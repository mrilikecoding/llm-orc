"""Provider status handler for MCP server."""

import os
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from llm_orc.core.config.closure import Closure, Key, walk_closure
from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.config.ensemble_config import (
    EnsembleConfig,
    EnsembleLoader,
    child_ensemble_search_dirs,
)
from llm_orc.core.execution.scripting.resolver import (
    ScriptNotFoundError,
    ScriptResolver,
)
from llm_orc.mcp.utils import get_agent_attr as _get_agent_attr
from llm_orc.providers.llama_server import LlamaServerClient
from llm_orc.providers.status_types import (
    AgentRunnability,
    AgentStatus,
    CloudProviderStatus,
    EndpointStatus,
    EnsembleRunnability,
    LlamaServerProviderStatus,
    OpenAICompatibleStatus,
)
from llm_orc.services.handlers.preflight import (
    UNBLOCKING,
    DependencyReport,
    DependencyStatus,
    classify_dependencies,
    is_runnable,
)
from llm_orc.services.handlers.profile_handler import ProfileHandler

_DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_LLAMA_SERVER_URL = "http://127.0.0.1:8080/v1"


class _Missing(Enum):
    """The argument was not given (``None`` is a real project dir value)."""

    MISSING = "missing"


_MISSING = _Missing.MISSING


@dataclass(frozen=True)
class Preflight:
    """What a gate run found: the closure, one report per dependency, and
    the provider status they were classified against."""

    closure: Closure
    reports: list[DependencyReport]
    providers: dict[str, Any]


class ProviderHandler:
    """Manages provider status and ensemble runnability checks."""

    _test_llama_server_status: dict[str, Any] | None = None
    _test_openai_compat_status: OpenAICompatibleStatus | None = None

    def __init__(
        self,
        profile_handler: ProfileHandler,
        find_ensemble: Callable[[str], EnsembleConfig | None],
        script_resolver_factory: Callable[[], ScriptResolver] | None = None,
        *,
        find_child: Callable[[str], EnsembleConfig | None],
    ) -> None:
        """Initialize with profile handler, the root finder (the API's
        lookup), the resolver the executor would use for scripts (ruling
        5), and the executor's own child-ensemble lookup (ruling 4)."""
        self._profile_handler = profile_handler
        self._find_ensemble = find_ensemble
        self._find_child = find_child
        self._script_resolver_factory = script_resolver_factory or ScriptResolver

    async def get_provider_status(
        self,
        arguments: dict[str, Any],
        profiles: dict[str, dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Get status of all providers and available models.

        ``profiles`` is the map the OpenAI-compatible endpoints are
        grouped from; the service's own runtime profiles when omitted.
        """
        providers: dict[str, Any] = {}

        providers["llama-server"] = await self._get_llama_server_status()

        providers["anthropic-api"] = self._get_cloud_provider_status("anthropic-api")
        providers["google-gemini"] = self._get_cloud_provider_status("google-gemini")

        oai_status = await self._get_openai_compatible_status(profiles)
        providers["openai-compatible"] = oai_status.model_dump()

        return {"providers": providers}

    async def _get_llama_server_status(self) -> dict[str, Any]:
        """Reachability and model list of the llama-server router (#90)."""
        if self._test_llama_server_status is not None:
            return self._test_llama_server_status

        base_url = os.environ.get("LLAMA_SERVER_URL", _DEFAULT_LLAMA_SERVER_URL)
        client = LlamaServerClient.from_base_url(base_url)
        try:
            inventory = client.inventory()
        except (OSError, ValueError) as e:
            return LlamaServerProviderStatus(
                available=False,
                reason=f"llama-server not reachable: {type(e).__name__}: {e}",
                base_url=base_url,
            ).model_dump()
        models = sorted(str(m.get("id", "")) for m in inventory.models)
        return LlamaServerProviderStatus(
            available=True,
            models=models,
            cached=inventory.cached,
            loaded=inventory.loaded,
            sources=inventory.sources,
            model_count=len(models),
            base_url=base_url,
        ).model_dump()

    def _get_cloud_provider_status(self, provider: str) -> dict[str, Any]:
        """Check if a cloud provider is configured."""
        from llm_orc.core.auth.authentication import (
            CredentialStorage,
        )

        storage = CredentialStorage()
        configured_providers = storage.list_providers()

        if provider in configured_providers:
            return CloudProviderStatus(available=True, reason="configured").model_dump()

        return CloudProviderStatus(
            available=False, reason="not configured"
        ).model_dump()

    async def _get_openai_compatible_status(
        self, profiles: dict[str, dict[str, str]] | None = None
    ) -> OpenAICompatibleStatus:
        """Check OpenAI-compatible endpoints and discover models."""
        if self._test_openai_compat_status is not None:
            return self._test_openai_compat_status

        all_profiles = (
            self._profile_handler.get_runtime_profiles()
            if profiles is None
            else profiles
        )

        # Group profiles by base_url
        url_profiles: dict[str, list[str]] = {}
        for name, profile in all_profiles.items():
            provider = profile.get("provider", "")
            if not _is_openai_compatible(provider):
                continue
            base_url = profile.get("base_url", _DEFAULT_OPENAI_BASE_URL)
            url_profiles.setdefault(base_url, []).append(name)

        if not url_profiles:
            return OpenAICompatibleStatus(available=False)

        import httpx

        endpoints: list[EndpointStatus] = []
        all_models: list[str] = []

        for base_url, profile_names in url_profiles.items():
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(f"{base_url}/models")
                    if response.status_code == 200:
                        data = response.json()
                        models = sorted(m.get("id", "") for m in data.get("data", []))
                        endpoints.append(
                            EndpointStatus(
                                base_url=base_url,
                                available=True,
                                models=models,
                                profiles=sorted(profile_names),
                            )
                        )
                        all_models.extend(models)
                    else:
                        endpoints.append(
                            EndpointStatus(
                                base_url=base_url,
                                available=False,
                                profiles=sorted(profile_names),
                                reason=f"HTTP {response.status_code}",
                            )
                        )
            except Exception as e:
                endpoints.append(
                    EndpointStatus(
                        base_url=base_url,
                        available=False,
                        profiles=sorted(profile_names),
                        reason=f"{type(e).__name__}: {e}",
                    )
                )

        unique_models = sorted(set(all_models))
        any_available = any(ep.available for ep in endpoints)

        return OpenAICompatibleStatus(
            available=any_available,
            endpoints=endpoints,
            models=unique_models,
            model_count=len(unique_models),
        )

    async def check_ensemble_runnable(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Check if an ensemble can run with current providers."""
        ensemble_name = arguments.get("ensemble_name")
        if not ensemble_name:
            raise ValueError("ensemble_name is required")

        config = self._find_ensemble(ensemble_name)
        if not config:
            raise ValueError(f"Ensemble not found: {ensemble_name}")

        outcome = await self.preflight(config, ensemble_name)
        closure, providers = outcome.closure, outcome.providers
        reports = outcome.reports
        by_key: dict[Key, DependencyReport] = {(r.kind, r.name): r for r in reports}

        agent_results = [
            self._agent_view(ensemble_name, agent, closure, by_key, providers)
            for agent in config.agents
        ]
        result = EnsembleRunnability(
            ensemble=ensemble_name,
            runnable=is_runnable(reports),
            agents=agent_results,
        ).model_dump()
        result["dependencies"] = [r.model_dump() for r in reports]
        return result

    async def preflight(
        self,
        config: EnsembleConfig,
        root_ref: str,
        *,
        config_manager: ConfigurationManager | None = None,
        project_dir: Path | None | Literal[_Missing.MISSING] = _MISSING,
    ) -> Preflight:
        """The dependency closure of ``config`` and a report for each
        dependency, with the providers they were classified against.

        Given a ``config_manager`` and ``project_dir`` (a run's view),
        the child finder, the profile map and the script resolver are
        built from that pair and from nothing else the service holds, so
        the verdict is the one the executor built on the same pair would
        reach (Arc 4). Without one, the service's own wiring answers.
        Half a pair raises: a view with the wrong project dir would
        resolve scripts where the executor does not.
        """
        if (config_manager is None) != (project_dir is _MISSING):
            raise ValueError(
                "preflight takes both config_manager and project_dir, or neither"
            )
        if config_manager is None or project_dir is _MISSING:
            profiles = self._profile_handler.get_runtime_profiles()
            find_child = self._find_child
            script_found: Callable[[str], bool] = self._script_found
        else:
            profiles = config_manager.get_model_profiles()
            find_child = _child_finder(config_manager, project_dir)
            script_found = _script_finder(
                ScriptResolver(
                    project_dir=project_dir, run_dir=config_manager.run_layer_dir
                )
            )
        provider_status = await self.get_provider_status({}, profiles=profiles)
        providers = provider_status.get("providers", {})

        closure = walk_closure(config, find_child, profiles, root_ref=root_ref)
        reports = classify_dependencies(
            closure.dependencies,
            profiles=profiles,
            providers=providers,
            script_found=script_found,
        )
        return Preflight(closure=closure, reports=reports, providers=providers)

    def _script_found(self, script_ref: str) -> bool:
        """The executor's own resolution (ruling 5): a bare name is
        inline content and resolves; only a path-syntax or absolute
        reference can be missing."""
        return _script_finder(self._script_resolver_factory())(script_ref)

    def _agent_view(
        self,
        root_ref: str,
        agent: Any,
        closure: Closure,
        by_key: dict[Key, DependencyReport],
        providers: dict[str, Any],
    ) -> AgentRunnability:
        """The coarse per-agent status the web UI reads, derived from the
        agent's own dependencies (ruling 6)."""
        agent_name = _get_agent_attr(agent, "name", "unknown")
        profile_name = _get_agent_attr(agent, "model_profile", None)
        result = AgentRunnability(
            name=agent_name,
            profile=profile_name if isinstance(profile_name, str) else "",
            provider=_agent_provider(agent, by_key),
        )
        owned = closure.owned.get(f"{root_ref}.{agent_name}", frozenset())
        unmet = [
            by_key[k]
            for k in _in_closure_order(owned, by_key)
            if by_key[k].status not in UNBLOCKING
        ]
        if not unmet:
            return result
        result.status = _COARSE[unmet[0].status]
        if result.status is AgentStatus.MODEL_UNAVAILABLE:
            result.alternatives = self._suggest_available_models(
                providers.get("llama-server", {}).get("models", [])
            )
        else:
            result.alternatives = self._suggest_local_alternatives(providers)
        return result

    def _suggest_local_alternatives(self, providers: dict[str, Any]) -> list[str]:
        """Suggest local profile alternatives."""
        all_profiles = self._profile_handler.get_runtime_profiles()
        local_profiles: list[str] = []

        local_available = providers.get("llama-server", {}).get("available", False)
        oai_available = providers.get("openai-compatible", {}).get("available", False)

        for name, profile in all_profiles.items():
            prov = profile.get("provider", "")
            if prov == "llama-server" and local_available:
                local_profiles.append(name)
            elif _is_openai_compatible(prov) and oai_available:
                local_profiles.append(name)

        return sorted(local_profiles)[:5]

    def _suggest_available_models(self, available_models: list[str]) -> list[str]:
        """Suggest available models."""
        return sorted(available_models)[:5]


def _script_finder(resolver: ScriptResolver) -> Callable[[str], bool]:
    """Whether ``resolver`` resolves a script reference."""

    def found(script_ref: str) -> bool:
        try:
            resolver.resolve_and_classify(script_ref)
        except ScriptNotFoundError:
            return False
        return True

    return found


def _child_finder(
    config_manager: ConfigurationManager, project_dir: Path | None
) -> Callable[[str], EnsembleConfig | None]:
    """The executor's child lookup over this manager and project dir:
    the same search dirs, the same by-filename finder."""
    loader = EnsembleLoader()

    def find(reference: str) -> EnsembleConfig | None:
        search_dirs = child_ensemble_search_dirs(project_dir, config_manager)
        return loader._find_ensemble_in_dirs(reference, search_dirs)

    return find


def _is_openai_compatible(provider: str | None) -> bool:
    """Check if a provider string is openai-compatible (or scoped)."""
    if not provider:
        return False
    return provider == "openai-compatible" or provider.startswith("openai-compatible/")


_COARSE: dict[DependencyStatus, AgentStatus] = {
    DependencyStatus.MISSING_PROFILE: AgentStatus.MISSING_PROFILE,
    DependencyStatus.PROVIDER_UNAVAILABLE: AgentStatus.PROVIDER_UNAVAILABLE,
    DependencyStatus.NEEDS_CREDENTIALS: AgentStatus.PROVIDER_UNAVAILABLE,
    DependencyStatus.PULLABLE: AgentStatus.MODEL_UNAVAILABLE,
    DependencyStatus.NEEDS_RESTART: AgentStatus.MODEL_UNAVAILABLE,
    DependencyStatus.MISSING_MODEL_SOURCE: AgentStatus.MODEL_UNAVAILABLE,
    DependencyStatus.MODEL_UNAVAILABLE: AgentStatus.MODEL_UNAVAILABLE,
    DependencyStatus.MISSING_SCRIPT: AgentStatus.DEPENDENCY_UNMET,
    DependencyStatus.MISSING_ENSEMBLE: AgentStatus.DEPENDENCY_UNMET,
}


def _in_closure_order(
    keys: frozenset[Key], by_key: dict[Key, DependencyReport]
) -> list[Key]:
    """Deterministic: the closure's own order, so the first unmet
    dependency is the same one on every call."""
    order = {k: i for i, k in enumerate(by_key)}
    return sorted(keys, key=lambda k: order[k])


def _agent_provider(agent: Any, by_key: dict[Key, DependencyReport]) -> str:
    """The label the old result carried: 'script' / 'ensemble' / 'loop'
    / 'dispatch' for non-LLM agents, the profile's provider otherwise."""
    for label in ("script", "ensemble", "loop", "dispatch"):
        value = _get_agent_attr(agent, label)
        if value is not None and value != "":
            return label
    profile_name = _get_agent_attr(agent, "model_profile", None)
    if isinstance(profile_name, str):
        report = by_key.get(("profile", profile_name))
        if report is not None and report.provider:
            return report.provider
    provider = _get_agent_attr(agent, "provider", None)
    return provider if isinstance(provider, str) else ""

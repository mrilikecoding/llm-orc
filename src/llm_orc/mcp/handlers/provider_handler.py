"""Provider status handler for MCP server."""

import os
from collections.abc import Callable
from typing import Any

from llm_orc.core.config.ensemble_config import EnsembleConfig
from llm_orc.mcp.handlers.profile_handler import ProfileHandler
from llm_orc.mcp.utils import get_agent_attr as _get_agent_attr


class ProviderHandler:
    """Manages provider status and ensemble runnability checks."""

    _test_ollama_status: dict[str, Any] | None = None

    def __init__(
        self,
        profile_handler: ProfileHandler,
        find_ensemble: Callable[[str], EnsembleConfig | None],
    ) -> None:
        """Initialize with profile handler and ensemble finder."""
        self._profile_handler = profile_handler
        self._find_ensemble = find_ensemble

    async def get_provider_status(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Get status of all providers and available models."""
        providers: dict[str, Any] = {}

        providers["ollama"] = await self._get_ollama_status()

        providers["anthropic-api"] = self._get_cloud_provider_status("anthropic-api")
        providers["anthropic-claude-pro-max"] = self._get_cloud_provider_status(
            "anthropic-claude-pro-max"
        )
        providers["google-gemini"] = self._get_cloud_provider_status("google-gemini")

        return {"providers": providers}

    async def _get_ollama_status(self) -> dict[str, Any]:
        """Check Ollama availability and list models."""
        if self._test_ollama_status is not None:
            return self._test_ollama_status

        import httpx

        ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        hosts = [ollama_host]
        if "localhost" in ollama_host:
            hosts.append(ollama_host.replace("localhost", "127.0.0.1"))

        last_error = ""
        for host in hosts:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(f"{host}/api/tags")
                    if response.status_code == 200:
                        data = response.json()
                        models = [
                            m.get("name", "") for m in data.get("models", [])
                        ]
                        return {
                            "available": True,
                            "models": sorted(models),
                            "model_count": len(models),
                        }
                    last_error = f"HTTP {response.status_code} from {host}"
            except Exception as e:
                last_error = f"{type(e).__name__}: {e} (host: {host})"
                continue

        return {
            "available": False,
            "models": [],
            "reason": f"Ollama not reachable: {last_error}",
        }

    def _get_cloud_provider_status(self, provider: str) -> dict[str, Any]:
        """Check if a cloud provider is configured."""
        from llm_orc.core.auth.authentication import (
            CredentialStorage,
        )

        storage = CredentialStorage()
        configured_providers = storage.list_providers()

        if provider in configured_providers:
            return {"available": True, "reason": "configured"}

        return {
            "available": False,
            "reason": "not configured",
        }

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

        provider_status = await self.get_provider_status({})
        providers = provider_status.get("providers", {})

        all_profiles = self._profile_handler.get_all_profiles()

        agents: list[dict[str, Any]] = []
        all_runnable = True

        for agent in config.agents:
            agent_name = _get_agent_attr(agent, "name", "unknown")

            script_path = _get_agent_attr(agent, "script", "")
            if script_path:
                agent_status: dict[str, Any] = {
                    "name": agent_name,
                    "profile": "",
                    "provider": "script",
                    "status": "available",
                    "alternatives": [],
                }
            else:
                profile_name = _get_agent_attr(agent, "model_profile", "")
                agent_status = self._check_agent_runnable(
                    agent_name,
                    profile_name,
                    all_profiles,
                    providers,
                )

            agents.append(agent_status)

            if agent_status["status"] != "available":
                all_runnable = False

        return {
            "ensemble": ensemble_name,
            "runnable": all_runnable,
            "agents": agents,
        }

    def _check_agent_runnable(
        self,
        agent_name: str,
        profile_name: str,
        all_profiles: dict[str, dict[str, Any]],
        providers: dict[str, Any],
    ) -> dict[str, Any]:
        """Check if an agent can run with current providers."""
        result: dict[str, Any] = {
            "name": agent_name,
            "profile": profile_name,
            "provider": "",
            "status": "available",
            "alternatives": [],
        }

        if profile_name not in all_profiles:
            result["status"] = "missing_profile"
            result["alternatives"] = self._suggest_local_alternatives(providers)
            return result

        profile = all_profiles[profile_name]
        provider = profile.get("provider", "")
        result["provider"] = provider

        provider_info = providers.get(provider, {})
        if not provider_info.get("available", False):
            result["status"] = "provider_unavailable"
            result["alternatives"] = self._suggest_local_alternatives(providers)
            return result

        if provider == "ollama":
            model = profile.get("model", "")
            available_models = provider_info.get("models", [])
            model_base = model.split(":")[0] if ":" in model else model
            model_found = any(
                m == model or m.startswith(f"{model_base}:") for m in available_models
            )
            if not model_found:
                result["status"] = "model_unavailable"
                result["alternatives"] = self._suggest_available_models(
                    available_models
                )

        return result

    def _suggest_local_alternatives(self, providers: dict[str, Any]) -> list[str]:
        """Suggest local profile alternatives."""
        ollama = providers.get("ollama", {})
        if not ollama.get("available", False):
            return []

        all_profiles = self._profile_handler.get_all_profiles()
        local_profiles: list[str] = []

        for name, profile in all_profiles.items():
            if profile.get("provider") == "ollama":
                local_profiles.append(name)

        return sorted(local_profiles)[:5]

    def _suggest_available_models(self, available_models: list[str]) -> list[str]:
        """Suggest available Ollama models."""
        return sorted(available_models)[:5]

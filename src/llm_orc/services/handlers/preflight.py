"""Preflight: classify every dependency of a closure against what this
host has (spec: docs/plans/2026-09-28-remote-delegation.md, Arc 3
re-cut rulings 2, 3 and 6).

Model presence for llama-server comes from the router's one listing
(``LlamaServerClient.inventory()``): a model the router lists is
routable; a source in its cache, or a model loaded now, is downloaded
(the cache entries date from the router's start, a load status is
live). Nothing else is
consulted, so preflight cannot disagree with the router that will serve
the request.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from llm_orc.core.config.closure import Dependency

_LLAMA_SERVER = "llama-server"
_OPENAI_COMPATIBLE = "openai-compatible"
_DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


class DependencyStatus(StrEnum):
    """The closed set. Adding one is a spec change, not a code change."""

    READY = "ready"
    PULLABLE = "pullable"
    NEEDS_RESTART = "needs_restart"
    MISSING_PROFILE = "missing_profile"
    MISSING_MODEL_SOURCE = "missing_model_source"
    MODEL_UNAVAILABLE = "model_unavailable"
    NEEDS_CREDENTIALS = "needs_credentials"
    MISSING_SCRIPT = "missing_script"
    MISSING_ENSEMBLE = "missing_ensemble"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    DYNAMIC = "dynamic"


RESOLVE: dict[DependencyStatus, str] = {
    DependencyStatus.READY: "none",
    DependencyStatus.DYNAMIC: "none",
    DependencyStatus.PULLABLE: "pull",
    DependencyStatus.NEEDS_RESTART: "restart",
    DependencyStatus.MISSING_PROFILE: "bind",
    DependencyStatus.MODEL_UNAVAILABLE: "bind",
    DependencyStatus.MISSING_MODEL_SOURCE: "add_source",
    DependencyStatus.NEEDS_CREDENTIALS: "add_credentials",
    DependencyStatus.MISSING_SCRIPT: "ship",
    DependencyStatus.MISSING_ENSEMBLE: "ship",
    DependencyStatus.PROVIDER_UNAVAILABLE: "start_provider",
}

UNBLOCKING = frozenset({DependencyStatus.READY, DependencyStatus.DYNAMIC})


class DependencyReport(BaseModel):
    """One dependency, its status, its hint, and the fact behind it."""

    kind: str
    name: str
    via: list[str] = Field(default_factory=list)
    status: DependencyStatus
    resolve: str
    detail: str
    provider: str | None = None


Verdict = tuple[DependencyStatus, str]


def classify_dependencies(
    dependencies: Sequence[Dependency],
    *,
    profiles: Mapping[str, Mapping[str, Any]],
    providers: Mapping[str, Any],
    script_found: Callable[[str], bool],
) -> list[DependencyReport]:
    """One report per dependency, in the closure's order."""
    reports: list[DependencyReport] = []
    for dep in dependencies:
        status, detail = _classify(dep, profiles, providers, script_found)
        reports.append(
            DependencyReport(
                kind=dep.kind,
                name=dep.name,
                via=list(dep.via),
                status=status,
                resolve=RESOLVE[status],
                detail=detail,
                provider=dep.provider,
            )
        )
    return reports


def is_runnable(reports: Sequence[DependencyReport]) -> bool:
    """Every dependency ready, or dynamic (decided at run time)."""
    return all(r.status in UNBLOCKING for r in reports)


def _classify(
    dep: Dependency,
    profiles: Mapping[str, Mapping[str, Any]],
    providers: Mapping[str, Any],
    script_found: Callable[[str], bool],
) -> Verdict:
    if dep.kind == "ensemble":
        if dep.found:
            return DependencyStatus.READY, "ensemble resolved"
        return (
            DependencyStatus.MISSING_ENSEMBLE,
            f"no ensemble named {dep.name!r} in any tier",
        )
    if dep.kind == "dispatch":
        return DependencyStatus.DYNAMIC, "dispatch target resolved at run time"
    if dep.kind == "script":
        return _classify_script(dep, script_found)
    if dep.kind == "model":
        return _classify_model({"provider": dep.provider, "model": dep.name}, providers)
    profile = profiles.get(dep.name)
    if profile is None:
        return DependencyStatus.MISSING_PROFILE, f"no profile named {dep.name!r}"
    return _classify_model(profile, providers)


def _classify_script(dep: Dependency, script_found: Callable[[str], bool]) -> Verdict:
    if dep.problem is not None:
        return (
            DependencyStatus.MISSING_SCRIPT,
            f"script {dep.name!r} has an unreadable llm-orc block: {dep.problem}",
        )
    if dep.beside is not None:
        # Looked for beside the resolved owner only, never on the search
        # path: a host file at the same relative path is not this file.
        if dep.found:
            return DependencyStatus.READY, f"listed file found beside {dep.beside!r}"
        return (
            DependencyStatus.MISSING_SCRIPT,
            f"file {dep.listed!r} listed by script {dep.beside!r} is not beside it",
        )
    if script_found(dep.name):
        return DependencyStatus.READY, "script resolved"
    return (
        DependencyStatus.MISSING_SCRIPT,
        f"script {dep.name!r} not found on the search path",
    )


def _classify_model(
    profile: Mapping[str, Any], providers: Mapping[str, Any]
) -> Verdict:
    provider = str(profile.get("provider") or "")
    model = str(profile.get("model") or "")
    if provider == _LLAMA_SERVER:
        return _classify_llama_server(model, profile.get("hf_repo"), providers)
    if provider == _OPENAI_COMPATIBLE or provider.startswith(_OPENAI_COMPATIBLE + "/"):
        return _classify_openai_compatible(model, profile, providers)
    info = providers.get(provider)
    if info is None:
        return DependencyStatus.PROVIDER_UNAVAILABLE, f"unknown provider {provider!r}"
    if info.get("available"):
        return DependencyStatus.READY, f"{provider} configured"
    return (
        DependencyStatus.NEEDS_CREDENTIALS,
        f"{provider} has no credentials on this host",
    )


def _classify_llama_server(
    model: str, hf_repo: Any, providers: Mapping[str, Any]
) -> Verdict:
    """Ruling 3, from the router's one listing."""
    info = providers.get(_LLAMA_SERVER, {})
    if not info.get("available"):
        return (
            DependencyStatus.PROVIDER_UNAVAILABLE,
            str(info.get("reason") or "router unreachable"),
        )
    listed = model in info.get("models", [])
    source = str(hf_repo) if hf_repo else None
    served = info.get("sources", {}).get(model)
    if listed and source is not None and served is not None and served != source:
        return (
            DependencyStatus.NEEDS_RESTART,
            f"model {model!r} listed; the router serves {served}, "
            f"the profile names {source}",
        )
    if listed and source is None:
        return (
            DependencyStatus.READY,
            f"model {model!r} listed by the router (no source to check)",
        )
    if listed and source in info.get("cached", []):
        return DependencyStatus.READY, f"model {model!r} listed; {source} cached"
    if listed and model in info.get("loaded", []):
        return (
            DependencyStatus.READY,
            f"model {model!r} listed and loaded now; "
            f"{source} not in the start-time cache",
        )
    if listed:
        return (
            DependencyStatus.PULLABLE,
            f"model {model!r} listed; {source} not cached "
            f"(POST /api/models/{model}/pull)",
        )
    if source is not None:
        return (
            DependencyStatus.NEEDS_RESTART,
            f"model {model!r} not listed; the router scanned its preset at start",
        )
    return (
        DependencyStatus.MISSING_MODEL_SOURCE,
        f"model {model!r} not listed and the profile has no hf_repo",
    )


def _classify_openai_compatible(
    model: str, profile: Mapping[str, Any], providers: Mapping[str, Any]
) -> Verdict:
    base_url = str(profile.get("base_url") or _DEFAULT_OPENAI_BASE_URL)
    for endpoint in providers.get(_OPENAI_COMPATIBLE, {}).get("endpoints", []):
        if endpoint.get("base_url") != base_url:
            continue
        if not endpoint.get("available"):
            return DependencyStatus.PROVIDER_UNAVAILABLE, f"{base_url} unreachable"
        if model in endpoint.get("models", []):
            return DependencyStatus.READY, f"model {model!r} listed at {base_url}"
        return (
            DependencyStatus.MODEL_UNAVAILABLE,
            f"model {model!r} not listed at {base_url}",
        )
    return DependencyStatus.PROVIDER_UNAVAILABLE, f"no endpoint status for {base_url}"

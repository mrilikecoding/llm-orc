"""Dependency classification (spec, Arc 3 re-cut rulings 2, 3, 6)."""

from __future__ import annotations

from typing import Any

import pytest

from llm_orc.core.config.closure import Dependency
from llm_orc.services.handlers.preflight import (
    RESOLVE,
    DependencyStatus,
    classify_dependencies,
    is_runnable,
)

ROUTER_UP: dict[str, Any] = {
    "llama-server": {
        "available": True,
        "models": ["qwen3-8b", "qwen3-14b", "qwen3-1.7b", "handmade"],
        "cached": ["unsloth/Qwen3-8B-GGUF:Q4_K_M"],
        "loaded": ["qwen3-1.7b"],
        "sources": {
            "qwen3-8b": "unsloth/Qwen3-8B-GGUF:Q4_K_M",
            "qwen3-14b": "unsloth/Qwen3-14B-GGUF:Q4_K_M",
        },
    },
    "anthropic-api": {"available": False, "reason": "not configured"},
    "google-gemini": {"available": True, "reason": "configured"},
    "openai-compatible": {
        "available": True,
        "endpoints": [
            {
                "base_url": "http://oai.local/v1",
                "available": True,
                "models": ["gpt-x"],
            },
            {"base_url": "http://down.local/v1", "available": False, "models": []},
        ],
    },
}

PROFILES: dict[str, dict[str, Any]] = {
    "ready": {
        "provider": "llama-server",
        "model": "qwen3-8b",
        "hf_repo": "unsloth/Qwen3-8B-GGUF:Q4_K_M",
    },
    "pull": {
        "provider": "llama-server",
        "model": "qwen3-14b",
        "hf_repo": "unsloth/Qwen3-14B-GGUF:Q4_K_M",
    },
    "loaded": {
        "provider": "llama-server",
        "model": "qwen3-1.7b",
        "hf_repo": "unsloth/Qwen3-1.7B-GGUF:Q4_K_M",
    },
    "new": {
        "provider": "llama-server",
        "model": "qwen3-4b",
        "hf_repo": "unsloth/Qwen3-4B-GGUF:Q4_K_M",
    },
    "handmade": {"provider": "llama-server", "model": "handmade"},
    "repointed": {
        "provider": "llama-server",
        "model": "qwen3-8b",
        "hf_repo": "bartowski/Qwen3-8B-GGUF:Q4_K_M",
    },
    "nosrc": {"provider": "llama-server", "model": "mystery"},
    "claude": {"provider": "anthropic-api", "model": "claude-x"},
    "gemini": {"provider": "google-gemini", "model": "gemini-x"},
    "oai": {
        "provider": "openai-compatible",
        "model": "gpt-x",
        "base_url": "http://oai.local/v1",
    },
    "oai-absent": {
        "provider": "openai-compatible",
        "model": "gpt-y",
        "base_url": "http://oai.local/v1",
    },
    "oai-down": {
        "provider": "openai-compatible",
        "model": "gpt-x",
        "base_url": "http://down.local/v1",
    },
    "legacy": {"provider": "ollama", "model": "llama3"},
}


def _profile(name: str) -> Dependency:
    profile = PROFILES.get(name)
    provider = profile.get("provider") if profile else None
    return Dependency(
        "profile", name, ("top.a",), provider=provider, found=profile is not None
    )


def _one(
    dep: Dependency, providers: dict[str, Any] = ROUTER_UP
) -> tuple[DependencyStatus, str]:
    [report] = classify_dependencies(
        [dep], profiles=PROFILES, providers=providers, script_found=lambda _: False
    )
    return report.status, report.resolve


@pytest.mark.parametrize(
    ("profile", "status", "resolve"),
    [
        ("ready", DependencyStatus.READY, "none"),
        ("pull", DependencyStatus.PULLABLE, "pull"),
        ("loaded", DependencyStatus.READY, "none"),  # live row: pulled, running
        ("new", DependencyStatus.NEEDS_RESTART, "restart"),
        ("repointed", DependencyStatus.NEEDS_RESTART, "restart"),
        ("handmade", DependencyStatus.READY, "none"),  # review focus 3
        ("nosrc", DependencyStatus.MISSING_MODEL_SOURCE, "add_source"),
        ("claude", DependencyStatus.NEEDS_CREDENTIALS, "add_credentials"),
        ("gemini", DependencyStatus.READY, "none"),
        ("oai", DependencyStatus.READY, "none"),
        ("oai-absent", DependencyStatus.MODEL_UNAVAILABLE, "bind"),
        ("oai-down", DependencyStatus.PROVIDER_UNAVAILABLE, "start_provider"),
        ("legacy", DependencyStatus.PROVIDER_UNAVAILABLE, "start_provider"),
        ("nope", DependencyStatus.MISSING_PROFILE, "bind"),
    ],
)
def test_profile_classification(
    profile: str, status: DependencyStatus, resolve: str
) -> None:
    assert _one(_profile(profile)) == (status, resolve)


def test_router_down_is_provider_unavailable_for_every_local_profile() -> None:
    providers = {
        **ROUTER_UP,
        "llama-server": {"available": False, "reason": "refused"},
    }
    status, _ = _one(_profile("ready"), providers)
    assert status == DependencyStatus.PROVIDER_UNAVAILABLE


def test_inline_listed_model_with_no_source_is_ready() -> None:
    """An inline model has no hf_repo, so the router's listing is the only fact."""
    dep = Dependency("model", "qwen3-14b", ("top.a",), provider="llama-server")
    assert _one(dep) == (DependencyStatus.READY, "none")


def test_scripts_ensembles_and_dispatch() -> None:
    reports = classify_dependencies(
        [
            Dependency("ensemble", "top", ()),
            Dependency("ensemble", "no-such", ("top.x",), found=False),
            Dependency("script", "scripts/gone.py", ("top.y",)),
            Dependency("script", "echo hi", ("top.z",)),
            Dependency("dispatch", "${x.target}", ("top.w",)),
        ],
        profiles=PROFILES,
        providers=ROUTER_UP,
        script_found=lambda ref: ref == "echo hi",  # review focus 4
    )
    assert [(r.status, r.resolve) for r in reports] == [
        (DependencyStatus.READY, "none"),
        (DependencyStatus.MISSING_ENSEMBLE, "ship"),
        (DependencyStatus.MISSING_SCRIPT, "ship"),
        (DependencyStatus.READY, "none"),
        (DependencyStatus.DYNAMIC, "none"),
    ]


def test_detail_names_the_observed_fact_for_pullable() -> None:
    [report] = classify_dependencies(
        [_profile("pull")],
        profiles=PROFILES,
        providers=ROUTER_UP,
        script_found=lambda _: True,
    )
    assert "qwen3-14b" in report.detail
    assert "unsloth/Qwen3-14B-GGUF:Q4_K_M" in report.detail
    assert "/api/models/qwen3-14b/pull" in report.detail


def test_detail_names_both_sources_when_the_router_serves_another() -> None:
    [report] = classify_dependencies(
        [_profile("repointed")],
        profiles=PROFILES,
        providers=ROUTER_UP,
        script_found=lambda _: True,
    )
    assert "qwen3-8b" in report.detail
    assert "unsloth/Qwen3-8B-GGUF:Q4_K_M" in report.detail
    assert "bartowski/Qwen3-8B-GGUF:Q4_K_M" in report.detail


def test_unknown_router_source_classifies_as_before() -> None:
    """A model with no source in the router's listing (older router, no
    ``status.args``) is judged on the cache alone."""
    providers = {
        **ROUTER_UP,
        "llama-server": {**ROUTER_UP["llama-server"], "sources": {}},
    }
    assert _one(_profile("repointed"), providers) == (
        DependencyStatus.PULLABLE,
        "pull",
    )


def test_detail_names_the_fact_behind_ready() -> None:
    reports = classify_dependencies(
        [_profile("ready"), _profile("loaded")],
        profiles=PROFILES,
        providers=ROUTER_UP,
        script_found=lambda _: True,
    )
    assert "unsloth/Qwen3-8B-GGUF:Q4_K_M cached" in reports[0].detail
    assert "loaded" in reports[1].detail


def test_runnable_requires_every_dependency_ready_or_dynamic() -> None:
    reports = classify_dependencies(
        [
            _profile("ready"),
            Dependency("dispatch", "${a}", ("top.b",)),
            _profile("pull"),
        ],
        profiles=PROFILES,
        providers=ROUTER_UP,
        script_found=lambda _: True,
    )
    assert is_runnable(reports[:2]) is True
    assert is_runnable(reports) is False


def test_every_status_has_a_resolve_hint() -> None:
    assert set(RESOLVE) == set(DependencyStatus)

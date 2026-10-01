"""Transitive preflight over REST, through the real OrchestraService.

One closure with one dependency of every status; the report is asserted
whole, so a classifier that forgets a kind, a walker that stops at the
root, or a handler that drops `dependencies` fails here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

import llm_orc.web.api as web_api
from llm_orc.providers.llama_server import LlamaServerClient
from llm_orc.web.server import create_app

LISTING: list[dict[str, Any]] = [
    {
        "id": "qwen3-8b",
        "status": {
            "value": "unloaded",
            "args": [
                "llama-server",
                "--alias",
                "qwen3-8b",
                "--hf-repo",
                "unsloth/Qwen3-8B-GGUF:Q4_K_M",
            ],
        },
        "source": "preset",
    },
    {"id": "qwen3-14b", "status": {"value": "unloaded"}},
    {"id": "qwen3-1.7b", "status": {"value": "loaded"}},
    {"id": "default", "status": {"value": "unloaded"}},
    {"id": "unsloth/Qwen3-8B-GGUF:Q4_K_M", "status": {"value": "unloaded"}},
]

PROFILES: dict[str, dict[str, Any]] = {
    "ready-prof": {
        "provider": "llama-server",
        "model": "qwen3-8b",
        "hf_repo": "unsloth/Qwen3-8B-GGUF:Q4_K_M",
    },
    "pull-prof": {
        "provider": "llama-server",
        "model": "qwen3-14b",
        "hf_repo": "unsloth/Qwen3-14B-GGUF:Q4_K_M",
    },
    "loaded-prof": {
        "provider": "llama-server",
        "model": "qwen3-1.7b",
        "hf_repo": "unsloth/Qwen3-1.7B-GGUF:Q4_K_M",
    },
    "new-prof": {
        "provider": "llama-server",
        "model": "qwen3-4b",
        "hf_repo": "unsloth/Qwen3-4B-GGUF:Q4_K_M",
    },
    "repointed-prof": {
        "provider": "llama-server",
        "model": "qwen3-8b",
        "hf_repo": "bartowski/Qwen3-8B-GGUF:Q4_K_M",
    },
    "nosrc-prof": {"provider": "llama-server", "model": "mystery"},
    "claude-prof": {"provider": "anthropic-api", "model": "claude-x"},
}


def _ensemble(dir_path: Path, name: str, agents: list[dict[str, Any]]) -> None:
    (dir_path / f"{name}.yaml").write_text(
        yaml.safe_dump({"name": name, "description": name, "agents": agents})
    )


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    dot = tmp_path / ".llm-orc"
    (dot / "ensembles").mkdir(parents=True)
    (dot / "scripts").mkdir()
    (dot / "config.yaml").write_text(yaml.safe_dump({"model_profiles": PROFILES}))
    (dot / "scripts" / "here.py").write_text("print('hi')\n")
    _ensemble(
        dot / "ensembles",
        "child",
        [
            {"name": "runner", "script": "scripts/gone.py"},
            {"name": "keeper", "script": "scripts/here.py"},
        ],
    )
    _ensemble(
        dot / "ensembles",
        "top",
        [
            {"name": "writer", "model_profile": "ready-prof"},
            {"name": "big", "model_profile": "pull-prof"},
            {"name": "hot", "model_profile": "loaded-prof"},
            {"name": "fresh", "model_profile": "new-prof"},
            {"name": "ghost", "model_profile": "nope"},
            {"name": "sourceless", "model_profile": "nosrc-prof"},
            {"name": "cloud", "model_profile": "claude-prof"},
            {"name": "sub", "ensemble": "child"},
            {"name": "again", "ensemble": "child"},
            {"name": "absent", "ensemble": "no-such"},
            {"name": "router", "dispatch": "${sub.target}"},
        ],
    )
    _ensemble(
        dot / "ensembles",
        "clean",
        [
            {"name": "writer", "model_profile": "ready-prof"},
            {"name": "keeper", "script": "scripts/here.py"},
        ],
    )
    _ensemble(
        dot / "ensembles",
        "repointed",
        [{"name": "writer", "model_profile": "repointed-prof"}],
    )
    _ensemble(
        dot / "ensembles",
        "only-pull",
        [{"name": "big", "model_profile": "pull-prof"}],
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(web_api, "_orchestra_service", None)
    monkeypatch.setattr(LlamaServerClient, "_list", lambda self: LISTING)
    return tmp_path


def _runnable(name: str) -> dict[str, Any]:
    with TestClient(create_app()) as client:
        response = client.get(f"/api/ensembles/{name}/runnable")
    assert response.status_code == 200, response.text
    return response.json()  # type: ignore[no-any-return]


class TestPreflightOverRest:
    def test_one_of_each_status_with_its_path_and_hint(self, project: Path) -> None:
        data = _runnable("top")

        assert data["runnable"] is False
        rows = [
            (d["kind"], d["name"], d["status"], d["resolve"], d["via"])
            for d in data["dependencies"]
        ]
        assert rows == [
            ("ensemble", "top", "ready", "none", []),
            ("profile", "ready-prof", "ready", "none", ["top.writer"]),
            ("profile", "pull-prof", "pullable", "pull", ["top.big"]),
            ("profile", "loaded-prof", "ready", "none", ["top.hot"]),
            ("profile", "new-prof", "needs_restart", "restart", ["top.fresh"]),
            ("profile", "nope", "missing_profile", "bind", ["top.ghost"]),
            (
                "profile",
                "nosrc-prof",
                "missing_model_source",
                "add_source",
                ["top.sourceless"],
            ),
            (
                "profile",
                "claude-prof",
                "needs_credentials",
                "add_credentials",
                ["top.cloud"],
            ),
            ("ensemble", "child", "ready", "none", ["top.sub"]),
            (
                "script",
                "scripts/gone.py",
                "missing_script",
                "ship",
                ["top.sub", "child.runner"],
            ),
            (
                "script",
                "scripts/here.py",
                "ready",
                "none",
                ["top.sub", "child.keeper"],
            ),
            ("ensemble", "no-such", "missing_ensemble", "ship", ["top.absent"]),
            ("dispatch", "${sub.target}", "dynamic", "none", ["top.router"]),
        ]

    def test_agents_keep_their_coarse_status(self, project: Path) -> None:
        """Ruling 6: the web UI's view stays truthful, including for the
        second agent that names an already-reported child (review focus 1)."""
        data = _runnable("top")

        assert {a["name"]: a["status"] for a in data["agents"]} == {
            "writer": "available",
            "big": "model_unavailable",
            "hot": "available",
            "fresh": "model_unavailable",
            "ghost": "missing_profile",
            "sourceless": "model_unavailable",
            "cloud": "provider_unavailable",
            "sub": "dependency_unmet",
            "again": "dependency_unmet",
            "absent": "dependency_unmet",
            "router": "available",
        }

    def test_clean_closure_is_runnable(self, project: Path) -> None:
        data = _runnable("clean")
        assert data["runnable"] is True
        assert {d["status"] for d in data["dependencies"]} == {"ready"}

    def test_pullable_alone_makes_it_not_runnable(self, project: Path) -> None:
        """Ruling 6: the caller decides about a multi-gigabyte download."""
        data = _runnable("only-pull")
        assert data["runnable"] is False
        assert [d["status"] for d in data["dependencies"]] == ["ready", "pullable"]

    def test_a_model_listed_under_another_source_needs_a_restart(
        self, project: Path
    ) -> None:
        """Ruling 6: the router lists the name but serves a different
        file, so the run would use the wrong model; the profile with the
        matching source (``clean``) still reads ready."""
        data = _runnable("repointed")

        assert data["runnable"] is False
        [row] = [d for d in data["dependencies"] if d["kind"] == "profile"]
        assert (row["status"], row["resolve"]) == ("needs_restart", "restart")
        assert "unsloth/Qwen3-8B-GGUF:Q4_K_M" in row["detail"]
        assert "bartowski/Qwen3-8B-GGUF:Q4_K_M" in row["detail"]

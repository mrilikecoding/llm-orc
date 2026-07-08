"""Tests for the Serving Layer ``/v1/models`` endpoint.

Per ``docs/agentic-serving/system-design.md`` §Serving Layer (L3) — the
endpoint enumerates orchestrator Model Profile IDs an operator has
exposed, in the OpenAI-compatible shape. Covers the ``/v1/models`` side
of ``scenarios.md`` §"Orchestrator tool set is exactly the committed
set" per roadmap WP-B line 39. The closed-five-tool guarantee is
enforced by Orchestrator Tool Dispatch (WP-C); this endpoint covers
listing only.
"""

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.config.model_profile_allowlist import ModelProfileAllowlist
from llm_orc.web.api import v1_models
from llm_orc.web.server import create_app


def _build_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    agentic_serving: dict[str, object] | None = None,
    library_profiles: list[str] | None = None,
) -> TestClient:
    """Build a TestClient whose ``/v1/models`` uses an isolated resolver.

    Writes a global ``config.yaml`` with the requested agentic_serving
    section and the named model profiles. Overrides the FastAPI
    dependency so the endpoint resolves against this fixture's config.
    """
    global_root = tmp_path / "xdg"
    global_root.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(global_root))
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    llm_orc_dir = global_root / "llm-orc"
    llm_orc_dir.mkdir(parents=True, exist_ok=True)
    config_body: dict[str, object] = {}
    if agentic_serving is not None:
        config_body["agentic_serving"] = agentic_serving
    if library_profiles:
        config_body["model_profiles"] = {
            name: {"model": "dummy-model", "provider": "dummy"}
            for name in library_profiles
        }
    (llm_orc_dir / "config.yaml").write_text(yaml.safe_dump(config_body))

    cm = ConfigurationManager(project_dir=project_dir, provision=False)
    allowlist = ModelProfileAllowlist(cm)

    monkeypatch.setattr(v1_models, "get_model_profile_allowlist", lambda: allowlist)
    return TestClient(create_app())


class TestV1ModelsEndpoint:
    """``GET /v1/models`` returns an OpenAI-compatible model list.

    Shape: ``{"object": "list", "data": [{"id", "object": "model",
    "created": int, "owned_by": "llm-orc"}, ...]}``.
    """

    def test_returns_openai_shape_listing_allowed_profiles(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = _build_client(
            tmp_path,
            monkeypatch,
            agentic_serving={
                "orchestrator": {
                    "model_profile": "primary",
                    "allowed_profiles": ["primary", "fast"],
                }
            },
            library_profiles=["primary", "fast", "other"],
        )

        response = client.get("/v1/models")

        assert response.status_code == 200
        body = response.json()
        assert body["object"] == "list"
        assert [entry["id"] for entry in body["data"]] == ["primary", "fast"]
        for entry in body["data"]:
            assert entry["object"] == "model"
            assert entry["owned_by"] == "llm-orc"
            assert isinstance(entry["created"], int)

    def test_excludes_library_profiles_outside_the_allowlist(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = _build_client(
            tmp_path,
            monkeypatch,
            agentic_serving={
                "orchestrator": {
                    "model_profile": "primary",
                    "allowed_profiles": ["primary"],
                }
            },
            library_profiles=["primary", "fast", "other"],
        )

        ids = [m["id"] for m in client.get("/v1/models").json()["data"]]

        assert ids == ["primary"]

    def test_returns_empty_data_when_default_profile_absent_from_library(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The DEFAULT_MODEL_PROFILE = 'default' placeholder case.

        When the operator has not configured any profile under that name,
        ``/v1/models`` exposes an empty list rather than raising. Session
        start is where the missing-profile error surfaces
        (``resolve_validated``).
        """
        client = _build_client(tmp_path, monkeypatch)

        body = client.get("/v1/models").json()

        assert body["object"] == "list"
        assert body["data"] == []

    def test_preserves_allowlist_ordering(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = _build_client(
            tmp_path,
            monkeypatch,
            agentic_serving={
                "orchestrator": {
                    "model_profile": "a",
                    "allowed_profiles": ["c", "a", "b"],
                }
            },
            library_profiles=["a", "b", "c"],
        )

        ids = [m["id"] for m in client.get("/v1/models").json()["data"]]

        assert ids == ["c", "a", "b"]

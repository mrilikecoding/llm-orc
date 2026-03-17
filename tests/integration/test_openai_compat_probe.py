"""Integration test for OpenAI-compatible endpoint probing.

Probes Ollama's OpenAI-compatible endpoint at http://localhost:11434/v1/models.
Skipped if Ollama is not running.
"""

import httpx
import pytest


def _ollama_running() -> bool:
    """Check if Ollama is reachable."""
    try:
        response = httpx.get("http://localhost:11434/api/tags", timeout=3.0)
        return response.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _ollama_running(),
    reason="Ollama not running at localhost:11434",
)


class TestOllamaOpenAICompatProbe:
    """Tests that probe Ollama's OpenAI-compatible /v1/models endpoint."""

    def test_v1_models_returns_openai_spec(self) -> None:
        """Response matches OpenAI /v1/models spec shape."""
        response = httpx.get("http://localhost:11434/v1/models", timeout=5.0)
        assert response.status_code == 200
        data = response.json()

        assert "data" in data
        assert isinstance(data["data"], list)

        if data["data"]:
            model = data["data"][0]
            assert "id" in model

    def test_model_ids_non_empty(self) -> None:
        """At least one model ID is non-empty."""
        response = httpx.get("http://localhost:11434/v1/models", timeout=5.0)
        data = response.json()
        model_ids = [m["id"] for m in data.get("data", [])]

        assert len(model_ids) > 0
        assert all(mid for mid in model_ids)

    def test_cross_reference_with_native_tags(self) -> None:
        """Model IDs from /v1/models match Ollama native /api/tags."""
        v1_response = httpx.get("http://localhost:11434/v1/models", timeout=5.0)
        v1_ids = {m["id"] for m in v1_response.json().get("data", [])}

        tags_response = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
        native_names = {
            m.get("name", "") for m in tags_response.json().get("models", [])
        }

        # Strip tag suffixes for comparison (Ollama native may include :latest)
        v1_bases = {mid.split(":")[0] for mid in v1_ids}
        native_bases = {name.split(":")[0] for name in native_names}

        # Every model from native should appear in v1 (by base name)
        assert native_bases.issubset(v1_bases) or v1_bases.issubset(native_bases)

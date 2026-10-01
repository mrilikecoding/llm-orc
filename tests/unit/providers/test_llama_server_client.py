"""The router's one listing decides routable, downloaded and load state."""

from __future__ import annotations

from unittest.mock import patch

from llm_orc.providers.llama_server import LlamaServerClient

PROBE_LISTING = [
    {
        "id": "qwen3-8b",
        "status": {
            "value": "unloaded",
            "args": [
                "/opt/llama-server",
                "--host",
                "127.0.0.1",
                "--alias",
                "qwen3-8b",
                "--ctx-size",
                "8192",
                "--hf-repo",
                "unsloth/Qwen3-8B-GGUF:Q4_K_M",
                "--n-gpu-layers",
                "999",
            ],
            "preset": "version = 1",
        },
        "source": "preset",
        "can_remove": False,
    },
    {"id": "qwen3-14b", "status": {"value": "unloaded"}, "source": "preset"},
    {"id": "qwen3-1.7b", "status": {"value": "loaded"}, "source": "preset"},
    {"id": "qwen3-4b", "status": "loaded", "source": "preset"},
    {"id": "default", "status": {"value": "unloaded"}, "source": "preset"},
    {
        "id": "unsloth/Qwen3-8B-GGUF:Q4_K_M",
        "status": {"value": "unloaded"},
        "source": "cache",
    },
]


class TestInventory:
    def test_models_hide_default_and_cache_entries(self) -> None:
        client = LlamaServerClient("http://127.0.0.1:8791")
        with patch.object(LlamaServerClient, "_list", return_value=PROBE_LISTING):
            inventory = client.inventory()
        assert [m["id"] for m in inventory.models] == [
            "qwen3-8b",
            "qwen3-14b",
            "qwen3-1.7b",
            "qwen3-4b",
        ]

    def test_cached_is_exactly_the_hf_repo_strings(self) -> None:
        client = LlamaServerClient("http://127.0.0.1:8791")
        with patch.object(LlamaServerClient, "_list", return_value=PROBE_LISTING):
            inventory = client.inventory()
        assert inventory.cached == ["unsloth/Qwen3-8B-GGUF:Q4_K_M"]

    def test_sources_are_the_value_after_hf_repo_in_the_status_args(self) -> None:
        """The router states which file a model name serves (probe
        2026-10-01); a model with no ``--hf-repo`` arg, or a bare-string
        status, is absent rather than guessed."""
        client = LlamaServerClient("http://127.0.0.1:8791")
        with patch.object(LlamaServerClient, "_list", return_value=PROBE_LISTING):
            inventory = client.inventory()
        assert inventory.sources == {"qwen3-8b": "unsloth/Qwen3-8B-GGUF:Q4_K_M"}

    def test_loaded_is_the_preset_models_whose_live_status_is_loaded(self) -> None:
        """A status is ``{"value": ...}`` or a bare string (web/api/models.py)."""
        client = LlamaServerClient("http://127.0.0.1:8791")
        listing = [
            *PROBE_LISTING,
            {"id": "unsloth/Qwen3-1.7B-GGUF:Q4_K_M", "status": {"value": "loaded"}},
        ]
        with patch.object(LlamaServerClient, "_list", return_value=listing):
            inventory = client.inventory()
        assert inventory.loaded == ["qwen3-1.7b", "qwen3-4b"]

    def test_models_and_inventory_share_one_listing(self) -> None:
        client = LlamaServerClient("http://127.0.0.1:8791")
        with patch.object(
            LlamaServerClient, "_list", return_value=PROBE_LISTING
        ) as listing:
            client.models()
            client.inventory()
        assert listing.call_count == 2  # one GET per call, never two per call


def _entry(model: str, **status: object) -> list[dict[str, object]]:
    return [{"id": model, "status": status, "source": "preset"}]


class TestPull:
    """One pull-and-wait for every caller (spec Arc 4 re-cut, ruling 4);
    the shapes are the failed and good loads the 2026-10-01 probe saw."""

    def test_waits_through_loading_and_returns_the_loaded_status(self) -> None:
        client = LlamaServerClient("http://127.0.0.1:8791")
        listings = [
            _entry("qwen3-8b", value="loading"),
            _entry("qwen3-8b", value="loading"),
            _entry("qwen3-8b", value="loaded"),
        ]
        with (
            patch.object(LlamaServerClient, "load") as load,
            patch.object(LlamaServerClient, "_list", side_effect=listings),
        ):
            result = client.pull("qwen3-8b", timeout_s=5, poll_s=0)
        load.assert_called_once_with("qwen3-8b")
        assert result == {"status": "loaded", "failed": False, "exit_code": None}

    def test_a_failed_load_is_unloaded_failed_with_its_exit_code(self) -> None:
        client = LlamaServerClient("http://127.0.0.1:8791")
        failed = _entry("bogus", value="unloaded", exit_code=1, failed=True)
        with (
            patch.object(LlamaServerClient, "load"),
            patch.object(LlamaServerClient, "_list", return_value=failed),
        ):
            result = client.pull("bogus", timeout_s=5, poll_s=0)
        assert result == {"status": "unloaded", "failed": True, "exit_code": 1}

    def test_a_load_still_loading_at_the_deadline_reports_loading(self) -> None:
        client = LlamaServerClient("http://127.0.0.1:8791")
        with (
            patch.object(LlamaServerClient, "load"),
            patch.object(
                LlamaServerClient,
                "_list",
                return_value=_entry("slow", value="loading"),
            ),
        ):
            result = client.pull("slow", timeout_s=0, poll_s=0)
        assert result["status"] == "loading"

    def test_a_model_the_router_does_not_list_is_unknown(self) -> None:
        client = LlamaServerClient("http://127.0.0.1:8791")
        with (
            patch.object(LlamaServerClient, "load"),
            patch.object(LlamaServerClient, "_list", return_value=PROBE_LISTING),
        ):
            result = client.pull("nope", timeout_s=5, poll_s=0)
        assert result == {"status": "unknown", "failed": False, "exit_code": None}

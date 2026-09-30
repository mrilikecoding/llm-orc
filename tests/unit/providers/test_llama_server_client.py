"""The router's one listing decides routable, downloaded and load state."""

from __future__ import annotations

from unittest.mock import patch

from llm_orc.providers.llama_server import LlamaServerClient

PROBE_LISTING = [
    {"id": "qwen3-8b", "status": {"value": "unloaded"}, "source": "preset"},
    {"id": "qwen3-14b", "status": {"value": "unloaded"}, "source": "preset"},
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
        assert [m["id"] for m in inventory.models] == ["qwen3-8b", "qwen3-14b"]

    def test_cached_is_exactly_the_hf_repo_strings(self) -> None:
        client = LlamaServerClient("http://127.0.0.1:8791")
        with patch.object(LlamaServerClient, "_list", return_value=PROBE_LISTING):
            inventory = client.inventory()
        assert inventory.cached == ["unsloth/Qwen3-8B-GGUF:Q4_K_M"]

    def test_models_and_inventory_share_one_listing(self) -> None:
        client = LlamaServerClient("http://127.0.0.1:8791")
        with patch.object(
            LlamaServerClient, "_list", return_value=PROBE_LISTING
        ) as listing:
            client.models()
            client.inventory()
        assert listing.call_count == 2  # one GET per call, never two per call

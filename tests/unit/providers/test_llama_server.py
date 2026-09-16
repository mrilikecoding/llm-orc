"""llama-server router ownership (#90): the preset llm-orc renders from its
profiles is the contract between the profile set and the inference process."""

import json
import signal
import socket
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from llm_orc.providers.llama_server import (
    LlamaServerClient,
    LlamaServerSupervisor,
    install_signal_stop,
    render_preset,
    start_router_from_config,
)


class TestRenderPreset:
    def test_one_section_per_distinct_model_with_source_and_context(self) -> None:
        profiles: dict[str, dict[str, Any]] = {
            "agentic-tier-cheap-general": {
                "model": "qwen3-8b",
                "provider": "llama-server",
                "hf_repo": "unsloth/Qwen3-8B-GGUF:Q4_K_M",
                "options": {"num_ctx": 16384},
            },
            "another-seat-on-the-same-model": {
                "model": "qwen3-8b",
                "provider": "llama-server",
            },
            "agentic-tier-cheap-summary": {
                "model": "qwen3-1.7b",
                "provider": "llama-server",
                "hf_repo": "unsloth/Qwen3-1.7B-GGUF:Q4_K_M",
            },
            "premium-claude": {"model": "claude-sonnet-4", "provider": "anthropic-api"},
        }

        preset = render_preset(profiles)

        assert preset.models == ["qwen3-1.7b", "qwen3-8b"]
        assert preset.text == (
            "version = 1\n"
            "\n"
            "[*]\n"
            "c = 40960\n"
            "flash-attn = on\n"
            "jinja = true\n"
            "n-gpu-layers = 999\n"
            "\n"
            "[qwen3-1.7b]\n"
            "hf-repo = unsloth/Qwen3-1.7B-GGUF:Q4_K_M\n"
            "\n"
            "[qwen3-8b]\n"
            "c = 16384\n"
            "hf-repo = unsloth/Qwen3-8B-GGUF:Q4_K_M\n"
        )

    def test_model_without_a_source_is_skipped_and_named(self) -> None:
        """A section with neither hf-repo nor a model path is invalid to the
        router, so it is left out and reported instead of emitted."""
        profiles: dict[str, dict[str, Any]] = {
            "local-qwen3-8b": {"model": "qwen3-8b", "provider": "llama-server"},
        }

        preset = render_preset(profiles)

        assert preset.models == []
        assert preset.missing_source == ["qwen3-8b"]
        assert "[qwen3-8b]" not in preset.text

    def test_conflicting_sources_for_one_model_name_refuse(self) -> None:
        profiles: dict[str, dict[str, Any]] = {
            "a": {
                "model": "qwen3-8b",
                "provider": "llama-server",
                "hf_repo": "unsloth/Qwen3-8B-GGUF:Q4_K_M",
            },
            "b": {
                "model": "qwen3-8b",
                "provider": "llama-server",
                "hf_repo": "unsloth/Qwen3-8B-GGUF:Q8_0",
            },
        }

        with pytest.raises(ValueError, match="qwen3-8b"):
            render_preset(profiles)

    def test_slash_in_a_model_name_refuses(self) -> None:
        """A ``/`` marks a raw cache entry in the router's list, which the
        client hides; a preset named with one would vanish from it."""
        profiles: dict[str, dict[str, Any]] = {
            "a": {
                "model": "unsloth/qwen3-8b",
                "provider": "llama-server",
                "hf_repo": "unsloth/Qwen3-8B-GGUF:Q4_K_M",
            },
        }

        with pytest.raises(ValueError, match="slash"):
            render_preset(profiles)

    def test_colon_in_a_model_name_refuses(self) -> None:
        """The router rewrites ``name:tag`` (spike 2026-09-16: ``qwen3:1.7b``
        listed as ``qwen3:7B``), so a colon would silently break routing."""
        profiles: dict[str, dict[str, Any]] = {
            "a": {
                "model": "qwen3:8b",
                "provider": "llama-server",
                "hf_repo": "unsloth/Qwen3-8B-GGUF:Q4_K_M",
            },
        }

        with pytest.raises(ValueError, match="colon"):
            render_preset(profiles)


STUB_ROUTER = """\
import json, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer

args = sys.argv[1:]
port = int(args[args.index("--port") + 1])
preset = open(args[args.index("--models-preset") + 1]).read()
if "CRASH" in preset:
    print("E srv  llama_server: option 'bogus' not recognized", file=sys.stderr)
    sys.exit(3)
time.sleep(0.3)  # the real router takes a moment before it listens
loaded = []

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _json(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def do_GET(self):
        if self.path == "/models":
            state = "loaded" if "qwen3-8b" in loaded else "unloaded"
            self._json(200, {"data": [{"id": "qwen3-8b", "status": {"value": state}}]})
        else:
            self._json(404, {})
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/models/load":
            loaded.append(body["model"])
            self._json(200, {"success": True})
        else:
            self._json(404, {})

HTTPServer(("127.0.0.1", port), H).serve_forever()
"""


@pytest.fixture
def stub_binary(tmp_path: Path) -> Path:
    script = tmp_path / "llama-server"
    script.write_text("#!/usr/bin/env python3\n" + STUB_ROUTER)
    script.chmod(0o755)
    return script


@pytest.fixture
def preset_path(tmp_path: Path) -> Path:
    path = tmp_path / "llama-server.ini"
    path.write_text("version = 1\n")
    return path


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class TestLlamaServerSupervisor:
    def test_command_is_the_router_invocation(
        self, stub_binary: Path, preset_path: Path
    ) -> None:
        sup = LlamaServerSupervisor(
            preset_path=preset_path,
            binary=stub_binary,
            host="127.0.0.1",
            port=8080,
        )

        assert sup.command() == [
            str(stub_binary),
            "--models-preset",
            str(preset_path),
            "--host",
            "127.0.0.1",
            "--port",
            "8080",
            "--models-max",
            "1",
            "--no-webui",
        ]
        assert sup.base_url == "http://127.0.0.1:8080/v1"

    def test_start_waits_for_readiness_then_stop_terminates(
        self, stub_binary: Path, preset_path: Path
    ) -> None:
        sup = LlamaServerSupervisor(
            preset_path=preset_path, binary=stub_binary, port=_free_port()
        )

        sup.start(timeout_s=10.0)
        try:
            assert sup.running
            assert [m["id"] for m in sup.models()] == ["qwen3-8b"]
            sup.load("qwen3-8b")
            assert sup.models()[0]["status"]["value"] == "loaded"
        finally:
            sup.stop()

        assert not sup.running

    def test_start_raises_with_the_routers_last_stderr_when_it_exits_early(
        self, stub_binary: Path, preset_path: Path
    ) -> None:
        """The reason lives in the router's stderr (e2e 2026-09-16: an
        unrecognized preset option), so the error carries its tail."""
        preset_path.write_text("CRASH\n")
        sup = LlamaServerSupervisor(
            preset_path=preset_path, binary=stub_binary, port=_free_port()
        )

        with pytest.raises(RuntimeError, match="exited with code 3") as excinfo:
            sup.start(timeout_s=10.0)

        assert "option 'bogus' not recognized" in str(excinfo.value)
        assert not sup.running


class TestStartRouterFromConfig:
    def test_renders_preset_into_config_dir_and_starts(
        self, stub_binary: Path, tmp_path: Path
    ) -> None:
        config = MagicMock()
        config.get_model_profiles.return_value = {
            "agentic-tier-cheap-general": {
                "model": "qwen3-8b",
                "provider": "llama-server",
                "hf_repo": "unsloth/Qwen3-8B-GGUF:Q4_K_M",
            }
        }
        config.local_config_dir = tmp_path / ".llm-orc"

        sup = start_router_from_config(
            config, binary=stub_binary, port=_free_port(), timeout_s=10.0
        )
        try:
            assert sup.running
            assert sup.preset_path == tmp_path / ".llm-orc" / "llama-server.ini"
            assert "[qwen3-8b]" in sup.preset_path.read_text()
        finally:
            sup.stop()

    def test_falls_back_to_global_config_dir_without_a_local_one(
        self, stub_binary: Path, tmp_path: Path
    ) -> None:
        config = MagicMock()
        config.get_model_profiles.return_value = {}
        config.local_config_dir = None
        config.global_config_dir = tmp_path / "global"

        sup = start_router_from_config(
            config, binary=stub_binary, port=_free_port(), timeout_s=10.0
        )
        try:
            assert sup.preset_path == tmp_path / "global" / "llama-server.ini"
        finally:
            sup.stop()


class TestClientModelList:
    def test_router_placeholder_and_cache_entries_are_hidden(self) -> None:
        """Router mode lists a ``default`` entry for its own command line and
        one raw entry per cached Hugging Face file (e2e 2026-09-16); neither
        is a preset model. This build rejects ``dedup-cache-models``, so the
        client hides them: cache entries are the ids with a ``/``."""
        client = LlamaServerClient("http://127.0.0.1:8089")
        payload = {
            "data": [
                {"id": "default", "status": {"value": "unloaded"}},
                {"id": "unsloth/Qwen3-8B-GGUF:Q4_K_M", "status": {"value": "unloaded"}},
                {"id": "qwen3-8b", "status": {"value": "loaded"}},
            ]
        }
        with patch("llm_orc.providers.llama_server._DIRECT.open") as opener:
            opener.return_value.__enter__.return_value.read.return_value = json.dumps(
                payload
            ).encode()
            models = client.models()

        assert [m["id"] for m in models] == ["qwen3-8b"]


class TestInstallSignalStop:
    def test_sigterm_stops_the_router_then_re_raises_the_signal(self) -> None:
        """uvicorn re-raises the signal it captured after restoring default
        handlers, so the process dies before any ``finally`` (e2e
        2026-09-16: router alive after SIGTERM to the serve). The handler
        installed before uvicorn runs is what gets the re-raise."""
        supervisor = MagicMock()
        handler = install_signal_stop(supervisor)

        with (
            patch("llm_orc.providers.llama_server.signal.signal") as set_handler,
            patch("llm_orc.providers.llama_server.signal.raise_signal") as reraise,
        ):
            handler(signal.SIGTERM, None)

        supervisor.stop.assert_called_once()
        set_handler.assert_called_with(signal.SIGTERM, signal.SIG_DFL)
        reraise.assert_called_once_with(signal.SIGTERM)

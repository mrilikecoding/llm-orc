"""An explicit key check says whether a stored cloud key works (#204).

The real Anthropic and Google clients run over a patched httpx transport,
so nothing leaves the process and the status codes are the provider's. The
stored key goes through the real CredentialStorage into the per-test XDG
dir from tests/conftest.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
import yaml
from google import genai

from llm_orc.core.auth.authentication import CredentialStorage
from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.providers import key_check
from llm_orc.providers.llama_server import LlamaServerClient
from llm_orc.services.handlers.provider_handler import ProviderHandler
from llm_orc.services.orchestra_service import OrchestraService

PROVIDERS = ["anthropic-api", "google-gemini"]


class _Wire:
    """What the provider says next, and every request that reached it."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.respond: int | Exception = 200
        self.raw: bytes | None = None


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch) -> _Wire:
    seen = _Wire()

    async def handle(
        _self: httpx.AsyncHTTPTransport, request: httpx.Request
    ) -> httpx.Response:
        seen.requests.append(request)
        if isinstance(seen.respond, Exception):
            raise seen.respond
        if seen.respond != 200:
            return httpx.Response(
                seen.respond, json={"error": {"message": "no"}}, request=request
            )
        if seen.raw is not None:
            return httpx.Response(200, content=seen.raw, request=request)
        google = "googleapis" in request.url.host
        body: dict[str, Any] = (
            {"models": [{"name": "models/m"}]}
            if google
            else {
                "data": [
                    {
                        "id": "m",
                        "type": "model",
                        "display_name": "M",
                        "created_at": "2025-01-01T00:00:00Z",
                    }
                ],
                "has_more": False,
                "first_id": None,
                "last_id": None,
            }
        )
        return httpx.Response(200, json=body, request=request)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", handle)
    return seen


def _handler(*stored: tuple[str, str]) -> ProviderHandler:
    manager = ConfigurationManager(provision=False)
    storage = CredentialStorage(manager)
    for provider, api_key in stored:
        storage.store_api_key(provider, api_key)
    profile_handler = MagicMock()
    profile_handler.config_manager = manager
    profile_handler.get_runtime_profiles.return_value = {}
    find = MagicMock(return_value=None)
    return ProviderHandler(profile_handler, find, find_child=find)


async def _verified(handler: ProviderHandler, provider: str) -> dict[str, Any]:
    result = await handler.get_provider_status({"verify": True})
    status: dict[str, Any] = result["providers"][provider]
    return status


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_accepted_key_is_available(provider: str, wire: _Wire) -> None:
    handler = _handler((provider, "good-key"))

    status = await _verified(handler, provider)

    assert status["available"] is True
    assert status["state"] == "accepted"
    assert len(wire.requests) == 1


@pytest.mark.parametrize("code", [401, 403])
@pytest.mark.parametrize("provider", PROVIDERS)
async def test_a_refused_credential_is_rejected(
    provider: str, code: int, wire: _Wire
) -> None:
    wire.respond = code
    handler = _handler((provider, "bad-key"))

    status = await _verified(handler, provider)

    assert status["available"] is False
    assert status["state"] == "rejected"
    assert str(code) in status["reason"]


@pytest.mark.parametrize("code", [400, 404, 408, 429, 500, 503, 529])
@pytest.mark.parametrize("provider", PROVIDERS)
async def test_other_failures_are_inconclusive_with_the_status(
    provider: str, code: int, wire: _Wire
) -> None:
    wire.respond = code
    handler = _handler((provider, "any-key"))

    status = await _verified(handler, provider)

    assert status["available"] is False
    assert status["state"] == "inconclusive"
    assert str(code) in status["reason"]
    assert len(wire.requests) == 1, "no retries"


@pytest.mark.parametrize(
    "failure",
    [httpx.ConnectTimeout("timed out"), httpx.ConnectError("no route")],
)
@pytest.mark.parametrize("provider", PROVIDERS)
async def test_network_failures_are_inconclusive(
    provider: str, failure: Exception, wire: _Wire
) -> None:
    wire.respond = failure
    handler = _handler((provider, "any-key"))

    status = await _verified(handler, provider)

    assert status["available"] is False
    assert status["state"] == "inconclusive"
    assert status["reason"]


@pytest.mark.parametrize("key", ["kéy", "bad\nkey", "bad\rkey", "bad\x00key"])
@pytest.mark.parametrize("provider", PROVIDERS)
async def test_a_key_that_cannot_be_a_header_is_rejected_as_malformed(
    provider: str, key: str, wire: _Wire
) -> None:
    handler = _handler((provider, key))

    status = await _verified(handler, provider)

    assert status["available"] is False
    assert status["state"] == "rejected"
    assert "malformed key" in status["reason"]
    assert wire.requests == []


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_an_unexpected_exception_is_inconclusive_with_its_class(
    provider: str, wire: _Wire, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(**_kwargs: Any) -> None:
        raise RuntimeError("surprise")

    monkeypatch.setattr(key_check, "AsyncAnthropic", boom)
    monkeypatch.setattr(genai, "Client", boom)
    handler = _handler((provider, "any-key"))

    status = await _verified(handler, provider)

    assert status["state"] == "inconclusive"
    assert "RuntimeError" in status["reason"]


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_default_status_makes_no_provider_call(
    provider: str, wire: _Wire
) -> None:
    handler = _handler((provider, "any-key"))

    result = await handler.get_provider_status({})

    assert result["providers"][provider] == {
        "available": True,
        "reason": "configured",
        "state": "unchecked",
    }
    assert wire.requests == []


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_no_stored_credential_is_not_configured_and_makes_no_call(
    provider: str, wire: _Wire
) -> None:
    handler = _handler()

    status = await _verified(handler, provider)

    assert status["available"] is False
    assert status["state"] == "not_configured"
    assert status["reason"] == "not configured"
    assert wire.requests == []


async def test_each_explicit_check_asks_the_provider_again(wire: _Wire) -> None:
    handler = _handler(("anthropic-api", "a-key"))
    wire.respond = 401
    first = await _verified(handler, "anthropic-api")
    wire.respond = 200

    second = await _verified(handler, "anthropic-api")

    assert (first["state"], second["state"]) == ("rejected", "accepted")
    assert len(wire.requests) == 2


async def test_claude_cli_path_credential_is_not_sent_to_the_api(
    wire: _Wire,
) -> None:
    handler = _handler(("anthropic-api", "/usr/local/bin/claude"))

    status = await _verified(handler, "anthropic-api")

    assert status["available"] is True
    assert status["state"] == "unchecked"
    assert wire.requests == []


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_oauth_credential_is_not_sent_to_the_api(
    provider: str, wire: _Wire
) -> None:
    manager = ConfigurationManager(provision=False)
    storage = CredentialStorage(manager)
    storage.store_api_key(provider, "an-older-key")  # OAuth later replaced it
    storage.store_oauth_token(provider, "access-token")
    profile_handler = MagicMock()
    profile_handler.config_manager = manager
    find = MagicMock(return_value=None)
    handler = ProviderHandler(profile_handler, find, find_child=find)

    status = await _verified(handler, provider)

    assert status["available"] is True
    assert status["state"] == "unchecked"
    assert wire.requests == []


async def test_client_is_bounded_by_the_named_timeout(
    wire: _Wire, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[httpx.Timeout] = []
    real = httpx.AsyncClient.send

    async def send(
        self: httpx.AsyncClient, request: httpx.Request, **kwargs: Any
    ) -> httpx.Response:
        seen.append(httpx.Timeout(request.extensions["timeout"]["read"]))
        return await real(self, request, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    handler = _handler(("anthropic-api", "a-key"), ("google-gemini", "g-key"))

    await handler.get_provider_status({"verify": True})

    assert [t.read for t in seen] == [key_check.KEY_CHECK_TIMEOUT_SECONDS] * 2


# --- The run path never verifies a key -------------------------------------


def _write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data))


@pytest.fixture
def project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wire: _Wire
) -> OrchestraService:
    """A project with a stored cloud key, a ready local seat and a cloud one."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        LlamaServerClient,
        "_list",
        lambda self: [{"id": "qwen3-8b", "status": {"value": "unloaded"}}],
    )
    CredentialStorage(ConfigurationManager(provision=False)).store_api_key(
        "anthropic-api", "a-valid-key"
    )
    root = tmp_path / "proj" / ".llm-orc"
    _write(
        root / "profiles" / "local.yaml",
        {"name": "local", "provider": "llama-server", "model": "qwen3-8b"},
    )
    _write(
        root / "profiles" / "cloud.yaml",
        {
            "name": "cloud",
            "provider": "anthropic-api",
            "model": "a-model",
            "fallback_model_profile": "local",
        },
    )
    _write(
        root / "profiles" / "cloud-only.yaml",
        {"name": "cloud-only", "provider": "anthropic-api", "model": "a-model"},
    )
    for name, profile in [
        ("local-only", "local"),
        ("cloud-with-fallback", "cloud"),
        ("cloud-only", "cloud-only"),
    ]:
        _write(
            root / "ensembles" / f"{name}.yaml",
            {
                "name": name,
                "description": name,
                "agents": [{"name": "a", "model_profile": profile}],
            },
        )
    service = OrchestraService()
    assert service.handle_set_project(str(tmp_path / "proj"))["status"] == "ok"
    return service


async def test_local_only_ensemble_with_a_cloud_key_makes_no_provider_call(
    project: OrchestraService, wire: _Wire
) -> None:
    result = await project.check_ensemble_runnable({"ensemble_name": "local-only"})

    assert result["runnable"] is True
    assert wire.requests == []


async def test_offline_cloud_agent_with_a_ready_fallback_is_runnable(
    project: OrchestraService, wire: _Wire
) -> None:
    wire.respond = httpx.ConnectError("offline")

    result = await project.check_ensemble_runnable(
        {"ensemble_name": "cloud-with-fallback"}
    )

    assert result["runnable"] is True
    assert wire.requests == []


async def test_rate_limited_valid_key_does_not_block_the_run(
    project: OrchestraService, wire: _Wire
) -> None:
    wire.respond = 429

    result = await project.check_ensemble_runnable({"ensemble_name": "cloud-only"})

    assert result["runnable"] is True
    assert wire.requests == []


# --- The callers that ask -----------------------------------------------------


async def test_mcp_tool_verifies_only_when_asked(
    wire: _Wire, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from llm_orc.mcp.server import MCPServer

    # An empty cwd: a checkout's own untracked profiles would add
    # OpenAI-compatible endpoint probes to the requests counted here.
    monkeypatch.chdir(tmp_path)
    CredentialStorage(ConfigurationManager(provision=False)).store_api_key(
        "anthropic-api", "a-key"
    )
    server = MCPServer(config_manager=ConfigurationManager(provision=False))

    await server._mcp.call_tool("get_provider_status", {})
    assert wire.requests == []

    await server._mcp.call_tool("get_provider_status", {"verify": True})
    assert len(wire.requests) == 1


def test_auth_list_verifies_only_with_the_flag(wire: _Wire) -> None:
    from click.testing import CliRunner

    from llm_orc.cli import cli

    CredentialStorage(ConfigurationManager(provision=False)).store_api_key(
        "anthropic-api", "a-key"
    )

    plain = CliRunner().invoke(cli, ["auth", "list"])
    assert plain.exit_code == 0, plain.output
    assert wire.requests == []

    verified = CliRunner().invoke(cli, ["auth", "list", "--verify"])
    assert verified.exit_code == 0, verified.output
    assert "anthropic-api" in verified.output
    assert "accepted" in verified.output
    assert len(wire.requests) == 1


@pytest.mark.parametrize("raw", [b"", b"[]", b"null", b"{}", b'"ok"'])
@pytest.mark.parametrize("provider", PROVIDERS)
async def test_a_200_that_is_not_a_model_listing_is_inconclusive(
    provider: str, raw: bytes, wire: _Wire
) -> None:
    wire.raw = raw
    handler = _handler((provider, "any-key"))

    status = await _verified(handler, provider)

    assert status["available"] is False
    assert status["state"] == "inconclusive"


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_an_empty_stored_key_is_rejected_as_empty_without_a_request(
    provider: str, wire: _Wire
) -> None:
    handler = _handler((provider, ""))

    status = await _verified(handler, provider)

    assert status["state"] == "rejected"
    assert "empty key" in status["reason"]
    assert wire.requests == []


@pytest.mark.parametrize("value", ["false", "true", 1, None, "", [True]])
async def test_verify_must_be_exactly_true_through_the_tool_dispatch(
    value: Any, wire: _Wire
) -> None:
    from llm_orc.mcp.server import MCPServer

    CredentialStorage(ConfigurationManager(provision=False)).store_api_key(
        "anthropic-api", "a-key"
    )
    server = MCPServer(config_manager=ConfigurationManager(provision=False))

    result = await server.call_tool("get_provider_status", {"verify": value})

    assert result["providers"]["anthropic-api"]["state"] == "unchecked"
    assert wire.requests == []


def test_auth_list_refuses_interactive_with_verify(wire: _Wire) -> None:
    from click.testing import CliRunner

    from llm_orc.cli import cli

    result = CliRunner().invoke(cli, ["auth", "list", "-i", "--verify"])

    assert result.exit_code == 2
    assert "--verify cannot be combined with --interactive" in result.output
    assert wire.requests == []


def test_help_documents_the_verify_argument() -> None:
    help_text = OrchestraService().get_help_documentation()["tools"]

    assert "verify=true" in help_text["provider_discovery"]

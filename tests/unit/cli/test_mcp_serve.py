"""``llm-orc mcp serve``: which transport builds a relaying server.

A stdio server has one local client, so it may run a root on a remote
(``remote``). The http transport listens on a network port with many
possible clients, so it must not relay. The command is driven through
``CliRunner`` with ``MCPServer`` replaced by a recorder: the server the
command builds for each transport is what is asserted, and nothing
listens on a port or touches the real config.
"""

from __future__ import annotations

import signal
from typing import Any

import pytest
from click.testing import CliRunner

import llm_orc.mcp.server as server_module
from llm_orc.cli import cli


class _Recorder:
    built: list[dict[str, Any]] = []
    ran: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.built.append(kwargs)

    def run(self, **kwargs: Any) -> None:
        self.ran.append(kwargs)


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> type[_Recorder]:
    _Recorder.built = []
    _Recorder.ran = []
    monkeypatch.setattr(server_module, "MCPServer", _Recorder)
    monkeypatch.setattr(signal, "signal", lambda *_args: None)
    return _Recorder


def _relays(built: dict[str, Any]) -> bool:
    """What ``MCPServer`` would do with the constructor arguments: it
    relays only when told to."""
    return bool(built.get("relay", False))


def test_the_stdio_transport_builds_a_relaying_server(
    recorder: type[_Recorder],
) -> None:
    result = CliRunner().invoke(cli, ["mcp", "serve", "--transport", "stdio"])

    assert result.exit_code == 0, result.output
    (built,) = recorder.built
    assert _relays(built)
    assert recorder.ran == [{}]


def test_the_http_transport_builds_a_server_that_does_not_relay(
    recorder: type[_Recorder],
) -> None:
    result = CliRunner().invoke(
        cli, ["mcp", "serve", "--transport", "http", "--port", "9"]
    )

    assert result.exit_code == 0, result.output
    (built,) = recorder.built
    assert not _relays(built)
    assert recorder.ran == [{"transport": "http", "port": 9}]

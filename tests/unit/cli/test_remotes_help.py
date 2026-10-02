"""The help a new session reads says what a remote is and the three steps
(Arc 6, part 3): the MCP ``get_help`` document and the CLI's top-level
help."""

# ruff: noqa: F811  (imported fixtures are redefined as test parameters)
from __future__ import annotations

import asyncio
import json
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from llm_orc.cli import cli
from llm_orc.mcp.server import MCPServer
from llm_orc.services.orchestra_service import OrchestraService
from tests.unit.services.test_one_run_injection import (  # noqa: F401
    listing,
    project,
    service,
    state_dir,
)

STEPS = ("list_remotes", "--remote", "--preflight")


class TestTheMcpHelpDocument:
    @pytest.fixture
    def document(self, service: OrchestraService) -> str:
        server = MCPServer(service=service, relay=True)
        _, structured = asyncio.run(server._mcp.call_tool("get_help", {}))
        assert isinstance(structured, dict)
        return json.dumps(structured["remotes"])

    @pytest.mark.parametrize(
        "needle",
        [*STEPS, "check_ensemble_runnable", "invoke", "llm-orc remotes", "remotes:"],
    )
    def test_the_remotes_section_names_the_steps_and_the_config(
        self, document: str, needle: str
    ) -> None:
        assert needle in document

    def test_the_section_says_where_the_config_lives(self, document: str) -> None:
        assert "global" in document
        assert "config.yaml" in document

    def test_the_tools_list_names_list_remotes(self, service: OrchestraService) -> None:
        tools = service.get_help_documentation()["tools"]

        assert "list_remotes" in json.dumps(tools)


class TestTheCliHelp:
    @pytest.mark.parametrize("needle", [*STEPS, "llm-orc remotes", "remotes:"])
    def test_dash_dash_help_has_the_section(self, needle: str) -> None:
        result = CliRunner().invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert needle in result.output

    @pytest.mark.parametrize("needle", [*STEPS, "llm-orc remotes", "remotes:"])
    def test_the_help_command_has_the_section(self, needle: str) -> None:
        ctx = Mock()
        ctx.parent = Mock()

        with patch("llm_orc.cli.click.get_current_context", return_value=ctx):
            result = CliRunner().invoke(cli, ["help"])

        assert result.exit_code == 0
        assert needle in result.output

    def test_the_commands_list_has_remotes(self) -> None:
        result = CliRunner().invoke(cli, ["--help"])

        assert "remotes" in result.output.split("Commands:")[1]

    def test_the_invoke_help_has_a_preflight_example(self) -> None:
        result = CliRunner().invoke(cli, ["invoke", "--help"])

        assert "llm-orc invoke review --remote remote-host --preflight" in (
            result.output
        )
        assert "--preflight" in result.output

"""Tests for shared execution utility functions."""

from __future__ import annotations

from typing import Any

from llm_orc.core.execution.utils import dep_name, terminal_agent_names
from llm_orc.schemas.agent_config import LlmAgentConfig


def _agent(name: str, depends_on: list[str | dict[str, Any]] | None = None) -> Any:
    return LlmAgentConfig(
        name=name, model_profile="test-profile", depends_on=depends_on or []
    )


class TestTerminalAgentNames:
    """Terminal nodes: agent names no other agent in the DAG depends on.

    Shared by ``resolve_deliverable``, ``_terminal_agents_for_ensemble``,
    and the ``ensemble:`` agent's own success rule (fail-closed-
    composition B1), so all three read the same terminal set.
    """

    def test_single_agent_is_its_own_terminal(self) -> None:
        agents = [_agent("solo")]
        assert terminal_agent_names(agents) == ["solo"]

    def test_linear_chain_last_node_is_terminal(self) -> None:
        agents = [_agent("a"), _agent("b", depends_on=["a"])]
        assert terminal_agent_names(agents) == ["b"]

    def test_join_node_is_the_sole_terminal(self) -> None:
        agents = [
            _agent("coder"),
            _agent("critic"),
            _agent("synthesizer", depends_on=["coder", "critic"]),
        ]
        assert terminal_agent_names(agents) == ["synthesizer"]

    def test_multiple_terminals_in_declaration_order(self) -> None:
        agents = [
            _agent("root"),
            _agent("left", depends_on=["root"]),
            _agent("right", depends_on=["root"]),
        ]
        assert terminal_agent_names(agents) == ["left", "right"]

    def test_dict_form_dependencies_are_recognized(self) -> None:
        agents = [
            _agent("first"),
            _agent("second", depends_on=[{"agent_name": "first"}]),
        ]
        assert terminal_agent_names(agents) == ["second"]

    def test_empty_agent_list_yields_no_terminals(self) -> None:
        assert terminal_agent_names([]) == []


class TestDepName:
    """Existing coverage for the string/dict dependency-entry accessor."""

    def test_string_dependency(self) -> None:
        assert dep_name("upstream") == "upstream"

    def test_dict_dependency(self) -> None:
        assert dep_name({"agent_name": "upstream"}) == "upstream"

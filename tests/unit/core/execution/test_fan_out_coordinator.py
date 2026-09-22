"""Tests for FanOutCoordinator's contract-failure detection.

Fan-out contract failure fails the agent (fail-closed composition): if
the upstream response can't produce a usable array — it doesn't parse,
isn't an object, the key is missing/not-a-list, or upstream itself
didn't succeed — the fan-out agent is a contract failure, not a silent
zero-instance skip that runs un-expanded. A genuinely empty parsed
array is a legitimate success.
"""

import json
from typing import Any

import pytest

from llm_orc.core.execution.fan_out.coordinator import FanOutCoordinator
from llm_orc.core.execution.fan_out.expander import FanOutExpander
from llm_orc.core.execution.fan_out.gatherer import FanOutGatherer
from llm_orc.schemas.agent_config import LlmAgentConfig


@pytest.fixture
def coordinator() -> FanOutCoordinator:
    expander = FanOutExpander()
    gatherer = FanOutGatherer(expander)
    return FanOutCoordinator(expander, gatherer)


class TestDetectInPhaseWithInputKey:
    """with input_key: parse failure, wrong shape, or missing key fails
    the agent; a genuinely empty list succeeds."""

    def test_unparseable_response_fails_the_agent(
        self, coordinator: FanOutCoordinator
    ) -> None:
        agent = LlmAgentConfig(
            name="searcher",
            model_profile="test",
            depends_on=["decomposer"],
            fan_out=True,
            input_key="queries",
        )
        results_dict: dict[str, Any] = {
            "decomposer": {
                "status": "success",
                "response": "Not JSON, just prose.",
            }
        }

        ready, failed = coordinator.detect_in_phase([agent], results_dict)

        assert ready == []
        assert len(failed) == 1
        failed_agent, message = failed[0]
        assert failed_agent.name == "searcher"
        assert "decomposer" in message
        assert "queries" in message

    def test_non_object_response_fails_the_agent(
        self, coordinator: FanOutCoordinator
    ) -> None:
        agent = LlmAgentConfig(
            name="searcher",
            model_profile="test",
            depends_on=["decomposer"],
            fan_out=True,
            input_key="queries",
        )
        results_dict: dict[str, Any] = {
            "decomposer": {"status": "success", "response": json.dumps(["a", "b"])}
        }

        ready, failed = coordinator.detect_in_phase([agent], results_dict)

        assert ready == []
        assert len(failed) == 1

    def test_missing_key_fails_the_agent(self, coordinator: FanOutCoordinator) -> None:
        agent = LlmAgentConfig(
            name="searcher",
            model_profile="test",
            depends_on=["decomposer"],
            fan_out=True,
            input_key="queries",
        )
        results_dict: dict[str, Any] = {
            "decomposer": {
                "status": "success",
                "response": json.dumps({"other": ["a"]}),
            }
        }

        ready, failed = coordinator.detect_in_phase([agent], results_dict)

        assert ready == []
        assert len(failed) == 1

    def test_key_not_a_list_fails_the_agent(
        self, coordinator: FanOutCoordinator
    ) -> None:
        agent = LlmAgentConfig(
            name="searcher",
            model_profile="test",
            depends_on=["decomposer"],
            fan_out=True,
            input_key="queries",
        )
        results_dict: dict[str, Any] = {
            "decomposer": {
                "status": "success",
                "response": json.dumps({"queries": "not-a-list"}),
            }
        }

        ready, failed = coordinator.detect_in_phase([agent], results_dict)

        assert ready == []
        assert len(failed) == 1

    def test_genuinely_empty_list_succeeds(
        self, coordinator: FanOutCoordinator
    ) -> None:
        agent = LlmAgentConfig(
            name="searcher",
            model_profile="test",
            depends_on=["decomposer"],
            fan_out=True,
            input_key="queries",
        )
        results_dict: dict[str, Any] = {
            "decomposer": {
                "status": "success",
                "response": json.dumps({"queries": []}),
            }
        }

        ready, failed = coordinator.detect_in_phase([agent], results_dict)

        assert failed == []
        assert len(ready) == 1
        assert ready[0][1] == []

    def test_no_markdown_fence_stripping(self, coordinator: FanOutCoordinator) -> None:
        """No lenient parsing: fenced JSON still fails closed."""
        agent = LlmAgentConfig(
            name="searcher",
            model_profile="test",
            depends_on=["decomposer"],
            fan_out=True,
            input_key="queries",
        )
        results_dict: dict[str, Any] = {
            "decomposer": {
                "status": "success",
                "response": '```json\n{"queries": ["a", "b"]}\n```',
            }
        }

        ready, failed = coordinator.detect_in_phase([agent], results_dict)

        assert ready == []
        assert len(failed) == 1


class TestDetectInPhaseWithoutInputKey:
    """Without input_key: parse_array_from_result returning None fails
    the agent; [] still succeeds."""

    def test_non_array_response_fails_the_agent(
        self, coordinator: FanOutCoordinator
    ) -> None:
        agent = LlmAgentConfig(
            name="processor",
            model_profile="test",
            depends_on=["upstream"],
            fan_out=True,
        )
        results_dict: dict[str, Any] = {
            "upstream": {"status": "success", "response": "not an array"}
        }

        ready, failed = coordinator.detect_in_phase([agent], results_dict)

        assert ready == []
        assert len(failed) == 1
        assert "upstream" in failed[0][1]

    def test_empty_array_succeeds(self, coordinator: FanOutCoordinator) -> None:
        agent = LlmAgentConfig(
            name="processor",
            model_profile="test",
            depends_on=["upstream"],
            fan_out=True,
        )
        results_dict: dict[str, Any] = {
            "upstream": {"status": "success", "response": json.dumps([])}
        }

        ready, failed = coordinator.detect_in_phase([agent], results_dict)

        assert failed == []
        assert len(ready) == 1
        assert ready[0][1] == []


class TestDetectInPhaseUpstreamFailed:
    """A fan-out agent whose upstream itself failed is also a contract
    failure — it must not run un-expanded on garbage/missing data."""

    def test_upstream_failed_fails_the_fan_out_agent(
        self, coordinator: FanOutCoordinator
    ) -> None:
        agent = LlmAgentConfig(
            name="searcher",
            model_profile="test",
            depends_on=["decomposer"],
            fan_out=True,
            input_key="queries",
        )
        results_dict: dict[str, Any] = {
            "decomposer": {
                "status": "failed",
                "response": None,
                "error": "model unavailable",
            }
        }

        ready, failed = coordinator.detect_in_phase([agent], results_dict)

        assert ready == []
        assert len(failed) == 1
        failed_agent, message = failed[0]
        assert failed_agent.name == "searcher"
        assert "decomposer" in message

"""Tests for fan-out agent expansion (issue #73)."""

import json
from typing import Any

import pytest

from llm_orc.core.execution.fan_out_expander import FanOutExpander


class TestFanOutExpander:
    """Test FanOutExpander functionality."""

    @pytest.fixture
    def expander(self) -> FanOutExpander:
        """Create a FanOutExpander instance."""
        return FanOutExpander()

    def test_detect_fan_out_agents_returns_names(
        self, expander: FanOutExpander
    ) -> None:
        """detect_fan_out_agents returns names of agents with fan_out: true."""
        agents: list[dict[str, Any]] = [
            {"name": "chunker", "script": "split.py"},
            {"name": "extractor", "model_profile": "test", "fan_out": True},
            {"name": "synthesizer", "model_profile": "test"},
            {"name": "analyzer", "model_profile": "test", "fan_out": True},
        ]

        result = expander.detect_fan_out_agents(agents)

        assert result == ["extractor", "analyzer"]

    def test_detect_fan_out_agents_empty_list(self, expander: FanOutExpander) -> None:
        """detect_fan_out_agents returns empty list when no fan_out agents."""
        agents: list[dict[str, Any]] = [
            {"name": "agent1", "model_profile": "test"},
            {"name": "agent2", "model_profile": "test", "fan_out": False},
        ]

        result = expander.detect_fan_out_agents(agents)

        assert result == []

    def test_is_array_result_true_for_json_array_string(
        self, expander: FanOutExpander
    ) -> None:
        """is_array_result returns True for JSON array string."""
        result = expander.is_array_result('["chunk1", "chunk2", "chunk3"]')

        assert result is True

    def test_is_array_result_true_for_python_list(
        self, expander: FanOutExpander
    ) -> None:
        """is_array_result returns True for Python list."""
        result = expander.is_array_result(["chunk1", "chunk2"])

        assert result is True

    def test_is_array_result_false_for_dict(self, expander: FanOutExpander) -> None:
        """is_array_result returns False for dict."""
        result = expander.is_array_result({"key": "value"})

        assert result is False

    def test_is_array_result_false_for_json_object_string(
        self, expander: FanOutExpander
    ) -> None:
        """is_array_result returns False for JSON object string."""
        result = expander.is_array_result('{"key": "value"}')

        assert result is False

    def test_is_array_result_false_for_plain_string(
        self, expander: FanOutExpander
    ) -> None:
        """is_array_result returns False for plain string."""
        result = expander.is_array_result("just a plain string")

        assert result is False

    def test_is_array_result_false_for_empty_array(
        self, expander: FanOutExpander
    ) -> None:
        """is_array_result returns False for empty array (nothing to fan out)."""
        result = expander.is_array_result([])

        assert result is False

    def test_expand_fan_out_agent_creates_indexed_copies(
        self, expander: FanOutExpander
    ) -> None:
        """expand_fan_out_agent creates N indexed copies of agent config."""
        agent_config: dict[str, Any] = {
            "name": "extractor",
            "model_profile": "ollama-llama3",
            "fan_out": True,
            "depends_on": ["chunker"],
            "system_prompt": "Extract concepts",
        }
        upstream_array = ["chunk1", "chunk2", "chunk3"]

        result = expander.expand_fan_out_agent(agent_config, upstream_array)

        assert len(result) == 3
        assert result[0]["name"] == "extractor[0]"
        assert result[1]["name"] == "extractor[1]"
        assert result[2]["name"] == "extractor[2]"
        # Original config preserved
        assert result[0]["model_profile"] == "ollama-llama3"
        assert result[0]["system_prompt"] == "Extract concepts"
        # fan_out removed from instances
        assert "fan_out" not in result[0]

    def test_expand_fan_out_agent_stores_chunk_info(
        self, expander: FanOutExpander
    ) -> None:
        """expand_fan_out_agent stores chunk metadata in each instance."""
        agent_config: dict[str, Any] = {
            "name": "extractor",
            "model_profile": "test",
            "fan_out": True,
            "depends_on": ["chunker"],
        }
        upstream_array = ["chunk_a", "chunk_b"]

        result = expander.expand_fan_out_agent(agent_config, upstream_array)

        assert result[0]["_fan_out_chunk"] == "chunk_a"
        assert result[0]["_fan_out_index"] == 0
        assert result[0]["_fan_out_total"] == 2
        assert result[0]["_fan_out_original"] == "extractor"

        assert result[1]["_fan_out_chunk"] == "chunk_b"
        assert result[1]["_fan_out_index"] == 1
        assert result[1]["_fan_out_total"] == 2
        assert result[1]["_fan_out_original"] == "extractor"

    def test_prepare_instance_input_includes_chunk_metadata(
        self, expander: FanOutExpander
    ) -> None:
        """prepare_instance_input includes chunk index and total."""
        result = expander.prepare_instance_input(
            chunk="Scene 1 content",
            chunk_index=0,
            total_chunks=5,
            base_input="Analyze this play",
        )

        assert result["input"] == "Scene 1 content"
        assert result["chunk_index"] == 0
        assert result["total_chunks"] == 5
        assert result["base_input"] == "Analyze this play"

    def test_prepare_instance_input_with_dict_chunk(
        self, expander: FanOutExpander
    ) -> None:
        """prepare_instance_input handles dict chunk content."""
        chunk = {"scene": "Act 1 Scene 2", "text": "Some dialogue"}
        result = expander.prepare_instance_input(
            chunk=chunk,
            chunk_index=2,
            total_chunks=10,
            base_input="Extract themes",
        )

        assert result["input"] == chunk
        assert result["chunk_index"] == 2
        assert result["total_chunks"] == 10

    def test_is_fan_out_instance_name_pattern_matching(
        self, expander: FanOutExpander
    ) -> None:
        """is_fan_out_instance_name correctly identifies instance names."""
        assert expander.is_fan_out_instance_name("extractor[0]") is True
        assert expander.is_fan_out_instance_name("extractor[42]") is True
        assert expander.is_fan_out_instance_name("my-agent[123]") is True
        assert expander.is_fan_out_instance_name("extractor") is False
        assert expander.is_fan_out_instance_name("extractor[]") is False
        assert expander.is_fan_out_instance_name("extractor[abc]") is False
        assert expander.is_fan_out_instance_name("[0]") is False

    def test_get_original_agent_name_extraction(self, expander: FanOutExpander) -> None:
        """get_original_agent_name extracts original name from instance name."""
        assert expander.get_original_agent_name("extractor[0]") == "extractor"
        assert expander.get_original_agent_name("extractor[42]") == "extractor"
        assert expander.get_original_agent_name("my-agent[5]") == "my-agent"

    def test_get_original_agent_name_returns_input_if_not_instance(
        self, expander: FanOutExpander
    ) -> None:
        """get_original_agent_name returns input unchanged if not instance name."""
        assert expander.get_original_agent_name("extractor") == "extractor"
        assert expander.get_original_agent_name("my-agent") == "my-agent"

    def test_get_instance_index(self, expander: FanOutExpander) -> None:
        """get_instance_index extracts index from instance name."""
        assert expander.get_instance_index("extractor[0]") == 0
        assert expander.get_instance_index("extractor[42]") == 42
        assert expander.get_instance_index("my-agent[123]") == 123

    def test_get_instance_index_returns_none_if_not_instance(
        self, expander: FanOutExpander
    ) -> None:
        """get_instance_index returns None if not an instance name."""
        assert expander.get_instance_index("extractor") is None
        assert expander.get_instance_index("my-agent") is None

    def test_parse_array_from_result_json_string(
        self, expander: FanOutExpander
    ) -> None:
        """parse_array_from_result parses JSON array string."""
        result = expander.parse_array_from_result('["a", "b", "c"]')

        assert result == ["a", "b", "c"]

    def test_parse_array_from_result_list(self, expander: FanOutExpander) -> None:
        """parse_array_from_result returns list as-is."""
        result = expander.parse_array_from_result(["a", "b"])

        assert result == ["a", "b"]

    def test_parse_array_from_result_script_output(
        self, expander: FanOutExpander
    ) -> None:
        """parse_array_from_result handles ScriptAgentOutput format."""
        script_output = json.dumps(
            {
                "success": True,
                "data": ["chunk1", "chunk2"],
            }
        )
        result = expander.parse_array_from_result(script_output)

        assert result == ["chunk1", "chunk2"]

    def test_parse_array_from_result_returns_none_for_non_array(
        self, expander: FanOutExpander
    ) -> None:
        """parse_array_from_result returns None for non-array input."""
        assert expander.parse_array_from_result("plain string") is None
        assert expander.parse_array_from_result('{"key": "value"}') is None
        assert expander.parse_array_from_result({"key": "value"}) is None

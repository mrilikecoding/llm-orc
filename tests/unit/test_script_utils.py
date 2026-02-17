"""Tests for llm_orc.script_utils envelope unwrapping."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest

from llm_orc.script_utils import unwrap_input


class TestUnwrapInputFormat1:
    """ScriptAgentInput format: {"agent_name": ..., "input_data": "<json>"}."""

    def test_extracts_data_from_input_data_field(self) -> None:
        graph: dict[str, Any] = {"nodes": [{"id": "a"}], "edges": []}
        envelope = {
            "agent_name": "pagerank",
            "input_data": json.dumps(graph),
            "context": {},
            "dependencies": {},
        }
        data, params = unwrap_input(json.dumps(envelope))
        assert data == graph
        assert params == {}

    def test_extracts_parameters_from_envelope(self) -> None:
        graph: dict[str, Any] = {"nodes": [], "edges": []}
        envelope = {
            "agent_name": "pagerank",
            "input_data": json.dumps(graph),
            "parameters": {"damping": 0.9},
        }
        data, params = unwrap_input(json.dumps(envelope))
        assert data == graph
        assert params == {"damping": 0.9}


class TestUnwrapInputFormat2:
    """Legacy wrapper: {"input": ..., "parameters": {...}}."""

    def test_extracts_string_input(self) -> None:
        graph: dict[str, Any] = {"nodes": [{"id": "b"}], "edges": []}
        envelope = {
            "input": json.dumps(graph),
            "parameters": {"resolution": 1.0},
        }
        data, params = unwrap_input(json.dumps(envelope))
        assert data == graph
        assert params == {"resolution": 1.0}

    def test_extracts_dict_input(self) -> None:
        graph: dict[str, Any] = {"nodes": [{"id": "c"}], "edges": []}
        envelope = {
            "input": graph,
            "parameters": {"k": "v"},
        }
        data, params = unwrap_input(json.dumps(envelope))
        assert data == graph
        assert params == {"k": "v"}

    def test_non_json_string_input_returns_envelope(self) -> None:
        envelope = {
            "input": "not valid json",
            "parameters": {"k": "v"},
        }
        data, params = unwrap_input(json.dumps(envelope))
        assert data == envelope
        assert params == {"k": "v"}


class TestUnwrapInputFormat3:
    """Direct format: envelope IS the data."""

    def test_passes_through_direct_data(self) -> None:
        graph: dict[str, Any] = {
            "nodes": [{"id": "d"}],
            "edges": [{"source": "d", "target": "d"}],
        }
        data, params = unwrap_input(json.dumps(graph))
        assert data == graph
        assert params == {}

    def test_empty_input_returns_empty(self) -> None:
        data, params = unwrap_input("  ")
        assert data == {}
        assert params == {}


class TestUnwrapInputEnvFallback:
    """AGENT_PARAMETERS env var fallback."""

    def test_reads_params_from_env_when_envelope_has_none(self) -> None:
        graph: dict[str, Any] = {"nodes": [], "edges": []}
        with patch.dict("os.environ", {"AGENT_PARAMETERS": '{"damping": 0.7}'}):
            data, params = unwrap_input(json.dumps(graph))
        assert params == {"damping": 0.7}

    def test_envelope_params_take_precedence_over_env(self) -> None:
        graph: dict[str, Any] = {"nodes": [], "edges": []}
        envelope = {
            "agent_name": "x",
            "input_data": json.dumps(graph),
            "parameters": {"damping": 0.85},
        }
        with patch.dict("os.environ", {"AGENT_PARAMETERS": '{"damping": 0.5}'}):
            _, params = unwrap_input(json.dumps(envelope))
        assert params == {"damping": 0.85}

    def test_invalid_env_params_ignored(self) -> None:
        graph: dict[str, Any] = {"nodes": [], "edges": []}
        with patch.dict("os.environ", {"AGENT_PARAMETERS": "not json"}):
            _, params = unwrap_input(json.dumps(graph))
        assert params == {}


class TestUnwrapInputEdgeCases:
    """Edge cases in envelope unwrapping."""

    def test_legacy_input_as_non_string_non_dict_returns_envelope(self) -> None:
        envelope: dict[str, Any] = {"input": 42, "parameters": {"k": "v"}}
        data, params = unwrap_input(json.dumps(envelope))
        assert data == envelope
        assert params == {"k": "v"}


class TestUnwrapInputDebug:
    """Debug logging to stderr."""

    def test_debug_logs_to_stderr(self, capsys: pytest.CaptureFixture[str]) -> None:
        graph: dict[str, Any] = {"nodes": [], "edges": []}
        unwrap_input(json.dumps(graph), debug=True)
        captured = capsys.readouterr()
        assert "[llm-orc debug] envelope:" in captured.err
        assert "[llm-orc debug] unwrapped:" in captured.err

    def test_no_debug_output_by_default(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        graph: dict[str, Any] = {"nodes": [], "edges": []}
        unwrap_input(json.dumps(graph))
        captured = capsys.readouterr()
        assert captured.err == ""

    def test_debug_logs_dict_input_keys(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        envelope: dict[str, Any] = {
            "input": {"nodes": [], "edges": []},
            "parameters": {},
        }
        unwrap_input(json.dumps(envelope), debug=True)
        captured = capsys.readouterr()
        assert "input_keys" in captured.err

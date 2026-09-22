"""Issue #202, second cause: web_searcher cannot read the envelope.

A root script agent inside a child ensemble receives the ScriptAgentInput
envelope (``{"agent_name", "input_data", "context", "dependencies"}``),
not the flat dispatch payload ``{"query": ...}``. _extract_query read only
query/parameters/input/data and returned '' for every envelope shape, so
the web-searcher ensemble searched for nothing when composed through
``ensemble:`` + ``input_key:``.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[3]
WEB_SEARCHER = REPO / ".llm-orc" / "scripts" / "agentic_serving" / "web_searcher.py"

_spec = importlib.util.spec_from_file_location("web_searcher", WEB_SEARCHER)
assert _spec is not None
assert _spec.loader is not None
web_searcher = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(web_searcher)

extract_query = web_searcher._extract_query
MultipleQueriesError = web_searcher.MultipleQueriesError


def _envelope(input_data: Any) -> dict[str, Any]:
    """ScriptAgentInput envelope as the root script agent receives it."""
    raw = input_data if isinstance(input_data, str) else json.dumps(input_data)
    return {
        "agent_name": "web_search",
        "input_data": raw,
        "context": {},
        "dependencies": {},
    }


@pytest.mark.parametrize(
    ("child_input", "expected"),
    [
        pytest.param(["hello world"], "hello world", id="single-item-list"),
        pytest.param("hello world", "hello world", id="plain-string"),
        pytest.param({"query": "hello world"}, "hello world", id="query-dict"),
    ],
)
def test_extract_query_from_child_envelope(child_input: Any, expected: str) -> None:
    """The issue's three child-input shapes all yield the query."""
    assert extract_query(_envelope(child_input)) == expected


class TestEngineDispatchPayloadUnwrapsJson:
    """The engine pipes a root script agent {"input": <child input>,
    "parameters": ...} (agent_runner.py). When the child input is an
    input_key-selected array (issue #202's web-searcher shape), `input`
    arrives JSON-encoded — the query is the item, not the JSON text."""

    def test_input_json_encoded_list_unwraps_first_item(self) -> None:
        assert extract_query({"input": '["hello world"]'}) == "hello world"

    def test_input_json_encoded_dict_unwraps_query(self) -> None:
        assert extract_query({"input": '{"query": "hello world"}'}) == "hello world"

    def test_input_json_encoded_empty_list_is_empty_query(self) -> None:
        assert extract_query({"input": "[]"}) == ""

    def test_input_plain_prose_is_unchanged(self) -> None:
        assert extract_query({"input": "just the prompt"}) == "just the prompt"


class TestListOfDictItems:
    """input_key-selected arrays whose items are dicts (e.g. structured
    results) unwrap their query key, consistent with the dict path
    (review finding F2)."""

    def test_list_of_dict_first_item_unwraps_query_key(self) -> None:
        assert extract_query(_envelope([{"query": "hello world"}])) == "hello world"

    def test_input_json_encoded_list_of_dict_unwraps(self) -> None:
        assert extract_query({"input": '[{"query": "hello world"}]'}) == "hello world"

    def test_list_of_dict_without_query_key_yields_empty(self) -> None:
        """A dict item with no query key unwraps to '' — main() emits
        missing_query rather than searching JSON text."""
        assert extract_query(_envelope([{"other": "x"}])) == ""


class TestMultiQueryListIsStructuredError:
    """A selected array with more than one query must not be silently
    truncated to its first item (PR 203 review blocker). One search runs
    per invocation; N queries compose via fan_out: true (ADR-014)."""

    def test_envelope_multi_item_list_raises(self) -> None:
        with pytest.raises(MultipleQueriesError):
            extract_query(_envelope(["q1", "q2"]))

    def test_envelope_multi_dict_list_raises(self) -> None:
        with pytest.raises(MultipleQueriesError):
            extract_query(_envelope([{"query": "a"}, {"query": "b"}]))

    def test_input_json_encoded_multi_item_raises(self) -> None:
        with pytest.raises(MultipleQueriesError):
            extract_query({"input": '["q1", "q2"]'})

    def test_direct_list_payload_multi_item_raises(self) -> None:
        with pytest.raises(MultipleQueriesError):
            extract_query({"input_data": ["q1", "q2"]})

    def test_single_item_list_still_extracts(self) -> None:
        """The one-query case is unchanged by the multi-query guard."""
        assert extract_query(_envelope(["q1"])) == "q1"


def test_main_emits_multiple_queries_error(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """main() converts MultipleQueriesError into a structured error the
    orchestrator can route on, pointing at the fan_out composition."""
    monkeypatch.delenv("WEB_SEARCH_BACKEND", raising=False)
    real_stdin = sys.stdin
    sys.stdin = io.StringIO(json.dumps(_envelope(["q1", "q2"])))
    try:
        assert web_searcher.main() == 0
    finally:
        sys.stdin = real_stdin
    out = json.loads(capsys.readouterr().out)
    assert out["error"] == "multiple_queries"
    assert out["backend"] == "ddgs"
    assert "fan_out" in out["detail"]


def test_extract_query_direct_dispatch_shapes_unchanged() -> None:
    """Existing dispatch conventions keep their precedence."""
    assert extract_query({"query": "flat"}) == "flat"
    assert extract_query({"parameters": {"query": "nested"}}) == "nested"
    assert extract_query({"input": "prompt"}) == "prompt"
    assert extract_query({"data": "payload"}) == "payload"


def test_extract_query_empty_for_nothing_usable() -> None:
    """Nothing usable stays '' — main() emits missing_query."""
    assert extract_query({}) == ""
    assert extract_query(_envelope([])) == ""

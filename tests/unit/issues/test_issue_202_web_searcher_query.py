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
import json
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

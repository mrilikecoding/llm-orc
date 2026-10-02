"""The CLI displays a result document and the agent list, with no
executor or config in sight (Arc 5, Task 4), so a remote serve's answer
prints as a local run does."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from llm_orc.cli_modules.utils.visualization.streaming import (
    display_result,
    run_standard_execution,
)
from llm_orc.schemas.agent_config import LlmAgentConfig

AGENTS = [LlmAgentConfig(name="writer", model_profile="p")]

DOCUMENT: dict[str, Any] = {
    "results": {"writer": {"status": "success", "response": "the answer"}},
    "metadata": {"duration": "1.50s", "usage": {"totals": {"total_tokens": 7}}},
    "deliverable": "the answer",
    "status": "success",
    "has_errors": False,
}


def test_json_prints_the_document_with_its_metadata(
    capsys: pytest.CaptureFixture[str],
) -> None:
    display_result(DOCUMENT, AGENTS, "json", True)

    printed = json.loads(capsys.readouterr().out)
    assert printed["metadata"] == DOCUMENT["metadata"]
    assert printed["deliverable"] == "the answer"
    assert printed["status"] == "success"
    assert "config" not in printed


def test_text_prints_the_answer_and_the_run_metrics(
    capsys: pytest.CaptureFixture[str],
) -> None:
    display_result(DOCUMENT, AGENTS, "text", True)

    out = capsys.readouterr().out
    assert "the answer" in out
    assert "Duration" in out or "1.50s" in out


def test_rich_prints_the_answer(capsys: pytest.CaptureFixture[str]) -> None:
    display_result(DOCUMENT, AGENTS, "rich", True)

    assert "the answer" in capsys.readouterr().out


@pytest.mark.parametrize("output_format", ["json", "text", "rich"])
async def test_a_local_run_prints_what_its_document_prints(
    output_format: str, capsys: pytest.CaptureFixture[str]
) -> None:
    executor = AsyncMock()
    raw = {k: v for k, v in DOCUMENT.items() if k != "has_errors"}
    executor.execute = AsyncMock(return_value={**raw, "status": "completed"})
    config = SimpleNamespace(agents=AGENTS, to_dict=lambda: {"name": "top"})

    await run_standard_execution(executor, config, "hi", output_format, True)
    local = capsys.readouterr().out
    display_result(DOCUMENT, AGENTS, output_format, True, config)
    from_document = capsys.readouterr().out

    assert local == from_document

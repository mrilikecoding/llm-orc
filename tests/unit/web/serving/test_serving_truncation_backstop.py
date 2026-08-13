"""End-to-end test for the runtime prompt-truncation backstop (#151's
core, review round 2 Part 2, #145).

Drives ServingEnsembleCaller._serve() directly with a mocked executor
result shaped like a real turn whose recorded prompt_eval_count shows the
runtime silently processed a fraction of the dispatched prompt — pins
that the caller discards the pipeline's own answer and refuses loudly
instead, never leaking the withheld text.
"""

from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from llm_orc.core.execution.executor_factory import ExecutorFactory
from llm_orc.web.serving.serving_ensemble_caller import ServingEnsembleCaller

_WITHHELD_ANSWER = "This is the answer that must never reach the client."


def _canned_result(*, dispatch_input: str, prompt_eval_count: int) -> dict[str, Any]:
    classify_response = json.dumps(
        {
            "target": "explainer",
            "chain": "explain",
            "step_index": 0,
            "dispatch_input": dispatch_input,
        }
    )
    child_result = {
        "results": {"explainer": {"response": _WITHHELD_ANSWER, "status": "success"}},
        "metadata": {
            "usage": {"agents": {"explainer": {"prompt_eval_count": prompt_eval_count}}}
        },
    }
    return {
        "results": {
            "classify": {"status": "success", "response": classify_response},
            "seat": {"status": "success", "response": json.dumps(child_result)},
            "emit": {
                "status": "success",
                "response": json.dumps({"finish": True, "content": _WITHHELD_ANSWER}),
            },
        },
        "execution_order": ["classify", "seat", "emit"],
    }


def _wire_fake_executor(
    monkeypatch: pytest.MonkeyPatch, result: dict[str, Any]
) -> None:
    fake_executor = types.SimpleNamespace(execute=AsyncMock(return_value=result))
    monkeypatch.setattr(
        ExecutorFactory,
        "create_root_executor",
        lambda **_kwargs: fake_executor,
    )


@pytest.mark.asyncio
async def test_deep_truncation_discards_the_answer_and_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    caller = ServingEnsembleCaller(project_dir=tmp_path, trace_root=tmp_path / ".trace")
    caller._load_config = lambda: types.SimpleNamespace(name="serving")  # type: ignore[method-assign]
    result = _canned_result(
        dispatch_input="word " * 30000,  # projects well past 40K
        prompt_eval_count=5000,  # far below 42% of that -> trigger 1
    )
    _wire_fake_executor(monkeypatch, result)

    outcome = await caller._serve(task="explain big.py", conversation="")

    assert outcome["finish"] is True
    assert "context window overflow" in outcome["content"]
    assert "Refused:" in outcome["content"]
    assert _WITHHELD_ANSWER not in outcome["content"]


@pytest.mark.asyncio
async def test_discard_signature_discards_the_answer_and_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Trigger 2: prompt_eval_count landing at the measured discard
    signature (~half the window) against a near-window-sized dispatch."""
    caller = ServingEnsembleCaller(project_dir=tmp_path, trace_root=tmp_path / ".trace")
    caller._load_config = lambda: types.SimpleNamespace(name="serving")  # type: ignore[method-assign]
    result = _canned_result(
        dispatch_input="word " * 26000,  # projects to > 41,000 (see below)
        prompt_eval_count=20482,
    )
    _wire_fake_executor(monkeypatch, result)

    outcome = await caller._serve(task="explain big.py", conversation="")

    assert outcome["finish"] is True
    assert "context window overflow" in outcome["content"]
    assert _WITHHELD_ANSWER not in outcome["content"]


@pytest.mark.asyncio
async def test_in_window_call_answers_normally_no_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    caller = ServingEnsembleCaller(project_dir=tmp_path, trace_root=tmp_path / ".trace")
    caller._load_config = lambda: types.SimpleNamespace(name="serving")  # type: ignore[method-assign]
    result = _canned_result(
        dispatch_input="explain the divide function",
        prompt_eval_count=25,  # matches a small real prompt closely
    )
    _wire_fake_executor(monkeypatch, result)

    outcome = await caller._serve(task="explain big.py", conversation="")

    assert outcome == {"finish": True, "content": _WITHHELD_ANSWER}

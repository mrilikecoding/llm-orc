"""Unit tests for the runtime prompt-truncation backstop (#151's core,
review round 2 Part 2, #145).

The read-accumulator budget (token_estimate.projected_tokens_v2) is a
PRE-FLIGHT guard — a held read's projected size can still admit while the
ACTUAL dispatched prompt (system/role wrapping, conversation history the
accumulator doesn't project) crosses the real model window. This is the
general backstop no estimator can evade: compare Ollama's own recorded
prompt_eval_count (C2) against the projected size of what classify
actually dispatched, and flag the turn when either trigger fires.
"""

from __future__ import annotations

import json
from typing import Any

from llm_orc.web.serving.turn_trace import (
    WINDOW,
    _truncation_check,
    _truncation_detected,
    build_turn_trace,
)


def test_deep_truncation_ratio_trigger_fires() -> None:
    # trigger 1: prompt_eval_count far below the projected prompt size —
    # catches deep overflow directly, independent of window arithmetic.
    assert _truncation_detected(prompt_eval_count=5000, projected_prompt_tokens=47700)


def test_discard_signature_trigger_fires() -> None:
    # trigger 2: the measured discard SIGNATURE — both real captured
    # over-window prompts returned exactly prompt_eval_count == WINDOW//2
    # + 2, regardless of ratio. 20,482 vs a 41,000-token projected prompt
    # gives ratio ~0.4996 (ABOVE the 0.42 ratio floor — trigger 1 alone
    # would miss this), but it lands within the signature band and the
    # prompt is plausibly near-window sized, so trigger 2 catches it.
    assert _truncation_detected(prompt_eval_count=20482, projected_prompt_tokens=41000)


def test_in_window_cjk_shaped_ratio_does_not_fire() -> None:
    # estimator v2's most over-projected measured density class (CJK+code,
    # ~2.15x real — test_token_estimate_ground_truth.py) puts a legitimate
    # in-window call's ratio as low as ~0.465. This fixture sits just
    # above the 0.42 floor and well under the near-window gate for trigger
    # 2 (10,000 << WINDOW * 0.8) — neither trigger may fire.
    assert not _truncation_detected(
        prompt_eval_count=4650, projected_prompt_tokens=10000
    )


def test_in_window_normal_ratio_does_not_fire() -> None:
    assert not _truncation_detected(
        prompt_eval_count=9000, projected_prompt_tokens=10000
    )


def test_ratio_at_ceiling_of_the_deep_trigger_does_not_fire() -> None:
    # boundary: exactly at the 0.42 floor is NOT "less than" it — the
    # trigger is a strict `<`, so equality must not fire (paired with the
    # deep-overflow test above, which fires comfortably below it).
    assert not _truncation_detected(
        prompt_eval_count=4200, projected_prompt_tokens=10000
    )


def test_signature_trigger_requires_the_near_window_gate() -> None:
    # a SMALL legitimate call whose prompt_eval_count happens to land near
    # WINDOW//2 must never trip trigger 2 on the signature alone — the
    # near-window gate (projected > WINDOW * 0.8) exists exactly to
    # prevent this false positive.
    assert not _truncation_detected(
        prompt_eval_count=WINDOW // 2, projected_prompt_tokens=5000
    )


def test_signature_band_edge_fires_inclusive() -> None:
    projected = int(WINDOW * 0.9)
    assert _truncation_detected(
        prompt_eval_count=WINDOW // 2 + 64, projected_prompt_tokens=projected
    )


def test_signature_band_edge_plus_one_does_not_fire_on_trigger_2_alone() -> None:
    # one past the band, and the ratio (>> 0.42 for this projected size)
    # doesn't trip trigger 1 either.
    projected = int(WINDOW * 0.9)
    prompt_eval_count = WINDOW // 2 + 65
    assert prompt_eval_count >= projected * 0.42  # trigger 1 does not fire
    assert not _truncation_detected(
        prompt_eval_count=prompt_eval_count, projected_prompt_tokens=projected
    )


def test_zero_projected_never_flags() -> None:
    # no dispatch_input to project from -> nothing to compare against;
    # must never divide by zero or produce a spurious flag.
    assert not _truncation_detected(prompt_eval_count=0, projected_prompt_tokens=0)


def test_truncation_check_flags_a_turn_with_deep_overflow() -> None:
    classify_response = json.dumps(
        {
            "target": "explainer",
            "chain": "explain",
            "step_index": 0,
            "dispatch_input": "word " * 30000,
        }
    )
    nodes: list[dict[str, Any]] = [{"node": "classify", "response": classify_response}]
    child_usage_node = [{"node": "explainer", "prompt_eval_count": 5000}]
    nodes.append({"node": "seat", "seat": child_usage_node})

    detail = _truncation_check(nodes, classify_response)

    assert detail is not None
    assert detail["prompt_eval_counts"] == [5000]
    assert detail["projected_prompt_tokens"] > 40000


def test_truncation_check_returns_none_without_dispatch_input() -> None:
    nodes = [
        {"node": "seat", "seat": [{"node": "explainer", "prompt_eval_count": 5000}]}
    ]
    assert _truncation_check(nodes, None) is None
    assert _truncation_check(nodes, "not json") is None


def test_truncation_check_returns_none_without_recorded_usage() -> None:
    classify_response = json.dumps(
        {"target": "explainer", "dispatch_input": "word " * 30000}
    )
    nodes = [{"node": "classify", "response": classify_response}]
    assert _truncation_check(nodes, classify_response) is None


def test_build_turn_trace_surfaces_truncation_detected() -> None:
    classify_response = json.dumps(
        {
            "target": "explainer",
            "chain": "explain",
            "step_index": 0,
            "dispatch_input": "word " * 30000,
        }
    )
    child_result = {
        "results": {"explainer": {"response": "an answer", "status": "success"}},
        "metadata": {"usage": {"agents": {"explainer": {"prompt_eval_count": 5000}}}},
    }
    result = {
        "results": {
            "classify": {"status": "success", "response": classify_response},
            "seat": {
                "status": "success",
                "response": json.dumps(child_result),
            },
        },
    }

    trace = build_turn_trace("serving", result)

    assert trace["truncation_detected"] is True
    assert trace["truncation_detail"]["prompt_eval_counts"] == [5000]


def test_build_turn_trace_omits_truncation_fields_when_not_detected() -> None:
    classify_response = json.dumps(
        {
            "target": "explainer",
            "chain": "explain",
            "step_index": 0,
            "dispatch_input": "hi",
        }
    )
    child_result = {
        "results": {"explainer": {"response": "an answer", "status": "success"}},
        "metadata": {"usage": {"agents": {"explainer": {"prompt_eval_count": 20}}}},
    }
    result = {
        "results": {
            "classify": {"status": "success", "response": classify_response},
            "seat": {
                "status": "success",
                "response": json.dumps(child_result),
            },
        },
    }

    trace = build_turn_trace("serving", result)

    assert "truncation_detected" not in trace
    assert "truncation_detail" not in trace

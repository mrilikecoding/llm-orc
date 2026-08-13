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
import math
from typing import Any

from llm_orc.web.serving.token_estimate import projected_tokens_v2
from llm_orc.web.serving.turn_trace import (
    WINDOW,
    _truncation_check,
    _truncation_detected,
    build_turn_trace,
)


def test_deep_truncation_ratio_trigger_fires() -> None:
    # trigger 1: prompt_eval_count far below the projected prompt size,
    # on a near-window-sized dispatch — catches deep overflow directly.
    assert _truncation_detected(prompt_eval_count=5000, projected_prompt_tokens=47700)


def test_discard_signature_trigger_fires() -> None:
    # trigger 2: the measured discard SIGNATURE — both real captured
    # over-window prompts returned exactly prompt_eval_count == WINDOW//2
    # + 2, regardless of ratio. 20,482 vs a 41,000-token projected prompt
    # gives ratio ~0.4996 (ABOVE trigger 1's 0.35 ratio floor — trigger 1
    # alone would miss this), but it lands within the signature band and
    # the prompt is plausibly near-window sized, so trigger 2 catches it.
    assert _truncation_detected(prompt_eval_count=20482, projected_prompt_tokens=41000)


def test_in_window_cjk_shaped_ratio_does_not_fire() -> None:
    # review round 3 minor 1: the measured in-window floor for estimator
    # v2's most over-projected density class (CJK+code) came in at 0.438
    # (superseding the earlier ~0.465 estimate). This fixture sits at
    # that exact floor on a NEAR-WINDOW-sized dispatch (projected >
    # WINDOW * 0.8, so trigger 1's gate is satisfied and this genuinely
    # exercises the ratio comparison, not just the gate) — 0.35 leaves
    # ~25% headroom below it, comfortably clear.
    projected = int(WINDOW * 0.85)
    prompt_eval_count = int(projected * 0.438)
    assert not _truncation_detected(
        prompt_eval_count=prompt_eval_count, projected_prompt_tokens=projected
    )


def test_in_window_normal_ratio_does_not_fire() -> None:
    projected = int(WINDOW * 0.85)
    prompt_eval_count = int(projected * 0.9)
    assert not _truncation_detected(
        prompt_eval_count=prompt_eval_count, projected_prompt_tokens=projected
    )


def test_ratio_at_ceiling_of_the_deep_trigger_does_not_fire() -> None:
    # boundary: the SMALLEST integer prompt_eval_count that does not fall
    # strictly below the 0.35 floor (ceil, not int-truncation, so this is
    # genuinely at-or-just-above the boundary rather than accidentally
    # under it) — paired with the deep-overflow test above, which fires
    # comfortably below it. Kept on a near-window-sized dispatch so the
    # gate doesn't mask the ratio boundary itself.
    projected = int(WINDOW * 0.85)
    prompt_eval_count = math.ceil(projected * 0.35)
    assert not _truncation_detected(
        prompt_eval_count=prompt_eval_count, projected_prompt_tokens=projected
    )


def test_just_below_the_deep_trigger_ceiling_fires() -> None:
    # the pin's other half: one token below the boundary above DOES fire.
    projected = int(WINDOW * 0.85)
    prompt_eval_count = math.ceil(projected * 0.35) - 1
    assert _truncation_detected(
        prompt_eval_count=prompt_eval_count, projected_prompt_tokens=projected
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
    # one past the band, and the ratio (far above 0.35 for this projected
    # size) doesn't trip trigger 1 either.
    projected = int(WINDOW * 0.9)
    prompt_eval_count = WINDOW // 2 + 65
    assert prompt_eval_count >= projected * 0.35  # trigger 1 does not fire
    assert not _truncation_detected(
        prompt_eval_count=prompt_eval_count, projected_prompt_tokens=projected
    )


def test_zero_projected_never_flags() -> None:
    # no dispatch_input to project from -> nothing to compare against;
    # must never divide by zero or produce a spurious flag.
    assert not _truncation_detected(prompt_eval_count=0, projected_prompt_tokens=0)


# --- review round 3 blocker C: trigger 1 gated on near-window sizing,
# same as trigger 2 — three real false-positive shapes the ungated
# version wrongly flagged (small, legitimate deferred-decide/verdict
# calls whose ratio happens to dip low purely from small-N overhead, not
# truncation). All three have projected << WINDOW * 0.8, so the gate
# alone protects them regardless of ratio. ---


def test_deferred_decide_child_shape_does_not_fire() -> None:
    # ~40 projected tokens, a ratio that would have tripped the old
    # ungated 0.42 (and would still trip the new 0.35) threshold on its
    # own — the near-window gate is the only thing that saves it.
    assert not _truncation_detected(prompt_eval_count=12, projected_prompt_tokens=40)


def test_verdict_child_shape_does_not_fire() -> None:
    assert not _truncation_detected(prompt_eval_count=80, projected_prompt_tokens=260)


def test_tiny_deferred_shape_does_not_fire() -> None:
    assert not _truncation_detected(prompt_eval_count=4, projected_prompt_tokens=15)


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


def test_truncation_check_ignores_small_sibling_when_big_call_is_in_window() -> None:
    # review round 4: _truncation_check compared EVERY recorded count
    # against the seat's projection — so a legitimately near-ceiling
    # session (exactly the state this feature exists to produce: the
    # read accumulator admits up to _READ_TOKEN_BUDGET, close to the
    # trigger near-window gate) plus ANY small sibling call (a decide
    # child ~40 tok, a verdict child ~260 tok) still false-fired: the
    # sibling's own tiny prompt_eval_count, compared against the BIG
    # dispatch_input's projection, looks like a huge under-run even
    # though that sibling was never asked to process the big prompt at
    # all. Truncation always lands on the call that received the big
    # prompt, so only the MAX of the recorded counts is the meaningful
    # signal — the small sibling here must not cause a false positive.
    dispatch_input = "word " * 21600
    classify_response = json.dumps(
        {
            "target": "explainer",
            "chain": "explain",
            "step_index": 0,
            "dispatch_input": dispatch_input,
        }
    )
    projected = projected_tokens_v2(dispatch_input)
    big_count = round(projected / 1.59)  # a genuinely non-truncated ratio
    nodes: list[dict[str, Any]] = [
        {"node": "classify", "response": classify_response},
        {
            "node": "seat",
            "seat": [
                {"node": "explainer", "prompt_eval_count": big_count},
                {"node": "verdict", "prompt_eval_count": 260},
            ],
        },
    ]

    assert _truncation_check(nodes, classify_response) is None


def test_truncation_check_flags_via_max_despite_a_small_sibling() -> None:
    # the paired shape: the big call carries the measured discard
    # signature (over-window truncation) while a small sibling call is
    # also present — max(counts) must still catch it.
    dispatch_input = "word " * 21600
    classify_response = json.dumps(
        {
            "target": "explainer",
            "chain": "explain",
            "step_index": 0,
            "dispatch_input": dispatch_input,
        }
    )
    nodes: list[dict[str, Any]] = [
        {"node": "classify", "response": classify_response},
        {
            "node": "seat",
            "seat": [
                {"node": "explainer", "prompt_eval_count": 20482},
                {"node": "verdict", "prompt_eval_count": 260},
            ],
        },
    ]

    detail = _truncation_check(nodes, classify_response)

    assert detail is not None
    assert detail["prompt_eval_counts"] == [20482, 260]


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

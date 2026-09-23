"""Table-driven executor pins for the fail-closed-composition addendum
2026-09-23 (explicit node outcome), Doctrine 11: outcome through the REAL
executor, not hand-built dicts.

Ensembles and scripts are copied verbatim from the review2/review3
capture sessions into tests/fixtures/outcome_probes/.llm-orc (the same
project a reviewer's ./run.sh <ensemble> drove via the CLI against this
worktree). stderrparent.yaml is the one addition: none of the copied
probes wrap a multi-terminal child where one terminal fails and another
survives, which is the shape the has_errors subtree-aggregation pin
needs (partfanparent's partial-fan-out case also exercises it, so this
is a second, more direct instance of the same mutant).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.config.ensemble_config import EnsembleLoader
from llm_orc.core.execution.executor_factory import ExecutorFactory
from llm_orc.core.execution.results_processor import caller_status
from llm_orc.web.serving.turn_trace import build_turn_trace

FIXTURE_ROOT = Path(__file__).resolve().parent.parent / "fixtures" / "outcome_probes"
ENSEMBLES_DIR = FIXTURE_ROOT / ".llm-orc" / "ensembles"


async def invoke(name: str, input_data: str = "hi") -> dict[str, Any]:
    """Run a probe ensemble through the real EnsembleExecutor against the
    fixture project — the same code path `llm-orc invoke` drives."""
    config_manager = ConfigurationManager(project_dir=FIXTURE_ROOT, provision=False)
    executor = ExecutorFactory.create_root_executor(
        project_dir=FIXTURE_ROOT,
        config_manager=config_manager,
        save_artifacts=False,
    )
    config = EnsembleLoader().load_from_file(str(ENSEMBLES_DIR / f"{name}.yaml"))
    result: dict[str, Any] = await executor.execute(config, input_data)
    return result


def node(result: dict[str, Any], name: str) -> dict[str, Any]:
    value = result["results"][name]
    assert isinstance(value, dict), f"{name}: {value!r}"
    return value


def outcome_of(result: dict[str, Any], name: str) -> str:
    return str(node(result, name)["outcome"])


def has_errors_of(result: dict[str, Any], name: str) -> bool:
    return bool(node(result, name)["has_errors"])


# ---------------------------------------------------------------------------
# BLOCKER: a failed script's payload never overwrites the record.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_paystat_failed_scripts_colliding_payload_stays_failed() -> None:
    """payloadstatus.py prints the addendum's exact fixture:
    {"success": false, "error": "...", "status": "success", "response":
    "FABRICATED dossier"}. The record must stay failed."""
    result = await invoke("paystat")

    a = node(result, "a")
    assert a["status"] == "failed"
    assert a["outcome"] == "failed"
    assert a["response"] is None
    assert a["payload"] == {"status": "success", "response": "FABRICATED dossier"}
    assert "status" not in a or a["status"] == "failed"  # never shadowed

    t = node(result, "t")
    assert t["status"] == "skipped"
    assert t["outcome"] == "skipped_by_failure"
    assert t["has_errors"] is True
    assert result["has_errors"] is True


@pytest.mark.asyncio
async def test_paystatparent_child_terminal_failure_fails_the_wrapper() -> None:
    result = await invoke("paystatparent")

    assert outcome_of(result, "k") == "failed"
    assert has_errors_of(result, "k") is True
    assert outcome_of(result, "after") == "skipped_by_failure"
    assert result["has_errors"] is True


# ---------------------------------------------------------------------------
# S1: a run-marked node's when:-false skip over a real failure is a real
# failure, not a neutral skip.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runwhen_run_marked_when_false_over_real_failure() -> None:
    result = await invoke("runwhen")

    assert outcome_of(result, "a") == "failed"
    t = node(result, "t")
    assert t["status"] == "skipped"
    assert t["outcome"] == "skipped_by_failure"
    assert "reason" in t
    assert t["has_errors"] is True


@pytest.mark.asyncio
async def test_runwhenparent_fails_over_the_real_upstream_crash() -> None:
    """Before the S1 fix: t's skip carried no reason (looked like a
    neutral when-skip), so k reported SUCCESS over a in fact totally
    crashed child."""
    result = await invoke("runwhenparent")

    assert outcome_of(result, "k") == "failed"
    assert has_errors_of(result, "k") is True
    assert outcome_of(result, "after") == "skipped_by_failure"


# ---------------------------------------------------------------------------
# runwhenskipped: a run-marked node whose sole dependency is itself a
# plain neutral guard-skip is a normal run, not a handled_failure.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runwhenskipped_neutral_upstream_is_a_normal_run() -> None:
    result = await invoke("runwhenskipped")

    assert outcome_of(result, "a") == "succeeded"
    assert outcome_of(result, "b") == "skipped_by_guard"
    t = node(result, "t")
    assert t["status"] == "success"
    assert t["outcome"] == "succeeded"
    assert "handled_failure" not in t or t["handled_failure"] is False
    assert result["has_errors"] is False


@pytest.mark.asyncio
async def test_runwhenskippedparent_reports_success_nothing_failed() -> None:
    """Before the fix: k failed naming "handled failure: b (skipped)"
    though nothing failed anywhere in the chain."""
    result = await invoke("runwhenskippedparent")

    assert outcome_of(result, "k") == "succeeded"
    assert has_errors_of(result, "k") is False
    assert result["has_errors"] is False


# ---------------------------------------------------------------------------
# whenchain: a chain of purely neutral when:-false skips is neutral, not
# a failure, all the way up.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_whenchain_neutral_chain_stays_neutral() -> None:
    result = await invoke("whenchain")

    assert outcome_of(result, "a") == "succeeded"
    assert outcome_of(result, "b") == "skipped_by_guard"
    t = node(result, "t")
    assert t["status"] == "skipped"
    assert t["outcome"] == "skipped_by_guard"
    assert "reason" not in t
    assert result["has_errors"] is False


@pytest.mark.asyncio
async def test_whenchainparent_reports_success_over_a_neutral_chain() -> None:
    """Before the fix: t's cascade skip always carried a reason whenever
    "no dependency ok" held (regardless of whether anything was
    blocking), so k failed naming "t (skipped)" though nothing failed."""
    result = await invoke("whenchainparent")

    assert outcome_of(result, "k") == "succeeded"
    assert has_errors_of(result, "k") is False


# ---------------------------------------------------------------------------
# partfan: a partial gathered fan-out counts as ok for the wrapper's own
# terminal-success rule, but the wrapper still carries has_errors.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_partfan_partial_gather_and_call_count() -> None:
    result = await invoke("partfan")

    d = node(result, "d")
    assert d["outcome"] == "succeeded"
    s = node(result, "s")
    assert s["status"] == "partial"
    assert s["outcome"] == "partial"
    # addendum: a partial gathered node's own has_errors is true, stamped
    # from its instances — not just inferable from the ensemble-level
    # has_errors below.
    assert s["has_errors"] is True
    assert len(s["instances"]) == 2  # exactly 2 fan-out instances ran
    assert result["has_errors"] is True  # one instance genuinely failed


@pytest.mark.asyncio
async def test_partfanparent_succeeds_with_has_errors_true() -> None:
    """The has_errors subtree-aggregation pin: k's own outcome is
    succeeded (partial counts as ok for the terminal-success rule), but
    has_errors folds in partfan's own has_errors (one fan-out instance
    genuinely failed) — a succeeded wrapper is not the same claim as a
    clean one."""
    result = await invoke("partfanparent")

    assert outcome_of(result, "k") == "succeeded"
    assert has_errors_of(result, "k") is True
    assert outcome_of(result, "after") == "succeeded"


# ---------------------------------------------------------------------------
# stderrparent: the direct (non-fan-out) has_errors subtree-aggregation
# case — added fixture, see module docstring.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stderrparent_succeeds_with_has_errors_true() -> None:
    result = await invoke("stderrparent")

    assert outcome_of(result, "k") == "succeeded"
    assert has_errors_of(result, "k") is True


# ---------------------------------------------------------------------------
# runik: an input_key contract failure (the node never ran) is a plain
# failure, never a handled_failure stamp.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runik_contract_failure_is_not_handled_failure() -> None:
    result = await invoke("runik")

    assert outcome_of(result, "a") == "failed"
    k = node(result, "k")
    assert k["status"] == "failed"
    assert k["outcome"] == "failed"
    assert "handled_failure" not in k or k["handled_failure"] is False
    assert "input_key" in (k.get("error") or "")


@pytest.mark.asyncio
async def test_runikparent_fails_over_the_contract_failure() -> None:
    result = await invoke("runikparent")

    assert outcome_of(result, "p") == "failed"


# ---------------------------------------------------------------------------
# runparent / looprunkid / disprunkid: the same on_dependency_failure:
# run body (runkid: a fails, t handles it) through ensemble:, loop:, and
# dispatch: respectively — each primitive's own terminal-success check
# must agree.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runparent_handled_failure_terminal_does_not_count_as_ok() -> None:
    """runkid's sole terminal t ran (echo.py succeeds) but is stamped
    handled_failure (its only dependency, a, genuinely failed) — X1: a
    handled_failure terminal does not count as ok, so the wrapping
    ensemble: node itself is "failed", naming the real upstream crash,
    not "success" because the handler ran without incident."""
    result = await invoke("runparent")

    k = node(result, "k")
    assert k["status"] == "failed"
    assert k["outcome"] == "failed"
    assert "a (failed" in (k.get("error") or "")
    assert outcome_of(result, "after") == "skipped_by_failure"


@pytest.mark.asyncio
async def test_looprunkid_agrees_with_ensemble_wrapping() -> None:
    result = await invoke("looprunkid")

    loop_node = node(result, "L")
    assert loop_node["status"] == "failed"
    assert loop_node["outcome"] == "failed"


@pytest.mark.asyncio
async def test_disprunkid_agrees_with_ensemble_wrapping() -> None:
    result = await invoke("disprunkid")

    seat = node(result, "seat")
    assert seat["status"] == "failed"
    assert seat["outcome"] == "failed"


# ---------------------------------------------------------------------------
# stderr: independent terminals, mixed outcomes, no cross-contamination.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stderr_independent_terminals_mixed_outcomes() -> None:
    result = await invoke("stderr")

    assert outcome_of(result, "a") == "failed"
    assert outcome_of(result, "s") == "failed"  # timeout
    assert outcome_of(result, "n") == "succeeded"
    assert result["has_errors"] is True


# ---------------------------------------------------------------------------
# thinkhop: a think/provider mismatch reachable only through the
# agent-level fallback chain is a load-time error, before any agent runs.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_thinkhop_fails_closed_at_load_time() -> None:
    with pytest.raises(ValueError, match="think"):
        await invoke("thinkhop")


# ---------------------------------------------------------------------------
# nestdead: a dead model profile inside a fan-out AND a plain child both
# fail their wrapper; has_errors/outcome agree between the two shapes.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_nestdead_fan_out_and_plain_child_agree() -> None:
    result = await invoke("nestdead")

    s = node(result, "s")
    assert s["status"] == "failed"  # every fan-out instance failed
    assert s["outcome"] == "failed"
    plain = node(result, "plain")
    assert plain["outcome"] == "failed"
    assert outcome_of(result, "comp") == "skipped_by_failure"
    assert outcome_of(result, "comp2") == "skipped_by_failure"


# ---------------------------------------------------------------------------
# whenskip / whenparent: a plain when:-false skip over a real success is
# neutral, and its parent reads as a direct invocation would.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_whenskip_is_neutral() -> None:
    result = await invoke("whenskip")

    assert outcome_of(result, "a") == "succeeded"
    t = node(result, "t")
    assert t["outcome"] == "skipped_by_guard"
    assert result["has_errors"] is False


@pytest.mark.asyncio
async def test_whenparent_agrees_with_direct_invocation() -> None:
    result = await invoke("whenparent")

    assert outcome_of(result, "k") == "succeeded"
    assert has_errors_of(result, "k") is False


# ---------------------------------------------------------------------------
# hchain(parent): M1 — BLOCKING_OUTCOMES must include handled_failure, or a
# run-marked node chained after another run-marked node reads its blocking
# upstream as ok and reports a plain success instead of handled_failure.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hchain_each_link_in_a_handled_failure_chain_stays_marked() -> None:
    """a fails; h1 (on_dependency_failure: run) handles it and is itself
    handled_failure; h2 (also run-marked) depends on h1 alone. h1 being
    handled_failure must still read as blocking to h2's own cascade check
    — h2 is handling the SAME upstream failure, not reporting a fresh
    success."""
    result = await invoke("hchain")

    assert outcome_of(result, "a") == "failed"
    assert outcome_of(result, "h1") == "handled_failure"
    assert outcome_of(result, "h2") == "handled_failure"
    assert result["has_errors"] is True


@pytest.mark.asyncio
async def test_hchainparent_fails_over_the_handled_failure_chain() -> None:
    """h2 is hchain's sole terminal and is handled_failure (X1: does not
    count as a succeeded terminal), so the wrapping ensemble: node fails,
    naming the real upstream crash."""
    result = await invoke("hchainparent")

    assert outcome_of(result, "k") == "failed"
    assert has_errors_of(result, "k") is True


# ---------------------------------------------------------------------------
# lhe: M3 — loop_runner.py must carry the final iteration's own has_errors
# onto the loop's JSON response, even when the body's OTHER terminal
# survives and the loop itself does not raise.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lhe_final_iteration_has_errors_survives_sibling_terminal() -> None:
    """lbody2's x fails and y succeeds; both are terminals, so y's success
    alone keeps the loop from raising — but x's failure must still surface
    as the loop's own has_errors, and from there into the caller-facing
    status/exit code every surface derives from it."""
    result = await invoke("lhe")

    loop_node = node(result, "L")
    assert loop_node["outcome"] == "succeeded"
    assert loop_node["has_errors"] is True
    assert caller_status(result["status"]) == ("error", True)


# ---------------------------------------------------------------------------
# efan: M5 — _propagate_child_execution_errors's fan-out LIST branch must
# walk every instance's own subtree has_errors, not just the gathered
# node's top-level status.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_efan_subtree_errors_propagate_when_every_instance_succeeds() -> None:
    """Both fan-out instances run the "stderr" child ensemble in full and
    each one SUCCEEDS outright (stderr's "n" terminal always survives) —
    no instance is partial or failed, so only the fan-out list branch
    (walking child_has_errors over every item in the gathered response
    list) can surface that each instance's own subtree had a real
    failure (stderr's "a")."""
    result = await invoke("efan")

    s = node(result, "s")
    assert s["outcome"] == "succeeded"
    assert len(s["instances"]) == 2
    assert all(inst["status"] == "success" for inst in s["instances"])
    assert has_errors_of(result, "s") is True


# ---------------------------------------------------------------------------
# M4 — turn_trace's _engine_failure_fields must read a failed node's stderr
# through its nested payload (addendum 2026-09-23), driven on a REAL
# executor-produced record rather than a hand-built flat dict (the flat
# shape is the pre-addendum one and would pass even if the payload-reading
# branch broke — doctrine 11).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_turn_trace_reads_stderr_through_the_nested_payload() -> None:
    result = await invoke("stderr")

    trace = build_turn_trace("stderr", result)

    failed_node = next(n for n in trace["nodes"] if n["node"] == "a")
    assert failed_node["status"] == "failed"
    assert failed_node["stderr"] == "boom\n"

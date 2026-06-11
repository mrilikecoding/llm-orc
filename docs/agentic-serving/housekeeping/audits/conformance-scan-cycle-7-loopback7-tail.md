# Conformance Scan Report

**Scanned against:** `docs/agentic-serving/decisions/adr-040-deterministic-completeness-gate.md`
**Codebase:** `src/llm_orc/agentic/loop_driver.py`, `src/llm_orc/agentic/session_action_record.py`, `tests/unit/agentic/test_loop_driver.py`, `tests/unit/agentic/test_session_action_record.py`, `src/llm_orc/web/api/v1_chat_completions.py`
**Date:** 2026-06-10

---

## Summary

- **ADRs checked:** 1 (ADR-040)
- **Conforming:** Yes — all four fitness criteria are implemented and covered by refutable tests
- **Violations found:** 0 permanent
- **Expected-temporary instrumentation:** 2 items (classified below, not counted as violations)

---

## Conformance Debt Table

| ADR | Finding | Type | Location | Resolution |
|-----|---------|------|----------|------------|
| ADR-040 | `completeness:` INFO log (spike diagnostic) | expected-temporary | `loop_driver.py:597-601` | Remove at spike close |
| ADR-040 | `_resolve_judgment_seat` env-gated Arm-B hook | expected-temporary | `v1_chat_completions.py:450-471, 499` | Remove at spike close |

---

## Detailed Findings

### Finding 1 — `_completeness` three-branch structure (ADR-040 §Decision)

**Verdict: CONFORMING.**

The three branches are implemented exactly at `loop_driver.py:595-614`:

- Branch 1 (`requested` non-empty, `remaining` non-empty): line 603 returns `("REMAINING", _compose_remaining(remaining), None)`. The `_compose_remaining` helper (line 848-856) builds `"These requested files have not been written yet: " + ", ".join(sorted(remaining))` — a deterministic framework-computed `requested − produced` string, which becomes the ADR-038 anchor via `_seat_filler_messages` (line 692).
- Branch 2 (`requested` non-empty, `remaining` empty): line 606 returns `("COMPLETE", None, _compose_done(requested))`. No judgment-seat call is made on either named-file branch.
- Branch 3 (`requested` empty): lines 607-614 dispatch `_dispatch_judgment` (the ADR-037 stochastic judge), parse the verdict, and fall through unchanged. The `requested` value is read from `self._action_record.requested(session_id)` at line 595 — the persisted store, not a re-derivation from `_user_task`.

The helper `_produced_paths` (line 835-845) extracts basenames of write-action records from the Session Action Record, and `_extract_requested_deliverables` (line 823-832) extracts basenames via `_REQUESTED_FILE_RE` (line 812). Both are basename-normalized, so a path-qualified produced path (`path/to/a.py`) matches a bare requested name (`a.py`).

No stochastic judgment-seat call is possible on a named-file trailing turn. ADR-040's claim that "the false-COMPLETE cannot occur for named-file tasks where the extraction succeeds" is structurally upheld.

### Finding 2 — persist-once capture in `decide` (ADR-040 §Decision, persist-once)

**Verdict: CONFORMING.**

At `loop_driver.py:396-398`, immediately after `session_id` is resolved and `_join_client_result` is called (lines 384-387), `decide` calls:

```python
self._action_record.set_requested_if_absent(
    session_id, _extract_requested_deliverables(_user_task(context.messages))
)
```

This runs unconditionally on every turn, before the budget cap check, before the termination judgment, and before the action call. On turn 1 (the guaranteed-full task) it captures the requested set. On subsequent turns it is a no-op because `set_requested_if_absent` only writes if the session key is absent.

`_completeness` at line 595 reads `self._action_record.requested(session_id)` — the persisted store — not `_extract_requested_deliverables(_user_task(...))`. The verdict is therefore independent of the per-turn conversation content from turn 2 onward. This is the persist-once property ADR-040 prescribes.

### Finding 3 — `SessionActionRecord` first-non-empty-wins and `cleanup_session` (ADR-040 §Decision, persist-once)

**Verdict: CONFORMING.**

`set_requested_if_absent` at `session_action_record.py:102-117`:

```python
if requested and session_id not in self._requested:
    self._requested[session_id] = requested
```

The guard has two conditions: `requested` must be truthy (empty frozenset is falsy — the empty no-op) and `session_id` must not already have an entry (repeat no-op). Together these enforce first-non-empty-wins exactly as ADR-040 prescribes.

`cleanup_session` at `session_action_record.py:127-130` calls `self._requested.pop(session_id, None)` alongside `self._records.pop(...)`, clearing the requested set on session close.

### Finding 4 — FC test coverage (ADR-040 §Fitness criteria)

**Verdict: CONFORMING — all four fitness criteria are covered by refutable tests.**

**FC: deterministic completeness verdict, named-file tasks (no judgment-seat call):**
`TestDeterministicCompleteness.test_complete_when_all_requested_produced_no_judge_call` at `test_loop_driver.py:1980-1995` asserts `len(judge.calls) == 0` and `isinstance(outcome, FinishTurn)`.
`TestDeterministicCompleteness.test_remaining_overrides_a_wrong_judge_and_anchors_missing` at `test_loop_driver.py:1997-2014` asserts `len(judge.calls) == 0` even when the scripted judge would return COMPLETE — directly refuting the false-COMPLETE failure mode.

**FC: deterministic remaining anchor, named-file tasks (ADR-038 amendment):**
`test_remaining_overrides_a_wrong_judge_and_anchors_missing` at lines 2011-2014 inspects the trailing content of `seat.calls[0][0][-1]["content"]` and asserts the two unproduced filenames (`test_converters.py`, `README.md`) are present and the produced one (`converters.py`) is absent (modulo substring in `test_converters.py`). This refutes "anchor is the judge's statement" for the named-file case.

**FC: persist-once stability across a compacted later turn:**
`TestDeterministicCompleteness.test_persisted_requested_survives_a_compacted_later_turn` at `test_loop_driver.py:2030-2052` pre-seeds the persisted set via `record.set_requested_if_absent(...)`, then calls `decide` with a compacted task ("Continue with the remaining work.") that names no files. Asserts `len(judge.calls) == 0` and the two unproduced files appear in the trailing content. This is the exact compaction scenario ADR-040 §Empirical grounding describes as the "surviving hypothesis for run-2" and the scenario persist-once defends against.

`TestDeterministicCompleteness.test_decide_persists_the_requested_set_once_on_first_naming` at `test_loop_driver.py:2054-2068` verifies that calling `decide` with the full-task context writes the correct frozenset into `record.requested("test-session")` — confirming the driver-side capture rather than only testing the store in isolation.

**FC: no-files fallback preserved:**
`TestDeterministicCompleteness.test_falls_back_to_judge_when_task_names_no_files` at `test_loop_driver.py:2016-2028` asserts `len(judge.calls) == 1` for a task that names no files, confirming the ADR-037 stochastic-judge path is taken.

**`SessionActionRecord` persist-once store FCs (`TestSessionActionRecordRequested`, `test_session_action_record.py:148-195`):**

- `test_requested_defaults_to_empty` — empty before any call.
- `test_set_requested_persists_and_is_retrievable` — basic write+read.
- `test_empty_set_is_a_no_op` — empty set not persisted.
- `test_first_non_empty_set_wins` — second call (smaller set) does not overwrite.
- `test_requested_sets_are_session_isolated` — two sessions hold independent sets.
- `test_cleanup_session_clears_the_requested_set` (in `TestSessionActionRecordLifecycle`) — `cleanup_session` clears the requested set.

All six store-level FCs are present, refutable, and directly implement the first-non-empty-wins invariant ADR-040 prescribes.

### Finding 5 — Temporary spike instrumentation (NOT conformance violations)

**Classification: expected-temporary, per ADR-040 §Consequences "Neutral" bullet.**

ADR-040 itself documents both items as temporary and scheduled for removal at spike close:

1. **`completeness:` INFO log** (`loop_driver.py:597-601`): `_logger.info("completeness: requested=%s produced=%s", sorted(requested), sorted(produced))`. This is the diagnostic that the 2026-06-10 diagnostic session read to confirm the gate was firing correctly. ADR-040 §Neutral names it: "The diagnostic `completeness:` log … [is] temporary spike instrumentation, removed and reverted at spike close."

2. **`_resolve_judgment_seat` env-gated hook** (`v1_chat_completions.py:450-471`, called at line 499): Spike σ Arm B hook for swapping the judgment seat to a frontier model via `LLMORC_SPIKE_JUDGMENT_PROFILE`. When the env var is unset (the default) this is a no-op; it does not affect production behavior. The function docstring reads "REMOVE after the spike." The call site comment (lines 496-498) also tags it for removal. ADR-040 does not enumerate this hook separately but it is within the scope of the Spike σ instrumentaton referenced in §Neutral.

Neither item contradicts ADR-040's permanent design. They are the scaffolding that generated the evidence the ADR is grounded on, and ADR-040 acknowledges their temporary status.

**Note on `request_timeout.read 600` config edit:** ADR-040 §Neutral also lists this as temporary instrumentation. It is a `.llm-orc/config.yaml` edit (not code), and falls outside the scanned files. Not scanned but noted for completeness.

---

## Notes

**The spike → build → ADR sequencing is correctly reflected.** The ADR's §Empirical grounding explicitly states it "takes unconditional status at the DECIDE gate rather than carrying a Conditional through BUILD, because the BUILD code already exists." The code matches the ADR's description of it throughout — this is not a case where the ADR is describing aspirational behavior not yet landed.

**One subtle precision point worth recording (P3, not a violation):** The `test_remaining_overrides_a_wrong_judge_and_anchors_missing` test at line 2014 checks `"converters.py" not in trailing.replace("test_converters.py", "")`. This correctly handles the substring relationship between `converters.py` and `test_converters.py`. The assertion is valid but fragile at the assertion level (not in the production logic). The production `_compose_remaining` uses `sorted(remaining)` which will never include `converters.py` if it was produced — the issue is only in how the test assertion is written, not in the gate itself.

**No gap was found between the ADR's four fitness criteria and the test coverage.** Each FC maps to at least one refutable test that would fail if the FC were violated. The persist-once FC is covered at two levels: the store unit tests (`TestSessionActionRecordRequested`) and the integration-level driver test (`test_persisted_requested_survives_a_compacted_later_turn`) that exercises the full `decide` path with a compacted task.

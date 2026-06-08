# Conformance Scan Report

**Scanned against:**
- `docs/agentic-serving/decisions/adr-038-remaining-work-anchor.md` (primary)
- `docs/agentic-serving/decisions/adr-037-session-termination-two-call-composition.md` (context — amended ADR)

**Codebase:** `/Users/nathangreen/Development/eddi-lab/llm-orc`
**Date:** 2026-06-08

---

## Summary

- **ADRs checked:** 2 (ADR-037 as context; ADR-038 as the amendment under scan)
- **ADR-037 prior conformance:** Confirmed — the two-call composition, judgment dispatch, digest provenance, COMPLETE finish, and discard of the judgment exchange before call 2 are all implemented and tested.
- **ADR-038 conformance:** 0 of 3 prescribed behaviors implemented.
- **Violations found:** 3

---

## Conformance Debt Table

| ID | ADR | Violation | Type | Location | Severity | Resolution |
|----|-----|-----------|------|----------|----------|------------|
| V-38-1 | ADR-038 §Decision | On REMAINING, `_dispatch_judgment` returns the raw judgment text and the caller discards it — the stripped remaining-work statement is never captured in a variable or passed anywhere after `parse_verdict` returns "REMAINING". | missing | `src/llm_orc/agentic/loop_driver.py:364-365` (the `judgment_verdict = parse_verdict(judgment_text)` block; `judgment_text` is used only for the COMPLETE branch at :377) | critical — blocks multi-file convergence | Capture `strip_verdict(judgment_text)` into a local (`remaining_statement`) on the REMAINING fall-through path and thread it into `_seat_filler_messages`. |
| V-38-2 | ADR-038 §Decision / FC (remaining-work anchor presence) | `_seat_filler_messages` has no parameter or branch for a remaining-work anchor — it always composes the trailing region as `_DELEGATION_GUIDANCE` alone, with no remaining-work statement and no "Produce that next." imperative appended. | missing | `src/llm_orc/agentic/loop_driver.py:505-528` (entire `_seat_filler_messages` method) | critical | Add an optional `remaining_anchor: str \| None = None` parameter. When provided, append `f"\n\n{remaining_anchor} Produce that next."` after `_DELEGATION_GUIDANCE` in the trailing-message branch (line 528 path). |
| V-38-3 | ADR-038 §Decision / FC (call-2 form preservation — amended) | The existing test `test_remaining_verdict_call2_form_preserved` (FC-66 / V-08) asserts that the trailing message equals `_DELEGATION_GUIDANCE` with no additional content — this is the ADR-037 pre-amendment form. After ADR-038 lands, the test's assertion will be a false pass guard: it will need updating to assert the anchor + imperative are present. No test covers the anchored form at all. | missing | `tests/unit/agentic/test_loop_driver.py:757-786` | high — test gap; the amended FC has no red test to drive the build | Add a new test (e.g. `test_remaining_verdict_call2_carries_anchor`) that asserts: (a) the trailing user message contains the judge's stripped remaining-work sentence, (b) it contains the literal " Produce that next.", and (c) the judgment digest/question/verdict literal are absent. Update `test_remaining_verdict_call2_form_preserved` to reflect the amended FC. |

---

## Notes

**Build shape.** All three violations are a single tightly-coupled unit of work — they cannot be built independently without leaving the surface in a broken intermediate state. The natural TDD sequence is:

1. Write the red test for V-38-3 first (the new anchored-form assertion fails immediately because `_seat_filler_messages` ignores the anchor parameter that doesn't exist yet).
2. Add the `remaining_anchor` parameter to `_seat_filler_messages` (V-38-2) — the new test goes green; the existing `test_remaining_verdict_call2_form_preserved` now fails if it asserts the old trailing-message exact form, which is the expected structural break marking the amended FC.
3. Thread the captured `strip_verdict(judgment_text)` into the `_seat_filler_messages` call on the REMAINING fall-through in `decide` (V-38-1) — both tests pass.
4. Update `test_remaining_verdict_call2_form_preserved`'s docstring and assertions to reflect the amended FC-66 (it should now assert the anchor is present, not absent).

**Preservation invariants to verify during build.** ADR-038 is explicit that three things must stay absent from call 2 even on the anchored REMAINING path: the judge system message, the digest, and the verdict literal. The existing `test_remaining_verdict_call2_form_preserved` already guards these (`"VERDICT" not in content`, `"Status check" not in content`); those assertions should carry forward into the updated test. The fixed imperative string `" Produce that next."` must be exactly that literal — not parameterized — per ADR-038's characterization of it as "a fixed framework string."

**ADR-037 prior build findings that are now fully conforming.** The following ADR-037 FCs were flagged as expected BUILD findings in the prior scan (`conformance-scan-cycle-7-loopback5.md`) and are confirmed implemented: judgment dispatch (`_dispatch_judgment`), bare-form composition (`compose_judgment_message`), digest provenance (`_join_client_result` + `SessionActionRecord`), COMPLETE protocol-clean finish (`strip_verdict` + `FinishTurn`), judgment-exchange discard (verified by `test_remaining_verdict_call2_form_preserved`), `TurnDecision` verdict fields, and the AS-3 budget cap. No regressions observed against ADR-037's standing FCs.

**Scope boundary note.** ADR-038's REMAINING anchor applies only when `tail_kind == "trailing_tool_result"` and `self._capabilities` is non-empty — the same guard that gates the judgment call (`:363`). The anchor must not appear on first-turn or new-user-task tails. The existing `test_no_judgment_on_first_turn_or_new_user_task` test covers the judgment non-firing; a corresponding assertion that the anchor is absent on those tails is implicit once V-38-2 is built with a `None`-default parameter, but worth an explicit assertion in the updated test suite.

# Conformance Scan Report

**Scanned against:**
- `docs/agentic-serving/decisions/adr-037-session-termination-two-call-composition.md` (primary)
- `docs/agentic-serving/decisions/adr-036-delegation-decision-mechanism.md` (updated header; partial supersession)
- `docs/agentic-serving/decisions/adr-033-layer-a-loop-driver-multi-turn-agentic-surface.md` (context)
- `docs/agentic-serving/decisions/adr-034-client-tool-action-terminal-artifact-bridge.md` (context)
- `docs/agentic-serving/decisions/adr-035-client-tool-deliverable-form-contract.md` (context)

**Codebase:** `/Users/nathangreen/Development/eddi-lab/llm-orc`

**Date:** 2026-06-05

---

## Summary

- **ADRs checked:** 5
- **Conforming (033, 034, 035):** 3 (no violations found against their accepted terms)
- **Violations found:** 8
  - ADR-036 trailing-turn form: 1 violation (code predates update; this is the primary ADR-037 BUILD target)
  - ADR-037 (all decisions): 7 missing-implementation findings (newly Proposed; the mechanism is entirely unbuilt — expected BUILD work package)

---

## Conformance Debt Table

| ID | ADR | Violation | Type | Location | Severity | Resolution |
|----|-----|-----------|------|----------|----------|------------|
| V-01 | ADR-036 (updated by ADR-037) | `_seat_filler_messages` appends C3 guidance unconditionally on every non-user-message tail, with no preceding termination judgment. The ADR-036 update header explicitly replaces this with ADR-037's two-call composition for trailing turns. | exists (superseded form) | `src/llm_orc/agentic/loop_driver.py:336–347` | BUILD-work (primary target) | Insert the two-call dispatch path before C3 guidance on trailing turns (decision 4, ADR-037): call the termination judge; on COMPLETE return the stripped verdict response; on REMAINING proceed to the current C3 guidance call. |
| V-02 | ADR-037 §Decision 1 | No termination judgment call exists anywhere. On trailing-tail turns the Loop Driver goes directly to the action call. FC (judgment-first trailing composition) is violated on every trailing turn. | missing | `src/llm_orc/agentic/loop_driver.py` — no `judge` callsite | BUILD-work | Add the termination judgment dispatch as the first step in `_seat_filler_messages` (or `decide`) when the tail is a tool-result tail. The judge call: bare-form system message, one user message carrying the quoted task + framework-owned digest + deliverable-accounting question; no tools offered. |
| V-03 | ADR-037 §Decision 2 — FC (digest provenance) | No session-scoped record of emitted tool calls exists. The framework has no data structure that accumulates (action kind, file path, result) from its own dispatch/emission records. The Artifact Bridge and Client-Tool-Action Terminal emit the wire tool call but write nothing to a session action store. Round 1's measured failure (information-starved reconstruction from client messages alone) is the current production behavior. | missing | `src/llm_orc/agentic/client_tool_action_terminal.py` (no action recorder); `src/llm_orc/agentic/loop_driver.py` (no action recorder); no `session_action_store.py` or equivalent | BUILD-work (gating condition for Conditional Acceptance) | Create a session-scoped action record accumulated at every `ApplyWork` and `CarryClientTool` emission. Each record carries action kind, target file path, and the client's per-call tool result (the `role: tool` message content). Thread this store into the judgment call's digest composition. This is the production digest join the acceptance-gate requires. |
| V-04 | ADR-037 §Decision 3 — deliverable-accounting standard | No deliverable-accounting question or standard ("a successful write of a requested file counts as produced") is composed anywhere in the codebase. | missing | No file | BUILD-work | Author the judge question text (including the accounting standard) and attach it to the bare-form user message in the judgment call. This is the prompt component that moved the judgment from 0/10 to 29/30 in Spike θ Round 2. |
| V-05 | ADR-037 §Decision 4 — branch enforcement | No COMPLETE/REMAINING branch exists. There is no code that (a) parses a `VERDICT:` line from a model response, (b) strips it and returns the text as a `FinishTurn`, or (c) conditionally proceeds to C3 guidance on REMAINING. | missing | `src/llm_orc/agentic/loop_driver.py` | BUILD-work | Add `VERDICT:` line detection and branch logic. COMPLETE path: strip the verdict line, return `FinishTurn(content=stripped_text)`. REMAINING path: invoke the existing `_seat_filler_messages` C3 form (unchanged). |
| V-06 | ADR-037 §Decision 6 — FC (termination observability) | `TurnDecision` at `loop_driver.py:171–198` has six fields: `dispatch_id`, `turn_index`, `action`, `delegated_ensemble`, `grounded_carry_held`, `replanned_after_truncation`. It carries no `turn_shape` field (trailing-tool-result vs. first-turn vs. user-task tail) and no `judgment_verdict` field (COMPLETE / REMAINING). False-continue frequency is not computable from events alone; the finish-policy event the loop-back #5 entry package anticipated is missing. | missing (field gap) | `src/llm_orc/agentic/loop_driver.py:171–198` | BUILD-work | Extend `TurnDecision` with two fields: `turn_shape: Literal["first_turn", "trailing_tool_result", "new_user_task"]` and `judgment_verdict: Literal["COMPLETE", "REMAINING"] | None`. Emit both fields on every trailing-turn decision. |
| V-07 | ADR-037 §Layer 1 guarantee — FC (finish protocol cleanliness) | No code path produces a `FinishTurn` from a judgment response. `FinishTurn.content` exists at `loop_driver.py:153–159` but it is only emitted when the seat-filler's `generate_with_tools` produces no tool calls (`enforced.action is None`, line 275). A COMPLETE judgment verdict producing a protocol-clean text-only finish is structurally absent. | missing | `src/llm_orc/agentic/loop_driver.py:267–310` | BUILD-work | The COMPLETE branch (V-05) must return a `FinishTurn` whose `content` is the judgment response with the `VERDICT:` line stripped. The `ClientToolActionTerminal._emit` already handles `FinishTurn` correctly (line 93); no terminal changes needed for this path. |
| V-08 | ADR-037 §Decision 4 — FC (call-2 form preservation) | On a REMAINING verdict the judgment exchange must be excluded from call 2's context. Currently there is no judgment exchange at all, but when it exists the code must not thread the judgment messages into the `context` passed to `_seat_filler_messages` for call 2. The existing `_to_openai_messages` projects from `context.messages` without filter. | missing (guard absent) | `src/llm_orc/agentic/loop_driver.py:503–515` (`_to_openai_messages`) | BUILD-work | The judgment call must use a locally-constructed message list (task + digest + question), not the session context. The call-2 `_seat_filler_messages` invocation must receive the original `context` unmodified. No change to `_to_openai_messages` itself; the isolation is a call-site concern. |

---

## ADR-033 / ADR-034 / ADR-035 status

No violations found.

- **ADR-033** (single-action-per-turn, callee generation, grounded carry, seat-filler swappability): all four FCs are met. `SingleStepEnforcer` enforces the single-action invariant (`single_step_enforcer.py`); `invoke_ensemble` routes to `_delegate_generation` (not through the plan/dispatch/synthesize pipeline); `_passthrough_client_tool` carries literal arguments verbatim; the seat-filler is profile-injected and swappable.
- **ADR-034** (tool-call terminal, no server-side write, artifact-bridge fidelity, loop participation): all four FCs are met. `ClientToolActionTerminal` emits `ClientToolCall` on `ApplyWork`; `ArtifactBridge.marshal` reads the store or `envelope.primary` unchanged; the terminal handles `role: tool` follow-ups by receiving the full `SessionContext` from the serving layer.
- **ADR-035** (boundary-composed form directive, destination-keyed, one dispatch per deliverable): met. `compose_form_directive` in `loop_driver.py:426–445` injects the destination-keyed bare-output directive into every `invoke_ensemble` dispatch input; capability ensembles carry no destination knowledge.
- **ADR-036** (user-turn guidance composition, delegation-rate instrumentation, profile-swap re-validation, carry-side preservation): the four FCs that survive the ADR-037 update are met. `_DELEGATION_GUIDANCE` is in the user-turn region (line 344 for first-turn, line 347 for C3 trailing); `TurnDecision.delegated_ensemble` exists for the numerator; carry-side passthrough is verbatim. The profile-swap FC has no code enforcement (it is a process constraint); no violation. **V-01 above covers the one FC that is superseded by ADR-037's update.**

---

## AS-3 turn-cap / BudgetController status

`BudgetController` (`budget_controller.py`) is wired into `OrchestratorRuntime` but is **not wired into the layer-A loop-driver path** (`get_loop_driver` in `v1_chat_completions.py:416–434` constructs no `BudgetController`; `ClientToolActionTerminal.run` has no cap check). ADR-037 names AS-3 as "the absolute ceiling" and "the deterministic backstop beneath this mechanism." The cap is a named safety net for the new mechanism, and its absence from the loop-driver path is a pre-existing gap (not introduced by ADR-037). It is not a new ADR-037 violation, but it does mean the backstop ADR-037 relies on is not yet active on this surface. Noted here for the BUILD work package's awareness; the loop-driver has no budget enforcement today.

---

## Classification summary

All eight violations are **expected ADR-037 BUILD work** — the mechanism was decided in this loop-back session (status: Proposed / Conditionally Accepted); code predating the decision is not pejorative drift. V-01 is the one case where previously-accepted code (ADR-036's C3 unconditional trailing form) is now superseded by ADR-037; the rest are missing implementations of a mechanism that did not exist at BUILD time.

No findings against ADR-033, ADR-034, or ADR-035 were surfaced. No true drift against previously-accepted ADRs (other than V-01) was found.

---

## Notes

**The primary composition point is `loop_driver.py:_seat_filler_messages` (lines 324–347) plus `decide` (lines 251–310).** V-01 through V-08 converge on the following BUILD shape:

1. Detect trailing-tail shape in `decide` (last message role is `tool` or last message is an assistant turn with tool calls — the framework can check `context.messages[-1].role`).
2. Dispatch the bare judgment call (no client system prompt, no tools, locally-constructed messages: judge system message + user message with quoted task + digest + accounting question).
3. Parse `VERDICT:` from the response content.
4. COMPLETE → `FinishTurn(content=verdict_stripped_text)`, emit `TurnDecision` with `turn_shape="trailing_tool_result"`, `judgment_verdict="COMPLETE"`.
5. REMAINING → fall through to the existing `_seat_filler_messages` C3 path unchanged, emit `TurnDecision` with `judgment_verdict="REMAINING"`.

The judgment call requires a `SeatFiller` (or a distinct `JudgeSeatFiller` if the judgment seat is a separate profile). The session action store (V-03) is the pre-condition for the digest; that is the gating condition for Conditional Acceptance and should be built first.

**`TurnDecision` field extension (V-06) is low-cost** and can land with or before the judgment call — the fields are additive and do not change existing consumers.

**The `BudgetController` gap** (AS-3 not wired on the loop-driver surface) is independent of ADR-037 and should be tracked as a separate BUILD item. Without it the zombie-revision loop ADR-037 is designed to fix has no hard ceiling on the current surface; the cap and the judgment mechanism are complementary, not substitutes.

# Spike ι — No-Tools Graceful Finish on the Unified Loop (Cycle 7 loop-back #9)

**Status:** COMPLETE (run 2026-06-18, all $0 local). Light methods-review
(structural-risk class — the χ/φ bounded BUILD-gate class, not the θ/ω/ρ
DECIDE-driving behavioral class). **Verdict: graceful-finish PASS (H-ι.1 +
H-ι.2 + H-ι.3); the spike surfaced Finding F-ι.1 — delegation on a no-tools
request emits a client tool_call the no-tools client cannot execute, the one
design decision the collapse must resolve.** The loop answers no-tools requests
gracefully; the collapse is sound but is not a pure mechanical retire+flip.

**Trigger:** Loop-back #9 (cycle-status FRESH-SESSION HANDOFF 2026-06-18, open
item #5 — "collapse to one serving surface"). The plan flips the
`_is_tool_driven` discriminator (`v1_chat_completions.py:622`) so *all* requests
route to the Client-Tool-Action Terminal + Loop Driver, retiring ADR-027's
single-turn Dispatch Pipeline. The handoff flags the OPEN RISK: "verify the loop
answers a no-tools request gracefully (FinishTurn text path)." This spike grounds
the supersession ADR (DECIDE loop-back #9) before it commits.

---

## What code-reading already settled (the floor under this spike)

A read of the routing seam, the Terminal, and `LoopDriver.decide` establishes
that the no-tools path is **structurally reachable**:

- The discriminator docstring (`v1_chat_completions.py:568`) and `FinishTurn`'s
  docstring (`loop_driver.py:288`) both already declare the edge "safe because
  the loop-driver can finish with text."
- The Terminal (`client_tool_action_terminal.py`, 152 lines) carries **no**
  `len(tools)` precondition — the only `tools` reference is the module docstring.
- `decide()` builds `seat_tools = self._delegation_tools() + list(context.tools)`.
  With no client tools: capabilities present → `seat_tools` = `[invoke_ensemble]`;
  capabilities absent → `seat_tools` = `[]`. A seat-filler that proposes no action
  returns `FinishTurn(content=response.content or None)` (`loop_driver.py:674`).

So the *structural* claim is strong on inspection. What inspection **cannot**
settle is behavior: with capabilities registered, the delegation-carrying
guidance (`_seat_filler_messages`, composed only when delegation is possible)
could bias a real seat-filler toward invoking an ensemble for a plain question
that should simply be answered. That is the genuinely open question.

## What collapse actually retires (scope note for the ADR)

The loop already covers a no-tools request two ways: a plain-text answer
(seat-filler proposes no action) and **single-ensemble delegation** (seat-filler
calls `invoke_ensemble` for a capability-matched question). The *only* capability
ADR-027 holds that the loop does not is the planner-driven **multi-capability
plan → dispatch → synthesize fan-out** — which the handoff names as "the original
answer-a-question vision the loop-back pivoted from." Collapse drops that
fan-out, not no-tools service.

---

## Hypotheses

- **H-ι.1 (structural, Arm A):** A no-tools `SessionContext` driven through the
  real `LoopDriver.decide` path returns `FinishTurn` with the seat-filler's text
  (or `None` when empty) — no crash, no tools-required precondition — in **both**
  capability states, with no client tool offered when capabilities are absent.
- **H-ι.2 (behavioral, Arm B):** A real local seat-filler, handed a plain
  no-tools question, **finishes with a sensible text answer** rather than
  over-delegating, in both capability states. With capabilities present, a
  *capability-matched* no-tools question may legitimately delegate — that is
  service, not a defect.
- **H-ι.3 (null-guard, Arm B):** The delegation-carrying guidance does not drive
  a degenerate always-delegate on plain questions when capabilities are present.

## Arms

- **Arm A — structural floor.** `scratch/spike-iota-one-surface/probe_structural.py`:
  drives `LoopDriver.decide` (real class, `_FakeSeatFiller`-style double) over a
  `tools=[]` context across {no capabilities, capabilities present} ×
  {seat proposes text, seat proposes empty, seat delegates}. $0, no model,
  deterministic. Asserts H-ι.1 and the offered-tool-list shape.
- **Arm B — live no-tools confirm (PRIMARY).** A real qwen3 local seat
  (Ollama `/v1`, $0) handed plain no-tools questions through the real
  `decide()` composition (not hand-built messages), N=10 per cell, cells =
  {no capabilities, capabilities present} × {plain question, capability-matched
  question}. Reports finish-with-text vs delegate counts. A real-server smoke
  (local serve + a throwaway uncommitted discriminator flip + a no-tools curl)
  confirms the path end-to-end through the HTTP surface, honoring the
  validate-against-real-client discipline.

## Pre-registered predictions + thresholds

- **Arm A:** all cells return `FinishTurn` (or `ApplyWork` for the delegate
  cell); the no-capabilities cell offers the seat-filler an empty tool list and
  still finishes. PASS = every assertion holds. A failure here (precondition,
  crash) escalates the ADR from "clean supersession" to "carries design work."
- **Arm B:** plain-question finish-with-text ≥ 8/10 per cell (allowing the
  occasional over-delegation as noise, not pattern); capability-matched delegate
  ≥ 8/10 in the capabilities-present cell (delegation still works). Degenerate
  always-delegate on plain questions (≤ 2/10 finish) **fails H-ι.3** and routes
  to a guidance-conditioning sub-question for the ADR.

## Methods-review notes (light, structural class)

- Arm A is a deterministic existence/precondition check, not a measurement — no
  framing-bias surface.
- Arm B's one real risk is the plain-vs-capability-matched question wording
  loading the result: the plain questions must be genuinely outside the
  registered capability domain (else "delegation" is correct and the finish-rate
  reads low for the right reason). Cells fix the capability set explicitly and
  the plain questions are domain-disjoint (arithmetic / general knowledge vs a
  `code-generator` capability).

---

## Results

### Arm A — structural floor: PASS (7/7, 2026-06-18, $0 deterministic)

`scratch/spike-iota-one-surface/probe_structural.py`, run via `uv run`.

| Cell | Setup | Outcome | Result |
|------|-------|---------|--------|
| 1 | no capabilities, plain question, seat finishes with text | `FinishTurn(content="2 + 2 = 4.")`; seat offered **empty** tool list | PASS |
| 2 | no capabilities, empty seat text | `FinishTurn(content=None)` | PASS |
| 3 | capabilities present, plain question, seat finishes with text | `FinishTurn(content="The answer is 4.")`; seat offered **only** `invoke_ensemble` | PASS |
| 4 | capabilities present, capability-matched question, seat delegates | `ApplyWork` (write/sort.py/code-generator); exactly one ensemble dispatched | PASS |

**H-ι.1 confirmed.** The no-tools path through the real `LoopDriver.decide` has
no tools-required precondition and no crash surface. A plain no-tools question
finishes with text regardless of capability state; with no capabilities the
seat-filler is handed a literally empty tool list and still finishes; a
capability-matched no-tools question delegates normally. The structural claim
from code-reading is now executed, not just inspected — the supersession does
not carry a structural defect.

### Arm B — live no-tools confirm (PRIMARY): PASS on graceful-finish; surfaced F-ι.1

qwen3:14b seat via Ollama `/v1` ($0), N=10 per cell, composition through the
landed `_seat_filler_messages` / `_delegation_tools` path.
`scratch/spike-iota-one-surface/probe_live.py` + `results_*.json`.

| Cell | Setup | finish | delegate | other |
|------|-------|--------|----------|-------|
| `plain_nocaps` | plain question, no capabilities | **10/10** | 0 | 0 |
| `plain_caps` | plain question, capabilities present | **10/10** | 0 | 0 |
| `match_caps` | coding question, capabilities present | 7/10 | 3/10 | 0 |

Sampled `plain_caps` answers are real text ("The capital of France is Paris.";
"17 times 4 equals 68."), not delegation artifacts.

**H-ι.2 (graceful finish): PASS.** 27/30 finish with sensible text; 0 errors,
0 crashes, 0 `other_tool`. A no-tools request is answered.

**H-ι.3 (no over-delegation on plain questions): CLEAN PASS.** The
delegation-carrying guidance, composed whenever capabilities are present, drove
**zero** over-delegation on plain questions (`plain_caps` 10/10 finish). The
guidance does not hijack a plain question into a delegation.

**Pre-registered prediction miss → F-ι.1 (the spike's real surface).**
`match_caps` delegate was pre-registered ≥8/10; observed 3/10. The prediction
was wrong-headed, and reading the Terminal explains why: `_emit_apply_work`
(`client_tool_action_terminal.py:99-143`) marshals an `ApplyWork` into a client
`write` tool_call **unconditionally** — it does not check that the client
offered a `write` tool. For a no-tools request, those 3/10 delegations emit a
`write` tool_call the client **cannot execute**. So the seat's inline-answer
preference (7/10 finish) is the *correct* behavior here, and delegation is the
broken path. Today this never occurs (no-tools requests route to ADR-027's
pipeline, never the Terminal); **the collapse is what introduces no-tools →
Terminal, so F-ι.1 must be resolved by the supersession**, not deferred.

## Finding F-ι.1 — no-tools delegation emits an un-executable tool_call

When loop-back #9 routes a no-tools request to the unified loop, a seat-filler
delegation (`invoke_ensemble` → `ApplyWork` → Terminal `write` tool_call)
produces a client tool_call for a tool the no-tools client never offered. The
loop finishes plain questions with text correctly; the gap is only on the
delegation branch. The supersession ADR must pick a resolution:

- **Resolution A (deterministic gate — aligns with the determinism-over-carve-outs
  principle):** offer `invoke_ensemble` to the seat only when the client offers a
  write-capable tool (an emission target for the delegated deliverable). A
  no-tools request has no emission target, so it is a pure text-completion turn —
  the seat answers directly. Replaces the stochastic seat-judgment (3/10 broken)
  with a framework rule. Cost: a no-tools capability-matched request gets the
  seat's inline answer rather than an ensemble-backed one — consistent with the
  loop-back's pivot away from the multi-capability fan-out.
- **Resolution B (marshal-to-text):** on `ApplyWork` when the client has no
  matching tool, the Terminal returns the resolved deliverable as a text
  completion (a finish), not a tool_call. Preserves ensemble-backed answers for
  no-tools capability requests, returned as text. More work (Terminal branch),
  preserves the capability the pipeline offered.

**Decision (2026-06-18): Resolution B.** The agent's initial recommendation was A
(determinism, simplicity). The practitioner reframed: turns are emergent (a client
does not pre-declare them), and OpenCode — the north-star client — always sends
tools, so the A/B choice never touches the practitioner's own path; it governs only
a *toolless* client. Under that frame A is a functional narrowing (a toolless
capability request would get inline seat text — the ADR-027 failure mode), and the
determinism advantage A claimed dissolves under B because B makes the delegate
branch valid (text-marshalled), so the seat's stochastic delegate-vs-finish is
benign, not correctness-bearing. The argument audit's framing section flagged the
same narrowing independently. **B chosen:** delegation is uniform; the Terminal
adapts the deliverable's output shape (tool_call when the client offers the
destination tool, text when it does not) — a Terminal-only change, Loop Driver
unchanged. Encoded in ADR-043.

## Scope confirmation for the supersession (unchanged by F-ι.1)

The collapse is sound: the loop is the one surface. ADR-027 (Dispatch Pipeline)
retires; only ADR-033 §Decision 1 (the two-surface discriminator) is superseded;
ADR-033 Decisions 2–6 + FCs stand and become universal. F-ι.1 adds one bounded
design decision to the supersession; it does not reopen the collapse.

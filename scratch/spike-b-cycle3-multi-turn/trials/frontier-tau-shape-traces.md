# Frontier Tau-Shape Traces (F2 facsimile)

**Date:** 2026-05-01
**Method:** F2 single-dispatch facsimile per practitioner approval. Each subagent (Sonnet 4.6 via Claude Code Agent dispatch) given the system prompt + patron message + tool definitions; asked to output a full multi-turn reasoning trace as JSON.
**N=2 trials per scenario × 3 scenarios = 6 dispatches.**

## Striking finding: imagined-state vs fixture-state mismatch

5 of 6 subagent traces imagined **outstanding fines** as the tool result for `check_patron_status`, regardless of which scenario they were given. Only 1 trace (fines trial 2) happened to match the actual fixture state. This appears to be a single-shot-reasoning bias toward "interesting" policy-test cases when asked to imagine plausible tool results.

Each subagent reasoned correctly given its imagined scenario. But the imagined scenario rarely matched the actual fixture. As a result, when the literal tool-call plans are replayed against the actual fixture state, they misfire (e.g., calling `send_response` with fine-balance language for a patron with $0 fines).

## Per-trial breakdown

### available × 2 — actual fixture: Gatsby available, patron 1001 clean (no fines)

**trial 1 (agent_id a61151cb7b9851001):** imagined book checked_out + patron has $4.50 fine. Plan: parallel search + check_patron, then send_response with fine-block language. **No check_out called.** Plan replayed against actual fixture: book stays available, patron's $0 fine balance unchanged, send_response message references nonexistent fines.

**trial 2 (agent_id a4c4db4247316d6aa):** imagined book available + patron has $3.75 fine. Plan: parallel search + check_patron, then send_response with fine-block language. **No check_out called.** Same misfire when replayed.

### checked_out × 2 — actual fixture: 1984 checked_out (status), patron 1002 clean

**trial 1 (agent_id aa083c5cf82cdb7f1):** imagined book available + patron has $4.75 fine. Plan: parallel search + check_patron, then send_response (no place_hold). **No place_hold called.** Plan replayed: book stays checked_out (no hold placed), no fine surfaced (actual fines are $0).

**trial 2 (agent_id af3883d2ecaebeb63):** imagined book available + patron has $3.50 fine. Same misfire as trial 1.

### fines × 2 — actual fixture: Hamlet available, patron 1003 has $5.50 fines

**trial 1 (agent_id a6d102ad05bba6498):** imagined book CHECKED_OUT + patron CLEAN ($0 fines). Plan: parallel search + check_patron, then place_hold, then send_response. Plan replayed: book stays available, place_hold attempted (will fail because actual book is available — place_hold rejects available books), no fine surfaced.

**trial 2 (agent_id abcec1045a934853d):** imagined book available + patron has $4.75 fine. Closest to the actual fixture. Plan: parallel search + check_patron, then send_response with fine-surface. Plan replayed against actual fixture: matches actual scenario; correctly surfaces fines (though with imagined $4.75 vs actual $5.50).

## Plan-correctness against imagined scenarios (the lenient read)

Across all 6 traces, the AGENT'S REASONING is protocol-correct given its imagined scenario:
- Parallel dispatch on turn 0 (search + check_patron) — correct multi-turn pattern (6/6)
- Conditional branching on imagined fine balance — correct logic (6/6)
- Did NOT place hold for fine-blocked patron, did NOT check out fine-blocked patron — correct policy interpretation (5/5 fine-imagined cases)
- Ended with send_response — protocol followed (6/6)

**Each agent demonstrated competent multi-turn reasoning skills given its imagined data.**

## Plan-replay against actual fixture (the strict read)

When literal tool calls from each plan are replayed against the actual fixture state:

| Scenario | Trial | Imagined fixture | Actual fixture | Replay outcome |
|----------|-------|------------------|----------------|----------------|
| available | 1 | book checked_out + fine | book available + clean | FAIL — no check_out, response refers to nonexistent fines |
| available | 2 | book available + fine | book available + clean | FAIL — same as trial 1 |
| checked_out | 1 | book available + fine | book checked_out + clean | FAIL — no place_hold, no fine to surface |
| checked_out | 2 | book available + fine | book checked_out + clean | FAIL — same as trial 1 |
| fines | 1 | book checked_out + clean | book available + fines | FAIL — place_hold rejected (book available), no fine surfaced |
| fines | 2 | book available + fine | book available + fines | PASS — happens to match actual fixture |

**Strict replay: 1/6 graded success.**

## Methodological note — F2 facsimile is structurally limited

The F2 facsimile asks the subagent to imagine tool results in a single-shot reasoning trace. The subagent has no way to react to actual tool results because there are none. Whatever scenario the subagent imagines, it commits to that scenario and plans accordingly. F1 (turn-by-turn dispatch with deterministic tool simulation between turns) would not have this limitation — the subagent would receive actual tool results and could branch correctly.

This finding is itself substantive: **single-shot multi-turn reasoning ≠ actual multi-turn reactivity**. The lit review's tau-bench reliability findings measure actual reactivity (the model receives and reacts to real tool results); F2 facsimile measures planning capability under imagined conditions.

The 6 traces collectively demonstrate that Sonnet 4.6 CAN reason correctly through the multi-turn protocol — the plan-correctness-given-imagined-scenario read is 6/6 valid plans. But it also demonstrates that F2 facsimile cannot substitute for F1 dispatch when the test is about reactivity to specific fixture states.

## Implication for Spike B synthesis

- The frontier-arm tau-shape data point under F2 facsimile is **methodologically limited**, not a clean reliability comparison against cheap-tier's 100% pass on the same fixture.
- The cycle's RQ-3 finding on this fixture is constrained to the cheap-tier data: **MiniMax M2.5 Free reliably handles single-control multi-turn library checkout** (12/12 scenario success post-fixture-fix).
- For frontier-tier multi-turn reactivity, the **real-session fixture (haiku authoring) gives an authentic test** — subagent has Read/Edit/Bash tools natively and runs real multi-turn behavior.
- **F2 cannot answer the cross-tier-on-tau-shape question.** Either we accept this scope limit and rely on the real-session fixture for cross-tier multi-turn evidence, or we'd need to invest in F1 turn-by-turn dispatch.

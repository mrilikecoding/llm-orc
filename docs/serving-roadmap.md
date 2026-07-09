# Serving Roadmap — to the North Star

**North star:** full model parity through composition (see `docs/serving.md`)
— the endpoint does everything a strong single model does behind a coding
tool, at ~zero marginal cost, and then exceeds it where composition has
structural advantages (verified acceptance, lossless memory, provenance).

**How progress is measured:** the 8-turn escalating ladder (build → edit →
tests → explain-why → second module → cross-file → run-tests → refactor),
driven through real OpenCode, scored per-turn against the same ladder run
with an Anthropic model (Haiku 4.5 / Sonnet 5) as the backend. The Cycle-7
benchmark harness on `research/agentic-serving-corpus` (`benchmark-runs/`)
is the automation to revive for this. Baseline 2026-07-08: 4/8 → 5/8 after
the pre-release fixes; ~55–60% of the Haiku-backed experience.
**Released as v0.18.0 (2026-07-09)** after an 8-angle review pass (8
correctness findings fixed pre-merge; deferred findings tracked as #91–#96).

## Stage 1 — Reliability: the accept-gate retry round (#89) — SHIPPED in v0.18.0

The dominant failure class — confirmed on BOTH seat tiers (local qwen3:8b
and qwen3.6-plus via Zen) — was one wrong expectation among ~5 generated
tests killing an otherwise-correct turn. Shipped: build-gated wraps
build-gated-round in the engine's `loop:` primitive (bounded, carry = the
executor's failure report). **Key finding from the A/B: structure is the
lever, not model size.** Live observation with retries wired: the residual
reject on hard turns is judge conservatism — judge false-reject rate is now
the measurement target (#84). The exit gate (ladder ≥ 7/8) is pending a
full post-release ladder rerun.

## Stage 2 — Memory rung 2′: lossless session record + selection (#82)

Replace the rung-1 window with the lossless per-session record and
deterministic referent selection (design:
`docs/plans/2026-07-08-serving-conversation-memory-design.md`). Clears the
cross-file/window-loss failure (turn 6: correct code, starved sandbox).
Entry gate: observe OpenCode's compacted wire on one long session before
hardening the divergence classifier (rebuild-from-wire is the interim
fallback). Exit gate: the cross-file ladder turn passes; a 20+-turn session
retains turn-1 referents.

## Stage 3 — Client execution surface (#83)

Files the conversation didn't write, and running things: thread client
read-tool results into the turn, and/or client-delegated execution (the
emit node's permission seam reused for a test-run tool_call — ADR-048
ODP-1). Closes run-tests (today: narration, not action) and fix/edit on
pre-existing files — the two biggest parity holes. Exit gate: "run the
tests" executes; editing a file the serve never wrote round-trips.

## Stage 4 — Shapes and seat tiering

Operator-curated catalog growth on the ADR-047 registry: a fix shape, a
refactor shape, grounded-explain (file-reading explain seat). Per-seat
escalation (local 14b / Zen qwen3.6-plus) on failure signals rather than by
default — the 2026-07-08 A/B showed the default stays local-first. The
elicit-then-build shape (criteria-first) graduates from fallback to a
selectable catalog entry.

## Stage 5 — Beyond parity: the plexus substrate (rung 3)

The session record ingests into a plexus knowledge graph; per-dispatch
slices become provenance-tracked lens queries, cross-session. This is where
the serve stops chasing a single model and passes it: memory with no
recency decay, provenance on every remembered fact, knowledge that
accumulates across sessions. Composes with the accept gate's contract
hierarchy (assume-guarantee direction noted at Cycle-8 close).

## Standing constraints

- 32GB rig is the permanent target; interactive latency is first-class.
- Local-first defaults; hosted seats (Zen subscription) are operator opt-in.
- Every stage lands with its ladder rerun and the parity score updated here.
- Deterministic control, model judgment only inside bounded, closed-set,
  gate-backstopped decisions.

## Issue index

Roadmap stages: #82 lossless record (Stage 2) · #83 client execution
(Stage 3) · #84 accept-gate harness + judge false-reject measurement ·
#85 sandbox hardening · #90 llama.cpp backend (bootstrapping).
Review debt remaining: #93 (executor reuse + registry-scan caching) ·
#95 (dead EnsembleConfig fields + prime machinery removal).
Shipped: #87, #88, #89 (v0.18.0); #86, #91, #92, #94, #96 and the safe
halves of #93/#95 (v0.18.1).

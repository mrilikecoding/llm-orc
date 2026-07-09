# Serving Roadmap — to the North Star

**North star:** full model parity through composition (see `docs/serving.md`)
— the endpoint does everything a strong single model does behind a coding
tool, at ~zero marginal cost, and then exceeds it where composition has
structural advantages. Concretely, "comparable or superior to a frontier
model (latency notwithstanding)" decomposes into three levers a monolithic
model does not have:

1. **Verified acceptance** — deliverables pass a deterministic executor and
   an independent judge before they ship; a raw model never checks its work.
2. **Lossless memory** — deterministic selection over the full history (and
   eventually cross-session substrate) instead of attention over a decaying
   context window.
3. **Zero marginal cost** — systematic coverage (more rounds, more
   verification, more retrieval) is free where every frontier token is
   billed.

Latency is the accepted trade; correctness and memory are where we compete.

## How progress is measured

The escalating multi-turn ladder driven through **real OpenCode** (never
harness-only), scored per turn against the same ladder with an Anthropic
model as the backend. Trajectory so far:

| Date | Battery | Score | Notes |
|------|---------|-------|-------|
| 2026-07-08 (arc start) | 8-turn ladder | 4/8 | silent wrong verdicts |
| 2026-07-08 (v0.18.0) | 8-turn ladder | 5/8 | all failures honest rejects |
| 2026-07-09 (Stage 2 core) | 9-turn todo app | **8/9** | multi-file build + deep recall pass; one honest reject |

The Cycle-7 benchmark harness (`research/agentic-serving-corpus` branch,
`benchmark-runs/`) is the automation to revive for a standing
parity-percentage arm (Haiku 4.5 / Sonnet 5 behind OpenCode as baseline).

## Current state (2026-07-09)

Released: **v0.18.0** (agentic serving) and **v0.18.1** (review-debt sweep).
Pending review: **PR #99** — Stage 2 core (wire observation + full-history
selection + gate-runner TestCase support).

Shipped capability: build (accept-gated, bounded retry), explain,
edit-existing (conversation-written files), within-session memory with
lossless file selection, deterministic routing with a guarded decider.
All-local (qwen3:8b) by default; operator seat overrides via `*.local.yaml`.

Key empirical facts the next work builds on:

- OpenCode's wire is **append-only** (stable prefix hashes, no compaction
  through 30+ messages) — `LLM_ORC_SERVE_WIRE_LOG` watches for the day that
  changes.
- **Structure beats model size**: the seat A/B showed a hosted frontier
  model does not fix the dominant failure classes; shape and verification
  changes do.
- The dominant hard-turn failure is now **spec-freedom divergence**:
  independently resampled tests and code disagree on choices the spec
  leaves open.

## The path, in order of leverage

### 1. TDD retry loop (#100) — next up

On a gate reject where round 1's tests loaded and ran, hold those tests
fixed as the spec and regenerate only the code (regenerate tests only when
they failed to collect). Directly attacks the spec-freedom failure (the one
lost turn in the 9-turn battery) and strengthens ADR-048's
criteria-primacy: round 1's tests become round 2's concrete criteria.
Exit gate: the cli.py-style integration turn passes; ladder ≥ 9/9.

### 2. Client execution surface (#83)

Files the conversation didn't write, and running things: thread client
read-tool results into the turn and/or client-delegated execution (emit's
permission seam reused for a test-run tool_call — ADR-048 ODP-1). Includes
the tool-mapping step (resolve emit outcomes against the client's
advertised tools instead of the hardcoded `write`). Closes run-tests and
pre-existing-file editing — the two biggest remaining parity holes.

### 3. Gate integrity pair (#98, #84)

#98: test-writing turns validate a shadowed composite in the shared exec
namespace — route "write tests" to a dedicated shape (the deliverable IS
the test file, run against the materialized workspace alone). #84: measure
judge false-reject rate on fixtures (with retries wired, judge conservatism
is the visible bottleneck on hard turns) and revisit ADR-048 §5's
AND-vs-weighted composition with data.

### 4. Shapes and seat tiering

Operator-curated catalog growth: fix shape, refactor shape,
grounded-explain (file-reading explain seat), elicit-then-build
(criteria-first) as a selectable entry. Escalation-on-signal seat tiering
(local 14b / hosted) — the default stays local-first per the A/B result.

### 5. Memory remainder (#82) and beyond-parity

Kept on #82: the server-side record + divergence classifier, deliberately
deferred until client compaction is actually observed on the wire; and
prose-turn retrieval beyond the recency tail (only files retrieve from deep
history today). Beyond parity: the record ingests into a plexus knowledge
graph, per-dispatch slices become provenance-tracked lens queries, and
memory becomes cross-session with no recency decay — where the serve passes
a single model rather than chasing it.

### 6. Bootstrapping and hardening

#90 llama.cpp as a built-in backend (drop the Ollama process dependency —
clear-cut default-model bootstrap); #85 sandbox hardening
(container/seccomp) before untrusted deployment; #93/#95 remainders
(executor reuse + registry-scan caching; dead-field and prime-machinery
removal).

## Standing constraints

- 32GB rig is the permanent target; interactive latency is first-class.
- Local-first defaults; hosted seats are operator opt-in, never tracked.
- Real-client validation at the earliest runnable point; every stage lands
  with its ladder rerun and the trajectory table updated.
- Deterministic control; model judgment only inside bounded, closed-set,
  gate-backstopped decisions.

## Issue index

Path: #100 TDD retry · #83 client execution · #98 test-shape gate ·
#84 judge measurement · #82 memory remainder · #90 llama.cpp ·
#85 sandbox hardening · #93/#95 remainders.
Shipped: #87 #88 #89 (v0.18.0) · #86 #91 #92 #94 #96 (v0.18.1) ·
#82-core + gate-runner TestCase support (PR #99).

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

**Battery realism ladder (named 2026-07-09):** the todo app is a toy.
Daily-driver parity means real complexity, and the ladder's upper rungs are
named: (a) the **self-referential meta-task** — answering questions about
the llm-orc codebase itself (or plexus, or their interaction) through the
serve, which exercises retrieval over a real repository rather than
conversation-written files; (b) the **fix-execution milestone** — the
serving layer executing a fix on a real codebase end-to-end (locate, edit,
run tests, verify); (c) the apex — the serve improving ITSELF
(self-hosting development: planning, chunking work, web search, fix
execution on its own repo). Rungs (a) and (b) hang off the client
execution surface (#83). Evaluation method for (c): an agent driving the
serve through the OpenCode CLI judges the serve's decisions against what
it would do itself — a shadow-comparison judge. Intermediate rungs get
designed from run evidence, not pre-specified.

Two generalizations the upper rungs force (named 2026-07-09):

- **Languages.** The verified-build gate is Python-scoped today (executor
  sandbox + adequacy checker), failing closed on everything else — and
  plexus is RUST, so the meta-task rung's plexus half is the first
  concrete non-Python need. Generalization is seat swaps behind the same
  `{requirement, code, tests}` contract: a per-language sandboxed executor
  and a per-language adequacy checker; the round, router, held carry, and
  gate composition are unchanged. Built when a rung demands a language,
  not speculatively.
- **Task shapes.** Fable-parity means evaluating a task's shape and
  deciding how to proceed — up to and including **an ensemble that designs
  other ensembles to suit the task at hand**. That is ADR-047's deferred
  composer-ensemble path, re-elevated from named-forward-direction to
  north-star mechanism: catalog growth is the manual rung
  (operator-curated shapes, deterministic selection via classify +
  registry), the compose-at-runtime primitive (ADR-047's four named engine
  gaps) is the enabling rung, and composer ensembles — composing from the
  registry's validated parts, output structurally validated before it
  registers, the composer itself a verified ensemble rather than a lone
  model — are the generative rung.

## Current state (2026-07-09)

Released: **v0.18.0** (agentic serving) and **v0.18.1** (review-debt sweep).
Merged and released: **PR #99** (Stage 2 core — wire observation +
full-history selection + gate-runner TestCase support) as **v0.18.2**;
**PR #101** (#100 TDD retry loop + the 2026-07-09 live-diagnosis fixes:
reject-noise filter, deliverable test-fence pollution, reflection-test
prompt ban, `held_round` + trace-depth observability) as **v0.18.3**.

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
- **Round-1 test quality gates everything** (2026-07-09 live diagnosis):
  the held TDD retry fires only when round 1's tests collected AND were
  judged adequate, so judge adequacy (#84) and test-writer quality (#98)
  directly bound the retry's live win-rate. The 8b seat's reflection-test
  mode (hasattr/callable — fails right code, passes wrong code) was banned
  in the seat prompt; failures moved to ordinary code/test disagreements.

## The path, in order of leverage

### 1. TDD retry loop (#100) — SHIPPED in v0.18.3

The loop body is now a deterministic router: a reject whose tests collected
and were judged adequate carries them under a HELD TESTS sentinel, and
round 2 dispatches to `build-code-round` — code only, adequacy carried,
executor as the live gate. Proven hermetically (the real graph through the
real engine) and live (route → held round → qwen3:8b regenerated code →
accept, `held_round: true`). Remaining exit gate: a full clean-session
ladder rerun once the #84/#98 bottleneck stops masking it — live diagnosis
showed the held path's entry condition is judge adequacy on round 1, which
moves the gate integrity pair up in leverage.

### 2. Client execution surface (#83)

Files the conversation didn't write, and running things: thread client
read-tool results into the turn and/or client-delegated execution (emit's
permission seam reused for a test-run tool_call — ADR-048 ODP-1). Includes
the tool-mapping step (resolve emit outcomes against the client's
advertised tools instead of the hardcoded `write`). Closes run-tests and
pre-existing-file editing — the two biggest remaining parity holes — and is
the enabler for both named upper battery rungs (the codebase meta-task
needs real-repo retrieval; the fix-execution milestone needs edit + run).

### 3. Gate integrity pair — #84 MEASURED AND CLOSED; #98 next

**#84 (done, released in v0.18.4):** the fixtures harness
(`benchmarks/judge_adequacy`, 16 labeled fixtures × 8 samples, live seat)
measured the model judge at FAR 0.0 (never accepts garbage tests) but FRR
25–67% on adequate tests, near-deterministically per fixture; three prompt
variants moved the miscalibration around without removing it. Every
inadequate class has a static signature, so the gate's adequacy signal is
now the deterministic value-bearing-assert checker (`adequacy_check.py`) —
FRR/FAR both 0 on all 16 fixtures by construction, one less model call per
round, judge conservatism retired as a failure class (ADR-048 amended).
The model seat survives as `adequacy-judge` for the harness. Live residual
after the swap: test-writer quality (reflection-style relapses now get
correctly rejected instead of stochastically judged) — which is #98's
territory.

**#98 (next):** test-writing turns validate a shadowed composite in the
shared exec namespace — route "write tests" to a dedicated shape (the
deliverable IS the test file, run against the materialized workspace
alone), reusing the deterministic adequacy checker.

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

Path: #98/#84 gate integrity (raised) · #83 client execution ·
#82 memory remainder · #90 llama.cpp · #85 sandbox hardening ·
#93/#95 remainders.
Shipped: #87 #88 #89 (v0.18.0) · #86 #91 #92 #94 #96 (v0.18.1) ·
#82-core + gate-runner TestCase support (v0.18.2, PR #99) ·
#100 TDD retry + live-diagnosis fixes (v0.18.3, PR #101).

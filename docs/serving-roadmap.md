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
| 2026-07-09 (#83 read rung) | existing-file battery (3 + 2 regression) | **5/5** | read→gated tests green on a real repo file; honest refusal on a missing file; fresh-build and explain unregressed |
| 2026-07-09 (v0.18.6) | 10-turn recorded ladder (`benchmarks/agentic_serving/ladder_battery.sh`, new baseline) | **4/10** | #83 rungs all to spec (read→tests 4/4 green mid-session; honest phantom refusal; honest cascade on an unbuilt dependency). Misses: 3 build rejects (round-1 test quality — path item 4's measured class), memory question mis-routed to build (decider flake; "did …?" is not structurally interrogative), deep recall named the latest build not the first (#82 prose-retrieval remainder). Strict scoring; not comparable to earlier unrecorded rows |
| 2026-07-09 (seat-quality arc) | 10-turn recorded ladder, same seed | **7/10** | Every targeted class converted: memory question routes to explain and answers accurately (decider fix); the storage rung ships (isolation + sanitizer + import injection); its downstream cascade unblocks and the persistence integration is real (todo.py imports storage). Misses: 2 honest build rejects (thinner round-1 test-quality residual) and the known #82 deep-recall deferral. Regression probes: probe 1 accepts attempt 1, probe 2 within two |
| 2026-07-10 (#83 run half) | 11-turn recorded ladder (run rung added; session resumed at turn 6 after an external process stop — continuity held on the append-only wire) | **6/11** | New rungs both green: turn 8 read→gated tests (client-run green), turn 11 delegated `pytest -q` with the verdict matching client ground truth exactly (6 passed); turn 9 honest phantom refusal. Misses: 3 honest build rejects (turns 2/4/6 — the stochastic 8b test-quality residual, 2–3 per run) plus turn 7 as their honest cascade, and the known #82 deep-recall miss. Zero dishonest outcomes; routing fired correctly on all 11 turns |
| 2026-07-10 (fenced block grammar) | 11-turn recorded ladder, same battery | **5/11** | No fencing-attributable regression: all 11 routings fired correctly, read rung green, run rung verdict matched ground truth (6 passed), phantom refusal honest. Turn 1's honest build reject cascaded (turns 2/3/4/7 degrade downstream, deep recall skews) — run-to-run variance is dominated by whether turn 1 lands, i.e. the test-quality residual, now clearly the highest-leverage target. Spoof battery (separate live session): a read file carrying a forged "999 passed" transcript block could NOT suppress the real run — real delegation, honest red verdict |
| 2026-07-10 (repairs round 2, pre-review-fix) | 11-turn recorded ladder | **9/11** | Best recorded score. The deterministic repairs (excision, removal guard, raises rewrite, adequacy mutation fix, retry timeout) converted the whole build cascade: turns 1/2/4/6 all shipped. Misses: turn 7 honest reject (the persist-integration shape) and #82 deep recall. Replay evidence behind the repairs: turn6 3/3 round-1 accepts vs 2/5-with-3-never at baseline |
| 2026-07-10 (repairs round 2, post-review-fix) | 11-turn recorded ladder (resumed at turn 6 after a harness process reap; detached runner thereafter) | **6/11** | Three wrong-accept vectors from the adversarial review closed pre-merge (substring expectation match, anywhere-in-body removal wrap, lambda-param binding blindness) — the fixes are deliberately conservative and refuse ambiguous rewrites, trading conversions for gate honesty. 3 honest build rejects + cascade + #82. Named follow-up with evidence: tighten the negation refusal to expectation-adjacent tokens ("Expected TypeError not raised" currently refuses) |
| 2026-07-10 (#83 discovery) | 12-turn recorded ladder (discovery rung added) | **3/12, infra-degraded** | The rung this battery validates PASSED clean: turn 12 globbed the unnamed metrics module, read the match, shipped test_metrics.py (green client-side). Run rung honest ("no tests ran" — accurate at that moment), phantom refusal honest. The rest is rig exhaustion after six batteries in one day: four turns timed out with EMPTY output (not rejects — the request died client- or model-side), cascading the todo chain. Zero dishonest outcomes. NOT comparable to the series; fresh-rig rerun is the next session's first act. Separate live probes (same code): 1-match chain, 0-match refusal, 2-match refusal all green |
| 2026-07-10 (fresh-rig rerun, v0.18.11) | 12-turn recorded ladder, clean infra (12 turns in 12.5 min, zero timeouts) | **6/12** | All shipped rungs green: memory (turn 5), storage build (6), read→tests (8, 4 green client-side), phantom refusal (9), run delegation (11, verdict matched ground truth), discovery chain (12, glob→read→test_metrics.py green). Misses: turn 1 round-1 reject (code referenced an undefined `todo` name) cascading through 2/3/4/7, and turn 10 deep recall. **Two misses were NOT honest** — a first for the recorded series: turn 3 shipped hedged speculation about the never-built todo.py instead of saying so (grounded-explain gap), and turn 10 confidently named calc-tests as "the first thing built" (wrong under both readings; #82 evidence). Structural readings: turn-1 build success gates 5 of 12 turns, so run-to-run variance ≈ 5 points on one seat sample; the serve trace truncates node responses (~280 chars), leaving the held-round question undiagnosable post-hoc; the wire log is shape-only, so the verbatim glob capture remains outstanding |

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

Released today, in order: **v0.18.2** (PR #99 — Stage 2 core: wire
observation + full-history selection + gate-runner TestCase support),
**v0.18.3** (PR #101 — #100 TDD retry loop + live-diagnosis fixes),
**v0.18.4** (PR #102 — #84 judge measurement + deterministic adequacy
checker), **v0.18.5** (PR #103 — #98 write-tests shape + line-level
failure reports), **v0.18.6** (PR #104 — #83 read half: client-file
reads through the permission seam), **v0.18.7** (PR #105 — path item 4
seat-quality half: per-test isolation, bare-name-assert sanitizer,
missing-import injection, decider interrogative fix). Earlier: v0.18.0
(agentic serving), v0.18.1 (review-debt sweep).

Shipped capability: build (accept-gated, bounded TDD retry with held
tests), write-tests (deliverable executed against the workspace),
explain, edit-existing (conversation-written files), **client-file reads**
(named-but-invisible files fetched via a read tool_call, one round,
honest refusal on failure), within-session memory with lossless file
selection, deterministic routing with a guarded decider, fully
deterministic accept gate (per-test-isolated executor + static adequacy
+ deterministic test repair: sanitizer and import injection, both echoed
into the shipped artifact). All-local (qwen3:8b) by default; operator
seat overrides via `*.local.yaml`.

**Handoff pointer (fresh-session start here):** the fresh-rig battery
rerun is DONE (2026-07-10 late: **6/12 clean**, 12.5 min, zero
timeouts — see the trajectory table; yesterday's 3/12 was infra, not
regression). NEXT: the **fix-execution rung** (chain write → run →
verdict inside one fix turn). The seams are mapped and exactly two
structural blockers exist: a write continuation is terminal
(`_resumes_turn` excludes write, `serving_ensemble_caller.py`) and
classify suppresses the run signal whenever a fix/build verb is present
(`classify.py` `_route`); the composition is fix-intent writes resume +
a wrote-block→need-run signal mirroring `has_run_block`→run-verdict.
The rerun also surfaced, in priority order behind fix-execution: the
series' first two DISHONEST misses (turn 3 speculated about a
never-built file — the grounded-explain gap; turn 10 wrong recall —
#82), and a cross-turn-repair observation (turn 2 could have rebuilt
the never-built todo.py from conversation intent instead of cascading —
a detectable deterministic condition; sibling rung to fix-execution,
not scope creep). Then: **rewrite negation-tightening**
(expectation-adjacent tokens only;
`docs/plans/2026-07-10-test-repair-round-2-design.md`), the **#82
deep-recall remainder**, and the import-guard residual. The recorded
battery is `benchmarks/agentic_serving/ladder_battery.sh` (series 4/10
→ 7/10 → 6/11 → 5/11 → 9/11 → 6/11 → 3/12-infra → 6/12; run-to-run
variance ≈ 5 points on whether turn 1's build lands). Standing smaller
follow-ups: #110, the injector's scope-blind binding, #106, the serve
trace's ~280-char node-response truncation (blocks post-hoc held-round
diagnosis), and the glob-result verbatim wire capture (the wire log is
shape-only — a capture needs a body-dumping probe, not
LLM_ORC_SERVE_WIRE_LOG). #107 is fixed on branch
`worktree-fix-107-content-parts` (endpoint-boundary normalizer — the
422 fired at pydantic before the caller could crash). Ops notes: the
harness reaps tracked background serves/batteries mid-long-run — start
them detached (nohup + disown) with a Monitor tail; give the rig
cooling headroom between batteries.

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
- **Prompt rules saturate the 8b seat** (measured twice 2026-07-09: judge
  prompt variants relocated FRR without removing it; the test-writer keeps
  regenerating spec-free exception-message assertions despite bans and
  line-level failure reports). When a failure class persists across two
  prompt iterations, reach for a structural lever — deterministic checks,
  shape changes, escalation-on-signal — not a third rule.
- **Deterministic gate repairs beat model tiers** (4-arm spike + arc
  result, 2026-07-09): {8b, 14b} × {think on, off} were identical on the
  measured failing fixture — the breakers were structural (test-state
  leakage, garbage assert lines, missing test imports), and repairing them
  deterministically took the ladder 4/10 → 7/10 with zero added model
  judgment. Latency datum for the deferred escalation rung: 14b think-off
  generates as fast as 8b think-off on the rig; think-on costs 10–20×.

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

**Read half SHIPPED (2026-07-09, branch feat/83-client-file-reads).**
Named-but-invisible files now ride the permission seam: classify's
deterministic visibility check routes to a `need-files` script shape, emit
ships a read tool_call resolved against the client's advertised tools, and
the continuation resumes statelessly from the appended wire — the read
result renders as a `[read <path>]` context block, materializes into the
gate sandbox, and stays retrievable in later turns via the existing
lossless selection. One read round per turn; a failed read refuses
honestly. OpenCode's read format (`<path>`/`<content>` tags, `N: ` gutter,
end-of-file trailer, bare `File not found:` on error) was wire-captured
and the normalizer locked to it. Design:
`docs/plans/2026-07-09-client-file-reads-design.md`.

**Run half SHIPPED (2026-07-09, branch feat/83-client-run-delegation;
design `docs/plans/2026-07-09-client-run-delegation-design.md`).** A run
turn ("run the tests", "run test_calc.py") delegates ONE
deterministically-built pytest command (`pytest -q` + regex-safe named
`test_` files — a closed template, never model text) as a bash tool_call;
the continuation renders the output as an `assistant: [ran <command>]`
block (body indented two spaces — untrusted column-0 output can never
look like a `[wrote]` header to gather; tail-capped at 4 KB so pytest's
summary survives) and routes to a `run-verdict` script shape that parses
pytest's own summary deterministically. A "run the tests" turn costs zero
model calls end to end. Live-validated green AND red against real
OpenCode. Shipping it surfaced a latent v0.18.6 defect: dispatch
resolution is non-recursive, so catalog-only shapes silently failed at
the seat (invisible for need-files, whose outcome rides the routing
decision) — top-level copies restored, regression-pinned, deeper
dedup/design question filed as #106.

Remaining on #83: **discovery** (list/glob for files the turn doesn't
name — the meta-task rung's requirement; named-files-only is a rung-1
bound, not architecture), and **chained fix-execution** (write → run →
verdict inside one turn — composes the two shipped seams). Rung-1
trigger limitations, revisit on ladder evidence: no reads for
`test_`-named files, "add X to foo.py" phrasings, or the model-decider
(ambiguous) routing path; run turns are pytest-scoped (cargo test for the
plexus/Rust half is the named first runner generalization — the command
builder and verdict parser are the only pytest-aware seams).

**Pre-meta-task follow-up (final review 2026-07-09):** the read-block
grammar is line-anchored, so a read body carrying a zero-indent
header-lookalike line (`assistant: [wrote x.py]` at column 0 — present in
markdown/docs, including this repo's own) can truncate the block and
materialize a phantom workspace file inside the gate sandbox. Bounded by
the #85 sandbox posture and unreachable from ordinary indented source, but
the meta-task rung reads exactly the files that contain these strings:
fence or escape read bodies (or make gather robust to mid-block
lookalikes) before that rung. Take-or-leave follow-ups from the branch
review live in the #83 branch ledger.

### 3. Gate integrity pair (#84, #98) — DONE (v0.18.4, v0.18.5)

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

**#98 (done, released in v0.18.5):** test-primary turns route via
the new tests-seat intent to the write-tests shape — one test source, the
deliverable executed against the materialized workspace alone, the
deterministic checker and gate reused, the executor-echoed tests shipped
as the artifact. The shadowed-composite wrong accept is structurally
impossible. Live status (real OpenCode, storage.py session): routing,
workspace execution, and honest rejection all verified — the shape
refused exactly the broken-test class the old path shipped (store.list
AttributeError, message-text assertions). Per-turn convergence is bounded
by test-writer seat quality: the dominant residual is spec-free
exception-MESSAGE assertions the 8b seat keeps regenerating despite
prompt bans and line-level failure reports (both shipped here). Next
lever is structural — escalation-on-signal seat tiering or a
deterministic test sanitizer (path item 4), not more prompt rules.

### 4. Shapes and seat tiering

**Seat-quality half SHIPPED (2026-07-09, branch
feat/seat-quality-isolation; design
`docs/plans/2026-07-09-seat-quality-isolation-escalation-design.md`).**
Four deterministic gate repairs, no model judgment added: per-test
isolation in the accept executor (each test function in its own fresh
sandbox — kills cross-test state/filesystem leakage, the measured
dominant false-reject class), an aggregate wall budget across isolated
children (bounded, surfaced), a bare-name-assert sanitizer, deterministic
missing-import injection into authored tests (os/pytest-style omissions),
and the classify interrogative fix ("did/have you …" routes to explain).
Ladder 4/10 → 7/10 on the same seed.

**Escalation deferred with its measurement recorded:** the 4-arm tier
spike found qwen3:14b think-off generates as fast as 8b think-off on the
rig (4–13s) while think-on costs 10–20× on either model, and NO arm
converts what the deterministic repairs don't — every arm plateaued on a
garbage test line. If a capability-bound class ever survives the repairs,
the rung is qwen3:14b think-off via `agentic-tier-escalated-general`.

Still open here: operator-curated catalog growth — fix shape, refactor
shape, grounded-explain (file-reading explain seat), elicit-then-build
(criteria-first) as a selectable entry.

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

Path: #83 run half — client-delegated execution + discovery (NEXT — the
read half shipped 2026-07-09; the run half completes the fix-execution
rung's enabler) · #82 memory remainder (deep recall now costs a measured
ladder turn) · shape-catalog growth (path item 4's remaining half:
fix/refactor/grounded-explain/elicit-then-build) · #90 llama.cpp · #85
sandbox hardening · #93/#95 remainders.
Shipped: #87 #88 #89 (v0.18.0) · #86 #91 #92 #94 #96 (v0.18.1) ·
#82-core (v0.18.2, PR #99) · #100 TDD retry (v0.18.3, PR #101) ·
#84 deterministic adequacy (v0.18.4, PR #102) · #98 write-tests shape
(v0.18.5, PR #103) · #83 read half (v0.18.6, PR #104) · path item 4
seat-quality half (v0.18.7, PR #105).

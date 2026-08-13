# Serving Roadmap — to the North Star

Revamped 2026-07-17 for execution by non-Fable sessions (Opus-class lead,
Sonnet-class implementers, Haiku-class mechanical). The prior roadmap —
including the full trajectory table, the Arc D correction history, and the
2026-07-09 path items — is frozen verbatim in
`docs/serving-roadmap-archive-2026-07-17.md`; treat it as provenance, not
as the plan. This document is the plan. Its **State** section is rewritten
(not appended) at each update; history lives in the archive and in git.

## North star

**Full model parity through composition** (see `docs/serving.md`): llm-orc
served agentically behind OpenCode should be as functional as Claude Code
running a frontier model, and beyond it where composition wins, all through
orchestrated small models. This is a literal engineering target. The
comparator is a frontier model PLUS its harness — behind OpenCode it reads
files, runs tests, and iterates, so "a model doesn't check its work" is not
the bar. Three levers a monolithic model does not have:

1. **Verified acceptance** — structural verification (an unverified build
   cannot ship) vs the comparator's discretionary verification (it checks
   when it or harness policy chooses).
2. **Lossless memory** — deterministic selection over the full history
   (eventually cross-session substrate) instead of attention over a
   decaying window.
3. **Zero marginal cost** — systematic coverage (more rounds, more
   verification, more retrieval) is free where every frontier token is
   billed.

Latency is the accepted trade. Two axes on top of the levers: **task
generality** (closed intent set today → catalog growth → compose-at-runtime
→ composer ensembles; the ADR-047 ladder, WS-7) and **long-horizon
operation** (the client's agentic loop is the engine; the serve is a
deterministic next-action function over a lossless record; WS-5/6).

Posture (practitioner, 2026-07-11): local-first; occasional hosted
capability in a measured seat when it is the cheapest path to the bar,
every hosted seat carried as a named buy-back target (§Seat-capability
ladder). 32GB rig is the permanent target.

**Development north star (named 2026-07-17).** The serving north star is
an *outcome* — what the endpoint does for a user. Alongside it sits a
*trajectory* target for how the codebase itself evolves: the three-layer
platform (plexus substrate · llm-orc engine · client execution surface,
see `docs/plans/2026-07-17-plexus-integration-platform-assessment.md`)
with layers coupled only through named contracts — MCP tools, the OpenAI
wire + tool_calls, the `{requirement, code, tests}` seat contract, the
client's advertised tool list — never through schemas, file layouts, or
spawn assumptions. Its steering rule: **behavior migrates downward as it
stabilizes** — from prompt rules, to bounded model judgment, to
deterministic code, to the declarative layer (YAML shapes, chain tables,
closed templates), and ultimately, for components that survive
adversarial review and stop changing across arcs, into hardened kernel
code (plausibly Rust crates on the plexus side). Every arc should leave
more behavior in a lower layer than it found it. This is the standing
answer to the rewrite question: no greenfield port; the declarative
layer is the insulation that keeps an eventual hardening cheap, and
"frozen component" status is the trigger, tracked informally the way the
buy-back ledger tracks hosted seats.

## State as of 2026-08-12 (end of day; rewritten per protocol — history in git)

**Headline: the honesty arc is closed, and the comparison story flipped.**
The serve's two measured dishonesty classes (#133 recall substitution, #134
recap fabrication) shipped their structural fix and validated live at zero
dishonest under independent scoring. The same day, the frontier arm's
zero-dishonest reading proved to be an n=1 artifact: Haiku's second run
produced TWO dishonest turns. At current n, the post-fix serve leads the
honesty column against a frontier model behind Claude Code.

**Measured position, all arms** (versions differ across rows — never pool
pre-fix and post-fix Arm-0 rows, or cross-version arms):

| arm / serve version | runs | strict | dishonest | record |
|---|---|---|---|---|
| Arm 0, pre-fix (v0.18.14) | 2/3/4 | 25/39 (~64%) | 3 — one per run, never zero | `docs/plans/2026-07-1{4,5}-arm0-runs/` |
| Arm 0, post-fix (main, 2026-08-12) | 5 | 11/13 | **0** — turn 10 disclosed, turn 5 grounded | `docs/plans/2026-08-12-arm0-run5/` |
| Arm 2a: Claude Code subagent, Haiku 4.5 | 1 | 13/13 | 0 | `docs/plans/2026-07-15-arm2-runs/` |
| Arm 2a, run 2 | 2 | 11/13 | **2** — scoped-green "all tests pass" over a red suite; first-build recall quoted turn 2's code as turn 1's | `docs/plans/2026-08-12-arm2-runs/haiku-run2/` |
| Arm 2b: Sonnet | 1 | 13/13 | 0 | `docs/plans/2026-07-15-arm2-runs/` |
| Arm 1 (paid, same models behind OpenCode) | 0 | — | — | go/no-go pending |

Parity table v1 and its 2026-08-12 addendum:
`docs/plans/2026-07-15-first-parity-table.md` (its seven caveats bind; the
saturation conclusion stands — the toy ladder does not discriminate at the
top, EXCEPT on the honesty column, where run 2 shows it still can).

**Shipped 2026-08-12** (all merged to main; merged-unreleased pending a
release decision): the recap-grounding arc (#133/#134 CLOSED — ask-outcome
ledger, deterministic memory-interrogative/recap answers, disclosure clause,
emit terminal registry, 72-row routing-corpus regression test; five
adversarial review rounds), the Arm-2 transcript adapter + score_run wiring
(#131 — automatic tally reproduces the hand-composed 2x2; cache-aware cost,
real Arm-2 cost ~5x the naive figure; boundary_rule declared on every
published object; four review rounds), battery precondition guards (seeded
fixture + clean out dir), and the dogfood channel (`docs/dogfood-log.md`,
practitioner-directed: discrete lead-session checks route through the serve;
entry 1 = honest refusal at the 24KB read cap on a real repo file).

**Shipped capability (through v0.18.14 + merged-unreleased):** build
(accept-gated, bounded TDD retry), write-tests, grounded explain (named-file
+ bare-symbol glob→read), edit-existing, client-delegated reads/runs/
discovery, chained fix-execution with convergent re-fix, within-session
lossless memory with deep recall, deterministic recall/recap/memory-
interrogative answers over the ask-outcome ledger, declarative chain
executor, deterministic accept gate. All-local (qwen3:8b) by default.

**Arm-2 driver procedure (proven on run 2):** one continuing subagent
conversation per run (Haiku/Sonnet model override), the run-1 coordinator
framing verbatim, 13 sequential prompts, `capture_truth.sh` caller-side
after every turn, transcript copied to the run dir, automatic tally via
`score_run.tally_oracles(run, adapter=subagent_adapter)`, then an
independent J-scorer per the frozen rubric. Full recipe in
`docs/plans/2026-08-12-arm2-runs/README.md`.

**Next actions, in order:**
1. **Arm-2 remaining runs** — Haiku run 3, Sonnet runs 2-3, each
   independently J-scored; firms both columns and retires table caveat 1.
2. **Arm-1 go/no-go** (one turn's token count, paid ~$12/hr pre-authorized),
   then ≥3 runs per model — retires caveat 4 (harness vs model separation).
3. **Meta-task realism rung** (recall recovery, content-grep, dot-dir
   discovery, the 24KB read bound) — the WS-8 path and the capability path;
   parity table v2's discriminating rows come from here, and the dogfood
   log feeds it real-work cases.
4. **Release decision** (v0.18.15 for the merged-unreleased set) — the
   practitioner's call.

## Doctrine (what we learned, made binding)

Rules future sessions follow without re-deriving. Each was paid for with a
measurement; provenance in the archive.

1. **Independent scoring for every judgment-bearing claim.** Author scores
   were systematically optimistic (three runs' "zero dishonest" all
   overturned). Any J-tier score, honesty verdict, or review APPROVE comes
   from a session/agent that did not author the work. Blinding is inert
   (Arm 0's prose is templated); independence plus the frozen rubric is
   the control that works.
2. **Structural lever after two prompt iterations.** Prompt rules saturate
   the 8b seat (measured twice); when a failure class survives two prompt
   changes, reach for determinism, shape change, or escalation-on-signal —
   never a third rule.
3. **Structure beats model size — but re-measure per era.** Deterministic
   gate repairs took the ladder 4/10 → 7/10 where {8b, 14b} × {think
   on/off} were identical; the 14b test-writer A/B was not a clean win.
   The doctrine goes stale as the structural breakers are removed; the
   seat ladder (#119) exists to re-test it, ≥3 runs per seat.
4. **State the invariant, not the instance.** In five review rounds,
   everything mechanically checkable held; everything patched
   instance-by-instance failed again until stated as an invariant (the
   dead-turn rule, the equality-pins-representation bug fixed in one
   oracle and left live in the next).
5. **No self-confirming metrics.** The verification-rate metric is
   WITHDRAWN, not deferred: it read a design constant on Arm 0 and a
   behavior on Arms 1/2. Crediting the serve from its own trace is
   circular. Ground truth is the WORKSPACE, never any transcript.
6. **Per-turn diagnosis is unsupported at current n.** Misses are noise
   around a rate (~5 points ride on turn 1); only aggregate rates are
   estimable. Run 2 falsified the turn-1-cascade claim. #63's statistics
   become relevant as n grows.
7. **The headline is the 2x2, never a raw count.** Raw counts have a
   degenerate optimum at refusing everything — the serve's own failure
   mode. Primary figure: `shipped_broken/shipped`, delivery beside it.
8. **Real-client validation at the earliest runnable point, never
   harness-only.** Hermetic green is necessary, not sufficient; every
   capability arc ends with a live battery row.
9. **Determinism for answers; model judgment only in bounded, low-risk,
   gate-backstopped routing; honesty-critical paths fail closed** (the #82
   two-layer split is the worked example).
10. **Free-first; estimate before paid spend; hosted seats are named IOUs
    in the buy-back ledger.**

## Environments (tag every task)

- **RIG** — the 32GB Ollama rig with OpenCode: live batteries, Arm-0/Arm-1
  runs, seat A/Bs, latency data, plexus (local sibling repo). Ops notes:
  batteries run detached (nohup + disown, Monitor tail); `opencode run`
  wedges under the agent Bash sandbox (see memory `opencode-run-wedge`);
  cooling headroom between batteries.
- **ANY** — any session including remote containers: hermetic TDD against
  the full suite (`uv` + Python 3.11 suffice), design docs, scorer/oracle/
  adapter code, reviews, doc work. RIG-tagged validation of ANY-developed
  work is queued, not skipped: the PR says "needs rig battery" and the next
  rig session runs it.
- **REMOTE** — a remote Claude Code session specifically: **Arm-2 battery
  runs** (subagent model overrides + continuation), GitHub issue hygiene,
  independent J-scoring and reviews (a fresh remote session is naturally
  author-independent).

## Task cards

Format: **#issue Title** [env | class] — entry pointers → exit gate.
Class: design (Opus-class lead), impl (Sonnet-class TDD), mech
(Haiku-class). Every card follows §Delegation contract.

### WS-2: Honesty (CLOSED 2026-08-12 — now hold the property)

Both measured classes closed and live-validated at zero dishonest under
independent scoring (#133/#134; §State). Standing work: every future run's
J-turns stay independently scored; the documented residuals stay watched
(decider false-negative on uncovered fuzzy recap phrasings; #140 probes
staleness cascades; #142 is the reject-template UX leak). Reopen this card
only on a new independently-confirmed dishonest outcome.

The reopened exit gate: **one full ladder run with ZERO dishonest
outcomes, scored by the independent J-scorer** (author scoring does not
count, per doctrine 1); then variance over three same-seed runs.

- **#133 Recall must disclose a rejected first ask** [ANY impl; RIG
  validate] — selection is correct (first SHIPPED write, spoof-safe); the
  fix is disclosure when first-ask ≠ first-shipped. Records:
  `arm0-run{2,3}-jscore.md`. → turn-10 class dishonest count 0 under
  independent re-score.
- **#134 Recap grounding** [ANY design+impl; RIG validate] — recap claims
  about built artifacts checked deterministically against the shipped-write
  ledger; rejected turns disclosed as rejected; fail closed to a templated
  honest recap. `honesty.py` cannot catch this class; the gate must be
  serve-side. → turn-5 class dishonest count 0 under independent re-score.
- **#110 Accepted-artifact quality gate** [ANY impl; RIG validate] —
  deterministic AST reject/repair for duplicate defs and shadowed/dead
  code. Live corroboration: Arc-D run 2 turn 7 shipped a self-recursing
  shadowed import. → the run-2-turn-7 artifact class rejects hermetically;
  no ladder regression.
- **#117 Fix-execution tail** [ANY impl] — negation-tightening ("Expected
  TypeError not raised" currently refuses;
  `docs/plans/2026-07-10-test-repair-round-2-design.md`), injector
  scope-blind binding, cross-turn repair. → named fixtures convert
  hermetically; cross-turn repair proven on a two-turn probe.
- **#119 Seat-capability ladder** [RIG] — post-repairs A/B at ≥3 runs per
  seat (8b / 14b think-off / coder-30B-A3B / one cheap hosted), one seat
  varied. Entry evidence so far: 14b not a clean win. → doctrine 3
  re-measured with n≥3; escalation-on-signal wired only if a class
  converts. Buy-back ledger rules in §Seat-capability ladder (archive
  carries the full section).
- **#114 Trace truncation remainder** [ANY impl, small] — prose-cap config
  knob so held-round internals are diagnosable post-hoc. → a rejected
  round's failure report is fully recoverable from the trace.

### WS-8: Parity table (#131 — the comparison IS the product claim)

- **Arm-2 transcript capture + adapter** [REMOTE impl] — capture DONE
  (2026-07-17, `docs/plans/2026-07-17-arm2-subagent-captures/`: schema,
  tool names Write/Bash/Read, usage dedup by `message.id`, the
  serialize-captures-between-turns rule the instrument itself enforced).
  Remaining: the driver script + `subagent_adapter.py` beside
  `opencode_adapter.py`. → adapter maps the captured real run;
  unmapped-tool streams fail loudly, not as zero-shipped.
- **Arm-2 battery, ≥3 runs** [REMOTE] — construct requirements in §State.
  → 3 runs with truth/oracle capture, J-turns scored independently, cells
  including deaths/unscored published.
- **Arm-1 go/no-go, then ≥3 runs** [RIG, paid] — one turn's token count
  first; `LADDER_MODEL=anthropic/...`. → 3 runs per model within budget.
- **Parity table** — v1 DONE 2026-07-15
  (`docs/plans/2026-07-15-first-parity-table.md`, n=1 per Arm-2 model,
  seven binding caveats; summarized in §State). v2 [ANY mech after more
  runs exist] strengthens it per §2-as-amended: interval-worthy n, Arm-1
  column, cost/solved-turn, deaths/unscored/legacy columns; and its
  discriminating rows come from the meta-task realism axis, since v1
  showed the toy ladder saturates at the top.
- **Adversarial honesty probes as a shared sub-battery** [RIG] — spoofed
  "999 passed" read, seeded-red, phantom asks, run against every arm. →
  per-arm probe outcomes beside the table.
- **#63 statistics framework** [ANY design] — becomes live as run count
  grows; per-doctrine-6 the table needs interval estimates before strong
  claims. → agreed test/interval procedure recorded in the rubric.
- **Named possibility, not an arm:** hybrid (frontier harness + llm-orc as
  MCP tools) — composition-as-tool vs composition-as-backend.

### WS-3: Client execution surface (single rounds → budgeted chains)

Capability-map principle (archive §WS-3 carries the full statement): the
serve upgrades any client — workspace-touching actions ride the client
seam; serve-native capabilities (gate, memory) never need the client;
per-client adapters normalize formats; honest refusal at the bottom.

- **#121 grep delegation** [ANY impl; RIG validate] — glob discipline
  generalized: template-built patterns from charset-checked stems,
  deterministic candidate rule, refuse-with-candidates or bounded read
  fan. **Build the deferred round-budget backstop here**
  (`max_rounds`/`rounds_consumed`; the read fan is the first chain not
  bounded by per-step idempotency). → meta-task exit: "how does classify
  decide routing?" answered through the serve via grep→read, grounded.
- **Meta-task rungs** [ANY impl; RIG validate] — (1) recall recovery:
  partial names ("the dispatcher") ground via distinctive-component
  matching (the spike's file-rarity signal); (2) content-grep for
  non-filename symbols (model-proposed pattern + charset-sanitize + grep
  find-or-refuse backstop); (3) `.llm-orc/` dot-dir discovery limitation —
  deferred to the self-hosting apex. Spike evidence:
  `docs/plans/2026-07-14-grep-read-spikes/` (naive content-grep REFUTED;
  filename stem is the clean deterministic signal).
- **#122 edit delegation** [ANY impl; RIG validate] — chain: read-current
  → apply → gate post-edit state → emit edit tool_call. → surgical edit
  ships gated on a real file.
- **#123 multi-file deliverables** [ANY impl; RIG validate] — N writes per
  turn; deliverable contract + form_gate go per-file. Battery turn 7's
  persist-integration shape is the standing miss this converts. → turn 7
  ships both files gated.
- **#124 command-template registry** [ANY impl] — closed builders keyed by
  intent (pytest today, `cargo test` with WS-4). Free-text bash never
  ships without #85. → second registry entry behind the same seam.
- **Client-agnosticism probe** [RIG, cheap] — one battery slice through a
  second OpenAI-compatible client (Aider or Cline), format differences
  absorbed by adapters. → guards against OpenCode overfit.

### WS-4: Language generalization (Rust first, for plexus)

- **#125 Rust gate** [ANY impl for builder/parser; RIG for plexus
  validation] — `cargo test` command builder + verdict parser (the only
  pytest-aware seams are named in the run-delegation design), then
  sandboxed Rust executor + adequacy checker (`#[test]` + `assert_eq!`
  discipline, extended from fixtures as Python's was). → one Rust
  fix-execution probe against plexus: locate, edit, `cargo test`, honest
  verdict.

### WS-5: Long-horizon operation

- **#126 Plan substrate spike + plan-shaped intent** [ANY design+impl; RIG
  battery] — spike (a) server-side plan artifact vs (b) `todowrite`
  mirror in one session; capability-map tilt says burden of proof is on
  the mirror. Then: oversized ask → decompose shape → plan in substrate →
  each wire round advances one step via the chain executor → honest
  progress reporting. Latency budget discipline per chain step (the 780s
  battery cap vs 720s seat budget is the standing datum). → 30+ turn
  battery on a real repo; turn 30 still advancing turn 1's plan with
  honest state.
- **Compaction observation** [RIG, passive] — #82's divergence-classifier
  entry gate; `LLM_ORC_SERVE_WIRE_LOG` watches for the first observed
  OpenCode compaction (the wire has stayed append-only through 30+
  messages).

### WS-6: Memory beyond parity

- **#127 Plexus consumer integration** [RIG design-first] — topology
  decision (spawn `plexus mcp` stdio vs shared SQLite + cursors; design
  around the mutual-subprocess circularity), session-record ingestion with
  tag-at-ingest from classify's routing facts (zero model calls), lens
  queries with `evidence_trail` receipts. Survey facts in the archive
  §WS-6. → cross-session recall probe: a fact from session A answered in
  session B with an evidence trail.
- **#82 remainder** [RIG] — divergence classifier, entry-gated on WS-5's
  compaction observation; prose retrieval beyond the recency tail.

### WS-7: Task-shape generality (the ADR-047 ladder)

- **#128 Elicit-then-build** [ANY impl; RIG validate] — clarifying
  questions on underspecified asks; the honest alternative to guessing
  spec-freedom (the dominant hard-turn failure). → an underspecified
  battery probe elicits, then builds against the answers.
- **#129 Refactor shape** [ANY impl; RIG validate] — behavior-preserving
  by contract: existing tests stay green on post-change state; composes
  read + build + run seams. → a rename/extract probe ships gated.
- **#130 Compose-at-runtime primitive** [ANY impl, evidence-gated] — the
  four named engine gaps; built when a flow demonstrably needs a shape the
  catalog lacks (AS-11). → a task that refuses today runs on a
  runtime-composed shape and passes its gate.
- **Composer ensembles** [design, after #130] — compose from validated
  registry parts, AS-2 validates before registration; the
  capable-model-composed baseline is the benchmark (ADR-047 pillar c). →
  a registered shape survives AS-2 + review with no operator edits.

### WS-9: Platform, hardening, hygiene

In dependency order: **#85** sandbox hardening [ANY impl] (container/
seccomp/rlimits; gates #124's bash widening and untrusted deployment) ·
**#84 remainder** [ANY impl] (adversarial harness vs a live
non-cooperative builder; ADR-048) · **#93** hot-path caching + I/O off the
event loop [ANY impl] (prerequisite for multi-session and WS-5 length) ·
**#90** llama.cpp backend [ANY impl; RIG validate] (drop the Ollama
process dependency; #64 folded in) · **#132** BitNet/bitnet.cpp
investigation [ANY research] (CPU seat; related to #90 — mainline
llama.cpp does not reliably load the 2B4T GGUF) · **#106** single home for
serving shapes [ANY impl] (regression pin holds meanwhile) · **#95**
dead-surface sweep [ANY mech, Haiku-grade].

Off-path backlog: #80 (meta-workflow viz) · #65 (artifact relocation) ·
#30/#66 (research). Closed 2026-07-11: #31 #78 #79 #64.

## Sequencing

- **Now (parallel, different environments):**
  - ANY/REMOTE: #133 and #134 (the dishonest classes — highest leverage,
    small, environment-agnostic), then #110.
  - REMOTE: Arm-2 capture → adapter → 3 runs.
  - RIG (next rig session): re-validate #133/#134 live with independent
    re-scoring; Arm-1 go/no-go; then the first parity table.
- **Next:** WS-3 #121 grep + meta-task rungs → meta-task exit gate
  (llm-orc half) → WS-4 #125 in parallel → plexus half → #122/#123/#124.
- **Then:** WS-5 and WS-6 as parallel arcs (WS-5's compaction observation
  feeds WS-6's divergence classifier; WS-6 ingestion needs nothing from
  WS-5).
- **Throughout:** WS-7 entries land inside WS-2/WS-3 arcs; #130 waits for
  demonstrated need; WS-9 items slot in where named as prerequisites;
  #119 when a rig session has battery budget.
- **Apex (self-hosting: the serve improving its own repo):** enter when
  the meta-task rung holds; judged by shadow comparison (an agent driving
  the serve through OpenCode judges its decisions against its own).

## Delegation contract (updated for non-Fable sessions)

Every arc: short design doc (`docs/plans/YYYY-MM-DD-*.md`) → TDD
implementation → live real-OpenCode validation at the earliest runnable
point (RIG; queued explicitly if the implementing session lacks the rig)
→ ladder rerun + a row appended to the archive's trajectory table →
**author-independent adversarial review with an explicit wrong-accept hunt
before merge** (the review record is five-for-five on finding real
blockers the author missed).

Session roles:
- **Lead (Opus-class):** designs, reviews, sequencing decisions, this
  document. On entry: read §State, §Doctrine, and the card being worked —
  not the archive.
- **Implementer (Sonnet-class):** one task card per arc, TDD, hermetic
  suite green, PR notes any queued RIG validation.
- **Mechanical (Haiku-class):** #95-grade sweeps, doc syncs, battery
  bookkeeping, table updates from existing records.
- **Scoring/review independence (doctrine 1):** J-scores and review
  APPROVEs come from a session or agent that did not author the work; a
  fresh remote session is naturally independent. The frozen rubric
  governs; corrections are amended in the rubric, never edited away.

## Standing constraints

- 32GB rig is the permanent target; interactive latency is first-class.
- Local-first defaults; hosted seats are operator opt-in, never tracked
  (`*.local.yaml`), each carried in the buy-back ledger.
- Real-client validation at the earliest runnable point; every capability
  stage lands with its ladder rerun and a trajectory row (archive table).
- Deterministic control; model judgment only inside bounded, closed-set,
  gate-backstopped decisions; honesty-critical paths fail closed.
- Ground truth is the workspace; independent scoring for judgment claims.

## Issue index

**WS-2:** #133 #134 #110 #117 #119 #114 · **WS-8:** #131 #63 · **WS-3:**
#121 #122 #123 #124 · **WS-4:** #125 · **WS-5:** #126 · **WS-6:** #127 +
#82 remainder · **WS-7:** #128 #129 #130 · **WS-9:** #85 #84 #93 #90 #132
#106 #95. Off-path: #80 #65 #30 #66. The shipped-issue ledger and release
mapping live in the archive.

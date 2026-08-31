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

## State (2026-08-30, evening rewrite)

### Next up

The gate is empty and the handoff's executable queue is EXHAUSTED: both
long arcs, every rule-13 small fix (#173, #175 env slice, #178-in-arc,
#179), the #172/#176 rework, and #168's live validation row are merged.
What remains forks three ways, and the fork is the practitioner's call:

1. **Push** — 11 merge units close eight GitHub issues on landing; CI's
   first look at the drift checker; `gh run watch` after.
2. **Arc-sized trio** — #174 (dead seat on non-build routes ships the
   engine envelope as the ANSWER; nearest to #168's just-validated
   family), #177 (unify the three file-vs-inline classifiers; changes
   what EXECUTES), #171 (the deliverable must participate; the general
   fix #166/#169/#173 closed slices of).
3. **Remaining live rows** — #166 #169 #173 #172 #176 #175; the
   fault-injection template is recorded in the #168 gate record, but the
   re-fix shapes need a seat-level injection design first.

Nothing is pushed (`git log --oneline origin/main..main` for the count; 10
merge units). The push — and the GitHub issue-closing that rides on it —
awaits the practitioner's explicit go. CI has never seen the drift checker;
#179 fixed the latent red that would have surfaced on first push.

**#172 #176 merged** after four review rounds (first review: four blockers,
including the #176 heuristic converting wrong-REJECT into wrong-ACCEPT;
round 2: the fix went structural — the runner's real namespace decides, not
AST guesses; round 3 caught the scan wrong-REJECTing produced-code classes
with `test_*`-named production methods; round 4 confirmed the code-name
snapshot fix on fresh probes). The merge into main was semantic — #168
rewrote the same runner functions — and every emitted string kept #168's
`_wire_safe` discipline; cross-arc probes on the merged tree confirm the
leak refusal and both load-error paths stay path-free. The round-4
confirmation was lead-run on fresh inputs after the reviewer's session hit
its model limit mid-round; that same reviewer had already pre-verified the
identical patch on its 46-shape matrix in round 3.

**Merged to local main this session, each with an author-independent
APPROVE after a wrong-accept hunt:**

- **#163** — an undigestable script is never cached. The confirmation round
  found no wrong-accept (~25 fresh probes; all 15 named instruments went red
  under their claimed mutants) and blocked once on two doc claims that did
  not derive; fixed and re-verified by the same reviewer.
- **#168** — a refusal reason names no absolute path; eight rounds and a
  three-part confirmation. The confirmation found the round-7 fix regressed
  round 5 (ndiff context lines are indented, so ordinary assertEqual diffs
  lost their evidence), so **#178 was taken in-arc**: the traceback parser
  is deleted and a `TestResult` subclass keeps the live exception at capture
  time. Found on the way: `addSubTest` does not delegate, so subtest
  failures were silently dropped — closed and pinned. Round 8b's residuals:
  a raising `__str__` no longer kills the runner child (guard in-arc);
  **#180 filed** (the report is unbounded and produced code controls the
  bound — 76KB measured; #114/#175 family); the vacuous-pass shapes
  (all-skipped suite reports all-passed) recorded on #84.
- **#173** — an inert re-fix candidate is never accepted, at source. The
  first review found the round-1 guard caught 2 of a 15-member class
  (`pass` alone — the injected smoke test's own body — still clobbered);
  reworked to a closed AST whitelist (structure and constants only), which
  supersedes #169's one-character pin explicitly. End-to-end pin included;
  zero wrong-rejects across 16 real-code shapes.
- **#179** (new, filed and fixed this session) — the doc-drift check knew
  zero names inside agent worktrees: `_SKIP` matched "worktrees" against
  absolute path parts, so rule 19's instrument was blind exactly where
  delegated agents run, and 4 of its own tests failed there. Relative parts
  close it; the breadth pin now names a tracked benchmarks instrument
  instead of a gitignored session artifact.

Prior handoff's merges (#166 #169 #170, rules 13-19) unchanged beneath.

### Process

Every first review this session found real blockers — the record is now
seven-for-seven — and both long-arc confirmations found their blocker by
RE-PROBING earlier findings with fresh inputs, not by re-reading. Rule 4's
re-verify clause is the one earning its keep. The #168 confirmation is the
sharpest datum: the round-7 fix traded one defect for a regression of round
5 on an equally ordinary input, which is what finally forced the structural
fix rule 18 had been pointing at since round 6.

### Open, and which are small

- **#175** — the env-scrub slice is MERGED (produced code runs under an
  empty environment; the child's true census — PEP 538's LC_CTYPE plus
  macOS's UID-bearing __CF_USER_TEXT_ENCODING — is pinned, and the
  pwd-route username is pinned as the accepted bound). The issue stays
  open for the closed-report-vocabulary half, riding #180/#142.
- **#180** — filed this session from #168 round 8b: the accept report is
  unbounded and unittest itself teaches the model to inflate it
  (`maxDiff = None`). #114/#175 family.
- **#171 #174 #177** — arc-sized, unchanged from the morning handoff.
- Untouched: **#165**, **#155** Arcs B/C, **#161/#162**, **#149 #151**.
- Still blocked on the practitioner: **#167** (volume paid runs) and
  **#141** (CLAUDE.md-confound spike's None condition).

### Queued, not skipped

**#168's live row LANDED** (dogfood entry 10; record
`docs/plans/2026-08-30-168-live-gate/`): a live-injected resolve crash
whose message named an absolute path reached the wire as
`(resolve: exited non-zero, status 1)` — both legs, direct wire and
`opencode run`, zero tool_use, nothing written. Found on the way: entry
9's PATH-broken precondition is EXTINCT on this main (#154), so that
replay now builds normally; live fault injection with a
confirmed-live mutant is the template for future rows. Also: a stale
serve from 08-17 was found still on :8765 — restart the serve before
every gate, a long-running serve validates the wrong code.

Still owed: rows for **#166 #169 #173** (re-fix/build refusal shapes),
**#172 #176** (library source, gate participants), **#175** (env
scrub) — each changes client-visible behaviour.

## Timeline

Done:

- [x] v0.18.0 — agentic serving backend (declarative ensemble behind OpenCode)
- [x] v0.18.1–0.18.7 — session record #99, TDD retry #100, write-tests #98, client reads (#83), per-test gate isolation
- [x] v0.18.8–0.18.12 — run delegation (#83), fenced block grammar, gate repairs, discovery glob (#83), fix-execution (#115), #107
- [x] v0.18.13 — convergent fix (#117 rungs 1.5+2), grounded explain (#118)
- [x] v0.18.14 — chain executor (#120), deep recall (#82 core)
- [x] Meta-task slice 1 — bare-symbol glob→read grounded explain
- [x] WS-8 instrument (#131 Arcs A–D) — battery, hidden oracles, frozen rubric, hashed manifests; 5-round review gate
- [x] Arm-0 column n=3; first parity table published
- [x] #133 #134 — honesty classes closed, live-validated at 0 dishonest (run 5)
- [x] Arm-2 adapter + automatic scoring (#131); Haiku run 2 scored (2 dishonest — frontier n=1 ceiling broken)
- [x] Battery precondition guards; CI fixes (twine metadata, pyasn1); dogfood channel (`docs/dogfood-log.md`)

Remaining, in order:

- [x] #131 — Arm-2 runs at n=3 per model, all independently J-scored (Haiku 35/39, 4 dishonest; Sonnet 39/39, 0)
- [x] #131 — Arm-1 GO'd + n=3 per model, independently J-scored (Haiku 38/39, 0 dishonest; Sonnet 39/39, 0); #147 filed
- [x] #146 — v0.18.15 RELEASED 2026-08-13 (PyPI + Homebrew green)
- [x] #148 — truncated-listing refuse merged + live-validated (3 review
  rounds; run-6 validation row); #149 filed (client-side flank)
- [x] #143 — CLOSED as an honest miss: component-subset refuted at
  class level; model gate refuted empirically (pre-registered bar,
  8b+14b, all variants); reopen rides #119 with the committed spike
- [x] #139 — context curve measured (flat through 32K recall / 24K
  synthesis; 4KB cap defensible; latency is the binding constraint)
- [x] #145 — repo-scale reads merged + live-validated (96KB cap,
  token-denominated read budget, runtime truncation backstop, #150
  fixed; five review rounds; dogfood entry 1 converted). classify.py
  refuses over-budget by design → chunked reads deferred; #151 open
- [x] #144 slice — serve-native dot-dir self-reference merged + live-
  validated (grounded self-reference for budget-fitting scripts; the
  literal classify-grounded exit stays open on the whale, riding
  #106/#151/chunked reads); #152 fixed+merged (routing fails closed)
- [x] #121 slice A — content-grep rung merged + live-validated (exit
  gate met: grep→menu→pick→AST-confirmed grounding); coverage bounds
  ride #153 (offset reads) and the truncated-listing trigger gate;
  #153 filed+fixed (client 50KB read cap refuses honestly)
- [x] #63 slice + #138 INSTRUMENT — statistics (Wilson/Fisher) and the
  volume-ladder instrument built, merged, and calibrated on arm 0
- [x] v0.19.0 released — script cache keyed on BYTES and shipped disabled
  (#160), instruments in the gate (#156), pipeline positive-completeness
  (#155 Arc A), plus #152 #154 #157 #158 #159 #164
- [ ] #167 volume paid runs (r=8 per level; awaiting practitioner go on
  cost) and #141 (awaiting go on the None condition), then parity table
  v2. #138 is CLOSED — it tracked the INSTRUMENT, which shipped; #167 is
  the spike itself.
- [ ] #126 — long-horizon 30-turn battery (#136 #137 feed the design)
- [ ] #140 memory spike (#139 closed: context curve measured); #127
  plexus substrate; #82 remainder (cross-session)
- [ ] #128 #129 #130 — task shapes toward compose-at-runtime
- [ ] #125 — Rust gate; #119 #135 — seat ladder on-signal
- [ ] #85 #84 #90 #93 #95 #106 #110 #114 #132 #142 — platform hardening as gates demand
- [x] #166 #169 #170 merged (empty deliverable never written; empty re-fix
  candidate never accepted; `make test` independent of the working
  directory), plus loop-protocol 13-19 and the doc-drift check
- [x] #163 #168 merged after confirmation rounds (#178 taken in-arc:
  the parser is deleted); #173 merged (inert class closed 15/15); #179
  filed+fixed+merged (drift check works in worktrees)
- [x] #172 #176 merged after four rounds (runner-namespace judgment, library
  source resolution rebuilt: explicit env path trusted on existence, cwd and
  packaged candidates content-gated non-empty, `("local","")` never escapes)
- [x] #175 env-scrub slice merged (empty child env, census pinned, bounds
  named); the vocabulary half stays open on #175, riding #180/#142
- [ ] #171 #174 #177 — the general fixes the two long arcs circled
- [ ] #161 #162 #165 — script-cache purity/imports and the -n auto flake;
  #155 Arcs B/C remainder
- [ ] North star: parity on real work, honesty column held at zero

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

## Epics

Issue lists live in GitHub: `gh issue list --label epic:<name>`. One line
per epic here; detail on the issues.

### epic:ws2-honesty — CLOSED 2026-08-12; hold the property
Zero dishonest under independent scoring, validated live. Watch: #140
(staleness cascades). Reopen only on a new independently-confirmed
dishonest outcome.

### epic:ws8-parity — the comparison IS the product claim
- [x] Instrument + Arm-0 column + parity table v1 + Arm-2 auto-scoring (#131)
- [x] #131 Arm-2 n=3; Arm-1 go/no-go + runs n=3 (all columns complete)
- [ ] #147 `_PASS_CLAIM_RE` false-positive family (three refuted captures)
- [ ] #141 CLAUDE.md confound · #167 volume-scaling paid runs
  (successor to the closed #138, which tracked the instrument) · #63
  statistics
- [ ] Parity table v2 (realism rows; interval estimates)

### epic:ws3-client-surface — the serve upgrades any client
- [x] read/run/discovery delegation, fix-execution, chain executor, meta-task slice 1
- [x] #143 closed (honest miss; reopen rides #119) · #145 repo-scale reads merged+live · #148 #150 read-seam hardening
- [x] #144 slice — serve-native dot-dir self-reference merged+live (grounded exit rides #106/#151/chunked reads)
- [x] #121 slice A — content-grep rung merged+live (exit gate met; bounds ride #153)
- [x] #153 offset reads merged+live (50–96KB grounding restored; the
  #121 bound converted); remaining flanks recorded on the issue
- [ ] #149 client-side truncation flank
- [ ] #122 edit delegation · #123 multi-file · #124 command registry · #117 fix-completion tail

### epic:ws5-long-horizon
- [ ] #126 30-turn battery + plan substrate (#136 #137 feed design)

### epic:ws6-memory
- [x] #139 context curve (flat through 32K recall / 24K synthesis; latency binds)
- [ ] #127 plexus integration · #82 remainder (cross-session record)

### epic:ws7-task-shapes
- [ ] #128 elicit-then-build · #129 refactor shape · #130 compose-at-runtime primitive

### epic:ws4-language
- [ ] #125 Rust gate (cargo runner, sandboxed executor, adequacy)

### epic:seat-ladder
- [ ] #119 escalation framework · #135 ori/eval A/B harness spike

### epic:ws9-platform
- [x] #146 v0.18.15 released
- [x] #152 fail-closed routing merged (readability-gated decisions; the misfire class refuses)
- [x] #154 interpreter PATH fragility fixed (.py runs under llm-orc's own interpreter; bare-PATH suite 50 failures -> 0)
- [x] #157 script-agent timeouts fixed (one resolver for inner and outer bounds; six nodes declare their own)
- [x] #158 script agents off the event loop (4.15s -> 1.03s; duplicate outer timeout retired)
- [x] #159 failure envelopes no longer cached (two-clause predicate; the corpus emits no boolean success)
- [x] #160 cache identity is the script's BYTES, and the cache ships DISABLED (two of six primitives are impure; a hit elided a write) — v0.19.0
- [x] #156 the 511 measurement instruments run in `make test` and CI (a regression in them used to corrupt evidence without failing a build)
- [x] #164 script-agent-architecture documents the cache that exists (the per-agent `cache:` key it showed is rejected by `extra="forbid"`)
- [x] #166 an empty build deliverable is never a client write (the fault was
  a LIVE seat with an empty artifact, not the dead seat the issue named)
- [x] #169 an empty re-fix candidate is never accepted, at source
- [x] #170 `make test` no longer depends on the ambient working directory
- [x] #155 Arc A — a node that cannot READ its input refuses (crashed shape/form_gate finished as an empty success); Arcs B/C still open
- [x] #163 an undigestable script is never cached (7 rounds + confirmation)
- [x] #168 a refusal reason names no absolute path (8 rounds + confirmation;
  #178 taken in-arc — the TestCase branch no longer parses tracebacks)
- [x] #173 an inert re-fix candidate is never accepted (closed AST whitelist)
- [x] #179 the doc-drift check knows its names inside agent worktrees
- [x] #172 #176 — an empty submodule is not a library; the gate's verdict
  comes only from tests the runner executes (four rounds)
- [ ] #151 runtime-window detector remainder · #155 Arcs B/C · #174 a dead seat ships the engine envelope as the answer · #175 vocabulary half (env route CLOSED; path-free literals still reach the wire) · #177 three file-vs-inline classifiers · #180 the accept report is unbounded · #161 cache purity · #162 cache misses imports · #165 `-n auto` flake · #85 sandbox hardening · #84 gate adversarial harness · #90 llama.cpp · #93 hot path · #95 dead surface · #106 shape home · #110 artifact quality · #114 trace cap · #132 BitNet · #142 reject templates

### epic:off-path
#80 #65 #30 #66 — parked, not on the north-star path.

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

Superseded by epic labels: `gh issue list --label epic:<name>`. Closed
2026-07-11: #31 #78 #79 #64. Closed since: #83 #98 #99 #100 #104 #105
#107–#109 #111–#113 #115 #116 #118 #120 #133 #134 #138 #139 #145 #152
#153 #154 #156 #157 #158 #159 #160 #164.

**#163 #166 #168 #169 #170 #172 #173 #176 #178 #179 are merged on local
main and still OPEN on GitHub**, because nothing is pushed. They close when
the push lands, not before — the roadmap said "closed" here first, which is
the kind of claim rule 15 exists for. #175 is merged in part (env slice)
and stays open by design for its vocabulary half.

Filed 2026-08-30 from review findings, all measured, none previously
tracked: #171 #172 #173 #174 #175 #176 #177 #178, then #179 (fixed+merged)
and #180 (open) in the evening session. #172 and #176 merged after four
review rounds.

Two closed issues gate work that is tracked elsewhere: **#138**
(instrument shipped; the paid runs are #167) and **#139** (curve
measured; #140 carries the remainder).

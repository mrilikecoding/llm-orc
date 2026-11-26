# Serving Roadmap — to the North Star

Revamped 2026-07-17 for execution by non-frontier sessions (hosted-large-class lead,
hosted-mid-class implementers, hosted-small-class mechanical). The prior roadmap —
including the full trajectory table, the Arc D correction history, and the
2026-07-09 path items — is frozen verbatim in
`docs/serving-roadmap-archive-2026-07-17.md`; treat it as provenance, not
as the plan. This document is the plan. Its **State** section is rewritten
(not appended) at each update; history lives in the archive and in git.

## North star

**Full model parity through composition** (see `docs/serving.md`): llm-orc
served agentically behind OpenCode should be as functional as the frontier harness
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

## State (2026-11-25 handoff)

**Ranked index:** the GitHub project "llm-orc kanban"
(https://github.com/users/mrilikecoding/projects/2). THIS DOCUMENT
GOVERNS; the board is its index. Moving a card to Done closes its issue
(a board automation), so a card goes to Done when the work is pushed,
and sits at "In review" while it is merged only locally. Pushing is
gated per push. The State section as it stood before this rewrite,
2026-09-11 through 2026-11-25, is frozen in
`docs/serving-roadmap-archive-2026-11-25.md`, including the findings
each remote-delegation arc's reviews deferred.

### Where things stand

- **Released:** v0.25.4 (2026-11-25; PyPI, GitHub release, tap formula;
  verified by a clean install). The remote host runs 0.25.3.
- **On local main, not pushed** (`f61b08d2`..`d8217cbb`, suite 6174,
  each fix independently reviewed):
  - #181: the build envelope ships the code the executor ran, and
    accepts only with that code in hand.
  - #165: every test runs from an empty directory with the router URL
    on a closed port, so the suite no longer depends on the checkout's
    untracked profiles, its library submodule, or what listens on :8080.
  - #189: the stale hand-written MCP tool list is gone; API-key
    OpenAI-compatible profiles now get `options`, `response_format` and
    the agent's `timeout_seconds` (they were dropped, so a hosted seat
    read to the 180 s default); `call_tool` dispatches every registered
    tool.
  - #191: `update_ensemble` applies `add_agents` and `remove_agents`
    (it used to write a backup and report an update it had not made),
    validates the result before an atomic write, and keeps the leading
    comment block. Every handler that turns a request-supplied name
    into a path now stays inside its directory: ensemble, profile and
    script CRUD, `get_script` and `test_script`, artifact delete,
    cleanup and reads, promote and demote, `library_copy`. Before this
    a crafted name could run a script above the scripts directory or
    remove a directory beside the artifacts base. The ensemble CRUD
    routes answer a refusal with 400 or 404, not 500.
  - #106: delete and update refuse a symlinked ensemble, so a
    project-scope delete can no longer unlink a serving shape.
- **Open issues:** 40. Today's session folded eight issues into four by
  structure (#175, #149, #161, #189), closed eleven as shipped or not
  planned, and released #199, #147 and #204. The record is on the
  issues.

### Standing directives

The target is a daily driver on existing repos, validated in real
OpenCode sessions. Free-first: local seats by default, paid runs only
in bounded slots with an estimate first. Delegate implementation and
review to cheaper subagents. The remote host runs releases, never a
checkout, and deploys on the practitioner's go. This repo names no
machine and no vendor tooling.

### Next tranche: the daily driver on existing repos

Confirmed by the practitioner 2026-11-25. The last
measurements: the 7-turn existing-repo probe went 1 of 7 for the serve
against 7 of 7 for both comparators (2026-09-11, on Ollama); T1 went 0
of 10 correct on llama-server against 3 of 8 on Ollama (2026-09-27),
every miss traced to the 8B test-writer (tests with no import,
self-contradicting expectations). Neither has been run since.

1. **Re-baseline (RIG, free).** T1 at r=5 and the 7-turn probe
   (`docs/plans/2026-09-11-daily-driver-probe/`) on current main with
   the llama-server backend, one row each. This is also #90's owed
   regression gate.
2. **Diagnose T1 before touching the build round.** Every existing-repo
   slice rides on the build round, and it delivered nothing on this
   backend. Find what changed between backends for the test-writer seat
   (template, thinking, quantization), then choose the lever by
   doctrine 2 and 3: structure first, a seat escalation (#119) only on
   a measured signal.
3. **Then, in the order the baseline decides:** #182's remaining slices
   (B, the edit contract: a build whose target exists ships the whole
   post-edit file and runs the existing suite; C is #123, code and
   tests in one turn; E is #124), #122 edit delegation, and #183 A and
   C (per-node elapsed in the trace, declared context size), since a
   gated build turn takes about five minutes.

The exit instrument is #182's: probe turns 1, 2, 3 and 6 convert to
correct workspace truth while turn 7 and the 13-turn ladder hold.
Nothing is planned past step 1's result.

Not in this tranche, picked up when the area is touched: #175 (the
refusal reason as a closed type), #149 (truncation the serve cannot
see), #155 (one known gap), #161 (script cache contract), #93, #110,
and the one-service work on #191 and #189.

### Gated on the practitioner

- Pushing local main, and deploying 0.25.4 to the remote host.
- The paid parity runs: #167 (volume ladder) and #141 (the
  instruction-file confound), then parity table v2.
- The remote host's certificate expires 2026-12-15 and its renewal
  daemon has been failing; the fix is in the homelab repo.
- The library submodule's invalid example ensembles
  (`neon-shadows-detective` carries a key the schema refuses).

### Known and left open

Recorded on the issues named; none is in the next tranche.

- #205: `invoke` with `input_file` reads any path the serve can read,
  and a child ensemble reference can load an ensemble-shaped file above
  the search directories. Both are documented behavior, so they wait on
  the access-level ruling. A serve already runs the scripts a client
  ships, so the containment above gives a typo'd name a refusal; it is
  not an access boundary.
- #191: profile CRUD, ensemble validate, runnable, execute and
  preflight, script get and test, and the artifact routes still answer
  a refused call with 500. `MCPServer.call_tool` has no production
  caller.
- #165: three timing tests failed once each under load on 2026-11-25
  and passed alone; the original `test_artifact_manager_integration`
  flake is still unexplained.
- A symlinked script category that points outside the scripts
  directory is refused; a symlinked script file that points outside
  still runs.

### Owed live rows

#166 #169 #173 #171 (a constructed non-participating shape via fault
injection), #172 #176, #175, #181 (a real build turn whose deliverable needs an
injected workspace import), and `update_ensemble` through a real serve.
#90's is step 1 above.

### Operating notes

Each was paid for; the stories are in the archive.

- `publish.yml` publishes the version in `pyproject.toml` on every push
  to main. Push the work at the old version, wait for CI, then push the
  release commit.
- A release is verified by a clean install from PyPI, never from the
  checkout.
- After a fix wave check the suite's wall time and `--durations`, not
  only the passed count. The suite is about two minutes in parallel
  (`make test`).
- A suite that is green in a subagent's clean worktree is not yet green
  on main. Merge one branch at a time with `--squash`, and run
  `make test` and `make lint` on main before committing.
- Run the real thing before calling a lifecycle feature done: a serve
  on its own ports with its own config and state dirs.
- Every first review round has found something the author missed, and
  fix waves introduce defects of their own. A fix to an instrument, a
  gate or a trust boundary gets an author-independent review, and so
  does the fix to its findings.
- A lenient rule in an instrument or a gate needs a captured case
  behind it. Unknown inputs must fall to the loud side.
- Ops: one serve per tree, restarted before every gate; the ladder is
  about 17 minutes and the probe about 10; hosted-mid implementers and
  reviewers, the lead reads every final diff.

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
- [x] Arm-2 adapter + automatic scoring (#131); hosted-small run 2 scored (2 dishonest — frontier n=1 ceiling broken)
- [x] Battery precondition guards; CI fixes (twine metadata, pyasn1); dogfood channel (`docs/dogfood-log.md`)

Remaining, in order:

- [x] #90 — Ollama dropped; the serve owns a llama-server router (2026-09-16)
- [x] #131 — Arm-2 runs at n=3 per model, all independently J-scored (hosted-small 35/39, 4 dishonest; hosted-mid 39/39, 0)
- [x] #131 — Arm-1 GO'd + n=3 per model, independently J-scored (hosted-small 38/39, 0 dishonest; hosted-mid 39/39, 0); #147 filed
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
- [ ] #85 #84 #93 #106 #110 #114 #132 #175 #189 — platform hardening as gates demand
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
- [x] #174 merged (a dead seat refuses honestly on every route; dead-seat
  refusal outranks accept/validity; stderr retained head+tail capped)
- [x] #177 merged (one resolve-and-classify observation; bare CWD refs
  anchored so PATH cannot re-resolve; identity names bytes AND location)
- [x] #171 merged (ablation control + public-binding re-fix surface; four rounds)
- [x] #182 D + B-1 merged (discovery before an unnamed build; edits keep the prior surface); #183 B measured (coder keeps thinking)
- [x] Arc 1 workspace-aware routing (#185) and arc 2 the sandbox mirrors the workspace (#184, #182 A) merged 2026-09-12
- [ ] #123 code+tests per turn (+ tests destination); #122 edit delegation; verified acceptance in the workspace; #183 A/C; #181
- [ ] #161 (holds #162) #165 — script-cache purity/imports and the -n auto flake;
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
11. **Pin the harm, not the mechanism.** Three of #177's four blockers
    were green pins measuring the code path a fix had just edited (a
    trap pin on the one shape where it held for free, an
    `assert_not_called` on the edited branch, identity pins checking
    only the digest suffix) while the defect ran a PATH impostor or
    crossed a cache. A pin must assert the OUTCOME — which program ran,
    which output crossed, what reached the wire — and go red under a
    mutant that reintroduces the defect it was written for.

## Environments (tag every task)

- **RIG** — the 32GB rig (llama-server backend) with OpenCode: live batteries, Arm-0/Arm-1
  runs, seat A/Bs, latency data, plexus (local sibling repo). Ops notes:
  batteries run detached (nohup + disown, Monitor tail); `opencode run`
  wedges under the agent Bash sandbox (see memory `opencode-run-wedge`);
  cooling headroom between batteries.
- **ANY** — any session including remote containers: hermetic TDD against
  the full suite (`uv` + Python 3.11 suffice), design docs, scorer/oracle/
  adapter code, reviews, doc work. RIG-tagged validation of ANY-developed
  work is queued, not skipped: the PR says "needs rig battery" and the next
  rig session runs it.
- **REMOTE** — a remote frontier-harness session specifically: **Arm-2 battery
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
- [ ] #141 instruction-file confound · #167 volume-scaling paid runs
  (successor to the closed #138, which tracked the instrument); #63
  and #131 closed 2026-11-25, their shipped scope complete
- [ ] Parity table v2 (realism rows; interval estimates)

### epic:ws3-client-surface — the serve upgrades any client
- [x] read/run/discovery delegation, fix-execution, chain executor, meta-task slice 1
- [x] #143 closed (honest miss; reopen rides #119) · #145 repo-scale reads merged+live · #148 #150 read-seam hardening
- [x] #144 slice — serve-native dot-dir self-reference merged+live (grounded exit rides #106/#151/chunked reads)
- [x] #121 slice A — content-grep rung merged+live (exit gate met; bounds ride #153)
- [x] #153 offset reads merged+live (50–96KB grounding restored; the
  #121 bound converted); remaining flanks recorded on the issue
- [ ] #149 truncation the serve cannot see (holds #151 and the #153 residuals)
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
- [ ] #155 Arcs B/C · #175 the refusal reason is a free string (holds #142 #180; env route CLOSED) · #161 script cache contract (holds #162) · #165 `-n auto` flake and test hermeticity · #85 sandbox hardening · #205 spike: access levels for a serve with clients it does not trust · #84 gate adversarial harness · #93 hot path · #106 shape home · #110 artifact quality · #114 trace cap · #132 BitNet · #189 code footprint (holds #95 #190 #195)

### epic:off-path
#80 #65 #30 #66 — closed 2026-11-25 as not planned.

## Delegation contract (updated for non-frontier sessions)

Every arc: short design doc (`docs/plans/YYYY-MM-DD-*.md`) → TDD
implementation → live real-OpenCode validation at the earliest runnable
point (RIG; queued explicitly if the implementing session lacks the rig)
→ ladder rerun + a row appended to the archive's trajectory table →
**author-independent adversarial review with an explicit wrong-accept hunt
before merge** (the review record is five-for-five on finding real
blockers the author missed).

Session roles:
- **Lead (hosted-large-class):** designs, reviews, sequencing decisions, this
  document. On entry: read §State, §Doctrine, and the card being worked —
  not the archive.
- **Implementer (hosted-mid-class):** one task card per arc, TDD, hermetic
  suite green, PR notes any queued RIG validation.
- **Mechanical (hosted-small-class):** #95-grade sweeps, doc syncs, battery
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

Filed 2026-08-30 from review findings, all measured, none previously
tracked: #171 #172 #173 #174 #175 #176 #177 #178, then #179 (fixed+merged)
and #180 (open) in the evening session. #172 and #176 merged after four
review rounds.

Two closed issues gate work that is tracked elsewhere: **#138**
(instrument shipped; the paid runs are #167) and **#139** (curve
measured; #140 carries the remainder).

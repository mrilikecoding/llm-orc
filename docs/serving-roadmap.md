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

## State (2026-09-16 handoff)

**Ranked index:** the GitHub project "llm-orc kanban"
(https://github.com/users/mrilikecoding/projects/2). THIS DOCUMENT
GOVERNS; the board is its index. Board "Done" = merged on main. Pushed
2026-09-16 on the practitioner's go: the 166-commit backlog, then PR
#186 (#90) merged and released as v0.20.0; CI is green on main again
(two pre-existing test failures and a whole-tree format check fixed on
the way). Pushing remains gated per push.

**Practitioner directives (2026-09-16):** drop Ollama (decided, not
debated further); the deployment target is a serve running natively on
ng-mini, proxied through a Dokku app, reachable on the tailnet from any
client; llm-orc must be deployable without coupling to additional
applications (#90's packaging goal). Standing directives from 2026-09-11
hold (daily driver on existing repos; real OpenCode sessions; Go spend
and paid comparison runs within reason; cheaper subagents; meter usage).

### Merged on local main this session

- **#90 — Ollama dropped; llama-server router owned by the serve**
  (PR #186, released v0.20.0; suite 4322 passed, lint 0, CI green). Evidence on the issue (spike table 2026-09-16). Shape:
  `OpenAICompatibleModel` carries `options` (`think` lifted into
  `chat_template_kwargs.enable_thinking`, 60x on qwen3), sampling
  passthrough, `response_format`, and llama-server `timings` under the
  `prompt_eval_count`/`eval_count` keys the truncation backstop reads;
  provider `llama-server` (local default, legacy fallback target;
  `LLAMA_SERVER_URL`); the serve renders `.llm-orc/llama-server.ini`
  from every llama-server profile (one section per model, `hf_repo:`
  source, `c = 40960` = `turn_trace.WINDOW`, one resident model) and
  supervises one router (`--no-backend` opts out; a router that exits
  before listening surfaces its stderr tail; SIGTERM to the serve stops
  the router — uvicorn re-raises captured signals after restoring
  handlers, so a `finally` never runs); `GET /api/models`,
  `POST /api/models/{name}/pull` (blocks until the router's real status);
  provider status / runnability / promotion read the router; the agent
  config field is `response_format` (was `ollama_format`); profiles are
  `local-qwen3-*`; model names carry no colon (the router rewrites
  `name:tag`) and no slash (raw cache entries). Verified live on the
  laptop: serve owning a real router, five preset models listed, pull
  blocked ~4 s to `loaded`, factory completion with timings, router gone
  on SIGTERM. Tests may not launch the real binary (autouse guard, shown
  red on two leaking serve tests before the fix).
- **Deploy scaffolding:** Dokku app `llm-orc` deployed
  (`~/Development/llm-orc-proxy`, nginx → `host.lima.internal:8765`;
  edge nginx set to 3600 s / buffering off / 50 m body);
  http://llm-orc.homelab.nate.green answers 502 until the serve exists on
  ng-mini. `deploy/ng-mini/` carries the launchd unit and setup notes.

### Evidence carried forward (2026-09-12)

Comparator row on the existing-repo probe: Go `qwen3.8-max` 7/7 ($0.27),
Sonnet 5 7/7 ($0.38) vs the serve 1/7. Ladder runs 7-8 on the merged
arcs: 10/13, 8/13; T1 decides the ladder (3 correct / 2 broken / 3 refused
over eight runs; refusals pre-date the arcs). Neither has been re-run on
the llama-server backend yet — that is the regression gate below.

### Next up (in order)

1. **Serve is up on ng-mini (2026-09-16 afternoon).** launchd agent,
   v0.20.0 from origin main, llama.cpp b10964 x64 binary (no Intel brew
   bottles; see `deploy/ng-mini/README.md`). http://llm-orc.homelab.nate.green
   answers 200; four tiers pulled over the tailnet; acceptance 1 and 2 of
   the handoff pass (tool call parsed through the tailnet URL). Measured
   on the box (i7-8700B, CPU-only): qwen3-8b 23 prompt tok/s, 4-6 gen
   tok/s; the trivial write_file turn took 560 s end to end because the
   serving ensemble ran classify + build-gated round + tests at that
   speed, and classified "create a text file" as a python_module build.
   Both numbers feed the gate below. HTTPS is on (practitioner's go):
   the homelab wildcard cert had expired 2026-08-23; renewed, valid to
   2026-12-15, `https://llm-orc.homelab.nate.green` answers 200 and http
   redirects. The weekly renewal daemon has been failing (root PATH lacks
   `certbot`), a homelab-repo fix owed to the practitioner before
   December. Full record: `docs/plans/2026-09-16-ng-mini-remote-serve-handoff.md`.
   **#194 merged on local main (2026-09-16 evening):** the serve exposes
   the full MCP tool set at `/mcp` (streamable HTTP, one `OrchestraService`
   behind REST, `/v1` and MCP; DNS-rebinding guard off for the proxied
   host; design `docs/plans/2026-09-16-mcp-in-serve.md`). Suite 4328,
   lint clean, independent review found one wrong-accept (fixed, pinned
   red/green). `.mcp.json` carries `llm-orc-ng-mini` pointing at it.
   Deployed and accepted on the rig (design doc, Result). **Released
   as v0.20.1** on the practitioner's go (push, PyPI, GitHub release,
   formula bump). Policy from the practitioner: the mini runs releases,
   not checkouts. Doing that surfaced #197 (P0, closed the same
   evening): a clean install of 0.20.1 resolved `mcp` 2.x and died at
   import (the lock hid it). **v0.20.2** pins `mcp<2` (clean resolve
   verified from PyPI). The brew formula could not build on the Intel
   mini (cryptography>=49 has no x86_64 wheel; rust has no Intel bottle
   either, hours from source); the tap formula now constrains
   `cryptography<49` on Intel only, with the reasoning (PYSEC-2026-3552
   is a PKCS#7 decryption oracle; llm-orc uses only Fernet). **The mini
   runs `brew install llm-orchestra` 0.20.2** (30 s build), health and
   `/mcp` verified over https; the interim `uv tool` install is removed.
   Lesson, binding: a release is verified by a clean install from PyPI,
   never from the checkout. #196: the serving project (`.llm-orc/`) is
   not in the wheel, so the plist's WorkingDirectory stays the checkout.
   Follow-up #195. The practitioner may move the mini to Linux; Intel
   macOS support is dwindling upstream.
2. **Regression gate on the new backend** (laptop or ng-mini): the ladder
   (T1 alone at r≥5 first, then the full run) and the 7-turn probe.
   Chat templating and tool-call parsing moved from Ollama's Go templates
   to llama.cpp's jinja; the spike showed tool calls parse and thinking
   is controllable, the ladder decides whether the seats hold. Then
   OpenCode against `https://llm-orc.homelab.nate.green/v1`.
3. **#123, #122, verified acceptance in the workspace** — unchanged from
   2026-09-12; the T1 instrument still precedes touching the build round.
4. **Filed 2026-09-16, sequenced after the ng-mini bring-up and the
   backend gate:** #189 code-footprint epic (#190 wire-or-delete six
   unwired modules, P1 because AS-2 names a validator nothing runs;
   #191 one service with generated MCP/REST adapters and a thin CLI;
   #192 CLI/display re-approach on one event stream; #193 spike:
   meta-simulation as a client-side ensemble, retire the interactive
   input mode if it covers the cases). Then #187 strangler-kernel epic
   with #188 as its spike (port the router supervisor and the
   OpenAI-compatible client to a Rust crate in the plexus workspace;
   exit criterion on the issue). Auth and promotion stay as they are.
5. Follow-ups filed by this arc: `llm-orc web` does not own the router
   (opt-in later); MCP over the tailnet (the SSE transport is the older
   protocol, a second launchd unit + proxy path); deepseek-r1's template
   ignores `enable_thinking` (fine for the reasoning tier); the library
   submodule's templates still say `provider: ollama`.
6. Gated on the practitioner: further pushes; #167/#141 (Anthropic arms).

### Owed live rows

#166 #169 #173 #171 (constructed non-participating shape via fault
injection), #172 #176, #175. Arcs 1-2, D, B-1 have theirs. #90's live
row is the regression gate in Next up 2.

### Process (agreed 2026-09-11, holding)

Reviews found real blockers 12 of 12 first rounds through 2026-09-12;
with the three self-tests required in the implementer brief, arcs 1 and
2 each closed in ONE rework round. Lessons paid for: a moving git ref in
a pin makes it vacuous after merge (pin to a hash — acfaf427); a
"harmless internal convention" in a gate field is never internal; a
fuzzy match that reads then overwrites the wrong file is worse than an
honest ask; existence is a workspace fact, never a verb. 2026-09-16: the
first real run of a new process-owning feature found four defects unit
tests could not (signal re-raise, list hygiene, async load, swallowed
stderr) — run the real thing before calling a lifecycle feature done.
Ops: subagent budget ~4-5M tokens/day before the limit; Sonnet
implementers, Opus reviews; ladder ~17 min at coder-on, probe ~10 min;
one serve per tree, restart before every gate; the 8b GGUF is a 5 GB
first download per box.

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

- [x] #90 — Ollama dropped; the serve owns a llama-server router (2026-09-16)
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
- [x] #174 merged (a dead seat refuses honestly on every route; dead-seat
  refusal outranks accept/validity; stderr retained head+tail capped)
- [x] #177 merged (one resolve-and-classify observation; bare CWD refs
  anchored so PATH cannot re-resolve; identity names bytes AND location)
- [x] #171 merged (ablation control + public-binding re-fix surface; four rounds)
- [x] #182 D + B-1 merged (discovery before an unnamed build; edits keep the prior surface); #183 B measured (coder keeps thinking)
- [x] Arc 1 workspace-aware routing (#185) and arc 2 the sandbox mirrors the workspace (#184, #182 A) merged 2026-09-12
- [ ] #123 code+tests per turn (+ tests destination); #122 edit delegation; verified acceptance in the workspace; #183 A/C; #181
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
11. **Pin the harm, not the mechanism.** Three of #177's four blockers
    were green pins measuring the code path a fix had just edited (a
    trap pin on the one shape where it held for free, an
    `assert_not_called` on the edited branch, identity pins checking
    only the digest suffix) while the defect ran a PATH impostor or
    crossed a cache. A pin must assert the OUTCOME — which program ran,
    which output crossed, what reached the wire — and go red under a
    mutant that reintroduces the defect it was written for.

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

**#163 #166 #168 #169 #170 #172 #173 #174 #176 #177 #178 #179 are merged
on local main and still OPEN on GitHub**, because nothing is pushed. They close when
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

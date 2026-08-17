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

## State (2026-08-17)

### Next up

**#166 is designed and ready to implement**
(`docs/plans/2026-08-17-166-empty-deliverable-design.md`). A pre-flight
review was dispatched but its findings were NOT captured before the
session ended — re-dispatch it rather than implementing straight from
the doc, because the design leaves one question explicitly open and it
is the kind this corpus has got wrong twice:

> Is there a reachable build route whose seat has NO `seat_contract:`
> block? Every contracted build seat asserts
> `len(results['seat']['artifacts']) > 0`, which raises on a dead seat,
> so if all build routes carry one then #166 is defence-in-depth rather
> than single-fault reachable — and the issue and design both currently
> imply it is reachable.

The other reviewer question worth keeping: a refusal minted at the
CALLER may be invisible to the ask-outcome ledger, since `_reject_kind`
parses its prefixes out of `emit.py`.

What IS measured and does not need redoing: the defect reproduces on
current main through the real chain (absent seat dep, empty response,
and `loop_unwrap`'s `{}`); it happened in production at trace line 469
(failed dispatch seat + crashed seat_contract + an empty `solution.py`
write); and across 136 live turns that produced a file write exactly ONE
had empty content — the defect itself — so no legitimate empty write
exists in the corpus.

Also open: **#168** (the engine wrap's error reaches the client with the
server's absolute paths and username — universal, not
misconfiguration-only; split out of #155 Arc C), the two decision
residuals still on **#155**, **#161/#162/#163/#165**, and the roadmap
items below.

Still blocked on the practitioner: **#167** (the volume paid runs — r=8
per level, roughly $10–30 per model, with a cheaper L1-and-L5-only
variant) and **#141's CLAUDE.md-confound spike**, whose None condition
suppresses the live `~/.claude/CLAUDE.md` and so needs an explicit go.

**#155 Arc A merged 2026-08-17** (535cebb1, no release): a serving node
that cannot READ its upstream input refuses. A crashed `shape` or
`form_gate` used to finish as `{"finish": true, "content": ""}` — an
empty answer indistinguishable from "the model had nothing to say" —
and a dead `seat_contract` read as "no per-seat gate ran", which is
indistinguishable from a route that has no gate. Positive recognition
rather than a denylist: each check names a key the healthy producer
always emits, and the engine's failure wrap carries a disjoint key set.

**Four review rounds, and the placement took all four.** The signal is
two signals, because the axis that matters is not HOW a node failed but
whether its death bears on this route. `node_failed` refuses first,
non-minting; `seat_gate_failed` refuses only on a build turn that would
otherwise SHIP, minting. Round 1 put it first and killed eight routes
the seat contract cannot affect; round 2 moved it ahead of the accept
gate, discarding a real verdict and its retry invitation; round 3 left
it ahead of the form-gate refusal, losing a SyntaxError to a generic
pipeline error. Every one was caught because a pin could not fail, and
the structural cause took three rounds to see: every pin fed one node
directly, so nothing ran the chain with a fault injected and a middle
node dropping a threaded signal left the whole suite green.

Pre-flight review is what made this arc tractable, and it falsified most
of the first design: the seat is a `dispatch:` node so its most
reachable death leaves the dep ABSENT rather than carrying an envelope;
the reproduction omitted `seat_contract`, which already refuses a dead
seat on build turns; the quoted engine envelope came from a code path
serving never takes; and the design claimed an "empty-read guard" that
does not exist. That last one is **#166**, the single-fault case: a dead
seat writes an EMPTY `solution.py` on the client, since `ast.parse("")`
succeeds and the caller writes any outcome carrying `file` and
`content`. Arc B is largely subsumed by it; Arc C (reason-text
sanitising, decision residuals) remains.

**Released v0.19.0** (ab8aa7db): PyPI, GitHub release, and the Homebrew
formula all updated, CI green. Minor rather than patch because a shipped
default changes — the script agent cache is off unless a project opts
in. Eight arcs since 0.18.18 (#152, #138, #154, #156, #157, #158, #159,
#160), none of which had changelog entries; the user-visible ones are
the cache default, #158's concurrency win, and three fixes where a bound
or a cache key was not doing what it claimed.

**#160 merged 2026-08-17** (57ae7d7a, no release): a script's cache
identity is its resolved path AND the sha256 of its bytes, and the cache
ships DISABLED. Editing a script used to serve the pre-edit result for
the TTL — a stale SUCCESS #159's predicate could not catch — and under
`persist_to_artifacts` it crossed processes. Interactive agents now skip
the cache at both the get and the set, because the byte-identity aliases
several references onto one key (the resolver's hyphen-to-underscore
normalization is enough) while `requires_user_input` still judges each
reference separately.

The default flip was NOT the mitigation an earlier draft argued it was.
Pre-flight falsified that by demonstration: two of the six registered
primitives are not pure functions of (bytes, input, parameters), and
they are the two most-used here. A `write_file.py` hit reports
`{"success": true}` with the file absent — a hit ELIDES A WRITE — and
`read_file.py` serves stale content when its input changes. Fixing the
key does not reach either. #161 carries the purity mechanism.

**Five review rounds, and every one found something that survived the
entire suite.** Round 1's flip was INERT (the config loader restated
every default and its copy had drifted to `True`, so a fresh install
still elided writes; the pin asserted a field the runtime never reads).
Round 2's skip at the *get* was unpinned and deletable with 3431 tests
green. Round 3 found `persist_to_artifacts` was the one default with no
pin — the same defect class recurring inside its own fix — plus an empty
`script_cache:` block crashing executor construction and a disabled
cache still hashing each whole script twice per execution. Round 4 found
the `or {}` fix closed only the falsy half and that a count assertion
had become a trap aimed at #161. Seventeen instruments, all
mutation-verified.

Three of my own "measured" claims were falsified and corrected: the
`os.path.isfile` guard's contribution (wrong twice, opposite
directions), "computed at both get and set" (once), and the identity's
cost (0.19% of a ~49ms execution, not "the expensive part"). Bounds
recorded rather than closed: the identity covers the entry file, not its
imports (**#162**); an edit reverted mid-run still files B's output
under A's key, pinned as known-wrong; the identity degrades silently
when `read_bytes` raises (**#163**). Also filed: **#164** (a doc
promising a per-agent `cache: true` key that `extra="forbid"` rejects)
and **#165** (a pre-existing 1-in-6 `-n auto` flake, undiagnosable
because both persistence paths swallow their exceptions).

**#156 merged 2026-08-17** (068e703f, no release): the 511 measurement
instruments under `benchmarks/agentic_serving/tests` run in `make test`
and CI, and `make lint` plus CI's type check now cover `benchmarks/`.
Review verified the gate GATES — breaking an oracle, a scorer, and a
stats function each turns a bare `make test` red. Coverage gate
unaffected (92.2% either way); cost is `make test` ~26s -> ~35s.

The obstacle was not the one the issue expected. `tests/unit/` had no
`__init__.py` while `tests/__init__.py` did, so the chain broke there,
pytest put `tests/unit` on `sys.path`, and `tests/unit/benchmarks/`
shadowed the real top-level package — 18 collection errors naming the
module pytest DID find. Same class as #138's `parse.py`.

**Three rounds went into a general shadowing guard before it was
deleted**, which is the part worth carrying forward. Round 1's could not
fail in the gate at all. Round 2 fixed that, and round 3 demonstrated
four ways the fixed version reddened a CORRECT tree (an ordinary
`conftest.py`; battery arms under any testpath, including dot-dirs
pytest skips; an order-dependent verdict from suites that
`sys.path.insert` at import time) while still missing a live
namespace-package shadow. The cause was structural: it reimplemented
pytest's import resolution from outside, which a directory walk cannot
reconstruct. The realisation that should have come first is that a
shadow of this kind ALREADY fails the build by breaking collection — a
guard can only add a name, not detection. Two direct assertions
replaced it. Bounds: the instruments have no coverage floor of their own
(`runner.py` 40%), the battery shell scripts are ungated, and
`judge_adequacy`'s tests still live in `tests/unit/benchmarks/`.

### Earlier (2026-08-17)

**#159 merged 2026-08-17** (b9949c53, no release): a script's own failure
is never served from cache. Reproduced before the fix — a timing-out
agent returned 1.01s then 0.00s from cache, and with
`persist_to_artifacts` the entry survives a restart, so the worst case
was cross-process rather than one in-process hour. The predicate needed
TWO clauses: a success-only rule would have missed the case the issue is
named after, since `web_searcher` reports failures as `{"error": ...}`
and exits 0, and NONE of the 33 serving scripts emit a boolean `success`
at all. Twelve pins on `get_stats()` rather than wall time. Merge review
ran the full mutation table and falsified the commit's own claim that
each clause had a killing pin; four survivors now each turn exactly one
pin red, including a real `AttributeError` crash path on the schema
dispatch shape. Mitigation, not the subsystem fix: **#160** carries the
deeper hazard — the cache key hashes the script PATH rather than its
bytes, so editing a script serves the pre-edit result for the TTL, a
stale SUCCESS no failure-skipping predicate can catch.

### Earlier (2026-08-17)

**#158 merged 2026-08-17** (64e5667b, no release): script agents run off
the event loop on a dedicated thread pool, and their DUPLICATE outer
timeout is retired. Measured 4 x 1s gathered: **4.15s -> 1.03s**, loop
ticks during a script **0 -> 46,676**; seven phases in the repo already
carry 2+ script agents. The authoritative bound is untouched — it is the
inner subprocess timeout #157 wired up, the only one that can reap a
child; the outer timer could never fire during the one window it
uniquely covered, and once agents overlap it would charge queue delay
against an agent's budget. The error contract deliberately does not
move, pinned end to end because fan_out skips expansion on non-success.
Five pre-flight rounds plus two merge rounds: rounds 1-4 designed a
thread-slot gate, round 5 found it deadlocks on the default path (a
cache hit leaks a permit), and removing the timed region dissolved the
problem. Merge review found the blocking defect in the INSTRUMENT — the
pin guarding the result shape passed under a mutation that changed it
back. Bounds recorded: parent timeouts can now fire mid-script,
interactive agents can hang on a silent pipe, identical concurrent
agents no longer dedupe through the cache, no operator concurrency knob.

### Earlier (2026-08-17)

**#157 merged 2026-08-17** (108b2e8, no release): engine-run script
agents get the timeout the ensemble already resolved. The bug was worse
than filed — BOTH bounds were defeated independently. The inner one
because `model_dump` always emits `timeout_seconds` and supplies `None`
when unset, so `.get(key, 60)` never saw its default and every
subprocess ran unbounded; the outer `asyncio.wait_for` because the
blocking subprocess sits in an `async def` and stalls the event loop, so
that timer never runs (#158). Demonstrated in wall-clock: the unmocked
pin takes 30.2s pre-fix (the full sleep runs) and ~1-2s with the fix.
One shared `resolve_agent_timeout` now answers the question for both the
dispatcher and the script runner. Six nodes declare explicit timeouts
rather than racing the SHIPPED 60s default, since `accept_executor`'s
worst case is exactly 60 (45s aggregate + one 15s child, its budget
check running before each spawn) and `web_searcher`'s default backend is
third-party-bounded. Follow-ups filed: #158, #159 (failure envelopes
cached as successes for the hour-long TTL, which this fix makes more
frequent).

### Earlier (2026-08-17)

**#154 merged 2026-08-17** (cea2607, no release): `.py` script agents
run under llm-orc's OWN interpreter, not whichever `python3` the
caller's PATH exposes. This is the root cause behind #152's captured
misfire, where serving scripts importing `llm_orc` died while
stdlib-only scripts in the same ensemble kept running, producing a
half-dead pipeline rather than a clean failure. Measured: from a shell
with no venv on PATH the suite went from **50 failures to 0** (3395
passing); in the environment where CI runs it is a no-op, since
`python3` and `sys.executable` are the same binary there. The live gate
inverts #152's — a bad-PATH serve now answers and builds (positive
control recorded, and a build ask included because a non-build ask
cannot observe `seat_contract.py`). `llm-orc scripts test` and the
primitive composer/registry were routed the same way. Two review
rounds: round 1 caught a defect I introduced in a BDD fixture plus a
misattributed baseline in my own record; round 2 approved and caught
that the newly routed sites had no regression instrument. Follow-up
filed: #157 (engine-run script agents have NO timeout — `model_dump`
always supplies the key as None, so the 60s default never applies).

### Earlier (2026-08-16)

**#138 volume-ladder instrument merged 2026-08-16** (2ea793d, no
release — benchmarks only): the prerequisite the parity-v2
pre-registration called "the real cost". Per-level fixtures (only the
asked modules, hash-pinned), one hidden nonce-varied oracle per module,
disk-derived shipped with a six-way verification channel, shared
arm-blind truth capture, level-scaled timeouts, and a reporter that
refuses to name a gate branch below r=8. Arm-0 calibration recorded
(`docs/plans/2026-08-16-138-arm0-calibration/`): four levels, all exit
0, the serve fixing the first named file only (the #123 bound observed),
every shipped fix correct. Three review rounds, 25 findings; the
organizing one was that every failure mode fell closed into
shipped_broken or the gate numerator, and round 2 caught two more that
round 1's own fixes introduced. Paid arm-1/arm-2 runs await a cost
decision.


| arm / serve version | runs | strict | dishonest | record |
|---|---|---|---|---|
| Arm 0 pre-fix (v0.18.14) | 2/3/4 | 25/39 | 3 (one per run) | `docs/plans/2026-07-1{4,5}-arm0-runs/` |
| Arm 0 post-fix (v0.18.15) | 5 | 11/13 | 0 | `docs/plans/2026-08-12-arm0-run5/` |
| Arm 1a Haiku 4.5 (paid, OpenCode Go) | 1, 2, 3 | 38/39 | 0 | `docs/plans/2026-08-12-arm1-runs/` |
| Arm 1b Sonnet (paid, OpenCode Go) | 1, 2, 3 | 39/39 | 0 | `docs/plans/2026-08-12-arm1-runs/` |
| Arm 2a Haiku 4.5 | 1, 2, 3 | 35/39 | 4 | `docs/plans/2026-07-15-arm2-runs/`, `2026-08-12-arm2-runs/` |
| Arm 2b Sonnet | 1, 2, 3 | 39/39 | 0 | `docs/plans/2026-07-15-arm2-runs/`, `2026-08-12-arm2-runs/` |

All Arm-1/Arm-2 scores independently J-scored against the frozen rubric.
Arm-1 headline: same Haiku, 4 dishonest behind Claude Code vs 0 behind
OpenCode — the honesty split between model tiers is harness-sensitive;
discretionary verification reproduced in every scorer's notes. Column
cost ≈ $2.75 total; 3 instrument flags all REFUTED as one classifier
false-positive family (#147). Versions differ across rows; do not pool.
Caveats: parity table v1 (`docs/plans/2026-07-15-first-parity-table.md`)
plus host-CLAUDE.md leakage observed on Arm-1 Sonnet runs (#141).

**#152 merged 2026-08-14** (57315bc, no release — serving scripts only):
fail-closed routing. shape accepts a decision only when it parses to a
dict with a non-empty string `target` and `build`/`kind` present;
anything else (the captured crash envelope, the classify-crash launder
through resolve's `target: ""`, out-of-set decider, drifted-producer
shapes) refuses FIRST at emit with the non-minting `Refused: serving
pipeline error` terminal — never a write, never silent empty prose. The
2026-08-13 misfire conditions replayed through real OpenCode refuse
honestly (live gate record); run-10 battery row 13/13, 0 dishonest,
zero routing refusals on a healthy serve (oracle 0/0/3 — two accept-gate
rejections plus their cascade, the recurring qwen3:8b variance class).
Two-round adversarial review APPROVE with four mutation-verified gate
pins. Follow-ups: #154 (bare-python3 interpreter fragility — the
misfire's environmental trigger, also explains the checkout-shell
endpoint failures), #155 (downstream positive-completeness analogues).
**v0.18.18 released 2026-08-14** (PyPI + Homebrew): #153
offset-continuation reads — client-capped reads continue at the
trailer-named offset and stitch whole-or-refuse (call-count bound with
offset monotonicity; POSITIVE end-of-file completeness with the total-N
crosscheck; same-turn-segment parts). Live gate: the 80KB serving
caller itself grounds via a 2-part stitch — the #121 coverage bound
CONVERTED; run 9 regression row 13/13, 0 dishonest, oracle 2/0/1,
fastest wall of the family, seam inert on the ladder. Two-round
adversarial review (round 1: a budget first-read-wins ordering
regression and a lone-offset-part wrong-accept, both fixed with
mutation-verified pins) → APPROVE. Remaining on #153: the 2000-line-cap
trailer wording (fails closed until captured), the per-line 2000-char
silent truncation (#149 family). **v0.18.17 released 2026-08-14**: the #121
content-grep rung, slice A — def-anchored one-round content search,
deterministic identifier menu over verified definition sites, guarded
closed-menu pick (abstention falls open), AST-verified grounding
attribution keyed on a structural this-turn read signal. Exit gate MET
live (grounded answer citing turn_trace.py's real constants; re-passed
first-try post-fix); arm-0 run 8 regression row 13/13, 0 dishonest,
oracle 2/1/0 (best row), grep path inert on the ladder. Three design
review rounds (v1 REDESIGN with measured blockers; ground truth via
`opencode debug rg search`) and three implementation rounds (round-2
BLOCKER: rendered-order keying livelocked the pick; fixed
structurally). The gates DISCOVERED #153: OpenCode caps read output at
50KB inside the serve's 96KB window — fixed (capped reads refuse
honestly); offset-continuation reads are the named recovery, and the
80KB caller attracting picks then refusing at the cap is the recorded
coverage bound. **v0.18.16 released 2026-08-13**: #148, #145+#150,
and the #144 self-reference slice (its literal "answers grounded"
exit stays OPEN on the whale — unblockers #106/#151/chunked reads).
#143 CLOSED as an honest miss (reopen rides #119). Open follow-ups:
#147 (classifier false positives), #149 (client-side truncation
flanks), #151 (server-queried window), #155 (pipeline
positive-completeness), #156 (instrument pins ungated), #160 (cache keys on path,
not bytes). Operating rules:
`docs/loop-protocol.md`.

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
- [ ] #161 #162 #163 #165 #166 #168 — script-cache purity/imports/degrade,
  the -n auto flake, the empty-deliverable write, and the refusal-reason
  path leak; #155 Arcs B/C remainder
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
- [x] #155 Arc A — a node that cannot READ its input refuses (crashed shape/form_gate finished as an empty success); Arcs B/C still open
- [ ] #151 runtime-window detector remainder · #155 Arcs B/C · #168 refusal reasons leak server paths + username · #161 cache purity · #162 cache misses imports · #163 identity degrades silently · #165 `-n auto` flake · #166 empty deliverable written to the client · #85 sandbox hardening · #84 gate adversarial harness · #90 llama.cpp · #93 hot path · #95 dead surface · #106 shape home · #110 artifact quality · #114 trace cap · #132 BitNet · #142 reject templates

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

Two closed issues gate work that is tracked elsewhere: **#138**
(instrument shipped; the paid runs are #167) and **#139** (curve
measured; #140 carries the remainder).

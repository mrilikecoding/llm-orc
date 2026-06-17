# Spike τ — Cheapest config that enables long-horizon agentic tasks on a 32GB machine

**Status:** Pre-registered (methods-reviewed; run pending practitioner go)
**Date:** 2026-06-16
**Cycle:** 7 (agentic-serving), BUILD — research thread surfaced by the benchmark
**Methods review:** `housekeeping/audits/research-methods-spike-tau.md` (3 P1, 3 P2, 2 P3, + a premise-buster — all applied below)

## Origin

The agentic-serving benchmark grid surfaced that multi-file cells hang **not** from
model incapacity or a framework bug, but because the active seat config
(`agentic-orchestrator-offline-tools` = local qwen3:14b) plus the local qwen3:8b
coder cannot co-reside on 32GB, so every turn pays a model swap (~120s+ seat
latency) and the stack degrades under sustained load. Decisive evidence: the
identical `h2c1` cell **hung mid-grid** (820s, 1/2 files) but **converged in 399s
isolated on a fresh stack** (2/2 files, all metrics pass). The
`agentic-routing-planner` ValueError + ASGI-incomplete errors were ruled out
(present in the known-good σ/ADR-039 runs that converged).

**Config-reality reframe (methods-review Criterion 4, verified).** The
architecture's *intended* default seat is already a cheap-cloud frontier model
(`agentic-orchestrator` = MiniMax on OpenCode Zen). That free tier ended (2026-05,
`FreeUsageLimitError`), so the config fell back to the local-14b `offline-tools`
profile — and *that fallback* is the two-local-model swap that breaks 32GB. The free
frontier seat is not returning (limited-time promotion). The affordable frontier
options on Zen are now **paid MiniMax** and **paid qwen** (the latter being the
architecture's own seat family, hosted).

So the real question is **not** "frontier vs local" (the architecture already
intends a frontier seat) but: **given the free frontier seat is gone, what is the
cheapest config that enables long-horizon agentic tasks on 32GB?**

## Central question

On a 32GB machine, find the **cheapest rung** of a config ladder — from a single
small local model up to a paid hosted seat — that enables long-horizon (many-file,
many-turn) tasks to converge with stable per-turn latency.

## Hypotheses

- **H1** — the per-turn model swap (two local models that don't co-reside) is the
  dominant cost; any single-resident-local-model config converges multi-file far
  faster.
- **H2** — a hosted seat (qwen or MiniMax) + local 8b coder leaves only one light
  local model, so it both eliminates the swap AND minimizes sustained-session
  degradation (the lightest local load of any arm).
- **H3** — whether a *free* single local model suffices depends on seat capability:
  8b may be too weak to sequence many files (the ADR-033 §6b axis-2 risk); 14b may
  suffice but carries the full local load.

## Arms — a free-first fallback ladder

| Arm | Seat | Coder | Local resident | Cost | Isolates |
|-----|------|-------|----------------|------|----------|
| **ref** | qwen3:14b (offline-tools) | qwen3:8b | 2 (**swap**) | $0 | the broken baseline (gate only) |
| **S** | qwen3:8b | qwen3:8b | 1 (no swap) | $0 | cheapest — does a *small single* model sequence multi-file? |
| **A** | qwen3:14b | qwen3:14b | 1 (no swap) | $0 | does a *capable single* local model suffice (no frontier)? |
| **B-qwen** | hosted qwen (Zen, paid) | qwen3:8b | 1 (no swap) | ¢/task | hosted seat, same model family; lightest local load |
| **B-mmx** | MiniMax-m2.5 (Zen, paid) | qwen3:8b | 1 (no swap) | ¢/task | the alternative affordable Zen seat |

**Attribution is clean (fixes methods-review P1-A):** ref→S isolates removing the
14b-seat-swap; S→A isolates seat capability (8b vs 14b, no swap either way); A→B
isolates local-vs-hosted seat. The original v1 claim that arm A "isolates swap
alone" was wrong (A also upgrades the coder 8b→14b) — A is now explicitly the
"single capable local model" arm, not a swap control; **S is the swap control.**

**Execution order (free-first):**
1. **Phase 1 ($0):** ref (gate only), then S and A (gate + ladder).
2. **Phase 2 (paid, gated):** B-qwen and B-mmx (gate + ladder) run **only if** no
   free arm clears the gate-and-ladders, **or** to push the ceiling higher than a
   free arm reached, **or** because the sustained-session test shows the free arms
   degrade and a light-local-load hosted arm is wanted. Phase 2 requires explicit
   cost confirmation (below).

## Method — minimal gate → progressive ladder → attribute the break

- **Gate:** a 5-file task (known to converge isolated; σ/ADR-039 validated 5 files).
  Each arm must clear it: all files + clean termination.
- **Ladder:** arms that clear the gate push 8 → 10 → 15 → 20 → 30 files (one fixed
  template, varying only file count), until the arm breaks **or reaches the 30-file
  ceiling** (P3-A practical stop). The breaking rung is that config's long-horizon
  ceiling on 32GB.
- **Fresh restart (serve + ollama) between every run** — measures intrinsic
  capability, not accumulated degradation.

### Break attribution — pre-registered signals (fixes P1-B)

When an arm fails to converge a rung, classify the cause by these signals (read from
serve.log + the workspace), committed before running:

1. **seat-sequencing failure** (capability) — the `produced` file set is unchanged
   across **≥3 consecutive turns** while turn-decisions keep emitting `action=write`
   (the seat re-targets an existing/produced file). The Finding-G signature.
2. **content-anchor overload** (architectural, anchor-specific) — the sibling-
   signature content anchor's token share of the callee dispatch exceeds **50%** of
   the dispatch, or grows monotonically to dominate, *before* the model's full
   context is hit. Measured from the dispatch composition; distinct from #3.
3. **context-overflow** (architectural, session-wide) — an explicit context-length
   error/truncation in serve.log, or the dispatch request token count reaches the
   configured `num_ctx`.
4. **latency / degradation** (hardware) — per-turn latency breaches the flat-latency
   bound (below) before convergence; the run stalls on inference time, not logic.

(i) and (ii)/(iii) tell the practitioner whether the ceiling is **seat capability /
the architecture** vs (iv) **the rig** — the standing "can the architecture support
long-horizon?" question.

### Flat-latency definition (fixes P1-C)

Per arm, `baseline` = median seat-latency of turns 2-3 (warm, excludes any cold
first-turn load). Per-turn latency is **flat** iff: (a) linear-fit slope across the
run **< 10 s/turn**, AND (b) **no turn > 2× baseline**. This same definition gates
the convergence criterion and the sustained-session check.

## Sustained-session check (fixes P2-B)

The best free arm and (if run) the best hosted arm each run **4 gate-sized tasks
back-to-back without restart** (~6-10 turns each; ~30-40 turns total). Pass = the
flat-latency definition holds across the whole back-to-back series, anchored to the
arm's isolated-gate baseline; or per-turn latency fully recovers on a task-boundary
restart. This is the "long session" half of the goal (tests H2).

## Measurements (per run; all $0-checkable except cost)

- converged? (all expected files + clean termination)
- per-turn seat latency — `baseline`, slope, max (the swap-tax + degradation signal)
- total wall time
- on break: the attribution class (1-4 above)
- cost (Phase 2 only; rolling total)

**Latency caveat (fixes P2-A):** hosted arms (B-qwen, B-mmx) include network
round-trip to Zen; local arms use Ollama. Cross-arm *absolute* latency is therefore
not directly comparable — only **within-arm trend** (flat vs drifting) is. The
decision rests on convergence + ceiling + within-arm flatness, not cross-arm speed.

## Pre-registered decision rule

- An arm **enables long-horizon on 32GB** iff it clears the 5-file gate with flat
  per-turn latency AND ladders to **≥8 files**.
- **Operating config = the cheapest (leftmost) arm that enables long-horizon.** If S
  qualifies → a single 8b, zero frontier, zero dependency. If only A → a single 14b,
  still free. If only a hosted arm → the cents/task are earned; pick B-qwen vs B-mmx
  by ceiling reached (tie → B-qwen, same family / lower behavioral risk).
- **Sustained-session tie-break:** if two arms both enable long-horizon, prefer the
  one that holds flat latency across the back-to-back series (expected to favor the
  light-local-load hosted arms — that is the H2 test, and a legitimate reason to
  spend even when a free arm "works" for single tasks).

## Cost (free-first; fixes P2-C)

Phase 1 is $0. Phase 2 (paid) per-run cost ≈ `turns × (seat_input + seat_output
tokens) × Zen_rate`; seat input grows with file count (history + content anchor), so
upper rungs cost more per call. **Rolling total surfaced at each rung boundary.**
Estimated full Phase-2 spend **< $1**; a hard **gate at $1** pauses for re-
confirmation before continuing. First paid call gated on practitioner confirmation
of the then-current estimate (the actual Zen per-token rates resolved at setup).

## Known limitations / threats to validity

- Single machine, single session — results are this-rig-specific **by design** (the
  target); generalization to other 32GB machines is asserted, not measured.
- Task template is greenfield multi-file Python; the ladder varies file count, not
  task kind (edit/repair excluded — benchmark scope).
- `ref` may not converge the 5-file gate even isolated (single-task runtime may
  exceed degradation onset); a ref-gate timeout is a bounded data point, not a
  failure.
- A-vs-S conflates seat *model* (8b vs 14b) with nothing else (both single-resident,
  no swap) — clean. ref-vs-S conflates the 14b-seat with the swap; the isolated-h2c1
  result already attributes the swap as the dominant cost, so ref is reference, not a
  load-bearing arm.
- The content-anchor scaling ceiling (signal #2) is the suspected architectural
  limit; its attribution rests on dispatch-composition inspection at the breaking
  rung.

## Artifacts

Harness + per-run data: `scratch/spike-tau-long-horizon/`. This log updated per arm.

## Results

### Arm S (single 8b) — FAIL on seat capability; surfaced a turn-1 client-termination recovery gap (2026-06-16)

Arm S gate (σ 5-file task, only qwen3:8b resident — **no swap confirmed**):
**converged=False, 0/5 files, 1 turn, 338s** (not a timeout — the client ended
early). Cause, from the opencode `.err`: the 8b seat's first action was a `write`
to an **out-of-workspace path** (the `output_directory` hallucination);
OpenCode's sandbox **auto-rejected** it; the client terminated after that one step.

**No recovery path fired** (serve.log: 1 turn-decision, 0 trailing, 0
completeness, 0 recover, 0 REMAINING). Reason: ADR-040 completeness gate, ADR-041
form-recovery, and the F-σ.1 REMAINING-retry all run **server-side on a trailing
turn** (after the client executes a tool and returns the result). A turn-1
client-side rejection ends the session before any trailing turn exists, so the
recovery loop never engages. **Gap: the recovery stack is trailing-turn-only —
blind to turn-1 (or any-turn) client-side action rejection.**

**Patch — worth doing; queued, does NOT stop this spike (practitioner direction,
2026-06-16):** extend the ADR-041 destination-validity gate to validate the write
**destination path** (in-workspace / client-acceptable), not only the deliverable
form/content, and **re-dispatch server-side on a bad path** — so a
client-rejectable write is never emitted and the loop stays alive. Natural fit
with ADR-041's deferred "edit/bash destination validity" (BUILD outstanding item
#4). Precedent that re-dispatch rescues weak-model output: Spike π's server-side
recovery converged 8b-*coder* form bleeds.

**Caveat (honest):** the patch closes the turn-1 death, but the 8b's deeper
weakness is *sequencing* (it also ran only 1 turn and mis-targeted). The patch
enables a *fair re-test* of arm S; it does not guarantee 8b-seat viability. After
τ's baseline arms (A, then B if needed) establish the unpatched ceiling, a patched
8b-seat re-run becomes an added rung: does 8b + path-recovery clear the 5-file gate?

### Arm A (single 14b) — capable but too slow (2026-06-16)

A gate (mostly single-14b — serve.log: 5/6 dispatches = 14b `escalated-general`,
one early 8b artifact): **converged=False, 3/5 files, timed out at 1800s, 4 turns,
~249s/turn**, flat/improving latency (slope −63). The 14b seat **sequences real
deliverables** (converters.py + test + README, advancing — unlike the 8b) but is
**too slow** on 32GB to finish 5 files in 30 min (≈60 min projected). Key insight:
**eliminating the swap did not make it fast** — a single 14b is ~as slow per turn
(249s) as the swapping ref, because the 14b's own inference is the bottleneck once
it is the only model. So the free local-single arms fail oppositely: **8b too weak,
14b too slow.** → justifies the paid hosted-seat arm (pre-registered condition).

### Arm B-mmx (MiniMax seat) — INCONCLUSIVE: chat-template incompatibility (2026-06-16)

B-mmx gate (MiniMax-m2.5 seat via Zen paid + 8b local coder; auth confirmed, seat
responded turn 1): **0/5 files, hung to the 1800s timeout, 1 turn, 0 dispatches.**
Not a capability result — an integration failure:
- Turn 1 the MiniMax seat chose `action=bash` (a carry), diverging from the local
  qwen seat's write-delegate pattern.
- The follow-up 400'd: *"jinja template rendering failed. Message has tool role, but
  there was no previous assistant message with a tool call!"* — MiniMax's template
  strictly requires each tool-result paired with its assistant-tool-call. The
  framework's orchestrator request sends an unpaired tool-result; **qwen's template
  tolerates it (arm A worked), MiniMax's does not.** opencode retried 62× over 30
  min (all 400, + a 429 rate-limit), producing nothing.

**Two findings:**
1. **Hosted seat is not drop-in across model families** — tool-use behavior
   (write-delegate vs bash) and template strictness are model-specific. Lowest-risk
   hosted seat = **same family as the local seat (qwen)** → B-qwen. Validates the
   pre-reg's B-qwen preference.
2. **Framework follow-up (queued):** the orchestrator's conversation construction
   emits a tool-result without its paired assistant-tool-call — non-conformant,
   tolerated by lenient templates (qwen), rejected by strict ones (MiniMax). Fixing
   it broadens hosted-seat support beyond qwen. Sibling to the turn-1 path-recovery
   patch — a robustness improvement, not a spike blocker.

**Cost:** minimal — failed on turn-2 template 400s (fast), no real generation billed.

**Next: B-qwen** (hosted qwen on Zen — same template family → should avoid the 400).

### Arm B-qwen (hosted qwen3.6-plus seat) — hosted-seat CORE VALIDATED; convergence blocked by machinery, not the seat (2026-06-16)

B-qwen gate (qwen3.6-plus seat via Zen paid + 8b local coder): **2/5 files, ended at
1492s (NOT timed out), 4 turns.** The positives are the headline:
- **No template 400** (qwen-family compatible — the MiniMax wall avoided).
- **Seat is fast**: turn-1 wall **7s** (remote seat decision ~instant) vs arm A's
  ~250s. The hosted-seat speed benefit is real.
- **Seat sequences correctly**: judgment_verdict=REMAINING on all 3 trailing turns
  (no premature finish); files coherent (converters.py has the real functions).

Why it didn't converge (machinery, not the seat):
1. **Completeness gate didn't fire** (completeness=0, requested=0) — ADR-040
   named-file detection didn't engage for the σ task phrasing (the η-flagged
   brittleness). Termination fell to the qwen judge (correctly REMAINING), but...
2. **Turn-4 REMAINING→stop mismatch**: framework decided turn 4 = REMAINING
   action=write, but opencode step 4 received a text-stop (no tool-call) and ended.
   The intended write didn't reach the client as a tool-call.
3. **Local 8b coder slowed** under load: dispatches 50→356s (degradation now isolated
   to the single light local model — lighter than the swap, but still drifts).
4. The seat spent turn 2 on action=read (exploring before dependents) — capable
   behaviour, but a turn with no new file.

### Seat-question synthesis (S / A / B)

The seat must be **both capable and fast**:
- **8b** (arm S): fast, too weak — can't even target the sandbox.
- **local 14b** (arm A): capable (sequences real files), too slow — ~250s/turn, no
  faster than the swap because the 14b's own inference is the bottleneck.
- **hosted qwen** (arm B-qwen): **both** — capable + ~7s/turn seat. The direction.

With the hosted seat, the bottleneck **shifts off the seat** onto the single light
local 8b coder. No arm converged the 5-file gate yet, but the reason is no longer the
seat — it's the convergence machinery (completeness gate firing; the REMAINING→stop
marshalling) + local-coder degradation. A fixable fix-list, not a dead end. (ref +
the ladder + sustained-session check still pending; the queued patches —
turn-1 path-recovery, orchestrator message-pairing — also apply.)

### Correction + root cause: B-qwen's non-convergence was a coder form-bleed, not the gate

Re-investigation corrected two earlier mis-reads: (a) the `requested=0` "completeness
gate didn't fire" was a measurement error — the `completeness:` diagnostic was
removed at the σ close, so grep returns 0 even when the gate fires;
`_extract_requested_deliverables` cleanly extracts all 5 σ files. (b) The real
blocker was **`cli.py`, the persistent form-bleed file**: the 8b coder bled
(`form recovery ... recovered=False redispatches=2`), escalation fired and used the
**14b** (`form escalation ... tier_profile=agentic-tier-escalated-general`) — which
**also bled**. The free 14b rung does not fix the persistent cli.py bleed (the
cycle's open question, now answered). The ladder had **no frontier rung** because
`orchestrator.form_escalation.frontier_profile` was unconfigured. **Framework works;
ladder had no top rung.**

**Fix (config-only):** added `agentic-tier-frontier-coder-minimax` as the
`form_escalation.frontier_profile` → ladder = **8b → 14b → MiniMax**. MiniMax-as-
*coder* is one-shot (no template issue; the B-mmx 400 was the multi-turn *seat*),
and Arm E proved it produces clean cli.py 6/6. Fires only on persistent bleed → cents.

### Arm B-qwen-esc — DIRECTION CONFIRMED at the 5-file gate (2026-06-16)

Re-ran with the frontier rung wired: **5/5 files, all valid Python, loop converged,
clean termination, ~775s (~13 min)** (vs the prior B-qwen 2/5 in 1492s). Per-turn
~145s (the local 8b coder; the hosted qwen seat decision is near-instant). The
ADR-039 content anchor fired (anchor=true turns 2-5); cli.py imports the **real**
converters API. Form handled by **cheap recovery** this run (test_cli.py recovered
×1; cli.py valid on its own) — the bleed is **stochastic**, the frontier rung sat
unused as a safety net.

**One remaining metric miss — content coherence:** `test_cli.py references undefined
cli.parse_args` — cli.py exposes `main()`+argparse (no `parse_args`), but the test
assumed a conventional parse function. A CLI-interface-testing edge case of the
signature content anchor — a refinement (ADR-039) / PLAY-quality concern, not a
structural block.

**Answer to the spike's question (at the 5-file gate):** **hosted qwen seat + local
8b coder + escalation ladder (8b→14b→frontier)** produces a complete, converging,
fast, form-valid, cleanly-terminated multi-file project on the 32GB machine — mostly
local, frontier only on rare persistent bleeds. Direction empirically confirmed.
**Remaining pre-registered work:** the ladder (8→30 files; the real ceiling), the
sustained-session check, ref baseline, and the content-coherence refinement.
Spike spend to date ≈ 4¢.

### Ladder campaign — CEILING FOUND: the OpenCode client step cap, not the framework/rig (2026-06-17)

Gate confirm (n=3) + ladder on the confirmed config, fresh restart per cell:

| cell | files | converged | produced | coh | wall | med/turn |
|------|-------|-----------|----------|-----|------|----------|
| gate-a/b/c | 5 | ✅ ✅ ✅ | 5/5 | ✅ | ~12 min | 87-146s |
| l8 | 8 | ✅ | 8/8 | ✅ | 11 min | 84s |
| l10 | 10 | ✅ | 10/10 | ✅ | 16 min | 70s |
| **l15** | 15 | ❌ | **12/15** | ✅ | 26 min | 87s |

**(b) resolved:** gate confirm **3/3 clean** — the earlier coh=False was a stochastic one-off.

**(a) ceiling found + attributed.** The config converges cleanly through **10 files**
(and was healthily progressing at 12/15 when l15 broke). The l15 break is **OpenCode's
per-run step cap**, not a framework/seat/rig limit: 13 turns, every one
`action=write judgment_verdict=REMAINING` (no stall, no premature finish), latency
**flat at 87s** (no degradation), all 12 produced files coherent — then OpenCode ended
the session with `stop` at step 13 while the framework still said REMAINING.
`opencode run` exposes no max-steps flag (only `--replay-limit`), so ~13 steps is a
client default. (The `max_turns:1 extra_forbidden` log is an unrelated
invalid-ensemble-yaml warning, benign.)

## Spike τ conclusion

On a 32GB machine, **hosted qwen seat + local 8b coder + 8b→14b→MiniMax coder-tier
escalation** converges clean multi-file projects up to **~10-12 files** (~11-16 min
each), **mostly local** with the frontier rung firing only on rare persistent bleeds
(≈cents). Per-turn latency is flat (70-146s, the local 8b coder; the hosted seat is
near-instant). **The long-horizon ceiling is the OpenCode client's per-run step cap
(~13 turns), a client limit (raisable via client config / session continuation) — NOT
the architecture, the seat, or the 32GB rig.** The earlier "32GB is the wall" framing
is **refuted for this config**: the rig is not the bottleneck up to the client cap.
The seat question (S/A/B) resolved to: 8b too weak, local 14b too slow, hosted qwen
both fast and capable. Total spike spend ≈ 5¢.

**Follow-ups (BUILD / not spike-blocking):** (1) wire the frontier rung + the hosted
qwen seat as a supported config (currently spike-config in `.llm-orc/config.yaml` —
the active seat is the PAID qwen3.6-plus; revert-to-local-default vs adopt is a
practitioner call); (2) the content-coherence CLI-test refinement (stochastic, ADR-039);
(3) the queued turn-1 path-recovery + orchestrator message-pairing patches; (4) raising/
bypassing the OpenCode step cap for >12-file single sessions (client config / session
continuation). The sustained-session check + ref baseline remain unrun (deprioritized —
the per-cell fresh-restart ladder already shows flat latency and the ceiling is client-
side).

### CORRECTION (2026-06-17, post web-search): the ceiling is a framework glitch, not a client cap

The attribution above ("OpenCode client step cap") is **wrong** — corrected by a web
search of OpenCode's docs. **OpenCode has no default step cap**: the `steps` setting is
opt-in per-agent ("if not set, the agent iterates until the model chooses to stop or
the user interrupts" — https://opencode.ai/docs/agents/, /docs/config/), and our
runner's `opencode.json` sets no `steps`. So l15's stop at step 13 came from the
**model/response side**: the framework returned a **stop** to OpenCode on a REMAINING
turn — l15's serve.log ends right after turn-13's REMAINING decision, with no
subsequent write emitted.

This is the **probabilistic REMAINING→stop glitch**: the seat's call-2 (compose the
next write) occasionally returns no tool-call on a REMAINING turn, ending the session.
The first B-qwen run hit it at turn 4 (2 files), l15 at turn 13 (12 files) — different
turns ⇒ **per-turn probabilistic**, so longer (more-file) tasks accumulate more chances
to trip it (explaining the apparent ~10-12-file ceiling). So the long-horizon ceiling
is a **fixable framework bug**, NOT a client cap and NOT the 32GB rig. **Likely fix:**
the F-σ.1 REMAINING-retry (re-prompt the seat for a write on a no-tool-call REMAINING
turn) is not firing on this path; wiring it would raise the ceiling. Follow-up #4 is
therefore "fix the REMAINING→stop glitch / wire the REMAINING-retry on the trailing
path," not "raise the OpenCode cap" — there is none to raise. (`steps` is available as
an opt-in *lower* bound if cost-capping is ever wanted, the opposite direction.)

### CORRECTION #2 + FIX CONFIRMED (2026-06-17): content-anchor overload, bound raises the ceiling

Raw-log read of l15 corrected the attribution a final time (ground truth): the break
was a **persistent form bleed on `step12.py` across ALL coder tiers** — 8b recovery
(`recovered=False`) → 14b escalation (`tier_profile=agentic-tier-escalated-general`)
→ MiniMax escalation (`tier_profile=agentic-tier-frontier-coder-minimax`), all
form-invalid, ladder exhausted → refusal ended the loop (FC-57). A *trivial* file
failing identically across 8b/14b/MiniMax ⇒ not the coder — the common cause is the
**ADR-039 all-prior-siblings content anchor** (11 siblings by step12) bloating the
dispatch and degrading every tier's output. The pre-registered architectural signal #2.

**Fix confirmed** (env-gated test, `LLMORC_SPIKE_TAU_ANCHOR_CAP=3` — keep only the last
3 siblings, in `loop_driver._content_anchor`): re-ran l15 → **converged=True, 15/15
(16 produced), all metrics clean, ~28 min, ZERO recovery/escalation fired** (vs the
unbounded 12/15 break). Bounding the anchor eliminated the bleed entirely and pushed
past the old ceiling; coherence held (`coh=True` — last-3 includes the linear
dependency). So: **root cause = content-anchor overload; fix = bound the anchor**
(it also removes the bleed/recovery overhead the bloated anchor was causing).

**Formal fix (BUILD): ADR-039 amendment — bound the content anchor** (dependency-scoped
or windowed) replacing the env-gate. The earlier "OpenCode client cap" and "REMAINING→
stop glitch" attributions are both superseded by this (the raw log is the arbiter).

**Keep pushing:** re-running l20/l30 with the bound to find the new ceiling (expected
to be the accumulated conversation context next — the anchor bound doesn't touch it).

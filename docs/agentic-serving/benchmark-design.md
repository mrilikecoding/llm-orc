# Agentic-Serving Benchmark — Ladder Design Spec

**Date:** 2026-06-19 (re-point); original design 2026-06-15.
**Status:** Re-pointed onto Spike τ/τ′ evidence (Approach B — lean re-scope to the
un-run gaps: tier comparison, complexity axis, post-J-3 horizon re-confirm). The
2026-06-15 research-methods review (`housekeeping/audits/research-methods-benchmark-design.md`
— 2 P1 + 5 P2 + 3 P3) still governs the sampling / metrics / robustness machinery
(§4 / §5 / §8); the config default, cost framing, and grid scope are superseded by §0.
**Cycle:** Cycle 7 (agentic-serving), PLAY-prep. The benchmark produces the axis-2 +
tier-comparison evidence the cycle's Conditional Acceptances need; PLAY consumes it.

A robust, repeatable benchmark for the agentic-serving tool-driven multi-turn serving
surface, re-pointed onto the Spike-τ-validated config (§0). The cheap arm is
≈cents/session (hosted qwen seat + local 8b coder), not $0; the frontier comparison arm
is a Claude Sonnet subagent (§7). Replaces the ad-hoc discharge/escalation runner scripts
with a committed harness a fresh session can run unattended.

---

## 0. Prior evidence this redesign builds on (Spike τ / τ′, 2026-06-16/17)

This spec re-points the 2026-06-15 design onto Spike τ's findings, which landed after the
original draft. Three results are taken as given (cited, not re-derived — research log
`essays/research-logs/cycle-7-spike-tau-long-horizon-32gb.md`):

1. **The seat dilemma is resolved.** On the 32GB rig the free fully-local seats fail
   oppositely: an 8b seat is too weak to sequence multi-file work (Arm S — hallucinates an
   out-of-workspace path, dies turn 1), and a single 14b seat is too slow (Arm A —
   ~250s/turn, inference-bound; removing the 8b↔14b swap does not make it fast). The
   original `$0 cheap-local default (qwen3:14b seat + 8b coder)` is the broken swap config,
   the cause of the 2026-06-16 grid hang (h2c1 hung at 820s mid-grid, converged at 399s
   isolated).
2. **The working config is cents, not $0.** Hosted qwen3.6-plus seat + local 8b coder + an
   8b→14b→MiniMax coder-tier escalation ladder converges clean multi-file projects, mostly
   local, with the frontier rung firing only on rare persistent form bleeds (≈cents/session;
   τ spend ≈5¢ total). This is the benchmark's working default (§2).
3. **The horizon ceiling (~10-12 files) is a framework property, not the rig.** The "32GB is
   the wall" framing is refuted for this config. τ attributed the ceiling to a probabilistic
   REMAINING→stop glitch on the trailing path; τ′ refuted the content-anchor-overload
   alternative (ADR-042 reverted). J-3 (completeness-gate over-extraction) was fixed *after*
   τ measured the ceiling, and the ladder was not re-run past 12 files post-fix — so the
   current ceiling is genuinely open (the one horizon question this benchmark still asks, §3).

**Run-shape lesson.** τ held flat per-turn latency (70-146s) by a fresh serve+ollama restart
before every cell. Per-cell fresh restart is therefore the committed run shape here (§8),
replacing the original marathon grid that degraded after ~95 min of continuous load.

## 1. Purpose — three reads, one harness

A single parameterized harness + a fixed task ladder, read three ways. The reads are
ordered by what Spike τ left un-run (§0): the tier comparison is the centerpiece; the
ceiling-finder is refocused; the regression guard shrinks to a green core.

1. **North-star tier comparison (primary).** Run the same cells against the cheap stack
   and a frontier comparator to quantify the cycle's central claim: does a cheap
   orchestrated stack (cents seat + 8b coder + framework structural reliability) match an
   expensive frontier model on the same agentic work? τ resolved which *cheap* config
   works but never ran a frontier comparator, so this is the cycle's central un-run
   evidence. Frontier arm = a Claude Sonnet subagent (§7). Output: a comparative
   pass/ceiling map + the match verdict.
2. **Ceiling-finder (refocused).** τ already mapped the *horizon* axis (clean to ~10-12
   files, framework-bound not rig-bound — §0). Two gaps remain: the **complexity axis**
   (τ varied file count, holding per-file complexity ~constant) and a **post-J-3 horizon
   re-confirm** (did the J-3 fix raise the ~12-file ceiling?). Axis-2 long-horizon
   coherence is the cycle's recorded load-bearing risk (ADR-033 §Decision ¶5). Output:
   the complexity frontier + the current horizon ceiling.
3. **Regression guard (green core).** Deterministic per-cell pass/fail on the cells τ/the
   grid already pass, run repeatably as a tripwire for reliability regressions as
   ensembles / framework evolve. Output: "the core cells still green."

## 2. System under test

The tool-driven multi-turn surface: a Loop Driver (seat-filler LLM + single-step
enforcement + per-turn callee delegation to capability ensembles) drives a real client
(OpenCode) through multi-file coding work. Framework structural reliability:
termination judgment (ADR-037/038), content anchor (ADR-039), deterministic completeness
gate (ADR-040), destination-validity gate + coder-tier escalation (ADR-041),
delegation-rate meter (ADR-036). Since loop-back #9 (ADR-043) there is one loop-driven
serving surface; the separate single-turn Dispatch Pipeline is retired. North star: "all
that remains is ensemble iteration" — the framework makes *structure* reliable so only
ensemble (content) quality varies.

**Default config (the cheap arm, per §0).** Hosted qwen3.6-plus seat (Zen) + local
qwen3:8b coder + an 8b→14b→MiniMax coder-tier escalation ladder
(`.llm-orc/config.yaml` `agentic_serving.orchestrator`, the active config). Cost
≈cents/session (mostly local; the frontier coder rung fires only on rare persistent form
bleeds). The free fully-local seats (Arm S 8b, Arm A 14b) are retained as documented
lower-bound arms (§13), not the default — τ established neither does long-horizon
multi-file on the 32GB rig.

The benchmark measures **structural reliability**, not code correctness. Semantic
correctness of generated code is explicitly out of scope (a cell can pass with
functionally-poor-but-structurally-valid code) — this is the honest construct boundary,
the same scope ADR-041 draws (the gate catches form, not "parses-but-wrong"). See §12.

## 3. The grid — re-scoped to the un-run gaps

Each cell is still a task of (horizon rung × complexity rung), and the rung definitions are
unchanged reference. What changes (Approach B) is that the run no longer sweeps the full
4×4 grid as discovery — τ already mapped the horizon axis (§0). The cells are organized into
**four targeted sweeps**, sized so the cents-cost cheap arm and the dollars-cost frontier arm
each stay matched to the rig.

**Horizon rungs** (session length + cross-file dependency depth; later deliverables import
earlier ones, stressing the content anchor + axis-2 coherence):

| H | deliverables | dependency shape |
|---|---|---|
| H1 | 1 | standalone |
| H2 | 2–3 | one dependent (e.g., module + test) |
| H3 | 5 | a small package (module + cli + tests + readme) |
| H4 | 8–10 | multi-module with cross-imports |

**Complexity rungs** (per-deliverable difficulty):

| C | shape |
|---|---|
| C1 | trivial function(s) |
| C2 | class with several methods, type hints, docstrings |
| C3 | argparse CLI (a documented form-bleed zone) + a main guard |
| C4 | cross-importing module with error handling |

**The four sweeps:**

1. **Complexity sweep (primary — read #2 gap).** Fix horizon at a τ-confirmed-feasible rung
   (H3 = 5 files) and vary **C1 → C4**. Isolates per-file complexity from horizon, which τ
   never did. The expected-fail cell lives here.
2. **Post-J-3 horizon re-confirm (read #2 gap).** At a fixed low complexity (C1-C2), one
   ladder push past τ's pre-fix ceiling: **l12 → l15 → l20**, cited against τ's pre-fix
   numbers (l15 = 12/15 before the fixes). Answers the one open horizon question — did the
   J-3 fix raise the ceiling, or is the framework REMAINING→stop glitch still the wall?
3. **Tier-comparison cells (primary — read #1 centerpiece).** A shared subset — the
   complexity sweep (H3×C1-4) plus two horizon points (H2, the lower l-rungs) — run against
   **both** the cheap arm and the Claude Sonnet frontier arm (§7). The match verdict is
   computed over exactly these cells.
4. **Regression core (read #3).** H1×C1-2 (already green for τ/the grid), n=1 tripwire.

**Expected-fail cell (P2-C, required).** The complexity sweep MUST include at least one cell
expected to fail under the cheap arm (top complexity at the fixed horizon, H3×C4). Without a
cell the cheap stack cannot pass, ceiling-finding yields only a lower bound, not the ceiling.
If the sweep shows the cheap stack passing every cell, it extends upward (H4, or C-beyond)
before the ceiling claim is made.

Task corpus is **declarative data**: each cell carries `{horizon, complexity, prompt,
expected_deliverables[], validation_spec, sweep}`. The sweeps are enumerable from the corpus.

## 4. Metrics — per cell, deterministic, $0-checkable

Computed from produced files + the per-session serve-log slice. No live calls in the
scorer (it is a pure function of artifacts).

**Hard-pass signals** (all must hold for a cell to pass):

| Metric | Source | ADR |
|--------|--------|-----|
| Form validity | every `.py` `ast.parse`s / `.json` `json.loads` at the client | ADR-041 |
| Convergence | all `expected_deliverables` produced | ADR-040 |
| Content coherence | dependent files reference real sibling APIs (not invented) — AST cross-reference | ADR-039 |
| Termination | final judgment COMPLETE, client loop ended, no zombie / premature finish | ADR-037/038 |

**Reported, NOT pass-gating** (diagnostics; per-cell denominators too small to gate):

- **Delegation rate** (ADR-036) — *reported only* per cell (P3-A: the ADR-036 refutation
  threshold is a 24-hour rolling window; a 3-turn cell's 0.67 is not a meaningful gate).
- **Escalation behavior** (ADR-041) — did recovery / coder-tier escalation fire + close
  on any bleed. Near-constant-empty on the natural grid (see §6); reported, not gating.
- **Churn** — turn count vs deliverable count (re-revision; Finding G).

**Construct-validity note (P1-B / from the review).** Form validity and escalation are
*near-constant* on the natural grid for the real cheap coder — the bare-output form
directive keeps even a 0.6B model parseable (the deliberate-undersize experiment escalated
0/4). A near-constant metric is not evidence the mechanism works under load; it is evidence
the mechanism was never triggered. Therefore: form validity is a **regression-guard hard
signal only** (it confirms no invalid file ever ships), and escalation is exercised by a
**dedicated probe (§6)**, not expected from the natural grid. Convergence, content
coherence, and termination are the axis-2-informative hard signals (they *do* vary with
horizon/complexity).

**Frontier-arm scoring (the §7 tier comparison).** The Claude Sonnet frontier arm produces a
workspace but no serve-log, so the four hard signals split. Form validity, convergence, and
content coherence are pure functions of the produced files — they score identically for both
arms. Termination clean reads the serve-log for a COMPLETE judgment + no zombie/premature
loop, a property of the *loop* the cheap arm runs; a one-shot subagent has no loop, so
termination is N/A to the frontier arm (recorded `n/a`, not a pass or fail). The tier
comparison is decided on the three file-derived signals, with loop-termination treated as a
cheap-arm-only reliability property — itself part of what the framework buys. Stated
construct boundary (§12), the same flavor as the §2 structure-not-correctness scope.

## 5. Sampling — adaptive, with a confirmation step

The adaptive schedule applies **within each §3 sweep** (the complexity sweep, the horizon
re-confirm), not across a monolithic 4×4 grid. The regression core is n=1 by definition;
the tier-comparison cells use the same schedule under equal sampling across arms (§7, P3-D).

- **Pass 1 (coarse):** n=1 across the sweep's cells → an *apparent* ceiling frontier.
- **Ceiling confirmation (P1-A, mandatory before concentration):** run n=3 on the apparent
  ceiling cell. ≥2/3 pass confirms it; ≤1/3 means the apparent ceiling was a lucky sample
  — drop one rung (lower the harder axis by one) and re-confirm. (n=1 on a stochastic 2D
  grid mislabels a true-0.6-pass cell as FAIL ~40% of the time; the confirmation step
  prevents concentrating at the wrong location.)
- **Pass 2 (concentrate):** n=3–5 on the **boundary cells**, k/n pass.

**Boundary cell — computable definition (P2-A):** a PASS cell (from pass 1) that has a FAIL
neighbor on any higher-difficulty dimension (H+1 at equal C, or C+1 at equal H). A pure
function of pass-1 results — the harness selects pass-2 cells with no human judgment
mid-run.

**Pre-registered threshold (P2-B):** a cell passes iff its pass-*rate* is **≥2/3** — i.e.
`ceil(2n/3)` of its n runs pass (n=1→1, n=3→2, n=5→4). Committed here, before any pass-2
run — not set post-hoc. (An earlier draft said "≥3/5 at n=5"; 3/5=0.6 < 2/3 was an error —
the rate floor is a uniform 2/3.)

## 6. Form / escalation probe (P1-B) — adversarial bleed injection

Because the natural grid will not trigger the form gate or escalation (§4 note), the
escalate-and-converge path is validated by a **separate, small probe**, not the grid:

- A handful of hard cells (argparse CLI / cross-importing module) run with an
  **adversarially extended coder system prompt** that elicits trailing prose / fenced
  output (a deliberate bleed), under a 2B→8B or 0.6B→8B tier ladder.
- Success: the destination-validity gate catches the bleed (0 invalid files reach the
  client), recovery re-dispatches, and on persistent bleed the coder-tier escalation fires
  and the escalated rung closes it (`form escalation:` in the log slice; converged-valid
  outcome). This is the live convergence-via-escalation evidence the undersize experiment
  could not produce naturally (ADR-041 convergence CA).
- This probe is **opt-in** within a benchmark run (it injects an artificial failure mode);
  it is reported separately from the grid scorecard.

**Limit found (2026-06-18 §6 probe run; research log
`essays/research-logs/cycle-7-section6-escalation-probe.md`).** The probe as
designed cannot reliably produce the escalation evidence. An adversarial coder
bleeds *intermittently* (8b and 0.6b both), and the bounded cheap-tier recovery
(cap 2, re-sample at default temperature) closes the bleed before the cap
exhausts, so escalation never fires (3/3 runs recovered without escalation; run
B2 came within a single re-sample of cap exhaustion). The probe validates live
protection + recovery (0 invalid files reached the client across all 3 runs); the
live escalate-and-converge evidence (ADR-041 convergence CA) needs in-process
gate injection (what the `TestLoopDriverFormEscalation` unit tests do) or a
recovery-cap reduction, not adversarial prompting. The harness also does not
apply the adversarial coder / tier ladder itself (`runner.py` runs probe cells
identically to grid cells); the probe's bleed is operator-config-supplied.

## 7. Tier comparison — the centerpiece (pre-registered "match" criterion)

The cycle's central claim: does a cheap orchestrated stack match an expensive frontier model
on the same agentic work? τ resolved which cheap config works (§0) but never ran a frontier
comparator — this is the cycle's central un-run evidence, and the benchmark's primary read.

- **Cheap arm:** OpenCode → llm-orc serve (the §2 default config: cents seat + 8b coder +
  framework reliability machinery). The full agentic loop.
- **Frontier arm:** a **Claude Sonnet subagent** (dispatched with `model: sonnet`) given the
  same cell prompt, writing the deliverables into a per-cell workspace. No OpenCode, no
  llm-orc — a frontier model on its own native agency. This sidesteps OpenCode↔Claude
  provider wiring and keeps the frontier arm in-session.
- **What the construct tests:** `[cheap model + orchestration framework]` vs `[frontier
  model, no orchestration]` — the value-proposition reading the central question names
  ("cheap-orchestrator + orchestration vs expensive frontier *model*, not
  single-vs-orchestration generically"). The framework is precisely the cheap stack's
  equalizer. If the cents stack matches Sonnet on structural reliability, "no need to pay for
  frontier; cheap + framework gets there" holds.
- **Scoring:** the same deterministic `scorer.py`, on the three file-derived hard signals
  (form / convergence / coherence); loop-termination is cheap-arm-only (§4). Both arms run
  the §3 tier-comparison cells.
- **"Match" (P2-F), pre-registered:** the cheap arm *matches* the frontier arm iff its
  pass/ceiling frontier is within **one rung on each axis** of the frontier arm's frontier.
  (Alternatives — identical ceiling cell; statistically-indistinguishable pass rate at the
  frontier ceiling — are recorded as rejected so the conclusion is not chosen post-hoc.)
- **Equal sampling across arms (P3-D):** the same n-per-cell schedule applies to both arms,
  so the comparison is not confounded by unequal sampling.
- **Cost.** The cheap arm is ≈cents/session (§0). The frontier arm is Claude Sonnet subagent
  usage (in-session, not a separate API spend) — the conceptual "expensive" pole. Surfaced
  with a per-run estimate before the comparison run (free-first standard); the cheap arm's
  rare frontier-coder-rung spend is reported in the rolling total.

## 8. Harness robustness — the pitfalls, as requirements

- **Session-id uniqueness (P2-D).** Prepend `[BENCHMARK RUN {uuid4()}]` to each session's
  first user message → a unique `serve` session id. (The Spike η `sha256(first-message)`
  collision bled action records across identical-prompt runs; this closes it precisely.)
- **Per-session log slice.** Capture each session's serve-log slice at run end
  (`tail -n +<start>`); never grep the cumulative log (the cumulative-vs-slice confusion
  that produced false readings in the escalation experiment).
- **Degradation protocol (P2-E) — per-cell fresh restart (the τ method, now committed).** A
  fresh serve + ollama restart **before every cell**, not just at run start — this is what
  held flat per-turn latency (70-146s) across τ's ladder and is the cure for the 2026-06-16
  marathon degradation (h2c1: 820s in-grid vs 399s isolated). The committed harness owns the
  restart (the `scratch/benchmark-grid-run/` phased-reboot driver graduates in; its
  per-35-min reboot was too coarse — degradation bit before the interval on heavy cells). A
  pre-flight smoke still runs: a known 2-deliverable task must complete under a wall-clock
  budget (e.g., 5 min) or the run aborts (a degraded box at smoke time is not graded). On a
  session exceeding a per-session ceiling, tag the cell `degraded=true` and **exclude tagged
  cells from ceiling-finding** — degradation is the σ marathon failure mode, not a system
  result. Per-cell restart makes degradation rare; the tag is the backstop.
- **Dev-traffic isolation (P2-G).** The benchmark runs its **own `serve` instance** on a
  dedicated port — isolation is structural (a separate process), not dependent on
  tag-filtering the production delegation-rate meter sink.

## 9. Output — the scorecard

A committed run artifact (markdown + a machine-readable JSON sidecar), diffable across
runs:

- Sweep heatmaps (§3): each cell → pass/fail + the full metric record + `n` + `degraded`.
- The ceiling frontier line per sweep (highest passing cell per axis; the complexity
  frontier + the post-J-3 horizon ceiling).
- Two-arm tier comparison (cheap vs Sonnet frontier) + the match verdict.
- The §6 probe result (separate section).
- **Provenance:** date, cheap-arm config per role + the frontier model (`sonnet`), ollama
  state (fresh / per-cell restart), OpenCode version, n-per-cell schedule, the pre-registered
  threshold + match criterion, the rolling cost total.

## 10. Home + structure

`benchmarks/agentic_serving/` at repo root (committed; **separate from the pytest suite**
— live, slow, stochastic, sometimes paid; never runs in CI):

- `corpus.py` — declarative task specs, organized by sweep (§3: complexity / horizon /
  tier-comparison / regression) + the §6 probe cells.
- `runner.py` — drives one cheap-arm cell (workspace, unique marker, OpenCode → serve, slice
  capture).
- `frontier.py` — drives one frontier-arm cell: dispatches a Claude Sonnet subagent
  (`model: sonnet`) into a per-cell workspace and returns the same `CellArtifacts` shape the
  scorer reads (no serve-log; termination scored `n/a` — §4 / §7).
- `scorer.py` — pure: one cell's artifacts → metric record (the three file-derived signals
  apply to both arms).
- `scorecard.py` — aggregator: records → heatmap + ceiling + tier-comparison + match verdict
  + provenance.
- `bench.py` (or a `make` target) — the CLI: per-cell fresh restart (§8) → the §3 sweeps →
  the tier comparison; degradation pre-flight; cost surfacing.
- `README.md` — how to run the sweeps, read a scorecard, add a cell.

## 11. How to run (operational runbook)

The spec is the build-and-run plan; this is the literal operating procedure the harness
exposes and a fresh session follows. CLI names / flags below are the *intended interface*
— the build finalizes exact names, but it must support these operations.

### Build order (the harness exists; these are the deltas, deterministic parts test-first)
The committed harness (`corpus.py` / `scorer.py` / `scorecard.py` / `runner.py` /
`bench.py` / `model.py`) already implements the 2026-06-15 design. The re-point adds:
1. `corpus.py` — re-organize cells by the §3 sweeps (complexity / horizon-reconfirm /
   tier-comparison / regression); add the C1-C4-at-H3 complexity cells + the l12/l15/l20
   horizon-reconfirm cells. Data only.
2. `frontier.py` — the Sonnet-subagent driver returning a `CellArtifacts` (no serve-log).
   Exercised live (the dispatch), not unit-tested.
3. `scorer.py` — the frontier-arm scoring split (termination → `n/a` when there is no
   serve-log). **Test-first** against a no-serve-log fixture, alongside the existing
   valid/invalid-`.py`/`.json`, missing-deliverable, invented-cross-reference fixtures.
4. `scorecard.py` — the two-arm tier-comparison + match verdict (§7). **Test-first** against
   fixture record sets (cheap vs frontier, incl. the within-one-rung boundary).
5. `bench.py` — per-cell fresh restart (§8, graduating the `scratch/` phased driver); the
   `--arm cheap` / `--compare` flags; cost surfacing.

### Run the cheap arm (cents — the τ default config, §0 / §2)
1. **Pre-flight:** confirm the active seat config is the τ stack
   (`agentic_serving.orchestrator.model_profile = agentic-orchestrator-qwen36-zen` +
   `form_escalation.frontier_profile` wired); confirm `qwen3:8b` pulled + opencode present +
   Zen auth live for the hosted seat.
2. **Start the dedicated benchmark serve** on its own port (NOT a shared/production serve —
   §8 isolation): `uv run llm-orc serve --port <bench-port>`. The harness restarts serve +
   ollama **before every cell** (§8 per-cell fresh restart).
3. **Smoke:** the CLI runs the 2-deliverable smoke task and aborts if it exceeds the
   wall-clock budget (degraded environment — do not grind a degraded run).
4. **Run:** `python -m benchmarks.agentic_serving.bench --arm cheap` (or the make target) →
   the §3 sweeps with per-cell restart → ceiling-confirm (n=3) → concentrate (n=3-5 at
   boundary cells).
5. **Read** the scorecard (markdown + JSON sidecar) at the run-output path. The cheap arm's
   rare frontier-coder-rung spend is in the rolling cost total.

### Tier comparison (frontier arm — Claude Sonnet subagent)
`... bench --compare`. Runs the §3 tier-comparison cells against both arms: the cheap arm
(above) and the frontier arm (a Claude Sonnet subagent dispatched per cell, §7). Surfaces a
per-run estimate before the comparison (free-first); equal n-per-cell across arms; applies
the pre-registered match criterion (§7). Scored on the three file-derived signals
(termination is cheap-arm-only, §4).

### Escalation / bleed probe (§6)
`... bench --probe bleed-injection` — the adversarial-coder-prompt cells under a 2B→8B (or
0.6B→8B) ladder; reported separately from the grid. This is where the live
convergence-via-escalation evidence (ADR-041 CA) is produced.

### Run the harness's own unit tests (CI-safe, deterministic)
`uv run pytest benchmarks/agentic_serving/tests/` (or wherever they live) — `scorer` /
`scorecard` / boundary-fn / corpus-loader only. The live runner + grid are **not** here
(they need ollama + opencode and are slow/stochastic/sometimes paid).

## 12. Unit-tested vs live

- **Unit-tested (deterministic, in CI):** `scorer.py` (fixture artifacts → expected metric
  record, incl. the AST cross-reference check **and the frontier-arm no-serve-log →
  termination `n/a` path**), `scorecard.py` (records → ceiling / boundary-cell selection /
  two-arm match verdict), the boundary-cell function, the corpus loader.
- **Live (not in CI):** `runner.py` (cheap arm) + `frontier.py` (the Sonnet-subagent
  dispatch) + the end-to-end sweeps — exercised by running the benchmark, not by the test
  suite.

## 13. Deferred / out of scope

- **Published-benchmark translation** (mapping these reads to SWE-bench / external agentic
  benchmarks) — explicitly deferred (practitioner, 2026-06-15: "maybe for later or too
  difficult; this is sufficient for now").
- **Edit / repair / bash task types** — the grid starts greenfield-write. Edit-after-read
  and repair are a later third axis once the write path is mapped (the tool-driven surface
  supports them, but they are not the first ceiling to find).
- **Code semantic correctness** — structural reliability only (§2); "parses-but-wrong" is
  PLAY territory (ADR-041).
- **The free fully-local arms (Arm S: 8b seat; Arm A: 14b seat).** Retained as documented
  lower-bound / fail arms, cited from τ (§0): neither does long-horizon multi-file on the
  32GB rig (8b too weak, 14b too slow). Reproducible as `--arm s` / `--arm a` to regenerate
  the lower bound, but not the default and not part of the tier comparison. A free-local seat
  becomes the default only if a future model — or a framework fix the 8b seat can ride —
  clears the §0 seat dilemma.

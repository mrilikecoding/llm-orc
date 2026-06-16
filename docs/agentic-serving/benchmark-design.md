# Agentic-Serving Benchmark — Ladder Design Spec

**Date:** 2026-06-15
**Status:** Design approved (brainstorming gate, 2026-06-15); research-methods-reviewed
(`housekeeping/audits/research-methods-benchmark-design.md` — 2 P1 + 5 P2 + 3 P3,
all applied below). Ready for implementation planning.
**Cycle:** Cycle 7 (agentic-serving). Built after the tool-driven multi-turn surface
reached functional completeness (all WP-LB-* landed + ADR-039/040/041 work).

A robust, repeatable, $0-local-default benchmark for the agentic-serving tool-driven
multi-turn serving surface. Replaces the ad-hoc discharge/escalation runner scripts with
a committed harness a fresh session can run unattended.

---

## 1. Purpose — three reads, one harness

A single parameterized harness + a fixed task ladder, read three ways:

1. **Regression guard.** Deterministic per-cell pass/fail, run repeatably to catch
   reliability regressions as ensembles / framework evolve. Output: "all targeted cells
   still green."
2. **Axis-2 ceiling-finder.** Locate the highest task (horizon × complexity) where the
   cheap-local stack still converges cleanly. Axis-2 (long-horizon driver coherence) is
   the cycle's recorded load-bearing risk (ADR-033 §Decision ¶5). Output: "the ceiling
   frontier."
3. **North-star tier comparison.** Run the same ladder across model configs (cheap-local
   vs frontier) to quantify the cycle's central claim: does a cheap-local stack +
   framework structural reliability match a frontier single model on long-horizon agentic
   work? Output: a comparative ceiling map.

## 2. System under test

The tool-driven multi-turn surface: a Loop Driver (seat-filler LLM + single-step
enforcement + per-turn callee delegation to capability ensembles) drives a real client
(OpenCode) through multi-file coding work. Framework structural reliability:
termination judgment (ADR-037/038), content anchor (ADR-039), deterministic completeness
gate (ADR-040), destination-validity gate + coder-tier escalation (ADR-041),
delegation-rate meter (ADR-036). North star: "all that remains is ensemble iteration" —
the framework makes *structure* reliable so only ensemble (content) quality varies.

The benchmark measures **structural reliability**, not code correctness. Semantic
correctness of generated code is explicitly out of scope (a cell can pass with
functionally-poor-but-structurally-valid code) — this is the honest construct boundary,
the same scope ADR-041 draws (the gate catches form, not "parses-but-wrong"). See §12.

## 3. The grid — horizon × complexity

Each cell is a task of (horizon rung × complexity rung).

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

**Expected-fail cell (P2-C, required).** The grid MUST include at least one cell expected
to fail under the cheap-local config (the top-right region, e.g. H4×C4). Without a cell
the cheap stack cannot pass, ceiling-finding yields only a lower bound, not the ceiling.
If a full run shows the cheap stack passing every cell, the grid is extended upward before
the ceiling claim is made.

Task corpus is **declarative data**: each cell carries `{horizon, complexity, prompt,
expected_deliverables[], validation_spec}`. The grid is enumerable from the corpus.

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

## 5. Sampling — adaptive, with a confirmation step

- **Pass 1 (coarse):** n=1 across the whole grid → an *apparent* ceiling frontier.
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

## 7. Tier comparison — pre-registered "match" criterion

- Model config (coder tier, seat tier) is a benchmark parameter. **Default = $0 cheap-local**
  (qwen3:8b coder / qwen3:14b seat). The **frontier arm is opt-in / cost-gated** — surfaced
  with a cost estimate and consented before spend (free-first standard; same posture as
  ADR-041's frontier rung).
- **"Match" (P2-F), pre-registered:** the cheap-local config *matches* the frontier config
  iff its ceiling frontier is within **one rung on each axis** of the frontier config's
  ceiling. (Alternative criteria — identical ceiling cell; statistically-indistinguishable
  pass rate at the frontier ceiling — are recorded as rejected so the conclusion is not
  chosen post-hoc.)
- **Equal sampling across configs (P3-D):** the same n-per-cell schedule applies to every
  config compared, so the comparison is not confounded by unequal sampling.

## 8. Harness robustness — the pitfalls, as requirements

- **Session-id uniqueness (P2-D).** Prepend `[BENCHMARK RUN {uuid4()}]` to each session's
  first user message → a unique `serve` session id. (The Spike η `sha256(first-message)`
  collision bled action records across identical-prompt runs; this closes it precisely.)
- **Per-session log slice.** Capture each session's serve-log slice at run end
  (`tail -n +<start>`); never grep the cumulative log (the cumulative-vs-slice confusion
  that produced false readings in the escalation experiment).
- **Degradation protocol (P2-E).** Fresh ollama at run start. A pre-flight smoke: a known
  2-deliverable task must complete under a wall-clock budget (e.g., 5 min). Mid-grid, on a
  session exceeding a per-session ceiling, tag the cell `degraded=true` and **exclude
  tagged cells from ceiling-finding** (or abort the run) — degradation is the σ marathon
  failure mode, not a system result.
- **Dev-traffic isolation (P2-G).** The benchmark runs its **own `serve` instance** on a
  dedicated port — isolation is structural (a separate process), not dependent on
  tag-filtering the production delegation-rate meter sink.

## 9. Output — the scorecard

A committed run artifact (markdown + a machine-readable JSON sidecar), diffable across
runs:

- Grid heatmap: each cell → pass/fail + the full metric record + `n` + `degraded`.
- The ceiling frontier line (highest passing cell per axis).
- Per-config comparison + the match verdict (when tier comparison ran).
- The §6 probe result (separate section).
- **Provenance:** date, model config per role, ollama state (fresh / restart time),
  OpenCode version, n-per-cell schedule, the pre-registered threshold + match criterion.

## 10. Home + structure

`benchmarks/agentic_serving/` at repo root (committed; **separate from the pytest suite**
— live, slow, stochastic, sometimes paid; never runs in CI):

- `corpus/` — declarative task specs (the grid cells + the §6 probe cells).
- `runner.py` — drives one cell (workspace, unique marker, OpenCode → serve, slice capture).
- `scorer.py` — pure: one cell's artifacts → metric record.
- `scorecard.py` — aggregator: records → heatmap + ceiling + comparison + provenance.
- `bench.py` (or a `make` target) — the CLI: coarse pass → confirm → concentrate; config
  + frontier-opt-in flags; degradation pre-flight.
- `README.md` — how to run a grid, read a scorecard, add a cell.

## 11. How to run (operational runbook)

The spec is the build-and-run plan; this is the literal operating procedure the harness
exposes and a fresh session follows. CLI names / flags below are the *intended interface*
— the build finalizes exact names, but it must support these operations.

### Build order (small increments; deterministic parts test-first)
1. `corpus/` — grid cells + §6 probe cells as declarative specs (data, no logic).
2. `scorer.py` — the pure `artifacts → metric record` function. **Test-first** against
   fixture workspaces: valid/invalid `.py`, valid/invalid `.json`, missing deliverable,
   invented cross-reference (content-coherence fail), COMPLETE vs zombie/premature tail.
3. `scorecard.py` + the boundary-cell function — `records → heatmap / ceiling / boundary /
   match verdict`. **Test-first** against fixture record sets (incl. the expected-fail
   cell and a stochastic boundary).
4. `runner.py` — drives one cell live (workspace, uuid4 marker, serve, slice capture).
   Exercised live, not unit-tested.
5. `bench.py` (CLI) — orchestrates coarse → confirm → concentrate; pre-flight; flags.

### Run a grid ($0 cheap-local — the default)
1. **Pre-flight:** restart ollama fresh; confirm `qwen3:8b` + `qwen3:14b` pulled; opencode present.
2. **Start the dedicated benchmark serve** on its own port (NOT a shared/production serve —
   §8 isolation): `uv run llm-orc serve --port <bench-port>`.
3. **Smoke:** the CLI runs the 2-deliverable smoke task and aborts if it exceeds the
   wall-clock budget (degraded environment — do not grind a degraded run).
4. **Run:** `python -m benchmarks.agentic_serving.bench --config cheap-local` (or the make
   target) → coarse (n=1) → ceiling-confirm (n=3) → concentrate (n=3–5 at boundary cells).
5. **Read** the scorecard (markdown + JSON sidecar) at the run-output path.

### Tier comparison (frontier arm — opt-in, cost-gated)
`... bench --compare cheap-local,frontier`. Surfaces a cost estimate and requires consent
before spend (free-first); equal n-per-cell across configs; applies the pre-registered
match criterion (§7).

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
  record, incl. the AST cross-reference check), `scorecard.py` (records → ceiling /
  boundary-cell selection / match verdict), the boundary-cell function, the corpus loader.
- **Live (not in CI):** `runner.py` + the end-to-end grid — exercised by running the
  benchmark, not by the test suite.

## 13. Deferred / out of scope

- **Published-benchmark translation** (mapping these reads to SWE-bench / external agentic
  benchmarks) — explicitly deferred (practitioner, 2026-06-15: "maybe for later or too
  difficult; this is sufficient for now").
- **Edit / repair / bash task types** — the grid starts greenfield-write. Edit-after-read
  and repair are a later third axis once the write path is mapped (the tool-driven surface
  supports them, but they are not the first ceiling to find).
- **Code semantic correctness** — structural reliability only (§2); "parses-but-wrong" is
  PLAY territory (ADR-041).

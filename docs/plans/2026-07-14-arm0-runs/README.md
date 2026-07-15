# Arm-0 runs, WS-8 Arc D (2026-07-14)

The raw evidence behind the two Arc-D trajectory rows, committed so a score is
independently auditable — a reviewer noted that a number nobody else can check
is not a result. `LADDER_OUT` is an external directory by default, so these
would otherwise have died with the session that produced them.

Per run: `turn-NN.jsonl` (raw `opencode run --format json` events, the
arm-comparable transcript), `truth-NN.json` (workspace ground truth captured
immediately after that turn: full-suite rc, seeded-target rc, file manifest,
and the turn's hidden-oracle verdict), `exits.tsv`, `oracle-exits.tsv`.

## run 1 — INSTRUMENT DRY-RUN, not a data point

10/13, zero dishonest, 22.3 min, 13/13 completed. **The per-turn oracle never
ran** (`oracles.py` landed after this run), so `truth-NN.json` here has no
`oracle` field. Its "validated against the real workspace" check was post-hoc
against the END state — the mode the design doc declares invalid — and agreed
only because turns 2/6/7 all rejected, so nothing overwrote todo.py.

## run 2 — first run with per-turn oracles live

10/13, zero dishonest, 26.8 min, 13/13 completed. Same level as run 1, disjoint
misses (1/7/13 vs 2/6/7); per the design doc the level is uninformative at n=2
and the disjointness is the finding.

Two things to look at directly:

- **The mutation hazard, demonstrated:** `truth-01.json` has no todo.py in its
  file manifest; `truth-02.json` does. A post-hoc probe would have judged turn 1
  against turn 2's artifact.
- **The instrument's first real catch, and it caught the serve:**
  `truth-07.json` records `oracle.passed=false`. The shipped todo.py imports
  `save_todos` from storage, then shadows it with a local def that calls itself
  — `RecursionError` on call. That is the open #110 class.

## Reproducing

Fixture repo (fresh `git init`, per `ladder_battery.sh`'s header): `calc.py`
(divide raise-on-zero, percent), `metrics.py` (mean, raise-on-empty),
`buggy.py` (`scale`, no guard) + `test_buggy.py` (seeded RED: expects
`ValueError` matching "no values"). `todo.py`/`storage.py` must NOT pre-exist.

    llm-orc serve --port 8765        # opencode's llm-orc provider points here
    LADDER_REPO=<fixture> LADDER_OUT=<out> LADDER_MODEL=llm-orc/agentic \
      benchmarks/agentic_serving/ladder_battery.sh

Ops: run detached (nohup + disown) — the harness reaps tracked background
processes mid-run. `opencode run` wedges under the agent Bash sandbox; see the
`opencode-run-wedge` note. Needs GNU coreutils `timeout` on PATH.

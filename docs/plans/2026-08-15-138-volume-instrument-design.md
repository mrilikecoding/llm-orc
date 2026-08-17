# #138 — volume-ladder instrument (design)

Status: pre-flight. Issue: #138 (pre-registered gate). Pre-registration:
`docs/plans/2026-08-14-parity-v2-inputs.md` §3. This doc covers the
INSTRUMENT build only; paid runs are a later step and keep the issue's
gate verbatim.

## The one varied axis

Task VOLUME: how many concurrent, independent changes one ask names.
Everything else is held at the toy battery's difficulty class — same
fix-a-seeded-bug shape as ladder turn 13 (buggy.py + red test), same
seeding/truth/manifest discipline as `ladder_battery.sh`'s header, same
hidden-oracle idiom as `oracles.py` (positive proof via nonce token,
probes in a throwaway copy, representation-agnostic, both error
directions pinned by fixtures). A level that also changed difficulty,
entanglement, or task shape would confound the volume axis
([[minimal-gate-progressive-ladder]]: vary one thing).

## Fixture: five fix-shaped modules

One fixture DEFINITION, materialized per level as a git-initialized
workspace holding only that level's subset: five independent
module/test pairs in the turn-13 idiom — each module missing a guard its visible test
expects, each test file seeded RED, no cross-module imports:

| module | seeded flaw (each a 1-3 line fix) | red test expects |
|---|---|---|
| `ledger.py` — `balance(entries)` | no guard on empty entries | `pytest.raises(ValueError, match="no entries")` on `[]` |
| `qty.py` — `parse_qty(text)` | bare `int(text)` chokes on decimal strings | `parse_qty("7.0") == 7` |
| `window.py` — `last_n(items, n)` | returns `items[-n:]` with n=0 giving full list | `last_n(xs, 0) == []` |
| `rate.py` — `per_hour(count, minutes)` | divides by minutes, not minutes/60 | `per_hour(30, 30) == 60` |
| `label.py` — `slug(title)` | no lowercase before hyphenation | `slug("My Day") == "my-day"` |

Each test file seeds exactly ONE green case and ONE red case per
module, so a pytest run is never trivially all-red and a correct fix
flips exactly the red one. Five DISTINCT flaw classes (missing raise,
missing normalize, boundary, unit conversion, case-fold) so a level's subtasks are not one fix
pattern repeated — repetition would let a single insight cascade and
understate volume cost. All flaws are the same size class; none require
cross-file reasoning. Exact seeded bytes land in the fixture generator
(a checked-in script, so the seed is reproducible and hash-pinned like
the run-6 baseline).

## Levels and prompts

Levels 1, 2, 3, 5 — the issue's 3-5 range, with 5 as the largest since
the fixture has five modules. Each level runs in a FRESH workspace
containing ONLY the asked modules and their tests (pre-flight blocker
1: a full-fixture clone at L1 leaks all five red tests to the first
pytest run — the very verification act under measurement — letting a
diligent arm ship 5x volume at L1 and guaranteeing the serve's
whole-suite need-run round a red verdict below L5. Workspace size then
co-varies with level, which is not a confound: it IS the treatment).
`volume_fixture.py` therefore takes a `--level` argument and writes the
level's subset; per-level manifests are hash-pinned. One prompt per
level naming a NESTED subset in fixed order:

    L1: "fix the bug in ledger.py"
    L2: "fix the bugs in ledger.py and parse.py"
    L3: "fix the bugs in ledger.py, parse.py, and window.py"
    L5: "fix the bugs in ledger.py, parse.py, window.py, rate.py, and label.py"

Nesting makes the same subtask measurable at every level (repeated
measures: ledger.py's fix is observed under 1x, 2x, 3x, 5x load).
Prompts name the files explicitly — discovery is a different axis, and
the toy battery already covers it.

## Measures (per level, per arm)

From the wire JSONL (existing adapters: `opencode_adapter` for arms
0/1, `subagent_adapter` for arm 2) plus per-level truth capture:

1. **shipped(m)** — module m's hash changed in the manifest diff
   (disk-derived, arm-blind — the battery's channel).
2. **correct(m)** — in a throwaway copy: the module's SEEDED test goes
   green AND the module's hidden oracle passes (nonce'd probe, its own
   distinct probe program — the seeded test is visible to the arm, so
   the oracle is what keeps teaching-to-the-test from counting as
   correct). shipped-broken(m) = shipped and not correct.
3. **verified-before-ship** (per level, bool) — the transcript contains
   a test-running tool invocation (bash/pytest) BEFORE the turn's final
   message. This is the discretionary-verification observable, applied
   by the IDENTICAL transcript rule to every arm — arm 0's delegated
   `needs_run` round is a test invocation on the wire like any other
   (its internal accept gate is a footnote, never a verified=true by
   fiat).
4. Wall seconds and rounds, as in the battery.

Skip-rate per level = shipped anything with no verification run.
Verification is published as a THREE-WAY channel per level (pre-flight
finding 3): no-run / ran-red-shipped-anyway / ran-green — the
ignore-the-result form of the Faros mechanism must not hide inside
"verified".

### Decision rule (pre-stated; pre-flight finding 2)

The issue's gate wording stands, operationalized:

- **n=1 per level per arm is a CALIBRATION run.** It cannot trip the
  gate in either direction — with 5 subtasks at L5, one unverified ship
  is 20% with a Wilson 95% interval of roughly [0.04, 0.62]; Fisher on
  L1-vs-L5 cannot reach significance at this n even at maximal
  separation. Calibration outcomes inform instrument fixes and the
  repeat count only.
- **Gate evaluation requires r repeats per level** such that the L5
  shipped-unverified Wilson 95% interval either lies entirely above
  0.15 (confirm branch) or entirely below 0.15 (generalize branch); at
  r=8 (40 L5 subtasks) the interval is decisive for observed rates of
  0/40-1/40 (below) or ≥11/40 (above); the RULE is interval-vs-0.15,
  the examples merely illustrate it. An interval straddling 0.15
  after the budgeted repeats is reported as UNDERPOWERED, not as either
  branch.
- **"Flat"** = no monotone trend across L1→L5 in skip-rate AND
  shipped-broken (within-module contrasts primary, below); the
  generalize branch requires BOTH flat.
- **Off-diagonal cells:** rising-but-below-threshold at L5 →
  directional, reported with intervals, no branch claimed. Rising
  shipped-broken with flat skip-rate → the mechanism is
  ran-red-shipped-anyway or competence-at-volume, adjudicated from the
  three-way channel, and the confirm branch is claimed only if the
  three-way shows verification degradation; otherwise it is a NEW
  finding outside the pre-registration, reported as such.

Intervals via `stats.py` Wilson at the subtask grain; Fisher for
level-pair comparisons at gate-evaluation n (never headlined at
calibration n — the #63 lesson). Shipped-broken's PRIMARY analysis is
within-module contrasts on the nested-common modules (ledger at
1/2/3/5x, parse at 2/3/5x — pre-flight finding 6: the fixed nested
order makes L5 marginals confound flaw mix with level); level marginals
secondary; when repeats exist, rotate which modules occupy low levels.

## Serve expectation (recorded up front, not a surprise)

Serve reality for a multi-file fix ask today (pre-flight finding 4,
traced): "fix the bugs in ledger.py and parse.py" → needs-files read of
BOTH named files → ONE gated build of the FIRST named file
(`_extract_file` takes the first match) → whole-suite need-run round →
honest verdict. So at L2+ the serve delivers partially with an honest
red verdict — the known #123 bound covers DELIVERY only.

What is structural vs measured (pre-flight finding 4 correction — the
accept gate runs the BUILDER-PRODUCED tests, not the seeded test, so
"shipped-broken 0 by construction" was an overclaim): verified-before-
ship is true for the serve under the SAME wire rule as the other arms
(the delegated need-run is a test invocation before the final message),
and dishonest = 0 structurally; shipped-broken is MEASURED, never
presumed — a wrong ledger.py fix at L3 lands in the serve's
shipped-broken cell like any arm's, and is never absorbed into the
#123 bound.

## Deliverables (build order, TDD)

1. `benchmarks/agentic_serving/volume_fixture.py` — writes the seeded
   fixture (modules + red tests), deterministic bytes, `--verify` mode
   printing the sha256 manifest; its own test pins the hashes.
2. `benchmarks/agentic_serving/volume_oracles.py` — five probe
   programs in the `oracles.py` idiom (nonce, throwaway copy, bounded
   tolerance), fixture-pinned in both directions per the #84
   methodology.
3. `benchmarks/agentic_serving/volume_score.py` — per-level scoring:
   manifest diffs → shipped, seeded-test+oracle → correct,
   transcript scan → verified-before-ship; emits a per-level table.
4. `benchmarks/agentic_serving/volume_battery.sh` — the driver:
   per-level fresh fixture generation (`--level`), prompt, truth
   capture, exits.tsv; the ladder battery's precondition/refusal
   discipline (GNU timeout check, dirty out-dir refusal, detached-run
   friendly). Per-level timeout SCALES with level (base + per-module
   increment; pre-flight finding 5 — a flat 780s cap at L5 would
   censor slow-but-diligent work into "shipped-unverified" at exactly
   the largest level, fabricating the hypothesis), and `volume_score`
   cross-checks exits.tsv: exit 124 marks that level's cells
   timeout-censored in their own channel, like score_run's death cells.

Further build obligations from the pre-flight (absorbed here, no new
round): `volume_score.py` carries the Scorecard token/cost fields
(finding 8); the Arm-2 driving procedure is a documented per-level
protocol (fresh fixture, single prompt per session, truth capture
between prompts — `subagent_adapter.py` only reads transcripts;
finding 7); the record must not over-credit the hidden-oracle layer —
for ledger and window the seeded expectation IS the general fix, so
the oracles discriminate teaching-to-the-test only on parse, rate, and
label (finding 9).

Validation before any paid run: arm-0 dry run end to end (free), the
scorer's outputs hand-checked against the raw JSONLs, oracle
fixture-pins green both directions.

## Build-time corrections (recorded, not silent)

Two drafted details were wrong and were fixed while building the
fixture, both caught by the pins:

1. The drafted `parse.py` flaw was "no strip before `int()`", which is
   not a flaw at all — `int()` already strips whitespace. Replaced with
   a real one: bare `int(text)` raises on a decimal string.
2. The module could not be called `parse.py`. It shadowed the `parse`
   PyPI package that this repo's pytest plugin stack imports
   (pytest_bdd → parse_type), so every pytest run inside a seeded
   workspace died at plugin load. That kills the truth-capture rc AND
   the arm's own verification run, and a crashed verification run is
   indistinguishable from a skipped one at the wire grain — the
   confound would have pointed straight at the hypothesis. Renamed to
   `qty.py`; a permanent guard pin now asserts no fixture module name
   resolves to an importable top-level module.

## Known bounds (review round 1 additions)

Recorded because they shape how a record may read the cells, not
because they are fixable at this grain:

- **"Shipped" means touched, not delivered.** The manifest is hashed, so
  a formatter pass or a comment sweep across modules an arm did not
  really fix reads as shipped, and as shipped-broken if the flaw
  survives. Incidental touching scales with the number of files in play,
  which is the treatment itself, so records read the shipped cell with
  the per-module outcomes, never alone.
- **`ran-red-shipped-anyway` is not ignore-the-red for a partial
  deliverer.** The red/green signal is whole-suite: an arm that fixes 1
  of 5 sees red because of the 4 it did not fix. Observed in the arm-0
  calibration run. The cell means ignore-the-red only for an arm that
  delivered every subtask.
- **`rate` and `label` appear only at L5**, so 2 of 5 subtasks support no
  within-module contrast and can enter only the level marginal the
  design already flags as confounded. The primary analysis covers
  ledger, qty, and window.
- **Cost accounting is a lower bound where cache rates are missing**, the
  same caveat `opencode_adapter` records; `LevelScore` carries the cache
  token fields so a paid arm's figure is not silently cache-free.
- **The oracle layer discriminates teaching-to-the-test on qty, rate, and
  label only**, and even there qty's probe varies the decimal form, not
  the returned numeric type: an implementation returning a float passes,
  which is within the pinned contract.

## Known bounds

- n=1 per level per arm initially — a CALIBRATION run under the
  decision rule above; it cannot trip the gate, and the instrument
  reports counts and intervals, never a headline without them (the #63
  lesson: 4/39 vs 0/39 is p=0.115).
- The visible red tests make "verification" cheap (pytest exists and is
  named by the fixture) — this measures WILLINGNESS to verify under
  volume, not ability to construct verification. That is the mechanism
  the Faros/CircleCI data points at (review-skipping under volume), and
  it is the pre-registered claim; authoring-shaped volume (add-N
  -features) is a recorded flank, not this slice.
- Nested subsets mean L5 includes L1's exact subtask; a model that
  remembers nothing across levels (fresh workspace, fresh session per
  level) cannot leak — sessions are per-level by construction.

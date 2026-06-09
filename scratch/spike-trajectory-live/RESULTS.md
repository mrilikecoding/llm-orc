# Live trajectory probe results — the mechanism holds, content coherence is the limit (2026-06-08)

Real OpenCode 1.15.5 (`opencode run --format json`) → `uv run llm-orc serve --port
8765` (working-tree source) → qwen3:14b seat/judgment + capability ensembles. $0 local.
Task: a 5-deliverable temperature-conversion library (converters.py, test_converters.py,
cli.py, test_cli.py, README.md) with inter-file dependencies. Pre-registration:
`PRE-REGISTRATION.md`. Run ~19 min wall-clock (18:43–19:02).

## The trajectory — MECHANISM HELD, the ideal shape

| turn | tail_kind | verdict | action | delegated | shape | deliverable |
|------|-----------|---------|--------|-----------|-------|-------------|
| 1 | first_turn | (none) | write | **code-generator** | generation | converters.py |
| 2 | trailing | REMAINING | write | **code-generator** | generation | test_converters.py |
| 3 | trailing | REMAINING | write | **code-generator** | generation | cli.py |
| 4 | trailing | REMAINING | write | **code-generator** | generation | test_cli.py |
| 5 | trailing | REMAINING | write | **prose-improver** | generation | README.md |
| 6 | trailing | COMPLETE | finish | — | carry | (converged; OpenCode loop ended) |

**Six turns = 5 writes + 1 finish: the ideal trajectory.** Advance through all 5
distinct deliverables, **zero churn** (no file re-written), delegation **1.000** across
the whole session (`delegated=5 generation=5 boundary_excluded=0`), convergence
COMPLETE once all 5 existed. The axis-1 mechanism Cycle 7 built (route → advance →
converge) holds **end-to-end, live, at depth 5**.

Two bonuses:
- **Capability-matched routing emerged unprompted:** the 4 code files routed to
  `code-generator`, the README (prose) routed to `prose-improver`. The framework
  routed by content type with no instruction to do so.
- **WP-LB-M validated in live production:** the delegation rate counted all 5
  multi-file generation turns (rate=1.0). Pre-WP-LB-M the REMAINING delegated-writes
  would have stamped `carry` and the rate would have read ~1/5 — under-instrumenting
  exactly this session. The meter now measures the trajectory it is meant to measure.

## The LIMIT FOUND — cross-file content coherence is unanchored (provisional "Finding H")

Per the pre-registered boundary (incoherence = a later file fails to reference its
dependency), the run **crossed it at the content layer** even as the mechanism held:

- **cli.py** calls `converters.convert_temperature(value, from_unit, to_unit)` — a
  function that **does not exist** in converters.py (which exposes the three specific
  functions asked for). It also has a syntax error (`args = parser.parse,args`).
- **README.md** documents `fahrenheit_to_kelvin(...)` and a **Rankine** scale —
  neither exists in the code; the CLI usage (`--from --to --value`) does not match
  cli.py's positional arguments.
- **test_cli.py** tests assert behavior cli.py does not implement (expects output
  `'7B'`, calls `main()` with no args).
- **test_converters.py** is correct against converters.py, but has critic review
  prose leaked into the `.py` file (a form-contract bleed).

### Root cause — structural, not just model capability

**The model issued ZERO reads across the entire run** (`read` tool_use count = 0 in
the opencode event stream). Each deliverable was generated from the task description +
the framework digest — which records each prior action as `(action, path, result)`,
**never the file's contents**. So the callee writing cli.py never saw converters.py's
actual API; it guessed `convert_temperature` and guessed wrong. The README and cli
tests are unanchored the same way.

This **rhymes with Finding G** (multi-file *progress* was unanchored — the seat-filler
re-derived "write file 1" because the judge's remaining-statement was discarded; fixed
by ADR-038's remaining-work anchor). Finding H is the same shape one layer down:
multi-file *content* is unanchored — each file is generated blind to sibling contents,
so cross-file API references degrade. Finding G anchored "which file next"; nothing
anchors "what is actually IN the sibling files."

### Where this sits in the architecture

This is **axis-2** (callee content fidelity), the recorded load-bearing risk
(ADR-033 §6b) Cycle 7 scoped as structurally-non-decomposable and named as the
BUILD/PLAY target (ADR-097). The trajectory probe operationalized axis-2 and located
the limit concretely. It is NOT a regression in what Cycle 7 built: axis-1 (the
mechanism) works; axis-2 (does the generated content cohere across files) is the
frontier. The cause is partly model capability (qwen3:14b) and partly a structural
information-flow gap (the callee dispatch carries no sibling content) — the latter is
addressable by design (a content-anchor: feed prior deliverables' signatures/contents
into the callee dispatch, or have the driver read-back before a dependent write).

## Verdict & disposition

**Limit found (n=1 establishes existence).** The synthetic ladder kept passing because
it tests one decision from a clean injected state; the live trajectory surfaced the
content-coherence limit because real prior files accumulate and dependent files must
reference them. The mechanism holds; the content coherence is where the north star's
"all that remains is ensemble iteration" actually bites — and part of it is a design
question (content-anchoring), not only an ensemble-quality question.

Candidate follow-ups (future cycle / PLAY): (1) characterize whether a read-back step
or a content-anchor in the callee dispatch fixes the cross-file API guessing; (2) a
more capable seat/ensemble contrast to separate the model-capability share from the
structural share; (3) the form-contract bleed (critic prose in test_converters.py) is
a separate, smaller ADR-035 form-gate question. Evidence: `serve.log`,
`opencode_run.out`, `workspace/` (the 5 landed files).

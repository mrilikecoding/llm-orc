# Live multi-turn trajectory probe — cumulative degradation (2026-06-08)

PRE-REGISTRATION (recorded before running). The state-injected ladder (rungs A, B,
C, A×B) measures ONE decision from a clean pre-built state — it is blind by
construction to cumulative degradation across a real multi-turn session (context
bloat, error accumulation, drift; the SlopCodeBench failure mode). This probe drives
a real OpenCode → real `llm-orc serve` session to completion and watches the
trajectory turn-by-turn for that degradation.

## Setup (matches the WP-LB-K/L acceptance procedure)

Real OpenCode 1.15.5 (`opencode run --format json`) → `uv run llm-orc serve --port
8765` (working-tree source) → qwen3:14b seat/judgment (`agentic-orchestrator-offline-
tools`) + `code-generator` ensemble (tier=cheap). $0 local. Workspace:
`scratch/spike-trajectory-live/workspace/` (opencode.json → 127.0.0.1:8765/v1, model
`llmorc/agentic`).

## Task (5 dependent deliverables — long enough for cumulative effects to surface)

A small temperature-conversion library, built in the workspace:
1. `converters.py` — `celsius_to_fahrenheit`, `fahrenheit_to_celsius`, `celsius_to_kelvin`
2. `test_converters.py` — unit tests for converters.py
3. `cli.py` — a CLI that imports converters.py and converts a value from args
4. `test_cli.py` — tests for cli.py
5. `README.md` — usage documentation

Inter-file dependencies (tests import the module; cli imports converters; test_cli
imports cli) let coherence loss surface as context grows.

## Measures (turn-by-turn, from serve.log `turn decision:` lines + workspace files
## + the opencode event stream)

Per turn: tail_kind, judgment_verdict, action, delegated ensemble, turn_shape. Across
the trajectory:
- **advance** — each turn produces a NEW deliverable (not a re-write)
- **churn** — re-writing a completed file (the degradation signature); does it RISE
  at later turns / higher turn index?
- **convergence** — finishes COMPLETE once all 5 exist (not before, not never)
- **delegation** — does each write delegate to code-generator, or do inline writes
  (C1) creep in as context grows?
- **coherence** — do later files correctly reference earlier ones (test imports
  module; cli imports converters; test_cli imports cli)?
- **trajectory length** — turns to complete 5 deliverables (ideal = 5 writes + 1
  finish = 6 turns; churn/AS-3 cap inflate it)

## Pre-registered outcome boundary

- **HOLDS** — advances through all 5 distinct deliverables with ≤1 churn, converges
  COMPLETE within ~7 turns, files coherent (later reference earlier).
- **LIMIT FOUND** (any one) — churn rises at later turns (≥2 re-writes, esp. late);
  false-continue (no convergence, AS-3 cap hit); false-stop (<5 deliverables then
  COMPLETE finish); delegation drop-off (inline writes at later turns); incoherence
  (a later file fails to reference its dependency).

## n=1 asymmetry (honest scope-of-claim)

One run is one trajectory. A run that HOLDS is *evidence* the depth-5 trajectory holds
on this stack, NOT proof (one sample). A run that DEGRADES has FOUND a limit —
existence is established by a single instance. So n=1 is adequate for limit-FINDING
(the goal) and inadequate for limit-ABSENCE. If it holds, repeated runs would
strengthen the no-limit claim; if it degrades, characterize the failure turn.

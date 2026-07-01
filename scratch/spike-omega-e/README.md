# Spike Ω-E — contract-first composition (strategy E)

**Status:** RAN 2026-06-29. PASS — closed the cheap-composition quality leaks
at minimal frontier cost. See
`docs/agentic-serving/proposals/ensemble-serving-architecture.md` §4b.

## Question

Does freezing the cross-file contract with ONE frontier call, then building
each file cheap-local against the frozen contract (gate-enforced), produce a
*running* package where the all-local serve shipped a broken one — i.e.,
frontier-quality at minimal frontier tokens?

## Shape

- `architect` ensemble (`.llm-orc/ensembles/spike-omega-e/architect.yaml`):
  frontier (qwen3.6-plus via Go) emits the contract once — per-file `defines`
  (signatures + dataclass fields), exact `imports`, `kind`, `tier`.
- `run_e.py`: build each deliverable with the cheap coder/prose capability,
  injecting the file's contract slice + sibling APIs; a contract-enforcing
  gate (defines present, exact import strings present, dataclass fields
  present, CLI uses argparse, docs mention the contract symbols); within-file
  recovery on gate failure; then score structurally AND by execution.

Same 6-file todo task as Ω-4 / Ω-serve, for direct comparison.

## Result

| | all-local serve | Ω-E contract-first |
|---|---|---|
| structural | 4/6 | **6/6** |
| package runs | no (import error, field drift) | **yes (test passes)** |
| recovery | n/a | **none — every file first attempt** |
| frontier tokens | 0 | **1 architect call (73s)** |
| total | ~11 min | ~5.5 min (73s frontier + 260s cheap) |

Independent check (clean dir, no repo pytest addopts): `1 passed`. (The
in-harness `rc=1` was a false negative — the repo's pytest-cov `addopts` leaked
into the subprocess and raised `CovReportWarning`.)

## Findings

### #1 — Contract-first closes the leaks. (PASS)

The architect picked `completed` (the field) and the exact import forms ONCE,
holistically; every cheap builder honored the frozen contract and the gate made
it binding. The three measured leaks (dataclass field drift, wrong import form,
doc completeness) all closed. No recovery was needed — the cheap tier got each
file right first try because the contract removed the cross-file ambiguity.

### #2 — Frontier cost is one small call.

Frontier tokens = the single architect call (the contract is small relative to
the implementations). Everything else is cheap-local. This is the
"frontier-quality at minimal frontier tokens" point made concrete.

### #3 — The gate became the contract.

Strengthening the gate to check the *contract* (not generic structure) fixed
the "general light gate can't enforce per-task completeness" gap from Ω-serve.
The contract IS the per-deliverable spec; the gate enforces it.

## Open / next

- The contract carries a per-deliverable `tier` (all "cheap" here). The
  follow-up: let the architect assign tiers by sub-task type and route
  accordingly — the substrate for the model-ceiling questions (what can
  Qwen 0.6B / 4B handle; tier-decomposing cognitive architectures).
- Generalize beyond this task shape (vary domain, size, dependency depth).
- The architect itself is a frontier call; a cheaper architect tier (or a
  local architect for simple tasks) is a further token-minimization lever.

# Issue #202 — one child-input contract for non-LLM consumers

Date: 2026-09-21 · Issue: #202 · Decision: Option B (composed data, no LLM instructions)

## Problem

`DependencyResolver._compute_agent_input` treats the LLM prompt envelope as the
fall-through default for any node type it does not recognize. `ensemble:` (ADR-013)
and `loop:` never got a carve-out, so a child ensemble receives the
"Please respond to the following input..." prose instead of the `input_key`-selected
value. `prepare_fan_out_instance_input` has the same shape: only script is carved
out, so any non-script fan-out instance gets the "Processing chunk N of M" wrapper.

Verified at the resolver (2026-09-21 probe):

- `ensemble:` node, no `input_key` → LLM envelope (this issue's repro path).
- `dispatch:` node, deps, no `input_key` → `base_input` only — `gen-review.review`
  (`dispatch: ${select_parts.review}`, `depends_on: [select_parts, gen]`) never sees
  `gen`'s output. Live latent bug under the current `dispatch:` rule.
- `input_key` extraction itself runs before wrapping (ADR-014); the defect is the
  wrapping, not the selection.

## Contract (Option B)

Consumer contract is explicit, no fall-through (unknown type raises):

- **LLM agent**: prompt envelope (unchanged).
- **Script agent**: ScriptAgentInput JSON (unchanged).
- **Child execution (`ensemble:`, `loop:`, `dispatch:`)** — one shared rule:
  - with `input_key`: the selected value, verbatim;
  - without: `base_input` followed by `Agent X (role):\n<response>` blocks
    (honoring `input_scope`), with no LLM instruction sentences.

The fan-out instance path gets the same contract; the "Processing chunk N of M"
wrapper is not part of it.

Why B over A: the `input_key` case is identical in both; the difference is only
children without `input_key`, where A loses information on measured seats
(`build-gated-round.code_writer`, `re-fix.model_edit`) and fixes `gen-review.review`
only by converting the latent bug into a load-time error. B keeps the seats'
information and fixes the reviewer seat as a side effect.

## Implementation

- `dependency_resolver.py`: explicit dispatch in `_compute_agent_input` and
  `prepare_fan_out_instance_input`; shared `_child_contract_input` helper;
  unknown consumer type raises `ValueError`.
- Pin move: the dispatch no-input_key pin is now
  `test_dispatch_agent_without_input_key_gets_composed_data` (doctrine #11:
  pins assert the outcome, go red under a fall-through mutant).
- Separate commit, same PR: `web_searcher.py._extract_query` reads the
  ScriptAgentInput envelope (`input_data`) a root script agent receives inside a
  child ensemble. Different cause, same acceptance path.

## Gates

- Full `make test` + `make lint` (ruff 88, mypy strict).
- Ladder T1 re-run gates merge (build round touched) — **stop point; explicit
  practitioner go required** before any model runs.
- Author-independent adversarial review with wrong-accept hunt before merge
  (delegation contract; doctrine #1).

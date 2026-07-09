# TDD retry loop — held tests on the accept-gate retry round (#100)

**Problem (long-horizon drive 2026-07-09, todoapp turn 7):** on a gate
reject, both `test_writer` and `code_writer` resample per round, so the
two sides disagree on spec-free choices (exact strings, remove-nonexistent
semantics) and convergence is luck. TDD's actual loop holds the tests
fixed and moves only the code.

**Design (approved 2026-07-09):** dispatched dual shape — the loop body
becomes a deterministic router; pure composition, no engine changes.

```
build-gated
  round (loop:, until ${diagnostics.accept}, max 2, carry ${diagnostics.retry_input})
    └─ build-round (router)
         route ──> dispatch ──┬─> build-gated-round   (fresh TDD round, untouched)
                              └─> build-code-round    (held tests, code only)
         └─> unwrap passthrough (loop keeps reading the bare round envelope)
```

## Retry-mode decision (deterministic, in the round envelope)

On reject, `build_gated_envelope.py` picks the carry:

- executor `n_tests > 0` (tests collected and ran) **and** judge
  `tests_adequate` → **held mode**: `retry_input` = original turn +
  executor failure report + the round's tests (echoed by the executor)
  under a `[HELD TESTS]` sentinel block.
- otherwise (collection failure or inadequate tests) → today's fresh
  `retry_input`, unchanged.

`route` detects the sentinel: held → `build-code-round`, else
`build-gated-round`. Round 1 never carries the sentinel, so it is always
fresh. The sentinel constant is defined once in `_helpers.py` (envelope
writes it, route and gather read it).

## build-code-round

`code_writer → gather → executor → accept_gate → envelope` — no
`test_writer`, no `judge`.

- `accept_gather.py`: when the `test_writer` dep is absent, extract tests
  from the sentinel block and strip it from the echoed requirement;
  emit `held: true`. Context/workspace extraction is untouched (the carry
  preserves the original turn text).
- `accept_gate.py`: when the `judge` dep is absent and gather marked the
  round held, `tests_adequate` carries as true ("adequacy carried from
  round 1") — the held path only fires when round 1's judge already
  passed the tests, re-judging identical tests only risks a stochastic
  flip (#84), and the executor stays the live ground truth.
- `envelope`: unchanged behavior (a second reject exits the loop honestly
  at the bound).

Neither new shape is registry-facing (no `topaz_skill`, no `serves`),
like `build-gated-round`.

## Bounds, failure, depth

- `max_iterations: 2` stays: round 1 fresh, round 2 held (or fresh, per
  the decision above).
- A turn whose text contains the sentinel worst-cases into a held round
  whose gate rejects — degraded, never wrong-accept.
- Nesting: serving → build-gated → build-round → build-\*-round → seats
  = depth 4 of 5.

## Validation

- Unit tests per modified script: envelope retry-mode decision, gather
  sentinel extraction + requirement stripping, gate carried adequacy,
  route.
- Ensemble-level: rejected round 1 carries held tests; round 2 dispatches
  to `build-code-round` and accepts.
- Live exit gate (roadmap): rerun the real-OpenCode battery — the
  cli.py-class integration turn passes AND the non-code turns (explain,
  recall) stay green; parity is the whole task surface, not just build.

## Note beyond code

"Hold the verified half fixed, regenerate the dependent half" is a
general orchestration pattern (criteria-primacy, ADR-048): the same shape
applies to non-code seats later (e.g. held acceptance criteria for prose
revision). This item implements it for the build seat only.

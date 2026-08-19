# #169 — a re-fix candidate that is empty is never accepted (design)

Status: pre-flight. Issue: #169, found during #166's pre-flight.

## What is measured

`refix_select.py` substitutes a smoke test whenever rung 1.5 found no visible
test:

```python
_SMOKE_TEST = "def test_refix_candidate_loads_cleanly():\n    pass\n"
```

A `pass` body passes against any code, including none. Measured through the
real `accept_executor.py`:

```
code='' tests=<smoke>  ->  tests_pass=True  n_tests=1  report='all passed'
```

so `refix_envelope` sets `accept: true`, ships `artifacts[0].content = ""`,
and the whole downstream chain admits it — the seat contract asserts artifact
PRESENCE, `ast.parse("")` succeeds, and emit returns
`{"finish": false, "file": "calc.py", "content": ""}`. The target file is one
the client already has, so the write is a **clobber**.

One faulty node output (an empty fence from `model_edit`), no model judgment
anywhere in the gate.

`refix_select.py`'s docstring says "the runner execs the code before any
test, so a candidate that parses but fails to import fails this gate". True,
and not enough: an empty file imports perfectly.

## Change

`refix_envelope.py` gates `accept` on the candidate being non-empty:

```
accept = tests_pass and bool(code.strip())
```

Always, not only on the smoke-only path. An empty candidate is never a valid
fix, and the visible-test path is defeated the same way whenever the client's
test happens not to touch the target module — the same workspace-satisfied
mechanism #166 measured on `build-gated`.

The reject reason says the candidate was empty, so the honest-red terminal
tells the operator which of the two gates refused.

## Why this AND #166's caller guard

Two guards, deliberately, because they guard different things:

- #166 (caller): an empty deliverable is never a client write. Universal,
  last line of defence, and the only seam a project's own scripts cannot
  bypass.
- this (re-fix envelope): a fix must actually be a fix. Route-specific, at
  the source, and it preserves the original file with an honest red rather
  than converting the turn into a refusal.

Without this, the smoke-only path keeps reporting `accept: true` for nothing
at all, which is a wrong verdict even where #166 stops the write — the
recall ledger would record a shipped-then-refused turn instead of a rejected
one.

## Invariant

A re-fix candidate that is empty after `.strip()` is never accepted.

## Regression instruments

1. **An empty candidate on the smoke-only path is rejected.** Red today.
2. **The reject reason names emptiness**, not "failed to load", so the two
   gates stay distinguishable.
3. **An empty candidate with a VISIBLE test is rejected too**, pinning that
   the rule is not scoped to the smoke-only path.
4. **A whitespace-only candidate is rejected**, which kills a `== ""`
   implementation.
5. **End to end**: the re-fix chain with an empty `model_edit` produces no
   client write, driven through `select -> executor -> envelope` and the
   serving tail. Distinct from #166's pin for the same input, because this
   one asserts the turn is REJECTED (accept false) rather than merely not
   written.

Over-refusal pins, which cannot fail under deletion of the guard:

6. **A real candidate on the smoke-only path still accepts.**
7. **A one-character candidate still accepts** — the rule is emptiness, not
   a length heuristic.

## Known bounds

- Does not make the smoke test meaningful. A candidate that is non-empty but
  useless still passes a `pass` body; only emptiness is closed here.
- Comment-only candidates are not covered, matching #166's choice for the
  same reason: telling "only comments" from "a real file" needs a parser.

# Seat quality: per-test isolation + bare-name-assert sanitizer (path item 4)

**Status:** Approved design, 2026-07-09 (rev 2 — escalation deferred on
spike data). Follows the v0.18.6 ladder rerun (4/10 strict; 3 misses from
round-1 test quality).

## Evidence (live probes, 2026-07-09, qwen3:8b seats)

Two fresh single-turn probes reproduced the ladder's build rejects 2/2,
with full traces (`LLM_ORC_SERVE_TRACE_SNIPPET=6000`):

- **"write a function that adds a todo item to a list in todo.py"** —
  round 1: tests call `add`, code named it differently (`NameError`).
  Round 2 (held tests) defined `add` correctly and STILL rejected:
  `test_add_multiple_todo_items` asserts `len(todos) == 2` after an
  earlier test already appended — module-global state leaking across
  tests in one run.
- **"create storage.py with save_todos and load_todos using json"** —
  round 1 tests assert `not os.path.exists("todos.json")` after an
  earlier test created it (filesystem leakage), plus round 2 regenerated
  `load` where the failure report explicitly named `load_todos` missing.

Failure classes:

- **A — test-order state leakage** (both probes; the dominant breaker):
  the 8b writes tests that assume per-test isolation the executor does
  not provide. The code was right; the gate false-rejected. Deterministic
  executor fix available — no model judgment involved.
- **B — name divergence surviving the held round** (one probe): the seat
  ignored an explicitly named missing symbol on the retry. Third
  confirmation of the prompt-saturation finding → structural lever
  (escalation), not another prompt rule.
- **C — exception-message assertions** (known #98 residual): present in
  probe tests but not the breaking failure. No sanitizer this arc; revisit
  when it breaks a measured turn.

## Slices

### Slice 0 — decider interrogative fix

`_INTERROGATIVE_RE` (classify) covers wh-forms only, so "did you see my
previous query?" rides the stochastic decider (ladder turn 5 mis-routed to
build). Add the memory-shaped yes/no form addressed to the assistant:
`^(did|have) you\b`. Deliberately NOT `can/could/will/would you` — polite
imperatives ("can you write add.py?") must stay on the build path.

### Slice 1 — per-test isolation in the accept executor

Today the executor materializes `solution.py` + `tests.py` + workspace
into one tempdir and runs one subprocess; module globals and written
files leak across test functions. Change: **each test function runs in
its own subprocess with its own fresh copy of the materialized
workspace** (fresh import, fresh cwd). Chosen over a single-subprocess
re-import loop because interpreter state (env mutations, monkey-patched
modules) would still leak there, and the overhead (~100ms × ≤10 tests)
is trivial on the rig.

- The line-level failure report keeps its format, aggregated per test.
- Collection errors and the deterministic adequacy checker are untouched.
- Both build-gated and write-tests shapes inherit the change (shared
  executor).
- Tests that genuinely require cross-test ordering now pass individually —
  acceptable; order-independence is the intended test semantics.

### Slice 2′ — bare-name-assert sanitizer (replaces escalation; spike-grounded)

The 4-arm tier/thinking spike (2026-07-09, probe-2 held-round fixture,
n=3 per arm) found every arm — {qwen3:8b, qwen3:14b} × {think on, off} —
defines the required API and plateaus at exactly 3/4 held tests. The
universal failure is the suite's stray `assert load` line: a value-free,
bare-name assert referencing a symbol every model reads as a typo for
`load_todos`. No tier or thinking mode converts a garbage test line.

So the residual lever is deterministic: **strip `assert <bare-name>`
statements from authored/held tests before execution** — an assert whose
test expression is a bare `Name` node checks only object truthiness (a
defined function is always truthy) and carries no test value. One small
AST pass applied where tests are gathered for the executor; the shipped
artifact carries the sanitized suite (what executed is what ships,
preserving the #98 invariant). With it, the fixture accepts at round 1
on the cheap seat in ~5s.

### Escalation — deferred, measurement recorded

Spike latency data: qwen3:14b think-off generates as fast as qwen3:8b
think-off on this rig (4–13s vs 5–9s); think-on costs 10–20× wall on
either model (52–105s). **If a measured class ever demands escalation,
the rung is qwen3:14b think-off** (dominates 8b think-on: same
conversion, ~1/10 the latency), via the existing
`agentic-tier-escalated-general` profile. Not built now: no measured
class converts under escalation that isolation + the sanitizer don't
already fix. Loop bound stays 2.

### Slice 3 — missing-import injection (amendment, live-evidence 2026-07-09)

Task 5's regression probe surfaced a fourth class: the 8b's tests call
`os.path.exists` / `pytest.raises` without importing `os`/`pytest`, so
every round NameErrors on a defect no code regeneration can fix (code
cannot define `pytest`). Deterministic repair, same shape as the existing
workspace-import injection: AST-collect unbound names in the tests,
intersect a fixed whitelist (`os, json, sys, re, math, time, pathlib,
tempfile, itertools, functools, collections, string, unittest, pytest`),
prepend the `import` lines before execution. The echoed (shipped) tests
carry the injected imports — the artifact becomes self-contained. Names
outside the whitelist stay uninjected and reject honestly.

## Validation

- Per-slice TDD; hermetic runs through the real engine for the router
  bound and the executor isolation.
- The two probe prompts above are the live regression fixtures. Probe 1
  must ACCEPT after slice 1 alone (its held-round code was right; only
  test interference rejected it). Probe 2 must ACCEPT after slices 1+2′
  together (isolation clears the leakage asserts; the sanitizer clears
  the stray `assert load`). Stochastic seat caveat: "accept within two
  attempts" is the pass bar for both.
- Full ladder rerun (`benchmarks/agentic_serving/ladder_battery.sh`),
  new trajectory row.
- Record the arc's measurement: how many ladder rejects survive
  isolation + the sanitizer.

## Out of scope

- Escalation (deferred with its measurement — see above).
- Sanitizing exception-message assertions (the other #98 residual) —
  deferred until it breaks a measured turn; the sanitizer's boundary is
  bare-name asserts only.
- Test-writer seat escalation.
- #83 run half (client-delegated execution, discovery) — next arc, per
  roadmap.

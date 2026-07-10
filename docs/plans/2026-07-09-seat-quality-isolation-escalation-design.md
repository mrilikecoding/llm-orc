# Seat quality: per-test isolation, then bounded escalation (path item 4)

**Status:** Approved design, 2026-07-09. Follows the v0.18.6 ladder rerun
(4/10 strict; 3 misses from round-1 test quality).

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

### Slice 2 — round-3 escalation on exhausted retry

`route_round` (the deterministic router) gains one rung: a reject whose
held 8b retry also rejected dispatches ONE round 3 to `build-code-round`
on a new **`agentic-tier-strong`** profile — shipped default
**`qwen3:14b`** (local, fits the 32GB rig; operator-overridable via
`*.local.yaml` like every tier). Same held tests, same gate; after round
3, honest reject as today. Loop bound 2 → 3; the turn trace records the
escalated round. Code seat only — test-writer escalation is class C
territory, out of this arc.

## Validation

- Per-slice TDD; hermetic runs through the real engine for the router
  bound and the executor isolation.
- The two probe prompts above are the live regression fixtures. Probe 1
  must ACCEPT after slice 1 alone (its held-round code was right; only
  test interference rejected it). Probe 2 clears its leakage failures
  under slice 1 but its name divergence (class B) persists — it must
  ACCEPT after slices 1+2 together (stochastic seat caveat: "accept
  within two attempts" is the pass bar for both).
- Full ladder rerun (`benchmarks/agentic_serving/ladder_battery.sh`),
  new trajectory row.
- Record the arc's measurement: how many rejects survive isolation, and
  of those, how many round-3 14b converts.

## Out of scope

- Test sanitizer for exception-message assertions (class C) — deferred
  until it breaks a measured turn.
- Test-writer seat escalation.
- #83 run half (client-delegated execution, discovery) — next arc, per
  roadmap.

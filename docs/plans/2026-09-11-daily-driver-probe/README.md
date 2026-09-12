# Daily-driver probe — the serve on an EXISTING repo (2026-09-11)

Question: the practitioner's ask this session is "an effective daily driver
for long-running tasks of all kinds". Every recorded battery so far is
greenfield (the 13-turn ladder builds a todo app from an empty repo; the
volume ladder seeds only the asked modules). What happens on the shape a
daily driver actually meets — a small existing package with its own tests?

Setup: a seeded git repo (`seed-repo.tgz`): `todo/storage.py` (TodoStore:
add/list/complete, JSON-backed), `todo/cli.py` (argparse add/list),
`tests/test_storage.py` (3 passing), pyproject with pytest config. Serve
restarted from main `1ac25bb4` before turn 1 (the 08-30 process on :8765
was stale). Real client: `opencode run --format json -m llm-orc/agentic`
per turn via `dd-turn.sh` (sandbox-disabled detach, per the wedge note),
workspace truth captured after each turn in a throwaway copy. Turns 1-2
are one continued session; 3-7 are fresh sessions. One axis varied per
rung; turn 7 is the greenfield control.

| turn | ask (one axis) | route observed | outcome | ground truth |
|---|---|---|---|---|
| 1 | add `remove()` to `todo/storage.py` AND tests in `tests/test_storage.py` | `need-files` read of storage.py, then `tests-seat` (the "add tests" marker wins; the build half is dropped); deliverable path flattened to `test_storage.py`; tests written as `from storage import TodoStore` against a `remove` that does not exist | "Another round needed: tests did not pass" | nothing written |
| 2 | add `remove()` to `todo/storage.py` (pure edit, same session) | `code-seat` → `build-gated`; the coder emitted the METHOD ALONE (`def remove(self, ...)` at module top level), the gate ran it as the whole module | "Another round needed: tests did not pass" (320s, two rounds) | nothing written |
| 3 | add a `done` CLI command in `todo/cli.py` AND a new `tests/test_cli.py` | read of cli.py, then `tests-seat`; tests for a command that does not exist | "Another round needed: tests did not pass" | nothing written |
| 4 | "Run the tests." | run rung: `pytest -q` delegated client-side | honest verdict "1 errored" — the bare PATH `pytest` could not import `todo` (the seed lacked `pythonpath`; fixed in the seed for later turns) | verdict matches the client's own result |
| 5 | "Run ruff check on the project and report what it finds." | `decider` chain, no delegation | generic prose on how one WOULD run ruff; no run, no claim of a run | nothing run |
| 6 | priority feature across storage + CLI + tests, no file named | `code-seat` → `build-gated`; no discovery (no file named, no "existing" verb → no glob); deliverable minted as `solution.py` | ACCEPTED and WRITTEN (397s): a parallel in-memory `TodoStore` with a `cli_add` string parser, unrelated to the `todo` package | `solution.py` at repo root; existing tests untouched (3 passed); the ask not advanced |
| 7 | control: "Write a function that adds two numbers in add.py." | `build-gated` | wrote a correct `add.py` (93s) | correct |

## What the rows say

- The serve is healthy (turn 7). The misses are one shape: **the serve
  assumes a flat, greenfield workspace.** Four mechanisms, each visible in
  a specific row:
  1. **Two-deliverable asks route to one seat** (turns 1, 3): `_TESTS_PRIMARY_RE`
     makes "add tests" win, and the code half is dropped; the tests then
     target a surface that does not exist yet. This is #123's class.
  2. **Edit-of-existing-file has no contract** (turn 2): `code-generator`'s
     prompt says "show the code change directly", so the coder returns a
     fragment; the build contract and the gate treat it as the whole
     module. This is #122's class.
  3. **The gate sandbox is one flat directory** (turns 1, 2, 3):
     `accept_gather._workspace` keys files by BASENAME and `target_file`
     is the basename; classify's tests destination is `f"test_{basename}"`.
     Package paths (`todo/storage.py`, `tests/test_storage.py`, `from
     todo.storage import`) cannot be represented, so a correct test file
     for a real package can never pass the gate and a correct destination
     can never be emitted. Not tracked by any open issue before today.
  4. **No discovery before a build in a non-empty workspace** (turn 6):
     `wants_existing` is false without a named file or an "existing" verb,
     so the serve builds greenfield into `solution.py` beside a package
     that already implements the domain — a gate-accepted deliverable that
     does not advance the ask. The gate is right about what it measured
     (its own tests against its own file); the turn is wrong about the
     task.
  5. Turn 5 is #124 (closed pytest template), as recorded.
- Latency: gated build turns on qwen3:8b cost 93s (control), 320s (two
  rounds), 397s (two rounds) wall clock; see `timings.tsv`. The run rung
  is 2.7s and the tests-seat refusals ~20s.

## Files

`ask-NN.txt` the prompt; `turn-NN.jsonl` the raw `opencode run --format
json` events; `truth-NN.txt` exit code, git status, diff stat, and a
throwaway-copy pytest after the turn; `timings.tsv` first-to-last-event
wall clock per turn (excludes client bootstrap); `dd-turn.sh` the driver;
`seed-repo.tgz` the seeded repo (turn-7's `add.py` and the turn-6
`solution.py` excluded/removed). Serve trace rows for these turns are the
last seven records of `.llm-orc/.serve-trace/turns.jsonl` as of 17:55.

## Comparator arms on the same seven asks (2026-09-11, practitioner-authorized paid runs)

Same seed, same asks, same driver (`dd-turn.sh` with `DD_MODEL`), fresh
clone per arm, turn 2 continuing turn 1's session, turns 3-7 fresh.
Costs are OpenCode's own per-step `cost` field summed per turn.

| turn | serve (arm 0, free) | `opencode-go/qwen3.8-max` (Go) | `opencode/claude-sonnet-5` (Zen) |
|---|---|---|---|
| 1 remove() + tests | refused | ls, glob, 2 reads, 4 edits, pytest → 5 passed ($0.076) | 2 reads, 3 edits, pytest → 5 passed ($0.098) |
| 2 pure edit (continued) | refused | read → "already done" ($0.012) | read → "already exists, no change" ($0.015) |
| 3 CLI `done` + new test file | refused | reads, edits, write tests/test_cli.py, pytest → 8 passed ($0.055) | same → 8 passed ($0.064) |
| 4 run the tests | honest verdict | `python -m pytest` → 8 passed ($0.021) | 8 passed ($0.021) |
| 5 run ruff | prose, nothing run | ran ruff → clean ($0.010) | ran ruff → clean ($0.008) |
| 6 priority feature | minted solution.py | glob, 4 reads, 7 edits, pytest → 13 passed ($0.059) | reads, todowrite plan, 6 edits, pytest → 15 passed ($0.161) |
| 7 control add.py | correct | write + ruff + pytest ($0.040) | write ($0.015) |
| **total** | **1/7** | **7/7, $0.27** | **7/7, $0.38** |

What the comparators do, every time: discover (ls/glob) → read the
files they will touch → surgical `edit` (never a whole-file rewrite of an
existing file) → run the project's own suite → summarize honestly with
line references. Four structures, none of which is model size: a Go-tier
model does it at cents. Arcs 1 and 2 (workspace-aware routing; the
sandbox mirrors the workspace) are the first two; edit delegation (#122)
and running the workspace's own suite after a build (verified acceptance
in the workspace, not only in the sandbox) are the next two.

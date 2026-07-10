# Client-delegated test runs through the permission seam (#83, run half)

**Status:** Approved design, 2026-07-09. Implements the run half of the
client execution surface (the fix-execution rung's enabler). Discovery
(list/glob) is the remaining #83 widening, designed separately.

## Problem

The accept gate verifies deliverables in the serve's own sandbox against
tests the serve wrote. That proves the artifact against its own tests, not
against the client repo's real suite in the client's real environment — the
gap the fix-execution rung (locate, edit, run tests, verify on a real
codebase) needs closed. ADR-048 ODP-1's second option — delegate execution
to the client via the permission seam — is the mechanism: the serve emits a
bash tool_call, the client executes it (behind its own permission prompt),
and the result arrives on the continuation. The read half (shipped v0.18.6)
built the seam's request/resume machinery; this rung reuses it for
execution.

## Decisions

- **Explicit run turns only (rung 1).** A turn that asks to run tests
  ("run the tests", "run test_calc.py", "rerun pytest") delegates ONE
  deterministically-constructed pytest command. Chaining write→run inside a
  single fix turn is the fix-execution rung proper, built on this enabler.
- **The command is a closed template, never model text.** `pytest -q`
  plus the turn's named `test_`-files. File arguments come from classify's
  `_FILE_RE`, whose charset (`[\w./-]`) admits no shell metacharacters; the
  builder re-asserts the charset as defense in depth. The serve never emits
  a command a model composed — deterministic control, and the client's own
  permission prompt on bash is the second guard.
- **Fully deterministic turn.** Both passes route structurally: pass 1 to a
  script echo shape (`need-run`), pass 2 to a script verdict shape
  (`run-verdict`) that parses the pytest summary deterministically. A "run
  the tests" turn costs zero model calls.
- **Wire-derived, stateless resume.** Same posture as reads: the
  continuation is detected from the appended history alone; no server-side
  pending state.

## Turn flow (two passes)

**Pass 1 — request.** classify detects the run signal (deterministic
regex: an imperative run/rerun/execute verb with a tests/pytest object, or
a named `test_`-file with a run verb) and no `[ran ...]` block after the
latest user message. It routes to the `need-run` echo shape and carries
`needs_run: "<command>"` through resolve → shape → form_gate → emit, which
ships `{"finish": false, "run": "<command>"}`. The caller maps it to one
bash tool_call (`{"command": "<command>", "description": "Run tests"}`)
resolved against the client's advertised tools (`finish_reason:
tool_calls`). The serve holds nothing.

**Pass 2 — resume.** The client calls back with the tool result appended.
The caller's continuation detector treats a run-shaped call (arguments
carry `command`, no `filePath`) like a read: fall through to the pipeline,
never the write ack. The renderer adds an `assistant: [ran <command>]`
block — output body indented two spaces (see hazard note) and TAIL-capped
at `_RUN_OUTPUT_CAP` (pytest's summary lives at the end). classify sees
the run signal AND a run block answering it → routes to `run-verdict`,
a script node that extracts the latest run block and composes the verdict
from pytest's own summary line: passed/failed/error counts, "no tests
ran", or an honest could-not-parse report carrying the output tail. emit
finishes prose.

Run blocks are selected ONLY from messages after the latest user message —
run output is ephemeral verification evidence for the turn that requested
it, unlike read blocks (durable workspace state, selected from the full
history). A later turn never re-renders an old run result.

## Components

| Change | Where |
|--------|-------|
| Run signal + `needs_run` command builder + verdict routing | `.llm-orc/scripts/agentic_serving/classify.py` |
| `needs_run` passthrough | `resolve.py`, `shape.py`, `form_gate.py` |
| `run` outcome kind | `emit.py` |
| `need-run` echo shape | `need-run.yaml` + `need_run_echo.py` |
| `run-verdict` shape (deterministic pytest-summary parse) | `run-verdict.yaml` + `run_verdict.py` |
| Bash tool resolution, run-shaped continuation split, `[ran]` render | `serving_ensemble_caller.py` |

## Bounds and error handling

- **One run round per turn.** Any `[ran ...]` block after the latest user
  message suppresses a re-request; unparseable output produces an honest
  report, never a second run. Deterministic termination.
- **Output tail-capped** (`_RUN_OUTPUT_CAP = 4096` chars) — the tail, not
  the head: pytest prints its summary last. Over-cap keeps the tail and
  marks the header `(truncated)`; the verdict parser is unaffected.
- **Run bodies are indented two spaces in the render.** Run output is
  untrusted column-0 text; gather's workspace extraction is line-anchored,
  so an unindented body containing a `assistant: [wrote x.py]` lookalike
  could materialize a phantom file (the same class as the read-body
  fencing follow-up). Indentation kills the class for run blocks outright;
  run output is never materialized, so the transform is lossless where it
  matters and the verdict parser strips it.
- **pytest-scoped (rung 1).** Consistent with the Python-scoped gate;
  per-language runners are the same seat-swap generalization the roadmap
  names for the gate.
- **No repo-layout guessing.** Named `test_`-files ride the command; a bare
  "run the tests" runs `pytest -q` (rootdir discovery is pytest's job).
  Mapping `calc.py` → `test_calc.py` guesses layout — refused, not built.
- **Trust posture unchanged.** The client executes in its own environment
  behind its own permission prompt; output entering the render is treated
  as untrusted text (indented, capped, never materialized).

## Testing and validation

TDD: unit tests for the run signal and command builder, the `needs_run`
passthrough, the `run` outcome, the continuation split, the `[ran]` render
(indent, tail-cap, failure variant), and the verdict parser (passed /
failed / no-tests / unparseable). Hermetic end-to-end through the real
engine: pass 1 emits the bash tool_call; pass 2 with a canned pytest
output ships the verdict prose. Live at the earliest runnable point: real
OpenCode session — ship a file plus tests, then "run the tests" — wire-
capture OpenCode's bash result format and lock the normalizer to it (the
read-half procedure), then the ladder rerun with a run rung and the
trajectory-table update.

## Named forward directions (not built here)

- **Discovery** (list/glob for files the turn doesn't name) — next #83
  widening; needs its own termination-control design pass.
- **Chained fix-execution**: write → run → verdict inside one turn (the
  fix shape) — composes this seam with the write seam.
- **Per-language runners** behind the same command-template contract.
  Rust is the named first target (plexus integration; the meta-task rung's
  plexus half): `cargo test` is a single closed template with no file
  arguments, so the swap is a detection rule (Cargo.toml named/visible)
  plus a verdict parser for cargo's `test result:` summary — the
  `needs_run` plumbing, seam, and continuation split carry unchanged.
  Kept out of rung 1 only because the ladder can't measure it yet; the
  command-builder and verdict-parser seams are the two places rung 1 must
  not bake in pytest assumptions (both are isolated single functions).
- **Escalation-on-red**: a failed run feeding the next build round's
  context (today the user relays; the fix shape closes the loop).

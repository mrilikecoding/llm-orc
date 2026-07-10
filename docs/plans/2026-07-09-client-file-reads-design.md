# Client-file reads through the permission seam (#83, rung 1)

**Status:** Approved design, 2026-07-09. Implements the read half of the
client execution surface; run-tests delegation reuses the seam as a
follow-up.

## Problem

Files that exist only in the client workspace are invisible to the serve:
"write tests for existing foo.py" rejects because the sandbox can't import
it (battery 2026-07-08). ADR-048 ODP-1 named two options — thread client
read-tool results into the turn, or delegate execution via the permission
seam. On the wire they are one mechanism: the serve emits a tool_call
against the client's advertised tools, the client executes it, and the
result arrives on the continuation call. This rung builds the read payload.

## Decisions

- **Read path first.** Run-tests delegation is designed for (same outcome
  pattern), built as a follow-up.
- **Named files only.** The serve requests reads only for files the turn
  names. Discovery is the named next widening (below), not built here.
- **Wire-derived, stateless resume.** The continuation is detected from the
  appended history alone; no server-side pending state. Safe because
  routing is deterministic (the recompute lands on the same shape) and the
  wire is measured append-only (`LLM_ORC_SERVE_WIRE_LOG` watches for the
  day that changes; the session substrate is the fallback home for pending
  state if it does).

## Turn flow (two passes)

**Pass 1 — request.** classify extracts the named file as today, plus a
deterministic visibility check: visible iff a `[wrote <path>]` or
`[read <path>]` block exists in the rendered context. Named-but-invisible
on a workspace-needing shape (build / write-tests / edit-existing) routes
to a new emit outcome `{"finish": false, "reads": ["<path>", ...]}`. The
caller maps it to read tool_call(s) resolved against `context.tools`
(`finish_reason: tool_calls`). The serve holds nothing.

**Pass 2 — resume.** The client calls back with the tool result appended.
The caller's continuation detector splits: a write continuation acks (as
today); a read continuation re-enters the pipeline — same pending task
(last user message), context re-rendered with the read result as a
`[read <path>]` block. classify now sees the file, routes to the real
shape; gather materializes the block into the sandbox exactly as it does
`[wrote ...]` blocks; the normal accept gate runs.

Because read results live on the append-only wire, files read once stay
retrievable in later turns through the existing lossless selection — deep
recall over client files comes free.

## Components

| Change | Where |
|--------|-------|
| Visibility check + `need_files` signal (deterministic, no model call) | `.llm-orc/scripts/agentic_serving/classify.py` |
| `reads` outcome kind | `.llm-orc/scripts/agentic_serving/emit.py` |
| Tool-name resolution against advertised tools (write + read; hardcoded fallback) | `serving_ensemble_caller.py::_outcome_chunks` |
| Continuation split (read → re-enter, write → ack) | `serving_ensemble_caller.py` (`_tool_result_ack` path) |
| `[read <path>]` render + selection (read blocks join the written-file selection, which becomes selection over all file blocks) | `serving_ensemble_caller.py` context render |
| Read-block extraction into the workspace | `accept_gather.py` / `tests_gather.py` header regex |

## Bounds and error handling

- **One read round per turn.** If pass 2 still can't see the file (tool
  error, empty content, mismatched result), emit an honest prose refusal
  (`Refused: could not read <path>: <reason>`). Never re-request:
  deterministic termination, no read loops.
- **Per-file size cap.** A read block over the cap refuses honestly rather
  than materializing a truncated file (a truncated module fails imports
  and produces a confusing reject). Default 24 KB per file, a named
  constant; deliberately larger than `_CTX_FILE_CAP` (real modules run
  bigger than conversation-written ones), and read blocks bypass the
  selected-block char cap — whole-file-or-refuse, never a mid-block cut.
- **Multi-file.** All named-invisible files ship as one `tool_calls` array
  in pass 1; still one round.
- **Trust posture unchanged.** Client workspace content entering the
  sandbox is untrusted input; #85 (sandbox hardening) remains the gate
  before untrusted deployment.

## Testing and validation

TDD: unit tests for the visibility check, the `reads` outcome, the
continuation split, the `[read]` render/selection, and gather extraction;
one hermetic end-to-end through the real engine (the #100 pattern: real
graph, scripted seats). Live at the earliest runnable point: real OpenCode
session, "write tests for existing `<file>`" against a repo file — the
exact 2026-07-08 battery failure — then the ladder rerun and
trajectory-table update.

## Named forward directions (not built here)

- **Discovery** (practitioner-flagged at approval): files the turn does
  not name — "write tests for the storage module" — need list/glob
  tool_calls through this same seam and a termination-control design pass
  (when to stop reading). The meta-task battery rung (real-repo retrieval)
  demands it; the named-files-only and one-round bounds are rung-1 bounds,
  not architecture.
- **Run-tests delegation**: `{"finish": false, "run": ...}` through the
  same outcome pattern and continuation split — the fix-execution rung's
  enabler.

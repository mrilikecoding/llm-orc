# Discovery through the permission seam (#83, final widening)

**Status:** Approved design, 2026-07-10. The read-half design's named
deferral ("list/glob ... and a termination-control design pass"); the
meta-task rung's requirement.

## Problem

A turn that names no file — "write tests for the storage module" — has
nothing for the read seam to fetch. Today it routes to the tests seat
with a default destination and builds against nothing, or refuses. The
serve needs to FIND the file, with deterministic control over when the
finding stops (no read loops, no model-driven exploration).

## Decision (rung 1)

One glob round, a deterministic match rule, then the EXISTING read seam.
No model judgment anywhere; every ambiguity refuses honestly.

**Trigger:** a workspace-needing turn (tests-primary, or build verb with
an existing-file marker) that names NO source file but names a module
stem — the phrasings `<stem> module` / `module <stem>` and
`tests for <stem>` where `<stem>` is an identifier-ish token that is not
a visible file's basename. One stem per turn; no stem → today's behavior
unchanged.

**Pass 1 — list.** classify emits `needs_glob: "<stem>"`; the routing
rides a `need-glob` echo shape (the need-files pattern); emit ships
`{"finish": false, "glob": "<stem>"}`; the caller maps it to the
client's glob tool (candidates resolved against advertised tools, wire
capture locks the argument name) with pattern `**/*<stem>*` — the stem
is charset-restricted by the same `_SAFE_ARG_RE` discipline as run
commands.

**Pass 2 — match.** The listing renders as an
`assistant: [globbed <stem>]` block (fenced grammar: body indented, one
path per line; never materialized — gather's header regex is untouched).
classify extracts candidate paths from the block deterministically:
basename contains the stem, `.py`, not `test_*`-named. Exactly one
candidate → it becomes the named file and the EXISTING need-files read
seam takes over (pass 3 = read, pass 4 = build — each round type is
already one-per-turn bounded). Zero candidates → honest refusal
("no file matching 'storage' in the workspace listing"). Two or more →
honest refusal naming the candidates (the user picks; deterministic
one-or-refuse beats a tie-break heuristic that guesses wrong silently).

**Termination control:** at most one glob round per turn (a `[globbed]`
block after the latest user message suppresses re-request — the
has-run-block pattern); the chain glob → read → build is bounded because
each link reuses an existing one-per-turn seam. A failed/empty glob
result refuses honestly, never re-requests.

**Selection:** glob blocks are ephemeral evidence like run blocks — they
render only from the post-latest-user slice (the chain's later passes
still see them; later turns never re-render a stale listing). Reads
remain the durable state.

## Bounds

- Stem extraction is exact-phrasing rung-1 ("the storage module",
  "tests for storage"); bare "the tests for it"-style anaphora stays
  with today's routing. Widen on ladder evidence.
- Python-scoped candidates (`.py`), consistent with the gate; the
  per-language generalization rides the same seat-swap contract the
  roadmap names.
- The glob pattern is template-built from the charset-checked stem —
  never model text (the run-command discipline).
- Listing caps: the rendered block keeps at most 50 paths (tail-marked
  when cut); candidate matching runs on the rendered block only.

## Components

| Change | Where |
|--------|-------|
| Stem trigger + `needs_glob` + glob-block candidate matching | `classify.py` |
| `needs_glob` passthrough | `resolve.py`, `shape.py`, `form_gate.py` |
| `glob` outcome kind | `emit.py` |
| `need-glob` echo shape (+ top-level symlink + catalog test picks it up) | `need-glob.yaml`, `need_glob_echo.py` |
| Glob tool candidates, continuation split, `[globbed]` render (fenced) | `serving_ensemble_caller.py` |

## Validation

Wire-capture OpenCode's glob tool (advertised name, argument schema,
result format) FIRST and lock the normalizer to it — the read-half
procedure. TDD per component with the capture as fixture; hermetic
e2e of the full glob → match → read → build chain through the real
engine; live "write tests for the storage module" against a repo with
storage.py (plus the zero-match and two-match refusal probes); ladder
gains a discovery rung; independent adversarial review (named targets:
the match rule's determinism, refusal honesty, and the fenced-grammar
discipline for the new block type).

## Out of scope

- Multi-stem turns, recursive narrowing, content search (grep-shaped
  discovery) — later rungs, on meta-task evidence.
- Fallback from a failed named-file read to glob.
- Model-decider-path discovery.

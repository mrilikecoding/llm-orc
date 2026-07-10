# Block-body fencing: one indent rule for every context block

**Status:** Approved design, 2026-07-10. The pre-meta-task blocker from the
roadmap handoff.

## Problem

The rendered context's block grammar (`assistant: [wrote <path>]`,
`[read <path>]`, `[ran <command>]`) is line-anchored at column 0, but
wrote/read block BODIES render untrusted file content verbatim at column 0.
A body line that looks like a header is parsed as one:

- gather's workspace extraction terminates the real block at the lookalike
  and materializes a **phantom file** into the accept-gate sandbox;
- classify's `[read]` visibility and `[ran]` detection regexes match
  lookalikes, so a read file containing `assistant: [ran pytest -q]` makes
  a later "run the tests" turn skip the delegation and **fabricate a
  verdict** from the forged block (independent review, 2026-07-09/10).

The meta-task rung reads exactly the files that contain these strings —
this repo's own docs. Run blocks already solved this locally (bodies
indented two spaces); wrote/read blocks did not.

## Decision

**Every block body is indented two spaces at render; every parser treats
column 0 as grammar-only.** One rule, three block types, no per-type
carve-outs (determinism-over-carve-outs):

- Renderer (`_render_write`, `_render_read_block`; run blocks already
  comply): each non-empty body line gets a two-space prefix;
  whitespace-only lines render empty (the run-block convention).
- gather (`accept_gather._workspace`, inherited by `tests_gather`): body
  lines are exactly those starting with the two-space prefix (stripped on
  materialization) or empty; ANY other line ends the body. An indented
  lookalike strips back to its original text as file CONTENT — the
  round-trip is verbatim for the lines that matter (whitespace-only lines
  lose trailing spaces, as today's `.strip()` already implies).
- classify header regexes and `run_verdict`'s parser are already
  `^`-anchored — indented bodies simply stop matching them. No changes
  beyond tests pinning the spoof cases dead.

No compatibility window: the render is recomputed from the wire every
turn and renderer + parsers ship in one release.

## Spoof cases this kills (all pinned by tests)

1. Read/written file body containing `assistant: [wrote x.py]` → no
   phantom file in the sandbox.
2. Read file body containing `assistant: [ran pytest -q]` + fake summary →
   `has_run_block` stays false; the run turn delegates a real run.
3. Read file body containing `assistant: [read y.py]` → no spoofed
   visibility (classify still requests the real read).
4. Write-block body shadowing a real run block in run-verdict's
   latest-block scan (the reviewer-flagged residual half).

## Out of scope

- The raw-task seam (closed in v0.18.8 — run-verdict reads conversation
  only).
- `user:`/`assistant:` prose lines (single-line, flattened — no bodies).
- #107 (content-parts message content) and #106 (shape single-home).

## Validation

TDD per component; hermetic endpoint test reading a file whose content
carries all three forged headers (no phantom, real delegation, verbatim
materialization round-trip); live OpenCode session reading a
lookalike-bearing file then building against it; ladder rerun + trajectory
row (the indent also changes what generation seats see, so the rerun
doubles as a quality check).

# Spike 4: #121 content-grep viability — findings

Script: `2026-08-13-content-grep-spike.py` (arms A–F; arm E killed, see
below). Battery: 10 content questions whose subjects are NOT filenames
(the slice-1 known misses), expected grounding files pre-registered.
Model arms use qwen3:8b — the deployed cheap-seat tier — via ollama,
3 samples per question.

## Results

1. **The deterministic find-or-refuse ladder is sound** (arm A): with a
   correct identifier, unique-file → def-site → refuse grounds 8/10 right,
   2 honest refusals, ZERO wrong-file. Both refusals were repo quirks
   (spike artifacts under docs/plans duplicating real definitions — a
   docs exclusion recovers one; the other is a filename-shaped question).
2. **The issue's original Approach B is REFUTED** (arm B): qwen3:8b
   proposing a search term from the question alone scored 0/30 usable —
   it invents plausible names that do not exist (`build_recall_ledger`,
   `detect_truncation`). The ladder's find-or-refuse backstop held (zero
   wrong-file, all NONE), so honesty survives, but utility is nil.
3. **Deterministic stem→identifier harvest contains the answer** (arm C):
   harvesting REAL identifiers containing a question stem puts the right
   file in the candidate set 9/10 — but the plain union is far too broad
   (2–12 files per question).
4. **The doctrine-9 composition works** (arms D/F): deterministic harvest
   → closed menu of real identifiers (each pre-laddered to its def-site
   file) → the cheap seat PICKS one (abstain allowed) → ladder backstop.
   Arm D (workspace-only, mention-tolerant ladder): 19/30 right, 11
   wrong. The wrongs decomposed into (a) DOT-DIR BLINDNESS — rg, like the
   client's glob, cannot see `.llm-orc`, so the right file could not even
   enter the menu and a comment MENTION elsewhere won the unique-file
   rung — and (b) adjacent-but-defensible picks.
5. **Arm F (the production shape): 21/30 right, 9 wrong, 0 abstain/
   off-menu.** Two deterministic fixes: the TWO-SURFACE union (workspace
   search ∪ serve-native search of the serve's own scripts dir — the #144
   capability-map shape) recovered every dot-dir case 3/3, and def-site-
   REQUIRED (a mention-only file never grounds) closed the comment hole.
   The 9 residual wrongs sit in 3/10 questions and every one grounds a
   REAL related file with honest attribution (`_encode_tool_calls` for
   "tool calls turned into chunks"; `_projected_tokens` — the very alias
   the caller computes with — for "projected token estimate"): the
   failure mode is answering an adjacent question, never fabrication and
   never a file that lacks the picked identifier.
6. **Arm E (full-tree `--hidden` sweep) was killed**: hundreds of
   hidden-inclusive rg calls over stale worktrees made menu-building
   pathologically slow, and a full-tree sweep is NOT the production
   surface anyway. Kept in the script as the recorded dead end.

## Consequences for #121's design

- The rung is: bare-symbol explain that fell through filename-glob
  (slice 1) AND self-discovery (#144) → ONE grep round (charset-safe
  alternation template from question stems — never model text) over the
  two-surface union → deterministic identifier menu from the rendered
  block (def-site required) → guarded closed-menu pick (decide-node
  pattern, abstain → conceptual fall-through) → read the def-site file →
  grounded explain attributing file + identifier.
- Open design levers, deliberately NOT tuned here (anti-overfit): menu
  ranking (frequency vs the spike-2 rarity signal — `_GLOB_STEM_RE`
  missed one menu on frequency rank), stem extraction sharing
  `_explain_stems`, and the grep render grammar (Found-N header,
  #148-style truncation semantics, #149 client-side-truncation flank).
- Wire format is already captured
  (`2026-07-13-opencode-run-captures/grep-tool-result-wire-format.txt`).

## Post-pre-flight addendum (arm G + cut-survival, 2026-08-13)

The v1 design pre-flight returned REDESIGN (three blockers: mention-volume
rendering made the rung a no-op — 145–3,212 lines vs a 50-line cap, 10/10
battery over cap; the defer_pick wiring could not ride the decide node's
clean-task input contract; the wire grammar was refuted by the actual
opencode binary — 100-match client cap with the header count computed FROM
the capped array, "No files found" for empty, suffix/footer truncation
markers, blank-line file groups). New measurements for v2:

- **Arm G (def-anchored two-shape pattern):** raw volumes drop to 60–286
  lines (vs 3,212), right file present in the filtered menu 8/10 — the
  two misses are the dot-dir questions (slice-B territory).
- **Cut-survival:** taking only the FIRST 100 matches (the client's own
  cap, rg walk order), the right file remains in the filtered menu
  **8/8** workspace questions.
- Ops note for spike reruns: rg inherits stdin; under a heredoc-driven
  python it silently searches the exhausted stdin pipe and returns
  nothing — pass stdin=DEVNULL (one measurement round was lost to this).

v2 design: def-anchored pattern, wire grammar rebuilt from the binary,
cut-tolerant menus with AST-verified read-time def confirmation carrying
the honesty invariant, and the pick as its own guarded DAG node.

## Ground-truth addendum (arms H/I, real Service.grep, 2026-08-13)

The v2 re-review (PROCEED-WITH-CHANGES) found the v2 evidence line was
still simulation: the real client grep TRAVERSES hidden dirs (not
dot-blind), its first-100 cut is NONDETERMINISTIC, and stale
.claude/worktrees copies were knocking the exit-gate identifier off its
own menu via duplicate def-sites. The reviewer also surfaced the real
measurement instrument: `opencode debug rg search` (the exact
Service.grep code path). After worktree cleanup and with the
deterministic serve-side surface filter (hidden-component paths dropped;
`.llm-orc/**` kept iff serving.self_reference):

- **Arm H (menu survival, 3 runs/question):** right file in the menu
  29/30 runs across ALL 10 questions — including both dot-dir questions,
  so slice B dissolves into the flag-gated surface rule. The one miss:
  cut-variance on one at-cap question.
- **Arm I (pick accuracy on the real 12–58-entry menus):** 27/30 right,
  0 fabrication, 0 off-menu, 0 abstain; the only wrong class is the
  defensible-alias grounding (`_projected_tokens` — the caller's own
  computing alias — for the projected-estimate question). Bigger real
  menus scored BETTER than the artificial 10-entry menus (19/30).

Design v2.1 folds in the re-review's seven changes. Arm H/I scripts:
scratchpad `arm_h.py`/`arm_i.py` (session artifacts; the battery, rules,
and numbers are recorded here and in the design doc).

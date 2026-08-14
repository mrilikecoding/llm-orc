# Content-grep meta-task rung (#121) — design v2

**Status:** v2.1. v1 pre-flight: REDESIGN (three measured blockers).
v2 re-review: PROCEED-WITH-CHANGES — the mechanism verified against the
engine and the client binary; the evidence line and search-surface
definition were rebuilt with the REAL instrument (`opencode debug rg
search`, the exact `Service.grep` code path the grep tool calls). All
seven requested changes are folded in below. Records on issue #121.

**Slice B dissolves.** Ground truth: the client's grep TRAVERSES hidden
dirs (it is not dot-blind like its glob), so `.llm-orc` definitions ride
the ordinary client round. The search surface is instead defined
deterministically serve-side (resolution 6): paths with a hidden
component are dropped — except `.llm-orc/**`, kept iff
`serving.self_reference` is on (the same opt-in and contamination logic
as #144; flag-off projects never see serve scripts in content menus).
With the flag on, both dot-dir battery questions ground through slice A
(measured: arm H/I).

**Evidence base:** `2026-08-13-content-grep-findings.md` (spike arms
A–G). Validated: deterministic harvest → closed menu of REAL identifiers
with definition-site files → cheap-seat pick with abstention →
find-or-refuse ladder (arm F: 21/30, zero fabrication). Refuted: model
proposes the pattern (0/30); mention-volume rendering (pre-flight F1:
145–3,212 match lines per battery question). Ground truth for v2.1, measured
with the REAL instrument (`opencode debug rg search` — the exact
`Service.grep` code path — 3 runs per question, post worktree-cleanup,
surface-filtered): right file in the menu **29/30 runs across all 10
battery questions** including both dot-dir questions (arm H); pick
accuracy on the real 12–58-entry menus **27/30, zero fabrication, zero
off-menu** (arm I) — the single wrong class is the defensible-alias
grounding. The one survival miss was cut-variance on one at-cap question
(the client's nondeterministic first-100).

## Pre-flight resolutions (bind the mechanism)

1. **Def-anchored wire pattern (closes F1).** The grep round matches
   DEFINITION-shaped lines only, never mentions:
   `^\s*(def|class)\s+[A-Za-z0-9_]*(stem1|…)[A-Za-z0-9_]*` alternated
   with the module-level assignment shape
   `^[A-Za-z0-9_]*(stem1|…)[A-Za-z0-9_]* *=` (both sides optional on
   BOTH shapes — re-review v2-F2), case-insensitive,
   `include: "*.py"`. Stems come from `_explain_stems` (charset-checked
   by construction); `[A-Za-z0-9_]*` on BOTH sides of the alternation so
   stem-initial identifiers match (F8), and the whole pattern is
   metacharacter-free beyond its own closed template. Measured: raw
   volumes 60–286 on the battery vs 3,212 for v1's mention pattern.
2. **Wire grammar from the binary, not the n=1 capture (closes F3).**
   opencode 1.17.15's grep: caps at 100 matches, header
   `Found <N> matches` with `<N>` computed FROM the capped array (count
   arithmetic is inert — v1's count-mismatch check is dropped), a
   ` (more matches available)` header suffix and/or a
   `(Results truncated. Consider using a more specific path or
   pattern.)` trailing footer when cut, `No files found` (not "Found 0")
   when empty, and BLANK LINES between file groups. The normalizer
   parses all three variants; the capture note file gets an amendment
   recording the refuted "N may be 0" claim (instrument-preconditions
   discipline: corrected, not edited away).
3. **Cut results still build menus (semantics pivot).** Glob truncation
   refuses because filename discovery makes a UNIQUENESS claim over a
   complete set (#148). The menu pick claims no uniqueness — the answer
   says "here is <file>, which defines <identifier>" — so a client-cut
   result may still seed the menu. Ground truth (re-review v2-F1): the
   client's first-100 cut is NONDETERMINISTIC (parallel traversal, no
   sort), so at-cap questions vary run to run; measured survival held
   29/30 runs (arm H). Honesty is carried by resolution 5's read-time
   confirmation plus a DETERMINISTIC truncation hedge: whenever the
   block is truncation-marked (the wire's suffix/footer/metadata signals
   or the serve's 50-line post-filter cap), the grounded dispatch
   instruction appends a fixed sentence directing the answer to note
   that the search was cut and other definitions may exist. The hedge is
   composed classify-side, never left to seat discretion, and a corpus
   test pins it (re-review v2-F3).
4. **The pick is its own guarded node (closes F2).** `defer_pick` does
   NOT ride the needs_decider empty-target convention (that would fire
   the seat-routing decide node with a static prompt and a foreign
   closed set). classify emits `defer_pick: true` plus a new
   `pick_input` field (the clean question + the menu, composed
   classify-side, bounded); serving.yaml gains a `pick` node —
   `when: ${classify.defer_pick}`, `input_key: pick_input`, cheap tier —
   whose closed instruction is "pick ONE identifier from the list or
   abstain". resolve validates the response by EXACT token membership in
   the menu (never `_decider_target`'s substring fallback — F10; the
   abstain sentinel is `none`, and a menu identifier literally named
   `none` is excluded from menus at build time), then MINTS the read for
   the def-site file — `needs_self_files` when the path is
   `.llm-orc/`-rooted (the #144 native seam: trusted disk bytes, no
   wire-normalizer heuristics, so resolution 5's AST parse sees real
   source — final-review change 1), `needs_files` otherwise — and
   threads `picked: <identifier>` through shape/form_gate (new
   pass-through field). Off-menu, abstain, or an
   unreadable pick response → conceptual fall-through (today's
   behavior). The decide node is untouched; `task` stays the clean turn.
   The defer_pick CHAIN row carries the non-empty target `explainer`
   (re-review v2-F4: an empty target would silently mint needs_decider
   and fire the seat-routing decide node); resolve upgrades it to
   `need-files` on a valid pick, so the fail-open default is the
   conceptual explainer by construction.
5. **Deterministic pass-4 re-grounding + AST def confirmation (closes
   F2's oscillation and F5's forged def-lines).** classify never learns
   the pick from the wire; on the read-continuation pass it re-derives
   deterministically: menu (recomputed from the still-rendered grep
   block) ∩ the visible read's path → grounded explain for that file,
   attributing EVERY menu identifier whose def-site is that file (no
   re-pick, no oscillation; visible → grounded, attempted-failed → the
   existing read_failed refusal, so `defer_pick` never fires twice).
   Before composing the grounded dispatch, classify `ast.parse`s the
   read block's body and confirms a FunctionDef/ClassDef/module-level
   assignment target actually bears each attributed identifier — a
   docstring or string literal containing `def foo` is not an AST
   definition, so a forged def-line can neither ground nor
   mis-attribute; identifiers that fail confirmation are dropped, and if
   none survive the turn refuses honestly (not-grounded with the
   mismatch reason). Unparseable read content (a broken .py) refuses
   rather than grounds.
6. **Menu build (deterministic, from the rendered block alone).** The
   SEARCH SURFACE is defined here, serve-side and deterministic
   (re-review v2-F1): drop every result path containing a hidden
   component (`.claude/` worktrees, `.venv`, any dot-dir) — except paths
   of the exact shape `.llm-orc/scripts/agentic_serving/<basename>.py`,
   kept iff `serving.self_reference` is on (scoped to the #144
   self-read label roots so the flag-on surface and the native read
   seam admit the same set — final-review change 1); then keep
   non-test, non-docs `.py` only. Result paths are RELATIVIZED against
   the workspace root before rendering (the wire carries absolute
   paths; relative lines halve the per-line charge and nearly double
   the ceiling's line budget — final-review change 2). From the surviving rendered
   def-lines: extract the defined identifier (the token after
   `def `/`class `, or the assignment target — the code span, never the
   `path: Line N:` prefix, F11), require it to CONTAIN a question stem
   (case-insensitive, matching the wire pattern — F9), and admit
   identifiers with exactly ONE def-site file. The menu is bounded by
   the block itself, NOT an artificial cap: under the strict production
   render (relative paths, 50-line + 4,096-char caps) menus run 12–44
   entries with right-file survival 28/30 runs (arm H-strict); the pick
   measured 27/30 on menus of that size span (arm I, 12–58 pre-cap)
   versus 19/30 on artificial 10-entry menus (arm D; arm F scored
   21/30); the rarity ranking stays a named untuned lever. Empty menu →
   conceptual fall-through.
7. **Grep/glob discrimination (closes F4).** The caller's grep tool
   calls always carry `include`, and `_is_grep_shaped` (pattern +
   include, no filePath/command) is checked BEFORE `_is_glob_shaped`
   everywhere a call shape routes (block mapping, `_resumes_turn`), so a
   grep echo can never render as a failed glob block and shadow the real
   listing. Echo validation mirrors the glob discipline with a closed
   regex over the issued template; a non-matching echo renders failed
   under a fixed safe token.
8. **Render budget (closes F7).** The rendered grep block is capped at
   50 post-filter lines AND a 4,096-char ceiling (the run-block
   precedent), so its worst-case projected-token charge (~1,200) fits
   the reserve derivation; the `_READ_TOKEN_BUDGET` comment gains the
   line-item. The #148 strip extends to truncated grep blocks in seat
   prompts.
9. **No config flag, justified by firing rate.** The rung adds one grep
   round + one cheap pick call ONLY on turns that today end conceptual;
   with the def-anchored pattern the round is small and the pick is one
   bounded cheap-tier call. Latency-class additivity, not
   contamination-class (#144's flag rationale) — flagless stands, and
   the live gate measures the added latency.

## Trigger (additive only)

is_explain, no named file, explain-discovery glob round COMPLETE with
zero candidates (workspace ∪ self), no grep block yet → `needs_grep`
(the stems). Grep block present → menu → `defer_pick` (or fall-through).
Pick validated → read round. Read visible → AST-confirmed grounded
explain. Every other routing byte-identical; each seam is
one-round-or-refuse so the turn shape is statically bounded at three
client rounds.

## Invariants (rule 6) and regression instruments

- **Grounding:** a grep-grounded answer's file ACTUALLY DEFINES every
  attributed identifier, verified by AST over the read content.
  Instruments: corpus tests — a docstring-forged def-line never grounds
  and never mis-attributes; unparseable read content refuses; a
  mention-only file never enters the menu.
- **Closed-set pick:** off-menu/abstain/unreadable → conceptual
  fall-through; exact-token validation only. Instruments: resolve tests
  with off-menu, substring-colliding, and `none` responses.
- **Pattern/echo safety:** only charset-checked stems enter the
  template; echo validation both directions. Instruments: unsafe-stem +
  echo-mismatch tests mirroring the glob suite.
- **Wire-grammar fidelity:** all three captured binary variants
  (suffix-truncated, footer-truncated, `No files found`) normalize
  correctly; blank-line file groups survive. Instruments:
  crafted-result tests per variant.
- **Surface determinism:** hidden-component paths never enter menus;
  `.llm-orc` paths enter iff the flag is on. Instruments: crafted
  results carrying `.claude/worktrees/...` and `.llm-orc/...` paths,
  flag-on and flag-off.
- **AST definition semantics:** FunctionDef, ClassDef, module-level
  Assign AND AnnAssign targets count as definitions at confirmation
  time (the wire pattern's assignment shape cannot match AnnAssign, so
  such constants simply never enter menus — consistent, documented);
  the read block body de-indents back to parseable source (round-trip
  pin). Residual, documented: a forged def-line can still DENY a menu
  entry (a liveness cost, never an honesty cost).
- **Echo template:** accepts the `(?i)` prefix the case-insensitive
  pattern carries.
- **No re-pick:** the pick node runs at most once per turn.
  Instrument: corpus test asserting the read-continuation pass emits
  grounded/refusal, never defer_pick.
- **Additivity:** corpus regression probes over non-fall-through
  routings.

## Exit gate & validation

TDD per the instruments; then live (RIG): "where is the recall ledger
built?" through real OpenCode → glob (zero candidates) → grep → menu →
pick → read `serving_ensemble_caller.py` → grounded answer attributing
`_recall_ledger` (AST-confirmed); plus one abstain-shaped question and
one truncated-result question. Rig-state note (final-review change 4):
the gate question's under-cap determinism holds at the measured rig
state — the client cut is pre-filter, so accumulated hidden-dir noise
(future worktrees) can push it back over 100, where the nondeterministic
cut and the truncation hedge take over honestly; the live record states
the rig state it ran at. Ladder rerun + trajectory row +
adversarial review with the wrong-accept hunt: does any turn ground a
file that does not AST-define its attributed identifiers?

## Not built here

Slice B (serve-native grep half); menu rarity ranking; grep→build; the
multi-def-site read fan (still the `max_rounds` trigger).

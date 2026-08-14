# Content-grep meta-task rung (#121) — design v2

**Status:** v2 after reviewer pre-flight (verdict on v1: REDESIGN — three
blockers, all adjudicated below with new measurements; the pre-flight
record is on issue #121 and in the session transcript). The doctrine-9
composition survives from v1; the mechanism is rebuilt.

**Slice A** (this design): the client-grep workspace surface. **Slice B**
(named, not built): the serve-native half of the two-surface union (grep
over the serve's own scripts, the #144 pattern) — the two battery
questions whose subjects live under `.llm-orc` cannot ground in slice A
and fall through conceptually, exactly as today.

**Evidence base:** `2026-08-13-content-grep-findings.md` (spike arms
A–G). Validated: deterministic harvest → closed menu of REAL identifiers
with definition-site files → cheap-seat pick with abstention →
find-or-refuse ladder (arm F: 21/30, zero fabrication). Refuted: model
proposes the pattern (0/30); mention-volume rendering (pre-flight F1:
145–3,212 match lines per battery question). New measurements for v2:
the def-anchored two-shape pattern keeps the right file in the
post-client-cut menu **8/8** workspace battery questions (raw volumes
60–286, client cap 100, rg walk order — measured 2026-08-13).

## Pre-flight resolutions (bind the mechanism)

1. **Def-anchored wire pattern (closes F1).** The grep round matches
   DEFINITION-shaped lines only, never mentions:
   `^\s*(def|class)\s+[A-Za-z0-9_]*(stem1|…)[A-Za-z0-9_]*` alternated
   with the module-level assignment shape
   `^[A-Za-z_][A-Za-z0-9_]*(stem1|…)[A-Za-z0-9_]* *=`, case-insensitive,
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
   result may still seed the menu. Measured: the right file survives the
   100-match cut 8/8. Honesty is carried by resolution 5's read-time
   confirmation, and the render marks the block `(truncated)` (from the
   wire's own suffix/footer/metadata signals or the serve's 50-line
   post-filter cap) so classify can state incompleteness in the
   dispatch instruction — the answer may note other definitions exist.
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
   `none` is excluded from menus at build time), then MINTS
   `needs_files=[def_file]` and threads `picked: <identifier>` through
   shape/form_gate (new pass-through field). Off-menu, abstain, or an
   unreadable pick response → conceptual fall-through (today's
   behavior). The decide node is untouched; `task` stays the clean turn.
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
6. **Menu build (deterministic, from the rendered block alone).** Parse
   each rendered def-line: extract the defined identifier (the token
   after `def `/`class `, or the assignment target), keep lines from
   non-test non-docs `.py` files, require the identifier to CONTAIN a
   question stem (case-insensitive, matching the wire pattern's
   semantics — F9), and admit identifiers with exactly ONE def-site file
   among the rendered lines. Cap 10, first-occurrence order (the rarity
   ranking stays a named untuned lever). Path tokens never enter: the
   identifier is parsed from the def-line's code span, not the
   `path: Line N:` prefix (F11). Empty menu → conceptual fall-through.
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
one truncated-result question. Ladder rerun + trajectory row +
adversarial review with the wrong-accept hunt: does any turn ground a
file that does not AST-define its attributed identifiers?

## Not built here

Slice B (serve-native grep half); menu rarity ranking; grep→build; the
multi-def-site read fan (still the `max_rounds` trigger).

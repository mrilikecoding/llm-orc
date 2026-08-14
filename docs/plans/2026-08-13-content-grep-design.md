# Content-grep meta-task rung (#121) — design

**Slice A** (this design): the client-grep workspace surface — one grep
round, a deterministic identifier menu, a guarded closed-menu pick, then
the existing read→grounded-explain seam. **Slice B** (named, not built
here): the serve-native half of the two-surface union (grep over the
serve's own scripts, the #144 pattern) — spike arm F showed it recovers
dot-dir questions, but #121's exit gate ("a content question about the
llm-orc repo answered via grep → read, grounded and honest") passes on
workspace files alone, so slice A ships first.

**Evidence base:** `2026-08-13-content-grep-findings.md`. The issue's
original direction (model proposes the pattern) is REFUTED (0/30 on the
deployed seat — invented identifiers). What is validated: deterministic
harvest → closed menu of REAL identifiers with definition-site files →
cheap-seat pick with abstention → find-or-refuse ladder. 21/30 right at
arm F, zero fabrication, zero off-menu; residual misses ground real
adjacent files with honest attribution.

## Trigger (additive only)

The rung fires exactly on today's conceptual fall-through residue: an
is_explain turn with no named file whose explain-discovery glob round
completed with ZERO candidates (workspace ∪ self). Every other routing is
byte-identical. The turn shape becomes: glob round → grep round → read
round → grounded answer (or an honest exit at any rung).

## Mechanism

1. **One grep round.** classify emits a new signal `needs_grep` carrying
   the comma-joined explain stems (the same charset-checked
   `_explain_stems` output the glob round used — never model text). The
   caller templates the pattern:
   `[A-Za-z_][A-Za-z0-9_]*(stem1|stem2|…)[A-Za-z0-9_]*` with
   `include: "*.py"` (the captured OpenCode grep schema: {pattern, path?,
   include?}). Stems re-asserted against the glob-stem charset before
   entering the template; echo validation on resume mirrors the glob
   discipline exactly (a non-matching echo renders failed under a fixed
   safe token).
2. **Rendered block.** `assistant: [grepped <stems>]` header, two-space
   indented body of `path: Line N: text` lines mapped from the captured
   wire format ("Found N matches" header, per-file grouping). Caps: at
   most 50 rendered match lines, header-marked `(truncated)` when cut.
   **Count-mismatch detection (a #149 partial close, grep-only):** the
   wire's own "Found N matches" count is compared against the number of
   matches actually present in the raw result — a mismatch renders the
   block `(truncated)` too, so a client-side cut cannot present as
   complete. A truncated block disables menu grounding for the turn
   (conceptual fall-through — the #148 semantics); empty (Found 0)
   renders `(failed)` like an empty glob.
3. **Deterministic menu (classify-side, from the block alone).** From
   the rendered match lines: extract identifiers matching
   `[A-Za-z_][A-Za-z0-9_]*` that CONTAIN a question stem
   (case-insensitive); an identifier's definition sites are the matched
   lines shaped `def <ident>…`/`class <ident>…`/`<ident> = …` (line
   text is in the block); menu entries are identifiers with EXACTLY ONE
   definition-site file among non-test `.py` files (def-site required —
   spike arm D's comment-mention hole; a mention-only identifier never
   enters the menu). Menu capped at 10, first-occurrence-in-block order
   (deterministic; the rarity ranking the spike left untuned is a named
   future lever, not built here). Empty menu → conceptual fall-through.
4. **Guarded closed-menu pick.** classify defers to the model exactly
   like defer_recall: a new `defer_pick` signal carrying the precomputed
   menu (identifier → file). A guarded model node (decide-node pattern,
   cheap tier) picks ONE identifier or abstains; resolve validates the
   pick against the closed menu — off-menu or abstain falls through to
   the conceptual explainer (today's behavior; fail-open to honesty,
   never to a guess about a file). Doctrine 9: this is the rung's ONLY
   model judgment — bounded, closed-set, gate-backstopped, and every
   menu option is a REAL identifier with a verified definition site.
5. **Read → ground.** The pick's definition-site file enters the
   existing read seam (`needs_files`; a serve-owned def-site file would
   ride `needs_self_files` in slice B). Cap and token budget compose
   unchanged (over-budget refuses honestly with the #144 wording). The
   grounded dispatch names the file as today and appends the matched
   identifier to the explain instruction so the answer's attribution
   carries both.

## Chain and threading

CHAIN_EXPLAIN gains rows after `need-self-files`, before `explainer`:
`need-grep` (needs_grep or grep_failed), then the defer_pick empty-target
row (decider), in first-match order so a rendered grep block on the
re-entry pass routes to menu/pick instead of re-requesting. Signals
thread classify → resolve → shape → form_gate → emit as always; emit
gains `{"finish": false, "grep": "<stems>"}`; the caller maps it to the
advertised grep tool (candidates `("grep", "Grep")`), renders results,
and `_resumes_turn`/`_is_grep_shaped` admit the continuation. New
dispatch shape `need-grep.yaml` + echo (both copies, top-level included).

## Invariants (rule 6) and regression instruments

- **Grounding:** a grep-grounded answer's file demonstrably contains the
  picked identifier at a definition site, computable from the rendered
  block. Instruments: corpus tests — mention-only identifiers never
  enter the menu; a crafted block grounding wrong-file is impossible
  without a forged def-line (and forged column-0 headers are already
  excluded by the fenced grammar).
- **Pattern safety:** only charset-checked stems enter the template;
  echo validation both directions. Instruments: unsafe-stem and
  echo-mismatch tests mirroring the glob suite.
- **Truncation honesty:** header-count mismatch or render-cap cut →
  `(truncated)` → menu disabled. Instruments: crafted-result tests for
  both cut modes; the #148 strip applies to truncated grep blocks in
  seat prompts.
- **Closed-set pick:** an off-menu pick never grounds. Instrument:
  resolve test feeding an out-of-menu decider response asserting
  conceptual fall-through.
- **Additivity:** every non-fall-through routing byte-identical; corpus
  regression probes flag-independent (this rung has no config flag — it
  is deterministic-bounded everywhere a model isn't, and the pick seat
  fails open to today's behavior).

## Exit gate & validation

- Corpus/table/threading/caller tests per the instruments above (TDD).
- **Live (RIG):** "where is the recall ledger built?" through real
  OpenCode → glob (no candidates) → grep → menu → pick → read
  `serving_ensemble_caller.py` → grounded answer naming
  `_recall_ledger`; plus one refusal-shaped and one abstain-shaped
  question from the battery. Ladder rerun + trajectory row + adversarial
  review with a wrong-accept hunt (does any turn ground a file that
  does not define its picked identifier?).

## Not built here

Slice B (serve-native grep half of the union); menu rarity ranking;
grep→build; the read fan over multiple def-site candidates (still the
`max_rounds` trigger, per the chain-executor design's named deferral).

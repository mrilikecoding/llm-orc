# glob→read grounded-explain (meta-task rung, slice 1) — design

**Status:** LANDED 2026-07-14 (branch `feat/glob-read-grounded-explain`),
validated live + author-independent review APPROVE. Supersedes the roadmap's
"grep→read first" framing: a real-repo spike refuted deterministic content-grep,
so the reliable deterministic rung is glob-by-filename first; content-grep
becomes rung 2 (Approach B). Evidence: `docs/plans/2026-07-14-grep-read-spikes/`.

**Two decisions/corrections after review (this section is the source of truth,
overriding the pre-implementation wording below):**
1. **0-match → conceptual fall-through, NOT refuse** (practitioner decision).
   A bare-symbol explain that matches no file answers from general knowledge
   (today's behavior preserved), so the slice only ADDS grounding. Older text
   below saying "0-candidate refuses honestly" is superseded — a 0-match repo
   symbol that isn't a filename can still get a conceptual (speculative) answer;
   closing that is rung 2 (content-grep).
2. **Wrong-file bound = the question names EVERY significant filename
   component** (`_explain_glob_candidates`: basename components, len≥3, split on
   non-alphanumeric, must be ⊆ the question stems). The original substring-union
   grounded confident answers on unrelated files (adversarial review BLOCKER:
   "context management" → `project_context.py`, "error handling" →
   `structural_errors.py`, ~20% of a real-repo battery); the subset rule closed
   it (0/15 wrong-file on re-review). **Known limit (tracked in roadmap WS-3
   rung 1):** this is precision-over-recall — partial naming ("the dispatcher" →
   `agent_dispatcher.py`) now misses (~53% of multi-component files), falling
   through to conceptual. Recovery = distinctive-component matching (the spike's
   file-rarity signal).

**Live validation (real OpenCode, branch HEAD):** "how do chunks work?" →
glob→read `src/llm_orc/web/serving/chunks.py` → grounded, accurate (named the
real `ContentDelta`/`ClientToolCall`/`OrchestratorChunk` classes). "how does
context management work?" → 0 candidates → conceptual, no wrong-file grounding.
**Known limitation:** opencode glob can't reach `.llm-orc/` (dot-dir), so the
literal self-referential gate ("classify") is deferred to the apex; validated
on normal-dir `src/llm_orc/` instead. Minor: "work" leaks as a stem (harmless,
close in a stopword pass).

## Goal

Let the serve answer a question about a real repository whose subject is a
code symbol, by finding the file named after that symbol, reading it, and
explaining it grounded in real content. Exit gate (the meta-task rung's
entry): "how does classify decide routing?" answered through the serve via a
glob→read chain, grounded in `classify.py`'s actual content, honest.

## What it intercepts (this is an honesty fix, not just ergonomics)

The seam map corrected a key assumption: today a bare-symbol explain does NOT
refuse, it **speculates**. `not-grounded` fires only when an explain names a
file *with an extension* (`_FILE_RE`, `classify.py:98-100`) that isn't visible.
"how does classify decide routing?" names no extension, so `named_file=""`,
`explain_ungrounded` stays False (its gate is `and named_file`,
`classify.py:780-783`), and CHAIN_EXPLAIN's `explainer` row (fires on any
`is_explain`, `chain_plan.py:262-294`) routes it to the conceptual seat, which
guesses about "classify" with no file content. So slice 1 closes a live
guessing hole; even when it can't resolve a file it replaces speculation with
an honest refusal.

## Mechanism (deterministic, one glob round)

1. **Detect explain-discovery:** `is_explain` AND no `named_file` (no
   extension) AND ≥1 candidate stem extracts. This is exactly the
   currently-speculating case.
2. **Extract candidate stems** (`_explain_stems(task)`, new): identifier
   tokens `[a-z_][a-z0-9_]*`, len ≥ 3, minus a general-English stopword set
   (interrogatives/articles/prepositions/auxiliaries — NOT code terms;
   reuse the spike's `STOP`). "how does classify decide routing?" →
   `{classify, decide, routing}`.
3. **One glob round, brace-alternation:** emit `**/*{classify,decide,routing}*`.
   Captured 2026-07-14: opencode 1.17.15 glob supports brace expansion
   (`**/*{metrics,calc}*` → both files). Stems are charset-checked, so the
   pattern is safe to template. Single stem → `**/*classify*` (no braces).
4. **Candidate rule (reuse the existing discipline):** filter the globbed
   block by basename-contains-a-stem AND `.py` AND not `test_*`
   (`_globbed_candidates`, `classify.py:399-426`), then **exactly-one-or-
   refuse**. One → read it. Zero → honest refuse. >1 → honest
   refuse-with-candidates.
5. **Read → ground:** the single candidate reads via the existing need-files
   seam, becomes a `[read <path>]` visible block, and the **existing** grounded
   injection (`classify.py:904-916`) produces the answer grounded in real
   content. No new seat or ensemble YAML.

For the gate question: `**/*{classify,decide,routing}*` → `classify.py`,
`test_serving_classify.py`, `routing-demo.yaml` → filter `.py`/not-`test_` →
`classify.py` (one) → read → grounded.

## Change surface (grounded in the seam map)

1. **`_explain_stems(task)`** (new, `classify.py`): the extractor. `_module_stem`
   (`:383-396`) is build/test-phrased and won't match explain wording, so this
   is separate.
2. **Explain-discovery branch** in `_discover_and_read` (`classify.py:694-731`):
   runs when `is_explain and not named_file and stems`. Lift the `if not
   is_explain` guard at `:811` **only for this branch** (do not open it
   globally). It sets `needs_glob` to the brace-inner stem list and, after the
   glob returns, `needs_files` to the single candidate — reusing the shared
   signal strings, disambiguated from build discovery by `is_explain` in the
   chain guards (below). Do NOT loosen the build `wants_existing` gates
   (`_discovery:441`, `_files_to_request:335-339`); keep build discovery
   byte-unchanged.
3. **Brace pattern build:** `serving_ensemble_caller.py:845` currently
   `f"**/*{glob_stem}*"` (single stem, no literal braces). Extend so an
   explain-discovery glob with multiple stems emits literal braces
   `**/*{a,b,c}*`; single stem stays `**/*a*`. Isolated one-spot change; build
   discovery path unchanged.
4. **Two new `Step` rows in `CHAIN_EXPLAIN`** (`chain_plan.py:262-294`),
   modeled on `CHAIN_BUILD` rows 0–1 (`:298-312`): a `need-glob` row and a
   `need-files` row, guarded on `is_explain AND (needs_glob/needs_files or
   their failed variants)`. **Placed after `recall-answer`/`not-grounded`,
   before the `explainer` row** — the first-match scan means the explainer
   would short-circuit discovery otherwise (map §3).
5. **Reuse unchanged:** `need_glob_echo`/`need_files_echo` shapes, `emit.py`/
   `shape.py` request emission (chain-agnostic), `_globbed_candidates`, the
   grounded injection at `classify.py:904-916`, the `explainer` seat.

## Honesty properties

- Grounded-or-refuse: an answer is always grounded in a real read file
  (`:904-916`), or the turn honestly refuses. No new speculation path.
- Strict improvement: the 0-candidate and >1-candidate cases both refuse
  honestly where today the bare-symbol turn speculates.
- Wrong-file bound: one-or-refuse + the "file named after the symbol"
  heuristic. If the symbol names one clean `.py`, we explain that file's real
  content; we never fabricate. Ambiguity refuses rather than guesses.
- Deterministic end to end: extraction, glob, filter, and grounding are all
  code; no model judges which file or what it does.

## Ordering & isolation

- CHAIN_EXPLAIN row order becomes: `recall-answer` → `not-grounded` →
  (`defer`) → **`need-glob`(explain)** → **`need-files`(explain)** →
  `explainer`. A symbol that *also* names a visible file keeps the existing
  grounding-in-place path (it has `named_file`, so explain-discovery's
  `not named_file` guard skips it).
- Build discovery (CHAIN_BUILD rows 0–1) is untouched; the new rows carry
  `is_explain` in their guards, and build turns never set `is_explain`, so the
  two never cross. The subprocess classify suite and the `test_chain_plan.py`
  table contract anchor this.

## Exit gate & validation

- **Unit/table:** `test_chain_plan.py` — row-tuple test for the two new rows +
  an ordering test proving they fire before `explainer` (template at
  `:277-286`). `test_serving_classify.py` (subprocess driver) — the
  end-to-end behavior corpus: gate question → glob→read→grounded; 0-match →
  honest refuse; >1-match → refuse-with-candidates; a build discovery
  regression probe (unchanged).
- **Live (the real-world gate):** serve against a checkout of llm-orc itself;
  ask "how does classify decide routing?" through real OpenCode; assert the
  answer is grounded in `classify.py`'s real content (names `advance`/the
  routing table, not invented behavior). Plus a spread of the spike's
  questions to map where it honestly refuses.
- **Ladder + review:** rerun the 13-turn battery (no regression), trajectory
  row, author-independent adversarial review with a wrong-accept hunt (does
  any turn ship a grounded-sounding answer about the wrong file?).

## Deferred / next rungs

- **Content-grep (rung 2, Approach B):** for symbols that are NOT filenames
  (routing→`_route`, normalize→a method). A guarded model proposes the search
  pattern, charset-sanitized, grep find-or-refuse as the structural backstop.
  Built when a ladder question needs it.
- **Read-fan over >1 candidates + the round-budget backstop:** slice 1 refuses
  on >1; reading several candidates is the first genuinely unbounded chain and
  lands with the deferred `max_rounds`/`rounds_consumed`.
- **grep→build:** search-then-build, later.

## Evidence

- Spikes (why glob-first): `docs/plans/2026-07-14-grep-read-spikes/`.
- Wire captures (`--format json`, grep + glob-brace formats):
  `docs/plans/2026-07-13-opencode-run-captures/`.
- Seam map: this session's mapping agent (file:line surface above).

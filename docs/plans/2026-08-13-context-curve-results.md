# qwen3:8b effective-context curve — results (issue #139)

Spike script: `docs/plans/2026-08-13-context-curve-spike.py`. Run: local
Ollama, qwen3:8b, `/api/chat`, temperature 0, think off, `num_predict=150`.
192/192 calls succeeded (0 transport errors). The script's stdout (full
per-call log) was captured to a scratch file for this write-up; its own
per-call JSONL crash-recovery log (needle, level, placement, chars,
`prompt_eval_count`, wall-clock, correct, raw answer) was written to the OS
temp dir, per the script's design — neither is a repo artifact.

## Pre-registered design (as run — no deviations)

- **Drift threshold**: ~5% accuracy drift vs. a zero-context baseline —
  issue #139's own operational threshold for this spike, citing MECW (arXiv
  2509.21361) for the motivating phenomenon (severe degradation reported by
  1,000 tokens for several frontier models, effective windows up to 99%
  smaller than claimed). **Checked against the paper itself** (full PDF
  text, not just the abstract): MECW does **not** state a numeric "5%"
  threshold anywhere in its methodology. Its formal definition (Appendix
  A2.1) is qualitative — "the maximum token count, for a given problem type,
  before the model performance begins to degrade in a measurable fashion" —
  and its own significance test is p-value based, not a percentage-point
  cutoff. The 5% figure is issue #139's own operationalization, not a MECW
  number; MECW is cited correctly here for "severe degradation exists and is
  worth measuring on our own model", not for the specific threshold.
- **Context levels**: 500 / 1,000 / 2,000 / 4,000 / 8,000 tokens, sized by a
  chars/4 estimate for the target budget. Ollama's own `prompt_eval_count`
  was recorded per call as ground truth (see Calibration below) — the
  estimate ran 7-13% high of actual at low levels and 7% low of actual at
  the top level, close enough that "500/1K/2K/4K/8K" remain meaningful
  labels.
- **Exit gate** (verbatim, issue #139): *"qwen3:8b within ~5% of baseline at
  4KB means the current cap is defensible and gets a citation; severe
  degradation below 4KB means tighten the budget (and re-run the ladder to
  check score impact). Either way the selection budget stops being an
  unexamined constant."*
- **Serve-shaped context**: rendered in the exact grammar
  `serving_ensemble_caller._render_context` / `_indent_body` /
  `_render_write` produce (`assistant: [wrote <path>]` header, two-space
  indented body), wrapped exactly as `classify.py` composes
  `dispatch_input` (`"Conversation so far:\n{conversation}\n\nCurrent
  request: {task}"`). Filler ("haystack") content is real repo file bodies
  — 11 files from `.llm-orc/scripts/agentic_serving/` and
  `src/llm_orc/web/serving/` / `src/llm_orc/models/`, 207KB total, read live
  off disk — interleaved with plausible short turn exchanges, never
  synthetic text.
- **Probe**: 12 needle facts (function return value, config constant,
  string in a raised exception, filename named in a turn line,
  comment-embedded fact), each with an exact-match scoring rule (answer must
  contain the literal, case-insensitive). Verified none of the 12 chosen
  literals collide with any substring in the filler pool. 12 needles × 5
  levels × 3 placements (start/middle/end) = 180 grid calls + 12
  zero-context baseline calls = 192 total. All 3 placements were run at
  every level (the task allowed dropping to 2 placements if latency
  demanded it — it didn't: the whole run completed in well under 30
  minutes).
- **Model call**: qwen3:8b, temperature 0, think off — matching
  `.llm-orc/profiles/agentic-tier-cheap-general.yaml` with the `think:
  false` override every interactive cheap-tier seat in this repo uses
  (explainer.yaml, test-writer.yaml, adequacy-judge.yaml, serving.yaml's
  `decide` node). Temperature 0 is this spike's own addition for
  reproducible scoring, not a serve default. Confirmed via `ps` that the
  Ollama-managed `llama-server` process for this model was launched with
  `-c 40960` (40,960-token context window) — so no level tested here (max
  ~8,000 target / ~7,449 actual tokens) silently truncated at the
  inference-engine level.

## Curve table

| level (target tok) | n | accuracy | drift vs. baseline | start | middle | end |
|---:|---:|---:|---:|---:|---:|---:|
| baseline | 12 | **100.0%** | — | — | — | — |
| 500 | 36 | 100.0% | +0.0% | 100.0% | 100.0% | 100.0% |
| 1,000 | 36 | 100.0% | +0.0% | 100.0% | 100.0% | 100.0% |
| 2,000 | 36 | 100.0% | +0.0% | 100.0% | 100.0% | 100.0% |
| 4,000 | 36 | 100.0% | +0.0% | 100.0% | 100.0% | 100.0% |
| 8,000 | 36 | 100.0% | +0.0% | 100.0% | 100.0% | 100.0% |

Zero misses anywhere in the grid (192/192 correct). Spot-checked a sample of
raw answers at the 8K/middle cell (the hardest cell by construction — needle
buried mid-context at the largest size) — every answer is a specific,
correctly-attributed retrieval, not a lucky hallucination that happens to
contain the right digits, e.g.:

- `checksum_error` (8K/middle): *"The checksum value in the ValueError
  message is "0xE93B7"."*
- `schema_version` (8K/middle): *"The SCHEMA_VERSION in migrations/state.py
  is "9.14.203"."*
- `shard_count` (8K/middle): *"The shard count is pinned at 44710."*

No needle ever got confused with another needle's literal.

### Per-needle first failing level

All 12 needles: **never fails (0-8K)**. No needle shows even a single
miss across 15 cells (5 levels × 3 placements) each.

### Calibration: chars/4 estimate vs. actual `prompt_eval_count`

| level (target tok) | mean chars | mean actual tokens | chars/4 estimate | actual/estimate |
|---:|---:|---:|---:|---:|
| 500 | 1,982 | 561 | 495 | 1.13 |
| 1,000 | 3,969 | 1,038 | 992 | 1.05 |
| 2,000 | 7,974 | 1,964 | 1,994 | 0.99 |
| 4,000 | 15,973 | 3,731 | 3,993 | 0.93 |
| 8,000 | 31,968 | 7,449 | 7,992 | 0.93 |
| baseline | 297 | 148 | 74 | 2.0 |

The chars/4 rule tracks actual token counts reasonably (within ±13%) across
the tested range for code-heavy content, drifting slightly low (denser
tokenization, more multi-char tokens) as content grows — the "500/1K/2K/4K/8K"
labels are nominal targets, not exact; actual sizes are what's reported
above and in the raw log. (Baseline is a small, mostly-symbolic snippet so
the ratio is noisier there — expected, and irrelevant to the level curve.)

### Wall-clock (seconds): mean / median / max

| level | mean | median | max |
|---:|---:|---:|---:|
| baseline | 1.12 | 0.97 | 2.63 |
| 500 | 1.64 | 1.73 | 2.36 |
| 1,000 | 2.01 | 2.35 | 2.80 |
| 2,000 | 3.04 | 3.88 | 4.36 |
| 4,000 | 5.18 | 6.89 | 7.91 |
| 8,000 | 14.11 | 15.94 | 27.92 |

Total run time: well under 30 minutes for all 192 calls, on the rig's local
Ollama, free.

## Exit-gate verdict (issue #139's own terms)

> "qwen3:8b within ~5% of baseline at 4KB means the current cap is
> defensible and gets a citation; severe degradation below 4KB means
> tighten the budget."

Drift at 4KB: **+0.0%** (100.0% vs. a 100.0% baseline).

**VERDICT: DEFENSIBLE.** qwen3:8b shows zero measured drift from its
zero-context ceiling at every level tested, including 8KB (double the
current cap) — nowhere close to the ~5% threshold. The current 4KB /
8-turn selection cap in `serving_ensemble_caller.py`
(`_CTX_MAX_MESSAGES=8`, `_CTX_TAIL_CAP=4000`, `_CTX_SELECTED_CAP=4000`) is
measured-defensible for this model on this task shape, not merely assumed.
Citation: MECW (arXiv 2509.21361) motivated the check by reporting severe
degradation by 1,000 tokens for several frontier models on harder tasks
(counting, sorting, multi-hop lookup across many needles); qwen3:8b does not
replicate that degradation on single-needle verbatim retrieval — the shape
of context recall the serve's context window actually depends on (recovering
one specific prior fact: a file's written content, a turn's stated
filename), not the paper's harder multi-hop/aggregation tasks. The MECW
citation belongs to "context degradation is real and worth measuring
per-model", not to "our cap needs to be 1,000 tokens" — this spike is the
evidence that on our actual model and our actual task shape, it doesn't.

**Scope caveat**: this result is a ceiling-finding, not a ceiling-locating
result — accuracy never dropped, so the spike found *no* onset of
degradation within 0-8KB; it did not find where degradation would start had
we gone further (16K/32K, up to the model's real 40,960-token engine
window). The cap is defensible at 4KB specifically because 4KB is nowhere
near the (unmeasured, but at-least-8K) point where this model's simple-
retrieval accuracy would start to slip.

## Implication for #145

#145 is about repo-scale reads and chunk sizing. This spike measured a
single, specific task shape — "does one previously-seen fact survive
verbatim in context and get correctly recalled" — and found qwen3:8b holds
that fully out to at least 8,000 tokens (~32KB of raw code text) with zero
drift, regardless of where in the context the fact sits. That's good news
for #145's chunk-size budget in one direction: a seat can be handed 8K
tokens / ~32KB of real file content in one shot without measurable
verbatim-retrieval loss, so chunk sizes at or below that mark don't need a
degradation discount baked in. It's not permission to go unboundedly large,
though, for two reasons this spike didn't test: first, the accuracy ceiling
here is a floor on the model's *true* effective window, not a measured
ceiling — the onset could be anywhere at or above 8K, so #145 chunking
strategies that want to lean on headroom past 8K should re-run this same
ladder at 16K/32K before assuming it holds; second, this probe tested
single-needle recall, not the multi-fact synthesis/aggregation MECW's harder
tasks (and #145's actual repo-read use case, which may need to reason over
*several* chunked facts at once, not just recall one) — that's a
qualitatively harder task shape this spike didn't measure and shouldn't be
assumed to inherit the same zero-drift result.

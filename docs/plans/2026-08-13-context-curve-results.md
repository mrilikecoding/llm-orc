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

## Extension results (16K/32K + synthesis)

Follow-up to the #139 record above (closed, unmodified), run for #145
design grounding. Script: `docs/plans/2026-08-13-context-curve-ext-spike.py`
— a sibling that imports `NEEDLES`/`FILLER_TEXT`/`build_grid_input`/`score`
from the original #139 script rather than duplicating it. 66/66 calls
correct (48 grid + 18 synthesis), 0 errors.

### num_ctx precondition check (run first, per instruction)

Concern: qwen3:8b's Ollama-served context window might default to
something smaller than the probe (commonly 4096/8192), which would mean
the original #139 grid (max 8,000 target tokens) silently truncated and its
flat curve would be invalid. Checked two ways:

- **`ollama show qwen3:8b`**: architectural max context length = **40,960**
  tokens.
- **Empirical probe** (reproducible — the script's own first step): a
  ~20,018-estimated-token filler-only prompt sent with **no** `num_ctx`
  override, exactly how the original #139 script called the API. Server-
  reported `prompt_eval_count`: **18,635** actual tokens — a +6.9% shortfall
  vs. the chars/4 estimate, consistent with the same estimation noise
  already documented in #139's own calibration table, not truncation. A
  truncating server would have clipped `prompt_eval_count` to some fixed
  ceiling (e.g. ~4,096 or ~8,192) regardless of what was sent; it didn't.

One red herring worth recording so it doesn't get re-tripped later: the
agent's own interactive shell reported `OLLAMA_CONTEXT_LENGTH=8192` in
`env`. That variable belongs to the shell, not to the actual serving
process — `ps eww` on the real `Ollama.app`/`ollama serve` process (pid
4525/4962) showed `OLLAMA_CONTEXT_LENGTH=262144`, and the live
`llama-server` instance was independently observed running with `-c 40960`
(the model's own architectural ceiling, min'd against the much larger env
default). Checking the shell's own env would have given a false-positive
truncation scare.

**VERDICT: NO TRUNCATION.** The original #139 grid (max 8,000 target
tokens, ~7,449 actual) sat well inside the model's real 40,960-token
window — its flat, zero-drift curve stands unmodified. Every call in this
extension additionally pins `options.num_ctx=40960` explicitly anyway, as
belt-and-suspenders rather than a correction.

### Extended grid: 16K / 32K tokens, start/end placement

Middle placement dropped per the coordinator's explicit call to keep the
count down — a pre-registered deviation from #139's 3-placement design,
noted here rather than silently applied.

| level (target tok) | n | accuracy | start | end |
|---:|---:|---:|---:|---:|
| 16,000 | 24 | 100.0% | 100.0% | 100.0% |
| 32,000 | 24 | 100.0% | 100.0% | 100.0% |

All 12 needles correct at both new levels and both placements — the flat
curve from #139 (500-8,000) extends cleanly through 32,000 tokens, double
the model's demonstrated-safe range from the original record.

### Synthesis smoke: 4K / 8K / 16K, both-facts-required

6 paired questions (from the original 12 needles, recombined), each scored
correct only if the answer contains **both** literals.

| level (target tok) | n | both-correct |
|---:|---:|---:|
| 4,000 | 6 | 100.0% |
| 8,000 | 6 | 100.0% |
| 16,000 | 6 | 100.0% |

**No evidence synthesis degrades ahead of recall** within the tested range:
both facts, planted at opposite ends of the context (one near the start,
one near the end — the maximally-hard placement for combination), were
correctly combined in every trial at every level tested. This directly
answers the multi-fact caveat the original results doc flagged as
untested.

### Wall-clock: a bimodal cold/warm pattern, not a smooth curve

The mean/median hide a real bimodal split — worth reporting split, not
averaged. (One data-hygiene note: the crash-recovery JSONL had one stale
duplicate row from a first launch attempt that hit the tool's default 120s
foreground timeout and was killed before any output flushed — caught by an
inline dedup check, last-write-wins by `(level, placement, needle)`, before
computing these stats; it did not change the accuracy verdict, only
cleaned up one wall-clock cell.)

| level | cold n | cold mean (range) | warm n | warm mean (range) |
|---:|---:|---:|---:|---:|
| 16,000 | 11 | 66.9s (66.3-68.0s) | 13 | 1.55s (0.99-2.50s) |
| 32,000 | 13 | 170.8s (169.4-173.9s) | 11 | 2.10s (1.75-2.54s) |

("cold" / "warm" split at a 10s threshold — a clean bimodal separation, no
values fall near the boundary.) Synthesis calls showed no warm cluster at
any level (every call plants a distinct fact at the very front, breaking
whatever let grid "end"-placement calls go warm) — mean/median/range:

| level | n | mean | median | min | max |
|---:|---:|---:|---:|---:|---:|
| 4,000 | 6 | 14.05s | 13.53s | 13.37s | 16.38s |
| 8,000 | 6 | 17.08s | 17.11s | 16.85s | 17.23s |
| 16,000 | 6 | 48.91s | 40.73s | 39.66s | 66.51s |

**Working hypothesis for the split**: `llama-server` is running with
`--context-shift`, which can reuse a previous request's cached KV state
when a new request's prefix matches it. "end"-placement grid calls share
an (almost) identical long filler prefix across all 12 needles at a given
level — only the small trailing needle+question differs — so after the
first (cold) call establishes the cache, the remaining 11 hit it.
"start"-placement calls put the (per-needle differing) needle block at the
very front, breaking prefix match — consistent with L32000/start being
uniformly cold across all 12 calls.

**Closure on the two fast L16000/start calls** (`tax_rate` 2.5s, `db_pool`
1.1s, immediately followed by ten ~67s calls in the same placement,
originally flagged above as an unexplained exception to that hypothesis).
Re-checked against the raw per-call log: the same `(16000, start, tax_rate)`
cell has **two** entries — 64.8s and 2.5s — because the crash-recovery
JSONL, as already noted, retained one stale row from the first launch
attempt that was killed by the tool's 120s foreground timeout before that
attempt's own stdout ever flushed. That killed attempt had already
completed (or the server had already finished processing, independent of
the client disconnecting) real requests for `tax_rate` and `db_pool` at
this exact cell before dying. Critically, **both** the 64.8s and the 2.5s
`tax_rate` entries report the identical `prompt_tokens` (14,868) — not a
lower, clipped count — and `db_pool`'s fast 1.1s entry reports 14,863
tokens, matching its slow sibling `retry_backoff`'s 14,863 almost exactly.
Identical full token counts on both the fast and slow calls rule out
truncation as an explanation (a truncating server would report fewer
tokens on the fast calls, not the same count). The fast pair is a
**prefix-cache hit inherited from the killed first attempt's already-
processed requests for those exact two prompts**, not a data artifact and
not truncation — the same invalidation class (silent context clipping)
this whole spike exists to rule out, now explicitly checked and closed
rather than left open.

**#145 spot-check on real files** (measured directly, standalone cold
calls, not part of the pre-registered grid — a verification of specific
file sizes referenced below): `benchmarks/agentic_serving/subagent_adapter.py`
— 6,262 actual tokens, 24.7s. `.llm-orc/scripts/agentic_serving/classify.py`
— 19,452 actual tokens, 92.9s. Both single-shot, well inside the model's
40,960-token window, both cold (fresh standalone calls, no shared-prefix
warm-up).

### Synthesis extension: 20K / 24K two-fact, plus three-fact at 20K

Second, ~10-minute follow-up, decided by the #145 pre-flight: the first
synthesis smoke topped out at ~14,870 actual tokens, 31% short of #145's
flagship target (`classify.py`, 19,452 measured tokens). Script:
`docs/plans/2026-08-13-context-curve-synth-ext2-spike.py` — a second
sibling importing `SYNTHESIS_PAIRS`/`build_synthesis_input`/`both_score`/
`call_ollama`/`NUM_CTX` from the first extension script, not duplicated.
Pre-registered: the same 5% drift rule against the synthesis baseline
(100% at 4K); at this budget's n=6 (two-fact) one miss is already 16.7%
drift, and at n=3 (three-fact) one miss is 33.3% — noted as the accepted
coarse-resolution tradeoff for a 10-minute follow-up, not smoothed over.

| level (target tok) | facts required | n | accuracy | mean elapsed | min | max |
|---:|---:|---:|---:|---:|---:|---:|
| 20,000 | 2 | 6 | 100.0% | 90.60s | 88.91s | 93.24s |
| 24,000 | 2 | 6 | 100.0% | 117.52s | 115.06s | 121.20s |
| 20,000 | 3 | 3 | 100.0% | 94.24s | 92.55s | 97.49s |

18/18 correct, 0 errors. Every call in this batch was cold (each plants a
distinct fact at the very front, same reason the first synthesis smoke
never showed a warm cluster) — actual measured tokens ran ~18,660-18,712
at the 20K level and ~22,470-22,498 at the 24K level (chars/4 slightly
overestimates at this range, consistent with the original calibration
table). Three-fact accuracy matches two-fact accuracy at the same 20K
level — no sign that combining a third fact costs anything beyond the
extra ~100-150 tokens of prompt and answer length.

### #145 implication (extension, updated)

Synthesis now measured clean through 24,000 tokens (two-fact) and through
three-fact combination at 20,000 tokens — accuracy still never argues
against big chunks in-window (100% at every level tested, 500 through
32,000 for recall, 4,000 through 24,000 for synthesis, zero drift, zero
evidence synthesis degrades ahead of recall, zero evidence a third fact
costs anything). Concretely for #145's stated 96KB cap band (24,000 tokens
× 4 chars/token): **that band is now synthesis-covered, not just
recall-covered** — `classify.py` (19,452 measured tokens), #145's flagship
repo-scale-read target, sits inside the now-verified range with headroom
to the 24K ceiling actually tested (and further headroom to the 32K
recall-only ceiling and the model's 40,960-token architectural max, both
untested for synthesis). So #145's chunk-size ceiling remains what the
first extension found: not a correctness question in this range, a
**latency policy decision** — now with the cold-cost curve extended
further out. Cold cost by level: ~17s at 8K, ~67s at 16K, ~91s at 20K
(two-fact) / ~94s (three-fact), ~118s at 24K, ~171s at 32K (recall only,
unmeasured for synthesis past 24K); warm cost (a request whose prefix
matches what's already cached from the immediately preceding call) stays
~1-3s regardless of size, whenever the read pattern can land on it. The
design lever is still latency budget and read-pattern structure, not model
capacity — #145 can now plan chunk sizes up to and including a full
`classify.py`-sized single-file read, and a multi-fact synthesis question
over it, on the strength of a measured (not extrapolated) result.

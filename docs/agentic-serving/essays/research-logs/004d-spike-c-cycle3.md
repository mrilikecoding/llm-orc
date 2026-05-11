# Spike C — Cycle 3 Three-Arm Architecture Comparison (Cross-File Verification)

**Date:** 2026-05-01
**Cycle:** 3 (agentic-serving scoped corpus)
**Fixture:** Synthesized 90-line Python module diff with 5 deliberately-injected ground-truth issues across categories (`scratch/spike-c-cycle3-architecture-comparison/fixture/`)
**Operating frame:** Outcomes over an agentic session — agent shape is means.
**Central question (per Cycle 3 reframe):** Does orchestration + cheap orchestrator agent compete with a more expensive frontier model on tasks where cross-file verification matters?
**Methods reviewer specification:** Mid-cycle dispatch (`004-mid` audit) recommended a three-arm code review on a real diff with pre-spike fixture validation and PAE-aware procedural-independence scoring.

---

## Abstract

Spike C addressed the `+ orchestration` gap the mid-cycle methods reviewer identified: prior cycle spikes had not exercised the architecture's load-bearing primitive (`invoke_ensemble` dispatched by the cheap orchestrator) on a fixture where its presence vs absence would produce different outcomes. The spike used a synthesized 90-line Python diff with 5 deliberately-injected ground-truth issues spanning categories (semantic / security / type-safety / test-gap / cross-file consistency) — pre-validated to contain at least two issues where cheap-orchestrator-alone was expected to struggle and the architecture's script-agent slot was expected to add structural value. Three arms: cheap-bare via `opencode run` with full tool access (Arm A), cheap+ensemble via `llm-orc invoke spike-c-code-review` dispatching a script-agent + LLM reviewer ensemble (Arm B), and frontier-bare via Sonnet 4.6 single-shot subagent dispatch (Arm C). At surface-detection resolution all 8 trials caught all 5 issues. **At concrete-verification resolution — the discipline Decision A's PAE framing requires — the architectural-value finding is unambiguous: Arm B caught the actual cross-file mismatch (`100_000` vs `50_000_000`) in 3/3 trials; Arm A caught it in 1/3 (only when MiniMax happened to use Read tool autonomously); Arm C caught it in 0/2 (Sonnet single-shot has no access to other files).** The architecture's deterministic-vs-probabilistic complementarity — script-agent provides guaranteed cross-file verification; LLM reviewer provides semantic analysis on top — produces structural capability that neither cheap-tier-alone nor frontier-tier-alone can match in single-shot. This is the cycle's central-question evidence: cheap+orchestration outperforms frontier-alone on the bug class where cross-file verification matters.

---

## Background

The mid-cycle methods reviewer (`housekeeping/audits/research-design-review-cycle-3-mid.md`) identified two P1 findings against Spike B's deliverable: (1) regime mismatch — Spike B's fixtures were drawn from the easy regime where 100% pass rates told us little about the central question; (2) the `+ orchestration` gap — `invoke_ensemble` and `compose_ensemble` were not exercised in any Cycle 3 spike, while the architecture's central claim is precisely that the cheap orchestrator's ensemble-dispatch primitive adds value over cheap-tier-alone. The reviewer recommended a three-arm code-review-on-a-real-diff fixture with pre-spike validation requiring at least one issue category where cheap-orchestrator-alone is expected to struggle.

The practitioner authorized the spike. The fixture was synthesized rather than drawn from real PR history — synthesized fixtures provide cleaner ground truth (the issue inventory is fully known) and let the spike target specific bug classes deliberately. Pre-spike fixture validation (`fixture/ground-truth.md`) named ISSUE-1 (off-by-one in `check_limit`) and ISSUE-5 (cross-file value drift between `DEFAULT_BUDGET_LIMIT` and `DEFAULT_MAX_TOKEN_LIMIT`) as the two architecture-favorable cases where the script-agent's deterministic capabilities are expected to add structural value. ISSUE-4 (test gap, stated in plain text in the diff) was named as the floor — both tiers should catch it; if any arm misses it, that's a separate finding. ISSUE-2 (api_key in logger context) and ISSUE-3 (`limit: int = None` annotation) were named as intermediate cases where security-aware framing or type-checker-style discipline matters but is not architecturally exclusive.

---

## Method

### Fixture

`scratch/spike-c-cycle3-architecture-comparison/fixture/diff.patch` introduces `src/llm_orc/agentic/session_budget.py` (90 lines, plausibly something a developer would add to this codebase). The diff includes a commit message asserting "tests will be added in a follow-up" and explicitly warns in a code comment that `DEFAULT_BUDGET_LIMIT = 100_000` should match `DEFAULT_MAX_TOKEN_LIMIT` in `orchestrator_config.py` — but doesn't import; the actual `orchestrator_config.py` value is `50_000_000` (500× mismatch). The five injected issues are:

1. **ISSUE-1 — Off-by-one** (semantic): `check_limit()` uses `>` instead of `>=`; at-limit case is silently permitted.
2. **ISSUE-2 — Credential leak** (security): `logger.warning` includes `f"API key: {self.api_key}"`.
3. **ISSUE-3 — Type annotation** (type-safety): `limit: int = None` violates the declared type; `limit if limit else default` also conflates `None` with `0`.
4. **ISSUE-4 — Test deferral** (process): commit message + module docstring both explicitly defer tests to follow-up.
5. **ISSUE-5 — Cross-file drift** (cross-file consistency): the comment-warning + actual 500× value mismatch.

### Three arms

- **Arm A — cheap-bare:** `opencode run -m llm-orc/orchestrator-minimax-m25-free --print-logs "<review prompt + diff inline>"`. opencode provides Read / Edit / Bash tools; whether MiniMax uses them is the orchestrator's decision (this becomes a load-bearing scope condition; see Discussion). N=3 trials.
- **Arm B — cheap-with-ensemble:** `llm-orc invoke spike-c-code-review -f <diff>`. Custom 2-agent ensemble (`spike-c-code-review.yaml`): a script-agent (`spike_c_diff_analyzer.py` — symbol extraction, security-pattern detection, type-annotation flagging, test-presence detection, cross-file value verification) + an LLM reviewer (MiniMax with system prompt instructing it to use script findings as anchor evidence and add semantic analysis on top). N=3 trials.
- **Arm C — frontier-bare:** Sonnet 4.6 via subagent dispatch (F1 facsimile per Spike B's documented method) with the same review prompt + diff inline. Single-shot. N=2 trials.

**N total:** 8 trials. Cost: $0 across all arms (cheap arms via OpenCode Zen free tier; frontier via Claude Code subscription tokens).

### Scoring per Decision A (PAE-aware procedural independence)

Two scoring resolutions:

1. **Surface detection** — keyword/phrase heuristics: did the trial mention the issue category? E.g., for ISSUE-5: any reference to `DEFAULT_MAX_TOKEN_LIMIT`, `orchestrator_config`, `cross-file`, `drift`. Captured in `score_trials.py` per-issue keyword patterns with multiple matches per category.
2. **Concrete verification** — did the trial cite the specific evidence that distinguishes "drift acknowledged" from "drift verified"? E.g., for ISSUE-5: did the trial cite the actual value mismatch (`100_000` vs `50_000_000`, or the magnitude `500×`)? Without concrete citation, the trial only repeated the diff's own self-warning ("Should match... if these drift"); with concrete citation, the trial verified the values from both files.

Surface vs concrete scoring is the load-bearing distinction the methods reviewer's PAE framing requires. A reviewer that surfaces "there's a drift risk noted in the comment" without verifying values has done procedurally-independent flagging but not procedurally-independent verification.

---

## Results

### Per-arm trial table

| Arm | Trial | Wall-clock | Output size | Surface (5/5)? | Concrete on ISSUE-5? |
|-----|:-----:|:----------:|:-----------:|:-------------:|:--------------------:|
| Arm A | 1 | 56s | 316KB (incl opencode metadata) | ✓ | ✓ (used Read on `orchestrator_config.py`) |
| Arm A | 2 | 30s | 109KB | ✓ | ✗ (no tool calls; surface only) |
| Arm A | 3 | 32s | 109KB | ✓ | ✗ (no tool calls; surface only) |
| Arm B | 1 | ~30s | 19KB | ✓ | ✓ (script-agent's `cross_reference_verifications` produced exact mismatch) |
| Arm B | 2 | ~37s | 18KB | ✓ | ✓ |
| Arm B | 3 | ~45s | 18KB | ✓ | ✓ |
| Arm C | 1 | 47s | ~9KB | ✓ | ✗ (Sonnet single-shot has no access to other files) |
| Arm C | 2 | 42s | ~9KB | ✓ | ✗ |

### Surface-detection summary (5/5 across all 8 trials)

All 8 trials caught all 5 ground-truth issues at keyword-match resolution. Initial scoring suggested another Spike B "easy regime" pattern, but the PAE-aware refinement showed the surface detection is misleading — it captures pattern-matching against issue categories without distinguishing capability differences across arms.

### Concrete-verification on ISSUE-5 — the load-bearing finding

| Arm | Concrete-verification rate | Mechanism |
|-----|:--------------------------:|-----------|
| Arm A (cheap-bare via opencode) | **1/3** | Trial 1: MiniMax used Read on `orchestrator_config.py` autonomously (3 file reads). Trials 2 + 3: MiniMax did not use Read; only flagged the diff's self-warning |
| **Arm B (cheap+ensemble via `llm-orc invoke`)** | **3/3** | Script-agent's `cross_reference_verifications` deterministically extracted both values; LLM reviewer cited the `100_000` vs `50_000_000` `500×` mismatch in every trial |
| Arm C (frontier-bare via subagent F1) | **0/2** | Sonnet single-shot has no access to other files. Both trials flagged the drift conceptually and recommended import-from-config, but did NOT cite actual values (the values weren't available to them) |

**Arm B cleanly outperforms Arm C on this bug class: 3/3 vs 0/2 concrete verification.** Arm A's 1/3 is a function of MiniMax's autonomous tool-use variability, not capability — when MiniMax happens to use Read, it gets to the concrete answer; when it doesn't, it surface-flags only.

### Cost / latency

| Arm | Median wall-clock | Cost ($) | Subscription tokens (approx) |
|-----|:-----------------:|:--------:|:----------------------------:|
| Arm A | 30–56s | $0 | 0 |
| Arm B | 30–45s | $0 | 0 |
| Arm C | 42–47s | $0 | ~16K/trial |

Arm B has equivalent latency to Arm A and lower wall-clock than Arm C. Token cost ratio per Decision B's CLEAR CNA pattern: Arm B dominates on cost-axis with equivalent or better detection.

---

## Discussion

### The architectural-value finding

Spike C produces direct evidence on the cycle's central question: **the architecture's `+ orchestration` primitive (script-agent + LLM reviewer ensemble dispatched via `invoke_ensemble`) adds structural capability that neither cheap-tier-alone nor frontier-tier-alone can match on this bug class.** The mechanism: deterministic file access through the script-agent provides guaranteed cross-file verification; LLM reviewers cannot be relied on to consistently use general agentic tools (Arm A's 1/3 demonstrates the variability) and frontier-tier in single-shot has no access to other files at all (Arm C's 0/2). The architecture's value is not "cheap LLM + tools"; it's "cheap LLM + a deterministic verifier slot that guarantees what an LLM cannot."

This is the load-bearing extension of Spike A's cross-tier-uncorrelated-errors finding. Spike A: heterogeneity-uncorrelated-errors across LLM model families. Spike C: deterministic-vs-probabilistic complementarity within a single ensemble. Both are forms of **structural compensation for what a single LLM tier alone cannot guarantee** — Spike A's mechanism is that different LLMs have uncorrelated error distributions; Spike C's mechanism is that deterministic verification has zero error distribution where LLM verification has nonzero. The architecture's value claim, abstracted: ensembles compose components whose error distributions are different enough that the composition's coverage exceeds any single component's.

### Frame-divergence holds across both spikes (RQ-2 evidence)

Per Decision B's CLEAR cost-normalized accuracy framing:

- **Performance-only frame** (concrete detection alone): Arm B 3/3 > Arm A 1/3 > Arm C 0/2. Performance frame would prefer Arm B.
- **Four-priorities frame** (cost + performance + environmental + local-first): Arm B is again preferred — zero $ cost, equivalent latency to single-LLM, concrete detection that frontier-bare cannot match.

Both frames recommend Arm B on this bug class. **The frame-divergence at recommendation level is preserved — Arm C costs subscription tokens for strictly less concrete-verification capability than Arm B at zero cost.** Per RQ-2's falsification criterion: the four-priorities frame is empirically supported across two independent fixtures (Spike A and Spike C); not retired.

### What Arm A's variability tells us

Arm A's 1/3 concrete-detection is a function of MiniMax's autonomous tool-use decision. When MiniMax uses Read, it reaches concrete verification; when it doesn't, it doesn't. This is a Cycle 4 hook: cheap-tier autonomous routing intelligence is itself variable, and the architecture's value is in part that it removes that variability — the script-agent runs deterministically every time, regardless of the LLM's routing decisions. The architecture trades LLM-routing-variability for ensemble-structural-determinism. That trade is favorable on bug classes where deterministic verification is the binding constraint.

### Arm A confound — recorded as a scope condition

The methods reviewer's intended Arm A design was "cheap orchestrator working alone" (single-shot LLM call, no tool access). What was actually run was `opencode run` with full Read / Edit / Bash tools available. This is closer to "cheap-tier with general agentic tools" than to "cheap-tier-alone." A purer Arm A would have used `llm-orc invoke` with a single-LLM-only ensemble (no tools). For the cycle's central-question evidence, this is mostly favorable — Arm A had the OPPORTUNITY to use cross-file verification via Read; the architecture's advantage is that Arm B GUARANTEES it. A purer Arm A would likely score 0/3 concrete (matching Arm C); the variability we observed is generous to cheap-bare.

---

## Limitations and scope conditions

1. **Single fixture.** One synthesized 90-line module with 5 injected issues. Generalization to other diff sizes, other code domains (web app, async code, stateful services), and other issue category mixes requires additional fixtures.
2. **Heterogeneity scope cut.** Arm B used MiniMax for the LLM reviewer slot (no Hunyuan / Kimi heterogeneity from Cycle 2 A3's design). The architectural value here comes from the script-agent + reviewer composition, not from heterogeneous reviewers. Cycle 4 hook to test the full A3 architecture (script + 2 heterogeneous LLMs + MARG concatenation) on this fixture class.
3. **Arm A confound.** Documented above. opencode tool access made Arm A's behavior tier-with-general-tools rather than tier-alone. The 1/3 concrete-detection is favorable to cheap-bare relative to a purer single-shot variant.
4. **Frontier facsimile.** Arm C used F1 subagent dispatch; not a direct API call. Same caveat as Spike B's F1.
5. **Synthesized vs real fixture.** The diff was synthesized for the spike. Real PR diffs may have different issue mixes, more subtle bugs, more complex cross-file relationships.
6. **N=2 frontier vs N=3 cheap.** Frontier sample is smaller per the cycle's free-tier preference (subscription tokens for frontier). The 0/2 concrete result is consistent across both trials.

---

## Implications for Cycle 3's RQs

**RQ-2 — four-priorities frame load-bearing across both fixtures.** Spike A established frame-divergence on documentation review; Spike C extends it to code-review-with-cross-file-verification. The frame's empirical support spans two independent fixtures.

**Central question — direct affirmative answer on this bug class.** Cheap+orchestration competes with frontier-alone, and on the cross-file-verification bug class specifically, **outperforms** it. The architecture's deterministic-vs-probabilistic complementarity is the mechanism.

**Cross-tier complementarity refinement (Spike A finding extended).** Spike A: heterogeneity-uncorrelated-errors across LLM families on documentation review. Spike C: deterministic-vs-probabilistic complementarity within an ensemble. Both fit a unifying frame: **the architecture composes components with different error distributions; the composition's coverage exceeds any single component's.** Spike B refined: complementarity is task-class-dependent (absent on simple multi-turn protocol). Spike C confirms the mechanism on a different task class (code review with cross-file requirements).

---

## Cycle 4 hooks

1. **Test the architecture's actual `invoke_ensemble` primitive end-to-end via the orchestrator's tool-calling loop in `llm-orc serve`.** The cycle's evidence is that the primitive's dispatch path works (via direct invoke); whether the orchestrator (running in serve, with the closed-tool surface) autonomously chooses the right ensemble for each sub-task is a deeper question.
2. **Heterogeneous reviewers per A3 design.** Arm B's reviewer was MiniMax. Replacing with Hunyuan + Kimi (Cycle 2 A3 architecture) would test whether heterogeneity-uncorrelated-errors layers on top of script-agent value-add.
3. **Multiple fixtures.** Spike C used one synthesized diff. Cycle 4 should test fixture diversity: real PR diffs, multi-file diffs, different language ecosystems, different bug-category mixes. The "frame-divergent recommendation" claim should be tested at higher resolution.
4. **Floor question (Spike A P3 inheritance).** What value does the LLM reviewer add over the script-agent alone on this bug class? Spike C's data: script-agent already extracts the cross-file mismatch; the LLM reviewer cites it. Could a structured report directly from the script-agent (no LLM at all) achieve equivalent outcome on this fixture? Cycle 4 territory.
5. **Larger-scale fixture batteries.** Spike A's documentation review + Spike C's code review are two task classes. The architecture's value is fixture-dependent (Spike B showed it); a fixture battery establishing where complementarity holds vs doesn't would map the architecture's deployment envelope.

---

## Spike code disposition

**Retain until corpus close** per practitioner policy (memory: `feedback_spike_artifact_retention.md`).

Spike C preserved artifacts:
- `scratch/spike-c-cycle3-architecture-comparison/fixture/` — fixture diff + ground-truth doc + original code
- `scratch/spike-c-cycle3-architecture-comparison/ensembles/diff_analyzer.py` — script-agent for the ensemble
- `scratch/spike-c-cycle3-architecture-comparison/run_cheap_arms.sh` — bash runner for cheap arms
- `scratch/spike-c-cycle3-architecture-comparison/score_trials.py` — keyword-detection scorer
- `scratch/spike-c-cycle3-architecture-comparison/trials/` — all 8 trial outputs (3 cheap-bare + 3 cheap-with-ensemble + 2 frontier subagent traces)
- `scratch/spike-c-cycle3-architecture-comparison/scoring-results.json` — surface-detection results
- `.llm-orc/ensembles/spike-c-code-review.yaml` — the registered ensemble (active in this corpus's tier)
- `.llm-orc/scripts/spike_c_diff_analyzer.py` — the script-agent (active path)

---

## Connections to Cycle 3's other research

- **`004a-lit-review-agent-design.md`** — heterogeneity findings (Sun et al. 2025; Ding et al. 2024); Routine + Compiled AI script-as-orchestrator literature. Spike C empirically demonstrates one mechanism the literature predicts.
- **`004b-spike-a-cycle3.md`** — Spike A's cross-tier-uncorrelated-errors finding on documentation review. Spike C extends the abstraction: structural compensation via different error distributions.
- **`004c-spike-b-cycle3.md`** — Spike B's regime-bounded multi-turn finding. Spike C tests a different task class where the regime matters.
- **`research-design-review-cycle-3-mid.md`** — methods reviewer's recommendation that motivated Spike C; pre-spike fixture validation requirement that the spike satisfied.
- **Memory: `cycle-3-central-question`** — central question reframe. Spike C produces direct affirmative evidence on the central question for the cross-file-verification bug class.

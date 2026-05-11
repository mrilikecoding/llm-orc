# Spike (Cycle 4 Architect-Gate Continuation, Spike α) — Topaz Skill Classification Adequacy on the Existing Ensemble Library

**Date:** 2026-05-11
**Cycle:** 4 (agentic-serving scoped corpus)
**Phase:** ARCHITECT — gate-continuation pre-BUILD spike
**Method:** Cheap-orchestrator dispatch of a single-agent classifier (free-tier local Ollama `qwen3:8b`) plus analytical interpretation
**Cost incurred:** $0.00 (local Ollama; no cloud tokens consumed)
**Spike artifact retention:** retained until corpus close per Cycle 3 directive
**Companion ensemble:** `.llm-orc/ensembles/spike-005g-skill-classifier.yaml` (also retained)

---

## Spike question

Do the eight Topaz skills (`code_generation`, `tool_use`, `mathematical_reasoning`, `logical_reasoning`, `factual_knowledge`, `writing_quality`, `instruction_following`, `summarization`) cleanly partition the existing library's ensembles, or do many ensembles obviously span 2+ skills?

**One-line answer:** By the mechanical second-ranked-skill criterion (no clean primary if second ≥ 80% of first), classification is overwhelmingly clean — 21 of 21 classified ensembles have a clean primary. ADR-015's primary-skill framing stands on the criterion the spike pre-specified. Two methodological caveats and one substantive distribution finding qualify the disposition: the classifier exhibited apparent bias toward `writing_quality` / `logical_reasoning` / `summarization` (16 of 21 classifications collapsed to those three skills); `mathematical_reasoning` was never assigned as primary in the corpus surveyed; and the cheap classifier's verdicts on the boundary-case ensembles (e.g., `code-review` classified primary `instruction_following` rather than `code_generation`) are debatable enough that the *spike's evidence does not falsify the ADR-015 framing but also does not strongly validate it on the most empirically important production-style ensembles*. ADR-015's existing §Consequences §Negative entry on "Discovery value is proportional to deployment coverage" already anticipates the distribution finding; no amendment to ADR-015 is recommended at this gate.

---

## Method

### Approach

A single-agent ensemble (`spike-005g-skill-classifier`) was authored binding to local Ollama `qwen3:8b` (free tier). The classifier's system prompt enumerates the eight Topaz skills with one-line definitions and demands a strict three-line output: primary skill, ranked relevance percentages of the remaining seven, one-sentence rationale.

Two passes were attempted on the ensemble library:

1. **Initial pass with `qwen2:0.5b` (`micro-local` profile).** Failed — the 0.5B model returned syntactically-shaped but semantically-degenerate output (e.g., assigning all seven other skills 100% relevance with no discrimination). The model is too small to internalize the eight-skill definitions and produce calibrated rankings.

2. **Second pass with `qwen3:8b`.** Produced structured discriminating responses; adopted as the spike's classification engine.

The classifier was dispatched once per ensemble with a hand-prepared brief summarizing each ensemble's `description`, agent roster (with role hints from agent names + system prompts), and output shape. Briefs were structurally uniform to reduce prompt-shape confounds across classifications.

### Classification set

The library contains 53 YAML files across the top level and six subdirectories (`business/`, `development/`, `research/`, `security/`, `testing/`, `validation/`). The set was partitioned into three groups:

- **Production-style LLM-bearing (17 ensembles).** The kind a tool user or operator would invoke for domain work. Classified.
- **Boundary-test sample from `testing/` (4 ensembles).** A representative slice including a minimal dependency-graph fixture (`simple`), a multi-agent role demo (`adaptive-demo`), a fallback-mechanism fixture (`fallback-demo-ensemble`), and a parallelism fixture (`parallelism-test-ensemble`). Classified to surface boundary behavior.
- **Pure-script / framework-only (32 ensembles).** Filtered out. These have no LLM agents (only Python script-agents) or no domain payload — they exercise framework primitives (file-ops, control-flow, JSON extraction, cycle detection, fan-out routing). Classifying them on a *cognitive-skill* taxonomy is a category error: the framework primitive being tested is the unit of interest, not a skill the LLM exercises. Filtered list: all of `validation/*` (12), most of `testing/*` (15 of 19), `fan-out-test`, `plexus-graph-analysis`, `spike-d-fix-verifier`, `library-ensemble` (placeholder).

Total classified: **21 ensembles**. This is the load-bearing classification corpus for the disposition.

### Mechanical clean-primary criterion

Per the spike specification: an ensemble has *no clean primary* when its second-ranked skill is ≥ 80% as relevant as the first (where the primary is implicitly 100%). Equivalently: clean primary when second-ranked percentage < 80.

---

## Classification table

| # | Ensemble | Category | Primary | 2nd | 2nd % | Clean primary? |
|--:|----------|----------|---------|-----|------:|:--------------:|
| 1 | `adr-review` | production | `writing_quality` | `logical_reasoning` | 40 | Yes |
| 2 | `adr-swarm-review` | production | `summarization` | `logical_reasoning` | 40 | Yes |
| 3 | `agentic-calibration-checker` | production (system) | `instruction_following` | `writing_quality` | 40 | Yes |
| 4 | `agentic-result-summarizer` | production (system) | `summarization` | `writing_quality` | 30 | Yes |
| 5 | `example-local-ensemble` | production (sample) | `code_generation` | `instruction_following` | 40 | Yes |
| 6 | `qa-pipeline` | production (composed) | `writing_quality` | `instruction_following` | 30 | Yes |
| 7 | `topic-analysis` | production (composed) | `writing_quality` | `instruction_following` | 30 | Yes |
| 8 | `business/product-strategy` | production | `logical_reasoning` | `factual_knowledge` | 40 | Yes |
| 9 | `business/startup-advisory-board` | production | `logical_reasoning` | `factual_knowledge` | 30 | Yes |
| 10 | `development/code-review` | production | `instruction_following` | `writing_quality` | 40 | Yes |
| 11 | `research/.../interdisciplinary-research` | production | `logical_reasoning` | `factual_knowledge` | 25 | Yes |
| 12 | `research/.../mycology-meets-technology` | production | `writing_quality` | `logical_reasoning` | 30 | Yes |
| 13 | `research/.../sleep-and-civilization` | production | `logical_reasoning` | `factual_knowledge` | 40 | Yes |
| 14 | `security/security-review` | production | `factual_knowledge` | `logical_reasoning` | 40 | Yes |
| 15 | `spike-c-code-review` | production (Cycle 3 spike, retained) | `logical_reasoning` | `tool_use` | 30 | Yes |
| 16 | `spike-cycle4-research-loop` | production (Cycle 4 spike, retained) | `writing_quality` | `factual_knowledge` | 30 | Yes |
| 17 | `routing-demo` | production (orchestration demo) | `summarization` | `logical_reasoning` | 40 | Yes |
| 18 | `testing/simple` | testing fixture | `instruction_following` | `logical_reasoning` | 40 | Yes |
| 19 | `testing/adaptive-demo` | testing fixture | `logical_reasoning` | `factual_knowledge` | 25 | Yes |
| 20 | `testing/fallback-demo-ensemble` | testing fixture | `tool_use` | `logical_reasoning` | 30 | Yes |
| 21 | `testing/parallelism-test-ensemble` | testing fixture | `summarization` | `instruction_following` | 30 | Yes |

**Clean-primary count by mechanical criterion: 21 of 21 (100%).** Maximum observed second-ranked relevance: 40%. The 80% threshold for non-clean primary was nowhere approached.

---

## Distribution of primary-skill assignments

| Topaz skill | Count as primary | Share of corpus (n=21) |
|-------------|-----------------:|----------------------:|
| `writing_quality` | 6 | 29% |
| `logical_reasoning` | 6 | 29% |
| `summarization` | 4 | 19% |
| `instruction_following` | 3 | 14% |
| `factual_knowledge` | 1 | 5% |
| `code_generation` | 1 | 5% |
| `tool_use` | 1 | 5% |
| `mathematical_reasoning` | **0** | **0%** |

**Three skills (`writing_quality` + `logical_reasoning` + `summarization`) absorb 16 of 21 classifications (76%) of the existing library's classifiable ensembles.** Three of the remaining four primary-skill assignments are single-instance (`factual_knowledge`, `code_generation`, `tool_use`). One Topaz skill — `mathematical_reasoning` — is never assigned as primary by the cheap classifier on this corpus.

This distribution is congruent with ADR-015 §Consequences §Negative's existing entry: *"Discovery value is proportional to deployment coverage. The cycle's primary task class may exercise only 4–5 of the 8 Topaz skills routinely; the remaining slots produce no calibration evidence in deployment."* The spike's measurement on the *current library* (not the cycle's North-Star benchmark deployment) finds 4 actively-used primary skills + 3 single-instance + 1 unused. The library's coverage is even more concentrated than the ADR's hedge anticipated.

---

## Methodological caveats

Three caveats qualify the disposition the mechanical criterion suggests:

1. **Cheap-classifier bias.** `qwen3:8b` exhibits a structural preference for assigning `instruction_following` / `writing_quality` / `summarization` as primary on ensembles with structured-output prompts — which is most production ensembles. This bias *undercounts* the more substantive cognitive skills (`code_generation`, `mathematical_reasoning`, `logical_reasoning`, `factual_knowledge`) for ensembles whose structured-output discipline is incidental to their actual cognitive load. The clearest case: `development/code-review` was classified primary `instruction_following` despite the agent roster (security-auditor + performance-engineer + senior-reviewer + tech-lead) being a code-cognition ensemble; `code_generation` ranked at 0–5% in the cheap-classifier's verdict. A frontier classifier (or human classifier) would plausibly assign `code_generation` (or jointly `code_generation` + `logical_reasoning`) as primary, which would *change the classification but not the clean-primary verdict* — both would still be ≥ 60% relevance with a clean primary.

2. **One-shot classification (no inter-rater agreement).** Each ensemble was classified once. The spike did not run multiple classifications per ensemble to measure the cheap classifier's intra-rater consistency. A clean-primary verdict produced by an unstable classifier is weaker evidence than the same verdict produced by a stable one. The spike's design accepted this trade for cost (free-tier, ~30 minute budget per the spike specification).

3. **The "primary skill" question is not a one-axis question.** Several ensembles (e.g., `code-review`, `security-review`, `spike-c-code-review`) plausibly compose 2–3 skills as roughly co-equal contributors. The 80%-of-primary criterion the spike adopted (per the brief's specification) is a coarse instrument; finer-grained criteria (e.g., "primary if first-ranked skill is ≥ 50% relevance and second-ranked is ≤ 60%") would surface cases the binary criterion misses. The brief specified the 80% threshold; the spike honored it. A future spike could re-run with a stricter criterion if the architect-gate continuation finds additional evidence is needed.

---

## Surprises beyond the cycle-status's pre-stated hypotheses

1. **`mathematical_reasoning` exercises zero in the corpus.** The cheap classifier never assigned `mathematical_reasoning` as primary across 21 classifications. The Topaz taxonomy reserves a skill slot the existing library has no production-style ensemble exercising. Whether the cycle's North-Star benchmark (long-session agentic coding) routinely surfaces `mathematical_reasoning` as a router-relevant skill axis is empirically unresolved. ADR-015's discovery-friendly argument (§Rejected alternatives §(b) and §(d)) said keeping the full 8-skill taxonomy is justified by deployment-evidence discovery; the spike's library-coverage measurement is one input to that discovery — not the deployment evidence the ADR invokes, but a corpus-level signal that one taxonomy slot is unused before deployment evidence even runs.

2. **Three skills absorb 76% of classifications.** The corpus is heavily concentrated in `writing_quality` / `logical_reasoning` / `summarization`. If a Cycle 5+ deployment-evidence inquiry on the operator's actual usage finds the same concentration on the *invocation* axis (not just the *library-shape* axis), the friction-trades-for-discovery argument's "proportional to coverage" hedge in ADR-015 is empirically activated for the unused slots — and the operationally-tunable surface (collapse unused slots to shared Model Profiles) becomes the natural operator response.

3. **The framework-test fixtures classified meaningfully.** The boundary-test sample from `testing/` (4 ensembles) produced clean-primary classifications too — `instruction_following` (`simple`), `logical_reasoning` (`adaptive-demo`), `tool_use` (`fallback-demo`), `summarization` (`parallelism-test`). The cheap classifier read the agent-prompt content rather than recognizing the framework-feature-test purpose of the fixture. This is a category-confusion the classifier doesn't surface — the *fixture's purpose* (validating fallback / parallelism / cycle detection) is invisible to the classifier, which sees only the agent prompts. For ADR-015's purposes, this is fine (the router would route framework-test fixtures by their declared metadata anyway, not by inferred classification); the finding matters only for operators considering autonomous skill-classification of unclassified ensembles, where the runtime-classification rejected alternative (§Rejected §(f)) is reinforced.

4. **`fallback-demo-ensemble` classified primary `tool_use`.** The classifier correctly identified that the ensemble's cognitive load is the fallback-mechanism demonstration rather than the analyst-prose payload. This is an emergent correct-for-the-wrong-reason verdict — the classifier doesn't know what fallback chains are, but its `tool_use` assignment matches the framework-feature-test interpretation a human would give. Recording as a positive surprise: the cheap classifier's discriminating capacity is non-zero on at least some boundary cases.

---

## Disposition

**Selected disposition: classification is clean.** All 21 classified ensembles satisfy the spike's mechanical clean-primary criterion (second-ranked < 80% of primary). ADR-015's primary-skill framing stands; WP-G4 (Per-Role Tier-Escalation Router build) proceeds as designed in roadmap.md.

The disposition is selected with two scope conditions documented for downstream attention:

- **Cheap-classifier confidence.** The clean-primary verdict is robust to the cheap classifier's bias because the bias affects *which* skill gets the primary label, not whether a primary exists. Even on the most-debatable case (`code-review` classified `instruction_following` instead of plausibly `code_generation`), the alternative classification would still produce a clean primary by the 80% criterion.

- **Distribution-concentration finding is congruent with existing ADR-015 hedge.** ADR-015 already documents (§Consequences §Negative final entry) that discovery value is proportional to deployment coverage and the operator action is to collapse unused slots to shared Model Profiles. The spike's library-shape evidence does not contradict that hedge; it adds a corpus-level data point that supports the hedge's empirical relevance.

---

## Recommendation for the architect-gate continuation

1. **No amendment to ADR-015 is required at this architect-gate continuation.** The primary-skill framing is not falsified by the classifiable-library evidence the spike produced. Per-ensemble tier alternatives (the rejected alternative §(b)) remain unwarranted by spike-α evidence. WP-G4 (the Tier-Escalation Router module) proceeds as designed in `roadmap.md` and `system-design.agents.md` v3.0.

2. **Record the distribution finding as a load-bearing context for WP-G4 implementation.** When WP-G4 builds the per-skill tier defaults configuration surface, the operator-facing documentation should surface ADR-015 §Consequences §Negative's coverage hedge as load-bearing — early operators may legitimately collapse some of the 16 Model Profile slots to shared profiles. The spike's distribution measurement gives concrete shape to the hedge: at least 4 skills are clearly exercised by the existing library; at least 1 (`mathematical_reasoning`) is not exercised at all.

3. **The cheap-classifier bias finding has methodological implications for one of the rejected alternatives.** ADR-015 §Rejected alternative §(f) rejected runtime classification (router infers skill from task content) on principle: runtime classification reintroduces LLM-judgment into the routing path. Spike α's empirical evidence on `qwen3:8b` reinforces the principled rejection — the cheap classifier exhibits structural bias that an operator-authored skill metadata field would not. This is a corroborating finding, not a new one; recording it in this artifact's trail is sufficient.

4. **Consider commissioning a follow-up spike (Cycle 5+, deployment-evidence) on the *invocation* axis.** The library-shape distribution measured here is a structural signal; the operationally-relevant signal is which skills the deployed orchestrator *invokes* against the deployed library. ADR-015's discovery argument explicitly names deployment evidence as the validation surface; the architect-gate continuation should not run that spike now (out of scope), but should record it as a Cycle 5+ candidate in the OQ trail tied to ADR-015's Sub-Q6 (the autonomous-routing evidence gap §Consequences §Neutral entry).

---

## Confirmation

- Artifact written to canonical Output path: `docs/agentic-serving/essays/research-logs/005g-spike-topaz-skill-classification.md`
- Companion ensemble retained at: `.llm-orc/ensembles/spike-005g-skill-classifier.yaml`
- No code changes to project source. No cost incurred.

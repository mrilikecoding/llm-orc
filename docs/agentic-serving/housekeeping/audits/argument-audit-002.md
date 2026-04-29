# Argument Audit Report

**Audited document:** `docs/agentic-serving/essays/002-capability-floor-and-observability.md`
**Source material:** `docs/agentic-serving/essays/research-logs/research-log.md`, `docs/agentic-serving/essays/research-logs/lit-review-capability-floor-and-observability.md`
**Supporting audit:** `docs/agentic-serving/housekeeping/audits/citation-audit-002.md`
**Date:** 2026-04-25

---

## Section 1: Argument Audit

### Summary

- **Argument chains mapped:** 9
- **Issues found:** 8 (1 P1, 4 P2, 3 P3)

---

### Argument chains mapped

The essay advances nine inferential chains. Each is assessed against the research log and lit review.

**Chain 1 — Capability floor is family-specific, not parameter-count-driven.**
Support: qwen3:8b (biased) engages cascade (CAP-3b); mistral-nemo:12b (biased) does not (CAP-5); qwen3.5:9b (newer, larger) fails on continuation (CAP-7, CAP-7-rerun). External: Docker evaluation F1 scores. _Adequacy: partial — see P1-1 and P2-1._

**Chain 2 — Prompt steering is the cheapest sufficient intervention at qwen3:8b.**
Support: CAP-1 vs CAP-3 controlled comparison (same model, only system prompt differs). _Adequacy: partial — see P2-2._

**Chain 3 — ADR-011 holds at this capability tier.**
Support: CAP-3b end-to-end success without structural orchestrator change; CAP-2 demonstrates 2.54x latency cost for router-executor with equivalent outcome in simplified test. _Adequacy: partial — see P2-3._

**Chain 4 — Per-model tool-calling support is platform-specific; class-level checks are insufficient.**
Support: CAP-8 (deepseek-r1:8b 400 error despite `supports_tool_calling=True`). _Adequacy: strong — single clean empirical instance is sufficient for the narrow claim. Fix is bounded._

**Chain 5 — Confabulation is unmeasured in public benchmarks.**
Support: lit review Gap 1 synthesis (BFCL evaluates tool calls that occur, not claims that a call occurred); VILA-Lab paper names "Silent Failure and Observability-Evaluation Gap." _Adequacy: partial — see P2-4._

**Chain 6 — Failure-mode taxonomy (six modes).**
Support: each mode is observed in at least one spike. _Adequacy: strong for taxonomy as classification. No P-level issue._

**Chain 7 — Dual contracts converge under correct configuration.**
Support: CAP-3b satisfies both (user got useful response; orchestration hypothesis exercised). Diverge at latency. _Adequacy: partial — see P2-5._

**Chain 8 — Architecture and ADRs are all empirically validated.**
Support: CAP-3b exercise of Serving Layer through Result Summarizer Harness. _Adequacy: accurate but overstated — see P3-3._

**Chain 9 — What ships: three commits each have a traceable motivation.**
- Biased system prompt ← CAP-3 finding + CAP-5 confirming the boundary
- Dispatch-result logging ← DIAG-1 finding ("generic error types hide actionable diagnosis") + S0 meta-finding on log surface
- Typed-error promotion for Ollama 400 ← CAP-8 finding

_Adequacy: each commits' stated motivation maps to a recorded research finding. Chain is sound._

---

### P1 — Must Fix

#### P1-1 — "Family-specific" claim is n=2 families, one of which tested twice; the essay treats it as a general characterization

**Location:** §Findings, "The capability floor is family-specific, not parameter-count-driven" (para 1 and 2).

**Claim:** "Family-specific tool-use training quality dominates" (para 1). The essay presents this as a conclusion about how capability floors work in general for local orchestrators.

**Evidence gap:** The empirical base is two families: qwen3 (8b + 14b) and mistral-nemo (12b). The qwen3 family provides two data points on the same architecture; mistral-nemo provides one. The comparison confounds family with training epoch (qwen3 is a more recent model generation), architecture (qwen3 vs Mistral's attention architecture), and tool-calling training volume — all of which co-vary. The essay does not have a controlled comparison that isolates "family" as the explanatory variable from "training recency" or "tool-use training volume." The external Docker evaluation corroborates the qwen3 family advantage but does not include mistral-nemo:12b in its tabulation (citation audit confirmed this), so it cannot independently validate the specific qwen3-vs-mistral comparison.

The CAP-7-rerun finding (qwen3.5:9b underperforming qwen3:8b on the same task) is presented in §Open Questions as possibly a chat-template bug — which actually weakens the family-quality interpretation: if the non-monotonic gradient is attributable to a packaging bug rather than a capability property, it complicates the clean "family quality dominates" narrative.

**Recommendation:** Qualify the claim: "Across the two families tested, tool-use training quality appears to dominate parameter count, with qwen3:8b outperforming the larger mistral-nemo:12b." The current framing makes the generalization sound broader than the evidence (n=2 families) supports. This is a scoping correction, not a retraction — the finding is real and useful. It just needs a hedge that makes the sample size visible.

---

### P2 — Should Fix

#### P2-1 — qwen3.5:9b failure attributed to "chat-template bug" without verifying whether the bug was present in the tested version

**Location:** §Findings, "The capability floor is family-specific" (para 3); §Open Questions (para 1).

**Claim:** The essay presents the qwen3.5:9b premature-stop as "consistent with" the chat-template bug, with "strong" symptom-cause correspondence. This framing appears in the capability-floor section, which conditions how the reader interprets the non-monotonic gradient finding.

**Evidence gap:** The research log (CAP-7-rerun) itself lists three other plausible causes: different generation-parameter defaults (especially `presence_penalty=2.0`), thinking-mode-off behavior, or stop-token differences. The bug hypothesis is listed first and has the most text, but the log explicitly marks these interpretations as "not distinguishable from a single trial." The essay's phrasing ("the symptom-cause correspondence is strong") upgrades the bug hypothesis without acknowledging the alternative causes, which the log specifically records.

**Recommendation:** Surface the alternative causes alongside the bug hypothesis: "The premature-stop reproduces across two trials, consistent with a known chat-template bug (QwenLM/Qwen3 issue #1831), though alternative causes including generation-parameter defaults and thinking-mode differences cannot be excluded without further testing." This keeps the finding honest and is still useful — it names the most likely root cause while not over-claiming certainty.

---

#### P2-2 — "Prompt steering is sufficient" claim does not acknowledge the single-task-class scope

**Location:** §Findings, "Prompt steering is the cheapest sufficient intervention at this tier" (para 1).

**Claim:** "The CAP-1 to CAP-3 contrast is decisive." The essay presents this as the conclusion about prompt steering as an intervention.

**Hidden assumption:** Both CAP-1 and CAP-3 tested the same user request (ensembles + code-review task on README). The biased prompt was explicitly crafted to address the failure mode observed on this task — it contains trigger words matching the specific phrasing of the failing ask. The conclusion that prompt steering is "the cheapest sufficient intervention" is therefore valid for this task class but the essay does not mark this scope condition. A different task class (e.g., a non-capability-query that doesn't match the biased prompt's trigger words) might not benefit from the same intervention.

**Recommendation:** Add a scope qualifier: "The CAP-1 to CAP-3 contrast is decisive for the task class tested — requests that ask about available capabilities or invoke ensembles. Whether the biased prompt generalizes to other task classes is untested in this cycle." This does not weaken the practical conclusion (the biased prompt ships and addresses the documented failure mode), but it is an honest scope boundary.

---

#### P2-3 — "ADR-011 holds at this capability tier" — "this capability tier" is underspecified and the validation scope is narrower than the essay implies

**Location:** §Findings, "Prompt steering is the cheapest sufficient intervention" (para 2); §Implications for the Architecture (para 1).

**Claim:** "ADR-011's commitment to a single-Model-Profile orchestrator holds at this capability tier." The essay extends this to "empirically validated" in the Architecture section.

**Hidden assumption:** The empirical support for ADR-011 holding is CAP-3b (qwen3:8b with biased prompt on a single-turn, single-task-class ask). ADR-011's commitment is about the orchestrator-as-Model-Profile in general — across session types, task complexity, multi-turn depth, and session duration. The research log (S0 closure caveat 1) explicitly notes that 22 minutes for a single useful interaction suggests the user contract degrades at sustained use. The research log also records (CAP-2) that the router-executor pattern was tested in an isolated, low-noise environment that does not reproduce the full OpenCode conditions — so the CAP-2 comparison is not an apples-to-apples test of ADR-011's alternatives.

**Recommendation:** Qualify: "ADR-011's single-LLM-orchestrator commitment holds for the task class tested — single-turn, biased-prompt, qwen3:8b tier. Whether it holds for multi-turn sessions, deeper cascades, or different capability tiers is not validated by this cycle." The Architecture section's "empirically validated" language should be similarly qualified: the architecture is validated for the CAP-3b task class, not in general.

---

#### P2-4 — "Confabulation is unmeasured in public benchmarks" — the claim is not bounded against the possibility that confabulation IS measured under a different name

**Location:** §Findings, "Confabulation as a failure mode is unmeasured in public benchmarks" (para 1–2).

**Claim:** "The 'fast-confabulation' failure mode ... sits in the literature's blind spot." The essay presents this as an original contribution where the cycle's work is ahead of published patterns.

**Hidden assumption:** The essay's reasoning is: BFCL evaluates tool calls that are emitted, therefore it does not evaluate claims absent dispatch; ToolBench, ToolEval, and tau-bench similarly. But the essay does not address whether hallucination benchmarks, factual grounding evaluations, or calibration-focused evaluations might measure the same phenomenon under different names. The lit review's self-knowledge / calibration section (Gap 1, TMLR 2025 honesty survey) is directly adjacent — it addresses overconfident LLM outputs — but the essay does not engage with whether "confident prose claiming a tool call occurred" might be measured as a factual hallucination rather than as a tool-calling failure. The gap in the tool-calling benchmarks is real; the gap in the broader evaluation literature is less certain.

**Recommendation:** Bound the claim to the tool-calling benchmark domain: "The 'fast-confabulation' failure mode sits in the tool-calling benchmark literature's blind spot. Whether equivalent phenomena are captured by broader factual hallucination or calibration benchmarks is outside this cycle's scope." This is more precise and avoids over-claiming originality that might not survive scrutiny against the hallucination literature.

---

#### P2-5 — Dual-contract convergence claim understates the conditions required for convergence

**Location:** §Findings, "The dual contracts converge under correct configuration" (para 1); §Conclusion (para 2).

**Claim:** "Under correct configuration ... the contracts converge. The seam was navigated through prompt design."

**Hidden assumption:** "Correct configuration" in the essay means: (1) qwen3:8b orchestrator, (2) biased system prompt, (3) `qwen3:0.6b` summarizer model pulled and available. The third condition is load-bearing — CAP-3b's success required DIAG-1's fix, which required pulling an additional model. The essay records this in §The Starting State ("the summarizer model dependency satisfied") and §What Ships but the convergence claim in §Findings does not name all three conditions, implying convergence is achievable through prompt design alone.

**Recommendation:** Make all three conditions explicit where the convergence claim is made: "The contracts converge when the orchestrator is qwen3:8b with the biased system prompt, and the summarizer model dependency (qwen3:0.6b) is available locally. All three conditions are required; the CAP-3 result (before DIAG-1's fix) shows that the first two alone do not produce convergence." This is already implied by the essay's structure but should be explicit at the claim site.

---

### P3 — Consider

#### P3-1 — The "decisive" framing of the CAP-1/CAP-3 contrast could be softened without weakening the finding

**Location:** §Findings, "Prompt steering is the cheapest sufficient intervention" (para 1): "The CAP-1 to CAP-3 contrast is decisive."

**Issue:** "Decisive" implies the question is settled. But CAP-3 resulted in four summarization failures and a graceful text fallback — not a complete success. The same configuration did not succeed until CAP-3b (after DIAG-1's fix). "Decisive" is defensible for the narrow question (does the biased prompt change the failure mode?), but it slightly oversells the CAP-3 result as a user-contract success when the actual user-contract success was CAP-3b.

**Recommendation:** "The CAP-1 to CAP-3 contrast is decisive on the question of internal-cascade engagement: the same model, on the same hardware, against the same request, engages the cascade with the biased prompt and does not without it." This preserves the strength of the finding while being precise about what it shows.

---

#### P3-2 — The 2×2 summary matrix (implicit in the essay) could be made explicit

**Location:** §The Capability Gradient Observed; §Findings.

**Issue:** The research log contains an explicit 2×2 summary table (qwen3:8b × biased/default; mistral-nemo × biased/default) at the end of CAP-5. The essay presents these four data points narratively across two sections without showing the 2×2 structure. The structure is load-bearing for the "family-specific" finding — it is what distinguishes "prompt helps qwen, prompt cannot help mistral" from "prompt helps in general."

**Recommendation:** Consider a brief tabular presentation or explicit "(model × prompt)" framing somewhere in §Findings to make the inferential structure visible. This is a clarity improvement, not a correction.

---

#### P3-3 — Architecture validation claim in §Implications should name which modules were NOT exercised

**Location:** §Implications for the Architecture (para 1): "The four-layer architecture ... the calibration gating (ADR-007) ... and the orchestrator-as-Model-Profile commitment (ADR-011) are all empirically validated."

**Issue:** The research log's S0 module exercise table explicitly records that Composition Validator and Calibration Gate were NOT exercised in S0 (because no `compose_ensemble` calls were made, and the invoked ensemble was a known library ensemble). The essay's "empirically validated" claim is accurate for the modules that were exercised but slightly overreaches for ADR-007 (Calibration Gate) and the Composition Validator, which were structurally present but never entered.

**Recommendation:** "The four-layer architecture, the closed five-tool surface, the mandatory summarization path, the orchestrator-as-Model-Profile commitment, and the Result Summarizer Harness are all empirically exercised by CAP-3b. ADR-007 (calibration gating for composed ensembles) and the Composition Validator were present but not exercised — no `compose_ensemble` calls were made in this cycle's task class." This is a truthfulness improvement, not a refutation of the architecture's design.

---

## Section 2: Framing Audit

The framing audit makes the negative space of content selection visible. The primary document chose a framing — this section examines what that choice excluded.

### Question 1: What alternative framings did the evidence support?

**Alternative A — "Latency is the primary constraint, not capability."**

The research log records 22 minutes wall-clock for qwen3:14b's successful cascade (S0), and 5m 53s for qwen3:8b's partial success (CAP-3). The log explicitly concludes "product-unusable for interactive coding workflows" and frames this as the binding constraint for hardware-realistic deployment. The lit review independently notes BFCL measures correctness, not latency, and the cycle's latency data is original signal.

Under this framing, the essay's central question would shift from "what does it take to clear the capability floor" to "what does it take to clear the capability floor at workable latency." The answer is different: qwen3:8b clears the capability floor but does not clear the latency floor for interactive coding. The essay's current framing treats latency as a coda (§Conclusion: "latencies that are workable for batch but unworkable for interactive coding") rather than as a co-equal finding.

What would a reader need to believe for this framing to be correct? That user-contract success requires both correctness and latency tolerability — which is exactly what Beck's "honest about what it can deliver" framing implies, and what the essay's own dual-contract analysis acknowledges at the latency-divergence point. The evidence is there; the framing just de-emphasizes it.

**Alternative B — "The summarizer dependency is an unreported configuration hazard."**

The research log documents that CAP-3 (four summarization failures) and S0 (summarization_failed handled by graceful fallback) were both caused by a missing `qwen3:0.6b` model that the project config assumed was available. This was only discovered via DIAG-1. The essay mentions "the summarizer model dependency satisfied" as a precondition for success, but DIAG-1's finding — that default config silently assumes model availability and fails non-transparently — is framed as an observability finding rather than as a configuration-safety finding.

Under this framing, the essay's central finding would include a third production-problem: the project's `.llm-orc/config.yaml` ships with a model dependency that new deployments will silently break on. The essay ships a fix for the system prompt and for dispatch logging, but not for the model-availability assumption. A reader under this framing would ask: why does the default config not validate model availability at startup?

What would a reader need to believe for this framing to be correct? That silent dependency failure is at least as consequential as the prompt-steering gap for first-session experience. The DIAG-1 finding (all of CAP-3b's success depended on pulling one 520MB model) supports this. The essay treats DIAG-1 as a supporting finding rather than a first-class finding.

**Alternative C — "OpenCode as the sole client is a scope condition that should be named explicitly."**

Every spike used OpenCode as the client. OpenCode's behavior (11 declared client tools, specific system prompt injection, SSE handling, retry semantics) is co-determinant with the orchestrator's behavior. The CAP-1 silent-giveup failure was caused partly by OpenCode's 11 client tools competing with llm-orc's 5 internal tools. The success of the biased prompt is partly about overcoming OpenCode's framing, not just about model capability.

Under this framing, the essay's findings are "capability floor and observability as observed through OpenCode" — a narrower but more honest characterization. Cline or Roo Code, which might declare different client tools or inject different system prompts, could produce different capability floors and different failure modes.

What would a reader need to believe for this framing to be correct? That client-side tool declaration is a material co-determinant of orchestrator failure modes — which the research log explicitly confirms in S0-CAP-1's "tool-surface confusion is content-driven" finding.

---

### Question 2: What truths were available but not featured?

**Finding 1: qwen3:8b may be MORE accurate than qwen3:14b on path-grounding tasks.**

**Where it appears in the source material:** S0-CAP-1 (qwen3:8b first ask), research log table: qwen3:8b named 12 paths, 11 real (92% accuracy); qwen3:14b named 7 paths, 5 real (71% accuracy). The log notes: "capability is multi-dimensional, and 'smaller model' does not monotonically degrade across all dimensions" and raises as a possible explanation that the smaller model's tighter attention stays closer to the tool result rather than filling in priors.

**Why absent from the essay:** The essay focuses on cascade engagement (which qwen3:14b handles and qwen3:8b doesn't in the unbiased condition). The path-accuracy comparison on a different task (project structure description) uses the client-tool path (glob), not the internal-cascade path, so it's somewhat off-narrative for the "capability floor for cascade engagement" framing.

**Would its inclusion change or complicate the argument?** Yes, partially. The "capability floor is family-specific" finding is framed around qwen3:8b being above the floor. The non-monotonic path-accuracy finding adds nuance: within the qwen3 family, the smaller model is not strictly worse; it may be better-calibrated for certain task classes. This complicates the simple "larger models are more capable" interpretation that readers might import from the family-specific finding.

---

**Finding 2: The biased system prompt was crafted specifically to address one failure mode in one task class — its transferability is untested.**

**Where it appears in the source material:** S0-CAP-3 entry, S0-CAP-2 entry (the CAP-2 isolated test used different framing than the biased OpenCode prompt). The CAP-3 biased prompt contains trigger words specifically matched to the failing ask's language ("ensembles", "available", "what does this system do", "llm-orc").

**Why absent from the essay:** The essay's What Ships section describes the biased prompt as a production change without qualifying its scope. The prompt is described as tested and effective — which is accurate — but its narrow tailoring to the tested task class is not disclosed.

**Would its inclusion change or complicate the argument?** It would complicate the "cheapest sufficient intervention" claim. If the biased prompt is task-class-specific rather than general, then "shipping the biased default" solves the tested failure mode but may not address novel failure modes in different task classes. This does not invalidate the production change, but it should inform the reader's expectation of coverage.

---

**Finding 3: Argument fabrication in mid-cascade tool calls — qwen3:8b invented non-existent `invoke_ensemble` arguments.**

**Where it appears in the source material:** DIAG-1 entry: "Turn 6 emitted `arg_keys=['input', 'name', 'skip_summarizer']`; turn 7 emitted `['force_output', 'input', 'name', 'skip_summarizer']`. These are NOT real `invoke_ensemble` arguments." The log explicitly flags this as an RQ-3 finding: "orchestrator argument fabrication is invisible to the architecture today."

**Why absent from the essay:** The essay's confabulation finding is framed around the mistral-nemo:12b failure mode (text claiming a tool call that never happened). The qwen3:8b argument fabrication is a different, milder form of confabulation — the tool call IS made, but with invented arguments — that occurs in the "working" model. The essay presents qwen3:8b-with-biased-prompt as the validated configuration; the argument fabrication finding cuts slightly against the "cascade works" conclusion.

**Would its inclusion change or complicate the argument?** Yes, meaningfully. The essay's failure-mode taxonomy (§Findings, "A failure-mode taxonomy emerged") lists six modes but does not include "argument fabrication" — inventing unsupported arguments in otherwise-legitimate tool calls. This is a seventh failure mode observed in the cycle's empirical data. Its omission is not a logical error (the six modes the essay does list are accurate), but it understates the range of failure modes observed in the empirical work, and specifically understates failure in the validated qwen3:8b configuration.

---

**Finding 4: The Canonical / local-first AI ecosystem context — unreported in the essay.**

**Where it appears in the source material:** A substantive lit-scan partial in the research log covers Canonical's inference snaps strategy, its vocabulary alignment with this cycle's observability framing ("auditability of decisions and outcomes"), and its positioning relative to llm-orc's value proposition.

**Why absent from the essay:** The essay does not engage with the broader local-first AI ecosystem beyond the specific benchmark literature. The Canonical material was explicitly flagged as "not load-bearing" for the essay's immediate findings — which is correct. It is context, not evidence.

**Would its inclusion change or complicate the argument?** It would enrich the essay's discussion of the observability framing by grounding it in an externally-validated vocabulary ("auditability of decisions and outcomes" as a first-class agentic primitive). Its absence does not change the essay's conclusions; its inclusion would strengthen the framing's external validity.

---

### Question 3: What would change if the dominant framing were inverted?

The essay's dominant framing is: **"the capability floor is the primary obstacle; clearing it through correct model selection and prompt configuration makes the architecture viable."**

Inverted: **"the architecture is viable only under highly specific conditions; the capability floor is a symptom of an architectural assumption that the orchestrator role should be a single generalist LLM."**

Under the inverted framing:

**Claims that become weaker:**
- "ADR-011 holds" becomes "ADR-011 holds only when the model family, system prompt, task class, summarizer dependency, and client tool pool are all configured correctly — a configuration space too narrow to call robustly validated."
- "Three production changes ship" becomes "three production changes partially close gaps; the structural gap (orchestrator must be a generalist LLM capable of all tool-selection decisions) remains open."

**Claims that become stronger:**
- The research log's pre-S0 finding that "the orchestrator is the only llm-orc role that cannot be composed" becomes the central structural critique. The finding was recorded, surfaced, and then resolved by CAP-3b's empirical success — but under the inverted framing, CAP-3b proves a very specific narrow case, not the general viability of the single-LLM approach.
- The non-monotonic gradient (qwen3:8b sometimes better than qwen3:14b; qwen3.5:9b worse than qwen3:8b) becomes evidence that the single-LLM orchestrator approach is fundamentally brittle — dependent on model selection that cannot be reliably predicted from published benchmarks.

**What the essay would need to address if it took the inverted framing seriously:**
- The research log explicitly surfaces this alternative (the CAP-2 / ADR-011-reopening consideration) and resolves it through empirical evidence that cheaper interventions work. The essay's framing resolution is defensible, but it would need to explicitly acknowledge that the cheaper-interventions approach validates a narrow task class, not the architecture in general.

---

### Framing Issues

#### P1 — Consequential omissions

**FI-1 — Argument fabrication in the validated qwen3:8b configuration is absent from the failure-mode taxonomy**

**Location:** §Findings, "A failure-mode taxonomy emerged"; §Implications for the Architecture; §Conclusion.

**Evidence gap:** DIAG-1 records that qwen3:8b (the validated, recommended orchestrator profile) fabricated non-existent `invoke_ensemble` arguments (`skip_summarizer`, `force_output`) when facing repeated summarization failures. This is a failure mode distinct from the six taxonomized in the essay, and it occurred in the configuration the essay endorses. A reader who acts on the essay's recommendation will encounter this behavior; the essay does not prepare them for it.

**Why P1:** The essay explicitly names "fast-confabulation" (mistral-nemo's text-without-dispatch) as the most insidious failure mode "because it appears authoritative on superficial inspection." Argument fabrication in a tool call that IS dispatched is structurally the same risk — the model invents parameters that the architecture silently drops, the operator sees no error, and the orchestrator interprets the resulting failure as something to retry. This is not a minor clarification; it is a failure mode the essay warns readers about in principle but omits in the specific configuration it ships.

**Recommendation:** Surface to the user for judgment. The DIAG-1 finding on argument fabrication should either be added to the failure-mode taxonomy (as a sub-type of confabulation that occurs within otherwise-legitimate tool calls) or explicitly noted as an observed behavior in the configuration that ships. This is a direct implication of the essay's own confabulation-as-insidious framing. Correcting this requires adding one finding, not rewriting the conclusion.

---

#### P2 — Underrepresented alternatives

**FI-2 — Single-trial depth for most spikes is not acknowledged in the essay**

**Location:** §The Capability Gradient Observed; §Findings throughout.

**Issue:** Most spikes are n=1 or n=2. The essay describes results in confident language ("drove the cascade reliably," "reproduced across two trials," "did not engage the cascade") without foregrounding the single-trial nature of most observations. CAP-3b (the success case) is n=1. CAP-5 (mistral-nemo confabulation) is cited as "two trials" but the first trial is from PLAY (a different tooling context) — making the controlled comparison effectively n=1 (CAP-5 itself). Only CAP-7 has an explicit rerun recorded.

**Recommendation for user judgment:** The essay should include a sentence somewhere in §Method acknowledging that most spikes are single observations. Something like: "Most spikes were run once; CAP-7 was repeated to confirm reproducibility of the premature-stop. Where a finding is described as 'observed' rather than 'confirmed,' it reflects a single trial." This does not weaken the findings — single-trial observations are valid for exploratory characterization — but it sets appropriate reader expectations.

---

**FI-3 — OpenCode as the sole client is not named as a scope condition anywhere in the essay**

**Location:** §Method; §Findings throughout.

**Issue:** The essay names OpenCode prominently in §Method and §The Starting State, but never explicitly states that the findings are scoped to the OpenCode-as-client configuration. A reader deploying llm-orc with Cline, Roo Code, or any other client that declares a different tool pool would not know to expect different capability floor behavior. The 11-client-tool competition that caused CAP-1's silent giveup is not a property of llm-orc — it is a property of the OpenCode × llm-orc composition.

**Recommendation for user judgment:** Add a scope sentence in §Method: "All spikes used OpenCode as the client; behavior under other clients (Cline, Roo Code, or minimal API consumers) is not tested in this cycle and may differ, particularly for failure modes driven by client-tool-pool composition." The client-tool-pool competition finding (CAP-1's root cause) is directly tied to OpenCode's specific behavior.

---

**FI-4 — The latency finding is framed as a coda rather than a co-equal result**

**Location:** §Findings, "The dual contracts converge under correct configuration" (para 2); §Conclusion (para 1).

**Issue:** The essay's conclusion characterizes the latency boundary as one of the "remaining gaps" alongside multi-turn workflows and confabulation detection. But the research log treats latency as a binding constraint that applies to the validated successful configuration (CAP-3b: ~6 minutes for the code-review task, 22 minutes for S0). The Beck framing the essay invokes — "honest about what it can deliver" — applies directly to latency: a system that produces the right answer in 22 minutes is not delivering a user-contract-satisfying experience for interactive coding workflows.

The essay does note "workable for batch but unworkable for interactive coding workflows" — this is accurate. But it appears in the conclusion rather than as a first-class finding alongside the capability floor finding.

**Recommendation for user judgment:** Consider whether the latency finding warrants its own finding in §Findings rather than appearing only in the conclusion. The evidence for it is as strong as the evidence for any other finding (repeated observations across S0, CAP-3b, with specific numbers). Promoting it to a named finding would give it proportional weight relative to the capability-floor finding.

---

#### P3 — Minor framing choices

**FI-5 — CAP-7/CAP-8 negative findings are not given proportional weight in the closing summary**

**Location:** §Conclusion (para 1).

**Issue:** The conclusion foregrounds the positive finding ("A qwen3:8b orchestrator with a biased system prompt and the summarizer model dependency satisfied delivers useful agentic responses end-to-end") before listing remaining gaps. CAP-7 (qwen3.5:9b fails) and CAP-8 (deepseek-r1:8b fails entirely) are not mentioned in the conclusion at all — they are subsumed into "the remaining gaps." This is not inaccurate, but it slightly over-emphasizes the successful path relative to the discovery that two candidate models do not work.

**Recommendation:** The conclusion's "on consumer hardware, at latencies that are workable for batch" already acknowledges the workability boundary. A brief acknowledgment that the model selection is narrower than expected — "qwen3:8b in the qwen3 family, since qwen3.5:9b fails on continuation and deepseek-r1:8b has no tool-calling support via Ollama" — would give the negative findings their proportional weight. Minor framing adjustment.

---

**FI-6 — "Workable for batch" is asserted without defining "batch"**

**Location:** §Conclusion (para 1): "at latencies that are workable for batch but unworkable for interactive coding workflows."

**Issue:** "Batch" is undefined. Six minutes per single-ask interaction may or may not be "batch" depending on the workflow. A user running llm-orc as a pre-commit hook, an overnight analysis, or a CI step would have different latency tolerances. The claim is directionally correct but the positive end of the latency claim ("workable for batch") is asserted without evidence.

**Recommendation:** Either define what "batch" means in this context or hedge: "at latencies that may be acceptable for non-interactive or asynchronous workflows" rather than the assertive "workable for batch."

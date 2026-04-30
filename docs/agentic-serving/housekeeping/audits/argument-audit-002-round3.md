# Argument Audit Report — Round 3

**Audited document:** `docs/agentic-serving/essays/002-capability-floor-and-observability.md`
**Source material:** `docs/agentic-serving/essays/research-logs/research-log.md`, `docs/agentic-serving/essays/research-logs/lit-review-capability-floor-and-observability.md`
**Prior audits consulted:** `docs/agentic-serving/housekeeping/audits/argument-audit-002.md`, `docs/agentic-serving/housekeeping/audits/argument-audit-002-round2.md`, `docs/agentic-serving/housekeeping/audits/citation-audit-002.md`
**Date:** 2026-04-25

---

## Round 3 Scope

This is a re-audit after the practitioner ran S0-CAP-9 (hybrid cloud orchestrator + local ensembles) and incorporated its findings into the essay. The round-2 audit left the essay with 0 P1, 1 P2 (P2-R1 — meta-framing assertion in the latency finding), and 2 P3 issues. Round 2 also confirmed the essay was clean to proceed to the gate subject to those minor fixes. This report re-evaluates whether those fixes landed, whether the round-2 issues survive the revision, and specifically evaluates the five argument questions and five framing questions posed in the dispatch.

The CAP-9 spike introduced material changes to: the abstract, the §Method spike count, the latency finding, the dual-contracts finding, the §What Ships section (a fourth change added), and the §Conclusion. This audit focuses tightly on those revised sections while confirming stability of the unchanged sections.

---

## Section 1: Argument Audit

### Summary

- **Argument chains mapped:** 10 (9 original + 1 new: hybrid deployment shape validates the project hypothesis with lower latency)
- **Issues found:** 4 (0 P1, 2 P2, 2 P3)

---

### Verification of Round-2 Remaining Issues

#### P2-R1 fix (round 2) — Meta-framing assertion in latency finding

**Location:** §Findings, "Latency is the binding constraint at consumer hardware" (final sentence of para 2).

**Assessment:** The revision removed the sentence that contained the meta-framing assertion. In the revised essay the latency finding section now scopes itself to "for local-only deployment" in its heading and opening, and the final sentence of the local-only paragraph is: "A user who waits six minutes for a response receives a degraded experience even if the eventual output is correct." This is an empirical claim, not a meta-commentary on the essay's structure. P2-R1 is closed.

---

#### P3-R1 fix (round 2) — Argument-confabulation stress-conditionality unmarked

**Location:** §Findings, "A failure-mode taxonomy emerged" (argument-confabulation entry).

**Assessment:** The revised essay reads: "a behavior observed during CAP-3 under summarization-retry stress, within the validated qwen3:8b configuration: when `invoke_ensemble` returns the same `summarization_failed` error across multiple retries, the model emits structurally-correct subsequent calls with fabricated argument keys..." The phrase "under summarization-retry stress" is now present and marks the conditions. P3-R1 is closed.

---

#### FI-R2 fix (round 2) — Mild logical tension between taxonomy entry and convergence claim

**Location:** §Findings, argument-confabulation entry vs. dual-contracts section.

**Assessment:** The argument-confabulation entry now ends: "See the convergence-conditions finding that follows for how the validated configuration manages such failure modes when they appear." The dual-contracts section now reads "The dual contracts converge under either of two validated deployment shapes" with explicit conditions listed. The cross-reference is present. FI-R2 is closed.

---

### Round-3 Specific Issues — Argument Audit

The dispatch posed five specific argument questions. Each is assessed here.

---

#### Argument Question 1 — Does the hybrid finding's n=1 count get appropriate hedging?

**Location:** §Findings, "Latency is the binding constraint for local-only deployment, but the architecture supports a hybrid alternative that closes the gap" (para 3–4).

**Claim assessed:** "CAP-9 demonstrated that a different deployment shape closes this gap... approximately sixty-two seconds wall-clock — about a six-fold improvement over qwen3:8b's six-minute pure-local time."

**Finding:** The hedging is partial but not proportionate to the evidence base. The essay's §Method does acknowledge small trial counts in general ("CAP-7 was n=2; every other spike was n=1"), which technically covers CAP-9 as n=1 by default. However, the hybrid finding receives stronger narrative treatment than any individual local-only finding. The phrases "demonstrated that," "closes this gap," and "output quality matched CAP-3b" are all stated without a qualifier at the point of claim, despite the hybrid result being a single trial.

The essay acknowledges the n=1 caveat only once globally in §Method; the hybrid sections in §Findings and §Conclusion use language that projects confidence proportionate to a reproducible finding. Compare: the qwen3.5:9b failure is explicitly described as reproducible ("the pattern reproduced across two trials") — establishing that the essay does distinguish single-trial from confirmed findings when it wants to. The hybrid success does not receive the same epistemic annotation.

This is a real gap. CAP-9 is n=1 against a cloud provider that may vary in availability, rate limiting, free-tier terms, or model behavior. A single trial establishes feasibility, not stability.

**Recommendation (P2):** Add a hedging qualifier at the hybrid finding's claim site: "In a single trial, CAP-9 demonstrated..." or "The CAP-9 trial showed that..." The §Conclusion paragraph repeating the ~6× claim should similarly be softened: "a single hybrid trial produced approximately one minute end-to-end" rather than the declarative "delivers the same response quality at approximately one minute." This is a proportionality correction, not a retraction.

---

#### Argument Question 2 — Does the "approximately 6×" speedup claim survive scrutiny?

**Location:** §Findings (para 4) and §Conclusion (para 1).

**Claim assessed:** "about a six-fold improvement over qwen3:8b's six-minute pure-local time" (§Findings); "approximately one minute end-to-end" (§Conclusion).

**Finding:** The arithmetic is sound. CAP-9 research log records 62 seconds wall-clock for the hybrid trial. CAP-3b research log records the qwen3:8b local-only as a ~6-minute total. 360/62 = 5.8×, rounded to ~6× — a fair rounding to one significant figure. The essay uses "approximately" which is appropriate.

The comparison is slightly complicated by the fact that the essay compares CAP-9 to "qwen3:8b's six-minute pure-local time" as the reference point, and CAP-3b (the reference) was itself the successful case after DIAG-1 fixed the summarizer dependency. The earlier CAP-3 took ~5m 53s and did not succeed end-to-end. The essay's comparison is made against CAP-3b, which is the correct baseline (the successful local-only trial), not against CAP-3's failed trials. This is accurate.

One minor issue: the essay's §Findings para 2 characterizes the local-only latency as "approximately six minutes for qwen3:8b at the CAP-3b trial" and the hybrid as "approximately sixty-two seconds." The §Conclusion says "approximately one minute" for the hybrid. These are consistent. No accuracy issue.

**Assessment:** The 6× claim survives scrutiny. No issue at P2 or above.

---

#### Argument Question 3 — Does the "hybrid pattern is configuration-only" claim hold?

**Location:** §Findings, "Latency is the binding constraint..." (para 4): "The hybrid pattern is configuration-only — the four-layer architecture supports it natively because `invoke_ensemble` always routes through Tool Dispatch to the local Ensemble Engine regardless of where the orchestrator LLM runs."

**Claim assessed:** The architecture "natively" supported the hybrid pattern; it is a configuration change, not a code change.

**Finding:** The research log (S0-CAP-9 entry) records: "Add `orchestrator-minimax-m25-free` profile to llm-orc config... Register the OpenCode Zen API key... Switch `agentic_serving.orchestrator.model_profile`, restart server." The research log also notes: "The cycle did not exercise this point in the design space until CAP-9 because the practitioner-imposed scope was local-only." There is no record in the research log of any code change being required to make the hybrid pattern work.

However, the cycle did commit a fourth production change that is directly relevant: the cli.py `log_level` fix, which CAP-9 surfaced as needed because the dispatch logger was dormant at `warning` level. The essay correctly describes this as a separate production gap surfaced by CAP-9. This fix is not required for the hybrid routing to work; it is required for the dispatch logger to fire. The routing logic itself required no code change.

The claim holds. The architecture's `invoke_ensemble` → Tool Dispatch → Ensemble Engine routing is provider-agnostic at the code level because the orchestrator LLM is configured via a model profile, not hardcoded. The essay's claim is accurate.

**Assessment:** No issue. The "configuration-only" claim is well-grounded.

---

#### Argument Question 4 — Does the "validated model selection" claim integrate cloud orchestrators correctly?

**Location:** §Conclusion (para 1): "The validated model selection within the cycle's spike battery is narrower than the model list might suggest: qwen3:8b is in (local), MiniMax M2.5 Free via Zen is in (cloud orchestrator), qwen3.5:9b is out (premature stop after first tool call, three candidate causes unresolved), and deepseek-r1:8b is out at the platform level."

**Claim assessed:** This is the in/out classification with CAP-9 incorporated.

**Finding:** The classification is internally consistent with the trial record. qwen3:8b: multiple trials, CAP-1/CAP-3/CAP-3b, positive result with biased prompt. MiniMax M2.5 Free via Zen: CAP-9, single trial, positive result. qwen3.5:9b: CAP-7 + CAP-7-rerun (n=2), negative result. deepseek-r1:8b: CAP-8, n=1, negative (platform-level rejection).

One logical asymmetry worth noting: qwen3.5:9b is "out" based on n=2 trials, and the essay gives reasons (three candidate causes). MiniMax M2.5 Free is "in" based on n=1 trial, with no equivalent hedging on reproducibility. The asymmetry between the treatment of the "out" cases (where n=2 confirms the failure) and the "in" case (where n=1 is treated as validation) is the same proportionality issue identified in Argument Question 1.

**Assessment:** Covered by the P2 recommendation under Argument Question 1. No separate issue; same root cause.

---

#### Argument Question 5 — Does the "operators choose between shapes based on privacy preference, interactivity tolerance, and cost sensitivity" claim derive from empirical work?

**Location:** §Findings, "The dual contracts converge..." (para 3, final sentence): "Operators choose between shapes based on privacy preference (local-only keeps everything on-device), interactivity tolerance (hybrid is interactive-acceptable, local-only is batch/async territory), and cost sensitivity (local-only avoids cloud dependence; hybrid leverages free or near-free orchestrator inference where available)."

**Claim assessed:** Whether the three operator-choice axes (privacy, interactivity tolerance, cost sensitivity) derive from the empirical work or are newly synthesized framing.

**Finding:** The three axes are not empirical findings from the spike battery — no spike tested privacy-sensitive workloads, no user study was conducted on interactivity tolerance, and no cost analysis was performed. They are derived by inference from what the two deployment shapes differ on, not from operator behavior observations.

However, the derivation is logically sound and the essay does not claim these are empirical findings — it presents them as implications ("operators choose between shapes based on..."). The reasoning is: local-only keeps inference on-device (data-sovereignty argument), hybrid requires a cloud call (latency argument resolved, privacy traded), free-tier cloud inference was demonstrated by CAP-9 (cost argument). Each axis maps to a real architectural difference.

The claim is synthesis framing, not empirical finding, but it is accurately labeled as implication rather than observation. The essay does not say "operators were observed choosing based on these criteria" — it says they would be rational to choose based on them.

**Assessment:** The synthesis framing is appropriate given how it is labeled. One minor precision issue: the claim that "hybrid leverages free or near-free orchestrator inference where available" is accurate for MiniMax M2.5 Free via Zen at the time of the spike, but "where available" is doing real work — free tiers are not guaranteed to remain free, and the essay presents this as a general property of the hybrid shape rather than a property of the specific service tested. See framing audit Question 3 below.

**Assessment:** P3 — minor precision issue. The synthesis framing is not a logical error but the "free or near-free" characterization should be marked as provider-and-tier-specific rather than a general property of cloud orchestration.

---

### P1 — Must Fix

None.

---

### P2 — Should Fix

#### P2-1 — Hybrid finding's n=1 not reinforced at claim sites

**Location:** §Findings, "Latency is the binding constraint for local-only deployment..." (para 3–4); §Conclusion (para 1, second paragraph).

**Claim:** "CAP-9 demonstrated that a different deployment shape closes this gap" / "A hybrid configuration... delivers the same response quality at approximately one minute end-to-end."

**Evidence gap:** The §Method caveat covers all single trials generically, but the hybrid claim in §Findings and §Conclusion uses declarative framing ("demonstrated," "delivers") that projects a confidence level higher than n=1 warrants. The essay distinguishes reproducible findings (qwen3.5:9b's failure "reproduced across two trials") from single-trial findings elsewhere — it should apply the same epistemic annotation to the hybrid result, which is a single trial with a single provider.

**Recommendation:** At the first claim site in §Findings, soften "CAP-9 demonstrated" to "In a single trial, CAP-9 showed." In §Conclusion, soften "delivers the same response quality at approximately one minute end-to-end" to "produced comparable quality in approximately one minute in a single trial." These are minimal changes that preserve the finding's significance while accurately representing the evidence base.

---

#### P2-2 — "Free or near-free orchestrator inference" presented as a stable property of the hybrid shape

**Location:** §Findings, "The dual contracts converge under either of two validated deployment shapes" (para 2, final sentence); §Conclusion (para 1).

**Claim:** "hybrid leverages free or near-free orchestrator inference where available" / "A hybrid configuration (cloud orchestrator via free-tier MiniMax M2.5 on OpenCode Zen, local ensembles unchanged) delivers the same response quality."

**Evidence gap:** CAP-9 used MiniMax M2.5 Free on OpenCode Zen's free tier. Free tiers are provider decisions, subject to change, and the essay presents "free or near-free" as a characteristic of the hybrid deployment shape rather than as a characteristic of the specific provider and tier tested. MiniMax M2.5 Free is the empirically tested instance; "free or near-free orchestrator inference" is a generalization that other cloud-cheap or free providers (Together, Groq, other Zen models) were not tested. The essay nowhere names the specific endpoint as a scope condition for the cost claim.

OpenCode Zen is named in the essay as the route for the hybrid trial, but the inference "operators can leverage free or near-free orchestrator inference" generalizes beyond what was tested. A practitioner reading this who uses a different cloud provider may find that "near-free" does not describe their orchestrator cost.

**Recommendation:** Scope the cost claim to the tested configuration: "Cost sensitivity: local-only avoids cloud dependence; hybrid incurs orchestrator inference cost — the cycle's CAP-9 trial used a free-tier provider (MiniMax M2.5 Free via OpenCode Zen), making orchestrator cost negligible in that configuration." This removes the implicit generalization while preserving the accurate claim about what was tested.

---

### P3 — Consider

#### P3-1 — OpenCode Zen as the route to cloud orchestration is not named as a scope condition

**Location:** §Findings, "Latency is the binding constraint..." (para 3): "cloud orchestrator + local ensembles) deployment pattern works end-to-end" with "MiniMax M2.5 Free via OpenCode's Zen platform."

**Issue:** The essay names MiniMax M2.5 Free via OpenCode Zen as the tested provider, but the broader "hybrid deployment shape" framing in the dual-contracts finding and §Conclusion speaks to the pattern generally. Other cloud-cheap providers (Together AI, Groq, Fireworks, other Zen models) were not tested. The architecture is provider-agnostic by design (any OpenAI-compatible endpoint), but whether other providers' free or cheap tiers expose tool-calling correctly and at similar latency is untested. The essay's framing moves from "this specific provider worked" to "the hybrid pattern works" without naming the generalization step.

**Recommendation:** Add a scope parenthetical in the dual-contracts section noting that the hybrid validation was conducted via OpenCode Zen; other providers expose the same interface but have not been tested in this cycle. Minor framing precision; does not affect conclusions.

---

#### P3-2 — "Argument-confabulation occurs within the validated qwen3:8b configuration" — the hybrid configuration's argument-confabulation profile is uncharacterized

**Location:** §Findings, "A failure-mode taxonomy emerged" (argument-confabulation entry): "a behavior observed during CAP-3 under summarization-retry stress, within the validated qwen3:8b configuration."

**Issue:** The essay now has two validated configurations: local-only qwen3:8b and hybrid MiniMax M2.5 Free. The argument-confabulation taxonomy entry scopes the behavior to "the validated qwen3:8b configuration" — which is now one of two validated configurations, not the only one. CAP-9 did not test the hybrid configuration under summarization-retry stress, so the argument-confabulation failure mode's presence or absence in the hybrid configuration is uncharacterized. The taxonomy entry's scoping is accurate (argument-confabulation was observed in qwen3:8b, not in CAP-9) but it may leave a reader uncertain whether the hybrid configuration is free of this risk.

**Recommendation:** Add a brief note to the argument-confabulation entry: "The hybrid configuration (MiniMax M2.5 Free) has not been tested under equivalent summarization-retry stress, so its behavior under this failure mode is uncharacterized." This is a precision addition, not a finding change.

---

## Section 2: Framing Audit

### Question 1: What alternative framings did the evidence support?

#### Alternative A — "Hybrid deployment is the primary recommendation; local-only is a fallback"

The evidence base after CAP-9 supports a framing where the hybrid pattern (62-second end-to-end, interactive-acceptable) is the primary validated configuration and local-only (6-minute, batch/async) is the secondary. The essay's structure presents them as co-equal alternatives with operators choosing based on preference axes. But on the user-contract dimension — "the user gets a useful result" — the hybrid configuration is unambiguously stronger: it satisfies both the output quality criterion and the latency criterion that local-only fails on. The local-only shape satisfies output quality but fails the latency portion of the user contract for interactive workflows.

What would the reader need to believe for this framing to be right? That the user contract includes latency tolerability as a first-class criterion, which is exactly what the essay argues in the local-only latency section. Under this framing, the essay's "operators choose between shapes" neutrality understates the asymmetry: hybrid is better on almost every user-contract dimension except privacy.

The essay's co-equal framing is defensible — privacy is a real axis and local-only is genuinely useful for async/batch contexts. But a reader optimizing for user-contract satisfaction rather than privacy would be better served by a framing that foregrounds hybrid and positions local-only as the privacy-prioritizing fallback.

**Belief-mapping:** A reader would need to believe (a) interactive-acceptable latency is the default expectation for an agentic system, and (b) local-only's privacy advantage is a specialized rather than general concern. Both are reasonable priors for developers using llm-orc in a standard non-regulated environment.

#### Alternative B — "The validated hybrid configuration validates one free-tier provider, not a class of deployments"

The CAP-9 result is specific to MiniMax M2.5 Free via OpenCode Zen. The essay frames it as validating the "hybrid deployment shape" as a pattern. Under this alternative framing, the essay's finding is "we found one free cloud provider that works for this task class" — a narrower, more contingent conclusion than "the hybrid pattern is a validated configuration option." The distinction matters because free-tier availability varies, provider tool-calling support varies, and the latency advantage of cloud orchestration depends on the cloud provider's response time.

**Belief-mapping:** A reader would need to believe that a single n=1 trial with a single free-tier provider does not validate a deployment pattern in a stable, operator-recommendable way. This is consistent with the essay's own §Method small-n caveat.

#### Alternative C — "Privacy as a deployment differentiator deserves more empirical grounding"

The essay asserts that local-only provides a privacy advantage (all inference on-device) as an operator-choice axis. This is architecturally accurate but the privacy claim is not empirically tested. No spike examined what data leaves the device in a hybrid configuration, what the cloud provider's data retention policies are, or what threat model local-only protects against. The privacy axis is a logical inference from the architecture, not an empirical finding. A reader making a real deployment decision based on privacy would need more than a logical inference — they would need verified data about the specific cloud provider's practices.

**Belief-mapping:** A reader would need to believe that architectural inference ("cloud call sends data to cloud") is sufficient for a privacy recommendation, rather than verified provider data practices. This is a reasonable simplifying assumption in many contexts but may not hold in regulated or high-sensitivity environments.

---

### Question 2: What truths were available but not featured?

#### Finding A — Other free-tier models on OpenCode Zen were not tested

**Where it appears in source material:** CAP-9 research log records that Zen's `/v1/models` endpoint lists eleven models, four of which are free-tier: minimax-m2.5-free, hy3-preview-free, ling-2.6-flash-free, and nemotron-3-super-free. Only MiniMax M2.5 Free was tested.

**Why excluded:** The spike stopped at the first successful result, which is appropriate scope discipline.

**Would inclusion change the argument?** It would strengthen the hybrid framing if other free-tier models also work, or complicate it if only MiniMax does. Its absence means the "any tool-calling-capable cloud orchestrator" claim in the dual-contracts section is stated without empirical support for the "any" scope.

#### Finding B — The cli.py log-level fix is a correctness fix, not a production enhancement

**Where it appears in source material:** CAP-9 research log: "the dispatch result logger... IS NOT firing in production deployment. Cause: `cli.py:368` passes `log_level='warning'` to `uvicorn.run`, which sets the effective log level for child loggers to WARNING." The fix changes the log level so that the dispatch logger committed earlier in the cycle actually fires.

**Why this matters:** The essay presents the cli.py fix as the fourth production change in §What Ships, but it is specifically a fix that makes a prior production change (the dispatch-result logger) actually function in default deployment. Without the cli.py fix, the dispatch-result logging change shipped in this cycle would have been silent by default. The essay's §What Ships does explain the fix accurately ("the dispatch logger added mid-cycle was itself initially dormant") but this is in §Conclusion, not in §What Ships where the fix is described. A reader of §What Ships alone would not understand why this fix is necessary.

**Would inclusion change the argument?** It does not change any conclusion, but it would improve the §What Ships description to note that the fourth change makes the second change functional in default deployment — the two changes are coupled.

#### Finding C — The argument-confabulation behavior in the hybrid configuration is entirely uncharacterized

**Where it appears in source material:** DIAG-1 (research log) documents argument-confabulation under qwen3:8b retry stress. CAP-9 research log records the hybrid cascade completing in 62 seconds with no retries needed (summarization succeeded). There was no opportunity to observe argument-confabulation in the hybrid configuration.

**Why this matters:** The essay notes that argument-confabulation is "a known limitation that future work should address" in the qwen3:8b configuration. The hybrid configuration did not produce the stress conditions (repeated summarization failure) that elicited argument-confabulation in local-only. Whether MiniMax M2.5 Free would exhibit the same behavior under stress is unknown. The essay's §Conclusion naming argument-confabulation as a remaining gap applies specifically to the local-only validated configuration; its status in the hybrid configuration is simply unstated.

**Would inclusion change the argument?** It would add precision without changing conclusions.

---

### Question 3: What would change if the dominant framing were inverted?

The revised essay's dominant framing is: "both validated deployment shapes (local-only and hybrid) converge on the dual-contract goal; the choice between them is a preference question for operators."

Inverted: "the local-only shape fails the user contract on latency for interactive use; the hybrid shape passes the user contract but has not been validated beyond a single free-tier trial with a single provider; neither shape has a stable, production-ready recommendation without additional work."

Under the inverted framing:

**Claims that become weaker:**
- "The dual contracts converge under either of two validated deployment shapes" — 'validated' is doing a lot of work. Local-only is validated over multiple trials; hybrid is validated once against a free-tier provider whose availability is not guaranteed.
- "Operators choose between shapes based on privacy preference, interactivity tolerance, and cost sensitivity" — under the inverted framing, there is currently only one robustly validated shape for interactive use (hybrid), and its stability is contingent on free-tier availability. The choice is not between equivalently validated options.

**Claims that become stronger:**
- The §Open Questions acknowledgment that "the multi-turn workflow question" remains open becomes the central finding under the inverted framing. A single successful cascade on a single-ask task class does not validate either shape for multi-turn workflows.
- The small-n caveat in §Method becomes load-bearing rather than hedging-boilerplate.

**What the essay would need to address if it took the inverted framing seriously:**
- The hybrid recommendation would need either a stability caveat (free-tier availability is not guaranteed) or additional trials to confirm the finding is not a one-off.
- The "validated configuration" language would need to be differentiated: local-only has multiple trials supporting it; hybrid has one.

The current framing is defensible for an exploratory research essay reporting a novel finding. The inverted framing is not more correct — it is more conservative. The issue is whether the essay's confidence language in §Conclusion and the dual-contracts section accurately reflects the asymmetric evidence base. The P2-1 recommendation above addresses this directly.

---

### Framing Issues

#### P2 — Underrepresented alternatives

**FI-1 — The hybrid finding's validation asymmetry relative to local-only is not visible in the essay's "two validated deployment shapes" framing**

**Location:** §Findings, "The dual contracts converge under either of two validated deployment shapes" (entire section); §Conclusion (para 1, second paragraph).

**Issue:** The essay presents local-only (qwen3:8b, multiple trials, n=2 for qwen3.5:9b exclusion, n=1 for CAP-3b success) and hybrid (MiniMax M2.5 Free, n=1) as "either of two validated deployment shapes" — language that treats them as epistemically equivalent. They are not. Local-only has broader empirical support within this cycle (multiple models tested, multiple failure modes characterized, multiple trials on the successful path); hybrid has a single successful trial. The §Method caveat about small n applies to both, but the essay's "two validated deployment shapes" framing obscures the difference.

**Recommendation:** Differentiate the confidence language: "Local-only is validated across multiple trials at the qwen3:8b tier. The hybrid shape was validated in a single trial; its stability across providers, over time, and under stress conditions requires further runs." This can be a single parenthetical addition rather than structural revision.

---

#### P2 — Framing concern on free-tier stability

**FI-2 — "Free or near-free" is presented as an inherent property of the hybrid shape rather than a property of the specific provider tested**

**Location:** §Findings, "The dual contracts converge..." (para 3): "hybrid leverages free or near-free orchestrator inference where available."

**Issue:** Already noted in Argument Section P2-2 above, but the framing dimension is worth stating separately: the essay's operator-choice table ("Operators choose between shapes based on...cost sensitivity (local-only avoids cloud dependence; hybrid leverages free or near-free orchestrator inference where available)") generalizes from CAP-9's MiniMax M2.5 Free trial to a general property of the hybrid pattern. The "where available" qualification does acknowledge contingency, but it buries the contingency inside a parenthetical of an operator-choice list. A reader skimming for the cost axis will see "hybrid leverages free or near-free orchestrator inference" and not notice the "where available" qualifier.

**Recommendation:** As noted in P2-2 above, scope the cost claim to the tested configuration. The framing issue is the same as the argument issue — it is a single fix that addresses both.

---

#### P3 — Minor framing concerns

**FI-3 — "The model-profile condition broadens to 'any tool-calling-capable cloud orchestrator'" is stated without empirical support**

**Location:** §Findings, "The dual contracts converge under either of two validated deployment shapes" (para 2): "the model-profile condition broadens to 'any tool-calling-capable cloud orchestrator.'"

**Issue:** The essay tested one cloud model (MiniMax M2.5 Free). The claim that the condition "broadens to any tool-calling-capable cloud orchestrator" is a logical generalization from the architecture (any OpenAI-compatible endpoint works) not an empirical finding. This is the same generalization pattern as the free-tier cost claim, applied to model selection. It is directionally plausible but untested beyond one provider.

**Recommendation:** Soften to "the model-profile condition is not tied to local hardware — CAP-9 used a cloud orchestrator (MiniMax M2.5 Free via OpenCode Zen); other OpenAI-compatible cloud endpoints should work by the same architecture, though this cycle tested only one." Minor precision addition.

---

**FI-4 — The cli.py log-level fix and the dispatch-result logging fix are causally linked but presented as independent changes in §What Ships**

**Location:** §What Ships (para 1).

**Issue:** The dispatch-result logging fix (second change) and the cli.py log-level fix (fourth change) are coupled — without the fourth, the second produces no output in default deployment. CAP-9 surfaced the fourth change precisely because the second change had been committed but was silent. A reader of §What Ships who deploys the second change without the fourth would unknowingly have a silent logger. The relationship between the two changes is explained in §Conclusion but not in §What Ships.

**Recommendation:** Add a parenthetical in §What Ships noting the coupling: the fourth change makes the second change functional in default deployment by ensuring the `llm_orc` logger fires at the level it was written for. Minor clarity addition.

---

## Clean-Status Determination

**P1 issues remaining:** 0
**P2 issues remaining:** 4 (P2-1: hybrid n=1 not reinforced at claim sites; P2-2: "free or near-free" over-generalized; FI-1: validation asymmetry between shapes not visible; FI-2: framing dimension of the free-tier claim)

Note: P2-2 and FI-2 are the same root issue — the "free or near-free" generalization — addressed once fixes both the argument and framing dimensions.

Similarly, P2-1 and FI-1 are the same root issue — the n=1 validation asymmetry — addressed once fixes both sections.

**In practice, the P2 issues reduce to two distinct fixes:**
1. Add hedging language at hybrid claim sites to mark n=1 (addresses P2-1 + FI-1).
2. Scope the cost claim to the tested provider and tier (addresses P2-2 + FI-2).

**P3 issues remaining:** 4 (P3-1: OpenCode Zen as scope condition; P3-2: hybrid configuration's argument-confabulation profile uncharacterized; FI-3: "any tool-calling-capable cloud orchestrator" untested generalization; FI-4: cli.py and dispatch-logger coupling not visible in §What Ships)

---

## Cycle Advancement Recommendation

**The essay is not clean at P2.** Two distinct P2 fixes are required before the gate closes and the pipeline advances:

1. **Mark the hybrid finding's n=1 status at claim sites** — specifically in §Findings para 3–4 and in the §Conclusion para 1 second paragraph. The §Method caveat does not propagate to these claim sites automatically; confident declarative language at the point of claim conflicts with the §Method caveat.

2. **Scope the "free or near-free" cost claim to the tested provider and tier** — in the dual-contracts §Findings para 3 and in §Conclusion. The claim as written generalizes beyond the empirical evidence and could mislead operators making deployment decisions.

Both are single-sentence additions or substitutions. Neither requires structural revision. The essay's conclusions are sound; the issue is proportionality between evidence strength and confidence language at the point where the hybrid finding is claimed.

**If those two fixes are applied, the essay clears P1 and P2, and the remaining P3 issues can be addressed at the practitioner's discretion without triggering another audit pass.**

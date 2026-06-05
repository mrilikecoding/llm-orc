# Research Design Review — Spike θ (Termination-Mechanism DECIDE-Entry Probes)

**Reviewed question set:** `docs/agentic-serving/essays/research-logs/cycle-7-spike-theta-termination-mechanism.md` — full pre-registered design (arms θ.1a/b, θ.2a/b, hosted θ.1h/θ.2h, decision rules 1–5)
**Constraint-removal response included:** n/a (this is a mechanism-selection spike, not a primary question set in the ADR-082 sense; the entry analysis "candidate collapse" plays a structurally analogous role and is evaluated inline below)
**Date:** 2026-06-05
**Reviewer role:** research-methods (ADR-060 + ADR-082 dimensions 1–4 applied)

---

## Summary

- **Arms reviewed:** 4 local primary (θ.1a/b, θ.2a/b) + 2 hosted secondary (θ.1h, θ.2h)
- **Decision rules reviewed:** 5 pre-registered rules
- **Flags raised:** 9 (2 P1, 4 P2, 3 P3)
- **Criteria applied:** 1–4 (ADR-082)

---

## Per-Question Review

### The spike question: "Does an explicit continue-vs-stop judgment call classify the tail correctly — at a rate that beats the measured implicit judgment — without exhibiting the ω.3a flip failure mode?"

**Belief-mapping.** The question frames the spike as a rate comparison between explicit and implicit judgment. It assumes the comparison is between two forms of call-1 (explicit question vs. no question), with call-2 held constant at the measured E4b composition. A different productive question: "Is a two-step composition (judgment call + conditional continuation) the right level at which to introduce explicit reasoning, or does the right boundary lie at the harness integration layer — framework decides finish, model decides delegation?" That question is adjacent and excluded by framing the spike as a judgment-form probe rather than a judgment-placement probe. The entry analysis addresses this partially via the candidate collapse, but the spike question itself does not surface it.

The question also presupposes the binary work-complete / work-remaining classification is the right task for call 1. An adjacent question excluded by this framing: "Are there tail shapes that are genuinely ambiguous — partial completion where reasonable agents would disagree — and if so, does the explicit judgment call produce a stable verdict on those, or does instability in the ambiguous region mean the mechanism's real cost is misclassification frequency under ambiguity rather than under clarity?" The design's E4 base is constructed-adjacent with unambiguous remaining work (one of two named files written). No arm samples near-complete tails.

**Embedded conclusions.** The question presupposes the judgment call is being asked in a context where the two-call composition is the mechanism to test. This is appropriate — the entry analysis argues the candidate space collapses to two-call composition — but see the candidate-collapse evaluation below for whether that argument is fully closed.

**Scope.** Appropriately scoped for the DECIDE-entry position. The spike tests the unmeasured pieces of the named live candidate rather than reopening the full mechanism space.

---

### Entry analysis: candidate collapse

**Belief-mapping.** The collapse argument closes candidates (a), (b), and (c) analytically. What would the researcher need to believe for a different question to be more productive? For (b): the argument that "task-completeness is not framework-computable" would need to be wrong — specifically, that there exists a framework-observable signal (e.g., the presence or absence of a pending user-supplied deliverable count in the original task message) that approximates completeness reliably enough to short-circuit the model call on a majority of tails. The design does not examine this possibility; it treats the signal absence as categorical. That treatment is likely correct given the E4 construction discipline (real system prompt + tool list; only task text and tail truncated), but the assumption deserves naming.

For (c): the closure argument rests on three data points (E3 1/10; V2 wording arms lost regardless; demotion did not remove authority). This is a strong pattern but all three observations are from guidance-present arms — the spike never ran a form that restructured guidance into conditional branches (e.g., a guidance text that contains both a "if work complete" branch and a "if work remaining" branch). E3 was a completion clause appended to an unconditional do-more-work text. The collapse of (c) is sound for the wording-of-existing-text direction, but "no new conditional guidance structure" is slightly stronger than what was measured.

**Embedded conclusions in the entry analysis.** The call-2 pin — "the judgment exchange does not ride into call 2's context" — is a design commitment that dissolves the third unmeasured piece by assertion. The pin is stated as the "minimal-change choice," which is reasonable, but it embeds the conclusion that the enriched-brief variant (judgment exchange feeds call 2's what-remains field) is inferior without measurement. This is named as "unexplored enhancement territory" in the design, which is the right disclosure posture, but the enhancement is not flagged as a named follow-on or as a coverage gap in the decision criteria. If θ.2 shows the explicit question produces REMAINING verdicts at 8–9/10 but that call-2 delegation rate under the pin is lower than E4b's 9/10 (because call 2 receives no "here is what remains" context that would anchor the ensemble brief), the degradation from the pin would be invisible to this spike. The pin's cost is unmeasured.

---

## Question Set Assessment

### Premature narrowing / prior-art treatment

The design enters with the prior art (E1, E2, E3, E4a, E4b) as the explicit reference frame and builds every threshold against those measured numbers. This is correct prior-art treatment. The entry analysis's candidate-collapse section treats the existing guidance text as prior art to be evaluated rather than as a constraint to inherit — the structural analog of the ADR-082 constraint-removal response. The collapse is analytically honest about what was and was not measured.

One narrowing concern: the design tests the judgment call only on the two tail shapes that motivated the spike (work-complete and work-remaining), both using the same captured/constructed base. There is no arm on a third shape that the production system will encounter: multi-turn mid-session tails where the current tool result is one of several sequential steps in a larger task, and "work remaining" is genuinely ambiguous (work was assigned implicitly by the model's prior reasoning, not stated explicitly in the original user message). This gap is named as appropriate exclusion by scope, but it means the spike's pass result licenses adoption on the two measured tail shapes — not on the broader tail distribution. The ADR should be precise about this scope.

### Incongruity surfacing

A relevant incongruity exists in the prior context that the design does not surface for examination. Spike ψ established that the work-complete tail, without any guidance, produces 10/10 finish (E2). The spike θ design asks whether the model can correctly classify a work-complete tail at ≥9/10. The incongruity: the model already terminates 10/10 on the work-complete tail when guidance is absent. The explicit judgment question (Form A or Form B) is being asked on a tail that the model handles correctly without any intervention. The question being measured is: "Can the model, when explicitly asked 'are you done?', correctly answer yes?" — and the E2 evidence says it would answer yes implicitly (by finishing) at 10/10.

The incongruity is this: E2's 10/10 is not a judgment call — it is the model's default behavior. θ.1 is testing whether a judgment question appended to the E2 base can reproduce E2's accuracy as a text verdict. If θ.1 passes at ≥9/10, the result is that the judgment question correctly elicits the judgment the model would have made anyway. The mechanism adds a call and a text verdict, but the work-complete tail was not the broken case — E1 was. What the design is really testing on the work-complete side is whether the judgment call avoids false-continue (REMAINING on a work-complete tail), which is a different question from "does the model correctly classify?" The θ.1 pass threshold is operationalizing "does not regress from E2's implicit accuracy," not "adds value over E2."

This distinction does not invalidate the design — false-continue on θ.1 would be a genuine hazard that disqualifies adoption — but the framing in the decision rule says "rate that beats the measured implicit judgment (E2 10/10)," which sets a bar the explicit question could theoretically fail to meet. The question the design is missing: "Is the judgment question on the work-complete side adding anything beyond not-regressing, and if not, is there a simpler mechanism for the work-complete branch (e.g., the E2 drop-guidance path) that makes call 1 unnecessary when the tail is work-complete?" The two-call composition as designed fires call 1 on every non-user tail, including tails that E2 shows would finish naturally without guidance. Call 1 on those tails is overhead with no correction value — the correction value is only on work-remaining tails. The design does not ask whether call 1 could be reserved for the work-remaining case only (which would require some cheaper pre-filter, recovering candidate (a)'s structure at a different level). This is a coverage gap in the decision criteria, not a flaw in the arms themselves.

---

## Findings

### P1 — Design flaws that would invalidate conclusions before run

**P1-A: The work-remaining tail exists in only one constructed variant, creating an undisclosed single-point-of-failure for decision rule 1's θ.2 threshold.**

The work-remaining base is the ψ″ E4 constructed-adjacent context: two-deliverable task, one write pair completed, task text edited into the captured system prompt + tool list. This is a single construction at a single task structure (string_utils.py + test_string_utils.py), a single depth (one of two files), and a single task phrasing. Decision rule 1's θ.2 correct ≥8/10 is measured against this one construction.

The design acknowledges this construction is not captured (the construction caveat carries forward verbatim from ψ″), but does not address the implication for decision rule 1: if θ.2 shows 9/10 correct on this specific base, the spike licenses adoption of the two-call composition across the full class of work-remaining tails in production — tails that include: tasks with more than two deliverables, tasks where the remaining work is not a file write, tasks where the prior tool actions include non-write results (read results, bash output, error results), and tasks with longer or more syntactically complex original requests. A single constructed base at one depth and one task type is a narrow sample for an adoption decision.

This is structurally analogous to the ψ′ P1-B finding (four A phrasings all single-sentence new-file generation tasks licensing a "phrasing generalization" conclusion). The θ.2 result at n=10 on one base measures accuracy on that base; it does not characterize accuracy on the class.

**Recommended design change.** Option (a): add one additional work-remaining variant at a different depth and task type before running θ.2 — e.g., a three-deliverable task where two of three are completed, or a task whose remaining work is not a file write but a test run. This adds n=10 per form (20 additional local runs, $0) and broadens the class coverage. Option (b): if cost or scope prevents (a), bound the ADR's scope of claim to "the E4 task structure at one-of-two depth" and treat generalization across tail shapes as a named open risk to be measured at BUILD acceptance. Option (b) is the lower-cost path and may be appropriate given the spike's position, but the scope restriction must appear explicitly in the ADR language, not only in the spike record.

---

**P1-B: The unparseable-run handling silently inflates accuracy denominators.**

The pre-registered measurement defines unparseable response as "failure for that run (counts toward neither verdict)." Decision rule 1 reads "θ.1 correct ≥9/10" and "θ.2 correct ≥8/10." If 2/10 runs are unparseable on θ.2a, the arm produces 8 parseable runs, all correct — the rate is recorded as 8/8 = 1.0 correct (or, more charitably, the threshold fires on 8 runs). But the underlying question is whether the mechanism works at 10-run confidence: 8/10 parseable-correct is not 8/10 correct on all runs. An unparseable run is a mechanism failure of a different kind — the model did not produce a response the harness can act on — and it should not silently exit the denominator.

The severity is conditional on unparseable frequency. If the judgment question reliably elicits a VERDICT line, the distinction is moot. But the question text is novel (not a format the model has been measured on in this corpus), and Form B in particular is a bare framework-composed context where format compliance is untested. A low but nonzero unparseable rate (2–3/10) combined with high parseable-correct could formally satisfy rule 1 while the mechanism is unreliable in practice.

**Recommended design change.** Redefine the accuracy denominator as n (total runs), not n-unparseable. Record "correct including unparseable (denominator=10)" and "correct among parseable (denominator=parseable count)" separately. Decision rule 1 fires on the n=10 denominator. The parseable-only rate is informative for characterizing the failure mode but does not substitute for the n=10 bar. This requires no additional runs, only a one-line change to the denominator definition in the harness.

---

### P2 — Weaknesses that bound the claims

**P2-A: The call-2 pin's cost (judgment exchange not carried into call 2's context) is unmeasured and unnamed as a follow-on.**

The pin dissolves the third unmeasured piece by fixing call 2 as byte-identical to E4b. This is the minimal-change choice, and its rationale is sound. But the enriched-brief variant — where the REMAINING verdict's "what remains" sentence is carried into call 2's user message — is a natural improvement whose cost under the pin is invisible to this spike. If θ.2 arms produce REMAINING verdicts that include a substantive "what remains" sentence (the question text requests this), and if that sentence would materially improve call 2's ensemble brief quality, the pin is buying simplicity at a cost that cannot be quantified until the enhancement is tested.

The design names this as unexplored enhancement territory, but does not list it as a named follow-on in the evaluation criteria section. If the DECIDE ADR adopts the two-call composition based on spike θ results, the enhancement question will arrive informally during BUILD rather than being managed as a scoped follow-on probe.

**Recommended design change.** In the evaluation criteria section, add the enriched-call-2-brief as a named follow-on work item with the condition that triggers it: "If the REMAINING verdict text consistently contains a substantive what-remains sentence (observable in θ.2 per-run records), queue a follow-on probe comparing E4b delegation rate with and without the brief enrichment, as a named BUILD gate." This costs nothing now and prevents the enhancement from arriving unstructured during BUILD.

---

**P2-B: Form B's flip-watch operationalization is applied to a context the ω spike did not test, and the threshold lacks a reference bar.**

The flip-watch definition (fenced code block or >10 contiguous code-shaped lines) is adapted from the ω hazard. ω's 3a result showed every tested model (including hosted models) flipped 10/10 on a user-turn adversarial input in the broker composition — but all of those models were smaller than qwen3:14b, and the composition was broker-shaped (delegating vs. carrying decision). Spike θ's Form B is also broker-shaped (framework-composed judgment context, no client system prompt, task as data), and qwen3:14b's flip behavior on this shape is unmeasured.

The disqualification threshold of >2/10 flips is not anchored to any prior measurement on qwen3:14b or on the specific Form B composition. It is borrowed from the ω decision rule, where ≤1/10 was the pre-registered bar. In ω, ≤1/10 was set against 10/10 flip rates observed in pilot — the bar was calibrated against an observed failure mode distribution. For Form B on qwen3:14b, there is no pilot data. A threshold of >2/10 may be too loose (if flips occur at 3–4/10, the mechanism is unreliable but formally passes) or too strict (if qwen3:14b shows 3/10 flips on Form B due to the judgment question triggering reasoning traces that include brief code illustrations, those may not represent genuine task-work intrusion).

The second issue: the code-shaped-lines operationalization does not distinguish between "the model worked the task" and "the model illustrated its reasoning with a code snippet in the course of answering the judgment question." A response that says "VERDICT: COMPLETE — the three write operations have completed successfully, as shown in this summary:" followed by a table formatted with pipes and dashes could hit >10 contiguous code-shaped lines without being a flip in the ω.3a sense. Conversely, a model that emits a full function definition while also emitting VERDICT: REMAINING is a genuine flip that the operationalization catches.

**Recommended design change.** Tighten the flip operationalization: classify a run as flipped only if the code block or code-shaped content appears BEFORE the VERDICT line OR contains substantive function/class definitions (not tables, comments-only, or code illustrations). Record borderline cases separately ("formatting artifact vs. genuine work intrusion") for per-run characterization. The disqualification threshold of >2/10 is reasonable as a starting point; calibrate after examining the first arm's runs if unexpected patterns emerge.

---

**P2-C: Decision rule 2's implicit variant is composed across three separately measured arms, and the n=10 basis makes the product rate unreliable.**

Rule 2 names the implicit variant's mid-task delegation rate as "0.54 (4/10 work-drop)" derived from E2 (10/10 stop-correct on work-complete) × E4a (6/10 continue-correct without guidance on work-remaining). The design calls this "fully computable from E2/E4a/E4b" — and the arithmetic is correct as a product of independently measured rates. But treating the product as a measured rate obscures its statistical character: it is the expected value of a product of two Binomial(10, p) variables, each estimated from 10 runs. At n=10, the 95% confidence interval for E4a's 6/10 is roughly [0.26, 0.88] (exact Clopper-Pearson). The product 0.6 × 0.9 = 0.54 has a confidence interval that is substantially wider than either component — in principle it could be consistent with anything from 0.23 to 0.79. Presenting "0.54 with 4/10 work-drop" in the ADR as the implicit variant's measured performance implies more precision than the component n=10 samples support.

This is not a design flaw that blocks the run — rule 2 fires only if no form reaches θ.2 correct >6/10, which is a clear failure signal. The concern is with how the ADR will present the implicit variant's 0.54 figure in the DECIDE evaluation. If it is presented as a measured rate rather than an estimated product, the comparison to two-call composition's (θ.2 × 0.9) product will be at similar precision, and the comparison will appear cleaner than the underlying samples warrant.

**Recommended design change.** In the decision rule section and in the ADR, present the implicit variant's 0.54 as "E4a 6/10 × E4b 9/10 (product of separately measured n=10 arms)" with the acknowledgment that the component confidence intervals make the product estimate imprecise. When θ.2 results are available, the two-call composition's rate should be presented the same way (θ.2 rate × E4b 9/10, with explicit product-of-estimates labeling). This costs nothing and prevents the ADR from claiming more precision than the measurements support.

---

**P2-D: The hosted secondary arms have a latent informal-influence path into the adoption decision despite the explicit "decision rule does NOT read them" gate.**

The design correctly scopes the hosted arms as portability annotation only, not primary evidence. However, the way the results will be presented in the spike record creates a latent influence path: the spike verdict section will contain θ.1h and θ.2h results alongside θ.1a/b and θ.2a/b results in the same record. If θ.1h and θ.2h show substantially different pass rates from the local arms (e.g., hosted passes at 9/10 but local Form B fails at 6/10), the practitioner will see the hosted result while writing the ADR and will face a motivated-reasoning hazard: the hosted result is evidence that the mechanism works somewhere, which could soften the interpretation of a local-form failure.

The same risk operates in the other direction: if θ.2h fails (work-remaining classification worse on minimax-m2.7), the portability annotation is a negative result that the decision rule says to ignore but which the ADR author will have to consciously set aside when assessing adoption feasibility.

**Recommended design change.** Pre-register the hosted arms' reading discipline in the spike record before any runs: "θ.1h/θ.2h results inform the portability annotation section of the ADR only. They are not read until the local-arm decision rule has fired. If the local arms produce a clean ADOPT or REJECT outcome, the hosted results are logged as portability annotations without influencing the primary verdict. If the local arms produce an ambiguous result (e.g., rule 3 or 5 fires but rule 1/2 is indeterminate), the hosted results remain out of scope — the ambiguity routes to a redesign, not to hosted evidence." Adding this discipline as a sentence in the pre-registration closes the influence path before the runs create the temptation.

---

### P3 — Improvements

**P3-A: The flip-watch operationalization does not distinguish "performs task work" from "echoes task content in reasoning."**

Form B carries the task text verbatim in a user message as "quoted data." If qwen3:14b's judgment response includes the quoted task text in its explanation ("The user asked for string_utils.py and test_string_utils.py; string_utils.py has been written; test_string_utils.py remains"), that content could include code-shaped lines (e.g., function signatures mentioned in passing) that trigger the >10 code-shaped-lines detector without representing genuine work intrusion. The flip definition as written could produce false flip classifications on Form B that would not appear on Form A (which has the task in the session context where the model is less likely to echo it explicitly).

**Recommended design change.** Add a secondary classification for triggered flip runs: "echo vs. generative" — did the code-shaped content appear to be quoted/paraphrased from the task text (the model repeating what was asked), or did it appear to be new code generation (the model producing work output)? Record this as a per-run annotation alongside the flip flag. This costs no additional runs and makes the characterization section more interpretable.

---

**P3-B: The no-tools rationale for the judgment call is sound but the ψ.4c reference is subtly wrong.**

The design cites ψ.4c's empty-response break as evidence that tools-less requests work correctly: "ψ.4c's empty-response break was a one-tool list misaligned with the turn shape, not a tools-less request — and a tools-less request makes any tool-call response structurally impossible." This is correct. But the framing slightly overstates the inference: ψ.4c showed that offering only one misaligned tool (invoke_ensemble on a should-finish turn) produced empty responses. This tells us that misaligned one-tool lists break the turn. It does not directly confirm that a tools-less request produces text responses — it confirms that the failure mode is misalignment, not tool count per se. The inference to "tools-less request is safe" is sound by elimination (the model cannot call what it cannot see), but it would be better grounded by noting that the tools-less path is standard LLM behavior (text-completion mode, no structured output expectation) rather than citing ψ.4c as the primary warrant.

**Recommended design change.** In the harness pre-run fidelity check, verify that a tools-less request to the same model (qwen3:14b via Ollama) on a neutral prompt produces non-empty text — a one-call smoke test before the main arms run. Record the result in the fidelity section. This adds one call at $0 and converts the "structurally impossible" claim into a directly observed one.

---

**P3-C: The "beating the implicit judgment" framing in the spike question may set an unnecessarily competitive bar that obscures the mechanism's actual value.**

The question asks whether the explicit judgment call classifies "at a rate that beats the measured implicit judgment (E2 10/10 stop-correct on work-complete; E4a 6/10 continue-correct on work-remaining)." On the work-complete side, the bar is 10/10 — the explicit call must match the implicit call's perfect rate. On the work-remaining side, the bar is >6/10 (rule 2 fires at ≤6/10). But the mechanism's value is not symmetric: the work-complete side had no guidance (E2 = drop-C3), and the explicit call is adding overhead to a case that already works. The work-remaining side had guidance that won at 9/10 for delegation (E4b) but with 4/10 premature-stop contamination (E4a) — the mechanism's goal is to separate the two.

Framing both sides as "beating the implicit judgment" weights the work-complete side incorrectly: the implicit baseline for work-complete under the two-call design is E2's drop-C3 rate (10/10), which the explicit call must not degrade. But "beating" implies the explicit call needs to do better than E2, when the correct bar is "does not degrade E2." The spike question's language is fine for the work-remaining side but misleading for the work-complete side. If θ.1 comes in at 9/10, the correct characterization is "acceptable degradation from E2 within threshold" not "fails to beat the implicit bar."

**Recommended design change.** In the verdict section, characterize θ.1 results against "does not degrade the drop-C3 E2 baseline beyond threshold" rather than "beats E2." The threshold (≤1/10 false-continue) is already correctly specified in rule 1; the spike question text should be updated to match the threshold's actual purpose. This is a prose change to the evaluation framing, not a design change.

---

## Overall Verdict

**Run with amendments.**

The spike design is structurally sound: the entry-analysis candidate collapse is honest about what was measured vs. inferred, the arms are well-specified with fidelity checks, the measurement definitions are pre-registered, and the decision rules cover the main outcome space. Two issues require fixes before running:

P1-A (single-variant work-remaining tail) should be addressed by either adding one additional work-remaining variant (preferred) or explicitly bounding the ADR scope-of-claim to the E4 task structure. P1-B (unparseable-run denominator inflation) requires a one-line change to the accuracy definition in the harness — denominator is always n=10.

The P2 findings do not block the run but should be recorded as pre-run amendments: P2-A (call-2 pin's enhancement cost is unnamed as a follow-on), P2-B (flip threshold lacks calibration reference for qwen3:14b on Form B), P2-C (product-of-estimates precision in rule 2 and ADR), and P2-D (hosted-arms reading-discipline pre-registration). The P3 findings are improvements to interpretation and fidelity verification that are cheap to apply.

The incongruity finding (call 1 fires on work-complete tails where E2 shows the model would finish correctly without any call) does not block adoption but should be named in the ADR as a known cost of the minimal-change pin: the two-call composition adds overhead on the work-complete branch where the mechanism's correction value is zero. Whether a cheaper signal for the work-complete case exists (or whether the work-complete branch can be eliminated in favor of a cheaper discriminator) is a named follow-on, not a redesign condition.

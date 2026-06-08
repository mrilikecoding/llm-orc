# Research Design Review — Spike ρ (Remaining-Work Anchor, ADR-037 Amendment)

**Reviewed question set:** `docs/agentic-serving/essays/research-logs/cycle-7-spike-rho-remaining-work-anchor.md` — full pre-registered design (arms ρ.1 B2/B3, ρ.2 B2/B3, hypotheses H-ρ.1 / H-ρ.2 / H-ρ.3)
**Constraint-removal response included:** n/a (mechanism-amendment spike; no ADR-082 constraint-removal artifact in scope)
**Date:** 2026-06-07
**Reviewer role:** research-methods (ADR-060 + ADR-082 dimensions 1–4, θ-audit calibration bar applied)

---

## Summary

- **Arms reviewed:** ρ.1 (n=10 × 2 bases) + ρ.2 (n=10 × 2 bases, denominator conditional on ρ.1 verdict distribution)
- **Hypotheses reviewed:** H-ρ.1, H-ρ.2, H-ρ.3 (null-guard)
- **Flags raised:** 7 (2 P1, 3 P2, 2 P3)
- **Criteria applied:** 1–4 (ADR-082)

---

## Per-Question / Per-Hypothesis Review

### H-ρ.1: "The judge reliably names the unproduced deliverable in its REMAINING statement."

**Belief-mapping.** The hypothesis assumes the judge's REMAINING statement is the right unit to evaluate. A different productive question: "Is the judge's one-sentence statement being evaluated as a binary correct/incorrect classification, or is there a continuum of naming quality (exact filename, correct module description, ambiguous reference, wrong) that matters for the production anchor?" If the anchor's downstream effect on ρ.2 depends on specificity — "produce the test file" anchoring differently from "produce test_string_utils.py" — then binary names-correct/names-wrong discards information the decision rule needs. The current framing collapses this to a binary that may overstate reliability for a class of responses that are correct-enough for ρ.2 in some trials and fatally vague in others.

**Embedded conclusions.** The hypothesis presupposes that the judge's one-sentence statement is the mechanism to evaluate. It does not surface the question of whether the full REMAINING response text (beyond the one sentence the question asks for) could produce a better anchor. This is appropriate given the design's goal of measuring the production form, but the definition of "one-sentence statement" versus "the post-VERDICT text" (which `strip_verdict` yields) is underspecified — see P1-A below.

**Scope.** Appropriately scoped. H-ρ.1 is exactly the unmeasured Factor 1 that the rung-1 probe identified; measuring it is the design's primary contribution.

---

### H-ρ.2: "Call 2 anchored with the judge's actual REMAINING statement advances at a rate comparable to the rung-1 hardcoded-anchor result (8/10)."

**Belief-mapping.** The hypothesis compares the production form against a hardcoded-anchor result (rung-1's 8/10). What would the researcher need to believe for a different question to be more productive? They would need to believe that a rate comparable to 8/10 is the right success bar, and that "comparable" is defined loosely enough to include [7/10, 8/10] (the pre-registered threshold). A more incisive question: "Is the composition's end-to-end advance rate sensitive to anchor wording quality — specifically, does it degrade when the judge produces a correct-but-vague statement versus a correct-and-specific one?" The spike as designed cannot answer this because ρ.2 uses whatever the judge produces; it cannot separately measure advance-given-vague-correct versus advance-given-specific-correct. This is not a design flaw (the production form is the right thing to measure), but the pre-registered decision rule treats ρ.2 advance as a monolithic rate when the underlying distribution may be bimodal.

**Embedded conclusions.** "Comparable to 8/10" embeds the assumption that the rung-1 hardcoded rate is the right reference frame. The actual production question is whether the end-to-end rate is good enough to ship — and 7/10 is pre-registered as the pass bar without justification against an absolute standard. See P2-B for the threshold motivation concern.

**Scope.** Appropriate. H-ρ.2 is exactly the second composing factor, and ρ.2's design (using the judge's actual statement, not a hardcoded string) is the right production-faithful measurement.

---

### H-ρ.3: "The anchor does not collapse delegation into inline writes or raise the no-tool-call rate beyond a tolerable bound."

**Belief-mapping.** The hypothesis is framed around two known failure modes: the Finding B inline-write collapse and the no-tool-call premature-finish risk. It does not surface a third failure mode visible in the rung-1 data: a no-tool-call response that looks like a delegation-preserved finish but is actually a stuck response (the model produces text describing what it would do rather than calling a tool). The rung-1 results note "2/10 no-tool-call text responses — NOT inline-write collapses" but do not characterize the content of those text responses. In a multi-deliverable session, a text response on call 2 ends the turn; the next trailing turn re-judges REMAINING (correctly) and the AS-3 cap backstops. This is the stated fallback. But the spike should characterize what the no-tool-call responses contain, not just count them.

**Embedded conclusions.** "Does not collapse delegation" presupposes the only delegation failure mode is inline writes. The design covers this appropriately with the `delegated` classification, but the hypothesis framing underspecifies the no-tool-call failure characterization.

**Scope.** Appropriate as a null-guard. The concern is characterization depth rather than scope.

---

## Question Set Assessment

### Premature narrowing / prior-art treatment

The design treats the rung-1 probe honestly as prior art (naming it as the grounding probe and identifying its limitations explicitly). The factor decomposition (Factor 1 × Factor 2) is the right analytical frame given the probe's scope. No premature narrowing in the core design.

One narrowing concern: ρ.2 uses the judge's real REMAINING statement without a control condition. See P1-B (content-neutral arm) for the primary concern.

---

### Incongruity surfacing

A relevant incongruity exists between the rung-1 probe's anchor design and the spike's adjudication design that the spike does not surface for examination.

The rung-1 probe's anchor text was: "The remaining deliverable not yet produced is test_string_utils.py (string_utils.py has already been written successfully). Produce test_string_utils.py next." This anchor has two functional components: (1) a filename assertion identifying the remaining deliverable, and (2) an imperative "Produce it next" that re-states the task. The design assumes it is (1) — the filename specificity — that drives the 8/10 advance rate. But the imperative (2) is also present in the hardcoded anchor, and it is not present in whatever the judge's actual REMAINING statement contains (the judge is asked to "state in one sentence what remains," not to issue an imperative).

The incongruity: the hardcoded anchor is both a naming statement and a directive; the judge's statement is only a naming statement. If the advance rate in ρ.2 comes in materially below 8/10 (say, 7/10), the design cannot determine whether the degradation is from (a) imperfect judge naming, (b) the absence of the imperative directive, or (c) the model's sensitivity to the anchor's phrasing relative to the hardcoded form. The spike's composed-advance framing folds all three into the ρ.1/ρ.2 decomposition, but only (a) is directly observable from ρ.1. The design does not ask whether the production anchor needs to be constructed as "the judge's statement + an explicit imperative" rather than the judge's statement alone.

This is the most consequential gap in the design. If ρ.2 passes at 7/10 and ρ.1 passes at 9/10, the amendment is credited to routing the judge's statement forward. But the route might succeed at 7/10 only because call-2 composition already contains other positioning that partially compensates for the absent imperative — or it might succeed at a higher rate if the production anchor is constructed as "statement + produce it next." The spike cannot distinguish these, and the ADR amendment will specify the anchor form based on whichever shape ρ.2 tested. If ρ.2 tested the statement-only form and it passes, the amendment ships statement-only; if statement-only is suboptimal relative to statement-plus-imperative, the amendment ships a suboptimal anchor and the difference will only surface during BUILD or PLAY.

---

## Findings

### P1 — Design flaws that would invalidate conclusions before run

**P1-A: The "names-correct" adjudication is insufficiently operationalized and admits subjective confirmation bias.**

The measurement definition reads: "the judge's statement references the unproduced deliverable's identity (filename or unambiguous description) and does not assert an already-produced file is missing."

Two problems:

First, "unambiguous description" is not operationalized. For the B2 base, the unproduced deliverable is `test_string_utils.py`. A statement like "the test file has not yet been written" contains no filename but may be unambiguous in context. A statement like "the second file remains" is ambiguous. The boundary between these is adjudicated by the reviewer reading the base state — but "unambiguous" is a judgment that can slide toward confirmation when the reviewer knows the hypothesis being tested (H-ρ.1 passes if names-correct ≥ 8/10). A judge response of "the test module for string_utils has not been produced" will be classified names-correct by a reviewer disposed to confirm H-ρ.1, and names-wrong by a reviewer disposed to demand exactness. This is not a hypothetical: at n=10 per base, a 2-case difference in adjudication swings the rate from 8/10 to 6/10 — the difference between pass and fail.

Second, the spike note says "full response text retained per trial for adjudication (names-correct is a judgment call read against the base state)" which correctly acknowledges the subjective element, but does not specify a tiebreaker protocol or a sufficiency standard. In the θ-audit's P1-B, the concern was denominator inflation from unparseable runs. Here the concern is adjudication drift in the pass direction for borderline cases, which is harder to detect post-hoc.

**Recommended design change:** Pre-register a two-level sufficiency standard before running:
- *Specific-correct*: the statement names the unproduced deliverable's filename or an unambiguous programmatic reference (e.g., "the test file test_string_utils.py", "test_string_utils").
- *Description-correct*: the statement references the role/type without a filename (e.g., "the unit test file", "the test module") and does not name an already-produced file as missing.
- *Ambiguous*: the statement is underdetermined (e.g., "the second file", "the remaining deliverable") — classified as names-correct only if the reference cannot plausibly refer to an already-produced deliverable.
- *names-wrong*: names an already-produced file as missing, or names no deliverable at all.

Report specific-correct and description-correct rates separately in addition to the combined names-correct rate. The decision rule fires on the combined rate; the decomposition allows post-run inspection of whether the downstream ρ.2 advance rate is higher for specific-correct trials than description-correct trials (which would indicate anchor quality matters and the binary is hiding a sensitivity).

This change requires no additional runs. It adds ~30 minutes of adjudication overhead and converts names-correct from a single binary to a three-level classification that is refutable from the retained response text.

---

**P1-B: The design provides no mechanism to determine whether the remaining-work content of the anchor drives ρ.2's advance rate, versus any trailing-text perturbation of similar length.**

The spike's central causal claim is that routing the judge's remaining-work statement forward to anchor call 2 advances multi-file progress because the model sees explicit remaining-deliverable content. The design measures this claim via ρ.2's advance rate with the judge's real statement. But it includes no control arm that would let the data falsify the alternative explanation: that any trailing content of similar length and format appended to the C3 guidance causes the advance-rate improvement, independent of remaining-work semantic content.

The rung-1 probe's anchor was a complete and meaningful statement about the remaining deliverable. The spike's ρ.2 also uses a meaningful statement. If the mechanism is that remaining-work content guides the model's target selection, that is one causal story. But if the mechanism is that a trailing sentence after the C3 guidance disrupts the model's tendency to re-derive "write file 1" by providing any additional contextual anchor (even semantically irrelevant content), the same advance rate would be observed — and the amendment would be shipping for the wrong reason, possibly with brittleness properties different from what the spike measured.

This matters for the amendment's robustness claims. If advance rate is driven by remaining-work content, the anchor wording matters and can be tuned. If it is driven by trailing-token perturbation, any wording produces the same rate and the amendment is actually more robust than the hypothesis claims — but also potentially achievable with a structurally simpler change. The spike cannot distinguish these with its current design.

**Recommended design change:** Add a content-neutral control arm to ρ.2 (or as a ρ.3 arm on one base, preferably B2, which is the rung-1 base so the comparison is most legible). The control anchor would be a trailing sentence of comparable length and format that contains no remaining-work information — e.g., "The session task contains multiple deliverables that should be produced in the order specified." or a paraphrase of the delegation standard ("Your role is to delegate file production to the appropriate ensemble."). This arm is n=10 on one base, adds 10 calls at $0 local, and can be run alongside ρ.2 with the same tooling.

If the control arm's advance rate is comparable to ρ.2's production-anchor advance rate, the mechanism is trailing-perturbation-driven and the amendment should be characterized accordingly. If the control arm's advance rate is materially lower (say, control ≤ 3/10 vs. production ≥ 7/10), the remaining-work content is doing the causal work and the hypothesis is validated at the mechanism level, not just the rate level. Either result is informative and neither adds significant complexity to the harness.

If a full control arm is out of scope, the minimum mitigation is to flag this as a named limitation in the ADR amendment: "The ρ spike establishes that routing the judge's remaining-work statement produces advance rate ≥ 7/10 but does not isolate whether remaining-work content or trailing-token presence is the causal mechanism. The amendment is warranted by the rate evidence; the mechanism isolation is a named follow-on."

---

### P2 — Weaknesses that bound the claims

**P2-A: The ρ.2 denominator is conditional on ρ.1's verdict distribution, creating a sample-size problem the pre-registered thresholds do not account for.**

The design specifies: "Denominator n (a verdict-COMPLETE or no-statement trial contributes `none` to ρ.2 — it produced no usable anchor)."

This means ρ.2 uses a full-n denominator of 10 per base, where some trials contribute `none` because ρ.1 produced a verdict-COMPLETE (false-stop) or a no-statement result. If H-ρ.1 holds (names-correct ≥ 8/10, verdict-COMPLETE ≤ 1/10), then at most 1-2 trials per base contribute `none` to ρ.2, and the effective denominator is 8-9. The advance ≥ 7/10 threshold means 7 out of 10 trials must advance — including the `none` trials.

This is the right denominator choice (the θ-audit's P1-B precedent: always n, not n-parseable). The problem is not the denominator but the threshold calibration: at n=10 with 1-2 `none` trials, ρ.2 needs 7 `advance` outcomes from at most 8-9 trials where a tool call was even made. That means ρ.2 requires the anchored call 2 to advance on 7/8 or 7/9 trials where it ran — a hidden 87.5-78% underlying success rate behind a stated 7/10 threshold. The [0.5, 0.7) Conditional-Acceptance band is also affected: a 6/10 advance rate with 2 `none` trials means 6/8 = 75% underlying success, which is above the 7/10 pass threshold's implied 87.5%.

The design does not flag this interaction between ρ.1's false-stop rate and ρ.2's effective rate. If verdict-COMPLETE comes in at 2/10 (one over the ρ.1 threshold), ρ.2 has 8 runnable trials; advance ≥ 7/10 requires 7/8 = 87.5% from those trials. The pre-registered bands assume the denominator is 10 with approximately 10 anchored runs.

**Recommended design change:** In the pre-registration and in the harness, track and report the `runnable-trial count` (trials where ρ.1 produced REMAINING with a usable statement) separately from n. The pre-registered thresholds fire on n=10; the runnable rate (advance / runnable-trials) is reported as a characterization statistic alongside the n-denominator rate, for the same reason θ's P1-B separated parseable-only rate from the n-denominator rate. Flag the interaction in the scope section: "If verdict-COMPLETE comes in above 1/10, ρ.2 interpretation requires the runnable-trial rate alongside the n-denominator rate."

---

**P2-B: The [0.5, 0.7) Conditional-Acceptance band is not motivated by reference to a specific risk model or prior measurement.**

The decision rule states: if ρ.1 passes but ρ.2's advance is in [0.5, 0.7), the amendment proceeds as Conditional Acceptance with the real-OpenCode multi-file convergence run as the discharge gate.

The 0.5 lower bound is not anchored to any measured baseline. The A_current arm in the rung-1 probe produced 0/10 advance. The "no anchor" baseline is 0. The Conditional-Acceptance band of [0.5, 0.7) does not relate to any measured prior that would justify treating 5-6/10 advance as "good enough to attempt discharge via acceptance run" rather than "below the threshold that warrants redesign."

The 0.7 upper bound is the pass bar, established by analogy with the rung-1 probe's 8/10 result. But the motivation for 7 rather than 8 as the pass threshold is that ρ.2 uses the judge's real statement rather than a hardcoded optimal string — a reasonable expectation of degradation. This reasoning is not stated in the design.

The [0.5, 0.7) band is operationally important: it determines whether a middling result routes to Conditional Acceptance (amendment ships with a discharge gate) or refutation (loop-back re-opens). Without a risk-calibrated motivation for the 0.5 lower bound, the band risks being captured by motivated reasoning post-run: a 5/10 result could be read as "in the band, proceed conditionally" when it is actually closer to the no-anchor baseline (0/10) than to the pass threshold (7/10).

**Recommended design change:** Anchor the [0.5, 0.7) band to a specific justification in the pre-registration. Options: (a) the band represents advance rates where the real-OpenCode acceptance run is expected to provide discriminating signal (i.e., the mechanism may work in the production path even if the spike's isolated composition underestimates it); or (b) the band represents rates that are an improvement over A_current (0/10) but insufficient for single-spike adoption, requiring discharge confirmation. Either justification should appear in the pre-registration verbatim. Additionally, if ρ.2 advance < 0.5 AND ρ.1 passes (good anchor naming, poor advance), the design should name a separate diagnostic: the anchor is trusted but call 2 is not using it, which is a different failure mode than "anchor is poisoned."

---

**P2-C: The pre-registered thresholds at n=10 do not adequately account for the statistical precision available at this sample size.**

The pass thresholds (ρ.1 names-correct ≥ 8/10, ρ.2 advance ≥ 7/10) are sharp cutoffs at a sample size where the confidence intervals are wide. At n=10, Clopper-Pearson 95% CI for 7/10 is approximately [0.35, 0.93]; for 8/10 it is [0.44, 0.97]. Two thresholds set 1/10 apart (7/10 pass vs. 6/10 fail, or 8/10 pass vs. 7/10 pass-ρ.1-only) are not statistically distinguishable from each other at this n.

This is not a reason to abandon the thresholds — θ operated at the same n and the methodology is calibrated for this corpus's cost-conscious $0 local design. But the design should not present pass/fail as a clean binary. The ADR amendment narrative should describe pass as "consistent with ≥ 7/10 true advance rate" not "advance is ≥ 70%"; and it should carry forward the θ-audit's P2-C recommendation (explicitly labeling composed-estimate rates and acknowledging the product-of-n=10 arms imprecision).

The spike design does not include this labeling discipline explicitly, though the "composed read" section gestures at it ("the production end-to-end advance rate is reported as the observed ρ.2 advance over n — it already folds Factor 1 in"). The composed estimate is ρ.1 × ρ.2 in expectation, but the design reports ρ.2's observed rate as the production rate directly — which is reasonable because ρ.2 uses the judge's actual output (not a hardcoded string). But the n=10 precision caveat should be stated explicitly in the pre-registration for the same reason θ's P2-C required it.

**Recommended design change:** Add one sentence to the "composed read" section: "At n=10 the pass/fail boundary carries wide confidence intervals (e.g., 7/10 pass is consistent with true rates from 0.35 to 0.93 at 95% CI); the decision rule is a structured evidence threshold, not a precision rate estimate. The ADR amendment will label ρ.2's observed rate as an n=10 measurement rather than a production rate estimate." No additional runs required.

---

### P3 — Improvements

**P3-A: The multi-base discipline is correct, but the B3 base construction should be more explicitly documented.**

The design specifies B3 as a "3-deliverable task (module + a second module + the test), two files written (two trailing tool pairs)." This is the appropriate θ-analog (the E4′ base: `string_utils.py + number_utils.py + test_string_utils.py`, two of three written). The construction discipline carries forward from θ, which is correct.

However, the B3 base construction note is significantly less explicit than θ's E4′ construction note. Spike θ specified the exact three-deliverable task text and the exact tail structure (which two of three files were written, in what order). Spike ρ's B3 specification leaves "module + a second module + the test" unspecified in the pre-registration. At run time, the harness author will choose a specific task text and specific tail structure; if that choice is not pre-registered, the B3 base is not fully pre-registered and the construction is a post-hoc design decision.

**Recommended design change:** Pin B3's task text and tail structure in the pre-registration before running. Recommended: use θ's E4′ base verbatim (string_utils.py + number_utils.py + test_string_utils.py, first two written, test outstanding) for continuity with the prior measurement corpus. Record the pin as a pre-registration note.

---

**P3-B: The no-tool-call trials in ρ.2 should be characterized by response content, not just counted.**

The design counts no-tool-call responses and gates on ≤ 2/10. The rung-1 probe noted these were "NOT inline-write collapses" but did not characterize their content further. At n=10 per base, 2-3 no-tool-call trials contain information about whether the anchor is producing a stuck-text response (model describes what it would do without calling a tool), a premature-finish response (model declares the task complete), or a refusal/confusion response (model does not understand the anchor format).

These failure modes have different remediation paths: stuck-text suggests the anchor needs an imperative (see the incongruity note above about the rung-1 anchor including "Produce it next"); premature-finish suggests the anchor is overloading the model's completeness assessment; confusion suggests anchor wording needs reformulation. Counting all three as `none` discards this diagnostic signal.

**Recommended design change:** Add a sub-classification for no-tool-call trials in ρ.2: `none-finish` (response declares completion or summarizes as complete), `none-text` (response describes next steps without a tool call), `none-other` (other). Full response text is already being retained per trial; the sub-classification adds ~15 minutes of annotation work per base and materially improves the diagnostic value of a ρ.2 fail or borderline result.

---

## Overall Verdict

**Run with amendments — two P1 findings require design changes before running.**

**P1-A** (names-correct adjudication validity) must be addressed before running ρ.1: the binary classification admits confirmation-direction drift on borderline cases that cannot be detected post-hoc. The two-level sufficiency standard (specific-correct / description-correct / ambiguous / names-wrong) is a pre-run adjudication discipline, not a post-hoc fix. This requires no additional calls, only a pre-registration amendment.

**P1-B** (absence of content-neutral control arm) is the most structurally important finding. The spike's causal claim — that remaining-work content anchors call 2 — cannot be isolated from the alternative (any trailing-text perturbation disrupts the stuck pattern). A 10-call control arm on B2 at $0 local resolves this. If a full arm is out of scope, the minimum is a named limitation in the pre-registration and the ADR, with the mechanism-isolation flagged as a named follow-on probe. Without either, the ADR amendment credit "routing the remaining-work signal forward fixes multi-file progress" carries the risk of being right for the wrong reason.

**P2-A** (conditional denominator interaction) requires a pre-registration note and harness tracking of the runnable-trial count; no design change.

**P2-B** (Conditional-Acceptance band motivation) requires a one-sentence justification in the pre-registration before running, so the band interpretation cannot drift post-result.

**P2-C** (n=10 precision labeling) is the θ-audit P2-C discipline carried forward; requires a one-sentence addition to the composed-read section.

**P3-A** (B3 construction pin) and **P3-B** (no-tool-call sub-classification) are cheap to apply and improve the diagnostic value of the results.

The spike's factor-decomposition design (ρ.1 × ρ.2, ρ.2 using the judge's real statement to fold Factor 1 in) is analytically sound and is the right design for the production-faithful measurement. The multi-base discipline, the fidelity commitment to the landed code path, and the full-n denominator discipline all meet the θ calibration bar. The core gap is mechanism isolation (P1-B), which is addressable at $0 and 10 additional calls.

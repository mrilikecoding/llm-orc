# Research Methods Review — Spike σ (Premature Finish, Finding I)

**Reviewed artifact:** `docs/agentic-serving/essays/research-logs/cycle-7-spike-sigma-premature-finish.md` — §PRE-REGISTRATION
**Constraint-removal response included:** n/a (mechanism-fix spike; no ADR-082 constraint-removal artifact in scope)
**Date:** 2026-06-09
**Reviewer role:** research-methods (ADR-060 criteria 1–4, ρ-audit calibration bar)

---

## Summary

- **Arms reviewed:** BASELINE (current code, no retry) + F-σ.1 (REMAINING-retry, R=1), n=8/arm
- **Primary outcome:** session completion (binary per run)
- **Flags raised:** 6 (2 P1, 3 P2, 1 P3)
- **Criteria applied:** 1–4 (ADR-082)

---

## Findings

### P1 — Design flaws that would invalidate conclusions before run

---

**P1-A: The ≥0.8 grounded threshold is not achievable with integer counts at n=8, making the stated decision boundary inoperable.**

The pre-registration declares GROUNDED when F-σ.1 session-completion rate ≥ 0.8. With n=8 runs, the only observable rates are multiples of 1/8: 0/8=0.000, 1/8=0.125, ..., 6/8=0.750, 7/8=0.875, 8/8=1.000. The threshold of 0.8 falls in the gap between 6/8=0.750 and 7/8=0.875 — neither value satisfies "≥ 0.8" unless the threshold is implicitly rounded. This means the effective grounded threshold at n=8 is ≥7/8=0.875, which is meaningfully stricter than the stated 0.8.

This matters because of the binomial distribution at that true rate. If the retry fix genuinely drives the session-completion rate to 0.80, the probability of observing ≥7 completions out of 8 is only 0.503. So even if the fix works exactly as hypothesized, the spike declares it INSUFFICIENT half the time. If the true rate is 0.85 (the rate the session-level model predicts given independent per-turn retry), P(≥7/8) is 0.657 — still a 34% chance of a false INSUFFICIENT verdict. The decision rule is sensitive to a single flip of one run: 6/8 routes to INSUFFICIENT, 7/8 routes to GROUNDED, and these two outcomes are separated by the variance of a single Bernoulli trial.

The pre-registration also does not define what happens when the observed fix rate is 6/8=0.750. This outcome satisfies "baseline ≤ 0.5" (if baseline confirms around 3/8) and shows a large lift from baseline — but the stated grounded criterion says INSUFFICIENT. There is no band or conditional path for 6/8 equivalent to ρ's Conditional-Acceptance band. The decision tree has a gap.

**Recommended design change (must address before run):** Choose one of:
- (a) Lower the grounded threshold to ≥0.75 (6/8), with explicit acknowledgement that this means a 5-REMAINING-turn session could still fail 25% of the time; or
- (b) Raise n to 10, which makes 0.80 achievable as 8/10=0.80; or  
- (c) Keep the ≥0.8 threshold (effective: ≥7/8) but add an explicit intermediate band: F-σ.1 in [6/8, 7/8) is "Conditional Grounded — sufficient for F-σ.2 comparison and production-config discharge gate, not sufficient for standalone adoption," motivated by reference to the session model and the baseline contrast. Document the band reasoning explicitly.

Option (c) is preferred: it preserves the spirit of the threshold while making the decision tree complete.

---

**P1-B: The hypothesis that the stall is "transient/sampling-driven" is assumed without being measured, and the spike design cannot distinguish it from a deterministic state-conditional refusal — which would make retry worthless.**

The pre-registration states: "a retry samples differently and is expected to recover most stalls." This is the mechanism claim. It rests on the assumption that the no-tool-call is a sampling artifact: the model happened to generate a text response on this draw, and at temperature ~0.8, a second draw will produce a tool call. That assumption is reasonable as a prior — and it is the right fix to try first. But the spike design provides no measurement to confirm or refute it.

The alternative hypothesis is that the no-tool-call on a REMAINING turn is state-conditional: when the conversation context is in a particular configuration (a specific anchor phrasing, a specific partial-work state, or a specific accumulated message length), the seat-filler consistently refuses to emit a tool call regardless of temperature. At temperature ~0.8 with qwen3:14b, this would mean the second sample is drawn from the same distribution and produces another no-tool-call with the same probability as the first — so R=1 retry recovers ~80% of stalls if they are independent, but recovers near-zero if they are state-conditional.

The critical difference: if stalls are transient, the retry-recovery count (secondary measure) will be high and session completion will lift. If stalls are deterministic, the retry-recovery count will be near-zero, session completion will not lift, and the spike correctly routes to F-σ.2 or a structural alternative. The spike's secondary measure (retry-recovery count) is designed to catch this — but only post-hoc, and only if F-σ.1 fails. The pre-registration should describe this as a measured hypothesis, not a background assumption.

More importantly: if stalls are state-conditional, the retried call will use the exact same `_seat_filler_messages` composition with no change to the context (same anchor, same conversation tail). The retry is literally the same call twice. The only thing that changes is the random sample. This is fine if the failure is sampling variance, but it should be stated explicitly as the mechanism being tested, not embedded as a background rationale.

There is also a third possibility the pre-registration does not consider: the no-tool-call is driven by a specific failure in the anchor composition on certain conversation states (e.g., the anchor text is long enough to push the message past a context window boundary, or the ADR-038 remaining-anchor format is being ignored when the model has already "decided" to finish). This would make F-σ.1 a weaker fix than F-σ.2 for a reason the retry-recovery count alone cannot diagnose.

**Recommended design change (must address before run):** Add a pre-registered characterization protocol for no-tool-call trials. When a REMAINING-arm run ends prematurely (stall is not recovered), retain and classify the no-tool-call response content before scoring the session as INCOMPLETE:
- *Stuck-text*: model produces text describing what it would do or summarizing remaining work
- *False-complete*: model declares the task finished with natural-language text
- *Context-refusal*: model produces a response inconsistent with the conversation state (confusion, repeated content, truncated output suggesting context overflow)

This is not an additional run. It is an annotation pass on already-retained serve.log output. It provides the only evidence that can distinguish transient-sampling from state-conditional refusal post-run, and it avoids a situation where the spike ends with INSUFFICIENT and no signal about which structural alternative to pursue.

Additionally, state explicitly in the pre-registration: "F-σ.1 tests the hypothesis that the no-tool-call is transient (temperature-sampling noise). The retry dispatches the identical `_seat_filler_messages` call a second time with no context change; the only difference between call 1 and call 2 is the sampled token sequence. If retry-recovery rate is high, the stall is confirmed as sampling-driven. If retry-recovery rate is near-zero, the stall is state-conditional and F-σ.2 (forceful-imperative change to the prompt composition) is the correct escalation path."

---

### P2 — Weaknesses that bound the claims

---

**P2-A: n=8 provides weak power to confirm the grounded threshold and zero formal stopping rule, yet the design acknowledges neither.**

At n=8 per arm and a true fix rate of 0.80, the probability of observing the grounded result (≥7/8 completions) is only 0.503. At a more conservative true rate of 0.75, it is 0.367. The spike will falsely declare INSUFFICIENT roughly half the time even if the fix genuinely achieves an 0.8 session-completion rate. This is not disqualifying — the corpus operates at $0 local and calibrated n=10 designs throughout — but the pre-registration should state it.

The cost argument for n=8 is valid (~5 min/run × 16 runs = ~80 min total). A sequential design with an early-stop rule is feasible. For example: if the fix arm reaches 6 completions in the first 6 runs, stop early (implied rate ≥1.0); if the fix arm reaches 0 completions in the first 5 runs, stop early (implied rate ≤0.0, clearly INSUFFICIENT). These are crude but cheap and appropriate for a $0 local corpus. The pre-registration notes "[methods-reviewer: confirm n adequacy to distinguish the rates; flag if a sequential/early-stop design is better given the per-run cost]" — this is a correct self-flag. The answer is: n=8 at ~0.5 power to confirm the stated threshold. Either add a 2-run safety buffer (n=10) or adopt a simple early-stop rule.

The one structural asset at n=8 is that the arm comparison is the main signal, not the arm-in-isolation rate. At the expected rates (baseline ≈3/8 completions, fix ≈6–7/8), the probability that the baseline arm happens to equal or exceed the fix arm by chance is approximately 0.069. That cross-arm comparison is the most legible evidence the spike produces, and it should be named as the primary inference (not the within-arm rate alone).

**Recommended design change:** Add one of:
- (a) A simple early-stop rule: run 4 fix-arm sessions first; if all 4 complete, the fix is clearly grounding and only the baseline arm needs to finish for the comparison; if 0/4 complete, the fix is insufficient and baseline need not run.
- (b) Bump to n=10, which costs 2 additional runs per arm (~20 min), makes ≥0.8 achievable as 8/10=0.80, and aligns with the ρ/ξ calibration bar.
- (c) Retain n=8 but add a note: "At n=8, observed rate ≥7/8 is consistent with a true rate in [0.50, 1.0] (95% Wilson CI = [0.47, 0.98]). The cross-arm comparison is the primary inference; the within-arm rate is a characterization estimate, not a precision rate."

---

**P2-B: "Retry-recovery count must explain the lift" is stated as a condition but is not operationalized — making it post-hoc adjudicable in either direction.**

The pre-registration says: "The retry-recovery count must explain the lift (mechanism, not noise)." This is the right instinct but it is incomplete as a measurement plan. What count of retries-that-recovered-to-a-tool-call is "sufficient explanation"? If the fix arm produces 7/8 session completions but the retry-recovery log shows retries fired only 2 times across all runs and both recovered, is the mechanism confirmed? (Yes — high recovery efficiency.) If retries fired 6 times and only 3 recovered, does that explain a 5/8 completion lift? (Unclear without a baseline rate.)

Without a pre-registered threshold or reference range, "must explain the lift" is an ex-post gate that can be satisfied by any positive retry-recovery count when the spirit is that the mechanism should be the primary driver. This is the same adjudication-drift concern as ρ's P1-A (names-correct classification).

**Recommended design change:** Operationalize the mechanism gate before running. Suggested form: "The mechanism is confirmed if retry-recovery rate (retries-that-resulted-in-a-tool-call / total-retries-fired) is ≥ 0.6 AND the count of retries-that-recovered equals or exceeds the session-completion improvement (fix completions − baseline completions). If retries fired but did not recover (retry-recovery rate < 0.4), the mechanism is not confirmed by retry — escalate to F-σ.2 regardless of the session-completion rate." This is a two-sentence addition to the pre-registration. No additional runs.

---

**P2-C: The coder=14b simplification is sound as an isolation, but its effect on session-completion rate is not bounded — if the 14b coder is substantially more reliable than 8b, the baseline rate may be higher than the pre-registered ~0.40, making the baseline arm uninformative about production.**

The pre-registration correctly notes that the no-tool-call is on the seat-filler's action call, not the coder's generation call, so coder model identity is orthogonal to whether the seat-filler emits a tool call. This is a valid isolation argument for the mechanism. The concern is a different one: baseline session completion is not purely a function of the no-tool-call stall rate. A session can also fail if the coder generates broken code that causes a downstream failure requiring correction, or if a downstream file is syntactically valid but structurally wrong in a way the judge cannot distinguish from COMPLETE. With a 14b coder, the file quality may be higher and these secondary failure modes less frequent — so the baseline arm's observed completion rate may be higher than ~0.40, not because the stall rate changed, but because sessions that would have failed for other reasons (under 8b) now complete. This would compress the fix-vs-baseline gap and reduce the spike's discriminating power.

The pre-registration acknowledges "coder=14b is NOT production (8b)" in the honest scope section. But it does not bound this effect on the baseline estimate. The ~0.40 baseline is derived from "~2/10 no-tool-call rate per REMAINING turn, 4 REMAINING turns in the 5-file task" — this is entirely a stall-rate model. It does not include the coder-quality secondary failures.

**Recommended design change:** Add one sentence to the honest scope: "The ~0.40 baseline estimate assumes session-completion failures are dominated by the stall failure mode. If the 14b coder reduces non-stall failures relative to the 8b coder in production, the observed baseline rate may exceed 0.40; a baseline rate above 0.50 with the current code would narrow the fix-vs-baseline gap and reduce the spike's discriminating power. If baseline arm exceeds 0.50, this should be flagged in the results as a confound requiring re-confirmation at the production-config discharge gate." No design change required, just explicit scope documentation.

---

### P3 — Improvements

---

**P3-A: The BASELINE arm runs current code — there is no diagnostic of which failure mode each INCOMPLETE session reflects.**

The baseline arm is designed to confirm the discharge finding: ~60% premature finish on 5-file sessions. That confirmation is important (the spike should open by verifying the symptom, not assuming it), and the design does this correctly. But the baseline arm does not pre-register how to classify each INCOMPLETE run. With n=8 baseline runs, some may fail because of the no-tool-call stall; others may fail for different reasons that are not visible in the pre-registration (e.g., a coder failure that produces invalid output the judge marks REMAINING on an already-completed file, consuming a turn; or the AS-3 cap firing due to compounding turns). Counting all INCOMPLETE runs as "premature finish due to stall" may overcount the stall rate if any INCOMPLETE run is attributable to a different cause.

The discharge data (2/2 runs) is consistent with 100% stall-driven failures because both runs were observed turn-by-turn. With n=8 baseline runs, the failure mode distribution may be messier.

**Recommended design change:** Pre-register a BASELINE session classification protocol:
- *Stall-confirmed*: turn log shows REMAINING verdict → no-tool-call action → FinishTurn at a turn where deliverables were missing
- *Non-stall incomplete*: session ended INCOMPLETE for another reason (AS-3 cap, coder failure producing a loop, etc.)

Report stall-confirmed count separately. The baseline arm should confirm stall-confirmed ≥ 4/8 (i.e., the stall is the dominant failure mode in this config) before the fix-arm comparison is interpreted as stall-specific evidence. This is an annotation pass on the serve.log, which is already pre-registered as retained.

---

## Per-Criterion Assessment

### Criterion 1: Belief-Mapping

The central framing — "a bounded retry of the action call recovers the stall" — is productive for this failure mode, but it forecloses a deeper question: is the stall a property of the retry surface (temperature sampling) or a property of the conversation state composition? A different productive question would be: "What would I need to believe for the stall to be a conversation-state problem rather than a sampling problem?" The design's answer is implicit: we would see near-zero retry-recovery rates and the stall would be state-conditional. The spike's secondary measure (retry-recovery count) does surface this — but only if the pre-registration operationalizes it as a falsifiable claim (P1-B).

The choice to commit to retry as the fix shape before characterizing the failure more is defensible on cost grounds (a retry is $0 and ~2 lines of code; re-characterization costs sessions). It is not premature narrowing given that the discharge data showed both stalls were transient-looking (the judge correctly identified REMAINING; the seat-filler just failed to produce a tool call). The retry framing is reasonable as a first fix. The concern is that the failure-characterization gap (P1-B) means a failed spike produces no diagnostic signal about why — leaving the escalation path (F-σ.2, structural alternative) uninformed.

### Criterion 2: Embedded Conclusions

The hypothesis contains one embedded conclusion that should be made explicit rather than assumed:

"A seat-filler no-tool-call on a `REMAINING` turn is incoherent (the framework already knows work remains)."

This is true from the framework's perspective, but it assumes the seat-filler experiences the REMAINING branch as a coherent context that should produce a tool call. The actual model may be receiving a message whose accumulated content pattern triggers a "I should summarize and stop" behavior that is locally coherent from the model's token-prediction perspective. The fix (retry) is correct regardless, but framing the no-tool-call as "incoherent" rather than "contextually motivated" may cause the researcher to under-weight the probability of a state-conditional stall (P1-B).

Suggested reframing: "A seat-filler no-tool-call on a `REMAINING` turn is a framework-level error — the driver has a REMAINING verdict and needs a tool call. Whether the model's behavior is sampling noise or a contextual tendency, the fix is the same (retry). The spike distinguishes these two causes via the retry-recovery rate."

### Criterion 3: Premature Narrowing / Prior-Art Treatment

The design treats ADR-038 / Spike ρ honestly as prior art: it names the refuted backstop assumption explicitly ("the ~2/10 no-tool-call rate is backstopped by the next re-judgment + the AS-3 cap") and identifies precisely why it fails under the real client ("a finish ENDS the loop"). This is correct prior-art treatment. No premature-narrowing concern in the design scope.

The one narrowing concern: the design jumps directly to a fix design (retry) without a diagnostic arm that would characterize the failure mode first. The discharge data supports this jump — 2 observations from 2 runs with a clear mechanism — but the spike should note that F-σ.1 is a mechanism-test as well as a fix-test (P1-B addresses this).

### Criterion 4: Incongruity Surfacing

The pre-registration does not surface an incongruity that is visible in the codebase context. The existing `_REMAINING_IMPERATIVE = "Produce that next."` string (already in production from Spike ρ) is a framework-level intervention that addresses the same "model doesn't know what to do next on a REMAINING turn" problem that Finding I represents — but at the prompt-composition layer rather than the retry layer. Spike ρ found that the imperative "removed the lone stuck/no-tool-call cases" in isolated single-decision tests. Under the real client, it did not eliminate the no-tool-call entirely (Finding I still occurred). The simplest-adjacent solution — strengthen the existing imperative rather than add a retry — is F-σ.2, which the design correctly holds as a fallback. The incongruity is not a design flaw but is worth naming: F-σ.1 (retry) adds a new mechanism; F-σ.2 (stronger imperative) is an adjustment to the existing mechanism. If F-σ.1 fails and F-σ.2 is tested, the spike's comparative structure will tell us something about which layer (execution policy vs. prompt composition) owns the stall failure.

---

## Overall Verdict

**Run with amendments — two P1 findings require design changes before running.**

**P1-A** (threshold gap at n=8) is a structural flaw: the ≥0.8 grounded threshold is not achievable with integer counts at n=8, which means the effective threshold is ≥7/8=0.875, and the decision tree has a gap for the 6/8=0.750 case. This requires either lowering the threshold to 0.75, bumping n to 10, or adding an explicit Conditional-Grounded band for 6/8. The fix is a pre-registration amendment — no additional runs. Recommended: option (c) — define the band explicitly, document the reasoning.

**P1-B** (uncharacterized failure mode) is the mechanistically important finding. The retry hypothesis rests on the stall being transient-sampling noise; if the stall is state-conditional, R=1 retry recovers near-zero stalls and the spike ends with INSUFFICIENT and no signal about what to do next. Adding a no-tool-call response-content classification protocol (stuck-text / false-complete / context-refusal) costs zero runs and converts a potentially uninformative INSUFFICIENT result into actionable diagnostic signal for the structural-alternative route. This is the ρ-audit P3-B concern elevated to P1 here because the escalation path is higher-stakes (DECIDE-loop-back vs. BUILD) and the spike has no other failure-mode signal.

**P2-A** (n=8 power) should be addressed by adding a simple early-stop rule or bumping to n=10. The 0.50 power at the stated grounded threshold is the central statistical concern.

**P2-B** (mechanism gate operationalization) requires one paragraph before running.

**P2-C** (coder confound scope note) requires one sentence in the honest scope section.

**P3-A** (baseline classification protocol) is a cheap annotation discipline that prevents over-attributing all INCOMPLETE baseline runs to the stall failure mode.

The core design is sound: live-multi-turn primary is the right choice for this failure mode; the BASELINE+F-σ.1 arm structure with cross-arm comparison is correct; the session-level binary outcome is the right primary measure; and the honest scope treatment (coder=14b, n=8 characterizes but doesn't prove) is appropriate. The pre-registration's self-flags are accurate. The amendments above tighten the decision boundary, operationalize the mechanism check, and ensure the spike produces actionable signal even on a null result.

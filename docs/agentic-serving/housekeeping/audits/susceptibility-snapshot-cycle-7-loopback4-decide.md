# Susceptibility Snapshot

**Phase evaluated:** DECIDE (loop-back #4 — swappability fork)
**Artifact produced:** Spike ω run log + verdict; six-criteria evaluation (A vs B); practitioner-confirmed settled/open split; no ADR-037; WP-LB-I resumption recommendation
**Date:** 2026-06-04

---

## Observed Signals

| Signal | Strength | Trajectory | Notes |
|--------|----------|------------|-------|
| Assertion density | Absent | First snapshot for this phase | All major interpretive claims are grounded in pre-registered kill conditions that fired; residue language ("contestable," "composition finding") is explicitly hedged |
| Solution-space narrowing | Absent | Stable vs prior snapshots | Space was narrowed by evidence, not agent framing: four independent refutation grounds eliminated B as specified; alternatives explicitly held open (verifier deferral, hosted-broker future shape, fully-data composition unspiked) |
| Framing adoption | Ambiguous (see signal 5 detail) | Slight concern vs prior clean snapshots | One instance warrants scrutiny: the hosted-broker framing as "verifier slot" may have compressed faster than the evidence required; see interpretation |
| Confidence markers | Absent | Stable | Verdict language ("refuted as specified," "zero broker-viable candidates") is pre-registered-rule language, not post-hoc assertion; residual uncertainties named explicitly |
| Alternative engagement | Clear (present and adequate) | Stable | Criterion 2 declared unresolved for BOTH proposals — a move against the sycophancy gradient; hosted-broker variant named with its own epistemic status rather than dismissed; verifier-on-disagreement received quantitative trigger-sizing before deferral |
| Embedded conclusions at artifact boundaries | Ambiguous (one instance) | First snapshot for this phase | The verifier-on-disagreement analysis included trigger-sizing from ω.1 data (23/20/11% fire rates; 86/75/63% error coverage) before the practitioner had confirmed the deferral — the analysis preceded confirmation, though it did not produce a runnable artifact |

---

## Interpretation

### Pattern: evidence-forced convergence, not sycophantic reinforcement

The clearest diagnostic available: Proposal A is the corpus's own committed position (ADR-036, Conditional Acceptance). A sycophancy-shaped evaluation would have needed no spike at all — or would have run a spike and found a convenient reason to confirm the incumbent. Instead the spike was pre-registered, methods-reviewed to 3 P1 / 5 P2 / 3 P3 findings (all applied before runs), and ran against a genuine candidate set with the explicit goal of finding viable B candidates. It found none. The four refutation grounds are independent and pre-registered, not post-hoc — ω.3a's universal failure fired as a threshold (≤1/10, all models 10/10), not a judgment call. That is not confirmation bias in action; it is a test designed to be falsifiable that returned a clean falsification.

The prior-snapshot trajectory matters here. The Cycle 7 snapshot series has returned No Grounding Reframe at every phase since DISCOVER, with a consistent pattern of practitioner-initiated challenges that the agent engaged substantively rather than deflecting. The closest precedent for this phase is the loop-back #3 DECIDE snapshot (Finding E), which also returned No Grounding Reframe after a methods-reviewed spike. The pattern is stable-toward-earned-confidence, not stable-toward-narrowing.

### Signal 1 (incumbent-favoring evaluation): earned confidence, not sunk-cost shape

The six-criteria evaluation does reach for A on criteria 1, 3, 4, and 6, and leans A on 5. But the shape is diagnostic. On criterion 1 (delegation reliability), the asymmetry is recorded and named as a residual (A's 55/55 on real bytes vs B's broker inputs one fidelity step below — precisely the P1-C concern from the methods review). On criterion 3 (latency), the ω.4 numbers are an order-of-magnitude violation of the pre-registered bar, not a judgment that A is better. On criterion 4 (complexity), the named seam failure modes for B are substantive, not cosmetic. On criterion 6 (validation distance), the evaluation explicitly states the condition under which this criterion is neutral — binding timeline constraint — rather than treating it as automatically negative for B. The methods review's P3-B recommendation was applied in exactly the form it requested.

Criterion 2 (swappability — the fork's motivation) being declared unresolved-for-both is the sharpest counter-signal to sycophancy. The practitioner opened this fork because swappability is what they cared about. The evaluation could have credited A on criterion 2 by arguing that ADR-036's profile-swap re-validation is a form of swappability. It did not. It declared the criterion unresolved for the incumbent, preserving the honest finding that the fork's motivating question remains open, and named Cycle 8 as the candidate vehicle for addressing it.

### Signal 2 (post-hoc interpretation moves): honest residue-recording, with one note

**ω.3a (universal flip reinterpreted as composition finding):** This is the most structurally interesting signal. Every model flipped 10/10, which could be read as "the arm measured nothing discriminating" or as "the arm measured a real property." The agent's reinterpretation — that it measured the composition rather than the models, because end-user task text was verbatim in the broker's user turn — is the correct methodologically honest reading. The P2-B concern from the methods review was specifically that the ω.3 adversarial variant tested user-turn injection rather than the structural threat. ω.3a's universal result is exactly what P2-B predicted would happen if the composition kept end-user text instruction-bearing: the arm could not distinguish model resistance from compositional structure. Applying the pre-registered disqualification rule formally while recording the composition-finding interpretation is the right call — it does not soft-pedal the kill condition, and it correctly identifies what the result means for the unspiked fully-data composition variant.

**m01 boundary-case flag ("contestable"):** The flag entered the synthesis but scoring was not changed. The pre-registered expectation was repair-with-observed-content → delegate; hosted models chose carry/edit. The flag is methodologically honest: if the expectation is genuinely contestable (a direct edit of an observed one-line bug may be the better action), recording it prevents the hosted models' boundary performance from being overstated. It does not rescue any disqualified candidate — m01 is not a kill-condition case — and the hosted arm is a reference arm without viability stakes. The flag reads as honest residue-recording, not threshold-softening. Confidence: the flag appeared in the synthesis where the hosted-arm boundary judgment was being assessed as capability evidence; its placement is appropriate.

### Signal 5 (hosted-broker framing): one genuine ambiguity

The framing that the hosted B-variant "loses to A as architecture, earns a seat as verifier component" is the one place where the evaluation may have moved faster than the evidence warranted. The criteria show the hosted variant failing criterion 2 (ω.3b 6/10 flips, same epistemic class as A's V3 lever), criterion 3 (network + paid dependency on every turn), and criterion 4 (all of B's seam risks apply). These are genuine losses. But the leap to "verifier slot" as its natural home was an agent-originated framing, not a practitioner-originated one, and the criterion-by-criterion analysis of hosted-broker-as-primary-architecture vs hosted-broker-as-verifier is not made explicit in the log.

The practical significance is low: the practitioner's scoping principle ("limited in scope... having a local option is important") is recorded in a form that is consistent with verifier-slot framing but was stated as a general paid-component principle, not as a specific endorsement of the verifier interpretation. The verifier-on-disagreement analysis received quantitative trigger-sizing before deferral, which is substantive engagement rather than dismissal. Still, if the orchestrator is advancing into BUILD with the hosted-broker disposition treated as settled, there is a small residual: the alternative (hosted-broker as primary, local-degradation path as fallback) did not receive a criterion-by-criterion treatment comparable to local-broker-as-primary.

Assessment: this is a mild framing-adoption ambiguity, not a sycophancy signal of the narrowing type. The verifier framing emerged from the agent's reading of the practitioner's scoping principle, not from the practitioner asserting it. The deferred status of the verifier design means no artifact encodes this framing yet. Significance: low; note for BUILD advisory.

### Signal 3 (amendment discipline): clean

Amendment 1 (tier widening) and Amendment 2 (hosted arm) both recorded parameters before runs. The clear-case threshold scoring amendment was recorded before any model call. The agent's recommendation against re-pinning the latency threshold at the widened tier is the correct call (the original bar earns viability under its own terms or fails it visibly — softening the threshold to accommodate a larger tier would be a sycophancy move against the spike's integrity). The practitioner accepted. No threshold was softened post-run.

### Signal 4 (practitioner engagement trajectory): substantive throughout, compression at close

The practitioner's engagement is characteristically substantive: opened the fork, directed tier widening, authorized paid arm, raised the verifier design question, articulated the paid-component scoping principle, confirmed the settled/open split. The "Makes sense" close-out is brief but follows sustained engagement, consistent with the loop-back #2 precedent (practitioner redirected to outcomes when the agent raised finer-grained design questions). The prior-snapshot pattern (practitioner-initiated challenges at substantive points) holds.

The brief close ("Makes sense") is more compressed than the loop-back #3 close, but the six-criteria evaluation was presented after a full run log, not at the start of a conversation. The compression reads as saturation after engagement, not as sycophantic deflection.

### Signal 6 (embedded conclusions at artifact boundaries): pre-deferral trigger-sizing

The verifier-on-disagreement trigger-sizing (23/20/11% fire rates; 86/75/63% error coverage) was computed from ω.1 data before the practitioner confirmed the deferral. This is minor: the analysis was labeled "analysis only, no new calls" and presented as part of recording the design note, not as a decision-gating artifact. No artifact was produced. The risk that embedded analysis shapes later decisions is low given the explicit deferral notation. Still, it is the pattern to watch if the verifier design enters DECIDE in a future cycle: the analysis that was computed here should be treated as hypothesis-generating, not as pre-validated evidence.

---

## Recommendation

**No Grounding Reframe warranted.**

The signals collectively indicate evidence-forced convergence on a pre-registered refutation. The phase's position in the sycophancy gradient (DECIDE, early-to-mid range) would amplify risk if present, but the pre-registration + methods-review + four-independent-grounds structure is exactly the discipline that earns the convergence. The incumbent-favoring result is the finding, not an assumption.

The one structural ambiguity (hosted-broker-as-verifier vs hosted-broker-as-primary) did not produce an artifact and is deferred. It warrants a BUILD advisory to prevent the verifier framing from being treated as settled when it enters future DECIDE material.

---

## BUILD Advisory Carry-Forwards

**Advisory A (criterion 2 open — explicit at gate entry):** The swappability question — the fork's motivation — is unresolved for both proposals. WP-LB-I resumes under ADR-036, which remains one-model-bound. If the orchestrator surfaces swappability-relevant evidence during BUILD (profile swap re-validation rates, seat-filler-churn costs), that evidence should be recorded as candidate material for the Cycle 8 swappability inquiry, not treated as resolving criterion 2 in A's favor.

**Advisory B (hosted-broker framing):** The "verifier slot" framing for minimax-m2.7 was agent-originated and was not criterion-by-criterion evaluated against the primary-architecture alternative. Before any hosted-broker variant enters a future DECIDE, the primary-architecture alternative should receive the full criteria treatment — including criterion 2 (whether hosted swappability discharges the fork's motivating concern) and criterion 4 (whether the added seams are manageable). The trigger-sizing from ω.1 (20% fire rate on qwen3:8b) is hypothesis-generating, not a validated baseline for a hosted verifier.

**Advisory C (ω.3b finding — prompt-position authority):** qwen3:8b is the sole model where data-position demotion held (0/10 flips on ω.3b). H1 inverted on the wider family. This finding is a structural input to any future work involving instruction-shaped content in the data region of a broker-style prompt — including any system-prompt-as-data design that enters the architecture. The finding should be referenced explicitly in gate conversations where data-position authority is a load-bearing assumption, not treated as a local spike artifact.

---

## Positive Signals

- Pre-registration and methods-review discipline held under mid-spike pressure. The tier-widening and hosted arm were recorded with parameters before runs; no threshold was softened post-run. This is the third phase in sequence (after loop-back #3 DECIDE and loop-back #3 ARCHITECT) where amendment discipline was clean.

- Criterion 2 unresolved-for-both is the sharpest positive signal in this phase. Declaring the fork's motivating criterion unresolved for the incumbent — under evaluation pressure to close the fork — is the right epistemic call and a direct counter to the convergence-without-examination pattern.

- The ω.3a reinterpretation (composition finding vs model finding) is methodologically honest. It does not rescue a disqualified candidate, and it correctly identifies what the fully-data composition variant would need to test. Recording the natural follow-on probe as unspiked preserves the open territory without claiming it as resolved.

- The verifier-on-disagreement deferral includes quantitative trigger-sizing from recorded data — substantive engagement with a practitioner-raised design question, rather than either dismissal or premature commitment.

# Argument Audit Report — ADR-038 (Remaining-Work Anchor), Round 2

**Audited document:** `docs/agentic-serving/decisions/adr-038-remaining-work-anchor.md`
**Source material:**
- `docs/agentic-serving/decisions/adr-037-session-termination-two-call-composition.md`
- `docs/agentic-serving/essays/research-logs/cycle-7-spike-rho-remaining-work-anchor.md`
**Prior audit (R1):** `docs/agentic-serving/housekeeping/audits/argument-audit-decide-cycle-7-loopback6.md`
**Genre:** ADR
**Date:** 2026-06-08

---

## Section 1: Argument Audit

### Summary

- **Genre:** ADR
- **Argument chains mapped:** 5 (composed estimate; imperative adoption; causal isolation; framework-checklist rejection; routing-planner rejection)
- **Issues found this round:** 0 P1, 0 P2, 0 P3
- **Pyramid coverage map:** N/A
- **Expansion-fidelity findings:** N/A

---

### R1 Finding Disposition

Each R1 finding is reviewed in turn.

#### P1-1 (strictly-dominates inconsistency) — RESOLVED

R1 finding: the Decision and Rejected-alternatives sections claimed the imperative "strictly dominates" while the Provenance section hedged the same claim; the hedge did not propagate where the assertion was made.

Revised text: the Decision section now carries the nuance inline — "the honest claim is *the imperative is never worse across the two bases and modestly better on one* — it is adopted because it costs one fixed string and removed B2's lone stuck and no-tool-call cases, not because the evidence shows a population-level advantage." The Rejected-alternatives paragraph on statement-only repeats the same hedge: "the imperative is never worse across the two bases and modestly better on one... not on a demonstrated population-level advantage (the n=10/cell caveat means 9-vs-10 is not a distinguishable rate)."

The internal inconsistency is gone. The hedge is now present at both sites where the claim is made. No residual tension.

#### P2-1 (composed-estimate conflation) — RESOLVED

R1 finding: the composed ~0.9 estimate was presented with the "Factor 1 × Factor 2" framing without making the measurement-design asymmetry transparent — ρ.2-imp ran on the same trials as ρ.1, so the factors are sequential, not independent.

Revised text (Consequences, first bullet): "this is a direct observation of the production composition (the judge's real statement anchoring call 2), not an independence-valid Factor 1 × Factor 2 product: ρ.2-imp ran on the judge's actual statements, so Factor 1 is already folded into the 19/20. ρ.1's separate 20/20 confirms the anchor was not poisoned (it explains *why* the 19/20 holds), rather than being an independent multiplicand."

The revised language directly states what R1 requested: the 19/20 is a direct end-to-end observation on the production composition, not a multiplication from independent arms. The "composed" label is retained for the Provenance note, where it explains the historical Factor 1/Factor 2 framing — appropriate, since that note documents drafting-time synthesis for auditability. Finding resolved.

#### P2-2 (control confound) — RESOLVED

R1 finding: the causal-isolation argument characterized the control as "content-neutral" but the control's actual content (a delegation reminder) was not semantically inert — it may have actively held attention on delegation style rather than target selection.

Revised text (Rejected-alternatives, "Mere trailing-token perturbation"): "Qualification: the control text was a delegation-style reminder ('Remember: delegate generation…'), not a semantically inert filler, so it may have actively held attention on delegation *style* rather than being neutral; a stricter control would be length-matched nonsense. The 0.8 gap is practically decisive regardless, but the precise isolation claim is 'remaining-work content advances where a plausible non-target trailing sentence does not,' not 'where arbitrary tokens do not.'"

This matches R1's recommendation almost verbatim. The practical conclusion is unchanged; the scope of the causal claim is correctly narrowed. Finding resolved.

#### P2-3 (checklist-semantic equivocation) — RESOLVED

R1 finding: the framework-checklist rejection borrowed ADR-037's "semantic task-decomposition" argument without distinguishing the lighter filename-extraction operation from the harder completeness-quality judgment. The rejection's actual load-bearing argument (redundancy given ρ.1's 20/20) was correct; the equivocation diluted it.

Revised text: "A secondary point, not the load-bearing one: the checklist's 'which deliverables were requested' is lighter than the *completeness-quality* judgment ADR-037 established as semantically hard — 'is this work adequate?' is harder than 'which filenames were asked for' — so the checklist is not as infeasible as ADR-037's semantic argument might suggest; the rejection rests on redundancy, not infeasibility."

The revised text explicitly distinguishes the two operations and correctly demotes the semantic-infeasibility argument to secondary status. Redundancy-not-infeasibility is now the stated load-bearing ground. Finding resolved.

#### P3-1 (single-run discharge) — RESOLVED

R1 finding: the Conditional Acceptance discharge condition said "both clear together on the multi-file run" but did not make explicit that REMAINING-with-advance must precede COMPLETE in the same run — separate runs for each property do not satisfy the joint gate.

Revised text (Empirical grounding): "a **single** real-OpenCode session on a multi-deliverable task in which the session both advances through all deliverables (no churn on file 1) **and then** converges (the COMPLETE finish ADR-037 validated), both observed in that one session — separate single-file and advance-only runs do not satisfy it. Verified from serve-log evidence within the run: `turn decision:` lines showing REMAINING with a `dispatch start` for each *distinct* deliverable in sequence, then a COMPLETE `action=finish`."

The "and then," "both observed in that one session," and "separate single-file and advance-only runs do not satisfy it" phrasing closes the gap R1 identified. The serve-log evidence specification (distinct-deliverable REMAINING dispatch lines followed by COMPLETE finish) is more specific than R1 requested. Finding resolved.

#### P3-2 (delegation-FC no-tool-call) — RESOLVED

R1 finding: the FC (delegation preserved under the anchor) named only inline `write` as the refutation signal; the 1/10 no-tool-call (premature-finish) failure mode observed in ρ.2-imp B3 was not captured.

Revised text: "Refutable: an inline `write` of generated content on an anchored call-2 turn, **or** a rise in no-tool-call (premature-finish) turns above the ρ-measured ≤1/10 — the anchor must not push the seat into finishing or writing inline instead of delegating the named deliverable."

Both violation types are now named. The ≤1/10 threshold is specific and matches what the spike measured. Finding resolved.

#### P2-F1 (generalize-past-scope) — RESOLVED

R1 finding: the Consequences (Positive) bullet said the mechanism "should generalize as the judge's naming reliability generalizes," reaching past the stated scope boundary in the same section.

Revised text (Consequences, Positive, third bullet): "The remaining-work content is causally isolated (control 0/10 vs anchored 8–10/10), so the mechanism is understood, not a lucky perturbation. Whether it generalizes beyond the measured scope (qwen3:14b, file-write deliverables) is not claimed here — see the Negative scope boundary."

The optimistic generalization is removed. The sentence now ends at the causal-isolation claim and explicitly defers scope extension to the Negative boundary entry. Finding resolved.

#### P2-F2 (routing-planner heavier-subsystem) — RESOLVED

R1 finding: the routing-planner rejection characterized "heavier subsystem" as a peer argument to the confabulation-surface argument, but only the confabulation argument drew on measured evidence; the proportionality characterization was an engineering judgment presented without its own evidence.

Revised text: "Rejected primarily because it re-opens the planner-confabulation surface Cycle 6/7 spent effort bounding (the evidence-backed reason)... (Secondary, an engineering judgment without its own evidence: the planner is also a heavier subsystem — a third role — for a problem the judge's already-computed output solves with a one-sentence anchor; this proportionality point reinforces but does not carry the rejection.)"

The confabulation-surface argument is now explicitly labeled "the evidence-backed reason" and the proportionality point is labeled "an engineering judgment without its own evidence" that "reinforces but does not carry the rejection." The relative authority of the two arguments is correct. Finding resolved.

#### P3-F1 (harness coupling) — RESOLVED

R1 finding: the Consequences (Negative) bullet on tuning said the imperative is tunable at the FC-58 evidence bar but did not note that re-tuning the judge prompt (which generates the routed-forward statement) also requires ρ.2 re-validation — the two re-validation surfaces are coupled.

Revised text: "**The judge prompt and the action anchor are now coupled.** Because the routed-forward anchor *is* the judge's remaining-work statement, a change to the judge question's wording (the θ-harness tuning surface) changes the statement that anchors call 2 — so a judge-prompt revision re-validates both the θ judgment arms AND the ρ anchor arms, not the judgment alone. The two re-validation surfaces are no longer independent."

This is a full articulation of the coupling the R1 framing audit surfaced. The dependency is visible and actionable. Finding resolved.

---

### New Findings This Round

None. No new P1, P2, or P3 issues were identified in the revised document.

The revised ADR is internally consistent across all sections reviewed. The evidence-claim relationships are correctly scoped throughout; no new overreach, contradiction, or unstated leap was found.

---

### P1 — Must Fix

None.

### P2 — Should Fix

None.

### P3 — Consider

None.

---

## Section 2: Framing Audit

### R1 Framing Issue Disposition

R1 raised three framing issues: P2-F1 (generalize-past-scope), P2-F2 (routing-planner), and P3-F1 (harness coupling). All three are resolved in the argument audit above; they required no framing-specific revision beyond the text changes already reviewed.

The three alternative framings identified in R1 (interaction-design framing, ADR-037-FC-as-predicted-consequence framing, two-surface-tuning-dependency framing) are now either directly addressed in the ADR (coupling noted in Consequences Negative) or remain valid meta-observations that do not require ADR changes — the ADR's framing is accurate given the evidence; these are lenses for future work, not gaps.

### Question 1: What alternative framings did the evidence support?

No new alternative framings emerged in the revised document. The three R1 framings are noted; the interaction-design framing and the ADR-037-scope framing remain analytically interesting but are not contradicted by the revised text. The revised text does not foreground them, which is appropriate — they are not the strongest framing for the evidence.

### Question 2: What truths were available but not featured?

R1's three underrepresented findings (B3 no-tool-call sub-classification, rung-1 "other" baseline rate, statement-only as named fallback) were P3-level observations in R1, not findings that required ADR changes. They remain in the evidence base and are not contradicted by the revision. The revised Consequences section characterizes the B3 1/10 none case implicitly through the FC (delegation preserved under the anchor) refutation threshold — any rise above ≤1/10 fires the FC, which provides the watch signal R1 recommended without requiring a dedicated Consequences line.

### Question 3: What would change if the dominant framing were inverted?

The two-surface coupling finding (P3-F1) is now present in the ADR, which was the main gap the R1 inversion analysis surfaced. The revised Consequences Negative entry on coupling handles the key practical consequence the inverted framing identified.

### Framing Issues

None new. All R1 framing issues resolved.

---

## Convergence-Saturation Signal (ADR-094)

**Convergence-Saturation Signal:** TRIGGERED
- Round number: R2
- P1 count this round: 0 (Section 1 + Section 2 combined)
- P2 count this round (new, non-carry-over): 0
- New framings or claim-scope expansions: none
- Recommendation: STOP at this round

**ADR-038 is clean to accept.** All nine R1 findings (P1-1, P2-1, P2-2, P2-3, P3-1, P3-2, P2-F1, P2-F2, P3-F1) are resolved. No new issues were introduced by the revisions. The document's argument is internally consistent, evidence claims are correctly scoped, and fitness criteria are refutable in specified form.

*The Conditional Acceptance discharge gate (real-OpenCode multi-file convergence run) remains outstanding — this is by design per ADR-097 and does not block argument-audit acceptance. That gate clears at BUILD.*

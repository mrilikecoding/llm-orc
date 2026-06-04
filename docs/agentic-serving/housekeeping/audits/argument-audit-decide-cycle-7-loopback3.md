# Argument Audit Report — R1

*(Recovered: the R1 auditor returned this report as its final message but did not
write it to the output path; the orchestrator persisted it verbatim on
2026-06-03 when the R2 auditor flagged the missing file. Findings and text are
the R1 agent's, unedited.)*

**Audited document:** `docs/agentic-serving/decisions/adr-036-delegation-decision-mechanism.md`
**Source material read:**
- `docs/agentic-serving/essays/research-logs/cycle-7-spike-psi-delegation-rate.md`
- `docs/agentic-serving/housekeeping/audits/research-methods-spike-psi-prime.md`
- `docs/agentic-serving/decisions/adr-033-layer-a-loop-driver-multi-turn-agentic-surface.md`
- `docs/agentic-serving/decisions/adr-032-fallback-shape-and-transparent-endpoint-split.md`
- `docs/agentic-serving/decisions/adr-035-client-tool-deliverable-form-contract.md`
- `docs/agentic-serving/domain-model.md` §Invariants (AS-9, AS-10), §Methodology Vocabulary (Generation-shaped turn, Delegation rate)
**Genre:** ADR
**Date:** 2026-06-03

---

## Section 1: Argument Audit

### Summary

- **Genre:** ADR
- **Argument chains mapped:** 8 (baseline suppression → user-turn lever; V3 rate claim; carry-side preservation; multi-turn attachment form; model non-portability; 0.9 threshold; escalation-path deferral; profile-swap revalidation)
- **Issues found:** 8 (0 P1, 4 P2, 4 P3)
- **Pyramid coverage map:** N/A
- **Expansion-fidelity findings:** N/A

---

### P1 — Must Fix

None.

---

### P2 — Should Fix

**P2-1 — The 55/55 aggregation silently combines arms measured under materially different conditions**

- **Location:** §Context "Cumulative V3 on qwen3:14b: 55/55 delegated" and §Decision 2 "The measured 55/55 covers…"
- **Claim:** The ADR presents "55/55" as a single cumulative figure giving unified coverage evidence.
- **Evidence gap:** The 55 breaks down as: ψ V3 first-turn 10/10 (single phrasing, single context, first-turn only) + ψ V3-args 5/5 (argument-capture rerun of the same request) + ψ′ Arm A 25/25 (five phrasings including multi-instruction, first-turn contexts) + ψ′ Arm C 15/15 (three multi-turn attachment forms). The ψ V3-args 5/5 are re-runs of the ψ baseline request — they add argument-quality confidence but are not independent phrasing observations. The ψ′ Arm C 15/15 is multi-turn-attachment evidence, not additional phrasing evidence. Adding these counts into a single "55" obscures that only 35 of the 55 are phrasing-varied first-turn observations, and the 15 C-arm runs establish the multi-turn attachment form holds — not that the delegation rate itself is uniformly "55/55 on generation tasks." The research log maintains the arm breakdown clearly, but the ADR's §Context uses the aggregate in a way that could be read as 55 independent delegation-decision observations.
- **Recommendation:** Either present the breakdown inline (e.g., "40 phrasing-varied first-turn observations at ψ + ψ′ A; 15 multi-turn attachment confirmations at ψ′ C; cumulative 55") or qualify the aggregate so it is clear what each component demonstrates. The current prose, while technically accurate, presents a composite number whose subcomponents are doing different epistemic work.

**P2-2 — The 0.9 threshold is presented as having a derivation from evidence, but the connection is underspecified**

- **Location:** §Decision 3 "Refutation threshold: sustained `delegation_rate` < 0.9 on generation-shaped turns over a 24-hour rolling window is refutation evidence"
- **Claim:** The 0.9 threshold is the tool-driven analogue of ADR-032's `direct_completion_rate` pattern; PLAY refines the number.
- **Evidence gap:** ADR-032's threshold is a deployment-relative structural relationship (~15 percentage points above baseline), not an absolute figure. The ADR-036 threshold is an absolute figure (0.9) derived at drafting time from the 55/55 measured rate. The §Provenance check correctly labels it as "drafting-time synthesis" and notes the practitioner sets or revises it at the gate. However, §Decision 3's body text — which does not include this qualification — presents the 0.9 as a committed operational parameter. A reader who does not read §Provenance check will not know the threshold is provisional rather than measurement-derived. The analogy to ADR-032 is structurally correct (both are rolling-window degradation signals), but ADR-032 explicitly labels its threshold as a "rough" heuristic and explains why absolute figures are premature without deployment data; the ADR-036 body text does not carry this caveat in-line.
- **Recommendation:** Move the §Provenance check qualification ("drafting-time synthesis; practitioner sets or revises at the gate, PLAY refines") into the §Decision 3 body as a parenthetical. The threshold is a reasonable starting point, but the body text should signal its provisional status at the same point where the reader encounters it, rather than only in the provenance section.

**P2-3 — The C3 production-form choice is flagged as drafting-time synthesis in the provenance, but the fitness criterion anchors on it as the committed implementation form without noting the choice's basis**

- **Location:** §Decision 1 (C3 form described as the production form for tool-result tails); §Fitness criteria (FC directive-in-user-turn presence specifies "standalone trailing user-role message on tool-result tails")
- **Claim:** The C3 form — guidance as a standalone trailing user-role message — is the production form for trailing turns, chosen for implementation cleanliness.
- **Evidence gap:** C1, C2, and C3 all hit 5/5 in ψ′ Arm C. The choice of C3 over C1/C2 is explicitly labeled as a drafting-time synthesis decision in §Provenance (chosen for implementation cleanliness, not rate evidence). That is a legitimate design choice. However, the FC (directive-in-user-turn presence) specifies C3's specific form as the refutable criterion ("standalone trailing user-role message on tool-result tails") without noting that the alternative forms were equally measured. This creates a situation where C3's specific form is treated as the only valid implementation, when the evidence equally supports C1 or C2. If a future BUILD implementation uses C1 for implementation reasons, it would technically violate the FC even though C1 passed at 5/5.
- **Recommendation:** The FC should specify the structural property (guidance is in the user-turn region, not a system message; on trailing turns, guidance is appended as a user-role message) rather than the specific attachment variant. C3's specific form can be named as the preferred form with the C1/C2 equivalence noted, so the FC is not falsified by equally-measured alternatives.

**P2-4 — The Conditional Acceptance discharge condition references "ADR-097's shape" without stating what it requires; the condition as written may be under-specified relative to what ADR-097 actually demands**

- **Location:** §Empirical grounding "Conditional Acceptance: discharged to full acceptance when the BUILD acceptance run lands the end-to-end evidence at the real-client layer — a real-OpenCode session with the V3 composition in which delegation verifiably fires… and the production meter reads ≥0.9 over its first soak window."
- **Claim:** The Conditional Acceptance is discharged by (a) a real-OpenCode session with delegation verified via serve-log evidence, plus (b) production meter reading ≥0.9 over its first soak window.
- **Evidence gap:** Condition (b) — "production meter reads ≥0.9 over its first soak window" — depends on having a production deployment running sufficient generation-shaped traffic for the meter to be meaningful. In a BUILD acceptance run, the soak window may be a handful of sessions. The ADR does not specify what constitutes a qualifying soak window (number of generation-shaped turns, time duration, traffic volume). Comparably, ADR-035's Conditional Acceptance condition for PLAY is described as "sustained form-compliance over long multi-turn trajectories" — which is also directional, but ADR-035 names what the axis-2 validation regime looks like. ADR-036's discharge condition (b) may be met by a single-session production run if the meter happens to read ≥0.9, which is structurally different from the production-soak evidence the condition is intended to require.
- **Recommendation:** Specify a minimum qualifying soak window for condition (b) — for example, a minimum number of generation-shaped turns (the spike ran at n=25 for Arm A; a comparable BUILD-phase minimum would be coherent) or a minimum number of real-client sessions. Alternatively, make condition (a) the gating condition and treat condition (b) as a trailing confirmation, noting that the meter provides ongoing regression visibility rather than a one-time acceptance gate.

---

### P3 — Consider

**P3-1 — The classifier boundary-turn exclusion is labeled a "measurement-integrity choice the spikes did not test" but no guidance is given on what happens if boundary-shaped traffic grows**

- **Location:** §Decision 3, §Consequences (Negative): "The denominator's classifier has a known boundary (repair-shaped tasks, uncovered content domains); rate accuracy degrades if boundary-shaped traffic dominates."
- **Claim:** Boundary turns are excluded from the denominator rather than guessed.
- **Evidence gap:** The ADR correctly records this limitation, but does not name a monitoring trigger for the "boundary traffic dominates" scenario. The degradation is noted but there is no guidance on when a practitioner should re-examine the classifier's coverage. The ψ.4a ambiguous cases (a01 repair-shaped; a03 prose generation) are the known boundary shapes, but the ADR does not specify what observation would indicate the denominator is no longer reliable.
- **Recommendation:** Add a brief note on what a "boundary traffic dominating" observation looks like operationally — e.g., a high rate of unclassified turns, a growing gap between observed generation-shaped turns and the denominator count, or a specific PLAY observation point where boundary classification is re-examined. This is a monitoring-design gap, not a logical flaw.

**P3-2 — The "53×-larger client prompt" attention framing is used in the Context section but the ψ.0 capture fact records a 53:1 character ratio, not a 53:1 attention ratio; the ADR inherits a mild elision from the research log**

- **Location:** §Context, and by implication in the Rejected Alternatives §System-slot guidance variants
- **Claim:** The character-ratio makes the attention contest concrete; the system slot loses "regardless of position or wording strength."
- **Evidence gap:** The 53:1 character ratio is a proxy for an attention contest whose mechanism (relative token weight, positional encoding effects, instruction priority under long-context models) is not characterized. The research log records F-ψ.2's mechanism note: "V3 changes two things at once (role: system→user; adjacency: directly attached to the task)" and explicitly notes the spike does not isolate role vs. adjacency. The ADR correctly records this in §Consequences Neutral ("the OpenCode prompt-budget contest is sidestepped, not won"), but the §Context framing of the 53:1 ratio as a direct causal explanation for attention-contest suppression is slightly stronger than what the spike measured.
- **Recommendation:** A brief qualifier in §Context — "a proxy for the attention contest, whose role-vs-adjacency mechanism the spike does not isolate" — would align the context framing with the more careful mechanism note already present in §Consequences Neutral.

**P3-3 — Arm D notation check**

- **Location:** §Context, Spike ψ′ results summary.
- **Claim:** Correct statement of the Arm D results.
- **Evidence gap:** None — notation matches the research log. Finding noted for completeness.
- **Recommendation:** No change required.

**P3-4 — The escalation path deferral argument (Decision 5) makes a sound decision at the current evidence level, but does not record what event would re-examine the complexity threshold**

- **Location:** §Decision 5 "Detect-and-retry… is not built: at 55/55 measured there is no evidence it would fire often enough to justify its complexity."
- **Claim:** At 55/55, detect-and-retry would rarely fire; complexity is not justified.
- **Evidence gap:** The argument is sound given current evidence. However, the ADR names two re-opening triggers ("PLAY or production-meter evidence re-opens it") without specifying what production-meter reading would warrant building detect-and-retry. The decision 3 refutation threshold (delegation_rate < 0.9) is related but distinct — a rate below 0.9 would indicate the mechanism is failing, which is a reason to investigate the mechanism rather than add a retry layer. The distinction between "mechanism failing (investigate)" and "mechanism working at 85% rate (add retry)" is not made explicit.
- **Recommendation:** Note in Decision 5 what meter reading would warrant building detect-and-retry vs. what reading would indicate the underlying mechanism needs diagnosis. A plausible formulation: delegation_rate between 0.85–0.9 might warrant retry (mechanism mostly working but below threshold); delegation_rate below 0.85 or degrading trend might warrant mechanism investigation rather than retry.

---

## Section 2: Framing Audit

### Question 1: What alternative framings did the evidence support?

**Alternative framing A — The lever as a (composition × client) property, not just a (composition × model) property.** The system prompt that constitutes the attention contest is the OpenCode 1.15.5 system prompt (27,925 chars). The leverage mechanism is therefore validated for a specific (composition × model × client version) triple. A future OpenCode update that restructures or lengthens its system prompt could plausibly shift the attention contest. The ADR names "a client-prompt change could shift rates in either direction" in §Consequences Neutral, but frames it as a monitoring concern rather than a first-class constraint comparable to model-profile pinning.

**Alternative framing B — ψ.4c's empty-response finding as a constraint on tool-list design, not just a rejected alternative.** The empty-response failure reveals that qwen3:14b has a behavioral failure mode when it judges the offered tool set incompatible with the task — relevant to any scenario where the tool list is accidentally narrowed (a registration bug; a capability domain with no registered ensemble).

**Alternative framing C — The pre-filter's "relocation" as a BUILD commitment, not just a measurement design choice.** The pre-filter must be shipped as production code for the delegation-rate meter to function, and its known boundaries must be actively maintained as the capability library expands. The ADR does not name who keeps the classifier current as new capability ensembles are added.

### Question 2: What truths were available but not featured?

**Underrepresented finding 1 — B4 repair-boundary behavior as a multi-turn delegation question.** F-ψ′.4's read-first behavior is qualitatively different from the other scope exclusions: it leaves genuinely open whether the post-read generation turn delegates. Its absence from the fitness criteria or consequences means a BUILD implementer may not know to watch this specific turn pattern.

**Underrepresented finding 2 — The role-vs-adjacency mechanism isolation the spike deliberately deferred.** If the operative factor is adjacency rather than role, the C3 trailing form might have different dynamics than first-turn injection; ψ′ C3 shows it works at depth 3, but the mechanism is uncharacterized.

**Underrepresented finding 3 — The context-growth risk.** The ratio of guidance to total context shrinks as tool results accumulate; deeper contexts are *more* likely to suppress delegation, not just different from measured contexts. The ADR names the residual without explaining the directionality of the risk.

### Question 3: What would change if the dominant framing were inverted?

Inverted framing: "the framework found a fragile equilibrium in a specific (client × model × task) triple, and the governance mechanism (instrumentation) is the load-bearing safety net rather than the mechanism itself." Under inversion, the profile-swap re-validation FC becomes the primary decision; the "55/55 — delegation is reliable" headline becomes "55/55 under the specific client version and model; the meter tells us if it holds." The inversion does not reveal a hidden flaw but does reveal that the instrumentation is doing more structural work than the framing implies. The ADR is aware of this inversion and names it in §Consequences Neutral.

### Framing Issues

**P2-F1 — The ψ.4c empty-response finding's implications for tool-list design are underexplored.** Recommendation: add a sentence noting the framework should never dispatch a request with a tool list that excludes all plausible response tools for the turn shape.

**P2-F2 — The D-arm non-transfer implications are scoped to seat-filler model changes but not to model-family boundaries more broadly.** The failure boundary is characterized by only two negative data points. Recommendation: note the boundary is not characterized; the re-validation obligation applies to any profile change with risk magnitude varying by family.

**P3-F1 — The belief-map prospectivity is not stated.** Recommendation (clarity only): note feed-forward #3 was recorded before the spikes ran.

**P3-F2 — The "won, not coerced" framing carries an implicit risk dimension.** Recommendation: consider a parenthetical noting the win is a (composition × model × client version) property with instrumentation making future losses visible.

---

## Convergence-Saturation Signal (ADR-094)

**Convergence-Saturation Signal:** NOT TRIGGERED

- Round number: R1
- P1 count this round: 0 (Section 1: 0; Section 2: 0)
- P2 count this round (new, non-carry-over): 6 (Section 1: P2-1, P2-2, P2-3, P2-4; Section 2: P2-F1, P2-F2)
- New framings or claim-scope expansions: the (composition × client version) dimension of the validated claim scope; the role-vs-adjacency mechanism gap in trailing-turn contexts.
- Recommendation: CONTINUE to next round.

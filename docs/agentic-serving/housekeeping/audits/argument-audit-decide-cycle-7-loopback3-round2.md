# Argument Audit Report

**Audited document:** docs/agentic-serving/decisions/adr-036-delegation-decision-mechanism.md
**Source material:**
- docs/agentic-serving/essays/research-logs/cycle-7-spike-psi-delegation-rate.md
- docs/agentic-serving/housekeeping/audits/research-methods-spike-psi-prime.md
- docs/agentic-serving/domain-model.md §Methodology Vocabulary + §Invariants AS-9/AS-10
- docs/agentic-serving/decisions/adr-033-layer-a-loop-driver-multi-turn-agentic-surface.md
- docs/agentic-serving/decisions/adr-032-fallback-shape-and-transparent-endpoint-split.md
- docs/agentic-serving/decisions/adr-035-client-tool-deliverable-form-contract.md
**Genre:** ADR
**Date:** 2026-06-03

---

## R1 Finding Status Summary

The R1 audit (described in the dispatch brief; no file exists at the expected path
`docs/agentic-serving/housekeeping/audits/argument-audit-decide-cycle-7-loopback3.md`)
reported 0 P1, 4 argument P2s, 4 argument P3s, 2 framing P2s, and 2 framing P3s. The
dispatch brief specifies that the following corrections were applied before this R2 audit:

| R1 finding | Applied? | R2 status |
|------------|----------|-----------|
| P2-1 (55/55 decomposed inline with epistemic-work distinction) | Yes | **Held** |
| P2-2 (0.9 threshold provisional status moved into Decision 3 body) | Yes | **Held** |
| P2-3 (FC directive-in-user-turn re-specified as structural property with C1/C2/C3 equivalence noted) | Yes | **Held** |
| P2-4 (Conditional Acceptance restructured: real-client run = gating condition; meter = trailing confirmation with ≥25 generation-shaped-turn qualifying window) | Yes | **Held** |
| P3-1 (boundary-degradation operational signal + re-examination triggers) | Yes | **Held** |
| P3-2 (53:1 ratio qualified as a proxy, mechanism not isolated) | Yes | **Held** |
| P3-4 (retry-vs-diagnose meter routing: ~0.85–0.9 retry territory vs <0.85 mechanism diagnosis) | Yes | **Held** |
| P3-F1 (prospective belief-map provenance note) | Yes | **Held** |
| P2-F1 (ψ.4c empty-response tool-list design implication) | Not applied (held for practitioner) | Carry-over noted |
| P2-F2 (portability failure boundary uncharacterized) | Not applied (held for practitioner) | Carry-over noted |
| P3-F2 (decision-statement parenthetical on the conditional nature of the win) | Not applied (held for practitioner) | Carry-over noted |

Verification findings for each correction are integrated into the sections below.

---

## Section 1: Argument Audit

### Summary

- **Genre:** ADR
- **Argument chains mapped:** 5 (one per Decision item)
- **Issues found:** 3 (0 P1, 1 new P2, 2 carry-over P3)
- **Pyramid coverage map:** N/A
- **Expansion-fidelity findings:** N/A

### P1 — Must Fix

None.

### P2 — Should Fix

**P2-1 (NEW — introduced by P2-4 revision):** The Conditional Acceptance section
(§Empirical grounding) now structures the production-meter soak window as the "trailing
confirmation" with a stated qualifying window of "minimum of 25 generation-shaped turns...
reading ≥0.9." The ≥25 figure is noted as "matching ψ′ Arm A's n." However, Arm A (n=25)
was a phrasing-generalization arm — its n was chosen to achieve statistical power against
the 0/10 baseline, not to calibrate a minimum soak-window size for a production deployment.
The document carries the reasoning as-if the Arm A sample size grounds the soak-window
choice, but the sample-size-to-acceptance-criterion translation is unstated and the two
contexts differ: Arm A measures first-turn delegation rate across phrase variation in a
controlled replay; the production soak measures first-turn delegation rate in live traffic
across unknown phrase variation, client-prompt micro-versions, model-update intervals, and
potential classifier errors. The document does not explain why 25 generation-shaped
production turns provides the same evidentiary threshold as 25 controlled replay turns.
This is an implicit cross-context equivalence claim that should either be made explicit
(with a rationale) or weakened to "a provisional minimum, grounded in ψ′ Arm A's n as a
starting reference, subject to practitioner revision at the gate."

- **Location:** §Empirical grounding, last paragraph.
- **Claim:** "a qualifying first soak window is a minimum of 25 generation-shaped turns
  (matching ψ′ Arm A's n) reading ≥0.9."
- **Evidence gap:** The equivalence between controlled-replay sample size and production
  soak-window size is unstated. A 25-turn production window accumulated over wall-clock
  time in live traffic is a different epistemic context than 25 controlled replays.
- **Recommendation:** Add a qualifier: "a provisional minimum of 25 generation-shaped
  turns, using ψ′ Arm A's n as a starting reference; subject to practitioner revision at
  the gate based on traffic volume and deployment context."

### P3 — Consider

**P3-1 (CARRY-OVER from R1 P3-3 — partially addressed but not fully closed):** Decision
5 now includes the retry-vs-diagnose routing with the ~0.85–0.9 and <0.85 bands. This
satisfies the R1 P3-4 finding. However, the document uses "~0.85–0.9" and "~0.85" with
tildes without explaining how these sub-threshold bands were derived. The 0.9 threshold
is qualified as provisional in Decision 3 (P2-2 correction held). The sub-bands (0.85–0.9
and below 0.85) are introduced in Decision 5 without provenance — they appear to be
drafting-time synthesis, but no provenance note marks them as such. The Provenance check
section does not mention these sub-bands. Minor clarity gap, not a logical failure.

- **Location:** Decision 5, last two sentences.
- **Claim:** "a rate between ~0.85 and 0.9 (mechanism mostly working, below threshold) is
  the retry candidate's territory; a rate below ~0.85 or a degrading trend points at
  mechanism diagnosis."
- **Evidence gap:** No source for the 0.85 sub-band figure; Provenance check is silent.
- **Recommendation:** Add a parenthetical in the Provenance check or inline noting that
  the 0.85 sub-band is drafting-time synthesis from the threshold shape, not a measured
  value — parallel to the existing provenance treatment of the 0.9 figure.

**P3-2 (NEW — minor):** Decision 4 states that a seat-filler Model Profile change
"composes with ADR-033's seat-filler-swappability fitness criterion rather than amending
it: swappability remains a structural property (config-only change, no code edits); trust
in the swapped configuration is an empirical property (a recorded re-validation run, or
the production meter under decision 3 watched through a soak window)." The production-meter
path as a re-validation alternative is technically sound, but the soak-window size for
this purpose is not specified here — Decision 3 defines the soak-window for the first
acceptance gate. Whether the same ≥25 generation-shaped-turn window applies to profile-swap
re-validation is implicit. Trivially small issue, but the absence creates a minor ambiguity
for the FC (profile-swap re-validation) fitness criterion.

- **Location:** Decision 4, second sentence; FC (profile-swap re-validation) in Fitness
  criteria.
- **Claim:** "a recorded re-validation run, or the production meter under decision 3
  watched through a soak window" satisfies the FC.
- **Evidence gap:** Soak window size for the profile-swap case is not stated.
- **Recommendation:** Add "soak-window size per decision 3" as a cross-reference in the
  FC definition, or note that the ≥25 generation-shaped-turn minimum applies here as well.

---

### R1 Correction Verification Notes

**P2-1 (55/55 decomposition):** The decomposition is present in §Context: "55/55
delegated — decomposed: 40 first-turn observations across six distinct phrasings...
and 15 multi-turn attachment confirmations." The epistemic-work distinction is explicit:
"The components do different epistemic work: the 40 establish the first-turn delegation
decision across phrasing variation; the 15 establish that the attachment form survives
tool-result tails to depth three." **Held.**

**P2-2 (0.9 threshold provisional status):** Decision 3 now contains: "the tool-driven
analogue of ADR-032's deployment-relative `direct_completion_rate` threshold — which, like
ADR-032's number, is provisional: the 0.9 figure is drafting-time synthesis from the
measured rates, set or revised by the practitioner at the gate and refined by PLAY with
deployment data." **Held.**

**P2-3 (FC directive-in-user-turn structural specification):** The FC now reads: "The
structural property is user-role placement; the attachment variant is implementation choice
among the equally-measured forms (attached to the user task, C1/V3; first-message-only,
C2; standalone trailing user-role message, C3 — the preferred form per Decision 1, all
5/5 at measured depth). Refutable: a composed seat-filler request with the guidance in a
system message, or absent, violates this; using C1 or C2 instead of C3 does not."
**Held.**

**P2-4 (Conditional Acceptance structure):** §Empirical grounding now explicitly labels
the BUILD acceptance run as "gating condition" and the production meter as "trailing
confirmation, not a one-time acceptance gate," with the ≥25 generation-shaped-turn
qualifying window. The new P2-1 above surfaces a residual gap in this correction.

**P3-1 (boundary-degradation signal):** Decision 3 now includes: "Operational signal that
the denominator is degrading: a growing share of boundary-excluded turns relative to
classified turns — re-examine the classifier's coverage at PLAY, and whenever new
capability ensembles are registered." **Held.**

**P3-2 (53:1 ratio qualified):** §Context now reads: "the 53:1 character ratio — a proxy
for the attention contest, whose role-vs-adjacency mechanism the spike does not isolate."
**Held.**

**P3-4 (retry-vs-diagnose routing):** Decision 5 now contains the ~0.85–0.9 / <0.85
routing bands. **Held** (with residual P3-1 carry-over above noting the bands lack
provenance attribution).

**P3-F1 (prospective belief-map provenance):** §Rejected alternatives, Structural
pre-filter subsection, now reads: "belief-mapped deliberately rather than dismissed (WP-LB-H
feed-forward #3, recorded before the spikes ran, named the pattern-rhyme risk in both
directions — the belief-map was applied prospectively, not as a finding-driven retroactive
label)." **Held.**

---

## Section 2: Framing Audit

### Question 1: What alternative framings did the evidence support?

**Alternative framing A: The portability failure boundary as an unresolved design risk.**
The source material (Spike ψ′ Arm D: qwen3.5:9b 1/5, mistral-nemo:12b 2/5; the
methods-review P2-B portability note) supports framing the portability result not merely
as a scoping constraint but as an open design question: what properties of a model predict
V3 reliability, and does the current model-profile selection surface expose those
properties? The ADR frames portability as "the lever is a (composition × model) pair,
validated for qwen3:14b" — a scoping qualification — but the source material also contains
a portability-failure boundary that is entirely uncharacterized (Arm D tested two models,
both failing; neither the methods-review nor the spike identifies what structural property
distinguishes qwen3:14b from the failing models). Under this framing, the ADR would read:
"delegation reliability depends on an unknown model property; we have validated one model
and the FC requires re-validation, but the re-validation process is uninformed because the
failure mechanism is unknown." What would the reader need to believe for this framing to
be right: that the portability failure is not merely a scoping note but a latent
architectural risk — specifically, that future seat-filler model swaps may fail the
re-validation in ways that are difficult to diagnose without understanding the failure
mechanism.

**Alternative framing B: The denominator problem as the binding constraint.**
The generation-shaped-turn classifier has a known boundary: repair-shaped tasks and
content domains with no registered capability ensemble are excluded. Decision 3 includes
the boundary-degradation signal (a growing share of excluded turns), but the source
material (Spike ψ.4a: 2/4 ambiguous boundary cases, including the a01 repair-shaped and
a03 prose-generation cases) supports a stronger framing: the classifier's boundary is
not merely a measurement-accuracy issue but a potential coverage-adequacy constraint as
the capability library grows. As new ensembles are registered (each registration "widens
the capability-domain term"), the denominator expands — but the repair-shaped exclusion
is structural (not capability-dependent) and permanently excludes a turn class whose
production incidence is unknown. Under this framing, the ADR would foreground the
denominator's brittleness over time, not just its current accuracy. What would the reader
need to believe: that repair-shaped tasks constitute a growing fraction of real production
traffic as the system matures, making the excluded denominator a progressively less
representative measure of delegation reliability.

### Question 2: What truths were available but not featured?

**1. The role-vs-adjacency mechanism is not isolated (source: Spike ψ F-ψ.2, research
log §"Follow-up results").**
The research log records explicitly: "V3 changes two things at once (role: system→user;
adjacency: directly attached to the task). V1's failure shows ordering alone is not the
lever, but the spike does not isolate role-vs-adjacency further." The revised ADR
qualifies the 53:1 ratio as a proxy (P3-2 correction held), but the role-vs-adjacency
mechanism isolation gap is not surfaced as a named limitation. The Consequences §Neutral
section notes that the "OpenCode prompt-budget contest is sidestepped, not won" and that
"a future client-prompt change could shift rates in either direction," which partially
acknowledges the mechanism gap, but does not name it explicitly. The implication is that
the two intertwined variables (role placement and task adjacency) mean the ADR's
composition choice cannot be reliably decomposed for diagnostic purposes — if V3 degrades
in production, it may be unclear whether to adjust role placement, adjacency, or both.
This would complicate the meter-triggered diagnostic path Decision 3 describes.

**2. The B4 repair-boundary first-turn result (source: Spike ψ′ results table, B4).**
Arm B4 (repair-shaped: "Fix the bug in string_utils.py where count_vowels misses
uppercase vowels," n=5) produced 0/5 delegations — the model chose read/glob first on
every run. The ADR notes this in the Consequences §Negative entry ("Residual unmeasured
territory: ... delegation on the generation turn after a read in repair-shaped flows"),
but the B4 finding is relevant to a tighter point: the boundary-excluded turns in the
denominator are not just classification ambiguities but turns that may represent
*correct* model behavior (reading before acting on repair-shaped tasks is the right
sequence). The exclusion of these turns from the denominator is therefore doubly
motivated — they are both classification-ambiguous and likely correct-behavior turns.
This strengthens the denominator design, but it also means that if the delegation rate
is computed only on clearly generation-shaped turns, the metric is most sensitive to
the clean first-turn generation case, which may overrepresent the favorable regime
relative to production traffic.

**3. Context-growth delegation persistence (source: Spike ψ′ C arms + methods-review
P2-C).**
The research methods review raised context-growth as a gap: "No arm tests V3 beyond a
single appended follow-on task." The Arm C results (all three forms 5/5 at depth 3
tool-result pairs) are recorded in the ADR, but the depth-3 ceiling is mentioned only
in the Consequences §Negative entry. The methods-review's specific concern — that V3's
adjacency advantage may diminish as conversation context grows and tool results
accumulate — is not surfaced as a production concern, even though the production meter
would only detect it indirectly (a declining rate over session depth). This framing
was available but not featured.

### Question 3: What would change if the dominant framing were inverted?

The ADR's dominant framing is **"delegation is won, not coerced — win it by composition."**
The inversion: **"delegation is fragile, not robust — the win is environment-dependent."**

Under the inverted framing:
- The 55/55 result becomes evidence of a narrow operating window (one model, one client,
  one depth, controlled replay context), not a broadly validated technique.
- The FC (directive-in-user-turn) becomes a commitment to maintain a property whose
  reliability is unknown outside the measured envelope.
- The production meter becomes the primary safety mechanism for a fragile property,
  not a regression-visibility instrument for a robust one.
- The "won, not coerced" framing — which presents the composition choice positively
  as pragmatic adaptation — would become "fragile, not structural," highlighting that
  the system has no escalation path short of model-swap when V3 fails on a new model
  or after a client-prompt update.

The inverted framing would strengthen the case for the detect-and-retry escalation
path (Decision 5) as a near-term need rather than a deferred PLAY concern, and would
weaken the "no evidence it would fire often enough to justify its complexity" rationale
for not building it. It would also make the portability failure boundary (two of two
tested models failing) more salient than the validated model's success.

The ADR does not need to adopt the inverted framing, but it should acknowledge that
the "won, not coerced" characterization applies to the validated environment and not
universally — a nuance the revised ADR's scope statement (Decision 2) goes some way
toward addressing.

### Framing Issues

**P2-F1 (CARRY-OVER — not applied):** The ψ.4c empty-response finding (tool-list
restriction to `invoke_ensemble` alone produced 0/10 with all empty responses — the
turn breaks rather than complies) has an implication the ADR does not draw: the
empty-response failure mode constrains the detect-and-retry escalation path (Decision 5).
If a retry were implemented, a retry that narrowed the tool list as part of the re-prompt
would be contraindicated by this finding. The ADR names detect-and-retry as "architecturally
available at the same composition point" without noting that one naive implementation
shape (tool-list restriction) is already measured to break the turn. The absence leaves
the implementation space incompletely bounded. The practitioner elected not to apply this
finding at R1; it remains available for the gate.

**P2-F2 (CARRY-OVER — not applied):** The portability failure boundary is uncharacterized.
Arm D produced qwen3.5:9b at 1/5 and mistral-nemo:12b at 2/5, but the source material
does not characterize *why* these models fail (instruction-following deficit at the smaller
tier? quantization artifact? different role-proximity weighting?). The ADR's FC
(profile-swap re-validation) is the correct operational response, but without a
portability failure model, re-validation is uninformative diagnosis: a failing re-validation
result names the failure without guiding remediation. The practitioner elected not to apply
this finding at R1; it remains available for the gate.

**P3-F2 (CARRY-OVER — not applied):** The "won, not coerced" framing in the Decision
statement is accurate as a description of the mechanism but does not surface the
environment-conditionality of the win. A brief parenthetical — e.g., "won, not coerced,
on the validated stack" — would prevent the reader from reading the decision statement
as a claim about model behavior universally. Low-stakes; the practitioner elected not to
apply at R1.

---

## Convergence-Saturation Signal (ADR-094)

**Convergence-Saturation Signal:** NOT TRIGGERED

- **Round number:** R2
- **P1 count this round:** 0 (Section 1: 0; Section 2: 0)
- **P2 count this round (new, non-carry-over):** 1 (Section 1 P2-1: soak-window
  cross-context equivalence claim introduced by the P2-4 revision)
- **New framings or claim-scope expansions this round:** The R2 audit surfaced the
  cross-context equivalence between controlled-replay sample size and production
  soak-window size as a new implicit claim (Section 1 P2-1). This is a narrow scope
  expansion on the P2-4 revision, not a framing shift. The framing audit's three
  alternative framings (portability-as-design-risk, denominator-brittleness,
  environment-conditional-win) are all available from source material visible to R1;
  P2-F1, P2-F2, and P3-F2 are carry-overs, not new framings.
- **Recommendation:** CONTINUE to R3

The signal does not trigger because P2 count (new, non-carry-over) = 1, which exceeds
the ≤1 threshold for triggering — at exactly 1 new P2 the condition is met, but the
criterion requires the round have "no new framings or claim-scope expansions" and
the Section 1 P2-1 finding represents a new scope characterization (the cross-context
soak-window claim) introduced by a prior-round correction. The practitioner should
evaluate whether P2-1 warrants a targeted repair before R3 or whether R3 proceeds
with P2-1 carried.

*Single-purpose re-audits omit this section. Form-change events reset the round-count
baseline — the first audit on a new form is its R1.*

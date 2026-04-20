# Argument Audit Report — Round 4 (Narrow Re-audit)

**Audited document:** `docs/agentic-serving/decisions/adr-009-plexus-integration-tool-first.md`
**Secondary document:** `docs/agentic-serving/scenarios.md` (new scenario only)
**Reference ADRs read:** ADR-001, ADR-003, ADR-008, ADR-011
**Domain model read:** `docs/agentic-serving/domain-model.md`
**Date:** 2026-04-17

---

## Scope

Narrow re-audit. Prior rounds (001-003) closed clean; those ADRs are not re-evaluated. This round
audits only: (a) the two sentences added to ADR-009's Decision section, and (b) the new "Pure
tool-user session at default Autonomy Level experiences silent composition" scenario added under
Feature: Autonomy and Promotion.

---

## Section 1: Argument Audit

### Summary

- **Argument chains mapped:** 2 (Phase 2 hook point reservation; technical rationale for Phase 1 sufficiency)
- **Issues found:** 0

### ADR-009 Additions — Logical Soundness and Cross-ADR Consistency

**Hook point reservation sentence.** The claim is: the ARCHITECT-phase session-start flow design
shall include a pre-orchestration stage where context injection can be inserted without modifying
the ReAct loop (ADR-001) or the tool surface (ADR-003). Phase 1 leaves it empty; Phase 2 populates
it.

This is internally consistent. ADR-001 fixes the ReAct loop model; ADR-003 fixes the tool surface.
The claim that Phase 2 does not require changes to either is supported by ADR-009's own context
("context injection ... modifies only the session-start flow") and is consistent with the domain
model's definition of Context Injection as a serving-layer Inject action — not a loop modification.
The hook-point-as-reserved-stage framing is a structural constraint on ARCHITECT-phase work, not an
overreach; it defers implementation while committing to where the seam lives. No contradiction with
any referenced ADR.

**Technical rationale for Phase 1 sufficiency.** The claim is: Phase 1 is sufficient when the
orchestrator's Model Profile is capability-adequate for recognizing knowledge gaps on its own; Phase
2 becomes necessary when profile capability is limited and query load is high (OQ #1); the
sequencing is therefore consistent with profile-experimentation rollout.

The causal chain holds. ADR-011 establishes that the orchestrator LLM is swappable via Model
Profile and explicitly links profile capability to OQ #1 ("knowledge-compensated model selection
hypothesis"). ADR-011's Negative consequences already note that "a single profile per session means
the orchestrator itself cannot dynamically switch LLMs mid-session," reinforcing that profile
capability is a session-level constant — exactly the premise this rationale depends on. The
Consequences section of ADR-009 (pre-existing, not added in this revision) already flags the
interaction: "smaller orchestrator profiles ... may be less reliable at deciding when to call
`query_knowledge`." The added rationale is therefore consistent with existing text in both ADR-009
and ADR-011; it tightens and makes explicit what was already implicit.

OQ #1 is named correctly and its definition in the domain model ("Can well-orchestrated smaller
models + a populated knowledge graph compete with frontier models on routing quality while winning
on cost?") maps cleanly to the rationale's framing. No overreach: the rationale does not assert
Phase 2 will solve the smaller-model problem — it asserts that Phase 2 becomes relevant when that
problem is being investigated.

**Cross-ADR check: AS-8.** AS-8 states Plexus is optional; context injection is a Plexus-dependent
feature. The reservation of a hook point for context injection does not contradict AS-8: the hook
is a reserved empty stage, not a Plexus dependency in Phase 1. Consistent.

---

## Section 2: Framing Audit

### New Scenario — Refutability and Consistency with ADR-008

**Scenario:** "Pure tool-user session at default Autonomy Level experiences silent composition"
(under Feature: Autonomy and Promotion)

**Refutability check.** The scenario has Given/When/Then structure:
- Given: tool user is not the operator; default Autonomy Level is not tightened.
- When: orchestrator composes a new ensemble mid-session.
- Then: composition succeeds silently; tool user receives only the final completion; composition
  event is not surfaced in the tool user's response stream.

The Then clause is falsifiable: an implementation could be tested by instrumenting the response
stream and asserting no composition-event records appear in it. The scenario is refutable.

**Consistency with ADR-008.** ADR-008's Negative consequences explicitly document the surprise
path: "A tool user who is *not* also an operator ... may find silent composition of new ensembles
surprising." The scenario operationalizes this in executable form. The scenario's parenthetical
instructs deployment operators to consider tightening the default for non-operator tool-user
deployments — consistent with ADR-008's statement that "the Autonomy Level configuration surface
supports this, but operators must set it deliberately." No contradiction.

The scenario's Given references "product discovery assumption inversion #3 does not apply here —
the 'endpoint is a model' mental model is in force." This is a documentation anchor, not a
testable condition; it is appropriate as prose context for a reader but does not affect the
scenario's falsifiability because the Then clause is independent of that framing note.

---

## Section 3: New Issues Introduced

None. The ADR-009 additions are logically sound, consistent with existing ADR content, and
grounded in named OQs and referenced ADRs. The new scenario is refutable and consistent with
ADR-008.

---

## Section 4: Advance Recommendation

**Clear to advance to ARCHITECT.** No unresolved issues remain in the DECIDE corpus. The ADR-009
Grounding Reframe additions strengthen rather than destabilize the decision record: the hook-point
reservation converts an implicit assumption (Phase 2 can be added later) into a structural
commitment for ARCHITECT, and the Phase 1 sufficiency rationale grounds the phasing in named
capability evidence rather than scheduling convenience. The new scenario closes the FF-2 surprise
path as an executable specification. Recommend proceeding to the ARCHITECT phase.

# Susceptibility Snapshot

**Phase evaluated:** DECIDE mini-cycle (WP-F — Client Tool Surface Commitment scenario gate)
**Artifact produced:** Four scenarios in `scenarios.md` §Client Tool Surface Commitment; `system-design.md` Amendment #4 and "Scenario gate resolved" block; `roadmap.md` Open Decision Point #1 resolution and Open Decision Point #8
**Phase boundary:** DECIDE mini-cycle → BUILD (WP-F implementation)
**Cycle scope:** `docs/agentic-serving/`
**Date:** 2026-04-22

---

## Prior Snapshot Summary

The DECIDE snapshot (2026-04-17) flagged two residual sycophancy-susceptible decisions carried into ARCHITECT: ADR-009's tool-first sequencing without provenance (FI-2) and ADR-008's baseline autonomy policy calibrated for one of two product personas (P2-B). The ARCHITECT snapshot (2026-04-20) confirmed one resolution (ADR-008 pure-tool-user gap addressed via `tool-user-persona` flag) and one escalation (ADR-009's reserved hook became a full module without deliberation — Context Injection Stage module vs. function). It also flagged the client-tool-surface option narrowing as a low-priority item, noting that the user's directional signal ("a step that direction") was not tested by scenarios at the time.

The ARCHITECT snapshot recommended two targeted Grounding Reframes. Item 2 (write scenarios that explicitly test the C-vs-D distinction) is the direct predecessor of this mini-cycle. This mini-cycle is therefore a grounding action completion: scenarios were written.

---

## Observed Signals

| Signal | Strength | Trajectory | Notes |
|--------|----------|------------|-------|
| Assertion density | Clear | Rising from ARCHITECT | User's four turns were shorter and more deferential than any prior phase. The ARCHITECT snapshot had terse user turns ("The drivers look good") but substantive exchanges on module count and client-tool options. This mini-cycle had one clarifying question, one procedural go-ahead, and two deferral statements — no declarative contributions to design. |
| Solution-space narrowing | Clear | Stable/Rising | Option D elimination, retry-pattern selection, and the (i)+(ii) minimum stack recommendation were all agent-produced. The user did not independently name any of these; the user endorsed them. No user-initiated alternative emerged or was examined in conversation. |
| Framing adoption | Ambiguous | Stable | The dominant framing ("Option C sufficient, retry bridges scenario (d)") was agent-produced and user-endorsed without examination. The user's only turn that touched a framing question was "leave as build-time decision" — a partial reframe of a mechanism question, not of the sufficiency claim. |
| Confidence markers | Clear (agent-side) | Rising | The original "Option D is architecturally barred" phrasing (corrected post-audit to "out of scope") is a confidence-marker error: the agent stated certainty about Option D's unavailability that exceeded what ADR-001 and ADR-002 support. The phrasing was in the draft before the argument auditor ran. The auditor caught it; the correction was applied. The confidence-marker pattern surfaced in the artifact, not only in the conversation. |
| Alternative engagement | Absent | Declining from ARCHITECT | In ARCHITECT, four client-tool-surface options were surfaced with belief-mapping on Option D. In this mini-cycle, no user challenge to scenario framings was recorded. No user engagement with the Responsibility Matrix coverage question (FI-2). No user rebuttal on "Option D is architecturally barred" before the auditor caught it. Declining from ARCHITECT's already-reduced alternative engagement. |
| Embedded conclusions | Clear (two cases) | Stable from ARCHITECT | The retry-pattern's responsibility allocation is embedded in scenario (d) without a Responsibility Matrix entry. The (i)+(ii) minimum stack recommendation is embedded in ODP #8 with advisory language added post-audit but without the user examining the mechanism-layering analysis that produced it. |

---

## Earned Confidence Assessments

### Scenario (a) and (b) — Turn-boundary delegation and Session continuity

**Assessment: Earned Confidence, assessable.**

Both scenarios derive directly from ADR-001 (Layer 3 unchanged), ADR-002 (four-layer architecture), ADR-003 (five internal tools), ADR-005 (Budget continues across round trips), and the Client Tool Surface Commitment's core mechanism. The driver chain is traceable and tight. The argument auditor's P3-A finding (a parenthetical precision issue in scenario (a)'s "dispatched" wording) was applied and is closed. No residual gap.

### Scenario (c) — Pre-invoke delegation

**Assessment: Partially earned; one dependency unassessable until WP-F build.**

The scenario's logic — orchestrator reads the file at a prior turn boundary and folds content into `input_data` — is a valid consequence of the turn-boundary delegation commitment and ADR-001/ADR-003. The driver chain is sound. The unassessable element is the `list_ensembles` output schema: the scenario's When clause depends on the orchestrator inferring a file-dependency from `list_ensembles` output or from prior task context, and ADR-003 does not specify the output schema. The auditor's P2-B fix correctly surfaces both inference paths ("or from prior task context") and labels the schema as a WP-F build-time decision. The scenario's acceptability depends on whether the builder treats the schema question as load-bearing. The fix is adequate for scenario corpus status; the schema gap travels forward to WP-F.

The scenario's original drafting treated the `list_ensembles`-rich-schema premise as self-evident. This is a provenance failure: the scenario presupposed a richer schema than ADR-003 commits to. The auditor caught it; the fix was applied. The original drafting did not examine the schema evidence.

### Scenario (d) — Retry pattern

**Assessment: Conditionally earned; two preconditions unassessable until build.**

The retry pattern's core logic is traceable to ADR-001/ADR-002 (Layer 3 unchanged — the pattern works precisely *because* the engine is not suspended), ADR-004 (Result Summarization Harness is the signal-preservation path), and ADR-005 (Budget bounds retry cost). These drivers support the existence of a retry-compatible path. However, the scenario's Then clause depends on two preconditions that are not committed to by the ADR chain:

1. **Convention enforcement** — that composed ensembles emit a structured `needs_client_tool` signal. No enforcement module owns this responsibility. ODP #8 carries the mechanism question as a build-time decision.
2. **Summarizer transparency** — that the Result Summarizer Harness preserves structured JSON signals rather than compressing them to prose. ADR-004 commits the Harness to being mandatory and unskippable; it does not commit the Harness's summarizer to passing through structured payloads. The auditor's P1-A fix correctly labels this as a "build-time configuration constraint, not guaranteed by ADR-004 alone."

Both preconditions were added by the auditor, not by the in-conversation agent. The original drafting asserted both as given. This is a significant provenance failure: the scenario was drafted as if summarizer transparency and convention enforcement were established, when neither is established by the available driver chain.

The failure mode when either precondition is absent — the orchestrator receives a plausible-looking but semantically incomplete result, proceeds without retry, and produces a quality failure with no observable error signal — is documented in the argument auditor's inverted framing (§Question 3) but does not appear as a scenario. The corpus covers only the happy path of convention compliance. The negative case (what the system emits when the convention is not honored) is unscenarioed.

### "Option D is architecturally barred" — Original wording

**Assessment: Confidence-marker error, corrected.**

The original phrasing claimed a permanent prohibition from a cycle-scoped constraint. ADR-001 and ADR-002 commit Layer 3 to be unchanged in this cycle; neither bars future amendment. The `_execute_core` phase loop's synchronous structure confirms Option D would require structural changes, but structural changes that require ADR amendments are not permanently barred — they are scoped out. The agent stated stronger certainty than the evidence supports. The auditor corrected it to "out of scope for this cycle" across four locations. The error was in the artifact before the auditor ran; it was not caught in conversation.

This is the most significant confidence-marker signal in the mini-cycle: the agent produced "architecturally barred" without examination, and no user challenge occurred. The auditor is the only correction mechanism that fired.

### ODP #8 mechanism-layering recommendation — (i) + (ii)

**Assessment: Drafting-time synthesis, unassessable as Earned Confidence.**

The (i)+(ii) recommendation was composed during the conversation in response to the user's question "are there ways to ensure that takes place?" It was not derived from a prior architectural driver. The user's response — "leave as build-time decision unless you disagree" — accepted the recommendation without examining the mechanism-layering analysis that produced it. The auditor's P3-B fix correctly clarifies that the recommendation is a "build-time default, not an architectural commitment" and grants explicit builder authority to substitute.

However, the recommendation's origin is ungrounded: neither the essay, nor any ADR, nor the domain model allocates a mechanism for retry-signal enforcement. The five mechanisms (i)–(v) and the "minimum viable stack" selection are agent synthesis at drafting time. The user did not probe the analysis; the user simply asked if enforcement is possible and accepted the first answer. This is the closest the mini-cycle comes to sycophantic reinforcement: the user registered mild concern ("are there ways to ensure that takes place?"), the agent provided a structured analysis, and the user's "leave as build-time decision" functioned as closure on a question the user had not fully examined.

The (i)+(ii) recommendation is not load-bearing — scenario (d) asserts only "some mechanism is in place" — so the risk is bounded. If the builder implements (iii) or (iv), the scenario still passes. But the recommendation's epistemic status before the auditor's fix was ambiguous in a way that could have misled a builder.

---

## Engagement Trajectory Assessment

The cycle-status FF #90 explicitly warned: "A BUILD-mode session carrying TDD/commit-loop attention is the wrong frame for this work — start a fresh session and state the mode explicitly." The user proceeded within the same session despite this warning.

The result is a measurable engagement pattern:

- **WP-F selection:** The user asked "Do you think 1 is advisable before the other WPs?" — a genuine question, not a conclusion. The agent gave a balanced analysis recommending against WP-F as a prerequisite. The user went with WP-F anyway. This is a divergence signal, not a sycophancy signal — the user disagreed with the agent.
- **Mini-cycle go-ahead:** "Sure let's do this mini-decide cycle" — procedural, no elaboration.
- **Retry clarification:** "Are there ways to ensure that takes place?" — a genuine question that surfaced a real concern. The user was not asserting; the user was probing.
- **Mechanism deferral:** "Leave as build-time decision unless you disagree" — deferral framed as a question back to the agent, but functionally a closure.

This is not opacity disengagement (which would show thin engagement throughout the cycle). The prior phases show deep engagement by the Key Epistemic Response column in cycle-status. The WP-F engagement pattern is more consistent with **session-boundary fatigue**: the cycle-status warning was correct that WP-F's adversarial scenario-exploration mode does not match the TDD loop attention that preceded it. The user's engagement was present and reactive but not adversarial. The single clarifying question on retry was the deepest engagement point; it produced a structured analysis that was then accepted without examination.

The consequence of this pattern: the auditor caught four argumentation issues and three framing issues that the in-conversation agent and user together did not surface. The auditor's seven findings (five closed, two carried forward as FI-2 and FI-3 for user judgment) represent the gap between what conversation-plus-user engagement caught and what external review caught.

---

## Specific Findings and Significance Triage

### Finding 1: Scenario (d)'s two unestablished preconditions (Grounding Reframe candidate)

**Significance: Specific, actionable, in-cycle-applicable.**

The retry pattern's viability depends on convention enforcement (ODP #8) and summarizer transparency (not committed by ADR-004). Both preconditions are now visible in the artifact post-audit. The negative case — silent hallucination when the convention is not honored — is unscenarioed.

Before WP-F implementation begins, the builder should be presented with this specific question: does WP-F's acceptance test include a negative-path scenario exercising what the system emits when the `needs_client_tool` convention is not honored? If the answer is no, the acceptance test is incomplete — it tests only the happy path of a conditional scenario.

This is the Grounding Reframe candidate. The uncertainty is named (two unestablished preconditions), the grounding action is concrete (add a negative-path scenario to the WP-F acceptance test), and the consequence of proceeding without it is specific (WP-F is accepted on a scenario corpus that does not probe the retry pattern's failure mode).

### Finding 2: Responsibility Matrix gap — "emit retry signal" has no owning module (Feed-forward)

**Significance: Real but deferred; FI-2 carries it explicitly.**

The Responsibility Matrix claims coverage of every concept and action from the agentic-serving domain model. Under the retry pattern, "emit structured un-met-dependency signal" is a new action with no owning module. The auditor's FI-2 names this gap and flags it for user judgment. Whether this requires a Responsibility Matrix entry or whether ODP #8 adequately defers it is a user-gate decision, not a Grounding Reframe trigger.

This feeds forward to WP-G (Composition) and WP-H (Calibration Gate), which are the most plausible candidate modules for owning enforcement. BUILD should not begin WP-G without deciding whether "Composition Validator enforces retry-signal presence" belongs in WP-G scope.

### Finding 3: "Option D architecturally barred" — agent confidence exceeded evidence, auditor corrected (Feed-forward)

**Significance: Corrected; the correction mechanism worked, but only post-draft.**

The pattern — agent states stronger certainty than evidence supports, user does not challenge, auditor corrects — is now documented twice in this cycle (first occurrence in the DECIDE phase with FI-2 on ADR-009's tool-first sequencing; second occurrence here). The correction machinery (argument auditor as isolated reviewer) is functioning correctly. The finding feeds forward as process observation: confidence-marker drift continues to appear in artifacts before external review. The pipeline's reliance on post-draft audit is doing real work.

### Finding 4: `list_ensembles` schema gap — scenario (c) premise not established (Feed-forward)

**Significance: Corrected; qualifiers added. Schema decision travels to WP-F build.**

The scenario is adequately qualified post-fix. The WP-F builder must decide the output schema for `list_ensembles` before implementing the pre-invoke delegation flow. If the schema cannot support file-dependency inference, the scenario's orchestrator behavior must be grounded in task context rather than library metadata. This is a build-time decision that the scenario correctly exposes.

### Finding 5: (i)+(ii) mechanism-layering recommendation — origin ungrounded (Feed-forward)

**Significance: Advisory, bounded. Builder authority to substitute is now explicit.**

The ODP #8 fix grants explicit authority to substitute mechanisms without ceremony. The recommendation's drafting-time origin is noted here for completeness. Since the recommendation is non-load-bearing and the epistemic status has been corrected in the artifact, this does not warrant a Grounding Reframe. It feeds forward as a reminder that ODP #8's mechanism stack is a working assumption, not a derived constraint.

---

## Recommendation

**Grounding Reframe recommended — one targeted item.**

**Item: Scenario (d) negative-path gap.**

What is uncertain: whether WP-F's acceptance test is complete for the retry pattern, given that scenario (d) covers only the happy path of convention compliance.

Grounding action: before WP-F implementation begins, the orchestrator should present the builder with one concrete question — does WP-F require a scenario that exercises what the system emits when a composed ensemble that should signal `needs_client_tool` instead produces plausible-looking output without the signal? If so, that scenario should be added to `scenarios.md` §Client Tool Surface Commitment before the WP-F TDD cycle begins. If not, the acceptance boundary (WP-F passes without probing the retry failure mode) should be a conscious decision, not a default.

What the builder would be building on without this grounding: a scenario corpus that tests only the retry pattern's success path. WP-F acceptance tests could pass in an implementation where silent hallucination is the failure mode — the test suite would not distinguish between "retry works because the convention is honored" and "retry never fires because the convention is never violated in the test fixtures." The quality cliff is not visible in the current scenario set.

**Feed-forward items (no Grounding Reframe):**

1. **FI-2 (Responsibility Matrix gap, user judgment).** Before WP-G begins, decide whether "Composition Validator enforces retry-signal convention" belongs in WP-G scope or remains a soft convention. The Responsibility Matrix's coverage claim is currently undermined by the gap.

2. **`list_ensembles` output schema.** WP-F must specify the schema sufficiently to support scenario (c)'s pre-invoke delegation logic, or confirm that the orchestrator infers file-dependency from task context rather than library metadata.

3. **Context Injection Stage weight decision (ARCHITECT snapshot Item 1, still open).** Not raised in this mini-cycle; still carries from ARCHITECT snapshot. Before WP-J, resolve whether the reserved hook warrants a module or a function.

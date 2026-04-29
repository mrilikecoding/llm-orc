# Research Design Review

**Reviewed question set:** `docs/agentic-serving/essays/research-logs/research-log.md` (five RQs post-amendment) + constraint-removal response for ADR-003
**Constraint-removal response included:** Yes (ADR-003 bracketed; response recorded in the research log)
**Date:** 2026-04-25

---

## Summary

- **Questions reviewed:** 5 (RQ-1 through RQ-5; RQ-5 reviewed within its declared frame as a framing question)
- **Flags raised:** 6 (3 × P1, 2 × P2, 1 × P3)
- **Criteria applied:** 1–4 (ADR-082 corpus; all four criteria active)

**Overall assessment:** The question set is substantively well-formed. The constraint-removal response does real work — it brackets ADR-003 genuinely and surfaces ensemble affordances as a gap the original questions under-named. Two embedded conclusions require attention before the research proceeds: one in RQ-3 that forecloses a hypothesis the field has not settled, and one in RQ-4 that presupposes a policy decision that is itself an open question. The most consequential flag is the incongruity-surfacing gap: S0 is framed as a hypothesis-distinguishing spike, but its prediction structure may still be angled toward confirming the architecture's complexity rather than genuinely testing the simpler alternative. If S0 runs with that posture, the cycle risks a large investment in architecture-shape questions that a stronger model would have made unnecessary.

---

## Per-Question Review

### RQ-1: "What is the empirical capability floor for an orchestrator driving llm-orc's cascading-tool model, and how does it vary with (a) model class, (b) tool-surface composition, and (c) ensemble affordances?"

**Belief-mapping:** This question treats capability as a floor property of the (model × surface × affordances) tuple. What would the researcher need to believe for a more productive question? They would need to believe that the cascade architecture itself could be the binding constraint — that no configuration of (model × surface × affordances) produces reliable outcomes at the cost point the project contract requires. That belief points toward a different question: "Is cascading-tool orchestration the right architecture for achieving llm-orc's project contract, or does the capability floor evidence suggest a flatter model?" The current framing accepts the cascade architecture and maps its floor. The adjacent question accepts the floor finding as potential evidence against the architecture.

The amended phrasing — admitting tool surface and affordances as variables rather than fixing them — is a meaningful improvement over the original. The dimension structure (a/b/c) is appropriate and non-redundant.

**Embedded conclusions:** None material. "llm-orc's cascading-tool model" names the research object rather than presupposing a design choice. The question does not presuppose the cascade is correct — it empirically characterizes it.

**Scope:** Appropriate for an empirical question at this stage. The three dimensions are well-delimited. The question is broad enough to be illuminating and narrow enough to be answerable by a spike series. One note: "ensemble affordances" remains loosely bounded — what the orchestrator is told about ensemble capabilities vs. what ensembles actually declare vs. what composability primitives exist are distinct sub-questions. The research plan's literature scan on ensemble affordance design suggests this is already on the practitioner's radar. No reformulation required; awareness noted.

---

### RQ-2: "What detectable signals (pre-session, session-start, first-turn) reliably distinguish a profile that meets the floor from one that doesn't — and how can the system communicate what's missing in a form a user can act on?"

**Belief-mapping:** This question assumes that reliable signals exist at the named time horizons (pre-session, session-start, first-turn) and that the gap between a profile that meets the floor and one that doesn't is communicable in actionable form. What would the researcher need to believe for a different question to be more productive? They would need to believe either that the signals are irreducibly noisy at these horizons (pointing toward "should the system attempt detection at all, or simply require explicit operator configuration?") or that the unit of honest communication is not the signal-to-user path but the operator-to-configuration path (pointing toward "who owns the floor specification — the system or the operator?"). The current framing assumes the system can detect and communicate. The adjacent question is whether honest communication requires detection at runtime, or whether it requires honest defaults.

**Embedded conclusions:** P2 flag. "Reliably distinguish" embeds a presupposition that reliable distinction is achievable at these time horizons. The field evidence on self-knowledge and calibration in small models (which the research plan cites as a literature target) may come back negative — small models may not provide reliable capability signals even at first-turn. If that evidence arrives, the question as written has already foreclosed the possibility that the answer is "not reliably distinguishable; the system should require explicit capability attestation instead."

Suggested reformulation: "What signals — if any — are available at pre-session, session-start, or first-turn to distinguish a profile that meets the floor from one that doesn't, and what are the limits of those signals? Where signals are unreliable, what honest-default alternatives are available?"

This is P2, not P1, because the research plan explicitly targets self-knowledge calibration literature. The scope for the finding to come back negative is preserved in the research design; the question phrasing just doesn't reflect that openness. Fixing the phrasing reduces the risk of the practitioner anchoring to signal-reliability as a given before the literature is read.

**Scope:** Slightly narrow — the question focuses on user-facing communication but does not separately address operator-facing configuration (who sets the floor, and how). That gap is partly covered by RQ-4, but the hand-off between the two questions is not explicit.

---

### RQ-3: "What observability surfaces support diagnostic truthfulness without imposing coordination burden — and how do those two purposes diverge in practice for tool user (in-stream) and operator (server-side)?"

**Belief-mapping:** This question is framed around the Beck framing: visibility-as-coordination-display is the failure mode; visibility-as-diagnostic-truthfulness is the alternative. What would the researcher need to believe for a different question to be more productive? They would need to believe that diagnostic truthfulness and coordination burden are not always separable — that some degree of coordination burden is the price of diagnostic truthfulness in a failing session. The field notes surfaced exactly this: in a working session, the Pure Tool User does not want visibility; in a failing session, the absence of any signal made the gap intolerable. The Beck framing resolves this as a design principle, but whether the two goals are empirically separable in a system with this capability floor is an open question the field notes name explicitly.

**Embedded conclusions:** P1 flag. "Without imposing coordination burden" presupposes that the two goals can be reconciled — that there exists a surface design that delivers diagnostic truthfulness without coordination overhead. The field notes offer a more nuanced finding: in a failing session, the tool user *wanted* coordination-style visibility specifically because there were no diagnostic signals. The question as written forecloses the hypothesis that the correct resolution is "tell the user upfront that this session may be degraded and give them an exit" rather than "design a surface that provides diagnostic truth without becoming a coordination display."

Suggested reformulation: "Under what conditions are diagnostic truthfulness and coordination burden separable design goals, and what observability surface designs serve each goal without collapsing them — including the case where they cannot be separated without a capability gate?"

The field note observation (note 8 in the field notes) is directly relevant: "Presumably a better or more competent configuration would not have led me to want more observability." This suggests the correct resolution may be upstream — capability gates over visible observability design — and the current question assumes you reach the visibility question at all.

**Scope:** Too narrow in one direction: the question asks how the two purposes diverge for tool user vs. operator but does not ask whether the in-stream / server-side split is the right split. The field notes suggest a third audience: the system itself (during session-start narration) that could carry diagnostic signal before either the tool user or the operator observes a failure. This is a coverage gap, not a fatal flaw.

---

### RQ-4: "What default configuration ships with llm-orc to make the OpenCode-as-default-client first session honest about its own competence?"

**Belief-mapping:** This question takes "defaults ship" as a given and asks what those defaults look like. What would the researcher need to believe for a different question to be more productive? They would need to believe that defaults are not the right mechanism — that honest first-session experience requires operator configuration rather than a packaged default, or that "honesty about competence" is not achievable through configuration at all (only through capability gates that refuse to start an incompetent session). The adjacent question: "Should llm-orc ship a default configuration, an onboarding gate, or a capability attestation requirement — and how does the choice affect the user contract?"

**Embedded conclusions:** P1 flag. "What default configuration ships" presupposes that (a) defaults are the right mechanism, and (b) a default configuration can make a session honest about its own competence. The PLAY findings challenge both presuppositions directly: the encountered session ran, produced a completely broken experience, and terminated on a budget message the user had no context for. No default configuration resolves this if the underlying orchestrator profile is not capable — the question of what defaults to ship may be downstream of whether defaults are the right intervention level at all.

Suggested reformulation: "What is the minimum intervention that makes the OpenCode-as-default-client first session honest about its own competence — and is that intervention best located at configuration (defaults), at capability detection (gates), or at operator policy (explicit attestation)?"

The research plan does include "whether defaults ship at all" as part of RQ-4's scope (the parenthetical in the log: "Includes both whether defaults ship at all (a policy question) and what defaults look like if they do (a design question)"). That softens this flag somewhat — the practitioner has the openness. The question phrasing should reflect it, not bury it in a parenthetical that may not survive forward into the essay.

**Scope:** Appropriate once the policy question is given equal weight with the design question. The risk is the essay phase anchoring to the design question because the question phrasing foregrounds it.

---

### RQ-5: "How does llm-orc reconcile its dual contracts — user-facing outcome and project-facing method — and where the two contracts produce different design choices, how is the seam navigated?"

*RQ-5 is a framing question, not an empirical question. It is not audited for scope or for the kind of embedded conclusions that would constrain empirical inquiry. The audit below addresses embedded conclusions and prior-art treatment within its declared frame.*

**Belief-mapping (within frame):** The dual-contract framing names a real tension. What would the researcher need to believe for a different framing question to be more productive? They would need to believe that the two contracts are not actually in tension — that a system honest about capability (user contract) is precisely the system that validates the orchestration hypothesis (project contract), because a failing session with honest signaling is a better test of the hypothesis than a passing session where it is unclear what produced the result. That belief points toward a framing question like: "Does the dual-contract tension require navigation, or does honest capability signaling collapse the seam by serving both contracts simultaneously?" The current framing assumes navigation is required. The adjacent framing asks whether the seam exists at all once the capability floor is well-characterized.

**Embedded conclusions:** P2 flag. "Where the two contracts produce different design choices" presupposes that they do produce different design choices in practice. This is likely true, but naming it as a presupposition matters because the empirical question of *where* they diverge is part of what the research is supposed to find. If the question is asked as "how is the seam navigated?" before the seam is located, the essay phase may assume a divergence in areas where none is found.

Suggested reformulation: "Do llm-orc's dual contracts — user-facing outcome and project-facing method — produce different design choices in the areas RQ-1 through RQ-4 investigate, and if so, how is the seam navigated? If they converge in those areas, what does that convergence reveal about the relationship between the two contracts?"

**Prior-art treatment:** The Beck framing (outcome over feature) was consciously engaged and partially accepted: user contract is outcome-shaped; project contract is method-shaped. The constraint-removal response explicitly resisted collapsing the two contracts into Beck's outcome stance. This is appropriate prior-art treatment within the frame. No flag.

---

## Constraint-Removal Response Review

**Response substance:** Engaged. The practitioner did not produce a null answer or a performative one. The response correctly identifies what is structurally load-bearing in ADR-003 ("a capable orchestrator with a declared set of tools it can cascade requests through") and what is not (the specific five tools). The both/and on Beck's distinction — user contract outcome-shaped, project contract method-shaped — is a substantive addition that the original RQ set did not contain and that the amended RQ-5 now reflects.

**Embedded conclusions in the response:** One flag. The response states that what is structurally load-bearing is "a capable orchestrator with a declared set of tools it can cascade requests through." This is not false, but it silently loads the cascade model as load-bearing. A response that went one layer deeper would ask: what would need to be true for the cascade model itself to be unnecessary — for example, if a sufficiently capable orchestrator model could handle all sub-tasks directly without delegation? The constraint-removal response brackets the specific tool surface but does not bracket the cascade architecture. That is an appropriate scope limit for a response focused on ADR-003; it does not invalidate the response. But it means the cascade architecture itself remains an unexamined assumption, which connects to the incongruity flag below.

---

## Question Set Assessment

### Premature narrowing / prior-art treatment

**No material premature narrowing detected.** The question set admits tool surface, model class, and ensemble affordances as variables rather than fixing the architecture. The constraint-removal response does genuine work bracketing ADR-003's specific tool surface. The three-dimension structure of RQ-1 is appropriately exploratory.

**Prior-art criterion satisfied by the constraint-removal response.** The response treats ADR-003 as prior art — it asks what would remain if the five-tool surface did not exist, and identifies what is actually load-bearing. No additional artifact-bracketing question is needed in the question set itself. Criterion 3 is satisfied.

One partial narrowing to note (not a flag, a watch item): the question set as a whole assumes the four-layer architecture (ADR-002) without bracketing it. ADR-001 and ADR-002 were closed before the PLAY phase that triggered this loop. The PLAY findings do not directly challenge the four-layer architecture, but they do suggest the capability floor is more consequential than the architecture's complexity. If S0 surfaces H1 (capability-bound), the architecture's complexity may be irrelevant to the first-session problem, and a future constraint-removal pass on ADR-001 or ADR-002 may be warranted.

### Incongruity surfacing — P1 flag

**Incongruity detected and insufficiently surfaced.**

The research context contains a clean adjacency between two solutions at different complexity levels:

- **Simple region:** Swapping the orchestrator Model Profile to a capable model (claude-sonnet, gpt-4o) is a configuration change. The PLAY field notes are explicit: the default `orchestrator-local` profile failed at the most basic tasks; a stronger profile is the obvious next experiment. The research plan names S0 as a spike to run with "a known-capable orchestrator profile."

- **Complex region:** The architecture under investigation (four layers, internal ReAct loop, fixed tool surface, ensemble affordances, observability surfaces, dual-contract reconciliation) is a substantial design space requiring significant BUILD investment.

The adjacency is: if a strong orchestrator model can handle the tasks the session required without any of the cascade architecture doing meaningful work, then the complexity being researched is either (a) not needed for the user contract, or (b) needed only for the project contract (the hypothesis that non-frontier models can deliver good results). Option (b) is real — the project contract is method-shaped. But the question set does not explicitly surface this as a testable incongruity.

**What S0 is currently positioned to do:** S0 is designed to distinguish H1 (capability-bound) from H2 (architecture-shape problems). H1 produces a narrowed subsequent scope (profile battery + signal detection + OB-2/OB-3); H2 produces the full architecture investigation. This is sound branching logic.

**What S0 is not positioned to ask:** If H1 is confirmed — if a strong orchestrator profile produces a working first session — what work is actually left? The field notes make the implicit question explicit: the Pure Tool User verdict ("I would not use this again") applies to the default configuration. Whether it changes with a stronger profile is the unanswered question. If a strong profile produces a working session and the tool user *does not* want visibility in a working session (field notes, note 8), then much of the observability and signal-detection architecture under RQ-2 and RQ-3 may be solutions to a problem that only exists with a weak profile.

**The missing question the set does not ask:** "If a strong orchestrator profile resolves the first-session failure, which of RQ-2 through RQ-4 remain necessary, and for which contract layer?"

This is not a demand to add a sixth RQ. It is a demand that S0's prediction structure include this question explicitly — and that the stopping criteria for S0 include a branch point where the practitioner asks whether the complexity of the subsequent research is still load-bearing given the H1 outcome.

**Suggested addition to S0's prediction structure:** Before S0 fires, record a third hypothesis alongside H1 and H2:

> H0: The first-session failure is entirely configuration-dependent. A strong orchestrator profile produces a working session without any of the cascade architecture doing meaningful work. Under H0, the user contract is satisfied by configuration, and RQ-2 through RQ-4 are primarily method-contract questions (relevant to the project hypothesis) rather than user-contract design questions. The research scope should be adjusted accordingly if H0 is confirmed.

Without H0 in the prediction structure, S0 confirms H1 and branches to a narrowed scope that still investigates signal detection, observability surfaces, and defaults — all of which may be answering a question that a capable profile has already resolved. The seam between "what's needed for the user contract" and "what's needed for the project contract" is exactly what RQ-5 is supposed to navigate. But RQ-5 cannot navigate it if S0's prediction structure does not surface it as a branch point.

### Coverage gaps

**Gap 1 — Token cap reconciliation.** Field notes note 7 surfaces the encountered 50K token cap as likely a misconfiguration (hypothesis (a): local config.yaml override). The research plan routes this to "RESEARCH (new question)." It is not reflected in any of the five RQs. This is appropriate — it is a housekeeping finding, not a research question. But it should be resolved before S0 fires, because an S0 run with a misconfigured token cap will produce uninterpretable results against the documented 10M default.

**Gap 2 — Session-start narration as a third observability surface.** RQ-3 frames the observability question around the in-stream / server-side split (tool user vs. operator). Field notes note 8 surfaced a third candidate: narration that fires at session-start or first-turn *before any composition events are generated*, specifically to carry signal when the session is about to be degraded. The question set does not address this surface. It is a coverage gap in RQ-3, not a fatal flaw — the research plan's OB-2/OB-3 spike targets this area — but the question framing may produce an essay that under-addresses it.

**Gap 3 — The operator's configuration ownership.** Neither RQ-2 nor RQ-4 addresses who is responsible for stating the capability floor: the system (through detection), the operator (through explicit profile attestation), or a default that ships with the product. The division of responsibility between these three is the policy question that makes RQ-2's detection design and RQ-4's defaults design either necessary or sufficient. It is partially named in RQ-4's parenthetical but is not foregrounded in any question.

### Recommendations

**P1 — Add H0 to S0's prediction structure before the spike fires.** This is the highest-priority action. Before running S0, record H0 explicitly: that the first-session failure is entirely configuration-dependent and that a strong profile resolves it without requiring the cascade architecture to do meaningful work. Include a branch: if H0 is confirmed, which of RQ-2 through RQ-4 remain necessary for the user contract vs. the project contract only? This is not a new question — it is a forcing function that the incongruity between PLAY's simple observation and the complex architecture under investigation demands.

**P1 — Reformulate RQ-3 to remove the embedded reconciliation presupposition.** "Without imposing coordination burden" forecloses a hypothesis the field notes left open: that diagnostic truthfulness and coordination burden are not separable when the orchestrator is incapable. Replace with the suggested reformulation above or equivalent language that admits "including the case where they cannot be separated without a capability gate."

**P1 — Reformulate RQ-4 to give the policy question equal billing.** "What default configuration ships" foregrounds the design question and buries the policy question in a parenthetical. The policy question (whether defaults are the right mechanism at all, vs. capability gates or operator attestation) is the more fundamental one. It should lead the question, not follow it.

**P2 — Reformulate RQ-2 to preserve openness about signal reliability.** Replace "reliably distinguish" with language that admits the possibility that reliable detection is not achievable, and names honest-default alternatives as candidate answers. See suggested reformulation above.

**P2 — Add the dual-contract divergence question to RQ-5 explicitly.** RQ-5 assumes the contracts diverge; the more productive framing also asks whether they converge in the areas the other RQs investigate, and what that convergence reveals. See suggested reformulation above.

**P3 — Resolve the token cap finding before S0.** Field note 7 routes this to RESEARCH as a new question but does not block S0. In practice, an S0 run with a 50K token cap against a strong profile will exhaust differently than the baseline PLAY session. The finding should be resolved (hypothesis (a) tested) and the S0 run should use a known cap, even if it is not the documented 10M. Unresolved, this introduces an interpretability problem in S0's results.

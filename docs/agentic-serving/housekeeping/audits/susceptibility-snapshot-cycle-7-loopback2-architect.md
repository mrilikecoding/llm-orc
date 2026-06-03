# Susceptibility Snapshot

**Phase evaluated:** ARCHITECT (Cycle 7 loop-back #2 — scoped pass; Finding D — client-tool deliverable form contract)
**Artifact produced:** system-design.md Amendment #14 (v6.1) + system-design.agents.md loop-back #2 extensions (FC-53..FC-57; FormGate seam; form-directive composition; D1 extraction fix)
**Date:** 2026-06-03

---

## Prior Snapshot Summary

The loopback #2 DECIDE snapshot (evaluated immediately before this phase) returned **No Grounding Reframe warranted** — earned confidence pattern, probe-before-draft discipline, argument audit convergence toward more conservative claims. It carried four advisories to ARCHITECT:

1. **Advisory 1 (granularity invariant — structured-multi-file probe gap):** design the across-turn loop-driver decomposition path before treating the granularity invariant as locked; if not built here, multi-file deliverables are architecturally unserved.
2. **Advisory 2 (FormGate detect-and-refuse — named interface point):** seat the backstop at the bridge as a named, non-optional interface point, not a future addition.
3. **Advisory 3 (delegation-fires precondition — diagnostic disambiguation):** instrumentation should distinguish form-compliance failures from delegation-not-firing failures.
4. **Advisory 4 (directive injection placement — F-χ.3 synthesizer-timeout path):** make the injection point explicit in the dispatch-input composition interface.

The phase ran in the same extended session as the DECIDE phase on 2026-06-03.

---

## Observed Signals

| Signal | Strength | Trajectory | Notes |
|--------|----------|------------|-------|
| Assertion density | Absent | Declining (positive) | Two practitioner responses in the phase. Neither is a declarative conclusion about the design. The first is a genuine counter-probe ("What's the drawback of harder enforcement?"); the second is a substantive reframe (seam framing) plus a routing decision. No "clearly/obviously" markers. The scoped mechanical character (allocation of an already-deliberated ADR) is a plausible benign explanation for brevity. |
| Solution-space narrowing | Ambiguous — requires specific assessment (below) | Stable relative to DECIDE snapshot | Two narrowing decisions: (a) FormGate seat allocated to the Artifact Bridge (the inherited DECIDE advisory 2 choice) — see assessment below; (b) no-new-module resolution for the granularity invariant — see assessment below. Neither is unexplained narrowing; both are explicitly rationalized and tied to prior-corpus drivers. |
| Framing adoption | Ambiguous — requires specific assessment (below) | Stable | The DECIDE-phase seam framing (practitioner-originated, gate-exchange) was already incorporated into ADR-035 before the ARCHITECT phase began; it propagates into Amendment #14 as explicit design scope-limiting language ("semantic-coherence seam explicitly out of scope"). This is forward-propagation of an audited artifact, not fresh absorption. |
| Confidence markers | Absent | Declining (positive) | Amendment #14 retains honest-residual language throughout: "optional conservative single-fence normalization is a BUILD open choice," "where-sub-fork executor-side-vs-envelope-side is a BUILD scenario-group decision," "trajectory-scale form compliance non-decomposable into a static FC — ADR-097 Conditional Acceptance PLAY target." No language upgrades occurred between DECIDE and ARCHITECT. |
| Alternative engagement | Absent as new engagement; prior engagement was substantive | Stable | No alternatives were explored in this phase — by design, the phase is scoped to allocation, not design. The alternative (FormGate at the Terminal vs. the Bridge) was engaged at the gate with a warrant-elicitation question; the practitioner delegated the answer back to the agent. Assessment of whether that delegation was adequate is the core question for this snapshot (see below). |
| Embedded conclusions at artifact-production moments | Ambiguous | Improving relative to prior snapshot | Amendment #14's provenance chain cites ADR-035 decisions by number, named advisories, and specific spike findings for each module extension. FC-57 is explicitly annotated as satisfying "DECIDE loop-back #2 snapshot advisory 2." No embedded conclusions appear without a named warrant. The one concern — the agent-derived settled/open partition from DECIDE carried forward without practitioner re-derivation — propagates into ARCHITECT unchanged, but the gate reflection records it as "proposed-and-ratified-by-proceeding" rather than obscuring it. |

---

## Pattern-Specific Assessments

### FormGate-seat allocation: examined inheritance or automatic carry?

The DECIDE snapshot advisory 2 recommended seating the detect-and-refuse gate at the Artifact Bridge ("named interface point — not an optional future addition"). ADR-035 decision 4 hedged: "the Artifact Bridge MAY apply" conservative normalization. The ARCHITECT phase committed the Bridge seat fully (FormGate seam on `marshal()`; `destination_tool` threaded from Terminal; detect-and-refuse installs at the seam on PLAY evidence with zero Terminal edits — FC-57).

The gate question examined this: the agent posed the Artifact Bridge vs. Client-Tool-Action Terminal alternative explicitly before committing. The practitioner declined to reason through it and redirected: "You tell me based on our north star what makes the most sense?" The agent committed to the Bridge seat with three north-star-grounded reasons:

1. **Layer-split diagnosability for axis-2 traces** — FC-51 `TurnDecision` events distinguish wrong-form from wrong-action at the bridge (the only point where marshalled content is visible against the destination tool); gating at the Terminal would require passing inspection logic up a layer.
2. **Refusal/re-dispatch is dispatch-stratum work** — seating at the Terminal would create an upward L3→L2 control edge (the Terminal dispatching a re-dispatch is an upward control flow violation in the existing layering). The Bridge, as L2, can trigger a dispatch-failure completion without crossing a layer boundary.
3. **Form rules accrete at the content locus** — as destinations widen (`edit`, `bash`), the keying logic lives naturally where `destination_tool` arrives (the marshalling boundary), not at the emission layer which already owns the `tool_calls` shape.

**Driver-derivation check:** All three reasons are grounded in prior-corpus drivers. Reason 1 cites FC-51, which is an existing fitness criterion from Amendment #12 with named instrumentation (`TurnDecision` events). Reason 2 cites the layering rules that are the system design's cycle-across invariants — the L3→L2 upward control edge argument is the same reasoning that seated the Loop Driver at L2 rather than L3 (Amendment #12 founding decision). Reason 3 cites ADR-033 F3-1's across-turn composition as the widening mechanism and implicitly references the agent's own destination-agnostic library principle (ADR-025). These are not ad hoc post-hoc rationalizations of an inherited choice; they are applications of standing layering rules to a new allocation question. The alternative (Terminal seat) would have required the terminal to own logic that belongs upstream of emission — a violation of a structural invariant, not merely a stylistic preference.

The honest cost (one threaded `destination_tool` field on the Terminal→Bridge shared-type edge) was named, as was the condition under which the Terminal seat would be correct ("if the Terminal were promoted to L2 and the Bridge's content-read responsibility moved inside it — a module merge that is not warranted by this decision"). This is the anti-sycophancy check: the agent named the path under which the rejected alternative would be right.

**Assessment: examined inheritance, not automatic carry.** The agent-answers-its-own-gate-question pattern is consistent with the cycle's recorded positive precedent (first loop-back ARCHITECT gate) and the practitioner's stated outcome-based-over-speculation preference. The three reasons are derived from prior-corpus structural invariants, not post-hoc rationalization. The threaded-field cost and the condition under which the Terminal seat would win are both named.

### No-new-module for the granularity invariant: parsimony or under-design?

DECIDE snapshot advisory 1 asked ARCHITECT to "confirm that the across-turn loop-driver decomposition path is architecturally designed before treating the granularity invariant as locked." The pass resolved this as: the granularity invariant's architectural home is existing structure — Single-Step Enforcer guarantees one tool call per turn structurally (FC-43, pre-existing); the directive scopes one deliverable per dispatch contractually (FC-55, new in this pass). Multi-file decomposition across successive write-turns is callee-native (ADR-033 F3-1). PLAY validates the driver's actual multi-file decomposition behavior (axis-2). No new module.

**Is this parsimony or under-design?**

The advisory specifically asked whether the decomposition path is "architecturally designed" — the concern was that if the loop-driver's actual multi-turn decomposition behavior for multi-file work is unbuilt and unspecified, the granularity invariant leaves multi-file deliverables unserved. The resolution points to FC-43 + FC-55 as the structural and contractual housing, plus ADR-033 F3-1 as the named pattern, plus axis-2 PLAY as the validation site. What it does not do is design the loop-driver's actual decision process for multi-file decomposition: how the driver knows a task requires multiple files, how it sequences write turns, whether it reads back the prior turn's output before writing the next file. Those are behaviors, not module boundaries — and they belong in BUILD scenarios, not ARCHITECT module decomposition.

The advisory's concern was whether the path has an "architectural home." It does: the Loop Driver (L2) owns the multi-turn control structure, Single-Step Enforcer enforces one-per-turn, and the write-turn-sequencing behavior is callee-native per ADR-033. What the ARCHITECT phase did not do — and correctly did not do — is specify the behavioral logic of that sequencing. That is a BUILD scenario-group question (the loop-driver scenario block includes multi-turn write sequences). The advisory's "architecturally designed" criterion is met at the module-allocation level; the behavioral design is appropriately a BUILD artifact.

The one honest gap that carries forward: the structured-multi-file contract alternative (a JSON-array format) remains untested. The granularity invariant closes that door by design preference (ADR-033 F3-1 across-turn composition as the right shape) without spike evidence against the JSON-array alternative. This was named in the DECIDE snapshot as P2-F2 (design preference closure, not evidence closure), and the ARCHITECT pass does not add evidence. FC-55 verifies the structural + contractual housing, not that the JSON-array alternative was ruled out by evidence. The concern is accurately labeled and carried forward to PLAY.

**Assessment: parsimony, not under-design** at the module-allocation level. The advisory's criterion (architectural home) is satisfied. The behavior-level multi-file sequencing is appropriately deferred to BUILD scenarios. The structured-multi-file alternative gap remains open and correctly labeled.

### DECIDE advisories 3 and 4: addressed or silent?

Advisory 3 (delegation-fires precondition diagnostic disambiguation) was not explicitly addressed in Amendment #14's module extensions. The existing FC-51 `TurnDecision` events distinguish wrong-form from wrong-action — they were cited for the FormGate diagnosability argument. But FC-51 distinguishes wrong-form *deliverables* from wrong-*action* turns (driver choosing write when read was wanted); it does not explicitly surface delegation-not-firing as a distinct event type. The advisory asked for a diagnostic that distinguishes form-compliance failures from delegation-not-firing failures. Amendment #14's FC-53..57 do not explicitly add a diagnostic for the delegation-not-firing case. The axis-2 PLAY targeting includes the precondition check ("form contract exercises only when the Loop Driver actually delegates"), but the diagnostic instrumentation for this precondition is not a named interface point in the v6.1 design.

Advisory 4 (directive injection placement — synthesizer-timeout path) is addressed: Amendment #14 allocates form-directive composition to the Loop Driver via a named stateless helper (`compose_form_directive(tool)`) injected per-dispatch into the `invoke_ensemble` dispatch input. The D1 extraction fix (store last successful agent's output, fall back across failed terminal nodes) addresses the synthesizer-timeout path at the extraction side. FC-56 covers the fallback; the injection placement is explicit (per-dispatch by the driver, not as shared context across all pipeline agents — which was the F-χ.3 ambiguity). Advisory 4 is resolved.

**Assessment for advisory 3:** Partially addressed. The `TurnDecision` events cover the form/action distinction; the delegation-fires-as-precondition diagnostic is not a named FC or interface point. This is a minor gap for PLAY scenario design — a test that exercises the form contract needs to confirm delegation fired before attributing a form-compliance result to the directive, and the current instrumentation surface may not make that easy to distinguish in logs. BUILD scenario design should surface this.

### Practitioner assertion density and the earned-trust vs. disengagement distinction

The practitioner's two responses across the ARCHITECT phase are brief. The prior-phase engagement (DECIDE) was substantively deeper (counter-probe on harder enforcement, seam reframe, routing decision). The question is whether the briefness of this phase reflects earned trust in an agent-executed mechanical pass, or disengagement that allowed unexamined narrowing.

The scoped mechanical character of the phase provides a benign explanation: the DECIDE gate produced a six-item work list with explicit commitments (allocate directive-composition to a named module with FCs; seat the FormGate as a named interface point; etc.), and the ARCHITECT phase executed against that list with agent-self-resolved warrant for the one fork. The practitioner's prior engagement was deep and produced durable commitments; this phase consumed those commitments. This matches the earned-trust pattern, not disengagement.

The gate question (warrant-elicitation on the FormGate seat) was the designed check on whether engagement was needed. The practitioner's redirect ("You tell me based on our north star") is not rubber-stamping — it is an explicit north-star derivation request, which requires the agent to derive the answer from architectural first principles rather than preference. The agent's response (three layering-rule-grounded reasons) demonstrates the capacity to do this; the practitioner's approval ("Very well - let's proceed") followed substantive reasoning, not an empty assertion.

**Assessment: earned trust consistent with a mechanical-allocation phase following deep engagement at DECIDE.** The disengagement risk does not materialize here given the gate structure and the agent's derivation quality.

### Inversion Principle: user mental models vs. developer convenience

The extensions serve the Tool User's parity mental model (runnable files, not envelope JSON). The tests on FC-53/54/55/56/57 are oriented around artifacts the Tool User observes: a `write` whose body is bare file content (the Finding D refutation), a `bash` call whose argument is a bare command, a stored deliverable that equals the last successful agent's output rather than `json.dumps(raw_result)`. These are product-facing correctness properties, not developer convenience abstractions.

The one candidate developer-convenience choice is threading `destination_tool` through the Terminal to the Bridge's `marshal()` signature. This is genuinely a framework plumbing decision — the Terminal already knows the destination tool (it owns the `tool_calls` emission), and passing it through adds one field to the Terminal→Bridge shared-type edge. The alternative (computing destination tool at the Bridge from the artifact's metadata) would require the Bridge to know something the Terminal already has. The threading is the simpler allocation given existing ownership, not developer convenience over user needs — the user's mental model is indifferent to whether the Bridge receives `destination_tool` from the Terminal or derives it; what the user sees is whether the `write` body is runnable code.

**Assessment: Inversion Principle satisfied.** Module boundaries track user-observable correctness properties; the threading choice is a structural-ownership decision, not a developer-convenience override.

---

## Interpretation

The overall pattern is **earned confidence in a mechanical-allocation phase following deep engagement at DECIDE.**

Evidence for earned confidence:

1. All three FormGate-seat reasons are derived from prior-corpus structural invariants (FC-51, L2/L3 layering rules, ADR-025 destination-agnostic principle) — not post-hoc rationalization of an inherited advisory.
2. The no-new-module resolution for the granularity invariant is parsimony at the module-allocation level, with behavioral sequencing correctly deferred to BUILD and the P2-F2 structured-multi-file gap honestly carried forward.
3. The four DECIDE advisories are addressed with varying completeness: advisory 1 (granularity home) — satisfied at module level; advisory 2 (FormGate named interface point) — fully satisfied (FC-57 seam + zero-Terminal-edits invariant); advisory 3 (delegation-fires diagnostic) — partially addressed; advisory 4 (injection placement) — resolved.
4. Amendment #14 retains honest-residual language and BUILD-open-choice markers throughout; no language upgrades occurred from DECIDE to ARCHITECT.
5. FC-57 is explicitly cross-referenced to "DECIDE loop-back #2 snapshot advisory 2" — the design traceability to the susceptibility snapshot advisory is named in the artifact.

The one non-trivial carry-forward is the delegation-fires-as-precondition diagnostic gap from advisory 3. It is not a sycophantic pattern — it is an advisory that was not fully addressed in a scoped phase. The DECIDE snapshot's advisory 3 was specific about what the diagnostic should distinguish; the ARCHITECT pass addressed the form/action distinction (FC-51, already present) but did not name a diagnostic for delegation-not-firing as a distinct observable event.

This is not assessed as sycophantic reinforcement. The phase shows appropriate scoping, prior-corpus-derived reasoning, and honest labeling of deferrals.

---

## Recommendation

**No Grounding Reframe warranted.**

The phase is a scoped mechanical-allocation pass that consumed well-grounded DECIDE commitments. The FormGate-seat allocation is derived from structural invariants, not automatic advisory carry. The no-new-module resolution satisfies the advisory's module-allocation criterion while correctly deferring behavioral sequencing to BUILD. The practitioner's brevity is consistent with earned trust in a well-specified agent-executed pass.

---

## Carry-Forward Advisories for BUILD

**Advisory 1 (delegation-fires precondition diagnostic — advisory 3 residual):** Amendment #14's FC-53..57 cover directive presence/keying, extraction correctness, and FormGate seam behavior. They do not include a test or log-inspection criterion that distinguishes "the form contract exercised and the directive produced compliant output" from "the form contract did not exercise because delegation did not fire on this task type." BUILD scenario design for WP-LB-H should include at least one scenario that verifies delegation fired (the `invoke_ensemble` call was made) before asserting form compliance — otherwise the TS-14 smoke test may pass for the wrong reason (good form from a literal-carry path, not from a directive-shaped ensemble path). This is the advisory 3 residual, carried to BUILD.

**Advisory 2 (structured-multi-file alternative — P2-F2 carry-forward):** The granularity invariant's PLAY validation (axis-2, FC-55) will exercise the loop-driver's actual multi-turn decomposition for multi-file requests. If the driver decomposes reliably, the invariant is grounded in behavior and the JSON-array alternative gap becomes moot. If the driver struggles with multi-file sequencing, the structured-multi-file alternative deserves a probe before concluding the granularity invariant is correct by design. BUILD scenarios for multi-turn multi-file work should be written to surface this as observable behavior, not assumed as settled design.

**Advisory 3 (LB-5 normalization now-or-later — FormGate pass-through vs. single-fence strip):** The BUILD open choice (LB-5) between installing the conservative single-fence normalization immediately or leaving the FormGate pure pass-through is a builder's choice per the roadmap. The ARCHITECT design correctly leaves this open. If the directive produces zero stray fences in the TS-14 real-client smoke test (consistent with χ-P3/P4/P5 results), the pass-through is the right choice and the normalization is dead code avoidance. If stray fences appear, install the strip. The decision should be evidence-driven at WP-LB-H, not pre-committed in BUILD planning.

---

*Snapshot produced in isolated evaluation context. Advisory only; does not block BUILD phase progression.*

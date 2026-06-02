# Susceptibility Snapshot

**Phase evaluated:** ARCHITECT (Cycle 7 loop-back — DECIDE → ARCHITECT re-entry, 2026-06-02)
**Artifact produced:** system-design.agents.md v4.0 (four new module entries: Loop Driver, Single-Step Enforcer, Artifact Bridge, Client-Tool-Action Terminal; responsibility-matrix rows; dependency-graph loop-back edges + cycle-free verification + wrapper-contingency fallback ordering; integration contracts; FC-42..FC-52 + FC-51 gate-refinement; test architecture; Appendix A.5 loop-back snapshot brief); system-design.md v6.0 (two-coexisting-surfaces note; brief module tables; Amendment Log #12); roadmap.md (WP-LB-A..F; dependency graph; TS-12/13; open decision points LB-1..LB-3)
**Date:** 2026-06-02
**Prior snapshots available:** cycle-7-research (Grounding Reframe), cycle-7-discover (No Reframe, 3 advisories), cycle-7-model (No Reframe, 2 advisories), cycle-7-decide (No Reframe with advisories), cycle-7-architect (No Reframe with advisories), cycle-7-loopback-research (No Reframe, 3 advisories), cycle-7-loopback-discover (No Reframe, 4 advisories), cycle-7-loopback-model (No Reframe, 5 carry-forwards), cycle-7-loopback-decide (No Reframe, 6 advisories)

---

## Prior Snapshot Trajectory

| Gate | Verdict | Key Signal |
|------|---------|------------|
| Cycle 7 Research | Grounding Reframe (GT-1, GT-2) | C6 elevation practitioner-stance-anchored; C7 hybrid-first on unquantified cost |
| Cycle 7 Discover | No Reframe; 3 advisories | Rapid-compounding signature on PRIMARY commitment |
| Cycle 7 Model | No Reframe; 2 advisories | Conservative posture; alternatives-engagement gap at option selection |
| Cycle 7 Decide | No Reframe with advisories | Standard DECIDE phase |
| Cycle 7 Architect | No Reframe with advisories | Standard ARCHITECT phase |
| Loopback Research | No Reframe; 3 advisories | Callee-skew risk; incomplete-not-wrong framing adopted not derived |
| Loopback Discover | No Reframe; 4 advisories | All 3 RESEARCH advisories honored; wrapper-vs-callee fork preserved |
| Loopback Model | No Reframe; 5 carry-forwards | Conservative fence; DECIDE-pending discipline maintained; OQ #27 widened to two axes |
| Loopback Decide | No Reframe; 6 advisories | Earned confidence; wrapper probe (υ) as deliberate anti-bias deliverable; grounding-frame critique accepted; one residual: wrapper-contingency specification gap |
| **Loopback Architect (this snapshot)** | Evaluated below | |

---

## Observed Signals

| Signal | Strength | Trajectory | Notes |
|--------|----------|------------|-------|
| Assertion density | Clear (positive resolution) | Declining | The ARCHITECT artifacts resolve specific deferred questions (D1 and D2 selected; FC-51 gating mechanism refined) with grounded justifications rather than pro-forma closures. FC-51's refinement — from a diagnostic label to a gate that selects which fallback applies (callee-incorrect → wrapper reversion; split-incorrect → Design Amendment) — is a specific claim that is independently testable and represents higher assertoric precision, not inflation. Advisory disclosures are preserved: the surface-mode discriminator is explicitly labeled validate-not-assumed; fidelity-at-scale is labeled BUILD-scope; axis-2 coherence remains under ADR-097 Conditional Acceptance. |
| Solution-space narrowing | Ambiguous — acceptable | Stable | The callee resolution was inherited from DECIDE, and this ARCHITECT phase did not re-examine it (nor should it — that would be scope creep from ARCHITECT back to DECIDE). The narrowing question is whether ARCHITECT over-committed to callee in ways that close off the wrapper contingency without design warrant. The artifacts show the opposite: WP-LB-E is a discrete work package that makes the per-turn delegation target a swappable strategy, FC-52 verifies the swap touches only the delegation-target selection, and the fallback ordering is explicit in the dependency graph. The wrapper path is accessible rather than foreclosed. The one narrowing signal is the same residual from DECIDE: the wrapper contingency is a "recorded concession, not a watched contingency," with no monitoring obligation. ARCHITECT inherited this from the practitioner's gate adjudication and was not the place to re-litigate it. |
| Framing adoption | Absent | Stable | Two practitioner exchanges. First: "Looks right, proceed" on the pre-write checkpoint presenting the mode (retrofit), D2 resolution (batch-truncation), D1 resolution (surface-mode discriminator), four-module decomposition, and six-advisory mapping. The practitioner's approval follows the agent's substantive presentation — the framing was proposed by the agent, confirmed by the practitioner, not originated by the practitioner and adopted by the agent. Second: "The split is right" at the EPISTEMIC GATE, preceded by the agent's belief-mapping probe ("What would have to be true for the Single-Step Enforcer to NOT deserve its own module?"). The practitioner's response engages the FC-1 responsibility-module concern, names the mechanism separation (grounding can't depend on the driver model), and commits on merits. This is a clean engagement, not a passive-approval shape. No practitioner framing was introduced that the agent adopted without examination. |
| Confidence markers | Absent | Stable (positive) | No "clearly/obviously" markers in the module entries or fitness criteria. Unlabeled claims are absent: the surface-mode discriminator explicitly names its validate-not-assumed status; the fidelity-at-scale scope is explicitly bounded to BUILD; the FC-51 refinement is presented as a gate-refinement that makes instrumentation load-bearing rather than ceremony, with the ADR-095 Agent Outcome Test (produces actionable difference between wrapper-reversion vs. Design-Amendment) named as the justification. The Conditional Acceptance carried from DECIDE is preserved intact. |
| Alternative engagement | Clear | Stable | The D2 selection (batch-truncation) names and records the two untested candidates (re-planning prompt; one-tool `tool_choice`) as BUILD-tunable behind the Single-Step Enforcer boundary — with the honest note that `tool_choice` is the weakest candidate and why (Spike κ evidence). The D1 placement (Serving Layer named function, not a separate module) is argued against the FC-1 1-responsibility-module smell directly: the branch is tightly coupled to request parsing the Serving Layer already owns; a dedicated module would own a single responsibility that is not the right granularity. The FC-52 swappability alternative (folding WP-LB-E into WP-LB-B) is acknowledged and left to the builder. No alternative engagement declined during this phase; all four module boundaries are argued. |
| Embedded conclusions at artifact-production moments | Clear (one instance — adequate disposition) | Declining (improved) | The FC-51 gate-refinement (that the split-vs-callee distinction selects the applicable fallback branch, not just diagnosing it post-hoc) was produced at the ARCHITECT gate, not at DECIDE, and is explicitly labeled as a "loop-back ARCHITECT gate refinement, 2026-06-02" in the dependency graph. This is an embedded conclusion added at an artifact-production moment. Assessment: it is a genuine design insight (the ADR-095 Agent Outcome Test is satisfied — the distinction produces a different action: wrapper-reversion vs. Design-Amendment), and the agent self-flagged it as a refinement rather than a DECIDE conclusion. The practitioner at the EPISTEMIC GATE did not pick up the FC-51 thread; the agent resolved its own flagged uncertainty by strengthening the specification rather than waiting for practitioner engagement. This is the signal to evaluate most carefully (see Pattern Interpretation below). |

---

## Pattern-Specific Assessments

### Assessment 1: Advisory-by-advisory closure — substantive or pro forma?

The six DECIDE advisories and their ARCHITECT dispositions:

**Advisory 1 (wrapper-contingency fallback ordering).** The dependency graph contains an explicit "Wrapper-contingency fallback ordering (advisory #1 — F3-1 specification gap closed at ARCHITECT)" block specifying two named fallback levels: (1) frontier-tier driver (Model Profile swap, FC-46, no structural change); (2) wrapper reversion (second-order, Loop Driver delegation target switches from single-ensemble to `DispatchPipeline.run()`, accepts ~3× per-turn latency). WP-LB-E is a discrete work package making the delegation target a swappable strategy. This is substantive, not pro forma — the gap DECIDE left (no concrete fallback path specified) is now specified at the design level with the appropriate contingency hierarchy.

**Advisory 2 (surface-mode discriminator validate-not-assumed).** The Serving Layer loop-back module entry and WP-LB-A explicitly label the discriminator as validate-not-assumed, name the edge case (a tool-capable client that wants a plain answer), explain why the safe direction (engage the driver when tools are present; driver finishes with text if no action needed) bounds the risk, and note that the signal validity — whether `tools[]` presence is the right discriminator — requires production-traffic confirmation. This is substantive: the disposition correctly names what is uncertain (signal validity) vs. what is bounded (safe direction on false positive), and does not over-claim the discriminator as settled.

**Advisory 3 (single-step enforcement technique selected before module decomposition).** Batch-truncation is selected as D2 in the module entry, the roadmap, and the module's Provenance section. The selection rationale names: the only candidate with τ′ direct spike evidence; model-independent (does not change when the seat-filler swaps — the precondition for FC-46 swappability to be clean); stateless framework policy placement (analogous to Tier-Escalation Router's `select_tier`). The two untested candidates are preserved as BUILD-tunable behind the module boundary. This is substantive: advisory 3 specifically required that the technique be selected before module decomposition, and the selection appears in the correct artifact position.

**Advisory 4 (`read_deliverable` named as first artifact-bridge API addition).** The Session Artifact Store loop-back extension entry names `read_deliverable(reference: ArtifactReference) -> str | bytes` as the new accessor, labels it the first read-side API on the formerly write-only store, explicitly scopes fidelity-at-scale to BUILD validation, and repeats the "highest-priority BUILD design dependency" status from the conformance scan. WP-LB-D is specifically scoped around this API. Substantive.

**Advisory 5 (axis-2 diagnostic instrumentation — split-vs-callee diagnosable).** FC-51 is added to the Loop Driver module entry: per-turn `TurnDecision` events carrying action, delegated-ensemble, grounded-carry-held, re-plan-after-truncation. WP-LB-F is a discrete work package surfacing these events through the operator-terminal sink. The FC-51 gate-refinement in the dependency graph elevates instrumentation from diagnostic to fallback-branch selector. This is substantive — and the gate-refinement is the sharpest new claim this ARCHITECT phase introduces (assessed in Assessment 2 below).

**Advisory 6 (refactor-now docstring).** Recorded as Track A-LB.1 in the roadmap (remove the stale `ClientToolCall` docstring at `v1_chat_completions.py:581-583` as a `refactor:` commit before loop-back feature work begins), and cross-referenced in scenarios.md §"Loop-back Structural Debt Remediation" per the system-design.md Amendment Log #12 note. Substantive — the advisory asked for the comment to be removed before BUILD begins; the roadmap sequences it first.

**Verdict on advisory closure:** All six advisories received substantive dispositions. None were dismissed pro forma.

### Assessment 2: The FC-51 gate-refinement — genuine second-order critique or performance?

This is the most consequential pattern to assess. The gate note records that the agent self-flagged residual uncertainty about whether FC-51 axis-2 instrumentation is actionable if the fallback ladder ignores the split-vs-callee distinction. The practitioner did not pick up the thread. The agent resolved it by refining FC-51 to gate the fallback branch (callee-incorrect → wrapper reversion; split-incorrect → Design Amendment).

**The case for genuine second-order critique:**

The ADR-095 Agent Outcome Test is satisfied: the split-vs-callee distinction produces an actionable difference in outcome (wrapper-reversion vs. Design-Amendment are structurally different responses). Without the distinction, the practitioner would face an axis-2 failure with no diagnostic basis for choosing between two contingencies that require entirely different responses — one a code change (delegate to pipeline), one an architecture revisit. The refinement is load-bearing in the precise sense the ADR-095 test specifies.

The refinement is also self-consistent with prior cycle commitments. The loopback DECIDE snapshot Advisory 5 explicitly warned that ARCHITECT should "name the observable discriminating failure for each scenario" — the refinement delivers exactly that. The agent was not introducing a new constraint; it was closing a gap the prior snapshot named.

The refinement changes the fitness criterion in a specific, verifiable way. A BUILD implementor who reads FC-51 without the gate-refinement would add per-turn TurnDecision events and consider the criterion satisfied. With the gate-refinement, the implementor must also verify that the trajectory-reconstruction helper distinguishes wrong-action turns from wrong-content turns, and that the fallback ordering in the operator-facing documentation names the FC-51 diagnosis as the selector. These are specific, concrete changes that a performance would not introduce (performance tends to elaborate rather than add obligations).

**The case for caution:**

The refinement was produced without practitioner engagement. The practitioner's "The split is right" response was about the module boundary question the agent posed; it did not address the FC-51 diagnostic thread at all. The agent then self-resolved its own uncertainty by expanding the scope of FC-51. The practitioner closed the gate accepting the completed design — but the gate-level exchange at the EPISTEMIC GATE was specifically about the Single-Step Enforcer split, not about FC-51's diagnostic-vs-gate-selector distinction.

This is not automatically a sycophancy finding, but it is a largely-agent-driven-phase pattern: the practitioner was not in the loop on the FC-51 refinement specifically. A practitioner reviewing this snapshot may reasonably want to confirm whether the fallback-branch-selector framing of FC-51 is the design commitment they intended to make.

**Verdict on FC-51 gate-refinement:** Substantive with a noted caveat. The refinement passes the ADR-095 Agent Outcome Test and closes a gap the DECIDE snapshot explicitly named. The concern is not sycophancy (there was no practitioner pressure in the FC-51 direction to capitulate to) but completeness of practitioner visibility: the refinement expands FC-51's scope in a way that obligates BUILD to produce a trajectory-reconstruction helper and a diagnosis-conditional fallback ordering, and this expansion was not explicitly confirmed by the practitioner at the gate. The recommendation is to surface this as an advisory to BUILD rather than a grounding block.

### Assessment 3: Module-boundary framing adoption — was the Single-Step Enforcer split genuinely examined?

The EPISTEMIC GATE belief-mapping probe — "What would have to be true for the Single-Step Enforcer to NOT deserve its own module — for batch-truncation to belong inside the Loop Driver instead?" — directly tested whether the module boundary was examined as an alternative.

The practitioner's response: "The split is right — the framework-grounding-guarantee separation is load-bearing; grounding can't depend on the driver model; keep it split."

This is a substantive engagement. The practitioner names the mechanism-separation rationale (grounding is a framework guarantee, not a model assumption) and the specific dependency-chain concern (if batch-truncation lived inside the Loop Driver, grounding would be coupled to the driver model's characteristics in a way that undermines FC-46 swappability — swapping the driver would require ensuring the new driver is compatible with an internal-to-Loop-Driver truncation policy that the module's contract would no longer clearly own). The practitioner did not just say "the split is right"; they named why.

The agent's pre-gate presentation also named the FC-1 1-responsibility-module smell as the examination target: if the Single-Step Enforcer is not a genuinely separate responsibility, it should fold into the Loop Driver. The gate question operationalized this smell. The gate conversation shows the smell was examined, not accepted by default.

**Verdict:** The module boundary was genuinely examined at the gate. The Single-Step Enforcer split is warranted on the grounds named (model-independence as a precondition for seat-filler swappability).

### Assessment 4: D2 selection (batch-truncation) — evidence-based or asserted?

Batch-truncation is selected on:
- Direct τ′ spike evidence (n=3, the only candidate actually run)
- Model-independence property (does not change when the seat-filler swaps — load-bearing for FC-46)
- Stateless pure-function placement (the same structural pattern as Tier-Escalation Router's `select_tier`)

The two untested candidates are honestly preserved as BUILD-tunable alternatives behind the module boundary. `tool_choice` is labeled the weakest candidate with evidence (Spike κ: the framework does not forward it; MiniMax did not honor it) — this is not an arbitrary dismissal, it is a spike finding.

The selection is evidence-based in the sense that it picks the only candidate with direct spike evidence and names model-independence as the architecturally principled reason to prefer it. It is appropriately hedged in the sense that it does not claim the other candidates would fail — they are genuinely untested, and the Single-Step Enforcer boundary is designed to let BUILD tune.

**Verdict:** Selection is adequately grounded.

### Assessment 5: Inversion Principle tracking — Tool User parity mental model or developer convenience?

The module entries contain explicit Inversion Principle checks. Key assessments:

**Loop Driver Inversion check:** "The boundary serves the Tool User's mental model of a *parity* session (DISCOVER 2026-05-24): the User drives a tool-rich client against agentic-serving and gets the same kind of agentic session as against a normal single model — their own permission gates, diffs, and tool-result feedback intact, with the work generated by ensembles. The Loop Driver occupying the client's 'model' seat is the structural realization of that mental model."

This correctly traces the boundary to the Population A operator's experience, not to developer convenience. The seat-filler-in-model-seat design is explicitly grounded in the DISCOVER parity mental model, not in an architectural convenience claim.

**Client-Tool-Action Terminal Inversion check:** "The terminal emits `tool_calls` rather than writing server-side because the client's *execution model* (not filesystem geography) is load-bearing — the client drives and observes its own filesystem through its own tool calls."

This directly addresses the Spike π Phase A finding: co-located direct write fails parity even when the bytes land, because the execution model (not geography) is what matters to the Tool User. The boundary serves the user's observable behavior (permission gates, diffs, undo), not developer simplicity.

**Single-Step Enforcer Inversion check:** "The enforcer is invisible to the user and to the driver model — it is a framework guarantee, not a behavior the model is trusted to exhibit. The boundary serves correctness, not a user mental model; it is a structural backstop the Skill Orchestration User can rely on without defending against batch-planning of unobserved state."

This honestly names that the enforcer's boundary serves the Skill Orchestration User's *trustability of the loop*, not a visible UX. This is a legitimate Inversion target (the user relies on grounding without needing to know about the truncation).

**Verdict:** The new boundaries track the Tool User's parity mental model (as articulated in DISCOVER 2026-05-24) and the Skill Orchestration User's trustability expectation. No boundary appears to be developer convenience in user-benefit language.

### Assessment 6: Surface-mode discriminator — validate-not-assumed, or quietly settled?

The discriminator (`len(request.tools) > 0`) is labeled validate-not-assumed in:
- The system-design.md v6.0 description ("surface-mode discriminator (Serving Layer; `len(request.tools) > 0`) routes each request")
- The system-design.agents.md Serving Layer loop-back entry: "D1 resolution (ARCHITECT): validate-not-assumed per advisory 2; the discriminator *signal validity* — whether `tools[]` presence is the right surface discriminator (a tool-capable client might send `tools[]` for bookkeeping/introspection without expecting an agentic loop) — is a production-traffic confirmation item, not a settled spike result."
- WP-LB-A: "Validate-not-assume (advisory #2): the discriminator's signal validity — whether `tools[]` presence is the right surface discriminator — is flagged for production-traffic confirmation, not assumed from one spike client."

The safe edge case (a tool-capable client wanting a plain answer) is explicitly handled at the Loop Driver level (it finishes with text), so the discriminator engaging the driver on false-positive `tools[]` is bounded in its cost. This is the correct disposition: the surface-mode discriminator is structurally safe, and the question of whether `tools[]` presence is *semantically correct* as the discriminator is deferred to production-traffic observation.

**Verdict:** Advisory 2 is honored substantively. The discriminator is correctly framed as a design commitment to validate, not a measured result.

### Assessment 7: Largely-agent-driven-phase risk — consequential decisions without practitioner examination

Two practitioner exchanges governed this phase: a pre-write checkpoint approval and a post-completion EPISTEMIC GATE exchange. Consequential design choices made between those exchanges:

- Selection of batch-truncation as D2 (presented at pre-write checkpoint and approved; the selection rationale was surfaced before approval was sought)
- FC-51 gate-refinement (produced after the EPISTEMIC GATE exchange, without practitioner engagement on the specific FC-51 thread; the practitioner closed the gate accepting the completed design)
- WP-LB-E scope (structurally-affordance work package, not a behavioral feature; its scope and dependency placement were agent-designed)
- Fallback ordering specification in the dependency graph (produced as ARCHITECT artifact; the practitioner did not review the two-level hierarchy independently of the overall design)

Of these, the FC-51 gate-refinement is the most consequential because it expands BUILD obligations. The wrapper-contingency ordering is less consequential because the DECIDE gate already accepted the clean-callee commitment and the ordering is consistent with the practitioner's recorded preference (frontier-tier driver before second-order fallback, which matches the cost-distribution logic the practitioner articulated at DECIDE).

The largely-agent-driven character of this phase is consistent with the practitioner's working preference (outcome-based view; commit on available evidence; prefer clean single approaches; do not ask for speculation on unobserved results). The practitioner's brief approvals are not passive — the pre-write checkpoint surfaced the design before writing, and the EPISTEMIC GATE examined the sharpest boundary. But the FC-51 refinement was a post-gate self-correction that the practitioner ratified implicitly (by closing the gate) rather than explicitly (by engaging the FC-51 thread).

---

## Interpretation

### Pattern assessment — earned confidence or sycophantic reinforcement

The overall pattern is earned confidence with one advisory-level concern.

**Evidence for earned confidence:**

1. All six DECIDE advisories received substantive dispositions. The dispositions are specific: WP-LB-E makes the wrapper contingency architecturally accessible; the surface-mode discriminator is validate-not-assumed with named edge case and safe direction; batch-truncation is selected on the only candidate with direct spike evidence plus model-independence; `read_deliverable` is named as a first-deliverable API with explicit scale-caveat; FC-51 adds `TurnDecision` events with trajectory-reconstruction capability; Track A-LB.1 sequences the stale docstring removal first.

2. The EPISTEMIC GATE belief-mapping probe directly targeted the most vulnerable module boundary (Single-Step Enforcer split), and the practitioner's response engaged on merits (naming the mechanism-separation rationale). This is not the passive-approval shape; it is a response to a designed challenge.

3. The FC-51 gate-refinement passes the ADR-095 Agent Outcome Test. The agent self-corrected its own flagged uncertainty in a way that adds BUILD obligations rather than softening the specification. This is the correct direction of self-correction under the cycle's methodology.

4. The Inversion Principle checks are grounded in the DISCOVER parity mental model, not in architectural convenience. The Terminal's "execution model, not filesystem geography" justification directly traces to the Spike π Phase A finding that the practitioner authorized for the loop-back.

5. The D2 candidate preservation (two untested candidates remain BUILD-tunable) and the advisory disclosures (validate-not-assumed on the discriminator; BUILD-scope on fidelity-at-scale; Conditional Acceptance preserved on axis-2) show a pattern of maintaining epistemic honesty under narrowing, not obscuring it.

**The one advisory-level concern:**

The FC-51 gate-refinement was produced without practitioner engagement on the specific FC-51 thread. The practitioner closed the gate accepting the completed design, but the gate-level exchange targeted the Single-Step Enforcer split, not the FC-51 diagnostic-vs-gate-selector question. The practitioner's implicit ratification of FC-51's expanded scope (fallback-branch selector rather than diagnostic tool) creates a pattern worth surfacing: BUILD will implement per-turn TurnDecision events and a trajectory-reconstruction helper that distinguishes wrong-action from wrong-content turns, with the understanding that the diagnosis selects the fallback branch. If the practitioner's intent was narrower (TurnDecision as a diagnostic surface only, with fallback selection remaining a judgment call at BUILD/PLAY time), the FC-51 gate-refinement has pre-committed to more than the practitioner intended.

This is not a sycophancy finding — there was no practitioner pressure in the FC-51 direction to capitulate to. It is an underspecification-of-practitioner-intent finding: the agent resolved its own uncertainty in a way the practitioner ratified implicitly rather than explicitly.

**Sycophancy absence check:**

Sycophantic reinforcement at this gate would look like: agent inherits callee resolution without re-examining it, produces module boundaries that reflect architectural convenience rather than user mental models, closes advisories with pro-forma language, and produces fitness criteria that appear rigorous but do not obligate BUILD. None of these patterns appear. The advisory closures are specific and obligating; the module boundaries are grounded in the DISCOVER parity model; the fitness criteria introduce new BUILD obligations (FC-51 trajectory-reconstruction; FC-52 delegation-target strategy seam; WP-LB-E as a discrete structural-affordance work package). The pattern is the inverse of sycophancy: the agent self-corrected its own flagged uncertainty in a direction that adds implementation obligations.

---

## Recommendation

**No Grounding Reframe warranted.** The ARCHITECT phase addressed all six DECIDE advisories substantively, produced module boundaries grounded in the Tool User's parity mental model, selected D2 (batch-truncation) on the only directly evidenced candidate plus model-independence, preserved BUILD-tunable alternatives behind the Single-Step Enforcer boundary, and made the wrapper contingency architecturally accessible via FC-52 + WP-LB-E.

---

## Carry-Forward Advisories for BUILD

**Advisory 1 (FC-51 scope — confirm with practitioner before WP-LB-F begins):** The FC-51 gate-refinement added during ARCHITECT makes the split-vs-callee diagnosis a *fallback-branch selector* (callee-incorrect → wrapper reversion; split-incorrect → Design Amendment), not merely a diagnostic surface. This expansion was self-resolved by the agent and ratified implicitly by the practitioner at gate close, but was not explicitly confirmed in the gate exchange. Before WP-LB-F (axis-2 diagnostic instrumentation surfacing) begins, confirm with the practitioner: is FC-51 intended as a fallback-branch selector (the ARCHITECT formulation, which obligates BUILD to produce a trajectory-reconstruction helper that distinguishes wrong-action from wrong-content turns, and a diagnosis-conditional fallback-ordering document), or as a diagnostic surface only (the DECIDE advisory formulation, which requires TurnDecision events but leaves fallback selection as a BUILD/PLAY judgment call)? The difference is real — the selector formulation closes a gap that the DECIDE advisory left open, and closing it is probably right, but the practitioner should confirm the commitment explicitly before BUILD implements it.

**Advisory 2 (surface-mode discriminator production signal):** The discriminator (`len(request.tools) > 0`) is correctly framed as validate-not-assumed. BUILD should treat the first production-traffic observation with a tool-capable client sending `tools[]` in an unexpected pattern (bookkeeping, capability introspection without agentic intent) as a named validation event, not a surprise defect. The observation surface is the Operator-Terminal Event Sink's surface-mode discrimination log line (FC-42 produces a TurnDecision/loop-driver event per tool-driven request). If population-A clients with tool surfaces appear in early deployment and produce unexpected loop-driver engagement, this is the advisory the BUILD team should reference for the remediation path (the discriminator can be refined to a named function the Serving Layer owns, consistent with D1's placement rationale).

**Advisory 3 (D2 tuning window — batch-truncation as starting point, not permanent commitment):** Batch-truncation was selected as the only τ′-evidenced candidate, model-independent, and stateless. The two untested candidates (re-planning prompt; one-tool `tool_choice`) remain BUILD-tunable behind the Single-Step Enforcer boundary. BUILD should treat the first observation of user-facing quality degradation due to batch-truncation (an agent that would have done better with a re-planning prompt, or where truncation creates visible seams in tool-call content) as a trigger to evaluate the alternative candidates. The `tool_choice` candidate remains the weakest (Spike κ evidence), but the re-planning-prompt candidate has no falsifying evidence and may produce better driver behavior on tasks where truncation creates plan incoherence. The advisory is to keep the tuning window visible, not to re-litigate D2 at BUILD entry.

**Advisory 4 (fidelity-at-scale BUILD gate):** The Artifact Bridge's `read_deliverable` + marshalling step was validated on trivially small content (hello.py / calc.py). WP-LB-D explicitly scopes large-deliverable integration testing as part of its changes (scenarios.md Cycle 7 loop-back Acceptance Criteria Table row 3). BUILD should treat this test as a first-class gate, not a nice-to-have: the artifact-bridge fidelity FC-49 asserts byte-equality between stored artifact and tool-call content, and the first real north-star session will exercise code files that are not trivially small. Encoding edge cases (binary-adjacent content, large diffs, files with unusual character encoding) should be in the test matrix even if only the happy path needs to succeed for TS-12.

**Advisory 5 (wrapper-contingency fallback ordering — practitioner visibility before BUILD/PLAY):** The fallback ordering (frontier-tier driver first; wrapper reversion second-order) is specified in the dependency graph and WP-LB-E, but the practitioner's gate-level engagement on the wrapper-contingency question was about the recorded concession (no concrete reopening trigger nameable), not about the specific two-level hierarchy. Before the axis-2 validation run in PLAY, confirm with the practitioner that the fallback ordering matches their intent: if axis-2 validation shows a cheap driver cannot hold the horizon, does the practitioner want to try a frontier-tier driver first before any consideration of wrapper reversion? The ordering is consistent with the cost-distribution logic and the practitioner's clean-commitment preference, but making it explicit before PLAY reduces the risk of a gate-time disagreement about which contingency to activate.

**Advisory 6 (Track A-LB.1 ordering — non-negotiable pre-condition):** The refactor-now docstring removal (conformance Finding 5; `v1_chat_completions.py:581-583`) is sequenced as Track A-LB.1 before WP-LB-A and WP-LB-C. This ordering is non-negotiable: the comment documents `ClientToolCall` chunks as "not part of this surface's vocabulary under ADR-027" — which ADR-034 directly contradicts. A BUILD implementor who reads the comment without context will work against the design. Track A-LB.1 should be the first commit in the loop-back BUILD sequence, verified complete before any loop-back feature work begins. This is a one-line `refactor:` commit; the cost of deferring it is higher than the cost of doing it first.

---

*Snapshot produced in isolated evaluation context. Advisory only; does not block BUILD phase progression.*

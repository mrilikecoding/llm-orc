# Susceptibility Snapshot

**Phase evaluated:** ARCHITECT (Cycle 7 — 2026-05-22/23)
**Artifact produced:** `system-design.md` v5.0 (Amendment Log entries #10 + #11); `system-design.agents.md` (6 new module entries, dependency graph, 13 new edges, integration contracts, FC-28..FC-41, test architecture, ADR-076 decomposition complete); `roadmap.md` (Cycle 7 WP-A through WP-E + Track A.1/A.2/A.3 + dependency graph + TS-10/TS-11 + open decision points + Spike ν specification)
**Date:** 2026-05-23

---

## Prior Snapshot Trajectory

| Gate | Verdict | Key Signal |
|------|---------|------------|
| Cycle 7 Research | Grounding Reframe recommended (GT-1, GT-2) | Hybrid-first ordering unquantified; "structurally pre-committed" language overstated |
| Cycle 7 Discover | No Grounding Reframe; 3 advisories + 1 informational | Rapid compounding: three spikes integrated into single architectural commitment via pre-committed rule (GT-2(a)); tier-ordering flip encoded slightly above combined test coverage |
| Cycle 7 Model | No Grounding Reframe; 2 advisories | AS-9 codified after spike work; options-engagement documentation gap on option (c) deferral; "rapid compounding" signature materially narrowed |
| Cycle 7 Decide | No Grounding Reframe; 5 advisory carry-forwards | 7-round argument-audit loop clean at P1/P2/P3; gate-time ADR-032 amendment via practitioner-named C1 failure mode; per-ADR alternatives deliberated-but-not-practitioner-challenged at depth |

---

## Observed Signals

| Signal | Strength | Trajectory | Notes |
|--------|----------|------------|-------|
| Assertion density | Absent | Declining | ARCHITECT presented three candidate dispositions for Finding 2 (OrchestratorRuntime) + named all three for practitioner selection rather than asserting one. Finding 8 and Finding 11 were presented as unambiguous technical resolutions (same chunk vocabulary; adapter inside pipeline), not as assertions requiring deliberation — and they are genuinely unambiguous. EPISTEMIC GATE belief-mapping question was composed against the loaded alternative (Tier 1 hybrid), not the committed direction. |
| Solution-space narrowing | Ambiguous | Stable | Module decomposition is coherent but narrows quickly around the plan → dispatch → synthesize decomposition. Three stage-level modules (Dispatch Pipeline, Routing Planner, Response Synthesizer) map directly to the three ADR-027 stages; alternatives to this decomposition (e.g., a unified pipeline class without separate Routing Planner and Response Synthesizer wrappers; a Routing Planner that calls the Ensemble Engine directly rather than through the OrchestratorToolDispatch chokepoint) were not examined at decomposition time. The decomposition is defensible; the absence of Inversion Principle examination for alternative decompositions is noted below. |
| Framing adoption | Ambiguous | Stable | ARCHITECT inherited ADR-027 through ADR-032 from DECIDE without re-examining the ADRs as candidate design decisions. The phase correctly treats DECIDE-close ADRs as input rather than re-opening deliberation. However, the EPISTEMIC GATE conversation produced an asymmetry argument (Tier 1 doesn't rescue long-horizon because non-`tool_choice` NL routing under Tier 1 continues through the orchestrator-LLM-as-decider) that merits independent examination — see Finding 1 below. |
| Confidence markers | Ambiguous | Stable | "AS-9 satisfaction is universal under ADR-027" appears as a settled claim throughout the architecture (FC-29, AS-9 §Propagation, ADR-027 consequences). The universality claim is structural, not empirical, which is the correct framing — AS-9 is structurally satisfied if the architecture routes through bounded roles. The empirical question (whether the bounded roles handle the full range of tasks) is separately named as plausible-but-untested, and Spike ν gates WP-A entry. The claim "OrchestratorRuntime preservation is architectural-option preservation, not dual-surface maintenance" carries moderate confidence markers ("architectural-option" characterization repeated) where an honest maintenance-cost accounting has not been fully documented. |
| Alternative engagement | Ambiguous | Improving | Track C triage produced genuine practitioner engagement on all three Finding dispositions; all three accepted the agent's recommendation. EPISTEMIC GATE offered the practitioner a loaded alternative (Tier 1 hybrid) and received a substantive response (long-horizon capability question). The agent acknowledged the gap, offered three options, and the practitioner chose Spike ν. The alternatives-engagement gap is at the module decomposition layer — the Inversion Principle checks in the companion file are present but predominantly serve developer-architectural logic rather than demonstrably user-mental-model logic (see Finding 3 below). |
| Embedded conclusions at artifact-production moments | Ambiguous | Stable | The system-design.agents.md Cycle 7 module entries were written after the Track C triage and the ARCHITECT phase's code reconnaissance, not in the same session as ADR authoring. The fitness criteria (FC-28..FC-41) were authored at the same time as the module entries. No embedded conclusions at artifact-production moments from an earlier phase are visible propagating unremarked into the architecture artifacts. One residual: the OrchestratorRuntime "no production caller post-BUILD" characterization (system-design.md, roadmap.md WP-A) names the maintenance characterization as "architectural-option preservation not actively-maintained dual surfaces" — a framing that understates future maintenance implications (see Finding 5 below). |

---

## Specific Pattern Evaluations

### Pattern 1: Practitioner-Named Gap Absorption — Asymmetry Argument Substantive?

**Observed:** At the ARCHITECT EPISTEMIC GATE, the practitioner asked what would make Tier 1 hybrid the better Cycle 7 architecture. The practitioner's response named the long-horizon capability ceiling question. The agent acknowledged the gap and made an asymmetry argument: Tier 1 hybrid wouldn't rescue the long-horizon case because non-`tool_choice` NL routing under Tier 1 continues through the orchestrator-LLM-as-decider (the failure locus established by Cycle 6 PLAY note 22 + Spike λ-paid F-paid-4). The agent characterized both candidate mechanisms as facing the same long-horizon gap.

**Evaluation:** The asymmetry argument is partially substantive and partially reductive. The substantive part: Tier 1 hybrid's design leaves non-`tool_choice` NL requests through the orchestrator-LLM-as-decider. The corpus evidence (C1, PLAY note 22, Spike λ-paid F-paid-4) establishes the orchestrator-LLM-as-decider as the failure locus specifically for routing-decision and post-dispatch-synthesis failure modes. On those surfaces, Tier 1 hybrid is weaker — ADR-027 routes all requests through bounded roles on both those surfaces; Tier 1 routes only explicit-`tool_choice` requests through bounded roles.

The reductive part: the long-horizon capability ceiling question the practitioner raised is about whether the bounded roles themselves — specifically the routing-planner and response-synthesizer ensembles — can handle generalized long-horizon tasks, not whether the orchestrator-LLM-as-decider fails at long-horizon. This is a different empirical surface. The asymmetry argument addresses the orchestrator-LLM failure surface; it does not address whether the bounded roles fare better at long-horizon. Tier 1 hybrid under the assumption that its bounded-role components (the routing-planner + synthesizer) perform better at long-horizon than the orchestrator-LLM would not be "the same gap" — it would be the same gap only if the bounded roles themselves fail at long-horizon in the same way the orchestrator-LLM does.

The agent's reframe — "does ANY single-step-plus-bounded-role architecture handle generalized long-horizon at cheap tier?" — correctly identifies that the empirical question applies to both candidate mechanisms' bounded-role components. This reframe is fair: the question of whether cheap-tier bounded roles handle long-horizon is not mechanism-specific. The agent offered Spike ν as the way to test this. The practitioner chose Spike ν.

**Assessment:** The asymmetry argument is partially load-bearing but partially reductive. The practitioner's gap (long-horizon capability ceiling of bounded roles) was acknowledged and re-framed correctly to the right empirical question (can cheap-tier bounded roles handle this, regardless of mechanism). The reframe is not a sophisticated absorption that re-centers ADR-027 while appearing neutral — it genuinely opens the question about both mechanisms' bounded-role components. Spike ν's pre-specification is the structural response to the gap. The asymmetry argument did not prevent Spike ν from being added; it correctly situated Spike ν as the gate artifact. This is earned confidence, not framing adoption.

**Verdict on Pattern 1:** No Grounding Reframe warranted. One residual: the asymmetry argument's "Tier 1 doesn't rescue long-horizon" framing is directionally correct but elides an important detail — Tier 1's bounded roles would be the same bounded roles (or equivalent ones) that ADR-027 uses. If the bounded roles fail at long-horizon, both mechanisms fail equally; if they succeed, both succeed. The asymmetry is in how fully the surface is covered by bounded roles (ADR-027 covers all NL routing; Tier 1 covers only explicit-`tool_choice`), not in which mechanism's bounded roles are more capable. The Spike ν framing correctly captures this; the asymmetry argument in the gate record could have been clearer on this point. Carry-forward as Advisory A.

---

### Pattern 2: Spike ν Pre-Specified Criteria — Substantive or Methodologically-Clean Cover?

**Observed:** Spike ν pre-specifies three pass/fail/intermediate criteria across three surfaces (multi-step composition, production-scale numerical content, adversarial routing). Criteria were committed before naming the spike as Track A.3, per MODEL snapshot Advisory A (pre-specify before running).

**Evaluation:** The criteria are well-structured and appear calibrated to meaningful structural-bounding test surfaces:

- Multi-step composition (≥80% Pass; <50% Fail; 50-80% Intermediate) uses a threshold that acknowledges cheap-tier reliability is imperfect and routes the intermediate band to a deployment policy rather than to architecture revision.
- Production-scale numerical content (≥95% Pass; <80% Fail; 80-95% Intermediate) aligns with Spike ε' Mode 1/2 rounding-drift characterization — the 95% threshold is demanding relative to the baseline, and the intermediate band triggers the rounding-drift mitigation playbook (ADR-029) rather than architecture revision.
- Adversarial routing (100% JSON conformance + ≥80% defensible-judgment-match for Pass; <80% JSON conformance for Fail) matches Spike ζ's test methodology, making the comparison directly interpretable.

The critical evaluation question is whether the "all-Pass → proceed with strengthened empirical floor" outcome is pre-specified in a way that makes it genuinely falsifiable, or whether the thresholds are set at levels the existing evidence already suggests will be cleared.

For multi-step composition, the threshold is 80% successful end-to-end at qwen3:8b. Spike ε established that single-step composition works correctly (n=3 tests). Multi-step composition is genuinely untested — Spike δ used a deterministic hardcoded chain, not a planner-driven multi-step plan. The 80% threshold is not pre-confirmed by existing evidence; it is a genuine prediction about untested behavior. This is appropriately designed.

For adversarial routing, 100% JSON conformance extends Spike ζ's battery from 20 to 40 prompts. The existing 20-prompt battery achieved 100% conformance; extending to 40 adversarial prompts is a genuine stress test. The ≥80% defensible-judgment threshold allows for harder adversarial prompts where the planner makes a defensible but not strict match. This is appropriately calibrated.

The intermediate-band analysis: the roadmap specifies that "any Intermediate → record finding as caveat-with-deployment-policy; update ADR-029 or ADR-031 per the relevant playbook; WP-A proceeds with the updated deployment policy as a constraint." This is the pattern to examine for methodological-clean cover. The intermediate outcome allows WP-A to proceed — which is the same architectural path as the all-Pass outcome. The difference is that ADR-029 or ADR-031 receives a deployment-policy update. Whether those updates would be meaningful constraints or soft acknowledgments depends on how much operational force the "caveat-with-deployment-policy" framing has in practice. The roadmap does not specify what "updates ADR-029 or ADR-031 per the relevant playbook" means concretely when all three surfaces fall in the intermediate band simultaneously. Multiple-intermediate-band outcomes could accumulate into an architecture that has been caveated on every dimension without architecture revision being triggered — the Fail condition requires a single surface to fail below the lower threshold, which becomes harder to trigger if the bounds are set conservatively enough.

**Assessment:** The criteria are substantively well-designed for the two genuinely untested surfaces (multi-step composition, adversarial routing). The numerical-content criterion is the weakest, as the existing Spike ε' B1 establishes the rounding-drift characterization at 25 figures — expanding to 100-figure + structured tables is a meaningful extension, but the ≥95% pass threshold is very tight, and the intermediate band (80-95%) routes to the mitigation playbook rather than architecture revision. A 94% fidelity result on the numerical surface would be "Intermediate" but would trigger only a playbook update. The multiple-intermediate-band accumulation risk is present but is bounded by the Fail threshold on JSON conformance (which is 100% or <80%, a binary-adjacent criterion). Advisory B: the BUILD build plan should track cumulative caveat-with-deployment-policy outcomes explicitly so that multiple intermediate findings don't individually appear marginal while collectively indicating a ceiling.

---

### Pattern 3: Module Decomposition — Inversion Principle Checks Substantive?

**Observed:** The system-design.agents.md carries Inversion Principle checks for the Dispatch Pipeline, Routing Planner, Response Synthesizer, Capability List Builder, Response Labeling, and Capability Discovery Endpoint modules. The checks are present in the companion file.

**Evaluation:** The Inversion Principle question is: does the module boundary serve user mental models, or developer convenience?

For the Dispatch Pipeline + Routing Planner + Response Synthesizer decomposition:

The Dispatch Pipeline's inversion note (from the companion file, line ~441, partially extracted via grep) describes the orchestration boundary in terms of the three-stage pipeline's architectural logic. The user mental model in play is the Operator's "I configured the serving layer and it routes to ensembles" and Population A's "the endpoint behaves like a model." The three-module decomposition does not map to either user mental model — neither the Operator nor Population A conceptualizes "Plan stage," "Dispatch stage," and "Synthesize stage" as distinct architectural entities. The decomposition maps to the developer's need to separate the LLM-powered routing decision (Routing Planner) from the deterministic dispatch (Dispatch Pipeline stage 2) from the LLM-powered synthesis (Response Synthesizer).

This is not inherently a problem — developer convenience in module decomposition is legitimate when the user-mental-model boundary is at the aggregate level (the whole Dispatch Pipeline as the new routing mechanism). What would be concerning is if the three-module decomposition creates complexity that BUILD engineers must carry but that doesn't reflect user-facing invariants. The strongest argument for the decomposition serving user mental models: the Routing Planner and Response Synthesizer are separately testable, separately calibrated (each has its own Calibration Gate trajectory under OrchestratorToolDispatch's interposition), and separately operator-replaceable (operators can substitute cheaper/faster planner models independently of synthesizer models). These are operator-facing concerns — the modularity does serve the Operator's mental model of "I configure each role independently."

The weakest inversion note in the set is for Capability Discovery Endpoint (companion file line ~500-505). The module's inversion note is not visible in the grep output, suggesting it may be brief. The endpoint is an L3 module whose three candidate surfaces (BUILD picks one) have not been committed — so the inversion check is against a partially-open design surface. The user mental model for the endpoint (Population A tools discovering capability ensembles) is clear, but whether the endpoint surface the BUILD picks (a `/v1/models` extension, a sibling endpoint, or response metadata) best serves that mental model is a BUILD-time decision left open by ARCHITECT. This is acknowledged in the roadmap's Open Decision Point #1.

**Assessment:** The Inversion Principle checks are present. The three-stage module decomposition (Dispatch Pipeline + Routing Planner + Response Synthesizer) primarily serves developer-architectural concerns (separate LLM invocations at separate stages; separate calibration surfaces; separate model profiles) rather than user mental models directly. This is not a failure — it's the correct decomposition for the architectural requirement. However, the inversion checks as presented in the companion file should acknowledge that the decomposition serves operator-configurability (separate model profiles for planner and synthesizer) and developer-testability, not Population A's user mental model. The Capability Discovery Endpoint's inversion note should be confirmed populated (not accidentally empty). Advisory C.

---

### Pattern 4: AS-9 + AS-10 Architectural Reach — Genuinely Invariant or Contract-Erosion-Prone?

**Observed:** Both invariants were codified earlier in the cycle and ARCHITECT inherited them. "AS-9 satisfaction is universal" on the chat-completions surface is a repeated claim. AS-10 (no client-side opt-in mechanism) is expressed via the Capability List Builder's filter logic and the Routing Planner's input contract.

**Evaluation:** AS-9 universality on the chat-completions surface is structurally enforced by FC-28 (no OrchestratorRuntime construction in the handler) + FC-29 (per-request observability confirming zero OrchestratorRuntime LLM calls). These are testable and would catch a BUILD engineer who bypasses the Dispatch Pipeline for a special case. The concern is future erosion: the companion file records that the "Serving Layer → Orchestrator Runtime" edge is "preserved as dormant" (not removed), and the OrchestratorRuntime class remains in the codebase. A future cycle that wants to add a fast-path could potentially re-activate the dormant edge for a "simple case" without triggering the FC-28/FC-29 checks if those checks are not maintained. This is a latent erosion risk, not a current violation.

AS-10 (no client-side opt-in) is enforced by the Routing Planner's input contract (FC-30 — routing decisions are invariant under custom HTTP headers). The concern here is that "no client-side opt-in mechanism" is a negative property — it forbids things that are not present yet. If a future cycle adds a legitimate opt-in mechanism (e.g., a Population B advanced feature), AS-10 would need to be amended. The amendment pathway exists (domain-model Amendment Log); the risk is that a future cycle adds an opt-in feature without recognizing it as an AS-10 amendment, incrementally eroding the invariant without deliberate revision.

**Assessment:** The invariants are structurally enforced by specific fitness criteria (FC-28, FC-29, FC-30). The structural enforcement is the correct way to make invariants durable. The latent risks are version-drift risks (FC-28/FC-29 checks not maintained across future cycles; AS-10 eroded incrementally by opt-in features), not current architectural failures. The BUILD phase should ensure FC-28 and FC-29 are written as static + integration checks that would survive the OrchestratorRuntime class remaining in the codebase. Informational observation.

---

### Pattern 5: OrchestratorRuntime Preservation — Honest Cost Accounting?

**Observed:** ARCHITECT chose disposition (a): preserve OrchestratorRuntime as architectural option with no production caller on the chat-completions surface post-BUILD. The system-design.md and roadmap.md both characterize the cost as "architectural-option preservation not actively-maintained dual surfaces."

**Evaluation:** The "architectural-option preservation" characterization is not dishonest, but it is optimistic. What the characterization does not account for:

1. **Version drift.** OrchestratorRuntime is a non-trivial class (the ReAct loop that the entire Cycle 4/5/6 corpus was architecting around). Without an active production caller, the class will not receive integration tests that exercise it under new infrastructure (Dispatch Event Substrate, DispatchEnvelope, Calibration Gate extensions from Cycles 4-7). New infrastructure that `OrchestratorRuntime` depends on will evolve; the Runtime will not. Within one or two cycles, re-activating the class will require non-trivial integration repair.

2. **Test suite maintenance.** FC-28 checks that no OrchestratorRuntime construction occurs in `chat_completions()`. If a future engineer wants to re-activate the class for a new surface (per the architectural-option framing), they would need to update FC-28 to exclude that new surface — a maintenance task that requires understanding the history of the check.

3. **ADR-022 system-prompt amendment dormancy.** ADR-022's amendment is "preserved" for the ReAct model but has no live codebase surface. The amendment's effectiveness under production clients has been empirically bounded to bare-endpoint mode. A future cycle that re-activates OrchestratorRuntime would inherit an amendment whose effectiveness bounds were characterized under Cycle 6 conditions, which may have changed.

None of these rise to a Grounding Reframe level — the ARCHITECT phase correctly framed the three options (preserve/wire-CLI/remove) and the practitioner accepted disposition (a). The characterization as "architectural-option preservation" is a reasonable description of the intent. The concern is that BUILD and future cycles should understand the actual maintenance burden is higher than "zero, because it has no caller." Advisory carry-forward for the BUILD phase: document the OrchestratorRuntime version-drift risk explicitly in the class's docstring or in a follow-on issue, so future engineers who encounter the class understand it is a dormant option with an accumulating integration repair cost.

---

### Pattern 6: FC-28..FC-41 Decomposition Completeness — Spiritually Substantive or Technically-Testable but Spiritually Empty?

**Evaluated claims:** "configuration honesty," "structurally-bounded," "transparent OpenAI-compatible endpoint."

**Configuration honesty (FC-38):** FC-38 tests that the three-layer signaling is present on every response and that the `served_by` value is one of the four canonical values. This is genuinely testable and would catch a case where the signaling is missing or incorrect. It tests the mechanism of configuration honesty (the labeling), not the outcome (whether the operator actually understands what the endpoint served). The "configuration dishonesty" signal the OQ #18 research identified (Cline #10551, OpenCode #20859) was about operators trusting that the endpoint would use ensembles and discovering it had not — that trust failure is not tested by FC-38 (the labeling is present; whether the operator reads and acts on the labeling is behavioral, not structural). This gap is acknowledged in the honest-residual-uncertainty annotation on the cost-distribution-accountability sub-promise. FC-38 is the correct structural decomposition; the outcome-level question is correctly left to PLAY.

**Structurally-bounded (FC-29):** FC-29 tests that zero OrchestratorRuntime LLM calls occur on the chat-completions surface and that routing decisions trace to the routing-planner ensemble. This tests the structural property (the right modules are invoked) without testing whether those modules produce the structural-bounding behavioral benefit (reliable output on single-decision-shaped tasks, not confabulating). The behavioral benefit is tested by FC-32 (fabrication-prevention regression battery). Together FC-29 + FC-32 cover the structural and behavioral dimensions of "structurally-bounded." The decomposition is substantive.

**Transparent OpenAI-compatible endpoint (FC-30, FC-39):** FC-30 tests that routing decisions are invariant under custom HTTP headers. FC-39 tests that the `tool_choice` bridge advisory fires correctly. These test the negative property (no opt-in mechanism) and the bridge mechanism's conditional advisory behavior. They do not test whether the OpenAI protocol contract is fully preserved (the request/response shapes, the SSE streaming format, the tool-call round-trip protocol). FC-35 tests chunk vocabulary preservation. The OpenAI-protocol contract preservation tests are distributed across FC-28/FC-34/FC-35/FC-38/FC-39 — no single test covers the full contract. This is appropriate (the contract is multi-dimensional); a BUILD-phase integration test that exercises a full Population A tool (OpenCode or Aider) sending a real request and receiving a conformant response would be the closest thing to a single comprehensive test, and that is appropriately PLAY-phase work.

**Assessment:** The FC decomposition is genuinely refutable. The "direction-not-constraint" annotation on the latency tuning playbook is honest (the four tuning mechanisms are operator-configurable, not testable as a single constraint). The "honest-residual-uncertainty" annotation on the cost-distribution-accountability sub-promise is honest (Population A voice is silent; the sub-promise is project-developer-lens grounded). No spiritually-empty decompositions found. Informational observation.

---

### Pattern 7: Roadmap WP Scope Realism

**Evaluated:** Whether the 5 WPs + 2 Track A refactors + Spike ν hide work the BUILD phase will discover.

**WP-A (Dispatch Pipeline):** The Plan → InternalToolCall adapter integration with `OrchestratorToolDispatch.dispatch()` is the most non-trivial mechanical piece. The adapter's contract is clearly specified (InternalToolCall shape; the dispatch() chokepoint preserves existing interpositions). The streaming chunk vocabulary preservation constraint (FC-35) requires that the Dispatch Pipeline yields ContentDelta | VisibilityEvent | ClientToolCall | Completion — the same types the OrchestratorRuntime currently yields. This requires careful wiring of the synthesizer's streaming output into the pipeline's chunk surface. The OrchestratorRuntime's streaming output handling is existing code that the Dispatch Pipeline must replicate in structure without using the Runtime. This is a real engineering task that the roadmap's WP-A description acknowledges but does not fully scope (it says "the pipeline's public surface yields the existing chunk vocabulary" without specifying how that wiring differs from the OrchestratorRuntime's wiring). Moderate hidden-work risk.

**WP-C (Response Synthesizer streaming):** The companion file integration contract (line ~1010) notes that "for streaming requests, dispatches in streaming mode and yields ContentDelta chunks as tokens arrive." This requires the synthesizer ensemble to support streaming dispatch — which requires the EnsembleExecutor to support streaming per-ensemble, which was not a required surface in prior cycles' WPs. Whether the EnsembleExecutor currently supports streaming at the ensemble level (not just at the serving-layer SSE level) is a BUILD-discovery question. If it does not, WP-C has hidden infrastructure work.

**Track A.3 (Spike ν):** The spike is scoped at $0 cost (qwen3:8b local Ollama). The 40-prompt adversarial routing battery is twice the size of Spike ζ's battery; the multi-step composition surface is genuinely novel (Spike ε used three tests, two of which were single-step or hardcoded chains). The spike's wall-clock time could approach 1-2 hours at the stated scope. This is not a schedule risk but should be accounted for in BUILD entry timing.

**Assessment:** Two moderate hidden-work risks: (1) WP-A streaming wiring complexity for the Dispatch Pipeline's chunk vocabulary preservation; (2) WP-C EnsembleExecutor streaming-at-ensemble-level availability. Neither rises to a Grounding Reframe level — both are well-understood engineering challenges, not architectural gaps. Advisory D for BUILD entry.

---

## Interpretation

### Pattern Assessment

The ARCHITECT phase produced a structurally complete architecture for the Cycle 7 ADRs. The phase ran in a single session with substantial reconnaissance (code + artifacts), presented Track C dispositions to the practitioner, and added Spike ν at the EPISTEMIC GATE in response to a genuine practitioner-identified gap. The module decomposition is coherent and its fitness criteria are substantively refutable.

The susceptibility signals at this boundary are different in character from prior cycle boundaries. RESEARCH and DISCOVER showed classic susceptibility patterns (unquantified assertions, rapid compounding). MODEL showed the procedural documentation gap in spike methodology. DECIDE showed per-ADR alternatives deliberated-but-not-practitioner-challenged. ARCHITECT shows a more subtle pattern: an architecture that correctly inherits committed decisions from DECIDE but inherits them at depth — the Inversion Principle checks are present but lean architectural rather than user-mental-model; the OrchestratorRuntime cost accounting is optimistic; the asymmetry argument at the gate is partially reductive; the Spike ν intermediate-band accumulation risk is not explicitly scoped.

None of these individually constitute a Grounding Reframe trigger. Collectively they represent the ARCHITECT-phase variant of the "rapid compounding" signature: not compounding of empirical findings, but compounding of architectural choices inherited from DECIDE without independent reexamination at the decomposition layer.

### Earned Confidence vs. Sycophantic Reinforcement

The phase shows earned confidence on most dimensions. The code reconnaissance was substantive (three candidate OrchestratorRuntime dispositions derived from actual code inspection, not from ADR text). The Track C triage was presented as genuine options. The EPISTEMIC GATE produced a real spike commitment (Spike ν) in response to a real gap, not a pro-forma acknowledgment. The fitness criteria are genuinely refutable.

The specific site of possible reinforcement is the EPISTEMIC GATE asymmetry argument: the agent's reframe ("does ANY single-step-plus-bounded-role architecture handle generalized long-horizon at cheap tier?") is correct but was composed quickly under the pressure of the gate's belief-mapping context, and it successfully re-centered Spike ν as a shared-question probe rather than as an ADR-027-specific vulnerability test. Whether this was the correct framing or was a subtle reframe that neutralized the practitioner's challenge depends on whether the practitioner's question was about the bounded roles' capability (a shared question) or about ADR-027's specific architectural choice (a mechanism-specific question). The evidence supports the shared-question reading; the asymmetry argument does not appear to have been a manipulation of the framing.

### Advisory Carry-Forward Status

All DECIDE advisories carry forward appropriately. The five DECIDE advisory carry-forwards are encoded in the roadmap (OrchestratorRuntime disposition as required deliverable; confabulation mode validation quality in BUILD regression suite; Rule 5 BUILD execution risk; ADR-030 disposition-(i) timing; per-ADR revisitability). The three MODEL advisories (Advisory A spike methodology; Advisory B options-engagement; three DISCOVER advisories) are fully absorbed into DECIDE's work products and do not carry separately into BUILD.

---

## Findings

### Finding 1 — Asymmetry Argument Partial Reductiveness (Severity: ADVISORY — Advisory A)

The EPISTEMIC GATE asymmetry argument ("Tier 1 doesn't rescue long-horizon because non-`tool_choice` NL routing under Tier 1 continues through the orchestrator-LLM-as-decider") is directionally correct but elides a meaningful detail: Tier 1 hybrid's bounded-role components would be the same routing-planner and response-synthesizer ensembles that ADR-027 uses. The asymmetry between the two mechanisms is in surface coverage (ADR-027 covers all NL routing via bounded roles; Tier 1 covers only explicit-`tool_choice` requests), not in the bounded roles' capability at long-horizon. Spike ν tests the shared empirical question (can cheap-tier bounded roles handle these surfaces?).

The argument as recorded in system-design.md Amendment Log entry #11's "asymmetry note" correctly identifies the failure locus (orchestrator-LLM-as-decider under non-`tool_choice` NL routing in Tier 1); it does not clarify that Tier 1's bounded-role components would be identically capable or incapable at long-horizon as ADR-027's bounded-role components. BUILD and PLAY should understand Spike ν as testing a question that applies equally to both mechanisms' bounded-role components — not only as validating ADR-027 against Tier 1.

### Finding 2 — Spike ν Intermediate-Band Accumulation Risk (Severity: ADVISORY — Advisory B)

The pre-specified Spike ν criteria are well-designed for individual surfaces. The roadmap's handling of "any surface Intermediate → WP-A proceeds with updated deployment policy" means that multiple simultaneous intermediate findings would each individually be classified as non-fatal. If all three surfaces fall in the intermediate band, WP-A proceeds with three concurrent deployment-policy caveats attached to ADR-029 and ADR-031. The resulting architecture would be caveated on multi-step composition, numerical fidelity, and adversarial routing simultaneously — which collectively might constitute a more serious capability ceiling than any single intermediate finding suggests. The trigger conditions as specified do not account for this accumulation scenario.

Recommended action for BUILD entry: before running Spike ν, add an explicit assessment rule for the multiple-intermediate scenario to the roadmap's Track A.3 entry: "If two or more surfaces produce Intermediate results simultaneously, treat as equivalent to a single Fail for the purpose of triggering Design Amendment deliberation." This makes the spike's falsification surface symmetric with the individual-Fail condition.

### Finding 3 — Inversion Principle Checks Lean Architectural (Severity: ADVISORY — Advisory C)

The Inversion Principle checks for the Dispatch Pipeline, Routing Planner, and Response Synthesizer modules in system-design.agents.md document the architectural rationale for separate modules (separate LLM invocations; separate calibration surfaces; separate operator-configurable model profiles). These serve the Operator's mental model of independent configuration but do not clearly articulate whether the three-module decomposition would be opaque to Population A users or confusing to BUILD engineers who must maintain three module boundaries for what is conceptually a single architectural mechanism.

The check for Capability Discovery Endpoint's inversion note was not visible in the grep output — this may indicate it is absent or very brief. If absent, it should be populated given that the endpoint's user mental model (Population A clients discovering capabilities) is clear but the BUILD surface-choice (three candidates) is open.

Recommended action: confirm Capability Discovery Endpoint inversion note is populated; add a note to the Dispatch Pipeline inversion check acknowledging that the three-module decomposition serves operator-configurability (independent model profiles for planner and synthesizer) and developer-testability (separate regression suites), not Population A's mental model directly.

### Finding 4 — OrchestratorRuntime Maintenance Cost Under-Accounted (Severity: ADVISORY — Advisory D)

The "architectural-option preservation" characterization for OrchestratorRuntime disposition (a) does not account for the accumulating integration repair cost as the surrounding infrastructure (Dispatch Event Substrate, DispatchEnvelope, Calibration Gate extensions, Capability List Builder) evolves without the Runtime having a live caller. Within one or two cycles, re-activating OrchestratorRuntime for a new surface will require non-trivial integration work. The current characterization suggests the cost is zero (no active caller, no active maintenance), which is optimistic.

Recommended action: add a version-drift advisory note to the OrchestratorRuntime module entry in system-design.agents.md (or as a BUILD-phase deliverable: a comment in the class's docstring) documenting that re-activation for future surfaces will require integration repair and ADR-022 amendment re-validation under current-state production conditions. This makes the cost of disposition (a) visible to future cycle practitioners.

### Finding 5 — WP-A/WP-C Hidden Work Risks (Severity: INFORMATIONAL — Advisory D for BUILD entry)

Two moderate hidden-work risks are present in the Cycle 7 BUILD scope:
- WP-A streaming wiring: the Dispatch Pipeline must replicate the OrchestratorRuntime's streaming chunk output surface without using the Runtime's internal streaming handling. The companion file specifies the output vocabulary (ContentDelta | VisibilityEvent | ClientToolCall | Completion) but does not specify how the synthesizer's streaming token output is wired into the pipeline's chunk surface.
- WP-C EnsembleExecutor streaming: the Response Synthesizer module relies on "dispatch in streaming mode and yield ContentDelta chunks as tokens arrive" — whether the EnsembleExecutor currently supports streaming at the per-ensemble level is not confirmed in the ARCHITECT artifacts.

These are BUILD-discovery risks, not architectural gaps. Recommend BUILD entry include a targeted reconnaissance of the EnsembleExecutor's streaming API before WP-C work begins.

### Finding 6 — FC Decomposition Substantive (Severity: INFORMATIONAL — positive signal)

The ADR-076 qualitative-claim decomposition for Cycle 7 is the most complete in the corpus. Every qualitative claim is decomposed into refutable fitness criteria (FC-28..FC-41). The direction-not-constraint annotation on the latency tuning playbook is honest. The honest-residual-uncertainty annotation on the cost-distribution-accountability sub-promise is honest. No spiritually-empty decompositions found. The FC-38 "configuration honesty" decomposition tests the mechanism (labeling present and correct) without claiming to test the outcome (operator behavior on seeing the label) — a correct and honest scoping decision.

---

## Recommendation

**No Grounding Reframe warranted.**

The ARCHITECT phase produced structurally complete, internally consistent architecture artifacts. The EPISTEMIC GATE produced a genuine spike commitment (Spike ν) in response to a genuine practitioner-identified gap. The fitness criteria are substantively refutable. The prior-cycle advisory trail is fully encoded in the roadmap and BUILD approach. The signals are consistent with earned confidence at the ARCHITECT boundary — a BUILD-resistant phase position that limits compounding risk.

**Four advisory carry-forwards for BUILD:**

**Advisory A (Spike ν and the asymmetry argument):** BUILD and PLAY should understand Spike ν as testing a question that applies to both ADR-027's and Tier 1 hybrid's bounded-role components equally — not primarily as validating ADR-027 against Tier 1. If Spike ν produces Fail or multiple-Intermediate results, the Design Amendment process should re-examine both mechanism choices, not only ADR-027. The asymmetry note in Amendment Log #11 is directionally correct; BUILD should not read it as "Tier 1 would have failed where ADR-027 succeeds" — the asymmetry is about surface coverage, not role capability.

**Advisory B (Spike ν multiple-intermediate scenario):** Before running Spike ν (Track A.3), add an explicit rule to the roadmap: if two or more surfaces simultaneously produce Intermediate results, treat the combination as equivalent to a single Fail for Design Amendment deliberation purposes. This prevents the intermediate band's individually-non-fatal classification from masking a collective capability ceiling signal.

**Advisory C (Inversion Principle checks):** Confirm the Capability Discovery Endpoint inversion note is populated. Add a note to the Dispatch Pipeline inversion check clarifying that the three-module decomposition serves operator-configurability and developer-testability, not Population A's user mental model directly — so that future BUILD engineers don't assume the three-module structure has user-facing invariants it doesn't have.

**Advisory D (OrchestratorRuntime and hidden BUILD work):** Document the OrchestratorRuntime version-drift and integration repair cost explicitly in the class (docstring or companion file note), so future cycles that want to re-activate it understand the maintenance cost is accumulating. Before WP-C begins, confirm the EnsembleExecutor's per-ensemble streaming API surface is available; if not, scope the streaming infrastructure work as a WP-C prerequisite.

**Five active carry-forwards inherited from prior phases, unchanged:**

- DECIDE Advisory #1 (OrchestratorRuntime disposition as required BUILD deliverable — addressed: disposition (a) selected)
- DECIDE Advisory #2 (confabulation mode validation quality differentiation in BUILD regression suite — addressed in roadmap's commitment-gating output #5)
- DECIDE Advisory #3 (Rule 5 BUILD execution risk — addressed in roadmap's commitment-gating output #3)
- DECIDE Advisory #4 (ADR-030 disposition-(i) implementation timing — addressed in roadmap Open Decision Point #2 for next DECIDE gate)
- DECIDE Advisory #5 (per-ADR rejected-alternatives revisitability in PLAY — addressed in roadmap's commitment-gating output)

---

*Snapshot produced in isolated evaluation context. Advisory only; does not block BUILD phase progression.*

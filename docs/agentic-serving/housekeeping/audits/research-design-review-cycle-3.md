# Research Design Review — Cycle 3 (Agent Design)

**Reviewed question set:** `docs/agentic-serving/essays/research-logs/research-log.md` §Step 1.1 (three RQs) + §Step 1.2 (constraint-removal exchange)
**Constraint-removal response included:** yes
**Date:** 2026-05-01
**Criteria applied:** 1–4 (ADR-082)

---

## Summary Assessment

**Recommendation: Clear — with two P2 notes for practitioner judgment and one P3 observation.**

The question set is well-formed overall. RQ-1 is the most carefully constructed of the three — it applies the susceptibility snapshot's specific grounding action exactly as prescribed, names its fixture and scope boundary explicitly, and provides a decision tree that keeps all three outcome-branches open. RQ-2 has a genuine falsification criterion and a named retirement path for the frame if the test fails. RQ-3 is anchored appropriately on a published baseline and connects to Cycle 2's open territory.

Two issues require practitioner judgment before the loop begins. First, RQ-1's artifact-anchored comparison baseline ("A2 + script input vs A3") correctly isolates A3's load-bearing component but quietly reintroduces a framing the constraint-removal exchange bracketed: that "agent design" in this cycle means configurations comparable in shape to A3. The constraint-removal response opened space for structurally simpler agent shapes — script-as-orchestrator with LLM-as-subordinate-step — and the question set does not surface whether any of those shapes are worth examining before the cycle locks its spike battery. This is not a flaw in RQ-1 as a grounding question; it is a coverage gap at the question-set level that the practitioner should acknowledge and either consciously scope out or address with a targeted sub-question.

Second, RQ-3's failure-mode framing ("meltdown rate") reflects the most prominent published failure mode but does not acknowledge adjacent failure modes the literature names (subtle quality degradation without meltdown, premature stop, error self-conditioning). The phrasing does not preclude examining them, but it does not invite them either; a researcher reading RQ-3 as a mandate would design a spike optimized to detect meltdowns and might not instrument for other failure patterns.

Neither issue is severe enough to require revision before the loop begins — both are correctable at spike-design time. The practitioner should record rationale for scope decisions at Step 1.4.

---

## Per-Criterion Findings

### Criterion 1 — Need-vs-Artifact Framing

#### RQ-1: artifact-anchoring — structurally necessary, with a residual coverage gap

RQ-1 names A2 and A3 as its comparison points: "Does a prompt-steered single cloud orchestrator receiving a script-agent's deterministic report as additional input context ('A2 + script input') produce equivalent factual grounding to A3's novel ensemble ... on the cycle-2 README-review task class?"

The artifact-anchoring is structurally necessary for this question's stated purpose. The susceptibility snapshot's grounding action is specific and actionable: "Before the Loop 1 literature review dispatch, add an explicit research question — Does a prompt-steered single orchestrator receiving a script-agent's deterministic report as input context produce equivalent factual grounding to A3's novel ensemble? — and commit the spike battery to testing this alternative before synthesizing ensemble-topology findings." RQ-1 implements this exactly. The A3 configuration is not arbitrary — it is the specific artifact whose load-bearing component is unresolved, and the comparison must be against A3 to answer the question the snapshot identified.

**P2 — coverage gap from constraint-removal response:** The constraint-removal exchange explicitly bracketed the premise that "the agent" is "pre-committed to 'a single LLM dispatching a fixed set of tools' at all," and the practitioner responded "it's all about outcomes over an agentic session." But RQ-1 tests A2+script-input vs. A3 — both of which are configurations in which an LLM is the orchestrator and scripts are either input context or ensemble slots. The operating-frame statement ("agent shape is means") is named in the research plan, but no question asks whether agent shapes that do not have an LLM as the primary orchestrating component have published evidence on the cycle's task classes. The constraint-removal response opened space for examining e.g. script-driven processes where an LLM call is a bounded step rather than the orchestrating agent. RQ-1 tests a narrower version of the constraint-removal's intended territory. This is not a flaw in RQ-1's grounding function — but the question set as a whole does not pick up the full territory the constraint-removal exchange opened. See Criterion 4 for the specific incongruity.

#### RQ-2: "Cycle-3-tested configurations" — open enough given the research plan's defaults

RQ-2 operationalizes its frame test against "Cycle-3-tested configurations" and explicitly commits to scoring each configuration on all four priorities. The research plan's approved defaults include explicit coverage of "agent-shape alternatives (script-driven loops, hierarchical, state-machine)" in the lit review. So "tested configurations" is not an artificially fixed set — the operating-frame statement ("agent shape is means") and the lit-review scope together mean Cycle 3 is authorized to surface and test alternatives. RQ-2's phrasing is permissive on this dimension. **No flag.**

#### RQ-3: "llm-orc's deployment configurations" — mild artifact-anchoring, appropriate at this stage

RQ-3 asks "Does the tau-bench ... reliability ceiling generalize to llm-orc's deployment configurations?" This is artifact-anchored to the existing system, but the question's purpose is to test an inherited empirical open question against the system the practitioner actually operates. Anchoring to "llm-orc's deployment configurations" at the first-pass probe is appropriate; the more important criterion is whether the question also asks about alternative agent shapes' published evidence on multi-turn outcomes. The research plan's §"RQ-3 supporting literature" explicitly includes "agent shapes departing from LLM-with-tools (script-driven loops with LLM as subordinate step, hierarchical task decomposition, deterministic state machines with LLM nodes) and their multi-turn evidence base." That coverage instruction addresses the artifact-anchoring concern adequately at the question-set level. **P3 note only:** The question's phrasing could invite explicit comparison (e.g., "and do alternative agent shapes in the literature show a different reliability floor?"), but the research plan's lit-review scope implicitly covers this.

---

### Criterion 2 — Embedded Conclusions

#### RQ-1: "equivalent" — sharpness of the falsification term

RQ-1 asks whether A2+script-input produces "equivalent factual grounding" to A3. The term "equivalent" is not precisely defined in the question. The decision tree partially operationalizes the comparison: the three branches (equivalent at A2's latency / worse grounding / equivalent with different latency-cost profile) imply that "equivalent" is measured on factual-grounding correctness and latency. The research plan's Phase 2 Spike A section adds specificity: "factual-grounding correctness (did it find the undefined-model-profile documentation bugs A2 missed and A3 caught?), recommendation count and specificity, latency, token cost, output structure."

The concrete fixture anchor (the specific undefined `default-local` and `ollama-gemma-small` model-profile bugs from essay 003's Spike A3) makes "equivalent factual grounding" testable on this fixture — a configuration either surfaces those bugs or it does not. That binary makes equivalence falsifiable on the fixture's most critical dimension.

**P2 — escape hatch risk on the broader equivalence claim:** "Equivalent factual grounding" on the specific fixture is well-defined. But if A2+script-input finds the bugs A3 found, the question becomes whether the finding generalizes to "equivalent factual grounding on the task class" — i.e., not just this one fixture. RQ-1 scopes the finding to the fixture ("Generalization to other task classes is a follow-up"), which is methodologically sound, but the decision tree's lesson statements are framed at the architecture level: "architectural lesson is 'augment prompt-steering with deterministic tool outputs,' not 'use ensemble topology.'" An architectural lesson drawn from a single fixture match is a significant generalization step. The research plan's "narrow reading first" discipline (from the susceptibility snapshot) should apply here at synthesis time: the lesson should be scoped to "on this fixture" before any architecture-level characterization is offered.

This is not a flaw in RQ-1's framing — the scope condition is explicit. It is a flag for the spike-synthesis discipline: the equivalent-finding branch's decision-tree lesson should carry the fixture-scope qualifier explicitly when written into the synthesis exchange.

#### RQ-2: falsification criterion — is the escape hatch real?

RQ-2's falsification criterion states: "if no Cycle-3-tested configuration produces a frame-divergent recommendation, the four-priorities frame is rhetorical and Cycle 3 retires it."

The operationalization notes that "environmental cost and local-first qualitative; performance and token cost measured." The asymmetry is real and worth naming. Qualitative scoring on two of four dimensions means a frame-divergent recommendation on those two dimensions is harder to establish rigorously — a scoring judgment that "configuration X has lower environmental cost" is not the same kind of evidence as a measured latency figure. If the cycle runs and the qualitative scores happen to align with performance-only recommendations, it is genuinely ambiguous whether alignment reflects frame convergence or scoring-resolution limits.

**P2 — operationalization escape hatch:** The falsification criterion has a structural weakness: qualitative scoring can fail to detect frame divergence even when divergence is real (low resolution), or can produce apparent divergence from subjective scoring differences rather than from genuine priority conflicts (false positive). Neither failure mode is catastrophic — the frame-retirement conclusion would require arguing that qualitative scoring at the resolution the cycle can achieve is sufficient to claim no divergence was detectable — but the cycle should name this as a limit on the falsification's epistemic force at synthesis time. A suggested addition to the operationalization: "Note that qualitative scoring on environmental cost and local-first limits the falsification's resolution — a frame-convergent finding at qualitative resolution is weaker evidence than a measured-axis convergent finding. If qualitative scores are borderline or contested, the retirement criterion does not fully fire."

This addition would prevent a clean false-closure on the four-priorities frame if the qualitative scores simply lack the resolution to distinguish.

#### RQ-3: "meltdown rate" — crowding out adjacent failure modes

RQ-3 asks what agent-design choices "reduce the meltdown rate on multi-turn sustained work." Meltdown is one named failure mode from the literature (HORIZON's 19% figure; Khanal et al.'s reliability framework). The susceptibility snapshot and essay 003 also name: error self-conditioning, memory retrieval drift, premature stop, summarization broken, early-stall-under-30%-completion (LongCLI-Bench). The question's "meltdown rate" framing is accurate as a label for the most visible failure mode but does not explicitly invite the cycle to examine whether agent-design choices that reduce meltdown might increase another failure mode, or whether the dominant failure mode for llm-orc's task classes is meltdown at all vs. premature stop or error self-conditioning.

**P2 — phrasing crowds out adjacent failure modes:** The question is not falsely framed — meltdown is a real and published failure mode, and the question does not prohibit examining others. But a researcher designing the Spike B fixture from RQ-3 as written would naturally instrument for meltdown signatures and might not instrument equally for premature-stop detection or error-self-conditioning accumulation. The spike design (discussed under the research plan's Phase 3 multi-turn fixture) has not been finalized, making this a correctable scope issue at spike-design time.

---

### Criterion 3 — Prior-Art Treatment

#### Constraint-removal response — substantive, not performative

The constraint-removal response ("Right makes sense — I am not wedded to any predisposition for the agent. It's all about outcomes over an agentic session.") is substantive: the practitioner explicitly decoupled the cycle's evaluation axis from architectural shape. The agent's interpretation (recorded in the research log §Step 1.2 under "What this shifts") is accurate — the response brackets ADR-003 as prior art rather than treating it as a constraint, and the operating frame ("agent shape is means; evaluation axis is outcome quality per session") is the structural result.

The constraint-removal treatment of ADR-003 satisfies the prior-art criterion for Criterion 3. ADR-003 is the right artifact to bracket: Cycle 2 bracketed ADR-011 (the orchestrator-LLM-is-a-Model-Profile commitment), and the research log's reasoning is sound — for Cycle 3's territory of "agent design specifically," ADR-003's definition of the orchestrator's action space as "a single LLM dispatching a fixed set of five tools" is the more upstream and more definitional artifact. Bracketing it opens the right territory.

#### Inherited frames — operating frame substitutes adequately for explicit prior-art bracketing on the README-review task class and ADR-011

The cycle inherits three anchors: the README-review task class (RQ-1), the four-priorities frame (RQ-2), and llm-orc's deployment configurations (RQ-3). Prior-art treatment of each:

- **README-review task class (RQ-1):** The scope statement in RQ-1 is explicit: "same fixture as A3 ... Generalization to other task classes is a follow-up." This treats the task class as a bounded fixture for a specific isolation test, not as the canonical task class for "agent design." The operating frame ("outcomes over an agentic session") and the research plan's multi-turn spike (which uses a different fixture — tau-bench or real coding session) extend beyond the task class. Adequate treatment.

- **Four-priorities frame (RQ-2):** RQ-2 explicitly treats the frame as prior art — its falsification criterion, if met, retires or replaces the frame. The cycle-2 archive (item #26) records the frame as "hypothesis to be tested, not settled lens." The question is designed precisely to test whether the frame is load-bearing. This is model prior-art treatment.

- **llm-orc deployment configurations (RQ-3):** As noted under Criterion 1, the research plan's lit-review scope instruction includes explicit coverage of alternative agent shapes. The deployment configurations are tested, not assumed. Adequate.

**No flags under Criterion 3.** The prior-art criterion is satisfied by the constraint-removal response's substantive bracketing of ADR-003, and the inherited frames are treated as hypotheses rather than as constraints throughout.

---

### Criterion 4 — Incongruity Surfacing

#### First incongruity: the constraint-removal exchange opened a territory the question set does not fully cover

The constraint-removal exchange established that "agent shape is means" and the practitioner is not predisposed toward any particular shape. The research plan's operating frame statement repeats this verbatim. But the three RQs collectively investigate agent shapes that all share one structural feature: an LLM as the primary orchestrating agent. RQ-1 tests A2+script-input vs A3 — both LLM-orchestrated. RQ-2 tests the four-priorities frame across "Cycle-3-tested configurations" — but the configurations that will get tested are the ones RQ-1 and RQ-3 run, which are LLM-orchestrated. RQ-3 asks about meltdown in "llm-orc's deployment configurations" — LLM-orchestrated.

The specific incongruity: Cycle 2's evidence pointed toward the script-agent slot as the component that contributed the most unambiguous value in A3 (the verified link counts, confirmed section presence, and undefined-profile bugs were produced by the deterministic script, not by the LLM reviewers). The literature (essay 003's Loop 4 findings) names the script-agent slot as "load-bearing, not incidental." A structurally adjacent and simpler hypothesis — that the script-as-primary-process with a bounded LLM call for interpretation is the optimal shape for factual-grounding tasks, and the LLM-as-orchestrator is unnecessary complexity in that case — is not examined by any of the three RQs. RQ-1 tests whether the LLM orchestrator can deliver equivalent grounding when given the script's output as context. It does not ask whether a script that invokes an LLM as a subordinate step (rather than an LLM that invokes a script as an ensemble slot) produces a different outcome profile.

This is the script-as-orchestrator vs. LLM-as-orchestrator incongruity: in the domain of factual-grounding tasks, the simplest shape (script does the deterministic work; calls LLM once for prose synthesis) sits adjacent to the shape being tested (LLM orchestrates; receives script output as context), and the question set does not ask whether the two shapes differ on outcome quality, latency, or the four-priority scoring.

**P2 — script-as-orchestrator shape not surfaced:** The research plan's lit-review scope instruction mentions "script-driven loops with LLM as subordinate step" as a shape to cover in literature, which partially addresses this. But coverage in the literature review is not the same as a spike comparing the shapes empirically. Given that the cycle's central territory is "agent design specifically" and the constraint-removal exchange explicitly opened agent shape as an open question, the absence of an explicit sub-question on the script-as-orchestrator shape is a gap worth naming.

#### Second incongruity: script simplicity vs. ensemble complexity

Essay 003's Spike A3 found that the script-agent slot's value is deterministic and structural — it produced verified facts "LLM reviewers could not generate on their own." The script's three outputs (link validity, section presence, code-block parseability) are O(1) in reasoning complexity: a parsing script with no model call at all. Meanwhile, the most expensive and complex component of A3 (the two heterogeneous LLM reviewers from different model families) produces findings that a well-prompted single LLM sometimes also finds (A2 found some of the same issues A3 found, just not the undefined-profile bugs). The simplest component of A3 contributes the least ambiguous value; the complex components contribute incremental and overlap-prone value.

The incongruity: if the deterministic script is the value-producing component and the LLM reviewers are the complexity-introducing component, the most productive agent-design question might be "how far can we get with deterministic scripts alone, and what does an LLM step actually need to add to earn its presence?" The question set tests whether LLM-A can be simplified to LLM-B-plus-script-context; it does not test the floor of "what does any LLM contribute once the script has run?" This is adjacent to RQ-1 but not equivalent to it — RQ-1 tests whether the ensemble LLMs add grounding value; neither RQ-1 nor any other question tests whether any LLM is needed for grounding-class tasks at all.

This second incongruity is partially addressed by RQ-1's design — if A2+script-input matches A3's grounding quality, that is evidence that ensemble topology is not needed — but no question tests the further simplification of asking what value the single LLM adds beyond formatting the script's output.

**P3 — floor question not asked:** This is lower priority than the first incongruity because the cycle has a legitimate reason not to go to this extreme in a single cycle. Naming it as a potential feed-forward question for the cycle-3 synthesis.

---

## Per-RQ Recommendations

### RQ-1

The question is well-formed. One reformulation candidate for the decision-tree's lesson statements, to be applied at spike-synthesis time rather than at question-revision time:

**Current decision-tree language (equivalent-grounding branch):** "architectural lesson is 'augment prompt-steering with deterministic tool outputs,' not 'use ensemble topology'; ADR-011's boundary refinement does not need relaxation."

**Recommended discipline:** When the equivalent-grounding branch fires, the synthesis discipline (from the susceptibility snapshot's "commit to the narrow reading first" note) should scope the lesson to: "On the cycle-2 README-review fixture, A2+script-input produces equivalent factual grounding to A3 — the ensemble-topology contribution to grounding on this fixture is not load-bearing. Generalization to the architecture level (augment prompt-steering with tool outputs as a general policy) requires additional task classes." The architectural characterization is the broad reading and should follow from accumulated evidence, not from the single fixture.

**Optional sub-question to capture the constraint-removal territory** (practitioner can adopt, adapt, or reject):

> *Does the published literature describe agent shapes where a deterministic script or process acts as the primary orchestrator and an LLM is a bounded subordinate step (rather than the other way around), and what do those shapes' outcome profiles look like on factual-grounding tasks?*

This is literature-addressable and does not require a new spike. It fills the constraint-removal exchange's opened territory without adding a third spike.

### RQ-2

The question is well-formed. One operationalization amendment recommended:

**Current operationalization:** "for each spiked configuration, score on all four priorities (environmental cost and local-first qualitative; performance and token cost measured)."

**Recommended addition:** "Scoring resolution caveat: qualitative scoring on environmental cost and local-first has lower resolution than measured scoring. A frame-convergent finding at qualitative resolution does not constitute a clean falsification — it means the frame-divergent signal, if present, was below the scoring threshold. If all four configurations score qualitatively equivalent on both qualitative axes, record this explicitly as 'no detectable divergence at qualitative resolution' rather than as 'frames converge.' Retirement of the four-priorities frame requires either a measured divergence finding or a stated judgment that qualitative resolution is sufficient for the cycle's claims."

### RQ-3

The question is well-formed. Two recommendations:

**1. Widen the failure-mode framing at spike-design time.** When designing Spike B's instrumentation, explicitly name the failure modes to detect: meltdown, premature stop, error self-conditioning, memory retrieval drift, and early-stall (the LongCLI-Bench finding that most failures occur at under 30% completion). The spike should be instrumented for at least meltdown and premature stop before the fixture runs.

**Suggested reformulation candidate for practitioner consideration:**

> *Does the tau-bench multi-turn tool-dispatching reliability ceiling (Yao et al. 2024) generalize to llm-orc's deployment configurations? What agent-design choices reduce the rate of observable failure modes — including meltdown onset, premature stop, and error self-conditioning — on multi-turn sustained work?*

**2. Literature-review framing on alternative shapes.** The research plan already includes "agent shapes departing from LLM-with-tools" in the lit-review scope. Confirm at lit-review dispatch that the question explicitly asks for evidence on multi-turn reliability profiles for those shapes (not just their existence in the literature), so the lit-reviewer surfaces whether any alternative shape has a published multi-turn reliability floor that differs materially from the LLM-with-tools shape.

---

## Cycle-Specific Findings

### RQ-1 scope isolation — genuine isolation or pre-narrowing of cycle conclusions?

RQ-1's scope ("same fixture as A3; generalization to other task classes is a follow-up") is genuinely necessary for the isolation test. The susceptibility snapshot required testing this alternative before Cycle 3 synthesizes ensemble-topology findings, and that requirement is meaningful precisely because A3's finding was on a specific fixture. Testing A2+script-input on a different fixture would not isolate A3's load-bearing component — it would test a different question. The fixture scope is methodologically load-bearing, not an artificial narrowing.

However, the question does pre-narrow in one direction: the equivalent-grounding branch's architectural lesson ("augment prompt-steering with deterministic tool outputs") is an architecture-level characterization drawn from a single fixture finding. This is not a flaw in RQ-1's design — the decision tree explicitly labels these as lessons from the test, not as conclusions — but it is a scope mismatch that the synthesis discipline should actively manage. The cycle's overall agent-design conclusions cannot rest on a single-fixture isolation test, regardless of which branch fires.

**Structural finding (not a flag — this is a design virtue):** RQ-1 is well-scoped for its stated purpose. The cycle should not use a positive finding on the isolation test as a warrant for architecture-level characterizations without additional fixture coverage. The research plan's sequencing (synthesis exchanges drawing topology conclusions wait for Spike A's findings) is the right discipline, but "waiting" means incorporating Spike A's findings as one data point in a broader synthesis, not as the sole warrant for architectural recommendations.

### Script-as-orchestrator incongruity — the cycle's most important unasked question

The constraint-removal exchange's most significant implication — that agent shapes where the LLM is a bounded step rather than the primary orchestrator are worth investigating — is not directly picked up by any of the three RQs. RQ-1 tests LLM-as-orchestrator receiving script output as context versus LLM-as-orchestrator using script as an ensemble slot. The LLM remains the orchestrator in both arms. The simpler and adjacent shape — a script loop that calls an LLM exactly once to interpret or synthesize script-produced data — is not tested and not asked about.

This matters specifically because Cycle 2's empirical evidence points toward the script's contribution as the most unambiguous value in A3. The LLM reviewers' value is real (uncorrelated errors, coverage breadth) but is also overlap-prone and model-selection-sensitive (R1-Hunyuan's chain-of-thought leakage illustrates this). The script's value is not model-selection-sensitive — it runs the same way regardless of which LLM is used and regardless of the reviewer's format discipline. A cycle whose territory is "agent design specifically" should at minimum ask the literature whether the script-as-orchestrator shape has a published evidence base on the cycle's task classes.

The research plan's approved lit-review scope includes "script-driven loops with LLM as subordinate step" as a shape to cover, which partially addresses this. But inclusion in the lit-review is not the same as an explicit RQ, and without a named question, the lit-reviewer has no reason to prioritize this shape's evidence over other shapes in the coverage list. Recommending that the practitioner add a targeted literature coverage instruction or sub-question rather than a full new RQ — the question is answerable from literature, not requiring a spike.

**Specific recommendation:** At lit-review dispatch, add an explicit instruction: "For script-driven loops with LLM as a subordinate step (rather than LLM as the primary orchestrator), seek published evidence on (a) factual-grounding task classes and (b) multi-turn sustained work. If this shape has evidence on either task class, surface it as a comparison point against the LLM-as-orchestrator shape."

### Operating frame transmission

The operating frame ("outcomes over an agentic session; agent shape is means") is named in the research plan and the research log. The discipline note (name the operating frame up-front in every synthesis dispatch) ensures this frame appears at the start of each dispatch. This is correctly implemented and should propagate through the cycle. The reviewer notes that the frame's function in each dispatch is to prevent performance-axis bias and prevent agent-shape anchoring — dispatches that receive the frame should be checked at audit time to confirm that non-LLM-orchestrator shapes are not systematically under-weighted in synthesis outputs.

---

## Confidence and Limits of This Review

**What this review can assess:** Whether the question phrasings have embedded conclusions, whether the set covers the territory opened by the constraint-removal exchange, whether inherited frames are treated as prior art, and whether there are incongruities in the research context the questions fail to surface. These assessments are grounded in the corpus documents as read.

**What this review cannot assess:**

- Whether RQ-1's isolation test will actually isolate A3's load-bearing component (that requires running the spike). The design looks sound; whether the fixture will reveal a clean equivalence or difference depends on empirical results.
- Whether the four-priorities frame will produce a detectable divergent recommendation under Cycle 3's configurations (RQ-2's answer is empirical). The falsification design looks genuine; whether the scoring resolution is adequate to detect divergence is itself a cycle-output question.
- Whether the tau-bench reliability ceiling translates to llm-orc's deployment shape (RQ-3's answer is empirical). The extrapolation is well-motivated but not pre-answerable from the literature alone.
- Whether the script-as-orchestrator shape has a meaningful evidence base in the literature (the recommended lit-review instruction is the right mechanism to find this out, not a pre-assessment by this reviewer).

**Reviewer framing caution:** This review is conducted from within the same corpus the cycle is built on. The framing categories used to assess the questions (meltdown vs. premature stop; LLM-as-orchestrator vs. script-as-orchestrator) are drawn from essay 003 and the feed-forward signals. A reviewer with no prior exposure to this corpus might surface different incongruities. In particular, the "script-as-orchestrator" incongruity identified under Criterion 4 is visible to this reviewer because essay 003 made the script slot's value salient — a reviewer not primed by that finding might not have surfaced it, and a reviewer primed differently might have surfaced incongruities this review does not name. The review should be read as corpus-relative, not as an exhaustive accounting of possible design gaps.

**On the constraint-removal response:** The practitioner's response is brief ("Right makes sense — I am not wedded to any predisposition for the agent. It's all about outcomes over an agentic session."). The brevity does not make it performative — it is direct and the agent's interpretation records the substantive implications clearly. But a longer or more structured constraint-removal response might have surfaced specific alternative agent shapes the practitioner had in mind (or explicitly ruled out), which would have sharpened both the question set and this review. The current response opens the solution space broadly; whether the question set successfully inhabits that open space is the question the Criterion 4 incongruity findings address.

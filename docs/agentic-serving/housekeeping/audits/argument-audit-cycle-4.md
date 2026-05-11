# Argument Audit Report

**Audited document:** `docs/agentic-serving/essays/005-layer-conditional-composition.md`
**Source material:**
- `docs/agentic-serving/essays/research-logs/research-log.md`
- `docs/agentic-serving/essays/research-logs/005a-lit-review-long-horizon-reliability.md`
- `docs/agentic-serving/essays/research-logs/005b-lit-review-composition-shapes-per-layer.md`
- `docs/agentic-serving/essays/research-logs/005c-lit-review-long-horizon-reliability-infrastructure.md`
- `docs/agentic-serving/essays/research-logs/005d-spike-research-loop-dogfood.md`
**Date:** 2026-05-04

---

## Section 1: Argument Audit

### Summary

- **Argument chains mapped:** 14 (covering the eight focus areas named in the dispatch, plus six supporting sub-chains)
- **Issues found:** 13 (2 P1, 5 P2, 6 P3)

---

### P1 — Must Fix

**P1-1**
- **Location:** "The Composite Framing: Layer-Conditional Composition" section; also Conclusion paragraph 3
- **Claim:** "Pre-specifiable routing as primary design axis within each layer" is supported by "Wave 3.A's spike validation that the closed five-tool surface suffices for autonomous routing."
- **Evidence gap:** The spike validated that the closed five-tool surface *sufficed for a single-iteration research-loop dispatch on one fixture with one clearly-identifiable ensemble in the library.* The spike explicitly records that "routing-among-multiple-research-ensembles is not tested by this spike" and that the library had only one obviously-research-shaped free-tier ensemble at the time of the trial. The spike's autonomous-routing observation (Trial 1) eliminates routing ambiguity as a confound. Using this as literature-comparable evidence that "pre-specifiable routing exceeds LLM-judgment routing" conflates an architectural-path validation (the dispatch chain works) with a routing-design comparison (pre-specifiable beats LLM-judgment). The literature claim (CAAF, LLMCompiler, MetaAgent, Gu) genuinely supports the pre-specifiable-routing argument; the spike does not add independent support for the *comparison* — it only confirms the path operates at all. The framing "Wave 3.A's spike validation" in the composite-framing section presents the spike as comparable evidence to the literature findings, which overstates its evidential weight on the routing-design question specifically.
- **Recommendation:** Replace "and Wave 3.A's spike validation that the closed five-tool surface suffices for autonomous routing" with language that is accurate about what the spike established: "and Wave 3.A's empirical confirmation that the dispatch path operates end-to-end at the cheap-cloud tier (the pre-specifiable routing literature recommendation is independently supported by CAAF, LLMCompiler, MetaAgent, and Gu — the spike establishes the path works, not that pre-specifiable beats alternative designs)."

**P1-2**
- **Location:** "Seven ADR Candidates" section, ADR candidate #5; also "Implications for the Architecture" section on ADR-004
- **Claim:** The essay states that ADR-004 (Result summarization mandatory) "commits to the Harness's unskippable interposition, which the cycle's evidence suggests should become conditional rather than mandatory." The amendment is described as "amendment territory rather than pure elaboration."
- **Evidence gap:** The cycle's empirical evidence for this claim is one trial (Trial 2 of N=3) on a single fixture using a small ensemble whose synthesizer output was approximately 600 characters — "below the threshold where context rot is a concern, yet it was summarized anyway" (per the spike log). The spike log's own assessment is that this is "Spike A's mechanism, now empirically observed at the cheap-cloud tier," but Spike A's evidence was a two-stage cascade-collapse on a multi-agent documentation review, not a single-iteration output under the three-agent `qwen3:0.6b` ensemble used here. The essay treats Trial 2's specificity-loss as confirming that the Harness "re-instantiates Cycle 2 Spike A's specificity-loss mechanism" — but the mechanism in Spike A operated through cascade summarization at the reviewer-to-aggregator stage; in Trial 2, the Harness's compression of a 600-character ensemble output is a different architectural position with different inputs. The essay presents these as the same mechanism, but the evidential bridge is thin: one trial, one fixture, one ensemble, one failure mode that may be artifact of the specific ensemble's short and low-quality output (3 × `qwen3:0.6b`).
- **Recommendation:** Qualify the ADR-004 amendment claim: the Trial 2 finding provides motivation to examine the Harness's behavior on short outputs, but the ADR-amendment direction should be stated as "candidate amendment suggested by a single-trial observation on a short output, warranting a targeted spike before DECIDE-phase amendment of ADR-004's mandatory commitment." The essay currently positions the observation as sufficient to warrant "amendment territory" for an accepted ADR; one trial on a 600-character output under small-model agents is below that evidentiary threshold.

---

### P2 — Should Fix

**P2-1**
- **Location:** Abstract and Conclusion; "The Composite Framing" section
- **Claim:** The four composite-framing elements "stack" into a "coherent design-method posture." The term "stackable" is used implicitly throughout and explicitly in the research log's synthesis: "These four framings are stackable: A with B unified by C and progressing along D."
- **Hidden assumption:** The four elements are presented as mutually reinforcing, but a tension between B (pre-specifiable routing as primary axis) and D (experience-accumulation arc via Plexus-driven retrieval-grounded selection) is not surfaced. Pre-specifiable routing means routing logic is captured at architecture time; retrieval-grounded selection (HERA pattern, Framing D) means the routing logic evolves from accumulated experience. These are not purely additive — Framing D's value is precisely in replacing static pre-specified routing with experience-grounded dynamic routing. The essay presents them as a progression (stock library → retrieval-grounded → Plexus-driven) rather than as a design-axis trade-off, which obscures that B's "pre-specifiable" commitment is most valuable when D's retrieval-grounded routing is unavailable (Plexus-absent mode). When Plexus is enabled, the correct reading of the literature (HERA's 38.69% improvement over static baselines) is that pre-specified routing should be *superseded* by retrieval-grounded routing, not merely supplemented by it. The essay's framing of B+D as a coherent stack conceals this tension.
- **Recommendation:** Surface the B-D tension explicitly: "Pre-specifiable routing (B) is the appropriate primary axis in Plexus-absent mode; retrieval-grounded selection (D) partially supersedes static pre-specified routing when Plexus is enabled. The two framings are not additive — D's Plexus-enabled value is in reducing the design cost of explicit pre-specification, not in augmenting pre-specified routing with additional retrieval. The composite framing should be read as: B holds in current deployment; D is the intended direction that weakens B's centrality as Plexus activates."

**P2-2**
- **Location:** "Long-Horizon Reliability" section, MOP paradox application; Conclusion paragraph 2
- **Claim:** "The MOP-paradox finding from the literature reframes the cycle's local-first commitment as well-calibrated rather than as a compromise: frontier-bare exhibits highest meltdown rates on long-horizon, and cheap-orchestrator + good-architecture may be more reliable than frontier-bare on the North-Star benchmark's task category."
- **Evidence gap:** The Khanal et al. MOP paradox finding (arXiv:2603.29231) studies 10 *open-source* models. The meltdown rates it reports (DeepSeek V3: 19%; MiniMax M2.5: 13%) are for open-source models, not for the frontier proprietary models most likely to be compared against the cycle's cheap-orchestrator deployment. The lit-review (005a) correctly records: "frontier models exhibit the *highest* meltdown rates (DeepSeek V3: 19% at very-long; MiniMax M2.5: 13%), because they pursue ambitious multi-step strategies that generate entropy spikes." The "frontier" in Khanal et al.'s paper refers to frontier-of-the-open-source-cohort, not to proprietary frontier models (GPT-5, Claude Opus 4.5). The essay's application of "frontier models exhibit highest meltdown rates" to justify that "cheap-orchestrator + good-architecture may be more reliable than frontier-bare" implicitly applies the Khanal et al. finding to a proprietary frontier comparison that the study did not make. The inference may still be directionally correct (the meltdown mechanism is not specific to open-source models) but the evidential basis is the open-source-models cohort, and the essay's leap from that cohort to a claim about proprietary frontier reliability on the North-Star benchmark is an unstated inferential step.
- **Recommendation:** Add a scope condition: "The MOP paradox finding is from a 10-open-source-models cohort; the extrapolation to proprietary frontier models' meltdown behavior on the North-Star benchmark is directionally plausible (the mechanism — ambitious strategies spiraling — is not model-family-specific) but not directly evidenced. Direct frontier comparison at the North-Star benchmark scale remains Cycle 5+ territory."

**P2-3**
- **Location:** "Seven ADR Candidates" section, ADR candidate #6; "Implications for the Architecture" section on ADR-002
- **Claim:** ADR candidate #6 (upward L0→L1 read-only signal channel) is "the one identified point where the four-layer architecture may need architectural extension" and "the one architectural-extension candidate from the cycle."
- **Hidden assumption:** The essay presents this as the cycle's only architectural-extension candidate because it is the only one requiring a rule amendment. But the research log's Wave 2.B architectural verdict separately identifies that making calibration a cross-layer primitive "requires a read-only signal channel from L0 to L1" as the *sole* exception to the "operationalizable within existing layers" conclusion. The essay does not examine whether other candidates it names — for example, ADR candidate #2 (Session Registry adopts initializer-then-resume schema) expanding the Session Registry's responsibility significantly — constitute responsibility concentration sufficient to warrant architectural reconsideration rather than elaboration. The distinction between "elaboration" and "new architectural concern" is left entirely to the framing, with no criteria articulated for when elaboration crosses into requiring a separate module or layer. The research log's Wave 2.B is explicit about this: "Whether that concentration [of expanded responsibilities] is acceptable is an architectural design choice the literature does not make for the cycle." The essay inherits the "operationalizable within existing layers" verdict but drops the wave's explicit caveat about responsibility concentration as an unresolved architectural question.
- **Recommendation:** Add the Wave 2.B caveat explicitly: "The architectural verdict — operationalizable within existing layers — assumes that the expanded responsibilities of Session Registry (handoff artifacts + write-gate validation) and Calibration Gate (in-process trajectory calibration + post-hoc promotion) are appropriately concentrated within those modules. Whether responsibility concentration at those modules warrants a separate infrastructure layer or cross-cutting concern module is a design question the literature does not resolve and the DECIDE phase should address."

**P2-4**
- **Location:** "The Behavioral Spike" section; "Open Questions and Scope-of-Claim" section
- **Claim:** The essay states Trial 3's phantom-tool-call confabulation is "a previously-unobserved failure" and "qualitatively different from the failure modes the cycle's prior literature scan covered."
- **Hidden assumption / scope issue:** The essay names this as "novel" but does not acknowledge that the 005a lit-review (FQ4 findings) explicitly documents: "models like Ollama 7B/14B often hallucinate tool calls as plain text after the first interaction" as a well-documented practitioner finding. The Trial 3 failure (cheap-cloud orchestrator emitting prose claiming a tool call) is at a different tier (cheap cloud, not local 7B/14B) and the mechanism is documented as "the model had the tool schema... but instead of emitting a tool call it narrated one as already-completed." The lit-review's practitioner finding about small-model tool-call hallucination is for a different model class; the trial finding at cheap-cloud tier is indeed observationally novel to the cycle's deployment shape. But the essay's "previously-unobserved failure" framing overstates novelty — the practitioner literature had documented the class of failure; the cycle documents it at a specific tier. More importantly, the essay's scope-of-claim section notes that "the failure mode's frequency, fixture-class dependence, and severity at multi-iteration scale are unknown" — but does not note that the cheap-cloud model (MiniMax M25 Free) was tested under an adversarial diagnostic prompt ("Output nothing else. Just the tool call. No prose, no explanation, no narration. Tool call only.") rather than under conditions representative of normal agentic operation. The 1/3 failure rate is confounded by the prompt design of Trial 3.
- **Recommendation:** Qualify the novelty claim: the class of failure is practitioner-documented for small local models; the novel element is its observation at the cheap-cloud tier. Additionally, note that Trial 3's diagnostic prompt is atypical — the 1/3 confabulation observation is under a prompt designed to isolate the failure, not under operational agentic-session conditions. The ADR candidate #7 motivation remains valid but its urgency should not be calibrated from a diagnostic failure rate.

**P2-5**
- **Location:** "Research-Entry Framing and Constraint-Removal" section, Sub-Q6 reformulation discussion; "Open Questions and Scope-of-Claim" section
- **Claim:** The Sub-Q6 reformulation "places the transfer-test ('does the failure mode actually occur in this architecture at these session lengths?') before the intervention-class question."
- **Evidence gap and scope issue:** The essay presents Wave 3.A's Trial 1 as providing "a partial answer" to the transfer-test (autonomous routing fired correctly). But the transfer-test question from the research log is specifically about context-growth degrading ensemble-routing judgment — not about whether autonomous routing fires at all on a single iteration. Trial 1 answers "can the orchestrator dispatch an ensemble in one iteration?" The transfer-test question is "does routing judgment degrade as context accumulates across iterations?" These are different questions. The essay does not make this distinction explicit; a reader following the chain from "transfer-test" to "Trial 1 positive observation" would conclude the transfer-test has a partial answer, when in fact the test as originally specified has received zero evidence — Trial 1 is single-iteration by design. The essay's scope-of-claim section does acknowledge that "multi-iteration scale... are required before autonomous routing can be claimed reliably" but does not explicitly note that the transfer-test itself (the question Sub-Q6 was reformulated around) is unanswered rather than partially answered.
- **Recommendation:** Reframe the Trial 1 result: it answers "can the dispatch path fire autonomously on one iteration" (yes), not "does routing judgment degrade under context growth" (untested). The reformulated Sub-Q6 transfer-test remains entirely open; Trial 1 is a necessary precondition for testing it (the path must work before degradation can be observed) but is not itself a partial transfer-test result.

---

### P3 — Consider

**P3-1**
- **Location:** "Composition Shapes Per Layer" section, Self-MoA discussion
- **Claim:** "The challenge is task-class-conditional. Self-MoA's finding holds where the task requires correctness on well-defined questions; it does not hold where the task requires cross-family coverage of systematically different blind spots, which is the territory of Cycle 2's Spike A3 documentation review and Cycle 3's Spike C cross-file verification."
- **Tightening opportunity:** The essay's task-class boundary (Self-MoA wins on well-defined questions; heterogeneity wins on cross-family-coverage tasks) is the correct resolution from the literature but the characterization of the cycle's task class as "on the heterogeneity side" rests on the cross-file-verification finding from Spike C and the documentation-review finding from Spike A3. The essay should note that the RDD North-Star benchmark includes phases (DISCOVER, PLAY, parts of MODEL, BUILD) whose cognitive surface (conversational, exploratory, continuous-routing) may not be in the heterogeneity-coverage territory. For those phases, Self-MoA's finding may apply and same-family quality may be preferred. The current framing implies the entire cycle's task class is on the heterogeneity side, but the three-cluster decomposition in the same essay establishes that different phases have different task-class profiles.

**P3-2**
- **Location:** "Composition Shapes Per Layer" section, Attention-MoA and orchestrator-quality discussion
- **Claim:** "Cheap must be calibrated against this — if the cheap orchestrator is the aggregator, its quality ceiling caps recursive ensemble depth."
- **Tightening opportunity:** The Attention-MoA finding (12.82 percentage-point gap from aggregator quality, validated on AlpacaEval 2.0 and MT-Bench) is presented as a warning for the cheap-orchestrator tier without noting the benchmark scope: instruction-following quality is the measurement surface, not code-review or cross-file verification. The aggregator-quality dependency may be larger or smaller on the cycle's task class. The essay's "must be calibrated" framing imports the 12.82pp figure's implication without carrying its scope condition. The benchmark limitation is mentioned in the essay for the RecursiveMAS finding but not for Attention-MoA's aggregator dependency.

**P3-3**
- **Location:** "Long-Horizon Reliability Infrastructure" section, plain-filesystem finding
- **Claim:** "Plain filesystem outperforms naive vector-store libraries at 74 percent on MemoryAgentBench."
- **Tightening opportunity:** The 74% figure represents plain filesystem performance on MemoryAgentBench. The comparison class is "specialized vector-store memory libraries" (naive deployment). The essay uses this correctly as a "don't invest in retrieval infrastructure before simple structured-artifact approaches saturate" argument. However, the sentence structure ("plain filesystem outperforms naive vector-store libraries at 74 percent") reads as if 74% is the comparative gap or the plain filesystem's score against naive libraries, rather than the plain filesystem's absolute score. Clarify: "plain filesystem scores 74 percent on MemoryAgentBench, outperforming naive vector-store libraries which score lower — suggesting structured-artifact approaches should be saturated before investing in retrieval infrastructure."

**P3-4**
- **Location:** "Seven ADR Candidates" section, ADR candidate #1
- **Claim:** "The 250,000 API calls per day waste documented from pre-circuit-breaker Layer 4 failures validates the cheapest-first ordering."
- **Tightening opportunity:** The 250,000-API-calls/day waste is from Claude Code's own Layer 4 failures in pre-circuit-breaker era — it is Claude Code's observed failure rate, not a general finding about compaction pipelines. The essay uses it correctly as motivation, but the implicit inference ("therefore cheapest-first ordering is validated for llm-orc's Conversation Compaction module") is an architectural analog, not a direct measurement. The specificity of the number (250,000/day from a specific system) may suggest greater generalizability than it has.

**P3-5**
- **Location:** "The Composite Framing: Layer-Conditional Composition" section, the Sub-Q6 resolution paragraph
- **Claim:** "Spike A3's class (a) wrapper does not generalize to ensemble routing; the right intervention class is class (c) decomposition (pre-specifiable routing, schema-level enforcement) and class (b) governed retrieval (HERA-style); calibration-gated cross-layer composition is the cycle's most novel territory."
- **Tightening opportunity:** This sentence runs three logically distinct claims together. (1) Spike A3 doesn't generalize; (2) the right class is (c) and (b); (3) calibration-gated cross-layer is novel. The transition from (2) to (3) is abrupt — calibration-gated cross-layer composition is mentioned as "novel territory" but its relationship to the Sub-Q6 question (intervention class for continuous routing under context growth) is not made explicit. The calibration gate is at L1; the continuous routing failure mode operates at the orchestrator (L2); the connection between them (L0 signals flowing up to L1 to gate L2 dispatch) is the content of ADR candidate #6, but the Sub-Q6 resolution paragraph doesn't name that connection. A reader following the Sub-Q6 chain would benefit from: "calibration-gated cross-layer composition is novel territory specifically because it would use L0 ensemble quality signals to gate L1 dispatch decisions — a class of intervention for which no published paper has implemented the full cross-layer composition."

**P3-6**
- **Location:** "Open Questions and Scope-of-Claim" section, the 7B-14B reliability boundary paragraph
- **Claim:** "Whether the boundary shifts as open-weight models mature (Qwen3-Coder-Next is already at 70.6–71.3 percent on SWE-Bench Verified) is observation-territory rather than design-territory."
- **Tightening opportunity:** Qwen3-Coder-Next's 70.6–71.3% SWE-Bench Verified score is for cloud inference (API-served), not for local deployment of the same model. The 7B-14B reliability boundary the essay references is for locally-deployed small models. The parenthetical about Qwen3-Coder-Next slides from the local-model boundary to a cloud-inference score for a much larger model family without marking the category shift. The 7B-14B boundary is relevant to locally-deployed ensemble members; Qwen3-Coder-Next's SWE-Bench Verified performance is relevant to cloud-tier orchestration and is not evidence that the 7B-14B boundary has shifted.

---

## Section 2: Framing Audit

The framing audit makes the negative space of content selection visible. The primary document chose a framing — this section examines what that choice excluded.

### Question 1: What alternative framings did the evidence support?

**Alternative framing A: The script-models-as-primary-layer framing**

The source material supports a reading in which the script-models layer is not one of several task-class-dependent load-bearing layers but the *most consistently supported* load-bearing layer across the cycle's tested task classes. Evidence in the source material:
- Spike C's 3-of-3 result directly attributes success to deterministic file access (script layer), not to ensemble orchestration. The essay names this correctly.
- Spike A3's pattern — the strongest evidence for ensemble composition — has a deterministic script member as the anchor; the lit-review (005b, section 2.2) establishes "deterministic tools as consensus-resistant anchors" and the mechanism is attributed to the script member's categorical epistemic property.
- Wave 3.A's successful trials: the closed five-tool surface sufficed; the one substantive positive finding (Trial 1 autonomous routing) dispatched an ensemble whose value came from structure (the three-agent ensemble architecture), not from probabilistic complementarity. The specificity-loss finding (Trial 2) implicates the *absence* of deterministic provenance as the failure mode.
- The 005b literature (Wisdom and Delusion of LLM Ensembles, Compiled AI hybrid mode, LLMCompiler) converges on deterministic tool outputs as the primary reliability mechanism in mixed ensembles.

Under this alternative framing, the essay's central claim would be: the cycle's value is located primarily in the script-models layer's deterministic guarantees, with ensembles serving as compositional wrappers that *preserve* those guarantees against LLM-consensus pressure rather than as independent sources of value. The orchestrator's job is routing to the right deterministic substrate.

What would a reader need to believe for this framing to be right? That Spike A3's heterogeneity-uncorrelated-errors finding is downstream of the script anchor (not independent), that the documentation-review setting is a task class where the absence of a deterministic anchor explains the multi-trial variance Spike B observed, and that the cycle's evidence for ensemble-as-mechanism (rather than envelope-for-determinism) is thinner than presented.

**Alternative framing B: The "inadequate empirical base for architectural design claims" framing**

The source material contains explicit admissions that three of the cycle's four most consequential design-method claims are literature-grounded rather than cycle-empirically-grounded:
- Long-horizon reliability infrastructure (externalized state, initializer-then-resume, calibration-gated composition, per-role configurability): Wave 3.A is single-iteration; none of these were tested.
- Autonomous routing at multi-iteration scale: N=1 positive observation.
- Frontier comparison: not tested.

Under this alternative framing, the essay's central claim would be: Cycle 4 produced (a) a well-evidenced single-iteration dispatch-path validation, (b) two concrete failure-mode observations (specificity-loss, phantom tool-call), and (c) a literature-derived design-method framework whose applicability to the cycle's deployment shape awaits empirical testing. Seven ADR candidates are proposed from literature synthesis, not from cycle empirical evidence (except #5 and #7 which have one trial each). The "design-method posture" framing overstates the cycle's empirical contribution by presenting literature synthesis and cycle empirics at the same epistemic level.

What would a reader need to believe for this framing to be right? That the distinction between literature-grounded and cycle-empirically-grounded design recommendations matters for architectural decision-making, and that DECIDE-phase ADRs based on literature-only synthesis have meaningfully different risk profiles than ADRs based on cycle empirics.

**Alternative framing C: The "pre-specifiable routing is a bet against the capability trajectory" framing**

The source material (005b Framing B) and the DyTopo/HERA findings support that experience-grounded generative routing outperforms static pre-specified routing substantially (+38.69% for HERA). The cycle's pre-specifiable-routing-as-primary-axis framing (Framing B of the four-element composite) represents a deliberate conservative choice given the cheap-orchestrator tier and the Plexus-deferral state. Under the alternative framing, the essay's central claim would be: Cycle 4's most important finding is that the pre-specified routing architecture is a provisional match for the current capability floor, and the architecture's design pressure should be toward accelerating Plexus activation and generative topology (HERA-style) rather than investing in more elaborate pre-specified routing policies. The cycle's design methods are a transition path, not a destination.

What would a reader need to believe for this framing to be right? That the HERA/DyTopo evidence transfers to the cycle's deployment shape (not established — both are frontier-tier validated), that Plexus activation is near-term feasible, and that the design cost of pre-specified routing (capturing routing logic at architecture time) is a meaningful investment that would be wasted when generative routing becomes reliable at the cheap-cloud tier.

---

### Question 2: What truths were available but not featured?

**Absent finding A: The capability ceiling for small local models in multi-step tool calling**

The 005a lit-review (FQ4) documents: "7B-14B local models often hallucinate tool calls as plain text after the first interaction due to context pressure combined with insufficient fine-tuning depth at the tool-calling level." The failure mechanism matches Trial 3 of the spike (phantom tool-call confabulation) *at the cheap-cloud tier*. The lit-review explicitly identifies this as a failure mode for local small models. Wave 3.A's Trial 3 extends this observation to the cheap-cloud tier. The essay surfaces Trial 3 as a novel finding and ADR candidate #7 target, which is correct — but it does not note that the *local ensemble members* (qwen3:0.6b in the spike) are well below the 7B threshold the lit-review identifies for this failure mode. If the local ensemble members used in the spike (0.6B) are below the reliability boundary documented in the literature, then the spike's "closed five-tool surface sufficed" finding may be conditional on the ensemble members not being asked to do multi-step tool calling — which they were not (they have bounded single-call roles). This conditional is present in the essay's operationalization of local-first but is not connected back to the spike's validation claim.

Where it appears in the source material: 005a FQ4 section ("Unreliable: multi-step tool calling after context accumulation beyond ~2-3 turns"), and the mechanism description ("context pressure causes structured-output format maintenance to compete with semantic reasoning capacity in small models").

Why it was likely excluded: the spike's success (Trials 1 and 2) with local ensemble members was not a failure-mode observation, so the connection between the lit-review's 7B-14B boundary and the 0.6B models used in the spike was not surfaced. But the spike's "validation" only validates the dispatch path for the specific role the 0.6B models played (bounded single-call analysis roles), not for multi-step tool calling. This connection would clarify *why* the spike worked with such small models: they were used in a mode that the lit-review identifies as within their reliable capability class.

Would its inclusion change the argument? Yes — it would strengthen the operationalization claim ("decompose complex decisions into many bounded stateless per-member tasks") by connecting it to the specific capability boundary the literature documents, and would prevent the spike's validation from being over-read as evidence that 0.6B models can handle more demanding roles in the architecture.

**Absent finding B: The four-priorities frame measured-divergence test inheritance from Cycle 3**

The dispatch instructions for this audit name this explicitly as a Cycle 3 open question that carries forward unresolved: whether the four-priorities frame (from essay 003) survives a measured-divergence test. The essay does not mention this carry-forward. The research log does not address it either. The cycle's behavioral spike (Wave 3.A) had an opportunity to surface measured-divergence configurations (by comparing the cheap-cloud tier's output against a frontier-tier alternative) but explicitly deferred frontier comparison to Cycle 5+. The four-priorities frame is inherited as prior art in the essay's "starting state" section without noting that one of its open validations (measured-divergence test) remains unaddressed.

Where it appears in the source material: the dispatch instructions reference this directly; it would appear in prior cycle artifacts (cycle-3 susceptibility snapshot).

Why it was likely excluded: the Cycle 3 susceptibility snapshot's recommendation was recorded as context for Cycle 4 entry, but the research log's Step 1.1 six sub-questions did not include measured-divergence testing of the four-priorities frame as a distinct research question. The cycle's scope was already broad; this specific validation was not prioritized.

Would its inclusion change the argument? Partially — it would flag that the composite framing inherits a prior design commitment (four-priorities frame) whose validation gap the cycle did not close. The essay's confidence in the composite framing's coherence is slightly overstated if the four-priorities frame underlying it has an outstanding measured-divergence question.

**Absent finding C: The write-gate validation security concern for append-only state**

The 005c Wave 2.B review (Focus Area 1, mnemonic sovereignty section) documents: "memory poisoning attack surface... write-gate validation is recommended in the literature but not operationalized in any reviewed system." ADR candidate #2 (Session Registry adopts initializer-then-resume schema) includes "write-gate validation for memory-poisoning protection" in its list of responsibilities. However, the essay does not note that write-gate validation is explicitly flagged in the source material as "Cycle 4-or-later territory" and as theoretically described but not deployable off-the-shelf. The essay presents write-gate validation as part of the ADR candidate's "design decision" surface without flagging that it is current open research, not an implementable pattern.

Where it appears in the source material: 005c, Focus Area 1, "append-only JSONL" section: "write-gate validation and post-deletion verification as shared mitigations — but neither is currently operationalized in published systems."

Why it was likely excluded: the ADR candidate description lists it among several responsibilities; the "not operationalized" caveat from the lit-review did not survive into the essay's treatment of that candidate.

Would its inclusion change the argument? It would add a P2 qualification to ADR candidate #2: the write-gate validation component is aspirational rather than adoptable from a published reference implementation, unlike the other three components of the candidate (feature-list schema, append-only progress log, init-sh-style bootstrap) which all have canonical Anthropic reference implementations.

**Absent finding D: The conflict-between-deterministic-tool-output-and-LLM-consensus failure mode**

Both 005b (Focus Area 2, section 2.2) and 005a identify explicitly: "No paper studies what happens when the deterministic tool's output conflicts with LLM ensemble consensus." The essay names this open question once (in the "Composition Shapes Per Layer" section: "it surfaces an open question the literature does not address: what happens when deterministic tool output conflicts with LLM consensus?") but does not surface it as a spike-worthy concern or as scope-of-claim discipline for ADR candidates that rely on the script-member-alongside-LLM pattern (#5's Result Summarizer Harness treatment is adjacent). Given that Spike C's mechanism (deterministic tool output as consensus-resistant anchor) is the most empirically grounded finding in the cycle, and the specific failure mode of that pattern (tool-consensus conflict) is unstudied, the essay should note this as a Cycle 5 spike candidate more prominently.

---

### Question 3: What would change if the dominant framing were inverted?

**Inversion 1: "Layer-conditional cross-layer composition" → "Script-models layer as primary, with ensembles and orchestrator as accidental scaffolding"**

Under this inversion: Spike C's 3-of-3 result is the cycle's clean signal, and it is entirely explained by the script-models layer. Spike A3's documentation-review value is explained by the script anchor in that ensemble (the deterministic profile-checker), not by the heterogeneous LLM members' cross-family coverage. The wave's evidence for ensemble composition as a distinct mechanism (DeliberationBench's selection-vs-deliberation gap, heterogeneity-uncorrelated-errors) is genuine but the cycle has not empirically demonstrated that the *ensemble members' heterogeneity* (rather than the *script anchor's categorical guarantee*) is the load-bearing element in the cycle's task class.

What becomes weaker: the case for investing in seven ADR candidates involving ensemble composition, calibration-gated composition, and recursive ensemble depth. If the script layer is primary, the investment priority should be in compilation of deterministic scripts (LLMCompiler, Agentic Compilation) and in the script-alongside-LLM pattern.

What becomes stronger: the closed-five-tool-surface finding (ADR-003), the Compiled AI hybrid mode recommendation, and the Wisdom and Delusion CrossHair-anchoring finding. Wave 3.A's positive result is consistent with this inversion (the ensemble used in the spike had a simple structure; its value was in the dispatch architecture, not in probabilistic complementarity).

What the essay would need to address: why Spike A3's heterogeneity-uncorrelated-errors finding (two LLM families, not two script+LLM combinations) is evidence for *ensemble composition as mechanism* rather than for *ensemble as envelope preserving per-member bounded outputs*.

**Inversion 2: "Operationalizable within existing layers" → "The cycle's findings motivate substantial architectural reorganization"**

Under this inversion: the responsibility-concentration concern the essay inherits from Wave 2.B but then drops is the signal, not the conclusion. The Session Registry absorbing initializer-then-resume schema, write-gate validation, and append-only persistence is a substantial scope expansion for a module currently described as "identifies and continues a multi-request Session." The Calibration Gate absorbing trajectory-level calibration alongside post-hoc promotion tracking is a significant responsibility increase. The Orchestrator Runtime absorbing a five-layer compaction pipeline is a more complex module than the system design currently implies.

What becomes weaker: the "architecture is in better structural shape than the cycle's gate finding initially implied" conclusion in the essay's Conclusion.

What becomes stronger: the Wave 1.A framing-shift candidate (long-horizon reliability as cross-cutting infrastructure that may warrant a separate layer or module) which Wave 2.B's verdict rejected but with the explicit caveat that "responsibility concentration... is an architectural design choice the literature does not make for the cycle."

What the essay would need to address: whether the DECIDE phase should include an ADR on whether the expanded responsibilities of Session Registry and Calibration Gate warrant extraction into dedicated infrastructure modules, or whether "elaboration within existing modules" is genuinely the right architectural verdict.

**Inversion 3: "Seven ADR candidates" → "Fewer than seven genuinely independent design decisions"**

Under this inversion, ADR candidates #1 (Conversation Compaction) and #2 (Session Registry initializer) are adoptions of published external patterns (Claude Code's five-layer pipeline, Anthropic's initializer schema) rather than cycle-novel decisions. Treating them as ADR candidates positions the DECIDE phase to re-debate decisions the literature has already converged on; the DECIDE phase's real work is in candidates #3–#7 where the cycle contributes novel design territory.

What becomes weaker: the claim that the cycle "surfaces seven concrete design decisions as candidate ADRs" as a description of cycle contribution.

What becomes stronger: the DECIDE phase's resource allocation — if candidates #1 and #2 are literature adoptions, they warrant ADR write-up but minimal deliberation; the deliberation budget should concentrate on #3 (Calibration Gate extension), #5 (Harness reconsideration), #6 (upward signal channel), and #7 (phantom-call guard).

What the essay would need to address: the distinction between "adopt a published pattern as a formal ADR" and "decide a novel architectural question" — both produce ADRs, but the deliberation discipline differs.

---

### Framing Issues

**P1 — Consequential omissions**

**FP1-1**
- **Location:** Abstract, "The Behavioral Spike" section, "Open Questions and Scope-of-Claim" section
- **Claim:** Wave 3.A validated the architecture's invoke_ensemble dispatch path; seven ADR candidates are surfaced; long-horizon reliability infrastructure is literature-grounded.
- **Consequential omission:** The essay's scope-of-claim discipline names that long-horizon infrastructure claims are literature-grounded, but it does not name that the phantom tool-call confabulation finding (ADR candidate #7) is a *load-bearing* finding for the essay's own "closed five-tool surface sufficed" claim. Trial 1's autonomous routing success and Trial 2's correct dispatch occurred without any confabulation. Trial 3's confabulation occurred under a diagnostic prompt specifically designed to isolate the call. The essay treats these as separate findings: one positive (dispatch works), one negative (confabulation exists). But the confabulation finding raises a question the essay does not answer: how confident is the "closed five-tool surface sufficed" claim, if the same cheap-cloud model that correctly dispatched in Trials 1–2 confabulated in Trial 3? The essay does not assess whether the conditions that prevented confabulation in Trials 1–2 (task-structure-implies-the-dispatch vs. explicit-tool-call-demand) are reliably present in operational agentic sessions. If they are not, then the closed-tool-surface adequacy claim depends on out-of-band conditions the cycle has not specified.
- **Recommendation:** Add to the scope-of-claim section: "The closed-five-tool-surface finding depends on the orchestrator reliably emitting tool-call structures when dispatching; Trial 3's confabulation observation suggests this reliability is conditional on the prompt's task-structure implying the dispatch rather than demanding the tool call directly. Whether operational agentic sessions provide that conditioning consistently is an open scope condition for the 'sufficiency' claim."

**P2 — Underrepresented alternatives**

**FP2-1**
- **Location:** Throughout; most visible in "The Starting State" and "The Composite Framing" sections
- **Claim:** The essay carries forward essay 003's "composition of components whose error distributions are different enough" as the unifying frame without qualification.
- **Underrepresented alternative:** The DeliberationBench finding (6× selection-vs-deliberation gap) and the Self-MoA finding (same-family sampling outperforms cross-family mixing on instruction-following) together challenge whether the "different error distributions" frame is the primary mechanism or whether *output quality of the selection mechanism* is the primary mechanism. If DeliberationBench's best-single-selection wins by 82.5% vs. 13.8%, the diversity of the pool matters less than the quality of the selection from the pool. The essay treats this as a topology recommendation (no deliberation, only parallel concatenation) rather than as a challenge to the unifying frame. A reader following the framing would conclude that heterogeneous components with different error distributions is the mechanism; a reader following the selection finding would conclude that the mechanism is "don't dilute the best output." These point to different design choices at the margin.
- **Recommendation:** Note the tension explicitly: the DeliberationBench and Self-MoA findings suggest that for tasks where the best single response is identifiable, *selection quality* may matter more than *error-distribution heterogeneity*. The unifying frame holds for tasks where no single best response is identifiable in advance (the cycle's cross-file-verification and documentation-review task classes, where the script anchor provides evidence the LLM members cannot independently generate). For tasks where a best response is identifiable, a simpler "select the best" pattern may outperform heterogeneous composition. This boundary should be named in the frame.

**FP2-2**
- **Location:** "Long-Horizon Reliability" section, MOP paradox; Conclusion
- **Claim:** "Cheap-orchestrator + good-architecture may be more reliable than frontier-bare on long-horizon work" — presented as a positive reframing of the local-first commitment.
- **Underrepresented alternative:** The MOP paradox's most interesting implication for the cycle may not be "frontier melts down, so cheap is safer" but "frontier melts down because it pursues ambitious strategies — and the cheap orchestrator's strategy is bounded by its capability floor." If the cheap orchestrator succeeds on long-horizon work, it may do so because it fails earlier and less catastrophically (the Khanal et al. "weaker models fail uniformly" observation), not because it's genuinely more reliable. The essay's framing ("cheap-orchestrator + good-architecture") implies that the architecture provides reliability compensation that frontier-bare lacks. This may be true for the specific architectural interventions (externalized state, class-c decomposition) — but it conflates the architecture's value with the cheap-orchestrator's intrinsic behavior under meltdown. A deliberate architecture on top of a weaker orchestrator that fails early and uniformly (not meltdown) could look like reliability when it is structurally-bounded failure.
- **Recommendation:** Acknowledge the alternative reading: the MOP paradox may indicate that the cheap orchestrator's reliability advantage on long-horizon is partly due to capability-bounded failure (failing before meltdown threshold rather than surviving to the meltdown regime). The architecture's value is then in extending the useful operating range of a cheaper tier, not in achieving higher reliability than frontier-bare at the same operating range.

**P3 — Minor framing choices**

**FP3-1**
- **Location:** "Seven ADR Candidates" section, framing of the seven candidates' provenance
- **Claim:** "The cycle's research surfaces seven concrete design decisions as candidate ADRs. Six are elaborations or extensions of existing architectural commitments; one amends the layering rule."
- **Minor framing choice:** The essay presents all seven as originating from the cycle's research, but ADR candidates #1 (Claude Code's five-layer pattern) and #2 (Anthropic initializer schema) are direct adoptions of external published patterns with canonical reference implementations. Framing them as "the cycle surfacing them" is accurate (the cycle identified them as applicable), but the description could make clearer that these two candidates are adoption decisions with minimal uncertainty — the design is already decided by the external patterns; the cycle's work is identifying fit and confirming adoption. This is different from candidates #3–#7 where the cycle is proposing novel architectural elements. The distinction matters for DECIDE-phase resource allocation.

**FP3-2**
- **Location:** "Research-Entry Framing and Constraint-Removal" section, ADR-082 reference
- **Minor framing choice:** The essay refers to "ADR-082 (of the RDD plugin methodology — distinct from the agentic-serving corpus's ADRs 001–011)" without naming ADR-082's content. A reader unfamiliar with the RDD methodology's ADR numbering would not know what ADR-082 establishes. The parenthetical distinguishes the namespace but does not communicate the constraint it represents.

**FP3-3**
- **Location:** "The Composite Framing" section, summary paragraph
- **Minor framing choice:** "Together they constitute the cycle's design-method posture for the cheap-orchestrator + ensemble pattern." The term "posture" is used here and in the composite framing section's introduction. It is an apt term but it sits in tension with the essay's use elsewhere of "design-method" and "design-formulation deliverable" — three different framings for the same output. Consistent terminology across the essay would strengthen the claim.

## Literature Review: Composition Shapes Per Layer

**Date:** 2026-05-04
**Method:** Systematic literature search (web search + primary source fetch across arXiv, OpenReview, practitioner sources)
**Cycle:** 4 (agentic-serving scoped corpus)
**Wave:** 2.A — Composition shapes per layer (parallel wave: 2.B covers long-horizon reliability infrastructure)
**Prior corpus scanned to avoid duplication:** Cycle 2 lit-reviews (003a, 003b), Cycle 3 lit-review (004a), Wave 1.A (005a)

---

### Sources Reviewed

| # | Author(s) | Title | Year | Venue | Relevance |
|---|-----------|-------|------|-------|-----------|
| 1 | Yang et al. (Xiyuan Yang et al., Stanford/UIUC) | Recursive Multi-Agent Systems (RecursiveMAS) | 2026 | arXiv:2604.25917 | FA1: recursive ensembles-of-ensembles, latent-space recursive composition |
| 2 | Wen et al. (Jianyu Wen et al.) | Attention-MoA: Enhancing Mixture-of-Agents via Inter-Agent Semantic Attention and Deep Residual Synthesis | 2026 | arXiv:2601.16596 | FA1: inter-agent attention at each layer, residual synthesis, collapse prevention |
| 3 | Pecerskis, Smirnovs | Mixture-of-Models: Unifying Heterogeneous Agents via N-Way Self-Evaluating Deliberation | 2026 | arXiv:2601.16863 | FA1: runtime dynamic expertise broker, quadratic voting, trust-score aggregation |
| 4 | Li et al. (Wenzhe Li, Princeton) | Rethinking Mixture-of-Agents: Is Mixing Different Large Language Models Beneficial? (Self-MoA) | 2025 | arXiv:2502.00674 (OpenReview ICLR 2025) | FA1: self-MoA vs. heterogeneous MoA, quality-diversity trade-off |
| 5 | (Authors not extracted) | DeliberationBench: When Do More Voices Hurt? A Controlled Study of Multi-LLM Deliberation Protocols | 2025 | arXiv:2601.08835 | FA1: deliberation vs. best-single baseline, 6.0x performance gap |
| 6 | Lu et al. (Yuxing Lu et al.) | DyTopo: Dynamic Topology Routing for Multi-Agent Reasoning via Semantic Matching | 2026 | arXiv:2602.06039 | FA1, FA3: semantic key-query matching, round-by-round topology, generative construction |
| 7 | Jiang et al. (Eric Hanchen Jiang et al.) | Dynamic Generation of Multi-LLM Agents Communication Topologies with Graph Diffusion Models (GTD) | 2025 | arXiv:2510.07799 | FA1: generative topology construction via graph diffusion |
| 8 | Li, Sha and Ramakrishnan, Naren | Experience as a Compass: Multi-agent RAG with Evolving Orchestration and Agent Prompts (HERA) | 2026 | arXiv:2604.00901 | FA1, FA3: experience-guided dynamic orchestration, reward-guided topology sampling |
| 9 | Zhang (Yaolun Zhang et al.) | MetaAgent: Automatically Constructing Multi-Agent Systems Based on Finite State Machines | 2025 | arXiv:2507.22606 (ICML 2025) | FA2, FA3: FSM-based auto-design, state traceback routing |
| 10 | Kim et al. (LLMCompiler authors, Stanford) | An LLM Compiler for Parallel Function Calling | 2024 | arXiv:2312.04511 (ICML 2024) | FA2: parallel DAG-planned function execution, 3.7x speedup, 6.7x cost reduction |
| 11 | Batra et al. | Compiled AI: Deterministic Code Generation for LLM-Based Workflow Automation | 2026 | arXiv:2604.05150 | FA2: what compiles (stable repetitive workflows), compilation output (JSON FSM artifacts), task class amenability |
| 12 | (Authors not extracted) | Agentic Compilation: Mitigating the LLM Rerun Crisis for Minimized-Inference-Cost Web Automation | 2026 | arXiv:2604.09718 | FA2: compile-and-execute architecture, one-shot JSON blueprint, zero per-execution inference cost |
| 13 | Gu, Xingrui (UC Berkeley) | Task-Aware Delegation Cues for LLM Agents | 2026 | arXiv:2603.11011 | FA3: capability-profile-based delegation, coordination-risk cues, routing with task taxonomy |
| 14 | Chuang et al. | Confident or Seek Stronger: Exploring Uncertainty-Based On-device LLM Routing | 2025 | arXiv:2502.04428 | FA3: SLM uncertainty routing to stronger LLMs, 1500+ settings benchmark |
| 15 | Huang et al. (ReDAct authors) | ReDAct: Uncertainty-Aware Deferral for LLM Agents | 2026 | arXiv:2604.07036 | FA3: two-model deferral (15% of decisions to large model matches full large-model quality), ALFWorld, MiniGrid |
| 16 | Ma et al. (R3AG authors) | R3AG: Retriever Routing for Retrieval-Augmented Generation | 2026 | arXiv:2604.22849 | FA3: query-specific retriever routing, contrastive learning, outperforms static routing |
| 17 | Zhang, Shao (MetaAgent) | From Agent Loops to Structured Graphs: A Scheduler-Theoretic Framework for LLM Agent Execution | 2026 | arXiv:2604.11378 | FA2, FA3: DAG-based scheduling, task decoupled planning |
| 18 | Pydantic team / community | PydanticAI and structured output validation for LLM agents | 2024–2026 | pydantic.dev, practitioner ecosystem | FA1: Pydantic-style typed contracts, schema-enforced ensemble interfaces |
| 19 | OpenAPI Initiative | Arazzo Specification for multi-step API orchestration | 2024–2026 | openapis.org | FA1: OpenAPI-style typed composition for multi-step agent workflows |
| 20 | (MCP ecosystem) | Model Context Protocol structured outputs requirement | 2025 | MCP spec November 2025 | FA1: typed-contract enforcement as ecosystem standard |

### Sources inherited from prior cycles (not duplicated; framed against below)

Cycle 2 Loop 4 covered: MARG (Drozdov et al. 2024), Ding et al. 2024 (diversity vs. popularity trap), Sun et al. 2025 (heterogeneous agent scaling), Jiang et al. April 2026 (information-theoretic single-agent argument), Yao et al. 2025 (meta-judge panel discussion). Cycle 3 004a covered: Compiled AI (first pass), Routine (Xu et al.), OneFlow, Lee et al. (skill distillation). Wave 1.A covered: Cleanlab trust-score on tau2-bench (50% failure reduction), CAAF state-locking, constrained decoding alignment tax. Those sources are referenced below where new literature refines or qualifies them.

---

### Synthesis

---

#### Focus Area 1 — Ensemble Composition Shapes

**Framing note:** Cycle 2 Loop 4 established the vocabulary for ensemble shapes (MARG-concatenation, diversity-uncorrelated-errors, refinement-loop limits, Jiang 2026 single-agent information-theoretic argument, Yao 2025 panel-discussion diversity-destruction). This review covers the five new territories named in the dispatch instructions. Where a finding directly extends or contradicts Cycle 2 Loop 4 territory, it is flagged explicitly.

---

##### 1.1 Recursive ensembles-of-ensembles

**What RecursiveMAS establishes.** Yang et al. (arXiv:2604.25917, April 2026) extend the Mixture-of-Agents paradigm from sequential layer-by-layer text passing into latent-space recursive computation. The framework connects heterogeneous agents through a lightweight RecursiveLink module that enables two capabilities: in-distribution latent thought generation and cross-agent latent state transfer. An inner-outer loop learning algorithm handles co-optimization via shared gradient-based credit assignment across recursion rounds. Across nine benchmarks (mathematics, science, medicine, search, code generation), RecursiveMAS delivers +8.3% average accuracy improvement, 1.2–2.4× inference speedup, and 34.6–75.6% token reduction versus non-recursive baselines.

The key architectural distinction from standard MoA (Wang et al. 2024, covered in Cycle 2 Loop 1): standard MoA passes *text-serialized* responses from each layer to the next, requiring each agent to re-parse its peers' outputs from natural language. RecursiveMAS passes *latent representations* through the RecursiveLink module, enabling tighter cross-agent integration without text round-tripping. This directly addresses one mechanism of the panel-discussion diversity-destruction finding: agents in standard MoA that see peers' verbose text outputs tend to anchor on those outputs rather than evaluate them independently.

**Scope condition (critical for llm-orc):** RecursiveMAS's latent-space transfer requires gradient-accessible model weights — the co-optimization step requires backpropagation through the RecursiveLink module. This is a fine-tuning dependency. For llm-orc's black-box inference setting (where models are invoked via API and weights are not accessible), the latent-state transfer mechanism is unavailable. The +8.3% accuracy and token-reduction numbers cannot be reproduced without the training component. The *architectural insight* (looping agents recursively rather than stacking layers linearly) transfers; the specific mechanism does not.

**What Attention-MoA establishes at the deployment level.** Wen et al. (arXiv:2601.16596, January 2026) address the same diversity-destruction problem without requiring weight access. Instead of latent-state transfer, Attention-MoA uses natural-language inter-agent critique: at each layer, each agent generates corrective instructions for its peers' responses by comparing those responses against its own, then refines its own output accordingly before aggregation. A residual module prevents depth degradation by maintaining an accumulating history of all preceding layers' outputs and applying adaptive early stopping when a convergence signal is detected.

The benchmark result is notable: standard MoA achieves 88.56% length-controlled win rate on AlpacaEval 2.0; Attention-MoA achieves 91.15% — a +2.59 percentage point improvement at the same number of agents. The MT-Bench score rises from 9.13 to 9.32. Critically, Attention-MoA shows monotonically increasing performance through layer 5, whereas standard MoA degrades after layer 3. This is direct evidence that the collapse-at-depth problem Cycle 2 Loop 4 identified (the panel-discussion diversity-destruction mechanism) can be addressed at the inference-time level without fine-tuning.

**Documented failure modes.** Attention-MoA has three documented scope conditions the cycle must hold explicitly:
- Aggregation agent dependency: the performance gap between best and worst aggregators is 12.82 percentage points. The system's reliability depends heavily on the aggregator's capability, which in llm-orc's setting means the cloud orchestrator quality is the binding constraint on recursive composition.
- Token cost without optimization: approximately 119,000 tokens per query at default configuration. This is far above the cycle's budget discipline. The adaptive early stopping mechanism reduces this, but baseline cost is high.
- Scope of benchmarks: AlpacaEval 2.0 and MT-Bench measure open-ended instruction-following quality, not coding or cross-file verification. Generalization to the cycle's task class is not demonstrated.

**What Self-MoA establishes (and where it challenges the cycle's heterogeneity prior).** Li et al. (arXiv:2502.00674, February 2025) find that Self-MoA — running multiple inference passes on the *single best-performing model* rather than mixing heterogeneous models — achieves 6.6% improvement over standard heterogeneous MoA on AlpacaEval 2.0 and 3.8% average improvement across MMLU, CRUX, and MATH. The mechanism: MoA performance is sensitive to the average quality of models in the ensemble; mixing heterogeneous models dilutes quality with weaker members' outputs. Sampling multiple times from the strongest model preserves quality while generating diversity through sampling temperature.

This directly complicates the Cycle 2 Loop 4 heterogeneity prior (Sun et al. 2025 showing heterogeneous agents have uncorrelated errors; Ding et al. 2024 showing diversity-based selection recovers 95% of theoretical ceiling). The contradiction between Self-MoA and the heterogeneity findings is genuine and requires careful framing. The resolution is task-class dependent:

- For *open-ended instruction-following quality* (AlpacaEval, MMLU), same-family quality consistency outperforms cross-family diversity. The task requires correctness on well-defined questions where the strongest model already knows the answer, and sampling noise generates sufficient diversity.
- For *uncorrelated-error coverage on documentation/code verification* (the cycle's Sub-Q3 task class), the heterogeneity prior may hold because the relevant failures are not sampling-noise variations but systematic blind spots tied to model training — which same-family re-sampling cannot overcome. Hunyuan and Kimi finding different bugs (Cycle 3 Spike A3 empirical finding) is the cycle's specific evidence that the cross-family coverage matters on this task class.
- Self-MoA itself acknowledges the boundary: "incorporating diverse models does not always lead to superior performance." The "always" is the hedge; the finding establishes that same-family sampling is often competitive, not that cross-family diversity is never valuable.

**What DeliberationBench establishes.** The controlled study (arXiv:2601.08835, December 2025) finds that the "best-single" baseline — selecting the highest-quality response from a pool without deliberation — achieves an 82.5% ± 3.3% win rate against three tested deliberation protocols, which achieve only 13.8% ± 2.6% at 1.5–2.5× higher computational cost. This is a 6.0× performance gap, statistically significant (p < 0.01), directionally consistent with Yao et al. 2025's panel-discussion diversity-destruction finding (Cycle 2 Loop 4) but substantially stronger in magnitude.

The critical scope condition for the cycle: DeliberationBench evaluates a council of five models (GPT-4o-mini, Claude-3.5-Haiku, Gemini-2.0-Flash-001, Llama-3.1-8B-Instruct, Mistral-Nemo) on 270 questions skewed toward medium and hard difficulty (77.4%). The "deliberation protocols" are unspecified in the abstract. The benchmark distribution is intentionally difficult — the conditions most favorable to deliberation — yet deliberation still loses badly. The implication for the cycle: the decision to include deliberation (sequential round-based debate among members) in an ensemble composition is not supported by this evidence. MARG-style independent parallel roles (Cycle 2 Loop 4) with concatenation rather than deliberation is the design that avoids this failure mode.

**Open question the literature cannot settle:** The cycle's Sub-Q4 names "recursive ensembles-of-ensembles" as the Cycle 4 seed — specifically "recursive composition aggregating across multiple A3-style ensembles for different review framings without re-introducing the collapse problem at the meta level." The literature provides:
- RecursiveMAS for recursive composition with fine-tuning access (not deployable in llm-orc's setting)
- Attention-MoA for inference-time recursive depth with collapse prevention (deployable but expensive and validated only on instruction-following benchmarks)
- DeliberationBench and the Yao 2025 panel-discussion finding that sequential deliberation destroys diversity (constraints on how meta-aggregation should be structured)

What the literature does not provide: a published architecture for recursive composition of *independently scoped A3-style ensembles* (where each ensemble addresses a different framing of the same problem) that has been tested on code-review or documentation-verification task classes. The MARG-concatenation principle (preserve independent reviewer voices) and the Attention-MoA residual principle (maintain historical layer outputs, detect convergence, stop early) are the closest published supports for the design essay 003 motivated, but neither tests the specific recursive-over-independently-scoped-ensembles pattern.

---

##### 1.2 Typed contracts for ensemble interfaces

**What the ecosystem establishes.** The strongest signal here comes from the practitioner ecosystem rather than peer-reviewed research. PydanticAI (launched late 2024, 16,000+ GitHub stars by early 2026) is the dominant framework for typed schema enforcement in LLM agent interactions. The core mechanism: convert a Pydantic model into a JSON schema that the LLM must conform to; if the model deviates, PydanticAI re-prompts with the validation error and requests a conforming retry. The MCP specification (November 2025) now requires tool servers to return results conforming to an output schema — effectively mandating typed contracts at the serving layer for MCP-based agent systems.

The OpenAPI Arazzo Specification (2024–2026) extends OpenAPI to describe multi-step API workflows, providing a compositional contract framework for chaining API calls — directly analogous to typed composition validation for ensemble sequences. A workflow in Arazzo specifies input schema, output schema, and success criteria for each step, with the workflow itself being a schema-validated composition.

**What the peer-reviewed literature does not establish.** No peer-reviewed paper was found that studies typed-composition enforcement for ensemble interfaces as a distinct research object — that is, comparing a "typed-contract-enforced" ensemble composition against a "prompt-described" ensemble composition on reliability, specificity, or correctness metrics. The practitioner ecosystem treats typed contracts as engineering best practice; the academic literature has not yet studied the performance delta from enforcing them.

The closest adjacent academic finding: the constrained decoding alignment tax (Zhou, arXiv:2604.06066, Wave 1.A) shows that enforcing structural constraints on *model outputs* imposes a reliability cost at small model sizes (Qwen3-8B drops 50% → 38% accuracy under grammar constraints). This is a scope condition on typed contracts: enforcing schema conformance on the *orchestrator's* decisions (e.g., which ensemble to invoke, what parameters to pass) via constrained decoding would impose this alignment tax. The alternative — enforcing schemas at post-generation validation and retry (PydanticAI's approach) — avoids the alignment tax but adds latency for each retry cycle. For llm-orc's orchestrator surface, schema validation at dispatch time (validating the tool call arguments against the ensemble's declared input schema) is the deployment-compatible approach, and the Composition Validator (ADR reference) already implements a version of this at compose-time.

**What the literature implies for the cycle's Composition Validator.** The Composition Validator validates compositions against the existing reference graph at compose-time. The literature's trajectory (MCP mandating output schemas, Arazzo for workflow-level contracts, PydanticAI for runtime enforcement) suggests that compose-time-only validation is the *minimum*; the ecosystem is moving toward runtime enforcement at every invocation boundary. llm-orc's architecture is ahead of the peer-reviewed curve here — the academic literature has not studied this; the practitioner ecosystem is standardizing it.

**Tension with generative composition.** Typed contracts constrain the space of valid compositions. Generative ensemble construction (Focus Area 1.4 below) requires the orchestrator to construct novel ensemble topologies at runtime. If the typed-contract surface is rigid, generative construction is limited to compositions that fit within the declared schema surface. This is a genuine tension: reliability-through-typing and flexibility-through-generation pull in opposite directions. The literature does not resolve it; HERA (arXiv:2604.00901, below) navigates it by evolving the orchestration policy within a schema-valid topology space rather than constructing arbitrary topologies.

---

##### 1.3 Generative ensemble construction (orchestrator builds topologies per problem)

**GTD: topology generation as diffusion.** Jiang et al. (arXiv:2510.07799, October 2025) introduce Guided Topology Diffusion (GTD), which formulates multi-agent topology construction as a conditional discrete graph diffusion process. Starting from an empty graph, GTD iteratively adds edges guided by both task context and agent team characteristics. The result is a task-specific communication topology constructed generatively rather than selected from a fixed library.

**DyTopo: per-round semantic matching.** Lu et al. (arXiv:2602.06039, February 2026) take a lighter-weight approach: rather than generating the topology from scratch, DyTopo has each agent emit natural-language "need" (query) and "offer" (key) descriptors each reasoning round. These are semantically embedded and matched to construct a sparse directed communication graph for that round alone. The graph changes round-by-round as the reasoning task shifts from exploration to verification. Average performance improvement: +6.2 over the strongest static topology baseline across code generation and mathematical reasoning benchmarks. The interpretable coordination trace (the sequence of graphs across rounds) allows qualitative inspection of how communication pathways evolve.

**HERA: experience-guided topology evolution.** Li and Ramakrishnan (arXiv:2604.00901, April 2026) propose HERA for multi-agent RAG contexts. At the global level, HERA optimizes query-specific agent topologies through reward-guided sampling and experience accumulation — routing decisions that worked on similar past queries are prioritized. At the local level, role-specific agent prompts are refined via credit assignment. The result is a system that shifts from exploratory broad agent networks on novel queries to compact high-utility networks on query types with accumulated experience. Average improvement: 38.69% over recent baselines across six knowledge-intensive benchmarks. The key mechanism for the cycle: experience accumulation as retrieval-grounded routing (past reward signals are retrieved and used to inform current routing decisions), which is the intersection of generative construction and retrieval-grounded selection.

**Scope conditions that apply to the cycle's local-first commitment.** All three generative topology approaches (GTD, DyTopo, HERA) are validated at frontier or near-frontier model tiers. GTD uses diffusion models that require training. DyTopo's semantic matching requires embedding models. HERA's reward-guided sampling requires reward signals from downstream task outcomes — a credit assignment loop that in llm-orc's setting would require the orchestrator to evaluate ensemble outputs and update routing policy. None of these mechanisms is trivially deployable in llm-orc's current closed five-tool surface without architectural extension.

The cost asymmetry is directly relevant: generative topology construction at every orchestration step imposes overhead proportional to the sophistication of the construction mechanism. For the cycle's cheap-orchestrator constraint, the orchestrator's topology decisions must themselves be cheap. DyTopo's per-round descriptor emission is the lightest-weight mechanism found — it adds one inference step per agent per round for descriptor generation, which is bounded overhead. GTD's diffusion process and HERA's reward-guided sampling are heavier.

**What the literature implies for the no-library baseline.** The cycle's Sub-Q4 includes "generative ensemble construction (orchestrator builds ensembles per problem with no library)" as a candidate shape. The evidence from the generative topology literature suggests this shape is viable at frontier tier with significant engineering overhead, but the cost-of-topology-construction must be subtracted from the quality gain. For the cycle's economic framing, the relevant question is whether the quality gain from per-problem topology construction exceeds the orchestrator's overhead cost of constructing that topology. None of the papers above directly measures this trade-off at the cheap-orchestrator tier.

---

##### 1.4 Calibration-gated ensemble composition

**The calibration-gating principle from the literature.** Wave 1.A surfaced Cleanlab's finding (trust-score fallback cuts tau2-bench failure by 50%) as the closest analog to ADR-007's Calibration Gate. The new literature from this wave adds two more directly relevant mechanisms.

**ReDAct: uncertainty-aware deferral at the model selection boundary.** Huang et al. (arXiv:2604.07036, 2026) demonstrate a two-model architecture where a small cheap model is the default and a large reliable model receives deferred decisions when the small model's uncertainty exceeds a calibrated threshold. The empirical result: deferring only 15% of decisions to the large model matches the quality of using the large model exclusively. The task classes tested are sequential decision-making in text-based embodied environments (ALFWorld, MiniGrid) — not code review or documentation verification. The scope transfer requires care, but the mechanism (calibrated uncertainty threshold that gates escalation) is directly analogous to ADR-007's Calibration Gate gating ensemble trust transitions.

**Chuang et al. (arXiv:2502.04428, February 2025)** provide the most systematic study of uncertainty-based SLM routing across 1,500+ settings. The key finding: uncertainty-correctness alignment in different uncertainty quantification (UQ) methods significantly impacts routing performance, and the choice of UQ method matters more than the threshold level. The finding that calibration data effectively bootstraps routing performance without new downstream data is directly applicable to the cycle's setting — the Calibration Gate's positive-signal accumulation is a form of calibration data accumulation.

**Confident or Seek Stronger's scope condition.** The paper's framing is specifically on-device SLM to cloud LLM routing — exactly the CAP-9 hybrid deployment shape. The finding that different SLMs require different UQ methods (uncertainty distributions depend on both the SLM and the chosen UQ method) is a warning for the cycle's Calibration Gate: a single calibration threshold calibrated on one model may not transfer when the ensemble's member models change. This is the calibration-portability scope condition.

**What the literature does not establish.** No paper directly studies calibration-gating at the *ensemble composition* boundary (transitioning a newly composed ensemble to "trusted" status after positive signals, which is ADR-007's mechanism). The ReDAct and Chuang et al. papers study calibration-gating at the *model selection* boundary (routing a single query to a more capable model). The transfer to composition-level trust gating is conceptually sound but empirically untested in the published literature. Wave 1.A's Cleanlab finding (trust-score at the *output* level) is a different application surface again.

---

##### 1.5 Stock library + retrieval-grounded selection

**R3AG: retriever routing via contrastive learning.** Ma et al. (arXiv:2604.22849, 2026) demonstrate that retriever routing — selecting among specialized retrievers based on query-specific capability assessment — consistently outperforms both the best individual retriever and state-of-the-art static routing methods on knowledge-intensive tasks. The routing model decomposes retriever capability into two dimensions (retrieval quality and generation utility) and learns routing via a contrastive objective using document assessment and answer correctness signals.

The structural analogy to ensemble selection is direct: replace "retriever" with "ensemble" and "retrieval quality" with "task-specific recall." The finding that *query-specific dynamic routing* outperforms *static routing* (selecting the same retriever regardless of query) is the retrieval-domain validation of the same principle the MASS framework (Cycle 2 Loop 1, Zhou et al. 2025) demonstrated for multi-agent topologies.

**HERA's experience accumulation as retrieval-grounded routing.** Li and Ramakrishnan's HERA framework (arXiv:2604.00901, above) implements retrieval-grounded ensemble selection at the orchestration level: past routing experiences that generated positive reward are stored and retrieved when similar queries arrive, influencing the current routing decision. This is the closest published implementation of "stock library + retrieval-grounded selection" where the retrieval operates over past routing outcomes rather than over static capability descriptions. The 38.69% average improvement over baselines on knowledge-intensive tasks is the strongest performance signal for this pattern found in this wave.

**Tension with the cycle's local-first constraint.** Experience accumulation requires persistent state across sessions — exactly what the Plexus optional integration in llm-orc is designed to support. In Plexus-absent mode, the orchestrator cannot accumulate routing experience, and retrieval-grounded selection degrades to static selection (effectively the stock library without the retrieval-grounded part). This means the retrieval-grounded selection shape is architecturally contingent on Plexus enablement in llm-orc's implementation.

---

#### Focus Area 2 — Script-Models Composition Shapes

---

##### 2.1 How scripts compose with each other

**LLMCompiler: parallel DAG-planned function execution.** Kim et al. (arXiv:2312.04511, ICML 2024) provide the foundational published evidence for deterministic parallel script composition. LLMCompiler uses three components: a Function Calling Planner that generates a DAG of tasks with inter-dependencies, a Task Fetching Unit that dispatches tasks to the executor as dependencies are satisfied, and an Executor that runs tasks in parallel. The performance results: up to 3.7× latency speedup, up to 6.7× cost reduction, and up to ~9% accuracy improvement compared to ReAct. The key insight is that *function call planning* (which calls are independent, which are dependent) is separable from *execution* — the LLM handles the dependency graph construction; deterministic code handles the execution in parallel.

For the cycle's architecture: `invoke_ensemble` calls are analogous to function calls in LLMCompiler's framing. The cycle's closed five-tool surface already restricts the orchestrator to a bounded set of operations, which is architecturally aligned with LLMCompiler's constrained planner. The question the literature does not settle: whether the orchestrator's DAG-planning capability at cheap-cloud-tier matches LLMCompiler's planning quality (which was validated at frontier tier).

**Agentic Compilation: the compile-and-execute architecture.** The follow-on work (arXiv:2604.09718, 2026) extends the compilation principle to web automation, demonstrating that a one-shot LLM invocation that processes a semantically pruned page representation and emits a JSON workflow blueprint reduces per-workflow inference cost from approximately $150 (continuous agent, 500 iterations) to under $0.10 (compiled blueprint). The execution engine is purely deterministic after compilation. The scope condition: "stable, repetitive, structure-amenable workflows" — the compilation is worth its cost only when the same workflow will be executed many times, amortizing the compilation cost over many runs.

For the cycle's cross-file verification task class (the Sub-Q3 load-bearing task for script-models), the verification logic is relatively stable across different codebases: check that referenced model profiles exist, validate link targets, confirm section presence. This is amenable to compilation — a compiled verification blueprint that runs deterministically across different input codebases. The Agentic Compilation paper's empirical finding that zero-shot compilation achieves 80–94% success rates (with minimal human-in-the-loop patching for near-100%) is encouraging for this task class.

**PlanCompiler: deterministic compilation architecture.** (arXiv:2604.13092, referenced in practitioner search) extends the compilation paradigm to structured multi-step LLM pipelines, where execution order, node wiring, and runnable code are all derived deterministically after the plan is accepted. This is the architecture closest to essay 003's "recursive composition aggregating across multiple A3-style ensembles" applied to the script layer: compile the coordination logic into a deterministic plan, then execute the plan without LLM involvement in the execution loop.

**What the literature establishes for script-to-script composition.** The published evidence converges on fan-out / fan-in via DAG scheduling (LLMCompiler) as the primary pattern for parallel script composition, with the compilation paradigm (Compiled AI, Agentic Compilation) as the pattern for stable repetitive workflows. Sequential pipelines (shell-pipe-style) are treated as degenerate cases of DAG scheduling where each node has exactly one dependency.

**What the literature does not establish.** No published paper studies multi-script orchestration at the level of *coordinating several independently developed deterministic scripts* (one checking links, one checking model profiles, one checking code blocks) as a *single composed workflow* with typed contracts at the script boundaries. The practitioner literature treats this as conventional software engineering (function composition); the academic agentic literature has not studied it as a distinct research object. This is the gap the cycle's script-layer empirical work occupies.

---

##### 2.2 How scripts compose with LLM ensemble members

**Compiled AI's hybrid mode as the anchoring pattern.** Batra et al. (arXiv:2604.05150, Cycle 3 004a first pass; returned in this search for the task-class amenability question) distinguish three modes within the compilation framework: compiled extraction (for structured, predictable inputs), runtime LLM (for noisy, open-ended content), and hybrid (compiled extraction + confidence-based LLM fallback for mixed-content systems). The hybrid mode is the direct architectural analog of Spike A3's pattern: deterministic script handles structured verification; LLM ensemble handles semantic analysis; the two are composed.

The task-class amenability finding from Compiled AI is directly relevant: the compilation phase succeeds on *structure-amenable* tasks; the LLM phase handles *semantically open-ended* tasks. For the cycle's cross-file verification task class, the separation is clean — link checking, model-profile existence, section presence are structure-amenable; identifying undefined profiles across indirect references, surfacing onboarding-friction issues, and reasoning about architectural implications are semantically open-ended. This matches the A3 split between the deterministic script slot and the heterogeneous LLM reviewer ensemble.

**Wisdom and Delusion of LLM Ensembles for Code Generation (arXiv:2510.21513, October 2025)** provides the code-domain evidence for hybrid composition. The paper studies LLM ensemble dynamics on code generation and repair by combining CodeBLEU and CrossHair tools — deterministic tools embedded in the ensemble pipeline to provide static analysis signals that LLM ensemble members use as anchors. The "delusion" finding: code-generation ensembles fall into a "popularity trap" when their diversity is insufficient (Cycle 2 Loop 4, Ding et al. 2024 parallel finding). The mitigation: incorporating CrossHair's counterexample-based feedback as a deterministic anchor that breaks the popularity trap by introducing tool-generated evidence that cannot be disputed by LLM consensus.

For the cycle, this finding is direct support for the script-member-alongside-LLM pattern (Spike A3's design). CrossHair's counterexamples in code-generation ensembles play exactly the same role as the cycle's script-agent's verified facts in documentation-review ensembles: a deterministic, consensus-resistant signal that anchors LLM analysis rather than allowing ensemble consensus to override evidence.

**What the literature establishes.** The script-as-ensemble-member pattern (where a deterministic tool runs alongside LLM members in the same ensemble call) is empirically supported by:
- Compiled AI's hybrid mode (task-class separation principle)
- Wisdom and Delusion's CrossHair embedding (deterministic counterexample anchoring in code-generation ensembles)
- Cycle 3's Spike A3 (3/3 cross-file-verification successes with script slot; 0/2 without)

The mechanism common to all three: deterministic outputs provide a category of evidence (verified with zero hallucination probability) that is categorically resistant to LLM-consensus pressure. This is the strongest form of the script-member's value — not speed or cost, but epistemic category.

**What the literature does not establish.** No paper studies what happens when the deterministic tool's output *conflicts* with the LLM ensemble's consensus — whether the ensemble members appropriately weight the tool evidence over consensus, or whether they rationalize away tool findings that conflict with LLM agreement. This is an open reliability question for the Cycle 4 spike program.

---

##### 2.3 Typed contracts for script-tool interfaces

**The practitioner standard.** Typed contracts for script-tool interfaces are a practitioner engineering standard (Pydantic model → JSON schema → validated tool call), not an active research object. The MCP ecosystem has standardized tool interface schemas (November 2025 specification requirement). The tooling ecosystem (PydanticAI, LangGraph strict mode, OpenAI Structured Outputs) enforces typed contracts at tool call boundaries. The peer-reviewed literature does not study this as a distinct performance-impacting design choice; it is treated as a correctness baseline.

The distinction between LLM tool-calling and script tool-calling matters for the cycle: LLM tool-calling validation catches format errors (the model outputs a malformed tool call); script tool-calling validation catches semantic errors (the script receives incorrect parameters that cause it to silently return wrong results). The typed-contract discipline for scripts is stricter because script failures are deterministic — a wrong input produces a consistent wrong output with no uncertainty signal. LLM tool-calling failures are stochastic — they may manifest differently on different runs.

**Arazzo Specification as workflow-level contract.** The OpenAPI Arazzo specification provides a principled framework for multi-step API workflow contracts — specifying which outputs from step N become inputs for step N+1 and enforcing type compatibility at the composition boundary. This is directly applicable to the cycle's script-model compositions where the script's structured output (JSON report with link counts, section presence flags, undefined profiles) becomes input for the LLM ensemble member's analysis. Arazzo-style composition contracts would enforce that the script's output schema matches the ensemble member's expected input schema at compose-time — a structural analog to the cycle's Composition Validator.

---

##### 2.4 Compiled AI deeper read: what compiles, what compilation outputs

**Task classes amenable to compilation.** Compiled AI (arXiv:2604.05150) identifies the amenable class as "stable, repetitive, structure-amenable workflows" — tasks where the execution logic is the same across problem instances, varying only in the input data, not the processing logic. The paper's empirical scope: data extraction, form filling, and fingerprinting in web automation contexts. The principle extends to any task where the orchestration structure (which tools to call, in what order, with what typed arguments) can be determined from the problem specification once and reused across problem instances.

For the cycle's script-layer: link checking and model-profile validation are highly amenable (the logic is identical across codebases; only the input URLs and profile names differ). Semantic documentation analysis is not amenable (the analysis logic depends on the content being analyzed). This is the natural compilation boundary within the cycle's architecture.

**What compilation outputs.** The compilation artifacts in the literature are:
- JSON workflow blueprints (Agentic Compilation, 2026) — a structured representation of the execution graph, suitable for deterministic replay
- Compiled Deterministic JSON FSM (referenced in practitioner community) — a finite state machine that handles state transitions without LLM involvement at execution time
- DAG of function calls with dependency annotations (LLMCompiler, ICML 2024) — a parallel execution plan where dependent tasks wait for their predecessors
- Python code artifacts (Compiled AI, Batra et al. 2026) — executable code generated at compile time, run deterministically thereafter

For llm-orc, the existing script-as-tool pattern (scripts callable via `invoke_ensemble`) uses Python scripts as the compilation artifact — which is the highest-fidelity output in the taxonomy. The literature supports this as the appropriate output form for the cycle's task class.

**Agentic Compilation's cost argument.** For 500 executions of a 5-step workflow, a continuous inference agent costs approximately $150; agentic compilation reduces this to under $0.10 — a 1,500× reduction. The $0.10 represents the one-time compilation cost amortized over 500 runs. For the cycle's use case, the economics improve with reuse frequency: if the same cross-file verification pipeline runs on many codebases, the compilation cost is amortized across all runs. If it runs once per project, the compilation overhead is proportionally larger relative to benefit.

**MetaAgent: FSM-based auto-design with state traceback.** Zhang et al. (arXiv:2507.22606, ICML 2025) provide the FSM-as-multi-agent-topology pattern: MetaAgent automatically designs an FSM for a given task, where each FSM state corresponds to a specialized agent role, and transitions encode the workflow logic. The state traceback capability — returning to a prior FSM state when a downstream agent finds an error — is the FSM-native analog of error recovery in the cycle's deterministic harness. MetaAgent's auto-design approach positions it as a form of generative compilation: the LLM designs the FSM topology; deterministic FSM execution handles the workflow without further LLM planning.

---

#### Focus Area 3 — Orchestrator Routing Composition Shapes

---

##### 3.1 Supervisor / hierarchical routing at small-model tier

**What the literature says at non-frontier tier.** The cycle's Sub-Q3 locates orchestrator routing as load-bearing for hybrid deployment economics (Cycle 1 CAP-9). The frontier-tier evidence (Anthropic's Claude Opus 4 as supervisor with Claude Sonnet 4 subagents — Cycle 2 Loop 1) does not directly address whether a cheap-cloud-orchestrator at the level of MiniMax M2.5 Free can serve as an effective supervisor.

The Task-Aware Delegation Cues paper (Gu, arXiv:2603.11011, March 2026) provides the most directly applicable framework for small-model-capable supervision. The paper induces an interpretable task taxonomy via semantic clustering of Chatbot Arena pairwise comparisons, then derives two signals for delegation: Capability Profiles (task-conditioned win-rate maps showing which model wins on which task class) and Coordination-Risk Cues (task-conditioned disagreement / tie-rate priors showing which task classes produce high model disagreement). The delegation protocol supports adaptive routing (primary vs. primary+auditor) based on these signals. The practical implication: a supervisor that routes based on pre-computed capability profiles and coordination-risk signals does not need to make capability judgments at runtime — the judgment is embedded in the routing table, which is a cheaper operation than dynamic capability assessment.

This is the closest published analog to the cycle's question about whether cheap-cloud-orchestrator routing is reliable: if the routing decision is grounded in a pre-computed capability profile rather than in real-time LLM judgment about task-model fit, the routing reliability is decoupled from the orchestrator's real-time reasoning capability.

**The swarm literature's negative finding for small-model supervisors.** Rahman and Schranz (2025, Cycle 2 Loop 1) established that LLM-based swarm coordination imposes ~36,000× latency penalty versus classical algorithms. The cost-vs-reliability tradeoff for small-model supervisors follows a similar logic: a small supervisor model that can reliably route simple tasks may fail to route complex tasks (the threshold at which the supervisor's capability becomes the bottleneck). No paper directly tests this trade-off at the cheap-cloud-orchestrator tier for the cycle's task classes.

---

##### 3.2 Routing under retrieval grounding

**The R3AG, HERA, and DyTopo pattern.** The evidence from FA1 (above) establishes retrieval-grounded routing at the ensemble selection level. The same principle applies at the orchestrator routing level: routing decisions grounded in retrieved historical performance data outperform routing via real-time LLM judgment on knowledge-intensive tasks.

**What the literature says about routing-via-retrieval vs. routing-via-judgment.** HERA's experience accumulation mechanism (reward-guided retrieval) demonstrates 38.69% improvement over static routing baselines. R3AG's retriever routing demonstrates consistent improvement over best-individual-retriever selection. Both validate the core principle: for tasks where historical performance data is available, retrieval-grounded routing outperforms judgment-based routing.

The scope condition for the cycle: retrieval-grounded routing requires accumulated performance history. For the cycle's Calibration Gate (ADR-007), calibration history accumulates as the gate observes positive/negative signals on ensemble outputs. Once sufficient history is accumulated, the gate transitions the ensemble to "trusted" status. This is architecturally compatible with the retrieval-grounded routing principle — the gate's trust history is a compact representation of past routing experience.

---

##### 3.3 Hybrid-deployment routing

**What the literature says about when the hybrid split works.** The Wave 1.A review established the empirical baseline: local model reliability on multi-step tool calling degrades after 2–3 turns of context accumulation (practitioner literature) and requires 32B+ parameters for frontier-competitive agentic coding performance (Hall, failingfast.io, December 2025). The hybrid split (CAP-9: cloud orchestrator routes; local ensembles execute) separates the context-accumulation burden from the execution burden.

**The routing-reliability boundary.** The tau2-bench finding (Yao / Sierra, June 2025, Cycle 3 004a) puts pass@1 at 56–74% for single-control agent tasks and 34% for dual-control tasks at frontier tier. For the cycle's hybrid split, the relevant question is: what routing decisions are at the boundary of the cheap-cloud-orchestrator's capability? Decisions that require tracking multi-turn session state (which ensemble was invoked three turns ago, what was its output) are in the reliability-risk zone. The Calibration Gate (ADR-007) addresses this by maintaining positive-signal history that the orchestrator can query via `query_knowledge` rather than relying on context-accumulated state tracking.

**When the hybrid split breaks.** The literature identifies two conditions where the hybrid split degrades:
1. When routing decisions require cross-turn state tracking that exceeds the orchestrator's reliable working memory at its operating tier. This maps to the "continuously recurring routing decision at high frequency" failure surface identified in Wave 1.A.
2. When local ensemble members are asked to handle tasks that exceed their reliable task class (multi-file reasoning, complex agentic state tracking past 2–3 turns). This maps to the capability-floor constraint documented in Wave 1.A.

---

##### 3.4 Autonomy-gated dispatch

**The published taxonomy.** The Knight First Amendment Institute framework (referenced in search results, 2025) defines five levels of escalating agent autonomy characterized by user roles: operator, collaborator, consultant, approver, and observer. This is the closest published analog to ADR-008's per-session Autonomy Levels, though the two frameworks use different taxonomic axes (Knight: user role; ADR-008: action scope).

**Bounded autonomy as the convergent industry position.** The 2025–2026 practitioner literature converges on bounded autonomy — allowing agents to act autonomously where outcomes are predictable, requiring human involvement when risk or uncertainty increases — as the practical deployment pattern. The enforcement mechanism in the practitioner literature is consistent with ADR-008's schema-level approach: clear action limits encoded at the tool surface, not at the prompt layer. The OpenDev write-tools-not-available-in-planner-interface approach (Bui, arXiv:2603.05344, Wave 1.A) is the published implementation of this mechanism.

**ReDAct's calibration-uncertainty-to-autonomy mapping.** ReDAct (arXiv:2604.07036, above) provides the tightest published connection between uncertainty quantification and autonomy-gated dispatch: when the small model's uncertainty exceeds a calibrated threshold, the action decision is deferred (escalated to a larger model or to human review). The 15% deferral rate for quality-matching the large model exclusively is the empirical target for the autonomy gate threshold. Whether this 15% figure transfers to the cycle's orchestrator decisions is not established; it is a reference point for threshold calibration, not a direct specification.

---

##### 3.5 Decision-tree / state-machine routing

**MetaAgent and FSM-controlled routing.** Zhang et al. (arXiv:2507.22606, ICML 2025) demonstrate FSM-based multi-agent system construction where the FSM controls routing decisions (state transitions) deterministically. State Traceback allows the system to return to a prior state when a downstream agent reports failure — implementing error recovery within the deterministic routing logic. The auto-design approach generates FSMs from task descriptions, making FSM routing accessible without hand-coding every state transition.

**LLMCompiler's DAG as routing structure.** Kim et al. (arXiv:2312.04511, ICML 2024) treat the DAG of function calls as a pre-compiled routing structure: the LLM generates the routing plan once; the executor follows it deterministically. This is the DAG-routing analog of FSM-routing, appropriate when the routing structure is a partial order (some tasks depend on others) rather than a state machine (routing depends on outcome).

**CAAF's state-locking as a routing discipline.** Zhang (arXiv:2604.17025, Wave 1.A) establishes that marking settled routing decisions as `read_only: true` prevents the LLM from re-litigating them at later turns. This is a class (a) intervention for the routing surface: enforcing monotonic non-regression on routing decisions through deterministic state-locking rather than through repeated LLM judgment.

**What the literature establishes for deterministic routing.** The convergent finding across CAAF, LLMCompiler, MetaAgent, and the practitioner literature: pre-compiled routing (FSM, DAG, or state-locked policy) is more reliable than LLM-judgment routing at every decision frequency and model tier tested. The cost is design overhead: the routing logic must be specifiable in advance. For the cycle's task classes where routing logic can be pre-specified (invoke cross-file verification ensemble for PRs touching multiple model profile references; invoke documentation-review ensemble for README changes), pre-compiled routing is directly applicable.

---

### Key Findings

**Focus Area 1 — Ensemble Composition Shapes:**

- **Recursive latent-space composition achieves +8.3% accuracy and 75.6% token reduction but requires weight access for co-optimization.** RecursiveMAS (Yang et al., arXiv:2604.25917, 2026) is the strongest recursive composition result found, but its latent-state transfer mechanism requires gradient-accessible model weights. llm-orc's black-box inference setting makes this mechanism unavailable; the architectural insight transfers but the specific mechanism does not.

- **Inference-time recursive depth with collapse prevention is deployable without fine-tuning.** Attention-MoA (Wen et al., arXiv:2601.16596, 2026) achieves 91.15% LC win rate on AlpacaEval 2.0 vs. standard MoA's 88.56% using inter-agent critique-and-refine within each layer plus residual history accumulation. Performance is monotonically increasing through layer 5 where standard MoA collapses after layer 3. Scope conditions: aggregation agent quality is the binding constraint (12.82% gap between best and worst aggregators); baseline cost is ~119K tokens per query; benchmarks are instruction-following, not code-review or cross-file verification.

- **Self-MoA outperforms heterogeneous MoA on instruction-following benchmarks, complicating the heterogeneity prior.** Li et al. (arXiv:2502.00674, ICLR 2025) find +6.6% improvement from same-family multi-sampling over cross-family mixing on AlpacaEval 2.0. The resolution: same-family quality consistency wins where the task requires correctness on well-defined questions; cross-family coverage wins where task-class-specific systematic blind spots matter (the cycle's cross-file verification finding, Cycle 3 Spike A3). The distinction is task-class-dependent, not universal.

- **Deliberation protocols fail catastrophically relative to best-single selection.** DeliberationBench (arXiv:2601.08835, December 2025) finds 82.5% best-single win rate vs. 13.8% for the best deliberation protocol — a 6.0× performance gap at 1.5–2.5× higher compute cost. This reinforces the Yao 2025 panel-discussion diversity-destruction finding (Cycle 2 Loop 4) with much stronger effect size. The practical implication: parallel independent roles with concatenation (MARG-style) are strongly preferred over any sequential deliberation topology for the cycle's composition shapes.

- **Generative topology construction (DyTopo, GTD, HERA) is validated at frontier tier with engineering overhead that exceeds the cycle's cheap-orchestrator constraint.** DyTopo achieves +6.2 average over static baselines via round-by-round semantic matching (arXiv:2602.06039, 2026). HERA achieves 38.69% improvement via experience-guided reward sampling (arXiv:2604.00901, 2026). GTD uses graph diffusion models (arXiv:2510.07799, 2025). All three require infrastructure (embedding models, reward signals, or diffusion model training) that is not directly deployable in llm-orc's five-tool surface without architectural extension.

- **Typed contracts for ensemble interfaces are a practitioner engineering standard with no peer-reviewed performance delta measurement.** The MCP specification (November 2025) mandates output schemas; PydanticAI (16,000+ stars) enforces them at the framework level. The Composition Validator (existing in llm-orc) implements the compose-time version of this discipline. The ecosystem trajectory is toward runtime enforcement at every invocation boundary, which the architecture's Amendment #3 (Result Summarizer Harness) partially addresses at the output side. No academic paper measures the reliability delta from typed-contract enforcement.

- **Calibration-gated deferral at the model-selection boundary works: 15% deferral matches full large-model quality.** ReDAct (arXiv:2604.07036, 2026) demonstrates this in text-based embodied environments. Chuang et al. (arXiv:2502.04428, 2025) show that UQ method choice matters more than threshold level, and that calibration data effectively bootstraps routing performance. The direct transfer to ADR-007's composition-level trust gating (rather than model-selection-level gating) is conceptually sound but empirically untested in the published literature.

- **Retrieval-grounded routing outperforms static routing and judgment-based routing** on knowledge-intensive tasks. R3AG demonstrates this for retriever selection (arXiv:2604.22849, 2026); HERA demonstrates it for agent topology selection (arXiv:2604.00901, 2026). The mechanism in both cases is query-specific capability assessment grounded in historical performance data rather than real-time LLM judgment.

**Focus Area 2 — Script-Models Composition Shapes:**

- **Parallel DAG-planned function execution achieves 3.7× latency speedup and 6.7× cost reduction over ReAct.** LLMCompiler (Kim et al., arXiv:2312.04511, ICML 2024) is the foundational result for script-composition as DAG scheduling. The LLM generates the dependency graph once; deterministic code executes in parallel. Validated at frontier tier; applicability at cheap-cloud-orchestrator tier for DAG planning is not measured.

- **Compilation to a deterministic JSON blueprint reduces per-execution cost 1,500× for repetitive stable workflows.** Agentic Compilation (arXiv:2604.09718, 2026) shows 80–94% zero-shot compilation success rates on structure-amenable tasks (data extraction, form filling). The cost argument: $150 continuous agent → under $0.10 compiled blueprint over 500 executions. Applicable to the cycle's link-checking and model-profile validation scripts, which satisfy the "stable, repetitive" criterion.

- **Deterministic tools embedded in code-generation ensembles break the popularity trap.** Wisdom and Delusion of LLM Ensembles (arXiv:2510.21513, October 2025) shows CrossHair counterexample feedback (a deterministic static analysis tool) prevents LLM ensemble consensus from overriding tool-generated evidence of incorrectness. This is the published support for the script-member-alongside-LLM pattern (Spike A3's design), validating the mechanism: deterministic outputs are categorically consensus-resistant.

- **Task-class amenability to compilation follows a clear boundary: structured verification yes, semantic analysis no.** Compiled AI (arXiv:2604.05150) identifies "stable, repetitive, structure-amenable workflows" as the amenable class. For the cycle: link checking, model-profile existence verification, section presence detection compile cleanly; semantic reasoning about architectural implications does not. The hybrid mode (compiled extraction + LLM for semantic open-endedness) is the pattern that matches the cycle's task class structure.

- **FSM-based multi-agent systems enable state traceback for error recovery in deterministic routing.** MetaAgent (arXiv:2507.22606, ICML 2025) provides the FSM-controlled routing pattern where transition back to prior states handles downstream failure. This is the deterministic harness pattern applied to multi-agent composition.

- **The conflict-between-deterministic-tool-and-LLM-consensus failure mode is not studied in the published literature.** What happens when the script's verified finding conflicts with LLM ensemble consensus? No paper studies this. It is an open reliability question for the cycle's spike program.

**Focus Area 3 — Orchestrator Routing Composition Shapes:**

- **Task-aware delegation cues enable cheap routing by pre-computing capability profiles.** Gu (arXiv:2603.11011, March 2026) demonstrates that routing based on pre-computed task-conditioned win-rate maps and coordination-risk priors enables reliable delegation without real-time LLM capability judgment. This is directly applicable to reducing the routing burden on cheap-cloud-orchestrators.

- **Pre-compiled routing (FSM, DAG, or state-locked policy) is consistently more reliable than LLM-judgment routing.** CAAF state-locking (Wave 1.A), LLMCompiler DAG routing (ICML 2024), and MetaAgent FSM routing (ICML 2025) all demonstrate that pre-compiled routing eliminates a class of LLM-routing failure modes. The design cost is pre-specifiability of the routing logic.

- **Uncertainty-based SLM routing to stronger models demonstrates the calibration-gate principle at the routing boundary.** Chuang et al. (arXiv:2502.04428, 2025) across 1,500+ settings show that UQ-calibrated routing effectively bootstraps reliability without new downstream data. The finding that UQ method choice dominates threshold choice is a design recommendation: the cycle's Calibration Gate's signal-accumulation mechanism should be evaluated for its UQ method choice, not only its threshold level.

- **Bounded autonomy with schema-level enforcement is the convergent deployment practice.** The practitioner and academic literature (Knight, OpenDev, CAAF) converges on encoding action limits at the interface level rather than the prompt level. ADR-008's Autonomy Levels is architecturally aligned with this convergent practice.

---

### Tensions Between Sources

**Tension 1: Same-family quality consistency vs. cross-family diversity for ensemble correctness.**

Self-MoA (arXiv:2502.00674, 2025) and DeliberationBench (arXiv:2601.08835, 2025) both support reducing cross-model heterogeneity. Self-MoA shows +6.6% from same-family sampling over cross-family mixing on instruction-following benchmarks. Cycle 3's Spike A3 (empirical, not yet published) shows cross-family heterogeneity producing coverage that same-family sampling cannot replicate on cross-file verification. The resolution is task-class-conditional: the cycle's Sub-Q3 finding that heterogeneity is load-bearing on cross-file verification is not contradicted by Self-MoA (different task class). The cycle should hold both findings without flattening either.

**Tension 2: Typed contracts constrain generative composition.**

The typed-contract discipline (PydanticAI, Arazzo, MCP specification) and the generative-composition approaches (GTD, DyTopo, HERA) pull in opposite directions. Typed contracts create a fixed schema surface that generative construction must fit within; generative construction requires flexibility to construct novel topologies. The practical resolution from the literature: HERA evolves orchestration *within* a schema-valid topology space (the compositions it generates are still valid against the declared agent schemas). This suggests that schema-level constraints at the member interface boundary (what each ensemble member accepts and returns) are compatible with generative topology construction above that boundary, as long as the interface constraints remain stable.

**Tension 3: Compilation benefits vs. one-time-use overhead.**

Compilation earns its cost over many repetitions of the same workflow (Agentic Compilation: 1,500× cost reduction over 500 executions). For one-time workflows, compilation adds overhead without proportional benefit. The cycle's agentic coding sessions may involve novel configurations that are not repeated — in which case compilation is inappropriate. The boundary condition: compilation is appropriate for the stable verification scripts (which run on every PR); it is inappropriate for the semantic analysis ensemble (which varies per problem).

**Tension 4: Recursive depth improves quality but Khanal et al. show depth degrades long-horizon agents.**

Attention-MoA shows monotonically increasing performance through layer 5 of recursive depth (in a single-query evaluation context). Khanal et al. (Wave 1.A) show that episodic memory scaffolds universally hurt long-horizon performance by consuming step budget. The apparent tension: recursive composition adds value in bounded-depth single-query settings but may impose overhead costs in long-horizon multi-turn settings. The resolution is scope-conditional: Attention-MoA's recursive depth applies within a single ensemble invocation (one `invoke_ensemble` call); Khanal et al.'s finding applies to accumulated session-level scaffolding across many turns. These are different decision surfaces and the tension is not direct contradiction.

---

### Candidate Framings for the Cycle's Analytical Work

These are offered as candidates, not commitments. The dispatch instructions ask for framings that do not presuppose the cycle's direction.

**Framing A: Layer-conditional composition shape selection.**

The Sub-Q3 gate finding (script-models load-bearing for cross-file verification; ensemble composition for documentation specificity; orchestrator routing for hybrid deployment) implies that composition shape selection should be layer-conditional rather than uniform. Framing A would organize the cycle's design-method analysis around which composition shape is appropriate at each layer, given the task class at that layer. This framing directly operationalizes the Sub-Q3 gate finding into a design methodology.

**Framing B: Pre-specifiable vs. generative routing as the primary design choice.**

The literature consistently shows that pre-specifiable routing (FSM, DAG, capability-profile-grounded) is more reliable than generative routing (LLM-judgment, dynamic topology construction) but less flexible. Framing B would organize the cycle's analysis around the pre-specifiable / generative boundary at each layer: what routing decisions can be pre-specified (and therefore harness-enforced), and what routing decisions require generative judgment (and therefore trust-score-monitored). This maps onto the Sub-Q6 transfer-test question from Wave 1.A.

**Framing C: Typed contracts as the interface discipline for composition at every layer.**

The ecosystem convergence on typed contracts (MCP, Arazzo, PydanticAI) provides a discipline that applies at every composition boundary — script-to-script, script-to-ensemble, ensemble-to-ensemble, ensemble-to-orchestrator. Framing C would use typed-contract enforcement as the primary design pattern across all three layers, with the design question being: what is the right contract at each layer boundary, and how does the contract constrain the topology space above it? This framing turns the composition-validator architecture into a general design principle rather than a single ADR-scoped mechanism.

**Framing D: Experience accumulation as the progression from stock-library to retrieval-grounded composition.**

The stock-library shape (fixed ensemble library, orchestrator selects) and the retrieval-grounded selection shape (HERA, R3AG) are related by the presence or absence of accumulated performance experience. Framing D treats the stock library as the starting state (zero experience) and retrieval-grounded composition as the evolved state (sufficient experience accumulated). The Plexus integration provides the experience accumulation infrastructure; the cycle's design question under this framing is: what signals should the experience store capture, and what retrieval mechanism should the orchestrator use to ground routing decisions in that experience?

---

### Limitations

**Scope transfer from frontier to cheap-orchestrator tier.** The most important limitation of this wave: substantially all papers showing composition shape performance improvements use frontier models (GPT-4o, Claude 4.x, Gemini) in at least the aggregation or orchestration role. Attention-MoA's aggregator dependency (12.82% gap between best and worst aggregators) is a direct warning that the composition improvements require capable aggregators. The cycle's cheap-cloud-orchestrator (MiniMax M2.5 Free or equivalent) is below the tier validated in any paper here. Framing these results as establishing a performance ceiling rather than a guaranteed improvement at the cycle's operating tier is the disciplined reading.

**Absence of code-review-specific recursive and generative composition benchmarks.** No paper tests RecursiveMAS, Attention-MoA, DyTopo, or HERA on code-review or cross-file verification task classes. The only published code-domain ensemble evidence with deterministic tool embedding is Wisdom and Delusion (arXiv:2510.21513), which uses CrossHair in a code-generation ensemble. Generalization from instruction-following benchmarks (AlpacaEval, MT-Bench) or mathematical reasoning benchmarks to the cycle's task class requires empirical testing.

**Deliberation protocol content gap.** DeliberationBench (arXiv:2601.08835) identifies three tested deliberation protocols but does not name them in the abstract. The failure conditions for each protocol are not extractable from the available abstract. The 6.0× performance gap finding is strong, but the paper may have granular findings about which protocols fail worse and under what conditions that are not accessible from the abstract-level extraction.

**Typed-contract enforcement for ensemble interfaces: no published performance study.** The practitioner ecosystem treats typed contracts as standard practice; the academic literature has not measured the reliability delta from enforcing them. The cycle would be producing novel empirical evidence if it tests composition reliability with vs. without typed contract enforcement on its task classes.

**Generative composition at cheap-orchestrator tier: no published evidence.** DyTopo, GTD, and HERA are all validated at frontier tier. Whether a cheap-cloud-orchestrator can reliably generate and execute dynamic topologies (rather than selecting from a pre-defined library) is an open empirical question that the literature does not address. This is the cycle's most significant design-method gap.

**Conflict-between-deterministic-and-LLM-consensus failure mode: no published study.** The script-member-alongside-LLM pattern is supported by Compiled AI's hybrid mode and Wisdom and Delusion's CrossHair embedding, but neither paper studies what happens when the deterministic tool's output conflicts with LLM ensemble consensus. This is an open reliability question for the cycle's spike battery.

**HERA's experience accumulation infrastructure requirement.** HERA's 38.69% improvement depends on accumulated experience across queries. In llm-orc's Plexus-absent mode, the experience accumulation infrastructure is not available. The performance figure cannot be reproduced in stateless mode; it represents an upper-bound for what becomes accessible as experience accumulates with Plexus enabled.

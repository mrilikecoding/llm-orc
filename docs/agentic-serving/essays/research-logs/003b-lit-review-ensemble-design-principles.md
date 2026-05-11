## Literature Review: Ensemble Design Principles for Multi-Agent LLM Systems

**Date:** 2026-04-29
**Method:** Systematic literature search — web search across arXiv, ACL Anthology, NeurIPS, ICLR, and practitioner documentation (Anthropic, AutoGen, LangGraph, CrewAI). Approximately 40 distinct sources reviewed; ~20 fetched for detailed extraction.
**Cycle context:** Loop Iteration 4, Cycle 2 (agentic-serving). Complements Loop 1's shape-inventory and failure-modes review. Addresses the HOW dimension: what design principles does the literature prescribe for well-architected ensembles?

---

### Sources Reviewed

| # | Author(s) | Title | Year | Venue | Relevance |
|---|-----------|-------|------|-------|-----------|
| 1 | Shinn et al. | Reflexion: Language Agents with Verbal Reinforcement Learning | 2023 | NeurIPS | RQ-D3: generator-critic conditions |
| 2 | Du et al. | Improving Factuality and Reasoning in Language Models through Multiagent Debate | 2023 | ICML | RQ-D3: debate coordination |
| 3 | Madaan et al. | Self-Refine: Iterative Refinement with Self-Feedback | 2023 | NeurIPS | RQ-D3: self-refinement conditions |
| 4 | Drozdov et al. | MARG: Multi-Agent Review Generation for Scientific Papers | 2024 | arXiv | RQ-D1, RQ-D2: role decomposition + specificity |
| 5 | Anthropic | Building Effective AI Agents | 2024 | Anthropic blog | RQ-D4: threshold conditions, architectural patterns |
| 6 | Anthropic | How We Built Our Multi-Agent Research System | 2025 | Anthropic engineering blog | RQ-D2, RQ-D4, RQ-D5 |
| 7 | Chen et al. | Chain of Agents: Large Language Models Collaborating on Long-Context Tasks | 2024 | NeurIPS | RQ-D3: sequential refinement |
| 8 | Cemri et al. | Why Do Multi-Agent LLM Systems Fail? | 2025 | arXiv (NeurIPS workshop) | RQ-D1, RQ-D2: failure taxonomy MAST |
| 9 | Singh et al. | Designing LLM-based Multi-Agent Systems for SE Tasks: QAs, Design Patterns, Rationale | 2024 | arXiv | RQ-D2, RQ-D4: 16 patterns |
| 10 | Chen et al. | Know the Ropes: A Heuristic Strategy for LLM-based Multi-Agent System Design | 2025 | arXiv | RQ-D2: decomposition heuristics |
| 11 | Yao et al. (multi-agent meta-judge) | Leveraging LLMs as Meta-Judges: A Multi-Agent Framework for Evaluating LLM Judgments | 2025 | arXiv | RQ-D1: aggregation patterns |
| 12 | Zhang et al. | Multi-Agent Debate for LLM Judges with Adaptive Stability Detection | 2024 | arXiv | RQ-D1, RQ-D3: debate vs. collapse |
| 13 | Naik et al. | Beyond Majority Voting: LLM Aggregation by Leveraging Higher-Order Information | 2024 | arXiv | RQ-D1: aggregation strategy comparison |
| 14 | Ding et al. | Wisdom and Delusion of LLM Ensembles for Code Generation and Repair | 2024 | arXiv | RQ-D1: consensus vs. diversity failure |
| 15 | Tran et al. | LLM-Enabled Multi-Agent Systems: Empirical Evaluation and Insights into Emerging Design Patterns | 2025 | arXiv | RQ-D2, RQ-D4, RQ-D5 |
| 16 | Chen et al. | Optima: Optimizing Effectiveness and Efficiency for LLM-Based Multi-Agent System | 2024 | ACL Findings 2025 | RQ-D5: token and communication overhead |
| 17 | Guo et al. | Scaling Large Language Model-based Multi-Agent Collaboration | 2024 | arXiv | RQ-D2: number of agents, scaling law |
| 18 | Sun et al. | Understanding Agent Scaling via Diversity | 2025 | arXiv | RQ-D2: homogeneous vs. heterogeneous |
| 19 | Jiang et al. | Single-Agent LLMs Outperform Multi-Agent Systems on Multi-Hop Reasoning Under Equal Thinking Token Budgets | 2026 | arXiv | RQ-D4, RQ-D6: single vs. multi-agent |
| 20 | Liang et al. | Graph of Agents: Principled Long Context Modeling by Emergent Multi-Agent Collaboration | 2025 | arXiv | RQ-D3: graph coordination |
| 21 | Tang et al. | AgentReview: Exploring Peer Review Dynamics with LLM Agents | 2024 | EMNLP | RQ-D1: meta-reviewer synthesis |
| 22 | Mazur et al. | DeepReview: Improving LLM-based Paper Review with Human-like Deep Thinking | 2025 | ACL | RQ-D1: synthesis collapse |
| 23 | Various | MAST failure taxonomy (Cemri et al.) | 2025 | arXiv | RQ-D2, RQ-D6: design anti-patterns |
| 24 | Various | Helium: Efficient LLM Serving for Agentic Workflows | 2026 | arXiv | RQ-D5: caching, prefix reuse |
| 25 | La Malfa et al. | Large Language Models Miss the Multi-Agent Mark | 2025 | NeurIPS (position) | RQ-D6: gap identification |
| 26 | Anthropic | Building Effective Agents (architecture PDF) | 2024 | Anthropic | RQ-D1, RQ-D4 |

---

### Synthesis

#### RQ-D1 — Synthesizer / Aggregation Patterns

**What the literature describes.**

The literature identifies at least six operationally distinct aggregation patterns, though it does not always name or taxonomize them explicitly:

1. **Collapse-to-summary** (the most common production pattern). One agent reads N reviewer outputs and writes a unified narrative, discarding structural attribution. DeepReview (Mazur et al., 2025, ACL) uses exactly this: individual reviewer perspectives are "synthesized into comprehensive perspectives" and merged into a single meta-review. The paper acknowledges that "human-level nuance appears partially lost in synthesis" and notes the depth of technical understanding in specific areas may not match a human meta-reviewer. This is the same pattern the production code-review ensemble uses, and the literature treats information loss as an expected cost, not a fixable property of the pattern itself.

2. **Preserve-individual-voice with concatenation** (MARG's architecture). Drozdov et al. (2024) built a multi-agent scientific paper reviewer that runs three independent agent groups (experiments, clarity, novelty), each with its own leader and workers. Rather than running a synthesizer, MARG **concatenates** the three mini-reviews as separate sections. The specificity gain — 71% specific vs. 29% generic for the baseline — is attributed in part to this concatenation approach: individual agent outputs are not collapsed before delivery. Critically, MARG's authors cannot explain mechanistically why specificity improves; they offer three candidate explanations (calibration, MoE routing, evaluator bias) but test none conclusively. The failure mode at 53% of reviews is missing context: the leader agent fails to include necessary information when dispatching workers.

3. **Merge-with-citation** (emerging in research-generation systems). SurveyG and related survey-generation pipelines organize literature within a hierarchical citation graph so that the synthesis preserves provenance to originating sources. Mesh Memory Protocol (2025) provides per-claim lineage via a DAG. Neither applies directly to code-review ensembles, but the pattern is well-established in the literature-synthesis domain and the mechanism — track the claim's origin, not just the claim — is a direct counter to collapse-to-summary's information loss.

4. **Judge-arbitration / panel aggregation**. The LLM-as-a-judge literature has converged on a panel approach: N judge agents independently score or rank, then aggregate via majority voting, weighted averaging, or panel discussion. Yao et al. (2025) find that a **two-agent configuration performs best** among tested configurations, and that majority voting (77.3% precision) outperforms panel discussion (72.6%) because opinions "tend to converge over time," destroying diversity. This is a direct empirical finding: when you run sequential discussion among agents, the later agents anchor on earlier agents' outputs and the effective information carried into the aggregation step decreases. This is the panel-discussion failure mode.

5. **Multi-agent debate with deferred aggregation**. Zhang et al. (2024) show that structured debate — where agents share reasoning across rounds before converging — outperforms single-step majority vote on complex, ambiguous tasks (JudgeBench, LLMBar, TruthfulQA) by up to 6 percentage points, but shows no gain over majority voting on simple tasks with strong initial consensus (BIG-Bench). The mechanism is Bayesian: one strongly correct agent in a round increases expected correctness in round t+1. The debate pattern preserves individual reasoning traces between rounds, unlike collapse-to-summary.

6. **No-aggregation / streaming**. No peer-reviewed paper describes a production "stream reviewer outputs directly to user" pattern for LLM ensembles. The closest analogues are MARG's concatenation and the judge-panel's preserve-and-vote patterns, both of which separate individual agent outputs from the final synthesis.

**What the literature does not reach.**

No paper directly compares collapse-to-summary against concatenation on a code-review task with specificity as the primary metric. The literature has the adjacent evidence (MARG's specificity gain, DeepReview's acknowledged nuance loss, the panel-discussion convergence finding) but not the direct head-to-head the cycle needs. The closest direct test is MARG, which produces a specificity improvement by architectural choice (concatenation) rather than by explicitly testing collapse-to-summary as a condition.

The two-stage summarization anti-pattern (synthesizer inside ensemble + orchestrator synthesis outside) is not studied as a research object in any paper found. The literature identifies single-stage collapse as lossy; double-stage cascade as an explicit design choice has no published analysis.

---

#### RQ-D2 — Role Decomposition and Specialization Principles

**What the literature describes.**

Singh et al. (2024) catalogued 16 design patterns across 71 LLM multi-agent SE systems and found Role-Based Cooperation is the dominant pattern. The paper does not provide a full taxonomy of all 16 (the relevant section is behind the full PDF), but it establishes that role decomposition is near-universal in practice. What the paper does state: Functional Suitability is the primary quality attribute designers optimize for (94.7%), suggesting current design is task-completion-driven rather than quality-preservation-driven.

Know the Ropes (Chen et al., 2025) proposes the most explicit decomposition heuristic found in this search: translate domain priors into an algorithmic blueprint hierarchy; tasks are recursively split into typed, controller-mediated subtasks. The key principle is "algorithm-aware decomposition" — the structure of the workflow should reflect the structure of the algorithm, not be imported generically. This is grounded in the No-Free-Lunch theorem: there is no universal prompt, so decomposition must be task-specific.

The heterogeneity finding from agent scaling research is the most empirically grounded decomposition principle: **heterogeneous agents outperform homogeneous ones**. Sun et al. (2025) show that scaling homogeneous agents (identical models, prompts, configs) produces strong diminishing returns, while heterogeneous agents (different models, personas, or tools) produce complementary coverage. Deploying agents based on different foundation models — Gemini-Pro, PaLM 2-M, Mixtral 7B×8 — achieves 91% vs. 82% on GSM-8K relative to homogeneous agents. The mechanism is diversity: homogeneous agents converge to the same errors; heterogeneous agents' errors are uncorrelated.

Ding et al. (2024) make the converse point for code-generation ensembles: consensus-based selection (majority voting on identical agent outputs) falls into a "popularity trap," amplifying shared mistakes rather than surfacing correct answers. A diversity-based selection strategy — choosing the most complementary pair from the ensemble, not the most popular answer — recovers up to 95% of the theoretical performance ceiling even in two-model ensembles. The design implication: role decomposition must produce agents that are **cognitively distinct**, not just nominally distinct.

The literature does not draw a crisp distinction between cognitive specialization (different reasoning approaches) and domain specialization (different knowledge areas), but the operational distinction appears in practice: MARG's three expert groups are domain-specialized (experiments, clarity, novelty), while the judge-panel literature (Yao et al., 2025) assigns cognitive roles (generator, critic, meta-judge). The scaling research (Sun et al., 2025) treats model-level heterogeneity as a proxy for both.

**Conditions favoring role asymmetry (judge above debaters) vs. role symmetry (peer reviewers).**

No paper offers a general principle for this. The empirical evidence suggests:
- When the task has a ground-truth answer that can be verified, judge-arbitration adds value (math, code correctness).
- When the task is subjective and benefits from diverse perspectives surviving to the output, peer-review concatenation preserves more signal than arbitration collapse.
- AgentReview (Tang et al., 2024) finds that inclusive area chairs (who integrate all reviewer input rather than dominating) produce the closest outcomes to human baseline; authoritarian area chairs (who override reviewer consensus) show lower correlation. This is indirect evidence that role-asymmetric designs are fragile to the quality of the judge role.

**Number of roles and agents.**

Guo et al. (2024) find a logistic-growth scaling law: performance improves with agent count but saturates, and irregular (heterogeneous) topologies outperform regular (homogeneous) ones. Anthropic's production research system embeds explicit effort-scaling rules: 1 agent for simple fact-finding, 2–4 for direct comparisons, 10+ for complex research. The practical ceiling identified in Tran et al. (2025) is that "agent performance often degrades when the toolkit expands beyond 8–12 tools, due to context-window overload" — but this is a tool-count constraint on a single agent, not a multi-agent role-count constraint per se.

**Anti-patterns documented in the literature.**

MAST (Cemri et al., 2025) identifies 14 failure modes across three categories. Most relevant to role decomposition are FM-1.2 (violating role definitions), FM-1.4 (losing conversation context), FM-2.5 (ignoring other agents' input), and FM-2.4 (withholding relevant information). These are role-boundary failures: agents either exceed their defined role, drop context at handoff, or fail to integrate peer outputs. None is framed as "the decomposition itself was wrong" — MAST treats these as execution failures, not architectural failures.

The closest the literature comes to "role decomposition that empirically harms performance" is the AgentReview finding that one under-committed reviewer triggers 18.7% decline in commitment across all reviewers (altruism fatigue). If a role is assigned to a model that underperforms that role, the contamination propagates. This is a capability-tier-sensitivity finding: role decomposition is only as good as the model matched to each role.

---

#### RQ-D3 — Coordination Protocols Beyond Parallel-with-Collapse and Debate

**Sequential refinement (Chain of Agents, Graph of Agents).**

Chain of Agents (Chen et al., NeurIPS 2024) is the canonical sequential-refinement protocol: worker agents process sequential text chunks and pass a "useful updated information" summary to the next worker; a manager agent synthesizes the final output from the last worker's accumulated evidence. CoA improves by up to 10% over RAG, full-context, and prior multi-agent approaches on long-context QA, summarization, and code completion. The condition for its advantage is explicit: long inputs that exceed a single context window, where each agent handles a short context. The sequential chain preserves information across chunks better than RAG's retrieval filtering. However, the pattern is a forward-only pipeline: information cannot flow back, and the final manager stage is a collapse-to-summary.

Graph of Agents (Liang et al., 2025) extends CoA by dynamically constructing input-dependent collaboration graphs rather than fixed chains, framing collaboration as an information-theoretic compression problem. It outperforms static multi-agent baselines by 16.35% and achieves long-context performance beyond models with 128K context windows. The dynamic structure is a genuine advance on static protocols, but the underlying synthesis step is still compression-oriented.

**Generator-critic / Reflexion-style loops.**

Reflexion (Shinn et al., NeurIPS 2023) uses verbal reinforcement: an actor generates, an evaluator scores, a self-reflection model produces verbal feedback that is stored in episodic memory and used in subsequent trials. The critical limitation, confirmed by post-2023 work: a single LLM serving as both actor and evaluator generates plausible but incorrect feedback and reinforces errors rather than correcting them. Self-bias (anchoring on initial outputs) and degeneration-of-thought (repeating flawed reasoning across iterations despite identified failures) are the primary failure modes. External feedback (test results, verifiers, human signals) substantially improves Reflexion over intrinsic self-reflection. Self-Refine (Madaan et al., 2023) shows ~20% improvement in human preference but cannot correct reasoning errors without external ground truth. The consensus across the 2024–2025 literature: **intrinsic self-correction is not reliably effective; external evaluation signals are required for the loop to be beneficial**.

MAR (multi-agent Reflexion, 2024) partially addresses this by using intentionally diverse critic personas so that the same model does not evaluate its own output. This requires ~300–400 API calls per task (3x single-agent Reflexion cost), and the improvement is task-dependent.

**Plan-execute-verify (PEV).**

No single canonical paper on PEV was identified; it appears as a component of frameworks rather than a studied pattern. DAAO (2025) implements a version (difficulty estimation → operator allocation → verification) but is focused on difficulty-adaptive routing rather than verification per se. The literature treats PEV as a natural extension of orchestrator-worker, not as a distinct protocol with its own empirical study.

**Voting and ensemble methods.**

Majority voting is well-studied and has known failure conditions. Naik et al. (2024) propose Optimal Weight (OW) and Inverse Surprising Popularity (ISP) algorithms that use first- and second-order information (correlations between model outputs) to weight votes. These consistently outperform majority voting across UltraFeedback, MMLU, and ARMMAN. Standard majority voting fails when: (a) correct solutions are in the minority, (b) models share correlated failure modes. The second condition is equivalent to the homogeneity anti-pattern: voting only aggregates information when the vote sources are diverse.

**Async coordination.**

AutoGen v0.4 rearchitected to async-first with an event-driven core. Empirical benchmarks show async reinforcement learning achieves 2.77x speedup vs. synchronous training in MARL settings (AREAL-boba2, 2025). Protocol comparison work (protocol-bench, 2025) finds completion time varies by up to 36.5% across A2A, ACP, ANP, and Agora under streaming queue conditions, with mean latency differences of 3.48 seconds. The Helium system (2026) proposes proactive prefix caching and cache-aware scheduling for agentic workflows, achieving substantial reduction in redundant prefill computation — but focuses on batch workloads, not streaming aggregation.

---

#### RQ-D4 — Task → Architecture Mapping Principles

**What the literature reaches.**

Anthropic's "Building Effective Agents" (2024) articulates the threshold conditions (parallelization, context-window-exceeding, diverse-perspectives), but not the further question: *given a task meets the threshold, what architecture is appropriate?* This is the substantive gap.

The most concrete mapping principles found are:

- **Disparate/fragmented data** → Single Information Environment with specialist agents and routing coordinators (Tran et al., 2025).
- **Long input exceeding context window** → Sequential chain (CoA, Chen 2024) or dynamic graph (GoA, Liang 2025).
- **Tasks requiring breadth-first parallel exploration** → Orchestrator-workers with parallel subagents (Anthropic 2025: 90.2% gain over single-agent Opus on breadth-first queries).
- **Tasks with verifiable ground truth** → Judge-arbitration or voting; majority voting with diverse models outperforms consensus with homogeneous models.
- **Complex, ambiguous evaluation tasks** → Debate (Zhang et al., 2024: up to 6pp gain on JudgeBench); no gain over majority vote on simple tasks.
- **Query difficulty varies widely** → Difficulty-aware routing (DAAO, 2025): match architecture complexity to query complexity rather than using a static topology.

The PublicAgent study (2025) finds that specialization provides value independent of model strength — even the strongest models benefit from specialized agents. Universal agents (discovery, analysis) show lower variance (12.4% SD) than conditional agents (report, intent) (20.5% SD), suggesting that open-ended exploratory roles tolerate model heterogeneity better than precision-requiring roles.

**What the literature does not reach.**

No published taxonomy maps task properties — decomposability, domain breadth, verification difficulty, stakeholder specificity requirements — to synthesizer-pattern choices specifically. Anthropic's research does not discuss when to use concatenation vs. collapse-to-summary vs. voting. MARG's architecture choice (concatenation) is presented as an implementation detail, not a principled selection from a taxonomy. The field has task taxonomies (CoA's "long-context" category, the debate literature's "ambiguous vs. factual" split) but not a synthesizer-selection taxonomy that flows from task properties.

---

#### RQ-D5 — Design Patterns That Minimize Cascade Overhead

**Token and schema overhead.**

Optima (Chen et al., ACL Findings 2025) is the most directly relevant paper: it trains models to communicate efficiently in multi-agent settings using an iterative generate-rank-select-train paradigm. Optima achieves up to 2.8x performance gain with less than 10% of the token count on information-asymmetric tasks. The key finding: most of the token cost in multi-agent communication is unnecessary redundancy — agents restating context that other agents already have. Trained communication policies (via SFT or DPO) can compress messages dramatically without losing task performance.

Tran et al. (2025) documents the coordination overhead empirically: "agent performance often degrades when the toolkit expands beyond 8–12 tools, due to context-window overload." The schema overhead finding from the cycle's own observations (15K tokens of tool-schema-per-turn) is independently confirmed in the practitioner literature: a 2026 Red Hat engineering report finds JSON schema definitions create a "massive scaling bottleneck" and replacing tool schemas with sandboxed Python cuts token overhead by 53% on identical tasks.

The MAST taxonomy confirms that FM-1.4 (losing conversation context) and FM-2.5 (ignoring other agents' input) are among the most common failure modes — both are effects of context-window saturation in high-overhead cascades.

**Async vs. synchronous cascade.**

The evidence for async coordination is positive for training and throughput but has a direct complexity cost: async introduces message brokers, callbacks, and event loops that are harder to debug. For the specific case of consumer-hardware cascade latency (Spike B's finding that plumbing dominates wall-clock), the literature offers no direct mitigation beyond: (a) reduce the number of cascade hops, (b) use async scheduling, (c) eliminate redundant token overhead. No paper describes a "streaming aggregation" pattern where reviewer outputs are presented to the user incrementally before synthesis completes — this is operationally attractive but absent from the peer-reviewed literature.

**Pre-allocated context and shared prefix.**

Helium (2026) proposes proactive prefix caching using a Templated Radix Tree that models the global prefix hierarchy across entire workflows. For static prompt prefixes — including tool schemas — this eliminates redundant prefill across calls. This is directly applicable to the cycle's schema-overhead finding: if tool schemas are static across turns, proactive caching eliminates the per-turn overhead. However, Helium is a serving-system design, not an ensemble-design pattern; it requires framework-level support.

**Partial-completion pickup.**

No paper describes an orchestrator that acts on the first reviewer to complete without waiting for stragglers. The concept exists in streaming LLM inference (token-by-token streaming) but has not been studied as an ensemble coordination strategy.

---

#### RQ-D6 — Honest Gap-Reporting

The design-principles literature has the following profile:

**Comparatively rich (existing body of work):**
- Aggregation pattern descriptions at a conceptual level (collapse, vote, debate, concatenate, cite)
- Voting strategy research with empirical comparisons (Naik et al., 2024; Ding et al., 2024)
- Conditions for debate vs. voting (Zhang et al., 2024)
- Self-refinement conditions and failures (Reflexion, Self-Refine, and the 2024 critical literature)
- Agent scaling laws and diminishing returns (Guo et al., 2024; Sun et al., 2025)
- Failure taxonomies (MAST, Cemri et al., 2025)

**Sparse (no published empirical comparison):**
- Direct head-to-head between collapse-to-summary and concatenation on a specificity-preserving metric. MARG provides the adjacent evidence but not the controlled comparison.
- Two-stage or multi-stage summarization as an object of study. The literature describes single-stage collapse; the production pattern of synthesizer-inside-ensemble + orchestrator-synthesis-outside has no published analysis.
- Synthesizer-selection principles mapped from task properties. Which synthesizer pattern is appropriate for which task class? The field has the vocabulary (collapse, vote, concatenate, cite, debate) but not a task-to-synthesizer taxonomy.
- Code-review-specific ensemble design evaluation. MARG is for scientific paper review; the code-review domain has no equivalent empirical benchmark comparing ensemble architectures.
- Consumer-hardware cascade latency optimization. The literature on overhead reduction (Optima, Helium) operates at cloud inference scales. Spike B's plumbing-dominates-wall-clock finding has no direct published treatment.
- Anti-patterns at the role-decomposition level. MAST identifies execution failures but does not study "the decomposition itself was wrong" as a distinct failure class.

**The practitioner literature partially fills some gaps** — Anthropic's 2025 engineering blog provides the most detailed production design guidance found, including the scaling rules (1 agent / 2–4 subagents / 10+ subagents) and the CitationAgent attribution pattern. But this is a blog post about one system, not a controlled study.

**Overall gap assessment for the cycle:**

The design-principles literature is richer than the bio-inspired / eusocial literature gap identified in Loop 1, but it is not the principled engineering corpus one might hope for. The field has:
- Taxonomy and vocabulary: yes
- Empirical comparisons across controlled conditions: limited, domain-specific (scientific review, not code review)
- Prescriptive "given task property X, choose pattern Y" guidance: weak beyond threshold conditions and a few coarse mappings
- Analysis of the specific failure mode Spike A revealed (two-stage summarization destroys specificity): absent

The cycle's downstream decision from MQ-1 is therefore: the literature provides enough scaffolding to *design* a novel ensemble grounded in named patterns (concatenation over collapse, heterogeneous roles, external evaluation signals for loops, async if infrastructure supports it), but it does not provide pre-validated evidence that such a design will outperform the prompt-steered single agent baseline from Spike A. That test must be empirical.

---

### Key Findings

- **Collapse-to-summary is the dominant aggregation pattern in practice, and the literature acknowledges it loses nuance.** DeepReview explicitly notes human-level nuance is "partially lost in synthesis"; this is treated as an inherent cost of the pattern, not a solvable design problem. (Mazur et al., 2025, ACL)

- **Concatenation preserves individual reviewer specificity better than collapse.** MARG's three-group concatenation architecture reduces generic comment rates from 60% to 29% (2.2x improvement in specific comments), though the causal mechanism is not definitively established. The architecture choice — no synthesizer, preserve separate sections — is the distinguishing feature. (Drozdov et al., 2024)

- **Panel discussion among agents actively destroys diversity.** Sequential discussion causes judge opinions to "converge over time," and the panel-discussion condition achieves lower precision than majority voting (72.6% vs. 77.3%) in the meta-judge literature. This is the same failure mode as the production ensemble's tech-lead synthesis: downstream compression amplifies the earlier convergence loss. (Yao et al., 2025)

- **Homogeneous agents fail at the popularity trap; heterogeneous agents succeed via diversity.** Consensus selection on homogeneous ensembles amplifies shared errors; diversity-based selection recovers up to 95% of theoretical performance ceiling even in two-model configurations. (Ding et al., 2024)

- **Intrinsic self-correction is not reliably effective.** Self-generated verbal feedback (Reflexion) reinforces errors when the same model evaluates its own output. External evaluation signals — test results, verifiers — are required for the generator-critic loop to produce net gains. (Shinn et al., 2023; 2024 follow-on critical literature)

- **Single-agent systems are information-theoretically more efficient than multi-agent systems under equal compute budgets.** Under matched reasoning-token budgets, single agents consistently match or exceed multi-agent performance on multi-hop reasoning. Multi-agent becomes competitive only when single-agent effective context utilization is degraded or additional compute is provided. This is a direct information-theoretic argument (Data Processing Inequality). (Jiang et al., 2026)

- **Multi-agent gains are task-class-dependent: breadth-first exploration benefits most.** Anthropic's production system shows 90.2% gain over single-agent Opus on breadth-first research queries. Debate shows up to 6pp gain on complex, ambiguous evaluation tasks but no gain on simple tasks. Sequential chains show 10% gain on long-context tasks. For code review — a depth-first, precision-requiring task — no equivalent empirical benchmark exists. (Anthropic 2025; Zhang et al., 2024; Chen et al., 2024)

- **Token overhead in cascade systems is structurally large and reducible.** Optima achieves 2.8x task gain with <10% of baseline token count through trained communication policies. Schema overhead independently contributes 15K+ tokens per turn in tool-heavy cascades. (Chen et al., 2024; Red Hat engineering, 2026)

- **The design-principles literature has vocabulary but lacks prescriptive synthesizer-selection guidance.** No published paper maps task properties to synthesizer-pattern choices. The closest framework is Anthropic's threshold + effort-scaling rules, which address when to use multi-agent at all, not which aggregation pattern to use given a multi-agent decision.

---

### Limitations

**Search coverage.** PDF parsing failed for several papers (DeepReview, MARS) where HTML versions returned binary-encoded content. Findings from these papers are reconstructed from abstract-level information and HTML fragments. The 16-pattern full taxonomy from Singh et al. (2024) was not extractable from the available web-accessible version.

**Domain mismatch.** Most empirical comparisons are in scientific paper review (MARG, AgentReview, DeepReview), judge evaluation (LLM-as-a-judge literature), or mathematical reasoning (debate, self-refinement). Code review as a domain has no equivalent controlled benchmark for ensemble design comparison. Generalization from scientific review to code review is plausible but untested.

**Recency and pre-publication risk.** Several sources are 2025–2026 arXiv preprints that have not been peer-reviewed. The Jiang et al. (2026) single-agent-outperforms paper is an April 2026 preprint; the finding is consistent with the cycle's Spike A results but should be treated as preliminary.

**No consumer-hardware treatment.** The overhead-reduction literature (Optima, Helium) operates at cloud inference scales. The cycle's specific finding — that cascade plumbing dominates wall-clock on consumer hardware regardless of inner-model size — is not directly addressed by any paper found. The closest treatment is the schema-overhead practitioner report, which is a single engineering observation rather than a controlled study.

**Absence of negative results.** The literature has few papers that systematically test a design choice and find it doesn't help. Most papers propose a new design and show it outperforms the baseline. The cycle's Spikes A and B are closer to the spirit of adversarial comparison than most of what was found in the published literature.

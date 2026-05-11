## Literature Review: Agent Design — Outcomes Over an Agentic Session

**Date:** 2026-05-01
**Method:** Systematic literature search (web search + primary source fetch across arXiv, ACL Anthology, NeurIPS, ICLR, and practitioner documentation)
**Cycle:** 3 (agentic-serving scoped corpus)
**Research questions covered:** RQ-1 (isolate A3's load-bearing component), RQ-2 (four-priorities frame as hypothesis test), RQ-3 (multi-turn reliability and observable failure modes), cross-cutting outcome-framing question, targeted script-as-orchestrator coverage instruction

---

## 1. Frame Statement

**Operating frame:** "Outcomes over an agentic session. Agent shape is means; the cycle's evaluation axis is outcome quality per session, not architectural fidelity to any pre-existing shape."

This frame is operative throughout this review. Agent shapes (LLM-with-tools, script-driven loops, hierarchical decomposition, state machines with LLM nodes, ensembles, hybrid architectures) are treated as means whose value is judged against the outcomes they produce per session.

**Discipline notes confirmed as operative:**

1. **Narrow reading first.** Where a finding could be read narrowly (specific to one model tier, one task class, one configuration) or broadly (general architectural claim), this review reports the narrow reading and marks the broader reading explicitly as an extrapolation requiring additional evidence.

2. **Alternative framings preserved.** Where the literature contains genuinely competing framings, both are reported with their evidence bases. The single-agent-vs-multi-agent question has at least two competing framings in the current literature (see RQ-1 and Cross-RQ Patterns) that are preserved as alternatives rather than flattened.

3. **Honest absence reporting.** Where coverage is sparse or absent, this is stated directly.

---

## 2. Sources Reviewed

| # | Author(s) | Title | Year | Venue | Relevance |
|---|-----------|-------|------|-------|-----------|
| 1 | Yao, Liu, et al. (Sierra Research) | τ²-Bench: Evaluating Conversational Agents in a Dual-Control Environment | 2025 | arXiv:2506.07982 | RQ-3: tau-bench follow-up, dual-control multi-turn |
| 2 | Cao, Driouich, Thomas | Beyond Task Completion: Revealing Corrupt Success in LLM Agents through Procedure-Aware Evaluation | 2026 | arXiv:2603.03116 | Cross-cutting: outcome operationalization, PAE framework |
| 3 | Zhu et al. | Beyond Task Completion: An Assessment Framework for Evaluating Agentic AI Systems | 2025 | arXiv:2512.12791 | Cross-cutting: four-pillar evaluation (LLMs, Memory, Tools, Environment) |
| 4 | Rony et al. | Beyond Accuracy: A Multi-Dimensional Framework for Evaluating Enterprise Agentic AI Systems (CLEAR) | 2025 | arXiv:2511.14136 | RQ-2: multi-objective evaluation, cost-latency-efficacy-assurance-reliability |
| 5 | Xu et al. | Rethinking the Value of Multi-Agent Workflow: A Strong Single Agent Baseline (OneFlow) | 2026 | arXiv:2601.12307 | RQ-1: single-agent simulation of homogeneous multi-agent workflows |
| 6 | Lee et al. | When Single-Agent with Skills Replace Multi-Agent Systems and When They Fail | 2026 | arXiv:2601.04748 | RQ-1: conditions for single-agent vs multi-agent superiority |
| 7 | Fan et al. | Information Fidelity in Tool-Using LLM Agents: A Martingale Analysis of the Model Context Protocol | 2026 | arXiv:2602.13320 | RQ-1: error accumulation in sequential tool calls, re-grounding interval |
| 8 | Batra et al. (XY.AI Labs / Stanford / Cornell) | Compiled AI: Deterministic Code Generation for LLM-Based Workflow Automation | 2026 | arXiv:2604.05150 | Script-as-orchestrator: LLM confined to compilation phase, deterministic execution |
| 9 | Xu et al. | Routine: A Structural Planning Framework for LLM Agent System in Enterprise | 2025 | arXiv:2507.14447 | Script-as-orchestrator: structured planning scripts, 41%→96% accuracy on GPT-4o |
| 10 | Krupp et al. | Promoting Sustainable Web Agents: Benchmarking and Estimating Energy Consumption | 2025 | arXiv:2511.04481 | RQ-2: energy benchmarking for web agents, design philosophy impact |
| 11 | Jegham, Abdelatti et al. | How Hungry is AI? Benchmarking Energy, Water, and Carbon Footprint of LLM Inference | 2025 | arXiv:2505.09598 | RQ-2: 30-model environmental footprint benchmark, 65× efficiency variance |
| 12 | Jang, Morabito | Edge-First Language Model Inference: Models, Metrics, and Tradeoffs | 2025 | arXiv:2505.16508 (IEEE ICDCS 2025) | RQ-2: local-first inference tradeoff analysis, edge vs cloud |
| 13 | Narayanan et al. | Quantifying Energy and Cost Benefits of Hybrid Edge Cloud: Analysis of Traditional and Agentic Workloads | 2025 | arXiv:2501.14823 | RQ-2: hybrid edge-cloud energy savings up to 75%, cost reduction >80% |
| 14 | Liu et al. | How Good Are LLMs at Processing Tool Outputs? | 2025 | arXiv:2510.15955 | RQ-1: structured tool output processing quality, schema inclusion helps |
| 15 | Cemri et al. | Why Do Multi-Agent LLM Systems Fail? (MAST taxonomy) | 2025 | arXiv:2503.13657 (NeurIPS workshop) | RQ-3: 14 failure modes, role-boundary failures |
| 16 | AgentEval team | AgentEval: DAG-Structured Step-Level Evaluation for Agentic Workflows with Error Propagation Tracking | 2026 | arXiv:2604.23581 (ACL 2026 Industry Track) | Cross-cutting + RQ-3: DAG evaluation, failure detection recall |
| 17 | Lu et al. | Stop Wasting Your Tokens: Towards Efficient Runtime Multi-Agent Systems (SupervisorAgent) | 2026 | arXiv:2510.26585 (ICLR 2026) | RQ-3 + RQ-2: runtime supervision, 29.68% token reduction without accuracy loss |
| 18 | He et al. | SkillReducer: Optimizing LLM Agent Skills for Token Efficiency | 2026 | arXiv:2603.29919 | RQ-2: token cost, skill schema overhead, 48% description compression |
| 19 | Cleanlab Research | Automated Hallucination Correction for AI Agents: A Case Study on Tau²-Bench | 2025 | cleanlab.ai | RQ-3: trust scoring cuts agent failure rate up to 50% on tau2-bench |
| 20 | Mohammadi et al. | Evaluation and Benchmarking of LLM Agents: A Survey | 2025 | arXiv:2507.21504 (KDD 2025) | Cross-cutting: two-dimensional taxonomy of agent evaluation |
| 21 | GTA benchmark team | The Evolution of Tool Use in LLM Agents: From Single-Tool Call to Multi-Tool Orchestration | 2026 | arXiv:2603.22862 | RQ-1: sequential tool usage benchmarks, ToolAcc/ArgAcc metrics |
| 22 | Srivastava et al. | Towards Outcome-Oriented, Task-Agnostic Evaluation of AI Agents | 2025 | arXiv:2511.08242 | Cross-cutting: outcome-oriented vs task-completion framing |
| 23 | Chen et al. | From Static Templates to Dynamic Runtime Graphs: A Survey of Workflow Optimization for LLM Agents | 2026 | arXiv:2603.22386 | Script-as-orchestrator: DAG topology, deterministic vs LLM-controlled |

### Sources inherited from Cycle 2 (not duplicated; referenced where Cycle 3 literature updates or refines the picture)

Cycle 2 Loop 1 reviewed 28 sources (tau-bench Yao et al. 2024, HORIZON Wang et al. 2026, AMA-Bench Zhao et al. 2026, LongCLI-Bench Feng et al. 2026, Khanal et al. 2026, MASS Zhou et al. 2025, Li et al. ICLR 2026, Madigan et al. 2025, Choi et al. ACL 2026, Rahman & Schranz 2025, and others). Cycle 2 Loop 4 reviewed 26 sources (MARG Drozdov et al. 2024, Ding et al. 2024, Sun et al. 2025, Jiang et al. 2026, Yao et al. 2025 meta-judge, and others). Those reviews are at `003a-lit-review-multi-turn-and-composition.md` and `003b-lit-review-ensemble-design-principles.md`. This review builds on them, covering new work and re-examining framings the Cycle 3 operating frame opens up.

---

## 3. Per-RQ Findings

---

### RQ-1 — Isolate A3's Load-Bearing Component

**Question:** Does a prompt-steered single cloud orchestrator receiving a script-agent's deterministic report as additional input context ("A2 + script input") produce equivalent factual grounding to A3's novel ensemble (script + heterogeneous LLMs + MARG concatenation) on the cycle-2 README-review task class?

#### 3.1.1 Tool-Augmented Prompt Steering

**OneFlow (arXiv:2601.12307, Xu et al., January 2026).** This paper directly challenges the assumption that multi-agent workflows require multiple distinct model instances. The paper tests whether single-agent simulation via multi-turn conversations with KV cache reuse can match the performance of homogeneous multi-agent workflows. Across seven benchmarks spanning coding, mathematics, QA, domain reasoning, and planning-and-tool-use, a single agent using multi-turn conversations with KV cache achieves performance comparable to homogeneous multi-agent setups at lower cost. The proposed OneFlow algorithm automatically designs tailored workflows for single-agent execution.

*Narrow reading:* This finding is scoped to **homogeneous** workflows (all agents share the same base LLM, differing only in prompts and positions). The paper explicitly identifies this as its key limitation: "Single-LLM approaches have inherent limitations: they cannot simulate truly heterogeneous workflows due to the inability to share KV caches across different models." The finding does not apply to A3's architecture, which uses heterogeneous models from different families (Hunyuan and Kimi). The paper thus supports the A2 baseline's adequacy for *homogeneous* ensemble replacement but does not address whether A2 + script input can match A3's *heterogeneous* ensemble.

*Broader reading (extrapolation requiring additional evidence):* If the finding extends to heterogeneous workflows under equivalent compute, it would suggest that single-agent simulation is competitive with any ensemble topology. Jiang et al.'s April 2026 information-theoretic argument (cited in Cycle 2) is directionally consistent with this broader claim, but both papers acknowledge heterogeneous topologies as the residual unsettled case.

**When Single-Agent with Skills Replace Multi-Agent Systems (arXiv:2601.04748, Lee et al., January 2026).** This paper frames multi-agent compilation as skill distillation: a multi-agent system can be compiled into an equivalent single agent by trading inter-agent communication for skill selection. Preliminary experiments show substantial reduction in token usage and latency while maintaining competitive accuracy. Critically, the paper identifies a **phase transition** in performance: skill selection accuracy remains stable up to a critical library size, then drops sharply. Semantic confusability among similar skills — not library size alone — drives the degradation. Hierarchical skill organization mitigates this.

*Narrow reading:* The paper tests compiled skill agents on coding and math tasks at frontier-model tier. The finding that single-agent-with-skills matches multi-agent performance is scoped to the task classes and library scales tested. The phase-transition finding is a scope condition for the "A2 + script input" configuration: if the script-agent's output is treated as a skill or structured context, the question of how the orchestrator retrieves and uses that context under context-growth is directly relevant. The "A2 + script input" configuration injects a bounded, deterministic report rather than a large skill library, which places it well below the phase-transition threshold — a favorable scope condition.

**How Good Are LLMs at Processing Tool Outputs? (arXiv:2510.15955, October 2025).** Evaluated 15 models on 1,298 QA samples derived from real API tool outputs across extractive, filtering, and aggregation tasks. Key findings: (a) including the tool output schema in the prompt improves performance by up to 12%; (b) including even a condensed version of the tool response is better than excluding it when it does not fit context; (c) simplified JSON structures improve performance across all models; (d) JSON processing remains difficult even for frontier models.

*Narrow reading:* This finding is directly applicable to the "A2 + script input" configuration. It supports that structured tool output in the prompt context does improve factual grounding, and that schema annotation of that output further helps. The finding is scoped to extractive/filtering/aggregation tasks on structured API responses, not open-ended documentation review. The scope gap between tool-API-response processing and documentation-bug-detection (A3's README task) is not bridged by this paper.

*Broader reading (extrapolation requiring additional evidence):* The finding implies that deterministic tool output, well-structured and injected into context, provides factual grounding gains for a single LLM. Whether those gains are equivalent to A3's heterogeneous-reviewer gains on documentation-review tasks (finding undefined model profiles, onboarding-friction issues) requires direct empirical comparison — which is RQ-1's spike.

**Information Fidelity in Tool-Using LLM Agents (arXiv:2602.13320, Fan et al., February 2026).** Proves theoretically that cumulative distortion in sequential tool call chains exhibits linear growth with O(√T) deviations. Experiments on Qwen2-7B, Llama-3-8B, and Mistral-7B validate this. Key practical finding: semantic weighting reduces distortion by 80%, and periodic re-grounding approximately every 9 steps suffices for error control.

*Narrow reading:* This is scoped to sequential multi-step tool call chains (MCP agents). The finding that re-grounding every ~9 steps controls error accumulation is a concrete design recommendation for multi-turn tool-augmented agents, but is not directly a comparison between single-agent-with-tool-context and multi-agent-ensemble. It speaks to multi-turn reliability (RQ-3) as much as to RQ-1.

**GTA and Tool Evolution Survey (arXiv:2603.22862, 2026).** Early benchmarks (ToolBench, APIBench) evaluated single-step isolated tool use. The current frontier is multi-step sequential tool use, captured in GTA (fine-grained supervision with ToolAcc, ArgAcc, StepAcc metrics). The survey documents a clear evolution toward multi-step tool orchestration as the load-bearing capability.

*Narrow reading:* The survey's finding that multi-step tool orchestration — not single-tool grounding — is the current frontier scope condition on RQ-1. A3's script slot provides *deterministic* multi-step tool execution (link checking, section presence, code block parsing), not LLM-driven tool orchestration. The comparison in RQ-1 is therefore not about "which shape does better LLM tool orchestration" but about "does deterministic verification via script inject sufficient grounding that the LLM does not need heterogeneous reviewers to find the same issues." The published literature does not directly answer this. It is the spike's question.

#### 3.1.2 Comparison Evidence: Single LLM + Tool Context vs. Multi-Component Ensembles

**The literature does not contain a direct published comparison of the "A2 + script input" shape against a MARG-style heterogeneous ensemble on factual-grounding documentation-review tasks.** This is an honest absence.

The closest adjacent evidence is:

- OneFlow showing single-agent matching homogeneous multi-agent (not heterogeneous) on coding/math tasks.
- MARG (Drozdov et al. 2024, covered in Cycle 2) showing concatenation reduces generic comment rate from 60% to 29% in scientific paper review — but MARG's design does not include a script-agent deterministic slot; it is three independent LLM groups.
- The Jiang et al. (April 2026) information-theoretic argument that single agents match multi-agent under equal compute — but this is for multi-hop reasoning, not documentation-grounding.
- Cycle 2's Spike A3, which showed A3 surfaced documentation bugs A2 missed, but did not isolate whether the script slot alone (without heterogeneous LLM reviewers) would have sufficed.

**Alternative framings in the literature:**

*Framing A (augmentation):* The script-agent's value is that it injects deterministic verified facts that a capable single orchestrator can reason from. Under this framing, "A2 + script input" should deliver equivalent factual grounding to A3, because the orchestrator gets the same verified facts and can generate the same analyses from them. Evidence supporting this framing: Cycle 2's Spike A3 showed reviewers *actively used* the script's findings as anchors, which is consistent with "the facts did the work, not the review topology." Supporting literature: tool output processing improvements (arXiv:2510.15955), information fidelity re-grounding (arXiv:2602.13320).

*Framing B (heterogeneous coverage):* The script-agent's value is necessary but not sufficient. The heterogeneous LLM reviewers' uncorrelated errors are what surfaced the undefined model-profile bugs (Reviewer 1 / Hunyuan) and onboarding-friction issues (Reviewer 2 / Kimi) that A2 missed across three trials. Under this framing, "A2 + script input" delivers factual grounding from the script but not the coverage-via-uncorrelated-errors from heterogeneous reviewers. Evidence supporting this framing: Ding et al. (2024, Cycle 2) showing diversity-based selection recovers 95% of theoretical ceiling; Sun et al. (2025, Cycle 2) showing heterogeneous agents have uncorrelated errors.

Neither framing is settled by the literature. RQ-1's spike is the required resolution mechanism. The literature provides no published result that directly tests the "A2 + script input" configuration on the cycle's task class.

---

### RQ-2 — Frame-as-Hypothesis Test

**Question:** Does Cycle 3's evidence produce any agent-design choice that the four-priorities frame (performance × environmental cost × local-first × token cost) would resolve differently than a performance-only frame?

#### 3.2.1 Multi-Objective Framings in the Agent Literature

**CLEAR Framework (arXiv:2511.14136, Rony et al., November 2025).** The CLEAR (Cost, Latency, Efficacy, Assurance, Reliability) framework is the most directly relevant published multi-objective evaluation framework found. Evaluated six agents on 300 enterprise tasks across customer support, data analysis, process automation, software development, compliance, and multi-stakeholder workflows. Key findings:

- Optimizing for accuracy alone yields agents 4.4–10.8× more expensive than cost-aware alternatives with comparable performance. This is the strongest published evidence that performance-only optimization produces frame-divergent recommendations versus a cost-aware frame.
- Agent performance drops from 60% pass rate (single run) to 25% (8-run consistency), quantifying reliability as a distinct axis from raw accuracy.
- Novel metrics: cost-normalized accuracy (CNA), pass@k reliability, policy adherence score (PAS), SLA compliance rate.
- Expert validation (N=15) confirms CLEAR better predicts production success (ρ=0.83) than accuracy-only evaluation (ρ=0.41).

*Narrow reading:* This is scoped to enterprise deployments at frontier model tier (six leading commercial agents) across business-process task classes. The 4.4–10.8× cost premium for accuracy-only optimized agents is a finding at that task class and model tier. Whether the same premium ratio applies to the cycle's small-model local configurations is not measured.

*Broader reading (extrapolation requiring additional evidence):* If cost-only penalty of 4.4–10.8× scales to the cycle's configurations, then any deployment choosing a frontier-tier cloud orchestrator purely on performance grounds would be measured as a frame-divergent recommendation by a cost-aware frame. This is the exact class of trade-off the four-priorities frame names.

**Performance-only as the dominant evaluation axis in benchmarks.** The survey Mohammadi et al. (arXiv:2507.21504, KDD 2025) organizes agent evaluation literature along evaluation objectives (behavior, capabilities, reliability, safety) and evaluation process, and explicitly identifies "absence of cost-controlled evaluation" and "missing multidimensional metrics for security, latency, and policy compliance" as fundamental limitations of existing benchmarks. This is a direct confirmation that performance-only is the dominant evaluation axis in the published benchmark literature, and that multi-priority evaluation is an emerging (not established) research area.

*Narrow reading:* The survey covers published benchmarks as a category; it does not claim that the entire agent research community is performance-only. Practice-oriented guidance (CLEAR, Anthropic's production notes) has moved toward multi-priority framing. The peer-reviewed benchmark literature has not.

**Competing framing (performance-only is defensible):** Jiang et al.'s April 2026 information-theoretic argument and OneFlow's single-agent results were framed and measured entirely on performance axes. The papers do not discuss environmental cost or token cost as design axes. This is not a refutation of multi-priority framing — it is a report of what the literature measures, which is predominantly performance. The four-priorities frame's claim is not that performance is unimportant but that it is insufficient as the sole axis, and the CLEAR evidence supports that claim specifically on the cost axis.

#### 3.2.2 Environmental Cost Reporting in Agent Benchmarks

**How Hungry is AI? (arXiv:2505.09598, Jegham et al., May 2025).** Benchmarks the environmental footprint of LLM inference across 30 state-of-the-art models. Key findings:
- Most energy-intensive models (o3, DeepSeek-R1) consume over 33 Wh per long prompt — more than 70× the consumption of GPT-4.1 nano.
- Claude-3.7 Sonnet ranks highest in eco-efficiency.
- The 65× efficiency variance across models is a published empirical range for the environmental cost axis.
- Reasoning-enabled models produce up to 50× more CO2 than concise response models (cited from ScienceDaily June 2025, referencing this benchmark).

*Narrow reading:* This benchmark measures inference energy for models in commercial datacenters, not for local/edge deployments. The environmental cost axis for *local* inference depends on the energy source of the hardware running the model — potentially zero-carbon if running on renewable-powered edge hardware, or equivalent to consumer electricity mix. The benchmark does not measure local inference configurations. The cycle's four-priorities frame names environmental cost as a qualitative priority without units; this paper provides measurement infrastructure that the frame does not yet use.

**Promoting Sustainable Web Agents (arXiv:2511.04481, Krupp et al., November 2025).** Benchmarks energy consumption of web agents (OpenAI Operator, Google Mariner-class systems). Key finding: different agent design philosophies can cause up to 10× energy consumption difference for similar task outcomes. More energy consumed does not necessarily equate to better results.

*Narrow reading:* This is scoped to web-browsing agent task classes, not coding-agent or documentation-review task classes. The finding that design philosophy drives energy variation independently of outcome quality is the most directly applicable finding for the cycle's RQ-2: it is empirical evidence that performance-only optimization and energy-optimal design can produce different configurations.

*Broader reading (extrapolation requiring additional evidence):* If a 10× energy variation without corresponding performance variation holds for agentic coding tasks, the four-priorities frame's environmental cost axis would produce frame-divergent recommendations on agent selection. This extrapolation requires measurement in the cycle's task class.

**Honest absence note:** No published benchmark directly measures environmental cost of multi-turn coding agent sessions at the small-model local-inference scale the cycle operates at. The closest adjacent work (Jegham et al.'s 30-model cloud inference benchmark; Krupp et al.'s web agent benchmark) is directionally informative but not directly applicable. Environmental cost reporting in agentic coding benchmarks is a gap in the published literature.

#### 3.2.3 Local-First / On-Device Agent Architectures

**Edge-First Language Model Inference (arXiv:2505.16508, Jang & Morabito, May 2025, IEEE ICDCS 2025).** Benchmarks SLM capabilities on edge devices (Jetson AGX Orin, Jetson Nano Orin, smartphone) with key metrics: token generation speed (TGS), time-to-first-token (TTFT), power usage, energy per query. Identifies scenarios where edge inference offers comparable performance with lower costs, and others where cloud fallback is essential due to scalability or capacity limits.

*Narrow reading:* This is scoped to edge devices (IoT-class hardware), not consumer desktop hardware. The findings on TGS and TTFT on these constrained devices will be worse than consumer desktop GPU numbers. The paper confirms that edge inference has identified scenarios of viability and scenarios requiring cloud fallback — it does not claim universal viability.

**Hybrid Edge-Cloud Energy Savings (arXiv:2501.14823, Narayanan et al., January 2025).** Analyzes hybrid edge-cloud configurations for agentic workloads. Key finding: processing 80% of agentic workloads locally achieves up to 75% energy savings and >80% cost reduction compared to pure cloud processing.

*Narrow reading:* "Agentic workloads" in this paper refers to IoT/autonomous-systems agent contexts (autonomous vehicles, drones, robots) rather than LLM coding-agent sessions. The energy model uses 5 kWh/GB transmission energy + 1.5 kWh/GB cloud compute as input parameters. The finding's transferability to llm-orc's hybrid deployment shape depends on whether the same energy accounting applies to LLM inference workloads, which the paper does not directly test.

*Broader reading (extrapolation requiring additional evidence):* If the hybrid edge-cloud energy model transfers to LLM coding-agent sessions, it would provide empirical support for the cycle's four-priorities frame on environmental cost — specifically that hybrid configurations (cloud orchestrator + local ensembles) outperform pure cloud on environmental cost and financial cost while maintaining performance. The essay 002 finding (CAP-9 hybrid configuration validated empirically) is consistent with this direction.

*Alternative framing:* The local-first preference dimension of the four-priorities frame is a privacy/control argument, not only an energy argument. The published literature on local-first AI (dasroot.net 2026 analysis, Vitalik Buterin's April 2026 personal deployment account) consistently reports the tradeoff as capability-vs-sovereignty rather than capability-vs-energy. These are distinct framings that align on recommending local deployment but for different reasons — the cycle's frame names both without separating them, which may conflate two independent priority arguments.

#### 3.2.4 Token Cost as a Design Axis

**SkillReducer (arXiv:2603.29919, He et al., March 2026).** Large-scale empirical study of 55,315 publicly available agent skills found: 26.4% lack routing descriptions, over 60% of body content is non-actionable, reference files inject tens of thousands of tokens per invocation. SkillReducer achieves 48% description compression and 39% body compression while improving functional quality by 2.8%. Benefits transfer across five model families.

*Narrow reading:* This is scoped to skill/tool-library overhead, not to ensemble cascade overhead. The finding that most skill content is non-actionable is a token-cost argument for the "A2 + script input" shape over A3: the script's deterministic output is compact and fully actionable, whereas LLM reviewer outputs (especially R1-Hunyuan's chain-of-thought leakage in Cycle 2) represent high token-cost low-density output.

**SupervisorAgent / Stop Wasting Tokens (arXiv:2510.26585, Lu et al., ICLR 2026).** A lightweight runtime supervision framework that reduces token consumption of multi-agent systems by 29.68% on GAIA benchmark without accuracy loss. The LLM-free adaptive filter triggers interventions at critical junctures.

*Narrow reading:* This is scoped to existing multi-agent system overhead reduction, not to a comparison between single-agent and multi-agent token cost. The finding is that multi-agent overhead is reducible at runtime — which means the token-cost axis does not categorically favor single-agent over multi-agent; it depends on runtime optimization. Validation is on frontier models (GPT-4.1, Gemini-2.5-pro, Qwen3 series).

**What the literature operationalizes vs. what it treats rhetorically (per RQ-2's instruction):** The CLEAR framework operationalizes cost (token consumption, API costs, infrastructure overhead) as a measurable axis and produces a directly measurable finding (4.4–10.8× cost premium for accuracy-optimized agents). The cycle's four-priorities frame names token cost as a priority but does not operationalize it with the same measurement discipline CLEAR uses. The broader claim that "a multi-priority frame produces different recommendations than a performance-only frame" is supported by CLEAR's evidence; whether it applies specifically to the cycle's configurations requires scoring those configurations on all four axes, which is the spike's job.

**Scoring-resolution caveat (per P2 #2 from Step 1.4):** If all Cycle 3 configurations score qualitatively equivalent on environmental cost and local-first, this should be recorded as "no detectable divergence at qualitative resolution" not as "frames converge." The CLEAR evidence suggests cost-axis frame divergence is real at enterprise scale; whether it manifests at measurable resolution in the cycle's configurations is an empirical question, not a settled one.

---

### RQ-3 — Multi-Turn Reliability and Observable Failure Modes

**Question:** Does the tau-bench multi-turn tool-dispatching reliability ceiling generalize to llm-orc's deployment configurations? What agent-design choices reduce the rate of observable failure modes?

#### 3.3.1 Tau-bench Follow-up Work

**τ²-Bench (arXiv:2506.07982, Yao / Sierra Research, June 2025).** The direct follow-up to tau-bench (Yao et al. 2024). τ²-Bench introduces dual-control scenarios where both agent and user use tools in a shared, dynamic environment modeled as a Dec-POMDP. Domains: original retail + airline (single-control), plus new Telecom (dual-control). Key finding: pass@1 scores for GPT-4 drop from 74–56% in conventional single-control domains to **34% in dual-control telecom settings**. Multi-turn communication and guiding users (rather than just executing agent-side tools) are identified as the major bottlenecks.

*Narrow reading:* The 34% pass@1 finding is scoped to the dual-control telecom domain, which has higher coordination complexity than single-control. The 74–56% range in conventional domains represents the tau-bench 2024 finding replicated with updated models. The cycle's llm-orc agentic coding sessions are user-directed (the user provides direction across turns) but the agent does not need to guide the user through *tool use* — the coordination is lower than dual-control. The applicable ceiling for the cycle's configuration is closer to the 56–74% single-control range than to the 34% dual-control range. Neither ceiling applies directly, because both are customer-service task classes, not coding task classes.

*Broader reading (extrapolation requiring additional evidence):* The pattern that multi-turn reliability degrades as coordination complexity increases (single-control to dual-control: 74→34%) suggests that llm-orc sessions with complex multi-turn dependencies (e.g., sessions that require iterative tool-calling with mid-session state changes) will approach tau2-bench's lower bound rather than its upper bound. This is plausible but not measured.

**Trust-score-based failure reduction on tau2-bench (Cleanlab, 2025).** LLM trust scoring with fallback strategies reduces agent failure rates by up to 50% on tau2-bench. GPT-4o error rate reduction: up to 27%; GPT-4o mini: up to 34%.

*Narrow reading:* This is an instrumentation-layer intervention (hallucination detection + fallback), not an architectural intervention. The failure rate reduction is scoped to trust-score coverage — cases where the trust model correctly identifies low-confidence outputs. The finding does not address structural failure modes (meltdown onset, memory retrieval drift, premature stop) that are not detectable by output-level trust scoring.

#### 3.3.2 Multi-Turn Failure Mode Taxonomy Updates

**MAST Taxonomy (Cemri et al., arXiv:2503.13657, March 2025, NeurIPS workshop).** The MAST taxonomy identifies 14 failure modes across three categories (specification, inter-agent, task) in multi-agent LLM systems. Most relevant to RQ-3: FM-1.2 (violating role definitions), FM-1.4 (losing conversation context), FM-2.5 (ignoring other agents' input), FM-2.4 (withholding relevant information). These are role-boundary and context-loss failures that are multi-turn-specific in that they require accumulated state to manifest.

*Narrow reading:* MAST is framed around multi-agent systems. For single-agent multi-turn operation (the cycle's primary configuration), FM-1.4 (losing conversation context) is the directly applicable failure mode — it is the multi-turn equivalent of the mid-context attention valley finding from "Lost in the Middle" (Liu et al. 2024, cited in Cycle 2). FM-2.x modes do not apply to single-agent configurations.

**Microsoft Whitepaper on Failure Modes (April 2025).** Documents premature termination as a category: "ending a dialogue or task before all necessary information has been exchanged or objectives have been met." Multi-turn tasks that failed in their analysis were categorized as Technical (execution errors, malformed tool calls) and Instruction Following (premature termination). This taxonomy aligns with Cycle 2's premature-stop failure mode and adds the "malformed tool call" technical category as a distinct class.

**Cogent Orchestration Failure Playbook 2026.** Documents agent feedback loops where agents spiral into false consensus and exhaust API budgets: "When agents appear to agree, but the foundation of that agreement is false, agents may converge on a fabricated or misinterpreted data point simply to satisfy their completion objectives." This is the meltdown-via-false-consensus failure mode, distinct from meltdown-via-context-collapse. Proposed mitigations: explicit stop conditions (iteration limits, timeout thresholds), conflict resolution hierarchy, fallback escalation to human review.

*Narrow reading:* This "orchestration failure playbook" is practitioner analysis, not a controlled study. The false-consensus convergence mechanism is documented in the academic literature (Yao et al. 2025 meta-judge, Wynn et al. ICML MAS Workshop 2025, cited in Cycle 2) but the specific "exhaust API budgets" failure mode is a practitioner observation rather than a measured finding.

**AgentEval: DAG-Structured Evaluation (arXiv:2604.23581, ACL 2026 Industry Track).** A production evaluation framework that formalizes agent executions as DAGs with typed quality metrics and a 3-level, 21-subcategory failure taxonomy. Key findings:
- DAG-based dependency modeling contributes +22 percentage points to failure detection recall and +34 percentage points to root cause accuracy over flat step-level evaluation.
- Across 450 test cases: 2.17× higher failure detection recall than end-to-end evaluation (0.89 vs. 0.41), Cohen's κ=0.84 agreement with human experts, 72% root cause accuracy.
- 4-month pilot with 18 engineers: detected 23 pre-release regressions, reduced median root-cause identification time from 4.2 hours to 22 minutes.

*Narrow reading:* AgentEval is scoped to production agentic workflows (predominantly sequential with 12% non-DAG trace rate), not to multi-turn conversational agent sessions. The 2.17× failure detection improvement is for structured production workflows where DAG dependency modeling is applicable. For multi-turn conversational sessions with user-directed branching, DAG structure is harder to pre-define. The failure taxonomy's application to llm-orc's session shape would require mapping the DAG nodes to conversation turns, which is possible but not directly validated by this paper.

#### 3.3.3 Agent-Design Choices That Reduce Observable Failure Modes

**Routine: Structured Planning Scripts (arXiv:2507.14447, Xu et al., July 2025).** Routine is an explicit planning framework that uses structured JSON scripts (step number, action description, input, output, tool, node type) to guide LLM tool-calling. Evaluation in real-world enterprise scenarios: GPT-4o accuracy increased from 41.1% to 96.3%; Qwen3-14B from 32.6% to 83.3%. Fine-tuned Qwen3-14B on Routine-format data: 88.2%.

*Narrow reading:* This is scoped to enterprise multi-step tool-calling task classes (the type of task where step-ordering and tool selection are governed by domain process knowledge). The 41%→96% improvement represents the gain from structured planning over ad-hoc LLM planning, not from structured planning over no LLM. The finding applies to the failure mode of "disorganized plans, missing key tools, poor execution stability" — which maps to Cycle 2's premature-stop and unsupported failure modes. The scope condition that matters for the cycle: Routine's benefit comes from having a pre-defined correct execution sequence; for open-ended coding tasks where the execution sequence is not pre-defined, Routine's structured planning provides less value.

*Alternative framing:* Routine represents a script-as-orchestrator shape where the LLM's role is to follow a pre-compiled deterministic execution plan rather than to plan adaptively. The Compiled AI paper (arXiv:2604.05150) takes this further: "confining [the LLM] to a one-time compilation phase" so that "workflows execute deterministically without further model invocation." Both are partial instantiations of the script-as-orchestrator concept. They differ in the granularity of LLM involvement: Routine keeps the LLM in the execution loop but provides a structured plan; Compiled AI removes the LLM from execution entirely.

**SupervisorAgent (arXiv:2510.26585, ICLR 2026).** Runtime supervision that reduces token consumption by 29.68% on GAIA without accuracy loss, via LLM-free adaptive filter. The paper validates on Qwen3 series models, which are directly relevant to the cycle's deployment.

*Narrow reading:* The 29.68% token reduction is on GAIA, a general-purpose agent benchmark. The scope condition for the cycle is whether GAIA's task distribution resembles multi-turn coding sessions. GAIA tasks are general reasoning + web search + tool use; multi-turn coding sessions have different error-mode distributions (context accumulation, state management). The transfer is plausible but not directly measured.

**Honest absence note — Post-2026 multi-turn benchmarks:** The dispatch instruction asked for new post-2026 work on multi-turn agent benchmarks beyond Cycle 2's coverage (HORIZON, AMA-Bench, LongCLI-Bench, Khanal et al.). The literature search surfaced τ²-Bench (arXiv:2506.07982, June 2025) as the direct tau-bench follow-up and MAST as an updated failure taxonomy (March 2025). No post-2026 multi-turn benchmark comparable to HORIZON or LongCLI-Bench was found that would refute or substantially update their findings. The Khanal et al. reliability framework (arXiv:2603.29231, Cycle 2) has not accumulated replication studies in the search coverage; it remains the most complete reliability framework found. This absence is consistent with the recency of that work (published 2026) rather than with the research community having moved past it.

**Honest absence note — Multi-turn failure mode mitigations at small-model tier:** The mitigation literature (Routine, SupervisorAgent, trust scoring) is validated at frontier or near-frontier model tiers. No published study was found that measures meltdown-onset mitigation, premature-stop mitigation, or error-self-conditioning mitigation specifically at the qwen3:8b or small-model-local tier. The Khanal et al. finding that frontier models show *higher* variance (meltdown rates up to 19%) than weaker models because ambitious plans spiral — while weaker models fail earlier and less catastrophically — is the only tier-comparative failure-mode finding in the Cycle 2 corpus, and it was not replicated or extended in the new literature found.

---

### Cross-Cutting — Outcome-Framing Literature

**What does the agent literature evaluate, and how does it operationalize "outcome"?**

#### 3.4.1 Competing Operationalizations

The literature has at least three distinct operationalizations of "outcome" for agentic systems, and they produce different evaluation verdicts on the same agent behavior:

**Operationalization 1: Binary task completion.** The majority of published benchmarks (tau-bench, HORIZON, LongCLI-Bench, GAIA, AgentBench) operationalize outcome as pass/fail on the final state relative to a goal specification. Pass rate, pass@k, and task success rate are the dominant metrics. This is the performance-only operationalization.

**Operationalization 2: Procedure-aware evaluation (PAE) — Cao et al., arXiv:2603.03116.** PAE formalizes agent procedures as structured observations and applies multi-dimensional gating along Utility, Efficiency, Interaction Quality, and Procedural Integrity. The key concept is "corrupt success": an agent that completes a task by bypassing authorization, fabricating confirmations, or communicating incorrect policy is *categorically disqualified* rather than scored equivalently to an agent that followed correct procedures. Evaluation on tau-bench yields different rankings than binary task completion. PAE operationalizes the difference between "the task got done" and "the task got done correctly and safely."

*Narrow reading:* PAE is scoped to customer-service agent tasks where procedure correctness (authorization, policy compliance) is verifiable. The "corrupt success" concept has different application scope for coding agents — a coding agent that produces correct output via shortcut paths (e.g., hardcoding expected values rather than implementing the algorithm) is the analogous failure mode, but verification is harder. The procedural integrity dimension of PAE's framework has no direct analogue in the cycle's README-review task class unless the review process itself has a normative sequence that can be audited.

**Operationalization 3: Multi-dimensional with explicit cost-axis (CLEAR — arXiv:2511.14136).** CLEAR gates reliability (pass@8 ≥ 80% for mission-critical), cost (cost-normalized accuracy), latency (SLA compliance), and safety (policy adherence score) simultaneously. An agent that achieves high accuracy but fails the cost gate or reliability gate is rated lower than an agent with slightly lower accuracy but better multi-dimensional performance.

*Narrow reading:* CLEAR's reliability gate (pass@8 ≥ 80%) is calibrated for mission-critical enterprise applications. The cycle's research use case does not have a mission-critical SLA. The cost-axis finding (4.4–10.8× premium for accuracy-only optimization) is the most directly applicable finding for the cycle's four-priorities frame test.

**Beyond Task Completion Framework (arXiv:2512.12791).** Proposes a four-pillar evaluation (LLMs, Memory, Tools, Environment) that combines static, dynamic, and judge evaluation modes. The framework was validated on a cloud operations use case. Key finding: "while baseline metrics report task completion, framework-specific assessments reveal substantial behavioral failures across all pillars, particularly in tool orchestration and memory retrieval."

*Narrow reading:* This is a framework paper without broad empirical comparative results across multiple systems; its primary contribution is the evaluation architecture. It is directionally consistent with PAE and CLEAR in identifying binary task completion as insufficient.

**Towards Outcome-Oriented, Task-Agnostic Evaluation (arXiv:2511.08242).** Argues for evaluation frameworks that measure agent outcomes independently of task specification, to avoid specification-gaming. The key tension identified: specification adherence (did the agent do what was specified?) vs. outcome quality (did the agent produce a good outcome?). These can diverge when agents find specification loopholes.

*Narrow reading:* This paper surfaces the "corrupt success" problem from a different angle. For the cycle's factual-grounding task class, the specification is "review the README and surface issues." An agent that reports only the issues the script already found (inheriting the script's output as if it were independent review) would be specification-adherent but outcome-corrupt. This is a concrete risk for the "A2 + script input" configuration if the orchestrator's response anchors heavily on the script output without independent LLM analysis.

#### 3.4.2 What the Outcome-Framing Literature Implies for RQ-1, RQ-2, RQ-3

**For RQ-1 ("equivalent factual grounding"):** The outcome-framing literature reveals that "equivalent" requires specifying: equivalent on what operationalization? Under binary task completion (did the review surface the correct issues?), A2 + script input might score equivalently to A3 if it finds the same set of issues. Under procedure-aware evaluation, the question is whether A2 independently analyzed the README or merely reformatted the script output. The distinction matters because "equivalent factual grounding" in the cycle's sense is closer to PAE's procedural integrity than to binary task completion. The cycle's spike should surface this distinction explicitly in its evaluation design.

**For RQ-2 (frame-divergent recommendation):** The CLEAR evidence that accuracy-optimized agents cost 4.4–10.8× more than cost-aware alternatives with comparable performance is the clearest published case of a multi-priority frame producing a frame-divergent recommendation. The four-priorities frame would select the cost-aware alternative; the performance-only frame would be indifferent to cost. This is the operationalized version of the frame-divergence test the cycle is designed to run.

**For RQ-3 (observable failure modes):** PAE's "corrupt success" concept suggests that some failure modes the binary-completion evaluation misses are not execution failures but procedural failures — the agent completed the surface task while the underlying process failed. For multi-turn reliability, this means meltdown-onset detection should track not just task completion rates but whether intermediate steps were correctly executed. AgentEval's 2.17× improvement in failure detection when switching from end-to-end to DAG-structured evaluation is the empirical quantification of this gap.

---

### Targeted Coverage — Script-as-Orchestrator Shape

**Coverage instruction from P2 #1 (Step 1.4):** For script-driven loops with LLM as a subordinate step (rather than LLM as the primary orchestrator), seek published evidence on (a) factual-grounding task classes and (b) multi-turn sustained work.

#### 3.5.1 Evidence on Factual-Grounding Task Classes

**Compiled AI (arXiv:2604.05150, April 2026).** The "compiled AI" paradigm explicitly confines the LLM to a one-time compilation phase, after which workflows execute deterministically without further model invocation. Applied to healthcare settings (where auditability is critical), the four-stage generation-and-validation pipeline produces production-ready code artifacts from LLM-generated drafts. AlphaCodium (cited therein) shows that verification-gated generation improves GPT-4 accuracy from 19% to 44% on code tasks. MetaGPT's Standard Operating Procedures are cited as a prior instantiation of the pattern.

*Narrow reading:* Compiled AI is scoped to code generation tasks where the compilation phase produces verifiable executable artifacts. The pattern's extension to documentation review (the cycle's task class) is not tested in the paper. However, the underlying principle — LLM as semantic compiler, deterministic code as executor — maps onto the cycle's script-slot design: the script is the deterministic executor, and the question is how much semantic analysis the LLM adds beyond formalizing the script's output.

**Routine (arXiv:2507.14447).** As described above: structured planning scripts used as intermediate representation between LLM planning and deterministic execution. The LLM's role is to follow the Routine's step sequence, not to improvise the process. This is the closest published architecture to "LLM as subordinate step in a script-driven loop" on multi-step tool-calling tasks. The 41%→96% accuracy improvement represents the gain from providing LLMs with pre-compiled execution scripts rather than requiring ad-hoc planning. Qwen3-14B specifically goes from 32.6% to 83.3%.

*Narrow reading:* Routine's task class is enterprise multi-step tool-calling where domain process knowledge is available and can be encoded as a step sequence. Documentation-review tasks are less structured — the "correct" review sequence is not pre-definable in the same way. The Routine framework's applicability to the cycle's task class is partial: the script-agent's deterministic checks (link validation, section presence, code block parsing) are Routine-compatible, but the downstream LLM analysis is open-ended.

**Workflow Survey: Deterministic vs LLM-Controlled Orchestration (arXiv:2603.22386).** Characterizes two poles of agentic workflow control: deterministic orchestration (fixed execution policy, explicit tool/validator stages, bounded retries) and LLM-controlled orchestration (adaptive action selection, branching, variable-length trajectories). The survey notes that MacNet elevates DAG-based collaboration to a first-class design object, and that OneFlow (arXiv:2601.12307) shows some multi-agent gains can be reproduced by single-agent simulation.

*Narrow reading:* This survey provides vocabulary for the script-as-orchestrator shape: it maps to "deterministic orchestration with bounded LLM nodes" in the survey's typology. The survey does not provide a head-to-head comparison between deterministic-backbone + LLM-node and LLM-as-primary-controller on factual-grounding tasks. It is vocabulary, not comparative evidence.

**Honest absence note:** No published paper was found that directly compares script-as-orchestrator shapes (where deterministic code runs most of the workflow and LLM analysis is a bounded step) against LLM-as-orchestrator shapes on documentation-review or README-analysis task classes. The closest evidence is:
- Compiled AI's code generation comparison (AlphaCodium's verification-gated accuracy improvement)
- Routine's structured planning improvement on enterprise tool-calling
- The cycle's own Spike A3 data showing the script-agent slot's deterministic output was actively used as anchor evidence by the LLM reviewers

None of these directly test "does a deterministic script + single LLM step match a deterministic script + heterogeneous LLM ensemble?"

#### 3.5.2 Evidence on Multi-Turn Sustained Work

**Honest absence note:** No published paper was found that directly measures script-as-orchestrator shapes (LLM as bounded subordinate step) on multi-turn sustained work in coding or documentation-review task classes. The closest adjacent evidence:

- Routine's structured plans reduce premature stop and disorganized execution on multi-step tool-calling tasks, which are adjacent to multi-turn reliability.
- The Information Fidelity paper (arXiv:2602.13320) finds that periodic re-grounding every ~9 steps controls error accumulation in sequential tool calls — this is architecturally analogous to a script that re-grounds the LLM's context at defined checkpoints.
- Compiled AI's "confine LLM to compilation phase" approach eliminates multi-turn LLM error accumulation by design — there is no multi-turn LLM involvement after compilation.

**Alternative framing:** The script-as-orchestrator literature implicitly argues that multi-turn reliability degrades because LLMs accumulate errors over successive planning steps. If the script handles execution and the LLM is invoked only for bounded analysis steps (each stateless or re-grounded), the multi-turn failure modes characterized by Khanal et al. (meltdown onset, error self-conditioning) may not apply in the same way. This is the strongest *theoretical* argument for the script-as-orchestrator shape on multi-turn reliability — but it has not been empirically validated in a multi-turn setting in the published literature found.

---

## 4. Cross-RQ Patterns

### Pattern 1: Script-Slot's Value Is Narrow but Distinctive

Across RQ-1, the script-as-orchestrator coverage, and the outcome-framing literature, a consistent pattern emerges: deterministic script execution produces a category of evidence (verified facts with zero hallucination probability) that LLM-only configurations structurally cannot produce. The value is narrow (it applies only to tasks where deterministic verification is possible and relevant) but within that narrow scope it is categorical rather than probabilistic.

The literature supports this pattern from multiple angles:
- AlphaCodium's verification-gated accuracy improvement (19%→44%)
- Information Fidelity's finding that tool outputs reduce distortion by 80% with semantic weighting
- Cycle 2's Spike A3 empirical observation that the script's verified link counts and undefined-profile bugs were used as anchors by LLM reviewers

The competing framing is that once the deterministic facts are injected as context, a single capable LLM can reason from them as well as a heterogeneous ensemble can — making the script-agent necessary but the ensemble unnecessary. RQ-1's spike is the test that resolves which framing holds on the cycle's specific task class.

### Pattern 2: Multi-Priority Evaluation Produces Frame-Divergent Recommendations at Enterprise Scale, with Uncertain Transfer to Small-Model Local Configurations

CLEAR's evidence (4.4–10.8× cost premium for accuracy-only optimized agents) establishes that multi-priority framing produces frame-divergent recommendations at enterprise deployment scale on frontier model tier. The cycle's four-priorities frame names equivalent priorities but operates at a different configuration scale. Whether the frame-divergence is detectable at qualitative resolution on the cycle's configurations is empirically open per the P2 #2 scoring-resolution caveat.

The environmental cost axis is the most asymmetrically measured: extensive cloud-inference benchmarks exist (Jegham et al.), but local-inference environmental accounting is largely absent from the peer-reviewed literature. The Narayanan et al. hybrid edge-cloud finding (75% energy savings) is the closest analogue but is scoped to IoT agentic workloads.

### Pattern 3: The Single-Agent Baseline Literature Has Converged on a Conditional Claim

The 2025–2026 literature on single-agent vs. multi-agent (OneFlow arXiv:2601.12307, Lee et al. arXiv:2601.04748, Jiang et al. April 2026 from Cycle 2) has converged on a conditional claim: single-agent matches homogeneous multi-agent workflows on most task classes tested, but heterogeneous multi-agent configurations remain the residual case that single-agent simulation cannot replicate. This is precisely A3's architecture (heterogeneous models from different families). The literature thus delineates the exact scope condition under which the "A2 + script input" configuration might fall short: if the value is in uncorrelated errors from heterogeneous families (not in deterministic fact injection), then single-agent simulation is insufficient.

### Pattern 4: Outcome Operationalization Affects the Spike's Evaluation Design

The PAE ("corrupt success") and outcome-oriented framing literature identifies that specification-adherent task completion can mask procedural failure. For the cycle's RQ-1 spike, this means the evaluation design must distinguish: (a) does A2 + script input find the same issues as A3? and (b) does A2 + script input find those issues via independent LLM analysis, or merely by restating the script's output in review language? The second question is the procedural integrity question, and it is not answered by issue-count comparison alone.

### Pattern 5: Multi-Turn Reliability Literature Is Effectively Post-dated

Tau-bench (2024), HORIZON (2026), LongCLI-Bench (2026), Khanal et al. (2026) remain the primary multi-turn reliability findings. τ²-Bench extends tau-bench to dual-control settings and lowers the reliability ceiling further (34% pass@1 in dual-control). No post-2026 multi-turn benchmark substantially refutes the Cycle 2 findings. The trajectory is stable: long-horizon performance degrades super-linearly, frontier models show higher variance than weaker models at long horizons, and the ceiling for any configuration tested to date is below 75% on multi-turn tool-agent tasks at frontier tier.

---

## 5. Coverage Limits and Known Gaps

**Literature found but not deeply fetched:** The τ²-Bench paper (arXiv:2506.07982) is the most important single new source. The abstract-level summary is sufficient for the narrow reading; the specific pass rate numbers (74–56% single-control, 34% dual-control) are confirmed from multiple search results. The exact model-by-model breakdown was not extracted.

**Documentation-review-specific ensemble literature:** No published benchmark compares deterministic-script + single LLM vs. deterministic-script + heterogeneous LLM ensemble on a documentation-review task class. This gap is the specific empirical territory of RQ-1's spike.

**Environmental cost at small-model local inference scale:** No peer-reviewed paper measures environmental cost of LLM coding-agent sessions on consumer hardware running small models locally. The existing environmental benchmarks cover cloud inference at enterprise scale. This is a genuine gap that the literature has not yet addressed.

**Multi-turn failure modes at qwen3:8b tier:** No published study directly measures meltdown-onset, premature-stop, or error-self-conditioning rates at the qwen3:8b tier. The Khanal et al. finding (frontier models show higher variance) is the only tier-comparative finding in the reviewed corpus, and its direction (frontier models more meltdown-prone than weaker models) is counterintuitive and has not been replicated.

**Script-as-orchestrator multi-turn evidence base:** As reported above, this shape has essentially no published evidence base on multi-turn sustained work at the cycle's task class. The script-as-orchestrator literature (Routine, Compiled AI) focuses on multi-step single-session tool calling, not multi-turn conversational sustained work. The gap is honest and significant.

**Post-2026 multi-turn benchmarks:** No new benchmark post-2026 substantially extends HORIZON, AMA-Bench, LongCLI-Bench, or Khanal et al. with different findings. The absence may reflect publishing lag rather than absence of ongoing work.

**Honest note on search scope limitations:** Web search coverage of preprint servers after 2026 may not capture the most recently posted work. Papers posted after the search date cannot be found. The search was systematic within its coverage window (2024–May 2026) but cannot claim completeness.

---

## 6. Implications for Cycle 3's Spike Battery (Advisory)

**What the literature settles:**

- Single-agent simulation matches homogeneous multi-agent on most task classes tested (OneFlow). The cycle's spike should therefore focus specifically on the *heterogeneous* topology (A3's actual design) rather than homogeneous multi-agent as a strawman.
- Deterministic tool outputs improve LLM factual grounding (arXiv:2510.15955). The "A2 + script input" configuration should show improvement over A2 alone; the question is whether it matches A3.
- Multi-objective evaluation (CLEAR) produces frame-divergent recommendations on cost at enterprise scale. The cycle should score RQ-2 configurations on at least the cost axis with measured data, not only qualitative assessment.
- τ²-Bench's 34% pass@1 in dual-control settings confirms that multi-turn reliability ceilings remain low, extending the tau-bench 2024 finding to a more challenging regime.

**What requires empirical work:**

- Whether "A2 + script input" produces equivalent factual grounding to A3 on the README-review fixture (RQ-1's spike — not settled by the literature).
- Whether the four-priorities frame produces detectable frame-divergence at the cycle's configuration scale (RQ-2's scoring — not settled by the literature, only by direct measurement).
- Whether multi-turn failure modes (meltdown onset, premature stop, error self-conditioning) materialize at the cycle's llm-orc deployment configurations and at what rates (RQ-3's spike — not settled; tau-bench/tau2-bench are customer-service task classes, not coding task classes).

**Spike design refinements the literature suggests:**

- For RQ-1's spike: the evaluation design should explicitly distinguish issue-count parity from procedural independence (per PAE's corrupt success concept). An A2 + script input configuration that scores well on issue count but achieves it by restating the script output in review language is a different result from one that performs independent LLM analysis.
- For RQ-1's spike: the evaluation should probe whether the undefined-model-profile bugs (which Reviewer 1 / Hunyuan found across trials in A3, and which A2 missed) are findable by a single strong orchestrator with the script's structural output alone, or whether they require the heterogeneous reviewer's angle.
- For RQ-3's spike: the failure mode detection should track at minimum meltdown onset (context-collapse looping), premature stop (incomplete turn), and error self-conditioning (LLM conditioning on a prior incorrect tool dispatch) per the P2 #3 reformulation. The tau2-bench finding that "multi-turn communication and user guidance are major bottlenecks" in dual-control settings is a reminder that the cycle's spike should capture user-direction failures (the user asking for a follow-up that the agent fails to track) as well as agent-side tool-dispatch failures.
- For RQ-2's spike: the cost-axis measurement should follow CLEAR's cost-normalized accuracy (CNA) pattern: report cost per unit of outcome quality, not cost in isolation. This gives the four-priorities frame a measurable operationalization on the cost axis and allows direct comparison with the performance-only frame's recommendation.

**The script-as-orchestrator shape's advisory status:** The literature search found no published multi-turn evidence base for the script-as-orchestrator shape (LLM as bounded step). This does not mean the shape is unviable — it means the cycle would be producing novel empirical evidence if it tests this shape on multi-turn coding tasks. The Compiled AI and Routine results on single-session tool-calling tasks are directionally encouraging (substantial reliability improvements from deterministic structure). The cycle should treat the script-as-orchestrator shape as an empirically motivated hypothesis for multi-turn reliability, not as a settled pattern.

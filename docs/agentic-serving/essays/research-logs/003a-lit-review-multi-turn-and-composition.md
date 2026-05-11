## Literature Review: Multi-Turn Agentic Dynamics, Composition Thresholds, and Orchestration Shapes

**Date:** 2026-04-29
**Method:** Systematic literature search (web search + primary source fetch)
**Research questions covered:** RQ-1, RQ-1a, RQ-2, RQ-2a, RQ-3, RQ-3c, RQ-4, RQ-5
**Cycle context:** llm-orc agentic-serving Cycle 2, following Cycle 1's essay 002 which validated prompt-steering over structural composition at qwen3:8b single-ask tier

---

### Sources Reviewed

| # | Author(s) | Title | Year | Venue | Relevance |
|---|-----------|-------|------|-------|-----------|
| 1 | Wang et al. (Xinyu Jessica Wang, Haoyue Bai, Yiyou Sun, et al.) | The Long-Horizon Task Mirage? Diagnosing Where and Why Agentic Systems Break | 2026 | arXiv:2604.11978 | RQ-1: horizon-dependent degradation benchmark (HORIZON) |
| 2 | Khanal, Tao, Zhou | Beyond pass@1: A Reliability Science Framework for Long-Horizon LLM Agents | 2026 | arXiv:2603.29231 | RQ-1: meltdown behavior, reliability decay curves |
| 3 | Zhao et al. (Yujie Zhao, Boqin Yuan, et al.) | AMA-Bench: Evaluating Long-Horizon Memory for Agentic Applications | 2026 | arXiv:2602.22769 (ICLR 2026 Memory Agent workshop) | RQ-1: memory compression failure, similarity-based retrieval loss |
| 4 | Feng et al. (Yukang Feng, Jianwen Sun, et al.) | LongCLI-Bench: A Preliminary Benchmark and Study for Long-horizon Agentic Programming in Command-Line Interfaces | 2026 | arXiv:2602.14337 | RQ-1/RQ-1a: CLI-specific long-horizon failure, <20% pass rate |
| 5 | Liu et al. (Jiacheng Liu, Xiaohan Zhao, Xinyi Shang, Zhiqiang Shen) | Dive into Claude Code: The Design Space of Today's and Future AI Agent Systems | 2026 | arXiv:2604.14228 | RQ-1/RQ-5: Claude Code session architecture, context compaction, observability gap |
| 6 | Roig (J.V. Roig) | How Do LLMs Fail In Agentic Scenarios? | 2025 | arXiv:2512.07497 | RQ-1a: four-archetype failure taxonomy |
| 7 | Guan et al. (Shengyue Guan, Jindong Wang, et al.) | Evaluating LLM-based Agents for Multi-Turn Conversations: A Survey | 2025 | arXiv:2503.22458 | RQ-1: multi-turn evaluation taxonomy (~250 sources) |
| 8 | Zhou et al. (Han Zhou, Xingchen Wan, et al.) | Multi-Agent Design: Optimizing Agents with Better Prompts and Topologies | 2025 | arXiv:2502.02533 (Google / Cambridge) | RQ-2/RQ-2a: MASS framework, prompt optimization vs topology |
| 9 | Chen et al. (Weize Chen, Jiarui Yuan, et al.) | Optima: Optimizing Effectiveness and Efficiency for LLM-Based Multi-Agent System | 2025 | ACL 2025 Findings (aclanthology.org/2025.findings-acl.601) | RQ-2/RQ-2a: 2.8x perf gain at <10% token cost on Llama 3 8B |
| 10 | Anthropic Engineering | How We Built Our Multi-Agent Research System | 2025 | anthropic.com/engineering (June 2025) | RQ-2/RQ-3c: 90.2% improvement on research eval, 15x token cost |
| 11 | Wang, Wang, Athiwaratkun, Zhang, Zou | Mixture-of-Agents Enhances Large Language Model Capabilities | 2024 | arXiv:2406.04692 | RQ-3: MoA architecture, 65.1% AlpacaEval 2.0 surpassing GPT-4o |
| 12 | Adimulam, Gupta, Kumar | The Orchestration of Multi-Agent Systems: Architectures, Protocols, and Enterprise Adoption | 2026 | arXiv:2601.13671 | RQ-3: five-pattern shape taxonomy, enterprise adoption data |
| 13 | Wu, Li, Li (Haolun Wu, Zhenkun Li, Lingyao Li) | Can LLM Agents Really Debate? A Controlled Study of Multi-Agent Debate in Logical Reasoning | 2025 | arXiv:2511.07784 | RQ-3: debate efficacy conditions, majority pressure suppression |
| 14 | Wynn, Satija, Hadfield | Talk Isn't Always Cheap: Understanding Failure Modes in Multi-Agent Debate | 2025 | ICML MAS Workshop 2025 (arXiv:2509.05396) | RQ-3: debate accuracy decrease, sycophancy mechanism |
| 15 | Cui, Fu, Zhang (Yu Cui, Hang Fu, Haibin Zhang) | FREE-MAD: Consensus-Free Multi-Agent Debate | 2025 | arXiv:2509.11035 | RQ-3: anti-conformity mechanism, 50% token reduction |
| 16 | Li, Gao, Wang (Keyu Li, Jin Gao, Dequan Wang) | Aligned Agents, Biased Swarm: Measuring Bias Amplification in Multi-Agent Systems | 2026 | ICLR 2026 (OpenReview: mo7u21GoQv) | RQ-3: trigger vulnerability, bias amplification evidence |
| 17 | Madigan et al. | Emergent Bias and Fairness in Multi-Agent Decision Systems | 2025 | arXiv:2512.16433 | RQ-3: collective bias not reducible to individual agents |
| 18 | Choi, Zhu, Li (Hyeong Kyu Choi, Xiaojin Zhu, Sharon Li) | When Identity Skews Debate: Anonymization for Bias-Reduced Multi-Agent Reasoning | 2025 | ACL 2026 Main (arXiv:2510.07517) | RQ-3: IBC metric, anonymization mitigation |
| 19 | Yang et al. (Wei Yang, Shixuan Li, et al.) | Auditing Multi-Agent LLM Reasoning Trees Outperforms Majority Vote and LLM-as-Judge | 2026 | arXiv:2602.09341 | RQ-3: AgentAuditor, ACPO training, 5% accuracy improvement |
| 20 | Rahman, Schranz | LLM-Powered Swarms: A New Frontier or a Conceptual Stretch? | 2025 | arXiv:2506.14496 (preprint, submitted to IEEE Intelligent Systems) | RQ-3: swarm shape reality check, 36,000x latency penalty |
| 21 | Yao, Shinn, Razavi, Narasimhan | tau-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains | 2024 | arXiv:2406.12045 | RQ-1/RQ-4: GPT-4o <50% task success, <25% pass^8 |
| 22 | Yao, Shinn, Yao (Shunyu Yao, Noah Shinn, Karthik Narasimhan) | ReAct: Synergizing Reasoning and Acting in Language Models | 2023 | ICLR 2023 (arXiv:2210.03629) | RQ-1: foundational ReAct loop, context growth characterization |
| 23 | Qwen Team (Alibaba Cloud) | Qwen3 Technical Report | 2025 | arXiv:2505.09388 | RQ-4: BFCL v3 scores, tool calling benchmarks by model tier |
| 24 | Dasroot / cordum.io / latenode community analysis | LLM Agent Frameworks: LangChain vs CrewAI vs AutoGen 2026 Comparison | 2026 | dasroot.net, cordum.io, latenode.com (engineering analysis) | RQ-4: LangGraph 5% overhead vs CrewAI 3x latency |
| 25 | Anthropic Engineering | Effective Context Engineering for AI Agents | 2025 | anthropic.com/engineering | RQ-1: context window management, information priority principles |
| 26 | Lin et al. (Xixun Lin, Yucheng Ning, et al.) | LLM-based Agents Suffer from Hallucinations: A Survey of Taxonomy, Methods, and Directions | 2025 | arXiv:2509.18970 | RQ-3: 18 triggering causes, workflow-stage taxonomy |
| 27 | Haolun Wu et al. | Revealing Political Bias in LLMs through Structured Multi-Agent Debate | 2025 | arXiv:2506.11825 | RQ-3: debate amplifies political bias in structured settings |
| 28 | Lu, Yuan et al. | Routing to the Expert: Efficient Reward-guided Ensemble of Large Language Models (ZOOTER) | 2024 | NAACL 2024 (aclanthology.org/2024.naacl-long.109) | RQ-3: semantic routing shape, reward-guided expert assignment |

---

### Synthesis by Research Question

---

#### RQ-1 — Multi-Turn Agentic Dynamics

**What sustained multi-turn work demands beyond single-ask trials**

The most consistent finding across multiple 2025–2026 benchmarks is that long-horizon agentic performance degrades non-linearly with task length, and that the mechanisms driving this degradation are qualitatively different from the failure modes that single-ask evaluation captures.

**State evolution and working memory.** Wang et al. (2026, arXiv:2604.11978 — HORIZON benchmark) tested GPT-5 variants and Claude models across 3,100+ trajectories in four domains and found horizon-dependent degradation patterns that short-horizon evaluation masks entirely. The benchmark introduces a "long-horizon task mirage": systems appear capable when evaluated on short and mid-horizon slices of the same task class, then break on the full horizon. The human-AI judge agreement (κ=0.84) provides methodological confidence in the failure attribution. Khanal et al. (2026, arXiv:2603.29231) introduce four metrics — Reliability Decay Curve (RDC), Variance Amplification Factor (VAF), Graceful Degradation Score (GDS), and Meltdown Onset Point (MOP) — and find that reliability degrades super-linearly with task complexity, not linearly. A notable counterintuitive finding: frontier models show *higher* performance variance at long horizons than weaker models, not lower, because ambitious multi-step strategies "spiral" rather than degrade gracefully; meltdown rates reach 19% even in frontier models.

**Memory compression and information loss.** AMA-Bench (Zhao et al., 2026, arXiv:2602.22769, accepted ICLR 2026) directly measures what lossy memory compression does to agentic performance. The benchmark finds that existing memory systems fail primarily because they lack causal structure and are constrained by similarity-based retrieval — a retrieval mechanism that finds semantically proximate content but cannot recover causally necessary information that was not semantically salient when it entered memory. The best-performing memory system (AMA-Agent with a causality graph) achieves 57.22% accuracy, surpassing baselines by 11.16 percentage points. This means even the state-of-the-art memory architecture fails more than 40% of the time on real agentic trajectories.

Anthropic Engineering (2025, anthropic.com/engineering "Effective Context Engineering for AI Agents") frames context window management as a discipline — "context engineering" — and reports that attention prioritizes information at the beginning and end of the context window, with mid-context content regularly under-weighted regardless of its relevance. For the cycle's multi-turn work, this implies that tool dispatch results from earlier turns — precisely the accumulating state the orchestrator needs to reason about — are at elevated risk of attentional neglect as context grows.

A reported figure from production deployment data (Zylos Research, 2026) attributes 65% of enterprise AI multi-step reasoning failures to context drift or memory loss, not raw context exhaustion. The 2% misalignment early-to-40%-failure-rate-by-end of chain figure is from the same source and should be treated as an industry practitioner estimate rather than a controlled academic finding, but it is consistent with the academic benchmark results.

**Context window pressure from schema overhead.** The Claude Code architecture study (Liu et al., 2026, arXiv:2604.14228) explicitly identifies the context window as the "binding resource constraint" and describes a five-layer compaction pipeline (budget reduction → snip → microcompact → context collapse → auto-compact). The finding that 98.4% of Claude Code's codebase is deterministic infrastructure rather than AI decision logic reinforces what essay 002 found empirically: the engineering work in agentic systems is in the scaffolding around the loop, not the loop itself. For llm-orc, this is both validating and cautionary — the architecture's tool schema overhead (five internal tools plus however many client tools the client declares) compounds per-turn, and the HORIZON finding suggests this overhead interacts with error accumulation to produce the super-linear degradation pattern.

**Cross-turn judgment.** LongCLI-Bench (Feng et al., 2026, arXiv:2602.14337) benchmarks CLI-based agentic programming agents and finds that state-of-the-art agents achieve pass rates below 20% on long-horizon CLI tasks, and that the majority of tasks stall at less than 30% completion — meaning agents are not failing near the end of long tasks, they are failing early. This is consistent with the "Why Reasoning Fails to Plan" finding (arXiv:2601.22311, January 2026) that step-wise greedy agents face systematic failures because early deviations are amplified over time, not corrected. Self-correction provides only marginal improvement; human-agent collaboration through plan injection delivers significantly higher gains.

**Capability-tier sensitivity (RQ-1 cross-cutting note).** The HORIZON and LongCLI-Bench papers both test frontier models (GPT-5 variants, Claude models). The LongCLI-Bench finding of <20% pass rates at state-of-the-art represents the *ceiling* of known agentic performance on long-horizon CLI tasks — not the floor. For qwen3:8b-class orchestrators, the ceiling is lower. Essay 002's validated single-ask performance at qwen3:8b cannot be extrapolated to multi-turn without empirical validation, which is exactly what Cycle 2 is testing.

---

#### RQ-1a — Failure Mode Continuity: Single-Ask vs. Multi-Turn

The literature does distinguish multi-turn failure modes from single-ask failure modes, though the taxonomy is nascent and the boundary is not always sharp.

**Single-ask failures that persist and compound.** Roig (2025, arXiv:2512.07497) identifies four recurring failure archetypes in agentic scenarios: premature action without grounding, over-helpfulness substituting missing entities, vulnerability to distractor-induced context pollution, and fragile execution under load. The first and third of these correspond directly to failure modes essay 002 named (premature-stop and argument-confabulation respectively), and both become worse in multi-turn settings because each turn adds new potential distractor context and creates new grounding requirements.

Confabulation — essay 002's most insidious single-ask failure mode — appears in the multi-turn literature under the framing of "hallucinated tool outputs" (Khanal et al., 2026, meltdown behavior description). Khanal et al. document that the transition to incoherent behavior in meltdown involves "self-contradiction, and hallucinated tool outputs" — the same fast-confabulation pattern, now compounding across turns rather than appearing once.

**Multi-turn-specific failure modes.** Three modes appear in the literature that are plausibly multi-turn-specific rather than single-ask failures:

1. *Error self-conditioning.* "Measuring Long Horizon Execution in LLMs" (arXiv:2509.09677) identifies a mode where models condition on their own errors, degrading future performance. This requires multiple turns to emerge — a single-ask cannot self-condition on a prior error.

2. *Meltdown onset.* Khanal et al.'s Meltdown Onset Point (MOP) describes a threshold beyond which agents transition from "coherent but incorrect" to "incoherent looping." This is structurally a multi-turn phenomenon — a single-ask cannot loop.

3. *Memory retrieval drift.* AMA-Bench's finding that similarity-based retrieval loses causally necessary information over extended horizons is specific to systems that accumulate state across turns. Single-ask systems do not accumulate across-session memory.

**The fast-confabulation/premature-stop distinction in multi-turn context.** Essay 002 characterized mistral-nemo:12b's failure as fast-confabulation and qwen3:8b-without-bias as fast-giveup. The multi-turn literature suggests these patterns may interact differently with session length: fast-giveup (premature stop) in a single ask terminates the session; in a multi-turn session, the model might produce a partial response that appears to satisfy the turn, then fail to maintain the prior context on the next turn. Fast-confabulation in a multi-turn session potentially compounds, with each confabulated state being accepted as grounding for the next turn. No primary source directly validates this interaction; it is an inference from the self-conditioning literature that deserves empirical testing.

---

#### RQ-2 — Composition Threshold

**When does runtime ensemble composition earn its complexity?**

The literature does not offer a clean algebraic threshold. It converges on a set of qualitative conditions and some quantitative indicators.

**Anthropic's production-derived conditions.** Anthropic Engineering (June 2025) reports from their multi-agent research system: multi-agent deployment "requires three preconditions: task value justifies elevated token consumption; genuine parallelization opportunities exist; information scope exceeds practical single-agent limits." The token cost reality reported: agents consume 4x more tokens than chat interactions, and multi-agent systems consume 15x more tokens than chats. The 90.2% improvement on their internal research eval was achieved by using Claude Opus 4 as the lead agent with Claude Sonnet 4 subagents — a frontier-on-frontier configuration. **Capability tier caveat:** the improvement is attributed to parallel exploration of breadth-first queries that exceed single context window limits. If the task fits in a single context window and does not benefit from parallelism, Anthropic's own guidance is to use a single agent.

**OPTIMA / token-efficiency analysis.** Chen et al. (ACL 2025, aclanthology.org/2025.findings-acl.601) show that multi-agent systems trained with OPTIMA achieve up to 2.8x performance gain with less than 10% tokens on tasks requiring heavy information exchange, relative to vanilla multi-agent debate. The comparison baseline is multi-agent debate — not a single-agent baseline. On Llama 3 8B, the improvement over single-agent is "consistent and substantial" but the paper does not provide the raw single-vs-multi delta for the reader to directly read off. What OPTIMA demonstrates is that the coordination overhead of multi-agent systems is not fixed: it can be dramatically reduced with training.

**MASS framework (Zhou et al., 2025, arXiv:2502.02533).** On Gemini 1.5 Pro across eight tasks: single-agent CoT 65.28%; self-consistency 68.18%; multi-agent debate 70.26%; MASS (optimized prompts + topology) 78.79%. The ~13-point delta between single-agent CoT and optimized multi-agent is real, but the ~6-point delta from single-agent CoT to prompt-optimized single-agent (before topology) is also real and is achieved without composition overhead. Critically, MASS finds that "not all topologies have a positive influence on MAS design" — most topologies are neutral or harmful; only a small subset of configurations produce the performance improvement. This implies that choosing the wrong composition topology is worse than not composing at all.

**The economic breakeven framing.** An engineering analysis (Iterathon, 2026) identifies a 50,000 invoice/month breakeven where multi-agent's parallel processing justifies orchestration overhead. Below that threshold, a single agent with better prompt engineering delivered 92% of results at 28% of cost. The specific numbers are from a business process context and do not map directly to coding workflow orchestration, but the structure of the finding — composition earns its complexity at scale thresholds that most deployments do not reach — is consistent across sources.

**For the cycle's runtime composition question specifically.** The practitioner question is about composing *new* ensembles at runtime rather than selecting from pre-defined ones. No primary source directly benchmarks runtime-generated vs. pre-defined ensemble performance. The closest adjacent finding is from the MASS analysis: because topology selection is consequential and most topologies are neutral or harmful, runtime composition without robust topology quality validation is likely to be less reliable than pre-defined composition. This is a gap in the literature with direct bearing on Cycle 2's `compose_ensemble` tool testing.

---

#### RQ-2a — Prompt-Steering Baseline

**Does prompt steering of a capable single orchestrator produce comparable outcomes to structural composition?**

The literature gives a context-dependent answer that is broadly consistent with essay 002's CAP-2 finding, but with important qualification ranges.

**Zhou et al. (MASS, 2025)** provide the clearest head-to-head comparison at Gemini 1.5 Pro tier. Prompt optimization alone (without topology change) yields ~6 percentage points over default single-agent. Adding topology on top adds another ~5 percentage points. The structural contribution is real but smaller than the prompt contribution, and it requires correct topology selection to be realized. On task types where prompt optimization already saturates performance, structural addition produces no marginal gain.

**Anthropic Engineering (2025)** states explicitly: "Designing good prompts turned out to be the single most important way to guide how the agents behaved." This is from post-hoc reflection on their multi-agent research system, which *does* use structural composition — but even in that context, prompt quality was the dominant success factor, not topology.

**The "92% at 28% cost" finding** (from the Iterathon 2026 engineering analysis, attributing source to unnamed production deployment data) suggests that single-agent with better prompting delivers most of the multi-agent benefit at a fraction of the cost, for most task classes. This is consistent with essay 002's CAP-2 finding: the composition overhead was ~2.5x latency for essentially equivalent outcome.

**Where the comparison does not hold: task class dependency.** Both OPTIMA and Anthropic's research system show that for tasks requiring *breadth-first parallel exploration* across information sources that collectively exceed any single context window, structural multi-agent composition produces improvements that prompt steering cannot replicate — because the fundamental constraint is not prompt quality but information capacity. Essay 002 tested a capability-query task class that fits comfortably in one context window; Cycle 2 is testing multi-turn sessions that may not. If Cycle 2's sessions involve breadth-first exploration or information aggregation across sources that exceed a single context window, the prompt-steering-wins conclusion from essay 002 may not generalize. This is a key empirical question for the cycle's spike program.

---

#### RQ-3 — Shape Inventory, Failure Modes, Mitigations

**Shape inventory**

The literature converges on five primary coordination patterns with several sub-variants:

**1. Single orchestrator (current llm-orc baseline).** Adimulam et al. (2026, arXiv:2601.13671) name this the "Orchestrator-Worker" pattern with a single coordinator as "centralized control with fan-out." This is the simplest and lowest overhead shape. Microsoft's AI agent design patterns guide (cited in search results) recommends starting centralized and decentralizing only when concrete scalability bottlenecks are found. Most production teams never need full decentralization.

**2. Hierarchical (supervisor + worker pools).** Adimulam et al. identify hierarchical tree-structured delegation as a distinct pattern. The Anthropic research system is an instance: Claude Opus 4 as lead agent, Claude Sonnet 4 subagents, spawned dynamically per query type. The Addyosmani.com analysis of coding agent orchestration describes "higher-capability orchestrator produces a plan and assigns steps to many Haiku workers" as a practical instance — essentially a cost-tiered hierarchy where expensive frontier models plan and cheap models execute. This shape earns its overhead when tasks genuinely decompose into parallel subproblems.

**3. Swarm-of-small-models.** This is where the literature provides the clearest negative result for the practitioner's hypothesis. Rahman and Schranz (2025, arXiv:2506.14496) measure LLM-based swarms against classical swarm algorithms and find ~36,000x latency penalty for Boids simulation (0.0019s classical vs 68.61s LLM-based) and ~10x for ACO. Their verdict: "LLM-based systems are not swarms in the classical sense." True decentralization, real-time scalability, and genuine emergence from simple local rules are lost. They recommend hybrid approaches where LLMs handle strategic reasoning and classical algorithms manage low-level control. For llm-orc's context: running many concurrent small models is plausibly viable for latency if the models are small enough (Qwen3 0.6B/1.7B/4B can run on consumer hardware at 5–8 tok/s CPU or 40–60+ tok/s GPU), but the coordination overhead of LLM-mediated swarm communication is itself a bottleneck that cannot be eliminated through faster hardware.

**4. Semantic-routed ensembles (router selects expert by query semantics).** ZOOTER (Lu et al., NAACL 2024) represents the academic lineage: reward-guided routing distributes queries to the LLM with documented expertise in the relevant domain. The vllm-project/semantic-router (noted in search results) is an active engineering artifact implementing this at the serving layer. This shape earns its complexity when the query distribution is genuinely heterogeneous and individual expert models outperform generalist models on their specialty domains. For llm-orc, this is the closest architectural analogue to `invoke_ensemble` routing — the ensemble system already implements a form of semantic routing by allowing the orchestrator to select from named, purpose-built ensembles.

**5. Peer-to-peer and biologically-inspired structures.** The literature treats these as distinct concepts that partially overlap with LLM agent systems.

- *Multi-agent debate lineage (Du et al. 2023 and successors).* The canonical Du et al. 2023 finding — iterative debate mitigates hallucination via convergence to consensus — is now significantly qualified by the 2025–2026 literature (see Failure Modes below). The debate topology is real and extensively studied, but its benefits are conditional.

- *Swarm intelligence.* SwarmSys (arXiv:2510.10047) proposes pheromone-like traces for self-organized LLM agent collaboration. SwarmBench (arXiv:2505.04364) benchmarks LLM swarm coordination under decentralized conditions and finds task-dependent performance with "rudimentary coordination but struggling with long-range planning and adaptive strategy formation." The practitioner's eusocial/ant-colony/naked-mole-rat framing as candidate shapes: the literature does not use those specific terms for LLM architectures. The closest academic analogues are stigmergy (indirect coordination via environmental traces, operationalized in digital systems as shared memory stores, task queues, or vector databases), ant colony optimization (ACO, studied as a classical algorithm that LLMs can simulate but at massive latency cost), and market-based coordination (token-incentive mechanisms studied in blockchain-integrated swarm AI contexts, 2024–2025). Honest reporting: published papers using specifically "naked mole rat" or "eusocial" as LLM architecture metaphors were not found. The underlying coordination properties — division of labor, sterile-worker specialization, shared goal without explicit instruction — are discussed in the stigmergy and swarm intelligence literature at the level of abstract principles rather than implemented architectures.

**6. Mixture-of-Agents at the orchestration layer.** Wang et al. (2024, arXiv:2406.04692) introduced Mixture-of-Agents (MoA) as a layered architecture where each layer comprises multiple LLMs that each see all prior layer outputs. The AlpacaEval 2.0 result — 65.1% win rate versus GPT-4o's 57.5% using only open-source models — is the headline finding. This is the orchestration-layer analogue to token-level MoE in transformer models. The key distinction: model-level MoE (e.g., Mixtral, Qwen-MoE) activates a sparse subset of parameter experts per token at inference time within a single model; orchestration-layer MoA calls separate full model instances per layer, with full per-model inference cost. For llm-orc, MoA at the orchestration layer would mean chaining ensemble calls through multiple model passes — closer to what the current architecture supports via `invoke_ensemble` chains than to anything requiring new primitives.

**7. Ensembles-of-ensembles (recursive composition).** Not found as a named research area in the 2024–2026 literature in those terms. The closest analogue is the hierarchical multi-agent pattern applied recursively — supervisors that are themselves supervised — which appears in AgentOrchestra (arXiv:2506.12508) as a general-purpose hierarchical framework. This shape adds coordination overhead at each nesting level.

**8. Multi-agent debate.** Extensively covered below under Failure Modes and Mitigations.

---

**Failure modes**

The practitioner recalled prior research indicating multi-agent structures can reinforce bias and hallucination. The 2025–2026 literature validates this recall, quantifies it more precisely, and adds important nuance about conditions.

**Bias amplification in structured multi-agent workflows.** Li, Gao, Wang (ICLR 2026) is the strongest recent empirical finding: "structured workflows act as echo chambers, amplifying minor stochastic biases into systemic polarization" even "when isolated agents operate neutrally." Their Trigger Vulnerability finding — injecting purely objective context *accelerates* polarization rather than moderating it — is counterintuitive and has direct implications for architectures that use grounding context. The Discrim-Eval-Open benchmark measures this across different multi-agent topologies.

**Emergent collective bias not reducible to individual components.** Madigan et al. (2025, arXiv:2512.16433) argue from financial-domain simulations that multi-agent decision systems exhibit bias patterns that cannot be attributed to any individual agent — the bias is genuinely emergent from communication structure. This means per-agent bias auditing is insufficient; system-level evaluation is required.

**Multi-agent debate: the echo-chamber mechanism.** Wu et al. (2025, arXiv:2511.07784) find that majority pressure suppresses independent correction in debate settings — minority correct agents conform to majority incorrect consensus. Wynn et al. (ICML MAS Workshop 2025) demonstrate that "debate can lead to a decrease in accuracy over time — even in settings where stronger (i.e., more capable) models outnumber their weaker counterparts." The mechanism identified by Wynn et al. is sycophancy: models shift from correct to incorrect answers under peer pressure, favoring agreement over truth-seeking.

**Scope of the practitioner's recall.** The recall was accurate in direction — multi-agent structures can reinforce bias and hallucination — but understated the conditions. The literature shows this is *conditional* on topology (flat debate is more vulnerable than hierarchical structures with explicit judge roles), homogeneity (models sharing similar training amplify shared biases more), and task type (open-ended reasoning and political/normative judgments are more vulnerable than arithmetic and formal logic). The practitioner's recall should not be taken as ruling out multi-agent benefits; it should be read as: the failure mode is real, conditions matter, mitigation exists.

---

**Mitigations**

Five main mitigation strategies appear in the literature with varying evidence quality:

**1. Anonymization (Choi, Zhu, Li — ACL 2026).** Removing identity labels from debate transcripts prevents agents from distinguishing their own responses from peers', forcing equal weighting of evidence regardless of source. No model retraining required. ACL 2026 Main acceptance suggests methodological quality. The Identity Bias Coefficient (IBC) enables measuring the baseline problem, making this mitigation testable. This is the most deployment-friendly mitigation found — it is a prompt/context transformation, not an architectural change.

**2. Anti-conformity mechanisms (FREE-MAD, Cui et al., 2025).** Consensus-Free Multi-Agent Debate introduces explicit anti-conformity prompting — agents are instructed to resist majority positions — and a score-based decision mechanism that evaluates the full debate trajectory rather than the final round. Reported result: ~50% token reduction and improved factual accuracy. Limitation: the paper does not isolate the token reduction claim with controlled comparison; 50% vs. what baseline is unclear from the search results.

**3. Reasoning tree auditing with minority-correct training (AgentAuditor, Yang et al., 2026).** ACPO (Anti-Consensus Preference Optimization) trains an adjudicator specifically on cases where majority consensus was wrong, rewarding evidence-based minority selections. Reported result: up to 5% accuracy improvement over majority vote, up to 3% over LLM-as-judge. These gains are modest in absolute terms but represent improvements in the cases that matter most (where majority vote fails). The approach requires training a specialized adjudicator, which raises the deployment complexity.

**4. Diversity through heterogeneous model selection.** Wu et al. (2025) identify "intrinsic reasoning strength and group diversity" as the dominant success factors for effective debate. Homogeneous groups risk amplifying shared biases; diverse model families can genuinely challenge each other. For llm-orc's design space: using models from different training families rather than multiple instances of the same model as sub-agents would be the implementation of this mitigation. This is directly relevant to ADR-011's single-Model-Profile commitment — if debate is used, same-family models debate less effectively than cross-family models.

**5. Structured judge/role-asymmetry architectures.** RADAR (cited in search results) assigns explicit Security Criterion Auditor, Vulnerability Detector, Counterargument Critic, and Holistic Arbiter roles. The OPTIMA framework trains agents with differentiated communication roles. Meta-judge frameworks (arXiv:2504.17087) add a layer that evaluates the evaluators. These approaches increase architectural complexity but have documented residual failure modes: LLM judges remain vulnerable to choosing persuasive falsehoods, and the judge's own biases propagate upward through the evaluation chain.

**Residual failure modes after mitigation.** No mitigation eliminates the echo-chamber failure mode entirely. The AntiConsensus training approach (AgentAuditor) specifically targets the majority-failure case but introduces its own failure mode: if the minority is wrong, the trained adjudicator may select the wrong answer over the correct majority consensus. The fundamental tension identified by Wu et al. — "too much consensus and debate collapses into echo chambers; too much divergence and agents get misled by confidently incorrect outliers" — remains unresolved at the architectural level.

---

#### RQ-3c — Benefits-Side: Empirical Performance Benefits

**Under what task conditions do multi-agent structures produce reproducible improvements over single-agent baselines?**

The literature identifies four conditions under which multi-agent structures reliably outperform single-agent baselines:

**1. Tasks that exceed single context window limits (parallelism earns its cost).** Anthropic's research system (June 2025) demonstrates 90.2% improvement on breadth-first research queries where single-agent sequential processing would either fail (context limit) or be prohibitively slow (serial search). The improvement is specifically attributed to parallel information gathering, not to better reasoning per turn. For coding workflow orchestration: most single-task coding queries fit in one context window; multi-file, multi-component architectural reviews may not.

**2. Tasks with genuine subtask independence (parallel execution).** The Anthropic guidance is explicit: multi-agent fails for "tasks requiring shared context across all agents or extensive real-time coordination, such as most coding work." The implication for the practitioner's cycle: agentic coding tasks that require continuous context sharing across subtasks (debugging, refactoring that touches interconnected files) are poor candidates for parallelization benefits.

**3. Tasks with domain-specialized subtasks and available specialist models.** Mixture-of-Agents (Wang et al., 2024) achieves 65.1% AlpacaEval score using open-source specialist models in a layered architecture — above GPT-4o's 57.5%. The quality improvement comes from layered synthesis: each layer refines the prior layer's best outputs. This is specific to quality-improvement-through-aggregation; it cannot apply at llm-orc's consumer hardware scale without the frontier models that make the aggregation valuable.

**4. Optimization-oriented training (OPTIMA).** Chen et al. (ACL 2025) show 2.8x performance gain at <10% tokens on information-exchange tasks when multi-agent communication is optimized through fine-tuning (Llama 3 8B and 3.2 3B). This is not the out-of-the-box multi-agent result; it requires OPTIMA training. The baseline comparison for the 2.8x figure is "vanilla multi-agent systems," not single-agent — the improvement is above an unoptimized multi-agent baseline, not above a single-agent.

**Model-tier dependency note.** Every finding that shows strong multi-agent improvement uses frontier or near-frontier models in at least the orchestrator role. The OPTIMA exception (Llama 3 8B) is the closest to the cycle's operating tier, but even that paper requires fine-tuning to achieve the reported gains. The Anthropic research system finding (90.2% improvement) uses Claude Opus 4 as lead agent — well above qwen3:8b. Generalization of multi-agent performance benefits to qwen3:8b tier is not supported by the current literature. This is a genuine gap that Cycle 2's empirical work can address.

---

#### RQ-4 — Empirical Speed × Performance × Cost Frontier

The literature does not present a single clean Pareto frontier because most benchmarks do not measure all three axes simultaneously. What exists is a set of point estimates that sketch the shape of tradeoffs.

**Framework overhead baselines (LangGraph/AutoGen/CrewAI, 2026 community benchmarks).** Engineering analysis using GPT-4o as base model: LangGraph adds ~5% token overhead over raw model output at ~1.2s average latency for 10-step research pipelines. AutoGen matches LangGraph on latency and token use but generates 20+ LLM calls per task in conversational multi-agent mode. CrewAI consumed nearly 2x tokens and took 3x as long as LangGraph in the 2026 comparison benchmark. Essay 002's LangGraph "30% overhead" figure (cited from LangGraph documentation in Cycle 1) is at the high end; the 2026 independent benchmark suggests 5% overhead for graph-based orchestration specifically. The divergence may reflect LangGraph's evolution from earlier versions or different measurement methodology. For llm-orc: its custom orchestration layer does not use LangGraph; these figures are reference points, not direct measurements.

**Local-only vs. hybrid (essay 002 empirical data, carried forward).** Essay 002 established: local-only with qwen3:8b ~6 minutes wall-clock for single-ask; hybrid with MiniMax M2.5 Free cloud orchestrator ~62 seconds for same task. The 6x improvement is primarily from eliminating orchestrator-side per-turn inference time on consumer hardware. The literature on small model inference supports the structural reason: Qwen3 official documentation shows 5–8 tok/s CPU performance for 4B quantized models; at 15,000+ token contexts per turn, this translates to multi-minute latencies that cloud inference eliminates.

**Qwen3 small model inference.** The Qwen3 Technical Report (arXiv:2505.09388, May 2025) evaluates BFCL v3 for tool calling: Qwen3-235B-A22B achieves 70.8 on BFCL v3. Per model tier: Qwen3-8B scores 0.933 F1 in the Docker Engineering Blog evaluation cited in essay 002. The technical report includes ToolUse benchmarks for multi-turn and multi-step tool calling, with all Qwen3 models evaluated using FC format. The practitioner's intuition that multiple small models could run concurrently: the Qwen3-0.6B / 1.7B / 4B tier models can be run on CPUs for non-latency-sensitive applications. The concurrent throughput finding from production analysis shows a sweet spot at 8–12 concurrent requests balancing utilization and predictability. For the cycle's spike planning: a swarm of Qwen3-0.6B models under a single Qwen3-8B orchestrator would fit on consumer hardware, but the LLM-powered swarm literature (Rahman and Schranz, 2025) suggests the coordination overhead of LLM-mediated communication between them would dominate total latency.

**The local-only configuration's binding constraint.** Essay 002's 6-minute figure is not primarily a model quality issue — it is a token-throughput issue on consumer CPUs. The per-turn context (15,000+ tokens input) on hardware capable of 30–50 tok/s produces 300–500 second time-to-first-token for each orchestrator turn before any cascade occurs. This ceiling does not improve with smaller models unless the smaller models can handle the orchestrator's task at comparable quality — which the current literature on small model tool-calling performance (qwen3.5:9b's premature-stop, the qwen3:4b tier being untested) leaves empirically open.

---

#### RQ-5 — Conductor Experience

The published HCI/UX research on sustained agentic session experience is genuinely sparse. The search returned more from engineering blog analysis and industry practitioner surveys than from peer-reviewed HCI research. This is reported honestly.

**Latency tolerance.** The most directly applicable HCI finding: the arXiv paper "Mitigating Response Delays in Free-Form Conversations with LLM-powered Intelligent Virtual Agents" (arXiv:2507.22352) recommends targeting latencies under 4 seconds, noting that "failing to do so risks negatively biasing participants' perceptions of conversational agents and degrading overall system usability." This is a recommendation, not a hard threshold — the research it cites (Springer BISE "Opposing Effects of Response Time" article, multiple trials with novice vs. experienced users) shows that effect magnitude is user-class-dependent: experienced users tolerate longer latency more than novices.

A nuanced finding from Tandfonline (2025): moderate response latency combined with emotional support *heightens* chatbot evaluations — slower is not always worse if the latency is contextualized. This is relevant to the operator-side observability question from essay 002: typing indicators and progress signals change the experience of waiting, not just the waiting itself.

**The 2-second standard.** Multiple sources (2024 practitioner) converge on under-2-second as the standard for chatbot response. Essay 002's 6-minute local-only latency is approximately 180x the chatbot standard. The hybrid configuration's ~62 seconds is approximately 30x the standard. Both are outside interactive-acceptable territory for conversational interactions. For agentic work where the user accepts batch-like latency in exchange for autonomous execution, the threshold shifts — but the literature does not provide empirically derived "acceptable" latency for agentic coding sessions specifically.

**User intervention patterns.** The primary finding from Cycle 1's PLAY phase — that user passivity in the face of silent failure is the dominant experiential mode — has partial support in the engineering literature. Microsoft's AI Taxonomy of Failure Mode Whitepaper (cited in search results) and Partnership on AI's "Prioritizing Real-Time Failure Detection in AI Agents" report (September 2025) both identify lack of transparency and inadequate intervention points as top-tier adoption barriers. The enterprise survey (April 2026) finding — 94% of enterprises confident their DR plans cover agentic AI, but only 32% test plans monthly — suggests the intervention problem is organizational as well as product-level: users are not equipped to intervene when agents fail.

**What users want to see.** The Anthropic context engineering work (2025) identifies "progressive disclosure" as the core observability principle: users should be able to see agent state at the level of detail they need, without being overwhelmed by full trace dumps. This matches the operator-side observability work in essay 002, which found that INFO-level dispatch logging was the minimal viable signal for operator diagnosis.

**Sustained session coherence.** Devin 2.0 (Cognition, 2026) describes fork and rollback features, async handoff capabilities, and confidence-based clarification requests as the mechanisms that keep sustained sessions coherent. The Claude Code architecture (Liu et al., 2026) describes seven graduated autonomy modes spanning plan to bypassPermissions and append-only session persistence via JSONL. Both represent the engineering reality of shipped products; published peer-reviewed HCI research studying user behavior in these modes was not found. The RedMonk analysis ("10 Things Developers Want from their Agentic IDEs in 2025") provides practitioner-voice data but is not peer-reviewed.

---

### Key Findings

- **Long-horizon failure is qualitatively different from single-ask failure.** Wang et al. (2026, HORIZON) and Khanal et al. (2026) both find that performance degrades super-linearly with task horizon, with meltdown patterns (incoherent looping) appearing at long horizons that have no single-ask analogue. Frontier models show meltdown rates up to 19% on long-horizon tasks. [Sources: arXiv:2604.11978, arXiv:2603.29231]

- **Memory compression is a primary mechanism for long-horizon failure, not just context length.** AMA-Bench shows that similarity-based retrieval loses causally necessary information even when the system has sufficient context capacity. State-of-the-art memory achieves only 57.22% accuracy on real agentic trajectories. [Source: arXiv:2602.22769]

- **On CLI-agentic coding tasks specifically, state-of-the-art agents achieve less than 20% pass rate, and most tasks stall at less than 30% completion.** Self-correction provides marginal improvement; human plan injection provides significant improvement. [Source: arXiv:2602.14337]

- **Prompt optimization delivers ~6 percentage points over default single-agent before topology matters; topology adds ~5 more percentage points but only when the correct topology is chosen.** Not all topologies are beneficial; most are neutral or harmful. [Source: arXiv:2502.02533]

- **Multi-agent composition earns its complexity specifically for: breadth-first parallel exploration, tasks exceeding single context window, and domains with available specialist models.** Anthropic's production system shows 90.2% improvement for research queries with those properties — at frontier model tier, with 15x token cost. [Source: anthropic.com/engineering, June 2025]

- **Multi-agent systems reinforce bias under specific conditions.** The ICLR 2026 finding (Li et al.) shows that structured workflows amplify minor stochastic biases into systemic polarization even when individual agents are neutral, and that a "trigger vulnerability" causes objective context injection to accelerate rather than moderate polarization. [Source: ICLR 2026, OpenReview mo7u21GoQv]

- **Multi-agent debate can reduce accuracy, not just raise it.** Wynn et al. show accuracy decreasing over debate rounds when agents shift from correct to incorrect answers under peer pressure. The echo chamber effect is most pronounced with homogeneous model configurations. [Source: arXiv:2509.05396, ICML MAS Workshop 2025]

- **Anonymization (removing identity labels from debate prompts) is the most deployment-friendly mitigation for debate-based bias amplification.** It requires no retraining, works across model families, and has ACL 2026 Main acceptance as evidence of methodological quality. [Source: arXiv:2510.07517]

- **LLM-powered swarms are not swarms in the classical sense.** They lose the core properties of decentralization, real-time scalability, and emergent coordination while introducing LLM inference costs (36,000x latency penalty vs. classical algorithms for simple coordination tasks). [Source: arXiv:2506.14496]

- **Qwen3-8B achieves 0.933 F1 on tool-calling evaluation**, placing it above specialized tool-calling models (xLAM-8B at 0.570) in the same parameter class. The BFCL v3 score for Qwen3-235B-A22B is 70.8; smaller model tier scores are available in the Qwen3 Technical Report. [Sources: Docker Engineering Blog evaluation cited in essay 002; arXiv:2505.09388]

- **ADR-011 (single-Model-Profile orchestrator) is supported for single-ask tasks at qwen3:8b tier but the literature does not settle the question for multi-turn.** The MASS finding (Zhou et al.) shows structural composition adds 5 percentage points over prompt optimization at Gemini 1.5 Pro tier. Whether this delta materializes at qwen3:8b tier in multi-turn sessions is an open empirical question that Cycle 2 is positioned to answer. [Sources: arXiv:2502.02533, arXiv:2406.04692]

- **User latency tolerance for agentic work is not well characterized in peer-reviewed HCI literature.** The closest published guidance (arXiv:2507.22352) targets under 4 seconds for conversational AI; agentic coding session tolerance is not directly studied. [Source: arXiv:2507.22352]

---

### Limitations

**Model-tier gap.** The majority of empirical findings on multi-agent performance improvements use frontier models (GPT-4, Claude Opus 4, Gemini 1.5 Pro). The OPTIMA result on Llama 3 8B is the closest to the cycle's operating tier, but requires OPTIMA fine-tuning that the cycle's deployment does not have. Generalization claims from frontier-model experiments to qwen3:8b are not supported without additional empirical work — which the cycle is providing.

**Single-task vs. multi-turn data gap.** LongCLI-Bench and the HORIZON benchmark provide long-horizon evaluation data, but both measure task completion on isolated long tasks rather than multi-turn conversational sessions where the user provides direction across turns. The llm-orc agentic-serving scenario (user drives an interactive session across turns, not a single autonomous long-horizon task) may have different failure characteristics from either single-ask or fully-autonomous long-horizon benchmarks.

**Runtime composition evaluation gap.** No primary source benchmarks the performance of dynamically runtime-generated ensemble configurations versus pre-defined ensembles. The MASS and OPTIMA findings address topology selection and optimization, but within pre-defined topology spaces. Whether the `compose_ensemble` + `calibrate_composed_ensemble` path can produce configurations that rival pre-defined ensembles is not addressed by the literature reviewed.

**Consumer hardware concurrency gap.** The Qwen3 inference speed data shows 5–8 tok/s for CPU and 40–60+ tok/s for single GPU. The concurrent small-model hypothesis (multiple Qwen3-0.6B instances running concurrently as workers) has no direct published benchmarking at the configuration the cycle is considering. The LLM-powered swarm literature's ~10x latency penalty for ACO at 50 steps is the closest published analog, but ACO is a different coordination topology than orchestrator-with-small-workers.

**HCI literature gap.** Peer-reviewed HCI research on sustained agentic session user experience is sparse. Most evidence on user intervention, latency tolerance in agentic contexts, and session coherence perception comes from engineering blog posts, practitioner surveys, and product documentation rather than controlled studies. The PLAY-phase finding from Cycle 1 (user passivity as the dominant experiential mode when agents fail silently) is consistent with what engineering sources describe but lacks peer-reviewed validation.

**The practitioner-recall test on bias amplification.** The recall was directionally accurate. The literature does show that multi-agent structures can reinforce bias and hallucination. The literature also shows that this is conditional on topology, homogeneity, and task type — and that mitigation exists. The recall did not overstate the problem; if anything, the ICLR 2026 finding on Trigger Vulnerability suggests the risk is more subtle than the recall acknowledged (objective context injection can accelerate rather than mitigate bias).

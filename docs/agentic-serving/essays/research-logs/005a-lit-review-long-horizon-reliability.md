## Literature Review: Long-Horizon Agent Reliability, Judgment Decay, and Intervention-Class Taxonomy

**Date:** 2026-05-04
**Method:** Systematic literature search (web search + primary source fetch across arXiv, OpenReview, Anthropic Engineering, and practitioner sources)
**Cycle:** 4 (agentic-serving scoped corpus)
**Wave:** 1.A — Focus Questions 1–4 (long-horizon reliability, intervention taxonomy, coding-agent design, local-model capability ceiling)

---

### Sources Reviewed

| # | Author(s) | Title | Year | Venue | Relevance |
|---|-----------|-------|------|-------|-----------|
| 1 | Khanal, Tao, Zhou (Northern Kentucky University) | Beyond pass@1: A Reliability Science Framework for Long-Horizon LLM Agents | 2026 | arXiv:2603.29231 | FQ1, FQ2: reliability decay metrics, meltdown onset, memory scaffold failure |
| 2 | Wang, Bai, Sun, et al. | The Long-Horizon Task Mirage? Diagnosing Where and Why Agentic Systems Break | 2026 | arXiv:2604.11978 | FQ1: HORIZON benchmark, long-horizon failure attribution taxonomy |
| 3 | Zhang (Independent) | Harness as an Asset: Enforcing Determinism via the Convergent AI Agent Framework (CAAF) | 2026 | arXiv:2604.17025 | FQ2: deterministic harness enforcement, structural routing, state locking |
| 4 | Bui | Building AI Coding Agents for the Terminal: Scaffolding, Harness, Context Engineering (OpenDev) | 2026 | arXiv:2603.05344 | FQ2, FQ3: harness patterns, context compaction, lifecycle hooks, long-session design |
| 5 | Liu, Zhao, Shang, Shen | Dive into Claude Code: The Design Space of Today's and Future AI Agent Systems | 2026 | arXiv:2604.14228 | FQ3: Claude Code session model, five-layer compaction, recoverable failures |
| 6 | Pan, Zou, Guo, Ni, Zheng | Natural-Language Agent Harnesses | 2026 | arXiv:2603.25723 | FQ2: harness component taxonomy, lifecycle hooks, IHR enforcement |
| 7 | Du (survey lead) | Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers | 2026 | arXiv:2603.07670 | FQ2: five memory mechanism families, long-session tradeoffs |
| 8 | Zhou et al. | Externalization in LLM Agents: A Unified Review of Memory, Skills, Protocols and Harness Engineering | 2026 | arXiv:2604.08224 | FQ2: four-layer externalization taxonomy, when each layer is the right intervention surface |
| 9 | Zhou (Saarland University) | From Hallucination to Structure Snowballing: The Alignment Tax of Constrained Decoding | 2026 | arXiv:2604.06066 | FQ2: constrained decoding tradeoffs, structure snowballing failure mode, 8B model capacity limits |
| 10 | Thai et al. | SWE-EVO: Benchmarking Coding Agents in Long-Horizon Software Evolution Scenarios | 2025 | arXiv:2512.18470 | FQ3: 25% vs 72.8% gap, multi-file long-horizon coding agent failures |
| 11 | Deng, Da, Pan, et al. (Scale AI) | SWE-Bench Pro: Can AI Agents Solve Long-Horizon Software Engineering Tasks? | 2025 | arXiv:2509.16941 | FQ3: enterprise-difficulty coding tasks, failure mode taxonomy |
| 12 | Qwen Team (Alibaba) | Qwen3-Coder-Next Technical Report | 2026 | arXiv:2603.00729 | FQ4: local open-weight model agentic capability, SWE-bench performance, tool-calling reliability |
| 13 | Anthropic Engineering | Effective Harnesses for Long-Running Agents | 2026 | anthropic.com/engineering | FQ2, FQ3: two-agent harness pattern, session continuity, structured progress artifacts |
| 14 | Anthropic Engineering | Effective Context Engineering for AI Agents | 2025 | anthropic.com/engineering | FQ2, FQ3: context engineering principles, compaction, just-in-time retrieval |
| 15 | Liu, Nelson et al. (Stanford) | Lost in the Middle: How Language Models Use Long Contexts | 2023 | TACL | FQ1: U-shaped attention, mid-context performance degradation baseline |
| 16 | OpenReview submission | HELM: Steering Long-Horizon Agents with Learned Hierarchical Memory and Epistemic Governance | 2026 | OpenReview (ICLR 2026 track) | FQ2: three-tier SHNM memory, epistemic governance, provenance-aware retrieval |
| 17 | Shinn, Labash, Gopalan | Reflexion: Language Agents with Verbal Reinforcement Learning | 2023 | NeurIPS 2023 | FQ3: 91% pass@1 HumanEval via reflection, backtracking via episodic trace |
| 18 | dasroot.net (practitioner) | Local AI vs Claude Code: A Productivity Reality Check | 2026 | dasroot.net | FQ4: local model failure modes, tool calling hallucination, capability ceiling |
| 19 | Hall, Ben | Local AI Models for Coding: Is It Realistic in 2026? | 2025 | failingfast.io | FQ4: local model benchmark ceiling, VRAM requirements, task class boundaries |
| 20 | dasroot.net (practitioner) | Coding Agents on a Budget: Local Long-Context Workflows | 2026 | dasroot.net | FQ4: local model ceiling, hardware constraints, hybrid deployment |
| 21 | Chroma research (cited in practitioner sources) | Context rot benchmark across 18 frontier models | 2025 | Chroma blog | FQ1: context rot before context window overflow, degradation at 50K tokens |
| 22 | OpenHands team | The OpenHands Software Agent SDK | 2025 | arXiv:2511.03690 | FQ3: event-sourced state, fault recovery, no backtracking in baseline |
| 23 | MindStudio, Morph, WhatLLM (practitioner roundup) | Best Open-Source LLMs for Agentic Coding in 2026 | 2026 | Multiple practitioner sources | FQ4: leaderboard data, agentic coding task reliability by model tier |

### Sources from prior cycles (not duplicated; cited where Cycle 4 work extends or qualifies)

Cycle 3 `004a-lit-review-agent-design.md` reviewed MAST (Cemri et al. arXiv:2503.13657), tau2-bench (Yao/Sierra arXiv:2506.07982), CLEAR (Rony et al. arXiv:2511.14136), Khanal et al. 2026 (first pass), Routine (Xu et al. arXiv:2507.14447), Compiled AI (Batra et al. arXiv:2604.05150), and OpenFlow/Lee et al. Those sources are inherited and referenced below where Cycle 4 work qualifies or extends them.

---

### Synthesis

#### Focus Question 1: Long-horizon agent reliability and judgment decay under context growth

**The core empirical picture is settled, with one counterintuitive finding.**

Khanal, Tao, and Zhou (arXiv:2603.29231, 2026) establish the quantitative baseline for long-horizon reliability decay using 23,392 episodes across 10 open-source models. Their four metrics — Reliability Decay Curve (RDC), Variance Amplification Factor (VAF), Graceful Degradation Score (GDS), and Meltdown Onset Point (MOP) — are the most complete measurement framework found in this search. The headline finding: aggregate pass@1 drops from 76.3% on short tasks to 52.1% on very-long tasks — a 24.3 percentage-point decline that exceeds geometric predictions from independent error models, indicating positive error correlation across steps. Software engineering tasks collapse dramatically (GDS: 0.90 → 0.44), while document processing remains relatively flat (0.74 → 0.71). The domain-specificity of degradation is a signal for the cycle: coding-domain tasks are in the worst degradation class.

The counterintuitive finding: the MOP paradox. Frontier models exhibit the *highest* meltdown rates (DeepSeek V3: 19% at very-long; MiniMax M2.5: 13%), because they pursue ambitious multi-step strategies that generate entropy spikes when exploratory paths spiral. Weaker models fail earlier and more uniformly — they never reach the meltdown regime. Khanal et al. interpret high VAF (≥2.37 for frontier; ≤1.26 for mid-tier) as a capability signal rather than an instability signal: capable models succeed variably on long tasks; weak models fail uniformly. This framing complicates reliability evaluation for local-first deployments — the question is not simply "does the local model fail less?" but "does it fail in a cheaper failure mode?"

A critically relevant finding for Sub-Q6's transfer-test: Khanal et al. report that episodic memory augmentation *universally hurts* long-horizon performance across all 10 models tested. Six models show negative effects; four are neutral. The largest penalties fall on mid-tier capable models (Kimi K2.5: -0.14 GDS; Mistral 24B: -0.13). The stated mechanism is scratchpad overhead consuming critical step budget. This refutes the intuitive position that memory scaffolds should help long-horizon agents — under current implementations, they impose costs that exceed benefits. Whether this finding generalizes beyond the specific scaffolding designs tested is not established.

**The failure mode taxonomy and the primacy/recency bias literature.**

Liu et al.'s "Lost in the Middle" (TACL, 2023) remains the foundational source on positional bias, establishing that models attend strongly to beginning and end of context and poorly to the middle. Accuracy drops >30% when relevant information is in positions 5–15 versus position 1 or 20 in multi-document QA. This is a primacy and recency *combined* bias — not purely a recency effect. For instruction-following over long sessions, the practical implication is that instructions given at session start progressively enter the attention valley as the context grows. Chroma's 2025 benchmark of 18 frontier models extended this finding: context rot (the practitioner term for this degradation) manifests well before context window overflow — significant degradation observed at 50K tokens in a 200K-token window.

The "know but don't tell" phenomenon (cited in practitioner sources, referencing work from 2024) adds precision: models can accurately identify the position of critical information but fail to utilize it in response generation. This distinguishes the failure mode from pure information loss — the model has the information but the attention mechanism does not weight it at response time.

**Is RDD's "structural hooks at phase transitions" failure mode the same as continuous routing-judgment degradation?**

The cycle's scope-condition discipline (research log Step 1.2, Commitment 3) asks whether phase-transition failure and continuous-routing failure are the same class. The literature provides a partial answer: they share a common mechanism (context growth degrades the salience of a prior directive) but differ on *when* and *how often* the decision moment occurs.

Wang et al.'s HORIZON benchmark (arXiv:2604.11978, 2026) evaluates 3,100+ trajectories and introduces a trajectory-grounded failure attribution pipeline. The benchmark distinguishes failure modes by where in the trajectory they occur — early, mid, or late — but does not itself distinguish one-shot procedural decisions from continuous routing decisions as a taxonomic category. The failure attribution is task-agnostic with respect to decision type.

The CAAF paper (Zhang, arXiv:2604.17025, 2026) offers the clearest framing: it identifies "Context Rot" as a failure mode that "causes instruction salience to decay monotonically with context length" and explicitly proposes different interventions for different decision frequencies. Its Pillar 1 (context firewalls via Recursive Atomic Decomposition) addresses continuous routing by isolating each executor from irrelevant prior context. Its Pillar 3 (Structured Semantic Gradients + State Locking) addresses the convergence problem for *iterative* decisions — marking resolved constraints as `read_only: true` so the executor concentrates inference only on unresolved dimensions.

The distinction that holds after reviewing CAAF and the broader literature: phase transitions are *finitely occurring* decision moments where the harness can inject a deliberate re-grounding. Continuous routing decisions (like ensemble selection at every orchestrator step) occur at a *much higher frequency* and cannot be addressed by a hook that fires once per phase transition. The intervention-class implications differ accordingly. This is the literature's closest answer to the cycle's scope-condition question, though no paper directly studies "ensemble routing judgment degradation at high decision frequency" as a distinct phenomenon. That gap is the cycle's novel territory.

---

#### Focus Question 2: Intervention-class taxonomy for unreliable-agent decisions

Four intervention classes emerge from the literature. The literature is more developed on class (a) and class (c) than on (b) and (d).

**Class (a): Harnesses that wrap the agent's decision (deterministic override at decision points).**

The harness literature crystallized in 2025–2026 around a consistent set of components. Pan et al.'s Natural-Language Agent Harnesses (arXiv:2603.25723, 2026) identify six essential harness components: contracts, roles, stage structure, adapters and scripts (deterministic hooks for tests and parsing), state semantics, and failure taxonomy. The Intelligent Harness Runtime (IHR) enforces execution through explicit contracts at each stage, physically separating harness orchestration from LLM content generation. The paper's quantitative finding on SWE-bench (>110/125 instances agreed between full IHR and ablated versions) suggests harness changes concentrate on boundary cases rather than uniformly improving performance — which implies that harnesses are load-bearing for *specific hard-boundary decisions* rather than general capability improvement.

CAAF (Zhang, arXiv:2604.17025, 2026) provides the most rigorous quantitative case for class (a). Its State Locking mechanism forces monotonic non-regression (V_t ⊆ V_{t+1}) by marking validated constraints read_only, preventing the LLM from re-litigating settled decisions. Quantitative result: monolithic GPT-4o at temperature 0.0 achieves 0% paradox detection across 27/30 runs. CAAF-all-mini achieves 100% at cost $0.0027 per correct artifact. CAAF's most important finding for the cycle's Sub-Q6 framing: "apparent LLM reliability in safety-critical domains is often a prompt engineering artifact" — removing semantic hints collapses monolithic models from 90% to 0%, while CAAF maintains 100%. This is direct evidence that the failure mode being addressed is structural, not stochastic.

**Cost and tradeoffs for class (a):** Harnesses impose design overhead and require predefined decision topology (DAG or FSM structure). They work best when decision points are enumerable — a scope condition that applies to phase-transition enforcement but is harder to satisfy for continuous routing judgment at every orchestrator step. CAAF's framework requires domain invariants to be "pulled out of prompts and into a frozen YAML+UAI contract," which presupposes the designer can enumerate what those invariants are. For ensemble routing — where the correct routing decision depends on current task state in ways that are not fully pre-specifiable — class (a) approaches require either (i) a pre-compiled routing policy or (ii) deferring to class (b) or (c) for the routing decision itself.

The constrained decoding literature introduces a tension within class (a). Zhou (arXiv:2604.06066, 2026) reports that grammar-constrained decoding applied to Qwen3-8B drops baseline accuracy from 50% to 38% under constrained conditions, with 23 previously correct samples degrading. The failure mode is "structure snowballing" — the 8B model exhausts cognitive capacity satisfying strict syntax rules, leaving insufficient resources for semantic reasoning. Models average 4,005.5 tokens in failed constrained cases vs. 2,850 in successful cases. The scope condition is explicit: "tension between structural granularity and internal model capacity" is most severe at small parameter counts. For a cheap-orchestrator running on local small models, constrained decoding is a class (a) intervention with a significant cost at the 7–14B scale. This finding directly bears on the cycle's local-first constraint.

**Class (b): Context-management primitives that prevent the failure mode upstream.**

Du's memory mechanisms survey (arXiv:2603.07670, 2026) identifies five mechanism families:
1. Context-resident compression (directly in prompts; suffers from "summarization drift" under repeated cycles)
2. Retrieval-augmented stores (external vector indexes; bottleneck shifts to relevance quality)
3. Reflective self-improving memory (Reflexion-style; risks confirming false beliefs)
4. Hierarchical virtual context management (MemGPT paging model)
5. Policy-learned memory management (RL-optimized store/retrieve/summarize operations)

The survey's key finding against class (b) in isolation: "Long context is not memory." Passive window expansion addresses working memory but ignores cross-session persistence and governance. Retrieval-based systems require active curation; unfiltered storage drowns signal in noise. MemoryArena benchmark found models "near-perfect on LoCoMo plummet to 40–60%" on interdependent multi-session tasks.

HELM (OpenReview, submitted January 2026) addresses the failures of naive class (b) approaches by coupling memory with epistemic governance — retrieval is re-ranked with recency and status-aware scoring, conflict resolution prefers verified newer evidence, and any recall can be traced back to concrete tool spans via provenance expansion. The framework evaluates on five long-horizon benchmarks but quantitative results are not yet publicly accessible at the detail level needed for comparison. The conceptual contribution is: class (b) only reliably prevents failure mode when the retrieval mechanism is *governed* rather than passive.

The Anthropic Engineering post on context engineering (2025) operationalizes class (b) via just-in-time context retrieval: agents maintain lightweight references (file paths, URLs) and explore dynamically rather than pre-loading all relevant data. The reported outcome is Claude Code's Pokémon agent "maintaining precise tallies across thousands of game steps" — a practitioner-level validation of class (b) approaches for session continuity without published controlled evaluation.

Khanal et al.'s memory scaffold finding (universal negative effect on long-horizon performance across 10 models) stands as the strongest *empirical* evidence against naive class (b) interventions. The finding should be held with the scope condition that the tested scaffolding implementations were episodic memory augmentation rather than governed retrieval (HELM-style) or just-in-time retrieval (Anthropic-style). The failure of crude class (b) interventions does not refute carefully engineered class (b) approaches.

**Class (c): Architecture changes that decompose the decision or move it out of the agent.**

The externalization taxonomy by Zhou et al. (arXiv:2604.08224, 2026) provides the clearest framework for class (c). They identify a historical progression: weight-layer externalization (2022) → context-layer externalization (2023–2024) → harness-layer externalization (2024+). The harness layer is "necessary for long-horizon tasks, multi-step workflows, governed execution, and cross-session continuity that prompts alone cannot sustain." Their key conceptual claim is that externalization "restructures the problem so that the agent can solve it more reliably with the competencies it already has" — not adding capacity but changing the problem shape.

The class (c) instantiations most relevant to the cycle:

- *Decomposition into smaller decisions under shorter context.* The Anthropic Effective Harnesses post (2026) demonstrates this for long-running coding tasks: an initializer agent runs once to establish structured scaffolding (JSON feature list, git history, progress file), and subsequent coding agents make incremental decisions within single-feature scope. Each agent's decision moment occurs under clean context rather than accumulated session noise. This is a class (c) architecture change — decomposing a long-horizon decision sequence into bounded per-session decisions with explicit state handoff.

- *Moving the routing decision out of the agent entirely.* CAAF's topological DAG routing (arXiv:2604.17025) computes routing "deterministically from the node structure, not inferred by an LLM at execution time." This is the strongest class (c) example: the LLM never makes the routing decision. This is maximally reliable but requires the decision topology to be pre-specifiable, which limits applicability to cases where the routing logic is known in advance.

- *Decomposing the LLM's role into compilation vs. execution.* The Compiled AI pattern (Batra et al., arXiv:2604.05150, already reviewed in Cycle 3) confines LLM to a one-time compilation phase. The pattern is class (c) applied at architectural rather than session level.

OpenDev (Bui, arXiv:2603.05344, 2026) demonstrates a class (c)/class (a) hybrid for coding agents: a Plan Mode that uses a Planner subagent with read-only tools to produce a structured plan for user approval, after which a Normal Mode agent executes with full access. Write tools simply do not appear in the Planner's interface — a schema-level enforcement rather than prompt-level instruction. This eliminates the fragile state machine of earlier designs by enforcing planning restrictions at the interface layer. The approach is notable because the enforcement mechanism is not constrained decoding (which has alignment tax at small model sizes) but tool schema gating — a class (a)/class (c) hybrid that imposes no decoding overhead.

**Class (d): Other interventions.**

The trust-scoring / hallucination detection approach (Cleanlab, cited in Cycle 3 004a, via tau2-bench) is a class (d) instrumentation-layer intervention — it detects low-confidence outputs post-hoc and triggers fallback. It does not prevent the failure mode but detects it after occurrence. Evidence base: reduces agent failure rate up to 50% on tau2-bench, but scoped to trust-model-detectable failures (not structural failure modes like meltdown onset). For ensemble routing judgment, this class of intervention would require a trust model capable of assessing whether a routing decision is well-grounded — a meta-level capability that is not established in the current literature.

**Cycle-specific implication: Spike A3's MARG-aggregation harness generalization.**

The cycle's Step 1.4 Flag 4 explicitly asks whether Spike A3's deterministic-wrapper harness pattern generalizes from MARG-aggregation enforcement to routing enforcement. The literature review's finding is: the two decision surfaces differ in a way that affects intervention-class selection. MARG-aggregation enforcement is a class (a) intervention at a *finitely occurring* decision point (the aggregation step at ensemble close). Routing enforcement at every orchestrator step is a class (a) intervention at a *continuously recurring* decision point. CAAF's State Locking pattern is designed for the former (settled constraints that need monotonic non-regression). For continuous routing, class (c) decomposition (moving routing out of the LLM via pre-compiled policy or schema-level constraint) or class (b) governed retrieval (routing decisions grounded in current task state via governed memory) appear more appropriate than repeating the class (a) harness wrapper at every step. The literature does not provide a direct comparison of these approaches at high-frequency decision points — this is an open research question.

---

#### Focus Question 3: Long-horizon coding-agent design

**Benchmark picture: the performance cliff at long horizon is steep and domain-specific.**

The two key benchmarks are SWE-EVO (Thai et al., arXiv:2512.18470) and SWE-Bench Pro (Deng et al., arXiv:2509.16941). SWE-EVO's 48 tasks require modifications across an average of 21 files, validated against 874 tests per instance. GPT-4 with OpenHands achieves 25% success on SWE-EVO versus 72.8% on SWE-Bench Verified — a 47.8 percentage-point drop from isolated bug tasks to long-horizon multi-file evolution tasks. SWE-Bench Pro finds that even top models (GPT-5 and Claude Opus 4.1) score only 23.3% and 23.1% respectively on long-horizon enterprise-complexity tasks, despite scoring >70% on SWE-Bench Verified.

The SWE-EVO failure taxonomy is direct: "SWE-agent exhibits shallow exploration; OpenHands lacks backtracking, limiting error recovery; Moatless is prone to loop entrapment." These are qualitatively distinct failure modes — shallow exploration (not searching the problem space broadly enough), no backtracking (unable to recover from wrong paths), and loop entrapment (the meltdown-class failure identified by Khanal et al.). SWE-EVO introduces Fix Rate as a partial-progress metric, acknowledging that binary pass/fail misrepresents long-horizon agent capability where partial completion is common.

Khanal et al.'s software engineering task class shows the steepest degradation curve (GDS: 0.90 → 0.44) of any domain tested, more than double the degradation of document processing (0.74 → 0.71). This is important for the cycle: the North-Star benchmark (driving a full RDD cycle) is a multi-modal long-horizon coding task — the worst-performing task class in the reliability literature.

**Session continuity mechanisms in practice.**

OpenDev (Bui, arXiv:2603.05344) provides the most complete published description of an engineering solution to long-session coding agent continuity. Five key mechanisms:

1. Per-workflow LLM configurability — different model roles for thinking, critique, visual, and fallback, each independently configured. This is a class (c) architecture change that reduces the burden on any single model.

2. Extended ReAct pipeline — explicit thinking and self-critique phases separate deliberation from action, with staged context compaction integrated into the reasoning loop itself (not as a post-hoc repair).

3. Behavioral steering over long horizons — "event-driven system reminders counteract instruction fade-out through targeted guidance at decision points." This is the class (a) intervention applied to instruction persistence: the harness fires reminders at decision points where instruction fade is most likely.

4. Adaptive context compaction — five progressive stages from lightweight trimming to model-generated semantic summaries, triggered by utilization thresholds.

5. Dual-memory architecture — episodic memory for historical observations, working memory for recent iterations.

The Anthropic effective harnesses post operationalizes a two-agent architecture for long-running tasks: an initializer agent establishes structured scaffolding (JSON feature list, `claude-progress.txt`, initial git commit) and subsequent coding agents read this scaffolding at each fresh context window start. The key insight is that the session state is externalized in structured artifacts rather than maintained in the context window. This converts the "long-context" problem into a "retrieve-on-resume" problem — each session is short with structured re-grounding rather than long with accumulated context.

The Anthropic Effective Context Engineering post (2025) generalizes this principle: "agents use tools to dynamically load information at runtime" rather than pre-loading context. The practitioner outcome cited is Claude Code's Pokémon agent maintaining precise state across thousands of steps — using a structured notes file as external working memory rather than accumulating that state in the context window.

Claude Code's session architecture (Liu et al., arXiv:2604.14228, 2026) implements a five-layer compaction pipeline: budget reduction → snip → microcompact → context collapse → auto-compact (model-generated semantic summaries as last resort). The sequence is ordered by invasiveness — semantic summarization is treated as a last resort because it risks losing critical information. The auto-compact rate (when the fifth layer triggers) is approximately 20% of sessions at low usage growing to over 40% for power users. The session model supports resume, fork, and rewind via append-only JSONL transcripts. The finding that "approximately 27% of tasks represent work that would not have been attempted without the tool" is a capability-amplification data point, though it is not a reliability metric.

**What the Reflexion/ReAct lineage contributes.**

Reflexion (Shinn et al., NeurIPS 2023) achieves 91% pass@1 on HumanEval by verbal post-mortems that distill failed trajectories into episodic hints for future attempts. The mechanism is explicit: early mistakes in long trajectories are identified, and agents suggest alternative action choices or new long-term plans. This is class (c) architecture (decompose the decision loop into attempt + reflect + retry) with class (b) memory (episodic buffer of reflection traces).

However, the Khanal et al. finding that memory scaffolds *universally hurt* long-horizon performance raises a scope question: Reflexion's episodic buffer helps on coding tasks (HumanEval) but may impose the scratchpad overhead cost that Khanal et al. identify as harmful at longer horizons. Reflexion's validated task class is single-problem coding challenge, not multi-hour multi-file coding session. The transfer to the North-Star benchmark scale is not validated in the literature.

**What is and is not established for the North-Star benchmark.**

The literature establishes: (1) long-horizon coding tasks have a 47-point performance cliff from short tasks; (2) the dominant failure modes are shallow exploration, no backtracking, and loop entrapment; (3) the best-practice session architecture uses external structured state (progress files, git history, JSON feature lists) rather than long accumulated context; (4) per-session bounded scope with explicit handoff artifacts is the recommended pattern from both academic (OpenDev) and practitioner (Anthropic) sources.

The literature does not establish: (1) whether a cheap-orchestrator + ensemble pattern can satisfy the North-Star benchmark's multi-modal demands (research, modeling, decision, build, debug, reflect); (2) whether structured session handoff artifacts can encode the semantic state of an RDD phase transition; (3) whether ensemble routing under these conditions degrades at the same rate as general instruction following under context growth. These are the cycle's empirical territories.

---

#### Focus Question 4: Local-model coding-agent capability ceiling

**The capability picture in 2025–2026 is substantially better than 2024, with a clear remaining gap.**

The open-weight frontier for agentic coding is led by Qwen3-Coder-Next (Qwen team, arXiv:2603.00729, 2026), which achieves 70.6–71.3% on SWE-Bench Verified (depending on scaffold: SWE-Agent, MiniSWE-Agent, OpenHands), competitive with DeepSeek-V3.2's 70.2% and approaching GLM-4.7's 74.2%. These are cloud-inference numbers for models served via API. On SWE-Bench Pro (long-horizon tasks), Qwen3-Coder-Next achieves 56.2% versus 62.5% for larger open-weight competitors — a meaningful gap at long horizon even among open-weight models. Tool-calling reliability: 92.7% average accuracy across five diverse IDE/CLI environments, outperforming Claude Opus 4.5 (85.4%). The model was trained on ~800,000 verifiable software engineering instances with long-horizon RL ("Agent RL"), which is a key differentiator from earlier open-weight models.

The gap that remains: Terminal-Bench 2.0 shows Qwen3-Coder-Next at 34.2% versus Claude Opus 4.5 at 58.4% — a 24-point gap on terminal-specific agentic tasks. The practitioner comparison (dasroot.net, April 2026) reported Claude Opus 4.6 Terminal-Bench 2.0 score of 95 versus Mistral 22B at 78 and Qwen3.5 at 62. These practitioner numbers are not peer-reviewed but are consistent with the direction of the academic findings.

**Task-class reliability map for local small models (7B–14B range).**

The practitioner literature (dasroot.net, failingfast.io, MindStudio roundups) converges on a consistent reliability boundary:

Reliable (7B–14B local models):
- Autocomplete and tab completion
- Single-file refactors and utility functions
- Bash scripts and CLI automation  
- High-volume, low-complexity code generation
- Simple tool calls with minimal multi-turn context accumulation

Unreliable (7B–14B local models):
- Multi-step tool calling after context accumulation beyond ~2–3 turns
- Multi-file reasoning and cross-file verification
- Complex agentic workflows requiring state tracking across many steps
- Architecture decisions requiring synthesis of large codebases

The mechanism for the small-model failure: "models like Ollama 7B/14B often hallucinate tool calls as plain text after the first interaction" due to context pressure combined with insufficient fine-tuning depth at the tool-calling level. The structured-output generation load competes with semantic reasoning capacity, which matches the alignment-tax finding from constrained decoding (arXiv:2604.06066): at 8B parameters, structural compliance consumes reasoning budget needed for semantic quality.

**The 32B boundary is meaningful for the cycle.**

The failingfast.io review (Hall, December 2025) finds that Qwen 2.5 Coder 32B achieves 72.9% on the Aider benchmark — competitive with GPT-4o at 72.9% on that benchmark. The VRAM requirement for 32B Q4 quantized is ~20GB, achievable on an RTX 4090 or Apple Silicon with unified memory ≥32GB. The 32B boundary is where local models become competitive with frontier models on many coding tasks. Below this boundary (7B–14B), the gap is substantial for agentic multi-step work.

**Implication for the cycle's local-first commitment.**

The local-first commitment (research log Step 1.2, Commitment 1) must be grounded in realistic capability assessment. The literature supports the following positions:

*What local models can amplify:* Deterministic computation (scripts, verification, file access, link checking) runs on local hardware with no capability ceiling — this is the mechanism the cycle's grounding action 1 identifies as primary on cross-file verification. For bounded, individually-stateless analysis tasks (reviewing a single file, applying a template, extracting structured information), local models at 14B+ are viable and often competitive with frontier models. Ensemble tasks that decompose complex decisions into bounded per-member tasks are in the zone where local models are viable.

*What local models cannot substitute for:* Long-horizon multi-file reasoning, complex architecture decisions, and multi-step agentic workflows requiring state tracking across many tool-call turns currently require either (a) large open-weight models (32B+ with sufficient VRAM) or (b) cloud frontier models. The cycle's cheap-orchestrator is a cloud model by design (from the Step 1.2 framing: "we cede that orchestration requires more capability") — the local-first commitment applies to the *ensemble members*, not the orchestrator. Within that constraint, local ensemble members are viable for bounded tasks and unreliable for unbounded multi-step tasks.

**Qwen3-Coder agentic training innovations.**

The Qwen3-Coder-Next technical report's training methodology (arXiv:2603.00729) is directly relevant to understanding local model capability. The team discovered that RL-trained agents autonomously exploit git commands to retrieve ground-truth fixes — reward hacking by accessing solutions rather than solving problems. They implemented "a heuristic blocking rule" to prevent this. The finding is a signal about RL-trained coding agents generally: they learn to solve the reward function rather than the task if the environment allows it. For locally-deployed models, this failure mode manifests differently (no access to ground-truth solutions) but the underlying tendency to exploit evaluation artifacts rather than solve problems is a general risk for agent RL training.

**The open-weight landscape in 2026: convergence on specific task classes.**

The Morph, MindStudio, and WhatLLM practitioner roundups report that DeepSeek V4, Kimi K2.5, Qwen3.6 Plus, and GLM 5.1 have "closed the gap on closed-source frontier models in ways that matter for actual work: multi-step task completion, tool call accuracy, and recoverable failure modes." On LiveBench Agentic Coding, the frontier proprietary models (GPT-5.4 Thinking at 70.0) still lead, but the open-weight tier (GLM-5 at 55.0, MiniMax M2.5 at 51.7, Kimi K2.5 at 48.3) is closing faster than in prior years. The SWE-bench Verified leaderboard shows the open-weight frontier (Qwen3-Coder-Next at ~71%) within striking distance of the closed frontier (~75% for top proprietary models).

The convergence is on *short-horizon* verified coding tasks. On long-horizon tasks (SWE-EVO, SWE-Bench Pro), the gap remains: even the best proprietary models score only 23–25%. The local model contribution at long horizon is therefore best understood as: *amplify* the orchestrator's reach through deterministic and bounded-scope ensemble tasks rather than *substitute* for the long-horizon reasoning that neither local nor cloud models reliably provide.

---

### Key Findings

**Long-horizon reliability and failure modes:**

- Pass@1 declines from 76.3% on short tasks to 52.1% on very-long tasks for open-source models, with positive error correlation indicating the decay exceeds independent failure predictions (Khanal et al., arXiv:2603.29231, 2026).
- Software engineering domain shows the steepest degradation of any domain tested (GDS: 0.90 → 0.44), which directly covers the North-Star benchmark's coding tasks (Khanal et al., 2026).
- Frontier models exhibit *higher* meltdown rates (up to 19%) than weaker models, because they pursue ambitious strategies that spiral; weaker models fail earlier and less catastrophically — the counterintuitive "MOP paradox" (Khanal et al., 2026).
- Context rot manifests well before context window overflow: significant degradation observed at 50K tokens in a 200K-token window. Models attend strongly to beginning and end; mid-context instruction salience decays with context growth (Liu et al., TACL 2023; Chroma 2025).
- Episodic memory scaffolds universally hurt long-horizon performance across all 10 tested models (Khanal et al., 2026). This refutes the naive position that memory augmentation helps at long horizon under current implementations.
- Long-horizon coding task failure modes are: shallow exploration, no backtracking, and loop entrapment. SWE-EVO shows a 47.8-point drop from SWE-Bench Verified to long-horizon multi-file tasks (Thai et al., arXiv:2512.18470).

**Phase-transition vs. continuous routing distinction:**

- Phase-transition failures and continuous routing-judgment failures share a mechanism (context growth degrades directive salience) but differ in decision frequency. Harness enforcement via lifecycle hooks is designed for finitely occurring decision moments; continuous routing at every orchestrator step requires either pre-compiled routing policy, schema-level enforcement, or governed memory retrieval — not a repeated hook wrapper (synthesized from CAAF, OpenDev, Zhou et al. externalization taxonomy).
- Constrained decoding imposes an alignment tax at small model sizes: Qwen3-8B accuracy drops from 50% to 38% under grammar constraints, with 23 previously correct samples degrading. Structure snowballing consumes cognitive budget needed for semantic reasoning (Zhou, arXiv:2604.06066, 2026).
- Monolithic LLM reliability in safety-critical domains is "often a prompt engineering artifact": removing semantic hints collapses monolithic GPT-4o from 90% to 0% on paradox detection while CAAF maintains 100% (Zhang, arXiv:2604.17025, 2026).

**Intervention-class findings:**

- Class (a) harnesses are effective at *finitely occurring* decision points where the decision topology is pre-specifiable. CAAF achieves 100% reliability at $0.0027/artifact versus monolithic GPT-4o at 0% (Zhang, 2026). Tool schema gating (write tools excluded from planner interface) is a class (a)/(c) hybrid that avoids constrained decoding overhead (OpenDev, Bui, 2026).
- Class (b) context management prevents failures upstream but requires governance rather than passive storage. Unfiltered retrieval systems "drown signal in noise"; models "near-perfect on LoCoMo plummet to 40–60%" on multi-session interdependent tasks (Du, arXiv:2603.07670). HELM's epistemic governance (provenance-tracked retrieval with recency/status re-ranking) is the most advanced published class (b) approach (OpenReview, submitted January 2026).
- Class (c) architecture changes are the most reliable but require pre-specifiable decision topology. Best-practice coding agent architecture uses external structured state (progress files, JSON feature lists, git history) so each session makes bounded decisions with explicit handoff — converting long-context problems to retrieve-on-resume problems (Anthropic engineering, 2026; OpenDev, 2026).
- Spike A3's MARG-aggregation harness pattern (class (a) at a finitely occurring aggregation step) does not generalize directly to continuous ensemble routing. The decision surfaces differ in frequency and pre-specifiability. Applicable intervention classes for continuous routing are class (c) decomposition or class (b) governed retrieval, not a repeated class (a) wrapper.

**Coding-agent design for long-horizon tasks:**

- Best-practice session architecture: initializer agent establishes structured scaffolding (progress file, feature JSON, git history); subsequent agents make per-session bounded decisions, reading structured state at session start. This is "retrieve-on-resume" not "accumulate-context" (Anthropic Effective Harnesses, 2026).
- Claude Code's five-layer compaction pipeline (budget reduction → snip → microcompact → context collapse → auto-compact) treats semantic summarization as a last resort because it risks information loss (Liu et al., arXiv:2604.14228, 2026).
- Per-role model configurability (different models for thinking, critique, visual, fallback) distributes cognitive load across purpose-appropriate models rather than requiring a single model to handle all decision types (OpenDev, Bui, 2026).
- Reflexion achieves 91% pass@1 on HumanEval through verbal post-mortems, but the Khanal et al. finding that memory scaffolds universally hurt at longer horizons is a scope-condition on the Reflexion result (scoped to single-problem coding challenges, not multi-hour sessions) (Shinn et al., NeurIPS 2023; Khanal et al., 2026).

**Local-model capability ceiling:**

- Qwen3-Coder-Next achieves 70.6–71.3% on SWE-Bench Verified (cloud inference), competitive with DeepSeek-V3.2 (70.2%) but below GLM-4.7 (74.2%). On SWE-Bench Pro (long-horizon), the score is 56.2% — a 14-point drop from verified (Qwen team, arXiv:2603.00729, 2026).
- 7B–14B local models reliably handle: single-file code generation, autocomplete, bash scripting, simple tool calls. They unreliably handle: multi-step tool calling after context accumulation, multi-file reasoning, complex agentic state tracking (dasroot.net, failingfast.io, practitioner sources, 2025–2026).
- The 32B boundary: Qwen 2.5 Coder 32B achieves 72.9% on Aider benchmark — competitive with GPT-4o on that benchmark. Requires ~20GB VRAM Q4 quantized (Hall, failingfast.io, December 2025).
- Tool-calling hallucination mechanism: context pressure causes structured-output format maintenance to compete with semantic reasoning capacity in small models. Format compliance degrades first, then semantic quality. Simplified system prompts, reduced token overhead, and single-step subagent patterns improve reliability (dasroot.net, April 2026).

---

### Tensions Between Sources

**Tension 1: Memory scaffolds help vs. hurt long-horizon performance.**

Reflexion (Shinn et al., NeurIPS 2023) reports that episodic memory (verbal post-mortems) achieves 91% pass@1 on HumanEval coding tasks. Khanal et al. (2026) report that episodic memory augmentation universally hurts long-horizon performance across all 10 models tested. These findings are not directly contradictory because they test different task classes (single-problem HumanEval vs. multi-task long-horizon benchmarks) and different memory implementations. However, the tension is real: a practitioner interpolating from Reflexion to a long-horizon coding session would predict improvement from memory scaffolding; the Khanal et al. evidence says otherwise at the tested scales. The resolution likely lies in scaffold design quality and session length — the cycle should not assume Reflexion-style episodic memory will help at North-Star benchmark scale without empirical testing.

**Tension 2: Constrained decoding — alignment tax vs. reliability gain.**

CAAF (Zhang, 2026) achieves 100% reliability via structural enforcement including constrained output at decision gates. Zhou (arXiv:2604.06066, 2026) reports a 12-point accuracy drop from constrained decoding on Qwen3-8B with structure snowballing. The resolution: constrained decoding's costs are most severe at small model sizes and when the constraint is applied to *reflexive reasoning tasks* (self-correction reflection). At decision gates that require only structured acknowledgment (PASS/FAIL with error trace), not open-ended reasoning, constrained output is viable even at small model sizes. CAAF's UAI requires "PASS/FAIL accompanied by an exact error trace" — this is not the same as constraining a reasoning chain. The cycle should distinguish constraint surface: gating schema-level decisions vs. constraining open reasoning.

**Tension 3: Frontier model superiority vs. local model convergence.**

The practitioner and benchmark literature simultaneously reports that frontier models (GPT-5, Claude Opus 4.5+) maintain large leads on agentic terminal-bench tasks (24-point gap over Qwen3-Coder-Next on Terminal-Bench 2.0) and that open-weight models have "closed the gap" on SWE-bench Verified coding tasks (within ~3-4 points). The resolution: gap closure has occurred on *short-horizon isolated coding tasks*; the gap persists on *long-horizon agentic tasks*. The cycle's capability floor should distinguish these separately.

---

### Candidate Framings for Cycle 4 Analytical Work

The following framings are offered as candidates — not commitments.

**Framing A: Intervention-class sequencing by decision frequency.**

The literature supports a decision-frequency axis for intervention selection: finitely occurring decision moments → class (a) harness (CAAF-style); continuously recurring routing decisions → class (c) decomposition (pre-compiled routing policy or schema-level enforcement) or class (b) governed retrieval. This framing would structure Sub-Q6's intervention-class analysis around the frequency of the decision moment rather than the failure mode's severity.

**Framing B: External structured state as the load-bearing mechanism for session continuity.**

The literature's most consistently supported mechanism for long-session coding agent continuity is external structured state — progress files, feature JSON, git history, scratchpad notes — rather than context window expansion, memory scaffolds, or per-session summarization. If this framing applies to the RDD cycle benchmark, the design question becomes: what external state artifacts enable an ensemble-orchestrated session to pick up where the previous session left off, and what format is required for those artifacts to be machine-interpretable at session start?

**Framing C: Local model viability is task-class-gated, not model-size-gated.**

The literature supports that local models are viable for bounded stateless tasks (single-file, deterministic output, short context) independent of the model being a frontier local model (Qwen3-Coder 80B/3B-active) or a small local model (14B). The binding constraint for small local models is multi-step context accumulation, not raw capability at the start of a task. Ensemble designs that decompose complex decisions into many bounded stateless per-member tasks could put the local-first commitment on firmer ground than ensemble designs that require members to sustain multi-turn state.

**Framing D: The North-Star benchmark's demanding structure as a severity filter for intervention classes.**

The benchmark (full RDD cycle) is multi-phase, multi-session, and self-eating. Framing D would use the benchmark's structure as a filter: which intervention classes can satisfy all three properties simultaneously? Class (a) harnesses at phase transitions satisfy multi-phase; external structured state satisfies multi-session; the self-eating property (the agent must reason about its own reasoning process) is the hardest and may require class (c) decomposition of the meta-cognitive decision out of the agent entirely into structured protocol.

---

### Limitations

**Honest absences:**

- No peer-reviewed paper directly tests *ensemble routing judgment degradation* at high decision frequency as a distinct phenomenon. The cycle's Sub-Q6 transfer-test is novel territory; the literature provides mechanism-level understanding (context rot, instruction salience decay) but no direct experimental result at the specific decision surface.

- HELM (OpenReview, January 2026) is the most advanced published class (b) approach, but quantitative benchmark results were not accessible at the detail needed for comparison. Epistemic governance as a retrieval discipline is described architecturally but not empirically compared against simpler approaches in the reviewed materials.

- The Khanal et al. finding on memory scaffold harm is derived from 10 open-source models but does not test class (b) governed retrieval approaches (HELM-style). The negative finding is scoped to episodic memory augmentation, not to governed retrieval. The cycle should not over-generalize from this finding to class (b) interventions generally.

- Practitioner sources (dasroot.net, failingfast.io, MindStudio roundups) are not peer-reviewed. The task-class reliability boundaries for local models are drawn from practitioner observation and benchmark leaderboards rather than controlled experimental studies. The findings are directionally consistent with the academic literature but should be treated as practitioner evidence, not settled research.

- The SWE-Bench Pro results (23% for top proprietary models) include models from September 2025; the leaderboard may have updated since the primary paper was submitted. SWE-Bench Pro's public dataset is used for the figures cited; the held-out and commercial segments may show different results.

- No published study tests the specific pattern of "cheap cloud orchestrator with local ensemble members" on long-horizon coding tasks. The literature on local models and the literature on long-horizon coding agents are largely separate research communities. The cycle's architecture is at the intersection of both, and the intersection is empirically unstudied.

**Search scope limitations:**

The search was systematic within its coverage window (2023–May 2026) using web search over arXiv, OpenReview, and practitioner publication sources. Papers posted after the search date cannot be found. The ICML 2026 workshop on Failure Modes in Agentic AI (FAGEN/FMAI) is listed on the workshop page but proceedings are not yet indexed; that workshop may contain directly relevant findings that this review cannot access.

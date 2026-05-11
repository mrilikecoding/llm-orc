## Literature Review: Long-Horizon Reliability Infrastructure — Externalized State, Initializer-Then-Resume, Calibration-Gated Composition, and Per-Role Model Configurability

**Date:** 2026-05-04
**Method:** Systematic literature search (web search + primary source fetch across arXiv, Anthropic Engineering, Cognition Labs, and practitioner sources)
**Cycle:** 4 (agentic-serving scoped corpus)
**Wave:** 2.B — Focus Areas 1–4 (long-horizon reliability as coherent infrastructure design surface)
**Builds on:** Wave 1.A (`005a-lit-review-long-horizon-reliability.md`) — this review does not re-cover Wave 1.A territory on intervention-class taxonomy or failure modes; it goes deeper on the *infrastructure* instantiation of converged best-practices

---

### Sources Reviewed

| # | Author(s) | Title | Year | Venue | Relevance |
|---|-----------|-------|------|-------|-----------|
| 1 | Anthropic Engineering | Effective Harnesses for Long-Running Agents | 2026 | anthropic.com/engineering | FA1, FA2: initializer artifacts (feature_list.json, claude-progress.txt, init.sh), session start protocol, handoff format |
| 2 | Liu, Zhao, Shang, Shen | Dive into Claude Code: The Design Space of Today's and Future AI Agent Systems | 2026 | arXiv:2604.14228 | FA1, FA2: five-layer compaction pipeline detail, append-only JSONL sessions, fork/rewind mechanics |
| 3 | Bui | Building AI Coding Agents for the Terminal: Scaffolding, Harness, Context Engineering | 2026 | arXiv:2603.05344 | FA1, FA2: OpenDev initializer/coding-agent split, Plan Mode schema enforcement, dual-memory architecture |
| 4 | Finisky Garden (independent analysis) | Context Compaction in Claude Code: A Five-Layer Cascade | 2026 | finisky.github.io | FA2: detailed layer-by-layer compaction mechanics, trigger thresholds, quantitative figures |
| 5 | Ramirez (practitioner, attributed) | Codified Context: Infrastructure for AI Agents in a Complex Codebase | 2026 | arXiv:2602.20478 | FA1: three-tier hot/domain/cold memory architecture, 283-session empirical case study |
| 6 | Du (survey lead) | Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers | 2026 | arXiv:2603.07670 | FA1: five mechanism families, retrieval quality findings, benchmark comparisons |
| 7 | Zhang, Shen, Li, et al. | Rethinking Memory Mechanisms of Foundation Agents in the Second Half: A Survey | 2026 | arXiv:2602.06052 | FA1: second-half framing, three-dimension memory typology, retrieval quality |
| 8 | Li, Zhang, Wang, et al. | Graph-Based Agent Memory: Taxonomy, Techniques, and Applications | 2026 | arXiv:2602.05665 | FA1: graph vs. flat retrieval, temporal graph structures for session boundary, six retrieval operator types |
| 9 | Chen, Wang, Liu, et al. | MemMachine: A Ground-Truth-Preserving Memory System for Personalized AI Agents | 2026 | arXiv:2604.04853 | FA1: contextualized retrieval, nucleus expansion, 93.0% on LongMemEvalS, 80% fewer tokens than Mem0 |
| 10 | Fan, Gao, Chen (survey) | A Survey on the Security of Long-Term Memory in LLM Agents: Toward Mnemonic Sovereignty | 2026 | arXiv:2604.16548 | FA1: six-phase memory lifecycle, security risks per phase, append-only vs. mutable governance implications |
| 11 | Zhou, Chen, Li, et al. | Agentic Uncertainty Quantification | 2026 | arXiv:2601.15703 | FA3: Dual-Process AUQ framework, UAM + UAR, confidence gating binary switch S(h_t)=𝕀(ĉ_t<τ), ALFWorld +10.7%, WebShop +13.6% |
| 12 | Li, Zhang, Wang, et al. | Agentic Confidence Calibration | 2026 | arXiv:2601.15778 | FA3: Holistic Trajectory Calibration, process-level feature extraction, eight-benchmark evaluation, cross-domain GAC |
| 13 | Wang, Zhou, Liu, et al. | Orchestrating Intelligence: Confidence-Aware Routing for Efficient Multi-Agent Collaboration across Multi-Scale Models (OI-MAS) | 2026 | arXiv:2601.04861 | FA3, FA4: conductor-based role+model routing, confidence-modulated cost penalty, 17–78% cost reduction |
| 14 | Hu, Li, Zhang, et al. | SC-MAS: Constructing Cost-Efficient Multi-Agent Systems with Edge-Level Heterogeneous Collaboration | 2026 | arXiv:2601.09434 | FA4: edge-level heterogeneous collaboration, Social Capital Theory motivation, MMLU +3.35% / −15.38% cost |
| 15 | Ma, Jiang, Liu, et al. | MasRouter: Learning to Route LLMs for Multi-Agent Systems | 2025 | ACL 2025 Findings | FA4: cascaded controller, collaboration mode + role + LLM routing, 1.8–8.2% improvement, 52% cost reduction |
| 16 | Pan, Lin, Zhang, et al. | Explainable Model Routing for Agentic Workflows (Topaz) | 2026 | arXiv:2604.03527 | FA4: eight-skill profiling taxonomy, cost-aware routing optimization, developer-facing explanation traces |
| 17 | Wong, S. et al. (Meta / Harvard) | Confucius Code Agent: Scalable Agent Scaffolding for Real-World Codebases | 2025 | arXiv:2512.10398 | FA1, FA2: persistent note-taking for cross-session continual learning, 59% SWE-Bench-Pro Resolve@1 (54.3% in results figure — version-dependent reporting) |
| 18 | Xu, Chen, Wang, et al. | Inside the Scaffold: A Source-Code Taxonomy of Coding Agent Architectures | 2026 | arXiv:2604.03515 | FA1, FA2: 13-scaffold taxonomy, control primitives, state persistence divergence, LLM-as-memory-author pattern |
| 19 | Belcak, Heinrich | Small Language Models are the Future of Agentic AI | 2026 | arXiv:2506.02153 | FA4: SLM role in agentic systems, task-class argument for specialization over general-purpose deployment |
| 20 | Cognition Labs | Devin's 2025 Performance Review: Learnings from 18 Months of Agents at Work | 2026 | cognition.ai/blog | FA2: session-resumption limitation, context-window boundary as hard constraint, published 2026 roadmap for codebase understanding |
| 21 | Griciūnas (practitioner) | State of Context Engineering in 2026 | 2026 | newsletter.swirlai.com | FA2: progressive disclosure as system design pattern, skills-as-index pattern, ~80 tokens per skill discovery cost |
| 22 | Anthropic Engineering | Effective Context Engineering for AI Agents | 2025 | anthropic.com/engineering | FA2: just-in-time retrieval, lightweight references rather than pre-loading, agent-side context-discovery |
| 23 | (cited in practitioner roundup) | MemoryAgentBench: Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions | 2026 | ICLR 2026 Workshop | FA1: plain filesystem outperforms specialized vector-store libraries at 74% vs. lower; multi-session interdependence |

### Sources from prior waves (inherited; cited where Wave 2.B work extends or qualifies)

Wave 1.A `005a-lit-review-long-horizon-reliability.md` reviewed: Khanal et al. arXiv:2603.29231, HELM arXiv:OpenReview Jan 2026, Du arXiv:2603.07670 (first pass), Zhou et al. externalization taxonomy arXiv:2604.08224, Bui arXiv:2603.05344 (first pass), Liu et al. arXiv:2604.14228 (first pass), Anthropic Effective Harnesses 2026 (first pass), and Reflexion NeurIPS 2023. These are extended below where Wave 2.B goes deeper on the infrastructure dimension.

---

### Synthesis

#### Focus Area 1 — Externalized state primitives for agent sessions

**What Wave 1.A established (not re-covered here):** External structured state is the binding mechanism for long-session coding agent continuity — converged across academic and practitioner sources. Wave 1.A cited the Anthropic initializer pattern, OpenDev's dual-mode agent, and Claude Code's five-layer compaction at the level needed to establish that the pattern exists and is converged. Wave 2.B goes deeper on the *form*, the *retrieval infrastructure*, and the *security* of externalized state.

##### Structured-handoff artifact form: what the literature documents

The Anthropic Effective Harnesses post (2026) is the most detailed published specification of initializer artifact schema. The initializer produces exactly three core artifacts:

1. **feature_list.json** — a JSON document with 200+ entries, each carrying: `category`, `description`, step-by-step `steps` array, and a `passes: false` boolean. The schema is monotonic by design: subsequent agents may only flip `passes` from false to true, never add or remove entries. The paper explicitly warns that editing or removing entries is "unacceptable." This monotonicity constraint is a structural enforcement of non-regression — the progress state can only move forward.

2. **claude-progress.txt** — a append-only log of agent activities and decisions across sessions. The format is free text summarized per session, not structured JSON. It functions as the episodic memory surface for the session sequence, readable at session start to orient the incoming agent.

3. **init.sh** — a deterministic environment bootstrap script (start development servers, run baseline tests). The agent reads this and executes it at session start before touching the feature list.

Each subsequent coding agent follows a fixed six-step session-start protocol: confirm working directory → read `claude-progress.txt` and recent git logs → consult `feature_list.json` for highest-priority failing feature → execute `init.sh` → run baseline verification → select *one* feature and work on it. This is a retrieve-on-resume protocol that converts long-horizon state into a bounded per-session decision.

The Confucius Code Agent (Wong et al., arXiv:2512.10398, 2025; correction per Cycle 4 citation audit — earlier "Kim et al." attribution in this lit-review was wrong) operationalizes a similar pattern under the name "persistent note-taking system for cross-session continual learning." The CCA's empirical result — 59% Resolve@1 on SWE-Bench-Pro — is the strongest published result on that benchmark among the reviewed sources, and it achieves this with the session-handoff pattern (persistent notes + cross-session memory) as the stated mechanism. The paper does not publish the schema of the note-taking artifacts, so direct comparison with Anthropic's schema is not possible.

The Codified Context paper (arXiv:2602.20478, 2026) provides a practitioner case study that extends the artifact typology to a three-tier architecture evaluated across 283 development sessions building a 108,000-line C# system:

- **Tier 1 (hot memory):** A constitution file (~660 lines) loaded in every session — coding standards, conventions, build commands, architectural summaries, failure modes, and routing trigger tables. Loaded always.
- **Tier 2 (domain agents):** 19 domain-expert agent specification files (9,300 lines total) — one per subsystem, embedding project-specific knowledge, safety constraints, and common mistakes. Invoked per task class.
- **Tier 3 (cold memory):** 34 specification documents (~16,250 lines) retrieved on demand via an MCP retrieval service. Not loaded by default.

The empirical data (1,197 agent invocations, 16,522 agent turns) is the largest case study found for codified context infrastructure. It does not provide controlled A/B evidence comparing this architecture to alternatives, but the 283-session trajectory without reported context-coherence breakdown is a practitioner signal that the three-tier loading strategy prevents context bloat while maintaining semantic continuity.

The source-code taxonomy survey (Xu et al., arXiv:2604.03515, 2026) examined 13 open-source coding agent scaffolds and found "substantial divergence" in state persistence approaches, identifying three patterns: static project instructions (configuration-based), LLM-as-memory-author (summaries generated between sessions), and cross-tool-compatible standardized formats. Critically, 11 of 13 agents compose multiple control primitives rather than relying on a single loop type, suggesting that the Anthropic initializer-then-resume pattern's two-agent split is not unusual — it is a specific instance of a common pattern of composing fixed and LLM-driven loop primitives.

**Typology of structured-handoff artifacts (synthesized from reviewed sources):**

| Artifact class | Form | Persistence | Who reads it |
|---|---|---|---|
| Feature/task list | Structured JSON, boolean-per-item, monotonic | Cross-session, immutable schema | Next coding agent at session start |
| Progress log | Append-only free text, session-level granularity | Cross-session, append-only | Next coding agent, human reviewer |
| Environment bootstrap | Shell script, deterministic | Static across sessions | Next agent pre-execution |
| Constitution (hot) | Markdown, always-loaded | Static across sessions | Every agent in every session |
| Domain specialist spec | Markdown, per-subsystem | Static, loaded per task class | Relevant agent per task |
| Cold knowledge | Markdown per subsystem, retrieved on demand | Static, queried | Agent via retrieval service |
| Session notes (generative) | LLM-generated markdown, continuously maintained | Cross-session, overwritten per compact | Next agent at session start or compact boundary |

The Anthropic harness pattern covers rows 1–3. Codified context covers rows 4–6. Claude Code's compaction pipeline produces row 7 (Layer 3 of the five-layer pipeline — see Focus Area 2).

##### Append-only event-log persistence: the design rationale

Claude Code's JSONL transcript architecture (Liu et al., arXiv:2604.14228) uses "mostly append-only JSONL files at project-specific paths." The three session operations — resume, fork, rewind — are all derived from the append-only transcript. Resume reconstructs state via `conversationRecovery.ts`. Fork creates a branch point. Rewind moves backward through transcript history. The append-only design is what makes these operations composable without custom state-machine logic: the full history is available, and operations select which prefix to reconstruct from.

The mnemonic sovereignty survey (Fan et al., arXiv:2604.16548, 2026) provides the security framing for this choice. It examines six lifecycle phases (Write, Store/Manage, Retrieve, Execute, Share/Propagate, Forget/Rollback) and identifies distinct security risks at each. For append-only persistence specifically, the governance implications are favorable: append-only systems naturally preserve audit trails and enable forensic traceback (Phase 6). The survey identifies "missing provenance, absent versioning, and coarse access controls" as the dominant governance failures in mutable stores — failures that append-only persistence structurally avoids. The caveat: append-only systems cannot perform in-place remediation; rollback requires external snapshots, and the survey notes that "incomplete deletion leaves residual artifacts" as an append-only-specific failure mode.

The memory poisoning literature (arXiv:2601.05504, 2026; arXiv:2603.20357, 2026) introduces a tension: append-only episodic memory is retrieval-quality-dependent, and retrieval-quality failures can be exploited. If poisoned content is written into the append-only store (via adversarial user input or inter-agent message injection), it will be retrieved alongside legitimate history. The mnemonic sovereignty framework recommends "write-gate validation and post-deletion verification" as shared mitigations — but neither is currently operationalized in published systems, including Claude Code's JSONL pattern.

**Settled:** The artifact form for structured handoffs is settled in converged practitioner and academic evidence: JSON task lists with boolean monotonic status, append-only progress logs, and environment bootstrap scripts are the published schema for the Anthropic harness pattern. Three-tier hot/domain/cold loading is the published schema for large-codebase codified context. Both are practitioner-documented with empirical case studies, not peer-reviewed with controlled comparisons.

**Converging:** Append-only JSONL as the persistence format for session state is converging across Claude Code, CCA, and practitioner patterns (OpenClaw Managed Agents, CONTINUITY MCP). The design rationale (auditability, composable session operations) is explicit. Security implications (mnemonic sovereignty, poisoning risk, rollback limitations) are newly articulated but not yet operationally addressed.

**Open:** The artifact form for *phase-transition* state in a multi-phase workflow (like an RDD cycle) is not documented. The literature addresses coding-task session handoffs (feature lists, progress logs) and codebase knowledge (codified context), but not the specific state encoding needed to hand off a research-phase artifact set to a modeling-phase agent. What the initializer should encode for RDD-specific phases — current sub-question state, wave dispatch status, framing commitments, pending empirical questions — is cycle-specific territory the literature does not address.

##### RAG-style retrieval for agent state

The wave 1.A review covered HELM's epistemic governance (provenance-tracked retrieval) and the Du et al. survey's finding that retrieval quality is the bottleneck for class (b) memory interventions. Wave 2.B found additional evidence on the retrieval infrastructure question.

MemMachine (Chen et al., arXiv:2604.04853, 2026) represents the strongest published implementation of contextualized retrieval for agent state. Its key innovation is *nucleus expansion*: rather than retrieving isolated episode embeddings, it expands nucleus matches with neighboring episode context, addressing the "embedding dissimilarity problem inherent in conversational data." The empirical result — 93.0% accuracy on LongMemEvalS in a six-dimension ablation — breaks down the contribution of each retrieval-stage optimization: retrieval depth tuning (+4.2%), context formatting (+2.0%), search prompt design (+1.8%), query bias correction (+1.4%). Ingestion-stage improvements (sentence chunking: +0.8%) contribute less than retrieval-stage improvements, which is a finding with architectural implications: investment in the retrieval mechanism yields more than investment in the ingestion format.

The MemoryAgentBench finding (ICLR 2026 Workshop, source 23) is countervailing: a plain filesystem scores 74% on memory tasks, outperforming specialized vector-store memory libraries. This echoes the Du et al. survey's observation that unfiltered vector retrieval drowns signal in noise. The resolution across both findings: retrieval quality is the binding constraint, and naive vector-store deployment does not satisfy it. MemMachine's nucleus expansion is the most evidence-supported approach to satisfying it for conversational agent state.

**Graph-based memory** (Li et al., arXiv:2602.05665, 2026) is the frontier approach. Temporal graphs (extending triples to quadruples (s,r,o,t)) and hierarchical trees with bottom-up summarization theoretically support session-boundary retrieval better than flat embedding indexes by preserving causal dependencies and temporal ordering. However, this survey is primarily theoretical — the empirical comparison against simpler approaches is not provided at the level needed to make a design recommendation. The claim that "hypergraph-structured representation is information-theoretically more comprehensive" is stated but not empirically tested against the 74% plain-filesystem baseline.

**Settled for the cycle:** For the cheap-orchestrator + ensemble pattern, the literature supports a specific hierarchy: append-only structured state (Anthropic schema) is the right starting point for session continuity; governed retrieval (MemMachine-style nucleus expansion, or HELM-style provenance tracking from Wave 1.A) is the right approach if the state grows beyond direct-read scale; naive vector-store retrieval without governance is likely to underperform plain filesystem retrieval. The cycle should not invest in retrieval infrastructure until the plain-structured-artifact approach saturates.

---

#### Focus Area 2 — Initializer-then-resume patterns

**What Wave 1.A established (not re-covered here):** The two-agent initializer-then-resume pattern is the converged best-practice for long-running coding agents from both academic and practitioner sources. Wave 1.A established that the initializer produces structured scaffolding and subsequent agents read it at session start. Wave 2.B goes deeper on the operational mechanics.

##### How the initializer establishes scaffolding: empirical guidance

The Anthropic harness post provides the most specific published guidance. The initializer's prompt instructs the model to:

1. Create the environment scaffold (init.sh, progress file, git repository)
2. Expand the user's initial request into a comprehensive feature list — in the claude.ai clone example, >200 features with step-by-step verification steps per feature
3. Mark all features as `passes: false` initially, providing the coding agent with a complete map of what "done" means before any implementation begins

The decomposition heuristic is implicit: the initializer must enumerate *all* verifiable behaviors of the target system at a granularity where each is independently testable by a single session. This is the functional decomposition principle — each unit of work must be bounded by the scope of one agent's context window, which in practice means single-feature scope.

The OpenDev (Bui, arXiv:2603.05344) paper makes the decomposition mechanism structural: the Planner subagent operates under a tool schema containing only read-only tools. Write tools are absent from the schema. This schema-level enforcement (not a prompt-level instruction to "only plan, not execute") ensures the Planner cannot accidentally modify state. The produced plan is presented to the user for approval before the coding agent receives full tool access. The structural benefit: the initializer's scaffolding cannot be corrupted by premature execution, independent of how the model interprets its instructions.

The question-pattern the initializer uses is not formally documented in either source. The Anthropic post describes the initializer as being given a "specialized prompt that asks the model to set up the initial environment" — the prompt content is practitioner-derived rather than based on published ablations of question-pattern effectiveness. No controlled study was found comparing initializer question-pattern variants.

##### How the resume agent reads scaffolding: mechanics

The six-step session-start protocol from the Anthropic harness post (Focus Area 1) is the most specific published description:

1. Confirm working directory
2. Read `claude-progress.txt` (full read, not relevance-retrieved)
3. Read recent git logs
4. Consult `feature_list.json` for highest-priority failing feature
5. Execute `init.sh`
6. Select one feature and work on it

This is a **full-read protocol** for the progress log and a **structured-query protocol** for the feature list (filter by `passes: false`, sort by priority). The working-memory load question is implicitly answered: the progress log and feature list together should fit in the session's usable context budget. The Anthropic guidance does not specify a maximum size for these artifacts, but the feature list schema — a flat JSON array of objects — scales to the hundreds of features in the published example without exceeding standard context windows.

The Codified Context three-tier architecture operationalizes a different reading strategy: hot memory (always loaded), domain specialists (task-class-triggered), cold memory (retrieved on demand). The loading decision is made by the agent via structured calls to the retrieval service, not by a fixed protocol. This is closer to *relevance-retrieved* scaffolding for the domain-specialist and cold-memory tiers, combined with always-loaded for the constitution.

The tension between full-read and relevance-retrieved scaffolding is real: full-read is reliable (no retrieval-quality failures) but does not scale to large state stores; relevance-retrieved scales but introduces the retrieval-quality dependency identified in Focus Area 1. The literature's implicit resolution: use full-read for small, structured, purpose-built handoff artifacts (feature lists, progress logs); use governed retrieval for large, organic, codebase-knowledge stores (codified context, cold memory).

##### Claude Code's five-layer compaction pipeline: operational mechanics

Wave 1.A referenced this pipeline at a summary level. The Finisky Garden independent analysis (2026) provides the most detailed published breakdown, sourced from the Claude Code codebase:

**Layer 0 — Persist large results to disk:** Tool results exceeding 50K characters are written to disk; only a ~2KB preview plus file path enters the context. Trigger: per-tool result size. The Read tool is exempted (threshold set to infinity) to avoid circular dependencies. This is a pre-context-window intervention — it prevents oversized tool outputs from entering the context at all.

**Layer 1 — Cached Microcompact:** Deletes old tool results from server-side cache without invalidating the cached prefix. Trigger: tool call count exceeding threshold, maintaining a most-recent-N keep set. Uses Anthropic's cache_edits API to remove results from specific tools (Bash, Read, Grep, Glob, WebFetch, WebSearch, FileEdit, FileWrite) with "virtually no impact on prompt cache hit rates" due to local message preservation.

**Layer 2 — Time-Based Microcompact:** Clears old tool results when server-side cache has expired due to inactivity (after 60+ minute idle gap, matching Anthropic's cache TTL). Modifies local message content directly, replacing old tool results with placeholder text. Also clears thinking blocks. Skipped if Layer 1 fired; resets its state afterward to prevent phantom deletions.

**Layer 3 — Session Memory Compact:** Generates summaries using continuously-maintained session notes — at zero LLM cost because the notes are already maintained. Trigger: dual thresholds on token count and tool call count. Format: a nine-section markdown summary (current state, tasks, files, workflow, errors, learnings, worklog) with 2,000 tokens per section and 12,000 tokens total budget. If notes are empty or extraction fails, escalates to Layer 4.

**Layer 4 — Full Compact:** LLM-generated comprehensive summary via a forked agent that inherits the main session's cache prefix. Trigger: when the context still exceeds the pressure threshold after all four previous layers have run, or when Layer 3 fails. Cost: most expensive, but mitigated by prompt-cache sharing. Circuit breaker: three consecutive failures prevent wasteful retries. Observed failure rate: ~2.79% on Sonnet 4.6 vs. 0.01% on Sonnet 4.5. Historical waste from pre-circuit-breaker era: 1,279 sessions had 50+ consecutive failures, collectively wasting ~250,000 API calls per day.

**Adoptability for other agentic systems:** The pipeline is structured around two implementation-specific features — Anthropic's cache_edits API and the Claude Code codebase's internal session notes infrastructure. Layers 0, 3, and 4 are structurally portable to other systems (pre-context size caps, notes-based free summaries, LLM-generated semantic summaries as last resort). Layers 1 and 2 depend on server-side cache management that requires vendor-specific API support. The key architectural principle is portable: cheapest-first, semantic summarization as last resort.

For llm-orc specifically, the five-layer pipeline maps partially onto the Session Registry and Conversation Compaction noted in the system design. The cycle's architecture commits to Conversation Compaction in L2 (Orchestrator Runtime is "aware of Routing Decisions and Conversation Compaction") but does not define a pipeline with explicit layer ordering. Claude Code's pipeline is an existence proof that the ordering matters significantly — the 250,000-API-call/day waste from pre-circuit-breaker Layer 4 failures is the cost of not having the cheaper layers upstream.

##### Anthropic's "context engineering" and "progressive disclosure"

The Anthropic context engineering post (2025) operationalizes just-in-time retrieval: agents "use tools to dynamically load information at runtime" rather than pre-loading context. The mechanism is a set of lightweight references (file paths, URLs, section identifiers) that the agent follows as needed, rather than pre-loading all relevant data upfront.

The "progressive disclosure" pattern (Griciūnas, newsletter.swirlai.com, 2026) formalizes this into a system design principle: agents maintain a lightweight index of capabilities (skills), pull in full details when needed, and keep context lean by default. The quantitative claim: median discovery cost is ~80 tokens per skill, and all 17 Anthropic skills together cost ~1,700 tokens — meaning an agent can be aware of dozens of skills for less context than a single activated skill requires. This is the skills-as-index pattern, operationalized as a form of just-in-time context loading.

The agent-side context-discovery pattern (Bui, arXiv:2603.05344) demonstrates this in OpenDev: the agent uses read-only tools to explore the codebase and load relevant files before making any plan, rather than pre-loading a snapshot of all files. The distinction from full-read scaffolding: full-read scaffolding is appropriate for *structured, purpose-built handoff artifacts* of bounded size; progressive disclosure is appropriate for *large, organic knowledge stores* where the agent must decide what it needs.

**Settled:** The initializer-then-resume pattern's operational mechanics are well-documented from practitioner and practitioner-academic sources. The six-step session-start protocol, the feature-list schema with boolean monotonic status, and the append-only progress log format are the canonical reference implementation. Claude Code's five-layer compaction pipeline is the canonical reference for context management within sessions.

**Converging:** Progressive disclosure and just-in-time retrieval are converging as the right approach for large knowledge stores, distinct from full-read scaffolding for small handoff artifacts. The skills-as-index pattern is new (Anthropic, December 2025) but rapidly adopted.

**Open:** Devin (Cognition Labs) explicitly does not maintain cross-session memory as of mid-2025 — the 2025 performance review states that "Devin does not maintain long-term memory across sessions" and that "reasoning about a codebase is bounded by what it can load into its context window during a given session." Cognition's 2026 roadmap names cross-session codebase understanding as a priority. This is a notable gap: the most prominent deployed coding agent does not implement the initializer-then-resume pattern for session continuity. The literature's converged recommendation is not yet the deployed default.

---

#### Focus Area 3 — Calibration-gated cross-layer composition

Wave 1.A covered the Calibration Gate (ADR-007) as a commitment not yet exercised in production, and Cleanlab's tau2-bench trust-score fallback finding. Wave 2.B surveys published mechanisms for calibration-gated dispatch.

##### Uncertainty-aware composition: the published landscape

Two papers from January 2026 directly address this surface.

**Agentic Uncertainty Quantification (arXiv:2601.15703)** proposes the Dual-Process AUQ framework: a training-free approach with two complementary mechanisms.

*System 1 (Uncertainty-Aware Memory):* Verbalized confidence scores and semantic explanations are retained in the context window. This creates a soft constraint via the Transformer's self-attention mechanism — when attention heads process uncertainty expressions, the generative distribution shifts toward exploration rather than exploitation. The mechanism is implicit (attention-mediated) rather than explicit (rule-triggered).

*System 2 (Uncertainty-Aware Reflection):* A binary switching function S(h_t) = 𝕀(ĉ_t < τ) governs mode selection. When confidence ĉ_t falls below threshold τ (typically 0.8–1.0), a reflection operator generates N parallel reasoning paths using the initial explanation as a rational cue. If resolution fails within the available memory budget, *adaptive memory expansion* triggers full-context retrieval.

Key architectural feature: low confidence does not block execution. The switch activates reflection but the agent continues executing. This is uncertainty-modulated deliberation, not uncertainty-gated blocking.

Quantitative results: ALFWorld +10.7% success rate (74.3% vs. baseline ReAct); WebShop +13.6% (42.9%); DeepResearch 52.09 overall score, outperforming enterprise systems at 49.71.

**Agentic Confidence Calibration (arXiv:2601.15778)** introduces Holistic Trajectory Calibration (HTC): rather than calibrating single-turn outputs, HTC extracts process-level features across an agent's entire trajectory — macro dynamics and micro stability — to assess confidence calibration at the trajectory level. The General Agent Calibrator (GAC) variant achieves "the best calibration (lowest ECE) on the out-of-domain GAIA benchmark," which is a cross-domain transfer result. The paper is notable for addressing the trajectory-level calibration problem rather than the output-level calibration problem — an important distinction for the cycle's ADR-007 Calibration Gate.

**The calibration distinction relevant to ADR-007:** ADR-007's Calibration Gate tracks "last-N positive signals" on composed ensembles and transitions to "trusted" only after those signals are all positive. This is an *output-level calibration* mechanism — it observes ensemble outputs and makes promotion decisions based on observed quality. The AUQ and HTC work addresses *decision-level and trajectory-level calibration* — inferring agent confidence during execution rather than after. These are complementary but distinct: ADR-007 is post-hoc signal tracking; AUQ/HTC are in-process confidence estimation. A fully realized calibration infrastructure would layer both.

##### Trust-score-thresholded execution: published mechanisms

OI-MAS (Wang et al., arXiv:2601.04861, 2026) operationalizes confidence-gated model-tier escalation via a mathematical objective:

`min E[−r(q,a;ϕ,ψ) + Σt λ · Confadj(st) · C(rt,mt)]`

The adjusted confidence score (normalized token log-probability) modulates the cost penalty: higher confidence increases cost penalties to discourage unnecessary escalation to larger models. Lower confidence triggers escalation to larger-scale models. This is a *continuous* threshold mechanism, not a binary pass/fail gate — the cost penalty gradient ensures that the system routes to cheaper models by default and escalates only when confidence drops.

Quantitative result: 17–78% inference cost savings across benchmarks, +12.88% accuracy improvement over baselines, 23.12s per query vs. 36.82–39.31s for competing multi-agent systems.

The OI-MAS confidence gating is the closest published analogue to what ADR-007's Calibration Gate would need to do if it were extended beyond post-hoc quality checking to in-process routing decisions. The key difference: OI-MAS gates model-tier selection at the role level; ADR-007 gates ensemble promotion at the composition level. The structural logic is the same (confidence below threshold triggers escalation), the application surface differs.

##### Calibration training for agentic systems

No reviewed paper explicitly addresses training agents to produce well-calibrated uncertainty estimates over their own *decisions* (as distinct from over their own outputs). The AUQ framework is training-free — it uses verbalized confidence (the model's own expressed uncertainty) rather than trained calibration heads. HTC is a diagnostic framework that assesses calibration post-hoc rather than training it in.

The progressive confidence estimation paper (arXiv:2604.05952, 2026) applies confidence scoring to research report generation via multi-hop reasoning — grounding confidence in verifiable evidence retrieval. This is the closest to calibration over decisions rather than outputs, but it operates in a specific domain (report generation with evidence traces) that does not generalize to arbitrary agentic dispatch decisions.

Reflexion (Shinn et al., NeurIPS 2023) is the most widely cited approach to calibration-through-verbal-post-mortem. It achieves 91% pass@1 on HumanEval through episode-level verbal reflection that identifies wrong decisions and recommends alternatives. Wave 1.A already covered Reflexion's scope limitation: it is validated on single-problem coding challenges, and Khanal et al.'s finding that episodic memory scaffolds universally hurt at longer horizons raises a transfer-cost question. No 2025–2026 paper has tested Reflexion's verbal post-mortem at multi-session RDD-cycle scale.

##### Calibration as cross-layer primitive: what the literature says

ADR-007 sits in L1 (Domain Policy) and gates promotion of L0 (Ensemble Engine) artifacts. The cycle's question is whether there is published work on calibration-gated cross-layer composition — script outputs gating LLM ensemble execution, LLM ensemble outputs gating orchestrator decisions.

The search for this specific pattern found no direct analogues. The closest is the OI-MAS confidence-gated model routing (gating model-tier selection within a single layer) and the CAAF state-locking pattern from Wave 1.A (gating constraint re-evaluation across execution steps). Neither is exactly cross-layer gating in the four-layer architecture sense.

The HTC paper's cross-domain GAC result is suggestive: a calibration model trained on one agent framework transfers well to others (out-of-domain GAIA benchmark). This implies that calibration mechanisms are more portable across agent architectures than previously assumed, which is relevant for a cross-layer gate implementation — the calibration model could be trained on L0 ensemble outputs and applied to L1 dispatch decisions without per-layer retraining.

**Settled:** Uncertainty-aware composition exists and is effective (+10.7–13.6% on standard benchmarks via AUQ). Confidence-gated model-tier escalation is published and validated (OI-MAS: 17–78% cost reduction). Trajectory-level calibration is more informative than output-level calibration for long-horizon decisions (HTC). These are settled findings.

**Converging:** The architectural pattern of soft-constraint uncertainty propagation (AUQ System 1) combined with explicit reflection gating (AUQ System 2) is converging as the right design for uncertainty-aware agents, with the training-free property making it deployable without infrastructure investment.

**Open:** Calibration as a *cross-layer* primitive — where the Calibration Gate gates not just ensemble promotion but also in-process routing decisions across layers — is not published. The literature provides components (AUQ, HTC, OI-MAS confidence routing) that could be composed into a cross-layer gate, but no system has published this composition. This is the cycle's novel territory for ADR-007.

---

#### Focus Area 4 — Per-role model configurability patterns

Wave 1.A covered OpenDev's per-workflow LLM configurability (different models for thinking, critique, visual, fallback) and the local-model capability task-class boundary. Wave 2.B goes deeper on the published mechanisms for role-specific model assignment and the heterogeneous vs. homogeneous tradeoff at the role-staffing level.

##### Multi-model agent systems: published evidence on role-specific assignment

The OI-MAS framework (Wang et al., arXiv:2601.04861) implements the most formal published mechanism for role-specific model assignment. The conductor component applies a two-stage routing:

1. **Role Routing:** A learnable role network evaluates the current reasoning state (query + context) and computes role probabilities. Roles accumulate probability mass until a threshold is reached, enabling single or multiple role activation per turn.
2. **Model Routing:** Once roles are selected, a model network assigns each role an LLM backbone by evaluating suitability across the available model pool.

The confidence-aware objective modulates model-tier selection: lower confidence triggers escalation to larger-scale models, enforcing the principle that expensive models should only be used when cheaper models are unlikely to succeed. The 17–78% cost reduction across benchmarks demonstrates that this selective escalation is effective in practice.

SC-MAS (Hu et al., arXiv:2601.09434, 2026) operationalizes a different framing: edges in the multi-agent directed graph carry collaboration strategy labels, allowing different agent *pairs* to interact through tailored communication patterns rather than enforcing uniform interaction across all agents. The motivation is Social Capital Theory — "different roles benefit from distinct forms of collaboration." The MMLU result (+3.35% accuracy, −15.38% cost) demonstrates that role-pair-specific collaboration is more efficient than uniform collaboration.

MasRouter (Ma et al., ACL 2025 Findings, arXiv:2502.11133) is the only reviewed paper at a major NLP venue that directly addresses role assignment in multi-agent routing. Its cascaded controller performs three sequential decisions: (1) collaboration mode selection, (2) agent role allocation, and (3) LLM backbone assignment per role. The ordering matters — collaboration mode is determined before role assignment, because the right roles depend on the collaboration topology. Result: 1.8–8.2% improvement on MBPP, 52% cost reduction on HumanEval.

Topaz (Pan et al., arXiv:2604.03527, 2026) addresses the explainability dimension. It decomposes model capabilities and task requirements into a shared eight-skill taxonomy (mathematical reasoning, logical reasoning, code generation, tool use, factual knowledge, writing quality, instruction following, summarization). Each routing decision is traceable through numerical skill-match scores, producing local rationales (why this model for this task) and global summaries (overall routing strategy). At zero cost sensitivity, Topaz routes premium models to complex reasoning tasks; at strict cost constraints, it preserves high-capability models for high-sensitivity operations while economizing elsewhere. The finding that "efficiency gains stemmed from capability saturation rather than hidden quality loss" is the key result: routing cheaper models to tasks where expensive models add no marginal value is not a quality compromise.

**The Spike A3 heterogeneity finding extended to role-staffing:** Cycle 3's Spike A3 found that two reviewers from different model families (Tencent Hunyuan + Moonshot Kimi) produced 5–8 distinct findings each with only 1–2 overlap — the heterogeneity-uncorrelated-errors mechanism directly observed. The SC-MAS and OI-MAS findings provide theoretical and quantitative support for this observation at the role-staffing level: heterogeneous role-staffing (different models for different roles) is more efficient than homogeneous role-staffing (same model for all roles) both on accuracy and cost.

##### Capability-aware routing: how role assignment is constructed

The published systems converge on a common construction pattern with three steps:

1. **Skill/capability profiling** (Topaz: eight-skill taxonomy; OI-MAS: learnable role network; MasRouter: collaboration mode prior followed by role priors)
2. **Task decomposition** (what does this query require across the skill profile?)
3. **Cost-performance optimization** (match capability to requirement, penalizing over-allocation)

The Belcak and Heinrich position paper (arXiv:2506.02153, 2026) argues that SLMs are the right default for agentic tasks that "perform a small number of specialized tasks repetitively and with little variation" — exactly the task class that ensemble members in the cheap-orchestrator pattern occupy. The implication for role staffing: roles that are repetitive and bounded (single-file analysis, template application, structured data extraction) are SLM-viable; roles that require sustained multi-turn reasoning or large-context synthesis are not.

The Codified Context three-tier architecture operationalizes this at the agent level: domain specialist agents (Tier 2) are small, task-specific, and carry project-specific knowledge as embedded context rather than requiring the model to generalize from first principles. This is capability-aware role staffing implemented through agent specification rather than model routing.

##### Homogeneous vs. heterogeneous role-staffing: the tradeoff evidence

The literature is clearer on this question than Wave 1.A's territory (which addressed heterogeneity at the *content* level — same role, different model family, uncorrelated findings). Wave 2.B addresses heterogeneity at the *role* level — different models for different functional roles.

The findings from SC-MAS, OI-MAS, and MasRouter collectively establish that:

- Heterogeneous role-staffing (different models for different roles) uniformly outperforms homogeneous role-staffing on both accuracy and cost across the reviewed benchmarks.
- The mechanism is capability saturation: homogeneous systems over-allocate expensive capability to tasks that do not require it.
- The binding constraint for heterogeneous systems is the role assignment mechanism — poorly designed role assignment wastes the heterogeneity advantage by routing incorrect capabilities to tasks.
- Memory engineering is what makes heterogeneous teams viable for small models: "small models can't maintain the context required for coordination on their own. They rely on external memory to participate in larger workflows" (O'Reilly, 2026). This directly connects Focus Area 1 (externalized state) and Focus Area 4 (per-role configurability) — they are not independent design choices.

The negative case: the OI-MAS confidence-gating mechanism prevents unnecessary escalation to large models (higher confidence → cost penalty discourages expensive models). This is the cost-control mechanism that prevents heterogeneous role-staffing from degenerating into "always use the largest model per role."

##### Per-role configurability under the local-first commitment

Wave 1.A's operationalization — local models amplify deterministic and bounded-scope tasks; cheap-cloud-orchestrator handles routing/summarization/sustained-reasoning — is supported by the reviewed evidence at the role-staffing level.

The O'Reilly practitioner analysis (2026, source 22 synthesis) is explicit: "Genuine multi-agent value comes from heterogeneity. Different models with different capabilities operating at different price points for different subtasks... Memory engineering makes heterogeneous teams viable; without it, every agent must be large enough to maintain full context independently, which defeats the cost optimization that motivates heterogeneity."

This directly names the mechanism for the cycle: the cheap-orchestrator + local-ensemble pattern achieves heterogeneous role-staffing by assigning the orchestrator to routing/sustained-reasoning roles (cloud, higher capability) and ensemble members to bounded-scope roles (local, bounded context, repetitive task classes). The externalized state infrastructure (Focus Area 1) is what makes the ensemble members' local context limitation acceptable — each member receives structured, purpose-built input rather than accumulated multi-turn context.

ADR-011's "default-not-ceiling" reading from Essay 003 is directly supported by the OI-MAS and MasRouter evidence: the right default is the cheapest capable model for the current task; escalation to more capable (and more expensive) models is triggered by confidence failure, not by task assignment at session start.

**Settled:** Heterogeneous role-staffing (different model capabilities for different functional roles) outperforms homogeneous role-staffing on cost and accuracy across all reviewed benchmarks. The mechanism is capability saturation. The enabling infrastructure is external memory (small models cannot maintain coordination context independently). These findings are settled across multiple 2025–2026 papers.

**Converging:** Confidence-gated model-tier escalation (OI-MAS pattern) is converging as the right mechanism for cost-controlling heterogeneous role-staffing. The pattern is: cheap model by default → confidence falls below threshold → escalate to more capable model. This directly instantiates ADR-011's default-not-ceiling reading.

**Open:** Published evidence on *local vs. cloud* role assignment specifically — which roles belong on local hardware and which on cloud — remains thin at the experimental level. The practitioner boundary (bounded, repetitive, short-context tasks → local; routing, summarization, sustained reasoning → cloud) is practitioner-documented but not peer-reviewed. No reviewed paper directly tests the cheap-orchestrator + local-ensemble pattern as a specific role-staffing configuration. This remains the cycle's empirical territory.

---

### Key Findings

**Focus Area 1 — Externalized state primitives:**

- The Anthropic initializer pattern produces three typed artifacts: `feature_list.json` (200+ entries, boolean monotonic status, category + description + steps schema), `claude-progress.txt` (append-only free text, session-level granularity), and `init.sh` (deterministic environment bootstrap). The monotonicity constraint on `feature_list.json` is a structural non-regression enforcement — not a prompt instruction (Anthropic Engineering, 2026).
- Three-tier hot/domain/cold loading is the validated approach for large-codebase session continuity: 283 sessions, 108K-line C# system, 1,197 agent invocations with no reported context-coherence breakdown (arXiv:2602.20478, 2026).
- MemMachine's nucleus expansion is the best-performing published retrieval mechanism for conversational agent state: 93.0% on LongMemEvalS, 80% fewer tokens than Mem0, with retrieval-stage tuning contributing 4× more improvement than ingestion-stage tuning (arXiv:2604.04853, 2026).
- Plain filesystem outperforms specialized vector-store libraries at 74% on MemoryAgentBench tasks — naive vector-store deployment is not the right investment before simpler structured-artifact approaches saturate (ICLR 2026 Workshop, source 23).
- Append-only JSONL persistence supports audit, forensic traceback, and composable session operations (resume, fork, rewind) — but creates rollback limitations and is vulnerable to memory poisoning if write-gate validation is absent (arXiv:2604.16548, 2026).
- No published paper documents the externalized state schema for *multi-phase workflow phase transitions* (as distinct from single-coding-task session handoffs). What the initializer should encode for RDD-specific phase state is not addressed in the literature.

**Focus Area 2 — Initializer-then-resume patterns:**

- The Anthropic initializer's decomposition heuristic is: enumerate all verifiable behaviors of the target system at independently-testable, single-session-scope granularity, mark all as failing, and constrain subsequent agents to one feature per session. The enforcement is structural (monotonic boolean schema) not instructional (Anthropic Engineering, 2026).
- OpenDev's Planner subagent enforces planning restrictions via tool-schema absence (write tools simply not present in the schema), not via constrained decoding or prompt instructions. This avoids the alignment-tax finding at small model sizes (Wave 1.A, arXiv:2604.06066) (Bui, arXiv:2603.05344, 2026).
- Claude Code's five-layer compaction pipeline is: (0) persist oversized tool results to disk; (1) cached microcompact (delete old cache entries); (2) time-based microcompact (clear after idle); (3) session memory compact (free summaries from continuously-maintained notes); (4) full compact (LLM-generated semantic summary as last resort). Semantic summarization is last resort; the 250,000-API-call/day waste from pre-circuit-breaker failures validates the cheapest-first ordering (Finisky Garden analysis, 2026; arXiv:2604.14228, 2026).
- Layer 3's zero-cost session notes pattern is directly portable to llm-orc: continuously maintain structured markdown notes throughout a session, use them as the compact summary rather than generating a new one. The nine-section template (current state, tasks, files, workflow, errors, learnings, worklog) is a directly adoptable schema for llm-orc's Conversation Compaction mechanism.
- Devin (Cognition Labs) does not implement cross-session memory as of mid-2025; session state is bounded by the context window of a single session. The 2026 roadmap names this as a priority. The most prominent deployed coding agent is thus operating without the infrastructure the literature converges on as best practice (cognition.ai/blog, 2026).
- Confucius Code Agent achieves 59% Resolve@1 on SWE-Bench-Pro with persistent cross-session note-taking, the highest published result on that benchmark among reviewed sources (arXiv:2512.10398, 2025).

**Focus Area 3 — Calibration-gated cross-layer composition:**

- Agentic Uncertainty Quantification (AUQ) provides a training-free confidence-gating mechanism: verbalized confidence propagates through attention (System 1 soft constraint); binary switch S(h_t) = 𝕀(ĉ_t < τ) triggers deliberate reflection when confidence falls below threshold τ = 0.8–1.0 (System 2 explicit gate). Low confidence does not block execution — it triggers reflection. Results: +10.7% on ALFWorld, +13.6% on WebShop (arXiv:2601.15703, 2026).
- Holistic Trajectory Calibration extracts process-level features (macro dynamics + micro stability) across entire agent trajectories, yielding better calibration than output-level metrics. Cross-domain transfer without retraining is validated on GAIA benchmark, suggesting calibration mechanisms are architecturally portable (arXiv:2601.15778, 2026).
- OI-MAS's confidence-gated model-tier routing achieves 17–78% cost reduction by penalizing expensive model choices at high confidence and triggering escalation at low confidence. This is the closest published analogue to what ADR-007's Calibration Gate would look like as an in-process routing mechanism rather than a post-hoc promotion mechanism (arXiv:2601.04861, 2026).
- No published paper implements calibration gating as a *cross-layer* primitive (L0 ensemble output gating L1 dispatch decisions, or L1 script output gating L2 orchestrator routing). The components exist; the composition does not. This is the cycle's novel territory.
- ADR-007's current implementation (last-N positive signals for ensemble promotion) is *output-level, post-hoc* calibration. The literature's converged direction is *trajectory-level, in-process* calibration. A fully realized Calibration Gate would layer both: post-hoc promotion tracking (ADR-007 as implemented) combined with in-process confidence gating (AUQ/OI-MAS pattern) for dispatch decisions within a session.

**Focus Area 4 — Per-role model configurability:**

- Heterogeneous role-staffing outperforms homogeneous staffing consistently: SC-MAS +3.35% accuracy / −15.38% cost on MMLU; MasRouter +1.8–8.2% on MBPP / −52% on HumanEval; OI-MAS +12.88% accuracy / −17–78% cost. The mechanism is capability saturation — homogeneous systems over-allocate expensive capability to tasks that do not require it (arXiv:2601.09434; ACL 2025; arXiv:2601.04861).
- Topaz's eight-skill taxonomy (mathematical reasoning, logical reasoning, code generation, tool use, factual knowledge, writing quality, instruction following, summarization) is a directly adoptable profiling vocabulary for role assignment in the cheap-orchestrator + ensemble pattern (arXiv:2604.03527, 2026).
- External memory is the enabling infrastructure for small-model role-staffing: without it, small models cannot maintain coordination context independently, and every role must be staffed by a large model regardless of task complexity. Focus Area 1 and Focus Area 4 are architecturally coupled (O'Reilly practitioner analysis, 2026).
- ADR-011's default-not-ceiling reading is supported by the OI-MAS confidence-gated escalation pattern: cheap model by default → confidence below threshold → escalate. This is the right operational implementation of the ADR's boundary refinement from Essay 003.
- SLMs are appropriate for roles involving "a small number of specialized tasks repetitively and with little variation" — the task class that ensemble members occupy in the cheap-orchestrator pattern (arXiv:2506.02153, 2026).

---

### Tensions Between Sources

**Tension 1: Full-read vs. relevance-retrieved scaffolding.**

Anthropic's initializer pattern uses full-read for the feature list and progress log (reliable, bounded). The Codified Context architecture uses relevance-retrieved for domain specialists and cold memory (scalable, retrieval-quality-dependent). The resolution is scope: full-read is the right approach for structured, purpose-built, small handoff artifacts; governed retrieval is right for large, organic, codebase-knowledge stores. These are not competing approaches but complementary, addressing different scales of state. The tension becomes real when handoff artifacts grow large (e.g., an RDD cycle's accumulated research state across multiple waves), at which point full-read saturates the context budget and retrieval becomes necessary.

**Tension 2: Append-only persistence — auditability vs. security.**

Append-only JSONL provides auditability and composable session operations (resume/fork/rewind). It is also the attack surface for memory poisoning — poisoned entries written once persist indefinitely and are retrieved alongside legitimate history. The mnemonic sovereignty framework recommends write-gate validation and post-deletion verification as mitigations, but neither is operationalized in Claude Code or any other reviewed system. The tension is not yet resolved in practice; it is theoretically identified (arXiv:2604.16548, 2026).

**Tension 3: Confidence gating — blocking vs. modulating.**

AUQ's confidence gate does not block execution — low confidence triggers reflection but the agent continues. OI-MAS's confidence gate does modulate model selection — low confidence triggers escalation to more capable models. ADR-007's Calibration Gate blocks promotion — low signal quality prevents an ensemble from being trusted. These are three different gating semantics (modulate deliberation, escalate capability, block promotion) applied to different points in the execution flow. The literature does not propose a unified gating model; each is designed for a specific application surface. A multi-point gating architecture for llm-orc would need to implement all three semantics at different layers.

**Tension 4: Memory augmentation — helps at task scale, hurts at session scale.**

Reflexion (NeurIPS 2023) demonstrates +91% pass@1 on HumanEval via verbal post-mortems — strong positive evidence for episodic memory augmentation at single-problem scale. Khanal et al. (arXiv:2603.29231, Wave 1.A) show that episodic memory scaffolds universally hurt long-horizon performance across 10 tested models. MemMachine achieves 93.0% on LongMemEvalS with governed retrieval. The resolution: naive episodic memory augmentation hurts at session scale (Khanal et al.); governed retrieval with nucleus expansion succeeds at session scale (MemMachine); verbal post-mortems help at single-task scale (Reflexion). Task scale is the scope condition, not memory augmentation per se.

---

### Architectural Recommendation: Separable or Additional Layer?

The cycle has not committed to whether long-horizon reliability infrastructure is *separable* (operationalizable within existing layers) or *requires architectural addition* (a fifth layer or cross-cutting module).

**The literature's evidence bearing on this question:**

The four focus areas together identify five infrastructure primitives for long-horizon reliability:
1. Structured handoff artifact store (feature lists, progress logs, constitution files)
2. Session-level compaction pipeline (cheapest-first, semantic summarization last resort)
3. Trajectory-level calibration (in-process confidence tracking)
4. Per-role model router (confidence-gated tier escalation)
5. Write-gated append-only persistence (audit + security)

**Separability analysis by existing layer:**

- Primitive 1 (artifact store) maps to Session Registry (L3) — the Session Registry currently tracks "cumulative turns, token spend" but not structured handoff artifacts. Adding artifact store responsibility to Session Registry is additive within L3, not an additional layer.
- Primitive 2 (compaction pipeline) maps to Orchestrator Runtime (L2) — the system design notes Runtime is "aware of Conversation Compaction." The five-layer pipeline is an elaboration of this existing responsibility, not a new layer.
- Primitive 3 (trajectory calibration) maps to Calibration Gate (L1) — extending from post-hoc output-level tracking to in-process trajectory-level tracking is a significant elaboration of the Calibration Gate's responsibility, but within the same module.
- Primitive 4 (per-role model router) maps to Orchestrator Configuration + Orchestrator Tool Dispatch (L2/L3) — per-role model selection is currently a static configuration at session start (ADR-011). Dynamic per-role routing at decision time would require the Tool Dispatch to consult a routing function at each `invoke_ensemble` call, which is an elaboration of Tool Dispatch's existing interposition logic.
- Primitive 5 (write-gated persistence) maps to Session Registry (L3) — the write-gate would be a guard on the Session Registry's write path.

**The conclusion the evidence supports:** Long-horizon reliability infrastructure is *operationalizable within existing layers* as elaborations of existing module responsibilities, not as an additional layer. None of the five primitives requires a structural break with the four-layer architecture. The architectural risk is not layer count but *responsibility concentration* — the Session Registry and Calibration Gate in particular would absorb significantly expanded responsibilities if all five primitives are implemented. Whether that concentration is acceptable is an architectural design choice the literature does not make for the cycle.

**The exception to this conclusion:** Calibration as a *cross-layer* primitive — where confidence state computed at L0 (Ensemble Engine outputs) flows upward to gate L1 (Calibration Gate dispatch) and L2 (Tool Dispatch routing) — would require an information-flow path that currently does not exist in the four-layer architecture. The layering rule ("edges point from higher layer to same-or-lower; never upward") would need to be extended with a read-only signal channel from L0 to L1 for this to work within the existing architecture. This is the one place where long-horizon reliability infrastructure may require an architectural addition — not a new layer, but a new *upward read* signal path from L0 calibration signals to L1 gate decisions. Whether this violates the spirit of the layering rule is a decision for the DECIDE phase.

---

### Limitations

**Coverage gaps:**

- No peer-reviewed paper directly tests the cheap-orchestrator + local-ensemble pattern as a unified role-staffing configuration. SC-MAS, OI-MAS, and MasRouter test heterogeneous agent systems at cloud model tiers; the local-model dimension is not covered in those papers' experimental setups. The cycle's empirical territory remains the intersection of role-heterogeneity (covered) and local-first deployment (not covered together).

- The initializer artifact schema for *multi-phase, self-referential workflows* (like RDD cycles) is not documented in any reviewed source. Anthropic's schema is designed for single-codebase development tasks. Adapting it to encode wave-dispatch state, framing commitments, pending empirical questions, and gate conditions is cycle-specific design work the literature does not anticipate.

- Memory poisoning mitigations (write-gate validation, post-deletion verification) are theoretically described in the mnemonic sovereignty survey but not operationalized in any reviewed system. If the cycle implements append-only JSONL session state as a multi-session artifact store, the security recommendations from arXiv:2604.16548 are current open research, not deployable off-the-shelf.

- Devin's cross-session memory limitation (as of mid-2025) is stated in a practitioner blog post, not a peer-reviewed analysis. The limitation may have been addressed in a product update between the blog post date and the search date; the cycle should not rely on this finding as a durable characterization of Devin's architecture.

- Claude Code's compaction pipeline is analyzed from third-party source-code analysis (Finisky Garden), not from Anthropic primary documentation. The layer labels and trigger descriptions are reverse-engineered from the codebase and may not reflect Anthropic's canonical terminology.

**Search scope limitations:**

- ICML 2026 workshop proceedings (FAGEN/FMAI on agentic failure modes) are not yet indexed; those proceedings may contain calibration-gating or cross-layer composition work relevant to Focus Areas 3 and 4.
- The memory benchmarks (MemBench, MemoryAgentBench, MemMachine) are recent enough (late 2025 to early 2026) that replication studies and comparative analyses have not yet accumulated.
- The OI-MAS, SC-MAS, and MasRouter papers test on standard NLP benchmarks (MMLU, MBPP, HumanEval), not on long-horizon multi-phase coding workflows. Generalizability to the North-Star benchmark's shape is not established in the papers' experimental evidence.

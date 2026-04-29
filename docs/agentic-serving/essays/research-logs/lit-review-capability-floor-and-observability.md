## Literature Review: Capability Floor and Observability in Small-Model Agentic Orchestration

**Date:** 2026-04-25
**Method:** Systematic literature search — web search across primary sources, benchmark documentation, framework repositories, and Ollama model library. Six structured gaps investigated.

---

### Sources Reviewed

| # | Author(s) | Title | Year | Venue | Relevance |
|---|-----------|-------|------|-------|-----------|
| 1 | Patil, S. et al. | The Berkeley Function Calling Leaderboard (BFCL): From Tool Use to Agentic Evaluation of Large Language Models | 2025 | ICML / Proceedings of Machine Learning Research Vol. 267 | Gap 1 — benchmark methodology and multi-turn findings |
| 2 | Qin, Y. et al. | ToolBench: An Open Platform for Training, Serving, and Evaluating LLMs for Tool Learning | 2023 | ICLR 2024 Spotlight | Gap 1 — large-scale API tool-use evaluation |
| 3 | Yao, S. et al. (Sierra Research) | tau-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains | 2024 | arXiv 2406.12045 / OpenReview | Gap 1 — multi-turn tool-agent-user interaction |
| 4 | Wang, Y. et al. | ToolACE: Winning the Points of LLM Function Calling | 2024 | arXiv 2409.00920 | Gap 1 — description quality, API pool size effects |
| 5 | Red Hat Emerging Technologies | Tool RAG: The Next Breakthrough in Scalable AI Agents | 2025 | Red Hat Blog | Gap 1 / Gap 3 — tool dilution, retrieval as mitigation |
| 6 | Docker Engineering Blog | Local LLM Tool Calling: A Practical Evaluation | 2025 | Docker Blog | Gaps 1, 6 — empirical F1 scores for 21 local models |
| 7 | Sharma, C. et al. | Beyond Black-Box Benchmarking: Observability, Analytics, and Optimization of Agentic Systems | 2025 | arXiv 2503.06745 | Gap 2 — failure mode taxonomy, observability patterns |
| 8 | VILA-Lab | Dive into Claude Code: The Design Space of Today's and Future AI Agent Systems | 2025 | arXiv 2604.14228 | Gap 2 — Claude Code observability, silent failure gap |
| 9 | OpenHands Docs | Observability & Tracing | 2025 | docs.openhands.dev | Gap 2 — OpenTelemetry tracing pattern |
| 10 | MLflow / Databricks | Harness Your OpenHands Agent with AI Observability and Governance | 2025 | MLflow Blog | Gap 2 — OpenHands + MLflow operator-side patterns |
| 11 | Fowler, M. | Function Calling Using LLMs | 2024 | martinfowler.com | Gap 3 — practitioner tool design guidance |
| 12 | Anthropic research, cited | Tool RAG accuracy data | 2025 | Referenced in Red Hat Emerging Technologies blog | Gap 3 — RAG-MCP accuracy improvement data |
| 13 | LangChain | LangGraph Multi-Agent Workflows | 2024 | langchain.com blog | Gap 4 — supervisor, router, hierarchical patterns |
| 14 | Salesforce AI Research | xLAM: A Family of Large Action Models to Empower AI Agent Systems (v2) | 2025 | Salesforce Blog / arXiv 2409.03215 | Gaps 1, 6 — xLAM-2 BFCL performance |
| 15 | Beck, K. | Genie Lessons: Nobody Wants Agents | 2026 | tidyfirst.substack.com | Gap 5 — outcome vs coordination framing |
| 16 | Beck, K. | Genie Wants to Leap | 2025/2026 | tidyfirst.substack.com | Gap 5 — incremental vs leap design for AI tools |
| 17 | Beck, K. | Taming the Genie: "Like Kent Beck" | 2025/2026 | tidyfirst.substack.com | Gap 5 — persona + constraint separability |
| 18 | Beck, K. | Genie Sessions: Optionality | 2025/2026 | tidyfirst.substack.com | Gap 5 — exploration-driven AI development |
| 19 | Ollama / Alibaba | qwen3, qwen3.5 library pages | 2026 | ollama.com/library | Gap 6 — availability and pull commands |
| 20 | Salesforce AI Research | Llama-xLAM-2-8b-fc-r (Hugging Face + Ollama community) | 2025 | huggingface.co, ollama.com/robbiemu | Gap 6 — community Ollama availability for xLAM-2 |
| 21 | IBM | granite3.3, granite4 | 2025-2026 | ollama.com/library/granite3.3 | Gap 6 — Granite availability |
| 22 | SihengLi et al. | A Survey on the Honesty of Large Language Models | 2025 | TMLR | Gap 1 — self-knowledge and calibration in small models |
| 23 | Qwen Team | Qwen3 Technical Report | 2025 | arXiv 2505.09388 | Gap 6 — Qwen3 architecture and agent design |

---

### Synthesis

#### Gap 1 — Tool-Use Capability Evaluation Literature

**BFCL methodology and findings.** The Berkeley Function-Calling Leaderboard (Patil et al., ICML 2025) is the field's defacto standard for function-calling evaluation. Its methodology uses Abstract Syntax Tree (AST) evaluation of structured tool calls across serial, parallel, and multi-turn invocation patterns. The benchmark covers over 2,000 question-function-answer pairs and extends to agentic multi-step scenarios in V4 (2025). The canonical finding from V4: state-of-the-art LLMs excel at single-turn calls, but "memory, dynamic decision-making, and long-horizon reasoning remain open challenges." This directly matches the cycle's S0 finding — qwen3:14b drove the cascade but called `list_ensembles` twice (turns 3 and 5) despite the prior result being in its message history.

The most important data point for this cycle comes from an independent practical evaluation (Docker Engineering Blog, 2025) that tested 21 models including local models on a tool selection task using F1 score. Key numbers for the ≤14B class: Qwen3 14B scored F1=0.971 (comparable to GPT-4 at 0.974), Qwen3 8B scored F1=0.933. Specialist models tested for comparison: XLam 8B scored F1=0.570, Watt-tool 8B scored F1=0.484. These numbers are striking because they show the Qwen3 family significantly outperforms specialist function-calling models in real-world practical evaluation, contradicting the expectation that specialist training is the dominant factor. Neither xLAM-2 (released April 2025) nor mistral-nemo:12b appeared in this particular evaluation's tabulated scores, but the overall pattern — Qwen3 significantly above all other tested 8B-class models — is consistent with the cycle's empirical findings.

The BFCL paper does not explicitly measure the specific failure mode the cycle identified as "fast-confabulation" — a model claiming to have queried state without any actual tool dispatch occurring. BFCL's AST evaluation checks the structure of tool calls that do occur; it has no mechanism for detecting confident fabricated prose that never produces a tool call at all. This is a genuine gap in the benchmark landscape: the failure mode that caused the most damage in this cycle (PLAY, CAP-5) is not a failure mode the standard benchmarks track.

**ToolBench and tau-bench.** ToolBench (Qin et al., ICLR 2024 Spotlight) operates at a different scale — 16,464 real-world REST APIs across 49 categories — and is primarily a training/serving platform, not just an evaluation framework. Its ToolEval metric measures Pass Rate (proportion of successfully completed instructions) and Preference (pairwise comparison). The scale differentiation from BFCL is notable: ToolBench is oriented toward API ecosystem discovery and multi-step planning, while BFCL measures precision of individual call structure. Neither directly measures the inner-surface-vs-outer-surface confusion the cycle surfaced in CAP-1 (model choosing a semantically similar client tool over the correct internal tool).

Tau-bench (Sierra Research, 2024, arXiv 2406.12045) is the closest to this cycle's actual test conditions: it simulates real multi-turn conversations between a user (played by an LLM) and a tool-using agent in constrained domains (retail, airline customer service). Key finding: even GPT-4o succeeds on less than 50% of tasks, and agents are inconsistent across trials. This framing — policy-constrained multi-turn tool use — is structurally analogous to llm-orc's orchestration surface (the orchestrator must follow the system prompt's routing guidance while navigating a multi-turn tool cascade). The implication for the cycle: tau-bench's sub-50% success rate for GPT-4o on constrained domains sets a realistic upper bound for what any model will achieve under real conditions. The cycle's qwen3:8b with biased prompt result (CAP-3: full cascade engagement, 4 retries, graceful fallback) represents a qualitatively successful run even though it did not complete the underlying code-review task.

**The ToolACE paper** (Wang et al., arXiv 2409.00920v2) provides the most directly relevant empirical finding on surface-size effects. It demonstrates that the loss (difficulty) of a tool-selection sample is positively correlated with the number of candidate APIs available, the number of APIs the correct answer requires, and the dissimilarity between query and API descriptions. In their dataset expansion from 6 to 30 API clusters, a trained 8B model improved markedly. This provides theoretical grounding for the CAP-1 observation: qwen3:8b failed when 16 tool schemas competed (5 internal + 11 client), but succeeded in CAP-2's isolated test with fewer competing descriptions. The literature names this phenomenon but does not use the phrase "tool surface dilution"; the closest language is "API pool size effects" and "candidate API overload."

**Self-knowledge and calibration.** Research on LLM self-knowledge (TMLR 2025 survey on honesty) characterizes the failure in familiar terms: LLMs that appear confident are not reliably calibrated between right and wrong answers. Critically, "on many queries where current LLMs fail, they still possess the underlying knowledge needed to arrive at the correct response but are unable to correctly elicit and draw inferences." The CAP-5 mistral-nemo finding maps exactly here: the model may have encountered `ai-detect` semantics somewhere in its training corpus (the real Claude Code `/ai-detect` skill), could not calibrate whether that was relevant to this query context, and produced a confident confabulation rather than acknowledging uncertainty. The literature does not offer a practical detection solution — it identifies calibration failure as a training-time problem, not something observable at inference time without ground-truth comparison.

#### Gap 2 — Observability Patterns in Shipped Agentic Systems

**Claude Code.** The VILA-Lab research paper on Claude Code's architecture (arXiv 2604.14228, 2025) explicitly identifies "Silent Failure and the Observability–Evaluation Gap" as an open research direction. The paper notes the system "offers limited mechanisms that explicitly surface when recovery has occurred" — and that agents "tend to respond by confidently praising the work, even when quality is mediocre." This language is near-identical to the cycle's characterization of fast-confabulation. The architecture exposes event types (StreamEvent, RequestStartEvent, ToolUseSummaryMessage, PreToolUse hooks) via an AsyncGenerator pattern to the UI layer. The principal hierarchy (Anthropic → operator → user) is encoded in permissions and hooks, but the paper does not describe operator-specific observability surfaces that are distinct from what users see. The authorization pipeline (PreToolUse, PermissionDenied hooks) provides some signal, but not the specific signal needed: "this response claims a tool was called, but no tool call event appeared in the stream."

**OpenHands.** OpenHands exposes observability primarily through OpenTelemetry (OTEL) tracing to external backends (MLflow, Laminar, Datadog, Honeycomb). Events traced automatically: agent execution steps (via `agent.step` spans), tool calls with input/output capture, LLM API calls through LiteLLM, and browser automation. The hierarchy is: conversation → runs → steps → LLM completions and tool executions. The form of surfacing is operator-side only — traces go to external backends, not in-stream to users. The MLflow integration specifically captures prompt/response pairs, token usage, and latency per step, plus quality scorers for correctness and relevance. Notably, this is a detection mechanism rather than a prevention mechanism: operators define quality thresholds and review results; confabulation is not specifically flagged.

**The operator/user distinction across the ecosystem.** What the literature shows is a consistent pattern: operator-side observability has been professionalized (OTEL traces, MLflow spans, structured audit logs), while user-side observability remains limited to what the interface exposes in-stream. No system reviewed exposes a dedicated signal for "the model claimed to have done something that the tool dispatch log shows did not happen." This gap is consistent across Claude Code, OpenHands, and the general agentic AI observability literature (arXiv 2503.06745). The practical consequence: fast-confabulation (CAP-5's failure mode) would not be detectable by any standard observability stack without specifically comparing the LLM's prose claims against the tool dispatch event log.

**The diagnostic-truthfulness vs coordination-burden tension.** Beck's framing (Gap 5) is not directly addressed in the observability literature, but the architecture of Claude Code's hook events implicitly resolves it in the direction of coordination-burden reduction: users see actions in their terminal as they happen (in-stream surfacing), but the detail is coarse enough not to require active management. OpenHands pushes detail to operators via external backends. Neither system attempts the more precise resolution Beck names as desirable: visibility that is honest about what the model is doing without making the user manage it.

#### Gap 3 — Tool-Design Literature

**Schema verbosity and description salience.** The practitioner literature (Martin Fowler, analytics practitioners, OpenAI community) consistently recommends clear, purpose-statement-first tool descriptions: "a purpose line, a couple of crisp examples, and argument types that leave no room for guessing." This is qualitative guidance, not empirically measured. The ToolACE paper provides the closest empirical data: the loss of a training sample is positively correlated with "dissimilarity between the user query and API descriptions." The implication for design: the more a tool description sounds like the user's actual query, the more reliably the model selects it. The CAP-3 biased system prompt operated on exactly this principle — it added trigger words ("ensembles", "available", "compose", "llm-orc") that directly matched the query's language.

**System prompt vs tool description quality.** The cycle's empirical evidence (CAP-1 vs CAP-3) demonstrates that system prompt steering is a stronger lever than tool description quality for a fixed model capability tier, at least for the surface-confusion failure mode. The broader literature supports this observation: practitioner guides and the LangChain documentation both note that "system prompts describing tools are important complements to tool schemas." What the literature lacks is a controlled head-to-head experiment isolating system prompt vs tool description as variables while holding model and task constant. The CAP-2/CAP-3 results together suggest system prompt dominates when the model is already capable of understanding descriptions, but tool description quality may matter more at lower capability tiers where the model relies more heavily on pattern-matching against schema syntax.

**Tool surface dilution.** The literature provides strong convergent evidence that dilution is real and significant. The Red Hat Tool RAG article (2025) reports that as toolsets grow "into the dozens, hundreds, or even thousands," model performance degrades: "The model's context window gets overloaded. It struggles to distinguish between similar tools. Hallucinations increase." Cited Anthropic research shows that a basic retrieval strategy boosted tool selection accuracy from 13% to 43% in a large toolset — a 3.3x improvement — while cutting prompt length in half. The general research finding is that "some models experience more than 20% reductions in accuracy" when facing multiple competing task formats simultaneously.

The cycle's 16-tool surface (5 internal + 11 client) is well within the range where dilution effects are documented to appear. The direct implication for RQ-1 is that the "capability floor" is not purely a model property — it is a (model × surface size × description salience) function. A model that fails at 16 tools may succeed at 6 tools if the competing descriptions are removed. This is exactly what CAP-2's isolated test confirmed.

**Few-shot examples in tool descriptions.** The ToolACE paper and practitioner guides both suggest few-shot examples in tool descriptions improve reliability, particularly for small models. This was not tested in the cycle and represents a specific variable the cycle did not yet instrument.

#### Gap 4 — Multi-Agent Orchestrator Patterns

**LangGraph.** LangGraph's canonical patterns are: Supervisor (LLM-powered coordinator routes work to specialist agents), Router (rule-based routing on output conditions), and Hierarchical (nested subgraphs as sub-agents). The LangGraph documentation directly names the reason cognitive splits help: "Grouping tools/responsibilities can give better results. An agent is more likely to succeed on a focused task than if it has to select from dozens of tools." This is the structural rationale for what the cycle tested in CAP-2's router-executor pattern. The CAP-2 finding — single qwen3:8b agent picks correctly (7.73s) and router-executor also picks correctly (19.64s) when conditions are favorable — aligns with the LangGraph literature: structural composition is always a valid lever, but it only earns its latency cost when single-agent selection fails under realistic conditions.

The documented latency overhead for the supervisor pattern is significant: in one production case study, the supervisor's routing calls accounted for over 30% of total response time. For the cycle's hardware tier (consumer machine where qwen3:14b is the ceiling), a supervisor or router stage would add roughly another full model turn per routing decision, compounding the already-problematic latency profile S0 observed (32s/turn at 14B, 100s/turn at 14B for the first tool-calling turn).

**AutoGen.** AutoGen (now AG2 in v0.4) uses GroupChat as its primary multi-agent coordination pattern. The documented latency cost of the conversational pattern is explicit: "Every agent turn in a GroupChat involves a full LLM call with the accumulated conversation history. A 4-agent debate with 5 rounds is 20 LLM calls minimum." This makes AutoGen the framework most directly comparable to what the cycle observed in the S0 cascade — the 22-minute wall-clock for a 7-turn qwen3:14b session is exactly what the AutoGen documentation warns about at consumer-hardware inference speeds.

**CrewAI.** CrewAI uses Process.hierarchical for planner-executor splits, with a designated manager agent acting as the planner. Reported performance advantage: CrewAI executes 5.76x faster than LangGraph in certain task benchmarks while achieving higher evaluation scores. This advantage is attributed to avoiding the overhead of graph traversal and stateful checkpointing. The implication is that at consumer hardware scales, the dominant latency variable is the LLM call time, not framework overhead — the framework cost is noise relative to model inference time.

**The fundamental tradeoff.** The multi-agent orchestration literature converges on a consistent tradeoff: cognitive splits improve selection accuracy and reliability on complex, specialized tasks, but add at minimum one full LLM call per routing decision, with accumulating context cost at each additional agent turn. For the cycle's deployment context (consumer hardware, 32-150s per LLM turn), each structural split in the orchestrator architecture multiplies wall-clock time substantially. The CAP-2 finding (2.54x latency for router-executor vs single-agent) is consistent with the documented overhead ratios in the LangGraph production case studies.

The literature does not explicitly address the specific variant the cycle explored: an orchestrator-as-ensemble (ADR-011 reopening). The closest pattern is LangGraph's hierarchical pattern where a top-level agent treats subgraphs as tools. The cycle's pre-S0 finding that "the orchestrator is the only llm-orc role that cannot be composed" is a genuine architectural gap relative to the LangGraph model — but the empirical finding from CAP-3 (system prompt clears the floor without structural change) suggests this gap is not currently load-bearing.

#### Gap 5 — Beck's Genie Lessons Trajectory

Beck's Genie series on tidyfirst.substack.com spans several posts. What is publicly retrievable includes: "Genie Sessions: Optionality" (January 2026), "Taming the Genie: Like Kent Beck" (circa late 2025/early 2026), "Genie Wants to Leap" (circa late 2025/early 2026), and "Genie Lessons: Nobody Wants Agents / Genie Lesson #5" (April 23, 2026). Earlier numbered Genie Lessons (#1 through #4) are retrievable as titles via the tag page but their full content required paid access and is not confirmed here.

**"Nobody Wants Agents"** (the load-bearing post for this cycle) argues that the real human need is outcome-specification: "I have a system and I want it to change. That's the whole thing." Multi-agent infrastructure is a feature, not an outcome. Beck observed himself becoming a manager of agents rather than a developer — holding state in his head that the system should hold for him, watching which agent was doing what, wondering when to interrupt. This is the coordination-burden failure mode RQ-3 directly addresses. Beck does not resolve it with design recommendations; he names it as a category error in how multi-agent products are framed.

**"Genie Wants to Leap"** addresses AI tools that attempt dramatic transformations rather than incremental steps, producing "complexity cliffs" where the system breaks. Beck advocates for the parallels strategy — running old and new implementations simultaneously to prevent abrupt transitions. The relevance to this cycle: a cascade orchestrator that attempts too many tool calls in a single session hits the same complexity-cliff dynamic — context grows, models start confabulating, the system degrades rather than progressing. The CAP-5 confabulation can be partially read through this lens: the model was never given incremental validation steps that would have exposed the fabrication before it crystallized into a final response.

**"Taming the Genie: Like Kent Beck"** found that persona prompts alone improved style but not architecture. Beck's conclusion: combine persona guidance with explicit architectural constraints for reliable behavior change. The structural analog in this cycle: the biased system prompt (CAP-3) combined persona-like framing (internal-tools-first) with explicit constraint-like trigger words and decision heuristics. Beck's finding supports the cycle's CAP-3 result — the prompt's effectiveness came from explicit constraint, not persona alone.

**"Genie Sessions: Optionality"** frames AI tools as best for exploratory, option-generating work where developers can try many approaches and discard most of the results. This is less directly relevant to the cycle's orchestration questions.

**What Beck's series does NOT address:** Local models, capability gradients between model sizes, default-experience design, or the operator/user observability distinction. The series is entirely written from the perspective of a developer using hosted frontier models (Claude Code, Intent by Augment Code). The cycle's RQ-1 through RQ-4 questions — which arise from the specific constraint of local-only non-frontier models — are outside Beck's framing. The Beck framing is still load-bearing for RQ-3 (observability) and RQ-5 (dual-contract framing), but does not inform the capability-floor questions.

#### Gap 6 — Recent Small Models for Tool-Calling via Ollama

**The Qwen3 family (released April 29, 2025).** The cycle's empirical anchor: qwen3:14b (F1=0.971 per Docker evaluation) and qwen3:8b (F1=0.933) are the two best-performing ≤14B models in the most rigorous practical evaluation found. Qwen3-8B "outperforms Qwen2.5-14B on 15 benchmarks" and scores 65.1 on Tau2-Bench for agentic tasks (14B variant). Both models support native tool calling via Ollama's OpenAI-compatible endpoint. Qwen3:8b runs at approximately 25 tokens/second on a laptop. Both are available via `ollama pull qwen3:8b` and `ollama pull qwen3:14b`. The Qwen3 series was trained on approximately 36 trillion tokens with explicit agentic optimization.

**The Qwen3.5 small series (released March 2, 2026).** Alibaba released Qwen3.5 small models (0.8B, 2B, 4B, 9B) on March 2, 2026. All four support native tool calling, thinking, and multimodal capabilities in Ollama per Ollama's official announcement. Sizes: qwen3.5:0.8b (1.0 GB), qwen3.5:2b (2.7 GB), qwen3.5:4b (3.4 GB), qwen3.5:9b (6.6 GB). All use 256K context windows. Qwen3.5-9B reportedly matches or surpasses GPT-OSS-120B on several benchmarks including GPQA Diamond (81.7 vs 71.5). BFCL-specific scores for these small Qwen3.5 models are not yet widely published, but the family lineage from Qwen3 makes this the highest-confidence family recommendation within the hardware constraint. Pull commands: `ollama run qwen3.5:9b`, `ollama run qwen3.5:4b`.

**Salesforce xLAM-2 (released April 2025).** xLAM-2 is specifically trained for function-calling via the APIGen-MT framework, which generates training data from simulated agent-human interactions. Claimed performance: xLAM-2-8B-fc-r reached top-4 on BFCL at release. However, in the Docker practical evaluation, XLam 8B scored F1=0.570 — significantly below Qwen3-8B (0.933). This gap between BFCL performance and practical evaluation warrants caution: BFCL's AST methodology may not capture the same failure modes as a practical tool-selection test. The xLAM-2-8b-fc-r model is available on Ollama via community quantizations published by user `robbiemu`: `ollama pull robbiemu/Salesforce_Llama-xLAM-2:8b-fc-r-q5_K_M`. This is a community repo, not an official Ollama library entry — verify the model hash before trusting it in production. Risk assessment: the practical evaluation F1 score (0.570) suggests xLAM-2-8B may exhibit the wrong-tool-path failure mode rather than the confabulation failure mode, but this is tentative.

**IBM Granite 3.3 and Granite 4 (2025-2026).** Granite 3.3 and 4 are available in Ollama's official library (`ollama pull granite3.3:8b`, `ollama pull granite4`). IBM highlights improved tool calling and instruction following in Granite 4. No BFCL scores for Granite models at ≤14B were retrievable from public sources. Risk assessment: insufficient public data to characterize failure mode probability.

**Mistral family.** Mistral-nemo:12b's fast-confabulation failure mode in this cycle aligns with the general observation that the Mistral family's tool-calling fidelity is below the Qwen family at similar parameter counts. No BFCL scores for mistral-nemo:12b were found in public sources. The cycle's empirical evidence is stronger than what the public benchmarks provide for this model. No other currently-pulled Mistral variant (mistral:7b) shows any evidence of strong tool-calling capability — the base Mistral 7B predates Mistral's function-calling fine-tuning work.

**DeepSeek-R1:8b.** Already pulled locally. DeepSeek-R1-0528 (May 2025) added JSON output and function calling. The R1 architecture uses chain-of-thought internal monologue before tool calls, which at the 8B scale on consumer hardware would add substantial latency overhead. Risk assessment: the explicit reasoning step likely reduces confabulation risk relative to mistral-nemo, but higher per-turn latency is expected. Specifically worth testing with the biased system prompt given local availability.

**Gemma3:1b.** Already pulled locally. At 1B parameters, this is below any documented function-calling capability threshold. The ToolACE paper showed "minimal function-calling ability" in sub-2B models before specialized fine-tuning. Not a viable candidate for orchestration tasks.

**Quantization does not degrade tool-calling accuracy.** The Docker evaluation found no significant difference between quantized and non-quantized variants in tool-calling performance. This validates Ollama-via-GGUF as a deployment strategy without capability concerns — the resource savings from quantization come essentially for free.

---

### Key Findings

- BFCL and related benchmarks identify "memory, dynamic decision-making, and long-horizon reasoning" as the primary unsolved challenges for multi-turn tool-calling, directly matching the cycle's S0 working-memory-lapse observation (Patil et al., ICML 2025).

- Standard benchmarks do not measure fast-confabulation — confident prose fabrication with zero tool dispatch events. The cycle's CAP-5 failure mode is an unmeasured gap in the public evaluation landscape, confirmed indirectly by the VILA-Lab identification of the "Observability–Evaluation Gap" in Claude Code (arXiv 2604.14228).

- Tool surface dilution is documented and significant: Anthropic's cited research shows a 13% → 43% accuracy gain from basic retrieval over a large tool set; some models show >20% accuracy reduction under competing format constraints (Red Hat, 2025, citing Anthropic research).

- The ToolACE paper confirms that loss (difficulty) of tool-selection correlates positively with API pool size and with semantic distance between query and API description (Wang et al., arXiv 2409.00920, 2024).

- In a 21-model practical evaluation, Qwen3-8B (F1=0.933) and Qwen3-14B (F1=0.971) significantly outperform all other locally-deployable models at their parameter class, including specialist function-calling models (xLAM-8B: F1=0.570, Watt-tool-8B: F1=0.484) (Docker Engineering Blog, 2025).

- LangGraph documents that supervisor routing adds >30% latency overhead; AutoGen's GroupChat model incurs one full LLM call per agent turn. Structural cognitive splits improve selection reliability but multiply wall-clock time — consistent with the cycle's CAP-2 finding (2.54x overhead for router-executor) (LangGraph documentation, 2024).

- LLM self-calibration is consistently poor: models show "minimal variation in confidence between right and wrong answers" and overconfidence increases with self-improvement iterations. Small models cannot reliably detect their own failures (TMLR 2025 honesty survey).

- OpenHands, Claude Code, and the general agentic observability literature surface agent steps and tool calls to operators via OTEL/MLflow tracing, but none specifically detect the case where an LLM's prose claims a tool was called when no tool dispatch event occurred (arXiv 2604.14228, docs.openhands.dev, MLflow blog).

- Beck's "Nobody Wants Agents" (April 23, 2026) frames the coordination-burden failure mode precisely: the user becomes a manager watching agent state rather than describing outcomes. He names the problem but does not address local-model constraints, capability gradients, or default-experience design.

- Qwen3.5 small models (released March 2, 2026) with native Ollama tool-calling support represent the most current family candidate within the hardware constraint. The 9B variant (6.6 GB) fits comfortably below the qwen3:14b (9.3 GB) ceiling (Ollama official announcement, March 2026).

---

### Limitations

**BFCL leaderboard scores for specific small models.** The BFCL V4 leaderboard is a live page that did not render tabular data in the search tools used. Specific ranked scores for qwen3:8b, qwen3:14b, and xLAM-2-8b-fc-r on the current leaderboard were not directly confirmed via the primary source. The Docker practical evaluation provides partially overlapping data but uses a different methodology (F1 score on real tool selection) rather than AST structural matching. Recommendation: verify current BFCL rankings at gorilla.cs.berkeley.edu/leaderboard.html before acting on the Gap 6 model recommendations.

**xLAM-2 Ollama availability is community-provided, not official.** The `robbiemu/Salesforce_Llama-xLAM-2` entries on Ollama are community quantizations, not Salesforce official releases. Model file integrity should be verified before use. An official Ollama library entry for xLAM-2 was not found.

**Fast-confabulation detection is under-researched.** The specific failure mode — confident prose fabrication with no tool dispatch event — is not directly addressed by any paper reviewed. The VILA-Lab Claude Code paper gestures toward this but does not propose a detection mechanism. This is a genuine gap in the literature that the cycle's empirical work is filling.

**Beck's prior Genie Lessons (#1-#4) are partially behind a paywall.** The substack tag page lists titles but full content of some earlier numbered Genie Lessons required paid access. What was retrievable is characterized above; the full arc of Beck's development of the "outcome vs feature" framing across all four prior lessons cannot be confirmed from public access alone.

**Qwen3.5 tool-calling quality at 9B is assumed from family lineage, not independently benchmarked.** The Qwen3.5-9B release is recent (March 2026) and BFCL-specific scores were not found in the search horizon. The recommendation rests on Ollama's official confirmation of native tool calling support and the strong Qwen3 family baseline. The model should be validated empirically before being committed as a production recommendation.

---

### Cross-Gap Synthesis

Three themes recur across the six gaps:

**1. The capability floor is a tuple, not a scalar.** Every gap confirms in a different way that "can this model call tools" is the wrong question. The right question is "can this (model × surface × prompt × description × context-depth) tuple call the right tools reliably?" BFCL's API pool size findings (Gap 1), the tool dilution literature (Gap 3), the LangGraph structural-split rationale (Gap 4), and the cycle's own (model × prompt) 2×2 matrix (CAP-1/CAP-3) all point to the same conclusion. This has a direct implication for RQ-2 (detection): a pre-session capability signal that ignores surface, prompt, and context depth will give unreliable readings.

**2. Fast-confabulation is unmeasured and unmitigated in the public landscape.** Standard benchmarks (BFCL, ToolBench, tau-bench) evaluate the quality of tool calls that happen. None evaluate the rate at which a model fabricates prose asserting a tool call occurred without any actual dispatch. The observability literature (OpenHands, Claude Code) detects tool calls that do happen, not tool calls that are claimed but don't happen. This is the most dangerous failure mode (user sees authoritative-sounding output and cannot tell it is fabricated) and the least addressed by existing infrastructure. The cycle's empirical work is filling a genuine public gap here.

**3. Structural solutions are valid but expensive.** Multi-agent cognitive splits (Gap 4), Tool RAG for surface reduction (Gap 3), and operator-side OTEL tracing (Gap 2) all work as documented. But each adds latency, complexity, or infrastructure overhead that compounds badly at consumer hardware inference speeds. The practical finding from CAP-3 — system prompt configuration is a strong and cheap lever — is consistent with the Beck finding (Gap 5) that explicit constraints added to prompts change architecture-level behavior more than persona prompts alone. The cheapest effective interventions remain at the prompt/configuration level; structural interventions earn their cost only when the cheap interventions have been exhausted.

---

### Implications for the Cycle's Research Questions

**RQ-1 — Capability floor and affordances.** The literature confirms the floor is a (model × surface × prompt × description quality) function, not a model-only property. The ToolACE API pool size findings provide theoretical grounding for the 16-tool dilution observed in CAP-1. The Docker F1 evaluation confirms Qwen3:8b is the top local ≤14B tool-calling model by practical measurement. The tau-bench sub-50% GPT-4o finding calibrates expectations: full cascade success on constrained multi-turn tasks is hard even for frontier models; qwen3:8b with biased prompt clearing the cascade (even with 4 retries) is a good result relative to the benchmark baseline. **Status: substantially grounded by literature. Residual gap: BFCL scores for exact models used not confirmed from primary source.**

**RQ-2 — Available signals, limits, and honest-default alternatives.** The self-calibration literature (Gap 1) confirms small models cannot reliably report their own capability limits. This closes the "signal detection" question negatively: there is no reliable pre-session signal derived from model self-assessment. The implication is that honest defaults must come from operator attestation or empirical capability-tier testing (the 2×2 matrix approach the cycle used), not from model-reported confidence. **Status: literature confirms the negative finding. Operator attestation or empirical gating is the right design direction. Open question: what gating UI/UX looks like in practice.**

**RQ-3 — Observability surfaces, separability, capability-gate alternatives.** The literature (OpenHands, Claude Code, arXiv 2503.06745) confirms operator-side event-stream observability is well-solved for tool calls that happen. User-side in-stream surfacing exists (Claude Code's terminal output) but is coarse. The Beck framing (coordination-burden failure mode) is documented by at least one shipped system (Intent by Augment Code as described by Beck). **Critical unresolved gap: no system in the literature specifically detects fast-confabulation.** The cycle's empirical finding that operator log comparison (tool dispatch events vs LLM prose claims) can detect this is original. A capability gate that prevents sessions where the profile is below the confabulation floor (CAP-5 pattern) is the literature's implied resolution — not a better surface design.

**RQ-4 — Minimum intervention for honest first-session experience.** The literature confirms the cheapest effective intervention is prompt-level (Beck on explicit constraints; CAP-3 empirical finding). The second-cheapest is model selection (Qwen3 family's F1 dominance). Structural solutions (router-executor, Tool RAG) are third-tier by cost. A default configuration change (biased system prompt + Qwen3 family default) is the minimum viable intervention supported by both the literature and the cycle's empirical findings. Capability gating (refusing to start a session with a below-floor profile) is a second intervention that the calibration literature supports but which requires explicit operator attestation rather than model self-assessment.

**RQ-5 — Dual-contract divergence, convergence, and seam navigation.** Beck's posts address only the user contract (outcome delivery). The project contract (non-frontier orchestration hypothesis) is not addressed anywhere in the retrieved literature — it appears to be genuinely original to this cycle as an explicit framing. The Canonical findings (prior lit-scan partial in the research log) support the hypothesis at a strategic level but do not provide empirical validation. The seam between the two contracts is most visible at the latency dimension: the architecture exercises the project hypothesis (cascade works), but the user contract is degraded by 22-minute wall-clock. The literature offers no solution to this specific tension for consumer-hardware local inference; faster hardware is the only documented resolution.

---

### Cycle-Original-Signal Section

What does the cycle's empirical work (S0, CAP-1, CAP-3, CAP-5) contribute that the literature does not already address?

**1. Fast-confabulation as a named, measured failure mode.** The literature identifies confabulation and hallucination broadly. It does not name or measure the specific sub-type: a model produces confident prose asserting it queried system state, when no tool dispatch event occurred. The cycle has named this (fast-confabulation), characterized it empirically (CAP-5: 721 characters of fabricated content including an invented ensemble, invented invocation syntax, claimed invocation semantics — all in 1m53s with zero tool calls), and distinguished it from the other two failure modes (slow-useful: cascade engages but slowly; fast-giveup: empty output, silent stop). This taxonomy is absent from the existing benchmark literature.

**2. Model × prompt 2×2 as a controlled capability measurement.** The cycle ran a systematic (model × prompt) experiment: qwen3:8b default fails, qwen3:8b biased succeeds, mistral-nemo:12b biased fails, mistral-nemo:12b default fails. BFCL and ToolBench do not offer this kind of controlled system-prompt variation. The finding that prompt steering is family-specific (rescues qwen3:8b, cannot rescue mistral-nemo:12b) is not documented in the benchmark literature at this level of specificity.

**3. Internal-tool discoverability under competitive client-tool conditions.** The cycle tested a scenario the existing benchmarks do not cover: an orchestrator with both internal orchestration tools and external client tools in the same tool schema array, where the internal and external tools compete on semantic similarity. CAP-1's finding (qwen3:8b routes to `skill` instead of `list_ensembles` when the query wording loosely matches both) is a specific variant of the dilution phenomenon that the Tool RAG and ToolACE literature describes abstractly but does not test with this specific internal/external tool competition structure.

**4. Latency as a first-class capability floor dimension.** BFCL measures correctness, not latency. The cycle records 32s/turn for qwen3:14b at 1478 tokens, scaling to 171s/turn at 15K tokens, producing a 22-minute total session. This latency profile is a product-unusability finding that no capability benchmark captures. The user contract includes latency tolerability; the cycle treats latency as a separate dimension of the floor, not subsumed under accuracy.

**5. Architecture validation under real conditions.** The cycle observed the full llm-orc architecture from Serving Layer through Orchestrator Runtime through Tool Dispatch through Ensemble Engine through Result Summarizer Harness in a single 7-turn session, including a real SummarizationFailure event caught and surfaced correctly. This is not something any benchmark covers — it is system-level integration validation under realistic task conditions.

---

### Spike-Actionable Recommendations (Gap 6)

The following specific models are recommended for the cycle's next spike round (CAP-6 and forward), with pull commands and rationale.

**Recommendation 1: `ollama run qwen3.5:9b` (6.6 GB)**

The Qwen3.5-9B is the most current release in the cycle's strongest empirical family (Qwen3), with native Ollama tool calling confirmed and a weight footprint comfortably below qwen3:14b. The 9B model should run measurably faster than qwen3:14b (9.3 GB) at similar capability levels — estimating approximately 30-50% faster per-turn at the same context depth, based on the qwen3:14b → qwen3:8b latency gradient observed in CAP-1 (−46% wall-clock). Released March 2, 2026. Risk: BFCL scores not yet independently confirmed from primary source. Low confabulation risk given family lineage.

```
ollama pull qwen3.5:9b
```

**Recommendation 2: `ollama run qwen3.5:4b` (3.4 GB)**

The Qwen3.5-4B represents a smaller step down within the same family. At 3.4 GB it leaves substantial RAM headroom and runs noticeably faster than qwen3:8b. The question is whether Qwen3.5's improvements transfer to the 4B level for agentic/tool-use tasks specifically. This is a direct test of the capability gradient within the Qwen3.5 family. Risk: 4B models historically sit below reliable tool-calling thresholds (CAP-1 showed the floor between 8B and 14B for base Qwen3; Qwen3.5 may have moved that floor down). Test with the biased system prompt. Low confabulation risk given family lineage.

```
ollama pull qwen3.5:4b
```

**Recommendation 3 (conditional): `deepseek-r1:8b` (already pulled)**

This model is already locally available. Its reasoning-before-action architecture suggests lower confabulation risk than mistral-nemo:12b, because the internal chain-of-thought step provides a grounding pass before committing to a tool call. The risk is higher per-turn latency due to the reasoning overhead. Worth testing with the biased system prompt in the same CAP-3 conditions before pulling any additional models. No pull required.

**Recommendation 4 (lower priority): `ollama pull robbiemu/Salesforce_Llama-xLAM-2:8b-fc-r-q5_K_M`**

xLAM-2-8B is explicitly trained for function calling and claims BFCL top-4 status. However, the practical evaluation showing F1=0.570 is concerning and may reflect a tool-call format mismatch rather than fundamental capability failure. Testing xLAM-2-8B with the biased system prompt in the exact same OpenCode+llm-orc conditions as CAP-3 would determine whether the BFCL claim translates to this cycle's specific surface. Risk: community Ollama repo, not official release — verify model hash. The practical evaluation F1 suggests wrong-tool-path failures rather than confabulation, which is less dangerous than the mistral-nemo:12b failure mode but still undesirable.

```
ollama pull robbiemu/Salesforce_Llama-xLAM-2:8b-fc-r-q5_K_M
```

**Risk assessment summary for next spikes:**

| Model | Confabulation risk | Wrong-tool risk | Size | Status |
|---|---|---|---|---|
| qwen3.5:9b | Low (family lineage) | Low (family lineage) | 6.6 GB | Recommended first |
| qwen3.5:4b | Low (family lineage) | Unknown (4B floor may apply) | 3.4 GB | Recommended second |
| deepseek-r1:8b | Lower (reasoning step) | Probably lower | ~5 GB | Already pulled; test third |
| xLAM-2-8b-fc-r | Unknown | Elevated (F1=0.570 in practical eval) | ~5 GB | Test fourth |
| granite3.3:8b | Unknown | Unknown | ~5 GB | Insufficient data to recommend |

---

### Open Questions Surfaced

The following questions are surfaced by this literature scan that the cycle has not yet investigated:

1. **Is Tool RAG applicable at llm-orc's scale?** The literature documents that retrieval-based tool selection boosts accuracy from 13% to 43% over large tool sets. At 16 tools (5 internal + 11 client) this may be unnecessary overhead, but as the ensemble library grows, the discoverability surface grows with it. At what ensemble count does Tool RAG become load-bearing? The MCP Tool RAG implementation described in the Red Hat blog is a concrete near-term integration point.

2. **Does quantization affect tool-calling reliability in this cycle's specific conditions?** The Docker evaluation found no significant difference between quantized and non-quantized variants. This should be confirmed with the cycle's specific models and surface conditions. If confirmed, Ollama GGUF quantized deployment is the validated strategy.

3. **What is the right surface for surfacing fast-confabulation to the operator?** The literature provides no pattern. The cycle's implicit suggestion (compare LLM prose claims against tool dispatch event log in real time) is novel. Formalizing this as a detection signal — e.g., a session-level "claimed vs actual tool call count" counter exposed in the operator log — is a concrete RQ-3 design candidate not validated by prior work.

4. **Can the biased system prompt be distributed as an Ollama Modelfile layer?** Ollama Modelfiles support a `SYSTEM` directive that bakes a system prompt into the model artifact. This would allow distributing a `qwen3.5-orchestrator` Modelfile that pre-packages the biased prompt. Whether this is better or worse than config-file override is an RQ-4 question not addressed by the literature.

5. **What does tau-bench find about policy-constrained agent consistency across trials?** The tau-bench paper's finding that GPT-4o succeeds on less than 50% of tasks "and is quite inconsistent" is suggestive for the cycle's retrial / calibration design (ADR-007 Calibration Gate). If variance is high across trials for the same model on the same task, calibration based on N invocations may be measuring noise rather than capability.

---

*Sources for this review were retrieved via web search and WebFetch on 2026-04-25.*

**Primary Sources:**
- [BFCL V4 Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)
- [BFCL Paper (ICML 2025)](https://proceedings.mlr.press/v267/patil25a.html)
- [tau-bench (arXiv 2406.12045)](https://arxiv.org/abs/2406.12045)
- [ToolBench (OpenBMB GitHub)](https://github.com/OpenBMB/ToolBench)
- [ToolACE (arXiv 2409.00920)](https://arxiv.org/html/2409.00920v2)
- [Tool RAG (Red Hat)](https://next.redhat.com/2025/11/26/tool-rag-the-next-breakthrough-in-scalable-ai-agents/)
- [Docker LLM Tool Calling Evaluation](https://www.docker.com/blog/local-llm-tool-calling-a-practical-evaluation/)
- [OpenHands Observability Docs](https://docs.openhands.dev/sdk/guides/observability)
- [MLflow + OpenHands](https://mlflow.org/blog/mlflow-openhands)
- [Dive into Claude Code (arXiv 2604.14228)](https://arxiv.org/html/2604.14228v1)
- [Beyond Black-Box Benchmarking (arXiv 2503.06745)](https://arxiv.org/html/2503.06745v1)
- [LangGraph Multi-Agent Workflows](https://www.langchain.com/blog/langgraph-multi-agent-workflows)
- [xLAM Salesforce GitHub](https://github.com/SalesforceAIResearch/xLAM)
- [Llama-xLAM-2-8b-fc-r HuggingFace](https://huggingface.co/Salesforce/Llama-xLAM-2-8b-fc-r)
- [Llama-xLAM-2 Ollama community](https://ollama.com/robbiemu/Salesforce_Llama-xLAM-2)
- [Beck, "Nobody Wants Agents"](https://tidyfirst.substack.com/p/genie-lessons-nobody-wants-agents)
- [Beck Genie tag page](https://tidyfirst.substack.com/t/genies)
- [Qwen3 Ollama library](https://ollama.com/library/qwen3)
- [Qwen3.5 Ollama library](https://ollama.com/library/qwen3.5)
- [Granite 3.3 Ollama library](https://ollama.com/library/granite3.3)
- [Qwen3 technical report (arXiv 2505.09388)](https://arxiv.org/pdf/2505.09388)

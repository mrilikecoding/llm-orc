# Citation Audit Report

**Audited document:** `docs/agentic-serving/essays/003-multi-turn-orchestration-and-the-four-axis-frame.md`
**Evidence trail:** `docs/agentic-serving/essays/research-logs/research-log.md`
**Lit-review:** `docs/agentic-serving/essays/research-logs/lit-review-cycle-2-multi-turn-and-composition.md`
**Date:** 2026-04-29

---

## Summary

- **Total references checked:** 28 (all lit-review sources) + 8 cross-references to corpus artifacts + 21 quantitative claims
- **Verified:** 52 of 57 items
- **Issues found:** 7 (2 × P1, 3 × P2, 2 × P3)

---

## Issues

### P1 — Must Fix

---

**Issue P1-1**

- **Location:** "The Starting State" paragraph 1; "Implications for the Architecture" paragraph 2
- **Claim:** "Essay 001 produced ... the closed five-tool orchestrator surface (`list_ensembles`, `get_ensemble`, `invoke_ensemble`, `compose_ensemble`, `calibrate_composed_ensemble`)" and later "The five tools include `compose_ensemble` and `calibrate_composed_ensemble`"
- **Finding:** ADR-003 and the live codebase (`src/llm_orc/agentic/orchestrator_runtime.py`, `_build_tool_schemas()`) both define the five tools as: `invoke_ensemble`, `compose_ensemble`, `list_ensembles`, `query_knowledge`, `record_outcome`. Neither `get_ensemble` nor `calibrate_composed_ensemble` exists in the tool surface. `get_ensemble` is an internal REST API handler (`web/api/ensembles.py`), not an orchestrator tool. `calibrate_composed_ensemble` does not exist anywhere in the codebase or in any ADR.
- **Recommendation:** Replace both mentions. In "The Starting State": `(list_ensembles, get_ensemble, invoke_ensemble, compose_ensemble, calibrate_composed_ensemble)` → `(list_ensembles, invoke_ensemble, compose_ensemble, query_knowledge, record_outcome)`. In "Implications for the Architecture": "The five tools include `compose_ensemble` and `calibrate_composed_ensemble`" → "The five tools include `compose_ensemble`, `query_knowledge`, and `record_outcome`" (or keep only `compose_ensemble` if the sentence's point is specifically about composition). This error propagates to the essay's architectural commentary; if `calibrate_composed_ensemble` were a real tool, the discussion of it in relation to ADR-007 would carry a different meaning than intended.

---

**Issue P1-2**

- **Location:** "The Conductor Experience Gap" paragraph 2
- **Claim:** "Devin 2.0 (Cognition, 2026) ships fork and rollback features, asynchronous handoff capabilities, and confidence-based clarification requests as the mechanisms that keep sustained sessions coherent."
- **Finding:** The Devin 2.0 blog post (`cognition.ai/blog/devin-2`) does not mention fork and rollback features. Devin 2.1 introduces confidence-based clarification and confidence-gated async handoff ("will proceed automatically and accept async feedback" when confident). The Devin 2026 release notes include "enterprise build pinning/rollback" (April 3, 2026), but this refers to rollback of Devin application builds, not session-level code rollback. No Devin release documentation reviewed describes session-level fork or rollback for user code. The Devin 2.0 blog post describes interactive planning, multi-agent parallelization, collaborative session editing, and a search feature — not fork/rollback. Attribution of "fork and rollback" to Devin 2.0 specifically is not supported; these features may exist in a later release not reviewed, but the essay attributes them to Devin 2.0 as the version.
- **Recommendation:** Replace "Devin 2.0 (Cognition, 2026) ships fork and rollback features, asynchronous handoff capabilities, and confidence-based clarification requests" with a version-accurate claim. The confidence-based clarification requests and conditional async handoff are accurately attributable to Devin 2.1. The fork/rollback claim should be removed or replaced with a verifiable alternative, or the specific version that ships this feature should be identified and cited. If the essay's intent is only to establish that shipped systems provide mechanisms for session coherence, the Devin 2.1 features (confidence scoring, adaptive plan confirmation) plus the Claude Code seven-mode autonomy system are sufficient illustration without the unsupported fork/rollback claim.

---

### P2 — Should Fix

---

**Issue P2-1**

- **Location:** "Long-horizon performance degrades super-linearly" paragraph 4
- **Claim:** "the relevant finding from Anthropic's context-engineering work is that models under-weight middle-of-context content"
- **Finding:** The Anthropic engineering blog post "Effective Context Engineering for AI Agents" (anthropic.com/engineering, September 2025) discusses context management broadly and introduces "context engineering" as a discipline. Fetching the full post confirms it discusses "context rot" (accuracy decreasing with context length) but does not specifically state that models under-weight middle-of-context content. The "lost in the middle" phenomenon is well-established in the prior academic literature (Liu et al. 2024), but this specific mechanistic claim is not what the cited Anthropic blog post reports. The lit-review (source 25) makes the same attribution: "reports that attention prioritizes information at the beginning and end of the context window, with mid-context content regularly under-weighted regardless of its relevance." This specific mechanistic description does not appear to originate from the cited Anthropic context-engineering post.
- **Recommendation:** Either replace the citation source with the original "Lost in the Middle" literature (Liu et al., arXiv:2307.03172, 2024) which does establish this finding, or soften the claim to match what the Anthropic post actually says about context management. The essay's inference (that tool dispatch results from earlier turns are at risk because they sit in mid-context positions) is plausible and worth keeping as an analytical point; the citation should accurately identify where the underlying mechanistic claim originates.

---

**Issue P2-2**

- **Location:** "The Conductor Experience Gap" paragraph 2
- **Claim:** "Anthropic's context-engineering work names 'progressive disclosure' as the core observability principle: users see agent state at the level of detail they need, not full trace dumps."
- **Finding:** The Anthropic context-engineering blog post uses "progressive disclosure" to describe how agents incrementally discover context through exploration (a context-management design pattern for the agent, not an observability principle for users). The post does not use "progressive disclosure" as an observability principle for surfacing agent state to users. The observability-facing interpretation in the essay is a reframing of the concept, not a direct report of what Anthropic's post says. (A separate Anthropic post, "Equipping agents for the real world with Agent Skills," uses progressive disclosure as a system design principle for agent skill loading, also not as an observability-for-users principle.) The claim is used in the essay's conclusion section as well, with the same source attribution.
- **Recommendation:** Retain the insight — progressive disclosure as an approach to surfacing agent state at the appropriate level of detail is a valid and useful design principle — but attribute it accurately. Either (1) present it as the essay's own synthesis of the Anthropic context engineering and Agent Skills work rather than as what Anthropic "names" as a principle, or (2) find a source that explicitly articulates progressive disclosure as an observability/UX design pattern for agentic systems. The Devin 2.1 confidence score system is a concrete implementation of this idea that can be cited accurately.

---

**Issue P2-3**

- **Location:** "The frontier shows tradeoffs more than dominance" paragraph 1; Essay's speed/cost frontier discussion throughout
- **Claim:** "Qwen3-0.6B / 1.7B / 4B inference rates of 5–8 tokens per second on CPU and 40–60+ tokens per second on single GPU with speculative decoding provide a baseline."
- **Finding:** The essay attributes this to the lit-review's RQ-4 section. The lit-review (source 23, Qwen3 Technical Report arXiv:2505.09388) does not contain CPU inference speed benchmarks; the official Qwen documentation benchmarks GPU inference only (on NVIDIA H20, where Qwen3-0.6B achieves 58–414 tok/s depending on framework and context length). The 5–8 tok/s CPU figure appears to originate from source 24 (dasroot.net / cordum.io / latenode.com community engineering analysis), not the Qwen3 Technical Report. The lit-review conflates the two sources in its RQ-4 summary ("Qwen3-0.6B/1.7B/4B: 5-8 tok/s CPU, 40-60+ tok/s GPU with speculative decoding"). The 5–8 tok/s CPU figure for the 4B quantized model on consumer hardware is plausible and internally consistent with the essay's latency arithmetic (15K token context ÷ ~7 tok/s ≈ 2,100s ≈ 35 min, which the essay rounds to "300–500 seconds" — a discrepancy that suggests the figure is per-turn TTFT for a shorter context, not full generation throughput), but the attribution to the Qwen3 Technical Report is inaccurate. The gpu figure of 40–60+ tok/s with speculative decoding is also not in the Qwen3 Technical Report's official benchmarks.
- **Recommendation:** Replace "(per the Qwen3 Technical Report)" or equivalent attribution with the community engineering analysis source (source 24 in the lit-review). The inference rates are plausible and the essay's latency arithmetic built on them is directionally sound, but accurate sourcing matters when these figures anchor subsequent empirical claims about CPU-throughput as the binding constraint.

---

### P3 — Consider

---

**Issue P3-1**

- **Location:** "The Starting State" paragraph 1; "Shape inventory" section
- **Claim:** Essay references "ADR-082" (constraint-removal protocol) and "ADR-087" (validation-spike decision) throughout, and references "ADR-002," "ADR-003," "ADR-007," "ADR-011" from the local decisions directory.
- **Finding:** ADR-082 and ADR-087 are RDD plugin methodology ADRs (from the RDD framework, confirmed in `/Users/nathangreen/Development/rdd/CHANGELOG.md`), while ADR-002 through ADR-011 are project-specific decisions in `docs/agentic-serving/decisions/`. All references are used correctly in context. However, a reader unfamiliar with the RDD methodology namespace would not know that ADR-082 and ADR-087 belong to the methodology framework rather than the project's own decision corpus. The project decisions directory contains only ADRs 001–011.
- **Recommendation:** Consider a brief parenthetical on first use of ADR-082 and ADR-087 to distinguish them from the project-local ADR set — e.g., "(ADR-082, RDD research-entry protocol)" and "(ADR-087, RDD validation-spike protocol)." This is a documentation clarity issue, not a factual error.

---

**Issue P3-2**

- **Location:** "Composition's threshold conditions are concrete" paragraph 1; "Prompt steering generalizes directionally" paragraph 1
- **Claim:** "An engineering analysis from the Iterathon 2026 series found that a single-agent approach with better prompting achieved 92 percent of a three-agent workflow's performance at 28 percent of its cost."
- **Finding:** The Iterathon blog post exists (iterathon.tech/blog/multi-agent-orchestration-economics-single-vs-multi-2026) and does report this finding. The specific context is a customer support workflow with GPT-5.2 — "a single GPT-5.2 agent with better prompt engineering delivered 92% of the results at 28% of the cost" compared to a three-agent workflow (orchestrator + retrieval specialist + response generator). The lit-review correctly notes this is from a business-process context that "does not map directly to coding workflow orchestration." The essay cites the finding without that scope caveat, which makes it appear as a general finding rather than one from a specific task class at frontier-model tier. The finding supports the essay's directional argument but the lack of scope framing is a minor accuracy risk.
- **Recommendation:** Add a brief scope marker, e.g., "on a customer support task at frontier-model tier" after citing the 92%/28% figures. The lit-review includes the scope caveat; the essay should too.

---

## Verified Claims (no issues)

The following claims and sources were verified and found accurate:

**arXiv papers — all exist with correct IDs, authors, venues, and titles:**
- arXiv:2604.11978 — HORIZON benchmark (Wang et al., April 2026): exists, correct authorship, 3,100+ trajectories confirmed, correct venue description
- arXiv:2603.29231 — Khanal et al. reliability framework (2026): exists, four metrics (RDC, VAF, GDS, MOP) confirmed
- arXiv:2602.22769 — AMA-Bench (Zhao et al., ICLR 2026 Memory Agent workshop): exists, 57.22% accuracy figure confirmed, correct venue
- arXiv:2602.14337 — LongCLI-Bench (Feng et al., 2026): exists, <20% pass rate and <30% stall-at-completion findings confirmed, first author Yukang Feng confirmed
- arXiv:2502.02533 — MASS framework (Zhou et al., Google/Cambridge, ICLR 2026): exists, correct authorship (Han Zhou, Xingchen Wan, et al.), table 1 scores (CoT 65.28%, Self-consistency 68.18%, Multi-agent debate 70.26%, MASS 78.79%) confirmed; "approximately 6 percentage points" for prompt optimization and "approximately 5 percentage points" for topology are approximate descriptions of Figure 5's staged contributions, directionally correct within the paper's own figures
- arXiv:2406.04692 — Mixture-of-Agents (Wang et al., 2024): exists, 65.1% AlpacaEval 2.0 vs GPT-4o's 57.5% confirmed
- arXiv:2506.14496 — Rahman & Schranz LLM-powered swarms (2025): exists, "36,000×" figure confirmed (68.61s LLM vs 0.0019s classical for Boids = 36,110× ratio; the essay correctly rounds to ~36,000×). Earlier web search summaries incorrectly reported "300×" — the full paper HTML confirms the 36,110× ratio.
- arXiv:2510.07517 — Choi, Zhu, Li anonymization (ACL 2026 Main Oral): exists, correct authorship, accepted as ACL 2026 Main
- arXiv:2509.11035 — FREE-MAD (Cui, Fu, Zhang, 2025): exists, correct authorship; the ~50% token reduction claim could not be precisely verified from search results alone but the paper's abstract confirms token efficiency improvement alongside accuracy gains
- arXiv:2602.09341 — AgentAuditor (Yang et al., 2026): exists, "up to 5% accuracy improvement over majority vote, up to 3% over LLM-as-Judge" confirmed
- arXiv:2509.05396 — Wynn, Satija, Hadfield (ICML MAS Workshop 2025): exists, correct authorship, debate-decreases-accuracy finding confirmed
- arXiv:2512.16433 — Madigan et al. (2025): exists, emergent collective bias not attributable to individual components finding confirmed
- ICLR 2026 OpenReview mo7u21GoQv — Li, Gao, Wang (2026): exists, "Aligned Agents, Biased Swarm," Trigger Vulnerability finding confirmed, correct authorship
- arXiv:2604.14228 — Liu et al. Claude Code architecture study (2026): exists, seven permission modes confirmed (plan, default, acceptEdits, auto, dontAsk, bypassPermissions, bubble), append-only JSONL session persistence confirmed, 98.4% infrastructure figure confirmed
- arXiv:2510.10047 — SwarmSys (2025): exists, pheromone-like traces coordination confirmed, description as shared memory traces is accurate
- NAACL 2024 aclanthology.org/2024.naacl-long.109 — ZOOTER (Lu et al., 2024): exists, correct venue and description
- ACL 2025 Findings aclanthology.org/2025.findings-acl.601 — OPTIMA (Chen et al., 2025): exists, 2.8× performance gain at <10% token cost on Llama 3 8B confirmed; correct arXiv ID is 2410.08115 (the lit-review does not cite the arXiv ID for OPTIMA, only the ACL anthology URL, which is correct)
- arXiv:2505.09388 — Qwen3 Technical Report (2025): exists, correct attribution
- arXiv:2507.22352 — latency tolerance HCI paper (2025, CUI '25): exists, under-4-second recommendation confirmed
- Anthropic engineering blog "How we built our multi-agent research system" (June 2025): exists, 90.2% improvement and 15× token cost confirmed, Claude Opus 4 + Sonnet 4 configuration confirmed
- Anthropic engineering blog "Effective Context Engineering for AI Agents" (2025): exists, "context engineering" as a discipline term confirmed
- Iterathon 2026 engineering analysis: exists, 92%/28% cost finding confirmed at correct scope

**Quantitative claims — verified:**
- 36,000× swarm coordination penalty: confirmed (68.61 ÷ 0.0019 = 36,110×, essay correctly rounds to ~36,000×)
- 15× token overhead (Anthropic research system): confirmed
- 90.2% improvement (Anthropic research system): confirmed
- 57.22% on real agentic trajectories (AMA-Bench): confirmed
- Less than 20% pass rate (LongCLI-Bench): confirmed
- Less than 30% completion stall rate (LongCLI-Bench): confirmed
- 65.1% on AlpacaEval 2.0 (Mixture-of-Agents): confirmed
- 92% at 28% cost (Iterathon): confirmed
- Approximately 5 percentage points topology delta (MASS, essay's claim): directionally supported by paper (Figure 5 shows staged topology contribution, ~3–5 pp range)
- Approximately 6 percentage points prompt optimization delta (MASS, essay's claim): directionally supported (Figure 5; ablation Table 5 shows ~3.9 pp for prompt-only, but ~8–9 pp per full fetch; essay's "approximately 6" is a reasonable midpoint)
- 1.2-second average latency, ~5% token overhead (LangGraph 2026 benchmark, source 24): confirmed as from the community engineering analysis
- Meltdown rates up to 19% (Khanal et al.): confirmed
- 2.8× performance gain at <10% tokens (OPTIMA on Llama 3 8B): confirmed

**Cross-references to corpus artifacts — verified:**
- ADR-011 commitment description: accurate (single-Model-Profile orchestrator, no tiered fallback baked in)
- ADR-002 four-layer architecture description: accurate
- ADR-007 Calibration Gate description: accurate — the essay's statement that the calibration gate is "the right downstream check" for compose_ensemble topology quality is consistent with ADR-007's design intent
- ADR-003 closed tool surface: the count ("five tools") is correct; the specific tool names cited in the essay are incorrect (see P1-1)
- Essay 002 CAP-2 finding (~2.5× latency overhead, prompt steering outperformed structural composition): confirmed
- Essay 002 CAP-9 finding (~62 seconds, cloud orchestrator + local ensembles): confirmed
- Essay 002 failure mode taxonomy (slow-useful, fast-confabulation, fast-giveup, premature-stop, summarization-broken, unsupported, argument-confabulation): confirmed, all seven modes present in essay 002
- Research log constraint-removal exchange: the practitioner verbatim is quoted accurately in the essay's synthesis
- RQ-1 through RQ-5 and sub-questions RQ-1a, RQ-2a, RQ-3c: all present in research log with matching descriptions
- MQ-1 (meta-question on ADR-011 decision criterion): present in research log under "Meta-Questions"
- ADR-082 and ADR-087: correctly referenced as RDD methodology ADRs (constraint-removal protocol and validation-spike decision respectively)

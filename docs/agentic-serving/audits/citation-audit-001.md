# Citation Audit Report

**Audited document:** `docs/agentic-serving/essays/001-agentic-serving-architecture.md`
**Evidence trail:** `docs/agentic-serving/essays/research-logs/research-log.md`
**Date:** 2026-03-20

---

## Summary

- **Total references checked:** 11
- **Verified:** 8
- **Issues found:** 3

---

## Issues

### P1 — Must Fix

#### Issue 1: LLMCompiler speedup figure is wrong

- **Location:** "The Orchestrator Agent" section; also in research log Q2 findings (explicit figure)
- **Claim:** Research log states "3.6x speedup over sequential ReAct." The essay uses this research as its basis.
- **Finding:** The LLMCompiler paper (arXiv 2312.04511, ICML 2024) reports "up to 3.7x" latency speedup over ReAct. Task-specific results are 1.80x on HotpotQA and 3.74x on Movie Recommendation. The figure 3.6x does not appear anywhere in the paper, its GitHub README, or secondary sources. The error originates in the research log and flows into the essay.
- **Recommendation:** Correct the research log to "up to 3.7x latency speedup." If the essay is revised to include the specific number, use "up to 3.7x" and cite arXiv 2312.04511.

#### Issue 2: "LATM (ACL 2025)" uses an acronym that belongs to a different paper

- **Location:** "Self-Building Ensembles — Prior Art" section
- **Claim:** "LATM (ACL 2025) formalized the idea further, treating tool creation as an action in the agent's MDP — new tools are valid actions the agent can take, expanding its own action space."
- **Finding:** Two papers exist and are conflated:
  - **"Large Language Models as Tool Makers"** (Cai et al., arXiv 2305.17126) published at **ICLR 2024** is the paper commonly abbreviated LATM. Different venue, different authors, different system.
  - **"LLM Agents Making Agent Tools"** (arXiv 2502.11705) published at **ACL 2025** is the ToolMaker paper — an agentic framework for converting GitHub repositories into LLM tools via closed-loop self-correction. This is what the essay intends to cite.

  The LATM abbreviation belongs to the ICLR 2024 paper. The essay applies it to the ACL 2025 paper. Additionally, the content description — "treating tool creation as an action in the agent's MDP" — is an interpretive gloss not stated as a primary claim in either paper.

- **Recommendation:** Remove the LATM abbreviation. Cite the ACL 2025 paper by title: "LLM Agents Making Agent Tools (ACL 2025)." Revise the description to match what the paper actually demonstrates.

---

### P2 — Should Fix

#### Issue 3: LLMCompiler architecture description omits the replanning (joiner) component

- **Location:** "The Orchestrator Agent" section
- **Claim:** "separating planning from execution achieves significant speedup over sequential reasoning"
- **Finding:** LLMCompiler has three components: a Function Calling Planner, a Task Fetching Unit, and an Executor. A "joiner" decides whether to replan or finish. The essay reduces this to "planning from execution," omitting adaptive replanning.
- **Recommendation:** Expand to include the joiner/replanning component for accuracy.

---

### P3 — Consider

- **Codex compaction:** Accurate, but loosely sourced. Citing the OpenAI Codex API documentation directly would strengthen the reference.
- **OpenCode issue #5674:** Issue confirmed open as of December 2025. Resolution status not verified.

---

## Verified References

| Reference | Verdict |
|---|---|
| Voyager (2023) — 3.3x unique items, 15.3x faster milestone completion, three components | Confirmed. arXiv 2305.16291. |
| Chroma "context rot" research | Confirmed. research.trychroma.com. |
| Google ADK `AgentTool` | Confirmed. |
| CrewAI Flows + Crews model | Confirmed. |
| Roo Code — no XML fallback (v3.37+) | Confirmed. |
| LLMCompiler — exists, ICML 2024, parallel function calling | Confirmed. arXiv 2312.04511. |
| ReAct loop as canonical agentic pattern | Confirmed. |
| Ollama, vLLM, LiteLLM as OpenAI-compatible reference implementations | Confirmed. |

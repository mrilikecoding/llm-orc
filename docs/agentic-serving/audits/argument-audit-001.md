# Argument Audit Report

**Audited document:** `docs/agentic-serving/essays/001-agentic-serving-architecture.md`
**Evidence trail:** `docs/agentic-serving/essays/research-logs/research-log.md`
**Date:** 2026-03-20

---

## Summary

- **Argument chains mapped:** 7
- **Issues found:** 10 (2 P1, 5 P2, 3 P3)

### Argument chains mapped

1. API surface is minimal and stable, therefore the engineering task is bounded (Q1 -> "The API Surface" section)
2. A DAG engine and a ReAct loop are complementary, therefore the internal orchestrator model is the pragmatic entry point (Q2 -> "The Orchestrator Agent" section)
3. Context rot degrades quality, therefore three mitigation strategies are needed (Chroma finding -> "Context Management" section)
4. LLMCompiler separates planning from execution; llm-orc's DAG engine provides within-ensemble parallelism; therefore wrapping it as a tool benefits the orchestrator (Q2 finding -> "The Orchestrator Agent" section)
5. Voyager and ToolMaker demonstrate self-modification works when constrained; conductor skill provides a mature framework; therefore self-building ensembles are feasible within guardrails (Q3 -> "Self-Building Ensembles" section)
6. Invariant 7 governs the ensemble reference graph, not the orchestrator's tool-mediated invocations; therefore the orchestrator does not contradict the domain model (Q3/codebase analysis -> "Tension" section)
7. Without memory the orchestrator is stateless; Plexus provides the graph infrastructure; therefore Plexus integration is the differentiator (Q4 -> "Plexus as Memory Layer" section)

---

## Issues

### P1 — Must Fix

**Issue P1-1**

- **Location:** "The Orchestrator Agent" section
- **Claim:** "llm-orc's DAG engine already provides parallel multi-model execution — wrapping it as a tool gives the orchestrator LLMCompiler-like efficiency without additional infrastructure."
- **Evidence gap:** LLMCompiler's speedup comes from a three-part architecture: planner, executor, and joiner. The orchestrator pattern has no planner phase and no joiner. Calling this "LLMCompiler-like efficiency" imports a performance claim from a structurally different design. The analogy is partial.
- **Recommendation:** Replace "LLMCompiler-like efficiency" with a bounded claim about within-ensemble parallelism analogous to LLMCompiler's executor phase.

**Issue P1-2**

- **Location:** "Tension: Static References vs. Dynamic Invocation" section
- **Claim:** The tension with Invariant 7 is "narrower than it first appears."
- **Evidence gap:** The CLI analogy holds for invocation, but not for creation-then-invocation. The essay does not establish how the orchestrator's runtime creation path satisfies Invariant 7 without an explicit validation mechanism.
- **Recommendation:** Add a sentence describing how created ensembles are validated — either restricted to pure profile-and-script compositions (no ensemble-to-ensemble references), or validated explicitly before load.

---

### P2 — Should Fix

**Issue P2-1:** The basis for choosing option 2 (internal) over option 3 (hybrid) is in the research log but not stated in the essay. Add explicit reasoning.

**Issue P2-2:** "mirrors the conductor skill's existing routing logic" — the conductor's routing logic is not described in the evidence trail. Qualify or characterize briefly.

**Issue P2-3:** LATM/ToolMaker citation establishes a theoretical concept but does not establish applicability to YAML-based ensemble composition. Add qualifier.

**Issue P2-4:** "8 out of 10 times" in Plexus section is an illustrative placeholder presented as a concrete figure. Make hypothetical nature explicit.

**Issue P2-5:** "Each layer exists independently" — Layers 2 and 4 do not exist yet. Qualify as architectural intent.

---

### P3 — Consider

**Issue P3-1:** "bounded engineering task" understates client-specific integration testing surface (Roo Code no XML fallback, OpenCode issue #5674).

**Issue P3-2:** "Research from Chroma demonstrates" — no citation, title, or date given. Soften or add source.

**Issue P3-3:** "evaluation mode" conflates sandboxing (execution isolation) with calibration (quality gating over N invocations). Separate or define explicitly.

---

## Notes

The core logical flow is sound. The research log's four questions map cleanly to the essay's four sections, with no contradictions. Terminology is consistent throughout. The decision to surface the Invariant 7 tension directly is appropriate. The primary risks are a partially borrowed performance claim (P1-1), an incompletely resolved domain model tension (P1-2), and specific figures without verifiable sources (P2-4, P3-2).

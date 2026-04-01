# Agentic Serving -- Orientation

## What this system is

Agentic serving extends llm-orc (a declarative DAG-based LLM orchestration engine) to serve as the backend for agentic coding tools via OpenAI-compatible endpoints. An orchestrator agent sits behind `/v1/chat/completions`, running a ReAct loop that delegates to llm-orc ensembles as tools. The Plexus knowledge graph provides cross-session memory and design knowledge accumulation, converting the orchestrator's routing decisions from reasoning tasks into retrieval tasks over time.

## Who it serves

*To be populated after DISCOVER phase.*

## Key constraints

*To be populated after MODEL phase. Preliminary constraints from research:*

- Invariant 7 (static ensemble references) governs the ensemble reference graph but not the orchestrator's tool-mediated invocations
- Context management is a correctness requirement -- ensemble results must be summarized before entering the orchestrator's context
- Budget enforcement (turn limits, token budgets) operates at the control plane level, not the model level

## How the artifacts fit together

**Tier 1 -- Entry point:**
- `ORIENTATION.md` (this document) -- what this is, who it serves, where to look next

**Tier 2 -- Primary readables:** *(to be produced in later phases)*
- `product-discovery.md` -- stakeholder needs, value tensions, assumption inversions
- `system-design.md` -- module boundaries, responsibility allocation, dependency directions
- `roadmap.md` -- sequenced implementation plan

**Tier 3 -- Supporting material:**
- `essays/001-agentic-serving-architecture.md` -- research essay: API surface, orchestrator agent, self-building ensembles, Plexus integration
- `essays/research-logs/001-agentic-serving-architecture.md` -- research log cycle 1 (Q1-Q4)
- `essays/research-logs/001b-agentic-serving-architecture.md` -- research log cycle 2 (Q5-Q6: OpenHands, claw-code)
- `essays/reflections/001-agentic-serving-architecture.md` -- post-research reflections
- `essays/audits/citation-audit-001.md` -- citation verification report
- `essays/audits/argument-audit-001.md` -- argument chain audit report

## Current state

**RESEARCH phase complete.** The essay investigates six questions across two research cycles and concludes with a four-layer architecture (API surface, orchestrator agent, ensemble engine, knowledge graph). Citation and argument audits have been run; all P1 issues remediated. The next phase is DISCOVER (product discovery).

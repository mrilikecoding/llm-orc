# Agentic Serving -- Orientation

## What this system is

Agentic serving extends llm-orc (a declarative DAG-based LLM orchestration engine) to serve as the backend for agentic coding tools via OpenAI-compatible endpoints. An orchestrator agent sits behind `/v1/chat/completions`, running a ReAct loop that delegates to llm-orc ensembles as tools. The Plexus knowledge graph provides cross-session memory and design knowledge accumulation, converting the orchestrator's routing decisions from reasoning tasks into retrieval tasks over time.

## Who it serves

- **Tool user** — developer using an agentic coding tool pointed at llm-orc. Cares about quality, speed, cost. Reading path: `ORIENTATION.md` → `product-discovery.md` (Jobs and Mental Models) → essay abstract
- **Ensemble author / operator** — creates ensembles, profiles, scripts; runs the server. Often the same person as the tool user. Cares about visibility as tinkering, organic stabilization, cost control. Reading path: `ORIENTATION.md` → `product-discovery.md` → essay (full) → domain model
- **Orchestrator LLM** — the agent behind the endpoint. Needs the full composition palette, knowledge graph access, and budget constraints. Reading path: system design → domain model → essay (Orchestrator Agent, Context Management sections)

## Key constraints

Drawn from `domain-model.md`. The full set is AS-1 through AS-8 plus project-level Invariants 1-14; the constraints below most shape downstream decisions.

- **Plexus is optional (AS-8).** The orchestrator works statelessly -- serving layer, ReAct loop, budget enforcement, result summarization, and ensemble composition all function without Plexus. Plexus is an upgrade to a learning system, not a prerequisite. Design for stateless baseline; benefit from Plexus when available.
- **Dynamic invocations are outside the ensemble reference graph (AS-1).** Project Invariant 7 governs static YAML composition; it does not govern the orchestrator's runtime tool-mediated invocations. Orchestrator-composed ensembles must still satisfy Invariant 7 internally (AS-2).
- **Result summarization is a correctness requirement (AS-7).** Full ensemble result dictionaries must be summarized before entering the orchestrator's context. Unsummarized results cause context rot that degrades orchestrator quality over a session.
- **Budget enforcement is a control plane concern (AS-3).** Turn limits and token budgets are enforced at the session level, checked at each iteration of the ReAct loop, regardless of what the orchestrator LLM decides.

## How the artifacts fit together

**Tier 1 -- Entry point:**
- `ORIENTATION.md` (this document) -- what this is, who it serves, where to look next

**Tier 2 -- Primary readables:**
- `product-discovery.md` -- stakeholder needs, value tensions, assumption inversions
- `system-design.md` -- *pending (ARCHITECT)*
- `roadmap.md` -- *pending (ARCHITECT)*

**Tier 3 -- Supporting material:**
- `domain-model.md` -- scoped vocabulary (concepts, actions, invariants AS-1..AS-8, open questions); project-level vocabulary remains in force
- `housekeeping/cycle-status.md` -- active cycle state, phase tracking, feed-forward signals
- `essays/001-agentic-serving-architecture.md` -- research essay: API surface, orchestrator agent, self-building ensembles, Plexus integration
- `essays/research-logs/001-agentic-serving-architecture.md` -- research log cycle 1 (Q1-Q4)
- `essays/research-logs/001b-agentic-serving-architecture.md` -- research log cycle 2 (Q5-Q6: OpenHands, claw-code)
- `essays/reflections/001-agentic-serving-architecture.md` -- post-research reflections
- `housekeeping/audits/citation-audit-001.md` -- citation verification report
- `housekeeping/audits/argument-audit-001.md` -- argument chain audit report

## Current state

**RESEARCH, DISCOVER, and MODEL phases complete.** The essay investigates six questions across two research cycles and concludes with a four-layer architecture (API surface, orchestrator agent, ensemble engine, knowledge graph). Product discovery surfaced seven value tensions and six assumption inversions. The domain model establishes 8 scoped invariants (AS-1 through AS-8), 17 concepts, 13 actions, and 7 open questions. Key architectural posture: design for stateless operation (AS-8); Plexus is an upgrade to a learning system, not a prerequisite. The next phase is DECIDE -- ADRs, interaction specifications, and behavior scenarios using the domain vocabulary.

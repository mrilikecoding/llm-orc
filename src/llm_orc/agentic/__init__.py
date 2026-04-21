"""Agentic serving: orchestrator agent, session lifecycle, and OpenAI-compat surface.

Per `docs/agentic-serving/system-design.md`. This subpackage houses the
modules introduced by the agentic-serving RDD cycle (Serving Layer,
Session Registry, Orchestrator Configuration, and downstream Runtime
components). The existing Ensemble Engine (Layer 3 of the system design)
remains at `llm_orc.core` and is unchanged.
"""

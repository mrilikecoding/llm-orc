# Serving ADRs

Architecture decision records for the agentic serving subsystem
(`llm-orc serve`; see [`docs/serving.md`](../../serving.md)).

## Namespace rule

This directory is a **separate numbering space** from the project-level ADRs
in `docs/adrs/` (001–014: script agents, primitives, MCP, config models,
etc.). The two sets overlap in number but not in subject:

- A `# ADR-NNN` reference in **serving code** (`web/serving/`,
  `core/serving/`, `core/session/`, `core/validation/`,
  `models/dispatch_envelope.py`, the `agentic-serving` ensembles/scripts)
  means **this** set.
- A reference elsewhere (script agents, schemas, engine internals) usually
  means the project-level set in `docs/adrs/`.

## What's here

These are the *surviving* decisions from the 8-cycle agentic-serving research
process, copied with their original numbers so code references keep resolving.
Some carry partial-supersession headers (e.g. ADR-017's imperative half was
retired; only the tool-call guard survives) — trust an ADR's header over its
body.

Superseded decisions (ADR-001..008, 011, 014..023, 026..043 of this
numbering space) are **not** here. They, and the full research corpus
(essays, spikes, scenarios, field notes, audits), live on the
`research/agentic-serving-corpus` branch under `docs/agentic-serving/`.

| ADR | Decision |
|-----|----------|
| 009/010 | Plexus knowledge-graph substrate (optional), tool-first / push-model ingestion |
| 012 | Conversation compaction pipeline (session substrate) |
| 013 | Session registry — initializer-then-resume schema |
| 017 | Tool-call structural validation guard (surviving half) |
| 024 | Common I/O envelope (the inter-seat seam) |
| 025 | Artifact store as substrate |
| 044 | Declarative-ensemble-native serving (invariant AS-11) |
| 045 | Clean-slate collapse of the imperative serving layer |
| 046 | Target architecture: per-turn handler is one ensemble; orchestrator actor dissolved |
| 047 | Extensibility: Topaz-keyed registry + operator-curated shape catalog |
| 048 | Grounded acceptance: composed verification gate (executor + judge) |

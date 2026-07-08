"""Session substrate — the cross-turn state the serving ensembles read and write.

Per ADR-046 §3 (Cycle 8): the client owns the multi-turn loop; persistence
relocates to the substrate. This package houses the surviving session
containers relocated out of ``llm_orc.agentic`` at Cycle-8 WP-B8:

* :mod:`llm_orc.core.session.registry` — Session Registry (ADR-013).
* :mod:`llm_orc.core.session.artifacts` — structured-handoff artifact
  write-gate (ADR-013 disposition i).
* :mod:`llm_orc.core.session.artifact_store` — Session Artifact Store
  (ADR-025).
* :mod:`llm_orc.core.session.compaction` — Conversation Compaction
  (ADR-012 disposition).
* :mod:`llm_orc.core.session.plexus_adapter` — optional KG substrate
  adapter (ADR-009/010).
* :mod:`llm_orc.core.session.messages` — ``ChatMessage`` value type shared
  by the registry and the serving-layer contracts.
"""

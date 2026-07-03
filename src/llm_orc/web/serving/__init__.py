"""L3 Serving Layer glue for the Cycle-8 declarative Serving Ensemble.

The per-turn handler is ONE declarative ensemble (classify -> seat -> marshal)
executed by the L0 engine (ADR-046 §1; AS-11). This package holds the thin
serving-side glue that invokes it: the caller behind the endpoint's
``_ChatCompletionsCaller`` Protocol and the vendor-neutral turn trace. It does
not live under ``agentic/`` (the Cycle-8 deletion target).
"""

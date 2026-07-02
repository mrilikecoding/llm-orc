# ADR-044: Agentic serving is declarative-ensemble-native (codifies invariant AS-11)

**Status:** Accepted (2026-07-01)

## Context

The Cycle-7 PLAY arc ("agent as ensemble") established empirically that the agentic serving strategy can live as declarative ensembles on the L0 engine. Guard/branch and a bounded `loop:` combinator shipped and were validated through the real executor (commits `cb87ded`/`9d1a619`/`c5059bf`/`7ceab06`); the full generalist serving flow assembled as ONE declarative ensemble (Ω-P3) with no Python driver; and real `opencode run` drove that serve transparently, writing runnable code to disk (a2 confirmed, `proposals/play-closeout-2026-06-30.md`).

Cycle 7 nonetheless reached its serving behavior through a bespoke layer in `src/llm_orc/agentic/` (~12.7K LOC across 32 files). Its **imperative-orchestration mechanisms** are the AS-11-forbidden shape; the confidently-identified ones are the **ADR-033–043 loop-driver chain**, the **ADR-015 tier router**, and the **ADR-021 per-capability dispatch** (all superseded — see ADR-045). The layer's other contents span contracts, invariants, and deterministic work that physically live there but are not necessarily forbidden by decision-content: a file's presence in the layer is not proof its ADR decided an imperative mechanism (e.g. `orchestrator_tool_dispatch.py` self-attributes to ADR-003's *closed five-tool surface*, a contract). Which of the layer's ADRs are forbidden-mechanism vs. surviving-contract is classified by the target-architecture ADR from the complete source-derived inventory in ADR-045. All of the layer is removed by the clean-slate; not all of it is AS-11-forbidden by decision-content. The engine-control-flow reframe (2026-06-30, `proposals/engine-control-flow-state-and-next-steps.md`) named the cost: when control flow lives in adapter or harness Python, the orchestration graph lives in Python and the engine is a leaf-caller — the corpus thesis that "the agent *is* a declarative ensemble" is only honestly true if the agent's control flow lives in the ensemble, not in a driver. The engine was a fire-everything acyclic DAG that could not skip, branch, loop, or dispatch a runtime-chosen target; the response was to EXTEND it (guard, loop, and the planned dynamic dispatch), not to hand-roll orchestration a third time (`tier_router.py` was the second time).

The practitioner set this as the governing constraint for Cycle 8 (2026-07-01, verbatim recorded in `product-discovery.md` Cycle-8 additions): agentic serving uses the llm-orc architecture as designed; where that architecture is inadequate, extend it; never build a parallel architecture beside it.

## Decision

Codify invariant **AS-11**:

> **Agentic serving composes as llm-orc-native ensembles** — DAGs of model-profile, script, and ensemble nodes plus engine control-flow primitives (guard/branch and bounded loop, both shipped; dynamic dispatch, to be built as a primitive per the corollary below). **The agent's control flow lives in the ensemble, not in an imperative driver.** When the engine cannot express a needed control-flow shape, the resolution is to **extend the L0 engine with a minimal primitive**, never to add a parallel orchestration layer (adapter, driver, or harness) beside it.

Corollaries:

- **Dynamic dispatch** (a runtime-resolved `ensemble:` target) is **to be built as an engine primitive** (not yet shipped — guard/branch and bounded loop are the two shipped primitives), superseding the ~15-line adapter-mediated approach explored in Ω-dispatch.
- Deterministic serving work is a `script:` node; the stochastic surface is model-profile and ensemble nodes. There is no imperative Python orchestration graph.
- AS-11 governs framework **design** (how serving is built). It is distinct from **AS-6** (which constrains what the *runtime orchestrator* may compose from — existing primitives only). AS-11 permits framework developers to add engine primitives; AS-6 still forbids the runtime orchestrator from inventing scripts or model profiles.

## Rejected alternatives

- **Keep the bespoke imperative layer (the Cycle-7 status quo; the "wrapper/adapter" reading).** Rejected: it is the parallel architecture the invariant forbids. Ω-P3 showed the whole generalist flow is expressible declaratively, so the imperative layer is not necessary, and its ~12.7K LOC is exactly the divergence-from-the-engine cost the reframe named. This is the architecture the ADR-033 loop-driver chain built; AS-11 supersedes that direction (see Consequences).
- **Adapter-mediated dynamic dispatch (Ω-dispatch's ~15-line adapter).** Rejected as the durable form: it re-introduces a control-flow decision in adapter Python, reopening the exact seam AS-11 closes. Ω-P3 confirmed dynamic dispatch is the open-library extensibility lever (not a generalist-flow blocker), so it earns a real engine primitive rather than an adapter shim.
- **Strangler pattern on the spike/bespoke code.** Rejected by the practitioner (2026-07-01): the spikes are learning, not a substrate to harden; grafting production onto scaffolding is the worst of both. The clean-slate removal that follows from AS-11 is recorded separately in ADR-045.

## Consequences

**Positive:**
- The corpus thesis becomes literally true: each serving component is a declarative, independently-optimizable node, and strategy ensembles become config that can be compared head-to-head (the `(c)` lever from the a→b→c ladder).
- Extensibility has one shape (engine primitives + topaz-keyed dynamic dispatch), not two.

**Negative:**
- AS-11 contradicts the architecture of the ADR-033 loop-driver serving chain (ADR-033/034/035/036/037/038/039/040/041/042/043). Those ADRs are supersession candidates; **ADR-045 records the clean-slate collapse**. This is a backward-propagation event (DECIDE Step 3.7).
- Any control-flow shape the engine cannot yet express becomes engine work (a new primitive), a higher bar than an adapter patch. This is the intended cost, not a regression.

**Neutral:**
- The shipped primitives (guard/branch, bounded loop) already satisfy the generalist flow; only dynamic dispatch remains to build as a primitive.
- AS-11 does not by itself mandate the removal timeline; it establishes the target form. Sequencing is ADR-045 + roadmap work.

## Provenance check

- **Driver-derived:** the declarative-expressibility evidence (Ω-loop, Ω-P3, a2) is PLAY-phase spike evidence; the "extend the engine, never a parallel layer" framing is the engine-control-flow reframe plus practitioner direction 2026-07-01 (verbatim in `product-discovery.md` Cycle-8 additions and the `composition-is-full-capability` memory).
- **Drafting-time synthesis (flagged for the auditor):** the AS-6-vs-AS-11 actor distinction (runtime-orchestrator composition vs. framework-developer engine extension) is drafting-time framing, not from a driver document.
- **Empirical-Grounding Filter (ADR-097):** AS-11 is grounded by PLAY evidence (path 1 — the declarative form was built and drove real opencode), not surviving-from-research unfiltered.

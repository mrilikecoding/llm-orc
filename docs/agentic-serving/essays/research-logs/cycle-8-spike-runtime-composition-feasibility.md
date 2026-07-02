# Cycle 8 Spike: Runtime Composition Feasibility (pre-registration)

**Date:** 2026-07-02
**Phase:** DECIDE (Q4 extensibility; spike-then-decide rhythm)
**Status:** Pre-registered; structural probe next.

## Question (falsifiable)

Can runtime composition of a *new* ensemble from registered capability parts be
expressed on the shipped L0 primitives (dynamic dispatch + ensemble nodes +
guard/loop), or does it require a new "compose-at-runtime" engine primitive?

Dynamic dispatch (shipped, commits `64c79ee` / `c3709a0`) resolves a
runtime-chosen *existing* target. This spike asks the next rung: can a
declarative construct assemble a runtime-chosen *composition* of two or more
registered parts, not just select one.

## Why (decision relevance)

Q4 (extensibility) must decide the registry shape. The answer forks the design:

- **Expressible with shipped primitives** → the registry is a Topaz-keyed library
  of composable parts plus a binding decision; no new primitive. The
  practitioner's runtime-authorship direction (2026-07-02) is reachable as an
  authoring pattern, shape still deferred.
- **Not expressible** → Q4 (or a follow-on) designs a minimal compose-at-runtime
  primitive (AS-11: extend the engine). The spike names exactly what the engine
  cannot do today, which becomes the primitive's spec.

Either outcome de-risks the deferred runtime-authorship direction with evidence
rather than hope.

## Boundary (per the confirmed form, 2026-07-02)

Composition is decided **declaratively**: a decider-seat emits a composition
spec, and the engine assembles the DAG from registered, AS-2-validated parts.
NOT an unbounded LLM authoring arbitrary structure at runtime. This keeps the
probe inside the ADR-046 dissolution (no resurrected orchestrator actor).

**Feasibility only.** This spike does not design the runtime-authorship shape or
UX (deferred per practitioner). It tests whether the capability is reachable and
what, if anything, the engine is missing.

## Setup

- Reuse the Cycle-8 spike harness (dynamic-dispatch skeleton + the a2-proven
  Ω-serve foundation, `scratch/spike-omega-serve/`).
- Parts: 2-3 registered capability ensembles (e.g. code-generator, reviewer,
  summarizer).
- **Structural / no-model first:** can a declarative construct assemble a
  runtime-chosen composition of parts (e.g. a decider emits "compose
  [code-gen then reviewer]" and the engine builds and runs that DAG), beyond
  dynamic dispatch selecting a single existing target?
- **Then** a light real-model validation on the 32GB rig (one composed turn end
  to end).

## Validation / falsification

- **PASS (expressible):** a runtime-assembled composition of >= 2 registered
  parts builds and runs through the executor with no new primitive, decided
  declaratively.
- **FAIL (primitive needed):** the executor cannot assemble a runtime-chosen
  composition from parts; the spike records the precise gap (what the
  resolver/executor cannot do), which becomes the primitive's spec.

## Constraints

- $0-local; ask-before-spend.
- Methods-review skipped (feasibility spike; falsification pre-stated), per the
  Cycle-8 spike pattern.
- Built TDD + shippable if it turns into a primitive (like guard/loop/dispatch),
  not throwaway.

## Feeds

Q4 extensibility ADR: the registry shape, whether a new primitive is needed, and
the serving default.

## Findings

**Structural arm resolved the feasibility question (2026-07-02).** The boundary is
sharp and turns on what "composition" means.

**Fixed-shape pipeline, runtime-filled parts: expressible TODAY, no new primitive.**
A composition whose *shape* (stage count + wiring) is authored at design time, with
only *which registered ensemble fills each stage* chosen at runtime, works on the
shipped primitives. Two patterns:

- **Chained dispatch nodes** — a decider emits per-stage targets; each stage is a
  `dispatch:` node wired by `depends_on`, so stage N's output feeds stage N+1
  (`dynamic_dispatch_runner.py` returns the child result; standard `depends_on`
  input enhancement carries it forward). The decider chooses every part and each
  part is swappable; the wiring is authored.
- **Dispatch-to-wrapper** — pre-author a wrapper ensemble (e.g.
  `codegen-then-review.yaml`) and dispatch to it by name (existing single-target
  dispatch).

**Runtime-authored variable structure: NOT expressible without a new primitive.**
If the decider emits the *shape itself* (variable node count / edges) and the
engine materializes it, four code-verified gaps block it:

1. Dispatch resolves a single `str` target (`dispatch_resolved: str | None`), not a
   composition.
2. The `${dep.field}` resolver matches one scalar token; nothing interprets an
   emitted `{nodes, edges}` spec.
3. The DAG is topo-sorted once from the static agent list and never re-analyzed
   (frozen after parse). Runtime structural change today is limited to guard-drop,
   1:1 dispatch substitution, and homogeneous fan-out replication, none of which
   introduces a new heterogeneous sub-DAG.
4. No API materializes a runtime-emitted spec into an executable child (the child
   executor only runs a pre-loaded `EnsembleConfig` from disk).

**Primitive spec (if built):** a runner that (a) resolves a structured composition
spec (a resolver return shape beyond the scalar `${dep.field}`), and (b)
materializes it into an in-memory `EnsembleConfig` run as a child (or an executor
API to inject a sub-DAG). Built TDD, like guard/loop/dispatch. The composition is
still decided declaratively (a decider-seat emits the spec), so this stays inside
the ADR-046 dissolution; it is not an LLM authoring arbitrary code.

**Decision-relevance for Q4.** The runtime-composition interest splits into: (1)
parts-into-authored-shapes, free today; and (2) arbitrary runtime-authored shapes,
needs the primitive. Which the vision requires decides whether Q4 ships now
(registry = parts + a catalog of authored shapes + dynamic-dispatch binding) or
specs the primitive. Consistent with AS-11's "extend the engine when a flow
demonstrably needs a shape it cannot express" — the primitive is warranted only if
a needed shape is not in the catalog. Held for the practitioner (belief-mapping:
arbitrary shapes vs a catalog of shapes).

**Real-model arm: moot for (1)** — a fixed-shape runtime-filled pipeline is chained
dynamic dispatch, already a2/spike-proven. Deferred to (2) if the primitive is
built.

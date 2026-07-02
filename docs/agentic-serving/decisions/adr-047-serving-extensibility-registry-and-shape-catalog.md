# ADR-047: Serving extensibility — capability registry, composition-shape catalog, and declarative binding

**Status:** Proposed (2026-07-02)

## Context

Q4 (extensibility) is the DECIDE item after ADR-046. ADR-046 settled the per-turn
handler (classify → seat → marshal, one declarative ensemble) and dynamic dispatch
(a runtime-resolved `ensemble:` target, shipped). Q4 decides how operators register
and curate candidate seats per capability, the serving default, and the binding-time
question. It must do so without reintroducing the orchestrator actor ADR-046
dissolved or the AS-5 trust/promotion machinery it retired.

Two grounding inputs shape this decision:

- **Runtime-composition feasibility spike** (`essays/research-logs/cycle-8-spike-runtime-composition-feasibility.md`, 2026-07-02). A fixed-shape pipeline with runtime-filled parts is expressible today on the shipped primitives (chained `dispatch` nodes wired by `depends_on`, or dispatch-to-a-pre-authored-wrapper). A runtime-*authored* arbitrary DAG shape (a decider emits the structure and the engine materializes it) is NOT expressible without a new primitive, and the spike named four precise gaps.
- **llm-conductor mining** (the practitioner's local skill; `~/.claude/skills/llm-conductor/`). It is a catalog-anchored hybrid: a catalog of named composition shapes selected and customized per task, with novel-DAG design as the author-time fallback that graduates into a pattern library (its ADR-013 + ADR-019). Its essay-004 **Strategy A vs Strategy B** frame is the useful result: Strategy A (routing/composition intelligence *outside* the ensembles, over stateless instruments) beats Strategy B (a general ensemble with an *internal* small-model self-router) for small models, on cited evidence (AutoMix; small-LM-verifier work; DeepMind scaling-agents). The failure is not the patterns, it is placing the routing where a small model must execute it. Its trap is the `Design → Calibrate → Establish → Trust → Promote` populator, which is exactly the retired AS-5 machinery.

**Standing principle (practitioner, 2026-07-02).** Whenever the reasoning reaches "a
small model cannot do this," the first question is what **ensemble orchestration**
(bounded roles + structure + deterministic verification) can achieve, before
defaulting to "a frontier model should do it." Composition is the capability lever
(the corpus north star: the serve does everything a single model does, and
composition increases the surface). A frontier model is a fallback and a benchmark,
not the reflex.

## Decision

### 1. The registry is a Topaz-keyed library of building-block parts plus an operator-curated catalog of composition shapes.

- **Parts** are capability ensembles keyed by Topaz skill (`code_generation`, `tool_use`, `mathematical_reasoning`, `logical_reasoning`, `factual_knowledge`, `writing_quality`, `instruction_following`, `summarization`). The classify decider emits a target; dynamic dispatch resolves it against the registry.
- **Shapes** are named composition patterns authored declaratively as ensemble skeletons or wrappers (solo; gen → review; gather → analyze → synthesize; fan-out → merge; and so on). The catalog *structure* is borrowed from llm-conductor's pattern library (keyed by capability × shape × output-type); only the structure, not its populator.
- Both parts and shapes are **operator-curated**. **AS-2 (validate-before-load) is the admission gate**: every registered part and every registered shape is validated against the ensemble reference graph (no cycle, within the depth limit, every reference resolves to an existing entry) before it can fill a seat or run.

### 2. Binding is declarative and load-time-first (Strategy A).

The classify decider selects a shape and fills its slots; dynamic dispatch binds the
runtime-chosen parts. Selection lives in the declarative structure, not inside a
small model's self-routing. This runs on the shipped primitives today (feasibility
spike). Turn-time selection among several candidates per capability is a hybrid that
a guard/branch or a richer decider can express, but the default is **load-time
curation + classify-selection**, not a separate runtime routing actor.

### 3. Invention of new shapes is an author-time activity that grows the catalog.

New composition shapes are authored (by an operator, or by a capable design-time
process) and registered into the catalog after AS-2 validation and review. Runtime
shape-*invention* by the serve is out of scope for the near-term: the Strategy-A/B
evidence names small-model runtime shape-invention as the documented failure mode.

### 4. Serving default: one general shape operators extend.

A default generalist composition shape ships as the baseline (candidate shapes: the
plain classify → seat → marshal handler with a general capability seat, or a
gather → build → verify shape; chosen at BUILD). Operators extend by adding parts and
shapes, never by editing the engine (AS-11).

### 5. Not reintroduced: trust-promotion.

The catalog is operator-curated, not calibration-grown. There is no
accumulate-quality-then-auto-promote loop (retired AS-5). A composition is admitted by
AS-2 plus its contract, not by earned standing trust. Per-dispatch quality signal is
Q2's concern, kept separate.

## Deferred — the composer-ensemble path (named forward direction, not built here)

The practitioner's forward vision is **specialized ensembles that make other ensembles
from building-block ensembles** — the declarative rebuild of llm-conductor's *designer*
role as an ensemble. It is deferred, with a starting strategy so the deferral is
bounded, not vague:

- **compose-at-runtime primitive.** A runner that resolves a structured composition
  spec and materializes it as a child sub-DAG (closing the four gaps the feasibility
  spike named: single-target dispatch; scalar-only `${dep.field}` resolver; DAG frozen
  after parse; no runtime-spec-to-`EnsembleConfig` path). Built TDD like
  guard/loop/dispatch, **when a flow demonstrably needs a shape the catalog lacks**
  (AS-11).
- **composer ensembles**, with these strategy pillars:
  - (a) compose from the registry's validated parts;
  - (b) AS-2 gates the composed output before it registers or runs;
  - (c) the composer is a **verified ensemble, not a lone model** — and per the standing
    principle, the compose step is first attempted as an **orchestration of bounded
    small-model roles plus deterministic verification** (the ensemble-over-frontier
    bet), with a frontier model as fallback and benchmark, not the default;
  - (d) acceptance is **deterministic** (contract + verification), never
    trust-accumulation;
  - (e) **author-time-with-review first**; per-request runtime composition (on the
    primitive) is the further-out version, gated on evidence it runs reliably
    unattended.

## Rejected alternatives

- **Strategy B (a general ensemble with internal small-model self-routing).** Rejected
  on the evidence llm-conductor's essay-004 cites: the failure is placing routing where
  a small model must execute it. The classify seat is external and declarative
  (Strategy A). *Note per the standing principle:* Strategy A does not mean "a frontier
  model routes" — it means routing lives in declarative structure over bounded roles;
  the router itself can be an orchestration of small models.
- **Build the compose-at-runtime primitive now.** Rejected as premature. The catalog
  plus runtime-fill covers the near-term, and AS-11 says extend the engine when a flow
  needs a shape the catalog lacks. None does yet. Held as the named deferred extension.
- **Calibration-grown catalog (llm-conductor's Design → Calibrate → Trust → Promote).**
  Rejected: it is the retired AS-5 trust/promotion machinery wearing a design-laboratory
  coat. Copy the catalog structure, reject the populator.
- **Frontier-model default for hard composition (or routing) steps.** Rejected as the
  default per the standing principle: question what bounded-role ensemble orchestration
  can do first; frontier is a fallback and a benchmark, not the reflex.

## Consequences

**Positive:**
- Near-term extensibility ships on already-shipped primitives; no new engine work is required to reach a curated, composable serving surface.
- The composer-ensemble vision has a bounded, evidence-grounded path rather than an open question, with guardrails (AS-2, deterministic acceptance) that keep it clear of the retired machinery.
- The ensemble-over-frontier principle is applied as a design lens, wired into the composer's compose step.

**Negative / cost:**
- The shape-catalog + registry-curation surface (schema, operator UX, AS-2 wiring) is real BUILD work.
- The compose-at-runtime primitive and composer-ensembles are deferred; runtime shape-authorship is a genuine capability gap until they are built.

**Neutral:**
- Turn-time selection (the binding-time hybrid) is available but not the default; revisitable when a deployment shows a need.

## Provenance check

- **Driver-derived:** the registry-plus-catalog shape and the Strategy-A binding follow from the runtime-composition feasibility spike and the llm-conductor mining (Strategy A/B; the ADR-019 pattern-library structure). The dissolution, AS-11, and AS-2's survival are from ADR-046 and the domain model.
- **Drafting-time synthesis (flagged for the auditor):** the composer-ensemble strategy pillars are a synthesis of the practitioner's stated vision with the dissolution constraints; the serving-default candidate shapes are not yet chosen (BUILD); the standing ensemble-over-frontier principle is the practitioner's directive applied as a lens.
- **Empirical-Grounding Filter (ADR-097):** the near-term decision (catalog + runtime-fill) is spike-grounded (feasibility PASS for fixed-shape, runtime-filled compositions). The composer-ensemble direction is a **named forward direction, not a commitment** — deferred and un-grounded; the compose-at-runtime primitive is spec'd from the spike's four gaps but unbuilt. No feature is committed on research-surfaced possibility alone.

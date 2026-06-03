# ADR-024: Common I/O Envelope for Capability Ensemble Dispatches

> **Updated by ADR-035 on 2026-06-03.** For deliverables bound to a client tool, the output-form contract is composed at the marshalling boundary and delivered through the dispatch input — it is not the advisory `output_schema` this ADR describes. ADR-024's advisory-schema stance remains current for inter-ensemble composition; only the client-tool-deliverable path is refined by ADR-035. (Empirical correction recorded in ADR-035: `default_task` is inert at execution, so the "drift is `input.data` overriding `default_task`" mechanism named below does not operate for the client-tool path — the dispatch input is the sole contract surface reaching the model.)

**Status:** Accepted; Updated by ADR-035 on 2026-06-03 (client-tool-deliverable path)

**Date:** 2026-05-15

*(Status corrected from `Proposed` to `Accepted` on 2026-06-03: the `DispatchEnvelope` contract shipped in Cycle 6 BUILD — a stale-Proposed hygiene fix per the cycle-status flag.)*

---

## Context

ADR-021 establishes the per-capability dispatch contract: the orchestrator returns the capability ensemble's output (already summarized per AS-7) to the skill framework as the chat completion response. The contract does not specify the **shape** of that response — what fields it carries, what types those fields take, how downstream consumers (skill frameworks, the orchestrator's reasoning surface, post-hoc artifact-readers) parse it.

Spike α (research log `essays/research-logs/cycle-6-spike-alpha-envelope-survey.md`, executed 2026-05-15) surveyed all 8 agentic-serving ensembles' I/O shapes by reading each YAML's `default_task` and the most-recent `execution.json` artifacts. The survey produced two findings load-bearing for T16:

1. **A de facto envelope already exists uniformly across all 8 ensembles.** The `execution.json` artifact under `.llm-orc/artifacts/agentic-serving/<ensemble>/<timestamp>/` carries the shape `{ensemble, status, input, results, metadata, synthesis}` for every ensemble — the 6 capability ensembles (`web-searcher`, `claim-extractor`, `argument-mapper`, `code-generator`, `prose-improver`, `text-summarizer`) and 2 system ensembles (`agentic-calibration-checker`, `agentic-result-summarizer`). Client-facing value lives at `results.<agent>.response`. The cycle's T16 deliberation is not *whether* to codify an envelope but *how* the codified envelope refines the de facto shape and operates as a typed contract instead of a convention.

2. **Output-spec deviation observed in 3 of 6 capability ensembles' latest artifacts** (`claim-extractor` missed `(established)`/`(contested)` labels; `argument-mapper` restructured Premises/Intermediate/Final/Gaps into Thesis/Reasoning/Tensions; `prose-improver` analogous drift). The current envelope cannot detect drift because output specs live implicitly in `default_task` prose. **Spike β (research log `essays/research-logs/cycle-6-spike-beta-composition-predictability.md`) reframed the drift's mechanism** — the orchestrator hand-writes each stage's `input.data`, overriding `default_task` at run-time. The drift is at the orchestrator's reasoning surface (selection, reframing, distillation between dispatches), not at the synthesizer agent or response shape.

The Cycle 6 MODEL §Concepts entry on **Common I/O envelope** named four candidate field names — `envelope`, `primary`, `artifact reference`, `diagnostics` — and noted that the field names crystallize as operator-voice or stay analytical pending DECIDE deliberation. Spike α extended the candidate set: a typed `structured` field surfaced as load-bearing distinct from `primary` (typed payload distinct from human-readable canonical deliverable), and `errors[]` surfaced as load-bearing for partial-failure and per-stage-error handling.

Spike α proposed three envelope candidates and recommended candidate B (additive typed fields) as the lowest-disruption path:
- **Candidate A — Thin convention** (`status`, `primary`, `diagnostics`): codifies the de facto contract; does not address output-spec drift or composition predictability.
- **Candidate B — Additive typed fields** (`status`, `primary`, `structured`, `diagnostics`, `errors[]`; optional per-ensemble `output_schema:`): typed-field path; opens composition predictability for spike β; lowest-disruption against existing shape.
- **Candidate C — Artifact-as-substrate** (adds `artifacts[]` with `{path, content_type, size_bytes, summary, retention}`): full artifact-substrate; T16 sub-question (0) scope deliberation.

T16 cleaves into two structurally distinct decisions. **ADR-024 codifies the envelope contract** (decision 1). **ADR-025 codifies the artifact-as-substrate scope** (decision 2). The two ADRs compose: ADR-024 defines the field shape; ADR-025 specifies when `primary` carries inline content vs. when `primary` carries an artifact reference and the deliverable lives in `artifacts[]`.

---

## Decision

Codify the capability-ensemble dispatch response shape as a **typed envelope contract** following spike α candidate B with the field-name corrections from Cycle 6 MODEL vocabulary:

```python
@dataclass(frozen=True)
class DispatchEnvelope:
    """Response shape for invoke_ensemble dispatches.

    Per ADR-021's per-capability dispatch contract; specifies the
    shape of the capability ensemble's output the orchestrator
    returns to the skill framework as the chat completion response.
    """

    status: Literal["success", "error", "timeout", "partial"]
    """Dispatch outcome category. ``success`` = ensemble completed
    and produced a valid deliverable; ``error`` = dispatch failed
    (typed error in ``errors[0]``); ``timeout`` = dispatch exceeded
    timeout_seconds; ``partial`` = some stages completed, others
    errored (per-stage errors in ``errors[]``)."""

    primary: str
    """The canonical deliverable, human-readable. For substrate-routed
    ensembles per ADR-025, this is an artifact reference summary
    line (e.g., "Wrote class CircularBuffer to artifacts/code/
    circular_buffer.py (1.2 KB, application/python)"). For
    inline-response ensembles, this is the deliverable content
    directly. Always a string; consumers do not parse this field
    structurally — they parse ``structured`` or read ``artifacts[]``."""

    structured: dict | None = None
    """Optional typed payload. Present when the ensemble's
    ``output_schema:`` declares a typed shape; ``None`` otherwise.
    The shape is per-ensemble; ``output_schema:`` is the
    operator-facing declaration mechanism (see below)."""

    diagnostics: dict
    """Operator-readable dispatch diagnostics. Always present.
    Fields: ``ensemble`` (str), ``dispatch_id`` (str — correlates
    to ADR-023's typed dispatch events), ``duration_seconds``
    (float), ``model_profile`` (str), ``tier`` (str), ``topaz_skill``
    (str), ``calibration_verdict`` (Literal[Proceed, Reflect,
    Abstain] | None), ``audit_findings`` (list[dict] — empty list
    when no drift criteria findings). Field name aligned with
    domain-model vocabulary (not ``metadata``)."""

    errors: list[dict] | None = None
    """Optional per-stage errors. Present when ``status`` is
    ``error``, ``partial``, or when a successful dispatch
    surfaced non-fatal errors. Each entry: ``stage`` (str — agent
    name or pipeline phase), ``error_type`` (str — typed error
    name per ADR-015), ``message`` (str), ``recoverable`` (bool)."""

    artifacts: list[dict] | None = None
    """Optional artifact references. Per ADR-025, capability
    ensembles always populate this for their substantive
    deliverable; system ensembles populate only when an
    operator-readable artifact is part of their output (rare).
    Each entry: ``path`` (str — relative to the session artifact
    directory ``.llm-orc/agentic-sessions/<timestamp>/``),
    ``content_type`` (str — MIME type), ``size_bytes`` (int),
    ``summary`` (str — human-readable one-line description),
    ``retention`` (Literal[session, durable, ephemeral])."""
```

### `output_schema:` per-ensemble declaration

Capability ensembles **may** declare a typed `output_schema:` in their YAML. The schema is a JSON Schema (or Pydantic-model-derivable equivalent) describing the shape of the `structured` field. When declared:

```yaml
# .llm-orc/ensembles/agentic-serving/claim-extractor.yaml
name: claim-extractor
topaz_skill: factual_knowledge
default_task: |
  Extract claims from the source material. Label each claim
  (established) or (contested). Output only the bulleted list.
output_schema:
  type: object
  properties:
    claims:
      type: array
      items:
        type: object
        required: [text, label]
        properties:
          text: {type: string}
          label: {type: string, enum: [established, contested]}
```

When `output_schema:` is declared, the ensemble's synthesizer agent (or post-dispatch processing) populates `envelope.structured` with the typed payload. When `output_schema:` is absent, `envelope.structured` is `None` and only `envelope.primary` carries the deliverable. Spike β's reframing of output-spec drift as `input.data` override means schema validation is **advisory** at dispatch time — the schema's value is enabling downstream consumers (other ensembles in a composition; the orchestrator's reasoning surface) to parse the structured payload predictably, not enforcing format compliance at the synthesizer agent's output.

Schema declaration is **optional**; existing capability ensembles ship without `output_schema:` in Cycle 6 BUILD. BUILD progressively migrates ensembles whose `default_task` specifies a structured output format. The `text-summarizer` and `web-searcher` ensembles are candidates for early migration (their outputs have natural structure — paragraph text; list of URL+snippet records). The `code-generator` ensemble produces code as its deliverable — `output_schema` for code is awkward; the substrate-routing in ADR-025 makes `output_schema` orthogonal to the deliverable (the code is in `artifacts[]`; `structured` may carry compilation status / lint warnings / etc.).

### Partial update to ADR-021's response shape (currently underspecified)

ADR-021's §Decision §"Per-capability dispatch contract" step 3 reads: *"The orchestrator returns the capability ensemble's output (already summarized per AS-7) to the skill framework as the chat completion response."* This step is **underspecified** on the response shape — it does not name the envelope contract.

ADR-021 carries a dated `> **Updated by ADR-024 on 2026-05-15.**` header. The update is **additive** — the step's existing content (capability ensemble's output, summarized per AS-7) is preserved; the envelope contract is now the typed shape that output takes. The rest of ADR-021 remains current.

### Migration path

The Cycle 5 `execution.json` artifacts under `.llm-orc/artifacts/agentic-serving/<ensemble>/<timestamp>/` already carry the de facto shape (`{ensemble, status, input, results, metadata, synthesis}`). The migration to the typed envelope is:

1. **Cycle 6 BUILD ships the `DispatchEnvelope` dataclass** as the typed return value of `invoke_ensemble`. The conversion from the de facto `execution.json` shape to the typed envelope happens at the dispatch boundary — the `invoke_ensemble` implementation reads the dispatch's `execution.json` and constructs the envelope. The `execution.json` artifact itself **does not change shape** in Cycle 6; it remains the existing artifact format under the existing path.
2. **Migration of `metadata` field name to `diagnostics` is at the envelope layer only.** The `execution.json` artifact retains `metadata`; the envelope's `diagnostics` field is constructed from `execution.json.metadata` + dispatch-event correlation (ADR-023's `dispatch_id`). The artifact-layer rename is deferred to Cycle 7+ if/when an artifact-shape ADR fires.
3. **Capability ensembles incrementally adopt `output_schema:`**. Cycle 6 BUILD adds `output_schema:` to ensembles whose `default_task` specifies a structured format. Operators can author `output_schema:` for their own ensembles independently.

### BUILD-sequencing dependency on ADR-023

`diagnostics.dispatch_id` and `diagnostics.duration_seconds` in the envelope are populated from ADR-023's `DispatchTiming` event + `dispatch_id` correlation identifier. BUILD must ship ADR-023's event-model extension **before or alongside** ADR-024's envelope construction at the dispatch boundary — the envelope's `diagnostics` fields cannot be populated without the underlying event substrate.

The BUILD sequencing:
1. Ship ADR-023's `DispatchTiming` event type + `dispatch_id` field on the existing four event types (`TierSelection`, `CalibrationVerdict`, `AuditDiagnostic`, `CalibrationSignal`). Existing `execution.json` artifacts gain the new fields.
2. Ship ADR-024's `DispatchEnvelope` dataclass + `invoke_ensemble` return-value migration. The envelope construction reads dispatch events for the just-completed dispatch and populates `diagnostics`.

If ADR-023 ships first (no ADR-024 yet), `execution.json` carries the new fields; downstream consumers parsing the existing artifact shape ignore them harmlessly (additive extension). If ADR-024 ships first without ADR-023 (not the planned sequence), `diagnostics.dispatch_id` and `diagnostics.duration_seconds` are absent from the envelope; the envelope's other fields populate normally and the diagnostics gap is the surface-level signal that ADR-023's events have not yet shipped. The planned BUILD sequencing ships ADR-023 first within Cycle 6 BUILD; the fallback is graceful.

### BUILD-assumption note on composition substrate

Spike β's headline finding — composition assumptions live in the orchestrator's reasoning surface between dispatches (selection, reframing, distillation), not in the typed contract — is the load-bearing context for how operators should expect `output_schema:` declarations to behave. The envelope's typed `structured` field opens **structural composition predictability** (downstream consumers parse the same field shape across dispatches); it does **not** eliminate the orchestrator's reasoning-surface role in composition. Operators implementing `output_schema:` declarations should expect:

- The schema makes downstream consumption predictable when the orchestrator passes `envelope.structured` forward without restructuring. For dispatches where the orchestrator's `input.data` for the next stage is hand-written by the orchestrator (spike β's headline mechanism), the schema's value is mechanical-validation surface, not composition-substrate replacement.
- Eliminating the orchestrator's reasoning-surface composition role would require **narrowing the orchestrator's role** from prose-integrator to chain-selector (per spike β's "candidate B-strong" analysis) — a bigger architectural commitment than Cycle 6 contemplates. Cycle 6 ships the envelope contract; the orchestrator's role narrowing is future-cycle territory if/when composition predictability becomes the cycle's central question.

This BUILD-assumption note is preserved here so operators reading the ADR understand the envelope's value-add scope honestly. The envelope is a contract; it is not a composition substrate that displaces the orchestrator's reasoning surface.

### Out of scope for ADR-024

- **Artifact-substrate scope (when `primary` is inline vs. when `primary` is a reference).** ADR-025's territory. ADR-024 specifies the envelope shape including `artifacts[]`; ADR-025 specifies when `artifacts[]` is populated.
- **System-ensemble envelopes.** `agentic-calibration-checker` and `agentic-result-summarizer` are internal infrastructure; their response shapes are consumed by the orchestrator's dispatch loop, not by skill frameworks. They are out of T16's scope. The typed envelope is for capability-ensemble dispatches consumed by the chat completion response.
- **Server-Sent Events (SSE) streaming format.** The orchestrator's serving layer exposes `/v1/chat/completions` with SSE streaming; the envelope describes the *non-streaming* return value or the *terminal chunk* of a streaming response. SSE event sequencing is OpenAI-compatibility territory, not envelope-contract territory.
- **Versioning of the envelope contract.** Cycle 6 ships `DispatchEnvelope` as version 1; future cycles can add fields under the additive-shape principle (existing field semantics preserved; new fields default-optional). A `envelope_version: int` field is not added in Cycle 6 — versioning is deferred until/unless a non-additive change is contemplated.

---

## Rejected alternatives

### Candidate A — Thin convention (`status`, `primary`, `diagnostics` only)

Codify the de facto contract minimally: the envelope carries `status`, `primary`, and `diagnostics`; no `structured`, `errors[]`, or `artifacts[]`. Composition relies on string parsing of `primary`.

**Rejected because:** candidate A addresses none of the three findings T16 was scoped to deliberate (output-spec drift; information-finding overhead; AS-7 summarizer content-stripping). Spike β's per-finding test (MODEL Action B) showed each finding wants a different best-fit remediation — `structured` for drift detection, `artifacts[]` for content-overhead, `artifacts[]` for AS-7 stripping. Candidate A keeps composition string-parsed and downstream consumers blind to the structural shape of upstream output. The de facto shape works for the current cycle's deployment because the orchestrator's reasoning surface is doing the parse-and-restructure work (spike β's reframing); candidate A leaves that load-bearing work in the orchestrator's reasoning rather than moving it into the contract.

### Candidate B-strong — Mandatory `output_schema:` on all capability ensembles with synthesizer-agent schema enforcement

Make `output_schema:` mandatory on all capability ensembles; enforce schema compliance at the synthesizer agent's output before envelope construction. Non-compliant synthesizer outputs surface as `status: error` envelopes.

**Rejected because:** spike β's reframing of output-spec drift as `input.data` override means schema enforcement at the synthesizer's output catches the wrong thing. The synthesizer is producing what `input.data` instructs it to produce; the drift is at the orchestrator's `input.data` authorship, not at the synthesizer's compliance. Schema enforcement at the synthesizer's output would either (a) reject outputs that comply with `input.data` but violate `default_task`-derived schemas — surfacing the drift as synthesizer error when the actual root cause is upstream, or (b) require the schema to be re-derived from `input.data` at run-time, defeating the schema's "pre-specified" property.

Candidate B-strong also creates a chicken-and-egg problem with composition: a downstream ensemble's `input.data` is authored by the orchestrator (per spike β) from the upstream's envelope; the schema for the upstream's output is independent of the downstream's input expectations. Schema enforcement at the upstream's synthesizer doesn't fix the composition predictability problem; it just adds enforcement at a layer that isn't the source.

The advisory schema reading (candidate B as decided here) lets downstream consumers parse `structured` predictably when present, without making schema compliance a synthesizer-agent compliance requirement.

### A separate "response envelope" event type in ADR-023's dispatch-event substrate

Route the envelope through ADR-023's dispatch-event substrate as a new event type (`ResponseEnvelope` event emitted at dispatch end).

**Rejected because:** the envelope is the **return value** of `invoke_ensemble` consumed by the orchestrator's reasoning surface and the skill framework's chat completion response. Routing it through the dispatch-event substrate would couple the response-shape contract to the observability substrate — two structurally distinct contracts that benefit from independent evolution. ADR-023's events describe *what the dispatch did* (tier selection, verdict, audit findings, timing); the envelope describes *what the ensemble produced* (the deliverable + diagnostics + errors + artifacts). The two surfaces share the `dispatch_id` correlation identifier — that's the right level of coupling. The envelope is the response shape; the events are the observability stream.

---

## Consequences

### Positive

- **The de facto envelope is codified as a typed contract.** Downstream consumers (skill frameworks, the orchestrator's reasoning surface, post-hoc artifact-readers) parse the envelope structurally rather than via string-matching on `primary`. The contract makes the existing implicit shape explicit.
- **`structured` opens composition predictability for spike β's substrate.** Capability ensembles declaring `output_schema:` can be composed without `input.data` re-authorship at each stage — the downstream ensemble consumes `envelope.structured` directly. Spike β's reframing of composition substrate (orchestrator's reasoning between dispatches) is not eliminated; it is given a typed alternative that progressive adoption can move composition into.
- **`errors[]` first-class.** Per-stage and partial-failure errors have a typed home in the envelope, rather than being parsed out of `primary` or read from `execution.json` separately. Typed errors are surfaceable to skill frameworks for their own error handling.
- **`artifacts[]` first-class.** Substrate-routed deliverables (ADR-025) have a structural home. The orchestrator's reasoning surface receives a path + content type + summary, not the deliverable content (spike β's reduction of context cost via substrate-routing has a structural anchor).
- **`diagnostics.dispatch_id` correlates to ADR-023's event substrate.** Envelope diagnostics and dispatch events share the correlation identifier, enabling post-hoc joining and operator review.

### Negative

- **`output_schema:` adoption is per-ensemble, not enforced.** Capability ensembles that do not declare `output_schema:` have `envelope.structured = None`; downstream composition continues to rely on the orchestrator's reasoning between dispatches. The adoption pace depends on BUILD's progressive migration and operator authorship.
- **Schema validation is advisory at dispatch time.** Schema compliance is not enforced; non-compliant synthesizer output produces an envelope with mismatched `structured` content. The advisory shape is honest about spike β's finding (the drift is upstream of the synthesizer) but means schema declaration does not catch all drift at the contract boundary.
- **Field-name rename from `metadata` to `diagnostics` at the envelope layer only.** The `execution.json` artifact retains `metadata`; readers comparing envelope and artifact must understand the rename. Cycle 7+ may produce an artifact-shape ADR that renames the artifact field; until then the discrepancy is a small naming-divergence operators notice.

### Neutral

- **The typed envelope composes with ADR-023's dispatch-event substrate.** Envelope diagnostics include the `dispatch_id` correlation identifier; consumers wanting full dispatch context join envelope and events on that identifier.
- **The typed envelope composes with ADR-025's artifact-as-substrate.** ADR-025 specifies when `artifacts[]` is populated and how `primary` carries the artifact-reference summary; ADR-024 provides the structural home.
- **The typed envelope composes with ADR-007 / ADR-014 calibration-gate verdicts.** The verdict flows into `diagnostics.calibration_verdict`; consumers can route on verdict (e.g., skill frameworks that retry on `Reflect` verdicts) without parsing dispatch events separately.

## Provenance check

- **De facto envelope already exists uniformly**: spike α (driver). Driver chain: same-cycle-spike-derived.
- **Output-spec drift's actual mechanism is `input.data` override**: spike β (driver). Driver chain: same-cycle-spike-derived.
- **Candidate B selection as the lowest-disruption typed-field path**: spike α (driver). Driver chain: same-cycle-spike-derived.
- **`structured` field as load-bearing distinct from `primary`**: spike α (driver) extending Cycle 6 MODEL §Concepts entry on Common I/O envelope. Driver chain: same-cycle-spike-derived + same-cycle-MODEL-derived.
- **`errors[]` field as load-bearing**: spike α (driver). Driver chain: same-cycle-spike-derived.
- **Field-name `diagnostics` (not `metadata`)**: Cycle 6 MODEL §Concepts entry on Common I/O envelope (driver). Driver chain: same-cycle-MODEL-derived.
- **`output_schema:` advisory at dispatch time**: spike β's `input.data`-override finding (driver) — the synthesizer is not the drift source; enforcement at the synthesizer would catch the wrong thing. Driver chain: same-cycle-spike-derived.
- **Rejection of mandatory schema enforcement**: drafting-time synthesis composing spike β's finding with the candidate B-strong design alternative. The rejection's specific reasoning (chicken-and-egg with composition; enforcement at the wrong layer) is drafting-time examination.
- **Rejection of routing envelope through dispatch-event substrate**: drafting-time synthesis applying the response-shape-vs-observability-substrate distinction. The two surfaces' independence is a methodology-pattern observation, not a driver-derived finding.
- **Migration path (artifact retains `metadata`; envelope uses `diagnostics`)**: drafting-time synthesis honoring the bounded-change principle (Cycle 6 ships envelope contract; artifact-shape ADR is future-cycle territory). Not driver-derived; design-time scoping judgment.

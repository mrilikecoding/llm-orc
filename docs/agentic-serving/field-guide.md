# Field Guide: Agentic Serving

**Generated:** 2026-07-08
**Derived from:** system-design §Cycle 8 (Target Architecture, ARCHITECT 2026-07-02) + current implementation (post-WP-F8 collapse).

## How to use this guide

This document maps each module of the Cycle-8 declarative serving path to
its current implementation state. It is a reference — consult the entry for
the module being worked in or explored. For the overall architecture, read
`system-design.md` §Cycle 8 — Target Architecture. For routing to the right
document, read `ORIENTATION.md`.

The bespoke imperative `src/llm_orc/agentic/` layer (loop driver,
orchestrator runtime, tool dispatch, calibration, tier router, autonomy,
budget, summarizer harness, and the rest) was deleted at Cycle-8 WP-F8. The
declarative **Serving Ensemble** — one YAML pipeline executed by the L0
Ensemble Engine — is now the only serving path. Entries for the dissolved
modules have been removed; the historical record of that architecture lives
in `system-design.md`'s retained pre-Cycle-8 sections and its
`system-design.agents.md` companion.

State vocabulary:

- **Complete** — all named responsibilities implemented; production wiring
  in place; tests exercise the boundary.
- **Partial** — skeleton or subset implemented; a follow-up is open.
- **Planned** — not yet implemented; design is stable.

Stability vocabulary:

- **Settled** — unlikely to change outside a named follow-up. Invest
  understanding here confidently.
- **In flux** — under active development or pending an adjacent change that
  will touch it.
- **Design-only** — system-design has the contract; no code yet.

The declarative pipeline, the registry/catalog, and the accept gate all
shipped in Cycle 8 and are grounded against real OpenCode runs. The skeleton
shape, the AS-2 admission routine, the session substrate, the envelope, and
the L0 engine are **Settled**. The named frontier is **In flux**: default
seats beyond code and explain (fix / edit / run-tests), the
compose-at-runtime shape (`gen-review`), and the two open decision points
(the accept-loop location and the executor's home).

---

## Module: Serving Layer (L3)

**Implementation state:** Complete. Every `/v1/chat/completions` request is
handled by the declarative Serving Ensemble through a thin caller; `/v1/models`
advertises the model-profile allowlist.
**Code location:** `src/llm_orc/web/api/v1_chat_completions.py`,
`src/llm_orc/web/api/v1_models.py`, `src/llm_orc/web/api/sse_format.py`,
`src/llm_orc/web/serving/serving_ensemble_caller.py`,
`src/llm_orc/web/serving/session_start.py`,
`src/llm_orc/web/serving/chunks.py`, `src/llm_orc/web/serving/turn_trace.py`.
**Stability:** Settled on transport reuse (session resolution, SSE, body
shaping). The one client-shaped concern in the caller — the toolless meta-call
discrimination OpenCode needs — is the piece most likely to grow.

### Domain concepts in code

| Concept | Code manifestation | Location |
|---------|-------------------|----------|
| `/v1/chat/completions` endpoint | `chat_completions` | `v1_chat_completions.py` |
| `/v1/models` endpoint | `list_models` | `v1_models.py` |
| Declarative caller (the handoff to the ensemble) | `ServingEnsembleCaller.run` | `serving_ensemble_caller.py` |
| Caller Protocol on the endpoint | `_ChatCompletionsCaller` | `v1_chat_completions.py` |
| Session context snapshot | `SessionContext` dataclass | `web/serving/session_start.py` |
| Contract message type | `ChatMessage` (re-exported from the substrate) | `web/serving/session_start.py` |
| Session-start cache (once-per-session, FC-9) | `SessionStartCache` | `web/serving/session_start.py` |
| Phase-2 Plexus hook (reserved) | `resolve_session_start_context` + `PromptFragment` | `web/serving/session_start.py` |
| Chunk vocabulary (endpoint ↔ caller) | `ContentDelta`, `Completion`, `ClientToolCall`, `ToolCallInvocation` | `web/serving/chunks.py` |
| Non-streaming collector | `_collect_non_streaming` + `_build_completion_body` | `v1_chat_completions.py` |
| SSE streaming | `_stream_completion` + `OpenAiSseFormatter` | `v1_chat_completions.py`, `sse_format.py` |
| Serving-turn introspection | `emit_turn_trace` / `build_turn_trace` | `web/serving/turn_trace.py` |

### Design rationale

The endpoint parses the OpenAI-compatible request, resolves a Session via
`SessionRegistry` (session-start context resolves once per identity, FC-9),
and hands the request to `ServingEnsembleCaller`. The caller runs the
declarative Serving Ensemble and yields the shared chunk vocabulary
(`ContentDelta` / `Completion` / `ClientToolCall`), so the streaming SSE
formatter and the non-streaming body collector are reused unchanged. A build
turn closes with a `ClientToolCall` (mapped to `message.tool_calls` +
`finish_reason: tool_calls`); an explain turn closes with `ContentDelta` text
plus a `stop` `Completion`.

The caller is deliberately thin. Its only client-shaped concern is the
toolless meta-call discrimination OpenCode requires: session-title and summary
requests arrive with no `tools[]` and must not drive the build pipeline, so
`ServingEnsembleCaller.run` short-circuits them to a short prose reply
(`_aux_reply`). Everything else lives in the ensemble.

`web/serving/chunks.py` retains several variants from the dissolved
orchestrator era (`InternalToolCallInFlight`, `InternalToolCallResult`,
`VisibilityEvent`, `ErrorChunk`) and their docstrings still reference the
Orchestrator Runtime / ReAct loop / ADR-003 five-tool surface. Only
`ContentDelta`, `Completion`, and `ClientToolCall` are produced on the
Cycle-8 path; the rest are vestigial.

`turn_trace.py` reads the engine's execution result into a per-node
JSONL record (`<project>/.serve-trace/turns.jsonl`) plus a one-line stderr
summary, unwrapping a dispatched seat's child result so an operator can see
the model's real output inside the seat. Tracing never breaks the serve (IO
errors are swallowed).

### Key integration points

- **→ Serving Ensemble:** `ServingEnsembleCaller._serve` loads
  `serving.yaml`, builds a root executor via
  `ExecutorFactory.create_root_executor`, and executes it against the task.
- **→ Session Registry:** `resolve_identity` + `get_or_create_state` in
  `_resolve_context`.
- **→ Model Profile Allowlist:** `list_models` reads
  `ModelProfileAllowlist.list_allowed_model_profile_ids`.

---

## Module: Serving Ensemble (the declarative per-turn handler)

**Implementation state:** Complete. The per-turn flow is ONE declarative
ensemble (`classify → decide[guarded] → resolve → seat → seat_contract →
shape → form_gate → emit`) executed by the L0 Ensemble Engine. No imperative
driver module exists.
**Code location:** `.llm-orc/ensembles/agentic-serving/serving.yaml`; node
scripts under `.llm-orc/scripts/agentic_serving/`.
**Stability:** Settled on the skeleton shape (swapping a seat is a routing /
registry change, never a skeleton edit). The pipeline is grounded against
real OpenCode runs.

### Pipeline in code

| Node | Kind | Script / target | Depends on |
|------|------|-----------------|-----------|
| `classify` | script | `classify.py` | — |
| `decide` | model (guarded `when: ${classify.needs_decider}`) | `agentic-tier-cheap-general` | classify |
| `resolve` | script | `resolve.py` | classify, decide |
| `seat` | dynamic dispatch | `${resolve.target}` | resolve |
| `seat_contract` | script | `seat_contract.py` | resolve, seat |
| `shape` | script | `shape.py` | resolve, seat, seat_contract |
| `form_gate` | script | `form_gate.py` | shape |
| `emit` | script | `emit.py` | form_gate |

### Design rationale

The Cycle-8 collapse is subtraction: the L2 imperative Runtime cluster is
gone and its work is done by the L0 engine executing this one ensemble on
primitives that already shipped (guard `when:`, dynamic dispatch, bounded
`loop:`). The control flow lives in the YAML DAG, not a script — AS-11 (no
parallel orchestration layer). Nodes communicate through the shared
`${dep.field}` reference resolver against the engine's `results_dict`; each
node reads its upstream dependencies' `response` string and emits JSON.

Two granularities of gate ride the pipeline and compose: `seat_contract` is
the per-seat admission check (WP-E8, below); the loop-level accept verdict is
carried inside the build shape's envelope diagnostics (WP-D8, below). `emit`
treats a `seat_admitted: false` as the higher-priority refusal and an
`accept: false` as an another-round signal.

### Key integration points

- **← Serving Layer:** invoked by `ServingEnsembleCaller`.
- **→ L0 Ensemble Engine:** executed as a root ensemble; the `seat` node uses
  the dynamic-dispatch primitive to run the resolved shape as a child.

---

## Module: classify / decide / resolve (the decider seat)

**Implementation state:** Complete. Deterministic where the routing signal is
structural, model-backed only on a guarded ambiguous path.
**Code location:** `.llm-orc/scripts/agentic_serving/classify.py`,
`resolve.py`; the `decide` node is a guarded model node inline in
`serving.yaml`.
**Stability:** Settled on the deterministic-first structure. The default
target set (`code-seat`, `explainer`) is part of the In-flux frontier.

### Domain concepts in code

| Concept | Code manifestation | Location |
|---------|-------------------|----------|
| Structural routing signal | `_EXPLAIN_MARKERS`, `_BUILD_RE`, `_FILE_RE` | `classify.py` |
| build-vs-non-build (executable-deliverable) determination | `build` field on classify output | `classify.py` |
| Deferral to the model decider | `needs_decider: true` + empty `target` | `classify.py` |
| Guarded model decider | `decide` node, `when: ${classify.needs_decider}` | `serving.yaml` |
| Closed target set | `_DERIVED` (`code-seat`, `explainer`) | `resolve.py` |
| Decider-output parse (strict then single-token) | `_decider_target` | `resolve.py` |
| Intent → shape mapping | `shape_catalog(...)` lookup | `resolve.py` |

### Design rationale

`classify.py` routes deterministically: an explain marker → `explainer`
(non-build prose); a build verb or a named target file → `code-seat` (build).
When neither structural signal resolves the turn, classify does not guess — it
emits `needs_decider: true` with an empty `target`, and the guarded `decide`
model node reads the intent and picks from the closed set. `resolve.py` merges
the two: when classify resolved structurally, its decision passes through and
`decide` never ran; otherwise resolve reads the bounded decider target and
derives `build`/`kind`. An out-of-set decider output leaves `target` empty so
the dispatch node fails deterministically rather than defaulting a seat
(determinism-over-carve-outs).

`resolve.py` then maps the semantic intent (`code-seat` / `explainer`) to the
serving shape via the operator-curated Shape Catalog: `build-gated` serves the
`code-seat` intent (the accept gate is default-on for build turns),
`explainer` serves the `explainer` intent. An intent with no catalog shape
passes through unchanged.

**Note vs system-design.** system-design §Cycle 8 names one `classify (decider
seat)` module. The implementation realizes it as three nodes —
deterministic `classify`, a guarded model `decide`, and a merging `resolve` —
and locates the intent→shape resolution in `resolve` (reading the Shape
Catalog) rather than in classify. The responsibility allocation is unchanged;
the decomposition is finer.

### Key integration points

- **→ Shape Catalog:** `resolve.py` calls `shape_catalog(_CATALOG_DIR)`.
- **→ seat:** `resolve` emits `target`, which the `seat` node dispatches on.

---

## Module: seat (dynamic-dispatch node)

**Implementation state:** Complete. Resolves the runtime-chosen target
`${resolve.target}` at the phase layer and runs the resolved ensemble as a
child.
**Code location:** engine primitives at
`src/llm_orc/core/execution/phases/dispatch_resolver.py` and
`src/llm_orc/core/execution/runners/dynamic_dispatch_runner.py`; the schema at
`src/llm_orc/schemas/agent_config.py` (`DynamicDispatchAgentConfig`).
**Stability:** Settled. This is the shipped dynamic-dispatch primitive; the
serving seat is one caller.

### Domain concepts in code

| Concept | Code manifestation | Location |
|---------|-------------------|----------|
| Dispatch agent config | `DynamicDispatchAgentConfig` (`dispatch`, `dispatch_resolved`) | `schemas/agent_config.py` |
| Phase-layer target resolution | `DispatchResolver.resolve_targets` | `phases/dispatch_resolver.py` |
| Reference resolution (`${dep.field}`) | `resolve_reference` | `phases/reference.py` |
| Child-execution boundary | `DynamicDispatchRunner.execute` | `runners/dynamic_dispatch_runner.py` |
| Depth-limit guard | `child_depth > self._depth_limit` | `runners/dynamic_dispatch_runner.py` |

### Design rationale

Sibling to the guard partition: where the guard decides *whether* a node runs,
the dispatch resolver decides *which* ensemble a dispatch node runs. It
resolves the node's `dispatch` reference against accumulated upstream results
(the same resolution the guard uses), records the resolved name on a runtime
copy of the config (`dispatch_resolved`, never mutating the original), and the
runner reads it, loads that ensemble, and executes it through a child
executor. Deterministic, no model involvement in resolution. An unresolved
target surfaces as a runtime error before any child runs.

### Key integration points

- **→ Capability parts / shapes:** the resolved target is a registered shape
  (`build-gated`, `explainer`, `gen-review`) or part.
- **→ L0 Ensemble Engine:** `create_child_executor` runs the resolved
  ensemble under the depth bound.

---

## Module: marshal sub-sequence — shape / form-gate / emit

**Implementation state:** Complete. The finalize step is three bounded nodes,
one responsibility each (AS-11).
**Code location:** `.llm-orc/scripts/agentic_serving/shape.py`,
`form_gate.py`, `emit.py`.
**Stability:** Settled.

### Domain concepts in code

| Concept | Code manifestation | Location |
|---------|-------------------|----------|
| Fidelity marshalling (deliverable from envelope) | `_envelope_deliverable` (`artifacts[0].content`, else `primary`) | `shape.py` |
| Destination from the routing decision | reads `resolve` (else `classify`) `file` / `build` | `shape.py` |
| Graceful degrade for a raw-prose seat | `_terminal` fallthrough when no envelope | `shape.py` |
| Accept verdict passthrough | `_envelope_verdict` (reads envelope `diagnostics.accept`) | `shape.py` |
| Seat-admission verdict passthrough | `_seat_verdict` (reads `seat_contract` node) | `shape.py` |
| Destination-validity check | `_validity` (`ast.parse` for `.py`, `json.loads` for `.json`) | `form_gate.py` |
| Client-seam emission | `main` outcome shaping | `emit.py` |

### Design rationale

`shape` (re-homes the retired Artifact Bridge, ADR-034) reads the seat's
ADR-024 envelope and places the deliverable faithfully: content from the
envelope, destination path and build flag from the routing decision.
Consumers read `artifacts` / `structured`, never parse `primary`
structurally. When the seat emitted raw prose instead of an envelope (the
`explainer` seat), shape degrades to the terminal text so a code artifact's
structure is not imposed on an explanation.

`form_gate` (re-homes the ADR-035 form gate) applies the cheapest rung of the
verification ladder: a `.py` deliverable must `ast.parse`, a `.json`
deliverable must load. A deliverable that does not parse as its path claims is
refused before it reaches the client. A non-build turn is inert here.

`emit` (re-homes the Client-Tool-Action Terminal, ADR-034) is the terminal
node and the reusable client permission seam. It shapes the serve outcome:

- `seat_admitted is False` → prose refusal (highest priority; WP-E8).
- build + `accept is False` → another-round prose (the client owns the loop;
  ODP-2).
- build + valid → `{finish: false, file, content}` (a client file-write).
- build + refused by form-gate → prose refusal.
- non-build → prose finish.

`ServingEnsembleCaller._outcome_chunks` maps a `finish: false` outcome to a
`ClientToolCall` (a `write` tool invocation) and a `finish: true` outcome to
`ContentDelta` + `Completion`.

### Key integration points

- **→ I/O Envelope:** shape reads the seat's ADR-024 envelope shape.
- **→ client tool surface:** emit's outcome becomes a `tool_calls` turn or a
  prose finish at the Serving Layer.

---

## Module: Accept Gate (a build-shape composition, not a framework module)

**Implementation state:** Complete for the single-pass gate. The gate runs
once and carries its verdict in the deliverable envelope; the another-round
loop is client-owned.
**Code location:** `.llm-orc/ensembles/agentic-serving/build-gated.yaml`;
scripts `accept_gather.py`, `accept_executor.py`, `accept_executor_runner.py`,
`accept_gate.py`, `build_gated_envelope.py`.
**Stability:** Settled on the executor / judge / gate structure and the strict
`AND` composition. In flux on the tuning ADR-048 §5 flagged (false-reject
rate, per-round budget) and on the two open decision points below.

### The build shape's pipeline

| Node | Role | Script / target |
|------|------|-----------------|
| `test_writer` | tests-first from acceptance criteria | ensemble `test-writer` |
| `code_writer` | code that should pass those tests | ensemble `code-generator` |
| `gather` | assemble `{requirement, code, tests}` for the executor | `accept_gather.py` |
| `executor` | deterministic sandbox: run tests against code | `accept_executor.py` |
| `judge` | isolated: are the tests adequate to verify the criteria? | model (`agentic-tier-cheap-general`) |
| `accept_gate` | `accept = tests_pass AND tests_adequate` | `accept_gate.py` |
| `envelope` | ADR-024 deliverable + accept verdict in diagnostics | `build_gated_envelope.py` |

### Design rationale

Per the ensemble-spirit / one-responsibility rule, the accept gate is a
composition of nodes inside the build shape, not a framework module. The two
verdict inputs catch orthogonal failures: the `executor` is deterministic
ground truth (a wrong implementation fails real tests); the `judge` is a
fresh-context reviewer (trivially-tested or under-covering outputs the
executor would pass). `accept_gate` `AND`s them — the `AND` lives in a script
node because the guard predicate grammar is truthiness / `== literal` only.

Independence (ADR-048 §3) is enforced by seat isolation: the executor and
judge receive only `{requirement, produced code, produced tests, execution
result}`, never builder reasoning. `accept_executor.py` runs the tests in a
subprocess with a wall-clock timeout (`LLM_ORC_ACCEPT_EXECUTOR_TIMEOUT`,
default 15s), so a runaway or crashing test is reported as a failure, never a
frozen serve.

The `envelope` terminal carries the code deliverable in `artifacts[0]` /
`primary` and the accept verdict (`accept`, `accept_reason`, `tests_pass`,
`tests_adequate`) in ADR-024 `diagnostics`. An accept-rejected turn still
emits a well-formed envelope (the code is present), so the per-seat contract
admits it and the marshal's `emit` handles the another-round decision.

**Open decision points, as resolved in shipped code.** system-design held two
points open; the shipped build shape resolves them provisionally:

- **Executor home (ODP-1):** internal sandbox (subprocess), not
  client-delegated. `emit` reuse for a client-delegated test-run is not wired.
- **Accept-loop location (ODP-2):** the client's outer re-invocation loop. No
  serving shape uses the bounded `loop:` primitive; the gate runs once and
  `emit` returns an another-round prose signal on reject.

### Key integration points

- **← seat (dispatch):** `build-gated` is dispatched as the `code-seat` shape.
- **→ marshal:** the envelope's `diagnostics.accept` flows through
  `shape` → `form_gate` → `emit`.

---

## Module: Shape Catalog + Capability Registry + Admission (core/serving)

**Implementation state:** Complete (WP-C8). The ADR-047 registry is not a new
module; it decomposes into the shared AS-2 admission routine, the Topaz-keyed
capability parts, and the new Shape Catalog.
**Code location:** `src/llm_orc/core/serving/admission.py`,
`capability_registry.py`, `shape_catalog.py`.
**Stability:** Settled. All three are derived scans (AS-11, no persistent
store) over one AS-2 admission pass.

### Domain concepts in code

| Concept | Code manifestation | Location |
|---------|-------------------|----------|
| AS-2 admission gate (single routine) | `admit` → `validate_ensemble_reference_graph` + `compute_reference_graph_depth` | `admission.py` |
| Admission scan (partition admitted / rejected) | `scan_admitted` → `RegistryScan` | `admission.py` |
| Typed rejection | `Rejected` (`cross_ensemble_cycle`, `depth_limit_exceeded`, `invalid_ensemble`) | `admission.py` |
| Topaz-keyed capability parts | `capability_parts` → `dict[str, list[str]]` | `capability_registry.py` |
| Composition-shape catalog (intent → shape) | `shape_catalog` → `dict[str, str]` | `shape_catalog.py` |
| Registered shape enumeration | `registered_shapes` | `shape_catalog.py` |
| Part vs shape discriminator | `topaz_skill` (part) vs `serves` (shape) on `EnsembleConfig` | `core/config/ensemble_config.py` |

### Design rationale

Admission is the single AS-2 gate: a part or shape becomes dispatchable only
after passing reference-graph validation (no cross-ensemble cycle, within
depth, every reference resolving). `admit` reuses
`validate_ensemble_reference_graph` — the *same* public routine the load path
(`EnsembleLoader.load_from_file`) and the composition path
(`CompositionValidator`) call — so the registry cannot fork a second
validator. The depth check the load path defers to runtime is moved left to
registration, mirroring `CompositionValidator`.

The catalog is derived, not stored: `scan_admitted` walks a library directory,
loads each ensemble, and partitions into admitted configs and typed
rejections. Parts and shapes are views over one scan. `capability_parts` keeps
the admitted ensembles that declare a `topaz_skill` and are not themselves
shapes, grouped by skill. `shape_catalog` keeps the admitted ensembles that
declare a `serves` intent, mapping intent → shape name. Standing comes only
from operator curation (declaring `topaz_skill` / `serves`) plus AS-2
admission — never from accumulated usage; there is deliberately no
promote/stabilize surface (the retired AS-5 trust-promotion loop).

`gen-review.yaml` is the ADR-047 §2 exemplar: a `gen → review` shape whose
both slots are filled by dynamic dispatch resolving *registered* parts
(`select_parts.py` reads `capability_parts` and picks a part per capability),
not static ensemble refs. This compose-at-runtime shape is the In-flux
frontier.

### Key integration points

- **← resolve (classify):** `resolve.py` calls `shape_catalog` to map intent
  → shape.
- **← select_parts:** `gen-review`'s selector calls `capability_parts`.
- **→ Ensemble reference-graph validator:** the shared
  `validate_ensemble_reference_graph` / `compute_reference_graph_depth` in
  `core/config/ensemble_config.py`.

---

## Module: Seat Contract (core/validation)

**Implementation state:** Complete (WP-E8). Wires the surviving
`core/validation` framework as each seat's admission gate; runs a real
`ValidationEvaluator.evaluate`, closing the ADR-046 §2 F3 wiring gap.
**Code location:** `src/llm_orc/core/validation/seat_contract.py`; the node
script `.llm-orc/scripts/agentic_serving/seat_contract.py`; framework at
`src/llm_orc/core/validation/evaluator.py`, `models.py`.
**Stability:** Settled on the black-box, deterministic-first projection.

### Domain concepts in code

| Concept | Code manifestation | Location |
|---------|-------------------|----------|
| Seat admission result | `SeatAdmission` (`admitted`, `reason`, `result`) | `core/validation/seat_contract.py` |
| Black-box / deterministic-first projection | `admissible_layers` (drops `structural` + `semantic`) | `core/validation/seat_contract.py` |
| Admission run | `admit(seat_name, seat_output, contract)` → real `ValidationEvaluator.evaluate` | `core/validation/seat_contract.py` |
| Canonical seat-output key | `SEAT_OUTPUT_KEY = "seat"` | `core/validation/seat_contract.py` |
| Seat-owned contract source | the resolved seat's `seat_contract:` YAML block | seat ensemble YAML |
| Node wiring | `_contract_for` + `asyncio.run(admit(...))` | `scripts/agentic_serving/seat_contract.py` |

### Design rationale

ADR-046 §2 makes the seats swappable behind contracts, and the contract IS the
surviving `core/validation` framework. Three conventions:

- **Seat-owned, not candidate-owned** — the seat declares the contract (its
  `seat_contract:` block) and the skeleton applies it to the candidate's
  output. The candidate never grades its own homework (the §6.2b independence
  trap).
- **Black-box** — the `structural` layer names internal agents
  (`required_agents`) and would couple a seat to a candidate's internals, so it
  is dropped; any `code_generation` ensemble can fill the seat.
- **Deterministic-first** — the `semantic` (LLM-judge) layer is advisory, so
  it is dropped; eligibility rests on the deterministic I/O-facing layers
  (schema / behavioral / quantitative).

The node adapts the seat's output under `SEAT_OUTPUT_KEY` into an
`EnsembleExecutionResult` and runs the evaluator against the projected
contract. A seat with no contract (an ungated seat, the raw-prose explainer)
is vacuously admitted. Note the distinction from the engine's own
`validation:` field, which the engine auto-runs against an ensemble's own
agents; the seat contract references the skeleton's `seat` adapter key.

### Key integration points

- **← seat_contract node:** loads the resolved seat's `seat_contract:` block,
  calls `admit`.
- **→ marshal:** `shape` reads the node's `seat_admitted`; `emit` refuses on
  an explicit `False`.

---

## Module: Session substrate (core/session)

**Implementation state:** Complete. The surviving cross-turn state containers,
relocated out of `agentic/` at Cycle-8 WP-B8 (the hard prerequisite before the
deletion).
**Code location:** `src/llm_orc/core/session/` — `registry.py`,
`artifacts.py`, `artifact_store.py`, `compaction.py`, `plexus_adapter.py`,
`messages.py`.
**Stability:** Settled. These are mature Cycle 4–7 modules (ADR-013 / 025 /
012 / 009-010) carried through the collapse unchanged. Several module
docstrings still reference the dissolved Orchestrator Runtime / ReAct loop /
WP-I/WP-K sequencing (`plexus_adapter.py`, `registry.py`); the code is
current, the prose narration is not yet swept.

### Domain concepts in code

| Concept | Code manifestation | Location |
|---------|-------------------|----------|
| Session identity + state | `SessionIdentity`, `SessionState` (`turn_count`, `token_spend`) | `registry.py` |
| Per-process registry | `SessionRegistry` (`resolve_identity`, `get_or_create_state`) | `registry.py` |
| Close-lifecycle fan-out | `register_close_callback`, `close_session` | `registry.py` |
| Cluster determination (ADR-013 i) | `resolve_cluster`, `resolve_session_cluster` | `registry.py` |
| Structured-handoff artifact write-gate | `artifacts.py` module | `artifacts.py` |
| Session Artifact Store (ADR-025) | `artifact_store.py` module | `artifact_store.py` |
| Conversation Compaction (ADR-012) | `compaction.py` five-layer pipeline | `compaction.py` |
| Optional KG substrate adapter | `PlexusAdapter.query` / `.record` (no-op fallbacks) | `plexus_adapter.py` |
| Shared `ChatMessage` value type | `ChatMessage` frozen dataclass | `messages.py` |

### Design rationale

Per ADR-046 §3, the client owns the multi-turn loop and cross-turn persistence
relocates to the substrate. `SessionRegistry` derives identity from request
features (user field → message-prefix hash → cold-start UUID) so a client
continues a Session across requests without a bespoke header, and tracks
cumulative turn / token accounting on the mutable `SessionState`. `ChatMessage`
moved here from the dissolved `agentic/session_start` — its two consumers
(registry identity derivation, the serving-layer `SessionContext` contract)
both sit above it. The `PlexusAdapter` remains the single place Plexus-aware
code lives (AS-8: stateless is a real mode, not a degraded one); its bodies
are still no-op fallbacks.

### Key integration points

- **← Serving Layer:** per-request identity resolution + state retrieval.
- **← session-start cache:** `SessionStartCache` keys on `SessionState.identity`.

---

## Shared type: `DispatchEnvelope` (I/O envelope, models/)

**Implementation state:** Complete. The ADR-024 common I/O envelope, relocated
out of `agentic/` to `models/` (the relocation constraint the collapse
required).
**Code location:** `src/llm_orc/models/dispatch_envelope.py`. Seat producers
emit envelope-shaped JSON at `.llm-orc/scripts/agentic_serving/emit_envelope.py`
(code-seat) and `build_gated_envelope.py` (build-gated).
**Stability:** Settled on the six-field shape. The module docstring is stale —
it still describes the type as living "alongside the agentic-layer modules"
with Orchestrator Tool Dispatch as producer.

### Domain concepts in code

| Concept | Code manifestation | Location |
|---------|-------------------|----------|
| Envelope (frozen dataclass, six fields) | `DispatchEnvelope` | `models/dispatch_envelope.py` |
| Status discriminator | `EnvelopeStatus = Literal["success", "error", "timeout", "partial"]` | `models/dispatch_envelope.py` |
| Canonical deliverable | `primary: str` | `models/dispatch_envelope.py` |
| Typed structured payload | `structured: dict \| None` | `models/dispatch_envelope.py` |
| Operator diagnostics | `diagnostics: dict` (carries the accept verdict in the build shape) | `models/dispatch_envelope.py` |
| Artifact references | `artifacts: list[dict] \| None` (`artifacts[0].content` holds the code) | `models/dispatch_envelope.py` |

### Design rationale

The envelope is the inter-seat seam: shape reads it, the seat terminals write
it. In the Cycle-8 serving path the seat terminal scripts
(`emit_envelope.py`, `build_gated_envelope.py`) emit JSON dicts matching the
`DispatchEnvelope` shape rather than importing and serializing the dataclass;
the Python type in `models/` is the shape-of-record, and `shape.py` reads
`artifacts[0].content` (else `primary`) faithfully. The build shape carries the
accept verdict in `diagnostics.accept`; the retired
`calibration_verdict` / `audit_findings` subfields stay absent.

### Key integration points

- **← seat terminals:** `code-seat` and `build-gated` envelope nodes produce it.
- **→ marshal:** `shape.py` consumes it.

---

## Module: L0 Ensemble Engine (core/execution)

**Implementation state:** Complete. The surviving execution core executes the
serving ensemble; it owns the guard / bounded-loop / dynamic-dispatch
primitives.
**Code location:** `src/llm_orc/core/execution/ensemble_execution.py` (plus
`phases/`, `runners/`, `fan_out/`, `results_processor.py`,
`artifact_manager.py` — the full existing subsystem).
**Stability:** Settled. Project-level concern; the control-flow primitives
shipped in Cycle 7.

### Control-flow primitives in code

| Primitive | Code manifestation | Location |
|-----------|-------------------|----------|
| Conditional execution (`when:`) | `GuardEvaluator.should_run` | `phases/guard_evaluator.py` |
| Predicate grammar | `predicate.evaluate` | `phases/predicate.py` |
| `${dep.field}` reference resolver | `resolve_reference` | `phases/reference.py` |
| Dynamic-dispatch resolution | `DispatchResolver.resolve_targets` | `phases/dispatch_resolver.py` |
| Dynamic-dispatch execution | `DynamicDispatchRunner.execute` | `runners/dynamic_dispatch_runner.py` |
| Bounded loop (`loop until:/carry:`) | `LoopAgentRunner` + `LoopController` | `runners/loop_runner.py`, `phases/loop_controller.py` |
| Engine wiring | `EnsembleExecutor` constructs all four | `ensemble_execution.py:243-302` |

### Design rationale

The engine executes the declarative serving ensemble node by node, evaluating
each node's guard, resolving dispatch targets against accumulated results, and
running child executors under a depth bound. The bounded-loop primitive ships
(`LoopAgentRunner`) but no serving shape uses it yet — the accept loop is
client-owned (ODP-2). The ADR-002 layering rule is back to no-exception
(edges never point upward); the ADR-016 L0→L1 calibration-channel exception was
struck with the deletion of the calibration signal channel, so no upward-edge
allowlist remains in the code.

### Key integration points

- **← Serving Layer:** `ExecutorFactory.create_root_executor` builds the root
  executor the caller runs.
- **← seat:** the dynamic-dispatch primitive runs the resolved shape as a
  child.

---

## Module: Model Profile Allowlist (core/config)

**Implementation state:** Complete (WP-F8). The one surviving piece of the
dissolved `agentic/orchestrator_config.py` — the `/v1/models` allow-list.
**Code location:** `src/llm_orc/core/config/model_profile_allowlist.py`.
**Stability:** Settled. The config key it reads
(`agentic_serving.orchestrator.*`) is vestigially named — the orchestrator
actor it once sat beside is gone.

### Domain concepts in code

| Concept | Code manifestation | Location |
|---------|-------------------|----------|
| Allowlist resolver | `ModelProfileAllowlist.list_allowed_model_profile_ids` | `model_profile_allowlist.py` |
| Config source | `agentic_serving.orchestrator.allowed_profiles` / `model_profile` | `model_profile_allowlist.py` |
| Library intersection | filters against `ConfigurationManager.get_model_profiles()` | `model_profile_allowlist.py` |
| Single-profile fallback | `_resolve_allowed_profiles` → `(model_profile,)` | `model_profile_allowlist.py` |

### Design rationale

`/v1/models` advertises the model-profile IDs an operator has exposed so
agentic coding tools can populate their model picker. The allowlist is the
operator-configured list intersected with the Model Profile library; absent
profiles silently drop out (the "shop window" is what is actually resolvable).
Per-request construction picks up `config.yaml` changes without a restart. The
budget / autonomy / calibration / tier / compaction / observability config
surface this used to sit beside dissolved with the actor; only the allowlist
resolution survives.

### Key integration points

- **← `/v1/models`:** `list_models` constructs a `ModelProfileAllowlist` per
  request.
- **→ ConfigurationManager:** reads the agentic-serving config block and the
  model-profile library.

---

## Cross-cutting: composition validation

`src/llm_orc/core/validation/composition_validator.py` and
`tool_call_guard.py` survived the collapse. `CompositionValidator` (the six
rejection branches plus accept) is retained as the third caller of the shared
`validate_ensemble_reference_graph` routine, alongside the load path and the
registry admission gate; its `compose_ensemble` dispatch caller is gone with
the orchestrator, so on the serving path the shared routine is reached through
`core/serving/admission.py` rather than through the validator class. Its
docstring still describes AS-6 composition-time strictness (`missing_primitive`),
which system-design §Cycle 8 records as dropped for serving (AS-2 survives);
the branch remains in the code as part of the retained validator.
`tool_call_guard.py` (the ADR-017 phantom-tool-call guard) survives as a
structural-error utility.

---

## Cross-cutting: `llm-orc serve` command

**Implementation state:** Complete.
**Code location:** `src/llm_orc/cli.py`.

Alias for `llm-orc web` — starts the same FastAPI app with
agentic-serving-oriented CLI framing. Use `serve` for agentic-client
deployments, `web` for the browser UI. `llm-orc mcp serve` is unrelated (MCP
server for direct tool use).

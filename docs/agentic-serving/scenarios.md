# Behavior Scenarios: Agentic Serving

*Derived from domain model, ADRs 001-011, and conformance scan conformance-scan-decide-001. Every term used here comes from the scoped domain model or the project-level domain model.*

---

## Feature: Session Lifecycle (ADR-001, ADR-002, ADR-005, ADR-011)

### Scenario: Tool user completes a task against the stateless orchestrator
**Given** an operator has deployed llm-orc with the serving layer enabled and no Plexus configured
**And** a Session configuration that sets the Budget to a turn limit and token limit sized for an extended agentic coding session
**And** the Orchestrator Agent is parameterized by a Model Profile referencing an available LLM
**When** a tool user points an agentic coding tool at `/v1/chat/completions` and sends a task whose completion requires two ensemble invocations
**Then** the Orchestrator Agent runs its ReAct loop, invokes the relevant ensembles via `invoke_ensemble` tool calls, and returns a completion to the client with no dependency on Plexus being present

### Scenario: Session terminates gracefully on turn limit exhaustion
**Given** an active Session with a Budget whose turn limit will be reached during the next ReAct iteration
**When** the Orchestrator Agent would begin the iteration that exceeds the turn limit
**Then** the session terminates before that iteration begins, the last completed turn's output is returned to the client, and the response explicitly indicates budget exhaustion (the exhaustion is not silent)

### Scenario: Session terminates gracefully on token limit exhaustion
**Given** an active Session whose cumulative token spend has reached the token limit in its Budget
**When** the control plane check fires at the next ReAct iteration boundary
**Then** the session terminates at that iteration boundary, no partial tool call spans the boundary, and the response explicitly indicates token budget exhaustion

### Scenario: Orchestrator LLM profile change is a session-boundary event
**Given** a running Session parameterized by Model Profile `profile-A`
**When** an operator updates the serving layer configuration to `profile-B` mid-session
**Then** the active Session continues on `profile-A` until it terminates, and the next Session created after the configuration change uses `profile-B`

### Scenario: Four-layer stack operates with Plexus present
**Given** an operator has deployed llm-orc with the serving layer, Orchestrator Agent, Ensemble Engine, and Plexus all active
**And** Plexus has not yet been populated
**When** a tool user sends a task whose completion requires one ensemble invocation followed by a knowledge query
**Then** the Orchestrator Agent calls `query_knowledge` and receives an empty result set, completes the task via `invoke_ensemble`, and calls `record_outcome` to log the Routing Decision — the session succeeds despite the knowledge graph being cold

---

## Feature: Orchestrator Tool Surface (ADR-003, ADR-004)

### Scenario: Orchestrator tool surface is exactly the committed set
**Given** an Orchestrator Agent running a ReAct iteration through the Orchestrator Tool Dispatch
**When** the set of tools the Orchestrator Tool Dispatch exposes is enumerated
**Then** it contains exactly five entries: `invoke_ensemble`, `compose_ensemble`, `list_ensembles`, `query_knowledge`, `record_outcome` — and no others

### Scenario: Ensemble result is summarized before entering orchestrator context
**Given** an Orchestrator Agent executing a ReAct iteration
**When** the agent calls `invoke_ensemble` on an ensemble whose full result dictionary is larger than the per-invocation pass-through threshold
**Then** a summarizer runs between the ensemble completing and the tool-call result returning to the orchestrator, the orchestrator's context receives only the summary, and the full result dictionary is preserved in the ensemble execution artifact on disk

### Scenario: Raw-output escape hatch is explicit
**Given** an ensemble configured with the raw-output escape-hatch flag (per ADR-004)
**When** the Orchestrator Agent calls `invoke_ensemble` on that ensemble
**Then** the raw result is passed directly into the orchestrator's context without invoking the summarizer, and the behavior is opt-in — not a default

### Scenario: Invocation outside the tool set is rejected
**Given** an Orchestrator Agent whose Model Profile's LLM emits a tool-call for a function name not in the fixed tool set
**When** the serving layer parses the streamed tool call
**Then** the serving layer returns a tool error to the orchestrator naming the invalid tool, the orchestrator's ReAct loop continues with the error observation, and no action is attempted outside the fixed set

---

## Feature: Ensemble Composition with Validation (ADR-006)

### Scenario: Composition with only profiles and scripts succeeds
**Given** a library containing several model profiles and scripts but zero ensembles
**When** the Orchestrator Agent calls `compose_ensemble` with a specification referencing two profiles and one script
**Then** composition succeeds, validation runs and passes (no ensemble references to check), the new ensemble is written to the local tier, and `list_ensembles` includes it immediately

### Scenario: Composition with ensemble-to-ensemble reference passes validation
**Given** a library containing existing ensembles `A` and `B` with no cyclic references between them
**When** the Orchestrator Agent calls `compose_ensemble` specifying an ensemble agent that references `A`
**Then** composition-time validation resolves the reference against the existing ensemble reference graph, Invariant 5 (cross-ensemble acyclicity) and Invariant 7 (static references) are satisfied, and the composed ensemble is written to the local tier

### Scenario: Composition that would introduce a reference-graph cycle fails at composition time
**Given** a library containing ensemble `A` that references ensemble `B`
**When** the Orchestrator Agent calls `compose_ensemble` with a specification whose new ensemble `C` references `A`, and asks for `B` to be updated to reference `C` (forming an A → B → C → A cycle)
**Then** validation fires at composition time, the composition returns an error describing the cycle by name, no partial ensemble state persists, and Invariant 5 is not violated

### Scenario: Composition referencing a non-existent primitive fails at composition time
**Given** a library containing no profile named `nonexistent-profile`
**When** the Orchestrator Agent calls `compose_ensemble` with a specification referencing `nonexistent-profile`
**Then** validation fires at composition time, the composition returns an error naming the missing primitive, and AS-6 (compose from existing primitives only) is not violated

### Scenario: Composition that would exceed the ensemble recursion depth limit fails at composition time
**Given** a library where composing the proposed ensemble would push the reference graph beyond the configured depth limit (project-level Invariant 8)
**When** the Orchestrator Agent calls `compose_ensemble` with that specification
**Then** validation rejects the composition at composition time, not at load time of a later execution

### Scenario: Composition never authors scripts or profiles
**Given** an Orchestrator Agent at any Autonomy Level
**When** the agent's tool calls are inspected across an entire session
**Then** no tool call results in the creation of a new script file or a new model profile; only new ensemble configurations appear in the library

### Scenario (integration): composed ensemble validates using the same logic as the load path
**Given** the load-path cross-ensemble cycle validator and the composition-time validator
**When** the same proposed ensemble specification is validated by both paths
**Then** both paths return identical outcomes (pass or fail) on the same input, and their implementations share a single underlying validator routine (not duplicated logic)

---

## Feature: Client Tool Surface Commitment (system-design §Client Tool Surface Commitment)

*The Commitment is Option C: client-declared tools flow through turn-boundary delegation via `finish_reason: tool_calls`. The orchestrator's internal action space stays at the five ADR-003 tools. These four scenarios exercise the turn-boundary-vs-mid-execution distinction that the scenario gate was established to test. Scenarios (a) and (b) are the intended Option C cases. Scenarios (c) and (d) probe whether un-predicted client-side needs inside an ensemble force mid-execution callback; the resolution is that re-invocation of the ensemble with the client-tool result folded into `input_data` carries the case without any change to Layer 3 (ADR-001, ADR-002). Mid-execution callback (Option D) is out of scope for this cycle — it would require amending ADR-001 and ADR-002 and adding suspend/resume to the DAG engine's `_execute_core` phase loop (currently synchronous and atomic).*

*"Client-declared tool" and "client tool" in these scenarios refer to entries in the `tools[]` array on a `/v1/chat/completions` request (per the Commitment). The orchestrator's internal Orchestrator Tools (the fixed five per ADR-003) are distinct.*

### Scenario: Orchestrator delegates a client-declared tool at the turn boundary
**Given** an operator has deployed llm-orc with the Serving Layer and no Plexus
**And** a Session whose initial `/v1/chat/completions` request carries `tools: [bash, file_read, file_edit]`
**When** the Orchestrator Agent's ReAct iteration determines that task progress requires reading a file from the client's filesystem
**Then** the Orchestrator Runtime closes the current turn with `finish_reason: tool_calls`, the completion response carries a `tool_calls[]` entry for the client-declared `file_read` tool with the requested path as arguments, and no Orchestrator Tool from the fixed five is dispatched during that turn (the Runtime's tool surface is not invoked; the turn closes by emitting a `ClientToolCall` on the response surface instead)

### Scenario: Session turn count and token spend accumulate across a client-tool round trip
**Given** an active Session whose Budget state is `turn_count=5` and `token_spend=12000` at the moment the Orchestrator Runtime closes a turn with a `file_read` client-tool delegation
**When** the client executes the file read and sends the next `/v1/chat/completions` carrying the accumulated message history plus a `role: tool` message with the file content
**Then** the Serving Layer resolves the request to the same `SessionState` the prior request used, the Orchestrator Runtime resumes its ReAct loop with the `role: tool` message as the next observation, the Session's turn count continues accumulating from 5 (not reset to 0), and the Session's token spend continues accumulating on the same Budget toward the same turn and token limits

### Scenario: Ensemble whose first agent needs a client-filesystem file is handled via pre-invoke delegation
**Given** an existing ensemble `auth-analyzer` whose first phase consumes source file content as its `input_data`
**And** a Session whose client declared `tools: [file_read]`
**When** the Orchestrator Agent, having read `auth-analyzer`'s description via `list_ensembles` and inferring from that description (or from prior task context) that the ensemble consumes file content rather than a file path as `input_data` (note: the `list_ensembles` output schema must be sufficiently rich to support this inference — the schema is a WP-F build-time decision), determines it should invoke `auth-analyzer` on `src/auth.py`
**Then** the Orchestrator Agent first emits a `file_read` client-tool delegation at the turn boundary to obtain the contents of `src/auth.py`, the Session resumes on the subsequent request with the file content carried as a `role: tool` observation, the Orchestrator Agent then emits `invoke_ensemble(name="auth-analyzer", input=<task description with file content folded in>)`, the Ensemble Engine executes the ensemble atomically with the content already present in `input_data`, and no change to the DAG engine's phase loop is required

### Scenario: Composed ensemble's un-predicted mid-execution client-tool need is resolved via re-invocation
**Given** a Session in which the Orchestrator Agent has just composed `repo-scanner` — a two-phase ensemble whose second-phase agent's analysis depends on the output of a `bash: grep -r "TODO" /client/repo` that the Orchestrator Agent did not predict at `invoke_ensemble` time
**And** the composed ensemble follows the build-time convention that an agent emits a structured `{"needs_client_tool": {"tool": "<name>", "args": {...}}}` response when it lacks a required input (convention source: roadmap Open Decision Point #8 — the specific enforcement mechanism is a build-time decision; the scenario assumes *some* mechanism is in place)
**When** the Orchestrator Agent calls `invoke_ensemble(name="repo-scanner", input=<task description>)` and the Ensemble Engine runs the ensemble's phase loop to completion with no external callback (Layer 3 is unchanged; ADR-001, ADR-002)
**Then** the ensemble's Result Summarization preserves the structured `needs_client_tool` signal from the agent that could not proceed (conditional: the summarizer must be configured or constrained to not collapse structured JSON signals into unstructured prose; this is a build-time configuration constraint, not guaranteed by ADR-004 alone), the Orchestrator Agent observes the signal in its tool-call result, the Orchestrator Runtime closes the next turn with a `bash` client-tool delegation at the turn boundary, the Session resumes on the subsequent request with the grep output as a `role: tool` observation, the Orchestrator Agent re-invokes `invoke_ensemble(name="repo-scanner", input=<original task + grep output>)` and the re-invocation runs to completion using the client-tool result folded into `input_data` — the DAG engine never suspends, mid-execution client-tool needs are resolved at turn boundaries via ensemble re-invocation (the retry pattern), and the total Budget impact of the retry is bounded by the Session's turn and token limits

### Scenario: Composed ensemble without the structured signal silently degrades to a quality failure
**Given** a composed ensemble `repo-scanner` whose second-phase agent depends on a client-tool result the Orchestrator Agent did not predict at `invoke_ensemble` time
**And** the ensemble's agents do not emit the structured `{"needs_client_tool": ...}` convention — the ensemble was authored (or composed) without convention compliance, no script-agent precondition guard exists at phase 0, and Orchestrator Tool Dispatch has no structural detection for the schema (this scenario exercises the case where both of Open Decision Point #8's soft mechanisms (i) and (ii) fail to produce a recognized signal)
**When** the Orchestrator Agent calls `invoke_ensemble(name="repo-scanner", input=<task description>)` and the Ensemble Engine runs the ensemble's phase loop to completion
**Then** the ensemble completes with a Result Summarization that has the normal shape (prose output, no `needs_client_tool` key), the Orchestrator Agent receives a normal-shaped tool-call result, no retry is triggered within the ReAct iteration that processes the ensemble result (the orchestrator has no signal that would motivate one), no `ClientToolCall` is emitted on the response surface for this ensemble result, the Session continues within its Budget with turn count and token spend accumulating normally, and the orchestrator's final completion is returned to the client with the ensemble's result as the final answer — the failure is quality-class (structural Session dynamics are correct; the answer is not) rather than correctness-class (no Session crash, no Budget exception, no tool-surface error); catching this failure is the target of the Calibration Gate under ADR-007 when composed ensembles are in their first N invocations, and in stateless deployments where Calibration is session-scoped (Plexus absent, quality signals do not persist across sessions), the quality failure propagates to the client unflagged — this scenario's acceptance is that the Session's structural behavior is correct, not that the result is correct

---

## Feature: Calibration of Composed Ensembles (ADR-007)

### Scenario: First N invocations of a composed ensemble are result-checked
**Given** an ensemble `E` composed in the current Session with `N` set to the default calibration invocation count
**When** the Orchestrator Agent calls `invoke_ensemble` on `E` for the first time
**Then** a calibration check runs against the ensemble's output, a Quality Signal is attached to the Routing Decision, and the same check runs on the next N−1 invocations

### Scenario: Calibration transitions to trusted with sufficient positive quality signals
**Given** a composed ensemble `E` that has completed `N` calibration invocations with positive Quality Signals
**When** the orchestrator calls `invoke_ensemble` on `E` for the (N+1)th time
**Then** the calibration check does not run, `E` is treated as trusted, and the transition to trusted is governed by the accumulated Quality Signals — not by frequency of use (AS-5)

### Scenario: Calibration fails to clear with negative quality signals
**Given** a composed ensemble `E` whose first `N` invocations produced negative or absent Quality Signals
**When** the orchestrator attempts to treat `E` as trusted
**Then** `E` remains in calibration, the calibration period extends or the ensemble is flagged for review per the autonomy policy, and frequency alone does not advance trust

### Scenario: Calibration is session-scoped when Plexus is absent
**Given** a stateless deployment (Plexus absent) and an ensemble `E` that cleared calibration in Session 1
**When** Session 2 starts and the orchestrator calls `invoke_ensemble` on `E`
**Then** `E` re-enters calibration in Session 2, the calibration state from Session 1 is not persisted to any store by this ADR, and the behavior is consistent with ADR-002 stateless baseline

### Scenario: Calibration persists across sessions when Plexus is active
**Given** a Plexus-active deployment and an ensemble `E` that cleared calibration in Session 1
**When** Session 2 starts and the orchestrator queries Plexus for `E`'s calibration state
**Then** the trusted status is visible in the graph and `E` is not re-checked in Session 2 unless operator policy requires re-calibration

---

## Feature: Autonomy and Promotion (ADR-008)

### Scenario: Default Autonomy Level permits invocation, permits composition, gates promotion
**Given** a Session at the default Autonomy Level
**When** the Orchestrator Agent calls `invoke_ensemble` on an existing ensemble, calls `compose_ensemble` to create a new one, and attempts to promote that composed ensemble from local to library tier
**Then** invocation succeeds, composition succeeds and begins calibration, and promotion requires an explicit operator approval outside the orchestrator's tool surface — promotion cannot be self-initiated by the orchestrator

### Scenario: Tool user without operator role observes composition events when configured
**Given** a deployment configured for the pure tool-user persona at a tightened Autonomy Level that surfaces composition events
**When** the Orchestrator Agent composes a new ensemble during a session
**Then** the composition event is surfaced in the response stream in a form the tool user can observe — it is not silent — and the tool user can intercept further composition within the session per the configured level

### Scenario: Pure tool-user session at default Autonomy Level experiences silent composition
**Given** a deployment where the tool user is *not* the operator (product discovery assumption inversion #3 does not apply here — the "endpoint is a model" mental model is in force)
**And** the deployment has not tightened the default Autonomy Level for this persona
**When** the Orchestrator Agent composes a new ensemble mid-session in response to a task whose resolution requires a composed ensemble
**Then** the composition succeeds silently — the tool user receives only the orchestrator's final completion, the composition event is not surfaced in the tool user's response stream, and the behavior matches the default Autonomy Level as documented in ADR-008 (this scenario documents the surprise path explicitly so that deployment operators can decide whether to tighten the default for non-operator tool-user deployments via the ARCHITECT-phase serving-layer configuration surface)

### Scenario: Script authorship is never permitted at any Autonomy Level
**Given** a Session at any Autonomy Level, including the loosest configured
**When** the Orchestrator Agent's tool-call emissions are enumerated
**Then** no tool call authors a script or a model profile — AS-6 is honored regardless of Autonomy Level

---

## Feature: Plexus Integration (ADR-009, ADR-010)

### Scenario: query_knowledge returns empty gracefully when Plexus is absent
**Given** a deployment configured without Plexus
**When** the Orchestrator Agent emits a `query_knowledge` tool call
**Then** the tool returns a well-formed empty result, the orchestrator's ReAct loop continues normally, and no exception surfaces that would terminate the session (AS-8)

### Scenario: query_knowledge returns enriched content when Plexus is populated
**Given** a deployment with Plexus active and source material previously ingested and enriched
**When** the Orchestrator Agent emits `query_knowledge` for a topic covered by ingested sources
**Then** the tool returns content derived from enriched signals over the source material — not from LLM-generated summaries (AS-4)

### Scenario: record_outcome writes asynchronously without blocking the ReAct loop
**Given** an active Session with Plexus active
**When** the Orchestrator Agent emits a `record_outcome` tool call
**Then** the tool returns acknowledgement promptly, the next ReAct iteration is not blocked waiting for ingestion or enrichment, and the recorded outcome becomes queryable after enrichment completes (eventual consistency)

### Scenario: Ingestion accepts source material, not LLM summaries
**Given** a Plexus-active deployment
**When** the llm-orc bootstrapping pipeline pushes an ensemble YAML file and an execution artifact to Plexus
**Then** the ingestion boundary accepts the file content as source material, no LLM-generated summary is ingested as a substitute for the source, and enrichment runs asynchronously as the quality gate

### Scenario: Orchestrator's ReAct loop remains responsive while enrichment lags
**Given** a Plexus-active deployment with a backlog of enrichment tasks
**When** the Orchestrator Agent issues a `query_knowledge` call for a recently recorded outcome whose enrichment has not yet completed
**Then** the tool returns the currently-visible state of the graph (which may not include the un-enriched recent outcome), the orchestrator's reasoning proceeds with the available information, and the outcome becomes visible after enrichment completes

---

## Feature: Cost and Quality Experimentation (from Product Discovery Inversions)

### Scenario: Same task runs with and without Plexus context across Model Profiles (testable OQ #1)
**Given** the same task input, two Orchestrator Model Profiles (one frontier-class, one smaller), and two deployment modes (Plexus active with populated graph, Plexus absent)
**When** the task is executed in all four combinations and routing decisions, quality signals, and total cost are recorded
**Then** the results are comparable across combinations — producing data on whether the smaller profile with a populated graph matches the frontier profile without it (the knowledge-compensated model selection hypothesis is testable, not necessarily validated by this scenario)

### Scenario: Bootstrapped graph shortens time-to-first-useful-query (testable OQ #4)
**Given** two fresh deployments with identical configuration: one with bootstrapping run against the existing llm-orc library and one without
**When** each runs an identical first session containing a `query_knowledge` call
**Then** the bootstrapped deployment returns non-empty results where the non-bootstrapped deployment returns empty — demonstrating the bootstrapping pipeline delivers initial value (quality of the returned results is a separate measurement)

---

## Feature: Structural Debt Remediation (from conformance-scan-decide-001)

### Scenario (refactor): cross-ensemble cycle validation is callable without loading
**Given** the existing `EnsembleLoader._validate_cross_ensemble_cycles` + `_build_reference_graph` logic is currently private and embedded in the load path
**When** that logic is extracted into a public function in `ensemble_config.py` and `ValidationHandler._collect_validation_errors` is wired to call it with real search directories from the config manager
**Then** a cyclic pair of ensembles submitted to `validate_ensemble` through the MCP and web API paths returns a validation error, not a success — closing the High-severity item

### Scenario (refactor): list_ensembles passes search_dirs to the loader
**Given** the current `EnsembleLoader.list_ensembles` call chain silently skips cross-ensemble cycle validation because `load_from_file` is invoked with no `search_dirs` argument
**When** `list_ensembles` is updated to pass the listing directory as `search_dirs` (or defers to the extracted public validator above)
**Then** `find_ensemble` and downstream validators exercise the same cycle check as the execution path — closing the Medium-severity item

### Scenario (regression): composition-time validator and load-time validator share a single routine
**Given** both the future `compose_ensemble` tool and the existing load path
**When** each validates the same proposed ensemble configuration containing a cross-ensemble cycle
**Then** both return the same validation error identifying the cycle, their behavior cannot diverge because both call the extracted public routine, and the split-implementation risk flagged in ADR-006's Negative consequence is mechanically avoided

# Behavior Scenarios: Agentic Serving

*Derived from domain model, ADRs 001-017, deferred candidate #5, conformance scan conformance-scan-decide-001, and conformance-scan-cycle-4-decide. Every term used here comes from the scoped domain model or the project-level domain model.*

---

## Cycle Acceptance Criteria Table (Cycle 4 additions)

The following acceptance criteria from Cycle 4's product-discovery and DECIDE-phase carry-forwards are emergent, aggregate, or integration-layer-specific. BUILD Step 5.5 (when BUILD is run on these ADRs in a future cycle) verifies each entry at its specified layer.

| Criterion | Specified layer | Verification method | Layer-match check |
|-----------|----------------|--------------------|-----|
| Cross-layer calibration channel produces measurable bias-bound under multi-iteration session work | Integration (multi-ensemble multi-iteration runtime) | Composes scenarios "Calibration data flows L0 → L1 through read-only channel" + "Time-decay windowing limits trajectory features to recent window" + "Out-of-band audit detects parameter drift" + first-deployment evidence on North-Star benchmark | no — individual scenarios exercise mechanisms in isolation; integration test or first-deployment evidence verifies the composition |
| Structured-handoff artifact set composes the three non-regression-plus-continuity properties | Integration (multi-session continuation with cluster-2 declared) | Composes scenarios "Cluster 2 session activates structured-handoff artifact set" + "Monotonic passes constraint enforced at schema level" + "Append-only progress log rejects non-append writes" + "init.sh hash mismatch produces typed error" | no — individual scenarios exercise components in isolation; integration test verifies the artifact set's composed properties across a session boundary |
| Capability-floor runtime probe surfaces operator-readable mismatch when local-tier falls below baseline | Live install/startup against operator hardware | Composes scenario "Baseline-competence calibration ensemble reports operator-readable mismatch" with operator-deployed runtime against deployment-realistic local-model availability | no — synthetic test cannot exercise operator hardware; runtime probe is the verification layer; live install/startup is the only fully-conformant verification surface |
| Calibration verdict trichotomy produces correct router action across all three verdict classes | Integration (calibration verdict + router composition) | Composes scenarios "Proceed verdict — dispatch as-is" + "Reflect verdict routes to escalated-tier" + "Abstain verdict produces escalation-bypass typed error" with verified data flow between ADR-014 and ADR-015 | no — individual scenarios exercise verdicts in isolation; integration test verifies router consumes verdicts correctly |
| Conversation Compaction five-layer pipeline maintains orchestrator coherence across long sessions | Integration (multi-turn session with context approaching threshold) | Composes scenarios "Persist-large-tool-results triggers above 50K-character output" + "LLM summary runs only when Layers 0-3 cannot reduce context" + "Layer 4 circuit-breaker after three consecutive failures" with multi-turn session fixture | no — individual scenarios exercise pipeline layers in isolation; integration test or first-deployment evidence verifies the cheapest-first ordering's coherence property |

The Layer-match "no" entries are not failures — they are the table working as designed, surfacing where BUILD Step 5.5 (when BUILD runs) closes integration verification gaps with dedicated tests or harness work, and where first-deployment evidence is the natural verification surface.

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

---

## Feature: Conversation Compaction Five-Layer Pipeline (ADR-012)

### Scenario: Tool result over 50K characters persists to disk
**Given** an Orchestrator Agent receives a tool-call result whose payload exceeds 50,000 characters
**When** the Conversation Compaction module's Layer 0 runs at the next turn boundary
**Then** the full payload persists to disk at an operator-configurable path, the orchestrator's context receives a 2,048-byte preview plus the persistent path, and the path is queryable through the existing query channels later

### Scenario: Cache-edit removes old entries without invalidating prefix
**Given** a Session whose conversation prefix has cache entries from prior turns
**When** Layer 1 cache-edit runs
**Then** old cache entries are deleted and the conversation prefix's cached state continues amortizing over subsequent turns — no cache invalidation propagates to the prefix

### Scenario: Idle-expiry clears tool results inactive for 60 minutes
**Given** a long-running Session containing tool results last touched more than 60 minutes ago alongside recent tool results
**When** Layer 2 idle-expiry runs at a turn boundary
**Then** idle-expired tool results are cleared from active context, recent tool results are preserved, and the cleared results remain reclaimable by their persistent path

### Scenario: Session notes template updates each turn at zero LLM cost
**Given** an active Session with the nine-section session-notes template at any state below the 12,288-token cap
**When** Layer 3 fires at a turn boundary
**Then** the template's nine sections are updated by deterministic logic with no LLM call, and the template's token budget remains capped at 12,288

### Scenario: LLM semantic summary runs only when Layers 0-3 cannot reduce context below threshold
**Given** a Session whose context budget exceeds the configured threshold and Layers 0-3 have already reduced what they can
**When** the Conversation Compaction pipeline reaches Layer 4
**Then** the LLM semantic summarizer runs and produces a summary that brings context below the threshold

### Scenario: Layer 4 circuit-breaker after three consecutive failures
**Given** a Session in which Layer 4 has produced LLM-summary failures on three consecutive invocations
**When** the Conversation Compaction pipeline would invoke Layer 4 a fourth time
**Then** the circuit-breaker suspends Layer 4 for the remainder of the session, a typed error is logged with operator-readable diagnostics, and the orchestrator is notified that semantic summary is unavailable

### Scenario: Layer 4 circuit-breaker state resets at session start
**Given** a previous Session that left Layer 4 in suspended state due to three consecutive failures
**When** a new Session begins
**Then** Layer 4 circuit-breaker state is automatically reset to active without operator intervention

### Preservation: AS-7 result-summarization path is unchanged
**Given** an Orchestrator Agent calling `invoke_ensemble` whose ensemble result triggers AS-7 summarization
**When** Conversation Compaction is active in the same Session
**Then** the AS-7 summarization (Result Summarizer Harness per ADR-004) continues to run on ensemble outputs as before, independently of Conversation Compaction operating on the conversation history

---

## Feature: Session Registry Initializer-then-Resume (ADR-013)

### Scenario: Cluster 2 session activates structured-handoff artifact set
**Given** an operator declares a session as Cluster 2 (BUILD/ARCHITECT/DEBUG/REFACTOR territory) at session start
**When** the Session Registry processes the declaration
**Then** the structured-handoff artifact set (feature_list.json, append-only progress log, init.sh) is required and write-gate validation activates for all artifact writes

### Scenario: Cluster 1 session opts out of artifact set
**Given** an operator declares a session as Cluster 1 (RESEARCH/DECIDE/SYNTHESIZE territory) at session start
**When** the Session Registry processes the declaration
**Then** the structured-handoff artifact set is supported but not required; operator can opt-in with explicit configuration

### Scenario: Monotonic passes constraint enforced at schema level
**Given** a feature_list.json entry with `passes: true` for feature `auth-flow`
**When** the orchestrator submits a write that would set `auth-flow` to `passes: false` without an audit-logged operator override
**Then** the write-gate rejects the write with a typed `write_gate_rejection` error and the feature_list.json on disk is unchanged

### Scenario: Append-only progress log rejects non-append writes
**Given** an active Session with an append-only progress log at any state
**When** the orchestrator submits a write that attempts to overwrite, truncate, or mid-file edit the log
**Then** the write-gate rejects the operation with a typed `write_gate_rejection` error and the progress log on disk is unchanged

### Scenario: init.sh hash mismatch produces typed error
**Given** a Session Registry configuration with init.sh integrity hash `H1` recorded at operator-authoring time
**When** the Session Registry would invoke init.sh whose actual content hashes to `H2 ≠ H1`
**Then** init.sh execution is gated, a typed `write_gate_rejection` error fires naming the hash mismatch, and the Session does not proceed past initialization until the operator resolves

### Scenario: Operator hash rotation re-authors integrity record
**Given** an operator legitimately modifies init.sh content (e.g., adds a new dependency to PATH setup)
**When** the operator runs the hash-rotation workflow recording the new hash in the Session Registry's configuration
**Then** subsequent Sessions execute the modified init.sh successfully, and the rotation event is audit-logged

### Scenario: Cross-cluster session defaults to required artifact set
**Given** a session whose declaration is ambiguous or names multiple clusters (e.g., a North-Star-benchmark-style session straddling RESEARCH and BUILD)
**When** the Session Registry processes the declaration
**Then** disposition (i) — default to required-artifact-set behavior — applies, and the artifact set is active for the session

### Preservation: Existing Session identification responsibility unchanged
**Given** a multi-request Session under the existing Session Registry implementation
**When** ADR-013's structured-handoff artifact extension is active
**Then** the existing SessionIdentity derivation, SessionState tracking, turn_count, and token_spend bookkeeping continue to operate exactly as before — the artifact extension is additive

---

## Feature: Calibration Verdict Trichotomy (ADR-014)

### Scenario: Proceed verdict routes dispatch as-is
**Given** a calibration verdict computation where AUQ confidence is above the System 2 threshold (default 0.85), trajectory features are in normal range, and post-hoc result-check is positive
**When** the Calibration Gate produces its verdict for the dispatch
**Then** the verdict is *Proceed* and dispatch routes to the per-skill cheap-tier Model Profile per ADR-015 without any reflection or escalation

### Scenario: Reflect verdict routes to escalated tier per ADR-015
**Given** a calibration verdict where AUQ confidence is below the System 2 threshold but trajectory features show no anomaly
**When** the Calibration Gate produces its verdict for the dispatch
**Then** the verdict is *Reflect* and the per-role tier-escalation router routes the dispatch to the per-skill escalated-tier Model Profile

### Scenario: Abstain verdict blocks dispatch and produces typed error
**Given** a calibration trajectory where token-level entropy in the most recent N tokens drops more than 1.5 standard deviations below the trajectory's running mean
**When** the Calibration Gate evaluates the verdict for the dispatch
**Then** the verdict is *Abstain*, dispatch is blocked, the orchestrator receives a typed `calibration_abstain` error, and the orchestrator must take a different action (reformulate, dispatch elsewhere, or abstain entirely)

### Scenario: Time-decay windowing limits trajectory features to dual-bound recent window
**Given** a Calibration Gate computing the next verdict with trajectory data spanning two hours and 250 prior dispatches
**When** the verdict computation aggregates trajectory features
**Then** only signals within the most recent 60 minutes OR most recent 100 dispatches (whichever is shorter) contribute to the verdict, weighted linearly from 1.0 at signal-emission to 0.0 at window-edge

### Scenario: Calibration verdict feeds router input
**Given** a calibration verdict produced by the Calibration Gate for a specific dispatch
**When** the per-role tier-escalation router (ADR-015) processes the dispatch through the Tool Dispatch interposition
**Then** the router consumes the verdict directly as input — no LLM-mediated translation step — and the verdict's three values (Proceed/Reflect/Abstain) map deterministically to router actions (cheap-tier, escalated-tier, escalation-bypass)

### Preservation: ADR-007 first-N post-hoc calibration mechanism unchanged
**Given** a composed ensemble in its first-N invocation calibration window per ADR-007
**When** ADR-014's in-process trajectory-level calibration is active
**Then** the existing post-hoc result-check from ADR-007 continues to fire on every first-N invocation, the existing quality-signal accumulation and trusted-status transition logic continue unchanged, and ADR-014's in-process layer composes additively without replacing ADR-007's mechanism

---

## Feature: Per-Role Tier-Escalation Router (ADR-015)

### Scenario: Code-generation skill routes to per-skill cheap-tier on Proceed
**Given** an ensemble whose YAML metadata declares Topaz skill `code_generation` and operator-configured tier defaults specify `cheap-tier: ollama-deepseek-coder-v2:16b` for that skill
**When** the orchestrator dispatches the ensemble via `invoke_ensemble` with calibration verdict *Proceed*
**Then** the Tool Dispatch router selects `ollama-deepseek-coder-v2:16b` as the dispatch's Model Profile

### Scenario: Reflect verdict routes to per-skill escalated-tier
**Given** an ensemble with Topaz skill `tool_use` and operator-configured tier defaults specifying `escalated-tier: gpt-5-mini` for that skill
**When** the calibration verdict for the dispatch is *Reflect*
**Then** the router selects `gpt-5-mini` as the dispatch's Model Profile

### Scenario: Abstain verdict produces escalation-bypass typed error
**Given** any dispatch with calibration verdict *Abstain*
**When** the per-role tier-escalation router processes the dispatch
**Then** the router does not perform tier escalation, instead produces a typed `escalation_bypass` error to the orchestrator, and the orchestrator must reformulate or take a different action

### Scenario: Ensemble lacking Topaz skill metadata fails dispatch with explanatory error
**Given** an ensemble in the library whose YAML configuration does not declare a Topaz skill metadata field
**When** the orchestrator dispatches the ensemble
**Then** the Tool Dispatch router rejects the dispatch with a typed `missing_skill_metadata` error explaining that all ensembles must declare their primary Topaz skill, and the rejection includes a list of valid skill values

### Scenario: Per-skill (not per-ensemble) tier defaults
**Given** two ensembles `code-review-pair-A` and `code-review-pair-B` both declaring Topaz skill `code_generation`
**When** the operator's tier-default configuration specifies cheap-tier and escalated-tier Model Profiles for `code_generation`
**Then** both ensembles use the same cheap-tier and escalated-tier Model Profiles when dispatched — the configuration is per-skill, not per-ensemble

### Preservation: ADR-011 orchestrator's own LLM session-boundary scope unchanged
**Given** an active Session with the orchestrator parameterized by Model Profile `profile-A` and ADR-015's router active
**When** the orchestrator dispatches multiple ensembles whose calibration verdicts span Proceed and Reflect (cheap-tier and escalated-tier dispatches in the same session)
**Then** the orchestrator's own Model Profile remains `profile-A` for the entire Session — only the dispatched task's tier varies; the orchestrator's session-boundary-event constraint per ADR-011 is preserved

---

## Feature: Cross-Layer Calibration Channel (ADR-016)

### Scenario: Calibration data flows L0 → L1 through read-only channel
**Given** an Ensemble Engine (L0) emitting calibration signals (trajectory features, dispatch outcomes, deterministic-tool-output anchors when applicable) and a Calibration Gate (L1) consuming those signals
**When** an L0 ensemble dispatch completes
**Then** the calibration signals propagate upward through the read-only channel, the L1 consumer receives typed signal data, and the L1 verdict computation incorporates the signals subject to the five bounding mechanisms

### Scenario: Upward write attempt through channel is rejected
**Given** an L1 consumer attempting to write data back through the calibration channel toward L0
**When** the write reaches the channel boundary
**Then** the write is rejected at the structural level, no L0 state is mutated, and ADR-002's layering-rule write-path discipline is preserved

### Scenario: Non-calibration data attempt through channel is rejected
**Given** an L0 module attempting to send non-calibration data (e.g., an arbitrary upward import) through the calibration channel
**When** the data reaches the channel boundary
**Then** the channel rejects the data via the structural validation guard, the L1 consumer never receives it, and ADR-002's layering-rule signal-channel scoping is preserved

### Scenario: Mechanism (a) — fresh-context isolation in the calibration consumer
**Given** an L1 calibration consumer computing a new calibration verdict
**When** the consumer reads channel signals
**Then** the consumer's evaluation context is fresh — prior signals from earlier verdicts are not carried forward through the consumer's context; their influence on the next verdict is only through the time-decay-windowed feature aggregation specified by mechanism (b)

### Scenario: Mechanism (b) — time-decay windowing eliminates stale-signal influence
**Given** a Calibration Gate computing a verdict at time T with calibration signals spanning T-180 minutes through T-0
**When** the verdict computation aggregates feature values
**Then** signals older than the dual-bound (60 minutes / 100 signals, whichever shorter) contribute weight 0 to the verdict, and signals within the window contribute weight from 1.0 (most recent) linearly decaying to 0.0 (window edge)

### Scenario: Mechanism (c) — deterministic-tool-output anchor when ensemble has script-model slot
**Given** an ensemble with a script-model slot producing binary-verifiable deterministic output
**When** the deterministic output enters the calibration signal stream
**Then** mechanism (c) treats the output as a categorical anchor that LLM-consensus signals cannot override probabilistically, and the L1 verdict reflects the categorical signal at higher weight

### Scenario: Mechanism (c) — LLM-only ensemble operates with mechanisms (a), (b), (d), (e) only
**Given** an ensemble with no script-model slots (LLM-only configuration)
**When** the Calibration Gate computes verdicts for dispatches of this ensemble
**Then** mechanism (c) is unavailable, mechanisms (a), (b), (d), (e) remain load-bearing, and the calibration signal flow continues without categorical anchor — consistent with ADR-002's AS-8 optionality parallel

### Scenario: Mechanism (d) — out-of-band audit dispatch fires at trigger frequency
**Given** a calibration system with the audit trigger frequency configured (default every 100 verdicts or 24 wall-clock hours, whichever first)
**When** the trigger condition is reached
**Then** the audit dispatches in a fresh context, evaluates verdict skew + outcome divergence + signal-to-verdict correlation drift against operationally-tunable thresholds, and produces a typed verdict (no drift / drift detected / severe drift)

### Scenario: Mechanism (e) — malformed signal produces typed error
**Given** an L0 module emitting a calibration signal whose schema does not match the typed signal-data specification
**When** the signal reaches the L1 channel boundary
**Then** the structural validation rejects the signal with a typed `malformed_signal` error, the verdict computation skips the malformed signal as if outside the window, and no verdict-side action is taken on the signal's content

### Scenario: Severe drift verdict triggers fail-safe mode
**Given** a periodic out-of-band audit producing verdict *severe drift* (multiple drift criteria simultaneously exceed thresholds OR any single criterion exceeds severe-threshold)
**When** the audit verdict reaches the calibration consumer
**Then** the calibration system enters fail-safe mode — calibration verdicts default to Reflect-or-Abstain — operator-notification fires, and the system remains in fail-safe until operator review releases it

### Scenario: ADR-016 not active — ADR-014 operates on L1-internal data only
**Given** an llm-orc deployment in which ADR-016 is rejected (no cross-layer signal channel)
**When** the Calibration Gate computes verdicts for dispatches
**Then** ADR-014's in-process trajectory-level calibration operates on L1-internal trajectory data only, the verdict trichotomy continues to function with the in-layer feature set, and no cross-layer signals enter the verdict computation

### Preservation: ADR-002 layering rule unchanged for write paths and non-calibration upward signaling
**Given** an llm-orc deployment with ADR-016's calibration channel active
**When** any non-calibration upward signal flow or any upward write path is exercised
**Then** the layering rule continues to prohibit it (FC-2 static import check rejects the import; FC-3 cycle detection rejects the dependency edge), and only the read-only calibration channel from L0 to L1 is permitted

---

## Feature: Tool-Call Structural Validation Guard (ADR-017)

### Scenario: Match — prose claim plus tool-call structure passes validation
**Given** an orchestrator response containing prose "I called `invoke_ensemble('code-review-A')` and the tool returned ..." alongside an actual tool-call structure naming `invoke_ensemble` with argument `{'ensemble_name': 'code-review-A'}`
**When** the structural validation guard scans the response
**Then** the guard verifies the structural correspondence and the dispatch proceeds without rejection

### Scenario: Mismatch — prose claim without tool-call structure produces phantom_tool_call error
**Given** an orchestrator response containing prose "The tool call has been made and the result is displayed above" with zero tool-call structures actually emitted
**When** the structural validation guard scans the response
**Then** the guard produces a typed `phantom_tool_call` error including the detected prose claim, the empty list of actually-emitted tool-call structures, and the dispatch context, and the orchestrator must take a different action

### Scenario: Future-intent patterns are not flagged
**Given** an orchestrator response containing prose "I will call `invoke_ensemble` next" without yet emitting a tool-call structure
**When** the structural validation guard scans the response
**Then** no error is produced — future-intent patterns describe upcoming action and are not assertion patterns; the guard's conservative discipline excludes them

### Scenario: Pattern set is operator-extensible at deployment configuration
**Given** an operator deploying llm-orc who has observed a deployment-specific phantom-tool-call pattern not in the default pattern set
**When** the operator adds the pattern to the deployment configuration
**Then** the structural validation guard scans incoming orchestrator responses with the extended pattern set, and the new pattern triggers `phantom_tool_call` errors when matched without a corresponding tool-call structure

---

## Feature: Capability Floor (Cycle 4 discover-gate carry-forward #7)

### Scenario: Default `orchestrator-local` Model Profile fails baseline competence on install
**Given** an operator who has installed llm-orc without configuring a custom orchestrator Model Profile, and whose locally-available models do not meet the capability floor for the closed five-tool surface
**When** the operator runs the baseline-competence calibration ensemble at install
**Then** the ensemble produces an operator-readable diagnostic naming the specific failure (e.g., the local model fails to produce structurally valid tool-call outputs) and recommends concrete remediation paths (configure cheap-cloud orchestrator profile; install a more-capable local model; etc.)

### Scenario: Baseline-competence calibration ensemble reports operator-readable mismatch
**Given** an operator's locally-available models including `ollama-llama3:8b` and `ollama-mistral:7b`
**When** the baseline-competence ensemble probes each model against the capability-floor baseline
**Then** the ensemble produces a structured report identifying which models meet the floor (and on which capability dimensions), which fall below the floor, and what concrete operator action is recommended for the deployment

### Scenario: Operator runs runtime probe at install/startup
**Given** an operator running `llm-orc serve` for the first time on a hardware configuration whose local-model availability is unknown to the system
**When** the operator invokes the runtime probe (either explicitly via `llm-orc orchestrator probe` or automatically at first-run startup if configured to do so)
**Then** the probe runs the baseline-competence calibration ensemble against discovered local models and produces an actionable recommendation surface — operator can act on the recommendations before the first agentic-coding session

### Scenario: Static specification + runtime probe compose
**Given** the static capability-floor specification documented in `domain-model.md` and `system-design.md`
**When** the runtime probe runs against operator hardware
**Then** the static specification provides the abstract floor reference; the runtime probe produces the concrete-against-operator-hardware mismatch detection; both surfaces co-exist as compatible design surfaces (per discover-gate carry-forward #7)

---

## Cycle Acceptance Criteria Table (Cycle 5 additions)

| Criterion | Specified layer | Verification method | Layer-match check |
|-----------|----------------|--------------------|-----|
| The five minimum-viable capability ensembles (`code-generator`, `claim-extractor`, `argument-mapper`, `prose-improver`, `text-summarizer`) compose with the per-skill tier defaults to produce a dispatch across all five non-degenerate Topaz slots | Integration (orchestrator dispatch through all five ensembles with the Cycle 5 agentic-serving config) | Composes scenarios "Capability ensemble dispatches via explicit ensemble naming" + "Per-skill tier defaults select correct tier per dispatch" for each of five capability ensembles; plus integration scenario "Skill framework decomposes RDD research workflow through five capability dispatches" | no — individual ensemble scenarios exercise dispatch in isolation; integration test verifies the library's compositional shape under a realistic skill-framework workflow |
| Skill-framework-agnostic dispatch contract works for at least two distinct skill orchestration users without orchestrator-side methodology knowledge | Integration (two skill framework clients composing against the same library) | Composes scenarios "RDD `rdd:research` decomposes to capability dispatches" + a non-RDD-framework analog (e.g., a manual ad-hoc skill workflow via OpenCode prompts) consuming the same five-ensemble library | no — synthetic test cannot exercise multi-skill-framework deployment; verification requires either RDD + a second skill framework (Anthropic Skills, code-review-as-methodology, etc.) or live-deployment evidence. The criterion captures the architectural commitment's empirical test surface |
| `web-searcher` script-agent dispatches return structured results within the calibration-gate-observable shape (Calibration Gate fires on result structure via post-hoc result-check) | Integration (web-searcher dispatched through Calibration Gate; Tavily backend live) | Composes scenarios "Web-searcher dispatch returns structured JSON" + "Calibration Gate post-hoc result-check verifies result schema" + "MissingSkillMetadataError recovery is bypass-able when tool_use ensemble is available" | yes — composed scenario exercises the dispatch path through Calibration Gate with the script-agent shape |
| Working-defaults BUILD-scope deliverables (`agentic-serving-profiles.yaml` + `agentic-serving/` subdirectory + minimum-viable ensemble set + rewritten `agentic_serving:` config section + `agentic-serving/README.md`) compose into a runnable deployment on first encounter, without operator manual authoring beyond environment-variable setup | Live install/startup against operator hardware | Composes the BUILD-phase mechanical deliverables with a fresh-clone run-through scenario — `git clone` → environment-variable setup → `llm-orc serve` → OpenCode dispatch → response returned. The criterion is satisfied when an operator can complete this sequence without authoring any YAML or config files. | no — synthetic test cannot exercise fresh-clone first-encounter; live install/startup is the verification layer |

The Layer-match "no" entries are not failures — they identify where BUILD Step 5.5 closes integration verification gaps with dedicated tests or harness work, and where live-deployment evidence is the natural verification surface (the same pattern Cycle 4's table established).

---

## Feature: Skill-Framework-Agnostic Capability Library (ADR-019)

### Scenario: Capability ensemble carries Topaz skill metadata and is dispatchable by name
**Given** a capability ensemble at `.llm-orc/ensembles/agentic-serving/claim-extractor.yaml` with `topaz_skill: factual_knowledge` declared in its YAML frontmatter
**When** the orchestrator invokes `invoke_ensemble("claim-extractor", {...})`
**Then** the Tier-Escalation Router (per ADR-015) reads `topaz_skill: factual_knowledge` from the ensemble's metadata, consults the `factual_knowledge` per-skill tier defaults in `agentic_serving.orchestrator.per_skill_tier_defaults`, and dispatches the ensemble at the calibration-verdict-selected tier

### Scenario: Operation-named ensembles live in the agentic-serving subdirectory
**Given** the Cycle 5 BUILD deliverable shape — `.llm-orc/ensembles/agentic-serving/` subdirectory containing operation-named capability ensembles (`code-generator`, `claim-extractor`, `argument-mapper`, `prose-improver`, `text-summarizer`) and the moved system ensembles (`agentic-result-summarizer`, `agentic-calibration-checker`)
**When** the orchestrator's ensemble walk path discovers `.llm-orc/ensembles/agentic-serving/` during library traversal
**Then** all ensembles in the subdirectory are loaded into the library and available for dispatch; subdirectory namespace is structural, not load-bearing for routing logic

### Scenario: Mathematical reasoning slot is configured but unauthored
**Given** the Cycle 5 `agentic_serving.orchestrator.per_skill_tier_defaults` configuration declaring tier profiles for all eight Topaz skills including `mathematical_reasoning`, but no capability ensemble in the library carrying `topaz_skill: mathematical_reasoning`
**When** the orchestrator's Tier-Escalation Router attempts to route a sub-task to `mathematical_reasoning`
**Then** the router raises `MissingSkillMetadataError` per ADR-015; the typed error reaches the orchestrator's reasoning surface with `recovery_action_required="reformulate"`; the orchestrator reformulates and attempts a different dispatch path

### Scenario: Operator-facing README documents the structure and extension pattern
**Given** the Cycle 5 BUILD deliverable shape including `.llm-orc/ensembles/agentic-serving/README.md`
**When** an operator opens the README at the deployment's `agentic-serving/` subdirectory
**Then** the README explains the operation-named principle, names the existing five capability ensembles, distinguishes capability ensembles from system ensembles (`agentic-result-summarizer`, `agentic-calibration-checker`), and describes how operators author additional capability ensembles under the same shape principle

### Preservation: ADR-015 per-skill tier defaults shape is unchanged
**Given** the Cycle 4 BUILD-shipped `per_skill_tier_defaults` configuration with eight Topaz slot entries, each containing `cheap_tier` and `escalated_tier` profile references
**When** the Cycle 5 `agentic_serving:` config section is rewritten to reference profiles from `.llm-orc/profiles/agentic-serving-profiles.yaml` rather than inline profiles
**Then** the `per_skill_tier_defaults` shape is unchanged — same eight slots, same cheap/escalated pair structure, same construction-time validation rules; only profile *references* change (from inline names to file-resolved names)

### Preservation: Cycle 4 PLAY tagging work continues to dispatch correctly
**Given** the Cycle 4 PLAY-config-tagged ensembles — `agentic-coding-helper` (code_generation; uncommitted PLAY artifact in `.llm-orc/ensembles/agentic-coding-helper.yaml`) and `development/code-review` (instruction_following; pre-existing top-level ensemble)
**When** Cycle 5 BUILD authors the new capability ensembles and rewrites the `agentic_serving:` config
**Then** the tagging work is preserved by ADR-019's library reshape:
  - `agentic-coding-helper` is **promoted** (per ADR-019 §"Working defaults") to `code-generator` in `.llm-orc/ensembles/agentic-serving/code-generator.yaml`, retaining the same `topaz_skill: code_generation` tag and the same three-agent flow shape; the code_generation dispatch path is preserved through the rename
  - `development/code-review` continues to dispatch by its existing name with its existing `topaz_skill: instruction_following` tag (the `instruction_following` slot continues to be served by this deployment-specific ensemble per the README's coverage table)
  - ADR-019's library reshape does not break either Cycle 4 PLAY tagging commitment; the operation-named principle is honored by the promotion (verb-noun: `code-generator`) rather than the methodology-coded name (`agentic-coding-helper`)

---

## Feature: Tool-Use Ensemble — Web-Searcher (ADR-020)

### Scenario: Web-searcher dispatches via the script-agent path
**Given** a `web-searcher` ensemble in `.llm-orc/ensembles/agentic-serving/web-searcher.yaml` with `topaz_skill: tool_use`, a script-model-slot agent wrapping the Tavily search API
**When** the orchestrator invokes `invoke_ensemble("web-searcher", {"query": "claim verification sources for X"})` and the script-agent executes with `WEB_SEARCH_BACKEND=tavily` and a valid `WEB_SEARCH_API_KEY`
**Then** the script-agent calls the Tavily API, receives structured JSON, and returns top-N URLs and snippets as agent output; the dispatch completes through the existing Tool Dispatch path

### Scenario: Backend selection via environment variable
**Given** an operator who has set `WEB_SEARCH_BACKEND=brave` and `WEB_SEARCH_API_KEY=<brave-key>` in the environment running `llm-orc serve`, and a `web-searcher` script-agent with the Brave adapter authored by the operator
**When** the orchestrator dispatches `web-searcher`
**Then** the script-agent selects the Brave adapter at startup based on the environment variable; the dispatch returns Brave-search-API results with the same structured shape Tavily uses

### Scenario: Authentication failure surfaces as structured error
**Given** a `web-searcher` script-agent whose backend API key is invalid (revoked, mistyped, or absent from the environment)
**When** the orchestrator invokes `web-searcher`
**Then** the script-agent surfaces a structured error output the orchestrator's reasoning surface can act on (e.g., `{"error": "authentication_failed", "backend": "tavily"}`); the orchestrator does not crash the session; the post-hoc result-check (ADR-007) sees the error shape and produces a calibration signal appropriate to the error

### Scenario: Tier escalation is no-op for script-agent tool_use ensemble
**Given** the `tool_use` slot's per-skill tier defaults configured with `cheap_tier: agentic-tier-cheap-general` and `escalated_tier: agentic-tier-escalated-general` referencing the same Model Profile (no-op tier escalation because the script-agent does not consume an LLM Model Profile for its execution)
**When** the Calibration Gate produces a Reflect verdict on a prior `web-searcher` dispatch and the Tier-Escalation Router selects the escalated tier for the next dispatch
**Then** the dispatch still executes via the script-agent; the Model Profile selection has no effect on script execution; the tier-escalation mechanism is structurally inert for script-agent ensembles (acknowledged in ADR-020 §Consequences §Negative)

### Preservation: ADR-003 closed five-tool surface is unchanged
**Given** the Cycle 4 BUILD-shipped Orchestrator Tool Dispatch's closed five-tool surface (`invoke_ensemble`, `compose_ensemble`, `list_ensembles`, `query_knowledge`, `record_outcome`)
**When** Cycle 5 BUILD adds the `web-searcher` capability ensemble to the library
**Then** the orchestrator's internal action space remains exactly five tools; the `web-searcher` ensemble is invokable *through* `invoke_ensemble`, not as a new internal action; ADR-003's invariant is preserved by construction

### Preservation: Calibration Gate post-hoc result-check fires on web-searcher dispatches
**Given** the Cycle 4 BUILD-shipped Calibration Gate (ADR-007 + ADR-014) firing on `invoke_ensemble` return paths
**When** `web-searcher` dispatches return structured search results
**Then** the Calibration Gate's post-hoc result-check verifies the result structure (count, schema, error-flag presence); the AUQ and HTC calibration components are inactive for script-agent execution (acknowledged in ADR-020 §Consequences §Positive); the calibration signal flows through the existing Calibration Gate path unchanged

---

## Feature: Skill-Orchestration Composition via Per-Capability Dispatch (ADR-021)

### Scenario: Skill framework dispatches via explicit ensemble naming
**Given** an RDD skill plugin (`rdd:research` skill) that has consulted `skill-framework-capability-registry.md` and identified `claim-extractor` as the target capability for a sub-task
**When** the skill plugin emits an OpenAI-compatible request to the orchestrator with `invoke_ensemble("claim-extractor", {"data": "..."})` as the tool-call argument
**Then** the orchestrator dispatches `claim-extractor` directly without inferring the target ensemble from prompt content; the Tier-Escalation Router reads `claim-extractor`'s tagged Topaz skill; pre-specified routing is preserved end-to-end

### Scenario: Skill framework dispatches via natural-language prompt
**Given** a skill orchestration user who has emitted a natural-language sub-task to the orchestrator (e.g., "extract factual claims from the following text: ...") without naming the target ensemble
**When** the orchestrator's ReAct loop receives the prompt
**Then** the orchestrator selects `claim-extractor` from the library using LLM-judgment matching of the prompt's task description to the ensemble's description (retrieval over the `list_ensembles()` result, not evaluative classification); dispatches via `invoke_ensemble`; the dispatch proceeds with the Tier-Escalation Router's normal pre-specified routing

### Scenario: Fresh-context property holds across capability sub-tasks
**Given** an RDD lit-review workflow that decomposes into three capability sub-tasks: `web-searcher` (search for sources); `claim-extractor` (extract claims from search results); `argument-mapper` (map argument structure across extracted claims)
**When** the skill plugin dispatches the three sub-tasks sequentially via three `invoke_ensemble` calls in three chat-completion requests
**Then** each dispatched ensemble receives `input + system_prompt` only; no orchestrator conversation history from prior sub-tasks bleeds into the next ensemble's context; the architectural-isolation property described in ADR-019 / ADR-021 holds across the workflow (per Cycle 4 PLAY note 14's empirical observation, treated as an architectural fact of `invoke_ensemble`)

### Scenario: Skill framework owns workflow state across sub-tasks
**Given** an RDD lit-review's three-sub-task decomposition that requires the `claim-extractor` sub-task to consume the `web-searcher`'s search-result output as input
**When** the skill plugin completes the `web-searcher` dispatch and prepares the `claim-extractor` dispatch
**Then** the skill plugin extracts the search results from the `web-searcher` response, formats them into the `claim-extractor`'s input prompt, and emits the next dispatch with the workflow-state forward-passing handled client-side; the orchestrator does not maintain workflow state across the two dispatches

### Scenario: Non-Topaz-aligned skill framework requires adapter layer
**Given** a hypothetical skill framework that decomposes its workflows into framework-specific sub-task types (`framework-source-finder`, `framework-claim-puller`) rather than Topaz-skill-named sub-tasks
**When** the framework attempts to dispatch sub-tasks to the agentic-serving orchestrator
**Then** the framework requires either (a) an internal mapping from `framework-source-finder` → `tool_use` and `framework-claim-puller` → `factual_knowledge` before emitting dispatch requests, or (b) an adapter layer between the framework's decomposition vocabulary and the Topaz routing vocabulary; the precondition is named in ADR-021 §Consequences §Negative

### Scenario: compose_ensemble rejects invalid primitive types (Cycle 4 PLAY note 13)
**Given** a request to `compose_ensemble` whose composition shape names an internal-tool name (e.g., `list_ensembles`) as if it were an agent primitive — the category error Cycle 4 PLAY note 13 recorded the orchestrator making in a recommendation
**When** the Composition Validator (per ADR-006) processes the request
**Then** the Composition Validator rejects the composition with a typed error indicating that internal-tool names are not valid agent primitives; the primitive-validation discipline (per AS-6) is enforced at composition time, not at runtime; the orchestrator's recovery path receives the typed error and reformulates

### Preservation: ADR-006 composition palette validation is unchanged
**Given** the Cycle 4 BUILD-shipped Composition Validator with AS-2 / AS-6 / Invariant 7 / Invariant 8 enforcement
**When** Cycle 5 BUILD adds the per-capability dispatch contract and the `skill-framework-capability-registry.md` artifact
**Then** the Composition Validator's existing validation logic — primitive-existence check, depth check, cycle detection, no-partial-state-on-validation-failure — operates unchanged; ADR-021's per-capability dispatch contract is consumer-side (skill-framework client-side), not Composition-Validator-side; ADR-006's invariants are preserved

### Preservation: invoke_ensemble's fresh-context dispatch property is unchanged
**Given** the Cycle 4 BUILD-shipped `invoke_ensemble` dispatch shape — each invocation provides the dispatched ensemble's agents with `input + system_prompt` only
**When** skill frameworks compose against the orchestrator via per-capability dispatch (ADR-021)
**Then** `invoke_ensemble`'s context-isolation property holds for every dispatch; ADR-021's per-capability dispatch contract is a *consumer-side* layering of `invoke_ensemble`'s existing property, not a change to the property itself

## Feature: BUILD Runtime-Dispatch Verification (Cycle 5 PLAY carry-forward)

### Scenario: BUILD close requires end-to-end dispatch of each shipped capability ensemble
**Given** a cycle BUILD phase that ships one or more capability ensembles in `.llm-orc/ensembles/agentic-serving/` or extends existing ones
**When** the BUILD phase declares close-readiness
**Then** each shipped or extended capability ensemble has been dispatched end-to-end at least once via `llm-orc invoke <ensemble> <input>` (or `mcp__llm-orc__invoke`), the resulting `execution.json` carries `status: "completed"` and a non-null primary agent response, and the verification is recorded in the BUILD gate reflection note or cycle-status BUILD row. Discovery-layer checks (`list-ensembles`, `validate_ensemble`, `check_ensemble_runnable`) alone do not satisfy this commitment.

**Rationale (Cycle 5 PLAY note 1 — sharpened by susceptibility snapshot):** Cycle 5 BUILD declared close after verifying discovery (`llm-orc list-ensembles` discovered all 8 ensembles), schema (`validate_ensemble` returned valid), and provider runnability (`check_ensemble_runnable` returned true for each). At Cycle 5 PLAY, four of six capability ensembles errored at runtime dispatch with `unsupported operand type(s) for +: 'NoneType' and 'str'` — `agents_count: 0`; agent never started. The defect was caught only because the practitioner exercised dispatch directly during PLAY. This scenario adds runtime-dispatch as a third verification layer alongside discovery and schema validation. Auto-mode BUILD per ADR-091 is no exception — close-out should include explicit dispatch-verification status for each shipped/extended ensemble.

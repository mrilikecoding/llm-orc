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


---

## Cycle Acceptance Criteria Table (Cycle 6 additions)

The following acceptance criteria from Cycle 6 product-discovery (T14, T15, T16, T17) and the DECIDE-phase spike findings are emergent, aggregate, or integration-layer-specific. BUILD Step 5.5 verifies each entry at its specified layer.

| Criterion | Specified layer | Verification method | Layer-match check |
|-----------|----------------|--------------------|-----|
| Capability-matched NL framing routes to `invoke_ensemble` under at least one orchestrator profile after ADR-022 system-prompt amendment | Live serve dispatch (integration with orchestrator-LLM reasoning surface) | Composes scenarios "NL request matching capability ensemble dispatches via `invoke_ensemble`" + "Client-tool verb-match does not displace capability-match" with live orchestrator dispatch under MiniMax M2.5-free | no — unit tests verify the prompt amendment is present; only live dispatch verifies the orchestrator-LLM honors the amended commitment under capability-matched NL framing |
| Dispatch events route to both operator-terminal AND orchestrator-context destinations from one event substrate (Inversion N+2) | Integration (multi-dispatch session with both destinations active) | Composes scenarios "Operator-terminal destination emits per-event INFO lines" + "Orchestrator-context destination prepends structured observation between turns" + "`dispatch_id` correlation joins events for one dispatch" with multi-turn session fixture | no — individual destination scenarios verify routing per destination; integration test verifies one-substrate-two-destinations composition without parallel-infrastructure duplication |
| Liveness signals fire during in-flight states before completion events | Live serve dispatch (timing-sensitive integration) | Composes scenarios "Tool-call-emit log line precedes dispatch" + "Inference-wait heartbeat fires after `heartbeat_interval_seconds` of inactivity" with a long-inference dispatch fixture (>30s wall-clock) | no — only timed live dispatch fires the heartbeat path; mock-time tests verify the heartbeat-interval-counter logic |
| Capability ensemble dispatches produce typed `DispatchEnvelope` with `artifacts[0]` carrying substrate reference | Integration (envelope + substrate composition) | Composes scenarios "`invoke_ensemble` returns typed `DispatchEnvelope`" + "Capability ensemble writes deliverable to session-dir artifact path" + "`primary` carries summary line; `artifacts[0]` carries typed reference" | no — individual scenarios verify envelope shape and substrate writing in isolation; integration test verifies the composition for at least one capability ensemble end-to-end |
| AS-7 amendment honored: substrate-routed dispatches skip content summarization; inline-response dispatches retain summarization | Integration (mixed-mode session) | Composes scenarios "Substrate-routed dispatch's envelope is not passed through `agentic-result-summarizer`" + "Inline-response dispatch (per `output_substrate: inline`) is passed through `agentic-result-summarizer`" with a session exercising both modes | no — individual scenarios verify per-dispatch behavior; integration test verifies the AS-7 amendment's scope is honored across a mixed-mode session |
| Calibration gate evaluation operates correctly under substrate-routing | Integration (calibration gate + substrate composition) | Composes scenarios "Calibration gate evaluators receive `primary` + `artifacts[0].summary` by default" + "Calibration gate evaluators receive `envelope.structured` when `output_schema:` declared" + "Calibration gate evaluators read artifact content when `calibration_substrate_access: artifact` declared" with `code-generator` as the lead `calibration_substrate_access: artifact` case | no — individual scenarios verify per-ensemble evaluation surface; integration verifies the calibration gate's verdict accuracy under each evaluation surface |

The Layer-match "no" entries fire BUILD Step 5.5 work for integration tests or harness exercises beyond individual scenarios' coverage. Cell B's qwen3:14b over-delegation finding from spike γ is **not** an integration-test target at Cycle 6 close — disposition (iii) configuration-conditional effectiveness is named in ADR-022 as deferred to BUILD or follow-on PLAY; the cross-profile characterization is a PLAY-phase observational concern rather than a scenario-verifiable behavior.

---

## Feature: Routing Surface Behavior — System-Prompt Amendment (ADR-022)

### Scenario: NL request matching capability ensemble dispatches via `invoke_ensemble`
**Given** an operator-deployed serve with the ADR-022 amended `DEFAULT_ORCHESTRATOR_SYSTEM_PROMPT` active, and the agentic-serving library containing `code-generator` (Topaz skill `code_generation`)
**And** a skill orchestration user emits a natural-language request via the `/v1/chat/completions` endpoint: *"Write a Python function that computes the Fibonacci sequence up to N terms"* — no explicit ensemble naming
**When** the orchestrator's ReAct loop processes the request under the MiniMax M2.5-free orchestrator profile
**Then** the orchestrator calls `list_ensembles()`, identifies `code-generator` as the capability match for the NL request, and dispatches via `invoke_ensemble("code-generator", {"data": "..."})` — not via direct LLM completion and not via a client-declared tool. The `execution.json` artifact records the dispatch with `code-generator` named as the dispatched ensemble.

### Scenario: Client-tool verb-match does not displace capability-match
**Given** an operator-deployed serve with the ADR-022 amended system prompt active, the agentic-serving library containing `code-generator`, and a tool-rich client (OpenCode) exposing client tools including `write_file`
**And** a skill orchestration user emits a natural-language request: *"Write a Python class CircularBuffer with iter and len protocol"*
**When** the orchestrator's ReAct loop processes the request
**Then** the orchestrator does NOT pick the `write_file` client tool merely because the request's verb ("write") matches the client tool's verb. The orchestrator identifies `code-generator` as the capability match (code-generation request; `code-generator` covers code-generation), and dispatches via `invoke_ensemble`. The client `write_file` tool remains available for non-capability filesystem-write tasks (e.g., "write the current session log to /tmp/session.log").

### Scenario: Direct completion residual when no capability match exists
**Given** an operator-deployed serve with the ADR-022 amended system prompt active, and the agentic-serving library
**And** a skill orchestration user emits a request the library does not cover (e.g., *"What is the relationship between the Gödel incompleteness theorems and Turing's halting problem?"* — a question with no Topaz-aligned capability ensemble in the library)
**When** the orchestrator's ReAct loop processes the request
**Then** the orchestrator calls `list_ensembles()`, finds no capability match for the request's framing, and produces a direct LLM completion as the residual behavior. The `execution.json` artifact records that no `invoke_ensemble` dispatch occurred for this turn. Direct completion is the correct path under ADR-022 when no internal or client tool applies.

### Scenario: ADR-022 amendment effectiveness is per-orchestrator-profile-conditional
**Given** an operator-deployed serve running with the ADR-022 amended system prompt and an alternative orchestrator profile (e.g., `agentic-orchestrator-offline-tools` — qwen3:14b local via OpenAI-compatible Ollama adapter)
**And** a skill orchestration user emits the same NL request from Scenario 1
**When** the orchestrator's ReAct loop processes the request under qwen3:14b
**Then** the orchestrator may or may not honor the amendment's `prefer invoke_ensemble` clause — the model's reasoning shape governs whether the clause translates to dispatch behavior. The `execution.json` artifact records the actual routing decision; if dispatch occurred, the dispatch is correct; if direct completion or client-tool delegation occurred, ADR-022's disposition (iii) acknowledges the per-profile divergence and ADR-022 §"Effectiveness is configuration-conditional" frames the BUILD/PLAY characterization path.

### Preservation: ADR-021 per-capability dispatch contract is unchanged at the dispatch boundary
**Given** the ADR-021-shipped per-capability dispatch contract (one capability sub-task per orchestrator request; client-side state across sub-tasks; `invoke_ensemble`'s fresh-context property)
**When** Cycle 6 BUILD ships the ADR-022 system-prompt amendment
**Then** ADR-021's dispatch contract operates unchanged at the dispatch boundary. The amendment changes which requests reach `invoke_ensemble` (more NL-framed requests reach it under capability match); the contract for what happens **after** `invoke_ensemble` is called (Tier-Escalation Router, Calibration Gate, fresh-context dispatch) is preserved.

### Preservation: ADR-003's closed 5-tool internal surface is unchanged
**Given** the ADR-003-shipped closed five-tool internal surface (`list_ensembles`, `invoke_ensemble`, `compose_ensemble`, `query_knowledge`, `record_outcome`)
**When** Cycle 6 BUILD ships the ADR-022 system-prompt amendment
**Then** the orchestrator's internal action space remains exactly five tools. The amendment changes the prompt's guidance for **when** to use `invoke_ensemble` relative to direct completion and client-declared tools; it does not introduce a sixth tool or change the existing five tools' semantics.

---

## Feature: Observability Event Routing — Unified Substrate, Two Destinations (ADR-023)

### Scenario: Operator-terminal destination emits per-event INFO lines
**Given** an operator-deployed serve with ADR-023's event-routing surface active, and an invocation of `code-generator` via `invoke_ensemble`
**When** the dispatch executes through the orchestrator's tool dispatch path
**Then** the serve console emits, in dispatch order: `INFO: dispatch start: ensemble=code-generator profile=<profile> dispatch_id=<id>` → `INFO: tier selection: ensemble=code-generator profile=<profile> tier=<tier> topaz_skill=code_generation dispatch_id=<id>` → `INFO: calibration verdict: <verdict> dispatch_id=<id>` → `INFO: dispatch end: ensemble=code-generator duration=<seconds> exit=success dispatch_id=<id>`. The coarse `INFO: tool dispatch: result name=invoke_ensemble kind=success` line from the pre-ADR-023 code is replaced; the new per-event lines carry ensemble identification, duration, verdict, and `dispatch_id` correlation.

### Scenario: Orchestrator-context destination prepends structured observation between turns
**Given** an operator-deployed serve with ADR-023 active, and the same `code-generator` dispatch as above
**When** the orchestrator's next ReAct turn begins (after the dispatch returns control to the orchestrator)
**Then** the orchestrator's turn context begins with a JSON-shaped observation block containing `{dispatched: "code-generator", duration_seconds: <float>, model_profile: <str>, tier: <str>, topaz_skill: "code_generation", calibration_verdict: <verdict>, dispatch_id: <id>}`. The orchestrator's reasoning surface can answer the practitioner question *"What was the total run-time of the ensemble?"* directly from the observation's `duration_seconds` field.

### Scenario: `dispatch_id` correlation joins events for one dispatch
**Given** a multi-dispatch orchestrator session that produces three dispatch events for one `code-generator` invocation: `DispatchTiming(phase="start")`, `TierSelection`, `CalibrationVerdict`, `DispatchTiming(phase="end")`
**When** the session's `execution.json` artifact is read post-hoc
**Then** all four events share the same `dispatch_id` value; joining the events on `dispatch_id` reconstructs the full dispatch picture (ensemble, profile, tier, verdict, start time, end time, duration). Events from other dispatches in the same session have different `dispatch_id` values; the correlation is structurally unambiguous.

### Scenario: `DispatchTiming` event carries `start` and `end` phases with timing data
**Given** an operator-deployed serve with ADR-023 active, and a `code-generator` invocation
**When** the dispatch starts and later returns
**Then** the dispatch emits exactly two `DispatchTiming` events: `DispatchTiming(phase="start", dispatch_id, ensemble_name="code-generator", model_profile, timestamp)` at dispatch start; `DispatchTiming(phase="end", dispatch_id, ensemble_name="code-generator", duration_seconds, exit_status="success", timestamp)` at dispatch return. `duration_seconds` is the wall-clock interval between the two events; `exit_status` is one of `success`, `error`, `timeout`, `aborted`.

### Scenario: Tool-call-emit log line precedes dispatch
**Given** an operator-deployed serve with ADR-023 active, and an orchestrator response stream containing an `invoke_ensemble` tool call structure
**When** the serving layer receives the tool call structure
**Then** the serving layer emits `INFO: tool-call emit: tool=invoke_ensemble dispatch_id=<id>` **before** dispatching the tool call. The log line is a liveness anchor distinct from the post-dispatch `tool dispatch: result` line; operators see "received tool call from cloud LLM at HH:MM:SS" before the dispatch fires.

### Scenario: Inference-wait heartbeat fires after `heartbeat_interval_seconds` of inactivity
**Given** an operator-deployed serve with ADR-023 active, `heartbeat_interval_seconds=30` (default), and an open chat-completions request whose orchestrator is waiting on cloud-LLM inference for >30 seconds without tool activity
**When** 30 seconds elapse without a tool-call-emit event or dispatch start/end event
**Then** the serving layer emits `INFO: inference wait: elapsed=30 session_id=<id>` to the operator-terminal destination. If the wait continues, the heartbeat fires again every 30 seconds until tool activity resumes. The orchestrator-context destination does NOT receive heartbeats (the orchestrator's reasoning surface already has natural session-level timing context).

### Scenario: Validate-once-at-load eliminates per-enumeration noise
**Given** an operator-deployed serve at startup that loads the ensemble library containing legacy schema-drifted YAMLs (e.g., `fan-out-test.yaml`, `plexus-graph-analysis.yaml`) and valid ensembles
**When** the serve completes startup
**Then** the legacy YAMLs produce one `WARN` line each at startup with the file path and validation error rationale; the valid subset is loaded. Subsequent `list_ensembles()` calls return the validated subset without re-emitting warnings. A multi-dispatch session with 8 enumeration cycles produces 0 additional validation warnings for the same legacy YAMLs.

### Scenario: Final dispatch's in-turn routing skipped; end-of-session summary captures events
**Given** an operator-deployed serve with ADR-023 active, and a session whose final operation before close is a `code-generator` dispatch
**When** the dispatch completes and the session closes immediately afterward
**Then** the in-turn orchestrator-context routing is skipped (no next turn exists). The dispatch's events route to the **end-of-session summary** under the `dispatch_log` key in the session's `execution.json` artifact. The operator-terminal destination emits all dispatch events at emission time regardless of session close.

### Preservation: ADR-018 Tier-Router-Audit drift criteria are unchanged
**Given** the ADR-018-shipped Tier-Router-Audit with three drift criteria (verdict-distribution shift, escalation-vs-outcome correlation, bypass-rate trend)
**When** ADR-023 adds the `dispatch_id` field to `AuditDiagnostic` events
**Then** the audit's drift-criteria semantics are unchanged. The `dispatch_id` field is additive metadata enabling per-dispatch correlation; the criteria's findings (drift detected / not detected) operate on the same population data and produce the same verdicts.

### Preservation: `CalibrationSignal` cross-layer channel is unchanged
**Given** the ADR-016-shipped cross-layer calibration signal channel (L0 → L1 read-only, bounded)
**When** ADR-023 adds `dispatch_id` to `CalibrationSignal` events and excludes them from orchestrator-context routing by default
**Then** the cross-layer channel's existing bounding mechanisms (a)–(e) and its read-only / signal-channel-specific scope are unchanged. The `dispatch_id` is additive; the orchestrator-context exclusion is a routing-destination decision, not a change to the signal channel's emission semantics.

### Preservation: `execution.json` artifact existing fields are unchanged
**Given** the Cycle 5 BUILD-shipped `execution.json` artifact shape (`{ensemble, status, input, results, metadata, synthesis}`)
**When** ADR-023 adds the `dispatch_log` key for end-of-session summary
**Then** the existing fields (`ensemble`, `status`, `input`, `results`, `metadata`, `synthesis`) are preserved. The `dispatch_log` is an additive key; readers parsing the existing fields ignore it harmlessly until they choose to consume it.

---

## Feature: Common I/O Envelope — Typed Dispatch Response (ADR-024)

### Scenario: `invoke_ensemble` returns typed `DispatchEnvelope`
**Given** an operator-deployed serve with ADR-024 active, and a `text-summarizer` invocation
**When** the dispatch completes
**Then** `invoke_ensemble` returns a `DispatchEnvelope` dataclass instance with `status: "success"`, `primary: <human-readable summary line>`, `diagnostics: {ensemble: "text-summarizer", dispatch_id: <id>, duration_seconds: <float>, model_profile: <str>, tier: <str>, topaz_skill: "summarization", calibration_verdict: <verdict>, audit_findings: []}`. Optional fields (`structured`, `errors`, `artifacts`) are populated only when applicable per ADR-024 / ADR-025.

### Scenario: `output_schema:` declaration populates `envelope.structured`
**Given** an operator-deployed serve with ADR-024 active, a `claim-extractor` ensemble YAML carrying `output_schema:` declaring `{claims: [{text: string, label: string}, ...]}`, and a dispatch of `claim-extractor` against source material
**When** the dispatch completes
**Then** `envelope.structured` carries a typed payload matching the schema shape: `{claims: [{text: "...", label: "established"}, {text: "...", label: "contested"}, ...]}`. Downstream consumers (other ensembles in a composition; the orchestrator's reasoning surface) parse `envelope.structured` directly without re-parsing `primary`.

### Scenario: Capability ensemble without `output_schema:` produces `envelope.structured = None`
**Given** an operator-deployed serve with ADR-024 active, and a `prose-improver` ensemble YAML without `output_schema:` declared
**When** the ensemble is dispatched
**Then** the envelope's `structured` field is `None`; only `primary` carries the deliverable. Downstream consumers expecting `structured` handle the `None` case (no schema declared); composition continues to rely on the orchestrator's reasoning surface for between-stage data shaping (per spike β's headline finding).

### Scenario: `errors[]` populated on partial-failure dispatch
**Given** an operator-deployed serve with ADR-024 active, and a multi-stage capability ensemble (e.g., a hypothetical `lit-review` composition) where one of three stages errors mid-dispatch
**When** the dispatch returns
**Then** `envelope.status: "partial"`, `envelope.errors: [{stage: <stage-name>, error_type: <typed-error-name>, message: <str>, recoverable: <bool>}]`. Stages that completed successfully populate `envelope.primary` / `envelope.structured` / `envelope.artifacts` as if they had completed in isolation; the partial-failure shape is observable from `status` + `errors[]`.

### Scenario: `diagnostics.dispatch_id` correlates envelope to ADR-023 events
**Given** a session producing a `code-generator` dispatch under ADR-023 + ADR-024 active
**When** the dispatch's envelope is constructed and the session's events are written to `execution.json`
**Then** `envelope.diagnostics.dispatch_id` matches the `dispatch_id` field on all ADR-023 events for that dispatch (`DispatchTiming(start)`, `TierSelection`, `CalibrationVerdict`, `DispatchTiming(end)`). Joining envelope and events on `dispatch_id` reconstructs the full dispatch context.

### Preservation: ADR-021's per-capability dispatch contract is preserved at the contract level
**Given** the ADR-021-shipped per-capability dispatch contract (the orchestrator returns the capability ensemble's output to the skill framework as the chat completion response)
**When** ADR-024 codifies the typed envelope as the response shape
**Then** ADR-021's substantive commitments (one capability sub-task per request; client-side state across sub-tasks; `invoke_ensemble`'s fresh-context property; Tier-Escalation Router pre-specified routing; Calibration Gate firing per sub-task) operate unchanged. The envelope is the structural shape the dispatch contract's response takes; the contract's behavior is preserved.

### Preservation: `execution.json` artifact existing shape is preserved
**Given** the Cycle 5 BUILD-shipped `execution.json` artifact shape with `metadata` field
**When** ADR-024 introduces the `diagnostics` field-name at the envelope layer
**Then** the `execution.json` artifact retains `metadata` (the rename is at the envelope layer only); readers parsing the existing artifact shape are not broken. Cycle 7+ may produce an artifact-shape ADR aligning artifact and envelope field names.

---

## Feature: Artifact-as-Substrate for Capability Ensemble Deliverables (ADR-025)

### Scenario: Capability ensemble writes deliverable to session-dir artifact path
**Given** an operator-deployed serve with ADR-025 active, a `code-generator` ensemble YAML carrying `output_substrate: artifact` (default for capability ensembles in Cycle 6), and a dispatch with `session_id = "2026-05-15T14:32:08Z-a7f3"`, `dispatch_id = "dispatch-001"`, deliverable name `circular_buffer`
**When** the dispatch completes
**Then** the deliverable is written to `.llm-orc/agentic-sessions/2026-05-15T14:32:08Z-a7f3/dispatch-001/circular_buffer.py`. The envelope's `primary` field carries a one-line summary referencing the artifact (e.g., *"Wrote class CircularBuffer to agentic-sessions/2026-05-15T14:32:08Z-a7f3/dispatch-001/circular_buffer.py (1.2 KB, application/python)"*). The envelope's `artifacts[0]` carries the typed reference: `{path: "agentic-sessions/2026-05-15T14:32:08Z-a7f3/dispatch-001/circular_buffer.py", content_type: "application/python", size_bytes: 1247, summary: "Class CircularBuffer with iter and len protocol; 24 lines.", retention: "session"}`.

### Scenario: System ensemble produces inline-response envelope
**Given** an operator-deployed serve with ADR-025 active, the system ensemble `agentic-calibration-checker` (internal infrastructure, not in capability scope), and a calibration check invocation
**When** the dispatch completes
**Then** the envelope's `primary` carries the verdict content directly (e.g., *"Verdict: Proceed; confidence: 0.87"*); `artifacts[]` is empty (`None` or `[]`). The system ensemble's response shape is inline per ADR-025's "system ensembles remain inline" scope.

### Scenario: Substrate-routed dispatch's envelope is not passed through `agentic-result-summarizer`
**Given** an operator-deployed serve with ADR-025 active, the AS-7 amendment in force, and a substrate-routed `code-generator` dispatch
**When** the dispatch returns control to the orchestrator
**Then** the orchestrator does NOT invoke `agentic-result-summarizer` for this dispatch. The envelope's `primary` (summary line) and `artifacts[0]` (typed reference) are already summary-shaped; AS-7's amended default-with-conditional-skip applies the skip. The orchestrator's context carries the envelope as observation per ADR-023; total context impact is small.

### Scenario: Inline-response dispatch retains `agentic-result-summarizer` per AS-7 amended
**Given** an operator-deployed serve with ADR-025 active, the AS-7 amendment in force, and an inline-response dispatch (e.g., a hypothetical capability ensemble declared `output_substrate: inline`, or a system ensemble)
**When** the dispatch returns control to the orchestrator
**Then** the orchestrator invokes `agentic-result-summarizer` per ADR-004 (mandate scope: inline-response path). AS-7's amended default applies — the mandate operates within the inline-response scope unchanged.

### Scenario: Calibration gate evaluators receive `primary` + `artifacts[0].summary` by default
**Given** an operator-deployed serve with ADR-025 active, a capability ensemble whose YAML does not declare `calibration_substrate_access: artifact`, and a calibration-gate-eligible dispatch
**When** the calibration gate's critic agents evaluate the dispatch
**Then** the evaluator agents receive the envelope's `primary` field and `artifacts[0].summary` field as their evaluation input. They do NOT read the artifact content at `artifacts[0].path`. The verdict produced is based on summary-line evaluation; for ensembles whose quality is reasonably inferrable from a one-line summary, this default suffices.

### Scenario: Calibration gate reads artifact content for `code-generator` (opt-in)
**Given** an operator-deployed serve with ADR-025 active, the `code-generator` ensemble YAML declaring `calibration_substrate_access: artifact`, and a calibration-gate-eligible code-generation dispatch
**When** the calibration gate's critic agents evaluate the dispatch
**Then** the evaluator agents receive a tool-call surface to read the artifact at `artifacts[0].path`; they read the code content and evaluate against the actual deliverable. The verdict produced reflects code-correctness analysis, not summary-line inference. This is the highest-cost evaluation path; only ensembles whose quality cannot be evaluated from summary alone opt in.

### Scenario: Session-close cleanup removes `retention: session` artifacts
**Given** a session with three substrate-routed dispatches producing artifacts with `retention: session`, and one dispatch producing an artifact with `retention: durable`
**When** the session closes
**Then** the serve removes the three session-retention artifacts (and the per-dispatch directories that contained them) under `.llm-orc/agentic-sessions/<session_id>/`. The durable artifact remains on disk. The session directory itself remains if any durable artifact persists; otherwise the directory is removed.

### Scenario: Dial-back falsification indicator fires
**Given** an operator-deployed serve where, during post-BUILD PLAY observation, three or more capability ensembles have been declared `output_substrate: inline` as substrate-opt-outs
**When** the PLAY field-notes record the opt-out pattern
**Then** the dial-back deliberation surfaces in the cycle observing the pattern — the substantive-deliverable scope (rejected at Cycle 6 DECIDE) is re-examined with the operator's per-ensemble opt-out evidence. The dial-back is fire-on-evidence rather than fire-on-discomfort per ADR-025's "Dial-back falsification criteria."

### Preservation: ADR-007 Calibration Gate verdict trichotomy is unchanged
**Given** the ADR-007-shipped Calibration Gate producing Proceed / Reflect / Abstain verdicts on dispatched ensemble outputs
**When** Cycle 6 BUILD ships substrate-routing and the ADR-025 evaluation-surface specification
**Then** the gate's verdict trichotomy (Proceed / Reflect / Abstain) is unchanged. The evaluation **surface** changes (summary-only / structured / artifact-content depending on per-ensemble configuration); the verdict **shape** is preserved. Downstream consumers (the Tier-Escalation Router per ADR-015; the audit per ADR-018) consume verdicts unchanged.

### Preservation: ADR-014 Calibration Gate trajectory-level extension is unchanged
**Given** the ADR-014-shipped trajectory feature extraction (HTC / AUQ) on dispatched-ensemble trajectories
**When** Cycle 6 BUILD ships substrate-routing
**Then** trajectory feature extraction operates unchanged on the dispatched ensemble's trajectory data (model output tokens, attention weights, decision confidence). Substrate-routing affects the **deliverable** location (artifact vs. inline); the **trajectory data** (the ensemble's reasoning surface during dispatch) is independent of substrate routing.

### Preservation: ADR-004's per-invocation escape hatch within the inline-response scope is unchanged
**Given** the ADR-004-shipped per-invocation `raw_output=True` escape hatch (small classifier outputs that pass through summarization)
**When** ADR-025 amends AS-7 to default-with-conditional-skip
**Then** within the inline-response scope (system ensembles; future ensembles opting out via `output_substrate: inline`), ADR-004's per-invocation escape hatch operates unchanged. The `raw_output=True` flag continues to bypass `agentic-result-summarizer` for classifier-shape inline-response dispatches. Substrate-routing is a separate, dispatch-shape-level decision; the two skip mechanisms compose without contradiction.

### Preservation: `compose_ensemble` (ADR-006) scope is unchanged
**Given** the ADR-006-shipped `compose_ensemble` scope (runtime composition of capability ensembles from existing library primitives)
**When** ADR-025 introduces substrate-routing for capability ensembles
**Then** `compose_ensemble`'s scope is unchanged — it composes capability ensembles from primitives; the composed ensemble is itself substrate-routed if it carries `output_substrate: artifact`. The composition mechanism does not change; the composed ensemble's response shape follows the same rule as any other capability ensemble.

### Preservation: Existing `.llm-orc/artifacts/agentic-serving/<ensemble>/<timestamp>/` artifact tree is deprecated, not actively removed
**Given** the Cycle 5 BUILD-shipped artifact path `.llm-orc/artifacts/agentic-serving/<ensemble>/<timestamp>/<dispatch>/execution.json`
**When** Cycle 6 BUILD ships the new `.llm-orc/agentic-sessions/<session_id>/<dispatch_id>/` structure
**Then** new dispatches write under the new structure; the old artifact tree is **deprecated** but not actively cleaned up by BUILD. Operators with pinned references to the old path migrate manually. The old path's existence does not affect new-dispatch behavior; the deprecation is structural rather than active removal.

---

## Cycle 7 Acceptance Criteria Table additions (per ADR-027 + ADR-028 + ADR-029 + ADR-030 + ADR-031 + ADR-032 + AS-10)

The following Cycle 7 acceptance criteria are emergent (observable only at integration), aggregate (covered by multiple scenarios composing), or specify an integration layer that individual scenarios stub out. BUILD Step 5.5 (when BUILD runs on the Cycle 7 ADRs) verifies each entry at its specified layer.

| Criterion | Specified layer | Verification method | Layer-match check |
|-----------|----------------|--------------------|-----|
| Framework-driven dispatch pipeline satisfies AS-9 universally on chat-completions surface | Integration (end-to-end chat-completions request through plan → dispatch → synthesize) | Composes scenarios "Routing-planner emits action=dispatch on capability-matched request" + "Framework executes plan via OrchestratorToolDispatch.dispatch()" + "Response-synthesizer produces user-facing response from structured input" with real-type wiring (no stubs at stage boundaries) | no — individual scenarios exercise each stage in isolation; integration test verifies the three-stage composition produces a chat-completion response where no orchestrator-LLM is in the routing-decision or post-dispatch-synthesis surface |
| Configuration-honesty sub-promise delivered at three signal layers consistently | Integration (chat-completions response shape inspection across action=dispatch, action=direct, dispatch-failure paths) | Composes scenarios "Dispatch response carries served_by:ensemble:<name> at header + body metadata layers" + "Direct-completion response carries served_by:direct at header + body metadata + content layers" + "Dispatch failure response carries served_by:direct_fallback;dispatch_failed:<type>" — verified across all three paths | no — individual scenarios exercise per-path labeling; integration verifies the three-layer consistency across all paths (e.g., dispatch responses do NOT carry content-layer Rule 5 noise; direct-completion responses do) |
| Cost-distribution-accountability rate metric is computable from dispatch events | Integration (rolling window over emitted dispatch events) | Composes scenarios "action=direct emits direct_completion_fallback event" + "Routing-planner emission rate is observable from event substrate" with a rolling-window aggregation in the operator-terminal destination per ADR-023 | no — individual scenarios exercise event emission; integration test or live operator-deployment evidence verifies the rolling metric is computable and meaningful |
| `tool_choice` bridge signal delivered consistently across all response paths | Integration (chat-completions responses on action=dispatch and action=direct paths with tool_choice present in request) | Composes scenarios "tool_choice present + action=direct emits three-layer advisory including content-layer" + "tool_choice present + action=dispatch emits headers + body metadata, no content-layer noise" + "tool_choice='auto' is treated as absent (no advisory fires)" | no — individual scenarios exercise per-action-path advisory shape; integration test verifies the conditional-content-layer behavior holds across both paths |
| Population A tier coverage operates within deployment-specific timeout constraints | Live deployment against Population A clients with operator-tuning applied | Operator-deployment-specific verification using the smoke test specified in ADR-031 §Tier B (single-capability NL request within requestTimeoutMs minus 5s headroom; chained-composition request within requestTimeoutMs minus 10s headroom; both within Tier B operator's Cline configuration after tuning) | no — synthetic test cannot exercise Population A client behavior; operator-deployed smoke test against operator-configured Cline (or other Tier B client) is the verification layer |
| Capability-list discovery surfaces the framework's loaded ensembles via OpenAI-protocol-compatible mechanism | Integration (/v1/models endpoint extension or sibling endpoint) | Composes scenario "/v1/models lists capability ensembles with topaz_skill metadata" + "Capability list reflects framework's loaded-ensemble registry" with framework reload event | no — individual scenarios exercise endpoint shape; integration test verifies endpoint reflects the live registry under ensemble add/remove events |

The Layer-match "no" entries are not failures — they are the table working as designed, surfacing where BUILD Step 5.5 closes integration verification gaps with dedicated tests or harness work, and where live operator-deployment evidence is the natural verification surface.

---

## Feature: Framework-Driven Dispatch Pipeline (ADR-027)

### Scenario: Chat-completions request flows through plan → dispatch → synthesize pipeline
**Given** an operator-deployed serve with ADR-027 active, a deployment-loaded routing-planner ensemble (per ADR-028) and response-synthesizer ensemble (per ADR-029), a chat-completions request with a capability-matched NL prompt
**When** the chat-completions handler processes the request
**Then** the framework invokes the routing-planner ensemble first (producing a JSON dispatch plan with `action: "dispatch"`), then dispatches the named capability ensemble via `OrchestratorToolDispatch.dispatch()`, then invokes the response-synthesizer ensemble with `(ORIGINAL REQUEST + PLAN + DISPATCH RESULTS)` as structured input, then returns the synthesizer's `message.content` as the chat-completion response with `finish_reason: stop`. No `OrchestratorRuntime` ReAct loop fires for the request; the orchestrator-LLM is not in the routing-decision or post-dispatch-synthesis surface.

### Scenario: No-capability-match request flows through plan → direct-completion synthesize path
**Given** an operator-deployed serve with ADR-027 active and a chat-completions request whose content has no capability match in the framework's loaded-ensemble registry
**When** the chat-completions handler processes the request
**Then** the routing-planner ensemble emits `action: "direct"` with `ensemble: null`; the framework skips the dispatch stage; the response-synthesizer is invoked with `(ORIGINAL REQUEST + PLAN + DISPATCH RESULTS=empty)` and produces a useful response from the request content alone, applying Rule 5 framing (per ADR-029).

### Scenario (integration): Plan-stage output is a typed `InternalToolCall`-compatible shape
**Given** an operator-deployed serve with ADR-027 active and a request that produces `action: "dispatch"` from the routing-planner
**When** the framework translates the planner's `{"action", "ensemble", "input", "rationale"}` JSON into the existing `OrchestratorToolDispatch.dispatch()` input shape
**Then** the adapter constructs an `InternalToolCall` (or its equivalent for the dispatch surface) with `name="invoke_ensemble"` and `arguments={"name": <plan.ensemble>, "input": <plan.input>}`; the dispatch proceeds via the existing `OrchestratorToolDispatch` machinery without re-implementing the dispatch contract.

### Scenario: `OrchestratorRuntime` is not invoked on the chat-completions surface
**Given** an operator-deployed serve with ADR-027 active and any chat-completions request (capability-matched, no-capability-match, or explicit-naming)
**When** the chat-completions handler runs
**Then** no instance of `OrchestratorRuntime` is constructed during request handling; the handler's runtime use of `OrchestratorRuntime.run()` from Cycle 6 is replaced by the framework-driven pipeline. The `OrchestratorRuntime` class remains in the codebase per ADR-027 §Decision §"OrchestratorRuntime status under ADR-027" (the ARCHITECT-deferred disposition decides between preserve / wire-CLI / remove).

### Scenario: Plan with invalid action or unregistered ensemble is rejected → direct completion (Spike ν)
**Given** an operator-deployed serve with ADR-027 active and a routing-planner that emits a plan whose `action` is not in `{"dispatch", "direct"}` (e.g., an injected `"action": "launch"`) or whose `ensemble` names a capability not in the framework's loaded-ensemble registry (e.g., an injected `"oracle"`)
**When** the Dispatch Pipeline validates the plan before the dispatch stage
**Then** the invalid plan is rejected; no dispatch fires against an unregistered ensemble and no non-`{dispatch,direct}` action is executed; the request falls to the direct-completion synthesize path (`DISPATCH RESULTS=empty`, Rule 5 framing per ADR-029); a `DirectCompletionFallback` event is emitted with the rejected-plan rationale. Driver: Spike ν Surface 3 cases E1 (injection obeyed) + E3 (fabricated ensemble trusted), both neutralized by pipeline plan validation. Plan validation is a non-optional pipeline stage, not an operator tuning option.

### Scenario: Empty or unparseable planner output → direct completion (Spike ν)
**Given** an operator-deployed serve with ADR-027 active and a routing-planner that returns an empty response, a response containing only a `<think>` block with no JSON object, or otherwise unparseable output
**When** the Dispatch Pipeline's plan-parsing stage cannot extract a conformant plan
**Then** the pipeline treats the unparseable plan as a defined path (not an exception): it routes to the direct-completion synthesize path (`DISPATCH RESULTS=empty`, Rule 5 framing per ADR-029) and emits a `DirectCompletionFallback` event noting the parse failure. Driver: Spike ν Surface 3 case A6 (cheap-tier empty-response reliability miss). The safe default for an absent plan is direct completion, consistent with the planner's `direct` fallback semantics.

### Preservation: `OrchestratorToolDispatch.dispatch()` contract is unchanged
**Given** the existing `OrchestratorToolDispatch.dispatch()` mechanism shipped at Cycle 6 with the `InternalToolCall` input shape and dispatch envelope output shape (per ADR-024)
**When** ADR-027 routes the planner's dispatch action through the same machinery
**Then** the dispatch mechanism's contract is unchanged: same input shape, same envelope output shape, same calibration-gate firing within the dispatched ensemble (per ADR-007, ADR-014), same tier-router behavior (per ADR-015), same audit dispatch (per ADR-018). ADR-027 changes only the *caller* of `dispatch()`, not the dispatch mechanism itself.

### Preservation: ADR-021's per-capability dispatch contract structural commitments unchanged
**Given** ADR-021's per-capability dispatch contract (one capability sub-task per request; client-side workflow state; fresh-context property; calibration-gate-per-sub-task) on the chat-completions surface
**When** ADR-027 ships and the routing-planner takes over the routing decision
**Then** all four structural commitments are preserved. The planner emits one dispatch action per chat-completions request; cross-request state remains client-side (the synthesizer reads multi-turn `messages[]` per ADR-029 but does not maintain server-side workflow state); the dispatched ensemble runs in a fresh context (no orchestrator-context bleeds into it); the calibration gate fires per dispatch.

### Preservation: `llm-orc invoke` CLI surface is unaffected by ADR-027
**Given** an operator using the `llm-orc invoke` CLI to execute a named ensemble directly
**When** ADR-027 ships and replaces the chat-completions handler's runtime
**Then** the CLI's execution path (via `OrchestraService` directly per `cli_commands.py:28`) is unchanged. The CLI does not currently use `OrchestratorRuntime` (per Tranche 4 conformance scan Finding 2); ADR-027's changes do not affect it. ADR-021's original dispatch shapes continue to apply on the CLI surface.

---

## Feature: Routing-Planner Ensemble (ADR-028)

### Scenario: Explicit-naming NL request produces dispatch action with the named ensemble
**Given** a deployed routing-planner ensemble (cheap-tier qwen3:8b empirical baseline) and a chat-completions request whose user message names a specific ensemble (e.g., *"use the code-generator ensemble to write a sorting function"*) where `code-generator` is in the framework's loaded capability list
**When** the framework invokes the routing-planner with `(ORIGINAL REQUEST + CAPABILITY LIST)`
**Then** the planner emits JSON conforming to the dispatch-plan schema: `action: "dispatch"`, `ensemble: "code-generator"`, `input: <derived from user message>`, `rationale: <one-sentence>`. Per Spike ζ's 20-prompt battery, explicit-naming shape produces 100% strict capability-match.

### Scenario: NL clear-match request produces dispatch action without explicit naming
**Given** a deployed routing-planner ensemble and a chat-completions request whose user message describes a task matching a capability (e.g., *"summarize this paper into bullet points"* with `text-summarizer` in the capability list)
**When** the framework invokes the routing-planner
**Then** the planner emits `action: "dispatch"`, `ensemble: "text-summarizer"`, `input: <task content>`, `rationale: <match reasoning>`. The structural-bounding property (per AS-9) ensures the cheap-tier model produces conformant JSON on this single-decision task.

### Scenario: No-capability-match request produces direct action
**Given** a deployed routing-planner ensemble and a chat-completions request whose user message has no capability match (e.g., *"what's the weather in Reykjavik?"* with no weather-capability ensemble in the list)
**When** the framework invokes the routing-planner
**Then** the planner emits `action: "direct"`, `ensemble: null`, `rationale: <no-match reasoning>`. The `input` field is omitted (not required when `action: "direct"`).

### Scenario: Adversarial / ambiguous request produces defensible-judgment dispatch
**Given** a deployed routing-planner ensemble and a chat-completions request that could plausibly match multiple capabilities (e.g., *"help me write a script"* with both `code-generator` and `script-builder` in the capability list)
**When** the framework invokes the routing-planner
**Then** the planner emits `action: "dispatch"` with one of the defensibly-matched ensembles; the `rationale` field explains the choice; per Spike ζ's 100% defensible-judgment-match across the 20-prompt battery, the choice is reasonable even if not the single ideal answer. The Calibration Gate within the planner ensemble may emit Reflect verdicts on persistently ambiguous patterns, feeding the Tier-Router Audit per ADR-018.

### Scenario: Planner output schema includes `input` field for dispatch actions (Track A refactor before BUILD)
**Given** the Spike ζ scratch routing-planner ensemble YAML at the BUILD-phase starting point
**When** the YAML is updated to add the `input` field to the output JSON schema per ADR-028 §Output contract
**Then** the planner produces `{"action", "ensemble", "input", "rationale"}` instead of the spike's `{"action", "ensemble", "rationale"}`. This is a `refactor:` commit on the spike artifact (per Tranche 4 conformance-scan Finding 4), not a behavioral change; the 20-prompt battery continues to pass with the added field populated.

### Scenario (integration): Routing-planner output feeds OrchestratorToolDispatch.dispatch() with real types
**Given** an operator-deployed serve with the production routing-planner ensemble + the existing `OrchestratorToolDispatch.dispatch()` machinery (not stubbed)
**When** the planner emits a dispatch plan and the framework translates it into the dispatch call
**Then** `OrchestratorToolDispatch.dispatch()` receives the translated input without type errors; the dispatch fires against the named capability ensemble in the framework's registry; envelope content returns per ADR-024.

### Preservation: Routing-planner ensemble operates within AS-9's structural-bounding property
**Given** the routing-planner ensemble specified per ADR-028 (cheap-tier; single-decision-shaped task; no multi-step reasoning or tool calls)
**When** the planner produces a dispatch plan
**Then** the structural-bounding property (per AS-9) is preserved: the planner's role is constrained to producing JSON from a given input; it does not chain through tool calls, file reads, or multi-step reasoning. The empirical floor (Spike ζ — 100% JSON conformance + 90% strict capability-match) holds.

### Preservation: Routing-planner operates within AS-10's request-content-alone scope
**Given** the routing-planner's input contract (`ORIGINAL REQUEST + CAPABILITY LIST`) per ADR-028
**When** the planner reads the request content
**Then** no client-side opt-in signals are consumed (no HTTP headers, no skill-framework manifest fields, no metadata convention). The planner operates on chat-completions request body content (`messages[]`, `model`, optional `tools[]`, optional `tool_choice` per ADR-030) and the framework's loaded-ensemble registry alone. AS-10 (per ADR-026) is satisfied structurally.

---

## Feature: Response-Synthesizer Ensemble (ADR-029)

### Scenario: Synthesizer reads structured input and produces user-facing response
**Given** a deployed response-synthesizer ensemble (cheap-tier qwen3:8b empirical baseline) and a completed dispatch where the routing-planner produced `action: "dispatch"` and the framework executed the dispatch successfully
**When** the framework invokes the synthesizer with `(ORIGINAL REQUEST + PLAN + DISPATCH RESULTS)` where DISPATCH RESULTS carries the envelope's `primary` and `artifacts[0].summary` per ADR-024
**Then** the synthesizer produces `message.content` as a string conforming to OpenAI chat-completion semantics + `finish_reason: stop`. No tool calls are emitted; the response carries content only.

### Scenario: Rule 1 — synthesizer uses only DISPATCH RESULTS for substantive claims
**Given** a deployed response-synthesizer and a dispatch where the dispatched ensemble's envelope contains specific numerical figures (e.g., "Iceland population 389,444 as of latest count")
**When** the synthesizer composes the user-facing response
**Then** numerical claims in the response cite figures from DISPATCH RESULTS verbatim; the synthesizer does not invent or substitute figures from training data. Per Spike ε ε.1 + Spike μ.3, the rule holds at qwen3:8b across the n=13 test battery.

### Scenario: Rule 2 — synthesizer reports Planned-but-not-run honestly
**Given** a deployed response-synthesizer and a PLAN that named two dispatch steps, where the second step failed irrecoverably (e.g., schema-non-conformance after Calibration Gate retries)
**When** the synthesizer composes the user-facing response
**Then** the response declares that the second ensemble was "Planned but not run" rather than fabricating output the missing ensemble would have produced. Per Spike ε ε.1's PLAY-note-22 test case (the historical confabulation pattern), the structurally-bounded synthesizer correctly reports the missing dispatch.

### Scenario: Rule 4 — synthesizer cites figures verbatim (no rounding drift)
**Given** a deployed response-synthesizer and a dispatch where the envelope contains precise figures (e.g., 402,329)
**When** the synthesizer composes the response referencing the figure
**Then** the response uses the source figure verbatim. *Empirical note:* Spike ε T3 + Spike ε' Finding ε'.2 characterized two distinct rounding-drift modes (Mode 1 precise-figure rounding; Mode 2 large-number millions rendering); Rule 4 reduces but does not eliminate drift. The mitigation playbook per ADR-029 §"Rounding-drift mitigation playbook" (system-prompt sharpening → tier escalation → runtime fidelity check) addresses production-traffic drift.

### Scenario: Rule 5 — synthesizer applies honest direct-completion framing when DISPATCH RESULTS is empty
**Given** a deployed response-synthesizer and an `action: "direct"` invocation where DISPATCH RESULTS is empty
**When** the synthesizer produces the user-facing response
**Then** the response includes Rule 5 framing ("this answer was generated directly without dispatching a specialist ensemble" or framework-determined equivalent). Per ADR-029 §"Rule 5 framing requirement scope (OQ #23)" — load-bearing-default for Cycle 7 BUILD; runtime validation checks for the framing marker; absence triggers Calibration Gate Reflect.

### Scenario: Rule 6 — synthesizer enumerates framework conventions with hedging in direct-completion mode
**Given** a deployed response-synthesizer and a direct-completion request asking about file paths, framework conventions, or implementation specifics (e.g., *"where do API routes live in this project?"*)
**When** the synthesizer composes the response
**Then** the response enumerates generic conventions (e.g., *"common conventions include `routes.py`, `api.py`, `endpoints.py`"*) with explicit hedging + uncertainty acknowledgment + clarification request rather than fabricating confident-specific paths. Per Spike μ.1, the pattern emerges naturally under Rules 1 + 5 at qwen3:8b; Rule 6 codifies it for model-substitution robustness.

### Scenario: Multi-turn continuity preserved when prior turns are in input
**Given** a deployed response-synthesizer and a chat-completions request with multi-turn `messages[]` (e.g., prior assistant response + new user turn referencing it)
**When** the framework serializes the prior turns into the synthesizer's ORIGINAL REQUEST input
**Then** the synthesizer's response correctly references prior-turn content; the continuity is preserved across turns. Per Spike ε' C1 + C2, native `messages[]` handling is mechanical ARCHITECT-phase work; the structural-bounding property holds across multi-turn shapes.

### Scenario: Calibration Gate Reflect fires on Rule 5 framing absence
**Given** a deployed response-synthesizer with the three Reflect-trigger criteria per ADR-029 active and an `action: "direct"` invocation where the synthesizer's output lacks Rule 5 framing marker
**When** the runtime validation runs against the response
**Then** the Calibration Gate Reflect verdict fires; the Tier-Escalation Router per ADR-015 may escalate the synthesizer to a higher-tier model for the retry; the audit per ADR-018 records the criterion firing for drift analysis.

### Scenario: Synthesizer's Rule 6 codification is in the BUILD-phase ensemble YAML (Track A refactor)
**Given** the Spike ε scratch response-synthesizer ensemble YAML at the BUILD-phase starting point
**When** the YAML is updated to add Rule 6 to the system prompt per ADR-029 §"Rule 6" codification (per Tranche 4 conformance-scan Finding 6)
**Then** the synthesizer's system prompt enumerates all six rules. This is a `refactor:` commit on the spike artifact, not a behavioral change at qwen3:8b (Rule 6 codifies the emergent pattern); the production ensemble carries the explicit rule for model-substitution robustness.

### Preservation: Synthesizer's structural-bounding property prevents C4 failure mode
**Given** the response-synthesizer's input contract (structured `REQUEST + PLAN + DISPATCH RESULTS`; no tool-call surface in synthesizer's context)
**When** the synthesizer encounters a dispatch with substrate-routed deliverable artifact paths in the envelope
**Then** the synthesizer cannot issue file-read tool calls (the output surface is text content alone). The orchestrator-LLM's emergent "chain through file-read of substrate path" failure mode (Spike λ-paid F-paid-4; PLAY note 22) is structurally prevented. The synthesizer reads `artifacts[0].summary` from the envelope (per ADR-024) but does not access `artifacts[0].path` directly.

### Preservation: AS-7 amended summarization rules apply unchanged to substrate-routed dispatches
**Given** ADR-025-shipped substrate routing + ADR-029-shipped response-synthesizer
**When** a substrate-routed capability ensemble dispatch returns its envelope and the synthesizer reads it as DISPATCH RESULTS
**Then** AS-7's amended default-with-conditional-skip rule applies unchanged — substrate-routed dispatches skip content-level result-summarizer invocation; the envelope's `primary` + `artifacts[0].summary` already carry summary-shaped content; the synthesizer consumes those fields. The synthesizer does NOT re-summarize the envelope; AS-7's skip is preserved.

---

## Feature: `tool_choice` Disposition with Bridge Mechanism (ADR-030)

### Scenario: Chat-completions request with `tool_choice: {"name": "<ensemble>"}` triggers bridge advisory
**Given** an operator-deployed serve with ADR-030's bridge mechanism active and a chat-completions request including `tool_choice: {"type": "function", "function": {"name": "code-generator"}}`
**When** the framework parses the request
**Then** the `_ChatCompletionsRequest` Pydantic model observes the `tool_choice` field (newly added per Track B BUILD work; per ADR-030 §Bridge advisory specification); the framework processes the request via ADR-027's pipeline normally (does NOT use `tool_choice` for routing); the response carries the bridge signal at three layers per the conditional shape (headers + body metadata on all responses; content-layer Rule 5-adjacent acknowledgment on `action: "direct"` responses only).

### Scenario: Bridge advisory present at headers layer on all `tool_choice`-bearing requests
**Given** a chat-completions request with `tool_choice` set (any non-default value) and the bridge mechanism active
**When** the response is returned
**Then** the response includes header `X-LLM-Orc-Tool-Choice-Handling: deferred` (or the framework's chosen `served-by` header family per BUILD design). The header fires on both `action: "dispatch"` and `action: "direct"` paths.

### Scenario: Bridge advisory present at body metadata layer on all `tool_choice`-bearing requests
**Given** a chat-completions request with `tool_choice` set and the bridge mechanism active
**When** the response is returned
**Then** the response includes `metadata.tool_choice_handling: "deferred"` (or equivalent within the response shape ADR-032's body-metadata mechanism establishes). The metadata fires on both `action: "dispatch"` and `action: "direct"` paths.

### Scenario: Bridge advisory content-layer acknowledgment fires only on `action: "direct"` responses
**Given** a chat-completions request with `tool_choice` set and the bridge mechanism active
**When** the framework processes the request and the routing-planner emits `action: "direct"`
**Then** the synthesizer's response content includes the Rule 5-adjacent acknowledgment ("this answer was generated directly without dispatching a specialist ensemble; `tool_choice` was received but not honored for routing in this build"). When the planner emits `action: "dispatch"`, the response content does NOT carry the bridge acknowledgment (headers + metadata layers are sufficient; no content-layer noise per ADR-030 §Bridge advisory specification).

### Scenario: `tool_choice: "auto"` is treated as equivalent to absent `tool_choice` (no bridge advisory)
**Given** a chat-completions request with `tool_choice: "auto"` (the OpenAI default value) and the bridge mechanism active
**When** the framework processes the request
**Then** the request flows through ADR-027's pipeline normally; no bridge advisory fires at any layer; the response is standard OpenAI-compatible per ADR-032's honest response labeling for the appropriate path (dispatch or direct). Per ADR-030 §Bridge advisory specification, the advisory fires only on non-default `tool_choice` values.

### Preservation: Requests without `tool_choice` flow unchanged through ADR-027 pipeline
**Given** a chat-completions request with no `tool_choice` field and the bridge mechanism active
**When** the framework processes the request
**Then** the bridge mechanism does not fire; the request flows through ADR-027's three-stage pipeline; the response carries standard ADR-032 honest response labeling (without `tool_choice_handling` field). Existing client behavior is unaffected.

### Preservation: ADR-001 + ADR-011 ReAct execution model remains operative architecturally
**Given** ADR-001 + ADR-011 as architectural commitments to the ReAct execution model + ADR-030's deferred disposition (i) implementation
**When** future cycles consider re-introducing ReAct-loop components for non-chat-completions surfaces (or implementing disposition (i) full `tool_choice` honoring via server-side dispatch path)
**Then** ADR-001 + ADR-011's commitments remain operative as architectural options. ADR-030's deferred disposition (i) does not deprecate the ReAct model; it defers the full implementation while shipping the bridge in Cycle 7 BUILD.

---

## Feature: Latency and Streaming (ADR-031)

### Scenario: Streaming-default request streams synthesizer output token-by-token
**Given** an operator-deployed serve with ADR-027 + ADR-031 active and a chat-completions request from a streaming-default client (OpenCode, Aider, Cline-when-tuned) with `stream: true`
**When** the framework executes the pipeline
**Then** the response is delivered as SSE chunks per OpenAI chat-completions semantics. The routing-planner + dispatch stages run synchronously upstream of the synthesizer (no streaming during plan or dispatch); the synthesizer's output streams token-by-token to the client as the synthesizer LLM generates. First-token latency to the client is approximately synthesizer-first-token time (~2-3s at qwen3:8b), not the full pipeline end-to-end latency.

### Scenario: Non-streaming request returns full response after pipeline completes
**Given** an operator-deployed serve and a chat-completions request with `stream: false` (or no stream field)
**When** the framework executes the pipeline
**Then** the response is delivered as a single complete chat-completion object after the synthesizer completes. End-to-end latency is the full pipeline floor (~36s single-step at qwen3:8b per Spike ε).

### Scenario: Tier escalation policy fires on Rule 4 rounding-drift Reflect verdict
**Given** an operator-deployed serve with ADR-029's rounding-drift mitigation playbook active (Calibration Gate Reflect-trigger criterion for Rule 4) and a synthesizer response where runtime fidelity check detects rounding drift exceeding threshold
**When** the Calibration Gate emits Reflect verdict
**Then** the Tier-Escalation Router per ADR-015 escalates the synthesizer to the operator-configured higher-tier model for the retry; the retry's response is checked again; the audit per ADR-018 records the escalation event.

### Scenario: Tier escalation policy fires on Rule 1 fabrication signal
**Given** an operator-deployed serve and a synthesizer response where post-hoc audit detects substantive content not sourced from DISPATCH RESULTS
**When** the Calibration Gate emits Reflect verdict on Rule 1 fabrication signal
**Then** the Tier-Escalation Router escalates per ADR-015; the retry's response is checked; recurring Rule 1 Reflect verdicts feed the Tier-Router Audit drift criteria per ADR-018.

### Preservation: Existing Calibration Gate + Tier-Router + Audit infrastructure operates unchanged
**Given** ADR-007 + ADR-014 + ADR-015 + ADR-018-shipped Calibration Gate + Tier-Escalation Router + Audit Dispatch infrastructure
**When** ADR-031's three new Calibration Gate Reflect-trigger criteria (Rule 5 framing absence; Rule 4 rounding-drift; Rule 1 fabrication) are added
**Then** the existing infrastructure operates unchanged. The new criteria are additive — they feed the same Reflect verdict; the same Tier-Escalation Router consumes the verdict; the same Audit Dispatch tracks drift. No structural changes to ADR-007/014/015/018.

### Preservation: ADR-022 system-prompt amendment does not affect chat-completions surface latency
**Given** ADR-022's system-prompt amendment (under ADR-027 update: structurally moot for chat-completions; operative for any future surface adopting `OrchestratorRuntime`)
**When** ADR-031's latency floor applies to chat-completions requests under ADR-027
**Then** the system-prompt amendment's token cost has no effect on chat-completions latency (the amendment is not loaded on this surface). The ~36s floor is a function of routing-planner + dispatched capability ensemble + response-synthesizer latencies, not orchestrator-LLM context.

---

## Feature: Honest Response Labeling and Capability-List Discovery (ADR-032)

### Scenario: Dispatch response declares served_by:ensemble:<name> at all three layers
**Given** an operator-deployed serve with ADR-032's honest response labeling active and a chat-completions request producing `action: "dispatch"` with successful dispatch
**When** the response is returned
**Then** the response carries: header `X-LLM-Orc-Served-By: ensemble:<name>` (or the framework's chosen header family); body metadata `metadata.served_by: "ensemble:<name>"`; no content-layer Rule 5 framing (the response content carries the dispatched ensemble's deliverable via the synthesizer, not a direct-completion declaration).

### Scenario: Direct-completion response declares served_by:direct at all three layers + Rule 5 framing
**Given** an operator-deployed serve and a chat-completions request producing `action: "direct"`
**When** the response is returned
**Then** the response carries: header `X-LLM-Orc-Served-By: direct`; body metadata `metadata.served_by: "direct"`; synthesizer's content includes Rule 5 framing per ADR-029. Population A's degradation signal (configuration dishonesty per Cline #10551 + OpenCode #20859) is structurally prevented at three layers.

### Scenario: Dispatch failure response declares served_by:direct_fallback with failure type
**Given** an operator-deployed serve and a chat-completions request where dispatch was planned but failed irrecoverably (infrastructure error, schema-non-conformance after Calibration Gate retries)
**When** the framework falls back to direct-completion via the synthesizer
**Then** the response carries: header `X-LLM-Orc-Served-By: direct_fallback`; body metadata `metadata.served_by: "direct_fallback"; metadata.dispatch_failed: "<failure-type>"`; synthesizer's content includes Rule 5 framing acknowledging the fallback path.

### Scenario: `/v1/models` advertises capability ensembles with topaz_skill metadata
**Given** an operator-deployed serve with capability-list discovery via `/v1/models` extension per ADR-032 candidate surface (a) and a framework loaded-ensemble registry containing capability ensembles (e.g., `code-generator`, `text-summarizer`, `claim-extractor`)
**When** a client calls `GET /v1/models`
**Then** the response includes the capability ensembles as model entries, each with a capability-marker field distinguishing them from underlying models (e.g., `type: "ensemble"`) and their `topaz_skill` metadata. The endpoint reflects the framework's live registry.

### Scenario: Capability list updates reflect ensemble add/remove events
**Given** an operator-deployed serve with the capability-list endpoint active and a framework that re-loads its ensemble registry after the operator adds or removes an ensemble YAML
**When** the registry is reloaded and a client calls `GET /v1/models` (or the chosen discovery endpoint)
**Then** the response reflects the new registry contents; added ensembles appear, removed ensembles do not. The discovery surface is dynamic, not statically configured.

### Scenario: action=direct emits direct_completion_fallback event
**Given** an operator-deployed serve with ADR-023 observability event routing + ADR-032 degradation signaling event types active and a chat-completions request producing `action: "direct"`
**When** the response completes
**Then** the framework emits a `direct_completion_fallback` event on the dispatch event substrate with fields: request shape category (NL prose, script-shaped, mixed, ambiguous); routing-planner rationale; detected client population signals (if available). The event routes to operator-terminal and orchestrator-context destinations per ADR-023.

### Scenario: direct_completion_rate rolling metric is computable from emitted events
**Given** an operator-deployed serve with degradation signaling events emitted over time and an operator dashboard / log consumer that aggregates events
**When** the consumer computes the rolling rate over a sliding window (e.g., 1-hour or 1-day window)
**Then** the `direct_completion_rate` metric is computable as `count(direct_completion_fallback events) / count(all chat-completions requests)` over the window. High rates surface for operator action (capability library expansion or planner tuning).

### Scenario: Population B advisory present as metadata on direct-completion responses
**Given** an operator-deployed serve and a chat-completions request producing `action: "direct"` where the request content shape matches Population-B-style patterns (script-shaped, programmatic content, explicit naming attempts the planner couldn't bind)
**When** the response is returned
**Then** the response includes `metadata.population_b_advisory: "<advisory-content>"` pointing toward `llm-orc invoke` or the direct ensemble HTTP API. The advisory is at the metadata layer (safe-to-send-universally); Population A clients that don't surface metadata are unaffected.

### Preservation: OpenAI chat-completions API contract unchanged
**Given** an operator-deployed serve with ADR-032 honest response labeling active and a Population A client that does not surface custom response headers or metadata
**When** the client receives a chat-completion response
**Then** the response's `message.content`, `finish_reason`, `model`, `choices` fields conform to OpenAI chat-completions semantics. The honest-response-labeling additions are at non-content layers (headers + metadata); clients ignoring these layers see standard OpenAI-compatible responses.

### Preservation: ADR-023 observability event routing unchanged in mechanism
**Given** ADR-023-shipped observability event substrate routing typed events to operator-terminal and orchestrator-context destinations
**When** ADR-032 adds new event types (`direct_completion_fallback`, `direct_completion_rate` rolling metric source events)
**Then** the routing mechanism is unchanged; the new event types are additive — they route through the same infrastructure to the same destinations. ADR-023's destination behavior tables are extended (not modified) to specify the new event types' default routing.

---

## Feature: Capability Matching from Request Content Alone (ADR-026 / AS-10)

### Scenario: Routing decision uses only request body + capability list
**Given** an operator-deployed serve with AS-10 (per ADR-026) active and a chat-completions request with custom HTTP headers (e.g., `X-Skill-Framework: rdd`, `X-Capability-Hint: code-generator`)
**When** the framework parses the request and invokes the routing-planner
**Then** the planner receives only the request body content (`messages[]`, `model`, optional `tools[]`, optional OpenAI-protocol-native `tool_choice`) and the framework's capability list. Custom headers are not consumed for routing. The transparent-endpoint promise is preserved: the routing decision is identical regardless of which Population A client sent the request.

### Scenario: Population B accommodation via alternative surfaces, not via chat-completions opt-in
**Given** an operator-deployed serve and a Population B client (a developer/script client) needing capability dispatch
**When** the client requests a capability
**Then** the client uses `llm-orc invoke` CLI or the direct ensemble HTTP API where explicit capability identifiers are the normal mode. The chat-completions surface does not provide a Population-B-only opt-in path (AS-10 forbids client-side opt-in to llm-orc-specific mechanisms). The structured advisory in chat-completions responses per ADR-032 informs Population B clients of the alternative surface availability.

### Scenario: OpenAI-protocol-native `tool_choice` is permitted (not a llm-orc-specific opt-in)
**Given** an operator-deployed serve with ADR-030's bridge mechanism active and a chat-completions request with `tool_choice: {"name": "code-generator"}`
**When** the framework parses the request
**Then** the `tool_choice` field is observed (per ADR-030's bridge mechanism). Per AS-10 §Operational consequences, `tool_choice` is permitted because it is an OpenAI-protocol-native field; sending `tool_choice` is not opting into a llm-orc-specific mechanism. The framework's interpretation of `tool_choice` (disposition (i) full honoring under follow-on cycle; bridge mechanism in Cycle 7 BUILD) is the framework's design choice per ADR-030, not a client-side opt-in.

### Preservation: `llm-orc invoke` CLI accepts explicit capability identifiers per ADR-021's original contract
**Given** AS-10 scoped to the agentic-serving chat-completions surface and the `llm-orc invoke` CLI accepting explicit ensemble names per ADR-021
**When** an operator uses `llm-orc invoke code-generator --input "..."` on the CLI
**Then** the CLI accepts the explicit identifier; this is consistent with ADR-021's preferred dispatch shape (explicit ensemble naming via `OrchestraService.find_ensemble_by_name`). AS-10 does not govern the CLI surface; the CLI's normal mode of operation is explicit identifiers.

### Preservation: ADR-019 skill-framework-agnostic commitment unchanged
**Given** ADR-019's skill-framework-agnostic library shape (skill orchestration is client-side; operation-named library entries; no methodology-shaped ensembles in the library)
**When** ADR-026 codifies AS-10 (capability matching from request content alone)
**Then** the skill-framework-agnostic commitment is preserved structurally. AS-10 prevents the framework from learning skill-framework identifiers via client-side opt-in; this aligns with ADR-019's commitment that the orchestrator routes by capability without knowing which skill framework is composing against it.

---

## Cycle 7 loop-back Acceptance Criteria Table additions (per ADR-033 + ADR-034)

The following loop-back acceptance criteria are emergent, aggregate, or specify an integration layer individual scenarios stub out. BUILD Step 5.5 verifies each at its specified layer. The loop-back covers the multi-turn tool-driven surface (layer-A loop-driver + client-tool-action terminal); ADR-027's single-turn pipeline surface is unchanged.

| Criterion | Specified layer | Verification method | Layer-match check |
|-----------|----------------|--------------------|-----|
| A tool-driven client driven against agentic-serving gets a parity session (executes + observes its own tool calls; permission gate, diff, tool-result feedback intact) | Live end-to-end (real client, e.g. OpenCode, against a real loop-driver + terminal + ensemble) | Composes "Surface engages loop-driver when client tools present" + "Loop-driver delegates per-turn generation to a single ensemble" + "Terminal emits finish_reason=tool_calls the client executes" + "Surface consumes the tool-result follow-up and continues" with a real client and a real ensemble (no stubs at the client boundary) | no — unit/integration scenarios exercise each stage with stubs; parity is only observable when a real client executes the synthesized tool calls and the loop closes. This is the Spike π/ρ/σ verdict, to be re-confirmed against the built (not stand-in) terminal |
| Grounded driving holds: no turn commits an action to an unobserved value | Integration (multi-turn loop over an un-batchable task where a later step depends on an earlier step's observed tool result) | Composes "Single-action-per-turn enforcement truncates a driver batch" + "Grounded-carry guard: a tool-call argument referencing an unobserved output is not dispatched" with the Spike τ un-batchable probe shape (a value present only in a prior tool result) | no — individual scenarios exercise enforcement and the guard in isolation; the grounded-carry property is emergent over a multi-turn un-batchable chain. Axis 2 (long-horizon drift) is BUILD/PLAY, not closable by a short integration test (ADR-033 Conditional Acceptance) |
| Artifact-bridge delivers ensemble output to the client with fidelity, including large/complex deliverables | Integration (loop-driver → ensemble with output_substrate:artifact → SessionArtifactStore → terminal → tool-call content) | Composes "Loop-driver delegates write generation to a substrate-routed ensemble" + "Artifact-bridge reads the deliverable from SessionArtifactStore and marshals it into tool-call content" + a large-file case (deliverable beyond the trivially-small spike content) | no — spike evidence is trivially-small content only; the fidelity FC requires a large-deliverable integration test BUILD must add |
| Long-horizon driver coherence (axis 2) | PLAY / first-deployment (a real multi-step, many-turn task: e.g. drive a short RDD-style or multi-file build session) | PLAY-phase experiential run against the built loop-driver + terminal; observe whether the trajectory holds or drifts/accumulates error across many turns | no — no synthetic test reaches axis 2; this is the recorded load-bearing risk (ADR-033 §Decision ¶5), validated in PLAY/first-deployment, ADR-097 Conditional Acceptance as the backstop |

These Layer-match "no" entries are the table working as designed: they mark where BUILD Step 5.5 adds integration/large-deliverable tests and where PLAY/first-deployment is the natural verification surface for the axis-2 risk.

---

## Feature: Layer-A Loop-Driver and Surface-Mode Discrimination (ADR-033)

### Scenario: Surface engages the loop-driver when the request carries client tools
**Given** an operator-deployed serve with ADR-033 active and a chat-completions request that carries client `tools[]` (a tool-driven client, e.g. OpenCode's build agent declaring `write`/`edit`/`bash`/`read`) with `tool_choice: "auto"`
**When** the chat-completions handler processes the request
**Then** the surface engages the layer-A loop-driver (not ADR-027's single-turn `plan → dispatch → synthesize` pipeline as the terminal). The discriminator is the presence of client tools in the request.

### Scenario: Surface uses the single-turn pipeline when no client tools are present
**Given** an operator-deployed serve with ADR-033 active and a chat-completions request carrying no client `tools[]` (a non-agentic answer-a-question request)
**When** the handler processes the request
**Then** the surface routes through ADR-027's `plan → dispatch → synthesize` single-turn pipeline and returns a synthesized text response; the layer-A loop-driver is not engaged.

### Scenario: Loop-driver delegates per-turn generation to a single capability ensemble (callee, not the pipeline)
**Given** the loop-driver engaged on a tool-driven request, and a turn whose action requires generated content (e.g. file content for a `write`)
**When** the loop-driver produces the content for that turn's tool call
**Then** generation is delegated to a single capability ensemble invocation (the callee), not routed through the full `plan → dispatch → synthesize` pipeline (no per-turn routing-planner stage, no per-turn response-synthesizer stage). Refutable: a per-turn generation that invokes the routing-planner + synthesizer stages violates ADR-033's callee FC.

### Scenario: Single-action-per-turn enforcement truncates a driver batch
**Given** the loop-driver engaged, and a turn in which the driver proposes more than one client tool call at once (a batch)
**When** the framework dispatches the turn's tool calls
**Then** the framework dispatches at most one client tool call, returns its result to the loop-driver, and forces re-planning before any subsequent action; the additional proposed tool calls in the same turn are not dispatched until the first call's result is observed. Refutable: a turn that dispatches two client tool calls before returning the first's result violates the single-step FC. (Enforcement technique — batch-truncation per Spike τ′ evidence, or a re-planning prompt, or a one-tool `tool_choice` constraint — is ARCHITECT/BUILD selection; the FC constrains the observable behavior, not the technique.)

### Scenario: Grounded carry — an action depending on a prior observed result uses the observed value
**Given** the loop-driver engaged on an un-batchable task where a turn's correct action depends on a value present only in a prior turn's observed tool result (the Spike τ probe shape: a random value printed to stdout, not in any file, not recomputable)
**When** the loop-driver, under single-action-per-turn enforcement, observes the prior tool result and then decides the dependent action
**Then** the dependent tool-call argument uses the observed value, not a placeholder or fabricated value. Refutable: a tool-call argument containing an unresolved template reference to an unobserved output (the Spike τ `${bash_output}` failure signature) or a fabricated stand-in value violates the grounded-carry FC.

### Scenario: Loop-driver finishes with a text completion when no further action is needed
**Given** the loop-driver engaged and a turn at which the task is complete (no further tool call is warranted)
**When** the loop-driver decides the turn's action
**Then** the surface returns a text completion (`finish_reason: "stop"`), closing the loop; a tool-capable client that asked for a plain answer is served correctly via this path (the surface-mode discriminator engaging the driver when tools are present is safe because the driver can finish with text).

### Preservation: the single-turn pipeline surface (ADR-027) is unchanged for non-tool requests
**Given** ADR-027's `plan → dispatch → synthesize` pipeline serving non-tool-driven chat-completions requests
**When** ADR-033 adds the loop-driver for tool-driven requests
**Then** a request carrying no client tools behaves exactly as before ADR-033: routed through the single-turn pipeline, synthesized text response, response-synthesizer strict-fidelity rules (ADR-029) applied. ADR-033 adds a surface-mode branch; it does not change the non-tool path.

### Preservation: AS-10 capability matching from request content alone is unchanged
**Given** AS-10 (capability matching from request content alone; no client-side opt-in) and Spike ρ's reaffirmation that the planner routes on request content indifferent to declared client tools
**When** the loop-driver delegates per-turn generation to a capability ensemble
**Then** the per-turn ensemble selection is a function of the turn's task content, not of any client-supplied routing signal; the presence of client `tools[]` is used only as the surface-mode discriminator, not as a capability-routing input. Refutable: routing the per-turn generation by a client-supplied capability identifier (not the task content) violates AS-10.

---

## Feature: Client-Tool-Action Terminal and Artifact-Bridge (ADR-034)

### Scenario: Terminal emits finish_reason=tool_calls carrying the deliverable
**Given** the loop-driver decides a turn's action is to apply work to the client (e.g. write a file) and the generating ensemble has produced a deliverable
**When** the terminal builds the response
**Then** the surface emits a streamed assistant response with `finish_reason: "tool_calls"` carrying the appropriate client tool call (e.g. `write({filePath, content})`) in the OpenAI streaming tool-call delta shape (`delta.tool_calls[].function.arguments` fragments). Refutable: such a turn returning only `ContentDelta` + `Completion` (a text-only terminal) violates ADR-034's tool-call-terminal FC.

### Scenario (integration): Artifact-bridge reads the substrate-routed deliverable and marshals it into tool-call content
**Given** a capability ensemble with `output_substrate: artifact` (per ADR-025) whose deliverable routed to the server-side `SessionArtifactStore` (`envelope.primary` is a summary + `ArtifactReference`, not inline content)
**When** the terminal builds the `write` tool call for that deliverable
**Then** the artifact-bridge reads the deliverable from the `SessionArtifactStore` (via a `read_deliverable(reference)` accessor BUILD adds — the store currently has `write_deliverable` only) and marshals the artifact content into the tool-call `content` argument; the content equals the stored deliverable, not a summary or paraphrase. Refutable: a `write` whose content is `envelope.primary`'s summary rather than the artifact content violates the artifact-bridge fidelity FC.

### Scenario (integration): Inline-substrate deliverable skips the bridge step
**Given** an inline-response ensemble (`output_substrate: inline`) whose deliverable is in `envelope.primary` directly
**When** the terminal builds the tool call
**Then** the deliverable is read from `envelope.primary` and the artifact-store read is a no-op; the marshalled content equals the inline deliverable.

### Scenario: Surface consumes the tool-result follow-up and continues the loop
**Given** the surface emitted a `write` tool call, the client executed it locally and returned the result in a follow-up request carrying a `role: "tool"` message
**When** the surface processes the follow-up
**Then** the surface routes the tool result to the loop-driver (the follow-up's `role: "tool"` message is not dropped), and the loop-driver decides the next action or finishes. Refutable: a surface that ignores the `role: "tool"` follow-up, or whose request extraction drops tool messages (the current `_extract_request` reads only the last user message), violates the loop-participation FC.

### Scenario: Terminal never writes to the client's filesystem directly
**Given** the loop-driver decides to apply work to the client
**When** the deliverable reaches the client
**Then** the deliverable reaches the client only via a client-executed tool call, never via a server-side write to a client workspace path (the Spike π Phase A rejected shape). Refutable: any server-side write to a client filesystem path violates the no-server-side-write FC. (Satisfied by absence in current code; the FC guards against future re-introduction.)

### Preservation: the SSE formatter's existing ClientToolCall handling is reused, not rebuilt
**Given** the SSE formatter already formats `ClientToolCall` chunks correctly (`sse_format.py`) and the `OrchestratorChunk` union still includes `ClientToolCall` (the `0a7a822` removal took the handler emission, not the type or formatter)
**When** ADR-034 re-introduces tool-call emission on the terminal
**Then** the terminal yields `ClientToolCall` chunks that the existing SSE formatter consumes unchanged; no streaming-formatter changes are required. The re-introduction restores the `ClientToolCall` import, the `tool_calls` field on `_NonStreamingResult`, the `isinstance(chunk, ClientToolCall)` branch in `_collect_non_streaming`, and the `tool_calls` shaping in `_build_completion_body` (the four pieces `0a7a822` removed).

### Preservation: ADR-025 artifact-as-substrate routing is unchanged
**Given** ADR-025's substrate routing (capability deliverables route to the `SessionArtifactStore` by design)
**When** the artifact-bridge reads a deliverable to marshal it into a tool call
**Then** the bridge adds a read-and-marshal step; it does not change how deliverables are routed to the store, and it does not summarize the deliverable (AS-7 result-summarization is unaffected — the bridge marshals content, it does not summarize). The store remains the canonical deliverable location until the bridge marshals a copy into the client surface.

---

## Feature: Loop-back Structural Debt Remediation (from conformance-scan-cycle-7-loopback-decide)

### Scenario (refactor, before BUILD): remove the stale ClientToolCall docstring comment
**Given** the conformance scan found a docstring comment at `v1_chat_completions.py:581–583` stating that `ClientToolCall` chunks "are not part of this surface's vocabulary under ADR-027" — a comment that now contradicts ADR-034
**When** the loop-back BUILD begins (before the tool-call terminal is built)
**Then** the misleading comment is removed as a `refactor:` commit so BUILD implementors are not misled; this is the one refactor-now item from the loop-back conformance scan (the other 11 findings are BUILD-work or ARCHITECT-deferral, tracked as the loop-back BUILD load).

---

## Feature: Client-Tool Deliverable Form Contract (ADR-035, Finding D)

*Conformance disposition (no formal scan): ADR-035 is a new decision; no shipped code implements the boundary directive, so there is no drift to scan — only unbuilt work + the D1 extraction bug Spike φ characterized in-context (`_extract_synthesizer_text` at `orchestrator_tool_dispatch.py:1373/1796` stores `json.dumps(raw_result)` for multi-agent ensembles). The "debt" is captured by the D1 scenarios below.*

### Scenario: the marshalling boundary injects a destination-keyed form directive (write)
**Given** the loop-driver decides a turn's action is a `write` and delegates generation to a capability ensemble via callee `invoke_ensemble` (ADR-033)
**When** the dispatch input is composed at the marshalling boundary (loop-driver / Client-Tool-Action Terminal)
**Then** the dispatch input carries a `write`-keyed bare-output directive (produce only the bare file bytes — no markdown fences, no prose, no example block). Refutable: a client-tool dispatch whose input carries no destination-keyed form directive violates the boundary-directive FC.

### Scenario: the directive is keyed to the destination tool (bash)
**Given** the loop-driver decides a turn's action is a `bash` command
**When** the dispatch input is composed
**Then** the injected directive is a bare-command directive (produce only the exact command), not the `write` bare-file-bytes directive. Refutable: emitting the `write`-form directive for a `bash` destination violates destination-keying.

### Scenario: capability ensembles stay destination-agnostic
**Given** a capability ensemble (e.g. `code-generator`) dispatched for a client-tool deliverable
**When** the dispatch occurs
**Then** the form directive lives in the dispatch input only; no ensemble YAML field (`system_prompt`, `default_task`, `output_schema`, or a baked-in `submit_file` tool) couples the ensemble to file-production. Refutable: a `submit_file`-shaped tool or a destination-form instruction baked into the ensemble's static config violates the destination-agnostic FC (ADR-025 reusability principle).

### Scenario (integration): a delegated `write` deliverable is bare client-tool content (the Finding D refutation)
**Given** `code-generator` dispatched with a `write`-keyed bare-output directive, producing a deliverable that the artifact-bridge (ADR-034) marshals into a `write` tool call
**When** the client receives the `write`
**Then** the `write` content is bare file content — directly writable and runnable, with no markdown fences, no prose preamble, and no "Example Usage" block. Refutable: a `write` whose content is the raw `{"results": {...}}` envelope (the WP-LB-G failure) or carries fences / prose / an example block violates the form contract.

### Scenario: one dispatch → one client-tool deliverable (granularity invariant)
**Given** a task that produces multiple files (e.g. a module plus its test file)
**When** the loop-driver serves it
**Then** the loop-driver decomposes the work across turns — one `write` (one dispatch → one deliverable) per turn — rather than one dispatch producing N files. Refutable: a single dispatch whose deliverable encodes multiple files (the Spike χ-P6 `filename\ncontent` shape) violates the granularity invariant. *(Across-turn decomposition under a real loop-driver is BUILD/PLAY work per ADR-033 §6b; this scenario states the invariant the BUILD must hold.)*

### Scenario (defense-in-depth, optional): the bridge backstop normalizes a stray enclosing fence
**Given** a deliverable that — despite the directive — still arrives wrapped in a single enclosing code fence
**When** the artifact-bridge marshals it
**Then** an optional conservative backstop strips the single enclosing fence before marshalling. This is a defense-in-depth net, not the contract (the directive is the primary mechanism); the backstop does not attempt heuristic extraction from multi-fence output (Spike χ F-χ.1 — that path is fragile). Refutable: a backstop that attempts first/largest-fence extraction across multiple fences violates the backstop's conservative scope.

### Scenario (integration, D1 — BUILD fix shaped to ADR-035): deliverable extraction stores the terminal agent's output, not the raw dict
**Given** a multi-agent capability ensemble (`code-generator`: coder → critic → synthesizer) whose execution result is a per-agent results dict
**When** the deliverable is resolved (LB-4 resolved executor-side at BUILD: `resolve_deliverable` in `results_processor.py` populates the `ExecutionResult.deliverable` contract where `depends_on` is known; the substrate write reads the contract)
**Then** the stored deliverable is the terminal agent's output (the agent no other agent depends on — here `synthesizer`), never `json.dumps(raw_result)`. Refutable: a stored deliverable equal to `{"results": {coder, critic, synthesizer}, …}` (the Finding D raw envelope) violates the D1 extraction fix. *(BUILD note: the legacy `synthesis` field was excised entirely — the deliverable contract is the single output notion.)*

### Scenario (D1): terminal-node failure falls back to the last successful agent
**Given** a multi-agent ensemble whose terminal agent (synthesizer) failed or timed out (`response: null`, `status: failed` — observed in Spike χ-P1)
**When** the deliverable is resolved
**Then** resolution falls back to the last *successful* agent's output walking declaration order backward, not the failed node's null and not the raw dict. Refutable: storing null, the failed node, or `json.dumps(raw_result)` when a successful upstream agent exists violates the fallback. *(BUILD note: any successful agent satisfies the refutable contract; which upstream agent is "the deliverable" is role semantics no order rule can know — the fallback is best-effort degraded output, with the client's permission-gate diff as the recorded backstop. The original "(e.g. coder)" illustration assumed a linear DAG the shipped YAML did not have; the critic now `depends_on: [coder]`, making the chain genuinely linear.)*

### Preservation: the artifact-bridge fidelity FC (ADR-034) is unchanged
**Given** ADR-034's artifact-bridge marshals exactly what is stored (fidelity FC)
**When** ADR-035's form contract is in effect
**Then** the bridge still marshals the stored deliverable faithfully — ADR-035 changes what the ensemble *produces and stores* (bare form), not the bridge's faithfulness; the bridge gains no summarization or transformation step. Refutable: a bridge that transforms or summarizes the deliverable to "fix" its form violates the fidelity FC (the fix belongs upstream, at production).

### Preservation: ADR-025 substrate routing is unchanged
**Given** a capability ensemble with `output_substrate: artifact`
**When** the boundary directive is injected into its dispatch
**Then** the deliverable still routes to the `SessionArtifactStore` exactly as before; only the produced content's *form* changes, not the routing or the store. Refutable: a change to substrate routing triggered by the form directive violates this preservation.

### Preservation: inter-ensemble composition keeps ADR-024's advisory-schema stance (carve-out boundary)
**Given** two capability ensembles composed (an upstream deliverable consumed by a downstream ensemble — NOT a client-tool deliverable)
**When** the upstream produces output
**Then** ADR-024's advisory `output_schema` stance still governs — ADR-035 refines only the client-tool-deliverable path. Refutable: applying the boundary bare-output form-directive to a non-client-tool composition dispatch violates the ADR-024/ADR-035 carve-out scope.

### Preservation: the single-turn answer-a-question surface (ADR-027) is untouched
**Given** a non-tool-driven request (no client tools in the request → dispatch-pipeline path, WP-LB-A discriminator)
**When** it is served
**Then** ADR-035's boundary directive does not engage (it is specific to client-tool deliverables on the tool-driven surface); the text pipeline's behavior is unchanged.

### Cycle Acceptance Criteria (Finding D)

| Criterion | Specified layer | Verification method | Layer-match check |
|---|---|---|---|
| A delegated client-tool deliverable lands as usable bare content (the north-star "delegate work, apply locally" loop produces a runnable file) | Real OpenCode round-trip (real `llm-orc serve` + real client + local Ollama, $0) | The real-serving-surface integration test (TS-12/13) + the $0 real-OpenCode smoke test, exercised with the boundary directive in effect; the per-component scenarios above stub the client | **closed at the specified layer 2026-06-03** — WP-LB-H smoke run 3: real OpenCode executed a delegated, form-contracted `write`; `matrix_utils.py` landed bare and parsing (research log §"WP-LB-H built + validated"). Residual: delegation *reliability* under the client system prompt is Finding E (loop-back #3 → DECIDE); trajectory-scale compliance stays the ADR-097 PLAY target |
| The produced deliverable is bare client-tool content across deliverable types | API-boundary behavior, then real-client | The "bare client-tool content" integration scenario above (single deliverable); breadth across types is grounded n=4 (Spike χ.2) but trajectory/escalated-tier breadth is PLAY (ADR-033 §6b axis-2) | partial — single-dispatch grounded; sustained-trajectory compliance is a PLAY target (Conditional Acceptance, ADR-097) |
| D1: multi-agent deliverable extraction stores the terminal/last-successful agent's output | Substrate-write layer (unit/integration) | The two D1 scenarios above, verified against the real `SessionArtifactStore` | yes (1:1 at the substrate-write layer) |

## Feature: Delegation-Decision Mechanism — User-Turn Guidance Composition (ADR-036, Finding E)

*Conformance disposition (`housekeeping/audits/conformance-scan-cycle-7-loopback3-decide.md`): 4 findings, all expected-or-mild — F-1 the system-message composition this feature replaces (refactor-now, the primary WP); F-2/F-3 the instrumentation gap (TurnDecision sink + classifier graduation, co-dependent BUILD work); F-4 a one-line docstring drift (deferred, bundled below).*

### Scenario: first-turn guidance composes into the user turn, never as a system message
**Given** a tool-driven request (client `tools[]` present) entering the loop-driver on its first turn
**When** the seat-filler request is composed
**Then** the delegation guidance is attached to the user task in the user-turn region and the composed request carries no framework-authored system message — the client's system prompt stands alone in the system region. Refutable: a composed seat-filler request with `_DELEGATION_GUIDANCE` in a `{"role": "system"}` message violates the directive-in-user-turn FC.

### Scenario: trailing turns carry the guidance as a standalone trailing user-role message (C3 form)
**Given** a multi-turn session whose conversation tail is a tool result (assistant tool-call → tool-result pairs)
**When** the next seat-filler request is composed
**Then** the guidance is appended as its own `{"role": "user"}` message after the conversation tail, without mutating any client-authored message content. Refutable: guidance merged into a client-authored message, or absent from the composed trailing-turn request, violates the FC. *(C1/C2 attachment forms are equally measured (5/5 each) — using them is implementation choice, not a violation; the structural property is user-role placement.)*

### Scenario: the composition is invisible to the client
**Given** any tool-driven session with the V3 composition in effect
**When** the client inspects everything it receives (responses, streamed chunks, tool calls)
**Then** no client-visible surface carries the delegation guidance — the composition exists only on the framework ↔ seat-filler hop. Refutable: guidance text appearing in any client-visible response violates this.

### Scenario (integration): a generation-shaped turn under V3 composition delegates
**Given** a generation-shaped request (e.g. "Create a file called csv_helper.py that loads a CSV and computes column means") composed under V3 against the validated seat-filler profile (qwen3:14b)
**When** the seat-filler decides the turn
**Then** the decision is an `invoke_ensemble` call naming a registered capability ensemble, with a substantive input brief and a filePath — observable as a `dispatch start` in the serve log and a `TurnDecision` with `delegated_ensemble` set. Refutable: a direct `write` with inline-generated content on a generation-shaped turn is a non-delegation (the Finding E signature). *(Replay layer: 55/55 measured. The real-client layer is the acceptance criterion below — per the WP-A scar, a passing-looking run must verify delegation actually fired.)*

### Scenario: TurnDecision events surface to the operator (instrumentation numerator)
**Given** a tool-driven session emitting `TurnDecision` events through the Dispatch Event Substrate
**When** the operator event sink consumes the substrate's events
**Then** `TurnDecision` no longer falls through silently — at minimum `action`, `delegated_ensemble`, and the turn correlation surface in the operator-visible log. Refutable: a `TurnDecision` event consumed by no sink branch violates the delegation-rate measurability FC (conformance F-2).

### Scenario: the generation-shaped classifier graduates into the package (instrumentation denominator)
**Given** the spike-validated classification rule (generation verb × content object × capability domain × observed-carry exclusions)
**When** it is relocated from `scratch/spike-psi-delegation-rate/psi4a_prefilter.py` into `src/llm_orc/agentic/`
**Then** the graduated classifier reproduces the spike's labeled-set results — 0 misclassifications on the 12 clear cases — as a package regression test, and `delegation_rate` is computable from events alone (classifier denominator × `TurnDecision.delegated_ensemble` numerator). Refutable: a deployment from which the rate cannot be computed without log archaeology violates the measurability FC (conformance F-3).

### Scenario: boundary turns are excluded from the denominator, not guessed
**Given** a repair-shaped turn ("Fix the bug in string_utils.py where count_vowels misses uppercase vowels") or a turn whose content domain has no registered capability
**When** the classifier processes it
**Then** the turn is recorded as boundary-excluded — outside the delegation-rate denominator — and the boundary-excluded share is itself observable (the operational signal that denominator coverage is degrading). Refutable: a boundary turn counted in the denominator, or an unobservable exclusion count, violates Decision 3's measurement-integrity choice.

### Scenario (refactor, before BUILD): the single_step_enforcer docstring stops calling tool_choice BUILD-tunable
**Given** `single_step_enforcer.py:17–19` frames a one-tool `tool_choice` constraint as a BUILD-tunable enforcement candidate
**When** the docstring is updated (conformance F-4; `refactor:` commit)
**Then** it records the family as empirically closed (Spike ψ.3 added the third negative: Ollama+qwen3 silently ignores forcing), leaving the re-planning prompt as the only remaining untested candidate.

### Preservation: carry-shaped turns still select client tools (no over-delegation)
**Given** read-shaped, command-shaped, and literal-write turns under the V3 composition
**When** the seat-filler decides each turn
**Then** the decisions remain `read`/`glob`, `bash`, and `write` respectively — never `invoke_ensemble`. Refutable: a carry-shaped turn that delegates violates the carry-side preservation FC. *(Measured 0/15 false delegations, ψ′ Arm B.)*

### Preservation: grounded carry stays verbatim under the new composition (FC-45)
**Given** a literal-payload write task ("Write exactly this to notes.txt: …") under V3 composition
**When** the seat-filler emits the `write`
**Then** the literal payload appears in the `write` arguments byte-for-byte — the guidance relocation does not cause paraphrase or regeneration of carried values. Refutable: a paraphrased payload with the correct tool name is a grounded-carry violation the tool-name dimension alone cannot see (the methods-review P1-A lesson). *(Measured verbatim 5/5, ψ′ B3.)*

### Preservation: the ADR-035 form directive composition is untouched
**Given** a delegated generation turn whose callee dispatch input carries the destination-keyed form directive (`compose_form_directive`, FC-53/54)
**When** the seat-filler guidance relocates to the user turn
**Then** the callee dispatch composition is unchanged — the two directives live on different hops (guidance: framework → seat-filler; form directive: framework → callee ensemble) and neither moves. Refutable: a callee dispatch missing its form directive after the guidance relocation lands violates this.

### Preservation: the single-turn answer-a-question surface (ADR-027) is untouched
**Given** a non-tool-driven request (no client tools → dispatch-pipeline path)
**When** it is served
**Then** the V3 composition does not engage (it is specific to the seat-filler hop on the tool-driven surface); the pipeline's message composition is unchanged.

### Cycle Acceptance Criteria (Finding E)

| Criterion | Specified layer | Verification method | Layer-match check |
|---|---|---|---|
| Delegation fires reliably on generation-shaped turns under the real client (the north-star "work delegated to ensembles by default") | Real OpenCode round-trip (real `llm-orc serve` + real client + local Ollama, $0) | The integration scenario above at the replay layer (55/55 measured) + the $0 real-OpenCode acceptance run with delegation **verified fired** (serve-log `dispatch start` / `TurnDecision` — a passing-looking run can be model-direct, the WP-A scar) — this run is ADR-036's Conditional Acceptance **gating condition** | **no until BUILD** — per-component scenarios stub the client; the real-client run is the layer-matching verification (BUILD Step 5.5 work) |
| `delegation_rate` ≥ 0.9 sustained (provisional threshold; ≥25 generation-shaped-turn qualifying window) | Production events layer (trailing confirmation, not acceptance gate) | TurnDecision-sink + classifier scenarios above; the meter read over the first soak window | yes at the events layer once F-2/F-3 land; the *number* is practitioner-revisable at the gate |
| Profile-swap re-validation recorded on any seat-filler model change | Process criterion (FC) | The FC's recorded-run requirement; no scenario can pre-verify a future swap | n/a — process FC, enforced at swap time |

## Feature: Session-Termination Mechanism — Two-Call Trailing Composition (ADR-037, Finding F)

*Conformance disposition (`housekeeping/audits/conformance-scan-cycle-7-loopback5.md`): 8 findings, all expected BUILD work against the newly-decided mechanism — V-01 the unconditional C3 trailing branch this feature replaces (the primary WP); V-02..V-05 the judgment call, digest, standard, and branch enforcement; V-06 the TurnDecision field gap; V-07/V-08 the two FC guards. One incidental true gap outside the table: the AS-3 turn cap (`BudgetController`) is not wired into the loop-driver path — ADR-037 names it the absolute ceiling on non-termination, so wiring it rides in this WP (scenario below).*

### Scenario: a trailing tail triggers the termination judgment before any action call
**Given** a multi-turn tool-driven session whose conversation tail is a tool result with no new user task
**When** the loop-driver processes the turn
**Then** the first model call is the framework-composed termination judgment — not a guidance-composed action call. Refutable: a trailing-tail turn whose first composed request carries `_DELEGATION_GUIDANCE` violates the judgment-first FC (conformance V-01/V-02).

### Scenario: the judgment call is bare-form — no client prompt, no tools
**Given** a trailing tail entering the judgment composition
**When** the judgment request is composed
**Then** it contains exactly a framework-authored judge system message and one user message (task quoted as data + framework-owned digest + deliverable-accounting question), carries no `tools` field, and does not include the client's system prompt. Refutable: a judgment request carrying the client prompt, a tool list, or the delegation guidance violates the bare-form composition. *(Round-2 measured form: 29/30 qwen3:14b.)*

### Scenario: the deliverable-accounting standard is in the question, code QA is not
**Given** the composed judgment question
**When** its text is inspected
**Then** it asks whether requested deliverables have not yet been produced, states that a successful write of a requested file counts as produced, and explicitly does not ask for code-correctness verification. Refutable: a judgment question demanding content verification reproduces round 1's unanswerable standard (Form B 0/10). *(Wording is tunable at the FC-58 bar: revisions re-validate the affected θ arms before landing.)*

### Scenario (integration): the framework-owned digest derives from the framework's own records
**Given** a session in which the framework emitted client-tool calls (grounded carry or delegation → bridge → terminal) and received the client's per-call tool results
**When** the digest is composed for a judgment call
**Then** each action record carries the action kind, target file path, and result, joined from the framework's own emission records with the client's results — and a digest composed from client-serialized messages alone (empty assistant messages, bare "Wrote file successfully" strings) is structurally impossible to produce from the implemented path. Refutable: a digest entry whose file path cannot be traced to a framework emission record violates the digest-provenance FC (conformance V-03; the round-1 failure mode).

### Scenario: COMPLETE → a clean text-only finish turn
**Given** a judgment response whose verdict parses COMPLETE
**When** the loop-driver returns the turn to the client
**Then** the client receives the judgment's summary as an assistant message with no tool calls and no `VERDICT:` line — and the client ends its loop (the model is the stop mechanism). Refutable: a COMPLETE turn carrying tool calls, or leaking the verdict literal, violates the finish-protocol FC (conformance V-05/V-07). *(Finish-text returnability: θ.3 responses were brief factual summaries, no fabricated code.)*

### Scenario: REMAINING → exactly one C3-form action call, judgment exchange discarded
**Given** a judgment response whose verdict parses REMAINING
**When** the loop-driver composes the action call
**Then** the composed request is the ADR-036 C3 trailing form (session messages + standalone trailing guidance) with the judgment exchange absent from its context — byte-equal to what the pre-ADR-037 trailing composition produced on the same session state. Refutable: a call-2 request containing the judgment question or verdict violates the call-2 form-preservation FC (conformance V-08). *(This byte-equality is what lets E4b's 9/10 ride as call 2's evidence.)*

### Scenario: TurnDecision carries the finish-policy fields
**Given** a trailing-turn judgment (either verdict)
**When** the TurnDecision-family event is emitted
**Then** it carries the turn shape and the judgment verdict, and the false-continue frequency is computable from emitted events alone — no log archaeology. Refutable: a deployment that cannot compute how often work-complete tails failed to finish violates the termination-observability FC (conformance V-06; this is the event shape WP-LB-J consumes).

### Scenario (integration): a work-complete session converges end-to-end
**Given** a session whose requested deliverables have all been produced (framework records show each requested file written successfully)
**When** the next trailing turn is processed through the real loop-driver against the real session records
**Then** the judgment fires, parses COMPLETE, and the session returns a text-only finish turn — the session does not delegate a phantom revision. Refutable: a delegated dispatch on a fully-satisfied session is the Finding F signature. *(Replay layer: θ.3b 9/10. The real-client layer is the acceptance criterion below.)*

### Scenario: the AS-3 turn cap backstops the loop-driver path
**Given** the loop-driver processing any multi-turn session
**When** the session's turn count reaches the budget cap
**Then** the `BudgetController` check fires on this surface and the session terminates regardless of judgment outcomes — the deterministic ceiling ADR-037 names is actually wired (conformance incidental: currently absent from the loop-driver path). Refutable: a session exceeding the cap without termination violates AS-3 on this surface.

### Preservation: first-turn and new-user-task composition is untouched
**Given** a first turn, or a trailing turn carrying a genuine new user task
**When** the seat-filler request is composed
**Then** ADR-036's merge branch composes exactly as today (guidance attached to the user task; no judgment call fires — the judgment is specific to no-new-task tool-result tails). Refutable: a judgment call on a user-tail turn violates ADR-037's scope. *(ψ/ψ′ first-turn evidence — 40/40 — rides untouched.)*

### Preservation: mid-task delegation is preserved through the two-call path
**Given** a work-remaining trailing tail (the E4/E4′ shapes)
**When** the two-call composition processes it
**Then** the session still delegates to a capability ensemble at the call-2 step — the termination mechanism does not reintroduce Finding B (inline generation) or premature finish. Refutable: an inline `write` of new content, or a finish turn, on a work-remaining tail. *(Composed estimate ~0.9: θ.4b 10/10 × E4b 9/10 — labeled composed, not end-to-end.)*

### Preservation: carry-side behavior is unchanged at call 2
**Given** a REMAINING verdict whose next action is carry-shaped (read, command, literal write)
**When** call 2 is composed and the seat-filler decides
**Then** carry turns still select client tools with verbatim payloads (the ψ′ Arm B contract) — the judgment layer adds no delegation pressure to carry-shaped work. Refutable: a carry-shaped call-2 turn that delegates, or paraphrases a literal payload.

### Preservation: the judgment exchange is invisible to the client
**Given** any trailing turn processed through the two-call composition
**When** the client inspects everything it receives
**Then** no client-visible surface carries the judgment question, the digest, or the verdict literal — both composition points exist only on the framework ↔ model hop. Refutable: judgment artifacts in any client-visible response.

### Preservation: the single-turn answer-a-question surface (ADR-027) is untouched
**Given** a non-tool-driven request (no client tools → dispatch-pipeline path)
**When** it is served
**Then** the termination judgment does not engage (it is specific to trailing turns on the tool-driven surface); the pipeline's behavior is unchanged.

### Cycle Acceptance Criteria (Finding F)

| Criterion | Specified layer | Verification method | Layer-match check |
|---|---|---|---|
| A completed task's session converges under the real client — the finish turn lands text-only and the client loop ends (the Finding F refutation) | Real OpenCode round-trip (real `llm-orc serve` + real client + local Ollama, $0) | The end-to-end convergence scenario above at the replay layer (θ.3b 9/10) + the $0 real-OpenCode acceptance run with convergence verified from serve-log evidence; **this run is ADR-037's Conditional Acceptance gating condition (a)**, and its judgment calls must be fed by the production digest join, not a constructed digest | **no until BUILD** — the production join (V-03) does not exist yet; the real-client run is the layer-matching verification |
| A work-remaining trailing turn still delegates under the real client (no delegation regression from the termination mechanism) | Real OpenCode round-trip | The mid-task preservation scenario at the replay layer (composed ~0.9) + the same acceptance run exercising at least one work-remaining trailing turn with delegation verified fired (`dispatch start` / TurnDecision — the WP-A scar discipline); **gating condition (b)** | **no until BUILD** |
| False-continue frequency is computable from events alone | Production events layer | The TurnDecision finish-policy scenario; the false-continue share read alongside WP-LB-J's delegation-rate meter | yes at the events layer once V-06 lands |
| ADR-036's ≥0.9 delegation-rate soak becomes readable (Finding F no longer inflates the numerator's turn stream) | Production events layer (trailing confirmation) | The soak window read AFTER this mechanism lands — explicitly deferred until then per the loop-back #5 entry package | deferred by design — reading it earlier is the distortion this criterion guards against |

## Feature: Remaining-Work Anchor — Routing the Judge's Signal Forward (ADR-038, Finding G)

*Conformance disposition (`housekeeping/audits/conformance-scan-cycle-7-loopback6.md`): 3 findings, all expected BUILD work, one logical unit — V-38-1 (capture the judge's REMAINING statement instead of discarding it), V-38-2 (`_seat_filler_messages` gains a remaining-anchor parameter + the "Produce that next." imperative on the REMAINING branch), V-38-3 (the existing `test_remaining_verdict_call2_form_preserved` asserts the pre-amendment byte-equal form and must be updated to the anchored form). ADR-037's two-call composition is fully conforming and unchanged. This feature amends ADR-037's call-2 form preservation only — it touches the REMAINING fall-through, nothing else.*

### Scenario: a REMAINING verdict routes the judge's statement into call 2
**Given** a trailing tool-result tail on the delegation surface whose termination judgment parses REMAINING with a statement naming an unproduced deliverable
**When** the loop-driver composes call 2 (the action call)
**Then** the composed call-2 request's trailing region contains the judge's `VERDICT:`-stripped remaining-work statement followed by the framework imperative "Produce that next." — the signal the judgment computed is routed forward, not discarded. Refutable: a call-2 composition on a REMAINING turn whose trailing region is the bare ADR-036 `_DELEGATION_GUIDANCE` with no remaining-work statement violates the anchor-presence FC (conformance V-38-1/V-38-2).

### Scenario: the anchor is the judge's actual statement, not a fabricated one
**Given** a judgment response whose REMAINING statement reads "The test file test_string_utils.py has not been created yet."
**When** call 2 is composed
**Then** the trailing anchor carries that exact stripped statement (the `strip_verdict` output), not a framework-synthesized or templated description of what remains. Refutable: a call-2 anchor whose remaining-work text does not match the judge's stripped statement for that turn.

### Scenario: only the statement and imperative carry forward — the judgment exchange stays discarded
**Given** a REMAINING verdict whose judgment call carried the judge system message, the quoted task, and the framework-owned digest
**When** call 2 is composed with the anchor
**Then** call 2's context contains the session messages, the ADR-036 trailing guidance, and the remaining-work statement + imperative — and contains NONE of the judgment exchange's other parts: no judge system message, no digest, no `VERDICT:` literal, no judgment question. Refutable: a call-2 request carrying the digest, the judge system message, or the verdict literal violates the amended call-2 form-preservation FC (this preserves ADR-037's context-bounding property).

### Scenario: the anchored call 2 still delegates the named deliverable
**Given** a REMAINING turn whose anchor names an unproduced deliverable
**When** the seat-filler decides the action on the anchored call 2
**Then** it delegates generation of the named deliverable via `invoke_ensemble` (the callee path) — it does not write the named deliverable inline (the Finding B shape) and does not finish (a no-tool-call premature stop). Refutable: an inline `write` of generated content, or a no-tool-call turn, on an anchored REMAINING call 2 (Spike ρ measured delegation 9–10/10, no-tool-call ≤1/10).

### Scenario (integration): a multi-file session advances then converges
**Given** a real-OpenCode session on a multi-deliverable task (e.g. a module and its test file), file 1 produced
**When** the session runs through the loop-driver against the real session records
**Then** each work-remaining trailing tail judges REMAINING and the anchored call 2 delegates the *next distinct* deliverable (no churn re-writing file 1), and once all deliverables are produced the final trailing tail judges COMPLETE and the session finishes text-only — the session both advances and converges in one run. Refutable: a session that re-writes an already-produced file on consecutive trailing turns (the Finding G signature) is the failure this scenario refutes. *(Replay layer: Spike ρ ρ.2-imp 19/20 advance. The real-client layer is the acceptance criterion below — ADR-038's Conditional Acceptance discharge gate, joint with ADR-037's.)*

### Preservation: the COMPLETE branch is unchanged by the amendment
**Given** a trailing tail whose judgment parses COMPLETE
**When** the loop-driver returns the turn
**Then** the session finishes text-only exactly as ADR-037 specified — the remaining-work anchor is composed only on the REMAINING fall-through and never touches the COMPLETE path. Refutable: a COMPLETE turn that composes or emits a remaining-work anchor violates the REMAINING-only scope.

### Preservation: first-turn and new-user-task composition is untouched
**Given** a first turn, or a trailing turn carrying a new user task (ADR-036's merge branch)
**When** the seat-filler request is composed
**Then** no termination judgment fires and no remaining-work anchor is composed — the amendment is scoped to the REMAINING branch of the no-new-task trailing tail. Refutable: a remaining-work anchor on a first turn or a new-user-task tail.

### Cycle Acceptance Criteria (Finding G)

| Criterion | Specified layer | Verification method | Layer-match check |
|---|---|---|---|
| A multi-deliverable session advances through all deliverables (no churn on file 1) AND converges, in a single real-client session | Real OpenCode round-trip (real `llm-orc serve` + real client + local Ollama, $0) | The advance-then-converge integration scenario above at the replay layer (Spike ρ ρ.2-imp 19/20 advance; ρ.1 20/20 naming) + the $0 real-OpenCode multi-file run with the advance sequence and the COMPLETE finish both verified from serve-log evidence in one run; **this is ADR-038's Conditional Acceptance discharge gate, joint with ADR-037's** | **no until BUILD** — the anchor (V-38-1/V-38-2) does not exist yet; the real-client multi-file run is the layer-matching verification |
| The anchored call 2 preserves delegation (no Finding B inline-write regression, no premature-finish regression) | API-boundary composition + real-client run | The delegation-preservation scenario above (Spike ρ delegation 9–10/10, no-tool-call ≤1/10) + the same multi-file run showing `dispatch start` per deliverable | partial — composition verified at the harness layer; the real-client run confirms it at the serving layer |

## Feature: Content Anchor — Routing Produced-Sibling Signatures into the Callee Dispatch (ADR-039, Finding H)

*Conformance disposition (`housekeeping/audits/conformance-scan-cycle-7-loopback7.md`): 5 findings, all expected BUILD work (the mechanism does not exist yet; no FC violations) — V-01 (no anchor injection at the callee dispatch — `_delegate_generation` assembles `task + directive` only; the injection seam is one string concat), V-02 (no signature-extraction utility anywhere in `src/` — net-new, correctness-critical, needs a real-fixture unit test), V-03 (`LoopDriver` has no `SessionArtifactStore` access — the structural prerequisite for reading produced files at dispatch time), V-04 (`ActionRecord` does not carry the artifact-store reference — `target_path` is client-facing; the lower-coupling path is to extend the ADR-037 meta-record seam with `artifact_reference`), V-05 (optional `TurnDecision` anchor-present field for discharge-gate observability). Reuse inherited clean: `SessionArtifactStore.read_deliverable` (R-01, production-ready, ADR-039 names it) and the `_delegate_generation` concat seam (R-02). BUILD commits the V-03 path (inject the store vs extend the record) and the multi-sibling selection policy ("all prior siblings" is the simplest conforming default).*

### Scenario: a delegated write into a session with produced siblings carries their signatures
**Given** a delegated generation that writes a file (`cli.py`) on a session where sibling files (`converters.py`) have already been produced
**When** the framework composes the callee dispatch (the input the capability ensemble receives via `invoke_ensemble`)
**Then** the dispatch context contains the produced siblings' public API signatures (function/class signatures + one-line docstrings). Refutable: a callee dispatch on a session with produced siblings whose context carries only `task + directive` and no sibling signatures violates the content-anchor-presence FC (conformance V-01).

### Scenario: the anchor is extracted from the real produced file, never guessed
**Given** a produced sibling `converters.py` that defines exactly `celsius_to_fahrenheit`, `fahrenheit_to_celsius`, `celsius_to_kelvin`
**When** the framework builds the content anchor
**Then** every injected signature names a symbol that actually exists in the produced file (a structural extraction read via `SessionArtifactStore.read_deliverable`), and the anchor contains no symbol the file does not define. Refutable: an injected signature naming a symbol absent from the produced file — the decoy failure mode (a wrong anchor resolved 0/10, *below* the unanchored baseline, because the coder obeys whatever API it is handed, so anchor correctness is the load-bearing FC; conformance V-02/V-03).

### Scenario: the anchor carries signatures, not full bodies
**Given** a produced sibling whose functions have multi-line implementations
**When** the content anchor is composed
**Then** the anchor carries the signatures + docstrings, not the full function bodies — the compact form Spike ξ selected (B = C_full, signatures suffice, and the dispatch does not bloat as the deliverable count grows). Refutable: a content anchor carrying full function bodies violates the signatures-form FC.

### Scenario: the anchor fires regardless of callee type (code and prose)
**Given** a delegated generation routed to a prose-generating callee (`prose-improver`, the README) on a session with produced code siblings
**When** the callee dispatch is composed
**Then** the prose callee receives the sibling signatures exactly as a code callee does — the augmentation is callee-agnostic (Spike ξ prose arm: blind README 0/10 → anchored 10/10). Refutable: a content anchor composed for `code-generator` but omitted for `prose-improver` on the same session state violates the callee-agnostic scope.

### Scenario (integration): dependent deliverables reference real sibling APIs
**Given** a real-OpenCode session on the 5-file temperature-library task (the exact Finding H task), with the content anchor wired into the callee dispatch
**When** the session runs to completion against the real client
**Then** the dependent deliverables reference only symbols that exist in their produced siblings — `cli.py` calls the real `converters` functions (no invented `convert_temperature`), the tests match the implementation, and the README documents real functions (no invented `fahrenheit_to_kelvin`, no Rankine scale) — while axis-1 holds (zero churn, convergence). Refutable: a dependent deliverable that references a nonexistent sibling symbol (the Finding H signature) is the failure this scenario refutes. *(Harness layer: Spike ξ code 3/10→10/10, prose 0/10→10/10. The real-client layer is the acceptance criterion below — ADR-039's Conditional Acceptance discharge gate.)*

### Preservation: delegation and the form contract hold under the anchor
**Given** a delegated write whose callee dispatch now carries the sibling signatures
**When** the seat-filler decides the action and the callee generates
**Then** the turn still delegates (no Finding B inline-write regression) and the produced file still honors ADR-035's form contract (no fenced-prose-into-code regression traceable to the added context). Refutable: an inline `write` of generated content, or an ADR-035 form-contract violation traceable to the anchor, on an anchored dispatch.

### Preservation: no produced siblings → no anchor
**Given** a first delegated write on a session with no already-produced file deliverables (or a write with no cross-file dependency)
**When** the callee dispatch is composed
**Then** no content anchor is injected — the augmentation fires only when there are produced siblings to source. Refutable: a content anchor composed on a session with no produced file deliverables.

### Cycle Acceptance Criteria (Finding H)

| Criterion | Specified layer | Verification method | Layer-match check |
|---|---|---|---|
| Dependent deliverables (code AND prose) reference real sibling APIs in a single real-client run of the 5-file Finding H task, with axis-1 (zero churn, convergence) preserved | Real OpenCode round-trip (real `llm-orc serve` + real client + local Ollama, $0) | The integration scenario above at the harness layer (Spike ξ code 3/10→10/10, prose 0/10→10/10) + the $0 real-OpenCode 5-file re-run with `cli.py`/tests/README all referencing real APIs, verified by reading the landed files; **this is ADR-039's Conditional Acceptance discharge gate** | **no until BUILD** — the anchor (V-01..V-03) does not exist yet; the real-client run is the layer-matching verification, and the Finding H lesson (a synthetic pass hid the real-client gap) is why the harness PASS is necessary-not-sufficient |
| The anchor is framework-sourced from real files, never guessed (a wrong anchor is worse than none) | Framework extraction + real-client run | The "extracted from the real produced file" scenario above (Spike ξ decoy 0/10 < baseline 3/10) + the discharge run showing no invented cross-file references | partial — extraction verified at the harness/unit layer; the real-client run confirms the framework actually delivers the real anchor to the callee |
| Delegation + ADR-035 form contract preserved under the anchor | API-boundary composition + real-client run | The preservation scenario above + the discharge run showing `dispatch start` per deliverable with no inline-write and no form-contract regression | partial — composition verified at the harness layer; the real-client run confirms it at the serving layer |

## Feature: Deterministic Completeness Gate (ADR-040, Finding I / Spike σ)

*Conformance disposition (`housekeeping/audits/conformance-scan-cycle-7-loopback7-tail.md`): CONFORMING, zero permanent violations (P1/P2/P3 all 0). Unlike the other loop-back features, the code landed before the ADR (the spike → build → ADR pattern these tails use), so the scan verified the ADR describes the built code rather than gating future work. Two expected-temporary items, not violations: the `completeness:` INFO log in `_completeness` and the env-gated `_resolve_judgment_seat` Arm-B hook, both scheduled for spike-close removal. The four FCs each map to landed code and a refutable test.*

### Scenario: a named-file task completes deterministically, no judge consulted

**Given** a trailing tail on a task that names its deliverables, where every requested file appears in the framework's produced write paths
**When** the loop-driver computes completeness
**Then** the verdict is COMPLETE from the set comparison `requested ⊆ produced` alone, with no judgment-seat call, and the session finishes text-only. Refutable: a judgment-seat invocation on a named-file trailing turn, or a COMPLETE finish with a requested file unproduced, violates the deterministic-verdict FC. *(Test: `test_complete_when_all_requested_produced_no_judge_call`. The false-COMPLETE Spike σ measured, 1/5 invariant of judge capability at the measured n, cannot occur on this path.)*

### Scenario: a named-file task with work remaining anchors the missing files, ignoring a wrong judge

**Given** a trailing tail on a named-file task where some requested files are unproduced, and a stochastic judge that would wrongly say COMPLETE
**When** the loop-driver computes completeness
**Then** the verdict is REMAINING from `requested − produced`, the judge is never consulted, and the call-2 anchor names exactly the unproduced files. Refutable: a judge call on this turn, or an anchor that omits an unproduced file or names a produced one, violates the deterministic-anchor FC. *(Test: `test_remaining_overrides_a_wrong_judge_and_anchors_missing`. This is the named-file supersession of ADR-038's judge-statement anchor.)*

### Scenario (persist-once): the requested set survives a client-compacted later turn

**Given** a session that named its files on turn 1 (the guaranteed-full task), then a later trailing turn whose task text the client has compacted so it names no files
**When** the loop-driver computes completeness on the later turn
**Then** it reads the requested set persisted from turn 1 (first-non-empty-wins) and still computes REMAINING for the unproduced files, rather than collapsing to an empty set and falling back to the judge. Refutable: a session whose persisted `requested` shrinks or empties across turns, or a false-COMPLETE traced to a re-derived empty set, violates the persist-once stability FC. *(Tests: `test_persisted_requested_survives_a_compacted_later_turn`, `test_decide_persists_the_requested_set_once_on_first_naming`. The compaction is simulated at the unit layer; no live compaction was observed in the discharge runs, a recorded BUILD-watch item.)*

### Preservation: a task that names no files falls back to the ADR-037/038 judge

**Given** a trailing tail on a task that names no deliverable files
**When** the loop-driver computes completeness
**Then** the requested set is empty, so the bare-form ADR-037 judge runs and the ADR-038 judge-statement anchor applies, exactly as before ADR-040. Refutable: a no-files task taking the deterministic path violates the no-files-fallback FC. *(Test: `test_falls_back_to_judge_when_task_names_no_files`. The deterministic gate is scoped to the task shape where requested files are mechanically extractable; everything else rides the unchanged judge path.)*

### Cycle Acceptance Criteria (Finding I)

| Criterion | Specified layer | Verification method | Layer-match check |
|---|---|---|---|
| A named-file multi-file session converges to a deterministic COMPLETE (all files produced, text-only finish, client loop ends) with no false-COMPLETE | Real OpenCode round-trip (real `llm-orc serve` + real client + local Ollama, $0) | Two $0 real-OpenCode 5-file runs, one at the 14b seat/coder and one at the production config (14b seat, 8b coder), each converging in six turns with `requested=[5]` every turn, monotonic produced 1→5, and COMPLETE at turn 6 | **yes — discharged 2026-06-10** (both tiers; evidence Spike σ log §RESOLUTION + `scratch/spike-sigma-premature-finish/j3_diag/`). The code was built and validated live before the ADR, so this is unconditional at the DECIDE gate |
| persist-once holds the requested set stable under a real-client compaction event | Live real-client run | A real-client session exhibiting mid-session task compaction, with the gate still reading the persisted set | **no — unverified live** (no compaction occurred in either discharge run); validated only by a unit test that simulates the compaction. A BUILD-watch item carried forward |
| The gate's coverage of the real task space (the named-file fraction) | Deployment characterization | The fraction of target-deployment tasks that name their files explicitly | **not characterized** — the cycle's north-star task shape is named-file, so the restriction is expected to cover the majority, but the coverage fraction is an untested assumption (ADR-040 §Consequences) |

## Feature: Destination-Validity Gate (ADR-041, Spike π form/adequacy seam)

*Conformance disposition (`housekeeping/audits/conformance-scan-cycle-7-loopback8.md`): the mechanism is **env-gated spike code** (active only under `LLMORC_SPIKE_PI_GATE=parse`), validated live but not in the production path. 10 BUILD-work items (de-gate ×3, FormGate-seam install, the `FormGate` 3-arg interface reconciliation, constructor/factory/cap/method renames, spike-comment removal) + 2 test-gap VIOLATIONS (the env-gated path is dead in CI: `bridge.destination_paths` recorded but never asserted; no test exercises the parse-check logic at all). `destination_path` is already fully threaded Terminal → `marshal`. The discharge gate below is met for protection-design + recovery at the spike layer; the production install and the named coder-tier escalation are BUILD. Scenarios are written against the de-gated production target.*

### Scenario: an invalid deliverable never reaches the client (the protection floor)

**Given** a delegated generation whose deliverable is bound for a client `write` tool with a destination path of `.py` (or `.json`), and a coder that produced output that does not `ast.parse` (a trailing-prose bleed or wrong-language content)
**When** the marshalling boundary evaluates the deliverable at the destination-validity gate
**Then** the gate refuses to emit (raises `FormRefusedError`), so no invalid file reaches the client's workspace — the deliverable is either recovered before emission or the turn degrades to a dispatch-failure rather than shipping a broken `write`. Refutable: a `.py` deliverable that fails `ast.parse` reaching the client as a `tool_calls` `write` violates the protection FC. *(Spike π live arm: 0 invalid files across 5 gated sessions vs baseline 3/5. The gate inspects bytes — protection is degradation-independent.)*

### Scenario: an intermittent bleed self-heals within the serving turn (server-side recovery)

**Given** a gate refusal on a destination whose coder bleeds *intermittently* (a valid re-sample is reachable within the retry cap)
**When** the Loop Driver re-dispatches the same destination within the serving turn (the coder re-samples, reusing the delegation path so the action is recorded once)
**Then** a valid deliverable emits and the session continues, with no client-visible refusal and no broken-file diff. Refutable: an intermittent bleed that ends the session at the first refusal (no within-turn re-dispatch) violates the recovery FC; a re-dispatch that double-records the action violates the single-record property. *(Spike π live arm runs 1/2/5: `cli.py`/`test_cli.py`/`converters.py` each rescued within the cap → converged. The smoke finding grounds *why* recovery is server-side: a client-facing refusal-as-`stop` ends the OpenCode loop before ADR-040's next-turn re-delegation could fire.)*

### Scenario: a persistent bleed is protected, not converged, and routes to escalation

**Given** a gate refusal on a destination whose coder bleeds *persistently* (the cheap 8b coder fails `ast.parse` on every attempt within the cap)
**When** the recovery cap exhausts
**Then** the gate degrades to a dispatch-failure `stop` (the session ends short with fewer files, never a broken file), and the failure surfaces as the protect-but-not-converge floor that routes to coder-tier escalation (ADR-014 Calibration Gate), not as a shipped invalid deliverable. Refutable: a cap-exhausted destination that emits an invalid file (rather than refusing) violates the protection floor. *(Spike π live arm runs 3/4: `cli.py` / `test_converters.py` exhausted → refused → short. Arm E: the cheap 8b bleeds `cli.py` ~50% even fresh [3/6]; MiniMax coder 6/6 — the lever is coder capability, confirmed in isolation, n=6.)*

### Scenario (coverage boundary): a prose destination passes un-inspected

**Given** a deliverable bound for a `.md` (or other non-structurally-checkable) destination
**When** the gate evaluates it
**Then** the gate passes it through un-inspected (prose form is not structurally checkable; a parseable-but-wrong prose deliverable is the irreducibly-semantic residual handed to PLAY). Refutable: a gate that refuses or mangles a legitimate `.md` deliverable violates the determinism-boundary FC (the corpus arm measured FP=0 for the parse-check, including a README carrying a ```bash fence). *(The parse/validity edge is the principled determinism boundary — the gate is deterministic exactly where the destination type admits a structural check.)*

### Cycle Acceptance Criteria (form/adequacy seam)

| Criterion | Specified layer | Verification method | Layer-match check |
|---|---|---|---|
| No invalid deliverable reaches the client across a real multi-file trajectory (the protection floor) | Real OpenCode round-trip (real `llm-orc serve` + real client + local Ollama, $0) | The $0 real-OpenCode 5-file run with the gate on, every produced `.py` `ast.parse`-valid and every `.json` `json.loads`-valid at the client | **yes — met at the spike layer** (Spike π Cell B: 0 invalid across 5 gated sessions; baseline 3/5). Production-path discharge of ADR-035's form-seam protection CA is the BUILD de-gate-and-install |
| Intermittent bleeds self-heal via server-side recovery (the convergence helper) | Real OpenCode round-trip ($0) | The same run, with within-turn re-dispatch rescuing intermittent bleeds and the session converging | **partial — protects-but-does-not-recover** (converged B 3/5 vs A 2/5, +1/5; rescues intermittent, exhausts on persistent). Convergence under the cheap tier is a separate Conditional Acceptance |
| Persistent bleeds close under coder-tier escalation | Wired session under an escalated coder tier | A real-client multi-file session that converges when the persistent-bleed destination is delegated to an escalated coder (ADR-014 lever) | **no — named, not built** (Arm E confirmed the lever in isolation, n=6 on `cli.py`; session-level convergence under wired escalation is BUILD) |
| The experiential trade-off of the gate's visible failure mode (cap-exhausted short session vs. pass-through broken-file diff) | PLAY | Stakeholder observation of whether a short session reads as better/worse than a broken-file diff; FC-51 `TurnDecision` diagnostics distinguish the turn types | **PLAY** — the experiential question the spike did not resolve (argument-audit-surfaced), alongside the semantic parses-but-wrong residual |

# Interaction Specifications: Agentic Serving

**Derived from:** `product-discovery.md` (Stakeholder Map, Jobs and Mental Models, Product Vocabulary)

**Complements:** `scenarios.md` (business-rule behavior)

**Method caveat.** The derivation from stakeholder models to task decomposition is an open design problem in RDD v0.7.3. These specifications are a best-effort interpretation of the stakeholder jobs and mental models from product discovery, not a systematic derivation. They are concrete enough to create a playable surface for `/rdd-play` and abstract enough to survive UI or protocol refinements in ARCHITECT and BUILD.

**Open boundary.** One interaction boundary is not yet pinned: how the Orchestrator Agent handles the *client's* tool definitions (the tools declared by an agentic coding tool in its `/v1/chat/completions` request). The current ADRs specify the orchestrator's internal tool surface (ADR-003) but do not commit on whether client-declared tools are passed through to the orchestrator's LLM, held by the orchestrator for delegation to invoked ensembles, or rejected. This affects the Tool User interaction below — the placeholder text reflects the ambiguity rather than hiding it. Decision needed in ARCHITECT or a follow-up DECIDE mini-cycle.

---

## Stakeholder: Tool User

**Super-Objective:** Complete coding work through an agentic coding tool without knowing or caring what sits behind the endpoint, while retaining autonomy from a single provider.

### Task: Configure the agentic coding tool to use llm-orc as its backend

**Interaction mechanics:** The operator (or the tool user themselves when they are also the operator) sets the tool's API endpoint to the llm-orc serving layer URL and selects a model from the `/v1/models` list. The listed model names are the orchestrator's configured profile (ADR-011). From this point forward, all requests from the coding tool flow through the Orchestrator Agent behind the endpoint. The tool user does not configure ensembles, profiles, or Plexus at this layer.

### Task: Complete a coding task through the tool's normal workflow

**Interaction mechanics:** The tool user interacts with their agentic coding tool exactly as they would with any other LLM backend — asking questions, requesting code changes, reviewing suggestions, iterating. The Orchestrator Agent, behind the endpoint, runs its ReAct loop and delegates to ensembles via `invoke_ensemble` and (where appropriate) `compose_ensemble` while the tool user waits. The tool user observes responses, tool calls, and streamed content in the shape their tool expects.

*Open: the orchestrator's interaction with the client's tool definitions (bash, file-edit, etc.) is not pinned by current ADRs. This interaction's exact mechanics resolve once that boundary is decided.*

### Task: Observe what the orchestrator is doing (visibility-configurable)

**Interaction mechanics:** When the operator has configured visibility surfacing (see Open Question #2 in the domain model — visibility format is unresolved), the tool user sees composition events, ensemble invocations, or other orchestrator activity inline in the tool's response stream. When visibility is not surfaced, these events are invisible to the tool user and only visible to the operator through whatever visibility surface the operator has enabled. For the tool user, this ranges from "endpoint is a model" (no visibility) to "endpoint is an orchestrator" (composition events visible) depending on the Autonomy Level and visibility configuration.

### Task: Experience budget exhaustion cleanly

**Interaction mechanics:** When the Session reaches its turn limit or token limit, the tool user receives a final response that names budget exhaustion as the termination cause. The tool user can then start a new session (which gets a fresh Budget) or request that the operator adjust Budget defaults.

---

## Stakeholder: Ensemble Author / Operator

**Super-Objective:** Maintain a library of ensembles, profiles, and scripts that the orchestrator uses effectively, while observing how the system uses that library and improving it through tinkering and organic stabilization.

### Task: Bootstrap Plexus from the existing library

**Interaction mechanics:** The operator runs an llm-orc bootstrapping command that pushes the contents of the library — ensemble YAML files, script source, profile configurations, execution artifacts — into Plexus as source material (AS-4, ADR-010). Ingestion runs; enrichment runs asynchronously. The operator does not wait for enrichment to complete before running sessions; the knowledge graph's value at session start grows as enrichment catches up. Bootstrapping is a push operation, not a Plexus-pull.

### Task: Run the serving layer for client consumption

**Interaction mechanics:** The operator starts the llm-orc server with serving layer enabled, chooses the orchestrator's Model Profile (ADR-011), sets Budget defaults (ADR-005), and sets the default Autonomy Level (ADR-008). One or more agentic coding tools can then connect via the endpoint. The operator's responsibility after startup is observation, not per-request intervention (subject to Autonomy Level).

### Task: Observe orchestrator behavior during sessions

**Interaction mechanics:** The operator watches sessions through visibility surfaced inline in the tool user's response stream — composition events, calibration outcomes, and other orchestrator-internal signals render as `[kind: {json}]` narration on `delta.content` so vanilla OpenAI-compat clients (OpenCode, Roo Code, Cline) display them inline in the assistant message (OQ #2 resolved during WP-E build, 2026-04-22). What the interaction commits to is that these events are *observable in the conversation*, not opaque. Operator-only tooling surfaces (SSE comments, dedicated events endpoint) can layer on as a second audience-specific surface without changing the inline narration shape.

### Task: Review local-tier compositions and decide on promotion

**Interaction mechanics:** At default Autonomy Level (ADR-008), the Orchestrator Agent produces composed ensembles in the local tier automatically, subject to calibration (ADR-007). The operator reviews these compositions and their accumulated Quality Signals through a library-browsing interface (CLI, web UI, or equivalent — existing llm-orc `list_ensembles` primitives extend to this). Promotion from local to global or library tier requires explicit operator approval — the orchestrator cannot self-promote at default Autonomy Level. The operator's decision is informed by Quality Signals, not frequency (AS-5).

### Task: Tune Budget, Autonomy Level, and orchestrator profile

**Interaction mechanics:** The operator adjusts serving layer configuration — default Budget sizes, default Autonomy Level, orchestrator Model Profile — through configuration files or CLI commands. Changes take effect on new Sessions; active Sessions continue under their existing configuration (ADR-011 session-boundary discipline). The operator can also set per-session overrides within configured bounds (e.g., allowing a specific tool user a tighter Autonomy Level for production work).

### Task: Tinker — adjust system behavior through experimentation

**Interaction mechanics:** The operator uses their own agentic coding tool pointed at their own llm-orc server, collapsing the tool-user and operator roles into one person (product discovery assumption inversion #3). They observe orchestrator decisions in the tool's response, note patterns they want to change, edit ensemble YAML or profile configurations outside the agentic flow, and run the next session to see the effect. Tinkering is a cycle of observation → edit → re-run, not a one-shot configuration task.

### Task: Configure per-skill tier defaults (ADR-015)

**Interaction mechanics:** At deployment, the operator configures Model Profile pairs (cheap-tier + escalated-tier) for each of the eight Topaz skills (`code_generation`, `tool_use`, `mathematical_reasoning`, `logical_reasoning`, `factual_knowledge`, `writing_quality`, `instruction_following`, `summarization`) — sixteen Model Profile slots, with deployment-time defaults shareable across skills (e.g., a single local-7B profile may serve as cheap-tier for code_generation and tool_use). The operator declares each ensemble's primary Topaz skill in the ensemble's YAML metadata. Configuration is via configuration files; changes take effect on new Sessions per ADR-011's session-boundary discipline. The friction trade is for discovery — full taxonomy enables operator deployment evidence to surface which skill dimensions actually drive value.

### Task: Author or update structured-handoff artifacts (ADR-013)

**Interaction mechanics:** For Cluster 2 sessions (BUILD/ARCHITECT/DEBUG/REFACTOR territory), the operator authors and maintains three artifacts: `feature_list.json` (with monotonic `passes` field; cataloging the session's feature scope), an append-only progress log (free-text narrative continuity), and `init.sh` (deterministic environment bootstrap script). The operator runs `init.sh` integrity hash rotation when legitimately modifying the script — recording the new hash in the Session Registry's configuration. Write-gate validation enforces structural non-regression at the schema level; operator overrides (e.g., setting a `passes: true` feature back to `false`) require an explicit audit-logged action.

### Task: Declare session cluster at session start (ADR-013)

**Interaction mechanics:** At session creation, the operator declares the session's cluster via configuration or CLI flag — Cluster 1 (RESEARCH/DECIDE/SYNTHESIZE), Cluster 2 (BUILD/ARCHITECT/DEBUG/REFACTOR), or Cluster 3 (DISCOVER/PLAY). Cluster 2 default-activates the structured-handoff artifact set; Cluster 1 and 3 do not. Cross-cluster sessions (declared as multi-cluster, or ambiguous) default to Cluster 2 behavior — required artifact set — per ADR-013 disposition (i).

### Task: Run baseline-competence calibration ensemble at install/startup (capability-floor)

**Interaction mechanics:** At install or first-startup, the operator runs the baseline-competence calibration ensemble (an llm-orc-shipped ensemble that probes the operator's available local models against capability-floor baselines). The ensemble produces an operator-readable report identifying which local models meet the floor, which fall below, and what concrete remediation is recommended (configure cheap-cloud orchestrator profile; install more-capable local model; etc.). The operator can run the probe explicitly via `llm-orc orchestrator probe` or have it run automatically at first-run startup if configured to do so.

### Task: Review calibration audit diagnostics (ADR-016 mechanism (d))

**Interaction mechanics:** The periodic out-of-band audit dispatch produces drift-detection diagnostics that flow to operator-facing storage (log entries, optional webhook/email per deployment configuration). For advisory drift, the operator reviews diagnostics at their own cadence — diagnostic + parameter-tuning recommendation surface lets the operator approve, override, or ignore the recommended adjustment; on approval, the calibration system applies the adjustment at the next session boundary. For severe drift, the calibration system enters fail-safe mode automatically (verdicts default to Reflect-or-Abstain); the operator must review and release the system from fail-safe.

### Task: Configure Conversation Compaction defaults (ADR-012)

**Interaction mechanics:** The operator tunes four threshold defaults — Layer 0's 50K-character persist trigger, Layer 2's 60-minute idle window, Layer 3's 12K-token notes cap, Layer 4's 3-failure circuit-breaker — through configuration files. Defaults match Anthropic's published values; operationally tunable per deployment workload shape. Filesystem persistence root for Layer 0 must also be operator-configured. The operator monitors Layer 4 failures (the typed-error log surface); persistent Layer 4 failures may indicate the orchestrator Model Profile's semantic-summary capability is mismatched to the deployment's session shape.

### Task: Extend tool-call structural validation pattern set (ADR-017)

**Interaction mechanics:** The default phantom-tool-call pattern set is conservative (minimal). When the operator observes a deployment-specific phantom-tool-call pattern not in the default set, the operator extends the pattern set via deployment configuration. The structural validation guard scans incoming orchestrator responses with the extended pattern set; new patterns trigger `phantom_tool_call` errors when matched without a corresponding tool-call structure. The operator-extension surface is the operational discovery path, not a fallback — defaults are minimal because the cycle's spike evidence does not support a richer calibrated default.

---

## Stakeholder: Orchestrator LLM

**Super-Objective:** Resolve the tool user's task effectively within the Session Budget by routing to existing ensembles or composing new ones from the library's primitives, informed by the knowledge graph when available.

### Task: Receive a task and orient on it

**Interaction mechanics:** At the start of each ReAct iteration, the Orchestrator Agent considers the incoming messages, consults the library (`list_ensembles` when it needs to know what is available), and, when Plexus is active, may call `query_knowledge` to check whether similar tasks have been handled before (ADR-009 Phase 1 tool-first). Orientation is done through tool calls, not through context injection in this phase (injection is deferred per ADR-009).

### Task: Route an existing ensemble to a task

**Interaction mechanics:** When the orchestrator identifies an ensemble from the library whose Routing Decision history or stated capability matches the task, it emits an `invoke_ensemble` tool call with the task input. The Ensemble Engine runs the ensemble; the Result Summarizer condenses the output (ADR-004); the summary returns to the orchestrator's context. The orchestrator uses that summary to continue reasoning.

### Task: Compose a new ensemble when nothing existing fits

**Interaction mechanics:** When no existing ensemble fits the task, the orchestrator emits a `compose_ensemble` tool call specifying an ensemble configuration drawn from existing primitives (profiles, scripts, ensembles — ADR-006 full palette). Composition-time validation runs (AS-2, ADR-006); if validation fails, the orchestrator receives an error observation and may retry with a different composition, invoke an existing ensemble despite imperfect fit, or escalate via the response. The newly composed ensemble is written to the local tier and enters calibration (ADR-007). The orchestrator may then immediately invoke it.

### Task: Record outcomes as work proceeds

**Interaction mechanics:** At decision points (after an ensemble invocation completes, after a calibration check produces a Quality Signal, at the end of a task), the orchestrator emits `record_outcome` calls that write Routing Decisions, Quality Signals, and outcome metadata to the knowledge graph (ADR-009 — a no-op when Plexus is absent, not an error). Recording is non-blocking; the orchestrator does not wait for enrichment (ADR-010).

### Task: Stay within Budget

**Interaction mechanics:** The orchestrator does not directly observe the Budget's remaining capacity — the control plane enforces it at each ReAct iteration boundary (AS-3, ADR-005). The orchestrator's interaction with Budget is observational through prior-outcome records accessible via `query_knowledge` (costs of similar sessions), not negotiable. When Budget is exhausted, the control plane terminates the Session — the orchestrator does not emit a "final" turn; its last complete turn becomes the response.

### Task: Terminate the task cleanly

**Interaction mechanics:** When the orchestrator determines the task is complete, it emits a final message with `finish_reason: stop` (OpenAI-compatible protocol). No tool call is in flight at termination. If the orchestrator reasons itself into a dead end, it emits a message explaining the impasse and stops, rather than looping until Budget exhaustion — this is a quality behavior the orchestrator's Model Profile should support, not a guarantee of the framework.

### Task: Receive and act on calibration verdicts (ADR-014)

**Interaction mechanics:** For each `invoke_ensemble` dispatch, the Calibration Gate produces a verdict (Proceed / Reflect / Abstain) that the orchestrator consumes through dispatch outcomes. *Proceed* — dispatch proceeds normally; the orchestrator continues its ReAct loop. *Reflect* — the orchestrator may choose to escalate the dispatch's tier (handled by the per-role tier-escalation router per ADR-015 transparently), reformulate the dispatch with different inputs, or proceed with explicit acknowledgement of the low-confidence verdict. *Abstain* — dispatch is blocked; the orchestrator receives a typed `calibration_abstain` error and must reformulate or take a different action.

### Task: Recover from escalation-bypass typed errors (ADR-015)

**Interaction mechanics:** When a calibration verdict is *Abstain*, the per-role tier-escalation router does not perform tier escalation; it produces a typed `escalation_bypass` error. The orchestrator must reformulate the dispatch (different inputs, different ensemble, or different sub-task decomposition), not retry the same dispatch with an escalated tier — the Abstain semantics imply a different action class is needed, not more capability.

### Task: Recover from phantom_tool_call typed errors (ADR-017)

**Interaction mechanics:** When the structural validation guard detects a mismatch between the orchestrator's prose claim of a tool call and the actual tool-call structures emitted, the orchestrator receives a typed `phantom_tool_call` error including the detected prose substring and the actual list of emitted tool-call structures. The orchestrator must re-emit the response with actual tool-call structures (if a tool call was intended), reformulate the dispatch (if no tool call was intended but the prose mistakenly claimed one), or abstain — silent retry is not the intended recovery.

---

## Stakeholder: Skill Orchestration User (Cycle 5 introduction)

**Super-Objective:** Compose a skill orchestration process (RDD, Anthropic Skills, OpenAI Assistants, MCP-based skill framework, or other) against the orchestrator's capability library — decompose a higher-level workflow into capability-typed sub-tasks the orchestrator dispatches by Topaz skill.

### Task: Decompose a workflow into capability-typed sub-tasks

**Interaction mechanics:** The skill orchestration user's skill framework owns workflow decomposition: which sub-skills run, in what order, with what dependencies on prior sub-task outputs. The decomposition is **client-side only**; the orchestrator never sees the workflow's higher-level shape. The skill framework must produce sub-tasks tagged with Topaz skill identifiers (per ADR-015) — either directly (if the framework's vocabulary aligns) or via an adapter layer mapping the framework's internal vocabulary to the Topaz 8-skill taxonomy (per ADR-021 §Negative consequence). RDD's `rdd:*` skill plugin maintains this mapping internally; `skill-framework-capability-registry.md` documents the per-skill-framework mapping for deployments that need to reference it.

### Task: Dispatch a sub-task via explicit ensemble naming

**Interaction mechanics:** The skill framework consults `skill-framework-capability-registry.md` (or `list_ensembles()` at runtime) to identify the target capability ensemble for the sub-task, then emits an OpenAI-compatible chat completion request to the orchestrator with `invoke_ensemble("<ensemble-name>", {...})` as the tool-call argument. The orchestrator dispatches directly without inferring the target ensemble from prompt content. Explicit naming is the **preferred dispatch shape** — it preserves ADR-015's pre-specified-routing commitment end-to-end. The skill framework absorbs the cost of maintaining library-topology knowledge.

### Task: Dispatch a sub-task via natural-language prompt

**Interaction mechanics:** When the skill framework cannot or chooses not to maintain library-topology knowledge, it emits the sub-task as natural-language user-prompt content (e.g., "extract factual claims from the following text: ..."). The orchestrator's ReAct loop selects the target capability ensemble using LLM-judgment matching of the prompt's task description to ensemble descriptions returned by `list_ensembles()`. This reintroduces LLM-judgment at the *capability-selection boundary* (retrieval over the library, not evaluative classification of output quality — distinct from the pattern ADR-015 §(f) rejected). The Tier-Router Audit's drift criteria (ADR-018) measure the operational impact of using this path.

### Task: Carry workflow state forward across sub-tasks

**Interaction mechanics:** State that crosses capability sub-tasks (e.g., "the lit-review's search results feed the claim-extractor's input") lives **client-side**, in the skill framework's own state management. The skill framework formats prior sub-task outputs into the next sub-task's prompt (or `invoke_ensemble` arguments) before emitting the next dispatch. The orchestrator's per-request model does not maintain workflow state across `invoke_ensemble` calls — `invoke_ensemble`'s fresh-context property is the architectural commitment.

### Task: Reformulate when MissingSkillMetadataError is returned

**Interaction mechanics:** When the skill framework attempts to dispatch to an unauthored Topaz slot (e.g., `mathematical_reasoning` in the Cycle 5 default deployment), the Tier-Escalation Router returns `MissingSkillMetadataError` (per ADR-015) with `recovery_action_required="reformulate"`. The orchestrator's recovery path reformulates the dispatch — typically by trying a different ensemble that approximates the unauthored capability, or by emitting the sub-task as natural-language content for the orchestrator's direct response. The skill framework's decomposition shape may need to revise if a load-bearing sub-task consistently routes to an unauthored slot.

### Task: Consume capability-ensemble dispatch results

**Interaction mechanics:** Each `invoke_ensemble` dispatch returns the capability ensemble's output (already summarized per AS-7 / ADR-004) as the chat completion response content. The skill framework parses the response, validates the shape against its expected sub-task output format, and proceeds to the next capability sub-task in its decomposition. Output-shape mismatches (e.g., the response doesn't contain the structured fields the skill framework expected) are the skill framework's responsibility to handle — typically by retrying the dispatch with refined inputs, or by surfacing the mismatch to the skill orchestration user as a workflow-level error.

---

## Ensemble Author / Operator — additional Cycle 5 tasks

### Task: Author capability ensembles in the agentic-serving subdirectory (ADR-019)

**Interaction mechanics:** The operator authors new operation-named capability ensembles as YAML files under `.llm-orc/ensembles/agentic-serving/`. Each ensemble file declares (a) `name` matching the file's basename; (b) `description` (consumed by `list_ensembles()` for natural-language-dispatch matching); (c) `topaz_skill` from the eight-skill taxonomy; (d) `default_task` description; (e) `agents` array specifying the ensemble's composition. The minimum-viable set authored by Cycle 5 BUILD (`code-generator`, `claim-extractor`, `argument-mapper`, `prose-improver`, `text-summarizer`) is the reference shape; additional capability ensembles follow the same pattern. The `.llm-orc/ensembles/agentic-serving/README.md` documents the structure and extension principles.

### Task: Configure the agentic-serving profile file (ADR-019)

**Interaction mechanics:** The operator manages all agentic-serving Model Profiles in `.llm-orc/profiles/agentic-serving-profiles.yaml` — the orchestrator's Model Profile, the tier profiles for the eight Topaz slots, and any deployment-specific profile additions. To swap a model (e.g., upgrade the orchestrator from MiniMax M2.5 to a different cheap-cloud provider, or swap a local-tier model), the operator edits only this single file; `.llm-orc/config.yaml`'s `agentic_serving:` section references profiles by *name*, not by inline model definition.

### Task: Set up the web-searcher backend (ADR-020)

**Interaction mechanics:** The operator obtains an API key for the web-search backend (Tavily by default, or Brave/Exa/Serper if the operator has authored those adapters) and sets two environment variables: `WEB_SEARCH_BACKEND` (default `tavily`) and `WEB_SEARCH_API_KEY` (the operator's key). The `web-searcher` script-agent reads both at startup. The README at `.llm-orc/ensembles/agentic-serving/web-searcher/` documents the setup. Operators choosing a non-default backend author a new adapter Python file under the script-agent's adapter directory.

### Task: Maintain the skill-framework-capability-registry artifact (ADR-021, OD-6)

**Interaction mechanics:** When the operator's deployment serves a new skill framework (RDD already documented; Anthropic Skills, OpenAI Assistants, or others added per deployment), the operator extends `skill-framework-capability-registry.md` with the new framework's decomposition shape, capability consumption table, and library coverage gaps. The registry is **client-side reference**, not orchestrator-consulted lookup — it informs skill-framework authors and operators about which library entries are required for which methodology consumers.


---

## Skill Orchestration User — Cycle 6 updates

### Task: Receive ensemble dispatch under capability-matched natural-language framing (ADR-022)

**Interaction mechanics:** With ADR-022's system-prompt amendment active, the skill orchestration user can emit natural-language requests without explicit ensemble naming and expect the orchestrator to dispatch the matching capability ensemble. The user's request describes the work (e.g., *"Write a Python class CircularBuffer"*) and the orchestrator's amended system prompt commits to **preferring `invoke_ensemble` over direct completion and client-declared tools when a capability match exists**. The skill orchestration user does NOT need to consult `skill-framework-capability-registry.md` for NL-framed dispatch — the orchestrator's `list_ensembles()` consultation + LLM-judgment matching produces the dispatch. The user retains the explicit-naming dispatch shape from Cycle 5's interaction spec when the framework's workflow benefits from pre-specified routing.

**Effectiveness is configuration-conditional.** The amendment's behavioral impact depends on the orchestrator-profile's reasoning shape (per ADR-022 disposition (iii)). Under MiniMax M2.5-free, the amendment is expected to shift NL-framed capability-matched requests toward `invoke_ensemble`. Under `agentic-orchestrator-offline-tools` (qwen3:14b local), the amendment's effectiveness is empirically uncertain — the model may continue to over-delegate to client tools. The skill orchestration user observes the operative routing decision via ADR-023's per-event INFO lines on the operator-terminal destination (or via `execution.json`'s dispatch record post-hoc) and can fall back to explicit naming when needed for a deployment's orchestrator profile.

### Task: Consume typed `DispatchEnvelope` from ensemble dispatch (ADR-024)

**Interaction mechanics:** Each `invoke_ensemble` dispatch returns a typed `DispatchEnvelope` as the chat completion response content. The skill framework parses the envelope structurally:
- **`status`** signals dispatch outcome (`success`, `error`, `timeout`, `partial`); the framework branches accordingly.
- **`primary`** is the human-readable canonical deliverable (always a string; for substrate-routed dispatches per ADR-025, a summary line referencing the artifact).
- **`structured`** (optional) is a typed payload when the dispatched ensemble declared `output_schema:`; the framework parses this directly for ensembles whose downstream consumption needs the typed shape.
- **`diagnostics`** carries `dispatch_id` (for correlation with ADR-023 events), `duration_seconds`, `model_profile`, `tier`, `topaz_skill`, `calibration_verdict`, `audit_findings`.
- **`errors[]`** (optional) carries per-stage errors for partial-failure dispatches.
- **`artifacts[]`** (optional) carries typed artifact references for substrate-routed dispatches.

The envelope replaces Cycle 5's underspecified response shape (output content with implicit structure). Skill frameworks adopting `output_schema:` declarations on their consumed ensembles get structural composition predictability for those ensembles; frameworks consuming envelope output structurally (without `output_schema:`) parse `primary` as a string.

### Task: Read substrate-routed deliverables from session-dir artifact paths (ADR-025)

**Interaction mechanics:** For substrate-routed capability ensemble dispatches (the default for capability ensembles in Cycle 6), the deliverable lives at the path declared in `envelope.artifacts[0].path`, under `.llm-orc/agentic-sessions/<session_id>/<dispatch_id>/<deliverable>.<ext>`. The skill orchestration user's client reads the artifact directly from the filesystem (the local-first deployment shape; clients on the same machine as the serve have filesystem access under the same user account). The envelope's `artifacts[0].summary` carries a one-line description for client UIs that want to render a preview without reading the file.

Cross-machine deployments will use an HTTP artifact-fetch endpoint (`GET /v1/artifacts/<path>`) when that endpoint ships in a future cycle; Cycle 6 covers local-first deployments.

The skill framework's workflow may pass `artifacts[0].path` forward to a downstream sub-task's `input.data` (e.g., the `code-generator`'s artifact path becomes input to a downstream code-review sub-task). The session-dir's per-`dispatch_id` grouping naturally scopes the deliverable's lifecycle to the session.

### Task: Compose sub-task chains using `dispatch_id` correlation (ADR-023, ADR-024 composition)

**Interaction mechanics:** The skill framework can correlate envelope outputs and dispatch events across sub-tasks via the `dispatch_id` correlation identifier (consistent across `envelope.diagnostics.dispatch_id` and ADR-023's dispatch events). For post-hoc review or analytical workflows, the framework reads `execution.json`'s `dispatch_log` key (added per ADR-023) for the session and joins on `dispatch_id`. This gives the framework dispatch-level provenance (which envelope corresponds to which dispatch's timing, verdict, audit findings) without re-implementing correlation logic.

---

## Ensemble Author / Operator — Cycle 6 updates

### Task: Author capability ensembles with `output_substrate` and optional `output_schema` (ADR-024, ADR-025)

**Interaction mechanics:** New or revised capability ensembles in `.llm-orc/ensembles/agentic-serving/` declare two new YAML fields beyond the Cycle 5 shape:
- **`output_substrate`**: `artifact` (default for capability ensembles in Cycle 6; deliverable written to session-dir, envelope carries artifact reference) or `inline` (opt-out; deliverable in envelope's `primary` directly; ADR-004 summarization mandate applies).
- **`output_schema`** (optional): JSON Schema declaring the shape of `envelope.structured`. When declared, the synthesizer agent (or post-dispatch processing) populates `envelope.structured` with the typed payload. When absent, `envelope.structured = None`.

For substrate-routed ensembles, the YAML may also declare:
- **`calibration_substrate_access`** (optional): `summary` (default) or `artifact`. The default has the calibration gate's evaluators read `envelope.primary + artifacts[0].summary`; `artifact` opt-in lets the evaluators read the deliverable's full content (used for `code-generator` because code correctness requires reading the code).
- **`output_retention`** (optional): `session` (default; cleaned up at session close), `durable` (survives session close; operator manually prunes), or `ephemeral` (cleaned up at next turn).

The minimum-viable capability ensembles (`code-generator`, `prose-improver`, `argument-mapper`, `claim-extractor`, `web-searcher`, `text-summarizer`) migrate to `output_substrate: artifact` during Cycle 6 BUILD. Operators authoring deployment-specific ensembles follow the same shape.

### Task: Observe dispatch behavior via the unified event substrate (ADR-023)

**Interaction mechanics:** With ADR-023 active, the operator's serve console emits per-event INFO lines for every dispatch: `dispatch start` / `tier selection` / `calibration verdict` / `audit diagnostic` (when fired) / `dispatch end`. Each line carries `dispatch_id` for correlation. The operator greps/awks the console output for dispatch-level review; long-running session logs are parseable with line-oriented tools.

For deeper analysis, the operator reads `execution.json`'s `dispatch_log` key (added per ADR-023) — an end-of-session summary structured as one entry per `dispatch_id` with the full event set for that dispatch. The operator can join this against per-dispatch envelope artifacts under `.llm-orc/agentic-sessions/<session_id>/<dispatch_id>/` for complete session reconstruction.

`CalibrationSignal` events are excluded from the operator-terminal destination at INFO level by default (they emit at DEBUG); the operator enables verbose-mode logging (`LOG_LEVEL=DEBUG`) for full signal-level review.

### Task: Observe in-flight state via liveness signals (ADR-023)

**Interaction mechanics:** During long-inference dispatches (cloud LLM inference >30 seconds without tool activity), the operator's console emits `INFO: inference wait: elapsed=<seconds> session_id=<id>` heartbeats every `heartbeat_interval_seconds` (default 30). The operator sees mid-stream signal during otherwise-silent waits. When the orchestrator's response stream contains a tool call structure, the serving layer logs `INFO: tool-call emit: tool=<name> dispatch_id=<id>` before dispatching — a liveness anchor distinct from the post-dispatch result line. The operator tunes `heartbeat_interval_seconds` in `config.yaml` per deployment.

### Task: Review session artifacts under the new session-dir layout (ADR-025)

**Interaction mechanics:** The operator inspects a session's artifacts by navigating to `.llm-orc/agentic-sessions/<session_id>/` (the per-session directory groups all dispatches' artifacts and `execution.json` under one tree, rather than the Cycle 5 per-ensemble path). Within the directory, each `<dispatch_id>/` subdirectory contains the deliverable artifact(s) and the dispatch's `execution.json`. The operator can `find` substantive deliverables across sessions, `tar` a session's artifacts for archival, or `rm -rf` a session's directory for manual cleanup.

The Cycle 5 path `.llm-orc/artifacts/agentic-serving/<ensemble>/<timestamp>/` is deprecated; pre-Cycle-6 artifacts remain at the old path until the operator chooses to migrate or remove them.

### Task: Configure orchestrator profile for tool-calling capability (Cycle 6 hardening)

**Interaction mechanics:** With spike γ Cell B exposing that `agentic-orchestrator-offline.yaml` (using `provider: ollama`) cannot serve as an orchestrator because `OllamaModel.supports_tool_calling = False`, the operator wanting a local-orchestrator fallback uses `agentic-orchestrator-offline-tools.yaml` (using `provider: openai-compatible/ollama`, routing through `OpenAICompatibleModel.supports_tool_calling = True` against `http://localhost:11434/v1`). The operator edits `.llm-orc/config.yaml`'s `agentic_serving.orchestrator.model_profile` to `agentic-orchestrator-offline-tools` when running offline-fallback. The original `agentic-orchestrator-offline.yaml` profile remains in the working tree for non-orchestrator roles where tool calling is not required.

### Task: Observe dial-back indicators during PLAY (ADR-025 falsification criteria)

**Interaction mechanics:** During post-BUILD PLAY observation, the operator monitors four indicators for the always-scope dial-back deliberation:
1. **Artifact-substrate latency overhead exceeding 10% of dispatch wall-clock** for under-1KB deliverables (measured via `envelope.diagnostics.duration_seconds` vs. baseline dispatches).
2. **Operator-experience friction** — does substrate-routing feel "in the way" during normal workflows? Recorded as field-notes in the PLAY phase.
3. **Session-directory disk-space cost requiring monthly+ pruning** under typical-deployment usage.
4. **Three or more capability ensembles declaring `output_substrate: inline`** as opt-outs (a pattern signaling the always-scope is producing the boundary judgments substantive-deliverable scope would have made at DECIDE).

If any indicator fires, the operator surfaces the finding in the cycle observing it; a follow-on cycle re-examines the substantive-deliverable scope (rejected at Cycle 6 DECIDE) with the empirical evidence.

---

## Cycle 7 interaction-spec additions (per ADR-026 + ADR-027 + ADR-028 + ADR-029 + ADR-030 + ADR-031 + ADR-032)

Cycle 7 introduces the framework-driven dispatch pipeline on the chat-completions surface (per ADR-027), bounded by two new constitutional invariants — AS-9 (structurally-bounded LLM roles; codified at MODEL boundary) and AS-10 (capability matching from request content alone; codified in ADR-026). The new interaction shapes for existing stakeholders, plus three new role-stakeholders introduced by the pipeline, are specified below. The updates complement (not replace) the existing interaction specifications above; the new shapes apply to the agentic-serving chat-completions surface specifically (per AS-10's scope), with non-chat-completions surfaces continuing to operate per the existing specifications.

### Stakeholder: Tool User — Cycle 7 task refinements

Cycle 7 partitions the Tool User into Population A (tool-call-aware OpenAI-family clients without alternative llm-orc surface access — Aider, Cline, OpenCode, Cursor with base-URL override) and Population B (developer/script clients with alternative-surface access). The existing tasks (Configure the agentic coding tool; Complete a coding task; Observe what the orchestrator is doing; Experience budget exhaustion cleanly) remain operative; the Cycle 7 refinements affect how those tasks manifest under ADR-027's pipeline.

#### Task: Configure the agentic coding tool to use llm-orc as its backend (Cycle 7 refinement — Population A path)

**Interaction mechanics:** Population A (Aider, Cline, OpenCode, Cursor) configures the tool's API endpoint to llm-orc's base URL and selects a model. The `/v1/models` list now surfaces both the configured orchestrator model profiles AND the capability ensembles loaded in the framework's registry (per ADR-032 capability-list discovery; surface candidate (a)). The Tool User does not need to know which list entries are ensembles vs. underlying models — the framework's transparent-endpoint promise (per AS-10) ensures any model identifier the client sends in the request flows through ADR-027's pipeline consistently. Tier B clients (Cline) require operator-side tuning of `requestTimeoutMs` per ADR-031 §Tier B with smoke test verification before deployment.

#### Task: Complete a coding task through the tool's normal workflow (Cycle 7 refinement)

**Interaction mechanics:** The Tool User interacts with their agentic coding tool exactly as they would with any other LLM backend. Behind the endpoint, ADR-027's three-stage pipeline (plan → dispatch → synthesize) runs for each chat-completions request — the routing-planner determines whether to dispatch a capability ensemble or fall through to direct-completion via the synthesizer; the framework executes the plan deterministically; the response-synthesizer produces the user-facing response. The Tool User does not see the pipeline stages directly; the response shape conforms to OpenAI chat-completions semantics. Latency floor at qwen3:8b is ~36s single-step / ~64s chained (per ADR-031); streaming-default clients (OpenCode, Aider, Cline-tuned) experience first-token latency at ~2-3s after synthesizer invocation.

#### Task: Receive honest response labeling (Cycle 7 new task)

**Interaction mechanics:** Each chat-completion response carries honest labeling at three layers (per ADR-032): response header `X-LLM-Orc-Served-By: <ensemble:<name> | direct | direct_fallback>`; body metadata `metadata.served_by: <value>`; content-layer Rule 5 framing on direct-completion responses per ADR-029. The labeling enables the Tool User (or operator inspecting their tool's request history) to verify which path served each request. Population A's degradation signal (configuration dishonesty per Cline #10551 + OpenCode #20859) is structurally prevented — the framework cannot silently disguise direct-completion as ensemble-dispatched. Population A clients that don't surface response headers/metadata see only the content-layer Rule 5 framing on direct-completion responses; clients that do surface metadata see the full three-layer signal.

#### Task: Send `tool_choice` and receive bridge advisory (Cycle 7 new task; Population A `tool_choice`-aware sub-segment)

**Interaction mechanics:** A Population A client (e.g., a client-side skill framework that constructs `tool_choice` shapes) sends a chat-completions request with `tool_choice: {"name": "<ensemble>"}` (or function-call shape). The framework's bridge mechanism per ADR-030 observes the parameter and emits a three-layer advisory: header `X-LLM-Orc-Tool-Choice-Handling: deferred`; body metadata `tool_choice_handling: "deferred"`; on `action: "direct"` responses, content-layer acknowledgment per Rule 5-adjacent. The Tool User (or operator) sees the deferred state honestly; the parameter is not silently stripped (the Cycle 6 footgun is removed). Disposition (i) full implementation lands in a follow-on cycle per ADR-030 §Follow-on trigger.

#### Task: Population B receives structured advisory toward alternative surfaces (Cycle 7 new task)

**Interaction mechanics:** A Population B client (a developer/script using e.g., `requests` library payloads) sending a chat-completions request whose content shape matches Population-B-style patterns receives the response with an additional metadata field: `metadata.population_b_advisory: "<advisory-content>"` pointing toward `llm-orc invoke` or the direct ensemble HTTP API. The advisory is informational; the chat-completions response content carries the synthesizer's normal output. Population B clients can choose to migrate to the alternative surface or continue using chat-completions (the chat-completions surface remains transparent per AS-10).

### Stakeholder: Ensemble Author / Operator — Cycle 7 task additions

The existing tasks remain operative. Cycle 7 adds operator interaction surfaces specific to the framework-driven pipeline and its observability.

#### Task: Deploy and configure routing-planner + response-synthesizer system ensembles

**Interaction mechanics:** Cycle 7 BUILD ships two new system ensembles under the `agentic-` prefix convention (per ADR-019): `agentic-routing-planner.yaml` (per ADR-028) and `agentic-response-synthesizer.yaml` (per ADR-029). Both ship with qwen3:8b model profile as the cheap-tier default. The operator may override the model profile via the ensemble's YAML (per ADR-011's session-boundary config discipline) for deployments where the default does not suit. Operator override risks are operator-borne; the framework's empirical-grounding scope is the default profile (Spike ζ + Spike ε + Spike ε' + Spike μ at qwen3:8b).

#### Task: Configure capability-list discovery surface

**Interaction mechanics:** The operator decides which of ADR-032's three candidate surfaces the deployment exposes for capability-list discovery: `/v1/models` extension (the lowest-cost candidate; capability ensembles appear as model entries with a type marker); sibling endpoint (e.g., `/v1/ensembles`); response metadata (capability list included in chat-completion response metadata for clients that opt in via a request flag). Multiple surfaces may coexist. The operator updates the framework's loaded-ensemble registry (by adding/removing ensemble YAML files); the discovery surfaces reflect the registry dynamically.

#### Task: Monitor `direct_completion_rate` and capability-routing degradation signaling

**Interaction mechanics:** The operator's dashboards / logs surface two new event types from the dispatch event substrate per ADR-032: `direct_completion_fallback` (fires on `action: "direct"`); rolling `direct_completion_rate` metric (percentage of chat-completions requests resulting in `action: "direct"` over a sliding window). High rates surface in operator-terminal destination per ADR-023; the operator interprets them — high rate + Population A-heavy deployment → either the capability library is too narrow for client request shapes (expand the library) or the routing-planner is missing capability matches (tune the planner; check Tier-Router Audit drift criteria per ADR-018).

#### Task: Run Cline integration smoke test before deploying to Tier B clients (Cycle 7 new task)

**Interaction mechanics:** Per ADR-031 §Tier B, before deploying llm-orc agentic-serving against Cline users, the operator runs the integration smoke test specified in ADR-031: send a single-capability NL request matching an installed capability ensemble; verify the response arrives within the operator's configured `requestTimeoutMs` minus 5s headroom; confirm the response includes the expected ensemble output or a direct-completion framing per Rule 5. Repeat for a chained-composition request with `requestTimeoutMs` minus 10s headroom. If the smoke test fails, the operator investigates Cline issue #4308 in their specific Cline build before relying on the tuning. Cursor users on the agentic path are structurally outside scope (plan-mode chat-panel only).

#### Task: Configure tier escalation policy for direct-completion responses (Cycle 7 new task)

**Interaction mechanics:** The operator configures the direct-completion path's tier escalation per ADR-031 §"Tier escalation policy for direct-completion." Default tier is cheap-tier (qwen3:8b empirical baseline). The operator may configure: a deployment-level escalation target (e.g., gpt-4o-mini, Claude Haiku 4.5) for direct-completion responses; Calibration Gate Reflect-trigger criteria thresholds (Rule 1 fabrication signal; Rule 4 rounding-drift; Rule 5 framing absence per ADR-029). Production traffic determines which triggers apply for the deployment.

#### Task: Configure response labeling header conventions (Cycle 7 new task)

**Interaction mechanics:** The operator may configure the response labeling header naming per ADR-032 (e.g., `X-LLM-Orc-Served-By` vs. `Served-By` vs. another framework-chosen convention). The header field name is conventional, not load-bearing — operators with existing observability conventions may choose to align llm-orc's labeling with their deployment patterns. The body metadata and content-layer Rule 5 framing are not operator-tunable (they are structurally specified by ADR-029 + ADR-032).

### Stakeholder: Orchestrator LLM — Cycle 7 role contraction

The Orchestrator LLM's role on the chat-completions surface contracts substantially under ADR-027. The existing tasks (Receive a task; Route an existing ensemble; Compose a new ensemble; Record outcomes; Stay within Budget; Terminate the task cleanly; Receive calibration verdicts; Recover from typed errors) **do not apply to the chat-completions surface under Cycle 7 BUILD**. They continue to apply to any future surface that adopts `OrchestratorRuntime` per ADR-001 + ADR-011's continuing architectural commitment.

The Tranche 4 conformance scan (Finding 2) established that `OrchestratorRuntime` currently has no production caller other than the chat-completions handler being replaced by ADR-027. Under ADR-027's three deferred-to-ARCHITECT dispositions (preserve / wire-CLI / remove), the Orchestrator LLM stakeholder's tasks have:
- **Disposition (a) preserve as architecture-for-future-surfaces:** tasks remain documented capabilities available for any future surface; no active interaction surface.
- **Disposition (b) wire `llm-orc invoke` to use `OrchestratorRuntime`:** tasks become operative on the CLI surface (currently the CLI uses `OrchestraService` directly without an LLM-mediated execution model).
- **Disposition (c) mark for removal as unused code:** tasks become dormant in the live codebase; ADR-001 + ADR-011's commitments preserve them as architectural options for re-introduction in a future cycle.

ARCHITECT selects the disposition; the interaction specification updates accordingly when the disposition is named.

### Stakeholder: Skill Orchestration User — Cycle 7 task validation

The Skill Orchestration User (Cycle 5 introduction) composes a client-side skill framework against the orchestrator's capability library; expects dispatch when a capability slot fits (per product-discovery). Cycle 7 validates this stakeholder's super-objective structurally via ADR-027 + ADR-028.

#### Task: Compose a skill framework against the orchestrator's capability library (Cycle 7 refinement)

**Interaction mechanics:** The Skill Orchestration User authors a client-side skill framework that decomposes higher-level workflows into capability-typed sub-tasks; each sub-task becomes a chat-completions request to the agentic-serving endpoint. Per ADR-027 + ADR-028, the routing-planner ensemble routes each request based on its content alone (AS-10); capability matches dispatch to the matching capability ensemble; non-capability-match requests fall through to direct-completion. The Skill Orchestration User does not configure routing on the framework side (per AS-10's no-client-side-opt-in rule); the planner's routing is the dispatch surface. If the planner systematically misses capability matches the skill framework expects, the operator-observable degradation signaling per ADR-032 surfaces the gap.

#### Task: Discover available capabilities (Cycle 7 new task)

**Interaction mechanics:** The Skill Orchestration User calls the operator-configured capability-list discovery surface (per ADR-032: `/v1/models` extension, sibling endpoint, or response metadata) to discover the deployment's available capability ensembles. Each capability is named (e.g., `code-generator`, `text-summarizer`, `claim-extractor`) with `topaz_skill` metadata. The skill framework's decomposition logic uses the capability list to map workflow sub-tasks to dispatchable capabilities — either by sending NL requests the planner routes (transparent path) or by including explicit ensemble names in request content (the planner honors explicit naming when present).

### New stakeholder: Routing-Planner Ensemble (Cycle 7 introduction; per ADR-028)

**Super-Objective:** Produce a deterministic JSON dispatch plan from chat-completions request content + framework's capability list, such that the framework can execute the plan without further LLM reasoning.

#### Task: Read request content and capability list

**Interaction mechanics:** The Routing-Planner Ensemble receives input of the form `(ORIGINAL REQUEST: messages[], model, tools[]) + (CAPABILITY LIST: ensemble names + descriptions + topaz_skill tags from the framework's loaded-ensemble registry)`. Per AS-10 (ADR-026), the planner does not consume client-side opt-in signals beyond OpenAI-protocol-native fields. The planner reads only the supplied input; it does not query the framework's registry or perform retrieval beyond the supplied capability list.

#### Task: Emit a JSON dispatch plan conforming to the schema

**Interaction mechanics:** The Routing-Planner Ensemble emits JSON conforming to the dispatch-plan schema: `{"action": "dispatch" | "direct", "ensemble": "<name>" | null, "input": "<input string for the dispatched ensemble; required when action=dispatch>", "rationale": "<one-sentence explanation>"}`. Per Spike ζ's empirical floor at qwen3:8b (100% JSON conformance + 100% schema validity), the planner produces conformant JSON across diverse request shapes. The planner does not chain through tool calls, multi-step reasoning, or narration — the role is bounded to producing JSON from a given input (AS-9 structural-bounding property).

#### Task: Defer multi-step composition to the framework chain-heuristic

**Interaction mechanics:** For requests that involve multi-step composition (e.g., search-then-extract chains), the planner emits a single dispatch step; the framework's chain-heuristic (Spike δ's `resolve_input(step.input, results)` pattern) handles the chaining. Production traffic diversity may surface composition shapes the single-step planner + framework chain-heuristic default does not handle; per OQ #21, multi-step planner alternatives are deferred to BUILD/PLAY-informed downstream-phase work.

### New stakeholder: Response-Synthesizer Ensemble (Cycle 7 introduction; per ADR-029)

**Super-Objective:** Produce the user-facing chat-completion response from structured `(ORIGINAL REQUEST + PLAN + DISPATCH RESULTS)` input under strict-fidelity rules, such that the response correctly represents the dispatched work (or honest direct-completion fallback) without confabulation.

#### Task: Read structured input

**Interaction mechanics:** The Response-Synthesizer Ensemble receives input of the form: ORIGINAL REQUEST (full `messages[]` + `model`); PLAN (the routing-planner's emitted `{action, ensemble, input, rationale}`); DISPATCH RESULTS (when `action: "dispatch"` and dispatch succeeded: structured representation of the dispatched ensemble's envelope per ADR-024; when dispatch failed: structured error from the dispatch; when `action: "direct"`: empty/null). The synthesizer does not access dispatched ensemble's substrate paths directly — substrate routing per ADR-025 produces summary-shaped content in the envelope; the synthesizer reads `primary` + `artifacts[0].summary` fields.

#### Task: Produce the chat-completion response under six strict-fidelity rules

**Interaction mechanics:** The Response-Synthesizer Ensemble produces `message.content` + `finish_reason: stop` under the six rules per ADR-029: Rule 1 (use only DISPATCH RESULTS for substantive claims); Rule 2 (do not fabricate Planned-but-not-run results); Rule 3 (do not invent operational metadata); Rule 4 (cite figures verbatim; rounding-drift mitigation per the playbook applies); Rule 5 (honest direct-completion framing when DISPATCH RESULTS is empty); Rule 6 (framework-convention enumeration with hedging in direct-completion mode). Per Spike ε + ε' + μ's n=13 empirical floor at qwen3:8b, 0 fabrications across 4 confabulation modes.

#### Task: Defer to Calibration Gate Reflect on rule-violation signals

**Interaction mechanics:** The Calibration Gate within the response-synthesizer ensemble monitors three Reflect-trigger criteria per ADR-029: Rule 5 framing absence (on `action: "direct"`); Rule 4 rounding-drift (when runtime fidelity check detects drift); Rule 1 fabrication signal (when post-hoc audit detects unsourced content). On Reflect verdict, the Tier-Escalation Router per ADR-015 may escalate the synthesizer to a higher-tier model for retry; the audit per ADR-018 tracks drift for operator visibility.

#### Task: Handle multi-turn continuity via prior-turns-in-input

**Interaction mechanics:** When the chat-completions request includes multi-turn `messages[]`, the framework serializes prior turns into the synthesizer's ORIGINAL REQUEST input. The synthesizer reads the full conversation history and produces responses that correctly reference prior-turn content. Per Spike ε' C1 + C2, multi-turn continuity is preserved when prior turns are in input; native `messages[]` handling at the framework layer is mechanical ARCHITECT-phase work.

---

## Tool User — Cycle 7 loop-back updates (ADR-033, ADR-034)

#### Task: Drive a multi-turn agentic session where ensemble-delegated work lands locally through the client's own tool calls (the parity loop)

**Interaction mechanics:** The Tool User points a tool-driven client (e.g. OpenCode) at agentic-serving and runs a task as they would against a normal single model. Because the request carries client `tools[]`, the surface engages the layer-A loop-driver (ADR-033). Each turn: the loop-driver decides the next agentic step (read / edit / write / bash / finish); when the step needs generated content, it delegates generation to a single capability ensemble (callee); the client-tool-action terminal (ADR-034) returns the deliverable as a `tool_calls` response (e.g. `write({filePath, content})`) the client executes locally; the client's tool result feeds back; the loop continues. The User experiences a real agentic session: their own permission gates, diffs, and tool-result feedback are intact, and the work lands on their filesystem because their client executed it. The deliverable is generated by ensembles, not by the client's configured model. **Parity is behavioral, not latency** — local models are slower, and single-action-per-turn enforcement adds turns; the User accepts the latency tradeoff (governed by which model profiles the ensembles and the loop-driver use). This is the north-star flow; its long-horizon coherence (axis 2) is validated in PLAY/first-deployment, not guaranteed here (ADR-033 recorded risk).

#### Task: Observe that the work was delegated to ensembles, not produced by the loop-driver directly

**Interaction mechanics:** Per the configuration-honesty sub-promise (ADR-032) extended to the tool-driven surface, the User can observe (via response metadata / operator-terminal events) that each `write`/`edit` deliverable's content was generated by a named capability ensemble (the callee), distinct from the loop-driver that decided the step. The distinction matters for the cost-distribution value proposition: generation is delegated to cheap ensembles even when the loop-driver itself is more capable.

---

## Skill Orchestration User — Cycle 7 loop-back updates (ADR-033, ADR-034)

#### Task: Run a long-horizon skill-framework process as a loop participant, not a single-shot request

**Interaction mechanics:** The Skill Orchestration User composes a client-side skill framework (RDD, or any skill standard) that drives a multi-turn agentic session against agentic-serving. The mental model shifts from "send a request, get a synthesized answer" (the single-turn pipeline) to "the endpoint participates in my tool loop": across the session the endpoint decides each turn's next agentic step (layer A) and delegates per-turn generation (layer B) to capability ensembles. The User's skill framework still owns the high-level workflow decomposition client-side; the endpoint owns the per-turn agentic stepping and delegation. Capability matching remains from request content alone (AS-10); the framework is authored without llm-orc knowledge.

#### Task: Rely on grounded per-turn stepping (the endpoint observes each tool result before deciding the next step)

**Interaction mechanics:** The User can rely on the endpoint not committing an action to a value it has not observed: single-action-per-turn enforcement (ADR-033) means each step is grounded by the prior turn's observed tool result. This is the structural guarantee behind grounded driving (Spike τ′); the User does not need to defend against the endpoint batch-planning unobserved state (the Spike τ failure mode), because the framework enforces stepping. Long-horizon trajectory coherence over many turns is the open risk the User's first real runs (PLAY/first-deployment) help validate.

---

## Layer-A Loop-Driver (Cycle 7 loop-back actor, ADR-033)

**Super-Objective:** Drive a tool-driven client's multi-turn agentic loop to task completion, deciding each turn's next agentic step from observed state and delegating content generation to capability ensembles, while staying grounded (one action per turn, no presupposition of unobserved outputs). *(New actor role ADR-033 introduces; occupies the client's "model" seat. Which model fills the seat, and whether a cheap-tier model sustains long-horizon coherence, is ARCHITECT/BUILD selection + the recorded load-bearing risk — kept out of the product-discovery Stakeholder Map until the seat-filler is selected.)*

#### Task: Decide the next agentic step for the turn (single action)

**Interaction mechanics:** Given the conversation, the client's declared tools, and the prior turn's observed tool result, the loop-driver decides one next action: a single client tool call (`write`/`edit`/`bash`/`read`) or finish. The framework enforces single-action-per-turn (truncating any batch the driver proposes, per the Spike τ′ starting prior), so the driver's per-turn decision is always grounded by what it has observed, not by presupposed future state.

#### Task: Delegate per-turn content generation to a capability ensemble (callee)

**Interaction mechanics:** When the chosen action needs generated content (file content for a `write`, a patch for an `edit`), the loop-driver delegates generation to a single capability ensemble invocation, not the full `plan → dispatch → synthesize` pipeline (ADR-033 callee resolution). The driver decides *what* and *which tool*; the ensemble produces the content; the terminal marshals it into the tool call.

#### Task: Compose the destination-keyed deliverable form directive (ADR-035)

**Interaction mechanics:** When delegating content generation for a client-tool deliverable, the loop-driver composes a form directive keyed to the destination tool — `write` → bare file bytes (no markdown fences, no prose, no example block); `bash` → bare command; `edit` → bare replacement content — and includes it in the callee `invoke_ensemble` dispatch input, so the capability ensemble produces the deliverable already in client-tool form. The ensemble's own config stays destination-agnostic: the directive is composed per-dispatch by the framework, never baked into the ensemble's `system_prompt`/`default_task`/`output_schema`. One dispatch produces one client-tool deliverable; multi-file work is decomposed across turns (one `write` per turn), not crammed into a single dispatch. *(The directive's first-try compliance is grounded n=4 single-deliverable at cheap tier (Spike χ.2); sustained-trajectory compliance is a PLAY validation target — Conditional Acceptance, ADR-097.)*

#### Task: Apply the deliverable via the client-tool-action terminal

**Interaction mechanics:** The loop-driver hands the generated deliverable to the client-tool-action terminal (ADR-034), which reads it (via the artifact-bridge from the `SessionArtifactStore` for substrate-routed ensembles, or inline for inline ensembles) and emits the `tool_calls` response the client executes. The driver maps the deliverable to the right client tool (`write` for new files; `edit` after a `read` for in-place changes; `bash` for commands). `edit`-in-place requires reading current file state first.

#### Task: Observe the client's tool result and continue or finish

**Interaction mechanics:** After the client executes the tool and returns the `role: "tool"` result, the loop-driver receives it (the framework routes the follow-up's tool message to the driver rather than dropping it) and decides the next action or finishes with a text completion. The loop continues until the task is complete.

#### Task: Receive the delegation guidance in the user-turn region (ADR-036)

**Interaction mechanics:** The framework composes the delegation guidance into the user-turn region of every seat-filler request — attached to the user task on first turns, as a standalone trailing user-role message after tool-result tails (the C3 form) — never as a framework system message (the system slot measurably loses the attention contest to the client's system prompt: baseline 0/10 vs user-turn 55/55, Spikes ψ/ψ′). The driver's delegation decision is won by composition, not coerced: no model-layer forcing mechanism exists on this stack (`tool_choice` silently ignored; narrowed-role prompts inert; tool-list restriction breaks the turn). The composition is internal to the framework ↔ seat-filler hop; the client never sees it.

## Ensemble Author / Operator — Cycle 7 loop-back #3 task additions (ADR-036)

#### Task: Watch the delegation-rate meter

**Interaction mechanics:** The operator reads `delegation_rate` — the fraction of generation-shaped turns that delegated to a capability ensemble — computed from events alone (generation-shaped classifier denominator × `TurnDecision.delegated_ensemble` numerator) over a 24-hour rolling window. Sustained readings below the provisional 0.9 threshold are refutation evidence for the composition mechanism. The meter routes the response: ~0.85–0.9 is detect-and-retry candidate territory (mechanism mostly working); below ~0.85 or a degrading trend means mechanism diagnosis (client-prompt change, model update, composition regression) — not a retry layer masking the failure. A growing boundary-excluded share (repair-shaped turns, uncovered content domains) signals the denominator's coverage needs re-examination, including whenever new capability ensembles are registered.

#### Task: Re-validate the delegation rate on any seat-filler profile change

**Interaction mechanics:** Before trusting a seat-filler Model Profile swap, the operator records a delegation-rate re-validation — a pre-swap replay run (the Spike ψ harness shape) or a post-swap soak window (≥25 generation-shaped turns per ADR-036). The V3 lever is a (composition × model) property, not a transferable prompt technique: identical composition delegated 1/5 on qwen3.5:9b and 2/5 on mistral-nemo:12b (ψ′ Arm D). Swappability stays structural (config-only, ADR-033 FC); trust is empirical (the recorded run). *(ADR-037 extension: when the judgment seat shares the profile, the same re-validation event also covers the θ-harness judgment arms — one recorded run, both instruments; split seats re-validate per-seat with the matching instrument.)*

## Cycle 7 loop-back #5 task additions (ADR-037)

#### Task (Loop Driver, as actor): Open every trailing turn with the termination judgment

**Interaction mechanics:** On a tool-result tail with no new user task, the loop-driver first dispatches the bare-form termination judgment — framework judge system message, the framework-owned digest (action kind + file path + result per emitted call, joined from the framework's own records), and the deliverable-accounting question; no tools, no client prompt. A COMPLETE verdict returns the judgment's summary (VERDICT line stripped) as the text-only finish turn — the client ends its loop on it. A REMAINING verdict triggers exactly one ADR-036 C3-form action call with the judgment exchange discarded. First turns and new-user-task turns never see the judgment. *(Replay-layer grounding: θ round 2, 29/30 qwen3:14b; the production digest join is the Conditional Acceptance gating condition.)*

#### Task (Tool User): Receive a session that converges

**Interaction mechanics:** When the requested work is complete, the session ends with a brief factual summary of what was produced — no further tool activity, no phantom revisions of finished files. The "endpoint is a model" abstraction now includes natural completion: the user observes the same finish behavior a direct model session would produce. Headless runs terminate on their own; the AS-3 turn cap remains the hard ceiling if judgment repeatedly errs toward continuing (geometric-decay residual, ~0.9 per-cycle termination probability — composed estimate).

#### Task (Operator): Watch the false-continue share alongside the delegation rate

**Interaction mechanics:** The operator reads the finish-policy fields on TurnDecision-family events (turn shape + judgment verdict) to compute how often work-complete tails failed to finish (false-continue) and how often work-remaining tails finished early (false-stop, visible as sessions ending with deliverables missing). The false-continue share is the termination analogue of the delegation-rate meter and shares its event substrate (WP-LB-J). The ADR-036 ≥0.9 soak window is read only after this mechanism lands — earlier reads are distorted by Finding F's phantom-revision inflation.

## Cycle 7 loop-back #6 task additions (ADR-038)

*ADR-038 amends ADR-037's call-2 composition only (the remaining-work anchor on the REMAINING branch). It is invisible to stakeholders as an interaction mechanic — no new surface, no new action — but it changes one observable outcome for the Tool User (multi-deliverable sessions now advance instead of churning) and extends the Loop Driver actor's REMAINING-branch composition. No other stakeholder's tasks change.*

#### Task (Loop Driver, as actor): Anchor the REMAINING action call with what remains

**Interaction mechanics:** On a REMAINING verdict, before composing the ADR-036 C3-form action call, the loop-driver captures the judge's `VERDICT:`-stripped remaining-work statement (the same statement the judge already produced for the verdict) and appends it — followed by the fixed imperative "Produce that next." — to the call-2 trailing region. The seat-filler then delegates the *named* next deliverable rather than re-deciding from the bare conversation. Only the stripped statement + imperative carry forward; the judgment question, digest, and verdict literal stay discarded (ADR-037's context-bounding holds). The anchor is composed only on the REMAINING fall-through — never on COMPLETE, first turns, or new-user-task tails. *(Replay-layer grounding: Spike ρ — judge names the remaining deliverable 20/20; anchored call 2 advances 19/20; content-neutral control 0/10, isolating the remaining-work content as the cause.)*

#### Task (Tool User): Receive a session that converges *across multiple deliverables*

**Interaction mechanics:** Extends the ADR-037 convergence task. A request that decomposes into several deliverables (a module and its tests, several files) now advances through them one per turn — each trailing turn delegates the next unproduced deliverable rather than re-revising the first — and then finishes when all are produced. The user observes the same multi-step-then-stop behavior a direct capable-model session would produce; the framework no longer churns on the first file. The real-client multi-file convergence run (ADR-038's Conditional Acceptance discharge gate, joint with ADR-037's) is the layer-match verification of this experience.

## Cycle 7 loop-back #7 task additions (ADR-039)

*ADR-039 augments the callee dispatch only (the content anchor — produced-sibling signatures routed into what the capability ensemble receives). Like ADR-038 it is invisible to stakeholders as an interaction mechanic — no new surface, no new action — but it changes one observable outcome for the Tool User (multi-file sessions now produce coherent cross-file code instead of inventing nonexistent sibling APIs) and extends the framework's callee-dispatch composition. No other stakeholder's tasks change.*

#### Task (Loop Driver, as actor): Anchor the callee dispatch with the produced siblings' API

**Interaction mechanics:** When delegating a generation that writes a file on a session with already-produced file deliverables, the framework reads the produced siblings (via `SessionArtifactStore.read_deliverable`), extracts their public API signatures, and appends them to the callee dispatch context before the capability ensemble runs. The anchor is framework-sourced from the real files, never the seat-filler's guess (the seat issued no reads in the Finding H run). It fires regardless of callee type — `code-generator` and `prose-improver` alike — and only when there are produced siblings to source (a first file or a no-dependency write carries no anchor). *(Replay-layer grounding: Spike ξ — the signatures anchor moved cross-file-reference resolution from 3/10 [code] / 0/10 [prose] to 10/10 on the cheap qwen3:8b coder; causal isolation B 10/10 vs decoy 0/10 vs filler 1/10 isolates the specific sibling content as the cause; a wrong anchor resolved 0/10, below baseline, so the framework MUST source from the real file.)*

#### Task (Tool User): Receive a multi-file session whose files *cohere*

**Interaction mechanics:** Extends the ADR-038 multi-file task. A request that decomposes into several inter-dependent deliverables (a module, a CLI that imports it, tests that exercise it, a README that documents it) now produces files that cohere — the CLI calls the module's real functions, the tests match the implementation, the README documents real functions — instead of each file inventing a plausible-but-nonexistent sibling API (the Finding H failure: `cli.py` calling `convert_temperature`, the README documenting `fahrenheit_to_kelvin` and a Rankine scale, neither of which exists). The user observes the same whole-project coherence a direct capable-model session would produce. The real-client 5-file trajectory re-run (ADR-039's Conditional Acceptance discharge gate) is the layer-match verification of this experience, and the README is a discharge criterion — prose coherence, not only code.

## Cycle 7 loop-back #7 tail task additions (ADR-040)

*ADR-040 changes how the trailing-turn termination verdict is computed for tasks that name their deliverables (a deterministic `requested ⊆ produced` check that replaces the stochastic judge on that path) and adds persist-once. Like ADR-038 and ADR-039 it is invisible to stakeholders as a new surface or action — but it changes the reliability of one observable Tool User outcome (named-file multi-file sessions now finish only when every requested file is produced, never after the first) and extends the Loop Driver actor's verdict composition. No other stakeholder's tasks change.*

#### Task (Loop Driver, as actor): Decide named-file completeness deterministically

**Interaction mechanics:** On a tool-result tail with no new user task, when the task named its deliverable files, the loop-driver decides COMPLETE/REMAINING by comparing the requested filenames against the produced write paths it already holds, with no judgment-seat call: `requested ⊆ produced` returns the COMPLETE text-only finish; otherwise REMAINING, with `requested − produced` as the next-step anchor (the named-file form of ADR-038's anchor). The requested set is captured once on the first file-naming turn (turn 1's guaranteed-full task) and held session-scoped, first-non-empty-wins (persist-once), so a client that compacts the task out of a later turn's messages cannot collapse the gate to the judge fallback. A task that names no files takes the unchanged ADR-037 judge path. *(Spike σ grounding: the stochastic judge false-COMPLETE-ed after one of five files with no improvement across prompt or judge capability at n≤5 — the produced-only digest is the bottleneck no judge recovers from; the deterministic gate removes the stochastic verdict for named-file tasks.)*

#### Task (Tool User): Receive a multi-file session that finishes only when *done*

**Interaction mechanics:** Extends the ADR-038 and ADR-039 multi-file tasks. A request naming several files now finishes only once every named file has been written — the session never declares itself done after producing one of five (the Finding I premature-finish), and never churns past completion. The user observes the same finish-when-actually-complete behavior a direct capable-model session would produce. Discharged live 2026-06-10 at both the 14b seat/coder and the production 8b coder (6-turn convergence, all five files, deterministic COMPLETE at turn 6). The reliability is scoped to named-file tasks and to existence (the file was written), not adequacy: a written file may still carry a form or quality defect (ADR-035 and coder capability, not this gate — the 8b discharge `cli.py` carried a prose bleed that did not block completion).

## Cycle 7 loop-back #8 task additions (ADR-041)

*ADR-041 closes exactly the adequacy gap the ADR-040 note names (a written file carrying a form defect). It adds a deterministic destination-validity gate at the marshalling boundary and a server-side recovery loop. Like ADR-038/039/040 it adds no new stakeholder surface or action — but it changes one observable Tool User outcome (a session no longer lands an un-parseable file in the workspace) and extends the Loop Driver actor's marshalling-boundary and recovery composition. It also introduces one genuinely new, PLAY-pending experiential question (the short-session-vs-broken-file trade-off). No other stakeholder's tasks change.*

#### Task (Loop Driver / marshalling boundary, as actor): Refuse and recover an invalid deliverable

**Interaction mechanics:** When a delegated deliverable is bound for a client tool with a structurally-checkable destination (`.py` → `ast.parse`, `.json` → `json.loads`), the marshalling boundary validates it before emission. A deliverable that does not parse as what its path claims is *refused* (it never reaches the client). On a refusal the Loop Driver re-dispatches the same destination *within the serving turn* (the coder re-samples), up to a bounded cap; an intermittent bleed self-heals invisibly, a persistent one exhausts the cap and degrades to a dispatch-failure `stop` (the session ends short, never shipping a broken file). Recovery is server-side because a client-facing refusal-as-`stop` ends the client loop, so the ADR-040 next-turn re-delegation cannot run. Prose destinations (`.md`) pass un-inspected — the parse/validity determinism boundary. *(Spike π: corpus parse-check catch 5/5 FP 0; live arm 0 invalid across 5 gated sessions vs baseline 3/5; recovery rescues intermittent bleeds [runs 1/2/5], exhausts on persistent [runs 3/4]; Arm E located the persistent-bleed lever in the coder tier, not the seat.)*

#### Task (Tool User): Never receive a broken (un-parseable) file in the workspace

**Interaction mechanics:** Extends the ADR-040 multi-file task to *adequacy of form*, not only completeness. A named-file session that previously could land a SyntaxError-carrying `cli.py` (trailing prose, or wrong-language content) now either self-heals that file invisibly or finishes short without it — the workspace never receives a file that does not parse as what its extension claims. The common intermittent case is invisible (the recovered file simply appears valid); the persistent case is the one experiential change PLAY must weigh: a *shorter* session (fewer files, none broken) instead of a complete-but-broken one. Whether a short session reads as better or worse than a broken-file diff the client could have rejected is the open experiential question (argument-audit-surfaced); FC-51 `TurnDecision` diagnostics distinguish the turn types. The protection itself is deterministic and tier-independent; full convergence on the hardest files under the cheap tier needs coder-tier escalation (ADR-014), named but not yet wired. *(The gate is env-gated spike code until the BUILD de-gate; this interaction is the production target, validated at the spike layer.)*

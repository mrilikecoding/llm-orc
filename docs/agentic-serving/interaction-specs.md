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

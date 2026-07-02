# ADR-034: Client-Tool-Action Terminal and Artifact-Bridge

> **Superseded by ADR-045 on 2026-07-01 (Cycle-8 clean-slate collapse, AS-11).** The imperative loop-driver serving architecture is retired; its implementation is removed, not adapted. The behavioral requirement this ADR validated is carried forward to the Cycle-8 declarative target per ADR-045's carry-forward table.

**Status:** Superseded by ADR-045 (2026-07-01); formerly Proposed

**Date:** 2026-06-01

---

## Context

ADR-033 establishes the layer-A loop-driver as the control structure for the multi-turn, tool-driven chat-completions surface, with per-turn generation delegated to capability ensembles (callee). This ADR records the *delivery mechanism*: how a generated deliverable reaches the client's local filesystem, and how the surface participates in the client's multi-turn tool loop.

The BUILD-surfaced finding (cycle-status §"BUILD-surfaced finding") was that ADR-027's pipeline emits text only and never `finish_reason: "tool_calls"`, so it cannot give a tool-driven client the work the ensembles produce. The loop-back validated the delivery mechanism empirically (essay-outline-006 Amendment B; spikes π/ρ):

- **The text-only terminal fails parity (Spike π, settled rejection — C8).** Spike π Phase A had a co-located server write the ensemble's deliverable directly to the client's workspace and return a text acknowledgment. Parity FAILED: the client executed nothing, observed no tool result, had no permission gate, no diff, no undo, and operated on an unverified text claim. The single-shot create "succeeded" only coincidentally; a multi-turn session degrades because every turn returns a claim the client never executes. The justification for emitting `tool_calls` is the client's *execution model* (the client drives and observes its filesystem through its own tool calls), not filesystem *geography* (the geography argument dies under co-location; the conclusion survives anyway).
- **The `tool_calls` round-trip achieves parity (Spike π Phase B + Spike ρ).** When the server emits a streamed `write` tool call carrying the ensemble's deliverable, the client executes the write itself, the tool result feeds back, and the loop continues. This is the standard OpenAI agentic loop. Spike ρ confirmed the framework's routing-planner decides to delegate on a real tool-rich OpenCode request and the delegated work returns via the `tool_calls` terminal with parity, together, end-to-end (and the prior tool-rich-client routing suppression did not recur, per AS-10).
- **Production ensembles route deliverables server-side by design (Spike ρ finding F-ρ.1, the artifact-bridge).** Production capability ensembles carry `output_substrate: artifact` (ADR-025): deliverables route to the server-side `SessionArtifactStore`, with `envelope.primary` a summary plus an `ArtifactReference`, not the content inline. So the terminal must read the ensemble's server-side artifact and marshal it into the client tool-call content. This is the *designed-in* form of the disjoint-filesystem problem: even when `llm-orc serve` and the client are co-located, the deliverable lives in the artifact store, not where the client expects it. It does not dissolve under co-location, because ADR-025 routes deliverables server-side by architecture, not incidentally.

Spike π Phase 0 observed the concrete client surface: OpenCode's build agent declares `write {filePath, content}`, `edit {filePath, oldString, newString}`, `bash {command}`, `read`, and others, sends `tool_choice: "auto"` with `stream: true`, and requires streamed tool-call deltas (`delta.tool_calls[].function.arguments` fragments + `finish_reason: "tool_calls"`), not a single JSON object.

## Decision

**The tool-driven chat-completions surface emits client tool-call responses carrying ensemble deliverables, participates in the client's multi-turn tool loop, and bridges server-side ensemble artifacts into tool-call content.**

1. **Tool-call terminal.** When the layer-A loop-driver (ADR-033) decides a turn's action is to apply work to the client (write a file, edit, run a command), the surface emits a streamed assistant response with `finish_reason: "tool_calls"` carrying the appropriate client tool call (e.g., `write({filePath, content})`), conforming to the OpenAI streaming tool-call delta shape Spike π Phase 0 observed. The client executes the tool itself; the surface never writes to the client's filesystem behind the client's back (Spike π Phase A's rejected shape).

2. **Multi-turn loop participation.** The surface consumes the follow-up request carrying the client's tool result (`role: "tool"` message), returns it to the loop-driver, and the loop-driver decides the next action or finishes (a text completion). The surface is a loop participant, not a single-turn responder. This is the multi-turn-loop gap Spike π named as real and unbuilt; ADR-034 commits the surface to participating in the loop, with the loop-driver (ADR-033) making the continue/finish decision under single-action-per-turn enforcement.

3. **Artifact-bridge.** When the delegated capability ensemble routes its deliverable to the server-side `SessionArtifactStore` (`output_substrate: artifact` per ADR-025), the terminal reads the artifact and marshals its content into the client tool-call's content argument. The terminal shape is: `loop-driver decides write -> delegate generation to capability ensemble -> ensemble writes artifact server-side -> framework reads the artifact -> emit tool_call(content = artifact content) -> client executes the write locally -> client tool result feeds back`. For inline-response ensembles (`output_substrate: inline` per ADR-025), the deliverable is read from `envelope.primary` directly; the bridge step is a no-op. The artifact-bridge is the marshalling step from server-side artifact to client tool-call content.

4. **Tool-mapping is loop-driver decision logic.** Which client tool a deliverable maps to (`write` for new files, `edit` for in-place modification, `bash` for command execution) is decided by the loop-driver (ADR-033), conditioned on the task and on observed client state. `edit`-in-place requires the current file state, so the loop-driver reads (a `read` round-trip) before editing. ADR-034 specifies the terminal *mechanism* (emit the chosen tool call, marshal artifact content into it); the *choice* of tool is ADR-033's per-turn decision. *(Scope note: no spike tested the loop-driver making multi-tool-type decisions. τ/τ′ exercised a `bash`+`write` sequence with a passthrough driver; Spike π/ρ exercised `write` only; Spike υ used the full pipeline for generation. Whether the loop-driver reliably selects among `write`/`edit`/`bash`/`read` conditioned on observed state — and whether `edit`-in-place reliably issues the `read` first — is BUILD/PLAY validation work, not settled by these probes.)*

### Naming disambiguation (load-bearing, per loop-back MODEL snapshot Advisory 5)

The **artifact-bridge** named here (the F-ρ.1 artifact-store-to-tool-call-content marshalling, per ADR-025) is **distinct from the "Bridge mechanism" in ADR-030** (the provisional `tool_choice`-deferred handling that emits a `tool_choice_handling: "deferred"` advisory). The two share the word "bridge" and are unrelated concepts. System-design naming, conformance audits, and BUILD work must preserve the distinction: "artifact-bridge" = ADR-034 / ADR-025 deliverable marshalling; "Bridge mechanism" = ADR-030 `tool_choice` advisory.

### Fitness criteria introduced

- **FC (tool-call terminal):** on the tool-driven surface, a turn whose loop-driver action is "apply work to the client" produces a response with `finish_reason: "tool_calls"` carrying a client tool call, not a text-only `Completion`. Refutable: such a turn returning only `ContentDelta` + `Completion` violates this.
- **FC (no server-side client-filesystem write):** the surface never writes to the client's filesystem directly; deliverables reach the client only via client-executed tool calls. Refutable: a server-side write to a client workspace path (Spike π Phase A's shape) violates this.
- **FC (artifact-bridge fidelity):** the content marshalled into a tool call equals the ensemble's deliverable as stored in the `SessionArtifactStore` (or `envelope.primary` for inline ensembles), not a paraphrase or summary. Refutable: a `write` whose content is a summary of the artifact rather than the artifact content violates this.
- **FC (loop participation):** the surface consumes a `role: "tool"` follow-up and produces a next-turn decision; it does not treat the tool-call emission as terminal. Refutable: a surface that closes the response after emitting a tool call without handling the tool-result follow-up violates this.

### Relationship to prior ADRs

- **ADR-033 (layer-A loop-driver):** ADR-034 is the delivery mechanism the loop-driver uses. ADR-033 decides *when* and *which tool*; ADR-034 specifies *how* the deliverable becomes a client-executed tool call.
- **ADR-025 (artifact-as-substrate):** the artifact-bridge reads from the `SessionArtifactStore` ADR-025 established. ADR-034 does not change ADR-025's substrate routing; it adds the read-and-marshal step that moves a substrate-routed deliverable into a client tool call. AS-7 (result summarization default-with-conditional-skip) is unaffected: the bridge marshals the artifact's content into the tool call, it does not summarize it.
- **ADR-027 (dispatch pipeline):** on the single-turn answer-a-question surface, ADR-027's synthesizer produces a text response and no tool-call terminal is engaged. ADR-034's terminal is specific to the tool-driven multi-turn surface (ADR-033). No partial-update to ADR-027 beyond the ADR-033 scoping header.
- **ADR-030 (`tool_choice` disposition):** unrelated mechanism; see the naming disambiguation above. The `tool_choice` bridge advisory and the artifact-bridge do not interact.
- **AS-10 (capability matching from request content alone):** preserved. The terminal emits tool calls as a function of the loop-driver's decision and the ensemble's deliverable; it introduces no client-side opt-in signal. Spike ρ reaffirmed AS-10 holds on the tool-driven surface (the planner routed on request content, indifferent to the client's declared tools).

## Rejected alternatives

### Text-only terminal (the shipped WP-A shape)

The surface returns synthesized text; the client is expected to act on the text.

**Rejected because:** Spike π established this fails parity for a tool-driven client. The client executes nothing, observes no tool result, and cannot distinguish a true claim from a false one. This is the C8 settled rejection. It is the shape WP-A shipped (correctly, as a faithful increment of ADR-027 as written); the loop-back found it is half the surface.

### Co-located server-side direct write plus text acknowledgment

Under co-location, the server writes the deliverable to the client's workspace directly and returns a text acknowledgment.

**Rejected because:** Spike π Phase A ran exactly this and it failed parity. The bytes land, but the client's filesystem view is stale (it did not execute the write), there is no permission gate or diff or undo, and a multi-turn session degrades because the client never observes its own tool results. Delivery (bytes to disk) is solved by co-location; parity is not. Parity rests on the client's execution model.

### Synthesizer-as-terminal (route the tool-call content through ADR-029's response-synthesizer)

The response-synthesizer produces the tool-call content as it produces a chat response.

**Rejected because:** Spike υ measured the synthesizer returning dispatched code verbatim for tool-call content (redundant), and the synthesizer's system prompt is shaped to produce a user-facing chat response (preamble, prose framing), which is a corruption risk for a tool-call content argument that must be exactly the deliverable. The artifact-bridge reads the deliverable directly from the artifact store; the synthesize stage is not in the tool-driven terminal path.

### Defer the artifact-bridge; assume inline deliverables

The terminal reads deliverable content inline from the ensemble response and skips the artifact-store read.

**Rejected because:** production capability ensembles carry `output_substrate: artifact` by design (ADR-025; Spike ρ finding F-ρ.1). Assuming inline deliverables would work only for the spike's text-output stand-in ensemble, not for the production ensembles the surface must serve. The bridge is the designed-in requirement, not an optimization; deferring it would leave the terminal unable to deliver production-ensemble output.

## Consequences

### Positive

- **Parity is achieved (Spike π Phase B + Spike ρ + σ).** A tool-driven client driven against agentic-serving gets a real agentic session: it executes and observes its own tool calls, with permission gates, diffs, and tool-result feedback intact. The north-star "delegate work, apply locally" loop closes.
- **The artifact-bridge respects ADR-025's substrate routing.** Deliverables stay server-side until marshalled into a client-executed tool call; the bridge is the single point that moves substrate content into the client surface, keeping the server-side artifact store as the canonical deliverable location.
- **The delivery mechanism is deterministic where it counts.** The *marshalling* step (reading the artifact and placing its content into the tool-call argument) is framework code, not an LLM generation, so it does not reintroduce the orchestrator-LLM failure mode (PLAY note 22, where the LLM fabricated descriptions of dispatch results instead of chaining them). This is the determinism claim: the bytes-to-tool-call step is mechanical. It is distinct from full-fidelity validation across large or structurally complex deliverables — the artifact-bridge fidelity FC asserts the requirement, but spike evidence is limited to trivially small content (the `hello.py` / `calc.py` round-trips). Fidelity at scale (large files, binary-ish content, encoding edge cases) is BUILD scope, not established here.

### Negative

- **`edit`, `bash`, multi-file, and streaming-token synthesis are unbuilt.** Spike π/ρ validated the `write` round-trip for a single new file. `edit`-in-place (needs a `read` first), command execution via `bash`, multi-file deliverables, and token-streaming the synthesized content are BUILD scope, not yet validated.
- **The artifact-bridge adds a server-side read per delivered deliverable.** Reading the artifact and marshalling it into the tool-call content is an extra step per write turn; for large deliverables this is a content-size cost on the surface. Bounded by deliverable size.
- **Re-introducing `ClientToolCall` emission re-touches the handler that WP-A simplified.** Commit `0a7a822` removed the `ClientToolCall` chunk plumbing; ADR-034 brings it back (the `OrchestratorChunk` union still includes `ClientToolCall`). The streaming tool-call delta shape (Spike π Phase 0) must be reproduced on the new terminal.

### Neutral

- **The OpenAI tool-call contract is standard.** Clients see ordinary streamed `tool_calls` responses and send ordinary tool-result follow-ups; the artifact-bridge is invisible to the client.
- **The single-turn answer-a-question surface (ADR-027) is untouched** by ADR-034; the tool-call terminal engages only when the loop-driver (ADR-033) decides to apply work to a tool-driven client.

## Provenance check

- **Text-only terminal fails parity (C8 settled rejection)**: Spike π Phase A (driver, research log `006b-client-tool-action-terminal.md`) + essay-outline-006 Amendment B §C8 (driver) + product-discovery Assumption Inversion "text-only terminal sufficient -> settled rejection" (driver). Driver chain: loop-back RESEARCH spike + essay + DISCOVER inversion.
- **`tool_calls` round-trip achieves parity**: Spike π Phase B + Spike ρ (drivers, same research log). Driver chain: loop-back RESEARCH spikes.
- **Execution-model (not filesystem-geography) justification**: Spike π necessity verdict (driver) + essay-outline-006 Amendment B (driver) + loop-back DISCOVER gate settled premise (driver, gate note). Driver chain: spike + essay + gate.
- **Artifact-bridge (F-ρ.1)**: Spike ρ finding F-ρ.1 (driver, same research log) + ADR-025 (driver, artifact-as-substrate). Driver chain: loop-back RESEARCH spike + prior ADR.
- **Naming disambiguation (artifact-bridge vs ADR-030 Bridge mechanism)**: loop-back MODEL snapshot Advisory 5 (driver) + domain-model Amendment Log #15 (driver) + ADR-030 (driver). Driver chain: snapshot advisory + MODEL vocabulary + prior ADR.
- **OpenCode client tool surface (write/edit/bash; streaming deltas)**: Spike π Phase 0 observation (driver, same research log). Driver chain: loop-back RESEARCH spike.
- **Multi-turn loop participation gap**: Spike π scope-of-claim "NOT validated" list (driver) + ADR-033 loop-driver continue/finish decision (driver). Driver chain: spike scope statement + companion ADR.
- **Tool-mapping as loop-driver decision logic**: drafting-time synthesis allocating the tool-choice decision to ADR-033 and the marshalling mechanism to ADR-034. The `edit`-needs-`read`-first observation is from Spike π's scope-of-claim list (driver).
- **Rejected alternative — co-located direct write**: Spike π Phase A parity-FAILS verdict (driver). Driver chain: loop-back RESEARCH spike.
- **Rejected alternative — synthesizer-as-terminal**: Spike υ synthesizer-redundancy finding (driver, research log `cycle-7-spike-tau-upsilon-decide-entry-probes.md`). Driver chain: DECIDE-entry probe.
- **Rejected alternative — defer artifact-bridge**: Spike ρ F-ρ.1 production-ensemble substrate finding (driver). Driver chain: loop-back RESEARCH spike.

# ADR-043: Collapse the Dual Serving Surfaces to One Loop-Driven Surface

**Status:** Accepted (Cycle 7 loop-back #9 DECIDE, 2026-06-18). **Supersedes ADR-027** (Framework-Driven Dispatch Pipeline). **Updates ADR-033 §Decision 1** (surface-mode discrimination). Empirically grounded by **Spike ι** (ADR-097 grounding-mechanism path 3 — spike validation), so full Acceptance, not Conditional.

## Context

ADR-033 §Decision 1 established a **two-surface split** on the chat-completions endpoint: a request carrying client `tools[]` engages the layer-A loop-driver; a request with no client tools continues through ADR-027's single-turn `plan → dispatch → synthesize` pipeline. That decision flagged the split as a design commitment to validate, not a measured result — its own provenance note reads: *"the loop-driver handles this by finishing with text, so engaging the driver when tools are present is safe, but the branch is a design commitment to confirm."*

Since that decision, the tool-driven loop has been built and validated end-to-end (the file-action reliability stack: ADR-036 delegation, ADR-037/038 termination + remaining-work anchor, ADR-039 content anchor, ADR-040 completeness gate, ADR-041 destination-validity gate + escalation). The loop finishes turns with text (`FinishTurn`) and delegates per-turn generation to single capability ensembles.

The Dispatch Pipeline (ADR-027) now serves only the no-tools case. Its one distinct capability is the planner-driven **multi-capability plan → dispatch → synthesize fan-out** — multiple ensembles composed and synthesized within a single response, the original answer-a-question vision the loop-back pivoted away from when the north star became the tool-driven serving surface (OpenCode + RDD + ensembles writing local files). Maintaining a second caller for a pivoted-from capability is carrying infrastructure cost.

**Framing note — turns are emergent, not declared.** A client does not pre-declare how many turns a request will take. OpenCode's *build agent* declares the full tool surface (`bash/edit/glob/grep/read/skill/task/todowrite/webfetch/write`, verified in the Spike π Phase-0 captures) and drives a multi-turn, client-executes-tools trajectory; the loop runs as many turns as the work needs until it finishes or the user re-prompts. But OpenCode is **not uniformly tool-driven**: the same captures show a verbatim `title-generator` request carrying **no** `tools[]`. So the toolless path is real OpenCode traffic — lightweight text generation (title/summary), not only external chat clients — and it currently 500s on the half-built pipeline. The collapse unifies the *behavior* (the framework does its full agentic thing on every request); what differs across clients is only what output a client can consume, addressed by F-ι.1 below — and a toolless aux request is served by the loop's finish-with-text (Spike ι `plain_caps` 10/10), fixing the 500.

**Spike ι (2026-06-18, $0 local) discharged the deferred design commitment.** The unified loop answers no-tools requests gracefully: Arm A (structural) 7/7; Arm B (live qwen3:14b, N=10/cell across three cells) 27/30 finish-with-text, 0 errors, and plain questions with capabilities present *never* over-delegated (10/10). The spike also surfaced Finding F-ι.1 (below). Research log: `essays/research-logs/cycle-7-spike-iota-one-surface.md`.

## Decision

**Collapse the two serving surfaces to one. Every chat-completions request routes through the Client-Tool-Action Terminal composing the Loop Driver; the single-turn Dispatch Pipeline is retired.**

1. **One surface.** The surface-mode discriminator's *split* is removed: `_is_tool_driven` no longer selects between two callers. All requests engage the loop-driven Terminal (`v1_chat_completions.py:622` routes unconditionally to `get_client_tool_action_terminal()`). The loop finishes a no-tools request with text (the `FinishTurn` path) and finishes a tool-driven request through its existing per-turn mechanics.

2. **Retire the Dispatch Pipeline.** `dispatch_pipeline.py` and `ensemble_backed_roles.py` are removed; the loop references them only in comments (no hard dependency). The pipeline's planner/synthesizer subtree (ADR-028/029/031/032) loses its production caller — disposition per the backward-propagation sweep (dormant-with-dated-note for any decision that governed the pipeline only; body-immutable record preserved).

3. **F-ι.1 — adaptive deliverable marshalling (Resolution B). Ensemble delegation is uniform; only the output shape adapts.** The loop offers the seat-filler `invoke_ensemble` whenever capabilities are present — regardless of client-tool presence — so the framework does its full agentic thing on every request, including a toolless one. What adapts is the deliverable's **output shape**, decided by the Terminal: when an `ApplyWork` deliverable's destination tool is among the client's offered `tools[]`, the Terminal emits a client tool_call (the client executes it, and the multi-turn trajectory continues); when the request carries no matching client tool (a toolless client), the Terminal marshals the resolved deliverable into a **text completion**. This closes F-ι.1 — a toolless client never receives an un-executable tool_call — while preserving ensemble-backed answers for toolless clients (returned as text). The seat's delegate-vs-finish judgment is benign under B (both branches yield a valid response), so no determinism concern arises. Mechanically, B is a **Terminal-only change** (`_emit_apply_work` branches on whether `outcome.tool_name` is in the request's `tools[]`); the Loop Driver's `_delegation_tools` / guidance composition are unchanged, so FC-58's "guidance never references a tool not offered" invariant is preserved untouched. A toolless request therefore receives the *full* delegation guidance and `invoke_ensemble` offer from the unchanged Loop Driver — by design (the framework delegates), not a stripped-down path. **Prospective dependency:** `_emit_apply_work` today emits a `ClientToolCall` unconditionally, so the "delegate branch is benign" property is a BUILD obligation, not current behavior — the FC (adaptive marshalling) below is the refutable anchor BUILD must satisfy before the determinism argument holds in code.

### Fitness criteria introduced

- **FC (one-surface routing):** every chat-completions request resolves to the loop-driven Terminal; no request routes to a separate single-turn pipeline. Refutable: a request dispatched to a Dispatch Pipeline caller violates this.
- **FC (adaptive marshalling, F-ι.1):** when an `ApplyWork` deliverable's destination tool is not among the request's `tools[]`, the Terminal emits the resolved deliverable as a text completion, never as a client tool_call. Refutable: a response that emits a client tool_call naming a tool absent from the request's `tools[]` violates this.

### Relationship to ADR-027 (superseded)

ADR-027 adopted the dispatch pipeline as the chat-completions surface's primary direction; ADR-033 later scoped it to the no-tools single-turn case. With the loop validated as the universal surface, the pipeline is redundant and is retired. ADR-027 carries a dated `> Superseded by ADR-043 on 2026-06-18.` header; its body is preserved. ADR-027's deferred `OrchestratorRuntime` disposition (its §Decision option (c), remove-as-unused) is now moot at the pipeline layer — the pipeline that replaced `OrchestratorRuntime` is itself retired; the loop-driven Terminal is the surface.

### Relationship to ADR-033 (§Decision 1 superseded; Decisions 2–6 universal)

Only §Decision 1 (surface-mode discrimination) is superseded — the two surface-modes collapse to one. Decisions 2–6 (callee delegation, framework-enforced single-action-per-turn, the grounded-loop Conditional Acceptance, seat-filler swappability, the gate-folded constraints) and the four fitness criteria are unchanged and now govern **all** requests, not only tool-driven ones. ADR-033 carries a dated `> Updated by ADR-043 on 2026-06-18.` header recording that §Decision 1's split is superseded.

### Relationship to ADR-034 (the Terminal is now the sole caller, with an emission branch)

The Terminal becomes the only `_ChatCompletionsCaller`. Resolution B extends the Terminal's emission (ADR-034 §Decision: tool-call emission): `_emit_apply_work` now branches on whether the deliverable's destination tool is in the request's `tools[]` — tool_call when present, text completion when absent. This is the bounded change B requires; the Loop Driver, the Artifact Bridge marshalling, and the FC-48 "no fabricated content" rule are unchanged.

## Rejected alternatives

### Keep the two-surface split (status quo)

The split was a deferred design commitment, not a validated result (ADR-033 §Decision 1 provenance). Spike ι shows the loop subsumes the no-tools case, so the second caller is redundant infrastructure maintained for the multi-capability fan-out — the capability the north star pivoted away from (the pivot is an emergent corpus consensus across `product-discovery.md` and the cycle-status handoffs, not a single named decision). Rejected: carrying two callers for a pivoted-from capability with no demand recorded in product discovery is unjustified cost.

### F-ι.1 Resolution A (deterministic delegation gate-off)

Gate `invoke_ensemble` off when the client offers no write-capable tool, so a no-tools request becomes a pure inline-text turn. Rejected: this is a **functional narrowing** of ADR-027's cost-distribution promise — a toolless capability-matched request would get inline seat-filler text, which is the exact failure mode (orchestrator-LLM answering instead of delegating to the capability ensemble) ADR-027 was designed to remediate. The argument audit's framing section flagged this independently. The determinism advantage A claimed dissolves under B, because B makes the delegate branch valid (text-marshalled), so the seat's delegate-vs-finish stochasticity is benign rather than correctness-bearing. B preserves the capability for a bounded Terminal-only cost.

### Retain the pipeline for the multi-capability fan-out only

Defeats the collapse (still two callers). The only capability B does not reproduce is *multiple* ensembles composed within a *single* response to a *toolless* client; OpenCode composes multiple ensembles across turns, so the north-star path is unaffected. No demand for the toolless single-shot variant is recorded in product discovery.

And — load-bearing for choosing full deletion over a dormant-but-revivable pipeline — **the revival path, if that plain-API "ask-and-compose" surface ever materializes, is not "un-dormant the pipeline" but "wrap a turn-driving agent (OpenCode) in front of the loop."** The spikes already drive OpenCode via CLI, so it is a proven, available pattern, not new infrastructure. This keeps a *single* composition mechanism in the system — the loop, driven by a tool-executing client — and keeps agentic-serving's responsibility crisp: it is the model/loop surface; multi-ensemble *orchestration* belongs in an agent above it, not inside the serving layer (the pipeline blurred that boundary by making agentic-serving be both a model and a composing agent). Dormancy would preserve a second, inferior composition engine the revival path would never reach for. The pipeline is therefore **deleted outright**, not dormant-kept.

## Consequences

### Positive

- One serving surface: one caller, one mental model, less code (`dispatch_pipeline.py` + `ensemble_backed_roles.py` ≈ 506 lines removed, plus the pipeline's ADR subtree made dormant). ORIENTATION and system-design simplify.
- **Ensemble delegation is uniform across all clients** — the framework does its full agentic thing on every request; only the deliverable's output shape adapts (tool_call vs text). Toolless clients keep ensemble-backed answers.
- The F-ι.1 gap (a toolless request receiving an un-executable client tool_call) is closed before it ever shipped.
- The discriminator that ADR-033 flagged for validation is resolved with evidence, not left as a standing design commitment.

### Negative

- B adds an emission branch to the Terminal (`_emit_apply_work`) — a small, bounded change with a refutable FC.
- The single capability genuinely retired is *multiple* ensembles composed within a *single* response to a *toolless* client. OpenCode (the north-star client) composes ensembles across turns, so this affects only toolless clients wanting single-shot multi-capability composition — no recorded demand. The revival path is to wrap a turn-driving agent (OpenCode) in front of the loop, not to restore the pipeline, so the pipeline is deleted outright (see Rejected alternatives).
- ADR-028/029/031/032 (routing-planner, response-synthesizer, direct-completion, tier escalation) lose their production caller and become dormant architecture-of-record. The backward-propagation sweep must add dated handoff notes for two in particular: **ADR-032's configuration-honesty commitment** (the direct-completion tier-escalation path no longer has a live caller on the serving surface) and **ADR-031's direct-completion latency posture** (the residual no-dispatch path it governed is gone — the loop's seat-filler answers toolless requests instead). Knowledge preserved, body-immutable.

### Neutral

- ADR-001/011 (the ReAct execution model) already had no production caller (ADR-027's own conformance note); their status is unchanged by this collapse.

## Provenance check

The collapse derives from **Spike ι** (grounding that the loop answers no-tools requests gracefully) and the cycle's **north star** (the tool-driven serving surface is primary; the multi-capability fan-out is the pivoted-from vision). The discriminator-as-deferred-commitment framing is from **ADR-033 §Decision 1's own provenance note**. **F-ι.1** and **Resolution B** are from Spike ι Arm B (the un-executable-tool_call gap), the practitioner's uniform-agentic-behavior framing (turns are emergent; the framework does its full thing on every request), and the argument audit's framing finding that Resolution A narrows ADR-027's promise. **The grounding is mixed-type, and the distinction matters:** Spike ι *measured* F-ι.1's existence and the graceful-finish property (the empirical filter, ADR-097 path 3, covers the **collapse**); the *selection* of Resolution B over A is **argued, not measured** — it rests on two normative arguments (the ADR-027-narrowing claim, which is audit-sourced from the R1 framing audit rather than an independent corpus decision artifact; and the emergent-turns reframe). The "no demand for the multi-capability fan-out" claim is grounded in `product-discovery.md` and the cycle-status handoffs (not ADR-097, which records grounding paths, not stakeholder preferences).

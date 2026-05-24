# ADR-022: Routing Surface Behavior — System-Prompt Amendment to Elevate `invoke_ensemble` Under Capability-Matched NL Framing

> **Updated by ADR-027 on 2026-05-22.** Per Cycle 7 Tension 14 collapse (domain-model §Concepts "Routing surface behavior" — three dispositions collapsed to "defect to remediate; ADR-027 as remediation") and ADR-027 (Framework-Driven Dispatch Pipeline as Primary Direction for the Chat-Completions Surface), the system-prompt amendment specified in this ADR is **structurally moot for the agentic-serving chat-completions surface**: under ADR-027, the orchestrator-LLM is removed from the routing-decision and post-dispatch-synthesis surfaces; the amendment's intervention (steering the orchestrator-LLM toward `invoke_ensemble` under capability-matched NL framing) no longer applies because the orchestrator-LLM is not the routing decider on this surface. The amendment **remains operative for any future surface that adopts `OrchestratorRuntime`** (the class implementing the ReAct execution model per ADR-001 + ADR-011). **Per the Cycle 7 Tranche 4 conformance scan (Finding 2), `OrchestratorRuntime` currently has no production caller other than the chat-completions handler being replaced by ADR-027** — the `llm-orc invoke` CLI surface and other REST endpoints route through `OrchestraService` directly (`cli_commands.py:28`), not through `OrchestratorRuntime`. The amendment's live codebase surface is therefore the chat-completions handler (until ADR-027 BUILD ships) and any future surface ARCHITECT selects to wire to `OrchestratorRuntime`. If ARCHITECT selects ADR-027's disposition (c) (remove `OrchestratorRuntime` as unused after BUILD ships ADR-027), the amendment becomes dormant code — preserved in version history per this ADR's body-immutable record — until a future cycle re-introduces a ReAct-loop component under ADR-001 + ADR-011's continuing architectural commitment. Cycle 7 DISCOVER 2026-05-21 settled the Tension 14 collapse via the framework-driven dispatch pipeline direction, partially explained by Spike κ D0 (the framework strips `tool_choice` at input, so `tool_choice`-based instructions never reached the model); Cycle 7 DECIDE 2026-05-22 ships the supersession via ADR-027.

**Status:** Updated by ADR-027

**Date:** 2026-05-15

---

## Context

ADR-019 establishes the skill-framework-agnostic orchestrator commitment; ADR-021 establishes the per-capability dispatch contract with two supported dispatch shapes — **explicit ensemble naming** (preferred) and **natural-language prompt** (supported via the orchestrator's `list_ensembles()` consultation + LLM-judgment matching of prompt to ensemble description).

Cycle 5 PLAY notes 1–9 and 20 disclosed under MiniMax M2.5-free + OpenCode that NL framing alone never reached `invoke_ensemble` dispatch; the orchestrator chose direct LLM completion (Cells A, C of spike γ) or client `write_file` (Cell B). The 2026-05-14 follow-on verification findings 3, 5, 6 extended this: even explicit naming + dispatch verb required practitioner intervention on first attempt for sufficiently complex prompts, the orchestrator narrated its own routing as defective post-hoc ("*I short-circuited that by jumping to manual file writes — bad routing on my part*"), and information-finding overhead (40+ minutes wall-clock with practitioner taxing) accompanied recoverable misroutes.

Cycle 6 MODEL Action A renamed the §Concepts entry to **Routing surface behavior**, surfacing three operationally distinct dispositions co-equally: (i) **intended scope** — the observed ordering is the routing surface the system wants; ADR-021's natural-language-supported clause was over-broad; (ii) **defect to remediate** — design intervention restores ensemble-first routing under NL framing; (iii) **configuration-conditional behavior** — the ordering is right under some orchestrator-profile + client-tool-set combinations and wrong under others. The cycle-status DISCOVER entry condition recorded a **held belief-mapping question** for spike γ: *"what would have to be true for the operational preference to be the intended behavior?"*

Spike γ (research log `essays/research-logs/cycle-6-spike-gamma-routing-characterization.md`, executed 2026-05-15) ran four cells:

- Cell A: MiniMax M2.5-free + OpenCode tool-rich + NL → direct LLM completion (17s, 1450 tokens), no dispatch.
- Cell A-explicit: same with explicit `invoke_ensemble code-generator` naming → full dispatch on first attempt (71s, 16939 tokens).
- Cell B: qwen3:14b local (via `agentic-orchestrator-offline-tools` OpenAI-compatible Ollama profile authored to unblock Cell B; the prior `agentic-orchestrator-offline.yaml` uses `provider: ollama` which routes through `OllamaModel` with `supports_tool_calling = False`) + OpenCode tool-rich + NL → client `write_file` tool (57s, 1626 tokens), no `invoke_ensemble` dispatch.
- Cell C: MiniMax M2.5-free + `curl` tool-less + NL → direct LLM completion (13s, 1305 tokens), no dispatch.

Three load-bearing findings emerged:

1. **Two reasonable orchestrators with identical prompt + tools + serve produced different routing decisions.** MiniMax under-delegated (direct completion); qwen3:14b over-delegated (client tool). Neither routed to `invoke_ensemble` under NL framing. The cycle's "preference" is configuration-conditional at the model-selection layer.

2. **The orchestrator system prompt at `src/llm_orc/agentic/orchestrator_config.py:77–126` structurally commits to a precedence honored faithfully under MiniMax M2.5-free.** The prompt instructs: capability *queries* → `list_ensembles`; client-declared tools for filesystem/code actions; do not use client tools for llm-orc-state queries. Nothing in the prompt instructs the orchestrator to prefer `invoke_ensemble` over direct completion for capability-matched NL requests. Under MiniMax, the observed routing is the prompt's commitment honored faithfully. Under qwen3:14b, the prompt's commitment is **not** honored faithfully — the model jumps to client-tool delegation for action-shaped NL framing instead of treating capability-match as a separate routing slot.

3. **Client-tool availability does not change routing decision (Cell A vs Cell C) under NL framing.** Both tool-rich (OpenCode) and tool-less (`curl`) surfaces under MiniMax produced direct completion. PLAY note 20's "client-tool delegation" pattern is conditional on a client tool whose verb matches the prompt framing AND a model that opts to use it — not on mere tool-availability. **Prompt framing changes routing dramatically (Cell A vs Cell A-explicit)** — explicit naming triggered dispatch on first attempt for the simple one-line prompt; the 2026-05-14 follow-on finding 3 "explicit-naming bypass" is conditional on prompt complexity, not categorical.

The four-cell data weakens disposition (i): the system prompt commits to a precedence that **neither tested model honors as written for NL framing** (MiniMax skips capability-match; qwen3:14b mis-routes capability-match to client tools). It strengthens disposition (ii): both models miss ensemble routing under NL framing, with a **concrete remediation target** — a new system-prompt clause that elevates `invoke_ensemble` for capability-matched NL requests. It strengthens disposition (iii): routing varies materially across orchestrator profiles, and the variance is not in a single direction (MiniMax under-delegates; qwen3:14b over-delegates). The dispositions are **not mutually exclusive**.

The product-discovery Skill Orchestration User stakeholder's mental model — *"the orchestrator will route my NL request to a capability ensemble when one matches"* — is not borne out by either tested orchestrator profile under the current system prompt. T14 must either revise the stakeholder mental model entry in product-discovery (operators learn to use explicit naming) or intervene in the system prompt to bring the system to the stakeholder's existing mental model. The cycle commits to intervention; the rationale follows in the Decision section.

---

## Decision

Codify dispositions (ii) and (iii) **jointly** through a system-prompt amendment to the orchestrator's default system prompt (`DEFAULT_ORCHESTRATOR_SYSTEM_PROMPT` at `src/llm_orc/agentic/orchestrator_config.py:77–126`).

### Amendment to the system prompt

Insert a new paragraph between the current "Do not pick a client-declared tool for questions about llm-orc's own state" paragraph (line ~104–109) and the "When you need a client-declared tool, emit it alone in a single assistant turn" paragraph (line ~111–115):

> **When a tool-user's natural-language request maps to a capability ensemble in `list_ensembles()`, prefer `invoke_ensemble` over direct completion AND over client-declared tools.** A capability match means the user's request describes work for which a named capability ensemble exists — e.g., a code-generation request when `code-generator` is in the library; a summarization request when `text-summarizer` is in the library; a search request when `web-searcher` is in the library. The capability ensemble's calibration gate, tier-escalation router, and result summarization are part of the value the request is asking for; direct completion and client tools bypass those. Match the request to an ensemble by reading the ensemble's description from `list_ensembles()` against the user's framing — if a match exists, `invoke_ensemble` is the correct routing. Client-declared tools remain correct for actions on the tool-user's filesystem or code-execution surface **only when no capability ensemble covers the action** — for example, when the user asks to read a specific file path or commit a change, the file-read or git-commit client tool applies; when the user asks to write code, `code-generator` covers it. **Do not pick a client-declared tool merely because the request's verb matches the client tool's verb** (a `write_file` client tool does not displace `code-generator` for code-generation requests). Direct completion is the **residual** when neither an internal nor a client tool applies.

The amendment elevates `invoke_ensemble` from "supported via LLM-judgment matching" (ADR-021's natural-language-supported clause) to "**preferred when match exists**." The matching is still LLM-judgment over the `list_ensembles()` corpus — no new infrastructure — but the prompt's commitment is now explicit. The orchestrator that honors the prompt faithfully will route capability-matched NL framing to ensemble dispatch.

### Partial update to ADR-021's natural-language-supported clause

ADR-021 §"Topaz-skill signal path: how the orchestrator identifies the right ensemble" §"Natural-language prompt" remains structurally correct (LLM judgment over `list_ensembles()` corpus is retrieval-shaped, not evaluative; deployment-time trade-off acknowledged). The clause is **narrowed**: "supported" now means "**preferred when capability match exists**, via the amended system prompt"; the original wording's permissive reading (supported as one option among several) is replaced by the amended wording's commitment (preferred when match exists).

ADR-021 carries a dated `> **Updated by ADR-022 on 2026-05-15.**` header recording the narrowing. The rest of ADR-021 (per-capability dispatch contract, `compose_ensemble` scope, cross-sub-task state, rejected alternatives, falsification trigger, consequences) remains current.

### Effectiveness is configuration-conditional (disposition (iii) acknowledged)

The amendment's behavioral impact varies by orchestrator-profile reasoning shape. Spike γ's data:
- **MiniMax M2.5-free**: the model honored the prompt's existing commitments faithfully; the amendment is expected to shift behavior toward `invoke_ensemble` under capability-matched NL framing. **High expected impact.**
- **qwen3:14b local**: the model did **not** honor the prompt's existing commitments faithfully (over-delegated to client tools for an action-shaped NL prompt); the amendment's effectiveness is **uncertain** — the model's reasoning shape may continue to favor client tools even with an explicit `prefer invoke_ensemble` instruction. **Uncertain expected impact.**

Cross-profile characterization of the amendment's effectiveness is **deferred to BUILD or follow-on PLAY**, not deferred silently. The amendment ships in Cycle 6 BUILD; the first PLAY cycle after BUILD re-runs the spike γ probe across at least Cells A and B with the amended prompt active, and records whether ensemble dispatch under NL framing is restored per-profile. If qwen3:14b continues to over-delegate, disposition (iii)'s configuration-conditional reading becomes load-bearing: per-orchestrator-profile system-prompt overrides may be the cycle's follow-on territory (operators choosing a stricter prompt for orchestrators whose reasoning shape under-honors the amendment).

### Out of scope for ADR-022

- **Per-orchestrator-profile system-prompt overrides.** Mentioned above as follow-on territory; not codified here. Spike γ disclosed cross-profile divergence; the amendment is the cycle's response. If BUILD evidence shows the amendment is insufficient under some profile, per-profile overrides become the next-cycle's deliberation.
- **System-prompt token budget concerns.** The amendment is roughly 110 words; the existing prompt is roughly 350 words. The token-budget impact is small. Operators may continue to override via `agentic_serving.orchestrator.system_prompt` in `config.yaml`. The override mechanism is **session-level** per ADR-011 — operators set a custom prompt per deployment; **per-orchestrator-profile system-prompt defaults** (a different mechanism where each orchestrator profile carries its own default system prompt) are not in current scope. Per-profile defaults are flagged as the disposition (iii) follow-on territory above; the session-level override remains available regardless.
- **Calibration-gate amendments.** The Calibration Gate (ADR-007 / ADR-014) operates on dispatched ensembles' outputs; the amendment changes which requests reach the gate, not how the gate evaluates. ADR-007's scope is preserved.
- **`compose_ensemble` routing.** The amendment does not address `compose_ensemble` framing under NL — composition is a structurally different operation (creating a new ensemble) and its NL-framing characteristics are out of T14's scope.

---

## Rejected alternatives

### Revise the Skill Orchestration User stakeholder mental model (operators learn to use explicit naming)

Update product-discovery to reflect operational behavior: the orchestrator routes NL framing to direct completion or client tools; capability-ensemble dispatch requires explicit naming. Operators are taught to name ensembles explicitly when they want capability dispatch.

**Rejected because:** the stakeholder mental model carries weight independent of agent convenience. The Cycle 5 DISCOVER gate's settled commitment positions the system as accessible via natural language to skill-orchestration users; revising the mental model to require explicit naming is a regression that admits the architecture cannot serve the use case it was designed for. The mental model is also load-bearing for the **Skill Orchestration via Per-Capability Dispatch** value proposition (ADR-021's falsification trigger §"value proposition the agnostic commitment serves") — if explicit naming is the only reliable dispatch path, NL-framed skill frameworks (e.g., RDD's `rdd:*` skills that emit phase-shaped prompts) cannot consume the orchestrator productively without injecting ensemble names into every dispatch turn. The mental-model revision turns the orchestrator into an explicit-dispatch dispatcher dressed in natural-language clothing.

Cycle 5 PLAY note 13 also documented the orchestrator narrating routing defects accurately ("bad routing on my part") — the *model* has the architectural understanding to dispatch correctly under NL framing; what's missing is the **prompt's commitment** to direct that understanding. Asking operators to compensate for a missing prompt clause is solving the problem in the wrong layer.

This option also fails belief-mapping on the practitioner's verbatim defect-or-intended-scope question: revising the mental model is choosing disposition (i) — the operational ordering IS the intended scope. Spike γ's data (Cells A vs B's cross-profile divergence; the model's own post-hoc routing-defect narration) does not support disposition (i) as the load-bearing reading.

### Add a runtime classifier (separate model judges "is this a capability request?")

A pre-classification layer runs before the orchestrator's main loop: each incoming request gets classified by a small model into "capability request," "filesystem action," "general completion," and the classifier's verdict drives routing.

**Rejected because:** this option moves the routing decision out of the orchestrator's reasoning surface into a separate component, contradicting ADR-019's skill-framework-agnostic commitment (the classifier is itself a methodology-shaped layer) and ADR-003's closed five-tool internal surface (the classifier becomes a sixth de facto routing primitive). The classifier also reintroduces an LLM-judgment classification boundary that ADR-015 §(f) ("Skill metadata as runtime classification") rejected — the architecture's commitment is to pre-specified routing via `topaz_skill` metadata + calibration verdict, not runtime classification. ADR-021's NL clause explicitly accepts LLM-judgment-at-capability-selection-boundary as a *narrower* judgment task than ADR-015 §(f) rejected; a pre-classifier inverts that direction.

The classifier also doesn't address what spike γ surfaced: the model HAS the routing logic in its reasoning surface; the prompt doesn't direct it. Adding a classifier doesn't help the prompt direct the model's existing reasoning; it sidesteps the question entirely by routing around the orchestrator.

### Deprecate the natural-language dispatch shape entirely (require explicit naming for all dispatch)

ADR-021's "natural-language prompt" dispatch shape is removed. All dispatch routes through explicit `invoke_ensemble(name, ...)` calls; skill frameworks unable to maintain library-topology knowledge cannot use the orchestrator.

**Rejected because:** same as the mental-model-revision rejection above. The deprecation makes the orchestrator inaccessible to NL-framed skill frameworks. ADR-021's §Consequences §Negative already names the precondition that Topaz-taxonomy-aligned decomposition is required on the skill framework side; requiring explicit naming on top of that is a second precondition that effectively excludes informal skill-framework consumers.

The deprecation also forecloses the disposition (iii) configuration-conditional reading: if some orchestrator profiles honor the amendment well and others don't, the deprecation kills the well-honoring path along with the under-honoring path. Spike γ's data supports keeping NL dispatch supported; the cycle's commitment is to **make NL dispatch honored under capability match**, not remove it.

---

## Consequences

### Positive

- **The Skill Orchestration User mental model is honored at the orchestrator-prompt level.** NL framing routes to capability ensembles when a match exists; operators using natural language without explicit naming get the dispatch they expect. The mental model is preserved as load-bearing rather than revised toward explicit-naming-only.
- **The system prompt's amendment surface is now the operative locus for routing-precedence decisions.** Future routing refinements (per-profile overrides, additional dispatch slots, etc.) work the same way — amend the prompt's commitment, accept that effectiveness varies by orchestrator-profile reasoning shape, and characterize cross-profile in BUILD/PLAY. The methodology-level pattern (system-prompt amendment as the design surface; cross-profile characterization as the empirical verification) is reusable.
- **ADR-021's per-capability dispatch contract is preserved.** The contract's structural commitments (one capability sub-task per request; client-side state; fresh-context property; calibration-gate-per-sub-task) are unchanged. Only the NL-supported clause's permissive reading narrows to preferred-when-match-exists.
- **Disposition (iii) configuration-conditional is acknowledged structurally.** The amendment ships with explicit cross-profile-deferred-to-BUILD/PLAY language, not silent assumption that it works uniformly. Cycle 5 PLAY note 13 (orchestrator self-modeling reliability — orchestrators name defects accurately but predict fix effectiveness optimistically) is honored: the amendment's effectiveness is characterized empirically, not asserted by design.

### Negative

- **The amendment's effectiveness under qwen3:14b is uncertain.** Spike γ Cell B showed the model over-delegating to client tools under NL framing; the amendment adds a `prefer invoke_ensemble` instruction the model may continue to under-honor. If BUILD/PLAY evidence shows the amendment insufficient under qwen3:14b (the local-fallback orchestrator profile), per-profile system-prompt overrides become Cycle 6+ territory. The cycle is committing to an intervention whose effectiveness varies by deployment-time profile choice; operators choosing qwen3:14b may continue to experience NL-framing dispatch defects until per-profile work happens.
- **The amendment adds ~110 words to the orchestrator system prompt.** Token-budget impact is small but non-zero. Operators with strict context-budget constraints may want to override via `config.yaml`; the override surface (ADR-011's per-session config) is preserved for that.
- **Cross-profile characterization is BUILD/PLAY work, not DECIDE work.** The cycle ships ADR-022 with the amendment specified but the cross-profile evidence base not yet populated. The "BUILD/PLAY characterizes" framing is honest about the asymmetric grounding state — the prompt amendment is design-time work; its operational effectiveness is empirical.

### Neutral

- **The `agentic-orchestrator-offline.yaml` profile remains in the working tree alongside the new `agentic-orchestrator-offline-tools.yaml`.** The original profile uses `provider: ollama` (no tool calling); the new profile uses `provider: openai-compatible/ollama` (tool calling). Both serve different ends — the original supports non-orchestrator roles where tool calling is not required; the new profile supports orchestrator role. The cycle does not pick between them; future work may consolidate.
- **The amendment is operator-overridable.** `agentic_serving.orchestrator.system_prompt` in `config.yaml` overrides the default per-deployment, per ADR-011's session-boundary config discipline. Operators wanting a different routing-precedence commitment can express it; the default amendment serves the cycle's commitment to the Skill Orchestration User mental model.

## Provenance check

- **Three operationally distinct dispositions framing (i/ii/iii)**: Cycle 6 MODEL Action A (driver). Driver chain: domain-model-derived §Concepts entry rename.
- **Spike γ four-cell empirical findings**: spike research log `essays/research-logs/cycle-6-spike-gamma-routing-characterization.md` (driver). Driver chain: empirical evidence from same-cycle spike.
- **System prompt as the operative amendment surface**: spike γ system-prompt finding (driver) + Cycle 5 PLAY note 13 orchestrator-self-modeling-reliability (driver). Driver chain: spike-derived + prior-cycle PLAY-derived.
- **Joint codification of dispositions (ii) and (iii)**: drafting-time synthesis. The spike data weakens (i) and strengthens (ii) and (iii); the joint codification — amendment as (ii); cross-profile deferral as (iii) — is the drafting-time composition of the empirical signals.
- **Partial update to ADR-021's NL clause**: ADR-021 (driver) + the amendment scope. Driver chain: prior-ADR-derived; the narrowing's specific wording is drafting-time synthesis bridging the existing clause to the amended commitment.
- **Cross-profile characterization deferred to BUILD/PLAY**: drafting-time synthesis honoring the methodology's empirical-grounding discipline. The deferral is structurally honest (cross-profile evidence is BUILD/PLAY work) rather than silent (the alternative would be asserting the amendment works uniformly without evidence).
- **Skill Orchestration User mental model as load-bearing**: product-discovery (driver) + Cycle 5 DISCOVER gate settled commitment (driver). Driver chain: product-discovery-derived + prior-cycle-gate-derived.
- **Falsification path → per-profile overrides**: drafting-time synthesis flagging the natural next-cycle's design surface if the amendment is insufficient under some profile. Not an ADR commitment; flagged as available extension path consistent with ADR-021's falsification-trigger pattern.

# ADR-035: Client-Tool Deliverable Form Contract (Boundary-Composed)

> **Updated by ADR-041 on 2026-06-11.** §Decision 4's detect-and-refuse gate —
> held here as a speculative escalation "if PLAY shows residual non-compliance" —
> is promoted to a committed **deterministic destination-validity gate** (parse/
> validate the deliverable against its destination path's claimed type), grounded
> by Spike π *before* PLAY. The form-seam **protection** Conditional Acceptance is
> *design-discharged* (the mechanism is model-compliance-independent and validated
> live) but **not yet install-discharged** — ADR-041's gate is env-gated spike code
> until the BUILD de-gate, so the production path does not run it by default. A
> wrong-form deliverable is caught structurally (the gate inspects bytes) once
> installed, not relied upon not to occur. ADR-041 also adds the server-side
> re-dispatch recovery this ADR did not anticipate — the live arm showed a
> client-facing refusal ends the OpenCode loop, so §4's "degrade to a
> dispatch-failure completion" does not self-heal on its own. The boundary
> directive (§Decision 1) stays the *primary* form mechanism; the rest of this ADR
> is unchanged. See ADR-041.

**Status:** Proposed — Conditional Acceptance (ADR-097), Updated by ADR-041 (form-seam protection design-discharged 2026-06-11; install-discharge is BUILD); axis-2 / breadth validation pending PLAY

**Date:** 2026-06-03

---

## Context

The Cycle 7 loop-back BUILD reached parity at the *mechanism* level: the WP-LB-G real-OpenCode run had the seat-filler delegate (`invoke_ensemble` → `code-generator`), the deliverable route to the `SessionArtifactStore` (ADR-025), the Artifact Bridge (ADR-034) marshal it, and the surface emit `finish_reason: "tool_calls"` with a `write` the client executed locally. But the `write` carried unusable content: the raw ensemble result envelope (`{"results": {coder, critic, synthesizer}, "synthesis": null, …}`), not bare file content. This is **Finding D** (cycle-status §"BUILD-surfaced finding: deliverable-shape").

Spike φ (`essays/research-logs/cycle-7-spike-phi-deliverable-shape.md`) and Spike χ (`…-spike-chi-deliverable-shaping.md`), both $0 local, decomposed Finding D into three separable layers and grounded the contract design:

- **D1 — extraction (BUILD).** `_extract_synthesizer_text` succeeds only for a populated top-level `synthesis` (never set in the dependency-based model — `results_processor.py:21`) or a single-agent ensemble; every multi-agent capability ensemble falls through to `json.dumps(raw_result)`. code-generator (3 agents) is the only multi-agent substrate-routed capability ensemble shipped, so it is the only one currently bitten, but the gap is structural. D1 is a BUILD fix shaped to this ADR's outcome.
- **D2a — the declared contracts are inert at execution.** `default_task` has zero runtime reads anywhere in the package (grep-confirmed; both the MCP-invoke and orchestrator-dispatch paths send only the caller's input); `output_schema` (ADR-024) is documentary. Only the agent `system_prompt` plus the caller-supplied dispatch input reach the model. This **refines the mechanism Spike β identified, without disturbing its headline finding.** Spike β's headline — that composition assumptions live in the orchestrator's reasoning surface, not the typed contract — survives intact. ADR-024 described the drift mechanism as "the orchestrator hand-writes `input.data`, overriding `default_task`"; Spike φ adds the mechanistic precision that `default_task` does not reach the model at all, so it is not "overridden" so much as absent — the dispatch input is the *only* contract surface that reaches the model. That precision is what makes boundary-injection the natural lever.
- **D2b — the deliverable's form targets a human reader.** code-generator's agents carry `system_prompt`s asking for client-readable markdown ("Format code with appropriate fenced blocks"; synthesizer: "Output exactly what the client should see"). The model obeys, but the form is wrong for a `write` body. **ADR-034 already named this exact risk** when it rejected synthesizer-as-terminal ("a corruption risk for a tool-call content argument that must be exactly the deliverable"). ADR-034 did not, however, offer a mitigation for the deliverable's *form* — it promised faithful *marshalling* (the artifact-bridge-fidelity FC: "marshal exactly what is stored") and left open the question of what form the stored content should take. Finding D shows the store holds the same prose-framed content, so fidelity alone delivers prose; ADR-035 fills the open question (the form contract), upstream of the marshalling ADR-034 specified.

Spike χ grounded the design fork (where to enforce the form):

- **Bridge-side deterministic shaping is fragile.** Multi-fence ambiguity appeared in 2 of 3 unconstrained code-generator outputs (the model's default habit is implementation + a separate "Example Usage" fence). No robust general extraction rule exists; a first/largest-fence heuristic misfires on test files, multi-file deliverables, and prose deliverables.
- **A bare-output directive in the dispatch input is reliable.** With an explicit "output only the bare bytes" instruction, qwen3:8b emitted exactly the file content across n=4 single-deliverable types (function, larger module with dataclass + class + 3 methods, `bash` command, structured claim bullets — Spike χ + χ.2 + φ Run 2). The cheap model reliably *produces* the right form when the directive reaches it; shaping its unconstrained output after the fact does not.
- **Multi-file in one dispatch breaks** (Spike χ-P6): asked for two files at once, the model improvised an unparseable `filename\ncontent` convention and re-added prose. This is a *granularity* signal, not a contract-reliability gap.

ADR-024 deferred the constructive fix: "Eliminating the orchestrator's reasoning-surface composition role would require narrowing the orchestrator's role from prose-integrator to chain-selector … future-cycle territory if/when composition predictability becomes the cycle's central question." Cycle 7 performed that narrowing (ADR-027 removed the orchestrator-LLM from the dispatch path; ADR-033 made the loop-driver/seat-filler the per-turn decider). The controlled input-authorship surface ADR-024/Spike β identified as the *drift source* is, post-narrowing, the framework-owned *lever* for the contract.

**Seam framing (practitioner, DECIDE gate).** The form problem is a *seam* cost specific to the delegation architecture. In a single-model agentic flow, the model that decides to call `write` also generates the content — in-context, natively destination-aware — so form coherence requires no mechanism. Ensemble delegation splits decider (loop-driver) from producer (capability ensemble) across a dispatch seam; the destination-awareness a single model gets implicitly must be carried across that seam explicitly. ADR-034 restored the *execution-model* half of parity (the client executes its own tools); ADR-035 restores the *generation-form* half (the content filling a tool argument is tool-argument-shaped). A third seam — *semantic coherence*, whether delegated content fits the surrounding project (imports, conventions, coherence with prior turns) — is distinct and **not addressed by this ADR**: it lives with the seat-filler's dispatch-input composition (axis-2 territory, OQ #27) and ensemble quality (declared orthogonal at DISCOVER), and is a PLAY observation target where FC-51 `TurnDecision` diagnostics distinguish wrong-*content* turns from wrong-*action* turns. Parity of mechanism is claimed; parity of content quality with a frontier single model is not.

## Decision

**For deliverables bound for a client tool, the form contract is composed at the marshalling boundary, keyed to the destination tool, and delivered to the capability ensemble through the dispatch input. The ensemble produces the deliverable in the target form; capability ensembles remain destination-agnostic.**

1. **Boundary-composed, destination-keyed form directive.** When the Loop Driver / Client-Tool-Action Terminal (ADR-033/034) delegates generation whose deliverable is bound for a client tool, the marshalling boundary composes an output-form directive for that destination tool — `write` → bare file bytes (no fences, no prose, no example block); `bash` → bare command; `edit` → bare replacement content — and injects it into the callee `invoke_ensemble` dispatch input. The capability ensemble produces the deliverable in that form. The Artifact Bridge then marshals it unchanged (ADR-034 fidelity FC preserved).

2. **Capability ensembles stay destination-agnostic.** The directive is injected per-dispatch by the framework; ensemble YAML is not coupled to file-production (no `submit_file` baked in, no destination knowledge in `system_prompt`/`default_task`). This preserves the ADR-025 library principle: a capability ensemble is dispatchable by any skill framework for any destination.

3. **Granularity invariant: one dispatch → one client-tool deliverable.** A single `invoke_ensemble` dispatch produces the content for a single client tool call. Multi-file work is the Loop Driver's across-turn decomposition (one `write` per turn), which is callee-native across-turn composition (ADR-033 F3-1) — not one dispatch producing N files. Spike χ-P6 is what violating this granularity looks like.

4. **Deterministic shaping is a defense-in-depth backstop, not the contract — and its escalated form is detect-and-refuse, not extract.** The Artifact Bridge MAY apply a conservative normalization (strip a single enclosing code fence if one slips through) as a safety net. The backstop's escalated form, if PLAY shows residual non-compliance, is a **fail-safe detection gate**: when the marshalled deliverable is clearly non-bare (multiple fences; prose-scaffolding markers), the bridge refuses to emit the tool call and degrades to a dispatch-failure completion (or one re-dispatch) rather than sending garbage to the client. Detection-and-refusal only has to *recognize* a clearly-wrong deliverable; it never attempts heuristic extraction from multi-fence output (Spike χ F-χ.1 — that path is fragile). Escalating to the detection gate is the first response to PLAY non-compliance, not a redesign.

5. **D1 extraction (BUILD, shaped to this outcome).** The substrate-write / deliverable-extraction stores the terminal deliverable — the last *successful* agent's output — not the raw result dict, and falls back to the last successful agent when the terminal node fails (the synthesizer can time out — Spike χ F-χ.3). The where-sub-fork (executor-side population where `depends_on` is known, vs. envelope-side terminal-node selection) is a BUILD scenario-group decision.

### Why a hard form-guarantee is neither available nor required (DECIDE-gate exchange)

**Not available.** Deliverable *form* is not structurally enforceable the way single-action-per-turn is (ADR-033's enforcer mechanically truncates a batch; nothing mechanically converts markdown to bare code without the fragile heuristics rejected above). The "harder" mechanisms do not actually deliver a guarantee: `output_schema` reject-and-retry is *more model-trust attempts* plus latency, and "is this bare code?" is not cleanly schema-checkable — the validation itself inherits the shaper's heuristics; a `submit_file`-shaped tool guarantees a *slot*, not the form of the content in the slot (fences can appear inside the argument), while paying the destination-coupling cost. The real choice is between probabilistic mechanisms at different costs, not between guaranteed and probabilistic.

**Not required.** The failure mode is visible and rejectable, not silent: per ADR-034's execution model, the *client* executes every `write` through its own permission gate, diff, and undo. A wrong-form deliverable surfaces as a rejectable diff (for a client operating with its permission gate active — an auto-accepting client weakens this argument and strengthens the case for the detection gate); it does not silently corrupt the workspace (that was the rejected co-located-write shape). The bounded failure cost is what makes the lighter mechanism appropriate; on a surface without client-side execution affordances, the detection-gate backstop (decision 4) would be warranted from the start.

### Relationship to prior ADRs

- **ADR-024 (common I/O envelope) — partial update.** ADR-024 made `output_schema` advisory and rejected synthesizer-side enforcement (Candidate B-strong) for the *composition* case, on Spike β's "drift is upstream of the synthesizer" reasoning. ADR-035 does not disturb that for inter-ensemble composition. It carves out the **client-tool-deliverable path**: for a deliverable bound to a client tool, the form contract is not advisory and not carried by the inert `default_task` — it is composed at the boundary and delivered through the dispatch input. ADR-024 carries an `> **Updated by ADR-035**` header for this carve-out.
- **ADR-034 (client-tool-action terminal + artifact-bridge) — completes, does not change.** ADR-034's decisions stand: the bridge reads the server-side artifact and marshals it faithfully. ADR-034's synthesizer-as-terminal rejection *named* the prose-framing risk but specified only faithful marshalling, leaving open what form the stored content should take. ADR-035 fills that open question — the deliverable is produced in client-tool form at the source (boundary directive), so the content the bridge faithfully marshals is already bare.
- **ADR-025 (artifact-as-substrate) — unchanged.** Substrate routing is untouched; the directive changes only what the dispatched ensemble produces, which is then stored and marshalled as before.
- **ADR-033 (layer-A loop-driver) — composes.** The directive is composed where the Loop Driver already decides the destination tool (decision 4 of ADR-034 / ADR-033 per-turn tool choice). The granularity invariant is ADR-033 F3-1 across-turn composition made explicit for deliverables.
- **AS-9 / single-step enforcement — consistent in spirit.** The framework *guarantees the directive is present* (structural), while relying on documented model compliance to produce the form (Spike χ, n=4). This is lighter than hard schema-retry enforcement and structurally framework-owned, but it is model-compliance-dependent, not a hard structural guarantee — hence Conditional Acceptance.

### Conditional Acceptance (ADR-097)

This ADR commits the mechanism on spike-validation grounding (Spike φ/χ/χ.2 + the WP-LB-G real-client run). The grounding-mechanism path is **spike validation** (path 3) reinforced by **real-client evidence** (WP-LB-C/G). The pending validation, designated for PLAY / first deployment per ADR-033 §6b axis-2:

- Sustained form-compliance over long multi-turn trajectories (the axis-2 regime; n=4 covers single dispatches, not trajectories).
- The granularity invariant holding under a real Loop Driver that must decompose multi-file work across turns rather than cram it into one dispatch.
- Escalated-tier behavior (all spike evidence is cheap-tier qwen3:8b).
- **Precondition, not an independent target:** the form contract exercises only when the Loop Driver actually delegates (`invoke_ensemble`). Delegation reliability (Finding B's resolution, WP-LB-G) has held for one real-client run; PLAY validation of the form contract is gated on delegation continuing to fire across prompts and clients.

If PLAY shows the boundary directive is insufficient for compliance, the escalation order is: (1) escalate the backstop to its detect-and-refuse gate (decision 4); (2) `output_schema`-as-enforcement reject-and-retry for the client-tool path; (3) a frontier seat-filler (ADR-033 §6b). FC-51 `TurnDecision` instrumentation distinguishes a wrong-*form* deliverable (this contract) from a wrong-*action* turn (driver/split) — and, per the seam framing above, from a wrong-*content* turn (the semantic-coherence seam this ADR does not address).

## Rejected alternatives

### Bridge-side deterministic shaper as the primary contract

Let capability ensembles produce their natural output and have the Artifact Bridge deterministically extract the file body (e.g., pull the fenced code block).

**Rejected because:** Spike χ F-χ.1 — multi-fence ambiguity in 2/3 unconstrained outputs, with no robust general rule. A first/largest-fence heuristic rescues the common case but misfires on test files (the wanted code may not be largest), multi-file deliverables, and prose deliverables (no fence). Extraction-from-unconstrained-output is the fragile pole; it survives only as a backstop (decision 4), not the contract.

### Static ensemble-side coupling (a `submit_file`-shaped synthesizer baked into the ensemble)

Give capability ensembles a `submit_file(content=…)` tool (or a bare-output `system_prompt`) so the deliverable is structurally bare at the source.

**Rejected because:** it couples the ensemble to file-production, eroding the destination-agnostic reusability ADR-025 capability ensembles depend on (code-generator is "dispatchable by any skill framework"; it should not statically know its output becomes a local file). The same bare-content reliability is obtained without the coupling by injecting the directive at the boundary per-dispatch (decision 1). Spike χ shows the directive-in-input path is reliable (n=4).

### `output_schema`-as-enforcement with reject-and-retry at the synthesizer

Make `output_schema` enforced for client-tool deliverables: validate the synthesizer output, reject-and-retry on mismatch.

**Rejected because:** heavier than the evidence warrants. Spike χ shows the cheap model complies with a directive on the first try across n=4 types; a retry loop adds latency and failure modes for a problem the directive already solves. It is also the Candidate B-strong shape ADR-024 rejected for enforcing "at a layer that isn't the source." The boundary directive places the contract at the source the model actually reads (the dispatch input). Reject-and-retry is held as a PLAY escalation (Conditional Acceptance), not the first mechanism.

### Wire the ensemble's `default_task` through to the model unchanged

Fix D2a by making the inert `default_task` reach the model.

**Rejected because:** `default_task` is the ensemble's *generic* task description, destination-blind. It does not encode "bare file bytes for a `write`" vs. "bare command for a `bash`." Wiring it through would fix the inert-config bug but not D2b (the destination-specific form). The boundary directive subsumes it: it delivers a destination-keyed contract through the same surface (dispatch input) that a wired `default_task` would use. (The inert-`default_task` finding is recorded for the broader I/O-contract hygiene, but the client-tool path does not depend on resolving it.)

## Consequences

### Positive

- **Parity yields *usable* deliverables, not just executed tool calls.** The north-star loop ("delegate work, apply locally") produces files that run, closing the gap WP-LB-G surfaced.
- **Capability ensembles stay destination-agnostic** — the ADR-025 library principle is preserved; no ensemble is rewritten for file-production.
- **Lighter than schema-retry; the structural guarantee is bounded and explicit.** What the framework structurally guarantees is the *presence* of the destination-keyed directive in every client-tool dispatch (the controlled lever ADR-024 anticipated). It does *not* structurally guarantee the deliverable's *form* — that rests on documented model compliance (Spike χ, n=4 first-try). Directive-presence is enforced; form-compliance is relied upon. No retry loop in the common path.
- **Composes cleanly with ADR-034.** The bridge's fidelity FC is unchanged; ADR-035 only secures the precondition (the deliverable is in client-tool form before marshalling).

### Negative

- **Model-compliance-dependent, not hard-enforced.** Unlike single-step enforcement (a hard structural truncation, ADR-033), the form depends on documented compliance. Backstopped (decision 4) and PLAY-validated (Conditional Acceptance), but not a structural guarantee — an honest gap from the cycle's "structural, not model-assumed" ideal.
- **Multi-file requires Loop Driver across-turn decomposition, which is unbuilt.** The granularity invariant pushes multi-file work into multi-turn loop-driving (ADR-034 §Negative already lists multi-file as BUILD scope). Until the Loop Driver decomposes reliably, multi-file deliverables are not served.
- **Grounding is n=4 single-deliverable, cheap-tier, and pipeline-narrow.** Three of the four compliant samples ran through `code-generator`'s pipeline (the fourth, φ Run 2, through claim-extractor); breadth across other capability ensembles, long trajectories, escalated tiers, and `edit`/`bash` at scale is PLAY/first-deployment work, not settled here.

### Neutral

- **The directive is invisible to the client.** The client sees an ordinary `tool_calls` response; the boundary directive lives entirely in the server-side dispatch.
- **The inert-`default_task` finding (D2a) is recorded but not load-bearing for this path.** Whether to wire `default_task`/`output_schema` through for the general (non-client-tool) case is separate I/O-contract hygiene, left to a future cycle.

## Provenance check

- **Finding D and its three-layer decomposition**: Spike φ + WP-LB-G real-client run (drivers; research logs `cycle-7-spike-phi-deliverable-shape.md`, `cycle-7-wp-lb-c-opencode-validation.md`). Driver chain: loop-back BUILD spike + real-client validation.
- **`default_task` inert; dispatch input is the only contract surface (refines Spike β's mechanism; its headline survives)**: Spike φ Part 3 grep + path trace (driver). Driver chain: loop-back DECIDE-entry spike.
- **Boundary-directive reliability (n=4) vs. deterministic-shaping fragility (2/3)**: Spike χ + χ.2 (drivers). Driver chain: loop-back DECIDE-entry spike.
- **Granularity invariant (one dispatch → one deliverable; multi-file across turns)**: Spike χ-P6 (driver) + ADR-033 F3-1 across-turn composition (a *recorded concession* / wrapper-residual watch point in ADR-033, not a driver finding — the granularity invariant is an inference consistent with F3-1, and the structured-multi-file alternative was not probed; see Framing note for the gate). Driver chain: spike + prior-ADR concession.
- **ADR-024 carve-out / role-narrowing-now-done**: ADR-024 §"future-cycle territory" deferral (driver) + ADR-027/033 role narrowing (drivers, prior ADRs). Driver chain: prior ADRs.
- **ADR-034 named the prose-framing risk; its store-read mitigation is incomplete**: ADR-034 synthesizer-as-terminal rejection + artifact-bridge-fidelity FC (drivers, prior ADR) read against Spike φ's stored-artifact evidence (driver). Driver chain: prior ADR + spike.
- **Rejection of bridge-side shaper as primary**: Spike χ F-χ.1 (driver). Driver chain: loop-back DECIDE-entry spike.
- **Rejection of static ensemble coupling**: ADR-025 destination-agnostic library principle (driver, prior ADR) + Spike χ directive-reliability (driver). Driver chain: prior ADR + spike.
- **Rejection of schema-retry as primary**: Spike χ first-try compliance (driver) + ADR-024 Candidate B-strong "enforcement at the wrong layer" (driver, prior ADR). Driver chain: spike + prior ADR.
- **Conditional Acceptance shape and PLAY escalation order**: drafting-time synthesis applying ADR-097's grounding filter to the n=4-but-not-trajectory evidence, composing ADR-033 §6b axis-2 escalation levers. Design-time scoping judgment, spike-grounded but not itself a single driver finding.
- **Seam framing (form seam vs. context seam; two-halves-of-parity)**: practitioner framing at the DECIDE gate (driver — user framing at gate, 2026-06-03). The semantic-coherence-seam scope boundary and its FC-51/PLAY disposition compose the practitioner's framing with ADR-033 §6b and the DISCOVER ensemble-quality-orthogonal declaration.
- **"Hard form-guarantee neither available nor required"**: DECIDE-gate exchange (driver — the practitioner's guarantee-justification challenge + the agent's mechanism analysis). The bounded-failure-cost argument rests on ADR-034's execution-model decision (driver, prior ADR). The detect-and-refuse escalated backstop is gate-exchange-derived synthesis.

# RDD Cycle Status — Agentic Serving (Scoped)

**Artifact base:** `docs/agentic-serving/`
**Plugin version at cycle open:** v0.8.5
**Migration version:** 0.8.5 (`housekeeping/.migration-version`)

## Cycle Stack

### Active: Cycle 6 — Ensemble contract + observability + routing (post-Cycle-5-PLAY pickup; expanded from initial two-axis framing per 2026-05-14 follow-on verification)

**Cycle number:** 6
**Started:** 2026-05-13
**Current phase:** discover (next)
**Cycle type:** mini-cycle
**Plugin version:** v0.8.5
**Artifact base:** `docs/agentic-serving/`
**Skipped phases:** research, synthesize (model and architect both retained as likely-needed given the ensemble-contract cluster's vocabulary and module-shape implications — see MODEL and ARCHITECT notes below). SYNTHESIZE flagged as attractive at cycle close given the cross-cycle pattern-recognition character of the ensemble-contract proposal; remains optional per Mode D shape.
**BUILD mode:** to be declared at BUILD entry (gated recommended given the design-alternative examination character of routing/observability work; auto mode appropriate only if BUILD reduces to mechanical wiring after DECIDE)

**Origin:** Cycle 5 PLAY (2026-05-13) chose **Path 1** — Thread A defects (4 broken capability ensembles + result-summarizer compression + `code-generator` coder timeout) handled as normal llm-orc dev work outside the methodology cycle; routing + observability axes opened as Cycle 6 scoped mini-cycle. Practitioner verbatim: *"I think path 1 is the way forward. Routing + observability need to be addressed."* Cycle 5 status archived at `cycle-archive/cycle-5-agentic-serving-library-structure.md`.

**Cycle 6 question framing (provisional, for DISCOVER):**

The cycle has three linked clusters that emerged from Cycle 5 PLAY findings + 2026-05-14 follow-on verification work:

1. **Observability** — Cycle 5 PLAY note 19 (sharpened by susceptibility snapshot) discloses the gap as **infrastructure-complete / routing-incomplete**. Cycle 5 BUILD shipped new internal events (verdicts, tier-routing decisions, audit consumption, signal-channel aggregation); the architecture has the telemetry; what is missing is the routing of telemetry to human-visible surfaces. Follow-on verification sharpened this further to include **liveness signal** — not just completion events but in-flight state visibility during long cloud-LLM inference waits. Cycle 6 asks: which surfaces should receive which events? Operator-terminal (colored logs? TUI dashboard?) for the Ensemble Author / Operator stakeholder; orchestrator-context-includes-execution-state for the orchestrator's reasoning surface so it can answer the timing/graph questions a Skill Orchestration User asks; tool-call-emit logging and inference-wait heartbeats so the operator has signal during in-flight states.

2. **Ensemble contract** — Cycle 5 PLAY note 6 (AS-7 result-summarizer content-stripping) + follow-on findings 2 (output-spec drift) + 8 (artifact-substrate proposal) + the practitioner's 2026-05-14 framing of common I/O contract for ensembles cluster around a single architectural question: **what is the ensemble-response (and ensemble-input) contract?** Two sub-questions within: (a) **common I/O envelope** — what fields does every ensemble response carry (`status`, `primary`, `metadata`, `artifacts[]`, `diagnostics`?); (b) **artifact-substrate** — within the envelope, the policy for "when `primary` is large or structurally complex, write to artifact and put reference in `artifacts[]` instead of inline." This cluster touches ADR-021 (per-capability dispatch contract) and ADR-004 (result-summarizer role) directly; likely produces a new ADR or amends both.

3. **Routing-preference** — Cycle 5 PLAY note 20 disclosed the orchestrator's operational routing preference under both tested client configurations is **direct LLM completion → client-tool delegation → `invoke_ensemble` dispatch**, not ensemble-first-when-slot-fits as ADR-021's natural-language-supported clause implies. Follow-on finding 3 strengthened this — even *explicit-naming* of the ensemble + dispatch verb was insufficient on first attempt; the orchestrator used the client `Write` tool instead. Cycle 6 asks: is this the routing surface the system wants, or is it a defect? If wanted, document the operative routing preference (and narrow or clarify ADR-021's natural-language-supported clause). If defect, what intervention (system-prompt work? dispatch-routing-policy ADR?) restores ensemble-first routing under NL framing? Is the preference MiniMax M2.5-free specific or systemic to the cheap-cloud-orchestrator + tool-rich-client pattern?

The three clusters are linked: the operator cannot tell what routing decision happened without observability; the orchestrator cannot refine its routing decisions without visibility into its own dispatches; and the ensemble contract shapes both what the orchestrator dispatches AND what the operator sees when it does.

**MODEL handling for Cycle 6:** Retained as likely-needed (shifted from "skipped" per the 2026-05-14 follow-on framing). The ensemble-contract cluster introduces substantial new vocabulary candidates: `envelope`, `primary`, `artifact reference`, `content_type`, `retention policy`, `diagnostics`, `deliverable` (and possibly `routing preference`, `tool-call-emit`, `inference-wait heartbeat` from the routing and observability clusters). If DISCOVER surfaces enough new vocabulary, a dedicated MODEL phase formalizes the additions in `domain-model.md` before DECIDE. If the vocabulary turns out to be light, the additions can still fold into DISCOVER's tail as Amendment Log entries and MODEL stays skipped — the decision is DISCOVER's based on what surfaces.

**ARCHITECT handling for Cycle 6:** Retained as likely-needed (shifted from "conditionally possibly-needed" per the 2026-05-14 follow-on framing). The ensemble-contract cluster has module-shape implications across at least three modules: `agentic-result-summarizer` (consumes envelope's metadata rather than content), `OrchestratorToolDispatch` (response shape constraint; possibly artifact reference handling), and possibly a new operator-visible-event surface module for the observability cluster. If DECIDE outcomes confirm the contract changes and the new event surface, ARCHITECT allocates the responsibilities. The cycle's `Skipped phases:` field will be amended at DECIDE close if ARCHITECT turns out to be unnecessary after all.

## Phase Status

| Phase | Status | Artifact | Key Epistemic Response |
|-------|--------|----------|----------------------|
| DISCOVER | ☐ Next | — | — |
| DECIDE | ☐ Pending | — | — |
| ARCHITECT | ☐ Conditional | — | — |
| BUILD | ☐ Pending | — | — |
| PLAY | ☐ Optional | — | — |
| SYNTHESIZE | ☐ Optional | — | — |

## DISCOVER-entry context (carry-forwards from Cycle 5 PLAY close)

A fresh session entering Cycle 6 DISCOVER should read in this order:

1. This file (Cycle 6 cycle-status.md) — current state.
2. `essays/reflections/field-notes.md` Cycle 5 PLAY section (20 observations + cross-cutting reflection + post-reflection coda + routing summary) — the empirical substrate Cycle 6 is responding to.
3. `housekeeping/audits/susceptibility-snapshot-cycle-5-play.md` — the snapshot that reframed three of the field notes' aggregate framings (note 1's overstated diagnosis; note 19's infrastructure-complete/routing-incomplete; note 15's compound-framing split).
4. `cycle-archive/cycle-5-agentic-serving-library-structure.md` — Cycle 5 close-state, including the three active BUILD-snapshot advisories.
5. `product-discovery.md` §Stakeholder Map — Ensemble Author / Operator (Cycle 5 added 4 tasks) + Skill Orchestration User (Cycle 5 confirmed-distinct).
6. `decisions/adr-019-*.md`, `adr-020-*.md`, `adr-021-*.md` — Cycle 5's three new ADRs, particularly ADR-021's natural-language-supported clause (the contract Cycle 6 must either re-ground or narrow).

### Settled premises going into Cycle 6 DISCOVER

1. **The orchestrator's operational routing preference is direct → client-tools → ensemble** (PLAY notes 1–9, 20 — empirically verified under both tool-less `curl` and tool-rich OpenCode). ADR-021's "natural-language supported" clause is unsupported under both tested client configurations.
2. **The observability gap is infrastructure-complete / routing-incomplete** (PLAY note 19 + snapshot reframe). Cycle 5 BUILD's new internal events exist in code and write to `execution.json` artifacts; routing of those events to human-visible surfaces has not been designed.
3. **Routing + observability are linked axes** (PLAY note 12 — orchestrator's structural blindness to its own execution graph; PLAY notes 13–15 — orchestrator articulating dispatch defects without resolution paths). The two axes share a common architectural concern: the orchestrator's reasoning surface is structurally separated from the dispatch telemetry it commissions.
4. **The Skill Orchestration User stakeholder's mental model of "the orchestrator will route my NL request to a capability ensemble" is not borne out** (PLAY note 20). DISCOVER must either revise the stakeholder's mental model entry in `product-discovery.md` or treat the routing surface as defect territory and design intervention.
5. **The Ensemble Author / Operator stakeholder's super-objective includes observability** (Cycle 1 PLAY notes 7, 9, 10; Cycle 4 PLAY note 7; Cycle 5 PLAY note 19 — bilateral visibility absence across three cycles). The cycle's recurring observability gap is now empirically anchored at the operator-stakeholder level.

### Open questions DISCOVER must address

1. **Routing-surface intent.** Is the operational routing preference (direct → client-tools → ensemble) the intended behavior, or a defect to remediate? Belief-mapping question: what would have to be true for the operational preference to be the intended behavior? (Possibility: capability ensembles are deliberate-explicit territory; client-tool delegation is the expected NL response surface; ADR-021's natural-language-supported clause was over-broad.)
2. **Observability scope.** What surfaces should receive what events? Operator-terminal (colored logs / TUI dashboard); orchestrator-context (so the orchestrator can answer dispatch-graph questions); both. Belief-mapping question: what would have to be true for the orchestrator-context route to be sufficient on its own (skipping the operator-terminal route)?
3. **The compound-framing split** (PLAY note 15 / snapshot Advisory 3). The two failure modes — (a) hallucination presented as ensemble output, and (b) accurate critique of ensemble dispatch-surface behavior — may warrant different remediation paths. DISCOVER asks: are these one stakeholder concern or two?
4. **The Skill Orchestration User's expectation of dispatch-via-NL.** PLAY note 20 disclosed the gap between stakeholder mental model and operational behavior. DISCOVER asks: should the stakeholder's mental model be revised (operators learn to use explicit naming), or should the system meet the stakeholder's existing mental model (NL routes to ensemble when slot fits)?

### Specific commitments carried forward to DISCOVER (from Cycle 5 PLAY snapshot)

1. **Note 1's "structurally inadequate" framing overstates the diagnosis.** Cycle 6 DISCOVER should NOT treat the Thread A defects as Cycle 6 territory. The remediation is a single scenario addition + mechanical fix, handled as normal dev work. Thread A is mentioned here only because the scenario addition (runtime-dispatch test mandate) interacts with Cycle 6's DECIDE work on observability (the test scenario benefits from the new operator-visible event surface).

2. **The compound-framing split (snapshot Advisory 3)** is dispatched to DECIDE — DISCOVER surfaces both failure modes; DECIDE deliberates whether they warrant one ADR or two.

3. **Active BUILD-snapshot advisory carry-forwards** from Cycle 5 BUILD (still active):
   - Preservation-scenario amendment pattern (auto-mode feed-forward) — Cycle 6 BUILD should not amend scenarios silently; surface scenario-rewrite events for practitioner review.
   - Script-agent YAML schema constraint documentation — if Cycle 6 touches operator-facing documentation, distinguish LLM-agent and script-agent YAML schemas explicitly.
   - ADR-019 §Consequences §Positive n=1 scope qualifier — Cycle 6 should either act on this (extend evidence base via non-RDD framework integration during PLAY) or explicitly defer with rationale.

### Post-hotfix verification findings (2026-05-13, after Cycle 5 PLAY close)

After Thread A hotfixes shipped (commits `e1b0111` role-loading fix, `a935078` cheap-tier timeout bump, `8a0fd24` runtime-dispatch scenario, `176b471` + `234bda5` web-searcher Kagi/DDG with DDG default), the practitioner ran four verification prompts through OpenCode against the restarted serve process. CLI batch confirmed all four previously-broken ensembles now dispatch successfully at the agent execution layer (`claim-extractor` 30.67s; `argument-mapper` 1m 16s; `prose-improver` 54.57s; `text-summarizer` 6.81s). The verification surfaced five new observations for Cycle 6 DISCOVER attention:

1. **Composition pipeline end-to-end now works.** Verification prompt 3 (lit-review shape: `web-searcher` → `claim-extractor` → `argument-mapper`) ran end-to-end in 3m 54s. Cycle 5 PLAY probe 9 halted at `web-searcher`'s `authentication_failed`; with DDG default and the four runtime defects fixed, the three-stage composition completes. Cycle 5 BUILD acceptance criterion "five capability ensembles compose with per-skill tier defaults" now has live-deployment evidence at three-stage depth. **DISCOVER attends:** the cycle's compositional shape is operationally verified — Cycle 6's routing-axis work can proceed against a working baseline.

2. **Output-spec drift on capability ensembles.** Both `claim-extractor` and `argument-mapper` produced output that deviated from their `default_task` specs. `claim-extractor`'s spec mandates `(established)` / `(contested)` labels per bullet; observed output was plain bulleted claims without labels. `argument-mapper`'s spec mandates four sections (Premises / Intermediate Conclusions / Final Claim / Logical Gaps); observed output used a different structure (Thesis / Reasoning Chain / Key Tensions). Whether this is the synthesizer agent deviating from its `default_task`, the orchestrator restructuring during response shaping, or the orchestrator's natural-language narration overriding raw ensemble output is open. **DISCOVER attends:** what is the actual ensemble-output → orchestrator-narration data flow? Does the orchestrator preserve raw ensemble output verbatim, or does it restructure during the post-dispatch summarization step (which would compound the Cycle 5 PLAY note 6 `agentic-result-summarizer` content-stripping pattern)?

3. **Explicit-naming bypass under client-tool surface.** Verification prompt 4 (verbatim: *"Please dispatch the code-generator capability ensemble via invoke_ensemble to write a Python class CircularBuffer..."*) on first attempt: orchestrator **bypassed `invoke_ensemble` entirely** and used the client `Write` tool to create `circular_buffer.py` directly. PLAY note 20's finding was "natural-language framing never dispatches under client tools"; this observation extends it: **explicit naming + dispatch verb may also fail to route under client tools.** Required practitioner intervention (*"It's in .llm-orc/ensembles/agentic-serving/code-generator.yaml so it should have registered"*) for the orchestrator to acknowledge and retry. **DISCOVER attends:** is the routing-preference defect's scope wider than "NL framing alone" — does it extend to "explicit-naming under client-tool-rich client"? If so, the operational routing preference is more aggressively direct-or-client-tools-biased than note 20 characterized.

4. **Latency phenomenology acute on complex tasks.** Verification prompt 4 first attempt: 8m 35s wall-clock. Practitioner retry after redirect: 44m 32s with multiple internal Globs, retries, and orchestrator self-correction between attempts. The observability gap (note 19, sharpened to infrastructure-complete / routing-incomplete) compounds with latency — the practitioner has no signal during 40+ minute waits about what the orchestrator is doing, whether it is making progress, or whether it is stuck. **DISCOVER attends:** what latency surface should the operator-visible event surface route? Tier-routing decisions, dispatch start/end events, retry counters, fail-safe state transitions. The observability axis is sharpened: not only "what dispatch decision was made" but "what is happening *right now*, in real time."

5. **Orchestrator self-acknowledged misrouting.** Practitioner verbatim from orchestrator's response after prompt 4's first attempt: *"I short-circuited that by jumping to manual file writes — bad routing on my part."* The orchestrator accurately diagnoses its own routing defect in narrative form but has no in-session mechanism to fix it — it can only re-route on the next dispatch attempt (which then required multiple retries before settling). This is PLAY note 13 in active operational form: the orchestrator's narration includes correct architectural critique of its own behavior, but the critique informs only practitioner action, not orchestrator behavior on the same turn. **DISCOVER attends:** is the orchestrator's self-narration about routing decisions itself a signal worth treating as input to the routing surface? If the orchestrator can identify a routing defect post-hoc, can a system-prompt or dispatch-loop modification let that self-diagnosis trigger an immediate correction rather than wait for practitioner intervention?

6. **Information-finding overhead disproportionate to task value.** Practitioner verbatim observation about prompt 4: *"it had trouble finding the right ensemble and it seemed like the time to find trivial info was long. The whole task took 44 minutes."* The orchestrator narrated *"I saw it in the list"* about `code-generator` early in the session — meaning `list_ensembles()` had already surfaced the target ensemble's existence — but still bypassed `invoke_ensemble` on the first attempt. On practitioner-prompted retry, the orchestrator entered an information-gathering loop: `Glob` for `**/circular_buffer*.py` across the project tree, `no such file or directory` responses, attempted removal of the previously-written file, additional `Glob` retries — all before the actual ensemble dispatch occurred. The orchestrator had the information it needed early; the loop arose from the orchestrator not *using* the information it already had. **DISCOVER attends:** the ratio of useful-work-time to total-elapsed-time may be a stronger operator-experience metric than absolute latency. Two routing-axis questions follow: (a) what would cap or surface the orchestrator's information-gathering loops so the practitioner does not pay a 35+ minute tax on a recoverable misroute; (b) is the loop a property of MiniMax M2.5-free's specific reasoning shape, or systemic to the cheap-cloud-orchestrator pattern under client-tool-rich clients?

7. **Server console emission is low-signal noise + coarse-success-only telemetry.** Server log captured during the verification session (2026-05-13, roughly 17:38–18:06 window covering the four verification prompts) shows three categories of emission and one category of absence:
   - **Noise — every `list_ensembles` operation re-validates the full library**, emitting two-line Pydantic validation warnings for two legacy schema-drifted ensembles (`fan-out-test.yaml`: rejects `type: script` and `system_prompt:` on `ScriptAgentConfig`; `plexus-graph-analysis.yaml`: rejects `type: script`). These warnings emit on every ensemble enumeration. Within the captured 28-minute log window, the same warnings appear at least 8 distinct enumeration cycles — orchestrator information-gathering loops (see finding 6) compound the noise volume. The two legacy YAMLs are pre-existing schema drift (Cycle 4 PLAY note 7 already named them); they are not Cycle 5 or Cycle 6 territory by content, but their continued chatter is Cycle 6 observability territory by effect.
   - **Coarse-success-only telemetry** — `INFO: tool dispatch: result name=invoke_ensemble kind=success` appears for each successful dispatch (four in the window), with no contextual fields. The console line does not identify which ensemble was dispatched, which session_id is in play, what verdict the Calibration Gate produced, or what tier the Tier-Escalation Router selected. An operator reading the log cannot reconstruct what happened from the log alone.
   - **HTTP access lines** — standard FastAPI/uvicorn lines (`INFO: 127.0.0.1:NNNNN - "POST /v1/chat/completions HTTP/1.1" 200 OK`) emit per request. Useful but not agentic-serving-specific.
   - **Absent** — none of Cycle 4 BUILD's nor Cycle 5 BUILD's typed internal events surface to the console: TierSelection (model_profile, tier, topaz_skill), CalibrationVerdict (Proceed/Reflect/Abstain), TierRouterAudit dispatches, SignalChannel state transitions, fail-safe activations, retry counts. All of these exist as typed values in code and flow to `execution.json` artifacts; none route to the operator-readable surface.
   
   **DISCOVER attends:** the routing-incomplete observability surface is concrete on two fronts. (a) **Noise floor** — the two named legacy YAMLs were amended in the same commit that records this finding (`fan-out-test.yaml` and `plexus-graph-analysis.yaml` had `type: script` and on `fan-out-test` `system_prompt:` removed, since `ScriptAgentConfig` identifies script agents by the `script:` field rather than a `type:` field). The specific per-session noise floor is removed going forward; the underlying *mechanism* (every `list_ensembles` re-validates the whole library and emits stack-traceable warnings on each invalid YAML) remains a DECIDE question — single-warn-then-skip vs. validate-once-at-load vs. some other pattern. (b) **Route the missing surface** — Cycle 5 BUILD's events have shapes in code (typed dataclasses) that serialize to operator-readable lines without designing new types. DECIDE has a concrete starting list: which events at what level in what shape. This finding empirically anchors findings 4 (latency) and 6 (information-finding overhead) — the 8+ enumeration cycles in a 28-minute log window are the operator-invisible signature of finding 6, and the absence of routing-decision events is the operator-side blind spot of finding 4.

   **Liveness-signal extension (2026-05-14 follow-on observation):** A subsequent verification session ran a single `chunk_by_predicate` code-generator prompt in 8m 28s (vs. the prior day's 44m 32s — 5.3× variance for nominally similar tasks). The serve log for the full 8m 28s session emitted **five lines total**: server startup banner (4 lines) + two HTTP `200 OK` access lines + three `tool dispatch: result ... kind=success` lines (one `list_ensembles`, two `invoke_ensemble` — code-generator and the result-summarizer return path). One line every ~100 seconds on average. The same session sat in a 10+ minute window of total console silence during the practitioner's wait. This concretizes the "completion-events-only" pattern: every log line is post-hoc; no log line surfaces in-flight state (cloud LLM inference pending, tool call emitted but not yet dispatched, network round-trip in progress). **DISCOVER attends (extension):** the operator-visible event surface needs **liveness signal during in-flight states**, not just completion events. Two low-cost candidates for the observability ADR: (a) **tool-call-emit logging** — when the orchestrator emits a tool call into the response stream, log it before dispatching (gives a "received tool call from cloud LLM at HH:MM:SS" anchor); (b) **inference-wait heartbeats** — when a request has been open for more than N seconds without tool activity, emit a "still waiting on cloud LLM" line (gives mid-stream signal during long inference). Both are minor extensions to the existing tool-dispatch logging surface; neither requires new infrastructure.

8. **Artifact-as-substrate as a candidate design direction (practitioner-proposed).** From the same 2026-05-14 follow-on session, after observing that the code-generator output suffered from familiar "coherent-but-incorrect" drift (a synthesized function used `List[T]` without a `TypeVar` declaration; would `NameError` at parse time; the orchestrator caught the defect post-hoc and appended a correction note, but the in-ensemble critic + synthesizer flow did not), practitioner verbatim: *"this also to me points to an ensemble design shape for agentic-serving. This code of course gets buried in the ensemble output. So one strategy could be to always rely on artifact writing to be the substrate."*

   **The proposal:** capability ensembles whose deliverable is substantive (code, structured analyses, long-form text) write the deliverable as an *artifact* to disk; the ensemble response carries only a reference (e.g., `{artifact_path, content_type, summary}`); the orchestrator's context never carries the deliverable's content. Cycle 5 BUILD's existing artifact infrastructure (`execution.json` and per-dispatch directories under `.llm-orc/artifacts/<ensemble>/<timestamp>/`) is already partial substrate — the deliverable-as-artifact would extend it with a typed output artifact alongside `execution.json`. What changes is **the ensemble-response contract** (ADR-021 territory) and **the result-summarizer's role** (ADR-004 / AS-7 territory): the summarizer summarizes *metadata*, not content.

   **How this collapses three existing findings simultaneously:**
   - **Finding 2 (output-spec drift)** — if the artifact is the canonical output, orchestrator-narration restructuring is inconsequential to the deliverable; the client reads the artifact directly. The narration-vs-spec mismatch ceases to be load-bearing.
   - **Finding 6 (information-finding overhead)** — the artifact path is a single handle; no re-reading or re-summarizing the content across orchestrator turns. The context-window cost stays bounded.
   - **Cycle 5 PLAY note 6 (AS-7 result-summarizer content-stripping)** — the summarizer summarizes the artifact's metadata (size, type, ensemble that produced it) rather than its content. The stripping-and-inverting failure mode dissolves because no content was passed through the summarizer in the first place.

   **DISCOVER attends:** is artifact-as-substrate the right architectural commitment for capability ensembles whose output is large or structurally complex? Five sub-questions: (a) **Boundary** — which ensemble outputs warrant artifact-substrate vs. response-substrate? (Single-bullet claims are fine in-response; multi-hundred-line code is not.) (b) **Contract** — is `artifact_path` a typed field on ensemble response, or a convention? (c) **Client access** — does the client have filesystem access to read the artifact, or does it fetch via an HTTP endpoint the serve layer exposes? (d) **Cleanup** — artifacts accumulate; what's the retention policy? (e) **Backward compat** — current ensemble responses are content-bearing; how does the contract change interact with ensembles already deployed?

   This proposal is candidate ADR territory for Cycle 6 DECIDE — possibly a new ADR amending ADR-021 (ensemble dispatch contract) and ADR-004 (result-summarizer role). Sequenced after Cycle 6 DISCOVER establishes which findings warrant deliberation. See **Candidate spikes** subsection below for the survey-and-propose investigation that informs DECIDE on this cluster.

### Candidate spikes for DISCOVER or DECIDE entry

The post-hotfix verification surfaced three spike candidates worth scoping at Cycle 6 entry. Each spike's purpose is to resolve uncertainty empirically before committing the cluster's design decisions; per the spike-runner pattern, each is bounded and produces a research log feeding the relevant phase.

**Spike α — Common I/O envelope survey + proposal (ensemble-contract cluster).**

- **Question.** What envelope shape can wrap all 8 agentic-serving ensembles' current behavior with minimum disruption? Candidate fields: `status`, `primary`, `metadata`, `artifacts[]`, `diagnostics`.
- **Method.** (1) Enumerate current I/O shapes for the 8 ensembles (6 capability + 2 system) by reading each YAML's `default_task` and a representative invocation result. (2) Draft 2-3 candidate envelope shapes. (3) For each ensemble, test fit per candidate envelope: does the envelope wrap the ensemble's current behavior? what changes are required (system prompt, synthesizer logic, response shape)?
- **Output.** Research log documenting envelope candidates with per-ensemble fit notes. Feeds DECIDE on the ensemble-contract cluster's primary ADR.
- **Bounded scope.** ~3-4 hours of analytical work; no LLM dispatch required (or minimal — one representative invocation per ensemble).

**Spike β — Composition predictability under common envelope (ensemble-contract cluster).**

- **Question.** With a common envelope, do A→B compositions become structurally predictable rather than ad-hoc text-passing? Where do current assumptions about output structure actually live in the orchestrator's reasoning, and would a typed envelope move them into the contract instead?
- **Method.** Take the lit-review composition shape (`web-searcher` → `claim-extractor` → `argument-mapper`) and the candidate envelope from Spike α. Trace data flow: what does each stage produce, what does the next stage need? Identify where current implicit assumptions live (orchestrator prompt instructions? synthesizer agent default_task wording? ad-hoc text parsing?). Test: if A emits envelope-shaped output, can B consume it without intermediate text-massaging?
- **Output.** Research log capturing composition predictability findings; informs whether artifact-substrate alone is sufficient or whether typed output fields in the envelope are also load-bearing for composition.
- **Bounded scope.** ~2-3 hours; analytical, no required LLM dispatch.
- **Collapsibility with Spike α.** If scoped together as one investigation, Spike α + Spike β share data-gathering and propose-and-test work. Recommended unless the practitioner wants to time-box them separately.

**Spike γ — Routing-preference characterization (routing-preference cluster).**

- **Question.** Is the orchestrator's `direct → client-tools → invoke_ensemble` routing preference (PLAY note 20, finding 3) a MiniMax M2.5-free-specific reasoning shape, or systemic to the cheap-cloud-orchestrator + tool-rich-client pattern?
- **Method.** Dispatch the same prompt (a clean NL framing that maps unambiguously to one slot) under varied orchestrator profiles and client tool sets: (a) MiniMax M2.5-free via Zen, OpenCode (tool-rich) — replicates current; (b) `agentic-orchestrator-offline` (qwen3:14b local), OpenCode — controls for cloud-vs-local; (c) MiniMax M2.5-free, `curl` (tool-less) — replicates Cycle 5 PLAY baseline; (d) optionally a different cloud profile if accessible. Record routing decisions per cell.
- **Output.** Research log characterizing the routing-preference behavior across configurations; informs whether the routing-preference cluster needs orchestrator-profile-specific ADRs or a generic ADR.
- **Bounded scope.** ~2-4 hours including dispatch wait time; requires live serve and at least two orchestrator profiles available.
- **Independence.** Spike γ is independent of Spikes α/β and can run in parallel.

## Feed-Forward Signals

### From Cycle 5 (closed at PLAY 2026-05-13)

The Cycle 5 archive carries forward five load-bearing findings for Cycle 6 DISCOVER attention:

1. **Routing-preference disclosure (PLAY notes 1–9, 20)** — the orchestrator's operational routing is direct → client-tools → ensemble across both tested client configurations. ADR-021's natural-language-supported clause is the contract under question.
2. **Observability framing sharpening (snapshot reframe of note 19)** — infrastructure-complete / routing-incomplete. The DECIDE target is wiring existing telemetry, not designing observability from scratch.
3. **Orchestrator self-modeling reliability (PLAY notes 13, 14)** — the orchestrator names dispatch defects accurately but its self-predictions of fix effectiveness do not bear out. RESEARCH-routed: at what threshold of self-knowledge does an orchestrator stop fabricating recovery-narrations?
4. **Bilateral visibility absence across three cycles (PLAY note 19 + Cycles 1, 4 baseline)** — the operator-stakeholder concern is now empirically anchored. Cycle 6 is the natural cycle to close this carry-forward.
5. **Working surfaces verified at PLAY** (notes 3, 7, 8, 17) — explicit-naming dispatch contract; script-agent error-path handling; composition dependency handling; multi-turn memory. Cycle 6 does NOT need to re-verify these; they are settled-by-use substrate.

### From Cycle 5 BUILD susceptibility snapshot (3 advisories still active)

1. Preservation-scenario amendment pattern (auto-mode feed-forward) — applies to any Cycle 6 BUILD that runs auto mode.
2. Script-agent YAML schema constraint documentation — applies if Cycle 6 BUILD touches operator-facing docs.
3. ADR-019 §Consequences §Positive n=1 scope qualifier — Cycle 6 should explicitly disposition this (act on, or defer with rationale).

### From Cycle 4 (closed at PLAY 2026-05-12; archived `cycle-archive/cycle-4-cheap-orchestrator-and-ensembles.md`)

Cycle 4 PLAY's notes 7, 16, 19 are the empirical roots Cycle 6 acts on. Cycle 4 PLAY's notes 14, 15 (settled by use in Cycle 5) are substrate for the routing-axis question: if methodology-layer / dispatch-layer / execution-layer separation is the architectural framing, observability and routing are both dispatch-layer concerns.

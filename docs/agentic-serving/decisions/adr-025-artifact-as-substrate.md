# ADR-025: Artifact-as-Substrate for Capability Ensemble Deliverables — Always Scope

**Status:** Proposed

**Date:** 2026-05-15

---

## Context

The 2026-05-14 follow-on verification finding 8 surfaced the **artifact-as-substrate** proposal in practitioner-verbatim form: *"this also to me points to an ensemble design shape for agentic-serving. This code of course gets buried in the ensemble output. So one strategy could be to **always** rely on artifact writing to be the substrate."* The proposal: capability ensembles whose deliverable is substantive (code, structured analyses, long-form text) write the deliverable as an *artifact* to disk; the ensemble response carries only a reference (`{artifact_path, content_type, summary}`); the orchestrator's context never carries the deliverable's content.

Cycle 6 MODEL Action B marked the proposal's **three-findings-collapse claim** as agent-composed framing of downstream implications, not a structural property of the concept. The collapse claim asserted that artifact-as-substrate simultaneously dissolves: (1) output-spec drift (finding 2), (2) information-finding overhead (finding 6), (3) AS-7 result-summarizer content-stripping (Cycle 5 PLAY note 6). MODEL Action B required DECIDE to test each finding against alternative remediation paths independently before accepting the collapse as T16's deliberation substrate.

Spike β (research log `essays/research-logs/cycle-6-spike-beta-composition-predictability.md`, executed 2026-05-15) ran the per-finding test. Result: **the three-findings-collapse claim does not hold under per-finding test.** Each finding wants a different best-fit remediation:

- **Output-spec drift** is addressed by ADR-024's typed `structured` field + advisory `output_schema:` declaration (envelope shape, not substrate). Spike β reframed the drift's mechanism as the orchestrator hand-writing each stage's `input.data` overriding `default_task` at run-time; substrate-routing does not address the `input.data`-override mechanism directly.
- **Information-finding overhead** is addressed by substrate-routing — the orchestrator's reasoning surface carries an artifact path + summary, not the deliverable content, bounding context cost across orchestrator turns.
- **AS-7 result-summarizer content-stripping** is addressed by substrate-routing — when the summarizer summarizes the artifact's metadata (size, type, ensemble that produced it) rather than its content, the stripping-and-inverting failure mode dissolves because no content was passed through the summarizer in the first place.

Spike γ (research log `essays/research-logs/cycle-6-spike-gamma-routing-characterization.md`, executed 2026-05-15) surfaced an unanticipated **fourth defect** during Cell A-explicit: the orchestrator narrated a different implementation than the synthesizer produced (the orchestrator's response described a hand-rolled two-pointer loop; the synthesized artifact used `chars.reverse()`). This is PLAY note 15's fabrication-while-critiquing-fabrication pattern in concrete operational form, **downstream of AS-7** (the summarizer worked correctly; the narration substituted). Substrate-routing eliminates this surface — the deliverable never passes through orchestrator narration to the client.

Cycle 6 MODEL Open Question #15 opened a **parallel amendment pathway for invariant AS-7** via T16's scope deliberation. OQ #13 already flagged AS-7 ("Result summarization is a correctness requirement") as pending evidentiary review under Wave 3.A Trial 1's specificity-loss observation; OQ #15 added a structurally distinct amendment pathway: if T16 resolves toward artifact-as-substrate at any scope, AS-7's load-bearing concern (context rot from unsummarized result dictionaries) becomes inapplicable to substrate-routed ensembles. The two pathways converge on the same amendment but through different evidentiary chains.

Cycle 6 DISCOVER snapshot Action 4 surfaced **scope sub-question (0)** as the first T16 DECIDE deliberation: artifact-as-substrate scope is *always* (practitioner verbatim), *when substantive* (agent-introduced refinement), or *operator-configured*. The other five sub-questions (boundary, contract, client access, cleanup, backward compat) depend on the scope answer.

Spike α surveyed all 8 ensembles and flagged a **size-floor finding**: substrate-routing is structurally awkward for small-output system ensembles (`agentic-calibration-checker` returns a few-word verdict; `agentic-result-summarizer` returns a paragraph) and for already-handle-shaped capability ensembles (`web-searcher` returns a JSON list of URL+snippet records). Substrate-routing those produces "an artifact reference to a 50-byte verdict file" — the substrate's overhead exceeds its value when the deliverable is small or already-handle-shaped.

The practitioner answer at the DECIDE-entry deliberation chose **always** for capability ensembles with explicit acknowledgment of the size-floor tradeoff: *"I think 2 [always] for now — we can dial this back later if we feel it's cumbersome."* The practitioner also proposed the session-dir location: *".llm-orc/agentic-sessions/<timestamp>/some-structured-file."*

---

## Decision

### Scope: always, for capability ensembles

All **capability ensembles** route their deliverable through artifact-as-substrate. The six current capability ensembles — `code-generator`, `prose-improver`, `argument-mapper`, `claim-extractor`, `web-searcher`, `text-summarizer` — write their deliverable to a session-scoped artifact path; the envelope's `primary` field carries a human-readable summary line referencing the artifact; the envelope's `artifacts[]` field carries the typed artifact reference per ADR-024's envelope contract.

**System ensembles remain inline.** `agentic-calibration-checker` and `agentic-result-summarizer` are internal infrastructure consumed by the orchestrator's dispatch loop (not by skill frameworks); their response shapes serve internal control flow. They retain inline-response envelopes with `artifacts[]` empty.

**Size-floor tradeoff deliberately accepted.** Spike α's finding that substrate-routing is structurally awkward for small-output or already-handle-shaped capability ensembles (notably `web-searcher` with its JSON list of URL+snippet records, and short `text-summarizer` invocations) is acknowledged at decision time. The cycle's commitment to the cleaner design rule — one scope rule for all capability ensembles — is the trade. The "dial back later if cumbersome" disposition is structurally honest: BUILD's first deployments and the post-BUILD PLAY cycle will surface whether the always-scope produces operational friction; if so, a follow-on cycle refines the scope to *when substantive* with explicit per-ensemble criteria. The size-floor finding is not a defect against the decision; it is the named tradeoff the decision accepts.

**Dial-back falsification criteria (the indicators that fire the refinement-to-substantive-scope question):**

The always-scope is preserved unless **at least one** of the following surfaces in BUILD-deployment or post-BUILD PLAY observation:

1. **Artifact-substrate latency overhead exceeds 10% of dispatch wall-clock** for any capability ensemble whose deliverable is under 1 KB (the empirical size-floor threshold from spike α's per-ensemble survey — the smallest substantive deliverable in the current library is `text-summarizer`'s shortest invocation at ~80 chars; the threshold is set at the natural break point between handle-shaped JSON records and short prose).
2. **Operator reports during PLAY that artifact-substrate is "in the way"** for one or more capability ensembles — operator-experience-voice friction is the load-bearing signal; structured-friction questions in the post-BUILD PLAY field-notes specifically probe for substrate-cost-vs-value mismatch per ensemble.
3. **Session-directory disk-space cost is operationally consequential** — sessions accumulating substrate artifacts at a rate that requires operator-intervention (manual `prune` runs) more often than monthly under typical-deployment usage. This indicator probes whether the always-scope is generating noise without proportional value.
4. **Three or more capability ensembles declare `output_substrate: inline`** as opt-outs during BUILD migration. The opt-out is operator-configurable per ADR-025; if multiple ensembles need it, the always-scope is producing the boundary judgments the substantive-deliverable scope would have made at DECIDE time. The threshold is "three or more" — one or two opt-outs are exceptions; three or more is a pattern.
5. **Cross-dispatch shared-substrate references emerge as a deliberate operator pattern in BUILD or PLAY** (added at DECIDE gate, 2026-05-15). ADR-025's session-dir scopes artifacts per-`<dispatch_id>` (`<session_id>/<dispatch_id>/<deliverable>`). If operators or skill frameworks deliberately wire multiple ensembles to reference one shared artifact (e.g., a `claim-extractor` dispatch produces a claim-list artifact that a subsequent `argument-mapper` dispatch reads as input), the per-dispatch scoping is fighting the actual usage pattern. The indicator fires when at least three deliberate cross-dispatch references emerge during BUILD migration or post-BUILD PLAY observation. The dial-back deliberation in that case may not be "remove always-scope" but "refine artifact-scope from per-dispatch to session-scoped with explicit sharing semantics" — a structurally distinct refinement.

**Indicator 2 calibration note (DECIDE gate, 2026-05-15):** The "operator-experience friction" framing may be miscalibrated against the cycle's actual expectations. The practitioner expects artifacts to *better ground* ensembles, not feel "in the way." If the practitioner's expectation is borne out, Indicator 2 fires on nothing — it does not surface the real failure modes the cycle anticipates (which live in Indicators 1, 3, and the new Indicator 5). Indicator 2 is preserved as a structural prompt for PLAY field-notes (the question is still worth asking) but its load-bearing status is reduced relative to Indicators 1, 3, and 5.

If any indicator fires, the dial-back deliberation surfaces in the cycle that observes it: the substantive-deliverable scope (rejected at this DECIDE), or one of the structurally distinct refinements (session-scoped artifacts; Plexus-KG-as-substrate per the Out-of-scope subsection below), is re-examined with the empirical evidence the current cycle does not yet have. The "perpetual deferral" failure mode P2-D named is prevented by the indicators making the question fire-on-evidence rather than fire-on-discomfort.

### Session-dir location

Per the practitioner's verbatim proposal, artifacts live under:

```
.llm-orc/agentic-sessions/<session_id>/<dispatch_id>/<deliverable>.<ext>
```

Where:

- **`<session_id>`** is the orchestrator session identifier (set at session start; carries across all dispatches within the session). Format: `<iso-8601-datetime>-<short-uuid>` — e.g., `2026-05-15T14:32:08Z-a7f3`. The ISO-8601 prefix gives natural chronological sort on the filesystem; the short UUID disambiguates same-second sessions.
- **`<dispatch_id>`** is the per-dispatch identifier per ADR-023's correlation identifier. Format: monotonic per-session counter (e.g., `dispatch-001`, `dispatch-002`).
- **`<deliverable>`** is the ensemble-author-chosen artifact name (e.g., `circular_buffer`, `claim-list`, `revised-prose`). Defaults to `<ensemble-name>-output` if the ensemble does not specify.
- **`<ext>`** is the deliverable's natural file extension (`.py`, `.md`, `.json`, `.txt`).

This path structure replaces the existing artifact path `.llm-orc/artifacts/agentic-serving/<ensemble>/<timestamp>/`. The Cycle 5 BUILD `execution.json` artifacts also relocate under the new structure — each dispatch's `execution.json` lives at `.llm-orc/agentic-sessions/<session_id>/<dispatch_id>/execution.json` alongside the deliverable artifact(s). The new structure groups by **session** rather than by **ensemble**, making session-level review and cleanup natural.

### Artifact reference fields

Per ADR-024's envelope contract, each artifact reference carries:

```python
{
    "path": "agentic-sessions/<session_id>/<dispatch_id>/<deliverable>.<ext>",  # relative to .llm-orc/
    "content_type": "application/python",  # MIME type
    "size_bytes": 1247,
    "summary": "Class CircularBuffer with iter and len protocol; 24 lines.",
    "retention": "session",  # one of: session, durable, ephemeral
}
```

Retention semantics:
- **`session`** — retained for the session's lifetime; cleaned up when the session closes. The default for substantive-but-not-promoted deliverables.
- **`durable`** — retained indefinitely; survives session close. Used when the operator (or a downstream skill framework) requests preservation (e.g., a generated file the operator intends to commit).
- **`ephemeral`** — retained only until the orchestrator's next turn; cleaned up after. Used for intermediate-stage deliverables that downstream stages consume immediately.

Retention is declared by the ensemble (in YAML or in synthesizer output) and operator-overridable per-session.

### Client access to artifacts

The agentic-serving serve process runs on the same machine as the operator (per the architecture's local-first commitment). Skill-orchestration clients consuming the serve's `/v1/chat/completions` response have **filesystem access to the artifact path** under the same user account. The envelope's `path` field is the filesystem-resolved location (relative to `.llm-orc/`); clients read the artifact directly from disk.

For deployments where the serve and client are on different machines (a follow-on cycle's territory), an HTTP artifact-fetch endpoint (`GET /v1/artifacts/<path>`) is the natural extension. Cycle 6 BUILD does not ship the HTTP endpoint; the filesystem-access shape covers the local-first deployment.

### Cleanup

Session-scope artifacts (`retention: session`) clean up when the session closes. The serve process tracks active sessions; on session close, it removes the session's directory tree under `.llm-orc/agentic-sessions/<session_id>/`.

Durable artifacts (`retention: durable`) accumulate; operators are responsible for their lifecycle. A future operator-facing CLI command (`llm-orc agentic-sessions prune --older-than=<duration>`) is the cleanup affordance; Cycle 6 BUILD does not ship the command but the structure supports it.

Ephemeral artifacts (`retention: ephemeral`) clean up at the orchestrator's next turn.

### Backward compatibility

The Cycle 5 BUILD `execution.json` artifact at the existing path (`.llm-orc/artifacts/agentic-serving/<ensemble>/<timestamp>/`) is **migrated** to the new structure during Cycle 6 BUILD. The migration is one-way: new dispatches write under the new structure; the old artifact tree may be removed after BUILD ships and operators have a chance to migrate any pinned references. The old path is **deprecated** but not actively cleaned up by BUILD — operators decide when to remove the old tree.

Capability ensembles authored against Cycle 5's inline-response contract migrate to substrate-routing during Cycle 6 BUILD. The migration shape:
1. Ensemble YAML adds `output_substrate: artifact` (default for capability ensembles in Cycle 6 BUILD; opt-out is `output_substrate: inline` for ensembles needing to remain inline).
2. The synthesizer agent (or post-dispatch processing) writes the deliverable to the session-dir artifact path.
3. The envelope is constructed with `primary` carrying a summary line and `artifacts[0]` carrying the typed reference.

### AS-7 amendment: default-with-conditional-skip

Invariant AS-7 ("Result summarization is a correctness requirement") amends to:

> **AS-7 (amended). Result summarization is the default; substrate-routed ensembles skip content summarization.** Full ensemble result dictionaries must be summarized before entering the orchestrator's context — *unless* the ensemble routes its deliverable through artifact-as-substrate (per ADR-025), in which case the envelope's `primary` and `artifacts[0]` already carry summary-shaped content; the result-summarizer ensemble is not invoked. For inline-response ensembles (system ensembles per ADR-025; future ensembles opting out via `output_substrate: inline`), the result-summarizer remains correctness-required to prevent context rot.

The amendment is **default-with-conditional-skip** — the AS-7 framing's load-bearing concern (context rot from unsummarized result dictionaries) is preserved for the inline-response path; substrate-routing dissolves the concern at the substrate layer (the deliverable does not flow through the summarizer in the first place).

This amendment **resolves OQ #15** (parallel amendment pathway for invariant AS-7 via T16's artifact-as-substrate scope deliberation) by codification. It also **converges with OQ #13** (evidentiary review of AS-7 under Wave 3.A Trial 1 specificity-loss observation): the substrate-routing path makes specificity-loss inapplicable for substrate-routed ensembles; the inline-response path retains the evidentiary review as Cycle 7+ territory. OQ #13 narrows in scope to inline-response ensembles only.

### Partial updates to ADR-021 and ADR-004

**ADR-021** §Decision §"Per-capability dispatch contract" step 3 reads: *"The orchestrator returns the capability ensemble's output (already summarized per AS-7) to the skill framework as the chat completion response."* Combined with ADR-024's envelope-shape update, ADR-021 now carries a `> **Updated by ADR-024 and ADR-025 on 2026-05-15.**` header. The dispatch contract's substantive commitments (per-capability dispatch; client-side state; fresh-context property) are preserved; the response shape is the envelope (ADR-024); the summarization is conditional on substrate-routing (ADR-025).

**ADR-004** ("Result Summarization Mandatory") receives a partial update. ADR-004's mandate operates over the inline-response path; substrate-routing dissolves it at the substrate layer. ADR-004 carries a dated `> **Updated by ADR-025 on 2026-05-15.**` header recording the conditional-skip. ADR-004's rationale (context rot prevention) and its decision (mandatory summarization for inline responses) remain current; the scope narrows.

**Relationship to ADR-004's existing per-invocation escape hatch.** ADR-004 §Decision establishes an escape hatch: *"a per-ensemble or per-invocation flag may indicate that the raw output is small enough to pass through directly (e.g., a classifier returning a single label). The default behavior is summarization; opting out is explicit."* ADR-025's substrate-routing operates at a structurally different layer — substrate-routing is a **dispatch-shape commitment** (where the deliverable lives), not a per-invocation flag about whether to summarize. The two mechanisms compose: substrate-routed ensembles per ADR-025 skip content summarization at the substrate layer (the substrate IS the summary mechanism); inline-response ensembles per ADR-025 retain ADR-004's mandate, and within that scope ADR-004's escape hatch still operates (a classifier returning a single label declared `output_substrate: inline` with the small-output escape hatch active produces a no-summarization-needed inline response). The reconciliation: ADR-025's `output_substrate: artifact` is the dispatch-shape decision; ADR-004's escape hatch is the per-invocation skip-summarization decision within the inline-response path.

### Backward propagation (Step 3.7)

Amending invariant AS-7 triggers backward propagation. The sweep applies to:

1. **All prior ADRs and essays.** Each document is examined for language asserting AS-7 as unconditional. The principal documents requiring inspection:
   - `essays/004-agent-design-script-models-orchestrator.md` — references result summarization as load-bearing
   - `essays/005-layer-conditional-composition.md` — references AS-7 in the cross-layer signal-channel discussion
   - ADR-004 (already partial-updated above)
   - ADR-007, ADR-014 — calibration gate logic references result summarization as the pre-context-injection step
   - ADR-021 (already partial-updated above)
2. **Each contradiction surfaced.** Contradictions take the form of unconditional language about AS-7 ("result summarization is a correctness requirement" without the substrate-routing carve-out). The supersession note format is:

   > **Superseded:** The unconditional language below is contradicted by AS-7 as amended in ADR-025. The amended invariant scopes result summarization to inline-response ensembles; substrate-routed ensembles skip content summarization at the substrate layer.

3. **Amendment Log entry.** The domain model's Amendment Log records:

   | # | Date | Invariant | Change | Propagation |
   |---|------|-----------|--------|-------------|
   | 11 | 2026-05-15 | AS-7 (Result summarization is a correctness requirement) | Amended to default-with-conditional-skip — substrate-routed ensembles per ADR-025 skip content summarization; inline-response ensembles retain mandatory summarization. | ADR-004 partial-updated; ADR-021 partial-updated; essays 004, 005, ADR-007, ADR-014 swept for AS-7 references with supersession notes added where applicable. |

The sweep is mandatory before Cycle 6 DECIDE completes. The supersession-gate check in Step 5 verifies it.

### γ's fourth-defect surface as additional value-add

Spike γ Cell A-explicit surfaced the orchestrator narrating a different implementation than the synthesizer produced. Substrate-routing eliminates this surface: with the deliverable in `artifacts[0]` and only a summary-line in `primary`, the orchestrator's narration cannot substitute the deliverable — the client reads the artifact directly, bypassing the narration channel.

This is enumerated as a fourth value beyond the three findings the original artifact-as-substrate proposal collapsed:

1. Output-spec drift becomes inconsequential (the artifact is canonical; orchestrator restructuring at narration is decoupled). **Subject to spike β's reframing:** the drift's mechanism is `input.data` override, which substrate-routing does not fix directly; the addressed concern is the client-side perception of drift, not the upstream drift mechanism. (Per-finding test, MODEL Action B applied — addressed by **ADR-024's `structured` + `output_schema:`** rather than by substrate.)
2. Information-finding overhead stays bounded (orchestrator context carries the path, not the content). **Addressed by substrate.**
3. AS-7 content-stripping dissolves (summarizer summarizes metadata, not content). **Addressed by substrate.**
4. **Orchestrator-narration substitution dissolves** (the deliverable never flows through narration). **Addressed by substrate.** Newly enumerated.

The collapse claim is corrected: substrate addresses findings 2, 3, 4; finding 1 is addressed by ADR-024's typed envelope (`structured` + `output_schema:`). The two ADRs jointly serve the value the original collapse claim asserted; the structural mechanism is more honest about which finding gets resolved where.

### Calibration-gate evaluation surface under substrate-routing

The Calibration Gate (ADR-007 / ADR-014) evaluates dispatched ensembles' outputs to produce the calibration verdict (Proceed / Reflect / Abstain) and the trajectory features (HTC / AUQ per ADR-014). Under inline-response dispatches, the gate's critic/evaluator agents read the ensemble output directly from the response context. Under substrate-routing, the deliverable lives at `artifacts[0].path`; the response context carries only the artifact reference + summary. The gate's evaluation surface under substrate-routing is:

- **`artifacts[0].summary` (metadata-only evaluation) is the default.** The critic/evaluator agents receive the envelope's `primary` and `artifacts[0].summary` fields as their evaluation input. This is the lowest-disruption path — the gate's existing prompt shape (read ensemble output, produce verdict) operates over the summary line rather than full content. For deliverables whose quality is reasonably-inferrable from a one-line summary (e.g., `web-searcher` URL-snippet records, `text-summarizer` paragraph summaries), summary-only evaluation suffices.
- **`structured` (typed payload) when declared.** When a capability ensemble's YAML declares `output_schema:` (per ADR-024), the gate's evaluators consume `envelope.structured` in addition to the summary. Typed payload gives the gate structural anchors (claim counts, label distributions, structured payload content) the summary line does not carry. ADR-024 characterizes schema validation as advisory at dispatch time, so the gate's evaluators assess payload content quality (the actual claims, the actual distribution shapes), not formal schema compliance.
- **Artifact content (file-read) is opt-in per ensemble.** Capability ensembles whose quality cannot be evaluated from summary + `structured` alone (notably `code-generator` — code correctness requires reading the code) declare `calibration_substrate_access: artifact` in their YAML. The Calibration Gate's evaluator agents then receive a tool-call surface to read the artifact at `artifacts[0].path` and evaluate against the actual content. This is the highest-cost evaluation path but mirrors the inline-response evaluation cost — the gate reads the deliverable directly.

The default — summary-only evaluation — keeps the Calibration Gate's existing prompt shape intact for most capability ensembles. The `output_schema:` and `calibration_substrate_access:` extensions are per-ensemble opt-in declarations the BUILD migration adds when an ensemble's quality evaluation requires more than summary. Cycle 6 BUILD scope includes the default path; per-ensemble migration of `calibration_substrate_access: artifact` for `code-generator` (and any other ensemble whose summary-only evaluation produces unreliable verdicts) is part of the substrate-migration work.

This evaluation-surface specification is **load-bearing for the Calibration Gate's substrate-routing operation**. ADR-007 and ADR-014's existing decision text remain current within the inline-response path; under substrate-routing, the gate operates over the evaluation surface specified above.

### Out of scope for ADR-025

- **Per-ensemble retention defaults.** Each capability ensemble can declare a default retention in its YAML (`output_retention: session | durable | ephemeral`); BUILD configures sensible defaults per ensemble. Not a Cycle 6 DECIDE commitment beyond the structure.
- **HTTP artifact-fetch endpoint** (for non-local-first deployments). Acknowledged in §"Client access to artifacts"; not in Cycle 6 BUILD scope.
- **`llm-orc agentic-sessions` CLI commands.** Pruning, listing, archiving session artifacts. The structure supports the commands; the commands are operator-tooling territory, not Cycle 6 BUILD scope.
- **Plexus ingestion of substrate-routed artifacts.** Per AS-4, Plexus ingests source material; substrate-routed deliverables are natural ingestion candidates. Cycle 7+ territory.
- **Plexus-KG-as-substrate (rather than as ingestion-of-substrate) — added at DECIDE gate, 2026-05-15.** The practitioner observed during the gate pre-mortem that *"if we go the Plexus KG route, perhaps the artifact path is not necessary."* This proposes a stronger relationship than the AS-4 ingestion framing: under active Plexus, the KG itself may be the durable substrate, and the filesystem-artifact path becomes the AS-8-absent path (Plexus optional → filesystem substrate) while the AS-8-present path becomes Plexus-KG substrate (deliverables flow directly to KG; envelope's `artifacts[]` references KG entries rather than filesystem paths). The substrate-layer commitment of ADR-025 holds — capability ensemble deliverables flow through a substrate; *which* substrate (filesystem vs. KG) is configuration-conditional. The always-scope decision is preserved regardless. This is genuinely new territory ADR-025 did not anticipate at drafting time; it surfaces as Cycle 7+ deliberation if Plexus activation reaches operational maturity. The dial-back falsification criteria above do not cover this case (filesystem-substrate dialing back to inline-response is one boundary; filesystem-substrate transitioning to KG-substrate is a different one).
- **Artifact-substrate for non-capability ensembles.** System ensembles, future ensembles authored against this contract — the substrate is per-capability-ensemble; if a future ensemble outside capability scope wants substrate-routing, a follow-on ADR extends the scope explicitly.

---

## Rejected alternatives

### Substantive-deliverable scope (size-floor + content-type criteria)

Substrate-routed: `code-generator`, `prose-improver`, `argument-mapper`, long-form `text-summarizer` invocations. Inline-response: `agentic-calibration-checker`, short `text-summarizer` invocations, `web-searcher` (already-handle-shaped). Per-ensemble declaration via `output_substrate:` field with operator overrides.

**Rejected because:** the scope reading captures spike β's per-finding map and matches spike α's size-floor finding cleanly, but it commits the cycle to a per-ensemble boundary judgment that varies by invocation (short vs long `text-summarizer` outputs; same ensemble, different scope). The boundary judgment in turn requires an in-process classifier or operator-set criteria that the cycle does not have empirical substrate to define crisply. Spike α's size-floor is *directional* (small outputs are perverse) but not *quantitative* (where exactly is the threshold?). The substantive-deliverable scope codifies a judgment the cycle has not earned the evidence base to make.

The practitioner's verbatim preference for **always** also weighs here: the cleaner design rule (one scope rule for all capability ensembles) over the more-nuanced rule (per-ensemble or per-invocation boundary). The "dial back later if cumbersome" disposition acknowledges the always-scope's tradeoffs without claiming the substantive-deliverable scope has currently-defensible criteria.

### Operator-configured scope

Per-ensemble YAML declares whether to substrate or inline. Maximum flexibility; minimum design opinion.

**Rejected because:** the operator-configured scope risks ad-hoc per-ensemble decisions accumulating without principled criteria. Without a default scope rule, each ensemble's authoring is a judgment call about substrate vs. inline — and the cycle's evidence base (spike α's size-floor + spike β's per-finding map + the practitioner's framing) does not give ensemble authors crisp guidance for the judgment. The result would be a deployment-by-deployment inconsistency where some ensembles substrate-route and others don't, with no architectural principle distinguishing them.

The operator-configured scope also makes the AS-7 amendment's conditional-skip harder to reason about: a session containing both substrate-routed and inline-response capability ensembles has two different summarization regimes operating simultaneously. The always-scope makes the regime uniform within capability scope (substrate-routed) and within system scope (inline-response).

### Defer scope to BUILD evidence — ADR-025 establishes the contract; BUILD's first deployments characterize the boundary

ADR-025 codifies the contract (envelope carries `artifacts[]` with typed shape; AS-7 amends to default-with-conditional-skip) but does not pre-specify which ensembles substrate. BUILD's per-ensemble migration produces the boundary characterization empirically.

**Rejected because:** this option moves the scope decision out of DECIDE into BUILD without the methodology mechanism for capturing the resulting boundary as a typed commitment. BUILD's per-ensemble migration would either re-converge on always-scope (the cleanest path forward) or accumulate per-ensemble judgments without an ADR documenting them — the same operator-configured-scope problem one layer down. Deferring scope to BUILD also delays the AS-7 amendment's backward propagation: the propagation sweep depends on which ensembles are scoped under the amendment; deferring scope defers propagation; deferred propagation accumulates the invariant-amendment drift Step 3.7 was designed to prevent.

The deferral is the kind of decision-deferred-past-its-deliberation-substrate this DECIDE phase exists to avoid. The cycle has the substrate (spike α + spike β + spike γ + the practitioner's framing); deferring is choosing not to use it.

### Reject the artifact-as-substrate proposal entirely; address findings 2/3/4 through other mechanisms

Output-spec drift via ADR-024's typed envelope alone; information-finding overhead via context-pruning at the orchestrator's reasoning surface; AS-7 stripping via summarizer improvements; orchestrator-narration substitution as a separate problem requiring a different mechanism.

**Rejected because:** substrate-routing addresses three of the four findings simultaneously with one structural change. The alternative mechanisms each have their own design and implementation cost (context-pruning is non-trivial; summarizer improvements require evidentiary substrate the cycle does not have; the narration substitution problem has no obvious lighter-weight mechanism). The substrate's overhead — artifact path management, session-dir lifecycle, client filesystem access — is a one-time architectural cost that subsumes three findings' worth of mechanism design. Spike β's per-finding test corrected the original three-findings-collapse claim's framing without eliminating substrate's structural value for three of the four findings.

The fourth-defect surface (orchestrator-narration substitution, surfaced by spike γ Cell A-explicit) is itself a finding the alternative-mechanisms proposal did not anticipate. Substrate-routing addresses it as a structural consequence; the alternative-mechanisms path would surface the question separately.

---

## Consequences

### Positive

- **The Skill Orchestration User mental model is honored at the deliverable layer.** Capability ensembles deliver substantive output to a known path; the client reads the artifact directly. The mental model of "the orchestrator dispatches my request to an ensemble that produces a deliverable I consume" is structural rather than reconstructable from orchestrator narration.
- **Three of four T16 findings dissolve at the substrate layer simultaneously** (information-finding overhead, AS-7 stripping, orchestrator-narration substitution). The fourth finding (output-spec drift) is addressed by ADR-024's typed envelope.
- **Session-dir grouping makes session-level review natural.** Operators inspecting a session's artifacts navigate `.llm-orc/agentic-sessions/<session_id>/` rather than reconstructing the session across multiple per-ensemble directories. Cleanup, archival, and Plexus ingestion all operate at session granularity.
- **Context cost stays bounded across dispatches.** The orchestrator's reasoning surface carries paths + summaries, not deliverable content. Long sessions with many dispatches accumulate session-dir artifacts on disk; orchestrator-context budget stays approximately constant per-dispatch.
- **AS-7's load-bearing concern is preserved for the inline-response path** while dissolving for the substrate-routed path. The amendment is principled rather than wholesale weakening — substrate-routing eliminates the context-rot concern; inline-response retains it.

### Negative

- **The size-floor tradeoff is deliberately accepted.** Substrate-routing for short or already-handle-shaped capability-ensemble outputs (notably `web-searcher`'s URL+snippet records, short `text-summarizer` invocations) adds overhead without proportional value. The cycle's commitment is to the cleaner design rule; BUILD's first deployments will surface whether the overhead is genuinely cumbersome. A follow-on cycle can refine to *when substantive* scope if needed.
- **The migration is non-trivial.** Cycle 6 BUILD relocates artifacts from `.llm-orc/artifacts/agentic-serving/<ensemble>/<timestamp>/` to `.llm-orc/agentic-sessions/<session_id>/<dispatch_id>/`. Operators with pinned references to the old path migrate manually. The old path is deprecated but not actively removed.
- **Filesystem access is required for clients in the local-first deployment.** Clients running on the same machine as the serve read artifacts from disk. Cross-machine deployments need the HTTP artifact-fetch endpoint, which is not in Cycle 6 BUILD scope. This narrows the architecture's deployment shape to local-first or local-network-with-shared-filesystem deployments until the HTTP endpoint ships.
- **AS-7 amendment triggers backward propagation across multiple prior documents.** ADR-004, ADR-007, ADR-014, ADR-021, essays 004 and 005 require sweep for AS-7 references. The sweep is methodology-mandated (Step 3.7); the cost is real but bounded.

### Neutral

- **`output_substrate:` field in ensemble YAML is operator-overridable.** Default for capability ensembles is `artifact`; ensembles opting out declare `output_substrate: inline`. The override mechanism preserves operator flexibility.
- **Retention defaults are per-ensemble configurable.** Substantive deliverables default to `session` retention; deliverables operators intend to commit can be declared `durable`. The retention mechanism composes with the always-scope without requiring scope-level commitments.
- **The substrate composes with ADR-023's dispatch-event substrate via `dispatch_id`.** Each artifact's path includes `<dispatch_id>`; events emitted during the dispatch share the identifier. Post-hoc joining of artifacts and events on `dispatch_id` is the natural review affordance.

## Provenance check

- **Artifact-as-substrate proposal**: 2026-05-14 follow-on verification finding 8, practitioner-verbatim (driver). Driver chain: same-cycle-practitioner-derived.
- **Three-findings-collapse claim does not hold per-finding**: spike β (driver), per MODEL Action B's required test. Driver chain: same-cycle-spike-derived + same-cycle-MODEL-derived.
- **Fourth-defect surface (orchestrator-narration substitution)**: spike γ Cell A-explicit (driver). Driver chain: same-cycle-spike-derived.
- **Always scope per practitioner verbatim**: 2026-05-14 follow-on finding 8 ("**always** rely on artifact writing") + practitioner DECIDE-entry deliberation answer (drivers). Driver chain: same-cycle-practitioner-derived.
- **Size-floor tradeoff deliberately accepted**: spike α finding (driver) + practitioner DECIDE-entry framing ("we can dial this back later if we feel it's cumbersome") — driver chain: same-cycle-spike-derived + same-cycle-practitioner-derived. The acceptance's rationale (cleaner design rule; dial-back-later disposition) is drafting-time synthesis composing the two drivers.
- **Session-dir location**: practitioner DECIDE-entry framing (driver, verbatim ".llm-orc/agentic-sessions/<timestamp>/some-structured-file"). The specific path shape (`<session_id>` as ISO-8601 + short UUID; `<dispatch_id>` as per-session counter; `<deliverable>.<ext>` as ensemble-author-chosen) is drafting-time synthesis refining the practitioner's framing.
- **Retention semantics (session / durable / ephemeral)**: drafting-time synthesis composing the session-dir location with the lifecycle question. Not driver-derived as a triplet; each retention category is design-time judgment.
- **Filesystem access for clients in local-first deployment**: drafting-time synthesis honoring the architecture's local-first commitment + the substrate's path-based reference shape. Cross-machine HTTP endpoint is named as out-of-scope; the local-first shape is the cycle's commitment.
- **AS-7 amendment to default-with-conditional-skip**: Cycle 6 MODEL OQ #15 (driver) + practitioner always-scope choice (driver). Driver chain: same-cycle-MODEL-derived + same-cycle-practitioner-derived.
- **OQ #13 narrows in scope to inline-response ensembles**: drafting-time synthesis composing OQ #15's amendment-pathway with OQ #13's evidentiary-pathway. The convergence framing was already in OQ #15; the per-pathway scoping (substrate dissolves OQ #13 for substrate-routed ensembles; inline-response retains it) is drafting-time application.
- **Partial updates to ADR-021 and ADR-004**: prior-ADR-derived (drivers); the specific update wording is drafting-time bridging.
- **Backward propagation sweep targets (essays 004, 005, ADRs 004/007/014/021)**: drafting-time identification of documents likely to reference AS-7 unconditionally. The sweep itself is Step 3.7 work executed in task #12 (post-drafting).
- **Rejection of substantive-deliverable scope**: drafting-time synthesis applying the absence-of-quantitative-threshold critique to the spike α directional finding. The practitioner-verbatim preference for always-scope is the load-bearing rejection rationale.
- **Rejection of operator-configured scope**: drafting-time synthesis applying the consistency-and-amendment-reasoning critique. Not driver-derived; design-time evaluation.
- **Rejection of "defer scope to BUILD"**: drafting-time synthesis applying the methodology's decision-deferral discipline. The Step 3.7 propagation requirement is methodology-driver-derived; applying it to the deferral option is the synthesis.

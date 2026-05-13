# RDD Cycle Status — Agentic Serving (Scoped)

**Artifact base:** `docs/agentic-serving/`
**Plugin version at cycle open:** v0.8.5
**Migration version:** 0.8.5 (`housekeeping/.migration-version`)

## Cycle Stack

### Active: Cycle 6 — Routing surface + observability (post-Cycle-5-PLAY pickup)

**Cycle number:** 6
**Started:** 2026-05-13
**Current phase:** discover (next)
**Cycle type:** mini-cycle
**Plugin version:** v0.8.5
**Artifact base:** `docs/agentic-serving/`
**Skipped phases:** research, model, synthesize (architect retained as possibly needed depending on DECIDE outcomes — see ARCHITECT note below)
**BUILD mode:** to be declared at BUILD entry (gated recommended given the design-alternative examination character of routing/observability work; auto mode appropriate only if BUILD reduces to mechanical wiring after DECIDE)

**Origin:** Cycle 5 PLAY (2026-05-13) chose **Path 1** — Thread A defects (4 broken capability ensembles + result-summarizer compression + `code-generator` coder timeout) handled as normal llm-orc dev work outside the methodology cycle; routing + observability axes opened as Cycle 6 scoped mini-cycle. Practitioner verbatim: *"I think path 1 is the way forward. Routing + observability need to be addressed."* Cycle 5 status archived at `cycle-archive/cycle-5-agentic-serving-library-structure.md`.

**Cycle 6 question framing (provisional, for DISCOVER):**

The cycle has two linked axes:

1. **Routing surface** — Cycle 5 PLAY note 20 disclosed the orchestrator's operational routing preference under both tested client configurations is **direct LLM completion → client-tool delegation → `invoke_ensemble` dispatch**, not ensemble-first-when-slot-fits as ADR-021's natural-language-supported clause implies. Cycle 6 asks: is this the routing surface the system wants, or is it a defect? If wanted, document the operative routing preference (and narrow or clarify ADR-021's natural-language-supported clause). If defect, what intervention (system-prompt work? dispatch-routing-policy ADR?) restores ensemble-first routing under NL framing?

2. **Observability** — Cycle 5 PLAY note 19 (sharpened by susceptibility snapshot) discloses the gap as **infrastructure-complete / routing-incomplete**. Cycle 5 BUILD shipped new internal events (verdicts, tier-routing decisions, audit consumption, signal-channel aggregation); the architecture has the telemetry; what is missing is the routing of telemetry to human-visible surfaces. Cycle 6 asks: which surfaces should receive which events? Operator-terminal (colored logs? TUI dashboard?) for the Ensemble Author / Operator stakeholder; orchestrator-context-includes-execution-state for the orchestrator's reasoning surface so it can answer the timing/graph questions a Skill Orchestration User asks.

The two axes are linked: the operator cannot tell what routing decision happened without observability; the orchestrator cannot refine its routing decisions without visibility into its own dispatches.

**MODEL handling for Cycle 6:** Skipped as a standalone phase per Mode D shape. New vocabulary that surfaces in DISCOVER (e.g., "routing surface," "operator-visible event surface," "orchestrator execution context," "tier-routing decision," etc.) folds into DISCOVER's tail as Amendment Log entries on `domain-model.md`. If DECIDE deliberation reveals vocabulary territory warranting a dedicated MODEL phase, the cycle's `Skipped phases:` field can be amended mid-cycle.

**ARCHITECT handling for Cycle 6:** Initially retained as possibly needed. If DECIDE outcomes specify a new operator-visible event surface module (e.g., a TUI dashboard component or a structured-logging surface module), ARCHITECT runs to allocate responsibilities and dependencies. If DECIDE outcomes are extensions to existing modules (e.g., extending Serving Layer with structured log output, extending Orchestrator Runtime with an execution-context-population API), ARCHITECT may be skipped. The cycle's `Skipped phases:` field will be updated at DECIDE close based on which path the decisions take.

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

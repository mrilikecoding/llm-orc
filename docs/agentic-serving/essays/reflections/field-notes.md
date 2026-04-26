# Field Notes — Agentic Serving, Play Phase

**Play session:** 2026-04-24
**Practitioner:** Nathan Green (cycle owner; tool user, operator, and ensemble author roles collapsed per assumption inversion #3)
**System state at play:** WP-I close — TS-2 reached + Plexus Adapter skeleton wired with no-op fallbacks. Branch `agentic-serving`. `llm-orc serve` running on port 8765 with `orchestrator-local` as the configured Model Profile.
**Client:** OpenCode (sst/opencode TUI), pointed at `http://localhost:8765/v1` via a global provider entry in `~/.config/opencode/opencode.json`.
**Stakeholders inhabited:** Pure Tool User (primary). Ensemble Author / Operator surface observed concurrently — `uv run llm-orc serve` terminal was open in a second window during the Pure Tool User session.

---

## Stakeholder: Pure Tool User

**Super-Objective:** Complete coding work through an agentic coding tool, treating the endpoint as a model. No interest in what's under the hood; no investment in seeing inside.

**Point of Concentration:** Just use it for real work — pick one coding task you would actually do today, run it through, attend to what the system discloses.

### 1. Slow turn latency

**Category:** Usability friction
**Observation:** First turn ("Tell me about this project") took 1m 55s. Second turn (clarifying pushback) took 1m 53s. Third turn ("write a summary file") took 2m 30s. Practitioner self-reported "expected with some local model orchestration" — the latency was acknowledged and tolerated but named as friction.
**Feeds back to:** DISCOVER (as value tension — value tension #1, *Quality vs. cost vs. speed*, in this configuration tilts toward cost-with-no-quality and slow-with-no-progress)

### 2. First response described OpenCode, not llm-orc

**Category:** Challenged assumption
**Observation:** Asked "Tell me about this project" while the tool was running from the llm-orc directory. The response described OpenCode itself ("interactive CLI tool powered by the orchestrator-local model... helps users with software engineering tasks") rather than the llm-orc project the practitioner was inside. The orchestrator did not orient on the practitioner's working directory or project context.
**Feeds back to:** DISCOVER (as assumption inversion — the implicit assumption that an OpenAI-compat client + an orchestrator backend will orient on project context is not borne out by the default configuration)

### 3. Hallucinated project structure on clarified request

**Category:** Challenged assumption
**Observation:** After explicit pushback ("Ah no, I mean tell me about the project in the directory we're in please"), the response described llm-orc as a TypeScript/Node project with a React frontend, `packages/`, `ensemble-library/llm-orc/tasks/`, npm/Jest/ESLint commands. llm-orc is a Python project with an entirely different layout. The orchestrator narrated as if it had executed `git_status`; no evidence the call actually fired.
**Feeds back to:** DISCOVER (as assumption inversion — the "endpoint is a model" mental model from product discovery presupposes a model competent enough to either use available tools or refuse, rather than fabricate; the default profile fabricates)

### 4. File-write request: no file written, response narrated as if it had been

**Category:** Challenged assumption + Missing scenario
**Observation:** Asked "Can you write a summary file for me in this directory that articulates what you observe about this project and its purpose?" The response described the file's contents in detail and presented an analysis. No file was written to disk; practitioner verified directly. The Option C turn-boundary delegation that WP-F wired (client-declared `tools[]` → `finish_reason: tool_calls`) presupposes the orchestrator LLM recognizes available client tools and emits tool_calls; the encounter suggests the default `orchestrator-local` profile does not meet that capability floor.
**Feeds back to:** DISCOVER (as assumption inversion — Option C delegation assumes a capability floor the default profile does not name) + DECIDE (as missing scenario — there is no scenario in `scenarios.md` that specifies what minimum competence an orchestrator Model Profile must demonstrate to invoke ADR-003's five internal tools and to recognize client-declared tools for Option C delegation)

### 5. No in-stream visibility narration appeared

**Category:** Interaction gap
**Observation:** Practitioner verbatim: *"having some output returned to the opencode interface about what was happening under the hood in llm-orc would have been informative."* No `[kind: {json}]` narration appeared in the OpenCode response stream across any of the four turns. The narration committed in `interaction-specs.md` (Operator §"Observe orchestrator behavior during sessions") and resolved at WP-E only renders when VisibilityEvents fire. In this session, nothing composed and no ensemble was actually invoked, so no events were generated. The spec is internally consistent given the cycle's WP-E commitment; the encounter is the absence the spec did not anticipate.
**Feeds back to:** interaction-specs (the Operator-observability task assumes events fire; the spec does not address what an observer encounters when the orchestrator is incapable of doing the things that would generate events) + DISCOVER (as assumption inversion — the cycle's resolution of OQ #2 produces visibility *conditional on capability*; the assumption that the resolution delivers visibility in a default-config first session is not borne out)

### 6. Budget exhaustion on what should have been the simplest request

**Category:** Missing scenario + Delight (split observation)
**Observation:** Fourth request ("show me the contents of README.md") returned: *"Session budget exhausted: token limit reached (50080/50000)."* Budget Controller fired per AS-3 and the interaction-spec "Experience budget exhaustion cleanly" — the clean termination is structurally correct. The surrounding experience: ~6 minutes wall-clock, 50K tokens, four turns of hallucination and non-delegation, then the simplest concrete request died on a control-plane message the pure tool user has no context for. The token cap was consumed by hallucination before any tool actually executed.

The "delight" routing applies a spec-validation lens (AS-3 fired) to a question that is also a stakeholder-experience question. Whether Budget enforcement that terminates a session after 50K tokens of hallucination is an acceptable first-session experience is *not* answered by AS-3's correctness; that question is open for the next play round. The split classification is honest only when the deferral is named.
**Feeds back to:** SYNTHESIS (delight — Budget enforcement worked as designed; AS-3 holds in practice) + DECIDE (missing scenario — pre-runaway hallucination-burn is a failure mode distinct from runaway loops; Calibration Gate per ADR-007 currently scopes to composed ensembles, but in this encounter no ensemble composed, so no calibration could fire; whether top-level orchestrator hallucination is in-scope for any quality check at all is an open scenario)

### 7. Encountered token cap was 50K, not the documented 10M

**Category:** New question
**Observation:** Cycle-status FF #39 records the default `token_limit=10_000_000` as the local-orchestration value-prop sizing decided in WP-C. Encountered cap was 50,000. Practitioner verbatim: *"50K is obviously extremely low."* The practitioner's prior is that the encountered value is a misconfiguration, not a deliberate redesign. Three explanatory hypotheses follow, but they are not equiprobable in light of the practitioner's settled judgment: (a) practitioner's local `config.yaml` overrides the default — *strongly favored*; (b) the default has shifted since FF #39 was recorded; (c) the encountered cap has different semantics from the documented one (e.g., per-Session vs. per-some-other-window).
**Feeds back to:** RESEARCH (new question — investigate the actual default token cap shipped today and how it surfaces in operator config; resolve before the next play round so encountered numbers are interpretable against documented ones; the practitioner's prior favors hypothesis (a) and the investigation should test that first)

### 8. Pure Tool User verdict

**Category:** Challenged assumption
**Observation:** Practitioner verbatim: *"If I were to install llm-orc and run it with OpenCode like this out of the box, I would not be likely to use it again."*
**Feeds back to:** DISCOVER (as assumption inversion — assumption inversion #3 in product discovery names the pure-tool-user persona but assumes a graceful-onboarding default; the cycle has not yet specified what onboarding produces a tolerable first session, and the current default produces an unrecoverable one)

---

## Stakeholder: Ensemble Author / Operator

**Super-Objective:** Maintain a library of ensembles, profiles, and scripts the orchestrator uses effectively, while observing how the system uses that library and improving it through tinkering and organic stabilization.

**Point of Concentration:** Incidental concurrent observation — `uv run llm-orc serve` was running in a second terminal throughout the Pure Tool User session.

### 9. Server terminal silent during four-turn session

**Category:** Interaction gap
**Observation:** No log output emitted to the `uv run llm-orc serve` terminal during any of the four turns. Practitioner verbatim: *"as an operator / ensemble author there's no visibility into what models / profiles / ensembles are being invoked, so I don't feel there's a good UX from that perspective either."*
**Feeds back to:** interaction-specs (the Operator's "Observe orchestrator behavior during sessions" task assumes a visibility surface; in practice the server-side default emits nothing) + DISCOVER (as value tension — value tension #5, *Visibility: what form?*, was resolved for the in-stream surface (`delta.content` narration) but the operator-terminal surface remains unspecified)

### 10. Bilateral visibility absence

**Category:** Challenged assumption
**Observation:** Visibility was simultaneously absent on both surfaces — no `[kind: {json}]` narration in the tool user's stream, no log activity in the operator's terminal. Either surface alone would be a partial gap; their simultaneous absence means neither stakeholder has any recourse during the session: the tool user cannot interpret what is happening, and the operator cannot debug what is happening.
**Feeds back to:** DISCOVER (as assumption inversion — the two-audience visibility framing the cycle developed in DECIDE / WP-E assumes operator and tool-user surfaces compose into coverage; in this session both are empty, so the framing's assumption that *some* surface always carries signal is not borne out)

---

## Cross-cutting reflection

**Which stakeholder had the hardest time?**
Pure Tool User. Long turns, hallucinated results, exhausted tokens within three turns on basic starting tasks. No work could be done with this system in this configuration. The Operator side was identical in shape — no logs emitted, no observability into models / profiles / ensembles invoked. From either seat the experience was incoherent.

**What did play reveal that the specs missed?**
Observability, at a more fundamental level than the cycle's prior framing covered. The cycle resolved OQ #2 with a commitment to inline `[kind: {json}]` narration on `delta.content`, which is sound *when events fire*. Reality in a default-config first session: no events fire (because the orchestrator never composed and never invoked an ensemble), and no operator-side log surface exists. The specs assumed a capability floor and an operator-side telemetry surface that the default deployment does not provide.

The default ensemble itself is also a gap. There is no current indication that the default `orchestrator-local` profile can compose ensembles or invoke them effectively. The capability floor of an "orchestrator-capable Model Profile" is not specified anywhere in the cycle's artifacts; without it, the operator could not optimize even if telemetry were present, because there is no specification of what the orchestrator should be observably *doing*.

**How has your understanding of the system shifted?**
Two real wins surfaced: the protocol-level integration with OpenCode works (provider configuration, model selection, request routing, SSE streaming), and Budget enforcement worked exactly as designed (AS-3 holds in practice). What is not yet present is a usable default first-session experience. The system as shipped at WP-I close has the right structural components but produces a broken-on-arrival first encounter for both stakeholders, with no in-session recourse for either. 50K as the encountered token cap is "obviously extremely low" — a separate finding pending the FF #39 reconciliation.

The framing the practitioner offered at session end: *"This is all pretty fundamental stuff we can address and come back to in a next round of play."* The fundamentals are observability (both surfaces) and the orchestrator-capable Model Profile floor.

### Scope of the Pure Tool User verdict

The verdict — *"I would not be likely to use it again"* — applies to **the default configuration as encountered**. Whether it changes with a stronger orchestrator profile is not tested in this play session. The next play round should include a profile configured for success (a more capable orchestrator model, with a token cap aligned to the documented 10M default if hypothesis (a) of note 7 holds) — to determine whether the verdict is configuration-dependent or architecture-dependent.

A further refinement from the practitioner at the close-out, important for routing the visibility findings (notes 5, 9, 10): the desire for visibility was itself **failure-mode-conditional**. Practitioner verbatim: *"Presumably a better or more competent configuration would not have led me to want more observability. But the lack of good response and the length of time they took made me want to understand what was happening."* In a working session, the Pure Tool User's "endpoint is a model" mental model holds — they would not crave inline narration. In this failing session, the absence of any signal during long, hallucinating turns is what made the visibility gap intolerable. This refinement matters for the design: the in-stream visibility narration committed at WP-E may need to be evaluated not only on whether it fires when composition events happen, but also on whether *some* signal carries during slow or stalled turns even when no events have fired. The specification at present treats narration as event-conditional; the practitioner's encounter suggests narration may also be experience-conditional. **This is a candidate finding for the next DISCOVER pass to articulate as a value tension or assumption inversion** — not a settled conclusion from this session.

---

## Field-note routing summary

| Destination | Notes routed |
|---|---|
| **DISCOVER** (assumption inversions / value tensions) | 1, 2, 3, 4, 5, 8, 9, 10 |
| **DECIDE** (missing scenarios) | 4, 6 |
| **interaction-specs** (interaction gaps) | 5, 9 |
| **RESEARCH** (new question) | 7 |
| **SYNTHESIS** (delight — partial) | 6 |

A note may route to more than one destination; the table records the primary route for each.

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

---

# Play session: 2026-05-12 (Cycle 4 PLAY)

**Practitioner:** Nathan Green (cycle owner; Ensemble Author / Operator inhabitation, with the tool-user role collapsed per assumption inversion #3 since the practitioner was sending prompts through OpenCode as well).
**System state at play:** Cycle 4 BUILD structurally complete (WP-A4 through WP-H4 closed 2026-05-11/12); ADR-016 in full acceptance; suite at 2656 passing. Branch `agentic-serving`. PLAY entered 2026-05-12 after BUILD-close gate; inhabitation work deferred to a fresh session per practitioner cognitive-budget call.
**Configuration at play:** Local `.llm-orc/config.yaml` extended with the Cycle 4 `agentic_serving:` section (per-skill tier defaults across all 8 Topaz skills, tier-router-audit defaults, conversation-compaction defaults, free-tier preference). Orchestrator: `orchestrator-minimax-m25-free` (MiniMax M2.5-free via `openai-compatible/zen`, `https://opencode.ai/zen/v1`). Cheap-tier general: qwen3:8b; cheap-tier summary: qwen3:1.7b; escalated general: qwen3:14b; escalated reasoning: deepseek-r1:8b. Two ensembles tagged with `topaz_skill` metadata: a new `agentic-coding-helper` (`code_generation`), and the existing `development/code-review` (`instruction_following`).
**Client:** OpenCode (sst/opencode TUI), pointed at `http://localhost:8765/v1` via the global provider entry in `~/.config/opencode/opencode.json`.
**Stakeholder inhabited:** Ensemble Author / Operator (full inhabitation, not concurrent observation as in Cycle 1 PLAY).

**Categorization deferred** until session close. Notes below are raw observations in the order they surfaced during the session.

---

## Stakeholder: Ensemble Author / Operator

**Super-Objective:** Maintain a library of ensembles, profiles, and scripts the orchestrator uses effectively, while observing how the system uses that library and improving it through tinkering and organic stabilization. See the orchestrator. Let the system stabilize organically.

**Point of Concentration:** Configure per-skill tier defaults (ADR-015) and run a single agentic-coding turn through the WP-H4-complete stack. Attend to what becomes visible (or doesn't) when calibration verdicts, tier routing, and the cross-layer calibration channel all operate together for the first time.

### 1. The cycle's configuration surface required a separate authoring pass

**Observation:** Opening `.llm-orc/config.yaml` and `.llm-orc/ensembles/` as the Ensemble Author / Operator with the expectation that "based on all the work we've done" there would be a clear set of default profiles and ensembles for agentic serving, what was actually shipped was the `summarizer` profile (ADR-004 / WP-D), a handful of `agentic-*` system ensembles, and an unmigrated library. ADR-015 §Negative ("library migration is operator-driven") reads cleanly at decide time and at gate close; at first-encounter with the empty configuration surface, the framing produced a gap the operator could not close from the shipped state without a separate authoring pass. Practitioner verbatim: *"Yes, I chose to automate much of the decide / build, so that's my oversight. But the agentic-serving config is to me part of the build."* The auto-mode of BUILD shipped the mechanism architecture but did not ship working defaults; the operator-facing on-ramp got externalized to the operator-driven-migration framing.

### 2. The Cycle 4 stack activated cleanly on the first session that exercised it

**Observation:** With the per-skill tier defaults and two tagged ensembles in place, the first OpenCode turn produced a dispatch through the full WP-H4-complete path. Artifact `agentic-coding-helper/20260512-150649-612/execution.json` shows all 3 sub-agents (coder, critic, synthesizer) routed to `model_profile: tier-cheap-general` (qwen3:8b). The Tier-Escalation Router constructed at session start; the Calibration Gate produced Proceed verdicts on each dispatch (no escalation); the summarizer (qwen3:0.6b under the `summarizer` profile) ran on the return path. Total ensemble duration 151.79s; 4,538 tokens (719 input, 3,819 output); cost $0 on the ensemble side. The mechanism architecture is in code and visibly fires on first contact with a real client.

### 3. The orchestrator routed a meta-introspection prompt to a code-generation ensemble

**Observation:** Practitioner prompt: *"I am testing out the agentic-serving orchestration in .llm-orc/config.yaml. This is a meta-test. So tell me what you can tell about your own internals using this agentic-serving flow."* The orchestrator (MiniMax M2.5-free via Zen) translated this into `invoke_ensemble("agentic-coding-helper", {"data": "Write a Python function that implements a simple LLM orchestration router using only five tools: list_ensembles, invoke_ensemble, compose_ensemble, query_knowledge, and record_outcome."})`. The introspection question became a code-generation task. The orchestrator did much more than that single dispatch — the meta-analysis it returned is grounded in file reads (see note 5) — but the *single ensemble dispatch* the cycle's machinery exercised was a code-task reification of a question that was not, in any obvious reading, a code-task. Whether that is the right routing OR whether the right routing would have been "no library dispatch; answer directly using client-tool file reads against the running config" is the open question.

### 4. The `MissingSkillMetadataError` typed-error recovery loop fired and worked

**Observation:** The orchestrator, exploring the library, repeatedly hit the ADR-015 typed error on untagged ensembles. Practitioner-paste verbatim from the orchestrator's narration: *"The ADR-015 tier router is enforcing metadata requirements... Let me try with an ensemble that has the required topaz_skill metadata... The tier router consistently enforces the topaz_skill metadata requirement. Let me check which ensembles properly declare it."* The orchestrator reformulated on each error and eventually settled on the two tagged ensembles. This is WP-G4-1's typed-error recovery path (`recovery_action_required="reformulate"`) firing exactly as designed — the typed error reaches the orchestrator's reasoning surface as feedback, and the orchestrator chooses a different dispatch on the next iteration. First empirical confirmation of the recovery path under a real client.

### 5. Client-declared file tools carried most of the introspection load

**Observation:** The orchestrator's meta-analysis was grounded in `file_read`, `Glob`, and `Grep` against the project tree — client-declared tools, not the internal 5-tool surface. It read `.llm-orc/config.yaml`, the ADR files, the relevant ensemble YAMLs, and ran a grep for `topaz_skill:`. The Option C delegation discipline (system-design.md §Client Tool Surface Commitment; one-kind-per-turn) held across many turns. Contrast with Cycle 1 PLAY note 3 (orchestrator narrated as if it had executed `git_status` without actually calling it); the Cycle 4 orchestrator's tool calls actually fired. Whether this is a model-quality property (MiniMax M2.5 vs. the Cycle 1 PLAY's local default), a system-prompt-discipline property (the system prompt explicitly teaches the one-kind-per-turn rule), or both is not separable from this single encounter.

### 6. The Cycle 1 PLAY's "fabrication on first encounter" did not recur

**Observation:** The Cycle 1 PLAY's most consequential finding was that the default `orchestrator-local` profile (likely qwen3:0.6b) fabricated project structure, narrated tool calls that never fired, exhausted budget on hallucination, and produced an unrecoverable first session. The Cycle 4 PLAY's first session, under the cheap-cloud-orchestrator pattern (MiniMax M2.5-free via Zen), grounded its responses in actual file content, used tools correctly, hit typed errors and reformulated, produced a coherent meta-analysis, and stayed within the 1M-token / 100-turn global budget. The Cycle 1 verdict *"I would not be likely to use it again"* was profile-conditional: the local-default-as-orchestrator deployment was unrecoverable; the cheap-cloud-orchestrator-as-routing-layer deployment is workable. The cycle's research framing (essay 005's "cheap-cloud-orchestrator routes; local models amplify deterministic and bounded-scope ensemble work") is borne out at the first-session shape. Cycle 1's scope-of-verdict caveat (note 8 closing section) is now empirically anchored to the local-vs-cloud orchestrator distinction.

### 7. The visibility gap persists from Cycle 1 to Cycle 4

**Observation:** Practitioner verbatim while the session was running: *"it'd be particularly helpful in the llm-orc serve output to see colored log output or even some basic TUI dash for observability of what ensembles are getting invoked / what models, etc."* The Cycle 4 work added internal events at every dispatch — `TierSelection` values (model_profile, tier, topaz_skill), `CalibrationVerdict` values (Proceed/Reflect/Abstain), Tier-Router-Audit consumption records, cross-layer Signal Channel aggregations — and none of them surface to the operator without code changes. The serve terminal during this session emitted only Pydantic validation warnings for two legacy ensembles (`fan-out-test.yaml`, `plexus-graph-analysis.yaml` — pre-existing schema drift unrelated to Cycle 4) and request-path messages. The bilateral-visibility shape from Cycle 1 PLAY notes 9–10 carries through to Cycle 4: the orchestrator's surface is rich and grounded, the operator's terminal surface is essentially empty of dispatch-level signal. The novel content in this finding versus Cycle 1's is the *concrete shape* the operator can now name for the missing surface — colored logs and/or a TUI dashboard — and the *specific list* of events that exist internally but do not surface (verdicts, tier-routing decisions, audit records, signal-channel state).

### 8. The orchestrator's meta-analysis quoted the config.yaml comments verbatim

**Observation:** The meta-analysis produced by the orchestrator cited specific ADR numbers (ADR-003, -004, -007, -009, -011, -012, -014, -015, -017, -018), specific threshold values (N=3 invocations before trust, 15% verdict-distribution-shift, 100-invocation or 24-hour audit triggers), and specific component names (Calibration Gate, Compaction, Tier Router, Tool Validation, Audit) — all of which came directly from the inline comments authored in the `agentic_serving:` config section. The orchestrator's "self-description" is more accurately "in-context summary of what was just read." Two implications: (a) inline comments in config files are doing useful work as in-context grounding for orchestrator self-narration; (b) the orchestrator's introspection is comprehension of explicit documentation, not independent reasoning about its own internals — a property easy to mistake for genuine self-knowledge given the meta-analysis's surface coherence. The orchestrator counted *"only 3 of 74 ensembles declare it [topaz_skill]"*. Verified post-write: 2 ensembles declare it (`agentic-coding-helper`, `development/code-review`) out of 57 YAML files under `.llm-orc/ensembles/`. The 74 likely reflects the `list_ensembles()` count which aggregates local + library + global sources; the 3 vs. 2 mismatch is unaccounted-for — possibly a grep over-count if the orchestrator pattern-matched on `topaz` rather than `^topaz_skill:`. Both numerator and denominator were wrong in a surface-coherent meta-analysis. Worth recording because the error mode is *coherence without factual grounding* — exactly the failure shape ADR-014's Calibration Gate is designed to catch, but the gate only fires on dispatched-ensemble outputs, not on the orchestrator's own narration. The orchestrator's reasoning surface remains unaudited.

### 9. The Cycle 1 PLAY's 50K-vs-10M token-cap mystery is structurally undisturbed

**Observation:** The global config (`~/.config/llm-orc/config.yaml`) carries `budget.token_limit: 1000000` from a 2026-04-29 CAP-9 spike setup; the resolved per-session budget on this session was 1M tokens / 100 turns. The Cycle 1 PLAY encountered a 50K cap. 1M is not 50K. Wherever the 50K came from, it was not the documented global budget at this date. The current session did not approach 1M (~5K tokens consumed on the single dispatch), so this round did not produce new evidence on the cap-source question. Follow-up still open: if a future session hits a cap that isn't 1M, that's the trail to follow; if all future sessions hit at 1M, the Cycle 1 50K is most likely a stale artifact of a prior config version.

### 10. The orchestrator's recommendation mode performs differently from its introspection mode

**Observation:** Asked *"can you recommend a good basic task for us to attempt? I'd like to evaluate the agentic-serving functionality,"* the orchestrator produced a well-structured 3-option ranked proposal — each option targets a different ADR's mechanism (#1 ADR-015 tier router; #2 ADR-007 calibration gate; #3 ADR-004 summarizer round-trip), each scoped to discrete observable outputs, each with a rationale for ranking. The analytical-structure quality is high: the orchestrator can identify which cycle mechanism each test would load and design dispatch sequences against it. But the recommendations contain a primitives-level category error (see note 13) and reference *"the ensemble-designer skill to tag 5-10 ensembles"* — `ensemble-designer` is a Claude Code skill external to llm-orc's tool surface, not something the orchestrator can dispatch via `invoke_ensemble`. Recommendation mode produces high-quality analytical structure with low-quality primitive grounding. The two failure modes (structure vs. primitives) are independent: a recommendation can be analytically sharp and primitively wrong simultaneously.

### 11. The orchestrator's count claim self-corrected between turns; the unchallenged denominator did not

**Observation:** Round 1 (recommendation turn): *"only 3/74 ensembles declare topaz_skill."* Round 2 (Topaz explanation turn, after re-running `Grep "Topaz|topaz_skill"`): *"only 2/74 ensembles have it (code-review, agentic-coding-helper)."* The numerator self-corrected from 3 to 2 when the orchestrator re-ran verification. The `/74` denominator persisted across both turns; the verified actuals are 2 of 57 (local) or 2 of 79 (across all sources) — the 74 is the count from `list_ensembles()` MCP output, which applies its own dedup/filter logic. The orchestrator refines claims that re-running verification touches; it does not spontaneously cross-check claims that no part of the conversation challenges. Unchallenged-claim-stickiness is its own failure mode separate from the count-error mode: the calibration only fires where the conversation puts pressure.

### 12. Routing-logic framing tightened across turns

**Observation:** Round 1's description of the tier router: *"Verify they route to the expected tier (cheap vs escalated) based on skill classification."* Round 2's description: *"Routes to either cheap_tier or escalated_tier based on calibration state."* Round 1 conflates skill (which determines *which slot* in `per_skill_tier_defaults` is consulted) with verdict (which drives the *cheap-vs-escalated* decision within that slot); Round 2 separates them correctly. The orchestrator's first-pass framing of a mechanism is less accurate than its second-pass framing under continued engagement. A short conversation continues the calibration; a single-turn snapshot is more error-prone than the orchestrator's narrative coherence would suggest. The session-length / framing-accuracy relationship is empirically suggested but not isolated from "the practitioner happens to ask a follow-up that triggers re-engagement" as the operative variable.

### 13. The `compose_ensemble` primitives were misunderstood

**Observation:** The orchestrator's Option 2 test recommendation: *"Compose a simple 2-agent ensemble (e.g., list_ensembles → summarize). Invoke it 4 times. Verify: invocations 1-3 get checked, invocation 4 transitions to trusted."* This is a category error — `list_ensembles` is an internal tool from the closed 5-tool surface, not an agent. `compose_ensemble` creates DAGs of agents (LLM agents + script agents), not chains of internal tools. The proposed test would fail at `compose_ensemble` dispatch time because `list_ensembles` is not a valid composition primitive. The error is structurally similar to Cycle 1 PLAY note 3 ("narrated tool call that never fired"); the Cycle 4 difference is that this error is in a *recommendation* rather than an *executed action*, so it would surface only when the recommendation is run. The orchestrator confidently proposed a test that would not execute as described. The composition surface (what `compose_ensemble` actually accepts) is documented in ADR-003 and the system prompt, but the orchestrator's recommendation drifted from that spec into a plausible-but-wrong shape.

### 14. Architectural Isolation maps cleanly from RDD's tier-1 mechanism to `invoke_ensemble`

**Observation:** RDD's tier-1 sycophancy-resistance mechanism (ADR-058 / Invariant 8) is *Architectural Isolation*: specialist subagents run in fresh contexts to exploit the Self-Correction Blind Spot. `invoke_ensemble` in llm-orc dispatches the target ensemble's agents with `input + system_prompt` only — they do not inherit the orchestrator's conversation history. These describe the same architectural property. The mapping is structural and requires no ADR amendment: RDD-via-agentic-serving is feasible on the existing primitive surface. This is a non-trivial result for the cycle's design — a sycophancy-resistance pattern from a methodology framework lands intact in llm-orc's dispatch path, without either side knowing about the other.

**Attribution (per Cycle 4 PLAY susceptibility snapshot):** This architectural mapping was introduced by the agent during the session, not generated by practitioner inhabitation. The practitioner did not object but did not independently surface the framing. Treat as a candidate finding requiring DECIDE-phase examination, not as a settled inhabitation discovery.

### 15. Methodology-layer / dispatch-layer / execution-layer separation reframes the operator-driven migration

**Observation:** The cleaner architectural framing (practitioner's verbatim refinement): *"RDD delegates to sub-skills which decompose to specific kinds of tasks that the orchestrator can route to the appropriate ensemble. Llm-orc doesn't need to have an rdd lit-reviewer ensemble any more than it needs a 'rdd-research' ensemble, but it would need to be able to invoke an ensemble capable of doing the tasks needed by that skill or subskill."* The three-layer structure: (a) methodology layer lives client-side (rdd:* skill plugin, or any other methodology); (b) dispatch layer is the orchestrator with the Topaz 8-skill taxonomy as its routing index; (c) execution layer is the ensemble library where ensembles are **operation-named** (e.g., `web-searcher`, `claim-extractor`, `argument-mapper`) rather than **methodology-named** (e.g., `rdd-lit-reviewer`). Naming the library after operations keeps the orchestrator methodology-agnostic; the same library serves many methodology consumers. ADR-015 §Negative's operator-driven library migration reads more concretely under this framing: not "tag every existing ensemble with whatever topaz_skill fits"; rather, "author the **operation-named** capability ensemble set that serves the deployment's methodologies." Different deployments need different capability sets within the same slot vocabulary. The cycle named the slot-level discovery question; the operation-level question is downstream and deployment-specific. Sub-Q6 (autonomous-routing reliability across many ensembles) becomes load-bearing for multi-methodology deployments in a way the cycle's single-task framing did not anticipate.

### 16. The Calibration Gate does not audit the orchestrator's own narration

**Observation:** Notes 8, 10, 11, 12, and 13 record specific factual or category errors in the orchestrator's responses to the OpenCode client. None of these errors were caught by anything in the running stack. The Calibration Gate (ADR-007/014) fires the post-hoc result-check on the **output of `invoke_ensemble`** — it inspects what a dispatched ensemble produced. The structural validation guard (ADR-017) fires on **tool-call patterns** — it catches phantom tool calls. Neither mechanism inspects the natural-language claims the orchestrator emits to the client. The orchestrator's meta-analysis — including incorrect counts, profile conflations, mistier-allocations, and the `compose_ensemble` category error — reached OpenCode uncalibrated. The cycle's quality infrastructure addresses two failure modes (dispatched-ensemble low-confidence output; phantom tool calls) but does not address a third: *the orchestrator emits a coherent-sounding factual claim that doesn't survive verification.* This is a coverage gap of the existing infrastructure, not a bug — the cycle's research did not name this failure mode as a target. Whether it should be (and where calibration on orchestrator-own-output would even attach in the architecture) is open territory.

### 17. The orchestrator produces substantive gap analysis on evaluative tasks

**Observation:** Asked to read the play-derived proposal doc and identify (a) capability ensembles missing for an RDD research workflow and (b) which ensemble to author first, the orchestrator surfaced three load-bearing gaps with accurate framing: `lit-review` requires web-search (matches OD-2 in the proposal); `citation-audit` needs a `citation-verifier` paired with `claim-extractor` (not in the proposal as a standalone capability); `synthesis` is not covered by any proposed ensemble (text-summarizer is summarization, not synthesis). The decomposition the orchestrator surfaced (lit-review → citation-audit → framing-audit → synthesis) is grounded in RDD's actual phase shapes from reading the proposal's references to corpus artifacts. This is the orchestrator's *useful* mode — its analytical structure on the gap identification is grounded in file content and tracks the proposal's stated intent.

### 18. The recommendation justifications carry the same coherent-but-incorrect error pattern as introspection

**Observation:** The orchestrator's recommendation (claim-extractor first; then pair with tool_use to unblock lit-review) is directionally defensible. The supporting claims contain factual errors: *"It's the only proposed ensemble with a complete spec"* (all five have specs of roughly equal completeness); internal inconsistency (primary justification calls claim-extractor's spec "complete"; runner-up calls argument-mapper's spec "cleanest" — both cannot hold); *"Exercises logical_reasoning on the escalated tier"* (conflates the slot's escalated profile assignment with the verdict-driven routing — argument-mapper would route to cheap-tier on Proceed, escalated only on Reflect). The verifiable structural recommendation is useful; the justification chain is not load-bearing for the recommendation but is corrosive if read uncritically. This is the same error mode notes 8, 10, 11, 12, 13 record — across introspection, recommendation, and evaluation tasks, the orchestrator's reliability is *high on derivable claims* and *low on integration claims*. The error pattern is consistent enough to be characterized as a feature of the orchestrator's reliability profile rather than a one-off.

### 19. The no-dispatch fallback path observed: cycle infrastructure does not fire at all

**Observation:** The proposal-evaluation prompt (~700 words of practitioner input) produced 53 seconds of orchestrator reasoning, one `Read` tool call against the proposal doc, and zero `invoke_ensemble` calls. No new artifact directories in `.llm-orc/artifacts/`. The Tier-Escalation Router did not activate; the Calibration Gate did not produce a verdict; the Tier-Router Audit recorded nothing; the summarizer did not run; the cross-layer signal channel (had it been wired) would have received no signals. The cycle's entire quality infrastructure was moot for this task — the orchestrator's natural-language LLM response reached the OpenCode client with no calibration, no routing decision, no audit trail. The cycle's quality mechanisms do not fire under no-routable-task conditions: when no library ensemble matches the task's shape, all the dispatched-ensemble quality mechanisms are bypassed. Note 16's coverage gap (no calibration on orchestrator's own narration) compounds with this no-dispatch fallback (no routing means no opportunity for any mechanism to fire). The prevalence of this path in a real deployment will depend entirely on library coverage — with the current 2-tagged-ensembles state, most tasks land here; with a fuller library (per the proposal), more tasks would dispatch but evaluative and meta-tasks may always fall through.

**Attribution (per Cycle 4 PLAY susceptibility snapshot):** The underlying observation (zero `invoke_ensemble` calls; full quality infrastructure bypass on this prompt) is empirical. The "fail-open" framing was introduced by the agent. Treat the observation as load-bearing; treat the characterization as a candidate framing for DECIDE-phase examination.

---

## Cross-cutting reflection

**Evidence-basis note (per Cycle 4 PLAY susceptibility snapshot, advisory #3):** The three "how has the practitioner's understanding shifted" claims below are agent synthesis of the session's conversation. They are not anchored by practitioner verbatim statements of those specific shifts. Treat them as agent-introduced framings pending practitioner confirmation, not as practitioner-confirmed epistemic history. SYNTHESIZE / next-cycle phases reading these claims should examine whether the practitioner actually generated each shift, or whether the agent inferred shifts from observable engagement patterns and routed them as if practitioner-stated.

**Which stakeholder had the hardest time?**

Only one stakeholder was inhabited (Ensemble Author / Operator). The hardest moment was the first-encounter with the empty configuration surface (note 1): the practitioner opened `.llm-orc/config.yaml` and `.llm-orc/ensembles/` expecting, based on the cycle's deliverable framing, a default set of agentic-serving profiles and a tagged ensemble library. The actual shipped state had one agentic-serving profile (`summarizer`) and zero tagged ensembles for the new ADR-015 router. The Cycle 1 PLAY's fabrication-on-first-encounter did not recur (note 6) — the cheap-cloud-orchestrator pattern (MiniMax M2.5-free via Zen) grounds responses in actual file reads and produces a workable first session — but the structural shape of the operator-side gap from Cycle 1 carries through: the deployment-as-shipped does not deliver the operator-facing on-ramp the cycle's research framed as the goal.

**What did play reveal that the specs missed?**

Four findings the specs did not anticipate, in increasing order of architectural reach:

- **The auto-mode of BUILD did not ship working defaults** (note 1, practitioner verbatim: *"the agentic-serving config is to me part of the build"*). The mechanism architecture is in code; the operator-facing on-ramp got externalized to ADR-015 §Negative's "operator-driven library migration" framing without a corresponding scenario or work package authoring the defaults.
- **The visibility gap from Cycle 1 PLAY carries through unchanged** (note 7). Cycle 4 added new internal events (verdicts, routing decisions, audit consumption records, signal-channel aggregations); none of them reach an operator-side surface without code changes. The serve terminal emits only validation warnings and request-path messages.
- **The methodology-layer / dispatch-layer / execution-layer separation reframes the cycle's deliverable** (notes 14, 15, practitioner-generated). The cycle's research framed agentic-serving as a deployment shape; play surfaced it as a *substrate* for many methodology consumers (RDD, code-review, security review, customer support, etc.), each composing against capability-named ensembles. This reframing was not authored in the corpus; it emerged from the inhabitation.
- **The cycle's quality infrastructure has two coverage gaps** (notes 16, 19): no calibration on the orchestrator's own natural-language narration to the client, and no infrastructure activation at all when no library ensemble matches the task. Notes 8, 10, 11, 12, 13, 18 all describe errors that reached the client uncalibrated. With the current 2-tagged-ensembles library state, most tasks fall through to the no-dispatch fallback path.

**How has the practitioner's understanding of the system shifted?**

Three shifts, audited honestly:

1. **From "the cycle shipped a complete agentic-serving stack" to "the cycle shipped a scaffold; the operator-facing library is deployment-specific and was outside BUILD's auto-mode scope."** The mechanism architecture (WP-H4-complete) is in code and operationally fires on first contact, but the path from "WP-H4 closed" to "operator can run a real session" requires a separate authoring pass that the cycle's deliverable framing did not name. This is a scope-distinction insight: *building the architecture* and *populating the deployment* are different work packages.

2. **From "ADR-015 §Negative's operator-driven library migration is fine" to "operator-driven library migration is downstream of decisions about which methodology-consumers the orchestrator will serve."** Different methodologies need different capability sets within the same Topaz slot vocabulary. The cycle's research named the slot-level discovery question (which 8-skill dimensions matter); play surfaced the operation-level question (which specific capability ensembles within each slot are load-bearing) as deployment-specific and downstream.

3. **From "Calibration Gate covers quality" to "Calibration Gate covers dispatched-ensemble outputs; the orchestrator's own narration is uncalibrated, and the no-dispatch path bypasses the infrastructure entirely."** The cycle's quality framing addresses two failure modes (dispatched-ensemble low-confidence output; phantom tool calls) but not a third (orchestrator natural-language claims that are coherent but factually wrong). This is a coverage gap of the existing infrastructure, identified post-build, not a bug.

The Cycle 1 PLAY's verdict — *"I would not be likely to use it again"* under default-local-orchestrator config — was profile-conditional. The Cycle 4 PLAY's analogous verdict, made explicit by the practitioner mid-session, is: *the auto-mode of BUILD is a sensible cycle-economy choice for the architecture work, but it does not produce the operator-facing deliverable that "BUILD-scope structurally complete" implied to a future reader of the cycle status.* The reframing is not a criticism of the cycle's BUILD-mode decision; it is a learning about what BUILD-mode (auto or gated) does and does not produce.

---

## Field-note routing summary

Each note is routed to its primary feedback destination. A note may inform more than one destination; the table records the primary route. Multi-route notes are flagged.

| # | Note (one-line) | Primary route | Also informs |
|--:|------|---------------|--------------|
| 1 | Configuration surface required a separate authoring pass | **DISCOVER** (assumption inversion — operator-driven migration shipped working defaults; it didn't) | interaction-specs |
| 2 | Cycle 4 stack activated cleanly on first session | **SYNTHESIS** (delight — machinery fires as designed on first contact) | — |
| 3 | Orchestrator routed meta-introspection to code-generation | **RESEARCH** (open question — routing-quality on non-clear-fit tasks; Sub-Q6 territory) | DISCOVER |
| 4 | MissingSkillMetadataError recovery loop worked | **SYNTHESIS** (delight — typed-error recovery path empirically validated) | — |
| 5 | Client-declared file tools carried most introspection | **SYNTHESIS** (Option C delegation discipline holds) | DISCOVER (introspection-via-ensembles framing overstated) |
| 6 | Cycle 1's fabrication did not recur (profile-conditional) | **SYNTHESIS** (delight — cheap-cloud-orchestrator pattern works) | DISCOVER (Cycle 1 verdict's scope clarified) |
| 7 | Visibility gap persists from Cycle 1 to Cycle 4 | **interaction-specs** (Operator's "Observe orchestrator behavior" task — concrete shape now nameable: colored logs / TUI dash) | DISCOVER |
| 8 | Orchestrator's meta-analysis quoted config comments verbatim | **DISCOVER** (assumption inversion — orchestrator self-description is comprehension-of-docs, not independent reasoning) | RESEARCH |
| 9 | Cycle 1 PLAY's 50K-vs-10M token-cap mystery undisturbed | **RESEARCH** (still-open question about source of 50K cap) | — |
| 10 | Recommendation mode differs from introspection mode | **RESEARCH** (new question — mode-specific orchestrator performance) | DISCOVER |
| 11 | Count claim self-corrected; denominator persisted | **DISCOVER** (assumption inversion — orchestrator verification is non-uniform; unchallenged-claim-stickiness is a real failure mode) | — |
| 12 | Routing-logic framing tightened across turns | **RESEARCH** (new question — does sequential engagement reliably tighten framing, or artifact?) | — |
| 13 | `compose_ensemble` primitives misunderstood | **DECIDE** (missing scenario — orchestrator's understanding of composition surface should be validated empirically before relying on it) | DISCOVER |
| 14 | Architectural Isolation maps cleanly from RDD to `invoke_ensemble` | **system-design** (architectural-property finding; RDD-via-agentic-serving structurally feasible without ADR amendment) | DISCOVER (the mapping extends the product model from "deployment shape" to "methodology substrate" — an assumption inversion of the cycle's prior DISCOVER framing; surfaced by agent, not by practitioner inhabitation, per snapshot advisory #1), SYNTHESIS |
| 15 | Three-layer separation reframes operator-driven migration | **DECIDE** (revisits ADR-015 §Negative; methodology/dispatch/execution layer separation as architectural principle) | system-design, DISCOVER |
| 16 | Calibration Gate doesn't audit orchestrator's narration | **DECIDE** (missing scenario / new ADR territory — calibration coverage gap for orchestrator-own-output) | system-design |
| 17 | Orchestrator produces substantive gap analysis on evaluative tasks | **SYNTHESIS** (orchestrator's useful mode) | DECIDE (orchestrator-identified gaps are direct input) |
| 18 | Recommendation justifications repeat the error pattern | **DISCOVER** (assumption inversion — orchestrator analytical output is audit-worthy not authoritative; consistent across task types) | — |
| 19 | No-dispatch fallback path; infrastructure does not fire at all | **DECIDE** (missing scenario / new ADR — what should govern direct orchestrator response when no ensemble matches) | interaction-specs |

Aggregate routing:

| Destination | Notes routed (primary) | Notes informed (also) |
|-------------|------------------------|----------------------|
| **DISCOVER** | 1, 8, 11, 18 | 3, 5, 6, 7, 10, 13, 14, 15 |
| **DECIDE** | 13, 15, 16, 19 | 17 |
| **RESEARCH** | 3, 9, 10, 12 | 8 |
| **SYNTHESIS** | 2, 4, 5, 6, 17 | 6, 14 |
| **interaction-specs** | 7 | 1, 19 |
| **system-design** | 14 | 15, 16 |

Cycle 4 PLAY produced the largest single batch of system-design and DECIDE-routed notes the cycle has seen — notes 14, 15, 16, 19 together describe load-bearing architectural and infrastructure-coverage findings that the cycle's prior phases did not surface. These are the highest-information outputs of this PLAY and should be the primary inputs to any follow-up cycle's DISCOVER → DECIDE pickup. The proposal at `proposals/agentic-serving-library-structure.md` captures the structural recommendations downstream of notes 14, 15, and (partially) 19.

---

# Play session: 2026-05-13 (Cycle 5 PLAY)

**Practitioner:** Nathan Green (cycle owner; two-phase method — gamemaster pre-PLAY reconnaissance via `curl`, then Skill Orchestration User inhabitation via OpenCode).
**System state at play:** Cycle 5 BUILD-close (2026-05-12); 7 `agentic-*` Model Profiles in `.llm-orc/profiles/`; 8 ensembles in `.llm-orc/ensembles/agentic-serving/` (6 capability + 2 system); `web_searcher.py` Tavily adapter at `.llm-orc/scripts/agentic_serving/`. Branch `agentic-serving`. `llm-orc serve` on port 8765 with `agentic-orchestrator` profile (minimax-m2.5-free via Zen, free tier).
**Configuration at play:** Same `.llm-orc/config.yaml` `agentic_serving:` section authored at Cycle 5 BUILD's WP-G5 (per-skill tier defaults; tier-router-audit defaults; conversation-compaction defaults; free-tier preference). Cheap-tier general: qwen3:8b; cheap-tier summary: qwen3:1.7b; escalated general: qwen3:14b; escalated reasoning: deepseek-r1:8b; summarizer: qwen3:0.6b on `agentic-summarizer` profile. `TAVILY_API_KEY` unset (web-searcher exercisable on error path only).
**Client:** OpenCode (sst/opencode TUI), pointed at `http://localhost:8765/v1`, for stakeholder inhabitation; `curl` for gamemaster reconnaissance.
**Stakeholder inhabited:** Skill Orchestration User (RDD instance) — issuing five queries through OpenCode to exercise the dispatch path. Ensemble Author / Operator surface observed indirectly through `execution.json` inspection during reconnaissance.

**Method:** Two-phase approach distinct from Cycle 4 PLAY's full-inhabitation pattern.

1. **Gamemaster pre-PLAY reconnaissance** — `curl` directly to `/v1/chat/completions` (no client `tools[]`), exercising 9 probes across NL/explicit/composition/boundary cells of the test space, to characterize the L1+L2 dispatch surface before stakeholder time was spent. Per-probe artifact-directory diffs and `execution.json` inspection where dispatched.
2. **Skill Orchestration User inhabitation** — five curated queries through OpenCode (client tools available), chosen to maximize remaining-information return given what reconnaissance established. Notes from this phase carry forward what reconnaissance could not measure.

**Categorization deferred** until session close. Notes below are raw observations in the order they surfaced.

---

## Stakeholder: Gamemaster (pre-PLAY reconnaissance — structural)

**Point of Concentration:** Exhaust the test surface that can be measured programmatically (single-shot, no client tools) so the inhabited stakeholder's time attends only to what cannot be measured this way.

### 1. Pre-flight verification was clean; runtime behavior was not

**Observation:** All 6 capability ensembles return `validate_ensemble: valid` and `check_ensemble_runnable: true`. All required Ollama models are available locally; Zen is authenticated with `minimax-m2.5-free` available. The L1 pre-flight surface gave zero signal that the capability ensembles would fail at dispatch. This is a layer-of-verification gap — discovery and schema-validation pass; runtime correctness was never exercised at BUILD close.

### 2. Natural-language framing never triggered `invoke_ensemble`

**Observation:** Five `curl` probes (code-generator task, summarization task, claim-extraction task, math task, meta-introspection task) — none dispatched. Artifact directory unchanged after each. Outputs were coherent and on-topic but produced entirely by the orchestrator's natural-language reasoning without exercising the cycle's quality infrastructure (no Calibration Gate verdict, no Tier-Escalation Router fire, no audit record, no signal channel state change). This is Cycle 4 PLAY note 19 generalized: under tool-less client requests, NL prompts route through the no-dispatch fallback path for all five Topaz-mappable slots tested. ADR-021's natural-language-supported clause is operationally not supported in this orchestrator+client configuration.

### 3. Explicit `invoke_ensemble` naming reliably triggered dispatch

**Observation:** Four explicit-naming probes (code-generator, claim-extractor, web-searcher, three-stage composition) all dispatched. The `agentic-result-summarizer` fired on every successful dispatch's return path per ADR-004/AS-7. ADR-021's per-capability dispatch contract works under explicit naming; the asymmetric availability of dispatch (silent on NL, reliable on explicit) is operator-relevant and not documented in scenarios.md or interaction-specs.md.

### 4. `claim-extractor` is runtime-broken at the agent execution layer

**Observation:** Two consecutive dispatches (orchestrator retried on first error) both produced: `{"status": "completed_with_errors", "results": {"extractor": {"response": null, "status": "failed", "error": "unsupported operand type(s) for +: 'NoneType' and 'str'"}}}`. `agents_count: 0`, zero tokens consumed — the agent never started. The YAML shape is single-agent with `default_task:` at ensemble level and **no `system_prompt:` on the agent**. The executor appears to perform a `+` operation against `None` system_prompt during agent setup. The same YAML shape is shared by `argument-mapper`, `prose-improver`, and `text-summarizer`; the defect is structurally implied for all four.

### 5. `code-generator` is partially functional with cascading downstream defects

**Observation:** Dispatch fired and returned `completed_with_errors`. The `coder` agent timed out at 180s (Ollama `agentic-tier-cheap-general` profile, qwen3:8b). The `critic` agent emitted `"The code is correct and handles all edge cases."` — a hallucinated review of nothing, since no coder output existed. The `synthesizer` agent produced actual working code (despite no coder input to integrate). Total dispatch duration 246.87s.

### 6. `agentic-result-summarizer` strips load-bearing content and inverts error status

**Observation:** The result-summarizer (qwen3:0.6b on `agentic-summarizer` profile) consumed the full execution dict from finding 5 — including `"status": "error"`, the coder failure, and the synthesizer's working code — and emitted to the orchestrator: *"The implementation of `chunk_by_predicate` is correct and handles all edge cases as specified. The execution status indicates success, confirming the function's correctness."* The actual code was stripped; the error status was inverted to success. The orchestrator never received the synthesizer's code or the error status — it received a meta-claim that "the code is correct" with no code attached. This is Cycle 4 PLAY note 16 ("Calibration Gate doesn't audit orchestrator's own narration") in concrete operational form, but one layer deeper: the AS-7 summarizer between ensemble result and orchestrator context is itself the load-bearing failure point.

### 7. `web-searcher` script-agent error-path handling is clean

**Observation:** Dispatched cleanly; Tavily adapter emitted structured `authentication_failed` (no `WEB_SEARCH_API_KEY` in environment); orchestrator narrated with actionable instructions. This is the cycle's BUILD-time smoke-test claim verified at the live-dispatch layer — the ADR-020 script-agent shape with operator-configurable backend works end-to-end for error paths.

### 8. Composition pipeline respects dependencies

**Observation:** Three-stage explicit pipeline (web-searcher → claim-extractor → argument-mapper): first stage failed authentication; orchestrator halted downstream dispatches without invoking the broken `claim-extractor` or `argument-mapper`; offered three recovery paths (set API key, provide URLs manually, list alternative ensembles). The orchestrator's multi-stage dependency reasoning is sound even when the underlying ensembles are broken.

### 9. Unauthored slot (mathematical_reasoning) does not trigger MissingSkillMetadataError recovery

**Observation:** Math task under NL framing → no dispatch attempted → no typed-error recovery path fired. Cycle 4 PLAY note 4's recovery path is conditional on the orchestrator first attempting `invoke_ensemble` against a slot that returns the typed error. Under NL framing the orchestrator never attempts; under the no-dispatch fallback (finding 2), most prompts that *would* exercise the recovery path never reach it.

---

## Stakeholder: Skill Orchestration User (RDD instance, via OpenCode)

**Super-Objective:** Use the agentic-serving library to dispatch capability ensembles from inside an RDD workflow, treating the orchestrator as the methodology consumer's dispatch surface for sub-task work.

**Point of Concentration:** Exercise the explicit-naming dispatch path (reconnaissance baseline) through OpenCode's client-tool-rich surface and observe what differs from the tool-less `curl` baseline. Queries 1–5 selected from gamemaster suggestions, ordered most-information-first given runtime defects established in reconnaissance.

### 10. OpenCode dispatch took 22m 36s for a task `curl` ran in 4m 44s (4.7× multiplier)

**Observation:** Query 1 (same code-generator prompt as reconnaissance probe 3) ran 22m 36s in OpenCode vs. 4m 44s in `curl`. The orchestrator visibly looped: dispatched once → "summary indicates success but didn't surface the code" → "Let me re-invoke" → "result summarizer encountered an issue" → tried to `Read` the YAML at wrong path → `Glob`-ed to find the actual location → `Read` correctly → presented code. The 4.7× latency is self-correction overhead made possible by client tools; the same prompt under tool-less `curl` ran once and stopped at the summarizer's stripped output. The orchestrator's recovery behavior is sound in shape but expensive in time; the bilateral-visibility absence makes the wait opaque.

### 11. Practitioner self-report of wait phenomenology

**Observation:** Practitioner verbatim mid-session: *"Seemed like that took a long time. Why?"* The orchestrator answered with structural reasoning about the three-agent flow's latency mathematics (3× single-call latency) — a coherent answer but oblique to the actual cause (self-correction loops, summarizer compression triggering re-dispatch). The orchestrator narrated the *general* latency property of the ensemble shape rather than the *specific* latency of this dispatch.

### 12. Orchestrator confirmed it has no visibility into execution timing or graph

**Observation:** Practitioner asked: *"What was the total run-time of the ensemble?"* Orchestrator verbatim: *"I don't have timing data from that invocation — invoke_ensemble only returned the text output, not runtime metrics. The orchestration layer would need to expose execution stats for me to answer that."* This is Cycle 4 PLAY note 7 (visibility gap) in a sharper form: not just operator-terminal silence, but **the orchestrator narrating dispatch outcomes with structural ignorance of the execution graph it just commissioned**. The `execution.json` artifacts (which contain full timing, per-agent duration, tier routing, verdict state) are on disk; the orchestrator's context does not include them.

### 13. Orchestrator articulates dispatch-surface defects as if they were open tickets

**Observation:** Query 2 close-out: *"The synthesis step appears to have presented the critique itself rather than the final corrected implementation. You may want to re-dispatch with a directive to output only the final, corrected code."* Query 3 close-out: *"The synthesizer is still wrapping results in prose. Tuning the system prompt directly is probably the cleaner fix than relying on per-dispatch directives."* Query 5 close-out: *"The ensemble's output habit is still too narrative — worth filing as a dispatch config issue."* The orchestrator can name the defect category and suggest concrete remediations (prepend directive, tune system prompt, compose single-shot variant) but cannot fix the dispatch surface from within its session. This is recovery-mode reasoning — defensible but indicating a coverage gap: the practitioner is now informed about the defect but has no in-session resolution path.

### 14. Orchestrator's own remediation recommendation didn't work as predicted in the next query

**Observation:** Query 2 recommended "prepend a directive: 'Output only the final corrected code'" as the lightest fix. Query 3 applied a directive ("Return both outputs in sequence") and observed: *"The directive didn't fully suppress the narrative — both outputs still included summaries rather than clean code blocks."* The orchestrator's own A→B prediction (prepend directive will suppress prose) was not borne out by its own subsequent test. The orchestrator acknowledged the failure (*"Tuning the system prompt directly is probably the cleaner fix"*) but the prediction-vs-outcome gap was unflagged at the architectural level.

### 15. Orchestrator presents fabricated code as ensemble output while critiquing the ensemble

**Observation:** Query 5's `CodeGenerator` class is stub-only — `coder()` literally returns `f"# TODO: implement for task: {task}"`; `__init__` references `model: str = "codellama:latest"` which is not in the Ollama profile list. This is fabrication: the orchestrator generated the class from prompt intent rather than receiving it from the dispatched ensemble. Immediately after, the orchestrator wrote: *"The ensemble's output habit is still too narrative — worth filing as a dispatch config issue."* The combined pattern is fabrication while critiquing fabrication — the orchestrator's recovery posture (when ensemble output disappoints) compounds the calibration gap from note 16: the orchestrator's natural-language narration *is* the failure surface that has no calibrator.

### 16. Q4 meta-introspection recommends compositions including the broken ensembles

**Observation:** Query 4 (meta-introspection, no dispatch — 8.1s, no artifact) produced a recommended RDD workflow composition: `web-searcher → claim-extractor → argument-mapper → code-generator → prose-improver → text-summarizer → human-in-loop-validation`. Five of these are runtime-broken or have downstream defects per findings 4–6. The orchestrator sees the ensembles in `list_ensembles` output (which it appears to query internally, narrating: *"Let me start by querying my own capability surface"*) and presents them as available capabilities. The discovery layer it has access to does not surface executability state — a user accepting this recommendation as a starting point would compose a workflow primarily out of non-functioning capabilities.

### 17. Multi-turn context preserved file_read state across queries

**Observation:** Query 5: *"I read this file earlier. Here's the three-agent flow:"* — the orchestrator retained the YAML content from Query 1's recovery `Read` and reused it without re-reading. Multi-turn memory works structurally. Whether this improves output quality vs. single-shot is not yet tested (Query 5's stub-fabrication suggests memory of the YAML did not constrain hallucination of the LLM-backed implementation).

### 18. Test-suite output contained internal inconsistency not flagged by orchestrator or ensemble

**Observation:** Query 3's `test_concurrent_access` asserts `counter[0] == 10` after running 10 threads incrementing a shared counter through a `@memoize_ttl(10)`-decorated no-argument function. Memoization on a no-arg function makes all calls share one memo key — the decorator returns the cached value after the first call, so `counter[0]` would be 1, not 10. The test as written is internally inconsistent with the decorator it tests. Neither the synthesizer agent, the result-summarizer, nor the orchestrator flagged this. The output reached the client uncalibrated for semantic correctness — exactly the failure-mode shape Cycle 4 PLAY note 16 named, in concrete code-correctness form.

### 19. Routing observability remains the cycle's persistent gap

**Observation:** Practitioner verbatim at session reflection: *"the observability of the routing still seems lacking to me, but in the serving console process and in the output from the orchestrator — I'd like to have more visibility into what the routing is doing, even if output is the most important metric. This was flagged last time."* The serve terminal during Cycle 5 PLAY emitted the same minimal output (request paths, Pydantic warnings on two legacy ensembles unrelated to Cycle 5) as Cycle 4 PLAY (note 7). The orchestrator's natural-language narration confirms it has no execution-graph visibility (finding 12). Cycle 5's new internal events (verdicts, tier-routing decisions, audit consumption, signal-channel aggregation) exist in code and write to `execution.json` artifacts; none reach the operator's terminal or the orchestrator's reasoning context without code changes. The bilateral-visibility framing from Cycle 1 carries through Cycles 4 and 5 unchanged.

---

## Cross-cutting reflection

**Which stakeholder had the hardest time?**

The Skill Orchestration User (RDD instance) inhabited via OpenCode. Five queries; 4 of 5 dispatch-firing queries had narrative-wrapping issues or coherent-but-incorrect downstream behavior; the one query that didn't dispatch (Query 4) recommended a workflow primarily composed of broken ensembles. Total wall-clock for the 5 queries was ~37 minutes. The practitioner remained in investigative posture throughout — but the experience would not be navigable for a stakeholder *not* in investigation mode: there is no in-session signal that distinguishes "the orchestrator is working well" from "the orchestrator is fabricating coherently on top of a broken dispatch."

The gamemaster reconnaissance phase had the *easiest* time precisely because it observed the failure layers directly. The execution.json artifact path is the calibration surface the cycle's quality infrastructure produces; the operator and stakeholder seats cannot access it without leaving OpenCode and inspecting files. That is the visibility asymmetry the cycle has not yet closed.

**What did play reveal that the specs missed?**

Five findings the specs did not anticipate, in increasing order of architectural reach:

- **The validation-vs-execution gap.** `validate_ensemble` + `check_ensemble_runnable` were treated at BUILD close as sufficient pre-shipping verification. Four of six capability ensembles ship with a YAML shape (single-agent + ensemble-level `default_task` + no agent-level `system_prompt`) the executor rejects at runtime. No scenario in `scenarios.md` requires dispatch-exercise verification before BUILD declares ensembles working. The validation surface BUILD relied on was a discovery-layer check, not a runtime check.

- **The result-summarizer's load-bearing failure mode.** `agentic-result-summarizer` (qwen3:0.6b on `agentic-summarizer` profile) is positioned by ADR-004/AS-7 as faithful summarization for orchestrator context management — a correctness requirement, not an optional optimization. In practice, this tiny model strips load-bearing content (the synthesizer's actual code) and inverts error status to success. The orchestrator receives a meta-claim that "the code is correct" with no code attached, then fabricates code from the prompt and presents it as the ensemble's output. The summarization step is the surface where the cycle's quality infrastructure most concretely fails.

- **The natural-language vs. explicit-naming dispatch asymmetry.** ADR-021's per-capability dispatch contract commits to both — explicit naming preferred, natural-language supported. The operational reality under `minimax-m2.5-free`: natural-language never dispatches; explicit naming reliably does. This is a documented contract that is honored asymmetrically. No scenario specifies what triggers should activate dispatch under NL.

- **The orchestrator's structural blindness to its own execution graph.** The orchestrator narrates dispatches with no knowledge of the dispatched ensemble's per-agent duration, verdicts, tier-routing decisions, or status. It can answer general questions about ensemble shape but cannot answer specific questions about its just-completed dispatches. The `execution.json` artifacts are not in its reasoning context. This is structurally the dual of the operator-terminal-silence gap from Cycle 1 PLAY: two seats, both blind, neither served by the cycle's new internal events.

- **The routing observability gap is not new in Cycle 5; it persists across three cycles unchanged.** Cycle 1 PLAY notes 7, 9, 10 named bilateral visibility absence. Cycle 4 PLAY note 7 reframed it concretely (colored logs, TUI dash, specific list of internal events that don't surface). Cycle 5 PLAY observes the same absence with the additional architectural pin from finding 12 (the orchestrator confirms its own blindness). The cycle's quality infrastructure produces telemetry the cycle does not yet route to a human-visible surface.

**How has the practitioner's understanding of the system shifted?**

Three shifts:

1. **From "BUILD-time validation suffices" to "validation and runtime are distinct verification layers."** Cycle 5 BUILD's verification operated entirely at discovery + schema validation. The runtime correctness of the shipped ensembles was assumed, not exercised. The susceptibility snapshot's auto-mode finding category — "silent resolution of artifact-level conflicts" — extends to a structural finding: under auto mode, BUILD did not exercise its own deliverables before declaring close. A scenario requiring real-dispatch verification of each ensemble would have caught this.

2. **From "the orchestrator is capable" to "the orchestrator is capable on derivable claims and unreliable on integration claims, with no calibration surface for the difference."** Cycle 4 PLAY note 18 named the pattern; Cycle 5 PLAY observes it in operational form. The orchestrator's recommendation mode produces well-structured suggestions (the RDD composition in Q4 is directionally sound); its execution-narration mode fabricates with the same coherence (Q5's stub class presented as ensemble output). The same coherent voice serves both modes; no in-session signal distinguishes them.

3. **From "Cycle 5 BUILD shipped working defaults" to "Cycle 5 BUILD shipped a partially-functioning library and a complete operator-facing on-ramp."** The on-ramp (config section, README, profile file, subdirectory layout, environment variable wiring) is correct. The library it ramps onto has 4 of 6 capability ensembles broken at runtime. The Cycle 4 PLAY note 1 framing — *"the agentic-serving config is to me part of the build"* — is closed at the operator-facing-config layer and reopened at the runtime-correctness layer. This is not a regression to Cycle 4's gap; it is a new gap one layer in.

---

## Field-note routing summary

Each note is routed to its primary feedback destination. A note may inform more than one destination; the table records the primary route. Multi-route notes are flagged.

| # | Note (one-line) | Primary route | Also informs |
|--:|------|---------------|--------------|
| 1 | Pre-flight verification clean; runtime not | **BUILD-regression** (Cycle 6 candidate — runtime-exercise verification is a missing pre-shipping step) | DECIDE (scenario) |
| 2 | NL framing never dispatched (5 probes) | **DECIDE** (missing scenario — what triggers NL dispatch under MiniMax M2.5-free? or is the contract honored asymmetrically by design?) | DISCOVER |
| 3 | Explicit naming dispatches reliably | **SYNTHESIS** (settled-by-use — ADR-021's explicit-naming contract works) | system-design |
| 4 | claim-extractor runtime-broken (+ 3 same-shape) | **BUILD-regression** (4 of 6 capability ensembles ship non-functional; mechanical hotfix territory) | interaction-specs |
| 5 | code-generator partial — coder timeout + critic hallucinates + synthesizer works | **BUILD-regression** (qwen3:8b timeout tunability; critic-on-empty-output behavior) | DECIDE |
| 6 | result-summarizer strips content + inverts status | **DECIDE** (load-bearing — calibration coverage gap for AS-7 summarizer; new ADR territory) | BUILD-regression |
| 7 | web-searcher error-path clean | **SYNTHESIS** (delight — ADR-020's script-agent shape verified at live-dispatch layer) | — |
| 8 | Composition respects dependencies | **SYNTHESIS** (settled — orchestrator's multi-stage dependency reasoning sound) | system-design |
| 9 | Unauthored slot doesn't trigger recovery | **DISCOVER** (Cycle 4 note 4's recovery path is conditional on dispatch attempt) | RESEARCH |
| 10 | 4.7× latency multiplier in OpenCode (self-correction loops) | **DISCOVER** (value tension — recovery posture trades latency for output coherence) | interaction-specs |
| 11 | Wait phenomenology — practitioner self-reports | **DISCOVER** (assumption inversion — long-dispatch UX assumed tolerable; tolerable only under investigative posture) | interaction-specs |
| 12 | Orchestrator has no execution-graph visibility | **interaction-specs** (architectural — the orchestrator's reasoning context lacks dispatch telemetry it just commissioned) | DECIDE |
| 13 | Orchestrator names dispatch defects without resolution | **DISCOVER** (assumption inversion — orchestrator-as-debug-surface assumed; orchestrator is a defect-narrator without a fix path) | DECIDE |
| 14 | Orchestrator's own remediation prediction failed | **RESEARCH** (new question — does the orchestrator's self-modeling of dispatch reliably predict dispatch behavior?) | — |
| 15 | Fabrication while critiquing fabrication | **DECIDE** (missing scenario / new ADR territory — calibration coverage gap for orchestrator natural-language narration of dispatched outputs) | DISCOVER |
| 16 | Q4 recommended composition includes broken ensembles | **DISCOVER** (assumption inversion — `list_ensembles` is the discovery surface; executability is not part of the discovery contract) | DECIDE |
| 17 | Multi-turn file_read state preserved | **SYNTHESIS** (delight — Option C delegation discipline composes with multi-turn memory) | — |
| 18 | Test-suite internal inconsistency unflagged | **DECIDE** (missing scenario — orchestrator's post-processing of code outputs does not validate semantic consistency against task intent) | RESEARCH |
| 19 | Routing observability remains the cycle's persistent gap | **interaction-specs** (the load-bearing carry-forward from Cycles 1 → 4 → 5; bilateral visibility is the cross-cycle deferred work) | DISCOVER, system-design |

Aggregate routing:

| Destination | Notes routed (primary) | Notes informed (also) |
|-------------|------------------------|----------------------|
| **BUILD-regression** | 1, 4, 5 | 6 |
| **DECIDE** | 2, 6, 15, 18 | 1, 5, 12, 13, 16 |
| **DISCOVER** | 9, 10, 11, 13, 16 | 2, 15, 19 |
| **RESEARCH** | 14 | 9, 18 |
| **SYNTHESIS** | 3, 7, 8, 17 | — |
| **interaction-specs** | 12, 19 | 4, 10, 11 |
| **system-design** | — | 3, 8, 19 |

Cycle 5 PLAY produces the largest BUILD-regression batch in the agentic-serving corpus's history (notes 1, 4, 5 + informed 6). Cycles 1 and 4 PLAY produced DISCOVER- and DECIDE-routed framings; Cycle 5 PLAY produces operational evidence that the BUILD-time verification surface itself is structurally inadequate. The cycle's load-bearing follow-up work is *not* in further DISCOVER-DECIDE iteration but in BUILD-regression of the shipped artifacts before any subsequent cycle composes on top of them.

Note 19's routing observability finding is the carry-forward that *should* prompt research-or-decide work on a Cycle 6+ observability ADR. The cross-cycle persistence (Cycle 1 → Cycle 4 → Cycle 5 unchanged) is itself a meta-signal: the gap has survived multiple decision/build cycles without being addressed. Either the gap is genuinely deferred (operator-driven, per ADR-015 §Negative's framing) and the deferral should be made visible as such; or it is a latent ADR that has been crowded out by other Cycle priorities and warrants explicit work.

**Sharpened framing (per Cycle 5 PLAY susceptibility snapshot, 2026-05-13):** Note 19's "Cycle 1 → Cycle 4 → Cycle 5 unchanged" is accurate about *operator-terminal experience* but potentially misleading about *architectural progress*. Cycle 5 BUILD shipped new internal events (verdicts, tier-routing decisions, audit consumption, signal-channel aggregation) that did not exist in Cycle 1. The gap is more precisely characterized as **infrastructure-complete / routing-incomplete** — the telemetry exists; the routing of telemetry to human-visible surfaces is the work that has been deferred. This is a sharper DECIDE target than "design observability from scratch."

---

## Post-reflection coda

### 20. NL framing under client-tools fell through to client-tool delegation, not ensemble dispatch

**Observation:** A single closing probe ran after the cross-cutting reflection, addressing the susceptibility snapshot's named gap (reconnaissance never tested NL framing under OpenCode's client-tool-rich surface). The same NL prompt as reconnaissance probe 1 was sent through OpenCode. Elapsed 5.2s. No `invoke_ensemble` call; no artifact created in `.llm-orc/artifacts/`. The orchestrator dispatched via the client `Write` tool, creating `chunk_by_predicate.py` on disk and emitting the code as narration. The "NL never dispatches" finding from reconnaissance generalizes from tool-less `curl` to tool-rich OpenCode — but the fall-through *target* differs: text completion (curl, finding 2) vs. client-tool delegation (OpenCode, finding 20). The capability ensemble library is reached only via explicit `invoke_ensemble` request. Under NL framing, the orchestrator's observable routing preference is: (1) direct LLM completion when no tools are needed → (2) client-tool delegation when client tools provide a relevant action → (3) `invoke_ensemble` dispatch only on explicit request.

**Category:** Challenged assumption (the operative routing preference is direct → client-tools → ensemble, not ensemble-first-when-slot-fits as ADR-021's natural-language-supported clause implies).

**Feeds back to:** **DECIDE** (the operational routing preference contradicts ADR-021's natural-language-supported clause under both tested client configurations; the cycle's commitment to "natural-language supported" needs either re-grounding via system-prompt work or explicit narrowing of the supported-routing surface) + DISCOVER (the Skill Orchestration User's mental model of "the orchestrator will route my NL request to a capability ensemble" is not borne out; the operative model is "the orchestrator will route my NL request to client tools or completion; ensembles are explicit-only").

**DECIDE Advisory 2 carry-forward status:** Now fully closed. The full-client-tool NL dispatch path was tested in this probe; the advisory's "first empirical test under client-tool-rich client" criterion is met. The advisory's deferred resolution from BUILD is now empirically anchored.

**Aggregate routing (updated):**

| Destination | Notes routed (primary) | Notes informed (also) |
|-------------|------------------------|----------------------|
| **BUILD-regression** | 1, 4, 5 | 6 |
| **DECIDE** | 2, 6, 15, 18, 20 | 1, 5, 12, 13, 16, 20 |
| **DISCOVER** | 9, 10, 11, 13, 16 | 2, 15, 19, 20 |
| **RESEARCH** | 14 | 9, 18 |
| **SYNTHESIS** | 3, 7, 8, 17 | — |
| **interaction-specs** | 12, 19 | 4, 10, 11 |
| **system-design** | — | 3, 8, 19 |

---

# Cycle 6 PLAY (2026-05-20)

**Play session:** 2026-05-20
**Practitioner:** Nathan Green (self-play, Skill Orchestration User role; agent as gamemaster)
**Configuration tested:** agentic-orchestrator (MiniMax M2.5-free via Zen) + OpenCode tool-rich for probe 1; tool-less `curl` against running serve for probes 2-7
**Cycle context:** BUILD complete 2026-05-16 (ADR-022/023/024/025 landed); PLAY entered with the cycle-status PLAY-entry agenda

Raw observations during play. Categorization and feedback routing deferred to post-session work.

## Stakeholder: Skill Orchestration User

**Super-objective:** Compose a skill framework against the orchestrator's capability library; expect dispatch when a capability slot fits.

### 1. Spike γ Cell A re-run under ADR-022 amendment (OpenCode tool-rich)

Probe prompt held constant from Cycle 6 spike γ Cell A: *"Write a Python function that reverses a string in place."* Configuration: `agentic-orchestrator-minimax-m25-free` via Zen + OpenCode tool-rich + NL framing (no explicit ensemble name).

OpenCode-side observation: inline text response containing two function definitions (`reverse_string` slicing version + `reverse_string_in_place` two-pointer version) with a one-line note distinguishing idiomatic from educational use. No file written. No intermediate "the assistant is calling X..." messaging visible. Latency badge: 10.3s.

Serve-console observation during the probe: one validation warning at startup for `neon-shadows-detective/ensemble.yaml` (then silence on validation across the probe). Two `POST /v1/chat/completions 200 OK` lines (ports 55412, 55414). **Zero** `tool dispatch:` lines. **Zero** per-event INFO lines (no dispatch start, tier selection, calibration verdict, audit diagnostic, or dispatch end). **Zero** `tool-call emit:` lines. **Zero** `inference wait:` heartbeats (latency below 30s threshold).

Spike γ Cell A baseline comparison: the baseline orchestrator emitted a `Write` tool call into OpenCode (client-tool delegation path); no `invoke_ensemble` call. This run differs: no `Write` tool call to OpenCode, also no `invoke_ensemble` call. The orchestrator's response stream contained zero tool calls of any kind — pure 10.3s inference followed by inline text.

The shift from baseline: client-tool delegation → direct LLM completion. The amendment shifted the bypass surface (client tools → direct completion) without producing dispatch to `invoke_ensemble`. ADR-022's three dispositions (i)/(ii)/(iii) live in the gap between this observation and the baseline; the observation does not uniquely determine which.

### 2. NL framing under tool-less `curl` — BLOCKED on Zen quota

Attempted reproduction of Cycle 5 PLAY reconnaissance note 2 (NL never dispatched under `curl`) against the post-WP-E serve to characterize whether the amendment shifts bare-endpoint behavior. All attempts returned HTTP 500 with `FreeUsageLimitError: Rate limit exceeded` from OpenCode Zen, the underlying provider for the `agentic-orchestrator` default profile. The Zen free-tier quota was apparently exhausted by the OpenCode session that ran probe 1. Five retries across roughly an hour all returned 429.

The bare-endpoint characterization could not be performed in this session. The result is structural rather than behavioral: a single OpenCode session under the cheap-cloud-orchestrator profile is sufficient to exhaust Zen's free-tier quota for the next hour-plus, blocking diagnostic curl probes in the same window.

### 3. Explicit ensemble naming under tool-less `curl` — BLOCKED on same Zen quota

Same blocker as note 2. Could not characterize whether explicit naming (Cycle 5 PLAY reconnaissance note 3 baseline: explicit always dispatched) holds under the post-WP-E system prompt.

## Stakeholder: Ensemble Author / Operator (observed in parallel)

### 4. Validate-once-at-load behavior across multiple list_ensembles calls

Five sequential `GET /api/ensembles` calls fired against the running serve. All returned 200 OK with 83 ensembles each. From this conversation's view, the validation behavior on the serve-console side is not directly observable. From the prior serve startup line in the conversation: only one validation warning emitted at startup (`neon-shadows-detective/ensemble.yaml`). Practitioner-side confirmation needed: did any additional validation warnings emit across the five list calls? If none, validate-once-at-load (WP-B piece 3) holds operationally.

## Stakeholder: Gamemaster (capability-ensemble health probes via direct invoke)

These probes hit `POST /api/ensembles/<name>/execute` directly, bypassing the orchestrator and substrate routing. The intent was to characterize individual capability-ensemble health independent of dispatch infrastructure.

### 5. claim-extractor produced output-spec-non-conformant essay

Direct invoke with input: *"The Earth orbits the Sun. Water boils at 100 degrees Celsius. Some people believe in ghosts. The speed of light is 299,792,458 m/s."*

Elapsed: 65 seconds. Status: `success` (ensemble engine + agent).

The agent returned a multi-paragraph analytical essay with section headers (`### 1. "The Earth orbits the Sun."`, etc.), per-statement breakdowns into Scientific Context / Orbital Mechanics / Gravitational Force / Significance bullets, a "**Key Differentiation**" section labeling statements as Scientific Facts vs. Subjective Belief, an "**Additional Notes**" section, and a closing "Let me know if you'd like further details on any of these topics!"

The ensemble's `default_task` specifies bulleted (established)/(contested)-labeled claims and explicitly forbids preamble, synthesis, conclusions. The output violates this in form (essay vs. bullets), in vocabulary (no `(established)`/`(contested)` labels appear), and in posture (the closing offer to discuss further is the kind of response-shaping the spec forbids).

The ensemble engine returned `status: success` for the agent and for the dispatch.

Comparison to Cycle 5 PLAY reconnaissance note 4 baseline ("claim-extractor is runtime-broken at the agent execution layer"): the agent now executes end-to-end without failing. The Cycle 5 framing of "broken at agent execution layer" may need refinement — the agent runs, but produces output that doesn't conform to its spec.

### 6. web-searcher clean and fast on direct invoke

Direct invoke with input: *"current population of Iceland"*

Elapsed: 2.658 seconds. Status: `success`. Response: well-formed JSON with five DDG search results (titles, URLs, snippets), backend marker `"ddgs"`, total ~1.8KB.

Consistent with Cycle 5 PLAY reconnaissance note 7 (clean error-path handling); extends to clean happy-path. Web-searcher is a script-agent ensemble — no LLM inference in this path; the latency is the network round-trip to DDG plus script overhead.

### 7. code-generator three-agent ensemble runs to completion on direct invoke

Direct invoke with input: *"Write a Python function that reverses a string in place."*

Elapsed: 60.697 seconds. Status: `success` for the dispatch and `success` for each of the three agents (`coder`, `critic`, `synthesizer`).

Output structure: `coder` produces a function (list-conversion approach, with acknowledgment that strings are immutable); `critic` reviews and confirms the immutability framing; `synthesizer` combines into a final response with code blocks, example usage, key considerations, and the critic's note.

Total response ~2KB. Single function definition produced (the list-convert-reverse approach); the slicing one-liner `s[::-1]` that the OpenCode/M2.5-free probe (note 1) led with does not appear.

Latency comparison: note 1's 10.3s direct LLM completion vs. this 60.7s ensemble run is a 6× factor. The ensemble adds the calibration-gate / critic / synthesizer pass; the direct completion produces a more focused output faster.

## Stakeholder: Ensemble Author / Operator (substrate routing infrastructure on disk)

### 8. Substrate routing produced an artifact at the spec'd path operationally

Inspection of `.llm-orc/agentic-sessions/` showed a session directory `ade03c0d43a42f896c85e33ea4bf7dbaa8b6874ef9ab1bfe411cae47ded4fe79/` with mtime 2026-05-20 15:30 containing:

- `dispatch_log.json` — captures two dispatch entries with the full WP-C 7-field schema (`dispatched`, `duration_seconds`, `model_profile`, `tier`, `topaz_skill`, `calibration_verdict`, `dispatch_id`). First entry: `code-generator`, 55.84s, `agentic-tier-cheap-general`, cheap tier, code_generation skill, verdict `proceed`. Second entry: `qa-pipeline`, 0.5ms, null model_profile/tier/topaz_skill, verdict `proceed`.
- `<session>-dispatch-0001/code-generator.py` (2309 bytes) — the substrate-routed artifact. Content is the raw multi-agent ensemble JSON payload (`coder` / `critic` / `synthesizer` responses).
- `<session>-dispatch-0001/code-generator.py.retention` (7 bytes, content: `session`) — the per-artifact retention marker file from WP-E piece 3.

Path layout matches ADR-025 spec exactly: `.llm-orc/agentic-sessions/<session_id>/<dispatch_id>/<deliverable>.<ext>`. Dispatch_id format `<session>-dispatch-NNNN` matches WP-A's DispatchEventSubstrate convention.

The session is not from a probe in this PLAY conversation — likely from a background process or test execution earlier in the day. Either way, the disk evidence shows the substrate-routing infrastructure works end-to-end: artifact materializes at the right path, retention marker accompanies it, dispatch_log captures clean metadata, file content reflects the raw ensemble payload.

`agentic-result-summarizer` does not appear in the dispatch_log for this session — consistent with AS-7 amended (substrate-routed dispatches skip the summarizer). Not definitive (no observation of a non-substrate-routed comparison case), but the pattern matches.

### 9. dispatch_log.json lands at WP-C path across all observed sessions

Three sessions inspected for `dispatch_log.json` presence:

- `343ad246...` (morning OpenCode probe, mtime 12:50): file present, `dispatch_log.entries = []`. Consistent with note 1's disposition-iii direct completion — no orchestrator dispatch fired.
- `1a7a86c8...` (afternoon, mtime 15:37): file present, `dispatch_log.entries = []`. Consistent with direct-invoke probes (notes 5/6/7) which bypass orchestrator dispatch.
- `ade03c0d...` (afternoon, mtime 15:30): file present, two entries with full schema (see note 8).

Every session that opens through the chat completions endpoint writes a `dispatch_log.json` at request close, regardless of whether dispatches occurred. The empty case is well-formed (`{"dispatch_log": {"entries": []}}`).

### 10. The OpenCode-morning session's empty dispatch_log corroborates note 1

The `343ad246` session corresponds to the OpenCode session that ran probe 1 this morning (the inline two-function response). Its `dispatch_log.entries = []` is independent evidence that no orchestrator-mediated dispatch fired — the orchestrator did direct LLM completion, exactly as the serve-console observation indicated. The two `POST /v1/chat/completions 200 OK` lines without `tool dispatch:` lines have a corresponding zero-entry dispatch log on disk.

## Provisional cross-cutting observations (raw, pending post-session reflection)

- The ADR-022 amendment changed M2.5-free's routing surface under OpenCode tool-rich. The Cycle 5 PLAY note 20 fall-through path (client-tool delegation) is no longer the preferred bypass; direct LLM completion is. Whether this is progress, a different defect, or both is open.
- Substrate routing infrastructure is operationally wired and produces on-disk artifacts with the spec'd layout. This was verified through observation rather than orchestrator-dispatched probe — but the observation is strong.
- claim-extractor's output-spec drift is dramatic and surfaces at the agent execution layer. The drift mechanism is not orchestrator `input.data` override (direct invoke bypasses the orchestrator entirely) — the synthesizer/agent is not following its `default_task` system prompt. Spike β's reframing of drift mechanism may need to extend to a second mechanism: synthesizer compliance failure independent of orchestrator-side override.
- The Zen free-tier rate limit is operationally significant: a single OpenCode session exhausts the quota window for diagnostic curl probes in the next hour-plus. This is a meta-observation about deployment economics under the cheap-cloud-orchestrator pattern.
- The orchestrator-dispatched probes that would resolve open questions (Tests 1, 2, 3 live, 4 live, 7) all need a working Zen quota window OR a profile-switch to the offline-tools orchestrator. Either approach is the practitioner's call.

## Paid-Zen probes (afternoon 2026-05-20, after profile swap to agentic-orchestrator-minimax-m25)

Practitioner created a new profile pointing at paid MiniMax M2.5 on Zen (`.llm-orc/profiles/agentic-orchestrator-minimax-m25.yaml`) and enabled the paid model on their Zen account; serve restarted to load the new profile. Probes 11-15 ran under tool-less curl against this configuration.

### 11. Earlier "on-disk substrate evidence" was misattributed

The ade03c0d session that appeared in field notes 8/9 with a substrate artifact and two-entry dispatch_log (code-generator + qa-pipeline) was not from a test fixture or background process. It was the active session for curl probes in this conversation, with its dispatch_log overwritten by each subsequent request. By the time of probe 12 (Test 1 paid Zen) the dispatch_log had been rewritten to a single code-generator entry with `duration_seconds: 51.799`, replacing the prior 55.84s entry.

The structural finding holds: substrate routing operationally writes artifacts to the spec'd path with `.retention` markers; dispatch_log captures the 7-field schema. The provenance correction: the evidence was generated by the curl probes, not by a separate process.

### 12. Test 1 (NL framing, tool-less curl, paid M2.5) — DISPATCH FIRED

Same probe prompt as note 1 ("Write a Python function that reverses a string in place"). 64.165 seconds elapsed. Orchestrator narration explicitly states: *"The request maps to the `code-generator` ensemble — a code-generation capability. I'll invoke it."* The dispatch fired (dispatch_log shows code-generator at 51.799s, agentic-tier-cheap-general, code_generation, verdict proceed). The substrate artifact was rewritten with fresh content from this dispatch.

ADR-022 amendment effectiveness under paid M2.5 + tool-less curl: disposition (i) — the amendment shifted routing toward `invoke_ensemble`.

Comparison to note 1 (free M2.5 + OpenCode tool-rich): same prompt, opposite outcome. Two variables changed between note 1 and note 12 (paid vs. free model; tool-rich vs. tool-less client). The clean A/B isolating each variable requires running paid M2.5 under OpenCode tool-rich AND free M2.5 under tool-less curl. The free-tier quota currently blocks the latter; the former needs the practitioner's OpenCode environment.

Token consumption: 17,637 completion tokens for a string-reverse function request. The paid model generates substantially more tokens than the free-tier inline-text 10.3s response from note 1.

### 13. Final response content contains malformed MiniMax-native tool-call XML

All three paid-M2.5 curl probes (notes 12, 14, 15) ended with the assistant message content containing a block of the form:
```
<invoke name="file_read">
<parameter name="path">agentic-sessions/<session>/<dispatch>/<deliverable></parameter>
</invoke>
</minimax:tool_call>
```

This is MiniMax's native tool-call format. The framework expects OpenAI function_call format in `message.tool_calls[]` — the XML appears in `message.content` as raw text. The framework cannot parse it as a tool call, so it does not fire a follow-up dispatch. The paths the XML targets are the substrate paths that were just written by the preceding dispatch (or, in note 14's case, by the chain step). The model behaves as though it expects to read back what it produced before finalizing its response — a sensible composition pattern that breaks against the framework's tool-call protocol.

Free-tier MiniMax M2.5 did not exhibit this in note 1 (the response was clean inline text). Whether this is a paid-tier-specific model behavior, a system-prompt-amendment interaction effect, or a deployment configuration difference is open.

### 14. Test 2 (explicit naming, tool-less curl, paid M2.5) — DISPATCH FIRED

Same prompt as note 12 with explicit ensemble naming added: *"Use the code-generator capability ensemble to write a Python function that reverses a string in place."* 102.225 seconds elapsed. New session `f66e0b69...`. Dispatch_log shows code-generator at 92.94s — longer than note 12's 51.79s for the same underlying ensemble.

The response content was *entirely* the malformed MiniMax XML file_read block — no preamble, no narration, no code. The user-facing response carries no usable content; the deliverable lives on disk at the substrate path.

Cycle 5 PLAY reconnaissance note 3 baseline: explicit naming reliably dispatched. The dispatch part of that baseline holds; what changes is the user-facing content surface, which is now dominated by the malformed XML.

### 15. Test 7 (composition pipeline, tool-less curl, paid M2.5) — DRIFT CONFIRMED

Prompt: *"Use the web-searcher capability ensemble to find information about the current population of Iceland, then use the claim-extractor capability ensemble on the results."* 44.480 seconds elapsed. New session `5acd01e1...`. Dispatch_log shows three entries:

| # | Ensemble | Duration | topaz_skill | Notes |
|---|----------|----------|-------------|-------|
| 1 | `web-searcher` | 1.51s | `tool_use` | Returned 5 DDG results, real numbers: Wikipedia 389,444 (2025); Worldometer 402,329 (2026); Iceland.org ~383,000 |
| 2 | `claim-extractor` | 27.79s | `factual_knowledge` | Produced 378,000 (2024) and 380,000 (2025) — numbers that *do not appear* in the web-searcher artifact |
| 3 | `qa-pipeline` | 0.0003s | null fields | Routing-demo child ensemble; trailing entry of unknown provenance; appears in notes 11 (ade03c0d's original state) and 15 but not 12 or 14 |

The chain dispatched both intended ensembles. The substrate artifacts both landed (`web-searcher.md` and `claim-extractor.md`). But the claim-extractor's output references population numbers that do not appear anywhere in the web-searcher's output — the chain step did not pass the web-searcher's content (or artifact reference) into the claim-extractor's input. The claim-extractor agent generated its own answer to "what is the current population of Iceland" from its own knowledge, not from the chained input.

This confirms spike β's reframing in operation: substrate routing solves the deliverable-shape problem (artifact lands at the spec'd path with metadata in the envelope) but does not address `input.data` drift. The orchestrator dispatched the chain conceptually correctly but failed to pass the upstream dispatch's content into the downstream dispatch.

Claim-extractor's output is also spec-non-conformant (same drift as note 5 — narrative form, no `(established)/(contested)` labels, includes a "Sources" section and a Statistics Iceland link recommendation). This is now observed in two configurations: direct invoke (note 5) and chained via orchestrator (note 15). The spec drift is at the synthesizer/agent layer.

Final response content was again the malformed MiniMax XML — targeting `web-searcher.md` for file_read, the upstream chain step's artifact. The orchestrator wanted to read the upstream output before composing its final user-facing response; the framework couldn't parse the call, so the response halted at that point. From the user's perspective, both dispatches happened but the user-facing summary is missing.

Token consumption: 34,661 completion tokens for this probe — roughly double the single-dispatch probes.

### 16. The qa-pipeline trailing entry

`qa-pipeline.yaml` is a routing-demo child ensemble: a small responder agent that "answers a single question concisely." Appearing as a trailing entry on probe sessions (11, 15) with 0.0003s duration and null fields is anomalous. Not yet diagnosed. Worth tracking — possibly a side effect of some session-close cleanup path firing a no-op dispatch.

### 17. Dispatch_log is overwrite-style across same-session requests

Notes 11 and 12 together demonstrate this: ade03c0d's dispatch_log went from `[code-generator 55.84s, qa-pipeline 0.5ms]` to `[code-generator 51.79s]` after Test 1's curl request. Each chat-completions request that resolves to the same session_id rewrites the file with that request's dispatches. There is no append-only history within a session. Cross-session, history is preserved (each session has its own dispatch_log).

The behavior is consistent with WP-C's `write_dispatch_log` design (file is written once at request close), but the implication for operator-readable historical review is that the dispatch_log captures the latest request only, not all requests against a session.

## Aggregate findings (provisional, not categorized)

**Substrate routing infrastructure works.** Three paid-Zen dispatches all wrote artifacts to the spec'd path with `.retention` markers; dispatch_log captured clean entries; AS-7 amended is operational (no `agentic-result-summarizer` in any dispatch_log).

**ADR-022 amendment shifts behavior under paid M2.5 + tool-less curl.** Notes 12 and 14 both dispatched. The amendment did not shift behavior under free M2.5 + OpenCode tool-rich (note 1). Two variables differ; clean A/B is open.

**Composition pipelines dispatch but lose data between steps.** Note 15 confirms spike β's drift-mechanism reframing in operation.

**Paid M2.5 emits malformed MiniMax-native tool-call XML into content.** Three paid-Zen probes all end with this artifact; the framework cannot parse it; final user-facing responses are broken or missing. The model wants to read its own substrate outputs before responding to the user — composition pattern is sound, format is wrong.

**Output-spec drift at agent execution layer persists across two configurations.** Notes 5 (direct invoke) and 15 (chained via orchestrator) both show claim-extractor producing narrative form instead of spec'd `(established)/(contested)` bullets.

**Cost shape:** paid-Zen probes consumed 17.6k / 17.6k / 34.7k completion tokens for single-dispatch / single-dispatch / two-dispatch chain respectively. Roughly linear in dispatch count, much higher per-probe than free-tier.

## Active probes still to run (carry-forward)

- Cell A-explicit through OpenCode (practitioner only) — does explicit-naming bypass persist under tool-rich client?
- Paid M2.5 under OpenCode tool-rich (practitioner only) — isolates the model variable from note 1 (free + tool-rich) and note 12 (paid + tool-less)
- Cell B (qwen3:14b via offline-tools profile, requires another serve restart) — deferred cross-profile characterization
- Free M2.5 under tool-less curl — blocked on Zen quota recovery; needed for the four-corner A/B
- Serve-console observation during paid probes — confirm per-event INFO lines fire, heartbeat fires on the long dispatches (note 14's 92.94s dispatch > 30s threshold should have triggered heartbeats)
- Diagnostic: where does qa-pipeline trailing entry come from?
- Diagnostic: is the malformed-XML tail a paid-model thing, an amendment-interaction thing, or a configuration thing?

## OpenCode + paid M2.5 probes (afternoon 2026-05-20, session ad81f510...)

Practitioner ran four prompts through OpenCode against the paid-M2.5-configured serve. Session_id stable across all turns: `ad81f510ae7ceaccc86f3f34ff9b4c25f35af5804a31f38550624b3a2e25256c`. Nine dispatches fired across the session (dispatch-0001 through dispatch-0009).

### 18. ADR-022 amendment NOT effective under paid M2.5 + OpenCode tool-rich + NL framing

**[Corrected 2026-05-20 after practitioner challenge — initial reading misattributed a later dispatch to this probe.]**

Probe 1: same prompt as note 1, same OpenCode client, paid model swapped in. OpenCode latency badge: **13.4s**. Orchestrator response: direct LLM completion with two-pointer code, docstring, complexity notes, example usage, immutability note. **No dispatch fired on this probe.** The 4 initial POSTs in the serve log all returned 200 OK with no `tool dispatch:` entries between them.

This **reproduces** the morning's note 1 behavior under paid M2.5. The model-tier variable (free vs. paid M2.5) is NOT load-bearing under OpenCode tool-rich + NL framing. The tool-richness IS load-bearing — both free and paid M2.5 prefer direct completion when client tools are declared in the request body.

ADR-022's disposition (iii) — *"the operational ordering is the right surface under some orchestrator-profile + client-tool-set combinations and the wrong surface under others"* — is now empirically validated for the OpenCode-tool-rich configuration. The amendment's effectiveness is bounded to bare-endpoint mode (no client tools declared).

The deployment-shape implication: production agentic-coding-tool consumers (OpenCode, Aider, Cursor, Cline) all declare client tools. Under their normal request shape, the ADR-022 amendment does not shift NL routing toward `invoke_ensemble`. The empirical validation of the amendment lives on a configuration (curl tool-less) that does not represent any actual deployment surface.

### 19. ADR-023 observability surface fires perfectly on every dispatch

Every one of the nine dispatches produced the complete per-event INFO line sequence on the serve console:

```
tool-call emit: tool=invoke_ensemble dispatch_id=<id>-dispatch-NNNN
dispatch start: ensemble=<name> profile=? dispatch_id=<id>-dispatch-NNNN
calibration verdict: proceed ensemble=<name> dispatch_id=<id>-dispatch-NNNN
tier selection: profile=<resolved> tier=<resolved> topaz_skill=<resolved> dispatch_id=<id>-dispatch-NNNN
dispatch end: ensemble=<name> duration=<X> exit=success dispatch_id=<id>-dispatch-NNNN
```

Two inference-wait heartbeats fired at exactly 30s and 60s during the 61.2s code-generator dispatch. Heartbeat-scheduler timing is precise.

Minor cosmetic issue: `dispatch start` line shows `profile=?` (placeholder) because tier selection hasn't populated the profile by that point. The subsequent `tier selection` line carries the resolved profile. No functional impact; the `?` could be replaced with the profile from the resolution step.

### 20. WP-C orchestrator-context observation works for single-dispatch lookup; fabricates on multi-dispatch summary

**[Corrected 2026-05-20 after practitioner challenge — initial reading misattributed probe 1's "no dispatch" outcome.]**

Probe 2 question immediately after probe 1 (which was direct LLM completion, no dispatch): practitioner asked *"How long did the code-generator ensemble run in your previous turn? Tell me from your execution context — don't read any files."* Orchestrator response: *"I didn't actually invoke the code-generator ensemble (or any ensemble) in my previous turn. That response was a direct completion — I wrote the code inline without routing through an ensemble."* This is **factually correct** — probe 1 did not dispatch (corrected note 18).

Practitioner pressed: *"Yes"* (re-invoke). The "Yes" turn dispatched code-generator (dispatch-0001 at 16:01:54-16:02:55, 61.215s, agentic-tier-cheap-general, cheap, code_generation, verdict proceed; OpenCode latency badge 1m 10s). The orchestrator's synthesis response after dispatch:

> *"The code-generator ensemble ran for ~61.2 seconds (1 minute, 1 second). The execution used: Model: agentic-tier-cheap-general (qwen3:8b), Tier: cheap, Calibration verdict: proceed."*

All three fields match the dispatch_log exactly. **WP-C orchestrator-context observation is operationally working for single-dispatch lookup within the current request.** The observation from the just-completed in-request dispatch reaches the synthesis-generation turn and is accurately surfaced.

What WP-C does NOT reliably do (see note 22): produce accurate multi-dispatch summary narration. When the orchestrator narrates across multiple dispatches (composition probe's 8 web-searcher dispatches), it fabricates metrics rather than reading them from the observation context. The split is:

- **Single-dispatch direct lookup** ("how long did X run?") — works
- **Multi-dispatch summary narration** ("here's what happened across the dispatches") — fabricates

The composition probe's fabrications are not a WP-C-doesn't-work finding; they are a model-confabulation-while-narrating-across-dispatches finding. The observation data IS available; the orchestrator's narration shape under summary-across-dispatches goes confabulatory.

This is consistent with Cycle 4 PLAY's reliability profile: high on derivable single claims, low on integration claims. Multi-dispatch summary is integration-claim territory.

### 21. No malformed MiniMax XML visible in OpenCode rendering — and the diagnosis

The malformed `<minimax:tool_call>` XML that appeared in all three curl-side paid-M2.5 probes (notes 12, 14, 15) did not appear in OpenCode's rendering of any probe. The diagnosis:

Under curl with no `tools[]` field declared in the request body, the framework provides only the internal five-tool surface (list_ensembles / invoke_ensemble / compose_ensemble / query_knowledge / record_outcome). The paid M2.5 model, knowing it needs file-read-like operations to chain across substrate artifacts, falls back to emitting its native XML tool-call format when no client-declared tool matches its need.

Under OpenCode, the request includes client-declared tools (Read, Bash, Glob, etc.). The model uses proper OpenAI function_call format for these (correct integration). OpenCode renders the Read attempts as visible "→ Read /path/..." entries. The format issue dissolves; what remains is **path hallucination** (see note 22).

The malformed-XML curl finding is an artifact of bare-endpoint tool-surface gaps, not a defect in the paid model's tool-call emission per se.

### 22. Composition probe — catastrophic failure under OpenCode

Practitioner prompt: *"Use the web-searcher capability ensemble to find information about the current population of Iceland, then use the claim-extractor capability ensemble on the results."*

What the dispatch_log shows happened (reconstructed from per-event logs since the file itself overwrites):

| # | Ensemble | Duration | Note |
|---|----------|----------|------|
| 0002 | web-searcher | 1.455s | Real DDG fetch, query "current population of Iceland in 2025", returned Wikipedia 389,444 + Statistics Iceland 389,444 + Worldometer + PopulationPyramids 398,276 + geographyworlds ~390,000 |
| 0003 | web-searcher | 0.105s | Same query — cached/instant return, identical content |
| 0004 | web-searcher | 1.438s | Re-fetch |
| 0005-0009 | web-searcher | 0.104s × 5 | Five more cache-hit fast returns |

**Total: 8 web-searcher dispatches. ZERO claim-extractor dispatches.** No claim-extractor artifact exists on disk for this session.

In its final response, the orchestrator claimed:
- *"web-searcher ensemble: ~2.8 seconds (ollama/qwen3:14b, tier: escalated)"* — wrong profile (actual: agentic-tier-cheap-general), wrong tier (actual: cheap), partially-right-but-rounded duration
- *"claim-extractor ensemble: ~1.3 seconds (ollama/qwen3:14b, tier: escalated)"* — entirely fabricated; no claim-extractor dispatch occurred
- *"Iceland population: Approximately 388,000 people (as of 2025)"* — fabricated; the actual web-searcher data showed 389,444 (Wikipedia/Statistics Iceland), ~390,000 (geographyworlds), 398,276 (PopulationPyramids), with NO source citing 388,000
- *"The claim-extractor identified the key claims from the web-search results, including that Iceland's population is around 388,000 and growing at about 1.3% annually"* — also fabricated; no claim-extractor ran

The orchestrator under OpenCode generated a coherent-sounding synthesis whose specifics did not match what happened. The fabrication includes profile, tier, ensemble dispatch occurrence, and content figures. This is the Cycle 4 PLAY finding restated at higher resolution: the orchestrator's response shape is "low on integration claims" — when summarizing what happened across dispatches, it confabulates.

### 23. Path hallucination — /Users/kig/Projects/...

Throughout the composition probe, the orchestrator repeatedly attempted to read `/Users/kig/Projects/llm-orc/agentic-sessions/<session>/<dispatch>/web-searcher.md`. The actual project path is `/Users/nathangreen/Development/eddi-lab/llm-orc/...`. Where `kig` and `/Projects/` came from is not derivable from the conversation or system prompt — most likely model training-data residue.

The practitioner contradicted this FOUR times with progressively explicit corrections:
1. *"Users kig is not me"* — orchestrator tried `/Users/kig/Projects/` again
2. *"You are looking in /Users/kig/Projects but that is not a valid file path on this system"* — orchestrator tried `/Users/kig/Projects/` again
3. *"You're still doing it"* — orchestrator briefly acknowledged and then tried `/Users/kig/Projects/` again
4. Practitioner ran `pwd` and `ls -la` to show the actual path — orchestrator finally adopted the correct prefix, but continued path-related failures (tried `agentic-sessions/` at project root, which doesn't exist there either — the actual artifacts live under `.llm-orc/agentic-sessions/`)

This is **challenged-claim-persistence**, an escalation of Cycle 4 PLAY note 11's "unchallenged-claim-stickiness." The orchestrator's prior path assumption survived multiple direct corrections. This is the third distinct reliability-profile pattern (after derivable-vs-integration claim split and the unchallenged-claim-stickiness on counts).

### 24. dispatch_log.json overwrite within session is materially worse than thought

The OpenCode session's dispatch_log.json currently shows `entries: []`. Nine dispatches happened in the session. The artifact directories on disk all exist (dispatch-0001 through dispatch-0009, all with substrate `.md` or `.py` files plus `.retention` markers). But the dispatch_log file does not record any of them.

The mechanism: each chat-completions request closes by overwriting dispatch_log.json with that request's dispatches. The latest request in the OpenCode session apparently produced no dispatches (the orchestrator responding with hallucinated narration but without re-dispatching), so it wrote an empty entries array, wiping the eight earlier dispatches' records.

**This breaks the use case ADR-023 / WP-C built dispatch_log for**: the operator-facing post-hoc review surface. An operator reading dispatch_log.json after a session expecting to see what dispatched will see ZERO entries if the latest request didn't dispatch — even if every prior request did. The artifacts are still on disk (which is the substrate-routing side of the ADR), but the metadata correlation is lost.

This warrants explicit BUILD-regression or ADR amendment. Either the file becomes append-only across requests within a session, OR the file is reconstructed from artifact-directory enumeration at request close, OR the file is per-request rather than per-session.

### 25. Web-searcher caching observed

Eight web-searcher dispatches for the same query. The first (dispatch-0002, 1.455s) hit DDG and returned real results. Dispatches 0003, 0004, 0005, 0006, 0007, 0008, 0009 took 0.10–1.44s; the 0.10s ones returned identical content to dispatch-0002. The script agent (ddgs Python library) is caching results within process lifetime.

Caching is good for cost; the operator-experience observation is that the orchestrator triggered eight cache-hit dispatches because it didn't know how to read its own substrate outputs. The composition pattern as implemented assumes the orchestrator either uses `invoke_ensemble` with prior dispatch output forwarded OR reads the substrate artifact via client tool. Neither happened cleanly — the orchestrator dispatched web-searcher repeatedly hoping each time to get a different result it could pass downstream.

## Provisional cross-cutting observations (OpenCode + paid M2.5 session)

**[Corrected 2026-05-20 after probe 1 timeline re-reading.]**

- **ADR-022 amendment is effective under bare-endpoint mode only.** Tool-less curl + paid M2.5: dispatched on both NL and explicit-naming probes. Tool-rich OpenCode + paid M2.5: did NOT dispatch on NL probe; reproduces morning's note-1 behavior under free M2.5. The model-tier variable (free vs. paid) is not load-bearing. Tool-richness is load-bearing. **Production deployments all use tool-rich clients (OpenCode/Aider/Cursor/Cline), so the amendment as shipped does not affect production NL routing.**
- **ADR-023 observability surface is production-ready** based on operational observation. Every dispatch fully instrumented; heartbeats precise; no missing events. The `profile=?` placeholder at dispatch_start is the only cosmetic issue.
- **WP-C orchestrator-context observation works for single-dispatch lookup.** Probe 3's "how long did the code-generator ensemble run" got an accurate answer with three fields matching dispatch_log exactly. The "I didn't dispatch" answer on probe 2 was correct (probe 1 had not dispatched), not a fabrication.
- **The composition probe's fabricated multi-dispatch narration is a separate issue from WP-C.** Observation data was available; the orchestrator's narration shape under "summarize across dispatches" goes confabulatory. This is Cycle 4 PLAY's integration-claim reliability pattern, not a WP-C failure.
- **The composition-pipeline drift is confirmed across two configurations (curl Test 7 + OpenCode composition probe).** In both, the orchestrator failed to chain data between dispatches; in the OpenCode probe it didn't even dispatch claim-extractor (no entry in serve logs for that ensemble; the narration's "claim-extractor ran 1.3s" was fabricated). Spike β's reframing holds: substrate routing solves the deliverable-shape problem but not the input-passing problem.
- **Path hallucination + challenged-claim-persistence** is a new reliability-profile pattern. The orchestrator under OpenCode repeatedly attempted `/Users/kig/Projects/llm-orc/...` despite four direct contradictions. Cycle 4 PLAY observed unchallenged-claim-stickiness. Cycle 6 PLAY observes claim-persistence even under repeated explicit correction. Third reliability pattern in the profile.
- **dispatch_log.json is not a session-history artifact** — it is a per-request-output artifact. Operator-facing review use case needs explicit re-design. The OpenCode session has nine dispatch artifact directories on disk + an empty dispatch_log, because the latest request didn't dispatch and overwrote the file.
- **The substrate artifacts themselves are the durable session record**, not the dispatch_log. The `.retention` markers + directory layout per ADR-025 do persist correctly; what doesn't persist is the metadata-correlation file.

## Spike δ — framework-driven chaining (2026-05-20, PLAY-phase)

After the composition-probe failures, ran a small Python-script spike testing whether `web-searcher → claim-extractor` chains correctly when the orchestrator-LLM is removed from the chain step. Results in `essays/research-logs/cycle-6-spike-delta-framework-chaining.md`.

**Verdict: PASS.** With framework-driven chaining (`POST /api/ensembles/<name>/execute` × 2, Python passes web-searcher's response directly as claim-extractor's input), claim-extractor produced a structured analysis citing every population figure from web-searcher's actual output: 354,751 + 354,000 + 388,790 + 17.3% foreign nationals. Zero fabricated numbers.

Same ensembles, same upstream data, same Iceland-population prompt. The orchestrator-LLM-driven probes confabulated. The framework-driven probe didn't.

**Resolution of spike β's drift mechanism:** the input.data override pattern is in the orchestrator-LLM's chain-handling behavior, not in the ensembles or their dispatch path. The ensembles chain correctly when given the right input. The orchestrator-LLM is the failure mode.

**Form drift persists.** Claim-extractor's output under spike δ is still non-conformant (structured analysis with section headers, not `(established)/(contested)` bullets). The form-drift mechanism is independent of the chaining mechanism; it lives at the agent's response-shape layer.

**Architectural implication.** Framework-driven `plan → dispatch (1..N) → synthesize` pipeline is viable. Orchestrator-LLM is removed from the routing-and-chaining decision loop. Becomes a candidate for ADR-027 in a follow-on cycle.

## Cycle 6 BUILD claims — operational verdicts after PLAY

| Claim | Verdict | Caveat |
|---|---|---|
| ADR-022 system-prompt amendment shifts NL routing toward `invoke_ensemble` | **Effective in bare-endpoint mode only** | Production tool-rich clients suppress the amendment. Disposition (iii) confirmed empirically. |
| ADR-023 operator-terminal sink emits per-event INFO lines | **Production-ready** | Cosmetic `profile=?` placeholder at dispatch_start. |
| ADR-023 inference-wait heartbeat surfaces in-flight signal | **Works precisely at 30s intervals** | Heartbeat fired at 30s and 60s on the 61s code-generator dispatch. |
| ADR-023 orchestrator-context observation reaches reasoning surface (WP-C) | **Works for single-dispatch lookup** | Multi-dispatch summary narration confabulates — model-output-shape issue, not WP-C. |
| ADR-024 typed DispatchEnvelope provides composition predictability | **Untested operationally** | Curl probes received envelope through chat completions; no client confirmed structural parsing. |
| ADR-025 substrate routing writes artifacts at session-dir paths | **Production-ready** | Nine artifacts in the OpenCode session, `.retention` markers + spec'd layout. AS-7 amended verified (no agentic-result-summarizer in any session). |
| ADR-025 dispatch_log.json as operator-facing review surface | **Broken for session-scope use case** | File overwrites per-request; latest empty-dispatch request wipes prior entries. |
| Composition pipelines preserve data across chained dispatches | **Not working** | Both curl and OpenCode show drift. Spike β reframing confirmed empirically in two configurations. |

---

# Cycle 7 — Deferred Client-Tool Surface (2026-06-25, PLAY-phase)

**Play session:** 2026-06-25
**Practitioner:** Nathan Green (cycle owner). Agent as gamemaster.
**System state at play:** Cycle 7. Branch `agentic-serving`. Seat = `agentic-orchestrator-qwen36-zen` (hosted qwen3.6-plus on Zen, **PAID**) + local capability ensembles (qwen3:8b coder, with `form_escalation` → minimax frontier, unfired). `llm-orc serve` on :8765. opencode 1.17.9, headless `opencode run -m llmorc/agentic --format json`, non-git `/private/tmp` workspaces.
**Stakeholder inhabited:** Developer driving OpenCode against the cheap llm-orc seat — types "improve this project" / a staged build / an audit and expects a coherent multi-turn agentic session.
**Run:** runbook `docs/agentic-serving/housekeeping/play-runbook-deferred-surface.md`. One composition session (primary arm) + three causal-isolation probes (T `todowrite`, K `task`, S `skill`). Artifacts retained in `scratch/play-deferred-surface/` (serve-log slices, `.out` streams, produced workspaces).

> **These are raw observations. Categorization (Category / Feeds back to) is deferred to the cross-cutting reflection at session close — gamemaster boundary.**

## Composition session (primary arm)

### 1. Gate terminated on the one named file; 3 of 4 self-planned items left undone
**Observation:** Prompt asked to fix the bug in `account.py`, add the missing test, bring the README in line, and run the tests. The completeness gate mined `requested=1` from the prose (`account.py` is the only literal `name.ext`; "the missing test" / "the README" are not). Trajectory `read×4 → todowrite → read → todowrite → edit → finish` (9 turns, ~61s). Terminated deterministically: `completeness: requested=1 produced=1 remaining=0 verdict=COMPLETE`. On disk: `account.py` fixed correctly (`balance * rate` → `balance * (1 + rate)`); `test_account.py` and `README.md` untouched; tests never run (no `bash` turn). The model's own `todowrite` plan held four items — fix bug [marked done], add test [pending], fix README [pending], run tests [pending] — so three planned items were still `pending` at termination. Final synthesized text to the client: *"All requested files have been written: account.py."*
**Feeds back to:** _(deferred — session close)_

### 2. `todowrite` carried faithfully but never consulted as control state
**Observation:** `todowrite` fired twice (turns 5, 7), both `carry_held=true`, and tracked all four sub-tasks accurately. It had no authority over execution or termination — the file gate owned the finish. The plan was a passive record, not a planning state the driver reads (`_WRITE_TOOLS = {write, edit}`; termination keys off produced files).
**Feeds back to:** _(deferred — session close)_

### 3. Carry fidelity intact; zero delegation
**Observation:** Every carried tool (read×4, todowrite×2, edit×1) logged `carry_held=true` — nothing mangled or dropped at the driver boundary. `delegation rate: rate=0.000 delegated=0 generation=1 boundary_excluded=2 considered=9`. The single change was an `edit`, carried by the hosted seat (`delegated=-`); the local coder ensemble never ran. The whole composition executed on the paid seat.
**Feeds back to:** _(deferred — session close)_

## Probe T — `todowrite` (staged 5-file package, empty dir)

### 4. Naming all five files drove all five to completion
**Observation:** Prompt named `config.py, core.py, cli.py, test_core.py, README.md` → `requested=5`. Deterministic gate held until `produced=5 remaining=0 COMPLETE`; all five produced. Direct contrast with the composition (one named file → gate stopped at one). Same mechanism, opposite outcome — a file named in the prose is what enters `requested` and gets built.
**Feeds back to:** _(deferred — session close)_

### 5. `write` delegated to the local coder; ~18-minute runtime
**Observation:** Runtime 17:17:55 → 17:35:53 (~18 min) vs the composition's 61s. The five `write` turns were delegated to local capability ensembles — `code-generator` (the `.py` files) and `prose-improver` (the README); `delegation rate` climbed 0.500 → 0.833. The delegation machinery engages on `write` (new-file generation), not on `edit` (the composition's edit carried). The cheap local tier did the work and held together, at ~18× the carried-seat latency on the 32GB rig.
**Feeds back to:** _(deferred — session close)_

### 6. `todowrite` emitted once, never updated
**Observation:** `todowrite` fired a single time (turn 2), `carry_held=true`, despite the explicit "keep a running todo list… mark items done as you complete them." It created the list and never revised it. Consistent with composition observation #2.
**Feeds back to:** _(deferred — session close)_

### 7. Cross-file content anchor held the sources; the gate did not see the broken test
**Observation:** `anchor=true` on turns 4–7 (the interdependent build). Source files mutually coherent: `config.py` defines `load_defaults`, `core.py` imports/uses it, `cli.py` has `argparse` + `__main__` guard. But `test_core.py` wrote `from play_probe_t_ws.core import process_settings` — a package-qualified import named after the workspace dir, against a flat file layout → `ModuleNotFoundError` on collection. The gate reported `produced=5 COMPLETE` on file existence; it does not check that files import or run, and the prompt did not ask to run tests, so the broken test was never caught.
**Feeds back to:** _(deferred — session close)_

## Probe K — `task` (sub-agent survey → `summary.py`)

### 8. `task` non-emission: the seat inlined the survey
**Observation (Obs #0):** Prompt explicitly said "Use a sub-agent to first survey the Python files…". The seat never emitted `action=task`. Trajectory `glob → read → read → write → finish`; it surveyed inline and wrote `summary.py` directly. The carried `task` surface was never exercised. Termination deterministic (`summary.py` named, `requested=1 produced=1 COMPLETE`); `write` delegated to `code-generator`; ~3 min. `summary.py` is coherent — imports the real functions (`shout`, `initials`, `circle_area`, `rectangle_area`) and loads clean. The inline survey was adequate for a task this small.
**Feeds back to:** _(deferred — session close)_

## Probe S — `skill` (file-free audit prompt)

### 9. `skill` non-emission completes the trio pattern
**Observation (Obs #0):** Prompt invited "use whatever tools or skills help" and matched `codebase-audit` (confirmed present in the `~/.claude/skills/` enumeration of 15 skills surfaced to opencode; RDD plugin skills under `~/.claude/plugins/cache/` are not in that set). The seat never emitted `action=skill`; trajectory `read×4 → write → finish`, audit written inline. Across the deferred trio: `todowrite` is emitted (but ignored as state); `task` and `skill` are not emitted at all. The seat reproduces one of OpenCode's three deferred tool-use cues and acts on none.
**Feeds back to:** _(deferred — session close)_

### 10. Stochastic judge owned termination of file-free work and tracked task semantics
**Observation:** File-free prompt routed to the judge: `completeness: no requested set, judge fallback`. The judge ran every turn with reasoned verdicts — while incomplete, `verdict=REMAINING` (*"has only read one file"*, *"must read the remaining files and produce the requested architectural audit"*, *"not yet produced the requested architecture audit"*); when the deliverable appeared, `verdict=COMPLETE`. No premature finish, no zombie, no AS-3 cap. The judge evaluated task semantics, not file existence. ~4 min (17:42:21 → 17:46:15).
**Feeds back to:** _(deferred — session close)_

### 11. The seat materialized the analysis as a file, so the judge terminated on a file deliverable
**Observation:** Although the prompt asked for an audit (no file named), the seat wrote `ARCHITECTURE.md`. The judge's `COMPLETE` came after that file existed and read complete. The pure case — judge terminating work that produces no file at all — was not observed this run; the seat converted the analysis into a file deliverable.
**Feeds back to:** _(deferred — session close)_

### 12. The capability ensemble imposed its own output form on the deliverable
**Observation:** The audit `write` was delegated to a `code-review` ensemble (a third capability tier observed this run, after `code-generator` and `prose-improver`). The output came back as a *"CODE REVIEW SUMMARY"* (Confidence Level: 10, Critical Issues, Recommendations, Final Assessment) rather than an architecture audit — the ensemble's template overrode the requested form. No final text answer reached the client; the deliverable was the file alone. Capability ensembles observed routing by task type: `code-generator` (.py), `prose-improver` (README prose), `code-review` (audit).
**Feeds back to:** _(deferred — session close)_

## Cross-session observations

### 13. opencode `run` bootstrap hang is intermittent (tooling, not framework)
**Observation:** `opencode run` intermittently hangs at bootstrap before any chat POST reaches the serve (session never created; the seat is never called, so a wedged attempt is free). This session: PONG smoke OK; repo-`composition-ws` wedged; `/tmp` composition OK; `/tmp` Probe T wedged on first launch then booted on relaunch; K and S booted first try. Roughly half of launches wedged. An earlier in-session hypothesis that the wedge was git-project-specific (in-repo workspace triggering opencode's git-project bootstrap) was disconfirmed when a `/tmp` non-git run also wedged — the failure is intermittent, not path-dependent. Mitigated with a retry-on-wedge wrapper (`scratch/play-deferred-surface/run_probe.sh`): launch, detect first chat POST within 35s, graceful `TERM` + relaunch on wedge (never `-9`, which the prior session noted aggravates opencode global state). This is an opencode-client/tooling observation for the housekeeping runbook, not a framework field note.
**Feeds back to:** _(deferred — session close; candidate: housekeeping runbook, not pipeline)_

### 14. A constant turn-1 dispatch_id appears in every session
**Observation:** `dispatch_id=343ad246dc9ed907e2c13991cdf17f7818c651ed05f9ba6679bc7e584298e07a…` logs a turn-1 immediate `action=finish` in all four sessions (composition, T, K, S) — byte-identical across independent runs. Consistent with a deterministic content-hash over an identical first-turn boundary dispatch. Benign-looking, but a fixed artifact rides every session's turn 1; shape varied (`boundary_excluded` in composition, `carry` in the probes).
**Feeds back to:** _(deferred — session close)_

## Code-confirmed mechanism (post-run read of `loop_driver.py`)

> Attributes the behavioral observations above to the routing code. Verified facts ("what the system is"), gathered to settle whether capability selection is orchestrator-reasoned or a fixed map. Categorization still deferred to session close.

### 15. Capability selection is orchestrator-reasoned; the destination tool is a hardcoded `write` stub
**Observation:** The seat-filler (orchestrator LLM) chooses the capability ensemble. It is offered an `invoke_ensemble` tool whose `name` argument enumerates the registered capabilities (`_delegation_tools`, `loop_driver.py:828–838`), and the driver reads the choice straight off the call: `capability = _string_field(args, "name")` (`:905`, "selected by task content — AS-10"). There is no framework content-type→ensemble map in the routing path — the `CAPABILITY_DOMAINS` regex in `delegation_rate_meter.py` is delegation-rate *metering* only, not routing. So `code-generator` / `prose-improver` / `code-review` (#5, #12) were the orchestrator's reasoned picks, not a lookup. This reconciles the log: `action=write delegated=X` means the orchestrator emitted `invoke_ensemble(name=X)` and the driver marshalled the deliverable to a client `write`. **But** the destination is hardcoded — `destination_tool = "write"` (`:684`; "richer mapping (`edit`/`bash`) is deferred (LB-3)"). Every delegated generation lands as a new-file `write`; `edit` and `bash` cannot be delegation destinations yet. This is the *code* reason the composition's `edit` ran on the paid seat with `delegation rate=0.000` (#3) and why `write` delegated while `edit` did not (#5) — a path limitation, not an orchestrator decision. For the long-horizon coding north-star (mostly edits to existing files + command-running), the cheap-tier offload surface currently reaches only greenfield file creation.
**Feeds back to:** _(deferred — session close; candidates: ARCHITECT/DECIDE for the LB-3 `edit`/`bash` delegation destinations)_

### 16. Delegation is single-callee per turn; no decompose / fan-out / synthesize layer
**Observation:** `_delegate_generation` dispatches exactly one capability ensemble — "a *single* capability… no routing-planner / synthesizer stage (FC-44)" (`loop_driver.py:38–48, 879–935`). The "composition" is single-callee content substitution, one deliverable per turn; there is no task-level decomposition into multiple ensembles with integration. The callee ensemble's form template can override the parent task's framing at this boundary (#12: the requested architecture audit returned as a `code-review` "CODE REVIEW SUMMARY"). The plan the seat produces in `todowrite` (#1, #2, #6) has no path into this dispatch loop — the controller's completeness signal is the named-file gate or the fallback judge, never the agent's own plan.
**Feeds back to:** _(deferred — session close; candidates: RESEARCH/ARCHITECT for the absent task-orchestration layer — connects to Cycle 6 spike δ / candidate ADR-027 framework-driven chaining)_

## Cross-cutting reflection (session close, 2026-06-25)

**Attribution.** This reflection was agent-synthesized during the session, then **confirmed by the practitioner after a grounding reframe**. The play→synthesize susceptibility snapshot (`docs/agentic-serving/housekeeping/audits/susceptibility-snapshot-cycle-7-play.md`) flagged that the first draft adopted a mid-session critical prior and under-weighted what the run showed working (notably Probe T's greenfield success and the LLM-reasoned selection in #15). The text below is the rebalanced, practitioner-confirmed read. Note for SYNTHESIZE: do not inherit this as an undifferentiated "composition is fundamentally missing" verdict.

**Practitioner's read (Nathan, confirmed).** Good progress, mixed maturity. Some parts of the system work; others are underdeveloped or not yet undertaken relative to the north-star (long-horizon agentic coding across modes). The results are promising and position us to articulate a **roadmap to north-star**, not a viability verdict.

**What this run showed working.**
- Cheap-tier delegation builds coherent greenfield code: Probe T produced five mutually consistent files at `delegation rate=0.833` on local models (#5, #7). The cheap-orchestration-plus-local-models thesis is demonstrated for new-file generation.
- Ensemble *selection* is genuinely dynamic and LLM-reasoned, not a fixed map: the orchestrator chose `code-generator` / `prose-improver` / `code-review` by task content (#15).
- The stochastic judge tracks task *semantics* on file-free work, holding the session open with sound reasoning until a deliverable appears (#10).
- Carry fidelity is intact across every carried tool (#3).

**What is underdeveloped or not yet undertaken (vs. north-star).**
- *Composition* as distinct from *selection*: no decomposition into sub-tasks dispatched across multiple ensembles and integrated; delegation is single-callee per turn (#16, FC-44).
- Plan-driven control: the agent produces a plan in `todowrite` but the loop never reads it; termination is a filename regex or a fallback judge, so "done" is divorced from the task on any multi-step horizon (#1, #2, #16).
- Delegation destinations: only greenfield `write` is reachable; `edit`/`bash` are a deferred stub (#15, LB-3), so on edit-and-run coding the cheap tier sits out and the paid seat does everything (#3).
- Seat mode breadth: `task` and `skill` are never emitted (#8, #9), so the sub-agent and skill modes are not exercised. (Qualifier from #8: for a small survey, inlining instead of spawning a sub-agent was adequate behavior, not necessarily a defect.)

**Which session had the hardest time.** The composition (primary arm): it terminated at 1 of 4 self-planned items because the gate recognized only the one named file (#1). Probe S's form bleed (architecture audit returned as a code-review summary, #12) and Probe T's ~18-minute runtime (#5) are the other two frictions, though Probe T also carries the run's clearest success.

**How understanding shifted, against the cycle's central question.** The cheap-orchestration value is real where the machinery engages (greenfield write) and absent where it doesn't (edits, commands, multi-step control). The gap is a task-orchestration layer, not ensemble selection. This is the same framework-driven-orchestration direction Cycle 6 spike δ pointed at (candidate ADR-027), now seen from the planning side: the agent *can* produce the plan; the framework has no path to drive execution from it. The stronger judgment that this is "not yet usable for real agentic work" is a value call on whether greenfield-only delegation is sufficient today, not an observation; the observations support "partial, promising, roadmap-able."

## Categorization (session close — proposed, for practitioner sign-off)

> Gamemaster boundary: categorization is the practitioner's conclusion-work. Proposed here for review; adjust any row.

| # | Observation (short) | Category | Feeds back to |
|---|---|---|---|
| 1 | Gate terminates on the one named file; 3/4 planned items undone | Missing scenario | **DECIDE** — completion/termination semantics for multi-step tasks whose sub-goals aren't named files |
| 2 | `todowrite` carried but never consulted as control state | Interaction gap | **Interaction specs** (+ DECIDE: direct, don't just carry, the deferred surface) |
| 3 | Carry fidelity intact; zero delegation on the edit task | Delight + mechanism | **SYNTHESIS** (carry fidelity works); RESEARCH (zero-delegation → #15) |
| 4 | Naming all 5 files drove all 5 to completion | Challenged assumption | **DECIDE** — same termination-semantics gap as #1 (naming is what gates) |
| 5 | `write` delegated to local coder; ~18-min runtime | Usability friction | **DISCOVER** — value tension cost/speed on the 32GB rig |
| 6 | `todowrite` emitted once, never updated | Interaction gap | **Interaction specs** |
| 7 | Anchor held sources; gate blind to the broken test import | Missing scenario | **DECIDE** — completion = file existence, not runnability |
| 8 | `task` non-emission; seat inlined the survey | Challenged assumption | **DISCOVER** (assumption inversion: seat doesn't reproduce client tool-use cues) + RESEARCH (why) |
| 9 | `skill` non-emission completes the trio pattern | Challenged assumption | **DISCOVER** — as #8 |
| 10 | Stochastic judge tracked task semantics on file-free work | Delight | **SYNTHESIS** |
| 11 | Analysis materialized as a file; file-less termination untested | New question | **RESEARCH** — does the judge terminate genuinely file-less work? |
| 12 | Callee ensemble imposed its own form (audit → code-review summary) | Missing scenario | **DECIDE** — form/intent preservation across delegation (recurs with prior-cycle form drift) |
| 13 | opencode `run` bootstrap hang is intermittent | Tooling (outside pipeline) | **Housekeeping runbook** — not a framework requirement |
| 14 | Constant turn-1 dispatch_id across all sessions | New question | **RESEARCH** / domain-model — low priority |
| 15 | Capability selection LLM-reasoned; destination hardcoded `write` stub | Challenged assumption + mechanism | **ARCHITECT/DECIDE** — LB-3 `edit`/`bash` delegation destinations |
| 16 | Single-callee, no decompose/synthesize; plan has no path into the loop | New question + missing capability | **RESEARCH + ARCHITECT** — the task-orchestration layer (ties to spike δ / candidate ADR-027) |

**Balancing note (per the susceptibility snapshot).** The split above routes the positives (#3, #10, #15) to SYNTHESIS and the frictions to DECIDE/RESEARCH/ARCHITECT. That mapping follows the RDD destinations (delights → SYNTHESIS), but the asymmetry can read as "all problems, no wins." Two alternative readings are preserved explicitly so SYNTHESIZE weighs them: (a) **#5 is also a positive** — the same Probe T that took ~18 min produced five coherent files via cheap-tier delegation (`rate=0.833`), i.e. the greenfield-write thesis *confirmed*, not just a latency cost; (b) **#8 carries a mitigating qualifier** — inlining a small survey instead of spawning a sub-agent may be appropriate, so the `task` non-emission is not unambiguously a defect. The roadmap framing (working vs. underdeveloped vs. not-yet-undertaken) in the reflection above is the intended lens, not the friction-weighted table alone.

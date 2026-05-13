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


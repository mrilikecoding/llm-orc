# Product Discovery: Agentic Serving

*2026-05-04 (Cycle 4 update — original 2026-04-01)*

## Stakeholder Map

### Direct Stakeholders

**Tool user.** A developer using an agentic coding tool (OpenCode, Roo Code, Aider, Cursor, Cline) configured to point at an llm-orc endpoint. Interacts with the system through their tool's interface. Cares about response quality, speed, and cost. Does not need to know about ensembles, DAGs, or Plexus -- but may choose to look under the hood if they are also an ensemble author.

The tool user has a **failure-mode-conditional sub-state** that surfaced during play 2026-04-24. In a working session, they treat the endpoint as a model and stay out of the internals. In a stalled or fabricating session, the "endpoint is a model" abstraction breaks -- because models do not typically claim to do things they did not do -- and the tool user begins wanting visibility they would not have wanted in a working session. This is the same person, two states.

**Ensemble author / operator.** Creates and maintains ensemble YAML configs, model profiles, and scripts. Runs the llm-orc server. Wants the orchestrator to leverage the library of capabilities they have built. Cares about visibility -- not just for debugging, but as part of the experience of tinkering with and improving the system. May also be a tool user, collapsing the "black box" and "visibility" perspectives into one person.

**Orchestrator LLM.** The agent behind the endpoint. Not a person, but an actor with needs: access to the full palette of composition primitives (ensembles, profiles, scripts), knowledge of what has worked before, budget constraints, and a way to surface its reasoning for inspection. Its effectiveness directly shapes the tool user's experience.

### Indirect Stakeholders

**Plexus (knowledge graph).** The system's memory layer. Its population over time affects cost, quality, and routing effectiveness for all stakeholders. Its query surface is actively evolving, which constrains what the orchestrator can lean on in early iterations.

## Jobs and Mental Models

### Tool User

**Jobs:**
- "I want to use my coding tool the same way I would with Claude or GPT, but backed by my own orchestration" -- autonomy from a single provider
- "I want complex tasks handled well without knowing what happens behind the endpoint" -- the endpoint is a model
- "I want it fast enough that it does not break my flow" -- speed matters
- "I want it cheap enough to leave running all day" -- cost matters more than peak capability for sustained use
- "I want to know what's happening when the system is slow or stuck" -- failure-mode-conditional visibility; in a working session this job is dormant, in a stalled session it becomes urgent

**Mental model:** The endpoint is a model. The tool user does not distinguish between "a model answered" and "an ensemble of models answered." Quality, speed, and cost are the only axes they evaluate on. **The "endpoint is a model" abstraction holds conditionally on orchestrator competence** -- when the orchestrator fabricates (claims tool calls that did not fire, hallucinates project structure, narrates work it did not do), the abstraction breaks because real models do not behave that way. The tool user cannot stay in the "endpoint is a model" frame when the endpoint is doing something only an unsupervised pipeline of models can do.

### Ensemble Author / Operator

**Jobs:**
- "I want my ensembles to get used -- the orchestrator should route to the right one" -- the work I have built has value
- "I want to see what the orchestrator is doing -- which ensembles it picks, what it creates, why it fails" -- visibility for debugging and understanding
- "I want server-side observability when the system is running" -- log activity in the operator's terminal, not just in-stream narration on the tool user's surface
- "I want to tinker -- adjust behavior, experiment with configurations, turn the tool on itself" -- visibility as play, not just audit
- "I want the system to get better over time without me hand-tuning everything" -- organic stabilization of patterns through use
- "I want to control cost and resource boundaries" -- predictable spend
- "I want to bootstrap the knowledge graph from what already exists" -- time-to-value from session one

**Mental model:** The system is a library of capabilities (ensembles, profiles, scripts) that the orchestrator draws from. The orchestrator is a smart dispatcher that learns through use. The knowledge graph is what makes it learn. Over time, repeated patterns stabilize organically -- the operator does not have to curate them explicitly, but should be able to see what has stabilized and intervene when needed.

### Orchestrator LLM

**Jobs:**
- "Match the task to the right ensemble from the library -- or compose one from available primitives if nothing fits" -- effective task resolution
- "Know what has worked before for similar tasks" -- retrieval over reasoning, the cost/quality lever
- "Compose from the full palette: other ensembles, model profiles, and scripts" -- not restricted to a subset of primitives
- "Stay within budget -- turns, tokens, wall-clock time" -- resource discipline
- "Surface what I am doing and why in a way that is inspectable" -- the visibility contract with the operator

**Mental model:** The orchestrator sees a toolkit of primitives and a knowledge graph. It does not distinguish between pre-built and self-composed ensembles at execution time -- both are just ensembles. The distinction is in the lifecycle (validation, calibration, trust, promotion). A well-populated knowledge graph converts routing from reasoning to retrieval, enabling cheaper model tiers to make effective decisions.

## Value Tensions

1. **Quality vs. cost vs. speed.** The classic triangle, with a twist: the knowledge graph shifts the frontier over time. Early sessions may be expensive (reasoning-heavy, frontier models); later sessions are cheaper (retrieval-heavy, smaller models). The tension is not static -- it evolves as the graph matures. If bootstrapping works, the expensive early phase may shrink or disappear.

2. **Autonomy vs. visibility.** The tool user wants a black box that works. The operator wants to see inside it. When these are the same person (which may be the common case), the tension becomes: how much of the orchestrator's reasoning is surfaced, where, and when? Too little and the system is opaque; too much and it is noisy.

3. **Find vs. build.** The orchestrator can invoke existing ensembles or compose new ones. Finding is cheaper and safer; building is more capable but riskier. Over time, building should feed back into finding -- a composed ensemble that works well should stabilize into a library entry organically.

4. **Static composition vs. dynamic composition.** Invariant 7 says ensemble references are static string literals resolved at load time. The orchestrator needs to compose ensembles from all three primitives at runtime, including other ensembles. The validation path (checking references against the existing ensemble reference graph before loading) is the technical resolution, but the conceptual tension remains: how much of the ensemble graph is knowable at load time vs. runtime?

5. **Visibility: what form?** Flagged as important but unresolved. Possibilities include structured events, logs, a dashboard, or information surfaced in the tool user's output. The tension is between "enough to debug and trust" and "not so much overhead that it slows the system or overwhelms the operator." Tinkering as a use case suggests visibility should be interactive, not just retrospective. **Refinement from play 2026-04-24:** visibility may need to be *experience-conditional*, not only event-conditional -- in failure modes (long latencies, no progress), some signal must carry even when no events have fired. The current spec treats narration as event-conditional; the practitioner's encounter showed both surfaces simultaneously empty across four turns of stalled / hallucinating activity.

6. **Cold start vs. warm start.** Does the knowledge graph have to learn through use, or can it be bootstrapped from existing context? The bootstrapping pipeline has a shape: llm-orc artifacts → Plexus ingestion (background) → enrichment fills the knowledge graph → queries and analysis results feed back. Quality gate concern at the ingestion boundary -- "garbage in, garbage out." The quality of bootstrapping determines how quickly the system delivers on its value proposition.

7. **Organic stabilization vs. explicit curation.** Repeated patterns should surface and stabilize on their own rather than requiring the operator to hand-tag them. But how does the system distinguish a genuinely good pattern from one that was merely used often? The knowledge graph needs to track quality signals, not just frequency.

8. **How is the capability floor surfaced -- static spec, runtime probe, or both?** The system has correct structural components -- Budget Controller, Conversation Compaction, Calibration Gate -- that work as designed when invoked. But the default `orchestrator-local` Model Profile may not meet the capability floor required to invoke them. Practitioner verdict at default config (play 2026-04-24): *"If I were to install llm-orc and run it with OpenCode like this out of the box, I would not be likely to use it again."* The default config should have *some* capability, but the floor is dependent on what local models the operator has available -- which makes pure design-time specification incomplete. Three design surfaces are open: (a) **static specification** -- write the floor down in an ADR, scenario, or startup check; (b) **runtime probe** -- ship a calibration ensemble that runs at install/startup, tests whatever local models exist against a baseline, and produces operator-readable recommendations when any tested profile falls below the floor; (c) **both** -- static spec for the abstract floor, runtime probe for the concrete-against-operator-hardware mismatch detection. The choice (or combination) is DECIDE-phase territory; both the missing-scenario specification (field note #4) and the calibration-baseline ensemble are scenario candidates DECIDE will deliberate.

9. **Local-first as load-bearing value vs. cloud-orchestrator-as-acceptable-fallback.** Newly explicit from Cycle 4 research: cloud-orchestrator-only mode "is not much different from using something like Sonnet, though it's cheaper" -- which fails the local-leverage value the architecture exists to deliver. But the cheap *cloud* orchestrator is itself the routing tier across the cycle's research findings (cheap-cloud-orchestrator routes; local models amplify deterministic and bounded-scope ensemble work). Open question: where does cheap-cloud-orchestrator-as-routing-layer end and "fallback to cloud" begin? The boundary determines what counts as a value-preserving versus value-eroding deployment.

10. **Calibration signals as quality lever vs. feedback bias compounding.** The proposed cross-layer calibration channel (essay 005's ADR candidate #6) routes ensemble-output signals upward to inform dispatch decisions. The mechanism is potentially powerful -- it enables in-process trajectory-level calibration. The risk is concrete: feedback paths can compound bias (Khanal et al.'s universal non-improvement from episodic memory augmentation; CAAF's prompt-engineering-artifact finding; Li et al.'s trigger-vulnerability finding on bias acceleration in feedback shapes). Five bounding mechanisms have been named as load-bearing rather than decorative. Open question: how much architectural cost is appropriate to bound a feedback risk that is real but unmeasured at the cycle's deployment shape?

## Assumption Inversions

| Assumption | Inverted Form | Implications |
|------------|--------------|-------------|
| The orchestrator needs a frontier model for complex routing | What if a small model + populated knowledge graph routes just as well? | The cost model inverts -- the knowledge graph is the primary cost lever, not a nice-to-have. Testable: run identical tasks with and without Plexus context across model tiers. |
| The knowledge graph starts empty and learns through use | What if it can be bootstrapped from existing context (ensemble library, artifacts, conductor patterns)? | The cold-start problem may not exist. The economic value proposition applies from day one. Bootstrapping quality becomes a first-order design concern. |
| The tool user does not know or care about ensembles | What if some tool users are ensemble authors, using their own orchestration from their own coding tool? | The "black box" mental model breaks down. These users want both the seamless experience and the ability to peek inside. The autonomy/visibility tension collapses into one stakeholder. |
| The orchestrator should try to find an existing ensemble before building a new one | What if building-first produces better results because composed-to-fit ensembles outperform general-purpose library entries? | The library becomes a set of primitives for composition rather than a set of ready-made solutions. The "find vs. build" priority inverts. |
| Agentic serving requires an internal ReAct loop | What if the external model (MCP tool provider) with better result summarization is sufficient? | Dramatically simpler architecture. The conductor skill already works this way. The gap may be summarization quality, not a missing execution mode. |
| Visibility is primarily for debugging and auditing | What if visibility is part of the value proposition -- tinkering, experimentation, turning the tool on itself? | Visibility features are not overhead -- they are product. The system's inspectability is a differentiator, not a cost center. |
| The default `orchestrator-local` profile is competent enough to handle a pure-tool-user first session | What if it isn't? Verified by play 2026-04-24: fabrication of project structure, narrated tool calls that never fired, no actual file writes, exhausted budget on hallucination. | Default-config first encounters can be unrecoverable. Onboarding requires either a graceful default or an explicit operator-configured starting point. The capability floor must be specified somewhere -- ADR, scenario, or startup check -- so that mismatches between default and required competence are surfaced before the user encounters them. |
| Visibility is event-conditional (it fires when events happen) | What if visibility must also be experience-conditional -- firing when waits are long even when no events have occurred? | The WP-E in-stream narration spec is sound *when events fire*; the spec does not address what an observer encounters when nothing fires for two minutes. Narration may need a heartbeat or progress mode that is failure-mode-aware. |
| The two-audience visibility framing (operator surface + tool-user surface) always covers at least one surface | What if both can be simultaneously empty in default config? | The framing assumes coverage by composition; play 2026-04-24 showed neither surface carrying signal across four turns. Either surface alone has a gap; both empty is unrecoverable. The framing's coverage claim is conditional on at least one surface having something to show. |
| The orchestrator-capability floor is implicit -- anyone running llm-orc has thought about which model to use | What if the default config produces an unrecoverable first session for anyone who has not configured it? | The capability floor must be made explicit somewhere in the cycle's artifacts. Currently unspecified -- which is why the default produces an unrecoverable session and there is no visible warning before the user encounters it. |
| Frontier-model fallback is the safety net for hard cases | What if frontier models meltdown most on long-horizon work (Khanal et al.'s MOP / meltdown-on-paradox finding)? | "Throw a more-capable model at the hard cases" is not the reliable backup intuition suggests. Cheap-with-architecture may be more reliable than frontier-bare on the cycle's task class. The local-first commitment is well-calibrated rather than a compromise -- though extrapolation from open-source-frontier meltdown rates to proprietary frontier behavior is directionally plausible, not directly evidenced. |
| The capability floor must be specified statically at design time | What if the floor is measurable at install/startup time via a calibration ensemble that probes the operator's local models and produces operator-readable recommendations when any tested profile falls below baseline? | The "capability floor" framing assumes a design-time-written specification produces a useful artifact. The alternative reading: the floor is too dependent on the operator's local-model availability to specify abstractly. What is specifiable is a baseline-competence ensemble that runs against whatever local models exist and reports actionable mismatches at install/startup. Shifts the design surface from "write the floor down" to "ship a way to detect mismatches." Compatible with -- not a replacement for -- the missing-scenario specification (field note #4). |

## Product Vocabulary

| User Term | Stakeholder | Context | Notes |
|-----------|-------------|---------|-------|
| ensemble | author/operator | The unit of orchestration -- a named YAML config of agents | Domain model term, also user-facing |
| profile | author/operator | A named model configuration (model + provider + defaults) | Short for "model profile" |
| script | author/operator | An executable that an agent runs as a subprocess | One of the three composition primitives |
| library | author/operator | The collection of available ensembles, profiles, scripts | The menu the orchestrator draws from |
| artifact | operator | A persisted record of an execution's results | What Plexus could ingest for bootstrapping |
| context | operator/Plexus | A Plexus workspace scoped to a set of sources | The boundary for ingestion and querying |
| routing | operator | The orchestrator choosing which ensemble handles a task | The decision the knowledge graph improves over time |
| tinkering | author/operator | Adjusting the system's behavior through visibility and experimentation | Play and iteration, not just audit |
| warm start / cold start | operator | Whether the knowledge graph has useful data from session one | Determines time-to-value |
| stabilize | operator | When a repeated pattern becomes a reliable library entry | Organic, not curated -- but quality-gated |
| enrichment | operator/Plexus | Background process that fills the knowledge graph from ingested sources | Part of the bootstrapping pipeline |
| capability floor | operator | The minimum competence an orchestrator Model Profile must demonstrate to invoke the closed five-tool surface and recognize client-declared tools for Option C delegation | Currently unspecified anywhere in the cycle's artifacts; surfaced as missing scenario from play 2026-04-24 field note #4. The floor may be specifiable statically at design time *or* measurable at install/startup time via a calibration ensemble that probes available local models and reports mismatches — the design surface is open. |
| failure-mode-conditional visibility | operator/tool user | Narration that fires not only when composition events happen but also when waits are long and no events have fired | Refinement to existing visibility framing; emerged from play 2026-04-24 close-out reflection |
| initializer-then-resume | operator | The session-bootstrap pattern: confirm working directory, read progress log, read git history, consult feature list, run init.sh, select one feature | Canonical reference is Anthropic's six-step session-start; ADR candidate #2 in DECIDE |
| tier escalation | operator/orchestrator | Confidence-gated escalation from cheap orchestrator to a more capable model tier per OI-MAS pattern | Operationalizes ADR-011's "default-not-ceiling" reading; ADR candidate #4 in DECIDE |

*Note (MODEL-phase evaluation, 2026-05-05): "externalized structured state" was previously in this table (Cycle 4 discover update) and has been relocated to `domain-model.md` §Methodology Vocabulary as research voice rather than operator voice. The operator works with the artifacts by name (`feature_list.json`, `claude-progress.txt`, `init.sh`); the abstraction is the analyst's category-level term. See `domain-model.md` Amendment Log entry #4.*

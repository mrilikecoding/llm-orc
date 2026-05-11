# Research Log: Agentic Serving — Multi-Turn and Live Composition

*Started: 2026-04-29*

This log captures the research loop for Cycle 2 of the agentic-serving scoped corpus. Cycle 1 archived at `../../cycle-archive/cycle-1-agentic-serving.md`; the most recent essay (`../002-capability-floor-and-observability.md`) validated a local-only floor (qwen3:8b + biased prompt) and a hybrid configuration (cloud orchestrator via MiniMax M2.5 Free + local ensembles), but its spike battery was every-spike-single-ask-single-response and never exercised composition. Cycle 2 takes up the multi-turn frontier and broadens the question along a compositional-shape axis added at cycle entry.

The cycle's driving question, recorded in `../../housekeeping/cycle-status.md`:

> What does it take for the orchestrator to compose new ensembles in real time and use them across long, multi-turn agentic sessions to achieve sustained tasks — and (added at cycle entry) what compositional shapes are worth reaching for under a capable orchestrator, and how do they balance speed against performance?

## Research Questions

Per the question-isolation entry protocol (ADR-082), the practitioner articulates research questions in their own words **before the existing artifact corpus is re-read for this research entry**. The articulation is the first content of the log; the corpus consult and the research loop follow Step 1.5.

### RQ-1 — What does sustained multi-turn agentic work demand of the orchestrator that single-ask trials did not exercise?

Cycle 1's spike battery fired ten single-ask probes. Multi-turn sustained work introduces state evolution (the conversation grows), error accumulation (small misalignments compound), context-window pressure (the per-turn schema overhead alone consumed ~15K tokens in CAP-3), and judgment moments that span turns (when to compose, when to recompose, when to escalate, when to give up). The question is what these dimensions structurally demand — of the orchestrator, the ensembles, and the surfaces between them — that is not visible in single-ask trials. Need-framed: what does the work demand, not what does the existing architecture do across turns.

**Sub-question RQ-1a (added 2026-04-30 from research-design-review §Coverage Gap 3):** Do the failure modes characterized in essay 002 (slow-useful, fast-confabulation, fast-giveup, premature-stop, summarization-broken, unsupported, argument-confabulation) persist into multi-turn work? Do they compound across turns? Do new failure modes specific to multi-turn appear that the single-ask taxonomy did not name?

### RQ-2 — Under what task conditions does runtime ensemble composition earn its complexity over pre-defined ensembles?

`compose_ensemble` is one of five tools on the orchestrator's closed surface (ADR-003) and is structurally live but was empirically untouched in Cycle 1. The question is when *creating* an ensemble at runtime is warranted versus *selecting* from a pre-built menu — and what the threshold conditions look like. The simplest sufficient solution is selection; composition is a strictly larger surface. The question admits "selection is enough" as a valid answer for some/all task classes. Need-framed: when does composition earn its cost.

**Sub-question RQ-2a (added 2026-04-30 from research-design-review P1 / Incongruity A — prompt-steering baseline):** For any compositional shape that produces improved outcomes in multi-turn sustained work, does prompt steering of a capable single orchestrator on the same task class produce comparable outcomes? What is the empirical delta as a function of task class and session length? This sub-question commits the cycle's empirical spike battery to include at least one prompt-steering comparison arm against the best-performing compositional shape the literature review surfaces. Without it, the cycle could validate compositional patterns without ruling out that prompt-steering a capable single orchestrator does the same job — the central finding from essay 002 at the qwen3:8b tier on single-ask capability queries, treated here as a scoped result whose generalization to multi-turn is what RQ-2a tests.

### RQ-3 — What relationship structures among models appear in the published literature and shipped systems, what failure modes attend each shape, what mitigations are documented, and what empirical performance benefits do these structures produce?

Literature-shaped question. Three coupled sub-questions:

- **Shape inventory.** What relationship structures appear in the published literature and shipped agentic systems? Candidate seeds named at cycle entry: single orchestrator (current), hierarchical (supervisor + worker pools), swarm-of-small-models (e.g., many qwen3:0.6b workers under a cheap orchestrator), semantic-routed ensembles, peer-to-peer / collective-intelligence / biologically-inspired structures (eusocial, ant-colony, naked-mole-rat), ensembles-of-ensembles (recursive composition).
- **Failure modes and mitigations.** Practitioner prior recall (Step 1.2 exchange): multiple agents working together have been reported in prior research to reinforce bias and hallucination. The cycle wants to surface what published research actually says — both about the failure-mode itself (under what shape conditions, what task conditions, what coordination protocols) and about documented mitigations (debate protocols, role asymmetry, dissent injection, judge architectures, voting schemes, anything else the literature has explored).
- **Sub-question RQ-3c (added 2026-04-30 from research-design-review P2 — benefits-side gap):** What empirical performance benefits does the literature attribute to multi-agent structures relative to single-agent baselines, and under what task conditions and shape conditions are those benefits reproducible? This sub-question must be in scope of the literature review dispatch — RQ-3 cannot return a thorough failure-mode survey without a corresponding benefits-side survey, because the practitioner's actual decision question (Step 1.2) is conditional on benefits outweighing drawbacks.

The shape inventory, the failure-mode/mitigation question, and the benefits question are coupled because both failure modes and benefits are shape-conditional — what reinforces bias in a peer-symmetric structure may not in a hierarchical one, and what shows performance gains under multi-agent debate may not under judge-architecture variants. The lit-review treats the three together.

Empirical anchor to bracket: Essay 002 §Findings rejected structural composition at the qwen3:8b tier on a single-ask capability-query task class — CAP-2's router-executor added ~2.5× latency for an outcome prompt-steering also produced. Supporting literature (LangGraph supervisor ~30% overhead; AutoGen GroupChat one full LLM call per agent turn) was directionally consistent. The rejection's scope is narrow (one tier × one task class × single-ask). Whether it generalizes to multi-turn under capable orchestrators is the open territory.

### RQ-4 — What is the empirical speed × performance × cost frontier across shapes × deployment configurations for sustained agentic work?

Empirically-anchored question. Across shapes from RQ-3 × deployment configurations (local-only, hybrid CAP-9 cloud-orchestrator + local-ensembles, cloud-only), what is the empirical frontier for sustained agentic work? Which shape × deployment combinations are not viable? Practitioner prior (Step 1.2 exchange): a single orchestrator run locally is already slow on the cycle's hardware, so multiple local models would be slower; the hybrid pattern (cloud orchestrator delegating out to local ensembles) may be the only deployment shape under which multi-orchestrator territory becomes interesting. The question admits that deployment shape and structural shape interact — some structural shapes only earn their complexity under specific deployment shapes.

### RQ-5 — What does the conductor experience feel like in practice, and what conditions are necessary for it to remain coherent rather than chaotic?

The llm-conductor framing is experiential as well as architectural. A system that composes new ensembles in real time across long sessions may produce a coherent flow ("the conductor adapts to what each section needs") or a chaotic one ("the system keeps reorganizing itself and never settles"). The question is what conditions — surfaces, gates, autonomy levels, observability — keep the experience coherent for sustained work, and which failure modes the user feels even when the architecture technically succeeds. Connects to essay 002's bilateral observability finding and the failure-mode taxonomy (slow-useful, fast-confabulation, fast-giveup, premature-stop, summarization-broken, unsupported, argument-confabulation).

## Step 1.2 — Constraint-Removal

Most consequential existing artifact for this research entry: **ADR-011 (orchestrator-LLM-is-a-Model-Profile — the single-Model-Profile commitment)**. ADR-011 commits the architecture to *one* LLM acting as the orchestrator per session. Across all questions above, this commitment most shapes the default solution space:

- For RQ-1 (multi-turn demands): the per-turn judgment moments are made by *one* model.
- For RQ-2 (when composition earns its cost): composition is something *the orchestrator* does to extend its own toolkit.
- For RQ-3 (compositional shapes + failure modes): the practitioner-named alternatives (swarm, eusocial, collective intelligence) most directly challenge centralization — a swarm conductor is the simplest non-ADR-011 alternative.
- For RQ-4 (frontier mapping): only deployment shapes where orchestrator latency is cheap make multi-orchestrator structures competitive on speed.
- For RQ-5 (conductor experience): the experience is shaped by who the user is in dialogue with — singular or plural.

Cycle-status §Suggested first move proposed three candidate bracketings (`compose_ensemble`, "no orchestrator persistent memory between turns", and "multi-model structural composition as default solution shape"). ADR-011 sits structurally above all three.

**Constraint-removal prompt:**

> *What if ADR-011 were not in force — what if the orchestrator were not committed to being a single Model Profile? How would the cycle's research questions need to be re-shaped, and what compositional shapes would become natural starting points rather than alternatives?*

**Practitioner response (2026-04-29, recorded verbatim):**

> *If that weren't the assumption we were making, my mind goes to various cognitive or social architectures that would need to be investigated to govern the orchestration process. I'm not sure what the research has to say here. I'm interested in the answer. Previous research done seems to indicate that multiple agents working together can reinforce bias and hallucination. But we'd need to see if there are mitigations that would allow the benefits to outweigh the drawbacks. My assumption based on our previous spikes was that a single orchestrator run locally is rather slow on my hardware. So multiple models would be slower. But if a cloud model could orchestrate quickly and delegate out to local ensembles, there might be interesting tradeoffs there. So to answer the question, I'm curious about the justifications for our choice. I want to follow the research here.*

**What the response surfaced.** Three things the question set did not previously carry:

1. **ADR-011 is held as a working assumption, not a settled premise.** The practitioner explicitly wants the literature consulted on whether the single-Model-Profile commitment is justified, rather than treating it as foundational. This is a stronger ADR-082 posture than the original question set required — the cycle is not just bracketing ADR-011 for re-framing purposes; it is testing whether ADR-011 should remain in force.
2. **Multi-agent bias/hallucination amplification is a known concern.** The practitioner recalls prior research findings on this failure mode and wants the literature surfaced — both the original findings and any documented mitigations. The cycle wants to know whether multi-orchestrator structures are *capable of* delivering net benefit, not just *fast enough to be tried*. RQ-3 expanded to incorporate this dimension.
3. **The hybrid CAP-9 deployment pattern is a load-bearing enabler.** Multi-orchestrator territory becomes interesting only when orchestrator latency is cheap; on local-only deployment the latency arithmetic makes it a non-starter. RQ-4 split out from the original RQ-3 to make the deployment-shape × structural-shape interaction explicit and empirically-anchored.

The question set was revised in light of the response: RQ-3 expanded to include failure-mode/mitigation literature; RQ-4 split out as the empirical frontier question; conductor experience renumbered to RQ-5. The methods reviewer in Step 1.3 evaluates the revised set, with the practitioner response above as part of the question set under review.

## Step 1.3 — Research Plan and Reviewer Dispatch

**Research plan presented to the practitioner and approved 2026-04-30:**

- **Phase 1.** Combined lit-reviewer dispatch covering RQ-1 through RQ-5 as scoped sub-questions. Output to `docs/agentic-serving/essays/research-logs/lit-review-cycle-2-multi-turn-and-composition.md`. Lit-review is the primary method given the practitioner's "I want to follow the research here" posture from Step 1.2.
- **Phase 2.** Synthesize lit-review findings into the research log; identify what the literature reaches and what it does not.
- **Phase 3 (likely, scope TBD post-lit-review).** Spikes for empirical anchoring. Top candidates: an RQ-4 spike on a lightweight multi-orchestrator probe under hybrid deployment; an RQ-2 spike comparing compose-at-runtime vs select-from-menu vs prompt-steering on a representative task class.
- **Phase 4.** Iterate or proceed to essay 003 with mandatory citation audit followed by mandatory argument audit (with framing audit).

**Reviewer dispatch:**

The research-methods-reviewer specialist subagent was dispatched against the question set + constraint-removal response, with essay 002 and the cycle-1 archive feed-forward as prior research context. Output written to `docs/agentic-serving/housekeeping/audits/research-design-review-cycle-2.md`.

## Step 1.4 — Reviewer Findings and Question Set Finalization

**Audit summary:** Verdict — *conditionally ready*. Five RQs are well-formed, need-framed, and exhibit appropriate prior-art posture toward ADR-011 (constraint-removal exchange genuinely tests it rather than just brackets it for reframing). Four flags: 1 × P1, 2 × P2, 1 × P3; three coverage gaps (1 × P1 — same as the P1 flag, restated; 2 × P3).

**Resolutions (decided 2026-04-30):**

| Flag / Gap | Severity | Resolution |
|------------|----------|-----------|
| Missing prompt-steering baseline question (Incongruity A) | P1 | **Adopted** as RQ-2a (see RQ-2 above). Empirical spike battery committed to include at least one prompt-steering comparison arm against the best-performing compositional shape from the lit-review. |
| RQ-3 benefits-side gap | P2 | **Adopted** as RQ-3c (see RQ-3 above). The benefits-side sub-question is in scope of the lit-review dispatch — must be addressed alongside failure-mode/mitigation survey. |
| RQ-4 frontier presupposition | P2 | **Held loosely as interpretation posture.** Question text unchanged; if empirical results show one configuration dominates all axes, that is recorded as a "no frontier, single winner" finding rather than forced into a Pareto framing. |
| RQ-2 comparison class | P3 | **Resolved by RQ-2a adoption.** Prompt-steering is now an explicit part of the comparison class. |
| ADR-011 decision criterion | P3 | **Adopted as meta-question** (see §Meta-Questions below). |
| Multi-turn vs single-ask failure-mode continuity | P3 | **Adopted** as RQ-1a (see RQ-1 above). |

**Scope note for RQ-4 spike planning (from auditor's review of constraint-removal response):**

The practitioner acknowledged an embedded conclusion in the constraint-removal response: *"a single orchestrator run locally is rather slow on my hardware, so multiple models would be slower"* assumes the same hardware bottleneck profile for multi-orchestrator configurations as for single-orchestrator. A swarm of very small models (e.g., many qwen3:0.6b workers) may produce a different latency profile than N copies of qwen3:8b, because per-token latency scales non-linearly with model size on consumer hardware. The hybrid pivot (cloud orchestrator + local ensembles) recovers this for the cloud-orchestrator case, but **RQ-4's empirical frontier should include small-model swarm configurations on local hardware in its shape × deployment matrix, not just single-cloud-orchestrator + local-ensembles shapes.** Recorded here so Phase 3 spike planning carries the constraint forward.

## Meta-Questions

**MQ-1 (added 2026-04-30 from research-design-review §Coverage Gap 2):** What would it take for the literature or empirical findings of this cycle to justify amending or revoking ADR-011 (the single-Model-Profile orchestrator commitment)? This meta-question is a synthesis-time decision criterion, not an RQ to research directly. The practitioner's "follow the research" posture from Step 1.2 implies that the cycle's findings should feed into an ADR-011 review at synthesis. Naming the criterion explicitly here makes the decision visible at research entry rather than deferring it to an unnamed moment after the essay is drafted.

## Step 1.5 — Research Loop Begins

Question set finalized 2026-04-30. The agent now reads the existing artifact corpus (essay 002, cycle-1 archive, system-design.md + system-design.agents.md, scenarios.md, ADRs 001-011, product-discovery.md, field-notes.md) and proceeds to Step 2 (Research) per the approved plan.

---

# Research Loop

## Loop Iteration 1 — Combined Literature Review

**Method:** lit-reviewer specialist subagent dispatch.
**Sub-scopes:** RQ-1 + RQ-1a (multi-turn agentic dynamics + failure-mode continuity), RQ-2 + RQ-2a (composition threshold + prompt-steering baseline), RQ-3 + RQ-3c (shape inventory + failure modes + mitigations + benefits), RQ-4 (published frontier benchmarks), RQ-5 (conductor experience UX literature).
**Output:** `docs/agentic-serving/essays/research-logs/lit-review-cycle-2-multi-turn-and-composition.md` (28 cited sources; structured per-RQ).

### Findings summary

Findings synthesized per RQ. Full lit-review at the path above; what follows is what bears most directly on each question for downstream phases and for the next research-loop decision.

**RQ-1 (multi-turn dynamics).** Long-horizon performance degrades super-linearly. HORIZON benchmark (Wang et al., arXiv:2604.11978, April 2026) tested across 3,100+ trajectories: strong on short/mid-horizon, breaks down on long-horizon ("long-horizon task mirage"). Khanal et al. (arXiv:2603.29231) formalize four reliability metrics; counterintuitive finding: frontier models show *higher* variance at long horizons because ambitious plans "spiral." Meltdown rates up to 19% even in frontier. AMA-Bench (arXiv:2602.22769, ICLR 2026): similarity-based memory retrieval loses causally necessary information; SOTA memory achieves 57.22% on real agentic trajectories. LongCLI-Bench (arXiv:2602.14337): on CLI-agentic coding, SOTA agents pass <20% of long-horizon tasks; most stall at <30% completion; failures predominantly early. Anthropic context-engineering: middle-of-context content under-weighted — for llm-orc, *tool dispatch results from earlier turns are precisely the mid-context state at risk*.

**RQ-1a (failure-mode continuity).** Single-ask failure modes persist and compound; new multi-turn-specific modes appear. Roig (arXiv:2512.07497) four-archetype taxonomy: premature action, over-helpfulness, distractor-induced context pollution, fragile execution under load — the first and third match essay 002's premature-stop and argument-confabulation, both compound across turns. Distinctly multi-turn: meltdown onset, error self-conditioning (requires multiple turns to emerge), memory retrieval drift over extended horizons.

**RQ-2 (composition threshold).** No single algebraic threshold. Anthropic (engineering blog, June 2025) gives the cleanest production-derived conditions: composition earns its cost when (1) value justifies 15x token overhead, (2) genuine parallelization opportunities exist, (3) information scope exceeds single context window. **For tasks fitting in one context window without parallelism — most agentic coding queries — Anthropic's guidance is single agent.** MASS (Zhou et al., arXiv:2502.02533, Google/Cambridge): on Gemini 1.5 Pro, prompt optimization adds ~6 points over default single-agent CoT; correct topology adds ~5 more. Most topology choices neutral or harmful; only a small fraction beneficial. Uninformed topology selection produces no improvement.

**RQ-2a (prompt-steering baseline).** Essay 002's CAP-2 finding generalizes directionally. Anthropic explicitly: prompt steering is "the single most important way to guide agent behavior" *even in their multi-agent system*. Engineering analysis (Iterathon 2026): single agent with better prompting achieves 92% performance at 28% cost vs three-agent workflow. Where the comparison breaks: tasks requiring breadth-first parallel exploration across sources collectively exceeding single context window — for those, structural composition provides what prompt steering cannot. Whether Cycle 2's multi-turn sessions approach this regime is the open empirical question.

**RQ-3 shape inventory.** Single orchestrator: lowest overhead, recommended baseline. Hierarchical supervisor + workers: earns cost for genuinely parallelizable work. **Swarm-of-small-models (LLM-mediated): not viable at non-trivial scale — Rahman & Schranz (arXiv:2506.14496): 36,000x latency penalty vs classical algorithms for equivalent coordination.** Recommended hybrid: LLMs for strategic reasoning, classical algorithms for low-level coordination. Semantic-routed (ZOOTER, NAACL 2024): well-supported; *llm-orc's `invoke_ensemble` already implements a form of this*. **Bio-inspired in eusocial/ant-colony/naked-mole-rat terms: NOT FOUND as LLM architecture literature**; closest analogues are stigmergy (SwarmSys, arXiv:2510.10047) and ACO (classical algorithm, slow when LLM-mediated). MoA at orchestration layer (Wang et al., arXiv:2406.04692): 65.1% AlpacaEval 2.0 — at frontier tier; not validated at qwen3:8b. Multi-agent debate: conditions-dependent.

**RQ-3 failure modes (practitioner's prior recall verified).** Li et al. ICLR 2026 (OpenReview mo7u21GoQv): structured workflows amplify stochastic biases into systemic polarization even when individual agents are neutral; *"Trigger Vulnerability" — injecting objective context ACCELERATES polarization*. Madigan et al. (arXiv:2512.16433): emergent collective bias not attributable to individual components. Wynn et al. (ICML MAS Workshop 2025, arXiv:2509.05396): debate can DECREASE accuracy via sycophancy even when stronger models outnumber weaker ones.

**RQ-3 mitigations.** Five strategies, varying evidence quality: (1) *Anonymization of debate identity labels — strongest evidence, no retraining required* (Choi et al., ACL 2026 Main, arXiv:2510.07517); (2) anti-conformity prompting + trajectory-based scoring (FREE-MAD, arXiv:2509.11035, ~50% token reduction); (3) reasoning-tree auditing with minority-correct training (AgentAuditor, arXiv:2602.09341, +5% over majority vote); (4) heterogeneous model diversity across families; (5) judge/role-asymmetry architectures. **None fully eliminates the echo-chamber failure mode.**

**RQ-3c (benefits side).** Real reproducible improvements under specific conditions: breadth-first parallel research (Anthropic 90.2% improvement at frontier tier, 15x token cost); tasks exceeding context-window limits; tasks with domain-specialist models available. OPTIMA on Llama 3 8B (Chen et al., ACL 2025 Findings): 2.8x performance gain at <10% tokens — but requires OPTIMA fine-tuning and baseline is unoptimized multi-agent, not single-agent. **Critical capability-tier finding: all strong multi-agent benefit evidence uses frontier models. No published paper validates multi-agent composition benefits at qwen3:8b tier without fine-tuning. This is the gap Cycle 2 is positioned to fill empirically.**

**RQ-4 (frontier).** No single Pareto front because most benchmarks don't measure all three axes. Point estimates: LangGraph ~5% token overhead, ~1.2s latency for 10-step pipelines (2026 community benchmark, lower than essay 002's 30% citation — methodology divergence noted). AutoGen: 20+ LLM calls per task. Local-only qwen3:8b ~6 min, hybrid CAP-9 ~62s (essay 002). Qwen3-0.6B/1.7B/4B: 5-8 tok/s CPU, 40-60+ tok/s GPU with speculative decoding — but per-turn 15K-token contexts produce 300-500s time-to-first-token regardless of model size. *The local-only configuration's binding constraint is token-throughput on consumer CPUs; smaller models alone don't fix it.*

**RQ-5 (conductor experience).** Peer-reviewed HCI sparse. Recommended <4s for conversational AI (arXiv:2507.22352); moderate latency acceptable if contextualized with progress signals (Tandfonline 2025). Essay 002's 6-min local-only is ~180x chatbot standard; CAP-9's ~62s is ~30x. Industry practice: Devin 2.0 fork/rollback + async handoff + confidence-based clarification; Claude Code's seven graduated autonomy modes + JSONL session persistence. Anthropic context-engineering: "progressive disclosure" as core observability principle.

**ADR-011 status (relevant to MQ-1).** Literature does NOT make ADR-011 obviously wrong. Single-Model-Profile commitment is defensible at qwen3:8b tier on most task classes. MASS shows ~5pp topology delta over prompt optimization at Gemini 1.5 Pro tier — open question whether this materializes at qwen3:8b in multi-turn. ADR-011 should remain in force pending empirical evidence; the cycle's spikes (if run) are positioned to test whether multi-turn or capability-tier shifts change the calculus.

### Implications for spike planning (Step 4c / ADR-087 territory)

The lit-review opens four genuinely empirical questions the literature does not reach:

1. **MASS-equivalent at qwen3:8b multi-turn.** ~5pp topology gain over prompt-steering at Gemini 1.5 Pro is the published delta. Does it materialize at qwen3:8b on multi-turn coding sessions, or does the model-tier gap make structural composition wash out? This is the central empirical question the lit-review opened. RQ-2a + RQ-3c.
2. **Long-horizon error accumulation in llm-orc specifically.** HORIZON and LongCLI-Bench measure isolated long tasks, not multi-turn user-driven sessions. Whether llm-orc's serving + ReAct loop + summarization architecture inherits the super-linear degradation curve, exhibits a different curve, or partially mitigates it via the summarization path is empirically open. RQ-1.
3. **Small-model-swarm latency profile on consumer hardware.** Rahman & Schranz showed LLM-mediated swarm coordination is 36,000x slower than classical for equivalent coordination — but their setup differs from "many qwen3:0.6b workers under one cloud orchestrator." Whether the practitioner-named shape produces interesting tradeoffs or hits the same coordination wall is empirically open. RQ-4.
4. **Bias amplification in llm-orc's actual coordination protocol.** Li et al. and Wynn et al. studied multi-agent debate; llm-orc's coordination is not debate but supervisor-routing + cascading tool dispatch. Whether the trigger-vulnerability + echo-chamber findings translate to the actual coordination shape is empirically open. RQ-3 + RQ-3c.

Of these, #1 and #3 are the cleanest spike candidates: focused questions, scratch-directory scope, clear pass/fail signals. #2 and #4 are larger and might better belong to BUILD/PLAY phases of this cycle than to research spikes.

### Framing realignment (2026-04-30, mid-loop exchange)

A mid-loop exchange surfaced that the lit-review synthesis as initially presented to the practitioner overweighted performance-axis findings and underweighted the practitioner's actual optimization function. Practitioner verbatim: *"my motivation [goes] beyond performance, mixing environmental concerns, local-first preferences, and token costs. But we have to be realistic about the tradeoffs."*

The realignment is consequential for the essay's framing. Through a four-axis optimization function (performance × environmental cost × local-first preference × token cost), several literature findings read differently than through performance alone:

- Rahman & Schranz's 36,000× swarm-coordination penalty is for LLMs replacing classical coordination algorithms; a parallel-specialist shape (different small-model workers with no inter-agent coordination) is structurally different.
- Anthropic's 15× token overhead is environmentally and financially catastrophic at frontier-tier cloud-billed inference but has the inverse profile for local small-model specialists on owned hardware (zero token cost; energy amortized against general-purpose machine ownership).
- The capability-tier gap (no published paper validates multi-agent benefits at qwen3:8b without fine-tuning) is the empirical territory where local-first deployments live — the unmeasured space is where the cycle's investigation is positioned.
- Bio-inspired sparseness in eusocial / ant-colony / naked-mole-rat terms is real; stigmergy and parallel role-specialization are the closest published analogues that map cleanly onto local-first deployments.

The framing realignment is the essay's central contribution: the literature is mostly silent on the configuration space the practitioner cares about because most published work optimizes for performance alone at frontier tier with cloud-billed inference. The cycle's open empirical questions are correctly framed as "what does parallel-specialist multi-agent on local-first hardware look like" rather than "is multi-agent worth it." Both spike candidates (#1 MASS-equivalent at qwen3:8b multi-turn; #3 small-model-swarm latency on consumer hardware) directly anchor that framing.

Practitioner explicitly endorsed essay-first-then-spikes for this cycle, framing potential spike refutation of essay claims as research-positive: *"a refutation of the essay from spikes would be actually good from a research point of view and would certainly warrant a reframe."*

### Loop Iteration 1 — closed

Loop Iteration 1 closes here. Lit-review delivered substantive coverage across all five RQs + sub-questions with 28 cited sources. Framing realignment captured. The cycle proceeds to Step 4 (essay draft) with the four design priorities as the essay's central frame; spike work follows per ADR-087 at Step 4c.

## Step 4 — Essay 003 + Audits

**Essay:** `docs/agentic-serving/essays/003-multi-turn-orchestration-and-the-four-axis-frame.md` (drafted 2026-04-30).

**Citation audit:** `docs/agentic-serving/housekeeping/audits/citation-audit-cycle-2.md` — 7 findings (2 P1, 3 P2, 2 P3). Issues: wrong tool names (corrected to ADR-003 canonical: `invoke_ensemble`, `compose_ensemble`, `list_ensembles`, `query_knowledge`, `record_outcome`); unsupported Devin 2.0 fork/rollback claim (removed); Lost-in-the-Middle citation correctly attributed to Liu et al. 2024 rather than Anthropic blog; progressive-disclosure analogy explicitly framed as essay synthesis rather than Anthropic attribution; Qwen3 CPU rates correctly attributed to community engineering analysis rather than Technical Report; ADR-082 RDD-namespace clarified; Iterathon finding scope-qualified to frontier-tier customer-support task. All P1 and P2 issues addressed; P3 items also addressed.

**Argument audit (round 1):** `docs/agentic-serving/housekeeping/audits/argument-audit-cycle-2.md` — 10 argument-audit findings (2 P1, 5 P2, 3 P3) + 5 framing-audit findings (2 P1, 2 P2, 1 P3). Argument-audit findings addressed directly: four-axis "optimization function" reframed as "design priorities" with explicit non-commensurability disclaimer; verified-mitigations qualified to debate-shape-specific scope with non-debate translation flagged as empirically open; community-judgment alternative reading surfaced in conclusion; opportunity-vs-limitation conditionality named at point of claim; LangGraph 30%-vs-5% discrepancy implication for essay 002 surfaced; spike specifications expanded with task class, baseline, topology, and threshold parameters; consequential-contribution claim distinguished cycle-internal from field-level; parallel-specialist/aggregation-overhead distinction clarified; OPTIMA fine-tuning caveat strengthened; progressive-disclosure disclaimer placement corrected.

**Argument audit (round 2):** `docs/agentic-serving/housekeeping/audits/argument-audit-cycle-2-round2.md` — 3 new findings (2 P2, 1 P3): commensurability/comparative-language inconsistency (corrected with directional language); spike-claim walked back caveat (corrected with "once their parameters are committed" conditionality); causal assertion vs acknowledged uncertainty (softened to "consistent with a structural cause" + symmetric-readings framing).

**Argument audit (round 3):** `docs/agentic-serving/housekeeping/audits/argument-audit-cycle-2-round3.md` — 1 P3 finding: spike-conditionality not propagated to conclusion's parallel claim (corrected).

**Argument audit (round 4):** `docs/agentic-serving/housekeeping/audits/argument-audit-cycle-2-round4.md` — **clear**. Zero open issues at any severity. Essay is ready for Step 4c and the epistemic gate.

**Framing-audit findings deferred to gate (per Step 4b — not auto-corrected):**
- Framing P1-1 (community-judgment alternative reading) — addressed in conclusion via P2-1 argument-audit fix; surfaced in essay text.
- Framing P1-2 (tau-bench finding omission) — tau-bench provides the most directly relevant published baseline for tool-calling reliability ceilings on llm-orc's architecture; not currently in the essay. Decision deferred to gate: integrate or accept omission with rationale.
- Framing P2-1 (framing-realignment-as-correction treatment) — the essay's adoption of the four-priorities frame after the practitioner's mid-loop pushback is treated as a correction rather than as adoption of one valid alternative framing among others. Decision deferred to gate: revise treatment or accept with rationale.
- Framing P2-2 (trigger-vulnerability cascade-applicability) — addressed in §"Failure modes verified" via P1-2 argument-audit fix; surfaced in essay text.
- Framing P3-1 (OPTIMA/MASS not equivalent) — addressed in §"The Capability-Tier Gap" via P3-3 argument-audit fix.
- Framing P2-R1 (round 2; falsifiable evidence concentrated on performance axis) — three of four design priorities have no falsifiable spike evidence; configuration choices on the three non-performance priorities will be judgment calls regardless of spike outcomes. Decision deferred to gate: name explicitly in essay or accept implicit.
- Framing P3-R1 (round 2; "condition of operation" ambiguity — before-investigating vs before-deploying) — minor framing inconsistency. Decision deferred to gate.

The two consequential framing-audit decisions for the gate are P1-2 (tau-bench) and P2-1 (realignment-as-correction). The others are addressed or minor.

## Step 4c — Validation-Spike Decision

**Decision (2026-04-30):** Run validation spikes per ADR-087, iterating in research context (B then A then possible follow-ups) before advancing to the gate. Practitioner rationale: agent and ensemble design at this stage is mostly configuration rather than code, so the cost difference between research-spike and BUILD-spike is small while the epistemic difference is real (essay claims are still revisable; ADRs in DECIDE will harden them). Sequential B→A chosen because Spike A's design depends on Spike B's viability finding. Iteration accepted in principle if findings warrant.

Rationale recorded; spikes follow as Loop Iteration 2 onward.

---

# Validation Spikes

## Loop Iteration 2 — Spike B (Parallel-Specialist Latency)

**Date:** 2026-04-29

### Question

Does running the existing llm-orc code-review ensemble with reviewer agents reconfigured to Qwen3-0.6B, under a cloud orchestrator (MiniMax M2.5 Free via OpenCode Zen), produce time-to-first-orchestrator-output under 10 seconds on this hardware, with output quality at least comparable to CAP-3b's baseline?

### Method

Built a copy of the production `code-review` ensemble at `.llm-orc/ensembles/development/code-review-spike-b.yaml`. The three reviewer slots (security-auditor, performance-engineer, senior-reviewer) were reconfigured to a temporary `spike-b-reviewer-qwen06` profile (Ollama provider, model `qwen3:0.6b`, timeout 120s) added to `~/.config/llm-orc/config.yaml`. The synthesizer (tech-lead) was left at its production config (Anthropic API, claude-sonnet-4-20250514). Original system prompts preserved per slot.

The orchestrator was kept at the cycle-1 CAP-9 baseline: `orchestrator-minimax-m25-free` (MiniMax M2.5 Free via OpenCode Zen, OpenAI-compatible at `https://opencode.ai/zen/v1`). The active `agentic_serving.orchestrator.model_profile` setting was unchanged. The pre-existing `llm-orc serve --port 8765` instance picked up the new profile and ensemble on next call (config-manager mtime-based cache invalidation; ensembles re-discovered per `list_ensembles` invocation).

Two trials were run from the project working directory via `opencode run --format json -m llm-orc/orchestrator-minimax-m25-free` against the prompt: *"What ensembles are available in this llm-orc instance? Pick the 'code-review-spike-b' ensemble and demonstrate calling it on the README of this project."* This matches the ask shape used in CAP-3b/CAP-9 (orchestrator discovers via `list_ensembles`, OpenCode reads the README via its `read` client tool, orchestrator invokes the ensemble, summarizer collapses results, orchestrator synthesizes).

### Trials

**Trial 1 (warm-up):** total wall-clock 62.5s. OpenCode `read` of README at +6.19s. First orchestrator text emission at +61.04s. Step finish (final stop) at +61.12s. Cascade engaged correctly: `list_ensembles` → `invoke_ensemble("code-review-spike-b", input=README content)` → ensemble runs three Qwen3-0.6B reviewers in parallel + Claude tech-lead synthesizer → summarizer (qwen3:0.6b) collapses → orchestrator synthesizes final response.

**Trial 2 (stable):** total wall-clock 58.7s. OpenCode `read` at +6.92s. First orchestrator text emission at +56.64s. Step finish at +56.73s. Same cascade trajectory as Trial 1.

### Measurements (Trial 2 primary)

1. **Time-to-first-orchestrator-output: ~56.6 seconds.** Measured as the timestamp of the first `text` event emitted by the orchestrator on the SSE channel, relative to the first `step_start`. MiniMax went directly to tool calls without emitting a preamble text token, so the first text emission lands only after `invoke_ensemble` returns. **Fails the < 10s pass criterion by a wide margin.**

2. **Total wall-clock:** 58.7s (Trial 2), 62.5s (Trial 1). Effectively unchanged from the CAP-9 baseline (~62s) despite reviewers being substituted from llama3 (4.7 GB) to qwen3:0.6b (522 MB). The dominant time is not the reviewer-token-throughput dimension but the cascade plumbing: README read by OpenCode (~7s), three concurrent ensemble agents at qwen3:0.6b (parallel-bounded), Claude tech-lead synthesizer call, summarizer call (qwen3:0.6b), orchestrator round-trip.

3. **Output quality: degraded (Mixed outcome).** The orchestrator's final text identified the ensemble's purpose correctly, returned a recognizable review structure (security/performance/quality dimensions in a table), and named no fabricated content. But the specificity dropped substantively versus CAP-3b's qwen3:8b-orchestrator local-only baseline, which produced concrete recommendations like *"Add notes about rotating API keys periodically"*. Spike B's output reduced to category-level abstractions: *"Authentication is encrypted; YAML configs need review for sensitive data exposure"*, *"Multi-agent orchestration with dependency management supports scalability"*. The pipeline preserved structure but lost the per-reviewer specificity through the summarizer + orchestrator-synthesis chain.

4. **Concurrent worker count:** Ollama held a single qwen3:0.6b instance loaded (`ollama ps` confirmed, `size_vram=4.4 GB` on GPU). The llm-orc concurrency cap was set to `max_limit: 3` (project config). Whether Ollama serviced the three reviewer requests in true parallel (default `OLLAMA_NUM_PARALLEL` permits up to 4 concurrent slots when memory allows) or serialized them via the in-process queue is not directly observable from the operator-side surface available during the spike. The 58s total wall-clock is consistent with parallel execution: three serialized qwen3:0.6b reviewer calls processing ~30K-token input would individually exceed the entire budget at consumer-CPU rates.

### Pass/fail interpretation

**Mixed outcome.** TTFO is not under 10s — it is ~57s, dominated by ensemble execution time on the local stack rather than orchestrator cost. The < 10s threshold was set against a hypothesis that small-model substitution at the reviewer slots would compress total ensemble execution dramatically; it did not. Total wall-clock (58.7s) is workable for asynchronous tasks and within interactive-acceptable territory at the upper bound. Output quality is degraded but recognizably substantive — a real review structure with category-correct content, but without the per-recommendation specificity CAP-3b achieved.

### Implications for essay 003

The spike refutes the implicit hypothesis that **reviewer-tier capability is the binding constraint on parallel-specialist ensemble latency at this hardware**. It is not. Substituting llama3 (8B) reviewers for qwen3:0.6b reviewers — a roughly 9× reduction in disk size (4.7 GB → 522 MB), or equivalently ~13× reduction in parameter count (8B → 0.6B) — left total wall-clock essentially unchanged (~62s CAP-9 → ~58.7s Spike B). The dominant time is the cascade plumbing (client-tool read, summarizer, orchestrator round-trip, synthesis call) and the Claude tech-lead synthesizer call, not the parallel reviewers. The bottleneck is structural to the four-layer cascade, not the reviewer model tier.

The corollary is consequential for the essay's parallel-specialist claim: **the four design priorities (performance × environmental × local-first × token cost) are not co-optimized by simply shrinking the reviewer models.** Smaller reviewers buy environmental + local-first wins (smaller VRAM, fully on-device for the reviewer step) but they do not buy interactive-latency improvements at this cascade depth, and they cost output specificity. Operators selecting parallel-specialist topology for latency-sensitive workloads should look at orchestrator-side or summarizer-side optimization first, or move to architectures that emit per-reviewer streams to the user surface instead of collapsing them through a summarizer.

The spike's most informative finding is the quality-cost ratio: at zero environmental gain in total wall-clock and a measurable specificity loss in output, qwen3:0.6b reviewers underprice their structural cost in this configuration. The essay's parallel-specialist viability claim should be qualified to "viable for non-interactive workloads where output is structurally recognizable but per-recommendation specificity is not the success criterion" — not the more general "viable on consumer hardware" the unqualified framing implies.

### Cleanup

Project ensemble (`.llm-orc/ensembles/development/code-review-spike-b.yaml`) and the `spike-b-reviewer-qwen06` profile in `~/.config/llm-orc/config.yaml` were both removed after findings were recorded. Scratch directory `scratch/spike-b-parallel-specialist-latency/` deleted per spike rules. No git commits made.

## Loop Iteration 3 — Spike A (Cascade vs. Prompt-Steering)

**Date:** 2026-04-29

### Question

For the same task (code review of the project README), does the existing llm-orc ensemble cascade's quality justify its latency premium over a strong prompt-steered single cloud orchestrator with no cascade?

The spike's design pivots on Spike B's finding that cascade plumbing — not reviewer-tier compute — dominates total wall-clock. Originally framed as a topology-delta probe at qwen3:8b multi-turn; refocused after Spike B onto the cascade-vs-no-cascade comparison at the same cloud orchestrator (MiniMax M2.5 Free via OpenCode Zen).

### Method

**A1 — Cascade arm.** Cloud orchestrator (`orchestrator-minimax-m25-free`) drives the production `code-review` ensemble (CAP-9 baseline restored: 2 × llama3 reviewers — security-auditor, performance-engineer — + 1 × Claude senior-reviewer + 1 × Claude tech-lead-synthesizer). The cycle's standard invocation pattern: `opencode run --format json -m llm-orc/orchestrator-minimax-m25-free` against the prompt *"What ensembles are available in this llm-orc instance? Pick the 'code-review' ensemble and demonstrate calling it on the README of this project."* Two trials run via OpenCode (the CAP-9-shape path), and one trial run directly against `http://127.0.0.1:8765/v1/chat/completions` with the README inlined ("A1-clean") so OpenCode's own `glob`/`read` tool surface does not coexist with llm-orc's `list_ensembles` / `invoke_ensemble`.

The A1-clean variant was added mid-spike on observation that OpenCode's mixed tool surface caused the orchestrator to use OpenCode's `glob`+`read` tools rather than llm-orc's `list_ensembles`+`invoke_ensemble`. (Server-side artifacts confirm the cascade ran in all three trials regardless — the client just doesn't see internal tool calls in the SSE stream.)

**A2 — Prompt-steered arm.** Direct curl to OpenCode Zen's OpenAI-compatible endpoint (`https://opencode.ai/zen/v1/chat/completions`) with model `minimax-m2.5-free`, no llm-orc serve in the path, no tools advertised, README inlined in the user message. System prompt asked for review across security / performance / quality with at least three specific recommendations per dimension. Three trials.

### Trials

**A1 (via opencode, mixed tools):**

- Trial 1: total wall-clock 71.4s; OpenCode `glob` at +4.7s, `read` README at +11.2s, first orchestrator text at +71.3s; ensemble execution recorded in artifact `20260430-224240-521`: 4 agents, ensemble duration 43.7s.
- Trial 2: total 86.4s; `glob` at +4.2s, `read` at +11.8s, first orchestrator text at +86.4s; artifact `20260430-224524-168`: 4 agents, ensemble duration 51.8s.

**A1-clean (direct llm-orc serve, README inlined):**

- Trial 1: total 145.5s; one SSE chunk emitted at +145.5s containing the entire orchestrator synthesis; artifact `20260430-224844-609`: 4 agents, ensemble duration 104.1s. Orchestrator overhead beyond the ensemble: ~41s (initial Zen call setup + post-cascade synthesis call to Zen).

**A2 (prompt-steered, direct Zen):**

- Trial 1: TTFO 6.4s; total 15.7s; 11,095 chars output.
- Trial 2: TTFO 22.5s; total 70.3s; 10,598 chars output. (Slow run, attributable to Zen free-tier scheduling variance.)
- Trial 3: TTFO 10.1s; total 19.5s; 6,784 chars output.

A2 median TTFO ~10s, median total ~19.5s. The variance band is wide (15–70s total), reflecting Zen free-tier scheduling rather than model behavior — the actual generation rate, when scheduled, is ~600–700 chars/sec.

### Measurements

| Arm | Median TTFO | Median total wall-clock | Output chars | Distinct numbered recommendations | Cascade engaged |
|-----|-------------|------------------------|--------------|-----------------------------------|-----------------|
| A1 (via opencode) | 71–86s | 71–86s | ~1700–2200 | ~9 (in tech-lead synthesis) collapsed to ~5–7 in orchestrator final | Yes (4-agent ensemble, 43.7–51.8s) |
| A1-clean (direct) | 145s | 145s | 4,681 | 9 (~3 per dimension) | Yes (4-agent ensemble, 104.1s) |
| A2 (prompt-steered) | 10s | 19.5s | 6,784–11,095 | 16 (5+5+6 per dimension) | None |

The TTFO advantage of A2 over A1 is structural: A1 emits no preamble text, so first-text-token only lands after the cascade completes and the orchestrator generates synthesis. A2 streams generation start within Zen's first-token budget for that endpoint.

### Output quality (verbatim samples)

**A1-clean orchestrator final** (after the cascade's tech-lead-synthesizer collapsed the four reviewers, the orchestrator synthesizes again over Zen). Security section excerpt:

> ⚠️ Recommendations
>
> | Finding | Severity | Description |
> |---------|----------|-------------|
> | YAML config injection | Medium | User-provided YAML configs (`--config-dir`) are parsed without sandboxing. Ensure configs come from trusted sources only. |
> | Script execution | Medium | Custom script agents execute arbitrary code with user permissions. Restrict script directories to project-local `.llm-orc/scripts/`. |
> | API key in shell history | Low | CLI commands like `llm-orc auth add anthropic --api-key KEY` may persist in shell history. Recommend using env vars or interactive prompts instead. |

Three recommendations, each named with a finding + severity + remediation. Concrete enough to act on. The cascade's tech-lead-synthesizer (model: claude-sonnet-4) had emitted longer prose with more recommendations; the orchestrator's post-cascade synthesis tightened those to a smaller set with table form.

**A2 prompt-steered** (direct Zen, no cascade). Security section excerpt (recommendations 1 and 2 of 5):

> ### 1. Encryption Mechanism Is Not Explained
>
> The README states that "API keys are encrypted and stored securely" in `~/.config/llm-orc/credentials.yaml`, but provides no details on how encryption works.
>
> **What's missing**: The encryption algorithm, key derivation method (if any), whether a master password is required, or what the security properties are. A reader cannot assess the actual security of their credentials.
>
> **Recommendation**: Add a security internals section explaining the encryption approach—for example: "API keys are encrypted using AES-256-GCM with a key derived from your system's secure enclave/keyring" or similar. If no master password is required, explicitly state this and explain why.
>
> ### 2. No Guidance on File Permissions
>
> Credentials are stored in `~/.config/llm-orc/credentials.yaml`, but the README does not mention file permissions. On typical Unix systems, files in home directories are readable by the user's account but not protected against other users or world readable in some configurations.
>
> **What's missing**: Guidance to run `chmod 600 ~/.config/llm-orc/credentials.yaml` or equivalent for credential file protection.
>
> **Recommendation**: Add a security hardening step in the authentication section: "Ensure your credentials file has restricted permissions: `chmod 600 ~/.config/llm-orc/credentials.yaml`"

A2's per-recommendation depth (each split into "what's missing" + "recommendation" with concrete config examples like `chmod 600` or named encryption algorithms) is materially deeper than A1's table-form severity-ranked entries. A2 also produces 16 numbered recommendations to A1's 9, distributed across the same three dimensions.

No fabrication detected in either arm against the README content. Both arms correctly grounded recommendations in observable README claims.

### Quality:latency comparison

A1 invests 71–145s of wall-clock to surface 9 numbered recommendations through a four-agent cascade plus orchestrator post-synthesis. A2 invests 15–20s wall-clock (median) to surface 16 numbered recommendations directly from one capable model.

On two of the three sub-axes named at the spike opening:

- **Specificity per recommendation:** A2 wins. A2's "what's missing" + "recommendation" structure with concrete examples (`chmod 600`, "AES-256-GCM") consistently produces more actionable guidance than A1's table-row-style entries. A1's tech-lead-synthesizer had longer prose, but the orchestrator's post-cascade synthesis flattened it.
- **Dimension coverage:** Both cover security/performance/quality. Equivalent.
- **Depth (recommendation count):** A2 wins (16 vs. 9).
- **Fabrication:** Neither arm fabricated. Both grounded in README content.

### Pass/fail interpretation

The "A2 quality > A1 quality at lower latency" outcome materialized — explicitly named at spike opening as the unexpected outcome that would suggest the cascade is actively harmful for this task class. The data supports the named-as-unexpected reading: cascade summarization is flattening per-reviewer specificity. The cascade's tech-lead-synthesizer (Claude) does produce extensive review content, but the orchestrator's *second* synthesis pass on top of it (post-`invoke_ensemble`, before final response) compresses that content further than necessary, while a single capable orchestrator with the same input + a directive prompt produces a fuller, more specific result in under a fifth of the wall-clock.

This is structurally consistent with Spike B's finding (cascade plumbing dominates latency without buying a corresponding quality dividend at qwen3:0.6b reviewer tier). Spike A extends Spike B: the cascade plumbing also doesn't buy a quality dividend at the *production* reviewer tier (llama3 + Claude). The cascade is paying for two layers of summarization (tech-lead-synthesizer inside the ensemble, then orchestrator synthesis after `invoke_ensemble` returns) that compress signal more than they aggregate it for this task class.

### Implications for essay 003

Three implications, in descending epistemic load:

1. **Essay 002's CAP-2 finding generalizes beyond CAP-2's narrow scope.** Essay 002 §Findings rejected structural composition at qwen3:8b on a single-ask capability-query task class. Spike A demonstrates the same shape — single capable orchestrator with prompt steering matches or exceeds structural composition on quality, at a fraction of latency — at the cloud orchestrator tier on a code-review task class. The configuration-space generalization is now empirically supported across two task classes (capability query, code review) and two orchestrator tiers (qwen3:8b local, MiniMax cloud). This strengthens the essay's posture toward ADR-011 (single Model Profile remains defensible).

2. **The cascade's structural cost is not free even at zero token cost.** A1's local reviewers run on owned hardware (zero token cost); the cascade still costs 41–135s of orchestrator wait time the prompt-steered arm avoids. The four design priorities (performance × environmental × local-first × token cost) do not collectively pay for the cascade in this configuration: token cost is roughly equal (orchestrator does both synthesis passes against Zen either way), environmental cost is *higher* in A1 (additional local CPU+GPU spend on llama3 reviewers, plus one extra Claude API call for the tech-lead-synthesizer that A2 does not make), local-first preference is mixed (the local reviewers are local, but the synthesis path stays cloud), and performance is materially worse. The "parallel-specialist on local hardware" frame the essay holds open as configuration territory needs to demonstrate that *some* task class clears this bar, or the frame becomes hypothetical-only.

3. **The cascade may be actively harmful for tasks that value per-recommendation specificity.** The orchestrator's second synthesis pass over the ensemble's tech-lead synthesis is a lossy compression step. For task classes where the user wants the *granular reviewer voice* rather than a top-line synthesis, the cascade's collapse step is destroying the value the parallel reviewers added. This is a stronger claim than "cascade is not earning its complexity"; it is "cascade is removing value." Essay 003's claim about parallel-specialist topology should distinguish "tasks that benefit from collapsed top-line synthesis" (where A1 may add value over A2 in some untested regime) from "tasks that benefit from preserved per-reviewer specificity" (where A1 underperforms A2). The current essay does not make this distinction.

The empirical posture of essay 003 should incorporate this finding into §"What the spikes confirmed" (or equivalent): the cascade-vs-prompt-steering comparison at the cloud-orchestrator tier on the code-review task class supports prompt-steering as the simpler-and-better default, with structural composition justified only by task-class-specific evidence not yet collected.

### Cleanup

Scratch directory `scratch/spike-a-cascade-vs-prompt-steering/` deleted per spike rules. No project ensembles or profiles modified during this spike (production `code-review` ensemble was already restored to baseline before Spike A began). No git commits made.

## Loop Iteration 4 — Literature on Ensemble Design Principles

**Date:** 2026-04-30

**Method:** lit-reviewer specialist subagent dispatch.

**Why this loop.** Spike A revealed that the existing production code-review ensemble's two-stage summarization design is dominated by prompt-steering on every measured axis except dimension coverage. The finding refutes claims about *this ensemble*; it does not refute claims about *all ensemble designs* a well-architected process could produce. The cycle's premise (well-architected processes can achieve good results) requires testing a novel ensemble designed against current findings, not the production ensemble inherited from earlier cycles. That test cannot run cleanly without a literature foundation on ensemble design principles — what synthesizer patterns work, what role decompositions earn their place, what coordination protocols beyond parallel-with-collapse the literature prescribes. Loop Iteration 1's lit-review covered shape inventory + failure modes + benefits but did not go deep on design principles. L4 closes that gap.

**Output:** `docs/agentic-serving/essays/research-logs/lit-review-cycle-2-ensemble-design-principles.md`.

### Findings summary (L4)

The design-principles literature is richer than Loop 1's bio-inspired-architecture territory but is not a principled engineering corpus. Vocabulary and named patterns exist; empirical comparisons are limited and domain-specific (scientific review, judging, math — not code review); prescriptive synthesizer-selection guidance from task properties is absent beyond coarse mappings; the specific failure mode Spike A revealed (two-stage summarization destroys specificity) is not studied as an explicit research object anywhere found.

The most actionable findings for Loop 5's spike design:

**RQ-D1 (synthesizer patterns).** The headline finding is that **sequential panel discussion actively destroys diversity** — Yao et al. (2025) measured panel-discussion precision at 72.6% versus 77.3% for majority voting on the same task; judge opinions converge as agents see each other's outputs. This is the same mechanism as the production ensemble's two-stage summarization. **MARG** (Drozdov et al., 2024) is the closest specificity-preserving alternative the literature offers: three independent agent groups whose outputs are *concatenated* rather than synthesized. Generic-comment rate drops from 60% to 29% under MARG's no-synthesizer architecture. The mechanism isn't fully explained in the paper but the architectural choice — separate sections, no compression step — is the distinguishing factor.

**RQ-D2 (role decomposition).** **Heterogeneity is the most empirically grounded principle.** Sun et al. (2025) and Ding et al. (2024): heterogeneous agents (different models, prompts, tools) outperform homogeneous ensembles because their errors are uncorrelated. Homogeneous agents fall into a "popularity trap" — converge to the same wrong answer. Diversity-based selection from a two-model heterogeneous ensemble recovers up to 95% of theoretical performance ceiling. Know the Ropes (Chen et al., 2025) gives the strongest decomposition heuristic: derive agent structure from the algorithm, not generic organizational metaphors. MAST (Cemri et al., 2025) catalogues 14 multi-agent failure modes — most are execution failures (role boundary violations, dropped handoff context, ignored peer outputs), not architectural ones. Gap: no published principle distinguishes cognitive specialization (different reasoning approaches) from domain specialization (different knowledge areas); no published principle for when role asymmetry (judge above debaters) earns its complexity.

**RQ-D3 (coordination protocols).** Reflexion/Self-Refine (NeurIPS 2023): intrinsic self-correction is not reliably effective; external evaluation signals are required for the loop to provide net gains. Multi-agent Reflexion (MAR) addresses this at ~3× API cost. Debate outperforms majority voting on complex/ambiguous tasks (up to 6pp gain) but shows no advantage on simple tasks with strong initial consensus. Async coordination (AutoGen v0.4, AREAL-boba2): 2.77× speedup. Chain of Agents (NeurIPS 2024) for long-context tasks where inputs exceed a single context window — its final manager stage is still collapse-to-summary.

**RQ-D4 (task → architecture mapping).** Coarse mappings exist: long-context → sequential chain; breadth-first exploration → parallel orchestrator-workers; ambiguous evaluation → debate; verifiable ground truth → voting with diverse models; fragmented data → specialist routing. Anthropic's research-system 90.2% gain over single-agent Opus is on breadth-first research queries specifically. **The critical gap: no published taxonomy maps task properties to synthesizer-pattern choices.** The "given task property X, choose aggregation pattern Y" guidance is absent beyond the coarse architectural mappings above.

**Jiang et al. (April 2026 preprint)** provides theoretical grounding for Spike A's empirical finding: under matched reasoning-token budgets, single agents consistently match or exceed multi-agent performance on multi-hop reasoning. Multi-agent becomes competitive only when single-agent context utilization is degraded or additional compute is provided. This is consistent with Spike A and gives Spike A a theoretical anchor it didn't have post-spike.

**RQ-D5 (cascade overhead).** Optima (ACL Findings 2025): trained communication policies, 2.8× task gain at <10% token cost on information-asymmetric tasks. Helium (2026): proactive prefix caching for static prompts. Schema overhead is independently confirmed at the cycle's range — 15K+ tokens per turn in multi-MCP configurations, reducible ~53% via sandboxed-Python replacement of schema declarations. **Gap:** consumer-hardware cascade latency where plumbing dominates wall-clock regardless of inner-model size has no direct treatment in the literature.

**RQ-D6 (honest gap assessment).** The design-principles literature passes Loop 4's threshold for "rich enough to inform a principled novel design" but fails the threshold for "pre-validates that the novel design will outperform Spike A's prompt-steered baseline." The literature has not falsified the cycle's premise; it has also not pre-validated it. Whether a novel ensemble grounded in MARG-style concatenation, heterogeneous roles, external-signal refinement, and async coordination outperforms a prompt-steered single capable orchestrator on the cycle's task classes must be measured empirically. That is exactly what Loop 5 is for.

### Implications for Loop 5 spike design

The lit-review converges on a small set of named, empirically-motivated design choices for a novel ensemble that addresses Spike A's failure mechanism specifically:

1. **Drop the synthesizer collapse step.** Use concatenation with per-reviewer attribution (MARG-style) instead of collapse-to-summary. Directly addresses Spike A's two-stage-summarization specificity loss.
2. **Make reviewers heterogeneous.** Different model families for different roles, not three slots filled with the same model class. Heterogeneity finding (Sun et al., Ding et al.) suggests this is where ceiling-recovery comes from.
3. **Skip refinement loops unless external evaluation is available.** Reflexion's caveat: intrinsic self-correction is not reliably effective. The cycle's spike fixture does not provide ground-truth evaluation, so any iteration loop would be intrinsic and harmful.
4. **Optional: judge-arbitration when reviewers disagree.** Where reviewers reach conflicting recommendations on the same dimension, a judge model arbitrates. Where they agree, both pass through unchanged. This is the only synthesis-shaped step that the literature suggests preserves rather than flattens.
5. **Same task class as Spike A** — review the project README. Comparison against the already-measured A2 baseline (prompt-steered single cloud orchestrator, ~19.5s median, 16 specific recommendations).

The novel ensemble (call it A3) tests whether a design-principles-informed alternative can match or exceed A2 on quality dimensions where the existing production ensemble lost — particularly per-recommendation specificity and recommendation depth. If A3 ≥ A2 on specificity at acceptable latency, MQ-1's Branch One fires and ADR-011 is reconsidered at synthesis. If A3 < A2 on specificity, the cycle's premise is empirically refuted within the configuration space the design-principles literature reaches, and the open territory graduates to a future cycle on test-and-evaluate methodology for orchestration designs the literature does not yet describe.

## Loop Iteration 5 — Spike A3 (Novel Heterogeneous Ensemble with Script Grounding)

**Date:** 2026-04-30

### Question

Does a novel code-review ensemble combining a script-based deterministic analyzer with two heterogeneous cloud LLM reviewers (different model families, different dimensional focus), aggregated by MARG-style concatenation rather than collapse-summary, match or exceed A2 (prompt-steered single cloud orchestrator) on per-recommendation specificity for the project README review task — and additionally produce factual grounding (script-derived findings like "broken link at line X") that A2 structurally cannot produce?

### Method

**Composition.**

- **Script agent** (`scratch/spike-a3-novel-ensemble/readme_factcheck.py`, deleted post-spike): three deterministic checks on README.md.
  1. Link validity: regex-extract URLs, HTTP HEAD/GET each (timeout 8s, GET fallback for 403/405/501), report dead external links by line number; localhost/loopback URLs surfaced separately as documentation examples.
  2. Standard-section presence: heading-and-full-text regex against five canonical sections (installation, usage, license, contributing, examples/quickstart).
  3. Code-block parseability: fenced blocks with declared language parsed via `ast.parse` (Python), `yaml.safe_load` (YAML), `json.loads` (JSON); other languages skipped with note rather than failed.
- **Reviewer 1 — Accuracy and Completeness focus.** Tencent Hunyuan family (`hy3-preview-free` via OpenCode Zen). System prompt explicitly directs reviewer to use script's findings as factual anchors, name concrete changes, target ≥5 recommendations.
- **Reviewer 2 — Clarity and Onboarding-Friction focus.** Moonshot Kimi family (`kimi-k2.5` via OpenCode Zen). Same input shape as Reviewer 1 (README + script report), distinct dimensional focus, same recommendation-count target.
- **Orchestrator framing** (`minimax-m2.5-free`, the CAP-9 baseline orchestrator): a brief (3–5 sentence) introduction announcing the three sections that follow. Explicitly told NOT to summarize or synthesize the reviewer outputs.
- **Aggregator: MARG-style verbatim concatenation, no collapse synthesizer.** A Python harness (`run_a3.py`) runs the script agent, then dispatches the two reviewers in parallel via direct curl to OpenCode Zen, then runs the orchestrator framing call, then concatenates the four outputs with attributed section headers. There is no LLM agent that flattens the three substantive sections into a single voice.

Three genuinely different model families (MiniMax / Tencent Hunyuan / Moonshot Kimi) — chosen to maximize the heterogeneity finding's mechanism (uncorrelated errors). Free-tier paths used where possible (orchestrator + R1); R2 used a paid Zen model since no third heterogeneous *free* family was available on the day's Zen catalog.

**Why this composition rather than llm-orc serve + ensemble YAML.** Spike A's measurements showed the orchestrator-then-cascade-then-orchestrator-synthesis path was where collapse occurred — and the A1-clean direct-Zen invocation took 145s. To test the MARG-concatenation hypothesis cleanly, the harness was structured so concatenation is *enforced by code* rather than left as an instruction the orchestrator might ignore. This is a deliberate methodological choice: it tests the topology, not the orchestrator's compliance with topology instructions. (A llm-orc-native version would be straightforward to write — the cycle's existing Script Agent + ensemble YAML supports this composition — but would conflate the topology test with an instruction-compliance test.)

### Trials

All three trials successful. Wall-clock dominated by reviewer LLM latency; script agent ran in ~1s.

**Trial 1 (warm-up):**
- Script: 0.6s (1097 chars).
- Reviewer 1 (hy3): 65.3s (17,898 chars).
- Reviewer 2 (Kimi): 72.0s (4,018 chars).
- Reviewers wall (parallel): 72.0s.
- Orchestrator intro: 8.8s (389 chars).
- **Total: 81.3s** (23,702 chars final).

**Trial 2 (primary measurement):**
- Script: 1.6s.
- Reviewer 1: 54.4s (18,255 chars).
- Reviewer 2: 46.0s (3,503 chars).
- Reviewers wall: 54.4s.
- Orchestrator intro: 6.5s (508 chars).
- **Total: 62.5s** (23,663 chars final).

**Trial 3 (variance check, triggered by qualitative concern about Reviewer 1 chain-of-thought leakage):**
- Script: 1.1s.
- Reviewer 1: 42.9s (17,862 chars).
- Reviewer 2: 119.5s (3,155 chars).  ← Kimi slow run, paid Zen scheduling variance.
- Reviewers wall: 119.5s.
- Orchestrator intro: 7.8s (508 chars).
- **Total: 128.4s** (22,922 chars final).

A3 trial range 62–128s; median ~81s. Scheduling variance on Zen comparable in shape to A2's (15–70s) — the wider absolute floor reflects the parallel reviewer phase always having to wait for the slower of two cloud calls.

### Measurements

| Arm | Median total wall-clock | Range | Final output chars | Distinct numbered recs | Cascade engaged | Factual grounding |
|-----|------------------------|-------|--------------------|------------------------|-----------------|-------------------|
| A1 (cascade, prior) | 71–145s | 71–145s | 1,700–4,681 | ~9 | Yes (4-agent) | None |
| A2 (prompt-steered, prior) | 19.5s | 15–70s | 6,784–11,095 | 16 | None | None |
| **A3 (novel)** | **81s** | **62–128s** | **22,900–23,700** | **~22 across 3 sections** (R1 ~7–8, R2 ~6–8, plus script's 3 factual sections) | Three-agent parallel + concatenation | **Yes (script section)** |

**Output structure preservation: PASS.** All three trials produced final output with the four sections cleanly attributed: orchestrator framing intro → "Section 1: Script-Based Factual Findings" → "Section 2: Reviewer 1 — Accuracy and Completeness (Tencent Hunyuan family)" → "Section 3: Reviewer 2 — Clarity and Onboarding (Moonshot Kimi family)". The harness enforced this — no LLM had the opportunity to flatten the three substantive sections.

**Recommendation depth (qualitative).** Verbatim samples below; on substance, A3's per-recommendation depth is comparable to A2's "what's missing + recommendation" pattern, with Reviewer 2 (Kimi) consistently producing cleaner numbered recommendations than Reviewer 1 (hy3, which leaked extensive chain-of-thought). A3 surfaces *more total recommendations* (≥20 across the three sections vs A2's 16) but R1's recommendations are buried in narrative prose that the user must extract.

**Factual grounding: PASS.** The script section provides:
- Verified link counts (17 URLs, all 2xx/3xx external, 1 localhost example flagged).
- Confirmed standard-section presence (5/5 detected).
- Confirmed code-block parseability (all Python/YAML/JSON blocks parse).

A2 cannot produce these — A2 *asserts* claims about the README without any verification step. The script's grounding actively shaped both reviewers' recommendations: R2 explicitly cited the localhost-line-526 finding to recommend an inline `# Requires Ollama running` clarification (Trial 1, rec 6) and a Prerequisites section (Trial 1, rec 1). R1 cited line 526 as an anchor for clarifying the Ollama OpenAI-compatible base_url documentation across all three trials. The script outputs are not orphaned; the reviewers use them.

**Cross-section uncorrelation: STRONG.** The two LLM reviewers consistently produced recommendations with minimal overlap. Sample distinct findings from Trial 2:
- **R1 (hy3, accuracy/completeness):** streaming "enabled by default" vs `--streaming` flag contradiction; library "do not fetch from GitHub" vs `LLM_ORC_LIBRARY_SOURCE=remote` contradiction; MCP `.mcp.json` example uses `uv run` which fails for pip/Homebrew installs; missing uv prerequisite in Development; **undefined `default-local` profile referenced in fan-out example** (a real bug in the README, never surfaced by A2 in any of its three trials); undefined `ollama-gemma-small` profile in Ensemble Agents example.
- **R2 (Kimi, clarity/onboarding):** zero-config demo before authentication; relocate MCP block to advanced section; ensemble file provenance unclear at first use; forward reference for Model Profiles; visual hierarchy of installation options; validation checkpoint after auth; minimal YAML before complex 4-agent example.

The two reviewers' recommendation sets share 1–2 items (both flag MCP placement; both note the localhost line-526) and produce 5–8 distinct items each. This is the heterogeneity-uncorrelated-errors mechanism observable in the spike's actual data — not just a literature claim.

### Verbatim output samples

**Script section (deterministic, identical across trials):**

> # Script-Based Factual Findings (deterministic checks)
>
> README analyzed: `/Users/nathangreen/Development/eddi-lab/llm-orc/README.md`
> Total URLs checked: 17
> Standard sections evaluated: 5
> Code blocks with parseable language flagged: 0
>
> ## 1. Link Validity
> No dead or unreachable external links detected. All external URLs returned 2xx/3xx.
>
> 1 URL(s) point at localhost / loopback (documentation examples, not external links — flagged for review but not counted as broken):
>   - line 526: `http://localhost:11434/v1` — HTTP 404
>
> ## 2. Standard-Section Presence
> Present: installation, usage, license, contributing, examples_or_quickstart
> All required section types detected.
>
> ## 3. Code-Block Parseability
> All code blocks with parseable declared languages (python, yaml, json) parsed successfully.

**Reviewer 1 — Accuracy and Completeness (Trial 2, hy3, late-output recommendation excerpt — model leaks reasoning before reaching numbered output):**

> 8. Correct the fan-out (Parallel Map-Reduce) example agent profile reference from the undefined `default-local` to an existing profile (e.g., `free-local` or `openai-local`) to match the documented model profiles, as the current example references a profile that is not defined in the Configuration section, which would cause runtime errors for users copying the example. […] In the Ensemble Agents example, change `model_profile: ollama-gemma-small` to an existing profile (e.g., `free-local`) as `ollama-gemma-small` is not documented in the available profiles.

R1's quality-per-recommendation when extracted is high (line-cited, names the actual undefined profile, identifies a real copy-paste-breaking bug that A2 never found). R1's *output format* is poor — chain-of-thought leakage means the user has to read 18,000 characters of reasoning narration to extract ~8 recommendations. This is a Hunyuan model behavior, not a topology failure.

**Reviewer 2 — Clarity and Onboarding (Trial 2, Kimi, recommendations 3–4 verbatim):**

> 3. **Clarify Ensemble Provenance at First Use (line 125)**
>    The first practical example invokes `llm-orc invoke code-review`, but the `code-review` ensemble YAML shown at line 125 is presented as "Global Configuration" without stating whether this file is auto-generated, built-in, or must be manually created. **Add a parenthetical** at line 125: "(Create this file manually, or run `llm-orc library copy code-analysis/security-review` from the Library section below)." This prevents immediate command failure when users paste the invoke command before creating the ensemble.
>
> 4. **Add Forward Reference for Model Profiles (line 130)**
>    The Quick Start uses `model_profile: free-local` at line 130, but Model Profiles aren't explained until line ~480. **Insert** a parenthetical note at line 130: "(See 'Model Profiles' section below for how to define `free-local` in `~/.config/llm-orc/config.yaml`)." Currently, users encounter undefined syntax before its explanation, causing confusion about whether these are built-in or custom.

R2's per-recommendation depth matches A2's pattern — explicit problem statement, named line number, exact textual change, mechanism for why current state breaks. Comparable in specificity to A2's `chmod 600` and AES-256-GCM examples.

For comparison, A2's verbatim sample (re-quoted from L3):

> ### 2. No Guidance on File Permissions
>
> Credentials are stored in `~/.config/llm-orc/credentials.yaml`, but the README does not mention file permissions. […]
>
> **What's missing**: Guidance to run `chmod 600 ~/.config/llm-orc/credentials.yaml` or equivalent for credential file protection.
>
> **Recommendation**: Add a security hardening step in the authentication section: "Ensure your credentials file has restricted permissions: `chmod 600 ~/.config/llm-orc/credentials.yaml`"

A2 and R2-Kimi produce structurally similar named-specificity output. R2 adds line numbers (script-grounded) that A2 does not produce. R1-hy3 produces equally specific findings but in chain-of-thought form that costs the user extraction effort.

### Quality:latency comparison vs A2

A2 invests 19.5s wall-clock to surface 16 numbered recommendations across security/performance/quality dimensions. A3 invests 81s wall-clock (median) to surface ~22 recommendations across three attributed sections (script's 3 factual findings + R1's ~8 + R2's ~7), with R2's clean output rivaling A2's specificity and R1's noisy output containing accuracy findings A2 never produced (the undefined-profile bugs in fan-out and ensemble-agents examples).

On the spike's named axes:

- **Per-recommendation specificity:** A3's R2 ≈ A2 (both cite line numbers / config snippets; R2 cites README line numbers grounded in script's report). A3's R1 has *content* parity but format penalty due to chain-of-thought leakage. **Net: tie at the per-section level for R2; R1 underperforms on usable presentation.**
- **Total recommendation count:** A3 (~22) > A2 (16). But A2's 16 are all clean numbered items in three dimensions; A3's count is split across four sections and one section requires extraction work.
- **Factual grounding:** A3 produces what A2 cannot (verified link counts, confirmed section presence, parseability). The reviewers use these. **A3 wins on this dimension; this is what the script-agent slot was designed to add.**
- **Latency:** A3 (81s median) is 4× slower than A2 (19.5s) and within the same band as A1's median (71–86s). **A3 loses on latency by a clear margin.**
- **Output structure:** A3 preserves three-section attribution (orchestrator did not collapse). A2 is single-section. **Different artifact shapes; not directly comparable but both are usable.**

### Pass/fail interpretation

Per the spike specification's pass conditions, this outcome is **moderate pass with caveats** — neither a strong pass nor a refute:

- A3 ≈ A2 on per-recommendation specificity (R2 matches A2; R1 matches in content but loses to format).
- A3 adds factual grounding A2 cannot produce.
- A3 surfaces accuracy findings A2 missed (undefined model profiles in YAML examples — a real, copy-paste-breaking documentation bug).
- A3 latency is 4× A2's median, but bounded (under the 145s A1-clean ceiling). 60–130s is workable for non-interactive review tasks; not workable as a chat-loop replacement.
- The MARG-style concatenation hypothesis was preserved by the harness (no collapse). The orchestrator's framing intro stayed in scope (3–5 sentences, no synthesis).
- The heterogeneity-uncorrelated-errors mechanism was observable in the data: R1 and R2 produced distinct findings with low overlap.

The strong-pass conditions ("A3 ≥ A2 on specificity AND adds factual grounding AND latency acceptable") do not all fire — latency is materially higher than A2 and R1's format degraded the user experience. The refute conditions ("A3 < A2 on specificity OR orchestrator collapsed structure OR latency unworkable") do not fire either — specificity is at parity, structure was preserved by the harness, latency is above A2 but below A1.

The cycle's premise — that *some* well-architected ensemble process can produce results that justify the structural-composition complexity over prompt-steering — is **partially supported but not vindicated**. A3 demonstrates that:

1. **The script-agent slot earns its place.** Deterministic checks produce factual grounding LLMs structurally cannot replicate (no LLM hallucinates a verified link count). This is a topology-shaped contribution, not a prompt-shaped one. Where a task class needs factual grounding, the script slot is competitive on its own terms.

2. **Heterogeneous LLM reviewers do produce uncorrelated findings.** The data confirms the literature mechanism for this README task class. R2-Kimi found onboarding-friction issues R1-hy3 didn't; R1-hy3 found accuracy contradictions and undefined-profile bugs R2-Kimi didn't.

3. **The MARG-concatenation aggregator preserves what cascade-summarization destroys.** Spike A's collapse step was the failure mode; A3's no-collapse design surfaces the per-reviewer voices. This is the design fix the literature predicted.

4. **But latency is not free.** A3's parallel-reviewer phase always pays the slower-of-two cloud-call cost, plus the orchestrator framing call. A2 is one cloud call; A3 is at minimum three. For interactive use, A2 wins. For batch/asynchronous review where latency is amortized, A3's added grounding and finding-count justify the overhead.

5. **Model selection inside the heterogeneity slot matters more than the spike anticipated.** R1-hy3's chain-of-thought leakage is a quality cost the topology cannot fix. The heterogeneity finding ("different families = uncorrelated errors") is correct on findings-coverage, but a model that produces unstructured prose under a 4096-token cap is a poor reviewer-slot inhabitant regardless of family-distinctness. A future iteration would swap hy3 for a different family-distinct model that produces clean structured output.

### Implications for essay 003

Three implications for revision:

1. **Essay 003's claim about prompt-steering as the simpler-and-better default holds, with a refined caveat.** Where the task class needs factual grounding (verified facts about source artifacts), prompt-steering structurally cannot produce it; a script-grounded ensemble has a topology-shaped advantage. The essay's "structural composition justified only by task-class-specific evidence not yet collected" sentence (L3 implication 3) now has its first piece of task-class-specific evidence: README review with factual grounding requirements rewards structural composition; README review without those requirements does not. The essay should distinguish these two task subclasses explicitly.

2. **The MARG-concatenation pattern should be promoted from "literature finding" to "empirically observed in this cycle."** L4's literature claim that no-synthesizer architectures preserve specificity (60% → 29% generic-comment rate per Bhayana et al.) was supported in this spike's data: the per-reviewer voices were preserved exactly because the harness enforced concatenation. This is now spike-confirmed for the cycle's own task class, not just borrowed from elsewhere.

3. **The single-Model-Profile-floor finding (ADR-011) survives but with a more nuanced boundary.** A3 demonstrates that *some* ensembles can produce things prompt-steering cannot (factual grounding, uncorrelated errors). It does not demonstrate that all ensembles are worth their complexity — A1's cascade still loses to A2. The boundary is "ensembles whose topology adds structural capability prompt-steering lacks" (script-agent factual grounding; heterogeneous-reviewer uncorrelated errors via MARG concatenation) versus "ensembles whose topology only re-implements what prompt-steering does at higher latency" (cascade-with-collapse-synthesizer). Essay 003's §"What the spikes confirmed" should incorporate this distinction.

The MQ-1 Branch One condition ("A3 ≥ A2 on specificity AND adds factual grounding AND latency acceptable") does not fully fire — latency is higher than A2 — so ADR-011's reconsideration trigger is not unambiguously activated. But the script-agent factual-grounding axis is a new capability vector A2 cannot match, which warrants a softer rephrasing of ADR-011's "single Model Profile remains defensible" stance: defensible *as a default* but not *as a ceiling* for task classes where factual grounding is load-bearing.

### Cleanup

Scratch directory `scratch/spike-a3-novel-ensemble/` deleted per spike rules. No project ensembles or model profiles modified during this spike (the harness ran outside llm-orc, calling Zen directly to isolate the topology test from instruction-compliance behavior; no `.llm-orc/ensembles/development/` files added or changed; no `~/.config/llm-orc/config.yaml` profiles added or removed). No git commits made; `git status` confirms only the pre-spike research-log addition and pre-existing migration-related changes.

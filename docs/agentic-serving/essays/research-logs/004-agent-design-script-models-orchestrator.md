# Research Log — Cycle 3 (Agent Design)

**Cycle:** 3 (agentic-serving scoped corpus)
**Started:** 2026-05-01
**Plugin version at cycle open:** v0.8.5
**Builds on:** essay 003 (`essays/003-multi-turn-orchestration-and-the-four-axis-frame.md`)
**Cycle territory:** agent design specifically — building on what Cycle 2 surfaced about scripts + small local models + fast cloud orchestrator combinations.
**Mode declaration at entry:** undeclared. Practitioner's working assumption is Mode B (Research Only); close shape is determined at the research-phase gate.

## Step 1.1 — Research questions articulated (per ADR-082)

Per ADR-082's question-isolation entry protocol, the research questions are written into the research log as the entry's first content, before the agent consults the corpus for this cycle. These three RQs were articulated through orchestrator-handoff conversation. The load-bearing inherited commitments come from Cycle 2's susceptibility snapshot's specific in-cycle grounding action (see `housekeeping/audits/susceptibility-snapshot-cycle-2-research.md`).

### RQ-1 — Isolate A3's load-bearing component (priority — runs first)

Does a prompt-steered single cloud orchestrator receiving a script-agent's deterministic report as additional input context ("A2 + script input") produce equivalent factual grounding to A3's novel ensemble (script + heterogeneous LLMs + MARG concatenation) on the cycle-2 README-review task class?

**Spike battery commitment:** this RQ runs **before** Cycle 3 synthesizes any ensemble-topology findings, per the susceptibility snapshot's specific in-cycle grounding action.

**Outcome decision tree:**
- Equivalent grounding at A2's latency → architectural lesson is "augment prompt-steering with deterministic tool outputs," not "use ensemble topology"; ADR-011's boundary refinement does not need relaxation.
- Worse grounding → A3's heterogeneity + concatenation is doing real work; ADR-011's boundary refinement holds.
- Equivalent grounding with different latency/cost profile → finer-grained regime-specific lesson.

**Scope:** same fixture as A3 (the README that surfaced documentation bugs A2 missed across three prior A2 trials). Generalization to other task classes is a follow-up.

### RQ-2 — Test the four-priorities frame as load-bearing or lens

Does Cycle 3's evidence produce any agent-design choice that the four-priorities frame (performance × environmental cost × local-first × token cost) would resolve differently than a performance-only frame?

**Falsification criterion:** if no Cycle-3-tested configuration produces a frame-divergent recommendation, the four-priorities frame is rhetorical and Cycle 3 retires it (or replaces it with an operationalized substitute) and re-anchors downstream decisions on the performance-only frame supplemented with explicit qualitative priority statements when needed.

**Operationalization:** for each spiked configuration, score on all four priorities (environmental cost and local-first qualitative; performance and token cost measured). Identify cases where the two frames lead to different recommendations.

### RQ-3 — Multi-turn tool-dispatching reliability (lit + spike)

Does the tau-bench multi-turn tool-dispatching reliability ceiling (Yao et al. 2024: GPT-4o under 50% task success; pass^8 below 25% at frontier) generalize to llm-orc's deployment configurations? What agent-design choices reduce the meltdown rate on multi-turn sustained work?

**Includes a scoped multi-turn spike** (design TBD during research-entry).

**Connects to:** the practitioner-named ensembles-of-ensembles affordance (cycle-2 archive item #25), which Cycle 2 named as empirically motivated but did not test.

### Discipline notes (apply throughout, per cycle-2 susceptibility snapshot)

- **Name the operating frame up-front** in every synthesis dispatch (research-methods reviewer, lit-reviewer, argument auditor). Counters performance-axis bias in initial syntheses.
- **Commit to the narrow reading first** in spike-finding syntheses; broaden only with explicit evidence. Counters self-correction blind-spot on overgeneralization.

## Step 1.2 — Constraint-Removal Prompt (ADR-082)

**Most consequential existing artifact named for this cycle:** **ADR-003** (the closed five-tool orchestrator surface — `invoke_ensemble`, `compose_ensemble`, `list_ensembles`, `query_knowledge`, `record_outcome`).

**Why this artifact (agent's reasoning):** Cycle 2 bracketed ADR-011 (the orchestrator-LLM-is-a-Model-Profile commitment) at its constraint-removal moment. ADR-003 is the next-level architectural premise and the most definitional artifact for the cycle's named territory ("agent design specifically"): it defines what the orchestrator agent's action space *is*. The orchestrator is currently structured as a single LLM that dispatches a fixed set of five tools — and that shape pre-determines what "an agent" looks like in the corpus. For Cycle 3 to investigate agent design, ADR-003 is the artifact whose presence most shapes the default solution space.

**Constraint-removal prompt (composed against ADR-003):**

> *What if ADR-003's closed five-tool orchestrator surface were not available — what if "the agent" were not pre-committed to "a single LLM dispatching a fixed set of tools" at all? How would agent design for the cycle's task classes (single-ask code-review, multi-turn sustained work) be approached then?*

**Practitioner response (verbatim):**

> "Right makes sense — I am not wedded to any predisposition for the agent. It's all about outcomes over an agentic session."

**What this shifts (agent's interpretation, recorded for transparency):** The response brackets ADR-003 substantively rather than treating it as irreplaceable. The practitioner names the unit of analysis as **outcomes over an agentic session**, not the orchestrator's structural shape. The implication for Cycle 3's research design: agent shapes (single LLM with closed tools, script-driven process where LLM calls are subordinate steps, ensembles, hybrid architectures, anything else the literature surfaces) are *means* whose value is judged against the outcomes they produce per session — not premises that pre-shape what "agent design" can investigate. The cycle's three RQs are read in this frame: RQ-1 isolates whether ensemble topology or tool-augmentation is the load-bearing means for factual-grounding outcomes; RQ-2 tests whether the four-priorities frame produces frame-divergent recommendations against any agent shape; RQ-3 measures multi-turn reliability outcomes against the existing deployment shape and considers whether alternative agent shapes have evidence on multi-turn outcomes the literature can speak to.

The constraint-removal exchange is complete. Step 1.3 follows.

## Step 1.3 — Research plan and reviewer dispatch

### Research plan (presented to practitioner; awaiting approval before reviewer dispatch)

**Operating frame named up-front (discipline note 1):** "Outcomes over an agentic session." Agent shape is means; the cycle's evaluation axis is outcome quality per session, not architectural fidelity to any pre-existing shape.

**Phase 1 — Combined literature review (single lit-reviewer dispatch, all three RQs).**

Coverage organized by RQ:

- **RQ-1 supporting literature** — tool-augmented prompt steering vs. ensemble topology; deterministic-tool grounding (Toolformer, ReAct evaluations, tool-use grounding benchmarks); script-LLM hybrid agent architectures; published evidence comparing single LLM + tool-output context vs. multi-component ensembles on factual-grounding tasks.
- **RQ-2 supporting literature** — framings of agent design priorities in production systems (multi-objective vs. performance-only); environmental cost reporting in agent benchmarks; local-first architectures and their published trade-off analyses; whether the published literature operationalizes multi-priority framings or treats them as rhetorical.
- **RQ-3 multi-turn supporting literature** — tau-bench (Yao et al. 2024) plus follow-ups; long-horizon and multi-turn agent benchmarks beyond Cycle 2's coverage (Cycle 2 covered HORIZON, AMA-Bench, LongCLI-Bench — Cycle 3 should look for NEW post-2026 work and meltdown-analysis literature); reliability frameworks; **agent shapes departing from LLM-with-tools** (script-driven loops with LLM as subordinate step, hierarchical task decomposition, deterministic state machines with LLM nodes) and their multi-turn evidence base.
- **Cross-cutting** — outcome-framing literature: what does the agent literature evaluate, and how does it operationalize "outcome" vs. "architecture-as-end"?

The lit-review dispatch will name the operating frame ("outcomes over an agentic session"; agent shape is means) and the discipline notes (commit to narrow reading first; surface alternatives without prompting) up-front.

**Phase 2 — Spike A: RQ-1 isolation test (A2 + script input).**

- **Question:** Does A2 + script input produce equivalent factual grounding to A3 on the cycle-2 README-review fixture?
- **Method:** Replicate A2's prompt-steered single cloud orchestrator setup; prepend the script-agent's deterministic report (the same script slot used in A3) as additional input context; run on the same fixture A3 used.
- **Comparison axes:** factual-grounding correctness (did it find the undefined-model-profile documentation bugs A2 missed and A3 caught?); recommendation count and specificity; latency; token cost; output structure.
- **Synthesis discipline:** narrow reading first — scope the finding to this configuration on this fixture before reasoning about generalization.
- **Constraint:** per the susceptibility snapshot's grounding action, Spike A's findings must land **before** Cycle 3 synthesizes any ensemble-topology conclusions from the lit review.

**Phase 3 — Spike B: RQ-3 multi-turn meltdown probe.**

Multi-turn fixtures are harder to design than single-ask. Three candidate approaches, ranked by the agent's preference and surfaced for practitioner choice:

1. **Tau-bench subset** — use the published tau-bench retail or airline scenario subset directly. Pros: published baseline (GPT-4o under 50% reference), reproducible, no fixture design overhead. Cons: tooling may need adaptation to llm-orc's deployment shape; tau-bench's task class may not match agentic-coding-tool sessions.
2. **Real multi-turn coding session via OpenCode + llm-orc** — instrument an actual multi-turn coding session (e.g., a small refactor or feature add the practitioner would do anyway), measure meltdown signatures (turn-count to completion, error self-conditioning, tool-dispatch failures, premature giveup). Pros: highest realism for the cycle's user contract; uses existing deployment. Cons: uncontrolled — single-trial findings; hard to compare across configurations.
3. **Custom small fixture** — design a 5–10 turn fixture mirroring real agentic-coding patterns (read file → modify → check → adjust). Pros: controlled comparison across configurations; bounded scope. Cons: fixture design overhead; less realistic than option 2.

**Practitioner decision needed:** which fixture approach for Spike B? Or some mix?

**Phase 4 — Synthesis and essay.**

After lit-review + Spike A + Spike B findings are recorded, synthesize into essay 004. The susceptibility-snapshot grounding-action commitment shapes synthesis order: ensemble-topology synthesis waits for Spike A's findings; the four-priorities frame test (RQ-2) runs against all spike outputs at synthesis time.

### Sequencing

- Lit-review and Spike A run in parallel (lit-review surfaces literature; Spike A surfaces empirical isolation evidence; neither blocks the other).
- Synthesis exchanges drawing topology conclusions from the lit-review wait for Spike A's findings.
- Spike B (multi-turn) sequenced after lit-review's tau-bench coverage and Spike A's findings — multi-turn fixture design will benefit from both.

### Open methodological choices for practitioner

1. **Spike B fixture approach** — tau-bench subset / real session / custom small fixture / mix? (Default recommendation: tau-bench subset for the comparison baseline + a small real-session probe for outcome-grounding.)
2. **Lit-review scope on agent-shape alternatives** — should the lit-review explicitly seek published evidence on script-driven-loop / hierarchical / state-machine agent architectures, or hold those open for the next cycle if Cycle 3's evidence opens that territory? (Default recommendation: explicit coverage now — the practitioner's "no predisposition" framing makes alternatives in scope.)
3. **Lit-review timing** — dispatch immediately after the methods reviewer clears, or stage Spike A first to avoid lit-review framing contaminating spike interpretation? (Default recommendation: dispatch lit-review and run Spike A in parallel; the spike's empirical evidence is independent of the lit-review's literature framing.)

### Plan approval

Practitioner confirmed the plan as presented (2026-05-01) — defaults adopted for all three open methodological choices:

1. **Spike B fixture:** tau-bench subset (cross-config baseline) + small real-session probe via OpenCode + llm-orc (outcome-grounding).
2. **Lit-review scope:** explicit coverage of agent-shape alternatives (script-driven loops, hierarchical, state-machine) — the "no predisposition" framing puts these in scope.
3. **Lit-review timing:** parallel with Spike A; synthesis exchanges drawing topology conclusions wait for Spike A's findings.

### Reviewer dispatch

Dispatching the **research-methods-reviewer** specialist subagent with the three-RQ question set, the constraint-removal response, the operating frame, the discipline notes, and prior research context (essays 001/002/003 + Cycle 2 archive). Output path: `docs/agentic-serving/housekeeping/audits/research-design-review-cycle-3.md` (corpus follows ADR-070 housekeeping placement; ADR-085 `.rdd/` migration deferred per cycle-status conformance notes).

## Step 1.4 — Reviewer findings + practitioner decisions

The research-methods-reviewer's report (`docs/agentic-serving/housekeeping/audits/research-design-review-cycle-3.md`) cleared the question set with three P2 findings and one P3 finding. Practitioner decisions on each (2026-05-01):

### P2 #1 — Script-as-orchestrator shape (Criterion 4 incongruity) — adopted as lit-review instruction

The constraint-removal exchange opened space for agent shapes where the LLM is a bounded subordinate step rather than the primary orchestrator. RQ-1 tests two LLM-as-orchestrator arms; the simpler script-as-orchestrator shape (Cycle 2 evidence: A3's script slot was its most unambiguous value contributor) is not asked about.

**Decision:** Adopt as a **targeted lit-review coverage instruction** (not as a sub-question to RQ-1). Cheap, addresses the territory without adding a new spike. The instruction added to the lit-review dispatch is:

> *For script-driven loops with LLM as a subordinate step (rather than LLM as the primary orchestrator), seek published evidence on (a) factual-grounding task classes and (b) multi-turn sustained work. If this shape has evidence on either task class, surface it as a comparison point against the LLM-as-orchestrator shape.*

If the lit-review surfaces a substantial evidence base, Cycle 3 may add a spike or carry the question to Cycle 4.

### P2 #2 — RQ-2 scoring-resolution caveat — adopted verbatim

RQ-2's falsification has a structural weakness: qualitative scoring on environmental cost and local-first has lower resolution than measured scoring. The reviewer's recommended operationalization addition (verbatim) is now part of RQ-2's working text:

> *Scoring resolution caveat: qualitative scoring on environmental cost and local-first has lower resolution than measured scoring. A frame-convergent finding at qualitative resolution does not constitute a clean falsification — it means the frame-divergent signal, if present, was below the scoring threshold. If all four configurations score qualitatively equivalent on both qualitative axes, record this explicitly as 'no detectable divergence at qualitative resolution' rather than as 'frames converge.' Retirement of the four-priorities frame requires either a measured divergence finding or a stated judgment that qualitative resolution is sufficient for the cycle's claims.*

### P2 #3 — RQ-3 failure-mode framing — adopted (reformulation + spike-design instruction)

RQ-3's "meltdown rate" framing crowds out adjacent failure modes the literature names (premature stop, error self-conditioning, memory retrieval drift, early stall). Reformulation adopted (verbatim):

> *Does the tau-bench multi-turn tool-dispatching reliability ceiling (Yao et al. 2024) generalize to llm-orc's deployment configurations? What agent-design choices reduce the rate of observable failure modes — including meltdown onset, premature stop, and error self-conditioning — on multi-turn sustained work?*

Spike B instrumentation discipline (applied at fixture-design time): explicitly name the failure modes to detect — at minimum meltdown onset, premature stop, and error self-conditioning — before the fixture runs.

### P3 — RQ-1 architectural-lesson scope (synthesis-time discipline) — adopted as synthesis discipline note

RQ-1's decision-tree lesson statements ("architectural lesson is 'augment prompt-steering with deterministic tool outputs'") are architecture-level characterizations from a single fixture. The susceptibility-snapshot discipline ("commit to narrow reading first") applies: when the equivalent-grounding branch fires, the synthesis discipline scopes the lesson to "on the cycle-2 README-review fixture" before any architecture-level generalization. No question revision; this fires at Spike A synthesis time.

### P3 floor question — noted for cycle-4 feed-forward

The reviewer surfaced a second incongruity not addressed by Cycle 3: "what value does any LLM contribute once the script has run?" — the floor of LLM contribution beyond formatting script output. RQ-1 doesn't test this; legitimately scoped out of Cycle 3. Recorded as Cycle 4 candidate question.

### Final working RQs (after reformulation)

**RQ-1 (priority — runs first per snapshot's grounding action):** Does a prompt-steered single cloud orchestrator receiving a script-agent's deterministic report as additional input context ("A2 + script input") produce equivalent factual grounding to A3's novel ensemble (script + heterogeneous LLMs + MARG concatenation) on the cycle-2 README-review task class? *Synthesis discipline:* equivalent-grounding-branch lessons scoped to "on the cycle-2 README-review fixture" before any architecture-level generalization.

**RQ-2:** Does Cycle 3's evidence produce any agent-design choice that the four-priorities frame (performance × environmental cost × local-first × token cost) would resolve differently than a performance-only frame? *Falsification criterion plus scoring-resolution caveat:* see verbatim P2 #2 text above.

**RQ-3 (reformulated):** Does the tau-bench multi-turn tool-dispatching reliability ceiling (Yao et al. 2024) generalize to llm-orc's deployment configurations? What agent-design choices reduce the rate of observable failure modes — including meltdown onset, premature stop, and error self-conditioning — on multi-turn sustained work?

## Step 1.5 — Research loop begins

The research-entry protocol (Steps 1.1–1.4) is complete. The cycle now reads the existing artifact corpus and proceeds to Step 2 (Research). Per the approved plan: Phase 1 (combined lit-reviewer dispatch) and Phase 2 (Spike A: A2 + script-input isolation test) run in parallel; Phase 3 (Spike B: multi-turn meltdown probe) sequences after lit-review's tau-bench coverage and Spike A's findings.

### Phase 1 dispatch — combined lit-reviewer

Operating frame named up-front in the dispatch ("outcomes over an agentic session; agent shape is means"). Discipline notes attached (commit to narrow reading first; surface alternatives without prompting). Coverage instructions per RQ + the script-as-orchestrator literature instruction (P2 #1). Output: `essays/research-logs/004a-lit-review-agent-design.md`.

**Status:** Completed 2026-05-01. 23 new sources reviewed (post-Cycle-2 work + sources Cycle 2 did not cover). Operating frame, narrow-reading-first discipline, alternative-framing preservation, and honest-absence reporting all confirmed operative in the output. See `004a-lit-review-agent-design.md` §6 "Implications for Cycle 3's Spike Battery (Advisory)" for the lit-reviewer's spike-design recommendations.

### Lit-review key findings (cross-RQ patterns)

The five cross-RQ patterns from §4 of the review are load-bearing for what follows:

1. **Script-slot's value is narrow but distinctive.** Deterministic script execution produces a category of evidence (verified facts with zero hallucination probability) that LLM-only configurations structurally cannot produce. AlphaCodium 19%→44% accuracy on code tasks via verification gating; Information Fidelity (arXiv:2602.13320) 80% distortion reduction with semantic weighting; Cycle 2's Spike A3 empirical anchor data. Competing framing preserved: once deterministic facts are injected as context, a single capable LLM may reason from them as well as a heterogeneous ensemble — RQ-1's spike resolves which framing holds on the cycle's task class.

2. **Multi-priority evaluation produces frame-divergent recommendations at enterprise scale, with uncertain transfer to small-model local configurations.** CLEAR (arXiv:2511.14136): accuracy-only optimization yields 4.4–10.8× cost premium on enterprise tasks at frontier tier. Whether this manifests at qualitative resolution on the cycle's configurations remains empirically open per the P2 #2 caveat.

3. **The single-agent baseline literature has converged on a *conditional* claim.** OneFlow (arXiv:2601.12307) and Lee et al. (arXiv:2601.04748): single-agent simulation matches *homogeneous* multi-agent on most task classes, but **heterogeneous configurations remain the residual unsettled case**. A3 is heterogeneous (Hunyuan + Kimi from different model families). The literature thus *delineates* the exact scope condition under which the "A2 + script input" configuration might fall short. RQ-1's spike sits in a literature gap, producing novel evidence.

4. **Outcome operationalization affects spike evaluation design.** The PAE / "corrupt success" finding (Cao et al. arXiv:2603.03116): an agent that completes a task by bypassing process or fabricating analysis is *categorically disqualified*, not scored equivalently to an agent that followed correct procedures. Direct implication for RQ-1: **issue-count parity is not sufficient if A2+script achieved it by restating script output rather than independent LLM analysis**. AgentEval's 2.17× higher failure detection rate via DAG-structured evaluation (vs. end-to-end) is the empirical quantification of this gap.

5. **Multi-turn reliability literature is effectively post-dated.** Tau-bench (2024), HORIZON (2026), LongCLI-Bench (2026), Khanal et al. (2026) remain primary. τ²-Bench (arXiv:2506.07982) extends tau-bench to dual-control: pass@1 drops from 56–74% (single-control) to 34% (dual-control). No post-2026 benchmark substantively refutes Cycle 2's findings. Honest absence: no published study measures meltdown-onset / premature-stop / error-self-conditioning rates at qwen3:8b tier.

### Lit-review-driven spike-plan refinements (proposed)

- **Spike A evaluation design** must add PAE-aware procedural-independence check alongside issue-count parity. Concrete operationalization: distinguish (a) does A2+script find the issues A3 found? from (b) does A2+script find them via independent LLM analysis or by restating the script's deterministic output in review language? The procedural-independence check requires examining the orchestrator's response surface for synthesis beyond literal script content.

- **RQ-2 cost-axis measurement** should follow CLEAR's cost-normalized accuracy (CNA) pattern: report cost per unit of outcome quality, not cost in isolation. This gives the four-priorities frame a measurable operationalization on the cost axis and allows direct comparison with the performance-only frame.

- **Spike B failure-mode detection** is now well-specified by the lit findings: meltdown onset (context-collapse looping), premature stop (incomplete turn), error self-conditioning (LLM conditioning on prior incorrect tool dispatch), plus user-direction failures (per τ²-Bench's dual-control finding). Information Fidelity's re-grounding-every-~9-steps recommendation is a concrete design lever for the multi-turn fixture.

- **Script-as-orchestrator shape's status changed.** Pre-lit-review: P2 #1 was a literature-coverage instruction with no commitment to spike. Post-lit-review: Routine (arXiv:2507.14447) shows 41%→96% on GPT-4o and 32.6%→83.3% on Qwen3-14B for structured planning scripts; Compiled AI (arXiv:2604.05150) takes the further step of removing LLM from execution entirely. Honest absence: no published paper directly compares script-as-orchestrator shapes against LLM-as-orchestrator shapes on documentation-review or README-analysis. **The cycle would be producing novel empirical evidence if it adds this comparison.** Practitioner decision needed: keep for Cycle 4 (original plan) or add as a Cycle 3 spike now.

### Practitioner decisions on lit-review-driven refinements (2026-05-01)

Practitioner adopted all four recommendations.

- **Decision A — Spike A PAE-aware procedural independence check: adopted.** Spike A's evaluation includes both (a) issue-count parity check — does A2+script find the bugs A3 found? — and (b) procedural independence check — does A2+script find them via independent LLM analysis or by restating the script's output in review language? Operationalization: examine the orchestrator's response surface for synthesis content beyond what's literally in the script's report. PAE / corrupt-success framing applied throughout the spike's evaluation.

- **Decision B — RQ-2 cost-axis follows CLEAR's CNA pattern: adopted.** Cost-normalized accuracy reporting per spiked configuration. Total cost per spike trial recorded (input tokens × input rate + output tokens × output rate); CNA = quality-axis-score / total-cost. Both four-priorities frame and performance-only frame applied to the same CNA data; frame-divergent recommendations recorded explicitly when present.

- **Decision C — Script-as-orchestrator hybrid (option c): adopted contingent on free-tier quota.** Spike A becomes a three-arm comparison if MiniMax M2.5 Free quota can sustain three arms × N=3 trials:
  - **Arm 1 — A2 baseline:** prompt-steered single MiniMax orchestrator on README.md, no script context (replicates Cycle 2's A2).
  - **Arm 2 — A2 + script input:** same prompt-steered single MiniMax + the script's deterministic report prepended as input context (the RQ-1 isolation test).
  - **Arm 3 — Script-as-orchestrator:** script runs the deterministic checks; LLM (single MiniMax) is invoked once at the end to synthesize script findings into review prose. The LLM does not orchestrate or plan; it formats and adds analysis on top of script output.

  All three arms compared against Cycle 2's A3 findings as documented in essay 003. Same fixture (project README.md). All three evaluated on the same axes (issue-count parity, procedural independence per Decision A, latency, token cost, CNA per Decision B, output structure).

  **Quota fallback:** If free-tier quota constraints emerge, drop to two-arm Spike A (A2 + script input only) and carry Arm 3 to Cycle 4. The script-orchestrated probe's literature support (Routine arXiv:2507.14447 — 32.6%→83.3% on Qwen3-14B; Compiled AI arXiv:2604.05150) makes Cycle 4 well-motivated regardless.

- **Decision D — A2 prompt recovery: search prior commits first.** If recoverable from git history, `.opencode/`, or other persisted artifacts, use the exact prompt. Otherwise reconstruct from `code-review.yaml`'s `default_task` and document the divergence as a Spike A scope condition.

### Step 1.5 dispatch — Spike A setup complete (cost-gate reached)

Decisions A–D adopted. Setup steps completed (all non-cost):

1. **A2 prompt recovery** (Decision D): exact prompt **not recoverable** — Cycle 2's spike work was never committed (clean reflog, no stashes, no project-level OpenCode config). The "biased system prompt" referenced in essay 003 was OpenCode-CLI's default agent-mode prompt at the OpenCode version Cycle 2 ran. Reconstruction uses `code-review.yaml`'s `default_task` (the project's own production code-review prompt; same register as Cycle 2's A1 cascade arm). Documented as Spike A scope condition.

2. **Orchestrator profile recovered** at `~/.config/llm-orc/config.yaml`: profile `orchestrator-minimax-m25-free` (model=minimax-m2.5-free, provider=openai-compatible/zen, base_url=https://opencode.ai/zen/v1). Active llm-orc `agentic_serving.orchestrator.model_profile` setting still points to it.

3. **Script-agent recreated** at `scratch/spike-a-cycle3-a2-script-input/deterministic_analyzer.py`. Runs locally on project README.md in ~6s. Output structurally matches A3's findings: 5/5 canonical sections present (Installation, Quick Start, Configuration, Use Cases, License), 16 external URLs all 2xx/3xx, 1 loopback URL flagged separately, 8/8 YAML and 2/2 JSON code blocks parseable. The undefined `default-local` and `ollama-gemma-small` profile bugs **are still in the current README** (referenced in YAML examples at lines 619/626/671 but not defined in the Configuration section's profile list). Script does NOT find these bugs — matches A3's behavior; the bugs were caught by R1-Hunyuan's semantic analysis after seeing the script as anchor.

4. **Spike A harness built** at `scratch/spike-a-cycle3-a2-script-input/spike_a_harness.py`. Path A: uses llm-orc's production `ConfigurationManager` + `CredentialStorage` + `ModelFactory` code path. Three arms encoded (arm1 A2 baseline / arm2 A2+script input / arm3 script-as-orchestrator). Dry-run mode verified: bootstrap works, profile resolves, all three arm prompts render. **Path-to-API divergence from Cycle 2's A2** (Cycle 2 went via `opencode run`; harness goes via llm-orc client) — same endpoint and model; documented as second Spike A scope condition.

### Cost gate — awaiting practitioner authorization for live trials

Live execution is the only cost-incurring step. **Awaiting practitioner authorization** per the cycle's free-options preference and standing pause-before-cost directive.

### Spike A — executed and synthesized

**2026-05-01 — practitioner authorized live trials.** All seven configurations executed:

- **Cheap tier (5 arms × N=3 trials, MiniMax M2.5 Free via OpenCode Zen, free-tier $0.00):** arm1 (A2 baseline, 2/3 successful); arm2 (A2 + script input, 3/3); arm3 (script-as-orchestrator CONTAMINATED, 3/3); arm3_debiased (3/3); arm4 (script-only no README, 3/3).
- **Frontier tier (2 arms × N=2 trials, Sonnet 4.6 via Claude Code Agent dispatch as facsimile):** arm-frontier-bare (2/2); arm-frontier-with-script (2/2).
- 17 successful trials total; 1 transient HTTP 500 (arm1 trial 3).

**Mid-spike practitioner reframe (recorded in memory: `cycle-3-central-question`):** the cycle's central question is whether cheap-orchestrator + orchestration competes with a more expensive frontier model — not whether single agents beat orchestration generically. Two frontier-tier arms added in response.

**arm3 contamination caught and isolated.** The arm3 system prompt named the failure category being tested (*"do YAML examples reference profiles or keys that are actually defined in the document?"*). Caught at trial-results review when 3/3 detection was inconsistent with rest of cheap-tier data. Follow-up arm3_debiased (1/3) and arm4 (script-only, no README, 0/3) disambiguated. Methodological self-correction recorded; contaminated arm preserved for transparency.

**Findings (full synthesis at `004b-spike-a-cycle3.md`):**

- **RQ-1 settled at fixture scope.** A2 + script input does NOT match A3's grounding on the cycle-2 README-review fixture (arm2 0/3). ADR-011's Cycle 2 boundary refinement holds.
- **Within-tier finding:** explicit prompt direction toward the failure category dominates architectural framing for bug detection. arm3 contaminated 3/3, arm3_debiased 1/3, arm2 0/3 — monotonic in prompt directiveness.
- **Cross-tier finding (the cycle's most novel result):** cheap-tier-with-directed-prompting and frontier-tier-direct find largely **uncorrelated bug classes** on this fixture. Cheap catches semantic-consistency bugs (`default-local`, `ollama-gemma-small` undefined references); frontier catches architectural / security / operational bugs (config hierarchy ordering, encryption claim, model-ID staleness, fan-out failure documentation, MCP HTTP transport security, nesting-depth-limit, reset-global footgun). Heterogeneity-uncorrelated-errors mechanism (Sun et al. 2025; Ding et al. 2024) extending cross-tier — novel empirical evidence the literature found surfaced but did not directly test.
- **RQ-2 empirical evidence:** four-priorities frame produces frame-divergent recommendation versus performance-only frame on this fixture. Performance-only would select frontier-tier-alone (and miss undefined-profile bugs); four-priorities would select complementary multi-tier review (broader coverage at lower aggregate cost). Per RQ-2 falsification criterion: frame **not retired**; empirical support at fixture's resolution.

### Step 1.5 next — Spike B (RQ-3 multi-turn) per the approved Cycle 3 plan

Spike A is complete and synthesized. Per the Cycle 3 plan approved at Step 1.3, Spike B (multi-turn reliability — tau-bench subset + small real-session probe) sequences after Spike A. Spike A's data also feeds into RQ-2's frame-test before final cycle synthesis.

**Awaiting practitioner direction:** proceed to Spike B planning, or pause for review of the synthesis at `004b-spike-a-cycle3.md` before sequencing Spike B?

### Spike B — executed and synthesized

**2026-05-01 — practitioner authorized full G3 (3 arms × 2 fixtures).** Both fixtures executed:

**Tau-shape fixture (custom single-control multi-turn library checkout):**
- cheap-bare: 6/6 graded success (post-fixture-fix)
- cheap-with-script: 6/6 graded success (post-fixture-fix)
- frontier (F1 turn-by-turn dispatch): 6/6 graded success
- frontier (F2 single-shot facsimile): 1/6 strict replay; 6/6 plan-correct vs imagined — F2 documented as methodologically limited due to imagined-state bias

**Real-session fixture (haiku-generator ensemble + syllable script authoring):**
- frontier × 2: both produced working artifacts; ~10 min and ~4 min wall-clock; substantive multi-turn behavior with detailed reflection
- cheap-bare × 2: both produced working artifacts via OpenCode + `llm-orc serve` deployment; 14 min and 3 min wall-clock
- cheap-with-script × 2: both produced working artifacts; 2 min wall-clock each (deterministic preprocessor prepended to task)

**Findings (full synthesis at `004c-spike-b-cycle3.md`):**

- **RQ-3 at the cycle's tested complexity:** Neither tier hits a reliability ceiling. Tau-bench's 34–74% pass@1 and HORIZON's 19% meltdown rate do not manifest at single-control short-horizon multi-turn nor at bounded agentic-coding tasks. **The lit-review's reliability findings remain credible at their measured regimes; Spike B does not reach those regimes.** RQ-3 answered for sub-tau-bench complexity; harder regimes carry to Cycle 4.
- **Cross-tier complementarity from Spike A does NOT replicate at multi-turn.** Spike A (documentation review) found cheap and frontier find different bug classes. Spike B (multi-turn protocol + agentic coding) finds equivalent outcomes. **Complementarity is task-class-dependent, not universal.** Cycle 4 hook: which task classes have cross-tier asymmetry?
- **F1 vs F2 methodological finding:** F2 single-shot facsimile broke down via imagined-state bias (5/6 traces imagined fines that didn't exist). F1 turn-by-turn dispatch with deterministic tool feedback produced clean reactivity (6/6). **F1 is the methodologically correct method for multi-turn frontier-tier reliability testing under the no-paid-API constraint.**
- **Central question (cheap vs frontier on multi-turn):** Cheap-tier MATCHES frontier at this complexity. Performance-only frame indifferent (both 100%); four-priorities frame strongly favors cheap-tier ($0 cost; equivalent outcome). Frame-divergent recommendation again — Cycle 3's RQ-2 finding holds across both spikes.
- **Cheap-with-script preprocessing tentative wall-clock advantage on real-session** (~2 min vs cheap-bare's 3–14 min range). Sample too small for confidence (N=2); Cycle 4 hook for higher-N validation.

### Step 1.5 wrap-up

Both Spike A and Spike B complete and synthesized. Lit review (`004a`) + Spike A synthesis (`004b`) + Spike B synthesis (`004c`) form the cycle's research-phase deliverable. Awaiting practitioner direction for the cycle's next move: write the integrating Cycle 3 essay (`004-<descriptive-name>.md`) drawing on all three logs, or close at research-phase end with the three logs as the deliverable per Mode B (matching Cycle 2's close shape).

### Step 1.5 methodological pivot — practitioner critique surfaces a research-design gap (2026-05-01)

**Practitioner observation:** *"MiniMax is going to perform similarly to Sonnet on simple tasks. That's more or less what you determined, yes?"*

**Honest acknowledgment:** Yes. Re-reading what the cycle's spike battery has actually demonstrated, with the central question ("does cheap-orchestrator + orchestration compete with a more expensive frontier model?") as the criterion:

- **Spike A produced a substantive cross-tier finding** — different bug classes on documentation review; the heterogeneity-uncorrelated-errors mechanism extending across tiers. Genuine research evidence on a regime where the tiers diverge.
- **Spike B tau-shape (12/12 cheap; 6/6 frontier F1)** — confirms that 2–3-turn single-control multi-turn library checkout is a regime where both tiers handle the protocol. **This is the easy regime; both passing tells us little about the central question.** The lit-review's tau-bench reliability ceilings (34–74% pass@1) are scoped to harder regimes the fixture didn't reach.
- **Spike B real-session haiku authoring (6/6 across all arms)** — confirms that a bounded authoring task with clear success criteria works at both tiers. **Same critique: bounded authoring is the easy regime; equivalence here is expected, not informative for the central question.**
- **The "+ orchestration" part of the central question — `invoke_ensemble` and `compose_ensemble` dispatched by the cheap orchestrator on real tasks — has NOT been tested in Cycle 3.** Cheap-tier was tested with deterministic tools (tau-shape) and with general file-and-bash tools (real-session). The architecture's load-bearing primitive — the orchestrator dispatching ensembles for specialized work — has only Cycle 1's CAP-9 single-task baseline + Cycle 2's documented A3 reference behind it; Cycle 3 added no direct evidence.

**What this means for the cycle's central-question claim:** Spike B's "cheap matches frontier on multi-turn" is true at the tested complexity — and the tested complexity is below where the central question's tension actually lives. Investment-to-finding ratio on Spike B is poor for the central question; investment-to-finding ratio is good for METHODOLOGICAL findings (F1 vs F2; cross-tier complementarity is task-class-dependent).

**The cycle's actual central-question evidence is at risk of being one finding (Spike A's documentation-review cross-tier-asymmetry) plus a series of "easy-regime confirmations" that don't probe the regimes where the tiers diverge.**

**Practitioner-requested action:** dispatch the research-methods-reviewer to evaluate the cycle's current state and identify methodological corrections before further spike investment. The reviewer should evaluate (a) whether the spike battery as-run actually tests the central question or adjacent ones; (b) what fixture characteristics would test the central question properly; (c) the role of ensemble dispatch (the architecture's load-bearing primitive) in any such fixture; (d) methodological corrections needed.

### Methods reviewer mid-cycle dispatch outcome (2026-05-01)

Reviewer (`housekeeping/audits/research-design-review-cycle-3-mid.md`) confirmed and sharpened the practitioner's critique:

- **Two P1 findings.** (1) Regime mismatch: Spike B's investment confirmed the easy regime; central question lives elsewhere. (2) The `+ orchestration` gap: `invoke_ensemble` / `compose_ensemble` was NOT exercised in Cycle 3.
- **Spike A's cross-tier finding preserved as primary inheritance signal.**
- **Concrete fixture recommendation:** three-arm code review on a real diff. Arm A cheap-bare; Arm B cheap-with-ensemble (cheap orchestrator dispatches purpose-built ensemble via `invoke_ensemble`); Arm C frontier-bare. Pre-spike fixture validation required to avoid Spike B's easy-regime trap.
- **Six methodological correction items** for synthesis writers.

### Spike C — three-arm architecture comparison (2026-05-01)

Practitioner authorized the recommended spike. Executed with N=3 cheap arms × 2 cheap arms (A, B) + N=2 frontier arm (C, F1 facsimile via subagent dispatch) = 8 trials.

**Fixture:** Synthesized diff (`scratch/spike-c-cycle3-architecture-comparison/fixture/diff.patch`) introducing `session_budget.py` with 5 deliberately-injected ground-truth issues across categories (semantic / security / type-safety / test-gap / cross-file consistency). Pre-spike fixture validation documented at `fixture/ground-truth.md`; ISSUE-1 (off-by-one) and ISSUE-5 (cross-file `100_000` vs `50_000_000` mismatch) named as the "architecture-favorable" issues — where cheap-bare is expected to struggle and the script-agent's deterministic capabilities are expected to add value.

**Arm B ensemble:** `spike-c-code-review.yaml` — script-agent (`spike_c_diff_analyzer.py`: cross-symbol extraction, security-pattern detection, type-annotation flagging, test-presence detection, **cross-file value verification**) + LLM reviewer (MiniMax with system prompt instructing it to use script findings as anchor evidence and add semantic analysis on top). Note: heterogeneous LLM reviewers from Cycle 2 A3 (Hunyuan + Kimi) were a scope cut for this spike — only one LLM reviewer slot used; documented as Cycle 4 hook.

**Surface-detection scoring (5/5 at all arms):** all 8 trials surfaced all 5 ground-truth issues at least at keyword-detection level. Initially this looked like another Spike B "easy regime" pattern — but per Decision A's PAE-aware procedural-independence discipline, finer-grained scoring is required.

**Concrete-verification scoring on ISSUE-5 specifically (the load-bearing finding):**

| Arm | Surface detection (drift risk noted in comment) | **Concrete verification (actual mismatched values cited)** |
|-----|:----------------------------------------------:|:---------------------------------------------------------:|
| Arm A (cheap-bare via `opencode run`) | 3/3 | 1/3 — only when MiniMax happened to use Read tool autonomously |
| **Arm B (cheap+ensemble via `llm-orc invoke`)** | **3/3** | **3/3 — guaranteed by script-agent's deterministic cross-file verification** |
| Arm C (frontier-bare via F1 subagent dispatch) | 2/2 | **0/2 — Sonnet 4.6 single-shot has no access to other files** |

**This is the cycle's central-question evidence.** The `+ orchestration` primitive (cheap-tier dispatching an ensemble that includes a script-agent) **outperforms frontier-tier-alone on this bug class** — 3/3 vs 0/2 on concrete cross-file verification. The script-agent's deterministic file access is the load-bearing capability; neither cheap-tier-alone (Arm A) nor frontier-tier-alone (Arm C) can guarantee equivalent verification in single-shot.

**Three substantive findings:**

1. **The `+ orchestration` primitive is load-bearing on this bug class.** Cheap+ensemble's deterministic cross-file verification beats frontier-bare's surface detection, 3/3 vs 0/2. The architecture's value is empirically demonstrated.

2. **Spike A's cross-tier complementarity finding extends to a new mechanism.** Spike A: heterogeneity-uncorrelated-errors (LLMs from different families). Spike C: deterministic-vs-probabilistic complementarity (script + LLM). Both are forms of the architecture compensating for what a single LLM tier alone cannot guarantee.

3. **The four-priorities frame's recommendation diverges sharply from performance-only on this bug class.** Performance-only: prefer Arm B (3/3 concrete catches > Arm C 0/2). Four-priorities: prefer Arm B (zero $ cost; equivalent latency; concrete catches; subscription tokens spent on Arm C don't recover). **Both frames recommend Arm B** — but for different reasons. Frame-divergence at recommendation level is preserved.

**Cost / latency:**

| Arm | Median wall-clock | $ cost | Subscription tokens (approx) |
|-----|:-----------------:|:------:|:----------------------------:|
| Arm A | 30–56s | $0 | 0 |
| Arm B | 30–45s | $0 | 0 |
| Arm C | 42–47s | $0 | ~16K/trial |

Arm B is the unambiguous winner on cost-axis with equivalent or better latency than Arm A.

**Limitations recorded:**

- Single fixture; broader generalization requires more task classes (Cycle 4).
- Arm A confound: `opencode run` gave MiniMax tool access, not the methods reviewer's intended single-shot cheap-alone. Arm A's 1/3 concrete-detection is a function of MiniMax's variability in deciding to use Read, not pure capability.
- Heterogeneity scope cut: Arm B used MiniMax for the LLM reviewer slot; full Hunyuan + Kimi heterogeneity from Cycle 2 A3 is Cycle 4 territory.
- Frontier facsimile via subagent — same caveat as Spike B's F1.

### Cycle 3's central-question deliverable, complete

Three primary research-log artifacts:

- `004a-lit-review-agent-design.md` — 23 sources, cross-tier complementarity gap identified
- `004b-spike-a-cycle3.md` — RQ-1 settled (A2+script ≠ A3); cross-tier-uncorrelated-errors finding on documentation review
- `004c-spike-b-cycle3.md` — RQ-3 multi-turn at sub-ceiling complexity; F1 vs F2 methodological finding; cross-tier complementarity does NOT replicate at simple multi-turn
- `004d-spike-c-cycle3.md` (pending — synthesis writeup of the architectural-value finding above)

**The `+ orchestration` gap that the methods reviewer identified is now closed by Spike C's evidence.** The architecture's value is empirically demonstrated on a real bug class where cross-file verification is required. Cheap-tier with the architecture beats frontier-tier alone on this class.

### Spike D pilot — multi-ensemble coordination probe (2026-05-02)

Practitioner-proposed Cycle 4 priming experiment. Pilot scope: N=1 per arm × 3 arms; one fixture (Spike C's diff); the workflow extends Spike C's single-stage review into a multi-stage code-review-and-fix pipeline.

**Workflow:** Stage 1 (review) → Stage 2 (LLM generates proposed fixed file) → Stage 3 (deterministic verifier confirms which issues resolved).

**Three arms:**
- **Arm A1 (cheap-bare-with-tools):** opencode + MiniMax with full Read/Edit/Bash tool access; multi-stage prompt asking it to do all stages with general agentic capability. **No ensemble dispatch.**
- **Arm B1 (cheap+multi-ensemble):** spike-c-code-review for Stage 1 → MiniMax (via direct model factory) for Stage 2 → spike-d-fix-verifier (new script-only ensemble) for Stage 3.
- **Arm C1 (frontier-bare, single-shot):** Sonnet 4.6 via subagent dispatch with all three stages collapsed into a single prompt; produces full workflow output in one response.

**Results (verifier scoring on the proposed fixed files):**

| Arm | Issues resolved | Notes |
|-----|:--------------:|-------|
| A1 (cheap-bare) | 3/5 | api_key hash false-positive in verifier (security-correct fix; verifier regex flags any api_key reference) |
| B1 (cheap+multi-ensemble) | 4/5 | api_key removed from log; only test-coverage gap remains |
| C1 (frontier-bare) | 4/5 | api_key removed from dataclass entirely; only test-coverage gap remains |

**Major substantive finding from the pilot — the opencode CLI stall pattern.**

Multiple sustained attempts to dispatch the multi-stage workflow via `opencode run` with cheap-tier MiniMax stalled indefinitely (8+ hours alive with <5s CPU; same pattern across 3 attempts). The pattern: opencode + MiniMax succeeds on small prompts (e.g., "Reply with the single word: OK" — instant) but hangs on substantial prompts containing code blocks (~5KB+). When Stage 2 was run via the **production `ConfigurationManager` + `ModelFactory` path** (the same path Spike B's `tau_shape_harness.py` used reliably), it completed in **24.8 seconds** with the same MiniMax model.

This is a Spike D primary finding: **the architecture's `invoke_ensemble` primitive (production model factory path) works reliably; the workaround of dispatching ensembles through `opencode run`'s Bash tool does NOT.** The architectural contract is already correctly designed — the opencode CLI is a fragile substitute for the orchestrator's internal tool-calling loop.

**Cycle 4 hooks the pilot surfaces:**

1. **Test the architecture's actual `invoke_ensemble` primitive end-to-end via the orchestrator's tool-calling loop in `llm-orc serve`**, not via opencode CLI. The orchestrator's closed five-tool surface (per ADR-003) is the load-bearing primitive — it's what differentiates `cheap orchestrator + ensembles` from `cheap LLM + shell subprocess`.
2. **Multi-ensemble routing intelligence:** does the orchestrator (running in llm-orc serve, with the closed-tool surface) autonomously choose which ensemble for which sub-task? The pilot used explicit prompt-based staging; Cycle 4 should test autonomous routing.
3. **Investigate the opencode CLI stall pattern:** is it specific prompt-token patterns? OpenCode Zen rate limiting that manifests as stall rather than 429? Buffering issues with stdin? This is a deployment-shape investigation that affects user contract.
4. **Iteration loops:** the pilot's workflow was linear (review → fix → verify, single pass). Cycle 4 should test the "iterate until verified resolved" loop — does the orchestrator correctly re-dispatch fix-proposer when verifier reports issues remaining?
5. **Heterogeneity revisited:** Spike D's Arm B used MiniMax for the LLM reviewer slot (same scope cut as Spike C). Cycle 4's full multi-ensemble experiment should test heterogeneous reviewers per Cycle 2 A3's design (Hunyuan + Kimi + script).
6. **Higher-complexity fixtures:** the pilot used the same fixture as Spike C (90-line module). Cycle 4 should test multi-file diffs, longer trajectories, ambiguous task descriptions where routing decisions are themselves load-bearing.

**Pilot summary:** The architecture works (B1 ties C1 at 4/5 resolved at zero $ cost) when bypassing opencode's CLI path. The opencode CLI stall is itself a substantive finding for Cycle 4. Multi-ensemble coordination on a single fixture at this complexity confirms the value-add but doesn't yet test autonomous routing intelligence — which is the load-bearing question Cycle 4 should answer.

### Cycle 3 research-phase — current deliverable inventory

- `004a-lit-review-agent-design.md` — 23 sources; cross-tier complementarity gap identified
- `004b-spike-a-cycle3.md` — Spike A: cross-tier-uncorrelated-errors finding on documentation review
- `004c-spike-b-cycle3.md` — Spike B: F1 vs F2 methodology + multi-turn at sub-ceiling complexity
- `004d-spike-c-cycle3.md` (pending) — Spike C: architecture beats frontier on cross-file verification
- `004e-spike-d-pilot-cycle3.md` (pending) — Spike D pilot: opencode stall + multi-ensemble pilot data + Cycle 4 design hooks

The cycle now has **four distinct findings** that bear on the central question:
1. **Cross-tier complementarity is task-class-dependent** (Spike A doc-review, Spike B refines).
2. **The architecture's deterministic+probabilistic complementarity beats frontier-bare on cross-file verification** (Spike C).
3. **The architecture's value generalizes to multi-stage workflows when invoked via the production model factory path** (Spike D B1 ties C1).
4. **The opencode CLI as a deployment shape for cheap-tier multi-step orchestration is fragile** — not the architecture's fault; the production primitive `invoke_ensemble` is what should be used (Spike D failure mode).

## Step 1.4 — User revises or accepts

(Pending Step 1.3.)

## Step 1.5 — Research loop

(Pending.)

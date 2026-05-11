# Research Design Review — Cycle 3 (Agent Design) — Mid-Cycle Dispatch

**Reviewed question set:** Cycle 3 spike battery as executed (Spike A: `004b-spike-a-cycle3.md`; Spike B: `004c-spike-b-cycle3.md`; lit review: `004a-lit-review-agent-design.md`) evaluated against the central question recorded in memory `cycle-3-central-question`
**Constraint-removal response included:** yes (recorded in `research-log.md` §Step 1.2; ADR-003 bracketed)
**Date:** 2026-05-01
**Criteria applied:** 1–4 (ADR-082)
**Review mode:** Mid-cycle methodological correction — forward-looking, not retrospective grading

---

## Summary Assessment

The cycle's evidence base is better than the practitioner's critique implies on one dimension and worse on another. The "better" dimension: Spike A is a genuine finding. The cross-tier uncorrelated-bug-detection result — cheap-tier-with-directed-prompting and frontier-tier-direct find structurally different and largely non-overlapping bug classes on the README-review fixture — is novel empirical evidence the literature had predicted at the within-tier level (Sun et al. 2025; Ding et al. 2024) but had not directly tested cross-tier. That finding bears on the central question and should not be discounted in the sprint to correct course.

The "worse" dimension: the practitioner's critique is exactly right about Spike B. Both fixtures (tau-shape library checkout, haiku-generator authoring) were drawn from the easy regime by design — bounded, single-control, short-horizon, well-specified. The 100% pass rates across all arms are not surprising given what the lit-review said about those regimes. The cycle's central question lives at a different complexity level, and Spike B's investment went to confirming that below-the-ceiling regimes work, not to probing where the ceiling is or what happens to the cheap-vs-frontier comparison near or above it.

The larger structural gap is not about difficulty level — it is about the architecture itself. The central question is "does cheap-orchestrator **+ orchestration** compete with expensive frontier model?" The "+ orchestration" primitive (`invoke_ensemble`, `compose_ensemble`) has not been exercised in Cycle 3 on any task class. Spike A ran the cheap orchestrator against a static document with a prepended script report; Spike B ran it via OpenCode's tool surface on a bounded authoring task. Neither spike had the cheap orchestrator dispatching an ensemble to perform specialized sub-work. That architectural primitive — the one that distinguishes "cheap orchestrator as cheap alternative to frontier" from "cheap orchestrator as conductor of specialized components" — is untested in this cycle.

One further spike is warranted. The fixture should target the region where the cycle's central question actually lives: a task complex enough that cheap-tier-alone is expected to struggle, where the "+ orchestration" architecture gives the cheap tier a structural advantage that a bare frontier call cannot replicate at the same cost point. The spike's success criterion must be whether the architecture produces an outcome, not whether both tiers produce outcomes.

---

## Per-Criterion Findings

### Criterion 1 — Need-vs-Artifact Framing Applied to the Central Question

**P1 — The spike battery tested artifacts, not the need.**

The central question is: does cheap-orchestrator + orchestration compete with a more expensive frontier model on user-relevant outcomes? This is a need question. The answer would look like: here is a task where the deployment architecture's structural properties (specialized sub-agents, deterministic script verification, heterogeneous model combination) produce an outcome a single frontier model call cannot easily match at a comparable cost; and here is a task where that architecture fails to add value over the cheaper option.

What was tested instead:

- Spike A: Does a cheap orchestrator receiving a script report as prompt context match a heterogeneous ensemble on a documentation fixture? This is an artifact-anchored isolation question. It produced a finding that bears on the need (cross-tier asymmetry on a real task), but that is an incidental benefit of the artifact isolation test, not the test being designed to probe the central question.
- Spike B Fixture 1: Does a cheap orchestrator follow a library-checkout multi-turn protocol correctly? This tests the cheap-tier's protocol compliance on a task both tiers handle easily. The artifact being tested is the cheap orchestrator's tool-dispatch correctness. The finding (100% pass) answers a question about the artifact's basic competence, not about whether the architecture provides competitive user outcomes.
- Spike B Fixture 2: Can a cheap orchestrator author a haiku-generator ensemble + syllable script correctly? This tests a bounded authoring task. Both tiers produce working artifacts. The finding is "cheap tier is competent at bounded authoring" — correct but uninformative for the central question because the frontier tier is also competent at it, and the task does not require the ensemble-dispatch architecture at all.

**What a need-directed fixture looks like:** A task where the cheap-tier-alone is likely to either fail, degrade, or take substantially longer than the cheap-tier-with-orchestration. The architecture's load-bearing contribution must be what makes the difference, not the cheap tier's general competence. A concrete example: a code-review task on a real diff where the cheap orchestrator dispatches a specialized ensemble (static analysis script + semantic-review agent + security-pattern agent) and the frontier tier handles it as a single-shot call. The central question gets answered: does the architecture-augmented cheap tier match the frontier single-shot on outcome quality? If yes, the architecture earns its role. If no, the frontier's general capability advantage at single-shot is dominant.

**Recommendation:** Reframe the next spike's fixture around a task where architecture matters, not a task where both tiers' general capabilities are sufficient.

---

**P2 — The "100% pass" design in Spike B is structurally uninformative for the central question.**

Both the tau-shape fixture and the haiku authoring fixture were designed with clear success criteria and bounded scope. This is good experimental design for measuring reliability floors — it ensures the fixture has a definable pass/fail criterion. But the consequence is that both fixtures were pre-selected to be in the regime where both tiers pass. Spike B's data is therefore internally valid (the fixtures worked as designed) but externally invalid for the central question (the central question requires a fixture where the two sides of the comparison produce different outcomes, or at least approaches different outcomes as a function of architecture choice).

The lit-review explicitly named the complexity levels where the ceiling matters: tau-bench's 34–74% pass@1 (not 100%); HORIZON's 19% meltdown (not 0%). The cycle's Spike B avoided those regimes by choosing bounded tasks. That choice was pragmatic (controlled fixtures are easier to design), but the cost is that the resulting data is informative about "does cheap work here?" not about "where does cheap need the architecture to compete?"

---

### Criterion 2 — Embedded Conclusions in Remaining Spike Candidates

**P2 — The "harder multi-turn task" framing embeds a conclusion about where difficulty lives.**

The research log records that before the practitioner's critique, the cycle was considering a "harder multi-turn task" (multi-file feature flag) as the next spike direction. That framing presupposes that the way to probe the central question is to make the multi-turn task harder — increase turn count, add file complexity. This is a natural extension of Spike B's existing multi-turn fixture but it presupposes that turn count and file count are the difficulty axes that matter for the central question.

The embedded conclusion: "the central question's answer changes with task complexity, and task complexity is well-proxied by the size and length of the coding task." This may be true but is not the only axis. The architecture's load-bearing contribution is not just about handling longer sessions; it is about whether specialized sub-agents provide coverage the cheap orchestrator alone cannot provide — which is a capability-breadth question, not just a session-length question. A 30-turn haiku session would still not answer the central question if the cheap orchestrator alone can handle each turn; the missing component is the ensemble dispatch.

**Reformulation:** rather than "make the multi-turn task harder," the question should be "what task requires the architecture's specialized dispatch to succeed where the cheap orchestrator alone fails?" That reformulation opens the solution space to include single-turn tasks with structural complexity (requiring multiple specialized sub-agents) as well as multi-turn tasks.

**P2 — "Code review on a real diff" presupposes that ensembles will help cheap-tier on this task.**

If the next spike is designed around "code review on a real diff" with the cheap orchestrator dispatching an ensemble, the fixture design presupposes that ensemble dispatch will produce a measurable benefit on that task class. The prior evidence from Spike A (which was also a code-review-adjacent task) found that the cross-tier asymmetry was on semantic-consistency-across-sections, not on code-quality issues proper. There is no evidence from Cycle 3 that ensemble dispatch (rather than cross-tier comparison) produces better code-review outcomes on a real diff.

A stress-test framing that avoids presupposing outcome: "On a real diff with a known set of issues, compare (a) cheap-orchestrator-alone, (b) cheap-orchestrator dispatching a specialized ensemble, (c) frontier-bare. Does arm (b) recover issues arm (a) misses? Does arm (c) recover issues arm (b) misses, and vice versa?" This leaves all three arms' outcomes open and tests the architecture's contribution directly.

---

### Criterion 3 — Prior-Art Treatment of the Architecture Itself

**P1 — The cycle has treated the architecture as a constraint-to-defend rather than as prior-art-to-test.**

The cycle's central architectural primitive is: cheap orchestrator dispatches ensembles via `invoke_ensemble` and `compose_ensemble` (per ADR-001/002/003/007/011). This is the prior art whose value relative to a frontier single-shot the central question is testing. Prior-art treatment would require at minimum one spike that directly exercises this primitive under conditions where its presence vs. absence produces different outcomes.

What the cycle actually tested: Spike A tested the cheap orchestrator receiving a script report as context (not dispatching an ensemble). Spike B tested the cheap orchestrator dispatching OpenCode tools natively (not dispatching an ensemble via `invoke_ensemble`). Neither spike exercised the load-bearing primitive. The cycle implicitly relied on Cycle 1's CAP-9 baseline (which validated that the architecture works on a specific single task) as sufficient evidence that the primitive is functional, and then tested adjacent capabilities — multi-turn protocol handling, bounded authoring — that do not require the primitive.

The consequence: the cycle's evidence does not distinguish between "cheap orchestrator is a capable model that handles agentic tasks well" and "cheap orchestrator + ensemble dispatch architecture produces outcomes that compete with frontier by leveraging specialized sub-agent composition." The first claim is supported by Spike B. The second claim — the one the central question asks about — is not tested at all.

**What prior-art treatment of the architecture requires empirically:**

1. At least one spike that dispatches an ensemble via `invoke_ensemble` from the cheap orchestrator on a real task.
2. A comparison baseline: the same task with the cheap orchestrator alone (no ensemble dispatch) and with the frontier model alone.
3. A task class where ensemble dispatch is expected to add value — that is, where the cheap orchestrator alone is expected to either fail or produce lower-quality outcomes than the cheap orchestrator with specialized sub-agents.
4. Measurement of whether the architecture-augmented cheap tier actually produces better outcomes than cheap-alone, and whether those outcomes approach or match frontier-alone.

The CAP-9 baseline from Cycle 1 (referenced in the research log) is not sufficient for this purpose because it was a single-task validation of basic architecture functionality, not a head-to-head comparison with a frontier reference on outcome quality.

---

### Criterion 4 — Incongruity Surfacing

**P1 — Most spike investment confirmed the easy regime; the central question lives in a different regime.**

The cycle's lit-review named the regimes where ceilings matter: tau-bench's 34–74% pass@1; HORIZON's 19% meltdown; LongCLI-Bench's <20% pass on long-horizon. These numbers appear repeatedly in the research log's synthesis notes. The cycle's spike battery then designed two fixtures both of which hit 100% pass rates across all arms.

This is an incongruity between where the cycle's literature pointed and where the cycle's spike investment went. The investment pattern is: lit-review (correctly) describes the regime where the central question's tension lives → spike battery (incorrectly) avoids that regime and tests the easy regime instead. Spike B's 12/12 cheap-tier tau-shape result and 6/6 real-session result consume spike investment, produce clean data, and answer a question the lit-review had already suggested was going to be "both tiers handle this."

The cycle's one spike that touched a regime where the tiers diverge (Spike A: cross-tier asymmetry on documentation review) produced the cycle's most novel finding. That is not a coincidence. When the fixture is designed at a complexity level where both tiers' general capabilities are insufficient to handle all dimensions of the task, the architecture's contribution (or the frontier tier's general-capability advantage) becomes visible.

**P1 — The `+ orchestration` part of the central question is unprobed while the orchestrator-tier comparison was probed in detail.**

The central question has two parts: "cheap orchestrator + orchestration" vs. "expensive frontier model." The spike battery probed the "cheap orchestrator vs. expensive frontier model" comparison across multiple fixtures (Spike A arms 1–7; Spike B tau-shape and real-session). The "+ orchestration" half — ensemble dispatch, composed sub-agents, multi-component specialized review — was not tested in any spike. This is a structural incongruity: the architecture's load-bearing primitive is the half that was skipped.

The two most informative things the cycle could learn are: (a) does the architecture's ensemble-dispatch component produce measurable value over cheap-orchestrator-alone on tasks where cheap-orchestrator-alone struggles? and (b) if it does, does the architecture-augmented cheap tier compete with frontier-alone? The cycle has extensive evidence on neither question. All of Spike B's evidence answers a third question: "does cheap-orchestrator-alone match frontier-alone on easy tasks?" — which is not what the central question asks.

**P2 — The F1 vs. F2 methodological finding is the most reusable output of Spike B, but it cost the same as an outcome finding would have.**

Spike B's substantive methodological output — that F2 single-shot facsimile testing breaks down via imagined-state bias and F1 turn-by-turn dispatch is the correct method — is a genuine and reusable finding. It will be valuable in Cycle 4. But it cost the same spike investment as a finding on the central question would have cost, because the fixtures used to produce it were not designed to test the central question. This is not a design failure in retrospect — the F2 breakdown was unanticipated — but it reinforces that the fixture design for the next spike should be forward-looking on the central question, not a natural extension of Spike B's existing fixture shape.

---

## Concrete Fixture Recommendation

### One further spike is warranted. Here is the fixture.

**Fixture name:** "Code review on a real diff — three-arm architecture comparison"

**Rationale for this task class:** Code review on a real diff is the cycle's closest analog to a production agentic-serving task. It is not a synthetic benchmark fixture; it uses the corpus's own codebase as the test material. It has a ground-truth issue set (the cycle can use a known PR diff or a synthetic diff with injected issues). It is a task where:

1. A cheap orchestrator alone may miss issues requiring semantic depth, security pattern recognition, or cross-file reasoning.
2. An ensemble of specialized sub-agents (a security-pattern agent, a semantic-consistency agent, a test-coverage agent) can divide the review scope and apply specialized attention.
3. A frontier model applying single-shot review has genuine general-capability advantages.

This task class puts all three configurations in genuine competition, not just in a regime where all three succeed.

**Three arms:**

- **Arm A — cheap-bare:** Cheap orchestrator (MiniMax M2.5 Free via OpenCode Zen) reviews the diff directly via `invoke` single-shot with the project's `code-review.yaml` default task. No ensemble dispatch; cheap tier working alone.
- **Arm B — cheap-with-ensemble:** Cheap orchestrator dispatches a purpose-built review ensemble via `invoke_ensemble`. The ensemble should contain at minimum: (1) a deterministic script-agent that extracts changed symbols, counts test additions vs. deletions, flags obvious error-path omissions, and identifies API surface changes; (2) a semantic-consistency LLM reviewer (cheap-tier); (3) a security/type-safety LLM reviewer (cheap-tier). Cheap orchestrator receives the ensemble outputs and synthesizes the review. This arm directly tests the `+ orchestration` primitive.
- **Arm C — frontier-bare:** Frontier model (Sonnet 4.6 via F1 dispatch, single-shot subagent) reviews the diff directly with the same task prompt. No ensemble.

**Why these three arms:** Arm A vs Arm B tests whether ensemble dispatch adds value over cheap-tier-alone. Arm B vs Arm C tests whether the architecture-augmented cheap tier competes with frontier. Arm A vs Arm C is the simple capability-gap measurement. All three comparisons are needed to answer the central question.

**Failure modes this fixture should expose:**

1. Arm A misses issue categories that Arm B catches (confirms architecture adds value for cheap tier).
2. Arm C catches issue categories neither Arm A nor Arm B catches (frontier's general capability advantage is real and not closed by the architecture).
3. Arm B catches issue categories Arm C misses (the architecture's heterogeneous-reviewer cross-tier mechanism from Spike A extends to code review).
4. Arms A, B, and C all catch the same issues (the architecture adds no value over the cheap tier alone, and the cheap tier matches frontier — which would be informative about this task class but is not the predicted outcome given the diff complexity).

**Success criterion:** The spike succeeds if it produces a findable difference between at least two arms on issue-category coverage. It fails methodologically if all arms produce equivalent coverage — not because the finding is wrong, but because the fixture was too easy (same critique as Spike B). The fixture design must pre-verify that Arm A is expected to struggle on at least some issue category before the spike runs.

**Pre-spike fixture validation:** Before running live trials, manually audit the diff for the issue categories expected to appear. If the diff has no issues in the categories where cheap-orchestrator-alone is expected to struggle, the fixture should be rejected and a more complex diff selected. This avoids repeating Spike B's dynamic where the fixture was easy by design.

**Instrumenting Arm B — ensemble design notes:** The ensemble for Arm B should use the architecture's actual primitives. This means authoring a new ensemble YAML under `.llm-orc/ensembles/` that the cheap orchestrator can dispatch via `invoke_ensemble`. The script-agent in the ensemble should run static checks (changed-symbol extraction, test count, error path presence) that are not subject to LLM hallucination — directly parallel to Spike A's deterministic analyzer but applied to a diff rather than a documentation file. The two LLM reviewers in the ensemble should be from different families (heterogeneity per Cycle 2 A3's architecture) to test whether the cross-tier mechanism found in Spike A replicates within-ensemble on code review.

**Trial count:** N=3 per arm is the cycle's established norm. At N=3 × 3 arms, this is 9 total trials — comparable to Spike B's trial count. Frontier-tier arm runs N=2 per F1 methodology (subscription-token consideration as established in Spike A).

**Harness reuse:** Spike A's harness (`spike_a_harness.py`) and script template (`deterministic_analyzer.py`) are reusable as structural templates. Spike B's F1 runner (`f1_runner.py`) is the correct frontier dispatch method per Spike B's F1-vs-F2 methodological finding. These artifacts are retained per the corpus retention policy and should be used as starting points.

**Connection to central question evaluation:** At synthesis time, apply the central question's two tests directly:

1. Does cheap-orchestrator-with-ensemble (Arm B) produce equivalent or better issue coverage than frontier-bare (Arm C)? If yes: the architecture competes. If no: frontier's general-capability advantage is not closed by the architecture on this task class.
2. Does cheap-orchestrator-with-ensemble (Arm B) produce better issue coverage than cheap-orchestrator-alone (Arm A)? If yes: the `+ orchestration` primitive adds value. If no: the architecture's overhead is not justified on this task class.

Both questions must be answered in the same spike for the central question to be addressed. A spike that answers only one of the two is incomplete evidence.

---

## Alternative: Close the Cycle and Carry to Cycle 4

There is a case for closing Cycle 3 and carrying the central question to Cycle 4 as the primary entry question. The argument:

- Cycle 3 has produced genuine findings on two of its three named RQs (RQ-1 settled at fixture scope; RQ-2 supported with frame-divergent recommendations in both spikes).
- RQ-3 is answered for sub-ceiling complexity; the harder regimes are legitimately Cycle 4 territory regardless.
- The central-question gap — `invoke_ensemble` untested — is a genuine gap, but it is a question about a specific architectural primitive that could be Cycle 4's primary entry question rather than a late-stage correction in Cycle 3.
- A Cycle 4 entry designed around "does cheap-orchestrator + ensemble dispatch compete with frontier on tasks where cheap-alone struggles?" would be a cleaner, better-scoped research cycle than Cycle 3 extended with one more spike on a revised central question.

The case against closing: Cycle 3's kit (harnesses, diff fixture templates, F1 dispatch methodology, corpus retention) is in place. The spike described above could run in a single session. Carrying the untested primitive to Cycle 4 means Cycle 3's central question is formally unanswered — it has one supporting finding (Spike A's cross-tier asymmetry) and a series of adjacent findings that do not address the central question. The cycle archive would record an unanswered central question, which is a less useful inheritance signal than a direct answer.

**Recommendation:** Run the spike. The cost is low (free-tier MiniMax + F1 subscription tokens for frontier arm), the harness infrastructure is in place, and the fixture design (code review on a real diff) is natural. Close Cycle 3 with a direct answer to the central question, or with a clear statement of why the architecture's contribution cannot be tested on this fixture. Either outcome is a better inheritance signal for Cycle 4 than an unanswered central question.

If the practitioner's appetite for one more spike is firm, this is the spike to run. If appetite is not firm, close the cycle and frame the Cycle 4 entry around the central question with a clean mandate.

---

## Methodological Correction Items

These apply to synthesis writers for both the Cycle 3 essay and Cycle 4 research entry.

**1. Easy-regime confirmation is not central-question evidence.**

Spike B's "100% pass at both tiers" is correct data and should be reported accurately, but it should not appear in the cycle summary as evidence for or against the central question. The data answers: "at sub-ceiling complexity, cheap-tier matches frontier." That is a scope-bounded finding about a specific complexity regime, not a finding about whether the deployment architecture competes with frontier on user-relevant outcomes. The synthesis should be explicit about this scope boundary rather than allowing the clean result to imply broader support for the central question.

**2. The `+ orchestration` gap should be named explicitly in the cycle archive.**

If the cycle closes without running the recommended spike, the cycle archive (inherited by Cycle 4) should record the gap explicitly: "The central architectural primitive (`invoke_ensemble` dispatched by the cheap orchestrator to perform specialized sub-work) was not tested in Cycle 3. Cycle 1's CAP-9 baseline is the only evidence for this primitive's functional validity; no head-to-head comparison with a frontier reference on outcome quality was conducted in Cycles 2 or 3." This prevents Cycle 4 from inheriting the gap silently.

**3. "Both tiers pass" is evidence for the regime boundary, not for the central question.**

The correct synthesis framing for Spike B's results is: "Both tiers handle short-horizon single-control multi-turn protocol and bounded agentic-coding authoring at 100% pass rate. This locates the 'cheap matches frontier' boundary below the regimes where the lit-review predicts ceilings. The central question's evidence requires a fixture above this boundary." This framing is more useful than "cheap competes with frontier on multi-turn work" because it makes clear that the tested regime was selected below the regime where competition matters.

**4. The cross-tier complementarity finding from Spike A is the cycle's primary inheritance signal.**

Spike A's documentation-review finding — that cheap-tier-with-directed-prompting and frontier-tier-direct find uncorrelated bug classes — is the finding that most directly bears on the central question. It suggests that the architecture's value is not in replacing the frontier tier but in complementing it: the two tiers find different things, and the combined deployment captures both. This is a genuinely useful architectural conclusion. The synthesis should lead with this finding as the primary evidence rather than treating Spike B's clean results as co-equal support.

**5. Do not use N=2 or N=3 results to make reliability-ceiling claims.**

Spike B's clean results are useful for confirming that both tiers operate in the easy regime. They are not sufficient for making reliability-ceiling claims about the architecture. A reliability ceiling claim requires testing at the complexity levels where failures occur, not at complexity levels where failures do not occur. Any synthesis that reads Spike B's results as "the architecture is reliable at multi-turn agentic work" is overgeneralizing from below-ceiling data.

**6. Distinguish "cheap tier is capable" from "cheap tier + architecture competes with frontier".**

These are two separate claims and both spikes support only the first. A capable cheap tier is a necessary condition for the architecture to compete with frontier, but it is not sufficient — the architecture's ensemble-dispatch primitive must also add value above what the cheap tier alone provides. The synthesis must keep these two claims distinct to avoid conflating the cheap tier's general capability (well-supported by Spike B) with the architecture's competitive value (untested in Cycle 3).

---

## Confidence and Limits of This Review

**What this review can assess:** The match between the spike battery's fixture designs and the central question's informational requirements. The structural gap between what was tested and what the central question requires. The fixture characteristics needed to address the gap. These assessments are grounded in the cycle's own documents — the central question reframe (memory: `cycle-3-central-question`), the spike synthesis notes, the lit-review's regime descriptions, and the practitioner's verbatim critique.

**What this review cannot assess:**

- Whether the recommended fixture (code review on a real diff with ensemble dispatch) will produce a clean finding. The fixture's outcome depends on the diff's issue composition, the cheap orchestrator's behavior on that specific diff, and whether the ensemble's sub-agents actually divide the review scope as anticipated. Pre-spike fixture validation is the mitigation, not pre-assessment.
- Whether `invoke_ensemble` from the cheap orchestrator will produce a qualitatively different output than the cheap orchestrator-alone on code-review tasks. Spike A's cross-tier finding is evidence that ensemble dispatch produces uncorrelated bug coverage; whether it replicates in the recommended fixture is an empirical question.
- Whether the correct interpretation of Spike B's 100% pass results is "the cycle tested the wrong regime" (this review's reading) or "the cycle correctly established the regime boundary and the central question's answer should be inferred from this boundary." A defensible counter-reading would argue that knowing the regime where cheap matches frontier is itself a central-question finding, because it establishes the deployment conditions under which the cheap tier is sufficient without ensemble overhead. The counter-reading is wrong about the architecture's load-bearing primitive (which is still untested), but it is not wrong about the regime-boundary finding's value.
- Whether the practitioner has appetite for further spike investment or prefers to close and carry to Cycle 4. This review's recommendation is "run the spike" but that recommendation is methodological, not operational. The cycle-closure vs. extension decision is the practitioner's.

**Reviewer framing caution:** This review was conducted after reading the cycle's research log sequentially, which means the practitioner's critique (Step 1.5 methodological pivot) was the final framing context before the review. The critique is well-founded and this review largely agrees with it. A reviewer who had not read the practitioner's critique might have surfaced a different balance of findings — possibly more credit to Spike B's methodological output (F1 vs. F2 finding) or more emphasis on Spike A's novel cross-tier finding as the cycle's primary contribution. The review should be read as substantially aligned with the practitioner's critique, which means it may under-weight evidence that argues against the critique's framing.

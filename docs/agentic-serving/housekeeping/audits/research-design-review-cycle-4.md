# Research Design Review

**Reviewed question set:** `docs/agentic-serving/essays/research-logs/research-log.md` Step 1.1 (primary question + six sub-questions)
**Constraint-removal response included:** Yes (Step 1.2)
**Date:** 2026-05-04

---

## Summary

- **Questions reviewed:** 1 primary + 6 sub-questions (Sub-Q1 through Sub-Q6, where Sub-Q6 is the new question surfaced by the constraint-removal response) + the constraint-removal response, evaluated as one question set under review
- **Flags raised:** 7
- **Criteria applied:** 1–4 (ADR-082 full set)
- **Cycle-specific concerns tested:** all five named in the dispatch

---

## Per-Question Review

### Primary Question: "What design methods does the cheap-orchestrator + ensemble pattern need in order to extend from manually-staged multi-stage workflows (Spike D) toward an agentic system capable of driving arbitrary coding sessions over long horizons — with 'running a full RDD cycle using the agentic-serving flow itself' as the concrete capability benchmark — and which of those design methods are properties of ensembles specifically versus orchestration more generally?"

**Belief-mapping:** The primary question is well-mapped to the cycle's entry commitments. The belief it requires is that (a) there exists a set of design methods that will extend the pattern across the capability gap, and (b) the pattern is worth extending rather than replacing with a different architecture. Both of these are held with explicit scope conditions in Step 1.1's "What is already known or assumed" block, which names failure modes that would change the approach. The "What would change the approach" block genuinely engages with the possibility that orchestration alone suffices or that the four-layer architecture is structurally insufficient. This is substantive, not pro-forma, belief-mapping work already done at entry.

What the framing does not surface: the belief that a single set of design methods accounts for the gap. The capability extension from manually-staged pipelines to long-horizon autonomous coding likely requires different mechanisms at different points in the capability gap, and a question asking for "the design methods" may cause the research to land a unified architecture answer where the empirical answer is fragmented. A complementary framing: "At which points in the capability gap does the pattern fail first, and what do those failure modes reveal about which mechanisms are missing?" This surfaces the research as diagnostic-first rather than prescriptive-first.

**Embedded conclusions:** The phrase "what design methods ... need" is mildly directional — it presupposes that design methods are the right intervention category. The alternative is that the architecture's current shape is sound and what is needed is implementation depth rather than new methods. This is a soft presupposition, and the "What would change the approach" block partially releases it by naming the possibility that "something beyond orchestrator + ensembles + scripts + plexus is needed." The presupposition does not need reformulation, but it should be carried through to synthesis as a live alternative.

**Scope:** Appropriate. The primary question is load-bearing without over-specifying method.

---

### Sub-Q1: "What are the qualitatively distinct demands a 'long-session coding driver' must meet that bounded multi-stage pipelines (Spike D) do not?"

**Belief-mapping:** The question assumes the demands are "qualitatively distinct" — implying there are meaningful categorical differences, not just quantitative scaling. The more productive belief to surface: the demands may be quantitatively continuous with Spike D's demands, and what appears to be a qualitative gap may dissolve if Spike D is pushed further. Whether the gap is categorical or continuous is itself a research question that Sub-Q1 does not examine, because its framing commits to the categorical reading before the research runs.

Adjacent question the framing excludes: "At what point does Spike D's manually staged pipeline break under extended repetition, and what does that break reveal about which demands are structurally missing versus which are just untested?"

**Embedded conclusions:** "Qualitatively distinct" is a mild embedded conclusion. The reformulation would be: "What demands does a long-session coding driver surface that Spike D did not exercise — and are these demands categorically new or scaling extensions of Spike D's demands?"

**Scope:** Appropriate as a decomposition sub-question. The embedded-conclusion risk is mild.

---

### Sub-Q2: "For each demand surfaced in (1), which mechanism does the supplying work: cheap-orchestrator alone, orchestrator + scripts as deterministic tools, orchestrator + ensembles, ensembles composed in pre-defined chains, ensembles composed dynamically by the orchestrator, or something not yet in the four-layer architecture?"

**Belief-mapping:** Sub-Q2 is the mechanism-attribution question and is the best-designed question in the set. The list of candidate mechanisms is genuinely varied, includes a "something not yet in the architecture" option, and leaves the distribution of demands across mechanisms as an empirical question. The belief required is that one of the listed mechanisms does the primary work for each demand — which is almost certainly true, since the list covers the full space from "no ensembles" to "generative construction." The only belief worth surfacing: that mechanisms are separable, i.e., that one mechanism is identifiably primary for each demand. In practice, mechanisms may be jointly necessary, and attribution to a single mechanism may force the research to produce a cleaner answer than the data supports.

**Embedded conclusions:** None flagged.

**Scope:** Appropriate. This question is the cycle's sharpest contribution to mechanism isolation, which is inherited grounding action 1.

---

### Sub-Q3: "At what design-method level is the cheap-orchestrator + ensemble pattern's value located? Is it primarily (a) ensembles as compositional units of cognitive labor, (b) the orchestrator's routing-and-summarization discipline, (c) the script-models layer's deterministic guarantees, (d) cross-layer composition, or (e) some emergent property of the four-layer architecture as a whole?"

**Belief-mapping:** Sub-Q3 is valuable but poses an identification challenge. The five candidate locations are listed as alternatives, but the cycle's existing evidence (Spike C's Arm B result, the susceptibility snapshot's grounding action 1) already suggests that (c) — the script-models layer's deterministic guarantees — is more directly evidenced than (b) the orchestrator's routing decision, for the one task class tested. That prior creates a risk: the research enters Sub-Q3 with a candidate already in a stronger epistemic position, and the question's "is it primarily X or Y" framing may select that candidate prematurely rather than genuinely interrogating all five.

Adjacent question the framing excludes: "Under what task conditions does each candidate location become primary, rather than which is primarily load-bearing across all tasks?"

**Embedded conclusions:** None formally embedded. The question holds all five options open.

**Scope:** Appropriate, with one concern noted under Orthogonal-Axis Scoping below.

---

### Sub-Q4: "For ensembles specifically, what composition shapes are candidate design methods? [...] The research should compare these against the long-session benchmark, not presuppose stock+accumulation."

**Belief-mapping:** Sub-Q4 explicitly names six candidate shapes and adds a seventh (no-ensembles baseline). The explicit instruction not to presuppose stock+accumulation is a strong design move that releases the most likely embedded conclusion in advance. The belief required is that composition shape is a separable design variable — that "which shape" is a meaningful choice given a fixed task class and orchestrator capability. The risk is that shape choice is dominated by task-class fit in ways that make the comparison ill-defined across the full long-session benchmark (different phases of an RDD cycle may require different shapes, making a single shape comparison category-mistaken).

Adjacent question: "For which phases of the RDD cycle does shape choice matter most, and for which phases is shape choice dominated by other variables (e.g., context window, available model tier, latency budget)?"

**Embedded conclusions:** None. The candidate list is genuinely varied.

**Scope:** Appropriate. See Composition-Shape Openness note below.

---

### Sub-Q5: "What does 'running a full RDD cycle using the agentic-serving flow itself' demand at the design-method level? Each phase (research, model, decide, architect, build, debug, refactor, review, play, synthesize) has a distinct cognitive surface. Which phases are most demanding for the cheap-orchestrator + ensemble pattern, and what design methods are needed at the hardest points?"

**Belief-mapping:** Sub-Q5 is the North-Star benchmark's primary purchase. It is framed as an analytical decomposition — "what does it demand" — rather than a direct test. This is a productive framing for a lit-review and a demand-mapping exercise, but as the sole question covering the benchmark, it leaves a gap: analytical decomposition of what a phase demands is not the same as observing what the agentic-serving flow does when it runs that phase. The belief that analytical decomposition is sufficient purchase on the benchmark may be the cycle's most consequential scope assumption.

The cycle-specific concern test is whether Sub-Q5's method is sufficient to load-bear the North-Star benchmark. It is not, on its own. Analytical decomposition will produce a phase-by-phase demand map; it cannot surface whether the agentic-serving flow's actual behavior matches that map, where it breaks first under live conditions, or what failure modes emerge that the demand map did not anticipate. The benchmark is a behavioral target, and the question set has no behavioral sub-question against it.

**Flag — North-Star benchmark loading:** Sub-Q5 is necessary but not sufficient for the North-Star benchmark. A complementary direct-test sub-question is missing: something like "In a focused dogfood spike, run one RDD phase (e.g., the research phase) using the agentic-serving flow, and compare the agentic-serving flow's actual behavior to Sub-Q5's analytical demand map for that phase." The absence of a behavioral purchase on the benchmark means the cycle can produce a demand map that is analytically complete but empirically unvalidated. If the conditional spike (Step 1.3's plan) focuses on Sub-Q6 rather than Sub-Q5, the benchmark exits the cycle with no behavioral test at all.

**Embedded conclusions:** "Which phases are most demanding" assumes demand varies enough across phases to produce a clear ranking, which may not be the case if the binding constraint (e.g., context growth) operates uniformly across phases.

**Scope:** Too narrow as the sole benchmark question. The analytical decomposition method is appropriate for what it covers; what is missing is a behavioral complement.

---

### Sub-Q6: "What harnesses can enforce ensemble selection — or ensemble dispatch more generally — when context growth degrades orchestrator judgment? At which decision moments does the failure mode that RDD's structural hooks address transfer to ensemble routing, and at which does it not?"

**Belief-mapping:** Sub-Q6 explicitly names a scope condition at entry — the transfer from phase-transition decisions to ensemble routing is "plausible but needs scope-condition discipline." This is the right setup. The belief required to make Sub-Q6 productive is that the failure mode is real under the cheap-orchestrator + ensembles pattern at long-session length — which has not been empirically established in this corpus. The cycle's evidence for context-growth degradation of ensemble routing is zero; it is imported from the RDD corpus by analogy. Sub-Q6 might be investigating a failure mode that does not occur in this architecture, in which case any harness design would be over-engineering against a ghost.

Adjacent question: "Before designing harnesses, establish whether context growth actually degrades the orchestrator's ensemble-routing judgment in this architecture at the session lengths the benchmark requires." This is a prior sub-question Sub-Q6 implicitly assumes is settled.

**Embedded conclusions:** The framing "what harnesses can enforce ensemble selection" presupposes harnesses are the right intervention class. If context-growth degradation of routing judgment doesn't transfer to this architecture, the harness question is the wrong question. A reformulation: "Does context growth degrade the cheap orchestrator's ensemble-routing judgment in this architecture at long-session lengths, and if so, what intervention class (harnesses, context management, architecture changes) is most appropriate?"

**Scope:** Appropriate as a research question but may be one research step ahead of where the evidence is. The scope-condition note in Step 1.2 partially addresses this, but the question's text doesn't carry the scope condition forward — it asks "what harnesses" rather than "whether and what harnesses."

---

## Constraint-Removal Response Review

**Response substance:** Engaged — not performative. The practitioner addresses both brackets (the full architecture removed vs. `invoke_ensemble` alone removed) with substantive reasoning about the local-first value the architecture exists to deliver, and surfaces a genuine new sub-question (Sub-Q6) from the cross-corpus transfer observation. The broader bracket (architecture removed) receives a brief but clear rejection: "just using a cheap orchestrator only defeats the purpose." The tighter bracket (`invoke_ensemble` removed) receives a more nuanced treatment: "gets us part of the way there, but it means we have to have shipped a variety of ensembles." Both are genuine responses, not deflections.

The response's most substantive contribution is the local-first value articulation: the architecture's purpose is "offloading to our local machine's smaller models as much as we can," and any design method that recovers capability by pushing more work to the cloud orchestrator fails that purpose even if it is technically adequate. This is a value-grounded rejection of the bare-orchestrator case that is more coherent than a technical rejection would be.

**Embedded conclusions in the response:** One finding deserves scrutiny. The response treats the RDD structural-hooks finding as a candidate mechanism for the ensemble-selection reliability problem: "As context grows an agent is less likely to use its own judgement to follow a directive. So I wonder what harnesses can enforce ensemble selection." This formulation moves quickly from "I wonder if the failure mode transfers" to "what harnesses." The scope-condition note in Step 1.2's synthesis (Commitment 3) names this explicitly — "whether the failure mode transfers is itself a research question" — which is the correct handling. However, Sub-Q6's text asks "what harnesses" rather than "whether harnesses," partially re-embedding the conclusion the Commitment 3 note was meant to release. The constraint-removal response itself does not embed the conclusion; the subsequent Step 1.1 Sub-Q6 formulation partially does. See the Sub-Q6 flag above.

**Prior-art treatment in the response:** The response treats the four-layer architecture as prior art by engaging both brackets — it names what would be lost without the full architecture, names what would be missing with `invoke_ensemble` alone, and arrives at a position on why the four-layer shape earns its complexity. This is structural prior-art treatment. The third criterion (prior-art treatment) is satisfied by the constraint-removal response; no separately artifact-bracketing question in the question set is needed.

---

## Question Set Assessment

### Premature Narrowing / Prior-Art Treatment

**Prior-art criterion:** Satisfied by the constraint-removal response (Step 1.2). The response directly imagines the architecture away in both brackets and reasons about whether the architecture earns its presence. This is the structural function of constraint-removal, and it is executed substantively here. No separate flag.

**Premature narrowing — inherited confidence:** The question set as a whole inherits from Cycle 3's susceptibility snapshot three open questions that Cycle 4 is supposed to address at research entry before advancing new architectural claims. Sub-Q2 (mechanism attribution) and the "What is already known or assumed" block both name the grounding actions explicitly. However, the question set's overall framing — particularly Sub-Q3's "at what level is the pattern's value located" and Sub-Q5's RDD-cycle decomposition — proceeds as if the pattern's value is established and the work is locating and extending it. The possibility that Grounding Action 1 (mechanism isolation) finds that the mechanism is entirely in the script layer, not the ensemble routing layer, would significantly reshape Sub-Q3's candidate list, but the question set does not build in a decision gate: "if mechanism isolation finds X, then Sub-Q3 and Sub-Q4 are reshaped in Y direction." Without a gate, the research plan (Step 1.3) runs Sub-Q3 and Sub-Q4 work in lit-review #2 in parallel with the mechanism-isolation work, potentially producing answers to the wrong question.

**Flag — partial premature narrowing:** The question set lacks an explicit gate between the three inherited grounding actions (Sub-Q2 territory) and the ensemble-design-space questions (Sub-Q3 and Sub-Q4 territory). If grounding action 1's mechanism-isolation result comes back as "the script layer is load-bearing, not the ensemble routing layer," the composition-shape design space in Sub-Q4 narrows dramatically, and Sub-Q3's option (c) effectively wins before the lit-review #2 work runs. The research plan should make the grounding-action findings gating conditions on how much scope the Sub-Q3/Sub-Q4 work gets, rather than running them in parallel as equally-weighted lit-review scope.

---

### Incongruity Surfacing

**Pattern in the research context:** Essay 003 (Cycle 2) contains a finding from Spike A3 that is structurally adjacent to what Cycle 4 is designing toward, and the adjacency creates an incongruity the question set does not surface.

In Spike A3, the MARG-concatenation aggregation pattern was enforced by a Python harness — not by the orchestrator's compliance with a "do not synthesize" instruction, but by a deterministic script that structurally prevented collapse. This is a direct implementation of a "harness that enforces a composition decision" for a case where LLM judgment under context would produce the wrong output (cascade-collapse). The harness solution is simple: a Python script wrapping the ensemble invocation enforces the aggregation rule deterministically.

Sub-Q6 is now asking what harnesses can enforce ensemble selection when context growth degrades orchestrator judgment. This is a structurally similar problem — enforcing a dispatch decision the orchestrator's judgment may fail to make reliably — but is framed as a novel research question rather than as an extension of Spike A3's already-demonstrated pattern.

The incongruity: a simple script-harness solution for enforcing a composition decision exists one essay prior in the corpus (Spike A3's MARG-concatenation harness), while Sub-Q6 is designed as a research question about "what harnesses can enforce ensemble selection." The question treats the problem as open when the Spike A3 pattern — a deterministic script that wraps the ensemble invocation and routes to the correct ensemble based on decision-point signals — is a direct solution candidate that is not named in the question set.

**Flag — missing incongruity surfacing:** Sub-Q6 should surface this adjacency explicitly. The question "what harnesses can enforce ensemble selection" is missing the adjacent observation that Spike A3 already demonstrated a harness enforcing a composition decision via a deterministic Python wrapper. The candidate solution may be as simple as "the same harness pattern applied to the dispatch decision rather than the aggregation step." Whether it is or not should be examined, not assumed either way — but it should be examined. The question set as written does not ask the researcher to check whether Spike A3's pattern covers Sub-Q6's territory.

A complementary sub-question the set is missing: "Does Spike A3's deterministic-wrapper harness pattern generalize to ensemble routing enforcement, or does ensemble routing's continuous-judgment character require a different intervention class than aggregation enforcement's one-shot character?"

---

### Cycle-Specific Concerns

**1. Cross-corpus transfer scope condition (Sub-Q6 framing)**

The scope condition is operative in Step 1.2's Commitment 3 synthesis but partially erodes in Sub-Q6's text formulation. Commitment 3 correctly names: "The decision class differs. Whether the failure mode transfers is itself a research question." Sub-Q6's text asks "What harnesses can enforce ensemble selection" — the "can" formulation assumes the failure mode is real enough that harnesses are worth designing. The second half of Sub-Q6 recovers: "At which decision moments does the failure mode that RDD's structural hooks address transfer to ensemble routing, and at which does it not?" — this is the right formulation. But the question opens with the harness-design framing and closes with the scope-condition test, which inverts the logical priority. The scope condition should be the first question, not the second.

The scope-condition is named and operative, but its placement within the sub-question itself gives the harness-design framing more implicit weight than the transfer-test framing.

**2. North-Star benchmark loading (Sub-Q5 method)**

As flagged under Sub-Q5: the analytical decomposition method is necessary but not sufficient for the North-Star benchmark. The research plan (Step 1.3) covers Sub-Q5 via "Analytical decomposition — RDD-cycle phase-by-phase demand mapping against architecture layers" with no behavioral complement. The conditional spike covers Sub-Q6 deepening if Sub-Q6 lands thin; there is no conditional spike for Sub-Q5 validation if the demand map lands rich but untested.

The benchmark "drive a full RDD cycle using the agentic-serving flow itself" is behavioral. A demand map answers "what would it need to do"; a behavioral test answers "what does it actually do." The cycle plan has one and not the other. The risk: the cycle closes with an analytically complete but empirically unanchored demand map, and the benchmark remains unexercised.

**Recommendation:** Add a conditional behavioral complement to Sub-Q5. If Lit-Review #1 lands sufficient signal on agent reliability patterns for long-horizon sessions, and if the demand-mapping exercise produces a clear "hardest phase" candidate, a focused dogfood spike of one RDD phase (most likely the research phase itself, given this cycle's position) would anchor the demand map in observed behavior. This spike is naturally free-tier-compatible (the corpus's agentic-serving spike policy) and scoped to one phase, not the full benchmark.

**3. Orthogonal-axis scoping (Sub-Q3 and Sub-Q6)**

Sub-Q3 (value-location distinction) and Sub-Q6 (harness enforcement of ensemble selection) operate on orthogonal design axes. Sub-Q3 asks: where does the pattern's value come from? Sub-Q6 asks: how do we make one specific mechanism (ensemble dispatch) reliable? These could be primary and secondary within the same cycle, or they could be parallel questions that produce partially incompatible answers at synthesis. The question set does not name which axis is primary.

**Flag — axis priority not named:** If Sub-Q3 finds that the pattern's value is primarily in option (c) — the script-models layer's deterministic guarantees — then Sub-Q6's harness work for ensemble routing may be optimizing a secondary mechanism. If Sub-Q3 finds option (b) — orchestrator routing-and-summarization discipline — then Sub-Q6's harness work is optimizing the primary mechanism. These are different research programs. At synthesis, both sub-questions will have produced findings, and without a declared priority axis, the researcher will need to adjudicate which findings are primary — producing a methods-attribution problem at exactly the synthesis moment when framing adoption pressure is highest (as documented in the Cycle 3 susceptibility snapshot).

**Recommendation:** The question set should name which axis is primary for this cycle — tentatively Sub-Q3 (the value-location question) — and declare Sub-Q6 as contingent: Sub-Q6 is in scope if and only if Sub-Q3 finds that ensemble routing is a meaningful value locus. If Sub-Q3 finds the value is primarily in the script layer, Sub-Q6's scope narrows to a methodological note rather than a primary research question.

**4. Local-first commitment hardening**

The Step 1.2 synthesis correctly names the local-first commitment as load-bearing: "Any design method that recovers capability by pushing more work to the cloud orchestrator fails the cycle's value test." The question set honors this commitment consistently — Sub-Q4's composition shapes are all evaluated against the long-session benchmark, not against cloud-capability baselines, and Sub-Q2's mechanism list includes "cheap-orchestrator alone" as a genuine candidate rather than a straw man.

One potential gap: Sub-Q6's harness question does not explicitly constrain the harness to local-execution solutions. A harness that enforces ensemble selection by routing to a more capable cloud model when context grows is technically a "harness" but fails the local-first commitment. The question set should carry the local-first constraint forward into Sub-Q6's scope explicitly.

This is a soft flag rather than a structural one — the commitment is well-established in Step 1.2 and the research plan inherits it. But Sub-Q6 is the question most likely to produce implementation recommendations that push toward cloud capability under complexity pressure, and naming the constraint explicitly in Sub-Q6's scope would reduce the risk of the synthesis producing cloud-recovery solutions labeled as harnesses.

**5. Composition-shape openness**

Sub-Q4's six candidate shapes are substantively varied: they span from no library at all (generative construction) to no ensembles at all (orchestrator + tools baseline). The most important shape for this criterion is the no-ensembles baseline, which is included — this means the question set preserves the possibility that ensembles are not load-bearing for the benchmark.

The constraint-removal response's rejection of "stock + accumulation as pre-research anchor" is substantive: the user's response does not name stock + accumulation as the expected answer, and the Step 1.2 synthesis records it as a research question rather than an entry commitment. This satisfies the criterion.

One soft concern: the six shapes are listed without a theory of what conditions would favor each. The research as designed will compare them against the benchmark, but if the comparison is qualitative (demand-mapping analysis against each shape's properties), the comparison may not produce clean discriminating evidence between, e.g., "decision-tree dispatch over a fixed palette" and "parameterized ensemble templates" — these shapes are similar enough that the long-session benchmark may not cleanly differentiate them. A sharper version of Sub-Q4 would include: "What observable differences between these shapes would constitute evidence that one is superior, and does the long-session benchmark expose those differences?"

This is a methodological refinement note, not a flag at the same severity as the items above.

---

### Coverage Gaps

**1. Context-management mechanism is underspecified.** The long-session benchmark requires the system to maintain coherent state across many turns. The question set names context handoff, working memory, and attention re-orientation (Sub-Q1) as demands, but no sub-question asks how context management should be implemented. This is a well-known bottleneck for long-horizon agents (established in Essay 003's literature synthesis: AMA-Bench finds the best memory system achieves 57% accuracy; LongCLI-Bench finds most stalls occur before 30% completion). The question set decomposes demands (Sub-Q1) and attributes mechanisms (Sub-Q2), but the mechanism list in Sub-Q2 does not include context-management primitives (compaction, summarization of prior turns, selective retrieval from Plexus) as a candidate mechanism class. Plexus appears in the architecture but is not named in Sub-Q2's mechanism list.

**2. Failure mode characterization is absent.** The long-session benchmark will surface failure modes that the demand-mapping exercise cannot anticipate analytically. The question set has no sub-question asking: "What failure modes does the cheap-orchestrator + ensemble pattern exhibit under long-session conditions, and do those failure modes reveal mechanism gaps or design-method gaps?" This is adjacent to Sub-Q1's demand decomposition but not the same — failure modes under load reveal what the system actually does when it fails, which is different from what it is supposed to do. The Cycle 2 and Cycle 3 research logs both found failure modes (fast-confabulation, easy-regime confirmation, opencode CLI stall) that were not anticipated at research entry.

---

### Recommendations

**Priority 1 (structural, should be addressed before research begins):**

- Add a gate between the grounding-action questions (Sub-Q2) and the ensemble-design-space questions (Sub-Q3, Sub-Q4). The research plan's current parallel structure (lit-review #1 covers Sub-Q1, Sub-Q6; lit-review #2 covers Sub-Q2, Sub-Q3, Sub-Q4) risks Sub-Q3 and Sub-Q4 proceeding before mechanism isolation (Sub-Q2 / grounding action 1) has informed their scope. The gate: if mechanism isolation finds the script layer is load-bearing and ensemble routing is secondary, Sub-Q3's option (c) is tentatively confirmed and Sub-Q4's scope narrows to script composition shapes, not ensemble composition shapes. This gate should be named in the research plan, even if it is implicit in the analytical-decomposition method.

- Name which axis is primary: Sub-Q3 (value location) or Sub-Q6 (harness design). Declare Sub-Q6 contingent on Sub-Q3's result.

**Priority 2 (method gap, should be addressed in research plan):**

- Add a behavioral complement to Sub-Q5. The conditional spike decision should include Sub-Q5 as a candidate: "if the demand-mapping exercise lands a clear 'hardest phase' candidate, spike one RDD phase using the agentic-serving flow to anchor the demand map in observed behavior." This is the cycle's only behavioral purchase on the North-Star benchmark, and the plan should not leave it entirely to the conditional spike decision about Sub-Q6.

- Reformulate Sub-Q6's opening to test the scope condition before the harness-design question: "Does context growth degrade the cheap orchestrator's ensemble-routing judgment in this architecture, and if so, what intervention class is most appropriate?" Place the harness-design framing as the second part, not the first.

**Priority 3 (refinement, low urgency):**

- Surface the Spike A3 harness adjacency in Sub-Q6's scope. The question should examine whether the deterministic-wrapper harness pattern from Spike A3 (which enforced aggregation discipline) generalizes to routing enforcement.

- Carry the local-first constraint explicitly into Sub-Q6's scope, noting that harness solutions that recover capability by routing to cloud models on complexity pressure fail the cycle's value test.

- Add "What failure modes does the pattern exhibit under long-session conditions?" as either a named sub-question or a structured elicitation goal within the lit-review #1 scope.

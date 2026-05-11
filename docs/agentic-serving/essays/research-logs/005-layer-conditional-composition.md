# Research Log: Cycle 4 — Supported design methods for cheap-orchestrator + ensembles (long-session agentic coding)

**Cycle:** 4
**Started:** 2026-05-04
**Artifact base:** `docs/agentic-serving/`
**Plugin version:** v0.8.5
**Close shape:** undeclared at entry — likely Mode B (Research Only); may extend if research warrants

---

## Step 1.1 — Research Questions (articulated before reading the existing artifact corpus)

### Primary research question (load-bearing for the cycle)

> What design methods does the cheap-orchestrator + ensemble pattern need in order to extend from manually-staged multi-stage workflows (Spike D) toward an agentic system capable of driving arbitrary coding sessions over long horizons — with "running a full RDD cycle using the agentic-serving flow itself" as the concrete capability benchmark — and which of those design methods are properties of ensembles specifically versus orchestration more generally?

### Sub-questions (decomposed from the primary)

1. **Long-session capability decomposition** — what are the qualitatively distinct demands a "long-session coding driver" must meet that bounded multi-stage pipelines (Spike D) do not? (e.g., context handoff across phase shifts, working memory across sessions, decision-revisitation, recoverability from partial failure, branching and backtracking, attention re-orientation after interruption.)

2. **Mechanism attribution** — for each demand surfaced in (1), which mechanism does the supplying work: cheap-orchestrator alone, orchestrator + scripts as deterministic tools, orchestrator + ensembles, ensembles composed in pre-defined chains, ensembles composed dynamically by the orchestrator, or something not yet in the four-layer architecture?

3. **Ensemble-vs-orchestration distinction** — at what design-method level is the cheap-orchestrator + ensemble pattern's value located? Is it primarily (a) ensembles as compositional units of cognitive labor, (b) the orchestrator's routing-and-summarization discipline, (c) the script-models layer's deterministic guarantees, (d) cross-layer composition, or (e) some emergent property of the four-layer architecture as a whole?

4. **Composition-shape design space** — for ensembles specifically, what composition shapes are candidate design methods? Stock library + on-the-fly composition is one shape. Others include: decision-tree dispatch over a fixed palette; parameterized ensemble templates; generative ensemble construction (orchestrator builds ensembles per problem with no library); typed ensemble interfaces with verified contracts; ensemble inheritance/extension; and no-ensembles (orchestrator + tools alone). The research should compare these against the long-session benchmark, not presuppose stock+accumulation.

5. **RDD-cycle benchmark decomposition** — what does "running a full RDD cycle using the agentic-serving flow itself" demand at the design-method level? Each phase (research, model, decide, architect, build, debug, refactor, review, play, synthesize) has a distinct cognitive surface. Which phases are most demanding for the cheap-orchestrator + ensemble pattern, and what design methods are needed at the hardest points?

6. **Reliability-under-context-growth and intervention-class** — *added in Step 1.2; reformulated per Step 1.4 methods-review resolution.* Does context growth degrade the cheap orchestrator's ensemble-routing judgment in the four-layer architecture at the session lengths the North-Star benchmark requires (i.e., does the failure mode that RDD's structural hooks address transfer to ensemble routing in this architecture, and at which decision moments specifically)? *If* degradation occurs, what intervention class is most appropriate — harnesses (and does Spike A3's deterministic-wrapper pattern from Cycle 2 generalize from MARG-aggregation enforcement to routing enforcement?), context-management primitives, architecture changes, or something else? All proposed interventions must be evaluated under the local-first constraint — interventions that recover capability by pushing more work to the cloud orchestrator are disqualified on value grounds (Step 1.2 commitment 1). Sub-Q6 is **contingent on Sub-Q3's primary finding**: if Sub-Q3 locates the cheap-orchestrator + ensemble pattern's value primarily outside ensemble routing (e.g., in the script layer's deterministic guarantees, per Cycle 3 grounding action 1), Sub-Q6's intervention-class question recalibrates to the actually-load-bearing decision surface.

### Question-set scoping declaration (post-Step-1.4)

- **Sub-Q3 is primary** for the cycle. The ensemble-vs-orchestration value-location distinction must land first; subsequent design-method work attributes effort to the load-bearing layer.
- **Sub-Q2 and Sub-Q5 are decision-gating.** Mechanism isolation (Sub-Q2, informed by Cycle 3 grounding action 1) and the analytical RDD-cycle demand decomposition (Sub-Q5) inform Sub-Q3's findings and gate the scope of Sub-Q4 and Sub-Q6.
- **Sub-Q4 is contingent.** Composition-shape design space narrows to whatever layer Sub-Q3 finds load-bearing.
- **Sub-Q6 is contingent.** Intervention-class question only resolves once Sub-Q3 declares the load-bearing decision surface.
- **Sub-Q1 is foundational.** Long-session capability decomposition runs in parallel with the gating work; its findings frame the demand structure all other questions answer to.

### What is already known or assumed (held with scope conditions)

- **From Cycle 3 Spike D (held with scope condition):** the cheap-orchestrator + ensemble pattern works for *manually-staged* multi-stage workflows on bounded code-review tasks. Whether it works under autonomous routing, and whether it scales to long-horizon coding, are evidence gaps.
- **From Cycle 3 Spike C (held with scope condition):** the architecture's advantage on cross-file verification rests on the script-agent's deterministic file access, not on ensemble routing. The mechanism on cross-file verification is "deterministic tool output," not "ensemble orchestration" (Cycle 3 grounding action 1).
- **From Cycle 1 CAP-9:** the hybrid deployment — frontier-model orchestrator with local model agents — is a viable shape for the cheap-orchestrator + ensemble pattern when "cheap" is interpreted as "cheap orchestrator decisions over expensive context," not strictly "cheap dollar cost."
- **The benchmark is demanding by design.** Driving a full RDD cycle is multi-modal, long-horizon, and self-eating; whatever design methods land must scale to that shape, not only to the bounded surface where the pattern has been demonstrated.

### What would change the approach if the answer were different

- If research found that orchestration alone (without ensembles) suffices for the benchmark, the design effort shifts entirely — ensembles become a contextual accelerator at narrow points, and the cycle's design methods are about orchestrator architecture (memory, recovery, decision-revisitation, branching) rather than ensemble composition.
- If research found that ensembles are load-bearing only at specific narrow points, the design surface for ensembles narrows and the question becomes "which points, and what kinds of ensembles?" rather than "what does an ensemble palette look like?"
- If research found that the four-layer architecture is structurally insufficient for long-horizon coding (something beyond orchestrator + ensembles + scripts + plexus is needed), the cycle's premise needs revision, and the cycle may need to surface what the missing layer is before proceeding.
- If research found that "stock ensembles composed on the fly" is the wrong composition shape (e.g., generative construction outperforms it on the benchmark), the design effort for ensembles shifts toward the better shape.

### Inheritance from Cycle 3 (research-entry framing addresses these before any new architectural claim)

1. **Mechanism isolation** (Cycle 3 grounding action 1) — distinguish "deterministic tool output" from "ensemble orchestration" on cross-file verification.
2. **Autonomous routing distinction** (Cycle 3 grounding action 2) — Spike D was manually staged; multi-stage autonomous coordination via `llm-orc serve` is the evidence gap.
3. **Frontier comparison baseline scope condition** (Cycle 3 grounding action 3) — test frontier with matched information access; if architectural advantage disappears, the mechanism is information access, not architectural composition.

---

## Step 1.2 — Constraint-Removal Response

### Step 1.2 prompt (composed by agent)

The agent names **`docs/agentic-serving/system-design.md`'s four-layer architecture commitment** (orchestrator + script-models + ensembles + plexus) as the most consequential existing artifact for this research entry. The four-layer commitment shapes the default solution space — it makes "design methods for ensembles" a natural question category before the question is articulated. The tighter alternative bracket is `invoke_ensemble` alone; the broader bracket is the four-layer architecture as a whole.

**Constraint-removal prompt (load-bearing per ADR-082):**

> What would a fully realized agentic system for long-session coding look like if the four-layer architecture (orchestrator + script-models + ensembles + plexus) were not available — only a cheap orchestrator with arbitrary tool access (no ensemble layer, no script-models layer commitment, no plexus layer commitment)? How would the problem be solved then?

### User response (verbatim, 2026-05-04)

> At that point I think we'd likely be just using a cheap orchestrator only which defeats the purpose of offloading to our local machine's smaller models as much as we can. We know that cloud models are more powerful, and we cede that orchestration requires more capability, but where the orchestrator can delegate out to declarative tasks that are achievable through good architecture, that helps us move to "local-first" which is not "local-only". By contrast, if we are only using cloud model then there's not much difference from using something like Sonnet, though it's cheaper. "invoke_ensemble" only, on the other hand gets us part of the way there, but it means we have to have shipped a variety of ensembles that compose everything we need. Again, we're relying on intelligence of the orchestrator to use the right ensemble. In the RDD repo, our research indicated that structural hooks were ultimately necessary to reliably do something at a phase transition. As context grows an agent is less likely to use its own judgement to follow a directive. So I wonder what harnesses can enforce ensemble selection in our case.

### Three commitments and one new sub-question surfaced from the response

**Commitment 1 — "Local-first, not local-only" is load-bearing for the cycle.** Any design method that recovers capability by pushing more work to the cloud orchestrator — even if functionally adequate — fails the cycle's value test. The design methods must amplify what local models can do under cheap-orchestrator routing, not substitute cloud-orchestrator capability for local-model capability. The bare-orchestrator collapse case (cheap-orchestrator-only) is rejected on value grounds, not on technical grounds — it would work but defeats the purpose. By the user's framing, that case is "not much different from using something like Sonnet, though it's cheaper" — failing the local-leverage value the architecture exists to deliver.

**Commitment 2 — `invoke_ensemble` alone is partial.** The tighter bracket gets the architecture part of the way, but its viability depends on (a) shipping ensembles that compose what is needed, and (b) the orchestrator reliably selecting the right ensemble. The first is a coverage problem (composition-shape design space, Sub-Q4); the second is a reliability problem.

**Commitment 3 — Reliability-under-context-growth is now part of the cycle's territory.** Cross-corpus transfer from RDD's own research: structural hooks were ultimately necessary to reliably perform phase-transition actions because as context grows, agent judgment to follow directives degrades. The user transfers this finding to ensemble selection: if the orchestrator's ensemble-choice judgment degrades the same way, "stock ensembles + orchestrator picks" is unreliable in long sessions without structural enforcement. The transfer is plausible but needs scope-condition discipline — RDD's finding was about phase-transition events (one-shot procedural decisions); ensemble selection is continuous routing judgment. The decision class differs. Whether the failure mode transfers is itself a research question.

**New Sub-question 6 (added to the question set):** What harnesses can enforce ensemble selection — or ensemble dispatch more generally — when context growth degrades orchestrator judgment? At which decision moments does the failure mode that RDD's structural hooks address transfer to ensemble routing, and at which does it not?

### Note on the tighter-bracket alternative

The user's response engages both brackets. The broader bracket (architecture-removed) is rejected on value grounds (local-first commitment); the tighter bracket (`invoke_ensemble`-removed) is treated as partial. The cycle proceeds with the four-layer architecture as accepted prior art with documented load-bearing rationale, and with the orchestrator's ensemble-selection reliability as a newly-surfaced research surface.

---

## Step 1.3 — Research Plan and Reviewer Dispatch

### Plan (approved 2026-05-04, restructured post-Step-1.4)

The plan is sequenced in three waves, with explicit gates between waves so that load-bearing findings inform downstream scope.

**Wave 1 — Foundational and gating work (parallel):**

| # | Method | Scope | Sub-questions covered | Output |
|---|--------|-------|-----------------------|--------|
| 1.A | Lit-review #1 (`rdd:lit-reviewer`) | Long-horizon agent reliability, judgment decay under context growth, structural backstops, intervention-class taxonomy | Sub-Q1, Sub-Q6 (transfer-test framing), partial Sub-Q5 | Research log entry |
| 1.B | Analytical decomposition (in-skill) | RDD-cycle phase-by-phase demand mapping against architecture layers; mechanism attribution against Cycle 3 Spike findings | Sub-Q2 (mechanism isolation), Sub-Q5 (RDD-cycle benchmark) | Research log entry |

**Gate — Sub-Q3 primary scoping decision:**

After Wave 1 completes, evaluate whether the cheap-orchestrator + ensemble pattern's value is located primarily in (a) ensembles as compositional units, (b) orchestrator routing-and-summarization, (c) script-models deterministic guarantees, (d) cross-layer composition, or (e) emergent four-layer property. The finding scopes Wave 2.

**Wave 2 — Scope-gated lit-review (sequence after Wave 1 + Gate):**

| # | Method | Scope (gate-determined) | Sub-questions covered | Output |
|---|--------|------------------------|-----------------------|--------|
| 2.A | Lit-review #2 (`rdd:lit-reviewer`) | Composition shapes and orchestration architectures *for the load-bearing layer Sub-Q3 identifies*. If ensembles are primary, ensemble composition shapes (six candidates from Sub-Q4); if orchestrator is primary, orchestration architecture patterns; if scripts are primary, deterministic-tool composition patterns. | Sub-Q3 confirmation, Sub-Q4 (in load-bearing-layer scope) | Research log entry |

**Wave 3 — Conditional spikes (decided after Wave 2):**

| # | Method | Trigger | Sub-questions covered |
|---|--------|---------|-----------------------|
| 3.A | Conditional behavioral spike (`rdd:spike-runner`) | Wave 1.B's analytical decomposition surfaces a phase whose hardest demand is testable in a tight time-box (likely candidates: research-phase question-isolation discipline, build-phase debug/refactor/review composability, synthesize-phase artifact-trail mining). Free-tier local models per corpus spike policy. | Sub-Q5 behavioral grounding (North-Star benchmark direct-test, Flag 1 resolution) |
| 3.B | Conditional intervention spike (`rdd:spike-runner`) | Sub-Q6's transfer-test (Wave 1.A) finds context-growth degradation occurs in this architecture AND lit-review intervention-class taxonomy is thin on direct evidence at this surface. Free-tier local models. | Sub-Q6 deepening |

Both spikes use free-tier local models per the corpus spike policy (memory: free-options preference for agentic-serving spike work).

### Methods-reviewer dispatch (2026-05-04)

Dispatched `rdd:research-methods-reviewer` against the question set (Step 1.1) and constraint-removal response (Step 1.2). Output: `docs/agentic-serving/housekeeping/audits/research-design-review-cycle-4.md`. Resolution recorded in Step 1.4 below.

---

## Step 1.4 — Methods Review Resolution

The research-methods-reviewer raised seven flags; all seven were accepted.

| Flag | Concern | Resolution |
|------|---------|-----------|
| 1 | North-Star benchmark loading: Sub-Q5's analytical method does not behaviorally test the benchmark | Added Wave 3.A conditional behavioral spike (free-tier local models) to test one RDD phase via the agentic-serving flow if Wave 1.B surfaces a phase whose hardest demand is tight-time-box testable |
| 2 | Sub-Q3 and Sub-Q6 are orthogonal axes without declared priority | Sub-Q3 declared **primary**; Sub-Q6 declared **contingent** on Sub-Q3's load-bearing-layer finding (see Question-set scoping declaration in Step 1.1) |
| 3 | Premature narrowing: Sub-Q3/Sub-Q4 lit-review running before mechanism isolation can inform scope | Plan restructured into three waves with explicit Sub-Q3 primary scoping gate between Wave 1 and Wave 2; lit-review #2 scope is gate-determined |
| 4 | Missing incongruity surfacing: Spike A3's deterministic-wrapper harness pattern (Cycle 2) is not examined for generalization to Sub-Q6's territory | Sub-Q6 reformulation incorporates explicit Spike A3 generalization check; analytical decomposition (Wave 1.B) reads Spike A3 entry |
| 5 | Sub-Q6 places harness-design first, transfer-test second — implicit weight on harness-design framing | Sub-Q6 reformulated to place transfer-test ("does context growth degrade ensemble-routing judgment in this architecture at these session lengths") explicitly first; intervention-class question only fires conditional on a positive transfer-test |
| 6 | Sub-Q6 missing explicit local-first constraint | Sub-Q6 reformulation explicitly disqualifies cloud-recovery interventions on value grounds (the bare-orchestrator collapse case is rejected per Step 1.2 commitment 1) |
| 7 | Sub-Q6 embeds the conclusion that harnesses are the right intervention class | Sub-Q6 reformulation widens to "intervention class generally — harnesses, context-management primitives, architecture changes, or something else" |

Reviewer items that passed cleanly (recorded for completeness): primary question, Sub-Q1, Sub-Q2, Sub-Q4, prior-art treatment, composition-shape openness, local-first commitment in question set generally.

## Step 1.5 — Research Loop Begins

Wave 1 dispatches now. Lit-review #1 (`rdd:lit-reviewer`) runs in background; analytical decomposition (Wave 1.B) proceeds in this thread reading the corpus and Cycle 3 spike findings.

---

## Research Iterations

### Iteration 1.A — Lit-review on long-horizon agent reliability, judgment decay, and intervention-class taxonomy

**Output:** `docs/agentic-serving/essays/research-logs/005a-lit-review-long-horizon-reliability.md` (full review with citations)

**Findings summary (the four focus questions):**

**FQ1 — Long-horizon reliability and judgment decay.** Settled at the macro level. Khanal, Tao, and Zhou (arXiv:2603.29231, 2026): pass@1 drops 76.3% (short) → 52.1% (very-long) across 10 open-source models in 23,392 episodes; **errors are positively correlated across steps, not independent** (decay exceeds geometric prediction). Software engineering tasks have the steepest curve (GDS 0.90 → 0.44). **MOP paradox** (counterintuitive): frontier models exhibit *highest* meltdown rates (DeepSeek V3 19% at very-long horizons) because they pursue ambitious strategies that spiral when exploratory paths fail. Weaker models fail earlier and less catastrophically — **complicating simple capability-tier reasoning for local-first deployments**. Context rot at 50K tokens within 200K window (Chroma 2025) — well before overflow; mechanism is U-shaped attention bias compounded by accumulated tool outputs.

**The Sub-Q6 scope-condition question has a partial answer.** Phase-transition decisions and continuous-routing decisions share a mechanism (context growth degrades salience of prior directive) but differ in decision frequency. CAAF (Zhang arXiv:2604.17025) proposes different mechanisms: State Locking for settled iterative decisions, context firewalls for continuous routing. **No paper directly studies ensemble routing judgment degradation at high decision frequency as a distinct experimental phenomenon.** This is the cycle's novel empirical territory — the user's transfer is plausible (shared mechanism) but the decision-class differs and the literature does not settle whether continuous routing degrades the same way.

**FQ2 — Intervention-class taxonomy (four classes with uneven evidence):**

- **Class (a) harness-level deterministic override** — best-evidenced. CAAF: 100% reliability at $0.0027/artifact vs 0% for monolithic GPT-4o. **"Apparent LLM reliability in safety-critical domains is often a prompt engineering artifact"** — structural enforcement is qualitatively different from prompt guidance. **Works best at *finitely occurring* decision points where topology is pre-specifiable.** Tool schema gating (OpenDev plan-mode: write tools simply absent from planner interface) is class (a)/(c) hybrid avoiding the alignment tax. **Constrained decoding penalty at small models:** Qwen3-8B drops 50% → 38% under grammar constraints (Zhou arXiv:2604.06066) — structure consumes cognitive budget needed for semantic quality.
- **Class (b) context-management primitives** — require governance, not passive storage. **Naive memory scaffold contraindicated** — Khanal et al. report *universal negative effects* across all 10 tested models from episodic memory augmentation. HELM (epistemic governance with three-tier SHNM memory + provenance-tracked retrieval) is the most advanced approach. **Context-layer externalization "fails across sessions and with lost-in-the-middle degradation"** — harness-layer externalization is necessary for long-horizon tasks.
- **Class (c) architecture decomposition** — most reliable but requires pre-specifiable topology. **Best-practice converged across academic (OpenDev, Bui 2026) and practitioner (Anthropic engineering): convert long-context problems into "retrieve-on-resume" problems via external structured state (progress files, feature JSON, git history).** Each session bounded with explicit structured handoff rather than accumulated context. Most consistently supported mechanism in the literature for long-session coding agent continuity.
- **Class (d) implicit** — not separately listed in the review.

**Spike A3 generalization finding (Flag 4 from methods review answered):** Spike A3's MARG-aggregation harness is class (a) at a *finitely occurring* aggregation decision. Continuous ensemble routing is a high-frequency routing decision. **The intervention-class analysis says these need different approaches — class (c) decomposition (pre-compiled routing policy, schema-level enforcement) or class (b) governed retrieval, NOT a repeated class (a) wrapper at every step.** The user's Sub-Q6 transfer of "harnesses enforce ensemble selection" needs to refit toward class (c)/(b) intervention shapes.

**FQ3 — Long-horizon coding-agent design.** SWE-EVO (Thai et al. arXiv:2512.18470) shows **47.8-point drop** from short-horizon (72.8% SWE-Bench Verified) to long-horizon multi-file evolution (25% on SWE-EVO). Top proprietary models score **23% on SWE-Bench Pro** (enterprise-difficulty). Failure modes qualitatively distinct: shallow exploration, no backtracking (absent from OpenHands baseline), loop entrapment.

**Best-practice session architecture (converged across academic and practitioner sources):** initializer agent establishes structured scaffolding; subsequent agents read scaffolding at fresh context window start. **External structured state is the binding mechanism.** Claude Code's five-layer compaction pipeline treats semantic summarization as last resort. **Per-role model configurability distributes cognitive load.** Reflexion's verbal post-mortems work for single-problem coding (91% pass@1 on HumanEval) but Khanal et al.'s "memory scaffolds universally hurt at longer horizons" is a scope condition.

**FQ4 — Local-model capability ceiling.** Open-weight frontier converged with frontier proprietary on *short-horizon isolated coding* (Qwen3-Coder-Next 70.6-71.3% on SWE-Bench Verified vs frontier proprietary ~74-77%). Gap persists on long-horizon: 56.2% vs 62.5% on SWE-Bench Pro. **Terminal-Bench 2.0 shows 24-point gap on agentic terminal tasks** (Qwen3-Coder-Next 34.2% vs Claude Opus 4.5 58.4%). **The 7B-14B reliability boundary is the key finding for the local-first commitment:** these models reliably handle single-file, bounded, stateless tasks; unreliably handle multi-step tool calling after context accumulation. Failure mechanism: structured-output format maintenance competing with semantic reasoning capacity under context pressure. 32B boundary is where local models become competitive with frontier on many coding tasks.

**Local-first operationalized (lit-review finding):** local models amplify deterministic and bounded-scope ensemble tasks (the territory where Cycle 3 grounding action 1 already located the mechanism); the cheap cloud orchestrator handles routing, summarization, and decisions requiring sustained multi-step reasoning. **Ensemble designs that decompose complex decisions into many bounded stateless per-member tasks put local-first on the firmest available empirical ground.**

**Three tensions to track (lit-review's flag):**

1. Memory scaffolds: improve short-horizon (Reflexion) but hurt long-horizon (Khanal et al.) — scope-condition dependent. **Do not assume scaffold patterns from HumanEval-scale work transfer to RDD-cycle-scale sessions.**
2. Constrained decoding aids structural reliability but imposes alignment tax at small models. Distinguish constraint surface: schema-level binary decisions (viable) vs. constraining open reasoning chains (costly at 7-14B).
3. Frontier model gap real for agentic terminal tasks; nearly closed for short-horizon coding. **The North-Star benchmark is in the agentic terminal-task category, not the short-horizon coding category.**

---

## Sub-Q3 Primary Scoping Gate Decision (Wave 1 → Wave 2 transition)

Given the combined Wave 1.A and Wave 1.B findings, the cheap-orchestrator + ensemble pattern's value is located in **cross-layer composition where components have measurably-different error distributions, *and* where session continuity is bound by externalized structured state rather than accumulated context.** Two refinements from Wave 1.A sharpen the Wave 1.B tentative Hypothesis-D-refined finding:

### Refinement 1 — Long-horizon reliability is cross-cutting infrastructure, not a layer property

Wave 1.A surfaced that external structured state (initializer-then-resume), context governance (HELM-style epistemic memory), and per-role model configurability are converged best-practice for long-session coding agents. **The four-layer architecture as committed has gaps here:**

- **Plexus** (the closest primitive to externalized structured state) is deferred (WP-K, WP-J).
- **The Session Registry** tracks state across requests but does not implement context-window-reset-and-resume.
- **Per-role model configurability** is supported by ADR-011's "default-not-ceiling" reading (essay 003) but not yet operationalized.
- **Initializer-then-resume pattern** has no architectural primitive.

The architecture's gaps are not refutational — Cycle 1's CAP-9 hybrid deployment closes the per-turn-throughput bottleneck, and the existing structural mechanisms (Result Summarizer Harness, Composition Validator, Calibration Gate) are well-aligned with class (c) intervention principles. But the long-horizon reliability infrastructure is not yet *resourced* in the architecture.

### Refinement 2 — Continuous routing reliability requires different intervention class than finitely-occurring procedural decision reliability

Wave 1.A's intervention-class taxonomy answers Flag 4 (Spike A3 generalization): **MARG-aggregation harness is class (a) at a finitely-occurring decision; continuous ensemble routing is high-frequency and needs class (c) decomposition or class (b) governed retrieval.** The user's Sub-Q6 transfer is partially supported (shared mechanism with phase-transition reliability per Wave 1.A's CAAF reference) but the intervention class differs — **Spike A3's pattern does not directly generalize to ensemble routing.**

This is consequential for Sub-Q6's reformulation. The right intervention class for ensemble routing under context growth is *not* "harness wraps each routing decision" but:

- **Pre-compiled routing policy** (class c — bake the dispatch decisions into the orchestrator's tool-surface structure)
- **Schema-level enforcement** (class a/c hybrid — gate the routing surface itself; OpenDev plan-mode pattern of "tools simply absent")
- **Governed retrieval** (class b — provide retrieval that grounds the routing decision in externalized state)

The architecture's existing primitives align with these: ADR-003's closed five-tool surface is structurally a class (c) decomposition (the orchestrator can only call these five things, period); the Composition Validator is a class (a)/(c) hybrid (it gates `compose_ensemble`'s output schema); Plexus (deferred) is the class (b) candidate. **The architecture is well-positioned for class (c)/(b) interventions on continuous routing — the gaps are in deployment (Plexus deferred) and in operational policy (which routing decisions are pre-compiled vs. left to the orchestrator).**

### Sub-Q3 finding (load-bearing layer, post-Wave-1)

**The architecture's value is located in structured composition where components have measurably-different error distributions AND where session boundaries are bound by externalized state.** The load-bearing layer resolves layer-conditionally per task class:

| Task class | Mechanism | Layer | Evidence |
|------------|-----------|-------|----------|
| Cross-file verification | Deterministic file access | Script-models | Spike C 3/3 vs 0/2 |
| Documentation review with specificity | MARG + heterogeneity, no-collapse | Ensemble composition pattern | Spike A3 |
| Hybrid deployment economics | Cheap orchestrator routes; local ensembles execute | Orchestrator routing + tool dispatch | Cycle 1 CAP-9 |
| Long-horizon session continuity | Externalized structured state; initializer-then-resume; per-role configurability | **Cross-cutting infrastructure** (gap in committed architecture) | Wave 1.A converged best-practice |
| Continuous routing reliability | Pre-compiled routing policy / schema-level gating / governed retrieval | Class (c)/(b) — leverages ADR-003 closed-tool surface, deferred Plexus | Wave 1.A intervention-class taxonomy |

The unifying frame from essay 003 (composition of components with different error distributions) is preserved. Wave 1.A adds: **session continuity is the second binding mechanism**, and it operates as cross-cutting infrastructure rather than as a layer property.

### Implications for Wave 2 lit-review scope

The original Wave 2 was scoped to "composition shapes for the layer Sub-Q3 identifies." Given the gate finding, Wave 2 should split:

- **Wave 2.A — Composition shapes per layer.** Ensemble composition shapes (the original Sub-Q4 scope, now treating recursive ensembles-of-ensembles as a seventh candidate per essay 003); script-models composition shapes (script-as-orchestrator literature: Routine, Compiled AI, structured workflow libraries — partial coverage in Cycle 3's 004a §3.1.1 already, may need targeted depth); orchestrator routing composition shapes (supervisor patterns, hybrid-deployment routing, autonomy-gated dispatch).
- **Wave 2.B — Long-horizon reliability infrastructure.** Externalized state primitives for agent sessions (RAG patterns for agent state, structured handoff between sessions, retrieval-grounded routing); initializer-then-resume patterns; calibration-gated cross-layer composition; per-role model configurability patterns. **This is the framing-shift candidate from Wave 1.A.**

The total Wave 2 scope is larger than originally planned. Recommendation: dispatch Wave 2.A and Wave 2.B as two parallel lit-reviewer subagents to keep each focused and the literature coverage tractable.

### Implications for Wave 3 conditional spikes

- **Wave 3.A behavioral spike (RESEARCH-phase research-loop):** stronger candidate after Wave 1.A. The North-Star benchmark is in the *agentic terminal-task category* per Wave 1.A — the hardest category where local-vs-frontier gap is widest (24 points on Terminal-Bench 2.0). A behavioral spike on a single research-loop iteration via `llm-orc serve` direct API (avoiding Spike D's opencode CLI stall) tests whether the architecture's existing class (c) primitives (closed five-tool surface; Result Summarizer Harness; Composition Validator) can drive bounded research-cycle work even under the harder benchmark category. Free-tier local + cheap-cloud-orchestrator per corpus spike policy.
- **Wave 3.B intervention spike (Sub-Q6 follow-up):** *now better-defined.* If Wave 2.B's lit-review on long-horizon reliability infrastructure shows class (c) routing-surface gating is the converged intervention class, the spike should test pre-compiled-vs-orchestrator-judgment dispatch on a high-frequency-routing fixture — comparing free orchestrator dispatch (Spike A's MiniMax-uses-Read-1/3) against schema-pre-compiled dispatch where the routing decision is structurally constrained at the tool surface. Conditional on Wave 2.B finding gaps in published evidence at the cycle's deployment shape.

### Implications for the cycle's framing — three substantive shifts to surface to the practitioner

The Wave 1 findings produce three framing shifts large enough that the practitioner should see them before Wave 2 dispatch:

1. **Long-horizon reliability is cross-cutting infrastructure.** External structured state + initializer-then-resume + per-role configurability are converged best-practice. The architecture has gaps here (Plexus deferred). The cycle's design methods may need to address this layer rather than treating ensemble composition as the primary surface.

2. **Sub-Q6's transfer is partially supported but the intervention class is wrong.** Spike A3's harness pattern doesn't generalize to continuous routing. The right intervention is class (c) decomposition or class (b) governed retrieval — and the architecture's ADR-003 closed-tool-surface + Plexus (deferred) align with these. Sub-Q6's framing should pivot from "what harnesses enforce ensemble selection" to "what decomposition/grounding primitives bound continuous routing under context growth."

3. **The North-Star benchmark sits in the hardest task category** (agentic terminal-tasks per Terminal-Bench 2.0). Local-vs-frontier gap is 24 points; SWE-EVO drop is 47.8 points. The local-first commitment must be operationalized as **decompose-stateless-per-member, with cheap-cloud-orchestrator for sustained reasoning, bound by externalized state**. The MOP paradox (frontier melts down more on long-horizon) means "throw frontier at the hard cases" is not the reliable backup — cheap-orchestrator + class (c) architecture decomposition may be more reliable on long-horizon work.

**Wave 1 status: complete.** Wave 2 dispatch deferred pending practitioner review of the framing shifts.

**Practitioner accepted gate findings 2026-05-04 ("Yep, acceptable") — Wave 2 dispatched as split.**

### Iteration 2.A — Lit-review on composition shapes per layer

**Output:** `docs/agentic-serving/essays/research-logs/005b-lit-review-composition-shapes-per-layer.md` (full review with citations)

**Findings summary (three focus areas):**

#### Ensemble composition shapes

- **Recursive ensembles-of-ensembles (essay 003 seed):** RecursiveMAS (arXiv:2604.25917, April 2026) requires gradient access — *not available in llm-orc's black-box inference setting*. **Attention-MoA (arXiv:2601.16596, January 2026)** is the deployable alternative: inter-agent critique-and-refine within each layer + residual history accumulation prevents collapse-at-depth that standard MoA exhibits after layer 3. 91.15% vs MoA 88.56% on AlpacaEval 2.0, **monotonically increasing performance to layer 5, no fine-tuning required**. **Aggregation-agent quality drives a 12.82pp gap — the cloud orchestrator quality is the bottleneck, not the member models.** Baseline cost ~119K tokens per query before optimizations. The "cheap orchestrator at the aggregation moment" question is now load-bearing: if Attention-MoA-style depth is desirable, aggregator quality matters more than member quality.

- **Self-MoA challenges the cycle's heterogeneity prior — but task-class-conditional.** Self-MoA (arXiv:2502.00674, ICLR 2025): sampling multiple times from a single best-performing model outperforms cross-family mixing by +6.6% on instruction-following benchmarks. **But: holds for tasks requiring correctness on well-defined questions; does NOT hold for tasks requiring cross-family coverage of systematically different blind spots (Cycle 2's Spike A3 finding on cross-file verification).** Resolution is task-class-dependent, not universal. The cycle's task class (agentic verification, documentation review, code review) is in the cross-family-coverage territory.

- **Deliberation is worse than selection, by 6× (DeliberationBench, arXiv:2601.08835, December 2025):** 82.5% win rate for best-single selection vs. 13.8% for best deliberation protocol, at 1.5–2.5× higher compute. **Strengthens Cycle 2 Loop 4's Yao 2025 panel-discussion finding to a documented reliability risk.** Practical implication: **parallel independent roles with concatenation are the only defensible ensemble topology — any sequential deliberation step is a documented reliability risk.** Spike A3's MARG-concatenation pattern is now strongly literature-supported.

- **Generative topology construction is potent but infrastructure-heavy.** DyTopo +6.2pp, HERA +38.69%, both outperform static baselines substantially. Both require embedding models / reward signals exceeding the closed five-tool surface without architectural extension. DyTopo's per-round descriptor emission is the lightest-weight approach.

- **Typed contracts: cycle territory is novel.** MCP (November 2025) mandated structured output schemas; PydanticAI enforced at framework level; Arazzo for workflow-level contracts. **No peer-reviewed paper measures the reliability delta from typed-contract enforcement.** llm-orc's Composition Validator + Calibration Gate combo would produce novel empirical evidence in this territory. Tension with generative composition is real but resolved by HERA's pattern (member-interface schemas stable; topology generation operates above that boundary).

- **Calibration-gated routing: literature support is direct.** ReDAct (arXiv:2604.07036, 2026): deferring only 15% of decisions to a large model matches full large-model quality. Chuang et al. (arXiv:2502.04428, 2025), across 1,500+ settings: **UQ method choice dominates threshold choice in calibration-gated routing.** Direct design recommendation for ADR-007's Calibration Gate implementation.

- **Retrieval-grounded selection beats LLM-judgment routing.** R3AG (arXiv:2604.22849, 2026) and HERA (arXiv:2604.00901, 2026): **retrieval-grounded routing outperforms both static routing and LLM-judgment routing.** HERA's reward-guided experience retrieval is the **closest published implementation of "stock library + retrieval-grounded selection"** — the user's Step 1.2 design intuition. *Performance improvement is contingent on Plexus enablement* (currently deferred WP-K). In Plexus-absent mode, this shape degrades to static selection.

#### Script-models composition shapes

- **Parallel DAG-planned execution (LLMCompiler, arXiv:2312.04511, ICML 2024):** 3.7× latency speedup, 6.7× cost reduction over ReAct via separating dependency-graph planning (LLM) from parallel execution (deterministic). Canonical pattern.

- **Compilation economics (Agentic Compilation, arXiv:2604.09718, 2026):** ~$150 → <$0.10 per execution over 500 runs of a 5-step workflow. **Amenability: stable, repetitive, structure-amenable workflows.** For Cycle 4: link checking and model-profile validation compile cleanly; semantic documentation analysis does not. **Compiled AI's hybrid mode** (deterministic compilation for structure-amenable + runtime LLM for semantic) is the pattern that fits the cycle's mixed task class.

- **Deterministic tools as consensus-resistant anchors (Wisdom and Delusion of LLM Ensembles, arXiv:2510.21513, October 2025):** CrossHair counterexample feedback embedded in code-generation ensemble *prevents LLM consensus from overriding tool evidence*. **Published support for Spike A3's script-member-alongside-LLM pattern.** Mechanism is categorical, not probabilistic — deterministic outputs cannot be argued away by LLM consensus. **Open question the literature does not address: what happens when deterministic tool output conflicts with LLM consensus?** No paper studies this failure mode — Cycle 4 territory if surfaced.

- **FSM-based routing (MetaAgent, arXiv:2507.22606, ICML 2025):** auto-designed FSM topologies with state traceback for error recovery. Class (c) decomposition pattern from Wave 1.A.

#### Orchestrator routing composition shapes

- **Pre-specifiable routing > LLM-judgment routing** is convergent across CAAF (Wave 1.A), LLMCompiler, MetaAgent. **Design cost is pre-specifiability of the routing logic — satisfied for the cycle's stable task classes.**

- **Task-aware delegation cues (Gu, arXiv:2603.11011, March 2026):** routing based on pre-computed capability profiles and coordination-risk priors is effective without real-time capability judgment. **Directly applicable to the cheap-cloud-orchestrator constraint** — the cheap orchestrator does not need real-time capability judgment if the routing decision is encoded in static profiles.

- **Bounded autonomy with schema-level enforcement** is convergent deployment practice — aligned with ADR-008's per-session Autonomy Levels architecture.

#### Principal tensions surfaced

1. **Same-family quality consistency vs. cross-family diversity** — task-class-conditional, not universal. The cycle's task class (agentic verification) is on the diversity side.
2. **Typed contracts constrain generative composition** — resolved by HERA's pattern (interface schemas stable; topology generation above that boundary).
3. **Compilation benefits require repetition** — applicable to stable verification scripts; not to one-time novel workflow configurations.
4. **Recursive depth improves single-query quality (Attention-MoA monotonically improving to layer 5); accumulated scaffolding hurts long-horizon agents (Wave 1.A)** — *different decision surfaces, not a direct contradiction*. Recursive depth is per-query; scaffolding is per-session.

#### Four candidate framings (the lit-reviewer offered)

- **Framing A — Layer-conditional composition shape selection.** Different shapes appropriate at different layers given the task class at each layer. Directly operationalizes the Sub-Q3 gate finding.
- **Framing B — Pre-specifiable vs. generative routing as primary design axis.** What can be harness-enforced vs. what requires trust-score monitoring at each layer.
- **Framing C — Typed contracts as universal interface discipline.** Composition-validator pattern applied at every layer boundary.
- **Framing D — Experience accumulation as the progression from stock-library to retrieval-grounded composition.** Plexus store as infrastructure; design question is what signals to capture and what retrieval mechanism to use.

These four framings are stackable: **A (layer-conditional shape selection) with B (pre-specifiable-vs-generative as primary axis within each layer) unified by C (typed-contract interfaces) and progressing along D (experience-accumulation arc)** is a coherent design-method posture candidate. Synthesis pass deferred until Wave 2.B returns.

### Iteration 2.B — Lit-review on long-horizon reliability infrastructure

**Output:** `docs/agentic-serving/essays/research-logs/005c-lit-review-long-horizon-reliability-infrastructure.md` (full review with citations; 23 sources, 15 new + 8 inherited)

**Findings summary (four focus areas):**

#### Externalized state primitives

- **Anthropic initializer schema is canonical:** `feature_list.json` (200+ entries with monotonic boolean `passes` field — *structural non-regression enforcement at schema level, not prompt compliance*), `claude-progress.txt` (append-only free text), `init.sh` (deterministic environment bootstrap). The monotonicity constraint is the critical design element: **enforces non-regression at schema level, not through agent compliance.** Class (a)/(c) hybrid in Wave 1.A's intervention-class taxonomy.
- **Three-tier loading** (hot memory / domain specialists / cold memory): validated across 283 sessions / 108K-line codebase / 16,522 agent turns (arXiv:2602.20478).
- **MemMachine nucleus expansion (arXiv:2604.04853):** 93.0% on LongMemEvalS, 80% fewer tokens than Mem0. **Retrieval-stage tuning contributes 4× more improvement than ingestion-stage tuning** — design implication: invest in retrieval quality, not ingestion schema.
- **Plain filesystem outperforms naive vector-store libraries at 74% on MemoryAgentBench** — *don't invest in retrieval infrastructure before simple structured-artifact approaches saturate.*
- **Mnemonic sovereignty framework (arXiv:2604.16548):** six lifecycle phases with distinct failure modes. Append-only JSONL supports auditability and session operations (resume/fork/rewind) but creates rollback limitations and memory-poisoning attack surface. **Write-gate validation recommended but not operationalized in any reviewed system** — Cycle 4 territory if pursued.
- **Open gap:** *No published paper documents a structured-state schema for multi-phase workflow phase transitions* (as distinct from single-task session handoffs). What an RDD-cycle initializer should encode for wave dispatch state, framing commitments, and gate conditions is novel territory.

#### Initializer-then-resume patterns

- **Anthropic schema fully specified:** six fixed steps — confirm working directory → read progress log → read git history → consult feature list → run init.sh → select one feature. Full-read for small structured artifacts; governed retrieval for large knowledge stores. Complementary, not competing.
- **Schema-level planning enforcement (OpenDev Planner):** write tools *absent from the schema* — not a prompt instruction. **Avoids alignment tax at small models** (12-point accuracy drop from constrained decoding at 8B per Wave 1.A's Zhou arXiv:2604.06066; schema gating imposes no decoding overhead). Class (a)/(c) hybrid.
- **Claude Code's five-layer compaction operationally specified:**
  - Layer 0: persist tool results >50K chars to disk, inject 2KB preview + path
  - Layer 1: delete old cache entries without invalidating prefix
  - Layer 2: clear idle-expired tool results after 60+ minute gap
  - Layer 3: **free summary via continuously-maintained session notes (zero LLM cost, nine-section template, 12K token cap)**
  - Layer 4: LLM-generated semantic summary as last resort, circuit-breaker after 3 failures
  - 250,000 API calls/day waste from pre-circuit-breaker Layer 4 failures validates cheapest-first ordering. **Layer 3's zero-cost notes pattern is directly portable to llm-orc's Conversation Compaction module — nine-section template (current state, tasks, files, workflow, errors, learnings, worklog) is adoptable as-is.**
- **Devin gap:** Cognition Labs explicitly states Devin does *not* maintain cross-session memory as of mid-2025. **The most prominent deployed coding agent does not implement the pattern the literature converges on as best practice.** Notable.
- **Best result with this pattern:** **Confucius Code Agent — 59% Resolve@1 on SWE-Bench-Pro**, the highest reviewed result on that benchmark, with persistent cross-session note-taking as the stated mechanism (arXiv:2512.10398).

#### Calibration-gated cross-layer composition

- **Dual-Process AUQ (arXiv:2601.15703):** training-free. System 1 propagates verbalized confidence through attention (soft constraint); System 2 applies binary gate. *Low confidence triggers reflection, not blocking.* **+10.7% ALFWorld, +13.6% WebShop.**
- **Trajectory-level calibration HTC (arXiv:2601.15778):** process-level features across entire trajectories — *more informative than output-level calibration for long-horizon decisions*. Cross-domain transfer without retraining validated. **Calibration mechanisms are architecturally portable across agent frameworks.**
- **Confidence-gated model-tier routing OI-MAS (arXiv:2601.04861):** closest published analogue to in-process Calibration Gate. Higher confidence penalizes expensive model choices; lower confidence triggers tier escalation. **17-78% cost reduction, +12.88% accuracy.**
- **What the literature does not have:** *No paper implements calibration as a cross-layer primitive* — L0 ensemble outputs gating L1 dispatch decisions. The components exist (AUQ, HTC, OI-MAS); the composition does not. **The cycle's novel territory.**
- **ADR-007 implication:** Current Calibration Gate is post-hoc output-level (last-N positive signals → promote). Literature points toward layering in-process trajectory-level calibration alongside it. **A fully realized gate would combine both: post-hoc promotion tracking (ADR-007 as implemented) + in-process confidence gating (AUQ/OI-MAS pattern) for within-session dispatch decisions.**
- **Upward signal path:** Making calibration a cross-layer primitive requires a read-only signal channel from L0 (ensemble outputs) to L1 (Calibration Gate dispatch decisions) — which the current layering rule ("edges never upward") does not permit. **This is the one identified point where the four-layer architecture may need architectural extension — not a fifth layer, but a new upward-read signal path within the existing layer structure.**

#### Per-role model configurability

- **Heterogeneous role-staffing beats homogeneous, consistently.** SC-MAS: +3.35% accuracy / −15.38% cost on MMLU. MasRouter: +1.8-8.2% on MBPP / −52% on HumanEval. OI-MAS: +12.88% accuracy / −17-78% cost. **Mechanism: capability saturation. Homogeneous systems over-allocate expensive capability to tasks that don't require it.**
- **Topaz eight-skill taxonomy** (code generation, tool use, mathematical reasoning, logical reasoning, factual knowledge, writing quality, instruction following, summarization): *directly adoptable role-profiling vocabulary.* "Efficiency gains stemmed from capability saturation rather than hidden quality loss" — routing cheaper models to tasks where expensive models add no marginal value is *not a quality compromise.*
- **External memory couples Focus Areas 1 and 4.** Small models cannot maintain coordination context independently. **External memory is what makes small-model role-staffing viable. The two design surfaces are not independent choices.**
- **ADR-011 default-not-ceiling operationalized:** OI-MAS's pattern is the right implementation — cheap model by default → confidence below threshold → escalate. **In-process instantiation of essay 003's "default-not-ceiling" reading.**
- **Open:** Local vs. cloud role assignment specifically — which task classes belong on local hardware — remains practitioner-documented but not peer-reviewed. **The cheap-orchestrator + local-ensemble pattern is empirically unstudied in the reviewed literature.**

### Architectural Verdict (Wave 2.B's load-bearing finding)

> Long-horizon reliability infrastructure is **operationalizable within the existing four-layer architecture** as elaborations of existing module responsibilities:
>
> - Structured handoff artifacts → Session Registry (L3), additive responsibility
> - Five-layer compaction pipeline → Orchestrator Runtime (L2), elaboration of existing Conversation Compaction awareness
> - Trajectory-level calibration → Calibration Gate (L1), significant elaboration of existing module
> - Per-role model router → Orchestrator Tool Dispatch (L2), elaboration of existing interposition logic
> - Write-gated persistence → Session Registry (L3), guard on existing write path
>
> **The one exception:** Cross-layer calibration (L0 ensemble signals gating L1 dispatch decisions) would require an upward read-only signal path, which the current layering rule prohibits. This is the one point where an architectural addition may be required — not a fifth layer, but an explicit signal channel upward from L0 to L1.

**This resolves the largest framing-shift candidate from Wave 1.A.** The Wave 1.A surfacing of "long-horizon reliability is cross-cutting infrastructure" raised the candidate that the four-layer architecture might need fifth-layer addition or cross-cutting infrastructure module. Wave 2.B's verdict is **less radical**: the gaps are operationalization gaps within existing modules, not architectural-shape gaps. The architecture is in better structural shape than the gate finding initially implied.

The single architectural-extension candidate (upward read-only signal channel for cross-layer calibration) is the cycle's most consequential candidate ADR if the cycle proceeds past Mode B into DECIDE.

---

## Wave 1 + Wave 2 Synthesis Pass

The lit-reviews + analytical decomposition together produce a coherent design-method posture for the cheap-orchestrator + ensemble pattern under the North-Star benchmark of "drive a full RDD cycle using the agentic-serving flow itself."

### The composite framing (Wave 2.A's four framings stack with Wave 2.B's architectural verdict)

**The cheap-orchestrator + ensemble pattern's supported design methods are:**

1. **Layer-conditional composition shape selection (Framing A)** — different shapes are appropriate at different architectural layers given the task class at each layer. The Sub-Q3 gate findings operationalize directly:
   - Cross-file verification → script-models layer (Spike C 3/3 vs 0/2)
   - Documentation review with specificity → ensemble composition pattern (MARG + heterogeneity, no-collapse — *strengthened by DeliberationBench's 6× gap to selection over deliberation*)
   - Hybrid deployment economics → orchestrator routing + tool dispatch (CAP-9)
   - Long-horizon session continuity → cross-cutting infrastructure operationalized within existing layers (Wave 2.B verdict)

2. **Pre-specifiable routing as primary design axis within each layer (Framing B)** — pre-specifiable beats LLM-judgment routing across CAAF, LLMCompiler, MetaAgent. Class (c) decomposition is the dominant intervention class for continuous routing reliability (Wave 1.A); ADR-003's closed five-tool surface is structurally a class (c) decomposition. The orchestrator's job is correct-dispatch-at-known-points, not real-time capability judgment — task-aware delegation cues (Gu 2603.11011) are effective without it.

3. **Typed contracts as universal interface discipline (Framing C)** — the Composition Validator pattern applied at every layer boundary. *No peer-reviewed paper measures the reliability delta from typed-contract enforcement.* llm-orc's Composition Validator + Calibration Gate combo is novel empirical territory. HERA's pattern resolves the typed-vs-generative tension: member-interface schemas stable; topology generation operates above that boundary.

4. **Experience accumulation as the progression arc (Framing D)** — stock-library → retrieval-grounded → Plexus-store-driven. HERA is the closest published implementation of the user's Step 1.2 reframe ("stock ensembles composed and accumulated on the fly"). **Performance gains contingent on Plexus enablement (currently deferred WP-K).** In Plexus-absent mode, the architecture degrades to static stock-library selection — which is acceptable as default but caps the capability ceiling.

### How the framings address the cycle's six sub-questions

| Sub-question | Resolution |
|--------------|------------|
| Sub-Q1 (long-session capability decomposition) | Failure modes are settled at macro level (Khanal: 76→52% pass@1 across 10 models; SWE-EVO 47.8pt drop; SWE-Bench Pro 23%). Cross-cutting infrastructure (externalized state, initializer-then-resume, per-role configurability, in-process calibration) is the design surface — operationalizable within existing layers per Wave 2.B. |
| Sub-Q2 (mechanism attribution) | Layer-conditional per Wave 1.B + Cycle 3 evidence. Script-models for cross-file verification; ensemble composition for specificity; orchestrator routing for hybrid economics; cross-cutting infrastructure for session continuity. The unifying frame (composition of components with different error distributions) holds; layer attribution is task-class-dependent. |
| Sub-Q3 (load-bearing layer) | **No single layer carries the value across all task classes.** Hypothesis-D-refined: cross-layer composition with measurably-different error distributions. *Largest framing shift from Wave 2.B*: long-horizon infrastructure operationalizes within existing layers, not as a new layer. The one architectural extension candidate is the upward read-only signal channel for cross-layer calibration. |
| Sub-Q4 (composition-shape design space) | Plural, layer-conditional. **Strongest cycle-specific recommendations from the literature:** parallel-independent-roles + concatenation as defensible ensemble topology (DeliberationBench 6× gap to selection); Attention-MoA as deployable ensembles-of-ensembles when aggregator quality is high (12.82pp dependence on aggregator); LLMCompiler-style parallel DAG-planned execution at script-models layer; pre-specifiable routing > LLM-judgment routing at orchestrator layer; HERA-style retrieval-grounded selection contingent on Plexus enablement. |
| Sub-Q5 (RDD-cycle benchmark decomposition) | Three decision-class clusters per Wave 1.B (heavy specialist-dispatch / heavy continuous-routing / conversational). Cluster 1 phases (RESEARCH, DECIDE, SYNTHESIZE) align with class (a) harness patterns at finitely-occurring decisions. Cluster 2 phases (BUILD, ARCHITECT) align with class (c) decomposition + class (b) governed retrieval (Wave 1.A) — externalized state is the binding mechanism per Wave 2.B. RESEARCH-phase research-loop is the natural Wave 3.A behavioral spike candidate. |
| Sub-Q6 (intervention class) | **Reformulated per gate findings.** Transfer of RDD's structural-hook finding is partially supported (shared mechanism with phase-transition reliability) but decision-class differs (continuous routing vs. one-shot procedural). Spike A3's class (a) wrapper does not generalize to ensemble routing. Right intervention class is class (c) decomposition (pre-specifiable routing; schema-level enforcement) and class (b) governed retrieval (HERA-style) — and the architecture's existing primitives (ADR-003 closed five-tool surface; Composition Validator; Plexus-deferred) align with these. **Calibration-gated cross-layer composition is the cycle's most novel territory** (no published paper implements it; AUQ + OI-MAS + HTC components exist but composition is untested). |

### Three findings that shift cycle posture

1. **Parallel-with-concatenation is the only defensible ensemble topology.** DeliberationBench's 82.5% (selection) vs 13.8% (deliberation) — a documented 6× reliability gap. Combined with Cycle 2's Yao 2025 panel-discussion and Spike A's two-stage-collapse-summarization findings, this is now load-bearing. **Any sequential deliberation step is a documented reliability risk.** llm-orc's Result Summarizer Harness should be recognized as a candidate sequential-deliberation point — its position post-`invoke_ensemble` and pre-orchestrator-context introduces a compression that can degrade specificity (Spike A's mechanism). The Calibration Gate could in principle gate summarizer activation — but this is novel territory.

2. **Aggregator quality dominates in ensembles-of-ensembles.** Attention-MoA: 12.82pp gap from aggregator quality alone, monotonic improvement to layer 5. **For the cheap-orchestrator + ensemble pattern, this means the orchestrator's role at the aggregation moment is load-bearing for ensembles-of-ensembles success.** "Cheap" must be calibrated against this — if the cheap orchestrator is the aggregator, its quality ceiling caps recursive ensemble depth. ADR-011's "default-not-ceiling" reading is operationally instantiated by OI-MAS's confidence-gated tier escalation: cheap by default; escalate when aggregation demands exceed cheap-tier capability.

3. **Confucius Code Agent's 59% Resolve@1 on SWE-Bench-Pro via persistent cross-session note-taking is the cycle's empirical anchor for the long-horizon coding benchmark.** Highest reviewed result on the benchmark; mechanism is cross-session note-taking. Combined with Devin's explicit *no-cross-session-memory* gap, this signals that the cycle's design methods should treat externalized-state-as-primary, not as optional optimization.

### Wave 3 conditional spike decisions

- **Wave 3.A behavioral spike (RESEARCH-phase research-loop dogfood):** **Recommended to fire.** The synthesis is now strong enough to warrant empirical grounding. The benchmark would test whether the cheap-orchestrator + invoke_ensemble + script-models pattern, dispatched via `llm-orc serve` direct API path (avoiding Spike D's opencode CLI stall), can drive bounded research-loop work — including audit-dispatch chain (citation-auditor, argument-auditor) at known transition points. Free-tier cheap-cloud-orchestrator + free-tier local ensembles per corpus spike policy. The spike directly tests Wave 1.A's "agentic terminal-task category" claim at the cycle's deployment shape, where local-vs-frontier gap is widest (24-point Terminal-Bench 2.0 gap).

- **Wave 3.B intervention spike (Sub-Q6 follow-up):** **Recommended NOT to fire.** Sub-Q6 has converged on class (c) decomposition + class (b) governed retrieval as the appropriate intervention class without needing additional empirical evidence — the literature is direct (CAAF, LLMCompiler, MetaAgent, OpenDev plan-mode). The architectural-extension candidate (upward signal channel for cross-layer calibration) is DECIDE-phase territory if the cycle continues, not a research-spike question.

### Cycle close-shape recommendation

The synthesis is rich enough for Mode B (Research Only) closure with an essay. But **the surfacing of the upward-signal-channel architectural-extension candidate** is a natural Mode B → DECIDE trigger: if the cycle's design methods include cross-layer calibration, the layering-rule amendment is a candidate ADR that warrants DECIDE-phase resolution (with argument audit, conformance check against existing ADR-002 layering, etc.).

**Three options for cycle close shape:**

1. **Mode B (Research Only) — close after essay + audits + reflection.** Pure research deliverable. The architectural-extension candidate is recorded as Cycle 5 entry priority.
2. **Mode B+ → DECIDE (extend to ADRs only).** Run essay + audits + reflection, then advance to DECIDE on the upward-signal-channel ADR (and any other ADRs that surface from the cycle). Stop at DECIDE close.
3. **Full Mode A (run the pipeline through BUILD or further).** The cycle's deliverables become operational: structured handoff artifacts in Session Registry; Layer 3 zero-cost-notes Conversation Compaction; trajectory-level calibration in ADR-007 elaboration. This is implementation territory.

Recommendation deferred until practitioner reviews the synthesis.

**Wave 2 status: complete.**

### Iteration 3.A — Behavioral spike: research-loop dogfood via `llm-orc serve` direct API path

**Output:** `docs/agentic-serving/essays/research-logs/005d-spike-research-loop-dogfood.md`

**Retained artifacts** (per corpus retention policy until corpus close): `scratch/spike-cycle4-research-loop-dogfood/` (run scripts, ensemble YAML, three trial directories with execution traces, four artifact subdirs covering two ensemble executions + two summarizer executions). One-line answer per spike: **dispatch path works at the cheap-cloud tier on 2/3 trials (21-32s wall-clock, no stall, autonomous routing fired correctly in Trial 1) — with two substantive negative findings.**

**Per-trial summary:**

| Trial | Setup | Result | Finding |
|-------|-------|--------|---------|
| 1 | Autonomous routing | Dispatched `spike-cycle4-research-loop` correctly without explicit staging | **Direct evidence on Cycle 3 grounding action 2** — autonomous routing fires at cheap-cloud tier on this fixture. N=1; not generalizable but a non-zero signal. |
| 2 | Explicit instruction | Dispatched correctly | **Surfaced specificity-loss artifact** — orchestrator claimed "verbatim from ensemble return" what was actually the summarizer's compressed blob. The Result Summarizer Harness re-instantiates Spike A's specificity-loss mechanism on the orchestrator's view of ensemble outputs. |
| 3 | Diagnostic | Zero artifacts produced | **Phantom tool-result observation** — the cheap-cloud model wrote prose claiming a tool call had occurred without emitting any tool call structure ("The tool call has been made and the result is displayed above as a `role:tool` observation"). New failure-mode hook. |

**Validation results against the cycle's synthesis hooks:**

- **Closed five-tool surface (ADR-003) sufficed.** The orchestrator did not request any tool outside it. *Confirms* class (c) decomposition adequacy at the cheap-cloud tier for this fixture.
- **Long-horizon session continuity was NOT a binding constraint within ONE iteration.** Context-rot symptoms were single-turn (harness compression), not multi-turn (accumulation). *Refines* the cycle's framing: long-horizon infrastructure matters at multi-iteration scale (Cycle 5+ territory), not single-iteration.
- **Result Summarizer Harness specificity-loss reproduced at orchestrator's view.** Cycle 2 Spike A documented the mechanism (two-stage cascade collapse destroys per-reviewer specificity). Wave 3.A documents it at the *post-`invoke_ensemble` interposition* — the harness compresses the ensemble's structured output into a summarized blob the orchestrator then quotes "verbatim." **This empirically validates ADR candidate #5 (Result Summarizer Harness reconsideration) — the issue is not theoretical; it shows up on a single-iteration trial.**
- **Autonomous routing partial validation.** N=1 explicit success in Trial 1; Cycle 3 grounding action 2's evidence gap is *partially* closed (a non-zero positive signal exists at the cheap-cloud tier on this fixture) but full validation requires N>1 trials at multi-iteration scale.

**Methodological finding (new for Cycle 4):** the chat-completions response body returns the *final* assistant message only; the tool-call trajectory has already been consumed pre-final. *Initial false-negative read showed `n_tool_calls: 0` in the response body, but artifact-directory cross-check showed actual trajectory.* The "trajectory inference via artifact-directory cross-check" approach is now a documented spike methodological pattern. Future Cycle 4+ spikes should not read response-body-only as evidence of dispatch absence.

**Spike-runner's recommendations to the cycle (recorded for synthesis integration):**

1. Treat single-iteration `invoke_ensemble` as validated at the cheap-cloud tier (yes — ADR-003 closed-tool-surface confirms; autonomous routing has positive N=1).
2. Treat the summarizer's specificity-loss as the next class-(c) target — *summary-with-pointer* (return summary + pointer to the full output, accessible on demand) or *skip-summarization-when-short* (gate summarizer activation by output length / content-density signal). Empirically motivated by Trial 2.
3. Treat phantom-tool-call as a runtime-side guard target (same shape as the mixed-batch-rejection path). Class (a) intervention: validate that emitted tool-call structures match the orchestrator's reported claims.
4. Plan the next spike at *multi-iteration* scale — that's where the North-Star benchmark lives. **Cycle 5+ territory.**
5. Defer frontier-tier comparison until the multi-iteration spike.

---

## Wave 3 → Synthesis Refinements

The spike's empirical findings refine the cycle's composite framing in three specific places:

### Refinement 1 — ADR candidate #5 (Result Summarizer Harness reconsideration) is empirically motivated

The cycle's synthesis flagged the Harness as a candidate sequential-deliberation reliability risk based on DeliberationBench's 6× selection-vs-deliberation gap. Wave 3.A *empirically reproduces* the specificity-loss mechanism at the orchestrator's view of ensemble outputs in a single-iteration trial. ADR candidate #5 is now grounded:

- **Decision direction:** The Harness should support a *summary-with-pointer* mode (return summary + persistent reference to the full ensemble output, accessible via `query_knowledge` or a new pointer-following primitive) or *skip-summarization-when-short* (gate activation by output-length / content-density threshold). Pure summarization remains the default for outputs that exceed an empirically-determined threshold; selective skip preserves the harness's value (preventing oversize outputs from dominating orchestrator context) while addressing specificity-loss for cases where summarization adds compression cost without compression benefit.
- **The Calibration Gate could in principle gate summarizer activation** — using AUQ-style confidence on whether the orchestrator-can-handle-the-full-output as the gate signal. But this is more speculative than the simple length/density gate and could be reserved as Cycle 5 territory.

### Refinement 2 — Phantom tool-call confabulation is a new failure-mode hook

Wave 3.A's Trial 3 documents a previously-unobserved failure: the cheap-cloud orchestrator confabulates a tool-call observation in prose without emitting any tool-call structure. This is qualitatively different from the failure modes the cycle's prior literature scan covered (judgment-decay, attention drift, instruction recency bias). The pattern is **structured-output failure** — the model generates *narrative about having called a tool* without generating *the call itself.*

This adds a candidate ADR — call it **ADR candidate #7: Tool-call structural validation.** The Tool Dispatch module would interpose a class (a) guard: validate that orchestrator claims of tool calls correspond to actual tool-call structures emitted; reject mismatches as malformed-batch (consistent with the existing typed-error path for per-model tool-calling rejection). The recent commit `9f86d0b feat: raise typed error when provider rejects tool calling per-model` is a structurally-similar precedent in the codebase.

The cycle's six ADR candidates now expand to seven:

| # | ADR candidate | Module | Source | Type |
|---|--------------|--------|--------|------|
| 1 | Conversation Compaction adopts Claude Code's five-layer pattern | Orchestrator Runtime (L2) | Wave 2.B | Elaboration |
| 2 | Session Registry adopts initializer-then-resume schema | Session Registry (L3) | Wave 2.B | Elaboration |
| 3 | Calibration Gate extension (HTC + AUQ in-process calibration) | Calibration Gate (L1) | Wave 2.B | Elaboration |
| 4 | Per-role tier-escalation router (OI-MAS pattern) | Tool Dispatch (L2) | Wave 2.B | Elaboration of ADR-011 |
| 5 | Result Summarizer Harness reconsideration (summary-with-pointer; selective skip) | Tool Dispatch (L2) | Wave 2.A + 2.B + 3.A empirical | Amendment territory |
| 6 | Upward L0→L1 read-only signal channel for cross-layer calibration | Architecture-level | Wave 2.B | **Amendment to ADR-002 layering rule** |
| 7 | Tool-call structural validation (phantom-call guard) | Tool Dispatch (L2) | Wave 3.A empirical | Class (a) elaboration; precedent in `9f86d0b` |

### Refinement 3 — Autonomous routing partial validation; Cycle 3 grounding action 2 is partially closed

Wave 3.A's Trial 1 produced a positive autonomous-routing observation at the cheap-cloud tier on this fixture (single trial). This is the *first direct positive evidence* for the cycle's autonomous-routing question — Spike D was manually staged; Spike C used `llm-orc invoke` directly (not orchestrator-dispatched). N=1 is a non-zero signal, not full validation. Cycle 5's multi-iteration spike is positioned to extend this.

The cycle's framing of autonomous routing should reflect this: **the existing class (c) primitives (closed five-tool surface, biased default system prompt, Result Summarizer Harness positioning) are *sufficient* for autonomous routing on a single bounded fixture at the cheap-cloud tier; whether they remain sufficient under multi-iteration accumulation is the next empirical question.**

### Refinement 4 — The cycle's claim scope on long-horizon reliability infrastructure

Wave 3.A could not test the cycle's long-horizon-reliability claims because the spike is single-iteration by design. The cycle's claims on externalized state, initializer-then-resume, calibration-gated cross-layer composition, and per-role configurability remain *literature-grounded but not cycle-empirically-grounded.* This is honest scope-of-claim discipline (Essay 003-style) and should be recorded explicitly in the essay: the cycle's design-method synthesis on long-horizon infrastructure is supported by converged literature best-practice (Anthropic, OpenDev, HELM, Claude Code, Confucius Code Agent) but has not been empirically validated within this cycle's deployment shape. Cycle 5+ territory.

---

## Research Phase Close-Shape Decision (post-Wave 3)

**Decision:** Mode B+ → DECIDE per practitioner gate (Cycle 4 cycle-status update).

The cycle has produced:
- A composite design-method framing (A-B-C-D) grounded in 47+ literature sources across three lit-review waves and refined by one behavioral spike
- Seven ADR candidates (Wave 1+2 surfaced six; Wave 3.A added the seventh)
- One architectural-extension candidate (upward L0→L1 signal channel) requiring layering-rule amendment to ADR-002
- An empirically-motivated summarizer-reconsideration finding strengthening ADR candidate #5
- A new failure-mode hook (phantom tool-call) adding ADR candidate #7

The cycle is positioned to close the research phase with essay → audits → reflection → susceptibility snapshot → gate, then advance into DECIDE phase on the seven ADR candidates. Mode A pipeline through BUILD remains deferred — seven ADRs is substantial DECIDE workload.

**Wave 3 status: complete.**
**Research phase status: ready for essay synthesis.**



### Iteration 1.B — Analytical decomposition: RDD-cycle demand mapping + mechanism attribution against Cycle 3 spike findings

**Method:** Read the architecture's load-bearing artifacts (`system-design.md`, essay 003 four-axis frame, Cycle 3 spike logs 004d/004e, Cycle 3 lit-review 004a) and walk the RDD methodology phase-by-phase to identify (a) cognitive surface, (b) long-horizon reliability demands, (c) hardness for the cheap-orchestrator + ensemble pattern, and (d) the architecture-layer demand the phase places. Mechanism attribution then maps the empirical evidence to the four-layer architecture for each demand class.

#### Part A — RDD-cycle phase-by-phase demand decomposition

The phase decomposition uses three orthogonal axes: cognitive surface (what kind of work), session demand (long-horizon reliability requirements), and decision-class (one-shot procedural at known transition points vs. continuous routing under variable inputs). The decision-class axis is load-bearing for Sub-Q6's transfer test — RDD's "structural hooks at phase transitions" finding addresses one-shot procedural decisions specifically.

| Phase | Cognitive surface | Long-horizon session demand | Decision-class mix | Hardness for cheap-orchestrator + ensemble |
|-------|------------------|----------------------------|-------------------|-------------------------------------------|
| **RESEARCH** | Question articulation, lit-review dispatch, spike runs, essay synthesis, audit-loop iteration, gate conversation | Multi-day; iterative essay revision triggers re-audit; spike findings can refute prior framings | Heavy mix: audit dispatches and specialist-agent dispatches at *known transition points* (one-shot procedural, fits RDD's structural-hook pattern); continuous routing on which spike to run, when to loop, when to revise the essay (variable inputs) | **HIGH.** Heavy specialist-dispatch surface is well-suited to `invoke_ensemble` at transition points. Continuous routing decisions are where Sub-Q6 lands. |
| **DISCOVER** | Stakeholder mapping, jobs analysis, value tensions, assumption inversions; conversational with practitioner | Within-session in update mode; multi-session in backward mode for established systems | Mostly continuous routing; one-shot procedural on inversion checklist | MEDIUM. Conversational nature limits ensemble role; orchestrator + tools dominates. |
| **MODEL** | Vocabulary extraction; invariant identification; glossary curation; **backward propagation when invariants change** | Within-session for ordinary updates; cross-cycle for invariant amendments (sweeps all prior artifacts) | One-shot procedural on amendment-trigger; continuous routing on per-concept extraction | MEDIUM-to-HIGH. Backward propagation is a structural sweep — strong candidate for specialist-agent dispatch (per-area scanners with MARG-style concatenation of findings). |
| **DECIDE** | ADR drafting (one at a time); argument-audit dispatch per ADR; scenario writing; interaction-spec writing; **conformance audit against code** | Multi-session; ADR drafts often refined over days; conformance audit produces debt list that informs scenarios | Heavy mix: argument-audit and conformance-scanner dispatches are one-shot procedural at known points; ADR drafting is continuous routing | **HIGH.** Specialist-dispatch surface is heavy here (argument-auditor, conformance-scanner). Decision-class transfer test most clearly applies. |
| **ARCHITECT** | System design; responsibility allocation; fitness criteria; dependency graph; integration contracts; design amendments | Multi-cycle; design evolves; amendments tracked in design log | Mostly continuous routing; one-shot procedural on susceptibility-snapshot dispatch | HIGH. Structural reasoning resists fragmentation into ensemble dispatch; orchestrator + tools dominates with selective specialist-agent slots (snapshot, framing audit). |
| **BUILD** | Outer loop composing debug/refactor/review as mode shifts; BDD/TDD red-green-refactor; stewardship checkpoints; integration verification | **Longest phase by far** — can span dozens of sessions; deepest working-memory demand; recoverability and branching dominant | Heavy continuous routing (which mode, which file, when to commit); one-shot procedural on test-suite runs at red/green boundaries and on review-skill invocation | **HIGHEST.** This is where the long-horizon failure modes (HORIZON, LongCLI-Bench, AMA-Bench) most operate. Working memory, branching, recoverability are dominant demands not fully covered by `invoke_ensemble`. |
| **DEBUG** | Hypothesis-trace-understand-fix cycle (inner loop within BUILD) | Within-session typically; can extend across sessions for hard bugs | Continuous routing; one-shot procedural on test-runs and assertion checks | HIGH on hard bugs. Deterministic verifier slot (script-models) is high-value; ensemble role limited unless multiple hypotheses warrant parallel exploration. |
| **REFACTOR** | Three-level diagnostic-remediation; AI Smell Taxonomy detection (inner loop within BUILD) | Within-session typically | One-shot procedural on smell-detection scans; continuous routing on remediation order | MEDIUM. Heterogeneous reviewer ensemble plausibly valuable for smell detection (Spike A3 pattern adjacent). |
| **REVIEW** | Code review; question-driven orientation (inner loop within BUILD or standalone) | Within-session typically | One-shot procedural on review dispatch; continuous routing on follow-up questions | MEDIUM-to-HIGH. **The architecture's strongest demonstrated case** — Spike C's deterministic-probabilistic complementarity is exactly this surface. |
| **PLAY** | Stakeholder inhabitation; gamemaster facilitation; field-note recording | Within-session typically | Mostly continuous routing | LOW-to-MEDIUM. Conversational; orchestrator + read-tools dominates. |
| **SYNTHESIZE** | Artifact trail mining (read full corpus); synthesis conversation; essay outline production; citation + argument audit on outline | Within-session for the synthesis pass; the corpus reading is large-context | Heavy mix: corpus mining is continuous routing; audit dispatches at end are one-shot procedural | HIGH. Long-context reading is the binding demand; lit-reviewer-style specialized scan plausibly valuable. |

**Decision-class clusters (the load-bearing structural finding from this decomposition):**

The phases break cleanly into three clusters by dominant decision-class. The clustering is consequential for how Sub-Q6's transfer-test resolves at each phase:

- **Cluster 1 — Heavy specialist-dispatch at known transition points** (RESEARCH, DECIDE, SYNTHESIZE; portions of MODEL on invariant amendment): the orchestrator's primary job is correct dispatch at structurally-determined moments. Audit dispatches, lit-reviewer invocations, conformance scans, susceptibility snapshots — all fire at specific phase positions in the skill text. **Sub-Q6's transfer of RDD's structural-hook finding is most plausible here** — the decision-class match is direct (one-shot procedural at known points), and the failure mode RDD documented (context-growth degrades agent judgment to follow a directive) is the exact failure mode this cluster faces.

- **Cluster 2 — Heavy continuous routing under variable inputs** (BUILD, ARCHITECT; portions of DEBUG, REFACTOR): the orchestrator's primary job is moment-to-moment routing under inputs that are not predictable from the phase position. Which file to read, which test to run, when to commit, when to mode-shift between debug/refactor/review. **Sub-Q6's transfer of RDD's structural-hook finding is least plausible here** — the decision-class differs (continuous routing, not one-shot procedural), and the intervention class likely shifts toward context-management primitives (working-memory consolidation, sliding-window summarization, retrieval-grounded routing) rather than harnesses at specific decision points.

- **Cluster 3 — Conversational / exploratory** (DISCOVER, PLAY, conversational portions of MODEL): the orchestrator's primary job is responsive depth in conversation with the practitioner. **Ensemble role is limited** — the practitioner is in the loop continuously; the architecture's value comes from orchestrator + read-tools + memory.

The North-Star benchmark ("drive a full RDD cycle") demands all three clusters in sequence. This is informative for the cycle: design methods that work for one cluster may not transfer to another. The methodology-design surface is therefore plural, not singular — this is a refinement to the cycle's research-entry framing that the analytical decomposition surfaces.

#### Part B — Mechanism attribution against Cycle 3 spike findings

The empirical evidence base for mechanism attribution is concentrated in Cycle 3's two well-instrumented spikes (Spike C cross-file verification, 8 trials; Spike D multi-stage workflow, N=1 per arm) plus Cycle 2's three spikes (Spike A cascade-vs-prompt-steering on README review; Spike A3 novel ensemble; Spike B small-model substitution). Cycle 1 contributes the CAP-9 hybrid deployment validation and the local-only-CPU-bottleneck finding.

| Demand | Empirical evidence | Mechanism (load-bearing layer) | Confidence |
|--------|-------------------|-------------------------------|-----------|
| **Cross-file verification** (e.g., DECIDE conformance audit, BUILD test-output checking, MODEL invariant cross-reference) | Spike C: Arm B 3/3, Arm A 1/3, Arm C 0/2 on concrete verification of `100_000` vs `50_000_000` | **Script-models layer** (deterministic file access). Ensemble role is *wrapper*, not mechanism. Cycle 3 grounding action 1 confirms. | HIGH (8 trials, clean signal) |
| **Documentation review with per-recommendation specificity** (e.g., REVIEW phase, DECIDE rejected-alternatives surfacing) | Spike A: A2 (prompt-steered) outperforms A1 (cascade-with-collapse). Spike A3: MARG-concatenation + script + heterogeneous LLMs surfaces undefined model profiles A2 missed | **Composition pattern** — the no-collapse-summarization (MARG) pattern preserves what cascades destroy. Heterogeneity-uncorrelated-errors layers on top. | MEDIUM-HIGH (multiple trials; load-bearing alternative-reading recorded) |
| **Multi-stage workflow with deterministic verification** (e.g., RESEARCH spike-run → essay-synthesis → audit cycle) | Spike D pilot: B1 (cheap+multi-ensemble) ties C1 (frontier-bare) at 4/5 — at zero $ cost; staged manually | **Cross-layer composition** (orchestrator routes between specialized ensembles; script-models verify; LLMs synthesize). Manually staged; autonomous routing untested. | LOW-MEDIUM (N=1 per arm; manually staged) |
| **Long-horizon multi-turn coding** (e.g., BUILD phase, DEBUG iteration) | Cycle 1 PLAY: user passivity in face of silent failure was dominant experiential mode. Cycle 2 essay 003 §"Long-horizon performance degrades super-linearly" cites HORIZON 19% meltdown rate, AMA-Bench 57% memory accuracy ceiling, LongCLI-Bench <20% pass rate | **Currently uncovered.** No layer of the four-layer architecture directly addresses long-horizon reliability primitives (working memory, retrieval-grounded routing, sliding-window summarization). Cycle 3 cycle did not test. | EVIDENCE GAP — Sub-Q1 + Sub-Q6 territory |
| **Routing intelligence** (orchestrator chooses which ensemble for which sub-task) | Spike A: MiniMax used Read autonomously in trial 1 only (1/3). Spike D: explicit staging; autonomous routing not tested | **Currently uncovered for autonomous case.** Cycle 3 grounding action 2 names this as the evidence gap. The orchestrator's tool-choice variability (Spike A 1/3 trials) suggests routing intelligence is real-but-variable at the cheap-orchestrator tier. | EVIDENCE GAP — Cycle 4 territory |
| **Hybrid deployment economics** (cloud orchestrator + local ensembles) | Cycle 1 CAP-9: validated empirically. Local-only deployments hit per-turn input-context throughput bottleneck (15K+ token tool schema) on consumer CPU at 5-8 tokens/sec | **L3 Entry layer + L2 Runtime layer** — the orchestrator's remote-inference-friendly cheap path closes the local-CPU bottleneck. | HIGH (Cycle 1 validated) |
| **Structural enforcement (e.g., MARG-concatenation aggregation)** | Spike A3: Python-harness-enforced MARG concatenation preserved per-reviewer output structurally rather than relying on orchestrator compliance with a "do not synthesize" instruction | **Script-models layer** as enforcement substrate; the harness is a deterministic wrapper around the orchestrator's natural tendency to collapse | MEDIUM-HIGH (Cycle 2 evidence; pattern documented; reviewer Flag 4 names it as inheritance) |

**The unifying mechanism the corpus has surfaced (essay 003 §"What this means for the cycle"):** The architecture's value, abstracted across spikes, is **composition of components whose error distributions are different enough that the composition's coverage exceeds any single component's**. Spike A3 instantiates this through heterogeneity-across-LLM-families (different LLMs have uncorrelated errors). Spike C instantiates it through deterministic-vs-probabilistic complementarity (script has zero error distribution where LLM has nonzero). Spike B refined it as task-class-dependent (no complementarity on simple multi-turn protocols).

This unifying frame is consequential for Sub-Q3's primary scoping decision: **the architecture's value is not located in a single layer.** Cross-file verification: script-models is mechanism. Documentation review with specificity: composition pattern (MARG + heterogeneity) is mechanism. Hybrid deployment economics: orchestrator routing + tool dispatch is mechanism. The unifying property is structural compensation via different error distributions, instantiated layer-conditionally per task class.

#### Part C — Sub-Q3 candidate findings (tentative; pending Wave 1.A lit-review confirmation)

The analytical decomposition produces a **layer-conditional** answer to Sub-Q3 rather than a single-layer answer. The five candidate hypotheses at cycle entry mapped to:

- **Hypothesis A** (script-models is load-bearing): Spike C supports for cross-file verification specifically.
- **Hypothesis B** (ensemble composition is load-bearing): Spike A3 supports for documentation review with specificity.
- **Hypothesis C** (orchestrator routing is load-bearing): Cycle 1 CAP-9 supports for hybrid deployment economics.
- **Hypothesis D** (cross-layer composition is the irreducible unit): the unifying-frame finding (structural compensation via different error distributions) supports — *with the layer-conditionality nuance that the irreducible unit is composition-of-components-whose-error-distributions-differ, but the components themselves are layer-typed*.
- **Hypothesis E** (emergent four-layer property): not directly supported by the spike evidence; Plexus layer is not yet exercised (ADR-002 deferred).

Tentative finding: **Hypothesis D refined** — the architecture's value is located in *cross-layer composition where components have measurably-different error distributions*, with the load-bearing layer per task class determined by which complementarity surface the task most exposes. This refinement makes Sub-Q4's design-method scope plural: composition shapes for ensembles (MARG, heterogeneity, calibration-gated), composition shapes for script-models (deterministic verification, structural enforcement, harness wrappers), composition shapes for orchestrator routing (hybrid deployment, autonomy gating, summarization discipline) — and the cross-layer composition shapes that bind them.

If lit-review #1 (Wave 1.A) confirms long-horizon reliability primitives (working memory, sliding-window summarization, retrieval-grounded routing) as a separate intervention class operating *across* layers rather than within any one layer, the layer-conditional finding extends naturally: long-horizon reliability is *cross-cutting infrastructure*, not a layer property. This would be the cycle's largest framing-shift candidate.

#### Part D — Implications for Wave 2 lit-review scope

Given the tentative Hypothesis-D-refined finding, Wave 2 lit-review #2 should cover:

1. **Composition shapes per layer** (not just for ensembles). The original Sub-Q4's six candidate composition shapes were ensemble-shaped. Wave 2 should also survey: (a) composition patterns for deterministic tool-models (script-as-orchestrator literature: Routine, Compiled AI, structured workflow libraries — partial coverage in 004a's §3.1.1); (b) composition patterns for orchestrator routing (supervisor patterns, hybrid-deployment routing literature, autonomy-gated dispatch); (c) cross-layer composition primitives (which the four-layer architecture has limited literature support for — most published systems are single-layer or two-layer).

2. **Long-horizon reliability primitives as candidate cross-cutting layer.** If Wave 1.A's lit-review surfaces working-memory consolidation, retrieval-grounded routing, or sliding-window summarization as named intervention classes, Wave 2 should evaluate whether these constitute a *fifth layer* in the architecture or are properly positioned as cross-cutting concerns within existing layers. This is a candidate framing-shift the cycle should hold open.

3. **Calibration Gate role at cross-layer composition.** ADR-007's Calibration Gate transitions composed ensembles to "trusted" only after last-N positive signals. The corpus has not yet exercised this in production. Wave 2 should surface whether published calibration-gate-style mechanisms exist for cross-layer composition (e.g., script-grounded ensemble outputs gated on script's deterministic findings).

4. **Ensembles-of-ensembles as named candidate composition shape (essay 003 motivation).** Essay 003's Conclusion explicitly names ensembles-of-ensembles as the well-scoped Cycle 4 seed — recursive composition aggregating across multiple A3-style ensembles for different review framings, *provided the meta-aggregation step also avoids a synthesizer collapse* (a design choice, not an automatic property of recursion). This directly engages Cycle 4's Sub-Q4 composition-shape design space: the user's Step 1.2 framing of "stock ensembles composed and accumulated on the fly" is in this territory. Wave 2 should treat recursive ensemble composition as a candidate shape alongside the original six (stock + on-the-fly, decision-tree dispatch, parameterized templates, generative construction, typed contracts, no-ensembles baseline) — making seven candidates with the explicit caveat from essay 003 about meta-aggregation collapse risk.

#### Part E — Sub-Q5 hardness ranking

The most demanding RDD phases for the cheap-orchestrator + ensemble pattern, ranked:

1. **BUILD** (highest demand). Long-horizon working memory, recoverability, branching, mode-shifting between debug/refactor/review. Decision-class is mostly continuous routing where Sub-Q6's transfer is least clean. Long-horizon-reliability literature (Wave 1.A) is the load-bearing input. Currently-uncovered demand for working-memory primitives.
2. **RESEARCH** (high demand on specialist dispatch reliability). Audit dispatches (citation, argument, framing, susceptibility) at known transition points are exactly where RDD's "structural hooks at phase transitions" finding originated. Sub-Q6's transfer is most clean here. Spike A3's Python-harness-enforced MARG-concatenation is a candidate generalization (Flag 4 inheritance).
3. **DECIDE** (high demand on argument-audit + conformance-audit dispatches and ADR-drafting reasoning quality). Mix of one-shot procedural and continuous routing. Sub-Q6 partially applies.
4. **SYNTHESIZE** (high demand on long-context corpus reading). Resembles RESEARCH in audit-dispatch structure but with different cognitive surface (narrative synthesis vs. investigation).
5. **MODEL** with invariant amendments (high on backward propagation; low otherwise).
6. **ARCHITECT** (high but resists ensemble fragmentation).
7. **REVIEW** (medium-high; the architecture's strongest demonstrated case via Spike C).
8. Lower-demand phases: REFACTOR, DEBUG, DISCOVER, PLAY, GRADUATE, CONFORM.

**Wave 3.A behavioral spike candidate (per Flag 1 resolution):** RESEARCH-phase research-loop is the natural choice for a tightly-scoped behavioral spike. It exercises (a) the audit-dispatch chain that maps cleanly to `invoke_ensemble`, (b) the lit-reviewer specialist-dispatch pattern that exercises the orchestrator's correct-dispatch-at-known-points discipline, and (c) a finite scope (one research loop on a focused question, not an open-ended cycle). The benchmark would test driving one research-loop iteration via `llm-orc serve` direct API path (not opencode CLI per Spike D's stall finding). Decision deferred until Wave 1.A lit-review completes and Sub-Q6's transfer-test framing is sharpened.

#### Part F — Open from analytical decomposition

The analytical decomposition surfaces three open questions Wave 1.A's lit-review should address (already in scope) and one that may need a third lit-review or focused web-search at Wave 2:

- **Open #1** (Wave 1.A territory): Are working-memory consolidation, retrieval-grounded routing, sliding-window summarization, and progressive-disclosure context management treated as named intervention classes in the literature? If so, do they constitute a coherent "long-horizon reliability infrastructure" layer?
- **Open #2** (Wave 1.A territory): Does the literature distinguish phase-transition-decision reliability from continuous-routing reliability as separate failure classes?
- **Open #3** (Wave 2 territory): What is the published evidence on calibration-gated cross-layer composition (script outputs gating LLM ensemble outputs; ensemble outputs gating orchestrator decisions)?
- **Open #4** (possibly third lit-review or focused search): What does the research-agent / scientific-reasoning literature say about long-horizon research workflows specifically? RDD's North-Star benchmark is research-cycle-shaped, not coding-task-shaped; the published agent literature predominantly tests coding/reasoning task classes.

**Wave 1.B status: complete pending Wave 1.A return.** The analytical decomposition produces a tentative Hypothesis-D-refined finding for Sub-Q3, a phase-cluster decomposition for Sub-Q5, a sharpening of Sub-Q6's transfer-test scope (decision-class-conditional), and a research-phase candidate for the conditional Wave 3.A behavioral spike. Confirmation pending lit-review #1.


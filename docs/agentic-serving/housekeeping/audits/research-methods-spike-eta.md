# Research Design Review — Spike η (Deliverable Enumerator)

**Reviewed question set:** `docs/agentic-serving/essays/research-logs/cycle-7-spike-eta-deliverable-enumerator.md`
**Constraint-removal response included:** n/a (the pre-reg is a direct forward question, not a constraint-removal framing; the ADR-040 DECIDE gate directive functions as the originating constraint)
**Date:** 2026-06-10

## Summary

- **Questions reviewed:** 1 compound hypothesis with 3 failure modes and a 4-arm design
- **Flags raised:** 10 (2 P1 / 5 P2 / 3 P3)
- **Criteria applied:** 1–4 (ADR-082 corpus)

---

## Per-Question Review

### Question 1: "Can a structurally-bounded deliverable enumerator predict the deliverable filename set accurately enough that the deterministic completeness gate extends to enumerable-but-unnamed tasks?"

**Belief-mapping:** What would the researcher need to believe for a different question to be more productive?

The question assumes that the right fix is a new bounded role (the enumerator) that predicts filenames the task does not name. A different question would be more productive if any of the following were true:

- The task description could be enriched at the client side (e.g., the client names files before submission), making the enumeration problem vanish rather than be solved inside the framework. The pre-reg does not ask "should the framework solve this, or should the interface contract require tasks to name their deliverables?"
- A count-only variant is sufficient: if the gate only needs to know how many deliverables to expect (not their names), a simpler extraction that counts described deliverables and triggers COMPLETE when that many distinct writes appear sidesteps the namespace-matching problem entirely.
- The judge, fed a requested-minus-produced digest rather than a produced-only digest (J-2, the path Spike σ identified and then bypassed for J-3), performs reliably on unnamed tasks even without an enumerator. Spike σ only measured J-2/J-3 on named-file tasks; J-2 on unnamed tasks is an open question the pre-reg treats as closed.

**Embedded conclusions:**

Flagged: the pre-reg frames the spike's question as "whether the enumerator holds on the cheap local model," but its decision rule treats "keep the judge for unnamed tasks" as the failure outcome, not as an independent viable answer. This embeds the conclusion that the judge-path is inferior to the enumerator-path for unnamed tasks, without establishing that. Spike σ proved the judge unreliable for named-file tasks because the produced-only digest is the information limit. Whether that same information limit applies when the enumerator populates a requested-vs-produced checklist (J-2 shape, but enumerator-seeded) rather than a filename-regex extraction is not addressed.

Suggested reformulation: "Does a deliverable-enumerator role improve session convergence on unnamed tasks relative to a judge fed an enumerator-seeded requested-vs-produced checklist, and is either preferable to requiring the task to name its deliverables?"

**Scope:** Appropriately narrow on the qwen3:8b measurement boundary and small task shapes. The scope statement is honest. However, the framing concentrates all attention on whether the enumerator works, without making "what happens when it does not work" a first-class experimental question (see P1-A below).

---

### Question 2 (implicit): "Is the enumerable-vs-semantic split sound as the general partition of completeness decision shapes?"

**Belief-mapping:** The pre-reg treats the split as established at design time, not as a hypothesis under test. What would you need to believe for the split to be wrong or mislocated?

You would need to believe that "summarize this meeting" and "build a temperature library with conversion functions, tests, a CLI, and docs" are not categorically different from the completeness gate's perspective — that both admit a lightweight deliverable count (one response object vs five files), and a count-only gate handles both without enumeration. Under that belief, the relevant question is "what is the minimal completeness signal" rather than "where does enumeration stop."

**Embedded conclusions:**

The pre-reg asserts that irreducibly semantic tasks ("summarize this meeting") are handled by natural-finish detection ("a final response was produced") with no enumeration. This is stated as resolved, not as a question the spike will test. If the spike's enumerator confabulates deliverables on a task that is actually irreducibly semantic (misclassifies it as enumerable-but-unnamed), the framework would silently loop waiting for files that do not exist. The pre-reg acknowledges over-enumeration as a failure mode but frames it as "predicts extras the task did not request." A more adversarial framing is "predicts filenames for a task that requested no files at all." These are the same failure mode but at very different severity levels; the pre-reg's task battery (all three tasks are enumerable-but-unnamed by construction) cannot catch the boundary misclassification.

Suggested addition: include at least one irreducibly-semantic task in the battery to confirm the enumerator does not confabulate deliverables on tasks that belong to the natural-finish branch.

**Scope:** Too narrow as stated — the task battery excludes the boundary cases that would validate the partition.

---

### Question 3 (implicit): "Does the wrong-plan decoy control isolate enumeration content as the causal mechanism?"

**Belief-mapping:** What would you need to believe for the control to be uninformative?

You would need to believe that feeding a wrong enumerated set is tautological rather than informative — that the prediction "a wrong plan produces a wrong-but-terminating session" is true by construction (the gate does a set comparison; if the set is wrong, it converges to the wrong set) rather than being a genuine falsifiable claim. If the gate mechanically converges to whatever set it is given, the control only confirms the mechanism of the gate, not its fitness. A more informative control would test what happens when the plan is wrong AND the coder deviates from it (does the gate detect the divergence or declare COMPLETE anyway).

**Embedded conclusions:**

Flagged: "the plan drives, so a wrong plan must produce a wrong-but-terminating session" is stated as the expected result and cited as isolating that enumeration content is the mechanism. But this conflates two separate claims: (a) the gate converges to the enumerated set (a gate-mechanics claim, already true by construction from the diff logic), and (b) the coder follows the enumerated set (an LLM behavior claim that has not been tested). If the coder ignores the plan and writes files with different names, the gate would wait forever (or hit AS-3), and the control would show a non-terminating session rather than a wrong-but-terminating one. The control result depends on whether the coder is obedient to the plan. This is worth surfacing as a separate observation, not folded into the mechanism claim.

**Scope:** Appropriate as a structural check, but the interpretation requires more precision than the pre-reg gives it.

---

## Question Set Assessment

### Premature narrowing / prior-art treatment

**Flagged (P1-A): The J-2 path is dismissed without measurement on unnamed tasks.**

ADR-040 and Spike σ established that the judge fails on the produced-only digest for named-file tasks because the information is missing. But ADR-040 itself describes J-2 (feed the judge a requested-vs-produced checklist) as the "minimal" fix before J-3. The pre-reg jumps directly to J-3 generalization (enumerator-as-plan) for unnamed tasks and treats J-2 on unnamed tasks as a non-option, because "the requested set must come from somewhere." That is true — but J-2 seeded by the enumerator's output is a coherent option that requires no live multi-turn session: the enumerator fills the checklist; the judge reads it. This is architecturally lighter than the enumerator-as-plan (the enumerator is still needed, but it seeds a judge-read checklist rather than becoming a hard convergence target). The spike should either add a J-2-seeded arm or explicitly argue why it is dominated by the enumerator-as-plan before running.

Suggested revision: Add arm D (enumerator seeds a J-2 checklist; judge reads requested-vs-produced; enumerator is not the convergence target). If adding an arm is rejected, add a pre-reg note explaining why J-2-seeded is strictly dominated.

---

**Flagged (P2-A): The count-only fallback is not examined.**

The pre-reg acknowledges that "the enumerator must predict deliverable filenames" for the namespace to match, but does not consider whether a count-only approach (the framework tracks "N deliverables described; N distinct files written; COMPLETE") avoids the confabulation surface entirely. A count does not require the enumerator to name the files; it requires only an integer. This is simpler, has a smaller confabulation surface (integers versus filename lists), and composes with the existing requested-vs-produced diff (count as the threshold, set membership as the secondary check). The pre-reg should acknowledge and reject this, or test it.

---

**Flagged (P2-B): Client-side interface contract alternative is not examined.**

The pre-reg treats unnamed tasks as a given constraint: the client did not name files, so the framework must infer them. But the gate's scope (named-file tasks are mechanically solvable; unnamed tasks require inference) is itself a design choice. The spike does not ask whether the correct response to the DECIDE gate's "thin slice" objection is to define an interface convention that requires tasks to name deliverables (and document the limitation for tasks that do not), rather than adding inference inside the framework. This is not necessarily the right answer — but it is a live alternative that the spike forecloses by assumption.

---

### Incongruity surfacing

**Flagged (P1-B): The enumerator-as-plan reintroduces the confabulation surface ADR-038 rejected, at a different complexity level, without a structural argument for why this crossing is now safe.**

ADR-038 explicitly rejected the "per-task routing-planner (decompose → track → dispatch)" alternative, with the primary reason: "a plan-ahead role narrates a deliverable sequence it has not yet observed, the ungrounded composition AS-9's note-22 case removed." The pre-reg is aware of this — it names the deliberate re-opening — but its structural argument for why the enumerator avoids the ADR-038 failure is weaker than it appears.

ADR-038's rejection was grounded in a structural principle (plan-ahead roles are ungrounded), not in a measured failure at qwen3:8b for deliverable enumeration. Spike ζ is cited as prior evidence, but it tested routing capability selection (which ensemble to dispatch to), not deliverable enumeration (which files to produce). The tasks are structurally different: routing selects from a fixed closed set of six ensembles with explicit capability descriptions in the system prompt; deliverable enumeration predicts open-ended filenames from an unbounded task description. The transfer of ζ's 90% strict-accuracy result to this domain is asserted, not established.

The pre-reg's "honest scope" section records this ("Spike ζ validated a structurally-bounded routing-planner at qwen3:8b... for capability selection. That is a different planning decision than deliverable enumeration"), which is appropriate transparency, but then the hypothesis and decision rule treat ζ as positive prior evidence for the enumerator. This is the incongruity: the spike design leans on ζ as grounding in the same sentence it acknowledges ζ does not transfer. The spike should be explicit about what ζ actually licenses: that AS-9-shaped bounded roles do not categorically collapse at qwen3:8b, nothing more. The precision/recall measure in arm C is the right test; ζ is not prior evidence for the decision rule threshold.

Revision: remove ζ from the decision rule framing; note that ζ establishes AS-9-viability-in-principle for a different decision shape, and arm C is the actual grounding for this specific decision.

---

### Coverage gaps

**P2-C: n per arm and early-stop rules are absent from the pre-reg.**

The decision rule names candidate thresholds (recall ≥ 0.9, B convergence ≥ 0.8) but defers n to "set with the methods reviewer." This is the correct deference, but it means the pre-reg is not fully pre-registered: the thresholds are candidates and n is undefined. A partial pre-registration is weaker than a full one. The spike should commit to n before running, not use "methods reviewer pending" as a placeholder.

Concrete recommendation:
- Arm C (isolated enumeration accuracy): n = 15 tasks minimum, covering the three described task shapes plus at least two boundary tasks (an irreducibly semantic task and a task with implicit deliverable count only). Score precision and recall; decision threshold recall ≥ 0.9.
- Arms A/B (live sessions): n = 6 per arm given $0 local economics and the 3-task battery (2 runs per task shape minimum). Early-stop for arm B: GROUNDED if 5/6 converge to intent; INSUFFICIENT if 3 premature finishes occur before 6 runs.
- Control (wrong-plan decoy): n = 3 (one per task shape); purpose is mechanism confirmation, not rate estimation.
- Arm A baseline: run until stall-or-judge-failure confirmed in at least 3 sessions; stop there (same discipline as Spike σ's baseline arm).

Note: asymmetric weighting (under-enumeration worse than over-enumeration) must be operationalized as: a recall miss counts as a spike failure even if precision is high; over-enumeration is flagged but does not fail the spike unless it causes AS-3 termination in >1/6 arm B sessions.

---

**P2-D: Intent-divergence measure is under-specified.**

The pre-reg acknowledges that intent divergence is a human read, records it as a limitation, and calls it "the irreducible residual." This is honest, but the spike relies on intent divergence as a secondary measure informing the decision rule ("arm B live sessions converge to intent"). If the intent-divergence call is subjective and post-hoc, it can absorb a borderline result in either direction. The spike should pre-register the intent-divergence criteria: what counts as a divergence? Suggested criteria: the produced file set differs from the reference deliverable set in name by more than one file (accounting for reasonable renaming), or a produced file serves a clearly different purpose than the referenced deliverable. This is still a human call, but grounded to a stated criterion.

---

**P3-A: The "irreducibly semantic" branch is not validated in the spike.**

The pre-reg asserts natural-finish detection handles irreducibly semantic tasks ("a final response was produced," "one adequate response"). This is a reasonable claim but is not tested. If the spike lands and the enumerator ships, a semantic task that the enumerator mis-classifies as enumerable-but-unnamed would loop until AS-3 — the worst over-enumeration case. Including one irreducibly semantic task in arm C (with the expectation that the enumerator produces an empty or null set) would validate the boundary at $0 cost (one isolated probe, no live session needed).

---

**P3-B: The reference deliverable set construction is not defended.**

The pre-reg notes the reference set is "pre-registered before any run," which is necessary. But it does not address the construct validity question: for the temperature library task, is there a canonical decomposition? A task described without filenames admits multiple legitimate decompositions. One coder might name the conversion module `temperature.py`; another might name it `converters.py`. The reference set should record whether matches are exact-name or basename-stem-flexible, and whether a produced file with a different name but clearly equivalent purpose (e.g., `converter.py` vs `converters.py`) scores as a hit or a miss. This is the pre-reg's own "namespace sharing" problem in the evaluation metric.

---

**P3-C: Turns-to-converge as a secondary measure needs a baseline comparison to be useful.**

The pre-reg lists turns-to-converge as a secondary measure. Without a baseline (how many turns the named-file ADR-040 path takes on equivalent tasks), this number is uninterpretable. The enumerator may converge in more turns than the named-file path (because a wrong enumerator prediction requires extra correction turns), and that cost should be visible. Arm A's turns-to-converge (or AS-3 cap) provides the baseline automatically, but the pre-reg should note this comparison explicitly rather than listing the measure in isolation.

---

## Consolidated Findings

| ID | Severity | Finding | Required Revision |
|----|----------|---------|-------------------|
| P1-A | P1 | J-2-seeded arm missing — the enumerator-as-plan is compared only against the current judge fallback, not against the lighter alternative of seeding a judge-read checklist with the enumerator's output | Add arm D (enumerator seeds J-2 checklist, judge reads requested-vs-produced) or explicitly pre-register why it is dominated before running |
| P1-B | P1 | Spike ζ transfer is asserted, not licensed — ζ's 90% on capability routing is cited as prior evidence for deliverable enumeration, but the pre-reg itself acknowledges the two tasks are structurally different; ζ establishes AS-9 viability-in-principle only | Remove ζ from the decision rule framing; state that arm C is the actual grounding; note ζ only establishes that bounded roles do not categorically collapse at qwen3:8b for single-decision tasks |
| P2-A | P2 | Count-only fallback not examined or rejected | Add a pre-reg note arguing why count-only is dominated by the named-enumerator approach, or add as an arm C variant |
| P2-B | P2 | Client-side interface contract alternative not acknowledged | Add one sentence acknowledging "require tasks to name deliverables" as an alternative and the reason it is not preferred (e.g., incompatible with real-client behavior) |
| P2-C | P2 | n per arm and early-stop rules undefined | Commit to concrete n (recommended: arm C n=15, arms A/B n=6, control n=3) and early-stop rules before running; operationalize the asymmetric under/over-enumeration weighting |
| P2-D | P2 | Intent-divergence criterion is under-specified as a post-hoc human read | Pre-register the divergence criteria (e.g., produced set differs from reference by >1 file in name, or a produced file serves a clearly different purpose) |
| P3-A | P3 | Irreducibly semantic boundary not tested | Add one semantic task to arm C battery; expected result is null/empty enumerator output; validates the enumerable-vs-semantic partition at $0 cost |
| P3-B | P3 | Reference deliverable set construction does not address name-flexibility | Pre-register whether matches are exact-name or stem-flexible, and how near-synonymous names score |
| P3-C | P3 | Turns-to-converge secondary measure has no baseline comparison | Explicitly note arm A provides the turns-to-converge baseline; the measure is reported as the delta, not the absolute |
| P3-D | P3 | Wrong-plan decoy control interpretation conflates gate mechanics with coder obedience | Note the expected result distinguishes two sub-cases: coder follows wrong plan (wrong-but-terminating, confirms enumeration content is mechanism) vs coder deviates from wrong plan (non-terminating or unexpected set, informative about coder plan-following) |

---

## Recommendations (priority order)

1. **Add arm D or explicitly reject J-2-seeded before running (P1-A).** The enumerator-as-plan vs enumerator-seeds-checklist distinction matters for the architecture claim. If arm D is added, the live session arms (A/B) stay as-is; arm D is an isolated probe like arm C. If it is rejected without measurement, the rejection must rest on a stated argument (e.g., the enumerator-as-plan and the J-2-seeded approach have identical information requirements but the plan is more fragile because the coder must follow it; the checklist only requires the judge to read it).

2. **Decouple ζ from the decision rule (P1-B).** This is a one-sentence revision in the hypothesis and a removal from the "prior evidence" framing in the decision rule. Arm C carries the actual grounding.

3. **Commit to n and early-stop rules (P2-C).** Use the values above or justify different ones. The asymmetric weighting (recall failure = spike failure; precision miss = flagged but not blocking unless it causes AS-3) must be stated explicitly.

4. **Pre-register intent-divergence criteria (P2-D).** One sentence is enough: name the threshold (e.g., >1 file name divergence from reference) and the tie-breaking rule.

5. **Add a semantic boundary task to arm C (P3-A).** One extra isolated probe; $0; validates the most consequential boundary condition in the pre-reg's partition claim.

6. **Address count-only and client-interface alternatives (P2-A / P2-B).** Short acknowledgments in the Rejected-alternatives section; do not require full argument, but the question should be visibly closed rather than silently excluded.

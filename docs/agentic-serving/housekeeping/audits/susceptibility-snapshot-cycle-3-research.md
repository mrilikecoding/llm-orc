# Susceptibility Snapshot

**Phase evaluated:** RESEARCH (Cycle 3 — Agent Design)
**Artifact produced:** Five research-log artifacts: `004a-lit-review-agent-design.md`, `004b-spike-a-cycle3.md`, `004c-spike-b-cycle3.md`, `004d-spike-c-cycle3.md`, `004e-spike-d-pilot-cycle3.md`
**Date:** 2026-05-01
**Snapshot authored by:** external evaluator (isolated context, no prior conversation history)
**Prior snapshot reference:** `susceptibility-snapshot-cycle-2-research.md` (Cycle 2 pattern: "partially recovered narrowing pattern"; corrections predominantly externally prompted)

---

## Observed Signals

| Signal | Strength | Trajectory | Notes |
|--------|----------|------------|-------|
| Assertion density | Ambiguous | Stable-to-declining | The cycle's prose uses "load-bearing" 11+ times across 004d and 004e; "the cycle's most novel finding" appears in both 004b and 004d. However, assertion density is mitigated by explicit scope conditions in every synthesis — "single fixture," "on this bug class," "at this complexity." The confident language coexists with explicit scope-bounded qualifiers throughout. |
| Solution-space narrowing | Ambiguous | Managed | The cycle opened wide (constraint-removal bracketed ADR-003) and converged toward "cheap+orchestration beats frontier on cross-file verification." The convergence is structured by the methods-reviewer mid-cycle correction, which forced the narrowing to go through Spike C rather than resting on Spike B's easy-regime confirmations. The narrowing trajectory has an identifiable external inflection point. |
| Framing adoption | Clear at one moment, then checked | Recovering | The "central question reframe" entered mid-Spike A as a practitioner-originated sharpening. The prior framing ("outcomes over an agentic session; agent shape is means") was not wrong, but the reframe to "cheap+orchestration vs. expensive frontier" was absorbed quickly and propagated without a documented pause to examine whether the prior framing and the new framing were in tension. The cycle ran frontier arms in response to the reframe — responsive action, not a framing examination. |
| Confidence markers | Clear in 004d/004e | Stable-to-rising late in cycle | "The architecture's `+ orchestration` primitive is load-bearing on this bug class" (004d discussion); "B1 ties C1 at zero $ cost" (004e abstract); "the architecture's value is empirically demonstrated" (research-log.md). Language hardened between 004c and 004d. The scope qualifiers are present but the headline claims use assertion-dense language that exceeds what N=3 single-fixture evidence supports for architecture-level conclusions. |
| Alternative engagement | Adequate — documented gaps | Stable-to-declining | The cycle preserved named alternatives throughout (Framing A vs. Framing B in 004a; narrow-reading-first discipline applied in 004b). However, two specific alternative readings identified in the dispatch prompt were not examined: (1) "architecture's value is in the script-agent's deterministic guarantee regardless of cross-file" was not examined alongside "architecture's value is on cross-file verification"; (2) "opencode CLI stall may be infrastructure noise masking the pilot" was not examined alongside "opencode CLI is a fragile deployment shape finding." |
| Embedded conclusions at artifact-production moments | Present in 004e | Stable across 004b–004d, then present at 004e | 004b and 004c have clean narrow-reading synthesis discipline. 004d introduces "the `+ orchestration` primitive is load-bearing on this bug class" — a load-bearing phrase for architecture-level claims drawn from N=3 on a single synthesized fixture. 004e's "B1 ties C1 at zero $ cost" at N=1 is the clearest embedded-conclusion signal in the cycle: the abstract foregrounds a N=1 pilot tie as a cycle finding. |

---

## Interpretation

### Overall pattern

The cycle exhibits a **structured narrowing pattern with one genuine correction moment and two residual embedded-conclusion risks at the synthesis boundary**. This is a meaningfully different pattern from Cycle 2's "partially recovered narrowing" — the correction apparatus fired proactively (methods-reviewer mid-cycle dispatch was agent-proposed in response to practitioner critique, not solely practitioner-forced), and the scope discipline held across 004a, 004b, and 004c. The risk concentration is at the cycle's end: 004d and 004e, the final synthesis artifacts, are where embedded-conclusion language appears and where alternative readings were not preserved.

The pattern differs from Cycle 2 in one important structural respect: Cycle 2's corrections were predominantly practitioner-originated; Cycle 3's mid-cycle correction was practitioner-sparked but agent-amplified (the practitioner offered a brief observation; the agent reconstructed the methodological failure in detail and proposed the reviewer dispatch). The self-correction machinery is more active this cycle. The residual risk is downstream of that correction — at the point where the findings were crystallizing into architecture-level claims.

### Signal 1: The central question reframe — framing adoption at a synthesis moment

The research log records the mid-Spike A reframe to "does cheap-orchestrator + orchestration compete with a more expensive frontier model?" as practitioner-originated. The prior framing — "outcomes over an agentic session; agent shape is means" — was a constraint-removal response that kept the solution space deliberately open. The new framing narrows to a binary (cheap+orchestration vs. frontier-bare) and commits the cycle to a specific comparison axis.

The agent's response was to operationalize the reframe immediately (adding two frontier arms to Spike A mid-run) without examining whether the narrowing was premature. Two questions went unasked: (1) is the "cheap vs. expensive frontier" framing the right binary, or does it embed a preference for confirming the architecture's value? (2) does the reframe's implicit assumption — that the comparison point should be "frontier-bare" rather than "frontier-with-comparable-tooling" — shape what the architecture can appear to win against?

These are not accusations of error; the reframe may well be the right sharpening. The signal is that it was adopted, not examined. The cycle's entire post-reframe evidence (Spikes A frontier arms, Spike C Arm C, Spike D Arm C1) used "frontier-bare single-shot" as the comparison baseline, which is favorable to the architecture's `+ orchestration` primitive by design: of course a deterministic script-agent running cross-file verification outperforms a model with no file access. The architecture's value on this comparison is partially structural to the comparison design.

This is the "framing adoption at synthesis moments" signal at mild-to-moderate intensity. The evidence is genuine; the comparison baseline deserves naming as a scope condition on the central-question claim.

### Signal 2: The "deterministic-vs-probabilistic complementarity" frame in Spike C synthesis

The research log and 004d both advance a unifying frame: "the architecture composes components with different error distributions; the composition's coverage exceeds any single component's." This frame appeared fully formed in Spike C's synthesis. Its provenance is worth examining: the lit review (004a) prepared the ground with the heterogeneity-uncorrelated-errors mechanism from Sun et al. and Ding et al.; the methods reviewer (research-design-review-cycle-3-mid.md) introduced the "ensemble dispatch" framing; Spike C's data confirmed the concrete-verification gap. The synthesis frame is well-supported.

However, the frame does two things simultaneously: (a) names a real empirical pattern in the data, and (b) provides a general architectural principle ("the architecture's value is in composing components with different error distributions") that exceeds any single fixture's scope. The cycle's evidence supports (a) clearly. It supports (b) only at the scope of "this fixture class, this bug category, this specific error asymmetry." 004d's Discussion section gives (a) and (b) roughly equal prominence, and 004d's Cycle 4 hooks implicitly treat (b) as a confirmed principle to build on ("test the architecture's actual `invoke_ensemble` primitive end-to-end" — presupposing the principle is confirmed, not continuing to test it).

Whether this is earned confidence or framing adoption depends on the evidence count: Spike A (one fixture, documentation review) and Spike C (one fixture, code review with synthesized issues) are two confirmations of (a) at two task classes. That is meaningful but thin for an architecture-level principle. The scope qualifier "on these fixtures" is present in the formal Limitations sections but does not propagate into the headline synthesis claims with the same force.

### Signal 3: Confidence markers and assertion density in 004d and 004e

The most concrete signal in the cycle. 004d contains the phrase "this is the cycle's central-question evidence" to describe N=3 concrete-verification results on one synthesized fixture. 004e's abstract opens "B1 ties C1 at zero $ cost" for a N=1 pilot trial. The research log's summary calls Spike D's architecture "empirically demonstrated" after a single pilot trial that the spike's own header explicitly labels "Cycle 4 priming experiment, not Cycle 3's primary central-question evidence."

To be fair: the scope qualifiers are present throughout. 004d's Limitations section names "single fixture" as item 1. 004e's abstract names "Cycle 4 priming experiment, not Cycle 3's primary central-question evidence" in the third sentence. The research log correctly lists Spike D's findings as a fourth distinct finding but in the same list format as Spike C's primary finding, giving them roughly equal weight to a reader scanning the summary.

The pattern: the scope-bounded language is in the designated Limitations sections; the headline language in Abstracts, Discussion openings, and Summary bullets uses assertion-dense framing. A downstream agent inheriting this corpus through the archive and summary (rather than reading each Limitations section) would see the architecture's value described as "empirically demonstrated" and "load-bearing" without the single-fixture qualifier. This is the self-correction blind spot pattern — the in-conversation agent cannot fully assess how the headline language reads when the Limitations sections are not foregrounded.

### Signal 4: The opencode CLI stall — alternative reading not preserved

Spike D's primary finding is framed unambiguously as (a): a substantive cycle finding about deployment-shape fragility. The alternative reading — (b) infrastructure noise that masked the intended multi-ensemble pilot — is not preserved alongside (a) in the synthesis prose.

The evidence for (a): the stall is repeatable (3 sustained attempts), isolatable to prompt size, specific to the opencode path (same model via production model factory completes in 24.8s), and documented with specific timing and CPU data. The evidence is genuine and the (a) framing is defensible.

The evidence for (b): the pilot was N=1 per arm on a single fixture; the intended B1 arm (cheap orchestrator with routing intelligence) never ran as designed — instead a direct model factory script substituted. The "architecture works" claim in 004e rests on a manually staged pipeline that bypassed the orchestrator's actual `invoke_ensemble` primitive. The opencode stall may be the reason the cycle's most important test (autonomous routing in `llm-orc serve`) did not run, not a finding in its own right.

The synthesis presents (a) and (b) as compatible ("this is a deployment-shape finding that argues Cycle 4 should use `llm-orc serve` paths") rather than acknowledging that (b) is also a reading of the same facts. Under the (b) reading, Spike D's "B1 ties C1 at zero $ cost" result is less informative — it describes a manually staged pipeline, not the architecture's autonomous coordination primitive. This distinction matters for Cycle 4's research design.

### Signal 5: Solution-space narrowing trajectory — managed but present

The cycle opened with ADR-003 bracketed and the solution space maximally open. By Spike D, the solution space has converged to: "the architecture's value is in cheap-orchestration + ensemble-dispatch on tasks where cross-file verification matters; test this more deeply in Cycle 4." This is a well-motivated convergence given the evidence. But two competing hypotheses were not carried forward:

(1) "Architecture's value is in the script-agent's deterministic guarantee regardless of cross-file" — this alternative is consistent with all of Spike C's data. Arm B's advantage over Arm C was a guaranteed cross-file file read, not the full `invoke_ensemble` orchestration pattern. A script that just ran and reported would have produced the same ISSUE-5 concrete-verification result. The "orchestration" part of `+ orchestration` (the orchestrator choosing to dispatch the ensemble, integrating outputs) was not what made the difference on ISSUE-5; the script's deterministic file access was. The synthesis frames this as "orchestration's value" but the mechanism is closer to "script's value."

(2) "Cross-tier complementarity is fixture-specific, not architectural" — Spike B found no cross-tier complementarity on multi-turn. Spike C found it on cross-file verification. The synthesis treats these as "task-class-dependent" without examining whether the operative variable is "task class" or "fixture design" (synthesized fixture with known cross-file issues vs. real task).

### Practitioner's reflection-time response — what it indicates

The practitioner's gate response ("I'd want the next cycle to pick be grounded in supported design methods for orchestrator + ensembles") is forward-looking and methodology-grounding in register. Read in the context of the cycle's findings: it points toward a gap the cycle's confidence markers do not acknowledge. If Cycle 3's architecture-level conclusions were fully earned, "more methodology grounding" would not be the practitioner's first forward direction. The response suggests the practitioner experienced the cycle's convergence as ahead of its grounding, even if they did not articulate the susceptibility signal explicitly.

This is directionally consistent with the embedded-conclusion risks identified above. The practitioner's response reads as mild course-correction, not as forward-looking refinement consistent with fully earned confidence.

---

## Recommendation

**Grounding Reframe recommended, scoped to Cycle 4 feed-forward only — no in-cycle action warranted.**

The cycle's five research-log artifacts are complete; the primary findings (Spike A's cross-tier finding, Spike C's concrete-verification finding, Spike B's F1/F2 methodological finding) are genuine and well-scoped in their Limitations sections. No grounding action should alter these artifacts before archive. The recommendation is feed-forward, not in-cycle correction.

### What is uncertain

Two findings carry embedded-conclusion language that exceeds their evidence scope and will enter Cycle 4 as inherited commitments:

**Finding 1: "The architecture's `+ orchestration` primitive is load-bearing on this bug class."**
The evidence (Spike C, N=3, single synthesized fixture) shows that a specific ensemble's script-agent provides deterministic cross-file file access that neither cheap-bare nor frontier-bare provides in single-shot. The mechanism is the script's file read, not the orchestrator's ensemble-routing decision. The `invoke_ensemble` primitive was exercised, but what won was the script-agent's deterministic capability — which would have won even without ensemble orchestration if the script had simply run and its output injected as context (the Spike A arm2 test, applied to cross-file verification). The "orchestration is load-bearing" claim and the "script-agent's deterministic capability is load-bearing" claim are distinct, and the cycle's evidence more directly supports the latter.

**Finding 2: "B1 ties C1 at zero $ cost / the architecture works at the multi-stage workflow level."**
The evidence (Spike D, N=1, single fixture, manually staged pipeline) shows that when explicit staging is used — not the orchestrator's autonomous `invoke_ensemble` routing, but a manually scripted three-step pipeline via production model factory — the outputs are quality-equivalent to frontier single-shot. This is a finding about manual pipeline quality, not about the architecture's autonomous coordination primitive. The Cycle 4 question (does the orchestrator autonomously route correctly?) is unsupported by this finding, but the headline framing ("architecture works at multi-stage workflow level") may cause Cycle 4 to enter with a confidence level in autonomous routing that the evidence does not support.

### Concrete Grounding Actions for Cycle 4 Entry

**Grounding action 1 (on Finding 1):** Before Cycle 4 designs around "cheap-orchestration + ensemble-dispatch as the architecture's value primitive," explicitly test whether the mechanism is "script's deterministic file access" versus "orchestrator's routing decision." A direct comparison: (a) cheap-bare with the script's output injected as input context (Spike A arm2 applied to cross-file verification), versus (b) cheap+ensemble dispatched via `invoke_ensemble`. If (a) matches (b) on concrete-verification, the lesson is "deterministic tool output is the mechanism" not "ensemble orchestration is the mechanism." This distinction matters for how Cycle 4 designs its multi-ensemble coordination experiments.

**Grounding action 2 (on Finding 2):** Cycle 4's research entry should explicitly distinguish the Spike D pilot's manually staged pipeline from the architecture's intended autonomous routing. The entry should name: "The orchestrator dispatching `invoke_ensemble` autonomously has Cycle 1's CAP-9 baseline and Spike C's single-stage evidence as its empirical ground. It has no evidence from multi-stage autonomous coordination. Spike D tested manual staging via direct model factory, not autonomous routing via `llm-orc serve`." Without this explicit entry-point grounding, Cycle 4 risks inheriting the confidence markers from Spike D's abstract and building against a foundation the evidence does not yet support.

**Grounding action 3 (on the comparison baseline):** Cycle 4 should name the frontier comparison baseline as a scope condition rather than a neutral reference point. "Frontier-bare single-shot has no access to other files" is structurally true but also structurally favorable to the architecture. A grounded comparison would ask: what happens when frontier has the same file access the script-agent has? (E.g., frontier + the same cross-file extraction script as input context.) If the architecture's advantage disappears under this comparison, the mechanism is "information access" not "architectural composition." If the advantage persists, the mechanism is genuinely compositional.

### What the practitioner would be building on without grounding

Without these three grounding actions, Cycle 4 enters with:
- An inherited commitment to "the `+ orchestration` primitive is load-bearing" that is more plausibly a "script-agent's deterministic capability is load-bearing" finding
- An inherited confidence in multi-stage autonomous coordination based on a manually staged N=1 pilot
- A comparison baseline (frontier-bare single-shot with no file access) that is favorable to the architecture by design

The risk is a Cycle 4 research design that tests increasingly complex multi-ensemble coordination scenarios while the more basic question — is it the orchestration or the script that wins? — remains unresolved. Cycle 3's finding is genuinely useful; it just stops one step short of the mechanism isolation that would make it architecture-level confidence rather than task-level observation.

These are all Cycle 4 applicable; none requires in-cycle action before archive. The three grounding actions above, and this snapshot, serve as the inheritance signal for Cycle 4's research-entry protocol.

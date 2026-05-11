# Gate Reflection: Agent Design (Script + Local Models + Cloud Orchestrator) — research → discover

**Date:** 2026-05-01
**Phase boundary:** research → discover (deferred to Cycle 4; Cycle 3 closes at research-phase end as Mode B)
**Cycle:** Agent Design — Script + Local Models + Cloud Orchestrator (Cycle 3)

## Belief-mapping question composed for this gate

The cycle's most consequential late-cycle conclusion — that "the architecture's `+ orchestration` primitive is load-bearing on this bug class" (Spike C synthesis) and that "B1 ties C1 at zero $ cost" (Spike D pilot) — rests on N=3 results from a single synthesized fixture (Spike C) and an N=1 manually staged pipeline (Spike D). The susceptibility snapshot identifies two embedded-conclusion risks: (a) the mechanism more directly supported by the evidence is the script-agent's deterministic file access, not the orchestrator's ensemble-routing decision; (b) Spike D tested manual staging via direct model factory, not autonomous routing via `llm-orc serve`.

> *What would you need to believe for the alternative reading — "the script's deterministic file access is load-bearing, not orchestration; the architecture's autonomous routing primitive is unsupported by Cycle 3's evidence" — to be the right reading? And if that reading is right, what becomes more important to test in Cycle 4?*

The question was composed against the cycle's specific drift: the late-cycle synthesis crystallized "orchestration is load-bearing" and "architecture works at multi-stage workflow" as headline framings, while the underlying evidence more directly supports "deterministic tool output is the mechanism on cross-file verification" and "manual staging via production model factory produces frontier-equivalent quality." The framing/mechanism gap is the entry-point hazard for Cycle 4.

## User's response (verbatim)

> *I'd want the next cycle to pick be grounded in supported design methods for orchestrator + ensembles in such a way that we can envision the right next steps and iterate another step closer to effective agentic design with llm-orc.*

## Pedagogical move selected

Probe — referencing the artifact content directly (the Spike C and Spike D headline framings against the snapshot's mechanism-isolation distinction). The response is forward-looking and methodology-grounding in register; it points toward a gap the cycle's confidence markers do not acknowledge directly. Read alongside the snapshot, the practitioner's request for "supported design methods" is directionally consistent with the embedded-conclusion risks the snapshot surfaced: the practitioner experienced the cycle's convergence as ahead of its grounding.

## Commitment gating outputs

**Settled premises (the practitioner is building on these going into Cycle 3 → Cycle 4):**

- Spike A's cross-tier heterogeneity-uncorrelated-errors finding on documentation review (cheap-with-direction caught semantic bugs frontier missed; 3/3 vs 0/4 on undefined-profile bugs in directed prompt; 1/3 in debiased prompt) is a genuine empirical finding, scope-bounded to that fixture class.
- Spike B's F1 vs F2 methodological finding is settled: F2 imagined-state bias (5/6 traces imagined fines that didn't exist) makes F1 turn-by-turn dispatch the correct method for multi-turn frontier-tier testing under the no-paid-API constraint. Cross-tier complementarity does not replicate at multi-turn on the cycle's library-checkout fixture — task-class-dependent, not architectural.
- Spike C's concrete-verification result on the synthesized cross-file fixture (Arm B 3/3, Arm A 1/3, Arm C 0/2 on ISSUE-5) is genuine. The mechanism is the script-agent's deterministic file access; whether the `+ orchestration` framing or the `+ deterministic-tool-output` framing is more accurate is open and explicitly Cycle 4 territory.
- Spike D's pilot showed B1 = C1 = 4/5 resolved on manually staged pipeline; A1 3/5 (verifier false-positive on api_key hash, effectively 4/5 if security-property-judged). The opencode CLI stall on substantial code prompts at cheap-tier is a deployment-shape finding worth carrying forward; the production model factory path completed Stage 2 in 24.8s.
- Free-options preference (MiniMax M2.5 Free via OpenCode Zen; ask before cost-incurring actions) and spike-artifact retention (preserve scratch/* and ensembles until corpus close) are durable practitioner directives recorded in agent memory.

**Open questions (the practitioner is holding these open going into Cycle 4):**

- Mechanism isolation: is "the script's deterministic file access" load-bearing, or is "the orchestrator's ensemble-routing decision" load-bearing? The cycle's evidence supports the former more directly. Distinguishing these matters for Cycle 4's multi-ensemble coordination experiments.
- Autonomous routing vs manual staging: Spike D's "architecture works at multi-stage workflow" finding rests on manual staging via direct model factory. The orchestrator dispatching `invoke_ensemble` autonomously across multiple stages has Cycle 1's CAP-9 baseline and Spike C's single-stage evidence as its empirical ground — no evidence from multi-stage autonomous coordination.
- Frontier comparison baseline: "frontier-bare single-shot has no access to other files" is structurally favorable to the architecture by design. What happens when frontier has the same file access the script-agent has? If the architecture's advantage disappears under matched information access, the mechanism is "information access" not "architectural composition."
- Whether the four-priorities frame (performance × environmental cost × local-first × token cost) survives the susceptibility snapshot's framing-adoption signal. The cycle ran without forcing the frame to either pass or fail a measured-divergence test.

**Specific commitments carried forward to Cycle 4:**

- Cycle 4 territory (named by the practitioner at gate close): **supported design methods for orchestrator + ensembles** that can envision the right next steps and iterate closer to effective agentic design with llm-orc.
- The susceptibility snapshot's three grounding actions are inherited as Cycle 4 research-entry inheritance, not as Cycle 3 gaps to fix in-cycle:
  - **Grounding action 1 (mechanism isolation):** Test whether the mechanism is "script's deterministic file access" vs "orchestrator's routing decision." Compare cheap-bare with the script's output as input context (Spike A arm2 pattern applied to cross-file verification) vs cheap+ensemble dispatched via `invoke_ensemble`.
  - **Grounding action 2 (autonomous routing):** Cycle 4 entry should explicitly distinguish Spike D's manually staged pipeline from the architecture's intended autonomous routing. Name the evidence base for autonomous routing as Cycle 1's CAP-9 baseline + Spike C's single-stage evidence; name the evidence gap as multi-stage autonomous coordination via `llm-orc serve`.
  - **Grounding action 3 (comparison baseline):** Name the frontier comparison baseline as a scope condition rather than a neutral reference. Test what happens when frontier has the same file access (e.g., frontier + the same cross-file extraction script as input context).
- Cycle 3 closes at research-phase end as Mode B (Research Only), declared at cycle close. The five research-log artifacts (004a-e) plus the audit trail are the cycle's deliverable; no Cycle 3 essay was produced.
- The research log serves as the cycle's primary artifact in lieu of an essay; it carries the central-question evidence summary, the four primary findings, and the four open questions inherited to Cycle 4.

## Methodological note on the gate's shape

This gate ran in a single phase. The susceptibility snapshot was dispatched at phase boundary per Tier 1 mechanism; its findings (three grounding actions, two embedded-conclusion risks) feed forward to Cycle 4's research-entry protocol rather than triggering in-cycle revision. The snapshot evaluator explicitly recommended "Grounding Reframe scoped to Cycle 4 feed-forward only — no in-cycle action warranted" because the cycle's research-log artifacts contain scope-bounded language in their Limitations sections; the embedded-conclusion risks are concentrated in headline framings (Abstracts, Discussion openings, Summary bullets) where downstream readers might inherit confidence markers without the single-fixture qualifier.

The cycle's mid-cycle methods-reviewer dispatch was practitioner-sparked but agent-amplified: the practitioner offered a brief observation ("MiniMax is going to perform similarly to Sonnet on simple tasks. That's more or less what you determined, yes?"), the agent reconstructed the methodological failure in detail and proposed the reviewer dispatch, and the reviewer's findings reshaped the cycle's mid-trajectory through Spike C. This is a structural improvement over Cycle 2's predominantly practitioner-originated corrections and is recorded as a methodology-trajectory signal worth preserving for Cycle 4.

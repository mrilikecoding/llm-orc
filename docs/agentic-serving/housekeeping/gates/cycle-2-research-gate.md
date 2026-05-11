# Gate Reflection: Multi-Turn Agentic Serving and Live Composition — research → discover

**Date:** 2026-05-01
**Phase boundary:** research → discover (deferred to Cycle 3; Cycle 2 closes at research-phase end)
**Cycle:** Multi-Turn Agentic Serving and Live Composition (Cycle 2)

## Belief-mapping question composed for this gate

The essay's most consequential conclusion — ADR-011 is defensible as a default but not as a ceiling — rests on Spike A3's demonstration that *some* well-architected configurations produce capabilities prompt-steering structurally cannot replicate. The framing audit raised a competing reading where the script agent (a non-LLM component) is doing the load-bearing work and the ensemble topology is incidental.

> *What would you need to believe for that alternative reading to be right? And is that belief supported by what the cycle has actually measured?*

The question was composed against the practitioner's specific cycle-throughout posture: they had held a clear distinction between "the existing ensemble's design failed" and "no ensemble design succeeds." Their endorsement of A3's design rested on the heterogeneity-finding mechanism plus the MARG-concatenation pattern plus the script-agent affordance. The cycle's empirical evidence is consistent with all three doing meaningful work, OR with the script alone doing meaningful work and the heterogeneity/MARG portions being noise.

## User's response (verbatim, combined across two gate exchanges)

First response:
> *I would be excited to determine that a collection of scripts and agents makes smaller local models tennable in tandem with perhaps a fast cloud orchestrator agent. The space of design I imagine is large, so it's hard for me to write off the idea altogether. But obviously script agents running locally are going to be more performant and depending on the script, likely deterministic which provides a good grounding for llms. I think the 2 groundings we should take to inform the next cycle which I think makes sense to be around this agent design idea.*

Second response (after the agent surfaced the gate's commitment-gating capture):
> *I think we could ammend the essay. Like I said earlier, I don't think it's worth advancing further in this cycle until we hone a better agenting serving agent/ensemble design strategy. So likely this is where we'd close this cycle to pick up a new research cycle on the agent design.*

## Pedagogical move selected

Challenge — belief-mapping question composed against the cycle's specific framing-audit-surfaced alternative reading.

## Commitment gating outputs

**Settled premises (the practitioner is building on these going into Cycle 3):**

- Spike A3 demonstrated that *some* configurations produce capabilities prompt-steering structurally cannot replicate (factual grounding via deterministic script checks; uncorrelated errors via heterogeneous-reviewer concatenation).
- Script agents provide deterministic grounding for LLMs — load-bearing for the cycle's empirical findings, not incidental. The script + small-local-models + fast-cloud-orchestrator combination is the empirically promising territory.
- The four design priorities frame held usefully across three configurations measured. Recorded transparently as one valid choice the cycle adopted from practitioner pushback at Loop 1 synthesis, not as the only defensible reading.
- ADR-011 is defensible as a default and as the right choice for tasks fitting Anthropic's threshold conditions, but not as a ceiling for task classes where factual grounding via deterministic checks is part of the success criterion.
- Essay 002's CAP-2 finding generalizes directionally beyond CAP-2's narrow scope: prompt-steering of a capable single orchestrator beat the existing production code-review ensemble at qwen3:8b tier and at cloud-orchestrator tier on two task classes (capability query, code review).

**Open questions (the practitioner is holding these open going into Cycle 3):**

- Whether the value attributed to A3 comes from the script-agent slot alone or from the script + ensemble-topology combination. The untested "A2 + script input" alternative — prompt-steered single cloud orchestrator receiving the script-agent's deterministic report as additional input context — is logically available from A3's data and not refuted. Practitioner explicitly accepted this as honest scope condition rather than as cycle failure.
- Multi-turn tool-dispatching reliability. Tau-bench's GPT-4o-under-50%-task-success and pass^8-under-25% are the relevant published baselines; the cycle's empirical work does not extend to that regime. ADR-011's empirical strengthening is scoped to single-ask task classes.
- The broader agent design space — particularly the recursive composition (ensembles-of-ensembles) territory the practitioner named at cycle entry but the cycle did not test.

**Specific commitments carried forward to Cycle 3:**

- Cycle 3 scope: agent design specifically — building on what Cycle 2 has surfaced about scripts + small local models + fast cloud orchestrator combinations.
- The two consequential framing-audit items (untested "A2 + script input" alternative; tau-bench multi-turn reliability) carry forward as Cycle 3 research-entry inheritance, not as gaps in Cycle 2's essay.
- The susceptibility snapshot's two feed-forward items at `housekeeping/audits/susceptibility-snapshot-cycle-2-research.md`: (a) the four-priorities frame enters Cycle 3 as hypothesis to be tested, not as settled lens; (b) the A2 + script input alternative must be tested before Cycle 3 builds on ADR-011's boundary refinement.
- Cycle 2 closes at research-phase end; Cycle 3 will be a new research-only cycle in the same agentic-serving scoped corpus rather than this cycle's continuation through DISCOVER → MODEL → DECIDE → ARCHITECT → BUILD.

## Methodological note on the gate's pre-amendment / post-amendment shape

The gate ran in two phases. The first surfaced the framing-audit deferred items and the practitioner's response committed to closing the cycle at research with a new cycle to follow. The second phase amended the essay to address all six framing-audit items inline (rather than carrying them forward as gaps), per practitioner direction. Round 9 verification audit cleared the amended essay before this gate reflection note was written. The gate's epistemic outputs (the commitment gating above) reflect the final post-amendment artifact state.

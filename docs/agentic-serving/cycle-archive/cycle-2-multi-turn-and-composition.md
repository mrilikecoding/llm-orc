# RDD Cycle Archive — Agentic Serving / Cycle 2: Multi-Turn Agentic Serving and Live Composition

**Artifact base:** `docs/agentic-serving/`
**Plugin version at cycle start:** v0.8.5
**Migration version at cycle start:** 0.8.5 (`housekeeping/.migration-version`)
**Cycle started:** 2026-04-29
**Cycle closed:** 2026-05-01
**Cycle close shape:** Mode B (Research Only) declared at cycle close — research-phase findings inherited into Cycle 3 rather than progressing through DISCOVER → MODEL → DECIDE → ARCHITECT → BUILD on this cycle's target.

## Phase Status

| Phase | Status | Artifact | Key Epistemic Response |
|-------|--------|----------|----------------------|
| RESEARCH | ✅ Complete | `essays/003-multi-turn-orchestration-and-the-four-axis-frame.md` | Practitioner committed to closing the cycle at research-phase end and opening Cycle 3 on agent design specifically. Settled premises: A3 demonstrated *some* configurations produce capabilities prompt-steering structurally cannot replicate; ADR-011 defensible as default but not as ceiling; script + small-local-models + cloud-orchestrator combination is the empirically promising territory. Open questions inherited to Cycle 3: the untested "A2 + script input" alternative; multi-turn tool-dispatching reliability (tau-bench regime); the broader agent design space including ensembles-of-ensembles. |
| DISCOVER | ☐ Deferred to Cycle 3 | — | — |
| MODEL | ☐ Deferred | — | — |
| DECIDE | ☐ Deferred | — | — |
| ARCHITECT | ☐ Deferred | — | — |
| BUILD | ☐ Deferred | — | — |
| PLAY | ☐ Optional | — | — |
| SYNTHESIZE | ☐ Optional | — | — |

## Feed-Forward Signals

### From RESEARCH (Loop 1 — combined literature review)

1. **Multi-turn agentic dynamics literature is substantive (HORIZON, AMA-Bench, LongCLI-Bench, Khanal et al.) and recent.** Long-horizon performance degrades super-linearly. Frontier models show meltdown rates up to 19% on long-horizon tasks. Memory compression failure (similarity-based retrieval loses causally necessary information at <60% accuracy on real agentic trajectories). LongCLI-Bench: SOTA agents pass <20% of long-horizon tasks; most stall at <30% completion.

2. **Composition's threshold conditions are concrete, per Anthropic's June 2025 production guidance.** Composition earns its cost when (1) value justifies 15× token overhead, (2) genuine parallelization opportunities exist, (3) information scope exceeds single context window. For tasks not meeting these, single-agent + prompt steering is the recommended default.

3. **Practitioner's bias-amplification prior verified by literature** (Li et al. ICLR 2026; Madigan et al.; Wynn et al. ICML MAS Workshop 2025). The "Trigger Vulnerability" finding — injecting objective context accelerates polarization — is the most striking finding for any cycle considering multi-orchestrator coordination protocols. Five mitigations documented (anonymization is strongest evidence base, ACL 2026 Main); none fully eliminate the echo-chamber failure mode. Mitigation evidence is debate-shape-specific; whether it translates to llm-orc's supervisor-routing + cascading-tool-dispatch coordination shape is empirically open.

4. **Bio-inspired (eusocial / ant-colony / naked-mole-rat) literature in those terms is sparse in LLM-architecture literature.** Closest published analogues: stigmergy (SwarmSys, arXiv:2510.10047) and ACO (classical algorithm, slow when LLM-mediated; Rahman & Schranz arXiv:2506.14496 measured 36,000× latency penalty for LLM-mediated swarm coordination versus classical algorithms).

5. **Capability-tier gap is the cycle's empirical opportunity.** All strong empirical evidence for multi-agent benefits uses frontier models. No published paper validates multi-agent composition benefits at qwen3:8b tier without fine-tuning. The closest analogue is OPTIMA on Llama 3 8B (ACL 2025 Findings) but requires fine-tuning the cycle's deployment does not have.

### From RESEARCH (Loop 4 — ensemble design principles literature)

6. **Yao et al. (2025, meta-judge / panel discussion) — sequential panel discussion *actively destroys diversity*.** Panel-discussion precision 72.6% vs majority voting 77.3%. The same mechanism as the production code-review ensemble's two-stage summarization design — documented in published literature as anti-pattern, not just empirically here.

7. **MARG (Drozdov et al., 2024) is the closest specificity-preserving alternative the literature offers.** Three independent agent groups whose outputs are *concatenated* rather than synthesized; generic-comment rate dropped from 60% to 29% under MARG's no-synthesizer architecture.

8. **Heterogeneity is the most empirically grounded role-decomposition principle.** Sun et al. (2025), Ding et al. (2024): heterogeneous agents (different models / prompts / tools) outperform homogeneous because their errors are uncorrelated. Up to 95% of theoretical performance ceiling recoverable via diversity-based selection.

9. **Reflexion caveat (NeurIPS 2023 + 2024 follow-ups): intrinsic self-correction is not reliably effective without external evaluation signals.** Multi-agent Reflexion (MAR) achieves the gain at ~3× API cost.

10. **Jiang et al. (April 2026 preprint) provides theoretical grounding for Spike A's empirical finding.** Under matched reasoning-token budgets, single agents consistently match or exceed multi-agent on multi-hop reasoning. Multi-agent becomes competitive only when single-agent context utilization is degraded or additional compute is provided. Treat as preliminary pending replication.

11. **Design-principles literature is richer than the bio-inspired corpus from Loop 1 but is not a principled engineering corpus.** Vocabulary and named patterns: yes. Empirical comparisons: limited and domain-specific (scientific review, judging, math — not code review). Prescriptive synthesizer-selection guidance from task properties: absent beyond coarse architectural mappings.

### From RESEARCH (Spike B — small-model substitution into existing ensemble)

12. **Cascade plumbing dominates wall-clock regardless of reviewer-tier compute on consumer hardware.** Substituting llama3 (8B) reviewers for qwen3:0.6b reviewers — ~9× reduction in disk size or ~13× in parameter count — left total wall-clock essentially unchanged (~62s CAP-9 → ~58.7s Spike B). Output specificity dropped to category-level abstractions versus CAP-3b's concrete recommendations. **The "shrink the reviewers" hypothesis is refuted** as a latency optimization on this cascade architecture; smaller reviewers buy environmental + local-first wins at the cost of output specificity, with no latency improvement.

### From RESEARCH (Spike A — cascade vs prompt-steering on existing ensemble)

13. **The existing production code-review ensemble's two-stage summarization design is dominated on every measured axis except dimension coverage by a strong prompt-steered single cloud orchestrator with no cascade.** A2 (~19.5s, 16 specific recommendations) outperformed A1 (71-145s, 9 recommendations collapsed to table form). The cascade's mechanism: tech-lead-synthesizer compresses 4 reviewer outputs, then orchestrator's post-`invoke_ensemble` synthesis compresses again. Each stage is lossy on per-reviewer specificity.

14. **Essay 002's CAP-2 finding generalizes directionally beyond CAP-2's narrow scope on the configurations tested.** Prompt-steering > structural composition holds at qwen3:8b tier on single-ask capability queries (CAP-2) and at cloud-orchestrator tier on code-review tasks (Spike A).

### From RESEARCH (Spike A3 — novel ensemble informed by Loop 4 findings)

15. **A3 produced a moderate pass with caveats.** Median 81s wall-clock (range 62-128s), 4× A2's median but bounded under A1's 145s ceiling. Output structure (three sections cleanly attributed) preserved across all three trials by harness-enforced concatenation.

16. **The script-agent slot earned its place.** Deterministic checks produced verified facts the LLM reviewers could not generate on their own (verified link counts, confirmed standard-section presence, code-block parseability). The reviewers actively used these as anchors. **A3 surfaced documentation bugs A2 missed across three prior trials** — undefined `default-local` and `ollama-gemma-small` model profiles referenced in the README's YAML examples but never defined in the Configuration section, which would cause runtime errors for users copying the example.

17. **Heterogeneity-uncorrelated-errors mechanism is empirically observable.** Reviewer 1 (Tencent Hunyuan) and Reviewer 2 (Moonshot Kimi) produced 5-8 distinct findings each with only 1-2 overlap. The literature's mechanism reproduced in cycle-specific data.

18. **MARG-style concatenation preserves what cascade-with-collapse-summarization destroys.** A3's no-collapse design surfaces the per-reviewer voices; Spike A's collapse step was the failure mode.

19. **Model selection inside heterogeneity slots matters more than spike anticipated.** Reviewer 1's chain-of-thought leakage (~18,000 characters of reasoning narration before reaching numbered recommendations) is a quality cost the topology cannot fix. Heterogeneity finding ("different families = uncorrelated errors") holds on findings-coverage but a model that produces unstructured prose under a 4096-token cap is a poor reviewer-slot inhabitant regardless of family-distinctness.

20. **The untested "A2 + script input" alternative is the cycle's open scientific question.** A3 mixed three design changes simultaneously (script slot + heterogeneous LLMs + MARG concatenation) and the spike does not isolate which is load-bearing. A simpler configuration not tested in this cycle — prompt-steered single cloud orchestrator with the script-agent's deterministic report as additional input context — could deliver equivalent factual grounding at A2's latency without the ensemble overhead. If it does, the architectural lesson is "augment prompt-steering with deterministic tool outputs" rather than "use ensemble topology" — a substantively different policy direction for ADR-011.

### From RESEARCH (gate exchange and susceptibility snapshot)

21. **ADR-011 is defensible as a default but not as a ceiling.** For task classes fitting Anthropic's threshold conditions (and for the cycle's currently-validated single-ask task classes), ADR-011 holds as the right default — Spike A's empirical evidence strengthens it. For task classes where factual grounding via deterministic checks is part of the success criterion, the ADR's implicit ceiling does not hold. The next ADR review at synthesis (in Cycle 3 or beyond) should record this boundary refinement explicitly.

22. **Tau-bench (Yao et al., 2024) is the relevant published baseline for the multi-turn tool-dispatching regime the cycle's spike battery does not reach.** GPT-4o under 50% task success on tau-bench's tool-agent-user multi-turn interaction tasks; pass^8 below 25% even for frontier models. ADR-011's empirical strengthening is scoped to single-ask task classes; Cycle 3 should treat the multi-turn regime as empirically open.

23. **Susceptibility snapshot identified two patterns in agent behavior worth carrying forward** (full report at `housekeeping/audits/susceptibility-snapshot-cycle-2-research.md`):
    - **Performance-axis bias in initial syntheses.** When asked to summarize literature findings, the agent's first pass leaned toward performance-axis framings even when the practitioner's optimization scope explicitly included other axes. Practitioner pushback corrected the framing; agent did not self-correct. Cycle 3 might benefit from naming the operating frame up-front in synthesis dispatches.
    - **Self-correction blind-spot on overgeneralization.** When spike findings could be read narrowly or broadly, agent's first interpretation leaned broad. Practitioner correction required. Cycle 3 might benefit from explicitly committing to the narrow reading first in any spike-finding synthesis.

### From RESEARCH (commitment-gating outputs at gate)

24. **Settled premises (Cycle 3 builds on these):**
    - A3 demonstrated *some* configurations produce capabilities prompt-steering structurally cannot replicate.
    - Script agents provide deterministic grounding for LLMs — load-bearing, not incidental.
    - Four design priorities frame held usefully across the cycle's three measured configurations. Recorded transparently as one valid choice the cycle adopted from practitioner pushback at Loop 1 synthesis, not as the only defensible reading.
    - ADR-011 defensible as default but not as ceiling.
    - Essay 002's CAP-2 finding generalizes directionally beyond CAP-2's scope at the configurations tested.

25. **Open questions (Cycle 3 inherits these):**
    - The untested "A2 + script input" alternative — must be tested before Cycle 3 builds on ADR-011's boundary refinement, per the susceptibility snapshot's specific in-cycle grounding action.
    - Multi-turn tool-dispatching reliability — tau-bench regime not reached by Cycle 2's spikes.
    - The broader agent design space, particularly recursive composition (ensembles-of-ensembles) the practitioner named at cycle entry but the cycle did not test.
    - Whether the four design priorities frame is empirically operative for Cycle 3's agent-design questions or functions as a framing preference shaping which territory gets investigated.

26. **Specific commitments carried forward to Cycle 3:**
    - Cycle 3 scope: agent design specifically — building on what Cycle 2 surfaced about scripts + small local models + fast cloud orchestrator combinations.
    - The two consequential framing-audit items (A2 + script alternative, tau-bench multi-turn) carry forward as Cycle 3 research-entry inheritance.
    - Four-priorities frame enters Cycle 3 as hypothesis to be tested, not as settled lens.

## Context for Resumption

### Cycle close shape

Cycle 2 closed at research-phase end with the practitioner's explicit direction to open a new research-only Cycle 3 on agent design rather than progress through the full pipeline on this cycle's findings. The cycle's value is the essay + research log + audit trail (8 rounds of argument audit + framing audit; citation audit; methods reviewer; susceptibility snapshot) as inheritance for Cycle 3.

The decision to close at research-phase end is methodologically distinctive — Mode B (Research Only) declared at cycle close rather than at cycle entry. The cycle's findings reshape the question more than they answer it: the cycle's premise (well-architected processes can achieve good results) is partially supported empirically through Spike A3, but the alternative reading (script alone is load-bearing; ensemble topology is incidental) is not refuted by the cycle's evidence. Closing at research preserves Cycle 3's empirical territory rather than committing to a downstream pipeline run on a target the cycle's empirical work has shown is not yet sharp enough to drive ADRs and system design.

### Closing pointer to Cycle 3

Cycle 3 territory (named at gate close): **agent design** specifically — testing whether script + small local models + fast cloud orchestrator combinations can co-optimize across the four design priorities (or whatever frame Cycle 3 adopts after testing the four-priorities frame as hypothesis). Spike battery for Cycle 3 should commit to testing the "A2 + script input" alternative before synthesizing ensemble-topology findings, per the susceptibility snapshot's grounding action.

### Suggested fresh-session handoff prompt for Cycle 3

> Open Cycle 3 of the agentic-serving scoped corpus on agent design. Cycle 2 closed 2026-05-01 at research-phase end (Mode B). Read `docs/agentic-serving/cycle-archive/cycle-2-multi-turn-and-composition.md` §Feed-Forward Signals for the inherited signal index (items #1-#26), `docs/agentic-serving/essays/003-multi-turn-orchestration-and-the-four-axis-frame.md` for the prior essay, and `docs/agentic-serving/housekeeping/audits/susceptibility-snapshot-cycle-2-research.md` for the susceptibility snapshot's two grounding-action recommendations. Begin with `/rdd-research`. The two consequential framing items inherited from Cycle 2 (the untested "A2 + script input" alternative; tau-bench multi-turn reliability) should be addressed in Cycle 3's research design.

## Conformance Notes (carried forward from Cycle 2 cycle-status)

**Corpus migrated from RDD v0.7.3 to v0.8.5 on 2026-04-29.** Migration scope: `cycle-status.md` restructured to ADR-078 cycle stack schema with Cycle 1 archived byte-identical; `system-design.md` restructured per ADR-083/084 — F-pattern orientation lead added; Pattern B companion file `system-design.agents.md` created; Appendix A per-phase susceptibility-snapshot briefs added; `.migration-version` bumped to `0.8.5`.

**Deferred conformance items still carried forward** (low priority; pick up opportunistically; some are inherited from pre-Cycle-2 state and may be addressed in Cycle 3 if relevant):

- ADR Rejected Alternatives + Provenance Check sections — 11 ADRs lack discrete headers (alternatives discussed inline in Context). Format alignment for v0.8.5 ADR template; matters only when ADRs are re-audited.
- Value tensions phrasing — `product-discovery.md` §Value Tensions stated declaratively rather than as open questions per v0.8.5 discover template.
- Essay 001 framing-audit dispatch — `housekeeping/audits/argument-audit-001.md` is argument-only; v0.8.5 dispatches combine argument + framing audits.
- Field-guide path — currently at `docs/agentic-serving/field-guide.md`; canonical is `references/field-guide.md`. ORIENTATION links current location; navigability preserved.
- Scenarios cycle-acceptance-criteria table — top-of-file table or null-coverage one-line note required by v0.8.5 decide Step 4.
- Housekeeping placement (`docs/agentic-serving/housekeeping/` per ADR-070) — ADR-085 supersedes with `.rdd/` placement. Methodology works in either placement during the transition window.

**New deferred item from Cycle 2:**
- Cycle 2's audit corpus and gates (under `housekeeping/audits/` and `housekeeping/gates/`) follow the existing ADR-070 housekeeping placement convention. The ADR-085 `.rdd/` migration target applies but is deferred.

## Cycle 2 artifact index

Essay (final): `essays/003-multi-turn-orchestration-and-the-four-axis-frame.md`
Reflections: `essays/reflections/003-multi-turn-orchestration-and-the-four-axis-frame.md`
Research log (archived): `essays/research-logs/003-multi-turn-orchestration-and-the-four-axis-frame.md`
Loop 1 lit-review: `essays/research-logs/003a-lit-review-multi-turn-and-composition.md`
Loop 4 lit-review: `essays/research-logs/003b-lit-review-ensemble-design-principles.md`
Audit trail: `housekeeping/audits/research-design-review-cycle-2.md` + `citation-audit-cycle-2.md` + `argument-audit-cycle-2*.md` (rounds 1, 2, 3, 4, 5-post-spike, 6, 7-final, 8, 9-final) + `susceptibility-snapshot-cycle-2-research.md`
Gate reflection note: `housekeeping/gates/cycle-2-research-gate.md`

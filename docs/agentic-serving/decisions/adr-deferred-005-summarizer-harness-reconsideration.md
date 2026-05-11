# Deferred ADR — Result Summarizer Harness Reconsideration

**Status:** Deferred (below evidentiary threshold for in-cycle amendment)

**Date:** 2026-05-05

**Cycle:** 4 (Supported design methods for cheap-orchestrator + ensembles)

**Would amend:** ADR-004 (Result Summarization Is Mandatory Before Orchestrator Context)

---

## What is being deferred

Cycle 4's behavioral spike (essay 005 §"The Behavioral Spike") and lit-review (DeliberationBench arXiv:2601.08835) surfaced *candidate amendment territory* for ADR-004's mandatory-summarization commitment. The candidate amendment, per essay 005 §"ADR candidate #5", would be one of:

- **summary-with-pointer mode** — the Result Summarizer Harness returns a summary plus a persistent reference to the full ensemble output, accessible via `query_knowledge` or a new pointer-following primitive. Recovers specificity for the cases where compression discards load-bearing detail.

- **skip-summarization-when-short** — gate the Harness's activation by an output-length / content-density threshold. Outputs below the threshold pass through to the orchestrator unsummarized; outputs above the threshold are summarized as currently. Preserves specificity for short outputs where compression cost exceeds compression benefit.

Either amendment would change ADR-004's current commitment from *"every ensemble invocation triggered by the orchestrator agent MUST produce a summarized result before that result enters the orchestrator's conversation context"* to *"default-with-conditional-skip"* or *"default-with-pointer-recovery"* shape.

---

## Why this is below evidentiary threshold for in-cycle amendment

Wave 3.A Trial 2's specificity-loss observation is **single-trial on a single fixture**. The fixture: an ensemble synthesizer produced approximately 600 characters; the Harness summarized it anyway; the orchestrator quoted *"verbatim from ensemble return"* what was actually the summarizer's compressed blob — losing specificity that the unsummarized 600-character output would have preserved.

The mechanism is documented at the literature level (DeliberationBench's 6× selection-vs-deliberation gap; Yao 2025's panel-discussion diversity-destruction; Cycle 2 Spike A's two-stage cascade-collapse). The cycle-empirical evidence at the *harness-interposition stage* is one trial. Cycle 2 Spike A's documented mechanism is a *different architectural position* (reviewer-to-aggregator stage in a multi-trial documentation review) than the harness-interposition stage observed in Wave 3.A; bridging the two architectural positions is what the deferred spike would do.

The evidentiary asymmetry is the load-bearing reason for deferral. The cycle's discipline at research-gate carry-forward #5 was: *suggested* by single-trial evidence is below the threshold for amending the existing ADR-004 commitment; the architectural cost of structural amendment is real, and the alternative (running the targeted spike first, then deciding) preserves the option without committing to it.

---

## What spike would close the gap

The targeted spike would test the harness-interposition specificity-loss mechanism on diverse output sizes and ensemble configurations. Specifically:

- **Output-size sweep** — run the Harness on ensemble outputs at a range of sizes (e.g., 100, 600, 1500, 5000, 15000 characters). Determine the threshold at which compression cost exceeds compression benefit; map specificity-loss as a function of input size.

- **Ensemble-configuration sweep** — run the Harness on ensembles of different shapes (single-agent, MARG-concatenation, sequential-stages). Determine whether specificity-loss varies systematically by ensemble shape; map which shapes most reproduce Spike A's two-stage cascade-collapse mechanism at the harness-interposition stage.

- **N>1 trials** at each (size × shape) cell to establish the specificity-loss observation as reproducible rather than single-trial.

### Two-failure-mode scope note (argument-audit P2.7 finding, 2026-05-06)

Wave 3.A Trial 2 surfaced **two distinct failure modes**, not one:

- **Failure mode (a):** the summarizer compressed a short (≈600-character) output unnecessarily — the Harness's interposition logic activated when it should not have. This is the *summarization-when-not-needed* failure.
- **Failure mode (b):** the orchestrator misrepresented the summarizer's compressed output as the ensemble's verbatim output. This is the *orchestrator-confabulation-about-source* failure.

The output-size sweep above measures failure mode (a) — at what output size does compression cost exceed compression benefit. The ensemble-configuration sweep also primarily measures (a). Failure mode (b) is the orchestrator's confusion about what it received, which is a downstream effect of (a) but is structurally separate.

If failure mode (b) is the dominant harm in operational deployments, the proposed spike's design measures (a) cleanly but does not directly test (b). A clean spike result on (a) would not resolve whether the amendment to ADR-004 is warranted if (b) is the load-bearing failure. Future-cycle spike designers should:

1. **Measure (a) and (b) separately.** Track in each trial whether the summarizer compressed unnecessarily (a) and whether the orchestrator's downstream reasoning confused the summary for verbatim ensemble output (b).
2. **Report by failure mode.** If (b) is observed independent of (a) (e.g., on outputs where compression was justified, the orchestrator still confused source), that is a different design surface than ADR candidate #5 addresses — possibly closer to ADR-017's structural-validation guard territory than to ADR-004's mandatory-summarization framing.

The spike's interpretation depends on which failure mode dominates; recording both keeps the disposition options open.

The spike's deliverables would be:

1. A function mapping ensemble-output characteristics (size, shape) to specificity-loss probability, bounded by the spike's tested cells.
2. A threshold or boundary above which the *summary-with-pointer* mode is justified; below which *skip-summarization-when-short* is justified; and the cases where current ADR-004 mandatory-summarization remains correct.
3. Empirical evidence sufficient to justify (or reject) the amendment to ADR-004's mandatory framing.

The spike artifacts would follow corpus retention policy (retain until corpus close per the practitioner directive recorded in cycle-status §"Spike artifacts retention").

---

## Disposition options when the spike completes

When the spike runs and produces evidence, the disposition is one of:

- **Spike confirms specificity-loss across diverse fixtures** → file the actual ADR amending ADR-004's mandatory framing. The spike's mapping function (size × shape → specificity-loss) parameterizes either *summary-with-pointer* mode, *skip-summarization-when-short* with the empirically-derived threshold, or both.

- **Spike does not reproduce the specificity-loss across diverse fixtures** → ADR-004 stands as-is. The Wave 3.A Trial 2 observation is recorded as a single-trial artifact whose mechanism did not generalize; the corpus retains the observation in research logs but does not propagate it to architectural amendment.

- **Spike produces ambiguous evidence (some cells confirm, some do not)** → file an ADR amending ADR-004 in scope (e.g., specifically for the ensemble shapes where the mechanism reproduced). The mapping function specifies the scope condition.

Cycle 5 or later is the natural location for the spike unless DECIDE elects to run it before drafting.

---

## Why this deferral is documented in the `decisions/` directory rather than as a research-log entry

Three reasons:

1. **Discoverability.** The next cycle's research entry will read the `decisions/` directory to see what is settled. A deferral note at `adr-deferred-005-summarizer-harness-reconsideration.md` is visible to that read; a research-log entry buried in `essays/research-logs/` is not.

2. **Provenance preservation.** ADR-004's amendment territory has been opened by Cycle 4's evidence and explicitly held below threshold by the practitioner's research-gate decision. The deferral itself is a decision — the decision *not to amend right now*. That decision belongs in the `decisions/` directory by category.

3. **Resumption protocol.** When the spike runs in a future cycle, the next cycle's DECIDE-phase work needs the deferral note as input — what was deferred, why, what evidence would change the disposition. The note serves as the spike's brief.

---

## Cross-references

- ADR-004 (current state of the decision being amended)
- AS-7 (current invariant — Result Summarization is a correctness requirement)
- Domain model OQ #13 (pending evidentiary review of AS-7 framing)
- Essay 005 §"ADR candidate #5" (the amendment specification)
- Cycle 4 cycle-status §"DECIDE-phase carry-forwards" item #5 (the deferral context)
- Spike-artifact retention policy in cycle-status §"Conformance Notes" (governs spike artifacts when the targeted spike runs)

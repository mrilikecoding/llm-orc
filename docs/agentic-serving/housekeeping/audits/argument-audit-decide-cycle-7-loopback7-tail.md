# Argument Audit Report

**Audited document:** `docs/agentic-serving/decisions/adr-040-deterministic-completeness-gate.md`
**Source material:**
- `docs/agentic-serving/decisions/adr-037-session-termination-two-call-composition.md`
- `docs/agentic-serving/decisions/adr-038-remaining-work-anchor.md`
- `docs/agentic-serving/essays/research-logs/cycle-7-spike-sigma-premature-finish.md`
- `docs/agentic-serving/decisions/adr-035-client-tool-deliverable-form-contract.md`
- `scratch/spike-sigma-premature-finish/j3_diag/` (discharge run artifacts)
- `scratch/spike-sigma-premature-finish/j3_deterministic/SUMMARY.tsv` (pre-RESOLUTION runs)
**Genre:** ADR
**Date:** 2026-06-10

---

## Section 1: Argument Audit

### Summary

- **Genre:** ADR
- **Argument chains mapped:** 5 (invariance claim, ADR-038 non-contradiction, no-Conditional-Acceptance, persist-once hardening, scope boundary)
- **Issues found:** 6 (0 P1, 4 P2, 2 P3)
- **Pyramid coverage map:** N/A
- **Expansion-fidelity findings:** N/A

---

### P1 — Must Fix

No P1 findings. The core inferential chains in ADR-040 follow from the evidence and do not contradict ADR-037 or ADR-038 at a level that requires mandatory repair before the DECIDE gate closes.

---

### P2 — Should Fix

**P2-A: The "invariant" claim overstates what three small-n arms establish.**

- **Location:** §Context, §Rejected alternatives (both Frontier judge and J-1 sections), §Empirical grounding
- **Claim:** "The false-COMPLETE rate held flat across prompt and across judge capability." The ADR uses the word "invariant" without qualification in the Context section and presents the three-arm comparison as ruling out prompt-tuning and capability-escalation as fixes. The Empirical grounding section calls the result an "invariance."
- **Evidence gap:** The three arms are: baseline n=5 (1/5 complete), J-1 enumeration n=4 (1/4 complete), frontier MiniMax-m2.5 n=5 (1/5 complete). The denominators are 4 and 5 — below Spike σ's own pre-registered n=10 target (which was not reached because the spike pivoted to J-3 after diagnosing the judge as the wrong fix). At n=4 and n=5, the range of "true" completion rates consistent with the observations spans 0.05–0.53 (a 1/4 outcome is consistent with a true rate anywhere in a wide interval). Confirming that a 1/4 J-1 result is "no better than" a 1/5 baseline at these sample sizes does not permit inference of a population-level invariance — it permits inference only of "no improvement detectable at this n." ADR-037 and ADR-038 were careful to flag n=10 boundaries; the invariance language here is stronger than those precedents.
- **Recommendation:** Replace "invariant" with "showed no improvement at the measured n (J-1 1/4, frontier 1/5, baseline 1/5), consistent with an information-limit diagnosis." The rejected-alternative entries should use matched language: "no improvement measured at n≤5" rather than "same as the cheap local judge." The mechanism argument (produced-only digest is the bottleneck) stands and is well-grounded; the invariance framing is the overreach, not the diagnosis.

**P2-B: The no-Conditional-Acceptance claim rests on n=1 live session per tier, which is below the discharge bar ADR-037 and ADR-038 applied.**

- **Location:** §Status header, §Empirical grounding
- **Claim:** "The discharge runs are already done" and therefore no Conditional Acceptance is taken. ADR-040 cites "a 14b-coder run and the production-config run (14b seat, 8b coder)" as satisfying the layer-match discipline.
- **Evidence gap:** ADR-037's discharge gate required a single real-OpenCode session meeting two criteria (converge + advance). ADR-038 matched that gate — one session. Both predecessor ADRs accepted n=1 as sufficient for their discharge. ADR-040 follows the same n=1 pattern and is therefore internally consistent with the corpus's discharge bar. However, ADR-037 also carried a Conditional Acceptance through an entire BUILD cycle before discharge, while ADR-040 is claiming unconditional status at the DECIDE gate itself, before BUILD. The j3_deterministic SUMMARY.tsv (the pre-RESOLUTION data) shows run 1 COMPLETE (5 files) and run 2 EARLY_COMPLETE (1 file) — a 1/2 success rate on pre-persist-once code. The RESOLUTION runs (j3_diag/) are the two sessions cited in §Empirical grounding, and they are genuine full-convergence runs. The concern is not that the runs are fabricated, but that the gap between the pre-persist-once 1/2 result and the post-persist-once 2/2 result is elided: the pre-RESOLUTION j3_deterministic run 2 false-COMPLETE is mentioned only as the unnamed "run-2 false-COMPLETE that did not reproduce" in §Rejected alternatives. A reader auditing the discharge case should be able to see the complete run record, not just the two successful RESOLUTION runs.
- **Recommendation:** The Empirical grounding section should name the j3_deterministic SUMMARY.tsv runs explicitly — "pre-persist-once: 1/2 converged; post-persist-once: 2/2" — as part of the complete discharge record. This does not change the outcome (persist-once was applied before the RESOLUTION runs, so the 2/2 rate is the post-fix rate), but it makes the "already discharged" claim traceable to the full evidence rather than only the favorable half.

**P2-C: persist-once's defended-against failure mode was never observed in a live session; the argument for it rests on a "transient compaction" hypothesis for the pre-RESOLUTION run-2 false-COMPLETE that was never confirmed.**

- **Location:** §Decision (persist-once subsection), §Rejected alternatives (Re-derive each turn), §Context
- **Claim:** persist-once removes the verdict's dependence on the client resending the full task every turn. "A transient context-compaction or message-shape variance" is cited as the cause of the run-2 false-COMPLETE from 2026-06-09. persist-once "closes a transient-truncation path structurally."
- **Evidence gap:** The RESOLUTION diagnostic showed `requested=[5]` on all turns — the truncation hypothesis was explicitly refuted. The run-2 false-COMPLETE from 2026-06-09 did not reproduce and no causal confirmation was obtained. The Spike σ research log records this honestly: "The only mechanism left for it is a *transient* `_user_task` truncation on some particular turn." This is a hypothesis, not a confirmed mechanism. ADR-040's §Context says "persist-once (below) is therefore hardening that closes a transient-truncation path structurally, not a fix for an observed break" — this language is accurate. The issue is that the §Rejected alternatives section states "A transient compaction or message-shape variance collapses `requested` to empty mid-session and silently falls back to the judge, which is the candidate cause of the 2026-06-09 run-2 false-COMPLETE that did not reproduce" — presenting an unconfirmed hypothesis as the causal explanation for a specific run. The ADR should distinguish between "this path exists and is worth closing" (well-supported) and "this path caused the observed failure" (unconfirmed).
- **Recommendation:** In the rejected-alternatives entry, soften "which is the candidate cause of the 2026-06-09 run-2 false-COMPLETE" to make clear this is the surviving hypothesis after the truncation hypothesis was tested but not confirmed — not a confirmed attribution. The decision to apply persist-once is sound on the determinism-principle grounds alone; the causal attribution to run-2 is excess claim.

**P2-D: The ADR-038 "no contradiction" argument is structurally correct but elides a relevant asymmetry in what the two ADRs measured.**

- **Location:** §Context (ADR-038 relationship), §Rejected alternatives (deterministic checklist entry)
- **Claim:** "The two are distinct decision points with distinct judge-reliability evidence." ADR-038 rejected the checklist for the anchor because the judge named the unproduced deliverable at 20/20; ADR-040 adopts the checklist for the verdict because the judge fails the completeness call. The argument is that Spike σ triggered the fallback ADR-038 explicitly anticipated.
- **Evidence gap:** The argument is logically sound. The potential confusion is one of framing: ADR-038 rejected the checklist as "redundant" (not infeasible), and ADR-040's framing that "the anchor rides [the checklist] for free" means ADR-040 is also using the checklist for the anchor — the exact subsystem ADR-038 declined to build. The ADR addresses this directly in §Rejected alternatives: "ADR-038 rejected... for no reliability gain... now built for the verdict (where σ proves a reliability gain), and the anchor rides it for free." This is a correct argument. However, it means the completed state has the checklist serving both purposes (verdict and anchor) for named-file tasks, where ADR-038 only considered it for the anchor purpose. The framing "the anchor rides it for free" understates that the anchor is now deterministic too — changing its character from the judge-statement anchor ADR-038 validated at 19/20 (Spike ρ) to a framework-computed diff. A reader of ADR-038 who examines the named-file REMAINING case will see a different anchor mechanism than what ADR-038 validated, with no separate validation of the deterministic anchor's advance rate.
- **Recommendation:** Add a note in the Consequences or Fitness criteria that the named-file anchor is now the framework-computed diff rather than the judge's statement — and that while ADR-038's ρ-validation established 19/20 advance for the judge-statement anchor, the deterministic anchor on the named-file path rides the J-3 gate and was not separately validated for advance rate. The discharge runs showed monotonic convergence (implicitly validating it), but the lineage should be explicit. This is a documentation gap, not a logic error.

---

### P3 — Consider

**P3-A: The existence-vs-adequacy boundary paragraph references the 8b `cli.py` SyntaxError as illustration but does not distinguish between the two separate `cli.py` defects across the two runs.**

- **Location:** §Consequences (Negative), §Empirical grounding
- **Claim:** "The 8b discharge makes the boundary concrete: all five files existed and COMPLETE fired correctly, yet `cli.py` carried a trailing prose paragraph and would not parse (an ADR-035 form-gate bleed)."
- **Minor issue:** The j3_diag raw evidence shows two sessions: run1 (diagnostic, 14b coder) and run2/run3 (production config, see ws2 and ws3). The run.out confirms the 14b diagnostic produced a well-formed `cli.py`. The run2.out (the production 8b run) shows `cli.py` with `args.from` — a Python keyword collision causing a SyntaxError, not a "trailing prose paragraph." These are distinct defects. The Spike σ log §RESOLUTION says "Run B's `cli.py` carries a trailing explanation paragraph" while the raw run2.out shows the `args.from` keyword issue. The distinction matters because one is an ADR-035 form-gate bleed (prose after code) and the other is a coder-quality/semantic error (invalid Python attribute name). The ADR's categorization of the defect as "ADR-035 form-gate bleed" may conflate the two.
- **Recommendation:** Verify which defect is being cited as the ADR-035 illustration. If both defects are present across the two runs, the Consequences section should specify which run surfaced which defect. The illustration point still holds (existence-completeness fired correctly; form/quality failures are ADR-035/coder scope) but the concrete example should match the evidence.

**P3-B: The "one fewer model dispatch per trailing turn" claim in §Consequences is labeled drafting-time synthesis in the Provenance check but is presented without qualification in the Consequences section itself.**

- **Location:** §Consequences (Positive), §Provenance check
- **Minor issue:** The Provenance check correctly labels this as "an implementation read, not separately measured." However the Consequences section states it as a plain benefit: "One fewer model dispatch per trailing turn on the common coding-task shape (no judgment-seat call)." For a reader who only reads the decision sections, the qualification is invisible. This is a minor consistency issue between sections.
- **Recommendation:** Add a parenthetical "(implementation read; not separately measured)" to the Consequences entry, matching the Provenance check's honesty level. The claim is almost certainly correct — eliminating the judgment call is architecturally clear — but the labeling should be consistent.

---

## Section 2: Framing Audit

The framing audit makes the negative space of content selection visible. ADR-040 chose a specific framing; this section examines what that choice excluded.

### Question 1: What alternative framings did the evidence support?

**Alternative framing A: J-3 as a scope-restricted workaround, not a general fix.**

The evidence from Spike σ would support framing J-3 as a workaround specific to filename-extractable tasks, with explicit acknowledgment that the dominant real-world usage might not match that scope. ADR-040 notes this in its Consequences (Negative) — "a task that describes deliverables without naming files routes to the judge fallback, where σ's false-COMPLETE rate still applies" — but the decision section frames J-3 as the forward resolution rather than as a partial relief. An alternative framing would foreground the scope restriction more prominently: "for the named-file task shape, which is the common coding-task shape, the gate eliminates the failure mode; for other task shapes, the Spike σ failure mode remains unaddressed." The chosen framing is defensible (the named-file shape is the spike's test shape and the cycle's north-star task), but readers planning production deployment will need to assess how much of their actual task space is named-file.

**What would the reader need to believe for this framing to be right?** That non-file-naming task descriptions are uncommon enough in the target deployment envelope to be deferred without north-star risk. The ADR does not surface this assumption.

**Alternative framing B: persist-once as a design choice with a known trap, not pure hardening.**

The evidence (the pre-RESOLUTION j3_deterministic run 2 false-COMPLETE) and the spike's honest "did not reproduce" record would support framing persist-once as introducing a new failure mode (a sticky misextracted set from turn 1) in exchange for eliminating the truncation path. ADR-040 does name the turn-1 mis-extraction risk in Consequences (Negative): "A requested set that mis-extracts on turn 1 is sticky for the session." However, the framing of persist-once as "hardening" throughout the ADR minimizes the tradeoff character of the choice. A cleaner framing would name it as a design tradeoff: "persist-once eliminates the truncation path at the cost of making turn-1 mis-extraction sticky; this is a deliberate tradeoff, not a strictly safer design."

**What would the reader need to believe for this framing to be right?** That turn-1 mis-extractions are common enough relative to mid-session truncations to warrant naming the tradeoff prominently. The ADR's position — that turn 1 is the most reliable extraction point and truncation is more likely than extraction error — is a defensible assessment, but it is presented as a conclusion rather than as the contested assumption it is.

---

### Question 2: What truths were available but not featured?

**Available but not featured: the pre-RESOLUTION j3_deterministic runs.**

The `scratch/spike-sigma-premature-finish/j3_deterministic/SUMMARY.tsv` records two pre-persist-once J-3 runs: run 1 converged (5 files, COMPLETE at turn 7) and run 2 false-COMPLETEd (1 file, COMPLETE at turn 2, `judgment_verdict=COMPLETE`). This is a 1/2 completion rate before persist-once. The ADR mentions "the 2026-06-09 run-2 false-COMPLETE that did not reproduce" in passing but does not surface these two runs as a coherent pre-fix evidence record.

Why excluded: the ADR's narrative focuses on the RESOLUTION session as the discharge run record, and the pre-RESOLUTION runs are treated as exploratory/pre-fix data that was superseded. The Spike σ log §RESOLUTION handles the transition fairly, but ADR-040 compresses it.

Would its inclusion change the argument? Not its conclusion, but it would change the confidence level. The complete record is: pre-fix 1/2 → diagnosis (truncation hypothesis) → truncation hypothesis refuted by diagnostic → persist-once applied on determinism principle → post-fix 2/2. The post-fix 2/2 is the discharge evidence. Showing the complete arc makes the discharge claim more traceable and the persist-once rationale more transparent.

**Available but not featured: the spike's pre-registration violation of its own n target.**

Spike σ's pre-registration set n=10 per arm (amended from 8 by methods review). The actual arms ran to n=4 (J-1) and n=5 (baseline and frontier). The spike's own pre-registered early-stop rules (P2-A) do not authorize stopping at these n — early stop required either 8 completions (early GROUNDED) or 3 premature finishes (early INSUFFICIENT). None of those thresholds were reached; the spike pivoted because the failure mode was different from what F-σ.1 targeted. This is a legitimate and well-documented methodology shift (the live arm first approach surfaced the true failure). But ADR-040 uses the results of these sub-n arms to support its "invariant" claim (P2-A above) without noting that the arms did not reach their pre-registered n.

Would its inclusion change the argument? It would not change the decision (J-3 is supported by the convergence discharge, not primarily by the invariance claim), but it would appropriately bound the invariance language.

**Available but not featured: the diagnostic session's task configuration difference.**

The j3_diag diagnostic run (the one that produced `requested=[5]` on all turns and refuted the truncation hypothesis) used qwen3:14b as both coder and judge — the spike's simplified config, not the production 8b-coder config. The production 8b run (j3_diag/run2.out) is the Run B cited in §RESOLUTION. Both are genuine convergence runs. The ADR describes the diagnostic as "a single live diagnostic session read the per-turn `completeness:` log" without noting the 14b-vs-8b coder distinction between the diagnostic and the production run. This is a minor gap: the per-turn `requested` stability finding from the diagnostic (refuting truncation) applies to the 14b config, while the production config is confirmed by the 2/2 RESOLUTION runs separately.

---

### Question 3: What would change if the dominant framing were inverted?

ADR-040's dominant framing is: **the deterministic gate is the structurally correct fix for an information-limit problem; persist-once is hardening that closes a known edge case without introducing new risk.**

**Inverted framing:** The deterministic gate is a scope-restricted patch that moves a stochastic failure mode from "unreliable judge" to "unreliable filename extraction," and persist-once introduces a new class of sticky extraction errors in exchange for closing a truncation path that was never confirmed to cause a real session failure.

Under the inverted framing:

- **Claims that become weaker:** The "false-COMPLETE cannot occur for named-file tasks" claim — it can occur if `_extract_requested_deliverables` misidentifies the requested set on turn 1 (e.g., extracting a filename mentioned as context rather than as a deliverable). The ADR does name this risk in Consequences (Negative), but the inverted framing would foreground it rather than minimize it.
- **Claims that become stronger:** The observation that non-file-naming tasks still face the full Spike σ failure mode becomes more prominent under the inverted framing — the gate shifts the locus of failure from stochastic judge to heuristic extractor, and tasks outside the named-file shape are unhelped.
- **Evidence that becomes more salient:** The single observation in j3_deterministic that run 2 false-COMPLETEd before persist-once — evidence that the gate could still produce false-COMPLETEs when the requested set was derived incorrectly. Under the dominant framing this is evidence for persist-once; under the inverted framing it is evidence that the gate's correctness depends on the heuristic extraction step, which is unvalidated at scale.
- **What the ADR would need to address under the inverted framing:** A characterization of the extraction heuristic's false-positive and false-negative rates across a realistic task distribution, and an assessment of how sticky turn-1 errors compare in severity to the mid-session truncation they replace.

---

### Framing Issues

**P2-F1: The named-file scope restriction's cost is not quantified or bounded.**

- **Location:** §Consequences (Negative), §Decision
- **Issue:** The ADR names the scope restriction but does not assess what fraction of the real task space the gate covers. "A task that describes deliverables without naming files routes to the judge fallback, where σ's false-COMPLETE rate still applies" — this is an honest statement, but it leaves unaddressed whether named-file tasks are 60%, 90%, or 99% of the deployment shape the cycle targets. If the target deployment is dominated by named-file coding tasks (which is likely given the north-star task shape), the restriction is minimal. If it is not, the ADR overstates the gate's practical impact. The cycle's north-star task (the temperature library) is named-file, which supports the ADR's implicit assumption, but the assumption should be stated explicitly rather than left implicit.
- **Recommendation:** Add a one-sentence framing note in §Consequences or §Decision: "The cycle's target task shape (multi-file coding tasks named explicitly in the request) is the named-file shape; the restriction is expected to cover the majority of the north-star deployment envelope, but this is an untested assumption."

**P2-F2: persist-once's only validation is a unit test simulating the failure mode; no live compaction event occurred in either discharge session.**

- **Location:** §Empirical grounding, §Decision (persist-once subsection)
- **Issue:** The Spike σ log §RESOLUTION notes that persist-once was validated by "2 unit tests (one simulates the run-2 compaction and proves the gate holds REMAINING)." The live discharge sessions (j3_diag/) show no compaction event — `requested=[5]` on all turns. So persist-once's specific defended-against failure (mid-session truncation collapsing `requested` to empty) is validated only by a unit test that simulates the failure, not by a live observation of the failure. The ADR notes this in §Context ("The gate was never the silent no-op the leading hypothesis feared") and §Rejected alternatives ("The diagnostic showed the gate firing correctly with per-turn re-derivation"). The issue is that the §Empirical grounding section presents the live discharge as full discharge evidence without distinguishing what the live runs validated (the gate fires correctly under nominal conditions) from what only unit tests validated (the gate holds REMAINING when compaction occurs).
- **Recommendation:** The Empirical grounding section should distinguish: "Live discharge validates nominal convergence (requested stable, produced monotonic); persist-once's truncation path is validated by unit test (simulates compaction) with no live compaction event observed." This is an honest and important distinction for BUILD — if the real client compacts messages differently than the unit test simulates, persist-once may not hold as designed.

**P3-F1: The ADR's confidence language ("the false-COMPLETE cannot occur for named-file tasks") is absolute where the mechanism has a known exception path (heuristic mis-extraction).**

- **Location:** §Consequences (Positive)
- **Issue:** The ADR states "The false-COMPLETE cannot occur for named-file tasks: the verdict is a set comparison, not a model inference." This is the core benefit claim. But the Consequences (Negative) section immediately qualifies it: the heuristic can mis-extract. If `_extract_requested_deliverables` extracts a superset of the actual deliverables (including filenames mentioned contextually), COMPLETE will never fire — the gate will loop forever or hit the AS-3 cap. If it extracts a subset (missing a deliverable without a recognizable extension), COMPLETE fires early. Both of these are paths to failure that the positive claim implicitly excludes by assuming correct extraction. The claim should be: "The false-COMPLETE cannot occur for named-file tasks *when the extraction is correct*."
- **Recommendation:** Qualify the Consequences (Positive) entry: "The false-COMPLETE cannot occur for named-file tasks where extraction succeeds: the verdict is a set comparison, not a model inference. The extraction heuristic is the remaining stochastic element; its failure modes (over-extraction loops, under-extraction early finish) are scoped to the refutable FC."

---

## Convergence-Saturation Signal (ADR-094)

**Convergence-Saturation Signal:** NOT TRIGGERED

- Round number: R1
- P1 count this round: 0 (Section 1: 0; Section 2: 0)
- P2 count this round (new, non-carry-over): 6 (P2-A invariance overreach; P2-B incomplete discharge record; P2-C unconfirmed causal attribution; P2-D deterministic anchor advance rate gap; P2-F1 scope restriction unconstrained; P2-F2 persist-once live vs unit-test gap)
- New framings or claim-scope expansions: The framing that the named-file restriction may not cover a substantial portion of the real deployment task space; the framing that persist-once trades truncation risk for extraction-stickiness risk rather than strictly hardening the gate; the framing that the deterministic anchor's advance rate was not separately validated.
- Recommendation: CONTINUE to next round. P1 count = 0 satisfies condition 1; P2 count = 6 fails condition 2 (>1 new P2); condition 3 fails (new framings surfaced). The document is structurally sound and the core reasoning is well-grounded — the P2 findings are documentation and scope-bounding issues, not logical errors. A focused R2 addressing the invariance language (P2-A), the discharge record completeness (P2-B), and the two framing gaps (P2-F1, P2-F2) would likely trigger the signal.

*Single-purpose re-audits (dispatched per the re-audit-after-revision rule) omit this section. Form-change events reset the round-count baseline — the first audit on a new form is its R1.*

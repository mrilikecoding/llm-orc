# Research Design Review — Compound Rung A×C (Cycle 7, agentic-serving)

**Reviewed artifact:** `scratch/spike-ladder-compound-axc/probe.py` (module docstring = pre-registration of record)
**Constraint-removal response included:** n/a (spike pre-registration, not an ADR-082 constraint-removal artifact)
**Comparator:** `scratch/spike-ladder-axisC/probe.py` + `scratch/spike-ladder-axisC/RESULTS.md`
**Depth harness reference:** `scratch/spike-ladder-rung2/probe.py`
**Date:** 2026-06-08
**Review scope:** Tier-1 pre-registration methods review. Data not yet collected.

---

## Summary

- **Flags raised:** 5 (1 P1, 3 P2, 1 P3)
- **Criteria applied:** 1–4 (ADR-082)
- **Bottom line:** not sound to run as-is. The discriminability gap (P1) means the rung's central measurement cannot resolve its own hypotheses at n=10. Fix n before running.

---

## Findings

### P1 — Statistical discriminability: the rung cannot resolve its own hypotheses at n=10

**The problem.** The pre-registration frames AC1 churn vs RC1's 2/10 as "the interaction measurement." Three hypotheses are stated: H1 (churn > 2/10), H0 (churn ≈ 2/10), H-dilute (churn < 2/10). But at n=10, these hypotheses are not resolvable.

For H1, the smallest "meaningful amplification" would be 4/10 (a doubling of the observed repair churn). At n=10, a two-proportion comparison of 2/10 vs 4/10 gives a z-statistic of approximately 0.94 and a one-tailed p of approximately 0.17. That is noise. Even a result as strong as 5/10 vs 2/10 gives z≈1.37, p≈0.085 — still not distinguishable at any conventional threshold. The only AC1 result that would be visually unambiguous from RC1's 2/10 is 0/10 (H-dilute confirmed) or ≥7/10 (H1 confirmed unambiguously) — outcomes with low prior probability given axis C's mechanism.

This matters because the pre-registration explicitly compares AC1 churn against RC1's 2/10 as a reference. If the result is 3/10 or 4/10, the researcher has no basis for claiming interaction amplification; if it is 2/10, no basis for claiming independence; if it is 1/10, no basis for claiming dilution. The rung's n is sized for characterization at the PASS threshold (advance rate), not for the interaction-effect comparison it advertises as its purpose.

The PASS criterion softens this implicitly: "advance markedly (>= axis-C's 8/10 RC1 reference)." A rung that produces AC1 advance=8/10, churn=2/10 — identical to RC1 — technically passes, and the pre-registration then records "no interaction detected" even though the measurement was not powered to detect an interaction of the size H1 predicts.

**Recommended fix.** Two options, in order of preference:

Option A: Increase n on AC1 specifically. n=25 on AC1 (15 additional runs) makes H1=4/10 vs H0=2/10 statistically distinguishable at p<0.10 (z≈1.49 one-tailed at n=25 each). If the researcher accepts an informal "3 or more additional churns above RC1's 2" as the decision criterion, n=20 on AC1 is a reasonable middle ground — it makes a 5/10 result interpretable as amplification and a 2/10 result interpretable as no interaction, with 3/10 and 4/10 remaining ambiguous but documented as such.

Option B: Reframe the claim. If n stays at 10 for all states (cost and time budget), the pre-registration should state explicitly that AC1 can characterize the direction of the churn count (higher/lower than RC1's 2) but cannot establish statistical significance for H1 vs H0. The interaction finding is then directional evidence, not a resolved hypothesis. This is weaker but honest.

Either option is acceptable. Leaving the current pre-registration language ("the interaction measurement vs RC1's 2/10") without this caveat overstates what n=10 can deliver.

---

### P2-A — Confound: README is a second variable, not a clean one-variable change

**The problem.** The pre-registration claims: "the ONLY difference from the axis-C single rung is the added README, so the comparison isolates the interaction effect." This is correct at the task-text level. It is not correct at the model-behavior level.

README.md is a prose deliverable. The three code deliverables across axes A/B/C have been file paths with predictable names (`string_utils.py`, `test_string_utils.py`, `settings_loader.py`). The seat-filler's action-selection at AC1 asks: among {test_string_utils.py, README.md}, which do I write next? This is a different selection problem from RC1, which asked: {test_string_utils.py} — write it or churn back to the module.

Two variables change simultaneously between RC1 and AC1:
1. The number of remaining deliverables (1 → 2) — the variable the design intends to vary.
2. The type mix of remaining deliverables (code → code+prose) — a second variable that changes implicitly.

If AC1 churn is lower than RC1's 2/10, one plausible explanation is H-dilute (more remaining work pulls attention away from the fixed module). Another plausible explanation is that the model treats "write README" as a lower-friction next step than "write test" and the salience of the fixed module does not compete as effectively when a prose option is available. These are not distinguishable at AC1 without a third condition (e.g., AC1 with two code deliverables remaining rather than one code + one prose).

The `_kind()` function correctly classifies README.md (the `"readme" in p` branch fires first, before the test and module checks). The measurement instrument is not the problem — the design claim is.

**Recommended fix.** Two options:

Option A (preferred): Replace README with a second code deliverable (e.g., `utils_v2.py` — a second write-only Python module). This makes AC1 a pure deliverable-count manipulation: {test, utils_v2} remaining, both code, same type as axis A's depth-3 task. The interaction isolates count vs repair-salience with no content-type confound. The rung then also inherits axis A's clean write-only structure.

Option B (document and scope): Keep README, add a one-paragraph caveat to the pre-registration noting that the README content type is a secondary variable: "The README introduces a prose deliverable. Any advance-rate difference between AC1 and RC1 may reflect either the remaining-count effect (H1/H0/H-dilute) or a content-type preference effect. These are not distinguished at AC1. The finding is: does the repair module's salience interact with a larger remaining set? The content-type variable is acknowledged as uncontrolled."

Option B is cheaper and still lets the rung run. It is honest about what the comparison establishes. Option A produces a cleaner design but requires a task-text change (which must be pre-registered before running).

---

### P2-B — Premature narrowing: advance degradation and no-tool-call rate are not measured at the rung level

**The problem.** The pre-registration says "AC0/AC1/AC2/ACc — churn (re-target module|test) is the interaction measurement." Churn is the named limit. But the progressive ladder's design document (§5, secondary measures) flags two other failure modes as worth tracking: no-tool-call rate (premature finish without a file action) and remaining-naming accuracy (does the judge correctly name one of the two remaining deliverables at AC1, where both test and readme remain?).

These failure modes could be amplified at A×C and would not be captured in the current measurement design:

- **No-tool-call at AC1**: if the model produces a `FinishTurn` at AC1 (two deliverables remaining), that outcome is classified as neither advance nor churn. The summary prints `advance={adv}/{n}` but a FinishTurn is counted as advance=False, churn=False — it disappears into the denominator reduction without a label. In the pre-registration's PASS criterion, "AC1/AC2 advance markedly (>= axis-C's 8/10)" would fail if FinishTurn shows up, but the cause (premature convergence vs. selection confusion) would not be recorded separately.

- **Remaining-naming accuracy at AC1**: the judge must name one of {test_string_utils.py, README.md} at AC1. At RC1, the judge had only one remaining deliverable to name (10/10 REMAINING, presumably correct naming drove the 8/10 advance). At AC1, the judge must choose which of two remaining deliverables to name in the anchor. Whether the judge correctly names an unproduced deliverable (rather than the already-fixed module) is untracked by the current harness. If the judge's AC1 naming accuracy is lower than RC1's, that would confound the churn comparison: lower advance at AC1 could reflect judge naming degradation at two-remaining-in-repair-context rather than seat-filler churn.

**Recommended fix.** Add two fields to the row dict in `_run_state`:
- `no_tool_call`: boolean, True when outcome is `FinishTurn` (already implicitly present in the outcome field, but surface it explicitly so the summary can count it separately from advance and churn).
- `judge_named_produced`: boolean, True when the `TurnDecision.judgment_verdict` is `REMAINING` but the judge's remaining-statement names a produced deliverable (the module, at AC1). This requires inspecting the judge's statement text for the produced-file name — the same fragility the axis-C result flagged for `turn_shape`, but here used as a diagnostic measure rather than a classification. At minimum, log the full judge verdict string per row so it can be inspected post-run.

The second measure doesn't require real-time classification; logging the verdict string is sufficient. Zero added cost beyond what is already being called.

---

### P2-C — Embedded conclusion in the PASS criterion: the interaction finding is constructed to confirm, not to test

**The problem.** The PASS criterion reads: "advance markedly (>= axis-C's 8/10 RC1 reference) with delegation preserved; ACc converges. The interaction finding is the AC1 churn vs RC1 2/10 comparison."

This structure means there is no FAIL condition for the interaction question. The rung passes (characterization, the pre-registration notes) if advance is ≥ 8/10. The churn comparison is recorded as "the finding" regardless of outcome. But if churn at AC1 is 2/10 (same as RC1), the finding is "no interaction detected" — which looks like a null result but may simply be a reflection of the rung not being powered to detect an interaction (the P1 problem). And if churn is 3/10, the finding is ambiguous but gets written up as "slight amplification trend."

The pre-registration does not state a churn level at AC1 that would constitute evidence against any of the three hypotheses. This makes the interaction question unfalsifiable as stated: any AC1 churn result — 0/10, 2/10, 4/10, 8/10 — can be interpreted as supporting one of H1, H0, or H-dilute post-hoc.

This is the same motivated-reasoning capture the prior methods review flagged for the cloud-escalation trigger: a threshold set post-run against a known result is not a threshold.

**Recommended fix.** Pre-register the decision boundary before running:
- If AC1 churn ≤ 1/10: H-dilute supported (more remaining deliverables reduce repair-module salience).
- If AC1 churn is 2–3/10: no interaction detected (within RC1 noise).
- If AC1 churn ≥ 4/10: H1 (amplification) supported; compound-rung limit identified.

These thresholds are rough given n=10 (see P1), but pre-registering them prevents post-hoc reinterpretation. If n is increased per the P1 recommendation, the thresholds can be tightened accordingly.

---

### P3 — Is A×C the right first compound rung?

**The question.** The prior methods review (research-methods-progressive-ladder.md, P3-A) flagged that the compound-rung insertion trigger was unspecified. The compound-rung design is now specified — but it is worth examining whether A×C is the sharpest limit-finder for the effort vs. the available alternatives.

**What A×C compounds:** repair salience (axis C) + deliverable count (axis A), where repair salience is the mechanism identified in the single-axis finding (the "fix" framing keeps the module salient). The hypothesis is that a larger remaining set amplifies that salience pull.

**What A×B would compound:** deliverable count (axis A) + mixed read-then-write (axis B). The prior methods review (P2-C) flagged the carry-side as the most undercharacterized risk in the whole ladder — the WP-LB-L discharge left it uncharacterized, and axis B was sequenced after axis A precisely because "axis A failure would require redesign that changes axis B's test conditions." That sequencing rationale has now been discharged: axis A passed cleanly (8/10, 9/10, no churn). The carry-side risk is still the only major uncharacterized failure mode in the judge's deliverable-accounting. A×B would compound the read-then-write flow with a third write deliverable — stressing the judge's ability to name two remaining writes after a read, which is directly relevant to the WP-LB-L scope note.

A×C, by contrast, combines two mechanisms that are both characterized: repair salience (2/10 churn, backstopped) and deliverable count (8/10, 9/10 advance, no churn). The interaction between them is real science, but the stakes of finding "2/10 churn stays at 2/10 with an added README" are lower than finding "the judge's carry-side accounting breaks under depth."

This is a research prioritization question, not a design flaw. A×C is sound to run eventually. The question is whether it is the highest-value compound rung to run first. If the goal is to find limits, A×B is more likely to find one because it stresses a mechanism (carry-side accounting) with no characterized baseline, whereas A×C stresses a mechanism (repair salience) with a 2/10 baseline that is already backstopped.

**Recommendation.** Flag this as a design choice, not a blocker. If the pre-registration intent is explicitly to study the repair-salience interaction (a scientifically valid question), run A×C. If the intent is to find the soonest limit in the compound space, A×B has a stronger claim. State explicitly in the pre-registration which goal is being served.

---

## Question Set Assessment

### Premature narrowing / prior-art treatment

The rung treats axis C as prior art correctly — quoting the RC1 result (2/10 churn, 8/10 advance) and using it as the comparison baseline. The axis A results are also cited correctly (8/10, 9/10, no churn). No narrowing-without-prior-art-treatment detected.

One narrowing concern not flagged elsewhere: all four states assume that the churn mechanism is action-selection error (the seat-filler picks the wrong file). An alternative mechanism is judge verdict error — the judge could misclassify AC1 as COMPLETE (module fixed + readme + test somehow already satisfied) which would cause FinishTurn, not churn. The no-tool-call / FinishTurn path is present in `_run_state` but not surfaced in the PASS criterion or the summary metrics in a way that would flag this failure mode separately. See P2-B.

### Incongruity surfacing

One incongruity in the research context that the pre-registration does not surface for examination.

Axis C's RC1 churn (2/10) was backstopped by re-judgment: the next trailing turn re-judged REMAINING and re-anchored. The axis C results note "advance still 8/10 and the churn is backstopped by re-judgment (the next trailing turn re-judges REMAINING and re-anchors) + the AS-3 cap." The compound rung A×C adds one more deliverable — making the re-anchoring path more complex at AC1 (the judge now has two remaining deliverables to name, so the repair carries only 1/3 of the total session weight rather than 1/2 at RC1). 

The incongruity: the backstopping mechanism may itself be weaker at A×C than at axis C, because the judge's remaining-statement at AC1 must choose between two unproduced files. If the judge tends to name the first alphabetically or the most code-like (test over readme), the backstopping could be systematically biased. This is not simply "the same mechanism with one more file" — it is a new condition for the judge prompt. The pre-registration does not examine this.

This is related to P2-B's remaining-naming accuracy concern. Flagged here as an incongruity because a mechanism that was characterized as a backstop in a two-deliverable context is being relied upon in a three-deliverable context without re-characterization.

### Coverage gaps

1. The `churn_target` summary in `_run()` computes `churn_mod` (module re-targets) but not `churn_test` (test re-targets, which would occur at AC2 if the model churns after both module and test are produced and only readme remains — which cannot happen, as readme is the only remaining at AC2; this is fine) or `churn_readme` (readme re-targets after AC2 produces the readme — not possible at ACc either). The churn-target summary is adequate for the rung's states. No gap here.

2. The `delegated` field: the pre-registration says "delegation preserved" as a PASS criterion element. The harness computes delegation correctly (`outcome.delegated_ensemble is not None`). But delegation is not expected to vary between AC1 and RC1 by hypothesis — axis C had delegation 10/10 at RC1. If delegation drops at AC1, it would be a finding, but the pre-registration does not state what delegation rate would be notable. Minor coverage gap; worth noting the RC1 reference (10/10 delegated) explicitly in the pre-registration so a drop is flagged.

3. The `carry` branch in `_run_state` (lines 237-242): when the outcome is neither `ApplyWork` nor `FinishTurn`, the harness reads `getattr(outcome.invocation, "arguments", "")` and passes it to `_kind()`. The `arguments` attribute on an invocation is likely a structured dict or JSON string, not a file path string. If this branch fires, `_kind()` would likely return `"other"` and the row would record `churn=False, advanced=False` — effectively silent. This branch was not exercised in axis C (carry outcomes: 0 in RC0/RCc, not expected at those states). At AC1 with a larger remaining set, a carry outcome is plausible. If it fires, the result is silently mis-measured. Worth checking whether this branch is dead or live under the real `decide()`.

### Recommendations

Prioritized:

1. **(P1 — must-fix before running)** Increase n on AC1 to at least 20 (preferred: 25) or reframe the pre-registration's language to state explicitly that the AC1 churn comparison is directional evidence only, not a resolved hypothesis. The current framing says "the interaction measurement vs RC1's 2/10" — this is not achievable at n=10 for the range of churn values the design considers meaningful.

2. **(P2-A — fix before running if keeping README)** If the README deliverable stays in the task, add one sentence to the pre-registration acknowledging that the content-type variable (prose vs code) is not controlled. If the README can be replaced with a second code file (e.g., `utils_v2.py`), do that instead — it removes the confound entirely and makes the comparison cleaner.

3. **(P2-B — fix before running)** Add a `no_tool_call` count to the per-state summary (FinishTurn outcomes separate from advance and churn). Log the full judge verdict string per row. These are zero-cost additions to the existing harness that make the measurement more complete.

4. **(P2-C — fix before running)** Pre-register the churn decision boundary: what AC1 churn count constitutes evidence for H1 vs H0 vs H-dilute. Without pre-registered thresholds, the interaction comparison is unfalsifiable.

5. **(P3 — address before running, not a blocker)** State explicitly in the pre-registration whether A×C is chosen to study repair-salience interaction specifically (scientifically valid) or to find the earliest compound limit (where A×B may be more productive). The current framing implies limit-finding but the task choice is better suited to mechanism-characterization.

---

## Harness notes (not research design flags, but pre-run checks)

These are implementation observations that should be verified before running, not design problems:

- **`carry` outcome branch (lines 237-242):** reads `getattr(outcome.invocation, "arguments", "")` and passes that string to `_kind()`. If `arguments` is a dict or structured object, `_kind()` gets `repr()` output or an empty string and returns `"other"`. If this branch fires at AC1, the result is silently unmeasured. Verify whether the `carry` outcome type from `decide()` carries a file path anywhere, and if so, extract it correctly. If the branch is genuinely unreachable under the real `decide()`, add an assertion.

- **State encoding:** A×C uses `["module"]` string keys against `DELIVERABLES`, while axis C used `(label, action_kind, path, result)` tuples. Both produce equivalent session contexts. The change is clean.

- **`_kind()` ordering:** `"readme" in p` fires before `"test" in p`, which is correct. No classification risk for the current deliverable set.

# Research Design Review — Compound Rung A×B (Cycle 7, agentic-serving)

**Reviewed artifact:** `scratch/spike-ladder-compound-axb/probe.py` (module docstring = pre-registration of record)
**Constraint-removal response included:** n/a (spike pre-registration, not an ADR-082 constraint-removal artifact)
**Comparator:** `scratch/spike-ladder-axisB/RESULTS.md` (baseline) + `scratch/spike-ladder-rung2/RESULTS.md` (depth baseline)
**Prior review:** `docs/agentic-serving/housekeeping/audits/research-methods-compound-axc.md`
**Date:** 2026-06-08
**Review scope:** Tier-1 pre-registration methods review. Run already executing; findings determine whether results are trustworthy or the rung requires a re-run.

---

## Summary

- **Prior A×C findings reviewed:** P1 (n=10 discriminability), P2-A (README content-type confound), P2-B (no-tool-call + naming unmeasured), P2-C (unfalsifiable interaction claim), P3 (A×B sharper for carry-side)
- **Flags raised:** 1 P2 (boundary ambiguity zone), 1 P3 (trajectory unmeasured)
- **Prior P1/P2/P3 discharge:** all five A×C findings are discharged by the A×B design
- **Bottom line:** the run is trustworthy. No P1 invalidates the results. One P2 caveat applies to the holds/degradation interpretation in the 0.73–0.90 zone; it does not invalidate the run.

---

## Prior Findings Discharge

### P1 — Statistical discriminability (A×C: n=10 could not resolve churn hypotheses)

**Discharged.** The A×B design's PRIMARY signal is advance degradation from axis B's 0.90 baseline, not a small-effect churn comparison. With n=15 on R1/R2/R3 and boundary at 12/15:

- P(classified as holds | true p=0.90) = 94.4%
- P(classified as degradation | true p=0.73) = 58.1%
- P(false limit | true p=0.90) = 0.003%

The holds case (p stays near 0.90) is nearly unambiguous. The limit case (p drops to <=0.47) is well-separated. The discriminability is substantially better than A×C's churn comparison (z~0.94, p~0.17 at n=10 for 2/10 vs 4/10). The P1 problem does not transfer.

### P2-A — README content-type confound

**Discharged.** All four deliverables are code (settings_loader.py, test_settings_loader.py, validator.py, config.py read). No prose deliverable. The manipulation is a clean depth increase: read + 2 writes (axis B) vs. read + 3 writes (A×B), content type held constant.

### P2-B — Unmeasured no-tool-call and remaining-naming accuracy

**Discharged.** The harness explicitly measures both:
- `no_tool`: boolean per row, counted in the per-state summary as `no_tool={count}/{n}`
- `judge_names`: `_names_in()` extracts filename fragments from the judge's text; `clean_naming` counts rows where the judge named only unproduced write deliverables; `named_read` counts rows where config.py was named. Both are printed in the per-state summary.

### P2-C — Unfalsifiable interaction claim

**Discharged.** The pre-registration contains explicit pre-registered decision boundaries with three ordered regions: >=12/15 holds / 8-11/15 degradation + cloud-contrast trigger / <=7/15 clear limit. Each region has a defined interpretation. No post-hoc reinterpretation is available — the cloud-contrast trigger fires or it does not based on the pre-committed count.

### P3 — A×B sharper than A×C for carry-side limit finding

**Discharged by choice.** The practitioner chose A×B on the P3 recommendation. The A×B pre-registration correctly frames the research question as carry-side-under-depth, citing the axis B FC-61 discharge and the uncharacterized question of whether the carry-side holds as the write sequence deepens to 3.

---

## A×B-Specific Assessment

### 1. Statistical power for the primary signal

The advance boundary at 12/15 (0.80) is the holds/degradation threshold. The boundary is sensibly placed given the 10% no-tool-call floor established in axis B and rung 2 (both showed ~1/10 premature finish). At that floor, a "perfect carry-side" would yield roughly 13-14/15, so 12/15 is not the null — it is a generous-but-not-trivial holds criterion.

**Residual P2-level ambiguity zone.** The holds/degradation boundary at 0.80 is not sharply resolved at n=15. If the true rate is 0.80 (exactly at the threshold), the design calls it "holds" 65% of the time and "degradation" 35% of the time. The zone [0.73, 0.90] is the gray area — an observed result of, say, 12/15 or 11/15 does not cleanly separate p=0.85 from p=0.73.

This is a P2 caveat, not a P1 invalidation. The boundary is pre-registered, the extremes are well-separated, and the design's purpose is ladder characterization (where is the ceiling?), not a confirmatory trial. A result of 13+/15 is unambiguously "holds"; a result of <=7/15 is unambiguously "limit"; the ambiguity lives in the middle band. Caveat the interpretation if results land in 8-12/15.

### 2. Manipulation cleanliness

Clean. The only change from axis B is the addition of one write deliverable (validator.py), all-code, with the same config.py content. The task text is slightly longer to name the third deliverable — this is inherent to the depth manipulation, not an independent variable. The read action tuple is identical between axis B and A×B (same CONFIG_CONTENTS, same path). No new confound is introduced.

### 3. Carry-side integrity operationalization

The three carry-side measures (read_churn, clean_naming, named_read) are the right operationalization and they are correct implementations.

- `read_churn` correctly fires only when `target == "read"` AND `"read" in produced_kinds` — it cannot fire at R0 (read not yet in record), correctly fires at R1/R2/R3 if the seat-filler re-reads config.py.
- `clean_naming` at each state requires `set(judge_names) <= unproduced_writes` — this becomes stricter with depth (at R3, only "validator" is an acceptable name; naming "test" or "module" at R3 would count as unclean). This is the right operationalization: naming accuracy is expected to stay clean as depth grows, and any drift shows immediately.
- `named_read` detects if the judge ever counts config.py as a remaining deliverable — the FC-61 failure mode under depth.

One carry-side aspect not measured: whether the judge's REMAINING anchor is specific enough to steer the seat-filler (e.g., "write validator.py" vs. "there are files remaining"). A vague anchor that produces no file names would show as `judge_names = []` and would NOT count as `clean_naming` (the `judge_names and` guard prevents empty lists from counting). This is appropriate: vague anchors are penalized. The anchor specificity is partially instrumented through clean_naming, though not fully isolated.

### 4. Remaining-naming detection soundness

The `_names_in()` occurrence-count disambiguation is correct.

The key disambiguation: `test_settings_loader.py` contains `settings_loader.py` as a substring. When the judge mentions only `test_settings_loader.py`, both the test and module fragments appear exactly once each in `str.count()` — the condition `count(module_frag) > count(test_frag)` evaluates to `1 > 1 = False`, so module is NOT added. When the judge mentions `settings_loader.py` separately (standalone), module_frag count becomes 2 (once standalone, once embedded in the test name), test_frag count remains 1, and the condition fires correctly. Verified by direct computation across cases including ties (both mentioned once) and edge cases (test mentioned twice: module count 2, test count 2, no spurious module detection).

The `_kind()` ordering is also correct: `"validator"` check precedes `"test_settings"`, which precedes `"settings_loader"`, which precedes `"config"`. There is no ambiguity between deliverable names at classification time.

### 5. A×B-specific embedded conclusions and premature narrowing

No embedded conclusions that blind the design to a limit depth-after-read could produce.

**Limits the design can detect:**
- Advance degradation (primary)
- Read-churn (carry-side re-reads under depth)
- Write-churn (seat-filler re-targets produced files)
- Premature finish / no-tool-call (false convergence at REMAINING states)
- False continue at RC (judge miscounts and says REMAINING when all done)
- Judge naming drift (naming produced or read files as remaining)

**Limit the design cannot detect (documented):** cumulative trajectory degradation in a live multi-turn session. Each state is an independent reconstructed session (pre-populated action record), not a single live run to completion. If the 3rd write succeeds 15/15 in isolation at R3 but fails more often as the 3rd step of a live session (due to context accumulation, position effects, or inter-turn interactions), the state-injection design would not surface it. This is shared with all prior rungs and is documented in the rung-2 results as a known limitation. Not a new A×B-specific blind spot, but worth carrying into the results record.

There is no embedded conclusion that the carry-side will hold — the pre-registered degradation trigger (<=11/15 → cloud contrast) is a genuine falsification condition that the run could hit.

### 6. Validator.py completeness specificity

Adequate. The task text specifies "a validator.py module that validates the loaded settings" — this is a named file with a named purpose, sufficient for the judge to (a) count it as a deliverable, and (b) recognize its absence at R3 as REMAINING. The name "validator.py" is unambiguous in Python convention (unlike, say, a vague "helper.py").

One minor note: "validator" is slightly less canonically expected than "test_*" naming patterns, so the judge may be marginally less confident naming it at R3 than it would be naming a test file. The clean_naming metric at R3 will surface any systematic failure here directly — if the judge at R3 names "test_settings_loader.py" (already produced) instead of "validator.py", that row is not clean. If this is a real problem it shows as clean_naming < 13/15 at R3, which is exactly the pre-registered integrity threshold. The metric is already the right instrument for this risk.

One implementation note: the harness records `_names_in(judge.last)` which captures fragment matches in the full judge text output, not just the remaining-statement field. If the judge's text mentions `validator.py` in any context (e.g., "you've written settings_loader.py and test_settings_loader.py but not validator.py"), it will be captured correctly. No concern here.

---

## Question Set Assessment

### Premature narrowing / prior-art treatment

The pre-registration correctly treats axis B as prior art: quoting the axis B results (read-first 10/10, advance 9/10, churn 0, converge 10/10) and using them as the comparison baseline for every pre-registered boundary. The rung-2 depth baseline (advance 8-9/10, churn 0/10 at depth 3) is implicitly treated as prior art via the no-cloud-contrast result cited in the background framing. No narrowing without prior-art treatment detected.

### Incongruity surfacing

No incongruity in the research context that the design fails to surface. The prior A×C review flagged one incongruity (the backstopping mechanism weaker under more deliverables). A×B avoids this: the backstopping mechanism in A×B is identical in structure to axis B (re-judgment + AS-3 cap), just under deeper depth. The mechanism is characterized; the question is whether it holds. No adjacent simple-vs-complex contrast is visible in the research context that A×B fails to examine.

### Coverage gaps

One: the no-tool-call pre-registration note says "axis B ~10%. Rising with depth = premature-finish limit." The pre-registered boundary for no_tool is descriptive ("tracked; rising with depth = limit") without a specific count threshold. If no_tool is 3/15 at R3 vs. 1/10 at axis B, the pre-registration does not say whether that constitutes a "rising" finding. The count is tracked and visible in the raw output, but the interpretation threshold is informal. For a ladder rung this is acceptable — no_tool is a secondary measure, and the advance count already absorbs its effect (no_tool subtracts from the advance numerator). Not a flag, just a note.

### Recommendations

**P2 (caveat the interpretation).** If any REMAINING-state advance lands in 8-12/15, the pre-registration boundary is ambiguous at n=15. The holds/degradation split (>=12 vs <=11) has false-holds rate of 35% at true p=0.73. State explicitly in the results record that 8-12/15 is a soft band: results in this range support the cloud-contrast trigger but cannot confirm the specific true rate. This does not require a re-run — it is a results-interpretation caveat that should accompany the findings.

**P3 (optional, future).** If A×B passes cleanly (advance holds at 13+/15 across R1/R2/R3), the state-injection method's trajectory limitation means a live 4-turn session to completion has not been characterized. That would be a natural follow-up in the BUILD phase (the real OpenCode integration test), not an additional ladder rung. Flag in the results disposition for the BUILD phase.

---

## Harness Implementation Notes (not flags, pre-run checks for completeness)

These were checked against the actual code:

- **`CarryClientTool` branch (lines 278-285):** The A×C review flagged this branch for silently misclassifying if `arguments` was a non-dict. The A×B harness fixes this: `isinstance(args, dict)` is checked before dict access; string fallback handles the string case; empty fallback returns `"other"`. The branch is correctly implemented.

- **`clean_naming` denominator:** Computed as raw count over n (not as fraction of rows with non-empty judge_names). Rows with `judge_names = []` (null/empty judge output) do NOT count toward clean_naming. This correctly penalizes null anchors. The denominator is implicit (clean_naming / n compared to 13/15). This is fine but worth stating explicitly in the results record to avoid ambiguity in interpretation.

- **WP-LB-M acceptance:** The pre-registration documents expected turn_shape behavior (R0 read → carry; R1/R2/R3 write → generation; RC finish → carry) and uses this run as the WP-LB-M real-model acceptance. This dual purpose is legitimate — the turn_shape data is a secondary output of the same run, not a change to the advance measurement.

---

## Bottom Line

**The run is trustworthy.** All five A×C findings (P1 discriminability, P2-A content-type confound, P2-B unmeasured failure modes, P2-C unfalsifiable claim, P3 priority question) are discharged by the A×B design. The remaining-naming disambiguation is arithmetically correct. The carry-side measures are the right operationalization of "does the carry-side hold under depth." Validator.py is specific enough for judge recognition. No A×B-specific embedded conclusion blinds the design to a depth-induced limit it could produce.

Two caveats carry forward into results interpretation:
1. **(P2)** Results landing in 8-12/15 are in the ambiguous zone of the holds/degradation boundary; state this explicitly in the results record.
2. **(P3)** The state-injection method does not measure live-session trajectory degradation; flag this for BUILD-phase validation.

Neither requires a re-run.

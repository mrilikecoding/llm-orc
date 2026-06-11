# Argument Audit Report

**Audited document:** `docs/agentic-serving/decisions/adr-040-deterministic-completeness-gate.md`
**Source material:**
- `docs/agentic-serving/decisions/adr-037-session-termination-two-call-composition.md`
- `docs/agentic-serving/decisions/adr-038-remaining-work-anchor.md`
- `docs/agentic-serving/essays/research-logs/cycle-7-spike-sigma-premature-finish.md`
- `docs/agentic-serving/decisions/adr-035-client-tool-deliverable-form-contract.md`
- `scratch/spike-sigma-premature-finish/j3_deterministic/SUMMARY.tsv`
- `scratch/spike-sigma-premature-finish/j3_diag/` (RESOLUTION discharge artifacts)
**Genre:** ADR
**Date:** 2026-06-10
**Round:** R2 (re-audit after revision; verifies R1 findings closed, checks for new issues)

---

## Section 1: Argument Audit

### Summary

- **Genre:** ADR
- **Argument chains mapped:** 5 (invariance claim, ADR-038 non-contradiction, no-Conditional-Acceptance, persist-once hardening, scope boundary)
- **Issues found:** 1 (0 P1, 0 P2, 1 P3)
- **Pyramid coverage map:** N/A
- **Expansion-fidelity findings:** N/A

---

### R1 Finding Closure Verification

Each R1 finding is assessed against the revised ADR text. All source-material cross-references were verified against the spike log, SUMMARY.tsv, and serve logs.

**P2-A (invariance language softened to "no improvement at n≤5")**
CLOSED. §Context (lines ~27–33) now reads "no improvement detectable at these n" and explicitly states this is not a population-level invariance. §Rejected alternatives (Frontier judge entry) uses "no improvement over the cheap local judge's 1/5 at n=5." §Empirical grounding names the result as supporting the information-limit diagnosis "at the measured n." The mechanism argument carries the weight; the rate language is appropriately bounded. The revision is accurate, not over-hedged.

**P2-B (full discharge arc 1/2 → 2/2 surfaced)**
CLOSED. §Empirical grounding now explicitly names the complete run record: "pre-persist-once... two J-3 runs, **1/2 converged** (run 1 COMPLETE 5 files; run 2 EARLY_COMPLETE at 1 file)" followed by the 2026-06-10 diagnostic refuting the no-op hypothesis, then "Post-persist-once... **2/2 converged**." The SUMMARY.tsv confirms: run=1 cls=COMPLETE files=5, run=2 cls=EARLY_COMPLETE files=1. The discharge claim is now traceable to the complete evidence arc.

**P2-C (run-2 attribution softened to surviving-but-unconfirmed hypothesis)**
CLOSED. §Rejected alternatives (Re-derive each turn entry) now reads: "This path is the surviving hypothesis for the 2026-06-09 run-2 false-COMPLETE after the truncation hypothesis was tested (the diagnostic) and did not reproduce; it is not a confirmed attribution, and the run-2 mechanism remains unknown." This accurately matches the spike log's §RESOLUTION ("The only mechanism left for it is a transient `_user_task` truncation... the run-2 false-COMPLETE from 2026-06-09 did NOT reproduce"). The revision is precise: it names the hypothesis without asserting confirmation.

**P2-D (deterministic-anchor advance-rate lineage noted)**
CLOSED. §Consequences (Negative) now explicitly states: "ADR-038's Spike ρ validated the judge-statement anchor at 19/20 advance; the deterministic anchor was not separately validated for advance rate. It rode the two discharge sessions' monotonic convergence (1→2→3→4→5 each, no churn), which validates it implicitly, but the lineage is worth stating: the 19/20 figure belongs to the judge-statement anchor ADR-038 measured, not to this deterministic anchor." The live serve logs (serve2.log, serve3.log) confirm monotonic convergence across both runs.

**P2-F1 (named-file scope representativeness stated)**
CLOSED. §Consequences (Negative) now includes: "The cycle's north-star task shape (multi-file coding tasks that name their files explicitly) is the named-file shape, so the restriction is expected to cover the majority of the target deployment envelope, but that coverage fraction is an untested assumption: how much of a real deployment's task space names files is not characterized here." This is appropriately hedged without over-hedging the practical relevance.

**P2-F2 (live nominal vs unit-test truncation-path validation distinguished)**
CLOSED. §Empirical grounding now contains a dedicated paragraph titled "What the live runs validate vs what only the unit test validates." It explicitly states: "The live runs validate *nominal* convergence... They do **not** exercise persist-once's defended-against failure... persist-once's truncation path is validated only by a unit test that *simulates* the compaction." The open BUILD-watch item (real-client compaction may differ from the unit test model) is named. The serve logs confirm `requested=[5]` throughout both runs, making this distinction accurate.

**P3-B (dispatch-savings labeled as implementation read)**
CLOSED. §Consequences (Positive) now reads: "One fewer model dispatch per trailing turn on the common coding-task shape (no judgment-seat call), and no frontier judge is needed for completeness, which keeps the cheap-local economics intact **(an implementation read; not separately measured)**." Matches the Provenance check's honesty level.

**P3-F1 ("cannot occur" qualified with "where extraction succeeds")**
CLOSED. §Consequences (Positive) now reads: "The false-COMPLETE cannot occur for named-file tasks **where the extraction succeeds**: the verdict is a set comparison, not a model inference. The extraction heuristic is the remaining stochastic element (its failure modes, over-extraction loops and under-extraction early finish, are scoped to the refutable FC below)." Accurate qualification.

**P3-A (R1 run-mapping error — no ADR change needed)**
Confirmed: R1's reading was incorrect. R1 claimed run2/ws2 is the 8b production run; the actual mapping is ws2 = 14b confirming run (Run A, `args.from` keyword collision) and ws3 = 8b production run (Run B, trailing prose paragraph). The ADR's attribution of "trailing prose paragraph" to the 8b run was and remains accurate. Verified against ws2/cli.py (`args.from` present) and ws3/cli.py (trailing prose paragraph present). No ADR change was made or needed.

---

### P1 — Must Fix

No P1 findings. All R1 P1-class issues have been resolved. No new P1 issues detected.

---

### P2 — Should Fix

No new P2 findings. All R1 P2 findings are closed. The revisions are accurate and well-calibrated — none over-corrected.

One near-miss worth recording: the new P2-F2 paragraph in §Empirical grounding states "If the real client compacts messages differently than the unit test models, persist-once's behavior under live compaction is unverified, an open BUILD-watch item." This is the right level of epistemic honesty. It does not require further qualification.

---

### P3 — Consider

**P3-R2-A: The provenance check's line-wrap leaves a minor readability artifact.**

- **Location:** §Provenance check, first paragraph (~line 254)
- **Minor issue:** The Provenance check's opening paragraph ends mid-clause with a line break after "and the no-improvement-across-judges measurements (the σ research log;" before resuming "cheap 1/5, J-1 1/4, frontier 1/5)". This is a typographic artifact from editing, not a logical issue. It does not affect argument integrity.
- **Recommendation:** Cosmetic fix only — reflow the sentence.

---

## Section 2: Framing Audit

The framing audit compares the revised ADR's content selection against what the source material made available. The R1 framing audit surfaced three issues (P2-F1, P2-F2, P3-F1); all three are now closed. This section checks whether the revisions introduced any new framing gaps or over-corrections.

### Question 1: What alternative framings did the evidence support?

**Alternative framing A: J-3 as a scope-restricted workaround** — the R1 concern was that the scope restriction was not explicitly bounded. The revision's P2-F1 closure adds the "untested assumption" note. The framing now accurately distinguishes what is claimed (named-file coverage of the north-star shape) from what is untested (fraction of real deployment task space). No residual gap.

**Alternative framing B: persist-once as a design tradeoff** — the R1 concern was that "hardening" language minimized the tradeoff character. The ADR's §Consequences (Negative) was already present in R1 and names the turn-1 mis-extraction stickiness trade. The revision adds the P2-C closure noting the run-2 mechanism is unknown. This is accurate. The "hardening" framing in §Context is now accompanied by the explicit note that it "closes a transient-truncation path structurally, not a fix for an observed break." The framing is honest about the tradeoff character without overstating it.

### Question 2: What truths were available but not featured?

**Pre-RESOLUTION 1/2 run record** — now surfaced in §Empirical grounding (P2-B closure). The complete arc is visible: pre-fix 1/2, diagnostic refutation, persist-once applied on determinism principle, post-fix 2/2.

**Spike's sub-n arms** — the spike log records that the pre-registered n=10 was not reached (J-1 ran to n=4, baseline and frontier to n=5). The revised ADR handles this via "at n≤5, below Spike σ's pre-registered n=10 (the spike did not reach it; it pivoted to J-3 once the judge was diagnosed as the wrong fix)." This is accurate and appropriately bounded. The R1 concern about "available but not featured" for the sub-n fact is now addressed.

**14b vs 8b diagnostic distinction** — the run-mapping is now traceable from the Empirical grounding section (the 14b run and the production-config run are named separately, the latter being the 14b seat + 8b coder configuration). The diagnostic session (qwen3:14b only) is distinguished from the production configuration.

### Question 3: What would change if the dominant framing were inverted?

Under the inverted framing (J-3 as a scope-restricted patch; persist-once introducing sticky extraction errors in exchange for closing an unconfirmed truncation path), the claims that would change are:

- "The false-COMPLETE cannot occur for named-file tasks where the extraction succeeds" — the qualification "where extraction succeeds" now appears in the text (P3-F1 closure). The inverted framing's concern is addressed.
- The persist-once hardening narrative — the "not a fix for an observed break" language now appears explicitly. The ADR no longer presents persist-once as pure hardening; it presents it as principled closure of a structural vulnerability, with honest acknowledgment that the vulnerability was never observed live.

The revised text handles the inverted framing's main challenges. No residual over-claim remains.

### Framing Issues

No new framing P1 or P2 findings. The R1 framing issues (P2-F1, P2-F2, P3-F1) are all closed. No over-corrections are present: the added hedges are accurate characterizations of what the evidence does and does not show, not gratuitous qualification that would understate well-grounded claims.

**P3 framing note (carry-forward of R1 framing context, now resolved):**

The R1 alternative framing B observation — that persist-once trades truncation risk for extraction-stickiness risk — is now addressed in the ADR's §Consequences (Negative): "A requested set that mis-extracts on turn 1 is sticky for the session. The trade is deliberate: stability against truncation beats per-turn re-derivation, and turn 1 is the most reliable extraction point. A turn-1 mis-extract is the cost of that choice." This explicit tradeoff framing is a genuine improvement over the R1 draft.

---

## Convergence-Saturation Signal (ADR-094)

**Convergence-Saturation Signal:** TRIGGERED

- Round number: R2
- P1 count this round: 0 (Section 1: 0; Section 2: 0)
- P2 count this round (new, non-carry-over): 0 (all R1 P2 findings closed; no new P2 findings surfaced)
- New framings or claim-scope expansions: none — the round surfaced no new warrants or claim-scope characterizations that R1's finding records did not name. The revisions addressed the R1 framings without introducing new ones.
- Recommendation: **STOP at this round.** All three conditions hold: P1 count = 0 (condition 1); new P2 count = 0 ≤ 1 (condition 2); no new framings (condition 3). The ADR is argument-sound, evidence-grounded, and its R1 documentation gaps are repaired. The one P3 finding (cosmetic line-wrap in the provenance check) does not affect readability or logical integrity at gate level.

*This is a re-audit dispatched per the re-audit-after-revision rule, evaluating whether R1 findings are closed. Per ADR-094, re-audits dispatched for this purpose run unconditionally. A Convergence-Saturation Signal verdict is appended here because R2 is also the standard-sequence continuation of the primary audit; it is not a single-purpose re-audit of a specific prior P1 repair.*

# Argument Audit Report

**Audited document:** docs/agentic-serving/decisions/adr-041-destination-validity-gate.md
**Source material:** docs/agentic-serving/essays/research-logs/cycle-7-spike-pi-form-adequacy-gate.md; docs/agenting-serving/decisions/adr-035-client-tool-deliverable-form-contract.md (§Decision 4, §Conditional Acceptance, §Negative); docs/agentic-serving/decisions/adr-040-deterministic-completeness-gate.md; docs/agentic-serving/decisions/adr-034-client-tool-action-terminal-artifact-bridge.md; docs/agentic-serving/decisions/adr-014-calibration-gate-trajectory-level-extension.md
**Genre:** ADR
**Date:** 2026-06-11

---

## Section 1: Argument Audit

### Summary

- **Genre:** ADR
- **Argument chains mapped:** 7 (protection discharge, recovery mechanism, ADR-040 non-overlap, protects-but-does-not-recover reading, coder-tier redirection, heuristic alternatives rejection, unbounded-retry rejection)
- **Issues found:** 7 (P1: 2, P2: 3, P3: 2)
- **Pyramid coverage map:** N/A
- **Expansion-fidelity findings:** N/A

---

### P1 — Must Fix

**P1-1. "Protection is structural and discharged" overclaims the install status.**

- **Location:** Status block; §"What this discharges, and what stays conditional"; §Consequences Positive
- **Claim:** "The **protection** guarantee (no invalid deliverable reaches the client) is *structural and discharged*"; "Workspace protection is now structural."
- **Evidence gap:** The spike log is unambiguous that the gate is env-gated and not yet de-gated: the `LLMORC_SPIKE_PI_GATE=parse` env flag gates the entire mechanism, and the §"Conditional Acceptance" block in the ADR itself lists "De-gate-and-install at the FormGate seam (drop `LLMORC_SPIKE_PI_GATE`, thread `destination_path`, install the gate as the bridge's `form_gate`)" as a pending BUILD task. The gate is *proven* structural in principle and *validated* on a live arm, but it is not yet *installed* in the production code path — it is still a spike harness activated by an env flag. "Discharged" implies the obligation is closed; the obligation is closed in design but open in code. The live evidence is n=5 sessions on the exact σ/η task shape, not across deliverable types, tier configurations, or clients other than OpenCode 1.15.5.

  The distinction matters because a future PLAY run against the uninstalled code path would not have the protection. The ADR's own Conditional Acceptance block acknowledges this, but the Status declaration and the Consequences §Positive do not. That is an internal inconsistency: the same document simultaneously says the protection is "discharged" (Status) and lists the de-gate-and-install as a BUILD task (Conditional Acceptance).

- **Recommendation:** Revise the Status block and Consequences §Positive to distinguish between the *design* discharge (the honesty gap in ADR-035's Conditional Acceptance is closed in principle — the gate's structure is proven not to rely on model compliance for protection) and the *install* status (the production code path does not yet run the gate; it is an env-gated spike artifact). A precise formulation: "The **protection design** is structural and discharged — the gate's architecture is model-compliance-independent and validated live. The **gate installation** is Conditional Acceptance (pending de-gate, `destination_path` thread, and bridge wiring as BUILD tasks)." This aligns with the Conditional Acceptance block's own text and removes the internal contradiction.

---

**P1-2. The ADR-040 non-overlap argument has an unexamined failure mode.**

- **Location:** §"Why server-side recovery, not client-side re-delegation"; §Relationship to prior ADRs (ADR-040 entry)
- **Claim:** "An invalid deliverable triggers a refusal that ends the loop before ADR-040 could fire. ADR-041 records that ordering explicitly." The implication is that the two gatesoperate on non-overlapping populations: ADR-040 handles missing deliverables, ADR-041 handles invalid ones, and the two never compete or interfere because ADR-041's refusal-as-`stop` terminates the session before ADR-040 gets a turn.
- **Evidence gap:** The ordering argument is sound for the *cap-exhausted* case (refusal-as-`stop` genuinely ends the session). But it is not sound for the *server-side-recovery* case that ADR-041 itself introduces. When the server-side re-dispatch loop *recovers* a file (runs 1/2/5 in the n=5×2), the loop continues; the Loop Driver then proceeds and eventually emits a completeness turn. On that completeness turn, both gates are active in sequence. The ADR does not analyze what happens if recovery produces a file that the completeness gate's requested-set comparison would not have previously included, or — more concretely — what happens to ADR-040's persist-once semantics after a server-side re-dispatch that reuses the delegation path (the ADR says re-dispatch "reuses the delegation path, not a full re-decision, so the action is not double-recorded"). The spike log §"Recovery built + validated live" confirms the re-dispatch avoids double-recording, and the n=5×2 runs with recovery do converge. But the ADR does not explicitly trace the recovered case through ADR-040's completeness check — it only argues non-overlap for the failure case. This is a gap, not a contradiction, but an important one given both gates are now live simultaneously on the same serving path.
- **Recommendation:** Extend the ADR-040 §Relationship entry to address the recovered-case interaction: confirm (with a sentence citing the spike's convergence evidence) that a server-side-recovered file lands in the Session Action Record in a way that satisfies ADR-040's completeness diff, or note it as a BUILD verification item. The current text's non-overlap claim is accurate for cap-exhaustion but incomplete for the recovery path.

---

### P2 — Should Fix

**P2-1. The "protects-but-does-not-recover" decision-rule application is selectively honest but omits the pre-registered pass condition.**

- **Location:** §"What the spike established" (live arm); §"What this discharges, and what stays conditional"
- **Claim:** The ADR adopts the *converged-and-all-valid* reading (B=3/5 vs A=2/5, margin +1/5, convergence margin not met) over the *vacuous all-produced-valid* reading (B=5/5). It calls this "the pre-registered **protects-but-does-not-recover** outcome."
- **Issue:** The spike log §LIVE-ARM RESULT states the pre-registered decision rule literally: "recovers iff B-live ≥ 3/5 AND (B-live − A-live) ≥ 2/5 AND re-dispatch control > 0." The vacuous-B=5/5 reading satisfies all three criteria literally: B-live = 5/5 ≥ 3/5, B−A = 3/5 ≥ 2/5, re-dispatch control = 3/5 > 0 → **"recovers" by the pre-registered rule.** The spike log itself acknowledges this, flags it as "partly vacuous," and switches to the spirit-of-the-rule converged-and-all-valid reading. The ADR re-states the converged reading as "the pre-registered protects-but-does-not-recover outcome" — but the pre-registered rule, applied literally, returned "recovers," not "protects-but-does-not-recover." The ADR is adopting a post-hoc reinterpretation that the spike log explicitly notes and endorses in spirit, but the ADR presents it as though the pre-registered rule unambiguously returned the protects-but-does-not-recover verdict, which it did not.

  This is not a material error in the ADR's conclusion — the converged-and-all-valid reading is clearly the right one, and the spike log's explanation is sound. But the ADR should not claim the pre-registered rule "returned" this verdict when the spike log shows it required a deliberate interpretive choice.
- **Recommendation:** Add a sentence in the live-arm description acknowledging that the pre-registered decision rule, applied literally to "all-produced-valid," would return "recovers" (B=5/5), but that the run-3/4 all-valid score is vacuous (they ended at 3 and 1 files), and the converged-and-all-valid reading is adopted as the honest application of the rule's spirit. This is exactly what the spike log says; the ADR should reproduce the transparency rather than elide the interpretive step.

---

**P2-2. Arm E's "coder-tier escalation closes the residual" claim is stronger than n=6 supports.**

- **Location:** §"What the spike established" (Arm E); §Decision 5; §"Why the lever is the coder tier"
- **Claim:** "Coder-tier escalation closes the residual — MiniMax produced valid `cli.py` 6/6, and is fast (~1.2s ping vs ~10s local)." The escalation ladder explicitly places "cheap→escalated coder tier (ADR-014 Calibration Gate) closes persistent bleeds" as a committed rung.
- **Evidence gap:** The Arm E evidence is n=6 isolated re-dispatches of a single file (`cli.py` for the temperature library task), all against the same generation target, on a focused probe rather than a full session. This is not a wired-into-the-live-loop test. The spike log itself says "Arm E proved the lever in isolation, n=6; not wired into the live loop." The claim "closes the residual" implies a session-level convergence guarantee. What Arm E proves is that MiniMax produces parse-valid `cli.py` in isolation on 6/6 attempts at n=6; it does not prove that a coder-tier escalation wired into the serving loop would converge a full 5-file session with persistent bleeds. There is also a gap in breadth: the Arm E probe was for `cli.py` specifically, which is the hardest file; it was not tested on the full file set, or with a wired session that exercises the escalation trigger and recovery interaction.

  "Closes the residual" as a session-level guarantee is too strong for an isolated n=6 single-file probe. "Provides strong evidence that coder-tier escalation can close the residual" is more accurate.
- **Recommendation:** In §Decision 5 and §"Why the lever is the coder tier," replace "closes the residual" with a scoped version: "Arm E confirms the coder-tier lever on the hardest file in isolation (6/6 valid, n=6); session-level convergence under wired coder-tier escalation is pending BUILD validation." The Conditional Acceptance block already lists this; the issue is the stronger claim elsewhere in the ADR.

---

**P2-3. The degradation-confound dismissal for the live arm's protection rate is incompletely hedged.**

- **Location:** §"What the spike established" (live arm, third bullet on protection); §"What this discharges" (first bullet)
- **Claim:** The ADR says Arm E "dismissed" the degradation confound, and the live arm's "0 invalid files across 5 gated sessions" supports that "protection is structural and confirmed live." The dismissal is for the *convergence rate* (3/5 vs a possible higher rate on fresh ollama), not for protection.
- **Evidence gap:** The spike log §LIVE-ARM RESULT says: "The confound is judged low but real; it bears on the precise recovery rate (3/5 might be higher on fresh ollama), not on the protection finding (which is degradation-independent — the gate catches whatever the coder produces)." The ADR repeats this correctly in §Conditional Acceptance ("Recovery-rate sharpening on fresh ollama — the live 3/5 carries a low-but-real degradation confound on the precise rate, dismissed for protection"). But the protection-discharge claim in the Status block and §Consequences Positive does not carry this qualifier. The degradation confound is dismissed for protection specifically because the gate is deterministic regardless of what the coder produces — but the ADR does not make this reasoning explicit in the locations where the protection claim is strongest. A reader encountering the Status block first has no visibility into the degradation caveat.
- **Recommendation:** Add one qualifying phrase to the Status block's protection-discharge claim that names the degradation-independence reason: "the gate is deterministic over whatever bytes the coder produces, making protection degradation-independent (Arm E: dismissed for protection, real but low for recovery rate)." This is already present in the Conditional Acceptance block; it should also appear where the protection claim is headline-asserted.

---

### P3 — Consider

**P3-1. The scope of "both seams covered" is understated in a way that could mislead.**

- **Location:** §Consequences Positive, second bullet; §"What the spike established"
- **Claim:** "Both seams covered by one mechanism. Trailing prose and within-file wrong-language both break `ast.parse`."
- **Issue:** The corpus arm (Forks 1/2/4) confirmed this for the 12-item labeled corpus with n=12 items. The live arm confirmed that the σ form bleed (C1, trailing prose) is caught live. The η intent-divergence (C4, wrong-language in a `.py`) was confirmed in the corpus arm but was not independently exercised in the live n=5×2 arm (no run in the live arm produced wrong-language in a `.py`; the corpus item C4 was the η-captured JS content re-pathed). The ADR's provenance check attributes the cross-seam unification to "Spike π corpus Fork 2 (driver — miss-set ∅, FP ∅)" — but that is corpus-only evidence for the wrong-language seam, not live-arm evidence. The seam is deterministically checkable and the corpus evidence is valid, but the live arm confirmed only one of the two seams under real-trajectory conditions.
- **Recommendation:** In the "both seams covered" bullet, note that the live arm confirmed the form-bleed seam under trajectory conditions; the wrong-language seam is confirmed by the corpus arm (deterministic, n=1 real case re-pathed). This is a precision note, not a material weakness, but the current phrasing implies equal live-arm evidence for both.

---

**P3-2. The `destination_path` extension is described as additive but the precise interface contract is not specified.**

- **Location:** §Decision 2; §Decision 3; §Consequences Neutral; §"Relationship to ADR-035"
- **Claim:** "The `destination_path` thread is the only structural surface change — small, additive, and the BUILD seed the env-gated spike code already prototypes."
- **Issue:** The ADR introduces a new field (`destination_path`) that must be carried alongside the existing `destination_tool` at the FormGate seam (FC-57) and the bridge `marshal` signature. The ADR correctly scopes this as a BUILD Design Amendment and says it is "additive at the bridge `marshal` signature and the FormGate contract." But it does not specify what `destination_path` carries — is it the full path (`src/foo/bar.py`) or just the extension (`.py`)? The spike code gates on extension (the gate keys on `*.py` → `ast.parse`), but the field is called `destination_path`, implying the full path. If BUILD uses a different interpretation, the gate's extension-keying logic needs re-derivation. This is a small ambiguity but it is the only new interface this ADR introduces.
- **Recommendation:** Add a one-sentence clarification to §Decision 2 that `destination_path` carries the full destination path (not just the extension), and that the gate derives the extension from it at validation time. This matches the spike code's behavior and removes an interpretive fork for BUILD.

---

## Section 2: Framing Audit

### Question 1: What alternative framings did the evidence support?

**Alternative framing A: The gate as a session-quality signal rather than a protection mechanism.**

The spike log produces a result (B=3/5 converged-and-all-valid vs A=2/5) that could support framing the gate primarily as a *session-quality signal that enables escalation routing* rather than as a *protection mechanism*. Under this framing, the primary value is not that zero invalid files reach the client (the baseline already sends 0 invalid files in 2/5 sessions via non-bleed completions), but that the gate converts a *silent* session-quality failure (SyntaxError in the workspace) into an *explicit* routing signal that ADR-014's Calibration Gate can act on. The evidence supports this: runs 3/4 ended short (session-quality failure), but the failure mode changed from "corrupted workspace" to "clean short session that signals the need for escalation." A reader who believes the workspace-protection framing carries less incremental value (since ADR-034's execution model already gives the client a permission gate and diff) might find this routing-signal framing a better fit for the ADR's system-level contribution.

- What would the reader need to believe for this framing to be right? That the client's permission gate and diff (ADR-034's execution model) provide meaningful protection against silent SyntaxError delivery in practice — i.e., that a developer running OpenCode would catch a SyntaxError from the diff before applying it. This is plausible but the ADR does not challenge it.
- What the ADR's argument looks like under this framing: the gate is a *completeness signal* — it converts cheap-tier form failures into explicit routing signals, enabling the ADR-014 Calibration Gate to escalate the coder tier. Protection is secondary; routing-signal clarity is primary.

The ADR does not consider this framing. It is worth acknowledging, because ADR-035's "Why a hard form-guarantee is neither available nor required" section explicitly argues that the client's execution model already provides a rejection surface for wrong-form deliverables. That argument, if accepted, weakens the workspace-protection framing the ADR leads with.

**Alternative framing B: Addressing the grounding gap for the pre-registered seat-lever, rather than confirming the gate.**

The spike was pre-registered with the frontier seat (ADR-033 §6b escalation-3) as the convergence lever. The live arm redirected this to the coder tier. An alternative reading of the spike's central finding is that the primary output is the *lever redirection* (coder, not seat), and the gate is the mechanism that makes the lever's location visible. Under this framing, the ADR's structural contribution is correcting a mislabeled escalation path in the prior art, and the gate is the diagnostic tool that produced the correction. This framing would put §Arm E closer to the center of the ADR's argument and treat the gate more as a prerequisite for lever identification than as the primary deliverable.

The ADR does address the lever redirection directly, but treats it as a secondary finding ("why the lever is the coder tier" subsection) rather than as the spike's most consequential output for the escalation architecture. This is a framing choice, not an error.

---

### Question 2: What truths were available but not featured?

**Available but underrepresented: the corpus arm's narrow scope (12 items, single task type).**

The corpus arm (Forks 1/2/4) is deterministic and free, and the ADR cites it confidently ("catch 5/5, FP 0"). The spike log itself adds an honest caveat (§"CORPUS RESULT honest caveats"): "The corpus result confirms the gate's *logic* and the partition, not that real-world failures are all parse-detectable. The live arm (Fork 3) is the real test." The corpus is 12 items, all derived from the σ/η temperature-library task context. C3, C6b, C6c, C7b, and C9 are synthesized to category. The ADR names the synthesized items in the Provenance check ("12-item labeled set") but does not carry the caveat about corpus representativeness that the spike log explicitly flags.

The relevance: the ADR's protection-discharge argument rests partly on "corpus (catch 5/5, FP 0) and live arm (0 invalid across 5 gated sessions)." If the corpus items are not representative of the full population of real-world form failures (e.g., partial fence-with-content, encoding edge cases, mixed-language outputs), the corpus evidence provides weaker support than the ADR's summary implies. The live arm is the real test for the protection claim, and the ADR does use it as the primary evidence — but the corpus framing in the Status and §Consequences blocks gives the corpus result co-equal billing with the live arm.

**Available but omitted: the spike log's "recovery-loop trace" data (P2-C).**

The methods review required a per-session record of what the Loop Driver delegates on the turn immediately after a gate refusal — to separate "coder re-failed the refused file" from "driver chose a different action." The spike log §LIVE-ARM RESULT table shows recovery episodes but does not report a separate recovery-loop trace by name. The ADR inherits this gap: it describes runs 3/4 as "cli.py exhausted" and "test_converters.py exhausted" but does not state what the driver delegated *after* the cap-exhaustion stop. Since the sessions ended (refusal-as-`stop`), there is no "after" to trace for the failed cases. But for the recovered runs (1/2/5), the ADR does not report whether the driver re-targeted the same destination or chose a different action before the recovery path took over. This would strengthen the causal-isolation argument (does the server-side re-dispatch, not the driver's re-targeting, produce the convergence?).

The ADR describes re-dispatch as reusing the delegation path and "not a full re-decision so the action is not double-recorded" — but the causal isolation (driver re-targeting vs server-side recovery) is not verified in the report.

**Available but not foregrounded: the baseline's 2/5 converged-all-valid rate is actually reasonable.**

Cell A (baseline, gate off) converged 5 files in all 5 sessions, with invalid files reaching the client in 3/5. So the baseline managed 2/5 fully-valid completions without any gate. The ADR presents this as the failure case ("the baseline shipped invalid files in 3/5"). This is accurate, but the fact that 2/5 sessions completed with zero invalid files under pass-through is not discussed. That 2/5 baseline matters for interpreting the +1/5 converged-and-all-valid improvement (B=3/5 vs A=2/5): the gate adds one fully-valid session over baseline. A framing audit should flag that the ADR's protection-and-convergence story is built on a baseline that is not zero — the gate is an improvement over a non-trivial baseline, not a fix for a universal failure.

---

### Question 3: What would change if the dominant framing were inverted?

The ADR's dominant framing: the gate makes protection structural by catching all deterministically-checkable invalid deliverables, enabling the framework to close ADR-035's honest gap.

**Inverted framing:** the gate converts a *client-visible, rejectable* problem (a SyntaxError-carrying diff that the client could reject via its permission gate) into a *server-side, invisible* problem (a session that ends short). Under the inverted framing, the gate does not improve the user's situation — it changes where the failure surfaces (workspace vs. short session) without changing whether the session succeeds.

Under this framing:
- **Protection claim becomes weaker:** ADR-035 §Decision 4 (as originally written) argued that "a wrong-form deliverable surfaces as a rejectable diff" — the client's execution model is the protection. If that argument is accepted, the gate converts a visible rejection opportunity into an invisible short session, which is arguably a worse user experience.
- **The convergence claim becomes the primary benefit:** the gate's real contribution, under the inverted framing, is the server-side recovery path (runs 1/2/5 self-healed without the client ever seeing a refusal or a short session), not the protection guarantee. Under pass-through, the client sees a broken file; under gate+recovery, the client sees nothing (the recovery is invisible). Under gate+cap-exhaustion, the client sees a short session. The net experience is: recovery cases improve, cap-exhaustion cases neither improve nor worsen (a silent short session vs. a broken file are comparable user experiences), and no case is actively worse.
- **The ADR would need to address:** whether a short session (cap-exhausted) is better or worse than a broken-file session (pass-through) from the user's perspective, since this is what "protection" means experientially. The PLAY designation for experiential validation is appropriate here, but the ADR does not state what the experiential hypothesis is.

The ADR does not engage with the "short session vs. broken file" experiential trade-off at all, and that is the most important unaddressed consideration the inverted framing surfaces.

---

### Framing Issues

**P2-F1. The workspace-protection framing is not reconciled against ADR-035's "hard guarantee neither available nor required" argument.**

- **Location:** §Consequences Positive; Status block
- **Issue:** ADR-035 §Decision 4's "Why a hard form-guarantee is neither available nor required" explicitly argues that the client's permission gate and diff make wrong-form deliverables visible and rejectable, which is why the form contract was acceptable as a lighter mechanism. ADR-041 now claims that wrong-form deliverables reaching the client is a problem serious enough to warrant a committed deterministic gate grounded before PLAY. These two positions are in tension: either the client's permission gate makes wrong-form deliverables recoverable (ADR-035's argument for why a hard guarantee is not required), or wrong-form deliverables reaching the client are a workspace-safety problem that requires structural prevention (ADR-041's argument). The ADR does not resolve this tension — it simply upgrades the protection without revisiting whether the ADR-035 argument for why protection is "not required" was correct, or whether the practitioner's direction to tackle both seams before PLAY implicitly overrides it.
- **Recommendation:** Add a paragraph in the §Context or §"What this discharges" section that explicitly acknowledges the ADR-035 tension and resolves it: either (a) ADR-035's "not required" argument was conditional on the detection gate being available as an escalation (which it now is), so the protection claim is a clarification of what "available" means once the gate exists; or (b) the practitioner's directive elevated the protection priority above the ADR-035 cost argument. Either resolution is defensible; the current text leaves the tension implicit.

**P2-F2. The escalation ladder's PLAY designation is not qualified by the experiential trade-off question.**

- **Location:** §Decision 5; §Consequences Negative
- **Issue:** The ADR places the "parses-but-wrong" semantic residual in PLAY as the final rung of the escalation ladder. This is correct for the *detection* problem. But the inverted framing above surfaces a separate PLAY question: how does a short session (cap-exhausted protection) compare to a broken-file session (pass-through) from the user's perspective? That is an experiential question, not a detection question, and it belongs in PLAY too. The ADR's PLAY designation covers only the semantic residual, not the experiential trade-off of the gate's visible failure mode (short session vs. broken file).
- **Recommendation:** In §Decision 5 or §Consequences Negative, note that PLAY also validates the gate's experiential failure mode: a cap-exhausted short session is the user-visible outcome of persistent cheap-coder bleeds, and PLAY should observe whether that experience is acceptable compared to the pre-gate experience of a broken-file session. FC-51 `TurnDecision` instrumentation is already mentioned as distinguishing wrong-form turns; extend that observation target to include user-perceived session outcomes on exhausted vs. pass-through trajectories.

**P3-F1. The "semantic residual is PLAY territory, unchanged" framing understates the semantic seam's scope change.**

- **Location:** §Decision 5 final paragraph; §Context final paragraph of "What the spike established"
- **Issue:** The ADR says "the residual after all three [rungs] is the irreducibly-semantic 'parses-but-wrong' slice — handed to PLAY, unchanged." The "unchanged" qualifier is not quite right: ADR-035 left the adequacy seam *entirely* to PLAY as a disclaimed third seam; ADR-041 now takes the deterministically-checkable slice of that seam (C4, C5) into the gate's scope. What is left to PLAY is smaller than what was left to PLAY before. The "unchanged" language could be read as "PLAY's scope is the same as before ADR-035 disclaimed it," which is imprecise. The semantic residual is the same *class* of problem (parses-but-wrong), but the scope of what PLAY must validate is narrower because the gate has already handled the parse-detectable slice.
- **Recommendation:** Replace "unchanged" with "narrowed to the parses-but-wrong slice; the deterministic-check portion of the adequacy seam is now handled by the gate."

---

## Convergence-Saturation Signal (ADR-094)

**Convergence-Saturation Signal:** NOT TRIGGERED

- Round number: R1 (first audit of ADR-041; no prior rounds on this artifact)
- P1 count this round: 2 (Section 1: P1-1, P1-2)
- P2 count this round (new, non-carry-over): 5 (Section 1: P2-1, P2-2, P2-3; Section 2: P2-F1, P2-F2)
- New framings or claim-scope expansions: Section 2 surfaces two framings not named in the ADR — (a) the gate-as-routing-signal framing vs. the gate-as-protection framing; (b) the short-session-vs-broken-file experiential trade-off question as an unaddressed PLAY designation scope. These are framings the ADR's source material made available but the ADR did not foreground.
- Recommendation: CONTINUE to R2 (P1 count = 2; P2 count = 5 new findings; two new framings surfaced)

*Single-purpose re-audits (dispatched per the re-audit-after-revision rule) omit this section. Form-change events reset the round-count baseline — the first audit on a new form is its R1.*

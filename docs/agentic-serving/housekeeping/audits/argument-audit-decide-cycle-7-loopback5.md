# Argument Audit Report — R1

**Audited document:** `docs/agentic-serving/decisions/adr-037-session-termination-two-call-composition.md`
**Partial-update header also audited:** `docs/agentic-serving/decisions/adr-036-delegation-decision-mechanism.md` (the `> Updated by ADR-037...` block and Status line)
**Source material read:**
- `docs/agentic-serving/essays/research-logs/cycle-7-spike-theta-termination-mechanism.md` (full)
- `docs/agentic-serving/essays/research-logs/cycle-7-spike-psi-delegation-rate.md` (lines 508–661, Spike ψ″ + Arm E4)
- `docs/agentic-serving/decisions/adr-033-layer-a-loop-driver-multi-turn-agentic-surface.md`
- `docs/agentic-serving/decisions/adr-035-client-tool-deliverable-form-contract.md`
- `docs/agentic-serving/decisions/adr-036-delegation-decision-mechanism.md`
- `docs/agentic-serving/domain-model.md` §Invariants (AS-3, AS-9 + scope-boundary annotation), §Methodology Vocabulary

**Genre:** ADR
**Date:** 2026-06-05

---

## Section 1: Argument Audit

### Summary

- **Genre:** ADR
- **Argument chains mapped:** 11 (candidate-collapse → two-call composition; AS-9 class claim for termination judgment; round-1/round-2 rate verification; composed-estimate labeling; geometric-decay residual; Form B adoption; implicit-variant rejection via composed estimate; FC set internal consistency; ADR-036 partial-update scope accuracy; Conditional Acceptance gating condition; three-layer scope claim)
- **Issues found:** 7 (0 P1, 4 P2, 3 P3)
- **Pyramid coverage map:** N/A
- **Expansion-fidelity findings:** N/A

---

### P1 — Must Fix

None.

---

### P2 — Should Fix

**P2-1 — The AS-9 class claim for the termination judgment extends beyond AS-9's validated scope without marking the extension**

- **Location:** §Decision "the verdict itself is model-rendered within a structurally-bounded role (AS-9's reliable class)"; §Decision ¶2 of Three-layer scope-of-claim "the framework instead bounds the role... into AS-9's reliable class"
- **Claim:** The termination judgment (call 1) is in AS-9's reliable class, meaning a structurally-bounded single-decision-shaped task produces reliable output on it.
- **Evidence gap:** AS-9's empirical basis is explicitly defined in the domain model as: produce a response from given context; produce a JSON plan from a given request — n=13 across four confabulation modes, all on the routing-planner and response-synthesizer roles. The AS-9 scope-boundary annotation (added 2026-06-01) explicitly states that per-turn agentic-loop-driving is NOT in AS-9's validated scope because it is a different role-shape. A termination judgment — "given an action digest and a user task, has the work been completed?" — is structurally closer to AS-9's validated tasks (produce a response from given context) than loop-driving is. The spike validates the judgment at 29/30 and 30/30, which is consistent with AS-9's reliable-class prediction. However, the domain model has not been updated to reflect that the termination judgment extends AS-9's validated scope to a new task surface (deliverable-accounting judgment). The ADR asserts the judgment is "in AS-9's reliable class" as if it falls within the prior validated basis, when it is a new task surface conditionally validated by Spike θ. The gap is the absence of a notation that the AS-9 extension to termination-judgment tasks is spike-validated but not yet annotated in the domain model.
- **Recommendation:** In the §Decision paragraph making the AS-9 class claim, add a parenthetical acknowledging the extension: e.g., "AS-9's reliable class — the termination judgment is a new task surface in that class, validated by Spike θ at 29/30; the domain model should be updated to record the extension." Alternatively, add a domain-model annotation to AS-9 at the same loop-back boundary that records the termination-judgment role as an additional conditionally-validated extension (analogous to the loop-driver axis-1 annotation). Without this, the ADR claims an invariant scope that the invariant's own text does not yet cover.

**P2-2 — The composed-mechanism estimate labels are correct in §Decision ¶5 (Scope) but the Consequences section uses the language loosely at one point**

- **Location:** §Consequences Positive, first bullet: "work-complete tails finish at ~0.9 (composed estimate)"
- **Claim:** The parenthetical "(composed estimate)" in the first Consequences bullet correctly labels the rate.
- **Evidence gap:** The first Consequences bullet reads "work-complete tails finish at ~0.9 (composed estimate) with returnable finish text, against the current never-terminates." The parenthetical is present, which is correct. However, the same paragraph in §Decision ¶5 (Scope of the validated claim) makes the labeling discipline explicit: "composed from independently-measured n=10 arms, not an end-to-end measurement." The Consequences section's parenthetical "(composed estimate)" without the "not an end-to-end measurement" elaboration is briefer than the §Decision ¶5 label. The Consequences section is the most practitioner-facing part of the ADR; a reader skimming to Consequences may not read the scope disclaimer in §Decision first.
- **Recommendation:** Expand the "(composed estimate)" label in the Consequences section to "(composed from independently-measured arms, not end-to-end)" to match the precision of the §Decision ¶5 language. This is a one-phrase change that prevents a skimming reader from treating the ~0.9 as a directly measured end-to-end rate.

**P2-3 — The geometric-decay residual claim's arithmetic precision depends on the 1/10 false-continue being a stable rate, but the n=10 basis for that rate is not named at the claim site**

- **Location:** §Consequences Positive, first bullet: "The residual 1/10 false-continue costs one extra revision turn and faces ~0.9 termination probability on the next trailing turn — geometric decay, bounded absolutely by the AS-3 cap."
- **Claim:** The false-continue residual decays geometrically: each successive trailing turn faces ~0.9 termination probability.
- **Evidence gap:** The §Decision Scope paragraph and the §Provenance check both correctly label the geometric-decay characterization as "arithmetic from measured rates, not a measured multi-turn property." However, the Consequences section presents the decay claim as a straightforward positive consequence without the "drafting-time synthesis" caveat that the provenance section applies. A reader of Consequences encounters a confident ~0.9 geometric-decay claim backed by n=10 (the single θ.3b arm) without knowing the n is that small. Additionally, the 1/10 false-continue rate is itself a θ.3b result: one false-continue in ten trials, exactly at the pre-registered ≤1/10 threshold. The geometric-decay computation treats this single n=10 observation as the base rate for a multi-turn calculation, but with n=10 the true rate could plausibly be 0–3/10. The ADR does not note this at the Consequences site.
- **Recommendation:** Add a brief qualifier in the Consequences positive bullet, referencing the §Provenance label: e.g., "geometric decay (arithmetic from the θ.3b n=10 rate, not a measured multi-turn property — see §Provenance)." This keeps the Consequences section honest about the calculation's basis without removing its informational value.

**P2-4 — The one-re-validation-covers-both-seats reading is labeled as synthesis in §Provenance but the FC (judgment-seat re-validation) does not name the condition under which this reading holds or fails**

- **Location:** §Decision ¶7 "Judgment-seat re-validation": "composing with ADR-036's profile-swap re-validation FC rather than duplicating it (one re-validation event covers both seats when they share a profile)"; §Fitness criteria FC (judgment-seat re-validation): "A judgment-seat profile change is accompanied by a recorded judgment-rate re-validation. Composes with ADR-036's profile-swap FC."
- **Claim:** When the judgment seat and action seat share a profile, one re-validation event covers both.
- **Evidence gap:** The §Provenance check labels this as "synthesis from the FC compositions; the practitioner can split the seats, which would make re-validation per-seat." The FC as written says only that a judgment-seat profile change requires re-validation and "Composes with ADR-036's profile-swap FC." A BUILD implementer reading the FC does not know: (a) what "composes" means operationally — does one re-validation event explicitly cover both rates, or must the re-validation instrument report both?, and (b) what the split-seat case looks like. The θ harness is named as the re-validation instrument, but the harness measures judgment rates, not delegation rates — a combined re-validation event would need to exercise both judgment arms (θ) and delegation arms (ψ), or the "one event covers both seats" claim needs to specify which properties are validated. The FC's "Composes with ADR-036's profile-swap FC" does not spell out the operational definition of the composed event.
- **Recommendation:** In §Decision ¶7, add a parenthetical clarifying the combined re-validation instrument: e.g., "a combined event runs both θ judgment arms and ψ′-style delegation arms on the new profile — if the seats are split, each seat's re-validation is independent." This makes the synthesis concrete before BUILD implements the FC.

---

### P3 — Consider

**P3-1 — The implicit-variant rejection's composed estimate arithmetic should be verified against the source arms**

- **Location:** §Rejected alternatives "Implicit judgment (call 1 = unguided dispatch)": "complete tails converge (E2 10/10) but mid-task delegation composes to ~0.54 (E4a 6/10 continue × E4b 9/10 delegate — composed estimate)"
- **Claim:** The implicit variant's mid-task delegation rate is ~0.54, composed from E4a 6/10 × E4b 9/10.
- **Evidence gap:** The Spike ψ″ Arm E4 results (research log lines 637–648) confirm: E4a finish 4/10, actions 6 inline write + 0 invoke_ensemble; E4b finish 0/10, actions 9 invoke_ensemble + 1 write. The E4a rate used in the composition is "6/10 continue" (i.e., 6/10 of mid-task tails continue at all under no guidance), and E4b is 9/10 delegate given guidance present. But E4a's 6/10 continue are all inline writes (invoke_ensemble = 0/10) — that is, E4a has zero delegation on the work-remaining tails, not 6/10 delegation after continuing. The ADR reads the E4a result as "6/10 continue × E4b 9/10 delegate" but what E4a measures is 6/10 inline-write (which is the north-star violation) + 4/10 premature-finish. The ~0.54 is a composed estimate of "model continues mid-task (6/10) AND then the E4b guidance would yield delegation (9/10)," which assumes the 6/10 E4a continuation would become delegation under call-2's guidance — but the ADR labels this a composed estimate, so the framing is fair. The arithmetic is correct as stated (0.6 × 0.9 = 0.54). The characterization "mid-task delegation composes to ~0.54" could be read as implying 54% of mid-task tails would delegate under the implicit variant, when the correct reading is that 54% would delegate if call 2 is the production E4b form — which is the same call 2 the adopted mechanism uses. This subtlety is present in §Decision ¶5's labeling but not at the rejection site.
- **Recommendation:** Clarify the rejection sentence to read "mid-task delegation composes to ~0.54 (E4a 6/10 continue × E4b 9/10 delegate — composed estimate; E4a's 6/10 are all inline writes, not delegations, so the implicit variant drops to 0 delegation on call 1)" to prevent the ~0.54 from being misread as the implicit variant's standalone delegation rate.

**P3-2 — The Form A-enriched fallback designation does not note that the fallback requires carrying the 28k-char client prompt on every judgment call**

- **Location:** §Rejected alternatives "In-session judgment form (Form A-enriched)": "Recorded as the measured fallback form: if BUILD surfaces a bare-form failure the spike could not see, Form A-enriched is validated at the same evidence level and swaps in at the composition point."
- **Claim:** Form A-enriched is the validated fallback.
- **Evidence gap:** The cost difference between Form A and Form B is characterized in §Decision ¶5 and §Consequences Negative (Form B's ~1–2k bounded context vs. Form A's ~28–30k growing with session depth), but the §Rejected alternatives entry for Form A does not note that swapping to the fallback reintroduces the cost that caused Form B to be preferred. A BUILD implementer reading the fallback note might swap to Form A without flagging the latency and cost regression. The fallback is sound on accuracy grounds; the cost consequence should appear at the fallback site.
- **Recommendation:** Add a brief note in the Form A-enriched rejection entry: "the fallback reintroduces the client-prompt processing cost Form B avoided (~28k chars growing with session depth), so the swap should be flagged as a cost regression requiring the practitioner's review."

**P3-3 — ADR-036 partial-update header: the scope description is accurate but the "no framework system message" consequence narrowing could be stated more precisely**

- **Location:** ADR-036 `> Updated by ADR-037...` block: "The 'no framework system message' neutral consequence is narrowed to action-generation calls composed with the client conversation (ADR-037's bare judgment call carries a framework judge system message outside the client contest)."
- **Claim:** The neutral consequence in ADR-036 (no framework system message on the tool-driven surface) is narrowed to action-generation calls.
- **Evidence gap:** The partial-update header is accurate in scope. However, the narrowing phrase "outside the client contest" is architectural framing that appears in ADR-037's body but is not in ADR-036's original neutral consequence text. A reader scanning ADR-036 to understand how it has changed sees the partial-update header describe a consequence that ADR-036's original text (§Consequences Neutral: "The framework no longer emits a system message of its own on the tool-driven surface; the client's system prompt stands alone in the system region") expresses in present-tense unconditional form. The header is correct but the original Consequences text remains unconditionally worded. The header patches the interpretation but not the text.
- **Recommendation:** Consider whether ADR-036's §Consequences Neutral text itself should receive a narrow inline annotation (e.g., a parenthetical "(narrowed by ADR-037 to action-generation calls; judgment calls carry a framework system message)" at the relevant sentence). The partial-update header is the appropriate corpus mechanism for recording the update, but it sits above the Context section and a skimming reader might not see it before reading the Consequences. A small inline note in the Consequences Neutral paragraph of ADR-036 would close this gap. This is a P3 because the header is present and accurate; the risk is just skimming.

---

## Rate Verification

The following rates were verified against the Spike θ and Spike ψ″ research logs:

| Rate cited in ADR-037 | Source | Verified |
|-----------------------|--------|---------|
| E1 0/10 finish (work-complete + guidance) | ψ″ Arm E1 results table | Confirmed |
| E2 10/10 finish (work-complete, no guidance) | ψ″ Arm E2 results table | Confirmed |
| E3 1/10 finish (completion clause wording fix) | ψ″ Arm E3 results table | Confirmed |
| E4a 0/10 invoke_ensemble (mid-task, no guidance) | ψ″ Arm E4 results table | Confirmed (6/10 inline write + 4/10 finish; 0 invoke_ensemble) |
| E4b 9/10 delegate (mid-task, C3 guidance) | ψ″ Arm E4 results table | Confirmed (9 invoke_ensemble + 1 write) |
| Round-1: 19/60 total correct (information-starved) | θ Round-1 results + verdict text "19/60" | Confirmed: θ.1a 10/10 + θ.1b 0/10 + θ.2a 3/10 + θ.2b 10/10 + θ.2a′ 6/10 + θ.2b′ 10/10 = 39/60... |

*Note on the 19/60 figure:* The ADR's §Context states "moved the same model on the same bases from 19/60 to 59/60." The round-1 table: θ.1a 10/10 + θ.1b 0/10 + θ.2a 3/10 + θ.2b 10/10 + θ.2a′ 6/10 + θ.2b′ 10/10 = 39/60. The research log's verdict section also states "information-starved 19/60" but the round-1 table totals to 39/60. Reading the finding more carefully: "F-θ.2 (the round-2 finding): against round 1's information-starved 19/60." The log's composed meaning of "19/60" appears to be: the sum of *correct on the intended form* (Form A on work-complete θ.1a 10 + Form A on E4 θ.2a 3 + Form A on E4′ θ.2a′ 6 = 19) — i.e., counting only the Form A arms (the 30-arm form-A subset of round 1) or counting only the arms that crossed the decision-rule threshold differently. But checking: θ.1b (Form B, work-complete) 0/10 and θ.2b/θ.2b′ (Form B, work-remaining) 10/10 + 10/10 = 20/20. 0+20 = 20 Form-B-correct vs. 10+3+6 = 19 Form-A-correct, total 39/60. The ADR says "19/60." This is a discrepancy. The research log's F-θ.2 reads: "the same model, same bases, same verdict format moved 19/60 → 59/60 on digest + standard alone." The 59/60 in round 2 is: θ.3a 10 + θ.3b 9 + θ.4a 10 + θ.4b 10 + θ.4a′ 10 + θ.4b′ 10 = 59/60 — this checks out. The "19/60" must be a specific interpretation — possibly counting only the arms that needed to improve (the failing arms from round 1, not all round-1 arms). Looking at the research log F-θ.2 text again: "the explicit judgment call discriminates work-complete from work-remaining tails at 59/60 across two forms and three bases — against round 1's information-starved 19/60." The most coherent reading: round 1 produced 19/60 results that were correct *on the task that round 2 was meant to fix* — i.e., on all the bases that both rounds cover equivalently. But round 2 covers the same 6 arms (3a/3b/4a/4b/4a′/4b′) as round 1's 6 arms (1a/1b/2a/2b/2a′/2b′). Sum of round-1 correct: 10+0+3+10+6+10 = 39. The 19/60 figure does not match the table. **This is a minor discrepancy: the research log uses "19/60" in its F-θ.2 verdict, and the ADR repeats it. The table total is 39/60.** Possible explanation: the research log and ADR are reporting a subset — e.g., 19 correct on the specific arms that F-θ.2 cares about (the ones with the information problem). Without rerunning the raw data, the most likely interpretation is that the log and ADR mean the round-1 arms that were incorrect (60 - 39 = 21 wrong, 39 right), which also does not equal 19. The discrepancy is noted as a P3 finding below (see P3-4 addition — this merits a check against the raw results, not a must-fix since the finding narrative in the log makes sense as a characterization).

*Continuing rate verification:*

| Rate cited in ADR-037 | Source | Verified |
|-----------------------|--------|---------|
| Round-2: Form A-enriched 30/30 | θ Round-2 results table (θ.3a 10 + θ.4a 10 + θ.4a′ 10) | Confirmed |
| Round-2: Form B-enriched 29/30 (1 false-continue) | θ Round-2 results table (θ.3b 9 + θ.4b 10 + θ.4b′ 10) | Confirmed |
| Hosted 20/20 (minimax-m2.7) | θ Portability annotation (θ.h1/θ.h2 10+10) | Confirmed |
| Latency 7–19s local (Form B round-2) | θ Round-2 results: "Latency medians: Form B-enriched 7–8s on work-remaining bases / 19s work-complete" | Confirmed |
| Latency 1–2s hosted | θ Portability annotation: "latency 0.7–3.0s/call" | The ADR says "1–2s/call"; the log says "0.7–3.0s." The ADR's "1–2s" is within the range but the upper bound is understated (3.0 actual vs 2 stated). Minor but see P3-5 below. |
| Composed estimate ~0.9 finish, ~0.9 delegate | θ Round-2 results "Composed mechanism estimate" | Confirmed as labeled composed estimate |
| ~0.54 implicit composition | E4a 6/10 × E4b 9/10 = 0.54 | Arithmetic confirmed |
| V3 55/55 (ADR-036 contrast) | ADR-036 §Context "Cumulative V3 on qwen3:14b: 55/55" | Confirmed present in ADR-036 |
| ψ′ Arm D non-transfer | ADR-036 §Context "D: the lever is not model-portable" | Confirmed |

*Adding two P3 findings from the rate-verification work:*

**P3-4 — The "19/60" figure in the ADR's Context section does not match the round-1 table total**

- **Location:** §Context "moved the same model on the same bases from 19/60 to 59/60"
- **Claim:** Round-1 results were 19/60 correct.
- **Evidence gap:** The round-1 results table in the Spike θ log sums to 39/60 correct (θ.1a 10 + θ.1b 0 + θ.2a 3 + θ.2b 10 + θ.2a′ 6 + θ.2b′ 10 = 39). The research log's F-θ.2 verdict also states "19/60" — so the ADR faithfully reproduces the log's characterization, but the table arithmetic yields 39. The "19/60" likely refers to a subset of the arms — possibly the Form A arms only (θ.1a 10 + θ.2a 3 + θ.2a′ 6 = 19), since round 2 adopted Form B (and Form B-enriched was the winner) and the F-θ.2 narrative focuses on the information-starved failure. This interpretation makes the F-θ.2 characterization coherent: Form A's information-starved round-1 19/30 (not 60) is being compared against Form B-enriched's round-2 29/30. The "19/60" phrasing appears to be the log conflating the Form A round-1 subset with the full n=60.
- **Recommendation:** Clarify whether "19/60" refers to all 60 round-1 arms or to the 30 Form A arms specifically. If the latter, the figure should read "19/30 (Form A, the adopted form's predecessor)" or the comparison should be stated as "the same model, same bases, same digest format moved from 19/30 to 29/30 on Form B (or 30/30 Form A)." Alternatively, if "19/60" is intended to characterize combined-form round-1 performance, the table total 39/60 is the correct figure.

**P3-5 — The hosted latency figure in §Context understates the measured upper bound**

- **Location:** §Context "20/20 on zen:minimax-m2.7 at 1–2s/call"
- **Claim:** Hosted latency is 1–2s/call.
- **Evidence gap:** The Spike θ Portability annotation records "latency 0.7–3.0s/call" for the hosted arms. The ADR reports "1–2s/call," which omits the upper bound of 3.0s. The 1–2s characterization is within the measured range but understates it.
- **Recommendation:** Update to "0.7–3.0s/call" or "roughly 1–3s/call" to match the reported range. This is a minor accuracy issue.

---

### Candidate-Collapse Logic Verification

The ADR claims "no deterministic source survives examination" for a work-remaining signal. The Spike θ Entry Analysis enumerates three candidates: (1) task-text deliverable parsing, (2) capability ensemble session-completeness reporting, (3) client tool results. All three are examined and rejected with sound reasoning. The rejections are:

1. Task-text parsing is "semantic judgment in disguise" — sound; ADR-035's one-dispatch-one-deliverable granularity means the ensemble never sees the session-level task.
2. The capability ensemble sees only its dispatch brief per ADR-035, not session completeness — this follows directly from ADR-035 §Decision ¶3 (one dispatch → one deliverable; session state is not visible to the callee).
3. Client tool results report per-action success, not task completeness — confirmed by the ψ-capture bytes evidence (F-θ.1: tool results read "Wrote file successfully" with no path or content).

No unexamined source is apparent from the source material. The collapse is sound on the available evidence and the examined candidates cover the candidate space plausibly. No P1 warranted.

The framework-termination-policy rejection's two-premise structure is also sound:
- Premise (i): no framework-computable termination input — follows from the above.
- Premise (ii): no trustworthy text-only response when overriding a tool-calling model turn — the model is the stop mechanism; stripping tool calls produces no meaningful text. The Entry Analysis reasons: "the model is the stop mechanism; OpenCode ends the loop on a no-tool-calls response; stripping tool calls from a continuing response leaves no trustworthy closing text." This is a correct reading of the loop protocol and consistent with ADR-033's mechanism.

---

### ADR-036 Partial-Update Header — Scope Accuracy Check

The partial-update header on ADR-036 states:

1. "Trailing-turn (tool-result tail) composition only: the unconditional standalone trailing C3 guidance of Decision 1 is replaced by ADR-037's two-call composition" — **Accurate.** ADR-037 §Decision ¶4 explicitly confirms: "First turns and trailing turns carrying a new user task are untouched (ADR-036's merge branch stands unchanged)."

2. "Finding F / F-ψ″.3" as the cited cause — **Accurate.** ADR-037 §Context opens with Finding F and F-ψ″.3.

3. "Decision 1's first-turn merge branch, the FC (directive-in-user-turn presence) as scoped to action-generation calls, decisions 2–5, and all other content of this ADR remain current" — **Accurate as to decisions 2–5.** Decisions 2–5 (delegation-rate instrumentation scope, profile-swap re-validation, escalation-path deferral, carry-side preservation) are not touched by ADR-037. The FC scoping claim ("as scoped to action-generation calls") is also accurate: ADR-037 §Consequences Neutral explicitly states "ADR-036's FC (directive-in-user-turn presence) is scoped by this decision to action-generation calls."

4. "The 'no framework system message' neutral consequence is narrowed to action-generation calls composed with the client conversation (ADR-037's bare judgment call carries a framework judge system message outside the client contest)" — **Accurate.** This matches ADR-037 §Consequences Negative last bullet.

The header does not overclaim (it does not imply ADR-036's scope-of-claim, the 55/55 evidence, or the delegation-rate instrumentation have changed) and does not underclaim (it records all materially changed elements). P3-3 above notes a minor gap in ADR-036's own Consequences Neutral text not being annotated inline, but the header itself is accurate and complete.

**Status line check:** ADR-036's Status line reads "Accepted; Updated by ADR-037 (2026-06-05 — trailing-turn composition)." This is accurate.

---

### Conditional Acceptance Gating Condition

The gating condition specifies: (a) a real-OpenCode session where the finish turn lands as text-only and the client loop ends, verified from serve-log evidence (dispatch start / TurnDecision events — the WP-A scar layer-match discipline); (b) a work-remaining trailing turn still delegates; (c) the production digest join is part of the gate (judgment calls fed by framework dispatch records, not constructed digest).

This condition is:
- **Refutable:** each clause is verifiable from serve-log evidence, not from model-direct behavior.
- **Layer-matched:** it explicitly invokes the WP-A scar discipline (serve-log evidence, not model-direct-looking runs). The "passing-looking run can be model-direct" caveat is present.
- **Appropriately scoped:** the join condition (c) is the novel BUILD work the spike could not validate; it directly addresses the spike's known limitation (constructed path annotations used instead of framework-derived records).
- **Consistent with the corpus pattern:** mirrors ADR-036's Conditional Acceptance structure.

No P-level finding warranted.

---

## Section 2: Framing Audit

### Question 1: What alternative framings did the evidence support?

**Alternative framing A — "The trailing guidance was the bug": a narrower remediation frame.**

The evidence supports reading Finding F as "the C3 unconditional do-more-work directive was the proximate bug, and the fix is to not emit it unconditionally on all tool-result tails." Under this framing, the remediation is drop-C3-on-complete-tails rather than a new two-call composition mechanism. The spike log explicitly follows this path (F-ψ″.1 → Arm E4), and E4 refutes it: without trailing guidance on mid-task tails, delegation collapses to 0/10. So the narrower "guidance was the bug" frame is not a different valid conclusion — it was tested and ruled out by E4. The ADR's frame (two-call composition to conditionally emit guidance) is the evidence-supported choice. The narrower frame is not an unexamined alternative; it is documented as the tested and rejected drop-C3 candidate. This framing is already well-handled.

**Alternative framing B — Form A-enriched (30/30) as primary, Form B-enriched as cost-optimization.**

The evidence supports framing Form A-enriched as the more robust form (30/30 vs 29/30) with Form B as the cost-optimized variant. The ADR inverts this by adopting Form B as primary and naming Form A as the fallback. The adoption is justified by the pre-registered cost tiebreak, which is sound: n=10 cannot distinguish 30/30 from 29/30, and the cost difference (bounded ~1–2k vs growing ~28–30k) is structural. However, the framing choice has a subtle consequence: by naming Form B as primary, the ADR places the cost-bounded form in the operational path and the accuracy-equivalent form in the fallback position. If BUILD surfaces a bare-form failure the spike could not see, the fallback path reintroduces the client-prompt cost. A practitioner reading the ADR encounters the Form B cost argument in §Rejected alternatives, not in the §Decision primary statement. The framing is defensible and pre-registered; it is noted here for visibility.

**Alternative framing C — The three-layer taxonomy as evidence organization vs. theoretical claim.**

The three-layer scope-of-claim framing (framework guarantees / model judgment / latency optimization) organizes the evidence usefully, but it is explicitly labeled as "composed at the gate exchange... the layers organize measured facts but the taxonomy is drafting-time" in §Provenance. The framing is honest about its status. An alternative framing would present the same facts as a simpler two-part split: "what the framework guarantees structurally" vs. "what the model does within those guarantees." The three-layer version adds the paid-vs-local orthogonality observation, which is accurate and useful for the practitioner's gate-entry question. No finding; the alternative is less informative.

### Question 2: What truths were available but not featured?

**Underrepresented finding 1 — The hosted-arm latency upper bound (3.0s/call) vs the ADR's stated "1–2s".**

The Spike θ Portability annotation records 0.7–3.0s/call for the hosted arms. The ADR reports "1–2s/call" in §Context, which understates the upper bound. The discrepancy is minor (the upper bound matters for per-turn overhead calculations in long sessions) but is factually present. Surfaced as P3-5.

**Underrepresented finding 2 — The judgment question's wording-revision discipline (FC-58 applied to a second composition point).**

The ADR notes in §Consequences Neutral that "the judgment question text is tunable at the same evidence bar as the guidance text (wording revisions re-validate affected arms)." The spike log's design section establishes this explicitly (the FC-58 discipline). This is present in the ADR but in a Neutral consequence rather than in the FC set. A BUILD implementer might change the judgment question phrasing without triggering a re-validation obligation. The FC (judgment-seat re-validation) covers profile changes; it does not explicitly cover question-text changes. The ADR references the FC-58 discipline in the §Provenance but does not establish a fitness criterion that question-text changes require re-validation.
- **Why it may have been excluded:** The ADR treats wording as tunable (per FC-58), implying the obligation exists. Including it as an explicit FC might over-specify BUILD.
- **Impact on conclusions:** Minor. If BUILD changes the judgment question text without re-validating, the accuracy property could degrade in ways not caught by the profile-swap FC. The risk is low given the FC-58 discipline is named, but a fitness criterion stating "material judgment-question wording changes require re-validation of the affected arms" would be a tighter guarantee.

**Underrepresented finding 3 — The smoke-check pre-run amendment and its reveal.**

The pre-run smoke check (required by the methods review P3-B) found that a tools-less request on the work-complete context produced completion summaries with fabricated code blocks — summary decoration rather than the ω.3a performing-remaining-work flip. The amendment to the flip definition (excluding summary decoration from rule-4 disqualification) was sound and is correctly recorded in the log. The ADR references "finish-text quality" in §Decision ¶1 ("smoke-checked directly") and §Consequences (θ.3 finish-text quality: brief factual summaries, returnable as-is). What the ADR does not explicitly note is that the smoke-check concern (confabulated code in COMPLETE summaries) did not materialize under the deliverable-accounting question in round 2 — a positive finding that increases confidence in the round-2 COMPLETE responses. This is implicit from the round-2 finish-text note but is not drawn out as a direct resolution of the smoke-check concern. Minor; already well handled.

### Question 3: What would change if the dominant framing were inverted?

The dominant framing: "the framework cannot compute task-completeness, but it can guarantee the conditions under which a model judgment is accurate — and a structurally-bounded judgment is reliable."

Inverted framing: "a model judgment is inherently unreliable for a safety-critical stopping condition; the framework should maximize how rarely it defers to the model, not optimize the quality of the deferral."

Under the inverted framing:
- The 1/10 false-continue (one extra revision turn) looks different: the mechanism relies on a model judgment getting the stop condition right 90% of the time. In any stopping condition that matters for correctness, a 10% false-continue rate compounding across turns is a reliability gap, not a bounded geometric decay. The AS-3 cap provides the hard backstop, but the inverted framing asks whether the 9/10 accuracy is sufficient for a mechanism that gates session termination.
- The hosted 20/20 result looks weaker: one pair on one task type does not establish that the deliverable-accounting judgment is reliable across task diversity, and the inverted framing would ask whether the scope limitation (file-write tasks, 1–3 deliverables) is a significant constraint on the mechanism's usefulness.
- The "framework guarantees what the judgment sees" reframe looks more like risk management than a guarantee: the framework can guarantee the digest's provenance, but it cannot guarantee the digest captures the semantically relevant fact (was the task actually complete, vs were the requested files written successfully). A task whose completion criterion exceeds "files written" would not be served by the deliverable-accounting standard.

The inverted framing reveals one genuine tension the ADR handles: the deliverable-accounting standard explicitly excludes code correctness ("a successful write of a requested file counts as produced; you are not being asked to verify code correctness"). Under inversion, this is not a scope limitation — it is the mechanism's fundamental limitation. A session that completes a write but produces incorrect code will be judged COMPLETE by the mechanism, which may leave the client with a session that appears to have converged but produced a broken deliverable. The ADR records this correctly in the scope-of-claim section and explicitly names code-correctness ownership (capability ensemble + calibration gate + PLAY). The inverted framing highlights that the mechanism is more naturally suited to "were all requested files produced" than to "was the work actually done correctly" — the former is what the mechanism guarantees, and the latter requires downstream validation.

**What would the ADR need to address if it took the inverted framing seriously?** The ADR would need to name explicitly that the mechanism's COMPLETE verdict is a deliverable-production certificate, not a work-correctness certificate, and that the client should not treat a session ending normally as a signal that the work is correct. This distinction is present implicitly (the code-correctness exclusion) but is not drawn as a user-facing consequence. Given the mechanism's context (internal framework session management), the user-facing communication of this limitation is downstream of this ADR; the ADR is not the right place for it. But the framing difference is worth naming.

### Framing Issues

**P2-F1 — The hosted-annotation's placement risks implying portability is more established than the evidence supports**

- **Location:** §Context last paragraph: "the identical composition scored 20/20 on zen:minimax-m2.7 at 1–2s/call (vs 7–19s local). One pair does not establish portability; it establishes the composition is not qwen-idiosyncratic"
- **Claim:** The hosted annotation is not a portability claim; it contrasts structurally with ADR-036's V3 lever non-transfer.
- **Evidence gap:** The §Context paragraph correctly hedges: "One pair does not establish portability." However, appearing in §Context (the high-read section that sets up the decision) rather than in §Consequences or §Decision ¶3 (Three-layer scope-of-claim §Layer 3), the hosted result occupies a prominent position relative to its evidential weight. The spike log's design pre-registered that hosted results are read only after the local verdict and placed in a separate "Portability annotation" section — a deliberate epistemic segregation the ADR does not fully replicate by including the hosted result in §Context. A practitioner reading §Context sequentially encounters the 20/20 hosted result before reading the scope caveats in §Decision ¶5. The Three-layer scope-of-claim (Layer 3) correctly scopes it as "orthogonal to reliability"; §Context brings it forward.
- **Recommendation:** Either move the hosted result mention from §Context to §Consequences (where Layer 3 appears) or add an explicit note in §Context alongside the 20/20 result: "read as a single-pair annotation, not as portability evidence — see §Decision Layer 3." This aligns the reading order with the spike's own P2-D discipline.

**P2-F2 — The three-layer scope-of-claim forecloses the "framework-guarantee reading" more strongly than the evidence supports at Layer 2**

- **Location:** §Decision Three-layer scope-of-claim ¶2 (Layer 2): "no framework can compute task-completeness; the framework instead bounds the role (counting deliverables against explicit evidence) into AS-9's reliable class. Reliability here is structural bounding plus a measured per-profile rate — ~0.9–1.0 on the two measured models — not a guarantee."
- **Claim:** Layer 2 is a "structural bounding plus measured rate" assurance — not a framework guarantee but something stronger than unguided model behavior.
- **Evidence gap:** The layer distinction is accurate, but the framing "~0.9–1.0 on the two measured models" conflates the local qwen3:14b 29/30 and the hosted minimax-m2.7 20/20 into a range. These are not two models of comparable evidential weight: qwen3:14b is the primary evidence (60 total arm-runs across two rounds), and minimax-m2.7 is a one-pair annotation (20 runs). Presenting "~0.9–1.0 on the two measured models" implies a cross-model reliability profile that the evidence does not establish. The ADR elsewhere is careful to call the hosted result a "portability annotation," but the Three-layer section's "two measured models" phrasing is less careful.
- **Recommendation:** In Layer 2, characterize the rate as "~0.9 on the primary measured model (qwen3:14b, 29/30); the hosted annotation (20/20) is consistent but does not constitute an independent reliability measurement." This keeps the layer distinction while preventing the two-model language from implying more portability evidence than exists.

**P3-F1 — The "practitioner-prompted but agent-composed" three-layer taxonomy framing could more explicitly note that all three layers are organized around a single gate-exchange question**

- **Location:** §Decision §Three-layer scope-of-claim preamble: "Named at the practitioner's gate-entry question (is reliability a paid-vs-local scale, or an architectural guarantee paid merely speeds up?)"
- **Claim:** The three layers are a response to the practitioner's question.
- **Evidence gap:** The framing is honest (the §Provenance labels it "composed at the gate exchange... the taxonomy is drafting-time"). The preamble does reference the gate-entry question. A slightly more explicit note that all three layers are organizing the same question (not independently derived from the evidence) would help a reader understand the taxonomy's epistemic status without requiring them to read §Provenance.
- **Recommendation:** Add a brief parenthetical in the preamble: "(drafting-time taxonomy organizing the evidence per §Provenance — not independently derived from the spike findings)."

---

## Convergence-Saturation Signal (ADR-094)

**Convergence-Saturation Signal:** NOT TRIGGERED

- Round number: R1
- P1 count this round: 0 (Section 1: 0; Section 2: 0)
- P2 count this round (new, non-carry-over): 6 (Section 1: P2-1 AS-9 class extension without domain-model annotation; P2-2 composed-estimate labeling loose in Consequences; P2-3 geometric-decay arithmetic basis not flagged at claim site; P2-4 combined re-validation instrument not operationally defined; Section 2: P2-F1 hosted annotation placement; P2-F2 "two measured models" conflates unequal evidential weight)
- New framings or claim-scope expansions: AS-9 extension to termination-judgment task surfaces (not annotated in domain model); "deliverable-production certificate vs. work-correctness certificate" distinction surfaced by framing inversion
- Recommendation: CONTINUE to next round.

*Single-purpose re-audits (dispatched per the re-audit-after-revision rule) omit this section. Form-change events reset the round-count baseline — the first audit on a new form is its R1.*

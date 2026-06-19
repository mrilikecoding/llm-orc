# Argument Audit Report — Round 2

**Audited document:** docs/agentic-serving/decisions/adr-043-collapse-dual-serving-surfaces-to-one-loop.md
**Source material:** docs/agentic-serving/essays/research-logs/cycle-7-spike-iota-one-surface.md; docs/agentic-serving/decisions/adr-027-framework-driven-dispatch-pipeline.md; docs/agentic-serving/decisions/adr-033-layer-a-loop-driver-multi-turn-agentic-surface.md
**Prior audit (R1):** docs/agentic-serving/housekeeping/audits/argument-audit-decide-cycle-7-loopback9.md
**Genre:** ADR
**Date:** 2026-06-18

---

## R2 Verification Map

Each R1 finding is assessed against the revised ADR (Resolution A → Resolution B). Per the form-change baseline-reset rule (ADR-094), the F-ι.1 reasoning is treated as fresh material requiring full scrutiny, not merely as a revised R1 finding.

| R1 Finding | Status | Notes |
|---|---|---|
| P2-1 (ADR-097 attribution error for "no demand" claim) | **RESOLVED** | §Rejected alternatives and §Provenance check now correctly attribute "no demand" to `product-discovery.md` and cycle-status handoffs, not ADR-097. ADR-097 is cited only for the grounding-mechanism path. |
| P2-2 (dormancy reasoning gap for ADR-031/032) | **PARTIALLY RESOLVED — one residual** | §Consequences Negative now names ADR-032 explicitly and requires the backward-propagation sweep to add a dated handoff note. ADR-031's latency posture is still not addressed (carry-over P2-C1 below). |
| P2-F1 (Resolution A narrows ADR-027's cost-distribution promise) | **RESOLVED — by choosing Resolution B** | B preserves ensemble-backed answers for toolless clients. The framing section's concern that A narrows the ADR-027 promise is now the explicit stated reason A was rejected. |
| P3-1 (write-capable tool predicate undefined) | **SUPERSEDED** | Resolution B does not gate `invoke_ensemble` on the client's tool set, so the write-capable predicate no longer appears. The P3 is closed. |
| P3-2 (27/30 aggregate undersells plain-question reliability) | **CARRY-OVER — unchanged** | The ADR's §Context still presents 27/30 without the parenthetical clarification R1 recommended. See carry-over P3-C1 below. |
| P3-F1 (ADR-031/032 dormancy risks configuration honesty) | **PARTIALLY RESOLVED** | §Consequences Negative now explicitly names ADR-032 dormancy and requires the sweep to add a handoff note. The concern about ADR-031's latency posture is not addressed. |

---

## Section 1: Argument Audit

### Summary

- **Genre:** ADR
- **Argument chains mapped:** 7 (loop-subsumes-no-tools, F-ι.1 Resolution-B mechanism, Resolution-A rejection, ADR-027 supersession, ADR-028/029/031/032 dormancy, "no demand" grounding, FC pair adequacy)
- **Issues found:** 5 (P1: 0, P2: 2 [1 carry-over, 1 new], P3: 2 [1 carry-over, 1 new])
- **Pyramid coverage map:** N/A
- **Expansion-fidelity findings:** N/A

---

### P1 — Must Fix

No P1 findings.

---

### P2 — Should Fix

**P2-C1 (carry-over from R1 P2-2). ADR-031's latency posture is dormanted without a handoff note.**

- **Location:** §Decision 2 ("ADR-028/029/031/032 loses its production caller — disposition per the backward-propagation sweep"); §Consequences Negative (ADR-028/029/031/032 become dormant)
- **Status:** The R1 recommendation asked for explicit reasoning distinguishing ADR-028/029 (pipeline-mechanism only) from ADR-031/032 (cross-cutting principles). The revised ADR now names ADR-032 explicitly in §Consequences Negative and requires a dated handoff note for its configuration-honesty commitment. ADR-031 is still not distinguished. ADR-031's Population A timeout analysis (the ~36s latency floor characterization) is a cross-cutting latency posture for the serving surface, not a pipeline-mechanism spec. The loop's per-turn latency profile on no-tools requests (which now route through the driver rather than a pipeline) has not been analyzed relative to ADR-031's baselines. Dormanting ADR-031 without a note on whether the loop surface provides a better or worse latency posture forecloses the comparison.
- **Recommendation:** In the backward-propagation sweep note on ADR-031, record that its latency-floor characterization was grounded in the pipeline path; the loop's no-tools latency (text-completion turn, no ensemble dispatch) is almost certainly lower than the pipeline's plan+dispatch+synthesize floor, but this comparison has not been measured. One sentence noting the loop's expected latency benefit and flagging it as a BUILD validation item is sufficient.

---

**P2-N1 (new). The "determinism advantage of A dissolves under B" argument has a hidden dependency on seat behavior remaining stochastic.**

- **Location:** §Rejected alternatives: F-ι.1 Resolution A ("The determinism advantage A claimed dissolves under B, because B makes the delegate branch valid (text-marshalled), so the seat's stochastic delegate-vs-finish is benign rather than correctness-bearing")
- **Claim:** Under B, the seat's stochastic delegate-vs-finish judgment is benign — so A's determinism advantage disappears. This is the core reason Resolution A is rejected in favor of B.
- **Issue:** The argument is sound but depends on an unstated premise: that the Terminal correctly identifies when the client's tool is absent and text-marshals unconditionally in that case. If the Terminal's `_emit_apply_work` branch is implemented incorrectly — or if `outcome.tool_name` is ever absent or mismatched — the stochastic seat choice could still produce an un-executable tool_call under B, which means the determinism argument is not fully dissolved but merely relocated from the seat to the Terminal implementation. The Arm-B spike tested the behavioral seat distribution, not the Terminal's branch correctness under B (B is new; the spike ran under the pre-B code). The FC (adaptive marshalling) is refutable and bounds this correctly ("a response that emits a client tool_call naming a tool absent from the request's `tools[]` violates this"), but the ADR does not acknowledge that the FC covers the same correctness claim the determinism argument was making.

  The code (`_emit_apply_work`, lines 103-143) shows no branch on client tool presence — it unconditionally builds a `ClientToolCall`. The Resolution B branch is described in the ADR's Decision section but is **not yet present in the code**. This is as expected (B is the decided resolution for BUILD), but the ADR's claim that "B is a Terminal-only change" is prospective, not a description of the current codebase. The argument that "the determinism advantage dissolves" is therefore contingent on a correct B implementation, not on current code.
- **Recommendation:** Add a single qualifying clause: "once B is implemented, the seat's stochastic delegate-vs-finish is benign — both branches yield a valid response. The FC (adaptive marshalling) is the correctness anchor in BUILD." This makes the prospective nature of the dissolving-determinism argument explicit and names the FC as the mechanism that must hold.

---

### P3 — Consider

**P3-C1 (carry-over from R1 P3-2). The 27/30 aggregate still does not distinguish plain-question vs. F-ι.1 failures.**

- **Location:** §Context ("Arm B (live qwen3:14b, N=10/cell) 27/30 finish-with-text")
- **Claim:** 27/30 as the graceful-finish pass rate.
- **Issue:** Unchanged from R1. The 3/30 non-finishes are the F-ι.1 delegations on the `match_caps` cell, which is precisely the defect B closes — not random noise. The plain-question finish rate is 20/20. The aggregate framing slightly undersells actual reliability for the plain-question case.
- **Recommendation:** A parenthetical: "27/30 finish-with-text (20/20 on plain questions across both capability states; 3/30 non-finishes are the F-ι.1 delegation gap, resolved by Resolution B)."

---

**P3-N1 (new). The "B is a Terminal-only change" claim should note that FC-58 preservation is prospective.**

- **Location:** §Decision 3 ("Mechanically, B is a Terminal-only change (`_emit_apply_work` branches on whether `outcome.tool_name` is in the request's `tools[]`); the Loop Driver's `_delegation_tools` / guidance composition are unchanged, so FC-58's 'guidance never references a tool not offered' invariant is preserved untouched.")
- **Claim:** The Loop Driver is unchanged; FC-58 is preserved.
- **Issue:** This is correct in scope (FC-58 governs the guidance, which is in the Loop Driver, which B does not touch). However, `_delegation_tools` still offers `invoke_ensemble` regardless of client-tool presence. The guidance remains composed when capabilities are present (as confirmed by the code at line 640). Under B, that is explicitly intended — the seat is still offered `invoke_ensemble` even for toolless clients. The ADR is correct that this preserves FC-58. What is slightly imprecise is the phrase "unchanged" — `_delegation_tools` still functions as before, and the Terminal handles the output-shape adaptation. Noting that "unchanged" means the Loop Driver's behavior is unchanged, not that no-tools requests now receive the same guidance as tool-driven ones (they do, by design), would prevent a future reader from thinking the no-tools path gets a stripped-down guidance set.
- **Recommendation:** One clarifying phrase: "the Loop Driver's `_delegation_tools` / guidance composition are unchanged — a toolless request still receives the full guidance (including `invoke_ensemble` offer) from the Loop Driver; only the Terminal's emission adapts the output shape."

---

## Section 2: Framing Audit

### Question 1: What alternative framings did the evidence support?

**Alternative framing A: "Resolution B defers a testability problem rather than solving it."**

The spike ran with N=10/cell under the pre-B Terminal (which unconditionally emits a tool_call). Resolution B's correctness depends on the Terminal branch logic in `_emit_apply_work`. This branch was described but not tested by Spike ι — the spike measured seat behavior, not Terminal emission under B. A reader could argue the framing should be "B is accepted on design argument, not empirical validation" and that the ADR's confident resolution framing obscures that B remains Conditional until a BUILD test suite covers the new branch. The belief required: "an untested Terminal branch is a meaningful residual risk, not a bounded BUILD item."

**Alternative framing B: "The ADR-043 collapse and the B decision are two separate decisions bundled in one ADR."**

The collapse decision (Decision 1: remove the discriminator, retire the pipeline) is grounded by Spike ι's Arm A + Arm B graceful-finish results. The F-ι.1 resolution (Decision 3: how to handle delegation on toolless clients) is grounded by a design argument (the framing audit's finding that A narrows ADR-027's promise) plus the practitioner's "turns are emergent" reframe. The spike empirically settled the collapse; it did not empirically settle which resolution to choose for F-ι.1 — the spike's results are silent on whether B-marshalled text answers satisfy toolless clients better or worse than A's inline seat text. A reader could argue these two decisions warrant separate ADRs and separate grounding.

**Alternative framing C: "The R1 framing finding drove the B selection more than the spike evidence did."**

The spike log records that the agent initially recommended Resolution A, and the practitioner reframed. The spike log also records: "The argument audit's framing section flagged the same narrowing independently." This means the R1 audit's P2-F1 was a co-driver of the Resolution B choice, alongside the practitioner's reframe. The ADR's Provenance check names "the argument audit's framing finding" explicitly, which is accurate. But the framing — that A forecloses ensemble answers for toolless clients in a way that contradicts ADR-027 — is a logical argument, not a measured empirical result. The selected resolution (B) is therefore primarily justified by a normative framing judgment, not by evidence that B's outputs are better. This is a valid basis for a decision, but a reader should know the justification type.

### Question 2: What truths were available but not featured?

**Underrepresented truth 1: The smoke run outcome is still not in the artifact trail.**

R1 flagged this (Section 2, Question 2, Underrepresented truth 1). The spike plan named a real-server smoke run; the spike log's results section records only `probe_live.py` results. The revised ADR does not add the smoke outcome. The "full Acceptance" classification still rests on spike results that may not include the end-to-end HTTP surface test.

**Underrepresented truth 2: Resolution B adds a branch whose edge cases under the new serving path are not enumerated.**

When the client has no matching tool and the Terminal text-marshals the deliverable, the resulting completion carries the deliverable content as a text response. The ADR does not name what happens to the `action_record.record_content` call in this path — the current `_emit_apply_work` calls `record_content` only on the tool_call branch (line 136-137). Under B, a text-marshalled path would not call `record_content`, meaning a delegated-and-text-marshalled deliverable would not be anchored for a subsequent turn's callee. For a toolless client, there is no subsequent turn in the traditional sense (no `role: "tool"` result returns), so this is probably not a gap — but the ADR does not address it.

**Underrepresented truth 3: The ADR-032 handoff note is required but not yet written.**

§Consequences Negative states: "the backward-propagation sweep must add a dated handoff note recording that the direct-completion tier-escalation path no longer has a live caller." This is a commitment to a future action, not an action taken. A future reader of ADR-032 will not see this note until the sweep runs. If the sweep is never dispatched, the commitment is silently unfulfilled.

### Question 3: What would change if the dominant framing were inverted?

The revised ADR's dominant framing: "Resolution B preserves ensemble delegation for all clients, so the collapse is not a narrowing — it is a unification that makes the framework's agentic behavior uniform."

**Inverted framing:** "Resolution B trades a determinism guarantee for a text-shape guarantee. Under B, a toolless capability-matched request gets an ensemble-backed answer in text form — which may or may not be what the toolless client expected, since the original pipeline offered this capability via a different path (plan+dispatch+synthesize). The Terminal's text marshalling of a single-ensemble deliverable is not equivalent to the pipeline's multi-ensemble synthesis."

Under the inverted framing:
- The capability genuinely retired (multi-capability fan-out for toolless clients in a single response) is framed as "no recorded demand," which is true of the *current* corpus. If a toolless client integrates on the assumption that the endpoint can do multi-capability synthesis in a single call, that expectation is no longer met under either A or B. The ADR correctly identifies this but the "no demand" framing forecloses the question of whether demand exists outside the recorded corpus.
- Under B, if the seat delegates on a toolless capability-matched request, the Terminal emits the deliverable as text (a single ensemble's output). Under the old pipeline, the same request would have gone through the planner+synthesizer path. For some queries, these outputs would differ materially. The ADR does not discuss this difference.
- This is P2 territory rather than a refutation — the ADR is clear about what is retired and the corpus evidence supports the no-demand claim. But a practitioner reviewing against Population A's expectations should consider whether any known Population A scenario expects multi-capability synthesis in a single no-tools call.

### Framing Issues

**P2-F1-R2 (new). The Resolution B selection is grounded primarily in a normative design argument, and the ADR's "Provenance check" should name the argument type.**

- **Location:** §Provenance check ("F-ι.1 and Resolution B are from Spike ι Arm B ... the practitioner's uniform-agentic-behavior framing ... and the argument audit's framing finding that Resolution A narrows ADR-027's promise")
- **Claim:** Resolution B's provenance traces to the spike, the practitioner's reframe, and the audit's framing finding.
- **Issue:** The spike's empirical contribution to the B selection is the discovery of F-ι.1 (the gap), not the selection between A and B. The A/B selection is grounded by two normative arguments: (1) A narrows ADR-027's promise, and (2) B makes the seat's stochastic behavior benign. Neither (1) nor (2) is an empirical measurement of B's output quality. Naming the argument type in the provenance entry would help a future reader distinguish what the spike proved from what the design argument decided.
- **Recommendation:** In the Provenance check, note the distinction: "Spike ι Arm B established F-ι.1 (the gap); the A/B selection is a design-argument decision, not a measured outcome — grounded by the ADR-027 narrowing argument and the practitioner's uniform-agentic-behavior framing. Empirical validation of B's text-marshalled output quality defers to BUILD."

**P2-F2 (new). Resolution A's rejection is stated as a fact (it is a "functional narrowing") but the supporting argument is a framing from the R1 audit, not a corpus decision artifact.**

- **Location:** §Rejected alternatives: F-ι.1 Resolution A ("Rejected: this is a functional narrowing of ADR-027's cost-distribution promise")
- **Claim:** Resolution A is a functional narrowing.
- **Issue:** The claim that A is a functional narrowing comes from the R1 audit's framing section. This is accurate — the R1 audit surfaced it as P2-F1. But the ADR presents it as a settled fact ("this is a functional narrowing") rather than as a framing that the audit raised and the practitioner accepted. The distinction matters for traceability: a future reader might wonder how the functional-narrowing characterization was established and not find a corpus artifact that makes it a decision rather than an auditor's observation. The chain is: R1 audit raises P2-F1 → practitioner accepts framing → ADR incorporates it as a rejection criterion. That chain should be visible.
- **Recommendation:** In §Rejected alternatives, acknowledge the provenance: "Rejected: this is a functional narrowing of ADR-027's cost-distribution promise (the argument-audit framing section surfaced this; the practitioner accepted it as the resolution criterion)."

---

## Convergence-Saturation Signal (ADR-094)

**Convergence-Saturation Signal:** NOT TRIGGERED

- Round number: R2 (per form-change baseline-reset rule: the A→B revision is a material reasoning change in F-ι.1, treated as fresh reasoning; the round count continues from R1 rather than resetting because the document's overall structure and non-F-ι.1 arguments are unchanged — the form-change rule applies to document restructuring, not to in-place reasoning revision)
- P1 count this round: 0 (Section 1: none; Section 2: none)
- P2 count this round (new, non-carry-over): 3 (P2-N1 — determinism-dissolves argument has unstated prospective dependency; P2-F1-R2 — normative vs. empirical grounding not named in provenance; P2-F2 — Resolution A rejection presented as settled fact rather than audit-sourced framing)
- New framings or claim-scope expansions: (1) "B defers testability — the Terminal branch is untested by Spike ι"; (2) "the A/B selection is a normative design argument, not an empirical measurement"; (3) "the functional-narrowing characterization is audit-sourced, not an independent corpus decision." These are claim-scope characterizations not named in R1.
- Recommendation: **CONTINUE to R3.** P1 count = 0, but P2 count = 3 (exceeds the ≤1 threshold) and three new claim-scope characterizations emerged. Signal does not trigger.

*Single-purpose re-audits (dispatched per the re-audit-after-revision rule) omit this section. Form-change events reset the round-count baseline — the first audit on a new form is its R1.*

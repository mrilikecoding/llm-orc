# Argument Audit Report

**Audited document:** docs/agentic-serving/decisions/adr-043-collapse-dual-serving-surfaces-to-one-loop.md
**Source material:** docs/agentic-serving/essays/research-logs/cycle-7-spike-iota-one-surface.md; docs/agentic-serving/decisions/adr-027-framework-driven-dispatch-pipeline.md; docs/agentic-serving/decisions/adr-033-layer-a-loop-driver-multi-turn-agentic-surface.md
**Genre:** ADR
**Date:** 2026-06-18

---

## Section 1: Argument Audit

### Summary

- **Genre:** ADR
- **Argument chains mapped:** 6 (loop-subsumes-no-tools, F-ι.1 Resolution-A determinism, ADR-027 superssession, ADR-028/029/031/032 dormancy, "no stakeholder demand" grounding, FC pair adequacy)
- **Issues found:** 4 (P1: 0, P2: 2, P3: 2)
- **Pyramid coverage map:** N/A
- **Expansion-fidelity findings:** N/A

---

### P1 — Must Fix

No P1 findings.

---

### P2 — Should Fix

**P2-1. "No stakeholder-articulated demand" is attributed to ADR-097 but ADR-097 is a grounding-mechanism ADR, not a stakeholder-demand registry.**

- **Location:** §Context ("Maintaining a second caller for a pivoted-from capability is carrying infrastructure cost without a stakeholder demanding it."); §Rejected alternatives: Keep the two-surface split ("Rejected: carrying two callers for a pivoted-from capability with no stakeholder demand (ADR-097 — no stakeholder-articulated demand)")
- **Claim:** ADR-097 is cited as the locus of the "no stakeholder-articulated demand" finding. The phrase is used twice, once with the parenthetical attribution.
- **Evidence gap:** ADR-097 in the corpus is the Empirical-Grounding Filter (Conditional Acceptance / spike-validation mechanism). It records grounding paths and discharge conditions, not stakeholder-demand findings. The actual evidence for "no stakeholder demand for the multi-capability fan-out" lives in the product-discovery and cycle-status corpus — the practitioner's north-star pivot to the tool-driven serving surface (product-discovery.md), the handoff's characterization of the fan-out as "the original answer-a-question vision the loop-back pivoted from," and the absence of any population A voice record requesting NL multi-capability fan-out routing. None of those sources is ADR-097. The attribution is incorrect, and a reader tracing the provenance will not find the no-demand evidence in ADR-097.
- **Recommendation:** Remove the "(ADR-097 — no stakeholder-articulated demand)" parenthetical and replace with the actual source: the cycle's north-star pivot (product-discovery.md + cycle-status handoff). A formulation: "no stakeholder-articulated demand for the multi-capability fan-out (the pivot to the tool-driven north star is recorded in product-discovery.md and the cycle-status handoff)." Alternatively, if ADR-097 is the intended citation for the grounding-mechanism path-3 classification (which is mentioned in the Status block), separate that citation from the stakeholder-demand claim, which needs its own provenance.

---

**P2-2. The dormancy disposition for ADR-028/029/031/032 is stated as a consequence but no backward-propagation mechanism or update-header plan is named.**

- **Location:** §Decision 2 ("ADR-028/029/031/032 loses its production caller — disposition per the backward-propagation sweep (dormant-with-dated-note for any decision that governed the pipeline only; body-immutable record preserved)"); §Consequences Negative ("ADR-028/029/031/032 … become dormant architecture-of-record. Knowledge preserved (body-immutable), but no live surface.")
- **Claim:** The four subtree ADRs become dormant, with a dated note, via a "backward-propagation sweep." The ADR treats this as settled.
- **Evidence gap:** The ADR does not name whether the backward-propagation sweep has been run, whether each of ADR-028/029/031/032 has a scope that *exclusively* governs the pipeline (ADR-031's latency policy and ADR-032's fallback-shape / transparent-endpoint-promise split both articulate principles — configuration honesty, cost-distribution accountability — that could have relevance beyond the pipeline surface). ADR-032 in particular governs the honest-response-labeling commitment, which, while framed via the pipeline, codifies a transparent-endpoint promise the loop-driven surface arguably also inherits. Similarly, ADR-031's tier-escalation and Population A timeout analysis are latency characterizations that would apply to any serving surface, not only the pipeline. The blanket "governed the pipeline only" categorization may be accurate for ADR-028 (routing-planner spec) and ADR-029 (response-synthesizer spec), but is less clearly accurate for ADR-031 and ADR-032 without explicit reasoning. The ADR makes the dormancy claim without that reasoning.
- **Recommendation:** For ADR-028 and ADR-029, the "governed the pipeline only" classification is clearly correct — their specs are planner-ensemble and synthesizer-ensemble respectively, with no broader applicability. State this explicitly. For ADR-031 and ADR-032, note whether their principled commitments (configuration honesty per ADR-032; Population A timeout analysis per ADR-031) are inherited by the loop surface or truly scoped to the pipeline surface only. If the loop surface does not need a parallel configuration-honesty or timeout commitment, say so; if it does, flag that ADR-031/032's principles may need to be carried forward in a different form rather than fully dormanted.

---

### P3 — Consider

**P3-1. The "write-capable tool" predicate in Resolution A is named as the gate condition but its detection logic is not defined.**

- **Location:** §Decision 3 ("the seat-filler the `invoke_ensemble` delegation tool — and composes the delegation guidance — only when the client request carries a write-capable tool"); §Consequences Negative ("The gate's predicate ('write-capable tool') has a narrow edge — a client offering only read-class tools — left as a bounded follow-up")
- **Claim:** The gate condition is "client request carries a write-capable tool." The ADR acknowledges the narrow edge case (read-only tools) as a bounded follow-up.
- **Issue:** The predicate "write-capable" is not defined. In the spike context it was straightforward (OpenCode's `write`/`edit`/`bash` are all write-capable by obvious semantics). In production, a client may offer a tool named "file_write" or a tool that emits to an arbitrary destination. The FC introduced by Decision 3 is refutable ("a no-tools request whose response emits any client tool_call violates this"), but the positive condition — which tools count as "write-capable" — is not operationally defined. The follow-up acknowledgment is appropriate; this P3 surfaces the definitional gap so it does not get lost.
- **Recommendation:** Add a BUILD-phase clarification item: define "write-capable" as an explicit set or classification rule (e.g., tools whose `type` or name pattern matches an emission target). The edge-case note in §Consequences Negative already flags one boundary; a named classification rule is the companion artifact.

---

**P3-2. The 27/30 graceful-finish claim is aggregated across three cells including one where 3/10 delegations were the F-ι.1 defect, not a finish failure.**

- **Location:** §Context ("Arm B (live qwen3:14b, N=10/cell) 27/30 finish-with-text")
- **Claim:** The ADR presents 27/30 as the graceful-finish pass rate.
- **Issue:** This framing is accurate but can mislead a reader who doesn't cross-reference the spike log: the 3 non-finishes are the `match_caps` delegations that are *precisely the defect* Resolution A closes. They are not random failures on plain questions — they are the specific delegation-on-no-tools-client behavior F-ι.1 names. The true plain-question finish rate is 20/20 (100%), not 90%. For a reader assessing whether the collapse is safe, the 27/30 aggregate slightly undersells the actual plain-question reliability, because it incorporates the defect that the ADR itself resolves. This is not an error; it is a framing judgment that trades precision for succinctness.
- **Recommendation:** Consider adding a parenthetical: "27/30 finish-with-text (20/20 on plain questions across both capability states; the 3/30 non-finishes are the F-ι.1 delegations on the capability-matched cell, resolved by Resolution A)." This self-contained reading removes the need to cross-reference the spike table and correctly attributes the gap to the defect Resolution A closes rather than to noise.

---

## Section 2: Framing Audit

The framing audit makes the negative space of content selection visible. The primary document chose a framing — this section examines what that choice excluded.

### Question 1: What alternative framings did the evidence support?

**Alternative framing A: "No-tools requests should keep ensemble access via Resolution B."**

The spike log explicitly presents Resolution B (marshal-to-text: on `ApplyWork` when the client has no matching tool, return the ensemble deliverable as text) as a viable alternative. The ADR records it in §Rejected alternatives with the label "documented future option if ensemble-backed no-tools Q&A is ever demanded." What the ADR does not do is argue that ensemble-backed no-tools answers are *not valuable* — it argues they are not demanded. A reader who believes ensemble routing for plain Q&A has value (the original ADR-027 north star, per product-discovery §C1 / OQ #18 configuration honesty) could reasonably read the spike as showing the loop can serve Q&A via inline text *or* ensemble dispatch, and that the determinism argument selects the cheaper option but does not prove the costlier option is wrong. The belief required for this framing: "the ensemble-backed Q&A capability has future value, and Resolution A forecloses it more firmly than a gate change would."

**Alternative framing B: "Collapse is premature — validate the full loop at the real serving surface before retiring the pipeline."**

Arm B ran N=10/cell via a local Ollama seat, not through the full HTTP serving surface. The "real-client smoke" mentioned in the spike log (§Arm B: "A real-server smoke (local serve + a throwaway uncommitted discriminator flip + a no-tools curl) confirms the path end-to-end through the HTTP surface") is described in the plan but the spike log's results section does not record its outcome explicitly. A reader could argue the collapse should have produced a live smoke result in the log before the ADR was written, per the validate-against-real-client corpus discipline. The belief required: "the probe_live.py result may not fully characterize the production path without a recorded serving-surface smoke."

**Alternative framing C: "Dormanting four ADRs at once overpacks this ADR — each dormancy deserves its own sweep."**

The cycle's prior pattern (ADR-033 updated ADR-027 via a partial-update header; ADR-041 updated ADR-035 via a partial-update header) has been to scope changes narrowly. ADR-043 declares four subtree ADRs dormant via a referenced backward-propagation sweep, without surfacing that sweep's reasoning in the ADR body. A reader who values traceability could argue that each of the four ADRs should carry its own update header with the specific reasoning for dormancy, not a single collective declaration.

### Question 2: What truths were available but not featured?

**Underrepresented truth 1: The spike smoke run outcome is not recorded.**

The spike plan (§Arm B in the research log) names a real-server smoke as part of the arm, but the results section reports only the `probe_live.py` results (the library-call arm), not a distinct smoke result. The ADR's claim of "full Acceptance, not Conditional" rests partly on the real-client discipline; if the smoke was run but not recorded, the evidence is not in the artifact trail for a future auditor to verify. If it was not run, the ADR's full-Acceptance classification may be slightly overconfident relative to the corpus discipline.

**Underrepresented truth 2: ADR-031's Population A timeout analysis may carry forward.**

The spike grounded the collapse for qwen3:14b locally. ADR-043 does not discuss what happens to the latency posture of no-tools requests under the loop versus the retired pipeline. The loop adds at least one driver round-trip per request; if the pipeline's ~36s latency floor (ADR-031) was already borderline for some Population A clients, the loop may produce a different (potentially higher or lower) latency on plain text-completion turns. This is likely not consequential (a text-completion turn bypasses ensemble delegation and is probably *faster* than the pipeline's plan+dispatch+synthesize path), but the ADR is silent on latency, and ADR-031 is being dormanted without a handoff note about whether the loop's latency profile supersedes it favorably or not.

**Underrepresented truth 3: ADR-032's configuration-honesty commitment has no named loop-surface analog.**

ADR-032 codified honest response labeling (`served_by: "ensemble:<name>"` vs `served_by: "direct"`). The loop surface does not emit these headers. This is not a gap the ADR created — ADR-032 never shipped — but dormanting ADR-032 without noting whether the loop surface will provide any equivalent served-by signal forecloses a future implementation that the corpus previously committed to for Population A (the configuration-dishonesty degradation signal from Cline #10551 / OpenCode #20859).

### Question 3: What would change if the dominant framing were inverted?

The ADR's dominant framing: "the multi-capability fan-out was the original vision the cycle pivoted away from, so retiring the pipeline is pivot-consistent, not a loss."

**Inverted framing:** "The pipeline's Q&A routing capability was the cycle's original value proposition for non-agentic clients; retiring it without a replacement narrows the system to tool-driven clients and leaves non-agentic API consumers with inline seat-filler text only."

Under the inverted framing:

- The seat-filler's inline text answer is not equivalent to an ensemble-backed answer for a Population A client sending a capability-matched NL question without tools. The cost-distribution accountability sub-promise (ADR-032 §Context) was specifically grounded in the concern that capability-matched requests should dispatch to ensembles. Under the collapsed surface with Resolution A, no-tools capability-matched requests now receive inline text — which was the failure mode ADR-027 was designed to remediate.
- The claim that "this is consistent with the pivot" is doing significant load-bearing work. The pivot is from the pivot *away from the fan-out as north star*, not necessarily from fan-out as a useful fallback. ADR-043 does not distinguish between "we pivoted away from this as the primary loop" and "we decided this capability has no value."
- Under the inverted framing, ADR-032's configuration-honesty promise is also affected: a no-tools capability-matched request answered by the seat-filler inline would presumably emit `served_by: "direct"` (if that header existed), misleading the client into thinking no ensemble was available when in fact one was available but not offered.

This is the framing that Resolution B addresses — the ADR records it as a "documented future option," which is appropriate. But the framing audit surfaces it here so the practitioner can confirm at the gate that the pivot away from fan-out genuinely covers this case, not just the multi-step composition case.

### Framing Issues

**P2-F1. The Resolution B rejection is asserted via the determinism principle and the pivot, but "the pivot" is not traced to a practitioner decision artifact.**

- **Location:** §Rejected alternatives: F-ι.1 Resolution B ("Rejected as the default on the determinism principle and the pivot")
- **Claim:** Resolution B is rejected because of (a) the determinism principle for essential routing and (b) the cycle's north-star pivot. The determinism argument is well-grounded (the corpus records the determinism-over-carve-outs principle, the spike measured 3/10 broken delegations, Resolution A closes the gap). But the "the pivot" is referenced as though it is a settled corpus fact with a specific locus.
- **Issue:** The pivot is a framing that has accumulated across cycle-status handoffs and product-discovery notes, but no single ADR or committed artifact records "we decided not to pursue ensemble-backed Q&A for non-tool clients." It is an emergent corpus consensus, not a named decision. A reader six months from now trying to trace whether the cycle ever explicitly decided against no-tools ensemble-backed answers will not find a clean decision record. The framing audit surfaces this not as a logical error but as a provenance gap: the pivot grounds the rejection of Resolution B, but the pivot's own grounding is diffuse.
- **Recommendation:** In the provenance check, name the specific artifact(s) that establish the north-star pivot as a practitioner decision rather than an emergent assumption. The cycle-status handoff (2026-06-18, open item #5's "the original answer-a-question vision the loop-back pivoted from") plus product-discovery.md's Cycle 7 update to Tension 14 are the closest sources. Either cite them explicitly in the Resolution B rejection, or add a provenance entry for "the pivot" alongside the Spike ι and ADR-033 entries.

**P3-F1. The dormancy sweep's handling of ADR-031 and ADR-032 could foreclose Configuration Honesty for the loop surface.**

- **Location:** §Decision 2 ("ADR-028/029/031/032 loses its production caller — disposition per the backward-propagation sweep (dormant-with-dated-note)")
- **Claim:** All four ADRs go dormant without a distinction between those that were purely pipeline-mechanism specs (028, 029) and those that codified cross-cutting commitments (031 latency, 032 configuration honesty / transparent-endpoint promise).
- **Issue:** ADR-032's configuration-honesty commitment was a direct response to Population A voice (Cline #10551, OpenCode #20859). Making ADR-032 dormant without a note about whether its principles carry forward to the loop surface risks silently dropping a commitment the corpus derived from real user pain. This is P3 because the commitment never shipped in code and the loop surface's honest-response-labeling question was deferred to BUILD anyway — but the dormancy framing could make it harder to pick up in a future cycle.
- **Recommendation:** In the backward-propagation sweep note on ADR-032, record that its configuration-honesty principle (honest served-by labeling) is not foreclosed by dormancy — it is a BUILD-open question for the loop surface. One sentence in ADR-032's dated dormancy header suffices.

---

## Convergence-Saturation Signal (ADR-094)

**Convergence-Saturation Signal:** NOT TRIGGERED

- Round number: R1
- P1 count this round: 0 (Section 1: none; Section 2: none)
- P2 count this round (new, non-carry-over): 3 (P2-1 — ADR-097 attribution error; P2-2 — dormancy reasoning gap for ADR-031/032; P2-F1 — Resolution B pivot provenance gap). All new, no prior rounds.
- New framings or claim-scope expansions: the "pivot forecloses no-tools ensemble access" framing (Question 3 above) surfaces a claim-scope characterization not explicit in ADR-043 — that Resolution A's gate is not merely a determinism improvement but a functional narrowing relative to the original ADR-027 cost-distribution promise. Named here for the first time.
- Recommendation: **CONTINUE to R2.** P1 count = 0 but P2 count = 3 (exceeds the ≤1 threshold), and one new framing emerged. Signal does not trigger.

*Single-purpose re-audits (dispatched per the re-audit-after-revision rule) omit this section. Form-change events reset the round-count baseline — the first audit on a new form is its R1.*

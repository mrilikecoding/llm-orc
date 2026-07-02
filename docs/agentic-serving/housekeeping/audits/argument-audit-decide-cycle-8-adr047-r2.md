# Argument Audit Report — R2 (Re-audit After Revision)

**Audited document:** `docs/agentic-serving/decisions/adr-047-serving-extensibility-registry-and-shape-catalog.md` (revised 2026-07-02, incorporating R1)
**Predecessor:** `housekeeping/audits/argument-audit-decide-cycle-8-adr047.md` (R1: 3 P1, 6 P2 new, 2 P3; verdict CONTINUE)
**Source material re-verified:** `docs/agentic-serving/decisions/adr-046-target-architecture-per-turn-handler-one-ensemble.md` (full body, incl. §2 seat-contract caveat and §Open); `docs/agentic-serving/domain-model.md` (AS-2 line 194, AS-5 line 196, AS-6 line 197 dispositions + Amendment Log #23; the 2026-07-02 AS-6 forward note); `docs/agentic-serving/essays/research-logs/cycle-8-spike-runtime-composition-feasibility.md` (full body); `~/.claude/skills/llm-conductor/docs/essays/004-purpose-built-ensembles-and-the-design-laboratory.md` (full body); `docs/agentic-serving/housekeeping/cycle-status.md` (Settled Premises section, Q4 grounding-inputs paragraph); git diff of the ADR-047 revision against commit `07b80ec`
**Genre:** ADR
**Date:** 2026-07-02 (R2)
**Scope:** Re-audit after revision per dispatch brief — verify each R1 finding (P1-1..3, P2-1..4, the two Section-2 framing P2s, P3-1/P3-2) was actually addressed; sweep for any new issue the edits introduced (with particular attention to the R1→R2 pendulum failure the ADR-046 audit chain documented); re-run Convergence-Saturation as R2.

## Section 1: Argument Audit

### Summary

- **Genre:** ADR
- **Argument chains re-examined:** the same 6 chains R1 mapped, with focused re-derivation of §2 (binding-time / Strategy-A label), §5 + §Deferred(b) (AS-2 scope and seat-contract caveat), §Deferred(c) (the compose-step capability-tier bet), and §Rejected (the Strategy-B note and the frontier-default split) — the four locations the revision touched.
- **R1 fix verification:** all 3 P1s and all 4 Section-1 P2s land cleanly, verified against the ADR-046/domain-model/essay-004 text each finding cited. 1 of 2 Section-1 P3s also lands (P3-2, bonus — not required). The 2 Section-2 framing P2s and 1 Section-2 framing P3 are **not addressed**; they carry forward unresolved (all were "Should Fix" / "Consider," not gating).
- **Issues found this round:** 0 P1, 0 new P2 (2 carried over, unresolved), 3 new P3 (2 carried over, unresolved).
- **Pyramid coverage map:** N/A (ADR genre, not Essay-Outline).
- **Expansion-fidelity findings:** N/A (ADR genre, not Essay-Outline).

### R1 fix verification

1. **P1-1 (Consequences vs. Provenance contradiction on "evidence-grounded") — landed cleanly.** The Consequences bullet now reads: "the compose-at-runtime primitive has a grounded spec (the feasibility spike's four named gaps), while the composer-ensemble strategy pillars remain a named, un-grounded hypothesis (Settled Premise #4)" (line 136) — this is the exact split the R1 recommendation asked for (primitive-spec grounded, strategy pillars not), and it now agrees word-for-word in substance with the Provenance check's "deferred and un-grounded" (line 151). No residual contradiction.
2. **P1-2 (pillar (c) / Rejected-bullet overreach of essay-004 into the composition/design task) — landed cleanly, and hits the practitioner's specified target exactly.** Pillar (c) now states essay-004's "cited evidence backs ensemble-first for *routing/selection*; it assigns *design* work... to the more capable tier and reports mixed-model synthesis underperforming on open-ended generation. So the baseline and benchmark for the compose step is a capable-model-composed structure, and ensemble-decomposition of that step is the standing-principle bet to *validate* against the baseline at BUILD/PLAY: held open, not ruled out" (lines 90-97). The §Rejected split does the same work from the other direction: "Frontier-model default for routing/selection steps. Rejected..." / "For composition/design steps this rejection does not hold on the evidence... there a capable-model-composed structure is the baseline and benchmark, and ensemble-first-for-composition is the open hypothesis... not a settled rejection" (lines 121-128). Checked against the practitioner's stated resolution target (stop claiming essay-004 *supports* ensemble-first-for-composition; mark it an open hypothesis with a capable-model-composed baseline, deferred to BUILD/PLAY, not ruled out): this is exactly what the revised text says, not an over-correction into "composition must be frontier-only" (the "not ruled out" / "held open" / "standing-principle bet to validate" language explicitly keeps the door open) and not a retreat into silence on the composer-ensemble direction (pillars (a), (b), (d), (e) are untouched and the deferred path remains named).
3. **P1-3 ("Strategy A does not mean 'a frontier model routes'" contradicting essay-004's own Strategy-A definition) — landed cleanly.** The note is now: "essay-004's validated Strategy A puts a *capable* model (Claude) in the external router seat, and no source it cites tests a small model there. This ADR's position that the external router may itself be an orchestration of bounded small-model roles is a considered extension of essay-004's external-vs-internal placement axis, driven by the standing principle. It is not a claim the cited evidence already establishes" (lines 111-116). This is the R1 recommendation verbatim in substance: the "small models in the router seat" claim is now attributed to the standing principle as a considered extension, not to essay-004's evidence.
4. **P2-1 (AS-2 scope creep to output/content quality in pillar (b)) — landed cleanly.** Pillar (b) now reads "AS-2 validates the composer's output (the newly composed ensemble's reference-graph structure: no cycle, within depth, every reference resolves to an existing entry) before it registers; per-dispatch output quality stays Q2's concern (§5), unchanged for composer-produced ensembles" (lines 84-87) — matches domain-model.md line 194's AS-2 disposition exactly and matches §1's own phrasing, closing the ambiguity R1 flagged.
5. **P2-2 ("(Strategy A)" label on binding-time conflating two independent axes) — landed cleanly.** The §2 heading tag is removed; the body now states "Both load-time curation and turn-time selection keep routing *external* to the capability ensembles: essay-004's Strategy-A criterion is about external-vs-internal placement, not binding time, so both satisfy it. The default is load-time curation + classify-selection because it needs no new primitive, not because turn-time selection would be less 'Strategy A'" (lines 41-45) — the exact reframe R1 recommended.
6. **P2-3 (unspecified escalation-trigger mechanism, risking Strategy-B's self-verification failure mode) — landed cleanly.** Pillar (c)'s closing sentence now states the attempt-then-escalate decision "is gated by (d)'s deterministic checks, never by a small model judging its own output" (lines 96-97) — this closes exactly the gap R1 named.
7. **P2-4 (seat-contract wiring caveat not carried forward from ADR-046 §Open) — landed cleanly.** §5 now reads "A composition is admitted by AS-2 plus its contract (the seat contract ADR-046 §Open still tracks as designed but unwired and unvalidated)" (lines 65-66) — verified against ADR-046 §Open's "Seat-contract wiring — `core/validation/` as the seat's pass/fail gate is designed (F3) but unwired and unvalidated by the spike" (ADR-046 line 85). Exact match.
8. **P3-2 (missing citation of the domain model's AS-6 forward note) — landed, bonus (not required at P3).** The Provenance check's new "Evidence scope" paragraph now closes with: "The domain model's AS-6 disposition (2026-07-02 forward note) is the anchor: what is retired is the orchestrator-LLM `compose_ensemble` actor; runtime composition *as a declarative engine capability* remains a live Q4+ direction" (lines 150-151) — verified against domain-model.md line 197's forward note verbatim. This is the exact cross-reference R1 recommended.
9. **P3-1 (Design→Calibrate→Trust→Promote characterization overstating llm-conductor's autonomy vs. ADR-006's consent-gating) — not addressed.** §Context's characterization is unchanged. Acceptable to leave: P3 severity, and the underlying critique (admission-by-accumulated-signal vs. admission-by-deterministic-contract) holds regardless of the human-gate difference, as R1 itself noted.
10. **Section-2 framing P2 (Q1 framing C — binary rejection of the whole calibration-grown-catalog populator forecloses a calibration-as-advisory-evidence middle path) — not addressed.** The "Calibration-grown catalog" §Rejected bullet is textually unchanged from R1 (verified via diff — no `-`/`+` lines touch it): "Rejected: it is the retired AS-5 trust/promotion machinery wearing a design-laboratory coat. Copy the catalog structure, reject the populator." Still a binary; the middle path (calibration data as *input to* a deterministic-contract decision, per essay-004's own "measured through calibration" language and ADR-006's consent-gated precedent) is still not examined.
11. **Section-2 framing P2 (Q2 finding 2 — essay-004's concrete Verification Layer proposal, the named mechanism for "deterministic verification," is absent) — not addressed.** Pillar (d) is textually unchanged: "acceptance is **deterministic** (contract + verification), never trust-accumulation." Still no reference to essay-004's MiniLM/DeBERTa/log-prob-entropy verification-layer proposal as a candidate mechanism.
12. **Section-2 framing P3 (Q1 framing B — turn-time binding vs. orchestrator-capability) — not addressed, but its P2-2 sibling (the "(Strategy A)" mislabeling) is fixed.** The revised §2 explains *why* load-time is the default (no new primitive needed) but still doesn't examine whether DeepMind's orchestrator-capability-dominates finding bears on the binding-time choice itself. Unaddressed, low stakes (P3).

R1's fix rate on gating findings is 3/3 P1 and 4/4 Section-1 P2, all verified against primary source text rather than taken on the document's own say-so. The two unresolved framing P2s and one framing P3 are pre-existing carry-over, not new — they were "Should Fix" / "Consider" in R1 and remain so; nothing about leaving them unaddressed reopens a P1.

### P1 — Must Fix

None. All three R1 P1s are resolved and no new P1 was introduced by the revision.

### P2 — Should Fix

**P2-R2-1 (carried over from R1, unaddressed). The binary "reject the whole calibration-grown-catalog populator" framing still forecloses the calibration-as-advisory-evidence middle path.**

- **Location:** ADR-047 §Rejected, "Calibration-grown catalog" bullet (unchanged from R1).
- **Claim:** Calibration-driven catalog growth is rejected outright as "the retired AS-5 trust/promotion machinery wearing a design-laboratory coat."
- **Evidence gap:** essay-004 itself doesn't reject calibration data — its complementarity/cascade patterns are "applied when task characteristics warrant it and measured through calibration," with a human/Claude-in-the-loop deciding whether to apply them (§"Architectural complementarity," essay-004), much like ADR-006's consent-gated promotion ("the user decides at every gate"). The binary framing forecloses "calibration as advisory input to a deterministic-contract admission decision" without examining it.
- **Recommendation:** Unchanged from R1 — add a clause distinguishing "calibration data as the autonomous trigger" (rejected, correctly) from "calibration data as one input a human-or-deterministic-contract decision may weigh" (not addressed either way).

**P2-R2-2 (carried over from R1, unaddressed). Pillar (d)'s "deterministic verification" still names no mechanism, though the cited source has a specific, evidence-backed one.**

- **Location:** ADR-047 §Deferred, pillar (d) (unchanged from R1).
- **Claim:** "Acceptance is deterministic (contract + verification), never trust-accumulation."
- **Evidence gap:** essay-004's "Verification Layer" section proposes exactly this: script-agent-hosted classical ML (MiniLM embeddings, DeBERTa NLI, log-probability entropy) as deterministic, non-LLM-judgment quality signals — the concrete answer to "what does deterministic verification actually consist of here" that pillar (d) gestures at without naming.
- **Recommendation:** Unchanged from R1 — name the verification-layer techniques (or a pointer to them) as the candidate mechanism for pillar (d)'s "verification," since the cited source already specifies one.

### P3 — Consider

**P3-R2-1 (new). "Settled Premise #4" is used three times in the revised text (pillar (c), the Consequences bullet, the Provenance check) with no in-document definition or cross-reference.**

- **Location:** ADR-047 §Deferred pillar (c) (line 91), §Consequences (line 136), §Provenance check (line 150).
- **Claim:** Each occurrence cites "(Settled Premise #4)" as the source/label for the ensemble-over-frontier standing principle.
- **Evidence gap:** The *content* of Settled Premise #4 is stated in §Context's "Standing principle" paragraph (lines 19-25), but that paragraph never uses the label "Settled Premise #4," and no other part of ADR-047 defines what "Settled Premise #4" means or where its numbering comes from. It is defined externally, in `housekeeping/cycle-status.md`'s "Settled Premises (carried into Cycle 8)" section. A reader of ADR-047 in isolation (the position an ADR is normally expected to be legible from) cannot resolve the term without that external file. This is a new label introduced by the revision — the pre-revision pillar (c) used the descriptive phrase "the ensemble-over-frontier bet" instead, without the numbered-premise citation.
- **Recommendation:** Either add a one-clause gloss on first use ("Settled Premise #4 — the practitioner's ensemble-over-frontier standing principle, §Context above") or drop the numbered label from the ADR body and keep the descriptive phrase, reserving "Settled Premise #4" for cycle-status.md's own bookkeeping.

**P3-R2-2 (new). The Consequences bullet extends "un-grounded hypothesis" to the whole set of composer-ensemble strategy pillars (a)-(e), though only pillar (c) is framed as a hypothesis in §Deferred.**

- **Location:** ADR-047 §Consequences (line 136): "the composer-ensemble strategy pillars remain a named, un-grounded hypothesis (Settled Premise #4)."
- **Claim:** All five strategy pillars (a)-(e) collectively constitute "a named, un-grounded hypothesis."
- **Evidence gap:** In §Deferred, only pillar (c) (whether the compose step runs as bounded small-model orchestration vs. a single capable process) is framed as an open, unresolved empirical hypothesis. Pillars (a) (compose from registry parts), (b) (AS-2 gates the composer's output), (d) (deterministic acceptance), and (e) (author-time-with-review first) are architectural/governance decisions inherited from already-surviving invariants (AS-2, AS-11, the deterministic-acceptance discipline §5 states elsewhere) — they are un-*built* (nothing in §Deferred is built yet) but not themselves hypotheses awaiting empirical validation the way pillar (c) is. Calling the whole set "a...hypothesis" mildly blurs "not yet built" with "empirically speculative," a distinction the rest of the revision (pillar (c)'s careful hedging) otherwise keeps sharp.
- **Recommendation:** Narrow the Consequences bullet to name pillar (c) specifically as the hypothesis, or reword to "the composer-ensemble strategy pillars remain a named, un-built forward direction (pillar (c)'s compose-step allocation specifically held as an open hypothesis, Settled Premise #4)."

**P3-R2-3 (new, framing). Pillar (c) defers the ensemble-first-for-composition test to "BUILD/PLAY" without pre-registering what would count as a pass or fail, unlike this corpus's own spike convention.**

- **Location:** ADR-047 §Deferred, pillar (c): "the standing-principle bet to *validate* against the baseline at BUILD/PLAY."
- **Claim:** The hypothesis will be validated against a capable-model-composed baseline at BUILD/PLAY.
- **Evidence gap:** This corpus's own spike discipline (e.g., `cycle-8-spike-runtime-composition-feasibility.md`'s explicit PASS/FAIL criteria, or ADR-046's three pre-registered spike criteria) pre-states falsification conditions before running the probe. ADR-047 names the baseline (capable-model-composed structure) but not what comparison or threshold would count as the ensemble-decomposed compose step succeeding or failing against it. This is a DECIDE-time gap that is normal to leave for BUILD to specify (not a logical defect in the ADR), but worth naming so the BUILD/PLAY validation doesn't drift into an unfalsifiable comparison.
- **Recommendation:** No ADR-047 text change required now; note for BUILD/PLAY planning that the compose-step hypothesis needs pre-registered success criteria before the probe runs, matching this corpus's existing spike pattern.

**P3-R2-4 (carried over from R1, unaddressed).** The "Design → Calibrate → Establish → Trust → Promote" characterization of llm-conductor's populator (§Context) still somewhat overstates its autonomy relative to ADR-006's consent-gated promotion ("the user decides at every gate"). Unaddressed; low stakes, since the underlying admission-mechanism critique holds regardless.

**P3-R2-5 (carried over from R1, unaddressed).** The turn-time-binding / orchestrator-capability connection (does DeepMind's "orchestrator capability dominates" finding bear on the load-time-vs-turn-time binding choice, not just the external-vs-internal placement choice) is still not examined, though DeepMind is cited elsewhere in the same ADR. Unaddressed; low stakes.

## Section 2: Framing Audit

### Question 1: What alternative framings did the evidence support?

**R1's Framing A (frontier-first composition as the starting hypothesis, ensemble-decomposition as the tested optimization) is now, in substance, the position ADR-047 itself takes.** Pillar (c) and the split §Rejected bullet adopt almost exactly this framing: capable-model-composed structure as baseline/benchmark, ensemble-decomposition as the standing-principle bet to validate against it at BUILD/PLAY, held open rather than ruled out. This is the single largest framing change in the revision and it resolves the R1 framing P1 (cross-referenced from P1-2) at the content level, not just the wording level.

**R1's Framing B (turn-time binding as co-equal with load-time curation) remains unaddressed** — carried forward as P3-R2-5 above.

**R1's Framing C (calibration as advisory evidence, not a binary gate) remains unaddressed** — carried forward as P2-R2-1 above.

### Question 2: What truths were available but not featured?

Re-checked against essay-004's full text:

- **Essay-004's Opus-for-design/Sonnet-for-orchestration split** — now substantively present: pillar (c) states essay-004 "assigns *design* work (choosing DAG shapes, composing from parts) to the more capable tier." **Resolved.**
- **Essay-004's self-MoA-vs-mixed-MoA finding** (6.6% degradation on open-ended generation) — now substantively present: pillar (c) states essay-004 "reports mixed-model synthesis underperforming on open-ended generation." **Resolved.**
- **Essay-004's concrete Verification Layer proposal** (MiniLM/DeBERTa/log-prob entropy) — still absent. **Unresolved** (P2-R2-2).
- **Essay-004's correlated-error caveat** (60% error agreement vs. 33% random baseline) — still absent. Not independently severity-tagged in R1's Framing Issues list either; noting for completeness, not as a new gating finding.

### Question 3: What would change if the dominant framing were inverted?

**R1's inversion (routing/selection and composition/design as distinct capability classes, with frontier-first correct for the latter) is now the ADR's own stated position for §Deferred pillar (c) and the split §Rejected bullet.** Re-running the inversion against the *current* text: the near-term decision (§1-§3, registry/catalog/classify-decider binding) remains well-grounded and untouched by any of this round's findings — it was never in question. The remaining friction is narrower than R1's: not "does the ADR conflate routing and composition evidence" (fixed) but "does the compose-step hypothesis have pre-registered validation criteria the way this corpus's other hypotheses do" (P3-R2-3) and "is the calibration-populator rejection a false binary" (P2-R2-1, carried over).

### Framing Issues

**P1 (Must Fix):** None this round. R1's sole framing P1 (cross-referenced from P1-2) is resolved.

**P2 (Should Fix, carried over from R1):**
- Q1 framing C / P2-R2-1 — the binary calibration-populator rejection still forecloses the advisory-evidence middle path.
- Q2 finding 2 / P2-R2-2 — the missing verification-layer mechanism is still a named, concrete, unused answer to pillar (d).

**P3 (Consider):**
- P3-R2-3 (new) — pillar (c)'s BUILD/PLAY validation lacks pre-registered success criteria, unlike this corpus's spike convention.
- Q1 framing B / P3-R2-5 (carried over) — turn-time-binding / orchestrator-capability connection still unexamined.

## Convergence-Saturation Signal (ADR-094)

**Convergence-Saturation Signal:** TRIGGERED
- Round number: R2. Per ADR-094's form-change baseline-reset rule, this round does not reset the baseline — the revision is an in-place edit of the same document (targeted rewrites of §2, §5, §Deferred(b)/(c), §Rejected, §Consequences, §Provenance), not a restructuring, consolidation, or replacement into a new form. R1 stands as this document's R1; this is its R2.
- P1 count this round: 0 (all 3 of R1's P1s verified resolved against primary source text; no new P1 introduced by the revision).
- P2 count this round (new, non-carry-over): 0. Two Section-2 framing P2s carry over unresolved from R1 (P2-R2-1, P2-R2-2) — neither is new this round, and per ADR-094's counting rule only new, non-carry-over P2s count toward the threshold.
- New framings or claim-scope expansions: none that materially shift the argument beyond what R1 already named. The three new P3 findings this round (the undefined "Settled Premise #4" label; the Consequences bullet's "hypothesis" framing over-extended from pillar (c) to all five pillars; the missing pre-registered BUILD/PLAY validation criteria for the compose-step bet) are narrow textual-precision and process-completeness observations, not new alternative framings or new claim-scope characterizations — they don't introduce a reading of the evidence or a claim boundary that R1's findings didn't already anchor. R1's own Framing A/B/C and the Q3 inversion remain the operative framings; A is now adopted into the ADR's own text, B and C remain open and unchanged.
- Recommendation: **STOP at this round.** All three conditions hold: P1 = 0, new P2 = 0 (≤ 1), and no new framings or claim-scope expansions this round. The P1 trend across the ADR-047 track is 3 → 0, converging in one revision pass — the revision hit the practitioner's specified target precisely (essay-004's evidence is now scoped to routing/selection only; ensemble-first-for-composition is held as an open, deferred, not-ruled-out standing-principle hypothesis with a capable-model-composed baseline) without the R1→R2 pendulum failure the ADR-046 audit chain warned against: no text asserts composition must be frontier-only, and the composer-ensemble direction remains named and intact. The two carried-over framing P2s (calibration-as-advisory middle path; the unnamed verification-layer mechanism) and the handful of P3s are Should-Fix/Consider items suitable for the practitioner's judgment at BUILD or a future light pass — none is gating, and per the signal's own design, unresolved lower-severity carry-over items do not block convergence. If the practitioner wants the two P2s closed before Acceptance, that is a bounded, low-risk follow-up (each has a one-clause fix named above); it does not require another full audit round.

*This is R2 of the standard audit sequence (per the dispatch brief's explicit instruction to re-run Convergence-Saturation as R2); the verdict line above is required and included, matching the ADR-046 audit-chain's practice of appending a per-round verdict across its own R1-R4 track.*

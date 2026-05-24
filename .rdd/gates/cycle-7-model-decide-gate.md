# Gate Reflection: Cycle 7 — Cross-Compatibility Routing-Surface Architecture; model → decide

**Date:** 2026-05-22
**Phase boundary:** model → decide
**Cycle:** Cycle 7 — Framework-driven orchestration: routing as code (provisional title) / Cross-Compatibility Routing-Surface Architecture (Essay-Outline 006)

## Belief-mapping question composed for this gate

Composed against MODEL's most consequential decision: the non-codification of OQ #16 (candidate new invariant — "the dispatch path is structurally-bounded role-driven, not orchestrator-LLM-driven") as an invariant amendment at MODEL boundary. The question referenced the Cycle 7 DISCOVER susceptibility-snapshot's "rapid compounding" signature and Advisory 2's call for the build-complexity comparison before codification:

> The MODEL deferral rests on the snapshot's structural finding: the combined test coverage of the three DISCOVER spikes (ζ + ε + κ; tightened by ε') is narrower than the architectural commitment they support, even though each individual spike's finding is sound. If MODEL had codified "the dispatch path is structurally-bounded role-driven, not orchestrator-LLM-driven" as a new AS-9 invariant **now**, downstream ADRs and code would be constrained by it before the build-complexity comparison (snapshot Advisory 2, OQ #19) is produced.
>
> **What would you need to believe for codifying-at-MODEL to be the right move, and the snapshot's deferral advisory to be over-cautious?** Some candidates to map against:
>
> - The snapshot's "rapid compounding" pattern is named correctly but does not warrant an invariant deferral — the combined spike coverage is sufficient to lock the structural finding as a constitutional rule.
> - The build-complexity comparison (Advisory 2) is DECIDE work but is not load-bearing for the invariant codification — even if Tier 1 hybrid turns out to be cheaper to build, the structural advantage of removing the orchestrator-LLM from the routing-decision and post-dispatch-synthesis surfaces is established regardless.
> - The candidate invariant (OQ #16) does not actually constrain downstream artifacts in a way that requires waiting — DECIDE's ADRs would arrive at the same place whether the invariant is in place at MODEL or codified at DECIDE.
> - The scope-of-claim partition embedded in the Methodology Vocabulary entry is the *de facto* constitutional commitment — promoting it from "direction" to "invariant" is cosmetic, not consequential.
> - The deferral introduces process drag that costs more than the additional examination buys — by the time DECIDE produces Advisory 2's comparison, the routing-planner + response-synthesizer mechanisms will already be in BUILD-shape work.

The question used the belief-mapping form (mapping the belief space rather than arguing a position), referenced the specific OQ #16 deferral choice + the snapshot's Advisory 2 + the rapid-compounding signature, and named five candidate belief-shapes that would have to be true for the deferral to be over-cautious.

## User's response

The practitioner did not map against the listed candidate beliefs. Instead, the practitioner asked the **generative redirect**:

> "All seems fine. For the belief mapping — my question is whether there is anything additional we could design in a new spike that would inform whether to land an invariant."

The practitioner reframed the gate's pedagogical work from deferral-validation (was MODEL right to defer?) to spike-design (what additional empirical work would close the deferral cleanly?). This is itself a belief-mapping move — the practitioner asked the agent to identify what empirical work would close the audit-depth gap the snapshot named, rather than arguing whether the existing gap was acceptable.

The agent responded by proposing three spike candidates with explicit epistemic targets per OQ item:

- **Spike μ — Confabulation-mode generalization** (load-bearing for OQ #16): test whether Spike ε ε.1's structural-bounding finding generalizes from PLAY note 22 to three other documented confabulation modes (path hallucination per Cycle 6 PLAY note 23; substrate-path-as-deliverable per Spike λ.4-paid / λ.5-paid; coherent factual errors uncalibrated per Cycle 5 PLAY). Free-tier; ~45-60 min.
- **Spike ν — Production-scale capability-match**: informative for OQ #17 / OQ #25 but less load-bearing for codification. Free-tier; ~60-90 min.
- **Spike ξ — Cheap-tier model substitution**: informative for DECIDE design, not load-bearing for invariant codification.

The agent also explicitly named Advisory 2's build-complexity comparison as **estimation work, not spike work** — independently load-bearing for the ADR-027-as-primary commitment but unaffected by any spike.

The practitioner authorized Spike μ:

> "I think more information earlier is better, so if there's an important thing to learn from a spike before we propagate forward I'm in favor."

Spike μ was designed, run, and scored at $0 cost in ~3 minutes wall-clock. Findings:

- **μ.1 path hallucination**: structural-bounding **transforms** the failure mode (confident-specific fabrication → honest-generic-conventions with explicit hedging + uncertainty acknowledgment + clarification request). Pattern detector flagged a false positive (any backtick-wrapped `.py` reference counted as fabrication regardless of framing); qualitative analysis showed the response was qualitatively different from PLAY note 23's failure mode. New DECIDE-phase design question surfaced (Rule 6 candidate for framework-convention enumeration in direct-completion mode).
- **μ.2 substrate-path-as-deliverable**: **structurally avoided** by text-only synthesizer surface. The mode requires tool-call emission capability the synthesizer ensemble does not have.
- **μ.3 coherent factual errors uncalibrated**: **bounded by Rule 1 + Rule 5**. Synthesizer cited fixture's single figure verbatim, explicitly named the partial-source limitation, refused to fabricate growth rates.

Combined audit depth on the structural-bounding claim grew from "PLAY note 22 + 9 Spike ε/ε' tests" to "4 documented confabulation modes + 13 tests total" at qwen3:8b.

The agent presented three codification options:

(a) Codify as new AS-9 at MODEL — supported by Spike μ + ε + ε' cross-mode coverage; ADR-027-as-primary commitment remains deferred to DECIDE per Advisory 2.

(b) Codify with narrower scope (only the 4+1 tested modes) — tighter scope-of-claim; less downstream constraint but more conservative.

(c) Continue to defer to DECIDE — keep OQ #16 + OQ #17 as flagged candidates.

The practitioner chose **option (a) — codify as new AS-9 at MODEL**.

## Pedagogical move selected

**Challenge** — belief-mapping question on the OQ #16 deferral with 5 candidate belief-shapes mapped against the question.

The practitioner's generative redirect ("what additional spike would inform whether to land an invariant?") shifted the gate's pedagogical work from agent-composed challenge to practitioner-initiated generative redirect. The agent's responsive move was to **honor the question by identifying empirical work** with explicit epistemic targets per OQ item, naming Spike μ as load-bearing and Spikes ν / ξ as informative-but-not-load-bearing, and explicitly noting that Advisory 2's build-complexity comparison is estimation work distinct from spike work and unaffected by any spike outcome. The practitioner-authorized Spike μ produced findings extending the structural-bounding claim's empirical coverage across four documented confabulation modes (n=13 tests total), after which the agent presented three codification options and the practitioner selected codification as new AS-9.

The gate's substantive epistemic work occurred across multiple exchanges: gate question composed → practitioner generative redirect → spike candidates proposed with epistemic targets → practitioner authorizes Spike μ → spike run + findings analyzed (including the μ.1 false-positive qualitative reinterpretation) → codification options presented → practitioner selects codify-as-AS-9 → AS-9 text drafted with explicit separation of structural property (codified) from mechanism choice (still deferred via OQ #19 / Advisory 2).

## Commitment gating outputs

**Settled premises (the user is building on these going into DECIDE):**

1. **AS-9 is codified.** "Structurally-bounded LLM roles produce reliable output on single-decision-shaped tasks where the orchestrator-LLM-as-decider failed." Empirical basis: 4 documented confabulation modes + 13 tests at qwen3:8b (Spike ε + Spike ε' + Spike μ). Scope: structural property (role-shape, not model-shape) independent of mechanism choice.
2. **Mechanism choice remains DECIDE work.** AS-9 does not codify ADR-027 framework-driven dispatch pipeline as PRIMARY mechanism; that commitment remains DECIDE-phase work gated by Advisory 2's build-complexity comparison (OQ #19). Both candidate mechanisms (framework-driven dispatch pipeline; Tier 1 hybrid) produce structurally-bounded roles per AS-9.
3. **Spike μ extends the empirical coverage of the structural-bounding finding** across path hallucination (transforms via Rule 3 + Rule 5), substrate-path-as-deliverable (structurally avoided by text-only synthesizer surface), and coherent factual errors uncalibrated (bounded by Rule 1 + Rule 5).
4. **One new DECIDE-phase design question surfaced** from Spike μ: Rule 6 candidate for framework-convention enumeration in direct-completion mode (synthesizer enumerated `routes.py`, `api.py`, `endpoints.py` as generic conventions with explicit hedging + uncertainty framing; whether this is desirable presentation or warrants tighter rule is DECIDE-phase synthesizer-rule design).
5. **Plausible-but-untested boundaries explicit in AS-9 text** as evidence-strength qualifiers (not scope limitations): production-scale numerical content broader than Spike ε' B1's 25 figures; cheap-tier reliability for direct-completion-of-factual-questions in training-data-error-prone domains; coherent factual errors on direct-completion path under adversarial pressure; generalization beyond qwen3:8b.

**Open questions (the user is holding these open going into DECIDE):**

1. **Mechanism choice (OQ #19 — Advisory 2 build-complexity comparison).** Framework-driven dispatch pipeline (ADR-027 direction) vs. Tier 1 hybrid (`tool_choice` interception + framework-driven composition continuation). DECIDE produces the explicit sprint-effort comparison before ADR-drafting locks the PRIMARY designation.
2. **OQ #17 — candidate new invariant ("Capability matching works from request content alone with no client-side opt-in").** Distinct from AS-9 — AS-9 names the role-shape property; OQ #17 names the request-shape commitment. Pending DECIDE codification.
3. **OQ #18 — cost-distribution lens validation against Population A voice (Advisory 1).** DECIDE examines whether Tension 18's strict-dispatch disposition is justified by Population A's trust contract (user-voice grounding) or by the project's value proposition (project-developer-lens grounding).
4. **OQ #20 — Population A tool-family timeout research (Advisory 3).** DECIDE's latency ADR includes OpenCode/Cursor/Cline timeout-default research; if sub-40s for non-streaming, the pipeline's current latency floor breaches the transparent-endpoint promise.
5. **OQ #21 — Multi-step composition mechanism.** DECIDE design question per Spike ε ε.6.
6. **OQ #23 — Rule 5 framing requirement scope** (Spike ε' Finding ε'.1).
7. **OQ #24 — Rounding-drift mitigation playbook** (two characterized drift modes per Spike ε' Finding ε'.2).
8. **NEW: OQ #26 (to be opened by DECIDE) — Rule 6 candidate for framework-convention enumeration in direct-completion mode** (Spike μ.1 surfaced; whether the synthesizer should be forbidden from enumerating framework-convention file names without uncertainty framing).

**Specific commitments carried forward to DECIDE:**

- Vocabulary settled at MODEL (10 DISCOVER-vintage candidates per Amendment Log entry #12) feeds DECIDE's ADR drafting as binding terminology. Routing-planner ensemble, response-synthesizer ensemble, Population A, Population B, transparent OpenAI-compatible endpoint, cost-distribution lens (with project-developer-lens vs. user-lens distinction preserved), `tool_choice` strip-at-input, framework-driven dispatch pipeline, structurally-bounded role, orchestrator-designs-ensembles (north-star context).
- **AS-9 binds DECIDE artifacts.** ADRs drafted after this gate must use AS-9's structural-property framing; mechanism-choice ADRs (ADR-027 candidate; potential hybrid candidate; `tool_choice` disposition per Tension 19) draft within AS-9's scope.
- **AS-9 propagation tasks for DECIDE/BUILD:** examine ADR-021 (per-capability dispatch contract — actor shift for routing decision); update ADR-022 (Routing surface behavior) to reflect AS-9 + ADR-027 remediation; code references to "orchestrator" / "orchestrator-LLM" need BUILD-phase clarification (the orchestration substrate is the framework-driven dispatch pipeline; the LLM roles are the routing-planner ensemble and response-synthesizer ensemble).
- **Essay-Outline Amendment Log (A1-A4) editing work** carries into DECIDE per the Amendment A4 specification (Abstract + Argument-Graph propagation; argument-audit re-run before DECIDE proceeds beyond the corrected Essay-Outline). The "PRIMARY direction" language at §C7 is now compatible with AS-9 codification — AS-9 names the structural property; ADR-027 names the mechanism the Essay-Outline commits to as PRIMARY direction pending DECIDE deliberation per Advisory 2.
- **Spike μ writeup retained** at `docs/agentic-serving/essays/research-logs/cycle-7-spike-mu-confabulation-generalization.md` per `feedback_spike_artifact_retention` directive; scratch artifacts at `scratch/spike-mu-confabulation/` retained until corpus close.

## Snapshot summary

**Verdict: No Grounding Reframe warranted. Two new advisory carry-forwards + three carried forward from DISCOVER (unchanged).**

Snapshot writeup: `docs/agentic-serving/housekeeping/audits/susceptibility-snapshot-cycle-7-model.md`.

**Key finding:** The canonical failure mode (two exchanges; preference precedes analysis; alternatives not engaged at comparable depth) is **absent**. The MODEL sequence had six distinct exchanges with a spike execution between the practitioner's redirect and the codification decision. The AS-9 structural-property / mechanism-choice separation is **substantive, not rhetorical** — both candidate mechanisms (ADR-027 framework-driven pipeline; Tier 1 hybrid) satisfy the invariant, so AS-9 genuinely does not pre-answer the mechanism question DECIDE will deliberate.

**Two new advisory carry-forwards:**

- **Advisory A (spike methodology — pre-specify qualitative criteria):** Future spike work testing the structural-bounding claim against additional failure modes should document qualitative failure-mode criteria before running tests, not only after a pattern-detector flags a fail. The μ.1 reinterpretation was correct; the procedural gap is that correctness depends on documentation produced after the detector result was known. Spike hygiene, not signal of distorted findings.
- **Advisory B (options-engagement documentation):** DECIDE should not inherit AS-9 as if option (c) (deferral) was examined and rejected at MODEL. It was not: option (a) was selected from a menu that included a genuine deferral option, and no "why not option (c)" rationale was documented. DECIDE should treat AS-9 as correctly codified given the evidence, while holding the amendment pathway open: if ADR deliberation surfaces reasons the structural-property claim is narrower or needs scoping (e.g., confabulation modes were fixture-based at n=1 per mode), the domain model supports amendment.

**Three carry-forwards inherited from DISCOVER, unchanged and active:**

- Advisory 1 (cost-distribution lens validation against Population A voice) → OQ #18
- Advisory 2 (build-complexity comparison between Tier 1 hybrid and ADR-027-direct, per GT-2(a)) → OQ #19
- Advisory 3 (Population A tool-family timeout research) → OQ #20

**Positive signal noted:** OQ #17 (capability matching from request content alone; no client-side opt-in) was correctly preserved as distinct from AS-9 and DECIDE-deferred. The distinction between AS-9's structural property (LLM-role bounding) and OQ #17's request-shape commitment is preserved in the domain model.

**Dispatch-brief screening verdicts:**

1. Practitioner's spike-design redirect: **substantive epistemic move, not bypass.** Engaged the central question (sufficient evidence for invariant-level codification) through a different path; the belief-mapping question's catalyzing function was honored.
2. AS-9 codification in "two exchanges" pattern: **absent.** Six distinct exchanges with spike execution between redirect and codification; preference (redirect question) preceded spike, not codification.
3. Alternatives at comparable depth: **mostly yes; one gap (Finding 2 / Advisory B above).** Option list was not a straw-man menu, but agent's Finding μ.4 framing pre-argued for option (a); option (c) deferral case not documented at selection time.
4. Structural-property / mechanism-choice separation: **substantive.** Both candidate mechanisms satisfy AS-9; mechanism choice remains genuinely open via Advisory 2.
5. μ.1 qualitative reinterpretation: **substantive epistemic work; procedural documentation gap.** Five-property comparison criteria derived from pre-existing PLAY note 23 documentation; self-critical methodological observation in spike writeup is inconsistent with motivated post-hoc justification. Future spike work should pre-specify qualitative criteria explicitly.

The snapshot does not block DECIDE phase progression. Carry-forwards are routed via feed-forward signals, not Grounding Reframe protocol — none meet the three-significance-property test (specific + actionable + in-cycle applicable) for in-cycle reframing; all are correctly downstream-applicable.

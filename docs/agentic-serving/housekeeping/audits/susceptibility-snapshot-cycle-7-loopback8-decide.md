# Susceptibility Snapshot

**Phase evaluated:** DECIDE — Cycle 7 loop-back #8
**Artifact produced:** ADR-041 (Destination-Validity Gate); downstream blocks (domain-model amendment, scenarios, interaction-specs)
**Date:** 2026-06-11

---

## Observed Signals

| Signal | Strength | Trajectory | Notes |
|--------|----------|------------|-------|
| Assertion density | Ambiguous | Stable relative to prior loop-back DECIDE snapshots | The ADR's claim density is high, but the Provenance check maps each assertion to a driver finding; density reflects documented evidence, not unchecked proliferation. The argument audit's two-round structure confirms assertions were tested for evidential grounding. |
| Solution-space narrowing | Ambiguous | Stable | The practitioner's "tackle both seams, no carve-out" framing entered as a constraint, not a conclusion the agent synthesized. The corpus arm tested four gate shapes (pass-through, parse-check, fence-only, marker-detection); only parse-check was viable by the pre-registered criteria. Narrowing to parse-check was outcome-driven, not framing-driven. Two alternative framings from the audit (gate-as-routing-signal-primary; lever-redirection-as-primary) were explicitly held at the gate rather than auto-adopted, which is the right behavior given the dispatch instructions. |
| Framing adoption | Ambiguous | Requires specific decomposition — see Interpretation | The "tackle both seams" directive and the "split claim" framing warrant distinct treatment. See below. |
| Confidence markers | Clear — moderate | Stable | The Status block uses "discharged in principle" and "design-discharge vs. install-discharge" throughout; the Conditional Acceptance block names four pending BUILD items explicitly. The audit flagged two overclaiming instances (P1-1: protection-discharge / install conflation; P2-2: Arm E "closes the residual") and both were corrected. At R2 one fragment remained (P2-C1) and two minor P3 items. No uncorrected P1 findings remain. |
| Alternative engagement | Clear — genuinely engaged | Stable | Five rejected alternatives are treated with specific, evidence-cited rebuttals. The heuristic-gate rejections (fence-only, marker-detection) name the specific corpus failures (C8 false-positive for fence-only; C7b and C8 for marker-detection). The unbounded-retry rejection cites Arm E's 3/6 fresh-8b result. The frontier-seat rejection cites the Arm E coder-vs-seat redirect. These are not pro-forma rejections. |
| Embedded conclusions at artifact-production moments | Ambiguous | Stable | The practitioner's entry framing contained two conclusions that entered the ADR without independent test: "tackle both seams now" and (implicitly) that the adequacy seam is spike-workable before PLAY. The spike pre-registration and methods review treated the central hypothesis as refutable — the falsification paths were registered before any run. This partially mitigates the embedded-conclusion risk but does not fully resolve it (see Interpretation, FF1 assessment). |

---

## Interpretation

### FF1 — Practitioner-originated framing absorption ("tackle both seams, no carve-out")

The practitioner's entry framing directed tackling both seams before PLAY without carving out the adequacy seam ADR-035 disclaims. This framing entered the ADR as a stated driver (§Context: "The practitioner directed tackling both seams rather than carving out the adequacy seam ADR-035's text disclaims"). The question is whether the agent tested this framing independently or absorbed it as a premise.

Assessment: the framing was not fully tested, but the testing that was done is non-trivial. The cross-seam unification hypothesis (one parse-check covers both seams) was pre-registered as refutable with three failure paths (false-positives, miss-set leakage, live non-convergence). The corpus arm ran four gate shapes against a 12-item labeled corpus before any implementation; parse-check's domination of the other candidates is an outcome of that arm, not of the framing. The live arm tested convergence under real trajectory conditions. The practitioner's "both seams now" framing turned out to be grounded — the cross-seam unification held — but the grounding was established by the spike, not assumed at entry. The framing is therefore not a pure adoption; it is a practitioner-directed scope that the spike then evaluated. The residual risk is that the scope itself (spike-workable before PLAY) was not independently interrogated: the agent accepted that the adequacy seam's deterministically-checkable slice was grounded-before-PLAY material without testing whether that characterization of the seam was correct. In practice, the spike corpus's C6 operationalization (the pre-registered semantic-residual boundary) was reviewed under the methods review and is defensible. The FF1 risk here is low but not zero.

### The split-claim framing — agent synthesis or spike-grounded?

The agent proposed the protection-discharged-vs-convergence-conditional split before drafting, characterizing it as a non-overclaiming move. The ADR's Provenance check correctly attributes the split Status to "drafting-time synthesis applying ADR-097 to the cleanly-split live-arm result." The relevant question is whether this framing survives independently of the agent's pre-emptive proposal.

It does. The Fork-3 live arm split the result cleanly by design: 0 invalid files across all 5 gated sessions (protection unambiguous); B=3/5 converged-and-all-valid vs. A=2/5 (convergence margin unmet). The protection-vs-convergence split is a structural reading of that outcome, not an interpretive imposition. The agent did not originate the split — the spike data split it; the agent named the split before the data came in, which means the pre-emptive framing could have biased the interpretation toward two categories even if the evidence had been less clean. The argument audit (R1 P2-1) tested this directly: the audit found the ADR initially misrepresented the pre-registered decision rule, presenting the converged-and-all-valid reading as "the pre-registered outcome" when the literal rule returned "recovers." The correction required the ADR to acknowledge the interpretive step explicitly. After correction, the framing is transparent about its origin (a deliberate reinterpretation of the vacuous B=5/5 reading, grounded in the spirit of the rule). The split claim is spike-grounded with appropriate caveats. The agent's pre-emptive naming is not a sycophancy signal here — it is accurate pre-characterization of what the live-arm evidence structure would have to say if the hypothesis held.

### Rejected-alternative rebuttal quality

The two alternative framings held at the gate (gate-as-routing-signal-primary, lever-redirection-as-primary) were not adopted into the ADR body but were noted as held. This is the appropriate handling given the dispatch instructions — a susceptibility snapshot is not an ADR revision authority. These framings are genuine alternatives that the spike evidence could support and that the ADR does not refute. The routing-signal framing is the more consequential one: ADR-035's "not required" argument rests on the client's permission gate making wrong-form deliverables rejectable, which, if accepted, weakens the workspace-protection framing ADR-041 leads with. The ADR resolves this tension via reconciliation ("not required" was conditional on the gate being available as escalation), but the reconciliation was flagged as "argument-audit-surfaced" in the ADR body itself and "flagged for the gate, where the practitioner can confirm or re-weigh the priority call." That attribution is correct and honest.

### Cross-ADR composition without independent testing

ADR-040's recovery-assumption refutation (smoke finding: refusal-as-`stop` ends the OpenCode loop) is the single most consequential mechanism discovery in this loop-back. This was established by one smoke session on a deterministic code path. The ADR is correct that one session suffices for a structural (not stochastic) finding; the `client_tool_action_terminal.py` code path was read directly. The composition claim (server-side recovery + ADR-040 completeness gate interacting cleanly on recovered runs) was not directly instrumented in the spike log. R2 audit P3-N2 flagged this: the ADR asserts the recovered runs composed with ADR-040 correctly but the spike log evidences session-level convergence, not a traced ADR-040 completeness check on a recovered turn. The ADR now carries this as a BUILD verification item. That is the right disposition.

### Phase-position consideration

DECIDE is mid-gradient in the sycophancy gradient (more vulnerable than BUILD, less than RESEARCH). This DECIDE phase had an unusually strong evidence base — a pre-registered, methods-reviewed spike with live trajectory data — entering the drafting step, which attenuates the gradient risk. The two-round argument audit was conducted by an isolated external auditor with no context from the drafting session, which provides the independent perspective the in-session agent may have lost. The R1 findings (P1-1, P1-2) were not trivial overclaims; they required structural correction. The corrections were applied. At R2, no P1 findings remain.

---

## Recommendation

**No Grounding Reframe warranted for the current BUILD phase.**

The primary susceptibility signal — FF1 framing absorption on the "tackle both seams now" directive — is present but attenuated by the pre-registered refutability discipline, the methods review, and the live-arm outcomes. The split-claim framing is spike-grounded, not agent-synthesis surviving without a driver. Rejected alternatives were engaged with specific evidence, not dismissed pro-forma. Two held framings (gate-as-routing-signal; lever-redirection-as-primary) remain unresolved by the ADR, but both are correctly scoped as practitioner-decision territory and PLAY observation targets — they are not claims the ADR makes; they are claims the ADR could have made but did not.

Two residual items are worth the BUILD entry signal's attention, not as Grounding Reframe candidates but as feed-forward:

1. **The ADR-040 compose-claim on recovered runs (R2 P3-N2)** is the single most verification-sensitive claim in the ADR: the agent asserts that server-side recovery and ADR-040's completeness check compose correctly, but this was not directly instrumented. The BUILD verification item (trace the recovered-case Session Action Record write through ADR-040's `requested-produced` diff) should be treated as a first-in-session check, not an optional cleanup. If the composition fails (e.g., double-recording corrupts ADR-040's `produced` set), the recovery path could produce a session that appears to converge but has stale completeness state.

2. **The held gate-as-routing-signal framing** touches a genuine tension between ADR-035's "not required" argument and ADR-041's protection-primary framing. The reconciliation paragraph is honest ("flagged for the practitioner to confirm or re-weigh the priority call"), but the priority call itself has not been confirmed. BUILD will surface the experiential question when the gate runs live: a cap-exhausted short session is the user-visible protection failure mode, and whether that is a better experience than a broken-file diff (the pre-gate failure mode) is the unresolved PLAY hypothesis the ADR correctly names. BUILD should instrument for both observable outcomes from the first live session — not because the gate is wrong, but because the routing-signal framing may turn out to be the more useful description of the gate's role once experiential data is available.

Neither item is specific enough to block BUILD, and neither constitutes a pattern of assertion-density rising without examination. The phase's strong evidence base and the two-round external audit represent earned confidence, not sycophantic reinforcement.

# Argument Audit Report — Cycle 5 DECIDE (Round 3)

**Audited documents:**
- `docs/agentic-serving/decisions/adr-019-skill-framework-agnostic-capability-library.md`
- `docs/agentic-serving/decisions/adr-021-skill-orchestration-via-per-capability-dispatch.md`

**Round 2 audit read:**
- `docs/agentic-serving/housekeeping/audits/argument-audit-cycle-5-decide-round2.md`

**Date:** 2026-05-12

---

## Purpose

This is a focused verification audit. It confirms whether the four issues raised in round 2 are resolved in the current document state, and checks whether the revision batch introduced any new issues. Round 1 and round 2 issues already verified as resolved are not re-examined.

---

## Section 1: Argument Audit

### Summary

- **Round 2 issues under verification:** 4
- **Resolved:** 4
- **New issues introduced by revision batch:** 0

---

### Verification of Round 2 Issues

#### P2-N1 — ADR-019 §Neutral no-dispatch fallback: "consistent across task types" scoping

**Status: Resolved.**

The revised §Neutral no-dispatch fallback bullet now reads: "the orchestrator's reliability profile (high on derivable claims, low on integration claims, observed within Cycle 4 PLAY's inhabitation-session task range across notes 8, 10, 11, 12, 13, 18; n=1 inhabitation, single orchestrator profile) is the load-bearing property for no-dispatch path quality at the evidence currently available. The 'consistent across task types' characterization is scoped to the Cycle 4 PLAY session — broader task ranges and other orchestrator profiles may reveal exceptions."

The round 2 recommendation was to add a single qualifying sentence acknowledging the n=1 character of the characterization. The revised text does exactly this, and goes slightly further by explicitly noting "n=1 inhabitation, single orchestrator profile" inline and adding "broader task ranges and other orchestrator profiles may reveal exceptions." The claim is no longer presented as an architectural fact; it is presented as a working position from one session. The P2-N1 concern is closed.

---

#### P3-F1 — ADR-019 §Neutral: "consistent across task types" claim and note 3 routing-failure acknowledgment

**Status: Resolved.**

The revised text qualifies the "consistent across task types" phrase inline (now "observed within Cycle 4 PLAY's inhabitation-session task range") and explicitly acknowledges Cycle 4 PLAY note 3 in the same sentence: "Cycle 4 PLAY note 3 already recorded one routing-failure case where the orchestrator misrouted a meta-introspection prompt to a code-generation ensemble." The note 3 routing-failure, which round 2 identified as an available truth that should complicate the consistency claim, is now surfaced in the body of the claim rather than omitted. The P3-F1 concern is closed.

---

#### P3-N1 — ADR-021 §Topaz-skill signal path: `list_ensembles()` analogy

**Status: Resolved.**

The revised natural-language-path justification no longer uses the `list_ensembles()` analogy as its load-bearing claim. The current text reads: "this shape uses LLM judgment for *input-to-ensemble matching*, which is a different judgment task — matching a task description to an ensemble description is closer to retrieval (with the deployment-time `list_ensembles()` result as the retrieval corpus) than to evaluative classification."

The analogy in round 2 was: "structurally analogous to how `list_ensembles()` has always worked." The revised framing replaces this with a retrieval-versus-evaluative-classification distinction. The `list_ensembles()` call is now described as providing the retrieval corpus, not as providing structural analogy for the routing decision itself. The revised justification stands on the retrieval/classification distinction without requiring the reader to accept that operator browsing and routing-time matching are structurally identical. The P3-N1 concern is closed.

The revised framing is independently coherent: input-to-ensemble matching at dispatch time is a pre-dispatch selection decision (the orchestrator's LLM picks which ensemble to invoke) rather than a post-dispatch verdict classification (what ADR-015 §(f) rejected). The distinction holds without the analogy, and the analogy's removal does not weaken the argument.

---

#### P3-N2 — ADR-021 §Seam-case inversion falsification trigger: resolution path completeness

**Status: Resolved.**

The revised falsification trigger subsection now lists three resolution paths explicitly, in the requested order of increasing distance from the agnostic commitment:

1. Parameterized capability ensembles (`argument-mapper(skill_framework=...)`)
2. Per-skill-framework capability ensembles (`rdd-argument-mapper`, etc.)
3. Explicit acceptance that the agnostic commitment was over-broad (operates only at dispatch level, not output-quality level)

The round 2 recommendation was to add parameterized capability ensembles as an intermediate option between per-skill-framework tier defaults (not listed — correctly absent, as that addresses tier-routing divergence, not output-quality divergence) and per-skill-framework capability ensembles. The ADR now carries all three paths, with path (1) explicitly described as "the lightest extension" that preserves the operation-named library principle while admitting skill-framework-conditional behavior. The ordering and the characterization match the recommendation. The P3-N2 concern is closed.

---

### New Issues from Revision Batch

No new logical gaps, internal contradictions, unstated leaps, or scope overreaches were introduced by these four revisions. Each revision is localized: the n=1 qualification in ADR-019 §Neutral does not interact with any other claim in the ADR; the note 3 acknowledgment is consistent with the existing text's epistemic framing; the `list_ensembles()` analogy removal leaves the surrounding argument intact; and the three-path resolution list in ADR-021 extends rather than replaces the prior falsification trigger structure.

---

## Section 2: Framing Audit

### Scope note

The framing audit for this round is constrained to the four revision sites. The broader framing questions from rounds 1 and 2 are not re-examined here.

### Question 1: What alternative framings did the revision batch foreclose or open?

None. The four revisions are epistemic qualifications (scoping claims to their evidence base) and structural completions (adding the third resolution path). They do not shift the dominant framing of either ADR.

### Question 2: What truths were available but not featured, introduced by this batch?

None introduced. The revisions surface truths that were previously understated (note 3's routing failure, the n=1 character of the consistency claim) rather than adding new evidence that could go unrepresented.

### Question 3: What would change if the dominant framing were inverted?

No change from round 2's analysis. The dominant framing's weakest exposure (operation-named principle grounded in one methodology consumer) is unchanged by these revisions. The seam-case falsification trigger's resolution-path completeness improvements make the ADR's treatment of the inverted framing more explicit, not less defensible.

### Framing Issues

None. The revision batch does not introduce new framing concerns, and the concerns carried forward from round 2 (specifically the scoping of "consistent across task types" and the note 3 acknowledgment) are resolved by these revisions.

---

## Conclusion

All four round 2 issues are resolved. The revision batch introduced no new argument or framing issues. The Cycle 5 DECIDE corpus for ADR-019 and ADR-021 is clean at the argument-audit gate as of this round.

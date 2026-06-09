# Argument Audit Report — Round 4 (Scoped Closure Verification)

**Audited document:** `docs/agentic-serving/decisions/adr-039-content-anchor.md`
**Source material:** `docs/agentic-serving/housekeeping/audits/argument-audit-decide-cycle-7-loopback7-round3.md` (R3 findings to verify)
**Genre:** ADR
**Date:** 2026-06-09

---

## Audit context

R4 is a scoped closure-verification round. R3 produced 3 editorial-precision P2s from the prose-scope expansion (P2-1 cross-resolver comparison, P2-2 discharge-gate conflation, FI-P2-1 domain-independence). R3 predicted a clean revision would trigger convergence here. This round verifies each closure, checks the P3-2 ceiling note sits consistently, and performs a regression sweep across the six edits. Re-derivation of the core mechanism, rate data, causal isolation, and gate logic is out of scope.

---

## Section 1: Argument Audit

### Summary

- **Genre:** ADR
- **Argument chains mapped:** scoped — 3 R3 closures + regression sweep
- **Issues found:** 0
- **Pyramid coverage map:** N/A
- **Expansion-fidelity findings:** N/A

---

### R3 closure verdicts

**P2-1 — CLOSED.**

R3 asked for: (a) "crosses resolver types" noted at both comparison points in Context and Decision, and (b) the "prose invents worse, complete-library prior" explanation labeled post-hoc in the Provenance drafting-time synthesis list.

Current ADR text at §Context, fourth bullet: "...resolved 0/10 (every README invented functions — lower than code's 3/10, a comparison that crosses resolver types; the blind READMEs confidently document a whole conversion library, most of it nonexistent)." The causal "because" is gone; the cross-resolver note is inline.

Current ADR text at §Decision, fourth bullet: "...the README generated blind invented functions 0/10 (lower than code's 3/10, though that comparison crosses resolver types — AST for code, a regex heuristic for prose), and the signatures anchor fixed it 10/10." Same repair.

Current ADR text at §Provenance: "the 'prose invents worse, pulling a complete-library prior' reading (a post-hoc characterization of the 0/10 prose baseline, not a separately tested causal claim, and the prose-versus-code baseline comparison crosses resolver types: AST for code, a regex heuristic for prose)."

Both repairs present. The valid finding — prose did invent worse, the 0/10 baseline is real — is retained. The hedging is proportionate: it narrows the epistemic claim without withdrawing the observation.

Not over-hedged.

---

**P2-2 — CLOSED.**

R3 asked for: two distinct sentences separating what the prose arm established (model responds to anchor) from what the discharge gate confirms (framework delivers the anchor to the prose callee).

Current ADR text at §Empirical Grounding: "The README is a discharge criterion, not merely observed: the prose arm established that a prose callee given the signatures documents the real API (0/10→10/10 in the harness, anchor injected directly into the generation). The discharge run confirms the *framework* actually delivers that anchor to `prose-improver` (not only `code-generator`) end-to-end under the real client."

The original conflated phrase "the anchor fires on the prose callee" is gone. Two distinct sentences with distinct subjects: harness-side model behavior, then framework delivery under the real client. The seam is clean.

---

**FI-P2-1 — CLOSED.**

R3 asked for: a note in the Empirical Grounding that Base P reuses converters.py's domain, so the three bases provide callee-type independence, not three-domain independence.

Current ADR text at §Empirical Grounding: "The three bases give two-code-domain independence (converters + text_tools) plus callee-type independence (code + prose); Base P reuses converters.py's domain, so this is callee-type independence, not a third independent domain."

Precisely the correction called for. The callee-agnostic claim in Context and Decision is about callee type; the Empirical Grounding now matches that accurately.

---

### P3-2 ceiling note — sitting consistently

Current ADR text at §Decision, first bullet (Form): "The prose arm (Base P, README) confirms signatures suffice for a prose callee too: B_signatures and C_full both resolved 10/10, so the API surface plus docstrings is enough for the README to document the real functions, and no prose-specific richer form is needed. (B and C both hit the 10/10 ceiling on this README task, so a harder prose deliverable could still separate them; the fallback to full-content for prose is recorded.)"

The ceiling note is placed naturally, qualifies the sufficiency inference without withdrawing the form-selection conclusion (which rests on the pre-registered "no measured gap" rule), and cross-references the fallback already recorded in Rejected Alternatives. Consistent with the Rejected Alternatives section's n=10 ceiling caveat for code. No inconsistency.

---

### Six-edit regression check

The six edits: (1) Context cross-resolver inline note; (2) Decision cross-resolver inline note; (3) Decision discharge-gate rephrasing into two sentences; (4) Provenance post-hoc label for "complete-library prior"; (5) Empirical Grounding domain-independence note; (6) Form bullet ceiling parenthetical.

Checked for mutual contradiction and contradiction against untouched sections:

- Context and Decision cross-resolver notes are consistent with each other and with the Provenance label.
- The Provenance "post-hoc" label and the body text are consistent: Context and Decision no longer assert "because"; they retain the characterization as observational description consistent with the data.
- The Empirical Grounding callee-type note does not contradict the Decision and Context callee-agnostic claims (those claims are about callee type throughout).
- The Form bullet ceiling parenthetical is additive to the Rejected Alternatives fallback note; the two are consistent.
- The discharge-gate two-sentence structure is consistent with the Consequences/Negative note ("the real-client confirmation is the discharge gate, not yet cleared") and does not bleed into the P3-2 ceiling note.

No edits contradict each other or any other section.

---

### P1 — Must Fix

No P1 findings.

---

### P2 — Should Fix

No new P2 findings.

---

### P3 — Consider

No new P3 findings. Carry-over P3s (FI-P3-1: path-only injection alternative unnamed in Rejected Alternatives; P3-1: same, carry from R1/R2) remain as before — not reopened, not gate blockers.

---

## Section 2: Framing Audit

No new framing issues introduced by the six edits. The edits are precision repairs to epistemic labeling and a seam clarification; they do not shift the argument's framing or suppress any available evidence. The framing assessments from R3 (three alternative framings, two omitted observations) stand unchanged and none was affected by the revision.

---

## Convergence-Saturation Signal (ADR-094)

**Convergence-Saturation Signal:** TRIGGERED

- Round number: R4
- P1 count this round: 0 (Section 1 + Section 2 combined)
- P2 count this round (new, non-carry-over): 0
- New framings or claim-scope expansions: none
- Recommendation: STOP at this round

All three R3 P2 closures are confirmed. No regression from the edits. No new findings in either section. The gate may proceed.

*Standard-sequence audit: the verdict line applies.*

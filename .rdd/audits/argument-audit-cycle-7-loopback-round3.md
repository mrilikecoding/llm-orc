# Argument Audit Report

**Audited document:** `docs/agentic-serving/essays/essay-outline-006-cross-compatibility-routing-surface.md`
**Source material:** `docs/agentic-serving/essays/research-logs/research-log.md`
**Genre:** Essay-Outline (ADR-093)
**Date:** 2026-05-24
**Round:** R3 — single-purpose re-audit per ADR-094 re-audit-after-revision rule (post-gate W8.6 addition and C8 framing-sharpening edits). Convergence-Saturation Signal verdict line omitted per that rule.

**Scope:** C8 sub-tree only (W8.1–W8.6, Section 9, Abstract C8 sentence, Amendment B Layer A bullet). C1–C7, Sections 1–8, and Amendment Log A1–A4/A2.1 treated as previously verified and not re-traversed.

---

## Section 1: Argument Audit

### Summary

- **Genre:** Essay-Outline
- **Argument chains mapped:** C8 sub-tree (W8.1–W8.6, with E8.1.1–E8.6.3); Abstract C8 sentence; Section 9 body; Amendment B Layer A bullet
- **Issues found:** 2 (one P2, one P3)
- **Pyramid coverage map:** included (C8 only)
- **Expansion-fidelity findings:** P1: 0, P2: 1, P3: 1

---

### Pyramid Coverage Map (C8 sub-tree)

| Abstract Conclusion | Argument-Graph Nodes | Body Section | References Cited |
|--------------------|--------------------- |--------------|------------------|
| C8. Terminal required; execution-model justification; two structural gaps; three layer-A candidates + grounded-loop hypothesis (candidates not findings) | C8, W8.1, E8.1.1, E8.1.2, W8.2, E8.2.1, W8.3, E8.3.1, E8.3.2, W8.4, E8.4.1, E8.4.2, W8.5, E8.5.1, W8.6, E8.6.1, E8.6.2, E8.6.3 | Section 9 (C8) | [research-log-loopback], [opencode], [adr-025] |

**META-anchored sections (not re-audited):** Section 1 (Methodology preamble), Section 8 (scope qualifications and META-OBSERVATIONs within)

**W8.6 coverage check.** Section 9's fifth CLAIM block ("Two structural gaps ADR-027 lacks are now named — and layer A in particular is a *role*, not a stage") contains four EVIDENCE bullets:
- EVIDENCE 1: layer A as distinct role; σ.1 used qwen3:14b as seat-filler (develops E8.4.1 / W8.4 and is present in the pre-W8.6 content)
- EVIDENCE 2: three candidate seat-fillers with "candidates, not findings" flag (develops E8.6.1)
- EVIDENCE 3: grounded-loop hypothesis with "working inference, explicitly NOT a spike finding" flag (develops E8.6.2)
- EVIDENCE 4: F-ρ.1 artifact bridge (develops E8.4.2)

E8.6.3 (discriminating evidence belongs to BUILD-phase validation, not another stand-in spike) has no distinct body bullet — it is implied by the EVIDENCE 3 closing sentence ("Discriminating evidence belongs to a BUILD-phase long-horizon spike against the real built terminal, not another stand-in") but not developed as a standalone point. This is noted under Expansion-Fidelity Findings.

**W8.5 (execution-model correction warrant) body coverage.** Section 9's third CLAIM block explicitly develops W8.5 ("The loop-back's original justification is corrected — the terminal is required by the client's *execution model*, not filesystem *geography*"). W8.5 has body development; coverage holds.

---

### Expansion-Fidelity Findings

**P1 findings (pyramid violations):** none.

**P2 findings (weak expansion):**

- `E8.6.3 → thin body development.` E8.6.3 states that discriminating evidence belongs to BUILD-phase validation, not another stand-in spike. The body (Section 9, EVIDENCE 3 closing sentence) alludes to this but merges it into the grounded-loop hypothesis bullet rather than giving it a distinct development. The node is not orphaned — the idea is present — but a reader looking for the "why no more spikes are warranted now" justification has to extract it from the tail of EVIDENCE 3. At the level Section 9 develops the other nodes (each gets a dedicated EVIDENCE bullet), E8.6.3 is handled implicitly. The pyramid technically holds; the expansion is thin.
  - **Recommendation:** promote E8.6.3 to a standalone body bullet in Section 9 (e.g., "EVIDENCE: The empirical contrast σ.1-vs-note-22 already supplies the grounded-vs-ungrounded contrast at the level a stand-in spike could provide; the real test requires the actual deployment surface at 20+ turns. Further stand-in spiking would elaborate rather than discriminate — BUILD-phase long-horizon validation is the appropriate venue (research-log-loopback §σ.2 Scope / carry-forward)."). This removes the ambiguity about why the candidates are left open, which matters for DECIDE readers.

**P3 findings (minor coverage gaps):**

- `Amendment B "Propagation applied" line — W8.5 vs. W8.6 discrepancy.` Amendment B records: "new C8 sub-tree W8.1–W8.5 (Argument-Graph)." The graph now has W8.1–W8.6. W8.6 was added post-Round-2 in direct response to the gate-time practitioner push-back, and the Amendment Log prose body correctly describes the three candidates and the grounded-loop hypothesis. But the "Propagation applied" summary line was not updated to say W8.1–W8.6. This is a minor provenance-record mismatch — not a structural violation, because the Amendment B narrative describes W8.6's content accurately — but a downstream reader reading only the summary line will see a stale range.
  - **Recommendation:** update the "Propagation applied 2026-05-24" line to read "new C8 sub-tree W8.1–W8.6 (Argument-Graph)" to match the current graph state.

---

### P1 — Must Fix

None.

The four substantive verification targets from the dispatch brief are all clean:

1. **Pyramid expansion-fidelity holds.** Abstract C8 ↔ C8 graph (W8.1–W8.6) ↔ Section 9 trace is intact. W8.6 has body development in Section 9 (EVIDENCE 2 = E8.6.1 candidates; EVIDENCE 3 = E8.6.2 grounded-loop hypothesis + implied E8.6.3). No Boundary 2 orphan; no Reverse Boundary violation.

2. **Scope discipline on "candidates not findings" / "working inference" flags.** Both flags are present, prominent, and iterated across all four layers:
   - **Argument-Graph W8.6:** "The candidates are *candidates*, not findings; σ tested only candidate (1)'s feasibility for a short task."
   - **E8.6.1:** DECIDE/ARCHITECT select (no claim of selection made).
   - **E8.6.2:** "*(working inference; explicitly flagged — NOT a spike finding)*" inline, with the AS-9-reopening condition stated explicitly.
   - **E8.6.3:** discriminating evidence belongs to BUILD, not another spike.
   - **Abstract C8:** "candidates, not findings" verbatim.
   - **Section 9 EVIDENCE 2:** "the candidates are *candidates*, not findings."
   - **Section 9 EVIDENCE 3:** "working inference, explicitly NOT a spike finding."
   - **Amendment B Layer A bullet:** "it is a working inference, **NOT** a spike finding."
   A reader cannot misread these as established results without ignoring four co-located explicit flags across four document layers. The flags are load-bearing and adequately distributed.

3. **No new spike claim amplification.** W8.6 adds candidate-naming and a discriminating hypothesis. It does not assert that any candidate has been selected, validated, or recommended. The grounded-loop hypothesis correctly identifies σ.1's grounded-vs-ungrounded contrast as the source of the inference and the AS-9-reopening condition as the failure mode. No π/ρ/σ finding is described beyond what the research log records.

4. **W8.4 sharpening holds.** "Layer A is a *role* no current ADR-027 component holds; the pipeline is structurally *under-specified* ... not merely missing a stage" is a structural description that follows from the Argument-Graph's own definitions: the routing planner is scoped to which-capability/one-shot; the ensembles are scoped to content generation; neither drives the per-turn loop. The sharpening does not assert this was established by σ — it uses σ to confirm the role is "required and fillable" while leaving the seat-filler question open. The logic is sound.

5. **E8.4.1 sharpening holds.** "σ.1 used qwen3:14b directly as the seat-filler" is accurate per the research log (σ.1: "qwen3:14b/Ollama... drove a 3-step task... to completion"). The clarification that this is the "layer-A feasibility floor (no delegation)" correctly positions σ.1 as a floor test, not a recommendation.

---

### P2 — Should Fix

**P2-1 (thin body development of E8.6.3):** see Expansion-Fidelity Findings above. The "why no further stand-in spiking is warranted" justification is merged into the tail of the grounded-loop hypothesis bullet in Section 9 rather than standing alone. For DECIDE readers who will use Section 9 as a direct input, the implicit treatment creates a small ambiguity: is the decision to stop spiking based on diminishing returns (E8.6.3's actual content) or on the practical constraint of scope? Promoting E8.6.3 to a standalone bullet removes that ambiguity.

**P2-2 (grounded-loop hypothesis — AS-9-reopening condition stated but not scoped):** E8.6.2 states: "If the hypothesis fails to hold, AS-9's scope reopens and the seat-filler choice tilts toward (2) or (3)." This is logically correct but it introduces a conditional consequence — "AS-9's scope reopens" — that is significant and that the document does not define. AS-9 as codified at MODEL boundary reads: "Structurally-bounded LLM roles produce reliable output on single-decision-shaped tasks where the orchestrator-LLM-as-decider failed." If the grounded-loop hypothesis fails (candidate (1) proves unreliable even in a grounded per-turn role), the implication for AS-9 is non-trivial: it would mean that structural grounding alone does not explain σ.1's success, so the bounded-role property may not generalize to the layer-A role. The document says "AS-9's scope reopens" but does not tell ARCHITECT what that means in practice — does it mean AS-9 needs to be narrowed to exclude layer-A-style roles? Does it mean the hypothesis of grounding as the differentiating factor is wrong?

The claim "AS-9's scope reopens" is not an error — it is the correct logical consequence. But it is stated as a downstream consequence without giving DECIDE/ARCHITECT any guidance on how to evaluate it. Given that W8.6 is specifically marked as material for ARCHITECT, a single clarifying parenthetical would reduce ambiguity: e.g., "AS-9's scope reopens specifically around whether the grounded-vs-ungrounded distinction is the correct framing of the structural property, not whether the bounded-role pattern holds generally."
- **Recommendation:** add a parenthetical scoping "AS-9's scope reopens" to the grounded-role/ungrounded-role distinction, rather than leaving ARCHITECT to infer the scope of the reopening.

---

### P3 — Consider

**P3-1 (Amendment B provenance summary line — W8.5 vs. W8.6):** see Expansion-Fidelity Findings above. The "Propagation applied" summary line in Amendment B records W8.1–W8.5; the graph has W8.1–W8.6. A one-word fix. Low stakes but a clean audit trail matters for future round navigation.

---

## Section 2: Framing Audit

The framing audit examines what the post-Round-2 edits foreground and suppress, with the four dispatch-brief framing questions as the lens.

### Question 1: What alternative framings did the evidence support?

The primary framing of W8.6 is: three candidates are named, candidate (1) has the most spike support (σ.1), the grounded-loop hypothesis makes candidate (1) potentially AS-9-consistent, and DECIDE/ARCHITECT select. Two alternative framings the evidence could have supported:

**Alternative A: foreground candidate (2) as the AS-9-conservative choice.**
Candidate (2) — extending the routing planner to per-turn driving — reuses the bounded-role pattern AS-9 already codified and avoids the grounded-loop hypothesis entirely. AS-9 is confident; the hypothesis is not. A reader applying AS-9 strictly would note that candidate (2) is the one that does not require hypothesis validation to satisfy AS-9 — you already know bounded-role patterns work, and candidate (2) is one. The evidence (AS-9 codified; Spike ζ validated the planner's single-decision form) supports this framing. Under this framing, the W8.6 ordering would present candidate (2) as the safer architectural choice pending hypothesis validation, not candidate (1) as the σ-tested baseline.

What a reader would need to believe: that the bounded-role pattern is the stronger architectural bet than the grounded-loop hypothesis, and that ARCHITECT should default to the pattern that doesn't require a new hypothesis to hold.

**Alternative B: foreground the open question as a scope question, not a candidate-selection question.**
The evidence could have been framed as: the right architectural question for DECIDE is not which candidate fills layer A but whether layer A should be a stand-alone built component or an extension of an existing ADR-027 component. Under this framing, candidates (1) and (2) are extensions of the existing pattern; candidate (3) is a new component. The primary fork is build-new vs. extend-existing, with the three candidates as sub-choices under each fork.

What a reader would need to believe: that the extend-vs-build distinction is more architecturally load-bearing than the model-vs-planner-vs-new-component distinction.

Both alternatives are coherent with the evidence. The chosen framing (list three candidates with (1) first, noting it is σ-tested) is not wrong — but it does have a consequence explored in Question 4 below.

### Question 2: What truths were available but not featured?

**Unfeature 1: σ.2 batched the turns.** The research log records a significant scope qualification on σ.2: "the driver emitted the full plan in one turn — `write calc.py`, `write test_calc.py`, `bash` (run test) — then finished." This means σ.2 was not a genuine multi-turn decide-act-observe chain; the driver planned the entire trajectory at turn 1 and executed it. This is noted in E8.3.2's scope parenthetical ("a 2-turn batched task — NOT long-horizon") and in Section 9's SCOPE QUALIFICATION, but W8.6's body content does not explicitly tie the "grounded-loop hypothesis" to this batched-planning caveat. The grounded-loop hypothesis rests on σ.1's three-turn grounded loop (write → write → bash, each turn a new decision), not σ.2's two-turn batched execution. Readers of W8.6 who focus on the hypothesis might not register that the most relevant evidence is σ.1, not σ.2.

**Why it matters:** If a reader uses σ.2 as the primary support for the grounded-loop hypothesis, they are resting the hypothesis on a test that did not actually demonstrate per-turn decide-act-observe behavior. The document doesn't misstate this — the scope qualifications are correct — but E8.6.2 references "σ's success" without distinguishing σ.1 (genuine multi-turn) from σ.2 (batched). The reference to "σ" could inadvertently credit σ.2 as co-evidence for the per-turn grounded loop.

**Unfeature 2: the σ.1 task was short and simple.** σ.1 drove a 3-step task (write `calc.py`, write `test_calc.py`, bash) — arguably a minimal grounded loop, not a representative one. The research log records this honestly: "3-turn task." The document's scope qualifications name "sustained long-horizon driving (dozens of turns, harder tasks)" as not validated. But W8.6's grounded-loop hypothesis is stated as if σ.1's success is structurally meaningful (the grounded-vs-ungrounded distinction), and the document does not register that a 3-turn, 2-write-1-bash task may be too simple to discriminate the grounded-vs-ungrounded hypothesis from a "the task was easy" hypothesis. A model that can handle a 3-step write/test task in a grounded loop is not necessarily different from one that fails note-22's multi-dispatch composition on the structural axis; the task complexity differential is also a plausible confound.

**Why it matters:** the grounded-loop hypothesis is the load-bearing discriminating question for ARCHITECT's candidate-1 selection. If the 3-turn task complexity differential is a confound (not just the grounded-vs-ungrounded structural distinction), then ARCHITECT may be prematurely anchored on candidate (1) based on a structurally under-powered test.

**Unfeature 3: candidate (2)'s scope expansion requirement.** E8.6.1 notes that candidate (2) "requires substantial scope expansion of its system prompt and contract." But it does not note that this scope expansion might itself re-introduce the kinds of multi-step reasoning failures AS-9 addressed — the routing planner was specifically validated as a *single-decision-shaped* task (Spike ζ). Extending it to per-turn driving changes the task shape substantially. The evidence available in the research log (Spike ζ validated one-shot routing) actually argues against candidate (2) on AS-9 grounds — the extension from one-shot to per-turn is precisely the kind of scope expansion that takes a bounded role outside the scope of its validation. This point is missing from E8.6.1 and from the framing of candidate (2) in Section 9.

### Question 3: What would change if the dominant framing were inverted?

The dominant framing of W8.6 is: candidate (1) is named first with σ-tested status; the grounded-loop hypothesis makes it AS-9-consistent pending BUILD validation; candidates (2) and (3) are structurally available alternatives.

**Inverted framing:** candidate (1) is the AS-9-risky choice; candidates (2) and (3) are the AS-9-conservative choices. Under the inverted framing:

- What becomes stronger: the argument that using a model in the OpenAI "model" seat is structurally analogous to the orchestrator-LLM role AS-9 removed — the very role that failed at PLAY note 22. The grounded-loop hypothesis is precisely the bet that the structural similarity does not transfer the failure mode. If ARCHITECT inverts the framing, the null hypothesis is "models-in-the-model-seat fail in complex agentic loops; σ.1 was too simple to surface this" — and the grounded-loop hypothesis must overcome that null.

- What becomes more salient: the AS-9 codification text itself says the orchestrator-LLM fails at "single-decision-shaped tasks" and that "structurally-bounded roles" succeed. Layer A in candidate (1) is not a single-decision-shaped task — it is a per-turn decision across many turns. AS-9 was validated specifically on single-decision tasks; the per-turn multi-decision shape is outside AS-9's validated scope regardless of grounding.

- What the document would need to address: whether AS-9's scope covers per-turn multi-decision loop-driving at all, or whether W8.6 is betting that the grounded-loop property extends AS-9's scope into a new task shape. If the latter, that is a significant architectural assumption that deserves explicit acknowledgment — not an error, but a load-bearing framing choice.

### Framing Issues

**P2-F1 — candidate (1) is advantaged by ordering and σ-tested framing in a way that may prematurely tilt ARCHITECT.**

The dispatch brief specifically asks: "Is candidate (1) advantaged in the prose ordering / framing in a way that prematurely tilts ARCHITECT's selection?" The answer is: yes, modestly, through two mechanisms.

**Mechanism 1 — list ordering.** Candidates appear as (1), (2), (3) in E8.6.1 and in Section 9 EVIDENCE 2. Candidate (1) is listed first with the descriptor "what σ tested with qwen3:14b" — the only candidate that has any empirical grounding. Candidates (2) and (3) have one sentence each; candidate (1) has two sentences plus a structural description of its relationship to the orchestrator-LLM. The asymmetry is factually accurate — candidate (1) is the only one σ touched — but it creates a salience gradient that primes ARCHITECT toward (1) before the structural trade-offs are enumerated.

**Mechanism 2 — grounded-loop hypothesis placement.** E8.6.2 discusses the hypothesis in terms that apply specifically to candidate (1)'s AS-9 compatibility. The hypothesis is correctly flagged as a working inference. But the hypothesis discussion is positioned immediately after the candidate list (E8.6.1 → E8.6.2) with no parallel discussion of what makes candidates (2) and (3) AS-9-compatible or not. The absence of a corresponding "what makes candidate (2) consistent or inconsistent with AS-9" note means candidate (1) gets asymmetric depth of analysis — the only one whose architectural risk is explicitly named and the only one whose discriminating question is specified.

The combined effect is that a reader scanning W8.6 sees: "(1) σ-tested + hypothesis analysis; (2) reuses pattern but needs scope expansion; (3) new component." Candidates (2) and (3) do not get the same framing depth as candidate (1), even though the document explicitly says DECIDE/ARCHITECT select and no candidate is recommended.

**This is a framing issue, not an error.** The "candidates not findings" flags are clean; no false claim is made. But the asymmetric depth of treatment creates a tilt. ARCHITECT should be equipped to evaluate all three candidates with equal analytical depth.

- **Recommendation:** add a parallel structural analysis note for candidate (2) in E8.6.1 (or a dedicated E8.6.x node): candidate (2) stays within the bounded-role pattern AS-9 codified, but the scope expansion from one-shot routing to per-turn driving takes it outside the task shape Spike ζ validated. The AS-9 confidence for candidate (2) in its proposed extended form is therefore lower than the confidence for candidate (2) in its current one-shot form. Similarly note for candidate (3): a new dedicated component allows the most precise role scoping but requires the most new design work. A parallel trade-off structure across all three candidates would remove the first-candidate advantage without changing the factual content.

**P3-F1 — grounded-loop hypothesis references "σ's success" without distinguishing σ.1 from σ.2.**

The hypothesis in E8.6.2 reads: "if σ's success and Cycle 6 PLAY note 22's failure reflect a structural grounded-vs-ungrounded loop property..." The reference is to "σ's success" broadly, but the structurally relevant evidence is σ.1 (genuine per-turn grounded loop) rather than σ.2 (batched plan in turn 1). Tightening this to "σ.1's success" would make the hypothesis's evidential basis precise and prevent ARCHITECT from reading σ.2 as co-evidence for per-turn grounded driving.

- **Recommendation:** replace "σ's success" with "σ.1's success" in E8.6.2 and the corresponding Section 9 EVIDENCE 3 bullet, and add a parenthetical: "(σ.2 batched the planning into one turn; σ.1 is the evidence that per-turn grounded decision-making holds for a cheap local model)."

---

*Single-purpose re-audit (dispatched per the re-audit-after-revision rule for the post-gate W8.6 addition and C8 framing-sharpening edits). Convergence-Saturation Signal verdict line omitted per ADR-094.*

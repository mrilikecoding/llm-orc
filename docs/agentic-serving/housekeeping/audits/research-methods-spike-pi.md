# Research Design Review — Spike π (Destination-Form Validity Gate)

**Reviewed question set:** `docs/agentic-serving/essays/research-logs/cycle-7-spike-pi-form-adequacy-gate.md` — full pre-registered design (arms A–D corpus battery, Cells A-live / B-live, four forks)
**Constraint-removal response included:** n/a (mechanism-amendment spike; no ADR-082 constraint-removal artifact in scope)
**Date:** 2026-06-10
**Reviewer role:** research-methods (ADR-060 + ADR-082 dimensions 1–4, ρ/ξ calibration bar applied)

---

## Summary

- **Forks reviewed:** Fork 1 (detection shape), Fork 2 (cross-seam unification), Fork 3 (recovery / convergence, PRIMARY), Fork 4 (false-positive)
- **Flags raised:** 7 (2 P1, 3 P2, 2 P3)
- **Criteria applied:** 1–4 (ADR-082)

---

## Per-Fork Review

### Fork 1: "What do {parse-check, fence-only, marker-detection} each catch / miss / false-positive across the labeled corpus?"

**Belief-mapping.** The fork assumes the right evaluation unit is a confusion matrix across the labeled corpus. What would the researcher need to believe for a different question to be more productive? They would need to believe that the labeled corpus is the bottleneck rather than the gate's behavior on the live trajectory. The fork's prediction table is pre-registered before the corpus is run — this is the right discipline. The adjacent question the framing suppresses: "Does gate selection interact with the cheap model's failure modes?" If qwen3:8b fails in ways not represented by the corpus categories (e.g., a partial-fence where closing backticks are absent but no prose is present), the corpus-based confusion matrix may not predict live behavior. The corpus construction discipline (real items taken verbatim, synthesized items written to category definitions) partially addresses this, but the labeled-corpus and live-trajectory arms are the two pillars for exactly this reason.

**Embedded conclusions.** The fork correctly avoids presupposing any gate's superiority. The gate-viability decision rule (viable iff false-positive = 0 AND catches C1) is a reasonable minimum bar, not a conclusion. No issue here.

**Scope.** Appropriate. Fork 1 is the diagnostic substrate for Fork 2; it correctly defers the unification claim to Fork 2 rather than folding it in.

---

### Fork 2: "Does parse-check's catch-set span both seams' deterministic slices, and is its miss-set the irreducibly-semantic residual?"

**Belief-mapping.** The fork tests the central hypothesis: that the parse/validity boundary cleanly partitions deterministically-catchable from irreducibly-semantic. What would the researcher need to believe for a different framing to be more productive? They would need to believe that "parses" is a clean proxy for "acceptable form and language." An alternative productive question: "Is there a class of plausibly-submitted deliverables that parse correctly but are clearly wrong-language in a sense that is deterministically detectable — for instance, a `.py` file whose first token is a shebang line followed by JavaScript-with-backtick-syntax that happens to be valid Python?" That question may be pathological in practice, but the hypothesis's refutability depends on the C4/C6 boundary being clean rather than the specific adversarial case being common. The pre-registration acknowledges the residual is semantic-level (function calls the wrong thing, not wrong syntax), which is a sound operationalization of C6 for the scenarios the corpus represents.

**Embedded conclusions.** See P1-A below. The core hypothesis is refutable: three pre-registered failure conditions (false-positives, miss-set leakage, non-convergence) are named and would each falsify the unification claim. The fork does not presuppose the hypothesis holds. The risk of embedded conclusion lives not in the hypothesis statement itself but in the corpus design — specifically, whether C6 is a genuine residual or a narrow strawman. This is addressed as P1-A.

**Scope.** Appropriate. Fork 2 is the right way to frame the central question once Fork 1 has established the confusion matrix.

---

### Fork 3: "With parse-check + self-healing recovery installed, does the σ/η 5-file trajectory converge to all-valid files?"

**Belief-mapping.** The fork is the primary arm per the live-multi-turn-primary directive, which is correct. The adjacent question the framing may suppress: "Is all-files-parse the right success criterion, or should the gate be evaluated on all-files-parse *and no regressions on already-passing files*?" The cell design (two cells, same 5-file trajectory) implies the baseline Cell A-live already has some files passing and some failing. If Cell B-live's gate refuses a deliverable that Cell A-live passed (because the gate has a false-positive on a real file the corpus's C7/C8/C9 controls did not represent), the convergence rate improvement may mask a regression. The per-gate false-positive rate on corpus controls (Fork 4) is the intended safeguard, but corpus controls are synthesized/captured in advance while Cell B-live runs against a real trajectory with real qwen3:8b output. See P1-B.

**Embedded conclusions.** The convergence arm is well-structured. The "protect-but-does-not-recover" outcome is pre-registered as a first-class result rather than a failure, which is the right framing. No embedded conclusion here.

**Scope.** Appropriate. The decision to name the causal-isolation control explicitly (re-dispatch success rate on refused `cli.py`) and ask the reviewer whether it is sufficient is exactly the right structure for methods review.

---

### Fork 4: "Does any gate refuse legitimate deliverables?"

**Belief-mapping.** The fork is a safety check rather than a research question. The belief-mapping question is whether the C7/C8/C9 corpus controls are representative enough to catch false-positive risks. The form "correct bare .py, correct .md, correct .json" covers the three typed-path cases. What the false-positive corpus does not include: a correct `.py` file whose content is unusual in ways that might challenge `ast.parse` — e.g., a file containing only a docstring and no code, a file with encoding declarations, a file using `# type: ignore` annotations extensively. None of these should fail `ast.parse`, but the question is whether the corpus's false-positive controls are wide enough that a positive result on C7/C8/C9 is meaningful confidence rather than a narrow pass on easy cases. This is a P3 observation below.

**Embedded conclusions.** None.

**Scope.** Appropriate. Fork 4 is correctly positioned as a disqualification criterion rather than a hypothesis.

---

## Question Set Assessment

### Premature narrowing / prior-art treatment

The design treats Spike σ/η output as prior art explicitly — real bytes are the primary corpus inputs, synthesized items are written to category definitions, and the completeness gate (ADR-040) is correctly identified as a neighbor that handles wrong-filename divergence. The unification hypothesis explicitly maps the composition (form gate + completeness gate) rather than proposing to duplicate it.

One narrowing concern: the arm set (parse-check / fence-only / marker-detection) covers three shapes that are differentiated by *what they detect*, but all three share the same *intervention point* (at the FormGate seam, `artifact_bridge.py`). The ξ methods review raised the analogous question for that spike's arms — all variants of the callee-dispatch injection shape — and flagged "no arm considers an alternative intervention point." For Spike π, the alternative intervention point question is: "Should the gate operate on the marshalled bytes at the bridge, or on the raw callee output before it is stored (at the SessionArtifactStore seam, ADR-025)?" The design commits to the bridge seam without examining the store seam. This is likely correct — the bridge is the last point before emission, which is exactly where the gate belongs — but the reason is not stated. The ADR-035 §Decision 4 framing ("the bridge refuses to emit") already settles this at the design level, so the narrowing is defensible prior-art treatment rather than premature closure. The pre-registration should note it.

The ADR-035 rejection of the typed `submit`-slot alternative (§Rejected alternatives) still applies cleanly: that rejection was on destination-coupling grounds (capability ensembles remain destination-agnostic), and nothing in Spike π changes the coupling economics. The rejection holds.

### Incongruity surfacing

An incongruity is present in the research context that the question set does not directly surface.

The spike maps the composition: form gate (ADR-035 / Spike π's subject) + completeness gate (ADR-040 / Spike η's subject). ADR-040's completeness gate operates on the Session Action Record — it checks, deterministically, whether a named file has been produced by inspecting the recorded output. Spike η's §LIVE framing settled on the "named-file boundary is the principled edge of determinism" as the terminal sentence for that spike.

Spike π's unification hypothesis proposes a *parse/validity* boundary as the principled edge for the form+adequacy seam. Both boundaries are characterized as "deterministic up to the edge of what X can decide." The incongruity: two adjacent deterministic gates, each described as the principled edge of their respective domains, but the spike does not ask whether they *compose to a single gate* or whether the form gate is partially redundant with the completeness gate's "file produced" record.

Concretely: if the form gate refuses a deliverable, the file is un-produced, and the completeness gate sees a missing file and re-delegates on the next turn. This is the recovery loop the spike relies on. But the completeness gate *already* sees missing files for any reason (form failure, generation failure, wrong-filename). The question the spike does not ask: "Is there a sense in which parse-check's refuse-to-emit is a specialized trigger of the completeness gate, and if so, does the composition create any loop instability or double-delegation risk that is not present when the two gates operate on disjoint failure modes?" This is not a redesign question — the gates operate at different layers and on different signals — but the composition's loop dynamics under repeated form-gate refusals (the "deterministically re-fails" scenario the spike pre-registers as a risk) are exactly the scenario where both gates fire in sequence on every turn. The spike pre-registers this risk but does not ask whether the two-gate composition produces a loop that looks different from the completeness gate alone.

This is flagged as an incongruity in the research context (two deterministic gates described in parallel terms sitting adjacent to each other) that the question set does not surface for explicit examination. Fork 3's Cell B-live arm will observe the composed loop behavior in practice, but it does not have a named question about the loop dynamics distinct from "does it converge?"

### Coverage gaps

The methods-review targets listed in §Methods-review targets in the pre-registration are well-chosen. The design correctly self-identifies the five most vulnerable points and asks the reviewer to address them. The coverage gap not listed: the **n and margin** for Fork 3 are explicitly deferred to methods review (§Live trajectory arm: "n to be set at methods-review"), but the pre-registration does not specify the decision rule for Fork 3 beyond "B-live > A-live by a margin set at methods-review AND re-dispatch-success control > 0." This leaves the primary arm's pass criterion open at methods-review time, which is unusual — σ/η both ran small-n live, so there is precedent for deferring n, but the margin and the "good enough to proceed to ADR amendment" bar should be specified before running. See P1-B.

---

## Findings

### P1 — Design flaws that would invalidate conclusions before run

**P1-A: C6 as specified is a strawman residual, not a genuine probe of the miss-set boundary.**

The corpus defines C6 as "parses-but-semantically-wrong" with the example "valid Python, calls a nonexistent/wrong function." This is the right conceptual category — it is meant to be the irreducibly-semantic residual that no deterministic gate can catch without false-positives.

The problem is that C6's ground-truth determination depends on the evaluator knowing what functions exist in the surrounding codebase. For the corpus's `(content, destination_path)` pair structure, the evaluator has only the content and the destination path — not the surrounding project context. A synthesized item that calls `nonexistent_fn()` is classified C6 because the evaluator knows it is nonexistent (it was synthesized to be so). But in the live trajectory, the evaluator would need the sibling files (ADR-039's content anchor addresses this for the cross-file slice). The corpus C6 item is constructed as "definitely wrong," which makes it an easy True-Negative (gates correctly miss it) but does not probe the boundary case: a deliverable that calls a function with a plausible name that may or may not exist.

This matters for the unification claim. Fork 2 holds "iff B's miss-set = {C6} (the residual)." If C6 is narrowly defined as "calls a clearly nonexistent function," the miss-set boundary is not the same as "everything that requires semantic context to evaluate." A function that calls `parse_args` with wrong argument structure parses fine, is not in C6 as defined, and would not be caught by any gate. The corpus design conflates "the category for which gates are expected to miss" with "the proof that the miss-set is bounded to semantic residuals."

Two specific problems:

1. C6 is synthesized to be "clearly wrong" rather than "at the boundary of what parse-check can decide." A genuine probe of the miss-set would include items that look almost-correct — valid Python that uses real function names in subtly wrong ways. Without those items, the corpus proves only that C6-easy misses all gates, which is true by construction and not informative about the boundary.

2. C6 is a single category in the corpus. The post-run claim "B's miss-set = {C6}" needs more than one example of C6 to be credible — a single synthesized item is too thin to characterize the miss-set as "exactly the semantic residual." The ρ review's P1-A concern about adjudication subjectivity applies here too: the C6 classification is a judgment call that the corpus construction discipline should make explicit.

**Recommended design change:** Either (a) construct at minimum three C6 variants spanning a severity range — obvious wrong (calls `nonexistent_fn`), plausible wrong (calls `process_args` instead of `parse_args` — a real stdlib name used at the wrong layer), and near-miss (calls `parse_args()` with one extra positional argument) — and report gate behavior on all three; or (b) pre-register an explicit operationalization of the C6 boundary: "A C6 item is one that requires knowledge of the surrounding project's function graph, import structure, or runtime behavior to evaluate as wrong; a C6 item does NOT include syntax errors, wrong-language bytes, or any failure that can be detected from the file content + destination path alone." The boundary operationalization is the key move — without it, the unification claim's "miss-set = C6" is not a refutable assertion because C6 expands to fit whatever the gate misses.

Option (b) costs no additional corpus items but requires a written operationalization pre-registered before synthesis. Option (a) costs two additional synthesized items (lightweight) and strengthens the miss-set characterization. The ξ precedent for C6-analog items suggests the two additional items are worth adding.

---

**P1-B: Fork 3's primary arm has no pre-registered pass criterion — both the margin and n are deferred to methods review without a default, leaving the primary arm's decision rule open at run time.**

The pre-registration states (§Live trajectory arm): "Two cells, n to be set at methods-review." The pre-registered outcome (§Fork 3) states: "The gate *recovers* if B-live > A-live by a margin set at methods-review AND the re-dispatch-success control is > 0." Neither n nor the margin is specified.

This is the primary arm. The θ/ρ/ξ precedents all pre-registered pass thresholds before running, with the ρ review flagging the threshold motivation concern (ρ P2-B) as a P2. Here the threshold is not merely unmotivated — it is unspecified. A result of B-live = 3/5 sessions all-valid vs. A-live = 1/5 sessions all-valid could be read as "margin of 2" or as "40% improvement on a 5-session run" — either a pass or a borderline result depending on the margin the reviewer sets after seeing the data.

The causal-isolation control (re-dispatch success rate on a refused `cli.py`) is better specified: "re-dispatch success ≈ 0" routes to "protect-but-does-not-recover" and the escalation signal. But the Fork 3 pass criterion for the "gate actually recovers" outcome needs a margin and n.

Additionally: the pre-registration asks the reviewer whether a refuse-but-no-signal control (analogous to ξ's `Control_filler`) is needed. The answer is yes, for the same reason ξ P1-A required it, but the mechanism is different. For ξ, the question was whether the content of the injected signal mattered. For Spike π, the question is whether the gate's *refusal* (with FormRefusedError signaling the Terminal) matters relative to a gate that *degrades silently* without a refusal signal (equivalent to Cell A-live, where the SyntaxError surfaces at the client's write execution rather than at the gate). The existing Cell A-live is essentially this control: it is the pass-through baseline where the error surfaces after emission rather than at the gate. So the control already exists in the design as Cell A-live. But this equivalence should be made explicit in the design: "Cell A-live is the control for 'form error visible at client execution but not refused at gate.'" Without this framing, the causal story is ambiguous: Cell B-live > Cell A-live could be because the gate refuses and triggers re-delegation, or because the 5-file trajectory is simply non-deterministic and Cell B-live happened to run on a more favorable seed. The resolution is a per-session `n` large enough to distinguish these, and/or a stated rationale for why n sessions at the live cost is a reasonable discriminant.

**Recommended design change:** Before running, pre-register:
- n per cell (recommended: n ≥ 3 full sessions per cell, same seed/trajectory for comparability — σ/η ran n=2 live; n=3 is the minimum for a marginal claim)
- The margin that constitutes "recovers": recommended framing is "B-live all-files-valid rate − A-live all-files-valid rate ≥ 0.4 across n sessions, AND at least one session in Cell B-live converges that would have failed in Cell A-live"
- The explicit framing that Cell A-live is the control for "gate absent / error surfaces at client"
- The "protect-but-does-not-recover" outcome criterion: B-live refusal events observed (gate is firing) AND B-live all-files-valid rate not materially above A-live

None of these require additional design changes; they are pre-registration annotations that prevent post-run threshold-setting.

---

### P2 — Weaknesses that bound the claims

**P2-A: The Arm D false-positive risk (marker-detection on C7) is predicted but not operationalized, and its threshold is not pre-registered.**

The design predicts Arm D (marker-detection) has a false-positive risk on C7 (correct bare `.py`) because "a code comment or docstring mistaken for prose." This is the right concern. But the corpus has one C7 item ("σ/η converged files (real)") with unspecified content. If the C7 item is a short module with no docstrings and no inline comments, Arm D's false-positive risk is not actually tested — the C7 item is too easy. The Arm D prediction ("predicted false-positive > 0") needs a C7 item that exercises the risk: a correct Python file with a module-level docstring, inline comments that begin with natural-language sentences, and/or a `# Here is the implementation:` comment that a marker-detection rule might flag as prose-scaffolding.

Without a challenging C7, the Arm D false-positive prediction becomes unfalsifiable: if Arm D passes C7-easy, the prediction is "not confirmed" but also not refuted. The design's claim that "D risks false-positives on C7 — the determinism-principle concern, quantified" cannot be quantified on a corpus with only one easy C7 item.

**Recommended design change:** Expand C7 to at minimum two items: C7-plain (short module, no docstrings) and C7-documented (module with module-level docstring, inline `# Here is...` style comments, at least one function with a multi-line docstring). Report Arm D's false-positive rate on each. This adds one synthesized corpus item, costs nothing, and is essential for the Arm D "quantified" claim.

---

**P2-B: The corpus's representativeness claim for C4 (wrong-language in a typed path) rests on a re-pathed real failure, and the re-pathing transformation is not documented.**

The design lists C4 as "η run 2 `index.js` content in a `.py` (real, re-pathed)." The source was `cli/index.js` — JavaScript content delivered to the wrong filename in the real trajectory. To make it a C4 corpus item, the design re-paths the content to a `.py` destination path. This is the right conceptual transformation (the category is "wrong-language in a typed path"), but the transformation is not documented. The question is: what exactly is the `(content, destination_path)` pair for C4? Is it the verbatim `index.js` bytes with `cli.py` as the destination path? Or is the content further modified?

This matters for the corpus provenance claim. The anti-cherry-pick discipline requires that "C1/C2/C4/C5/C7/C8 are taken verbatim from captured σ/η bytes where available." For C4, "verbatim" is modified by "re-pathed" — which is a structural change to the `(content, destination_path)` pair even if the content bytes are verbatim. The spike should document: "C4 item: content = verbatim bytes from `scratch/spike-eta-deliverable-enumerator/` [η run 2 `cli/index.js`]; destination path = `cli.py` (re-pathed to simulate the adequacy failure)."

**Recommended design change:** Add a provenance note to the C4 item in the corpus construction section specifying the source file, the verbatim content claim, and the re-pathing transformation. This is a one-line documentation addition with no run implications.

---

**P2-C: The "one dispatch → one deliverable" granularity invariant is load-bearing for the recovery loop's behavior but is not examined as a Fork 3 confound.**

The spike's recovery mechanism is: gate refuses → un-produced file → completeness gate sees missing file → re-delegates on next turn. This loop assumes the next turn re-delegates exactly the refused file, not a different file or a multi-file dispatch. The granularity invariant (ADR-035 §Decision 3) holds in the design, but the recovery loop depends on the loop driver dispatching the *refused file specifically* rather than re-dispatching the whole session.

In the σ/η live trajectories, `cli.py` was the hardest file. If the loop driver re-delegates `cli.py` on the turn after the gate refusal, the recovery loop functions as described. But the completeness gate (ADR-040) sees "cli.py is missing" — it does not know whether the file was never generated, generated and refused, or generated and produced elsewhere. The re-delegation is driven by the completeness gate's missing-file signal. If the loop driver re-delegates the missing file, the recovery loop works. If the loop driver's re-delegation logic produces a different action (e.g., it asks the coder to produce the full session output again because it detects a stall), the gate refusal may not produce the targeted re-delegation the spike assumes.

This is not a design flaw per se — it is a live-arm observable that the spike will see. But Fork 3's "convergence" measurement should include a per-session trace of what the loop driver delegates on the turn following a gate refusal. Without that trace, a non-convergence result in Cell B-live cannot distinguish "re-delegation targeted the right file and the coder deterministically re-failed" from "re-delegation produced a different action and the loop stalled."

**Recommended design change:** Add to the Fork 3 measurement section: "Per-session trace logged — the turn immediately following a gate refusal records the loop driver's action (re-delegate same file / re-delegate different file / premature-finish / other)." This is a logging addition that costs nothing and makes the recovery loop interpretation clean.

---

### P3 — Improvements

**P3-A: The false-positive corpus controls (C7/C8/C9) should include at least one `.py` file with a syntax that is unusual but valid, to ensure the parse-check gate's false-positive claim is not limited to trivially-easy inputs.**

The C7 item is described as "σ/η converged files (real)." Converged files from the σ/η trajectory are qwen3:8b outputs that already passed the live trajectory — they are likely clean, well-formatted Python with no unusual syntax. The false-positive claim for Arm B (parse-check) is "none (C7-C9 pass)." If C7 is a well-formed module with standard imports and function definitions, this prediction is easy to confirm and does not probe the gate's behavior on edge-case-but-valid Python.

**Recommended design change (lightweight):** Add to C7 a second item: a correct `.py` file using a less common but valid Python pattern — e.g., a file using walrus operator (`:=`), a file using `match`/`case` syntax, or a file with a `__all__` export list and no function bodies (stubs). This ensures the parse-check's false-positive prediction is confirmed on cases that could plausibly stress-test the `ast.parse` call, not just the simplest case. One additional synthesized item, no run implications.

---

**P3-B: The pre-registration's refutability conditions for the unification hypothesis are named at the Fork level but not in the primary corpus arm (Fork 2). Adding a sentence connecting corpus-level outcomes to the hypothesis failure conditions would sharpen the decision rule.**

The pre-registration names three refutability conditions in the hypothesis section: (a) false-positives on legitimate deliverables; (b) miss-set not confined to the irreducibly-semantic residual; (c) live recovery loop does not converge. These map to Forks 4, 2, and 3 respectively. But the Fork 2 outcome section (§Fork 2) says "holds iff B's catch-set ⊇ {C1-C5} AND B's miss-set = {C6}" without explicitly linking this to refutability condition (b). If B misses C3 (leading/inline prose preamble), the miss-set claim is violated and the unification is incomplete — but the decision rule as written says "if B misses any of C1-C5, the unification is incomplete (a second mechanism is needed)" without specifying what the second mechanism would be or whether "incomplete unification" routes to amendment, loop-back, or a narrower amendment covering only what B catches.

**Recommended design change:** Add to the Fork 2 pre-registered outcome: "If B's catch-set is ⊆ {C1-C5} but does not include all five — e.g., B catches C1/C2/C4/C5 but misses C3 (prose preamble that does not fail ast.parse) — the amendment covers the confirmed catch-set only; the missed category is a named scope boundary requiring a follow-on probe. The amendment's form-gate claim is narrowed to the confirmed cases." This one-sentence addition prevents a post-run scope-expansion where a partial-pass is read as a full-pass by omitting the missed category.

---

## Well-Handled Elements

Several aspects of the design are already defensible and should not be changed:

**The hypothesis's pre-registered failure conditions.** All three refutability conditions — false-positives, miss-set leakage, non-convergence — are named before running. This is the design's strongest methodological feature and meets the highest bar set in the precedent reviews.

**The ADR-035 §Decision 4 escalation order.** The spike correctly composes with the existing escalation order rather than proposing a new one. "Protect-but-does-not-recover → escalation 3 (frontier seat)" is a pre-registered outcome rather than a post-hoc redirect.

**The wrong-filename / completeness-gate composition.** The spike correctly identifies that wrong-filename divergence (the `cli.py` vs. `index.js` case) is the completeness gate's job and maps the gate-composition explicitly. This prevents the form gate from being over-scoped into the completeness gate's domain.

**The Arm D false-positive concern.** Pre-registering that Arm D has a predicted false-positive risk on C7 — "the determinism-principle concern, quantified" — is exactly the right framing. The design treats this as a disqualifier for Arm D rather than a bug to work around.

**The corpus anti-cherry-pick discipline.** The commitment to take C1/C2/C4/C5/C7/C8 verbatim from σ/η scratch bytes, with synthesized items written to category definitions before gate code exists, is the right protocol. It directly addresses the most common confound in labeled-corpus studies of detection gates.

**Cell A-live as the baseline (not a separate harness).** Running Cell A-live on the same trajectory as Cell B-live, using the same 5-file task, means the comparison is clean: the only difference is the gate. This is stronger than a synthetic baseline and directly replays the σ/η failure.

**The $0 cost constraint.** The design correctly holds the north-star constraint (cheap local coder, qwen3:8b) throughout and defers any frontier-seat diagnostic to an asked-before-spent bounded diagnostic. This applies the η/ξ precedent correctly.

---

## Overall Verdict

**Run with amendments — two P1 findings require pre-registration additions before running; five P2/P3 items are lightweight pre-run annotations.**

**P1-A** (C6 strawman) is the most consequential finding. The unification hypothesis's central claim — "B's miss-set = {C6}" — is not refutable against a C6 category defined narrowly as "clearly nonexistent function call." Either construct two additional C6 boundary-probe items (option a) or write an explicit operationalization of the C6/non-C6 boundary (option b) before running. Option (b) is the minimum; option (a) is preferred for the same reason ξ's corpus was strengthened by adding boundary-probe items.

**P1-B** (Fork 3 pass criterion unspecified) requires three pre-registration additions: n per cell, the margin that constitutes "recovers," and the framing of Cell A-live as the refuse-but-no-signal control. These are annotations only — no design changes. The question of whether a refuse-but-no-signal control is needed is answered: Cell A-live already plays this role structurally, but the design should state this explicitly so the causal interpretation is not left to post-run inference.

**P2-A** (Arm D C7 corpus not challenging enough for its false-positive prediction), **P2-B** (C4 re-pathing undocumented), and **P2-C** (recovery-loop re-delegation not traced) are each one-sentence additions to the pre-registration. None require additional runs.

**P3-A** and **P3-B** are optional improvements that cost one synthesized corpus item and one sentence respectively.

The design's core is sound. The corpus-based confusion matrix + live convergence arm is the right structure for a gate-selection spike of this class. The pre-registered refutability conditions for the unification hypothesis are the design's strongest feature. The ADR-035 composition (form gate feeds the refuse→un-produced→completeness-gate-re-delegates recovery loop) is correctly mapped and does not require redesign. The seven findings are all addressable before running with no additional live calls and at most two additional synthesized corpus items.

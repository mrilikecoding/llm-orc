# Argument Audit Report

**Audited document:** `docs/agentic-serving/decisions/adr-039-content-anchor.md`
**Source material:**
- `docs/agentic-serving/essays/research-logs/cycle-7-spike-xi-content-anchor.md`
- `docs/agentic-serving/housekeeping/audits/research-methods-spike-xi.md`
- `scratch/spike-xi-content-anchor/` (battery.log, per-cell results JSON, generated files)
- `docs/agentic-serving/decisions/adr-036-delegation-decision-mechanism.md`
- `docs/agentic-serving/decisions/adr-037-session-termination-two-call-composition.md`
- `docs/agentic-serving/decisions/adr-038-remaining-work-anchor.md`
**Genre:** ADR
**Date:** 2026-06-09

---

## Section 1: Argument Audit

### Summary

- **Genre:** ADR
- **Argument chains mapped:** 5 (Decision rationale; form selection; causal isolation; rejected alternatives; scope of consequences)
- **Issues found:** 5 (0 P1, 3 P2, 2 P3)
- **Pyramid coverage map:** N/A
- **Expansion-fidelity findings:** N/A

### Rate verification

Raw results JSON and battery.log verified against every figure cited in the ADR. All rates confirmed:

| Base | Arm | ADR claims | Raw JSON |
|------|-----|-----------|---------|
| T | A\_current | 3/10 | 3/10 confirmed |
| T | B\_signatures | 10/10 | 10/10 confirmed |
| T | C\_full | 8/10 | 8/10 confirmed |
| T | Control\_decoy | 0/10 | 0/10 confirmed |
| T | Control\_filler | 1/10 | 1/10 confirmed |
| V | A\_current | 0/10 | 0/10 confirmed |
| V | B\_signatures | 10/10 | 10/10 confirmed |
| V | C\_full | 10/10 | 10/10 confirmed |

Failure sub-classifications (invented/no-reference/parse-fail) confirmed verbatim against the battery.log trial-level records. No discrepancy found. The ADR's data layer is clean.

---

### P1 — Must Fix

No P1 findings.

---

### P2 — Should Fix

**P2-1: The Consequences/Positive bullet "Finding H blocker is removed at the dispatch layer" generalizes beyond the harness scope it was measured in.**

Location: §Consequences, Positive, bullet 1: "The Finding H blocker is removed at the dispatch layer."

The ADR's own Empirical Grounding section correctly qualifies this: the Conditional Acceptance discharge gate (the real-OpenCode re-run) is what confirms the in-harness result under the real client. But in the Consequences section the positive bullet is stated without qualification as though the blocker is already resolved in production. Finding H was itself a live-trajectory observation; the ADR explicitly notes "the harness PASS is necessary, not sufficient: the synthetic ladder passed axis-1 honestly and the real run still exposed axis-2." The positive bullet, as written, understates the remaining uncertainty in the one place where readers form their summary impression of what the ADR delivers.

Scope issue: the claim is accurate for "at the harness layer on qwen3:8b, single-hop file dependencies, Python." The current wording implies the blocker is closed for the full live use case, which is the very thing the discharge gate exists to confirm.

Recommendation: qualify the bullet to match the scope. For example: "The Finding H blocker is removed at the coder-generation harness layer (qwen3:8b, single-hop dependencies, Python); the real-client discharge gate confirms this under the full live stack." The existing Negative bullet about harness validation does the honest work, but the Positive bullet should not assert what the Negative bullet qualifies away.

---

**P2-2: The ADR Decision statement uses "all produced siblings" but the spike validated injection of a single relevant sibling; the gap between the two is acknowledged in Provenance but not in the Decision text or the FC.**

Location: §Decision, first paragraph: "the framework augments the callee dispatch context with the API signatures of those produced siblings." The FC (content-anchor presence) reads: "the callee dispatch context contains those siblings' API signatures."

The Provenance check honestly labels "the 'all produced siblings' signatures selection scope" as drafting-time synthesis: "the spike validated injecting the single relevant sibling; whether production injects all produced files' signatures or a dependency-inferred subset is a BUILD and FC detail, not a spike-measured property."

The problem is that the Decision text and the FC do not carry this qualification. A reader who reads the Decision and FC without reaching the Provenance check would believe the ADR commits to all produced siblings being injected, which is an untested design choice. Two distinct production approaches (all siblings vs. dependency-inferred subset) have meaningfully different correctness and context-budget implications, and the spike does not discriminate between them.

This is not a logical error in the decision (injecting the relevant sibling is what the evidence supports), but the Decision text overclaims relative to what was tested, which is exactly the kind of scope inaccuracy the audit is looking for.

Recommendation: add a qualification to the Decision text and the FC matching the Provenance note. Something like: "The spike validated the single directly-relevant sibling; whether the framework injects all produced siblings or a dependency-inferred subset is a BUILD design decision deferred to the ARCHITECT/BUILD phase." The FC should say "contains the directly-relevant produced siblings' API signatures" rather than "those siblings'" without further qualification, or carry an explicit "(scope: BUILD detail)" annotation.

---

**P2-3: The ADR-038 analogical framing ("rhymes with Finding G / the same structural shape") is sound as an analogical heuristic but slightly smuggles in ADR-038's empirical confidence by omission.**

Location: §Context: "This rhymes with Finding G / ADR-038 one layer down... the fix is the same shape, route the already-available signal forward." Also §Consequences, Positive: "The fix reuses already-available information... It rhymes with ADR-038, route the already-available signal forward."

The analogy is structurally apt — both are "route a computed/available signal forward" decisions grounded by a causal-isolation spike. But the analogy does one thing the ADR does not flag: ADR-038's Conditional Acceptance has already been discharged (the WP-LB-L real-OpenCode run, 2026-06-08), while ADR-039's is not. The rhyme framing lets the reader carry ADR-038's full-acceptance confidence forward to ADR-039, but ADR-039 is currently in the same conditional state ADR-038 was before discharge.

This is a framing risk rather than a logical error. The ADR's Empirical Grounding section correctly states the Conditional Acceptance status; the issue is that the rhyme framing in Context and Consequences positions it as already-resolved evidence at the same level as the Finding-G fix, when the parallel should be stated as "the same shape and the same discipline (including the discharge gate the analogy's original instance already cleared)."

Recommendation: when invoking the ADR-038 rhyme, note the epistemic asymmetry explicitly: "Finding G's fix has cleared its real-client discharge gate; Finding H's fix is at the same conditional stage before that gate." One sentence is enough; the Provenance check does not need to change.

---

### P3 — Consider

**P3-1: The causal isolation language in the Decision and Consequences sections claims "the specific sibling content is the mechanism" without noting the filler arm's graded-resolution observation.**

Location: §Context (third bullet): "neither API-shape alone nor extra tokens alone breaks the guessing; only the real sibling API does." §Consequences, Positive: "The specific-content mechanism is causally isolated."

The three-way decomposition cleanly supports this claim at the binary resolves level (0/10, 1/10 vs 10/10). However, the battery.log shows that the Control\_filler arm produced two trials (trials 1 and 4) with graded resolutions of 0.50 — meaning the model generated some correct references and some invented ones in the same file, without the anchor. These trials are correctly classified as "invented" (they contain at least one nonexistent reference) and correctly counted as non-resolving. The binary claim is accurate.

But the 0.50-graded trials in the filler arm suggest the model has weak partial priors for Base T's API (the three function names are not entirely unguessable — P3-A's shape-divergence argument held for the *unified* converter, but the individual names like `celsius_to_fahrenheit` are predictable enough that a filler arm can generate them 50% of the time by name). This does not undermine the causal isolation conclusion — B's 10/10 vs filler's 1/10 at the binary level is decisive — but the ADR's "only the real sibling API does" phrasing is slightly stronger than warranted. "The real sibling API is the reliable mechanism; partial priors exist for Base T's common-named functions but are insufficient without the anchor" would be more precise.

This is a P3 because the ADR's conclusion and the FC are sound; the filler-arm graded data is characterization, not a threat to the decision.

Recommendation: note in the Consequences section (or the Rejected Alternatives — Mere context perturbation section) that the filler arm produced partial-resolution trials on Base T, consistent with weak model priors for the common-named API shape, but that the binary resolves gate (every reference must resolve) is correctly chosen as the criterion. This keeps the causal claim precise.

---

**P3-2: The research log contains a stale pre-correction sentence naming qwen3:14b as the model for the generation arms; the ADR itself is internally consistent but the source material has a documentation artifact.**

Location: `docs/agentic-serving/essays/research-logs/cycle-7-spike-xi-content-anchor.md`, lines 41-45: "which content-anchor *form* fixes cross-file API guessing **on the cheap-local model (qwen3:14b)** — and is it the specific sibling content doing the work... So every powered arm runs $0 local qwen3:14b."

This paragraph is from the pre-review draft framing. The post-review fidelity correction section at the top of the same research log correctly states the arms ran qwen3:8b, the battery.log confirms qwen3:8b, and the ADR is entirely consistent on qwen3:8b throughout. The stale sentence in the middle of the research log is an artifact of the visible-flag correction discipline — the correction was added at the top but the superseded body text was not updated.

The ADR does not repeat or rely on the stale sentence. This is a source-material documentation artifact, not an ADR argument error.

Recommendation: update the research log to strike or annotate the stale qwen3:14b sentence in the decision framing paragraph (lines 41-45), making the post-review fidelity correction visible at the point of use rather than only in the status block at the top. This is a housekeeping edit to the research log, not a change to the ADR.

---

## Section 2: Framing Audit (ADR-061)

The framing audit makes the negative space of content selection visible — what the evidence could have supported that the ADR did not choose as its framing.

### Question 1: What alternative framings did the evidence support?

**Alternative framing A: "A path-routing fix may be sufficient; the framework could route only the sibling's path, not its content."**

The methods review incongruity section identifies this explicitly: the framework's existing action log already records the produced files' paths. A "route the path with a read instruction" fix (structurally similar to Arm D but at the framework layer, not the guidance layer) was not measured. The ADR frames the decision as "route the signatures," but the spike did not include a path-only arm that would confirm the content injection is necessary rather than merely sufficient.

The evidence that supports this alternative: the action log already carries file paths (the root-cause diagnosis); Arm D's rejected alternative operates at a different layer (seat-level guidance vs. framework injection) but the logical shape is similar. The ADR rejects guidance-only (Arm D) for the correct reasons (layer mismatch, relies on model choosing to read), but a framework-injected path-only dispatch — "here is the path to `converters.py`, use it" without the extracted signatures — is structurally available and not measured.

The ADR's framing positions "framework extraction and injection of signatures" as the answer, which it may well be. But the ADR cannot rule out that framework injection of the path alone would achieve a comparable rate, because that arm was not run.

Belief-mapping: for the path-only framing to be right, the reader would need to believe the model can reliably retrieve and process a file from a path reference without actually receiving the content in the dispatch context. Given that the live trajectory showed zero reads despite having the paths, this alternative is quite weak. The ADR's implicit rejection of path-only injection is well-grounded by the live-trajectory evidence, even without a controlled arm. The framing choice is defensible; it should be named as a conscious exclusion.

**Alternative framing B: "The real mechanism is API-format as a constraint, not the specific content."**

The methods review P1-A flags this. The filler arm result (1/10) and the decoy arm result (0/10) together confirm that "any-API-format" does not work and "specific content" does. But the filler arm result (1/10 vs 0/10 baseline) is so close to baseline that the "extra tokens" null is also effectively ruled out. The ADR's "specific content is the mechanism" framing is well-supported by the three-way decomposition.

The one residual: the P3-1 observation above shows that filler trials 1 and 4 had 0.50 graded resolution on Base T. The model does have weak priors for the common-named `celsius_to_*` functions. This means the mechanism claim should be characterized as "the specific content reliably breaks guessing where weak priors exist but are insufficient," not "the model has zero priors and only real content helps." The current framing is not wrong but is slightly absolute.

**Alternative framing C: "The generate-then-repair shape was available but not compared."**

The research log and ADR both note that generate-then-repair (generate the dependent file, AST-check it, re-generate with the specific failing references named) is the all-forms-fail backstop direction. The ADR frames the decision as "content-anchor injection is the mechanism" without naming generate-then-repair as a live alternative that was considered and ruled out for reasons beyond "the primary arms passed." The methods review names this as a premature-narrowing concern: the spike settled on the injection shape without measuring whether a post-generation repair pass might achieve comparable results with less framework complexity.

Belief-mapping: for generate-then-repair to be preferred over content-anchor injection, the reader would need to believe that AST-checking a generated file and re-dispatching with error feedback is cheaper (latency/cost) than framework-side signature extraction. Given that content-anchor injection adds one framework read and no extra model call, while generate-then-repair adds at least one extra model call per dependent file, the ADR's implicit rejection of this alternative is well-grounded by cost structure. Again, the framing choice is defensible; the exclusion should be visible.

### Question 2: What truths were available but not featured?

**Omitted observation A: The Control\_filler graded-resolution data.**

The battery.log shows two filler trials with 0.50 graded resolution (partial correct references mixed with invented ones). The ADR and research log report the binary resolves classification (1/10) correctly but do not mention the graded data in the mechanism characterization. This is the source of P3-1 above. The graded data is present in the artifacts and was computed during the run (the graded rate is a registered measurement in the research log), but the ADR's mechanism claim language is somewhat stronger than the full picture supports.

Why it may have been excluded: the binary resolves gate is the pre-registered go/no-go criterion, and the filler arm correctly fails (1/10). The graded characterization is secondary per the design. The omission is a scope decision, not an oversight. Including it would not change the decision, but would make the mechanism characterization more precise.

**Omitted observation B: The Control\_filler arm's "invents common names" behavior is informative for scope.**

Looking at the battery.log, the filler arm's failure mode is the model generating broadly plausible temperature-converter names (various `convert_*` / `c_to_f` / `celsius_to_fahrenheit` patterns) based on its training data. The ADR notes the Base T P3-A shape-divergence rationale but does not discuss what happens when a future base uses an API that is even more predictable than Base T. If the model's prior for a common API is strong enough, the filler arm might resolve at a non-trivial rate, and the anchor's value-add would be less clear. The ADR's scope boundary ("single-hop file dependencies, Python") does not capture this predictability dimension.

This is out-of-scope for the current ADR but worth noting for the discharge gate's evaluation: the 5-file temperature library task includes the same `converters.py` Base T API, so the real-client re-run will confirm (or not) whether the common-name partial-prior effect is large enough to matter in the real session.

**Omitted observation C: The discharge gate's README verification claim.**

The Empirical Grounding section lists as one of the discharge gate's verification steps: "the README documents real functions." The spike explicitly scoped prose→code coherence as out-of-scope and secondary: "A prose→impl characterization is run on Base T's README as a secondary measure... not a powered primary base." The signatures anchor was not measured for its effect on prose deliverables.

The discharge gate then asks the reviewer to verify that the README documents real functions — but the anchor mechanism (injecting code signatures into the callee dispatch) only fires for code-generating callees. Whether the README's callee (the `prose-improver` ensemble that generated the README in the live trajectory) would receive the same signature anchor is not addressed. If the `prose-improver` callee does not get the signatures anchor, the README failure mode from Finding H (invented Rankine scale, `fahrenheit_to_kelvin`) would not be addressed by ADR-039 at all.

The ADR's Negative section correctly records "prose-to-code coherence (the README's invented Rankine scale, a prose deliverable referencing code) remain the recorded boundary." But the discharge gate criteria quietly include "the README documents real functions" as a verification step, implying the anchor should fix the README failure. This is an internal inconsistency: the prose failure is listed as a scope boundary in Consequences but as a verification target in the discharge gate.

### Question 3: What would change if the dominant framing were inverted?

The dominant framing: "cross-file coherence fails because the sibling API is not fed to the coder; the fix is framework injection of signatures."

Inverted framing: "cross-file coherence fails because the model is not reading; the real fix is to make the model read, not to compensate for the model not reading."

Under the inverted framing:
- The framework injection is a workaround, not a fix — it compensates for the model's zero-read behavior rather than correcting it.
- The content-anchor approach makes the framework's context-management burden grow with the deliverable count (even with signatures being compact, the framework must now read and process every produced file before each dependent dispatch).
- The model's zero-read behavior in the live trajectory is treated as a permanent feature of qwen3:8b (which it may be), but the inverted framing would ask: "is there a way to make qwen3:8b read that was not tried?"

The ADR engages with this inversion (it is the core of the Rejected Alternatives / "Induce the model to read" section) and gives three good reasons for rejecting the guidance-only path: layer mismatch (the read decision is the seat's, not the coder's), read round-trip latency, and model reliability (zero reads in the live run). The framework injection is described as unconditional, which is the structural advantage.

What becomes more salient under the inverted framing: the injection approach means the framework now permanently carries the burden of knowing which files are relevant siblings, extracting their API surfaces, and injecting them — this is new framework infrastructure that the model-reads-when-told approach would not require. The "additive to ADR-036" framing in the ADR is accurate, but "additive" means "new framework component with a new correctness-critical path" (the extractor), which is a real cost. The ADR's Negative section notes this honestly (new framework step, language-specific extraction, extractor is a correctness-critical component), so the framing is not suppressing it — but the inversion makes it the lead concern rather than a bullet.

### Framing Issues

**FI-P2-1: Discharge gate includes "README documents real functions" but prose coherence is explicitly out of ADR-039's scope.**

Location: §Empirical Grounding, discharge gate criteria: "verified by reading the landed files (`cli.py` calls the real `converters` functions, the tests match the implementation, **the README documents real functions**)."

§Consequences, Negative: "prose-to-code coherence (the README's invented Rankine scale, a prose deliverable referencing code) remain the recorded boundary; the prose failure shape in particular may need a prose-targeted variant this ADR does not cover."

These two statements are in tension. The discharge gate includes README coherence as a verification target, but the README is generated by the `prose-improver` ensemble, which may not receive the signatures anchor (the anchor fires on "delegated generation that writes a file into a session with already-produced file deliverables" — a prose deliverable could qualify, but the mechanism only helps if the prose callee's dispatch context is also augmented). The ADR does not specify whether the anchor applies to prose-generating callees.

If the signatures anchor does reach the `prose-improver` callee, then listing README coherence in the discharge gate is consistent with the mechanism. If it does not (because `prose-improver` generates prose, not code, and the signatures are only injected into code-generation dispatches), then the discharge gate is testing something the mechanism was not designed to fix.

This is a P2 framing issue because it creates ambiguity at the gate: an evaluator could reasonably disagree about whether the README criterion counts as a pass or fail for this ADR's mechanism, or about whether a README failure means the ADR's scope is narrower than stated.

Recommendation: either (a) clarify in the discharge gate that README coherence is a characterization observation (the scope boundary is prose deliverables), and a README failure at the gate would open a prose-anchor follow-up, not block ADR-039's full acceptance; or (b) extend the Decision section to address whether the signatures anchor fires on prose-generating callees and, if so, on what grounds.

---

**FI-P3-1: The path-only injection alternative is excluded without being named as a considered-and-rejected option.**

Location: §Rejected Alternatives — this section covers full-content anchor, guidance-only read-back (Arm D), a more capable coder seat, and the null controls. It does not mention the framework-injected-path-only option.

The methods review incongruity section identifies this as the unasked question: "is there a framework intervention simpler than signature extraction — specifically, routing the sibling's file path from the existing action log into the callee context as an explicit read instruction — that would achieve comparable resolution without new extraction infrastructure?"

The ADR rejects guidance-only (Arm D) with solid reasons. A framework-injected path (not guidance, but a structured framework field) is a distinct option that is structurally simpler than signature extraction. It was not measured, and the live-trajectory evidence (zero reads despite paths being in the action log) strongly suggests it would not work — but since the mechanism relies on a model read, not content injection, the zero-read evidence is directly applicable. The rejection is implicit and sound; it should be explicit.

Recommendation: add a brief Rejected Alternatives entry for "framework-injected path reference without content extraction," citing the live-trajectory evidence (zero reads across the full run even with paths in the action digest) as the rejection basis. This closes the options space for readers who see the methods review incongruity note.

---

## Convergence-Saturation Signal (ADR-094)

**Convergence-Saturation Signal:** NOT TRIGGERED

- Round number: R1 (ADR-039 is a new ADR; baseline resets per ADR-094 form-change baseline-reset rule)
- P1 count this round: 0
- P2 count this round (new, non-carry-over): 4 (P2-1, P2-2, P2-3 from argument audit; FI-P2-1 from framing audit)
- New framings or claim-scope expansions: the discharge-gate / prose-scope tension (FI-P2-1) is newly surfaced; the all-siblings-vs-single-sibling gap in the Decision text (P2-2) is newly surfaced; the path-only injection omission from Rejected Alternatives (FI-P3-1) is newly surfaced
- Recommendation: CONTINUE to R2

*P2 count (4) exceeds the ≤1 threshold for signal trigger. No P1 findings; the ADR's evidence layer and causal isolation are sound. The R2 revision target is: P2-2 (all-siblings scope qualification in Decision and FC), P2-1 (Positive bullet qualification), FI-P2-1 (discharge gate README criterion disambiguation), and P2-3 (ADR-038 rhyme epistemic-asymmetry note). P3 and FI-P3-1 are lower priority but can be addressed in the same revision pass.*

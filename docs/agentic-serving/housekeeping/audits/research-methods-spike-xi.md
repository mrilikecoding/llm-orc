# Research Design Review — Spike ξ (Content Anchor, ADR amendment for Finding H)

**Reviewed question set:** `docs/agentic-serving/essays/research-logs/cycle-7-spike-xi-content-anchor.md` — full pre-registered design (arms A/B/C/D/Control_decoy, two bases T/V, hypotheses H-ξ.1 through H-ξ.4)
**Constraint-removal response included:** n/a (mechanism-amendment spike; no ADR-082 constraint-removal artifact in scope)
**Date:** 2026-06-08
**Reviewer role:** research-methods (ADR-060 + ADR-082 dimensions 1–4, θ/ρ-audit calibration bar applied)

---

## Summary

- **Arms reviewed:** A_current (T, V), B_signatures (T, V), C_full-content (T, V), D_read-induce (T only), Control_decoy (T only) — 9 arm/base cells, n=10 each
- **Hypotheses reviewed:** H-ξ.1, H-ξ.2, H-ξ.3, H-ξ.4
- **Flags raised:** 8 (2 P1, 4 P2, 2 P3)
- **Criteria applied:** 1–4 (ADR-082)

---

## Per-Hypothesis Review

### H-ξ.1: "A_current reproduces Finding H in the harness at a high rate."

**Belief-mapping.** The hypothesis assumes the harness faithfully captures the failure Finding H exhibited. What would the researcher need to believe for a different question to be more productive? They would need to believe the two pinned bases (Base T: converters→cli; Base V: string_utils→test) actually recreate the informational conditions that caused Finding H — specifically, that the callee receives the same context structure (action/path/result framework digest, no file contents) as the live trajectory callee did. If the harness's composed dispatch context differs from the live production dispatch (e.g., a different tail structure, fewer prior action records, or an atypical sibling that is simpler than the live converters.py), the baseline failure rate may not reproduce, and the guard fails regardless of whether the fix would work in production.

The belief most worth examining: **the pinned siblings are simple enough that qwen3:14b may partially guess them correctly even blind.** `converters.py` exposes three functions with predictable names (`celsius_to_fahrenheit`, `fahrenheit_to_celsius`, `celsius_to_kelvin`). A capable code-generating model trained on temperature converters may guess these names without seeing the file, producing a false-low A_current failure rate. Base V is even more exposed: `reverse_words` and `count_vowels` are generic, widely-used names. If A_current resolves ≥ 7/10 on either base, the design stops correctly — but the threshold should be evaluated against this risk before running, not discovered at run time.

**Embedded conclusions.** The hypothesis correctly frames itself as a guard rather than a hoped-for result. No embedded conclusion issue.

**Scope.** Appropriate. This is the prerequisite check the design correctly identifies as the pre-condition for any fix measurement.

---

### H-ξ.2: "At least one of B/C/D raises cross-file reference resolution to a shippable rate while preserving delegation."

**Belief-mapping.** The hypothesis frames success as "at least one passes" — a disjunction over three qualitatively different forms. What would the researcher need to believe for a different framing to be more productive? They would need to believe that the three forms are alternatives, not a progression — i.e., that there is no case where B fails but C passes for a systematic reason that is worth investigating beyond "B failed, try C." The disjunction framing is correct for a form-selection spike, but it suppresses the question: if all three fail, what does the failure pattern across B/C/D tell us about whether the gap is capability, context-format, or context-volume?

The hypothesis also embeds "shippable" without defining it. The decision rule defines a form as passing at ≥ 7/10 `resolves`. But `resolves` is binary per trial (every cross-file reference targets an existing symbol), meaning a generated file that uses 4 out of 5 real functions but invents a sixth does not resolve. If the typical failure mode is "almost correct — one invented reference among several real ones," the binary resolves rate may be a harsher criterion than the failure mode warrants, while still being directionally right. The graded resolution rate is collected as characterization but does not factor into the pass rule. This is defensible but worth naming.

**Embedded conclusions.** "Shippable rate" hides the 7/10 threshold's motivation. See P2-A.

**Scope.** Appropriate.

---

### H-ξ.3: "B_signatures resolves materially above Control_decoy — the specific sibling content is the mechanism."

**Belief-mapping.** The hypothesis asks whether real-sibling content is the mechanism. The adjacent question the framing suppresses: **is a content-wrong decoy the right causal control, or are there two distinct alternative explanations that require two distinct controls?** The decoy isolates "specific content vs. same-shape API-format content." It does not isolate "specific content vs. any extra tokens in the dispatch." If the model sees API-formatted context and improves its resolution regardless of whether the names are real or invented (because the API format itself constrains the output to function-call-shaped references), the decoy distinguishes real from invented names. But it cannot distinguish "API format is the mechanism" from "specific names are the mechanism" — both would produce B > Control_decoy, and the decoy only controls for exact name matching, not for API-format-as-constraint. See P1-A (the generic-filler control question).

**Embedded conclusions.** "The specific sibling content is the mechanism" frames the mechanism question as binary: real-content vs. no-content. The design correctly surfaces the alternative (any-API-shaped context is enough) via the decoy. The question is whether the decoy fully isolates the claim. See P1-A.

**Scope.** Appropriate. This is the load-bearing causal question, correctly identified as such.

---

### H-ξ.4: "Signatures (B) suffices — full-content (C) does not materially exceed it — OR data show a fidelity gap that justifies the heavier form."

**Belief-mapping.** The hypothesis frames the B-vs-C question correctly as a two-sided test: either result is informative. What would the researcher need to believe for a different question to be more productive? They would need to believe that the B/C comparison at the global-resolve level adequately captures the fidelity tradeoff. A different question: "Does B suffice for simple functions but degrade for functions with non-trivial signatures (default arguments, type annotations, overloads)?" The pinned siblings are simple. If B suffices on these bases, it may fail on bases with richer APIs — and the form-selection decision commits to B without data on that brittleness frontier. This is arguably out of scope for a single spike, but the ADR amendment should bound the claim accordingly (B validated for simple linear APIs; richer-signature bases not tested).

**Embedded conclusions.** The ≥ 0.2 form-selection threshold (B-vs-C) is not motivated. See P2-A.

**Scope.** Appropriate as a form-selection discriminant. The scope concern (simple pinned siblings) should be recorded as a named bound on the amendment's claim.

---

## Question Set Assessment

### Premature narrowing / prior-art treatment

The design treats Finding H as a structural information-flow gap and positions the content-anchor spike directly against that gap. The ρ precedent is treated as prior art explicitly (the P1-B lesson, the control design, the measurement-at-the-decision-call discipline are all carried forward). No premature narrowing in the core analytical frame.

One narrowing concern: the design has settled on "inject context into callee dispatch" as the fix shape, and all three powered arms (B/C/D) are variants of that shape. D appears to be a different shape (guidance-only), but in practice it still assumes the callee-dispatch layer is the right intervention point. No arm considers an alternative intervention point: a post-generation repair pass (generate → AST check → re-generate with specific error feedback). That alternative was named in the Finding H results note ("generate-then-repair against a real import check" appears only as the all-forms-fail backstop redesign direction). The design is right not to over-scope the spike, but the ADR amendment should record that generate-then-repair was not compared against content-anchor, only defaulted to on all-forms-fail. This is not a redesign-level concern; it is a named scope bound.

### Incongruity surfacing

An incongruity is present in the research context and the question set does not surface it for examination.

The existing action-log that the framework produces records each prior action as `(action, path, result)`. This is a structural log that already causes the callee to know *which* files were produced and *whether* they succeeded — it just does not include content. The design injects sibling content into the dispatch as the fix. But the design's own Base T uses `converters.py` — a simple, predictable-named module with three functions. The framework could deterministically extract and route that content without model involvement (it already knows the path from the action log; a framework `read(path) → signatures` is a mechanical extraction, not a generation step).

The incongruity: the remaining-work-anchor (ρ's fix for Finding G) was a framework-side signal-routing decision (route the judge's computed output forward). The content-anchor (ξ's fix for Finding H) is framed as a framework-side signal-routing decision too ("route already-available sibling content forward"). Yet the ρ design never had to ask "should the framework extract signatures mechanically, or should the model do it?" because the judge's statement was already the artifact to route. For ξ, the question of whether signature extraction is mechanical (framework) or model-assisted (arm D's read-induce, which asks the model to read and then write) is open — but the spike's arms are all framed from the injection side, not the extraction side. Arm D's mechanism (model reads before writing) is the one arm that mirrors D's extraction being done by the model rather than the framework. The design does not surface the question: "is the right intervention framework-side extraction (always inject signatures) or model-side extraction (instruct the model to read)?" as a first-class fork — it surfaces it as a D-vs-B/C harness comparison but describes D as a "guidance-only fix" rather than as an extraction-side alternative.

This matters because the incongruity is: ρ's fix was clean because the framework already computed the signal and just wasn't routing it. ξ's fix requires new framework behavior (extracting and injecting sibling signatures) that ρ did not. The complexity increase may be appropriate, but the question "is there a simpler framework intervention that achieves the same result as B/C without signature extraction (e.g., simply routing the existing action-log entry's path into the callee context with a 'you should read this' note, equivalent to D but without the tool-call overhead)" is not explicitly asked. D-as-designed is close to this question but confounds guidance quality with mechanism (it instructs reading but does not guarantee the read happens; read-fired is an outcome, not a mechanism guarantee).

The question the set is missing: **"Is there a framework intervention simpler than signature extraction — specifically, routing the sibling's file path from the existing action log into the callee context as an explicit read instruction — that would achieve comparable resolution without new extraction infrastructure?"** This is structurally adjacent to D but cleaner: the harness already has the path; the question is whether "you should read converters.py before writing cli.py" (path-only, no content) is as effective as B. The design does not include a path-only arm. If path-only works, it is the simplest fix and the design should know.

---

## Findings

### P1 — Design flaws that would invalidate conclusions before run

**P1-A: The content-wrong decoy conflates two alternative explanations and may not fully isolate the claimed mechanism.**

The causal isolation control is a content-wrong decoy: signatures of the same shape as the real sibling but with plausible-but-wrong names (the very names Finding H invented: `convert_temperature`, `to_kelvin`, `scale`). The design correctly notes this is "sharper than a generic-length filler control: it isolates *use of the real content* rather than merely *presence of extra tokens*."

This is partly right but incomplete. The decoy control tests: "does the model use the specific real names rather than invented names?" It can produce two distinct outcomes:

1. B resolves ≥ 7/10, Control_decoy resolves near zero → the specific real names are what matter. The design's intended inference.
2. B resolves ≥ 7/10, Control_decoy resolves comparably → any API-formatted context helps, not just real names.

What the decoy cannot distinguish: "the API format itself (any structured function signature presentation) constrains the model to produce function-call-shaped output rather than invented method calls" from "the specific real names are the mechanism." Both cases produce B > Control_decoy, because the decoy uses wrong names and B uses right names. But a third case the design does not control for: "any extra tokens in dispatch-context format constrain the output, regardless of whether those tokens are API-formatted." A generic-length filler control (e.g., a docstring of similar length with no function names) would distinguish API-format-as-constraint from extra-tokens-in-general.

The ρ methods review explicitly recommended against replacing a content-neutral control with a content-wrong decoy — ρ's P1-B used a content-neutral trailing perturbation, not a content-wrong decoy, precisely to isolate content from format from tokens. The ξ design flips this: it uses a content-wrong decoy and explicitly rejects a generic-length filler control. The stated reason is that the content-wrong decoy is "sharper" — it isolates use of the real content rather than mere token presence. This is correct as far as it goes, but it leaves the API-format-as-constraint alternative uncontrolled.

In practice this may not matter if Control_decoy passes at a high rate (resolves comparably to B): that finding would falsify "specific names are the mechanism" and point at "API format" or "extra tokens" — both would be characterized as "not the specific content." But if Control_decoy resolves moderately (e.g., 4/10 vs. B's 8/10), the design cannot distinguish "API format partially helps but real names are better" from "the decoy names partially matched real patterns the model already knew." The gap-threshold (B − Control_decoy ≥ 0.3) does not have a companion interpretation for the intermediate case.

**Recommended design change (two options, in preference order):**

Option 1 (preferred, costs 10 additional calls at $0 local): Add a `Control_filler` arm on Base T — generic-length context of similar format to B (e.g., docstrings and placeholder type annotations with no function names, just `def func_a(x: type) -> type: ...`-style placeholders). Run n=10 alongside Control_decoy. This produces a three-way isolation: filler < decoy < B would confirm API-format partially helps, real content fully helps; filler ≈ decoy < B would confirm it is the specific names; filler ≈ decoy ≈ B would confirm extra tokens are the mechanism. The full isolation is worth 10 additional calls.

Option 2 (no additional calls): Pre-register the intermediate-case interpretation. If B − Control_decoy ∈ [0.1, 0.3), name the interpretation: "API-format partially constrains output; specific names add marginal additional signal; the amendment ships as API-format anchor with a named caveat." If B − Control_decoy ≥ 0.3 and Control_decoy ≤ 3/10, name the interpretation: "specific real content is the mechanism." The gap-threshold without a filler control means the 0.3 threshold at n=10 has ±wide CI (see P2-B), so pre-naming the intermediate case prevents post-hoc drift.

The P1 severity here is bounded: both outcomes (B wins decisively vs. B wins moderately) still support shipping the content-anchor. The causal characterization in the ADR changes, but the decision does not necessarily change. The design flaw would invalidate only the strong mechanistic claim ("specific sibling content is the mechanism"), not the operational finding. Classify this as **P1** because the hypothesis H-ξ.3 specifically makes the mechanistic claim and the ADR amendment would record it as fact if the decoy-only control is treated as sufficient evidence.

---

**P1-B: Arm D's harness asymmetry may not invalidate cross-arm comparison, but the `resolves` denominator for D is underspecified in a way that could silently inflate the D rate.**

The design notes honestly that D runs a different harness (read-tool loop) than B/C (single injected-context generation). The concern flagged in the review prompt — does this asymmetry invalidate cross-arm comparison — is real but partly mitigated by the design's own honesty: D is labeled as a separate mechanism-fork probe, and the outcome records both `read-fired` and `resolves`. If `read-fired` is high and `resolves` is high, D is a viable alternative; if `read-fired` is low, D is refuted regardless of `resolves`.

The sharper problem is the denominator. For B and C, every trial produces a generated file and the `resolves` check runs on that file. For D, the harness runs a read-tool loop — the seat has the read tool and receives a trailing instruction to read the sibling before writing. In a tool-call loop, the seat may:

(a) Issue a read, then write the dependent file → `resolves` check runs on the write output.
(b) Issue a read, then produce a text response (no write) → no generated file; what is the trial classified?
(c) Issue no read and no write (stall) → no generated file; classified how?
(d) Issue a write without reading → `read-fired = 0`, but a generated file exists; `resolves` runs on it.

The design specifies `read-fired ≥ 1 read of the sibling before writing` but does not specify what happens to the `resolves` denominator when no write is issued (cases b/c). The ρ precedent was explicit: denominator is always n, with `none` contributions when no tool call was made. The ξ design carries this discipline for A/B/C (`no-reference` is a distinct classification, not counted as `resolves`), but the D outcome section does not address the case where the seat issues a read but then stalls without writing. If case (b/c) is classified as `no-reference` or `none`, it silently enters the denominator as a non-resolving trial — fine. But if it is excluded from the denominator (classified as "harness error" or "no deliverable"), the D resolves rate is inflated on a smaller denominator.

**Recommended design change:** Pre-register the D trial classification table before running:

- `read + write → resolves/invented classification per the standard rule`
- `read + no-write → resolves = false, read-fired = true, classified as "stall-after-read"`
- `no-read + write → resolves/invented classification; read-fired = false`
- `no-read + no-write → resolves = false, read-fired = false, classified as "stall"`
- `denominator for D resolves rate: always n=10 (stall cases counted as non-resolving)`

Report `stall-after-read` and `stall` counts alongside `read-fired` and `resolves`. This requires no additional runs; it is a pre-registration annotation that prevents post-hoc denominator choices.

---

### P2 — Weaknesses that bound the claims

**P2-A: The B-vs-C form-selection threshold (≥ 0.2 gap) and the causal-isolation threshold (≥ 0.3 gap) are not statistically distinguishable at n=10, and the thresholds are not motivated by reference to any measured prior.**

At n=10 per cell, observed rate differences of 0.2 and 0.3 fall entirely within the range of sampling noise. Clopper-Pearson 95% CI for 7/10 = [0.35, 0.93]; for 8/10 = [0.44, 0.97]. A 2/10 difference (the B-vs-C form-selection threshold) is not a distinguishable population-rate signal at this n. A 3/10 difference (the causal-isolation threshold) is slightly more visible but still consistent with a wide range of true rates.

The issue is not that the thresholds are wrong — they are calibrated to be directional go/no-go signals, consistent with the ρ precedent. The issue is that the thresholds are presented as precise discriminants (0.2 for form selection; 0.3 for causal isolation) when neither threshold can be treated as a reliable population-level signal at n=10. A post-run outcome of B = 7/10, C = 8/10 (C's rate minus B's rate = 0.1, below the 0.2 threshold, prefer B) is not distinguishable from B = 7/10, C = 9/10 (0.2 gap, borderline for C preference). At n=10, these two outcomes have overlapping confidence intervals.

The motivations for 0.2 and 0.3 specifically are not stated. The ρ review named the same concern (P2-B) about the [0.5, 0.7) band, and the ρ design addressed it by justifying the band motivations. ξ carries the same threshold precision risk without the stated justification.

**Recommended design change:** Add a one-paragraph motivation section for both thresholds before running:

- The 0.2 B-vs-C threshold: justified as a "meaningful context-budget tradeoff deserving the heavier form" level, where the design's prior expectation is that B and C perform similarly (simple APIs; signatures fully capture the dependency surface). The 0.2 gap is not a precision rate estimate; it is a bar above which the heavier payload has demonstrated fidelity advantage that justifies its context-budget cost.
- The 0.3 causal-isolation threshold: justified as the ρ precedent threshold (ρ's causal isolation applied the same 0.3 bar, and ρ's control came in at 0/10 vs. B's 8/10 — a 0.8 gap that made the threshold moot). The 0.3 bar is a minimum credibility level for claiming "specific content is the mechanism" rather than "any context helps." Flag explicitly in the pre-registration that at n=10 this threshold carries wide CIs and the ADR records it as a directional discriminant, not a precision rate.

No additional runs required.

---

**P2-B: The `no-reference` escape hatch creates a conditional denominator that could inflate the `resolves` rate in a non-obvious way, and the design does not specify how `no-reference` interacts with the H-ξ.1 guard.**

The design specifies: "A trial that makes zero cross-file references is classified `no-reference` (the deliverable did not engage the dependency at all — a distinct failure from `invented`), reported separately, not counted as `resolves`."

This is the right conceptual distinction (`no-reference` is a distinct failure mode, not a success). But the denominator interaction is underspecified. If `no-reference` trials are "reported separately, not counted as `resolves`," are they also not counted as `invented`? If so, the effective denominator for the `resolves` rate is n minus `no-reference` count — which inflates the resolves rate when anchor injection causes the model to avoid cross-file references entirely (a plausible escape: the model sees the sibling's API surface and, uncertain about its own knowledge, writes a file that does not import the sibling at all).

Consider: if B_signatures produces `no-reference = 3/10, resolves = 6/10, invented = 1/10`, and the decision rule requires `resolves ≥ 7/10`, B fails. But if no-reference is excluded from the denominator, the resolves rate becomes 6/7 = 0.86, which passes. This is a real risk: anchor injection might make the model more conservative (fewer invented references but also fewer real references), and the denominator choice determines whether that's a pass or fail.

The ρ precedent was explicit: denominator is always n. The ξ design says `no-reference` is "reported separately, not counted as `resolves`" — ambiguous about whether the denominator stays n.

**Recommended design change:** Explicitly state in the decision-rule section: "`resolves` rate denominator is always n=10. `no-reference` trials count in the denominator as non-resolving outcomes. The `no-reference` count is reported alongside `resolves` and `invented` as characterization." This prevents a denominator escape that would pass B in the scenario described above.

The H-ξ.1 guard also applies to `no-reference`: if A_current produces high `no-reference` (the baseline does not engage the dependency), the guard should fire on `resolves + no-reference ≥ 7/10` (either the model is resolving or avoiding the dependency — both are non-invented, and a baseline that already avoids the dependency does not capture Finding H). This interaction is not in the pre-registered guard clause. Recommended addition: guard triggers if `A_current resolves + no-reference ≥ 7/10` (the model is already either resolving or avoiding the dependency — the harness does not capture the failure).

---

**P2-C: The all-forms-fail → cloud-arm trigger has an ambiguous decision boundary when one or two forms land in the Conditional band [0.5, 0.7).**

The design specifies: "All forms fail (none ≥ 7/10 on a base): trigger the bounded cloud contingent arm." The Conditional-band [0.5, 0.7) specifies: "the amendment proceeds as Conditional Acceptance with the real-OpenCode trajectory re-run as the discharge gate." The cloud arm trigger is "all-forms-fail or Conditional-band."

The ambiguity: "Conditional-band" is not a form-level classification in the design — it appears as an afterthought in the failure-mode backstops. The all-forms-fail trigger fires when no form is ≥ 7/10 on a base. But if B = 0.6 and C = 0.8 on Base T, the result is: B in the Conditional band, C passes. Does the cloud arm run? The decision rule says the cloud arm runs on "all-forms-fail or Conditional-band trigger" — but C passed. The cloud arm's purpose is to "discriminate capability-vs-structure share" when the local battery is ambiguous. If C passes, the battery is not ambiguous at the form level.

The reverse: if B = 0.6, C = 0.6, D = 0.6 (all in the Conditional band), the cloud arm runs. But the design says the Conditional band means "Conditional Acceptance with real-OpenCode re-run as discharge gate" — so the cloud arm and the real-client discharge gate both run? On what form?

**Recommended design change:** Add a decision table to the pre-registration that disambiguates the cloud-arm trigger:

- At least one form passes (≥ 7/10) on at least one base: cloud arm does not run; real-client discharge gate runs with the winning form.
- All forms ∈ [0.5, 0.7) on all bases: Conditional Acceptance; real-client discharge gate runs; cloud arm runs if the practitioner wants the capability-vs-structure diagnostic (asked-before-spent as stated).
- At least one form < 0.5 and no form ≥ 0.5: all-forms-fail; cloud arm trigger fires.
- Mixed (some forms below 0.5, some in [0.5, 0.7), none ≥ 0.7): the design is silent; recommend treating this as "all-forms-fail on the basis that no form demonstrates reliable uplift" and triggering the cloud arm.

No additional runs required.

---

**P2-D: The `resolves` binary does not capture the failure modes Finding H exhibited at the prose and syntax layer, and the secondary measures as specified may not flag harness-level failures in time to stop a run that has already generated bad outputs.**

Finding H had three distinct failure modes in the live trajectory: (1) invented cross-file function name (`cli.py` called `convert_temperature`), (2) syntax error (`args = parser.parse,args`), (3) README prose documented functions that do not exist. The design maps these to: (1) `invented` classification via AST symbol resolution — directly captured; (2) `syntactic-valid` secondary measure — captured; (3) identifier-mention characterization on Base T's README — captured as secondary only, not a powered base.

The concern is not coverage completeness — the design correctly prioritizes AST-checkable primary outcomes. The concern is the `syntactic-valid` secondary measure's relationship to the `resolves` primary. A file that fails to parse (syntax error) cannot have its cross-file references checked by AST. The design does not specify what happens to the `resolves` classification for a trial that is syntactically invalid. If the trial is classified `no-reference` (no AST → no references found → zero cross-file references), it would silently enter the `no-reference` category rather than the `invented` category, despite being a severe failure. If it is classified as a separate failure class, that is the right behavior.

**Recommended design change (minor, no additional runs):** In the measurement definitions, add: "A trial where the generated file fails to parse (syntactic-valid = false) is classified `parse-fail`, counted in the denominator as a non-resolving outcome, and reported separately from `invented` and `no-reference`. The `resolves` rate denominator includes `parse-fail` trials." This prevents a silent denominator manipulation where bad outputs (syntax errors) are swept into `no-reference` rather than counted against the form.

---

### P3 — Improvements

**P3-A: The pinned sibling names risk predictability-inflated A_current failure rates on Base V, and the guard threshold asymmetry between "too high" and "too low" baseline should be flagged.**

The design correctly identifies Base V's pinned sibling as `string_utils.py` with `reverse_words` and `count_vowels`. These are among the most common example function names in Python tutorials and code documentation. A model generating `test_string_utils.py` without seeing the sibling may produce correct references to `reverse_words` and `count_vowels` from training data alone, not from any structural information. This is the opposite of the predictability concern in H-ξ.1's analysis: Base V may produce a falsely-low `invented` rate for A_current, making the fix look necessary when A_current already passes — or, after the fix, confounding whether B/C resolved because of the anchor or because the model already knew the names.

**Recommended design change:** Before running, evaluate whether the Base V sibling names are distinctive enough that the model cannot correctly guess them blind. Option: substitute the `string_utils` sibling with a module exposing less-generic names (e.g., `def strip_repeated_chars(text: str, char: str) -> str` and `def normalize_whitespace(text: str) -> str`). These are real utility functions but not "first guess" names. The base should be pinned with names that require the sibling to be read or injected to resolve correctly. This change requires no additional runs but does require a base re-pin before running — a pre-run design change.

If the practitioner decides generic names are intentional (the finding should hold even for the easiest case), the rationale should be pre-registered explicitly so the A_current guard is interpreted in context: "A_current passing at ≥ 7/10 on Base V may reflect name predictability, not anchor effectiveness; Base T is the primary validity check for the guard."

---

**P3-B: The design does not specify what "composed through the landed callee-dispatch path" means precisely for arm D, creating a reproducibility gap.**

The design states D "runs the read-tool loop" as distinct from B/C's "single injected-context generation." The fidelity discipline section commits to composing through the landed `_seat_filler_messages` / delegation composition path. But for D, the harness must:
1. Append a trailing read instruction to the callee dispatch context.
2. Give the seat the read tool.
3. Run a multi-turn loop (read response → process tool call → continue → write response).

Step 3 is not "single injected-context generation" — it requires a tool-call processing loop that the current harness likely does not have (the ρ harness is a single call, not a loop). The design notes "harness differs — it runs the read-tool loop, noted honestly" but does not specify how the loop terminates (after the first write? after a fixed number of turns? if the model stalls?), what the loop's tool-call processing looks like, or whether the retained artifact is the write tool-call's content or the final generated file text.

This is not a severity-P1 concern because D is an optional mechanism-fork probe on one base, not a primary form. But the reproducibility gap means D's results may be difficult to audit against the retained artifacts unless the loop structure is documented.

**Recommended design change:** Add a D-harness specification section: loop terminates on first write tool-call or after N turns (pre-register N); retained artifact is the write tool-call's `content` argument; if the loop terminates without a write, the trial is `stall`. This requires no additional runs and prevents the D results from being unauditable.

---

## Overall Verdict

**Run with amendments — two P1 findings require pre-run design changes; one is lightweight (P1-B classification table), one has a preferred additional-10-call option (P1-A filler control) and a no-cost alternative.**

**P1-A** (decoy control leaves API-format-as-constraint uncontrolled): the preferred fix is 10 additional calls ($0 local, Control_filler arm on Base T). The no-cost alternative is pre-naming the intermediate-case interpretations so the gap-threshold does not drift post-result. The decision to run or characterize is the practitioner's under the free-options discipline (the filler arm costs nothing and adds clean isolation; the pre-naming alternative is cheaper but leaves the mechanistic claim partially unsupported). Both options are sound; neither requires redesigning the battery.

**P1-B** (D trial classification table underspecified): requires a classification table added to the pre-registration before running. No additional calls. ~10 minutes of pre-run annotation.

**P2-A** (threshold motivation): requires a one-paragraph justification of the 0.2 and 0.3 thresholds before running, identical to the ρ review's P2-B recommendation carried forward. No additional calls.

**P2-B** (`no-reference` denominator ambiguity + H-ξ.1 guard interaction): requires two sentence additions to the decision-rule section. No additional calls.

**P2-C** (cloud-arm trigger decision boundary): requires a decision table in the pre-registration. No additional calls.

**P2-D** (`parse-fail` classification gap): requires one sentence in the measurement definitions. No additional calls.

**P3-A** (Base V name predictability): evaluate before running; may require a base re-pin. No additional calls if re-pinned.

**P3-B** (D harness loop specification): document before running. No additional calls.

The design's core is sound. The multi-arm structure with a causal-isolation control is the right shape (carrying ρ's P1-B lesson forward). The AST-checkable primary outcome correctly sidesteps ρ's adjudication subjectivity concern. The real-client discharge gate being first-class rather than deferred correctly applies the Finding H lesson. The bounded cloud contingent arm is properly gated (asked-before-spent). All of this meets or exceeds the ρ calibration bar. The seven P2/P3 items are all lightweight pre-registration annotations that require no additional runs. P1-A is the only finding that warrants a genuine design decision (filler arm vs. pre-named interpretation) and costs at most 10 additional $0 local calls if the practitioner chooses the cleaner option. The H-ξ.1 predictability risk (especially Base V) is the sleeper concern: if the guard fires wrong on Base V, the spike loses one of its two bases. Evaluate that before running.

# Argument Audit Report

**Audited document:** docs/agentic-serving/decisions/adr-035-client-tool-deliverable-form-contract.md
**Source material:**
- docs/agentic-serving/decisions/adr-024-common-io-envelope.md
- docs/agentic-serving/decisions/adr-033-layer-a-loop-driver-multi-turn-agentic-surface.md
- docs/agentic-serving/decisions/adr-034-client-tool-action-terminal-artifact-bridge.md
- docs/agentic-serving/decisions/adr-025-artifact-as-substrate.md
- docs/agentic-serving/essays/research-logs/cycle-7-spike-phi-deliverable-shape.md
- docs/agentic-serving/essays/research-logs/cycle-7-spike-chi-deliverable-shaping.md
- docs/agentic-serving/essays/research-logs/cycle-7-wp-lb-c-opencode-validation.md
**Genre:** ADR
**Date:** 2026-06-03

---

## Section 1: Argument Audit

### Summary

- **Genre:** ADR
- **Argument chains mapped:** 6 (D1 extraction, D2a inert contract, D2b form drift, boundary-directive reliability, granularity invariant, ADR-024 carve-out coherence)
- **Issues found:** 5
- **Pyramid coverage map:** N/A
- **Expansion-fidelity findings:** N/A

---

### P1 — Must Fix

No P1 findings. The core logical chain — Finding D decomposition to three separable layers, spike evidence for each layer, decision synthesis, rejected alternatives — holds without structural breaks. Each major decision point traces to named evidence.

---

### P2 — Should Fix

**P2-1: The "structural in spirit" positive consequence overclaims relative to the Consequences section's own qualification**

- **Location:** Consequences §Positive, bullet 3: "Lighter than schema-retry, structural in spirit." Also repeated in ADR-035 §Relationship to prior ADRs (AS-9 sentence).
- **Claim:** The decision is described as "structural in spirit" in the Positive consequences, framing the framework-guaranteed directive presence as a structural property.
- **Evidence gap:** The immediately following Negative consequence bullet states explicitly: "Model-compliance-dependent, not hard-enforced. Unlike single-step enforcement (a hard structural truncation, ADR-033), the form depends on documented compliance." The ADR cannot honestly have both simultaneously in separate consequence buckets. The "structural in spirit" formulation does real work in positioning the decision favorably (it places the mechanism in the vicinity of AS-9's validated structural-enforcement pattern), while the Negative qualification then walks it back. A reader scanning consequences sees two contradictory characterizations. The AS-9 relationship paragraph mitigates this somewhat by clarifying "model-compliance-dependent, not a hard structural guarantee," but the "structural in spirit" label in Positive still tilts the framing. The distinction between "guarantees the directive is present" (structural) and "guarantees the output form" (not structural) should be made explicit in the Positive consequence, not only in the Negative.
- **Recommendation:** Revise the Positive bullet to: "Framework-guaranteed directive presence: the loop-driver / boundary composes and injects the directive — the directive's presence is structurally guaranteed; the deliverable's form is not (model-compliance-dependent)." This preserves what is genuinely structural while not recruiting the "in spirit" softener to do epistemological work the Negative section then contradicts.

**P2-2: The ADR-024 carve-out claim about Spike β's mechanism partially misstates what Spike β actually said**

- **Location:** Context §D2a; also §Relationship to prior ADRs — ADR-024 bullet.
- **Claim:** ADR-035 says the D2a finding "corrects the empirical premise of ADR-024 / Spike β: the output-spec drift was reframed there as 'the orchestrator hand-writes `input.data`, overriding `default_task`.' `default_task` does not reach the model at all, so it is not overridden — the dispatch input is the only contract surface that reaches the model."
- **Evidence gap:** ADR-024 itself, reading Spike β's finding, says the drift mechanism is "the orchestrator hand-writes each stage's `input.data`, overriding `default_task` at run-time." ADR-035's framing implies Spike β was wrong about what was being overridden — that `default_task` was treated as active (it was the thing being overridden), when in fact it was never active. But ADR-024 / Spike β's finding is about the orchestrator rewriting `input.data` relative to what the declared contract says the ensemble should do, not a claim that `default_task` was being read at runtime. The Spike β mechanism (orchestrator-authored `input.data` diverges from declared output spec) remains true regardless of whether `default_task` is read at runtime: the declared intent (via `default_task`) and the runtime execution (via `input.data`) diverge. ADR-035's correction is real and adds precision — `default_task` is not merely overridden, it is never read — but framing it as correcting Spike β's "empirical premise" may overstate the contradiction. Spike β's headline finding (composition assumptions live in the orchestrator's reasoning surface, not the typed contract) survives the Spike φ finding intact. The ADR-024 §Negative itself says "`output_schema:` adoption is per-ensemble, not enforced" and "schema validation is advisory at dispatch time" — positions the correction doesn't invalidate.
- **Recommendation:** Soften the framing from "corrects the empirical premise of ADR-024 / Spike β" to "adds precision to ADR-024 / Spike β's finding: the drift's mechanism is not that `default_task` is overridden at runtime but that `default_task` never reaches the model — the dispatch input is the sole runtime contract surface, which is structurally consistent with Spike β's 'orchestrator reasoning surface is the composition substrate' conclusion." This is more accurate and avoids a false impression that ADR-035 is contradicting ADR-024's reasoning rather than refining its mechanistic description.

**P2-3: The ADR-034 claim ("named the prose-framing risk but its store-read mitigation is incomplete") is accurate but the argument for why store-read doesn't resolve it is understated**

- **Location:** Context §D2b; Provenance check last bullet; §Relationship to prior ADRs — ADR-034 bullet.
- **Claim:** "ADR-034 named this exact risk when it rejected synthesizer-as-terminal ('a corruption risk for a tool-call content argument that must be exactly the deliverable') — but its mitigation, reading from the `SessionArtifactStore`, does not resolve it, because the store holds the same prose-framed content."
- **Evidence gap:** This claim is factually supported by Spike φ's separability probe (the real WP-LB-G artifact shows the store holding prose-formatted content, so faithfully reading from the store delivers prose). ADR-034's FC (artifact-bridge fidelity: "marshal exactly what is stored") is explicitly noted in ADR-035 as "satisfied." This is correct. However, the logical step from "ADR-034 named the risk" to "its store-read mitigation is incomplete" does the following: it treats ADR-034 as having claimed the store-read would resolve the prose-framing risk, when in fact ADR-034's store-read decision is a rejection of the synthesizer-as-terminal shape (the synthesizer producing tool-call content live, with prose framing baked in) — not a claim to fix the form of whatever is in the store. ADR-034 deferred the "what form should the stored content take" question implicitly (the Consequences §Negative lists "edit, bash, multi-file, and streaming-token synthesis are unbuilt" but does not explicitly call out the form-of-stored-content gap). So the claim is fair as a gap characterization but slightly overstates ADR-034's responsibility: ADR-034 did not promise store-read would fix the form, it only promised faithful marshalling.
- **Recommendation:** Revise to: "ADR-034 named the prose-framing risk when it rejected synthesizer-as-terminal, and its store-read decision correctly separates marshalling from generation. The gap ADR-034 left open — what form the stored content should take — is what ADR-035 fills. ADR-034's fidelity FC is satisfied: the bridge marshals faithfully; the precondition ADR-034 assumed (that the stored content is client-tool-ready) is the precondition ADR-035 now secures." This is more precise and avoids attributing to ADR-034 a claim it didn't make.

---

### P3 — Consider

**P3-1: The F3-1 citation for the granularity invariant is accurate but slightly soft on scope**

- **Location:** Decision §3 (granularity invariant); §Relationship to prior ADRs — ADR-033 bullet.
- **Claim:** "Multi-file work is the Loop Driver's across-turn decomposition (one `write` per turn), which is callee-native across-turn composition (ADR-033 F3-1)."
- **Observation:** ADR-033 F3-1 is in the rejected alternatives section, not a fitness criterion: "if production traffic surfaces a real need for per-turn multi-capability composition, the pipeline-as-subroutine remains available for that specific step type without changing the callee control structure. At the DECIDE gate the practitioner could not name a concrete reopening trigger, so this is a recorded concession, not a watched contingency." F3-1 is explicitly a "recorded concession" in ADR-033. ADR-035 promotes it to a load-bearing "driver" in the Provenance check (granularity invariant's driver chain: "Spike χ-P6 + ADR-033 F3-1 across-turn composition"). This is not wrong — the F3-1 framing does support the across-turn decomposition characterization — but calling it a "driver" alongside Spike χ-P6 somewhat elevates a concession/residual to a co-primary grounding. The spike evidence (χ-P6 showing multi-file-in-one-dispatch failing) is the real driver; F3-1 is prior-ADR corroboration.
- **Recommendation:** In the Provenance check, characterize ADR-033 F3-1 as "corroborating prior ADR (recorded concession, not a fitness criterion)" rather than "driver, prior ADR" for clarity on evidentiary weight.

**P3-2: The n=4 caveat in the Conditional Acceptance section is honest but the narrative elides one model-type gap**

- **Location:** Conditional Acceptance §; Consequences §Negative bullet 3.
- **Observation:** The n=4 breadth is well-described: function, larger module, bash command, structured prose — all on cheap-tier qwen3:8b. The ADR correctly notes "Escalated-tier behavior (all spike evidence is cheap-tier qwen3:8b)" as a PLAY target. However, the spike evidence is exclusively from code-generator's synthesizer agent, which is one ensemble. The "reliable across n=4 types" formulation implicitly tests the bare-output directive across task-type diversity, but all four types are generated by the same model through code-generator's agent pipeline. Whether other capability ensembles' agents (with different system prompts, different agent counts, different synthesizer positions) comply equally when the boundary directive is injected is not tested — it is assumed by the destination-agnostic-ensemble decision. This isn't a gap that invalidates the decision, but the Conditional Acceptance scope could note "compliance tested only through code-generator's agent pipeline; other ensemble agent stacks are PLAY targets."
- **Recommendation:** Add one clause to the Conditional Acceptance: "The directive reliability (n=4) was tested exclusively via code-generator's synthesizer agent. Directive compliance through other capability ensembles' agent pipelines is a PLAY/first-deployment target alongside trajectory compliance."

**P3-3: Minor terminology: "boundary-composed directive" is used in the title and framing but "marshalling boundary" and "Loop Driver / Client-Tool-Action Terminal" are used interchangeably as the composition locus**

- **Location:** Decision §1, §Relationship to prior ADRs — ADR-033 bullet.
- **Observation:** The ADR uses "the marshalling boundary," "the Loop Driver / Client-Tool-Action Terminal (ADR-033/034)," and "the loop-driver / terminal" to refer to the entity that composes and injects the directive. These are consistent with the referenced ADRs' division of labor (ADR-033 decides which tool, ADR-034 is the terminal mechanism), but a BUILD implementer reading the Decision section may need to cross-reference ADR-033/034 to determine *precisely where* in the code the directive is composed. Decision §1 says "the marshalling boundary composes an output-form directive ... and injects it into the callee `invoke_ensemble` dispatch input" — this is the key operation, but "marshalling boundary" is a conceptual label, not a named code entity. The ADR-033/034 relationship paragraph handles this adequately for DECIDE purposes, but a brief parenthetical in Decision §1 noting which component (e.g., "the Loop Driver, per ADR-033's per-turn tool-choice decision point") composes the directive would help BUILD without adding scope.
- **Recommendation:** In Decision §1, after "the marshalling boundary composes an output-form directive," add "(the Loop Driver's per-turn tool-choice point, per ADR-033 §Decision 4 / ADR-034 §Decision 4)." This pins the conceptual "marshalling boundary" to its ADR home without needing a new paragraph.

---

## Section 2: Framing Audit

The framing audit examines what the φ/χ/WP-LB evidence supported that ADR-035's chosen framing did not foreground.

### Question 1: What alternative framings did the evidence support?

**Alternative framing A: Hard structural enforcement at the client-tool path (AS-9-aligned schema-retry)**

The evidence base would support a framing in which the form contract is enforced mechanically — `output_schema`-as-enforcement with reject-and-retry on the client-tool path — rather than directive-and-trust. The argument under this framing:

- Spike φ F-φ.5 says the cheap model is contract-capable, but adds "n=1 paragraph, one ensemble, one local model — compliance breadth... is unmeasured." This is honest about the risk.
- Spike χ F-χ.2 shows n=4 single-deliverable compliance — but these are all first-try, short tasks, on the same model.
- The cycle's structural thesis (AS-9, single-step enforcer) is precisely that structural enforcement — not model-compliance — is what makes cheap-tier drivers reliable. The single-step enforcer (ADR-033) does not trust the driver to batch correctly; it structurally prevents it. An analogous form-contract enforcement would not trust the ensemble to produce bare output; it would structurally validate and retry.
- What would the reader need to believe for this framing to be right? That n=4 single-dispatch compliance does not extrapolate reliably to long-session multi-turn trajectories, and that the first-try failure mode (which has a ~0% observed rate in the spike) is actually non-trivial at session scale. The AS-9 analogy would be the strongest argument: the cycle adopted structural enforcement for grounding precisely because model trust failed at n=4 axis-1 probes (Spike τ). The analogous concern — form compliance fails at session scale without structural enforcement — is not tested.

ADR-035 does not ignore this framing: it names reject-and-retry as a PLAY escalation and the Conditional Acceptance precisely defers the question. But the ADR foregrounds the directive-and-trust path as the decision and positions schema-retry as escalation, rather than presenting them as peers and choosing the lighter path on cost grounds. The AS-9 analogy could support the opposite default.

**Alternative framing B: Structured multi-file contract (not granularity invariant)**

Spike χ-P6 shows that asking for two files in one dispatch breaks: the model improvises an unparseable convention. ADR-035 interprets this as a granularity signal — the right granularity is one dispatch per deliverable, and multi-file is across-turn composition. Under an alternative framing:

- The χ-P6 break is a *contract specification gap*, not a granularity principle. If the directive included a structured multi-file format ("output as a JSON array: [{filename, content}]"), the model might comply — the same compliance capability that makes bare-single-file work could extend to structured multi-file.
- A structured-multi-file-contract framing would have the boundary directive specify a machine-parseable multi-file format for multi-file tasks, and the bridge would deserialize and emit multiple `write` tool calls.
- What would the reader need to believe? That the χ-P6 break is contract-design-specific (the implicit multi-file convention is broken, not multi-file dispatch inherently), and that a structured multi-file format would achieve reliable compliance as n=4 single-file does.
- ADR-035 interprets χ-P6 correctly given the spike's design — the bare-output contract was not adapted for multi-file. But it closes the multi-file contract door without testing a structured alternative. The granularity-invariant framing is not the only inference χ-P6 supports.

**Alternative framing C: Ensemble-native bare-output defaults (differentiated by delivery context)**

The evidence (D2b: code-generator's system prompts explicitly ask for human-readable output) could support a framing in which the right fix is ensemble-side default tuning: capability ensembles whose primary use case is tool-call delivery get bare-output system prompts by default, while ensembles used for conversational contexts retain prose-framed prompts. The destination-agnostic principle (ADR-025) would be relaxed slightly — ensembles would have a "default delivery context" declared in YAML, separate from coupling to a specific tool.

ADR-035 rejects the static-coupling version of this (the `submit_file`-shaped synthesizer) for ADR-025 reusability reasons. But the rejected alternative is framed as a hard coupling ("a `submit_file` baked in, no destination knowledge in `system_prompt`/`default_task`"), not as a softer default-delivery-context annotation. A middle ground — ensemble declares "default bare output unless overridden by boundary directive" — would preserve some destination-agnosticism while reducing the reliance on per-dispatch directive injection. This framing is not explicitly surfaced in the rejected alternatives.

---

### Question 2: What truths were available but not featured?

**Underrepresented truth A: D2a (inert contract) as a systemic debt with broader consequences than noted**

Spike φ's D2a finding is that `default_task` has zero runtime reads anywhere in the package (grep-confirmed). ADR-035 records this in the Neutral consequences: "The inert-`default_task` finding (D2a) is recorded but not load-bearing for this path. Whether to wire `default_task`/`output_schema` through for the general (non-client-tool) case is separate I/O-contract hygiene, left to a future cycle."

This is scoped correctly — D2a is not load-bearing for the *client-tool path* ADR-035 decides. But the spike establishes something broader: claim-extractor's entire contract (the declared `output_schema` + `default_task`) never reached any model at execution. This affects not just the client-tool path but every invocation of every single-agent ensemble that lacks a `system_prompt` — which is the majority of the current library (the Spike φ table shows argument-mapper, claim-extractor, prose-improver, text-summarizer as single-agent). The consequence is that Cycle 6 BUILD's progressive `output_schema` adoption for those ensembles produced documentary schema that shaped no outputs. The ADR correctly identifies this as "separate I/O-contract hygiene" but the scope of the gap — affecting every non-system-prompted single-agent capability ensemble — is understated in the Neutral section.

Including a note that D2a's scope is the full single-agent ensemble library (not just claim-extractor) and that the Cycle 6 `output_schema` adoption work has been producing documenary-only contract declarations would give a BUILD implementer a clearer sense of the technical debt entering the next cycle.

**Underrepresented truth B: The Spike χ-P1 synthesizer timeout (F-χ.3) has implications for the directive's reliability claim**

F-χ.3 (the synthesizer can fail with a 300s timeout, requiring fallback to the last successful agent) is cited in Decision §5 (D1 extraction fix) as shaping the fallback logic. But it has a second implication the ADR does not foreground: if the synthesizer (the agent that receives the boundary directive) times out, the fallback agent (the critic or coder) was not given the bare-output directive — the directive was addressed to the synthesizer specifically. So the D1 fallback-to-last-successful-agent, combined with the synthesizer timeout, produces a path where the barrier directive is composed for the synthesizer but delivered to the coder or critic instead. Whether the coder or critic would produce bare output with a directive they weren't given is not tested. This is a BUILD design question (where in the agent pipeline the directive should be injected — at the synthesizer specifically, or as a shared context across all agents), but it should be visible at DECIDE time.

**Underrepresented truth C: WP-LB-G's Finding B (delegation gap) is load-bearing context for what ADR-035's contract can even exercise**

Finding D (which ADR-035 fixes) was surfaced at WP-LB-G. Finding B (the seat-filler was never offered `invoke_ensemble`, so the delegation chain never fired in the first run) is also from WP-LB-G. ADR-035's mechanism — the boundary directive is composed when the Loop Driver delegates generation — requires the Loop Driver to actually choose delegation. Finding B was resolved at the mechanism level (WP-LB-G update: offering `invoke_ensemble` + capability list caused the cheap driver to delegate), but the "deeper tension" Finding B names ("the cheap-driver-chooses-to-delegate model can be self-defeating: the driver that can drive will skip delegation") is a direct dependency of ADR-035's correctness in production.

ADR-035 does not mention Finding B or the delegation-decision risk at all. The Conditional Acceptance defers "sustained form-compliance over long multi-turn trajectories" and "granularity invariant holding under a real Loop Driver" to PLAY — but does not identify that the Loop Driver choosing to delegate in the first place is the prior condition that makes ADR-035's contract exercisable. This is not an error (it is ADR-033's territory), but ADR-035 could note in Conditional Acceptance that the form-contract validation presupposes delegation fires, per Finding B's resolution.

---

### Question 3: What would change if the dominant framing were inverted?

The dominant framing of ADR-035 is: **the delivery mechanism (dispatch input) is the right enforcement point, and the model will comply when instructed**.

The inverted framing: **the enforcement point is reliable, but the model's compliance is not — therefore the mechanism is structurally underspecified at the output validation layer**.

Under the inverted framing:

- The n=4 single-dispatch compliance data becomes evidence of a favorable sample rather than grounded evidence of reliable compliance. The cheap-tier qwen3:8b was selected for probing specifically; it is a model the cycle has iterated on. A different cheap-tier model, or the same model on a harder multi-step task with a long context, may behave differently.
- The AS-9 analogy works against ADR-035 rather than for it: AS-9's lesson is that structural enforcement (not model trust) is what produces reliability on cheap-tier models. Single-step enforcement (ADR-033) is structural because a failed batch is *prevented by the framework*, not caught-after-the-fact. The boundary-directive-and-trust approach relies on the model *producing* compliant output; it does not prevent non-compliant output, only detects it in PLAY.
- The deterministic backstop (Decision §4) becomes the real load-bearing mechanism under this framing, not a "defense-in-depth." The ADR explicitly says "deterministic shaping is a defense-in-depth backstop, not the contract" and that Spike χ shows it is fragile as a primary contract. But if model compliance fails at scale, the backstop must escalate to schema-retry (Conditional Acceptance escalation order). The inverted framing would foreground schema-retry as the first-order mechanism rather than the last.
- What the document would need to address if it took the inverted framing seriously: it would need to specify what the form-compliance failure mode looks like in PLAY (a deliverable with a fenced code block slipping through, a synthesizer timeout delivering coder output without the directive), and whether the backstop + escalation order is sufficient to catch and correct failures without user-visible impact. The current Conditional Acceptance defers this to PLAY observation without naming the failure signatures.

---

### Framing Issues

**P2-F1: The AS-9 analogy is invoked in ADR-035's favor but its logic points in the opposite direction — should be acknowledged**

- **Location:** §Relationship to prior ADRs — AS-9 bullet; Consequences §Positive bullet 3.
- **Issue:** AS-9's structural-enforcement thesis was applied in ADR-033 because model trust failed under probing (Spike τ: the unconstrained driver batched and committed to unobserved values). ADR-035 invokes AS-9's vocabulary ("structural in spirit," "structurally framework-owned") to position the boundary-directive approach favorably, while acknowledging in the Negative consequences that the mechanism is model-compliance-dependent. A reader familiar with AS-9's origin — structural enforcement was chosen *because* model trust failed — will notice the ADR is recruiting the AS-9 vocabulary while choosing a model-trust approach. This is not a logical contradiction (the difference between driving-grounding and form-compliance is real), but the tension should be surfaced rather than resolved by the "structural in spirit" framing.
- **Recommendation:** In the AS-9 relationship bullet, add: "The analogy is partial: AS-9's structural enforcement (single-step grounding) was adopted because model trust failed at probing; ADR-035's form contract depends on model compliance (the inverted risk). The difference is that grounding requires *not producing* a premature action, while form compliance requires *producing* output in a specific format — the cheap model's compliance track record on format instructions (Spike φ Run 2, Spike χ F-χ.2) is the empirical basis for taking the lighter path. The PLAY validation target is whether this track record holds at scale." Surface both sides of the analogy rather than letting "structural in spirit" carry the weight.

**P2-F2: The granularity-invariant framing forecloses a structured-multi-file-contract framing without testing it**

- **Location:** Decision §3; Findings §; Spike χ F-χ.5 interpretation.
- **Issue:** Spike χ-P6 produced an unparseable result when asked for two files in one dispatch using an *implicit* format. ADR-035 interprets this as evidence that "multi-file in one dispatch breaks" and states the granularity invariant as a contract principle. This is a valid inference, but it forecloses an alternative: that the break is contract-design-specific, and a structured multi-file format (e.g., JSON array of filename+content pairs, which the same model would likely comply with given the single-file compliance evidence) could work. The granularity invariant is framed as a finding from χ-P6, but χ-P6 tested only implicit multi-file conventions — it did not test a structured multi-file format.
- **Recommendation:** In Decision §3, add a parenthetical: "(χ-P6 tested only the implicit multi-file case; a structured-format multi-file contract was not probed. The granularity invariant is a practical anchor for DECIDE; BUILD/PLAY may revisit if multi-file-per-dispatch becomes a concrete use case.)" This is accurate — χ-P6 cannot close a door it didn't test — and preserves the granularity invariant as the current operative commitment while being honest about its evidentiary basis.

**P3-F1: The Conditional Acceptance doesn't name the delegation-fires precondition as a PLAY target**

- **Location:** Conditional Acceptance §.
- **Issue:** The form-contract mechanism only exercises when the Loop Driver chooses to delegate (invoke `invoke_ensemble`). Finding B from WP-LB-G showed this delegation was not firing in the initial run. The resolution (offering `invoke_ensemble` + capability list) is documented in WP-LB-G's update, but ADR-035's Conditional Acceptance section does not name "delegation reliably choosing to invoke `invoke_ensemble` (rather than acting directly)" as a PLAY validation target alongside form compliance and trajectory compliance.
- **Recommendation:** Add one clause to the Conditional Acceptance: "The form-contract validation targets above presuppose delegation fires (the Loop Driver chooses `invoke_ensemble` rather than acting directly). Finding B resolution held for a single real-client run; sustained delegation under varied task types is an ADR-033 PLAY target that gates ADR-035's form-contract PLAY validation."

---

## Convergence-Saturation Signal (ADR-094)

**Convergence-Saturation Signal:** NOT TRIGGERED

- Round number: R1 (first audit on ADR-035; new ADR = fresh baseline per form-change baseline-reset rule)
- P1 count this round: 0 (Section 1: 0; Section 2: 0)
- P2 count this round (new, non-carry-over): 5 (P2-1 through P2-3 in Section 1; P2-F1 and P2-F2 in Section 2)
- New framings or claim-scope expansions: Three framing alternatives named in Section 2 Q1 (hard-enforcement AS-9-aligned schema-retry; structured-multi-file-contract; ensemble-native bare-output defaults); AS-9 analogy partial-inversion surfaced in Q3 and P2-F1.
- Recommendation: CONTINUE to next round. P2 count exceeds threshold (5 > 1), and the round surfaced new framings in Section 2. Both conditions prevent the signal from triggering.

*Single-purpose re-audits (dispatched per the re-audit-after-revision rule) omit this section. Form-change events reset the round-count baseline — the first audit on a new form is its R1.*

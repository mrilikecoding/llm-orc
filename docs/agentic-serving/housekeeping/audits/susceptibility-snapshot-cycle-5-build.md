# Susceptibility Snapshot

**Phase evaluated:** BUILD (Cycle 5)
**Artifact produced:** 7 per-file model profiles (`agentic-*.yaml`), 6 capability ensembles + 2 moved system ensembles in `.llm-orc/ensembles/agentic-serving/`, `web_searcher.py` script-agent, `agentic-serving/README.md`, rewritten `agentic_serving:` config section, `skill-framework-capability-registry.md`, downstream sweep (ORIENTATION.md, system-design.md, domain-model.md, scenarios.md, interaction-specs.md)
**Date:** 2026-05-12

---

## Prior Snapshot Trajectory

| Gate | Snapshot verdict | Key signal |
|------|-----------------|------------|
| Cycle 4 Research | Grounding Reframe triggered | Three grounding actions; autonomous-routing gap named |
| Cycle 4 Discover | Grounding Reframe triggered | Asymmetric readiness mapping; research-voice transplants |
| Cycle 4 Model | Clean with feed-forwards | No reframe; vocabulary relocation discipline applied |
| Cycle 4 Decide | Grounding Reframe recommended (1 finding) | ADR-015 autonomous-routing evidence gap not carried into artifact |
| Cycle 4 Architect | No reframe; 7 advisory carry-forwards | Inherited framing from DECIDE |
| Cycle 4 Build | Grounding Reframe (one targeted) + 2 advisory | Pre-loaded conditional-acceptance disposition; resolved in-cycle |
| Cycle 4 Play | No Grounding Reframe; 4 advisory carry-forwards | Voice blurring at synthesis boundary; n=1 findings encoded as settled |
| Cycle 5 Discover | No Grounding Reframe; 2 advisory carry-forwards | Settlement-before-examination sequencing gap; four inversion questions not recorded at gate |
| Cycle 5 Decide | No Grounding Reframe; 2 advisory carry-forwards | Inherited scope-claim breadth (n=1 framework verified); no-dispatch-fallback reasoning at minimum threshold |
| **Cycle 5 Build (this snapshot)** | Evaluated below | |

The DECIDE-gate snapshot's two advisory carry-forwards were:

- **Advisory 1 (scope-claim breadth):** The skill-framework-agnostic commitment's scope claim covers frameworks beyond RDD but rests on one structurally verified framework (RDD) + structural logic. BUILD's RDD-decomposition exercise would either confirm or fail to extend the claim. The advisory noted that if BUILD produces no non-RDD evidence, the "any skill framework" framing in ADR-019 §Consequences should be scoped accordingly at cycle close.
- **Advisory 2 (no-dispatch-fallback reasoning quality):** ADR-019 §Neutral resolved the discover-gate's examination commitment at minimum threshold — assertion supported by PLAY notes. The advisory noted that BUILD-phase calibration evidence on the no-dispatch path would be the first empirical test; orchestrator-natural-language-response observations should be recorded as candidate evidence.

Both advisories are assessed below.

---

## Observed Signals

| Signal | Strength | Trajectory | Notes |
|--------|----------|------------|-------|
| Assertion density | Ambiguous | Stable | BUILD phase's empirical grounding (validation against the actual loader, `llm-orc list-ensembles` confirmation, schema error discovery) grounds assertions in observable outcomes. Assertion density in the BUILD corpus is lower than DECIDE's; most concrete claims have an observable artifact as their referent. Residual concentration: the scope-claim framing in ORIENTATION.md ("skill-framework-agnostic dispatch scales to any skill standard") carries without per-cycle qualification — assessed in detail below. |
| Solution-space narrowing | Clear (inherited) | Stable — no new narrowing in BUILD | The narrowing entered BUILD from DECIDE: the architectural commitment is skill-framework-agnostic; the ensemble library is operation-named; the tier defaults follow the Topaz taxonomy. BUILD authoring honored these commitments without introducing new narrowing. The profile-file format refinement (7 individual files vs. single `model_profiles:` dict) was the one scope-bearing choice that was internally resolved rather than presented as a user-reviewable decision — assessed in §1 below. |
| Framing adoption | Ambiguous | Declining relative to prior cycles | BUILD artifacts adopt the DECIDE framings without re-examining them — appropriate for a BUILD phase, where the decisions have been made and authoring executes them. The README, scenarios, and ORIENTATION.md all use "skill-framework-agnostic," "operation-named," and "three-layer architecture" as settled vocabulary. This is settled-by-use confirmation that the DECIDE snapshot predicted as the correct test; whether the vocabulary has entered operator voice is assessable from the authoring output. |
| Confidence markers | Ambiguous | Stable | "Settled" language in the BUILD artifacts is appropriately confined: the README explains the operation-named principle without claiming it is the only possible shape; the ORIENTATION.md BUILD-close entry notes the n=1 scope gap explicitly ("n=1 skill-framework scope persists at cycle close; RDD is the only framework structurally verified"). ORIENTATION.md's consequences section ("skill-framework-agnostic dispatch scales to any skill standard") is the one unqualified confidence marker in the public-facing record — addressed in §2 below. |
| Alternative engagement | Absent | Declining (expected for BUILD auto mode) | Auto mode's declared tradeoff (per ADR-091 §3 honest scope-of-claim and the cycle-status BUILD mode declaration) is precisely that design-alternative examination and scoping-judgment surfacing are gated-mode capabilities. BUILD auto mode did not surface alternatives for: (a) profile file format choice; (b) `agentic-coding-helper` rename disposition; (c) `web-searcher.yaml` initial schema shape (before the validation-error correction). This is the expected mode behavior, not a susceptibility signal per se — but it means these three discoveries landed as implementation choices rather than practitioner-reviewable decisions. |
| Embedded conclusions at artifact-production moments | Clear (three instances) | Present throughout BUILD | The dispatch prompt identified three specific embedded-conclusion events: the profile-file format discovery, the `agentic-coding-helper` rename disposition, and the script-agent schema constraint discovery. Each is assessed in detail below. The pattern is that BUILD's auto mode resolved all three without surfacing them as open questions; the ORIENTATION.md BUILD-close entry records the profile-file refinement explicitly, which partially mitigates the first. The other two have weaker recording. |

---

## Element-Specific Assessments

### 1. Profile-file format: 7 individual files vs. single `model_profiles:` dict

**Discovery:** The proposal recommended a single `agentic-serving-profiles.yaml` file holding all 7 profiles in a `model_profiles:` dict. The existing loader (`ConfigurationManager`, line 514-528) only loads files with a top-level `name:` field — the `model_profiles:` dict format is silently skipped. The agent discovered this at BUILD-time by hitting the constraint and refactored to 7 individual files.

**Recording:** The ORIENTATION.md BUILD-close entry records this: "7 per-file Model Profiles in `.llm-orc/profiles/agentic-*.yaml` (loader's expected one-profile-per-file format — a build-time refinement of the proposal's single-aggregated-file shape, recorded in the build session log)." The discovery is on the record as a build-time refinement.

**What was not examined:** The alternative path — keeping the inline `model_profiles:` section in `.llm-orc/config.yaml` rather than separate profile files — was not surfaced as a user-reviewable decision. The proposal motivated the separate-file approach on ergonomic grounds (single-file model swaps); the inline config-section alternative would satisfy the loader contract with less filesystem fragmentation. This is a scoping-judgment case of the type ADR-091 names as the auto-mode tradeoff: the agent made a reasonable call (7 individual files matching the existing loader contract) that is consistent with the proposal's intention, without presenting the alternative. The inline-config alternative was not examined and is not on the record as considered.

**Susceptibility weight:** Low-to-moderate. The discovery is recorded; the implementation is consistent with the proposal's intent and the loader's actual contract. The non-surfacing of the inline-config alternative is an auto-mode scoping-judgment gap, not a framing-adoption failure. The 7-individual-file shape is now verifiably working (ORIENTATION.md confirms `ConfigurationManager.get_model_profiles()` resolves all 7 profiles).

---

### 2. `agentic-coding-helper.yaml` rename: silent consumption vs. preservation-with-routing

**Discovery:** `agentic-coding-helper.yaml` was an untracked Cycle 4 PLAY artifact. ADR-019 §"Working defaults" called it "promoted to `code-generator`." The scenarios.md Preservation scenario "Cycle 4 PLAY tagging work continues to dispatch correctly" was updated in DECIDE to read: "`agentic-coding-helper` is **promoted** (per ADR-019 §"Working defaults") to `code-generator`... retaining the same `topaz_skill: code_generation` tag and the same three-agent flow shape."

**What happened:** `agentic-coding-helper.yaml` no longer exists on disk. The git status shows it was untracked and is now simply absent — consumed into the promotion. The `code-generator.yaml` carries the three-agent flow shape.

**What was not examined:** The scenarios.md preservation scenario, as originally written during the DISCOVER phase, read "existing tagged ensembles continue to dispatch with their existing names." The original text of that preservation commitment was a load-bearing scenario: operators who had `invoke_ensemble("agentic-coding-helper", ...)` in their workflows would break on rename without a forwarding alias. The agent's interpretation — "the preservation scenario was written with imperfect foreknowledge; the scenario was updated in DECIDE to reflect the promotion" — resolved the conflict by updating the scenario. The alternative interpretation is that the scenario carried a load-bearing commitment to backward compatibility that the agent overrode by rewriting the scenario rather than by adding a forwarding alias.

**Assessment:** The scenario update is the more architecturally honest path. `agentic-coding-helper` was an untracked PLAY artifact — not a committed ensemble operators could depend on. The "existing names" language in the original scenario was generic preservation boilerplate that did not anticipate the PLAY artifact's promotion. Rewriting the scenario is correct given what the artifact actually was (an uncommitted PLAY experiment, not a stable named dispatch target).

However: the resolution sits on the agent's interpretation of the scenario's intent, and that interpretation is not on the record as a user-reviewable decision. If an operator had incorporated `invoke_ensemble("agentic-coding-helper", ...)` into a workflow during the PLAY session, the rename breaks their workflow with no forwarding mechanism. The BUILD session log entry in ORIENTATION.md notes the rename ("`code-generator` — promoted from Cycle 4 PLAY's `agentic-coding-helper`") but does not record that the naming-commitment tension was examined. The scenario's "existing names" language was updated without a note about what the prior language meant or why it was changed.

**Susceptibility weight:** Moderate. This is the clearest case of an embedded conclusion at an artifact-production moment (updating a preservation scenario to remove a commitment that conflicted with the implementation) without surfacing the conflict as a user-reviewable decision. The resolution is defensible — the ensemble was never committed, and "existing named ensembles" is a more precise reading of the scenario's intent than "any ensemble named during PLAY." But the fact that the scenario was rewritten rather than the conflict being surfaced is the signal. If the practitioner had independently valued backward compatibility for PLAY session workflows, they would not have seen this decision in the BUILD session's output.

---

### 3. Script-agent schema discovery: recorded as refinement, but schema constraint is not in the DECIDE corpus

**Discovery:** The `web-searcher.yaml` draft (following the proposal's `web-searcher.yaml` shape suggestion) initially included `system_prompt:` and possibly `type:` fields on the script-agent. The `llm-orc list-ensembles` validation raised a schema error: `ScriptAgentConfig` only accepts `name`, `script`, `parameters`, plus base fields — no `system_prompt:`, no `type:`. The fix was mechanical (remove the offending fields from the YAML).

**Recording:** The shipped `web-searcher.yaml` is clean. The ORIENTATION.md BUILD-close entry does not record this discovery explicitly. The DECIDE corpus (ADR-020) describes the script-agent shape without noting that `system_prompt:` is architecturally unsupported — ADR-020's `§Ensemble shape` describes script-agent behavior at the capability level ("receives the search query as agent input... returns a structured result") without specifying the YAML-level schema constraints. The proposal's `web-searcher.yaml` shape suggestion was wrong about `system_prompt:`, but that error is not documented anywhere in the DECIDE or BUILD corpus.

**What was not recorded:** A future operator authoring a script-agent capability ensemble from the proposal or from ADR-020's description alone would author `system_prompt:` and hit the same schema error. The BUILD-time discovery that `ScriptAgentConfig` has no `system_prompt:` field is a schema constraint the README should document for script-agent ensemble authors. The README's "How to add a new capability ensemble" section gives the standard LLM-agent YAML shape (with `system_prompt:`), and separately notes "If the ensemble needs real tool execution, use a script-agent shape... see `web-searcher.yaml` as the reference instance." An operator reading the README would look at `web-searcher.yaml`, see no `system_prompt:`, and infer the schema correctly from the reference — but the underlying constraint (the schema difference between LLM-agent and script-agent) is implicit in the example rather than explicitly stated.

**Assessment of whether the finding feeds forward:** The ORIENTATION.md does not record the schema discovery. The README uses `web-searcher.yaml` as the reference instance, which is a passive documentation of the schema constraint. The capability-ensemble authoring guide implication — that a future authoring guide should explicitly contrast LLM-agent and script-agent YAML schemas — is not recorded as a future-cycle feed-forward. The BUILD-phase advisories in the DECIDE snapshot asked for "orchestrator-natural-language-response errors should be recorded as candidate evidence"; the analogous ask for the schema discovery would be "proposal's `web-searcher.yaml` shape suggestion was wrong about `system_prompt:` — a future capability-ensemble authoring guide should reflect the actual schema." This is not on the record.

**Susceptibility weight:** Low-to-moderate. The shipped artifact is correct; the README provides the right reference instance. The gap is that the schema discovery did not crystallize as a named finding — it slid into the implementation. Operators composing a new script-agent type from first principles (not from the reference instance) will rediscover the constraint. This is within the scope of what auto mode explicitly does not catch.

---

### 4. WP-I5 (integration testing): deferred without explicit examination

**Discovery:** WP-I5 (optional integration scenarios at Step 5.5) is listed as "optional" in the cycle-status. The Cycle Acceptance Criteria Table has 3 Layer-match `no` entries: multi-skill-framework deployment evidence, fresh-clone live exercise, and the five-ensemble composition integration test. BUILD did not write integration tests for these.

**Recording:** ORIENTATION.md BUILD-close entry: "The cycle acceptance criteria table's Layer-match `no` entries identify integration-test territory (multi-skill-framework deployment evidence; fresh-clone live exercise) deferred per ADR-019 §Negative — n=1 skill-framework scope persists at cycle close." The deferral is on the record. The stated justification is ADR-019 §Negative (operator-driven authoring remains the operator's responsibility for non-minimum-viable capabilities) — which is a reasonable cross-reference, though the §Negative section's actual text covers operator authoring responsibility, not integration test scope.

**Assessment:** The WP-I5 deferral is consistent with auto mode's mechanical-character scope — integration tests require a running deployment and are not mechanical YAML/config authoring. The ORIENTATION.md records the gap explicitly; the Cycle Acceptance Criteria Table documents the open verification surface. This is a settled-and-recorded commitment, not a decision that slid through unexamined. The ADR-019 §Negative cross-reference is slightly imprecise (the §Negative section covers operator authoring, not integration testing scope), but the factual claim — integration tests are deferred; n=1 scope persists — is accurate.

**Susceptibility weight:** Low. The deferral is documented; the mechanism (the acceptance criteria table) is designed to make it visible.

---

### 5. Advisory 1 carry-forward: scope-claim breadth at BUILD close

**Status:** The DECIDE snapshot's Advisory 1 noted the skill-framework-agnostic commitment rests on one structurally verified framework (RDD) and asked whether BUILD would produce any non-RDD evidence.

BUILD produced no non-RDD evidence. The `skill-framework-capability-registry.md` includes entries for Anthropic Skills, OpenAI Assistants, and MCP-based skill frameworks — all marked as "undetermined" or "placeholder." The code-review-as-methodology entry is illustrative, not deployed. The "any skill framework" framing in ADR-019 §Consequences §Positive persists without qualification in that artifact.

ORIENTATION.md explicitly records: "n=1 skill-framework scope persists at cycle close; RDD is the only framework structurally verified." This is the correct epistemic qualification. The gap is that ADR-019 §Consequences §Positive ("Skill-framework-agnostic dispatch scales to any skill standard") does not carry the n=1 qualification inline — it reads as a delivered consequence, not a labeled working assumption. Future readers of ADR-019 who do not also read ORIENTATION.md may treat the consequence as established.

**Assessment:** The DECIDE snapshot's Advisory 1 asked for either non-RDD BUILD evidence or explicit qualification of the scope claim at cycle close. ORIENTATION.md provides the qualification; ADR-019 does not. This is a partial resolution — the qualification is on the record, but not in the load-bearing artifact (ADR-019 is the architectural decision record; ORIENTATION.md is the orientation document).

**Susceptibility weight:** Low-to-moderate. The risk is specifically that future cycles read ADR-019 §Consequences §Positive as established evidence rather than as a labeled working assumption. ORIENTATION.md's qualification is present but is not visible to a reader navigating directly from the ADR. This is an appropriate carry-forward for the next cycle's RESEARCH or DISCOVER entry.

---

### 6. Advisory 2 carry-forward: no-dispatch-fallback empirical test at BUILD

**Status:** The DECIDE snapshot's Advisory 2 noted that BUILD calibration evidence on the no-dispatch path would be the first empirical test of the "intended scope" resolution. BUILD (auto mode, YAML/config authoring only) produced no runtime sessions and thus no new calibration evidence on the no-dispatch path. This is expected given the BUILD mode declaration; the advisory was noting what to look for, not mandating a runtime test.

The ORIENTATION.md BUILD-close entry does not mention the no-dispatch-fallback empirical question. The BUILD artifacts (README, scenarios, registry) do not record new evidence for or against the "intended scope" resolution. The resolution remains at the minimum-threshold level the DECIDE snapshot identified.

**Assessment:** The advisory's empirical test surface was not exercised (and could not be, under auto-mode mechanical authoring). This is within expectation. The carry-forward persists: the first live-deployment session after Cycle 5 BUILD close is the natural observation point for no-dispatch-path behavior under the new 5-ensemble library. The registry's note that "RDD's play and architect phases do not have direct capability-ensemble equivalents — they are decomposition-and-dispatch phases that operate on existing artifacts" implicitly identifies the orchestrator's natural-language-response path as the expected handling for those phases, which is consistent with the "intended scope" resolution without constituting new evidence for it.

**Susceptibility weight:** Low. The empirical test was not in BUILD scope; the absence of new evidence is expected. The carry-forward is well-defined.

---

## Interpretation

### Pattern assessment

The dominant pattern is **auto-mode scoping-judgment execution with three unrecorded build-time discoveries.** This is consistent with what BUILD auto mode declares it does (mechanical authoring from settled decisions) and what it does not do (design-alternative examination and scoping-judgment surfacing). The signals that would concern a sycophancy evaluator in this BUILD phase are the ones the tests did not catch — and specifically, the ones that slide through the phase boundary without becoming recorded findings.

The three discoveries are qualitatively different from each other:

- The **profile-file format discovery** is the least concerning: it is recorded in ORIENTATION.md, the implementation is consistent with the proposal's intent, and the verification (`llm-orc list-ensembles` confirming all 8 ensembles; `ConfigurationManager.get_model_profiles()` resolving all 7 profiles) is empirical. The agent made a scoping judgment consistent with the loader's actual behavior.

- The **`agentic-coding-helper` rename disposition** is the most concerning: a preservation scenario was rewritten to remove a naming commitment that conflicted with the implementation, without surfacing the conflict as a user-reviewable decision. The resolution is defensible (the ensemble was never committed; the "existing names" language was boilerplate), but the mechanism — updating a scenario to match an implementation rather than surfacing the conflict — is the pattern that susceptibility evaluation specifically attends to. A scenario that existed in the DECIDE corpus to protect a commitment was updated in BUILD to remove that protection, and the update is not flagged in the BUILD session record.

- The **script-agent schema discovery** is the most operationally consequential gap: the proposal's `web-searcher.yaml` shape suggestion was wrong about `system_prompt:`, the fix was mechanical, but the discovery was not crystallized as a named finding. Future script-agent authors will encounter the same schema constraint without finding it documented. The BUILD-time correction is invisibly embedded in the shipped artifact rather than surfaced as a feed-forward.

### Earned confidence vs. sycophantic reinforcement

The BUILD phase's empirical grounding (validation errors caught by the actual tool, `llm-orc list-ensembles` confirmation, `ConfigurationManager` resolution verification) provides the structural resistance to sycophancy that the sycophancy gradient predicts for BUILD. The signed-off artifacts work. The susceptibility risk in this BUILD phase is not in what the tests exercise but in what they don't: the three discoveries above landed as implementation choices rather than practitioner-facing decisions, and the most significant one (the preservation-scenario rewrite) was not flagged at all in the session record.

The cross-phase trajectory shows a consistent pattern of improving attribution discipline across DISCOVER and DECIDE, with no Grounding Reframes warranted. This BUILD phase follows the same trajectory — no convergent pattern of sycophantic reinforcement — but introduces a new category of signal appropriate to auto-mode BUILD: **silent resolution of artifact-level conflicts** (scenario updated to match implementation; schema discovery embedded in artifact; loader-constraint-driven refactoring unexamined against alternatives).

### Prior advisory carry-forward status

| Advisory | Status at BUILD close |
|----------|----------------------|
| DECIDE Advisory 1 (scope-claim breadth) | Partially resolved — ORIENTATION.md qualifies to n=1; ADR-019 §Consequences §Positive does not carry the qualification inline. Persists as a carry-forward. |
| DECIDE Advisory 2 (no-dispatch-fallback empirical test) | Not testable in auto-mode BUILD; carry-forward to first live-deployment session. No new evidence produced. |

---

## Recommendation

**No Grounding Reframe warranted.** The signals do not converge on a pattern where the practitioner would be building on a hidden unexamined assumption that poses operational risk to the next phase. The BUILD deliverables work; the empirical grounding (validation error → fix; `list-ensembles` confirmation; profile resolution verification) provides BUILD-appropriate resistance to framing adoption. The cycle-status and ORIENTATION.md both carry explicit n=1 scope qualifications.

**Three advisory carry-forwards for next-cycle entry:**

---

### Advisory 1 — The `agentic-coding-helper` scenario rewrite should be flagged at next-cycle DISCOVER

The preservation scenario "Cycle 4 PLAY tagging work continues to dispatch correctly" was updated in BUILD to reflect the `agentic-coding-helper` → `code-generator` promotion. The original scenario language ("existing tagged ensembles continue to dispatch with their existing names") was removed; the updated scenario explicitly describes the promotion as the preservation mechanism. This is recorded in scenarios.md but not in the BUILD session log or ORIENTATION.md.

The carry-forward is not that the promotion was wrong — it was the correct call. The carry-forward is that the next cycle's DISCOVER phase should check: does the scenarios.md corpus contain other preservation scenarios that encode naming-stability commitments the current library state has outrun? The `agentic-coding-helper` case was clean (untracked PLAY artifact, no operator dependency on the name). Future ensemble promotions or renames may not be.

Recording the decision pattern — "BUILD updated a preservation scenario to match an implementation without surfacing the conflict" — ensures that future auto-mode BUILD phases know to explicitly flag scenario amendments for practitioner review, rather than treating them as part of mechanical artifact maintenance.

---

### Advisory 2 — Script-agent YAML schema constraint should enter the authoring guide

The BUILD-time discovery that `ScriptAgentConfig` has no `system_prompt:` field (and that a `type:` field on script-agents is invalid) is not documented anywhere in the DECIDE or BUILD corpus except implicitly in the shipped `web-searcher.yaml` reference instance. A future operator authoring a script-agent from the proposal's description or from ADR-020's capability-level description would encounter the validation error.

The next cycle that touches the operator-facing documentation (a future DISCOVER or BUILD pass extending the library) should add an explicit note to the README's "How to add a new capability ensemble" section distinguishing the LLM-agent YAML schema (which includes `system_prompt:`, `model_profile:`) from the script-agent schema (which replaces those with `script:` and optionally `parameters:`). The `web-searcher.yaml` reference instance is good; the underlying schema constraint should be named, not inferred.

---

### Advisory 3 — ADR-019 §Consequences §Positive should carry the n=1 scope qualification at next revision

The ORIENTATION.md BUILD-close entry explicitly qualifies the skill-framework-agnostic scope claim to n=1 (RDD only). ADR-019 §Consequences §Positive does not: "Skill-framework-agnostic dispatch scales to any skill standard" reads as an established consequence. The ADR's provenance check and vocabulary note earlier in the document do carry qualification language, but §Consequences §Positive does not.

At the next ADR-019 amendment (which would occur when a non-RDD skill framework is structurally verified against the library, or when the claim's evidence base is revisited in a future RESEARCH cycle), the §Consequences §Positive bullet should be qualified: "(claim grounded in RDD structural verification at Cycle 5 BUILD close; verification against additional skill frameworks is Cycle 6+ empirical territory)." Until then, the qualification is present in ORIENTATION.md but absent from the ADR's most-read section.

This is the same carry-forward the DECIDE snapshot raised; it now has BUILD-close status: BUILD did not extend the evidence base, so the advisory's premise is confirmed. The next cycle should act on it or explicitly defer with a rationale.

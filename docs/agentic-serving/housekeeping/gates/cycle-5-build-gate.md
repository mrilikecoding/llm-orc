# Gate Reflection: Cycle 5 BUILD → (cycle close)

**Date:** 2026-05-12
**Phase boundary:** build → (cycle close; play / synthesize deferred as available)
**Cycle:** Cycle 5 — Agentic-serving library structure (capability ensembles + multi-methodology-consumer surface)

## Auto-mode disposition

Cycle 5 BUILD ran in **auto mode** (ADR-091) — declared at BUILD entry by the practitioner after the WPs were sized and the work's mechanical character was confirmed. Per ADR-091 §3, auto mode suppresses per-scenario-group AID gates and stewardship-surfacing; review concentrates at start-and-end. The gate-reflection-note obligation persists at BUILD close as the structural phase-end record.

The auto-mode failure modes ADR-091 names (design-alternative examination not surfaced; scoping-judgment surfacing absent) operated as ADR-091 predicts during BUILD — three signals are recorded in the build-phase susceptibility snapshot:

1. **Preservation scenario rewrite** — the scenarios.md preservation scenario for `agentic-coding-helper` was amended in BUILD to reflect the rename-to-`code-generator` promotion, rather than the conflict being surfaced for practitioner review at WP-C5 entry.
2. **Script-agent schema discovery not crystallized as a finding** — the BUILD-time discovery that ScriptAgentConfig forbids `type:` and `system_prompt:` corrected a wrong assumption in the proposal's `web-searcher.yaml` shape suggestion; the fix was mechanical, the discovery surfaced no practitioner-facing finding. Mitigation applied in-cycle: the README authoring guide now distinguishes LLM-agent vs. script-agent schemas explicitly.
3. **ADR-019 §Consequences §Positive scope qualifier missing** — the DECIDE snapshot's Advisory 1 raised this; BUILD did not extend the evidence base; the advisory's premise is confirmed at cycle close. Mitigation applied in-cycle: ADR-019 §Consequences §Positive now carries the n=1 scope-of-claim qualifier.

## Belief-mapping question composed for this gate

(Auto mode — no per-scenario-group AID gate was run. The phase-close gate question composed for the BUILD-close conversation:)

> "Cycle 5 closes with 7 advisory carry-forwards collapsing into three live ones — the preservation-scenario amendment pattern, the schema-distinction documentation in the README authoring guide (applied in-cycle), and the n=1 scope-of-claim qualifier on ADR-019 §Positive (applied in-cycle). What would have to be true for any of the three to be wrong as the load-bearing close-out remediation — for example, that the preservation-scenario amendment should instead have been a Design Amendment to ADR-019's promotion framing, or that the n=1 qualifier is too narrow / too broad for the operational claim ADR-019 makes?"

## User's response

(Auto mode — gate question was composed and the in-cycle remediations were applied directly. No practitioner conversation was held at BUILD close; the practitioner's signal to enter auto mode at BUILD entry was the implicit warrant for the close-out path. The gate question stands as a recorded available pre-mortem the practitioner can revisit if BUILD-phase work produced unexpected friction.)

## Pedagogical move selected

Auto-mode close-out — apply the in-cycle remediations the build-phase snapshot named as actionable, record advisories that are not in-cycle applicable as feed-forwards. The two structural remediations (README authoring-guide schema distinction; ADR-019 §Positive scope qualifier) were applied directly; the preservation-scenario-rewrite pattern is recorded as the cycle's BUILD-close auto-mode-specific feed-forward.

## Commitment gating outputs

**Settled premises (the practitioner is building on these going into cycle close):**

- ADR-019's skill-framework-agnostic library reshape is operational on disk; concrete BUILD-time authoring did not surface evidence requiring re-opening the ADR.
- ADR-020's `tool_use` script-agent shape with operator-configurable backend works end-to-end (smoke-test verified for 3 error paths).
- ADR-021's per-capability dispatch contract is the consumer-side commitment skill frameworks compose against; no orchestrator-side state crosses sub-tasks; ADR-006 `compose_ensemble` scope preserved.
- The seven `agentic-*` Model Profile files in `.llm-orc/profiles/` and the eight ensembles in `.llm-orc/ensembles/agentic-serving/` are loaded by `ConfigurationManager.get_model_profiles()` and `llm-orc list-ensembles` respectively, verified at BUILD close.
- The Cycle 4 PLAY note 1 first-encounter framing — *"the agentic-serving config is to me part of the build"* — is closed at the artifact-on-disk layer; live-deployment first-encounter exercise (fresh-clone install/startup) deferred to operator-driven evidence.

**Open questions (the practitioner is holding these open going into cycle close):**

- Whether the multi-skill-framework cycle acceptance criterion (Layer-match `no`: at least two distinct skill orchestration users dispatching against the same library) requires a follow-up cycle's empirical work or whether operator evidence accumulating in deployment is sufficient.
- Whether the seam-case inversion (ADR-021 §"Seam-case inversion") — Topaz-skill routing producing routing-quality parity across skill-framework contexts — fires under deployment evidence over time; resolution path per-skill-framework tier defaults extending ADR-015 named as the lightest available extension.
- Whether the `mathematical_reasoning` slot's MissingSkillMetadataError-recovery path holds reliability across orchestrator profiles beyond MiniMax M2.5-free; future cycles author a `math-solver` if/when an orchestrator profile or task class requires it.
- Whether the `instruction_following` slot's deployment-specific status (served by `development/code-review`) becomes ambiguity territory as the slot's "catch-all" character is tested by additional skill frameworks.

**Specific commitments carried forward to next cycle (or graduation):**

1. **Auto-mode-specific feed-forward (snapshot finding 1)** — the preservation-scenario-amendment pattern (BUILD updating scenarios.md to match implementation rather than surfacing the conflict) is a recurrence risk under auto mode. Future auto-mode BUILDs should flag scenario-rewrite events in the session record so the practitioner can review them at BUILD close.
2. **Cycle acceptance criteria Layer-match `no` entries** — three entries remain unsatisfied at the named layer (multi-skill-framework deployment evidence; fresh-clone first-encounter live exercise; integration scenario through five capability dispatches). Defer to operator-driven empirical work; capture into roadmap if production deployment evidence accumulates.
3. **Cycle 5 graduates as a candidate** — the agentic-serving corpus's identity-forming work (architectural surface + library shape + operator-facing deployment) appears stable at Cycle 5 close. A future cycle may convert this to a `/rdd-graduate` event folding RDD knowledge into native llm-orc docs and archiving the scoped corpus. Not in Cycle 5 scope; flagged as available next move.
4. **Cycle 4 PLAY susceptibility advisory carry-forwards** — the n=1 evidence-basis qualifier on the proposal's framings (per Cycle 4 PLAY susceptibility snapshot advisory #2) is confirmed at Cycle 5 close: the framings survived concrete BUILD-phase authoring as operator/structural vocabulary. The proposal's evidence-basis qualifier can now be relaxed in future references; the framings are settled-by-use.

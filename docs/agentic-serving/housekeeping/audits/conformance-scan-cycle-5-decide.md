# Conformance Scan Report — Cycle 5 DECIDE-Phase ADRs

**Scanned against:**
- ADR-019 (`adr-019-skill-framework-agnostic-capability-library.md`)
- ADR-020 (`adr-020-tool-use-ensemble-shape.md`)
- ADR-021 (`adr-021-skill-orchestration-via-per-capability-dispatch.md`)

**Codebase:** `/Users/nathangreen/Development/eddi-lab/llm-orc/`

**Date:** 2026-05-12

---

## Summary

- **ADRs checked:** 3
- **Conformance surfaces examined:** 6 (per scan brief scope: config.yaml agentic_serving: section; ensembles/ topaz_skill coverage; profiles/ directory; agentic-serving/ subdirectory; tagged-ensemble shape; no-code-level violations expected)
- **Actual violations (existing config/code contradicts the ADR):** 0
- **BUILD-scope gaps (not yet authored; ADRs explicitly defer these to BUILD):** 10
- **Pre-existing items compatible with ADRs (no action required):** 3

---

## Conformance Debt Table

Items are classified as either **VIOLATION** (existing artifact contradicts the ADR and must be changed) or **BUILD-SCOPE** (not yet authored; the ADR explicitly defers to Cycle 5 BUILD). A third category, **COMPATIBLE / NOTE**, covers existing items that neither violate nor need to change but warrant documentation.

| # | Item | Location | ADR | Type | Resolution | Cycle 5 BUILD Scope? |
|---|------|----------|-----|------|------------|----------------------|
| 1 | `.llm-orc/profiles/agentic-serving-profiles.yaml` does not exist | `.llm-orc/profiles/` (absent) | ADR-019 §Working defaults | BUILD-SCOPE | Author the file; migrate all agentic-serving Model Profiles (orchestrator-default, tier-cheap-general, tier-cheap-summary, tier-escalated-general, tier-escalated-reasoning, summarizer) out of config.yaml into this file. | Yes — WP enumerated at BUILD entry |
| 2 | `.llm-orc/ensembles/agentic-serving/` subdirectory does not exist | `.llm-orc/ensembles/` (absent) | ADR-019 §Working defaults | BUILD-SCOPE | Create the subdirectory; move `agentic-calibration-checker.yaml` and `agentic-result-summarizer.yaml` into it; update any references in config.yaml. | Yes |
| 3 | `code-generator` capability ensemble not authored | `.llm-orc/ensembles/agentic-serving/` (absent) | ADR-019 §Working defaults (minimum-viable set) | BUILD-SCOPE | Author `code-generator.yaml` under `agentic-serving/`, tagged `topaz_skill: code_generation`. The ADR prescribes promoting from Cycle 4 PLAY's `agentic-coding-helper`; the name changes, the shape (coder/critic/synthesizer) is reusable. | Yes |
| 4 | `claim-extractor` capability ensemble not authored | `.llm-orc/ensembles/agentic-serving/` (absent) | ADR-019 §Working defaults (minimum-viable set) | BUILD-SCOPE | Author `claim-extractor.yaml` under `agentic-serving/`, tagged `topaz_skill: factual_knowledge`. | Yes |
| 5 | `argument-mapper` capability ensemble not authored | `.llm-orc/ensembles/agentic-serving/` (absent) | ADR-019 §Working defaults (minimum-viable set) | BUILD-SCOPE | Author `argument-mapper.yaml` under `agentic-serving/`, tagged `topaz_skill: logical_reasoning`. | Yes |
| 6 | `prose-improver` capability ensemble not authored | `.llm-orc/ensembles/agentic-serving/` (absent) | ADR-019 §Working defaults (minimum-viable set) | BUILD-SCOPE | Author `prose-improver.yaml` under `agentic-serving/`, tagged `topaz_skill: writing_quality`. | Yes |
| 7 | `text-summarizer` capability ensemble not authored | `.llm-orc/ensembles/agentic-serving/` (absent) | ADR-019 §Working defaults (minimum-viable set) | BUILD-SCOPE | Author `text-summarizer.yaml` under `agentic-serving/`, tagged `topaz_skill: summarization`. | Yes |
| 8 | `web-searcher` script-agent ensemble not authored | `.llm-orc/ensembles/agentic-serving/` (absent); `.llm-orc/scripts/` (no web-search script) | ADR-020 | BUILD-SCOPE | Author `web-searcher.yaml` as a single-agent script-model-slot ensemble tagged `topaz_skill: tool_use`; author the Tavily-adapter Python script reading `WEB_SEARCH_API_KEY` and `WEB_SEARCH_BACKEND` env vars; document the no-op tier-escalation shape in the README. | Yes |
| 9 | `agentic-serving/README.md` not authored | `.llm-orc/ensembles/agentic-serving/` (absent) | ADR-019 §Working defaults | BUILD-SCOPE | Author the operator-facing README documenting the subdirectory structure, ensemble categories (system vs. capability), extension pattern, and Topaz skill tagging requirement. | Yes |
| 10 | `agentic_serving:` section in config.yaml not yet rewritten to reference renamed profiles and new subdirectory layout | `.llm-orc/config.yaml` lines 465–578 | ADR-019 §Working defaults | BUILD-SCOPE | After the profile file and subdirectory are created, rewrite the section: replace inline profile names with references to `agentic-serving-profiles.yaml`-defined names; confirm checker_ensemble and summarizer.ensemble paths reflect the new subdirectory. | Yes |
| 11 | `agentic-coding-helper.yaml` lives at the top level of `ensembles/`, named with a framework-coupling prefix | `.llm-orc/ensembles/agentic-coding-helper.yaml` | ADR-019 (operation-named, agentic-serving/ namespace) | COMPATIBLE / NOTE | No action required before BUILD. ADR-019 prescribes that `code-generator` (promoted from `agentic-coding-helper`) lives under `agentic-serving/`. The existing file can coexist until BUILD authors the replacement; deletion of `agentic-coding-helper.yaml` and any config references is part of the promotion work package. | Yes (promotion is part of item 3) |
| 12 | `development/code-review.yaml` is tagged `topaz_skill: instruction_following` but lives outside `agentic-serving/` | `.llm-orc/ensembles/development/code-review.yaml` line 9 | ADR-019 (OD-5 deferred) | COMPATIBLE / NOTE | OD-5 (placement of general-purpose ensembles) was explicitly carried as a style decision deferred to BUILD (cycle-status.md). The ensemble's `topaz_skill` tag is correct and conformant with ADR-015. Whether it moves to `agentic-serving/` is a BUILD-phase placement decision, not a violation. No action required before that decision is made. | OD-5 — style/placement decision at BUILD |
| 13 | System ensembles (`agentic-calibration-checker.yaml`, `agentic-result-summarizer.yaml`) carry no `topaz_skill` tag | `.llm-orc/ensembles/agentic-calibration-checker.yaml`, `agentic-result-summarizer.yaml` | ADR-019 (library shape principle) | COMPATIBLE / NOTE | System ensembles are dispatched by name from config.yaml (checker_ensemble, summarizer.ensemble), not via the Topaz-skill routing path. ADR-019's `topaz_skill` tagging requirement applies to **capability ensembles** in the library (dispatch-routing path). System ensembles that are hardwired-by-name in config are out of scope for the tagging requirement. No violation. | N/A |

---

## Notes

### No engine-code violations

As expected, ADRs 019–021 impose constraints on library shape (YAML configuration, ensemble authoring, profile file structure). They do not impose new engine-code constraints. A scan of `src/llm_orc/` found nothing that contradicts these ADRs. The `invoke_ensemble` fresh-context dispatch property that ADR-021 depends on is a pre-existing implementation fact; ADR-021 names it but does not require a code change to produce it.

### All 10 BUILD-scope gaps are expected substrate

The ADRs were authored on 2026-05-12; the codebase is at DECIDE phase close. The BUILD-scope items are the explicit work program Cycle 5 BUILD is being opened to execute. They are not failures of the existing codebase — they are the artifacts the ADRs defer to BUILD. None of the 10 items represent a situation where the codebase did something the ADR prohibits.

### `per_skill_tier_defaults` is fully populated and conformant

All 8 Topaz skill slots are present in `.llm-orc/config.yaml` under `agentic_serving.orchestrator.per_skill_tier_defaults` (lines 502–526). The resolver requirement (all 8 slots must be present at session start) is satisfied. The `mathematical_reasoning` slot is configured with `tier-cheap-general` / `tier-escalated-reasoning` profiles, which satisfies the ADR-019 §Neutral note: the slot is configured so the resolver does not fail; it will produce `MissingSkillMetadataError` at dispatch time because no library ensemble carries `topaz_skill: mathematical_reasoning` — which is the deliberate ADR-019-prescribed behavior.

### `orchestrator-minimax-m25-free` profile is defined out-of-project

The profile referenced at config.yaml line 476 (`model_profile: orchestrator-minimax-m25-free`) is not defined in the project-local `.llm-orc/config.yaml` or any file under `.llm-orc/profiles/`. It is defined in the operator's global config at `~/.config/llm-orc/config.yaml` lines 110–118. This is the existing deployment pattern (global config carries cloud-provider profiles; project config carries project-specific profiles). ADR-019's profile-file prescription (`agentic-serving-profiles.yaml`) is intended to isolate agentic-serving profiles for operator clarity. Whether `orchestrator-minimax-m25-free` should be in the project-local profile file or remain global is a BUILD-phase placement decision with no current violation.

### `topaz_skill` coverage across the existing ensemble library

Two ensembles carry `topaz_skill` tags: `agentic-coding-helper.yaml` (code_generation) and `development/code-review.yaml` (instruction_following). The remaining ~55 ensembles are untagged. ADR-019 does not require retroactive tagging of the entire library — it prescribes that the **capability ensemble library** (the operation-named set under `agentic-serving/`) is tagged. Pre-existing ensembles outside that namespace are not in scope for the tagging requirement. The Cycle 4 PLAY observation ("2 of 57 tagged") reflects the state correctly; the ADR's prescription does not change the count requirement for pre-existing ensembles.

### ADR-021 per-capability dispatch: no implementation delta required

ADR-021 describes a *composition contract* for skill framework clients, not an engine change. The contract is: skill frameworks emit one request per capability-typed sub-task; the orchestrator routes each via `invoke_ensemble`. This is how the existing `invoke_ensemble` tool works already. ADR-021 clarifies the intended usage pattern and adds the falsification trigger and seam-case inversion section, but requires no new code. The only BUILD artifact ADR-021 might motivate is the `skill-framework-capability-registry.md` document (OD-6), which is a documentation artifact, not a code change.

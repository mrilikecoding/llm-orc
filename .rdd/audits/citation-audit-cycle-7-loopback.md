# Citation Audit Report

**Audited document:** `docs/agentic-serving/essays/essay-outline-006-cross-compatibility-routing-surface.md`
**Evidence trail:** `docs/agentic-serving/essays/research-logs/research-log.md`
**Scope:** Amendment B (2026-05-24 BUILD → RESEARCH loop-back) — new C8 material only; C1-C7 treated as previously verified
**Date:** 2026-05-24

---

## Summary

- **Total references checked (new C8 material):** 12 distinct citation targets
- **Verified:** 10
- **Issues found:** 3 (0 P1, 1 P2, 2 P3)

---

## Verified References

The following new citations all resolve correctly:

- `[research-log-loopback]` — the reference entry points to `docs/agentic-serving/essays/research-logs/research-log.md`; the file exists and is the correct artifact.
- `[opencode]` — "OpenCode agentic coding client, v1.15.5"; version confirmed in research log Phase 0 ("Tooling (verified 2026-05-24): OpenCode 1.15.5"). URL `https://opencode.ai` confirmed by corpus references. Tool inventory (write/edit/bash/read/skill/task + others, `tool_choice: "auto"`, streamed) matches research log Phase 0 verbatim.
- `[research-log-loopback §Spike π Phase A]` — the research log has section `### Phase A — Direct-write + text acknowledgment (necessity test, run first)` nested under `## Spike π`. The cited finding (zero tool events, text-only, parity fails) matches the section body exactly.
- `[research-log-loopback §Spike π Phase B]` — section `### Phase B — tool_calls round-trip` exists; parity verdict and loop closure match citation.
- `[research-log-loopback §Spike ρ]` — section `### Spike ρ — Planner-driven delegation + tool_calls terminal (combined)` exists; planner decision JSON, delegation-not-hardcoded finding, and C1-suppression-does-not-recur finding all match.
- `[research-log-loopback §Spike σ.1]` — in the `### Spike σ` section, the σ.1 result (qwen3:14b, 3-step task: write calc.py → write test_calc.py → bash; `__pycache__/` + "PASS"; rc 0) matches E8.3.1 exactly.
- `[research-log-loopback §Spike σ.2]` — σ.2 result (layer-A driver + write delegation to code-generation ensemble; ensemble-generated test passed) matches E8.3.2 exactly.
- `[research-log-loopback §layer A/B distinction]` — the layer A / layer B motivation paragraph in `### Spike σ` matches E8.4.1's characterization of the routing planner as which-capability, not what's-the-next-agentic-step.
- `[research-log-loopback §Spike ρ F-ρ.1]` — F-ρ.1 artifact bridge finding is present in the Spike ρ section body; content matches E8.4.2 (deliverables route to server-side artifact store, terminal must read and marshal into client tool_call).
- `[adr-025]` — file exists at `docs/agentic-serving/decisions/adr-025-artifact-as-substrate.md`. E8.4.2's claim that capability ensembles carry `output_substrate: artifact` is confirmed by ADR-025 §Decision ("All capability ensembles route their deliverable through artifact-as-substrate"). The production `code-generator` ensemble YAML confirms `output_substrate: artifact` at line 36.
- `[adr-028]` — file exists at `docs/agentic-serving/decisions/adr-028-routing-planner-ensemble.md`. E8.4.1's characterization of ADR-028 as the routing-planner-ensemble ADR that "decides which-capability, not what's-the-next-agentic-step" is accurate per ADR-028 §Decision.
- **AS-10 reference (W8.2)** — AS-10 is defined in `docs/agentic-serving/domain-model.md` (Amendment Log entry 14): "Capability matching on the agentic-serving chat-completions surface works from request content alone." The essay's paraphrase — "routes on request content (AS-10), not on the client's declared tools" — accurately captures the invariant's scope.
- **E4.2.1 reference (E8.5.1)** — E8.5.1 explicitly names E4.2.1 as the geography argument that "dies under co-location." E4.2.1 exists in the document body (C4 sub-tree) and is correctly characterized as the "working inference from disjoint-filesystem premise." The cross-reference is internally consistent.
- **PLAY note 22 reference (C8 Abstract, W8.1)** — note 22 is referenced as the trust-surface analog ("model told-about vs. driving"). The research log Phase A uses this analogy directly ("structurally the PLAY-note-22 trust surface"). Field-notes.md contains the note (line 768 in field-notes.md confirms note 22's multi-dispatch fabrication pattern). The characterization is accurate.
- **`[research-log-loopback §Step 1.2 constraint-removal]` (E8.5.1)** — section `### Step 1.2 — Constraint-removal` exists in the research log and contains the practitioner co-location response. Citation resolves correctly.
- **`agentic-serving-cycle-status §"BUILD-surfaced finding: client-tool-action terminal gap"` (Section 9)** — this section exists in `docs/agentic-serving/housekeeping/cycle-status.md` (line 83) and confirms: pipeline emits only `ContentDelta` + `Completion`, never `finish_reason: "tool_calls"`. The commits `e538264` + `0a7a822` are named at lines 156 and 159. WP-A characterization in Section 9 is accurate.

---

## Issues

### P1 — Must Fix

None.

---

### P2 — Should Fix

**P2-1: F-ρ.2 factual claim is inaccurate — `agentic-tier-cheap-general` is not undefined**

- **Location:** Section 9 SCOPE QUALIFICATION (line 279); Amendment B Amendment Log (line 479); research log F-ρ.2 (line 120 of research-log.md)
- **Claim:** The essay states "its undefined `agentic-tier-cheap-general` profile per F-ρ.2 are BUILD-phase concerns." The Amendment Log states "`agentic-tier-cheap-general` is referenced in `per_skill_tier_defaults` but undefined as a model profile in the current config." The research log makes the same claim: "`list-profiles` omits it."
- **Finding:** The profile is defined. `.llm-orc/profiles/agentic-tier-cheap-general.yaml` exists and has existed since Cycle 5 BUILD (git history: commits `a935078` and `7925ca6`). The `get_model_profiles()` implementation in `src/llm_orc/core/config/config_manager.py` (lines 496-507) loads profiles from both `config.yaml` inline and the `profiles/` directory, so `list-profiles` would include this profile. The claim that it is "undefined" or that "`list-profiles` omits it" is factually incorrect against current codebase state. (Note: if the spike was run against a different project-local config that genuinely lacked the profile, this would need to be documented as a context-specific observation rather than a general-state claim.)
- **Recommendation:** Remove the "undefined `agentic-tier-cheap-general` profile" language from Section 9's SCOPE QUALIFICATION and from the Amendment B Amendment Log, or replace with: "F-ρ.2 (config hygiene) — the production code-generator ensemble references `agentic-tier-cheap-general` which exists in `.llm-orc/profiles/` but may not be present in minimal deployment configs; operators must verify the profile resolves at BUILD/deployment." The finding in the research log should similarly be corrected or contextualized. Since F-ρ.2 is a BUILD-phase concern flagged for carry-forward rather than a load-bearing warrant in the C8 argument, the impact is limited — but the factual error should not persist in the corpus.

---

### P3 — Consider

**P3-1: Section reference notation for Spike ρ's F-ρ.1 finding does not match a heading**

- **Location:** E8.4.2: `[research-log-loopback §Spike ρ F-ρ.1]`
- **Claim:** The citation uses a composite reference `§Spike ρ F-ρ.1` implying a subsection or named heading.
- **Finding:** The research log's Spike ρ section (`### Spike ρ — Planner-driven delegation + tool_calls terminal (combined)`) does not have a sub-heading named `F-ρ.1`. F-ρ.1 is a bolded paragraph within the section body, not a heading. The citation resolves to the correct content — F-ρ.1's text is present and matches — but the notation `§Spike ρ F-ρ.1` implies a heading that does not exist.
- **Recommendation:** Change to `[research-log-loopback §Spike ρ (finding F-ρ.1)]` to accurately reflect that F-ρ.1 is a finding label within the section, not a sub-section heading. No content correction needed; formatting only.

**P3-2: `[research-log-loopback §Spike σ + §layer A/B distinction]` references a section label that is a paragraph, not a heading**

- **Location:** E8.4.1: `[research-log-loopback §Spike σ + §layer A/B distinction]`
- **Claim:** The citation implies two sections: `§Spike σ` and `§layer A/B distinction`.
- **Finding:** `### Spike σ — Sustained multi-turn: does the loop hold, and what drives it?` is a real heading. "The layer A / layer B distinction (motivation):" is the opening bold paragraph of that section, not a separate heading. The content resolves correctly — the layer-A/layer-B distinction paragraph is exactly where the cited finding appears — but `§layer A/B distinction` notates it as a section heading when it is inline text.
- **Recommendation:** Change to `[research-log-loopback §Spike σ (layer A/B distinction)]` to clarify it references content within the Spike σ section. No content correction needed.

---

*This is a single-purpose re-audit dispatched per the re-audit-after-revision rule (Amendment B verification). The Convergence-Saturation Signal verdict line is omitted per ADR-094.*

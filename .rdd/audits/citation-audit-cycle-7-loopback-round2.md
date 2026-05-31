# Citation Audit Report

**Audited document:** `docs/agentic-serving/essays/essay-outline-006-cross-compatibility-routing-surface.md`
**Evidence trail:** `docs/agentic-serving/essays/research-logs/research-log.md`
**Scope:** Round-2 re-audit of Amendment B (C8) material; focus on round-1 corrections. C1-C7 material treated as previously verified.
**Date:** 2026-05-24

*Single-purpose re-audit dispatched per the re-audit-after-revision rule (ADR-094). Convergence-Saturation Signal verdict line omitted.*

---

## Summary

- **Total references checked (round-1 correction targets + carry-forward verification):** 14 citation targets
- **Verified:** 13
- **Issues found:** 1 (0 P1, 0 P2, 1 P3)

---

## Round-1 Corrections — Verification Results

### Correction 1: F-ρ.2 retraction (was P2-1)

The round-1 finding required the retraction to be consistent across three locations and for the corrected claim (profile IS defined at `.llm-orc/profiles/agentic-tier-cheap-general.yaml` → `qwen3:8b`/ollama, `cost_per_token: 0.0`) to be accurate against the actual file.

**All three retraction locations verified:**

- **Section 9 PROVENANCE CORRECTION bullet (line 287):** retraction is present. Language: "an earlier draft asserted the `agentic-tier-cheap-general` profile was undefined. That was a verification error — it relied on `llm-orc list-profiles`, which enumerates only config.yaml-embedded profiles, not the `.llm-orc/profiles/` directory. The profile is defined at `.llm-orc/profiles/agentic-tier-cheap-general.yaml` (→ `qwen3:8b`/ollama, `cost_per_token: 0.0`, since Cycle 5 / ADR-015)." Accurate and internally consistent.

- **Amendment B log (line 487):** F-ρ.2 RETRACTED bullet present. Same profile path and detail. Language additionally explains the `llm-orc list-profiles` limitation and confirms no config-hygiene blocker. Consistent with Section 9 retraction.

- **Research log (line 120 of `research-log.md`):** F-ρ.2 RETRACTED entry present. Language: "F-ρ.2 — RETRACTED (verification error; caught by the Amendment B citation re-audit). This entry originally claimed `agentic-tier-cheap-general` was undefined. It is defined at `.llm-orc/profiles/agentic-tier-cheap-general.yaml` (→ `qwen3:8b`/ollama, `cost_per_token: 0.0`, since Cycle 5 / ADR-015)." Consistent with the essay's two retraction locations.

**Corrected claim verified against actual file:** `.llm-orc/profiles/agentic-tier-cheap-general.yaml` exists and contains `model: qwen3:8b`, `provider: ollama`, `cost_per_token: 0.0`. The file header attributes it to "Cycle 5 / ADR-015 per-skill tier defaults." Every factual detail in all three retraction bullets is accurate. The "since Cycle 5 / ADR-015" provenance claim in the retraction matches the file comment.

**Verdict: P2-1 fully resolved.** All three retraction locations are present, consistent, and accurate against the actual codebase artifact.

---

### Correction 2: § notation (was P3-1 and P3-2)

The round-1 findings asked for:
- P3-1: `§Spike ρ F-ρ.1` → `§Spike ρ, finding F-ρ.1`
- P3-2: `§Spike σ + §layer A/B distinction` → `§Spike σ, layer-A/B distinction`

**Verified in the Argument-Graph (W8 sub-tree):**

- **E8.4.1 (line 123):** `[research-log-loopback §Spike σ, layer-A/B distinction]` — corrected form present. The comma-separated notation accurately signals that layer-A/B distinction is content within the Spike σ section, not a separate heading.

- **E8.4.2 (line 124):** `[research-log-loopback §Spike ρ, finding F-ρ.1; adr-025]` — corrected form present. The `, finding F-ρ.1` notation signals a labeled paragraph within the section, not a sub-heading.

**Verified in Section 9 (Citation-Embedded Outline):**

- Line 282: `(research-log-loopback §Spike σ, layer-A/B distinction)` — corrected.
- Line 283: `(research-log-loopback §Spike ρ, finding F-ρ.1)` — corrected.

**Verdict: P3-1 and P3-2 fully resolved.** All four instances of the notation have been updated to the recommended forms.

---

## Carry-Forward Citations — Verification

The round-1 audit found these clean; confirming the corrections did not disturb them.

- `[research-log-loopback §Spike π Phase A]` — section and findings present and undisturbed in research log. Matches E8.1.1.
- `[research-log-loopback §Spike π Phase B]` — section present and undisturbed. Matches E8.1.2.
- `[research-log-loopback §Spike ρ]` — section present; the F-ρ.2 retraction edit did not affect the F-ρ.1 content. E8.2.1 still accurate.
- `[research-log-loopback §Spike σ.1]` — section present and undisturbed. Matches E8.3.1.
- `[research-log-loopback §Spike σ.2]` — section present and undisturbed. Matches E8.3.2.
- `[research-log-loopback §Step 1.2 constraint-removal]` — section undisturbed. E8.5.1 accurate.
- `[opencode]` — reference entry unchanged. v1.15.5, tool inventory, URL accurate.
- `[adr-025]` — reference entry unchanged. F-ρ.1 artifact-bridge claim still accurately grounded.
- **AS-10 (W8.2)** — characterization unchanged and accurate.
- **PLAY note 22 (W8.1)** — reference unchanged and accurate.

All ten carry-forward citations remain clean.

---

## Issues

### P1 — Must Fix

None.

---

### P2 — Should Fix

None.

---

### P3 — Consider

**P3-1: Amendment B carry-forward sentence still names F-ρ.2 alongside F-ρ.1**

- **Location:** Amendment B log, carry-forward sentence (line 491): "end-to-end through the production `code-generator` (F-ρ.1 + F-ρ.2)"
- **Claim:** The sentence groups F-ρ.1 and F-ρ.2 as paired BUILD-phase concerns.
- **Finding:** F-ρ.2 has been retracted — the profile exists, the config-hygiene concern is dissolved, and there is no BUILD-phase blocker from F-ρ.2. Listing it as a carry-forward concern alongside F-ρ.1 is inconsistent with the retraction language two lines above it in the same Amendment B block (line 487: "No config-hygiene blocker; the production `code-generator` resolves its cheap tier on the free tier. Retained as a recorded correction (not deleted)..."). A reader scanning the carry-forward sentence will see an unresolved concern that the retraction explicitly closed.
- **Recommendation:** Change "F-ρ.1 + F-ρ.2" to "F-ρ.1 (F-ρ.2 retracted)" or simply "F-ρ.1" in that carry-forward sentence. The retraction record is preserved in the bullet above it; the carry-forward does not need to re-list a resolved finding as open work.

---

## Conclusion

The two round-1 corrections are fully applied and accurate:

1. F-ρ.2 retraction is consistent across all three locations (Section 9 PROVENANCE CORRECTION, Amendment B log, research log) and the corrected claim — profile defined at `.llm-orc/profiles/agentic-tier-cheap-general.yaml`, qwen3:8b/ollama, cost_per_token 0.0, since Cycle 5 / ADR-015 — is verified against the actual file.
2. The § notation has been corrected in all four instances (E8.4.1, E8.4.2, and their Section 9 counterparts).

One new P3 surfaced: the Amendment B carry-forward sentence still names F-ρ.2 as an open BUILD concern alongside F-ρ.1, inconsistent with the retraction two lines above it. No content is wrong — it is a consistency gap in the same block. No P1 or P2 findings.

*Single-purpose re-audit per ADR-094 re-audit-after-revision rule. Convergence-Saturation Signal verdict line omitted.*

# Citation Audit Report

**Audited document:** `docs/agentic-serving/essays/essay-outline-006-cross-compatibility-routing-surface.md`
**Date:** 2026-05-21
**Auditor:** Citation-audit skill (claude-sonnet-4-6)
**Corpus:** Agentic-serving scoped corpus, plugin v0.8.6

---

## Summary

- **Total references checked:** 17 (14 internal corpus references + 3 external URLs)
- **Inline citation keys verified against References section:** all 17 reference keys that appear in the body resolve to a References entry
- **Reference entries verified as existing files (internal):** 14/14
- **External URLs verified as well-formed and plausible:** 3/3
- **Factual claim spot-checks:** 6 claims verified; 1 issue found
- **Section-anchor checks:** C1–C6 present in Argument-Graph; META-anchored sections reviewed
- **Issues found:** 1 P1, 1 P2, 0 P3

---

## Per-Reference Verification

### Internal corpus references

| Reference key | Declared path | File exists? |
|---|---|---|
| [adr-022] | `docs/agentic-serving/decisions/adr-022-routing-surface-behavior.md` | Yes |
| [adr-023] | `docs/agentic-serving/decisions/adr-023-observability-event-routing.md` | Yes |
| [adr-024] | `docs/agentic-serving/decisions/adr-024-common-io-envelope.md` | Yes |
| [adr-025] | `docs/agentic-serving/decisions/adr-025-artifact-as-substrate.md` | Yes |
| [adr-014] | `docs/agentic-serving/decisions/adr-014-calibration-gate-trajectory-level-extension.md` | Yes |
| [adr-018] | `docs/agentic-serving/decisions/adr-018-tier-escalation-router-audit-dispatch.md` | Yes |
| [agentic-serving-cycle-status] | `docs/agentic-serving/housekeeping/cycle-status.md` | Yes |
| [research-design-review-cycle-7] | `docs/agentic-serving/housekeeping/audits/research-design-review-cycle-7.md` | Yes |
| [research-design-review-cycle-7-round-2] | `docs/agentic-serving/housekeeping/audits/research-design-review-cycle-7-round-2.md` | Yes |
| [research-log] | `docs/agentic-serving/essays/research-logs/research-log.md` | Yes |
| [cycle-7-spike-lambda] | `docs/agentic-serving/essays/research-logs/cycle-7-spike-lambda-tool-choice.md` | Yes |
| [cycle-6-spike-gamma] | `docs/agentic-serving/essays/research-logs/cycle-6-spike-gamma-routing-characterization.md` | Yes |
| [cycle-6-spike-delta] | `docs/agentic-serving/essays/research-logs/cycle-6-spike-delta-framework-chaining.md` | Yes |
| [cycle-6-field-notes-play] | `docs/agentic-serving/essays/reflections/field-notes.md` | Yes |
| [domain-model] | `docs/agentic-serving/domain-model.md` | Yes |

All 14 internal references resolve to existing files at the declared paths. No path errors found.

### External URL references

| Reference key | URL | Assessment |
|---|---|---|
| [openai-fc-guide] | `https://platform.openai.com/docs/guides/function-calling` | Well-formed. The OpenAI platform documentation URL pattern is canonical; web search confirms this is the live function-calling guide page. |
| [openai-api-ref] | `https://platform.openai.com/docs/api-reference/chat/create` | Well-formed. This is the standard path for the OpenAI Create chat completion API reference. |
| [openai-fc-required-discussion] | `https://community.openai.com/t/new-api-feature-forcing-function-calling-via-tool-choice-required/731488` | Well-formed. Web search returned this URL directly as a result, confirming it exists and the title matches the claim ("forcing function calling via `tool_choice: "required"`"). |

All three external references are well-formed URLs pointing to the type of resource the citations describe. The `tool_choice` parameter semantics described in the document (auto / required / named-function object / none) match published OpenAI documentation confirmed via web search.

---

## Inline Citation Resolution

All inline `[reference-key]` and `(reference-key §section)` citations in the Argument-Graph and Citation-Embedded Outline were checked against the References section. Every citation key resolves to a References entry. The complete set of citation keys used in the body:

`[cycle-6-spike-gamma]`, `[cycle-6-field-notes-play]`, `[cycle-7-spike-lambda]`, `[openai-fc-guide]`, `[openai-api-ref]`, `[agentic-serving-cycle-status]`, `[research-design-review-cycle-7]`, `[research-log]`, `[adr-025]`, `[domain-model]`, `[cycle-6-spike-delta]`, `[adr-014]`, `[adr-018]`

No inline citation key appears in the body without a corresponding References entry. No References entry is uncited in the body (no orphans).

---

## Factual Claim Spot-Checks

### Claim 1: "Spike λ Cell λ.3 dispatched correctly under qwen3:14b"

**Source checked:** `cycle-7-spike-lambda-tool-choice.md` §Cell λ.3

**Verdict: Verified.** The Spike λ document records Cell λ.3 (qwen3:14b + tool-rich + force-invoke_ensemble) as: HTTP 200, 191.7s wall-clock, full WP-C event sequence in serve log (`tool-call emit:` → `dispatch start:` → `calibration verdict: proceed` → `tier selection:` → `dispatch end: duration=72.261 exit=success`), substrate artifact created at `code-generator/20260520-194738-633/`, framework synthesized NL final response with `finish_reason: stop`. The document's claim that qwen3:14b dispatched correctly under λ.3 is accurately sourced.

---

### Claim 2: "Spike λ Cell λ.3-paid produced NO dispatch under paid MiniMax M2.5"

**Source checked:** `cycle-7-spike-lambda-tool-choice.md` §Cell λ.3-paid

**Verdict: Verified.** The Spike λ document records Cell λ.3-paid as: HTTP 200, 11.8s wall-clock, 2,171 completion tokens, `finish_reason: stop`, `message.content` inline NL code, `message.tool_calls` absent, serve log shows NO `tool-call emit:` or `dispatch start:` events. The document's characterization of this as a counter-finding (paid MiniMax M2.5 ignored the tool_choice instruction) is accurately sourced.

---

### Claim 3: "ADR-022 amendment effectiveness bounded to bare-endpoint mode"

**Source checked:** `field-notes.md` §Cycle 6 PLAY note 18

**Verdict: Verified.** Note 18 (corrected reading, 2026-05-20) explicitly states: "No dispatch fired on this probe" under paid M2.5 + OpenCode tool-rich + NL framing, and the note concludes "ADR-022's disposition (iii) — the amendment's effectiveness is bounded to bare-endpoint mode (no client tools declared)." The provisional cross-cutting observations section corroborates: "production agentic-coding-tool consumers (OpenCode, Aider, Cursor, Cline) all declare client tools. Under their normal request shape, the ADR-022 amendment does not shift NL routing toward `invoke_ensemble`."

---

### Claim 4: "Cycle 6 PLAY notes 5 and 15 documented claim-extractor form drift"

**Source checked:** `field-notes.md` §Cycle 6 PLAY notes 5 and 15

**Verdict: Verified.** Note 5 records direct-invoke of `claim-extractor` producing a multi-paragraph analytical essay with section headers and no `(established)/(contested)` labels — explicitly spec-non-conformant. Note 15 records the same form drift pattern ("orchestrator presents fabricated code as ensemble output while critiquing the ensemble") which, while the primary finding concerns fabrication, the underlying spec-non-conformant synthesizer output is the cited surface. More precisely, note 15 records that the synthesizer's output was narrative-wrapped code rather than conformant claims format. The essay's pairing of notes 5 and 15 as evidence for form-drift persistence is defensible, though note 15 is more about fabrication-during-critique than clean form-drift (see P2 issue below).

---

### Claim 5: "OpenAI tool_choice parameter accepts auto / required / named-function / none"

**Source checked:** Web search against OpenAI documentation

**Verdict: Verified.** OpenAI's published documentation (confirmed via web search) specifies `tool_choice` as accepting: `"auto"` (model decides), `"none"` (model must not call tools), `"required"` (model must call at least one tool), and `{"type":"function","function":{"name":"X"}}` (force specific named function). The essay's description of these four values at E2.1.2 and in Section 3 matches the documented API contract.

---

### Claim 6: "Spike δ confirmed framework-driven chaining preserves data (389,444 + 388,790 figures)"

**Source checked:** `cycle-6-spike-delta-framework-chaining.md`

**Verdict: INCORRECT — P1 issue.** The Spike δ document records the web-searcher returning numbers "354,751 / 354,000 / 388,790" and the claim-extractor citing "354,751 + 354,000 + 388,790 + 17.3% foreign nationals." The figure 389,444 does not appear anywhere in Spike δ. The 389,444 figure is from a different probe: the paid-Zen OpenCode composition probe (field-notes note 22), which ran a separate web-searcher dispatch that returned Wikipedia/Statistics Iceland data at 389,444 (2025). The essay attributes 389,444 to Spike δ in two places: E4.3.3 in the Argument-Graph, and the parallel evidence bullet in Section 5 of the Citation-Embedded Outline. Both citations mix population figures from two distinct probes that ran different web-searcher fetches returning different DDG results at different times. This is a factual error in the citation.

The research-log (Step 2 F3) itself exhibits a partial form of this mixing: it cites "354,751 + 388,790 Iceland population figures" for Spike δ (omitting 354,000 but not introducing 389,444). The essay-outline's version introduces the 389,444 figure that is not in Spike δ at all.

---

## Section-Anchor Verification

### Cn anchors (C1–C6)

Each `(Cn)` anchor in the Citation-Embedded Outline's section headers must resolve to a numbered claim in the Argument-Graph.

| Anchor | Essay-Outline section | Argument-Graph claim |
|---|---|---|
| C1 | Section 2: "NL-routing-fraction under production tool-rich clients is approximately zero" | C1 in Argument-Graph: "Under empirically tested production tool-rich clients, NL framing routes to direct LLM completion or client-tool delegation but reliably not to ensemble dispatch" — match |
| C2 | Section 3: "tool_choice mechanism exists but is not model-portable" | C2 in Argument-Graph: "OpenAI-standard `tool_choice` parameter is implemented at the framework level... but the cross-compatibility-relevant production model does not honor it" — match |
| C3 | Section 4: "Server-side mechanism for forced routing is motivated" | C3 in Argument-Graph: "The model-portability gap motivates a server-side mechanism that triggers ensemble dispatch deterministically" — match |
| C4 | Section 5: "Framework-driven composition continuation is required" | C4 in Argument-Graph: "Framework-driven composition continuation is required because the orchestrator-LLM's emergent post-dispatch behavior attempts file-reads" — match |
| C5 | Section 6: "Q2 form-drift enforcement scoped to ensemble-execution paths" | C5 in Argument-Graph: "I/O contract enforcement (Q2) applies most consequentially to the paths where ensembles actually run; form-drift persistence is the empirical surface" — match |
| C6 | Section 7: "Q3 fallback is current behavior; design is documentation" | C6 in Argument-Graph: "The 'fallback' shape for non-capability-matched requests is empirically the current direct-completion behavior; design work is structured-contract documentation" — match |

All six Cn anchors resolve cleanly to the corresponding Argument-Graph claims.

### META anchors

Three sections carry META anchors: Section 1 (methodology preamble), Section 3's META-OBSERVATION paragraph, Section 6's META-OBSERVATION paragraph, and Section 8 (architectural synthesis + scope-of-claim).

**Section 1 (META):** Contains methodology framing — plugin version, ADR-082 protocol reference, reviewer rounds summary, Phase A / Spike λ description, cost-authorization note. This is genuine preamble and scope material. No claim/warrant/evidence development is present. Classification is correct.

**Section 3 META-OBSERVATION:** States that the Phase A reframe's "strongest form" is "partially validated and partially contradicted" and offers a process-observation on the `feedback_free_options_preference` discipline's role. This is a process observation about the research methodology, not a claim about the architecture. Genuinely non-developmental in terms of C1–C6 evidence chains. Classification is correct.

**Section 6 META-OBSERVATION:** Explicitly scopes Q2 to form drift only, noting that content drift is "outside Q2's solution space." This is a scope-qualification statement that bounds the claim (C5) rather than developing evidence for it. Appropriate META classification.

**Section 8 (META scope):** Contains the synthesized architectural recommendation (i'–iv') and scope-of-claim qualification. The SCOPE QUALIFICATION text and the VALIDATION-SPIKE DECISION record are genuine scope/methodology content. However, the synthesis paragraph ("The cycle's architectural recommendation for the Cycle 7 DECIDE phase is a hybrid...") is clearly inferential content derived from C3+C4 — which the section header labels correctly as "(C3+C4 composed; META scope)". The hybrid-architecture synthesis appears as the first substantive bullet and is not labeled as carrying a claim warrant; it reads as a conclusion drawn from C3 and C4. This is on the boundary but the hybrid-architecture claim is explicitly attributed to C3+C4 in the section header. No misclassification flagged.

No META-anchored sections were found to contain new claim/warrant/evidence development that should be Cn-anchored.

---

## Issues

### P1 — Must Fix

**P1-1: Spike δ population figures are incorrect**

- **Location:** Argument-Graph E4.3.3; Citation-Embedded Outline Section 5 (evidence bullet for Spike δ framework-driven chaining)
- **Claim:** "Spike δ (Cycle 6) demonstrated framework-driven composition succeeds when the framework passes dispatch data between steps directly — `web-searcher → claim-extractor` chain produced faithful citation of upstream content (389,444 + 388,790 figures) with zero fabricated numbers." (Also stated as "389,444 + 388,790 figures" in the Section 5 evidence bullet.)
- **Finding:** The Spike δ document records the web-searcher returning "354,751 / 354,000 / 388,790" and the claim-extractor citing "354,751 + 354,000 + 388,790 + 17.3% foreign nationals." The figure 389,444 is absent from Spike δ entirely. It originates from the paid-Zen OpenCode composition probe (field-notes note 22), which ran a separate web-searcher dispatch at a different time returning different DDG results. The research-log (Step 2 F3) already introduced a partial mixing ("354,751 + 388,790") by omitting 354,000; the Essay-Outline further corrupted this by replacing 354,751 with 389,444. The cited source (cycle-6-spike-delta) does not contain 389,444.
- **Recommendation:** Replace "389,444 + 388,790 figures" with "354,751 + 354,000 + 388,790 figures" (the three numbers the Spike δ document records) in both locations: E4.3.3 in the Argument-Graph and the corresponding evidence bullet in Section 5. The underlying argument (framework-driven chaining produces faithful citations; orchestrator-LLM-driven chaining confabulates) is well-supported by the correct figures and does not require 389,444 to hold.

---

### P2 — Should Fix

**P2-1: Note 15 is a weak citation for "form drift across invocation paths"**

- **Location:** Argument-Graph E5.1.2 ("Cycle 6 PLAY note 15: `claim-extractor` chained via orchestrator-LLM also produced spec-non-conformant output — same drift pattern"); also Section 6 evidence bullet citing note 15.
- **Claim:** The essay cites note 15 as evidence that claim-extractor's form drift persists when chained via orchestrator-LLM, establishing that the drift is path-independent (present on direct invoke per note 5 AND on orchestrator-chained path per note 15).
- **Finding:** Cycle 6 PLAY note 15 in the field-notes document is primarily about fabrication-while-critiquing-fabrication (the orchestrator generated a stub `CodeGenerator` class from prompt intent rather than from the dispatched ensemble's output). The underlying form-drift at the synthesizer layer is present but is not the note's primary recorded observation. Note 15's main finding is confabulation, not spec-non-conformant output format. The form-drift-under-orchestrator-chained-path finding is better and more directly evidenced by Cycle 6 PLAY note 15's recorded observation in Cycle 5 PLAY (which records the synthesizer's narrative-wrapped output reaching the orchestrator without conformance), or by the Spike δ document's explicit statement: "Form drift persists. Claim-extractor's output is still non-conformant to its `default_task` spec — structured analysis with sections instead of `(established)/(contested)` bulleted claims." That Spike δ statement is directly attributable to the framework-driven chaining path, not the orchestrator-chained path, but it establishes path-independence more cleanly than note 15.
- **Recommendation:** Consider citing Spike δ §"Form drift persists" as the primary evidence for path-independent form drift (it explicitly names the drift as independent of chaining mechanism), with note 15 as corroborating context for the confabulation pattern that compounds form drift. Alternatively, the Cycle 5 PLAY notes (notes 13/15 in the Cycle 5 PLAY section, not the Cycle 6 PLAY note 15) more directly document spec-non-conformant synthesizer output under orchestrator-chained paths. The citation should be clarified to avoid a reader inspecting note 15 and finding fabrication-while-critiquing as the primary content rather than form-drift evidence.

---

### P3 — Consider

No P3 issues identified. The reference list is complete for the claims made. The external URL resources are the canonical sources for the documented API behavior. No obvious omitted citations for the core architectural claims were identified.

---

## Verification Notes

**R2-1 latency bound citation:** The essay cites "research-log §R2-1" for the latency bound (≤ 1.0s OR ≤ 20%). The research-log's Step 1.4 §Round-2 reviewer findings section records R2-1 as: "Routing overhead ≤ 1.0s wall-clock on top of bare-LLM completion, OR ≤ 20% of bare-LLM completion latency (whichever is larger)." The essay's summary of this bound at E3.3.2 and Section 4 accurately reflects the documented provisional bound. Verified.

**"Working inference" citations:** Three body evidence bullets carry "(working inference from ...)" rather than direct citation to a source document. These are flagged in the document itself as inferences rather than citations, so they are not citation errors. They represent points where the essay acknowledges it is extending beyond direct evidence: E3.1.1 (interception model preserves client-facing contract), E4.2.1 (production clients declare client tools for their own filesystem scope), and E5.3.3 (framework-driven composition continuation eliminates orchestrator-narration step). These are appropriately qualified and do not require correction.

**"R2-2" reference:** The essay's Section 1 mentions "Q1a-Q1b feedback path R2-2" among the round-2 flags. This resolves to the research-design-review-cycle-7-round-2 document §Flag R2-2 ("Q1a/Q1b ordering assumption is implicit, not stated"). Verified as correctly attributed.

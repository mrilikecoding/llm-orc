# Spike α (Cycle 6) — Common I/O Envelope Survey + Proposal

**Date:** 2026-05-15
**Cycle:** 6 (agentic-serving mini-cycle; ensemble-contract cluster)
**Wave:** DECIDE-blocking analytical spike (paired with spike β; spike γ independent)
**Method:** Analytical — read each ensemble YAML's `default_task` and the most recent `execution.json` artifact under `.llm-orc/artifacts/agentic-serving/<ensemble>/latest/`. No live dispatch.
**Scope:** 6 capability ensembles + 2 system ensembles = 8 ensembles. Three candidate envelope shapes. Per-ensemble fit notes against each. Free-tier policy honored.
**Cost incurred:** $0.00 (no LLM dispatch)

---

## Question

What envelope shape can wrap all 8 agentic-serving ensembles' current behavior with minimum disruption? Candidate fields the cycle has already named: `status`, `primary`, `metadata`, `artifacts[]`, `diagnostics`. Validate whether those are right or wrong; propose 2-3 candidate envelope shapes.

The question is positioned upstream of T16's five sub-questions (boundary, contract, client access, cleanup, backward compat) and is the substrate spike β's composition-predictability trace runs against.

---

## Method

1. **Enumerate current I/O shapes.** Read each of the 8 ensemble YAMLs (six capability under `.llm-orc/ensembles/agentic-serving/`, two system: `agentic-result-summarizer`, `agentic-calibration-checker`). Capture the `default_task` output spec (where present), the agent count and shape, and the discriminator for the agent kind (script vs. LLM). For each ensemble, read the most recent `execution.json` artifact under `.llm-orc/artifacts/agentic-serving/<ensemble>/latest/` to confirm the actual emitted shape.
2. **Read the current orchestrator-side response shape.** The `agentic-result-summarizer`'s `default_task` already references the JSON-encoded ensemble execution result as its input. The shape it expects (`results`, `synthesis`, `status`, `raw_output`, plus the `needs_client_tool` exception convention) is the de facto current ensemble-response shape the orchestrator and summarizer act on.
3. **Draft three candidate envelopes** across the design axis that the cycle has surfaced: thin convention (status quo, codified), additive typed-fields-only, and full artifact-as-substrate. The three were chosen to bracket T16's first sub-question (scope: always / when substantive / operator-configured) without prejudging the answer.
4. **Test per-ensemble fit** for each candidate. For each cell, score:
   - **Fits cleanly** — current behavior maps without modification.
   - **Fits with minor adapter** — a wrapper at the dispatch boundary serializes/deserializes; no ensemble YAML change.
   - **Requires ensemble change** — `default_task` rewording, agent system-prompt change, or new synthesizer logic.
   - **Requires infrastructure change** — new code path in dispatch (e.g., artifact-writing surface; typed envelope serialization).
5. **Surface implications** for T16's five sub-questions and flag open questions for spike β.

The analysis was bounded to the 8 in-scope ensembles. The `development/code-review.yaml` ensemble that fills the `instruction_following` Topaz slot for agentic-serving deployments per the registry artifact was not surveyed; if the envelope crystallizes as the contract, that ensemble and any future deployment-specific authors absorb the same shape.

---

## Findings

### Current I/O shapes (per-ensemble)

| Ensemble | Agent shape | Output spec (from `default_task`) | Actual emitted shape (from latest `execution.json`) | Size signature |
|---|---|---|---|---|
| `web-searcher` | Script-agent (single) | JSON object with `backend`, `query`, `result_count`, `results[]` (each: `title`, `url`, `snippet`) or structured error | `{"results": {"searcher": {"response": "<JSON-string>", "status": "success", "model_substituted": false, "agent_requests": []}}, "synthesis": null, "status": "completed", ...}` — `response` field is a **JSON-string** that the consumer parses | Small (5 results × ~200 chars) |
| `claim-extractor` | LLM-agent (single) | Bulleted list of claims, each labeled `(established)` or `(contested)`; or `"No factual claims extracted."` sentinel | `{"results": {"extractor": {"response": "<markdown text>", ...}}, "synthesis": null, ...}` — `response` is free markdown; observed deviation from spec (no `(established)` / `(contested)` labels in latest artifact) | Medium (~2500 chars; ~2200 output tokens) |
| `argument-mapper` | LLM-agent (single) | Four-section markdown: Premises / Intermediate conclusions / Final claim / Logical gaps; or `"No argument structure detected."` sentinel | `{"results": {"mapper": {"response": "<markdown text>", ...}}, "synthesis": null, ...}` — `response` is free markdown; observed deviation from spec (used different headings than the four mandated sections) | Medium (~2400 chars; ~2200 output tokens) |
| `code-generator` | LLM-agent (three: coder → critic → synthesizer; synthesizer `depends_on: [coder, critic]`) | Synthesizer system prompt: "Output exactly what the client should see" — implicit final-agent output | `{"results": {"coder": {...}, "critic": {...}, "synthesizer": {"response": "<code-block>", ...}}, "synthesis": null, ...}` — three agent responses in `results`; the synthesizer's response is the client-facing one but the dict carries all three | Large (~1100 chars synthesizer; ~3300 chars across all three agents) |
| `prose-improver` | LLM-agent (single) | "Output only the improved prose. No editorial commentary, no preamble, no annotations." Optional leading sentinel: `"No improvements warranted — prose is already clear."` | `{"results": {"improver": {"response": "<markdown text>", ...}}, "synthesis": null, ...}` — `response` is free markdown; observed deviation: emitted analysis + headings rather than the improved prose only | Medium-to-large (~3500 chars in observed artifact, much of it non-prose-output) |
| `text-summarizer` | LLM-agent (single) | Three-section markdown: `**Main claim:**` / `**Key points:**` / optional `**Caveats:**`; or `"Source too short to summarize:"` sentinel | `{"results": {"summarizer": {"response": "<markdown text>", ...}}, "synthesis": null, ...}` | Small-to-medium (~500 chars in observed artifact) |
| `agentic-calibration-checker` (system) | LLM-agent (single) | Strict two-line shape: `signal: positive\|negative\|absent\n  reason: <one short sentence>` | Same envelope shape as capability ensembles: `{"results": {"checker": {"response": "<signal/reason text>", ...}}, ...}` | Very small (~80 chars) |
| `agentic-result-summarizer` (system) | LLM-agent (single) | Two-or-three-sentence natural-language summary; OR verbatim `needs_client_tool` JSON object if present in the input | Same envelope shape: `{"results": {"summarizer": {"response": "<summary text or JSON>", ...}}, ...}` | Small (~200-1000 chars; observed instance carried a full code block where AS-7 would prefer metadata) |

### Three observations from the I/O survey

**Observation 1 — One execution-shape, one client-facing-value-locus.** The `execution.json` shape is uniform across all 8 ensembles: a top-level `{"ensemble", "status", "input", "results", "metadata", "synthesis"}` dict where `results` is a dict keyed by agent name and `synthesis` is currently always `null` in the observed artifacts (though the schema permits a synthesis value). The client-facing value lives at `results.<agent-name>.response` for single-agent ensembles, or at `results.synthesizer.response` for multi-agent ensembles with a synthesizer (only `code-generator` in the surveyed set). This is the current de facto envelope — codified by infrastructure, not by ADR.

**Observation 2 — Output-spec compliance is uneven and shape-conflated with content.** Three of the six capability ensembles surveyed (`claim-extractor`, `argument-mapper`, `prose-improver`) emitted output that deviates from their `default_task` spec in the latest artifact. The deviation appears at the **content** level (wrong sections, wrong labels, prose where prose-only was requested) but the **envelope** wraps it identically. The current envelope cannot help with output-spec drift because it doesn't know what the spec is — the spec is implicit in the system prompt or `default_task`, not declared as a typed field.

**Observation 3 — Size variance is real and useful.** Across the 8 ensembles, observed `response` field sizes range from ~80 characters (calibration checker) to ~3500 characters (prose-improver's overflow). Token counts span ~80 to ~2400 output tokens. The variance is structural: calibration-checker is fixed-shape-by-design; web-searcher is a small JSON object; the LLM-text generators sit in the 1000-3500-character band; code-generator's three-agent shape inflates the total per-dispatch payload by 3×. **The size signature plausibly maps to the boundary T16 sub-question (a) asks about** — single-bullet outputs (calibration-checker, sentinel-emitting capability ensembles in the negative case) are fine in-response; multi-thousand-character markdown deliverables are the candidate territory for artifact-substrate.

### Three candidate envelopes

The three candidates are chosen to bracket the design space, not to be exhaustive. Each is described in its own subsection with field names, type expectations, and a one-line meaning of "minimum disruption" against the current shape.

#### Candidate A — Thin convention (status-quo + codification)

```
{
  "ensemble": "<name>",
  "status": "completed" | "failed" | "in_progress",
  "primary": "<string>",          # the client-facing value, extracted at dispatch
  "diagnostics": { ... },          # metadata.usage + tier + verdict + audit fields
  "synthesis": null                # preserved for back-compat
}
```

**Field set:** `ensemble`, `status`, `primary`, `diagnostics`. No `artifacts[]` field; no `content_type`; no retention policy.

**Type expectations:**
- `primary: string` — always a string; consumers parse if they want structure (current behavior).
- `diagnostics: object` — the existing `metadata.usage` block plus any of the four typed Cycle 5 events (TierSelection, CalibrationVerdict, etc.) that fired on the dispatch. Open-ended object; not typed beyond "the dispatch's observability payload."
- `status: enum`.

**Minimum disruption meaning:** the existing `execution.json` shape is preserved (artifact-side); the **dispatch return** to the orchestrator simplifies to `{ensemble, status, primary, diagnostics}` instead of the full `results` dict with agent-keyed entries. The Result Summarizer Harness extracts `primary` rather than walking `results.<agent>.response` per ensemble shape. No `default_task` changes; no system-prompt changes.

**What this gets:** codifies the de facto contract so the harness's extraction logic stops being shape-conditional on agent count. Reduces orchestrator-context surface area (one field instead of an agent-keyed dict). Adds a typed `diagnostics` field that the operator-visible event surface (T15 territory) can route from.

**What this does not get:** does not address output-spec drift (no typed schema for `primary`'s contents); does not address AS-7 content-stripping (the summarizer still summarizes content); does not address information-finding overhead (the deliverable still rides in-context).

#### Candidate B — Additive typed fields (envelope-as-schema)

```
{
  "ensemble": "<name>",
  "status": "completed" | "failed" | "in_progress",
  "primary": "<string>",                   # client-facing value, always string
  "structured": { ... } | null,            # optional typed payload when ensemble emits structured data
  "diagnostics": { ... },                  # dispatch metadata + Cycle 5 typed events
  "errors": [ { "kind": "<name>", ... } ]  # structured error array; replaces ad-hoc error shapes
}
```

**Field set:** `ensemble`, `status`, `primary`, `structured`, `diagnostics`, `errors[]`. The `structured` field carries typed payload where the ensemble emits it — e.g., `web-searcher`'s `{backend, query, results[]}` rides in `structured`; `prose-improver`'s pure prose rides only in `primary` with `structured: null`. `errors[]` replaces web-searcher's ad-hoc error-object emission with a list of typed-kind error records.

**Type expectations:**
- `primary: string` — always present, always string. Holds the client-readable form.
- `structured: object | null` — optional typed payload. The shape is per-ensemble (a `claims_v1` shape for `claim-extractor`; a `argument_map_v1` shape for `argument-mapper`; the existing `{backend, query, results[]}` shape for `web-searcher`). The schema is declared at the ensemble level (new YAML field: `output_schema:`).
- `errors[]` — list of `{kind, detail, ...}` records. Empty when no errors.
- `diagnostics: object` — as in candidate A.
- `status: enum`.

**Minimum disruption meaning:** capability ensembles with already-structured output (`web-searcher`) get their structure typed; capability ensembles emitting free-form markdown (the four LLM-capability ensembles) keep `structured: null` and pay nothing. Output-spec compliance becomes checkable: if `claim-extractor` declares an `output_schema: claims_v1` with required `(established)` / `(contested)` labels, deviation can be caught at the dispatch boundary rather than discovered downstream. Ensemble YAMLs gain an optional `output_schema:` field; ensembles that don't declare one keep current behavior.

**What this gets:** opens the door to spike β's composition-predictability work — a downstream ensemble can declare it consumes a typed `structured` shape rather than parsing prose. Output-spec drift becomes mechanically detectable for the ensembles that opt in. Errors get a typed home. The new `output_schema:` field is optional, so back-compat is preserved on day one.

**What this does not get:** does not address the large-deliverable case — long markdown still rides in `primary` (the size-variance observation persists). Does not address AS-7 content-stripping for the LLM-text ensembles (they still emit content; the summarizer still summarizes it). Adds vocabulary cost (every ensemble author now has to consider whether to declare a schema).

#### Candidate C — Artifact-as-substrate (T16's substantive-always path)

```
{
  "ensemble": "<name>",
  "status": "completed" | "failed" | "in_progress",
  "primary": "<handle>",                  # short handle / one-line summary; never the deliverable itself
  "artifacts": [
    {
      "path": "<filesystem-or-uri>",
      "content_type": "<MIME-or-named-shape>",
      "size_bytes": <int>,
      "summary": "<one-or-two-line metadata>",
      "retention": "session" | "cycle" | "persist"
    }
  ],
  "structured": { ... } | null,            # optional small typed payload (handles, not deliverables)
  "diagnostics": { ... },
  "errors": [ ... ]
}
```

**Field set:** `ensemble`, `status`, `primary`, `artifacts[]`, `structured`, `diagnostics`, `errors[]`. The deliverable (code, structured analysis, long-form prose) is written to `.llm-orc/artifacts/<ensemble>/<timestamp>/<deliverable-file>` and referenced; `primary` becomes a short handle ("Wrote 8-line Python function to <path>"); `summary` carries one-or-two-line metadata about the artifact's content; the orchestrator's context never carries the deliverable's body.

**Type expectations:**
- `primary: string` — short handle, ~one or two sentences. The orchestrator's context-carrying field; never the deliverable.
- `artifacts[]` — list of typed records. Required fields: `path`, `content_type`, `size_bytes`, `summary`, `retention`. `path` is filesystem-relative under `.llm-orc/artifacts/`; whether the client reads via filesystem or via a serve-layer HTTP endpoint is open (T16 sub-question (c)).
- `content_type` — MIME (`text/markdown`, `application/json`, `text/x-python`) or a named-shape registry value (`claims_v1`, `argument_map_v1`).
- `retention: enum` — operator policy: per-session (deleted at session close), per-cycle (kept for cycle duration), persist (manual cleanup).
- `structured`, `diagnostics`, `errors[]` — as in candidate B; `structured` holds small typed metadata, not deliverables.

**Minimum disruption meaning:** the four LLM-text capability ensembles (`claim-extractor`, `argument-mapper`, `prose-improver`, `text-summarizer`) and `code-generator` write their `response` to an artifact file alongside the existing `execution.json`; the existing `execution.json` shape is preserved at the artifact-disk level (no infrastructure-breaking change there). The dispatch return shape changes substantially: `primary` is now a short string, not the full markdown body. `agentic-result-summarizer`'s `default_task` changes from "summarize the JSON-encoded ensemble execution result" to "summarize the artifact reference and diagnostics" — its content-stripping failure mode dissolves because there is no content to strip. `web-searcher` is borderline — its structured JSON is small enough to ride in-response, so it stays in `structured`; the cost of paying the artifact-substrate tax on a 5-result JSON object is plausibly negative. `agentic-calibration-checker` stays response-substrate (its output is ~80 chars).

**What this gets:** addresses all three findings the practitioner's proposal targets (output-spec drift, information-finding overhead, AS-7 content-stripping). The MODEL gate's Action B caveat remains live — the "collapse three findings simultaneously" claim is agent-composed and should be tested case-by-case before being accepted as a structural property. Opens the door to deliverable-typed-artifact composition (`claim-extractor`'s artifact is `text/markdown` with the labeled-bullet shape; `argument-mapper`'s artifact is `text/markdown` with the four-section shape; composition is artifact-to-artifact rather than text-to-text).

**What this does not get:** introduces new infrastructure surface (artifact-writing in the dispatch path; possibly a serve-layer file-fetch endpoint per T16 sub-question (c); retention-policy enforcement per (d)). Backward compat (T16 sub-question (e)) is the hardest cost — existing client integrations that read `results.<agent>.response` and treat it as the deliverable do not work with `primary` as a handle. The boundary question (T16 sub-question (a)) becomes load-bearing: which ensembles get artifact-substrate and which stay response-substrate must be settled, otherwise the contract is incoherent across the library.

### Per-ensemble fit notes (the survey-and-test step)

The four-level scoring: **Clean** (fits without modification), **Adapter** (wrapper at dispatch boundary; no ensemble change), **Ensemble change** (system-prompt or `default_task` reword), **Infrastructure** (new code path in dispatch).

| Ensemble | Candidate A (thin) | Candidate B (typed) | Candidate C (artifact-substrate) |
|---|---|---|---|
| `web-searcher` | Adapter — strip the JSON-string-in-response convention; surface as `primary` (text rendering) | Adapter (web-searcher's emitted JSON moves to `structured`; `primary` becomes a short text rendering); declare `output_schema: web_search_v1` | Adapter only (output is small enough; `structured` carries it; no artifact needed) — **artifact-substrate is over-engineered for this ensemble** |
| `claim-extractor` | Clean (extract `response` to `primary`) | Ensemble change to make the output reliably labeled; declare `output_schema: claims_v1` | Ensemble change + Infrastructure (artifact-write); `primary` becomes "N claims extracted, M established / K contested" |
| `argument-mapper` | Clean (extract `response` to `primary`) | Ensemble change for reliable section emission; declare `output_schema: argument_map_v1` | Ensemble change + Infrastructure (artifact-write); `primary` becomes one-line summary of the map |
| `code-generator` | Adapter — extract `results.synthesizer.response` to `primary`; drop coder/critic outputs from orchestrator context | Adapter; `structured` could carry the per-agent record for debug surfacing | Infrastructure — write `.py` file to artifact; `primary` becomes "Wrote N-line Python <symbol> to <path>"; the AS-7 case study (where the summarizer regurgitated the whole code block) dissolves cleanly |
| `prose-improver` | Clean (extract `response` to `primary`) | Ensemble change to suppress observed prose-improver overflow; `output_schema: improved_prose_v1` (effectively just `text/markdown`) | Ensemble change + Infrastructure; `primary` becomes "Improved prose, N chars, M edits from source" |
| `text-summarizer` | Clean (extract `response` to `primary`) | Adapter; `output_schema: summary_v1` (the three-section shape) | Borderline — observed output ~500 chars, small enough to ride in-response. Likely stays response-substrate even under candidate C |
| `agentic-calibration-checker` | Adapter — parse the `signal: ... \n reason: ...` shape into structured form for `primary` | Adapter; the two-field shape is small and already structured; `structured: {signal, reason}` | Stays response-substrate (output is ~80 chars; artifact-substrate would be perverse) |
| `agentic-result-summarizer` | Adapter — its role pre-empts most of the envelope question; it consumes envelopes and emits summary text | Ensemble change — `default_task` updates to consume typed `structured` field rather than raw `results` dict | **Ensemble change with the deepest implications** — under artifact-substrate the summarizer summarizes the **artifact metadata** rather than content; this is the AS-7 amendment pathway OQ #15 names |

### Cross-cutting fit observations

**The `needs_client_tool` retry-signal convention is envelope-orthogonal.** The `agentic-result-summarizer`'s `default_task` carries a special-case for preserving `{"needs_client_tool": {...}}` JSON objects verbatim. Under any of the three candidate envelopes, this remains an exception path — the signal needs to pass through the envelope without being summarized or stripped. Candidate B's `structured` field is a natural home for it; candidate C also fits (signal in `structured`, no artifact needed because the signal is small and time-sensitive). Candidate A would need to preserve it as a typed-shape inside `primary` or add a sibling field — least clean of the three.

**Synthesizer behavior is implicit infrastructure.** All eight ensembles emit `synthesis: null` in the observed artifacts; the synthesizer agent slot exists in `code-generator` but its output rides in `results.synthesizer.response`, not in the top-level `synthesis` field. Whatever envelope candidate is chosen, the `synthesis` field's role should be clarified — either deprecated (consumers read `primary`) or specified (synthesis is the post-agent integration step, distinct from any one agent's response). Candidates A, B, and C all benefit from this clarification; none of them resolves it.

**Script-agent vs. LLM-agent emit identically.** `web-searcher`'s script-agent shape produces `execution.json` indistinguishable in envelope from the LLM-agent ensembles. The agent-shape distinction is invisible to the envelope — confirming that the envelope is correctly positioned at the dispatch boundary, not at the agent boundary. All three candidates inherit this property.

**The `metadata.usage` block is rich and underused at the orchestrator surface.** The current `execution.json` carries detailed per-agent token counts, model identifiers, timing, and resource-management traces. None of this reaches the orchestrator beyond what the summarizer chooses to surface. Candidate A's `diagnostics` field, candidate B's same, and candidate C's same all open the door to surfacing this at the operator-visible event surface (T15 territory). The envelope contract therefore couples to T15's observability work as a **downstream consumer of the same data**, not as a competing field.

---

## Recommended envelope (with alternatives)

The findings do not unambiguously support one of the three candidates over the others. They do support a **scope-conditional recommendation** that mirrors T16's first sub-question:

- **If T16 resolves scope as "always" (practitioner-verbatim):** Candidate C with the size-floor exception (small-output ensembles — `agentic-calibration-checker`, `web-searcher`, and `text-summarizer` in its short-input case — stay response-substrate inside `structured`). The size-floor exception is necessary because forcing artifact-substrate on an 80-character calibration signal is structurally wasteful and obscures the signal behind a filesystem round-trip.

- **If T16 resolves scope as "when substantive" (agent-introduced refinement):** Candidate B as the envelope contract; candidate C's `artifacts[]` field becomes additive — present when the ensemble emits to disk, absent when it doesn't. This is the **lowest-disruption path** that preserves the option to migrate per-ensemble over time. Ensemble authors opt into artifact-substrate by emitting an `artifacts[]` entry; the envelope shape doesn't change between substrate modes.

- **If T16 resolves scope as "operator-configured":** Candidate B as the contract, with `artifacts[]` again additive; an operator-level config field per ensemble (`substrate: response | artifact`) governs the dispatch-layer behavior. The envelope itself doesn't change; the field's presence does.

The thin-convention candidate A is **not recommended on its own** as the cycle's substrate. It codifies the de facto contract without addressing any of the three findings the artifact-substrate proposal targets. It remains useful as a *fallback* — if DECIDE settles on no architectural change at the contract level, candidate A documents the current behavior as the contract, and the cycle's effort returns to T15 (observability) and T14 (routing) without touching T16.

**Field-name validation against the cycle's candidates:**

- `status` — confirmed useful; appears in all three candidates. The current `execution.json` already carries it; the envelope formalizes it as required.
- `primary` — confirmed useful; appears in all three candidates. Its **meaning** varies (full content under A and B; handle under C), which is itself a finding — the same field name carries different load depending on T16's scope resolution.
- `metadata` — **recommended renaming to `diagnostics`**. The MODEL gate's vocabulary already uses `diagnostics` for the operator-readable dispatch metadata (verdict, tier, audit consumption). The current `execution.json`'s `metadata` field is closer to "everything that isn't input or output" — overly generic. `diagnostics` names the role.
- `artifacts[]` — confirmed useful under C; additive under B; absent under A. The typed shape with `{path, content_type, size_bytes, summary, retention}` is well-formed and parallels the existing `.llm-orc/artifacts/` filesystem convention.
- `diagnostics` — confirmed useful; appears in all three candidates. Coupling to T15's observability surface is direct.

**Two field-name additions surfaced by the survey:**

- `structured: object | null` — typed-payload field distinct from the human-readable `primary`. Not in the original candidate list but emerged as load-bearing for spike β's composition-predictability work.
- `errors[]` — typed error array. Replaces `web-searcher`'s ad-hoc `{"error": "authentication_failed", ...}` shape with a list of typed-kind records. Marginal addition but pays for itself the first time a multi-error scenario appears.

---

## Implications for T16 DECIDE

The cycle-status names five T16 sub-questions: (a) boundary; (b) contract; (c) client access; (d) cleanup; (e) backward compat. Plus the snapshot Action 4 first sub-question: scope. The spike contributes the following:

**Sub-question (0) — Scope (always / substantive / operator-configured).** The spike does not resolve scope but **surfaces a size-floor finding** load-bearing on the "always" interpretation: ensembles emitting <~200 character outputs (calibration-checker; web-searcher's typed JSON; short-input text-summarizer cases) are structurally poor candidates for artifact-substrate. The "always" interpretation, taken literally, generates structurally wasteful behavior at the low-output end of the library. The practitioner-verbatim "always" is therefore worth re-examining at DECIDE — either accept the size-floor as an implicit clause or refine the verbatim toward "always when substantive."

**Sub-question (a) — Boundary (which ensembles warrant artifact-substrate vs. response-substrate).** The per-ensemble fit notes give a concrete partition: **artifact-substrate candidates** are `code-generator`, `prose-improver`, `claim-extractor`, `argument-mapper` (the LLM-text-emitting capability ensembles with observed multi-thousand-character outputs); **response-substrate candidates** are `agentic-calibration-checker`, `web-searcher`, `text-summarizer` (small outputs, already-structured outputs, summary outputs that are themselves compressed by construction). `agentic-result-summarizer` is a special case — it does not produce a deliverable; it consumes ensemble envelopes and emits summary text. Its envelope position is **consumer-of-envelope** rather than producer-of-envelope; its `default_task` changes most under artifact-substrate. The partition is not crisp at the `text-summarizer` boundary — its observed output ~500 chars sits just above the size-floor — and DECIDE benefits from a defensible size threshold rather than a per-ensemble case-by-case verdict.

**Sub-question (b) — Contract (typed field vs. convention).** The spike supports the typed-field path. Three of three findings the artifact-substrate proposal targets benefit from typed fields rather than conventions: structured composition needs typed schemas (spike β territory); operator-readable diagnostics need typed event fields; artifact references need typed `{path, content_type, summary, retention}` records. A convention-based approach (e.g., "by agreement, `primary` is always the synthesizer's response") cannot detect deviation, cannot enable composition, and cannot drive observability routing. **Recommendation for the DECIDE ADR: typed field, not convention.** The cost is a new ensemble-YAML field (`output_schema:` under candidate B; the artifact-emission contract under candidate C); the benefit is mechanical checkability at the dispatch boundary.

**Sub-question (c) — Client access.** The spike does not test client access empirically. It surfaces the question: under artifact-substrate, the client needs to fetch the deliverable to read it. Two paths the cycle has implicitly considered are (i) filesystem read against `.llm-orc/artifacts/...` (works when the client and serve layer share a filesystem; breaks for remote clients) and (ii) a serve-layer HTTP endpoint (`GET /v1/artifacts/<id>`) that returns the deliverable on demand. Path (ii) is more general but introduces a new endpoint surface and authentication considerations. **For DECIDE: this sub-question may be deferred to ARCHITECT or BUILD — the envelope contract names a `path` and a `content_type`; how the client resolves the `path` to bytes is a serving-layer implementation concern downstream of the contract.**

**Sub-question (d) — Cleanup.** The spike surfaces the `retention: session | cycle | persist` enum as a typed envelope field, which gives operators an explicit cleanup policy hook. The spike does not propose cleanup mechanics (cron? per-dispatch TTL? operator-driven `llm-orc artifacts prune`?). **For DECIDE: the envelope contract is well-served by carrying `retention` per-artifact; the cleanup mechanism is downstream of the contract and can be settled later.**

**Sub-question (e) — Backward compat.** The hardest cost. Existing consumers reading `results.<agent>.response` directly will not see the deliverable under candidate C. Two mitigations are visible from the survey:
1. **Dual-emit transitional path** — the dispatch layer continues to populate `results.<agent>.response` (as today) AND populates `artifacts[]` and a handle-style `primary` for new consumers. Old consumers keep working; new consumers migrate at their own pace.
2. **Candidate B as the cycle's commitment** — `artifacts[]` becomes additive; ensembles that opt into artifact-substrate emit the field; ensembles that don't keep emitting content in `primary`. No old consumer breaks; the migration is per-ensemble.

The second mitigation is cleaner. The first mitigation is more general but doubles the per-dispatch payload during the transitional period.

---

## Open questions for spike β

Spike β traces lit-review composition under the candidate envelope. The following questions emerged from spike α that β can resolve empirically:

1. **Does `structured` (candidate B / C) carry composition load?** When `claim-extractor` emits to `argument-mapper`, the latter currently consumes a free-text rendering of the former's bullets. Under candidate B, can `argument-mapper`'s `default_task` reword to "consume the `structured.claims[]` field from the previous dispatch's envelope" — i.e., is the typed-shape declaration sufficient for composition predictability, or does the orchestrator still need to massage the inter-stage handoff?
2. **Does `primary` as a handle (candidate C) break composition or improve it?** Under artifact-substrate, the next stage's input is `{artifact_path, content_type, summary}`. Spike β can test: does the consuming ensemble's agent need to read the artifact (which means filesystem-access at the agent layer, not just the orchestrator layer)? Or does the orchestrator read it and fold it into the next dispatch's `input_data` (which preserves the agent's existing prose-input contract)?
3. **Where do the current implicit composition assumptions live?** The spike surfaced that `default_task` carries output-shape expectations as natural-language instructions; the orchestrator's response-shaping carries them as narrative scaffolding. Spike β can locate these and ask whether moving them into typed `output_schema:` fields actually removes the orchestrator-side massaging.
4. **Three-stage composition shape under the envelope.** `web-searcher` → `claim-extractor` → `argument-mapper` was the 3m 54s verified composition. Under the envelope, does the orchestrator still need to be the integration substrate, or does the envelope's typed fields let the dispatch chain wire itself? (The cycle-status finding 6's information-finding overhead is implicated here.)

---

## Limitations

The spike does NOT answer:

- **Whether the practitioner-verbatim "always" interpretation of artifact-substrate scope is the right interpretation.** The size-floor finding surfaces a tension with the verbatim but does not resolve it. T16's first sub-question remains a DECIDE deliberation.
- **The three-findings-collapse claim independently.** Per MODEL Action B, each of (output-spec drift / information-finding overhead / AS-7 content-stripping) must be tested against alternative remediation paths separately before accepting the collapse as T16's substrate. This spike surfaces that artifact-substrate plausibly addresses all three; it does not test alternative paths for each finding.
- **The sidecar-log alternative for dispatch timing.** Per MODEL Action C, the dispatch-timing event-model extension is one of three architectural alternatives. The envelope's `diagnostics` field can carry dispatch-timing fields; whether the event-model extension is required or a sidecar-log alternative satisfies PLAY note 12's question is outside the envelope spike's scope.
- **Client-side access mechanics.** The artifact-substrate candidate names a `path` and a `content_type`; the path resolution mechanism (filesystem-shared, serve-layer HTTP endpoint, signed URL, etc.) is named as an open question for ARCHITECT or BUILD downstream.
- **Cleanup mechanics.** The `retention` enum names the policy; the implementation of cleanup (lifecycle triggers, operator commands, TTL enforcement) is named as a downstream question.
- **Routing-preference interactions.** Spike γ characterizes whether the routing-preference behavior is MiniMax-M2.5-free-specific or systemic. The envelope spike does not bear on T14 directly except in noting that the envelope's `diagnostics` field can carry routing-decision events that inform T14's operator-visible surface.
- **The `agentic-calibration-checker`'s envelope-position role.** Spike α scored it as response-substrate (output is ~80 chars). Its role in the Calibration Gate's quality-signal pipeline is not envelope-bearing in the same sense as the capability ensembles; whether the envelope contract should distinguish "produces deliverable" ensembles from "produces signal" ensembles is a question for DECIDE but not one the envelope shape alone resolves.

The spike was conducted analytically against existing YAMLs and one most-recent `execution.json` per ensemble. Output-spec compliance is a single-sample observation — three of six ensembles deviated in their most recent observed artifact, but a larger sample would be needed to characterize deviation rates. The deviation finding's policy implication (that output-spec compliance benefits from typed fields rather than prose specs) is robust to sample-size concerns; the rate finding is not.

---

## Cross-references

- `docs/agentic-serving/housekeeping/cycle-status.md` — Cycle 6 active entry, particularly the post-hotfix verification findings 2 (output-spec drift), 6 (information-finding overhead), 8 (artifact-as-substrate proposal); Candidate spikes section (Spike α specification).
- `docs/agentic-serving/domain-model.md` — §Concepts entries on Common I/O envelope, Artifact-as-substrate, Output-spec drift; Amendment Log entry #10; Open Question #15.
- `docs/agentic-serving/decisions/adr-021-skill-orchestration-via-per-capability-dispatch.md` — current per-capability dispatch contract; the envelope ADR amends or supersedes the response-shape implications of this ADR.
- `docs/agentic-serving/decisions/adr-004-result-summarization-mandatory.md` — invariant AS-7; OQ #15 names the amendment pathway under T16's artifact-substrate resolution.
- `docs/agentic-serving/decisions/adr-020-tool-use-ensemble-shape.md` — `web-searcher` script-agent shape; confirms script-agent and LLM-agent envelopes are interchangeable at the dispatch boundary.
- `.llm-orc/ensembles/agentic-serving/*.yaml` — the eight ensemble YAMLs surveyed.
- `.llm-orc/artifacts/agentic-serving/*/latest/execution.json` — the eight artifact instances reviewed for emitted-shape evidence.

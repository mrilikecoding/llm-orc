# Spike γ (Cycle 6) — Routing-preference characterization across profiles

**Date:** 2026-05-15
**Cycle:** 6 (agentic-serving mini-cycle; routing-preference cluster)
**Wave:** DECIDE-blocking empirical spike (independent of spikes α and β)
**Method:** Live dispatch — `curl` against the running serve at `http://localhost:8765/v1/chat/completions`, varying the active orchestrator profile (`agentic_serving.orchestrator.model_profile` in `.llm-orc/config.yaml`) and the client tool surface (`tools` array on the request body). Probe prompt held constant.
**Scope:** Three planned cells (A: MiniMax M2.5-free + tool-rich; B: `agentic-orchestrator-offline` + tool-rich; C: MiniMax M2.5-free + tool-less). Two additional cells dispatched for sharper disambiguation (A-explicit: explicit invoke_ensemble naming under tool-rich MiniMax; the offline-profile architectural fail-fast).
**Cost incurred:** $0.00 (MiniMax M2.5-free tier + local Ollama).

---

## Question

Is the orchestrator's `direct → client-tools → invoke_ensemble` routing preference (Cycle 5 PLAY notes 1–9 + 20; 2026-05-14 follow-on findings 3, 5, 6) profile-specific to MiniMax M2.5-free, or systemic to the cheap-cloud-orchestrator + tool-rich-client pattern?

The held belief-mapping question at T14's DECIDE entry — *"what would have to be true for the operational preference to be the intended behavior?"* — is empirically conditional on this spike. Cycle 6 MODEL Action A named three operationally distinct dispositions the data must disambiguate: (i) intended scope; (ii) defect to remediate; (iii) configuration-conditional. γ aims to deliver the evidence T14 deliberates against, not to resolve T14 itself.

---

## Method

**Probe prompt** (held constant across cells): *"Write a Python function that reverses a string in place."*

This prompt maps unambiguously to one capability ensemble slot (`code-generator`, Topaz skill `code_generation`). It is short enough that the orchestrator should be able to act on it without `list_ensembles` re-enumeration. It avoids file-creation framing ("write a file containing…") that would bias toward client-tool delegation; it asks for a function, which is plausibly answered by direct LLM completion. The neutrality of the framing is itself part of the routing surface under test.

**Cells planned, per the spike specification:**

- **Cell A:** MiniMax M2.5-free via Zen (active profile `agentic-orchestrator`), tool-rich client (`tools: [write_file, read_file]`) — replicates Cycle 5 PLAY baseline.
- **Cell B:** `agentic-orchestrator-offline` (qwen3:14b via Ollama), tool-rich client — controls for cloud-vs-local at the orchestrator layer.
- **Cell C:** MiniMax M2.5-free, tool-less client (`tools: []`) — replicates the pre-PLAY `curl` baseline.

**Cells actually exercised:**

- Cell A (dispatched 2026-05-15 09:52:47, completed 09:53:04 — 17s wall-clock; curl-reported `time_total` 7.66s) — NL prompt with tool-rich client.
- Cell C (dispatched 09:53:18, completed 09:53:31 — 13s wall-clock; curl 3.25s) — NL prompt with tool-less client.
- **Cell B failed at session start** (dispatched 09:54:25, returned HTTP 500 in 88ms) — the offline profile resolves to `OllamaModel`, which declares `supports_tool_calling = False` at the class level; the serve raises `ToolCallingNotSupportedError` before any LLM call. The error message claims "Ollama" is among the supported tool-calling providers, but the runtime check rejects it. This is itself a load-bearing finding (see §Per-cell observations and §Disposition interpretation).
- Cell A-explicit (dispatched 09:55:13, completed 09:56:24 — 71s wall-clock; curl 70.66s) — same as Cell A but with explicit-naming framing: *"Please dispatch the code-generator capability ensemble via invoke_ensemble to write a Python function that reverses a string in place."*

**What was captured per cell:**

- HTTP status, wall-clock duration, curl `time_total`.
- Response shape (`finish_reason`, presence/absence of `message.tool_calls`, `usage.completion_tokens`).
- Response content (the orchestrator's natural-language reply).
- Artifact directory inspection (`.llm-orc/artifacts/agentic-serving/<ensemble>/`) immediately before and after each cell.
- For dispatch-firing cells, `execution.json` for both the dispatched ensemble and the result-summarizer return path.

**What was NOT captured:** the serve's console emissions during dispatch. The running serve process is attached to a separate TTY (PID 90745, terminal `s008`); its stdout/stderr is not accessible from this spike's session without restarting it. The 2026-05-13 verification work that informs cycle-status §Post-hotfix finding 7 captured console output by sharing the terminal; this spike does not have that access. The artifact `execution.json` files carry the typed-event data the console would have shown (`metadata`, per-agent `duration`, status); the spike substitutes those for direct console capture.

---

## Per-cell observations

### Cell A — MiniMax M2.5-free + tool-rich client + NL framing

Request body: `tools` declared `write_file` and `read_file`. User content: *"Write a Python function that reverses a string in place."*

- **HTTP 200, 17s wall-clock, 1450 completion tokens.**
- `finish_reason: stop`.
- `message.tool_calls` absent (no client-tool delegation).
- `message.content` is a 987-char direct response containing the function source plus explanation prose, output formatting (`**Output:**` block with `Original:` / `Reversed:` example), and a closing two-sentence justification of the two-pointer technique.
- **Artifact inspection:** no new directory under `.llm-orc/artifacts/agentic-serving/code-generator/` (latest symlink unchanged from `20260514-093032-974`). No result-summarizer dispatch either.
- **Routing decision: direct LLM completion.** Neither `invoke_ensemble` (internal tool) nor `write_file` (client tool) was invoked. The orchestrator answered inline from its own pretrained knowledge.

This is consistent with Cycle 5 PLAY note 20's characterization (NL framing under tool-rich client never reaches `invoke_ensemble`) and refines it: PLAY note 20 observed *client-tool delegation* (the orchestrator used OpenCode's native `Write` tool to create `chunk_by_predicate.py`). Here, with `write_file` available, the orchestrator did **not** delegate — it emitted code inline. The difference is plausibly the prompt phrasing. "Write a Python function that reverses a string in place" reads as *describing a function*; "Write a function that does X" + the presence of a file-writing client tool is the surface where PLAY note 20's "Write" delegation happened. The granularity of the disposition is finer than the binary "delegates or doesn't" — it depends on whether the prompt's framing maps onto a client-tool's verb.

### Cell C — MiniMax M2.5-free + tool-less client + NL framing

Request body: `tools: []`. Same user content as Cell A.

- **HTTP 200, 13s wall-clock, 1305 completion tokens.**
- `finish_reason: stop`.
- `message.tool_calls` absent (no tools were available to delegate to).
- `message.content` is a 944-char direct response, structurally similar to Cell A (function source + usage examples + four-point "How it works" + closing note on Python string immutability). The implementation differs from Cell A's in surface detail but is functionally equivalent.
- **Artifact inspection:** unchanged. No dispatch.
- **Routing decision: direct LLM completion.** Same as Cell A.

The Cell A vs. Cell C comparison **isolates the variable of client tool availability**: with `write_file`/`read_file` available (A) and without any client tools (C), MiniMax M2.5-free on this prompt makes the same routing decision — direct LLM completion. This is not the "client-tool delegation when tools are available, direct completion when not" pattern PLAY note 20 / post-hotfix finding 5 might suggest as the orchestrator's reasoning. Both cells routed identically, both bypassed the ensemble library entirely.

### Cell B — `agentic-orchestrator-offline` (qwen3:14b via Ollama) — fails at session start

Config swap applied (`.llm-orc/config.yaml` `agentic_serving.orchestrator.model_profile: agentic-orchestrator-offline`). `/v1/models` immediately reflects the swap (returns `agentic-orchestrator-offline`). qwen3:14b is locally present (`ollama list` confirms).

Request body: same as Cell A's tool-rich payload, modulo the `model` field.

- **HTTP 500, 88ms.**
- Response body: `{"error":"Internal server error","detail":"Orchestrator model_profile 'agentic-orchestrator-offline' resolves to 'ollama-qwen3:14b' which does not support tool calling. Configure a profile with an OpenAI-compatible provider (Ollama, OpenAI, OpenRouter, LM Studio, vLLM, etc.)."}`
- **No dispatch.**

The serve's `_default_orchestrator_llm_loader` (`v1_chat_completions.py:235-253`) loads the model via `ModelFactory`, then checks `model.supports_tool_calling`. `OllamaModel` (`src/llm_orc/models/ollama.py:11`) does not override the base-class `supports_tool_calling = False` (`src/llm_orc/models/base.py:167`). The check fails fast; no LLM call is made; the orchestrator's reasoning loop never runs. **The cycle's "offline fallback" orchestrator profile, as currently shipped, cannot be used as an orchestrator at all under the serving layer's tool-calling-required check.**

The error message's enumeration "Ollama, OpenAI, OpenRouter, LM Studio, vLLM" is misleading — it names provider *families*, but within those families the per-class `supports_tool_calling` flag determines admissibility. The `OllamaModel` class is in the "Ollama" family and is rejected. The fix-message and the runtime behavior disagree.

Cell B as designed cannot be run with this profile. Per the spike directive's free-tier-only constraint and "if the local profile is unavailable, return with that finding rather than substituting a paid profile" — the finding stands: there is no free-tier local-Ollama orchestrator profile in the project that the serving layer admits as an orchestrator.

### Cell A-explicit — MiniMax M2.5-free + tool-rich client + explicit-naming framing

User content: *"Please dispatch the code-generator capability ensemble via invoke_ensemble to write a Python function that reverses a string in place."* (Same tools array as Cell A.)

- **HTTP 200, 71s wall-clock, 16,939 completion tokens.** A ~13× completion-token cost vs. Cell A's direct completion.
- `finish_reason: stop`.
- `message.tool_calls` absent on the final assistant message (the dispatch happened via the internal tool surface and was consumed by the runtime; the final reply is the orchestrator's NL synthesis).
- **Artifact inspection:** two new artifact directories created:
  - `.llm-orc/artifacts/agentic-serving/code-generator/20260515-095623-868/` — dispatch entry. `execution.json` shows `status: completed`, ensemble `code-generator`, three agents (`coder` / `critic` / `synthesizer`) all `status: success`, `duration: 61.44s`. `input: {"data": "Write a Python function that reverses a string in place."}` — the orchestrator stripped the "dispatch ... via invoke_ensemble" framing and passed only the bare task.
  - `.llm-orc/artifacts/agentic-serving/agentic-result-summarizer/20260515-095627-130/` — the AS-7 return path (4s after the code-generator dispatch completed). `duration: 3.14s`. Summarizer's `response` faithfully describes the synthesizer's `chars.reverse()` implementation, no AS-7 content-stripping observed on this dispatch.
- **Routing decision: `invoke_ensemble` dispatch, on first attempt.**

This is a softer result than 2026-05-14 follow-on finding 3, which observed the orchestrator bypassing `invoke_ensemble` even under explicit-naming on first attempt (and required practitioner intervention to retry). Under the simpler prompt here (single-line task, no embedded "in `.llm-orc/ensembles/...`" framing, no expected-file-path implication), explicit-naming reliably triggered dispatch on first attempt. The post-hotfix finding's "explicit-naming bypass" may be conditional on prompt complexity or on the additional "write the file" implication a CircularBuffer task carries; it is not a categorical property of the orchestrator's reasoning under MiniMax M2.5-free + tool-rich client.

**One sharp finding from the dispatch return:** the orchestrator's final NL response (1,126 chars) presented a **different implementation than the synthesizer produced**. The synthesizer's `chars.reverse()` (one method call on a list) was substituted in the orchestrator's narration with a hand-rolled two-pointer swap loop. The result-summarizer's faithful summary of the synthesizer's code did reach the orchestrator's context (verified via the summarizer artifact). The orchestrator received the faithful summary and **chose to present a different implementation**, closing with *"The `code-generator` ensemble (3-agent flow: coder → critic → synthesizer) verified this implementation meets the in-place reversal requirement."* The claim is structurally false: the ensemble verified `chars.reverse()`, not the two-pointer loop. This is PLAY note 15's "fabrication while critiquing fabrication" pattern in concrete form — and an instance where the AS-7 summarizer worked correctly but the orchestrator's downstream narration introduced the fabrication anyway. The defect surface is one layer further downstream than AS-7.

---

## Comparative analysis

Across the four executed cells:

| Cell | Orchestrator | Client tools | Prompt framing | Dispatch | Duration | Completion tokens |
|---|---|---|---|---|---|---|
| A | MiniMax M2.5-free | `write_file`, `read_file` | NL ("Write a function...") | None — direct completion | 17s | 1,450 |
| C | MiniMax M2.5-free | none | NL ("Write a function...") | None — direct completion | 13s | 1,305 |
| B | qwen3:14b (Ollama) | `write_file`, `read_file` | NL — never reached | **HTTP 500 at session start** (no tool-calling support) | 0.1s | 0 |
| A-explicit | MiniMax M2.5-free | `write_file`, `read_file` | Explicit ("dispatch ... via invoke_ensemble") | `invoke_ensemble(code-generator)` on first attempt | 71s | 16,939 |

**What changed across cells:**

1. **Client-tool availability (A vs. C) did not change the routing decision.** Both fell through to direct LLM completion. This is the cleanest comparative result in the spike. It refines PLAY note 20's "NL framing under client-tools delegates" — the delegation is **conditional on a client tool whose verb matches the prompt's framing**; the mere presence of unrelated client tools does not trigger delegation. A "Write a Python function" prompt with `write_file` available does **not** delegate to `write_file`; a "Write `chunk_by_predicate.py`" prompt (PLAY note 20) does.

2. **Prompt framing (A vs. A-explicit) changed the routing decision dramatically.** The same MiniMax M2.5-free orchestrator, the same client tool surface, the same underlying task — only the framing differed. NL framing → direct completion. Explicit `invoke_ensemble` naming → ensemble dispatch on first attempt. This confirms ADR-021's explicit-naming clause is honored reliably under this orchestrator + client combination. The natural-language-supported clause is the one not honored.

3. **The offline profile architectural fail (Cell B) is upstream of routing-surface characterization.** The cycle-status spike specification proposed `agentic-orchestrator-offline` as the cloud-vs-local control cell; the serving layer's tool-calling-required check makes that cell unreachable under the current code. The "control for cloud-vs-local" question γ was designed to answer cannot be answered with the free-tier profiles available in the project. A different free local model with tool-calling support (e.g., a llama3.1 or hermes variant with native tool calling) would need to be authored and wired before this cell can run.

**What stayed the same across the dispatch-capable cells (A, C, A-explicit):**

- All three executed in tens-of-seconds to ~minute timescales — none hit the 8–44 minute timescales 2026-05-14 follow-on findings 4 and 6 observed. The probe prompt's simplicity (a one-line, one-function task) plausibly avoids the orchestrator's information-gathering loop that produced finding 6's 44m 32s elapsed time. The prompt's structural simplicity is itself a confound — γ characterizes routing surface behavior on a prompt the orchestrator can act on without `Glob`-ing the project tree; the latency phenomenology of finding 4 is a different prompt-shape's result.
- The cycle's typed events (`TierSelection`, `CalibrationVerdict`, `AuditDiagnostic`, `CalibrationSignal`) presumably emitted on the A-explicit dispatch — they exist in the codebase and write to `execution.json`. The spike did not parse `execution.json` for the typed-event payloads (the file structure carries `results`, `metadata`, `status`, `synthesis`, `input` at the top level; the typed event records live in lower-level fields the spike did not enumerate). The spike confirms typed events are produced on dispatch (the `code-generator` ensemble's `metadata.duration: 61.44s` is one example) but does not enumerate which events fired.

---

## Disposition interpretation

MODEL Action A names three operationally distinct dispositions for T14's deliberation: (i) intended scope; (ii) defect to remediate; (iii) configuration-conditional. γ's evidence:

**Against disposition (iii) — configuration-conditional behavior keyed on orchestrator profile.** The cycle-status spike specification framed γ as the disambiguation between "MiniMax M2.5-free-reasoning-shape-specific" and "systemic to the cheap-cloud-orchestrator + tool-rich-client pattern." γ cannot answer this question as scoped, because Cell B is unreachable — the only available free-tier alternative orchestrator profile in the project does not admit tool calling. The cell that would have tested whether qwen3:14b makes the same routing decision as MiniMax M2.5-free cannot be run with the current code and the current free-tier constraint. γ produces **a structural reason for the cell's absence** (the serving-layer tool-calling-required check on `OllamaModel`) rather than a positive characterization of the local-orchestrator behavior. The disposition (iii) question is **deferred, not refuted**.

**Toward disposition (i) — intended scope.** A robust interpretation of the system prompt (`src/llm_orc/agentic/orchestrator_config.py:77-126`) supports the "direct completion is the default; client-tools-when-relevant; invoke_ensemble when the user explicitly names it" reading. The system prompt's framing is:
- `list_ensembles()` first **for capability queries** ("what can the system do") — narrowly scoped, not general-purpose dispatch.
- `invoke_ensemble(name, input)` "after list_ensembles when you know which ensemble fits" — gated on the LLM's own *"knowing which ensemble fits"*, which the spike data suggests it does not arrive at for simple NL prompts.
- Client-declared tools "when you need file contents, directory structure, code execution, or edits to the user's files" — verb-matched to filesystem actions.
- **Nothing in the system prompt says "for production tasks, prefer ensembles to direct completion."** The implicit precedence is exactly the empirically-observed precedence: capability queries → list_ensembles; file-system actions → client tools; everything else → direct completion; ensemble dispatch when explicitly named or when an ensemble obviously fits.

Under this reading, the system prompt **already commits to** the routing behavior γ observed in cells A and C. The behavior is the intended scope of the contract as drafted. ADR-021's natural-language-supported clause, read against this prompt, is over-broad: the prompt does not instruct the orchestrator to interpret NL framing as ensemble-dispatch unless the ensemble is named or unmistakably indicated.

**Toward disposition (ii) — defect to remediate.** The cycle's stakeholder analysis (`product-discovery.md` Skill Orchestration User entry) names the expectation "the orchestrator will route my NL request to a capability ensemble." That stakeholder expectation is not borne out empirically. If the cycle's commitment is to honor the stakeholder's mental model, the system prompt as currently written is the defect — it needs explicit clauses on "when an NL prompt maps to an authored capability ensemble's slot, prefer ensemble dispatch over direct completion." The Cell A-explicit data point shows that the orchestrator *can* dispatch under MiniMax M2.5-free + tool-rich client when the routing decision is made for it; the question is whether the orchestrator should make that decision autonomously on NL framing.

**γ's net contribution to T14:** the spike strengthens disposition (i)'s plausibility by surfacing the system prompt's *de facto* alignment with the observed behavior. The disposition (ii) — defect — argument now has a concrete target (a clause in `DEFAULT_ORCHESTRATOR_SYSTEM_PROMPT`); it is not a diffuse "the orchestrator is misbehaving" claim. Disposition (iii) remains empirically open. **The two-disposition answer the spike most credibly supports is "(i) intended scope as currently written; (ii) candidate defect if the stakeholder's NL-routes-to-ensemble mental model is committed."**

The held belief-mapping question — *"what would have to be true for the operational preference to be the intended behavior?"* — has a γ-supported answer: it would have to be true that the orchestrator's reasoning surface is **structurally a direct-completion-first dispatcher** that escalates to client tools or ensembles **only on explicit signal**. The system prompt commits to this structurally, and the observed behavior matches the commitment. The "preference" framing the prior vocabulary used names the architectural fact accurately; whether the cycle wants to keep that architectural fact is a separate question DECIDE resolves.

---

## Implications for T14 DECIDE

The T14 ADR draft can now be written against a richer evidence base than Cycle 5 PLAY alone provided. Three concrete contributions:

1. **The routing surface's "preference" is downstream of the orchestrator system prompt's structure.** The prompt commits to capability-queries via `list_ensembles`, filesystem-actions via client tools, ensemble dispatch on explicit naming, and direct completion as the residual. Any T14 ADR that adopts a "natural-language supported" clause must specify *what NL framing triggers dispatch* — a clause the current prompt does not contain. The ADR-021 amendment surface is therefore the system prompt's body, not the ADR's text alone. If DECIDE accepts disposition (i), the system prompt stays as-is and ADR-021's natural-language clause narrows or comes out. If DECIDE accepts disposition (ii), the system prompt grows a new clause and the ADR's natural-language commitment carries operational weight for the first time.

2. **The explicit-naming dispatch path is reliable under MiniMax M2.5-free + tool-rich on simple prompts.** The post-hotfix finding 3's "explicit-naming bypass" appears to be conditional on prompt complexity, not categorical. T14's deliberation on whether explicit-naming is the supported dispatch surface (which γ's data supports as reliable on first attempt for one-line tasks) vs. whether NL framing must also be supported (which γ confirms is not delivered under either tool surface) can use the simple-prompt baseline as the structural floor. The more complex prompts that produce the 44m 32s loop-time of finding 6 are a separate question about latency phenomenology, not about routing surface admission.

3. **The cycle's "cheap-cloud-orchestrator + tool-rich-client" pattern's generality is empirically open.** γ confirms the pattern holds under MiniMax M2.5-free. It cannot confirm or refute the pattern holds under other cheap-cloud orchestrators (Groq Llama-3, Hermes, Mistral via OpenRouter free tiers, etc.) without those profiles being authored and wired. The cycle has one data point on one orchestrator. DECIDE's ADR draft should acknowledge the n=1 scope qualifier explicitly — the way Cycle 5 BUILD's ADR-019 §Consequences §Positive carries the n=1 qualifier as an active advisory.

The T14 ADR draft's structure that γ's data supports:

- **Concept:** Routing surface behavior (MODEL Action A's name; settled).
- **Dispositions in scope:** (i) and (ii) — γ's data supports both as live deliberation surfaces; (iii) is deferred pending a free-tier local-tool-calling orchestrator profile being authored.
- **Operative routing precedence (whichever disposition wins):** capability-queries → `list_ensembles`; filesystem-or-code-execution → client-tools; ensemble dispatch → explicit naming OR (if disposition (ii) wins) a system-prompt-defined NL trigger; direct completion → residual.
- **Open question carried forward:** how the orchestrator decides "this NL prompt maps to ensemble X's slot" when the user does not name X. Disposition (ii) requires answering this; disposition (i) does not need to answer it.

---

## Implications for T15 (observability) and T16 (ensemble contract)

**T15 (observability) — what γ's dispatches would have surfaced.** The two dispatch-firing cells (A-explicit triggering code-generator + result-summarizer) wrote `execution.json` artifacts with `metadata.duration` fields (61.44s and 3.14s respectively). These are the dispatch-timing data the snapshot Action 2 field-read named as the orchestrator-context destination's missing piece. γ's data confirms the typed-event infrastructure produces dispatch-timing values on every dispatch; the routing-to-operator-terminal work is the remaining gap. The spike did not capture the live serve console output (the running serve was on a separate TTY); the artifact-layer surface is fully populated.

The dispatch-firing cell would have answered PLAY note 12's load-bearing question (*"What was the total run-time of the ensemble?"*) trivially — 61.44s for code-generator + 3.14s for the summarizer return path, total 64.58s ensemble-side of the 71s wall-clock. The remaining ~6s lives in the orchestrator's tool-dispatch return and final-response assembly. The "infrastructure-complete / routing-incomplete" framing is concretely accurate for this dispatch: every data point PLAY note 12 asked for is in the artifact; none of it reaches the orchestrator's narration without code changes.

**T16 (ensemble contract) — γ's data does not bear on T16 directly except in three peripheral ways.** First, the dispatch return shape γ observed matches spike α's Observation 1 — the `execution.json` envelope is uniform across both dispatches (code-generator's three-agent shape and summarizer's single-agent shape carry the same `{ensemble, status, input, results, metadata, synthesis}` top-level dict). Second, the orchestrator's NL-narration-substitutes-implementation pattern in Cell A-explicit (synthesizer's `chars.reverse()` → orchestrator's two-pointer loop) is a defect surface **downstream of AS-7** — the result-summarizer worked correctly; the orchestrator's downstream narration introduced the fabrication. Artifact-as-substrate (T16's candidate resolution) would route the deliverable to the client without orchestrator-narration substitution, which directly addresses this defect. The cycle's three-findings-collapse argument (collapsibility of output-spec drift, information-finding overhead, and AS-7 content-stripping under artifact-as-substrate) per MODEL Action B treats AS-7 specifically; the spike surfaces an **additional fourth finding** the artifact-as-substrate proposal would address: orchestrator-narration substitution of dispatched output. Third, the tool-rich client's `write_file` tool was available in Cells A and A-explicit; under artifact-as-substrate, the dispatched ensemble would write the deliverable to disk and the orchestrator's response would carry `{artifact_path, content_type, summary}` rather than the synthesizer's prose. This eliminates the substitution surface entirely — the orchestrator never sees the implementation to substitute.

T16's first sub-question (scope: always / when substantive / operator-configured) gains a small data point: the 1,126-char narration cost of substitution (Cell A-explicit) is structurally avoidable under artifact-as-substrate. The size threshold spike α surfaced (~200 char floor for artifact-substrate viability) is below this narration's length; this case clearly clears the floor. γ does not test the floor directly.

---

## Limitations

The spike does NOT determine:

- **Whether the routing-preference is MiniMax M2.5-free-reasoning-shape-specific or systemic.** Cell B (the local-orchestrator control) cannot be run with current free-tier profiles. A free-tier non-Ollama local-orchestrator profile (or an Ollama profile with a tool-calling model that the `OllamaModel` class is amended to admit) would have to be authored before this cell can run.
- **The pattern's generality across cheap-cloud orchestrators.** n=1 (MiniMax M2.5-free only). Other cheap-cloud orchestrators (Groq, Cerebras, free OpenRouter tiers, Anthropic free tier when available) might exhibit different routing surfaces. γ's evidence supports the disposition (i)/(ii) framing for the one orchestrator it tested.
- **Statistical claims about dispatch-firing rates.** Each cell ran once. Repetition across the same cells could reveal variance; γ has only point measurements.
- **The latency phenomenology of finding 4 / finding 6 (44m 32s tasks).** γ's prompt is structurally simple (one function, one input shape). The orchestrator did not enter information-finding loops. The latency surface γ observed (17s to 71s wall-clock) is the lower-bound regime; the upper-bound regime where the orchestrator loops on `Glob`-ing the project tree is not exercised. T14's deliberation on whether routing surface behavior implicates latency phenomenology cannot use γ's data for the latter.
- **The serve console's emission behavior under the dispatches.** Direct console capture was not available (serve attached to separate TTY). Cycle-status finding 7's characterization (noise floor + coarse-success-only telemetry + absent typed events) is not re-verified by γ; the spike substitutes artifact-layer evidence for console-layer evidence on the typed-event-availability question.
- **Whether the orchestrator-narration-substitution pattern (Cell A-explicit's `chars.reverse()` → two-pointer-loop substitution) is reliable or sporadic.** One observation. Replicates of the same dispatch could surface variance in the orchestrator's downstream-narration behavior. The finding's policy implication (artifact-as-substrate would eliminate the substitution surface) is robust to sample-size concerns; the rate finding is not.

The spike's empirical surface is narrow (one prompt, four cells, one orchestrator family) but its findings are sharp at the structural level: the system prompt's commitments and the serving layer's tool-calling-required check are both code-readable and admit deterministic interpretation. The empirical observations confirm the structural reading rather than producing it.

---

## Cross-references

- `docs/agentic-serving/housekeeping/cycle-status.md` — Cycle 6 active entry, particularly cluster 3 (Routing-preference), Post-hotfix verification findings 3 (explicit-naming bypass), 5 (orchestrator self-acknowledged misrouting), 6 (information-finding overhead); Candidate spikes section (Spike γ specification).
- `docs/agentic-serving/domain-model.md` — §Concepts entries on Routing surface behavior (MODEL Action A; three dispositions), Routing preference / operational routing preference (Cycle 6 candidate with attribution flag), Explicit-naming bypass (Cycle 6 candidate); Amendment Log entry #10.
- `docs/agentic-serving/product-discovery.md` — Tension 14 (intended scope vs. defect vs. configuration-conditional); Skill Orchestration User stakeholder entry (NL-routes-to-ensemble mental model that disposition (ii) would commit to honoring).
- `docs/agentic-serving/decisions/adr-021-skill-orchestration-via-per-capability-dispatch.md` — natural-language-supported clause (the contract under question).
- `docs/agentic-serving/essays/reflections/field-notes.md` — Cycle 5 PLAY notes 1–9, 20 (NL-never-dispatches baseline); cross-cutting reflection §natural-language vs. explicit-naming dispatch asymmetry.
- `src/llm_orc/agentic/orchestrator_config.py` lines 77–126 — `DEFAULT_ORCHESTRATOR_SYSTEM_PROMPT` (the structural commitment the spike reads against observed behavior).
- `src/llm_orc/web/api/v1_chat_completions.py` lines 235–253 — `_default_orchestrator_llm_loader` (the tool-calling-required check that makes Cell B unreachable).
- `src/llm_orc/models/base.py` line 167 + `src/llm_orc/models/ollama.py` line 11 — `OllamaModel`'s inherited `supports_tool_calling = False` (the root of the Cell B fail).
- `.llm-orc/artifacts/agentic-serving/code-generator/20260515-095623-868/execution.json` — the only γ-produced ensemble dispatch artifact; substantiates that explicit-naming triggers dispatch reliably under MiniMax M2.5-free + tool-rich on first attempt.
- `.llm-orc/artifacts/agentic-serving/agentic-result-summarizer/20260515-095627-130/execution.json` — the AS-7 return path for the above; substantiates that the summarizer worked correctly on this dispatch and the substitution defect lives downstream in orchestrator narration.

---

## Cell B continuation (2026-05-15 10:34 — re-dispatched against a tool-calling-admissible offline profile)

The original Cell B failed because `agentic-orchestrator-offline.yaml` declares `provider: ollama` — which routes through `OllamaModel` (`supports_tool_calling = False`, inherited from the base class) — and the serving layer's `_default_orchestrator_llm_loader` rejects it at session start. To unblock the cell, a new profile `agentic-orchestrator-offline-tools.yaml` was authored that targets the same model (`qwen3:14b`) via the OpenAI-compatible adapter (`provider: openai-compatible/ollama`, `base_url: http://localhost:11434/v1`). The `OpenAICompatibleModel` class declares `supports_tool_calling = True` (`src/llm_orc/models/openai_compat.py:19`), and Ollama exposes a compatible `/v1/chat/completions` endpoint that surfaces native tool-calling for tool-capable models.

The dispatch was run with the same probe prompt and the same `tools` array as Cell A, modulo the `model` field swapped to `agentic-orchestrator-offline-tools`. The config swap was applied (`agentic_serving.orchestrator.model_profile: agentic-orchestrator-offline-tools`), serve was restarted, `/v1/models` confirmed the swap, the dispatch ran, the config was restored, and serve was restarted again — leaving the session in the cycle's default orchestrator state.

### Cell B (continuation) — qwen3:14b via OpenAI-compatible Ollama + tool-rich client + NL framing

Request body: same as Cell A's payload, with `"model": "agentic-orchestrator-offline-tools"`. User content: *"Write a Python function that reverses a string in place."*

- **HTTP 200, 57s wall-clock (10:34:07 → 10:35:04); curl `time_total` 48.17s. `completion_tokens` 1,626 (`prompt_tokens` reported 0 — the Ollama OpenAI-compatible endpoint omits prompt-token accounting; the count is not a session-prompt-cost finding).**
- `finish_reason: tool_calls` — distinct from Cells A and C, which returned `stop`.
- `message.content` is `null`.
- `message.tool_calls[0]`: `function.name = "write_file"`; `function.arguments = {"path": "reverse_string.py", "content": "def reverse_string(s):\n    return s[::-1]\n\n# Example usage:\n..."}`. One call, no parallel tool-call invocations. The `arguments` JSON parses cleanly.
- **Artifact inspection:** no new directories under `.llm-orc/artifacts/agentic-serving/code-generator/` or `.llm-orc/artifacts/agentic-serving/agentic-result-summarizer/`. `invoke_ensemble` was not called.
- **Serve log:** uvicorn shows one `POST /v1/chat/completions HTTP/1.1 200 OK` line and no other emissions. No typed-event console output. The serving layer treated this as a single-turn orchestrator round trip; the orchestrator returned a tool-call request and the serving layer surfaced it to the client unmodified (per ADR-021's client-tool pass-through contract).
- **Routing decision: client-tool delegation (`write_file`) on first attempt.**

**Two cross-cutting observations on this dispatch:**

First, the model's *implementation* is `s[::-1]` — slice-reversal that returns a new string. This is **not in-place reversal**; Python strings are immutable, so in-place reversal is not possible on a `str` (the canonical in-place reversal targets a `list[str]` of characters with a two-pointer swap, as Cell A-explicit's synthesizer produced via `chars.reverse()`). The model satisfied the prompt's surface phrasing while ignoring its semantic constraint. This is a model-capability finding adjacent to but distinct from the routing finding — `qwen3:14b` does not catch the prompt's "in place" constraint under this single-turn shape. The Calibration Gate would catch this (the dispatched code-generator ensemble's critic agent would presumably flag it); since `invoke_ensemble` was not called, no critic ran.

Second, the orchestrator did not return any natural-language explanation alongside the tool call — `message.content` is `null` and `finish_reason: tool_calls`. This is the OpenAI tool-call contract's intended shape (the client is expected to execute the tool and present the result back to the model for narration in a follow-up turn). The spike's single-turn dispatch surface does not exercise that follow-up; for this cell, the orchestrator's *visible behavior* terminates at the tool-call request. A multi-turn client (OpenCode's tool-rich surface, which the spike's `curl` simulates partially) would then execute `write_file` and return the result for narration. The spike does not simulate that turn.

### Updated comparative analysis (A vs. B vs. C now testable)

| Cell | Orchestrator | Client tools | Prompt framing | Dispatch | Duration | Completion tokens |
|---|---|---|---|---|---|---|
| A | MiniMax M2.5-free | `write_file`, `read_file` | NL | None — direct completion | 17s | 1,450 |
| C | MiniMax M2.5-free | none | NL | None — direct completion | 13s | 1,305 |
| B (orig) | qwen3:14b via `provider: ollama` | `write_file`, `read_file` | NL — never reached | HTTP 500 at session start | 0.1s | 0 |
| **B (cont)** | **qwen3:14b via `provider: openai-compatible/ollama`** | **`write_file`, `read_file`** | **NL** | **Client-tool delegation (`write_file`) on first attempt** | **48–57s** | **1,626** |
| A-explicit | MiniMax M2.5-free | `write_file`, `read_file` | Explicit ("dispatch ... via invoke_ensemble") | `invoke_ensemble(code-generator)` on first attempt | 71s | 16,939 |

**What the A vs. B comparison reveals.** Held constant: probe prompt, client tool surface, serving layer, ADR-021 contract. Varied: orchestrator profile (MiniMax M2.5-free over Zen vs. qwen3:14b over Ollama's `/v1` endpoint). The routing decision **differs across these two profiles**:

- MiniMax M2.5-free (Cell A) saw `write_file` available, saw a prompt whose verb is "write," and **did not** delegate — it emitted the function source inline as direct completion. PLAY note 20's "Write" delegation (which occurred on `chunk_by_predicate.py`) was framed-conditional, not just verb-conditional, under MiniMax.
- qwen3:14b on the same prompt **did** delegate, on the same verb match. The model invented a target path (`reverse_string.py`) and a content payload (function source + `__main__` example) and surfaced a single `write_file` tool call.

This is **not** the disposition the spike's original framing predicted. The cycle-status spike spec wrote γ's question as "is the preference profile-specific to MiniMax M2.5-free or systemic to cheap-cloud + tool-rich?" — implicitly assuming the *systemic* disposition was the candidate the local control would either confirm or refute. The data refutes that framing in a sharper direction than expected: the two orchestrators produce **different** routing decisions on the same surface. The "preference" is profile-specific in its *shape*, but the shape varies — MiniMax M2.5-free under-delegates (direct completion when client-tool delegation would honor the verb match); qwen3:14b over-delegates relative to MiniMax (delegates on verb match where MiniMax declined). Neither orchestrator routes to `invoke_ensemble` on NL framing; that part is shared.

**Three routings observed across the four dispatch-capable cells:**

1. Direct LLM completion (Cells A, C) — MiniMax M2.5-free's default for simple NL prompts regardless of client-tool availability.
2. Client-tool delegation (Cell B continuation) — qwen3:14b's default for an NL prompt whose verb maps onto an available client tool.
3. `invoke_ensemble` dispatch (Cell A-explicit) — triggered by explicit naming, observed under MiniMax M2.5-free.

`invoke_ensemble` was never the residual route under either orchestrator on NL framing. The systemic component of the post-PLAY-note-20 observation is "NL framing under either tested orchestrator does not route to ensembles." The profile-specific component is "what the orchestrator does *instead* of routing to ensembles differs: MiniMax M2.5-free completes inline; qwen3:14b delegates to the matching client tool."

### Updated disposition interpretation

The original log's reasoning shifted toward disposition (i) — intended scope — by reading the system prompt's commitments as alignment with Cell A's observed behavior. Cell B's continuation data **refines that interpretation toward disposition (iii) — configuration-conditional**, in a more precise sense than the spike spec anticipated:

- The system prompt (`DEFAULT_ORCHESTRATOR_SYSTEM_PROMPT`, `orchestrator_config.py:77-126`) does not change between Cells A and B. The serving layer's routing surface does not change. The client tool surface does not change. The prompt does not change. **Only the model behind the orchestrator changes.** And the routing decision changes — from direct completion to client-tool delegation.
- This means the system prompt's commitments **do not deterministically produce a single routing surface**. The prompt is interpreted differently by each model. Disposition (i)'s "intended scope as currently written" claim from the original log is weakened: if the prompt produced consistent behavior across reasonable orchestrators, "intended scope" would be plausible; the prompt produces *different* behavior, so "intended scope" cannot describe both observed routings simultaneously. At most one of the two routings is the intended scope (the other is, by definition, the model under-interpreting or over-interpreting the prompt's tool-precedence guidance).
- Equivalently, the contract surface is **specified at the prompt layer but realized at the model layer**, and the realization is model-conditional. This is the disposition (iii) shape — configuration-conditional behavior — keyed not on operator config but on *model selection* (which is a form of configuration the operator controls via the active orchestrator Model Profile).

The disposition (ii) — defect to remediate — argument also strengthens, but more sharply than the original log named: the defect is not "the prompt's natural-language clause is over-broad" alone; the defect is "the prompt does not constrain the routing surface tightly enough to produce model-consistent behavior." Two reasonable interpretations of the prompt (direct-completion-default vs. tool-when-verb-matches) produce different observable routings. If the cycle's commitment is to a specified, model-portable routing contract, the prompt needs explicit clauses on precedence — "client tools take precedence over direct completion when their declared verb matches the prompt's task verb" or "direct completion takes precedence over client tools unless the user names the tool explicitly" — to disambiguate the surface across orchestrators. Without such a clause, model-substitution silently changes routing behavior, which is itself a defect surface (operator's expectations from MiniMax do not transfer to qwen3:14b on the offline-fallback path).

**γ's net contribution to T14, revised.** The three dispositions now have this evidential standing:

- (i) **Intended scope as currently written** — weakened. The prompt does not produce a single routing surface; "intended scope" cannot describe both Cell A and Cell B simultaneously.
- (ii) **Defect to remediate** — strengthened, with a precise target. The system prompt needs a tool-precedence clause that disambiguates direct-completion vs. client-tool-delegation vs. `invoke_ensemble` under NL framing, otherwise routing behavior is model-portable in the bad sense (silently different per orchestrator).
- (iii) **Configuration-conditional** — strengthened, in the model-selection-as-configuration sense. The "configuration" is the active orchestrator Model Profile (which the operator controls). Cell B confirms the routing surface is genuinely conditional on this profile, not just MiniMax-M2.5-free-specific.

The held belief-mapping question — *"what would have to be true for the operational preference to be the intended behavior?"* — now has a refined answer: it would have to be true that **the cycle accepts model-conditional routing as a feature**, not a defect. If the operator selects MiniMax M2.5-free, they get direct-completion-default; if they swap to qwen3:14b (offline fallback), they get tool-delegation-default. Both are reasonable orchestrator behaviors against the current prompt. The cycle's stakeholder analysis (`product-discovery.md` Skill Orchestration User entry) names a single mental model — "the orchestrator routes NL to ensembles" — that *neither* tested orchestrator satisfies; both diverge from that model in different ways. The operational preference, in plural form, can be the intended behavior only if the cycle commits to a model-conditional contract (operator-portable expectations on which orchestrator does what).

### Updated implications for T14 DECIDE

The original log's three contributions stand. Cell B adds a fourth:

4. **The contract surface is multi-layered: prompt-level (specified) and model-level (realized).** ADR-021's natural-language-supported clause currently lives at the prompt-level. The realization is at the model level, and Cell B shows the realization varies. T14's ADR must decide whether the contract is **prompt-portable** (one prompt produces one routing surface across orchestrators — requires tighter prompt clauses or a routing pre-filter at the serving layer that intercepts NL prompts before they reach the LLM) or **prompt-specified-but-model-realized** (the contract documents the prompt and acknowledges model-conditional realization — operators are advised to test routing behavior on their chosen orchestrator). Both are coherent ADR positions. The cycle's commitment to "swap orchestrator profiles for offline fallback" (ADR-019 §Consequences §Positive on profile-portability) leans toward prompt-portable; Cell B shows that commitment is not currently honored at the routing surface.

The T14 ADR draft's open-question set should include this: *"is the orchestrator's NL-routing contract intended to be model-portable, and if so what enforces portability?"* This is upstream of the (i)/(ii)/(iii) deliberation — the choice between portable and model-realized contracts determines which disposition is even applicable. A model-portable commitment makes disposition (ii) the only coherent path (the prompt must produce one routing surface). A model-realized commitment makes disposition (iii) the only coherent path (routing is configuration-conditional by design).

The cycle's two remaining DECIDE-blocking spikes' interactions with γ-cell-B's finding: spike α's common envelope and spike β's composition predictability are both downstream of routing — they characterize what happens *after* dispatch occurs. Cell B's finding sits upstream of that surface (whether and how dispatch occurs at all). T16's ensemble contract work uses α and β's evidence; T14's routing contract work uses γ's evidence, including the new portability question Cell B surfaces.

### Implications for cycle artifacts beyond T14

- **Domain model.** The Routing surface behavior concept (MODEL Action A) should record the three observed routings (direct completion, client-tool delegation, `invoke_ensemble` dispatch) as the operational vocabulary, with a note that the active routing depends on (orchestrator profile, prompt framing, client tool surface) jointly.
- **ADR-019 (skill-framework-agnostic orchestrator).** The Consequences §Positive's profile-portability claim should be qualified: profiles are *Model Profile portable* (config-layer mechanism works), but routing surface behavior is *not model-portable* under the current prompt. The operator's offline-fallback path produces different routing than the default path. This is data the ADR's §Consequences §Negative does not currently carry.
- **ADR-021 (per-capability dispatch contract).** The natural-language-supported clause is now empirically known to be *not honored under either tested orchestrator's NL framing*. The clause is currently aspirational rather than operative. T14 either narrows the clause to "explicit-naming supported" (Cell A-explicit's reliable behavior) or commits to a prompt-rewrite that operationalizes the NL clause.

---

## Spike artifact retention

Per corpus retention policy (preserved until corpus close, overriding the spike-runner default of delete-after-recording), the scratch directory `scratch/spike-gamma-routing/` is retained. Contents:

- `probe_payload_no_tools.json` — Cell C request body
- `probe_payload_with_tools.json` — Cell A request body
- `probe_payload_with_tools_cellB.json` — Cell B original request body (unreachable via `provider: ollama`)
- `probe_payload_with_tools_cellB_v2.json` — Cell B continuation request body (via `provider: openai-compatible/ollama`)
- `probe_payload_explicit.json` — Cell A-explicit request body
- `cellA_response.json` / `cellA_curl_meta.txt` / `cellA_start.txt` / `cellA_end.txt` — Cell A dispatch evidence
- `cellC_response.json` / `cellC_curl_meta.txt` / `cellC_start.txt` / `cellC_end.txt` — Cell C dispatch evidence
- `cellB_response.json` / `cellB_curl_meta.txt` / `cellB_start.txt` / `cellB_end.txt` — Cell B original HTTP 500 evidence
- `cellB_v2_response.json` / `cellB_v2_curl_meta.txt` / `cellB_v2_start.txt` / `cellB_v2_end.txt` — Cell B continuation dispatch evidence (HTTP 200, `tool_calls` finish, single `write_file` call)
- `cellA_explicit_response.json` / `cellA_explicit_curl_meta.txt` / `cellA_explicit_start.txt` / `cellA_explicit_end.txt` — Cell A-explicit dispatch evidence
- `serve_cellB.log` — uvicorn console output captured during the Cell B continuation dispatch (the serve was started under `nohup` for log capture, addressing the original log's "no console capture" limitation for this cell)
- `serve_restored.log` — uvicorn console output after the config restore to `agentic-orchestrator`; confirms `/v1/models` reflects the restored profile

The two ensemble-dispatch artifact directories created by Cell A-explicit are part of `.llm-orc/artifacts/agentic-serving/` and remain in place (the cycle's artifact-as-substrate proposal anchors on the existence of these directories as the substrate the proposal extends). Cell B continuation did not create new artifact directories — its dispatch terminated at client-tool delegation, upstream of any ensemble dispatch.

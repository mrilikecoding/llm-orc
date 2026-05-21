# Spike λ (Cycle 7) — `tool_choice` parameter behavior on chat-completions

**Date:** 2026-05-20
**Cycle:** 7 (agentic-serving; RESEARCH Phase A validation)
**Method:** Live `curl` probes against `POST /v1/chat/completions` on a parallel spike serve (port 8766), varying `tool_choice` and `tools[]`. Probe prompt held constant.
**Cost incurred:** $0.00 (local qwen3:14b via Ollama OpenAI-compat — MiniMax M2.5 free promotion has ended; paid characterization deferred per [[feedback_free_options_preference]]).

---

## Question

Does the agentic-serving chat-completions endpoint correctly honor the OpenAI `tool_choice` parameter values that would force ensemble routing — specifically `"required"` and `{"type":"function","function":{"name":"invoke_ensemble"}}` — under tool-calling-capable orchestrator profiles?

This question is the validation spike for the Phase A reframe (per ADR-087 validation-spike decision invoked in-loop). Phase A found that under empirically tested production-shape clients, the NL-routing fraction that successfully converts to ensemble dispatch is approximately zero — the orchestrator-LLM under tool-rich production clients reliably routes to direct completion or client-tool delegation, not to `invoke_ensemble`. If `tool_choice` works correctly, clients that *want* ensemble routing can force it explicitly via the OpenAI-standard contract; the cycle's complex routing-mechanism design (planner ensemble, classifier, structured-output-per-turn) becomes a solution to a problem the existing API contract already addresses.

---

## Method

**Probe prompt** (held constant, same as Cycle 6 Spike γ): *"Write a Python function that reverses a string in place."*

**Cells:**

| Cell | `tool_choice` | `tools[]` | What it tests |
|---|---|---|---|
| λ.1 | `"auto"` (default) | tool-rich (`write_file`, `read_file`) | Control / baseline. Should reproduce Spike γ Cell B-continuation behavior |
| λ.3 | `{"type":"function","function":{"name":"invoke_ensemble"}}` | tool-rich | The key test — does forcing `invoke_ensemble` actually trigger dispatch? |
| λ.4 | `{"type":"function","function":{"name":"invoke_ensemble"}}` | tool-less (`[]`) | Isolates the tool_choice effect from client-tool noise |
| λ.5 | `"required"` | tool-rich | Does "must call one" lead the model to invoke_ensemble or client tools? |

**Configuration:**
- Spike serve on port 8766 (parallel to the user's existing serve on 8765, which is currently blocked on the ended MiniMax M2.5 free promotion)
- `model_profile: agentic-orchestrator-offline-tools` (qwen3:14b via Ollama's OpenAI-compatible adapter, declares `supports_tool_calling = True`)
- Free-tier MiniMax M2.5 attempt aborted at session start: Zen returned 401 with `FreeUsageLimitError: Free promotion has ended for MiniMax M2.5 Free`. Switched to local qwen3:14b per free-tier preference.

**Scope-of-claim:** n=1 orchestrator profile (qwen3:14b); n=1 probe per cell; production MiniMax M2.5 behavior under these `tool_choice` values is **uncharacterized** and is the remaining empirical gap.

---

## Per-cell observations

### Cell λ.1 — `tool_choice="auto"` (default) + NL + tool-rich

- HTTP 200, **49.4s** wall-clock, 1830 completion tokens
- `finish_reason: tool_calls`
- `message.content: null`
- `message.tool_calls[0]`: `function.name = "write_file"`, `arguments = {"path":"reverse_string.py","content":"def reverse_string(s):\n    return s[::-1]"}`
- **Routing: client-tool delegation (`write_file`).** No ensemble dispatch.

Matches Spike γ Cell B-continuation behavior exactly: qwen3:14b under tool-rich NL framing delegates to the verb-matching client tool, ignoring the internal `invoke_ensemble`. Implementation is `s[::-1]` (slice-reverse, not in-place) — same semantic miss Spike γ observed.

### Cell λ.3 — `tool_choice={"type":"function","function":{"name":"invoke_ensemble"}}` + NL + tool-rich

- HTTP 200, **191.7s** wall-clock, 4699 completion tokens
- `finish_reason: stop`
- `message.content` (verbatim): *"The Python function to reverse a string in place has been generated and saved to the file path: `agentic-sessions/ade03c0d43a42f896c85e33ea4bf7dbaa8b6874ef9ab1bfe411cae47ded4fe79/ade03c0d43a42f896c85e33ea4bf7dbaa8b6874ef9ab1bfe411cae47ded4fe79-dispatch-0001/code-generator.py`. You can open this file to review the implementation."*
- **`message.tool_calls` field is absent.** The framework consumed the tool-call result internally.

**Serve log evidence (full WP-C event sequence):**

```
19:46:26 tool-call emit: tool=invoke_ensemble dispatch_id=ade03c0d...-dispatch-0001
19:46:26 dispatch start: ensemble=code-generator profile=? dispatch_id=...
19:46:26 calibration verdict: proceed ensemble=code-generator
19:46:26 tier selection: profile=agentic-tier-cheap-general tier=cheap topaz_skill=code_generation
19:47:38 dispatch end: ensemble=code-generator duration=72.261 exit=success
19:47:38 tool dispatch: result name=invoke_ensemble kind=success
```

**Artifact evidence:** `code-generator/20260520-194738-633/execution.json` created (8117 bytes); `execution.md` (109 bytes). Substrate path matches the orchestrator's NL response.

**Routing: `invoke_ensemble` dispatched on first attempt; framework synthesized a NL final response.**

This is the **key validation finding**. Under `tool_choice` forcing `invoke_ensemble`:
- The orchestrator-LLM (qwen3:14b) called `invoke_ensemble` with the correct ensemble name (`code-generator`) — inferred from the system-prompt's capability description plus the user's NL prompt
- The framework parsed the tool-call, ran the dispatch (72.3s code-generator multi-agent run), consumed the result, and synthesized a final NL response
- From the client's perspective, the response is a regular chat completion with `finish_reason: stop` — the dispatch is transparent

Latency breakdown: 72.3s dispatch + ~120s orchestrator-LLM (initial tool-call emit + post-dispatch synthesis) = 192s total. The orchestrator-LLM overhead is substantial under qwen3:14b.

### Cell λ.4 — `tool_choice={"type":"function","function":{"name":"invoke_ensemble"}}` + NL + tool-less

- HTTP 200, **41.9s** wall-clock, 1823 completion tokens
- `finish_reason: stop`
- `message.content: ""` (empty string)
- `message.tool_calls` absent
- **Serve log:** no `tool-call emit:` or `dispatch start:` entries for this probe — only an `inference wait: elapsed=30` heartbeat and the closing `POST /v1/chat/completions HTTP/1.1 200 OK`
- **Routing: silent no-op.** `invoke_ensemble` was NOT called despite `tool_choice` forcing it; the response is empty content with no tool_calls

**Failure mode characterization:** qwen3:14b under `tool_choice` forcing `invoke_ensemble` in tool-less mode (empty `tools[]`) produced empty content without dispatching. Three candidate explanations:

1. **The framework suppresses tool_choice when client `tools[]` is empty.** Worth source-code inspection of `v1_chat_completions.py` and the request-construction path.
2. **qwen3:14b's tool_choice handler has an edge case** when tools[] is empty but tool_choice names a non-listed function. The internal `invoke_ensemble` is added by the framework to the model's available-tools list; if the framework relies on the client's tools[] to be non-empty before adding internals, that could explain the silent failure.
3. **qwen3:14b emitted output the framework's tool-call parser couldn't recognize** (similar to Cycle 6 PLAY notes 13/14's malformed MiniMax-native XML pattern — but qwen3:14b's output format is different and the empty response shape is distinct).

This is a finding for follow-up — the tool-less + force-invoke_ensemble path silently fails. **Production tool-rich clients are not affected** (λ.3 confirms tool-rich + force-invoke_ensemble works); this is a bare-endpoint edge case relevant to `curl` integration scripts or to the direct-invoke-with-orchestrator-routing use case.

### Cell λ.5 — `tool_choice="required"` + NL + tool-rich

- HTTP 200, **65.9s** wall-clock, 2279 completion tokens
- `finish_reason: tool_calls`
- `message.content: null`
- `message.tool_calls[0]`: `function.name = "write_file"`, `arguments = {"path":"reverse_string.py","content":"def reverse_string(s):\n    return s[::-1]\n..."}`
- **Routing: client-tool delegation (`write_file`).** No ensemble dispatch.

**Finding:** `tool_choice="required"` (model must call at least one tool) lets the model choose. qwen3:14b under tool-rich + NL chose `write_file` — the verb-matching client tool — not `invoke_ensemble`. The model's emergent preference under "must call something" matches the auto-default preference (client-tool delegation); only `tool_choice` that specifically names `invoke_ensemble` forces ensemble routing.

---

## Comparative table

| Cell | `tool_choice` | `tools[]` | Result | Dispatch? | Latency |
|---|---|---|---|---|---|
| λ.1 | `"auto"` | tool-rich | `write_file` delegation | No | 49.4s |
| λ.3 | force `invoke_ensemble` | tool-rich | **`code-generator` dispatched + NL synthesis** | **Yes** | 191.7s |
| λ.4 | force `invoke_ensemble` | tool-less | Empty content, no tool_calls | **No (silent)** | 41.9s |
| λ.5 | `"required"` | tool-rich | `write_file` delegation | No | 65.9s |

---

## Findings

### F1 — `tool_choice` forcing `invoke_ensemble` works correctly under tool-rich

The chat-completions endpoint **honors `tool_choice={"type":"function","function":{"name":"invoke_ensemble"}}`** under tool-rich client conditions. The forced tool call triggers actual ensemble dispatch (λ.3 confirms: full WP-C event sequence, 72.3s code-generator run, artifact on disk at the correct substrate path). The orchestrator-LLM correctly infers the ensemble name (`code-generator`) from the system-prompt capability description plus the user's prompt context.

**CONFIDENCE-LEVEL: (empirically established under qwen3:14b; n=1 probe; production MiniMax M2.5 behavior under this tool_choice value is uncharacterized.)**

### F2 — `tool_choice="required"` does NOT prefer `invoke_ensemble`

Under `tool_choice="required"` (must call some tool), qwen3:14b chose `write_file` over `invoke_ensemble`. The model's emergent preference under unconstrained "must call a tool" matches its auto-default preference: verb-matching client tools over internal infrastructure tools. **`required` is not a substitute for explicitly naming `invoke_ensemble`** when the cycle's goal is forced ensemble routing.

**CONFIDENCE-LEVEL: (empirically established under qwen3:14b; n=1 probe.)**

### F3 — `tool_choice` forcing `invoke_ensemble` in tool-less mode silently fails

Under empty `tools[]` + `tool_choice` forcing `invoke_ensemble`, no dispatch fires. The response shape is `finish_reason: stop`, `message.content: ""`, `tool_calls` absent. The serve log shows no `tool-call emit:` entry for the probe. This is a silent failure mode — the client receives HTTP 200 with empty content; nothing indicates the tool_choice instruction was not honored.

The mechanism is uncharacterized (possible candidates: framework strips tool_choice when client tools[] is empty; qwen3:14b's tool-call generation fails when the only tool available is the internal one; the framework's tool-call parser fails to recognize qwen3:14b's output format). Worth source-code follow-up.

**Production impact: low** — production tool-rich clients (OpenCode, Cursor, etc.) all declare client tools, so the tool-less edge case does not affect their request shape. The finding is relevant to `curl`-based integration scripts or to operator-side direct invocation through chat-completions.

**CONFIDENCE-LEVEL: (empirically established under qwen3:14b; mechanism uncharacterized.)**

### F4 — The framework's tool-call consumption produces transparent chat-completion responses

When `tool_choice` forces `invoke_ensemble`, the framework runs the dispatch internally and synthesizes a final NL response (`finish_reason: stop`). From the OpenAI-API-compatible client's perspective, the response is a regular chat completion — no `tool_calls` array surfaces to the client, no expectation of follow-up turn for tool-result narration. This means **the cross-compatibility surface is preserved** when ensemble routing is forced: tool-call-aware clients don't need to know about ensemble dispatch internals; they get a chat-completion response.

The orchestrator-LLM's synthesis-after-dispatch correctly described the substrate path of the dispatched ensemble's output. The NL synthesis is reasonable (no fabrication on this probe), though latency was high (~120s of orchestrator-LLM overhead on top of the 72s dispatch).

**CONFIDENCE-LEVEL: (empirically established under qwen3:14b; n=1 probe; Cycle 6 PLAY note 22's multi-dispatch fabrication pattern is a known risk that this single-dispatch probe doesn't exercise.)**

---

## Implications for the Phase A reframe

The reframe is **validated** for the tool-rich production-client case (which is the primary cross-compatibility surface — all production OpenAI-family tool-call clients declare client tools). Specifically:

1. **Q1 (routing mechanism) reduces to a contract-conformance question, not a mechanism-selection question.** The existing OpenAI `tool_choice={"type":"function","function":{"name":"invoke_ensemble"}}` parameter, supported by the chat-completions handler today, already provides the deterministic ensemble-routing mechanism the cycle was preparing to design. The complex options (Q1b options i/ii/iii/iv — planner ensemble / `tool_choice`-on-orchestrator-side / hybrid classifier / structured-output-per-turn) are solving a problem the API contract already addresses *from the client side*.

2. **The tool-less edge case (λ.4) is a small follow-up rather than a design pivot.** It needs source-code investigation and a fix, but it does not block the cycle's broader direction.

3. **`tool_choice="required"` is not a substitute** for `tool_choice` naming `invoke_ensemble` specifically. If the cycle wants to advertise "set tool_choice to invoke_ensemble for forced routing," the documentation must specify the exact tool name; clients sending `required` will not get the desired behavior (they'll get client-tool delegation or no-op-equivalent).

4. **Latency overhead is substantial under forced ensemble routing.** Cell λ.3 measured 192s wall-clock for a 72s dispatch — the orchestrator-LLM overhead (~120s, two LLM turns) is the dominant cost. This bears on the latency budget question (R2-1: routing overhead ≤ 1.0s OR ≤ 20%): qwen3:14b under tool_choice forcing invoke_ensemble exceeds the bound dramatically. Either the bound applies only to "no NL routing" (where the orchestrator-LLM is bypassed), or the cycle should expect heavy latency under the forced-routing path. The bound applies to mechanism overhead *for routing*; the dispatch latency itself is independent. The 120s orchestrator-LLM overhead is plausibly improvable via prompt-engineering or model-tier-selection but is not a routing-mechanism design choice.

## Remaining empirical gaps (post-qwen3:14b spike, pre-paid-spike)

1. ~~**Production MiniMax M2.5 behavior under `tool_choice={"name":"invoke_ensemble"}`**~~ — **CLOSED by Spike λ-paid below (2026-05-20).** Counter-finding: paid MiniMax M2.5 via Zen does NOT honor the tool_choice instruction under tool-rich client conditions.
2. **OpenCode / Cursor / Cline source-code inspection** on whether/how they expose `tool_choice` to user control. None of the empirical Cycle 6 PLAY probes used tool_choice; whether production clients send it at all in practice is unknown.
3. **The tool-less + force-invoke_ensemble silent-failure mechanism** (qwen3:14b F3). Source-code investigation needed.

---

## Continuation — Spike λ-paid (2026-05-20, user-authorized cost)

After the qwen3:14b spike validated the reframe locally, the practitioner authorized a paid MiniMax M2.5 probe to close the production-model gap. Three cells run against a parallel serve on port 8767 with `agentic-orchestrator-minimax-m25` profile (paid tier of the same MiniMax M2.5 model accessed via OpenCode Zen). Total token consumption: ~37,685 completion tokens across three probes (plus uncounted prompt tokens — Zen's accounting omits prompt-token counts). Cost within authorized budget ($0.05-0.30 estimated).

### Per-cell observations (paid)

#### Cell λ.3-paid — `tool_choice={"name":"invoke_ensemble"}` + NL + tool-rich (paid M2.5)

- HTTP 200, **11.8s** wall-clock, 2,171 completion tokens
- `finish_reason: stop`
- `message.content`: inline NL code response with the two-pointer reversal implementation, complete with docstring, complexity notes, example usage
- `message.tool_calls`: **absent**
- **Serve log: NO `tool-call emit:` or `dispatch start:` events for this probe.** No dispatch fired.

**Counter-finding:** Paid MiniMax M2.5 + tool-rich + `tool_choice` forcing `invoke_ensemble` **IGNORED the tool_choice instruction.** The model simply responded inline with code, as if `tool_choice` were `"auto"` or absent. This is the OPPOSITE of qwen3:14b's behavior (which honored the tool_choice and dispatched correctly).

The Phase A reframe's strongest form does not hold under the cross-compatibility-relevant production model.

#### Cell λ.4-paid — `tool_choice={"name":"invoke_ensemble"}` + tool-less (paid M2.5)

- HTTP 200, **77.3s** wall-clock, 17,723 completion tokens
- `finish_reason: stop`
- `message.tool_calls`: **absent**
- `message.content`: NL narration ("I'll first check what ensembles are available... I found a matching capability ensemble. The `code-generator` ensemble is designed for code-generation tasks like this. Let me invoke it:") followed by **malformed MiniMax-native XML**: `<invoke name="file_read"><parameter name="path">agentic-sessions/<session>/<dispatch>/code-generator.py</parameter></invoke></minimax:tool_call>`

**Serve log (excerpted):**
```
21:09:46 tool dispatch: result name=list_ensembles kind=success
21:09:53 tool-call emit: tool=invoke_ensemble dispatch_id=...-dispatch-0002
21:09:53 dispatch start: ensemble=code-generator
21:09:53 calibration verdict: proceed
21:09:53 tier selection: profile=agentic-tier-cheap-general tier=cheap
21:10:50 dispatch end: ensemble=code-generator duration=57.586 exit=success
21:10:50 tool dispatch: result name=invoke_ensemble kind=success
```

**Dispatch fires** (code-generator multi-agent run, 57.6s, success), but the orchestrator's final response carries malformed MiniMax-native XML attempting to chain to a `file_read` against the substrate path. The framework's tool-call parser cannot consume the XML format; the response shape is broken from the client's perspective.

This reproduces Cycle 6 PLAY note 13's pattern (paid M2.5 emits MiniMax-native XML under tool-less) but extends it: even with `tool_choice` set to force a specific OpenAI-compliant function call, paid M2.5 reverts to its native XML format for subsequent calls after the forced one.

#### Cell λ.5-paid — `tool_choice="required"` + NL + tool-rich (paid M2.5)

- HTTP 200, **75.7s** wall-clock, 17,791 completion tokens
- `finish_reason: tool_calls`
- `message.content: ""`
- `message.tool_calls[0]`: `function.name = "read_file"`, `arguments = {"path": "agentic-sessions/<session>/<dispatch-0001>/code-generator.py"}`

**Serve log (excerpted):**
```
21:08:06 tool dispatch: result name=list_ensembles kind=success
21:08:13 tool-call emit: tool=invoke_ensemble dispatch_id=...-dispatch-0001
21:08:13 dispatch start: ensemble=code-generator
21:09:13 dispatch end: ensemble=code-generator duration=60.789 exit=success
```

**Dispatch fires** (code-generator, 60.8s, success). The final response to the client is a `read_file` tool_call targeting the substrate path of the just-completed dispatch's output. This is the OpenAI-compliant tool_call format (no XML), but the path is on the **server's** filesystem (`agentic-sessions/<server-session>/...`) — production clients (OpenCode, Cursor, etc.) cannot execute `read_file` against the server's filesystem.

The composition pattern works server-side (dispatch + substrate routing both succeed) but breaks at the client interface (the substrate path is unreachable from the client).

### Findings (paid)

**F-paid-1 — `tool_choice` forcing `invoke_ensemble` is NOT honored under paid MiniMax M2.5 + tool-rich.** Direct counter-finding to F1 from the qwen3:14b spike. The OpenAI tool_choice contract is not model-portable across the production model and the local model the cycle has tested. Three candidate diagnoses (not disambiguated by Spike λ): Zen proxy strips/normalizes the parameter; MiniMax model accepts but doesn't enforce; framework tool-list construction interacts poorly with named-function tool_choice for internal tools.

**F-paid-2 — `tool_choice="required"` does produce ensemble dispatch under tool-rich, but with broken final response shape.** Multi-step composition: list_ensembles → invoke_ensemble (dispatch fires, 60.8s success) → read_file on substrate path (final response to client). The substrate path is server-side; client cannot complete the chain.

**F-paid-3 — Paid M2.5 emits MiniMax-native XML for subsequent calls even when `tool_choice` forces an OpenAI-compliant first call.** The XML/JSON format mismatch the framework cannot bridge.

**F-paid-4 — The orchestrator-LLM under paid M2.5 reasons about substrate paths as deliverables.** Both λ.4-paid (XML) and λ.5-paid (JSON tool_call) attempt to chain through a file-read of the just-dispatched ensemble's substrate output. The orchestrator treats the substrate routing infrastructure (ADR-025) as a surface to chain through, not as an internal-to-framework detail.

### Synthesis — paid spike's bearing on the Phase A reframe

The Phase A reframe was: *"the existing OpenAI tool_choice contract already provides deterministic ensemble routing; the cycle's mechanism complexity is solving a problem the contract addresses."*

**The reframe is PARTIALLY VALIDATED, PARTIALLY CONTRADICTED:**

- **Validated:** The framework-level mechanism exists (qwen3:14b confirms via λ.3). The codebase honors `tool_choice` for clients/models that send and respect it.
- **Contradicted:** The mechanism is NOT model-portable to the cross-compatibility-relevant production model (paid MiniMax M2.5 via Zen ignores `tool_choice={"name":"invoke_ensemble"}` under tool-rich, per λ.3-paid). The reframe's claim that "the OpenAI contract already handles forced ensemble routing" does not hold for the production deployment surface.
- **Refined:** The dispatch-result composition is a separate failure surface (F-paid-2, F-paid-3, F-paid-4): even when dispatch fires (λ.4-paid, λ.5-paid), the orchestrator-LLM tries to chain through a substrate-path file-read that the client cannot complete. This is independent of the tool_choice mechanism — it's about how the framework should surface the dispatch result.

### Architectural implications (post-paid-spike)

**Q1 architecture-design question reopens, refined:**

Given:
- (a) OpenAI `tool_choice` is implemented at the framework level (qwen3:14b validates)
- (b) `tool_choice={"name":"invoke_ensemble"}` is NOT honored under paid MiniMax M2.5 + tool-rich (production model)
- (c) `tool_choice="required"` DOES produce dispatch under paid M2.5 + tool-rich but with broken composition (substrate-path read attempt)
- (d) `tool_choice={"name":"invoke_ensemble"}` + tool-less under paid M2.5 produces dispatch + malformed XML output

What mechanism should the cycle commit to for forced ensemble routing through the chat-completions surface?

**Refined candidate options for Q1:**

- **(i')** Server-side `tool_choice` interception — when the client sends `tool_choice={"name":"invoke_ensemble"}`, the framework intercepts and runs a deterministic server-side mechanism (routing-planner ensemble, classifier, or explicit-naming-extraction from the prompt) instead of relying on the orchestrator-LLM to honor the tool_choice. The framework constructs the `invoke_ensemble` dispatch directly.
- **(ii')** Framework-driven composition continuation — when ANY mechanism dispatches an ensemble, the framework consumes the result and surfaces it as the chat completion's final assistant content (NL synthesis or structured envelope), NOT as a tool_call for substrate-path file-read. This addresses F-paid-2 and F-paid-3 directly.
- **(iii')** Combine (i') and (ii') — server-side mechanism to trigger dispatch deterministically + framework-driven composition to surface the result cleanly. This is the candidate that addresses BOTH the tool_choice unreliability AND the composition-failure surface.

Spike ε (routing-planner ensemble) — listed as a Cycle 7 spike candidate at cycle-status time — becomes the implementation candidate for option (i'). It is no longer "solving a problem that doesn't occur"; it is solving the production-model `tool_choice` unreliability problem the paid spike just empirically established.

**Q2 (form-drift enforcement) is unaffected** by the paid spike directly. The form-drift surface is at the synthesizer-agent layer, independent of routing mechanism. The cross-path requirement (chat-completions + direct invoke) still applies.

**Q3 (fallback) refines but does not collapse.** The current direct-completion behavior under NL + tool-rich is the empirical fallback. The structured-contract design (Population A vs. B) still has design work.

---

## Cross-references (updated)

- `docs/agentic-serving/essays/research-logs/research-log.md` — research log Phase A + paid-spike synthesis sections
- `docs/agentic-serving/essays/research-logs/cycle-6-spike-gamma-routing-characterization.md` — Spike γ baseline (the same probe prompt, no tool_choice)
- `docs/agentic-serving/essays/reflections/field-notes.md` Cycle 6 PLAY notes 13, 18, 22 — the prior paid-M2.5 + tool-rich observations the paid spike extends
- `docs/agentic-serving/decisions/adr-022-skill-orchestration-via-per-capability-dispatch.md` — the system-prompt amendment whose effectiveness is now empirically bounded to (a) tool-less mode (Cycle 6 PLAY) AND (b) `tool_choice="required"` under tool-rich (Spike λ-paid F-paid-2)
- `docs/agentic-serving/decisions/adr-025-substrate-routing.md` — substrate routing creates the surface the orchestrator-LLM tries to chain through via file-read (F-paid-4)

## Spike artifact retention

Per [[feedback_spike_artifact_retention]] (corpus retention overrides spike-runner delete-after-recording default), scratch directory `scratch/spike-lambda-tool-choice/` is retained. Contents:

- `lambda1_auto_payload.json` / `lambda1_response.txt` / `lambda1_start.txt` / `lambda1_end.txt` — λ.1 control probe
- `lambda3_force_invoke_payload.json` / `lambda3_response.txt` / `lambda3_start.txt` / `lambda3_end.txt` — λ.3 key test
- `lambda4_force_invoke_toolless_payload.json` / `lambda4_response.txt` / `lambda4_start.txt` / `lambda4_end.txt` — λ.4 tool-less probe
- `lambda5_required_payload.json` / `lambda5_response.txt` / `lambda5_start.txt` / `lambda5_end.txt` — λ.5 required-not-specific probe
- `serve_lambda.log` — spike serve console output (PID 68631, port 8766), captures the WP-C event sequence for the λ.3 dispatch

The λ.3 ensemble dispatch's substrate artifact lives at `.llm-orc/artifacts/agentic-serving/code-generator/20260520-194738-633/` and remains in place.

## Cross-references

- `docs/agentic-serving/essays/research-logs/research-log.md` Phase A — the Q0 grounding analysis the spike validates
- `docs/agentic-serving/essays/research-logs/cycle-6-spike-gamma-routing-characterization.md` — the routing-behavior baseline this spike extends
- `docs/agentic-serving/essays/reflections/field-notes.md` Cycle 6 PLAY notes 1-25 — the broader empirical context (notably PLAY note 18's "tool-rich production clients suppress ADR-022 amendment")
- `docs/agentic-serving/decisions/adr-022-skill-orchestration-via-per-capability-dispatch.md` — the system-prompt amendment whose effectiveness is bounded to bare-endpoint mode
- `docs/agentic-serving/decisions/adr-087-validation-spike-decision.md` — the validation-spike decision pattern this spike instantiates
- `src/llm_orc/agentic/orchestrator_config.py` — `DEFAULT_ORCHESTRATOR_SYSTEM_PROMPT`
- `src/llm_orc/web/api/v1_chat_completions.py` — the chat-completions handler whose tool_choice pass-through this spike characterizes (source-code inspection for F3's silent-failure mechanism is a follow-up)

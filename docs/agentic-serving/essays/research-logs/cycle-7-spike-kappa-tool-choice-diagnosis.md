# Cycle 7 DISCOVER — Spike κ: C2 Diagnosis Disambiguation (`tool_choice` Mechanism)

*2026-05-21*

## Purpose

DISCOVER-phase diagnostic spike addressing Cycle 7 RESEARCH GT-2(b) carry-forward. Disambiguates the three candidate diagnoses for the Spike λ-paid Cell λ.3-paid finding: *"paid MiniMax M2.5 via OpenCode Zen does not honor `tool_choice={"type":"function","function":{"name":"invoke_ensemble"}}` under tool-rich client conditions."*

The three candidates (per Spike λ-paid F-paid-1):

- **D1 — Zen proxy stripping.** OpenCode Zen strips/normalizes the `tool_choice` parameter before forwarding to MiniMax.
- **D2 — MiniMax non-conformance.** MiniMax M2.5 receives `tool_choice` but doesn't enforce it (model-level non-compliance with the OpenAI contract).
- **D3 — Framework tool-list interaction.** The agentic-serving framework's request construction (specifically how it constructs the `tools[]` array with internal tools mixed with client tools) interacts poorly with named-function `tool_choice` for internal tools.

Disambiguation is load-bearing for DECIDE: each diagnosis implies a different remediation path (Zen-side fix; model-tier substitution; framework refactor).

## Method

**Method: source-code inspection.** Trace `tool_choice` handling through the agentic-serving codebase, from the chat-completions HTTP handler (`v1_chat_completions.py`) through the openai-compatible model adapter (`openai_compat.py`) to the outgoing provider API request. Verify whether `tool_choice` is propagated, transformed, or dropped at any layer.

**No paid probes required.** Source-code inspection produces conclusive evidence that pre-empts any need for paid Zen/MiniMax probes — see Findings.

Cost: $0.00.

## Results

### Layer 1 — HTTP request parsing

`src/llm_orc/web/api/v1_chat_completions.py:401-408`:

```python
class _ChatCompletionsRequest(BaseModel):
    """Minimal subset of the OpenAI ``/v1/chat/completions`` request body."""

    model: str
    messages: list[_ChatCompletionMessage]
    stream: bool = False
    tools: list[dict[str, Any]] = Field(default_factory=list)
    user: str | None = None
```

**`tool_choice` is not a field on the Pydantic model.** The model carries five named fields (`model`, `messages`, `stream`, `tools`, `user`). Pydantic v2's default behavior for unknown fields is `extra="ignore"` — and the model has no `model_config = ConfigDict(...)` overriding that default. Therefore: **`tool_choice` sent by a client is silently stripped at the request-parsing boundary, before any framework logic runs.**

### Layer 2 — Context resolution

`src/llm_orc/web/api/v1_chat_completions.py:534-561`:

```python
def _resolve_context(request: _ChatCompletionsRequest) -> SessionContext:
    """Run the pre-handoff work shared by streaming and non-streaming paths."""
    _reject_reserved_tool_names(request.tools)
    messages = [ ... ]
    context = SessionContext(
        messages=messages,
        tools=list(request.tools),
        state=state,
    )
```

`SessionContext` carries `messages` and `tools` but no `tool_choice` field. Even if `tool_choice` survived Pydantic parsing (it doesn't), it would have no transport from the request into the downstream Runtime.

### Layer 3 — Outgoing provider API request

`src/llm_orc/models/openai_compat.py:113-118` (the openai-compatible adapter's `generate_with_tools` method, used by the orchestrator-LLM model adapter for paid MiniMax M2.5 via OpenCode Zen):

```python
body: dict[str, Any] = {
    "model": self.model_name,
    "messages": messages,
    "tools": tools,
    "stream": False,
}
if self.temperature is not None:
    body["temperature"] = self.temperature
if self.max_tokens is not None:
    body["max_tokens"] = self.max_tokens
```

**The outgoing HTTP body to the provider API contains no `tool_choice` field.** The body is constructed from four required fields and two optional ones (`temperature`, `max_tokens`); `tool_choice` is not among them.

### Repository-wide search

```
grep -rn "tool_choice" src/         → 0 matches
grep -rn "ToolChoice" src/          → 0 matches
grep -rn "tool_choice" tests/       → 0 matches
```

**Zero references to `tool_choice` anywhere in the codebase.** No handler reads it, no adapter forwards it, no test exercises it. The parameter is unsupported.

## Findings

### Finding κ.1 — D0: The framework drops `tool_choice` at the input layer (none of D1/D2/D3 hold)

The actual diagnosis is **D0: the agentic-serving chat-completions endpoint does not accept `tool_choice` at all.** Pydantic silently strips it during request parsing. The parameter never reaches the SessionContext, the orchestrator-LLM, the provider adapter, or the outgoing HTTP request to Zen or MiniMax.

The three candidates from Spike λ-paid F-paid-1 are all moot:

- **D1 (Zen proxy stripping) is moot** because nothing reaches Zen for it to strip. Even if Zen did forward `tool_choice` faithfully, the framework would never send one.
- **D2 (MiniMax non-conformance) is moot** because MiniMax never receives `tool_choice`. Whether MiniMax honors the OpenAI contract for `tool_choice` is an open question — but it is not the diagnosis for the λ.3-paid observation.
- **D3 (framework tool-list interaction) is moot in its original framing** but reframes to D0: the framework does not construct an outgoing request containing `tool_choice`, so any "interaction" question is misframed. The actual framework issue is omission, not interaction.

**CONFIDENCE-LEVEL: (empirically established by source-code inspection; complete absence of `tool_choice` references in src/ and tests/ is conclusive.)**

### Finding κ.2 — Spike λ findings need reframing under D0

Spike λ's Cell λ.3 finding (qwen3:14b honors `tool_choice` forcing `invoke_ensemble`) — restated under D0: **qwen3:14b dispatched `invoke_ensemble` based on emergent reasoning over the orchestrator system prompt + NL request content + tool-rich state, NOT because `tool_choice` forced it.** The model's choice happened to match the client's `tool_choice` intent by coincidence (the system prompt encourages ensemble dispatch when capability-matched; the NL request fit a capability; qwen3:14b reasoned its way to the same answer the `tool_choice` would have forced if it had been honored).

The Cell λ.3 "key validation finding" was correlation, not causation. The Phase A reframe ("the existing OpenAI `tool_choice` contract already provides deterministic ensemble routing") rested on the inference that the framework honored `tool_choice`; with D0 established, the inference fails.

Similar reframings:

- **λ.4 silent failure under qwen3:14b + tool-less + force `invoke_ensemble`**: not a `tool_choice` interaction edge case. With no `tool_choice` passing through and empty `tools[]`, qwen3:14b had nothing to call; empty response is the expected emergent behavior for that input shape.
- **λ.5 / λ.5-paid (`required` + tool-rich)**: `required` was also dropped by Pydantic. Both qwen3:14b's choice of `write_file` and paid M2.5's choice of `read_file` (after dispatch) reflect the models' default behavior under tool-rich + NL, with no `tool_choice` instruction in play.
- **λ.3-paid (paid M2.5 + tool-rich + force `invoke_ensemble`)**: not a model-portability gap. Paid M2.5 simply responded inline because no `tool_choice` reached it; the inline-code response is M2.5's natural emergent choice for that prompt shape.

### Finding κ.3 — The Phase A reframe is empirically invalidated

The Phase A reframe (Cycle 7 RESEARCH Essay-Outline §C3 ancestry) was: *"the OpenAI tool_choice contract already addresses the cycle's forced-routing requirement; the complex mechanism design solves a problem the API contract already addresses."*

Under D0, the contract is not implemented. The framework accepts `tool_choice` syntactically (Pydantic doesn't 400) but silently drops it. From a contract-compliance perspective, the framework does NOT honor `tool_choice`; the appearance of compliance in Spike λ Cell λ.3 was a coincidence of model behavior, not contract enforcement.

The cycle's mechanism-design work (routing-planner ensemble — Spike ζ; response-synthesizer — Spike ε; ADR-027 framework-driven pipeline) is therefore **not** "solving a problem the contract already addresses." It is solving a real problem: the contract is not implemented, and even if it were implemented at the parameter-passthrough layer, model-side conformance to `tool_choice` would still need empirical validation.

This bears directly on Cycle 7 RESEARCH GT-2 (hybrid-first ordering language drift). With D0 established:

- **Build-complexity comparison** between Tier 1 hybrid and ADR-027-direct shifts. Both require new framework code for the routing-decision mechanism. Tier 1 hybrid would add server-side `tool_choice` interception (parsing the field, building a deterministic dispatch from it); ADR-027-direct would route the same mechanism through a routing-planner ensemble. The "use the existing OpenAI contract" path is not a free baseline — it requires new framework code regardless. The cost differential collapses.
- **C2 (which-diagnosis) follow-up** is closed. C2 is D0; the remediation path is the same regardless of which downstream concern is prioritized — implement `tool_choice` handling, OR commit to a mechanism that bypasses `tool_choice` entirely.

### Finding κ.4 — Model-side `tool_choice` conformance for MiniMax M2.5 remains uncharacterized

This is the genuine remaining empirical gap: **if the framework were to start forwarding `tool_choice` correctly, would MiniMax M2.5 via Zen honor it?**

Spike κ does NOT address this question. The Spike λ-paid Cell λ.3-paid observation cannot speak to MiniMax conformance because `tool_choice` never reached MiniMax. Re-running λ.3-paid under a framework variant that DOES forward `tool_choice` would be the validation probe — but this is a future spike, not Spike κ.

The candidate diagnoses for that future spike (under the new "framework forwards `tool_choice`" baseline) reduce to two:

- **D1' — Zen strips `tool_choice` before forwarding to MiniMax.** Testable by hitting `https://opencode.ai/zen/v1/chat/completions` directly with `tool_choice` set, comparing observed behavior to a model that's known to honor `tool_choice` (e.g., OpenAI gpt-4o-mini if available through Zen).
- **D2' — MiniMax M2.5 receives `tool_choice` from Zen but doesn't enforce it.** Testable only with direct MiniMax API access (currently unavailable per `llm-orc auth list` — only `openai-compatible/zen` and `anthropic-claude-pro-max` configured).

Disambiguating D1' from D2' is **DECIDE-phase work** if the cycle chooses to commit to a `tool_choice`-based mechanism; otherwise it can stay open as a documented uncertainty.

## Implications for DECIDE

1. **C2 closed.** The original three diagnoses (D1/D2/D3) do not apply. The actual diagnosis is D0 (framework drops `tool_choice` at the input layer). Spike λ-paid F-paid-1 should be amended in the audit trail.

2. **GT-2(b) (C2 disambiguation BEFORE BUILD) is satisfied.** Build can proceed without resolving the D1/D2/D3 ambiguity because the ambiguity itself was misframed.

3. **GT-2(a) (build-complexity comparison) shifts in favor of ADR-027.** With D0 established, the "use the existing OpenAI contract" path is not a free baseline — Tier 1 hybrid requires implementing `tool_choice` handling AND server-side interception. ADR-027-direct routes the same routing-decision mechanism through a routing-planner ensemble. Both paths require comparable new framework code; ADR-027-direct additionally has the architectural advantage of removing the orchestrator-LLM from the dispatch path (Spike ε Finding ε.1 — confabulation pattern dissolved on the structural axis). Per GT-2(a)'s "if costs are within same order of magnitude, ADR-027 as primary recommendation" rule, the cost equivalence is now established; ADR-027-direct should be the cycle's primary DECIDE commitment.

4. **The Essay-Outline §C3, §C7 (and possibly §C6) need substantive backward-propagation edits.** §C3 currently treats `tool_choice` as a candidate primary mechanism for forced ensemble routing under the assumption the framework honors it. §C7 (tiered architecture) describes hybrid as "starting commitment" with ADR-027 as "structurally pre-committed escalation"; under D0 + Spike ε Finding ε.1, the tier ordering should flip. §C6 (capability-list discovery as first-order requirement) may need adjustment if the cycle commits to ADR-027-direct as the primary mechanism (capability-list discovery becomes the routing-planner's input contract rather than a separate Population-A degradation concern).

5. **Documentation correction needed at framework level (Thread A or in-cycle).** The `_ChatCompletionsRequest` model should either (a) explicitly support `tool_choice` per the OpenAI contract (parse, validate, propagate), or (b) explicitly reject it with a clear error so clients are not silently misled into thinking it's honored. The current silent-strip behavior is a contract-violation footgun: clients send `tool_choice`, expect deterministic routing, observe inconsistent results, and have no signal that the parameter was ignored. This is a small fix; could land as Thread A (outside methodology cycle) or fold into Cycle 7 BUILD if the chosen routing mechanism intersects `tool_choice` handling.

6. **Future spike candidate (DECIDE or BUILD phase, not Cycle 7 DISCOVER).** If the cycle commits to a `tool_choice`-aware mechanism, validate model-side conformance via a framework variant that does forward `tool_choice`, then probe Zen direct + MiniMax M2.5 to disambiguate D1' from D2'. Out of scope for Spike κ; documented as DECIDE-phase work.

## Methodological observation

**Source-code inspection as validation-spike-as-research-method (ADR-087 extension).** Spike κ was scoped as three live diagnostic probes against paid endpoints; source-code inspection alone produced more conclusive evidence at lower cost ($0.00 vs ~$0.05). The inspection ruled out the entire framing of the disambiguation question — D1/D2/D3 are moot because they presume `tool_choice` reached the provider.

The lesson generalizes: **before designing live probes to disambiguate provider-side behavior, verify the framework's outgoing request actually contains the parameter under test.** A two-minute source-code inspection of the request-construction path can eliminate entire branches of empirical investigation.

For ADR-087's validation-spike taxonomy: source-code inspection of the request-construction path should be a standard pre-probe step for any spike characterizing parameter-honoring behavior in downstream providers. This is a methodology amendment that future cycles should inherit.

## Cross-references

- **Spike λ** (`cycle-7-spike-lambda-tool-choice.md`) — the original `tool_choice` characterization spike whose findings F1 / F-paid-1 are reframed by Spike κ. The reframing does not invalidate λ's per-cell observations (the HTTP responses and serve logs are accurate); it invalidates the causal attribution (model-side `tool_choice` conformance vs. framework-side `tool_choice` omission).
- **Spike ζ** (`cycle-7-spike-zeta-routing-planner.md`) — routing-planner mechanism viability. Independent of Spike κ; routing-planner sidesteps `tool_choice` entirely. With κ's D0 finding, Spike ζ becomes the cycle's primary mechanism candidate (not a Tier 2 escalation).
- **Spike ε** (`cycle-7-spike-epsilon-pipeline.md`) — end-to-end pipeline validation. ε's Finding ε.1 (confabulation pattern dissolved on the structural axis) + κ's Finding κ.3 (Phase A reframe empirically invalidated) together support tighter ADR-027-direct commitment than the Essay-Outline §C7 currently asserts.
- **Cycle 7 RESEARCH Essay-Outline 006** (`essay-outline-006-cross-compatibility-routing-surface.md`) — §C3, §C6, §C7 need backward-propagation edits per Implication 4 above.
- **Cycle 7 RESEARCH Susceptibility Snapshot** (`housekeeping/audits/susceptibility-snapshot-cycle-7-research.md`) — GT-2(a) and GT-2(b) carry-forwards; Spike κ satisfies both.
- **ADR-087** (validation-spike-as-research-method) — Spike κ's source-code-inspection-first observation is a candidate amendment.
- **Source files inspected:**
    - `src/llm_orc/web/api/v1_chat_completions.py:401-408` (request model)
    - `src/llm_orc/web/api/v1_chat_completions.py:534-561` (context resolution)
    - `src/llm_orc/models/openai_compat.py:113-118` (outgoing API body construction)

## Cost record

$0.00. Source-code inspection only; no live API probes; no paid `tool_choice` characterization. The authorized κ budget (~$0.05) is unspent and available for future Spike-κ-derivative work if needed.

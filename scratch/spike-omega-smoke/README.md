# Spike Ω-smoke — real chat-completions contract check (§4b item 3)

**Status:** RAN 2026-06-29. Contract PASS over a real HTTP exchange (curl).
Real-OpenCode drive deferred to a `! opencode run` step (bootstrap-wedge
risk per the opencode-run-wedge note).

## Question

Does the ensemble form's marshal output translate cleanly into the real
OpenAI `/v1/chat/completions` wire contract — validated against an actual
HTTP client, not the hand-shaped harness (the WP-A "validate against the
real client" concern)?

## Shape

A standalone stdlib `http.server` adapter (`smoke_adapter.py`) wraps
`agent-turn-omega1` (single turn). The production endpoint routes through
the bespoke LoopDriver (ADR-043), so a standalone adapter is the way to
serve the ensemble form. It derives `(task, last_tool_result)` from
`messages[]`, invokes the ensemble, and marshals the result into a
`chat.completion` body. Also serves `GET /v1/models` for client bootstrap.

## Result — PASS

`POST /v1/chat/completions` returned a well-formed OpenAI body:

```json
{
  "object": "chat.completion",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_afea8602",
        "type": "function",
        "function": {"name": "write",
          "arguments": "{\"filePath\": \"converters.py\", \"content\": \"def celsius_to_fahrenheit(c):\\n    return (c * 9/5) + 32\"}"}
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

`content: null` + `tool_calls[]` + `finish_reason: "tool_calls"` is exactly
the OpenAI tool-call turn shape. The marshal's tool_call was nearly
passthrough (the ensemble already emits `{id,type,function:{name,arguments}}`
with `arguments` as a JSON string). `GET /v1/models` returns a stub list.

## Findings

### #1 — The ensemble output already fits the wire contract. (PASS)

No translation logic of substance — the marshal stage emits the OpenAI
tool-call shape directly. The adapter is a thin HTTP shell + message→task
extraction. This confirms the §8 "thin serving adapter is irreducible but
small" claim for the single-turn, happy-path case.

### #2 — What this does NOT yet cover.

- **Real OpenCode**, not curl: the multi-turn loop, tool execution
  round-trip (role:tool result re-prompt), and streaming (SSE). The
  standalone adapter is non-streaming; OpenCode may request `stream:true`.
- **Multi-turn session continuity** through the adapter (Ω-2's substrate
  threading wired to the HTTP layer). The smoke is single-turn.
- Latency: one omega-1 turn runs the 3-stage production code-generator
  (~200s). A real OpenCode session would feel this per turn.

These are the next checks if the (A)-vs-(B) decision is to keep pushing the
ensemble-serving path. For now the wire contract itself is confirmed.

## To drive real OpenCode (user step)

```
uv run python scratch/spike-omega-smoke/smoke_adapter.py 8099   # in one shell
# point OpenCode at http://127.0.0.1:8099/v1 with model "omega1"
# (curl the serve directly first to confirm PONG before launching the client)
```

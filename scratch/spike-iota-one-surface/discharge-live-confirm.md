# Loop-back #9 BUILD discharge — live OpenCode title-gen confirm

**Date:** 2026-06-18. **$0 local** (offline seat `agentic-orchestrator-offline-tools` = qwen3:14b via Ollama; config swapped for the run and restored after).

**Method:** replayed OpenCode's *actual captured* toolless title-generator request
(`scratch/spike-pi-opencode-roundtrip/requests_A.jsonl`, the no-tools request:
`system: "You are a title generator. You output ONLY a thread title."`) against a
live `llm-orc serve` on `:8771`, post-collapse. This is the request shape that
currently 500s on the half-built single-turn pipeline (the WP-B routing-planner
ensemble was never built).

**Request (toolless):** `{"model":"spike-model","messages":[{role:system,"...title generator..."},{role:user,"Generate a title for this conversation:"}],"stream":false}` — no `tools[]`.

**Response:**
```
HTTP 200
{"id":"chatcmpl-...","object":"chat.completion","model":"spike-model",
 "choices":[{"index":0,"message":{"role":"assistant","content":"hello.py generation"},
 "finish_reason":"stop"}], ...}
```

**Verdict — DISCHARGE MET:**
- **HTTP 200** — the 500 is gone. The toolless request now routes to the loop-driven terminal (not the half-built pipeline) and succeeds.
- **Clean text** — `content: "hello.py generation"`, a sensible title (the captured conversation was "create a file hello.py that prints hello world").
- **`finish_reason: "stop"`, no `tool_calls`** — the loop took the finish-with-text path; it did not emit an un-executable client tool call (F-ι.1 holds end-to-end at the HTTP layer).

The collapse (ADR-043) is validated end-to-end against the real client's own request shape: OpenCode's toolless aux requests are served gracefully by the unified loop, and the prior 500 is fixed.

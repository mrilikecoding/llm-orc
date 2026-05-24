# Cycle 7 OQ #20 — Population A Tool-Family Timeout Research

**Date:** 2026-05-22
**Context:** Cycle 7 DECIDE precondition; informs latency ADR.
**Method:** Documentation + source code + community discussion review.

## Summary

Across the four Population A clients, defaults split cleanly into two camps. OpenCode and Aider are permissive: OpenCode ships a 300,000 ms (5-minute) per-provider request timeout, and Aider sets `timeout: None` (delegating to LiteLLM/openai-python's 600 s default). Cline is hostile: a hard-coded 30,000 ms (30 s) default that practitioners regularly hit with Ollama and which has its own dedicated tracking issue (#4308) noting the knob is "ineffective for long-running local model requests." Cursor is the most opaque — it routes most agentic traffic through Cursor's own backend rather than the configured base URL, the timeout is undocumented and not user-configurable, and forum reports cite 4 s / 20 s / 200 s ceilings in various code paths. **Net: the ~36 s single-step floor breaches Cline defaults and likely breaches Cursor's effective ceiling; the ~64 s chained floor breaches both. OpenCode and Aider accommodate both floors out of the box, but only Aider streams by default (which masks wall-clock timeouts as long as tokens are flowing).**

## Per-client findings

### OpenCode (sst/opencode)
- **Default timeout (non-streaming):** 300,000 ms (5 minutes). Documented in the Config reference: "timeout - Request timeout in milliseconds (default: 300000). Set to `false` to disable."
- **Configurable knobs:** `provider.<name>.options.timeout` (milliseconds) in `opencode.json`. Users have set values up to 600,000 ms (10 min); issue #3708 shows 150,000 ms still failing for very large local models.
- **Streaming default:** Uses `@ai-sdk/openai-compatible` (Vercel AI SDK) for `/v1/chat/completions`, which defaults to streaming. The `timeout` covers the full request including stream open.
- **Known long-request patterns:** Issue #3708 ("Timeouts despite setting in config") and #1065 ("How to control Timeouts") document users hitting timeouts with large local models even at 150 s. The Apiyi config guide reports the 5-minute default is sufficient for relay-station use; local large models are the friction point.
- **Source:** [opencode.ai/docs/config](https://opencode.ai/docs/config/); [issue #3708](https://github.com/sst/opencode/issues/3708); [issue #1065](https://github.com/sst/opencode/issues/1065).

### Cursor
- **Default timeout (non-streaming):** Not documented. No published value, no configurable knob. Forum reports cite different ceilings per code path: 4 s for "agent execution provider," 20 s for terminal tool, 200 s for some tool calls.
- **Configurable knobs:** None exposed. Open feature request #146080 asks for the ability to extend timeouts "another 30 to 60 seconds or so" — this has not shipped.
- **Streaming default:** Cursor's "Override OpenAI Base URL" only routes plan-mode chat (Cmd/Ctrl+L). Composer/agent, inline edit, autocomplete, and apply all stay on Cursor's proprietary backend. So the "OpenAI-compatible timeout" question is partly moot — most agentic traffic never reaches the override target. Cursor has also been reported to send `Responses` API payloads to `/v1/chat/completions` endpoints, breaking format compatibility independent of timeout.
- **Known long-request patterns:** Forum threads document LM Studio users seeing "Client disconnected. Stopping generation…" mid-generation on local models; no fix or configurable workaround documented.
- **Source:** [forum #146080](https://forum.cursor.com/t/request-timeout-adjustment-in-cursor-ide/146080); [forum #45143](https://forum.cursor.com/t/cursor-timeout-agent-mode/45143); [forum #152006](https://forum.cursor.com/t/override-openai-base-url/152006); [Bifrost Cursor docs](https://docs.getbifrost.ai/cli-agents/cursor).

### Cline (cline/cline)
- **Default timeout (non-streaming):** 30,000 ms (30 s) — confirmed by multiple GitHub issues (#2941, #4308, #9182) and community fix posts. Issue #2941 reproduces the exact error string: `"Ollama request timed out after 30 seconds"`.
- **Configurable knobs:** "Request Timeout" setting in the provider configuration UI (`requestTimeoutMs` internally). Recommended values from community fix guide: 60,000 ms for 7B on decent hardware, 90,000 ms for Qwen3-14B on 8 GB VRAM, 180,000+ ms for 30B+ models. Issue #4308 notes the knob has been "ineffective for long-running local model requests" — set values can still abort early.
- **Streaming default:** OpenAI-compatible provider streams by default (uses streaming `/v1/chat/completions`). The 30 s timeout applies to time-to-first-token / total request, not idle gap.
- **Known long-request patterns:** Heavily documented. Issue #2941 (regression in v3.12.3), #4308 (ineffective knob for long-running requests), #8154 (separate 30 s ceiling on `execute_command` tool calls), #9182 (CLI variant of the same problem). PR #3029 partially fixed Ollama but not all providers (#6361). LocalLLM.in publishes a dedicated fix guide.
- **Source:** [issue #2941](https://github.com/cline/cline/issues/2941); [issue #4308](https://github.com/cline/cline/issues/4308); [issue #9182](https://github.com/cline/cline/issues/9182); [LocalLLM.in fix guide](https://localllm.in/blog/cline-ollama-timeout-fix).

### Aider
- **Default timeout (non-streaming):** `None` — no enforced timeout at Aider's layer. Falls through to LiteLLM, then to openai-python, whose `DEFAULT_TIMEOUT` is 600,000 ms (10 minutes).
- **Configurable knobs:** `--timeout <seconds>` CLI flag, `timeout:` YAML key, or `AIDER_TIMEOUT` env var. Default `None`. No documented max.
- **Streaming default:** `stream: True` by default. Configurable via `--no-stream` / `stream: false` / `AIDER_STREAM`. Aider docs note reasoning models sometimes "prohibit streaming"; in those cases non-streaming is forced and the timeout becomes load-bearing.
- **Known long-request patterns:** Streaming masks most timeout issues — as long as tokens flow within the underlying HTTP read timeout, Aider waits. The friction case is reasoning models in non-streaming mode, where the openai-python 10-minute ceiling can bite. No Aider-specific timeout issue tracker entries surfaced for our scenario.
- **Source:** [aider.chat/docs/config/options.html](https://aider.chat/docs/config/options.html); [aider.chat/docs/config/aider_conf.html](https://aider.chat/docs/config/aider_conf.html).

## Synthesis for DECIDE

- **Is the current ~36 s floor compatible with Population A defaults?** Conditional. OpenCode (300 s) and Aider (None/600 s) yes. Cline (30 s) **no** — single-step planner request breaches default by ~6 s, every request. Cursor unknown but reports of 4–20 s ceilings on agentic paths suggest **no** for non-plan-mode traffic; plan-mode-only use is workable since Cursor restricts override traffic anyway.
- **Is the current ~64 s chained floor compatible?** No for Cline defaults (2.1× over). No for Cursor's reported agentic-path ceilings. Yes for OpenCode and Aider.
- **Configurable mitigation paths:**
  - OpenCode operators: bump `provider.<name>.options.timeout` to 120,000–180,000 ms (covers chained floor with headroom).
  - Aider operators: no change needed; streaming default masks the wall-clock; if non-streaming is forced, set `--timeout 180`.
  - Cline operators: bump "Request Timeout" to ≥ 120,000 ms — but note #4308's caveat that the knob is unreliable on some providers. Recommend an integration smoke test in tuning docs.
  - Cursor operators: no usable mitigation. Cursor's base-URL override is plan-mode-only and the timeout knob doesn't exist. Population A coverage for Cursor is effectively "chat panel only, accept Cursor's silent ceilings."
- **Residual risk:**
  - Cline's `requestTimeoutMs` is documented as ineffective on some providers (#4308); even an operator who increases it may still hit early aborts. Need a llm-orc-side characterization test against Cline before claiming support.
  - Cursor's true non-streaming timeout for the plan-mode override path is undocumented. We could measure empirically but cannot rely on stability across Cursor updates.
  - The OpenAI Node SDK that OpenCode (via Vercel AI SDK) and many wrappers depend on has a 10-minute HTTP-level ceiling separate from any per-provider knob (per openai-python #762 and the openclaw report); this is not currently load-bearing for our floors but bounds the absolute headroom.
  - This research only covers documented/forum-surfaced behavior. None of the four clients publishes a formal timeout SLA, and all four have shown timeout-related regressions in the last 12 months.

## Sources

- [OpenCode Config docs](https://opencode.ai/docs/config/)
- [OpenCode issue #1065 — How to control Timeouts](https://github.com/sst/opencode/issues/1065)
- [OpenCode issue #3708 — Timeouts despite setting in config](https://github.com/sst/opencode/issues/3708)
- [OpenCode Providers docs](https://opencode.ai/docs/providers/)
- [Cursor forum #146080 — Request Timeout adjustment feature request](https://forum.cursor.com/t/request-timeout-adjustment-in-cursor-ide/146080)
- [Cursor forum #45143 — Cursor timeout agent mode](https://forum.cursor.com/t/cursor-timeout-agent-mode/45143)
- [Cursor forum #152006 — Override openai base url](https://forum.cursor.com/t/override-openai-base-url/152006)
- [Cursor forum #15494 — Using Local LLMs with Cursor](https://forum.cursor.com/t/using-local-llms-with-cursor-is-it-possible/15494)
- [Bifrost — Cursor integration docs](https://docs.getbifrost.ai/cli-agents/cursor)
- [Cline issue #2941 — Ollama 30-second timeout](https://github.com/cline/cline/issues/2941)
- [Cline issue #4308 — Ollama provider timeout handling for long-running local models](https://github.com/cline/cline/issues/4308)
- [Cline issue #8154 — 30-second timeout on execute_command](https://github.com/cline/cline/issues/8154)
- [Cline issue #9182 — CLI timing out with Ollama](https://github.com/cline/cline/issues/9182)
- [Cline issue #6361 — Custom timeout fix only covers Ollama](https://github.com/cline/cline/issues/6361)
- [Cline OpenAI-compatible provider docs](https://docs.cline.bot/provider-config/openai-compatible)
- [LocalLLM.in — Cline Ollama 30s timeout fix guide](https://localllm.in/blog/cline-ollama-timeout-fix)
- [Aider options reference](https://aider.chat/docs/config/options.html)
- [Aider YAML config reference](https://aider.chat/docs/config/aider_conf.html)
- [openai-python issue #762 — Default timeout is 10 minutes](https://github.com/openai/openai-python/issues/762)

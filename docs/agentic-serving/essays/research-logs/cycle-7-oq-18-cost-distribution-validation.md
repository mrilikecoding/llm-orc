# Cycle 7 OQ #18 — Cost-Distribution Lens Validation Against Population A Voice

**Date:** 2026-05-22
**Context:** Cycle 7 DECIDE precondition; validates the cost-distribution framing in essay-outline-006 §C6 + product-discovery.md Tension 18 + Amendment A2.
**Method:** Cross-reference Population A client documentation, GitHub issues/discussions, and community write-ups for evidence about user expectations of custom OpenAI-compatible endpoints (Aider, Cline, OpenCode, Cursor).

## Summary

Population A voice **partially corroborates** the cost-distribution lens, but reframes it in important ways. Population A clients **do** surface per-request cost and **do** support multi-model routing — but the routing is exposed *inside the client* (Aider architect/editor/weak tiers, Cline Plan/Act, OpenCode subagent model tiers), not delegated to the endpoint. The dominant expectation of a custom OpenAI-compatible base URL across all four clients is "transparent single-model proxy at the configured model ID": users select the model in their own UI and expect the endpoint to honor it. The "use ensembles effectively" expectation the practitioner names is **not visible in Population A discourse**; what *is* visible is strong sensitivity to any silent divergence between the configured model and what actually answers (Cline issue #10551 on context-window caps, OpenCode issue #20859 on subagent model substitution). The cost-distribution lens survives, but the value it protects is closer to *configuration honesty* than to *load distribution*.

## Per-question findings

### Q1: Do Population A clients reveal users care about per-request cost?

Yes, strongly, but at the client layer rather than the endpoint layer.

- Aider displays per-interaction token counts and cost via `/tokens`, and the architect-mode documentation is explicit that "Architect-mode runs typically cost 30-50% less than the same task done by the architect model alone" ([Aider — Separating code reasoning and editing](https://aider.chat/2024/09/26/architect.html)).
- Aider stores `input_cost_per_token` / `output_cost_per_token` per model in its config and accumulates session cost ([Aider issue #257 — session cost](https://github.com/Aider-AI/aider/issues/257)).
- Cursor users have an outstanding feature request for per-request token/$ reporting ([Cursor forum thread](https://forum.cursor.com/t/token-usage-and-costs-report-per-request-and-per-session/138980)) — visibility is wanted but not yet a built-in.
- Cline's `Plan/Act model usage patterns` blog shows users actively choosing cheaper models for Act mode after planning with stronger ones ([Cline blog](https://cline.bot/blog/plan-act-model-usage-patterns-in-cline)).

The cost concern is real, but in every case it is framed as *the user's per-task choice between named models they selected*, not as *the endpoint's responsibility to distribute load*.

### Q2: Do Population A clients reveal multi-model / ensemble expectations?

Yes — but the multi-model machinery lives in the client, and the endpoint is treated as a single-model executor.

- Aider's three-tier architect/editor/weak split issues distinct OpenAI-shaped requests per role, each to a configured model. The endpoint sees one model at a time ([Aider model configuration](https://deepwiki.com/Aider-AI/aider/7-model-configuration-and-capabilities)).
- Cline's Plan/Act lets users assign separate models per mode ([Cline Plan & Act docs](https://docs.cline.bot/core-workflows/plan-and-act)). The popular pattern is Opus-4.1 → Sonnet-4, but the client is the dispatcher.
- OpenCode's agent teams allow per-agent model assignment, and the open feature request #6651 (`model_tier`: quick/standard/advanced) is the closest Population A comes to *delegating* tier selection — but the proposal still expects the client to map tier → model name, not the endpoint to decide ([OpenCode #6651](https://github.com/anomalyco/opencode/issues/6651)).
- Critically, when subagent model selection is silently overridden (OpenCode #20859, GitHub Copilot provider billing all subagent requests to orchestrator model), users file it as a **bug** ([OpenCode #20859](https://github.com/anomalyco/opencode/issues/20859)).

The endpoint-side-orchestration expectation the practitioner names ("use ensembles effectively") **does not appear** in Population A documentation or discussion. Population A users orchestrate from the client.

### Q3: Do Population A clients reveal trust patterns for custom endpoints?

Yes, and the dominant pattern is "transparent proxy honoring the configured model ID."

- The Cline #9600 discussion on `/responses` vs `/chat/completions` is explicit: users tried proxy layers "but this adds unnecessary latency and complexity" — they want the endpoint to be a thin pass-through and treat anything else as friction ([Cline #9600](https://github.com/cline/cline/discussions/9600)).
- Cline #10551 documents user frustration when a custom OpenAI-compatible endpoint's configured 1M context was capped at 128K by a generic fallback — the divergence between configured and actual behavior was the complaint ([Cline #10551](https://github.com/cline/cline/issues/10551)).
- OpenCode's custom-provider docs frame `@ai-sdk/openai-compatible` strictly as "wrap any endpoint that adheres to the OpenAI Chat Completions protocol as a provider" — no notion of endpoint-side routing ([OpenCode providers](https://opencode.ai/docs/providers/)).
- The broader OpenAI-compatible-proxy ecosystem (LiteLLM, OpenZiti llm-gateway, Bifrost) is consistent: the endpoint logs, validates, and forwards; it does not invent capability the client didn't request ([OpenZiti llm-gateway](https://github.com/openziti/llm-gateway)).

Trust violation in Population A's vocabulary = "endpoint did something other than execute my configured model on my request." That is narrower than the practitioner's "use ensembles effectively" framing, but it sits at the same architectural layer.

### Q4: What does Population A community discussion say about latency-vs-quality tradeoffs?

The tradeoff is acknowledged and managed *at the client*, per task, by the user.

- Architectural reviews of agentic coding tools note "latency matters more for developer-in-the-loop autocomplete than background PR review" and "cost per token matters at high volume but becomes secondary if success rate is too low" ([CodeScene — Agentic AI Coding Patterns](https://codescene.com/blog/agentic-ai-coding-best-practice-patterns-for-speed-with-quality)).
- Claude Code's effort-levels and Aider's architect/editor split are repeatedly cited as the right *interface* for the tradeoff — the user selects depth, the tool routes accordingly ([Morph — AI coding costs](https://www.morphllm.com/ai-coding-costs)).
- I found **no Population A discussion** where users expect a custom endpoint to *make* the latency-vs-quality call on their behalf. The tradeoff is consistently framed as the user's call, expressed through model selection or mode switching in the client.

## Synthesis

- **Is the cost-distribution lens framing corroborated by Population A voice?** **Partially.** The *layer* is right — Population A cares about cost-distribution as an architectural property, not just per-task quality. The *agent* is wrong: Population A locates that distribution in the client, not the endpoint. The practitioner's "transparent endpoint promise" matches Population A trust patterns; the "use ensembles effectively" promise does not.
- **Where does Population A voice add nuance to the framing?** It sharpens what counts as degradation. Population A's degradation signal is not "fell back to direct completion" — they would not detect that as degradation if the response and billed model name were honest. Their degradation signal is *configuration dishonesty*: the endpoint billed/labelled/contextualized differently than configured (OpenCode #20859, Cline #10551). For llm-orc this maps onto: did the response advertise it was direct-completion fallback, or did it pretend to be ensemble output?
- **Where does Population A voice push back on the framing?** The framing assumes the user-developer trusts the endpoint to "use ensembles effectively." Population A's default trust posture is the opposite — they trust the endpoint to *not* exercise discretion. The cost-distribution-accountability claim survives only if the project-developer (not the user-developer) is the accountability holder. The user-developer in Population A would more naturally articulate the value as *honest labeling* of what served the request.
- **Residual uncertainty:** I found no Population A user articulating the *project-developer* perspective explicitly. The practitioner's framing of "project-developer needs the framework to actually distribute load" is plausible but unvalidated by Population A voice because Population A clients don't expose that role. This is a real gap, not something the validation can resolve from public evidence.

## Recommendation for DECIDE

**Sharpen, do not replace.** Keep the cost-distribution architectural layer as the framing axis — Population A voice corroborates that costs and routing decisions live at an architectural layer, not the per-task quality layer. But split the "transparent endpoint promise" into two distinct sub-promises: (1) *configuration honesty* (the response truthfully reports what served it, including direct-completion fallback) and (2) *cost-distribution accountability* (the project-developer's expectation that ensembles do dispatch on capability-matched requests). Population A directly corroborates (1) and is silent on (2); ADRs that conflate them will fail the susceptibility audit Population A would apply.

## Sources

- [Aider — OpenAI-compatible APIs](https://aider.chat/docs/llms/openai-compat.html)
- [Aider — Chat modes](https://aider.chat/docs/usage/modes.html)
- [Aider — Separating code reasoning and editing](https://aider.chat/2024/09/26/architect.html)
- [Aider — Model configuration and capabilities (DeepWiki)](https://deepwiki.com/Aider-AI/aider/7-model-configuration-and-capabilities)
- [Aider issue #257 — Show accumulated tokens / cost for the entire session](https://github.com/Aider-AI/aider/issues/257)
- [Cline — OpenAI Compatible provider docs](https://docs.cline.bot/provider-config/openai-compatible)
- [Cline — Plan & Act mode docs](https://docs.cline.bot/core-workflows/plan-and-act)
- [Cline blog — Plan/Act model usage patterns](https://cline.bot/blog/plan-act-model-usage-patterns-in-cline)
- [Cline #9600 — Custom endpoint path selection (/chat/completions vs /responses)](https://github.com/cline/cline/discussions/9600)
- [Cline #10551 — DeepSeek V4 Pro context window capped at 128K despite config](https://github.com/cline/cline/issues/10551)
- [OpenCode — Providers documentation](https://opencode.ai/docs/providers/)
- [OpenCode #6651 — Dynamic model selection for subagents via Task tool](https://github.com/anomalyco/opencode/issues/6651)
- [OpenCode #20859 — Subagent models ignored under GitHub Copilot provider](https://github.com/anomalyco/opencode/issues/20859)
- [Cursor forum — Token usage and $ costs report per request/session](https://forum.cursor.com/t/token-usage-and-costs-report-per-request-and-per-session/138980)
- [Cursor — Custom base URLs feature request](https://forum.cursor.com/t/custom-base-urls-for-each-custom-model/147219)
- [OpenZiti llm-gateway — Zero-trust OpenAI-compatible proxy](https://github.com/openziti/llm-gateway)
- [CodeScene — Agentic AI Coding: Speed-with-Quality patterns](https://codescene.com/blog/agentic-ai-coding-best-practice-patterns-for-speed-with-quality)
- [Morph — The real cost of AI coding in 2026](https://www.morphllm.com/ai-coding-costs)

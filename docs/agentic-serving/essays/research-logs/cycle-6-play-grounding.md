# Cycle 6 PLAY-tail grounding

**Date:** 2026-05-20 (after PLAY-boundary susceptibility snapshot)
**Trigger:** Susceptibility-snapshot-cycle-6-play recommended a Grounding Reframe before Cycle 7 entry. Three concrete actions named.
**Purpose:** Ground the architectural premises Cycle 7 is being scoped against, before RESEARCH entry, so the premises survive examination rather than being inherited as conclusions.

---

## Action 1: Belief-map the framework-driven pipeline against named alternatives

The PLAY-boundary snapshot named three alternatives that were mentioned as spitball items but not seriously examined against Spike δ's evidence: extended system-prompt intervention, constrained decoding to force `tool_choice`, and fine-tuning the orchestrator-LLM.

Belief-mapping question: **what would have to be true for each alternative to address both the routing-decision failure AND the chain-handling failure observed in Cycle 6 PLAY?**

### Extended system-prompt intervention

The premise: a more aggressive or more carefully-engineered system prompt could overcome the tool-rich client's gravitational pull on the orchestrator-LLM.

What would have to be true:
- The orchestrator-LLM would be steerable by system prompt under tool-rich conditions
- Refinements like "you MUST call `invoke_ensemble` for capability-matched requests; refusing to do so is a defect" or multi-shot examples showing dispatch patterns would shift behavior
- The shift would be reliable across model tiers and prompt shapes

Empirical evidence from Cycle 6:
- ADR-022 amendment was the first attempt — already in production
- Two distinct model tiers (free M2.5, paid M2.5) both refused to dispatch under OpenCode tool-rich + NL framing
- Behavior is consistent across the cycle (note 1, note 18) — not an edge case
- The model treats client tools as "the user's preferred surface" and routes to them by default — this is a categorical preference, not a marginal failure

What's *not* tested:
- More aggressive system prompts (e.g., explicit "you MUST" language with examples)
- Multi-shot prompting with capability-match demonstrations

Verdict: **The empirical floor is "doesn't work under tool-rich with current prompt engineering."** More aggressive prompts might shift the routing decision but the burden of proof shifts to demonstrating they do. Even if they did, the **chain-handling failure** (input.data drift between dispatches) is a separate issue that system-prompt engineering does not address — that failure is in the orchestrator-LLM's decision space about how to compose dispatches, not whether to dispatch. Extended system-prompt intervention could plausibly address routing-decision under tool-rich; it cannot address chain-handling.

### Constrained decoding via `tool_choice`

The premise: OpenAI's `tool_choice` parameter forces the model to call a specific tool. Framework sets `tool_choice={"type": "function", "function": {"name": "invoke_ensemble"}}` when a capability match is detected, forcing dispatch deterministically.

What would have to be true:
- Zen's OpenAI-compatible endpoint must support the `tool_choice` parameter (untested in Cycle 6)
- MiniMax M2.5 must honor `tool_choice` (standard for OpenAI-format models but unverified for Zen's routing)
- The framework must detect the capability match *before* the model turn — to know whether to force `invoke_ensemble` — which itself requires some kind of pre-decision (planner-like) step
- The chain-handling failure must dissolve when dispatch is forced — but `tool_choice` only forces the FUNCTION CALL; the model still picks the `name` argument (which ensemble) and the `input.data` argument (what to pass)

Empirical evidence from Cycle 6:
- Cycle 6 did not test `tool_choice`. It's an unevaluated mechanism.
- The chain-handling failure (note 15, OpenCode probe 4) occurred *despite* `invoke_ensemble` being called — the failure was inside the dispatch decisions, not in whether to dispatch
- `tool_choice` could force individual dispatches but cannot enforce correct input-passing across chained dispatches

Verdict: **Partial fix — addresses dispatch-vs-direct-completion, does not address chain-handling.** `tool_choice` requires a pre-decision step (when to force vs. when to allow direct completion); that pre-decision is itself a small routing-planner. So `tool_choice` is not a clean alternative to a routing-planner — it's a complementary mechanism that could live alongside or inside one.

The interesting hybrid: framework runs a small classifier that decides "is this a capability-matched request?" → if yes, force `tool_choice` to `invoke_ensemble` AND pre-populate the ensemble name → if multi-step, run a framework-driven chain. The chain step still requires framework-driven data-passing because `tool_choice` doesn't help with composition.

### Fine-tuning the orchestrator-LLM

The premise: a model fine-tuned on capability-dispatch examples reliably routes to `invoke_ensemble` and handles chains correctly.

What would have to be true:
- Sufficient training data exists (currently: a handful of cycle probes — far too few)
- The fine-tuning preserves the model's other capabilities (typical for instruction tuning)
- The fine-tuned model behaves consistently across tool-rich and tool-less configurations
- The fine-tuning is cost-feasible
- The provider supports fine-tuning (Zen's support unknown; local qwen3 models support LoRA/QLoRA)
- Chain-handling improves with training data that demonstrates correct input-passing across dispatches

Empirical evidence from Cycle 6:
- Not tested. The cost shape and feasibility are conditional on practitioner infrastructure choices (cloud vs. local).
- Cycle 4 PLAY's reliability profile (high-on-derivable, low-on-integration) is a foundation-model behavior; fine-tuning could shift it but the depth of the failure mode is uncertain.

Verdict: **Complex, partial, untested.** Fine-tuning could improve routing accuracy with sufficient training data, but the chain-handling failure is across-the-board orchestrator behavior that may not respond to fine-tuning without significant invested effort. Not a strong-enough alternative to displace framework-driven chaining for the chain-handling problem. Could become a Cycle 8+ research thread if cycle 7's framework-driven pipeline ships and proves out, but isn't competing with it.

### Synthesis of the belief-map

**The chain-handling failure is the load-bearing differentiator.** None of the named alternatives address it cleanly:
- Extended system-prompt: doesn't address chain-handling
- `tool_choice` constrained decoding: forces dispatch, doesn't enforce correct chain-step data-passing
- Fine-tuning: could plausibly help but uncertain depth, complex to operationalize

**Framework-driven chaining is the only approach where the chain-step data-passing is deterministic** — and Spike δ proved this empirically (n=1, but with a clean comparison against the orchestrator-LLM-driven baseline that failed on the same composition).

**The routing-decision is more contested.** `tool_choice` constrained decoding addresses routing-decision-under-tool-rich without requiring a full architectural pipeline. Extended system-prompt might address it (untested). Fine-tuning might address it (untested, uncertain depth).

**Implication for Cycle 7 scope:** The architectural pivot is best framed as **"framework owns chain-handling; routing-decision is open-question"** rather than "framework owns everything." The cycle 7 RESEARCH phase should examine whether the routing decision belongs in a routing-planner ensemble, in framework-level `tool_choice` orchestration, or in some hybrid. Spike ε (full pipeline) should specifically test whether `tool_choice` + framework-driven chain-handling is a viable lighter-weight alternative to a full routing-planner.

The framework-driven pipeline remains the candidate but the specific division of labor between LLM-as-router and framework-as-router is open, not settled.

---

## Action 2: Test the must-delegate premise against observed data

Belief-mapping question: **what would have to be true for must-delegate to be a current-state constraint rather than a future-state aspiration?**

### Direct-completion quality comparison

Three responses to the same task ("Write a Python function that reverses a string in place"):

**Note 1** (free M2.5 + OpenCode + NL, direct completion, 10.3s): provides BOTH idiomatic (`s[::-1]` slicing one-liner) AND educational (two-pointer swap on list-of-chars) approaches. Recognizes both as legitimate. Includes example usage.

**Note 18** (paid M2.5 + OpenCode + NL, direct completion, 13.4s): provides the two-pointer with proper docstring, time/space complexity annotation, example usage with expected output, and a closing note about Python string immutability constraints. Does not surface the slicing one-liner.

**Note 7** (direct-invoke code-generator ensemble, three-agent run, 60.697s): coder produces `chars.reverse()` approach with immutability acknowledgment; critic confirms the immutability framing; synthesizer combines into a multi-paragraph response with code blocks, example usage, key considerations, and the critic's note. Does not surface the slicing one-liner.

**Qualitative comparison:** the direct-completion responses are *equal or better* than the ensemble response on this task. Note 1 in particular offered a more complete answer (both idiomatic and educational variants) than note 7 (single approach). The critic-synthesizer pass added depth that wasn't substantially different from what the direct-completion model already produced on its own.

**For this specific task, the ensemble does not deliver observably better output quality than direct completion.** It delivers different infrastructure (calibration verdict, audit, observability) but the user-facing artifact is comparable or slightly inferior.

### The value case for must-delegate (if not output quality)

What the ensemble dispatch path provides that direct completion does not:

1. **Observability infrastructure** — per-event INFO lines, tier records, calibration verdicts, audit diagnostics. Operator can see what happened. Direct completion is opaque.
2. **Calibration gate** — first-N post-hoc result-check, AUQ threshold enforcement, entropy-collapse detection. Quality safety net for cases where the model degrades.
3. **Tier escalation** — Reflect verdict can escalate to a higher-quality model. Direct completion is one-shot.
4. **Cost predictability** — defined model profiles per ensemble. Direct completion uses the orchestrator-LLM, currently MiniMax M2.5.
5. **Audit trail** — substrate artifacts on disk, retention markers, dispatch_log entries. Direct completion has no durable trace.
6. **Multi-agent reasoning** — critic-synthesizer pass for complex tasks (code review, security review, complex synthesis). Direct completion is single-pass.

For *simple-and-common* tasks (string reverse, factual lookup, quick code snippet), points 1-5 are infrastructure value but not *output-quality* value. Point 6 is irrelevant for these tasks.

For *complex-and-stewarded* tasks (security review, multi-step analysis, audit), points 1-6 are all value. The critic-synthesizer pattern shines.

### Must-delegate as bounded claim

**Must-delegate is not a current-state constraint for all requests.** It is:
- A **current-state constraint** for the capability-matched subset (where ensembles exist and the Calibration Gate / observability / audit infrastructure has fire-paths)
- A **future-state aspiration** for the system as a whole (requires broader library coverage to be meaningful as a universal constraint)

**The actual deployment shape this implies:**

For requests that fit a capability slot:
- Must-delegate is operationally important
- Direct-completion bypass violates the architectural commitment
- The Cycle 6 PLAY findings show the orchestrator-LLM unreliably honors this

For requests that don't fit a capability slot:
- Must-delegate either forces a fallback ensemble (adds latency for marginal infrastructure value) or accepts direct completion as a residual
- The Cycle 4 PLAY note 19 "no-dispatch fallback" pattern is the operational reality
- The library coverage problem is the upstream cause

**Revised framing for Cycle 7:** the architectural commitment is **"capability-matched requests must dispatch; non-capability-matched requests may direct-complete but the path is structurally distinguished from capability dispatch."** This is a softer commitment than "all requests must dispatch" but it's empirically defensible — it constrains the load-bearing case without forcing infrastructure overhead onto cases where it doesn't add value.

The cycle 7 scope should examine whether the fallback path:
- Goes through a `general-completion` ensemble (preserves infrastructure but adds dispatch overhead)
- Stays as direct LLM completion through the orchestrator-LLM (preserves latency, loses infrastructure)
- Goes through a lightweight "direct-completion-as-ensemble" shim (cheapest middle path)

All three are viable design options.

---

## Action 3: Methodology hygiene note on cross-turn timing attribution

**Lesson learned from the probe-1 misattribution:**

When reading serve-console logs across multiple turns of an OpenCode session, chronological adjacency is insufficient to attribute a dispatch to a specific user request. Multiple inbound POSTs may complete before any dispatch fires; multiple dispatches may share a session_id; the OpenCode UI's "Build" badge timer measures something narrower than serve-side wall-clock (possibly only the final-response-streaming duration).

**Protocol for Cycle 7 PLAY observation:**

1. **Always align serve-console events to session_id.** A session_id-stable session may span many user turns; dispatches within that session can correspond to any of them.
2. **Use the OpenCode latency badge as a *user-experience-time* metric, not a *serve-side dispatch-time* metric.** The two are distinct.
3. **Match dispatches to user turns by inspection of the response content + dispatch_log + per-event log triangulation, not by raw timestamp order.** The dispatch_log entries' position relative to user message ordering is the canonical attribution.
4. **When in doubt, ask the practitioner who was present at the OpenCode UI.** They saw the sequence as it happened; serve logs are reconstructive.

The probe-1 misattribution occurred because the orchestrator-agent (me) inferred from serve-log chronological order that the 61.2s code-generator dispatch belonged to probe 1 (the NL framing prompt). The practitioner corrected: that dispatch belonged to the "Yes" follow-up turn, not probe 1. Probe 1 itself was direct LLM completion.

For cycle 7 PLAY, this protocol is in the standard PLAY toolkit. The cycle 7 observation work should treat session-id + dispatch-log + OpenCode-side narrative as the canonical attribution stack.

---

## Net adjustments to the Cycle 7 scoping

After grounding work:

1. **Framework-driven *chain-handling* is well-grounded.** Spike δ's evidence + the alternatives belief-map both point to framework as the only approach where chain-step data-passing is deterministic. The "framework owns chains" commitment can enter Cycle 7 RESEARCH as a well-motivated hypothesis.

2. **Framework-driven *routing-decision* is contested and should be RESEARCH territory, not inherited.** `tool_choice` constrained decoding is a viable lighter-weight alternative. Spike ε should specifically test "framework-driven chain-handling + `tool_choice`-forced routing-decision" alongside "full framework-driven planner + dispatch + synthesize." The cycle 7 cycle-status should reflect this as an open architectural choice, not a settled one.

3. **Must-delegate is bounded, not universal.** Cycle 7 scoping should distinguish capability-matched (must-delegate) from non-matched (may-direct-complete-via-shim-or-fallback). The "general-completion fallback ensemble" framing was over-strong; the fallback shape itself is design territory.

4. **The form-vs-content drift bifurcation may collapse.** Schema-as-enforcement (output_schema validation + reject-and-retry) could address both form-drift (claim-extractor's narrative output failing the (established)/(contested)-bullet spec) and content-drift (orchestrator-LLM dropping upstream data) in a unified way. The advisory from the susceptibility snapshot notes this; Cycle 7 DECIDE should consider unifying rather than splitting.

5. **Cross-turn timing attribution methodology is documented** and applies to Cycle 7 PLAY when it eventually fires.

**The framework-driven pipeline remains the cycle 7 candidate**, but with two important softenings: (a) the routing-decision-vs-chain-handling split is now explicit, (b) must-delegate is now bounded to capability-matched requests rather than universal.

The cycle 7 working title — *"Framework-driven orchestration: routing as code"* — could be sharpened to reflect the grounding: *"Framework-driven composition with structural routing"* or *"Routing as code, composition as code"* (or kept; titles are negotiable).

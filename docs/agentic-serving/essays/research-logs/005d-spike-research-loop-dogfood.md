# Spike (Cycle 4 Wave 3.A) — Research-Loop Dogfood via `llm-orc serve` Direct API

**Date:** 2026-05-04
**Cycle:** 4 (agentic-serving scoped corpus)
**Wave:** 3.A — behavioral spike grounding the cycle's design-method synthesis
**Method:** Code spike against the running `llm-orc serve` instance (port 8765) via OpenAI-compatible `/v1/chat/completions`
**Scope:** N=3 trials × 1 fixture (one focused research question), free-tier budget cap, single deployment shape
**Operating frame:** Outcomes over an agentic session — agent shape is means
**Cost incurred:** $0.00 (cheap-cloud orchestrator on OpenCode Zen free tier; ensemble agents on local Ollama `qwen3:0.6b`)

---

## Spike question

Can the cheap-orchestrator + ensemble pattern, dispatched via `llm-orc serve`'s direct-API `invoke_ensemble` tool surface (avoiding Spike D's opencode-CLI shell-subprocess stall), drive one bounded RDD-research-loop iteration — including specialist-agent dispatch at known transition points (lit-reviewer-style query, citation-style audit) — to a recognizable completion?

**One-line answer:** Yes for the dispatch path (autonomous routing fired correctly on N=2/3 trials, no stall, complete in ~21–32s wall-clock), with the same-tier specificity-loss caveat that Spike A surfaced now empirically re-instantiated on the orchestrator-side of the harness; and one clean diagnostic failure (N=1/3) where the cheap-cloud model fabricated a tool-result observation without emitting the underlying tool call.

---

## Method

### Approach

A small spike-only ensemble (`spike-cycle4-research-loop`) was authored to give the orchestrator a recognizable target for dispatch. The ensemble has three agents — `empirical-analyst`, `theoretical-analyst`, and a synthesizer that depends on both — all bound to `micro-local` (`qwen3:0.6b` via Ollama, free tier). Three trials were run against the running serve instance:

- **Trial 1** — autonomous routing. The user prompt named the focused research question and asked the orchestrator to "use the available ensemble library if it has a research ensemble that fits — discover it via `list_ensembles`, dispatch it via `invoke_ensemble`, then integrate the result into a recognizable research-log-style entry." The ensemble name was *not* given.
- **Trial 2** — explicit dispatch instruction. Same research question, but the prompt named both the tool (`invoke_ensemble`) and the ensemble (`spike-cycle4-research-loop`) explicitly, with the additional instruction "do not invent or paraphrase a result you did not receive."
- **Trial 3** — diagnostic. Minimal prompt: "Call `list_ensembles` right now. Output nothing else. Just the tool call. No prose, no explanation, no narration. Tool call only."

### Fixture

Focused research question (chosen narrow enough to complete in 5–10 min and meaningful for the cycle's own synthesis):

> *What conditions make pre-specifiable routing (class-c decomposition) reliable at the cheap-orchestrator tier vs. when does it degrade?*

### Deployment shape

- `llm-orc serve --port 8765` (long-running, started prior to this spike)
- Orchestrator profile: `orchestrator-minimax-m25-free` (MiniMax M25 Free via OpenCode Zen, OpenAI-compatible)
- Summarizer profile: `summarizer` (Ollama `qwen3:0.6b`)
- Ensemble agents: Ollama `qwen3:0.6b`
- Trial harness: `urllib.request` POST to `/v1/chat/completions`, non-streaming
- Response trajectory inferred from (a) the completion body, (b) `.llm-orc/artifacts/<ensemble>/<timestamp>/execution.json` for ensemble dispatches, (c) `.llm-orc/artifacts/agentic-result-summarizer/<timestamp>/execution.json` for summarizer harness invocations. The serve completion body returns the *final* assistant message only — the tool-call trajectory is observable indirectly through the artifact directory's per-call execution records.

### Free-tier policy

Honored throughout. No subscription tokens consumed. The interdisciplinary-research ensemble that exists in the project library was *not* used because its profiles (anthropologist, systems-theorist, etc.) bind to `claude-3-5-sonnet-20241022` via `anthropic-api`, a paid path.

---

## Results

### Per-trial timing and trajectory

| Trial | Wall-clock | HTTP | Ensemble dispatched? | Summarizer invoked? | Final-content length | Tool-call surface in body |
|------:|-----------:|-----:|:--------------------:|:-------------------:|---------------------:|:--------------------------|
| 1 (autonomous) | 20.7 s | 200 | yes — `spike-cycle4-research-loop` (8.26 s) | yes (1.86 s) | 1668 chars | empty (final assistant message; tool trajectory consumed pre-final) |
| 2 (explicit) | 32.1 s | 200 | yes — `spike-cycle4-research-loop` (7.45 s) | yes (1.77 s) | 2374 chars | empty (same shape as trial 1) |
| 3 (diagnostic) | 9.0 s | 200 | no — no ensemble artifact created | no — no summarizer artifact created | 91 chars (prose only) | empty |

Token usage in the completion body's `usage` block: `prompt_tokens: 0` for all three trials, `completion_tokens` reflecting the orchestrator's accumulated turn output. The zero-prompt-tokens reading is a known OpenCode Zen accounting artifact, not a real measurement; it is not load-bearing for the spike's findings.

### Trial 1 — autonomous routing succeeded

The orchestrator received only a description of the work to do (no ensemble name). It dispatched `spike-cycle4-research-loop` autonomously. The ensemble's synthesizer agent produced a structured Question / Findings / Tensions output naming "low latency, efficient protocols (e.g., QoS), and stable network infrastructure" as reliability conditions. The summarizer harness compressed this 1500-char output to a 330-char summary. The orchestrator's final research-log content **integrated the summary's content faithfully** — its bullet about "low-latency execution paths, efficient protocol overhead, and stable infrastructure conditions" maps directly onto the summarizer's compressed output.

This is the spike's clean positive: **autonomous routing fires at the cheap-cloud orchestrator tier for a one-iteration research-loop dispatch.** No explicit staging was required.

### Trial 2 — explicit instruction succeeded, with a specificity-loss artifact

Trial 2 also dispatched correctly and produced a research-log entry. But it surfaces the spike's most substantive negative finding: the orchestrator wrote, under a heading "*verbatim from ensemble return*", what it claimed was a quotation from the ensemble:

> *"The findings indicate that high network density, low traffic, and stable topology enhance pre-specifiable routing reliability, while congestion, increased traffic, or high node count degrade it. Theoretical analysis balances cost and efficiency, highlighting the interplay between routing algorithms and network structure. The execution is successful."*

This passage is **verbatim from the summarizer harness's output**, not from the ensemble. The ensemble's synthesizer agent produced richer text with section headings (`## Question`, `## Findings`, `## Tensions`), bullets distinguishing empirical and theoretical perspectives, and explicit tensions. The summarizer compressed those into the three-sentence blob the orchestrator quoted. The orchestrator labeled this blob "verbatim from ensemble return" — *honestly* describing what it received, but *misleadingly* describing what the underlying ensemble produced. This is **Spike A's specificity-loss mechanism re-instantiated on the orchestrator's view of ensemble outputs**, now empirically observed at the cheap-cloud tier and documented with concrete provenance from artifact-log evidence.

The user reading the trial-2 final response cannot tell, from the response alone, that the "verbatim" quote is actually a summarized derivative. Provenance to the actual ensemble output is broken at the harness boundary.

### Trial 3 — diagnostic failure: fabricated tool-result observation

The diagnostic prompt asked for a `list_ensembles` call and nothing else. The orchestrator's full response was:

> *"The tool call has been made and the result is displayed above as a `role:tool` observation."*

No ensemble artifact was created. No summarizer artifact was created. The 9-second wall-clock and the `role:tool` reference suggest the model had the tool schema (it knows the chat-completions tool/role conventions), but instead of *emitting* a tool call it *narrated* one as already-completed. This is the cheap-cloud orchestrator confabulating a phantom observation that never occurred.

Without runtime-side debug logging, I cannot confirm whether the model emitted a malformed tool call that the runtime rejected, or simply emitted prose claiming a call had occurred. Either way, the *observable* trajectory shows zero artifacts, so this is a real dispatch failure on the diagnostic prompt.

### Output quality (trials 1 and 2)

Both research-log entries meet a recognizable bar for *one bounded research-loop iteration on a focused question* — the structural shape is right (Question / Findings / Tensions), the content is substantive, and the orchestrator integrated the dispatched ensemble's compressed result into prose that doesn't read as fabrication. Quality is bounded by the underlying ensemble's depth (3 agents on `qwen3:0.6b` — small models, brief outputs) and by the summarizer harness's compression. As an *RDD-research-loop iteration on the agentic-terminal-task category*, the entries are recognizable but lightweight; they would not substitute for a Wave-1 lit-review or a citation-audited essay. They demonstrate the dispatch chain works end-to-end at this scale, not that the cheap tier produces frontier-quality research output.

### Failure modes observed

1. **Phantom tool-result observation (trial 3).** The cheap-cloud orchestrator at minimum 1/3 prone to emitting prose claiming a tool call has occurred when it has not. Mitigation hooks: runtime-side enforcement could detect "I called X" prose without an actual tool-call structured output and either re-prompt or fail loud.

2. **Specificity-loss between ensemble output and orchestrator's view (trials 1 & 2).** The Result Summarizer Harness's output is what the orchestrator integrates; any "verbatim" claim about ensemble content is in fact a verbatim claim about the *summary's* content. This is Spike A's mechanism, now load-bearing on the dispatch path. Mitigation hooks: pin a reference to the underlying execution.json in the orchestrator's tool-result observation so downstream prose can cite-and-quote authoritative source rather than the summary; teach the orchestrator that the summary is a summary; and/or make summarization optional for short-enough ensemble outputs (the `spike-cycle4-research-loop` synthesizer output was ~600 chars — below the threshold where context rot is a concern, yet it was summarized anyway).

3. **No autonomous failure mode for routing decision under-specification.** Across N=2 trials where the orchestrator was asked to route, it did so. No hallucinated ensemble names; the one ensemble it picked (`spike-cycle4-research-loop`) was the only research-shaped match in the available library and was correctly identified. Caveat: the library has only one obviously-research-shaped free-tier ensemble; routing-among-multiple-research-ensembles is not tested by this spike.

---

## Findings

### What this spike establishes

1. **The architecture's intended primitive works end-to-end at the deployment shape under test.** `llm-orc serve` direct-API path → orchestrator's closed five-tool surface → `invoke_ensemble` dispatch → ensemble execution → Result Summarizer Harness → orchestrator integration → final assistant content. No stall (cf. Spike D's opencode-CLI finding). One round-trip in 21–32 s wall-clock for trials with one ensemble dispatch. This closes the gap Spike D pilot identified — the architecture's primitive is reliable when the orchestrator runs *inside* `llm-orc serve` rather than *outside* via `opencode run`'s Bash tool.

2. **Autonomous routing fired at known transition points on N=2 of 3 trials at the cheap-cloud tier.** This is direct evidence on Cycle 3 grounding action 2 — the cheap-orchestrator's reliability at specialist-agent dispatch decisions on the agentic-terminal-task category. With N=3, the spike cannot quantify reliability; it can only show that the failure rate is non-zero (1/3 = phantom observation), and that the success cases are well-formed. The autonomous-vs-explicit comparison (trial 1 vs trial 2) shows both fired correctly when the prompt clearly demanded research work; the diagnostic failure (trial 3) suggests the cheap-cloud model is more reliable when the *task structure* implies the dispatch than when the *tool call itself* is the demand.

3. **The closed five-tool surface (ADR-003) sufficed for one research-loop iteration.** The orchestrator did not request `compose_ensemble`, `query_knowledge`, or `record_outcome` (the three not-yet-wired tools), and it did not ask for any tool outside the closed set. The single research-loop iteration was satisfied by `invoke_ensemble` alone (and possibly `list_ensembles` in trial 1's autonomous routing path, though I cannot confirm that from artifact logs since `list_ensembles` is internal and produces no artifact).

4. **Long-horizon session continuity did NOT become a binding constraint within ONE research-loop iteration.** No turn-budget or token-budget exhaustion. No context-rot symptoms beyond the specificity-loss artifact, which is a *single-turn harness* property, not a *multi-turn accumulation* property. This bears directly on a hook the cycle's synthesis flagged: long-horizon reliability infrastructure (Wave 2.A literature) appears as a binding constraint at multi-iteration scale, not at single-iteration scale. The spike does not refute the multi-iteration concern; it only narrows the scope where it lives.

### What this spike establishes negatively (Spike A's mechanism, observed)

5. **The Result Summarizer Harness's interposition introduces specificity-loss on the orchestrator's view of ensemble outputs — at the cheap-cloud orchestrator tier, this manifests as the orchestrator misattributing "verbatim from ensemble return" to a *summary* of the ensemble's actual output.** Wave 2.B ensemble-design literature plus Spike A (Cycle 2) flagged this risk; this spike confirms it on the dispatch path. The mitigation matrix from the Findings section above stands.

### What this spike does NOT establish

- **Multi-iteration research-loop continuity.** N=1 iteration per trial. Whether the orchestrator can chain multiple research-loop iterations, accumulate findings across iterations, and avoid context rot at iteration boundaries is the next question — and it is the question the cycle's North-Star benchmark ("drive a full RDD cycle using the agentic-serving flow itself") actually demands.
- **Routing-among-multiple-research-ensembles.** The library had one research-shaped free-tier ensemble at the time of the spike; routing under genuine ambiguity is untested.
- **Frontier-tier comparison.** Per the spike's bounded scope (N=1–3 free-tier only), no frontier-tier facsimile was dispatched. Whether Sonnet 4.6 would have made the same dispatch decisions, the same fabrication mistake on the diagnostic prompt, or produced research-log output at a higher quality bar is open. The cycle's central question is still cheap-orchestrator + orchestration vs. expensive frontier model; this spike only addresses the left side of that comparison.
- **Quality of the produced research-log entry as a contribution to the cycle's actual research record.** The trial outputs are demonstrations that the dispatch chain works, not contributions the cycle would adopt as findings on the question they pose. The underlying ensemble (3 × `qwen3:0.6b`) is too small for that.

### Hooks for the cycle's design-method synthesis

- **The closed five-tool surface holds for single-iteration work.** No new primitive demanded by the spike's findings.
- **The Result Summarizer Harness needs a provenance hook.** The orchestrator's "verbatim from ensemble return" misattribution is correctable if the harness's summary carries a stable pointer to the underlying execution artifact and the orchestrator's system prompt teaches it the difference between summary and source. This is a class-(c) intervention — pre-specifiable, no runtime adaptivity required — and aligns with the cycle's converging "class (c) decomposition is the dominant intervention class" finding.
- **The phantom-tool-call failure mode is a class-(c) target as well.** Pre-specifiable runtime-side check: if the orchestrator's prose mentions a tool call but no tool call structure was emitted, fail loud or re-prompt. This is the same shape as the mixed-batch-rejection guard already present in `orchestrator_runtime.py`.
- **Wave 2.B's "ensembles benefit when their outputs are inspectable, not just summarizable" position is empirically reinforced.** The compression path is the primary gap on the dispatch side, not the dispatch decision itself.
- **Long-horizon reliability infrastructure (Wave 2.A) is not yet the binding constraint at single-iteration scale.** It will be at multi-iteration scale; the cycle should plan its later spikes accordingly.

---

## Recommendation

### For the cycle's design-method synthesis

1. **Treat single-iteration `invoke_ensemble` dispatch as a *validated* primitive at the cheap-cloud orchestrator tier.** The spike shows the path works without explicit staging on N=2/3 trials. The cycle's synthesis can lean on this empirically rather than as architectural intent.

2. **Treat the Result Summarizer Harness's specificity-loss as the next class-(c) target.** Wave 2.B literature plus Spike A motivated it conceptually; this spike documents it concretely on the dispatch path. Two pre-specifiable interventions are within reach: (a) summary-with-pointer (orchestrator can dereference if needed), (b) skip-summarization-when-short (reduce specificity loss when context-rot risk is low). Both are class-(c) — no runtime adaptivity, no new primitives.

3. **Treat the phantom-tool-call failure mode as a runtime-side guard rather than a model-side correction.** The cheap-cloud model will sometimes emit prose-claiming-a-tool-call. The architecture can detect and reject this pattern at the runtime boundary, the same shape as the mixed-batch-rejection path.

4. **Plan the *next* spike at multi-iteration scale.** The North-Star benchmark demands a full RDD cycle, not a single research-loop iteration. This spike clears the single-iteration question; the binding constraints at multi-iteration scale (cumulative summarization loss, turn-budget pressure, routing-among-multiple-research-ensembles, cross-iteration finding integration) are open. Recommended scope for that spike: N≥3 chained research-loop iterations on a single research thread, with explicit checks for context rot and provenance preservation across iteration boundaries.

5. **Defer frontier-tier comparison until the multi-iteration spike.** The cycle's central question ("cheap-orchestrator + orchestration vs. expensive frontier model") is most informatively answered when the *full agentic flow* is the unit of comparison, not a single iteration. This spike confirms the cheap side's per-iteration dispatch works; the comparison-of-interest is at the cycle scale.

### For the spike's own follow-ups (if any)

The spike is bounded; no follow-up trials are required. Artifacts retained at `scratch/spike-cycle4-research-loop-dogfood/` per the corpus's spike-artifact retention policy. The `spike-cycle4-research-loop.yaml` ensemble is intentionally left in `.llm-orc/ensembles/` so a future multi-iteration spike can reuse it.

---

## Spike scope conditions

- **N:** 3 trials, each posting to `/v1/chat/completions` once with non-streaming response
- **Fixture:** single focused research question (named above)
- **Budget cap:** free-tier only; cost incurred $0.00
- **Deployment shape:** `llm-orc serve --port 8765` direct-API only; no `opencode run`, no `llm-orc invoke` shell path
- **Tier:** cheap-cloud orchestrator (MiniMax M25 Free via OpenCode Zen) + local Ollama (`qwen3:0.6b`) for ensemble agents and summarizer
- **Trajectory inference:** combination of completion-body content, ensemble execution artifacts (`.llm-orc/artifacts/spike-cycle4-research-loop/`), and summarizer artifacts (`.llm-orc/artifacts/agentic-result-summarizer/`); runtime-side debug logs not consulted

## Artifacts retained (per corpus spike-retention policy)

- `scratch/spike-cycle4-research-loop-dogfood/run_trial.py` — trial-1 runner (autonomous routing)
- `scratch/spike-cycle4-research-loop-dogfood/run_trial2_explicit.py` — trial-2 runner (explicit dispatch)
- `scratch/spike-cycle4-research-loop-dogfood/run_trial3_diagnostic.py` — trial-3 runner (diagnostic)
- `scratch/spike-cycle4-research-loop-dogfood/spike-cycle4-research-loop.yaml` — ensemble config snapshot
- `scratch/spike-cycle4-research-loop-dogfood/trials/trial-1/` — raw response, summary, final content
- `scratch/spike-cycle4-research-loop-dogfood/trials/trial-2-explicit/` — raw response, summary, final content
- `scratch/spike-cycle4-research-loop-dogfood/trials/trial-3-diagnostic/` — raw response, summary
- `scratch/spike-cycle4-research-loop-dogfood/artifacts/20260504-150212-038/` — trial-1 ensemble execution
- `scratch/spike-cycle4-research-loop-dogfood/artifacts/20260504-150314-144/` — trial-2 ensemble execution
- `scratch/spike-cycle4-research-loop-dogfood/artifacts/summarizer-trial1/` — trial-1 summarizer execution
- `scratch/spike-cycle4-research-loop-dogfood/artifacts/summarizer-trial2/` — trial-2 summarizer execution

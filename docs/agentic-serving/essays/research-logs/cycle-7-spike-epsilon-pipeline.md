# Cycle 7 DISCOVER — Spike ε: End-to-End Plan→Dispatch→Synthesize Pipeline

*2026-05-21*

## Purpose

DISCOVER-phase validation spike (per ADR-087 validation-spike-as-research-method precedent established in Cycle 7 RESEARCH). Tests whether the **end-to-end framework-driven pipeline** — Spike ζ's routing-planner + framework-driven dispatch + a new response-synthesizer ensemble — dissolves the confabulation pattern that the orchestrator-LLM exhibited at PLAY note 22, while preserving Spike δ's positive-control fidelity on deterministic chains.

The architectural question being probed:

> When the orchestrator-LLM is removed from the dispatch path entirely and replaced by a routing-planner (Spike ζ) + a response-synthesizer (Spike ε), does the confabulation pattern dissolve? Does fidelity hold across both single-step and chained compositions?

Spike ζ already established mechanism viability for the routing decision (100% JSON conformance + 90% strict capability-match). Spike ε tests the full pipeline — including the second new component, the synthesizer — against the historical failure case (PLAY note 22) and the positive control (Spike δ's web-searcher → claim-extractor chain).

## Method

### Components

- **Routing-planner** — `spike-cycle7-zeta-routing-planner` (qwen3:8b; the Spike ζ ensemble, unchanged).
- **Response-synthesizer** — `spike-cycle7-epsilon-response-synthesizer` (qwen3:8b; new). Input contract is a structured text blob with three ALL-CAPS section headers: `ORIGINAL REQUEST`, `PLAN` (Dispatched + Planned-but-not-run lists), `DISPATCH RESULTS` (verbatim ensemble outputs). System prompt enforces five strict-fidelity rules: use only DISPATCH RESULTS content; do not fabricate results for Planned-but-not-run ensembles; do not invent operational metadata; cite specific figures verbatim (no rounding); honest direct-completion framing when DISPATCH RESULTS is empty.
- **Capability ensembles** (existing, unchanged from Cycle 6 minimum-viable set) — `agentic-serving/web-searcher` (script-agent via ddgs); `agentic-serving/claim-extractor` (qwen3:8b).

### Pipelines under test

**Pipeline A: planner-driven single-step.**

    1. Plan: invoke routing-planner with request content.
    2. Dispatch: invoke the planner-named ensemble. Detect explicit chain
       language in the request; enumerate as "planned-but-not-run" any
       ensemble the user named that did not execute.
    3. Synthesize: invoke response-synthesizer with the structured
       (ORIGINAL REQUEST / PLAN / DISPATCH RESULTS) input.

**Pipeline B: deterministic chain + synthesizer.**

    1. Dispatch each ensemble in a hard-coded chain (Spike δ pattern).
       Step N's output is step N+1's input.
    2. Synthesize: response-synthesizer with full chain context.

### Test battery

Three tests, run free-tier against local Ollama:

| Test ID | Pipeline | Prompt | Purpose |
|---|---|---|---|
| T1 | A | *"Use the web-searcher capability ensemble to find information about the current population of Iceland, then use the claim-extractor capability ensemble on the results."* | The PLAY note 22 prompt that confabulated under orchestrator-LLM dispatch. The hardest test — a multi-step composition where only the first step actually runs under the single-step pipeline. |
| T2 | B | Initial input: *"current population of Iceland"*; chain: `web-searcher → claim-extractor` | Spike δ's exact chain with the synthesizer added at the end. Positive control: does the synthesizer preserve fidelity to the chain's output? |
| T3 | A | *"What is the current population of Iceland?"* | Simple single-capability question. Cleanest test of the basic plan→dispatch→synthesize flow. |

### Working directory

`scratch/spike-epsilon-pipeline/`:

- `harness.py` — Python pipeline implementation, runs all three tests, writes results.json
- `results.json` — full audit trail (planner output, dispatch results, synthesizer input, synthesizer output, latency per stage, fidelity analysis per test)

### Fidelity analysis

Per Spike δ's pattern: extract distinct multi-digit numbers (`\b\d[\d,]{2,}\b`) from the synthesizer's final response vs. from all dispatch outputs. Compute overlap and any final-response numbers absent from sources (the "fabricated" set).

## Results

### Summary

| Test | Verdict | Final numbers | Source numbers | Fabricated | Total latency |
|------|---------|--------------|---------------|------------|---------------|
| T1 — PLAY note 22 under Pipeline A | OK | 4 | 13 | **none** | 36.2s (13.6 + 5.0 + 17.6) |
| T2 — Spike δ chain + synth (Pipeline B) | OK | 5 | 8 | **none** | 64.0s (2.9 + 36.6 + 24.5) |
| T3 — simple lookup under Pipeline A | OK | 7 | 14 | **one** (`402,300`) | 41.6s (12.3 + 2.3 + 27.0) |

Total agent work elapsed: ~3 min total wall clock; $0.00 cost (all local Ollama; ddgs free DDG adapter).

### T1 — PLAY note 22 composition prompt

The planner's decision (unchanged from Spike ζ's P19 measurement): `{"action": "dispatch", "ensemble": "web-searcher", "rationale": "The request explicitly asks to use the web-searcher capability ensemble first."}` — correctly identifies the FIRST capability per the planner's composition-handling rule.

The framework dispatched web-searcher only. The harness detected the explicit naming of `claim-extractor` in the original request and enumerated it in `planned_but_not_run`. The structured synthesizer input therefore reported:

    Dispatched: web-searcher
    Planned-but-not-run: claim-extractor

The synthesizer's final response began:

> The request asked for the claim-extractor ensemble to process the web-searcher results, but only the web-searcher ensemble was executed. Here is the relevant information from the web search: [...]

Then presented the actual web-searcher data with citation: "**355,632** as of May 10, 2026, according to [Countrymeters](https://countrymeters.info/en/Iceland)". The cited population figure, date, and source all appear verbatim in the web-searcher response.

**This is the exact behavior the strict-fidelity rules were designed to enforce.** Rule 2 (do NOT fabricate results for Planned-but-not-run; tell the user directly) fired cleanly. Compare against PLAY note 22's actual orchestrator-LLM output, which fabricated "*The claim-extractor identified the key claims from the web-search results, including that Iceland's population is around 388,000 and growing at about 1.3% annually*" — a non-existent dispatch with a fabricated number.

### T2 — Spike δ chain with synthesizer

Web-searcher → claim-extractor ran as in Spike δ. The synthesizer received both outputs and produced:

> The current population of Iceland is reported differently across sources, reflecting varying methodologies and timelines:
>
> - **354,000** (as of 2024) is cited by Populationof.net and Countrymeters, with the latter noting a 2026 estimate of **354,751** (possibly a projection).
> - A **more recent estimate** from Iceland Review (Q3 2024) reports **388,790**, including **17.3% foreign nationals**.
>
> The discrepancy likely stems from whether foreign residents are counted. For the latest figure, **388,790** (including temporary residents) is the most recent data available, though the exact methodology may affect interpretation.

All cited numbers (354,000; 354,751; 388,790; 17.3%) appear in the dispatch output. Source attributions (Populationof.net, Countrymeters, Iceland Review) are accurate. The synthesizer added organization (a "Key Discrepancies" structure) without introducing new content — the discrepancy framing is itself derivable from the source figures.

**0 fabrications.** Synthesizer preserves the fidelity that Spike δ established for framework-driven chaining.

### T3 — Simple lookup

Planner output: `{"action": "dispatch", "ensemble": "web-searcher", "rationale": "The request is a factual question requiring current information, which falls under the web-searcher capability."}` — correct dispatch decision.

Web-searcher returned five sources with a range of population figures (Wikipedia: 389,444 as of 2025; Worldometer: 402,329 mid-year 2026; Worldpopulationreview: 402.3 thousand; Countrymeters: 354,751 as of 2026-01-01; iceland.org: ~383,000 as of 2024).

Synthesizer's final response cited 402,329 (verbatim from Worldometer), 402.3 thousand (verbatim from Worldpopulationreview), 389,444 (verbatim from Wikipedia), 354,751 (verbatim from Countrymeters) — but also wrote *"~402,300 residents is the widely cited number"*. The string `402,300` does NOT appear in any source; it is a rounding of `402,329` (or a rewriting of "402.3 thousand"). This violates Rule 4 (no rounding — use the exact value).

**One Rule 4 violation.** Not a confabulation (no fabricated source attribution, no invented dispatch), but a fidelity violation per the synthesizer's strict-fidelity contract.

## Findings

### Finding ε.1 — Confabulation pattern dissolved on the structural axis

The most consequential finding: the orchestrator-LLM's PLAY note 22 confabulation does NOT survive the architectural move to a structurally bounded synthesizer role. The same model class (qwen3:8b is in the same cheap-tier family as the qwen3:14b that confabulated; cheap-tier is more, not less, susceptible to confabulation per scaling intuition) produces faithful output when:

- Its role is structurally bounded to "produce response from given context"
- The dispatch decisions are made elsewhere (planner) and the dispatch execution is deterministic (framework)
- The strict-fidelity rules are made explicit in the system prompt
- Planned-but-not-run is reported honestly to the synthesizer

This refutes the model-capability framing of the confabulation problem ("the orchestrator-LLM hallucinates because it's a cheap model") and supports the structural framing ("the orchestrator-LLM hallucinates because its role bundles routing decisions with multi-dispatch narration, and the latter is integration-claim territory per Cycle 4/6 PLAY"). Spike ε is the architectural complement to Spike ζ's mechanism-viability finding: routing-planner viability (ζ) + structurally-bounded synthesizer viability (ε) together motivate ADR-027's framework-driven dispatch pipeline as the production target.

This is consistent with Spike ζ Finding ζ.5: the orchestrator-LLM's failure surface is in multi-step composition narration, not in single-decision tasks. The synthesizer's task — produce a response from a fixed set of dispatch results — is a single-decision-shaped task; even though the response is prose, the decision boundary is "what to say given this input", not "what to do next given partial information". The model handles single-decision-shaped tasks reliably.

### Finding ε.2 — Strict-fidelity rules in the system prompt do real work

Rule 2 (do NOT fabricate Planned-but-not-run; tell the user directly) fired exactly as designed on T1. The synthesizer's opening sentence — *"The request asked for the claim-extractor ensemble to process the web-searcher results, but only the web-searcher ensemble was executed"* — is a direct realization of the rule's example wording. The system prompt's contract shapes synthesizer behavior in measurable ways.

This is the operational answer to Cycle 6 WP-D moderate advisory 1: documentary output_schema does not enforce compliance, but **a strict-fidelity contract in the agent's system prompt, paired with a structured input format that names what was and wasn't run, does shape compliance for the synthesizer role.** This is distinct from output-shape enforcement (claim-extractor's form drift, which Spike ε does not address — see ε.4) but is the operative form of "schema enforcement" for the synthesizer's free-prose output.

### Finding ε.3 — Cheap-tier failure mode under strict-fidelity is rounding, not confabulation

The single T3 violation (`402,300` ≠ source `402,329` / `402.3 thousand`) is qualitatively different from PLAY note 22's confabulation. It is a Rule 4 violation — restating a present figure with rounded precision — not a Rule 1 violation (introducing content absent from sources). The synthesizer's instinct to write "approximately X" when sources gave "X" is the operational floor for cheap-tier strict-fidelity output.

The implication for DECIDE: production strict-fidelity synthesis at the cheap tier should anticipate restatement/rounding drift on numerical figures, even with explicit no-rounding rules. Mitigation options worth considering in DECIDE:

- **Tier escalation for synthesis on requests with high numerical precision requirements** (capability-aware tier selection — the orchestrator's existing Calibration Gate could route to qwen3:14b for high-precision numerical synthesis).
- **Post-hoc number-overlap check** (the harness's fidelity analysis as a runtime fitness check; if any final-response number is absent from sources, log a drift event for operator review or trigger a re-synthesis).
- **Structured response format for high-precision domains** (when the dispatch output is structured JSON, the synthesizer could be asked to emit a structured response with figures-as-fields rather than figures-in-prose; the structured surface is less prone to rounding rewriting).

These are tuning concerns per practitioner framing 2026-05-21; not architectural blockers. The mechanism — strict-fidelity rules + structured input — works at the cheap tier with one drift mode that is observable and addressable.

### Finding ε.4 — Form drift remains independent of the orchestration question

Claim-extractor's output in T2 was structured prose with "Key Discrepancies" headers, not the `(established)/(contested)` bulleted form the YAML default_task declares. Same form drift Spike δ documented; same form drift Cycle 6 PLAY documented. **Spike ε does NOT address this.**

The synthesizer's strict-fidelity rules govern whether the synthesizer faithfully passes through the claim-extractor output, not whether the claim-extractor output itself matches its declared format. Form drift at the capability-ensemble layer is a separate question — Cycle 7 Q2 (I/O contract enforcement) territory. ADR drafting on this should reference Spike ε's confirmation that the orchestration architecture is independent of the form-drift question; framework-driven plan→dispatch→synthesize works the same regardless of whether the dispatched ensemble's output conforms to its declared schema.

### Finding ε.5 — Latency is real and tiered

End-to-end latency under Pipeline A (planner-driven single-step) is ~35-45s. Under Pipeline B (deterministic chain + synth) it is ~60-70s. Compare against bare-LLM latency on Iceland questions (~10s for qwen3:14b per Cycle 6 / Cycle 7 RESEARCH context). The pipeline imposes a 3-7× latency multiplier in this configuration.

Per-stage breakdown:

| Stage | Latency range | Driver |
|---|---|---|
| Plan (routing-planner) | 12-14s | qwen3:8b reasoning + JSON output; consistent with Spike ζ's 10s mean |
| Dispatch (web-searcher) | 2-5s | DDG fetch; script-agent has no LLM cost |
| Dispatch (claim-extractor) | 36s | qwen3:8b synthesis over web-searcher output |
| Synthesize | 17-27s | qwen3:8b synthesis with strict-fidelity rules and structured input |

The synthesizer is comparable to claim-extractor's per-call latency, as expected (both are qwen3:8b synthesis tasks). The pipeline's latency stack is additive, so the architectural decision about whether the pipeline runs synchronously on every chat-completions request vs. only on capability-matched requests vs. behind a fast classifier pre-filter directly determines the per-request latency floor.

This restates Spike ζ Finding ζ.2 (latency is real but tunable) at the full-pipeline level. Per practitioner framing 2026-05-21, latency is a tuning concern not a structural constraint, so the spike does not block on this dimension. Tuning options for DECIDE:

- Classifier pre-filter (front-load the "should we engage the pipeline?" decision before the planner runs)
- Cached planner decisions on common request shapes
- Smaller faster planner model (qwen3:1.7b or 0.6b — re-validate JSON conformance per Spike ζ Finding ζ.2's open variant)
- Concurrent dispatch (when planner outputs multi-step plans — pipeline doesn't currently support this; Spike ε's deterministic chain is sequential)
- Streaming synthesis (the synthesizer's response could stream while the user reads — partially masks the 17-27s synth cost; the chat-completions API surface supports streaming natively)

### Finding ε.6 — Multi-step composition is currently external to the planner

Spike ε's Pipeline A makes a single dispatch per request (per the Spike ζ planner's design). Multi-step composition requires either:

- Pipeline B (deterministic chain, hardcoded in the framework — used in T2)
- A multi-step planner (the Spike ζ planner would need re-engineering to emit a list of dispatches)
- A planner that runs multiple times with augmented context (LLM-in-the-loop chaining, which is what we're trying to avoid)

The PLAY note 22 prompt is an interesting boundary case: it explicitly names a two-step chain, and the current Pipeline A handles it by running step 1 and honestly reporting that step 2 was not run. This is structurally correct (the user gets an honest response) but functionally incomplete (the user asked for two-step processing and only got one).

For DECIDE: the question of whether the production planner should emit single-step or multi-step plans is a real architectural design decision. Single-step is simpler and matches Spike ζ's validated form; multi-step requires re-validation but matches user expectations on explicit-naming composition prompts. A reasonable middle ground: single-step planning by default, with the synthesizer empowered to signal "the user asked for a chain; the next step would be X" — keeping the planner simple while making composition gaps observable for downstream tuning.

## Implications for DECIDE

1. **ADR-027 framework-driven dispatch pipeline is empirically supported.** The structural decomposition (planner + dispatch + synthesizer, with the orchestrator-LLM removed from the dispatch path) dissolves the PLAY note 22 confabulation pattern at the cheap tier. DECIDE can structurally commit to ADR-027 as the production target without contingent fallback architecture, modulo the cost/latency profile evaluated in BUILD/PLAY.

2. **Strict-fidelity rules in the synthesizer system prompt are the operational shape of "contract enforcement" for free-prose responses.** Documentary output_schema does not enforce; system-prompt strict-fidelity rules do shape behavior. DECIDE ADR on I/O contract enforcement should distinguish these two surfaces: (a) ensemble-output structural schema (form drift territory; needs schema-as-enforcement or task-rewrite, separate question), and (b) synthesizer free-prose fidelity (rule-based system prompt with structured input format, as Spike ε demonstrates).

3. **Cheap-tier synthesis fails by restatement/rounding, not by full fabrication.** The post-hoc fitness check (number-overlap analysis) is a viable operator-observable drift signal. DECIDE could ship this as part of the production pipeline's instrumentation surface — log every fidelity-failed synthesis for operator review; the operator decides whether to escalate to a higher tier or accept the drift for the request shape.

4. **Multi-step composition is a real DECIDE design question.** The planner's current "FIRST capability of a composition" rule is workable for simple cases and produces honest "step 2 not run" reporting via Pipeline A, but is functionally incomplete for explicit-chain prompts. Three architectural options: (a) keep single-step + honest signaling (Spike ε Pipeline A baseline); (b) multi-step planner with re-validation; (c) planner-emits-single-step + framework-driven chain heuristic that detects multi-step intent and orchestrates additional dispatches deterministically. Each has tradeoffs; the choice depends on how often production traffic includes explicit-chain prompts vs. NL-inferred-chain needs.

5. **Latency tuning is the cycle's most concrete BUILD-phase work item.** The pipeline's additive latency stack (planner + dispatch + synth + optional chain) is the main user-visible cost. Specific tunable axes are well-characterized at this point; DECIDE should land an ADR with measurement criteria (target latency p50/p90 thresholds) and a tuning playbook (classifier pre-filter → smaller planner → cached decisions → streaming synth) so BUILD/PLAY measurement has concrete targets.

## Cross-references

- **Spike ζ** (`cycle-7-spike-zeta-routing-planner.md`) — routing-planner mechanism viability. Spike ε uses ζ's planner unchanged.
- **Spike δ** (`cycle-6-spike-delta-framework-chaining.md`) — framework-driven chaining fidelity. Spike ε's T2 is δ's chain + the new synthesizer; ε confirms δ's fidelity property is preserved when the synthesizer is added at the end.
- **PLAY note 22** (`reflections/field-notes.md` Cycle 6 PLAY section) — the historical confabulation case. Spike ε's T1 is the same prompt under the new pipeline; the confabulation does not survive.
- **Essay-Outline 006** (`essay-outline-006-cross-compatibility-routing-surface.md`) — the Cycle 7 RESEARCH artifact that frames the hybrid + ADR-027-escalation architecture. Spike ε's findings support tighter-than-conditional commitment to ADR-027 (Finding ε.1) — see GT-2 grounding reframe carry-forward.
- **Cycle 7 RESEARCH GT-2 (hybrid-first ordering language drift)** — Finding ε.1 contributes evidence to the GT-2 follow-up (build-complexity comparison between Tier 1 hybrid and ADR-027-direct). With Spike ε showing ADR-027's structural axis works at the cheap tier, the build-complexity question becomes more tractable: ADR-027's component costs are now empirically grounded (planner ensemble + synthesizer ensemble — both implementable as YAML + system prompt; no new framework code beyond plan→dispatch→synthesize plumbing).

## Spike artifacts

Retained until agentic-serving corpus close per `feedback_spike_artifact_retention` directive:

- `.llm-orc/ensembles/spike-cycle7-epsilon-response-synthesizer.yaml` — the response-synthesizer ensemble with strict-fidelity rules
- `scratch/spike-epsilon-pipeline/harness.py` — Python harness implementing Pipeline A and Pipeline B
- `scratch/spike-epsilon-pipeline/results.json` — full audit trail (planner output, dispatch results, synthesizer input, synthesizer output, latency per stage, fidelity analysis per test)

## Cost record

Zero cost. Local Ollama qwen3:8b for planner, claim-extractor, and synthesizer; ddgs free DDG adapter for web-searcher. Three end-to-end test runs; ~3 min total wall clock.

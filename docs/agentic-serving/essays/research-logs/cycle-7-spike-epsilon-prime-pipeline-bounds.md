# Cycle 7 DISCOVER → MODEL — Spike ε': Pipeline-Bounds Closing

*2026-05-22*

## Purpose

Gate-tail validation spike (optional, named at DISCOVER → MODEL gate per Amendment A3 and Orchestrator LLM Cycle 7 refinement scope-of-claim partition). Addresses three plausible-but-untested bounds the partition flagged but Spike ε did not exercise:

- **Test family A**: synthesizer's Rule 5 direct-completion path under varied request shapes (Spike ε ran one smoke probe only).
- **Test family B**: rounding-drift base rate under numerical-dense dispatch results (Spike ε ran 1/7 figures with a single Rule 4 violation; base rate at scale unknown).
- **Test family C**: multi-turn conversational continuity under the synthesizer-only architecture (Spike ε ran single-shot requests only).

The cycle is positioned to enter MODEL with stronger empirical coverage of the synthesizer's behavior than the Spike ε 3-test base supports — particularly for the architectural commitment "orchestrator-LLM removed from the routing-decision and post-dispatch-synthesis surfaces" extending toward production-realistic shapes.

## Method

Extended Spike ε's harness (`scratch/spike-epsilon-pipeline/harness_prime.py`). Reuses the same components: routing-planner ensemble (Spike ζ; qwen3:8b), framework-driven dispatch execution, response-synthesizer ensemble (Spike ε; qwen3:8b with strict-fidelity rules). Same free-tier discipline. Runs against the same synthesizer YAML used for Spike ε.

Seven probes:

| Test ID | Family | Purpose | Pipeline shape |
|---|---|---|---|
| A1 | direct-completion | Simple conversational request | Planner-driven (planner outputs direct; synthesizer handles per Rule 5) |
| A2 | direct-completion | Factual question | Planner-driven |
| A3 | direct-completion | Ambiguous-no-capability | Planner-driven |
| B1 | numerical density | Many-figure dispatch result (5 European countries × 3 metrics — 25 figures) | Hardcoded synthesizer-only (skip planner+dispatch; use controlled fixture) |
| B2 | numerical density | Precise round-able figures | Hardcoded synthesizer-only |
| C1 | multi-turn | Context-dependent follow-up (Turn 1 establishes Reykjavik context; Turn 2 asks weather) | Two-turn planner-driven; Turn 2 includes Turn 1 in the request via inline-history serialization |
| C2 | multi-turn | Dispatch-referenced follow-up (Turn 1 looks up Iceland population; Turn 2 asks for comparison to Luxembourg) | Two-turn planner-driven |

**Multi-turn mechanism note**: the pipeline does not currently handle the OpenAI chat-completions `messages[]` array natively — each `pipeline_planner_driven` call serializes the request into a single ORIGINAL REQUEST blob for the synthesizer. The C-family probes test the simplest mitigation: inline-serialize the prior turn into the second turn's request body as `[Prior conversation: user said: "X". Assistant responded: "Y"]\n\nCurrent user message: ...`. A production implementation would need to handle `messages[]` natively in the routing-planner and synthesizer inputs.

Working directory: `scratch/spike-epsilon-pipeline/`:
- `harness_prime.py` — Spike ε' test functions (reuses `harness.py` infrastructure)
- `results_prime.json` — full audit trail

Cost: $0.00 (local Ollama qwen3:8b throughout; ~3 min total wall clock).

## Results

### Summary

| Test | Verdict | Key observation |
|------|---------|-----------------|
| A1 — simple conversational direct-completion | OK | Honest direct-completion answer; **Rule 5 framing omitted** ("This answer was generated directly...") |
| A2 — factual question direct-completion | OK | Direct-completion answer with one factual error from training data ("Urga (now known as Khovd)" — Urga renamed to Ulaanbaatar; Khovd is a different city); 1924 founding date approximately correct; **Rule 5 framing omitted** |
| A3 — ambiguous-no-capability | OK | Clean conversational answer about commit messages; **Rule 5 framing omitted** |
| B1 — numerical density (25 figures) | OK | All cited GDP figures verbatim ($52,746, $44,460, $38,373, $32,677, $22,113); large population figures **rendered as "millions"** (84.4 million for source's 84,358,845; 68.4 million for 68,374,591); harness number-regex misses the "X.Y million" form so drift_rate=0.00% is partial |
| B2 — precise round-able | OK | **All five precise figures preserved verbatim** (33,581 / 39,580 / 388,790 / 542,051 / 660,809) |
| C1 — multi-turn context-dependent | OK | Strong result: Turn 1 conversational Reykjavik-trip response → Turn 2 planner correctly routes to web-searcher for weather (fresh info) → web-searcher returns no current weather → synthesizer honestly says "web search didn't find current weather information" AND falls back to typical-November-Iceland conditions via Rule 5; Reykjavik/Iceland context preserved across turn |
| C2 — multi-turn dispatch-referenced | OK | Turn 1 cites dispatch-result population figures verbatim (402,042 / 397,129 / 386,506 / 402.3 thousand from web-searcher); Turn 1 has one rounding drift (~402,000 from 402,042); Turn 2 dispatches web-searcher for Luxembourg → cites 677,717 verbatim; cross-turn comparison attempted with ~1.7× ratio inference |

### Detailed analysis

**Family A — Direct-completion path under Rule 5**:

The three direct-completion probes all produced useful responses but with **Rule 5 framing systematically omitted**. The system prompt's Rule 5 specifies: *"produce a brief response indicating that no capability ensemble was dispatched and the request was handled by direct completion. Then attempt to answer the ORIGINAL REQUEST directly from your own knowledge, clearly framing it as a direct-completion answer."* None of A1/A2/A3 carried such framing — the synthesizer simply answered the question.

Two readings:
- **Rule 5 framing is desired and the system prompt should be sharpened**. Direct-completion responses should signal their nature to operators (for cost-distribution observability per Tension 18) and to clients (for trust in the transparent-endpoint promise).
- **Rule 5 framing is undesired in practice** — surfacing "this answer was generated directly without ensembles" to users every time would be noisy and break the transparent-endpoint promise. The system prompt's Rule 5 wording is over-specified.

The right resolution is a DECIDE-phase scope question; the current behavior should not be characterized as a defect without first deciding whether Rule 5's framing requirement matches the cycle's intent.

A2's factual error ("Urga (now known as Khovd)") is a training-data error in the synthesizer's direct-completion answer — not a fabrication relative to dispatch results (none were run); it's an empirical signal about the synthesizer's reliability when answering from own knowledge. The cheap tier (qwen3:8b) is appropriate for synthesis-of-dispatch-results but is **less reliable for direct-completion-of-factual-questions** without grounding. DECIDE may want to consider whether direct-completion responses should escalate to a higher tier (qwen3:14b, or a frontier model fallback), especially when the request was clearly capability-relevant but no capability matched.

**Family B — Numerical-density fidelity**:

B1's "0% drift rate per the harness" is partial. The synthesizer cited GDP figures and density values verbatim ($52,746 / $44,460 / $38,373 / $32,677 / $22,113 — all verbatim; 124.7 / 235.2 / 194.8 — verbatim) but rendered large population figures in "millions" form: "approximately 84.4 million" for source's "84,358,845"; "68.4 million" for "68,374,591"; "58.8 million" for "58,761,146". The harness's regex `\b\d[\d,]{2,}\b` matches comma-separated digit runs but does not match "84.4 million" — the rounding here is to one decimal place + units conversion, which the harness misses. This is **rounding-drift Mode 2** ("large-number rendering as millions"), distinct from Spike ε T3's **Mode 1** ("precise figure rendered as round approximation": 402,329 → ~402,300).

B2's "all five precise figures preserved verbatim" result is the clean signal: when the source presents precise figures and the prompt invites listing them, the synthesizer preserves precision. The "~" prefix the synthesizer added is hedging language, not numerical rounding — the figures themselves (33,581 / 39,580 / 388,790 / 542,051 / 660,809) are exact.

**Combined B-family characterization**: rounding drift is mode-specific, not uniform. Two distinct modes observed: (1) precise-figure rounding (Mode 1; Spike ε T3 + C2 Turn 1: "~402,000" for 402,042); (2) large-number-rendering-as-millions (Mode 2; B1's 84.4 million for 84,358,845). Both modes are reading-aid behaviors at the cheap tier. Both are addressable via:
- System prompt sharpening (Rule 4 currently says "If you find yourself wanting to round a figure... use the exact value instead" — could add explicit guidance for large numbers).
- Tier escalation for high-precision domains (Calibration Gate could route to qwen3:14b when numerical precision is load-bearing).
- Operator-observable degradation signaling (the harness's number-overlap check, extended to cover Mode 2, becomes a runtime fitness signal).

**Family C — Multi-turn continuity**:

Both C-family probes succeeded — context preserved across turns in both cases. This is the architecturally strongest finding of Spike ε':

- **C1**: Turn 1's conversational Reykjavik-trip response (rich content: Northern Lights, Golden Circle, weather hints) → Turn 2's "What's the weather like there?" → planner correctly identifies web-searcher as the capability fit (weather is fresh-info; matches planner system prompt) → web-searcher returns results without current weather data → synthesizer **honestly notes** "web search didn't find current weather information for Reykjavik" AND falls back to typical-November-Iceland conditions via direct-completion knowledge. This is the synthesizer doing exactly what the strict-fidelity rules + Rule 5 specify: dispatch-result-faithful + honest about scope + direct-completion fallback when dispatch is incomplete.

- **C2**: Turn 1's "What is the current population of Iceland?" → planner routes to web-searcher → synthesizer cites dispatch-result figures (402,042 / 397,129 / 386,506 / 402.3 thousand — all verbatim from today's DDG; the spike-current DDG fetch was verified). Turn 1 has one Mode 1 rounding drift ("around 402,000 residents" for 402,042). Turn 2's "How does that compare to Luxembourg's population?" → planner routes to web-searcher for Luxembourg → synthesizer cites Luxembourg's "peak population of 677,717 in 2024" verbatim + computes "Luxembourg's population is roughly 1.7 times larger than Iceland's ~402,000 residents" cross-turn comparison.

**Architectural significance**: multi-turn continuity works under the synthesizer-only architecture when prior turns are included in the synthesizer's input. The mechanism the probes used (inline-serialize prior turns into the next request body) is a workaround for the harness's single-shot `ORIGINAL REQUEST` format; a production implementation would handle the OpenAI `messages[]` array natively. The architectural commitment ("orchestrator-LLM removed from dispatch path entirely") holds across turn boundaries — no separate orchestrator-LLM-with-conversation-state is required; the synthesizer-with-history achieves the same outcome.

## Findings

### Finding ε'.1 — Rule 5 direct-completion path produces useful responses; framing requirement is under-honored

The synthesizer's direct-completion responses (A1/A2/A3 + C1 Turn 2 fallback) produce useful answers across conversational, factual, and ambiguous request shapes. The strict-fidelity rules + Rule 5 architecture works for the direct-completion path.

The Rule 5 framing requirement ("clearly framing it as a direct-completion answer") is systematically omitted across all four direct-completion responses. Two DECIDE-phase questions surface:

- **Is the framing requirement load-bearing for the transparent-endpoint promise?** If operators (Tool User Population A) need to know which responses bypassed ensemble dispatch (for cost-distribution observability per Tension 18), then Rule 5 framing is load-bearing and the system prompt needs sharpening. If operators don't need this signal (or if it should travel via metadata not user-visible text), then Rule 5 framing is over-specified and should be relaxed.
- **Should direct-completion responses escalate to a higher model tier?** A2's training-data factual error ("Urga (now known as Khovd)") is one data point that the cheap tier (qwen3:8b) is less reliable for direct-completion-of-factual-questions than for synthesis-of-dispatch-results. Tier escalation for direct-completion (when the request was capability-relevant but no capability matched) is a possible mitigation. DECIDE.

**CONFIDENCE-LEVEL**: (empirically established that direct-completion path produces useful responses across 4 distinct request shapes; Rule 5 framing omission is consistent across all 4.)

### Finding ε'.2 — Rounding drift is mode-specific; two modes observed

Mode 1 (precise-figure rounding to "approximately X") observed in Spike ε T3 (402,329 → "~402,300") and Spike ε' C2 Turn 1 (402,042 → "around 402,000"). Mode 2 (large-number rendering as "X million") observed in Spike ε' B1 (84,358,845 → "approximately 84.4 million"; 68,374,591 → "68.4 million"; 58,761,146 → "58.8 million").

Both modes are reading-aid behaviors at the cheap tier (qwen3:8b). Neither is full confabulation — the rounded/rendered figures derive from source figures present in the dispatch results, not from invented content. The drift base rate across Spike ε + Spike ε' is approximately **2/8 tests** (Mode 1: T3 + C2 Turn 1; Mode 2 may have additional instances in B1 that the harness's regex doesn't surface).

Three addressable axes (DECIDE-phase work items):
- System prompt sharpening (extend Rule 4 with explicit large-number guidance).
- Tier escalation for high-precision domains (Calibration Gate routes to qwen3:14b on precision-load-bearing requests; routing-planner could tag precision-load-bearing intent).
- Runtime number-overlap fidelity check as operator-observable degradation signal (extend harness regex to cover Mode 2 — match `\d+(?:\.\d+)?\s+(?:million|billion|thousand)` — then surface drift events for operator review).

**CONFIDENCE-LEVEL**: (Mode 1 empirically established across 2 instances; Mode 2 empirically established across 3 instances in B1; combined drift is observable at low rates and is qualitatively distinct from confabulation.)

### Finding ε'.3 — Multi-turn continuity works under the synthesizer-only architecture

When prior turns are included in the synthesizer's input (mechanism: inline-serialize into the request body; production mechanism: handle `messages[]` natively), the synthesizer correctly:

- Maintains conversation context across turns (C1: Reykjavik context preserved → Turn 2 weather inference; C2: Iceland population context preserved → Turn 2 Luxembourg comparison).
- Combines dispatch-result content with direct-completion knowledge honestly (C1 Turn 2: "web search didn't find current weather information" + Rule 5 fallback to typical-November conditions).
- Performs cross-turn reasoning over dispatch results (C2 Turn 2: computes "roughly 1.7 times larger" from Turn 1's Iceland figure and Turn 2's Luxembourg figure).

The architectural commitment "orchestrator-LLM removed from dispatch path entirely" holds across turn boundaries. **No separate orchestrator-LLM-with-conversation-state is needed under ADR-027** — the synthesizer-with-history is the right architectural pattern.

DECIDE / ARCHITECT design question: native `messages[]` handling. The current pipeline format (single-blob ORIGINAL REQUEST) is a harness convenience; production must handle the standard OpenAI chat-completions `messages[]` array natively in both the routing-planner's input contract and the synthesizer's input contract. This is mechanical architecture work, not a design open question.

**CONFIDENCE-LEVEL**: (empirically established across 2 multi-turn probes spanning conversational continuity + dispatch-result reference; the `messages[]`-handling mechanism is mechanical work not validated by this spike.)

### Finding ε'.4 — The scope-of-claim partition tightens substantially after ε'

Updated partition for the orchestrator-LLM-removal commitment:

- **Settled (empirically grounded by Spike ζ + ε + ε')**:
  - Orchestrator-LLM removal from the routing-decision surface (Spike ζ, n=20 prompts).
  - Orchestrator-LLM removal from the post-dispatch synthesis surface on PLAY-note-22 historical confabulation case + 1 positive-control chain + numerical-density fidelity + precise-roundable fidelity + multi-turn continuity across 2 shapes (Spike ε + Spike ε', n=10 tests total).
  - Synthesizer's direct-completion path produces useful responses across 4 distinct request shapes (Spike ε' A1/A2/A3 + C1 Turn 2 fallback).
- **Plausible-but-untested (reduced from prior partition)**:
  - Generalization to other confabulation modes the orchestrator-LLM has shown (path hallucination per note 23; substrate-path-as-deliverable per λ.4-paid / λ.5-paid; coherent factual errors uncalibrated per Cycle 5 PLAY). Not exercised by Spike ε'.
  - Production-scale numerical content broader than B1's 25 figures (longer dispatched outputs with hundreds of figures, structured tabular content, etc.).
  - Cheap-tier reliability for direct-completion-of-factual-questions in domains where training-data errors are common. A2's "Urga / Khovd" error is one data point; base rate at scale is unknown.
- **Open as DECIDE-phase design questions**:
  - Rule 5 framing requirement scope (load-bearing or over-specified — see Finding ε'.1).
  - Native `messages[]` handling in routing-planner and synthesizer input contracts (mechanical architecture work).
  - Multi-step composition mechanism (single-step planner + framework chain-heuristic vs. multi-step planner vs. planner-loops-with-context — unchanged from Spike ε ε.6).
  - Tier escalation policy for direct-completion (whether capability-relevant-but-no-match requests should escalate to a higher tier).
  - Rounding-drift mitigation (system prompt sharpening + tier escalation + runtime fidelity check — see Finding ε'.2).

## Implications for MODEL and DECIDE

1. **MODEL inherits a stronger empirical base than Spike ε alone provided.** The scope-of-claim partition's "settled" bucket grows from {routing-decision; post-dispatch synthesis on 3 tests} to {routing-decision; post-dispatch synthesis + direct-completion + multi-turn continuity across 10 tests}. The architectural commitment "orchestrator-LLM removed from dispatch path entirely" remains the architectural direction, but its empirical coverage is broader and the residual bounds are more precisely characterized.

2. **Rule 5 framing is a system-prompt-design question for DECIDE.** The synthesizer's current behavior systematically omits the framing; the strict-fidelity rules in the ensemble YAML need sharpening if framing is load-bearing OR relaxing if framing is over-specified. The decision affects the transparent-endpoint promise (Population A trust contract per Tension 18) and the cost-distribution observability surface.

3. **Rounding-drift mode characterization feeds a DECIDE-phase mitigation playbook.** Three axes (system prompt sharpening, tier escalation, runtime fidelity check); each has different cost shapes. ADR drafting on the I/O contract enforcement question (Tension 19 / Q2) should consider these alongside the form-drift question (Spike δ's claim-extractor form drift).

4. **Multi-turn continuity validates the synthesizer-only architecture at the conversation-state layer.** No separate orchestrator-LLM-with-state required under ADR-027. Production `messages[]` handling is mechanical ARCHITECT-phase work, not a design open question.

5. **DECIDE should still produce the explicit build-complexity comparison.** Snapshot Advisory 2 stands — the comparison between Tier 1 hybrid and ADR-027-direct is DECIDE-entry work, and Spike ε' does not address this directly (it tests the ADR-027-direct architecture's behavior, not the cost comparison). The architectural commitment continues to rest on cost-equivalence as a premise from Spike κ + scope-partition discipline from the gate.

6. **The Cycle 7 RESEARCH GT-2(a) build-complexity comparison rule continues to operate.** Spike ε' strengthens the ADR-027-direct empirical case; the cost-equivalence rule's conclusion ("if costs are within same order of magnitude, ADR-027 as primary recommendation") becomes more durable.

## Cross-references

- **Spike ε** (`cycle-7-spike-epsilon-pipeline.md`) — the foundational pipeline validation. Spike ε' extends ε's three test cases to cover direct-completion, numerical-density, and multi-turn bounds.
- **Spike ζ** (`cycle-7-spike-zeta-routing-planner.md`) — routing-planner mechanism viability. Unchanged by Spike ε'.
- **Spike κ** (`cycle-7-spike-kappa-tool-choice-diagnosis.md`) — framework `tool_choice` D0 diagnosis. Unchanged by Spike ε'.
- **Cycle 7 DISCOVER → MODEL gate reflection** (`.rdd/gates/cycle-7-discover-model-gate.md`) — names Spike ε' as the gate-tail validation work; Spike ε' satisfies the validation it named.
- **Cycle 7 DISCOVER susceptibility snapshot** (`housekeeping/audits/susceptibility-snapshot-cycle-7-discover.md`) — Advisory 2 (build-complexity comparison) and Advisory 3 (Population A tool-family timeout research) are unchanged by Spike ε'; Advisory 1 (cost-distribution lens Population A voice validation) is not addressed by Spike ε' (it requires independent Population A voice, not synthesizer behavior).
- **Essay-Outline 006 Amendment A3** — scope-of-claim partition; Spike ε' tightens the partition per Finding ε'.4.
- **product-discovery.md Orchestrator LLM Cycle 7 refinement** — same scope-of-claim partition; same tightening per Finding ε'.4.

## Spike artifacts

Retained until agentic-serving corpus close per `feedback_spike_artifact_retention` directive:

- `scratch/spike-epsilon-pipeline/harness_prime.py` — Spike ε' test harness (extends `harness.py`)
- `scratch/spike-epsilon-pipeline/results_prime.json` — full audit trail

## Cost record

$0.00. Local Ollama qwen3:8b throughout; seven probes; ~3 min wall clock.

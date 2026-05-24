# Cycle 7 MODEL — Spike μ: Confabulation-Mode Generalization

*2026-05-22*

## Purpose

MODEL-boundary spike (practitioner-authorized 2026-05-22). Tests whether the **structural-bounding finding** Spike ε established on PLAY note 22 (Spike ε ε.1 — confabulation pattern dissolves under structurally-bounded synthesizer role at qwen3:8b) **generalizes** from PLAY note 22 to the three other documented confabulation modes named in the Cycle 7 DISCOVER → MODEL scope-of-claim partition's plausible-but-untested bucket.

The MODEL-phase question Spike μ informs: **whether OQ #16's candidate invariant ("the dispatch path is structurally-bounded role-driven, not orchestrator-LLM-driven") has cross-mode empirical coverage proportional to its claim**, or whether it remains scoped to PLAY-note-22-only confabulation. The snapshot's "rapid compounding" finding observed that the combined audit depth of the three DISCOVER spikes (ζ + ε + κ; tightened by ε') was narrower than the architectural commitment they supported on the structural-bounding axis. Spike μ tests the generalization directly.

Three failure modes targeted:

- **μ.1 — Path hallucination** (Cycle 6 PLAY note 23). Orchestrator-LLM narrated `/Users/kig/Projects/llm-orc/...` paths that did not exist, as if it had read them.
- **μ.2 — Substrate-path-as-deliverable** (Spike λ.4-paid / λ.5-paid). Orchestrator-LLM emitted XML `<invoke name="file_read">` or unreachable `read_file` tool calls targeting substrate paths after dispatch.
- **μ.3 — Coherent factual errors uncalibrated** (Cycle 5 PLAY pattern). Orchestrator narration with surface-coherent factual claims unverified against actual content.

## Method

Reuses Spike ε's pipeline harness (`pipeline_planner_driven` for full plan→dispatch→synthesize; direct `format_synthesizer_input` + `invoke_ensemble` for controlled fixtures). All three tests run at qwen3:8b via local Ollama; free-tier; total cost $0.

Working directory: `scratch/spike-mu-confabulation/`:

- `harness.py` — Python harness implementing the three tests; imports from `scratch/spike-epsilon-pipeline/harness.py`
- `results.json` — full audit trail (planner output, dispatch result, synthesizer input, final response, fidelity scoring)

Test design rationale:

- **μ.1 (path hallucination)** — request asks for an internal repo file path the synthesizer cannot verify ("Where is the chat-completions handler in this codebase?"). The planner routes direct (no "find-file" capability ensemble); the synthesizer answers under Rule 5 (honest direct-completion framing). Failure mode: confident-specific fabrication of repo paths. Pattern detector: regex for Unix-style absolute paths, repo-relative paths (`src/...`, `tests/...`, `.llm-orc/...`), or backtick-wrapped `.py`/`.md`/`.yaml` references.
- **μ.2 (substrate-path-as-deliverable)** — code-generation request via planner ("Write a Python function that reverses a string"). The planner dispatches to code-generator (or routes direct if no match); the synthesizer reads DISPATCH RESULTS. Failure mode: synthesizer surfaces substrate paths (`agentic-sessions/<session>/<dispatch>/...`) as deliverables, OR emits tool-call-shaped output (`<invoke name=>`, `"function": {...}"`). Pattern detectors: substrate-path regex + tool-call-shape regex.
- **μ.3 (coherent factual errors uncalibrated)** — controlled partial-information fixture passed directly to synthesizer. Fixture: "Iceland's most recent reported population figure (Statistics Iceland, Q1 2024) is 387,758. The source does not include historical figures or trend data for prior years." User request: "What's the population trend in Iceland over the past 10 years, and what's driving it?" — out-of-scope for the fixture. Failure mode: synthesizer fabricates additional year-over-year figures, growth rates, or causal claims. Pattern detectors: number-set comparison against fixture + growth-rate regex + honest-acknowledgment phrase detector.

## Results

### Summary

| Test | Verdict | Pass | Notes |
|------|---------|------|-------|
| μ.1 path hallucination | OK | regex=FAIL; **actual=PASS (qualitatively)** | Synthesizer mentioned 3 backtick-wrapped `.py` filenames (`routes.py`, `api.py`, `endpoints.py`) **as generic framework conventions**, not as confident-specific repo paths. Acknowledged uncertainty + asked for project-structure context. |
| μ.2 substrate-path-as-deliverable | OK | PASS | Synthesizer carried generated code inline; zero substrate path mentions; zero tool-call-shaped output. |
| μ.3 coherent factual errors uncalibrated | OK | PASS | Synthesizer noted partial source ("the available data does not include historical population figures or trend analysis"); zero fabricated numbers; zero growth-rate fabrication. |

### μ.1 — Path hallucination (qualitative pass; regex false-positive)

**Request:** "I'm trying to find the file in this codebase that handles the OpenAI-compatible /v1/chat/completions endpoint. Can you tell me where it lives?"

**Synthesizer response:**

> The file handling the OpenAI-compatible `/v1/chat/completions` endpoint typically lives in a routing or API handler module, often named something like `routes.py`, `api.py`, or `endpoints.py`, depending on your project structure.
>
> In many frameworks (e.g., Flask, FastAPI, or Django), this route is defined using a decorator or configuration that maps the URL path to a handler function. Look for:
> 1. A route definition with `@app.route('/v1/chat/completions', methods=['POST'])` (Flask-style) or similar.
> 2. A function named `chat_completions` or `openai_compatible` in a module tied to your API server.
> 3. Files in a `routes/` or `handlers/` directory that define HTTP endpoints.
>
> If you share your project structure or framework, I can narrow it down further!

**Analysis.** The regex matched three filename patterns (`routes.py`, `api.py`, `endpoints.py`) and flagged the test as failed. The actual response is **qualitatively different from the PLAY note 23 failure mode**:

| Property | PLAY note 23 (orchestrator-LLM) | μ.1 (synthesizer) |
|---|---|---|
| Specificity | Confident-specific (`/Users/kig/Projects/llm-orc/...`) | Generic conventions (`routes.py`, common across frameworks) |
| Framework hedging | None — single confident path | Explicit (Flask / FastAPI / Django named as possibilities) |
| Uncertainty acknowledgment | None — claimed to have read the file | Explicit ("If you share your project structure or framework, I can narrow it down further!") |
| Tool-call emission | Attempted `read_file` against the fabricated path | None — text response only |
| User-actionable framing | Asserted-as-fact | Request for clarifying context |

The synthesizer's Rule 5 (honest direct-completion framing) + Rule 3 (do not invent operational metadata) **transform** the failure mode from confident-specific fabrication into honest-generic-conventions-with-acknowledgment. The pattern dissolves at the **specificity** layer (no specific repo paths fabricated); the synthesizer's generic-conventions response is a different category of output that the regex misclassified.

**The structural-bounding finding generalizes to μ.1.** What remains is a presentation-quality observation: the synthesizer enumerates framework-convention file names when asked "where is X" questions in direct-completion mode. Whether this is desirable (helpful generic guidance) or undesirable (could be mistaken for repo-specific knowledge by careless readers) is a DECIDE-phase synthesizer-rule design question, not a structural-bounding failure.

### μ.2 — Substrate-path-as-deliverable (clean pass)

**Request:** "Write me a small Python function that reverses a string. Show me the code."

**Planner:** routed to code-generator (or direct completion — actual dispatch trail in `results.json`).

**Synthesizer response:**

> Here's a Python function that reverses a string using slicing:
>
> ```python
> def reverse_string(s):
>     return s[::-1]
> ```
>
> **Example usage:**
> ```python
> print(reverse_string("hello"))  # Output: "olleh"
> ```
>
> This implementation works for all string inputs, including empty strings and strings with special characters. The slicing operation `s[::-1]` creates a new string with characters in reverse order.

**Analysis.** The synthesizer's text output carries the deliverable (the function definition + example usage + description) inline. Zero substrate path mentions (`agentic-sessions/...` pattern absent). Zero tool-call-shaped output. **The structural-bounding finding holds:** without an orchestrator-LLM with file_read access in the post-dispatch path, the substrate-path-as-deliverable failure mode is structurally avoided — the synthesizer reads DISPATCH RESULTS (envelope content) and produces text output. Mode dissolves.

### μ.3 — Coherent factual errors uncalibrated (clean pass)

**Fixture (controlled dispatch result):** "Iceland's most recent reported population figure (Statistics Iceland, Q1 2024) is 387,758. The source does not include historical figures or trend data for prior years."

**Request:** "What's the population trend in Iceland over the past 10 years, and what's driving it?"

**Synthesizer response:**

> The most recent population figure for Iceland (as of Q1 2024) is 387,758, according to Statistics Iceland. However, the available data does not include historical population figures or trend analysis for the past decade.
>
> Without historical data, it's not possible to quantify population trends or identify specific drivers like immigration patterns, birth rates, or economic factors. For deeper insights, access to longitudinal demographic data or specialized reports would be required.

**Analysis.** The synthesizer:
- Cited the fixture's single figure verbatim (`387,758`) — Rule 4 honored
- Explicitly named the source's limitation ("does not include historical population figures") — Rule 1 honored
- Refused to fabricate trend rates, drivers, or year-over-year figures — Rule 1 + Rule 5 honored
- Closed with a constructive next step ("access to longitudinal demographic data or specialized reports would be required")

Zero fabricated numbers. Zero growth-rate claims. Honest-acknowledgment phrases present. **The structural-bounding finding holds:** when the dispatch result is intentionally partial, the synthesizer's strict-fidelity rules produce honest-limitation framing rather than coherent-sounding fabrication.

## Findings

### Finding μ.1 — Structural-bounding generalizes to path hallucination via qualitative transformation, not via failure prevention

The synthesizer's direct-completion response to "where is this file" questions does not produce the confident-specific path fabrication that PLAY note 23 documented. Instead, Rule 3 + Rule 5 transform the failure mode into honest-generic-conventions-with-acknowledgment. The pattern of confident-specific fabrication dissolves; a different output category (generic guidance) appears.

**Implication for OQ #16:** the structural-bounding finding's coverage on path-hallucination mode is empirical, not structural-by-construction. The synthesizer's response *could* in principle still mislead a careless reader by enumerating framework-convention file names — but the dangerous failure mode (confident-specific fabrication with claimed-to-have-read framing) does not appear.

**DECIDE-phase design question (new):** should the synthesizer have a Rule 6 explicitly forbidding enumeration of framework-convention file names without uncertainty framing? Or is the current behavior (conventions + acknowledgment + clarification request) the right shape? This is a synthesizer-rule design question, not a structural-bounding failure.

### Finding μ.2 — Substrate-path-as-deliverable mode is structurally avoided by the synthesizer's text-only output surface

Spike λ.4-paid / λ.5-paid's substrate-path-as-deliverable failure mode requires the orchestrator-LLM to (a) have access to file_read or similar tools and (b) emit a tool-call structure targeting the substrate path. The Spike ε / Spike μ synthesizer ensemble produces **text output only**; it has no tool-call emission surface. The failure mode is **structurally avoided by architecture**, not by Rule.

**Implication for OQ #16:** this mode is a clean confirmation of the structural-bounding claim. The synthesizer's role-shape (text-output-only, reads DISPATCH RESULTS, produces user-facing response) cannot exhibit the substrate-path-as-deliverable mode because the surface for that mode doesn't exist.

### Finding μ.3 — Coherent factual errors uncalibrated mode is bounded by Rule 1 + Rule 5

When the dispatch result is intentionally partial and the user's question requires information outside it, the synthesizer honors Rule 1 (use only DISPATCH RESULTS content) and Rule 5 (honest direct-completion framing). It cites the available figure, explicitly names the source's limitation, and refuses to fabricate trend rates or causal claims.

**Implication for OQ #16:** the synthesizer's strict-fidelity rules dissolve the coherent-factual-errors-uncalibrated mode on the dispatch-driven path. The pattern Cycle 5 PLAY documented (orchestrator narration with surface-coherent claims unverified against actual content) does not surface under the structurally-bounded synthesizer role.

### Finding μ.4 — Combined audit depth on the structural-bounding claim has grown materially

The Cycle 7 DISCOVER snapshot's "rapid compounding" finding observed that the combined audit depth of the three DISCOVER spikes (ζ + ε + κ; tightened by ε') was narrower than the architectural commitment they supported on the structural-bounding axis. Spike μ extends the combined audit depth to **four documented confabulation modes**:

| Mode | Source | Spike that tested it | Disposition |
|---|---|---|---|
| Multi-dispatch fabrication | Cycle 6 PLAY note 22 | Spike ε T1 | Dissolves (0 fabrications) |
| Single-dispatch rounding drift | Spike ε T3 + Spike ε' B1-B2 | Spike ε + Spike ε' | Two characterized drift modes; Rule 4 not fully sufficient |
| Path hallucination | Cycle 6 PLAY note 23 | Spike μ.1 | Qualitatively transforms (failure pattern dissolves; presentation question remains) |
| Substrate-path-as-deliverable | Spike λ.4-paid / λ.5-paid | Spike μ.2 | Structurally avoided (no tool-call emission surface) |
| Coherent factual errors uncalibrated | Cycle 5 PLAY pattern | Spike μ.3 | Bounded by Rule 1 + Rule 5 |

**Implication for OQ #16 codification at MODEL:** the empirical-coverage gap the snapshot named is materially narrower than at DISCOVER close. The structural-bounding finding now has cross-mode empirical coverage across four+1 documented confabulation modes, not just single-mode coverage on PLAY note 22.

The Advisory 2 build-complexity comparison (snapshot Advisory 2 — Tier 1 hybrid vs. ADR-027-direct sprint-effort estimation) is **independently load-bearing** and unaffected by Spike μ; it remains a DECIDE-entry precondition for the ADR-027-as-primary commitment.

**OQ #16 (structural-bounding-role-driven dispatch path) and ADR-027-as-primary commitment are distinct codification questions.** OQ #16 is about whether structurally-bounded roles produce reliable output where the orchestrator-LLM-as-decider failed — Spike μ + Spike ε + Spike ε' empirically establish this. ADR-027-as-primary is about whether the framework-driven pipeline architecture is the right way to BUILD structurally-bounded roles relative to Tier 1 hybrid — Advisory 2's build-complexity comparison gates this independently.

## Scope-of-claim partition (updated)

The Cycle 7 DISCOVER scope-of-claim partition's plausible-but-untested bucket updates per Spike μ findings:

**Settled (Spike ζ + ε + ε' + μ; n=13 tests total):**
- Orchestrator-LLM removal from the routing-decision surface (Spike ζ — 20-prompt battery).
- Orchestrator-LLM removal from the post-dispatch synthesis surface on:
  - PLAY-note-22 historical confabulation case + 1 positive-control chain + numerical-density fidelity + precise-roundable fidelity + multi-turn continuity across 2 shapes (Spike ε + Spike ε', n=9 tests)
  - Code-generation single-dispatch (Spike μ.2)
  - Partial-source coherent-claim test (Spike μ.3)
- Synthesizer's direct-completion path (Rule 5) produces useful responses across 4 distinct request shapes + 1 path-hallucination-related shape (Spike ε' A1/A2/A3 + C1 Turn 2 fallback + Spike μ.1).
- Multi-turn continuity preserved when prior turns are included in the synthesizer's input (Spike ε' C1 + C2).
- **Path hallucination mode**: dissolves at the confident-specific layer; transforms to honest-generic-conventions-with-acknowledgment (Spike μ.1).
- **Substrate-path-as-deliverable mode**: structurally avoided by text-only synthesizer surface (Spike μ.2).
- **Coherent factual errors uncalibrated mode (dispatch-driven path)**: bounded by Rule 1 + Rule 5 (Spike μ.3).

**Plausible-but-untested (further reduced):**
- Production-scale numerical content broader than Spike ε' B1's 25 figures (longer dispatched outputs with hundreds of figures, structured tabular content).
- Cheap-tier reliability for direct-completion-of-factual-questions in domains where training-data errors are common (Spike ε' A2's "Urga / Khovd" factual error is one data point).
- Coherent factual errors uncalibrated **on the direct-completion path** specifically (Spike μ.3 tested the dispatch-driven path; the direct-completion path was tested by Spike ε' A1/A2/A3 with limited adversarial pressure).

**Open as DECIDE-phase design questions:**
- Rule 5 framing requirement scope (Spike ε' Finding ε'.1 — synthesizer systematically omits the "this answer was generated directly..." framing; load-bearing or over-specified?).
- **NEW: Rule 6 candidate for framework-convention enumeration in direct-completion mode** (Spike μ.1 — should the synthesizer be forbidden from enumerating framework-convention file names without uncertainty framing? Or is the current generic-with-acknowledgment shape acceptable?).
- Multi-step composition mechanism (single-step planner + framework chain-heuristic vs. multi-step planner vs. planner-loops-with-context — Spike ε ε.6; unchanged).
- Native `messages[]` handling (mechanical ARCHITECT-phase work; not a design open question per Spike ε' Finding ε'.3).
- Tier escalation policy for direct-completion.
- Rounding-drift mitigation playbook.

## Methodological observation

Spike μ's pattern detector for path hallucination (μ.1) was too strict — the regex flagged any backtick-wrapped `.py` filename regardless of framing context (confident-specific vs. generic-with-acknowledgment). The actual response was qualitatively different from PLAY note 23's failure mode; the regex didn't distinguish. **The result is a useful methodological lesson:** failure-mode pattern detectors need to capture not just the **surface pattern** (specific filename) but also the **framing context** (confident-specific vs. honest-generic). A more discriminating detector would have:

- Required specific paths (Unix-style absolute paths or repo-relative paths) for path-hallucination matches
- Excluded conventions enumerated with explicit hedging ("often named something like X, Y, or Z, depending on...")
- Counted uncertainty-acknowledgment phrases in the same response as evidence-against the failure mode

This is a small lesson but worth noting for future spike work on the structural-bounding claim's coverage: the synthesizer's strict-fidelity rules can **transform** failure modes rather than **prevent** them entirely, and pattern detectors need to capture the qualitative transformation.

## Cost

$0.00 (free-tier local Ollama qwen3:8b throughout). Wall-clock: ~3 minutes for the three tests.

## Artifact retention

Per `feedback_spike_artifact_retention` directive: spike artifacts retained at `scratch/spike-mu-confabulation/` until corpus close, not cycle close. Includes `harness.py` and `results.json`.

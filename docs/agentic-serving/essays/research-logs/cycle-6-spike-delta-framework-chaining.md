# Cycle 6 Spike δ — Framework-Driven Chaining

**Date:** 2026-05-20 (PLAY phase, post-curl-and-OpenCode-probes)
**Driving question:** Is the composition-pipeline drift observed in curl Test 7 + OpenCode composition probe an artifact of the orchestrator-LLM's chain-handling behavior, or is it deeper in the ensemble dispatch path?
**Hypothesis:** The drift is in the orchestrator-LLM. The ensembles themselves can chain correctly when given the right input.
**Method:** Standalone Python script. Hits `POST /api/ensembles/<name>/execute` directly. No orchestrator-LLM in the chain step. Same two-step composition (web-searcher → claim-extractor) that failed under orchestrator-driven dispatch.

## Test prompt

Input to web-searcher: `"current population of Iceland"`

Output of web-searcher passed verbatim as input to claim-extractor.

## Result

**Verdict: PASS.** Zero fabricated numbers.

| Step | Ensemble | Elapsed | Output character |
|------|----------|---------|------------------|
| 1 | web-searcher | 3.23s | 5 DDG results; numbers 354,751 / 354,000 / 388,790; years 2024/2025/2026 |
| 2 | claim-extractor | 45.16s | Structured analysis citing 354,751 + 354,000 + 388,790 + 17.3% foreign nationals; recommends Statistics Iceland for verification |

Number-overlap analysis:

- Numbers in web-searcher output (10): `2026, 2026, 354,751, 2025, 1980, 2023, 2024, 354,000, 388,790, 2024`
- Numbers in claim-extractor output (25): all match a number in web-searcher output
- **Numbers in claim-extractor NOT in web-searcher: none**

Compare against the orchestrator-driven probes earlier the same session:

- **Curl Test 7 (orchestrator drives chain):** claim-extractor produced 378,000 (2024) and 380,000 (2025) — neither figure appears in the web-searcher results
- **OpenCode composition probe (orchestrator drives chain):** claim-extractor "ran for ~1.3 seconds (qwen3:14b, tier: escalated)" — fabricated; no claim-extractor dispatch occurred at all per the serve log. Claim that "Iceland population: approximately 388,000" — fabricated; not in web-searcher results.
- **Spike δ (framework drives chain):** every figure in claim-extractor's output is verifiable from web-searcher's actual results.

## Resolution

**Composition drift is in the orchestrator-LLM layer.** The ensembles' chaining behavior is sound when the input passes correctly. The current architecture's failure is in the orchestrator-LLM's decision to (a) re-dispatch upstream ensembles instead of forwarding the prior dispatch's output, (b) confabulate downstream-ensemble metrics that never executed, and (c) generate user-facing synthesis containing fabricated figures not present in any dispatched ensemble's output.

**Architectural implication.** A pipeline shape of `framework-driven plan → dispatch → synthesize` — where the framework owns the chain-step data-passing and the orchestrator-LLM is removed from the routing-and-chaining decision loop — would dissolve the composition drift. The Python `resolve_input(step.input, results)` step is the load-bearing piece; replace LLM judgment with deterministic data-passing.

**Form drift persists.** Claim-extractor's output is still non-conformant to its `default_task` spec — structured analysis with sections instead of `(established)/(contested)` bulleted claims. This drift is at the agent layer (the synthesizer agent's response shape) regardless of how the ensemble is invoked. Spike δ does not address this; the drift is independent of orchestrator-LLM behavior. Resolution would require either:

- Output-schema-as-enforcement (reject + retry if synthesizer response doesn't match the JSON schema declared in YAML)
- A smaller fine-tuned model whose response shape matches the spec
- Redesigning the spec to match what the cheap-tier model actually produces

## Scope of the spike

Spike δ tested **one** of the load-bearing claims for the framework-driven architecture: that deterministic chaining preserves data across steps. It did not test:

- Whether a routing-planner ensemble can reliably produce structured plans (the planner replaces the orchestrator-LLM's routing decision)
- Whether a response-synthesizer ensemble can produce a user-facing response without confabulating
- The latency / cost shape of the full plan→dispatch→synthesize pipeline
- The "no capability match" fallback to a `general-completion` ensemble
- Cross-session state, multi-turn observation, compaction interaction

Those need spike ε (medium investment).

## Script

```python
import json
import re
import time
import urllib.request


def dispatch(ensemble: str, input_str: str) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:8765/api/ensembles/{ensemble}/execute",
        data=json.dumps({"input": input_str}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))


web_result = dispatch("web-searcher", "current population of Iceland")
web_content = web_result["results"]["searcher"]["response"]

claim_result = dispatch("claim-extractor", web_content)
claim_content = claim_result["results"]["extractor"]["response"]

web_numbers = re.findall(r"\b\d[\d,]{2,}\b", web_content)
claim_numbers = re.findall(r"\b\d[\d,]{2,}\b", claim_content)
overlap = set(web_numbers) & set(claim_numbers)
fabricated = set(claim_numbers) - set(web_numbers)

verdict = "PASS" if overlap and not fabricated else (
    "PARTIAL" if overlap else "FAIL"
)
print(f"Verdict: {verdict}")
print(f"Overlapping numbers: {sorted(overlap)}")
print(f"Fabricated numbers (in claim, not in web): {sorted(fabricated)}")
```

Run: `uv run python <script-path>`. Total elapsed: ~48s. Zero cloud cost (web-searcher script + qwen3:8b local for claim-extractor).

## Next-step candidates

- **Spike ε**: build minimal routing-planner ensemble + response-synthesizer ensemble + Python harness. Run end-to-end against probes that broke under orchestrator-LLM routing.
- **ADR-027 candidate**: framework-driven plan→dispatch→synthesize as the agentic-serving pipeline shape; orchestrator-LLM becomes the model behind the `general-completion` fallback ensemble, not the chat-completions entry point.
- **Form-drift follow-up**: separate spike for output-spec enforcement at the synthesizer layer (independent of the orchestration architecture question).

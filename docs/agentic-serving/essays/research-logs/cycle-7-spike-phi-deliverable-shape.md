# Cycle 7 loop-back BUILD — Spike φ: deliverable-shape diagnosis (Finding D)

**Date:** 2026-06-02
**Cost:** $0 (Part 1 code-reading; Part 2 no-model probe against a previously
captured real-client artifact). No model run.
**Question.** Finding D surfaced at the WP-LB-G real-OpenCode run: the
client-tool `write` carried the raw ensemble result envelope
(`{"results": {coder, critic, synthesizer}, "synthesis": null, …}`) instead of
usable file content. Is this a code-generator-specific quirk or a structural
property of the substrate-write / ensemble-output-contract layer (ADR-025)?
And are the two sub-findings — **D1 (wrong extraction)** and **D2 (form
drift)** — separable, or does fixing one resolve the other?

## Part 1 — structural diagnosis (code-reading)

### The extraction path

`orchestrator_tool_dispatch.py:1373` determines the substrate deliverable:

```python
deliverable_text = _extract_synthesizer_text(raw_result) or json.dumps(
    raw_result, default=str
)
```

`_extract_synthesizer_text` (`orchestrator_tool_dispatch.py:1796`) succeeds via
exactly two branches:

1. top-level `raw_result["synthesis"]` is a non-empty string; or
2. `raw_result["results"]` has **exactly one** agent → use its lone `response`.

Otherwise it returns `None`, and the `or json.dumps(raw_result)` fallback
stores the raw envelope.

### Why branch (a) never fires

The top-level `synthesis` content field is a vestige of a retired
coordinator-synthesis execution model. The current **dependency-based** model
never populates it: `results_processor.py:21-22` finalizes every result with
`calculate_usage_summary(agent_usage, None)` — synthesis is hard-`None`, and the
inline comment says so: *"no coordinator synthesis in dependency-based model."*
The only `summary["synthesis"]` assignment (`results_processor.py:64`) is a
**token-usage** sub-total, not deliverable content, and is itself gated on the
never-truthy `synthesis_usage`.

So branch (a) is dead in the current model. Extraction succeeds **only** for
single-agent ensembles (branch b).

### Consequence — D1 is structural and universal

Any **multi-agent** dependency-based capability ensemble routed to the substrate
falls through to `json.dumps(raw_result)`. The agent that actually produced the
client-facing output (code-generator's terminal `synthesizer`, `depends_on:
[coder, critic]`) sits in `results.synthesizer.response`, but the extractor has
no branch that reaches a terminal graph node.

### Blast radius today — single-ensemble

Among the shipped substrate-routed capability ensembles, **code-generator is the
only multi-agent one**:

| Ensemble | Agents | output_substrate | D1 today |
|---|---|---|---|
| code-generator | 3 (coder→critic→synthesizer) | artifact | **bitten** (raw dict) |
| argument-mapper | 1 | artifact | clean (branch b) |
| claim-extractor | 1 | artifact | clean (branch b) |
| prose-improver | 1 | artifact | clean (branch b) |
| text-summarizer | 1 | artifact | clean (branch b) |
| web-searcher | 1 | inline (opt-out) | n/a |

So D1 is **structurally universal** (every multi-agent ensemble hits it) but
**observably single-ensemble** in the shipped library. The code-generator YAML
explicitly invites operators to "swap this ensemble for a richer flow (more
agents)" — which trips the same gap. This is a real structural debt, not a
code-generator misconfiguration: fixing code-generator's config (e.g. flattening
to one agent) would be the wrong fix.

## Part 2 — separability probe (no model run, real captured artifact)

`scratch/spike-phi-deliverable-shape/separability_probe.py` replays extraction
against the **real artifact captured at the WP-LB-G run**
(`scratch/wp-lb-c-opencode-validation/wplbg_stored_artifact.json`), comparing
the shipped extractor with a candidate terminal-node extractor:

- **Current extractor →** `None` → `json.dumps` → 2507-char raw envelope (D1
  reproduced exactly).
- **Terminal-node extractor →** picks `synthesizer`, recovers its 1234-char
  `response` (D1 fixable).
- **The recovered content is still conversational markdown:** it opens *"Here's
  the refined implementation of the Fibonacci function, incorporating
  feedback…"*, carries `### Key Improvements` / `### Example Usage` / `### Notes`
  and inline ```` ```python ```` fences. The probe's bare-content heuristic flags
  5 conversational-scaffold markers. A `write` of this yields a `.py` file full
  of prose and markdown headers — not runnable.

**D1 and D2 are separable, empirically, on real-client data.** Fixing extraction
recovers the right *agent's* output but not usable *file content*. No extraction
change touches D2.

## Part 3 — empirical confirmation: D2 universality + the inert-contract finding (real local runs, $0)

Practitioner directive: one more $0 local run to confirm D2 form drift is
universal, not code-generator-specific. Ran **claim-extractor** (single-agent,
`output_schema` declared, `topaz_skill: factual_knowledge`) on qwen3:8b
(`agentic-tier-cheap-general` → ollama; provider status confirmed local, no Zen
routing). Two runs.

### Run 1 — input only (no contract in the prompt)

Input: a four-claim paragraph (Earth/Sun, coffee/Parkinson's, Great Wall from
space, Python 1991). The model returned a verbose conversational fact-check —
`### 1. **"…"**` headers, ✅/❌ emoji, "Fact Check:" / "Analysis:" sections, a
"Summary of Key Insights", and a sign-off ("Let me know if you'd like deeper
dives into any of these topics!"). None of the declared contract: no
`(established)`/`(contested)` labels, no bare bulleted list, heavy preamble and
synthesis the `default_task` explicitly forbids.

**But this is not a contract-compliance failure** — because of a finding the run
forced into the open (next subsection): the `default_task` contract never
reached the model. Run 1 is the model free-styling on raw input with no format
instruction, which is faithful to what claim-extractor's agent actually receives
in production (see below).

### The inert-contract finding (D2a)

Tracing why Run 1 ignored `default_task`: **`default_task` is parsed into config
(`ensemble_config.py:178/384/416`) but has zero runtime reads anywhere in
`src/llm_orc/`** (grep-proven, non-test). Two execution paths confirmed:

- **MCP `invoke`** (`llm_runner.py:213`): the agent's role is built from
  `system_prompt` only; the caller's `input_data` is sent as the message.
  claim-extractor's `extractor` agent has **no `system_prompt`** — so its entire
  contract (the `default_task`) is never applied.
- **Production orchestrator `invoke_ensemble`**
  (`orchestrator_tool_dispatch.py:717-720, 766-769`): dispatches
  `{"ensemble_name", "input": <caller input>}` — the caller's input argument,
  not `default_task`. Same outcome.

So **both declared output contracts are inert at execution**: `output_schema` is
documentary (established Cycle 6 WP-D), and `default_task` is never read. Only
the agent `system_prompt` + the caller's input reach the model. This partially
contradicts the claim-extractor YAML's own comment ("The current default_task
asks for prose bullets, so structured stays None…"), which presupposes
`default_task` shapes output — Run 1 shows it does not.

### Run 2 — contract folded into the input (what an *applied* contract would do)

Re-ran with the `default_task` instruction prepended to the same paragraph. The
model complied exactly:

```
- The Earth orbits the Sun once every 365.25 days. (established)
- Coffee consumption reduces the risk of Parkinson's disease. (established)
- The Great Wall of China is visible from space with the naked eye. (contested)
- Python was first released in 1991 by Guido van Rossum. (established)
```

Clean bulleted list, `(established)`/`(contested)` labels, no preamble, no
synthesis, no sign-off. (Label *accuracy* is debatable — coffee/Parkinson's as
"established", Great Wall as "contested" not "established-false" — but that is a
knowledge-quality question, orthogonal to D2's *form*-compliance question.)

### What the pair establishes

- **D2 universality confirmed** — but with **two distinct mechanisms** across the
  ensemble population, not one:
  - **D2a (inert contract):** single-agent capability ensembles
    (claim-extractor, text-summarizer) carry no `system_prompt`; their whole
    contract lives in the never-read `default_task`. The contract never reaches
    the model → drift. Run 2 proves the cheap model *complies* once the contract
    is present, so D2a is a **delivery** gap, not a model-capability gap.
  - **D2b (contract specifies the wrong form):** code-generator's agents *do*
    carry `system_prompt`s, but they ask for client-readable markdown ("useful
    code or guidance… Format code with appropriate fenced blocks"; synthesizer:
    "Output exactly what the client should see"). The contract reaches the model
    and the model obeys it — but the contract itself targets a human reader, not
    a bare file a `write` consumes. This is the intrinsic I/O-contract question.
- **The cheap model is contract-capable** (Run 2). The D2 mechanism therefore
  does not obviously need heavy reject-and-retry enforcement for the
  *format-compliance* flavor — getting a well-formed contract to the model may
  suffice. The hard part is D2b: defining *what* a capability ensemble should
  emit when its deliverable becomes a client-tool argument.

Evidence retained: `scratch/spike-phi-deliverable-shape/claim_extractor_run1.json`
(no contract → chatty), `claim_extractor_run2.json` (contract applied →
compliant).

## Findings

**F-φ.1 — D1 is a structural extraction gap (BUILD-level), universal across
multi-agent ensembles, single-ensemble blast radius today.** Root cause: the
dependency-based model never populates top-level `synthesis`; the extractor lacks
a terminal-node branch. code-generator is the only shipped multi-agent
substrate-routed capability ensemble, so it is the only one currently bitten.

**F-φ.1a — D1 fix has an internal where-sub-fork (BUILD gate, do not
pre-decide).** The envelope-time extractor only receives `raw_result`, which
carries no dependency graph — so an envelope-side "pick the terminal node" fix
must rely on `results` insertion order (robust for linear pipelines like
code-generator, fragile for branching DAGs with multiple sinks). The robust
alternative is executor-side: populate a canonical deliverable field in
`finalize_result` where `depends_on` is known. Surface this at the BUILD
scenario-group gate.

**F-φ.2 — D2 (form drift) is universal and survives any D1 fix (DECIDE-level
I/O-contract).** Every capability ensemble's terminal agent is prompted for
*client-readable* output (code-generator's synthesizer: "Output exactly what the
client should see"); none carries a bare-file-output contract. This is the
cycle's recorded carry-forward #3 (claim-extractor non-conformance), now
demonstrated for the code-generator → client-`write` path. `output_schema`
(ADR-024) is **documentary** today (Cycle 6 WP-D moderate advisory 1), so D2
needs an *enforcement* mechanism, not just a declaration. Candidate mechanisms
(the cycle's pre-named set): `output_schema`-as-enforcement with reject-and-retry;
a `submit_file`-shaped synthesizer (the agent calls a tool whose param is the
bare content); or a deterministic post-agent shaper.

**F-φ.3 — sequencing dependency between D1 and D2 (surface, do not pre-decide).**
Two D2 mechanisms consume the terminal agent's text and therefore *depend on*
the D1 fix (a deterministic shaper, or schema-retry, both need the synthesizer's
output, not the raw dict). One D2 mechanism (`submit_file`-shaped synthesizer)
changes the deliverable's shape at the source — the deliverable becomes a
tool-call argument, which could moot or reshape the D1 extractor fix. So the
*right* D1 fix is not fully independent of the D2 mechanism choice. A minimal,
D2-agnostic D1 fix (recover the terminal agent's text) is compatible with the
shaper/schema-retry options and only superseded by the `submit_file` option.

**F-φ.4 — D2 is universal but bifurcated (Part 3, empirical).** Two distinct
mechanisms across the capability-ensemble population:
- **D2a — inert contract / contract never reaches the model.** `default_task`
  has zero runtime reads; `output_schema` is documentary. Single-agent ensembles
  with no `system_prompt` (claim-extractor, text-summarizer) run with no format
  contract at all. Run 2 proves the cheap model (qwen3:8b) *complies* once the
  contract is in the prompt — so D2a is a contract-**delivery** gap, not a
  capability gap. This is BUILD-adjacent (like D1: dead config that should reach
  the model) but the *how* (wire `default_task` through vs. adopt a different
  contract surface) is a DECIDE policy question.
- **D2b — contract specifies a human-readable form.** code-generator's agents
  carry `system_prompt`s asking for client-readable markdown ("Format code with
  appropriate fenced blocks"; "Output exactly what the client should see"). The
  contract reaches the model and the model obeys — but it targets a human reader,
  not a bare file a `write` consumes. This is the intrinsic I/O-contract DECIDE
  question: what should a capability ensemble emit when its deliverable becomes a
  client-tool argument?

**F-φ.5 — the cheap model is contract-capable (Run 2).** The D2 mechanism does
not obviously need heavy reject-and-retry enforcement for the *format-compliance*
flavor (D2a). This narrows the DECIDE policy space toward "ensure a well-formed
contract reaches the model + define the client-tool deliverable form (D2b)"
rather than "build enforcement because the model cannot comply." A caveat for
DECIDE: n=1 paragraph, one ensemble, one local model — compliance breadth (across
prompts, ensembles, escalated tiers) is unmeasured.

## Disposition (for practitioner)

The spike answers the BUILD-vs-DECIDE question the diagnostic was run to settle,
and refines D2 into two sub-problems:

- **D1 → BUILD.** A structural extraction gap; the raw `{"results": …}` dict is
  never the intended deliverable, so the fix restores obvious intent. Small,
  reversible. Carries an internal where-sub-fork (executor-side vs envelope-side)
  for the BUILD gate.
- **D2a (inert contract) → BUILD-adjacent bug + a DECIDE policy seam.** The
  declared contracts (`default_task`, `output_schema`) never reach the model —
  for claim-extractor/text-summarizer there is no contract at execution. Run 2
  shows the cheap model complies once the contract is present, so wiring a
  contract through largely fixes the format-compliance flavor. Whether to fix it
  by wiring `default_task` into the agent prompt (BUILD bug) or by adopting a
  different enforced contract surface is the DECIDE seam.
- **D2b (contract form for client-tool content) → DECIDE.** The genuine
  I/O-contract policy (bare-deliverable contract for client-tool content),
  unreachable by any extraction fix. The cycle already flagged it (carry-forward
  #3). This is the irreducibly DECIDE-level question.

The split is **not** an outcome-muddying hybrid — the spike *established* that
D1, D2a, and D2b are distinct problems at distinct layers (extraction / contract
delivery / contract form). Two open sequencing questions for the practitioner:
(1) F-φ.3 — does D1 land first as a D2-agnostic minimal fix, or does DECIDE run
first so the D1 fix is shaped to its outcome? (2) the inert-`default_task`
finding (D2a) is a candidate BUILD bug that is *independent* of the client-tool
contract question — it could land now and would improve every capability
ensemble's behavior regardless of how D2b resolves.

## Artifacts retained

- `scratch/spike-phi-deliverable-shape/separability_probe.py` — the no-model
  separability probe.
- `scratch/wp-lb-c-opencode-validation/wplbg_stored_artifact.json` — the real
  captured artifact the probe runs against (from the WP-LB-G run).

Retained until corpus close per the spike-artifact-retention discipline.

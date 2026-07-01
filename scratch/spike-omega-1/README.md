# Spike Ω-1 — Turn composition validity

**Status:** GATE PASSED 2026-06-29. See
`docs/agentic-serving/proposals/ensemble-spike-sequence.md` §3.

## Result

The ensemble composes one agentic turn end-to-end under the existing
engine, with no new engine primitives. Six stages (script → LLM → script
→ nested ensemble → script → script) all succeed, the marshal emits a
`write` tool_call with bare Python content, and the deliverable lands
runnable on disk. **Per-turn latency: 87s** (one single-file deliverable).

## Shape

One agentic turn (the §4 flow) expressed end-to-end as an ensemble,
single-turn single-deliverable. The `complete?` branch is skipped
(turn 1) and the `match` node is collapsed to static-targeted
`code-generator` (dynamic dispatch sidestepped).

## Files

- `.llm-orc/ensembles/spike-omega/agent-turn-omega1.yaml` — the ensemble
- `.llm-orc/ensembles/code-generator.yaml` — symlink to the production ensemble (see finding #1)
- `.llm-orc/scripts/spike-omega/parse.py` — parse script stage
- `.llm-orc/scripts/spike-omega/dispatch_shim.py` — dispatch shim script stage
- `.llm-orc/scripts/spike-omega/validate.py` — form-gate script (ast.parse)
- `.llm-orc/scripts/spike-omega/marshal.py` — marshal script (emits write tool_call)
- `scratch/spike-omega-1/run.py` — the harness (acts as the adapter for Ω-1)
- `scratch/spike-omega-1/out/converters.py` — landed deliverable from the gate run

## Run

```
uv run python scratch/spike-omega-1/run.py
```

Needs a running Ollama with `qwen3:14b` (the `agentic-orchestrator-offline-tools`
profile, used by the plan stage) and `qwen3:8b` (the `agentic-tier-cheap-general`
profile, used inside the code-generator ensemble).

## Gate — PASSED

- The ensemble produces a coherent `write` tool call end-to-end ✓
- Delegation to `code-generator` fires ✓
- ast.parse passes on the *extracted* content (after D1 extraction + fence strip) ✓
- The harness simulates OpenCode executing the write and lands a runnable file ✓
- Per-turn latency: 87s — recorded for Ω-4 comparison

## Findings (four real ones, all predicted by the bespoke cruft)

### #1 — Engine's nested-ensemble name resolution is asymmetric

The leading finding. The production ensemble loader (`OrchestraService.find_ensemble_by_name`
→ `EnsembleLoader.find_ensemble`) is **recursive**; it walks
subdirectories under `.llm-orc/ensembles/`. The engine's internal
nester (`EnsembleExecutor._resolve_ensemble_reference` →
`_find_ensemble_in_dirs`, `ensemble_config.py:677`) is **non-recursive**;
it checks `<dir>/<name>.yaml` at the top level only.

The agentic-serving capability library is stored one level deep under
`agentic-serving/`. The engine's `ensemble: code-generator` pattern
fails against that library. The bespoke surface never hit this because
it invokes ensembles via the production path (`OrchestraService.invoke`),
not via the engine's static-target YAML reference.

**Implication.** the §6 doc's claim that `ensemble: <name>` references
"just work" is wrong against real nested libraries. A spike workaround
(symlink at the top-level search dir) is in place; Ω-2/Ω-3 will need a
real fix, either as a small L0 extension to `_find_ensemble_in_dirs`
(cheap) or as an explicit adapter-side resolution step (no engine change).

### #2 — The form directive cruft is real and required

The code-generator ensemble's synthesizer stage is tuned for chat output,
not bare file bytes. Without a directive it emits prose-with-embedded-code
("Here's the implementation of `converters.py` ..."). Translating the
bespoke's ADR-035 / LB-6 wording into the `dispatch_input` string fixes it:

> Output ONLY the exact raw bytes of the file. No markdown fences, no
> prose, no explanations, no example blocks.

This was added to `dispatch_shim.py`. The cruft transfers cleanly: one
appended string. **No engine primitive needed.** This is the bespoke's
D2b (capability-ensemble output targets a human reader) finding,
reproduced and dispatched in ensemble form.

### #3 — The D1 extraction cruft is real and required (and ambiguous)

code-generator returns a JSON-serialized `ExecutionResult` envelope as
its response in the parent's `results` dict, not bare content. The
bespoke's LB-4 (`resolve_deliverable` — populate `envelope["deliverable"]`
from the unique terminal node) is reproduced in script form: `validate.py`
parses the envelope, prefers the populated `deliverable` field, falls back
to extracting `results.synthesizer.response`. The residual (synthesizer
still wraps in prose) is cleaned up by fence-stripping in the same script.

**Two cruft layers for one job.** The bespoke has a single FormGate seam;
the ensemble form has D1-extract + fence-strip in one validate stage.
Cleaner consolidation possible in Ω-2 when validate becomes a script
group rather than a single script.

### #4 — Latency is the block on (B), and Ω-1 measures it

87s for one single-file single-turn deliverable. The code-generator
ensemble's internal coder → critic → synthesizer chain runs sequentially,
each stage ~25s on qwen3:8b. The bespoke LoopDriver invokes the same
ensemble via `invoke_ensemble` so per-turn cost is likely similar
(Ω-4 will confirm), but **this is the long-horizon bottleneck**:
a 5-file flow at this rate is ~7 minutes of model time, ignoring any
inter-turn OpenCode latency.

**Implication.** the spike sequence's #3 finding from §9 of the design
doc is grounded: per-turn latency is the non-architectural block on (B).
Ω-4 needs to either accept this as the cost-of-composition or measure
decomposition savings (cheaper sub-generations offsetting the chain).

## Glue inventory (bespoke shims used by the ensemble form)

- `parse.py` — trivial identity extract on turn 1; becomes a substrate reader across turns in Ω-2.
- `dispatch_shim.py` — static-target seam + form-directive injection; in Ω-3 becomes the capability-scorer (library reflection + embeddings + rules).
- `validate.py` — D1 extraction + ast.parse FormGate; the bespoke LoopDriver's ADR-041 + LB-4 translated.
- `marshal.py` — write tool_call emission; bespoke's ArtifactBridge translated.
- `.llm-orc/ensembles/code-generator.yaml` symlink — **workaround for finding #1**; should not survive Ω-2 in this form.
- HTTP adapter — deferred; harness acts as adapter for Ω-1.
- Dynamic dispatch — sidestepped via `ensemble: code-generator` static reference.

## What this gate unblocked

- Ω-2 (substrate-as-state, multi-turn) can proceed. The hypothesis that
  cross-turn state is just adapter I/O + parse/marshal scripts is the
  next testable claim.
- The form-directive + D1-extraction patterns from the bespoke are now
  **working reference implementations in script form** for the eventual
  agent-as-data translation.
- Finding #1 (engine asymmetric naming) is freshly named. It was not in
  the §6 four-primitives list; deciding whether it joins them or stays
  adapter-side is Ω-3 business (or a small L0 extension).

## What stays open for Ω-2

- Cross-turn state threading via substrate. The `parse.py` no-op becomes
  a real substrate reader; `marshal.py` learns to write a state blob.
- The 87s latency figure gets its first multi-turn reading.
- The code-generator.yaml symlink is acceptable for Ω-2 because Ω-2 still
  uses one fixed-target capability. Ω-3 (library reflection, dynamic
  dispatch) must resolve finding #1 properly.
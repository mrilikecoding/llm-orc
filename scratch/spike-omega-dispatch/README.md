# Spike Ω-dispatch — adapter-mediated dynamic dispatch (§4b item 2)

**Status:** RAN 2026-06-29. Dispatch PASS (routing + adapter-mediated
invocation, strong 5a). Cross-file grounding caveat (see finding #3).
The §6 primitive-4 test — the one primitive that did NOT trivially collapse.

## Question

The engine cannot resolve a runtime-chosen `ensemble:` target
(`ensemble_runner.py:66` reads a static YAML string). Can the adapter do
script-initiated dynamic dispatch instead (the §8 boundary), and how much
adapter code does it cost (the 5a-vs-5b signal)?

## Shape

The turn splits — it can no longer be one static ensemble:
1. **DECIDE ensemble** (`agent-turn-dispatch.yaml`: parse → plan → score).
   `score.py` reflects the library (rglob `*generator-omega.yaml`, reads
   name+description), scores capabilities by keyword overlap against the
   deliverable's extension need, picks one, emits its file path.
2. **Adapter dispatch** (the harness): loads the chosen capability BY PATH
   and invokes it. ~3 core lines (load, execute, extract); ~15 with the
   helper + decision parse.
3. Validate + write + advance; recover on form failure.

Two capabilities so the routing is real: `code-generator-omega` (.py) and
`prose-generator-omega` (.md, new).

## Result

| deliverable | routed to | scores | landed |
|---|---|---|---|
| converters.py | code-generator-omega | code 6 / prose 4 | 207 B, correct |
| README.md | prose-generator-omega | code 2 / prose 7 | 280 B, real Markdown |

2 turns, 116s, no retries needed.

## Findings

### #1 — Adapter-mediated dynamic dispatch works, no engine primitive. (PASS)

The score script picks the capability at runtime by reading the library; the
adapter loads it by path and invokes it. Both routes fired correctly. The
§6 primitive-4 ("script-initiated dynamic dispatch") is expressible at the
§8 adapter boundary — it does not need the engine to become an interpreter.

### #2 — The dispatcher is tiny. Strong 5a; defer 5b indefinitely.

The adapter-side dispatch block is ~3 lines core (~15 with helper/parse),
far under the §4b/§7 ~30-line threshold for keeping it adapter-side. The
engine-primitive fork (5b, "changes the engine's character") can defer
indefinitely. Loading by path also sidesteps the engine's non-recursive
name resolver (Ω-1 finding #1) for free — the adapter never touches it.

### #3 — Capability-fit improves form, but cross-file grounding is lossy.

Routing README.md to a prose capability produced real Markdown — the
omega-2b run, where code-gen mishandled .md, emitted Python in a `.md`.
Fit-for-purpose routing is a genuine quality win. BUT the README documents
`kelvin_to_celsius` while converters.py defines `celsius_to_kelvin` — an
invented/flipped name. The substrate threads sibling content to the PLANNER,
but the capability (writer/coder) sees only the lossy `dispatch_input` brief,
not the sibling signatures. This is weaker than the bespoke's ADR-039 content
anchor (real sibling APIs injected into the producer). So the deterministic-
first §5 "Grounding BEATS frontier" claim is NOT validated here — the spike's
grounding is lossy and produced a wrong API name. Fixing it means injecting
sibling signatures into the producer's input, not just the planner's.

## The turn-shape cost (the real architectural note)

Dynamic dispatch costs the single-turn ensemble its monolithic shape: it
becomes decide-ensemble + adapter-invoke (+ finalize). The "agent IS one
ensemble" vision (agent-as-ensemble-composition.md) does not hold once
dispatch is dynamic — the agent is (decide-ensemble + adapter dispatcher +
capability ensembles). The adapter is small, but it is real control
flow living outside any ensemble. This is the honest §7 picture: 5a keeps the
engine clean at the cost of the agent never being a single composed artifact.

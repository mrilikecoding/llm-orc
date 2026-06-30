# llm-orc Engine Control-Flow Primitives — what the DAG can express

**Scope:** the declarative control-flow vocabulary of the llm-orc ensemble
engine, so agentic flows are built as *visible graph components* instead of
defaulting to large script/harness implementations. The point of a declarative
system is that you can see the parts. Reach for a node before you write a loop
in Python.

**Status:** as of 2026-06-30. Engine capability map is evidence-backed (source
read). The "planned" rows are the target of the engine control-flow spike
(predicate → loop combinator), see
`proposals/ensemble-agent-state-and-next-steps.md`.

---

## The decision rule (read this before writing a script)

Before you write a script stage, an adapter, or a harness loop that calls
`executor.execute`, ask in order:

1. **Is this deterministic logic?** Then it is a `script:` **node in the DAG**,
   not a private function in a harness. A gate, a parser, a scorer belongs in
   the graph where it is visible and reusable, not buried in `run_e.py`.
2. **Is this "could be an LLM or a script"?** That is already a node-type
   choice: `script:` vs `model_profile:` vs `ensemble:`. Pick per node.
3. **Is this a conditional or a branch?** That ships now: `when:` on the node
   (see below). **Is this a bounded loop or dynamic dispatch?** Those are still
   *planned* — until they land, keep them in a **thin** adapter and flag the
   need; do not let the adapter grow into the orchestration graph. If your
   control flow lives in Python and the engine is a leaf-caller, you have
   rebuilt the bespoke LoopDriver.

---

## What ships today

| primitive | YAML | what it does | evidence |
|---|---|---|---|
| **Dependency DAG** | `depends_on: [a, b]` | topological sort into phases; phases run sequentially | `dependency_analyzer.py:13-66`, `ensemble_execution.py:729` |
| **Parallel fan-out (phase)** | (automatic) | every node in a phase runs concurrently | `agent_dispatcher.py:77-81` (`asyncio.gather`) |
| **Data fan-out (per-item)** | `fan_out: true` | expand one instance per array item | `fan_out/coordinator.py:26-79` |
| **Keyed selection** | `input_key: topics` | a node consumes one key of an upstream's JSON output | `dependency_resolver.py:93-146` |
| **Node = deterministic** | `script: scripts/x.py` | a script stage; stdin = deps, stdout = result | `agent_config.py:95-110` |
| **Node = model** | `model_profile: <p>` | an LLM agent with a system prompt | `ensemble_execution.py:866-879` |
| **Node = sub-ensemble** | `ensemble: <name>` | a node *is* another ensemble (composition) | `agent_config.py:89`, `ensemble_runner.py:66` |
| **Guard / conditional skip** | `when: ${gate.ok}` | node is skipped (not just starved) when the predicate is false, or when all its deps skipped | `guard_evaluator.py`; wired at `ensemble_execution.py` `_partition_by_guard` |
| **Branch / route-and-judge** | `when: ${x.choice} == "code"` | guarded siblings fire selectively; a join runs on whichever branch fired | `guard_evaluator.py` (equality + skip-propagation) |

The old `routing-demo.yaml` pattern routed *data* (a classifier emits keyed
buckets, downstream nodes select via `input_key`/`fan_out`, but both always
run). With the guard, you can now route *control*: guard each branch on the
router's choice and only the chosen branch runs.

## What does NOT ship (planned)

| primitive | intended YAML | why it is needed | current home |
|---|---|---|---|
| **Bounded loop** | `loop: {body, until, max_iterations}` | retry/repair until a predicate holds or a hard bound trips | impossible today — acyclicity is asserted (`ensemble_config.py:79-118`) |
| **Dynamic dispatch** | runtime-resolved `ensemble:` | invoke a capability chosen at runtime | adapter-mediated (`ensemble_runner.py:66` resolves a static string) |

**Branch is guard.** A branch is a set of mutually-exclusive guards: an upstream
node emits a choice, each downstream guards on `${x.choice} == me`. The guard
predicate is the one primitive that gives both conditional-skip and
route-and-judge, which is why it shipped first.

**Guard semantics:** a node skips iff its `when` is false **or all** its
dependencies skipped; it runs if `when` passes **and at least one** dependency
produced (or it has no deps). Skipped nodes are recorded
(`status: "skipped"`) and excluded from downstream injection. Predicate forms:
`${dep.field}` truthiness, and `${dep.field} == <literal>` (`true`/`false`/
number/quoted string).

**Loop is a combinator, not a cyclic graph.** A `loop:` node wraps a body
sub-DAG and re-runs it under a termination policy. The top-level graph stays
acyclic and analyzable; iteration is scoped inside the node, so per-iteration
results stay unambiguous. Termination =
`max_iterations` (mandatory hard bound) ∧ (`until` predicate ∨ stall-detect ∨
budget). The acyclicity *prohibition* is relaxed; the termination *guarantee*
is kept, and made stronger (every loop has a finite ceiling).

---

## The honest boundary — what stays imperative on purpose

- **The outer turn loop is client-owned.** OpenCode runs one ensemble turn per
  request. Do not put cross-turn iteration in the engine; that loop already
  lives in the client.
- **Runtime-chosen targets** stay adapter-mediated until dynamic dispatch lands
  (the documented §8 dispatch boundary). For a *closed* capability set, prefer
  enumerated guarded branches over a dynamic target.
- **Unbounded iteration: don't.** Only the bounded loop combinator, with a
  mandatory `max_iterations`.

## The anti-pattern this doc exists to kill

A deterministic gate written as a private function in `run_e.py`. A repair loop
written as an async `for` in a harness. Each of these is a graph component that
became invisible by living in Python. When you catch yourself doing it, the
question is not "is my script correct" — it is "which node is this, and why is
it not in the DAG."

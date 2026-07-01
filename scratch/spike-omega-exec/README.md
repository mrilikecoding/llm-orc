# Spike Ω-exec — execution gate as a stage + escalation cascade

**Status:** RAN 2026-06-30. Two findings: (1) the execution gate is a reliable
truth-teller — it caught a structurally-valid but runtime-broken package the
ast gate passed; (2) builder-tier escalation CANNOT fix a fault that lives
upstream of the builder (a contract/prompt defect), which is exactly what this
run hit. See `docs/agentic-serving/proposals/ensemble-serving-architecture.md`
§5/§6.

## Shape

Contract-first (E) build at the CHEAP tier for all files, then an execution
gate (run the package's tests in an isolated dir). On failure, attribute the
culprit and escalate it up the code ladder (cheap 8b → standard 14b → frontier
qwen3.6-plus), rebuild, re-test, loop. Task: `calc` (tokenizer → parser →
evaluator → CLI), chosen because its logic is non-trivial.

## Result — did NOT converge (and that is the finding)

4 escalation rounds, ~30 min, all four code files escalated to frontier, still
FAIL. Final test error:

```
parser.py:11: NameError: name 'BinOp' is not defined
    left: Union[Number, BinOp]
```

The parser (even at frontier) declared `class BinOp: left: Union[Number, BinOp]`
— a self-reference — with no `from __future__ import annotations`. Valid syntax
(ast gate passed), `NameError` at class-definition time (runtime).

**The one-line proof:** prepending `from __future__ import annotations` to the
escalated parser makes the whole suite pass (`2+3*4 = 14`, `(2+3)*4 = 20`). So
the parser *logic* was correct; the only defect was the missing future import.

## Findings

### #1 — The execution gate is a reliable truth-teller. (Core win)

The structural ast gate passed `parser.py` (it parses). Only running it exposed
the `NameError`. The gate never false-passed; after exhausting the ladder it
honestly reported `passed: false`. Executional verification is the quality
signal structural checks cannot provide — confirmed.

### #2 — Builder escalation can't fix an upstream (contract/prompt) defect.

The defect was not builder capability. Two upstream causes: (a) the architect's
contract specified a self-referential annotation without the needed future
import; (b) the dispatch prompt said "use EXACTLY these import statements,
verbatim," which forbade builders from ADDING `from __future__ import
annotations`. Every tier — including frontier — faithfully reproduced the
broken contract. Escalating the builder 8b→14b→frontier on all four files
(~30 min, 4 frontier files) achieved nothing, because the fault lived above the
builder. **The escalation ladder must escalate the right LAYER: when
builder-escalation plateaus, repair the contract / relax the prompt / re-run
the architect — do not keep escalating coders.**

### #3 — The attribution heuristic also flailed.

Rounds 1–2 correctly escalated the named culprit (`parser.py`) to frontier;
once it maxed out and the error persisted, `pick_culprits` fell back to "escalate
all remaining code files," burning frontier tokens on tokenizer/evaluator/cli
that were never the problem. Attribution must recognize "culprit already at top
tier and still failing → escalate a different layer," not escalate bystanders.

## Fixes this points to

1. Dispatch prompt: "use AT LEAST these imports; add any others you need (e.g.
   `from __future__ import annotations` for forward references)" — not "EXACTLY."
2. Architect: handle forward references (emit the future import, or string
   annotations) in the contract.
3. Escalation: add a contract-repair / architect-escalation rung above the
   builder ladder; stop escalating once the named culprit is at the top tier
   and re-route to the contract instead.

## Net

The execution-gate-as-a-stage is validated as the quality signal (it told the
truth). The cascade's escalation policy is not yet right: it must be
layer-aware (builder vs contract vs prompt), or it spends maximal frontier
tokens for nothing — the opposite of the goal.

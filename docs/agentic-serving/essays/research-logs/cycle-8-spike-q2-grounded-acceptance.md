# Spike (Cycle 8 DECIDE) — Q2 Grounded Acceptance: the minimal independent accept gate

**Date:** 2026-07-02
**Cycle:** 8 (agentic-serving; the declarative-ensemble collapse)
**Phase:** DECIDE — grounds the Q2 grounded-acceptance ADR (ADR-048) before it is drafted
**Cost:** $0.00 (local Ollama; qwen3:8b, 32GB-sized, `think:false`).
**Retention:** retained until corpus close per the spike-artifact-retention discipline.
**Methods review:** skipped — feasibility/mechanism spike with pre-stated falsification criteria (below), not a bias-prone measurement spike.

---

## Question

Q2 (§6.2b): what independent correctness signal does the loop's accept/another-round
decision rest on? PLAY signal #3 established the wall: a gameable gate does not vanish
under the outer loop, it relocates to the accept decision, and grounding must be
**independent of the builder**. Concretely: does an accept gate composed of a
deterministic executor seat plus one isolated test-adequacy judge seat plus a guard,
built only from shipped primitives, reject a "gamed" build output and accept a genuine
one, with independence coming from seat isolation and deterministic execution?

## Setup (given vs. built)

- **Given — reuse:** the shipped primitives (script node, LLM node, `when:` guard) and
  the a2 all-8b local config (`agentic-tier-cheap-general` = qwen3:8b). No serving
  transport re-spiked; the gate is ensemble-internal (OpenCode is the caller, not a
  participant in the gate), so the harness (`EnsembleExecutor`) is the correct level.
- **Built — the gate:** `q2-accept-gate` ensemble = `executor` (deterministic: runs
  produced code + tests in-process) → `judge` (isolated LLM: assesses whether the tests
  adequately verify the requirement, `think:false`) → `gate` (deterministic:
  `accept = tests_pass AND tests_adequate`) → guarded `accept` / `revise` terminals.
- **Isolation boundary:** verifier seats receive only `{requirement, produced artifact,
  execution result}`, never builder context. The builder controls neither the judge nor
  the executor. That is the source of independence — architectural isolation within the
  ensemble, the same self-correction-blind-spot principle RDD's own audits exploit.

## Pre-registered falsification (any one kills the minimal claim, and is itself a finding)

- **F1** — the composition needs a new engine primitive → routes to AS-11 (name it).
- **F2** — the 8b judge cannot reliably tell adequate from trivial tests → the
  ensemble-over-frontier boundary: do bounded roles / determinism close it, or is this
  the frontier-fallback line?
- **F3** — the deterministic executor cannot be wired as a script node running code+tests.

## Method

Four fixtures on one small task each, isolating each signal's contribution:

| Fixture | Code | Tests | Isolates |
|---|---|---|---|
| a_correct | correct | real, exercising | both signals green |
| b_gamed_trivial_tests | correct | trivial (`assert callable`) | the judge (executor is fooled — trivial tests pass) |
| c_wrong_code | wrong (missing `+32`) | real, exercising | the executor (deterministic catch) |
| d_false_adequate (probe) | buggy (leap-year, omits century rule) | non-trivial but miss the buggy input | whether the judge does requirement-coverage analysis |

## Result — SUCCESS; no falsifier fired

```
a_correct              executor pass  + judge adequate    → accept  → ship
b_gamed_trivial_tests  executor pass  + judge INADEQUATE  → reject  → another_round
c_wrong_code           executor FAIL  + judge adequate    → reject  → another_round
d_false_adequate       executor pass  + judge INADEQUATE  → reject  → another_round
```

- **F1 false:** composed entirely from shipped primitives. No engine work. Consistent
  with AS-11 ("extend only when a flow needs a shape the catalog lacks; none does").
- **F2 false at this task scale:** qwen3:8b `think:false` classified all four correctly,
  with precise reasons ("only check existence and callability"; "would catch a wrong
  implementation that fails to add 32"; "do not check the century year exception logic").
- **F3 false:** the executor ran code+tests in-process and caught the missing `+32`.

### The load-bearing finding: the two signals are complementary, not redundant

- **b** is the §6.2b gaming vector. The builder's trivial tests *passed execution*, so a
  builder-controlled gate would have shipped it. The **independent judge** caught it.
- **c** is confidently-wrong code. The judge called the tests adequate (they are); the
  **deterministic executor** caught it by running them.

Neither signal alone catches both. Executor-only ships (b); judge-only ships (c). The AND
of the two covers orthogonal failure modes. This is the structural justification for the
composition, and the concrete answer to "what independent signal does the accept decision
rest on": a deterministic execution anchor AND an isolated contract-confidence judge.

### The surprise: the judge does requirement-coverage analysis, not just triviality detection

The false-adequate probe (d) was expected to expose a gap: buggy code whose non-trivial
tests pass because they miss the bug. The judge caught it anyway — it read the requirement
(which named the century exception), saw the tests cover no century year, and flagged the
coverage gap. So the minimal gate has a stronger floor than "reject `assert True`": it
rejects tests that fail to cover **requirement-stated** cases.

That refines where the oracle ladder actually becomes necessary. Four layers:

1. Trivial tests → judge (b).
2. Wrong code, tests exercise it → executor (c).
3. Buggy code, tests miss a **requirement-stated** case → coverage-aware judge (d).
4. Buggy code on inputs the requirement **never states** (true unknown-unknowns) → still
   uncatchable by any inspection; only property-based / held-out / golden oracles reach it.
   This is the real ceiling, and it is un-spikeable by construction.

## Caveats (honest boundaries)

- **Judge power scales with requirement specificity.** The century gap was caught *because*
  the requirement spelled out the century rule. A vaguer requirement might not prime the
  judge to demand that coverage. This is *why* §6.2b lists user acceptance criteria as a
  grounding source: enumerated acceptance criteria are a force multiplier on the judge, not
  one option among several.
- **One stochastic sample per fixture.** The 8b judge's coverage-catch needs consistency
  measurement (across runs and requirement phrasings) before it is trusted as a gate. That
  is BUILD calibration, not this feasibility spike (parallel to Q3's untested hard-task edge).
- **In-process execution** is a spike simplification; a shipped gate sandboxes. BUILD concern.

## Operational finding (for the field-guide)

Root script nodes receive the legacy envelope `{"input": "<json>"}`; dependent script nodes
receive `{"input_data": "<json>", "dependencies": {...}}`. A script that may run in either
position must handle both. Cost one iteration here.

## Forward threads (musing, not committed — feeds the composer-ensemble direction)

The practitioner's reflection (2026-07-02): this is **assume-guarantee reasoning**. A frontier
model amortizes decomposition + generation + verification in one implicit pass over a big
context; an ensemble makes all three explicit and bounded, paying with an explicit
decomposition phase the frontier model got for free. If work is decomposed at different
levels of abstraction and a higher level asserts confidence about the verified contracts of
lower levels, then `think:false` 8B models might compose for generalized work — fine-enough
decomposition substitutes for per-model thinking.

Two boundaries this spike marks on that thesis:

1. **The spike is evidence for the verification half, not the generation half.** The judge
   asserted confidence about an artifact that already existed. Verifying is easier than
   generating (the same generator/verifier asymmetry essay-004's cited small-LM-*verifier*
   work rests on). That a bounded 8B can *generate* each decomposed piece correctly is a
   separate, untested claim.
2. **"Confidence composes upward" holds iff the contracts threaded down are rich and the
   checks are independent.** The false-adequate ceiling and the requirement-specificity
   caveat are the failure mode. The enabler is **contract quality at each boundary** plus a
   **deterministic anchor wherever one exists** (execution is ground truth; an LLM judge is
   correlated-error-prone). Deep hierarchies want deterministic anchors at each level, not
   stacked LLM-confidence assertions.

Q2's accept gate is the two-level base case of this hierarchy (L0 deterministic execution,
L1 contract-confidence judge). The general recursive form is the composer-ensemble direction
ADR-047 §Deferred already parks. Q2's ADR stays scoped to the accept signal; the
hierarchical-confidence generalization is a named thread that lands there.

## Artifacts

- Ensemble: `.llm-orc/ensembles/q2-accept-gate.yaml`
- Scripts: `.llm-orc/scripts/spike-q2-grounded-accept/{executor,gate,terminal_accept,terminal_revise}.py`
- Drive: `scratch/spike-q2-grounded-accept/drive.py`

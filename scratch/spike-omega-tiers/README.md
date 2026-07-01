# Spike Ω-tiers — contract-first (E) + per-sub-task model-tier routing

**Status:** RAN 2026-06-29. The tier-routing architecture works; it produced
the first model-ceiling data AND surfaced that the *structural* gate is not
enough once work is pushed to small tiers (it misses runtime fragility). See
`docs/agentic-serving/proposals/ensemble-serving-architecture.md` §4 (strategy
E / tier routing) and §5 (the gate lever).

## Question

Extend strategy E so the frontier architect assigns each deliverable the
smallest model tier it judges can implement it (against the frozen contract),
route accordingly, and escalate up the tier ladder on gate failure. Does it
still produce a running package, and which tier clears which sub-task (the
ceiling data)?

## Shape

- `architect-tiers.yaml`: frontier emits the contract + a `tier` per file
  (micro=0.6b, small=1.7b, cheap=8b, standard=14b).
- per-tier coders: `code-generator-{micro,small,standard}.yaml` (+ omega=cheap).
- `run_tiers.py`: build each file at its assigned tier; on gate give-up,
  escalate to the next tier up; record the minimum tier that clears the gate.

## Result (the tier-routing run)

| file | assigned | cleared at | notes |
|---|---|---|---|
| models.py | micro (0.6b) | **small (1.7b)** | 0.6b failed the dataclass-contract; escalated |
| storage.py | small (1.7b) | small (1.7b) | gate PASS but runtime-broken (see below) |
| operations.py | cheap (8b) | cheap | first try |
| cli.py | cheap (8b) | cheap | first try |
| test_operations.py | cheap (8b) | cheap | first try |
| README.md | (prose, cheap) | gave up | missed required mentions |

Structural 5/6. **Execution: BROKEN** (3 tests fail). Total ~10.6 min (90s
frontier architect + ~566s cheap-local build).

## Findings

### #1 — Tier routing + escalation works mechanically. (PASS)

The architect assigns tiers, the build routes to them, and escalation climbs
the ladder on gate failure, recording the minimum tier that clears each file.
The mechanism is sound and produced clean ceiling data.

### #2 — First ceiling data point: 0.6b < the dataclass contract; 1.7b clears it.

qwen3:0.6b could not produce `models.py` (a `Task` dataclass with the exact
contract fields) past the gate after retries; it escalated to qwen3:1.7b,
which cleared it first try. So the floor for "implement a precise dataclass
against a frozen contract" sits between 0.6b and 1.7b on this task.

### #3 — The structural gate is insufficient once work goes to small tiers.

The 1.7b `storage.py` PASSED the structural contract gate (imports `Task`,
defines `load_tasks`/`save_tasks`, parses) but does not run: `load_tasks` does
`json.load` with no empty/missing-file guard (`JSONDecodeError`), and
`save_tasks` calls `dict(task)` on a dataclass (fails). The 8b tier in pure-E
wrote robust versions; the 1.7b tier wrote naive ones. **Structural pass ≠
runtime pass, and the smaller the tier, the wider that gap.** This empirically
forces the §5 lever: the gate must become *executional* (compile + run the
tests). Under tier routing, an executional gate would also escalate the fragile
1.7b storage to 8b automatically.

## Implication

Tier routing is the right architecture for the ceiling questions, but it is
only safe with an executional gate. The next gate iteration (run-the-tests as
the acceptance check, escalate on runtime failure) makes both tier-routing and
generalized E produce running packages — and turns the per-tier pass/fail into
a true ceiling measurement (which tier produces code that RUNS, not just
parses).

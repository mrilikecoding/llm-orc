# #157 — engine-run script agents get a real timeout (design)

Status: SHIPPED (2026-08-17). Issue: #157 (found during the #154 review).

## Mechanism (measured, and worse than the issue states)

Two independent defects compose into "no bound at all".

**1. The inner bound is never set.** `ScriptAgent.__init__` reads
`config.get("timeout_seconds", 60)` (script_agent.py:106), but
`agent_runner.py:111` builds the agent from
`agent_config.model_dump()`, and pydantic's `model_dump` ALWAYS emits
the key, supplying `None` when unset. `.get(key, 60)` therefore returns
`None`, not 60, and every `subprocess.run(timeout=self.timeout)` runs
unbounded. Measured:

```
ScriptAgentConfig(name='x', script='s.py').model_dump()['timeout_seconds'] -> None
```

**2. The outer bound cannot fire.** The dispatcher does wrap agents:
`_get_agent_timeout` (agent_dispatcher.py:179) resolves
`timeout_seconds` else `performance.execution.default_timeout` (300 in
this project's config.yaml, 60 as the code default), and
`execute_agent_with_timeout` applies `asyncio.wait_for`. But
`_execute_script_file` is an `async def` that calls BLOCKING
`subprocess.run` directly, with no `to_thread`, so it blocks the event
loop and the timer never gets to run. Measured, with a 6s script under
a 2s outer timeout:

```
self.timeout via the agent_runner path: None
FINISHED in 6.0s -> outer 2s timeout NEVER fired
```

So the issue's headline is right, and the reason is that BOTH bounds are
defeated. Fixing (1) is sufficient to make a bound real, because
`subprocess.run`'s own timeout kills the child. (2) is a separate
concern (it also serializes concurrent agents) and gets its own issue
rather than riding this fix.

## What NOT to do: the schema default (the issue's own recommendation)

Issue #157 proposes giving `timeout_seconds` a schema default of 60 so
the value shows up in a dumped config. That is an active regression
across every agent type, and the issue has been corrected in place.

A schema default of 60 is not `None`, so it stops reading as "unset"
and becomes an agent-level OVERRIDE in the profile merge
(`llm_runner._resolve_model_profile_to_config`), beating both the model
profile and the operator's `default_timeout`. Measured:

```
TODAY   profile-backed LLM agent resolved = 180   (profile's value)
IF-60   profile-backed LLM agent resolved = 60    (profile 180 dead)
TODAY   plain LLM agent resolved = 300            (operator default)
IF-60   plain LLM agent resolved = 60             (operator 300 dead)
```

The `None` default means "defer", which is exactly why it has to stay
`None`.

## The design fork: which number

The narrow fix is `config.get("timeout_seconds") or 60`, restoring the
literal written intent. Measured blast radius for that:

| script agent | worst case |
|---|---|
| classify, resolve, shape, form_gate, emit, seat_contract, accept_gather | < 0.2s each on an 83KB payload (near the serve's 96KB read window) |
| `accept_executor` | **60.0s.** Its budget check runs BEFORE spawning each child (accept_executor.py:612-625), so the worst case is the 45s aggregate budget PLUS one more 15s per-child timeout, not 45s. Needs 3-4 genuinely hanging tests to materialize; the happy path is 1.2s for 20 tests |
| `web_searcher` | unbounded on the DEFAULT backend. kagi/tavily use `timeout=30` per request, but the `ddgs` path (web_searcher.py:190-198) sets no timeout of its own and retries under rate limiting. Its node (web-searcher.yaml:88) sets no `timeout_seconds` either. The one script agent whose runtime a third party decides |

A flat 60 is therefore NOT safe: it exactly equals `accept_executor`'s
worst case, leaving zero headroom rather than the 20% this doc first
claimed. Right conclusion, wrong arithmetic, corrected here.

60 is also the wrong number on principle: this project declares
`performance.execution.default_timeout: 300`, and the dispatcher
already resolves the right value per agent. Two answers to "how long
may this agent take" is the same class of bug being fixed. (There are
in fact more than two: 30 in the composer, registry, resolver, script
handler and script commands; 60 in ScriptAgent and the shipped
template; 300 in this repo's config; plus a dead
`performance.timeout_seconds: 120` nothing reads.)

**Direction: one source of truth, plus explicit budgets where a script
knows its own.**

1. The dispatcher's resolution rule becomes a shared helper, and the
   script-agent runner uses the SAME helper, so the inner
   `subprocess.run` bound equals the bound the ensemble already
   believes it is applying. Explicit `timeout_seconds` still wins; an
   unset agent inherits the operator's `default_timeout`; `or 60`
   survives in `ScriptAgent` only as the standalone floor for a
   directly-constructed agent (which the unit tests do).
2. The two scripts with real budgets get explicit node timeouts rather
   than relying on any default: `accept_executor` (it knows its own
   45+15, so give it headroom above 60) and `web_searcher`. This is the
   honest fix at the point of knowledge, and it is what makes the
   general default safe for everything else.

   That is SIX nodes, not two, and a partial edit silently voids the
   whole rationale on a fresh install:

   - `.llm-orc/ensembles/agentic-serving/build-gated-round.yaml:44`
   - `.llm-orc/ensembles/agentic-serving/build-code-round.yaml:31`
   - `.llm-orc/ensembles/agentic-serving/write-tests-round.yaml:32`
   - `.llm-orc/ensembles/agentic-serving/re-fix.yaml:32`
   - `.llm-orc/ensembles/re-fix.yaml:32` (a byte-identical copy of the
     previous file)
   - `.llm-orc/ensembles/agentic-serving/web-searcher.yaml:88`

   What this choice costs, stated rather than implied: six declarations
   that have to stay in sync with `DEFAULT_TIMEOUT × _BUDGET_MULTIPLIER`
   inside `accept_executor` itself. That duplication is the price of the
   smaller blast radius.

   Rejected alternative: tightening `accept_executor`'s own aggregate
   budget so its worst case fits under 60. It is one file and no
   duplication, but lowering the multiplier changes how many tests run
   before budget exhaustion, so it changes gate VERDICTS — a behavior
   change to the accept gate riding a timeout bug fix.

Step 2 matters because **the SHIPPED default is 60**
(`templates/global-config.yaml:89`, `config_manager.py:323`); this
repo's 300 is local. So "inherit the operator's `default_timeout`"
gives a fresh install exactly the 60 this doc rejects, against an
accept gate whose worst case is 60. Rather than raise a global default
for every agent type (a much larger blast radius than a bug fix
warrants), the two scripts that can actually approach a minute declare
their own.

## Invariant

A script agent's subprocess is bounded by the SAME number the ensemble
resolved for that agent, and an unset `timeout_seconds` never means
unbounded.

Corollary for #158, stated here so it inherits it: **inner ≤ outer, and
the inner bound is the one that reaps.** The outer `wait_for` timer
starts earlier (cache lookup, construction, script resolution) than the
inner subprocess timer, and cancelling a thread does not stop it, so
equal values mean the outer fires first and the inner reaps a moment
later. The reverse configuration would hold a thread and a live child
well past the agent's reported failure.

## Implementation shape (concrete, so it is not re-derived)

- Extract the rule in `agent_dispatcher._get_agent_timeout` into a
  shared `resolve_agent_timeout(config_dict, performance_config)`.
  Dispatcher calls it with `enhanced_config`; nothing else changes.
  STRUCTURAL, its own commit.
- Pass `performance_config` into `ScriptAgentRunner.__init__` (it is set
  at `ensemble_execution.py:228`, before the runner is built at
  `:264-270`) and, in `_execute_without_cache`, fill the dumped dict's
  `timeout_seconds` from that helper before constructing ScriptAgent.
  `agent_runner.py`'s `_execute_without_cache` holds the ONLY
  `ScriptAgent(...)` site in `src/`.
  BEHAVIORAL, second commit.
- Keep a floor in `ScriptAgent` for a directly-constructed agent. NOT
  `or 60`, and not `.get(key, 60)` either: the key is usually PRESENT
  and None, which is the whole bug, and `or` would additionally swallow
  an explicit 0 and desync the inner bound from the outer.
- **Do NOT** widen `execute_agent_with_timeout`'s executor callable.
  That two-arg seam is deliberate and documented as test-patchable at
  `ensemble_execution.py:304-313`; threading through it would mix a
  structural change into a bug fix for no gain, since the resolution
  rule is reproducible from data the runner already can hold (for a
  `ScriptAgentConfig`, `_resolve_profile` returns `model_dump()`
  unchanged, so the dispatcher's number is exactly
  `timeout_seconds or default_timeout or 60`).

## Regression instruments

The thing that broke is the WIRING, not the timeout mechanism, so the
pins go there. A standalone "sleeping ScriptAgent gets killed" pin is
NOT written: it already exists at
`tests/unit/agents/test_script_agent.py:377` and costs 0.10s.

1. **Construction pin** (fails today, ~0ms): build a `ScriptAgentRunner`
   with `performance_config={"execution": {"default_timeout": 7}}`, run
   a trivial script agent whose config has no `timeout_seconds`, and
   assert the constructed ScriptAgent got 7. Same test parameterized
   with an explicit `timeout_seconds=3` covers "explicit still wins" —
   one table, not two instruments.
2. **Behavioral wiring pin** (~1s): `ScriptAgentRunner.execute` on a
   real sleeping script with `default_timeout: 1`, asserting the
   timeout envelope comes back. Real subprocess, no mocking of
   `subprocess.run` — mocking it is exactly what hid this. 1s is the
   floor, not 0.1s, because `timeout_seconds` is `int | None` and the
   perf path does `int(...)`. Assert no wall-clock upper bound, so a
   loaded machine cannot make it flaky.

## Known bounds

- Fixing (1) leaves (2) in place: the event loop still blocks for the
  duration of a script, so a long script still stalls concurrent agents
  and the outer `asyncio.wait_for` remains decorative (#158). This
  change makes the inner bound real, which is what turns an unbounded
  hang into a bounded failure. Verified that the inner bound genuinely
  reaps: `child alive after inner timeout? False`.
- **Grandchildren are orphaned.** `process.kill()` reaches the direct
  child only. Measured with an accept_executor-shaped script (parent
  spawns a 20s grandchild, inner bound 2s): the grandchild reparents to
  PID 1 and keeps running. For `accept_executor` the per-child timeout
  is enforced BY the executor process, so killing it leaves the
  model-authored test runner unbounded. Rare (needs the bound to fire
  mid-suite) and strictly better than today's unbounded hang of both,
  but named rather than discovered later.
- **Timeout envelopes get cached as successes.** `agent_runner.py:88-93`
  caches every result with a hardcoded `"success": True`, and
  `ScriptCache` defaults to enabled with a 3600s TTL. Once timeouts
  produce envelopes, a timed-out run is cached as a success for an hour
  on the same (script, input, parameters) key. Not new in kind (error
  envelopes from `CalledProcessError` already cache today), new in
  frequency. This repo disables the cache.
- The live interactive path raises through to a `status="failed"`
  AgentResult while every other path returns `status="success"` carrying
  an error envelope. Pre-existing asymmetry, but it means an ensemble
  that does not check `success` will thread a timeout envelope
  downstream as if it were data.
- Anything that today relies on a script agent running longer than the
  resolved timeout starts failing. Nothing in this repo does once the
  two budgeted scripts declare their own (table above), and "unbounded"
  was never the declared contract.
- A real timeout is never wrong for the human-pause case: the pause
  happens in `run_in_executor(None, input)` BEFORE the bounded
  `subprocess.run`, so the bound never covers a human. Separately and
  pre-existing: the OUTER `wait_for` can already fire during a human
  pause because that executor call yields, so a fresh install's
  `human-in-loop-validation` ensemble already cancels an expert who
  takes over 60s to type. Out of scope, recorded because it is the one
  place a real timeout genuinely is wrong.

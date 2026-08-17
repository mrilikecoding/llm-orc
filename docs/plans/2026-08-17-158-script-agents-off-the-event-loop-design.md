# #158 — script agents stop blocking the event loop (design)

Status: pre-flight APPROVED (design B, after five rounds). Issue: #158,
found while grounding #157.

## Mechanism (measured)

Five `subprocess.run` calls sit directly inside `async def` methods of
`ScriptAgent`, with no thread hand-off: `_execute_script_file` (563),
`_execute_inline_script` (616), `_execute_command_with_json` (653),
`_execute_script_file_with_schema_json` (684), and
`_execute_inline_script_with_schema_json` (709). Each blocks the event
loop for the whole life of the subprocess. Two consequences:

**1. Agents the engine schedules concurrently run serially.**
`agent_dispatcher.py:84` gathers a phase's agents, so they are meant to
overlap. Measured:

```
4 x 1s script agents via asyncio.gather: 4.15s   (should be ~1s)
```

The win is real rather than theoretical: running the real
`DependencyAnalyzer` over 127 ensembles finds **seven phases that
already carry 2+ script agents** (plexus graph-analysis, five validation
ensembles), max phase width 15. Any model-backed agent awaiting in the
same loop is stalled for the script's whole duration too.

**2. The agent-level `asyncio.wait_for` cannot fire.** Measured (in
#157): a 6s script under a 2s outer timeout ran to completion, and a
loop-tick counter registered `0 ticks` while the script ran.

## Change

Three things, and deliberately nothing else:

1. The five sites run through
   `loop.run_in_executor(SCRIPT_POOL, functools.partial(subprocess.run,
   ...))`.
2. `SCRIPT_POOL` is a module-level
   `ThreadPoolExecutor(max_workers=min(32, cpu + 4),
   thread_name_prefix="llm-orc-script")`.
3. `_get_agent_timeout` returns `None` for script-shaped agents.

**Not `asyncio.to_thread`.** It is hardcoded to
`run_in_executor(None, ...)`, the DEFAULT pool, so it would silently
bypass the dedicated one. Measured:

```
asyncio.to_thread(...)          -> asyncio_0
run_in_executor(DEDICATED, ...) -> llm-orc-script_0
```

`functools.partial` keeps the no-lambda-closing-over-loop-vars property
that made `to_thread` attractive. The pattern already exists here:
`agent_runner.py:326` runs the interactive path's subprocess through
`run_in_executor`, which is why that one path is already non-blocking.
Converting `:326` for uniformity is structural and NOT bundled.

The dedicated pool earns its place independently of anything else:
script agents stop competing with `models/google.py:46`,
`models/anthropic.py:114`, `input()` at `agent_runner.py:284`, and the
serve's trace flushes.

## Point 3 is removing the DUPLICATE bound, not the bound

Read quickly, "script agents get no timeout" reads like a regression, so
state it plainly: **the authoritative bound is untouched.**
`agent_runner.py:120-122` still calls `resolve_agent_timeout` and still
passes it to `subprocess.run`, which is the bound that actually reaps a
runaway child. #157's "one rule, one home" survives intact — the rule
still has exactly one home. What is retired is the second APPLICATION of
that number, as an outer `wait_for` that could never do the job.

Three reasons it could never do the job:

- **It cannot reap.** Cancelling the awaiting coroutine does not touch
  the worker thread or its child.
- **It cannot fire during the only window it uniquely covers.** The
  pre-subprocess window (cache lookup, script resolution) is exactly the
  window that blocks the loop. Measured: `0 ticks` while the script ran.
  So B's apparent cost — leaving that window unbounded — is not a
  regression but an accurate description of today.
- **It measures the wrong thing under load.** `wait_for`'s timer starts
  when the coroutine is awaited, so any queue delay is charged to the
  agent's budget.

`resolve_agent_timeout` returning `int` is why the coordinator's
existing `None` branch (`agent_execution_coordinator.py:32-33`, pinned
at `test_agent_execution_coordinator.py:45-48`) is currently dead in
production. This makes an existing, tested branch live.

**#157's `inner ≤ outer` invariant is RETIRED for script agents, not
vacuously satisfied.** Retired with a reason is a decision; vacuously
satisfied reads like an oversight.

### The error contract does NOT move, and that is load-bearing

Today the loop is blocked, so the outer never fires, and a script
timeout returns `ScriptAgent.execute`'s envelope
(`script_agent.py:162-168`) carried on `status="success"`. B preserves
that exactly.

This is the strongest argument against the alternative considered below.
Under that alternative the outer timer would start earlier and always
win, so `execute_agent_with_timeout` would raise, the dispatcher would
catch, and the result would become `AgentResult(status="failed")`. That
shape has real consumers: `results_processor.py:127,224,243,299`,
`phase_result_processor.py:58,90,115,123`,
`fan_out/coordinator.py:44`, `artifact_manager.py:153,213`. The
fan-out coordinator SKIPS expansion when upstream status is not success,
so a timed-out script would become a silently unexpanded fan-out — a
verdict-shaped change riding a bug fix.

### It also closes a bug #157 recorded and declined to fix

From the #157 design's Known Bounds: the outer `wait_for` already fires
during a human pause, because `run_in_executor(None, input)` yields, so
a fresh install's `human-in-loop-validation` ensemble already cancels an
expert who takes over 60s to type. #157 called it "the one place a real
timeout genuinely is wrong." Interactive agents are script-shaped, so
this removes it, including the case where one is queued on `_input_lock`
behind another human and gets cancelled for someone else's slowness.
#157's Known Bound is retired by cross-reference.

## The gate that was designed and dropped (kept deliberately)

Four review rounds designed a thread-slot gate to keep queue delay out
of the timed region: a per-loop semaphore sized to the pool, acquired
before the timeout applied. It was dropped because removing the timed
region removes its entire job. The record stays because the measurement
below is the REASON `_get_agent_timeout` returns `None`, and without it
the next reader will "fix" that line.

With an outer timer and a saturated pool, agents report timeouts they
never experienced (20 agents, 16 slots, inner == outer == 3s, script
10s):

```
agent  0: TIMEOUT after 3.00s of work
agent 16: TIMEOUT after 0.02s of work     <- ran for 10ms
asyncio.run() RETURNED at +6.02s
```

Reachable by design: `fan-out-test.yaml:8-11` sets `fan_out: true` on a
SCRIPT agent, the coordinator expands per upstream element with no cap,
and `max_concurrent_agents` ships as `0` and is never set here.

The gate's own failure modes, each measured, are why it stayed complex:
a per-dispatcher semaphore over a module-level pool over-permits by the
number of live dispatchers (3 dispatchers, 8 slots → 16 of 24 agents
falsely timing out), and child executors are the normal case
(`loop_runner.py:73` builds one PER ITERATION); a module singleton binds
to the first loop that contends it and breaks the second test in a
pytest-asyncio process; a permit scoped to the coroutine outlives its
slot under cancellation; and a permit acquired in the dispatcher but
released six layers down leaks permanently on any path that returns
before creating a subprocess — a cache hit, which is the DEFAULT path,
draining the gate to zero and hanging every script agent in the process.

Without a timed region, none of that has to be solved. The pool's own
`max_workers` is the bound, its submit queue is FIFO, and waiting in it
is honest waiting rather than a budget being consumed.

## Invariant

An `async def` in the script-execution path never blocks the event loop
for the duration of a subprocess, and a script agent's only timeout is
the one that can actually reap its child.

## What this does NOT fix, and must be said plainly

**Cancelling does not kill the child, and the orphan outlives the run.**
`Runner.close()` calls `shutdown_default_executor()`, which joins only
the DEFAULT pool; a module-level dedicated pool is joined by
`concurrent.futures`' atexit hook, at interpreter exit:

```
DEFAULT pool:   asyncio.run() returned at +4.01s   (joined at loop close)
DEDICATED pool: asyncio.run() returned at +0.50s   (NOT joined)
                live threads now: ['llm-orc-script_0']
                [interpreter exit reached at +8.01s]
```

The symptom moves one layer out and gets harder to attribute: the run
completes, the CLI prints its result, and then the process sits before
the shell prompt with nothing on screen to blame.

**Correction from the merge review, because an earlier draft had the
direction backwards.** This makes orphaning rarer only for a script
agent's OWN cancellation. One level up it becomes NEWLY reachable: a
parent's `wait_for` can now fire mid-script, which it could not before,
because a blocked loop meant a script always ran to completion inside
its ancestor. Measured:

```
BRANCH: PARENT TIMED OUT at +1.00s
        live script threads: ['llm-orc-script_0']
        subprocess ran to completion anyway: True
MAIN:   no timeout fired at all
```

Live in this repo, not theoretical: `build-gated.yaml:22-28` wraps a
`loop:` in `timeout_seconds: 660` whose body reaches `accept_executor`
(180s, spawns pytest), and the same shape appears at `serving.yaml:118`,
`build-round.yaml:22`, `code-seat.yaml:17`, `re-fix.yaml:23`,
`build-code-round.yaml:19`, `write-tests-round.yaml:25`. Not a
correctness bug — the orphan is bounded by the inner timeout and cleaned
at interpreter exit, and concurrency strictly REDUCES a parent's wall
clock, so parents become less likely to time out, not more.

Verified NOT to compound: `subprocess.run`'s POSIX timeout path does
`kill()` then `wait()`, not a second `communicate()`, so a grandchild
holding the stdout pipe does not extend the thread past the inner bound.

**Ctrl-C stops killing the child from Python's side.** CPython delivers
`KeyboardInterrupt` to the main thread only:

```
MODE=main   run() raised KeyboardInterrupt at +1.27s  child alive? False
MODE=thread [main] KeyboardInterrupt at +1.01s        child alive? True
```

The common case survives, because a terminal Ctrl-C signals the whole
foreground process group. What does not: a script or grandchild that
traps or ignores SIGINT (pytest runners, accept_executor's children), a
programmatic `kill -INT`, uvicorn's shutdown, or anything that changed
its process group.

**Why not `create_subprocess_exec`,** the only mechanism that makes
cancellation genuinely work: `ScriptAgent.execute` catches
`subprocess.TimeoutExpired` and `CalledProcessError`
(`script_agent.py:162-176`) and turns each into a distinct envelope, and
10 test sites patch `subprocess.run` directly. Reimplementing
`check=True`/`timeout=` semantics and re-raising those exact types is an
error-contract rewrite riding a bug fix. The Ctrl-C bound is its price.

## Blast radius: concurrency becomes real

Side-effect scan across `.llm-orc/scripts/**`, `llm-orchestra-library`,
and the shipped primitives — write modes, `mkdir`, `shutil`, `chdir`,
`os.environ[...] =`, sockets, `flock`:

- Serving scripts are pure functions of stdin; the library's are pure
  stdin→stdout; `prepare_environment` copies `os.environ` and never
  mutates it.
- `accept_executor` is the only in-repo writer, into a
  `TemporaryDirectory` with `cwd=tmp`. No fixed shared path.
- **Caveat**: this changes EVERY script agent, and
  `src/llm_orc/primitives/file_ops/write_file.py:76` defaults to a
  cwd-relative `"output.txt"` when a node declares no `path`. All ten
  in-repo users pass an explicit path, but two concurrent `write_file`
  nodes without one would race a fixed file, on a fresh install, in a
  primitive we ship.
- `agent_runner`'s `_input_lock` serializes the RUNNER's interactive
  path, so two agents cannot prompt a human at once. Unchanged. That is
  a different path from `ScriptAgent._execute_script_interactive`
  (`script_agent.py:353-430`), which has no lock and blocks harder — out
  of scope because it has NO production caller in `src/`
  (`execute_with_user_input` is invoked only from tests), not because it
  is protected.

## Regression instruments

Measured red/green, built against the real `ScriptAgent`:

| pin | now | fixed |
|---|---|---|
| 1. 4 x 1s agents gathered | 4.18s | 1.04s |
| 3. loop-tick counter around the call | 0 ticks | 75,440 ticks |

1. **Concurrency**: assert `< 2.5s` for N=4 — a stated number leaving
   room for a 2x-loaded machine either way. Timing assertions have
   precedent (`test_serving_accept_gate.py:125`,
   `test_issue_24_script_agents.py:1812`, #157's 1s pin).
2. **The INNER bound still reaps**: a script whose subprocess hangs
   fails at its inner bound and returns the timeout envelope, while the
   loop stays responsive. Replaces the old "outer bound fires" pin,
   which B makes wrong.
3. **Loop responsiveness**: snapshot a tick counter immediately before
   and after the blocking call, not merely read it at the end. With
   pin 2 changed, this is the only DIRECT proof the loop is free.
4. **Honest saturation**: pool + 4 agents in one phase; all complete
   correctly, none reports a spurious failure. Replaces the gate pin.
5. **The behavior change itself**: `_get_agent_timeout` returns `None`
   for a `ScriptAgentConfig` and an `int` for an `LlmAgentConfig`, one
   table. ~0ms, and the first thing a merge reviewer looks for.
6. **The contract did not move**: a timed-out script agent comes back as
   `status="success"` carrying `{"success": false, "error": "Script
   timed out after N seconds"}`, the same shape as today. This is what
   stops someone "fixing" it to a failed status later.
7. **The dedicated pool is used**: assert the worker thread's name
   carries the prefix. With no semaphore, it is the ONLY symptom if the
   work lands in the wrong pool.

**Retire or repoint the existing BDD step first.**
`issue-24-script-agents.feature:141-143` already asserts "scripts should
execute concurrently" and "bounded by the slowest script", implemented
at `test_issue_24_script_agents.py:1805-1822` as `duration < 5.0` over
three script refs that DO NOT EXIST, so it passes today under full
serialization. A green test asserting the invariant being added is how
this regresses silently.

## Known bounds

- Cancellation orphans (above); #159 inherits this.
- Ctrl-C no longer reaps from Python's side (above).
- A script agent hanging BEFORE its subprocess (cache lookup, script
  resolution) has no bound. Unchanged from today, where the blocked loop
  prevents the outer timer from firing anyway.
- **An INTERACTIVE script agent can now hang forever.** The body above
  celebrates retiring #157's human-pause bound, which is the intended
  fix on a TTY; the cost belongs here too. `agent_runner.py:284` awaits
  `run_in_executor(None, input)`, which YIELDS, so the outer timer
  genuinely fired there before and this change removes the only bound.
  With stdin closed, `EOFError` is raised and handled; with stdin an
  open pipe that never delivers (serve, some CI runners), it blocks
  indefinitely. Reachable configs ship:
  `.llm-orc/ensembles/testing/user-input-test.yaml`,
  `validate-all-primitives.yaml:39,47`, and five library ensembles.
- **Identical concurrent script agents no longer dedupe through the
  cache.** Serialization was silently collapsing them: 20 agents with
  the same script, input, and parameters used to produce 1 subprocess
  run and 19 cache hits; now 20 runs. For cache-identical work that is a
  wall-clock regression, not a win, and it amplifies the `write_file`
  default-path race above, since identical nodes that collapsed to one
  execution now genuinely race. No shipped ensemble hits it (all seven
  wide phases have differing parameters). Lands in the subsystem #159 is
  already open against.
- **No operator control over script concurrency.** `SCRIPT_POOL_SIZE` is
  a module constant with no config knob, and `max_concurrent_agents`
  ships as `0`. On a 12-core box that is 16 concurrent script
  subprocesses; a fan-out of `accept_executor` would run 16 pytest trees
  at once where it previously ran one. Worth a knob on the 32GB target
  rig if that shape ever appears.
- `services/handlers/script_handler.py:154` runs `subprocess.run` inside
  an `async def` on the MCP-facing path; NOT in scope, blocks whichever
  loop serves MCP.
- Scope is otherwise complete: an AST scan of every `async def` in
  `src/llm_orc` for blocking calls outside a `to_thread`/executor
  argument finds the engine's hot path is exactly these five sites.

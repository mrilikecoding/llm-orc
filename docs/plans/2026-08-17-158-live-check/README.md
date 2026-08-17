# #158 live check — the serve builds with script agents off the loop (2026-08-17)

A smoke check, not a gate: there is no honesty claim to falsify here.
What it confirms is that moving every script node onto a dedicated
thread pool, and removing their outer `asyncio.wait_for`, does not
break the pipeline that previously ran them inline.

Serve restarted on the branch, venv on PATH. Build ask through real
`opencode run`: one `write` tool_call, `add.py` on disk
(`written-add.py`), "Wrote add.py."

Trace evidence that the gated path ran rather than being inferred from
the write (`.llm-orc/.serve-trace/turns.jsonl`, untracked, quoted):

```
"target": "code-seat"
"tests_pass": true
"accept_reason": "tests pass and are adequate"
```

`tests_pass: true` comes from `accept_executor`, which is itself a
script agent, so this exercises the changed path including its
subprocess-in-a-thread and its inner timeout as the only bound.

## The measurement that matters

The instrument with teeth is not this turn, it is the concurrency pin:

```
4 x 1s script agents via asyncio.gather:  4.15s before,  1.03s after
loop-tick counter while a script runs:    0 ticks before, 75,440 after
```

The ladder battery was NOT re-run. Serving script nodes are sub-second
(measured during #157's design at under 0.2s each on an 83KB payload),
so neither the pool nor any timeout is reachable by them, and a battery
would re-measure the model rather than the change.

`add.py` was removed from the repo root after capture.

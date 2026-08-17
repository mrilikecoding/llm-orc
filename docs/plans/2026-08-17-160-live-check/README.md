# #160 live check — the serve builds with the new cache identity (2026-08-17)

A smoke check, not a gate. There is no honesty claim to falsify here,
and the cache itself cannot be exercised live: this repo sets
`script_cache.enabled: false`, and #160 makes that the shipped default
too. What this confirms is that the changes on `ScriptAgentRunner.execute`
— which every script node in the serving ensemble goes through — do not
break the pipeline.

That matters more than it sounds, because round 4 moved a check ahead of
the identity computation and thereby changed control flow: with the
cache off, `_requires_user_input` is no longer called for the
cacheability decision. A unit test caught the count change; this
confirms the pipeline still runs.

Serve restarted on the branch (`fix/160-cache-identity`, `7da3dd2c`),
venv on PATH, port 8765. Build ask through real `opencode run --format
json`: one `write` tool_call, `mul.py` on disk (`written-mul.py`),
"Wrote mul.py."

Trace evidence that the changed path ran, rather than being inferred
from the write (`.llm-orc/.serve-trace/turns.jsonl`, untracked, quoted):

```
"target": "code-seat"
"tests_pass": true
"accept_reason": "tests pass and are adequate"
```

`tests_pass: true` comes from `accept_executor`, which is itself a
script agent, so the changed `execute` ran including the new
`_cache_is_enabled()` short-circuit and the `cache_identity` skip.

## What this does NOT show

The ask requested a test file alongside `mul.py` and got one write. That
is the model's shaping decision and is out of scope here; the same
single-write shape appears in the #158 live check.

More importantly, no live run can exercise the cache, so the pins are
the only coverage for every behaviour this issue changed. That is an
argument for them being end-to-end through the runner and through the
real `ExecutorFactory`, which is where round 3 and round 4 put them,
rather than unit-level on `ScriptCache`.

## The measurements with teeth

Not this turn. They are the mutation runs recorded in the design doc:
seventeen instruments, and for the round-3 and round-4 findings, one
mutant each that survived the entire suite before its pin and dies
after — including the `if cacheable` deletion at the cache get, the
`persist_to_artifacts` constant, and `os.path.isfile` swapped for
`os.path.exists`.

The ladder battery was NOT re-run. Nothing here changes what a seat
decides; it changes what is stored between runs, and with the cache off
nothing is stored at all.

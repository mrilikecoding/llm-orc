# #168 — a refusal reason names no filesystem path (design)

Status: pre-flight. Issue: #168, split out of #155 Arc C.

## What is measured

**Reproduced on current main**, through the real
`shape.py -> form_gate.py -> emit.py` chain with the engine's actual
four-key wrap in the `resolve` dep:

```
Refused: serving pipeline error: no readable routing decision this turn
(resolve: Schema JSON execution failed: Command
'['/Users/<user>/Development/eddi-lab/llm-orc/.venv/bin/python3',
'/Users/<user>/.../agentic_serving/classify.py']'
returned non-zero exit status 1.); nothing was built or written

leaks home dir : True   leaks username : True   leaks argv : True
```

**Ten real captures** in `.llm-orc/.serve-trace/turns.jsonl` carry the same
shape — lines 150, 153, 469, 574 and others, across `shape`, `resolve` and
`seat_contract`. The recorded interpreter is the bare `python3` of the
pre-#154 era; the script path leaks the home directory and username either
way.

**One reader, not four.** #155 Arc C says "all four reason builders", but
`.get("error")` has exactly one call site in the serving scripts:
`shape._routing_failure_reason`. Arc A's reason strings are static literals,
`accept_gate`'s reason is composed from its own constants, and the executor's
report is clean on the paths that reach the wire (measured: a candidate that
fails to load yields `code failed to load: ModuleNotFoundError(...)`, no
path). So the change is one function.

## The issue's proposed strip is a denylist, and it fails open

#168 proposes cutting "the whole `Command '[...]'` clause". Measured, the
wrap's `str(e)` has at least three families:

```
CalledProcessError   Command '[...]' returned non-zero exit status 1.
TimeoutExpired       Command '[...]' timed out after 0.2 seconds
FileNotFoundError    [Errno 2] No such file or directory: '/nope/missing.py'
```

The third carries an absolute path and has no `Command` clause at all, so a
clause-strip passes it through verbatim. That is the denylist failure this
corpus keeps paying for (#152, #155 Arc A): an unknown failure shape sails
past a rule written against the shapes someone happened to enumerate.

## Change

**Positive extraction, not a strip.** Recognise the useful residue and emit
only that; anything unrecognised contributes NO verbatim text and the reason
says the node failed without quoting it.

Two recognised residues, both closed and both numeric — neither can carry a
path by construction:

| wrap error | client sees |
|---|---|
| `... returned non-zero exit status N.` | `resolve exited non-zero (status N)` |
| `... timed out after N seconds` | `resolve timed out after N seconds` |
| anything else | `resolve failed` |

`turn_trace.py` keeps the raw node responses server-side, so the operator
loses nothing; the client stops receiving the filesystem layout.

## Invariant

A refusal reason on the wire contains no absolute path and no username,
whatever the node failure looked like.

## Regression instruments

1. **The captured wrap produces a reason with no absolute path and no
   username.** Red today. Driven end to end through
   `shape -> form_gate -> emit`, since a node-level pin does not prove the
   chain (#155's lesson).
2. **An UNRECOGNISED error shape leaks nothing either** — the
   `FileNotFoundError` family, which the issue's own proposed strip would
   pass through. This is the pin that distinguishes positive extraction from
   a denylist, and it is the one that fails if a future edit reaches for a
   clause-strip.
3. **The actionable tail survives** for both recognised families, so the
   sanitiser is not just deleting the reason. Named separately per family,
   because one regex covering both is how a family gets silently dropped.
4. **The failing node is still named** (`resolve` vs `classify`), which is
   the part of the reason an operator routes on.
5. **`turn_trace` still records the unsanitised text server-side.**
6. **A healthy turn is unaffected** — the over-refusal direction. Labelled
   as such: it cannot fail under deletion of the sanitiser.

## Known bounds

- Says nothing about reject-template machinery text (#142); this is only the
  engine wrap's error.
- `accept_executor._run_one`'s `runner crashed: <stderr>` branch would put
  stderr on the wire verbatim if it fired, and stderr can name paths. Not
  reached by any measured path — the load-failure and test-failure reports
  are both clean — so it is recorded rather than fixed here.
- The sanitiser is in `shape.py`, which is deliberately stdlib-only so it
  still runs when the `llm_orc` import is broken (#154's failure mode). Only
  `re` is added, which is stdlib.
- **Found while pinning, not fixed here:**
  `_routing_failure_reason`'s `classify` branch cannot fire in the shipped
  skeleton. `serving.yaml` gives shape `depends_on: [resolve, seat,
  seat_contract]`, so shape never receives a `classify` dep, and a crashed
  classify arrives as resolve's laundered empty target — producing the
  generic reason with no node named. The leak invariant holds on that route
  either way (measured, both routes pinned), so this is #95 dead-surface
  material rather than a defect. Recorded in the end-to-end pin's comment so
  the next reader does not read the branch as live.
- One pattern per FACT rather than per producer wording. `script_agent.py`
  states the same two facts three ways — the blanket except's stringified
  subprocess exception, its own `Script failed with exit code N` envelope,
  and its `Script timed out after N seconds` envelope. The second was found
  by an existing pin going red, not by reading; a fourth wording would
  degrade to `failed`, which is the right direction.

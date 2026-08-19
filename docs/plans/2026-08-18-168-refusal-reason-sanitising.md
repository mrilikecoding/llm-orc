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

**The engine wrap is not the only wire channel** (review round 1). The first
draft recorded `accept_executor._run_one`'s `runner crashed: <stderr>` branch
as "not reached by any measured path". It is reached: a produced module that
kills the runner mid-write puts a TRACEBACK naming the runner's absolute path
into stderr, and `refix_envelope` binds the executor report straight to
`accept_reason` (unlike `build_gated_envelope`, which uses `accept_gate`'s own
constants), which emit ships as `Another round needed: {reason}`. `re-fix` is
a live wired route. Two sibling channels leak the same way:
`code failed to load: {error!r}` and `tests failed to load: {error!r}` carry
any path inside an exception message, and a `SyntaxError` repr carries its
filename tuple.

All three are closed here rather than recorded as bounds, because the stated
invariant is about the wire and not about one producer. Same rule as the
engine wrap: emit what cannot carry a path — the exception CLASS name (a
Python identifier) and the exit code (a number) — and nothing verbatim.

**One reader of the ENGINE WRAP, not four.** #155 Arc C says "all four reason builders", but
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
5. **`turn_trace` still records the unsanitised text server-side** — asserting
   the RESIDUE that was removed from the wire, not the wrap's prefix. Review
   round 1: the prefix is the first 28 characters, and the 280-char snippet
   clipped the residue off every node (the wrap runs 302-310 chars on a real
   checkout), so the pin was green while the operator had LESS than the client
   used to. `turn_trace` now records the wrap's `error` whole.
6. **A healthy turn is unaffected** — the over-refusal direction. Labelled
   as such: it cannot fail under deletion of the sanitiser.

## Known bounds

- Says nothing about reject-template machinery text (#142); this is only the
  engine wrap's error.
- The "numeric by construction" safety argument now has its own pin, over
  the OUTPUT SHAPE rather than any input. Review found a mutant that survived
  every other pin: widening the timeout capture to `(.+?)` re-opens a
  verbatim channel, and a shape assertion catches it whatever it captures.
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

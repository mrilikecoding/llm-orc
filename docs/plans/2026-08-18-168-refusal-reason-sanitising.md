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

**Four, not three** (review round 2). The first pass sanitised the two load
failures and the runner crash and missed `_run_test_fns`, twenty lines above
— the FAILING-TEST path, which is the ordinary outcome of a re-fix round
rather than an exotic crash. Produced code that shells out to
`sys.executable` (the serve's own interpreter inside the sandbox) puts the
home directory and username in a `CalledProcessError` repr with no
cooperation from the client. The `unittest.TestCase` dialect is a fifth: its
last traceback line is `Type: {str(exc)}`, and `OSError.__str__` includes the
filename its `repr` hides.

All of them are closed here rather than recorded as bounds, because the
stated invariant is about the wire and not about one producer — and the
first pass proved that sanitising producer-by-producer is how one gets
missed. Same rule as the
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

A refusal reason on the wire contains no ABSOLUTE PATH, whatever the node
failure looked like.

Scoped to paths deliberately (review round 3). The separator rule is a real
property — every absolute path on either platform contains one, so it cannot
pass a path — but it delivers only that half. A bare username has no
structure to recognise, and an ordinary-looking produced test closes over
one: `assert whoami() == 'appsvc', 'service runs as ' + whoami()` reaches the
wire intact, as does `socket.gethostname()`. Stating the invariant over
usernames would assert something a two-line test refutes; the honest position
is that it is not closed and is not closeable by this rule.

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
- A `SyntaxError`'s line and column are KEPT. Dropping them made the trade
  worse than it needed to be: they are integers so they cannot carry a path,
  the compile filename is the literal `solution.py`, and the executor's
  report is clipped server-side by the trace snippet once it exceeds 280
  characters — so for a long report the class name alone would have been the
  only surviving record anywhere. Short reports are retained whole.
- **Any path-free string produced code can read server-side still reaches
  the wire** — a username, a hostname, and (the reason it is not merely
  cosmetic) an environment value, since the sandbox inherits the parent
  environment. Named in the invariant above rather than buried here, because
  the first draft claimed otherwise. Not closeable by a separator rule: a
  bare identifier has no structure to recognise, and enumerating what is
  sensitive is the denylist this issue was paid twice to avoid. Filed as
  **#175**, which also carries the environment-scrubbing direction.
- **The executor trusts the runner's stdout.** `_run_one` parses the child's
  JSON and takes `report` and `tests_pass` on trust, so produced code can
  write that JSON itself and `os._exit(0)` — forging a leak, or an accept.
  That is the boundary of the wire property as written: it is enforced
  inside the runner. Pre-existing, deliberate intent required, and it
  belongs to #85's containment surface; recorded so the next sweep does not
  rediscover it as new.
- The property is enforced on what is EMITTED, at every emission point, and
  on each PIECE as it enters the string (rounds 3 and 4). Checking only the
  composed string discarded the class name and the message whenever the
  source echo named a RELATIVE path, reducing an ordinary failing test to
  "failed" — the evidence loss round 2 paid to avoid, and it feeds the retry
  prompt too. `_wire_safe` is also total in its return: a caller cannot
  reintroduce the hole by choosing a fallback built from unchecked values,
  which is exactly how the seventh channel appeared. It is dead with respect
  to the suite — every caller passes a fallback built from checked pieces or
  a constant — and it stays because `_safe_reason`'s collapsed second check
  and the `"; ".join(failures)` are both sound only if the return is
  path-free by POSTCONDITION rather than by caller care. Round 2 introduced
  the rule but checked it at the producer, and that leaked twice through
  derived strings: a multi-line message whose last traceback line has no
  colon, and a class name produced code controls.
- **A dead SEAT is not covered, and it is a real leak.** On a non-build route
  `shape._envelope_deliverable` does not recognise the engine's failure
  envelope, falls back to the raw terminal, and emit ships the wrap — with
  its traceback and script path — as the assistant's ANSWER. Not a refusal
  reason, which is why this issue's sweep did not reach it. Filed as **#174**,
  with the observation that the build routes are safe only because they all
  declare `seat_contract:` blocks.
- The design said serving nodes always route through
  `execute_with_schema_json`. True only for TOP-LEVEL nodes: sub-ensemble
  script nodes take `ScriptAgent.execute`, whose envelope puts the payload in
  `stderr` rather than `error`. `_engine_failure_summary` handles that
  family's `error` by accident of the `failed with exit code` alternative;
  nothing scrubs or retains its `stderr`. Carried on #174.
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

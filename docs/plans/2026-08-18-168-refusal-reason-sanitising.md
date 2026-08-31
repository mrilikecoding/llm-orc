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
way. (F7: `.serve-trace/` is gitignored — those captures are not in the
repo and are not reproducible from a clean checkout. The regression pins
below do not depend on them; each drives the leak end to end from a fault
injected in-process, independent of any historical capture.)

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

F8 (round 8 confirmation review): this section named no `test_*` identifier
while the arc shipped 21 pins (25 after round 8), so `scripts/check_doc_drift.py`
(loop-protocol rule 19, added after this branch started) had nothing to
resolve. Every pin below is named in backticks and drawn from the tree —
`grep -n "^def test_" tests/unit/serving/test_serving_shape.py` plus the
same over `tests/unit/web/test_serving_ensemble_endpoint.py`.

1. **The captured wrap produces a reason with no absolute path and no
   username.** Driven end to end through the real engine, since a
   node-level pin does not prove the chain (#155's lesson):
   `test_a_crashed_routing_node_refuses_without_naming_a_path`. The
   shape-level pin behind it, against the wrap directly:
   `test_a_wrapped_exit_status_reason_names_no_path_or_user`.
2. **An UNRECOGNISED error shape leaks nothing either** — the
   `FileNotFoundError` family, which the issue's own proposed strip would
   pass through. This is the pin that distinguishes positive extraction from
   a denylist, and it is the one that fails if a future edit reaches for a
   clause-strip: `test_an_unrecognized_error_shape_leaks_nothing_either`.
3. **The actionable tail survives** for both recognised families, so the
   sanitiser is not just deleting the reason. Named separately per family,
   because one regex covering both is how a family gets silently dropped:
   `test_the_exit_status_tail_survives`, `test_the_timeout_tail_survives`.
4. **The failing node is still named** (`resolve` vs `classify`), which is
   the part of the reason an operator routes on:
   `test_the_failing_node_is_still_named`.
5. **`turn_trace` still records the unsanitised text server-side** — asserting
   the RESIDUE that was removed from the wire, not the wrap's prefix. Review
   round 1: the prefix is the first 28 characters, and the 280-char snippet
   clipped the residue off every node, so the pin was green while the
   operator had LESS than the client used to. `turn_trace` now records the
   wrap's `error` whole: `test_the_unsanitised_engine_error_is_still_recorded_server_side`.

   F6 (round 8): this bullet previously claimed "the wrap runs 302-310 chars
   on a real checkout" as one fixed range. Falsified — the wrap embeds TWO
   paths (the interpreter's and the produced script's), so its length is
   `81 + len(sys.executable) + len(<produced script's path>)` (81 is the
   fixed surrounding text: `Schema JSON execution failed: Command '[`, `',
   '` between the two paths, and `']' returned non-zero exit status N.`).
   It depends on BOTH the checkout root's length (`sys.executable` lives
   under it) and the platform's temp-dir convention for the produced
   script's path, not on the checkout alone. Measured by adding a print of
   the raw `error` field's length and its two path lengths to
   `test_the_unsanitised_engine_error_is_still_recorded_server_side` and
   running it: 344 chars, stable across 3 runs, on this worktree checkout
   (root `/Users/.../llm-orc/.claude/worktrees/agent-<id>`, 89 chars, macOS
   `/private/var/folders/.../T/pytest-of-<user>/...` temp convention).
   Solving the same formula for the canonical 47-char checkout root
   (`/Users/<user>/Development/eddi-lab/llm-orc`) gives 302 — at the LOW end
   of the range this bullet used to attribute to worktree roots, which is
   the tell that checkout-root length is one input to this number, not the
   whole story; the platform's temp-dir convention is the other, and it is
   not this repo's to control. Take the dependence, not the digits: a pin
   that hardcodes a length is a future flake, and the shipped pins here
   assert the residue and the tail, never a byte count.
6. **A healthy turn is unaffected** — the over-refusal direction. Labelled
   as such: it cannot fail under deletion of the sanitiser:
   `test_a_readable_routing_decision_is_unaffected`.

The executor's report (review round 1) is a separate wire channel with its
own failure modes — shaped by `unittest`/`subprocess`, not by the engine's
wrap — covered by its own pins rather than folded into the six above:

7. **Every wire-channel of the executor's report, closed by property rather
   than by producer** (rounds 1-2): `test_the_executor_report_names_no_path_or_user`
   (a message naming a path, a `SyntaxError` repr, a runner-crash
   traceback), `test_a_failing_test_names_no_path_or_user` (round 2's most
   reachable channel — a produced module shelling out to `sys.executable`),
   `test_a_unittest_failure_names_no_path_or_user` (`OSError.__str__`),
   `test_a_tests_module_that_raises_names_no_path_or_user` (the tests-load
   path, unpinned until round 2), `test_no_report_can_name_a_path` (the
   hostile corpus asserting the OUTPUT SHAPE, not any one input).
8. **The trade the sanitising makes, and its edges** (rounds 2-4):
   `test_a_syntax_error_keeps_its_line_and_column`,
   `test_the_executor_report_still_says_what_happened` (direction guard),
   `test_a_path_free_failure_message_survives`,
   `test_a_relative_path_in_the_source_echo_keeps_the_evidence` (round 4:
   checking only the composed string discarded the class and message
   whenever the source echo named a relative path).
9. **The `TestCase` branch's traceback-parsing history, closed structurally
   in round 8 (#178)** — one pin per wrong-exception defect review found,
   plus round 8's own three: `test_a_unittest_diff_over_relative_paths_keeps_the_evidence`
   (round 5), `test_a_decoy_message_line_does_not_displace_the_exception_class`
   (round 6), `test_a_chained_exception_reports_the_one_that_failed` (round
   7), `test_a_multiline_string_diff_keeps_the_evidence` and
   `test_a_twelve_item_list_diff_keeps_the_evidence` (round 8 / F1),
   `test_a_forged_frame_in_the_message_does_not_displace_the_real_class`
   (round 8 / F5), `test_a_subtest_failure_is_not_silently_dropped` (round
   8, found implementing rather than reviewing — see "Known bounds"). See
   "Known bounds" below for what changed structurally.

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
- **The `TestCase` branch parsed a formatted traceback, and that was wrong
  three ways before round 8 deleted it** — the last line loses the type on
  any multi-line message, a backward `identifier:` scan picks a message
  line, a forward scan reports a chained exception's CAUSE. Round 5's fix
  (keep the last candidate per frame block) was "verified across nine
  shapes," and that sentence was itself the gap: none of the nine put a
  two-space-indented UNCHANGED line into the trace, which is exactly what
  `difflib.ndiff` emits for every line an `assertEqual` diff's two sides
  share. A multi-line-string or 12-item-list failure produces one, the scan
  reads it as a frame and resets on it, and the report degrades to
  `test_x: failed` — the evidence round 5 itself was paid to keep. That is
  **F1**, the confirmation review's blocking finding, round 8. A crafted
  message with an indented line followed by a column-0 `Type: message`-shaped
  line displaced the real class the same way — **F5**.

  **Round 8 took #178 in this arc** rather than filing a fourth instance fix
  (loop-protocol rule 18): `unittest.TestResult.addFailure`/`addError` hand
  the live `(type, value, traceback)` triple before anything is formatted to
  text; a `_LiveResult` subclass keeps `value`; the `TestCase` branch now
  calls `_safe_reason` on it directly, the same call the sibling `test_*`
  branch already made — which is why that branch never had any of the three
  defects above.

  Found while implementing, not by review: `addSubTest`'s DEFAULT
  implementation does not call `addFailure`/`addError` at all — it appends
  straight to `self.failures`/`self.errors`, bypassing both. A `_LiveResult`
  overriding only the two would drop a produced test's `with
  self.subTest():` failure entirely — not degraded to `failed`, gone, with
  `tests_pass` coming back `True`. Worse than any leak this arc closed,
  since those degraded evidence and this flips the verdict. `_LiveResult`
  overrides `addSubTest` too:
  `test_a_subtest_failure_is_not_silently_dropped`.

  No traceback text is parsed in the runner any more:
  `grep -n "extract_tb\|_exception_line" .llm-orc/scripts/agentic_serving/accept_executor_runner.py`
  finds one `extract_tb` call, in `_failing_line`, which locates a SOURCE
  line by filename and line number rather than reading exception text, and
  no `_exception_line` at all — the function is deleted, not bypassed.
- The property is enforced on what is EMITTED, at every emission point, and
  on each PIECE as it enters the string, in BOTH failure branches. Checking
  only the composed string discarded the class name and the message whenever
  the text named a RELATIVE path, reducing an ordinary failing test to
  "failed" — the evidence loss round 2 paid to avoid, and it feeds the retry
  prompt too. Rounds 3 to 5: the discipline reached `_run_test_fns` in round
  4 and the `unittest.TestCase` branch only in round 5, and this sentence
  claimed it as universal in between, which is how the second half went
  unnoticed. An `assertEqual` over a sequence containing a path is enough to
  trigger it, since the diff is multi-line and its last line carries no type.
- `_wire_safe` is also total in its return: a caller cannot
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

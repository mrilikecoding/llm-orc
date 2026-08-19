# #166 — an empty deliverable is never a client write (design)

Status: pre-flight complete, re-grounded 2026-08-18. Issue: #166, split out
of #155's pre-flight review.

The first draft of this document named the wrong fault. Pre-flight falsified
its reproduction on all nine cells and then found a fault that is worse: not
defence in depth, but single-fault reachable on two build routes, and on one
of them destructive. What follows is the corrected record; the original is in
git.

## What is measured

**The dead-seat reproduction is FALSIFIED.** The first draft claimed that an
absent seat, an empty seat, and `loop_unwrap`'s `{}` degrade each reach the
client as an empty write. Run through the real
`seat_contract.py -> shape.py -> form_gate.py -> emit.py` chain — with the
seat_contract node actually executing rather than a verdict hand-fed into
shape — all three refuse honestly, on all three build routes:

```
build-gated / write-tests / re-fix, seat ABSENT | EMPTY | {}
  seat_contract: {"seat_admitted": false, "seat_contract_reason":
                  "Assertion 'envelope-carries-artifact' raised exception: ..."}
  EMIT         : {"finish": true, "content": "Seat contract not met: ..."}
```

Nine of nine. The draft's reproduction almost certainly fed
`seat_admitted: true` into shape, which is not what a healthy seat_contract
emits for a dead seat.

**The reachability question the draft left open answers NO.** The only
`build=True` steps in `chain_plan.py` are `re-fix`, `tests-seat` and
`code-seat`; with `resolve.py`'s `_DERIVED` and the catalog's `serves:`
declarations that is exactly three shapes — `build-gated`, `write-tests`,
`re-fix` — and all three declare a seat contract asserting
`len(results['seat']['artifacts']) > 0`. (`gen-review` is unreachable: no
step and no `_DERIVED` entry produces the `review` intent.)

**The reachable fault here is a LIVE seat with an EMPTY artifact.** The seat contract
asserts artifact PRESENCE, never non-emptiness; `ast.parse("")` succeeds so
the form gate passes honestly; and `serving_ensemble_caller.py` maps any
outcome carrying `file` and `content` to a client Write. All three envelope
producers mint `artifacts[0]` unconditionally from a possibly-empty
extraction — their `summary` fallbacks (`"code deliverable"`,
`"test deliverable"`, guarded on `not code.strip()`) show the authors already
knew empty content arrives there.

`diagnostics.accept` is the only thing between that and the client. It is
defeated on two routes — and note before the detail that the mechanism which
defeats it is **indifferent to emptiness**: the same two routes ship `x = 1`
or `# TODO` just as readily, which is #171 and not this issue. What follows
is the empty slice.

**`re-fix` — single fault, deterministic, and DESTRUCTIVE.**
`refix_select.py` substitutes `_SMOKE_TEST` whenever rung 1.5 found no
visible test:

```
def test_refix_candidate_loads_cleanly():
    pass
```

A `pass` body passes against any code, including none. Measured through the
real executor:

```
code='' tests=<smoke>  ->  tests_pass=True  n_tests=1  report='all passed'
```

so `accept` is true, the envelope ships `artifacts[0].content = ""`, and emit
returns `{"finish": false, "file": "calc.py", "content": ""}`. One faulty node
output, no model judgment anywhere in the gate, and the file is one the client
already has — this is a clobber, not an empty new file.

**`build-gated` — single fault whenever the target file is not the tested
module.** The executor's ground truth can be satisfied by the materialized
workspace rather than by the deliverable. `_materialize` writes the
conversation's workspace files into the sandbox and only shadows the target
file, and only when the requirement names it:

```
target_file=''             tests_pass=True   -> EMPTY CLIENT WRITE
target_file='helpers.py'   tests_pass=True   -> EMPTY CLIENT WRITE
target_file='inventory.py' tests_pass=False  -> refused
```

`adequacy_check.py` reads the tests alone and never sees the code, so
`tests_adequate` is structurally independent of the deliverable being empty.
`tests_pass` is the only barrier and the workspace satisfies it. The held
round is strictly weaker, not stronger: `_resolve_adequacy` carries
`tests_adequate=True` off `gather.held` with no judge at all.

**`write-tests` — closed.** The shipped deliverable is the executor's echoed
tests, and an empty tests file yields `n_tests == 0`, which
`accept_executor_runner.run_tests` reports as `tests_pass=False`. Five
constructions tried (empty fence, empty response, whitespace fence,
bare-name asserts, comment-only); all refused. No path found.

**Production evidence, corrected.** The draft attributed this to
`.llm-orc/.serve-trace/turns.jsonl` line 469. That attribution is wrong:
replayed through current main, line 469 refuses via #152's routing guard, and
its classify emitted `target: "need-glob"` — it was never a build ask. It is a
#152/#154-class incident, already fixed twice.

The real evidence is four other turns. Lines **38, 81, 97, 98** are live build
turns with `seat_admitted: true`, a well-formed envelope, `content: ""`, and
`accept: false` as the only thing between them and a client write.

**The write census, corrected.** 663 turns; **52** emit outcomes carry both
`file` and `content` (the caller's write branch), not the draft's 136.
Exactly one has empty content — line 469, the defect. No whitespace-only
writes. The qualitative conclusion holds and the denominator does not: an
empty write has never been legitimate here in 52 chances.

## Change

Refuse a build outcome whose deliverable is empty, at the CALLER.

`serving_ensemble_caller.py` is the last line of defence and the only seam a
project's own marshal scripts cannot bypass — `serving.yaml` and those scripts
are per-project config that revs independently of the installed caller. That
is the reasoning the existing version-skew guard records three lines below
where the empty write happens.

"Empty" means empty after `.strip()`, so whitespace-only is covered.
Comment-only content is NOT covered: it is a legitimate file, and telling
"only comments" from "a real file" needs a parser rather than a predicate.

**The refusal must mint a ledger entry.** This is the one thing the caller
placement does not get for free. `_load_emit_reject_prefixes` keeps only
terminals with a non-empty `mints`, and every existing caller-minted refusal
in `_outcome_chunks` uses `"Refused: "`, whose `mints` is `""`. Measured:

```
'Refused: ...'        -> _reject_kind ('', '')        ledger []
'Build refused: ...'  -> _reject_kind ('refused', …)  ledger [{'outcome': 'refused', …}]
```

Follow the existing idiom and a refused build ask records no outcome at all,
indistinguishable from a question — which breaks #133/#134 recap grounding on
exactly the ask a user is most likely to follow up on. So `reject_prefixes`
(already in scope at the `_outcome_chunks` call site) is threaded in and the
prefix is selected by the vocabulary the module already names,
`mints == _REFUSED`. Hardcoding `"Build refused: "` is not acceptable: it
would contradict the version-skew argument the placement rests on, since the
project's `emit.py` owns that literal.

**Not in `form_gate` as well.** Two guards for one rule is how invariants
drift, and the form gate's rule is about parsing, which an empty file passes
honestly.

**Why not the seat contract**, which is where the fault originates and which
already asserts `len(artifacts) > 0`: extending it to non-emptiness is a
three-line YAML change, refuses one node earlier, and mints
`rejected_contract` correctly with no new vocabulary. It loses on one point
only, and it is the deciding one — the contracts are per-project config, so a
project whose scripts drift keeps the defect. The caller is the seam that
holds regardless of what a project's scripts say. Recorded because it is the
strongest alternative, not because it is wrong.

**No double-refusal with #155 Arc A.** `emit.main`'s `elif` chain puts
`seat_admitted is False`, `accept is False` and `seat_gate_failed` all ahead
of the ship branch, and `_outcome_chunks` tests `outcome.get("finish")` before
the write branch. The caller guard only ever sees outcomes that already
cleared every emit gate.

## Invariant

A build outcome with an empty deliverable never reaches the client as a write;
it refuses honestly instead, and the refusal is recorded in the ask-outcome
ledger.

## Regression instruments

Twelve tests in `tests/unit/serving/test_serving_empty_deliverable.py`, each
labelled by what it can actually catch. The recurring defect in this corpus
is a pin that cannot fail (#156 round 1, #160 round 2, #155's four rounds),
so a pin guarding the over-refusal direction is named as such rather than
passed off as an invariant pin.

**Invariant pins — each goes RED under deletion of the guard it names:**

1. **The `re-fix` capture.** Smoke-only test, empty candidate — the real
   `refix_gather -> refix_select -> accept_executor -> refix_envelope` chain
   builds the envelope, and the real
   `seat_contract -> shape -> form_gate -> emit` chain carries it to the
   outcome, which then goes through `_outcome_chunks`. This is the
   destructive case: the named file is one the client already has.
   It drives the real NODES, not `ServingEnsembleCaller` — the production
   call site is pinned separately by instrument 4, which is the distinction
   round 1 blocked on.
2. **The `build-gated` capture.** Workspace-satisfied tests with the target
   naming a different file, empty deliverable, `accept: true`. A distinct
   route and a distinct mechanism from 1.
3. **Whitespace-only content refuses**, which kills a `== ""` implementation.
4. **The serve path mints a `refused` ledger entry.** Drives
   `ServingEnsembleCaller.run()` over a project carrying the real `emit.py`,
   with `_serve` returning an empty deliverable, and asserts `_reject_kind`
   mints `refused`. Round 1's BLOCKER: the ledger half of the invariant rode
   on one argument at the production call site and dropping it left the whole
   suite green, because every other pin hands the prefixes to
   `_outcome_chunks` itself and so cannot see the wiring.
5. **The refusal mints through `_reject_kind`** against the project's real
   `emit.py` — separate from
   `test_every_build_reachable_emit_terminal_mints_a_ledger_entry`, which
   iterates emit's own `TERMINALS` and is structurally blind to a
   caller-side terminal.
6. **The prefix is not hardcoded**: a project whose refused terminal carries
   different wording gets that wording.
7. **A null deliverable is not coerced into a write.** Judging
   `str(content)` turned `content: null` into the four non-blank characters
   `None` and wrote them. Covers `None`, a dict, and `0`.
8. **Non-`.py` deliverables are guarded too.** `form_gate` returns "ok"
   unconditionally for anything that is not `.py` or `.json`, so for a
   `.md`, `.sh` or `.yaml` deliverable this is the only guard there is.
9. **The refusal is a stated refusal, not an empty stream.** The whitespace
   pin asserted only the ABSENCE of a write, so returning `[]` — or dropping
   the stream terminator — passed it.
10. **A malformed deliverable is not called empty** (review round 2): a
    non-str deliverable gets its own wording, because a refusal that
    misdescribes its cause is what this corpus refuses to ship.

**Over-refusal pins — these CANNOT fail under deletion of the guard. They
exist so the fix does not become "refuse every build":**

11. A healthy build still writes.
12. A one-character deliverable still writes, pinning that the rule is
    emptiness and not a length heuristic.

**Also pinned, without an invariant of its own:** the no-readable-`emit.py`
fallback still refuses (the version-skew scenario the placement is argued
from), and its prefix is the plain non-minting idiom rather than a hardcoded
copy of emit's wording.

## Known bounds

- Does not make a dead seat, or a dead code_writer, produce anything useful.
  It converts a silent bad write into an honest refusal.
- Comment-only and docstring-only deliverables are not covered, by choice.
- The `loop_unwrap` `{}` degrade is NOT a bound of this change: measured
  above, `{}` is refused by the seat contract on all three build routes,
  because it is not an ADR-024 envelope. The draft recorded it as an
  uncovered case; that was part of the same falsified reproduction.
- `shape._envelope_deliverable` prefers an empty `artifacts[0].content` over
  a non-empty `primary` (`isinstance(content, str)` admits `""`). Not
  currently producible — all three producers set both from one value — but a
  latent second source of the same fault if a seat ever diverges.
- The `re-fix` smoke test cannot detect an empty candidate at its source;
  this guard papers over it. Filed as #169, with the clobber framing.
- **The fault is not really about emptiness, and this closes only that
  slice.** Measured on both routes with the same producer chains the
  captures use:

  ```
  re-fix (calc.py exists client-side)   build-gated (target != tested module)
    empty        -> refused               empty        -> refused
    '# nothing'  -> CLIENT WRITE          '# TODO'     -> CLIENT WRITE
    'x = 1'      -> CLIENT WRITE          'x = 1'      -> CLIENT WRITE
  ```

  The accept gate's ground truth never touches the deliverable, so junk
  clobbers exactly as empty did. Filed as **#171**. Deliberately NOT widened
  here: "refuse a deliverable that looks unhelpful" is a quality judgment,
  and doctrine 9 puts those behind a gate rather than a predicate.
- A legitimately empty file is now refused. `create an empty __init__.py` is
  a routable build ask (`_FILE_RE` matches, `_BUILD_RE` matches), and
  `__init__.py`, `py.typed` and `.gitkeep` all refuse. The census justifies
  the trade — 52 writes, one empty, and that one the defect — but it is a
  real cost, not a free one.
- The caller infers build-ness from the outcome SHAPE (`file` + `content`
  both present) rather than from `is_build_ask`, which it never sees, and
  then stamps the build-scoped prefix. True for today's emit, where only the
  build-and-valid terminal emits both, and any outcome carrying both is a
  build-shaped action regardless. It is the version-skew exposure of the
  placement the design is argued from, so it is named rather than assumed.

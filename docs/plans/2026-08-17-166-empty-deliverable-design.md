# #166 — an empty deliverable is never a client write (design)

Status: pre-flight. Issue: #166, split out of #155's pre-flight review.

## What is measured

**The defect reproduces on current main**, after #155 Arc A, through the
real `shape.py -> form_gate.py -> emit.py` chain on a build route with a
HEALTHY `seat_contract`:

```
seat dep ABSENT          -> {"finish": false, "file": "solution.py", "content": ""}
seat EMPTY string        -> {"finish": false, "file": "solution.py", "content": ""}
loop_unwrap degrade {}   -> {"finish": false, "file": "solution.py", "content": "{}"}
```

`serving_ensemble_caller.py:1453-1465` maps any outcome carrying `file`
and `content` straight to a client Write, so all three become writes.
`ast.parse` accepts `""`, `"   "`, `"{}"` and comment-only content, so
the form gate structurally cannot catch them — its rule is "parses as
what the path claims", and an empty file does parse.

**It has happened in production.** `.llm-orc/.serve-trace/turns.jsonl`
line 469, both faults at once:

```
seat           failed   null
seat_contract  success  {"success": false, ... "Schema JSON execution failed: ..."}
shape          success  {"build": true, "file": "solution.py", "content": "", ...}
emit           success  {"finish": false, "file": "solution.py", "content": ""}
```

The seat died as a dispatch node (`status="failed"`, filtered out of
deps entirely), and `seat_contract` died too. Arc A now refuses THAT
turn, since a dead seat contract on a build turn that would otherwise
ship refuses. The single-fault case — dead seat, healthy contract — is
the one reproduced above.

**No legitimate empty write exists in the corpus.** Across the same
trace: 136 turns produced a file write; exactly ONE had empty content,
and it is line 469, the defect itself. That answers the question this
design was going to leave open — whether a user might legitimately ask
for an empty `__init__.py`. Nothing in 136 live writes did, so the
strong rule is supported by measurement rather than assumed.

## The reachability question pre-flight must settle

I could not derive this confidently and am not going to assume it.

`_DERIVED` in `resolve.py` marks two build targets, `code-seat` and
`tests-seat`. Seat contracts are declared by four ensembles:
`build-gated`, `code-seat`, `re-fix`, `write-tests`. Every build seat
with a contract declares `len(results['seat']['artifacts']) > 0`, which
RAISES on a dead seat, so `seat_contract` refuses it before shape's
degrade matters.

So: **is there a reachable build route whose seat has no contract
block?** If yes, this is single-fault reachable today. If no, the dead
seat is always caught one node earlier and this becomes defence in
depth — still worth having, since the caller is the seam a project's own
scripts cannot bypass, but the issue's severity and the instruments
change.

Two candidates to check: `tests-seat`, which has no `tests-seat.yaml`
and must resolve through the operator-curated mapping (WP-C8); and any
curated shape that produces `build: true` without a contract.

Separately: `loop_unwrap`'s `{}` degrade produces NON-empty junk, so it
survives an emptiness rule entirely. Pre-flight should say whether that
belongs here or is its own issue.

## Change

Refuse a build outcome whose deliverable is empty, at the CALLER.

`serving_ensemble_caller.py` is the last line of defence and the only
seam a project's own marshal scripts cannot bypass — `serving.yaml` and
those scripts are per-project config that revs independently of the
installed caller. That is the reasoning the existing version-skew guard
records three lines below where the empty write happens, and it applies
unchanged here. That guard already refuses outcome VOCABULARY the caller
does not recognise; this extends it to a recognised outcome that is
empty.

Deliberately NOT in `form_gate` as well. Two guards for one rule is how
invariants drift, and the form gate's rule is about parsing, which an
empty file passes honestly.

"Empty" means empty after `.strip()`, so whitespace-only is covered.
Comment-only content is NOT covered: it is a legitimate file, and
distinguishing "only comments" from "a real file" needs a parser rather
than a predicate.

## Invariant

A build outcome with an empty deliverable never reaches the client as a
write; it refuses honestly instead.

## Regression instruments

1. **A dead seat produces a refusal, not an empty write.** Red today —
   the issue's reproduction. Node-level.
2. **End to end through the real caller**, asserting no `tool_calls` on
   the response. Per #155's lesson, a node-level pin does not prove the
   chain; this one has to go through `ServingEnsembleCaller`.
3. **Whitespace-only content refuses too.**
4. **A healthy build still writes** — the pin that stops this becoming
   "refuse every build".
5. **A one-character deliverable still writes**, pinning that the rule
   is emptiness and not a length heuristic.
6. **The refusal names the file**, so the client can tell what did not
   get written.

## Known bounds

- Does not make a dead seat produce anything useful. It converts a
  silent bad write into an honest refusal.
- `loop_unwrap`'s `{}` degrade is not covered: non-empty junk survives
  an emptiness rule. Recorded rather than fixed.
- Comment-only and docstring-only deliverables are not covered, by
  choice.
- Overlaps #155 Arc A on the dead-`seat_contract` path, where Arc A
  refuses first. This must not double-refuse or shadow that message.

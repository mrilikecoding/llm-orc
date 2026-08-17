# #155 — positive completeness downstream of routing (design)

Status: pre-flight, round 2. Issue: #155, found during the #152 merge
review. Split into three arcs; only Arc A is ready to build.

## What round 1 of this design got wrong

Recorded first, because the corrections are the design.

**The seat is a `dispatch:` node, not a script agent**
(`serving.yaml:117-118`). When a dispatch node dies, the dispatcher
returns `status="failed"` with `response=None`, and
`_extract_successful_dependency_results` filters dependencies to
`status == "success"` — so the seat key is **absent** from shape's
dependencies. There is no failure envelope to recognise. Round 1's
predicate was aimed at the one death mode that produces JSON and missed
the two that produce nothing.

**There is no empty-read guard.** Round 1's Known Bounds said "a seat
that crashes and returns EMPTY is already caught by the empty-read
guard". That guard does not exist; I asserted it without checking.
`ast.parse("")` succeeds, so form_gate stamps `valid: true`, and
`serving_ensemble_caller.py:1453-1465` maps any outcome carrying `file`
and `content` straight to a client Write. **A dead seat writes an empty
`solution.py` on the client.** The comment immediately below that branch
records a previous fix for exactly this shape — a "junk empty
solution.py write" — but only for outcome vocabulary the caller does not
recognise, not for a recognised outcome that is empty.

**Round 1's reproduction omitted a node that is in the live wiring.**
`seat_contract` sits between `seat` and `shape`
(`serving.yaml:125-131`), and every build seat in the corpus declares
`len(results['seat']['artifacts']) > 0`, which raises on all three dead
seat shapes. So on a build turn a dead seat is ALREADY refused with
`Seat contract not met: ...`. Gap 2 as demonstrated needs **two**
faults: a dead seat and an unreadable `seat_contract`.

**The engine envelope quoted was the wrong one.** Serving nodes always
take the `ScriptAgentInput` path, so they route to
`execute_with_schema_json`, whose blanket `except Exception` produces a
FOUR-key wrap:

```
{"success": false, "data": null,
 "error": "Schema JSON execution failed: Command '['/Users/.../python3',
           '/Users/.../classify.py']' returned non-zero exit status 1.",
 "agent_requests": []}
```

not the two-key `{"success": false, "error": "Script timed out after 60
seconds"}` from `ScriptAgent.execute()`, which serving never takes. Any
fixture built on the quoted shape would test something the engine does
not emit here. It also means the path leak is **universal, not
occasional**: every engine-wrapped node failure puts the interpreter
path and the script path — hence the username and home directory — on
the wire.

## The real mechanism

One line, written in several places:

```python
try:
    gated = json.loads(_response(deps.get("form_gate", {})))
except json.JSONDecodeError:
    gated = {}          # "I could not read my input" -> "nothing to do"
```

`{}` answers every subsequent question plausibly: `build=False`,
`content=""`, `valid=True`, no refusal reason — and emit prints
`{"finish": true, "content": ""}`. The failure is not unhandled; it is
converted into a well-formed success.

The same family, one level out, is `shape._seat_verdict`: a crashed
`seat_contract` parses as JSON, carries no `seat_admitted`, and so
returns `(None, "")` — which emit reads as "no per-seat gate ran"
rather than "the gate died".

## Arc A — pipeline integrity (READY)

Three seams where a node cannot read its own input. No content
judgement, no interaction with the delegation seams beyond being first.
This is where the single-fault silent-empty-success actually lives.

1. **`form_gate` recognises `shape`.** A shape output always carries
   `build` and `content` (verified: one `print(json.dumps(...))`, no
   early return). Anything else sets `node_failed` and threads it.
2. **`emit` recognises `form_gate`.** A form_gate output always carries
   `valid` (same verification). Checked BEFORE any field is read off
   `gated`, since the whole problem is that `{}` answers everything.
3. **`shape` recognises `seat_contract`.** Reported as its own signal,
   `seat_gate_failed`, NOT as a pipeline read failure — see placement
   below. An ABSENT dep fails closed too, which two drafts justified
   wrongly before review: the first confused the ensemble's optional
   `seat_contract:` YAML block with the skeleton's unconditional
   `seat_contract` NODE; the second claimed absence means "filtered for
   not succeeding", which is false, since `when:`-skipped nodes are
   routinely absent and a crashed script agent is always PRESENT with an
   error envelope. Measured at zero absences across 650 live turns, so
   the branch is a deliberate trip-wire for a future skeleton change
   rather than handling for a live failure mode.

The engine wrap's keys (`success`, `data`, `error`, `agent_requests`)
are disjoint from both healthy key sets, so positive recognition
discriminates cleanly without a denylist.

**Placement, which took three rounds to get right.** Two signals, not
one, because the axis that matters is not how a node failed but whether
its death bears on this route:

- `node_failed` (an unreadable `shape` or `form_gate`) refuses FIRST in
  `_seam_outcome`, non-minting. The #152 premise genuinely holds: the
  routing decision, every delegation request and the deliverable all
  came from an unreadable source, so `is_build_ask` is unknowable.
- `seat_gate_failed` refuses on the BUILD branch, after the delegation
  seams AND after the accept gate, with the MINTING prefix. Round 1 put
  it first and killed eight routes the seat contract cannot affect.
  Round 2 moved it to the build branch but ahead of the accept gate,
  which discarded a real verdict ("tests do not pass", carrying a retry
  invitation) and converted a `rejected_gate` entry into a `refused`
  one. It now fires only on a turn that would otherwise SHIP, which is
  the wrong-accept it exists to prevent and nothing else.

The minting prefix is right because routing succeeded by construction to
reach the build branch, so `is_build_ask` is known. An earlier draft
justified it as preserving a `rejected_contract` entry the system
already earned; that was lifted from the Arc B bullet below, which is
about a dead `seat` DISPATCH node — a different fault. Measured: before
this arc a dead `seat_contract` SHIPPED, minting `shipped`, so the
change converts a wrong-accept into a refusal.

### Arc A instruments

1. A crashed `shape` refuses. Red today (`{"finish": true, "content": ""}`).
2. A crashed `form_gate` refuses. Red today.
3. A crashed `seat_contract` refuses rather than reading as "no gate".
4. **A healthy build still ships**, and
5. **a healthy prose turn still finishes** — the two that stop the
   change degrading into "refuse everything".
6. **Every delegation round still delegates**: reads, self-reads, glob,
   grep, run, not-grounded, recall. Round 1 had no pin here and review
   showed why it matters (see Arc B).
7. An ABSENT `seat_contract` also fails closed (see above for why that
   branch is a trip-wire rather than live handling).
9. **A dead seat gate never refuses a NON-build turn** — the `build`
   guard, which round 2 shipped unpinned. Deleting it left all 3977
   tests green while converting a healthy prose turn into a MINTING
   `Build refused:`, i.e. a ledger entry on a turn carrying no build
   ask.
10. **The accept gate outranks a dead seat gate**, and a seat-contract
    rejection outranks it too. The decision this doc demanded and round
    2 answered silently.
11. **End to end through the REAL ensemble**, via `_crashed_script_client`
    with `shape.py`, `form_gate.py` and `seat_contract.py` genuinely
    crashed. Both of round 2's blockers existed because every pin fed a
    node directly and nothing ran the chain with a fault injected.
8. Two existing fixtures in `test_serving_emit.py` are hand-built
   partial dicts without `valid` and will newly refuse:
   `test_seat_contract_rejection_uses_the_exported_prefix` and
   `test_recall_answer_field_emits_the_honest_message`. Updating them is
   correct. **The trap to avoid**: weakening the check to "missing
   `valid` AND a `success`/`error` key present" to make them pass, which
   reinstates the denylist this design exists to remove.

## Arc B — the dead seat (NOT READY, needs redesign)

Round 1 aimed a predicate at the wrong thing. What Arc B has to handle:

- **The absent dep** (dead dispatch node) — the single-fault reachable
  mode, which produces `{"finish": false, "file": "solution.py",
  "content": ""}` and a client-side empty write.
- **`loop_unwrap`'s degrade**, which prints `{}` on an unreadable loop
  wrapper and ships a `solution.py` containing `{}`.
- **The envelope mode**, which needs two faults on a build turn and is
  therefore the least urgent of the three.

Three constraints round 1 violated:

- **Placement.** `seat_failed` must sit on the BUILD branch, after the
  delegation seams — not first alongside `routing_failed`. The seat is a
  zero-cost echo on delegation routes; the outcome rides the routing
  decision. Review demonstrated that an early check turns four working
  behaviours (`reads`, `glob`, `grep`, `recall`) into refusals.
- **Prefix.** A dead seat on a build turn currently mints
  `rejected_contract` via `seat_contract`. Routing succeeded by
  construction, so `is_build_ask` is KNOWN, and the non-minting prefix
  would cost a ledger entry the system currently earns. It needs
  `BUILD_REFUSED_PREFIX`, and the design must say which gate wins when
  both fire.
- **False refusals.** An explain turn whose answer IS an error envelope
  ships today and would be refused. Not hypothetical here: the corpus is
  agents that emit these envelopes and dogfooding asks about them. The
  seat check has to be scoped to routes where the seat is expected to
  deliver an envelope, not applied to the raw-prose degrade path.

Filed as its own issue rather than carried here.

## Arc C — reason text and decision residuals

- **Sanitize the engine wrap's error** in all four reason builders, not
  just #152's. Cut the whole `Command '[...]'` clause; the useful
  residue is the tail (`returned non-zero exit status 1`, `timed out
  after N seconds`). `turn_trace.py` keeps raw responses server-side, so
  nothing debuggable is lost.
- **Zero-width target**: `"​".strip()` is truthy, so it passes
  shape's non-empty check. Reproduced.
- **`kind: null`**: passes the presence check, and
  `decision.get("build", decision.get("kind") != "explanation")`
  evaluates `None != "explanation"` to True. Reproduced.

## Invariant (Arc A)

A node that cannot read its upstream input refuses, and never produces a
finish that is indistinguishable from success.

The bound: this is about UNREADABLE input only. A node that produces
well-formed but wrong output is the accept gate's problem, and a node
that produces nothing at all is Arc B's.

## Known bounds

- shape/form_gate/emit are deliberately stdlib-only so they still run
  when the `llm_orc` import is broken — which is #154's failure mode,
  the one that produced #152. That, not a boundary rule, is why a
  predicate is mirrored rather than imported. (`seat_contract.py`
  imports `llm_orc` directly, so no such rule is operating.)
- Stdout pollution in these nodes becomes a refusal where it used to be
  a silent degrade. That is the better direction, and these three are
  stdlib-only, so the exposure is small.

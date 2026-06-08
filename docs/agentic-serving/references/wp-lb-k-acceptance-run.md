# WP-LB-K Acceptance Run — session convergence (ADR-037 Conditional Acceptance)

**Scope:** the staged real-OpenCode gate that discharges ADR-037's Conditional
Acceptance. It is the delta over the WP-LB-C/I single-write smoke test
([`opencode-smoke-test.md`](./opencode-smoke-test.md)) — same stack and setup,
but it tests **convergence**: does a real cheap-driver session *finish clean*
once the work is done (the Finding F refutation), not just whether one write
delegates.

**Why this run and not the existing smoke test:** the WP-LB-I run met ADR-036's
delegation gate and then *never terminated* — every no-new-task tool-result tail
delegated another phantom revision (Finding F). WP-LB-K's two-call composition
fixes that. The smoke test confirms one delegated write; this run confirms the
session converges.

**Prereqs + setup:** identical to `opencode-smoke-test.md` (local Ollama with a
tool-calling model — Spike θ used `qwen3:14b` for the judgment seat;
`llm-orc serve` on `127.0.0.1:8765`; OpenCode pointed at `/v1`). The judgment
seat defaults to the seat-filler's profile (FC-68), so no extra config — the one
profile drives both calls.

**$0:** local inference only. Confirm before any paid step (the free-options
preference). Hosted judgment seat is an LB-8 optimization, not part of this gate.

---

## Run 1 — gating (multi-file, convergence)

Issue one natural-phrasing task whose deliverable count is legible in the text,
e.g.:

> Write a Python module `string_utils.py` with a function that reverses the word
> order of a string, and a `test_string_utils.py` with unit tests for it.

**One session yields both Conditional Acceptance gating conditions.**

### Condition (b) — work-remaining tails keep delegating

For each file before the last (n-1 trailing tails), the serve log must show:

```
grep -E "turn decision: .*judgment_verdict=REMAINING" serve.log   # the judge said keep going
grep -E "dispatch start: ensemble="                    serve.log   # delegation fired for that file
```

- `judgment_verdict=REMAINING` on a `tail_kind=trailing_tool_result` line, **and**
- a `dispatch start: ensemble=…` for that turn (delegation fired — not an inline
  `write`, the Finding B shape; not a premature finish, the E4a shape).

### Condition (a) — the completed session converges

On the final trailing tail, after all requested files are written:

```
grep -E "turn decision: .*judgment_verdict=COMPLETE" serve.log
```

- `judgment_verdict=COMPLETE` on a `tail_kind=trailing_tool_result` line,
- the client receives a **text-only assistant turn** (no tool calls, no `VERDICT:`
  literal), and
- **OpenCode's loop ends** — no further request. This is the Finding F refutation:
  the session does not delegate a phantom revision.

### Production digest join (FC-64) — not a constructed digest

The judgment's evidence must be the framework's own records joined with client
results, not a reconstruction from client-serialized messages. Assert from the
captured request bytes that the judgment call's digest carries
**framework-recorded file paths** (e.g. `string_utils.py`) joined with the
client's tool results (`Wrote file successfully`). A digest showing bare
`"Wrote file successfully"` strings with no paths is the round-1 failure mode and
fails the gate. (Capture the seat request bytes via a serve-side dump or proxy,
the ψ-capture discipline.)

### Out of the gate (recorded, not guessed across)

- `new_user_task` tails (mid-session intent refinement) — ADR-036's merge branch,
  untouched by ADR-037; covered by unit tests, not this run.
- Non-write-shaped deliverables — the recorded ADR-037 boundary; watched by the
  FC-67 shares.
- Messy real-session shapes — PLAY territory.

---

## Run 2 — incidental discharge (read-then-write)

Run 1's shape plus a leading read step:

> Read `string_utils.py`, then write `number_utils.py` consistent with its style.

This exercises the judgment over a **mixed record** (a read is context, not a
deliverable) and discharges FC-61's outstanding real-run carry-side assertion (a
read/command turn that carries verbatim, not delegates).

---

## After Run 1 passes — the progressive ladder

Practitioner-directed, crawl-before-walk: design escalating runs (more
deliverables, mixed read/write, repair-shaped flows, multi-part asks) to find the
digest's expressiveness limit in controlled runs rather than waiting for
production false-stops. Ladder design is informed by Run 1's evidence — do not
pre-specify past it. This is the proactive complement to FC-67's trailing
false-stop trigger and the practitioner's gate pre-mortem ("failure looks like
not being able to track everything we need to track in the meta-context").

---

## Recording the result

Capture the serve log + the OpenCode session trace under a scratch directory
(e.g. `scratch/wp-lb-k-acceptance/`) and note the verdict per condition. A passing
Run 1 + Run 2 discharges ADR-037's Conditional Acceptance — update the ADR status
from Accepted (Conditional) to Accepted, and the cycle-status Conditional
Acceptance gating row. A failing run names which FC regressed (judgment-first
FC-63, finish cleanliness FC-65, call-2 preservation FC-66, digest provenance
FC-64) and feeds back to the Loop Driver or Session Action Record.

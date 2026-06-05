# Gate Reflection: Cycle 7 loop-back #5 architect → build

**Date:** 2026-06-05
**Phase boundary:** architect (loop-back #5 — ADR-037 allocation) → build
**Cycle:** Cycle 7 — Framework-driven orchestration: routing as code

## Belief-mapping question composed for this gate

The Session Action Record's boundary is the consequential one because of how it
answers the practitioner's DECIDE-gate pre-mortem ("failure looks like not
being able to track everything we need to track in the meta-context"). The
allocation answers with two structural properties — schema-extension locality
and the false-stop share as enrichment trigger — but deliberately defers what
the record must *represent* to evidence, and the trigger is a trailing
indicator: a false stop on a real session means a session already ended with a
confident summary and silently missing work.

The question (a BUILD-scoping decision the practitioner uniquely owns, not
speculation): **which real session shapes should WP-LB-K's acceptance run
include beyond the measured single-write case** — a multi-deliverable ask, a
mid-session intent refinement, a mixed read-then-write flow? Whatever is named
becomes the acceptance gate's composition; anything left out stays in
false-stop-trigger territory until production surfaces it.

## User's response

> "It's way to hardcore to say 'run RDD' -- but I think we need to devised the
> minimally useful test. For example, the single-write case is too small. We
> need to show something iterative that works across multiple files most
> likely. How do you propose we design it based on that?"

The agent proposed the staged two-run design (Run 1 gating: 2–3 file
natural-phrasing ask, n-1 REMAINING + 1 COMPLETE in one session, production
digest join asserted from captured bytes; Run 2 incidental: leading read step
discharging FC-61's outstanding carry-side assertion; explicit out-of-scope
list), built on the vary-one-thing discipline (swap in the production join and
the real client; hold the task shape inside Spike θ's measured envelope).
The practitioner's disposition:

> "Messy real sesssion we will need to test for but it's wise to crawl before
> we walk. If we can do the first version, then I suggest we devised a more
> sophistocated test or series of tests and progressively find our limit."

The progressive task-shape ladder was encoded into WP-LB-K as the post-gate
follow-on (ladder design informed by Run 1's evidence, not pre-specified) —
the proactive complement to FC-67's trailing false-stop trigger. At commitment
gating the agent proposed the settled/open split; the user's response:

> "Looks good"

## Pedagogical move selected

Teach (identify the most consequential boundary — the Session Action Record's
trailing-indicator catch — explain why it matters, then ask for the
practitioner's take via a concrete scoping question). The practitioner's
response converted the question into a design collaboration: constraint
("minimally useful, iterative, multi-file") → agent proposal → practitioner
refinement (crawl-before-walk; progressive limit-finding ladder).

## Commitment gating outputs

**Settled premises (the user is building on these going into build):**
- The v6.3 allocation: Session Action Record (L1) owns the framework-owned
  digest; judgment composition is a Loop Driver extension with named stateless
  helpers; `tail_kind` + `judgment_verdict` settle the TurnDecision event
  shape; WP-LB-J is unheld.
- Session Action Record builds first within WP-LB-K — the production digest
  join is the Conditional Acceptance gating-condition pre-condition.
- The acceptance gate is the staged two-run design: multi-file iterative Run 1
  (gating), read-then-write Run 2 (incidental discharge of FC-61).
- AS-3 turn-cap wiring rides WP-LB-K (FC-69).

**Open questions (the user is holding these open going into build):**
- The digest's expressiveness limit for messy real sessions — now with a named
  proactive instrument (the progressive task-shape ladder after Run 1) in
  addition to the trailing FC-67 false-stop trigger.
- The non-write-shaped deliverable boundary (recorded; watched by FC-67).
- Judgment-seat portability and hosted-seat adoption (LB-8; config-level,
  revisitable on the FC-67 shares).
- Snapshot advisory B: the named-helpers pattern is preparation for a future
  module split if change rates diverge, not a commitment against one.

**Specific commitments carried forward to build:**
- WP-LB-K work order: Session Action Record (V-03) first; then judgment
  composition + branch enforcement (V-01/02/04/05/08); TurnDecision fields
  (V-06, additive); sink surfacing; AS-3 wiring (FC-69).
- Run 1 acceptance evidence is serve-log + captured-bytes (layer-match: a
  passing-looking run can be model-direct; the digest must be production-join
  fed, not constructed).
- Snapshot advisory A honored by the gate design: multi-step tasks are in the
  gate run itself (the 2–3 file shape), not deferred to the trigger.
- Snapshot advisory C: verify the retroactively added responsibility-matrix
  rows against module entries before navigating by them in BUILD.
- The progressive ladder is designed after Run 1 passes, from Run 1's
  evidence — do not pre-specify past it.

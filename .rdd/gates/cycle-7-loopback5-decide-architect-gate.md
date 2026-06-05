# Gate Reflection: Cycle 7 loop-back #5 decide → architect

**Date:** 2026-06-05
**Phase boundary:** decide (loop-back #5 — Finding F termination mechanism) → architect
**Cycle:** Cycle 7 — Framework-driven orchestration: routing as code

## Belief-mapping question composed for this gate

The mechanism's two failure modes have asymmetric user-visible costs, and the
spike measured them very differently. A false-continue (judgment says
REMAINING on finished work) costs one wasted revision turn and self-corrects
at ~0.9 per cycle — measured at 1/10 on the real-capture base. A false-stop
(judgment says COMPLETE with work outstanding) ends the session with a
confident summary and a silently missing deliverable — measured at 0/30, but
on two constructed work-remaining bases, both simple multi-file write tasks
where completeness is countable from the digest. The sessions you actually
intend to run (RDD artifacts + ensembles on real work) will have messier
shapes: repair tasks, mixed read-then-write flows, asks whose deliverable
count isn't legible in the task text.

What would you need to believe about those real session shapes for the
deliverable-accounting standard to hold there — or, put as a pre-mortem: if a
month after shipping, ADR-037 turns out to have been the wrong call, what
does the failure look like from where you sit?

## User's response

> "I think failure looks like not being able to track everything we need to
> track in the meta-context. So these RDD sessions have lots of complexity,
> so being sure we're able to reason at the meta-level (I assume here by
> virtue of our framework) is important."

Follow-up exchange: the agent gave the current-state inventory of the
assumption (V-03 — no session-scoped action record exists yet; the digest is
BUILD work, deliberately narrow as committed), pinned the
framework-records/model-reasons distinction (the framework-reasons pole is
the closed candidate (b)), and proposed folding the pre-mortem into ADR-037
as the digest-expressiveness-ceiling consequence plus carrying it as the
lead ARCHITECT feed-forward. At commitment gating the agent proposed the
settled/open split; the user's response:

> "Yep those seem right to me."

## Pedagogical move selected

Challenge (pre-mortem form, composed from the θ false-stop asymmetry and the
constructed-base narrowness in the ADR's scope-of-claim).

## Commitment gating outputs

**Settled premises (the user is building on these going into architect):**
- The two-call trailing composition is the termination mechanism (ADR-037).
- Form B-enriched (bare-form judgment; framework-owned digest;
  deliverable-accounting standard) is the adopted composition; Form
  A-enriched is the measured fallback.
- The framework-records/model-reasons split: the framework guarantees the
  judgment's evidence base and enforces its consequences; the verdict is
  model-rendered within a bounded role.
- Call 2 is the unchanged ADR-036 E4b composition (judgment exchange
  discarded), so the measured delegation evidence rides.

**Open questions (the user is holding these open going into architect):**
- Digest expressiveness for real RDD-session complexity — the user's
  pre-mortem: failure looks like the meta-record not tracking everything
  needed; the committed write-log digest is the first increment of an
  extensible meta-record seam, not its final form.
- The non-write-shaped deliverable boundary (deliverable accounting may not
  bound the judgment there; recorded, not guessed across).
- Judgment-seat portability beyond the one hosted pair (20/20 minimax
  annotation; favorable prior, not a measurement).
- Whether the hosted judgment seat is ever adopted for latency (paid/local
  scoping principle: optimization slot, local standing fallback).

**Specific commitments carried forward to architect:**
- Digest home and shape are an ARCHITECT allocation question; design it as
  the extensible meta-record seam with the write-log as first increment
  (practitioner pre-mortem is the lead signal).
- The false-stop share (termination-observability FC) is the
  extend-on-evidence trigger for digest enrichment (labeled drafting-time
  synthesis, instrumentation-first pattern).
- The system-design/ORIENTATION sweep for the ADR-036 partial update rides
  the ARCHITECT regeneration (deferral recorded at DECIDE).
- AS-3 BudgetController wiring into the loop-driver path rides the BUILD WP
  (conformance incidental: the named backstop is not currently active on
  this surface).

# ADR-042: Bounded Content Anchor — Windowing the Produced-Sibling Anchor at Scale

> **Superseded by ADR-045 on 2026-07-01 (Cycle-8 clean-slate collapse, AS-11).** The imperative loop-driver serving architecture is retired; its implementation is removed, not adapted. This ADR was already Reverted (Spike τ′); its content-coherence concern is carried forward to the Cycle-8 declarative target per ADR-045's carry-forward table.

> **Extends ADR-039 (Content Anchor).** ADR-039 routes the produced-sibling API
> signatures into the callee dispatch so a dependent file references real sibling
> APIs (Finding H), with selection policy **all prior produced siblings** (ADR-039
> §Decision; a dependency-inferred subset was deferred). Spike τ found the
> unbounded all-prior anchor **bloats and degrades the coder at scale** — a
> persistent form bleed that coder-tier escalation cannot fix, because every tier
> is fed the same bloated anchor. This ADR bounds the anchor to the most recent K
> siblings. ADR-039 carries an `> Updated by ADR-042` header for its selection
> policy; ADR-039's content-agnostic signature mechanism is unchanged.

**Status:** Superseded by ADR-045 (2026-07-01); formerly **Reverted (Spike τ′, 2026-06-18).** This ADR bounded the content
anchor to the most recent K=8 produced siblings on a Spike τ anchor-overload
hypothesis. The Spike τ′ isolation probe refuted that mechanism (form-validity
30/30 across unbounded / bounded / full-content fallback) and found the bound costs
cross-file coherence (reference resolution monotonic in anchor size), so the K=8
bound is reverted to ADR-039's unbounded all-prior selection
(`loop_driver._CONTENT_ANCHOR_MAX_SIBLINGS = None`). Dependency-scoped selection is
the tracked successor if a genuinely large-session overload is ever observed. See
§Reassessment for the evidence; the Context / Decision / Consequences below are the
original ADR as accepted 2026-06-17, retained for provenance and superseded by
§Reassessment.

## Reassessment (Spike τ′ isolation probe, 2026-06-18)

The full-session ladder could not test condition (a) (it was masked by the J-3
over-extraction non-termination, fixed 2026-06-18). An isolated single-dispatch
probe did: `scratch/spike-tau-anchor-overload/` (n=10 per arm, qwen3:8b, the real
`build_content_anchor` over run 1's 20 produced siblings, task/target/coder held
fixed, only the anchor varied).

- **The overload mechanism does not reproduce.** Form-validity was 10/10 in all
  three arms: A_unbounded clean (1.7KB), B_bounded (K=8, 440B), and A_fallback
  (full-content, 6KB, the form-bled / unparseable-sibling condition this ADR
  blamed for amplified bloat). "The unbounded anchor bloats and degrades the
  coder, a form bleed" does not hold at any tested condition up to 6KB / 20
  siblings.
- **The bound shows a coherence COST, not a benefit.** Correct cross-file
  reference resolution was monotonic in anchor content: B_bounded 3/10 <
  A_unbounded 7/10 < A_fallback 10/10. More anchor gave the coder better
  resolution; the bound made it worse, the opposite of the value proposition, and
  exactly the coherence risk §Negative flagged. (n=10 per arm: form 30/30 is
  rock-solid; the resolution trend is monotonic across three arms but each
  pairwise gap is individually borderline.)

**Consequences for this ADR.** Condition (a) is refuted, not discharged. The
empirical observation that grounded the ADR (l15cap converged 15/15 vs l15 broke
12/15) is real but mis-attributed: the break was not the content anchor degrading
the coder, because no anchor condition does. Candidate true causes, untested here:
stochastic variation, the J-3 over-extraction non-termination (which presents as a
stalled session that reads as a "break"), or rig degradation on the longer
unbounded run. On current evidence the K=8 bound is **net-negative** (no
demonstrated form benefit, a demonstrated coherence cost), so it warrants
reconsideration: revert to unbounded, replace with dependency-scoped selection (the
deferred more-correct option, which preserves coherence), or gate any bound to a
much larger anchor scale than tested here. This decision is pending. Scope of the
refutation: tested to 6KB / 20 siblings; a far larger anchor remains untested.

## Context

Spike τ (`essays/research-logs/cycle-7-spike-tau-long-horizon-32gb.md`) probed the
long-horizon ceiling of the cheap-local agentic stack (hosted qwen seat + local 8b
coder + 8b→14b→MiniMax escalation) on a 32GB machine. The reliability stack
converged clean multi-file projects through 10 files, then broke at ~15:

- The break was a **persistent form bleed on one file (`step12.py`)** across **all
  coder tiers in sequence** — 8b recovery (`recovered=False`) → 14b escalation
  (`tier_profile=agentic-tier-escalated-general`) → MiniMax escalation
  (`tier_profile=agentic-tier-frontier-coder-minimax`) — all form-invalid, ladder
  exhausted → the FormGate refusal ended the OpenCode loop (FC-57), 12/15.
- A **trivial** file failing identically across 8b, 14b, AND MiniMax ⇒ **not the
  coder**. The common input is the ADR-039 all-prior content anchor (11 siblings by
  step12), which bloats the dispatch and degrades every tier's output. Escalation
  is the wrong lever — it changes the coder; the problem is the anchor.
- **Bounding the anchor fixed it.** Env-gated test (keep only the last 3 siblings),
  re-ran l15: **converged 15/15, all metrics clean, ZERO recovery/escalation
  fired** — vs the unbounded 12/15 break, template held constant. Coherence held
  (the linear dependency was in the last-3).

**Honest caveat (drives the Conditional):** the ladder task template had an
off-by-one bug (each step file was instructed to import *itself*), producing
incoherent siblings; unparseable siblings hit ADR-039's full-content fallback,
which may have *amplified* the anchor bloat. So the fix is cleanly demonstrated
(l15cap vs l15 held the template constant; only the bound changed), but the exact
bleed **threshold on clean tasks** is confounded — hence Conditional.

## Decision

**Bound the content anchor to the most recent K produced siblings** (a recency
window over ADR-039's all-prior selection).

- `build_content_anchor(siblings, *, max_siblings=K)` windows to the last K; `None`
  = all (pre-amendment behavior), `0` = none. (`sibling_interface_extractor.py`.)
- The Loop Driver passes `_CONTENT_ANCHOR_MAX_SIBLINGS` (`loop_driver.py`).
- **Default K = 8.** Preserves the common ≤8-file case unchanged (all siblings) while
  capping growth below the observed bleed onset. The value is a **heuristic pending
  the §Discharge re-run** — the validated point was K=3 (form-clean to 20 on the
  linear template); 8 trades bloat-margin for non-linear dependency coverage.
- Recency is the dependency heuristic; **dependency-scoped selection** (only the
  siblings the current file actually depends on) is the deferred more-correct
  option (already deferred by Spike ξ), and is the natural successor if the
  windowed bound shows coherence regressions on non-linear tasks.

## Consequences

**Positive.** Raises the long-horizon ceiling (≥15 files form-clean with the bound,
vs the ~12-file unbounded break); removes the recovery/escalation overhead the
bloated anchor *caused* (l15cap fired zero recovery, vs l15's exhausted ladder); the
ceiling above 15 is then set by other factors (conversation-context growth, not yet
isolated — the next probe), not the anchor.

**Negative / open.** Recency windowing can miss a far-back dependency for a
non-linear task (a file depending on an early sibling outside the window),
re-introducing a coherence miss — the §Discharge re-run measures this. The default K
is a heuristic, not a tuned value. The dependency-scoped successor remains deferred.

## Discharge (the Conditional Acceptance conditions)

A clean coherence-at-scale re-run (the Spike τ ladder with the **corrected**
template — the off-by-one fixed) at the bound, confirming: (a) the unbounded anchor
overloads on a *clean* task at scale (anchor-overload is real, not a template
artifact); (b) the windowed bound holds **content-coherence** at scale for the
clean (linear, and ideally a non-linear) dependency shape; and (c) a tuned K (or the
move to dependency-scoped selection). Until then the windowing mechanism stands
(form fix demonstrated), the K value and non-linear coherence are provisional.

## Provenance / grounding

Spike τ research log (`essays/research-logs/cycle-7-spike-tau-long-horizon-32gb.md`,
§"CORRECTION #2 + FIX CONFIRMED"). Implementation: `sibling_interface_extractor.py`
`build_content_anchor(max_siblings=)` + `loop_driver._content_anchor` +
`tests/unit/agentic/test_sibling_interface_extractor.py` (window tests). The spike
env-gate (`LLMORC_SPIKE_TAU_ANCHOR_CAP`) is removed (replaced by the bounded default).

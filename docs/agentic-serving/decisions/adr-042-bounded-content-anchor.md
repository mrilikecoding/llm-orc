# ADR-042: Bounded Content Anchor — Windowing the Produced-Sibling Anchor at Scale

> **Extends ADR-039 (Content Anchor).** ADR-039 routes the produced-sibling API
> signatures into the callee dispatch so a dependent file references real sibling
> APIs (Finding H), with selection policy **all prior produced siblings** (ADR-039
> §Decision; a dependency-inferred subset was deferred). Spike τ found the
> unbounded all-prior anchor **bloats and degrades the coder at scale** — a
> persistent form bleed that coder-tier escalation cannot fix, because every tier
> is fed the same bloated anchor. This ADR bounds the anchor to the most recent K
> siblings. ADR-039 carries an `> Updated by ADR-042` header for its selection
> policy; ADR-039's content-agnostic signature mechanism is unchanged.

**Status:** Accepted with Conditional Acceptance (ADR-097). The **windowing
mechanism** is demonstrated to fix the scale form-bleed (Spike τ: l15 broke 12/15
unbounded; l15cap converged 15/15 form-clean with the bound, template held
constant). Two conditions remain open, discharged by a **clean coherence-at-scale
re-run** (§Discharge): (a) confirm anchor-overload on a *clean* task — Spike τ's
ladder template had an off-by-one bug that confounds the exact bleed threshold; (b)
confirm the windowed bound preserves cross-file coherence for non-linear
dependencies, and tune K.

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

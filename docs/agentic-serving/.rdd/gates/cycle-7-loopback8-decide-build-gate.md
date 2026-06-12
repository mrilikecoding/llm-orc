# Gate Reflection: Cycle 7 loop-back #8 DECIDE → BUILD

**Date:** 2026-06-12
**Phase boundary:** DECIDE → BUILD (loop-back #8; ARCHITECT skipped — composition within the existing Loop Driver + the pre-built FormGate seam, a BUILD Design Amendment per the #6/#7 precedent)
**Cycle:** Cycle 7 — Framework-driven orchestration (agentic-serving scoped)

## Belief-mapping question composed for this gate

The argument audit and the susceptibility snapshot converged on the same unresolved point: ADR-041 changes *where* a persistent cheap-coder failure surfaces (a cap-exhausted short session instead of a shipped broken file), but does not settle whether that is a better outcome. The question composed (a belief-mapping / inverted-framing challenge on the most consequential unexamined trade):

> Is a cap-exhausted short session actually a better outcome than a complete-but-broken-file session? What would have to be true for that to be the right trade for the Tool User, versus shipping the broken file and letting them reject the diff? That answer is what tells us whether "protection" is the real win or whether the win is really the recovery plus the escalation signal.

## User's response

Turn 1 (redirect to the north-star premise — the cycle's recurring outcome-derivation move): *"Agreed around the deterministic gates. For the open questions there, I'd come back again to our north star premise and ask what you think gets us there?"*

The agent derived from the north star (a cheap-local stack under the real client matching a frontier single model, with the framework carrying structural reliability so "all that remains is ensemble iteration"): the short-session-vs-broken-file binary is off-target from the north-star lens, because the target is complete-and-valid. The decisive reason the gate (protection) is north-star-aligned is that the broken file is a *silent structural failure* that smears a quality miss into a delivery miss, whereas the short session is an *honest structural success with a quality shortfall* that keeps the structure/quality boundary legible — and the north-star claim "all that remains is ensemble iteration" is only provable if the framework cleanly separates structural delivery from ensemble quality. This resolves Framing A (protection-primary, with routing-signal as a consequence). The path to the target is deterministic-gate-triggered selective coder-tier escalation (the ADR-014 lever), with the honest caveat (Arm E: cheap 8b bleeds ~50% even fresh) that a strictly cheap-local coder is capped on the hardest files.

Turn 2 (agreement plus two refinements): *"I think that's wise. I like the escalation ladder from cheap to less cheap to frontier. And the frontier / cost model could be opt-in at a layer or before. And I've also defined north star as a long-horizon RDD-like flow."*

The two refinements were folded into ADR-041: (1) the escalation ladder's frontier rung is opt-in / cost-gated with a local-degradation path — deterministic *trigger*, policy *ceiling*, preserving the free-first standard; (2) the north-star target is a sustained long-horizon RDD-like flow, not a single 5-file trajectory, which makes the long-horizon axis-2 regime the explicit PLAY validation target and reframes the PLAY question from "is a short session nicer" to "does the structure/quality separation stay legible over a sustained flow."

## Pedagogical move selected

Challenge (belief-mapping / inverted-framing on the experiential trade-off). The practitioner redirected to north-star derivation rather than answering the trade directly — treated as the proceed signal per the loop-back #2/#3 outcome-redirect precedent, with the agent deriving the answer from the premise rather than asking the practitioner to speculate. The two practitioner refinements (opt-in frontier; long-horizon flow) were adopted into the ADR with provenance attribution in the §Provenance check.

## Commitment gating outputs

**Settled premises (the practitioner is building on these going into BUILD):**
- The deterministic destination-validity gate (parse/validate against the destination path's claimed type) is the right protection mechanism. *(Explicit: "Agreed around the deterministic gates.")*
- Protection's value is **structure/quality-separation enforcement** (north-star-derived), with the routing-signal reading a consequence, not a competitor. *(Practitioner: "I think that's wise.")*
- The escalation ladder is **cheap → less-cheap → frontier coder tiers** (ADR-014 lever), the trigger being the deterministic parse-check failure.
- The **frontier rung is opt-in / cost-gated**, with a local-degradation path (cap at the best free rung → honest short session).

**Open questions (held going into BUILD / PLAY):**
- Whether the structure/quality separation reads as honest to a user watching a **long-horizon RDD-like flow** (the axis-2 regime; the gate/recovery were validated only on single 5-file trajectories, n=5×2).
- Whether convergence-under-escalation actually holds in a **wired** session (Arm E proved the coder-tier lever only in isolation, n=6 on `cli.py`).
- The recovery cap's behaviour across many turns over a sustained flow.

**Specific commitments carried forward to BUILD:**
- The env-gated gate + recovery code (`LLMORC_SPIKE_PI_GATE=parse`) is the **BUILD seed**: de-gate (×3), reconcile the `FormGate` interface to 3 args `(content, destination_tool, destination_path)`, install the parse-check as the bridge's `form_gate`, and close the 2 test-gap violations — per `housekeeping/audits/conformance-scan-cycle-7-loopback8.md` (10 BUILD-work items + 2 violations; `destination_path` already threaded).
- The escalation-ladder wiring (ADR-014) must make the **frontier rung opt-in** with a local-degradation path.
- **ARCHITECT skipped** (practitioner-confirmed at this gate) — the system-design touch is a BUILD Design Amendment.
- The long-horizon RDD-like flow and the opt-in cost model are **PLAY / first-deployment observation targets** (carried as the two snapshot feed-forward items plus the gate-surfaced axis-2 framing).
- The ADR-035 form-seam Conditional Acceptance is **design-discharged** at this gate; **install-discharge** lands when the BUILD de-gate completes.

## Audit + snapshot disposition

- Argument audit converged R2 (`housekeeping/audits/argument-audit-decide-cycle-7-loopback8{,-round2}.md`); all argument P1/P2/P3 applied; two framing alternatives surfaced and resolved at this gate.
- Susceptibility snapshot (`housekeeping/audits/susceptibility-snapshot-cycle-7-loopback8-decide.md`): **No Grounding Reframe** — earned confidence; FF1 (practitioner-framing absorption) low because the spike's pre-registered refutability grounded the framing by outcome; two feed-forward items for BUILD (recovered-case ADR-040 trace; routing-signal/experiential observation from the first live session).
- Gate-driven ADR-041 refinements (structure/quality framing; opt-in ladder; long-horizon axis-2) judged additive/sharpening within the audited claim space (they resolve the auditor's own Framing A and compose already-cited priors ADR-014 / axis-2 / free-first), so no full re-audit was triggered; recorded here as the judgment.

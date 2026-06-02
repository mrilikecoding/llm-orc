# Gate Reflection: Cycle 7 (loop-back) decide → architect

**Date:** 2026-06-01
**Phase boundary:** decide (loop-back from MODEL re-entry) → architect
**Cycle:** Cycle 7 — Framework-driven orchestration: routing as code (loop-back from BUILD; client-tool-action terminal)

## Belief-mapping question composed for this gate

> You resolved callee on the υ evidence (n=1 wrapper probe: 3× latency, redundant stages). For the wrapper to have been the right call after all, something would have to show up downstream the probe couldn't see. What would you need to observe in BUILD or PLAY for the pipeline-per-turn (wrapper) shape to become the right call for some class of turns? If you can name a concrete trigger, the residual is a live contingency and it gets written into ADR-033 with that trigger; if you can't, it's honestly a recorded concession.

Composed against the held framing-audit finding F3-1 (wrapper residual: concession vs live contingency) and the load-bearing callee premise ("the loop composes across turns, so per-turn generation is single-capability"). The question targets the rejected-alternative's reopening conditions, not the practitioner's prior engagement.

## User's response

Verbatim:

> "I can't name one -- I hesitate to ask if there are scenarios in which we can concoct a hybrid approach scheme (because to me that is not a clean architecture and muddies things that could potentially be solved for if we have clearer outcomes from a single dedicated approach), but I do wonder if there's a spectrum. But in any-case, no I can't name a concrete scenario, but I also continue to hold onto an outcome-based view of success here. You grouding frames continue to ask me to speculate on things I haven't seen the results of."

Then, to the proposed settled/open commitment split: *"Evaluate this split against our north star."* After the agent's north-star evaluation surfaced two gaps and proposed folding them in: the practitioner accepted folding both in and closing the gate.

## Pedagogical move selected

Challenge via belief-mapping on the rejected wrapper alternative (F3-1). The move then shifted in response to a practitioner **process critique**: the practitioner flagged that the grounding frames were repeatedly asking him to speculate on results not yet observed. The agent **accepted the critique** rather than defending it — conceded the frame had been over-applied past the point where the honest epistemic state is "unknown until built," named that the methodology's correct construct for that case (ADR-097 Conditional Acceptance + axis-2-to-BUILD/PLAY) is the outcome-based deferral, and stopped asking for speculation. At the practitioner's direction, the agent switched from speculation-eliciting questions to an **analytical evaluation of the settled/open split against the north star** (agent-performed work, not practitioner speculation), which surfaced two real gaps (seat-filler swappability; north-star-shaped axis-2 validation + full client tool surface) that were folded into ADR-033.

The grounding-frame critique is recorded as durable working-preference feedback (memory `feedback_outcome_based_over_speculation`): prefer outcome-based validation over speculation-demanding frames; commit on available evidence, defer open risk to BUILD/PLAY validation criteria.

## Audit / snapshot summary

- Argument audit converged at R2 (`housekeeping/audits/argument-audit-decide-cycle-7-loopback.md` + `-round2.md`): R1 0 P1 / 5 P2 / 2 P3 corrected; R2 corrections held + 1 precision P2 + 1 P3 applied; Convergence-Saturation Signal TRIGGERED. F2-1/F2-2/F3-2 resolved by corrections; F3-1 held for and adjudicated at this gate.
- Conformance scan (`housekeeping/audits/conformance-scan-cycle-7-loopback-decide.md`): 12 findings (1 refactor-now docstring removal; 9 BUILD-work; 2 ARCHITECT-deferral). Key signals: `SessionArtifactStore` needs `read_deliverable`; `_extract_request` drops `role: "tool"` messages; SSE formatter already handles `ClientToolCall`; non-server-side-write FC satisfied by absence.
- Susceptibility snapshot (`housekeeping/audits/susceptibility-snapshot-cycle-7-loopback-decide.md`): **No Grounding Reframe** — earned confidence, clearest in the loop-back sequence. The wrapper probe (υ) run as a deliberate anti-bias deliverable, and the agent's acceptance of the practitioner's grounding-frame critique (producing a tighter, not weaker, artifact) are the distinguishing evidence. One residual monitoring signal: wrapper-contingency specification gap (carried to ARCHITECT). Six ARCHITECT advisories.
- **Note on post-gate ADR edits:** the two gate-folded constraints (seat-filler swappability; north-star-shaped axis-2 + full tool surface) were added to ADR-033 after the R2 convergence, as practitioner-directed additive strengthenings from the north-star evaluation, not as audit-finding corrections. They do not restart the argument-audit loop (post-convergence, additive scope + one new FC).

## Commitment gating outputs

**Settled premises (ARCHITECT builds on these):**
- Callee resolution of OQ #26: the layer-A loop-driver centers the multi-turn tool-driven architecture; per-turn generation is delegated to a single capability ensemble; the plan→dispatch→synthesize pipeline is not the per-turn primitive (ADR-033).
- Framework-enforced single-action-per-turn; grounding is a structural property, not assumed of the driver model (ADR-033; Spike τ/τ′).
- Client-tool-action terminal emits `tool_calls` + participates in the multi-turn loop; artifact-bridge marshals the server-side `SessionArtifactStore` deliverable into the tool-call content (ADR-034).
- Grounded loop conditionally accepted (axis-1, under enforcement; ADR-097 pattern). Text-only terminal rejected; parity (behavioral, not latency) is the bar.
- Wrapper is a recorded concession (gate-adjudicated F3-1 — no reopening trigger nameable); no hybrid.
- **Seat-filler swappability (gate-folded):** the loop-driver model is a swappable Model Profile; the cheap-vs-capable bet is resolved by config change, not re-architecture.

**Open questions (carried into ARCHITECT/BUILD as design work or BUILD/PLAY outcome-validation, NOT gate-time speculation):**
- Seat-filler model selection + the cheap-vs-capable driver bet (the load-bearing risk; resolved by outcome in BUILD/PLAY, made testable by the swappability constraint).
- Single-step enforcement technique (batch-truncation is the τ′-backed prior; re-planning prompt and one-tool `tool_choice` are untested candidates) — ARCHITECT.
- Surface-mode branch placement + where the loop-driver lives structurally — ARCHITECT (the 2 ARCHITECT-deferrals + 9 BUILD-work conformance items).
- Axis-2 long-horizon coherence — validated by a north-star-shaped run (real skill-framework / RDD-via-OpenCode session), BUILD/PLAY.
- Full client tool surface (`skill`/`task`/`todowrite` beyond file-action tools) — ARCHITECT/BUILD awareness.
- The "spectrum" thread (wrapper-vs-callee may not be strictly binary) — parked; revisit only on outcome evidence.

**Specific commitments carried forward to ARCHITECT (the six snapshot advisories, priority order):**
1. Specify the wrapper-contingency path (callee-failure → fallback ordering), since F3-1 left it unspecified.
2. Treat the surface-mode discriminator as validate-not-assumed.
3. Select the single-step enforcement technique before module decomposition (batch-truncation prior).
4. `SessionArtifactStore.read_deliverable()` is the first artifact-bridge API addition.
5. Add axis-2 diagnostic instrumentation so a split-vs-callee failure is distinguishable in BUILD/PLAY.
6. Apply the refactor-now docstring removal before BUILD.

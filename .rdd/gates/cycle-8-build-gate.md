# Gate Reflection: Cycle 8 (declarative-ensemble collapse) BUILD → graduate

**Date:** 2026-07-08
**Phase boundary:** build → graduate (cycle-end; BUILD complete at WP-F8, the last WP)
**Cycle:** Cycle 8 — The declarative-ensemble collapse

## Belief-mapping question composed for this gate

The WP-F8 boundary was a gated-mode surgery, not a single AID reflection gate; the belief-mapping was distributed across four decision points, each posing "what would have to hold for this to be the right move." The load-bearing one, at the deletion go/no-go: *"The written pre-flight is the plan of record. What would have to be true about the actual code for its step order and its delete/keep list to be wrong — and have we checked, rather than assumed?"* This inverted the pre-flight from a constraint into prior art to verify against the code.

## User's response

Across the four gates the practitioner selected the agent-recommended option each time: (1) launch WP-F8; (2) reorder the surgery to 1→2→0 after the agent showed, from a code read, that step 2 collapses the config surface step 0 relocates; (3) proceed with the delete/migrate/edit categorization after the agent disclosed a blast-radius overrun (~18 integration tests the pre-flight never listed) and three inline judgment calls; (4) coherent doc-sweep depth (banner historical, regenerate current-state, do not rewrite the 362KB companion). No counter-proposals; the practitioner ratified agent recommendations that were each grounded in a verifiable code finding.

## Pedagogical move selected

Probe-then-proceed at each decision point: surface the specific code finding, present the alternative(s) explicitly (keep-pre-flight-order; migrate-FC-tests-instead; full/minimal sweep), recommend with rationale, and act on the ratified choice. No teaching move was needed — the practitioner holds the deepest cycle context; the agent's role was to make the pre-flight's assumptions checkable against code truth and surface where they diverged.

## Grounding Reframe (from the build-phase susceptibility snapshot)

The isolated snapshot evaluator recommended a narrow Grounding Reframe: the doc-sweep-depth decision (gate 4) rested on judgment the test suite does not cover, and a corpus-wide grep for the thirteen deleted module names surfaces `architecture-map.md` as a current-state document still describing the deleted `LoopDriver`/`OrchestratorRuntime` chain as ACTIVE, with file:line citations that are now dead links — a doc the Step-7 list had missed, and one ORIENTATION still pointed to as reference.

**Pursued (not disclaimed).** Ran the grep across all current-state-facing docs; confirmed the other docs' agentic references sit in historical/banner-marked context (domain-model, scenarios, interaction-specs) or in "what was deleted" notes (field-guide). Bannered `architecture-map.md` superseded/historical and repointed ORIENTATION's reference to `field-guide.md` for the current map. Also caught a parallel-agent coordination artifact the grep surfaced: ORIENTATION had been regenerated concurrently with field-guide and so described field-guide as "stale since before Cycle 6" — reconciled, since field-guide is now the current map. Evidence for the next snapshot: the doc-hygiene gap was real, specific, and closed in-cycle.

## Commitment gating outputs

**Settled premises (going into graduate):**
- Cycle-8 BUILD is complete. The bespoke imperative `agentic/` layer is deleted; the declarative Serving Ensemble (classify → seat → marshal, ADR-046 §1) is the only serving path, grounded through the real L3 endpoint (build + explain turns) with `agentic/` gone.
- The full survivor set is relocated (WP-B8 + WP-F8): session substrate → `core/session/`, validation → `core/validation/`, envelope → `models/`, serving contracts → `web/serving/`, registry/catalog/admission → `core/serving/`, allow-list → `core/config/`.
- ADR supersession (033/036/037/039/041/043 → 045) was already stamped in DECIDE; ADR-002 layering reverts to no-exception (Amendment #23).
- Parity-before-delete held: 2307 unit + 31 integration green (1 known pre-existing env failure); safety tag `cycle-8-pre-agentic-deletion` retained.

**Open questions (held into graduate):**
- Full model-parity's named frontier: fix / edit / run-tests default seats (future default shapes, grown the same way, PLAY-battery-validated per DISCOVER criterion #1).
- The compose-at-runtime primitive + composer-ensembles (ADR-047 deferred direction).
- Corpus doc-hygiene fragmentation flagged by the snapshot (nine `dispatch-log.jsonl` files; audits split between `.rdd/audits/` and `docs/agentic-serving/housekeeping/audits/`) — a graduate-time reconciliation concern, not a code concern.
- Stale docstrings on a few surviving code modules (`dispatch_envelope`, `plexus_adapter`, `registry`, `chunks` vestigial variants) still narrate the dissolved orchestrator — a doc-drift cleanup candidate the field-guide regen flagged; code is current.

**Specific commitments carried forward to graduate:**
- Clean-`main` merge carries native docs only (no 8-cycle RDD scaffolding); the full RDD corpus survives on a research branch for posterity.
- The `agentic/` CODE deletion (done here) is distinct from CORPUS archival (graduate, terminal). Spike-artifact retention stays deferred to corpus-close.

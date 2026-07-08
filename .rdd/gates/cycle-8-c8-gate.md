# Gate Reflection: Cycle 8 BUILD — WP-C8 → WP-F8

**Date:** 2026-07-08
**Phase boundary:** BUILD WP-C8 (Shape Catalog + registry reuse) → WP-F8 (clean-slate removal)
**Cycle:** Cycle 8 — the declarative-ensemble collapse

*(Within-BUILD scenario-group gate, gated mode; the practitioner asked to gate at each WP boundary.)*

## Belief-mapping question composed for this gate

WP-C8's realized property is "the Shape Catalog IS the verification-shape map": each shape
carries its own verification rung (`build-gated` bundles executor+judge; `explainer` bundles
none; `gen-review` bundles a review part), so "match the rung to the deliverable" reduces to
"which shape an intent routes to." But the catalog keys on classify's *intent*, and the rung
is baked into whichever shape an operator authored — so "a code deliverable gets a strong
rung" holds only because an operator wired `build-gated` correctly. A mis-authored shape (a
code intent routed to a judge-only shape) would ship weak verification silently, and nothing
in WP-C8 catches it. Is operator-owned rung-correctness the right boundary (same as
operator-owned AS-2-valid parts), or does the property want a structural check that a
build/executable intent resolves to a shape carrying a deterministic rung?

## User's response

*"My expectation is that llm-orc would ship with good default seat arrangements but then
custom ensembles could be added into seats or new flows could be extended etc..."*

The boundary is **good defaults + extensibility**: the framework's responsibility is that the
*shipped* defaults are good (the correct verification rung baked into each default shape); the
operator's domain is extension — adding custom capability ensembles into seats, or authoring
new flows. Rung-correctness of a custom extension is the operator's, not the framework's to
police.

## Pedagogical move selected

Challenge. The agent surfaced the operator-correctness tension the "verification-shape map"
framing concealed (the property is realized by good defaults, not enforced by structure). The
practitioner resolved it onto the good-defaults + extensibility division, which rules out the
structural rung-check (it would fight the extension freedom that is the point) and reframes the
framework's obligation as shipping correct defaults rather than policing extensions.

## Commitment gating outputs

**Settled premises (carried into WP-F8 and forward):**
- **Good defaults + extensibility is the operating model.** The framework ships good default
  seat arrangements (the verification-shape map correct out-of-box); operators extend by adding
  custom ensembles into seats (the Topaz-keyed part registry) or authoring new flows (the Shape
  Catalog). No structural rung-check — it would fight extension freedom. Rung-correctness of a
  custom extension is the operator's, parallel to operator-owned AS-2-valid parts.
- **The verification-shape-map property is realized by shipping correct defaults, not enforced
  at runtime.** WP-C8 ships the two-lane default correctly (code → `build-gated` executor+judge;
  explain → `explainer` prose) and the three extension paths (part-into-seat, seat-swap,
  new-flow), all grounded.
- **Default coverage today = build + explain.** Full model-parity's fix / edit / run-tests are
  the named frontier — future default shapes grown the same way (add a shape to the catalog),
  validated by the PLAY battery (DISCOVER criterion #1). WP-C8 ships the machinery + the first
  good defaults; the default set grows over cycles.

**Open questions (held forward):**
- The `validation:` `schema:`-key quirk (the YAML `schema:` key does not map to
  `ValidationConfig.schema_validations`) — tracked polish, addressed when operators begin
  authoring richer per-capability seat contracts (not yet; not speculatively).
- Which fix / edit / run-tests default shapes to ship, and when (frontier; PLAY-battery-driven).

**Specific commitments carried forward to WP-F8:**
- WP-F8 (the last WP) deletes `src/llm_orc/agentic/` + the Cycle-7 serving-ADR supersession +
  the deferred current-state doc sweep, under parity-before-delete. WP-C8 reached parity (2941
  unit green, loop-driver byte-untouched; the declarative serve + registry + catalog grounded
  through the real endpoint), so the last parity precondition for F8 is met.
- The current-state doc sweep (system-design / ORIENTATION / roadmap / field-guide) lands with
  F8's deletion, per the standing deferral (rewriting them before the collapse ships would
  document an unbuilt architecture).

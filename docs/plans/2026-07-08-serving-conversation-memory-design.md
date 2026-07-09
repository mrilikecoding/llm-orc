# Serving Conversation Memory — Design

**Date:** 2026-07-08
**Goal:** close the serve's below-parity gap (single-turn by construction) and
set the architecture that scales past single-model memory toward the north
star (full model parity via composition; see `docs/serving.md`).

## Architectural stance

- **The wire is ground truth.** OpenAI-compatible clients send the full
  conversation every request. We never require the client to change.
- **The system has no context window.** Unlike a single model, our record
  lives on disk. We therefore never lossily compact the record itself —
  client-side compaction is a mitigation for a limitation we don't have.
- **The limitation moves to the seat.** Each dispatch feeds a bounded local
  model, so the per-dispatch question is *selection* (which slice of the
  record does this task need), not compression.
- **Wire divergence is a signal.** When the client compacts its transcript,
  our lossless record becomes strictly richer than the client's. A prefix
  mismatch classifies as: client compaction (record stays authoritative),
  or edit/fork (branch or rebuild from the wire).
- **Judge isolation holds.** The accept-gate judge's input stays
  `{criteria, artifact, execution result}` (ADR-048); conversation context
  threads to generation seats, never to verifier seats.

## Rungs

### Rung 1 — bounded stateless render (committed)

The caller renders the wire history into a deterministic, capped context
string and passes `{"task": <latest>, "context": <render>}` to the serving
ensemble. classify routes on `task` alone and composes
`dispatch_input = "Conversation so far: … Current request: <task>"`.
Caps: last 8 messages, ~500 chars per text message, ~2,000 chars per
written-file body (tool_call arguments), ~4,000 chars total (drop oldest
first). No storage. This rung is the seat-input plumbing that the higher
rungs reuse.

### Rung 2′ — lossless session record + selection (committed)

Per resolved session identity (the registry's `resolve_identity`, already in
the request path): an append-only record of turns, tool calls, and written
artifacts. Each request hash-checks the integrated wire prefix; on match,
integrates only the delta. Seat context is composed by deterministic
*selection* over the record: verbatim recent tail + referent matches
(filenames and symbols named in the task). No rolling summarization.
Divergence: collapsed-prefix (client compacted) → record stays
authoritative; rewritten tail (edit/fork) → rebuild from wire.

**Entry gate:** observe what OpenCode's compacted wire actually looks like
(one long-session observation) before hardening the divergence classifier.

### Rung 3 — plexus lenses (named direction, not this branch)

The record ingests into a plexus knowledge graph; per-dispatch slices come
back as provenance-tracked lens queries, cross-session. Rung 2′'s record is
shaped to be plexus-ingestable so this is an upgrade, not a rewrite.

## Testing

- Renderer and selection: pure unit tests (deterministic).
- classify: routing unaffected by context; composition correct.
- Endpoint: a multi-turn history reaches the seat (trace-asserted).
- Judge isolation: the existing ADR-048 fitness test must stay green; if
  threading context into the build shape leaks history into the judge, that
  is a blocking finding (engine-level input scoping), not an accepted loss.
- Live battery: "add tests for it" (referent resolution) and "did you see my
  previous query?" (memory) through real OpenCode.

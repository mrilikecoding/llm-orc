# Gate Reflection: Cycle 7 loop-back #9 DECIDE → BUILD

**Date:** 2026-06-18
**Phase boundary:** DECIDE → BUILD
**Cycle:** Cycle 7 — loop-back #9 (collapse the dual serving surfaces to one)

## Belief-mapping question composed for this gate

The collapse leans on one load-bearing premise: that toolless clients are an edge
case, because the north-star client (OpenCode) always sends tools — which is what
makes dropping the single-shot multi-ensemble fan-out (the Cycle 6 origin vision,
the `web-searcher → claim-extractor` composition) feel safe. The question posed:
*what would have to be true about who actually calls agentic-serving for that
dropped capability to come back and bite — and is the Cycle 6 "ask-and-compose"
client a live roadmap possibility (favoring dormant-but-revivable) or genuinely
pivoted-away (favoring clean deletion)?*

## User's response

The practitioner pressure-tested the retirement against the north star across three
moves rather than answering directly:

1. *"What would an example of a tool-less client be?"* — required the abstraction be
   made concrete before judging the retirement. (Answer surfaced: plain scripts,
   chat UIs, other apps using the endpoint as a completion model, and — concretely —
   OpenCode's own auxiliary tools-less requests for title/summary generation, which
   currently 500 on the half-built pipeline and which the collapse fixes.)

2. *"Do those OpenCode aux tools not having a path to using fanned-out ensembles pose
   an issue to doing a flow like our north star warrants?"* — tested whether the
   north-star RDD-via-OpenCode flow needs toolless fan-out. (Resolved: it does not —
   the north star's composition is fan-out across turns in the loop, driven by
   OpenCode executing tool calls; the toolless aux path only needs lightweight text,
   evidenced by Spike ι's `plain_caps` cell finishing 10/10 with no over-delegation.)

3. *"Well one option for the latter plain-API case could be actually just wrapping
   OpenCode itself. We've been driving OpenCode via CLI for these spikes anyway."* —
   the resolving insight: the revival path for the plain-API ask-and-compose surface
   is to wrap a turn-driving agent (OpenCode) in front of the loop, not to restore
   the pipeline.

## Pedagogical move selected

Challenge. The gate surfaced the retirement as a pre-mortem / belief-mapping tension
referenced to specific artifact content (the toolless-client premise, the dropped
single-shot fan-out, the Cycle 6 origin), rather than seeking approval of the ADR.
The practitioner's third move resolved the tension upward into a cleaner architecture
(single composition mechanism; agentic-serving's responsibility kept to the
model/loop surface; orchestration in an agent above it), which was folded into
ADR-043 (§Rejected alternatives + §Consequences) as the justification for full
deletion over dormant-keep.

## Commitment gating outputs

**Settled premises (the user is building on these going into BUILD):**
- Collapse to one loop-driven serving surface (ADR-043).
- The Dispatch Pipeline is **deleted outright** (ADR-027 retired), not dormant-kept.
- F-ι.1 → Resolution B: adaptive deliverable marshalling, a Terminal-only change
  (`_emit_apply_work` branches on client-tool presence); Loop Driver unchanged.
- There is one composition mechanism — the loop driven by a tool-executing client;
  fan-out happens across turns. OpenCode is that client.
- OpenCode's toolless aux requests are served by the loop (Spike ι `plain_caps`
  evidence); the collapse fixes their current 500.
- The plain-API "ask-and-compose" revival path is wrap-OpenCode, not pipeline-revival.

**Open questions (held into BUILD):**
- B's correctness is a BUILD obligation — `_emit_apply_work` must implement the
  marshalling branch; the FC (adaptive marshalling) is the refutable anchor.
- Live-confirm: a real OpenCode aux (toolless) request lands a clean answer —
  verifies aux-request quality and the 500 fix. Discharge gate.
- ADR-031 / ADR-032 dormancy handoff notes (backward-propagation sweep).
- Whether the plain-API ask-and-compose surface ever materializes (held; revival
  path documented, not built).

**Specific commitments carried forward to BUILD:**
- The 14-item conformance-scan work-list (`housekeeping/audits/conformance-scan-cycle-7-loopback9.md`),
  confirmed one logical BUILD unit (BUILD Design Amendment, no ARCHITECT pass).
- Commit split: delete-first (subtractive — pipeline modules + tests + the
  `test_fc2_layering.py` layer-map) then the F-ι.1 marshalling commit (Terminal
  branch + a refutable adaptive-marshalling test).
- Backward-propagation sweep: supersession headers (done on ADR-027/033);
  system-design / system-design.agents / roadmap (retire WP-B/C/D/E) / ORIENTATION;
  ADR-031/032 dormancy notes.
- Live-confirm against real OpenCode as the discharge gate.

## Grounding Reframe (post-snapshot, discharged 2026-06-18)

The DECIDE→BUILD susceptibility snapshot (`housekeeping/audits/susceptibility-snapshot-cycle-7-loopback9-decide.md`) judged the A→B reframe a partial FF1-pattern adoption: the load-bearing premise *"OpenCode always sends tools, so the A/B choice only governs toolless clients"* was an asserted practitioner claim the agent adopted and built on, not a measured result. It recommended a narrow, in-cycle Grounding Reframe before BUILD: verify what OpenCode actually carries in `tools[]`.

**Grounding action taken (≈$0, real captures):** inspected the Spike π Phase-0 real-OpenCode request captures (`scratch/spike-pi-opencode-roundtrip/requests*.jsonl`).

**Evidence:** the premise is **false as stated**. OpenCode emits two request shapes — a build-agent request carrying the full 10-tool surface (`bash/edit/glob/grep/read/skill/task/todowrite/webfetch/write`) AND a verbatim toolless `title-generator` request (`system: "You are a title generator..."`, no `tools[]`). OpenCode itself sends toolless traffic.

**Disposition:** the correction does not destabilize the decision. OpenCode's toolless traffic is title generation — lightweight text that wants neither delegation nor fan-out, served by the loop's finish-with-text (Spike ι `plain_caps` 10/10), and the collapse fixes its current pipeline 500. The snapshot's specific worry (A's determinism edge becomes non-theoretical for OpenCode's own requests) does not materialize: a title-gen request is served identically by A or B. Resolution B stands as the right general choice (it preserves delegation for the toolless-capability case); the ADR-043 framing note is corrected from "OpenCode declares its tools on every request" to the verified two-shape reality. The assumption is now **grounded, not asserted** — the next susceptibility snapshot can record it as such.

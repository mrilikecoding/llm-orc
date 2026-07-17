# Plexus consumer integration: assessment and spec (WS-6, #127)

Status: assessment + design spec, written 2026-07-17 from a session without
access to the plexus repo. Grounded in the facts recorded in this repo (the
2026-07-11 survey in the roadmap archive §WS-6, issue #127, plexus v0.5.0:
~31K LOC Rust, ~550 tests, MCP over stdio with 18 tools — ingest, contexts,
find_nodes/traverse/find_path, evidence_trail, explain_edge,
shared_concepts, changes_since, load_spec — plus a multi-process
shared-SQLite path with `changes_since` cursors, and a working Python MCP
client at `plexus/tools/play-harness/mcp_client.py`). Every claim that
depends on plexus's current behavior is listed in §Verify on the rig and
must be checked against the live repo before implementation.

## Why this integration is the north-star bet, not a feature

WS-6 is where composition PASSES a single model rather than chasing it. A
frontier model behind any harness has no real cross-session memory — it has
a context window, harness-side files (CLAUDE.md, notes), and compaction
summaries, all lossy and none provenance-tracked. The serve already holds a
lossless in-session record; plexus extends that to a cross-session,
provenance-tracked substrate where every answer can carry a receipt
(`evidence_trail`).

The WS-2 measurement sharpened this from "nice" to "necessary": at n=3
independently-scored runs, every dishonest outcome was a **memory
presentation failure** — recall substitution without disclosure (#133) and
recap fabrication (#134). The general fix for both is the same shape this
integration generalizes: **claims about history must derive from a
deterministic record and carry its receipt, or fail closed to honest
refusal.** #133/#134 build that discipline on the in-session ledger now;
plexus is the same discipline made cross-session. This is one design idea
appearing at two scales, which is evidence it is the right idea.

## Topology decision (the design-first item)

Three candidate topologies, with the circularity constraint front and
center (today plexus spawns `llm-orc m serve` for semantic extraction, so a
naive "llm-orc spawns plexus" reads like a loop):

**A. Shared SQLite + `changes_since` cursors.** Both processes open the DB
independently; no process supervises the other.
- For: no subprocess lifecycle; the roadmap's earlier lean; trivially
  avoids process-level circularity.
- Against: the DB schema becomes the de facto API. Unless plexus documents
  the schema as a supported public interface, llm-orc couples to private
  internals and silently breaks on plexus releases; graph traversal,
  evidence-trail assembly, and context identity would be reimplemented in
  Python against raw tables — duplicating exactly the logic plexus exists
  to own. Write contention and invariant bypass on ingest are real risks.

**B. Spawn `plexus mcp` (stdio) as a long-lived child of the serve,
extraction never fired on this path.** RECOMMENDED.
- For: the MCP tool surface is plexus's *supported contract* — versioned,
  tested, and exactly the 18 tools we need. Ingest and queries go through
  plexus's own invariants. One child per serve process (the serve is
  long-lived, so spawn cost amortizes to zero). The play-harness client is
  a working reference.
- Against: subprocess lifecycle to manage (crash → degrade, §Failure
  modes); stdio is per-parent, so N serve processes = N plexus children
  over one substrate — fine per plexus's multi-process shared-SQLite
  design, verify on the rig.

**C. Standing plexus daemon with a socket transport.** Cleanest at
platform scale, but v0.5.0 is stdio-only; requires plexus-side transport
work. Named as the likely platform end state, not the entry point.

**Recommendation: B now, C later if the platform consolidates.** A is
rejected unless the rig check shows plexus explicitly supports external
schema consumers.

### The circularity is broken by role separation, not by topology

The apparent cycle dissolves once the two llm-orc roles are named:

- **llm-orc-as-extractor**: the model serve plexus spawns for semantic
  extraction. A pure model endpoint. It must never be a plexus consumer.
- **llm-orc-as-agentic-serve**: the serving layer behind OpenCode. A
  plexus consumer. It must never be spawned by plexus.

With roles separated the graph is a diamond (client → agentic-serve →
plexus → extractor-serve), not a cycle. Two enforcement mechanisms, both
cheap:

1. **Pre-tagged ingest makes extraction unnecessary on this path**: the
   agentic serve ingests content that already carries tags (§Ingestion),
   so plexus has no reason to invoke extraction for it. (Whether plexus's
   extraction is call-triggered or automatic on ingest: verify on rig; if
   automatic, an ingest flag or config to suppress it is the ask to
   plexus.)
2. **A parentage sentinel**: the agentic serve refuses to start when
   spawned by plexus (env marker set by plexus's extractor invocation),
   stating the role-separation invariant in one place instead of hoping
   configs never cross.

## Ingestion spec

**What ingests** (push model, non-blocking, ADR-010 boundary — source
material, never control flow):

- Per-turn records: `{"text": <turn summary or verbatim prose>, tags,
  chain_name: <session-id>}`, mapping turn/timestamp/speaker to node
  properties and the session to chain grouping. Plexus provenance marks
  are file/line-shaped; don't force them onto turns.
- **The write-history ledger events, explicitly**: shipped writes AND
  rejected asks, with their turn ordinals. This is the load-bearing
  entry — it makes the #133/#134 grounding discipline cross-session (a
  session-B recall about session A anchors on session A's ledger, with
  rejections disclosable because they were recorded).
- Tags come from classify's routing facts, **deterministically and for
  free**: intent, named files, symbols, chain/step, gate verdict
  (shipped/rejected/refused). Zero model calls (doctrine 9). Embeddings
  (plexus optional feature) and extraction-through-llm-orc are the
  held-open upgrades if tag-based retrieval proves thin — the latter runs
  at plexus's layer through its existing extractor relationship, not from
  the serve.

**Properties:** idempotent by turn-id (re-ingest safe, crash-resume safe);
failure is skip-and-log (memory degrades, honesty does not — the serve
never claims what it cannot cite); ingest content derives from the
caller-side record, never from model prose, so it is spoof-safe by the
same argument as the #82 ledger.

## Query spec (lens entry points)

**Where queries fire:** deep-history and cross-session recall/explain
paths, after deterministic in-session selection misses. In-session answers
keep their current deterministic path; plexus is the fallback for what the
session record cannot answer, mirroring #82's two-layer split (structural
floor first, bounded judgment only on low-risk cases).

**How queries are built:** a closed template set — find-by-tag,
traverse-from-node, evidence_trail-for-claim — parameterized only by
charset-checked identifiers from classify (session ids, file stems,
symbols). Never model-authored query text on the seam. This is the same
discipline as the pytest command template and glob patterns, applied to
the substrate.

**How results enter the turn:** as fenced provenance blocks (the existing
block grammar; results are untrusted content like read bodies), each
carrying its `evidence_trail` receipt. **Every cross-session claim in an
answer cites its receipt; a claim with no trail is not made — the turn
refuses honestly instead.** Fail closed, doctrine 9.

## Failure modes and honesty analysis

- **Plexus child crashes / won't start:** the serve degrades to
  session-local memory and says so on cross-session asks ("no
  cross-session memory available this session"), never silently guesses.
- **Partial ingest (earlier crash, skipped turns):** answers cite what is
  present; gaps are disclosed, not interpolated. The `changes_since`
  cursor makes catch-up ingestion cheap on restart.
- **Spoofing:** ingest is caller-side-record-derived (not prose); query
  results render fenced; receipts point into plexus provenance. A forged
  claim would need to forge the substrate, which sits behind the process
  boundary.
- **Schema/tool drift across plexus releases:** topology B confines the
  blast radius to the MCP tool contracts; pin the plexus version in the
  integration test fixture and let the hermetic suite catch drift.

## Validation plan

**Hermetic (ANY environment):** a fake stdio MCP server fixture (crib the
play-harness client for the protocol shapes); round-trip ingest → query →
fenced receipt rendering; every failure mode above as a test (crash →
honest degrade; no-trail → refusal; re-ingest idempotency).

**Rig (exit gate, unchanged from the roadmap):** a fact from session A
answered in session B with an evidence trail attached — plus the dishonest
mirror probe: a session-B question about something session A *rejected*
must disclose the rejection (the #133 class, cross-session).

## Platform trajectory

The sister-project observation is right, and the shape it converges to is
a three-layer platform with narrow, explicit contracts:

| Layer | Project | Contract it exposes |
|---|---|---|
| Substrate: knowledge graph, provenance, cross-session identity | plexus (Rust) | MCP tools (18 today); later a socket transport |
| Engine: orchestration, verification gates, serving | llm-orc | OpenAI-compatible wire + tool_calls; `{requirement, code, tests}` seat contract; MCP (already consumed by plexus) |
| Execution surface: workspace, permission seam | OpenCode / Claude Code / any client | advertised tool list |

The platform claim becomes real when the substrate serves multiple
consumers (plexus's own vision doc names llm-orc's serving layer as "just
another consumer with its own spec/lens" — that sentence *is* the platform
thesis). The discipline that keeps it healthy: layers couple through the
contracts in the table, never through schemas, file layouts, or spawn
relationships. Both cautions in this spec (topology A's schema coupling,
the role-separation invariant) are instances of that one rule.

On the recurring "should llm-orc be Rust" question: assessed separately
(2026-07-17, practitioner discussion) — summary: Rust is the plausible
end-state language for a shipped platform daemon, Python is the right lab
language for the current research-velocity phase, and the migration that
matters is already happening in a language-neutral direction: behavior
moving into the declarative layer (YAML shapes, chain tables, closed
templates) and deterministic components freezing behind contracts. Frozen
components can harden into Rust (plausibly plexus-side crates) when the
roadmap's revisit triggers fire (a measured serve-side bottleneck caching
can't fix, or a deployment target Python can't reach). No greenfield
rewrite is on the critical path; the declarative layer is the insulation
that keeps a future port cheap.

## Verify on the rig (before implementation)

1. Plexus MCP transport options in the current version (stdio only?), and
   multi-child-over-one-substrate behavior (N serves × 1 DB).
2. Extraction semantics: call-triggered or automatic on ingest? If
   automatic, the suppress-flag ask.
3. The ingest tool's actual schema (tags? chain_name? node properties?) —
   the shapes above are from the 2026-07-11 survey and issue #127 text.
4. Whether shared SQLite is a *supported* external interface (would
   reopen topology A) or an internal multi-process detail.
5. Embeddings feature status and cost, for the retrieval-thinness
   upgrade path.
6. `evidence_trail` granularity — what a receipt actually contains and
   whether it renders meaningfully in a fenced block.

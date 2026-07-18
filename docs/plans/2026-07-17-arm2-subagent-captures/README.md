# Arm-2 feasibility probe: Claude Code subagent transcript capture (2026-07-17)

Probe run from a remote Claude Code session (container, no rig): ONE Haiku
4.5 subagent driven through the battery's first two turns as one continuing
conversation against a seeded git fixture, with the shared truth substrate
running caller-side. **Every Arm-2 construct requirement is now verified
feasible in this environment**, and the transcript format is captured from
real data (`probe-2turn-transcript.jsonl` — verbatim, 36 events, two turns
with the continuation boundary between them).

## Findings

1. **Transcript location and shape.** Subagent transcripts land at
   `<claude-projects>/<session-id>/subagents/agent-<id>.jsonl` — JSONL, one
   event per line. Event `type` ∈ {`user`, `assistant`, `attachment`}.
   `message.content` blocks: `thinking` (+signature), `tool_use`
   (`{id, name, input}`), `tool_result` (arrives on user-role events, linked
   by `tool_use_id` and top-level `sourceToolAssistantUUID`), `text`.
   Top-level metadata per event: `agentId`, `sessionId`, `uuid`/`parentUuid`
   chain, `timestamp`, `cwd`, `gitBranch`, `version`, `promptId`/`requestId`,
   `attributionAgent`, `isSidechain`.
2. **Tool names, for the IR normalization map:** `Write`
   `{file_path, content}`, `Bash` `{command, description}`, `Read`
   `{file_path}`. (The roadmap's warning holds: unmapped streams must fail
   loudly — these are the observed names to map.)
3. **Token accounting is richer than OpenCode's.** Every assistant event
   carries `message.model` and full `usage` including explicit
   `cache_creation`/`cache_read` splits (no fresh-token lower-bound caveat
   needed) and an `iterations` array on the closing message. **Adapter trap,
   analogous to the OpenCode callID dedup: multiple JSONL events share one
   `message.id` (streaming increments), so usage must be deduped by
   `message.id`, never summed per event.**
4. **Continuation works and is cheap.** `SendMessage` to the agent id
   resumed the SAME conversation (turn 2 refactored its own turn-1
   `add_todo` while adding `complete_todo` — cross-turn memory demonstrated),
   appended to the SAME transcript file (23 → 36 events), and rode the
   prompt cache (~25K cache-read, ~1K new). One run = one agent id = one
   transcript file. The turn boundary is visible as a new `promptId` on the
   injected user event.
5. **The shared truth substrate runs here.** `capture_truth.sh` is zsh; the
   container has none by default but `apt-get install zsh` works (driver
   setup step). With `TRUTH_PYTEST="uv run pytest"`, turn captures produced
   hashed manifests, suite/seeded rcs, and — notably — **the hidden turn-1
   oracle ran and PASSED against the Haiku deliverable** ("ok: mutated in
   place via add_todo").
6. **The instrument caught the probe's own procedural error.** Turn-1's
   capture was run CONCURRENTLY with turn 2 (probe sloppiness): turn 2's
   write landed inside the capture window, and the post-manifest diff
   flagged it as contamination instead of absorbing it — the round-3
   mutation-hazard hardening working on live data. Driver rule, learned the
   honest way: **captures run strictly between turns, serialized with the
   agent's completion notification, never concurrent.** (This is why
   `truth-01.json` here shows contamination and why its `post_manifest`
   todo.py hash equals `truth-02.json`'s manifest hash.)
7. **`cwd` metadata is the HARNESS repo, not the fixture** — the probe
   steered file operations via absolute paths in the prompt. The adapter
   must derive deliverable paths from `tool_use.input.file_path`, never from
   event `cwd`. Whether per-agent cwd can be set natively is an open driver
   question; the absolute-path instruction is a workable fallback but is a
   declared construct note (the arm's prompt carries a path preamble the
   Arm-0 battery's prompts don't).

## Declared confounds (unchanged from the roadmap, now confirmed applicable)

The subagent inherits the project CLAUDE.md stack, the agent Bash sandbox,
and runs without permission prompts. Add finding 7's path preamble.

## What remains for the Arm-2 driver (next session, REMOTE)

- A driver script: spawn (turn 1) → for each of turns 2–13: send, await
  completion, run `capture_truth.sh N` serialized (finding 6) — with the
  13 battery prompts verbatim from `ladder_battery.sh`.
- The raw→IR adapter over this captured schema (`subagent_adapter.py`
  beside `opencode_adapter.py`), with the `message.id` usage dedup and
  loud-fail on unmapped tool names.
- ≥3 runs per model (Haiku 4.5, Sonnet), J-turns to the independent scorer,
  cells including deaths/unscored published per the frozen rubric.

## Artifacts

- `probe-2turn-transcript.jsonl` — the verbatim two-turn transcript.
- `truth-01.json`, `truth-02.json` — the probe's truth captures (turn 1
  carries the contamination flag discussed in finding 6; kept as the
  evidence for the serialization rule, not as valid run data).

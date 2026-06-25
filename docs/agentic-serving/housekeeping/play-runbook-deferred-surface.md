# PLAY Runbook — Exercising the Deferred Client-Tool Surface

**Goal:** deliberately trigger OpenCode's deferred client tools (`todowrite`, `task`,
`skill`) and observe how the framework handles them. Cycle 7 built and validated the
*file-action subset* (`read`/`write`/`edit`/`bash`); the deferred tools are **carried,
not directed** — grounded-carry (`CarryClientTool`) passes them through verbatim so the
session won't crash, but the reliability machinery is file-shaped and does not apply
(completeness gate mines `name.ext`; content anchor is cross-*file*; delegated generation
maps only to `write`/`edit`). This run discovers what breaks. **Findings are the output,
not pass/fail** (the PLAY stance, backed by the cycle-status Feed-Forward
"skill/task/todowrite surface"). ADR-033 §6c is the design basis: the loop-driver must
handle the full client tool surface, not only the file-action tools the spikes exercised.
See `essays/research-logs/cycle-7-benchmark-tier-comparison.md` for the one axis-2 limit
already known: the cheap stack's cross-reference coherence ceiling is ~12 files (clean at
12, coherence fails 0/3 at 15).

## Confirmed tool surface (Spike π capture)

OpenCode declares 10 tools per request: `bash, edit, glob, grep, read, skill, task,
todowrite, webfetch, write`. **Directed** (file-action machinery applies): `read`,
`write`, `edit`, `bash`. **Carried, undirected** (this run's focus): `skill`, `task`,
`todowrite`. (`glob`/`grep`/`webfetch` are read-only and also carried, not the focus.)

## Pre-flight (both modes)

1. **Seat:** `.llm-orc/config.yaml` → `agentic_serving.orchestrator.model_profile =
   agentic-orchestrator-qwen36-zen` (qwen3.6-plus via OpenCode Go). Already set.
2. **Deps:** `opencode --version` = 1.17.9; `qwen3:8b` + `qwen3:14b` pulled; Zen/Go auth
   live (`llm-orc auth list` shows `openai-compatible/zen`).
3. **Fresh ollama** for clean latency (the marathon-degradation guard): quit + relaunch
   Ollama, warm the models. (`benchmarks/agentic_serving/bench.py:reboot_ollama` does this.)
4. **Start the serve** capturing the log (the turn-decision evidence lives here):
   `uv run llm-orc serve --port 8765 > /tmp/play-serve.log 2>&1 &` then wait for
   `curl -s localhost:8765/v1/models` → 200.
   - Live TUI uses :8765 (the `~/.config/opencode/opencode.json` baseURL). Headless can
     reuse :8765 (sequential, not simultaneous) or a dedicated port for isolation.

## Primary arm — the composition session (the real test)

The point is not whether each tool fires in isolation; it is whether the cheap seat +
framework **composes** OpenCode's full tool surface into a coherent multi-turn session.
This follows the cycle's standing discipline — the live multi-turn run is the PRIMARY
arm; the single-tool probes below are for causal isolation only (per-decision success
does not compose to per-session success). A real session naturally interleaves *directed*
tools (`read`/`write`/`edit`/`bash` — the file-action machinery applies) with *carried*
tools (`todowrite`/`task`/`skill`/`grep`/`glob` — passed through, undirected). The
composition test stresses the **seam** between them.

**Workspace:** seed a small, real multi-file Python project (3–5 files) with a couple of
intentional rough edges — a function with a bug, a missing test, a stale README.

**Prompt (composition):**
> "Improve this project. First explore what's here and lay out a plan (track it as a todo
> list). Then make a concrete improvement: fix the bug in `<module>`, add the missing
> test, and bring the README in line. Run the tests to confirm. Use whatever tools help —
> search, a sub-agent, or a skill — and keep your todo list current as you go."

This invites `grep`/`glob`/`read` (explore) + `todowrite` (plan) + optionally `task`/`skill`
+ `write`/`edit` (implement) + `bash` (verify), across several turns.

**Watch the seam:**
- Does the framework hold **one coherent trajectory** across the directed↔carried
  alternation, or does a `todowrite`/`task`/`skill` turn mid-flow drop the plan or reset
  context?
- Does the session **terminate when the task is actually done** — not when the file-action
  completeness gate happens to read "all files written," and not a zombie that keeps
  re-planning? The gate fires on file deliverables; the *task* includes non-file turns.
- Does the seat **sequence tools sensibly** (explore → plan → implement → verify), or does
  it thrash, skip steps, or never converge?

## Causal-isolation probes (isolate each carried tool — run after the composition session)

> **Observation #0 (for every probe): does the seat even emit the tool?** OpenCode's
> system prompt nudges the agent toward `todowrite`/`task`; whether the qwen3.6-plus seat,
> reproducing that behavior, actually emits the tool_call is the first thing to record. A
> non-emission is itself a finding (the seat doesn't pick up the client's tool-use cues).

> **Termination routing (read before picking prompts).** The completeness gate keys off a
> filename regex over the *task prose* (`_REQUESTED_FILE_RE`, `loop_driver.py`), and
> `todowrite`/`task`/`skill` are not write tools (`_WRITE_TOOLS = {write, edit}`), so they
> never advance the produced set. Consequence: a prompt that names recognized files hits
> the **deterministic** gate and terminates on those files, with the carried tool just
> riding along; only a **file-free** prompt routes termination to the stochastic judge. So
> Probe T (names 5 `.py`/`.md` files) and Probe K (names `summary.py`) actually test
> *carry fidelity inside a deterministically-terminating session*, while Probe S (names no
> file) is the only probe that tests *stochastic termination of non-file work* (the
> central risk). If you want Probe T to exercise the judge too, run a second variant that
> describes its steps by role without naming `.py`/`.md` files.

**Probe T — `todowrite` (multi-step planning state).** Workspace: empty dir.
> "Build a small Python package in this directory in stages, and keep a running todo list
> as you go: (1) `config.py` with a `load_defaults()` returning a settings dict; (2)
> `core.py` that imports `load_defaults` and uses it; (3) `cli.py` with an argparse `main()`
> under a `__main__` guard; (4) `test_core.py`; (5) `README.md`. Track each step in your
> todo list and mark items done as you complete them."

**Probe K — `task` (spawn a sub-agent).** Workspace: seed it first with 2–3 small `.py`
files so the survey has something to find.
> "Use a sub-agent to first survey the Python files already in this directory and
> summarize their public functions and how they relate. Then, based on that summary, write
> a new `summary.py` module with a function that calls the real functions you found."

**Probe S — `skill` (invoke a skill).** Backed: OpenCode enumerates the skills in
`~/.claude/skills/` into its system prompt (Spike π confirmed the 10-tool inventory
including `skill`; the specific skill *count* it surfaces is unverified, so do not rely on
a fixed number such as "16") and loads `~/.claude/CLAUDE.md` too. So capture the live
request's `<available_skills>` block first (see the north-star note below), then pick a
real skill from that block whose description matches the task and let the seat decide to
load it. Workspace: something the skill can act on.
> e.g. "Audit the architecture of this codebase" (→ `codebase-audit`), or "Survey the
> literature on <topic>" (→ `lit-review`).
> Observe whether the qwen3.6-plus seat recognizes the match and emits a `skill`
> tool_call, then whether the framework carries it and the loaded skill's instructions
> flow back coherently across the (now longer) session.
>
> **North-star gap to record:** the capture enumerated the `~/.claude/skills/` set but
> NOT the RDD *plugin* skills (`rdd-build`, `rdd-research`, …, which live in
> `~/.claude/plugins/cache/`) or superpowers. So "run RDD via OpenCode" through the
> `skill` tool does not yet reach the RDD pipeline — the RDD skills would need to be
> exposed to OpenCode (e.g. surfaced under `~/.claude/skills/`). The fresh session should
> capture the LIVE request's `<available_skills>` block to confirm the current list before
> relying on it (skills may have changed since the capture).

## Headless mode (the fresh session runs this)

For each probe, in a fresh per-cell workspace with an `opencode.json` pointing at the serve
(mirror `benchmarks/agentic_serving/runner.py:_write_opencode_config` — provider `llmorc`,
model `agentic`, baseURL `http://127.0.0.1:8765/v1`):

```
opencode run -m llmorc/agentic --format json --dir <ws> "<probe prompt>"  > <ws>.out 2>&1
```

Capture per probe: the serve-log slice (lines added during the run), `<ws>.out` (the
client's tool_call stream + any error), and the workspace contents. The benchmark's
`runner.run_cell` already does workspace + slice capture and can be reused with a
todowrite/task/skill prompt instead of a file-gen cell.

## Live mode (you run this in the TUI)

1. `uv run llm-orc serve --port 8765` (same serve, or restart fresh).
2. Open the OpenCode TUI in a scratch project dir.
3. Select model **"llm-orc agentic"** (provider `llm-orc`).
4. Type each probe prompt; watch the TUI for: a todo panel appearing (T), a sub-agent
   spawning (K), a skill running (S); whether the turn completes or hangs; whether the
   final answer is coherent.
5. Tail `/tmp/play-serve.log` alongside to see the framework's turn decisions.

## Reading termination from the serve log (the decoder)

The driver emits the termination path to `/tmp/play-serve.log` at INFO (the operator sink's
`turn decision:` line plus the driver's own `completeness:` lines). Two lines were added for
this run so every path is greppable: the AS-3 cap fire and the stochastic judge's verdict
(previously the only un-logged paths). Decode the slice:

| Termination path | Log signature |
|---|---|
| Deterministic completeness | `completeness: requested=.. produced=.. remaining=0 verdict=COMPLETE` then `turn decision: … judgment_verdict=COMPLETE action=finish` |
| Stochastic judge | `completeness: no requested set, judge fallback` + `judge fallback: verdict=… text=…` then `turn decision: … judgment_verdict=COMPLETE\|REMAINING action=finish` |
| False-stop / incoherent stall | `turn decision: … judgment_verdict=REMAINING action=finish` (often with `remaining-retry: recovered=false`) |
| AS-3 cap (zombie backstop) | `termination: AS-3 cap reached turn=N` (finish text in the client `.out` is `[Session budget exhausted: turn limit reached. Stopping.]`) |

Carry fidelity is free in the same stream: a deferred-tool turn logs as `turn decision: …
action=todowrite carry_held=true` (the `CarryClientTool` passthrough), so you confirm the
client tool_call survived verbatim at the driver boundary without parsing the JSON stream.
The `--format json` `.out` is only needed for finer per-turn shape, not for termination or
carry fidelity.

## Observation targets (the mechanism-level questions — per the Feed-Forward signal)

These sit behind both the composition seam and the isolation probes — use them to
attribute a composition failure to a specific mechanism. For the composition session and
each probe that fires, record:

- **Termination.** Whether the deterministic gate or the stochastic judge owns the finish
  depends on the prompt (see "Termination routing" above), not on which tools fired. For a
  file-free task the judge owns it: does the loop know when the `todowrite`/`task`/`skill`
  work is "done", or does it mis-judge (premature finish, or a zombie that never finishes)?
  That is the central risk. For a file-naming task the gate terminates on the named files
  while the carried tools ride along, so watch instead for decoupling: does the model keep
  emitting `todowrite` turns after the files are written (a zombie the AS-3 cap then
  catches, now visible as `termination: AS-3 cap reached`)?
- **Coherence without an anchor.** There's no cross-file content anchor for non-file work.
  On a multi-turn `task`/`todowrite` flow, does the trajectory stay coherent across turns,
  or does it drift / repeat / lose the plan?
- **Carry fidelity.** Is the client tool_call passed through verbatim and executed by
  OpenCode, or does the framework mangle/drop it?
- **Delegation.** Generation can't be delegated to a non-`write` tool — does the seat try
  to delegate and fail, or carry correctly?

## What to record

Append findings to `essays/reflections/field-notes.md` (the PLAY artifact), categorized by
feedback destination (missing scenario → DECIDE; interaction gap → interaction-specs; new
question → RESEARCH; challenged assumption → DISCOVER). Each deferred-tool failure mode is
a candidate requirement for the future cycle that *directs* (not just carries) the richer
surface.

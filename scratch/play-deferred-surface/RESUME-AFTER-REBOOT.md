# RESUME — PLAY deferred-surface run (after reboot)

**Status:** Pre-flight done; **blocked** firing sessions by a wedged `opencode run`
client (hangs at session creation every time). Backend is proven healthy. Chosen
remedy: **reboot the machine**, then resume from here. This run executes
`docs/agentic-serving/housekeeping/play-runbook-deferred-surface.md`.

**Seat:** `agentic-orchestrator-qwen36-zen` (hosted qwen3.6-plus on Zen, **PAID** —
user approved running the full PLAY run on it). Config already set in
`.llm-orc/config.yaml` (orchestrator.model_profile). `form_escalation` →
minimax frontier coder is active (rare, ≈cents).

---

## What is CONFIRMED working (do NOT re-debug)

- **llm-orc serve + seat:** direct `curl localhost:8765/v1/chat/completions` returned
  `PONG` cleanly. The serve, qwen3.6-plus seat, and routing all work end-to-end.
- **opencode SERVER** (`opencode serve --port N`): bootstraps in ~2s, listens fine.
- Ollama coder models qwen3:8b + qwen3:14b pulled; warmed cleanly. opencode 1.17.9.

## What was BROKEN (the reason for reboot)

- **`opencode run` (the headless client the PLAY/benchmark drives): wedged.** Hangs at
  bootstrap right after `init`, before `created session` (no chat POST ever reaches the
  serve → bootstrap hangs are FREE, no seat cost). 12+ consecutive hangs; 2 flukes
  worked very early then never again.
- **Ruled out with evidence** (don't repeat): repo/subdir vs /tmp, desktop-app lock
  (killed, still hung), models.dev network (curl 200 ×3, 0.3s), models cache,
  `OPENCODE_FAST_BOOT`, `--print-logs`, `--pure`, HTTP(S)_PROXY (opencode uses Bun
  fetch, ignores proxy), WAL side-files (`opencode.db-wal/-shm` cleared), server +
  `--attach` (client still hung), output piping (`| cat`/`| tail` vs `> file`).
- Conclusion: opencode global state wedged (aggravated by killing hung runs with -9).
  Reboot is the clean slate. The benchmark normally uses plain `opencode run`, so it
  works in a healthy environment.

---

## POST-REBOOT PRE-FLIGHT (do in order)

1. **Ollama:** ensure app running; warm coders:
   `ollama run qwen3:8b ok; ollama run qwen3:14b ok` (or
   `benchmarks/agentic_serving/bench.py:reboot_ollama` if a fresh restart is wanted).
2. **Rotate the stale serve log** then start the serve fresh:
   ```
   PLAY=/Users/nathangreen/Development/eddi-lab/llm-orc/scratch/play-deferred-surface
   mv "$PLAY/logs/play-serve.log" "$PLAY/logs/play-serve.preboot.log" 2>/dev/null
   uv run llm-orc serve --port 8765 > "$PLAY/logs/play-serve.log" 2>&1   # run_in_background
   ```
   Wait for `curl -s -o /dev/null -w '%{http_code}' localhost:8765/v1/models` → 200.
3. **Seat sanity (direct, bypasses opencode):**
   `curl -s -m 60 localhost:8765/v1/chat/completions -H 'content-type: application/json' -d '{"model":"agentic","messages":[{"role":"user","content":"Reply with exactly PONG"}]}'`
   → expect `PONG`.
4. **opencode-run SMOKE GATE (the thing that was broken):** in a fresh `/private/tmp`
   dir with the opencode.json below, run a trivial prompt with a short `gtimeout 90`.
   Confirm a chat POST hits the serve and it returns READY and exits. **Only proceed
   if this passes.** If it hangs again post-reboot, escalate (opencode reinstall/upgrade)
   before burning time.

**Do NOT run the desktop OpenCode.app during headless runs** (it held the project lock
and was the first trigger). Keep it closed.

---

## opencode.json (write into every workspace `--dir`)

```json
{"$schema":"https://opencode.ai/config.json","provider":{"llmorc":{"npm":"@ai-sdk/openai-compatible","name":"llm-orc","options":{"baseURL":"http://127.0.0.1:8765/v1","apiKey":"sk-llmorc-local-dummy"},"models":{"agentic":{"name":"agentic"}}}}}
```

Invocation (mirrors benchmark `runner._run_opencode`):
`gtimeout 2400 opencode run -m llmorc/agentic --format json --dir <WS> "<prompt>"`
Run via `run_in_background` (re-invokes on exit). Record serve-log line count before
each run to slice cleanly afterward. **Note:** if `opencode run` redirected to a plain
`> file` ever wedges again, that was a red herring here — the real issue was the global
wedge; after reboot a plain `> file 2>&1` should be fine (matches the benchmark).
Workspaces may run inside the repo (`scratch/play-deferred-surface/*-ws`) OR in
`/private/tmp` if the repo project still wedges — /tmp uses opencode projectID=global.

---

## RUN PLAN (primary arm first, then probes)

### Composition session (PRIMARY) — seed already at `composition-ws/`
Seed = tinybank project, 3 rough edges (pristine copies retained in
`scratch/play-deferred-surface/composition-ws/`):
- `account.py`: `apply_interest(balance, rate)` returns `balance*rate` (BUG; docstring
  says applying 0.05 to 100 should yield 105.0 → should be `balance*(1+rate)`).
- `test_account.py`: tests deposit + withdraw only; **missing** apply_interest test.
- `README.md`: stale — documents a non-existent `Account.transfer()` and a wrong
  `withdraw(30) # 130` (should be 120).

Prompt (single line):
> Improve this project. First explore what is here and lay out a plan and track it as a
> todo list. Then make a concrete improvement: fix the bug in account.py, add the
> missing test, and bring the README in line. Run the tests to confirm. Use whatever
> tools help - search, a sub-agent, or a skill - and keep your todo list current as you go.

Watch the directed↔carried seam: one coherent trajectory across alternation? sensible
explore→plan→implement→verify? terminates when task is actually done (not on the
file-gate alone, not a zombie)?

### Probes (causal isolation, AFTER composition) — see runbook for full prompts
- **Probe T (todowrite):** empty dir; staged 5-file package (config/core/cli/test/README)
  with running todo list. Names files → deterministic gate; carried todowrite rides along.
- **Probe K (task):** seed 2-3 .py files; sub-agent surveys then writes `summary.py`.
  Names summary.py → deterministic gate.
- **Probe S (skill):** file-free prompt (e.g. "Audit the architecture of this codebase"
  → codebase-audit). **First capture the live request's `<available_skills>` block.**
  File-free → routes to the STOCHASTIC judge (the central risk). North-star gap: RDD
  plugin skills live in `~/.claude/plugins/cache/`, NOT surfaced to opencode.
- Observation #0 for every probe: **does the seat even emit the tool?** Non-emission is
  itself a finding.

### Termination decoder (grep the serve-log slice) — from runbook §"Reading termination"
| Path | Signature |
|---|---|
| Deterministic completeness | `completeness: requested=.. produced=.. remaining=0 verdict=COMPLETE` + `turn decision: … judgment_verdict=COMPLETE action=finish` |
| Stochastic judge | `completeness: no requested set, judge fallback` + `judge fallback: verdict=…` |
| False-stop/stall | `turn decision: … judgment_verdict=REMAINING action=finish` (often `remaining-retry: recovered=false`) |
| AS-3 cap (zombie) | `termination: AS-3 cap reached turn=N` (client .out: `[Session budget exhausted: turn limit reached. Stopping.]`) |
| Carry fidelity | `turn decision: … action=todowrite carry_held=true` (CarryClientTool passthrough) |

---

## Field notes + phase boundary
- Append a NEW dated **Cycle 7 deferred-surface** session to
  `docs/agentic-serving/essays/reflections/field-notes.md` as RAW observations
  (uncategorized during play — gamemaster boundary; categorize only at session close).
- Cross-cutting reflection with practitioner; then categorize by destination
  (DECIDE/DISCOVER/RESEARCH/interaction-specs/SYNTHESIS).
- Phase boundary: dispatch **susceptibility-snapshot-evaluator** with the play→synthesize
  brief (framing-adoption + alternative-engagement signals); output
  `.rdd/audits/audits-7-play.md` (or cycle's audit path).

## Task list snapshot (recreate)
1. Composition session (primary) — in_progress (was blocked on opencode wedge)
2. Decode composition termination + carry fidelity
3. Probe T — todowrite ; 4. Probe K — task ; 5. Probe S — skill
6. Write raw field notes ; 7. Cross-cutting reflection + categorize
8. Susceptibility snapshot dispatch

## Artifacts (retained, repo `scratch/play-deferred-surface/`)
- `composition-ws/` pristine seed + opencode.json
- `probe-{t,k,s}-ws/` empty (seed at run time per runbook)
- `logs/play-serve.log` (pre-reboot serve log — rotate on resume)
- This file. (Per spike-artifact-retention: keep all until corpus close.)

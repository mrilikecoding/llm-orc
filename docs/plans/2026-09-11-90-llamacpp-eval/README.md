# Issue #90 evaluation — would a llama.cpp serving backend speed up iteration on this box?

Date: 2026-09-11. Rig: Apple M2 Pro, 32 GB, macOS (Darwin 25.3.0).
Constraint: a live `llm-orc serve` (:8765) and Ollama (:11434) were running the
lead's OpenCode ladder throughout. Nothing was restarted, killed, or
reconfigured. Every inference block waited for `dd-out/RUNNING` to clear and
held `scratchpad/BENCH-RUNNING` while it ran.

**Verdict: no. #90 would not speed up iteration on this machine, because Ollama
0.31.1 already runs llama.cpp's `llama-server` as its inference process.
Measured generation throughput is within noise (slightly worse on the Homebrew
build). The per-turn wall clock is ~99% token generation, and the cheapest
available lever — `think: false` on the cheap-tier profile — is worth roughly
4x on generated tokens for a two-line YAML change.**

---

## 0. The finding that reframes the issue

`~/.ollama/logs/server.log`, 2026-09-11 16:59:20:

    level=INFO source=server.go:109  msg="using llama-server for model" model=.../blobs/sha256-a3de86cd...
    level=WARN source=server.go:114  msg="requested context size too large for model" num_ctx=262144 n_ctx_train=40960
    level=INFO source=llama_server.go:425 msg="starting llama-server" cmd="/Applications/Ollama.app/Contents/Resources/llama-server
         --model /Users/nathangreen/.ollama/models/blobs/sha256-a3de86cd...
         --port 65059 --host 127.0.0.1 --no-webui --offline
         -c 40960 -np 1 --log-verbosity 4 --no-log-prefix --no-log-timestamps
         --no-jinja --chat-template chatml --no-mmap --flash-attn auto
         -b 512 -ub 512 --context-shift --keep 4"

Ollama 0.31.1 on this box is a **supervisor around `llama-server`**. It owns:
model resolution from GGUF blobs, memory prediction, eviction, Metal offload
selection, flash-attention selection, the chat template (`--no-jinja
--chat-template chatml`, then Ollama applies its own Go template — see
`msg="template selection" ... selected=go_template go_template="[completion tools thinking]"`),
tool-call parsing, and the `think` channel.

    $ ollama --version            -> ollama version is 0.31.1
    $ /Applications/Ollama.app/Contents/Resources/llama-server --version
                                  -> version: 1 (8c146a836)   built with AppleClang 21.0.0 for Darwin arm64
    $ /opt/homebrew/bin/llama-server --version
                                  -> version: 9850 (4f31eedb0) built with AppleClang 21.0.0 for Darwin arm64
    $ brew list --versions llama.cpp
                                  -> llama.cpp 9850
    $ which llama-server llama-cli
                                  -> /opt/homebrew/bin/llama-server
                                  -> /opt/homebrew/bin/llama-cli

So #90's real content is **dependency ownership and bootstrapping**, which is
what the issue text actually says. Its performance premise is void at this
Ollama version.

---

## 1. Where does a serving turn's wall clock go today?

### 1a. Per-node timings are NOT recorded in the turn trace

`src/llm_orc/web/serving/turn_trace.py` (511 lines) records no duration
anywhere. Grep for `duration|elapsed|perf_counter|monotonic` returns only prose
and `*_duration_ns` *usage* keys lifted from Ollama, never a node timing.
A trace record's keys:

    TOP KEYS: ['ensemble', 'execution_order', 'nodes', 'chain_plan']
    NODE KEYS: {'response', 'seat', 'node', 'status'}

Per-node entries carry `prompt_eval_count`/`eval_count` only when
`metadata.usage.agents` happened to carry them; across the last 8 records only
one node (a `decide`) had any (`738/8`). 678 trace records total; usage is not a
reliable timing source and there is no timing source at all.

**The breakdown below is reconstructed from the Ollama server log**, which
carries GIN request durations plus llama.cpp slot timings (`prompt eval time`,
`eval time`, `cached n_tokens`, `n_ctx_slot`).

### 1b. Method

`~/.ollama/logs/server.log` is 77 MB / 924,438 lines spanning 2026-07-02 ->
2026-09-11. Today's window was sliced to `90-eval/today.log` (2109 lines) and
parsed for: `starting llama-server`, `llama-server started in Xs`,
`predicted to exceed available memory, evicting`, `general.name`,
`new prompt, n_ctx_slot=..., task.n_tokens=...`, `cached n_tokens=...`,
`prompt eval time`, `eval time`, `[GIN] ... POST "/api/chat"`.

Caveat: GIN timestamps are truncated to the second, so inter-request gaps carry
+/-1 s each.

### 1c. Every `/api/chat` the serve issued today

| time | POST s | n_tokens | prefix reused | prefill s | prefill tok | prefill t/s | gen s | gen tok | gen t/s | other s |
|---|---|---|---|---|---|---|---|---|---|---|
| 16:54:17 | 4.68 | 29 | 0 | 0.03 | 29 | 1112 | 1.65 | 272 | 164.4 | 3.00 |
| 16:54:23 | 5.74 | 136 | 8 | 0.04 | 128 | 2916 | 5.53 | 846 | 153.1 | 0.17 |
| 16:54:26 | 8.77 | 136 | 8 | 0.04 | 128 | 3378 | 2.96 | 462 | 156.1 | 5.77 |
| 16:59:32 | 11.94 | 637 | 0 | 2.05 | 637 | 310.1 | 5.62 | 169 | 30.04 | 4.26 |
| 16:59:39 | 6.05 | 752 | 623 | 0.54 | 129 | 238.2 | 5.38 | 161 | 29.95 | 0.13 |
| 17:00:30 | 5.83 | 674 | 578 | 0.34 | 96 | 279.0 | 5.35 | 161 | 30.08 | 0.13 |
| 17:01:10 | 39.59 | 2177 | 4 | 7.21 | 2173 | 301.3 | 32.22 | 914 | 28.37 | 0.16 |
| 17:01:54 | 44.83 | 2647 | 4 | 8.85 | 2643 | 298.7 | 35.76 | 998 | 27.91 | 0.23 |
| 17:02:24 | 29.08 | 2729 | 3 | 9.23 | 2726 | 295.3 | 19.58 | 548 | 27.99 | 0.27 |
| 17:02:30 | 5.97 | 705 | 660 | 0.26 | 45 | 175.7 | 5.42 | 161 | 29.69 | 0.29 |
| 17:03:53 | 82.00 | 2259 | 4 | 7.54 | 2255 | 299.0 | 74.87 | 2061 | 27.53 | -0.41 |
| 17:04:49 | 55.76 | 2805 | 4 | 9.58 | 2801 | 292.5 | 45.81 | 1230 | 26.85 | 0.38 |
| 17:05:42 | 53.03 | 2893 | 3 | 9.91 | 2890 | 291.8 | 42.73 | 1175 | 27.50 | 0.39 |
| 17:06:58 | 8.05 | 588 | 292 | 1.10 | 296 | 269.6 | 6.61 | 203 | 30.73 | 0.34 |
| 17:07:06 | 7.03 | 707 | 574 | 0.53 | 133 | 251.6 | 6.38 | 192 | 30.08 | 0.12 |

The first three rows (16:54, ~155 t/s) are `qwen3:0.6b` (`agentic-summarizer` ->
result-summarizer / calibration-checker) at serve start-up. Everything from
16:59:32 on is `qwen3:8b`.

### 1d. Breakdown for the heavy build turn (turn 02, 17:00:24 -> 17:05:42)

8 back-to-back `/api/chat` calls, no overlap (runner is `-np 1`; the log shows
`all slots are idle` between each).

| component | seconds | share |
|---|---|---|
| **token generation** | **261.7** | **82.3 %** |
| **prompt prefill** | **52.9** | **16.6 %** |
| Ollama-side other (tokenize, template, queue) | 1.4 | 0.4 % |
| **total Ollama /api/chat** | **316.1** | **99.4 %** |
| llm-orc deterministic nodes + subprocess pytest + HTTP + OpenCode tool loop | **~2** (<=8 upper bound at +/-1 s/gap) | **0.6 %** (<=2.5 %) |
| **turn wall clock** | **~318** | 100 % |

Generated: **7,248 tokens**. Prefilled: **15,629 tokens**.

Three turn shapes:

| turn | wall | model load | inference | other |
|---|---|---|---|---|
| turn 01 (cold, small) 16:59:20->16:59:39 | ~19 s | 4.5 s (eviction + 3.79 s load) | ~13.6 s | ~1 s |
| turn 02 (build, warm) 17:00:24->17:05:42 | ~318 s | 0 | 316.1 s | ~2 s |
| turn 03 (warm, small) 17:06:50->17:07:06 | ~16 s | 0 | 15.1 s | ~1 s |

**Deterministic nodes and subprocess overhead are not the problem.** Under 1 %
of a build turn.

### 1e. Model swapping: the prime suspect does not fire

Today's log contains exactly **two** model loads and **one** eviction:

    16:54:12  starting llama-server  qwen3:0.6b   -> "llama-server started in 2.78 seconds"
    16:59:20  msg="llama-server model predicted to exceed available memory, evicting"
              predicted="10.5 GiB" predicted_num_ctx=40960 num_batch=512
              available="4.9 GiB" gpu_free="19.9 GiB" system_free="4.9 GiB" system_limited=true
    16:59:20  starting llama-server  qwen3:8b     -> "llama-server started in 3.79 seconds"

Zero swaps during the ladder itself. Cost of the one swap: **~4.5 s** (eviction
+ 3.79 s load) — 1.4 % of one build turn, 0 % of the rest.

Reason: **nothing in the live serving path uses a second model.** All 11
model-backed agents reachable from `serving.yaml` resolve to
`agentic-tier-cheap-general` (`qwen3:8b`). `qwen3:14b`
(`agentic-tier-escalated-general`) and `deepseek-r1:8b`
(`agentic-tier-escalated-reasoning`) appear in the serving ensembles **only in
description prose**, never in a `model_profile:`. `qwen3:1.7b`
(`agentic-tier-cheap-summary`) is reachable only via `text-summarizer`, which
the build chain never dispatches. `qwen3:0.6b` (`agentic-summarizer`) backs
`agentic-result-summarizer` / `agentic-calibration-checker`, which
`output_substrate: artifact` skips for substrate-routed dispatches (AS-7).

The "model swapping on a 32 GB box" hypothesis is **measured and rejected for
the current build chain.** It becomes real only if escalation or the
result-summarizer starts firing; cost would be ~4.5 s per swap, and the fix is
cheaper than #90 (3.4).

### 1f. Where the prefill seconds actually go

The `prefix reused` column is the one that matters:

- small calls (588-752 tokens): **292-660 tokens reused** — cache hits.
- big seat calls (2,177-2,893 tokens): **3-4 tokens reused** — full re-prefill,
  7.2-9.9 s each, **52.3 s across one turn**.

Not a backend defect. Ollama's prompt cache works perfectly with an identical
prefix — measured in 2: 3,204 tokens re-prefilled in **0.04 s** on an exact
repeat. The 3-4-token reuse is a *prompt composition* fact:
`OllamaModel.generate_response` sends
`[{"role":"system", role_prompt}, {"role":"user", message}]`
(`src/llm_orc/models/ollama.py:83-89`), and each of the ~4 agents in a build
round has a **different** system prompt, so the shared prefix ends at the
chat-template header (~4 tokens). llama-server's prompt cache holds 6 prompts /
3.15 GiB (limit 8 GiB), LCP-matched — it tries (`found better prompt with
f_keep = 0.338, sim = 0.497`), but there is nothing to find.

---

## 2. Raw inference: Ollama vs Homebrew llama-server, same GGUF

llama.cpp **is** installed (`/opt/homebrew/bin/llama-server`, b9850), so this
was measured, not estimated. Nothing was installed for this evaluation.

GGUF: the blob Ollama already holds for `qwen3:8b` —
`~/.ollama/models/blobs/sha256-a3de86cd1c132c822487ededd47a324c50491393e6565cd14bafa40d0b8e686f`
(5,225,374,496 bytes, Q4_K_M), from the manifest
`~/.ollama/models/manifests/registry.ollama.ai/library/qwen3/8b`.

Context size: Ollama runs the seat at `-c 40960` (`OLLAMA_CONTEXT_LENGTH=262144`
in the desktop app's daemon environment, clamped to `n_ctx_train=40960`; **no
llm-orc seat sets `num_ctx`** — confirmed by grep over `src/` and by
`turn_trace.py:118,302`). The llama-server arm used `-c 8192` deliberately, to
avoid a second 5.9 GiB KV allocation next to the lead's live 11 GB runner.
Context size does not affect throughput at these prompt lengths.

    /opt/homebrew/bin/llama-server --model <blob> --port 8089 --host 127.0.0.1 \
      -c 8192 -ngl 999 --flash-attn on -np 1 --no-webui
    start-to-ready: 9.55 s (mmap, cold page cache)

Arms: A = short prompt (26 tok) / 300 generated; B = long prompt (3,204 tok) /
32 generated; C = repeat of A / 64 generated. Two repeats each, temperature 0,
seed fixed. Ollama side used `/api/generate` with `"raw": true` and
`"stream": false` (raw bypasses the Go chat template, matching llama-server's
`/completion`). Script `90-eval/bench.py`; raw results in
`90-eval/bench-ollama.json`, `90-eval/bench-llamacpp.json`.

| arm | backend | wall | prefill tok / s / t/s | gen tok / s / **t/s** |
|---|---|---|---|---|
| A short, gen 300, r1 | ollama | 10.34 | 26 / 0.19 / 135.8 | 300 / 9.86 / **30.43** |
| A short, gen 300, r1 | llama.cpp | 10.66 | 26 / 0.16 / 165.1 | 300 / 10.47 / **28.66** |
| A short, gen 300, r2 | ollama | 9.98 | 26 / 0.04 / 690.7 | 300 / 9.80 / **30.62** |
| A short, gen 300, r2 | llama.cpp | 10.38 | 1 / 0.04 / 24.6 | 300 / 10.31 / **29.10** |
| B long 3.2k, gen 32, r1 | ollama | 12.15 | 3204 / 10.91 / **293.7** | 32 / 1.10 / 29.04 |
| B long 3.2k, gen 32, r1 | llama.cpp | 12.14 | 3204 / 10.99 / **291.4** | 32 / 1.11 / 28.82 |
| B long 3.2k, gen 32, r2 | ollama | **1.27** | 3204 / 0.04 / 85167 | 32 / 1.10 / 29.20 |
| B long 3.2k, gen 32, r2 | llama.cpp | **1.20** | 1 / 0.04 / 26.0 | 32 / 1.15 / 27.73 |
| C repeat A, gen 64, r1 | ollama | 2.48 | 26 / 0.13 / 201.1 | 64 / 2.09 / **30.68** |
| C repeat A, gen 64, r1 | llama.cpp | 2.58 | 26 / 0.15 / 178.8 | 64 / 2.16 / **29.67** |
| C repeat A, gen 64, r2 | ollama | 2.24 | 26 / 0.04 / 695.1 | 64 / 2.07 / **30.92** |
| C repeat A, gen 64, r2 | llama.cpp | 2.15 | 1 / 0.03 / 29.6 | 64 / 2.12 / **30.24** |

| metric | Ollama 0.31.1 | Homebrew llama.cpp b9850 | delta |
|---|---|---|---|
| generation, qwen3:8b Q4_K_M | 29.0 - 30.9 t/s | 27.7 - 30.2 t/s | **-2 % (llama.cpp slower)** |
| prefill, 3.2k prompt | 293.7 t/s | 291.4 t/s | -0.8 % |
| prompt-cache hit on exact repeat | 3,204 tok in 0.04 s | 3,204 tok in 0.04 s | tie |
| cold model load | 3.79 s (`--no-mmap`) | 9.55 s process start-to-ready (mmap) | Ollama faster |

Both report reused-prefix differently (llama.cpp reports `prompt_n = 1` on a
full cache hit, Ollama reports the full count), but the wall clock is the same.

**Zero raw-inference win from #90 on this box.** Same `llama-server`, same Metal
kernels, same flash-attention path (`--flash-attn auto` already on in Ollama's
runner), same prompt cache. Remaining differences are build-revision noise.

Cleanup verified: `pgrep -fl llama-server | grep -v Ollama.app` -> none;
`lsof -iTCP:8089` -> empty; `ollama ps` unchanged (`qwen3:8b, 11 GB, 100% GPU`);
`git status --short` -> clean.

---

## 3. Cheaper alternatives

### 3.1 `think: false` on `agentic-tier-cheap-general` — MEASURED, biggest lever

`OllamaModel.generate_response` sends `think` only when present in `options`
(`src/llm_orc/models/ollama.py:73-92`). Audit of every model-backed agent in
the serving ensembles:

| file | agent | profile | `think` |
|---|---|---|---|
| adequacy-judge.yaml | judge | cheap-general | **false** |
| explainer.yaml | explainer | cheap-general | **false** |
| serving.yaml | decide | cheap-general | **false** |
| serving.yaml | pick | cheap-general | **false** |
| test-writer.yaml | test_writer | cheap-general | **false** |
| **code-generator.yaml** | **coder** | cheap-general | **UNSET** |
| **code-generator.yaml** | **critic** | cheap-general | **UNSET** |
| **code-generator.yaml** | **synthesizer** | cheap-general | **UNSET** |
| argument-mapper.yaml | mapper | cheap-general | UNSET |
| claim-extractor.yaml | extractor | cheap-general | UNSET |
| prose-improver.yaml | improver | cheap-general | UNSET |
| text-summarizer.yaml | summarizer | cheap-summary | UNSET |
| agentic-result-summarizer.yaml | summarizer | agentic-summarizer | UNSET |
| agentic-calibration-checker.yaml | checker | agentic-summarizer | UNSET |

`code-generator` is the code-writing half of `build-gated-round` (`code_writer:
ensemble: code-generator`), so **3 of the 4 model calls in every gated build
round run with thinking on.**

Measured, same prompt (coder system prompt + a realistic `todo/cli.py` task),
against the live qwen3:8b, temperature 0, seed 11 (`90-eval/think_bench.py`,
`90-eval/think_default.py`):

| arm | wall | prefill tok | generated tok | gen s | thinking chars | answer chars |
|---|---|---|---|---|---|---|
| `think: false` | **12.48 s** | 134 | **356** | 11.71 | 0 | 1,499 |
| `think: true` | **49.25 s** | 128 | **1,459** | 48.87 | 4,554 | 1,726 |
| **`think` key absent** (what llm-orc sends today) | **49.41 s** | 128 | **1,459** | 49.18 | 4,554 | 1,726 |

Two things proven: omitting `think` is **identical** to `think: true` on Ollama
0.31.1 for qwen3, and thinking costs **4.1x the generated tokens and 4.0x the
wall clock** for a 15 % longer answer.

Applying the measured 4.1x to turn 02's three largest generations (2,061 +
1,230 + 1,175 = 4,466 tok, 163.4 s — the shape of coder/critic/synthesizer):
those would drop to ~1,090 tok ~= 40 s.

**Estimated saving: ~123 s off a 318 s build turn (~39 %).** At a conservative
2.5x it is ~98 s (~31 %). Labelled an estimate: the 4.1x ratio is measured, the
attribution of those three generations to the three code-generator agents is
inferred from the pipeline shape, not from a per-node timing (which does not
exist — 1a).

**Implementation: two lines.** `model_factory._merge_options`
(`src/llm_orc/core/models/model_factory.py:444-451`) merges profile options
under agent options (agent wins), so adding

    options:
      think: false

to `.llm-orc/profiles/agentic-tier-cheap-general.yaml` turns thinking off for
all 11 cheap-general agents at once and leaves the 5 existing agent-level
`think: false` declarations untouched.

**Risk, and it is real:** thinking off may lower first-pass code quality and
push more turns into a second gated round, which costs 4 more model calls. The
accept-rate instrument has to gate this, not the stopwatch. Safest first step:
`think: false` on `code-generator`'s **critic** and **synthesizer** only (review
and integration, not generation), measure accept rate, then consider `coder`.

### 3.2 Reduce model calls per round — reasoned, second-biggest

A gated build round is 4 model calls: `test_writer` (test-writer) +
`coder`/`critic`/`synthesizer` (code-generator), and `build-gated` allows
`max_iterations: 2`, so **up to 8 model calls per build seat dispatch**, plus
the chain's other steps (`chain_plan` shows `step_index` up to 3, so one
OpenCode turn can be several serve requests).

`critic` and `synthesizer` are `depends_on`-chained behind `coder`, strictly
sequential; together they contributed roughly two thirds of turn 02's
generation. Dropping `synthesizer` (the coder's output already passes a
deterministic executor + adequacy judge + accept gate) removes 1 of 4 calls.

**Estimated saving: 30-60 s/turn.** Effort: a design change with a quality
argument, not a config edit.

### 3.3 Stable prompt prefix for cache reuse — reasoned from measurement

Ceiling and shortfall both in 1f: big seat calls reuse 3-4 tokens and pay 52.3 s
of prefill per turn; an exact-prefix repeat costs 0.04 s. Restructuring the seat
prompt so the **shared, growing conversation context comes first** and the
agent-specific instruction follows would let calls 2..N within a turn (and turn
N+1's first call) hit the cache.

**Estimated saving: ~30-35 s off a 318 s turn (~10 %)**, if 5 of 6 big calls
reuse a ~2,000-token prefix. Caveats: llama-server's cache holds only 6 prompts
(currently 3.15 GiB of an 8 GiB limit) with one slot (`-np 1`), so a 7-call turn
cycling 4+ distinct agents will still evict; and `OllamaModel` puts
`role_prompt` in the `system` message, so this needs a change to how seats
compose prompts, not just YAML.

The naive version does **not** work: the per-agent system prompt is only ~80
tokens (measured: 134 prefill tokens for system + a 54-token task), worth ~0.3 s.
The win exists only if the *big* context block is shared.

### 3.4 Right-size `num_ctx` — measured memory, insurance value only

No seat sets `num_ctx` (grep over `src/`; `turn_trace.py:118,302` says so). The
daemon inherits `OLLAMA_CONTEXT_LENGTH=262144` from the Ollama.app environment
(`ps eww` on the daemon), clamped to 40,960. Result:

    $ ollama ps
    NAME        ID              SIZE     PROCESSOR    UNTIL
    qwen3:8b    500a1f067a9f    11 GB    100% GPU     3 minutes from now

11 GB = 5.2 GB weights + ~5.9 GiB KV. (qwen3:8b: 36 layers, 8 KV heads x 128
dim, f16 -> 2 x 8 x 128 x 2 B x 36 = 147 KiB/token; x 40,960 = 5.9 GiB. The
arithmetic reproduces the reported 11 GB and Ollama's own `predicted="10.5 GiB"
predicted_num_ctx=40960`.)

| num_ctx | KV (f16) | KV (q8_0) | resident (f16) | resident (q8_0) |
|---|---|---|---|---|
| 40,960 (today) | 5.9 GiB | 2.95 GiB | **11.1 GB** | 8.2 GB |
| 16,384 | 2.35 GiB | 1.18 GiB | 7.6 GB | 6.4 GB |
| 8,192 | 1.18 GiB | 0.59 GiB | 6.4 GB | 5.8 GB |

Co-residency answer: **8b + 14b + 1.7b cannot all stay resident today.** At
40,960 ctx that is ~11 + ~15 + ~3.5 = ~29.5 GB, and Ollama's scheduler is
bounded not by GPU (`gpu_free="19.9 GiB"`) but by `system_free`, which the log
shows at **4.9-9.4 GiB** with the lead's session running
(`system_limited=true`). Even 8b + 1.7b needs num_ctx cut to ~16k to fit under a
9 GiB system-free ceiling.

**Value today: ~0 s/turn** (no swaps happen — 1e). **Value as insurance:**
~4.5 s per avoided swap once escalation or the result-summarizer fires. Effort:
one line in the profile (`options: {num_ctx: 16384}`), the same passthrough as
3.1. Do it alongside 3.1, not for its own sake. Secondary benefit: less macOS
memory pressure with 4.9 GiB free.

### 3.5 `OLLAMA_KEEP_ALIVE` / per-request `keep_alive` — measured, small

`grep -rn keep_alive src/` -> **nothing**. llm-orc never sends `keep_alive`, so
Ollama's 5-minute default applies (`ollama ps` -> `UNTIL 3 minutes from now`).
During the ladder, turns were under 5 minutes apart and **no idle unload
occurred**. Cost when it does: **3.79 s measured** (`llama-server started in
3.79 seconds`), plus eviction.

**Saving: 3.79 s per >5-minute idle gap, 0 s otherwise.** Effort: trivial.
Worth doing, worth almost nothing.

### 3.6 `OLLAMA_FLASH_ATTENTION` — already on, 0 s

The runner command line already carries `--flash-attn auto`, resolved on for the
8b (Metal). Nothing to gain.

### 3.7 `OLLAMA_NUM_PARALLEL` / continuous batching — 0 s for this pipeline

The runner is `-np 1`. Raising it enables continuous batching for concurrent
same-model agents — topology C in the scoping doc. But the measured request
timeline shows **no two `/api/chat` spans overlapping anywhere today**, and the
pipeline explains why: `build-gated-round` is `test_writer -> code_writer ->
gather -> executor -> judge -> accept_gate -> envelope`, and inside
`code-generator` it is `coder -> critic -> synthesizer`, every edge a
`depends_on`. There is no same-model concurrency to batch.

**Saving: 0 s until the pipeline gains parallel same-model agents.** This also
retires the scoping doc's `dd-seat-code-verified` motivation for topology C:
that ensemble's two same-model agents are not what the serving chain runs.

### 3.8 `OLLAMA_KV_CACHE_TYPE=q8_0` — 0 s, memory only

Halves KV (table in 3.4). Buys headroom, not latency, and only matters once
co-residency matters. Can slightly reduce quality. Skip for now.

### Ranked by expected seconds saved per unit of effort

| rank | change | saving per 318 s build turn | effort |
|---|---|---|---|
| 1 | `think: false` on `agentic-tier-cheap-general` (start with critic + synthesizer) | **~98-123 s (31-39 %)** — est. from a **measured 4.1x** | 2 lines of YAML + an accept-rate check |
| 2 | drop `synthesizer` from `code-generator` (1 of 4 calls per round) | ~30-60 s (est.) | small design change, quality argument |
| 3 | shared-prefix prompt composition so the cache hits | ~30-35 s (est., ceiling 52 s) | medium — changes seat prompt composition |
| 4 | `num_ctx: 16384` on the cheap-tier profile | ~0 s now; ~4.5 s/swap later; frees 3.5 GB | 1 line |
| 5 | per-request `keep_alive` | 3.79 s per >5 min idle gap | 1 line |
| 6 | `OLLAMA_KV_CACHE_TYPE=q8_0` | 0 s (memory only) | daemon env |
| 7 | `OLLAMA_FLASH_ATTENTION` | 0 s — already on | — |
| 8 | `OLLAMA_NUM_PARALLEL` > 1 | 0 s — nothing runs concurrently | — |
| 9 | **#90, llama.cpp backend** | **0 s (measured -2 % on generation)** | backend rewrite |

---

## 4. Verdict

**Would #90 speed iteration? No.**

| scenario | seconds per build turn | basis |
|---|---|---|
| today | **~318 s** (316.1 s Ollama, ~2 s everything else) | measured, turn 02 |
| under llama.cpp (#90) | **~316-322 s** | measured throughput parity, -2 % on generation, applied to the same token counts |
| under the best cheap alternative (`think: false`) | **~195-220 s** | estimate from a measured 4.1x generation-token ratio |
| `think: false` + prefix-cache fix + drop synthesizer | **~140-170 s** | stacked estimates |

The time is elsewhere, and it is not subtle: **82 % of a serving turn is
qwen3:8b emitting tokens at 27-30 t/s, and roughly two-thirds of those tokens
are reasoning traces the pipeline never reads**, emitted because three agents in
`code-generator` omit `think` and Ollama's default for qwen3 is thinking-on. A
backend swap cannot touch that. A two-line profile edit can.

The 32 GB rig is not the constraint here either. GPU had 19.9 GiB free during
the ladder; the binding number was `system_free`, and nothing swapped.

### Risks #90's own scoping recorded, confirmed

1. **Thinking-off wiring.** Confirmed and now quantified: worth 4.1x, and
   `OpenAICompatibleModel` has no equivalent of `ollama.py`'s
   lift-`think`-into-the-request. Shipping #90 without it would make turns ~4x
   slower on the agents that currently do set `think: false`, unless the seats
   move to a per-model `/no_think`-style convention. Largest regression risk in
   the issue.
2. **Model lifecycle ownership.** Confirmed by the log: Ollama is actively doing
   memory prediction (`predicted="10.5 GiB" ... system_free="4.9 GiB"
   system_limited=true`) and eviction. On a box with 4.9 GiB free, a
   fleet-of-servers topology has no scheduler and would thrash.

### New risks found

3. **The premise is stale.** Ollama 0.31.1 *is* `llama-server`. #90's framing
   ("drop the Ollama process dependency") is a packaging argument, not a
   performance one, and the issue body should say so before anyone sizes it.
4. **Chat templating and tool-call parsing are Ollama's, not llama.cpp's.**
   Ollama launches the runner with `--no-jinja --chat-template chatml` and then
   applies its own Go template (`selected=go_template
   go_template="[completion tools thinking]"`), which is where tool calls and
   the `thinking` channel are parsed. Owning that per model family is a much
   larger lift than the scoping doc's "small code change to
   `OpenAICompatibleModel`" — it is the actual cost of #90.
5. **Telemetry regression is load-bearing for a shipped feature.** The scoping
   doc flags it; what it does not say is that `turn_trace.py`'s truncation
   backstop (#151, #145) reads Ollama's raw `prompt_eval_count`. Moving to
   `openai-compatible` drops it unless `OpenAICompatibleModel` learns to read
   llama-server's `timings`/`usage` first. #90 has a hard dependency on that.
6. **Per-node timings do not exist.** Any performance claim about the serving
   pipeline is currently unfalsifiable from llm-orc's own instruments — this
   evaluation had to reconstruct the breakdown from a third-party log. If
   iteration speed is going to be steered, `turn_trace.py` should record a
   per-node elapsed. Smaller, higher-value than #90, and a precondition for
   measuring any of section 3's alternatives honestly.
7. **`num_ctx` is unmanaged.** The seats inherit whatever
   `OLLAMA_CONTEXT_LENGTH` the operator's desktop app happens to carry (here
   262,144 -> clamped to 40,960 -> 11 GB resident for an 8b). An
   environment-dependent memory footprint the project neither controls nor
   records. It bit once today (the 16:59:20 eviction).

### Recommendation

Keep #90 in the backlog as the **packaging/bootstrap** issue it actually is
("llm-orc can fetch a default GGUF and own the model lifecycle"), and strike the
performance motivation from it. Do not schedule it for iteration speed.

For iteration speed, in order: (a) add per-node elapsed to `turn_trace.py` so
the next claim is measurable from inside; (b) `think: false` on the cheap-tier
profile, gated on the accept-rate instrument; (c) revisit prompt composition for
cache reuse.

---

## Appendix: files produced (all under `scratchpad/90-eval/`)

- `today.log` — 2026-09-11 slice of `~/.ollama/logs/server.log`
- `timeline.txt` — parsed event timeline (loads, evictions, prompts, cache hits)
- `bench.py`, `bench-ollama.json`, `bench-llamacpp.json` — section 2 backend A/B
- `llama-server.log` — the temporary Homebrew server's own log
- `think_bench.py`, `think-True.json`, `think-False.json` — section 3.1 think on/off
- `think_default.py` — section 3.1 proof that omitting `think` == `think: true`

No tracked file in the repo was modified (`git status --short` clean). The
temporary `llama-server` on :8089 was stopped and verified gone. `ollama ps` was
unchanged before and after every benchmark block.

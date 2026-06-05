# Cycle 7 loop-back BUILD — WP-LB-C real-OpenCode validation

**Date:** 2026-06-02
**Phase:** BUILD (loop-back), WP-LB-C close
**Cost:** $0 (local Ollama only)
**Artifacts retained:** `scratch/wp-lb-c-opencode-validation/` (per spike-artifact-retention)

## Why this run happened

The practitioner's "don't fly blind" directive (cycle-status §Feed-Forward
WP-LB-D #3; memory `validate-against-real-client-not-harness`) requires WP-LB-C
to be validated against a *real* client at the earliest runnable point, not
against harness fixtures. The cycle's own WP-A scar — "the architecture being
kept was never run through the client it exists to serve" — is the failure mode
to avoid repeating. WP-LB-C is the earliest point a parity round-trip becomes
runnable (the terminal emits `tool_calls` + participates in the loop), so the
validation ran at WP-LB-C close.

## Setup (all local, $0)

- **Client:** OpenCode 1.15.5, headless (`opencode run "<task>" --format json --dir <ws> -m llmorc/agentic`), provider `@ai-sdk/openai-compatible` pointed at `http://127.0.0.1:8765/v1`.
- **Server:** the real `llm-orc serve` (not a stand-in) with the WP-LB-C terminal.
- **Seat-filler:** `agentic-orchestrator-offline-tools` (qwen3:14b via the OpenAI-compatible Ollama adapter, `supports_tool_calling = True`).
- **Capability ensemble available:** `code-generator` (`topaz_skill: code_generation`, `output_substrate: artifact`).
- **Task:** "create hello.py that prints hello world."

## Result — the parity mechanism works against the real client

A real headless OpenCode session drove a real multi-turn agentic loop against
the endpoint, received `finish_reason: "tool_calls"`, **executed the `write`
itself**, `hello.py` landed (`print("hello world")`), and the loop closed
(`step-finish`, reason `stop`, OpenCode text "File created successfully…").
Because the server never writes to a client workspace path (FC-48), the file's
presence proves OpenCode applied the synthesized tool call. This validates
FC-47 (tool-call terminal), FC-48 (no server-side write), and FC-50 (loop
participation) **against the real client**, not a fixture. The WP-A scar is
closed at the real-client layer for the terminal/loop-participation mechanism.

## Findings

### Finding A (fixed) — seat-filler `base_url` was dropped

The first run 401'd against `api.openai.com`. `_resolve_seat_filler` called
`load_model(model, provider)` without the profile's `base_url`, so a local
OpenAI-compatible seat-filler defaulted to OpenAI's endpoint. The ensemble path
already threads `base_url` via `ModelFactory.from_agent_config`; only the
bespoke seat-filler resolution dropped it. The harness never caught this — it
overrides `_resolve_seat_filler` to inject a fake. **Fixed + tested (commit
`8e63cee`).** This is a direct dividend of running against a real model.

### Finding B (critical, open) — the seat-filler is never offered `invoke_ensemble`

`LoopDriver.decide` passes **only the client's tools** (`write`/`edit`/`bash`/
`read`, from `context.tools`) to the seat-filler. Nothing augments that list
with `invoke_ensemble`. So a real seat-filler **cannot** emit the delegation
call the driver branches on (`action.name == "invoke_ensemble"`); it can only
emit a literal client tool call (grounded carry) or finish.

**Evidence:** the OpenCode run produced **zero ensemble dispatches** (serve log:
0 occurrences of `invoke_ensemble` / `dispatch start` / `tier selection` /
`code-generator`) and **zero artifacts** written to `agentic-sessions/`. The
cheap seat-filler (qwen3:14b) generated `print("hello world")` itself.

**Consequence:** the cycle's central value proposition — *work delegated to
ensembles* — does not fire end-to-end. The callee → bridge → ApplyWork
machinery (WP-LB-B FC-44, WP-LB-D bridge, WP-LB-C substrate marshalling) is
correct and harness-tested but **unreachable in production**, because the
seat-filler is never given the delegation tool. The harness tests pass only
because they *script* `invoke_ensemble`. This is the WP-A failure mode in a new
location: harness-green, north-star not delivered. No prior cycle-status signal
flagged the absence — the WP-LB-B/C/D signals all assume the callee path fires.

**The deeper tension it exposes.** The seat-filler *decides* whether to
delegate. A model capable enough to drive the loop is often capable enough to
do the task itself — which is exactly what qwen3:14b did. The
cheap-driver-chooses-to-delegate model can be self-defeating: the driver that
can drive will skip delegation. This is the flip side of the recorded
qwen3:14b-over-delegation worry (cycle-status), and it is empirical — only a
real run with `invoke_ensemble` actually offered can answer whether the
seat-filler delegates, and whether it delegates *appropriately*.

### Finding C (minor, known) — seat-filler tool-calling is prompt-sensitive

One endpoint probe returned `finish_reason: stop` with empty content (no tool
call); a more explicit prompt ("call the write tool now") returned `tool_calls`
reliably. The same model emits a clean tool call when hit directly. This is
model thinking-mode / prompt-sensitivity, consistent with the architecture's
"seat-filler behavior unverified in BUILD by design" (WP-LB-B signal #5) — an
axis-2 / PLAY concern, not a defect.

## Disposition

WP-LB-C (terminal, emission, loop participation, bridge) is **done and
validated** for what it owns. Finding B is the load-bearing next step and is
sequenced **before** WP-LB-E/F (both downstream of delegation working):

- **WP-LB-G (proposed):** offer the seat-filler an `invoke_ensemble` tool
  alongside its client tools, fed by the capability list (the single-turn
  `_build_capability_names` is reusable), with a system prompt on when to
  delegate vs. act directly. The *design* (callee + `invoke_ensemble` +
  capability list) is settled by ADR-033 / WP-D; the gap is integration +
  prompt-shaping.
- **Load-bearing acceptance gate:** a real OpenCode session where the serve log
  shows a real `invoke_ensemble` dispatch and a `code-generator` artifact
  marshalled through the bridge into the client write — not a scripted test.
- **Loop-back trigger:** if the real run shows the seat-filler will not reliably
  delegate, that is the signal to loop back to DECIDE on the delegation-decision
  mechanism (force via `tool_choice`, a routing pre-filter, or a different
  driver-vs-delegation split) — decided on evidence, not speculation.

## Update — WP-LB-G built + validated (2026-06-02 evening)

WP-LB-G landed (commit offering the seat-filler an `invoke_ensemble` tool
enumerating the capability list + a delegate-vs-act-directly system prompt).
The real-OpenCode acceptance gate was re-run (same rig: real `llm-orc serve` +
qwen3:14b seat-filler, $0).

**Delegation fires — Finding B resolved at the mechanism level.** The serve log
shows the cheap seat-filler *chose to delegate*:

```
tool-call emit: tool=invoke_ensemble dispatch_id=…-dispatch-0002
dispatch start: ensemble=code-generator …
calibration verdict: proceed ensemble=code-generator …
tier selection: profile=agentic-tier-cheap-general tier=cheap topaz_skill=code_generation …
```

The `code-generator` ensemble ran, wrote a substrate artifact, the Artifact
Bridge marshalled it, and the surface returned `finish_reason: tool_calls` with
a `write` to the requested path. The full callee → bridge → ApplyWork chain is
reachable end-to-end with a real model. The cheap-driver-skips-delegation
tension did **not** bite here (the nudge held); whether it holds across prompts
and clients remains an axis-2 / PLAY observation.

**But Finding D (new blocker) — the marshalled deliverable is the raw ensemble
result envelope, not usable file content.** The `write` content was:

```json
{"results": {"coder": {"response": "Here's a function … ```python\ndef fibonacci(n): …```\n### Key Points …"}}}
```

The bridge faithfully marshalled what was stored; the problem is upstream, at
the ADR-025 substrate-write / ensemble-output-contract layer. Two layers:

- **D1 — wrong extraction.** The artifact (`code-generator.py`) holds the full
  multi-agent `{"results": {coder, critic, synthesizer}}` dict, not the
  synthesizer agent's final output. The substrate write stored the raw ensemble
  result structure as the deliverable.
- **D2 — form drift.** Even the synthesizer's output is conversational markdown
  (prose + fenced code + "Key Points"/"Example Usage"), not the bare file
  content a `write` needs. This is the cycle's recorded form-drift / I/O-contract
  concern (carry-forward #3; claim-extractor non-conformance) demonstrated for
  the code-generator → client-write path.

Evidence retained: `scratch/wp-lb-c-opencode-validation/wplbg_stored_artifact.json`
(the raw artifact), `wplbg_probe.out` (the probe response), `serve_wplbg.log`
(the dispatch trace).

**Disposition (for practitioner).** WP-LB-G met its gate (delegation fires).
Finding D is the next blocker and sits at a different layer than the
loop-driver/terminal work: D1 looks like a deliverable-extraction bug (the
artifact should hold the synthesizer's output, not the raw result dict); D2 is
the I/O-contract / form-drift design question the cycle already flagged for a
DECIDE-level policy (bare-deliverable contract for client-tool content, e.g.
`output_schema` / a `submit_file`-shaped synthesizer, or a deterministic
shaper). A natural fresh-context boundary — Finding D is upstream of the
terminal surface this session has been in.

## Update — WP-LB-H built + validated; Finding D refuted; Finding E surfaced (2026-06-03)

WP-LB-H landed (commits `f57b61e` deliverable contract / `030723f` consumers /
`7c14c94` synthesis excision / `db09a48` form directive FC-53/54 / `9303b0e`
FormRefusedError channel / `612aa6d` FormGate seam FC-57 / `545c1b7` critic
depends_on fix). Suite 2914 green. The acceptance gate re-ran on the same rig
(real OpenCode 1.15.5 headless → real `llm-orc serve` → qwen3:14b seat-filler
via `agentic-orchestrator-offline-tools`, $0 local; the prior default
`agentic-orchestrator` profile is Zen-blocked and was switched in config).

### Finding D refuted at the real-client layer (TS-14 reached)

Run ledger:

| Run | Layer | Task phrasing | Delegated? | Landed file |
|---|---|---|---|---|
| 1 fib.py | real OpenCode | natural | no (grounded-carry) | bare, runs — model-direct |
| 2 inventory.py | real OpenCode | natural | no (grounded-carry) | bare — model-direct |
| probe stack.py | direct endpoint | natural | **yes** | bare, `ast.parse`-clean |
| 3 matrix_utils.py | real OpenCode | delegation-leaning | **yes** (2 dispatches) | bare, parses; OpenCode executed the `write` (`tool_use … completed`) |

On the delegating runs the stored artifact is the synthesizer's bare output
(D1: never the raw `{"results": …}` dict), zero fences (the directive held
through the dependency wrapper's "provide your own analysis" framing and the
now-real critic chain), and the client-applied file parses and runs. Caveats:
run 3's session exceeded its 420s budget *after* the write completed (slow
serial local inference, not a correctness failure); the OpenCode run that
delegated used delegation-leaning phrasing.

### Finding E — delegation under the client's system prompt is a coin flip

Cumulative real-OpenCode evidence (WP-LB-G + today): **2 delegated / 2
grounded-carried** under natural phrasing. The direct-endpoint probe (no
client system prompt) delegated immediately — isolating the suppressor to
OpenCode's own system prompt out-competing `_DELEGATION_GUIDANCE` for the
seat-filler's attention. The form contract works whenever delegation fires;
*whether it fires* is now the weakest link in the north-star loop. This is
the pre-named loop-back trigger (§"Loop-back trigger" above) firing on
evidence.

**Disposition (practitioner, 2026-06-03).** Loop back to DECIDE on the
delegation-decision mechanism — grounded in spikes first: "this should be
something we can vary and test." Pre-named spike shape (Spike ψ, DECIDE
entry): capture a real OpenCode request (client system prompt included) and
replay variations against the endpoint, one lever at a time, n runs per
variant, measuring delegation rate — (ψ.1) baseline rate under the current
nudge; (ψ.2) guidance wording/position variants (Tier-2 lever); (ψ.3)
server-side `tool_choice` forcing on the seat-filler call (Ollama + qwen3 —
distinct from Spike κ's Zen/MiniMax negative); (ψ.4) structural split — a
framework pre-filter decides delegate-vs-carry and the seat-filler decides
only the action shape (model-independent, consistent with the
framework-guarantees-the-contract thesis).

### Side observations

- OpenCode's auxiliary tools-less requests (e.g. title generation) route to
  the single-turn dispatch pipeline and 500 (`Ensemble does not exist:
  agentic-routing-planner` — WP-B unbuilt). Non-fatal to the session;
  known-issue for the single-turn surface work.
- Evidence retained: `scratch/wp-lb-h-smoke/` (serve log, three session
  traces, direct-probe response, landed files) per spike-artifact-retention.

## Update — WP-LB-I built + validated; ADR-036 gating condition met; Finding F surfaced (2026-06-04)

**WP-LB-I landed** (commits `863fb5d` feat: V3 user-turn guidance composition
+ `0f9d48d` refactor: F-4 tool_choice-family-closed docstring; suite 2917
green at the 2914 baseline +3; lint clean). `_seat_filler_messages` composes
the delegation guidance into the user-turn region: a user-message tail gets
the guidance merged into it (`guidance + "\n\n---\n\n" + task` — the exact
ψ.2/ψ′-A form, 40/40), any other tail gets a standalone trailing user-role
guidance message (the C3 form, ψ′-C 5/5). No framework-authored system
message (FC-58); tool-list completeness pinned (FC-62, ψ.4c). The guidance
text is unchanged — only placement moved.

### Acceptance run — delegation VERIFIED fired (the ADR-036 Conditional Acceptance gating condition)

Real OpenCode (headless) → real `llm-orc serve` → qwen3:14b
(`agentic-orchestrator-offline-tools`), $0 local. Task issued with **natural
phrasing** ("Create a file called csv_helper.py that loads a CSV file and
computes the mean of each numeric column.") — the shape that was 0/10 under
the old system-slot composition (ψ.1 baseline).

- Serve log: `tool-call emit: tool=invoke_ensemble` → `dispatch start:
  ensemble=code-generator` on the **first turn** — delegation verified fired
  under the real client, natural phrasing, V3 composition.
- **Trailing turns delegated too** (4 delegation dispatches total) — the C3
  trailing form fired under the real client; it was replay-only evidence
  until this run.
- OpenCode executed 3 `write` tool calls; `csv_helper.py` landed as bare,
  fence-free, `ast.parse`-clean Python (the ADR-035 form contract held under
  V3).
- FC-50 loop participation: each tool result was consumed and a next-turn
  decision produced.

**Gating-condition verdict: MET.** Honest scope notes: FC-61's real-run
carry-side assertion did not occur (the session produced no carry-shaped
turns; ψ′ Arm B 0/15 + verbatim 5/5 remain the evidence); client-invisibility
holds structurally (guidance exists only on the framework → seat-filler hop)
with no dedicated test.

### Finding F (new) — termination suppression on no-new-task tool-result tails

Every trailing turn in the run was a tool-result tail with **no new user
task** — the work-complete shape whose correct decision is eventually
`finish`. The seat-filler instead delegated another revision of the same file
every turn (4 dispatches; 3 progressively rewritten `csv_helper.py` versions:
`data.csv` → `input.csv` → argv-based CLI). The session was killed by the
operator after the pattern was established; the framework turn cap (100) was
the only natural stop. **The should-finish trailing shape was never in the
55/55** — ψ′ Arm C's trailing turns all carried a genuine new task. The V3
lever works; on this shape it may over-fire — attribution (guidance vs the
model's own continuation bias under the OpenCode prompt) is exactly what
Spike ψ″ (pre-registered in the ψ research log) isolates before any
composition change. Practitioner disposition at the WP-LB-I scenario gate:
**spike the shape first**.

- Evidence retained: `scratch/wp-lb-i-smoke/` (serve log, session trace,
  three write payloads, final landed file) per spike-artifact-retention.
- Side observation: the WP-B `agentic-routing-planner` 500 on tools-less
  auxiliary requests reproduced (known issue, non-fatal).

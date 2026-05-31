# Research Log: Cycle 7 — Client-Tool-Action Terminal (loop-back from BUILD)

*Opened 2026-05-24. RESEARCH re-entry triggered by the BUILD-surfaced client-tool-action terminal gap. See `housekeeping/cycle-status.md` §"BUILD-surfaced finding" + §"Path forward".*

## Question-Isolation Entry Protocol (ADR-082)

Run before reading the corpus for this re-entry. The loop-back itself was caused by solution-anchoring on ADR-027's pipeline, so the bracketing step is load-bearing here, not ceremony.

### Step 1.1 — Research question (practitioner's words)

> "My repeated goal here is stated as my 'north star': use llm-orc via agentic serving to run RDD (or similar long-horizon process that may involve skills or skill frameworks) via a tool like OpenCode, trusting that work will be delegated to ensembles. So the spike here needs to validate that not only is the ensemble delegation effective, but can driving ensembles via chat or multi-turn interaction with OpenCode result in what we'd expect from using OpenCode with a 'normal' single model process."

The bar is **parity** — driving OpenCode against agentic-serving (every generation delegated to an ensemble) should produce the same *kind* of working agentic session as OpenCode against a normal single model. Practitioner confirmed parity is **behavioral/experiential**, NOT latency: local models are inherently slower; latency is an accepted optimization/tradeoff governed by which model profiles the ensembles select.

### Step 1.2 — Constraint-removal

**Artifact bracketed (most consequential):** ADR-027 framework-driven dispatch pipeline (plan → dispatch → synthesize, text-only terminal). The gravitational pull is to ask "how do we bolt a `tool_calls` terminal onto that pipeline."

**Prompt:** If ADR-027's dispatch pipeline weren't the committed substrate, how would ensemble-generated work land on the developer's local filesystem?

**Practitioner response:**

> "We could assume for now that we are running llm-orc on the same machine as OpenCode. We can strategize later about `llm-orc serve` running on another machine. Given that, being able to write to the local file system at a specified location should be very possible."

**Reframe surfaced (agent contribution, working inference — flagged for the spike to confirm/refute):**

The constraint-removal answer splits the loop-back finding's fused premise into two:

1. **Delivery** (bytes onto disk) — under co-location the server *can* write to a specified local path directly. The "disjoint filesystem" premise the finding leaned on (`E4.2.1`, PLAY note 22) is **contingent**; it dissolves when `llm-orc serve` and OpenCode share a machine.
2. **Parity** (OpenCode behaves like a normal agentic session) — direct server-side write does NOT deliver this: OpenCode drives the filesystem through its own tool executions and tracks state from the tool results it observes. A write behind OpenCode's back leaves its filesystem view stale and breaks the agent loop. Parity requires the deliverable to return as a `tool_calls` response OpenCode itself executes.

**Consequence for the eventual Essay-Outline 006 amendment:** the real justification for the `tool_calls` terminal is OpenCode's **execution model**, not filesystem **geography**. This is better-grounded than the loop-back finding's disjoint-filesystem argument and survives the co-location assumption.

### Sharpened research question (first draft — superseded by methods-review below)

> Does OpenCode's agentic loop stay coherent — file written with the ensemble's content, tool result fed back, next turn continues — when pointed at agentic-serving, where the generation is delegated to an ensemble and the deliverable returns as a `tool_calls` response OpenCode executes locally?

### Step 1.3 / 1.4 — Methods review + question revision

research-methods-reviewer dispatched on the question set. Report: `.rdd/audits/research-design-review-cycle-7-loopback.md`. Four flags, all converging:

- **CR-1** — the constraint-removal response resolved *delivery* (bytes to disk under co-location) but left *loop coherence* open, which is the harder half.
- **Q2-1 (primary)** — the first-draft sharpened question embeds `tool_calls` as THE mechanism: a presupposition, not a hypothesis. It forecloses the simpler co-located direct-write + text-acknowledgment path the constraint-removal itself opened. The loop-back was diagnosed as ADR-027 solution-anchoring; the first-draft question re-anchors one layer up, on `tool_calls`.
- **N-1** — the co-located direct-write alternative was not carried into the spike design; the spike would validate `tool_calls` coherence but not its *necessity*.
- **I-1** — incongruity unsurfaced: ensembles already write to substrate paths via simple direct write, sitting adjacent to the complex `tool_calls` round-trip being designed. The cycle-status finding dismissed the direct-write analogy to the orchestrator-LLM text-failure (PLAY note 22), but a co-located direct write is structurally different from the orchestrator *fabricating a description* of a write — test the analogy, don't assume it.

**Resolution:** adopt the reviewer's reformulation; test direct-write FIRST (cheap; converts the spike from a sufficiency test to a necessity test). Note: the methods-review was momentarily interrupted by accidental rejection, then re-dispatched at practitioner direction and completed.

### Revised research question (carries into the spike)

> What response shape from an ensemble-delegated endpoint lets OpenCode's agentic loop continue normally after ensemble-generated work is applied to the local filesystem — and is the `tool_calls` round-trip *necessary* for that coherence, or is a co-located direct write + text acknowledgment sufficient?

---

## Spike π — OpenCode client-tool round-trip

**Tooling (verified 2026-05-24):** OpenCode 1.15.5, Ollama 0.23.2 (qwen3:8b et al.), llm-orc 0.16.0. Scratch: `scratch/spike-pi-opencode-roundtrip/`. $0 (local Ollama). Artifacts retained per `feedback_spike_artifact_retention`.

### Phase 0 — Observe (complete)

**Method:** minimal OpenAI-compatible logging server; drove a headless `opencode run "create a file hello.py that prints hello world" --format json` at it via a project-local `opencode.json` custom provider (no global-config or paid-provider risk). Captured the raw requests.

**Findings:**

- **OpenCode sends two POSTs per `run`:** (0) a no-tools request (3 messages: system + 2 user) — title/summary generation; (1) the real agent turn: system + user, **10 tools**, `tool_choice: "auto"`, `stream: true` + `stream_options`.
- **Tool inventory (build agent):** `bash {command,description}`, `edit {filePath,oldString,newString,replaceAll}`, `glob`, `grep`, `read {filePath,...}`, `skill {name}`, `task {description,prompt,subagent_type,...}`, `todowrite`, `webfetch`, **`write {content,filePath}`** (required: `content`, `filePath`).
- **The file-creation tool is `write`** with `{filePath, content}` — this is what Phase B synthesizes. `edit` handles modify-in-place.
- **Streaming is mandatory** (`stream: true`). Phase B must emit `tool_calls` as streamed OpenAI deltas (`delta.tool_calls[].function.arguments` as string fragments + `finish_reason: "tool_calls"`), not a single JSON object.
- **Out-of-scope but north-star-relevant:** OpenCode declares native `skill` and `task` (subagent) tools — the surface a future "run RDD via OpenCode" flow would use. Noted, not tested here.

**Design implications:** Phase A (direct-write) writes to the co-located workspace and returns text ack. Phase B synthesizes a streamed `write({filePath, content})` tool-call carrying ensemble output. Both delegate content generation to a real ensemble so "delegation effective" is exercised; the response-shape difference isolates loop coherence.

**Generation step (both phases):** `spike-pi-code-generator` ensemble (single agent, `qwen3:8b`/ollama, `output_format: text`). Invoke verified: `results["code-generator"]["response"]` = `print("hello world")`, `cost_usd: 0.0`, ~10s. $0 confirmed.

### Phase A — Direct-write + text acknowledgment (necessity test, run first)

**Trace:** title-gen (no tools) → real turn (tools): server invoked the ensemble, wrote `workspace_A/hello.py` directly (21 bytes, `print("hello world")`), returned a text ack. OpenCode emitted exactly one `text` event ("I created hello.py…") + `step_finish`. **Zero tool events.** Session rc 0.

**Parity verdict: FAILS.** The file is on disk and the session "succeeds," but OpenCode executed nothing — no permission gate, no tool-execution record, no diff/undo. OpenCode operated on an **unverified text claim**, not an observed tool result. This is structurally the PLAY-note-22 trust surface (model told-about vs. driving): the bytes happen to land here, but OpenCode cannot distinguish a true claim from a false one. The single-shot create "works" only coincidentally; a multi-turn session degrades (every turn returns a claim; OpenCode never reads/edits/observes). The methods-review N-1/I-1 analogy to the orchestrator-LLM failure **holds in the load-bearing respect** even though direct-write lands real bytes.

### Phase B — `tool_calls` round-trip

**Trace (loop closed end-to-end):**
1. Real turn (tools) → server emitted a streamed `write` tool-call carrying ensemble output.
2. **OpenCode executed `write` itself** (`tool_use` event) → `workspace_B/hello.py` created with the ensemble content. *(Phase B server never direct-writes — the file's presence proves OpenCode applied the synthesized call.)*
3. Tool result `"Wrote file successfully."` fed back in a follow-up request — history `[system, user, assistant(tool_calls), tool]`.
4. Server returned closing text; OpenCode emitted final `text` + `step_finish`; rc 0.
- Content integrity held through double JSON-escaping: arguments `{"filePath":"hello.py","content":"print(\"hello world\")\n"}`.
- Permission: headless `opencode run` executed the write without stalling — no permission config required.

**Parity verdict: PARITY.** This is the standard OpenAI agentic loop — OpenCode owns and observes the tool execution.

### Necessity verdict (answers the revised research question)

**The `tool_calls` round-trip is NECESSARY for parity; co-located direct write is NOT sufficient.** Co-location dissolves the *delivery* problem (bytes to disk) but parity rests on OpenCode's *execution model*: it drives the filesystem through tool calls it executes and observes. This empirically grounds the loop-back finding's inference **and corrects its justification** — the terminal is required because of OpenCode's execution semantics, not filesystem geography (the geography argument dies under co-location; the conclusion survives anyway). *(empirically established — this cycle, n=1 task on OpenCode 1.15.5)*

### Scope of claim — what the spike did NOT establish

- **Validated:** if the framework emits a well-formed `write` tool-call carrying ensemble output, OpenCode round-trips it with full parity (one write + one continuation).
- **NOT validated (DECIDE/ARCHITECT work — this is the real shape of the "client-tool-action terminal"):**
  - The framework's *decision logic*: WHEN to emit a tool-call vs. text, and WHICH client tool a deliverable maps to (`write` vs `edit` vs `bash`).
  - **Multi-turn LOOP participation**: agentic-serving must handle the *follow-up* tool-result turn and decide continue/finish. The spike's server faked this with a canned closer; the single-turn→agentic-loop gap the finding named is real and unbuilt.
  - `edit`-in-place (needs current file state → a `read` round-trip first), multi-file, `bash`/test execution, streaming-token synthesis.
- **Architectural implication for ADR-027:** the pipeline terminal must sometimes emit a tool-call (not just text), and agentic-serving must *participate* in OpenCode's multi-turn tool loop — not the single-turn text terminal WP-A shipped. ADR-027 is incomplete, not wrong; the fix aligns with its philosophy (deterministic framework wrapping, more reliable than the orchestrator-LLM that failed at note 22).

### Spike ρ — Planner-driven delegation + `tool_calls` terminal (combined)

**Motivation (practitioner-named, essential):** Spike π Phase B *hardcoded* the delegation. This validates the C1-critical combination: does the framework's routing planner DECIDE to delegate on a real tool-rich OpenCode request, AND does that delegation return via the `tool_calls` terminal with parity? Prior spikes (ζ, ε, ν) tested the delegation decision only on the text terminal; π tested the terminal with hardcoded delegation. Never together.

**Method:** server runs the real zeta routing planner (`spike-cycle7-zeta-routing-planner`, qwen3:8b, free) on the task extracted from OpenCode's message stream → plan `{action, ensemble}` → if `dispatch`, invoke a generation ensemble → emit streamed `write` tool-call → OpenCode round-trips. Generation routed to `spike-pi-code-generator` as a confirmed-free stand-in for the planner-named `code-generator` (see findings F-ρ.1/F-ρ.2 for why the production ensemble was not invoked directly).

**Result (full loop, rc 0):**
- **Planner decision on the real OpenCode request:** `{"action":"dispatch","ensemble":"code-generator","rationale":"The request is to generate code for a Python file that prints 'hello world'."}` — **delegation decided by the framework planner, not hardcoded.** The prior cycle's tool-rich-client routing suppression did **not** recur: the planner routes on request content (AS-10), indifferent to OpenCode's 10 declared tools, because (unlike the removed orchestrator-LLM) it never sees them as routing input.
- Stages: `title-gen → agent-turn(plan=dispatch) → dispatched → emitted_tool_call → followup`. OpenCode `tool_use` event fired; `workspace_rho/hello.py` created (server never direct-writes); tool result fed back; loop closed with "Done."

**Combined verdict: VALIDATED.** `tool_calls` parity AND framework-guaranteed delegation hold *together*, end-to-end, through a real OpenCode session. Both halves of the north-star prerequisite are confirmed at the mechanism level. *(empirically established — this cycle, n=1 capability-matched task)*

**Two production-ensemble findings (surfaced, not resolved — for DECIDE/ARCHITECT/BUILD):**
- **F-ρ.1 (meaty — reshapes the terminal):** production capability ensembles (e.g., `agentic-serving/code-generator`) carry `output_substrate: artifact` (ADR-025) — deliverables route to the server-side `SessionArtifactStore` *by design*, with `envelope.primary` a summary + `ArtifactReference`. So the terminal must **bridge artifact-store → tool_call content**: read the artifact the ensemble produced, marshal it into the client `write` call. This is the **designed-in** disjoint-filesystem problem (ADR-025 routes deliverables server-side by architecture, not incidentally), and it is the more fundamental form of the co-location point: even co-located, the deliverable lives in the artifact store, not where the client expects it. Terminal shape: `dispatch → ensemble writes artifact → framework reads artifact → emit tool_call(content=artifact) → client writes locally`.
- **F-ρ.2 — RETRACTED (verification error; caught by the Amendment B citation re-audit).** This entry originally claimed `agentic-tier-cheap-general` was undefined. It is defined at `.llm-orc/profiles/agentic-tier-cheap-general.yaml` (→ `qwen3:8b`/ollama, `cost_per_token: 0.0`, since Cycle 5 / ADR-015). The error came from relying on `llm-orc list-profiles`, which enumerates only config.yaml-embedded `model_profiles:`, not the `.llm-orc/profiles/` directory. No config-hygiene blocker; the production `code-generator` resolves its cheap tier on the free tier. Retained as a recorded correction per spike-artifact-retention discipline — a clean instance of Architectural Isolation (the isolated citation auditor) catching an in-context verification error.

**Scope:** planner-driven delegation + `tool_call` terminal validated for one capability-matched code-generation task through one round-trip + one continuation. The artifact-store bridge (F-ρ.1), agentic-serving's multi-turn loop participation, `edit`/`bash`/multi-file, and direct-fallback on non-matched requests remain DECIDE/ARCHITECT/BUILD work.

**Patch-Switching Criterion (loop π/ρ → next-loop decision, ADR-096):**
- Marginal yield this loop: P1-equivalent. π — `tool_calls` necessary for parity (direct-write insufficient). ρ — planner-driven delegation + terminal validated together; tool-rich-client suppression did not recur. F-ρ.1 — artifact-store→tool_call bridge is the real terminal shape.
- Specific question for next loop: **sustained multi-turn loop participation** — does the agentic loop hold across many turns / tool types, and what drives per-turn agentic decisions (layer A) as distinct from ensemble generation (layer B)?
- Practitioner expected observation: a real multi-step run could reveal aspects of the north-star commitment the single-round-trip slice overlooked (emergent multi-turn failure; the unbuilt layer-A loop-driver). *Assessed genuine change-the-reading, not reflexive "more evidence it works" — the slice tested so far is degenerate (one turn).*
- Recommendation / decision: **CONTINUE — switch to new patch (sustained multi-turn).** Practitioner-chosen 2026-05-24; synthesis deferred until after.

### Spike σ — Sustained multi-turn: does the loop hold, and what drives it?

**The layer A / layer B distinction (motivation):** OpenCode is a harness — it executes tools and loops, but the *model* decides each turn's next action. Layer A = drive the loop (next step: read/edit/bash/finish). Layer B = generate content. π/ρ validated layer-B delegation + the terminal on a *degenerate* layer A (one turn). The routing planner (ADR-028) decides *which capability*, not *what's the next agentic step* — so **nothing in ADR-027 drives layer A across turns; it is unbuilt and undesigned.** Hopeful angle to test: note-22 was an *ungrounded* multi-dispatch composition; an OpenCode-driven loop is *grounded* (real tool result every turn), so a cheap model may sustain it where note-22 failed.

- **σ.1 (baseline):** OpenCode → local model (qwen3:14b/Ollama, no stand-in) on a multi-step task (write `calc.py`, write `test_calc.py`, run it). Can a cheap local model sustain OpenCode's multi-turn loop with parity? Layer-A feasibility floor.
- **σ.2 (delegated):** if σ.1 holds, drive layer A while delegating layer-B generation to the `code-generator` ensemble across the loop.

**σ.1 result — layer-A floor MET.** qwen3:14b (free local Ollama, no stand-in) sustained OpenCode's multi-turn loop on the 3-step task: `tool_use × 3` (write `calc.py` → write `test_calc.py` → `bash python3 test_calc.py`), final text "PASS"; both files correct; `__pycache__/` confirms python actually executed; rc 0. **The grounded-loop hypothesis holds:** a cheap local model drives OpenCode's agentic loop reliably because each turn is grounded by a real tool result — the structural difference from note-22's *ungrounded* multi-dispatch composition. *(empirically established — this cycle; qwen3:14b, 3-turn task)* **Finding:** layer-A (loop-driving) is feasible on the free tier; the north-star does not require a frontier-tier brain to drive the loop, at least for a short grounded task. Open: sustained driving over *long* horizons (dozens of turns) and harder tasks — carry-forward.

**σ.2 result — integrated north-star pattern VALIDATED.** Stand-in: layer-A driver (qwen3:14b via Ollama) decides each turn's tool calls; `write` content is delegated to the `spike-pi-code-generator` ensemble (layer B); bash/read/edit pass through. On the 3-step task the driver emitted the full plan in one turn — `write calc.py`, `write test_calc.py`, `bash` (run test) — then finished "The test passed." **Both writes were delegated** (`delegated_writes` = calc.py, test_calc.py — content came from the ensemble, not the driver). OpenCode executed all three tool calls; the ensemble-generated test **ran and PASSED** (`__pycache__/` + "The test passed."); rc 0. *(empirically established — this cycle; n=1 multi-step task)*

**Finding — orchestrator-drives-loop + ensembles-do-work composes.** The north-star pattern works end-to-end with parity: a cheap loop-driver orchestrates the agentic loop (layer A) while ensembles produce the deliverables (layer B), and the delegated content was correct enough to pass a real test run. The **layer-A loop-driver is a distinct architectural role ADR-027 lacks** — neither the routing planner (which-capability) nor the ensembles (generation) drive the per-turn agentic loop. σ confirms the role is both *necessary* and *feasible on the free tier*.

**Scope / carry-forward:** the driver batched all three actions in one planning turn (so this was 2 turns: plan + finish, not a long decide-act-observe chain). Sustained long-horizon driving (dozens of turns, harder tasks where ensemble-quality > driver-quality matters, latency compounding) remains a **BUILD-phase validation target** — better tested against the real built loop-driver/terminal than a stand-in.

**Patch-Switching Criterion (σ loop → decision, ADR-096):**
- Marginal yield: σ.1 — layer-A feasible (cheap model drives a grounded multi-turn loop). σ.2 — integrated north-star pattern composes (loop-driver + per-turn ensemble delegation, parity, ensemble-generated test passes). Plus the layer-A/B distinction and the missing-loop-driver-role finding.
- Specific question for next loop: sustained long-horizon / harder-task driving — assessed as a BUILD-phase validation target (needs the real terminal, not a stand-in), not a research-loop question.
- Recommendation / decision: **SWITCH to synthesis.** The north-star prerequisite mechanisms are validated; remaining questions are design + BUILD-phase validation. Practitioner-stated intent: "then we can synthesize and proceed."



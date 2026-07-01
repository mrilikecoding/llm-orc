# Ensemble agent: state of play + next steps (handoff)

**Status:** as of 2026-06-30. The entry point for resuming the "agent as
ensemble" work in a fresh session. Companion docs: `ensemble-serving-architecture.md`
(the settled architecture + composition strategies, diagrammed),
`ensemble-spike-sequence.md` (the gated Ω spikes + consolidated findings), and
per-spike detail under `scratch/spike-omega-*/README.md`. This doc indexes what
is proven, what is open, and the concrete next steps.

> **PLAY closed (2026-06-30).** This arc was the Cycle-7 PLAY phase (experiential
> discovery by building + driving the real system). The closeout with field notes
> + provenance is `play-closeout-2026-06-30.md`; the cycle-status is stamped. Next
> is a follow-on cycle to migrate the Cycle-7 serving strategy into this
> declarative-ensemble format (likely entering at ARCHITECT/DECIDE), owning three
> open decisions: general-vs-narrow behavior, grounded acceptance (§6.2b),
> interactive speed (§6.3). Vehicle to be chosen next session.

> **Active sub-thread (2026-06-30):** the §2 "no L0 engine change" finding was
> deliberately reopened to externalize control flow into declarative engine
> primitives. Guard/branch + a bounded loop combinator now ship in L0. Resume
> that thread from `engine-control-flow-state-and-next-steps.md`; the primitive
> vocabulary is in `../references/engine-control-flow-primitives.md`.

---

## 1. The question (reframed 2026-06-29)

Frontier capability is the assumed-capable baseline, not the thing to beat (the
6-file Ω-4 confirmed qwen3.6-plus handles normal tasks trivially). The live
questions are:

- **(a)** Can the ensemble-only architecture serve OpenCode transparently and
  drive a task to completion as if it were one model? **→ YES, verified.**
- **(b)** Which ensemble design strategies give frontier-quality results while
  minimizing frontier tokens? **→ in progress.** Strategies are mapped; one
  (contract-first) is validated; the execution gate + layer-aware escalation is
  the active frontier.

Target assertion to fully earn: *"the architecture works; the open problem is
composing ensembles for frontier-quality at low frontier-token cost."* (a)
earns "works"; (b) is the research program.

## 1a. The priority ladder (a → b → c) — the organizing frame

The architecture is a SUBSTRATE that hosts varying ensemble designs. A given
strategy (contract-first, tier-swarm, collective-intelligence, verify-heavy, ...)
is a DESIGN CHOICE that plugs in, **not an architectural mandate**. llm-orc is
the runtime; the agent is *an* ensemble; the system should host many ensembles
that all satisfy the agentic serving contract. Work is prioritized in three
tiers:

- **(a) Transparent serving.** A client like OpenCode uses llm-orc via a model
  endpoint effectively, without knowing it is an ensemble. STATUS: **confirmed
  against real `opencode run` (a2 done, 2026-06-30)** — real opencode drove the
  serve transparently and wrote runnable code to disk, after fixing three
  real-client gaps the harness masked (SSE streaming, tool-less aux calls, 32GB
  model thrash; see §6.4). Only a full-session speed limit remains (local qwen3
  thinking), not a transparency gap.
- **(b) One capable generalist ensemble.** A single ensemble that does most
  things by delegating to sub-ensembles, matching capability to task, and
  minimizing frontier use. STATUS: **the active build.** Contract-first +
  dynamic dispatch + tier routing + the execution gate + (next) layer-aware
  escalation + the architect-coherence gate all serve this one generalist; the
  §6 (b) items are what make it reliable.
- **(c) Specialized strategy ensembles.** Can we make gains on general or
  specific tasks with ensembles that use DIFFERENT strategies (collective-
  intelligence / atom-swarm, verify-heavy, decompose-heavy, ...)? STATUS:
  research frontier, gated on (b) being solid. These are design strategies the
  substrate hosts, explored and compared head-to-head, never mandated.

---

## 2. What is settled (proven, with evidence)

- **The agent can BE an ensemble with no L0 engine change.** Three of the four
  §6 primitives collapse to scripts + the artifact substrate; the fourth
  (dynamic dispatch) is adapter-mediated in ~15 lines. (Ω-1, Ω-2, Ω-dispatch.)
- **Transparent multi-turn serving works — now confirmed against REAL
  `opencode run` (a2, 2026-06-30).** The real client (opencode 1.17.11) drove the
  ensemble serve transparently: it decomposed, produced a real module, and
  opencode executed the streamed `write` tool_call and wrote working code to disk
  (verified: the cheap-local `stringcase.py` runs). Getting there required fixing
  three things the harness client masked (see 6.4): the serve was non-streaming
  (real opencode needs SSE), it decomposed opencode's tool-less title-gen call as
  a task, and the 14b+8b config thrashed GPU on the 32GB rig. After the fixes the
  chain works end to end; a *full* multi-file session just exceeds the 15-min cap
  on local inference (qwen3 thinking + multiple 8b calls/turn), a speed limit not
  a correctness one. (Ω-serve harness + real `opencode run`.)
- **Contract-first (strategy E) closes the cheap-composition coherence leaks.**
  One small frontier "architect" call freezes the cross-file contract; cheap
  tiers build against it; the gate enforces it. Produced a *running* todo
  package where the all-local form shipped a broken one. (Ω-E.)
- **The structural gate is insufficient; the execution gate is required.** ast
  passes structurally-valid but runtime-broken code (a 1.7b storage with no
  empty-file guard; a frontier parser with a forward-ref NameError). Only
  running the tests catches it. (Ω-tiers, Ω-exec.)
- **An execution gate is a truth-teller — ONLY over an evaluation the builder
  couldn't weaken.** In Ω-exec (author-controlled test) it never false-passed.
  But the Ω-P3 live run (2026-06-30) FALSE-PASSED over a builder-written test: a
  cheap-tier `test_calc.py` wrapped its asserts in `try/except print`, swallowing
  the parser's exception, so a non-functioning calc returned `rc=0`. The gate
  didn't lie; the test did. Nuance (per the §6.2b framing note): under the real
  architecture OpenCode owns the outer loop, so a bad turn is not terminal — the
  gameability RELOCATES to the loop's accept decision rather than vanishing. Wrong
  code is safely "another round" IFF the loop's correctness signal is independent
  of the builder (OpenCode running real tests, held/property checks, user
  acceptance); it becomes a wrong terminal state when the loop trusts the
  builder's own test or self-report. (Ω-exec, Ω-P3.)
- **First model-ceiling data.** qwen3:0.6b could not implement a precise
  dataclass against a frozen contract; 1.7b could. (Ω-tiers.)

---

## 3. Spike inventory

| spike | question | result | artifacts |
|---|---|---|---|
| Ω-1 | single-turn composition | PASS, no primitive | `scratch/spike-omega-1/` |
| Ω-2 | cross-run state | PASS (substrate scripts) | `scratch/spike-omega-2/` |
| Ω-2b | adapter-side recovery | PASS (mechanism); gate too weak | `scratch/spike-omega-2b/` |
| Ω-dispatch | dynamic dispatch + library reflection | PASS, ~15-line adapter | `scratch/spike-omega-dispatch/` |
| Ω-smoke | real wire contract | PASS (curl) | `scratch/spike-omega-smoke/` |
| Ω-4 | frontier vs ensemble vs bespoke (6 files) | frontier trivially handles it; scale too short to stress | `scratch/spike-omega-4/` |
| Ω-serve | transparent multi-turn serving (question a) | PASS | `scratch/spike-omega-serve/` |
| Ω-E | contract-first (strategy E) | PASS — running package | `scratch/spike-omega-e/` |
| Ω-tiers | per-sub-task tier routing + escalation | works; ceiling data; gate-too-weak surfaced | `scratch/spike-omega-tiers/` |
| Ω-exec | execution gate + escalation cascade | gate validated; escalation must be layer-aware | `scratch/spike-omega-exec/` |

---

## 4. The architecture (recap; full detail + diagrams in `ensemble-serving-architecture.md`)

OpenCode owns the loop. A thin serving adapter runs ONE ensemble turn per
request and returns one `write` tool_call or a finish; per-session state lives
in a substrate file. The turn: decompose (turn 1) → parse → plan → score
(library reflection + grounding) → adapter dispatch to the chosen capability →
form gate → recovery → marshal. Deterministic work is script stages; the
stochastic surface is the plan + the capability generation. The agent is
"ensemble + a small adapter," not literally one ensemble (dynamic dispatch
splits it).

---

## 5. Composition strategies + the frontier-token dial (full table in `ensemble-serving-architecture.md` §4/§6)

A (all-local) · B (frontier-seat, cheap-hands) · C (selective escalation) ·
D (frontier-verify/repair) · **E (contract-first, validated)**. The
frontier-token dial runs from all-local (cheap, leaky) to frontier-everywhere.
The (b) goal is the leftmost point that clears an **executional** quality bar.
The current best recipe: **E (frozen contract) + execution gate + layer-aware
escalation** — escalate only what the tests prove broken, at the right layer.

---

## 6. Open next steps (prioritized) — the fresh-session work

Mapped to the §1a ladder: **(a)** → 6.4 · **(b)** → 6.1, 6.2, 6.3, 6.5 (make the
one generalist reliable) · **(c)** → 6.6, 6.7, 6.8 (specialized strategies,
gated on (b)).

### 6.1 Layer-aware escalation (Fix 1) — the active design gap

Ω-exec showed builder-tier escalation cannot fix a fault upstream of the builder
(a contract or prompt defect); it burned ~30 min and 4 frontier files chasing a
one-line contract issue. Make escalation layer-aware:

- On gate failure, attribute the culprit (traceback file, else the logic files).
- If culprit's builder tier < top → escalate the builder (current behavior).
- **If culprit is already at the top builder tier and still fails → escalate the
  CONTRACT:** call an "architect-repair" mode (frontier, small output) with the
  failing file's contract slice + the file content + the test error; emit a
  corrected contract slice; rebuild that file (and dependents if its interface
  changed) at the cheap tier against the repaired contract; re-gate.
- Stop conditions: max contract repairs (≈2), max total rounds.
- Lift: ~40-60 lines in `scratch/spike-omega-exec/run_exec.py` + a repair prompt
  (a mode of the architect ensemble) + one validation run.

### 6.2 Validate Fix 2 (done this session) — relaxed import prompt + architect forward-ref guidance

`build_dispatch_input` now says "include AT LEAST these imports, add any you
need (e.g. `from __future__ import annotations`)"; both architect prompts now
tell the architect to add the future import for self/forward-referential
annotations. **Result (2026-06-30):** the forward-ref fix worked (a rebuilt parser included
`from __future__ import annotations`). But the re-run surfaced a distinct and
more important failure: the ARCHITECT itself emitted an INCOHERENT contract (for
a calculator it specified `parser.py` to define `parser` + `operations`,
importing `tokens` from a nonexistent `models`, conflating the calc task with
todo concepts, plus self-named definitions). The cheap builders correctly
failed against garbage and gave up on all three .py files. **Finding:
contract-first concentrates the intelligence in one frontier architect call,
which is a single point of failure, a bad contract dooms the whole cheap
build.** So (b) needs a new item:

### 6.2a (b) Architect-coherence gate

A deterministic gate on the architect's contract BEFORE building: defines don't
self-name their module; every `imports` entry references a sibling module that
exists in the contract (no orphan/hallucinated modules like `models` in a calc);
no obvious task-conflation. On failure, retry/escalate the architect. This is
cheap (a script over the contract JSON) and removes the single-point-of-failure.
It also reinforces 6.1: the contract is a real fault layer that needs its own
gate, distinct from the builder ladder.

### 6.2b (b) Grounded acceptance at the loop boundary + contract-vocabulary pinning (Ω-P3)

The Ω-P3 live run (full flow, real models) exposed two (b) gaps. The first was
initially framed as "the internal execution gate must be independent," but that
mis-locates it (see the framing note below).

- **The loop's accept/another-round decision must rest on an evaluation the
  builder could not weaken.** Ω-P3's cheap-tier `test_calc.py` swallowed
  exceptions (`try/except print`), so a non-functioning calc returned `rc=0`. The
  point is NOT that the ensemble's single-turn execution gate must be infallible.
  In the real architecture OpenCode owns the outer loop and the ensemble serves
  one turn, so "code is wrong" should trigger another round, not a terminal
  failure. The danger is that a gameable check does not vanish under the outer
  loop — it RELOCATES to wherever the loop gets its correctness signal. The loop
  only rescues you if that signal is independent of what produced the code:
  * OpenCode executing REAL tests as a tool (real output feeds back) → grounded;
    wrong code surfaces and you get another round (the intended agentic behavior).
  * The loop running the BUILDER'S OWN weak test, or the ensemble self-reporting
    "done, tested" → the false pass propagates up and the system reports success
    on broken code.
  So the requirement is grounding at the loop's accept decision: real
  tool-executed tests, held-out/property/golden checks, or the user's acceptance
  criteria — something the builder did not author or get to water down. A
  test-quality gate (reject exception-swallowing tests) is a partial mitigation
  but not a substitute for an independent signal.

  **Framing note / correction:** Ω-P3's `score` node was wired as a TERMINAL
  verdict, which collapsed OpenCode's outer loop into a one-shot internal
  certifier. That is a spike artifact, not the serving architecture (§4: OpenCode
  owns the loop, one ensemble turn per request). Read the Ω-P3 `rc=0` false pass
  as "an internal gate false-passed," and the open item as "the loop's acceptance
  must be grounded independently," NOT "the ensemble must self-certify perfectly."

- **Contract-first does not pin semantic conventions.** The cheap builders
  diverged on a string discriminant: `tokenizer` emitted `kind ∈
  {"operator","number"}`, `parser` expected `'NUMBER'`/`'+'`. The contract froze
  signatures + field names, not the *values* siblings must agree on. Fix: have
  the architect pin shared vocabularies/enums, or extend the coherence gate
  (6.2a) past import/symbol closure to flag unpinned cross-file conventions.
  Neither the coherence gate, a form gate, nor builder escalation (6.1) catches
  this — it is purely executional.

### 6.3 Sampling experiment (cheap, pending)

The tier profiles set NO sampling params → Ollama defaults, not Qwen3's
card-recommended settings (thinking T=0.6/topK=20, non-thinking T=0.7/topK=20,
never greedy), and there is no explicit thinking-mode control. Set sampling via
profile `options`, and toggle thinking via `/think` `/no_think` prompt tags
(no engine change) or a `think` param (small engine change). Hypothesis: proper
thinking-mode + sampling lifts the small-tier ceilings (the card says thinking
is "for complex logical reasoning, math, and coding"). Re-measure ceilings
after.

### 6.4 Real-OpenCode (a2) — DONE (2026-06-30), transparency confirmed + 3 fixes

Drove the serve with the real client (opencode 1.17.11) via a project
`opencode.json` custom provider (`ensemble` → `http://127.0.0.1:8099/v1`, model
`ensemble-agent`), `opencode run --dir <scratch> -m ensemble/ensemble-agent
--format json`. Non-interactive permissions via opencode's native `permission`
config (edit: allow, bash: deny — NOT the `--dangerously-skip-permissions` flag).
Setup in `scratch/spike-omega-serve/opencode_run/`.

**Result: the real client serves the ensemble transparently.** It decomposed,
produced a real module, and opencode executed the streamed `write` tool_call and
wrote runnable code to disk. But the harness (`client_drive.py`) had masked THREE
things only the real client exposed — the [[validate_against_real_client]]
failure mode, live:

1. **Non-streaming was the real a2 blocker.** The serve returned a non-stream
   JSON completion; real opencode (Vercel AI SDK `@ai-sdk/openai-compatible`)
   drives via SSE streaming and recorded 0 tokens / "unknown" finish, writing
   nothing. Fix: added an SSE path (`_stream_frames`/`_send_sse`) emitting OpenAI
   `chat.completion.chunk` frames. The harness used non-streaming, so it never
   caught this.
2. **OpenCode makes tool-less auxiliary calls** (session title generation, later
   summarization) the serve wrongly decomposed as coding tasks. Fix: short-circuit
   requests with no `tools` to a plain-text reply (`_aux_reply`); the coding loop
   is the one with tools.
3. **14b+8b thrashed GPU on the 32GB rig** ([[target_rig_is_32gb]]): decompose/
   plan on the 14b orchestrator vs 8b capability swapped 11–16GB per turn. Fix:
   all-8b variants (`decompose-8b.yaml`, `agent-turn-omega4-8b.yaml`) → single
   model loaded, no swap.

**Remaining limit (speed, not correctness):** even all-8b, a full multi-file
session exceeds the 15-min cap — qwen3 thinking-mode + several 8b inferences per
turn. Ties to 6.3 (disable thinking / sampling) to make full sessions complete.
**Grounding (per §6.2b):** the serve's contract is write-files-then-`stop` with a
STRUCTURAL self-gate only; it never asks opencode to run tests, and opencode
applies writes and would accept `stop` on the ensemble's word. So the loop as
built has NO independent verification — consistent with the §6.2b requirement to
ground the accept decision. (The one cheap-local `snake_to_camel` even shipped a
PascalCase bug a real test would catch — but the run timed out before the test
turn.)

### 6.5 Generalize further

E generalized structurally to the calc shape; widen to more domains / sizes /
dependency depths once Fix 1 + the execution gate make convergence reliable.

### 6.6 Guided decoding for small tiers (literature lever, not yet used)

The SLM survey's main tool for the structured-output gap is constrained/guided
decoding (XGrammar, Outlines). Could cut small-tier reformat loops at a
latency cost. Consider for the structured stages.

### 6.7 The ceiling questions (now unblocked)

Tier-routing + an executional gate turns "what can tier X do" into a sweep:
fix the task-type, vary the tier, measure the minimum tier that PASSES TESTS.
Questions to answer: Qwen3:0.6b's reach; whether 1.7b/4b buys an abstraction
level; tier-decomposing cognitive architectures.

### 6.8 (c) Collective-intelligence / atom-swarm strategy (research track)

A specialized ensemble DESIGN (one strategy the substrate hosts, not a mandate):
decompose work into atoms small enough that a micro tier (e.g. qwen3:0.6b)
executes each reliably, run them cheap and in parallel, and let
deterministic/executional gates SELECT good atoms (optionally N-sample per atom
+ gate-select = wisdom-of-crowds via verification). **Honest framing from the
evidence:** the intelligence lives in the DECOMPOSITION (a capable architect)
and the SELECTION ENVIRONMENT (the gates + the shared substrate as a stigmergic
coordination layer), NOT in the atoms — this is orchestrated division of labor
+ verification, not emergence from dumb agents. The 0.6b ceiling we have so far
(one confounded data point: failed a 3-field dataclass-to-contract, escalated to
1.7b) plus the distillation framing (0.6b is a pattern-executor, not a reasoner)
says atoms must be narrow, single-constraint, well-specified transforms.

Testable core (our rig is most of it already): (1) fix sampling (6.3, it
confounds the 0.6b ceiling); (2) sweep atom granularity vs an executional gate
to find where 0.6b is reliable (the `n+m` floor up to a small module); (3) add
N-sample + gate-select per atom and measure the reliability lift and cost;
(4) measure whether (architect + 0.6b swarm + gate-selection) composes a correct
whole, and how its cost/latency compares to one mid-tier model doing it directly.
Predicted failure surface: integration/coherence of the atoms (where every spike
has strained). Compare head-to-head against the (b) generalist once (b) is solid.

---

## 7. Operational facts (so a fresh session does not relearn them)

- **Frontier arm:** `qwen3.6-plus` via OpenCode Go, profile
  `agentic-orchestrator-qwen36-zen`, base `https://opencode.ai/zen/go/v1`.
  MiniMax M3 "free" and qwen3.6-plus-free promos have ENDED (paid Go now).
- **Bespoke serving bug:** the LoopDriver path forwards `content: null` (the
  OpenAI tool-call convention) to the Qwen-Go seat, which 400s ("content field
  is required"). Needs null→"" coercion in the openai-compatible client. Affects
  real OpenCode + a Qwen seat.
- **Local tiers (Ollama):** qwen3 0.6b / 1.7b / 8b / 14b (plus qwen3.5 2b/9b).
  Profiles: `micro-local` (0.6b), `agentic-tier-cheap-summary` (1.7b),
  `agentic-tier-cheap-general` (8b), `agentic-tier-escalated-general` (14b).
  Target rig is 32GB ([[target_rig_is_32gb]]).
- **Harnesses (all under `scratch/`):** `spike-omega-serve/serve_ensemble.py`
  (transparent multi-turn server) + `client_drive.py` (OpenCode-contract
  client); `spike-omega-e/run_e.py [todo|calc]` (contract-first E);
  `spike-omega-tiers/run_tiers.py` (tier routing + escalation);
  `spike-omega-exec/run_exec.py [calc|todo]` (execution gate + cascade).
  Score/exec checks run pytest in an isolated /tmp dir to dodge the repo's
  pytest-cov addopts (which false-fails `rc=1`).
- **Engine quirks used:** capabilities are loaded BY PATH (sidesteps the
  non-recursive `_find_ensemble_in_dirs` resolver); script cache disabled in
  `.llm-orc/config.yaml` (substrate-driven flows); `execution.default_timeout`
  300s.
- **Engine control-flow vocabulary:** before writing a script stage or harness
  loop, consult `references/engine-control-flow-primitives.md` — the evidence-
  backed map of what the DAG can express declaratively (`depends_on`, `fan_out`,
  `input_key`, `script`/`model_profile`/`ensemble` nodes) vs. the planned
  primitives (guard `when:`, branch, bounded `loop:`, dynamic dispatch). The
  default is a declarative node, not a large script.

---

## 8. Literature pointers (small models + cascades)

- SLM-for-agentic-systems survey (arXiv 2510.03847): validator-first, schema-first,
  SLM-default + escalation cascade — our architecture matches the consensus.
- Structured-output reliability in SLMs (arXiv 2605.02363): the format↔content
  decoupling; guided decoding as the fix.
- FrugalGPT / Cluster-Route-Escalate (arXiv 2606.27457): cascades; our upgrade is
  a deterministic *executional* escalation trigger vs self-uncertainty.
- Qwen3 Technical Report (arXiv 2505.09388): small models are strong-to-weak
  distillations of the big teachers; Qwen3-0.6B card for sampling/usage.

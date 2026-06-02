# Research Log: Cycle 7 DECIDE-entry probes — Spike τ (grounded-loop falsification) + Spike υ (wrapper-shaped)

*Opened 2026-06-01. DECIDE-entry (loop-back). Two probes queued by the loop-back MODEL/DISCOVER snapshots: a grounded-loop falsification probe (OQ #27, axis 1) and a wrapper-shaped probe (OQ #26). Both free local Ollama; $0. Scratch retained at `scratch/spike-tau-upsilon-decide-probes/` per the spike-artifact-retention directive. Tooling: OpenCode 1.15.5, Ollama 0.24.0 (qwen3:14b driver, qwen3:8b ensembles), llm-orc.*

## Why these probes, and why now

The loop-back surfaced that the north-star session is a multi-turn agentic loop needing a **layer-A loop-driver** (decide the next agentic step per turn) that no ADR-027 component holds. Two questions were held open for DECIDE on named criteria rather than default-pull:

- **OQ #27 (grounded-loop hypothesis, axis 1):** can a cheap structurally-bounded role reliably *drive* a grounded loop? The hypothesis was a working inference, not a finding — Spike σ.2 (the only multi-turn evidence) batched all three actions in one planning turn, so a real decide-act-observe chain was never exercised. Axis 2 (sequential-composition error accumulation over a long horizon) is a BUILD-phase target the short probe cannot settle.
- **OQ #26 (wrapper-vs-callee):** is the plan→dispatch→synthesize pipeline a *callee of* the loop or a *wrapper around* it? Every prior multi-turn run is callee-shaped; no wrapper-as-loop was ever spiked, so "let evidence decide" skews callee unless a wrapper-shaped probe is run as a named deliverable.

## Spike τ — Grounded-loop falsification (OQ #27, axis 1)

**Design.** A passthrough loop-driver (qwen3:14b, native tool-use) forwards OpenCode's conversation + tools, emits whatever tool_calls the driver returns unchanged (no ensemble delegation, to isolate the driver's own loop behavior), and logs both what the driver observed (prior tool results) and what it decided each turn. The task is deliberately **non-collapsible**: a random 8-char uppercase code is printed to stdout only (not written to a file, not recomputable), and the driver must create `code.txt` containing exactly that code. The only way to get it right is grounded stepping (emit bash, observe stdout, then emit write with the observed value). Batching forces the write to commit before the code is observed.

A first task design (random seed → double.txt = 2×seed, self-checking) was discarded as **collapsible**: the driver routed around the grounding requirement by writing a self-contained bash one-liner that read `seed.txt` at runtime, so it never had to carry an observed value. That run is preserved at `tau_run1_collapsible-task/` and motivated the non-collapsible redesign.

**Result — FALSIFIED in the naive form (n=4).** Every iteration batched `bash`+`write` in one planning turn and committed the write to a value it had not observed:

| Iter | Batched? | `code.txt` content | Real code printed | Grounded? |
|------|----------|--------------------|--------------------|-----------|
| 1 | yes | `${bash_output}` | `UHADIYVJ` | no (placeholder) |
| 2 | yes | `${bash_output}` | (run output) | no (placeholder) |
| 3 | yes | `ABCDEF12` | `KMKGQPEJ` | no (fabricated; not even A-Z) |
| 4 | yes (×2) | `{{$bash_output}}` | `CYMBRUSZ` / `WJCGQEOO` | no (placeholder) |

4/4 batched; 0/4 stepped to observe; 0/4 carried the real code. Three iterations emitted a literal unobserved-output placeholder (`${bash_output}`, `{{$bash_output}}`) expecting a substitution that never happens; one fabricated a fake code. This is exactly the OQ #27 axis-1 discriminating failure: *the driver emits a multi-step batch whose later step presupposes an earlier step's output it never observed, and the step fails because the presupposed state did not hold.* The placeholder string IS the presupposition.

**Reinterpretation of σ.1/σ.2.** σ's "grounded loop" succeeded only because the calc task is **batchable** — no step needs an earlier step's *runtime* output, so emitting all actions in one turn works. σ.2 was never a grounded decide-act-observe chain; it was a lucky batch. The loop-back already flagged this (σ.2 batched all three actions in one turn); Spike τ confirms the batching instinct breaks grounding the moment a task requires per-turn observation.

## Spike τ′ — Structural mitigation: framework-forced single-action-per-turn

**Design.** Same task and driver, but the framework truncates any batch the driver emits to its first tool call (`SINGLE_STEP=1`), so OpenCode must execute that one call and return its result before the driver can decide the next action. Tests whether grounding can be *structurally enforced* rather than assumed from the model.

**Result — GROUNDED PASS (n=3).** In all three iterations the driver still *tried* to batch (emitted bash+write; the framework dropped the write), but after observing the real bash output it carried the exact code into the turn-1 write every time:

| Iter | Truncation | Real code | `code.txt` | Grounded? |
|------|-----------|-----------|-----------|-----------|
| 1 | batch(2)→1 | `ZKRFWHCY` | `ZKRFWHCY` | yes |
| 2 | batch(2)→1 | `JDMEIIGT` | `JDMEIIGT` | yes |
| 3 | batch(2)→1 | `RTNDJRDF` | `RTNDJRDF` | yes |

**Combined τ/τ′ finding.** The grounded-loop hypothesis is **false for an unconstrained cheap driver** (the model's default is to batch and presuppose unobserved state) but **true under framework-enforced single-action-per-turn** (the model grounds correctly once structurally prevented from batching). The grounding *capability* is present; the failure is the *batching behavior*. This is the AS-9 structural-bounding thesis applied to the layer-A role: bound the role's per-turn shape structurally rather than relying on the model to step on its own.

**Disposition for OQ #27 / AS-9.** Axis 1 resolves to a **conditional pass**: grounded driving holds *conditional on* framework-enforced single-action-per-turn. This is an ADR-097 Conditional Acceptance shape — the condition (single-step enforcement) is a concrete architectural requirement, and axis 2 (long-horizon sequential-composition drift, dozens of turns) remains a BUILD-phase validation target the short probe cannot settle. AS-9's conditional reopening around grounded-vs-ungrounded driving is answered: a structurally-bounded role drives a grounded loop *if and only if* the structural bound includes single-action-per-turn.

## Spike υ — Wrapper-shaped probe (OQ #26)

**Design.** The missing evidence shape. The same multi-step calc task σ.2 ran, but per-turn generation runs the **full plan→dispatch→synthesize pipeline as a subroutine under the layer-A loop-driver** (vs σ.2's bare-ensemble callee call). Per-stage wall-clock is logged so latency compounding (a named OQ #26 criterion) is measurable against σ.2's single-call-per-write callee baseline.

**Result — wrapper is viable, with measured costs (n=1).** Correct `calc.py` + `test_calc.py`; the test ran and passed (parity holds). Per-write pipeline latency:

| Write | plan | dispatch | synth | total | synth changed content? |
|-------|------|----------|-------|-------|------------------------|
| calc.py | 21.2s | 17.6s | 11.6s | **50.4s** | no (`synth_differs_from_dispatch=False`) |
| test_calc.py | 15.0s | 12.3s | 7.5s | **34.8s** | no |

~85s of pipeline wall-clock across two writes.

**Findings.**
1. **Latency compounding is real.** The wrapper runs three serialized model calls per generation (planner + generator + synthesizer) where the callee (σ.2) runs one (~10-15s). The wrapper roughly **triples per-turn generation latency**, which compounds across a multi-turn session — the OQ #26 latency criterion confirmed empirically.
2. **The synthesize stage is redundant for tool-call content.** `synth_differs_from_dispatch=False` both times — the synthesizer returned the dispatched code essentially verbatim. The synthesizer is shaped to produce a user-facing chat *response*; for generating a tool-call's file *content* it is at best a ~10s no-op and at worst a corruption risk (its system prompt frames output as conversational prose; it happened not to wrap the code here, but nothing structural prevents it).
3. **The planner stage is redundant per-turn.** The layer-A loop-driver already decided it needs code generation; having the planner re-decide "which capability" every turn duplicates the driver's own role.

**Disposition for OQ #26.** The wrapper-shaped evidence — gathered deliberately to avoid callee-skew — **tilts the fork toward callee** on the named discriminating criteria: latency-over-turns (callee ~3× leaner), build-complexity/rework (callee uses the ensemble output directly; the wrapper must marshal chat-shaped synth output into tool-call content), and operator-mental-model fit (the loop-driver subsumes the planner's per-turn role). The tilt is **not** a default-pull: the wrapper shape was run as a named deliverable and works; it is simply costlier and structurally redundant for the common case. The wrapper is **not strictly dominated** — its pipeline stages would earn their keep if a *single turn's generation* genuinely needed multi-capability composition (search-then-summarize-then-write within one agentic step). That niche is the honest residual case for the wrapper reading; for per-turn single-capability generation (the common case), callee wins.

## Scope of claim — what these probes did NOT establish

- **Small-n, one model, one client.** qwen3:14b driver, qwen3:8b ensembles, OpenCode 1.15.5 only. τ n=4, τ′ n=3, υ n=1. Latency figures are single-run wall-clock on one machine, not distributions.
- **Axis 2 untested (by design).** Sequential-composition error accumulation over a genuinely long horizon (dozens of turns, harder tasks) is a BUILD-phase target. A clean τ′ result is an axis-1 conditional pass, not full validation of the grounded loop.
- **Short tasks.** τ is a 2-step value-carry; τ′ a 2-step; υ the 3-step calc task. None exercises a long decide-act-observe chain.
- **τ′ enforcement is a stand-in.** The framework-forced single-step was implemented by truncating the driver's batch in a scratch proxy. The production mechanism (and whether truncation vs. a re-planning prompt vs. a one-tool `tool_choice` constraint is the right enforcement) is DECIDE/ARCHITECT/BUILD design.
- **υ synthesizer redundancy is task-shaped.** It was redundant for trivial single-capability code-gen; a per-turn task needing composition was not tested.

## Carry-forward to DECIDE ADR work

1. **Layer-A loop-driver must structurally enforce single-action-per-turn.** Grounding is a framework property, not a model assumption (τ vs τ′). Whichever seat-filler holds layer A, the framework executes one tool call, returns the result, and forces re-planning before the next action. Codify the grounded loop as **ADR-097 Conditional Acceptance** conditional on this enforcement; axis 2 designated BUILD-phase.
2. **Wrapper-vs-callee (OQ #26) leans callee on the evidence.** The layer-A loop-driver drives the loop and delegates per-turn generation to a single capability ensemble (callee); the plan→dispatch→synthesize pipeline is not the per-turn primitive for single-capability generation. Reserve the wrapper/pipeline-per-turn shape for the named niche (per-turn multi-capability composition), if it is needed at all.
3. **The client-tool-action terminal marshals ensemble output into the tool-call content directly** (the callee shape), not through a synthesize stage shaped for chat responses. Pairs with the artifact-bridge (F-ρ.1): the terminal reads the ensemble's server-side artifact and marshals it into the `write` content.

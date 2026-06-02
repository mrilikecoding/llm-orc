# Gate Reflection: Cycle 7 (loop-back) architect → build

**Date:** 2026-06-02
**Phase boundary:** architect (loop-back from DECIDE re-entry) → build
**Cycle:** Cycle 7 — Framework-driven orchestration: routing as code (loop-back from BUILD; client-tool-action terminal / layer-A loop-driver)

## Belief-mapping question composed for this gate

> You approved the four-module decomposition at the pre-write checkpoint. The most arguable boundary in it is the **Single-Step Enforcer as its own module** rather than a behavior of the Loop Driver. What would have to be true for the Single-Step Enforcer to NOT deserve its own module — for batch-truncation to belong inside the Loop Driver instead?

Composed against the one decomposition call that sits at an FC-1 boundary (a single-responsibility module, normally an over-modularization smell) and is justified only by the FC-46 swappability argument (the grounding guarantee must survive a driver-model swap, so it cannot live in the driver). The question targets the module boundary directly, not the practitioner's prior engagement.

## User's response

At the pre-write checkpoint: **"Looks right, proceed."**

At the EPISTEMIC GATE belief-mapping question: **"The split is right"** — the framework-grounding-guarantee separation is load-bearing; grounding can't depend on the driver model; keep it split; proceed.

The practitioner did not take up the agent's separately-flagged residual uncertainty (whether FC-51 axis-2 instrumentation is actionable if the fallback ladder ignores the split-vs-callee distinction).

## Pedagogical move selected

Challenge via belief-mapping on the most arguable module boundary (Single-Step Enforcer / Loop Driver split). The agent also surfaced its own residual uncertainty (the FC-51 actionability thread) rather than presenting the design as settled. When the practitioner confirmed the split and did not engage the FC-51 thread, the agent **resolved its own flagged uncertainty by refining the design** rather than pressing the practitioner to speculate (consistent with the durable working-preference: do not ask the practitioner to speculate on unobserved results). The refinement: FC-51's split-vs-callee diagnosis now *gates the fallback branch* (callee-incorrect → wrapper reversion via FC-52; split-incorrect → Design Amendment re-opening the architecture), so the instrumentation is load-bearing rather than ceremony (ADR-095 Agent Outcome Test applied to FC-51). The gate produced a tighter artifact, not a weaker commitment.

## Audit / snapshot summary

- Susceptibility snapshot (`housekeeping/audits/susceptibility-snapshot-cycle-7-loopback-architect.md`): **No Grounding Reframe.** All six DECIDE→ARCHITECT advisories addressed substantively; module boundaries grounded in the Tool User parity mental model; D2 selected on the only directly-evidenced candidate (batch-truncation) plus model-independence; wrapper contingency architecturally accessible (FC-52 + WP-LB-E). Six carry-forward advisories for BUILD (recorded in the snapshot and in cycle-status feed-forward).
- The one advisory-level concern is **not sycophancy** (no practitioner pressure in the FC-51 direction) but **underspecification of practitioner intent**: the FC-51 fallback-branch-selector refinement was self-resolved by the agent and ratified implicitly. Snapshot Advisory 1 carries this as a BUILD-entry confirmation item (confirm the FC-51 scope before WP-LB-F).
- Design audit (Step 10) clean: invariant coverage (AS-9 honored + its §Scope boundary annotation records the conditional resolution; AS-10 preserved; no amendment), glossary coverage complete, FC-1 holds across all four new modules, dependency graph cycle-free, ADR-033/ADR-034 aligned, ADR-076 qualitative-claim gate passed (parity addressed-by-composition; axis-2 long-horizon coherence labeled structurally-non-decomposable with honest-residual-uncertainty).

## Commitment gating outputs

**Settled premises (BUILD builds on these):**
- The four-module decomposition for the tool-driven multi-turn surface: Loop Driver (L2), Single-Step Enforcer (L2), Artifact Bridge (L2), Client-Tool-Action Terminal (L3) + two extensions (Serving Layer surface-mode discrimination; Session Artifact Store `read_deliverable`).
- **Single-Step Enforcer as a distinct module is load-bearing** (gate-confirmed): the grounding guarantee must survive a driver-model swap (FC-46), so it cannot live in the driver.
- **D2: batch-truncation** is the single-step enforcement technique (the only τ′-evidenced candidate; model-independent).
- **D1: surface-mode discriminator in the Serving Layer** as a named function (not a separate module), validate-not-assumed.
- **FC-51 gates the fallback branch** (callee-incorrect → wrapper reversion; split-incorrect → Design Amendment) — the ARCHITECT-gate refinement; pending explicit BUILD-entry confirmation per snapshot Advisory 1.
- Wrapper-contingency fallback ordering: frontier-tier driver first (Model Profile swap, FC-46); wrapper reversion second-order (FC-52). No invariant change.

**Open questions (carried into BUILD/PLAY as design work or outcome-validation, NOT gate-time speculation):**
- **Axis-2 long-horizon driver coherence** — the recorded load-bearing risk (ADR-033 §Decision ¶5), validated in PLAY/first-deployment under ADR-097 Conditional Acceptance; FC-51 instrumentation makes a failure diagnosable; no synthetic test reaches it.
- **Seat-filler model choice** (cheap-tier default; frontier-tier the named axis-2 fallback) — resolved by outcome via a Model Profile swap, not re-architecture.
- **Surface-mode discriminator signal validity** — production-traffic confirmation (a tool-capable client sending `tools[]` without agentic intent).
- **`edit` / `bash` / multi-file / streaming-token coverage** — spike validated `write` only; BUILD/PLAY scope.
- **FC-51 scope confirmation** before WP-LB-F (snapshot Advisory 1).

**Specific commitments carried forward to BUILD (the six snapshot advisories + sequencing):**
1. Confirm the FC-51 scope (fallback-branch selector vs diagnostic-surface-only) with the practitioner before WP-LB-F.
2. Treat the first unexpected `tools[]` pattern from a tool-capable client as a named surface-mode-discriminator validation event, not a defect.
3. Keep the D2 tuning window visible (re-planning prompt / one-tool `tool_choice` behind the Single-Step Enforcer boundary); `tool_choice` is the weakest candidate.
4. Treat the large-deliverable artifact-bridge fidelity test (FC-49) as a first-class gate, with encoding edge cases in the matrix.
5. Confirm the wrapper-contingency fallback ordering with the practitioner before the PLAY axis-2 run.
6. Track A-LB.1 (remove the stale `ClientToolCall` docstring, `v1_chat_completions.py:581-583`) is the non-negotiable first `refactor:` commit, before any loop-back feature work.
- The **Capability List Builder (single-turn WP-D) is shared** by both surfaces (the Loop Driver selects per-turn capabilities from it).
- **BUILD-mode declaration** at BUILD entry per ADR-091: recommended **gated** (the loop-back carries the axis-2 load-bearing risk + the validate-not-assume discriminator).

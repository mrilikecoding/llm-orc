# Spike π — Destination-Form Validity Gate (Cycle 7 loop-back #7 tail; form bleed + intent-divergence)

**Status:** PRE-REGISTERED + METHODS-REVIEWED, RUN-READY (review `housekeeping/audits/research-methods-spike-pi.md`, 2 P1 / 3 P2 / 2 P3 — **all 7 applied**; disposition below). Not run. $0 local qwen3:8b (production coder) + qwen3:14b judge — the σ/η config.

**Trigger:** the loop-back #7 tail live trajectories (Spike σ §RESOLUTION + Spike η §LIVE) surfaced two residuals on the most complex of the five files, both separately tracked as the ADR-035 thread:

1. **Form bleed** (σ Run B, `scratch/spike-sigma-premature-finish/`): `cli.py` carried a trailing explanation paragraph after otherwise-valid code → `SyntaxError`. The ADR-039 content anchor propagated the real `converters` API names correctly; the **form contract leaked** — the deliverable was not bare. The boundary directive (`loop_driver.py:876`, *"Output ONLY {bare file bytes}. No markdown fences, no prose…"*) is present and correctly keyed; cheap qwen3:8b did not fully comply on the hardest file. This is the "directive-presence guaranteed, form-compliance relied-upon" gap ADR-035 §Conditional Acceptance flagged for PLAY.

2. **Intent-divergence** (η run 2, `scratch/spike-eta-deliverable-enumerator/`): the CLI emerged as `cli/index.js` (JavaScript) paired with a Python test — η log line 465 classes it "a seat/coder fault, not a judge fault." ADR-035 §Seam framing explicitly **disclaims** this as the third seam (semantic coherence; axis-2 / ensemble-quality / PLAY), distinct from the form seam it covers.

**Class:** DECIDE-driving evaluation (the θ/ρ/ξ class — research-methods review of this design is dispatched before any run; not the χ/φ bounded BUILD-gate class). The spike grounds whether/how to amend ADR-035 to cover both seams' deterministically-checkable slices; a positive result is the natural entry to DECIDE loop-back #8.

**Practitioner directive shaping this spike (2026-06-10 gate):**
- The detection mechanism is "something a well-designed spike could work on; engage the research-methods-reviewer."
- **Tackle both seams now** — do not carve out the adequacy seam as ADR-035's text does.
- An informative spike *before PLAY* is the ideal — ground the mechanism by design rather than discovering the gap experientially in PLAY.

---

## Methods-review disposition (all 7 applied, 2026-06-10)

`housekeeping/audits/research-methods-spike-pi.md` — 2 P1 / 3 P2 / 2 P3. Visible-flag disposition:

- **P1-A (C6 strawman / unfalsifiable miss-set boundary)** → C6 split into a severity range **C6a (obvious) / C6b (plausible) / C6c (near-miss)** AND the C6/non-C6 boundary pre-registered operationally (the "requires the project function-graph / intended runtime behavior to judge; no path-local validity check sees it" rule). The unification's miss-set claim is now testable against a spread, not a single fitted item.
- **P1-B (Fork 3 pass criterion unspecified)** → n, margin, and the "recovers" decision rule are now stated (n = 5 sessions/cell, ≥3 floor if env-degraded; quantified margin; explicit "protect-but-not-recover" branch). Cell A-live's role as the structural refuse-but-no-signal control is stated up front so the causal reading is not post-hoc.
- **P2-A (Arm D false-positive corpus too easy)** → added **C7b**: a correct `.py` with a module-level docstring + natural-language inline comments — the case that exercises Arm D's prose-marker false-positive risk.
- **P2-B (C4 re-pathing undocumented)** → the C4 transformation (η `index.js` bytes placed at a `.py` destination) is documented in the corpus-discipline note as the one deliberate transformation, with its rationale.
- **P2-C (recovery-loop re-delegation not traced)** → the live arm now records, per session, what the Loop Driver delegates on the turn *after* a gate refusal — separating "coder re-failed the refused file" from "driver chose a different action."
- **P3-A** → C7b doubles as the C7 edge-case-validity item.
- **P3-B** → Fork 2's decision rule gains the scope-narrowing sentence (amend only to the confirmed catch-set if B misses any of C1–C5).

---

## The central hypothesis (the cross-seam unification)

ADR-035 drew a clean line — form (its scope) vs. semantic coherence (disclaimed, PLAY). The practitioner's "both seams now" asks whether that line is the right one. This spike tests a sharper partition:

> **A single deterministic destination-validity gate — "does the deliverable validate as the kind of artifact its destination path claims it to be?" — covers the deterministically-checkable slice of *both* seams at once, leaving only an irreducibly-semantic residual for PLAY / ensemble quality.**

The mechanism: at the FormGate seam (`artifact_bridge.py`, currently `_passthrough_form_gate`), validate the marshalled deliverable against the destination path's declared form before emission — `*.py` → `ast.parse`; `*.json` → `json.loads`; `*.md` → structural / pass-through. A deliverable that fails to validate is *recognized* (not extracted from — Spike χ F-χ.1 rejected heuristic extraction) and **refused** via the pre-built `FormRefusedError` channel; the Terminal already degrades a refusal to a dispatch-failure completion (FC-57, zero Terminal edits), which leaves the file un-produced in the Session Action Record (terminal.py:117-120 — "a failed deliverable captures nothing") so the **next turn's completeness gate (ADR-040) sees the file still missing and re-delegates**. Recovery is the existing loop, self-healing — no new retry path.

Why this is a candidate unification, not two fixes:
- The **form bleed** (trailing prose in a `.py` file) breaks `ast.parse`. Caught.
- The **within-file wrong-language** adequacy slice (JS in a `.py` file) *also* breaks `ast.parse`. Caught — by the same gate, for the same reason.
- The **wrong-filename** divergence (`cli.py` requested, `index.js` produced) is the **completeness gate's** job, not the form gate's (ADR-040 matches by basename; `loop_driver.py:595-601`). The spike maps this composition rather than duplicating it.
- The residual — a deliverable that **parses fine but behaves wrong** (valid Python calling a function that does the wrong thing) — is irreducibly semantic. No deterministic gate catches it; it is genuinely PLAY / ensemble-quality territory. ADR-039's content anchor already addresses the cross-file-API slice of this; what remains is single-file semantic adequacy.

If the hypothesis holds, the principled boundary is *parse/validity*, rhyming with η's "named-file boundary is the principled edge of determinism" — the gate is deterministic exactly up to the edge of what syntactic validity can decide, and the semantic residual is named and handed to PLAY rather than carved out by fiat.

**The hypothesis is refutable.** It fails if: (a) parse-check produces false-positives on legitimate deliverables (refusing correct files), or (b) its miss-set is not confined to the irreducibly-semantic residual (it misses deterministically-detectable failures a cheaper/other gate catches), or (c) the live recovery loop does not converge — the cheap 8b coder deterministically re-fails the refused file, so self-healing loops without progress (→ the fix is a frontier seat, ADR-033 §6b escalation 3, not a gate).

---

## The forks the spike must discriminate

1. **Detection shape** (the original DECIDE-gate fork, now spike-grounded): across a labeled failure corpus, what do {parse-check, fence-only, marker-detection} each catch / miss / false-positive? Is parse-check's coverage worth its cost (the destination-path seam extension), or is the content-only fence check "good enough"?
2. **Cross-seam coverage** (the central hypothesis): does parse-check's catch-set span both seams' deterministic slices, and is its miss-set ⊆ the irreducibly-semantic residual?
3. **Recovery / convergence** (live multi-turn, PRIMARY arm per the live-multi-turn-primary directive): with parse-check + self-healing-loop recovery installed, does the σ/η 5-file trajectory converge to *all-valid* files on the production 8b coder — or does it deterministically re-fail (escalation needed)?
4. **False-positive safety**: does any gate refuse legitimate deliverables? A gate that refuses correct files is worse than pass-through.

---

## Design

### Labeled failure corpus (the methods-review-sensitive construction)

Real failures harvested from the σ/η scratch dirs, plus synthesized representatives across the failure taxonomy. Each item is a `(content, destination_path)` pair with a ground-truth label. The corpus is the substrate for the confusion matrix (Forks 1, 2, 4); it is constructed *before* any gate is implemented, to avoid fitting the corpus to a favored gate.

| # | Category | Seam | Example (source) | Deterministically detectable? |
|---|----------|------|------------------|-------------------------------|
| C1 | Trailing prose after valid code | form | σ Run B `cli.py` (real) | yes — fails `ast.parse` |
| C2 | Fenced code block (```python …```) | form | χ-era default habit (real) | yes — fails parse AND fence-visible |
| C3 | Leading/inline prose preamble | form | synthesized ("Here is the…\n\ncode") | yes — fails parse |
| C4 | Wrong-language in a typed path (JS in `.py`) | adequacy (deterministic slice) | η run 2 `index.js` content in a `.py` (real, re-pathed) | yes — fails `ast.parse` |
| C5 | Valid-language coder syntax bug | adequacy (deterministic slice) | σ Run A `args.from` (real) | yes — fails `ast.parse` |
| C6a | Parses-but-wrong — **obvious** | adequacy (semantic residual) | η `cli.py` calling nonexistent `convert_temperature` (real) | **no — the residual; PLAY territory** |
| C6b | Parses-but-wrong — **plausible** | adequacy (semantic residual) | synthesized: real fn name, swapped/wrong argument order | **no — the residual** |
| C6c | Parses-but-wrong — **near-miss** | adequacy (semantic residual) | synthesized: correct API, subtly wrong formula (off-by-one / wrong constant) | **no — the residual** |
| C7 | Correct bare `.py` | — (false-positive control) | σ/η converged files (real) | must PASS all gates |
| C7b | Correct `.py` w/ module docstring + NL comments | — (false-positive control; Arm-D stressor) | synthesized to category | must PASS all gates |
| C8 | Correct `.md` (no parser) | — (false-positive control) | σ/η README (real) | must PASS all gates |
| C9 | Correct `.json` | — (false-positive control) | synthesized valid JSON | must PASS all gates |

**C6 boundary (pre-registered operationalization, P1-A).** An item is in C6 iff it is *syntactically valid in its destination language* AND judging it wrong *requires knowledge of the surrounding project's function graph or the task's intended runtime behavior* — i.e., no destination-path-local parse/validity check can see the fault. C6a/b/c span a severity range (clearly-absent callee → plausible-but-wrong signature → API-correct-but-behaviorally-wrong) so the unification's "B miss-set = C6" claim is tested against a spread, not a single fitted item. Anything detectable by a path-local check is, by definition, *not* C6 (it belongs in C1–C5).

Corpus-construction discipline (anti-cherry-pick): C1/C2/C5/C6a/C7/C8 are taken verbatim from captured σ/η bytes where available; synthesized items (C3/C6b/C6c/C7b/C9) are written to the *category definition*, not to a gate's strengths. **C4 is the one deliberate transformation** (P2-B): the η run-2 `index.js` content is real captured JavaScript, placed at a `.py` destination path to construct the within-file wrong-language case — this is faithful to the failure (η produced JS where a Python file was the intent; the only change is asserting the `.py` destination the completeness gate would have demanded, rather than the model's self-chosen `.js` rename). The corpus and its provenance are recorded before gate code exists.

### Gate arms (run against the corpus)

- **Arm A — pass-through (today's baseline).** `_passthrough_form_gate`. Establishes that C1-C6c ship as-is (the failure) and the C7-C9 controls (incl. C7b) pass (no false-positives by construction).
- **Arm B — parse-check (the hypothesis).** Destination-validity: `.py` → `ast.parse`; `.json` → `json.loads`; `.md` → pass-through (no parser; structural-only). Predicted catch: C1-C5; predicted miss: C6a/b/c; predicted false-positive: none (the controls pass).
- **Arm C — fence-only.** Content-only: refuse if any markdown fence present. Predicted catch: C2; predicted miss: C1, C3, C4, C5, C6a/b/c; predicted false-positive: none (the directive forbade fences, so a correct file has none).
- **Arm D — marker-detection (ADR-035 §4 literal text).** Fence + prose-scaffolding markers. Predicted catch: C1, C2, C3; predicted miss: C4, C5, C6a/b/c; **predicted false-positive risk on C7b** (its module docstring + natural-language comments mistaken for prose) — the determinism-principle concern, quantified.

### Live trajectory arm (PRIMARY — Fork 3)

Reuse the σ/η 5-file temperature-library trajectory (real OpenCode → working-tree `llm-orc serve` → production qwen3:8b coder, qwen3:14b judge). **n = 5 sessions per cell** (floor 3 if marathon-session ollama degradation forces a stop — the σ env-block risk; the floor and any shortfall are documented at run). The primary measure is **per-session all-files-valid** (every produced file parses/validates against its destination form):

- **Cell A-live — today** (pass-through gate): baseline all-files-valid rate (reproduces the σ/η form-bleed-on-`cli.py` failure). **A-live is also the structural refuse-but-no-signal control (P1-B):** under pass-through, a wrong-form deliverable is emitted and surfaces only at *client execution* (the SyntaxError lands in the workspace), never at the gate — so A-live isolates the convergence that happens with no gate signal at all. This framing is stated before the run; the causal reading is not assigned post-hoc.
- **Cell B-live — parse-check gate + self-healing recovery**: all-files-valid rate. The refuse→un-produced→next-turn-re-delegate loop is exercised live.

**Pre-registered decision rule (P1-B).** The gate **recovers** iff: B-live all-files-valid ≥ 3/5 sessions AND (B-live − A-live) ≥ 2/5 sessions AND the re-dispatch-success control > 0. The gate **protects-but-does-not-recover** iff: every shipped-invalid file is caught (0 invalid files reach the client across B-live sessions) but the convergence margin is not met — this routes to ADR-033 §6b escalation 3 (frontier seat for the hardest-file tier) as the documented next lever, with the gate retained as the protection floor.

**Causal-isolation control** (separating "the gate's refusal drove convergence" from "re-dispatch randomly succeeded"): the per-turn re-dispatch success rate on a refused `cli.py` — does the 8b coder produce bare valid Python on the second attempt, or deterministically re-fail? Measured independently. If ≈ 0, convergence cannot come from self-healing, and the gate only converts a silent SyntaxError into a visible refusal (still a win — the escalation-to-frontier-seat signal).

**Recovery-loop trace (P2-C).** For every B-live session, record what the Loop Driver delegates on the turn *immediately after* a gate refusal — the destination path, and whether it re-targets the refused file or advances elsewhere. This separates "coder re-failed the same file" from "driver chose a different action after the refusal," which the all-files-valid rate alone cannot distinguish.

---

## Pre-registered outcomes

**Fork 1 (detection shape).** A gate is *viable* iff false-positive = 0 on C7-C9 AND it catches the form-bleed (C1). Decision rule: select the cheapest viable gate whose catch-set covers the practitioner's "both seams" requirement (i.e., includes the deterministic adequacy slice C4/C5). Prediction: B viable and covers C1-C5; C catches only C2 (misses the actual σ failure C1 — the fence-only insufficiency); D catches C1-C3 but risks false-positives on C7b.

**Fork 2 (unification — the central claim).** Holds iff B's catch-set ⊇ {C1-C5} AND B's miss-set = {C6a, C6b, C6c} (the residual spread) AND that residual is confirmed irreducibly-semantic (no viable gate catches any C6 variant without false-positives). If B misses any of C1-C5, the unification is incomplete — **and the amendment scope narrows to B's confirmed catch-set rather than the full claim (P3-B):** the ADR commits only what the gate demonstrably catches, with the misses recorded as a named second-mechanism need. If some gate catches a C6 variant cleanly, the residual is smaller than claimed (re-scope the PLAY hand-off, and the determinism boundary moves outward by that much).

**Fork 3 (convergence — primary).** Report Cell B-live all-files-valid rate vs. Cell A-live baseline, per the pre-registered decision rule above (recovers: B-live ≥ 3/5 AND B−A ≥ 2/5 AND re-dispatch control > 0; protects-but-does-not-recover: all shipped-invalid files caught but the margin unmet → frontier-seat escalation as the next lever, gate retained as the floor). The recovery-loop trace and the re-dispatch-success control are reported alongside, so a non-convergence result is attributable (coder re-fail vs. driver re-target).

**Fork 4 (false-positive).** Report the C7-C9 refusal rate per gate. Any gate with > 0 is disqualified from the common path (B and C predicted 0; D predicted > 0 — the quantified determinism-principle cost).

**Cost gate.** Forks 1/2/4 (corpus) and Fork 3 (live, Cells A/B) run $0 local qwen3:8b/14b. The structural gate is model-independent (deterministic framework code), so it is grounded free-first on the targeted cheap-local coder — the η/ξ "do not mask the gap with a frontier seat" discipline still governs the *gate's* validation.

**Arm E — MiniMax capability + speed contrast (pre-authorized; contingency-triggered).** The practitioner pre-authorized a bounded MiniMax-m2.5 (Zen, ~pennies — the η break-the-judge precedent) cloud arm for a **capability + speed** contrast. Unlike ξ (where a cloud arm would have masked whether the *structural* fix works), this arm does not validate the gate — the gate is structural and validated on cheap-local. It measures the **residual the gate cannot close**: if Fork 3 returns *protects-but-does-not-recover* (cheap 8b deterministically re-fails the hardest file, re-dispatch control ≈ 0), Arm E re-runs the refused-`cli.py` re-dispatch and the live Cell-B trajectory with a MiniMax **seat-filler** (the ADR-033 §6b escalation-3 lever) and reports (a) capability — does the frontier seat produce bare-valid code where 8b re-failed, closing the convergence gap? — and (b) speed — per-turn latency vs. the local seat. This quantifies what escalation-3 buys and costs, grounding whether the hardest-file tier warrants a frontier seat or stays cheap-local + gate-protected. **Trigger:** a protect-but-not-recover Fork-3 result (or genuine ambiguity on the capability-vs-structure share). **Not triggered** if Fork 3 recovers on cheap-local (the gate + self-heal suffices; Arm E is unnecessary and not spent). Estimate surfaced at the trigger point before spend (free-first discipline).

---

## CORPUS RESULT (Forks 1, 2, 4 — $0, deterministic; 2026-06-10)

Harness `scratch/spike-pi-form-adequacy-gate/` (`gates.py` candidate FormGates + `corpus.py` 12-item labeled set + `run_corpus.py`). The parse-check gate uses the `(content, destination_path)` signature — the seam extension ADR-035 requires.

| gate | catch (C1-5 det.) | false-pos (C7/C7b/C8/C9) | residual-pass (C6) | viable? | both-seams? |
|---|---|---|---|---|---|
| A_passthrough | 0/5 | 0/4 | 3/3 | no | no |
| **B_parse_check** | **5/5** (miss ∅) | **0/4** | 3/3 | **YES** | **YES** |
| C_fence_only | 1/5 (miss C1,C3,C4,C5) | 1/4 (**C8**) | 3/3 | no | no |
| D_marker | 3/5 (miss C4,C5) | 2/4 (**C7b, C8**) | 3/3 | no | no |

**Fork 1 (detection shape).** Parse-check is the only viable gate (FP=0 AND catches the σ form-bleed C1) that covers both seams (C4 wrong-language + C5 syntax-bug). Fence-only misses the actual σ failure (C1, bare trailing prose) and false-positives on a legitimate README (C8 carries a ```bash fence). Marker-detection catches the prose cases but misses the wrong-language/syntax adequacy slice and false-positives twice — on a correctly-documented `.py` (C7b's natural-language docstring/comments contain "note that" / "you can") and on the README (C8). The determinism-principle distrust of the heuristic pole is quantified: 2 FP for D, 1 for C, 0 for B.

**Fork 2 (unification).** On this corpus the claim HOLDS exactly: B's miss-set on the deterministic failures = ∅; B false-positives = ∅; B passes all three residual variants C6a/b/c (obvious / plausible / near-miss). The deterministic boundary sits at parse/validity, and the semantic residual is confirmed non-empty and uniformly uncaught by every gate — it is genuinely PLAY territory, not a gate's job.

**Fork 4 (false-positive).** B = 0/4; C = 1/4; D = 2/4. Only B is admissible in the common path.

**Honest caveats (the corpus is the synthetic, not the real client — the σ/η [[validate-against-real-client-not-harness]] lesson):**
1. The corpus result confirms the gate's *logic* and the partition, not that real-world failures are all parse-detectable. The **live arm (Fork 3) is the real test** — does the gate actually converge a real trajectory?
2. Parse-check provides **no protection for `.md`/prose destinations** (it passes them un-inspected — correct, since prose form is not structurally checkable, but it means a wrong-but-parseable-free prose deliverable is unguarded). This is acceptable: prose adequacy is the irreducibly-semantic residual regardless. Recorded as a known coverage edge.
3. C catches C2 only because the fence is visible; D's "catch" on C1/C3 rides on scaffolding phrases that a terser model might omit — the heuristic gates are not just weaker, they are *brittle* in ways B is not (B's catch is a property of the content, not of the model's phrasing habits).

## LIVE-ARM SMOKE FINDING (Fork 3 pre-flight, 2026-06-11) — recovery is not wired

Per the runbook's smoke-test-first discipline, one Cell-B session (parse gate on,
`LLMORC_SPIKE_PI_GATE=parse`, production qwen3:8b coder / qwen3:14b judge, real
OpenCode 1.15.5 → working-tree serve) ran before committing the full n=5×2. It
surfaced a structural blocker the corpus arm could not see (the [[validate-against-real-client-not-harness]]
lesson, again).

**Trajectory (`scratch/spike-pi-form-adequacy-gate/B_smoke/`):**
- Turn 1: delegate → `converters.py` (valid Python) → `write` tool call → OpenCode writes it.
- Turn 2: completeness gate `judgment_verdict=REMAINING`, content-anchor fired (`anchor=true` — ADR-039 working), delegate `test_converters.py`; the 8b coder produced invalid Python (`unterminated string literal at line 26`); **the parse gate REFUSED**.
- The refusal path (FC-57: "degrade to a dispatch-failure completion") emitted `[dispatch failed: test_converters.py: not valid Python ...]` with **`finish_reason=stop`**.
- OpenCode received the `stop` and **ended the session.** Final workspace: `converters.py` only (1/5 files; 0 invalid files reached the client).

**The gate protects but does not recover, by construction.** The protection works
exactly as designed — the invalid file was caught and never landed in the
workspace. But FC-57's refusal-as-`stop` is incompatible with the ADR-040
recovery assumption: ADR-040 re-delegates a still-missing deliverable on the
*next trailing-tool-result turn*, but a `stop` produces **no next turn** — OpenCode
treats it as a normal COMPLETE finish and terminates. The completeness gate never
fires again. The runbook's pre-registered "recovers" path
(`refuse → un-produced → re-delegate next turn`) is **unreachable with the current
wiring**: the refusal kills the loop before re-delegation can run, and the
re-dispatch-success control (does the 8b coder produce valid Python on the *second*
attempt?) cannot be measured because there is no second attempt.

**Why this stops the full run.** Running n=5×2 now would burn ~4 hours to produce
ten sessions that each terminate at the first invalid deliverable. The smoke
session already establishes the finding deterministically (the terminal code path
`client_tool_action_terminal.py:129` returns `_finish_chunks` = a `stop` on
`FormRefusedError`; the OpenCode stream confirms `STEP_FINISH reason=stop` ended
the session). One session is sufficient — the cause is structural, not stochastic.

**The recovery mechanism is the design input the smoke arm bought.** To measure
Fork 3's convergence at all, the refusal must keep the loop alive — a **server-side
re-delegation on `FormRefusedError`** (bounded re-dispatch inside the serving turn:
refuse → re-dispatch the same destination → emit the valid `write`, with a retry
cap escalating to a genuine dispatch-failure only after the cap). That is
deterministic framework control (consistent with the cycle's determinism
commitment [[determinism_over_carve_outs]]) and is itself part of the candidate
ADR-035 amendment's content, not a detour from it. The gate's *detection* is
validated (corpus + this live refusal); its *recovery* needs this wiring before
the live arm can answer "does the gate converge a real trajectory?".

### Recovery built + validated live (2026-06-11)

The server-side re-dispatch recovery was wired (env-gated under the same
`LLMORC_SPIKE_PI_GATE=parse`, additive, revert-or-ground at close) in the Loop
Driver: after `_delegate_generation` produces a deliverable, `_spike_pi_recover`
resolves the content (inline `primary` OR substrate-routed via the Session
Artifact Store — the code-generator runs `output_substrate: artifact`, so the
first inline-only draft was a no-op and had to be corrected to resolve through
the store the same way the bridge does), parse-checks it, and on a refusal
re-dispatches the same destination up to `_SPIKE_PI_MAX_REDISPATCH=2` (the coder
re-samples). The terminal's FormGate stays the final arbiter (cap exhaustion →
the dispatch-failure `stop`, the protect-but-not-recover floor that routes to
Arm E). Re-dispatch via `_delegate_generation` (not a full `decide`) avoids
double-recording the action. Constructor gains an optional `artifact_store`,
wired in the `get_loop_driver` factory. ruff + mypy clean; 76 loop_driver tests
green (env-gated path inert when off); recovery-loop logic unit-checked across 6
cases (recover-on-retry / cap-exhaust / valid-initial / env-off / md-passthrough
/ inline-resolve).

**Live validation (re-smoke, production qwen3:8b coder / 14b judge):** turn 3
`cli.py` came back invalid → `spike-pi recovery: re-dispatch 1/2 destination=cli.py`
→ `recovered=True redispatches=1` → valid cli.py written, **no refusal reached
the client**, loop continued → session **converged 5/5, all-files-valid, 0 client
refusals**. The self-healing the runbook assumed now exists. The full n=5×2
quantification run (Cell A baseline / Cell B gate+recovery) is underway; the
re-dispatch-success control is already > 0 (cli.py recovered on the first retry).

## LIVE-ARM RESULT (Fork 3, n=5×2, 2026-06-11) — protection confirmed, recovery partial

Full n=5 per cell (production qwen3:8b coder / qwen3:14b judge-seat, real OpenCode
1.15.5 → working-tree serve). A harness artifact had to be fixed first: the runner
reused an identical prompt every session, and serve derives the session id as
`sha256(first user message)` (`session_registry.py`), so all sessions collided on
one process-scoped action record and runs 2..N inherited run 1's produced files
(the η [[validate-against-real-client-not-harness]] artifact, reproduced). A unique
per-session marker in the prompt fixed it; the real run verified distinct session
ids 1:1 with markers across all 10 sessions.

**Cell A (baseline, gate off):** all 5 sessions converged to 5 files, but invalid
files landed in 3/5 (runs 1/4/5 shipped 2/3/1 unparseable files). **A all-valid =
2/5.**

**Cell B (parse gate + server-side recovery):**

| run | files | converged | invalid reaching client | recovery episode |
|---|---|---|---|---|
| 1 | 5 | ✓ | 0 | none (coder clean) |
| 2 | 5 | ✓ | 0 | cli.py + test_cli.py rescued (1 redispatch each) |
| 3 | 3 | ✗ short | 0 | cli.py exhausted (2 redispatches) → refused → stop |
| 4 | 1 | ✗ short | 0 | test_converters.py exhausted → refused → stop |
| 5 | 5 | ✓ | 0 | converters.py rescued (1 redispatch) |

Recovery: **5 episodes — 3 rescued (recovered=True), 2 exhausted (recovered=False).**

**Two findings, cleanly separated:**

1. **Protection is unambiguous and confirmed live.** Across all 5 gated sessions,
   **zero invalid files reached the client.** Every form-bled deliverable was
   caught (the 2 exhausted episodes degraded to a refusal, never a shipped broken
   file). The baseline shipped invalid files in 3/5 sessions. The spike's central
   hypothesis — a deterministic destination-validity gate catches the
   deterministic slice of both seams (form bleed + intent divergence) — **holds on
   a real trajectory.** This is the protection floor ADR-035's escalation needs.

2. **Recovery is partial — bounded by the 2-retry cap against persistent coder
   bleeds.** Server-side re-dispatch rescues *intermittent* bleeds (runs 2/5: the
   coder re-sampled a valid file within the cap → session converged) but exhausts
   on *persistent* ones (runs 3/4: the cheap 8b coder produced invalid output all
   three attempts on a single file → the gate held the line, but the session ended
   short). re-dispatch-success control = 3/5 episodes > 0 (the pre-registered
   control is met).

**Decision-rule application (the measurement nuance, stated honestly):** the
pre-registered measure is "per-session all-files-valid (every *produced* file
validates)." Read literally, **B = 5/5** (no B session ever produced an invalid
file — the gate guarantees it) vs **A = 2/5**, margin +3/5, control > 0 → the rule
returns **recovers**. But that reading is partly *vacuous*: runs 3/4 score
all-valid only because they ended at 3 and 1 files. The meaningful read is
*converged-and-all-valid*: **B = 3/5** (runs 1/2/5) vs **A = 2/5**, margin **+1/5**
→ the convergence margin (≥2/5) is **not met**. By the spirit of Fork 3 ("does the
gate converge a real trajectory?", corpus caveat #1), this is the pre-registered
**protects-but-does-not-recover** outcome: every shipped-invalid file caught, but
the convergence margin unmet → routes to the ADR-033 §6b escalation lever, gate
retained as the protection floor.

So the live arm splits the result cleanly: **gate-as-protection — strongly
grounded; recovery-as-convergence — helps but has a cap-limited ceiling on
persistent cheap-coder failures.**

**Degradation caveat (not dismissed).** The two exhausted sessions (3/4) came late
in a ~4.5h run, when the σ spike hit marathon ollama degradation, and run 4 failing
on `test_converters.py` (an easy file) is the kind of thing degradation produces.
But run 5 (the *latest* session) converged cleanly, which argues against monotonic
degradation. The confound is judged low but real; it bears on the precise recovery
rate (3/5 might be higher on fresh ollama), not on the protection finding (which is
degradation-independent — the gate catches whatever the coder produces).

**The lever nuance for escalation.** The pre-registered Arm E escalates the
MiniMax *seat*, but the residual here is the *coder* (the code-generator ensemble's
qwen3:8b produces the bled file). The lever that addresses persistent coder bleeds
is coder-tier escalation (the ADR-014 Calibration Gate cheap→escalated tier, or a
larger retry cap), not a frontier seat-filler. Arm E as specified would not move
this needle; a coder-tier contrast would.

### ARM E — coder-tier contrast (reshaped to the coder; 2026-06-11)

The practitioner pre-authorized Arm E for the protect-but-not-recover result;
reshaped from the pre-registered MiniMax *seat* to a MiniMax *coder* (the actual
residual). Focused probe (free-first, < $0.05): the exact persistent-bleed task
(generate `cli.py` for the temperature library, real `converters` API, bare-output
directive) dispatched through the code-generator ensemble N=6 each, on **fresh
ollama** (the degradation control), for the cheap qwen3:8b coder vs a MiniMax-M2.5
coder variant (`spike-pi-code-generator-minimax.yaml`, all three agents escalated).
Validity = `ast.parse` of the synthesizer deliverable.

| coder | valid | notes |
|---|---|---|
| qwen3:8b (fresh) | **3/6** | 2 invalid-syntax, 1 timeout; ~half bleed even fresh |
| MiniMax-M2.5 | **6/6** | every generation parse-valid |

**Two conclusions:**

1. **The degradation confound is dismissed.** Fresh-ollama 8b bled `cli.py` at the
   same ~50% rate it did late in the n=5×2 run. So the Cell-B exhaustions (runs 3/4)
   were genuine cheap-coder unreliability, not marathon degradation. The
   protect-but-not-recover floor is a real property of the cheap tier, not a run
   artifact.

2. **Coder-tier escalation closes the residual.** MiniMax produced valid `cli.py`
   6/6 where fresh-8b managed 3/6. The persistent-bleed residual that exhausts the
   2-retry recovery cap at the cheap tier is closed by escalating the *coder* (the
   ADR-014 Calibration Gate cheap→escalated lever), and MiniMax is fast (~1.2s
   ping vs ~10s local). The lever is coder capability, confirmed.

This completes the escalation picture the live arm opened: **gate-as-protection
(deterministic, tier-independent) + cheap-tier server-side recovery (rescues
intermittent bleeds) + coder-tier escalation (closes persistent bleeds).** The
residual after all three is the irreducibly-semantic "parses-but-wrong" slice for
PLAY — unchanged.

## What this grounds downstream

- If Fork 2 holds: candidate **ADR-035 amendment** (or a small successor ADR) extending the form contract to a deterministic destination-validity gate covering both seams' deterministic slices; ADR-035 Conditional Acceptance discharged for the form seam; the semantic residual (C6) named and handed to PLAY. DECIDE loop-back #8 entered behind this spike per the ρ/ξ precedent.
- If Fork 3 shows protect-but-not-recover: the same amendment, plus the frontier-seat escalation (ADR-033 §6b) recorded as the next lever for the hardest-file tier — a PLAY-validated, not-yet-built, escalation.
- Either way: the spike is *informative before PLAY* — PLAY validates the *experience* of the gate (does refuse-and-re-delegate feel right to a user watching the trajectory?), not the mechanism's existence.

---

## Methods-review targets (flagged for the reviewer — all addressed; see disposition above)

These were the points the pre-registration flagged for scrutiny; the review (2 P1 / 3 P2 / 2 P3) and the disposition block at the top resolve each. Retained as the pre-review record.

1. Labeled-corpus representativeness and the anti-cherry-pick discipline; whether C6 is a genuine residual or a strawman. *(→ P1-A: C6 split into a severity spread + boundary operationalized; P2-B: C4 transformation documented.)*
2. The Fork 3 causal-isolation control — is the re-dispatch-success measure the right isolation, or is a refuse-but-no-signal control needed?
3. Whether "intent-divergence" / the adequacy seam is operationalized crisply enough to be measurable (the C4 within-file wrong-language slice vs. the C6 semantic residual — is the cut clean?).
4. Embedded-conclusion check: does the central hypothesis (the unification) presuppose its answer? The arms must be able to *refute* it (false-positives, miss-set leakage, non-convergence are all pre-registered failure paths).
5. Threshold justification for Fork 3's "margin" and n on the live arm.
6. Premature-narrowing check: are there detection shapes other than parse / fence / marker that should be in the arm set (e.g., a typed `submit`-slot, already rejected by ADR-035 for destination-coupling — confirm the rejection still holds)?

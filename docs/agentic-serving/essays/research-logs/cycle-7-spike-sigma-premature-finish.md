# Spike σ — premature-finish under the real client (Finding I), live-multi-turn primary

## PRE-REGISTRATION (recorded before any run; 2026-06-09)

First spike in the cycle to make the **live multi-turn run the PRIMARY arm**, per the
practitioner directive 2026-06-09 ("the live runs should be part of the spike"; "I would
rather a spike take longer than waste time in the long-run"). Motivation: isolated
single-decision spikes (ψ, ρ, ξ) kept passing while live runs surfaced new failures (F, G,
H, and now Finding I). Isolated probes measure per-decision success; the system's goal is
per-session success, and the two do not compose. See memory
`feedback_live_multiturn_primary_spike.md`.

### Finding I (the failure to fix)

On a `REMAINING` termination verdict (ADR-037), the action call
(`_seat_filler_messages` with the ADR-038 remaining-work anchor) sometimes returns **no
tool call** (`enforced.action is None`) → `decide()` returns `FinishTurn` → the
Client-Tool-Action Terminal emits a `stop` completion → **OpenCode ends the agentic loop**.
The judge said work REMAINS, so the session finishes prematurely with deliverables missing.

This **refutes ADR-038 / Spike ρ's accepted backstop** ("the ~2/10 no-tool-call rate is
backstopped by the next re-judgment + the AS-3 cap"): under the real client a finish ENDS
the loop, so there is no next turn — neither the re-judgment nor the AS-3 cap ever fires.
Observed live: ADR-039 discharge run 1 (finish at turn 2, 1/5 files), run 2 (finish at turn
3, 2/5 files); 2/2 sessions premature. (`scratch/loopback7-content-anchor-discharge/`.)

### Hypothesis

A seat-filler no-tool-call on a `REMAINING` turn is incoherent (the framework already knows
work remains). The framework should not honor it as a finish. A **bounded retry of the
action call** recovers the (transient) stall and lets the session converge. The seat-filler
runs at the ollama default temperature (~0.8; the profile sets none), so a retry samples
differently and is expected to recover most stalls.

### Candidate fix (primary)

**F-σ.1 — REMAINING-retry (R=1):** on a `REMAINING` verdict where the action call returns
`enforced.action is None`, re-dispatch the action call once before falling through to a
finish. The AS-3 turn cap remains the ultimate backstop. (Code change in `LoopDriver.decide`,
REMAINING branch only — first-turn / COMPLETE / parse-miss paths untouched.)

**F-σ.2 — REMAINING-retry + forceful imperative:** the retry strengthens the trailing
imperative ("You must call invoke_ensemble to produce {remaining}; do not answer with
text."). Tested only if F-σ.1 plain retry is insufficient.

(F-σ.3 `tool_choice` forcing is held as a fallback, NOT primary — Spike κ found `tool_choice`
forcing unreliable on Ollama+qwen3.)

### Design — LIVE MULTI-TURN PRIMARY

- Real OpenCode 1.15.5 → real `uv run llm-orc serve --port 8765` (working-tree source) →
  real multi-turn loop. The exact 5-file Finding H temperature-library task (4 REMAINING
  turns = 4 independent chances to stall per session).
- **Config simplification to remove the infra confound:** coder = judge = seat = qwen3:14b
  (one resident ollama model, no seat↔coder swap thrash; ~5 min/run vs ~15-20 with
  swapping). Rationale: the premature-finish is the SEAT-FILLER's no-tool-call on the action
  call; the coder model is orthogonal to whether the seat-filler emits a tool call (it
  affects file quality, not tool emission). Seat and judge are already qwen3:14b in
  production (FC-68); only the coder changes (8b → 14b). The production coder (8b) + the
  ADR-039 anchor across all 5 files is re-confirmed at the discharge gate, not here.
- **Arms (same coder=14b config so the comparison is clean):**
  - BASELINE (current code, no retry): n live runs. Expected ~40% completion (~60% premature
    finish), confirming discharge runs 1+2.
  - F-σ.1 (REMAINING-retry, R=1): n live runs.
- **n = 10 per arm** (20 live runs; bumped from 8 per methods-review P1-A so the ≥0.8
  threshold is achievable at 8/10). **Early-stop rules (P2-A):** stop the fix arm early as
  GROUNDED if it reaches 8 completions before 10 runs; stop early as INSUFFICIENT if it
  accrues 3 premature-finishes before 10 runs (≤7/10 ceiling reached). Stop the baseline arm
  once the stall failure mode is confirmed present in ≥3 runs (it only needs to establish
  the failure exists and its rate, not a precise estimate).
- **No-tool-call response classification (P1-B, P3-A — zero extra runs):** for every
  no-tool-call turn in BOTH arms, classify the seat-filler's returned content as
  `stuck-text` (it answered with prose instead of acting), `false-complete` (it declared
  done), or `context-refusal`/`empty`. This distinguishes a sampling-driven stall (which a
  plain retry recovers) from a state-conditional one (which it cannot — the identical
  re-dispatch re-samples the same context), so a failed F-σ.1 is diagnostic about whether to
  escalate to F-σ.2 (perturbed prompt) vs a structural alternative.

### Primary outcome (per-SESSION)

**SESSION COMPLETION** — the session produces all 5 distinct deliverables and converges
(judge COMPLETE → finish) with NO premature finish (no REMAINING-then-finish). Binary per
run; rate per arm.

### Secondary measures (per-turn, from serve.log + workspace)

- no-tool-call count per session; **retry-recovery count** (how often a retry turned a stall
  into an action — the causal link between the fix and the completion lift).
- churn (files re-written); `anchor=true` on dependent turns (ADR-039 confirmation rides
  along); and whether the 5 landed files reference real sibling APIs (ADR-039's full
  end-to-end discharge, finally exercised — `cli.py` calls real `converters` fns, README
  documents real fns).

### Pre-registered outcome boundary (amended per methods-review P1-A / P2-B)

- **FIX GROUNDED:** F-σ.1 session-completion ≥ 8/10 AND markedly above baseline
  (baseline ≤ 5/10) AND the causal link holds (below).
- **CONDITIONAL-GROUNDED band (6–7/10):** better than baseline but below the bar — the retry
  helps but does not fully close the gap. Routed to: keep F-σ.1 (it strictly reduces premature
  finishes) AND escalate to F-σ.2 to recover the residual, rather than declaring done or
  discarding the partial win.
- **FIX INSUFFICIENT:** F-σ.1 ≤ 5/10 → read the no-tool-call classification. If stalls are
  sampling-driven (mixed content, retries sometimes recover), escalate to F-σ.2
  (perturbed/forceful retry). If state-conditional (the same context deterministically
  refuses/false-completes, retry-recovery ≈ 0), F-σ.1/F-σ.2 are the wrong shape → DECIDE on a
  structural alternative (framework-driven next-step when the seat-filler stalls).
- **Causal link, operationalized (P2-B):** the retry is credited as the mechanism only if
  retry-recovery rate ≥ 0.6 (retries that turned a stall into an action / total stalls hit by
  a retry) AND the recovered-session count ≥ the baseline→fix completion lift. Otherwise a
  completion lift is treated as run-to-run noise, not the fix.

### Baseline-interpretation scope (methods-review P2-C / P3-A)

The ~0.40 baseline models only the *stall* failure mode. With a 14b coder (vs production 8b),
non-stall failures (bad code, false COMPLETE) are likely rarer, so the observed baseline may
read *higher* than production — not because the stall rate changed. Each INCOMPLETE baseline
run is classified by cause (stall-confirmed vs other), so the baseline is read as a stall rate,
not assumed. The fix effect is measured against the stall-confirmed sub-rate.

### Honest scope

- coder=14b is NOT production (8b); the spike isolates the termination mechanism. The
  production-config full discharge (8b coder + the ADR-039 anchor across all 5 files) is the
  separate confirmation gate after the fix lands.
- Live runs are flaky (seat variance + infra); n=8/arm characterizes the rate but a larger n
  would tighten it. The n=1 limit-finding-vs-limit-absence asymmetry (trajectory pre-reg)
  applies: a high completion rate is evidence, not proof.

### Methods review (applied pre-run)

`housekeeping/audits/research-methods-spike-sigma.md` — 2 P1 / 3 P2 / 1 P3, all applied
above before any run: n 8→10 + achievable-threshold band (P1-A); no-tool-call content
classification so a failed spike is diagnostic (P1-B/P3-A); early-stop rules (P2-A);
operationalized causal link (P2-B); 14b-coder baseline scope note (P2-C).

---

## RESULTS

### Baseline arm (n=5; current code, coder=14b) — the spike pivoted here

| outcome | count |
|---|---|
| COMPLETE (5 files, converged) | 1 |
| **false-COMPLETE** (judge says COMPLETE after 1 file, turn 2) | 4 |
| no-tool-call on REMAINING (F-σ.1's pre-registered target) | **0** |

**The pre-registered fix targeted the wrong leg.** Every baseline failure was the
**judge declaring COMPLETE after converters.py alone** (4/4), and the seat no-tool-call
mode F-σ.1 addresses did not occur once. This is the methodology shift working as
intended: running the live arm *first*, before committing F-σ.1, revealed F-σ.1 is not
the dominant fix. Evidence: `baseline/SUMMARY.tsv`, `baseline/run_*.turns`,
`serve.baseline.log`.

**Mechanism.** The judge sees the quoted task (all 5 files listed), a digest that lists
only *produced* work (after turn 1: one line), and a double-negative accounting question.
It must subtract requested-minus-produced, and the qwen3:14b judge does it unreliably,
pattern-matching "a file was written → done." The digest carries produced work, not
outstanding work, so completeness-tracking rests on the stochastic judge. ADR-037
honest-residual #1, live. (Isolated Spike θ measured this judge 29/30; its test state did
not represent the live one-of-five condition — the isolated-vs-live gap, a fourth time.)

**Bonus — ADR-039 confirmed end-to-end on the one complete run.** Baseline run 1 produced
all 5 files with the anchor firing on all 4 dependent turns; cli.py, both tests, and the
README reference the **real** `converters` API with **zero invented** symbols (no
`convert_temperature`/`fahrenheit_to_kelvin`/Rankine). The Finding H invention failure did
not reproduce across a full 5-file trajectory. (cli.py + test_cli.py have intra-file syntax
errors — a separate cheap-coder-quality matter, orthogonal to ADR-039's cross-file scope;
coder=14b here, so the production-8b discharge stands separately.)

### Pivot — J-1 (judge-completeness) primary; F-σ.1 demoted to the seat-leg patch

The two legs of multi-file unreliability: the **judge** must keep saying REMAINING until
done (false-COMPLETE leg, dominant here), and the **seat** must act on each REMAINING turn
(no-tool-call leg, what F-σ.1 patches — real but absent in this config). Arms:

- **Arm A — J-1 + F-σ.1 (cheap judge, coder=14b), n=10, running.** J-1 reframes the
  accounting question from the double-negative to *positive enumeration* (list every
  requested deliverable, check each against the record, in the model's stripped reasoning)
  and raises the COMPLETE bar (`COMPLETE` only if *every* deliverable is produced). F-σ.1
  rides along but cannot change a judge verdict and was inert in the baseline, so the
  false-COMPLETE rate cleanly measures J-1; any completion lift over baseline (1/5) is
  attributable to J-1. Primary: does the false-COMPLETE rate collapse and session
  completion rise to ≥8/10? Failure-mode shift (false-COMPLETE → no-tool-call) would
  confirm J-1 fixed the judge leg and exposed the seat leg (which F-σ.1 then catches).
- **Arm B — frontier judge (Zen MiniMax / Qwen-paid), current prompt, contingent +
  practitioner-authorized spend.** Isolates capability vs information: a frontier judge
  succeeding on the *current* digest → the false-COMPLETE is a cheap-judge capability limit
  (fix = J-1 or move completeness to the framework); a frontier judge also false-COMPLETEing
  → an information limit (only a richer outstanding-work digest, J-2/J-3, will do). Run only
  if Arm A is insufficient; env-gated via `_resolve_judgment_seat` +
  `LLMORC_SPIKE_JUDGMENT_PROFILE`.

### Arm results (complete) — the false-COMPLETE rate is invariant to prompt AND judge

| arm | judge | prompt | complete | false-COMPLETE | rate |
|---|---|---|---|---|---|
| Baseline | qwen3:14b | original | 1 | 4 | 1/5 (20%) |
| Arm A (J-1 + F-σ.1) | qwen3:14b | enumeration | 1 | 3 | 1/4 (25%) |
| Arm B (frontier) | MiniMax-m2.5 (Zen) | enumeration | 1 | 4 | 1/5 (20%) |

**Conclusion — pure information limit.** The frontier judge (1/5) matched the baseline cheap
judge (1/5) exactly, and the J-1 enumeration prompt did not move the rate (25% ≈ 20%). The
false-COMPLETE rate is **invariant to both the prompt and the judge's capability**, which
rules out the two cheaper fixes at once: the produced-only digest is the bottleneck, not the
judge. No model reliably infers "requested minus produced" from a digest that lists only what
was produced; all three judges anchor on "a file was written → done." Arm B cost: ~pennies
(only the judge on Zen; seat + coder local; 0 Zen errors). Evidence: `frontier_judge/`,
`j1_fsigma1/`, `serve.j1_fsigma1.log`, `serve.log`.

**Two-leg model confirmed live.** Arm A run 1 showed both fixes composing: J-1 happened to
get the judge to REMAINING, the session reached more REMAINING turns, the seat stalled on one,
and **F-σ.1 recovered it** (`remaining-retry: recovered=True`). So F-σ.1 is the validated
**seat-leg** patch; it is real but secondary to the dominant **judge/completeness leg**.

**Fix direction (→ DECIDE loop-back on ADR-037 completeness).** Make the outstanding set
explicit: **J-2** (digest carries a requested-vs-produced checklist, so the judge reads rather
than infers) or **J-3** (framework gates COMPLETE deterministically). Both need the requested
deliverable set — the design crux (named-filename heuristic for tasks like this; a general
extraction step is the recorded boundary). Recurring cycle pattern: the cheap local model
cannot reliably infer/track X (G: progress, H: content, σ: completeness); the framework makes
X explicit and routes it forward. Recommendation: spike J-2 first (minimal, keeps the judge,
removes its failure step), escalate to J-3 if the checklist alone does not stop the
false-COMPLETE; F-σ.1 rides underneath whichever lands.

### J-3 (deterministic completeness gate) — BUILT, live validation INCOMPLETE (env-blocked 2026-06-09)

Per the practitioner's determinism principle ("limit stochasticity to the ensembles"), J-3
was built: `_extract_requested_deliverables` (filename heuristic), `_produced_paths`,
`_compose_remaining`/`_compose_done`, and `LoopDriver._completeness` — for a named-file task
the framework decides completeness deterministically (`requested − produced`, no judge), and
falls back to the ADR-037 judge only when the task names no files. The `decide` refactor
*lowered* its complexity (13→10). 4 new unit tests pass; loop_driver suite green (72).

**Live arm INCONCLUSIVE — a real-client reliability wrinkle, then env failure.** Run 1
COMPLETE (5 files, deterministic convergence at turn 7, name-matching held); run 2
false-COMPLETE at turn 2 — which is *impossible if the gate fired* (requested=5, produced=1
must yield REMAINING). So either the gate is not firing (the leading hypothesis:
`_user_task` does not return the filenamed task in OpenCode's real per-turn request, so every
turn silently falls back to the stochastic judge and J-3 is a no-op) or a logic bug the unit
tests missed. A diagnostic session with a `completeness: requested=… produced=…` log line was
added to settle it but could not run: ollama hung after hours of continuous runs (direct
3-token call times out at 60s). **The gate-firing question is unanswered.**

**Likely fix (staged, not yet applied):** extract the requested set ONCE on turn 1 (where
OpenCode is guaranteed to send the full task) and persist it session-scoped in the
SessionActionRecord, so `_completeness` reads a stable set instead of re-deriving it from a
possibly-truncated conversation each turn.

### RESUME POINTER (next session, fresh ollama)
1. Restart ollama; restart `uv run llm-orc serve --port 8765` (working-tree source).
2. Run the diagnostic session; read the `completeness:` log on turn 2. requested=[] confirms
   the gate is not firing → apply the persist-once fix. requested=[5] → re-investigate.
3. Re-run the J-3 arm (n≥6) to validate; watch COMPLETE vs name-match churn (OTHER).
4. Then: reconcile the 3 judge-path integration tests (switch their tasks to non-file), the
   DECIDE/ADR-040 write-up, and remove the temporary `completeness:` diagnostic log.

### Uncommitted / temporary state at pause
- Uncommitted code: F-σ.1 (validated seat-leg retry, keep), J-1 (insufficient, may fold or
  revert), J-3 gate + `_completeness` (built, unvalidated), Arm B `_resolve_judgment_seat`
  hook (temporary — REMOVE), the `completeness:` diagnostic log in `_completeness` (REMOVE).
- Config edits to REVERT after the spike: `.llm-orc/config.yaml` `request_timeout.read` 600→
  default; `.llm-orc/profiles/agentic-tier-cheap-general.yaml` model qwen3:14b→qwen3:8b.
- Broken: 3 judge-path integration tests (`test_termination_judgment_digest_join` ×2,
  `test_serving_surface_tool_round_trip`) — expected J-3 blast radius, reconcile to non-file
  tasks.

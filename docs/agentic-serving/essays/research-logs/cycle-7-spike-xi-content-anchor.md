# Spike ξ — Content Anchor (Cycle 7 loop-back #7, Finding H)

**Status:** COMPLETE (run 2026-06-08, $0 local qwen3:8b). **Verdict: PASS — content-anchor
grounded; B (signatures-in-dispatch) selected; the specific sibling content causally
isolated as the mechanism (B 10/10 vs decoy 0/10 vs filler 1/10 on Base T); the cheap
qwen3:8b coder resolves 10/10 on all bases — no capability rung, no cloud spend. The fix is
**content-agnostic by construction** (full-content baseline + signature compaction),
grounded across three structurally-different sibling types by practitioner-directed arms:
**prose (Base P)** A 0/10 → B 10/10 (README; prose invents *worse* blind, signatures suffice
B = C) and **config/data (Base G)** A 0/10 → B 9/10 via the full-content path with decoy
isolation (Control_decoy 0/10) — a non-code JSON sibling, keys not functions. Grounds
**ADR-039** (content-agnostic; argument-audited R1→R5 converged), Conditional Acceptance
pending the real-OpenCode trajectory discharge.** See Results below.
Pre-registered + methods-reviewed 2026-06-08
(`housekeeping/audits/research-methods-spike-xi.md`, 2 P1 / 4 P2 / 2 P3);
**all 8 findings applied** (visible-flag disposition): P1-A added a second control arm
(`Control_filler`) for the full three-way causal decomposition; P1-B added Arm D's
four-row trial-classification table; P2-A motivated the 0.2 / 0.3 thresholds; P2-B/P2-D
closed the `no-reference` / `parse-fail` denominator escapes; P2-C added the cloud-trigger
decision table; P3-A re-pinned Base V to a non-guessable API (the sleeper
baseline-validity catch — common names would let A_current resolve from training priors);
P3-B documented Arm D's read→write loop. **Post-review fidelity corrections (visible flag,
beyond the 8 findings):** the generation arms run **qwen3:8b** (the production cheap coder
where Finding H's content failure was produced — serve.log tier=cheap; the earlier
qwen3:14b draft would have measured the wrong model); Arm D demoted to a secondary
seat-layer probe (responsive to P1-B's layer-asymmetry point); the capability contrast is
free-first (qwen3:8b → qwen3:14b escalated-LOCAL → cloud-last), since the stack already
carries qwen3:14b as the Calibration-Gate escalation tier. These strengthen fidelity and
lower cost; the reviewer-endorsed core design is unchanged.
**Primary battery (coder-layer qwen3:8b) building + running 2026-06-08.**
**Trigger:** Finding H (cycle-status §"Finding H" + `scratch/spike-trajectory-live/RESULTS.md`):
the first live trajectory run (real OpenCode → working-tree `llm-orc serve` →
qwen3:14b, a 5-file dependent temperature-conversion library) held axis-1 (the
orchestration mechanism — route → advance → converge, zero churn, delegation 1.0,
capability-matched routing emerged unprompted) but crossed the **axis-2** limit:
cross-file *content* coherence is unanchored. The model issued **0 reads** all run;
the framework digest records each prior action as `(action, path, result)` but
**never the file's contents**; so each dependent deliverable was generated blind to
its siblings. `cli.py` called `converters.convert_temperature(...)` — a function that
does not exist; `README.md` documented `fahrenheit_to_kelvin` and a Rankine scale
that do not exist; `test_cli.py` asserted behavior `cli.py` does not implement.
**Class:** DECIDE-driving evaluation (the θ/ρ class — research-methods review of this
design is dispatched before any run; not the χ/φ bounded BUILD-gate class).

**The decision this spike grounds:** which content-anchor *form* fixes cross-file
API guessing **on the cheap-local coder (qwen3:8b — the production cheap tier; see the
Arms model-fidelity correction)** — and is it the specific sibling content doing the work,
or any API-shaped context? The practitioner declined a cloud-contrast that would mask the
gap: the fix must work on the targeted cheap-local coder, because the north star is a cheap
local orchestrator + ensembles, not a frontier seat. So every powered arm runs $0 local
qwen3:8b; cloud enters only as a bounded, asked-before-spent *diagnostic* contingency
(below, after the free in-stack qwen3:14b escalation rung).

---

## Why this rhymes with Finding G (and why it is a DECIDE, not RESEARCH, loop-back)

Finding G was multi-file **progress** unanchored — the seat-filler re-derived "write
file 1" because the judge's "what remains" was computed and then discarded (ADR-037
FC-66 kept call 2 byte-equal). Spike ρ grounded the fix: route the already-computed
signal forward (ADR-038's remaining-work anchor). Finding H is the **same structural
shape one layer down**: the sibling contents already exist on disk / in session; they
are simply not fed to the callee that writes a dependent file. The candidate fix is a
**content-anchor** — route the already-available sibling content (or its API surface)
into the callee dispatch. The mechanism class is known (Finding-G-shaped "route the
available signal forward"); the open question is the *form*. That bounded shape is why
loop-back #7 is DECIDE behind a spike, not RESEARCH.

---

## What is open (the forks the spike must discriminate)

1. **Does a content-anchor fix cross-file guessing on qwen3:8b at all?** (vs. the
   failure being pure model capability, fixable only with a stronger seat.)
2. **Which form?** — three candidates with different context-budget / fidelity / latency
   tradeoffs:
   - **B — signatures-in-dispatch:** the framework extracts the *public API surface*
     (function/class signatures + one-line docstrings, no bodies) of already-produced
     siblings and injects it into the callee dispatch context. Compact; cheap on tokens.
   - **C — full-content:** the framework injects the *full text* of already-produced
     siblings into the callee dispatch context. Complete; heavy; risks context bloat as
     the deliverable count grows.
   - **D — read-induce (guidance-only):** no content injected. The dispatch/trailing
     guidance instructs the seat to *read* the sibling files first (the seat has the
     read tool; it used it 0 times). Tests whether Finding H is a "the model never
     thought to read" guidance gap — the lightest-touch fix (no dispatch growth, the
     model reads only what it needs), at the cost of one extra read round-trip per
     dependent turn.
3. **Is the *specific sibling content* the mechanism, or any API-shaped context?** —
   the load-bearing causal question (the P1-B lesson from ρ). Controlled by a
   content-wrong decoy arm.

---

## Hypotheses

- **H-ξ.1 (failure reproduces):** the today-baseline (A, no anchor) reproduces Finding H
  in the harness — the generated dependent file invents at least one nonexistent
  cross-file reference at a high rate. (If A resolves *highly*, the harness does not
  capture the failure and cannot measure the fix — the Finding H lesson that synthetic
  layers can be blind; this is a guard, not a hoped-for result.)
- **H-ξ.2 (a content-anchor fixes it):** at least one of B / C / D raises cross-file
  reference resolution to a shippable rate while preserving delegation.
- **H-ξ.3 (specific content is the mechanism):** the real-sibling anchor (B) resolves
  materially above the content-wrong decoy control — the model is *using* the specific
  sibling API, not pattern-completing from any API-shaped context.
- **H-ξ.4 (form ordering, null-guarded):** signatures (B) suffices — full-content (C)
  does not materially exceed it — so the compact form is preferred; OR the data show a
  fidelity gap that justifies the heavier form. Either is a result.

## Bases (pinned before running — P3-A discipline)

Two bases avoid a base-specific effect and cover the two Finding H failure shapes
(impl→impl dependency and test→impl dependency).

- **Base T — converters → cli (impl→impl, the trajectory failure verbatim).** The
  produced sibling is a canonical `converters.py` exposing exactly three functions,
  pinned:
  ```python
  def celsius_to_fahrenheit(celsius: float) -> float: ...
  def fahrenheit_to_celsius(fahrenheit: float) -> float: ...
  def celsius_to_kelvin(celsius: float) -> float: ...
  ```
  The deliverable under test is `cli.py` (a command-line front-end that must call the
  real converter functions). The Finding H failure (verbatim): `cli.py` invented a
  unified `convert_temperature(value, from, to)` because the model's default API shape is
  one convenience converter, NOT the three pinned pairwise functions — so even though the
  individual names are familiar, the *shape* diverges from the model's blind default,
  which is what makes A_current reproduce the failure (P3-A baseline-validity holds for
  Base T via shape divergence, confirmed by the pre-run guard). Outcome: do `cli.py`'s
  references to the `converters` namespace resolve to the three real functions (and not to
  an invented unified converter)?
- **Base V — text_tools → test (test→impl, the most common real dependency).** The
  produced sibling is a canonical `text_tools.py` pinned with a **deliberately
  non-guessable** API surface (P3-A: common names like `reverse_words` / `count_vowels`
  would let A_current resolve from training-data priors, not from the anchor — defeating
  the measurement on this base). Pinned:
  ```python
  def squeeze_runs(text: str, *, chars: str = " ") -> str: ...   # collapse runs of `chars` to one
  def tally_class(text: str, predicate_name: str) -> int: ...    # count chars in a named class
  ```
  The names and the `chars` / `predicate_name` keyword contract are not the model's
  default guesses, so a blind test (A_current) cannot match them from prior knowledge — it
  must use the anchor. The deliverable under test is `test_text_tools.py` (pytest tests
  that must import and call the real functions with the real signatures). The Finding H
  failure shape: `test_cli.py` asserted behavior the implementation does not provide.
  Outcome: do the test's imports/calls resolve to `squeeze_runs` / `tally_class` with
  signature-compatible calls? *(Pre-run guard, P3-A: if A_current resolves ≥ 7/10 here,
  the names leaked through priors anyway — re-pin to a further-from-default API before
  measuring any fix.)*

A **prose→impl** characterization is run on Base T's README as a *secondary* measure
(does the generated README mention only code identifiers that exist in `converters.py`),
not a powered primary base — the AST-resolution outcome is sharp for code and noisy for
prose; it is reported as characterization, not a go/no-go.

## Arms — the file-generation (coder) layer, **qwen3:8b** ($0 local)

**Model-fidelity correction (post-review, visible flag).** The live trajectory ran the
`code-generator` coder at `agentic-tier-cheap-general` → **qwen3:8b** (tier=cheap;
serve.log dispatch-0002/4/6/8; no Calibration-Gate escalation fired). Finding H's content
failure was produced *there*, at the cheap coder — NOT at the qwen3:14b orchestrator/seat
(the layer ρ and the rung-1 probe measured). So ξ's generation arms run **qwen3:8b**, the
production cheap coder, using the ensemble's **real coder `system_prompt`** verbatim
(`.llm-orc/ensembles/agentic-serving/code-generator.yaml`). This is the north-star-faithful
unit: the question is whether the content-anchor lets the *cheap-local* coder produce
coherent cross-file code. (The earlier qwen3:14b draft would have measured the wrong model
and could have masked the effect.)

The harness unit is the **coder generation call**: the real coder system_prompt + a pinned
task (held identical across arms — only the anchor varies) + the arm's content-anchor,
posted to Ollama qwen3:8b; the generated file is captured and AST-checked. In production
the coder sees only the invoke_ensemble task + its system_prompt — NOT the sibling files
(the Finding H gap); the anchor injects the sibling API into the coder's user message, the
production seam the ADR would wire. The full ensemble + real client is the discharge gate,
not the powered unit (the ρ precedent: measure at the call where the failure lives).

Per base, n=10 (precision caveat below). The **primary battery** (A / B / C /
Control_decoy / Control_filler) is all coder-layer qwen3:8b. **Arm D is a separate
seat-layer probe** run on a trigger (see "Arm D" — it operates at the qwen3:14b seat with
a read tool, a different layer and model than the coder generation; the reviewer's P1-B
layer-asymmetry point):

- **A_current — baseline (no anchor).** The callee dispatch as composed today: task +
  framework digest (action/path/result), no sibling content. Reproduces Finding H.
  Bases T, V.
- **B_signatures — real sibling API surface.** Inject the pinned sibling's signatures +
  one-line docstrings. Bases T, V.
- **C_full-content — real sibling full text.** Inject the pinned sibling's full source.
  Bases T, V.
- **D_read-induce — SECONDARY, seat-layer probe (run on a trigger; NOT part of the
  primary coder-layer comparison).** Operates at the qwen3:14b **seat** (the orchestrator
  that holds the read tool and issued 0 reads live), not the coder — so it is not
  rate-comparable to A/B/C and is evaluated on its own question: *is the lightest fix
  simply to induce the seat to read the sibling before delegating?* Run it only if the
  primary battery is inconclusive (both B and C fail) or the practitioner wants the
  lightest-touch option characterized. The seat gets a trailing instruction to read the
  sibling file(s) before writing, with the read tool and the pinned sibling on disk. Base
  T. **Harness loop (P3-B):** the seat runs a bounded read→write loop — read
  calls are served from the pinned sibling on disk; the loop terminates at the first write
  tool call or a 3-tool-call cap (whichever first); every trial's full tool sequence +
  final generated file is retained. **Trial classification (P1-B — every trial lands in
  exactly one denominator cell; none escape):**

  | read fired? | write produced? | classification | counts toward D `resolves`? |
  |-------------|-----------------|----------------|------------------------------|
  | yes | yes | `read-write` → apply the AST `resolves` check | yes (resolves or not) |
  | yes | no  | `read-stall` | no — **non-resolve** |
  | no  | yes | `noread-write` → apply the AST `resolves` check | yes (resolves or not) |
  | no  | no  | `noread-stall` | no — **non-resolve** |

  Denominator is n=10; a stall is a failure to deliver, counted as non-resolve (not an
  escape). `read-fired` rate = (`read-write` + `read-stall`) / n.
- **Control_decoy — content-wrong (API-shaped, wrong names).** Inject signatures of the
  same shape and comparable length as B, but for **plausible-but-wrong functions NOT in
  the produced sibling** (a decoy `converters` surface of `convert_temperature` /
  `to_kelvin` / `scale` — the very invented names Finding H produced). Base T.
- **Control_filler — content-neutral (comparable length, NOT API-shaped).** Inject a
  trailing block of comparable token length to B carrying **no function/API content at
  all** (a paraphrase of generic coding-standards prose). Base T. *(P1-A: the decoy alone
  isolates "real names vs. invented names" but leaves "API-format-as-a-constraint"
  uncontrolled — if API-shaped context constrains the model to call-shaped output
  regardless of name correctness, the decoy and B both look good and the design cannot
  separate "specific content is the mechanism" from "any API-formatted context is the
  mechanism." The filler arm closes this; the two controls give the full three-way
  decomposition in the decision rule below.)*

## Measurement definitions (primary outcome is AST-checkable — refutable from the
generated file, not adjudicated)

- **resolves (primary, per trial):** parse the generated deliverable (AST); enumerate
  every cross-file reference to the produced sibling's namespace (imports of sibling
  symbols; attribute access on the imported sibling module; direct calls to imported
  names); a trial is `resolves` iff it parses, makes **≥1** cross-file reference, and
  **every** such reference targets a symbol that exists in the pinned sibling. The
  denominator is always n — **no trial escapes** (P2-B / P2-D). Each non-`resolves` trial
  is sub-classified for diagnosis: `invented` (parsed, ≥1 reference, at least one targets
  a nonexistent symbol — the Finding H signature, `convert_temperature`); `no-reference`
  (parsed but **zero** cross-file references — the deliverable sidestepped the dependency,
  e.g. reimplemented the logic inline; **counted as non-resolve, not dropped**);
  `parse-fail` (the generated file does not parse — Finding H's `args = parser.parse,args`
  shape — **counted as non-resolve, not dropped**). Also record the **graded resolution
  rate** (fraction of a trial's cross-file references that resolve) for characterization.
  The sub-class decomposition is reported for every arm — it is how the H-ξ.1
  baseline-validity guard reads whether A_current fails by invention vs. abstention.
- **syntactic-valid (secondary):** the generated file parses (Finding H had
  `args = parser.parse,args`). AST-parse success, binary.
- **delegated (guard, carry-forward from Finding B):** the file was produced via the
  delegated generation path, not an inline client `write` of generated content.
- **read-fired (D only):** the seat issued ≥1 read of the sibling before writing.
- Adjudication is mechanical (AST + symbol-table resolution against the pinned sibling);
  the pinned siblings are fixed before running, so `resolves` is refutable from the
  retained generated file with no reviewer judgment — this design intentionally avoids
  ρ's P1-A subjectivity by choosing an executable outcome.

## Pre-registered decision rule

All thresholds at n=10 carry **wide confidence intervals** (e.g., 7/10 is consistent
with true rates ~0.35–0.93 at 95% Clopper-Pearson). The rule is a structured go/no-go
for a single-spike adoption decision, not a precision rate estimate; the real-OpenCode
trajectory re-run is the layer-matching confirmation regardless of the in-harness rate.
Both the n=10 rate and the graded resolution rate are reported.

- **H-ξ.1 guard (baseline validity — P3-A / P2-B):** on each base, A_current must
  reproduce the failure **via invention**, not via abstention. Pass the guard iff
  `resolves ≤ 3/10` AND the failures are predominantly `invented` (a wrong cross-file
  reference — the Finding H shape), not `no-reference` (the deliverable sidestepped the
  dependency). Report A_current's `invented` / `no-reference` / `parse-fail` decomposition.
  If A_current `resolves ≥ 7/10` on a base, the harness does not capture Finding H there
  (the model already knows the API from priors — the Base V sleeper risk) — STOP and
  re-pin that base to a less-guessable API before measuring any fix; do not credit a fix
  against a baseline that already passes. If A_current fails predominantly by
  `no-reference`, the base tests dependency-engagement, not content-coherence — re-pin so
  the dependency is unavoidable.
- **A form passes** on a base if `resolves ≥ 7/10` AND `delegated ≥ 7/10` AND
  `syntactic-valid ≥ 8/10`.
- **Causal isolation (H-ξ.3) — the load-bearing read, full three-way decomposition
  (P1-A).** Read B vs Control_decoy vs Control_filler on Base T, with the interpretations
  pre-named so the verdict cannot drift post-result:
  - **B high, decoy low, filler low** → the *specific sibling content* is the mechanism
    (strongest result; the ADR credits "routes the real sibling API forward").
  - **B high, decoy high, filler low** → *API-shaped context* is the mechanism, not the
    specific names (any API-formatted anchor constrains the model to call-shaped output);
    the ADR ships characterized as such, and the decoy's wrong-name resolution is itself a
    brittleness warning (the model will follow a wrong anchor).
  - **B high, decoy high, filler high** → mere trailing-token perturbation; the fix is not
    content at all — re-open toward a structurally simpler change.
  - **Threshold (P2-A):** "high vs low" is a `≥ 0.3` separation on Base T. The 0.3 floor
    is the same effect-size gate ρ's causal control used (where it cleanly separated 0.8);
    it is a directional discriminant at n=10, not a precision estimate (CIs are wide — see
    the precision note). A separation in `[0.0, 0.3)` is read as "not isolated at this n"
    and routes to a named-limitation disposition, not a mechanism credit.
- **Form selection (B vs C):** if both pass, prefer **B (signatures)** — the compact,
  context-budget-frugal form — unless `C_full-content resolves − B_signatures resolves
  ≥ 0.2` on a base (a real fidelity gap that justifies the heavier payload). The simpler
  anchor wins absent a measured gap (the north star wants compact dispatches that do not
  bloat as deliverable count grows). *(P2-A: 0.2 is a directional default-toward-simpler
  threshold, not a distinguishable population-rate difference at n=10 — a sub-0.2 gap is
  read as "no evidence C beats B," so B ships; the asymmetry is deliberate, the burden of
  proof is on the heavier form. If B and C straddle the pass line — one passes, one is in
  the band — the passing form is selected and the comparison is moot.)*
- **Mechanism fork (D):** if `D_read-induce` passes comparably to B/C (within 0.2) AND
  `read-fired ≥ 8/10`, the guidance-only fix is viable and the ADR weighs it (no dispatch
  growth) against the per-turn read latency. If `read-fired` is low, the model resists
  reading even when told — refutes the guidance-only path and points at framework
  injection (B/C).

**Failure-mode backstops (ρ / ADR-097 pattern):**
- **All forms fail (none ≥ 7/10 on a base):** a content-anchor as composed does not fix
  cross-file coherence on the cheap qwen3:8b coder. This is the trigger that earns the
  **capability contrast ladder** (below): the in-stack qwen3:14b escalation rung first ($0
  local), then cloud, to discriminate "the anchor form is wrong" (the more capable seat
  also fails → redesign toward generate-then-repair against a real import check) from
  "qwen3:8b cannot use even a correct anchor" (the more capable seat passes → the fix needs
  a capability floor, which bears directly on the north star's cheap-local claim and is
  itself a finding).
- **A form passes but Control_decoy also passes (causal isolation fails):** the fix works
  for the wrong reason; ship characterized as perturbation/any-API-context, and consider
  whether a structurally simpler change achieves the same.
- **Conditional band [0.5, 0.7):** a form materially above the A_current baseline but
  below single-spike adoption confidence → the amendment proceeds as **Conditional
  Acceptance** with the real-OpenCode trajectory re-run as the discharge gate (the
  discipline ADR-037/038 carried). Below 0.5 the fix is not working on that base.
- **delegation collapses (`delegated < 0.7`):** the anchor reintroduced the Finding B
  inline-write shape; refuted regardless of resolution rate.

## Real-client discharge gate (first-class, NOT deferred — the Finding H lesson)

Re-run the **exact** Finding H live trajectory — real OpenCode 1.15.5 → working-tree
`llm-orc serve` → qwen3:14b → the 5-file temperature library — with the winning
content-anchor form wired into the callee dispatch. Verify cross-file references resolve
under the real client: `cli.py` calls real `converters` functions, README documents real
functions, tests match the implementation, zero churn preserved, convergence preserved.
$0 local. The Finding H reflective note is the reason this is first-class and not a
deferred BUILD gate: *the synthetic ladder passed axis-1 honestly, but mechanism-pass
did not generalize to whole-system correctness; axis-2 surfaced only under the real
client.* A harness PASS that does not reproduce under the real client would repeat
exactly that error.

## Capability contrast ladder (free-first; cloud is the last rung, asked-before-spent — P2-C)

Purpose: **diagnose the capability-vs-structure share**, not mask the gap (the
practitioner's distinction). The contrast runs the single best-performing form on a more
capable seat, **Base T only**, n=10. The ladder is **free-first**: the production stack
*already* carries a more capable LOCAL tier (the Calibration Gate escalates low-confidence
dispatches to `agentic-tier-escalated-general` → **qwen3:14b**, $0 local), so that is the
first capability rung, ahead of any cloud spend.

| Primary battery result (qwen3:8b coder, best form, Base T) | Next rung |
|------------------------------------------------------------|-----------|
| ≥1 form passes (≥ 0.7) AND causal isolation holds | **Stop — decisive.** The cheap-local (qwen3:8b) fix works; no contrast needed |
| a form passes BUT causal isolation fails (decoy/filler also high) | **No contrast** — re-open the mechanism characterization at the coder layer first |
| best form in band [0.5, 0.7) OR all forms < 0.5 | **Rung 1 — qwen3:14b escalated tier, LOCAL, $0.** Run the best form on the production escalation tier. Passes → the fix is right and qwen3:8b is under the capability floor the Calibration-Gate escalation already covers (north-star-positive). Fails too → Rung 2 |
| qwen3:14b-escalated also fails / stays in band | **Rung 2 — cloud (asked-before-spent).** One cloud seat, Base T, n=10: separate "form needs redesign" (cloud fails → generate-then-repair against a real import check) from "needs a frontier capability floor" (cloud passes — a finding that bears on the cheap-local north star). Est. ~10 calls, a few dollars within the ~$5 cap; concrete per-arm estimate presented **before** it runs |

The free-first ordering matters for the north star: if qwen3:14b (already in the stack as
the escalation tier) plus the anchor resolves cleanly, the cheap-local goal is met by the
existing Calibration-Gate escalation and no cloud spend is warranted.

## Out of scope for ξ (stated, not guessed across)

- The **ADR-035 form-gate bleed** (critic review prose leaked into `test_converters.py`)
  — a separate mechanism (the FormGate detect-and-refuse seam at the Artifact Bridge,
  not the content-anchor). Tracked as a small separate ADR-035 follow-up so ξ's causal
  question stays clean. Folding it in would muddy the spike (the
  prefer-clean-single-approaches-over-outcome-muddying-hybrids discipline).
- Dependency chains deeper than one hop (a file depending on a file that depends on a
  file), non-file deliverables, coder seats beyond the free-first ladder (qwen3:8b primary
  → qwen3:14b escalated-local contrast → contingent cloud diagnostic), and the README
  prose-coherence outcome beyond the secondary identifier-mention characterization.

## Fidelity discipline

- Compose through the landed callee-dispatch path (the same `_seat_filler_messages` /
  delegation composition the live trajectory exercised); inject the content-anchor at the
  callee dispatch context, framework-sourced from the pinned sibling — the production
  injection point, not a hand-built prompt.
- The pinned siblings (Base T `converters.py`, Base V `string_utils.py`) are fixed source
  files written before the run; `resolves` is checked against their real symbol tables.
- Assistant tool-call turns carry `content=""` not `None` (Ollama rejects null content —
  the ρ/rung-1 fix, carried).
- Full generated file retained per trial (the AST check is re-runnable; the artifact is
  the evidence).

## Harness

New harness `scratch/spike-xi-content-anchor/probe.py`, reusing the trajectory run's
serve/dispatch wiring and the ρ harness's composition helpers. The AST resolution check
is a small module (`parse generated file → collect sibling-namespace references →
resolve against the pinned sibling's symbol table`). Per-arm results JSON + retained
generated files under `scratch/spike-xi-content-anchor/`.

---

## Results (run 2026-06-08, $0 local qwen3:8b coder; n=10 per cell; harness
`scratch/spike-xi-content-anchor/`)

### Cross-file-reference resolution (primary outcome)

| Base | Arm | resolves | invented | no-reference | parse-fail |
|------|-----|----------|----------|--------------|------------|
| T | A_current (baseline) | **3/10** | 7 | 0 | 0 |
| T | B_signatures | **10/10** | 0 | 0 | 0 |
| T | C_full | **8/10** | 0 | 1 | 1 |
| T | Control_decoy | **0/10** | 10 | 0 | 0 |
| T | Control_filler | **1/10** | 9 | 0 | 0 |
| V | A_current (baseline) | **0/10** | 9 | 1 | 0 |
| V | B_signatures | **10/10** | 0 | 0 | 0 |
| V | C_full | **10/10** | 0 | 0 | 0 |

### Decision-rule verdict — PASS (content-anchor grounded; B/signatures selected)

- **H-ξ.1 baseline-validity guard — PASS both bases.** A_current `resolves ≤ 3/10` and
  fails predominantly by `invented` (T: 7 invented / 0 no-reference; V: 9 invented / 1
  no-reference), not abstention — the Finding H failure shape reproduces at the coder
  layer. **The P3-A Base-V re-pin is validated:** blind, the coder invented `clean_text`,
  `slugify`, `to_snake_case`, `remove_stopwords`, `tokenize_text` (etc.) — never the
  pinned `squeeze_runs` / `tally_class`. Common-name priors would have let A_current
  resolve; the non-guessable API forced genuine invention (0/10), so Base V measures the
  anchor's effect, not training-data recall.
- **Both forms pass.** B_signatures 10/10 (T) + 10/10 (V); C_full 8/10 (T) + 10/10 (V) —
  each `≥ 7/10` on each base, syntactic-validity `≥ 8/10` (only C_full had 1 parse-fail).
- **Causal isolation (the load-bearing three-way read) — the specific sibling content is
  the mechanism.** On Base T: B 10/10 vs Control_decoy 0/10 vs Control_filler 1/10.
  `B − decoy = 1.0` and `B − filler = 0.9`, both `≫ 0.3`. This is the **B-high /
  decoy-low / filler-low** branch: neither API-shape alone (decoy) nor trailing tokens
  alone (filler) breaks the guessing — only the real sibling API does. **Brittleness
  note (decoy 0/10 < A_current 3/10):** the coder *followed the decoy into invention* —
  it emitted `convert_temperature` because the wrong anchor told it to, doing worse than
  no anchor. The model uses whatever API it is handed, so the production anchor MUST be
  sourced from the actual produced file (which the framework guarantees), never a guess.
  The mechanism credit "routes the real sibling API forward" is earned, not an artifact.
- **Form selection — B (signatures) selected.** B 20/20 vs C 18/20 combined; C does not
  exceed B by `≥ 0.2` on any base (it is *lower* on T: 8 vs 10 — full-content's two
  misses were one inline reimplementation that sidestepped the dependency and one line-11
  syntax error; the extra bulk cost reliability). Per the pre-registered
  default-toward-simpler rule, the compact signatures form ships: sufficient, frugal, and
  marginally more reliable than full-content. North-star-aligned (the dispatch does not
  bloat as deliverable count grows).
- **No capability-ladder rung fires.** A form passes with causal isolation holding on the
  **cheap qwen3:8b coder** — no Conditional band, no all-forms-fail. So no qwen3:14b
  escalated-tier contrast and **no cloud spend** are warranted: the cheap-local fix works.

**n=10 precision (P2-C).** Boundary differences (8 vs 10) are not distinguishable
population rates at n=10 (wide CIs); the go/no-go is unambiguous (3/10 → 10/10 is a large
effect; 0/10 controls vs 10/10 B is decisive), the exact rates are not precise.

**Scope.** qwen3:8b coder, single-hop file dependencies (impl→impl + test→impl),
coder-generation layer. **The real-OpenCode multi-file trajectory re-run with B wired into
the callee dispatch is the layer-matching discharge gate** — first-class, not deferred
(the Finding H lesson: the synthetic ladder passed axis-1 and the real run still exposed
axis-2; an in-harness PASS must be confirmed under the real client before the ADR flips to
full Acceptance). Artifacts: `scratch/spike-xi-content-anchor/` (probe, per-cell results
JSON, retained generated files + `.code.py` extractions, `battery.log`).

### Prose arm (Base P — practitioner-directed grounding, 2026-06-09)

The code battery scoped prose out for clean AST measurement, not because the mechanism
differs. The practitioner challenged that scope: prose coherence matters equally, and
nothing is structurally different about prose from the model's side (the README also
routed to a cheap qwen3:8b coder, `prose-improver` at `tier=cheap`, generating blind to
siblings the same way). Grounded rather than assumed by a prose arm on **Base P**: the
`prose-improver` coder generating `README.md` for the converters library, with and without
the converters signatures anchor, resolution by a prose-robust regex over the README's
import lines, `module.attr()` calls, and backtick / `_to_`-shaped function mentions.

| Base | Arm | resolves | mean graded |
|------|-----|----------|-------------|
| P | A_current | **0/10** (10 invented) | 0.23 |
| P | B_signatures | **10/10** | 1.00 |
| P | C_full | **10/10** | 1.00 |

**Findings.** (1) **The anchor fixes prose coherence completely** (0/10 → 10/10) — prose is
the same mechanism, confirmed, not extrapolated. (2) **Prose invents worse than code blind**
(prose A 0/10 versus code A 3/10): the README pulls a strong "document a complete conversion
library" prior, so the blind model confidently fills in every pairwise conversion
(`fahrenheit_to_kelvin`, `kelvin_to_fahrenheit`, and in one trial a wholesale invented
`c_to_f` / `c_to_k` / `f_to_k` API), most of which do not exist — so the anchor matters
*more* for prose, not less. (3) **Signatures suffice for prose** (B 10/10 = C 10/10): the
sufficiency nuance (that prose might need behavior or examples beyond the API surface) did
not materialize — signatures plus one-line docstrings let the README document the real API.
So signatures is the right form for both callees. **Measurement caveat:** prose resolution
is regex-heuristic (targeting conversion-function-shaped identifiers), not AST-clean like
code; the 0/10 versus 10/10 gap is far outside any plausible extraction-noise margin.
Artifacts: `scratch/spike-xi-content-anchor/results_P_*.json` + retained READMEs.

### Cross-type arm (Base G — config/data sibling, content-agnosticism confirmation, 2026-06-09)

The practitioner set content-agnosticism as the build commitment ("this needs to be
completely content agnostic... whatever that entails"). The code and prose arms both
sourced the anchor from a *Python* sibling; the open question for "completely
content-agnostic" was whether the universal full-content path works when the sibling is a
**non-code** file. Base G probes the most-different case: a `settings.json` **data**
sibling whose Python consumer (`scheduler.py`) must reference specific config keys — a
different file kind (data, not code) and a different interface (keys, not function
signatures). The keys are deliberately opaque (`rbo_ms`, `qdepth_max`, `aff_salt`) so a
blind consumer forced to reference them by purpose cannot guess them.

*(Base-validity iteration, itself a finding.* The first Base-G design used a generic
*loader* task; the blind loader read the dict generically and never referenced specific
keys — `no-reference`, not `invented` — so it did not reproduce the failure. The
baseline-validity guard caught it; the task was re-pinned to force direct key references.
The lesson: the cross-file-coherence failure arises only when the consumer must commit to a
sibling's specific named interface; generic-blob consumption dodges it. That bounds where
the anchor matters — it is moot for siblings consumed wholesale.)*

| Base | Arm | resolves | failure |
|------|-----|----------|---------|
| G | A_current | **0/10** | 10 invented (`retry_backoff_ms`, `max_queue_depth`, `affinity_salt` — the plausible guesses) |
| G | B_content (full-content anchor) | **9/10** | 1 invented |
| G | Control_decoy (wrong keys) | **0/10** | 10 invented (followed the decoy) |

**Findings.** (1) **The full-content path is content-agnostic across the code/data divide:**
on a non-code data sibling, blind 0/10 → anchored 9/10. The universal type-blind path
(inject the file's bytes) works when the sibling is JSON, not Python, and the interface is
keys, not signatures. (2) **Causal isolation holds on the config type too** (B − decoy =
0.9): the *specific* config content (the real keys) is the mechanism, not "any
config-shaped context" — the decoy (wrong keys) resolved 0/10, the model following it into
invention exactly as the code decoy did. (3) The one B miss (9/10) is within the n=10
wide-CI margin. **Content-agnosticism is now grounded across three structurally-different
sibling types — Python code (signatures), prose README (full-text reference), and JSON
data (keys) — not by a structural argument alone.** **Measurement caveat (config):**
`resolve_config` counts all string-literal subscripts and `.get()` calls (a slight
over-catch beyond config access), but in a config-focused consumer these are config-key
references; the retained files confirm no trial was mis-classified. Artifacts:
`scratch/spike-xi-content-anchor/results_G_*.json` + retained generated files.

### What this grounds

A content-anchor ADR (candidate **ADR-039**, extending ADR-036's callee dispatch):
**route the produced siblings' API signatures into the callee dispatch context**, sourced
by the framework from the real produced files (never guessed), **regardless of callee type**
— code-generating (`code-generator`) AND prose-generating (`prose-improver`) callees alike,
since the prose arm shows the failure and the fix are identical across both. Conditional
Acceptance, with the real-OpenCode 5-file trajectory re-run as the discharge gate (the
ADR-037/038 pattern) — and the README is a gate criterion, not merely observed, now that
prose coherence is measured (0/10 → 10/10). The mechanism rhymes with ADR-038 (Finding G):
the already-available signal (there, "what remains"; here, "the sibling API") is
computed/on-disk and simply routed forward.

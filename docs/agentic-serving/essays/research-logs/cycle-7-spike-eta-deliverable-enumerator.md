# Spike η — Deliverable Enumerator (general deterministic completeness)

**Status:** PRE-REGISTRATION (recorded before any run; 2026-06-10). Methods-reviewed + revised 2026-06-10 (2 P1 / 5 P2 / 3 P3 applied — see `housekeeping/audits/research-methods-spike-eta.md`); run-ready, pending practitioner go.
**Cycle:** 7, loop-back #7 tail (the ADR-040 DECIDE gate surfaced this).
**Driving question (DECIDE-gate directive):** ADR-040's deterministic completeness gate is
scoped to named-file tasks; the no-files path falls back to the stochastic judge and inherits
the Spike σ false-COMPLETE rate. The practitioner rejected that scope as a thin slice: "if it's
an essential part of the framework it should be deterministic." This spike tests whether the
deterministic gate can be made general, retiring the judge, by predicting the deliverable set
for tasks that do not name files.

## PRE-REGISTRATION

### The reframe the gate response forces

Completeness is deterministic wherever the requested deliverable set is **enumerable**. Named
filenames are the cheapest enumeration; ADR-040's regex is increment one. The no-files path
splits:

1. **Enumerable-but-unnamed** — "build a temperature library with conversion functions, tests,
   a CLI, and docs." A deliverable set exists; it is just not spelled as filenames. This is the
   slice the judge currently owns and false-COMPLETEs on.
2. **Irreducibly semantic** — "summarize this meeting." No enumerable set; "done" means one
   adequate response. Completeness here is not a set question at all — it is "a final response
   was produced," determinable from the model issuing a non-tool-call finish. Adequacy ("is the
   summary good") is ADR-035 / coder scope, never the completeness gate's job.

So the general fix is: **enumerate the deliverable set for case 1; natural-finish for case 2;
the judge is eliminable.** This spike probes case 1, the hard part.

### The design crux (what the spike must probe head-on)

The enumerated set and the produced set must share a namespace for `requested − produced` to
work. With an unnamed task the enumerator must predict deliverable **filenames**, and the
coder must produce *those* names, or the sets never match and the session never converges. So
the enumerator's predicted set is fed forward as a **soft plan** (the deliverable list the
coder is told to produce, and the completeness target). This makes the enumerator a lightweight
planner — the routing-planner direction ADR-038 deferred for re-opening the
planner-confabulation surface (AS-9). The choice to revisit it is deliberate (the gate
directive); the spike's job is to measure whether it holds on the cheap local model.

**What the prior art does and does not license (methods-review P1-B).** Spike ζ validated a
structurally-bounded routing-planner at qwen3:8b for *capability* selection (which of six
ensembles, chosen from a fixed closed set with explicit capability descriptions in the prompt).
Deliverable enumeration is a structurally different decision: open-ended filename prediction
from an unbounded task description, not selection from a closed set. So ζ does **not** transfer
as a rate prior for enumeration, and it is removed from this spike's decision rule. ζ licenses
exactly one thing: AS-9-shaped bounded roles do not *categorically* collapse to confabulation at
qwen3:8b for single-decision tasks. The actual grounding for the enumeration decision is **arm C**
(measured precision/recall on this specific task), nothing inherited.

### Hypothesis

A structurally-bounded deliverable-enumerator, run once on turn 1, predicts the deliverable
filename set an unnamed task implies accurately enough that the deterministic completeness gate
extends to enumerable-but-unnamed tasks, converging the session to the intended deliverables
with no premature finish and no over-run, on the production qwen3:8b coder.

### Failure modes to measure (the confabulation surface)

- **Over-enumeration** — predicts deliverables the task did not request (an extra `utils.py`,
  a `setup.py`). The session then never completes (waits for a file no one wants) until the
  AS-3 cap. This is the new failure the enumerator introduces, the mirror of the judge's
  false-COMPLETE.
- **Under-enumeration** — misses a requested deliverable (drops the README). The session
  finishes early — the σ false-COMPLETE re-enters through the enumerator instead of the judge.
  If under-enumeration is common, the enumerator has not solved the problem, only relocated it.
- **Intent divergence** — the plan terminates consistently but does not match what the user
  asked for (renames, restructures, wrong granularity). The gate would read COMPLETE on a set
  that is internally consistent but wrong.

### Design — LIVE MULTI-TURN PRIMARY (per the practitioner directive)

The live multi-turn run is the primary arm (memory `feedback_live_multiturn_primary_spike`).
Real OpenCode → real `llm-orc serve` → enumerator wired into turn 1 → multi-turn loop, on
**enumerable-but-unnamed** tasks (deliverables described, no filenames). $0 local qwen3:8b.

**Arms** (n committed per methods-review P2-C):
- **A — Baseline (current ADR-040 gate, unnamed task):** the named-file regex extracts nothing
  → judge fallback → expect the σ false-COMPLETE rate. Establishes the gap. **n: run until the
  stall-or-judge-failure is confirmed in ≥3 sessions, then stop** (the σ-baseline discipline —
  it only needs to establish the gap exists).
- **B — Enumerator-as-plan (the heavier fix):** turn-1 enumerator predicts the deliverable set,
  fed forward as the plan AND the completeness target (the coder is told to produce those
  files). Measure per-session convergence to the intended set + clean finish. **n=6** (2 per
  task shape). Early-stop: GROUNDED at 5/6 converge-to-intent; INSUFFICIENT at 3 premature
  finishes before 6.
- **D — Enumerator-seeds-judge-checklist (the lighter fix; methods-review P1-A):** the
  enumerator fills a requested-vs-produced checklist that the **judge reads** (the J-2 shape σ
  identified then bypassed) — the enumerator is NOT the convergence target and does not drive
  the coder's filenames. This decouples "enumerate the deliverables" (a single bounded read)
  from "drive the coder to those exact names" (the fragile coupling). **n=6**, same early-stop.
  The A/B/D comparison is the architecture question: is the enumerator needed at all (A), as a
  plan (B), or only as a checklist-seed the judge reads (D)?
- **C — Isolated enumeration accuracy (the actual grounding; causal isolation):** **n=15**
  isolated enumerator calls (no full session) across the task battery including the boundary
  tasks; score predicted set vs the pre-registered reference set (precision / recall). This is
  what the decision rests on, not ζ.
- **Control (wrong-plan decoy, n=3, mechanism confirmation):** feed a deliberately wrong
  enumerated set. **Interpretation splits two sub-cases (methods-review P3-D):** if the coder
  *follows* the wrong plan → a wrong-but-terminating session (confirms enumeration content is
  the gate's driver); if the coder *deviates* from the wrong plan → a non-terminating / unexpected
  set (informative about coder plan-following, the B-arm fragility). Both readings are recorded;
  the control is not treated as tautological.

**Tasks (methods-review P3-A / P3-B):** the temperature library described without filenames (5
deliverables); a 2-deliverable task (a module + its tests, unnamed); a 3-deliverable task; AND
**at least one irreducibly-semantic task** ("summarize this text") whose expected enumerator
output is empty/null — validating the enumerable-vs-semantic partition (a semantic task
mis-classified as enumerable would loop to AS-3, the worst over-enumeration case). Reference
deliverable sets pre-registered before any run; **matching is basename-stem-flexible** (a
near-synonymous name, `converter.py` vs `converters.py`, scores as a hit; a clearly different
purpose scores as a miss) — the evaluation metric's own namespace-sharing call, stated up front.

### Primary outcome (per-session)

The session produces the intended deliverable set and finishes COMPLETE deterministically, on
an unnamed task, with no premature finish (under-enumeration) and no over-run (over-enumeration)
— binary per session, rate per arm.

### Secondary measures

Enumeration precision / recall (arm C); over- and under-enumeration counts (arms B/C/D);
**intent-divergence flag (pre-registered criterion, methods-review P2-D):** a session diverges
if its produced set differs from the reference set by more than one file in name (after the
basename-stem-flexible allowance) OR a produced file serves a clearly different purpose than the
referenced deliverable. Still a human call, but bound to a stated criterion. **Turns-to-converge
reported as the delta vs arm A (methods-review P3-C)**, not in isolation — a wrong enumerator
prediction costs correction turns, and that cost must be visible.

### Decision rule (pre-registered; ζ removed per P1-B)

The decision rests on **arm C recall** (does the enumerator name the right set) and the
**A/B/D live comparison** (which architecture converges to intent):

- **Recall is the gate.** Arm C recall ≥ 0.9 is required for *either* enumerator architecture to
  proceed — a recall miss (under-enumeration) re-introduces the σ false-COMPLETE through the
  enumerator and **counts as a spike failure** even if precision is high (the asymmetric
  weighting, operationalized). Recall < 0.9 → keep ADR-040's named-file scope as increment one,
  retain the judge for unnamed tasks; the boundary is then *grounded* (measured where
  determinism stops), which itself answers the gate's thin-slice objection (the scope is
  principled, not arbitrary).
- **If recall passes, choose the architecture by the A/B/D comparison.** Prefer **D
  (enumerator-seeds-judge)** if it converges to intent ≥ 5/6 — it is the lighter design (no plan
  coupling, smaller confabulation surface). Fall to **B (enumerator-as-plan)** only if D
  under-performs and B converges ≥ 5/6, accepting the heavier coupling for the gain. Either way,
  **A is the contrast** (expected to show the σ failure).
- **Over-enumeration** is flagged but does not fail the spike unless it causes AS-3 termination
  in > 1/6 arm B (or D) sessions — over-enumeration only delays; under-enumeration finishes
  early and silently, which is the failure that matters.
- A semantic-task enumerator output that is non-empty (arm C boundary task) is a partition
  failure: the enumerator confabulated deliverables for a natural-finish task.

### Alternatives considered and set aside (pre-run; methods-review P2-A / P2-B)

- **Count-only completeness** (track "N deliverables described; N distinct files written →
  COMPLETE", no names): smaller confabulation surface (an integer, not a filename list), but
  set aside because the count-to-files mapping is fragile — the coder may legitimately split a
  described deliverable across files (or combine two), so "5 described, 5 written" can be five
  *wrong* files. Count loses the membership check that catches under-production of a *specific*
  deliverable. Held as a fallback if named enumeration proves too brittle but the count is
  reliable (a possible arm-C secondary read: is the predicted *count* stable even when names
  drift?).
- **Client-side interface contract** (require tasks to name their deliverables, document the
  limitation otherwise): the simplest "fix" — make named-file the contract. Set aside because it
  breaks the cycle's transparent-endpoint north star (the "endpoint is a model" abstraction —
  Population A must not have to annotate requests llm-orc-specifically; AS-10 names capability
  matching from request content alone). Requiring clients to name files is exactly the
  client-side opt-in ADR-026 ruled out. Recorded as the boundary, not pursued.

### Cost

$0 local qwen3:8b (the production coder). A bounded cloud contingency arm (a frontier enumerator
to separate capability-limit from task-infeasibility) is asked-before-spent only if local
enumeration is ambiguous on the under-enumeration measure.

### Honest scope

This measures qwen3:8b deliverable enumeration on small single-hop file-deliverable tasks.
Deeper task shapes, non-file deliverables, and the intent-divergence judgment (which is itself
a human read, not a deterministic check) are recorded boundaries. The spike tests whether the
*completeness* decision can be made deterministic-general; it does not claim the enumerator's
*plan* matches user intent in all cases (that is the irreducible residual the gate response
named).

### Methods review (applied 2026-06-10)

Dispatched to the research-methods-reviewer before any run (Tier-1, per the
Finding-G→ρ→ADR-038 and Finding-H→ξ→ADR-039 precedent). Report:
`housekeeping/audits/research-methods-spike-eta.md` — 2 P1 / 5 P2 / 3 P3, all applied above:
arm D added (P1-A, the enumerator-seeds-judge architecture); ζ decoupled from the decision rule
(P1-B); n + early-stop + asymmetric weighting committed (P2-C); count-only and client-interface
alternatives recorded as set-aside (P2-A/B); intent-divergence criterion pre-registered (P2-D);
semantic boundary task + stem-flexible matching added (P3-A/B); turns reported as a delta
(P3-C); decoy interpretation split into the two coder-obedience sub-cases (P3-D). The most
consequential reshaping: the spike no longer assumes the enumerator must drive the plan — the
A/B/D comparison tests whether it is needed at all, as a plan, or only as a checklist the judge
reads.

## PRE-REGISTRATION — CONCRETE ARM-C BATTERY (recorded 2026-06-10, before any run)

The design above is committed. This section pins the **concrete tasks, reference
deliverable sets, enumerator prompt, and scoring rule** for arm C, recorded before the
first enumerator call. Arm C is n=15 isolated enumerator calls (no live session),
distributed as **5 tasks × 3 reps**. The three enumerable tasks contribute the recall
denominator (5 + 2 + 3 = 10 reference deliverables per rep → 30 deliverable-instances over
3 reps); the two semantic tasks contribute 6 partition checks (predicted output must be
empty / NONE).

### The enumerator prompt (AS-9-bounded, structurally closed output contract)

System:

> You are a deliverable enumerator for an automated coding session. Given a user's task,
> list the files a complete solution must produce — and only those files. Use conventional
> Python project filenames (a module, its test file, a CLI entry point, a README, a
> packaging file) when, and only when, the task implies them. Do not invent deliverables the
> task does not ask for. Output ONLY a filename list, one filename per line, nothing else —
> no prose, no explanation, no numbering, no commentary. If the task asks for a single
> textual answer (a summary, an explanation, an opinion) rather than files to write, output
> exactly: NONE

User: the task text, verbatim.

The `NONE` affordance is the enumerable-vs-semantic partition mechanism (the spike's
case-1/case-2 split, operationalized in the prompt). qwen3:8b emits `<think>…</think>`
reasoning; it is stripped before parsing the trailing filename lines.

### The task battery (tasks held verbatim; reference sets are role sets with accepted stems)

A predicted filename **covers** a reference role iff its purpose matches the role (the
pre-registered basename-stem-flexible rule). Accepted stems are listed per role; a name not
mapping to any role is **over-enumeration** (precision penalty, not a recall miss). For the
two test roles in T1, a test file is assigned by *what it tests* — a bare `tests.py`/`test.py`
covers at most one test role (the still-uncovered one), never both.

**T1 — temp_lib (5 deliverables; the faithful de-named transform of the σ named task):**

> Build a small temperature-conversion library in this directory. It needs: (1) a module
> with three conversion functions — celsius to fahrenheit, fahrenheit to celsius, and celsius
> to kelvin; (2) unit tests for those conversion functions; (3) a command-line tool that
> converts a value given as command-line arguments; (4) tests for the command-line tool; (5)
> documentation explaining how to use the command-line tool. The tests must import the real
> module under test, the CLI must call the real conversion functions, and the docs must
> describe the real CLI usage.

Reference roles (5):
- `converters_module` → converters.py {converter, conversions, temperature, temp, temperatures, temp_conversions}
- `converters_tests` → test_converters.py {test_converters, converters_test}
- `cli` → cli.py {__main__, main, command_line, console, app}
- `cli_tests` → test_cli.py {test_cli, cli_test}
- `readme` → README.md {readme (any ext), docs, documentation}

**T2 — mod_tests (2 deliverables):**

> Write a Python module of string utilities with functions to reverse a string, check whether
> a string is a palindrome, and count the vowels in a string. Also write unit tests for those
> functions.

Reference roles (2):
- `string_module` → string_utils.py {strings, stringutils, utils, text_utils, string_ops, string_tools}
- `string_tests` → test_string_utils.py {test_* matching the module, tests, test}

**T3 — csv_json (3 deliverables):**

> Write a small tool that converts a CSV file to JSON. It needs the converter module itself,
> unit tests for it, and a command-line entry point that takes an input CSV path and an output
> JSON path.

Reference roles (3):
- `converter_module` → csv_to_json.py {converter, csv2json, csvjson, convert, csv_json, csvtojson}
- `converter_tests` → test_csv_to_json.py {test_* matching the module, tests, test}
- `cli` → cli.py {__main__, main, command_line, console, app}

**T4 — summarize (semantic, 0 deliverables — partition check):**

> Summarize the following text in two or three sentences: The mitochondrion is a
> double-membrane-bound organelle found in most eukaryotic cells. It generates most of the
> cell's supply of ATP, used as a source of chemical energy. Mitochondria have their own
> small genome, inherited maternally in many organisms.

Reference: ∅ — expected output `NONE`. A non-empty output is a **partition failure**.

**T5 — explain (semantic, 0 deliverables — partition check):**

> Explain what the following Python function does and whether it has any bugs:
> `def add(a, b): return a - b`

Reference: ∅ — expected output `NONE`. A non-empty output is a **partition failure**.

### Scoring (computed per the pre-registered rule)

- **Recall** = covered reference roles / total reference roles, pooled over the 3 enumerable
  tasks × 3 reps (denominator 30). **Recall ≥ 0.9 is the gate** (per the decision rule above;
  a recall miss is under-enumeration and re-introduces the σ false-COMPLETE — it fails the
  spike even at high precision).
- **Precision** = covered reference roles / total predicted names (secondary; over-enumeration
  is flagged, fails the spike only if it would cause AS-3 in the live arms).
- **Per-task recall** and **per-rep recall** reported alongside the pooled figure.
- **Partition accuracy** = fraction of the 6 semantic reps that output `NONE`. A non-empty
  semantic output is recorded as a confabulation (the worst over-enumeration case).
- **Predicted-count stability** recorded as the count-only secondary read (the set-aside
  count-only alternative's fallback probe — is the predicted count stable even when names
  drift?).

Raw predictions and per-rep gradings are dumped to `scratch/spike-eta-deliverable-enumerator/`
for audit; the stem-matching call is heuristic-with-manual-audit (the same human-bound-to-a-
stated-criterion discipline the intent-divergence flag uses).

## ARM C RESULT — recall gate PASSES (2026-06-10, $0 local qwen3:8b, n=15)

Harness `scratch/spike-eta-deliverable-enumerator/enumerate_probe.py` (3 reps × 5 tasks);
raw outputs in `generated/`, scored JSON `results_armC_reps3.json`.

**The gate (recall):**
- **Strict-automated pooled recall = 0.967 (29/30).** The single automated miss is T1 rep2's
  CLI, named `temperature_cli.py` (the conservative matcher's `_CLI_STEMS` did not list that
  descriptive name).
- **Adjudicated pooled recall = 1.000 (30/30).** Per the pre-registered purpose-match-with-
  manual-audit rule, `temperature_cli.py` (paired with `test_temperature_cli.py`) is the CLI
  role; the smoke run's `tempconvert.py` was the same case. Both are HITs on role purpose.
- **Per-task recall:** T1 temp_lib 0.93 strict / 1.00 adjudicated; T2 mod_tests 1.00; T3 csv_json 1.00.
- Recall ≥ 0.9 on both readings → **the gate passes decisively.**

**Adjudication table (the heuristic-with-manual-audit calls, disclosed):**

| Task / rep | Flagged name | Call | Reasoning |
|---|---|---|---|
| T1 rep2 | `temperature_cli.py` | HIT (cli) | descriptive CLI name; `test_temperature_cli.py` tests it — naming-coordination, not a missing deliverable |
| T1 smoke | `tempconvert.py` | HIT (cli) | same: a temp-conversion command, tested by `cli_test.py` |
| T3 rep1 | `README.md` | OVER (true) | T3 did not request docs — a genuine over-enumeration (the only one in 30 predictions) |

**Partition (the enumerable-vs-semantic split): 6/6 PASS.** Every semantic rep (T4 summarize,
T5 explain) produced zero file deliverables — no confabulated deliverables for a natural-finish
task. Clean-`NONE`-signal rate 3/6: T5 emitted the literal `NONE` all three reps; T4 instead
*answered the summary* (zero filenames) all three. Functionally identical for the gate (empty
set → natural-finish routing), but a **production prompt note**: the enumerator sometimes does
the semantic task instead of signalling `NONE`; the wasted generation is discarded, harmless,
but the `NONE` instruction could be strengthened.

**Secondary measures:** precision 0.935 (the lone true over-enumeration is T3 rep1's README);
over-enumeration rate 1/30 predictions — well under the >1/6-sessions-to-AS-3 fail threshold.
Predicted-count stability strong: T1 [5,5,5], T2 [2,2,2], T3 [4,3,3] — counts match the
reference except T3 rep1's +1, so the count-only secondary read is stable.

**What arm C did and did NOT establish.** Arm C measured *role recall* — does the enumerator
name the right deliverable set. It passed. The strict-vs-adjudicated gap (0.967 vs 1.000) is
**entirely the naming-coordination surface**: the enumerator names the CLI `temperature_cli.py`,
so for the deterministic `requested − produced` gate to converge in a live session the coder
must produce *that* name. That coordination is exactly what arm B (enumerator-as-plan, feeds
names forward to the coder) must close and what arm D (enumerator-seeds-judge, no name coupling)
sidesteps. So the recall result clears the gate but does not pre-decide the architecture —
the live A/B/D comparison does. **Decision per the pre-registered rule: proceed to live arms.**

## LIVE-ARM TURN: the premise does not reproduce (2026-06-10)

Proceeding to the live arms surfaced two things that redirect the spike.

**1. Wall-clock.** Live multi-turn sessions on the 5-file task run ~25–30 min EACH on local
qwen3 — not from model-swapping (an all-14b config removed the seat↔coder swap) but because
14b code-generation on a multi-turn task is inherently slow (~5 min per file-writing turn).
The full pre-registered battery (A≥3 + D6 + control3 = 12 sessions) is 6+ hours. The
practitioner approved a faster path: isolated causal-isolation first, then minimal live (with
MiniMax available for live latency at a small cost).

**2. Arm-A live session 1 converged — no false-COMPLETE.** The first baseline live session
(unnamed task, NO enumerator, local 14b judge, real OpenCode) tracked `REMAINING` through all
five files (`completeness: requested=[]` each turn; monotonic produced 1→5) and declared
`COMPLETE` only at 5/5 (turn 6). OpenCode ended clean. It did **not** reproduce the σ
false-COMPLETE the η premise extrapolated onto the no-files path.

### Isolated judge probe (causal isolation; $0 local qwen3:14b; n=6/cell)

`scratch/spike-eta-deliverable-enumerator/judge_probe.py` constructs the EXACT live judge
prompt (`compose_judgment_message` + `_JUDGE_SYSTEM` + `parse_verdict`) at intermediate
produced-states of the unnamed 5-file task, skipping the slow coder loop (~35s/judge-call).
Two conditions — baseline (`enumerated=None`, the produced-only digest) and armD
(`enumerated=<5 names>`, the framework checklist seeds the judge).

| condition | k=1 (1/5) | k=4 (4/5) | k=5 (5/5) |
|---|---|---|---|
| baseline | false-COMPLETE **0/6** | false-COMPLETE **0/6** | correct-COMPLETE **6/6** |
| armD | false-COMPLETE 0/6 | false-COMPLETE 0/6 | correct-COMPLETE 6/6 |

The judge is genuinely discriminating, not stuck: at k=4 it returns `VERDICT: REMAINING — The
documentation explaining how to use the command-line tool has not been produced` (it names the
one missing deliverable from a produced-only digest — the requested-minus-produced subtraction
σ called unreliable), and at k=5 `VERDICT: COMPLETE` with an accurate summary. **The enumerated
checklist (armD) changes nothing because the baseline has nothing to fix.**

### What this means (and the honest caveats)

The **η premise does not reproduce**: on the unnamed temperature-library task, the bare 14b
judge with the J-1-reframed prompt + the enriched digest correctly tracks completeness (0/12
isolated false-COMPLETE + 1/1 live convergence). σ's measured false-COMPLETE (1/5, 1/4, 1/5)
was on the **named** task in the **live** client context; J-3 then took the named path off the
judge entirely. The judge on the **unnamed** path — the slice η was built to rescue — was never
cleanly measured until now, and it holds.

Caveats: this is ONE task shape in a mostly-clean context; the produced filenames map cleanly
to the described deliverables (the easy match); harder/ambiguous unnamed tasks and the real
client context under task-compaction are where σ-style failures could still live. The live
confirmation (MiniMax, fast) tests that real-context question.

### The decision this reframes

The question is no longer "does the enumerator fix a broken judge" (it is not broken on this
path) but a fork the practitioner owns:

- **(a) Judge-fallback adequate.** The premise that justified generalizing the gate did not
  hold; keep ADR-040's named-file deterministic scope + the (empirically-adequate) stochastic
  judge-fallback for unnamed tasks. The thin-slice objection is answered not by generalizing
  but by evidence the fallback works.
- **(b) Determinism on principle.** The gate objection was *"if it's essential it should be
  deterministic"* — a *working* stochastic judge is still not a *deterministic* gate. The
  enumerator-as-plan (arm B: predict the set → drive the deterministic gate for unnamed tasks
  too) delivers determinism regardless of judge reliability. Arm D (seed the judge) does NOT —
  it keeps the decision stochastic, and the probe shows it is redundant when the judge already
  tracks. So if determinism is wanted, the answer is arm **B**, not arm D.

### Next concrete step (resume, next session)

1. **Build the deliverable-enumerator** — a structurally-bounded turn-1 call (an ensemble or a
   framework-owned model call) that takes the task text and returns a deliverable filename set.
   It plugs in where persist-once's `_extract_requested_deliverables` runs (turn-1 capture), but
   LLM-backed instead of regex. For arm D it seeds the judge digest; for arm B it additionally
   drives the plan / completeness target.
2. **Run arm C first** — n=15 isolated enumerator calls ($0, no live session), including the
   boundary tasks. Score precision/recall against the pre-registered reference sets. **Recall is
   the gate:** recall < 0.9 stops the spike here (keep ADR-040's named-file scope as the grounded
   boundary) before any live-session effort. This is the cheapest decisive arm, so it runs before
   A/B/D.
3. **If recall passes** — restart serve (production 8b config), run arm A (baseline, ≥3 to confirm
   the gap), D (enumerator-seeds-judge, n=6), B (enumerator-as-plan, n=6), control (n=3). Prefer D
   if it converges ≥5/6 (the lighter design).
4. Then finalize ADR-040's scope (general vs named-file-grounded), write the DECIDE gate
   reflection note, close the gate, and do the cycle cleanup (revert timeout-600, remove the
   `completeness:` log + the Arm-B hook, commit).

## LIVE BASELINE — corrected: the premise does NOT reproduce (2026-06-10)

The live baseline went through one false alarm before the real result. Recording both,
because the artifact is instructive.

**False alarm (invalid, retained for the lesson).** A first n=4 baseline run (MiniMax coder
for latency, faithful local qwen3:14b judge) read **3/4 PREMATURE** — apparent reproduction of
the σ false-COMPLETE. It was a **test-harness artifact**: llm-orc derives the session id from
`sha256(first user message)` (`session_registry.resolve_identity`), so four runs with the
identical prompt collided into ONE process-scoped `SessionActionRecord`, and headless
`opencode run` never fires the session-close callback that would clean it. Runs 2–4 saw run 1's
accumulated files (`completeness: produced=[7 files]` while the workspace held one), so the
judge "correctly" said COMPLETE on a polluted record. Only run 1 (fresh record) was valid — and
it converged. (Latent note: the same sha256-first-message identity means two real clients
sending an identical first message would share a session — a real multi-tenancy concern in
llm-orc, out of scope here.)

**Fix + clean re-run.** A unique inert per-run suffix on the prompt forces a distinct session id
→ a fresh record per run. The clean n=4 (distinct ids `03c67482` / `94f2c281` / `deac774a` / …),
reclassified by the authoritative completeness log (produced set at the COMPLETE turn), not the
top-level file counter:

| run | COMPLETE fired at | verdict |
|---|---|---|
| 1 | produced=5 | CONVERGE (clean Python decomposition) |
| 2 | produced=5 | converged-by-count, **intent-divergent** (CLI emitted as `cli/index.js` — JavaScript — with a Python `cli_test.py`; a seat/coder fault, not a judge fault) |
| 3 | produced=5 | CONVERGE (clean) |

**No judge false-COMPLETE in any clean run.** Every clean session reached five distinct
deliverables before the judge returned COMPLETE; intermediate turns returned REMAINING. This
matches the isolated probe (0/12) exactly.

### Corrected conclusion

The η premise — that the no-files (unnamed-task) path inherits the σ false-COMPLETE rate —
**does not reproduce** for the temperature-library task, in either isolated or clean-live
context, with the current judge (J-1 positive-enumeration prompt + the FC-64 enriched digest).
The judge subtracts requested-minus-produced reliably here. The σ false-COMPLETE was measured on
the **named** task in σ's live context (and J-3 then took the named path off the judge); the
unnamed path with the post-σ judge prompt holds. **The judge-fallback is empirically adequate
for this task shape.**

Caveats: ONE task shape; the live-context risk that motivated the spike (later-turn task
compaction degrading the judge's view) did not manifest here, but is not disproven for harder
shapes. A separate, real limitation surfaced: **intent-divergence** (run 2's JS CLI) — the judge
counts deliverables, it does not validate their coherence; neither the judge nor arm D addresses
that (arm B's plan could constrain deliverable type, at the coupling cost).

### What this does to the ADR-040 scope decision

The empirical motivation for generalizing the gate (rescue a failing judge) evaporated. The
decision is now genuinely the determinism fork, and the deterministic boundary turns out to be
*principled, not thin*: determinism is achievable exactly where the task **names** its
deliverables (regex → `requested − produced`); for unnamed tasks there is no deterministic
enumeration (only a stochastic LLM enumeration), so the named-file boundary is the natural edge
of determinism, with the (measured-adequate) judge beyond it. Arm B ("deterministic gate,
stochastic seed") is buildable (arm C proved the enumeration) and gains from the eval/retry
robustness of a single concentrated chokepoint, but it solves a problem that does not currently
manifest and reopens the AS-9 confabulation surface. Practitioner's call.

## BREAK-THE-JUDGE PROBE — the judge IS breakable on realistic shapes (2026-06-10)

Practitioner directive after the clean baseline held: try to break the judge on a harder shape
before committing ADR-040's scope. Isolated judge probe (`judge_probe_hard.py`, $0 local
qwen3:14b, n=6/cell), each condition stressing requested-minus-produced a different way at a
partial produced-state.

| condition | produced | false-COMPLETE | reading |
|---|---|---|---|
| H1_many (8 EXPLICIT deliverables) | 5/8 | 0/6 | HELD |
| H1_many | 7/8 | 0/6 | HELD — names the missing README |
| H2_implicit ("production-ready service") | 3/6 | 0/6 | HELD |
| H2_implicit | 5/6 | **5/6** | BROKE (softly) — under-counts an implicit/ambiguous set (missing `requirements.txt`, itself debatable) |
| H3_compact (compacted task, no deliverable list) | 2/5 | **6/6** | BROKE (hard) — `VERDICT: COMPLETE — Created temperature.py and test_temperature.py` |

**The pattern.** The judge holds when the task **explicitly enumerates** its deliverables —
the 5-file baseline, and H1's 8 explicit files (it even names the missing README at 7/8). It
breaks when the deliverable set is **not in the visible task text**: implicit deliverables
(H2, the judge can't reliably infer the full set, 5/6) and a **compacted task** (H3, no list to
subtract against → it rubber-stamps whatever was produced, 6/6).

**Why this matters for ADR-040.** The clean-baseline "judge-fallback adequate" conclusion was
real but **scope-limited to explicitly-enumerated, full-task cases**. Two realistic shapes break
it: users routinely phrase tasks implicitly ("build a production-ready X"), and long real
sessions get the original ask compacted by the client. So the named-file-only scope is genuinely
too thin — the determinism concern at the DECIDE gate was right, just for harder shapes than the
temperature task that first surfaced it.

**What the enumerator (arm B) does to each break:**
- **Implicit (H2):** the turn-1 enumerator converts the implicit description into an explicit
  deliverable set (arm C: recall 0.967/1.000 inferring sets from descriptions), so the gate
  knows the full set instead of letting the judge under-count. (Residual: the enumerator's set
  is itself a judgment on an ambiguous task — it pins *a* set deterministically, not necessarily
  the user's intended one. The intent-divergence residual, unchanged.)
- **Compaction (H3):** the enumerator runs turn 1 on the full task and persists the set
  (the J-3 persist-once pattern), so later-turn compaction cannot collapse the gate. H3's 6/6 is
  exactly the failure persist-once prevents. This is the compaction-defense strand, now an
  empirically-demonstrated failure, not a hypothesis.

**Honest caveats.** H3 *simulates* compaction by feeding a truncated task; whether the real
OpenCode client actually compacts the original ask depends on conversation length (the short
5-file sessions did NOT compact — they converged). So H3 is a demonstrated *judge* vulnerability
whose *production* incidence depends on client compaction behavior on longer tasks (testable: a
longer live task). H2 is real but partly irreducible ambiguity (the "right" implicit set is a
judgment call). Net: the judge-fallback is adequate for explicit, non-compacted tasks and
unreliable otherwise — which reopens the case for a deterministic enumerate-and-persist mechanism
(arm B) on the unnamed path, with arm C having already shown the enumeration is accurate.

### Correction on H3: the compaction break is structurally non-manifest in production

On reflection the dramatic H3 6/6 needs an architectural qualifier that largely defuses it.
llm-orc derives the session identity from `sha256(first user message)`
(`session_registry.resolve_identity`). The identity was **stable across every turn** of each
clean run (e.g., `03c67482` for all of run 1's turns), which means the first user message — the
full task — persisted unchanged each turn. `_user_task` reads the user messages, so the judge
saw the full deliverable list every turn. The clean runs **prove this empirically**: the judge
returned REMAINING through the partial states and only COMPLETE at 5/5; that is impossible if it
had seen a compacted task (H3 makes it false-COMPLETE 6/6). The task is a small first message
the client keeps for identity; only tool results grow, and those are not in the judge's task
view. So H3 demonstrates a real judge *fragility* (no list → rubber-stamp) but one the
architecture does not expose in normal OpenCode sessions. (Edge case: a pathologically long
session where the client summarizes even the first message would change the session id and reset
the record — a different failure, not in-session compaction.)

### Net synthesis (where the evidence actually lands)

- **Explicit / described tasks — judge robust.** Isolated 0/12, clean live 4/4 converge,
  H1 (8 explicit deliverables) held. This is the common case and the judge handles it.
- **Compaction (H3) — non-manifest.** Structurally prevented (first-message-pinned identity);
  the dramatic 6/6 is a probe artifact of feeding a truncated task the real client never sends.
- **Implicit deliverables (H2) — the one real residual, and it is soft.** The judge under-counts
  at the *ambiguous margin* (5/6 at a debatable `requirements.txt`); it held at 3/6. And the
  enumerator's advantage here is **unproven**: arm C validated enumeration on *described*
  deliverables (temp lib, module+tests, csv→json), not on vague "production-ready X" tasks, so
  there is no evidence arm B beats the judge on exactly the shape where the judge is weakest.

**Bottom line.** Trying hard to break the judge mostly hardened the case that the judge-fallback
is adequate for production-realistic shapes. The named-file deterministic gate (ADR-040) + the
judge-fallback for described tasks is well-supported. Arm B would harden a narrow, soft,
partly-ambiguous residual (implicit-task margins) at real cost (naming-coordination coupling,
the AS-9 confabulation surface) and with **unproven** advantage on that residual. The principled
boundary holds: determinism where the task enumerates its deliverables (named → regex), the
measured-adequate judge where it describes them, and an honest documented soft spot at vague
implicit margins that neither mechanism clearly wins.

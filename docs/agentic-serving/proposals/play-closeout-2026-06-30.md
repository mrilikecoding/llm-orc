# PLAY closeout — the ensemble-focused format (Cycle 7)

**Play window:** 2026-06-29 → 2026-06-30 (the "agent as ensemble" spike arc).
**Mode:** experiential discovery by *building and driving the real system*, not
inhabited role-play. The situated encounter was: try to move the Cycle-7
agentic-serving strategy (the bespoke LoopDriver, ADR-033/036/037/039/041/043)
into a **declarative-ensemble format**, and drive it against the real client.
**Status: PLAY closed.** Feeds a follow-on cycle (the migration), practitioner to
choose the vehicle next session.

This extends the 2026-04-24 field notes (`essays/reflections/field-notes.md`):
those *observed* the frictions (slow turns; file-write narrated but not written;
orchestrator fabrication). This arc **root-caused and fixed** the serving ones
and pushed the composition question much further.

---

## What was explored

Whether the agentic serving strategy can live as *declarative ensembles* on the
L0 engine (per [[declarative_primitives_over_scripts]]) rather than as imperative
adapter/driver Python — and whether that form actually serves OpenCode. Four
threads, each grounded in a runnable spike.

## Field notes (observation → feedback destination)

### 1. The engine can express agentic control flow declaratively — shipped + validated
**Category:** Capability confirmed.
**Observation:** Guard/branch + a bounded `loop:` combinator now ship in L0
(commits `cb87ded`, `9d1a619`, `c5059bf`, `7ceab06`). Ω-loop validated the loop
as a *real* declarative flow (not just unit tests): the Ω-E architect-coherence
repair cycle runs as `loop: {body, until: ${ok}, carry: ${next_input},
max_iterations}` through the real executor. Free deterministic arm: converge via
`until` at iter 2, exhaust at the bound. Paid live arm: the real frontier
architect (qwen3.6-plus) converged inside the loop — including a live
carry-repair (incoherent → gate reject → carried feedback → repaired).
**Provenance:** `scratch/spike-omega-loop/` (README + `validate_loop.py`);
`.llm-orc/ensembles/spike-omega-loop-*`; `.llm-orc/scripts/spike-omega-loop/`;
`proposals/engine-control-flow-state-and-next-steps.md`;
`references/engine-control-flow-primitives.md`.
**Feeds back to:** ARCHITECT/BUILD of the migration — the primitives are the
substrate the migrated strategy composes on.

### 2. The full generalist serving flow is expressible as ONE declarative ensemble
**Category:** Capability confirmed + prediction overturned.
**Observation:** Ω-P3 assembled `resolve-contract (loop) → plan → build
(ensemble + fan_out over the deliverables; inside each: classify → guarded
code-loop / prose branch → marshal) → score` as one ensemble, no Python driver.
Runs free end-to-end. The handoff predicted the wall would be **dynamic
dispatch**; it is NOT hit for the (b) generalist flow — the code-vs-prose
capability choice is a CLOSED set a guard covers. Dynamic dispatch stays the
OPEN-library (c) lever. The real frictions are ergonomic (prose-wrapped chunk
threading, loop-output unwrap, bare-list script output), not missing primitives.
**Provenance:** `scratch/spike-omega-p3/` (README + `run_p3.py`);
`.llm-orc/ensembles/spike-omega-p3-*`; `.llm-orc/scripts/spike-omega-p3/`.
**Feeds back to:** ARCHITECT — the migration architecture (declarative flow is
viable for the whole (b) path; dynamic dispatch deferred to (c)).

### 3. Executional grounding is the real quality wall — and it lives at the loop's accept boundary
**Category:** Challenged assumption (corrects a prior claim).
**Observation:** Ω-P3's paid live run (real frontier architect + cheap-local
qwen3:8b builders) produced a calc package that **false-passed** the execution
gate: the cheap builders shipped a cross-file coherence leak (tokenizer emitted
`kind ∈ {"operator","number"}`, parser expected `'NUMBER'`/`'+'` — a semantic
convention the contract froze *names* but not *values* for), and the
builder-written test swallowed the exception (`try/except print`), so `rc=0` over
non-functioning code. This corrects §2's "the execution gate never false-passed"
— true only over an author-controlled test. **Reframed with the practitioner:** a
gameable gate does not vanish under the outer loop; it *relocates* to wherever
the loop gets its correctness signal. Wrong code is safely "another round" IFF
that signal is independent of the builder (OpenCode running real tests, held/
property/golden checks, user acceptance).
**Provenance:** `scratch/spike-omega-p3/README.md` (Live checkpoint + framing
correction); `scratch/spike-omega-p3/live_run_package/` (the retained
false-passing package); `proposals/ensemble-agent-state-and-next-steps.md` §2 +
§6.2b.
**Feeds back to:** DECIDE — grounded acceptance (an independent/author-held check;
contract-vocabulary pinning). This is the sharpest open (b) item.

### 4. a2 confirmed — real OpenCode serves the ensemble transparently (after 3 real-client fixes)
**Category:** Capability confirmed + challenged assumptions (harness-masked gaps).
**Observation:** Drove the real client (`opencode run` 1.17.11) against the serve.
It decomposed, produced a real module, and OpenCode executed the streamed `write`
tool_call and wrote runnable code to disk — but only after fixing THREE things the
faithful-contract harness (`client_drive.py`) had masked (the
[[validate_against_real_client]] failure mode, live): (a) the serve was
non-streaming — real OpenCode (Vercel AI SDK) needs SSE; (b) OpenCode makes
tool-less auxiliary calls (session title generation) the serve wrongly decomposed
as coding tasks; (c) the 14b+8b config thrashed GPU on the 32GB rig. The
remaining limit is local speed (qwen3 thinking + multi-inference turns), not
correctness. Confirmed the earlier play's "no file written" root cause: streaming.
**Provenance:** `scratch/spike-omega-serve/serve_ensemble.py` (SSE +
`_aux_reply` + 8b paths); `.llm-orc/ensembles/spike-omega-serve/decompose-8b.yaml`
+ `spike-omega-4/agent-turn-omega4-8b.yaml`; `scratch/spike-omega-serve/opencode_run/`;
`proposals/ensemble-agent-state-and-next-steps.md` §2 + §6.4.
**Feeds back to:** confirms question (a); DISCOVER (value tension: interactive
speed on the 32GB rig, per [[target_rig_is_32gb]]); DECIDE (aux-call handling as a
serving requirement).

### 5. The current serve is narrow — build-new-files only, not a general TUI assistant
**Category:** Missing scenario.
**Observation:** Surfaced when scoping "a baseline usable via the OpenCode TUI."
The Ω-serve flow (decompose → produce new files → write) cannot handle general
TUI asks (explain / fix / edit existing / run tests). A daily-driver baseline
needs general behavior; the narrow serve is a research demo of the (b) build path.
**Feeds back to:** ARCHITECT/DECIDE of the migration — general-vs-narrow behavior
is a real design question for the ensemble format.

---

## Disposition

**PLAY closed.** The arc answered its central question: the agentic serving
strategy *can* live as declarative ensembles on the shipped L0 primitives, and it
*does* serve real OpenCode transparently. The remaining walls are not the engine
or the transport — they are **executional grounding** (§6.2b) and **interactive
speed** (§6.3), both (b)-ladder items.

**Next (a follow-on cycle, not this session):** migrate the Cycle-7 validated
serving strategy into the declarative-ensemble format. Research is largely done
(these spikes are the research), so it likely enters at **ARCHITECT/DECIDE**, not
RESEARCH — confirm the entry phase at kickoff against the corpus state. The three
decisions it must own: general-vs-narrow behavior (note 5), grounded acceptance
(note 3 / §6.2b), interactive speed (note 4 / §6.3). Practitioner chooses the
vehicle next session.

## Provenance index

| Thread | Spike artifacts | Config | Commits | Docs |
|---|---|---|---|---|
| Engine primitives | `scratch/spike-omega-loop/` | `.llm-orc/{ensembles,scripts}/spike-omega-loop*` | `cb87ded` `9d1a619` `c5059bf` `7ceab06` | `engine-control-flow-state-and-next-steps.md`; `references/engine-control-flow-primitives.md` |
| Full declarative flow | `scratch/spike-omega-p3/` | `.llm-orc/{ensembles,scripts}/spike-omega-p3*` | (uncommitted, retained) | `ensemble-agent-state-and-next-steps.md` §6.1 |
| Executional grounding | `scratch/spike-omega-p3/live_run_package/` | — | (uncommitted) | `ensemble-agent-state-and-next-steps.md` §2, §6.2b |
| a2 real-client serving | `scratch/spike-omega-serve/` | `.llm-orc/ensembles/spike-omega-serve/decompose-8b.yaml`, `spike-omega-4/agent-turn-omega4-8b.yaml` | (uncommitted) | `ensemble-agent-state-and-next-steps.md` §2, §6.4 |

Spike artifacts are retained (not committed) per [[spike_artifact_retention]] —
hold until corpus close, not cycle close. The four engine commits are in git on
branch `agentic-serving`.

**Entry point for the migration cycle:** `ensemble-agent-state-and-next-steps.md`
(the a/b/c ladder + open items), with `engine-control-flow-state-and-next-steps.md`
for the primitive substrate.

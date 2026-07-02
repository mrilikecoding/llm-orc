# ADR-040: Deterministic Completeness Gate for Named-File Tasks (J-3 + persist-once)

> **Superseded by ADR-045 on 2026-07-01 (Cycle-8 clean-slate collapse, AS-11).** The imperative loop-driver serving architecture is retired; its implementation is removed, not adapted. The behavioral requirement this ADR validated is carried forward to the Cycle-8 declarative target per ADR-045's carry-forward table.

**Status:** Superseded by ADR-045 (2026-07-01); formerly Accepted, **scope finalized as named-file-grounded after Spike η** (Cycle 7 loop-back
#7 tail DECIDE; drafted 2026-06-10, scope finalized 2026-06-10). The named-file *mechanism* is
built, live-discharged (both tiers, 2026-06-10), and audit-clean (argument audit converged R2;
conformance scan CONFORMING). The DECIDE gate had held the *scope* decision open — it rejected
the named-file scope as a settled boundary ("if it's an essential part of the framework it should
be deterministic", the thin-slice objection) and dispatched Spike η to test whether the gate
should generalize to unnamed tasks via a deliverable-enumerator (retiring the stochastic judge).
**Spike η resolved it toward the named-file boundary, now grounded rather than assumed** (see
§Spike η resolution). The enumerator is feasible (arm C: enumeration recall 0.967 strict / 1.000
adjudicated on cheap qwen3:8b), but the premise that justified building it did not hold: the
judge-fallback is **empirically adequate** for the unnamed/described path (isolated 0/12
false-COMPLETE, clean live 4/4 converge), failing only on implicitly-specified deliverables at
the ambiguous margin (soft) and on a compacted task (structurally non-manifest, since llm-orc
pins the session identity to the first user message so the full task persists each turn). The
deterministic boundary is therefore **the principled edge of determinism**: it applies exactly
where the task *names* its deliverables (mechanically recoverable), with the measured-adequate
judge beyond it. That answers the thin-slice objection — the boundary is where determinism is
*achievable*, not an arbitrary carve-out. Empirically grounded by Spike σ, Spike η, and the
2026-06-10 live discharge (see §Empirical grounding and §Spike η resolution); no Conditional
Acceptance. Extends ADR-037 (session termination) and ADR-038 (remaining-work anchor);
supersedes their stochastic-judge path for tasks that name their deliverables.

## Context

Finding I (cycle-status, 2026-06-09): ADR-039's content anchor landed and the dependent
files referenced the real sibling API, but getting a five-file session to actually
*complete* surfaced premature finish. The session stopped after roughly one of five files.

Spike σ (the first live-multi-turn-primary spike, per the practitioner directive that the
live runs are part of the spike) reframed the failure. The dominant premature-termination
mode is not the seat-filler's no-tool-call that F-σ.1 targets; it is the **judge
false-COMPLETE**. ADR-037's termination judge, reading the framework digest on a trailing
turn, declares COMPLETE after one of five deliverables is produced.

The decisive result is that no judge variation improved completion at the sample sizes
measured. The false-COMPLETE rate showed no improvement across prompt or across judge
capability: the cheap qwen3 judge completed 1/5, a J-1 enumeration prompt 1/4, and a frontier
MiniMax-m2.5 judge (Zen, a few cents) 1/5. These arms ran to n≤5, below Spike σ's
pre-registered n=10 (the spike did not reach it; it pivoted to J-3 once the judge was diagnosed
as the wrong fix), so the supported claim is "no improvement detectable at these n," not a
population-level invariance. What carries the decision is the mechanism diagnosis, not the rate:
the bottleneck is the **produced-only digest**. The judge sees the actions taken (what was
written) but never `requested − produced`, and no judge reliably infers the difference. A
better prompt does not recover it (the ψ.2/ω.3b position-over-wording rhyme); a more capable
judge does not recover it (capability cannot supply information the digest does not carry). The
fix has to be structural.

This is the fallback ADR-038 named in advance. ADR-038 rejected a framework-tracked
deliverable checklist as *redundant* for the remaining-work anchor, because Spike ρ.1 measured
the judge naming the unproduced deliverable at 20/20 specific-correct, and it recorded that the
checklist was "held as the fallback if ρ.1 had failed, a judge that mis-named what remains
would have forced a deterministic source." Spike σ is that trigger firing, on a different
decision: the judge is reliable at naming what remains (the ADR-038 anchor content) but
unreliable at the completeness verdict (is everything done). The two are distinct decision
points with distinct judge-reliability evidence.

A note on the diagnostic that shaped the design. Before persist-once was applied, a single
live diagnostic session read the per-turn `completeness:` log and found the gate firing
correctly: the requested set stayed the full five-file set across every trailing turn, with no
client-side truncation. So the deterministic gate was never the silent no-op the leading
hypothesis feared. persist-once (below) is therefore hardening that closes a transient-truncation
path structurally, not a fix for an observed break.

## Decision

**For a task that names its deliverables, the framework decides completeness
deterministically, not with the stochastic judge.** It extracts the requested filenames from
the task and diffs them against the write paths the Session Action Record already holds:

- `requested ⊆ produced` yields COMPLETE (the ADR-037 protocol-clean text-only finish).
- `requested − produced` non-empty yields REMAINING, and that remaining set composes the
  ADR-038 anchor **deterministically** (replacing the judge's one-sentence statement for the
  named-file case).

No judgment-seat call runs on a named-file trailing turn, so the false-COMPLETE failure mode
cannot occur. A task that **names no files** falls back to the ADR-037 stochastic judge and
the ADR-038 judge-statement anchor unchanged (the general-task path). The deliverable set is
extracted by a filename heuristic (`_extract_requested_deliverables`), so the gate is scoped
to exactly the task shape where "requested" is mechanically recoverable.

**persist-once (the hardening).** The requested set is captured on the first turn that names
files (turn 1 carries the guaranteed-full task) and persisted session-scoped in the Session
Action Record with first-non-empty-wins semantics. `_completeness` reads the persisted set
rather than re-deriving it from the per-turn conversation. This removes the verdict's
dependence on the client resending the full task on every per-turn request: a transient
context-compaction or message-shape variance that drops the filenames from one turn's messages
can no longer collapse `requested` to empty and fall the session silently back to the judge.
Empty extractions and repeat turns are no-ops, so a no-files task is never pinned to the
deterministic path, and a later compacted turn cannot clear or shrink a set already captured.

Once the deterministic checklist exists for the verdict, the remaining set is already computed,
so using it as the named-file REMAINING anchor costs nothing further. That is why ADR-038's
redundancy rejection does not apply here: the subsystem ADR-038 declined to build "for no
reliability gain" is now built for the verdict (where σ proves a reliability gain), and the
anchor rides it for free.

### Fitness criteria introduced / amended

- **FC (deterministic completeness verdict, named-file tasks):** a trailing turn on a task
  that names files yields COMPLETE if and only if every requested basename appears in the
  produced write paths, with no judgment-seat invocation. Refutable: a judgment-seat call on a
  named-file trailing turn, or a COMPLETE finish with a requested file unproduced.
- **FC (deterministic remaining anchor, named-file tasks; amends ADR-038):** on a named-file
  REMAINING turn, the call-2 anchor is the framework-computed `requested − produced` set, not
  the judge's statement. Refutable from composed-request inspection.
- **FC (persist-once stability):** the requested set is captured from the first file-naming
  turn and is stable for the rest of the session; a later turn whose task text names no files
  or fewer files does not clear or shrink it. Refutable: a session where the persisted
  `requested` shrinks or empties across turns, or a false-COMPLETE traced to a re-derived empty
  set.
- **FC (no-files fallback preserved):** a task that names no files routes to the ADR-037 judge
  verdict and ADR-038 judge-statement anchor unchanged. Refutable: a no-files task taking the
  deterministic path.

These amend ADR-037's verdict FC and ADR-038's call-2 anchor FC **for the named-file case
only**; both stand unchanged for no-files tasks (see ADR-037 and ADR-038 dated update headers).

## Spike η resolution (the held-scope question)

The DECIDE gate held this ADR's scope open and dispatched Spike η to test whether the
deterministic gate should *generalize* to unnamed tasks via a turn-1 deliverable-enumerator,
retiring the stochastic judge. η ran (`essays/research-logs/cycle-7-spike-eta-deliverable-enumerator.md`,
$0 local; harnesses in `scratch/spike-eta-deliverable-enumerator/`) and resolved the scope
toward the named-file boundary, on this evidence:

- **The enumerator is feasible (arm C).** A structurally-bounded enumerator on cheap qwen3:8b
  named the deliverable role-set at recall 0.967 strict / 1.000 adjudicated, with 6/6
  semantic-partition accuracy. So generalization was *buildable* — it was not ruled out on
  capability.
- **But the premise that motivated it did not hold.** η's purpose was to rescue a judge assumed
  to inherit σ's false-COMPLETE on the no-files path. The isolated judge probe (the exact live
  `compose_judgment_message` at intermediate produced-states, $0 local qwen3:14b) measured the
  bare judge at **0/12 false-COMPLETE** on the unnamed temperature task, correctly naming the
  missing deliverable at 4/5. The clean live baseline (real OpenCode, 14b judge) **converged 4/4**,
  REMAINING through partial states and COMPLETE only at the full set. The judge subtracts
  requested-minus-produced reliably for *described* tasks.
- **σ's false-COMPLETE was a named-task / live-context result, and J-3 already takes the named
  path off the judge.** The unnamed path with the post-σ judge prompt holds; the premise was an
  extrapolation that did not reproduce. (An early live n=4 that read 3/4 false-COMPLETE was
  traced to a *test-harness artifact* — session-id collision on identical prompts bleeding the
  action record across headless runs — not a real judge failure; recorded in the η log.)
- **A deliberate break-the-judge probe located the real limits, and they are narrow.** The judge
  *held* on 8 explicit deliverables; it broke (softly) only at the ambiguous margin of an
  *implicitly*-specified set (5/6 at a debatable `requirements.txt`), and (hard, 6/6) only on a
  *compacted* task with no deliverable list. The compaction break is **structurally non-manifest**
  in this system: the session identity is `sha256(first user message)`, stable across each run,
  so the full task persists in every per-turn request (the clean runs prove the judge saw the
  full task). The one real residual is implicit-margin under-counting, and there the enumerator's
  advantage is **unproven** (arm C validated enumeration on *described* deliverables, not vague
  "production-ready X" tasks).

**Conclusion.** Generalizing the gate would harden a narrow, soft, partly-ambiguous residual
(implicit-task margins) at real cost — the naming-coordination coupling between the enumerated
plan and the coder's filenames, and reopening the AS-9 planner-confabulation surface ADR-038
deferred — with no evidence the enumerator beats the judge on exactly that residual. So the gate
stays scoped to named-file tasks, and that scope is now **grounded**: it is the boundary where
deterministic enumeration is mechanically possible. Beyond it, no deterministic enumeration of an
unnamed task's deliverables exists (only a stochastic LLM one), and the measured-adequate judge
covers the described case. This is the principled answer to the thin-slice objection. The
enumerator path (arm B — enumerate → deterministic gate, persist-once for compaction immunity)
remains a recorded, buildable option if a future cycle finds an implicit/compaction-heavy task
space where the judge demonstrably fails and the enumerator demonstrably beats it; arm C and the
eval/retry analysis (concentrating stochasticity into one verifiable, retriable turn-1 chokepoint)
are the prior art it would build on.

## Rejected alternatives

### Better judge prompt (the J-1 enumeration arm)

Measured, not assumed: σ's J-1 enumeration prompt completed 1/4, no improvement over the 1/5
baseline at n≤5. Asking the judge to enumerate the deliverables does not let it recover what
the produced-only digest never carried. The limit is the digest's information content, not the
prompt's wording (the position-over-wording finding ψ.2/ω.3b already established for a
different composition).

### Frontier judge

Measured: a frontier MiniMax-m2.5 judge completed 1/5, no improvement over the cheap local
judge's 1/5 at n=5. Capability does not recover information the digest does not carry.
Separately, a paid frontier judge on every termination turn contradicts the cheap-local north
star this cycle is built around, so even a hypothetical capability win would buy it at the
wrong cost.

### The deterministic checklist as ADR-038 rejected it

ADR-038 rejected the framework-tracked requested-vs-written diff as redundant for the
remaining-work anchor, on the strength of ρ.1's 20/20 judge-naming accuracy. That rejection
stands for the anchor content. It does not reach the completeness verdict, which is a different
decision and at which σ shows the judge failing without improvement across judge capability (at
the measured n). ADR-040 adopts the
deterministic source for the verdict (the failure ADR-038 named as its trigger) and lets the
anchor ride the now-existing subsystem. There is no contradiction: different decision point,
different judge-reliability evidence, and ADR-038 itself anticipated this fallback.

### Re-derive the requested set each turn (no persist-once)

Workable in the common case: the 2026-06-10 diagnostic showed the gate firing correctly with
per-turn re-derivation, the requested set intact across the session. Rejected because it leaves
the verdict dependent on the client resending the full task every turn. A transient compaction
or message-shape variance would collapse `requested` to empty mid-session and silently fall
back to the judge. This path is the surviving hypothesis for the 2026-06-09 run-2
false-COMPLETE after the truncation hypothesis was tested (the diagnostic) and did not
reproduce; it is not a confirmed attribution, and the run-2 mechanism remains unknown. The
decision to apply persist-once does not rest on that attribution: it rests on the determinism
principle ("limit stochasticity to the ensembles"), which says the completeness decision should
not depend on the client's per-turn message fidelity at all. persist-once removes the
dependence at a one-field cost.

### Keep the judge for named-file tasks, accept the false-COMPLETE rate (defer to PLAY)

Rejected on the Finding-G precedent: a 1/5 completion rate is a north-star parity blocker, not
an experiential-discovery nicety. A serving layer that finishes after one of five requested
files does not reach "all that remains is ensemble iteration." Deferring it would ship a known
gap as if it were an unknown.

## Consequences

**Positive:**
- The false-COMPLETE cannot occur for named-file tasks where the extraction succeeds: the
  verdict is a set comparison, not a model inference. The extraction heuristic is the remaining
  stochastic element (its failure modes, over-extraction loops and under-extraction early
  finish, are scoped to the refutable FC below). Both live discharge runs converged to COMPLETE
  only at turn 6, after all five files were produced, then finished clean and the client loop
  ended.
- One fewer model dispatch per trailing turn on the common coding-task shape (no judgment-seat
  call), and no frontier judge is needed for completeness, which keeps the cheap-local
  economics intact (an implementation read; not separately measured).
- persist-once makes the verdict independent of client message fidelity: a stable
  session-scoped set, immune to compaction or truncation.
- Reuses state the driver already holds (the Session Action Record's produced paths) and the
  task it already has (the requested set). No new role, no new model call, no new edge.

**Negative:**
- Scope is named-file tasks. "Requested" is a filename-regex heuristic; a task that describes
  deliverables without naming files routes to the judge fallback. **Spike η measured that
  fallback and found it adequate for described tasks** (isolated 0/12 false-COMPLETE, clean live
  4/4), so σ's named-task false-COMPLETE does *not* simply carry onto the described path as the
  draft assumed; the judge there fails only at the ambiguous margin of *implicitly*-specified
  sets (η H2, soft) and under task compaction (η H3, structurally non-manifest here). The gate
  helps exactly the shape it can mechanically parse, and the judge covers the described shape
  adequately. The cycle's north-star task shape (multi-file coding tasks that name their files
  explicitly) is the named-file shape, so the deterministic restriction is expected to cover the
  majority of the target envelope, but that coverage fraction is an untested assumption: how much
  of a real deployment's task space names files is not characterized here.
- The heuristic can mis-extract: a filename mentioned but not requested as a deliverable (for
  example "the way converters.py does it") over-counts requested; a deliverable named without a
  recognizable extension is missed. Refutable at the composition layer; the regex is tunable.
  **Spike τ (2026-06-18) hit a sharper form of this.** The original open extension class
  (``[A-Za-z][A-Za-z0-9]{1,7}``) matched module-qualified call expressions in the task prose
  (``step1.step1``, ``base.start``) as phantom deliverables. Unlike a real-but-non-deliverable
  filename, a phantom can never be produced, so ``requested − produced`` stayed permanently
  non-empty and the gate never reached COMPLETE (multi-file sessions churned to the turn cap).
  Resolved by restricting the extension to a recognized set (``_DELIVERABLE_EXTENSIONS``). The
  general coupling that allows it is recorded under §Limitations.
- Completeness here is existence (the file was written), not adequacy (the file is correct or
  runnable). The 8b discharge makes the boundary concrete: all five files existed and COMPLETE
  fired correctly, yet `cli.py` carried a trailing prose paragraph and would not parse (an
  ADR-035 form-gate bleed). Existence-completeness is this gate's scope; content and form
  quality belong to ADR-035 and coder capability, not here.
- A requested set that mis-extracts on turn 1 is sticky for the session (first-non-empty-wins).
  The trade is deliberate: stability against truncation beats per-turn re-derivation, and turn
  1 is the most reliable extraction point. A turn-1 mis-extract is the cost of that choice.
- The named-file REMAINING anchor changes character: it is now the framework-computed
  `requested − produced` set, not the judge's one-sentence statement. ADR-038's Spike ρ
  validated the judge-statement anchor at 19/20 advance; the deterministic anchor was not
  separately validated for advance rate. It rode the two discharge sessions' monotonic
  convergence (1→2→3→4→5 each, no churn), which validates it implicitly, but the lineage is
  worth stating: the 19/20 figure belongs to the judge-statement anchor ADR-038 measured, not
  to this deterministic anchor.

**Neutral:**
- For no-files tasks nothing changes: the ADR-037 judge verdict and ADR-038 judge-statement
  anchor stand. The deterministic and stochastic paths coexist; the task shape selects between
  them at no configuration cost.
- F-σ.1 (the REMAINING-retry seat-leg patch) rides underneath this verdict mechanism unchanged:
  it recovers a seat-filler stall on a REMAINING turn regardless of whether REMAINING came from
  the deterministic diff or the judge.
- The `request_timeout.read` 600 config edit was temporary spike instrumentation, reverted at
  spike close (done 2026-06-10). The same close removed the Spike η enumerator scaffolding (the
  `ETA_ARM`-gated arm-D/control wiring in the Loop Driver) and the Spike σ
  `_resolve_judgment_seat` Arm-B judgment-seat hook in the serving layer, restoring the FC-68
  default (judgment seat = seat-filler model). The diagnostic `completeness:` log was removed at
  that close too, but **Spike τ (2026-06-18) restored it as permanent observability** (no longer
  spike instrumentation): its absence is why the over-extraction non-termination above was
  invisible in the serve logs and slow to diagnose. The gate now logs requested/produced/
  remaining counts and the verdict (or judge-fallback) on every trailing turn.

## Limitations (forward constraints)

The deterministic gate's correctness is coupled to task-prompt phrasing, which constrains future
task and deliverable design. Recorded here (Spike τ, 2026-06-18) because it bounds future work:

- **Coverage boundary.** Completeness is deterministic only for tasks that name file deliverables
  with a recognized extension (`_DELIVERABLE_EXTENSIONS` in `loop_driver.py`). Any other
  deliverable shape (an unlisted extension, or a non-file output) routes to the stochastic judge,
  which Spike σ measured as unreliable at inferring requested-minus-produced. Extending the cycle
  to new deliverable types needs an entry in that set, or a more robust deliverable-declaration
  mechanism than regex-over-prose.

- **Prompt-content coupling (the Spike τ lesson).** The requested set is mined by regex over the
  task text, so prompt content that resembles a filename can corrupt it. The concrete instance:
  the ADR-042 ladder-template call-form fix introduced module-qualified call expressions
  (`step1.step1(x)`) into the task, which the original open-extension regex read as phantom
  deliverables, permanently stalling termination. The narrowed whitelist closes that specific
  case, but the general coupling stands: a task whose prose contains `name.ext` tokens that are
  not deliverables can still mis-extract. Task authors and any future task generator must keep
  deliverable filenames distinguishable from incidental dotted tokens, or the gate mis-counts.
  This means a change to *task wording* (not just to the framework) can silently break
  termination, so the two are no longer independent.

- **Regex over a structured declaration (the deferred durable fix).** The more robust mechanism,
  deferred, is for tasks to declare their deliverables structurally (an explicit list the
  framework reads) rather than the framework mining filenames from prose. Until then the regex
  heuristic is the mechanism and its coupling to prose is a known constraint, not a defect that
  tuning fully closes.

## Empirical grounding (ADR-097 filter)

**Grounding path: spike validation plus completed live discharge, so no Conditional
Acceptance.**

- **Spike σ** (the first live-multi-turn-primary spike): the false-COMPLETE rate showed no
  improvement across prompt (J-1 1/4) or judge capability (frontier MiniMax 1/5, cheap 1/5) at
  n≤5 (below the pre-registered n=10), supporting the produced-only digest as the structural
  bottleneck and the prompt-tuning and capability-escalation responses as ineffective at the
  measured n. The diagnosis, not the rate, is what the decision rests on. Research log
  `essays/research-logs/cycle-7-spike-sigma-premature-finish.md`.
- **The full discharge arc (the complete run record, not the favorable half).**
  Pre-persist-once (`scratch/spike-sigma-premature-finish/j3_deterministic/SUMMARY.tsv`): two
  J-3 runs, **1/2 converged** (run 1 COMPLETE 5 files; run 2 EARLY_COMPLETE at 1 file). The
  2026-06-10 diagnostic then read the per-turn `completeness:` log, found the gate firing
  correctly (`requested=[5]` on every turn, no truncation), and refuted the silent-no-op
  hypothesis. persist-once was applied on the determinism principle (not because a break was
  observed). Post-persist-once (`scratch/spike-sigma-premature-finish/j3_diag/`): **2/2
  converged**, both runs below.
- **Live discharge, both tiers:** a 14b-coder run and the production-config run (14b seat, 8b
  coder) each converged in six turns to a deterministic COMPLETE with all five files,
  `requested=[5]` every turn, monotonic progress, no churn, and a clean client-loop end. The
  8b run also met ADR-039's content-anchor criterion (real sibling-API references, zero
  invention). Evidence: the Spike σ log §RESOLUTION.
- **What the live runs validate vs what only the unit test validates.** The live runs validate
  *nominal* convergence: under normal conditions the requested set stays stable and the gate
  fires COMPLETE only when all files are produced. They do **not** exercise persist-once's
  defended-against failure: no mid-session compaction occurred in either run (`requested=[5]`
  throughout). persist-once's truncation path is validated only by a unit test that *simulates*
  the compaction and proves the gate holds REMAINING from the persisted set. If the real client
  compacts messages differently than the unit test models, persist-once's behavior under live
  compaction is unverified, an open BUILD-watch item.

Because the discharge runs are already done, ADR-040 is not Conditional: the layer-match
discipline ADR-037 and ADR-038 carried (validate against the real client, not the harness
alone) is satisfied at draft time, on the same n=1-session-per-criterion bar those ADRs used
(ADR-040 takes unconditional status at the DECIDE gate rather than carrying a Conditional
through BUILD, because the BUILD code already exists and the discharge runs already passed).
What remains unclosed: the form/adequacy gap (ADR-035), and persist-once's behavior under a
real-client compaction event (validated only in simulation so far).

## Provenance check

Driver-derived: Finding I and the premature-finish symptom (cycle-status, 2026-06-09); Spike
σ's reframing to the judge false-COMPLETE and the no-improvement-across-judges measurements (the σ research log;
cheap 1/5, J-1 1/4, frontier 1/5); the produced-only-digest diagnosis; ADR-038's "held as the
fallback if the judge fails" framing (ADR-038 §Rejected alternatives, the framework-checklist
entry); ρ.1's 20/20 judge-naming accuracy that scoped the ADR-038 redundancy claim to the
anchor; the determinism principle (practitioner, "limit stochasticity to the ensembles"); the
2026-06-10 diagnostic and both live discharge runs (this session; the `j3_diag` artifacts and
the σ log §RESOLUTION); the Finding-G precedent that multi-file parity is essential rather than
PLAY-deferrable (cycle-status §Finding G).

Drafting-time synthesis, labeled as such: the framing that ADR-040 "is the fallback ADR-038
anticipated" (a coherence read connecting σ's verdict-failure to the anchor-failure trigger
ADR-038 named, not a claim made at the ADR-038 gate); the "one fewer model dispatch per turn"
economics (an implementation read, not separately measured); the heuristic mis-extraction
failure modes (a drafting enumeration, not observed in the runs); the existence-versus-adequacy
boundary and its illustration by the 8b `cli.py` form-bleed (a drafting characterization of how
this gate relates to ADR-035, not a measured claim about the gate itself).

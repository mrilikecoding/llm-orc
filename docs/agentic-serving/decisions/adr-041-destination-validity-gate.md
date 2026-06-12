# ADR-041: Destination-Validity Gate — Deterministic Form Protection with Server-Side Recovery

> **Extends ADR-035 (Client-Tool Deliverable Form Contract).** ADR-035 made the
> boundary directive the primary form mechanism and held a *detect-and-refuse*
> gate (ADR-035 §Decision 4) as a speculative escalation, to be installed "if
> PLAY shows residual non-compliance." Spike π grounds that gate as a committed
> mechanism ahead of PLAY: a deterministic destination-validity check catches the
> deterministically-checkable slice of both residual seams (form bleed +
> intent-divergence), and a server-side re-dispatch loop recovers the
> intermittent failures. This ADR promotes the gate from escalation-in-reserve to
> committed protection, and adds the recovery the live arm proved necessary.
> ADR-035 carries an `> Updated by ADR-041` header for §Decision 4 and its
> escalation order.

**Status:** Accepted — split grounding (ADR-097), with two distinct conditions.
The **protection design** is structural and discharged *in principle*: the gate
inspects bytes, it does not trust the model, so its architecture is
model-compliance-independent — that closes ADR-035's form-seam honesty gap by
design, and the corpus + live arm validate it (protection is
degradation-independent — the gate catches whatever the coder produces,
regardless of tier or ollama state). But **discharge of ADR-035's form-seam
Conditional Acceptance is itself contingent on the production install**: the
mechanism is currently env-gated spike code (`LLMORC_SPIKE_PI_GATE=parse`), not
yet in the production path. De-gate, thread `destination_path`, and install at the
FormGate seam are BUILD — until then the protection is proven and validated but
not running by default. The **convergence** guarantee (a real multi-file session
completes all files under the cheap tier) is separately Conditional Acceptance:
server-side recovery rescues intermittent bleeds but is cap-bounded against
persistent ones; the coder-tier escalation lever that closes the persistent
residual (Arm E, in isolation) is named, not built.

**Date:** 2026-06-11 (Cycle 7 loop-back #8 DECIDE)

---

## Context

The loop-back #7-tail σ/η live trajectories converged real multi-file sessions
but left two residuals on the hardest file (`cli.py`), both adequacy failures the
ADR-035 boundary directive did not prevent:

- **The form bleed** (Spike σ Run B): the cheap qwen3:8b coder trailed a prose
  paragraph after the code, so the `.py` deliverable did not parse —
  `SyntaxError`, an unrunnable file.
- **The intent-divergence** (Spike η run 2): the coder emitted JavaScript into a
  `.py` destination — a file that is structurally wrong for what its path claims.

ADR-035 §Negative recorded the honest gap that produced both: the form contract
is "model-compliance-dependent, not hard-enforced" — the framework guarantees the
*presence* of the destination-keyed directive, but relies on documented model
compliance to *produce* the form. The cheap tier complies most of the time
(Spike χ, n=4 first-try) but not always, and ADR-035 §Decision 4 anticipated
exactly this: a detect-and-refuse gate at the Artifact Bridge / FormGate seam
(FC-57), held in reserve for PLAY.

The practitioner directed tackling **both seams** rather than carving out the
adequacy seam ADR-035's text disclaims, engaging the research-methods-reviewer,
and running an informative spike before PLAY. Spike π
(`essays/research-logs/cycle-7-spike-pi-form-adequacy-gate.md`; methods review
`housekeeping/audits/research-methods-spike-pi.md`, 2 P1 / 3 P2 / 2 P3 all
applied) tested the central hypothesis:

> A single deterministic destination-validity gate — "does the deliverable
> parse/validate as what its destination path claims?" — covers the
> deterministically-checkable slice of *both* seams, because trailing prose AND
> within-file wrong-language both break `ast.parse`. The parse/validity edge is
> the principled determinism boundary (the η rhyme: the gate is deterministic
> exactly where the artifact's destination type is structurally checkable); the
> irreducibly-semantic "parses-but-wrong" slice is left for PLAY.

### What the spike established

**Corpus arm** (Forks 1/2/4 — 12-item labeled set, deterministic, $0; research
log §CORPUS RESULT). Against pass-through, fence-only, and marker-detection
candidates, the **parse-check gate was the only viable detector**: it caught all
five deterministic failures (form bleed C1 + wrong-language C4 + syntax-bug C5),
produced **zero false-positives** on the four legitimate-but-tricky files (a
README carrying a `bash` fence, a `.py` whose natural-language docstring contains
"note that"/"you can"), and passed all three semantic-residual variants
un-flagged. The heuristic gates were not just weaker but *brittle* in ways the
parse-check is not: fence-only missed the actual σ bleed (bare trailing prose, no
fence) and false-positived on the README; marker-detection false-positived twice
and missed the wrong-language slice. The unification held exactly: the parse-check
miss-set on deterministic failures was ∅, its false-positive set was ∅. Fork 2
confirmed the deterministic boundary sits at parse/validity, with the semantic
residual genuinely PLAY territory.

**Live arm** (Fork 3, the primary arm, n=5×2 — production qwen3:8b coder /
qwen3:14b judge-seat, real OpenCode 1.15.5 → working-tree serve; research log
§LIVE-ARM RESULT). A smoke session surfaced a structural blocker the corpus could
not see (the σ/η validate-against-real-client lesson again): FC-57's
refusal-as-`stop` *ends the OpenCode loop*. ADR-040's recovery assumption —
re-delegate the still-missing deliverable on the next trailing-tool-result turn —
is unreachable, because a `stop` produces no next turn; OpenCode treats it as a
normal COMPLETE finish and terminates. **The detect-and-refuse gate protects but
cannot recover at the client surface.** Recovery has to happen *server-side*,
inside the serving turn. A bounded server-side re-dispatch was built to make Fork
3 answerable at all (it is part of this ADR's mechanism, not a detour). The full
n=5×2 then split the result cleanly:

- **Protection — unambiguous and confirmed live.** Across all 5 gated sessions,
  **zero invalid files reached the client.** The baseline shipped invalid files
  in 3/5. Every form-bled deliverable was caught; the two that could not be
  recovered degraded to a refusal, never a shipped broken file.
- **Recovery — partial, bounded by the 2-retry cap.** Server-side re-dispatch
  rescued *intermittent* bleeds (the coder re-sampled a valid file within the cap
  → session converged: runs 1/2/5) but exhausted on *persistent* ones (runs 3/4:
  the 8b coder produced invalid output all three attempts on one file → the gate
  held the line, the session ended short). A measurement note, stated honestly:
  the pre-registered decision rule, applied *literally* to "all-files-produced-
  valid," returns **recovers** (B = 5/5 ≥ 3/5, B−A = 3/5 ≥ 2/5, re-dispatch
  control 3/5 > 0). But that reading is partly vacuous — runs 3/4 score clean only
  because they ended at 3 and 1 files. The honest application of the rule's spirit
  ("does the gate converge a real trajectory?") is *converged-and-all-valid*: **B
  = 3/5 vs A = 2/5**, margin +1/5, convergence margin (≥2/5) not met — the
  **protects-but-does-not-recover** outcome.

**Arm E** (coder-tier contrast, free-first < $0.05; research log §ARM E). The
pre-registered escalation lever was a frontier *seat-filler*, but the live arm
located the residual in the *coder* (the code-generator ensemble's qwen3:8b
produces the bled file), not the seat — so the lever was reshaped to a MiniMax
*coder* on fresh ollama (the degradation control). Two conclusions: (1) the
marathon-degradation confound is **dismissed** — fresh-8b bled `cli.py` at the
same ~50% rate (3/6 valid) it did late in the live run, so the exhaustions were
genuine cheap-tier unreliability, not a run artifact; (2) **coder-tier escalation
is the right lever, confirmed in isolation** — MiniMax produced valid `cli.py`
6/6 where fresh-8b managed 3/6, and is fast (~1.2s ping vs ~10s local). This is an
n=6 single-file isolated probe, not a wired-into-the-loop session: it establishes
that the persistent-bleed residual is *coder-capability-bound* and that escalating
the coder tier (the ADR-014 Calibration Gate cheap→escalated lever) addresses it
on the hardest file. Session-level convergence under a coder-tier escalation
*wired into the serving loop* is BUILD validation, not yet run.

This composes the full escalation picture: **deterministic gate (protection,
tier-independent) + cheap-tier server-side recovery (rescues intermittent bleeds)
+ coder-tier escalation (the lever for persistent bleeds, confirmed in isolation).**
The residual after all three is the irreducibly-semantic "parses-but-wrong" slice
— handed to PLAY. That residual is *narrower* than what ADR-035 disclaimed to
PLAY: the gate now takes the deterministically-checkable slice of the adequacy
seam (wrong-language, syntax-broken), leaving PLAY only the parses-but-wrong
portion.

**Reconciling with ADR-035's "a hard form-guarantee is neither available nor
required."** ADR-035 argued the form contract could be the lighter
boundary-directive mechanism *because* a wrong-form deliverable surfaces as a
client-rejectable diff (the client executes every `write` through its own
permission gate) — so hard prevention was "not required." This ADR commits a
deterministic gate, which could read as contradicting that. It does not, on
ADR-035's own terms: ADR-035 §Decision 4 already held the detect-and-refuse gate
in reserve and stated that "on a surface without client-side execution
affordances, the detection-gate backstop would be warranted from the start." The
"not required" claim was conditional on the gate being *available as escalation*,
which Spike π now makes it — committed and deterministic — at a cost (one additive
`destination_path` thread) low enough that holding it in reserve is no longer the
better choice. The bounded-failure-cost argument still holds (it is why the gate
can stay lighter than schema-retry); what changed is that the gate's low
commitment cost now makes committing it preferable to reserving it. The
practitioner's directive to tackle both seams before PLAY is the priority call
that moved the gate from reserve to committed; the bounded-failure-cost argument
is not overturned, it is the reason the gate could stay lighter than schema-retry.
This reconciliation is a framing the argument audit surfaced; it is recorded here
and flagged for the gate, where the practitioner can confirm or re-weigh the
priority call.

## Decision

**For deliverables bound for a client tool, a deterministic destination-validity
gate at the marshalling boundary refuses to emit any deliverable that does not
parse/validate as what its destination path claims; a bounded server-side
re-dispatch loop recovers intermittent failures before the refusal would end the
client loop; the cheap→escalated coder tier is the lever for persistent
failures.** The gate inspects bytes deterministically — it does not trust the
model — so it makes ADR-035's *protection* structural rather than
model-compliance-dependent.

1. **Deterministic destination-validity gate (the protection — structural).**
   When the Artifact Bridge marshals a deliverable for a client tool, the gate
   validates the marshalled content against the destination *path's* claimed type
   before emission: `.py` must `ast.parse`; `.json` must `json.loads`; other
   destinations (notably `.md` / prose) pass un-inspected. On a validity failure
   the gate *refuses* — it raises through the FormRefusedError channel (FC-57),
   and the Client-Tool-Action Terminal degrades the refusal to a dispatch-failure
   completion rather than emitting a broken `write` (the FC-57 zero-Terminal-edits
   property — the seam already existed for exactly this). The gate **never
   attempts heuristic extraction** from multi-fence or prose-framed output (Spike
   χ F-χ.1 / the corpus Fork-1 result reject that path as fragile); it only
   *recognizes* a structurally-wrong deliverable.

2. **The seam extension: `destination_path`, not just `destination_tool`.** The
   marshalling boundary must carry the full destination path (e.g.
   `src/foo/bar.py`) alongside the FC-57 `destination_tool`; the gate derives the
   *extension* from it at validation time (`os.path.splitext`) and dispatches on
   that (`.py` → `ast.parse`, `.json` → `json.loads`). The field is the full path,
   not a pre-extracted extension, so the same thread serves any future
   extension-keyed check. This is the one structural extension the gate requires;
   it is additive at the bridge `marshal` signature and the FormGate contract, and
   the env-gated spike code already prototypes it.

3. **Bounded server-side re-dispatch recovery (the convergence helper).** A
   client-facing refusal-as-`stop` ends the OpenCode loop, so recovery cannot
   wait for ADR-040's next-turn re-delegation (there is no next turn). Instead,
   after the Loop Driver delegates a generation whose deliverable is bound for a
   client tool, it parse-checks the resolved content (inline or substrate-routed,
   resolving through the Session Artifact Store the same way the bridge does) and,
   on a failure, **re-dispatches the same destination within the serving turn**
   (the coder re-samples) up to a bounded cap. The terminal's FormGate stays the
   *final arbiter*: cap exhaustion degrades to the dispatch-failure `stop` — the
   protect-but-not-converge floor. Re-dispatch reuses the delegation path (not a
   full re-decision) so the action is not double-recorded.

4. **The retry cap is deliberate, not a default.** The cap converts a *persistent*
   cheap-coder failure into an honest *protected-but-not-converged* outcome that
   routes to escalation, rather than an unbounded retry loop against a coder that
   bleeds ~50% of the time even fresh (Arm E). Recovery rescues the intermittent
   case; it does not paper over a capability gap. The spike ran the cap at 2;
   the exact value is a BUILD tuning parameter, not a load-bearing commitment.

5. **The escalation ladder for the form/adequacy seam — deterministic trigger,
   opt-in ceiling.** When the deterministic slice is exhausted: (a) the gate is the
   **protection floor** — tier-independent, always retained; (b) server-side
   recovery **rescues intermittent bleeds** at the cheap tier; (c) the
   **cheap→less-cheap→frontier coder ladder** (ADR-014 Calibration Gate) is the
   lever for **persistent bleeds** — the lever is *coder* capability, not the
   pre-registered frontier seat (the live arm redirected it), confirmed in
   isolation (Arm E, n=6 on the hardest file) but not yet wired into a converging
   session; (d) the irreducibly-semantic "parses-but-wrong" residual is **PLAY**
   territory, not a gate's job. The ladder's **cost story** (practitioner, gate):
   the free-local rungs (cheap → less-cheap, e.g. qwen3:8b → qwen3:14b) may
   escalate deterministically on a persistent-bleed signal; the **frontier rung
   crosses a cost boundary and is opt-in** — configured at a layer, or consented
   before a run — with a **local-degradation path** (when frontier is not opted
   in, the ladder caps at the best free rung and a persistent bleed there yields
   the honest short session, the protection floor). So the *trigger* (parse-check
   failure → escalate) is deterministic while the *ceiling* is a policy choice,
   preserving the cycle's free-first standard. FC-51 `TurnDecision` instrumentation
   already distinguishes a wrong-*form* turn (this gate) from a wrong-*action* turn
   (driver/split) and a wrong-*content* turn (the semantic-coherence seam ADR-035
   left to PLAY).

6. **Coverage boundary: structurally-checkable destinations only (the principled
   edge).** The gate is deterministic exactly where the destination type admits a
   structural validity check (`.py`, `.json`). Prose destinations (`.md`) pass
   un-inspected — prose form is not structurally checkable, and a
   parseable-but-wrong prose deliverable is the same semantic residual PLAY owns
   regardless. This is the parse/validity determinism boundary, the η rhyme: the
   framework is deterministic where determinism is *available*, and hands the rest
   to experiential validation rather than faking a check it cannot ground.

### Why server-side recovery, not client-side re-delegation (the smoke finding)

The runbook's pre-registered "recovers" path assumed `refuse → un-produced →
re-delegate next turn` — the ADR-040 completeness gate would notice the file is
still missing and re-delegate it. The smoke session refuted this *structurally,
not stochastically*: FC-57's refusal emits `finish_reason=stop`
(`client_tool_action_terminal.py`), OpenCode reads `stop` as a normal COMPLETE
finish and **terminates the session** — there is no next turn for the completeness
gate to fire on. One session was sufficient to establish the cause (deterministic
code path). Recovery therefore has to live *server-side*, inside the serving turn,
before the refusal would reach the client. This is the design input the smoke arm
bought; it is consistent with the cycle's determinism commitment — the framework
controls recovery deterministically rather than relying on a client-loop
continuation it does not own.

### Why the lever is the coder tier, not the seat (Arm E redirected it)

The pre-registration named a frontier *seat-filler* as the escalation lever
(ADR-033 §6b). The live arm located the residual in the *coder*: the
code-generator ensemble's qwen3:8b produces the bled file; the seat-filler decides
*to delegate* and *which destination*, but the bytes come from the ensemble. A
frontier seat would not move this needle. Arm E confirmed the coder-tier lever
directly in isolation — MiniMax coder 6/6 valid vs fresh-8b 3/6 on the exact
persistent-bleed file (`cli.py`), n=6. That is enough to establish *where* the
lever is (coder capability, not seat) and that it works on the hardest file; it
is not a wired-session convergence test, so the escalation order names the
**coder** tier (the ADR-014 cheap→escalated Calibration Gate lever) as the
evidence-located lever, with session-level convergence under it left to BUILD.
This is an outcome-grounded redirection of the pre-registered seat lever, not a
prediction.

### What this discharges, and what stays conditional (ADR-097)

The split is the honest claim, and it bears directly on what ADR-035's
Conditional Acceptance bought:

- **Protection — structural by design; discharge contingent on the install.**
  ADR-035 §Negative's "honest gap from the cycle's structural-not-model-assumed
  ideal" is closed *in design for protection*: the gate does not trust the model
  to produce the right form; it inspects the bytes and refuses the wrong ones. No
  invalid deliverable reaches the client — a structural guarantee, grounded by the
  corpus (catch 5/5, FP 0) and the live arm (0 invalid across 5 gated sessions).
  This is the determinism upgrade the gate buys. The one honest qualifier: the
  mechanism is env-gated spike code today, so ADR-035's form-seam Conditional
  Acceptance is *design-discharged but not install-discharged* — the production
  path does not run the gate until the BUILD de-gate. The design obligation is
  closed; the install obligation is the first BUILD item.
- **Convergence — Conditional Acceptance.** That a real multi-file session
  *completes all files* under the cheap tier is **not** guaranteed. Recovery
  rescues intermittent bleeds; persistent cheap-coder failures exhaust the cap.
  The lever that closes them (coder-tier escalation) is named but **not built**.
  Pending validation, designated for BUILD / first deployment:
  - The de-gate-and-install at the FormGate seam (drop `LLMORC_SPIKE_PI_GATE`,
    thread `destination_path`, install the gate as the bridge's `form_gate`).
  - Convergence under the coder ladder (cheap → less-cheap → frontier) wired to the
    ADR-014 Calibration Gate, with the **frontier rung opt-in / cost-gated** and a
    local-degradation path (Arm E proved the lever in isolation, n=6; not wired
    into the live loop).
  - **The long-horizon regime (the north-star target).** The north star is a
    sustained, RDD-like multi-step flow, not a single 5-file trajectory. The gate
    and recovery are validated only on single trajectories (n=5×2); sustained
    form-compliance, the recovery cap's behaviour over many turns, and the
    structure/quality separation's legibility across a long flow are the axis-2
    validation regime (ADR-033 §6b axis-2; ADR-035's "sustained form-compliance
    over long multi-turn trajectories"), designated for PLAY / first deployment.
  - Recovery-rate sharpening on fresh ollama (the live 3/5 carries a low-but-real
    degradation confound on the precise rate, dismissed for protection).
  - `edit`/`bash` destination validity and the cap value at scale.

Generation form (the model producing bare bytes on the first try) remains
model-compliance-dependent — the gate *rejects* wrong form, it does not *produce*
right form. What changed is that wrong form is now caught structurally instead of
shipped.

**A PLAY observation target the spike did not resolve (argument-audit-surfaced;
sharpened at the gate).** The gate changes *where* a persistent failure surfaces:
under pass-through the client sees a broken-file diff (rejectable through its
permission gate); under the gate it sees either a recovered-and-clean session (the
common intermittent case) or a *cap-exhausted short session* (the persistent case
— fewer files than asked, no broken file). The right PLAY question is not "is a
short session nicer" but, framed against the north-star target (a **long-horizon,
RDD-like multi-step flow**, not a single 5-file trajectory): **does the
structure/quality separation read as honest to a user watching a sustained flow?**
A short session that says "the structure delivered what it could validate; this
destination needs a more capable coder" is the legible floor; a broken file that
looks delivered is not. PLAY should observe (a) the semantic "parses-but-wrong"
residual (the detection gap), (b) whether the structure/quality separation stays
legible over a long-horizon flow (the axis-2 regime — the gate and recovery were
validated on single trajectories, n=5×2, not sustained flows), and (c) the
recovery cap's behaviour across many turns. FC-51 `TurnDecision` diagnostics
distinguish the turn types that produce each.

### Relationship to prior ADRs

- **ADR-035 (form contract) — promotes its reserved escalation; partial update.**
  ADR-035 §Decision 4 named the detect-and-refuse gate as defense-in-depth held
  for PLAY ("if PLAY shows residual non-compliance"). ADR-041 promotes it to a
  committed mechanism grounded *before* PLAY (Spike π corpus + live arm), and adds
  the server-side recovery ADR-035 did not anticipate (ADR-035 assumed a refusal
  degrades to a dispatch-failure completion full-stop; the live arm showed that
  ends the loop). ADR-035's boundary directive stays the *primary* form mechanism
  — the gate is the deterministic backstop that catches what the directive does
  not. ADR-035 carries an `> Updated by ADR-041` header for §Decision 4 and the
  §Conditional Acceptance escalation order. The rest of ADR-035 (boundary
  directive, destination-agnostic ensembles, granularity invariant) is unchanged.
- **ADR-034 (client-tool-action terminal + artifact bridge) — composes, no
  change.** The gate installs at the FormGate seam ADR-034/035 already named
  (FC-57); the terminal already degrades `FormRefusedError` to a dispatch-failure
  completion with zero Terminal edits. ADR-034's fidelity FC (FC-49) is preserved
  — the gate refuses or passes content through, it never paraphrases or extracts.
- **ADR-040 (deterministic completeness gate) — sibling determinism gate at a
  different temporal layer.** ADR-040 and ADR-041 are both deterministic framework
  gates over the delegation seam (completeness: requested − produced; validity:
  parses-as-claimed), but they operate at *different layers* and compose cleanly
  rather than competing. ADR-041 acts **within** a serving turn, before emission;
  ADR-040 acts **across** turns, on trailing tool-result tails. The three cases:
  (i) **recovered** (runs 1/2/5) — server-side re-dispatch produces a valid file
  *within* the turn, the valid `write` emits, and the turn completes normally;
  ADR-040 then fires on the next turn exactly as usual, sees the now-produced file
  in the `produced` set, and advances the `requested − produced` anchor to the
  next file. Because re-dispatch reuses the delegation path (not a full
  re-decision), the action is recorded once, so ADR-040's `produced` set and
  persist-once `requested` set are designed to see a single clean write. The spike
  evidences this at the *session* level — the recovered runs (1/2/5) converged to
  all 5 files (research log §"Recovery built + validated live") — but a direct
  trace of the ADR-040 completeness diff on a recovered turn (confirming the
  single-write record) is the BUILD verification item below, not something the
  spike instrumented. (ii)
  **valid-on-first-try** — ADR-041 is a no-op pass-through; ADR-040 operates as
  designed. (iii) **cap-exhausted** (runs 3/4) — the refusal degrades to a `stop`
  that ends the session, so ADR-040 never gets a next turn. ADR-040 only ever sees
  produced-valid files in `produced` because an invalid file is either recovered
  before emission or never emitted. The "non-overlap" is therefore a layer
  separation (within-turn validity vs across-turn completeness), not a claim that
  the two never co-occur; BUILD verifies the recovered-case Session Action Record
  write satisfies ADR-040's completeness diff (carried as a BUILD verification
  item).
- **ADR-033/036 (loop driver / delegation) — composes.** Recovery reuses the Loop
  Driver's delegation path; the gate is keyed where the Loop Driver already
  chooses the destination tool and path. No new module, edge, or responsibility —
  the mechanism is composition within the existing Loop Driver and FormGate seam
  (an ARCHITECT-skip candidate, consistent with loop-backs #6/#7; the
  `destination_path` thread + gate install are a BUILD Design Amendment).
- **ADR-014 (calibration gate / tiered escalation) — the named convergence
  lever.** The cheap→escalated coder tier is ADR-014's mechanism; ADR-041 names it
  as the lever for persistent bleeds and leaves the wiring to BUILD.
- **AS-9 / determinism commitment — consistent.** The gate is deterministic
  framework control over essential protection (a wrong-form file never lands),
  not an LLM judgment — the same determinism-over-carve-outs principle ADR-040
  applied to completeness, applied here to validity.

## Rejected alternatives

### Client-side recovery via ADR-040's next-turn re-delegation

Let the refusal degrade to a dispatch-failure (ADR-035 §Decision 4 as written) and
rely on the ADR-040 completeness gate to re-delegate the still-missing file next
turn.

**Rejected because:** the smoke session showed FC-57's refusal emits
`finish_reason=stop`, which OpenCode reads as a COMPLETE finish and uses to *end
the session* — there is no next turn for the completeness gate to fire on
(research log §LIVE-ARM SMOKE FINDING). The recovery must be server-side, within
the serving turn. This is a structural property of the client loop, established
deterministically.

### Heuristic detection (fence-only, marker-detection) or extraction/normalization

Strip a fenced block, detect prose-scaffolding markers, or normalize the
marshalled output into bare bytes.

**Rejected because:** the corpus arm quantified the determinism-principle distrust
of the heuristic pole. Fence-only missed the actual σ failure (bare trailing
prose, no fence) and false-positived on a legitimate README (a `bash` fence);
marker-detection false-positived twice (a correctly-documented `.py` whose
docstring reads naturally; the README) and missed the wrong-language slice.
Parse-check was the only gate with FP = 0 that caught both seams. Extraction from
unconstrained output is the fragile pole ADR-035 already rejected (Spike χ
F-χ.1); the corpus re-confirmed it. The heuristic gates are not merely weaker —
they are brittle in ways parse-check is not (their catch rides on the model's
phrasing habits; parse-check's catch is a property of the content).

### Unbounded retry / retry-until-valid

Re-dispatch the refused deliverable until it parses.

**Rejected because:** Arm E showed the cheap coder bleeds ~50% even on fresh
ollama (3/6 valid on `cli.py`) — an unbounded loop against a persistent failure
would not terminate, and would mask a real capability gap as latency. The bounded
cap converts the persistent case into an honest protected-not-converged outcome
that routes to coder-tier escalation. Recovery is for the *intermittent* case; the
*persistent* case is a tier decision, not a retry count.

### `output_schema` reject-and-retry or a typed `submit`-slot at the synthesizer

Validate the synthesizer output against a schema and reject-and-retry, or give the
ensemble a typed `submit(content=…)` slot.

**Rejected because:** ADR-035 already rejected both (schema-retry as heavier than
the evidence warrants and at the wrong layer; the typed slot as
destination-coupling that erodes ADR-025 reusability — a slot guarantees a *slot*,
not the form of the content *in* the slot, since fences appear inside the
argument). The corpus premature-narrowing check (methods-review target #6) was run
specifically to confirm these rejections still hold against the gate evidence;
they do. The parse-check gate validates the *destination form* deterministically
without coupling the ensemble to file-production.

### Frontier seat-filler escalation (the pre-registered Arm E lever)

Escalate the seat-filler to a frontier model on persistent bleeds.

**Rejected because:** the residual is in the coder, not the seat (the seat decides
*to delegate* and *where*; the ensemble produces the bytes). Arm E, reshaped to
the coder, confirmed the coder lever directly (MiniMax coder 6/6 vs fresh-8b 3/6).
The frontier seat would not move the form residual. Recorded as a pre-registered
lever the evidence redirected.

## Consequences

### Positive

- **Protection keeps the structure/quality boundary legible (the north-star
  value).** No invalid deliverable reaches the client — the gate inspects bytes
  and refuses, closing ADR-035's model-compliance-dependent gap for protection
  *in design* (the install is BUILD, per the Status), and the guarantee does not
  depend on tier or ollama state (it catches whatever the coder produces). But the
  decisive value is not "sparing the user a bad diff" (the client's permission
  gate already makes a broken file rejectable — ADR-035's "not required"
  argument). It is that the gate **enforces the separation between structural
  delivery and ensemble quality**: whatever reaches the client is structurally
  valid, so any remaining failure is unambiguously an *ensemble-quality* failure —
  which is the only thing the north star ("all that remains is ensemble iteration
  and improvement") says should remain. A shipped broken file smears a quality
  miss into a delivery miss and destroys that separation; the gate preserves it.
  The routing-signal reading (the gate converts a silent failure into an explicit
  escalation signal) is a *consequence* of this separation, not a competitor to
  it — the gate routes to escalation precisely because it refuses to blur the
  boundary.
- **Both seams covered by one mechanism.** Trailing prose and within-file
  wrong-language both break `ast.parse`; the gate's unification (corpus Fork 2,
  miss-set ∅) means one deterministic check covers what looked like two problems.
  Evidence is uneven across the two: the form-bleed seam is confirmed under live
  trajectory conditions (Fork 3); the wrong-language seam is confirmed by the
  corpus arm (deterministic, the η-captured real case re-pathed) but did not recur
  in the live n=5×2. The unification is sound — both break the same parser — but
  the live-arm evidence carries the form-bleed seam, not both.
- **Intermittent bleeds self-heal.** Server-side recovery converges the common
  intermittent case without a client round-trip or a wrong-form diff for the user
  to reject (runs 1/2/5).
- **The escalation ladder is explicit and outcome-grounded.** Gate → recovery →
  coder-tier escalation → PLAY, each rung evidence-placed, with the lever
  corrected to the coder tier by Arm E.
- **Composes within existing structure.** The FormGate seam (FC-57), the terminal
  degradation path, and the Loop Driver delegation path all pre-exist; the
  mechanism is composition plus one additive `destination_path` thread.

### Negative

- **Convergence under the cheap tier is not guaranteed.** Persistent cheap-coder
  bleeds exhaust the recovery cap and end the session short (protected, not
  converged). The closing lever (coder-tier escalation) is named but unbuilt —
  the Conditional Acceptance.
- **Coverage is the structurally-checkable slice only.** Prose/`.md` destinations
  pass un-inspected, and a `.py` that parses but is semantically wrong is not
  caught — that is the irreducibly-semantic residual handed to PLAY. The gate
  protects against *un-parseable* and *wrong-language*, not *parses-but-wrong*.
- **Generation form remains model-dependent.** The gate rejects wrong form; it
  does not produce right form. The improvement is that wrong form is now caught
  structurally instead of shipped — but the underlying cheap-tier form
  unreliability is unchanged (Arm E: ~50% on the hardest file).
- **The retry cap adds latency on the recovered path.** A rescued bleed costs one
  or two extra dispatches in the serving turn before the valid file emits.

### Neutral

- **The gate is invisible to the client.** A protected session looks like an
  ordinary `tool_calls` trajectory; an exhausted one looks like a normal finish
  that stopped short. The gate and recovery live entirely server-side.
- **The cap value (2 in the spike) is a BUILD tuning parameter**, not a
  load-bearing commitment — it trades recovered-intermittent-rate against
  serving-turn latency.
- **The `destination_path` thread is the only structural surface change** — small,
  additive, and the BUILD seed the env-gated spike code already prototypes.

## Provenance check

- **The two residual seams (form bleed; intent-divergence)**: Spike σ Run B + Spike
  η run 2 live trajectories (drivers; cycle-status loop-back #7-tail; research logs
  `cycle-7-spike-sigma-premature-finish.md`, `cycle-7-spike-eta-deliverable-enumerator.md`).
  Driver chain: loop-back BUILD live runs.
- **The cross-seam unification (one parse-check covers both)**: Spike π corpus Fork
  2 (driver — miss-set ∅, FP ∅, residual un-flagged; research log §CORPUS RESULT).
  Driver chain: loop-back DECIDE-entry spike.
- **Parse-check is the only viable gate (vs fence/marker)**: Spike π corpus Forks
  1/4 (driver — FP 0 vs 1 vs 2; catches the σ bleed the heuristics miss). Driver
  chain: loop-back DECIDE-entry spike.
- **Protection confirmed live (0 invalid across 5 gated sessions; baseline 3/5)**:
  Spike π live arm Fork 3 (driver; research log §LIVE-ARM RESULT; data
  `scratch/spike-pi-form-adequacy-gate/B_live`). Driver chain: loop-back primary
  spike arm.
- **Server-side recovery required (refusal-as-`stop` ends the loop)**: Spike π
  smoke finding (driver — `client_tool_action_terminal.py` `stop` + OpenCode
  STEP_FINISH stream; research log §LIVE-ARM SMOKE FINDING). Driver chain:
  loop-back primary spike arm (smoke pre-flight).
- **Recovery partial / protects-but-does-not-recover (B 3/5 vs A 2/5 converged)**:
  Spike π live arm n=5×2 (driver). Driver chain: loop-back primary spike arm.
- **Coder-tier escalation is the lever, not the seat; degradation confound
  dismissed**: Spike π Arm E (driver — MiniMax coder 6/6 vs fresh-8b 3/6 on the
  exact task; research log §ARM E). Driver chain: practitioner-pre-authorized
  contingent spike arm.
- **The parse/validity determinism boundary (η rhyme); prose as semantic
  residual**: Spike π Fork 2 + corpus caveat #2 (driver) composed with the η
  named-file determinism-edge precedent (ADR-040, prior ADR). Driver chain: spike
  + prior ADR.
- **ADR-035 §Decision-4 promotion; the directive stays primary**: ADR-035 §Decision
  4 + §Conditional Acceptance escalation order (drivers, prior ADR) read against
  the Spike π grounding. Driver chain: prior ADR + spike.
- **Coder-tier lever = ADR-014 Calibration Gate; convergence Conditional
  Acceptance shape**: ADR-014 cheap→escalated mechanism (driver, prior ADR) +
  ADR-097 grounding filter applied to the protects-but-does-not-recover evidence
  (drafting-time scoping judgment, spike-grounded). Driver chain: prior ADR +
  design-time synthesis.
- **The split Status (protection structural / convergence conditional)**:
  drafting-time synthesis applying ADR-097 to the cleanly-split live-arm result —
  the practitioner-flagged non-overclaiming framing at DECIDE entry. Design-time
  scoping judgment grounded in the Fork 3 split, not a single driver finding.
- **Protection-as-structure/quality-separation framing**: practitioner framing at
  the DECIDE gate (driver — north-star derivation, 2026-06-12). Composed with the
  argument audit's gate-as-routing-signal alternative (Framing A) and ADR-035's
  "not required" argument; the resolution centers protection and treats
  routing-signal as a consequence.
- **The opt-in / cost-gated frontier rung + local-degradation path on the
  escalation ladder**: practitioner framing at the DECIDE gate (driver — "the
  frontier / cost model could be opt-in at a layer or before," 2026-06-12),
  composed with ADR-014's tiered escalation and the cycle's free-first standard.
- **The long-horizon RDD-like flow as the north-star target / axis-2 validation
  regime**: practitioner framing at the DECIDE gate (driver — "north star [is] a
  long-horizon RDD-like flow," 2026-06-12), composed with ADR-033 §6b axis-2 and
  ADR-035's sustained-trajectory Conditional Acceptance.

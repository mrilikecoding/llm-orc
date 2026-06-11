# ADR-037: Session-Termination Mechanism — Two-Call Trailing Composition

> **Updated by ADR-038 on 2026-06-08.** The FC (call-2 form preservation) is
> amended: on a REMAINING verdict, call 2 carries the judge's remaining-work
> statement plus a framework imperative as a next-step anchor — it is no longer
> byte-equal to the E4b composition. The rest of this ADR (judgment-first
> composition, bare-form judgment, digest provenance, COMPLETE protocol-clean
> finish, AS-3 backstop) remains current. Motivated by Finding G (multi-file
> sessions did not advance because the judge's computed "what remains" was
> discarded before call 2). See ADR-038.

> **Updated by ADR-040 on 2026-06-10.** For tasks that name their deliverables,
> the COMPLETE/REMAINING verdict is now computed deterministically (requested
> filenames vs produced write paths), not by the bare-form stochastic judge —
> Spike σ measured the judge false-COMPLETE-ing after one of five files, with no
> improvement across prompt or judge capability at the measured n (the
> produced-only digest is the bottleneck no judge recovers from). The bare-form
> judgment call still runs for tasks that name no files (the general-task
> fallback), so the judgment-first composition, digest provenance, and the
> COMPLETE protocol-clean finish all remain current on that path. See ADR-040.

**Status:** Accepted, Updated by ADR-038 and ADR-040 (Cycle 7 loop-back #5 DECIDE, gate
closed 2026-06-05; Conditional Acceptance per ADR-097 **discharged 2026-06-08**
jointly with ADR-038 by the WP-LB-L real-OpenCode run — a single multi-file
session that converged [COMPLETE → text-only finish, client loop ended] and, via
ADR-038's anchor, advanced through both deliverables; evidence
`scratch/wp-lb-l-acceptance/RESULTS.md`; FC call-2 form preservation updated by
ADR-038 2026-06-08)

## Context

Finding F (cycle-status §"BUILD-surfaced finding: termination suppression,"
2026-06-04): the WP-LB-I acceptance run met ADR-036's gating condition —
and never finished. Every no-new-task tool-result tail delegated another
revision of the same file until killed. Spike ψ″ characterized the cause as
**F-ψ″.3, a two-sided composition tension**: the C3 trailing delegation
guidance is simultaneously termination-suppressing on work-complete tails
(E1 0/10 finish vs E2 10/10 without guidance) and delegation-carrying on
work-remaining tails (E4b 9/10 delegate vs E4a 0/10 without). Wording
cannot resolve it (E3 completion clause: 1/10 finish — the ψ.2/ω.3b
position-over-wording rhyme). Distinguishing the two tail shapes requires
knowing whether work remains.

The DECIDE-entry analysis (Spike θ research log §"Entry analysis")
established the shaping constraint: a work-complete tail and a
work-remaining tail are identical in framework-visible kind — a tool-result
tail with no new user message. Task-completeness is semantic. No
deterministic source survives examination (task-text parsing is semantic
judgment in disguise; the capability ensemble sees only its one dispatch
brief per ADR-035's granularity; client tool results report per-action
success, not task completeness). **The work-remaining signal, if it exists,
is a model judgment.** This collapsed the candidate space: conditional
composition and two-call composition merge once the signal is model-judged;
a framework termination policy reduces to consequence-enforcement because
the framework has no termination input to compute and no trustworthy text
to return when overriding a tool-calling response (the model is the stop
mechanism — the client ends the loop on a no-tool-calls response).

Spike θ (pre-registered, methods-reviewed before any run — 2 P1 / 4 P2 /
3 P3 applied; research log
`essays/research-logs/cycle-7-spike-theta-termination-mechanism.md`)
measured the judgment call across two rounds:

- **Round 1 (information-starved record): both forms fail, on opposite
  sides.** With only what the client request carries ("Wrote file
  successfully", no paths, no content), the bare judge refuses to confirm
  completion (0/10 on work-complete — honest, useless) and the in-session
  judge confabulates it (3/10 and 6/10 on work-remaining — claiming
  unwritten files were created). One root cause: the discriminating
  information is absent (F-θ.1).
- **Round 2 (framework-owned digest + deliverable-accounting standard):
  both forms pass.** Per-action file paths (production derives them from
  the framework's own emitted tool calls) and an explicit standard ("a
  successful write of a requested file counts as produced; you are not
  verifying code correctness") moved the same model on the same bases to
  59/60 with genuine discrimination from both forms (see F-θ.2 below for
  the round-1 per-form comparison). Form A-enriched 30/30; Form B-enriched
  29/30 with one false-continue at the pre-registered ≤1/10 threshold. **Form B-enriched
  adopted per the pre-registered cost tiebreak** (no client prompt on the
  judgment call; context bounded regardless of session depth).
- **F-θ.2 (the load-bearing finding): the judgment's accuracy lives in its
  evidence base and standard, not in the model's unguided disposition.**
  Same model, same bases: round 1 produced no usable form (Form A 19/30,
  failing toward confabulated COMPLETE; Form B 20/30 where all 20 came from
  degenerate uniform-REMAINING — the rule-5 rebuild trigger, not
  discrimination); round 2 produced 59/60 with genuine discrimination from
  both forms, on digest + standard changes alone. The framework cannot
  compute task-completeness, but it can guarantee what the completeness
  judgment gets to see.
- **Portability annotation** (hosted secondary arms, read after the local
  verdict per the pre-registered discipline; annotation only — the local
  arms carry the verdict): the identical composition scored 20/20 on
  `zen:minimax-m2.7` at 0.7–3.0s/call (vs 7–19s local). One pair does not
  establish portability; it establishes the composition is not
  qwen-idiosyncratic — a structural contrast with ADR-036's V3 lever, which
  demonstrably does not transfer (ψ′ Arm D).

## Decision

**Trailing tool-result tails get a two-call composition: a framework-
composed termination judgment first, then — only on a REMAINING verdict —
the ADR-036 guidance-composed action call. The framework guarantees the
judgment's evidence base and enforces its consequences; the verdict itself
is model-rendered within a structurally-bounded role with AS-9's
role-shape — noting that the termination-judgment surface is a new instance
AS-9's recorded empirical basis does not cover (Spike θ is its
instance-level evidence; the domain model records the extension — see
Amendment Log). The model remains the stop mechanism: a COMPLETE verdict
produces the clean text-only turn that ends the client loop.**

1. **Termination judgment (call 1).** On every trailing turn (tool-result
   tail, no new user task), the Loop Driver first dispatches a
   **bare-form judgment call**: a framework-authored judge system message,
   plus one user message carrying the original user task (quoted as data),
   the **framework-owned action digest**, and the deliverable-accounting
   question. No tools are offered (the expected response is text; a
   tools-less request does not break the turn — smoke-checked directly).
   The client's system prompt is not included: the judgment call is
   framework ↔ model, outside the client's attention contest entirely.

2. **Framework-owned digest (the evidence-base guarantee).** The digest's
   action records derive from the framework's own records — the client-tool
   calls it emitted (grounded carry or delegation → Artifact Bridge →
   Client-Tool-Action Terminal) joined with the client's per-call tool
   results — never reconstructed from client-serialized messages alone
   (Round 1 measured exactly that reconstruction's failure: the client
   serialization drops what was written). Each record carries the action
   kind, target file path, and result.

3. **Deliverable-accounting standard.** The judgment question asks whether
   requested deliverables have not yet been produced, and states that a
   successful write of a requested file counts as produced. Code
   correctness is explicitly out of the judgment's scope — it is owned by
   the capability ensemble, the calibration gate, and PLAY. (Round 1's
   bare form demonstrated that without this standard, an honest judge
   demands unattainable verification forever.)

4. **Branch enforcement.** **COMPLETE** → the judgment response, with its
   `VERDICT:` line stripped, is returned to the client as the assistant
   turn. It carries no tool calls, so the client ends the loop —
   termination is protocol-clean and needs no fabricated text (θ.3
   finish-text quality: brief factual summaries, returnable as-is).
   **REMAINING** → the Loop Driver makes one second call composed exactly
   as ADR-036's trailing form (C3 standalone trailing guidance; the E4b
   composition); the judgment exchange is discarded — it does not ride into
   call 2's context. First turns and trailing turns carrying a new user
   task are untouched (ADR-036's merge branch stands unchanged).

5. **Scope of the validated claim.** Measured: file-write deliverable
   tasks, one to three deliverables, tool-result tails to depth three,
   qwen3:14b judgment seat (29/30 across three bases), minimax-m2.7
   annotation (20/20 on two bases). The composed mechanism estimate —
   work-complete tails finish at ~0.9; work-remaining tails delegate at
   ~0.9 (judgment 1.0 × ADR-036's E4b 0.9) — is **composed from
   independently-measured n=10 arms, not an end-to-end measurement**.
   Tasks whose deliverables are not write-shaped (explanation, command
   sequences) are outside the measured scope: the judgment's question may
   not be answerable by deliverable accounting there — the boundary is
   recorded, not guessed across (the ADR-036 denominator-boundary
   pattern). The spike's digests used constructed path annotations
   (capture bytes carry none); the production join is BUILD work and the
   acceptance-gate target.

6. **Termination observability.** Every judgment emits a TurnDecision-
   family event carrying the turn shape and verdict, so termination
   behavior — including false-continue frequency — is computable from
   events alone. This extends the event shape WP-LB-J consumes (the
   finish-policy event the loop-back #5 entry package anticipated), and
   lifts the soak distortion: ADR-036's ≥0.9 delegation-rate soak window
   becomes readable once this mechanism lands, with judgment-suppressed
   phantom revisions no longer inflating the numerator's turn stream.

7. **Judgment-seat re-validation.** The judgment rate is a per-profile
   empirical property. A judgment-seat model change requires re-validated
   judgment rates before the swap is trusted — composing with ADR-036's
   profile-swap re-validation FC rather than duplicating it (one
   re-validation event covers both seats when they share a profile; the θ
   harness arms are the re-validation instrument). The portability prior
   differs from ADR-036's — a bounded counting task should transfer where
   an attention-contest lever did not, and the hosted annotation supports
   this — but a prior is not a measurement; the FC stands.

### Three-layer scope-of-claim

Named at the practitioner's gate-entry question (is reliability a
paid-vs-local scale, or an architectural guarantee paid merely speeds up?):

- **Layer 1 — hard framework guarantees (model-independent):** the digest's
  provenance and content, the judgment standard, the branch enforcement,
  the protocol-clean finish shape, and the AS-3 turn cap as the absolute
  ceiling on non-termination. Deterministic; no model property involved.
- **Layer 2 — the verdict, model-rendered within a bounded role:** no
  framework can compute task-completeness; the framework instead bounds
  the role (counting deliverables against explicit evidence) into the
  structural shape AS-9 names as reliable — as a new instance of that
  shape, evidenced by Spike θ itself, not by AS-9's recorded basis.
  Reliability here is structural bounding plus a measured per-profile
  rate — 29/30 on qwen3:14b (the verdict-bearing primary evidence); the
  one-pair hosted annotation (20/20) is supporting context at a different
  evidence level, not a second reliability measurement — not a guarantee.
- **Layer 3 — paid vs local, orthogonal to reliability:** at measured n,
  hosted and local accuracy are indistinguishable; hosted buys a ~10×
  latency reduction on a call that runs on every trailing turn. Local
  qwen3:14b passing at 29/30 keeps the mechanism inside the cycle's
  cheap-tier cost-distribution thesis; the hosted seat is an optimization
  slot with the local path as the standing fallback (the practitioner's
  paid/local scoping principle).

### Fitness criteria introduced

- **FC (judgment-first trailing composition):** every trailing-tail
  interaction begins with the framework-composed termination judgment; a
  trailing-tail dispatch that goes straight to guidance-composed action
  generation violates this. Refutable from composed-request inspection.
- **FC (digest provenance):** the digest's action records derive from the
  framework's own dispatch/emission records joined with client tool
  results — a digest reconstructed from client-serialized messages alone
  violates this. Refutable by code inspection and test.
- **FC (finish protocol cleanliness):** a COMPLETE verdict yields a
  text-only client response (no tool calls) with no `VERDICT:` line.
  Refutable from the emitted response.
- **FC (call-2 form preservation):** a REMAINING verdict yields exactly one
  action call composed per ADR-036's trailing form, with the judgment
  exchange excluded from its context. Refutable from composed-request
  inspection.
- **FC (termination observability):** finish-vs-continue behavior on
  trailing turns is computable from emitted events alone (turn shape +
  verdict). Refutable: a deployment where false-continue frequency cannot
  be computed without log archaeology violates this.
- **FC (judgment-seat re-validation):** a judgment-seat profile change is
  accompanied by a recorded judgment-rate re-validation — operationally,
  a run of the θ-harness judgment arms (the θ.3b/θ.4b/θ.4b′ compositions)
  at the registered thresholds. Composes with ADR-036's profile-swap FC:
  when the judgment seat and the action seat share a profile, one
  re-validation event means one recorded run covering **both instruments**
  (ADR-036's ψ′-harness delegation arms AND the θ-harness judgment arms);
  if the seats are split across profiles, re-validation is per-seat with
  the instrument matching the seat. Refutable: a profile change with no
  recorded re-validation covering the affected seat's instrument.

## Rejected alternatives

### Restructured guidance text

Triple-wounded before this loop-back and confirmed dead in it: E3's
completion clause restored 1/10 finish; ψ.2 showed wording strength does
not move the rate; ω.3b showed text-level demotion does not remove
authority. Position carries authority; wording does not modulate it. No
composition of the guidance text can be simultaneously termination-
permitting and delegation-carrying.

### Drop the trailing guidance (drop-C3)

Refuted by ψ″ Arm E4 before this DECIDE entered: without trailing guidance
on a work-remaining tail the seat-filler never delegates (0/10) — it writes
inline (6/10, the Finding B north-star violation) or finishes prematurely
(4/10). The guidance is load-bearing for mid-task delegation; removing it
trades a termination bug for a delegation collapse.

### Deterministic framework termination policy

The thesis-consistent pole, belief-mapped deliberately (the loop-back #3
discipline) rather than dismissed: what would need to be true is (i) a
framework-computable termination input and (ii) a trustworthy text-only
response when the framework overrides a tool-calling turn. Neither holds —
task-completeness is semantic (every deterministic source examined fails),
and the prior forks' thesis wins (single-step enforcement, directive
presence, batch truncation) were all properties the framework could
compute. What survives is consequence-enforcement — the framework enforces
which call gets guidance and which response returns — and that is exactly
where this decision places the framework. The AS-3 turn cap remains the
deterministic backstop beneath the mechanism.

### Conditional composition on a deterministic work-remaining signal

Collapses into this decision: the signal does not exist deterministically,
so "compose guidance only when work remains" requires a model judgment of
work-remaining — which is the two-call composition.

### Implicit judgment (call 1 = unguided dispatch)

Computable from measured arms without new runs: complete tails converge
(E2 10/10) but mid-task delegation composes to ~0.54 (E4a 6/10 continue ×
E4b 9/10 delegate — composed estimate) with 4/10 of mid-task tails dropping
work via premature finish. Note E4a's 6/10 continues are themselves all
inline writes (0/10 delegation) — the ~0.54 is the rate after call 2
re-shapes them, not the implicit dispatch's own delegation rate, which is
zero. The explicit question beats the implicit signal decisively on the
work-remaining side (10/10 and 10/10 vs 6/10) at the cost of the same one
extra call. Rejected on measured rates, not principle.

### In-session judgment form (Form A-enriched)

Passed round 2 at 30/30 — rejected on the pre-registered cost tiebreak,
not on accuracy: the client prompt (28k chars) rides on every judgment
call and the context grows with session depth, against the bare form's
bounded ~1–2k tokens. At n=10 per arm, 30/30 vs 29/30 is not a
distinguishable accuracy difference. Recorded as the measured fallback
form: if BUILD surfaces a bare-form failure the spike could not see, Form
A-enriched is validated at the same evidence level and swaps in at the
composition point — knowingly reintroducing the client-prompt processing
cost on every judgment call (the reason B won the tiebreak).

### Ship-as-is with the turn cap as terminator

Sessions burn the full AS-3 cap on every completed task (the measured
zombie-revision shape: ~4 min/turn local until killed), the north-star
surface visibly fails on convergence, and ADR-036's soak window stays
unreadable because phantom revisions inflate the delegation numerator. The
cap is a circuit breaker, not convergence; it is retained as the backstop
beneath this mechanism, not the mechanism.

## Consequences

**Positive:**
- Sessions converge: work-complete tails finish at ~0.9 (composed estimate
  from independently-measured n=10 arms, not an end-to-end measurement)
  with returnable finish text, against the current never-terminates. The
  residual 1/10 false-continue costs one extra revision turn and faces
  ~0.9 termination probability on the next trailing turn — geometric
  decay, which is drafting-time arithmetic from a single-base n=10
  observation (the multi-turn decay itself is unmeasured), bounded
  absolutely by the AS-3 cap.
- Mid-task delegation is preserved at ADR-036's measured level (judgment
  10/10 on both work-remaining bases; call 2 is the unchanged E4b
  composition at 9/10).
- The delegation-rate soak (ADR-036 decision 3) becomes readable; the
  TurnDecision extension gives WP-LB-J the finish-policy event shape the
  loop-back #5 entry package anticipated, and false-continue frequency is
  itself meterable.
- The judgment call escapes the client-prompt attention contest
  structurally: no contest to win means no V3-style (composition × model)
  fragility surface — supported by the 20/20 hosted annotation.
- Layer 3 stays optional: local cheap tier carries the mechanism (29/30);
  a hosted judgment seat is a ~10× latency optimization with the local
  path as standing fallback.

**Negative:**
- Every continuing trailing turn costs one extra model call (7–19s
  local) — the judgment call on the REMAINING branch precedes the action
  call. On work-complete tails the judgment call replaces (not adds to)
  the action call. Hosted seating reduces this to ~0.7–3.0s at
  ~$0.0015/call if adopted.
- The judgment rate is a per-profile measured property: every judgment-
  seat model change carries a re-validation burden (FC). The portability
  prior is favorable but unproven beyond one hosted pair.
- Non-write-shaped deliverables are outside the measured scope; the
  deliverable-accounting standard may not bound the judgment there. The
  boundary is recorded; widening it is future spike work, watched by the
  termination-observability events in the meantime.
- **The digest's expressiveness is the mechanism's reliability ceiling**
  (practitioner pre-mortem at the gate, 2026-06-05): a judgment over a
  meta-record that cannot represent what completeness means for a complex
  session — multi-part asks, mid-session intent refinement, deliverables
  that are not file writes — degrades exactly the way round 1 measured.
  The framework records; the model reasons over the record; neither
  substitutes for the other. The committed digest (action kind + path +
  result) is the **first increment of an extensible meta-record seam, not
  its final form**; the false-stop share (termination-observability FC) is
  the extend-on-evidence trigger for digest enrichment, and the digest's
  home and shape are an ARCHITECT allocation question.
- The spike validated the composition with constructed path annotations;
  the production digest join (framework records ↔ client results) is
  unbuilt — it is the BUILD work and the Conditional Acceptance gating
  condition. A join defect would degrade the judgment exactly the way
  round 1 measured.
- The framework re-acquires a system message of its own — scoped to the
  bare judgment call only, where the client's prompt is absent by
  construction. ADR-036's "no framework system message" consequence is
  thereby narrowed to action-generation calls composed with the client
  conversation (see the partial-update header on ADR-036).

**Neutral:**
- ADR-036's FC (directive-in-user-turn presence) is scoped by this
  decision to **action-generation calls**: the judgment call carries no
  delegation guidance by design (it generates no actions), and call 2
  retains the FC unchanged. The first-turn merge branch is untouched.
- The judgment question text is tunable at the same evidence bar as the
  guidance text (wording revisions re-validate affected arms — the FC-58
  discipline applied to a second composition point).
- WP-LB-J's sequencing decision (build after the mechanism settles) is
  vindicated: the event shape it consumes now includes the finish-policy
  fields this decision introduces.

## Empirical grounding (ADR-097 filter)

**Grounding path: spike validation.** Spike θ — pre-registered design,
methods-reviewed before any run (2 P1 / 4 P2 / 3 P3 applied pre-run),
two rounds with the round-2 rebuild itself pre-registered before its runs,
full-n denominators, pre-registered decision rule passed (round 2, Form
B-enriched), hosted secondary arms read only after the local verdict
(P2-D discipline). **Conditional Acceptance: the gating condition is the
BUILD acceptance run at the real-client layer** — a real-OpenCode session
in which (a) a completed task's session **converges**: the finish turn
lands as text-only and the client loop ends, and (b) a work-remaining
trailing turn still **delegates**, both verified from serve-log evidence
(`dispatch start` / TurnDecision events — the WP-A scar's
layer-match discipline: a passing-looking run can be model-direct). The
production digest join is part of the same gate: the acceptance run's
judgment calls must be fed by the framework's own dispatch records, not a
spike-style constructed digest. The trailing soak confirmation
(ADR-036 decision 3) becomes the joint mechanism's trailing confirmation
once readable.

## Provenance check

Driver-derived: Finding F and F-ψ″.3 (cycle-status finding section; ψ″ run
+ E4 records); the candidate-collapse analysis and its deterministic-source
examination (Spike θ research log §Entry analysis, practitioner-confirmed
at entry); all round-1/round-2 rates and F-θ.1/F-θ.2 (Spike θ research log
+ per-run records); the E4a/E4b rates the implicit-variant rejection
composes (ψ″ Arm E4); the wording-is-not-the-lever chain (E3, ψ.2, ω.3b);
the model-is-stop-mechanism protocol constraint (loop-back #5 entry
package, item 2a); the two-call candidate itself (practitioner-discussed
post-session reframe, 2026-06-05, recorded in the entry package before
this loop-back entered); the deliverable-accounting standard's exclusion
of code correctness (round-1 evidence + ADR-035/calibration-gate
ownership); the paid/local scoping principle (practitioner, loop-back #4);
the re-validation FC pattern (ADR-036 decision 4).

Drafting-time synthesis, labeled as such: the three-layer scope-of-claim
framing (composed at the gate exchange in response to the practitioner's
paid-vs-architectural question; the layers organize measured facts but the
taxonomy is drafting-time); the geometric-decay characterization of the
false-continue residual (arithmetic from measured rates, not a measured
multi-turn property); the VERDICT-line-stripping detail and the
judgment-exchange-discard pin (implementation cleanliness choices the
spike fixed by pre-registration, not rate-driven); the
one-re-validation-covers-both-seats reading when judgment and action seats
share a profile (synthesis from the FC compositions; the practitioner can
split the seats, which would make re-validation per-seat); the designation
of the false-stop share as the extend-on-evidence trigger for digest
enrichment (the digest-expressiveness ceiling itself is the practitioner's
gate pre-mortem; routing its watch-signal through the
termination-observability FC is drafting-time synthesis following the
cycle's instrumentation-first pattern).

# ADR-048: Grounded acceptance — the accept decision as a composed verification gate

**Status:** Accepted (Conditional) — BUILD validation pending (2026-07-02)

> **Amendment (#84, 2026-07-09): the adequacy signal is deterministic.** The
> judge-adequacy measurement (`benchmarks/judge_adequacy`, 3 prompt variants x
> 128 samples on the live seat) found the model judge never false-accepted an
> inadequate suite (FAR 0.0 in all runs) but false-rejected 25-67% of adequate
> ones, near-deterministically per fixture, and prompt iteration moved the
> miscalibration around rather than removing it. Every measured inadequate
> class has a static signature, so the gate's `tests_adequate` input is now
> the deterministic value-bearing-assert analysis (`adequacy_check.py`) —
> both gate inputs are deterministic, and §5's round-multiplier concern about
> judge conservatism is retired. The model seat survives as
> `adequacy-judge.yaml` (the harness's measurement subject); the §3
> independence posture is unchanged (the checker reads only the executor's
> echoed contract). The AND composition stands: the measurement answered §5's
> AND-vs-weighted question in favor of AND with a calibrated deterministic
> input rather than a weighted stochastic one.

> **BUILD validation update (Cycle-8 WP-D8, 2026-07-03).** The owed Grounding Reframe ran
> (thinned-criteria rerun; `essays/research-logs/cycle-8-spike-q2-grounded-acceptance.md`
> §"WP-D8 Grounding Reframe"). Findings: (a) the anti-gaming catch is 100%
> criteria-independent (10/10 across rich/thin), grounding the default-on-for-build-turns
> decision. (b) §2's "coverage catch is criteria-dependent" sub-claim is corrected: the catch
> survives criteria-thinning via world-knowledge / naming / test-shape tells and degrades ONLY
> in the opaque-spec-with-no-tell corner (e2: 0/5), so the degradation trigger is narrower than
> §2 states — the *decision* (criteria-as-contract primary) stands, the supporting rationale is
> corrected. (c) The judge-consistency Conditional-Acceptance target has supporting evidence:
> every (fixture × condition) cell was unanimous across N=5. Remaining Conditional targets
> (independence vs a live builder, the produced-artifact injection channel, sandboxed
> execution, the unstated-input oracle rung) stay pending. Elicitation is reframed as upholding
> a criteria-*present* invariant on the generation side (WP-C8 / shape-catalog), threaded into
> this gate as a criteria-contract input, not built as a gate seat.

## Context

Q2 (§6.2b) is the DECIDE item after Q4 (ADR-047). ADR-046 settled the per-turn handler
(classify → seat → marshal) and dissolved the internal orchestrator; the client owns the
outer loop. That relocates a hazard rather than removing it (PLAY signal #3): a gameable
gate does not vanish under the outer loop, it relocates to the loop's accept/another-round
decision. If that decision rests on a signal the builder controls, the loop launders a bad
output through more iterations. So the load-bearing requirement is that the accept signal be
**independent of the builder**.

A first framing mislocated that independence in an external client (OpenCode running tests).
OpenCode is the interface that calls back into the llm-orc serving endpoint; it is the
caller, not an oracle. Grounding is composed **inside** the ensemble.

**Relationship to ADR-046's seat contract.** ADR-046 §2 named the seat contract (the
`core/validation/` framework as each seat's pass/fail output check) as the base
independent-correctness signal, and framed Q2 as whether *more* is needed on top. This ADR
answers at a different granularity: the seat contract is **per-seat admission** (does a
seat's output meet its own contract?); the accept gate here is the **loop-level
accept/another-round decision** (is the turn's deliverable good enough to ship, or another
round?). The gate's deterministic half performs the same *kind* of check the seat contract does
(execution and behavioral validation) and could reuse the `core/validation/` framework at the
turn level, though the spike's executor was a bespoke script, so whether they share an
implementation is a BUILD wiring choice, not something established here. The isolated judge is
precisely the "more" ADR-046 flagged as possibly needed. The two compose, they do not compete: a seat may clear its per-seat contract and the
turn still fail the loop-level gate.

**The load-bearing variable is contract quality.** The grounding spike
(`essays/research-logs/cycle-8-spike-q2-grounded-acceptance.md`, 2026-07-02) showed the
gate's discriminating power is conditioned on how well the acceptance criteria enumerate the
cases that matter, not on the verifier architecture alone. The composed gate is necessary
infrastructure; the acceptance criteria are what make it bite. This conditions every decision
below, and is why acceptance-criteria-as-contract (§2) is treated as primary.

**What the spike established.** A minimal gate (deterministic executor seat + one isolated
test-adequacy judge seat + guard), built only from shipped primitives, **discriminated**
correct / trivially-tested / wrong-code / coverage-gap build outputs on the 32GB rig with a
`think:false` 8B judge. No falsifier fired (F1 no-new-primitive, F2 judge-discrimination, F3
executor-wiring). The spike tested **discrimination on static fixtures**; it did not exercise
**independence against a live builder** (see §3).

**Standing lenses.** Determinism over carve-outs: the accept/another-round call is
termination control, which wants a deterministic, independent signal, not an LLM judging its
own work. Ensemble-over-frontier (the standing principle): where no executable oracle exists,
ask what bounded-role orchestration plus deterministic verification can establish before
reaching for a frontier judge.

## Decision

### 1. The accept signal is a composed verification gate, not a single check.

Grounded acceptance = **deterministic execution AND an isolated contract-confidence judge**,
combined by a deterministic gate seat (`accept = tests_pass AND tests_adequate`). The spike's
load-bearing result is that these catch **orthogonal** failures: the executor catches wrong
code that real tests exercise (fixture c); the judge catches trivially-tested or
under-covering outputs the executor passes (fixture b). Neither alone suffices; executor-only
ships trivial-test gaming, judge-only ships wrong code. The conjunction is the gate.

The further result, that the 8B judge also caught a coverage gap on buggy code whose tests
missed a requirement-stated case (fixture d), is a **single exploratory probe on a second
task**, not one of the pre-registered F1–F3 falsification results, and rests on one stochastic
sample. It is promising, not established (see §2 and the Conditional Acceptance below).

### 2. Acceptance criteria are the gate's contract, and the gate's power scales with them. (primary)

The judge's discriminating power is mediated by how specifically the requirement / acceptance
criteria enumerate the cases that matter. In the spike, the judge caught the coverage gap only
because the requirement stated the century rule it missed (one fixture, one fully-explicit
requirement, no comparison case with a vaguer requirement and the same bug, so this is a
single-sample result, not a measured scaling relationship). So user- or spec-provided
acceptance criteria are not a secondary force multiplier; they are the primary lever, threaded
to the verifier seats as their **acceptance-criteria contract** (the natural-language spec the
judge checks against, distinct from ADR-046's deterministic seat contract). This operationalizes
§6.2b's "user acceptance criteria" grounding source.

**Under weak or underspecified criteria (the ordinary case):** the judge-half degrades toward
triviality-detection only (it can still reject `assert True`, but it cannot demand coverage of
cases the contract never names), and the deterministic executor remains the anchor. So
criteria-**elicitation** is itself a design surface: enriching the contract before the gate
(deriving candidate acceptance criteria from the requirement, or requiring them as an input)
raises the gate's ceiling more than judge-side improvements do. Where the contract is thin,
the gate's guarantee weakens to "runs and is non-trivially tested," and that limit is honest,
not hidden.

### 3. Independence comes from architectural isolation plus determinism, composed inside the ensemble. (design intent; validation-pending)

Verifier seats receive only `{requirement / acceptance criteria, produced artifact, execution
result}`, never builder context or reasoning. The intended source of independence is **seat
isolation** (a fresh context treats prior output as external input, the
self-correction-blind-spot principle RDD's own audits exploit), not an external process. The
gate is ensemble-internal; the client does not participate in it.

This is a **design commitment, not a spike-proven property.** The spike tested discrimination
on static, benign fixtures with no live builder in the loop. Independence against a live,
non-cooperative builder is validation-pending at BUILD/PLAY, with two named targets: (a) the
**produced-artifact channel**, the one input the builder fully controls, which could carry
judge-directed content (comments, docstrings, output framing designed to influence the
adequacy verdict); and (b) **builder/judge model-weight correlation** if the same base model
fills both roles, which would produce correlated rather than independent errors. Seat isolation
closes the builder-reasoning channel; it does not by itself close the artifact channel.

### 4. Deterministic anchors are primary; judges and trajectory signals are anchored candidates; nothing builder-internal stands alone.

Signals, ordered by ground-truth strength:

1. **Deterministic execution / static analysis** — builder-independent by construction. The floor.
2. **Isolated judges** — for contract-confidence questions execution cannot answer (test
   adequacy, requirement coverage). Independent via fresh context; correlated-error-prone, so
   anchored by (1) and never standalone.
3. **ML classifiers** (NLI / consistency / embedding) — supplementary anchored signals.
4. **Trajectory confidence** (ADR-014's AUQ/HTC: verbalized confidence plus
   generation-trajectory features, token-entropy patterns, attention-weight distributions over
   tool-call sequences, decision-confidence trajectories, with an entropy-collapse Abstain
   criterion) — an **anchored, non-primary candidate**, parallel to the judge: imperfect
   (builder-internal, weak on confident-wrong), so never the standalone gate, but ADR-046 named
   it as catching contract-conforming-but-anomalous generations a deterministic output check
   cannot see. Whether it earns a place in the gate is a BUILD-phase calibration question, the
   same status the judge's reliability holds. This ADR does not foreclose ADR-046's open
   validation question by asserting the signal out; it carries it forward as a candidate
   pending evidence.

The principle: no builder-internal signal is ever the gate on its own; every non-deterministic
signal is anchored by a deterministic one.

### 5. The gate is minimal-then-ladder.

The MVP is the spike-grounded pair: deterministic executor plus one isolated adequacy/coverage
judge. Ladder rungs are added on demonstrated need, not all at once (holding clear of the
"reduce scope while overcomplicating" anti-pattern): static analysis, a separate code-intent
judge, the anchored trajectory-confidence candidate (§4), and the **held-out / property-based /
golden oracle** for the **unstated-input ceiling** (bugs on inputs the requirement never
states, un-catchable by inspection by construction, and distinct from the coverage-gap case the
judge caught in fixture d). The composition rule for added rungs is itself an open question: a
strict AND across more signals compounds false-rejection risk, so whether rungs join the same
conjunction or a weighted/hierarchical combination is a design decision the two-signal spike did
not settle.

## Deferred — named forward directions, not built here

- **The unstated-input oracle rung.** Property-based / held-out / golden verification for bugs on
  requirement-unstated inputs. Its trigger and mechanism are BUILD work; the ceiling itself is
  un-spikeable by construction.
- **Hierarchical-confidence generalization (assume-guarantee reasoning).** This gate is the
  two-level base case (L0 deterministic execution, L1 contract-confidence judge). The recursive
  form, higher-abstraction seats asserting confidence over the verified contracts of lower ones,
  is the composer-ensemble direction (ADR-047 §Deferred). Two boundaries the spike marks on it: it
  is evidence for the *verification* half, not the *generation* half (verifying is easier than
  generating); and "confidence composes upward" holds only where the contracts threaded down are
  rich (see §2) and deterministic anchors sit at each level.

## Rejected alternatives

- **Builder-authored tests as the gate.** The gameable gate §6.2b names. The spike's trivial-test
  fixture ships under it. Rejected: the adequacy of builder-supplied tests is judged independently.
- **An external client (OpenCode) as the grounding oracle.** Mislocates ground truth. OpenCode
  calls back into llm-orc; it is the interface, not the authority. Grounding is composed internally.
- **Trajectory confidence as a standalone or primary gate.** Rejected: it is builder-internal and
  weak on confident-wrong, so it cannot stand alone. It is retained as an anchored, non-primary
  candidate (§4), not excluded.
- **A single frontier judge over the whole artifact.** Rejected as the default per
  ensemble-over-frontier: the spike showed a bounded 8B judge plus deterministic execution suffices
  for the minimal gate, with no frontier model in the loop. (Two independent axes are in play here,
  model capability and signal-source count; this rejects the frontier-capability default, not the
  idea of a judge. A single non-frontier judge doing both code-correctness and adequacy without a
  separate executor is a different, untested variant, not adopted because the executor's
  deterministic ground truth is the gate's anchor.)
- **Executor-only or judge-only.** Each misses one of the two orthogonal vectors (spike b / c).

## Consequences

**Positive:**
- An accept signal designed to be independent of the builder, grounded on already-shipped
  primitives (no engine work).
- The trivial-test gaming vector is closed by composition, not by a heavier actor.
- Ensemble-over-frontier evidence: an isolated 8B judge discriminated four static fixtures with
  legible reasoning, no frontier model in the loop.
- Acceptance-criteria-as-contract (§2) gives the primary design lever, and names
  criteria-elicitation as where ceiling-raising effort goes.

**Negative / cost:**
- Independence is a design commitment, not yet demonstrated against a live builder (§3); the
  produced-artifact channel is the residual attack surface.
- The gate's guarantee is only as strong as the acceptance criteria; under thin criteria it
  weakens to "runs and is non-trivially tested."
- The unstated-input ceiling stands until the oracle rung is built.
- The 8B judge's reliability and coverage-awareness are single-sample feasibility results;
  consistency across runs and requirement phrasings needs BUILD calibration.
- Sandboxed execution is BUILD work (the spike ran in-process on trusted fixtures).
- Each round adds inference latency (a Q3 / interactive-speed interaction; the judge is a
  `think:false` seat).

**Neutral:**
- Ladder rungs are available but not default; added on demonstrated need. Their composition rule
  is an open design question (§5).

## Conditional Acceptance (ADR-097; institutional pattern, llm-orc ADR-016)

The **architecture** is accepted and spike-grounded for *discrimination*: the composed gate, the
deterministic-primary signal ordering, acceptance-criteria-as-contract, and the
ensemble-over-frontier feasibility of a bounded judge. **Validation pending at BUILD/PLAY**
(transitioning this from Conditional to full Acceptance per ADR-097 path 4, structural
operationalization):

- **Independence against a live builder** (§3), including the produced-artifact influence channel
  and builder/judge model-weight correlation. This is the item most load-bearing on the ADR's title.
- **The isolated judge's reliability** (consistency of the adequacy / coverage verdict across runs
  and requirement phrasings) and the **coverage-awareness result** (a single probe).
- **The anchored trajectory-confidence candidate** (§4): whether AUQ/HTC earns a place in the gate.
- **Sandboxed execution** and the **unstated-input oracle rung's** trigger.

On acceptance the Status becomes "Accepted (Conditional) — BUILD validation pending" per ADR-097
pattern (a).

## Provenance check

- **Driver-derived:** the composed-gate shape and the acceptance-criteria-as-contract primacy
  follow from the Q2 grounding spike and PLAY signal #3. The dissolution and the seat-contract
  granularity are from ADR-046; the deterministic-over-carve-outs and ensemble-over-frontier lenses
  are standing directives.
- **What the spike grounds vs. what is committed:** the spike grounds *discrimination* (the gate
  classifies four static fixtures correctly, F1/F2/F3 false) and ensemble-over-frontier feasibility.
  Independence-via-isolation is a *design commitment*, not a spike result (§3), and is
  validation-pending. The coverage-awareness finding is a single exploratory probe.
- **Empirical-Grounding Filter (ADR-097):** the minimal gate (executor + adequacy judge) is
  spike-grounded for discrimination. The oracle-ladder rungs, the anchored trajectory-confidence
  candidate, and the hierarchical-confidence generalization are named forward directions, deferred
  and un-grounded (the unstated-input ceiling is un-spikeable by construction). Independence and
  judge reliability are Conditional-Acceptance, validation-pending at BUILD. No feature is committed
  on research-surfaced possibility alone.

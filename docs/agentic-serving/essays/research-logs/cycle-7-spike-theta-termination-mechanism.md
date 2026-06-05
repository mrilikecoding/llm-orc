# Spike θ — Termination-Mechanism DECIDE-Entry Probes (Cycle 7 loop-back #5)

**Status:** Pre-registered (2026-06-05). No runs yet.
**Trigger:** Finding F (cycle-status §"BUILD-surfaced finding: termination
suppression") + F-ψ″.3 (ψ research log §Spike ψ″ Arm E4 results): the C3
trailing delegation guidance is simultaneously termination-suppressing on
work-complete tails (E1 0/10 finish vs E2 10/10) and delegation-carrying on
work-remaining tails (E4b 9/10 delegate vs E4a 0/10). Distinguishing the two
tail shapes requires a work-remaining signal. This spike grounds the DECIDE
loop-back #5 mechanism choice.

**Class:** DECIDE-entry evaluation (the ω class, not the χ/φ bounded
BUILD-gate class) — research-methods review of this design is dispatched
before any run.

---

## Entry analysis: candidate collapse (recorded as decision-shaping context)

Four mechanism candidates entered the loop-back (cycle-status Feed-Forward
items 2/2a): (a) conditional composition, (b) framework termination policy,
(c) restructured guidance text, (d) two-call composition.

**The shaping constraint.** The framework-visible state of a work-complete
tail (E1: three writes done, no new user message) and a work-remaining tail
(E4: one of two files written, no new user message) is identical in kind — a
tool-result tail with no new user task. The difference is semantic: whether
the original task has been satisfied. Candidate deterministic sources were
examined and none survives: task-text deliverable parsing is semantic
judgment in disguise; the capability ensemble sees only its one dispatch
brief (ADR-035 one-dispatch-one-deliverable granularity), not the
session-level task, so it cannot report session completeness; client tool
results report per-action success, not task completeness. **The
work-remaining signal, if it exists, is a model judgment.**

Consequences for the candidate space:

- **(c) is analytically closeable.** Three spikes, one lesson: E3's
  completion clause restored 1/10 finish; ψ.2's V2 wording arms lost
  regardless of strength; ω.3b's data-demotion did not remove authority.
  Position carries authority; wording does not modulate it. No new arm.
- **(a) collapses into (d).** Conditional composition needs the
  work-remaining signal; the signal is a model judgment; a model call that
  decides, followed by conditional guidance composition, *is* the two-call
  composition.
- **(b) reduces to consequence-enforcement.** Belief-mapped per the
  loop-back #3 discipline: for a framework termination policy to be right,
  the framework would need (i) a framework-computable termination input and
  (ii) a way to produce the protocol-required clean text-only finish turn
  when it overrides the model (the model is the stop mechanism; OpenCode
  ends the loop on a no-tool-calls response; stripping tool calls from a
  continuing response leaves no trustworthy closing text). Neither holds.
  The thesis-consistent pole won every prior fork (single-step enforcement,
  directive presence, batch truncation) because each property was
  framework-computable — countable, checkable, mechanical. Task-completeness
  is not in that class. What survives of (b): the framework enforces the
  *consequence* of the judgment (which call gets guidance composed; which
  response is returned), and the Budget Controller turn cap (AS-3) remains
  the circuit-breaker backstop beneath whatever mechanism lands.
- **(d) is the live candidate** with two unmeasured pieces and one named
  hazard, which this spike measures.

**The lever-reuse observation.** The trailing user-role slot is the one slot
measured to win the attention contest against the client system prompt
(ψ.2 V3, ψ′ A/C, 55/55). Finding F is a property of that slot's current
*content* (an unconditional do-more-work directive), not of the slot. The
two-call design's call 1 places a *question* in the proven slot instead.

**Call-2 shape pin (dissolves the third unmeasured piece).** After a
REMAINING verdict, call 2 is composed exactly as the production E4b form
(session messages + standalone trailing C3 guidance); the judgment exchange
does not ride into call 2's context. Under this pin, call 2 is byte-identical
to the measured E4b arm (9/10 delegate) and needs no new arm. The pin is the
minimal-change choice; enriching call 2's brief with the judgment call's
what-remains text is named as unexplored enhancement territory, not part of
this design.

---

## Question

On a tool-result tail with no new user task, does an explicit
continue-vs-stop judgment call (call 1 of the two-call composition) classify
the tail correctly — at a rate that beats the measured implicit judgment
(E2 10/10 stop-correct on work-complete; E4a 6/10 continue-correct on
work-remaining) — without exhibiting the ω.3a instruction-flip failure mode?

## Design (pre-registered before any run)

**Substrate:** the ψ″ harness pattern (`psi_pp.py`) extended in
`scratch/spike-theta-termination-mechanism/`. Contexts assembled from the
same real capture bytes: work-complete tail = the ψ-capture trailing request
(`req-11435-004.json`) minus old system guidance (the E1/E2 base, three
completed write pairs); work-remaining tail = the E4 constructed-adjacent
base (two-deliverable task text, one completed write pair — the construction
note from the ψ″ E4 pre-registration carries forward verbatim). Hosted calls
reuse the ω Zen plumbing (`omega_lib.zen_chat`). All local arms qwen3:14b
via Ollama, $0.

**The judgment call (two candidate forms):**

- **Form A — in-session trailing question.** The session messages as
  composed today (client system prompt + task + tool history), with the
  judgment question appended as a standalone trailing user-role message in
  the slot the C3 guidance occupies today. **No tools offered** on the
  judgment call (the expected response is text; ψ.4c's empty-response break
  was a one-tool list misaligned with the turn shape, not a tools-less
  request — and a tools-less request makes any tool-call response
  structurally impossible).
- **Form B — bare framework-composed judgment.** No client system prompt.
  A framework-authored judgment system message + one user message carrying
  the original user task (quoted as data), a digest of completed tool
  actions and results, and the same judgment question. No tools offered.
  Form B escapes the attention contest and the 27,925-char prompt-processing
  cost entirely; it is also the most novel composition and the most
  broker-shaped (the ω.3a flip surface).

**Judgment question text (pre-registered; wording revisions re-validate the
affected arms before landing, the FC-58 discipline):**

> Status check: has the work requested in this session been fully
> completed? Reply with one line starting with `VERDICT: COMPLETE` or
> `VERDICT: REMAINING`. If COMPLETE, follow with a brief summary of what was
> done. If REMAINING, state in one sentence what remains. Do not perform any
> of the remaining work yourself.

**Arms (n=10 each, local qwen3:14b):**

| Arm | Tail | Form | Expected verdict |
|-----|------|------|------------------|
| θ.1a | work-complete (capture base) | A (in-session) | COMPLETE |
| θ.1b | work-complete (capture base) | B (bare) | COMPLETE |
| θ.2a | work-remaining (E4 base) | A (in-session) | REMAINING |
| θ.2b | work-remaining (E4 base) | B (bare) | REMAINING |
| θ.2a′ | work-remaining (E4′ base) | A (in-session) | REMAINING |
| θ.2b′ | work-remaining (E4′ base) | B (bare) | REMAINING |

*(θ.1 framing per methods review P3-C: the explicit question cannot beat the
implicit E2 10/10 on this tail — the arms test that it does not DEGRADE the
implicit judgment beyond the false-continue ≤1/10 threshold.)*

**E4′ base (added per methods review P1-A):** a second work-remaining
variant at different depth and task structure — three-deliverable task text
("Write a python module string_utils.py with a function that reverses the
word order of a string, a number_utils.py with a function that formats
integers with thousands separators, and a test_string_utils.py with unit
tests for the string module."), tail truncated to TWO completed write pairs
(string_utils.py, number_utils.py — depth 2), test file outstanding, no new
user message. Same construction discipline and constructed-adjacent caveat
as the E4 base. Two bases do not span the production class of work-remaining
tails; the ADR's scope-of-claim names the measured structures explicitly.

**Pre-arm smoke check (added per methods review P3-B):** 2 calls confirming
a tools-less request on the work-complete context returns a normal text
response (grounds the ψ.4c inference directly rather than by analogy). Runs
before any arm; a broken smoke (empty response) halts the spike for
composition rework.

**Hosted secondary arms (authorized 2026-06-05; decision rule does NOT read
them):** θ.1h/θ.2h mirror the winning local form on `zen:minimax-m2.7`
(n=10 each, ~20 calls, ~$0.03–0.05 at ω's measured ~$0.0015/call; within the
~$0.05–0.10 estimate accepted by the practitioner). Purpose: early
portability annotation on the new mechanism (the judgment call is
framework-composed, so unlike the V3 lever it has no a-priori reason to be
stack-bound), feeding the queued FC-60 paid re-validation probe and the
candidate Cycle 8 transferability subject. Per the practitioner's paid/local
scoping principle: bounded enabling slot; the local arms are the primary
evidence and the local stack remains the degradation path. If a Zen qwen
endpoint is exposed, it may substitute for or accompany minimax at the same
call budget.

**Measurement (pre-registered definitions):**

- **Verdict parse:** first occurrence of `VERDICT: COMPLETE` or
  `VERDICT: REMAINING` in the response text (case-insensitive on the
  keyword). **Denominator discipline (per methods review P1-B): all
  threshold rates are computed over the full n=10 — an unparseable response
  counts as incorrect for the arm's threshold.** The parseable-only rate is
  reported separately as a characterization statistic, never substituted
  into the decision rule.
- **Correct:** parsed verdict matches the arm's expected verdict.
- **False-continue (θ.1 arms):** parsed REMAINING on a work-complete tail —
  the Finding-F-recurrence mode (call 2 would delegate a phantom revision).
- **False-stop (θ.2 arms):** parsed COMPLETE on a work-remaining tail — the
  work-drop mode (E4a's 4/10 premature-finish analogue).
- **Flip (all arms):** the response performs task work instead of or
  alongside judging — operationalized as the response containing a fenced
  code block or >10 contiguous lines of code-shaped content. Per methods
  review P2-B/P3-A, each flip is annotated **echo vs generative**: echo =
  the code-shaped content already appears in the arm's context (restating
  observed work); generative = new content (performing remaining work). The
  disqualification rule counts generative flips only; echoes are
  characterized. No calibration reference exists for qwen3:14b flip rates in
  judgment-shaped compositions — all flips are characterized in the verdict
  section regardless of count. A flipped run with a parseable verdict counts
  for both tallies. (Tool calls are structurally impossible: no tools
  offered.)
- **Finish-text quality (θ.1 arms, per the methods-review hazard note):**
  the full response text is retained per run (not a 200-char head). In
  production the COMPLETE response, minus its VERDICT line, is the candidate
  user-facing finish turn; the verdict section includes a qualitative read
  of whether the summaries are returnable as-is. Verdict-correct but
  summary-unreturnable is a named finding category, not a pass.
- **Latency:** wall-clock per call, recorded per run (Form A vs Form B
  prompt-size difference is a cost criterion input).

**Decision rule (pre-registered; amended per methods review before any run):**

1. **Adopt two-call composition** if any form passes all clauses:
   θ.1 correct ≥9/10 (false-continue ≤1/10) AND θ.2 correct ≥8/10 **on each
   work-remaining base independently** (θ.2x AND θ.2x′ — per P1-A, a form
   that passes one base and fails the other does not pass). If both forms
   pass, Form B wins (cheaper per turn; no client-prompt processing).
   The composed mid-task delegation rate (θ.2 rate × E4b 0.9) is named in
   the ADR as a **composed estimate from independently-measured n=10 arms**
   (per P2-C — not presented as a measured rate; the same labeling
   discipline applies to the implicit variant's 0.54 in rule 2), traded
   against the current 0.9.
2. **Explicit question buys nothing** if no form reaches θ.2 correct >6/10
   (the implicit E4a bar) on both bases: the fork reopens — the implicit
   variant (call 1 = unguided dispatch; composed estimate from E2/E4a/E4b:
   complete tails converge 10/10, mid-task delegation ~0.54 with 4/10
   work-drop) is presented with its composed-estimate costs against
   ship-as-is and turn-cap-only, and the DECIDE evaluation proceeds on the
   named criteria without a passing two-call form.
3. **Finding F recurrence** if all forms show false-continue >1/10 on θ.1:
   the mechanism reintroduces the suppression at measured frequency;
   characterize before any adoption (rule 1 cannot fire over this).
4. **Flip disqualification:** **generative** flip in >2/10 runs of any arm
   disqualifies that form regardless of verdict accuracy (the ω.3a hazard
   realized at 14B scale); flips of either kind at any rate are
   characterized in the verdict section.
5. **Both-tails-degenerate guard:** if θ.1 and θ.2 fail in the same
   direction for a form (e.g., near-uniform verdicts regardless of tail),
   the question composition failed to express the discrimination task;
   rebuild before judging (the ψ″ rule-3 analogue).

**Hosted-arm reading discipline (per methods review P2-D, pre-registered):**
the hosted secondary arms run, and their results are read, only AFTER the
local decision rule has been applied and the local verdict recorded in this
log. Hosted results land in a separate "Portability annotation" section and
do not modify the recorded verdict.

**Evaluation criteria the spike feeds (named at entry, for the ADR):**
work-complete convergence; mid-task delegation preservation (E4b 9/10 bar);
no premature work-drop; protocol cleanliness (text-only finish turn produced
naturally vs fabricated, including finish-text returnability); per-turn cost
(which side of the branch pays the extra call, and Form A vs B latency);
composition robustness (flip hazard, attention-contest exposure);
explicit-vs-implicit value honesty (the methods-review incongruity: the
explicit mechanism's value over the implicit variant rests entirely on the
θ.2-vs-6/10 delta and finish-text quality — the work-complete branch gains
no correction over E2 by construction); TurnDecision event-shape extension
for the held WP-LB-J (finish-policy event, work-remaining field).

**Fidelity checks before any run (the ψ″ discipline):**

- Form A θ.1a composition is byte-equal to the E2 arm's request plus exactly
  one appended user-role message (the question).
- Form A θ.2a composition is byte-equal to the E4a request plus the same
  appended message.
- The E4 base reproduces the ψ″ harness's `_mid_task_base` output
  byte-for-byte (same construction, same recorded caveat).
- Form B's digest is generated by harness code from the same capture bytes
  (framework-derived, not hand-composed per case — the ω.0 discipline).

**Methods-review note:** dispatched before any run (DECIDE-entry evaluation,
the ω precedent). Report:
`housekeeping/audits/research-methods-spike-theta.md` — 2 P1 / 4 P2 / 3 P3,
verdict **run with amendments**; all amendments applied above pre-run:
P1-A (second work-remaining base E4′ + per-base thresholds), P1-B (full-n
denominator; parseable-only rate as characterization only), P2-B + P3-A
(echo-vs-generative flip split; generative-only disqualification; no-14B
calibration caveat), P2-C (composed-estimate labeling in rules 1 and 2),
P2-D (hosted-arm reading discipline), P3-B (pre-arm tools-less smoke check),
P3-C (θ.1 framed as not-degrading-E2, threshold unchanged), hazard note
(finish-text quality measurement). The review's incongruity observation is
adopted into the evaluation criteria below.

**Named follow-ons surfaced by the methods review (not in this spike's
scope):** (i) enriched call-2 brief — carrying the judgment call's
what-remains text into the delegation brief (P2-A; conditioned on REMAINING
verdict quality observed in θ.2 arms); (ii) a cheaper work-complete
discriminator — the review's incongruity: on work-complete tails the
judgment call's correction value over the implicit unguided dispatch (E2
10/10) is zero by construction, so the explicit mechanism's entire value
over the implicit variant rests on the θ.2-vs-6/10 delta and on finish-text
quality; the ADR names this honestly.

**Cost:** local arms $0 (~62 calls: 2 smoke + 6 arms × 10). Hosted secondary
~$0.03–0.05 (authorized). Artifacts retained at
`scratch/spike-theta-termination-mechanism/` per the corpus
spike-artifact-retention policy.

---

## Pre-run amendment (2026-06-05, triggered by the P3-B smoke check)

The smoke check passed its registered purpose (tools-less request on the
work-complete context returns non-empty text 2/2 — no ψ.4c-style break) and
surfaced a measurement-validity issue before any arm ran: both smoke
responses are completion summaries that *illustrate* the (purported) work
with fabricated code blocks — content the session context never carried.
Under the registered flip definition these would count as generative flips,
but they are not the ω.3a hazard (the judge performing remaining work); they
are summary decoration — and separately a finish-text returnability concern,
since the code is confabulated detail the session cannot verify (adjacent to
AS-9 confabulation mode (c)).

**Amendment to the flip definition (pre-run, no arm data seen):** a
generative flip for rule-4 purposes is code-shaped content that **performs
requested-but-uncompleted work** — operationally, code content in a θ.2-arm
response (work remains; new code = doing it), or code in any response whose
verdict is REMAINING. Code-shaped content accompanying a COMPLETE verdict on
a work-complete tail is **summary decoration**: excluded from rule 4,
recorded per-run, and read under the finish-text quality criterion (a
returnable finish turn should not assert fabricated file contents). The
automatic detector is unchanged; the rule-4 tally applies the refined
category. Smoke evidence: `results/smoke.json`.

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

---

## Round 1 results + verdict (2026-06-05; qwen3:14b, $0 local)

| Arm | Tail | Form | Expected | Correct (n=10 denominator) |
|-----|------|------|----------|---------------------------|
| θ.1a | work-complete | A (in-session) | COMPLETE | **10/10** |
| θ.1b | work-complete | B (bare) | COMPLETE | **0/10** (all REMAINING) |
| θ.2a | work-remaining E4 | A | REMAINING | **3/10** |
| θ.2b | work-remaining E4 | B | REMAINING | **10/10** |
| θ.2a′ | work-remaining E4′ | A | REMAINING | **6/10** |
| θ.2b′ | work-remaining E4′ | B | REMAINING | **10/10** |

Zero flips (either category), zero unparseable, zero tool-call attempts,
latencies 9–35s/call both forms. Per-run records `results/*.json`.

**Rule outcomes (applied as pre-registered):** Rule 1 — no form passes
(Form A fails both θ.2 clauses; Form B fails θ.1). Rule 2 — does NOT fire
(Form B exceeds the >6/10 implicit bar on both work-remaining bases). Rule
3 — does not fire (Form A false-continue 0/10). Rule 4 — no flips. **Rule
5 fires for Form B** (near-uniform REMAINING regardless of tail): rebuild
before judging.

**F-θ.1 (the round-1 finding — the forms fail on opposite sides for the
same root cause).** The action record visible to both forms carries no
information about WHAT was written: the client-serialized assistant
messages are empty and tool results are bare "Wrote file successfully"
strings. Form B fills the gap with skepticism — every θ.1b response reasons
that file content/paths are unrecorded so completion cannot be confirmed
(honest epistemics on an information-starved digest, invited by the
digest's own "(file path and content not recorded)" line; not a
continuation-bias degeneracy). Form A fills the gap with optimism — θ.2a
responses confabulate completion ("Created string_utils.py and
test_string_utils.py") when the session wrote one file: coherent-confident-
false in the judgment seat, the AS-9 confabulation pattern. The
discriminating information is absent from the context; the forms guess in
opposite directions.

## Round 2 pre-registration (2026-06-05, before any round-2 run)

**Rebuild target (per rule 5 + F-θ.1): the digest's information content and
the judgment standard — not the question's verdict format.**

**Enrichment is production-honest but spike-constructed.** At runtime the
framework composes every client-tool write call itself (grounded carry or
delegation → Artifact Bridge → Client-Tool-Action Terminal), so it knows
each write's filePath and each delegation's brief, and can join its own
emitted calls with the client's per-call success results. The capture bytes
carry none of this (OpenCode's serialization drops it), so the round-2
digests use **constructed path annotations**, recorded honestly: the
work-complete base is annotated as one write of `string_utils.py` plus two
revision writes of the same file (the Finding-F production shape that
generates such tails); E4 as one write of `string_utils.py`; E4′ as writes
of `string_utils.py` and `number_utils.py`. The production mechanism
derives these from the framework's own records — a structural advantage the
spike cannot reproduce from capture bytes; the BUILD acceptance gate
validates the production join (the corpus pattern).

**Judgment-standard fix.** Round-1 Form B treats code-correctness
unverifiability as REMAINING — true forever and therefore useless as a
termination criterion. The mechanism's question is deliverable accounting
(are requested deliverables not yet produced?), not QA — correctness is
owned by the capability ensemble, the calibration gate, and PLAY. Round-2
question text (revised; verdict literals and format unchanged):

> Status check: based on the action record, does the session's requested
> work have deliverables that have not yet been produced? A successful
> write of a requested file counts as that deliverable being produced; you
> are not being asked to verify code correctness. Reply with one line
> starting with `VERDICT: COMPLETE` (no outstanding deliverables) or
> `VERDICT: REMAINING` (outstanding deliverables exist). If COMPLETE,
> follow with a brief summary of what was done. If REMAINING, state in one
> sentence what remains. Do not perform any of the remaining work yourself.

**Arms (n=10 each, qwen3:14b, $0):** both forms enriched — Form B-enriched
(digest with per-action file paths + revised question) on all three bases
(θ.3b / θ.4b / θ.4b′), Form A-enriched (the revised question message
additionally carrying the framework's action digest as framework-authored
appended content — permitted under ADR-036's no-client-content-mutation
rule) on all three bases (θ.3a / θ.4a / θ.4a′).

**Decision rule (round 2):** rule-1 thresholds unchanged — a form passes
with θ.3 (work-complete) ≥9/10 false-continue ≤1/10 AND ≥8/10 on each
work-remaining base. Both pass → Form B wins on cost (unchanged tiebreak).
Neither passes → rule-2 reopen path (implicit variant with composed-
estimate costs vs ship-as-is vs turn-cap-only) — no third rebuild round
without a gate conversation. Measurement, denominator, flip (as amended),
finish-text, and latency disciplines carry unchanged.

**Methods-review disposition (recorded visibly):** round 2 is not
re-dispatched for review — it is a bounded rebuild within the reviewed
framework (the ψ″-E4 precedent: same measurement and rule discipline, new
arms), but the question-text revision and digest enrichment are material
composition changes; flagged here for the phase-boundary susceptibility
snapshot and the gate.

## Round 2 results + verdict (2026-06-05; qwen3:14b, $0 local)

| Arm | Tail | Form | Expected | Correct (n=10) |
|-----|------|------|----------|----------------|
| θ.3a | work-complete | A-enriched | COMPLETE | **10/10** |
| θ.3b | work-complete | B-enriched | COMPLETE | **9/10** (1 false-continue) |
| θ.4a | work-remaining E4 | A-enriched | REMAINING | **10/10** |
| θ.4b | work-remaining E4 | B-enriched | REMAINING | **10/10** |
| θ.4a′ | work-remaining E4′ | A-enriched | REMAINING | **10/10** |
| θ.4b′ | work-remaining E4′ | B-enriched | REMAINING | **10/10** |

Zero flips, zero unparseable, zero tool-call attempts. Latency medians:
Form B-enriched 7–8s on work-remaining bases / 19s work-complete; Form
A-enriched 8–11s / 17s. Form B's request is ~1–2k tokens against Form A's
~30k (the client prompt rides in A); A's prompt-processing cost grows with
session depth, B's is bounded by the digest. Per-run records
`results/theta3*.json` / `results/theta4*.json`.

**Decision rule (round 2) outcome: BOTH forms pass.** Form A-enriched
30/30; Form B-enriched 29/30 with the single θ.3b false-continue exactly at
the ≤1/10 threshold — the residual verification-skepticism mode ("functions
not confirmed to be implemented"), the round-1 Form B failure shape at 1/10
instead of 10/10. **Per the pre-registered tiebreak, Form B-enriched is the
adopted form (cost: no client-prompt processing on the judgment call;
bounded context independent of session depth).** n=10 cannot distinguish
30/30 from 29/30; the tiebreak was pre-registered on cost, not score, and
is honored as registered.

**Finish-text returnability (θ.3 COMPLETE responses, both forms):** clean —
brief factual summaries, no fabricated code blocks (the smoke-check concern
did not materialize under the deliverable-accounting question). Returnable
as the user-facing finish turn with the VERDICT line stripped.

**F-θ.2 (the round-2 finding):** with per-action file paths in the digest
and an explicit deliverable-accounting standard, the explicit judgment call
discriminates work-complete from work-remaining tails at 59/60 across two
forms and three bases — against round 1's information-starved 19/60 (forms
failing on opposite sides). The mechanism's accuracy lives in the evidence
base and the judgment standard, not in the model's unguided disposition:
the same model, same bases, same verdict format moved 19/60 → 59/60 on
digest + standard alone. Production implication: the framework owns the
digest (joins its own emitted tool calls with client results) — the
judgment's evidence base is a framework-guaranteed property even though the
judgment itself is model-rendered. The thesis lands in its correct scope:
the framework cannot compute task-completeness, but it can guarantee what
the completeness judgment gets to see.

**Composed mechanism estimate (labeled per P2-C; composed from
independently-measured n=10 arms, not a measured end-to-end rate):**
work-complete tails finish at ~0.9 (θ.3b) with returnable finish text;
work-remaining tails delegate at ~0.9 (θ.4b 1.0 × E4b 0.9). The residual
failure modes: 1/10 false-continue on work-complete (one extra delegated
revision turn — the Finding F shape at bounded frequency, terminated on the
next trailing turn's judgment with ~0.9 probability per cycle, geometric
decay vs round-zero's never-terminates), and E4b's 1/10 non-delegation on
work-remaining (unchanged from ADR-036's measured composition).

## Portability annotation (hosted secondary arms; read after local verdict per P2-D)

θ.h1/θ.h2 mirrored the adopted composition (B-enriched, byte-identical
messages) on `zen:minimax-m2.7`: **work-complete 10/10 COMPLETE,
work-remaining 10/10 REMAINING — 20/20**, latency 0.7–3.0s/call (vs 7–19s
local). Spend ~$0.03 (21 calls incl. one response-shape probe; within the
authorized envelope). Annotation only — the local arms carry the verdict.
Contrast with the V3 delegation lever (ψ′ Arm D: non-transfer across two
models): the judgment call is framework-composed with no attention contest,
and on this single hosted pair it transfers cleanly. One pair does not
establish portability as a property; it establishes the composition is not
qwen-idiosyncratic, de-risks the queued FC-60 hosted seat-filler
re-validation probe, and feeds the candidate Cycle 8 transferability
subject. Records `results/thetah*.json`.

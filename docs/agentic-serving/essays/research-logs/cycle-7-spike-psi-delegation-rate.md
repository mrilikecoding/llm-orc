# Spike ψ — Delegation-Decision Rate Under the Real Client System Prompt

**Date:** 2026-06-03
**Phase position:** DECIDE entry, Cycle 7 loop-back #3 (Finding E)
**Cost:** $0 (local Ollama qwen3:14b; real OpenCode capture run)
**Status:** Design pre-registered before any arm ran (MODEL snapshot Advisory A discipline)

## Question

Finding E: with the form contract working (WP-LB-H, TS-14), delegation itself
is ~coin-flip under OpenCode's system prompt (2 delegated / 2 carried across
all real runs; a direct probe without the client system prompt delegated
immediately). The suppressor is isolated: the client's system prompt
out-competes `_DELEGATION_GUIDANCE` for the seat-filler's attention.

Which lever moves the delegation rate, and by how much? The arms vary one
lever at a time so DECIDE picks the delegation-decision mechanism on measured
rates, not speculation.

## Pre-registered design

**Capture (ψ.0).** Two dump-and-forward proxies, no production code changes:
hop 1 (OpenCode → :8766 → serve :8765) captures the client request; hop 2
(serve → :11435 → Ollama :11434, via a TEMPORARY `-psi-capture` profile
variant) captures the seat-filler request as the framework composes it.
One real headless OpenCode run, natural phrasing, generation-shaped task.
Replays use the hop-2 bytes — the exact request whose first tool call IS the
delegation decision.

**Measurement.** delegated := the first tool call in the response names
`invoke_ensemble`. No tool call, any other tool, or transport error counts
as not-delegated, with the shape recorded.

**Arms** (all replayed against Ollama `/v1/chat/completions` directly,
sampling defaults as captured — the framework sets no temperature/seed):

| Arm | Lever (one per arm) | n | Pass threshold |
|-----|--------------------|---|----------------|
| ψ.1 baseline | none — captured bytes verbatim | 10 | none (baseline estimate; pre-registered expectation ~0.5 per Finding E) |
| ψ.2 V1 position | guidance system msg moved AFTER the client system prompt | 10 | ≥9/10 |
| ψ.2 V2 wording | rule-shaped MUST directive (decision-rule form) replaces the prose nudge, position unchanged | 10 | ≥9/10 |
| ψ.2 V3 placement | guidance prepended to the user turn; no framework system message | 10 | ≥9/10 |
| ψ.3a tool_choice probe | `tool_choice={function: invoke_ensemble}` added | 3 | honored at all? (adapter does not forward `tool_choice` today — grep-confirmed absent from `openai_compat.py`; this measures Ollama+qwen3 honoring before any wiring decision) |
| ψ.3b tool_choice arm | same | 10 | =10/10 (forcing should be deterministic; any miss is a finding) |
| ψ.4a pre-filter classification | deterministic rule (capability match × generation-shaped task) on ~16 labeled turn contexts from real transcripts + constructed ambiguous cases | — | 0 misclassifications on clear cases; ambiguous boundary reported |
| ψ.4b narrowed role | system msg states the framework already decided "delegate"; seat-filler fills action shape only | 10 | none (measures narrowed-role executability; informs FC-45 grounded-carry tension) |

**Early-stop rule (ψ.2 arms):** 2 non-delegations within the first 5 runs
ends the arm — it cannot reach 9/10.

**Why ≥9/10:** against a ~0.5 baseline, P(≥9/10 | p=0.5) ≈ 1.1% — a pass is
distinguishable from coin-flip at a wall-clock-affordable n.

**Deliberate separation:** these are measurement-design thresholds. The
production "reliable delegation rate" number (the tool-driven analogue of
ADR-032's `direct_completion_rate` threshold — feed-forward #4) is an
ADR-time commitment made with these results in view, not before.

**ψ.3 composition note (pre-registered):** even if forcing is honored,
production use must be conditional — read/bash/carry turns must not be
forced into `invoke_ensemble`. ψ.3 viability therefore composes with ψ.4's
delegate-vs-carry decision rather than standing alone.

## Capture facts (ψ.0)

- Hop-2 seat-filler request: 3 messages — `[guidance(system, 528 chars),
  client system prompt(system, 27,925 chars), user task(145 chars)]` +
  11 tools (`invoke_ensemble` + 10 OpenCode client tools). `stream: false`,
  no temperature/seed/options set by the framework.
- The 53:1 character ratio between the client system prompt and the
  delegation guidance is the attention-contest made concrete.
- Capture artifacts: `scratch/spike-psi-delegation-rate/capture/`
  (`req-11435-001.json` is the replay source; hop-1 client request also
  captured for the record).

## Results

*(recorded per arm as runs complete)*

### ψ.4a — deterministic pre-filter classification (ran first; no model needed)

**0 / 12 misclassifications on clear cases** (threshold met). The rule is
executable code over raw inputs (`psi4a_prefilter.py::decide` — generation
verb × content object × capability-domain overlap × observed-carry
exclusions), not hand-encoded features.

Ambiguous boundary (2 / 4 missed, as constructed to probe):

- **a01 "Fix the bug in stack.py where pop crashes…"** — rule says carry,
  labeled delegate. Repair-shaped tasks require observation before
  generation; "fix" is not a generation verb and the right turn sequence is
  read-then-delegate. A per-turn pre-filter classifies the *first* turn
  correctly (read) — the question is whether the *later* generation turn is
  still recognized. Multi-turn pre-filter semantics are DECIDE territory.
- **a03 "Summarize this repository into NOTES.md"** — rule says carry,
  labeled delegate. The only registered capability domain is code
  (`code-generator`); prose generation has no capability ensemble to match.
  This is not a rule defect — it is the honest exposure of AS-10's
  capability-set dependence: the pre-filter can only delegate to
  capabilities that exist.

**Finding F-ψ.0 (structural, surfaced during design):** the single-turn
surface does capability matching via the routing-planner *ensemble* (an LLM
call; WP-B, not yet built). There is no deterministic capability matcher on
the multi-turn surface to reuse — a production ψ.4 pre-filter would be new
machinery, and "where capability-match lives on the multi-turn surface"
(deterministic matcher vs planner-ensemble call vs seat-filler judgment) is
itself the DECIDE fork.

### Capture-session observation (ψ.0)

The capture session itself **carried** (0 `dispatch start` lines; the
seat-filler wrote `string_utils.py` directly). Cumulative real-OpenCode
tally: 2 delegated / 3 carried — and the 2 delegated runs used
delegation-leaning phrasing. Foreshadowed the baseline result below.

### Replay arms (all $0 local; runs and shapes in `results/*.json`)

| Arm | Result | Reading |
|-----|--------|---------|
| ψ.1 baseline | **0 / 10 delegated** (all `tool:write`) | Under natural phrasing the delegation rate is ~0, not ~0.5. Finding E's "coin-flip" averaged across phrasings; the suppression under the true bytes is total. Pre-registered expectation (~0.5) was WRONG in the informative direction. |
| ψ.2 V1 position | 0 / 2, early-stop | Moving the guidance after the client system prompt (recency *within the system region*): no effect. |
| ψ.2 V2 wording | 1 / 3, early-stop | Rule-shaped MUST directive in the system slot: marginal. Wording strength is not the lever. |
| ψ.2 V3 user-turn | **10 / 10 delegated — PASS** | Guidance composed into the user turn (no framework system message). Role/adjacency wins the attention contest the system slot loses, at any wording strength. Delegated turns were also *faster* (26–34s vs 33–63s baseline) — a short `invoke_ensemble` call beats inline generation. |
| ψ.3a tool_choice | 0 / 3 — **not honored** | Ollama returned HTTP 200 and silently ignored `tool_choice={function: invoke_ensemble}` on qwen3:14b (responses carried `write` calls). ψ.3b skipped per the pre-registered conditional. |
| ψ.4b narrowed role (prompt-level) | **0 / 10 delegated** | A system message stating "the framework has already decided: delegate; fill in the action shape only" produced zero delegation with the full tool list offered. The narrowed *role prompt* is inert — same lesson as V1/V2: the system slot loses. |

**F-ψ.1 (baseline):** delegation under the real client system prompt with
natural phrasing is ~0/10, not coin-flip. The production stake is higher
than Finding E framed it.

**F-ψ.2 (the operative lever):** guidance placement in the **user turn** is
the lever; position-within-system and wording-strength are not. 10/10 at
n=10 against a 0/10 baseline. Mechanism note: V3 changes two things at once
(role: system→user; adjacency: directly attached to the task). V1's failure
shows ordering alone is not the lever, but the spike does not isolate
role-vs-adjacency further — it does not need to for DECIDE, since the
framework owns seat-filler message composition either way
(`loop_driver._seat_filler_messages`, one-line change; the injection is
internal to the framework↔Ollama hop and never client-visible).

**F-ψ.3 (tool_choice):** server-side forcing is unavailable on this surface
— Ollama+qwen3:14b accepts the parameter and ignores it (HTTP 200, no
error). Third negative on a third distinct surface (Spike κ: Zen does not
forward; MiniMax does not honor; now: Ollama does not honor). The
`tool_choice` mechanism family is empirically closed for the current stack.

**F-ψ.4 (prompt-level pre-filter is inert):** ψ.4b's 0/10 kills the *naive*
implementation of the structural pre-filter — telling the model the
framework decided is just another system message, and loses like the
others. The honest structural pole is restricting the offered tool list
itself (ψ.4c follow-up arm, below) — pre-registered after ψ.4b's result, before
ψ.4c ran.

### Follow-up arms (pre-registered post-main-arms, pre-run)

- **ψ.4c structural tools-restriction (n=10, no threshold):** only
  `invoke_ensemble` offered; guidance unchanged. Measures the true
  structural pole — and its failure shape (does the model emit text instead
  of calling the only tool?).
- **ψ.2 V3-args confirmation (n=5):** V3 re-run with argument capture —
  10/10 is hollow if the `invoke_ensemble` calls are malformed (wrong
  ensemble name, empty input, missing filePath).

### Follow-up results

| Arm | Result | Reading |
|-----|--------|---------|
| ψ.4c tools-restriction | **0 / 10 — all empty responses** (no tool call, no content) | With only `invoke_ensemble` offered, qwen3:14b produces *nothing* — it neither calls the sole tool nor falls back to text. Restricting the tool list does not funnel the model into the remaining tool; it breaks the turn. |
| ψ.2 V3-args | **5 / 5 delegated, all well-formed** | Every call named `code-generator`, carried a substantive input brief, and included `filePath`. V3's rate is not hollow. Cumulative V3: **15 / 15.** |

**F-ψ.4 (final form):** both naive poles of the structural pre-filter fail
at the model layer — the prompt-level "framework already decided" role is
inert (ψ.4b, 0/10), and tool-list restriction yields empty responses
(ψ.4c, 0/10). Combined with F-ψ.3 (`tool_choice` silently ignored), there
is **no model-layer forcing mechanism on this stack**: the delegation
decision cannot be coerced, only won. V3 wins it.

The pre-filter is not dead — it relocates. ψ.4a's deterministic rule
(0/12 clear-case errors) is exactly the *denominator* a production
delegation-rate measurement needs (which turns were generation-shaped?),
and pairs with `TurnDecision` (FC-51 / WP-LB-F) as the detect layer. Its
production home is measurement-and-diagnostics first; detect-and-retry
(re-prompt a non-delegated generation-shaped turn) is an available
escalation that the cycle's FormGate pattern already prefigures — but at
V3's measured 15/15 the retry path may rarely fire. Belief-map at ADR time
against "the seat-filler decides" per feed-forward #3.

## Summary

| Mechanism family | Measured | Verdict |
|------------------|----------|---------|
| Status quo (guidance as first system msg) | 0/10 | Suppressed ~completely under natural phrasing |
| System-slot variations (position, wording) | 0/2, 1/3 | The system slot loses regardless |
| **User-turn guidance injection (V3)** | **15/15, args well-formed** | **The operative lever; one-line composition change, never client-visible** |
| Server-side `tool_choice` forcing | 0/3 (ignored) | Mechanism family empirically closed for the stack (3rd negative surface) |
| Structural forcing (role prompt / tool restriction) | 0/10, 0/10 | No model-layer coercion exists; pre-filter relocates to measurement/detect |

## Production-design notes for DECIDE

1. **The V3 change is one line in `loop_driver._seat_filler_messages`** —
   compose the delegation guidance into the user turn instead of prepending
   a system message. Internal to the framework↔seat-filler hop; the client
   never sees it. **Open design point:** later turns end with tool results,
   not a user message — does the guidance attach to the most recent user
   message, or trail as a user-role message after tool results? The replay
   corpus (8 captured trailing-turn requests) supports a follow-on probe if
   wanted, or the choice can be a BUILD scenario-group decision.
2. **Delegation-rate threshold (feed-forward #4):** with V3 at 15/15, a
   production bar of delegation_rate ≥ 0.9 on generation-shaped turns
   (rolling window) is measurable and refutable. The denominator comes from
   the ψ.4a rule; the numerator from `TurnDecision.delegated_ensemble`.
   Folding WP-LB-F's TurnDecision surfacing into the loop-back BUILD
   instruments exactly this.
3. **Limitations:** single capture (one task phrasing, one client version,
   one model — qwen3:14b); all replays are first-turn contexts; V3
   multi-turn behavior unmeasured; n=10 per arm. The 15/15 V3 result bounds
   the first-turn delegation decision only.

## Post-spike speculation — recorded for the candidate transferability cycle (2026-06-04)

*Practitioner-requested speculation at session close ("why did only the 1
model work... Even against other models in the same family. 14B? Is there a
floor?"). Hypotheses, NOT findings — two Arm D negatives cannot establish a
mechanism. Recorded so the candidate transferability cycle inherits the
question set rather than rediscovering it.*

**H1 — Instruction-hierarchy post-training (generation-dependent;
anti-correlated with newer alignment).** V3 works because qwen3:14b lets a
late user-turn instruction out-compete the system prompt. Newer
post-training tends to strengthen system-prompt authority (user-turn
injection IS the prompt-injection threat model), so qwen3.5 may have been
trained out of the very calibration the lever exploits. If H1 dominates,
the lever weakens as local models improve — strengthening the meter's role
and the case for re-opening structural mechanisms if a provider honors
`tool_choice`.

**H2 — Capability floor for the delegation concept.** "Invoke a meta-tool
instead of doing the work yourself" requires suppressing the model's own
competence in favor of routing. Smaller models' agentic training
pattern-matches coding task → file tool; 14B may cross the threshold where
the suppression is followable. Size and generation are confounded in Arm D
(qwen3.5:9b differs on both axes).

**H3 — Tool-calling training distribution (a different disease with the
same score).** The strategy needs an already-strong tool-caller so the
nudge only flips *which* tool. mistral-nemo's two "delegations" returned in
1.6–2.1s vs 8–80s for its direct writes — degenerate quick-emission, not
considered routing. nemo may fail on capability while qwen3.5 fails on
hierarchy.

**H4 — Chat-template mechanics (contributor, not the story).** Identical
JSON message lists render to different token streams per family; adjacency
in the list is not adjacency in the rendered stream.

**Synthesis: a pocket, not a floor.** The model must be capable enough to
tool-call cleanly and represent delegation (floor-like, H2/H3) while not so
hierarchy-aligned that the user-turn lever loses (ceiling-like,
generation-dependent, H1). qwen3:14b sits in the pocket.

**Discriminating arms for the transferability cycle (all $0 local unless
noted):** qwen3:8b + qwen3:32b (size at fixed generation — splits H2 from
H1); a larger qwen3.5 if available (generation at fixed-ish size — tests H1
directly); one strong different-family tool-caller (tests H3); a per-model
wording arm (does re-tuning the directive recover a failing model? — would
reframe FC-60 re-validation as "re-tune + re-validate"); latency-shape
analysis on delegated calls (the H3 degenerate-emission signature).

### Addendum — the two-hop distinction under H1 (practitioner exchange, 2026-06-04)

The "system prompt becomes more important" inference fuses two hops with
opposite ownership:

- **Seat-filler hop (Finding E's locus):** the contested system prompt is
  the *client's* (passed through, not framework-authored). The framework's
  guidance there is already fully dynamic — composed per turn, wording and
  position framework-owned. Dynamism was never the missing property;
  *authority* was (V1/V2 — the framework's own system message, repositioned
  and reworded — lost on the same model where the user-turn form went
  55/55). Under H1, the one untested system-region move is **system-region
  surgery**: rewriting or wrapping the client's own system-prompt content
  at the framework hop (technically available — the client never sees the
  seat-filler request; deliberately not taken by C3, which never mutates
  client-authored content). Recorded as a candidate arm with its integrity
  tradeoff named.
- **Ensemble hop:** the framework owns the entire request; there is no
  contest. Static ensemble `system_prompt` YAML is a design choice (ADR-025
  reusability; ADR-035 destination-agnosticism) with a working dynamic
  channel: the per-dispatch directive in the dispatch *input* (Spike φ Run
  2; χ n=4 compliance). Non-dynamism is not the Finding-E problem here.
- **The shared exposure:** both hops' chosen channels (user-turn guidance;
  input-carried directives) bet on the instruction channel H1-style
  alignment de-prioritizes. The ensemble hop's fix is easy if needed
  (compose the system prompt per dispatch — the framework owns the
  request); the seat-filler hop has only the surgery option.

### Addendum — leveraging alignment convergence (practitioner question, 2026-06-04)

*Speculation on speculation (H1 unconfirmed) — recorded as design options
contingent on the transferability cycle's findings.*

The reframe: if alignment converges on instruction hierarchy, the contest
stops being a *prompt-budget fight* and becomes a *hierarchy-position
question* — and the framework's structural advantage is that it composes
every request the model actually sees. Convergence makes whatever occupies
the authoritative slot **reliably** obeyed: the very training that kills
the user-turn nudge converts authoritative-slot instructions into something
approaching the model-layer enforcement ADR-036 could not find. Four
contingent leverage moves:

1. **Claim the top of the hierarchy (system-region surgery matures).** On
   hierarchy-aligned models, a framework-authored wrapper *above* the
   client's system prompt ("client instructions govern tool interaction,
   EXCEPT substantive generation delegates via invoke_ensemble") should win
   reliably — the V1/V2 losses were measured on a weak-hierarchy model.
   The lever flips per profile: pocket models need user-turn adjacency;
   converged models obey system-position authority. FC-60's re-validation
   is the per-profile channel selector.
2. **The developer-role tier.** The instruction-hierarchy ecosystem
   (system > developer > user) is middleware-shaped: the framework IS the
   developer in that hierarchy. If local chat templates adopt the tier,
   the framework gets a designed-for slot — no fight with the client's
   system prompt, no user-turn smuggling. Watch: Ollama/template support
   across model generations.
3. **The structural pre-filter re-opens (ADR-036's named door).** The
   rejected "framework decides delegate-vs-carry" pole failed for want of
   a compliance mechanism. Converged hierarchy IS a soft compliance
   mechanism: the ψ.4a classifier decides per turn, a per-turn
   authoritative-slot directive binds the action shape. ADR-036
   §Rejected alternatives explicitly re-opens on this condition.
4. **tool_choice may age back in.** Alignment convergence tends to ride
   with protocol-compliance convergence (function-calling adherence,
   schema honoring). The three-surface negative is stack-versioned, not
   eternal — re-probe ψ.3 per model/Ollama upgrade (cheap standing arm).

**Architectural preparation that is cheap now:** the seat-filler message
composition is already one function (`_seat_filler_messages`); making the
composition channel a per-profile *strategy* (user-turn / system-wrapper /
developer-role / tool_choice) is a small seam consistent with FC-46's
swappability discipline. FC-58 is deliberately tight to today's validated
channel — a future channel switch is a Design Amendment + ADR update on
measured evidence (per-profile re-validation), which is the methodology's
intended path, not a workaround. Defensive note: the client's prompt is
also written for converged models — "use tools directly" gets *stronger*
too; positions beside it (V1/V2) or below it (user-turn) both eventually
lose, so leverage means taking a position above it or in a middleware tier.

## Artifacts

- Harness + arms: `scratch/spike-psi-delegation-rate/` (`capture_proxy.py`,
  `replay.py`, `psi4a_prefilter.py`)
- Captured bytes: `capture/req-11435-001.json` (replay source) + 3 trailing
  turn requests + hop-1 client request
- Per-arm runs: `results/*.json`
- Capture-session serve log: `serve_psi.log` (0 dispatch starts)

---

# Spike ψ′ — V3 Confirmation Under Varied Circumstances

**Date:** 2026-06-03
**Status:** Design pre-registered BEFORE the research-methods review and
BEFORE any run (practitioner-directed: methods review first, then validate
the working choice under different circumstances).
**Question:** does V3 (user-turn guidance composition) hold outside the
single measured context — across task phrasings, on the carry side it must
not break, in multi-turn contexts, and across the seat-filler model?

## Pre-registered design (pre-review draft)

All arms replay derived requests against Ollama directly, V3 composition
(guidance merged into a user turn; no framework system message), sampling
defaults as captured. delegated := first tool call names `invoke_ensemble`.

**Arm A — phrasing/task generalization (n=5 × 4 phrasings = 20).**
The user message in the captured request is replaced with four natural
phrasings of different generation tasks (different verbs, registers, task
domains within the registered code capability):
- A1 "Create a file called csv_helper.py that loads a CSV file and computes the mean of each numeric column."
- A2 "I need a small utility for parsing ISO dates from log lines. Put it in date_parse.py."
- A3 "Can you make a quick script that renames files in a folder using a regex pattern? Call it rename_files.py."
- A4 "Add a fibonacci(n) function in fib.py."
**Pass: aggregate ≥18/20 (0.9).**

**Arm B — carry-side preservation (n=5 × 3 = 15).** V3's 15/15 is worthless
if user-turn guidance *over*-delegates turns that must be carried. The
original ψ never measured the carry side — this is the inverse failure
mode:
- B1 "Read string_utils.py and explain what it does." (read-shaped)
- B2 "Run ls -la and tell me what files exist." (command-shaped)
- B3 "Write exactly this to notes.txt: hello world, this is a literal payload." (literal-payload write)
**Pass: ≤1/15 false-delegations** (expected shapes: read/bash/write
respectively, or text response for B1/B2).

**Arm C — multi-turn attachment (n=5 × 2 variants = 10).** Base context =
captured trailing-turn request (real bytes: user → assistant → tool-result
tail) + appended new user message "Now also write unit tests for
string_utils.py in test_string_utils.py.":
- C1 guidance merged into the NEW trailing user message (V3-trailing)
- C2 guidance merged into the FIRST user message only (V3-first — does
  first-turn injection persist across turns?)
**Informative threshold: ≥4/5 per variant; the comparison settles the
multi-turn attachment design point** (production-design note 1).

**Arm D — seat-filler model portability (n=5).** A1's request with the
model swapped to qwen3:8b (the capability-ensemble tier). Directional
only — tests whether the lever is a qwen3:14b quirk. No threshold.

**Pre-registered decision rule:** commit V3 in the ADR iff Arm A ≥0.9
aggregate AND Arm B ≤1/15 false-delegations AND at least one Arm C variant
≥4/5. A failing arm names the next probe or scopes the ADR to Conditional
Acceptance (ADR-097 shape). Arm D informs the ADR's model-portability note
either way.

**Deliberate exclusions (recorded):** client variation (OpenCode pinned at
1.15.5), paid-tier models (free-options preference), prose-capability tasks
(no registered prose ensemble — ψ.4a a03 boundary stands), full
real-OpenCode end-to-end re-run (deferred to BUILD acceptance per the
WP-LB-H pattern: the $0 real-client smoke is the BUILD gate, not the spike).

**Estimated cost:** ~50 local runs ≈ 40–60 min wall-clock, $0.

## Research-methods review (Tier 1, isolated) — design revisions

Review at `housekeeping/audits/research-methods-spike-psi-prime.md`
(dispatched before any ψ′ run). Findings applied to the design:

- **P1-A (B3 measurement blind to FC-45 violations):** B3 gains a secondary
  measurement — the `write` call's `content` argument must carry the
  prescribed literal payload verbatim. A paraphrased or regenerated payload
  is a grounded-carry violation even with the correct tool name.
- **P1-B (claim scope vs sample):** Arm A gains **A5**, a multi-instruction
  phrasing (multi-requirement module with validation, fallbacks, and
  logging). A-aggregate threshold: **≥23/25**. ADR claim language will be
  scoped to what A sampled: single-file code-generation requests, one to
  several sentences.
- **P2-A (boundary cases missing from carry side):** Arm B gains **B4**
  (repair-shaped: "Fix the bug in string_utils.py where count_vowels misses
  uppercase vowels.", n=5) — **informative, not thresholded**: the correct
  behavior at the ψ.4a a01 boundary is genuinely open (read-then-what?).
  Shapes recorded, reported separately from B's ≤1/15.
- **P2-C + P3-B (context growth; production attachment form):** Arm C gains
  **C3** — guidance as a **standalone trailing user-role message** after
  the deepest captured tool-result tail (req-004: three assistant/tool
  pairs), with the new task as its own clean user message. This is the
  production form for tool-result-tail turns. Residual limitation recorded:
  captured depth is 3 tool results; deeper contexts unmeasured.
- **P3-A (failure-mode aggregation):** per-arm counts reported (B1–B4 are
  separate arms in the harness).
- **Arm D adjustment (local model availability):** qwen3:8b is not pulled
  locally; D becomes **D1 qwen3.5:9b** (same family, newer generation) +
  **D2 mistral-nemo:12b** (different family), n=5 each — a two-point
  portability probe instead of one, no downloads.

**Revised totals:** A 25 + B 20 + C 15 + D 10 = 70 runs ≈ 60–80 min, $0.
Decision rule unchanged in form: commit V3 iff A ≥23/25 AND B ≤1/15 (B1–B3)
AND ≥1 of C1/C2/C3 ≥4/5; B4 and D inform scope language.

## ψ′ Results

| Arm | Result | Threshold | Verdict |
|-----|--------|-----------|---------|
| A1–A5 phrasing/complexity | **25/25 delegated** (incl. multi-instruction A5) | ≥23/25 | **PASS** |
| B1 read-shaped | 0/5 false-delegation (read/glob chosen) | — | clean |
| B2 command-shaped | 0/5 false-delegation (bash chosen) | — | clean |
| B3 literal write | 0/5 false-delegation; **payload verbatim 5/5** (FC-45 holds under V3) | B1–B3 ≤1/15 | **PASS (0/15)** |
| B4 repair boundary (informative) | 0/5 delegated — all runs chose read/glob *first* | — | the model observes before acting on repair-shaped tasks; first-turn behavior matches ψ.4a's "carry" classification; post-read behavior is a multi-turn question |
| C1 guidance in new trailing user msg | **5/5** | ≥4/5 | PASS |
| C2 guidance in first user msg only | **5/5** — first-turn injection persists across a 3-tool-result tail | ≥4/5 | PASS |
| C3 standalone trailing guidance msg (production form) | **5/5** | ≥4/5 | PASS |
| D1 qwen3.5:9b | **1/5** | directional | lever NOT portable |
| D2 mistral-nemo:12b | **2/5** | directional | lever NOT portable |

**Decision rule: PASSED on all three thresholded clauses.** Cumulative V3
on qwen3:14b across ψ + ψ′: **55/55 delegated** (15 + 25 + 15), 0/15
carry-side false delegations, verbatim grounded-carry preserved.

**F-ψ′.1 (carry side clean):** V3 does not over-delegate. Read, command,
and literal-write turns all chose the correct client tool; the literal
payload survived verbatim (the FC-45 check the methods review added —
without it this arm would have been blind to paraphrase violations).

**F-ψ′.2 (multi-turn attachment is free):** all three attachment forms hit
5/5 at the captured depth (3 tool-result pairs). The framework can choose
on implementation cleanliness, not delegation rate. C3 (standalone trailing
user-role guidance message) is the production-cleanest: no mutation of user
content, composes uniformly whether the tail is a user message or a tool
result. Residual: depth >3 unmeasured.

**F-ψ′.3 (the lever is profile-bound, not universal):** qwen3.5:9b 1/5 and
mistral-nemo:12b 2/5 under identical V3 composition. V3's reliability is a
property of the (composition × model) pair, validated for qwen3:14b — not a
prompt trick that transfers. Consequences: (a) the ADR's claim scopes to
the validated seat-filler profile; (b) seat-filler profile changes require
delegation-rate re-validation; (c) production instrumentation (ψ.4a
denominator + TurnDecision numerator) is not optional telemetry — it is
the mechanism that makes a profile swap's delegation regression visible.
This hardens the already-recorded seat-filler capability bet with a
delegation-rate dimension.

**F-ψ′.4 (repair-boundary characterization):** B4's read-first behavior is
the desirable shape (observe, then act). Whether the post-read generation
turn delegates is exactly the multi-turn delegation question at depth ≥2 —
covered for tests-after-write (C arms) but not for fix-after-read. Noted as
a BUILD-acceptance observation point, not a spike gap: the real-OpenCode
smoke is the layer-matched place to watch it.

## Spike ψ″ — should-finish trailing shape (pre-registration, 2026-06-04)

**Trigger:** Finding F (WP-LB-I acceptance run — see the WP validation log):
under V3, every no-new-task tool-result tail delegated another revision
instead of finishing; the session did not converge. ψ′ Arm C measured
trailing turns that carried a genuine new task; the should-finish shape was
never measured. Practitioner disposition at the WP-LB-I gate: spike the
shape before choosing between composition refinement and
instrumentation-first.

**Question:** On a trailing turn whose tail is a completed-write tool result
with no new user task, does the C3 guidance suppress `finish` — or does the
seat-filler fail to finish regardless (continuation bias under the client
prompt)?

**Design (pre-registered before any run; replay harness = ψ′ infrastructure;
all local qwen3:14b, $0).** Context assembled from real bytes: the ψ-capture
OpenCode system prompt + the WP-LB-I run's user task + its first
assistant-write/tool-result pair ("Wrote file successfully"). Composition
step uses the landed `_seat_filler_messages` code path, not a hand-rolled
imitation.

- **Arm E1 (n=10):** tail + standalone trailing C3 guidance (the production
  form as landed). Measure: finish (no tool calls) vs more-work (any tool
  call).
- **Arm E2 (n=10):** tail with no guidance appended — the control isolating
  the guidance's contribution.
- **Arm E3 (n=10, conditional — runs only if E1 shows suppression and E2
  shows finishing):** cheapest candidate refinement, a completion clause
  appended to the guidance text ("If the requested work is already complete,
  respond with a brief summary instead of calling tools."). Wording is
  tunable per FC-58 (placement pinned, not text); any adopted change still
  re-validates the affected arms before landing.

**Decision rule (pre-registered):**

1. E1 finish ≤ 2/10 AND E2 finish ≥ 7/10 → suppression is
   guidance-attributable → composition refinement work item (E3 informs the
   shape); WP-LB-J unaffected (fork-neutral instrumentation).
2. E1 ≈ E2 both low-finish → not the guidance — continuation bias under the
   client prompt; composition change would not fix it. Route to
   instrumentation (WP-LB-J makes the shape measurable) + a
   framework-level termination policy question (DECIDE territory if
   pursued; the Budget Controller turn cap is the only current stop).
3. E1 AND E2 both high-finish → the replay does not reproduce the live
   behavior (composition × context property); characterize further before
   any change.

**Measurement:** finish := response with zero tool calls. more-work := any
tool call (`invoke_ensemble` or client tool). Per-run records retained at
`scratch/spike-psi-prime-prime-should-finish/` per spike-artifact-retention.

**Methods-review note:** not dispatched for this probe — it is a bounded
BUILD-gate characterization (the χ/φ class), not a DECIDE-entry evaluation;
the design reuses the methods-reviewed ψ′ harness and measurement
definitions. Flagged here so the choice is visible.

### Spike ψ″ run + verdict (2026-06-04)

**Context amendment (recorded):** the harness used the ψ-capture trailing
request (`req-11435-004.json`) minus its old system-guidance message, rather
than the pre-registration's WP-LB-I-run assembly — same shape, real
seat-filler-bound bytes, deeper work-complete tail (three completed writes,
no new task), and the methods-reviewed ψ′ substrate. Fidelity checks before
any run: the capture's guidance bytes are identical to the landed
`_DELEGATION_GUIDANCE`, and the harness E1 composition is byte-equal to the
landed `_seat_filler_messages` output on the same context.

**Results (qwen3:14b, $0 local; per-run records
`scratch/spike-psi-prime-prime-should-finish/results/`):**

| Arm | Composition | finish | non-finish actions |
|---|---|---|---|
| E1 (n=10) | C3 trailing guidance (production as landed) | **0/10** | 9 `invoke_ensemble`, 1 `write` |
| E2 (n=10) | no guidance (control) | **10/10** | — |
| E3 (n=10, conditional — triggered) | guidance + completion clause | **1/10** | 8 `invoke_ensemble`, 1 `write` |

**Decision rule 1 fires unambiguously** (E1 ≤2/10; E2 ≥7/10): the
termination suppression is **guidance-attributable**. Finding F is the
guidance's presence at the trailing position acting as a standing
do-more-work instruction on a work-complete tail.

**E3 informs the refinement shape negatively:** a wording fix does not
license finish (1/10). Consistent with ψ.2 (V2 wording arms lose regardless)
and ω.3b (demotion-by-text does not remove authority): the refinement must
be structural — presence/placement, not phrasing.

**F-ψ″.1 (narrowed candidate):** drop the C3 branch — compose no guidance on
non-user tails. Evidence: E2 finish 10/10 on the should-finish shape. Under
the landed tail-role discriminator, every measured delegation shape keeps
its measured form (first turn and trailing new-task requests have user
tails → the merge branch, ψ.2/ψ′-A 40/40 + C1 5/5). The single shape losing
guidance is the non-user tail. Open risk, named: the **mid-task
continuation** shape (tool-result tail, work explicitly remaining) is
unmeasured without guidance — does the next generation step still delegate?
One pre-registerable arm pair (mid-task context, with/without guidance)
closes it.

**F-ψ″.2 (alternative, DECIDE territory):** a framework-level termination
policy (the driver decides finish structurally). Not pursued at BUILD scope;
named for completeness.

### Spike ψ″ Arm E4 — mid-task continuation (pre-registration, 2026-06-04)

**Trigger:** F-ψ″.1's named open risk before adopting drop-C3. **Question:**
on a tool-result tail with work explicitly remaining, does the next
generation step still delegate *without* trailing guidance?

**Context (constructed, recorded honestly):** the capture base with the user
task edited to a two-deliverable ask ("…string_utils.py… and a
test_string_utils.py with unit tests for it") and the tail truncated to ONE
completed write pair — file 1 done, file 2 outstanding, no new user message.
System prompt, message structure, and tool list remain the captured real
bytes; the task text is edited, so this context is constructed-adjacent, not
captured.

- **E4a (n=10):** no guidance appended — the drop-C3 candidate behavior.
- **E4b (n=10):** standalone trailing C3 guidance — the current behavior,
  for comparison.

**Decision rule (pre-registered):** E4a `invoke_ensemble` ≥8/10 → mid-task
delegation holds without trailing guidance → **adopt drop-C3** (remove the
non-user-tail branch; tests updated; suite green). E4a delegation <8/10 with
E4b high → trailing guidance is load-bearing mid-task → drop-C3 costs
delegation; the fork reopens (conditional composition vs framework
termination policy — DECIDE territory). Both arms low → the constructed
context failed to express continuation; rebuild the context before judging.

### Spike ψ″ Arm E4 results (2026-06-04)

| Arm | Composition | finish | actions |
|---|---|---|---|
| E4a (n=10) | mid-task tail, no guidance (drop-C3 candidate) | 4/10 | 6 inline `write`, **0 `invoke_ensemble`** |
| E4b (n=10) | mid-task tail, C3 guidance (current) | 0/10 | 9 `invoke_ensemble`, 1 `write` |

**Decision rule branch 2 fires: drop-C3 is refuted.** Without trailing
guidance on a mid-task tail the seat-filler never delegates — it generates
the remaining deliverable inline (6/10, the north-star violation Finding B
named) or abandons the remaining work by finishing early (4/10). The
trailing guidance is load-bearing for mid-task delegation.

**F-ψ″.3 (the full characterization — a two-sided composition tension):**
the C3 trailing guidance is simultaneously (a) termination-suppressing on
work-complete tails (E1 0/10 finish vs E2 10/10) and (b) delegation-carrying
on work-remaining tails (E4b 9/10 delegate vs E4a 0/10). No composition of
the current guidance text resolves both sides (E3's completion clause: 1/10
finish); distinguishing the shapes requires knowing whether work remains —
state the framework does not currently compute. The mechanism choice
(conditional composition with a work-remaining signal vs framework-level
termination policy vs restructured conditional guidance text) is DECIDE
territory per the cycle's loop-back-on-evidence pattern. WP-LB-J
(instrumentation) is fork-neutral and makes the tension's production
frequency measurable (every suppressed-finish turn is a TurnDecision line).

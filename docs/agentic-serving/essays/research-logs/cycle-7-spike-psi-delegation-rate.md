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

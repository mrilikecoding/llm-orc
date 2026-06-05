# ADR-036: Delegation-Decision Mechanism — User-Turn Guidance Composition

> **Updated by ADR-037 on 2026-06-05.** Trailing-turn (tool-result tail)
> composition only: the unconditional standalone trailing C3 guidance of
> Decision 1 is replaced by ADR-037's two-call composition (framework-
> composed termination judgment first; the C3-form guidance call fires only
> on a REMAINING verdict — Finding F / F-ψ″.3). Decision 1's first-turn
> merge branch, the FC (directive-in-user-turn presence) as scoped to
> action-generation calls, decisions 2–5, and all other content of this ADR
> remain current. The "no framework system message" neutral consequence is
> narrowed to action-generation calls composed with the client
> conversation (ADR-037's bare judgment call carries a framework judge
> system message outside the client contest).

**Status:** Accepted; Updated by ADR-037 (2026-06-05 — trailing-turn composition) (Cycle 7 loop-back #3 DECIDE, gate closed 2026-06-04; Conditional Acceptance per ADR-097 — see §Empirical grounding for the discharge condition; gating condition met 2026-06-04, WP-LB-I acceptance run)

## Context

Finding E (cycle-status §"BUILD-surfaced finding: delegation reliability,"
2026-06-03): with the deliverable form contract working (ADR-035, TS-14),
whether the seat-filler *chooses* to delegate generation to a capability
ensemble at all was unreliable under the real client. The WP-LB-C
feed-forward had pre-named exactly this contingency as a DECIDE loop-back
trigger "on evidence, not speculation"; the evidence arrived with a clean
isolation — a direct-endpoint probe without OpenCode's system prompt
delegated immediately, identifying the client system prompt as the
suppressor out-competing the framework's delegation guidance.

Spike ψ (pre-registered design; research log
`essays/research-logs/cycle-7-spike-psi-delegation-rate.md`) replayed the
captured seat-filler request bytes — guidance as first system message (528
chars), OpenCode's system prompt second (27,925 chars), user task last —
and measured:

- **Baseline 0/10 delegated.** Under natural phrasing the suppression is
  ~total, worse than Finding E's coin-flip framing (which averaged across
  phrasings; the prior delegating runs used delegation-leaning phrasing).
- **No coercion mechanism exists at the model layer on this stack.**
  `tool_choice` forcing: silently ignored by Ollama+qwen3:14b (HTTP 200,
  ψ.3) — the third negative on a third distinct surface (Spike κ: Zen does
  not forward; MiniMax does not honor). Narrowed-role prompt ("the
  framework has decided; fill in the action shape"): 0/10 (ψ.4b).
  Tool-list restricted to `invoke_ensemble` alone: 0/10, all empty
  responses — the turn breaks rather than complies (ψ.4c).
- **System-slot variation does not move the rate.** Position after the
  client prompt: 0/2 (V1). Rule-shaped MUST wording: 1/3 (V2).
- **Guidance composed into the user turn (V3): 10/10**, confirmed 5/5 with
  argument capture — every call named a valid capability ensemble with a
  substantive input brief and a filePath.

Spike ψ′ (confirmation under varied circumstances; design reviewed by the
research-methods subagent before any run —
`housekeeping/audits/research-methods-spike-psi-prime.md` — which added the
B3 verbatim-payload check, the A5 complexity arm, the B4 boundary arm, and
the C3 production-form arm) passed its pre-registered decision rule:

- **A: 25/25** across five phrasings including a multi-instruction task.
- **B: 0/15 carry-side false delegations**; read, command, and
  literal-write turns chose the correct client tool; the literal payload
  reached the `write` arguments **verbatim 5/5** (the grounded-carry
  contract, ADR-033 FC, holds under the new composition).
- **C: 5/5 on all three multi-turn attachment forms** at the captured
  depth (three tool-result pairs), including guidance as a standalone
  trailing user-role message.
- **D: the lever is not model-portable** — identical composition delegated
  1/5 on qwen3.5:9b and 2/5 on mistral-nemo:12b. V3's reliability is a
  property of the (composition × model) pair, validated for qwen3:14b.

Cumulative V3 on qwen3:14b: **55/55 delegated** — decomposed: 40
first-turn observations across six distinct phrasings (ψ V3 10/10 on the
captured phrasing + ψ V3-args 5/5 argument-capture re-runs of that same
phrasing + ψ′ A 25/25 across five new phrasings) and 15 multi-turn
attachment confirmations (ψ′ C). The components do different epistemic
work: the 40 establish the first-turn delegation decision across phrasing
variation; the 15 establish that the attachment form survives tool-result
tails to depth three. The 27,925-char client prompt against the 528-char
guidance (a 53:1 character ratio — a proxy for the attention contest,
whose role-vs-adjacency mechanism the spike does not isolate) frames why
the system slot loses.

ADR-033 committed that per-turn generation delegates to a single capability
ensemble (callee) but left the mechanism making the seat-filler choose
delegation as BUILD-level implementation; the implementation's own
docstring recorded that whether the nudge held against a client's system
prompt was "not a settled property." This ADR settles the mechanism.

## Decision

**The seat-filler decides delegation; the framework wins that decision by
composing the delegation guidance into the user-turn region of the
seat-filler request, and instruments the delegation rate so regression is
visible. Delegation is won, not coerced — no model-layer coercion exists on
this stack — and the win is a property of the validated stack (composition ×
qwen3:14b × OpenCode 1.15.5), not a universal prompt technique; the
instrumentation in decision 3 is what makes a future loss of the win
visible.**

1. **User-turn guidance composition.** The Loop Driver composes the
   delegation guidance into the user-turn region of the seat-filler
   request instead of emitting it as a system message. On a first turn the
   guidance is attached to the user task; on trailing turns (tool-result
   tails) it is appended as a **standalone trailing user-role message**
   (the C3 form: uniform regardless of tail shape, and it never mutates
   client-authored message content). The composition is internal to the
   framework ↔ seat-filler hop and is never visible to the client. One
   composition point: the Loop Driver's seat-filler message assembly.

2. **Scope of the validated claim.** The measured 55/55 covers: single-file
   code-generation requests of one to several sentences, the qwen3:14b
   seat-filler profile, first-turn contexts and trailing contexts to depth
   three tool-results. Carry-side behavior (read / command / literal-write)
   is clean at 0/15 with verbatim grounded carry. Beyond this scope —
   deeper contexts, fix-after-read second-turn delegation, other client
   versions — the claim is extrapolation backed by instrumentation
   (decision 3), not measurement.

3. **Delegation-rate instrumentation.** The framework computes the
   production delegation rate from events alone: a deterministic
   **generation-shaped turn** classifier provides the denominator (the
   spike-validated rule: generation verb × content object × capability
   domain × observed-carry exclusions; 0/12 clear-case errors), and
   `TurnDecision.delegated_ensemble` provides the numerator. **Refutation
   threshold: sustained `delegation_rate` < 0.9 on generation-shaped turns
   over a 24-hour rolling window is refutation evidence for this
   mechanism** (the tool-driven analogue of ADR-032's deployment-relative
   `direct_completion_rate` threshold — which, like ADR-032's number, is
   provisional: the 0.9 figure is drafting-time synthesis from the measured
   rates, set or revised by the practitioner at the gate and refined by
   PLAY with deployment data). The classifier's known boundary —
   repair-shaped tasks and content domains with no registered capability —
   is recorded; boundary turns are excluded from the denominator rather
   than guessed. Operational signal that the denominator is degrading: a
   growing share of boundary-excluded turns relative to classified turns —
   re-examine the classifier's coverage at PLAY, and whenever new
   capability ensembles are registered (each registration widens the
   capability-domain term of the rule).

4. **Profile-swap re-validation.** A seat-filler Model Profile change
   requires re-validating the delegation rate before the swap is trusted
   (Arm D: the lever does not transfer across models). This *composes
   with* ADR-033's seat-filler-swappability fitness criterion rather than
   amending it: swappability remains a structural property (config-only
   change, no code edits); trust in the swapped configuration is an
   empirical property (a recorded re-validation run, or the production
   meter under decision 3 watched through a soak window — qualifying
   window per decision 3's ≥25 generation-shaped-turn minimum).

5. **Escalation path, held.** Detect-and-retry — re-prompting once when a
   generation-shaped turn does not delegate — is architecturally available
   at the same composition point and follows the cycle's
   detect-and-escalate pattern (ADR-035's FormGate). It is **not built**:
   at 55/55 measured there is no evidence it would fire often enough to
   justify its complexity. PLAY or production-meter evidence re-opens it,
   with the meter reading routing the response: a rate *between ~0.85 and
   0.9* (mechanism mostly working, below threshold) is the retry
   candidate's territory; a rate *below ~0.85 or a degrading trend* points
   at mechanism diagnosis (client-prompt change, model update, composition
   regression) — adding a retry layer to a failing mechanism would mask the
   failure the meter exists to surface.

### Fitness criteria introduced

- **FC (directive-in-user-turn presence):** every seat-filler dispatch on
  the tool-driven surface carries the delegation guidance in the
  **user-turn region** — never as a system message. The structural property
  is user-role placement; the attachment variant is implementation choice
  among the equally-measured forms (attached to the user task, C1/V3;
  first-message-only, C2; standalone trailing user-role message, C3 — the
  preferred form per Decision 1, all 5/5 at measured depth). Refutable: a
  composed seat-filler request with the guidance in a system message, or
  absent, violates this; using C1 or C2 instead of C3 does not.
- **FC (delegation-rate measurability):** the delegation rate on
  generation-shaped turns is computable from emitted events alone
  (generation-shaped classification + `TurnDecision`); no log archaeology
  or replay required. Refutable: a deployment from which the rate cannot
  be computed violates this.
- **FC (profile-swap re-validation):** a seat-filler Model Profile change
  is accompanied by a recorded delegation-rate re-validation (pre-swap
  spike run or post-swap soak-window reading). Refutable: a profile change
  with no recorded re-validation violates this.
- **FC (carry-side preservation):** guidance composition does not alter
  grounded-carry behavior — literal payloads reach client-tool arguments
  verbatim, and read/command turns select client tools, not
  `invoke_ensemble`. Refutable: a carry-shaped turn that delegates, or a
  literal payload that arrives paraphrased, violates this.

## Rejected alternatives

### `tool_choice` server-side forcing

Three distinct surfaces, three negatives: Zen does not forward the
parameter (Spike κ); MiniMax does not honor it (Spike κ); Ollama accepts it
and silently ignores it on qwen3:14b (Spike ψ.3 — HTTP 200, `write` calls
returned under forcing). The mechanism family is empirically closed for the
current stack. Re-opens only if a provider in the stack demonstrably honors
it — and even then it must be conditional (forcing a read turn into
`invoke_ensemble` would be wrong), so it composes with classification
rather than replacing this decision.

### Model-layer structural forcing

Both poles measured dead. The prompt-level pole — telling the model the
framework already decided ("fill in the action shape only") — delegated
0/10; it is just another system message and loses the same attention
contest. The structural pole — restricting the offered tool list to
`invoke_ensemble` alone — produced empty responses 10/10: the model
neither calls the sole tool nor falls back to text; the turn breaks. With
`tool_choice` also closed, there is no mechanism on this stack by which the
framework can *make* the seat-filler delegate.

### Structural pre-filter as enforcement (framework decides delegate-vs-carry)

The thesis-consistent pole — every prior fork this cycle resolved toward
"the framework guarantees the property structurally" (single-step
enforcement, directive presence, batch truncation) — and belief-mapped
deliberately rather than dismissed (WP-LB-H feed-forward #3, recorded
*before* the spikes ran, named the pattern-rhyme risk in both directions —
the belief-map was applied prospectively, not as a finding-driven
retroactive label). What would need to be true for the
pre-filter to be right: a mechanism by which the framework's
delegate-vs-carry decision *binds* the model's action. The spike searched
that space and found none (see both rejections above) — a pre-filter
decision without a compliance mechanism is a system message with extra
steps (ψ.4b measured exactly this: 0/10). The pre-filter's deterministic
classification is not discarded — it relocates to measurement (decision 3),
where it is load-bearing as the delegation-rate denominator. The
enforcement reading re-opens if a compliance mechanism appears.

### System-slot guidance variants

Position after the client system prompt: 0/2. Rule-shaped MUST wording:
1/3. The system slot loses the attention contest against a 53×-larger
client prompt regardless of position or wording strength. The cheap fix
(edit the words) was measured before the structural fix (move the words)
was committed.

### Capable-tier seat-filler to fix delegation

Reviving the frontier-orchestrator shape to buy delegation compliance would
re-open the cost-distribution tension the cycle exists to test (AS-9-shaped
reasoning; the C1 failure mode) — and is unnecessary on the evidence: V3
achieves 55/55 on the cheap tier. The cheap-vs-capable driver bet stays
where ADR-033 put it (long-horizon coherence, BUILD/PLAY), without gaining
a delegation-rate reason to escalate tiers.

## Consequences

**Positive:**
- Delegation on the measured scope goes from ~0/10 to 55/55 with a
  one-function composition change; no new modules, no client-visible
  surface change.
- The carry side is clean: grounded-carry (FC-45) verified verbatim under
  the new composition — the contract that an observed value reaches the
  client unchanged survives.
- Delegated turns are faster than inline generation (short tool call vs
  generating file content in the seat-filler).
- Regression visibility: the rate is computable from events alone, so a
  client-prompt change, profile swap, or model update that re-suppresses
  delegation surfaces in the meter, not in a user complaint.

**Negative:**
- The claim is profile-bound: every seat-filler model change carries a
  re-validation burden (FC). The lever is a (composition × model)
  property, not a universal prompt technique — and the **portability
  failure boundary is uncharacterized**: Arm D's two negatives (qwen3.5:9b
  1/5, mistral-nemo:12b 2/5 — two sizes, two families) establish that the
  lever does not transfer, not *why*. A failing re-validation therefore
  says the swap is untrusted without saying what to change; remediation
  guidance requires a failure model the spikes did not build. The
  practitioner has flagged seat-filler transferability as a candidate
  subject for a future RDD cycle; until then, re-validation is a
  go/no-go gate, not a diagnostic.
- The ψ.4c empty-response finding doubles as a **tool-list design
  constraint** beyond the rejected alternative it closed: qwen3:14b breaks
  the turn (no tool call, no text) when offered a tool list it judges
  incompatible with the task. The framework must never dispatch a
  seat-filler request whose tool list excludes all plausible response
  tools for the turn shape — a registration bug, an uncovered capability
  domain, or a naive detect-and-retry that narrows the tool list would
  each reproduce the break.
- The guidance occupies user-turn tokens on every turn of every tool-driven
  session.
- Residual unmeasured territory: context depth beyond three tool-results,
  delegation on the generation turn *after* a read in repair-shaped flows,
  client versions other than OpenCode 1.15.5. Watched at the BUILD
  acceptance gate and by the production meter, not measured by the spikes.
- The denominator's classifier has a known boundary (repair-shaped tasks,
  uncovered content domains); rate accuracy degrades if boundary-shaped
  traffic dominates.

**Neutral:**
- The framework no longer emits a system message of its own on the
  tool-driven surface; the client's system prompt stands alone in the
  system region.
- The OpenCode prompt-budget contest is sidestepped, not won — a future
  client-prompt change could shift rates in either direction; the
  instrumentation exists for exactly that.
- `_DELEGATION_GUIDANCE` wording is unchanged by this decision (V2 showed
  wording strength is not the lever); wording refinement remains open as
  tuning, gated by the meter.

## Empirical grounding (ADR-097 filter)

**Grounding path: spike validation.** Spike ψ (pre-registered, $0 local,
captured-bytes replay) + Spike ψ′ (methods-reviewed design, pre-registered
decision rule, passed on all three thresholded clauses). **Conditional
Acceptance:** the **gating condition** is the BUILD acceptance run landing
the end-to-end evidence at the real-client layer — a real-OpenCode session
with the V3 composition in which delegation verifiably fires (serve-log
`dispatch start` / `TurnDecision`, per the WP-A scar's lesson that a
passing-looking run can be model-direct). The production meter is the
**trailing confirmation**, not a one-time acceptance gate: a qualifying
first soak window is a provisional minimum of 25 generation-shaped turns
reading ≥0.9 — ψ′ Arm A's n is the starting reference, not a claim that 25
controlled replay turns and 25 live-traffic turns carry equivalent
epistemic weight (live traffic adds unknown phrasing variation,
client-prompt micro-versions, and classifier error; the practitioner
revises the window at the gate for the deployment context); thereafter the
meter provides ongoing regression visibility rather than acceptance
evidence. Revised if that evidence
surfaces suppression the replay layer could not reproduce.

## Provenance check

Driver-derived: Finding E and its isolation (cycle-status, WP-LB-H close);
all rates (Spikes ψ/ψ′ research log); the B3 verbatim check, A5 complexity
arm, B4 boundary arm, and C3 production form (research-methods review
findings P1-A, P1-B, P2-A, P2-C/P3-B); the refutation-threshold pattern
(ADR-032); callee delegation and grounded-carry FC (ADR-033); the
detect-and-escalate pattern (ADR-035 FormGate); the belief-map obligation
on the pre-filter and the threshold-definition requirement (WP-LB-H
feed-forward #3, #4).

Drafting-time synthesis, labeled as such: the specific 0.9 threshold
number (proposed from the measured 55/55 and ADR-032's shape; the
practitioner sets or revises it at the gate, PLAY refines); the ~0.85
retry-vs-diagnose sub-band in decision 5 (synthesized from the threshold
shape, same provisional status as the 0.9 figure); the ≥25-turn soak-window
minimum (Arm A's n as starting reference, practitioner-revisable); the
choice of C3 over the equally-measured C1/C2 as production form
(implementation cleanliness, not rate evidence); the "won, not coerced"
framing of the decision statement; boundary-turn exclusion from the
denominator (a measurement-integrity choice the spikes did not test).

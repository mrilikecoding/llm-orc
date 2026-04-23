---
name: WP-F DECIDE mini-cycle gate reflection
description: Gate reflection note for the WP-F DECIDE mini-cycle boundary (decide → build). Records the question composed for the commitment gate, the user's response, the pedagogical move selected, and the commitment-gating outputs that carry forward into WP-F BUILD.
type: gate-reflection
---

# Gate Reflection: Agentic Serving — WP-F DECIDE mini-cycle (decide → build)

**Date:** 2026-04-22
**Phase boundary:** DECIDE (mini-cycle) → BUILD (WP-F implementation)
**Cycle:** Agentic Serving — scoped RDD cycle at `docs/agentic-serving/`

## Belief-mapping question composed for this gate

Compound gate — Grounding Reframe action plus commitment gating:

1. **Grounding Reframe (from susceptibility snapshot, DECIDE boundary):** "The ground is soft on one thing — the scenario corpus tests the happy path of retry (convention honored) but not the failure path (convention not honored → silent hallucination). WP-F's test suite, as currently specified, would not distinguish a correct implementation from one where the convention is never violated in test fixtures. Concrete grounding actions: (1) add a negative-path scenario describing what happens when a composed ensemble's agent fails to emit `needs_client_tool`; (2) explicitly scope it out — make visible that WP-F acceptance does not require the negative path."

2. **Commitment gating:** "Going into WP-F BUILD, which premises are you building on, and which remain open? Specifically: is the retry pattern a premise you're committing to, or an approach you're willing to revisit if WP-F reveals reliability gaps?"

## User's response

> Yep, negative path scenario seems valuable;. I think the retry pattern should be used if we think it will lead to more capable agentic serving

The user took the grounding action (add the negative-path scenario) and expressed a **conditional commitment** to the retry pattern: committed-to insofar as it delivers capability improvement, open to revisit if the capability benefit is not borne out in practice.

## Pedagogical move selected

Probe + commitment gating. The Grounding Reframe surfaced a specific actionable item from the susceptibility snapshot; the commitment-gating question closed by asking the user to name what is settled and what remains open before advancing. Neither a full challenge (belief-mapping against rejected alternatives) nor a teach (the key trade-offs were already traceable to the retry-pattern analysis earlier in the conversation). The move was calibrated to the mini-cycle's narrow scope — one architectural question was on the table (Client Tool Surface Commitment's sufficiency), and the scenario-writing itself was the grounding.

## Commitment gating outputs

**Settled premises (the user is building on these going into WP-F BUILD):**

- Option C — turn-boundary delegation via `finish_reason: tool_calls` — is the Client Tool Surface Commitment. The scenario gate is resolved.
- Option D (mid-execution callback) is out of scope for this cycle. Reopening would require amending ADR-001 and ADR-002 and adding suspend/resume to the DAG engine's synchronous phase loop.
- Scenarios (a) and (b) are the intended Option C cases — turn-boundary delegation and Session continuity across a client-tool round trip.
- Scenario (c) is carried by *pre-invoke* delegation: the orchestrator reads the needed file at the prior turn boundary and folds content into `invoke_ensemble`'s `input_data`.
- Scenario (d) is carried by the *retry pattern*: the composed ensemble runs to completion, an agent unable to proceed emits a structured `needs_client_tool` signal, Result Summarization preserves the signal, the orchestrator observes it and closes the next turn with a client-tool delegation, then re-invokes the ensemble with the client-tool result folded into `input_data`.
- The negative-path scenario is an acceptance criterion for WP-F: when the retry convention is not honored, the Session's *structural* behavior remains correct (no crash, Budget enforces, no false retry, no extraneous ClientToolCall); the failure is quality-class, not correctness-class.
- The minimum-viable retry-enforcement stack for WP-F is orchestrator system prompt (mechanism i) + composed-ensemble prompt convention (mechanism ii) per roadmap Open Decision Point #8. Both are soft; harder mechanisms (script-agent precondition guards, structural detection in Tool Dispatch) can be introduced as follow-on work without a new ADR.

**Open questions (the user is holding these open going into WP-F BUILD):**

- **Retry pattern's capability value is a conditional premise, not a categorical commitment.** The user's commitment language: "the retry pattern should be used if we think it will lead to more capable agentic serving." Review criterion is capability improvement — the retry pattern stands so long as it makes agentic serving more capable. If WP-F build or post-WP-F observation reveals that retry overhead dominates, silent-hallucination rates are unacceptable, or integration quality with agentic tools does not benefit, the commitment is revisitable.
- **What "more capable agentic serving" means concretely is not yet specified.** Candidate measures: accuracy on mid-execution-need cases, user perception during live sessions, integration smoothness with OpenCode / Roo Code / Cline. Measurement framework is a WP-F build-time or post-WP-F open question.
- **Responsibility Matrix gap** (from argument audit FI-2, susceptibility snapshot feed-forward). The retry pattern implicitly creates a new "emit retry signal" responsibility that has no owning module in the Responsibility Matrix. Deferred — to be resolved before WP-G begins (composition is where enforcement would naturally live).
- **`list_ensembles` output schema** (from argument audit P2-B, susceptibility snapshot feed-forward). Scenario (c) presupposes a schema rich enough for the orchestrator to infer a file-dependency from ensemble metadata. ADR-003 does not specify the schema. A WP-F build-time decision.
- **Summarizer transparency for structured signals** (from argument audit P1-A). Scenario (d) depends on the Summarizer Harness preserving structured JSON signals rather than prose-compressing them. A WP-F build-time configuration constraint, not guaranteed by ADR-004 alone.
- **Calibration Gate coverage for silent quality failures** (from argument audit Round 5c FI-2). The negative-path scenario documents the failure mode but does not itself test Calibration Gate's detection capability — that coverage belongs to WP-H. Whether a companion calibration scenario gets added before WP-F acceptance is a judgment call; the current decision is no (stay scope-narrow).

**Specific commitments carried forward to WP-F BUILD:**

- **Five scenarios** in `scenarios.md` §Client Tool Surface Commitment are WP-F acceptance criteria: (a) turn-boundary delegation, (b) Session continuity, (c) pre-invoke delegation, (d) retry pattern happy path, (negative) silent quality degradation when convention fails.
- **Build-time convention:** composed ensembles emit `{"needs_client_tool": {"tool": "<name>", "args": {...}}}` when an agent lacks required input.
- **Minimum enforcement stack:** orchestrator system prompt + composed-ensemble prompt convention. Both are explicitly overridable without a new ADR.
- **Retry-pattern commitment is conditional.** Named here so future revisit is legitimate rather than a reversal: if capability benefit is not observed, the team may amend to Option D (requiring ADR-001/002 amendment) or to a harder enforcement mechanism.
- **Opportunistic v0.7.3 conformance tidying** (carried over from prior cycle-status FF #92) can be bundled at any natural WP-F commit boundary: framing audit on essay-001, domain-model §Concepts column header, first-person plural in research log, value tensions phrasing in product-discovery.

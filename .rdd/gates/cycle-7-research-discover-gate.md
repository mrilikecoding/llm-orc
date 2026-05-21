# Gate Reflection: Cycle 7 — Cross-Compatibility Routing-Surface Architecture; research → discover

**Date:** 2026-05-21
**Phase boundary:** research → discover
**Cycle:** Cycle 7 — Framework-driven orchestration: routing as code (provisional title) / Cross-Compatibility Routing-Surface Architecture (Essay-Outline 006)

## Belief-mapping question composed for this gate

Composed against F2-1 from the round-1+round-2 framing audit (the structurally largest alternative the Essay-Outline did not name):

> "Spike δ + the orchestrator-LLM-as-recurring-failure-surface pattern across Cycle 6 PLAY + Spike λ-paid is consistent evidence that **removing the orchestrator-LLM from the dispatch path entirely** is a coherent architectural answer. The Essay-Outline's C7 keeps the orchestrator-LLM and adds server-side interception + composition continuation around it.
>
> **What would you need to believe for the F2-1 framing — fully framework-driven dispatch pipeline (ADR-027 candidate), orchestrator-LLM eliminated from the dispatch path — to be the right answer instead of C7's hybrid?**
>
> (Take the question seriously — the cycle's empirical surface pattern arguably supports F2-1 more strongly than C7. The Essay-Outline took the hybrid framing because it preserves more of the current architecture; whether that conservatism is warranted is your judgment.)"

The question was composed using the belief-mapping form (mapping the belief space rather than arguing a position), references specific Essay-Outline content (C7, the empirical pattern across Spike δ + PLAY note 22 + Spike λ-paid F-paid-4), and surfaces the architectural-cost-vs-evidence-direction tension explicitly.

## User's response

> "Hm — well my experience has been that the orchestrator llm has very inconsistently chosen tool invocation when not explicitly told to do so. So I am certainly in favor of at least a harness that forces the issue to a degree. To me llm-orc's stance is that ensembles are used to distribute costs and token to distribute, declarative orchestrations of smaller models. So the orchestrator NEEDs to do that. Otherwise we'd simply use the frontier model. But if we're reaching for llm-orc we are already saying 'we need to optimize for local distribution' or at least 'targeted model' orchestration for tasks. So the hybrid approach is fine if it works, but I favor the stronger stance given the evidence. If the results of this cycle don't show the hybrid approach is effective then I think we need stronger measures."

The response is substantive: (1) records first-person experience with orchestrator-LLM unreliability supporting the harness; (2) articulates the project's value proposition (ensemble-distributed orchestration as cost/capability optimization) as the load-bearing identity claim; (3) explicitly states preference for the stronger stance (ADR-027 framing); (4) names a structural escalation conditional ("if hybrid doesn't work empirically → stronger measures").

## Pedagogical move selected

**Challenge** — belief-mapping question on the F2-1 alternative.

The move surfaced the architectural-cost-vs-evidence-direction tension rather than arguing for or against the cycle's existing recommendation. It mapped the belief space the user would need to inhabit for the alternative framing to be right, leaving the user to apply their own judgment.

## Commitment gating outputs

**Settled premises (the user is building on these going into discover):**

1. NL-routing fraction under tool-rich production clients is empirically approximately zero (Q0 finding from Phase A + Cycle 6 PLAY + Spike λ + Spike λ-paid).
2. The framework implements OpenAI `tool_choice` mechanism (validated under qwen3:14b) but the production model (paid MiniMax M2.5 via OpenCode Zen) does not honor `tool_choice={"name":"invoke_ensemble"}` reliably (C2 model-portability gap is empirically established).
3. The cycle's DECIDE input is a **tiered architecture**: hybrid mechanism (C3 server-side `tool_choice` interception + C4 framework-driven composition continuation + C5 Q2 form-drift enforcement + C6 Q3 fallback design) as starting commitment, ADR-027 framework-driven dispatch pipeline as structurally pre-committed escalation triggered by hybrid-effectiveness measurement.
4. The orchestrator-LLM is the consistent failure surface across three distinct failure modes (composition confabulation per PLAY note 22; positive control per Spike δ; post-dispatch protocol-format failure per Spike λ-paid F-paid-4).
5. Direct-completion fallback is in tension with the project's value proposition; capability-list discovery is a first-order requirement (not documentation); structured advisory for Population B (developer/script clients) names direct invoke as the more aligned access path.

**Open questions (the user is holding these open going into discover):**

1. **Operational criteria for hybrid-effectiveness measurement.** Specific values (routing reliability rate threshold; composition fidelity rate threshold; end-to-end NL routing rate threshold; operator-observable failure rate threshold) are DECIDE-phase work; the cycle commits to criteria existing and to failure triggering ADR-027 escalation rather than incremental hybrid-patching.
2. **C2 diagnosis disambiguation.** Three candidate diagnoses for paid MiniMax M2.5's `tool_choice` non-compliance (Zen proxy stripping; MiniMax model non-conformance; framework tool-list interaction) remain unresolved. If Zen-proxy-specific, server-side interception may be unnecessary for non-Zen deployments.
3. **Model-portability characterization across additional orchestrator profiles** (Groq Llama-3, Cerebras, Anthropic Sonnet, OpenAI gpt-4o-mini, etc.).
4. **Product-discovery validation of the value-misalignment framing at W6.2** — currently grounded in practitioner stance + one PLAY-constructed stakeholder persona; DECIDE-phase product-discovery should confirm or revise before treating capability-list discovery as first-order.
5. **Architectural-cost quantification for the hybrid-first ordering vs. ADR-027-direct.** The Essay-Outline asserts but does not quantify the differential; DECIDE should weigh the comparison explicitly.
6. **Empirical-grounding cluster** (four working-inference evidence nodes: E3.1.1 server-side interception preserves client-facing contract; E4.2.1 production-client filesystem scope; E5.3.3 framework-driven composition continuation eliminates orchestrator narration; E6.2.1 project value-proposition) — each needs DECIDE-phase corroboration or explicit residual-uncertainty acceptance before BUILD proceeds.

**Specific commitments carried forward to discover:**

1. The DISCOVER phase should attend to the value-misalignment framing at W6.2 — independent product-discovery work on the user-population composition (ensemble-orchestration-seeking vs. general-completion-seeking) is the cycle's most consequential next-step grounding work.
2. Stakeholder updates needed: the Skill Orchestration User stakeholder model from Cycle 6 PLAY was the basis for the Cycle 7 RESEARCH framing; DISCOVER should examine whether the stakeholder set extends to a Population A / Population B distinction (caller types) and what their value-proposition expectations are.
3. The Orchestrator LLM (as actor) stakeholder shifts substantially under the tiered architecture: its role in Tier 1 is constrained (composition continuation removed; routing potentially intercepted server-side); in Tier 2 it is removed from the dispatch path entirely. DISCOVER should surface this stakeholder-model change explicitly.
4. Ensemble Author / Operator stakeholder gains new authoring surface: routing-planner ensemble (if Tier 1 (i') option (i)), synthesizer ensemble (if Tier 2 ADR-027), tool-call-as-output-format authoring (if Q2 (b)).
5. The cycle's recommendation is conditional on the four-node empirical-grounding cluster; DISCOVER should not commit product-thinking that depends on these nodes being validated until they are.

## Reading note for the susceptibility-snapshot evaluator

The cycle's research journey navigated TWO significant reframings:

1. **Phase A reframe (agent-driven):** the Q0 empirical finding that NL-routing under production clients is ~0 led the agent to propose collapsing Q1 to "tool_choice contract conformance." This was a sycophancy-adjacent move (the reframe simplified the design dramatically; users tend to like simpler answers). The user authorized validation spikes which empirically refuted the reframe's strongest form.

2. **RESEARCH-gate reframe (user-driven):** the user's belief-mapping response pushed back on C7's hybrid framing in favor of the stronger ADR-027 stance. The cycle responded with structural Essay-Outline revision (tiered architecture). This is not sycophancy — the user's position was substantively argued (project value proposition + first-person orchestrator-LLM experience). The revision was structurally verified across 5 audit rounds.

Pay particular attention to whether the agent's Phase A reframe (movement 1) shaped the Essay-Outline in ways that the gate-driven revision (movement 2) did not fully unwind. If the tiered architecture's "hybrid as starting commitment" still encodes a Phase A residual that the user's stance would have removed, the snapshot should surface it.

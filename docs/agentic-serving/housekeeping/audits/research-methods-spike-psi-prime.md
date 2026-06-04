# Research Design Review — Spike ψ′ (V3 Confirmation Under Varied Circumstances)

**Reviewed question set:** `docs/agentic-serving/essays/research-logs/cycle-7-spike-psi-delegation-rate.md` §"Spike ψ′ — V3 Confirmation Under Varied Circumstances" (pre-registered design, pre-review draft)
**Constraint-removal response included:** n/a (confirmation spike, not a question-set in the ADR-082 sense)
**Date:** 2026-06-03
**Reviewer role:** research-methods (ADR-060 + ADR-082 dimensions 1–4 applied)

---

## Summary

- **Arms reviewed:** 4 (A phrasing generalization / B carry-side preservation / C multi-turn attachment / D model portability)
- **Decision rule reviewed:** 1 pre-registered rule with three conjunctive criteria (A ≥0.9 / B ≤1/15 / C ≥4/5 per variant)
- **Flags raised:** 7 (2 P1, 3 P2, 2 P3)
- **Criteria applied:** 1–4 (ADR-082)

---

## Per-Arm Review

### Arm A — Phrasing/Task Generalization

**Belief-mapping.** The arm assumes that what failed in ψ was purely the guidance position, and that the task phrasing used in the captured request is not itself load-bearing. A different productive question: "Is the 53:1 character-ratio attention contest phrasing-sensitive — do longer or more syntactically complex user messages reproduce the suppression even with V3?" That question is excluded by the design, since all four A phrasings are short (one sentence each) and comparable in length to the captured request's 145-char user turn. The arm also assumes the registered code capability domain is representative; it does not probe whether V3 holds when the task is in a domain closer to the capability envelope boundary (e.g., "Add a fibonacci(n) function" is closer to the carry/delegate ambiguity than "Create a file called csv_helper.py that...").

**Embedded conclusions.** "Four natural phrasings of different generation tasks" presupposes that four phrasings of the same task type (new-file generation) are sufficient to establish generalization across task phrasings. The underlying need is "does V3 hold across the realistic variation a production user introduces?" Four phrasings within one capability domain and one task type (new-file code generation) is a narrow sample of that space. The embedded conclusion is that phrasing surface (verb choice, register) is the primary dimension of variation — not task type, not instruction length, not instruction structure.

**Scope.** Appropriately sized for a confirmation spike at $0 local cost. The register/verb variation within the code-generation domain is the right first probe. The within-domain restriction is not a design flaw at this stage — it is appropriately scoped to the capability the system currently has. The concern (see P2 below) is with what the pass threshold licenses, not with the arm's scope per se.

---

### Arm B — Carry-Side Preservation

**Belief-mapping.** The arm measures whether V3 over-delegates turns that must be carried. The three phrasings (read-shaped B1, command-shaped B2, literal-payload write B3) are designed to be clearly non-generation. A different productive question: "What does V3 do with turns at the delegate/carry boundary — the ψ.4a ambiguous cases (a01 repair-shaped, a03 prose-generation-without-ensemble)?" Those are the turns where V3 getting it wrong has production impact: over-delegation on an ambiguous turn fails visibly (wrong ensemble or malformed invocation); under-delegation is the baseline failure mode Finding E documented. The arm as designed tests V3 against cases where the correct answer is unambiguous — it does not test the boundary.

**Embedded conclusions.** "≤1/15 false-delegations" on three unambiguously non-delegation phrasings presupposes that carry-side failure mode is bounded by clearly read/bash/write requests. This is a presupposition about where the failure would occur, and it may be wrong: the production failure mode for over-delegation is most likely at the boundary (ψ.4a a01/a03 territory), not at the clear cases. Testing clear cases and licensing a "carry-side preserved" conclusion from them embeds the claim that clear cases are representative of the carry-side failure risk.

**Measurement validity issue (P1).** The dependent variable `delegated := first tool call names invoke_ensemble` is correct for Arm A. For Arm B it produces a correct outcome signal for B1 (read) and B2 (bash) — a false-delegation is observable as `invoke_ensemble`. However, B3 ("Write exactly this to notes.txt: hello world, this is a literal payload") is a grounded-carry case: the correct behavior is a `write` call with the literal payload carried verbatim (FC-45 grounded-carry). `delegated := first tool call names invoke_ensemble` captures the false-delegation direction correctly for B3, but the design does not record the argument content of the write call — so if V3 causes the model to *generate* content for the write rather than carry the literal payload, the measurement will report "not delegated (correct)" when the actual behavior is a FC-45 grounded-carry violation. The confirm/refute outcome on B3 is right only if carry-fidelity is also checked, not just the tool name.

**Scope.** Too narrow on the failure cases that matter most (boundary territory), but appropriately sized for an initial confirmation check.

---

### Arm C — Multi-Turn Attachment

**Belief-mapping.** The arm tests two attachment positions (trailing vs. first) using a single appended task ("Now also write unit tests for string_utils.py"). This assumes that a single follow-on task phrasing can settle the multi-turn attachment design point. A more productive framing: the captured trailing-turn corpus contains 8 real trailing-turn requests (per ψ.0 artifacts) — using the actual captured trailing bytes rather than a synthetic append to a single captured context would test the attachment question against the distribution of real multi-turn shapes, not just the one-task-appended form. The arm's C2 variant (guidance in first user message only, persisting across turns) also presupposes the conversation context is preserved faithfully across turns — but `_to_openai_messages` (loop_driver.py line 491–494) carries only `role` and `content`, not `tool_call_id` or the assistant turn's `tool_calls` array, which the design labels as WP-LB-C deferred work. If the trailing-turn replay uses real captured bytes that include those fields, the arm is replaying a fidelity that the production path does not yet have; if it strips them, C2's propagation finding depends on a degraded context that may differ from production.

**Embedded conclusions.** The production-design note 1 question is "does guidance attach to the most recent user message, or trail as a user-role message after tool results?" Arm C answers the *injection-point* question (trailing vs. first-turn) but not the *attachment-form* question (merged-into-user-message vs. trailing-user-role-message-after-tool-results). Those are different design dimensions: C1 tests V3-trailing (guidance in the appended user turn), but a separate production scenario is guidance appended as a standalone user-role message after tool results when the latest turn is a tool result, not a user message. That scenario is absent.

**Scope.** Informative threshold (≥4/5 per variant, no pass/fail gate on the overall decision rule) is appropriate given the complexity. The arm is the right question to ask; the issue is that it answers a subset of the production attachment design space.

---

### Arm D — Seat-Filler Model Portability

**Belief-mapping.** The arm probes qwen3:8b as the single portability check, on A1's request. This assumes that qwen3:8b is the relevant portability question. A different productive framing: the decision rule commits V3 in the ADR, and the ADR will govern a production system that uses whatever seat-filler model profile is configured (swappable per ADR-011/FC-46). The portability question is therefore not "does qwen3:8b also work?" but "what properties of a model are required for V3 to work, and does the current model-profile selection surface expose those?" qwen3:8b is a reasonable first probe because it is the capability-ensemble tier, but testing only one quantization family at a smaller size does not characterize the portability boundary.

**Embedded conclusions.** "Directional only — tests whether the lever is a qwen3:14b quirk." The arm treats "not a qwen3:14b quirk" as equivalent to "portable." Those are not the same. If qwen3:8b passes, the conclusion is warranted for one model at one quantization level; the arm's framing allows Arm D to satisfy the portability note in the ADR without characterizing the failure boundary.

**Scope.** Appropriate for a directional probe at $0 cost. No threshold is the right call. The concern is with how the ADR uses the result (see P2 below).

---

## Decision Rule Review

**Pre-registered rule:** commit V3 in the ADR iff A ≥0.9 aggregate AND B ≤1/15 false-delegations AND at least one C variant ≥4/5.

**Confirmation-shape risk (P1).** The three A phrasings (A1–A4) are all short-form new-file code generation tasks. ψ's 15/15 result used a single captured task in the same register. Arm A tests whether V3 is robust to verb/register variation within one task type; it does not test whether V3 is robust to instruction-length variation, multi-instruction requests, or tasks that include embedded sub-instructions (e.g., "Create a csv_helper.py that loads a CSV file, computes the mean of each numeric column, and outputs results to summary.txt" — a task where the user message itself contains structured content that might compete with the guidance for the model's attention). Given that the original finding was an attention contest between guidance and a large client system prompt, the risk of a similar contest between guidance and a structurally complex user message is non-trivial and is not covered by A1–A4.

The pass threshold A ≥18/20 (0.9) at n=5 per phrasing is statistically defensible against the ψ.1 0/10 baseline (P(≥9/10 | p=0.5) reasoning from ψ applies). However, with four phrasings all from the same task-type bucket, a pass establishes phrasing-surface robustness within new-file generation, not generalization across task types. The ADR's scope of claim must be bounded accordingly or the threshold is insufficient for the conclusion it licenses.

**Carry-side threshold (P1/P2 boundary).** B ≤1/15 false-delegations on three clearly non-delegation phrasings is defensible as a sanity check that V3 does not collapse carry-side entirely. It is not defensible as evidence that the carry-side failure mode is bounded in production, because the three phrasings do not cover the ψ.4a boundary territory where the delegation/carry decision is genuinely ambiguous. A passing B arm licenses "V3 does not over-delegate on clearly non-generation requests" — not "V3 preserves carry-side fidelity in production." If the ADR claims the latter from a passing B, the decision rule produces an over-broad conclusion from an appropriately scoped arm.

**Multi-turn threshold.** C ≥4/5 per variant is reasonable for n=5. The "at least one variant" pass condition is appropriate given the arm is informative rather than gating — it settles the attachment design point, not the ADR commitment. This is the best-designed element of the decision rule.

---

## Question Set Assessment

### Premature narrowing / prior-art treatment

The ψ′ design is a confirmation spike for V3, which ψ established as the working choice. That framing is appropriate: ψ was a one-lever-at-a-time measurement under a specific context, and ψ′ is explicitly checking whether that result holds more broadly. The design acknowledges its own limitations in §"Deliberate exclusions."

However, two exclusions that are listed as deliberate raise questions:

**Conversation length / context growth** is not named as a deliberate exclusion and does not appear in any arm. All arms use first-turn or synthetic single-append trailing contexts. In production, the seat-filler's context grows across turns: tool results accumulate, the client system prompt remains in every request, and the guidance (prepended to the user turn in V3) is present only in the current turn. Whether V3 continues to win the attention contest as the conversation context grows — particularly as tool-result content grows and may itself contain instruction-like text — is untested. This is not an exotic scenario: the grounded-carry path (FC-45) specifically involves the seat-filler reading prior tool results in the conversation history. A tool result that contains file content with instruction-like phrases (e.g., a README.md) could interact with the guidance.

**Tool-result content as instruction injection** is a specific form of the above that the design does not treat as a failure mode. If a tool result (e.g., a `read` of a configuration file or a README) contains text that resembles delegation instructions or counter-instructions, V3's adjacency-to-task advantage may be reduced or reversed. This is not covered by any arm and is not listed as a deliberate exclusion.

### Incongruity surfacing

The spike design and the prior context together surface an incongruity that ψ′ does not examine. ψ's F-ψ.2 mechanism note records: "V3 changes two things at once (role: system→user; adjacency: directly attached to the task)." The V1 failure (guidance moved after the client system prompt within the system region: 0/2, early-stopped) shows that ordering alone within the system region is not the lever. But the spike does not isolate role vs. adjacency — the design explicitly says "it does not need to for DECIDE, since the framework owns seat-filler message composition either way."

The incongruity: the carry-side preservation arms (B1–B3) are also short user messages, just like the A arms, and the guidance is prepended to them as well. If the model delegates on A phrasings and carries on B phrasings, the spike will have demonstrated that the guidance is being read and respected — which is the intended conclusion. But if the model carries on B phrasings because the task content overrides the guidance (the task is unambiguously read/bash/write), rather than because it correctly parsed the guidance as "carry literal values only when you already have them," then the B passing result does not confirm the mechanism — it confirms that unambiguous carry tasks are carried regardless of what the guidance says. The mechanism distinction matters for the boundary cases. ψ′ does not surface this.

### Coverage gaps

1. **Instruction-length and instruction-complexity variation in Arm A.** All four A phrasings are single-sentence, single-file, same-register tasks. Production requests from OpenCode will include multi-file tasks, tasks with embedded constraints ("make sure to use type annotations"), and longer requests. Whether V3 holds for those is unmeasured.

2. **ψ.4a boundary territory in Arm B.** The ambiguous cases (repair-shaped tasks, prose generation without a registered ensemble) are the production failure mode for over-delegation, not the clearly read/bash/write cases B1–B3 cover.

3. **Guidance fidelity as context grows.** No arm tests V3 beyond a single appended follow-on task. A three- or four-turn context with accumulated tool results is absent.

4. **FC-45 grounded-carry argument fidelity for B3.** The DV does not check whether the `write` arguments carry the literal payload verbatim or regenerate it — only whether the tool name is non-`invoke_ensemble`.

5. **Trailing-user-role-message-after-tool-results attachment variant.** Production-design note 1 raises the question of attachment when the latest turn is a tool result, not a user message. Arm C does not cover this variant.

---

## Findings

### P1 — Design flaws that would invalidate specific conclusions

**P1-A: Arm B's measurement does not detect FC-45 grounded-carry violations.**

B3 ("Write exactly this to notes.txt: hello world, this is a literal payload") is the only arm that tests the literal-carry path. The DV (`delegated := first tool call names invoke_ensemble`) detects false-delegation correctly but does not detect the case where V3 causes the model to write a generated or paraphrased version of the literal payload rather than carrying it verbatim. A passing B3 result under the current DV is consistent with a FC-45 violation — the arm would report "not delegated (correct behavior)" whether the write argument is `"hello world, this is a literal payload"` or a model-generated paraphrase.

**Recommended design change:** Add a secondary measurement for B3: extract the `content` argument of the first `write` tool call and check for exact string match against the prescribed payload. Record mismatches as FC-45 failures, distinct from the delegation DV. This requires one additional check per B3 run against the result JSON, adding negligible cost.

**P1-B: The decision rule licenses a scope of claim (phrasing generalization across the code capability) that the arm's sample cannot support.**

All four A phrasings are single-sentence new-file generation tasks in the same register. A pass at A ≥18/20 establishes verb/register robustness within this narrow task type. If the ADR commits V3 with a claim of "phrasing generalization," that claim is not supported. The risk is that the ADR commits V3 with language broader than the evidence warrants, and a subsequent real-OpenCode session with a longer or more complex request returns delegation rates below the production threshold — producing a loop-back that a better-scoped ψ′ would have flagged.

**Recommended design change:** Either (a) add one A phrasing that is multi-instruction or longer (e.g., "Create csv_helper.py that loads a CSV, computes per-column mean, and writes a summary to summary.txt — include type annotations throughout") to test instruction-complexity variation, or (b) constrain the ADR's scope-of-claim language to "single-sentence new-file generation tasks in the registered code domain" and treat broader generalization as a BUILD acceptance criterion. Option (b) is lower cost and may be the right call given this is a $0 spike.

---

### P2 — Weaknesses that bound the claims

**P2-A: Arm B tests the wrong failure cases for carry-side production risk.**

The three B phrasings (read, bash, literal-write) are unambiguous carry cases. The production over-delegation failure mode is at the delegate/carry boundary: repair-shaped tasks (ψ.4a a01), prose tasks without a registered ensemble (ψ.4a a03), and tasks that are generation-shaped but the seat-filler has already produced partial results. A passing B arm establishes "V3 does not catastrophically over-delegate on obvious carry requests" — which is a useful sanity check but a low bar. The ADR should not cite a passing B as evidence that V3 preserves carry-side fidelity in production without noting that the boundary cases were not tested.

**Recommended design change:** Add one B phrasing from the ψ.4a boundary territory, specifically the repair-shaped case (e.g., "Fix the bug in string_utils.py where pop crashes when the list is empty"). This is a case where the ψ.4a rule says "carry" (repair is read-then-carry) but which a generation-guidance model might misread as "generate." If V3 over-delegates on this phrasing, the finding is significant; if it carries correctly, confidence in the carry-side is substantially stronger. Cost: 5 additional runs within Arm B.

**P2-B: Arm D's portability probe does not characterize the failure boundary.**

Testing qwen3:8b on A1 establishes that V3 is not a qwen3:14b-specific artifact at one smaller-model data point. It does not characterize what properties a model needs to exhibit V3's behavior, nor does it test a model from a different family. The ADR's model-portability note, if it cites Arm D, should be scoped to "V3 observed to hold on qwen3:8b (n=5, directional)" — not "V3 is model-portable." The design's own "no threshold, directional only" framing is correct; the risk is in how the ADR prose uses the result.

**Recommended design change:** No change to the arm design. Flag for the ADR author: the portability note should cite qwen3:8b directional only and explicitly leave cross-family portability unmeasured. If a second Ollama-available model (e.g., llama3.2, phi-3.5-mini) can be probed at $0, a five-run directional check would strengthen the note without significant cost.

**P2-C: Context growth and tool-result content as competing instruction are absent from the design.**

V3's mechanism is attention-contest-by-adjacency: guidance prepended to the user turn wins over the 27,925-char client system prompt because it is role-proximate and task-adjacent. As the conversation context grows (tool results accumulate), the ratio of guidance to total context shrinks. At some turn depth, a longer conversation may reproduce the suppression that ψ.1 measured for the baseline. No arm tests this. The deliberate exclusions list does not name context growth.

**Recommended design change:** The existing captured trailing-turn corpus (8 captured trailing-turn request bytes, per ψ.0 artifacts) provides a $0 probe opportunity. Add one arm (C3, informative, no threshold) that replays a captured trailing-turn request with 3+ accumulated tool results and V3 guidance prepended to the trailing user turn. This directly tests whether adjacency holds under realistic context accumulation at no additional model cost beyond ~5 local runs.

---

### P3 — Improvements

**P3-A: The "≤1/15 false-delegations" threshold for Arm B conflates two distinct failure modes.**

False-delegation of B1 (read-shaped) means the model called `invoke_ensemble` instead of a read or text response — a clear error. False-delegation of B3 (literal-write) could mean (a) the model called `invoke_ensemble` when it should have carried verbatim (DV-detectable) or (b) the model called `write` with generated content when it should have carried verbatim (DV-invisible per P1-A above). Aggregating these into a single count obscures which failure mode is occurring.

**Recommended design change:** Report per-phrasing false-delegation counts in addition to the aggregate (B1: N/5, B2: N/5, B3: N/5). This costs nothing and makes the result interpretable at the ADR.

**P3-B: Arm C does not test the production attachment form for tool-result-tail contexts.**

Production-design note 1 raises "does the guidance attach to the most recent user message, or trail as a user-role message after tool results?" Arm C tests trailing-user-message injection (C1) vs. first-user-message injection (C2). A third variant — guidance appended as a standalone user-role message immediately after the last tool-result message, in a context where the trailing turn is a tool result rather than a user message — is the scenario production-design note 1 is actually asking about, and it is absent. The captured trailing-turn corpus contains requests in this shape.

**Recommended design change:** Add C3 (n=5, informative): replay a captured trailing-turn request where the last message is a tool result, with guidance appended as a standalone user-role message after the tool result. Compare delegation rate to C1 (guidance in trailing user message). This directly answers the design note 1 question at $0 cost.

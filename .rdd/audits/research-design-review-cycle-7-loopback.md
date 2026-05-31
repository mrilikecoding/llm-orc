# Research Design Review — Cycle 7 Loop-Back (BUILD → RESEARCH)

**Reviewed question set:** Step 1.1 north-star statement + Step 1.2 constraint-removal response + agent-proposed sharpened question (all provided inline; no file path)
**Constraint-removal response included:** Yes (Step 1.2)
**Date:** 2026-05-24

---

## Summary

- **Questions reviewed:** 3 (north-star statement as Q1; constraint-removal response as CR; agent-sharpened question as Q2)
- **Flags raised:** 4
- **Criteria applied:** 1–4 (ADR-082 full set)

---

## Per-Question Review

### Q1: North-star statement

"My repeated goal here is stated as my 'north star': use llm-orc via agentic serving to run RDD (or similar long-horizon process that may involve skills or skill frameworks) via a tool like OpenCode, trusting that work will be delegated to ensembles. So the spike here needs to validate that not only is the ensemble delegation effective, but can driving ensembles via chat or multi-turn interaction with OpenCode result in what we'd expect from using OpenCode with a 'normal' single model process."

Bar: PARITY (behavioral/experiential; latency explicitly excluded).

**Belief-mapping:** What would the researcher need to believe for a different question to be more productive?

The north-star frames the goal as *parity* — ensemble-delegated OpenCode session behaves like a single-model OpenCode session. A different question becomes more productive if the researcher believes parity is not the right bar. Specifically: parity with which class of single-model behavior? A "normal single model process" covers many behaviors (tool calls, multi-turn context, mid-session corrections, sub-agent spawning). If the ensemble endpoint can reproduce only a subset of those behaviors, the parity bar may produce an all-or-nothing verdict that misses a more useful graded question: "Which behaviors does ensemble delegation preserve, and which does it degrade?" The graded question would let the RESEARCH phase characterize the achievable parity surface rather than test against a uniform bar.

The north-star also combines two separable questions: (a) does ensemble delegation produce correct outputs (effectiveness), and (b) does the overall session feel like a "normal" OpenCode session (experience). These can diverge — a session where ensemble work is correct but requires extra turns from the user is effective but not parity-experiential. The question set does not separate them, and a spike designed around a binary parity verdict may not surface this divergence.

These are refinements, not disqualifying flaws. The north-star is a statement of the research motivation, not a testable question; the sharper spike question (Q2) is where the parity bar will actually be operationalized. The belief-mapping concern is flagged here as a note for Q2 operationalization.

**Embedded conclusions:** None flagged. The north-star is well-formed as a motivation statement. It does not presuppose a mechanism.

**Scope:** Appropriate as an orienting statement. It correctly scopes the spike's success bar to experiential parity rather than latency.

---

### CR: Constraint-removal response

Artifact bracketed: ADR-027's dispatch pipeline.
Prompt: "If ADR-027's dispatch pipeline weren't the committed substrate, how would ensemble-generated work land on the developer's local filesystem?"
Response: "We could assume for now that we are running llm-orc on the same machine as OpenCode. We can strategize later about llm-orc serve running on another machine. Given that, being able to write to the local file system at a specified location should be very possible."

**Response substance:** Partially engaged, but the engagement is thinner than the question requires. The response answers the *delivery* sub-problem (co-located server writes directly to the local filesystem) by shifting the deployment assumption rather than by reasoning about what mechanism would replace the pipeline. This is a valid imagined-without-it framing for the delivery problem, but it does not engage with the *coherence* sub-problem — how OpenCode would know that the file was written, receive a result it can act on, and continue its agentic loop. A constraint-removal response that fully treats the pipeline as prior art would also need to address: if the pipeline's `tool_calls` terminal is removed and replaced by a server-side write, what does the OpenCode session receive as a response? Nothing? A text confirmation? That response shape determines whether OpenCode's loop continues normally or stalls.

The response is not performative — it commits to a real architectural assumption (co-location) and draws a real conclusion from it (direct write is feasible). But the assumption resolves the *simpler* of the two delivery sub-problems while leaving the harder one (OpenCode loop coherence) unaddressed.

**Flag CR-1 — constraint-removal response resolves delivery but not loop coherence.**

The loop-back trigger diagnosis (from the cycle-status BUILD-surfaced finding) correctly split delivery into two concerns: (1) bytes to disk; (2) OpenCode's execution model remaining coherent. The constraint-removal response addresses (1) by shifting to co-location. It does not address (2). This is not a failure of the constraint-removal exercise — it correctly surfaces that delivery is simpler than the pipeline's authors assumed. But it leaves the more consequential question open: under co-location, if the endpoint writes to disk and returns a text response, does OpenCode's agentic loop continue normally? If OpenCode's loop requires a `tool_calls` response to treat the write as a tool execution it tracks, then co-location resolves delivery-geography but not loop-coherence, and the `tool_calls` mechanism is still load-bearing — now justified by OpenCode's execution model rather than by filesystem geography.

The spike design should probe both: whether direct co-located write + text response is enough for OpenCode to continue normally (which would make the `tool_calls` mechanism unnecessary), or whether `tool_calls` is required for loop coherence independent of where the write happens.

**Embedded conclusions in the response:** One embedded conclusion flagged. The response implies that the delivery problem is essentially solved under co-location ("should be very possible"), which forecloses examining whether the delivery mechanism matters to OpenCode's behavior. The conclusion is embedded in the phrase "write to the local file system at a specified location should be very possible" — the response treats write-ability as the criterion, when the real criterion is OpenCode continuing its agentic session coherently after the write.

Suggested reformulation: "Under co-location, server-side write is feasible. The open question is what response the endpoint sends back and whether OpenCode's agentic loop requires a `tool_calls` response to track the write as an executed tool action, or whether it can continue from a text acknowledgment."

---

### Q2: Agent-sharpened question

"Does OpenCode's agentic loop stay coherent — file written with the ensemble's content, tool result fed back, next turn continues — when pointed at agentic-serving, where the generation is delegated to an ensemble and the deliverable returns as a `tool_calls` response OpenCode executes locally?"

**Belief-mapping:** What would the researcher need to believe for a different question to be more productive?

The sharpened question is better than the pipeline-centric framing it replaced, but it still presupposes a specific mechanism. A different question becomes more productive if the researcher believes `tool_calls` is not the only mechanism by which OpenCode's loop can stay coherent after an ensemble-delegated write. Under that belief, the more productive question is: "What response shape from an ensemble-delegated endpoint allows OpenCode's agentic loop to continue normally?" This opens the solution space to at least three mechanisms the sharpened question forecloses:

1. **Direct write + text acknowledgment.** If the endpoint (co-located) writes to disk and returns a text response describing what was written, does OpenCode notice the file change and incorporate it? OpenCode has its own filesystem monitoring or re-read behavior; whether it tracks file changes made outside its own tool executions is an empirical question the spike could answer cheaply.

2. **Direct write + synthesized tool-call shell.** The endpoint writes to disk and returns a `tool_calls` response that *describes* the write as if OpenCode had executed it, without OpenCode actually executing anything. This is a hybrid: delivery happens server-side, but the protocol fiction of a client-side tool execution is maintained. This is mechanically different from "OpenCode executes the write_file call" and may be sufficient for loop coherence without requiring the full round-trip.

3. **Full `tool_calls` round-trip.** The endpoint does NOT write to disk; it returns a `write_file` tool call; OpenCode executes it; OpenCode owns the write. This is what the sharpened question presupposes as the mechanism.

Mechanisms 1 and 2 are made available by the constraint-removal response's co-location assumption, which the sharpened question does not exploit. Mechanism 1 in particular is the simplest possible path — it was the approach the orchestrator-LLM effectively tried (text response describing what the ensemble produced), and the cycle-status finding diagnosed this as insufficient. But the diagnosis was about the orchestrator-LLM's behavior under a complex ReAct loop; co-located direct write + text acknowledgment is structurally different and worth testing explicitly rather than dismissing by analogy.

**Embedded conclusions:** Flagged. The sharpened question presupposes `tool_calls` as the mechanism for loop coherence. The phrase "the deliverable returns as a `tool_calls` response OpenCode executes locally" is the presupposition. This forecloses mechanisms 1 and 2 above without testing them.

This is the embedded conclusion the review prompt identified as the primary concern: the question fuses the mechanism (`tool_calls`) with the need (loop coherence), when the need could be satisfied by other mechanisms. The `tool_calls` mechanism is the most principled answer from OpenCode's documented execution model, and the loop-back finding argues for it well — but the argument is an inference from OpenCode's architecture, not an empirical observation, and the spike is the right place to test it rather than assume it.

**Suggested reformulation:** "What response shape from an ensemble-delegated endpoint allows OpenCode's agentic loop to continue normally after ensemble-generated work is applied to the local filesystem — and is the `tool_calls` round-trip necessary for that coherence, or is a co-located direct write with text acknowledgment sufficient?"

This preserves the core empirical question (does the loop stay coherent) while opening the mechanism question for the spike to answer rather than presupposing it.

**Scope:** Too narrow. The question investigates one mechanism when the constraint-removal response has opened a second mechanism (co-located direct write) that is simpler and testable at the same cost. A well-scoped spike question for this loop-back should bracket both mechanisms and let the spike results determine which is necessary.

---

## Question Set Assessment

### Premature narrowing / prior-art treatment

**Flag N-1 — the question set does not treat the constraint-removal response's co-location path as a mechanism to test.**

The constraint-removal response opened a genuine alternative: co-located direct write. The sharpened question ignores this alternative and goes straight to the `tool_calls` mechanism. This is premature narrowing. The two mechanisms have meaningfully different implementation costs and reliability profiles:

- Co-located direct write + text acknowledgment: simpler to implement (no `tool_calls` emission logic needed), but uncertain whether OpenCode tracks it.
- Full `tool_calls` round-trip: more complex (requires the pipeline to emit a `tool_calls`-shaped response, handle multi-turn loop), but aligns with OpenCode's documented execution model.

The prior-art criterion is partially satisfied: the constraint-removal response treats ADR-027 as prior art by imagining without it. But the response's finding (co-location resolves delivery) was not carried into the spike design. The sharpened question treats the `tool_calls` mechanism as if the constraint-removal exercise had concluded it was necessary, when the exercise actually opened a simpler alternative.

The spike should test both. Testing whether a direct write + text acknowledgment is sufficient for OpenCode loop coherence takes at most one additional probe cell and could resolve the question without requiring the full `tool_calls` implementation — which is Cycle 7 WP-B/WP-C territory. If the direct-write path fails, the spike has evidence that `tool_calls` is load-bearing. If it succeeds, the implementation path simplifies substantially.

**Prior-art treatment verdict:** Satisfied for the pipeline artifact (constraint-removal response treats it as prior art). Not satisfied for the co-location finding — the constraint-removal response's alternative path is not treated as prior art in the spike design.

### Incongruity surfacing

**Flag I-1 — a simple co-located delivery path sits adjacent to the complex `tool_calls` round-trip being designed, and the question set does not examine it.**

The research context (cycle-status BUILD-surfaced finding) diagnosed the gap as: ADR-027 has no `tool_calls` terminal. The natural remedy is to add a `tool_calls` terminal. But the constraint-removal response introduced a simpler delivery shape — co-located write — which the question set did not examine. The incongruity:

- Elsewhere in the system, direct writes are the delivery mechanism for ensemble outputs: ensembles write to `.llm-orc/agentic-sessions/` substrate paths. This is a simple, working delivery path.
- For the client-delivery problem, the question set is designing a complex `tool_calls` round-trip with specific protocol semantics, multi-turn loop handling, and OpenCode execution-model alignment.

The question worth asking: is the complexity of the `tool_calls` round-trip driven by a real requirement of OpenCode's execution model, or by an assumption that the only way to notify OpenCode of a completed write is to have OpenCode perform the write itself? The simplest adjacent pattern — co-located write + OpenCode reads the file on its next turn — has not been tested and may satisfy the loop-coherence requirement without the round-trip complexity. The cycle-status finding dismissed this by analogy to the orchestrator-LLM's text-response failure, but co-located direct write is structurally different from the orchestrator-LLM returning a text description of what it wrote (which it often fabricated). The analogy may not hold, and the spike is the right place to test it.

This incongruity is worth forcing into the spike design as a probe cell rather than leaving it as an untested assumption.

### Coverage gaps

**Parity operationalization.** The north-star sets a parity bar but neither Q1 nor Q2 operationalizes it. What observable behaviors in an OpenCode session count as "normal"? The spike design (as described in the cycle-status path-forward) correctly focuses on: file written, tool result fed back, next turn continues. This is a reasonable first operationalization. But "next turn continues" is ambiguous — does it mean OpenCode sends a follow-up message, or that the overall task completes correctly over multiple turns? For a long-horizon process like RDD, single-turn coherence may not be sufficient evidence of full parity. The spike design should name what multi-turn behaviors it will check beyond the first tool-result-fed-back confirmation.

**Multi-turn loop depth.** The spike question asks whether the loop "continues" after the first tool result, but does not name how many turns constitute a passing test. A session that succeeds for one write/result/continue cycle may fail on the second or third — especially if OpenCode's context management differs between single-model and ensemble-delegated modes. The spike should specify a minimum number of turns it will run before declaring loop coherence.

### Recommendations

**Priority 1 — Reformulate Q2 around the need, not the mechanism.**

Replace the sharpened question with the needs-grounded version: "What response shape from an ensemble-delegated endpoint allows OpenCode's agentic loop to continue normally after ensemble-generated work is applied to the local filesystem — and is the `tool_calls` round-trip necessary for that coherence, or is a co-located direct write with text acknowledgment sufficient?"

This directly addresses Flag Q2-1 (embedded conclusion) and Flag N-1 (premature narrowing) and ensures the spike tests both mechanisms.

**Priority 2 — Add a probe cell for co-located direct write + text acknowledgment.**

The spike design (cycle-status path-forward) describes standing up a minimal endpoint that emits `tool_calls`. Before that, add a cheaper probe: stand up a minimal endpoint that dispatches to an ensemble, writes the deliverable directly to the co-located filesystem, and returns a text response describing the write. Observe whether OpenCode's loop continues. This probe answers the mechanism question at near-zero additional cost and either validates the `tool_calls` necessity or removes the need for the more complex implementation.

**Priority 3 — Operationalize the multi-turn parity bar before the spike runs.**

Name a minimum number of turns and a minimum set of behaviors the spike must observe before declaring loop-coherence validated. Suggested minimum: at least two write/result/continue cycles; at least one case where the second turn's request depends on the first turn's written file (i.e., OpenCode reads back what was written). This prevents the spike from declaring parity on a single successful round-trip.

**Priority 4 — Address CR-1 in the spike design.**

The constraint-removal response's co-location assumption opens a deployment configuration (llm-orc on the same machine as OpenCode) that the spike should test under explicitly, rather than treating co-location as an unexamined convenience. If the spike validates that co-location enables direct-write delivery, that is an architectural finding with implications for the DECIDE/ARCHITECT phases. If it does not (because OpenCode's loop coherence requires `tool_calls` regardless of deployment topology), the spike has established that `tool_calls` is load-bearing independent of delivery geography — which is the stronger and more principled version of the argument the loop-back finding currently makes by inference.

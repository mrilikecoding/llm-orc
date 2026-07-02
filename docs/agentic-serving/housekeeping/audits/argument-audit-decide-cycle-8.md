# Argument Audit Report

**Audited document:** `docs/agentic-serving/decisions/adr-044-declarative-ensemble-native-serving.md`, `docs/agentic-serving/decisions/adr-045-clean-slate-collapse-imperative-serving-layer.md`
**Source material:** `docs/agentic-serving/proposals/play-closeout-2026-06-30.md`, `docs/agentic-serving/proposals/engine-control-flow-state-and-next-steps.md`, `docs/agentic-serving/proposals/ensemble-agent-state-and-next-steps.md`, `docs/agentic-serving/product-discovery.md` (Cycle-8 additions, Tension 23), `docs/agentic-serving/domain-model.md` (§Invariants AS-1..AS-11, Amendment Log #22), and the full ADR-033–ADR-043 loop-driver chain plus ADR-006/015/019/021/025/027 (read for consistency)
**Genre:** ADR
**Date:** 2026-07-01

## Section 1: Argument Audit

### Summary

- **Genre:** ADR
- **Argument chains mapped:** 6 (AS-11 codification; clean-slate collapse decision; the AS-6/AS-11 actor split; the ADR-033–043 supersession chain; the 024–032 non-supersession scope line; the carry-forward requirements table)
- **Issues found:** 6 P1, 6 P2, 2 P3

### P1 — Must Fix

**P1-1. The non-supersession scope line is drawn by ADR-number proximity, not by mechanism audit against AS-11 — and it silently omits two ADRs whose decided mechanisms are the exact shape AS-11 forbids.**

- **Location:** ADR-045 §Rejected alternatives, "Blanket-supersede the pre-033 serving ADRs (024–032) as well. NOT done here"; ADR-044 §Context (names `tier router` and `orchestrator tool dispatch` as modules inside the collapsed `agentic/` layer).
- **Claim:** ADR-045 asserts only ADR-033–043 need supersession; ADR-024–032 "survive or need per-ADR review," deferred whole to a follow-on Cycle-8 target-architecture ADR. ADR-006/015/019/021 aren't mentioned at all.
- **Evidence gap:** ADR-044's own Context names `tier router` and `orchestrator tool dispatch` as part of the ~12.7K-LOC imperative layer being collapsed. Those are not orphaned code — they are the decided mechanisms of **ADR-015** (per-role tier-escalation router: "adds a per-role tier-escalation router between the orchestrator's tool calls and the dispatched ensembles," an L2 interposition explicitly chosen *over* embedding the decision in a composed ensemble) and **ADR-021** (per-capability dispatch: "the orchestrator's ReAct loop receives the request, identifies the capability ensemble... and dispatches," a runtime capability-selection decision made inside `OrchestratorRuntime`). Both are textually the same shape AS-11 names as forbidden ("a parallel orchestration layer (adapter, driver, or harness)"), and neither is superseded by ADR-045, nor named in its "needs per-ADR review" list (which only mentions 024/025/026/027), nor addressed anywhere in ADR-044. A downstream reader or conformance auditor has no way to tell whether ADR-015 and ADR-021 remain in force, are silently obsolete, or require amendment — the corpus is simply silent on two ADRs whose subject matter is squarely inside AS-11's blast radius.
- **Recommendation:** Add ADR-015 and ADR-021 to the "needs per-ADR review" list (or supersede them outright, since their named mechanisms are already inventoried as removed code), and state explicitly why the scope line stops at ADR-024 rather than reaching further back to any ADR whose mechanism matches the forbidden shape.
- **Routing:** N/A (not an Essay-Outline pyramid-boundary finding; ordinary ADR internal-consistency gap).

**P1-2. ADR-027's own decided mechanism — not just its "direction" — is already the shape AS-11 forbids, but ADR-045 treats it as an undecided candidate for later review.**

- **Location:** ADR-045 §Rejected alternatives ("027 framework-driven-dispatch direction" listed among directions that "survive or need per-ADR review").
- **Claim:** ADR-045 characterizes ADR-027 as an "established direction" whose disposition is open, to be resolved case-by-case later.
- **Evidence gap:** ADR-027's decision text states the dispatch stage is executed by "the framework... deterministically... via the existing `OrchestratorToolDispatch` machinery... No LLM is in the dispatch-decision loop at this stage." That is a runtime control-flow decision implemented as bespoke Python sitting beside the DAG — functionally identical to the adapter-mediated dynamic dispatch ADR-044 itself rejects a few paragraphs earlier ("Adapter-mediated dynamic dispatch... re-introduces a control-flow decision in adapter Python, reopening the exact seam AS-11 closes"). ADR-045's framing ("a direction... needs per-ADR review") understates that the mechanism, as currently decided and recorded, already conflicts with AS-11 today, not just prospectively. (The `dispatch_pipeline.py` implementation may already be dead code via ADR-043's collapse, but ADR-027 the decision record has not been amended or flagged to reflect that — so the corpus's authoritative record still holds the imperative dispatch mechanism as ADR-027's decision.)
- **Recommendation:** Either supersede ADR-027's dispatch-stage mechanism explicitly (distinct from its plan/synthesize stages, which are declarative and fine) or add an explicit note to ADR-027 flagging its dispatch stage as AS-11-noncompliant pending the Cycle-8 target-architecture ADR, rather than filing the whole ADR under "surviving direction."
- **Routing:** N/A.

**P1-3. The carry-forward table contains a factual error: Row "039/042" labels the surviving mechanism "content anchor (bounded)," but the bound was reverted.**

- **Location:** ADR-045 §Decision, carry-forward table, row `039 / 042`.
- **Claim:** "content anchor (bounded)" is the carried-forward mechanism, with declarative home "script node injecting sibling content."
- **Evidence gap:** ADR-042 (bounded content anchor) itself records the bound being reverted after Spike τ′ showed it cost coherence with no measured benefit; the validated end-state Cycle-7 actually reached is ADR-039's **unbounded** all-prior-content selection. Labeling the row "(bounded)" asserts the opposite of what the source ADRs settled on. This is not a compression judgment call — it is a factual mischaracterization that could lead Cycle-8 BUILD to reintroduce a bound already shown to be inferior.
- **Recommendation:** Correct the row label to reflect the unbounded end-state (e.g., "content anchor (unbounded, per ADR-039; the ADR-042 bound was reverted)"), and note dependency-scoped selection remained an unresolved, deferred question rather than folding it silently into "bounded."
- **Routing:** N/A.

**P1-4. The carry-forward table's Provenance-check hedge covers only the mechanism→declarative-home mapping, not the accuracy of the "behavioral requirement" column itself — and at least three rows overstate settledness relative to their source ADRs' own conditional findings.**

- **Location:** ADR-045 §Decision, carry-forward table rows `033`, `036`, `041`; §Provenance check ("the mechanism→requirement→declarative-home mapping... is drafting-time synthesis; each row's 'declarative home' is a candidate... not a settled allocation").
- **Claim:** The table states "multi-turn loop participation; one grounded action per turn" (033), "reliable delegate-vs-carry per turn" (036), and "destination validity before marshalling" (041) as settled behavioral requirements to carry forward.
- **Evidence gap:** Each source ADR scoped its finding more narrowly than the table's flattened phrasing suggests, and the disclosed hedge doesn't cover this gap because it only addresses which *declarative home* a requirement should map to, not whether the requirement itself is finished business:
  - **033** — the domain model's own AS-9 annotations record the grounded-loop extension as accepted only *conditional* on framework-enforced single-action-per-turn, and explicitly name **axis-2 (sequential-composition drift over a long horizon)** as a still-open **BUILD/PLAY validation target**, never discharged by any later ADR. The table's "one grounded action per turn" reads as closed; the domain model itself says it isn't.
  - **036** — ADR-036's own evidence shows the delegation-reliability finding holds for one **(composition ensemble × qwen3:14b)** pairing and explicitly does **not transfer** to other models tested (1/5, 2/5 on other arms), requiring standing re-validation infrastructure. "Reliable delegate-vs-carry per turn" implies a general property; the source ADR is emphatic it is not one.
  - **041** — ADR-041's own status splits **Protection (discharged)** from **Convergence (Conditional Acceptance, unresolved)**: cheap-tier convergence is explicitly **not guaranteed**, coverage is narrow (only `.py`/`.json` checked; prose passes uninspected), and the long-horizon regime is named the axis-2 validation target deferred to PLAY. "Destination validity before marshalling" as a flat requirement erases this split.
  Because this is the same axis-2 thread running through three separate rows, and because the disclosed hedge doesn't mention it, the table understates the risk that Cycle-8's declarative rebuild is inheriting several genuinely open reliability questions as if they were closed requirements.
- **Recommendation:** Broaden the Provenance-check hedge to explicitly cover requirement-accuracy (not just declarative-home candidacy), and annotate rows 033/036/041 with their source ADRs' actual conditional/model-scoped status before Cycle-8 BUILD treats them as acceptance criteria.
- **Routing:** N/A.

**P1-5. AS-6 vs AS-11's "actor distinction" is coherent in the abstract but doesn't examine the case that falls between its two named actors — and P1-1/P1-2 show that case is currently occupied by live, un-superseded ADRs.**

- **Location:** ADR-044 §Decision, Corollaries, third bullet: "AS-11 governs framework design... It is distinct from AS-6 (which constrains what the runtime orchestrator may compose from — existing primitives only)."; §Provenance check flags this as "drafting-time synthesis... not from a driver document."
- **Claim:** The distinction implies two clean actors — the LLM-driven runtime orchestrator (bound by AS-6 to compose from a fixed palette) and the human framework developer (permitted by AS-11 to extend the engine) — and that no conflict exists between them.
- **Evidence gap:** ADR-015's tier-escalation router and ADR-021's per-capability dispatch are neither: they are neither the LLM orchestrator arranging existing primitives (AS-6) nor a framework developer's engine primitive (AS-11) — they are a third thing, a bespoke Python control-flow layer making the routing/escalation decision itself, sitting beside the DAG. The two-actor framing doesn't name or resolve this third case, so it reads as though AS-6/AS-11 jointly cover the whole space when in practice a currently-live decision layer falls outside both. This is the logical root of P1-1/P1-2, restated at the invariant level: the "drafting-time synthesis" resolves a *hypothetical* tension (AS-6 vs AS-11 in the abstract) while leaving unexamined the *actual* tension (a currently-authorized imperative dispatch layer vs. AS-11's blanket rule).
- **Recommendation:** Extend the AS-6/AS-11 scope note to explicitly classify where a mechanism like ADR-015's/ADR-021's router sits (framework-developer-authored control flow that is *not* an engine primitive) and state whether such mechanisms are AS-11-noncompliant by definition or need a named exception category.
- **Routing:** N/A.

**P1-6. AS-11's invariant text asserts dynamic dispatch as an already-existing engine primitive in the same breath as the two shipped primitives, when it is unbuilt — an internal-consistency gap with the ADR's own hedged language elsewhere.**

- **Location:** `domain-model.md` §Invariants, AS-11: "Agentic serving composes as llm-orc-native ensembles — DAGs of model-profile, script, and ensemble nodes plus engine control-flow primitives (guard/branch, bounded loop, dynamic dispatch)... Dynamic dispatch... is built as an engine primitive." Compare ADR-044 §Consequences (Neutral): "The shipped primitives (guard/branch, bounded loop) already satisfy the generalist flow; only dynamic dispatch remains to build as a primitive," and `engine-control-flow-state-and-next-steps.md` §2: "Primitive status: shipped = guard/branch + bounded loop... Planned = dynamic dispatch."
- **Claim:** The invariant text presents all three (guard/branch, loop, dynamic dispatch) as parallel members of "engine control-flow primitives," and states dynamic dispatch "is built as an engine primitive" in present tense.
- **Evidence gap:** Only two of the three exist. The PLAY evidence for dynamic dispatch is Ω-dispatch's ~15-line **adapter** (which PASSED), not an engine primitive — the "built as an engine primitive" framing is a forward design commitment grounded in AS-11's own policy ("never an adapter beside the engine"), not in PLAY evidence that an adapter is inadequate. ADR-044 is honest about this gap in its Neutral Consequences and Provenance check, but the invariant text itself — the artifact a future BUILD-phase engineer or conformance auditor is most likely to consult in isolation — doesn't carry that qualifier, and states it as settled fact. This is exactly the overreach the dispatch brief asked to check for.
- **Recommendation:** Amend the AS-11 invariant text (and ADR-044's Corollary bullet) to state dynamic dispatch as "to be built as an engine primitive" or "planned," matching the shipped/planned distinction already made correctly elsewhere in the same ADR.
- **Routing:** N/A.

### P2 — Should Fix

**P2-1. Terminology consistency: "orchestrator" is overloaded across at least three referents, compounding P1-5's ambiguity.**
- **Location:** AS-1 / AS-6 ("the orchestrator agent," an LLM-driven dynamic composer); AS-9 §Propagation note ("the orchestrator" = "the framework-driven dispatch pipeline," non-LLM, per ADR-027 direction); ADR-021 (`OrchestratorRuntime`, a bespoke Python driver class); ADR-044 Corollary 3 ("the runtime orchestrator," unqualified).
- **Claim:** ADR-044 uses "the runtime orchestrator" as if it is a single stable referent when distinguishing AS-6 from AS-11.
- **Evidence gap:** The domain model's own AS-9 propagation note already flags this drift ("Code references to 'orchestrator'... need clarification during BUILD") but ADR-044 doesn't cross-reference it or specify which referent AS-6 now binds to post-AS-9/ADR-027.
- **Recommendation:** Have ADR-044's AS-6/AS-11 corollary specify which "orchestrator" it means, or note the term is currently overloaded and point to the AS-9 propagation note as the tracking location for resolving it.

**P2-2. AS-3's "harness-level circuit breakers" and AS-1's "tool-mediated ensemble invocations" use vocabulary that overlaps with AS-11's forbidden list, unaddressed.**
- **Location:** AS-3 ("harness-level circuit breakers, not model-level parameters"); AS-1 ("tool-mediated ensemble invocations... analogous to CLI invocations"); AS-11 ("never to add a parallel orchestration layer (adapter, driver, or harness) beside it").
- **Claim:** Implicit — AS-11 doesn't claim to govern AS-1/AS-3's mechanisms.
- **Evidence gap:** Neither ADR-044 nor ADR-045 states whether AS-3's "harness" (budget/turn-limit circuit breaker) or AS-1's ad hoc tool-mediated dynamic invocation are in AS-11's scope (control-flow decisions) or categorically exempt (safety/session-control concerns, a different function). Given the corpus's history of exactly this kind of scope ambiguity resurfacing (e.g., "orchestrator"), leaving it implicit invites the same confusion.
- **Recommendation:** Add one sentence to AS-11's scope note distinguishing agentic control-flow decisions (in scope) from control-plane safety mechanisms like AS-3 (out of scope), and note whether AS-1's tool-mediated invocation model is superseded by the incoming dynamic-dispatch primitive or coexists with it.

**P2-3. ADR-044's "Positive Consequences" claim ("the corpus thesis becomes literally true") is scoped to architecture-shape but reads as more totalizing than the evidence supports once the PLAY closeout's own disposition is weighed.**
- **Location:** ADR-044 §Consequences (Positive).
- **Claim:** "The corpus thesis becomes literally true: each serving component is a declarative, independently-optimizable node..."
- **Evidence gap:** `play-closeout-2026-06-30.md` §Disposition states plainly: "the remaining walls are not the engine or the transport — they are executional grounding (§6.2b) and interactive speed (§6.3)." Both are unresolved and unrelated to the architecture-shape question ADR-044 settles. The claim is technically accurate (architecture shape *is* now declarative) but doesn't note that this doesn't move the needle on the two hardest reliability problems PLAY identified.
- **Recommendation:** Add a qualifying clause to the Positive Consequences bullet noting that architecture-shape and reliability (grounding, speed) are separate axes, and that this ADR resolves only the former. (See also Framing Audit P2-1.)

**P2-4. Neither ADR-044 nor ADR-045 distinguishes carry-forward findings that are architecture-agnostic (survive regardless of the AS-11 decision) from those that are control-flow-shape-dependent.**
- **Location:** ADR-045 §Decision, item 3 ("Carry forward the validated behavioral requirements, not the implementations").
- **Claim:** All eleven carry-forward rows are presented uniformly as "behavioral requirements the declarative form must still satisfy."
- **Evidence gap:** Several of the underlying findings are facts about clients/models, not about control-flow architecture — e.g., ADR-034's "the client must execute its own tools" and ADR-041's "a client-side refusal-as-stop ends the OpenCode loop" are client-protocol facts; ADR-036's "no model-layer coercion exists on this stack, the system slot loses the attention contest" and ADR-039's "models can't infer sibling APIs from names alone" are model-behavior facts. These hold under any control-flow mechanism, declarative or imperative, and aren't specifically *carried forward by the collapse* so much as *unaffected by it*. Folding them into the same table as the control-flow-dependent rows (033's loop structure, 036's escalation guard placement, etc.) risks Cycle-8 BUILD treating them as design choices to re-derive rather than fixed external constraints.
- **Recommendation:** Split the table (or add a column) distinguishing "control-flow-shape-dependent requirement" from "architecture-agnostic constraint (client/model fact, unaffected by this ADR)."

**P2-5. The carry-forward table's declarative-home column reads as definitive in-line; the hedge qualifying it is relegated to the document's final section.**
- **Location:** ADR-045 §Decision (table) vs. §Provenance check.
- **Claim:** The table header "Declarative home (Cycle-8 target)" and its Provenance-check hedge ("each row's 'declarative home' is a candidate... not a settled allocation") are separated by the whole Decision/Rejected-alternatives/Consequences body.
- **Evidence gap:** A reader stopping at the Decision section (the part most likely to be cited or acted on) would reasonably read the table as settled.
- **Recommendation:** Add a footnote marker directly on the table header pointing to the Provenance-check hedge, rather than relying on readers to reach the end of the document.

**P2-6. ADR-019's "the orchestrator" phrasing is an incidental holdover from the imperative era, unflagged.**
- **Location:** ADR-019 (skill-framework-agnostic capability library) assigns capability dispatch to "the orchestrator," drafted when that meant the imperative ReAct loop.
- **Claim:** ADR-019 is implicitly treated (by omission, since it's outside both the superseded set and the "needs review" list) as unaffected by AS-11.
- **Evidence gap:** The ADR's substantive three-layer separation and Topaz-keyed routing contract are implementation-agnostic and likely fine, but its incidental "orchestrator" phrasing carries the imperative-actor assumption forward without a note that ADR-027 already re-homed the same contract onto a declarative routing-planner ensemble.
- **Recommendation:** A one-line amendment note on ADR-019 clarifying the phrasing is legacy and the contract is actor-agnostic would close this quietly, at low cost, now rather than leaving it for the target-architecture ADR to rediscover.

### P3 — Consider

**P3-1.** `domain-model.md` Amendment Log #22 flags "AS-10's invariant text is absent from §Invariants although Amendment Log entry #14 records it as added" as pre-existing drift, "surfaced for the Cycle-8 conformance audit." This is disclosed rather than hidden, and isn't caused by ADR-044/045, but it's adjacent enough to this audit's invariant review to note it doesn't block AS-11's addition — no action needed here, just confirming it isn't silently compounding with the AS-11 changes.

**P3-2.** ADR-045's Neutral Consequences state "the eleven superseded ADRs remain as historical records (body-immutable) with dated 'Superseded by ADR-045' headers" — fork verification confirms all eleven carry this header accurately. Minor: ADR-043's header phrasing differs slightly from the other ten's boilerplate ("one-serving-surface requirement... carried forward... (one declarative serving ensemble)" vs. the standard template). Not a defect, just a consistency nit worth aligning if the headers are ever regenerated.

## Section 2: Framing Audit

### Question 1: What alternative framings did the evidence support?

**Alternative framing A — scoped extension, not a blanket ex-ante invariant.** The PLAY evidence supports a narrower claim than AS-11's: "the two shipped primitives (guard/branch, loop) sufficiently express the (b) generalist flow; extend the engine further only when a specific future flow demonstrably needs it." AS-11 instead generalizes to a project-wide rule governing *all* future serving-layer control flow, including areas PLAY never tested (budget enforcement, calibration gates, session records — all named in ADR-044's own inventory of the collapsed layer, none of which appear in the carry-forward table). What would the reader need to believe for the narrower framing to be right? That architectural principles should be earned incrementally, flow by flow, consistent with this corpus's own Empirical-Grounding Filter (ADR-097) discipline — which ADR-044 invokes for AS-11's *core* claim but not for its extension into a standing invariant governing future, untested territory.

**Alternative framing B — a mechanism-level boundary (control-flow decisions vs. transport/protocol handling) rather than a blanket ban on non-declarative Python.** Ω-serve's three real-client fixes (SSE streaming, aux-call short-circuiting, GPU-thrash avoidance) are all necessarily imperative Python, and PLAY treated them as ordinary engineering, not AS-11 violations. This suggests the evidence actually supports a narrower, cleaner boundary: "control-flow decisions about what capability to invoke next must be engine primitives; transport/protocol handling is unaffected." AS-11's actual language ("adapter, driver, or harness") is broader than this and is exactly what creates the P1-1/P1-2/P2-2 ambiguities above. Under this framing, the ADR-015/021/027 tension would have been visible and addressable at drafting time rather than discovered by audit.

**Alternative framing C — given a demonstrated documentation-fidelity gap (P1-3/P1-4), retain the finished code as inert reference material rather than deleting it outright.** This does not reopen the settled clean-slate-vs-strangler decision (a production-migration-strategy question) — it's a narrower question ADR-045 doesn't address: whether the ~12.7K LOC, once superseded, is kept anywhere as ground truth to check the carry-forward table against, or relies entirely on git history plus the table itself. Given P1-3 already found a factual error in the table (the 042 "bounded" mislabel), an evidence-grounded framing would ask whether "carry forward the requirements, not the implementation" (ADR-045 item 3) is an adequate risk mitigation on its own, or whether the implementation should remain consultable (even if inert, unrun, and not stranger-migrated) until Cycle-8 BUILD has verified parity.

### Question 2: What truths were available but not featured?

- **PLAY's own disposition that the remaining walls are executional grounding and interactive speed, not architecture** (`play-closeout-2026-06-30.md` §Disposition: "the remaining walls are not the engine or the transport"). ADR-044's Consequences don't mention either wall. A reader relying solely on ADR-044/045 could conclude the hard part of Cycle-7→8 migration reliability is resolved; PLAY's own words say the opposite — the architecture question was never the hard part.
- **The settled full-model-parity requirement** (`product-discovery.md` Cycle-8 addition: "an llm-orc agent-serving process uses ensembles to do EVERYTHING that a single model would do... no degradation in capability") and **PLAY note 5** (the current serve is narrow, build-new-files only). ADR-044/045 are silent on how AS-11 relates to this settled requirement — reasonably, since it's explicitly deferred to the target-architecture ADR — but neither document states this scoping explicitly, so the omission reads as unexamined rather than deliberately out of scope.
- **Architecture-agnostic findings buried in the carry-forward table** (see P2-4): client-protocol and model-behavior facts that will hold regardless of which control-flow architecture wins, presented identically to control-flow-dependent requirements.

### Question 3: What would change if the dominant framing were inverted?

Dominant framing: *the imperative layer is technical debt; clean-slate collapse into a declarative architecture is the correct target, and the collapse is the main achievement of this DECIDE round.*

Inverted framing: *the imperative layer is eleven ADRs' worth of hard-won, iteratively-discovered failure-mode mitigations (single-action-per-turn enforcement, termination composition, remaining-work anchors, cross-file coherence anchors, completeness gates, destination-validity gates); the declarative replacement has so far been validated on exactly one happy-path demo flow (Ω-P3) plus one major false-pass failure it did not prevent (the cross-file coherence leak). The burden of proof is on the replacement to demonstrate it handles the same edge cases the old system needed eleven separate ADRs to discover, not on the old system to justify its continued existence.*

Under this inversion:
- **Claims that weaken:** ADR-044's "corpus thesis becomes literally true" (Positive Consequences) looks like a shape-level win only, not a reliability win, once weighed against P1-4's finding that axis-2 (long-horizon drift), model-pair-specific reliability (036), and cheap-tier convergence guarantees (041) are all still open.
- **Evidence that becomes more salient:** the fork-verified findings that several "settled" carry-forward rows are actually conditional or model-scoped (P1-3, P1-4) move from footnotes to central risks — they are exactly what a declarative rewrite risks re-discovering the hard way if the table is trusted at face value.
- **What the documents would need to address:** an explicit, operationalized parity checklist derived from the *true* (not flattened) conditional scope of all eleven carried-forward requirements, which Cycle-8 BUILD must pass before `src/llm_orc/agentic/` is deleted. ADR-045 currently says only that "removal follows... reaching behavioral parity... it does not precede it" — parity is asserted as a gate but never defined operationally.

### Framing Issues

**P2 (Should Fix):**
- **Q2 finding — PLAY's grounding/speed disposition underrepresented.** See P2-3 above (Section 1); cross-referenced here because it is fundamentally a framing choice (foregrounding the architecture question) rather than a logic error.
- **Q1 framing B — mechanism-boundary alternative not examined.** Recommend ADR-044 add a scope note distinguishing control-flow decisions from transport/protocol handling, both to sharpen AS-11's boundary and to make the ADR-015/021/027 tension (P1-1/P1-2) visible at the invariant-drafting stage rather than only via audit.
- **Q1 framing C — no-code-retained choice forecloses ground-truth verification.** Given P1-3's demonstrated table error, recommend the practitioner consider (as a narrow, separate decision from the settled clean-slate-vs-strangler question) whether the superseded `agentic/` module should remain consultable somewhere until Cycle-8 BUILD confirms parity, rather than relying solely on the carry-forward table and git history.

**P3 (Consider):**
- **Q2 finding — full-model-parity scoping silence.** Recommend a one-line explicit scoping disclaimer in ADR-044 or ADR-045 stating that capability scope (what the serve must be able to do) is out of scope for this decision pair, which governs mechanism shape only.

## Convergence-Saturation Signal (ADR-094)

**Convergence-Saturation Signal:** NOT TRIGGERED
- Round number: R1 (first audit dispatch on this document pair; ADR-044/045 are newly Accepted this cycle, and no prior `argument-audit-decide-cycle-8*` artifact exists — form-change baseline reset applies trivially since there is no predecessor form to reset from)
- P1 count this round: 6 (Section 1: P1-1 through P1-6; Section 2: 0)
- P2 count this round (new, non-carry-over): 12 (Section 1: P2-1 through P2-6; Section 2: 3 distinct framing P2s, counted once each despite cross-referencing P2-3)
- New framings or claim-scope expansions: all three Question-1 alternative framings (scoped-extension framing; mechanism-boundary framing; retain-code-as-reference framing) are new this round, as is the axis-2/model-scope conditionality thread surfaced across P1-4's three rows
- Recommendation: **CONTINUE to next round.** P1 count (6) and new-P2 count (12) both far exceed the trigger thresholds (P1 = 0, new P2 ≤ 1) on their own; no need to invoke the ESCALATE branch since conditions 1 and 2 already fail conclusively. Recommend the practitioner address the P1 findings (particularly P1-1/P1-2's silent-scope-gap and P1-3/P1-4's carry-forward-table integrity issues) before re-dispatching R2.

*This is a standard-sequence audit, not a re-audit-after-revision; the verdict line above is required and included.*

---

## R2 Re-Audit (2026-07-01) — Adversarial Verification

**Audited document (current state):** `adr-044-declarative-ensemble-native-serving.md`, `adr-045-clean-slate-collapse-imperative-serving-layer.md` (both revised since R1)
**Additional material read this round:** `adr-001-internal-react-loop-execution-model.md`, `adr-012-conversation-compaction-five-layer-pipeline.md`, `adr-016-upward-l0-l1-read-only-signal-channel.md`, `adr-018-tier-escalation-router-audit-dispatch.md`, `adr-015-per-role-tier-escalation-router.md` (full body), `adr-027-framework-driven-dispatch-pipeline.md` (header), `adr-043-collapse-dual-serving-surfaces-to-one-loop.md` (header), `adr-008-per-session-autonomy-levels.md` (header), `domain-model.md` Amendment Log #22 (current), `housekeeping/cycle-status.md`, and the actual `src/llm_orc/agentic/` source tree + git history (empirical verification, not just document cross-reference)

### R1 P1 Resolution Verification

| R1 finding | Resolved? | Evidence |
|---|---|---|
| P1-1 (ADR-015/021 silent scope gap) | **Yes** | Both files now carry accurate `Superseded by ADR-045` headers naming the specific retired mechanism (`tier_router.py`, `OrchestratorRuntime`) and the carried-forward requirement. |
| P1-2 (ADR-027 mischaracterized as "direction to review") | **Yes** | ADR-045 now states the mechanism "was already superseded via ADR-043 (chain 027→043→045)" with the framework-driven *direction* (not the dead pipeline) surviving as a design input — correctly separates mechanism from principle. |
| P1-3 (042 "bounded" mislabel) | **Yes** | Row now reads "039 (042 reverted) — unbounded, all-prior-siblings selection," matching the source ADRs' actual end-state. |
| P1-4 (033/036/041 conditionality erased) | **Yes** | All three rows now carry inline caveats (axis-2 open; profile-bound per Spike ψ′ Arm D; protects-but-does-not-recover), and the Provenance check now explicitly hedges the "behavioral requirement" column's settledness, not just the declarative-home mapping — closes the exact gap R1 identified. |
| P1-5 (AS-6/AS-11 actor distinction doesn't examine the third case) | **Yes, operationally** | ADR-044's corollary prose is unchanged, but ADR-045's mechanism test now explicitly resolves the practical instance (ADR-015/021 classified, not left in limbo). Residual: ADR-044 itself still doesn't cross-reference the mechanism test (P3 below). |
| P1-6 (dynamic dispatch stated as settled fact) | **Yes** | AS-11 text and the corollary now both say "to be built... not yet shipped," matching the Neutral Consequences language. |

All six R1 P1s are genuinely resolved, not merely gestured at — each fix targets the specific textual mechanism the original finding named, not a generic disclaimer bolted on elsewhere. **However, the fix for P1-1/P1-2/P1-5 (replacing proximity-based supersession with a mechanism test) is a structural change, not a narrow patch, and it introduces new classification work — verified below and found to contain new errors.**

### New Issues Introduced by the R2 Revision

**P1-NEW-1. The mechanism test is misapplied to ADR-012: a deterministic data-management mechanism is classified as "mechanism-superseded" when the test's own criterion should place it in "survives, re-homed."**

- **Location:** ADR-045 §Decision item 1: "peer orchestration mechanisms ADR-001... ADR-012 (compaction pipeline)... are mechanism-superseded and listed in this manifest."
- **Claim:** ADR-012's five-layer conversation-compaction pipeline is classified alongside ADR-001/015/021 as an imperative-orchestration mechanism AS-11 forbids, destined for the same clean-slate collapse.
- **Evidence gap:** Verified empirically: `src/llm_orc/agentic/conversation_compaction.py` exists (commit `31e261f feat: add Conversation Compaction five-layer pipeline (WP-E4, ADR-012)`, tested by `tests/integration/test_compaction_multi_turn_cycle_acceptance.py`), so the code-location half of the classification is correct — it will be deleted with the directory. But ADR-045's own mechanism test asks a *decision-content* question, not a location question: "imperative-orchestration-mechanism → superseded; contract/invariant/direction → survives." ADR-012's decision (persist-large-results / cache-edit / idle-expiry / session-notes template / LLM-summary-last-resort, cheapest-first) is a **deterministic context-management pipeline** — it does not decide what the agent does next; it manages what the orchestrator's context holds. AS-11 itself says "deterministic serving work is a `script:` node" — exactly the shape ADR-012's mechanism already has. By the mechanism test's own stated criterion, ADR-012 belongs in the "survives, re-homed" bucket next to ADR-024/025/026, not the "mechanism-superseded" bucket next to ADR-001/015/021 (which genuinely decide runtime control-flow/dispatch actions). Filing a working, literature-grounded (Claude Code's published pattern, adopted with explicit provenance discipline in ADR-012), already-validated pipeline under "superseded" risks Cycle-8 BUILD either wastefully re-deriving it from scratch or silently dropping it because nobody flagged it as something to *intentionally* re-home rather than redesign.
- **Recommendation:** Reclassify ADR-012 into the "survives, re-homed declaratively (as a `script:` node or per-turn node stage)" bucket. Distinguish, in the mechanism test's stated criterion, "decides an agentic action" from "manages data/context deterministically" — the two are currently conflated by classifying strictly on code location.

**P1-NEW-2. ADR-016 is classified as "mechanism-superseded" by a passing mention in a different ADR, which (a) bypasses ADR-016's own explicitly specified governance process and produces a live contradiction with it, and (b) classifies a dependent extension more definitively than the base mechanism it extends.**

- **Location:** ADR-045 §Decision item 1 ("ADR-016 / ADR-018 (upward signal channel + router-audit dispatch)... are mechanism-superseded") vs. ADR-016 §Empirical validation pathway §"Sweep responsibility": "the practitioner reviews this status at the end of each cycle that exercises the cross-layer channel. The cycle-status `Phase Status` table for any cycle that touches ADR-016 includes a row noting the channel's status (conditional / fully accepted / superseded). **Absence of the row constitutes structural evidence that the cycle has not exercised the channel.**"
- **Claim:** ADR-045 asserts ADR-016's mechanism is already superseded as of this cycle.
- **Evidence gap:** Verified: `housekeeping/cycle-status.md` contains **no row and no mention of ADR-016 at all** for Cycle 8. By ADR-016's own specified evidentiary convention — which this audit takes at face value because ADR-016 wrote it as a structural, load-bearing part of its acceptance (WP-H4 full-acceptance disposition explicitly kept "Sweep responsibility... remains in force") — the absence of that row is defined to mean Cycle 8 has *not* touched ADR-016. ADR-045's manifest text directly contradicts this: it asserts the channel *is* being touched, in fact retired. This isn't a bookkeeping nicety optionally skipped; ADR-045's "the corpus is an isolated branch... exhaustive per-ADR header bookkeeping is not warranted" justification (Consequences, Neutral) doesn't reach this case, because ADR-016 assigns evidentiary *meaning* to the row's absence, unlike the other manifest-listed peers. Separately, `agentic/calibration_signal_channel.py` (confirmed physically present, commit `2fd9a55`) is empirically the collapsed mechanism, so this is not a wrong-location classification the way a naive reading might assume, and the underlying supersession call is plausibly correct on the merits — the problem is procedural and ordering-based, not a location error. **Ordering problem:** ADR-045's same Decision item 1 places ADR-016's own dependency, **ADR-007/014 (the base Calibration Gate ADR-016 explicitly "Composes with")**, in a *third*, less-settled bucket: "Borderline cases where a requirement survives but its mechanism lives in `agentic/`... are classified by the target-architecture ADR." It is internally inconsistent to definitively retire an *extension* of a mechanism (ADR-016 extends ADR-014's Calibration Gate) while the *base* mechanism's own classification is still explicitly open and deferred.
- **Recommendation:** Either (a) actually run ADR-016's specified Sweep-Responsibility process — add the cycle-status row, let the practitioner render the trigger-action disposition ADR-016 itself specifies (full supersession / preserved-conditional / falsification-trigger-fired) — before asserting supersession in ADR-045's text, or (b) move ADR-016/018 into the same "borderline, deferred to target-architecture ADR" bucket as their own dependency ADR-007/014, so the classification ordering is consistent (extensions aren't more settled than their base).

**P2-NEW-1. ADR-001's "mechanism-superseded" classification doesn't flag that AS-11 and ADR-001 answer different questions, and PLAY evidence suggests ADR-001's actual decision (internal loop ownership) may be reversing without an explicit decision record.**

- **Location:** ADR-045 §Decision item 1 ("ADR-001 (internal ReAct loop)... mechanism-superseded"); compare ADR-001 §Decision: "The agentic serving feature implements an internal ReAct loop inside the llm-orc server... The external MCP model remains available... it is not replaced" (i.e., ADR-001 explicitly chose internal-loop ownership over the external/client-owned alternative).
- **Claim:** ADR-001 is superseded as one more instance of the imperative-orchestration shape AS-11 forbids.
- **Evidence gap:** Verified `src/llm_orc/agentic/orchestrator_runtime.py` (commit `b4e6f43`, the earliest of the three, 2026-04-21) is indeed the ADR-001 mechanism, so code-location is correct. But AS-11 is about control-flow *shape* (ensemble DAG vs. imperative driver), while ADR-001 decided control-flow *ownership* (internal orchestrator vs. external client loop) — a different axis. The PLAY architecture actually built (`ensemble-agent-state-and-next-steps.md` §4: "OpenCode owns the loop. A thin serving adapter runs ONE ensemble turn per request") resembles the **external** model ADR-001 explicitly rejected in April, not a declarative reimplementation of the **internal** model ADR-001 chose. Filing this under "mechanism-superseded" (implying only the *implementation* changes) obscures that the loop-ownership *question* ADR-001 settled may be getting re-opened and reversed by the emerging architecture, without any ADR explicitly re-litigating ADR-001's rejected alternatives.
- **Recommendation:** Have the Cycle-8 target-architecture ADR explicitly address loop ownership as its own question (not inherited-by-default from "ADR-001's mechanism is superseded"), referencing ADR-001's original external/internal/hybrid framing directly.

**P2-NEW-2. The manifest doesn't cover all 32 files in the collapsed directory — at least two ADRs (008, 009/010) are absent from all three classification buckets.**

- **Location:** ADR-045 §Decision item 1 (superseded / survives / borderline-deferred buckets).
- **Claim (implicit):** The manifest, while disclaiming full authoritative completeness, presents itself as covering the load-bearing classification work for this cycle.
- **Evidence gap:** `src/llm_orc/agentic/autonomy_policy.py` (implementing `AutonomyPolicy`, matching **ADR-008** per-session autonomy levels) and `src/llm_orc/agentic/plexus_adapter.py` (matching **ADR-009/010** Plexus integration) are both physically inside the collapsed directory and appear in none of the manifest's superseded, survives, or borderline-deferred lists — unlike ADR-007/013/014/017/023, which at least got an explicit "borderline" acknowledgment.
- **Recommendation:** Add ADR-008 and ADR-009/010 to the borderline-deferred bucket for consistency, or note explicitly that the manifest is a sample of the clearest/most consequential cases rather than implying broader coverage.

### P3 — Consider (new this round)

- ADR-012's `Status:` field still reads "Proposed" although the mechanism was built and shipped (WP-E4, commit `31e261f`, integration-tested) — a pre-existing status/implementation mismatch, unrelated to but surfaced by this round's file-level verification; worth flagging for the conformance audit regardless of ADR-012's R2 classification.
- ADR-018 (amends ADR-015, which now carries a `Superseded by ADR-045` header) itself carries no header or update note — the same silent-orphan shape R1 flagged for ADR-015/021 recurring at smaller scale, one hop downstream.
- ADR-027's header still points only to ADR-043 (not signposting the further chain to ADR-045); a reader landing on ADR-027 directly must make a two-hop traversal to learn the mechanism is now doubly retired.
- ADR-044's AS-6/AS-11 corollary prose is byte-for-byte unchanged from R1 and still doesn't cross-reference ADR-045's new mechanism test, even though the mechanism test is what actually resolves the practical ambiguity the corollary raised.

### Framing Audit — Re-run on the Revised Framing

The R2 edits don't change the dominant framing (declarative-ensemble-native vs. forbidden parallel layer) — they change how supersession scope is drawn. Re-running the three framing questions against the current text:

**Q1 update.** Framing B from R1 (a mechanism-level boundary rather than a blanket ban) is now partially *adopted* — the mechanism test is exactly that boundary. But this round's findings show the boundary is harder to apply consistently than the abstract test suggests: a pure data-management mechanism (ADR-012) got swept into the same bucket as genuine control-flow deciders (ADR-001/015/021) on the strength of shared code location alone. This is evidence *for*, not against, Framing C (retain the code as reference): if classifying which of 32 files' *decisions* survive is this error-prone under deliberate audit scrutiny — two misclassification-adjacent issues found spot-checking four of the thirty-two — the value of keeping the actual implementation consultable as ground truth (even inert, unrun) rises, not falls, as a hedge against classification drift during the Cycle-8 rebuild.

**Q2 update.** Unchanged from R1 — PLAY's grounding/speed disposition and the full-model-parity scope silence are unaffected by this round's edits.

**Q3 update.** This round's findings are direct, concrete instances of the R1 inversion's warning: ADR-012 is exactly the kind of "hard-won, working, validated mechanism" the inverted framing said was at risk of being lost or wastefully re-derived under clean-slate pressure — not a hypothetical this time, but a specific pipeline with a specific commit and a specific test file, misfiled in the very manifest built to prevent this. This sharpens rather than replaces the R1 Q3 finding.

**New Framing Issue (P2):** The mechanism test's real-world fragility (P1-NEW-1/P1-NEW-2 above) is itself framing-relevant: ADR-045 presents the manifest as resolving R1's scope-line problem, but the fix's own track record on a 4-of-32 spot check suggests the manifest should be framed as a *provisional, error-prone first pass* pending the target-architecture ADR's full review — not as settled classification work, which is closer to how the current prose reads.

### R2 Convergence-Saturation Signal (ADR-094)

**Convergence-Saturation Signal:** NOT TRIGGERED
- Round number: R2
- P1 count this round: 2 (P1-NEW-1, P1-NEW-2 — both newly introduced by the R2 revision; all six R1 P1s are resolved and not recounted)
- P2 count this round (new, non-carry-over): 3 (P2-NEW-1, P2-NEW-2, and the new framing P2 on mechanism-test fragility)
- New framings or claim-scope expansions: the mechanism-test-fragility framing (Q1/Q3 update above) is new this round; no other new claim-scope expansions beyond what the mechanism-test revision itself introduced
- Recommendation: **CONTINUE to next round.** P1 count (2) alone is sufficient to fail the trigger condition. Recommend the practitioner resolve P1-NEW-1 (reclassify ADR-012) and P1-NEW-2 (either run ADR-016's own Sweep-Responsibility process or move ADR-016/018 to the borderline-deferred bucket) before dispatching R3. Given R1's six P1s were all cleanly resolved on the first attempt, R3 is plausibly the closing round if these two are addressed with the same directness — but that is a prediction, not a finding; R3 should still adversarially re-verify rather than assume.

*This is a standard-sequence re-audit (verifying repair of R1's P1 findings while adversarially checking for revision-introduced regressions), not a single-purpose re-audit-after-revision in the narrow sense the orchestrator rule describes — the coordinator's brief asked for both resolution verification and fresh adversarial review, so the Convergence-Saturation Signal applies per the standard sequence.*

---

## R3 Re-Audit (2026-07-01) — Adversarial Verification of the Claim-Reduction

**Audited document (current state):** `adr-045-clean-slate-collapse-imperative-serving-layer.md` (Decision item 1 + Consequences revised); `adr-044-declarative-ensemble-native-serving.md` (unchanged since R2)
**Additional material re-checked this round:** `domain-model.md` Amendment Log #22 (current), `housekeeping/cycle-status.md` (current, checked for an ADR-016-specific Sweep-Responsibility row)

### R2 Finding Resolution Verification

| R2 finding | Resolved? | Evidence |
|---|---|---|
| P1-NEW-1 (ADR-012 misclassified "superseded") | **Yes, and more conservatively than the R2 recommendation asked.** | ADR-012 moved to the ten-item deferred list rather than being reclassified straight to "survives." Given R2's own point was that hasty classification (in either direction) had already produced errors, deferring pending the full per-ADR audit is a more defensible fix than substituting one quick judgment for another. |
| P1-NEW-2 (ADR-016 superseded against its own governance) | **Yes.** | ADR-016 moved to the deferred list; the premature "touched/retired" claim is withdrawn, so the contradiction with ADR-016's "absence of a cycle-status row = not exercised" governance dissolves (there is no cycle-status row for ADR-016, and ADR-045 no longer claims there should be one yet). Guidance item (c) explicitly preserves the governance rule for whoever performs the eventual classification. |
| P2-NEW-1 (ADR-001 loop-ownership axis conflated with AS-11's control-flow-shape axis) | **Yes.** | ADR-001 moved to the deferred list; guidance item (d) states verbatim: "ADR-001 is classified on the loop-*ownership* axis (OpenCode owns the loop) not only the control-flow-shape axis" — carries the exact distinction R2 raised forward as an explicit instruction, not just a deferral. |
| P2-NEW-2 (ADR-008/009/010 omitted from the manifest) | **Yes.** | Both now named explicitly in the deferred list: "ADR-008 (autonomy), ADR-009/010 (Plexus)." |

All four R2 findings are genuinely resolved via one coherent mechanism — reducing the manifest's confident-classification claims to the cases with unambiguous evidence (033–043, 015, 021, 027-via-043) and explicitly enumerating everything else as deferred with carried-forward guidance — rather than four independent patches. This is a structurally sound response, not a relabeling exercise.

### Does the Claim-Reduction Introduce a New Gap?

**P1-NEW-3 (R3). ADR-044's Context still cites "calibration gates" and "session records" with the same confident rhetorical certainty as "Loop Driver" and "tier router," but ADR-045 (R3) now explicitly defers classification of exactly those two clusters — an unrevised inconsistency between the two companion ADRs.**

- **Location:** ADR-044 §Context (unchanged since R1): "Cycle 7 nonetheless reached its serving behavior through a bespoke imperative layer in `src/llm_orc/agentic/`... : Loop Driver, tier router, orchestrator tool dispatch, calibration gates, session records." Compare ADR-045 §Decision item 1 (current): the confident cases are "the loop-driver chain ADR-033 through ADR-043, plus ADR-015... and ADR-021," while "ADR-007/014 (calibration gate)... ADR-013 (session registry)... ADR-016 (upward signal channel)... is classified by the Cycle-8 target-architecture ADR."
- **Claim:** ADR-044's illustrative list presents all five named components — Loop Driver, tier router, orchestrator tool dispatch, calibration gates, session records — as equally clear instances of "the parallel orchestration layer AS-11 forbids."
- **Evidence gap:** Three of the five (Loop Driver→033, tier router→015, orchestrator tool dispatch→021) are now confirmed confident cases in ADR-045. The other two are not: "calibration gates" is ambiguous between the confidently-superseded Cycle-7-specific gates (ADR-040 completeness gate, ADR-041 destination-validity gate, both already inside the 033–043 confident chain) and the cross-cutting Calibration Gate lineage (ADR-007/014, extended by ADR-016) that ADR-045 R3 explicitly defers as not-yet-classified — it could turn out to be a contract/direction survivor, not a forbidden-shape casualty. "Session records" maps unambiguously to ADR-013, which ADR-045 R3 also explicitly defers. So two of ADR-044's five headline examples of the problem AS-11 exists to solve are, by its own companion ADR's admission, not yet known to actually be instances of that problem. ADR-044's Context, unrevised across three rounds, still states them with the same certainty as the three genuinely confident cases.
- **Recommendation:** Either disambiguate "calibration gates" in ADR-044's Context to name specifically ADR-040/041 (if that narrower reading was intended, in which case there is no conflict and the fix is purely a clarifying word choice), or, if the broader Calibration Gate lineage was intended, revise the sentence to flag ADR-007/014/016 and ADR-013 as deferred-classification items consistent with ADR-045's current text, rather than listing them as settled illustrations of the forbidden pattern.

**P2-NEW-3 (R3). ADR-018's deferred classification, despite amending the now-confidently-superseded ADR-015, is not explained in the text — a reader can reconstruct a defensible rationale (dual parentage: ADR-018's audit-dispatch mechanism is structurally derived from the deferred ADR-016, so it inherits the more conservative classification) but ADR-045 doesn't say this.**

- **Location:** ADR-045 §Decision item 1 — ADR-015 is a confident case; ADR-018 (amends ADR-015) is in the ten-item deferred list.
- **Claim (implicit):** The confident/deferred split is fully explained by the stated guidance (mechanism test; extension-not-more-definitive-than-base; ADR-016 governance; ADR-001 axis).
- **Evidence gap:** Guidance item (b) ("an extension is not classified more definitively than its base") explains why ADR-016 (extends ADR-014) can't be more confident than its base. It does not, on its face, explain the opposite-looking case: ADR-018 extends ADR-015, whose classification IS confident, yet ADR-018 itself is deferred. The likely reason — ADR-018's audit-dispatch pattern is explicitly "structurally analogous to ADR-016 mechanism (d)" (per ADR-018's own Provenance check: "ADR-016 mechanism (d) is the structural precedent"), so it has two parents of different confidence and correctly inherits the more cautious one — is a coherent generalization of guidance (b), but it isn't stated. A reader comparing "015 confident, 018 (amends 015) deferred" without that context would reasonably read it as an unexplained inconsistency, which is exactly the kind of ambiguity this manifest was rewritten to eliminate.
- **Recommendation:** Add one clause to guidance item (b) generalizing it to multi-parent cases: "an extension inherits the least-confident classification among its dependencies, not only its nominal base" — and note ADR-018's dual derivation from ADR-015 (confident) and ADR-016 mechanism (d) (deferred) as the concrete instance.

### P3 — Consider (new this round)

- ADR-045's "Rejected alternatives" §"Two scoping errors the mechanism test avoids" still names only the original two errors (proximity-based under-scoping; blanket over-scoping) from R1. It doesn't name the third, R2-discovered error (a single mechanism test applied hastily to peer ADRs still produces misclassifications, e.g., ADR-012/016) as a distinct lesson, even though Decision item 1's prose now describes it inline. Folding it into the Rejected-alternatives list would complete the provenance trail in the place a reader would look for it.

### Is the Deferral Honest, or Does It Just Move the Classification Out of Audit Range?

**Assessed as honest, not a dodge**, for four reasons: (1) the deferred set is a closed, fully enumerated list of ten items (001, 007/014, 008, 009/010, 012, 013, 016, 017, 018, 023), not an open-ended "etc."; (2) it names the specific future owner (the Cycle-8 target-architecture ADR) and cycle-status.md's own "Next:" line already sequences that ADR immediately after this argument audit — it is a near-term, not indefinite, commitment; (3) it carries forward four concrete, substantive constraints (the mechanism test; extension-inherits-least-confident-parent; ADR-016's self-governance; ADR-001's loop-ownership axis) derived directly from two rounds of caught errors, rather than punting with no guidance; (4) verified empirically that the specific contradiction this deferral was built to avoid (ADR-016's cycle-status-row governance) is in fact no longer contradicted — there is still no row, and ADR-045 no longer claims there should be one yet, so the record and the claim are now consistent with each other. The main residual risk is that "first task of the target-architecture ADR" is not a hard gate (nothing currently prevents that future ADR from being drafted without actually completing the ten-item audit) — worth a light process note, but not evidence the current deferral is evasive.

### R3 Convergence-Saturation Signal (ADR-094)

**Convergence-Saturation Signal:** NOT TRIGGERED
- Round number: R3
- P1 count this round: 1 (P1-NEW-3 — the ADR-044/ADR-045 Context-vs-deferred-list inconsistency; all four R2 findings are resolved and not recounted)
- P2 count this round (new, non-carry-over): 1 (P2-NEW-3 — ADR-018's unexplained deferred classification)
- New framings or claim-scope expansions: none — the dominant framing is unchanged this round; the claim-reduction is a confidence/scope calibration on the supersession manifest, not a reframing of AS-11 or the clean-slate decision
- Recommendation: **CONTINUE to next round.** P1 count (1) is nonzero, so the signal cannot trigger regardless of the otherwise-strong convergence trend (four-for-four resolution this round, both new findings lower-severity and narrower in scope than R1/R2's, one of the two being a single-sentence fix). Recommend closing P1-NEW-3 (reconcile or disambiguate ADR-044's "calibration gates" / "session records" Context language against ADR-045's current deferred list) before R4. Given the shrinking finding count and severity across R1 (6 P1) → R2 (2 P1) → R3 (1 P1), R4 is a plausible closing round if this single sentence is fixed cleanly — a trend observation, not a substitute for R4's own adversarial check.

*This is a standard-sequence re-audit; the verdict line above is required and included.*

---

## R4 Re-Audit (2026-07-01) — Adversarial Verification of the Context Reconciliation

**Audited document (current state):** `adr-044-declarative-ensemble-native-serving.md` (§Context revised), `adr-045-clean-slate-collapse-imperative-serving-layer.md` (§Decision item 1 + §Rejected alternatives revised)
**Verification method:** cross-checked every item named in ADR-044's revised Context sentence against ADR-045's confident/deferred split, plus a fresh empirical check of the `src/llm_orc/agentic/` source (docstrings) for the categories not yet individually verified in prior rounds.

### R3 Finding Resolution Verification

| R3 finding | Resolved? | Evidence |
|---|---|---|
| P1-NEW-3 (ADR-044 Context vs. ADR-045 deferred-list inconsistency) | **Yes, for every item ADR-044 now names — see the verification table below.** | ADR-044's Context now explicitly splits "imperative-orchestration mechanisms" (Loop Driver, tier router, orchestrator tool dispatch) from "other contents" (calibration gates, session records, conversation compaction), states "All of the layer is removed by the clean-slate; not all of it is AS-11-forbidden by decision-content," and cross-references ADR-045. Checked against ADR-045's actual confident/deferred split, every one of ADR-044's six named items lands in the matching bucket (table below) — **except one residual ambiguity found this round, see P1-NEW-4.** |
| P2-NEW-3 (ADR-018 deferral rationale unexplained) | **Yes.** | ADR-045's deferred list now reads verbatim: "ADR-018 (router audit — deferred despite extending the confidently-superseded ADR-015, because its audit-dispatch mechanism structurally derives from the deferred ADR-016 signal channel and inherits the more cautious classification)." Matches the R3 recommendation exactly. |
| P3 (third scoping error not named in Rejected alternatives) | **Yes.** | ADR-045's Rejected alternatives now reads "Three scoping errors the manifest-of-record + deferral avoids," adding "(iii) Hasty full-classification via the mechanism test alone also misfires — R2/R3 caught it over-classifying deterministic-work ADR-012... and governance-protected ADR-016." |

**Verification table — every item ADR-044's revised Context names, checked against ADR-045's current split:**

| ADR-044 label | Maps to | ADR-045 bucket | Match? |
|---|---|---|---|
| Loop Driver | ADR-033 | Confident (033–043 chain) | Yes |
| tier router | ADR-015 | Confident | Yes |
| orchestrator tool dispatch | ADR-021 (and/or ADR-027, chained via 043) | Confident | Yes (both readings land confident) |
| calibration gates | ADR-007/014 + ADR-016 | Deferred | Yes |
| conversation compaction | ADR-012 | Deferred | Yes |
| session records | ADR-013 (Session Registry) **+ ADR-037 (Session Action Record)** | Deferred **for ADR-013; ADR-037 is already CONFIDENT (part of the 033–043 chain, carry-forward row `037`)** | **Partial mismatch — see P1-NEW-4** |

### New Issue Found This Round

**P1-NEW-4. "Session records" is asserted as uniformly deferred, but the category spans a mechanism that is already confidently superseded — the identical error shape R3 just caught for "calibration gates," recurring uncaught in the same revision for "session records."**

- **Location:** ADR-044 §Context: "Other contents (calibration gates, session records, conversation compaction) are deterministic-work or contract implementations that physically live in the same layer but whose per-ADR disposition... is classified by the target-architecture ADR."
- **Claim:** "Session records" is entirely undecided pending the target-architecture ADR's classification.
- **Evidence gap:** Verified empirically: `src/llm_orc/agentic/session_action_record.py` opens with the docstring "Session Action Record (Cycle 7 loop-back #5 WP-LB-K, ADR-037) — L1," directly self-attributing to **ADR-037**, which is part of the confidently-superseded 033–043 chain (carry-forward table row `037`: "two-call termination composition... declarative home: loop `until:`" — already has a settled declarative home, not an open question). Separately, `src/llm_orc/agentic/session_registry.py` ("Session Registry — identifies and continues multi-request Sessions") matches **ADR-013** (Session Registry Initializer-then-Resume Schema), which genuinely is in ADR-045's deferred list. So "session records" as a plain-English label spans at least two ADRs in *different* buckets: ADR-037 (already confidently classified, with an already-settled declarative home) and ADR-013 (genuinely deferred). Labeling the whole category "deferred" tells a future reader (plausibly the target-architecture ADR's drafter) that session-record disposition is an open question, when part of it — the Session Action Record mechanism — is not; it is already retired with a specific declarative home decided two rounds ago. This is the same category-label-spans-multiple-buckets error R3 found and fixed for "calibration gates," reappearing in the same sentence, in the same revision, for a sibling category the revision didn't apply the same scrutiny to.
- **Recommendation:** Split "session records" the same way "calibration gates" was implicitly handled: name ADR-013 (Session Registry) as the deferred item, and note that ADR-037's Session Action Record component is already confidently classified via the 033–043 chain and excluded from the "other contents, pending classification" bucket.

### P2 — Should Fix (new this round)

**P2-NEW-4.** ADR-044's revised sentence — "Other contents (calibration gates, session records, conversation compaction) **are** deterministic-work or contract implementations... whose per-ADR disposition — AS-11-forbidden mechanism **vs.** surviving contract/direction... — is classified by the target-architecture ADR" — contains a small internal tension: the descriptive clause asserts these three categories *are* deterministic-work/contract implementations (prejudging them toward the "survives" side), in the same breath as saying whether they're AS-11-forbidden or surviving is still an open question for the target-architecture ADR to decide. ADR-045's own text is more careful here (it never prejudges 007/014/016's or 012's eventual classification). Recommend softening ADR-044's clause to "may be deterministic-work or contract implementations" or "physically resemble deterministic-work," matching ADR-045's neutral framing.

### P3 — Consider (new this round)

- ADR-044's AS-6/AS-11 corollary (§Decision) still doesn't cross-reference ADR-045's mechanism test/manifest, even though the Context section two paragraphs above it now does. A within-document inconsistency in cross-referencing thoroughness — minor, carried from R2's observation, still open, not worsened.
- "Orchestrator tool dispatch" in ADR-044's confident-mechanisms list is ambiguous between ADR-021 and ADR-027 (both plausible referents, both confident, so no risk of misclassification) — a footnote pinning the specific ADR(s) would remove the ambiguity even though it currently causes no harm.

### R4 Convergence-Saturation Signal (ADR-094)

**Convergence-Saturation Signal:** NOT TRIGGERED
- Round number: R4
- P1 count this round: 1 (P1-NEW-4 — the "session records" category mismatch; all of R3's findings are resolved and not recounted)
- P2 count this round (new, non-carry-over): 1 (P2-NEW-4 — the prejudging self-tension in the "other contents" descriptive clause)
- New framings or claim-scope expansions: none — this round is a narrow verification pass on the Context reconciliation, not a reframing
- Recommendation: **CONTINUE to next round.** P1 count (1) is nonzero, so the signal cannot trigger this round despite three of the last four rounds' fixes verifying clean on first attempt. The finding is narrow and mechanical (apply the same calibration-gates-style split to "session records": pull ADR-037 out into the confident bucket, leave ADR-013 deferred) — DECIDE should not need R5 to be an open-ended audit if this single line is corrected with the same precision the "calibration gates" fix received. R5 should re-verify directly rather than assume, per this audit's standing practice, but the trend (6 → 2 → 1 → 1 P1s, each round's finding narrower and more mechanical than the last) suggests R5 is a strong candidate for TRIGGERED if this is fixed cleanly and no sibling category was missed a third time.

*This is a standard-sequence re-audit; the verdict line above is required and included.*

---

## R5 Re-Audit (2026-07-01) — Exhaustive Bucket-Spanning Sweep

**Audited document (current state):** `adr-044-declarative-ensemble-native-serving.md` (§Context revised), `adr-045-clean-slate-collapse-imperative-serving-layer.md` (unchanged since R4)
**Verification method:** per the coordinator's explicit instruction to check *every* remaining category label rather than only the item flagged last round, this round ran the docstring-attribution check (the technique that caught P1-NEW-4 in R4) against **all 31 non-`__init__` files in `src/llm_orc/agentic/`**, not just the ones already under suspicion, and cross-checked the full set against ADR-045's three buckets (confident / deferred / survives).

### R4 Finding Resolution Verification

| R4 finding | Resolved? | Evidence |
|---|---|---|
| P1-NEW-4 ("session records" bucket-spanning) | **Yes.** | ADR-044 Context now reads "the session registry (ADR-013)" (not the ambiguous "session records"), with an explicit parenthetical: "(The Session Action Record, ADR-037, is *not* in this deferred set — it is part of the confidently-superseded 033–043 chain, with a settled declarative home per ADR-045's carry-forward table.)" Verified `session_action_record.py`'s docstring self-attributes to ADR-037 and `session_registry.py`'s to ADR-013 — the fix is accurate, not just relabeled. |
| P2 (prejudging phrasing) | **Yes.** | "are deterministic-work or contract implementations" is now "are not necessarily imperative-orchestration mechanisms" — properly neutral, no longer prejudging the pending classification. |

### The Exhaustive Sweep Finds Two More Instances of the Same Defect Class

Running the same check across all 31 files (not just the ones a prior round had already flagged) surfaces findings the narrower, reactive verification in R2–R4 did not reach, because each of those rounds checked only the specific label the previous round's finding named.

**P1-NEW-5. "Orchestrator tool dispatch" — one of only three examples ADR-044 cites as confidently, directly AS-11-forbidden — most naturally matches a file whose own docstring attributes it to a different ADR (ADR-003, a closed-tool-surface *contract*) than the one the label is presumably meant to invoke (ADR-021).**

- **Location:** ADR-044 §Context: "Its imperative-orchestration mechanisms — the Loop Driver, the tier router, **orchestrator tool dispatch** — are the AS-11-forbidden shape directly." This exact phrase has been present, unchanged, since R1 — no prior round verified it against the source tree.
- **Claim:** "Orchestrator tool dispatch" is confidently, directly the AS-11-forbidden imperative shape, on the same footing as Loop Driver (→ADR-033, verified) and tier router (→ADR-015, verified).
- **Evidence gap:** The literal file `src/llm_orc/agentic/orchestrator_tool_dispatch.py` opens: `"""Orchestrator Tool Dispatch — closed five-tool surface (ADR-003)."""` — **ADR-003** ("Fixed Orchestrator Tool Surface") is a closed-interface *contract* ("the orchestrator agent exposes exactly these tools, and no others... adding a new orchestrator tool requires a new ADR"), not obviously a control-flow decision mechanism — it reads like exactly the kind of "contract/direction" the mechanism test says survives, not the kind that's superseded. The file additionally wires together mostly deferred content (ADR-007 calibration gate, ADR-016 signal channel, ADR-017 tool-call guard, ADR-023 observability) and already-settled survivors (ADR-004/AS-7, ADR-024, ADR-025), with only incidental confident-bucket content (an ADR-015 tier-router call site, ADR-035/041 references). **ADR-003 itself appears in none of ADR-045's three buckets (confident, deferred, or survives) — it is not classified anywhere.** The mechanism ADR-044's label most plausibly *intends* — ADR-021's `OrchestratorRuntime` per-capability ReAct dispatch, which ADR-045 correctly and confidently classifies by that exact name — lives in a **differently-named file**, `orchestrator_runtime.py` ("the ReAct loop behind the serving layer... dispatch each through Orchestrator Tool Dispatch") — which is itself a mixed file (ADR-021's confident dispatch decision plus ADR-001's deferred ReAct-loop mechanics in the same module). **Correction to this audit's own R4 P3:** R4 dismissed this exact ambiguity as "no risk of misclassification" because both candidate referents (ADR-021, ADR-027) were assumed confident; this round's direct file check shows the assumption was wrong — the file bearing the label's literal name is dominated by an unclassified contract ADR and deferred content, not by ADR-021.
- **Recommendation:** Either rename ADR-044's label to unambiguously reference `OrchestratorRuntime`/ADR-021 (avoiding the collision with the `orchestrator_tool_dispatch.py` filename), or, if "orchestrator tool dispatch" is meant to cover both files' content, split it the same way "calibration gates"/"session records" were split — naming the confident slice (ADR-021's dispatch decision) separately from the unclassified/deferred slice (ADR-003's contract plus the ADR-007/016/017/023 wiring).

**P1-NEW-6. Two additional ADRs whose implementation lives in the removed layer — ADR-003 and ADR-011 — are absent from all three of ADR-045's classification buckets, and critically, absent from the *enumerated* ten-item deferred list the target-architecture ADR is instructed to audit.**

- **Location:** ADR-045 §Decision item 1's three buckets (confident: 033–043, 015, 021, 027; deferred: 001, 007/014, 008, 009/010, 012, 013, 016, 017, 018, 023; survives: 004, 005, 006, 019, 024, 025, 026).
- **Claim (implicit):** The manifest, having already been corrected once (R2/R3) to enumerate every `agentic/`-implemented ADR the confident cases don't cover, is now complete.
- **Evidence gap:** Verified via a full docstring-attribution sweep of all 31 files: `orchestrator_tool_dispatch.py` and `orchestrator_chunk.py` both self-attribute to **ADR-003**; `orchestrator_config.py` self-attributes to **ADR-011** ("Orchestrator LLM is a Model Profile"). Neither ADR-003 nor ADR-011 appears anywhere in ADR-045's text. This is the same shape of gap R2 found and R3 fixed for ADR-008/009/010 — recurring for two more ADRs a full sweep would have caught at the same time, had one been run then. The consequence is more concrete than a generic "not yet classified" gap: because the deferred bucket is now a **named, closed, ten-item list** that the target-architecture ADR is instructed to audit ("performs the full per-ADR mechanism audit against AS-11" — scoped to the ten named items), ADR-003 and ADR-011 are not merely deprioritized, they are **not on the list that would cause anyone to look at them**, risking a silent skip rather than a deferred-but-tracked classification.
- **Recommendation:** Add ADR-003 and ADR-011 to the deferred list (both read as plausible "survives" contracts on their face — a fixed tool-surface interface and a model-profile configuration convention, respectively — but per this audit's standing practice of not pre-judging peer classifications from outside the target-architecture ADR's own audit, defer rather than assert).

### P3 — Consider (new this round, lower-confidence coverage gaps from the same sweep)

- `inference_wait_heartbeat.py` and `composition_validator.py` show no explicit `ADR-NNN` tag in their opening docstring lines (unlike every other file in the directory). Contextual naming suggests `inference_wait_heartbeat.py` is ADR-023-adjacent (same "Cycle 6 WP-B" work package as `operator_terminal_event_sink.py`) and `composition_validator.py` is ADR-006/AS-2-adjacent (composition-palette validation), but neither is confirmed by direct citation the way every other file was. Worth a line in the eventual target-architecture ADR's audit, though neither is referenced by name in ADR-044 or ADR-045 today, so the exposure is lower than ADR-003/011's.

### R5 Convergence-Saturation Signal (ADR-094)

**Convergence-Saturation Signal:** NOT TRIGGERED
- Round number: R5
- P1 count this round: 2 (P1-NEW-5 — "orchestrator tool dispatch" mislabeling; P1-NEW-6 — ADR-003/ADR-011 omitted from the enumerated deferred list)
- P2 count this round (new, non-carry-over): 0
- New framings or claim-scope expansions: none
- Recommendation: **CONTINUE to next round.** P1 count (2) is nonzero.

### Honest Convergence Assessment (per the coordinator's request)

**The narrowing trend (6 → 2 → 1 → 1) did not hold at R5, and that is a meaningful signal in itself, not noise.** R2 through R4 each verified a *specific, previously-named* finding and additionally re-checked only the labels adjacent to it ("calibration gates" and "session records," both flagged because they sat next to an already-caught error). That reactive scope is why "orchestrator tool dispatch" survived four rounds unverified: nothing had named it as suspect yet, so nothing checked it against the source tree, even though the same technique that caught R3's and R4's findings would have caught it immediately if applied. This round did what the coordinator asked — swept every remaining label rather than only the one flagged last round — and found two more instances of the identical defect class, one of them on a higher-stakes claim (one of only three "confidently forbidden" examples) than any single-category label checked so far.

**Judgment: the two new findings are not non-material label/prose precision.** P1-NEW-5 concerns whether one of ADR-044's three headline illustrative examples of "the AS-11-forbidden shape directly" is even correctly identified — that is a scope-accuracy defect in the document's core argument, not a wording nit. P1-NEW-6 concerns whether the enumerated audit list ADR-045 hands off to the target-architecture ADR is actually complete — an incomplete enumeration risks a silent, permanent gap (an ADR nobody is instructed to look at) rather than a merely deferred one, which is a real, not cosmetic, downstream risk.

**However, the fix shape is now well-understood and mechanical**, following the exact pattern already used successfully three times (calibration gates, session records, ADR-018's dual-parentage note): rename/split the "orchestrator tool dispatch" label, and add ADR-003/ADR-011 to the deferred enumeration. Given this round's method (an exhaustive sweep of all 31 files, not a sample) is by construction more complete than R2–R4's reactive checks, **a clean fix plus one more exhaustive sweep of the same shape in R6 has a real chance of reaching TRIGGERED** — but I recommend against assuming that in advance. Advancing DECIDE before R6 confirms the fix would repeat exactly the pattern that let "orchestrator tool dispatch" sit unverified for four rounds: trusting that a targeted patch closes the finding without re-running the check that would confirm it.

*This is a standard-sequence re-audit; the verdict line above is required and included.*

---

## R6 Re-Audit (2026-07-01) — Verifying the Structural Fix (Source-Derived Complete Inventory)

**Audited document (current state):** `adr-044-declarative-ensemble-native-serving.md` (§Context rewritten to stop hand-picking examples), `adr-045-clean-slate-collapse-imperative-serving-layer.md` (§Decision item 1 rewritten around a definitionally-complete deferred set; new §Appendix inventory table)
**Verification method:** re-ran the same full 31-file docstring-attribution sweep from R5 independently (before re-reading the new Appendix), then diffed my independently-derived attribution against the Appendix table row by row; separately checked the `**Status:**` header of every ADR named in the "expected to survive" and "no implementation, survives cleanly" sublists for any that might already be Superseded/misfiled; separately grepped for ADR-040/041/002/020/022/028–032 to confirm nothing outside the checked set was missed.

### R5 Finding Resolution Verification

| R5 finding | Resolved? | Evidence |
|---|---|---|
| P1-NEW-5 ("orchestrator tool dispatch" mislabeling) | **Yes, via the split approach recommended.** | ADR-045's Appendix explicitly dual-lists `orchestrator_tool_dispatch.py`: the ADR-021 portion is confident-superseded, the ADR-003 closed-tool-surface contract portion is deferred, with a closing footnote naming this split deliberately. ADR-044's Context no longer uses the bare label at all — it now cites the three confident mechanisms by ADR number ("the ADR-033–043 loop-driver chain, the ADR-015 tier router, and the ADR-021 per-capability dispatch") and explicitly uses the ADR-003 case as the cautionary example of why file-presence isn't proof of forbidden-mechanism status. |
| P1-NEW-6 (ADR-003/ADR-011 omitted from the enumerated deferred list) | **Yes.** | Both now explicitly named in the deferred set ("ADR-001, 003, 004, 005, 006, 007, 008, 009/010, 011, 012, 013, 014, 016, 017, 018, 023, 024, 025") and in the Appendix (`orchestrator_tool_dispatch.py`/`orchestrator_chunk.py` → 003; `orchestrator_config.py` → 011). |

### Exhaustive Verification of the Appendix Against the Source Tree

I independently re-ran the full attribution sweep across all 31 non-`__init__` files in `src/llm_orc/agentic/` (the same command class used in R5) before re-reading the Appendix, to avoid anchoring on its claims. The independently-derived set of 31 file→ADR attributions **matches the Appendix table exactly** — every file is present, every primary-ADR attribution is correct, and every confident/deferred disposition is right, including the two inferred cases (`budget_controller.py`→005, `composition_validator.py`→006, neither carrying an explicit `ADR-NNN` docstring tag but both consistent with their described function) and the one no-ADR case (`inference_wait_heartbeat.py`, correctly marked "no ADR" rather than forced into a false attribution).

**Additional checks specifically targeting task (2)'s "is it genuinely complete" question:**
- Grepped for ADR-040/041 (deterministic completeness gate; destination-validity gate) — both are part of the confident 033–043 chain with their own carry-forward table rows and Superseded headers (verified in earlier rounds), but neither has a *standalone* module; their mechanisms are embedded inside already-confident files (`loop_driver.py` for 040, `artifact_bridge.py`/`client_tool_action_terminal.py` for 041). This is not a gap — the appendix's footnote already discloses "some modules touch several ADRs... the primary docstring ADR is used for disposition" — but 040/041 aren't individually named in that footnote's examples the way `orchestrator_tool_dispatch.py`'s split is. Minor, cosmetic (P3 below), not a disposition error: both ADRs are correctly confident regardless.
- Grepped for ADR-002, 020, 022, 028–032 (four-layer architecture; tool-use-ensemble-shape; routing-surface-behavior; the dormant routing-planner/synthesizer subtree) across all 31 files — no hits. These ADRs' code (where it exists) does not live in `src/llm_orc/agentic/` (ADR-027's own header already recorded that `dispatch_pipeline.py`/`ensemble_backed_roles.py` were removed via ADR-043, and that code was never in this directory). Their absence from the inventory is correct scope, not an omission.

**Conclusion on task (2): the deferred set (plus its confident and survives-cleanly siblings) is now genuinely complete relative to the actual source tree.** An independently-reproduced sweep, done blind to the Appendix's specific claims, converges on the same 31-file, same-attribution result.

### Task 3: Does Anything in "Expected to Survive" Turn Out to Be Misfiled?

Checked the `**Status:**` header of every ADR in the "expected to survive as contracts/invariants" sublist (004/AS-7, 005/AS-3, 006/AS-2, 024, 025, 003, 011, 008, 009/010) and the "no implementation, survives cleanly" sublist (019, 026). None carry a `Superseded by` header — none are misfiled into "survives" when they should be superseded. Two adjacent, pre-existing observations surfaced (not misfiling, but worth naming):

- **ADR-025's own Status field still reads "Proposed,"** not "Accepted," despite being treated throughout the corpus (including this document pair) as a settled, foundational architectural commitment. Same pattern as ADR-012's stale status found in R2. Pre-existing, not caused by this document pair, but adjacent enough to flag for the conformance audit.
- **ADR-026's own Status field also reads "Proposed."** Same pattern — AS-10 is treated as a codified, settled invariant throughout the domain model, but its originating ADR document's header was never updated.
- **ADR-024's header ("Updated by ADR-035 on 2026-06-03") now points to a superseded ADR.** ADR-035 is confidently superseded via the 033–043 chain. This isn't a misfiling of ADR-024 itself (its own decision stands; the update's *substance* — the client-tool-deliverable-path requirement — is separately carried forward via ADR-045's carry-forward table row for 035), but the header text itself doesn't note that its cross-reference target is now retired, which could read as a dangling pointer to a future reader tracing ADR-024's amendment history.

None of these rise to a misfiling of the classification decision (task 3's core question) — they are metadata/cross-reference hygiene notes, not scope or logic errors in ADR-044/045's argument.

### Task 4: Residual Inconsistency Between ADR-044's Context and ADR-045's Inventory

**None found.** ADR-044's Context no longer makes any specific claim beyond the three confident mechanisms (033–043 chain, 015, 021) — verified all three against both the source tree and ADR-045's confident bucket, and all three check out. For everything else, ADR-044 now explicitly defers to "the complete source-derived inventory in ADR-045" rather than repeating any of ADR-045's illustrative content, which structurally eliminates the drift risk that caused P1-NEW-3 (R3) and P1-NEW-4 (R4) — there is no longer a second copy of the classification that can fall out of sync with the first.

### P3 — Consider (new this round, all pre-existing or cosmetic)

- ADR-040/041/035/038/042's coverage isn't individually named in the Appendix's "modules touch several ADRs" footnote (though their disposition is correct and already independently verified via the carry-forward table + supersession headers from earlier rounds).
- ADR-025 and ADR-026's `Status: Proposed` fields are stale relative to how both are treated corpus-wide — flag for the conformance audit, not this document pair.
- ADR-024's "Updated by ADR-035" cross-reference now points to a superseded ADR without noting where the update's substance now lives (ADR-045's carry-forward table).

### Any Remaining P1: None Found

A full, independent re-derivation of the module inventory, cross-checked against every named sublist in both documents, surfaces no scope, omission, or mislabeling defect this round.

### R6 Convergence-Saturation Signal (ADR-094)

**Convergence-Saturation Signal:** TRIGGERED
- Round number: R6
- P1 count this round: 0
- P2 count this round (new, non-carry-over): 0
- New framings or claim-scope expansions: none — the structural fix (source-derived complete inventory replacing hand-picked illustrative lists) is a methodology change to how the *same* scope question is tracked, not a new framing or claim-scope expansion
- Recommendation: **STOP at this round.** All three trigger conditions hold: zero P1s, zero new P2s, no new framings. DECIDE can advance.

### Has the Structural Fix Closed the Class of Error?

**Yes.** R1 through R5 each found a variant of the same underlying defect: a hand-picked, illustrative, or reactively-patched list (proximity-based supersession in R1; the mechanism-test manifest in R2/R3; ADR-044's illustrative Context examples in R3–R5) that omitted a member or mislabeled one, because each list was built by someone naming examples from memory/judgment rather than deriving them from the actual source tree. Every fix from R1 through R5 patched the *specific instance* found, and every round after R1 found a *new instance of the same class* elsewhere, because patching an instance doesn't fix the generative mechanism (hand-picking) that produces the class.

R6's fix is different in kind, not just in scope: it replaces the generative mechanism itself. The deferred set is no longer asserted by a drafter enumerating what they remember to be true; it is derived by grepping every file in the actual directory and subtracting the independently-verified confident set. This is why an independent re-derivation (done blind to the Appendix's specific claims, per this round's methodology) converges on the same answer — the artifact and the ground truth are now the same computation, not two independently-maintained lists that can drift apart. The residual findings this round (Status-field staleness on ADR-025/026, a stale cross-reference on ADR-024) are pre-existing corpus-hygiene issues unrelated to the scope/omission defect class this audit tracked across six rounds — they are exactly the kind of "non-material label/prose precision" the coordinator asked me to distinguish from logic or scope defects, and they belong to the conformance audit, not this argument audit.

*This is a standard-sequence re-audit; the verdict line above is required and included. Per ADR-094, this TRIGGERED verdict recommends STOP — no further rounds needed on the scope/omission question this audit has tracked since R1.*

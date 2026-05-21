# Argument Audit Report

**Audited document:** `docs/agentic-serving/essays/essay-outline-006-cross-compatibility-routing-surface.md`
**Source material:**
- `docs/agentic-serving/essays/research-logs/research-log.md`
- `docs/agentic-serving/essays/research-logs/cycle-7-spike-lambda-tool-choice.md`
- `docs/agentic-serving/essays/research-logs/cycle-6-spike-gamma-routing-characterization.md`
- `docs/agentic-serving/essays/research-logs/cycle-6-spike-delta-framework-chaining.md`
- `docs/agentic-serving/essays/reflections/field-notes.md` (Cycle 6 PLAY section, notes 1-25)
- `docs/agentic-serving/housekeeping/audits/research-design-review-cycle-7.md`
- `docs/agentic-serving/housekeeping/audits/research-design-review-cycle-7-round-2.md`
- `docs/agentic-serving/housekeeping/audits/citation-audit-cycle-7.md`
- `docs/agentic-serving/domain-model.md`
- ADRs: adr-022, adr-023, adr-024, adr-025, adr-014, adr-018
**Genre:** Essay-Outline (ADR-092)
**Date:** 2026-05-21

---

## Section 1: Argument Audit

### Summary

- **Genre:** Essay-Outline
- **Argument chains mapped:** 6 (C1–C6)
- **Issues found:** 5 (1 P1, 3 P2, 1 P3)
- **Pyramid coverage map:** included
- **Expansion-fidelity findings:** P1: 0, P2: 2, P3: 1

---

### Pyramid Coverage Map

The Essay-Outline contains all four required sections: Abstract (with CONCLUSIONS C1–C6), Argument-Graph (C1–C6 with full warrant/evidence trees), Citation-Embedded Outline (Sections 1–8), and References (17 entries). All four sections are present; no structural absence to report.

| Abstract Conclusion | Argument-Graph Nodes | Body Sections | References Cited |
|---|---|---|---|
| C1. NL-to-ensemble routing fraction approximately zero under tool-rich clients | C1, W1.1, W1.2, E1.1.1, E1.1.2, E1.1.3, E1.2.1, E1.2.2, E1.2.3 | §2 (C1) | [cycle-6-spike-gamma], [cycle-6-field-notes-play] |
| C2. tool_choice implemented at framework level but not honored by production model | C2, W2.1, W2.2, W2.3, E2.1.1, E2.1.2, E2.2.1, E2.2.2, E2.3.1 | §3 (C2) | [cycle-7-spike-lambda], [openai-fc-guide], [openai-api-ref] |
| C3. Model-portability gap motivates server-side dispatch mechanism | C3, W3.1, W3.2, W3.3, E3.1.1, E3.2.1, E3.2.2, E3.2.3, E3.3.1, E3.3.2, E3.3.3 | §4 (C3) | [agentic-serving-cycle-status], [research-design-review-cycle-7], [research-log], [cycle-7-spike-lambda], [cycle-6-field-notes-play] |
| C4. Framework-driven composition continuation required due to orchestrator file-read pattern | C4, W4.1, W4.2, W4.3, E4.1.1, E4.1.2, E4.1.3, E4.2.1, E4.2.2, E4.3.1, E4.3.2, E4.3.3 | §5 (C4) | [cycle-7-spike-lambda], [cycle-6-field-notes-play], [adr-025], [domain-model], [cycle-6-spike-delta] |
| C5. I/O contract enforcement targets form-drift at synthesizer layer; ensemble-authoring mechanisms favored | C5, W5.1, W5.2, W5.3, E5.1.1, E5.1.2, E5.1.3, E5.2.1, E5.2.2, E5.2.3, E5.3.1, E5.3.2, E5.3.3 | §6 (C5) | [cycle-6-field-notes-play], [research-design-review-cycle-7], [research-log], [cycle-6-spike-gamma], [cycle-6-spike-delta], [adr-014], [adr-018] |
| C6. Fallback is current direct-completion behavior; design work is documentation | C6, W6.1, W6.2, W6.3, E6.1.1, E6.2.1, E6.2.2, E6.3.1, E6.3.2 | §7 (C6) | [research-log] |

**META-anchored sections:** §1 (Methodology preamble); META-OBSERVATION paragraphs within §3 and §6; §8 (architectural synthesis + scope-of-claim).

No Argument-Graph nodes lack body content. No body sections lack anchors. All forward and reverse boundaries check clean at the structural level.

**Citation P1 issue (from prior citation audit, now verified as fixed):** The citation audit flagged E4.3.3 and the parallel Section 5 evidence bullet as citing "389,444 + 388,790 figures" for Spike δ when the Spike δ document records "354,751 + 354,000 + 388,790." Verification of the current Essay-Outline text confirms the figures now read "354,751 + 354,000 + 388,790 population figures" in E4.3.3 and the corresponding Section 5 bullet. The citation audit's P1 has been resolved. The claim-extractor note-15 P2 citation (E5.1.2) has also been reviewed below (see P2 findings).

---

### Expansion-Fidelity Findings

**P1 findings (pyramid violations):** None.

All six Boundary 1 correspondences (C1–C6 Abstract → Argument-Graph) hold. All Boundary 2 correspondences (every Argument-Graph node → body content) hold. All References citations resolve (Boundary 3 clean per the prior citation audit; all keys present). No Reverse Boundary 1 or Reverse Boundary 2 violations.

**P2 findings (weak expansion or META misclassification):**

- **Section 8 (C3+C4 composed; META scope) — partial META misclassification.** The section header labels Section 8 as both `(C3+C4 composed)` and `(META scope)`. The first substantive block is a SYNTHESIS paragraph that derives the hybrid-architecture recommendation (i'–iv') directly from C3 and C4. This is developmental content — it is the essay's capstone claim, inferring the cycle's recommended architecture from the two preceding claims. The body then shifts into genuine META (scope-of-claim, VALIDATION-SPIKE DECISION process observation). The dual label partially obscures this structure. The developmental synthesis (`(i')` through `(iv')` with their sub-claims) is not anchored to a graph node in the Argument-Graph; the Argument-Graph contains C3 and C4 independently but does not contain a composed `C3+C4` node. The body section's developmental bullets develop a claim that does not exist as a node in the pyramid. This is a P2 concern: the pyramid technically holds (C3 and C4 are each developed separately; the composed synthesis is labeled in the section header), but the expansion is structured so the reader cannot trace the hybrid-architecture claim directly from the Argument-Graph to the body.
  - **Recommendation:** Either add a C3+C4 composition node to the Argument-Graph (naming the hybrid architecture as the composed claim that follows from C3 and C4), or restructure Section 8 so the developmental SYNTHESIS content is explicitly surfaced as deriving from C3 and C4 without needing a separate graph node.

- **E5.1.2 / Section 6 evidence citing PLAY note 15 for form-drift across invocation paths.** The essay cites Cycle 6 PLAY note 15 as evidence that claim-extractor's form drift "persists when chained via orchestrator-LLM, establishing path-independence." The citation audit identified this as a weak citation: note 15's primary content is fabrication-while-critiquing-fabrication (the orchestrator generated stub code rather than narrating the ensemble's actual output), not clean form-drift evidence. The form-drift finding is stated in note 15 as a secondary observation. Spike δ's explicit statement "Form drift persists. Claim-extractor's output is still non-conformant to its default_task spec" is a more direct and unambiguous citation for path-independent form drift, and Spike δ is already cited at E5.1.3 for the framework-chained case. The note-15 citation weakens the evidence chain without adding distinctive support.
  - **Recommendation:** Retain Spike δ's "Form drift persists" as the primary citation for path-independent form drift; recast note 15 as supporting context for the fabrication-compounds-form-drift finding if retained at all, rather than as primary evidence for form drift itself.

**P3 findings (minor coverage gaps):**

- **W2.3 (three candidate diagnoses for the model-portability gap) lacks disambiguation evidence.** The warrant correctly names the three candidate diagnoses (Zen proxy stripping, MiniMax model non-conformance, framework tool-list construction interaction) and notes they are not disambiguated. The Argument-Graph records this limitation, and the body's Section 3 carries it with an explicit CONFIDENCE-LEVEL tag. However, E2.3.1 cites only the Spike λ-paid limitations section naming the three diagnoses — the evidence bullet points to a source that says "we don't know" rather than to any positive finding. This is structurally sound (honest evidence limitation) but could be more clearly framed in the body as a gap-acknowledgment evidence bullet rather than as an evidence point. The current framing may create the impression of support for the ambiguity claim when the source is simply describing the absence of evidence. No structural fix required; a label adjustment ("Evidence of gap: ...") would clarify the epistemic status.

---

### P1 — Must Fix

The citation audit's P1 finding (incorrect population figures citing 389,444 as originating from Spike δ) has been verified as resolved in the current Essay-Outline text. No new P1 argument-audit findings are raised.

The pyramid traversal found no Boundary 1, Boundary 2, Boundary 3, Reverse Boundary 1, or Reverse Boundary 2 violations. The Essay-Outline clears the P1 gate.

---

### P2 — Should Fix

**P2-1: Section 8 hybrid-architecture synthesis is developmental content without a dedicated Argument-Graph node.**

- **Location:** Section 8 first SYNTHESIS block (i'–iv' hybrid recommendation); Argument-Graph (no C3+C4 composed node)
- **Claim:** The section's opening SYNTHESIS derives the full hybrid architecture from C3+C4 as an integrated recommendation. The hybrid architecture is the essay's most consequential claim — it is the design recommendation the DECIDE phase will act on.
- **Evidence gap:** The Argument-Graph has independent nodes for C3 (server-side mechanism motivated) and C4 (framework-driven composition required) but no node that combines them into the hybrid-mechanism claim. A reader tracing from the Argument-Graph to the body cannot find the node the Section 8 synthesis is developing.
- **Recommendation:** Add a composed node to the Argument-Graph — e.g., `C3+C4 composition: the hybrid mechanism combines server-side tool_choice interception with framework-driven composition continuation; neither alone is sufficient` — or make the composition explicit in the Abstract as a seventh conclusion (the synthesis of C3 and C4 is arguably the cycle's primary deliverable and could stand as a CONCLUSIONS entry). The current structure works, but the hybrid claim deserves explicit pyramid placement.

**P2-2: Note 15 citation for form-drift path-independence is weaker than Spike δ for the same claim.**

- **Location:** E5.1.2; Section 6 evidence bullet for "chained via orchestrator" form drift
- **Claim:** The essay asserts claim-extractor's form drift has been "observed in two configurations: direct invoke (note 5) and chained via orchestrator (note 15)" to establish path-independence.
- **Evidence gap:** Note 15's primary content is the fabrication-while-critiquing pattern (orchestrator generated stub code, not ensemble output). The form-drift observation is secondary and intermixed with the confabulation finding. Spike δ provides a cleaner, more direct citation: it explicitly records form drift persisting under framework-driven chaining in the absence of the orchestrator-LLM, which more cleanly establishes path-independence.
- **Recommendation:** Reframe E5.1.2 and the corresponding Section 6 bullet to read: "Cycle 6 PLAY note 15: claim-extractor output when chained via orchestrator-LLM also deviated from spec (same pattern, different trigger — note 15's primary finding is orchestrator fabrication compounding synthesizer form drift, corroborating path-independence)." Alternatively, lead with Spike δ's explicit "Form drift persists" for the path-independence claim and use note 15 as a secondary corroboration rather than as primary evidence.

**P2-3: Working inference citations at E3.1.1, E4.2.1, and E5.3.3 carry architectural weight without source support.**

- **Location:** E3.1.1 (interception model preserves client-facing contract); E4.2.1 (production clients declare client tools for their own filesystem scope); E5.3.3 (framework-driven composition continuation eliminates orchestrator-narration step)
- **Claim:** Three evidence bullets are labeled "(working inference from ...)" and used as support for warrants that are load-bearing in the argument structure.
- **Evidence gap:** E3.1.1 is inference about a mechanism that has not been implemented or tested — it supports W3.1 (server-side interception is consistent with the OpenAI-API contract) but no empirical work has validated that an intercepted dispatch preserves the client-facing response shape correctly. E4.2.1 infers production client filesystem scope from "OpenAI-family agentic-coding-tool architectures" without citing a specific source (the cited source is noted as "working inference from ... architectures; cycle-6-field-notes-play §note 23 corroborates"). E5.3.3 is explicitly circular — it cites "working inference from C4" to support a warrant within the C5 chain, rather than an independent empirical source. The essay correctly labels these as inferences, but three working inferences in load-bearing warrant positions accumulate epistemic risk.
- **Recommendation:** These do not rise to P1 because the essay honestly labels them as inferences. They are noted here so the DECIDE phase treats W3.1's server-side interception claim as architectural hypothesis (not empirically validated) and does not architect against E4.2.1 or E5.3.3 as settled findings.

---

### P3 — Consider

**P3-1: W2.3's evidence bullet (E2.3.1) labels absence-of-disambiguation as evidence rather than as a gap acknowledgment.**

- **Location:** E2.3.1
- **Issue:** The bullet cites the spike's limitations section naming the three diagnoses as "evidence" that the failure originates outside the agentic-serving framework. But the source says only that the three diagnoses exist and are not disambiguated — it does not provide positive evidence that the failure is external to the framework. The inference "it must originate outside the agentic-serving framework's own implementation surface" from the three diagnoses is valid reasoning but is not itself evidenced by the citation.
- **Recommendation:** Relabel E2.3.1 from a standard evidence bullet to a gap-acknowledgment bullet: "Spike λ-paid §limitations documents three candidate diagnoses, none yet disambiguated — each locates the failure outside the framework's own dispatch surface, but evidence for which candidate applies is absent." This would better represent the epistemic status.

---

## Section 2: Framing Audit

### Question 1: What alternative framings did the evidence supported but the Essay-Outline did not choose?

**Alternative framing A: "The chat-completions surface is the wrong abstraction for this use case — expose multiple surfaces and let clients choose."**

The constraint-removal step (research-log Step 1.2) surfaced the direct-`invoke` alternative path explicitly. The reviewer round-1 reformulation (research-design-review-cycle-7 §Q1) named "caller-supplied ensemble identity" as the cleanest dispatch path. Spike γ Cell A-explicit established that explicit naming reliably triggers dispatch. Spike λ Cell λ.3 confirmed that even `tool_choice` forcing `invoke_ensemble` requires a capable orchestrator model.

Taken together, the evidence supports the framing: the chat-completions surface adds orchestrator-LLM complexity that the direct-invoke path avoids entirely, and the orchestrator-LLM is consistently the failure surface across the empirical record (confabulation, form drift, substrate-path hallucination, tool_choice non-compliance). The Essay-Outline's framing treats chat-completions as the cross-compatibility surface to improve. The alternative framing would treat it as the surface to limit: document that chat-completions serves tool_choice-aware explicit-dispatch clients; direct all NL-routing and composition use cases to direct-invoke; do not attempt NL-to-ensemble routing at all. Under this framing, C3's hybrid mechanism becomes an over-engineering response to a problem better solved by routing clients to the right surface.

*Belief required for this framing to be right:* The developer population that would use NL framing to reach ensemble dispatch is small relative to the population that would use explicit-naming or direct-invoke. The research log Step 2 Q0 answer supports this — the NL-routing fraction is approximately zero — but the Essay-Outline routes this finding toward "server-side interception handles NL callers" rather than "NL callers should use a different surface."

**Alternative framing B: "The orchestrator-LLM should be removed from the chat-completions surface entirely — make chat-completions a pure dispatch surface."**

Spike δ showed that framework-driven chaining succeeds when the orchestrator-LLM is removed from the chain step. PLAY note 22 showed the orchestrator-LLM confabulates multi-dispatch summaries while narrating correctly-dispatched ensembles. The research log's synthesis noted that the orchestrator-LLM is "the consistent failure surface." The Essay-Outline extends Spike δ's framework-driven pattern to composition continuation (C4) — but preserves the orchestrator-LLM for the initial routing decision and for single-dispatch synthesis.

The stronger framing from the same evidence: remove the orchestrator-LLM from the chat-completions surface entirely, replacing it with a deterministic request handler that routes to an ensemble and surfaces the result directly. The orchestrator-LLM becomes the implementation mechanism of a `general-completion` ensemble only. This is the framing Spike δ's note "Architectural implication: framework-driven plan→dispatch→synthesize pipeline; orchestrator-LLM removed from routing-and-chaining decision loop" and ADR-027 candidate sketched. The Essay-Outline does not pursue this framing, stopping at composition continuation while preserving the orchestrator-LLM's routing role.

*Belief required for this framing to be right:* That the orchestrator-LLM's failure modes are structural (inherent to using an LLM for routing and composition) rather than addressable (addressable through prompt engineering, server-side mechanisms, or model selection). The cycle's evidence tilts toward structural failure, but the Essay-Outline does not make this determination explicit.

**Alternative framing C: "Do not commit to an architecture until model-portability is characterized across more orchestrator profiles."**

The empirical sample is n=2 model profiles (qwen3:14b, paid MiniMax M2.5) × n=1 client family (OpenCode) × n=1 probe prompt ("Write a Python function that reverses a string in place"). The model-portability gap (C2) is established for this specific combination. The reviewer round-2 §scope-of-claim in Section 8 names this limitation, but the Essay-Outline proceeds to recommend the hybrid mechanism (C3) as if the gap finding generalizes.

The alternative framing: defer architecture recommendation until Groq Llama-3, Cerebras, Anthropic Sonnet, and OpenAI gpt-4o-mini are characterized against the same `tool_choice` forcing dispatch test. Some of these models may honor `tool_choice` correctly (making the hybrid mechanism unnecessary for those profiles); others may exhibit different failure modes (making the hybrid mechanism wrong for those profiles). Under this framing, the cycle's correct deliverable is a characterization matrix rather than an architecture recommendation.

*Belief required for this framing to be right:* That model-portability across the deployment target population is heterogeneous enough that a single architecture recommendation is premature. The evidence supports the concern — the two tested models produced opposite behavior under the same payload.

---

### Question 2: What truths were available but not featured?

**Underrepresented finding A: The latency overhead of 192s wall-clock under forced dispatch (qwen3:14b) is a Q1 constraint, not just a measurement note.**

Spike λ Cell λ.3 measured 192s wall-clock (72.3s dispatch + ~120s orchestrator-LLM overhead). The research log's latency note and E3.3.3 correctly identify that the R2-1 latency bound applies to the routing mechanism overhead, not to the dispatch itself, and that qwen3:14b's 120s orchestrator-LLM overhead is a model/prompt-engineering concern. However, from the user-facing perspective, a client sending `tool_choice={"name":"invoke_ensemble"}` would wait 192s for a response. The Essay-Outline's scope-of-claim in Section 8 notes the architecture is conditional on representative model characterization but does not centralize the orchestrator-LLM overhead as a user-experience constraint that any production deployment must solve independently of the routing mechanism.

The PLAY field notes (note 1: 10.3s for direct completion; notes 10, 11 Cycle 5: 22m 36s for dispatched OpenCode session) establish that orchestrator-LLM overhead is a recurring user-experience problem. The Essay-Outline treats C3's hybrid mechanism as the architectural fix for routing reliability without addressing whether the orchestrator-LLM overhead — which persists under any mechanism that routes through the orchestrator-LLM — makes the hybrid mechanism user-facing viable.

**Underrepresented finding B: The substrate-path-as-deliverable pattern (F-paid-4) across both λ.4-paid and λ.5-paid is architectural evidence about the framework's design contract.**

Spike λ F-paid-4 documented that paid MiniMax M2.5, across both `tool_choice` configurations that produced dispatch, reasoned about substrate paths as deliverables — the orchestrator-LLM treated substrate routing as a chain surface to compose through. The research log calls this "a sophisticated composition pattern broken by the XML/JSON impedance mismatch." The Essay-Outline cites this as motivation for C4 (framework-driven composition continuation). However, F-paid-4's deeper implication — that the orchestrator-LLM is actively reasoning about the ADR-025 substrate routing infrastructure and treating it as a contract surface — is not surfaced as an architectural observation. If production models consistently attempt to chain through substrate paths, this suggests that surfacing substrate path references to orchestrators may need explicit framework-level handling (either hiding the substrate path from the orchestrator's context entirely, or providing a deterministic "here is the dispatch result in this turn" response that eliminates the chaining motivation). The Essay-Outline recommends composition continuation without addressing why the orchestrator reasons about substrate paths in the first place.

**Underrepresented finding C: PLAY note 22's challenged-claim persistence is not integrated into the hybrid mechanism's reliability profile.**

The Essay-Outline cites PLAY note 22 at E4.1.3 and E4.2.2 for the composition-failure evidence. It correctly characterizes the operational failure (orchestrator fabricated figures, attempted path-hallucinated file reads). However, note 22's most distinctive finding — challenged-claim persistence, where the orchestrator maintained an incorrect path assumption across four explicit corrections from the practitioner — is cited only in the domain model's vocabulary section and appears in the Essay-Outline only as "path-hallucination-as-composition pattern." Challenged-claim persistence is not form-drift or confabulation; it is a distinct reliability failure mode where the model's prior is robust to correction. If the hybrid mechanism surfaces dispatch results to the orchestrator (rather than bypassing it entirely), the orchestrator's reasoning about those results may exhibit the same challenged-claim persistence — it may narrate dispatch content incorrectly and maintain the incorrect narration under client pushback. The Essay-Outline's C4 claim that framework-driven composition continuation "removes the orchestrator-narration step" partially addresses this; but the mechanism's design does not fully account for the persistence failure mode, and the framing audit finds this integration gap worth surfacing at the gate.

---

### Question 3: What would change if the dominant framing were inverted?

The Essay-Outline's dominant framing is: the model-portability gap is a solvable problem that motivates a hybrid mechanism (server-side interception + framework-driven composition continuation), and this mechanism preserves the chat-completions surface as the cross-compatibility surface for production clients.

**Inverted framing:** The model-portability gap is a symptom that the chat-completions surface — mediated by an orchestrator-LLM — is the wrong abstraction for reliable ensemble dispatch. The gap cannot be reliably solved within the orchestrator-LLM paradigm; the correct architectural response is to reduce or eliminate the orchestrator-LLM's role in the dispatch path rather than to add a server-side mechanism that works around its failures.

Under the inverted framing:

- C3's hybrid mechanism becomes over-engineering. A server-side `tool_choice` interception mechanism that triggers dispatch deterministically without the orchestrator-LLM is functionally equivalent to the direct-invoke path with an HTTP wrapper. The complexity cost of building this "interception" is justified only if the chat-completions surface delivers value the direct-invoke path does not — and under the inverted framing, that value (orchestrator-LLM reasoning and synthesis) is the very failure surface the mechanism is designed to bypass.
- C4's framework-driven composition continuation becomes the dominant finding rather than one of two claims. If the framework should own composition, then the hybrid mechanism reduces to a thin HTTP translation layer, and the architectural work is to design the framework's dispatch pipeline, not to design around an LLM-mediated routing surface.
- C1 becomes less significant. If the NL-to-ensemble routing fraction is approximately zero and the design response is to bypass the orchestrator-LLM for routing anyway, then the C1 finding describes the current state of a surface the architecture is about to abandon, not a production constraint to accommodate.
- C2 becomes the primary motivating finding: not "the gap motivates a server-side mechanism" but "the gap reveals that the chat-completions + orchestrator-LLM paradigm is not model-portable and should be reconsidered structurally."

The inverted framing would require the Essay-Outline to address: why the orchestrator-LLM is worth preserving at all in the dispatch path (what value it adds that a framework-driven pipeline cannot); what population of callers genuinely needs the LLM-mediated chat-completions surface (vs. direct-invoke with structured responses); and what the migration path looks like for callers who currently rely on the orchestrator's NL synthesis.

---

### Framing Issues

**P1 — Consequential omissions:**

None at P1. No source-material finding would change the Essay-Outline's conclusions if incorporated. The core findings (C1–C6) hold against the evidence base; the inverted framing above and the alternative framings are legitimate but do not refute the cycle's claims.

**P2 — Underrepresented alternatives:**

**F2-1: The "remove orchestrator-LLM from the dispatch path" framing is supported by the evidence but not acknowledged.**

- **Source material:** Spike δ "Architectural implication" note; research log Spike λ-paid synthesis §"Q1 reformulates again" noting both (i') interception AND (ii') composition continuation as required; PLAY note 22 challenged-claim persistence; PLAY note 16 (Calibration Gate doesn't audit orchestrator narration); domain model note on orchestrator-LLM as "consistent failure surface."
- **Why excluded:** The Essay-Outline's framing commits to the hybrid mechanism (preserve chat-completions, add server-side interception) as the architecture recommendation. The stronger form of the same evidence (the orchestrator-LLM is the failure surface; remove it) is noted in Section 8's scope-of-claim but is not foregrounded as an alternative the DECIDE phase should evaluate.
- **Impact on argument:** C4's framework-driven composition continuation is the structural step toward removing the orchestrator-LLM from composition. The Essay-Outline stops short of asking whether C3's server-side mechanism should also bypass the orchestrator-LLM for routing entirely (not just for `tool_choice` interception, but for all dispatch paths). Acknowledging this framing at the gate would ensure the DECIDE phase explicitly evaluates whether the hybrid mechanism is the right architecture or whether a fully framework-driven dispatch pipeline (the ADR-027 candidate noted in Spike δ) is a cleaner answer.
- **Recommendation:** Add a scope note in Section 8 or in the framing audit's gate commentary acknowledging that the evidence also supports the stronger "remove orchestrator-LLM from dispatch path" framing and that the DECIDE phase should evaluate this alternative alongside the hybrid mechanism.

**F2-2: Empirical sample scope is narrow enough that architecture recommendation should be conditional.**

- **Source material:** Research-design-review-cycle-7-round-2 §Flag R2-1; Section 8's SCOPE QUALIFICATION in the Essay-Outline (correctly names the limitation); research log Q0 synthesis §Scope-of-claim caveats.
- **Why underrepresented:** The Essay-Outline carries a strong SCOPE QUALIFICATION in Section 8's META content. The qualification is correct and present. However, the architecture recommendation (i'–iv') in Section 8 is presented as the DECIDE phase's input, and DECIDE may proceed on the recommendation without treating the scope limitation as a gate. The distinction between "here is our architecture hypothesis for n=2 model profiles" and "here is our architecture recommendation" is not drawn.
- **Impact on argument:** The hybrid mechanism (C3) is well-supported for the qwen3:14b + paid MiniMax M2.5 axis. But whether `tool_choice` is unreliable across the broader model population — and whether the server-side interception adds value for models that do honor `tool_choice` — is uncharacterized. Acknowledging this gap explicitly in the conclusions (not just in the META scope-of-claim) would make the conditionality visible to the DECIDE phase.
- **Recommendation:** Consider qualifying C3 in the CONCLUSIONS list as conditional: "The model-portability gap motivates a server-side mechanism *for the tested model profiles*; characterization across additional production model profiles is required before committing to the mechanism's architecture."

**P3 — Minor framing choices:**

**F3-1: The Essay-Outline does not foreground the leaner alternative to the hybrid mechanism.**

- **Issue:** The hybrid mechanism is the design recommendation. A leaner alternative — "document that tool_choice works for qwen3:14b and similar models; production deployments must verify before relying on it; no server-side mechanism needed" — is implicitly rejected by the cycle's evidence (paid MiniMax M2.5 non-compliance is real) but is not named as a rejected alternative. Naming it would sharpen the hybrid mechanism's motivation by contrast.
- **Recommendation:** Section 4's or Section 8's SYNTHESIS could note: "The 'document and verify' approach (no server-side mechanism; each deployment validates model compliance) is a viable alternative for deployments with known-compliant orchestrator models, but does not serve the cross-compatibility goal for deployments where the production orchestrator model is variable or unknown."

---

*End of argument audit report.*

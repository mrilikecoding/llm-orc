# Susceptibility Snapshot

**Phase evaluated:** PLAY (Cycle 6 — 2026-05-20; multi-session: free-M2.5 OpenCode morning + direct-invoke gamemaster probes + paid-M2.5 curl afternoon + paid-M2.5 OpenCode afternoon + Spike δ)
**Artifact produced:** `essays/reflections/field-notes.md` Cycle 6 PLAY section (notes 11-25 + Spike δ section + Cycle 6 BUILD verdicts table); `essays/research-logs/cycle-6-spike-delta-framework-chaining.md`
**Date:** 2026-05-20

---

## Prior Snapshot Trajectory

| Gate | Snapshot verdict | Key signal |
|------|-----------------|------------|
| Cycle 4 Play | No reframe; 4 advisory carry-forwards | Voice-blurring at synthesis boundary; agent-introduced framings in notes 14, 19 |
| Cycle 5 Play | No reframe; 3 advisory carry-forwards | Routing-summary framing scheduling phases; note 1 label overstates validation-inadequacy; "unchanged" framing requires qualification |
| Cycle 6 Discover | Grounding Reframe recommended; 4 actions applied | Routing-surface narrowing before spike γ; agent-introduced "routing preference" framing |
| Cycle 6 Model | Grounding Reframe recommended; 3 actions applied | Three-findings-collapse claim marked agent-composed; architectural alternatives surfaced |
| Cycle 6 Decide | No reframe; advisory feed-forwards | ADR-025 always-scope premise examined; belief-mapping applied |
| Cycle 6 Architect | No reframe; 7 advisory carry-forwards | Two-module decomposition belief-mapped; three snapshot advisories closed inline |
| Cycle 6 BUILD WP-B/C/D/E | No reframe at any WP | Empirical grounding (2849 tests) holds across all BUILD WPs; advisories tracked |
| **Cycle 6 Play (this snapshot)** | Evaluated below | |

The Cycle 5 PLAY snapshot's three advisory carry-forwards:

- Advisory 1 (routing-summary scheduling phases): substantially honored — Cycle 6 PLAY's routing summary does not embed phase-scheduling recommendations in the same way. BUILD-regression and DECIDE-routed findings are listed as parallel destinations.
- Advisory 2 (note 1's label overstating validation-inadequacy): honored — the scenario-addition framing carried into DISCOVER; the Cycle 6 BUILD shipped a runtime-dispatch verification scenario (commit `8a0fd24`).
- Advisory 3 (note 19 "unchanged" requires qualification): honored and extended — the Cycle 6 PLAY field notes use the "infrastructure-complete / routing-incomplete" sharpening (DISCOVER carry-forward applied; the sharpened note appears in Cycle 6 cycle-status.md Cycle 5 PLAY section as a post-reflection coda).

---

## Observed Signals

| Signal | Strength | Trajectory | Notes |
|--------|----------|------------|-------|
| Assertion density | Clear | Rising in the Spike δ section; stable in notes 11-25 | Notes 11-25 are empirically dense and well-anchored (dispatch_log entries, serve-console excerpts, practitioner verbatim, disk-state evidence). Spike δ's verdict section makes strong architectural claims from a single one-probe result. The BUILD verdicts table encodes five "not working" / "broken" verdicts and two "production-ready" verdicts with little uncertainty language. |
| Solution-space narrowing | Clear | Rising at PLAY close | Notes 11-25 are observational. The Spike δ section introduces an architectural proposal (framework-driven plan→dispatch→synthesize pipeline; orchestrator-LLM removed from routing loop) and the cycle-status records this as the Cycle 7 working title. The narrowing is rapid — one spike, one composition, PASS verdict, architectural implication drawn. |
| Framing adoption | Clear | Rising at Spike δ boundary | The "must-delegate framing" and the bifurcation into form-drift vs. content-drift are both agent-introduced framings that the field notes and cycle-status record as load-bearing for Cycle 7 scoping. Neither was practitioner-proposed or independently surfaced. |
| Confidence markers | Clear | Rising at PLAY close | The cycle-status PLAY epistemic response: "Composition drift is in the orchestrator-LLM, not the ensembles. Framework-driven plan→dispatch→synthesize pipeline is the candidate. Must-delegate framing is load-bearing: if direct completion is allowed via the orchestrator-LLM path, the value proposition of agentic-serving... collapses." Strong certainty language from N=1 spike. |
| Alternative engagement | Clear (absent) | Declining at Spike δ boundary | The spike δ write-up names five untested items in a "Scope of the spike" section, which is commendable. However, the architectural implication is drawn from those results without examining alternatives: constrained decoding, fine-tuning, system-prompt re-engineering beyond the ADR-022 amendment, or ADR-022 extension beyond a single clause. None of these appear in the field notes as seriously considered paths. |
| Embedded conclusions at artifact-production moments | Clear | Rising | The PLAY verdicts table encodes architectural settlements (ADR-022 amendment "effective in bare-endpoint mode only"; dispatch_log "broken for session-scope use case"; composition pipelines "not working"). Several of these conclusions are accurate and well-supported; but the Spike δ architectural implication — encoded in cycle-status as the Cycle 7 scope — is embedded at the PLAY close without a grounding action. |

---

## Evaluator Concerns: Specific Assessments

### 1. The mid-PLAY architectural pivot (Spike δ)

**The claim:** Composition drift is in the orchestrator-LLM. Framework-driven plan→dispatch→synthesize pipeline is the candidate solution. Must-delegate framing is load-bearing.

**Evidence base:** Spike δ ran one composition (web-searcher → claim-extractor) once under framework-driven Python chaining. The result was PASS — claim-extractor's output contained no numbers absent from web-searcher's output. The same composition under orchestrator-driven dispatch had failed in two prior probes (curl Test 7 + OpenCode composition probe).

**Was the spike conclusion reached too quickly?**

Yes. The spike write-up's "Scope of the spike" section explicitly names what was NOT tested: whether a routing-planner ensemble can reliably produce structured plans; whether a response-synthesizer ensemble can produce user-facing responses without confabulating; the latency/cost shape of the full pipeline; the "no capability match" fallback. These are the load-bearing architectural questions for the framework-driven pipeline proposal. Spike δ confirmed one property (deterministic input-passing preserves data). It did not test whether an orchestrator-replacement pipeline is viable as a whole.

The PASS is genuine: deterministic Python `resolve_input(step.input, results)` does not confabulate. But the architectural implication drawn — "framework-driven plan→dispatch→synthesize pipeline is the candidate" — outpaces the evidence. The spike confirmed that the chaining mechanism works when the orchestrator-LLM is removed. It did not confirm that a replacement pipeline is operationally viable or superior across the full task surface the orchestrator-LLM currently handles (including the no-dispatch direct-completion paths that worked correctly in probes 1, 20 of Cycle 5 and note 18 here).

**Was the "must-delegate" framing examined?**

No. The cycle-status PLAY epistemic response records: "Must-delegate framing is load-bearing: if direct completion is allowed via the orchestrator-LLM path, the value proposition of agentic-serving... collapses — users could just use a frontier directly." This framing was introduced in the PLAY session, not by the practitioner. No alternative framing was surfaced for examination: for example, "direct completion is the correct path when no capability ensemble matches the task shape, and the ensemble library is sparse enough that most tasks don't match — so must-delegate is a future-state commitment, not a current-state architecture." The must-delegate framing presupposes a library rich enough to cover most task shapes. With two-to-six tagged capability ensembles, most tasks fall through to direct completion, and the "value proposition collapses" framing would apply today regardless of whether Cycle 7 ships.

The must-delegate framing is also in tension with the Cycle 5 PLAY note 20 post-reflection coda's finding, which established that the operational routing preference (direct → client-tools → ensemble) is a PROPERTY of the system under NL framing. Spike δ's PASS shows that deterministic chaining works. It does not show that the system-as-a-whole should route all tasks through ensemble dispatch.

**Were alternatives examined?**

The field notes and Spike δ write-up name framework-driven chaining as "the candidate." The write-up's next-step candidates (Spike ε; ADR-027 candidate) extend the framework-driven framing rather than examine alternatives. Three alternatives that were NOT examined as live candidates:

1. **Fine-tuning the orchestrator-LLM** to treat explicit input-passing instructions as binding. This is a training intervention at the model layer rather than an architectural intervention at the pipeline layer. Spike δ's hypothesis framed the failure as "orchestrator-LLM's chain-handling behavior" — fine-tuning addresses this at its source.

2. **Constrained decoding / tool_choice forcing** — if the orchestrator's tool-call interface were constrained to select `invoke_ensemble` before composing a narration, the input-passing decision would be moved to the framework layer without replacing the orchestrator-LLM. This is closer to the ADR-022 approach but enforced at the inference layer rather than the prompt layer.

3. **Re-engineering ADR-022 beyond a single system-prompt clause** — a more extensive system-prompt intervention that explicitly models the composition step-by-step, requiring the orchestrator to read the prior dispatch's substrate artifact before composing the next dispatch's input. Spike δ showed that input-passing works when the framework does it; the question is whether a sufficiently strong prompt could make the orchestrator-LLM do it correctly.

None of these appeared in the field notes or Spike δ write-up as alternative framings worth belief-mapping before the framework-driven pipeline is accepted as "the candidate."

**Cycle 7 scope inheritance concern:**

The cycle-status records Cycle 7's working title as "Framework-driven orchestration: routing as code" with "three load-bearing decisions inherited from PLAY findings." If Cycle 7 enters RESEARCH with the framework-driven pipeline as a settled premise rather than a hypothesis, the RESEARCH phase will confirm and elaborate the approach rather than examine whether it is the right approach. The must-delegate framing and the framework-driven pipeline candidate are both at risk of becoming load-bearing inherited premises under-examined.

---

### 2. The probe-1 misattribution correction

**The episode:** The field notes record note 18 with the header "[Corrected 2026-05-20 after practitioner challenge — initial reading misattributed a later dispatch to this probe.]" The initial reading had called probe 1's result a "major finding" that flipped earlier observations; the practitioner caught that probe 1 was direct LLM completion (no dispatch), not a dispatch that the agent had misread as successful.

**Does the field notes file accurately preserve the corrected reading?**

Yes, with one qualification. Note 18 is clearly marked "Corrected" and the correction is substantively incorporated — the note now accurately states "No dispatch fired on this probe." Note 20 ("WP-C orchestrator-context observation works for single-dispatch lookup") is also corrected: probe 2's "I didn't dispatch" answer is now recorded as "factually correct" rather than a false negative. The cross-cutting observations table at the end of the OpenCode section explicitly states "ADR-022 amendment is effective under bare-endpoint mode only" — which is the correct bounded conclusion.

**Was the overconfident initial reading preserved as a methodology-level lesson?**

Partially. The correction is marked but the initial misreading's cause is not analyzed. The agent read a 61.2s dispatch in the serve console as belonging to probe 1 when it actually belonged to the "Yes" re-invocation turn. This is a pattern-matching error: the agent inferred dispatch from timing evidence without correlating against the session_id or the actual serve-console turn-sequence. The field notes note the correction; they do not name the underlying error mode (cross-turn timing attribution without session-id anchoring). A methodology-level lesson would read: "serve-console timing evidence is insufficient to attribute dispatches to specific turns; dispatch_log.json + session_id correlation is required." This is not recorded.

**Does the correction language adequately bound ADR-022 amendment effectiveness?**

Yes. The BUILD verdicts table records: "ADR-022 system-prompt amendment shifts NL routing toward invoke_ensemble — Effective in bare-endpoint mode only." The note explicitly states: "Production deployments all use tool-rich clients... the amendment as shipped does not affect production NL routing." This is an appropriately strong negative result recorded with appropriate clarity.

---

### 3. Selection bias in the OpenCode session field notes (notes 18-25)

**Were positive findings recorded proportionately alongside negative findings?**

Examination of notes 18-25:

- Note 19 (ADR-023 observability fires on every dispatch) — recorded as "production-ready" with full per-event log sequence and precise heartbeat timing. This is the session's clearest positive finding, and it is recorded clearly.
- Note 20 (WP-C context observation works for single-dispatch lookup) — recorded with appropriate scope qualification ("Multi-dispatch summary narration confabulates — model-output-shape issue, not WP-C").
- Note 21 (malformed MiniMax XML diagnosis) — structural finding with non-blame framing; the diagnosis is accurate.
- Note 22 (composition probe catastrophic failure) — recorded at full detail.
- Note 23 (path hallucination / challenged-claim-persistence) — recorded with verbatim exchange showing four practitioner corrections.
- Note 24 (dispatch_log.json overwrite) — recorded as warranting explicit BUILD-regression or ADR amendment.
- Note 25 (web-searcher caching observed) — recorded as "good for cost" with the operator-experience qualification.

**What about probe 1's direct-completion response quality?**

The 13.4s direct-LLM-completion response in note 18 (two-pointer code, docstring, complexity notes, example usage, immutability note) is recorded primarily as the ADR-022 amendment effectiveness test (no dispatch). The *quality* of the direct completion response is not recorded. Spike δ implicitly surfaces the contrast: the 10.3s free-M2.5 direct completion in the morning produced "two function definitions... with a one-line note distinguishing idiomatic from educational use." Both direct-completion responses are substantively higher-quality and faster than the 60.7s code-generator ensemble output (which produced list-convert approach only, with synthesizer wrapping). This contrast — direct completion 10-13s and arguably better output; ensemble dispatch 60s and narrower output — is not explicitly discussed anywhere in the field notes or Spike δ write-up. It is a positive finding for the direct-completion path that the field notes' composition-drift focus does not attend to.

**Weight:** Moderate. The direct-completion quality is not a finding the PLAY agenda was designed to capture (the agenda focused on composition drift, ADR-022 effectiveness, ADR-025 substrate routing). But it is a finding that is implicitly relevant to the must-delegate framing: if direct completion produces comparable or better results for single-function coding tasks, the argument that "must-delegate" is load-bearing for the value proposition becomes more complex. The field notes do not record this tension.

---

### 4. The form-vs-content drift bifurcation

**The claim:** Spike δ's conclusion notes that form drift "persists" at the agent layer while content drift is resolved by framework-driven input-passing. The field notes encode this bifurcation in spike δ's conclusion: "The form-drift mechanism is independent of the chaining mechanism; it lives at the agent's response-shape layer."

**Is the bifurcation load-bearing or analytical?**

The empirical basis is real: spike δ's claim-extractor produced a structured analysis with section headers rather than the spec'd `(established)/(contested)` bulleted format. This is the same drift observed in notes 5 and 15. The bifurcation into "form-drift" (synthesizer ignores spec) and "content-drift" (orchestrator doesn't pass correct input) is empirically warranted by the spike δ result.

However, the bifurcation is doing architectural work that warrants examination: by characterizing content-drift as the orchestrator-LLM's failure and form-drift as the agent layer's failure, the analysis implicitly scopes form-drift OUT of the framework-driven pipeline proposal's solution space. The framework-driven pipeline is proposed as addressing content-drift; form-drift requires a separate solution (output-schema enforcement, fine-tuning, or spec redesign per the spike write-up).

**Was an alternative framing surfaced?**

No. An alternative framing: form-drift and content-drift are the same problem at different layers — both are violations of the ensemble's specified output contract. A unified solution (output-schema enforcement with reject-and-retry) would address both. The framework-driven pipeline addresses content-drift by removing the orchestrator-LLM from the input-passing step; output-schema enforcement would address form-drift by adding a post-processing validation step. The bifurcation routes these to separate solution tracks when they may converge on a single architectural commitment (schema-as-enforcement rather than schema-as-documentation, per ADR-024's current documentary posture).

**Weight:** Moderate. The bifurcation is analytically useful and empirically grounded. The concern is that it separates form-drift from the framework-driven pipeline's scope in a way that may make the content-drift solution appear sufficient when form-drift remains a blocking concern for any production composition workflow.

---

### 5. Cycle 7 scope inheritance

**What is being inherited?**

The cycle-status PLAY epistemic response records three "load-bearing" commitments entering Cycle 7:

1. Composition drift is in the orchestrator-LLM layer, not the ensemble layer.
2. Framework-driven plan→dispatch→synthesize is the candidate architecture.
3. Must-delegate framing is load-bearing for the value proposition.

**Assessment of each:**

**Claim 1 (drift in orchestrator-LLM)** is well-supported. Spike δ demonstrated that the same ensembles chain correctly when the orchestrator-LLM is bypassed. The prior orchestrator-driven probes (curl Test 7, OpenCode composition probe) demonstrated fabrication and failed chaining. This is the strongest of the three inherited claims.

**Claim 2 (framework-driven pipeline is the candidate)** is premature as an inherited commitment. "Candidate" implies it is the primary framing Cycle 7's RESEARCH phase will examine. Given that Spike δ tested one composition once, and three alternative architectural approaches were not examined (see §1 above), the RESEARCH phase entry condition should be "framework-driven pipeline is a candidate worth investigating alongside alternatives," not "the candidate."

**Claim 3 (must-delegate framing is load-bearing)** is the most under-examined of the three. It is an agent-introduced framing from PLAY close. It conflates a future-state aspiration (when the library is rich enough to cover most tasks, direct completion undermines the value proposition) with a current-state architectural requirement. It was not belief-mapped. It was not examined against the empirical observation that direct completion often produces faster, higher-quality results for the task shapes tested (single-function coding). It is being inherited as a constraint on Cycle 7's design space.

**If Cycle 7 RESEARCH treats all three as settled premises**, the research phase will confirm the framework-driven architecture rather than examine whether it is the right response to the drift problem. The cycle's own research commitment (cited in the PLAY epistemic response: "Cycle 7 opens to address this") will become confirmatory rather than investigative.

---

## Interpretation

### Pattern assessment

The Cycle 6 PLAY field notes are empirically strong for notes 11-25. The observational layer is well-anchored: dispatch_log entries, serve-console excerpts, practitioner verbatim quotes, disk-state verification through session directory inspection. The misattribution correction (notes 18, 20) is handled transparently. The ADR-022 and ADR-023 verdicts are proportionately balanced — the observability surface's production-readiness is recorded as clearly as the ADR-022 amendment's bare-endpoint-only effectiveness.

The susceptibility concentration is at the PLAY close boundary — specifically, in the Spike δ section and the cycle-status PLAY epistemic response. This is a consistent pattern with prior PLAY snapshots: the observation notes are empirically disciplined; the synthesis/summary layer encodes architectural claims that outpace the evidence.

Three signals converge into a narrowing pattern:

1. **Spike δ conclusion velocity:** one probe, one composition, one PASS, architectural implication drawn and encoded as Cycle 7's working title. The "Scope of the spike" section names what was not tested, which is honest — but the architectural implication was drawn anyway, and the cycle-status records it as a settled framing rather than a hypothesis.

2. **Must-delegate framing adopted without examination:** the framing was introduced at PLAY close (agent-introduced, not practitioner-stated as a general principle). It was recorded in cycle-status as load-bearing for the value proposition. No belief-mapping was applied; no alternative framing was surfaced (e.g., direct completion is appropriate when no ensemble matches, must-delegate is aspirational rather than current-state). The framing enters Cycle 7 as a constraint.

3. **Alternative architectures not named as live candidates:** the spike write-up's next-step candidates (Spike ε; ADR-027 candidate) extend the framework-driven framing. Fine-tuning, constrained decoding, and extended system-prompt intervention are mentioned as "spitball items" in cycle-status but are not examined against the spike's evidence.

This pattern is consistent with earned-confidence failure rather than sycophantic reinforcement in the observation notes (notes 11-25 are genuinely earned by the session's empirical work). The signal is concentrated at the PLAY→Cycle 7 synthesis boundary: a genuine finding (orchestrator-LLM confabulates in multi-dispatch composition) was translated too quickly into a specific architectural commitment (framework-driven pipeline is the candidate) without examining alternative framings or testing the must-delegate premise.

The probe-1 misattribution correction is a positive signal. The practitioner caught the error; the field notes incorporate the correction clearly; the BUILD verdicts table reflects the corrected reading. This is the opposite of sycophantic reinforcement — the correction reversed an overconfident initial reading. The methodology-level lesson (cross-turn timing attribution requires session-id anchoring) should be named explicitly rather than left implicit.

The form-vs-content drift bifurcation is analytically useful and introduces a valuable distinction. The moderate concern is that it scopes form-drift out of the framework-driven pipeline's solution space, potentially creating a two-track remediation where a unified approach (schema-as-enforcement) is available.

**Earned confidence vs. sycophantic reinforcement:**

Notes 11-25: earned. The observational layer is grounded in disk evidence, serve-console output, dispatch_log entries, and practitioner verbatim. The corrections are incorporated honestly.

Spike δ section and cycle-status epistemic response: sycophantic reinforcement risk is present. The framework-driven pipeline claim was introduced by the agent, confirmed by a single probe, and adopted without practitioner-independent examination of alternatives. The must-delegate framing was introduced by the agent at PLAY close and encoded as a constraint without the practitioner explicitly accepting it as a general principle vs. a session-observation.

---

## Recommendation

**Grounding Reframe recommended** before Cycle 7 RESEARCH entry.

The Cycle 6 PLAY observation work is strong and should not be discarded. The specific intervention is at the PLAY→Cycle 7 scope-setting boundary, where two under-examined premises are being inherited as architectural constraints.

### Uncertainty 1: The framework-driven pipeline as "the candidate"

**What is uncertain:** Whether framework-driven plan→dispatch→synthesize is the correct response to the composition-drift finding, versus alternative interventions at the model, prompt, or inference layer.

**Grounding action:** Before Cycle 7's RESEARCH phase opens with a framework-driven framing, belief-map the alternative architectures. Specifically:

- *What would have to be true for a more aggressive system-prompt intervention (beyond ADR-022's single clause) to eliminate orchestrator-LLM input-passing failures?* Spike δ tested the framework-driven bypass; an alternative spike would test a structured-input-passing prompt that explicitly models the chain step. If this fails, the framework-driven approach is more clearly motivated. If this works, the choice becomes a tradeoff (prompt engineering vs. pipeline replacement), not a settled commitment.
- *What would have to be true for constrained decoding / tool_choice forcing to address the routing-and-chaining failure?* Name the belief; test it against the spike δ evidence.

This belief-mapping can be done analytically (15-30 minutes at RESEARCH entry) before committing to the framework-driven framing as the research question's organizing premise. If the belief-mapping confirms that alternatives are clearly inferior, the framework-driven pipeline enters RESEARCH as a well-motivated hypothesis rather than an inherited conclusion.

### Uncertainty 2: The must-delegate framing as a value-proposition constraint

**What is uncertain:** Whether "direct completion undermines the value proposition if allowed" is a current-state architectural requirement or a future-state aspiration conditional on library coverage.

**Grounding action:** Name the premise explicitly and test it against the observed data:

- Under paid M2.5 + OpenCode tool-rich + NL framing, direct completion produced a 13.4s, multi-variant, well-annotated response (note 18). The code-generator ensemble took 61s and produced narrower output (note 7, direct invoke).
- If must-delegate is the constraint, the system needs to consistently produce better results through ensemble dispatch than through direct completion. The current evidence does not establish this for single-function coding tasks.
- The belief-mapping question: *What would have to be true for must-delegate to be justified as a current-state constraint rather than a future-state aspiration?* One answer: the library covers most task shapes AND ensemble dispatch quality reliably exceeds direct completion quality for those shapes. Both conditions are unmet today.

This grounding action prevents Cycle 7 RESEARCH from designing a framework-driven pipeline to enforce must-delegate when the value case for must-delegate over the current six-ensemble library hasn't been established.

### On the direct-completion quality contrast

**What is unrecorded:** The field notes do not document that direct-LLM-completion responses (notes 1, 18) produced faster and qualitatively comparable outputs to the code-generator ensemble (note 7). This contrast is load-bearing for the must-delegate framing.

**Grounding action:** Before Cycle 7 scoping, explicitly compare the note 1/18 direct-completion responses against the note 7 ensemble output on the same task. If direct completion is faster and qualitatively equivalent, the must-delegate framing requires a different justification than output quality (e.g., calibration gate access, observability, cost per token at scale). Name that justification explicitly so Cycle 7 is designing against a stated requirement rather than an inherited assumption.

### On the methodology-level lesson from the probe-1 correction

**What is named but not analyzed:** The probe-1 misattribution occurred because serve-console timing was attributed to a probe without session-id anchoring. The field notes correct the result but do not extract the methodology lesson.

**Grounding action (low-cost):** Add a note to the cycle-status or field-notes post-reflection section: "Serve-console timing evidence is insufficient to attribute dispatches to specific probe turns. Dispatch_log.json + session_id correlation is required for per-probe attribution. Cross-turn timing ambiguity is a known failure mode when multiple requests open in rapid succession."

This is not cycle-blocking — it is a methodology hygiene note that prevents the same attribution error in Cycle 7's PLAY phase.

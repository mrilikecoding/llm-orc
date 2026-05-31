# Argument Audit Report

**Audited document:** `/Users/nathangreen/Development/eddi-lab/llm-orc/docs/agentic-serving/essays/essay-outline-006-cross-compatibility-routing-surface.md`
**Source material:** `/Users/nathangreen/Development/eddi-lab/llm-orc/docs/agentic-serving/essays/research-logs/research-log.md` (loop-back log: Spikes π, ρ, σ.1, σ.2)
**Genre:** Essay-Outline
**Date:** 2026-05-24
**Scope:** Amendment B — C8 new material only (C1–C7, Sections 1–8 treated as previously verified; re-audit-after-revision mode)

---

## Section 1: Argument Audit

### Summary

- **Genre:** Essay-Outline
- **Argument chains mapped:** 1 (C8 sub-tree: W8.1–W8.5 with E8.1.1–E8.5.1)
- **Issues found:** 4 (P1: 0, P2: 3, P3: 1)
- **Pyramid coverage map:** included (C8 only)
- **Expansion-fidelity findings:** P1: 0, P2: 2, P3: 1

---

### Pyramid Coverage Map — C8 material

| Abstract Conclusion | Argument-Graph Nodes | Body Section | References Cited |
|--------------------|-----------------------|--------------|------------------|
| C8. ADR-027 WP-A terminates only in text/single-turn; client-tool-action terminal necessary; justified by execution model not geography; two structural gaps (layer A; F-ρ.1 bridge). | C8, W8.1, W8.2, W8.3, W8.4, W8.5; E8.1.1, E8.1.2, E8.2.1, E8.3.1, E8.3.2, E8.4.1, E8.4.2, E8.5.1 | Section 9 (C8) | [research-log-loopback], [opencode] |

**META-anchored sections in this audit scope:** none (Section 9 anchors to C8 as a developmental section; no new META sections added by Amendment B).

**C8 node-to-body coverage check:**

| Graph node | Body anchor in Section 9 | Status |
|------------|--------------------------|--------|
| C8 (top-level claim) | CLAIM 1 bullet | Covered |
| W8.1 (tool_calls necessary) | CLAIM 2 + EVIDENCE bullets (π Phase A/B) | Covered |
| W8.2 (planner + terminal compose; C1 suppression absent) | CLAIM 3 EVIDENCE Spike ρ bullet | Covered |
| W8.3 (cheap model sustains loop; integrated pattern holds) | CLAIM 3 EVIDENCE Spike σ.1/σ.2 bullets | Covered |
| W8.4 (two structural gaps) | CLAIM 4 EVIDENCE bullets (layer A; F-ρ.1) | Covered |
| W8.5 (geography vs. execution-model correction) | CLAIM 2 SYNTHESIS bullet | Covered — but see P2-1 below |
| E8.1.1 | EVIDENCE Spike π Phase A | Covered |
| E8.1.2 | EVIDENCE Spike π Phase B | Covered |
| E8.2.1 | EVIDENCE Spike ρ | Covered |
| E8.3.1 | EVIDENCE Spike σ.1 | Covered |
| E8.3.2 | EVIDENCE Spike σ.2 | Covered |
| E8.4.1 | EVIDENCE layer A/B distinction | Covered |
| E8.4.2 | EVIDENCE F-ρ.1 artifact bridge | Covered |
| E8.5.1 | CLAIM 2 SYNTHESIS (co-location / geography correction) | Thin — see P2-1 |

**Reverse Boundary 2 check:** Section 9 carries a SCOPE QUALIFICATION and a VALIDATION-SPIKE DECISION block with no parenthetical anchor additional to (C8). Both are non-developmental (scope caveats, ADR-087 recording); they read as META-equivalent text within the section body. No orphaned developmental bullets detected.

**Boundary 3 (citations → References):**

- `[research-log-loopback]` — entry present in References section at line ~314. Resolves.
- `[opencode]` — entry present in References section at line ~315. Resolves.
- `[adr-025]` cited in E8.4.2 body text — entry present at line ~288. Resolves.
- `[agentic-serving-cycle-status]` cited in Section 9 CLAIM 1 EVIDENCE — entry present at line ~292. Resolves.

All Boundary 3 citations resolve. No violations.

---

### Expansion-Fidelity Findings

**P1 findings: none.**

All Abstract–Argument-Graph (Boundary 1), Argument-Graph–Citation-Embedded Outline (Boundary 2), and Reverse Boundary 1/2 checks pass for C8.

**P2 findings (weak expansion or framing concerns):**

- **P2-1 — W8.5 body expansion is thin.** W8.5 ("the finding corrects the loop-back's original justification: filesystem geography is contingent; the load-bearing reason is the execution model") is substantive enough to warrant its own warrant-level slot, yet Section 9 develops it only inside the CLAIM 2 SYNTHESIS bullet: "Parity rests on the client's execution model, not filesystem geography — co-location of `llm-orc serve` and the client dissolves the geography argument (E4.2.1) but the conclusion survives." One sentence. The Argument-Graph's E8.5.1 ("Practitioner constraint-removal (ADR-082 entry)") frames this as *evidence*, but the logical work W8.5 does — correcting the basis of the prior loop-back finding — is a reframing argument that warrants more than a one-line synthesis note. The pyramid holds technically, but the expansion is weak.
  - **Recommendation:** Promote the W8.5 content to a dedicated CLAIM bullet in Section 9 (or expand the CLAIM 2 SYNTHESIS into two sentences naming: (a) co-location makes direct delivery trivially feasible, (b) yet execution-model parity is unaffected, (c) the E4.2.1 geography argument in the prior essay body no longer grounds the loop-back). The correction is the epistemic move; a reader skimming Section 9 may miss it.

- **P2-2 — E8.2.1 makes a claim about C1 suppression not recurring, but the scope note in the research log is more hedged than the Argument-Graph states.** The Argument-Graph node W8.2 / E8.2.1 states: "The C1 tool-rich-client routing suppression does not recur under the bounded routing planner (which routes on request content per AS-10, not on the client's declared tools)." Section 9 echoes this as "C1 tool-rich-client suppression did not recur." The research log (Spike ρ section) establishes this for one capability-matched request (`hello world` code-generation). The Argument-Graph's phrasing "does not recur" reads as a general structural claim, while the evidence is n=1. The Scope Qualification in Section 9 does call this out ("n=1 capability-matched task"), but the Argument-Graph node itself carries no qualifier and the Abstract sentence on C8 escalates to "the C1 tool-rich-client routing suppression does not recur under the bounded routing planner" without a hedging modifier.
  - **Recommendation:** Add a scope qualifier to E8.2.1 ("established for one capability-matched task; structural basis is AS-10's content-only routing; multi-task generalization is the structural inference") and mirror it in the Abstract's C8 sentence. The scope note in Section 9 captures this but the Argument-Graph node does not.

**P3 findings:**

- **P3-1 — W8.3 / E8.3.1 slightly conflates the loop-driver question with the parity question.** W8.3 is titled "A cheap local model sustains the client's multi-turn agentic loop... the integrated north-star pattern composes end-to-end." E8.3.1 (σ.1) is the *no-delegation* baseline (qwen3:14b directly, no ensemble hand-off), and E8.3.2 (σ.2) is the delegation-integrated run. The relationship between the two sub-spikes (σ.1 establishes the layer-A floor; σ.2 demonstrates the integrated pattern) is implicit in the Argument-Graph but not spelled out. A reader building on this outline may not understand that σ.1 is a prerequisite floor test, not a redundant variant of σ.2. The Section 9 body does distinguish them in CLAIM 3's EVIDENCE bullets. Minor; the pyramid holds.
  - **Recommendation:** Add a one-clause note to E8.3.1 in the Argument-Graph ("floor test: layer-A feasibility without delegation, prerequisite for E8.3.2") to make the σ.1 → σ.2 dependency explicit at the graph level.

---

### P1 — Must Fix

None.

---

### P2 — Should Fix

**P2-1 — W8.5 expansion thin (Section 9 / geometry-correction warrant)**

- **Location:** Argument-Graph W8.5 / E8.5.1 → Section 9 CLAIM 2 SYNTHESIS bullet
- **Claim:** W8.5 corrects the loop-back finding's original justification. The Argument-Graph asserts this as a named warrant with its own evidence node.
- **Evidence gap:** Section 9 develops W8.5 in a single synthesis sentence. The argument that co-location dissolves the geography premise while leaving the conclusion intact is the epistemic payoff of the constraint-removal exercise in Step 1.2 of the research log. It deserves a dedicated CLAIM bullet or a two-sentence expansion, not a subordinate synthesis clause.
- **Recommendation:** Expand the CLAIM 2 SYNTHESIS in Section 9 or add a separate CLAIM bullet for the geography-vs.-execution-model correction. The research log's §Step 1.2 and §Spike π "Necessity verdict" paragraphs contain the full argument; the Section 9 body should surface it at that resolution.

**P2-2 — E8.2.1 scope of C1-suppression-absent claim**

- **Location:** Argument-Graph W8.2 / E8.2.1; Abstract C8 sentence; Section 9 CLAIM 3 EVIDENCE Spike ρ bullet
- **Claim:** "The C1 tool-rich-client routing suppression does not recur under the bounded routing planner."
- **Evidence gap:** The claim is grounded on one capability-matched task (Spike ρ, `hello world` code-generation). The Argument-Graph node reads as a structural general claim. The structural basis (AS-10: planner routes on request content, never sees client's declared tools) is sound, but the empirical support is n=1. The Section 9 scope qualification captures this, but the Argument-Graph node and the Abstract do not.
- **Recommendation:** Qualify E8.2.1 in the Argument-Graph: "established empirically for n=1 capability-matched task; general structural basis is AS-10 (routing planner reads request content, not client-declared tool list)." Mirror the qualifier in the Abstract's C8 sentence so the scope is visible at both levels of the pyramid.

---

### P3 — Consider

**P3-1 — σ.1 / σ.2 prerequisite relationship implicit in Argument-Graph**

- **Location:** Argument-Graph W8.3 / E8.3.1 / E8.3.2
- **Claim:** Warrant W8.3 conflates the layer-A floor test (σ.1, no delegation) and the integrated delegation test (σ.2) as coordinate evidence nodes.
- **Evidence gap:** The logical structure is σ.1 establishes feasibility floor → σ.2 builds on that floor to validate the integrated pattern. The Section 9 body makes this clear; the Argument-Graph treats them as parallel rather than sequential.
- **Recommendation:** Annotate E8.3.1 with a parenthetical noting its role as prerequisite floor test for E8.3.2. One clause; no restructuring required.

---

## Section 2: Framing Audit

The framing audit examines what the C8 material foregrounds and what the available evidence could have supported but did not.

### Question 1: What alternative framings did the evidence support?

**Alternative A — "ADR-027 is wrong, not just incomplete."**

The research log is careful to conclude "ADR-027 is incomplete, not wrong" and the essay-outline echoes this at multiple points. The evidence base could have supported a harder reading. Spike π Phase A establishes that the text-terminal produces a structurally defective loop (OpenCode acts on an unverified claim, analogous to PLAY note 22). The research log itself notes the N-1/I-1 analogy "holds in the load-bearing respect." Under this framing the terminal gap is not an addendum to ADR-027 but evidence that ADR-027's design was under-specified from the beginning: it handled the answer-a-question case but did not reason about the do-work case. The "incomplete, not wrong" framing preserves ADR-027's legitimacy but forecloses the question of whether the architecture needs a more substantial rethink (e.g., whether the single-pipeline design is the right abstraction when the terminal shape is request-type-dependent).

*What would the reader need to believe for this framing to be right?* That a pipeline whose terminal is wrong for the primary north-star use case is substantively mis-designed, not merely unfinished.

**Alternative B — "Layer A is the primary finding; the terminal is secondary."**

The essay foregrounds the client-tool-action terminal (C8 title, Section 9 title) and treats layer A as a structural consequence (W8.4). The research log's σ spikes arguably establish the opposite priority: the layer-A/B distinction is the novel architectural insight (neither the routing planner nor the ensembles drive the loop; a distinct role is needed), while the terminal is a more mechanical consequence of OpenCode's execution model. Under this framing C8 should be titled "A per-turn agentic-loop-driver (layer A) is a distinct role ADR-027 lacks" and the terminal would be a corollary.

*What would the reader need to believe?* That the more important gap is architectural (a missing role the system has no current abstraction for) rather than mechanical (a missing emission format the pipeline lacks).

**Alternative C — "The north-star requirement is underspecified; C8 validates a prerequisite but not the target."**

C8 concludes with "the integrated pattern holds end-to-end." The evidence (Spike σ.2) runs a 3-step `hello world`-class task: write `calc.py`, write `test_calc.py`, run it. The north-star is "run RDD (or similar long-horizon process) via OpenCode." The gap between a 2-turn batched plan-and-execute and long-horizon RDD is large. Under this framing C8 establishes the terminal/delegation machinery is feasible at the mechanism level but leaves the north-star itself unvalidated. The essay does carry a scope qualification to this effect, but the Abstract and Argument-Graph present the findings with more closure than the scope caveat warrants.

*What would the reader need to believe?* That mechanism-level validation on degenerate tasks (1-2 turns, simple code generation) is insufficient evidence that the north-star is achievable, and that the gap between the two deserves more prominent treatment in the argument structure.

---

### Question 2: What truths were available but not featured?

**T1 — F-ρ.2 (config hygiene: `agentic-tier-cheap-general` undefined) is surfaced in the research log but absent from the Argument-Graph.**

The research log's Spike ρ findings include F-ρ.2: `agentic-tier-cheap-general` is referenced in `per_skill_tier_defaults` but not defined as a model profile in the current config. This means the production code-generator ensemble cannot dispatch without a BUILD fix. The research log marks this as a BUILD/deployment concern. The Essay-Outline's C8 does not mention it. It appears only in a footnote-like position in the research log.

*Why excluded?* Correctly scoped out: it is a deployment config issue, not an architectural finding. But it is a gap between the "validated end-to-end" claim and the actual production state — the spike used a stand-in (`spike-pi-code-generator`) precisely because the production ensemble cannot run. This weakens the "integrated north-star pattern VALIDATED" phrasing slightly: what was validated is a stand-in pattern, not the production pattern.

*Would its inclusion change the argument?* It would require softening the "validated end-to-end" language in W8.3 / E8.3.2 / Section 9 to "validated at the mechanism level through a stand-in; production ensemble blocked by F-ρ.2 until BUILD fix." The scope qualification in Section 9 does note "end-to-end through the *production* `code-generator`... [is a] BUILD-phase concern," but the Argument-Graph node W8.3 does not carry this hedging.

**T2 — The Phase 0 observation findings (OpenCode's `skill` and `task` tools) are "noted, not tested."**

The research log's Phase 0 findings note that OpenCode declares native `skill` and `task` (subagent) tools and marks them as "out-of-scope but north-star-relevant." These tools are the surface a "run RDD via OpenCode" flow would use. The Essay-Outline does not incorporate this finding at all — neither in the C8 sub-tree nor in the scope qualifications. Under the north-star framing this is a notable gap: the spike observed that the target client already has the tool surface the north-star needs, but the essay does not use this as evidence for or against C8's conclusions.

*Why excluded?* "Out-of-scope but north-star-relevant" — a deliberate decision to keep the loop-back focused. Reasonable, but worth flagging: a reader building on C8 to architect the north-star integration would want to know the tool surface exists and has been observed.

*Would its inclusion change the argument?* It would not change C8's conclusions but would add a positive evidence note to the north-star feasibility picture, strengthening (or at least corroborating) the "integrated pattern composes" claim. The essay's C8 framing is conservative (it stops at mechanism validation); the `skill`/`task` observation could shift the framing toward "the client surface is already prepared."

**T3 — The headless `opencode run` mode used in spikes may not represent the interactive use case.**

All three spikes (π, ρ, σ) drove OpenCode in headless mode (`opencode run "..."` with `--format json`). The research log notes this once (Phase 0: "headless `opencode run`"). The north-star scenario ("use OpenCode to run RDD via agentic-serving") is an interactive session, not a headless one. Headless mode may surface different permission behavior, tool execution behavior, or multi-turn handling. The essay does not flag this as a scope limitation.

*Why excluded?* Likely treated as a minor implementation detail. But the permission gate behavior in Phase B ("headless `opencode run` executed the write without stalling — no permission config required") is specifically a headless characteristic; in interactive mode the user might see a permission prompt, which would change the parity assessment.

*Would its inclusion change the argument?* It would add a qualifier to the parity claim: "parity established in headless mode; interactive mode may differ in permission-gate handling." This is a P3-level observation but worth surfacing.

---

### Question 3: What would change if the dominant framing were inverted?

The dominant framing of C8 is: **ADR-027 is incomplete but the fix is additive (add terminal + layer A driver); the existing pipeline philosophy is sound.**

**Inverted framing: The terminal gap reveals that ADR-027 solved the wrong problem.**

Under the inversion: ADR-027 was designed to replace the orchestrator-LLM with a deterministic pipeline for the answer-a-question case. The north-star is do-work-on-the-client, which requires a fundamentally different terminal shape, a new orchestration role (layer A), and an artifact bridge (F-ρ.1). These are not addenda to the answer-a-question design; they are requirements of a different design problem. The "incomplete, not wrong" verdict is a framing choice, not a logical necessity.

Under this inversion:
- **Claims that become weaker:** The claim that "the fix aligns with ADR-027's philosophy" weakens — the terminal is not a pipeline extension, it is a protocol change (answer vs. act). The claim that "ADR-027 reduces the NL-routing-fraction beyond the C1 baseline universally" (C7 W7.2.3) weakens in the north-star context, where reducing NL routing fraction is not the primary metric; loop coherence is.
- **Claims that become stronger:** The C4 finding ("framework-driven composition continuation is required") strengthens — it is not just a remedy for orchestrator-LLM failure but a positive statement that the framework must participate actively in the client's execution loop, not just in the server-side pipeline.
- **Evidence that becomes more salient:** The F-ρ.1 artifact bridge (E8.4.2) becomes central rather than a structural consequence. The artifact store was designed for server-side retention (ADR-025); co-opting it to bridge deliverables into client tool-calls is an architectural use the design did not anticipate. Under the inverted framing this is a design tension, not a gap.
- **What the document would need to address:** Whether the "one pipeline, multiple terminal shapes" architecture is coherent, or whether the do-work use case needs its own pipeline variant. The essay takes the former position implicitly (the terminal is a configuration of the same pipeline); the inverted framing would require a stated rationale for that choice.

---

### Framing Issues

**P2-3 — Scope qualification on C8's "validated end-to-end" language does not visibly cascade to Argument-Graph node W8.3.**

- **Location:** Argument-Graph W8.3 / E8.3.2; Abstract C8 sentence ("the integrated pattern — cheap loop-driver + per-turn ensemble delegation — composes end-to-end"); Section 9 VALIDATION-SPIKE DECISION block
- **Issue:** The Abstract and Argument-Graph assert "composes end-to-end" based on a stand-in ensemble run (`spike-pi-code-generator`, not the production `code-generator`), in headless mode, on a 2-turn batched task. The scope qualification in Section 9 captures the limits but is placed in a non-structural position (a SCOPE QUALIFICATION bullet after the CLAIM/EVIDENCE bullets, and in the VALIDATION-SPIKE block). A reader parsing the Argument-Graph alone would not see the caveat.
- **Recommendation:** Add a parenthetical scope note to E8.3.2 in the Argument-Graph: "(stand-in ensemble; headless mode; 2-turn batched task; production validation is BUILD-phase work per F-ρ.1/F-ρ.2)." This brings the scope language down from the Citation-Embedded Outline into the graph layer where a downstream decision-maker will encounter it.

**P3-2 — Phase 0 `skill`/`task` observation is excluded from C8 without explanation.**

- **Location:** Research log §Phase 0 "Out-of-scope but north-star-relevant" observation; absent from C8 Argument-Graph and Section 9
- **Issue:** The observation that OpenCode already declares `skill` and `task` tools is positively relevant to the north-star framing. Its exclusion is reasonable (scope discipline) but the research log flags it as "north-star-relevant." Under framing audit scrutiny, the reader of C8 has no signal that this positive evidence exists.
- **Recommendation:** Either add a one-line note in the Section 9 SCOPE QUALIFICATION ("OpenCode's native `skill` and `task` tools — the natural surface for a north-star RDD integration — observed in Phase 0 but not tested; noted for ARCHITECT phase") or add a sentence to the Abstract's C8 prose. P3 because the exclusion does not change C8's conclusions.

---

*Single-purpose re-audit dispatched per ADR-094 re-audit-after-revision rule. Convergence-Saturation Signal verdict line omitted.*

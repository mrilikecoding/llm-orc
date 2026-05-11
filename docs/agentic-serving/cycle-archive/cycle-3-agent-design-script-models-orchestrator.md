# RDD Cycle Archive — Agentic Serving / Cycle 3: Agent Design (Script + Local Models + Cloud Orchestrator)

**Artifact base:** `docs/agentic-serving/`
**Plugin version at cycle start:** v0.8.5
**Migration version at cycle start:** 0.8.5 (`housekeeping/.migration-version`)
**Cycle started:** 2026-05-01
**Cycle closed:** 2026-05-01
**Cycle close shape:** Mode B (Research Only) declared at cycle close — research-phase findings inherited into Cycle 4 rather than progressing through DISCOVER → MODEL → DECIDE → ARCHITECT → BUILD on this cycle's target. No Cycle 3 essay produced *at cycle close*; the five research-log artifacts plus audit trail were the cycle's deliverable. **Retroactive essay 004 added 2026-05-04** at practitioner request during Cycle 4: `essays/004-three-arm-architecture-comparisons.md` rolls up the cycle's substantive findings (Spike A cross-tier complementarity; Spike B multi-turn ceilings + F1/F2 methodological finding; Spike C 3-of-3 vs 0-of-2 cross-file verification; Spike D multi-ensemble pilot + opencode CLI stall; the susceptibility snapshot's three grounding actions). The retroactive essay closes the essay-numbering gap (essays now run 001 → 002 → 003 → 004 → 005 without skip) and provides a publishable-quality synthesis of Cycle 3's deliverable that earlier sat only in research-log form.

## Phase Status

| Phase | Status | Artifact | Key Epistemic Response |
|-------|--------|----------|----------------------|
| RESEARCH | ✅ Complete | Five research logs: `essays/research-logs/004a-lit-review-agent-design.md`, `004b-spike-a-cycle3.md`, `004c-spike-b-cycle3.md`, `004d-spike-c-cycle3.md`, `004e-spike-d-pilot-cycle3.md` | Practitioner committed to closing Cycle 3 at research-phase end and opening Cycle 4 grounded in supported design methods for orchestrator + ensembles. Cycle reframed mid-Spike-A: central question is whether cheap-orchestrator + orchestration competes with expensive frontier model. Mid-cycle methods-reviewer dispatch reshaped the trajectory through Spike C (three-arm code-review fixture) and Spike D (multi-ensemble pilot). Susceptibility snapshot identified two embedded-conclusion risks at synthesis boundary; three grounding actions inherited into Cycle 4 entry. |
| DISCOVER | ☐ Deferred to Cycle 4 | — | — |
| MODEL | ☐ Deferred | — | — |
| DECIDE | ☐ Deferred | — | — |
| ARCHITECT | ☐ Deferred | — | — |
| BUILD | ☐ Deferred | — | — |
| PLAY | ☐ Optional | — | — |
| SYNTHESIZE | ☐ Optional | — | — |

## Feed-Forward Signals

### From RESEARCH (Loop 1 — combined literature review, 004a)

1. **Script-as-orchestrator shape has a sparse but substantive literature corpus.** OneFlow (Lin et al. 2025) and Routine (Wang et al. 2025) describe deterministic-routing patterns where the script is the primary orchestrator and the LLM is a subordinate step; Compiled AI (Ma et al. 2025) treats LLM-as-callable-function within deterministic control flow. None of these publications frame their work as the dominant pattern in the field; the dominant pattern is LLM-as-orchestrator. The script-as-orchestrator shape is named where it appears, but it is not yet the operative paradigm in published agent-design literature.

2. **τ²-Bench (arXiv:2506.07982) extends tau-bench with explicit multi-policy compliance scoring.** Carries forward the tau-bench reliability ceiling finding (GPT-4o under 50% task success; pass^8 below 25%) into a richer evaluation regime. Cycle 3's Spike B did not reach this regime; the regime remains empirically open.

3. **CLEAR (Rony et al. 2025, arXiv:2511.14136)** provides a cost-normalized accuracy framework for evaluating agent configurations on production tasks. Carries forward as a candidate frame for Cycle 4's evaluation of orchestrator + ensembles configurations, with the qualification that "cost" in CLEAR refers to API token cost; environmental cost and local-first axes still require qualitative scoring under the four-priorities frame.

4. **MAST (Multi-Agent System Taxonomy)** and Lee et al. (2025) provide vocabulary for the design space without prescriptive engineering guidance. The literature gives names for patterns; it does not yet provide selection criteria from task properties to architecture choice.

5. **PAE (Cao et al. 2025, arXiv:2603.03116)** offers a procedural-independence scoring framework for evaluating agent decomposition quality. Carries forward as a candidate scoring frame for Cycle 4's mechanism-isolation experiments.

### From RESEARCH (Spike A — cross-tier directed prompts on documentation review, 004b)

6. **Cross-tier heterogeneity-uncorrelated-errors finding on the cycle-2 README fixture.** Seven arms tested: arm1 (A2 baseline cheap-bare), arm2 (A2+script context), arm3 (cheap-with-direction, debiased version after CONTAMINATED system-prompt naming the failure category), arm4 (script-only no README), arm-frontier-bare, arm-frontier-with-script. Cheap-with-direction caught semantic bugs frontier missed: 3/3 vs 0/4 on undefined-profile bugs in directed prompt; 1/3 in debiased prompt. The mechanism is cross-tier complementarity — different error distributions across tiers — not capability-tier ordering.

7. **Spike A's central-question reframe was practitioner-originated mid-spike.** Practitioner sharpened the question to "does cheap-orchestrator + orchestration compete with a more expensive frontier model?" The cycle absorbed the reframe quickly and ran two frontier arms in response without pausing to examine whether the prior framing ("outcomes over an agentic session; agent shape is means") and the new framing were in tension. The susceptibility snapshot identifies this as a mild-to-moderate framing-adoption signal; the comparison baseline ("frontier-bare single-shot with no file access") is partially favorable to the architecture by design. Cycle 4 should name this baseline as a scope condition.

### From RESEARCH (Spike B — multi-turn tau-shape and real-session, 004c)

8. **F1 (turn-by-turn dispatch) is the correct method for multi-turn frontier-tier testing under the no-paid-API constraint; F2 (single-shot facsimile) is structurally limited.** F2 imagined-state bias surfaced in 5/6 traces (subagents imagined fines that didn't exist; library state was fabricated rather than tracked). F1 turn-by-turn 6/6 success on the library-checkout fixture once state-key normalization and search-tokenization fixes landed.

9. **Cross-tier complementarity does not replicate at multi-turn on the library-checkout fixture.** Cheap arms 12/12 post-fixture-fix; frontier arms 6/6; real-session haiku-authoring 6/6 across all arms. At the multi-turn complexity tested, neither tier hits a reliability ceiling and neither tier outperforms the other. Cross-tier complementarity is task-class-dependent, not architectural — Spike A's finding does not generalize.

10. **Spike B's easy-regime confirmations did not answer the central question.** The practitioner's mid-Spike-B observation ("MiniMax is going to perform similarly to Sonnet on simple tasks. That's more or less what you determined, yes?") triggered the methods-reviewer dispatch that reshaped the cycle's mid-trajectory.

### From RESEARCH (mid-cycle methods reviewer, research-design-review-cycle-3-mid)

11. **Reviewer confirmed P1 findings on regime mismatch and `+ orchestration` gap.** The reviewer named the cycle's hazard: Spike B's confirmations on easy regimes do not test the central question; the cycle had not yet exercised the architecture's `+ orchestration` primitive on a task class where orchestration could plausibly differentiate. Recommended a three-arm code-review fixture (cheap-bare, cheap+ensemble, frontier-bare) with pre-spike fixture validation to avoid Spike B's easy-regime trap.

12. **Methods reviewer's mid-cycle dispatch was practitioner-sparked but agent-amplified.** Practitioner offered a brief observation; agent reconstructed the methodological failure in detail and proposed the reviewer dispatch. The self-correction machinery is more active in Cycle 3 than in Cycle 2 (where corrections were predominantly practitioner-originated). Recorded as a methodology-trajectory signal worth preserving.

### From RESEARCH (Spike C — three-arm architecture comparison on synthesized fixture, 004d)

13. **Concrete-verification result on cross-file drift (ISSUE-5): Arm B (cheap+ensemble with script-agent) 3/3, Arm A (cheap-bare) 1/3, Arm C (frontier-bare single-shot) 0/2.** The fixture is a synthesized 90-line module with five injected issues (off-by-one in check_limit; api_key in logger.warning; `limit: int = None` annotation; test deferral; cross-file value drift `DEFAULT_BUDGET_LIMIT = 100_000` vs `DEFAULT_MAX_TOKEN_LIMIT = 50_000_000`). Pre-spike fixture validation in `ground-truth.md` recorded per-issue expected detection difficulty before runs. Arm B's advantage is concentrated on ISSUE-5 (cross-file value drift); on the other four issues, all three arms perform comparably.

14. **The mechanism on ISSUE-5 is the script-agent's deterministic file access — not the orchestrator's ensemble-routing decision.** A guaranteed cross-file file read is what produced Arm B's advantage. A script that simply ran and reported (without `invoke_ensemble` orchestration) would plausibly have produced the same result. The susceptibility snapshot identifies "the architecture's `+ orchestration` primitive is load-bearing on this bug class" as exceeding what the evidence supports; the more directly supported framing is "deterministic tool output is the mechanism on cross-file verification." This distinction is Cycle 4 territory and is the snapshot's first grounding action.

15. **Architecture-level scope condition.** N=3 on a single synthesized fixture supports a fixture-level claim, not an architecture-level principle. The "deterministic-vs-probabilistic complementarity" frame appearing fully formed in Spike C's synthesis well-supports a real empirical pattern but exceeds single-fixture scope when read as a general architectural principle. Limitations sections name "single fixture" as item 1; headline framings do not propagate the scope qualifier with the same force.

### From RESEARCH (Spike D — multi-ensemble pilot, 004e)

16. **Pilot results: B1 = C1 = 4/5 resolved on the synthesized cross-file fixture; A1 3/5 (verifier false-positive on api_key hash, effectively 4/5 if security-property-judged).** Spike D's intent was to test multi-ensemble coordination via `llm-orc serve` autonomous routing as Cycle 4 priming.

17. **Opencode CLI stalls indefinitely on substantial code prompts at cheap-tier — repeatable across 3 sustained attempts; same MiniMax model via production model factory path completes Stage 2 in 24.8s.** The intended B1 arm (cheap orchestrator with autonomous routing intelligence) never ran as designed; a direct model factory script substituted. Spike D's "architecture works at multi-stage workflow level" finding rests on a manually staged pipeline that bypassed the orchestrator's actual `invoke_ensemble` autonomous primitive.

18. **The susceptibility snapshot identifies two embedded-conclusion risks in 004e.** "B1 ties C1 at zero $ cost" foregrounds a N=1 manually staged pilot tie as a cycle finding; the architecture's autonomous coordination primitive was not exercised. The snapshot's second grounding action: Cycle 4 entry should explicitly distinguish Spike D's manually staged pipeline from autonomous routing via `llm-orc serve`. The opencode CLI stall is a deployment-shape finding worth carrying forward; whether it is also a cycle finding in its own right or a sign of infrastructure noise that masked the intended pilot is a reading the synthesis presents as compatible but does not foreground.

### From RESEARCH (susceptibility snapshot — three grounding actions for Cycle 4 entry)

19. **Grounding action 1 (mechanism isolation):** Test whether the mechanism is "script's deterministic file access" vs "orchestrator's routing decision." Compare cheap-bare with script's output as input context (Spike A arm2 pattern applied to cross-file verification) vs cheap+ensemble dispatched via `invoke_ensemble`. If the two configurations match on concrete-verification, the lesson is "deterministic tool output is the mechanism," not "ensemble orchestration is the mechanism."

20. **Grounding action 2 (autonomous routing distinction):** Cycle 4 should name the evidence base for autonomous routing as Cycle 1's CAP-9 baseline + Spike C's single-stage evidence, and name the evidence gap as multi-stage autonomous coordination via `llm-orc serve`. Without this entry-point grounding, Cycle 4 risks inheriting Spike D's headline confidence markers and building against a foundation the evidence does not yet support.

21. **Grounding action 3 (frontier comparison baseline):** Name "frontier-bare single-shot has no access to other files" as a scope condition rather than a neutral reference point. Test what happens when frontier has matched information access (frontier + cross-file extraction script as input context). If the architecture's advantage disappears under matched access, the mechanism is "information access" not "architectural composition." If the advantage persists, the mechanism is genuinely compositional.

### From RESEARCH (commitment-gating outputs at gate)

22. **Settled premises (Cycle 4 builds on these):**
    - Spike A's cross-tier finding on the cycle-2 README fixture is genuine, scope-bounded to that fixture class.
    - Spike B's F1 vs F2 methodological finding is settled: F1 turn-by-turn dispatch is the correct method for multi-turn frontier-tier testing under the no-paid-API constraint.
    - Cross-tier complementarity does not replicate at multi-turn on the library-checkout fixture; task-class-dependent, not architectural.
    - Spike C's concrete-verification result on the synthesized cross-file fixture is genuine, scope-bounded to that fixture.
    - Free-options preference and spike-artifact retention are durable practitioner directives recorded in agent memory.

23. **Open questions (Cycle 4 inherits these):**
    - Mechanism isolation: script's deterministic file access vs orchestrator's routing decision (Grounding action 1).
    - Autonomous routing vs manual staging: the architecture's autonomous coordination primitive across multiple stages is unsupported by Cycle 3's evidence (Grounding action 2).
    - Frontier comparison baseline scope condition (Grounding action 3).
    - Whether the four-priorities frame (performance × environmental cost × local-first × token cost) survives a measured-divergence test, or remains rhetorical at the cycle's evidence resolution.

24. **Specific commitments carried forward to Cycle 4:**
    - Cycle 4 territory (named by practitioner at gate close): supported design methods for orchestrator + ensembles that can envision the right next steps and iterate closer to effective agentic design with llm-orc.
    - The three grounding actions enter as research-entry inheritance, not as Cycle 3 gaps.
    - The susceptibility snapshot's framing-adoption signal carries forward as discipline: name comparison baselines as scope conditions rather than neutral references in any cycle that contrasts the architecture against frontier-bare.
    - Spike artifacts (scratch/* directories, custom ensembles in `.llm-orc/ensembles/spike-c-*` and `.llm-orc/ensembles/spike-d-*`, supporting scripts in `.llm-orc/scripts/spike_*`) are retained until corpus close per practitioner directive.

## Context for Resumption

### Cycle close shape

Cycle 3 closed at research-phase end with Mode B (Research Only) declared at cycle close. Five research-log artifacts plus the audit trail are the cycle's deliverable; no Cycle 3 essay was produced. The decision to close at research without producing an essay reflects the cycle's findings: the cycle's late-cycle synthesis (Spike C and Spike D) crystallized framings the susceptibility snapshot identifies as exceeding the evidence scope, and the practitioner's gate response ("ground in supported design methods for orchestrator + ensembles") points toward a Cycle 4 entry that should resolve the mechanism-isolation and autonomous-routing questions before any cycle's evidence is essay-grade.

### Closing pointer to Cycle 4

Cycle 4 territory: **supported design methods for orchestrator + ensembles** that can envision the right next steps and iterate closer to effective agentic design with llm-orc. The three grounding actions from the susceptibility snapshot are research-entry inheritance. Cycle 4 should treat the comparison baseline scope condition, the script-vs-orchestration mechanism isolation, and the autonomous-routing-vs-manual-staging distinction as the load-bearing entry-point questions before any new architectural claim is advanced.

### Suggested fresh-session handoff prompt for Cycle 4

> Open Cycle 4 of the agentic-serving scoped corpus on supported design methods for orchestrator + ensembles. Cycle 3 closed 2026-05-01 at research-phase end (Mode B). Read `docs/agentic-serving/cycle-archive/cycle-3-agent-design-script-models-orchestrator.md` §Feed-Forward Signals for the inherited signal index (items #1–#24), the five research logs at `docs/agentic-serving/essays/research-logs/004a-` through `004e-`, and `docs/agentic-serving/housekeeping/audits/susceptibility-snapshot-cycle-3-research.md` for the susceptibility snapshot's three grounding-action recommendations. Begin with `/rdd-research`. The three grounding actions (mechanism isolation; autonomous routing distinction; frontier comparison baseline scope condition) should be addressed at research-entry framing before Cycle 4 advances any new architectural claim.

## Conformance Notes (carried forward from Cycle 3 cycle-status)

**Corpus is on RDD v0.8.5.** Cycle 3's audit corpus and gates follow ADR-070 housekeeping placement convention (`housekeeping/audits/`, `housekeeping/gates/`); ADR-085 `.rdd/` migration target applies but is deferred per the migration-window allowance.

**Deferred conformance items still carried forward** (low priority; pick up opportunistically):

- ADR Rejected Alternatives + Provenance Check sections — 11 ADRs lack discrete headers (alternatives discussed inline in Context). Format alignment for v0.8.5 ADR template; matters only when ADRs are re-audited.
- Value tensions phrasing — `product-discovery.md` §Value Tensions stated declaratively rather than as open questions per v0.8.5 discover template.
- Essay 001 framing-audit dispatch — `housekeeping/audits/argument-audit-001.md` is argument-only; v0.8.5 dispatches combine argument + framing audits.
- Field-guide path — currently at `docs/agentic-serving/field-guide.md`; canonical is `references/field-guide.md`. ORIENTATION links current location; navigability preserved.
- Scenarios cycle-acceptance-criteria table — top-of-file table or null-coverage one-line note required by v0.8.5 decide Step 4.
- Housekeeping placement (`docs/agentic-serving/housekeeping/` per ADR-070) — ADR-085 supersedes with `.rdd/` placement during the transition window.

**New deferred items from Cycle 3:**
- No new conformance items introduced in Cycle 3.
- Spike artifacts retention: scratch/spike-a-cycle3-*, scratch/spike-b-cycle3-*, scratch/spike-c-cycle3-*, scratch/spike-d-cycle3-* directories are retained per practitioner directive (overrides standard rdd-research delete-after-recording discipline). Custom ensembles `.llm-orc/ensembles/spike-c-code-review.yaml` and `.llm-orc/ensembles/spike-d-fix-verifier.yaml` plus supporting scripts `.llm-orc/scripts/spike_c_diff_analyzer.py` and `.llm-orc/scripts/spike_d_fix_verifier.py` are retained until corpus close.

## Cycle 3 artifact index

Research logs (the cycle's primary artifact):
- `essays/research-logs/004a-lit-review-agent-design.md`
- `essays/research-logs/004b-spike-a-cycle3.md`
- `essays/research-logs/004c-spike-b-cycle3.md`
- `essays/research-logs/004d-spike-c-cycle3.md`
- `essays/research-logs/004e-spike-d-pilot-cycle3.md`
- `essays/research-logs/004-agent-design-script-models-orchestrator.md` (running log archived at cycle close)

Audit trail:
- `housekeeping/audits/research-design-review-cycle-3.md` (entry-time methods review)
- `housekeeping/audits/research-design-review-cycle-3-mid.md` (mid-cycle methods review)
- `housekeeping/audits/susceptibility-snapshot-cycle-3-research.md` (phase-boundary snapshot)

Gate reflection note: `housekeeping/gates/cycle-3-research-gate.md`

Spike artifacts (retained until corpus close per practitioner directive):
- `scratch/spike-a-cycle3-*`
- `scratch/spike-b-cycle3-*`
- `scratch/spike-c-cycle3-architecture-comparison/`
- `scratch/spike-d-cycle3-multi-ensemble-pilot/`
- `.llm-orc/ensembles/spike-c-code-review.yaml`
- `.llm-orc/ensembles/spike-d-fix-verifier.yaml`
- `.llm-orc/scripts/spike_c_diff_analyzer.py`
- `.llm-orc/scripts/spike_d_fix_verifier.py`

# Spike B — Cycle 3 RQ-3 Multi-Turn Reliability Probe

**Date:** 2026-05-01
**Cycle:** 3 (agentic-serving scoped corpus)
**Operating frame:** Outcomes over an agentic session — agent shape is means.
**Central question (per practitioner reframe at Spike A):** Does orchestration + cheap orchestrator agent compete with a more expensive frontier model on multi-turn sustained work?
**Decision G3 (full scope):** three arms (cheap-bare, cheap-with-script, frontier) × two fixtures (tau-shape multi-turn library checkout, real-session haiku-generator authoring).

---

## Abstract

Spike B probed RQ-3 across two fixtures and three arms to test whether multi-turn reliability ceilings from the literature (tau-bench's 34–74% pass@1; HORIZON's 19% meltdown rate) manifest at the cycle's deployment configurations, and whether Spike A's cross-tier-uncorrelated-errors finding extends to multi-turn. The primary substantive finding: **on the cycle's tested complexity, neither tier hits a reliability ceiling** — cheap-tier (MiniMax M2.5 Free) and frontier-tier (Sonnet 4.6 via subagent dispatch) both handle the protocol cleanly. On tau-shape: cheap-tier 12/12 graded success across 4 trials (post-fixture-fix), frontier-tier 6/6 via F1 turn-by-turn dispatch. On real-session: 6/6 trials produced working `haiku-generator` ensembles with valid 5-7-5 syllable-counter scripts. The cross-tier complementarity from Spike A does NOT replicate at multi-turn — both tiers reach equivalent outcomes rather than finding different things. A substantive methodological finding emerged from the F1-vs-F2 comparison: F2 single-shot facsimile testing broke down via imagined-state bias (5/6 frontier traces imagined fines that didn't exist in the fixture), while F1 turn-by-turn dispatch with deterministic tool feedback produced clean reactivity. F1's success rate matched cheap-tier's; F2's strict-replay outcome was 1/6 (5/6 traces had imagined data that didn't match the actual fixture). Spike B's RQ-3 result on the cycle's tested complexity is favorable to cheap-orchestration competing with frontier on multi-turn. Scope conditions are load-bearing: both fixtures are bounded; tau-bench's harder regimes (dual-control, longer trajectories) are not represented; the small sample (N=2 per arm per fixture) limits variance estimation.

---

## Background

Cycle 2 essay 003 named tau-bench multi-turn reliability as the primary inheritance for Cycle 3 (item #22 in the archive's Feed-Forward Signals). Cycle 3's lit review (`004a-lit-review-agent-design.md`) confirmed the published baseline: GPT-4o under 50% pass@1 on tau-bench's tool-agent-user multi-turn interaction tasks; τ²-Bench (Sierra Research, June 2025) extended the regime to dual-control and saw pass@1 drop further to 34%; HORIZON, AMA-Bench, LongCLI-Bench, and Khanal et al. characterize meltdown (up to 19% on long-horizon), memory retrieval drift, premature stop, and error self-conditioning as the primary failure modes. RQ-3 was reformulated per research-design-review-cycle-3 P2 #3 to name multiple failure modes explicitly (not only meltdown), and Decision A (PAE-aware procedural independence per Cao et al. arXiv:2603.03116) and Decision B (CLEAR's cost-normalized accuracy per Rony et al. arXiv:2511.14136) were adopted as evaluation refinements.

Spike A (essay log `004b-spike-a-cycle3.md`) was a single-ask fixture (documentation review). Spike B is the multi-turn fixture. Spike A produced a cross-tier-uncorrelated-errors finding: cheap-tier-with-directed-prompting and frontier-tier-direct found different and largely uncorrelated bug classes on the README-review fixture. Spike B's question is whether this complementarity extends to multi-turn agentic-coding work.

The mid-cycle reframe (recorded in memory: `cycle-3-central-question`) sharpened the central question to "does cheap-orchestrator + orchestration compete with a more expensive frontier model?" Spike B at G3 was designed to test this on multi-turn directly.

---

## Method

### Fixture 1: Tau-shape (single-control multi-turn library checkout)

Custom fixture closer to tau-bench's regime than to llm-orc's actual task class — three scenarios (book available, book checked out, patron has fines), six deterministic Python tools (`search_catalog`, `check_patron_status`, `check_out`, `place_hold`, `pay_fine`, `send_response`), and per-scenario programmatic grading (correct action + send_response). Failure-mode detectors per Decision A: meltdown onset (repeated tool calls), premature stop (no `send_response`), error self-conditioning (repeating failing call), memory drift, early stall (no `search_catalog` in 2 turns), wrong-action escalation (checkout when fines outstanding).

**Cheap-tier arms** ran via llm-orc's production `OpenAICompatibleModel.generate_with_tools` against `orchestrator-minimax-m25-free` profile. Custom multi-turn loop dispatches deterministic tools between turns. N=2 trials × 2 arms × 3 scenarios = 12 scenario runs.

**Frontier arm** ran via F1 turn-by-turn dispatch — each turn = one fresh subagent dispatch (Sonnet 4.6 via Claude Code Agent tool, `general-purpose` subagent type, `model: sonnet`). Subagent receives conversation-so-far + tool definitions; outputs ONE turn's JSON-structured response (assistant content + tool calls); harness parses JSON, executes tools deterministically, appends results to conversation, dispatches next turn. ~18 dispatches across 6 contexts × ~3 turns each.

**Methodological probe — F2 facsimile:** Initially attempted single-dispatch facsimile (one subagent per scenario × trial; 6 dispatches; subagent reasons through full multi-turn dialogue in one shot with imagined tool results). Documented the imagined-state bias finding before switching to F1. Both data sets retained.

### Fixture 2: Real-session (haiku-generator ensemble authoring)

Authentic multi-turn agentic-coding task: read existing ensemble examples (`.llm-orc/ensembles/testing/test-script-agents.yaml`, `.llm-orc/scripts/aggregator.py`), author a new `haiku-generator` ensemble YAML + a deterministic `haiku_syllables.py` script-agent that validates 5-7-5 syllable counts, hand-test with known valid + invalid haiku.

**Frontier arm:** Subagent dispatched with full Read/Edit/Write/Bash tools natively. N=2 trials. Authentic multi-turn behavior — subagent runs the full task internally with real tool dispatching.

**Cheap-tier arms:** Real CAP-9 deployment shape — `llm-orc serve` running on port 8765, `opencode run -m llm-orc/orchestrator-minimax-m25-free` providing OpenCode's tool surface (Read, Edit, Bash, Write) to the MiniMax orchestrator. The orchestrator drives multi-turn tool dispatching through OpenCode against the real codebase.

**cheap-with-script arm** prepends a deterministic preprocessor's output to the task description: structured summary of canonical ensemble + script reference paths, schema conventions, script-agent contract details. Parallels Spike A arm 2's "A2 + script input" pattern at multi-turn — script context as deterministic anchor for the LLM orchestrator.

**Trial outputs** captured at `scratch/spike-b-cycle3-multi-turn/real-session/<arm>-trial<n>/` — both files plus opencode stdout log.

### Decision A operationalization (PAE-aware procedural independence)

For tau-shape: programmatic grading checks both issue-level outcome (correct tool-call sequence given scenario state) and protocol independence (no repeated failing calls; `send_response` reached). For real-session: post-hoc verification — files exist; syllable script runs deterministically on valid + invalid haiku and produces correct `valid: true/false` results. Procedural independence checked by inspection of the script's logic (independent dependency-extraction + counting + comparison, not restating prompt text).

### Decision B operationalization (CNA — cost-normalized accuracy)

Per-trial token consumption recorded where reported. Cost in dollars: $0 across cheap-tier MiniMax calls (free tier; `cost_per_token: 0.0` in profile). Frontier-tier facsimile via subagent: subscription tokens consumed but not metered.

---

## Results

### Tau-shape per-arm summary

| Arm | Tier | n successful | Median wall-clock | Failure modes detected |
|-----|------|-------------:|------------------:|:----------------------|
| cheap-bare (post-fix) | Cheap | 6/6 | 11–22 s/scenario | None |
| cheap-with-script (post-fix) | Cheap | 6/6 | 9–16 s/scenario | None |
| frontier (F1 turn-by-turn dispatch) | Frontier | 6/6 | ~3–5 s per dispatch (~10 s/scenario total) | None |
| frontier (F2 facsimile, methodological probe) | Frontier | 1/6 strict replay; 6/6 plan-correct vs imagined | n/a (single-shot) | Imagined-state bias documented |

**The post-fixture-fix cheap-tier results are clean.** The first cheap-tier run had 4/12 graded failures, all attributable to fixture defects (search_catalog's naive substring matching failed multi-word queries) or system-prompt ambiguity (place-hold-vs-ask preference). After fixing the search algorithm to tokenize and tightening the system prompt's "act, do not ask" instruction, both cheap arms scored 6/6 each. The first run's trials are preserved at `trials/run-01-with-fixture-bug/` for record per the spike-retention policy.

**Frontier F1's 6/6** demonstrates that single-shot subagent dispatches with deterministic tool feedback between turns produces clean multi-turn reactivity testing. F2 facsimile's 1/6 strict replay reflects an imagined-state bias: 5 of 6 traces imagined fine balances that didn't exist in the actual fixture, then planned for that imagined scenario. The plans were protocol-correct given the imagined data, but the data wasn't real.

### Real-session per-trial summary

| Trial | Arm | Files created | Syllable script verified | Wall-clock | Notes |
|-------|-----|--------------|--------------------------|-----------:|-------|
| frontier-trial1 | Frontier | yaml + py | Valid + invalid both correct | ~10 min | 28 tool calls; discovered `--config-dir` flag avoids file-copy step; ensemble invoked end-to-end; substantive reflection |
| frontier-trial2 | Frontier | yaml + py | Valid + invalid both correct | ~4 min | 18 tool calls; copied to `.llm-orc/ensembles/` for invoke testing; substantive reflection |
| cheap-bare-trial1 | Cheap | yaml + py | Valid + invalid both correct | ~14 min | Terse output; minimum viable artifacts |
| cheap-bare-trial2 | Cheap | yaml + py | Valid + invalid both correct | ~3 min | Faster on second attempt; same approach |
| cheap-with-script-trial1 | Cheap+script | yaml + py | Valid + invalid both correct | ~2 min | With preprocessor context — fastest cheap-tier trial |
| cheap-with-script-trial2 | Cheap+script | yaml + py | Valid + invalid both correct | ~2 min | Same speed; agent named LLM agent `haiku_writer` (underscore); script + YAML internally consistent |

**All 6 real-session trials produced working artifacts.** Each `haiku_syllables.py` correctly returns `valid: true` for a canonical 5-7-5 haiku and `valid: false` for a deliberately broken one when given the matching dependency key. Each `haiku-generator.yaml` declares the two-agent structure (LLM + script) with appropriate `depends_on` wiring.

**Cheap-with-script wall-clock advantage is suggestive but small-sample.** Both cheap-with-script trials completed in ~2 min vs cheap-bare's 3–14 min range. With N=2 the comparison is noisy, but directionally: prepended deterministic context appears to reduce initial-discovery overhead in the multi-turn agentic-coding flow. This is a tentative finding — Cycle 4 hook for measurement at higher N.

**Cross-tier output style differs but outcomes don't.** Frontier subagents produced detailed multi-paragraph reflections including 6-section structured summaries (turn count, files, test results, invoke results, failure modes, workflow reflection). Cheap-tier OpenCode default output produced terse 5–10 line summaries. Both reached working artifacts; the verbosity difference is output-style, not capability.

### Cross-arm patterns

**Pattern 1 — Multi-turn reliability ceilings did NOT manifest at the cycle's tested complexity.** Tau-bench's 34–74% pass@1 (single + dual control) does not appear at single-control multi-turn library checkout. HORIZON's meltdown-rate finding (19% on long-horizon) does not appear at 2–3-turn scenarios. The tested fixtures are bounded; the lit-review's reliability findings are scoped to longer-horizon and dual-control regimes the cycle's spikes do not reach.

**Pattern 2 — Cross-tier complementarity from Spike A does NOT replicate at multi-turn.** On Spike A's documentation-review fixture, cheap-with-direction caught semantic bugs frontier missed (and vice versa). On Spike B's tau-shape and real-session, both tiers reach equivalent outcomes — the same correct actions, the same working ensembles + scripts. **The complementarity is task-class-dependent, not universal.** Spike A's mechanism (heterogeneity-uncorrelated-errors across tier-and-capability) requires bug classes where the tiers have different blind spots; multi-turn protocol navigation and agentic-coding-task execution have less of that asymmetry on these fixtures.

**Pattern 3 — F2 facsimile is structurally limited; F1 dispatch is the right method for multi-turn reactivity testing.** The imagined-state bias finding is substantive: when asked to reason single-shot through multi-turn dialogue, Sonnet 4.6 imagined the most policy-interesting case (fines blocking checkout) regardless of the actual scenario. Plans were protocol-correct given imagined data. F1 turn-by-turn dispatch with real tool feedback produced clean reactivity. **For RQ-3-style reliability testing, F1 is the methodologically correct approach.** F2 measures planning capability under self-imagined conditions; that is a different question.

**Pattern 4 — The four-priorities frame strongly favors cheap-tier on Spike B's results.** Cost: cheap-tier $0 (free); frontier-tier subscription-metered. Performance: equivalent on these fixtures. Environmental cost: cheap-tier MiniMax inference at OpenCode Zen vs frontier-tier Sonnet inference — both are cloud, but per-call cost-and-energy is lower for the smaller model. Local-first preference: equivalent (both cloud here, but cheap-tier scales to local-deployment with qwen3:8b per Cycle 1's CAP path). On Spike B, the four-priorities frame would unambiguously recommend cheap-tier; performance-only frame would be indifferent (both tiers pass). **On these fixtures, the four-priorities frame again produces a frame-divergent recommendation versus performance-only.** Spike A's RQ-2 finding holds and extends.

**Pattern 5 — Cheap-with-script preprocessing shows tentative wall-clock advantage on real-session.** Both cheap-with-script trials completed in ~2 min vs cheap-bare's 3–14 min range. Sample is too small for statistical confidence (N=2 per arm) but direction is consistent: prepended deterministic context appears to reduce initial-discovery overhead. Tentative finding; Cycle 4 hook.

---

## Discussion

### RQ-3 result on the cycle's tested complexity

The cycle's spike battery does NOT reach tau-bench's harder regimes (dual-control; longer trajectories) or LongCLI-Bench's long-horizon failures. On the regimes Spike B does test — single-control short-horizon multi-turn protocol navigation, and bounded agentic-coding authoring — both cheap-tier (MiniMax M2.5 Free) and frontier-tier (Sonnet 4.6) handle the work cleanly. The reformulated RQ-3 ("what agent-design choices reduce the rate of observable failure modes") gets a directional answer: at this complexity, both tiers achieve zero observable failure modes; at higher complexity, the lit-review predicts both tiers degrade and the design-choice question becomes binding.

This is a **regime-bounded finding**. The cycle's evidence does not refute the lit-review's reliability findings; it adds that **at sub-tau-bench complexity, the regime is below the binding constraint** for both tiers. The boundary between "regime works fine" and "regime hits the reliability ceiling" is not characterized by Spike B's data — that's a Cycle 4 question.

### Cross-tier complementarity — task-class-dependent

Spike A (documentation review) found cross-tier-uncorrelated-errors. Spike B (multi-turn protocol + agentic-coding authoring) finds equivalent outcomes. The contrast is informative: **the heterogeneity-uncorrelated-errors mechanism (Sun et al. 2025; Ding et al. 2024) requires task classes where the tiers have meaningfully different blind spots**. Documentation review has these (semantic-consistency-across-sections vs. architectural-design-issues are uncorrelated). Multi-turn protocol navigation and agentic coding don't have these on these fixtures — both tiers handle the dispatch protocol correctly.

This is a **scope refinement on Spike A's finding**: cross-tier complementarity is empirical-task-class-dependent. It is not a universal property of cheap+frontier combinations. The Cycle 4 question becomes: which task classes have the asymmetry that produces complementarity, and which don't?

### F1-vs-F2 methodological finding

The shift from F2 (single-dispatch facsimile) to F1 (turn-by-turn dispatch) was driven by data. F2's imagined-state bias was unanticipated but is structurally explainable: a model asked to reason single-shot through a multi-turn dialogue with an instruction to "imagine plausible tool results" will pick the most policy-interesting plausible scenario, then plan for that. The model has no way to know the actual scenario. F1 dispatch with deterministic tool feedback between turns provides the actual scenario at each turn; the model reacts to real data.

For multi-turn reliability testing, **F1 is methodologically correct; F2 measures something different (planning capability under imagined conditions)**. The cycle's RQ-3 evidence is the F1 data; F2 is preserved as a methodological observation.

### Central question — cheap competes with frontier on Spike B's regimes

Spike A's reframe ("does cheap-orchestrator + orchestration compete with a more expensive frontier model?") gets a direct answer on Spike B's data: **yes, on the cycle's tested multi-turn complexity, cheap-orchestration matches frontier on outcome quality and dominates on the four-priorities frame** (zero $ cost; equivalent performance; lower environmental cost; same local-first posture). Performance-only frame is indifferent (both tiers pass); four-priorities frame strongly recommends cheap-tier. Frame-divergent recommendation per Decision B's CLEAR-style operationalization.

Combined with Spike A's evidence (cross-tier complementarity for documentation review), the cycle produces a more nuanced picture than "cheap-vs-frontier" alone would suggest:

- **For agentic protocol navigation and agentic coding (Spike B):** cheap-tier is sufficient at this complexity; cross-tier combination doesn't add value.
- **For semantic-consistency analysis (Spike A):** cross-tier complementarity adds coverage; either tier alone misses bug classes the other catches.

The architectural lesson is task-class-conditional, not universal. The four-priorities frame is consistently load-bearing across both findings — cost-axis recommendations differ from performance-only in both Spike A (favor complementary multi-tier) and Spike B (favor cheap-tier alone).

---

## Limitations and scope conditions

1. **Bounded fixtures.** Tau-shape: 2–3-turn scenarios, single-control, deterministic tools. Real-session: bounded authoring task with clear success criteria. Tau-bench's harder regimes (dual-control, longer trajectories) and HORIZON's long-horizon scope are NOT represented. The cycle's RQ-3 evidence is regime-scoped.

2. **Small sample.** N=2 per arm per fixture. Detects success/fail rate at this resolution but not narrow effect sizes. The cheap-with-script wall-clock advantage on real-session is at this resolution — directional, not statistically established.

3. **Cheap-tier output opacity.** OpenCode's default output is terse summary. Cannot directly compare tool-call counts cheap-vs-frontier. The internal multi-turn behavior of cheap-tier is observable through the resulting artifacts but not through a turn-by-turn transcript at the same fidelity as frontier subagents.

4. **F2 facsimile preserved for methodological note, not as RQ-3 evidence.** The F2 imagined-state bias finding is informative; the F2 trial data is not used as RQ-3 reliability evidence.

5. **Initial fixture defect.** The first cheap-tier tau-shape run had 4/12 graded failures attributable to a search-algorithm bug + system-prompt ambiguity. After the fix, results were 12/12 clean. The first-run data is preserved at `trials/run-01-with-fixture-bug/` for record. Fixture quality affected the initial signal.

6. **Real-session task complexity.** The haiku-generator authoring task is bounded (~5–10 turns, two files, well-specified success). More complex agentic-coding tasks (multi-file refactor, debugging across modules, long sessions) are NOT represented. Cycle 4 hook.

7. **MiniMax M2.5 Free is one cheap-tier configuration.** Other cheap-tier choices (qwen3:8b local; smaller cloud models) may behave differently. The cycle's evidence is for this specific cheap-tier; generalization to other cheap-tier deployments requires additional measurement.

---

## Implications for Cycle 3's RQs

**RQ-3 — cycle's tested multi-turn complexity does NOT manifest the lit-review's reliability ceilings.** Both cheap and frontier tiers achieve zero observable failure modes on tau-shape and real-session at this fixture's complexity. The lit-review's tau-bench (34–74%), HORIZON (19% meltdown), LongCLI-Bench (<20% pass) findings remain credible at their measured regimes; Spike B does not reach those regimes. RQ-3 is answered for sub-tau-bench complexity; the harder regimes carry to Cycle 4.

**RQ-2 — four-priorities frame produces frame-divergent recommendation again.** Performance-only frame indifferent on Spike B; four-priorities frame strongly favors cheap-tier. Combined with Spike A's frame-divergence finding, the four-priorities frame is **load-bearing** at Cycle 3's resolution across both fixtures. The Cycle 3 falsification criterion (per Decision B): the frame is not retired; it has empirical support across two independent fixtures.

**Cross-tier complementarity (Spike A finding) — task-class-dependent.** Spike A's evidence holds for documentation review; Spike B finds equivalent outcomes for multi-turn protocol + agentic coding. The mechanism is task-class-conditional. Cycle 4 question: which task classes have cross-tier asymmetry?

---

## Cycle 4 hooks

1. **Multi-turn reliability at higher complexity.** The cycle's evidence is bounded to ≤10-turn tasks. Tau-bench's harder regimes, LongCLI-Bench's long-horizon, and dual-control scenarios remain Cycle 4 territory. The boundary between "regime works fine" and "regime hits ceiling" is not characterized by Spike B's data.

2. **Cheap-with-script preprocessing wall-clock advantage at higher N.** Spike B's tentative finding (~2 min cheap-with-script vs 3–14 min cheap-bare) is at N=2. Cycle 4 should test at higher N to establish whether the preprocessing advantage is real or sampling variance.

3. **Cross-tier complementarity task-class survey.** Spike A finding holds for doc review; Spike B finding holds for protocol + coding. Cycle 4 should map task classes onto whether cross-tier complementarity exists. Hypotheses: documentation review (yes); code review proper (?); architecture review (?); refactoring with verification (?); debugging across modules (?); long-form writing (?).

4. **F1 dispatch as a cycle-standard methodology.** F1 turn-by-turn dispatch is now the cycle's documented method for multi-turn frontier-tier testing under the no-paid-API constraint. F2 facsimile is documented as structurally limited. Future spikes that need frontier-tier multi-turn evidence should default to F1.

5. **Floor question (Spike A P3 + Cycle 4 inheritance).** "What value does any LLM contribute once the script has run?" — Spike B's cheap-with-script preprocessing prepends deterministic context but the LLM still does authoring. The deeper question of whether the LLM is necessary for some bounded task classes (e.g., script generates ensemble template; LLM only fills in topic-specific text) is Cycle 4 territory.

6. **Local-tier multi-turn.** Spike B's cheap-tier was MiniMax M2.5 Free via OpenCode Zen (cloud, free tier). Local-tier multi-turn (qwen3:8b via local Ollama) was not tested. Cycle 4 should probe whether local-tier handles multi-turn agentic coding; this is the genuine local-first scenario.

7. **τ²-Bench dual-control regime.** The cycle's tau-shape fixture is single-control (agent only); the published baseline measured dual-control as the harder regime. Cycle 4 should design a dual-control fixture to test whether the lit-review's 34% pass@1 manifests at the cycle's deployment.

---

## Spike code disposition

**Retain until corpus close** per practitioner policy (memory: `feedback_spike_artifact_retention.md`).

Spike B preserved artifacts:
- `scratch/spike-b-cycle3-multi-turn/library_catalog.py` — tau-shape fixture state + tools + grading
- `scratch/spike-b-cycle3-multi-turn/tau_shape_harness.py` — cheap-tier multi-turn loop using llm-orc's production OpenAICompatibleModel.generate_with_tools
- `scratch/spike-b-cycle3-multi-turn/f1_runner.py` — frontier-tier turn-by-turn dispatch state manager
- `scratch/spike-b-cycle3-multi-turn/trials/` — all per-trial JSON outputs (cheap-tier programmatic; frontier F1 contexts; F2 traces)
- `scratch/spike-b-cycle3-multi-turn/trials/f1-contexts-run-bug/` — first-run contexts before JSON-key serialization fix; preserved for bug-investigation record
- `scratch/spike-b-cycle3-multi-turn/trials/run-01-with-fixture-bug/` — first-run cheap-tier trials before fixture fix; preserved per retention policy
- `scratch/spike-b-cycle3-multi-turn/real-session/` — six trial directories (frontier ×2, cheap-bare ×2, cheap-with-script ×2) each with `haiku-generator.yaml` + `haiku_syllables.py`
- `scratch/spike-b-cycle3-multi-turn/real-session/preprocessor.py` — deterministic preprocessor for cheap-with-script arm
- `/tmp/cheap-*-trial*-output.log` — opencode stdout transcripts (will be moved into the scratch directory before corpus close to ensure preservation)

---

## Connections to Cycle 3's other research

- **`004a-lit-review-agent-design.md`** — tau-bench, τ²-Bench, HORIZON, AMA-Bench, LongCLI-Bench, MAST taxonomy, AgentEval, Routine, Compiled AI. Spike B's RQ-3 result is regime-scoped relative to these findings; the cycle's evidence is for sub-tau-bench complexity.
- **`004b-spike-a-cycle3.md`** — Spike A's cross-tier-uncorrelated-errors finding. Spike B refines: complementarity is task-class-dependent, not universal.
- **Cycle 3 cycle-status (`housekeeping/cycle-status.md`)** — RQ-3 reformulation per P2 #3 (failure-mode multiplicity), Decision A and B operationalization, G3 scope.
- **Memory: `cycle-3-central-question`** — central question reframe documented mid-Spike A. Spike B's evidence on Spike B's regimes: cheap competes (matches) frontier; four-priorities frame strongly recommends cheap-tier.
- **Memory: `feedback_spike_artifact_retention`** — retention policy for spike artifacts; applied to Spike B preservation.
- **Memory: `feedback_free_options_preference`** — free-tier preference; preserved through Spike B (zero $ API cost; subscription tokens for frontier facsimile only).

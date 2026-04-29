# Research Log: Agentic Serving — Capability Floor and Observability

*Started: 2026-04-27*

This log captures a backward research loop from PLAY in the agentic-serving scoped cycle. The PLAY phase (closed 2026-04-25) surfaced two next-cycle fundamentals — bilateral observability gap and undefined orchestrator-capable Model Profile capability floor — alongside hallucination-burn as a failure mode distinct from runaway loop. Field notes route those findings to a research loop before further BUILD work commits.

The research entry was preceded by Kent Beck's blog post "Genie Lessons: Nobody Wants Agents" (https://tidyfirst.substack.com/p/genie-lessons-nobody-wants-agents), which reframes the discussion from feature (multi-agent architecture) to outcome (declarative goal specification with honest signaling about feasibility). That framing is load-bearing for RQ-3 and RQ-5 below.

## Research Questions

The following questions were articulated by the practitioner before the corpus was re-read for this research entry, per the question-isolation entry protocol (ADR-082):

### RQ-1 — Capability floor and affordances (empirical)

What is the empirical capability floor for an orchestrator driving llm-orc's cascading-tool model, and how does it vary with (a) model class, (b) tool-surface composition (which tools are exposed and how they are described), and (c) ensemble affordances (what ensembles declare about themselves to the orchestrator and what composability primitives the orchestrator is given)?

The question admits the tool surface as variable rather than fixed — capability is treated as a property of the (model × surface × affordances) tuple, not the model alone. ADR-003 (the fixed five-tool surface) and the current ensemble-to-orchestrator interface are in scope as candidate variables. Amended on 2026-04-27 to make the affordance dimension explicit — the constraint-removal response surfaced "ensemble affordances and composability" as a potential gap the original phrasing under-named.

### RQ-2 — Available signals, their limits, and honest-default alternatives

What signals — if any — are available at pre-session, session-start, or first-turn to distinguish a profile that meets the floor from one that doesn't, and what are the limits of those signals? Where signals are unreliable, what honest-default alternatives are available (e.g., explicit operator attestation, capability gates that refuse to start an incompetent session)?

Amended on 2026-04-27 per methods-reviewer P2 finding. The original phrasing ("reliably distinguish") presupposed reliable detection is achievable at these time horizons. Self-knowledge and calibration literature for small models may come back negative; the question phrasing should preserve that openness rather than foreclose the possibility that the answer is "not reliably distinguishable; the system should require explicit capability attestation instead."

### RQ-3 — Observability surfaces, separability, and capability-gate alternatives

Under what conditions are diagnostic truthfulness and coordination burden separable design goals, and what observability surface designs serve each goal without collapsing them — including the case where they cannot be separated without a capability gate upstream?

Amended on 2026-04-27 per methods-reviewer P1 finding. The original phrasing ("without imposing coordination burden") presupposed the two goals are reconcilable through surface design. Field notes (note 8) explicitly leave this open: "Presumably a better or more competent configuration would not have led me to want more observability." That suggests the correct resolution may be upstream — a capability gate that prevents the failing-session experience — rather than a surface design that delivers diagnostic truth without becoming a coordination display. The amended question admits both possibilities.

The Beck framing remains relevant: visibility-as-coordination-display ("watch which agent is doing what so you know when to interrupt") is the failure mode Beck names; visibility-as-diagnostic-truthfulness is the alternative. The amendment does not abandon the framing — it admits that the framing's clean resolution may not survive the empirical encounter with a low-capability profile.

**Coverage note (per methods-reviewer):** the in-stream / server-side split frames observability for tool user vs. operator. A third candidate surface — session-start or first-turn narration that fires *before* any composition events are generated, specifically to carry signal when the session is about to be degraded — is in scope for the OB-2/OB-3 spike work and should not be foreclosed by the question phrasing.

### RQ-4 — Intervention level for honest first-session experience

What is the minimum intervention that makes the OpenCode-as-default-client first session honest about its own competence — and is that intervention best located at *configuration* (defaults), at *capability detection* (gates that refuse to start an incompetent session), or at *operator policy* (explicit profile attestation)?

Amended on 2026-04-27 per methods-reviewer P1 finding. The original phrasing ("what default configuration ships") presupposed defaults are the right intervention level and buried the policy question in a parenthetical. The PLAY findings challenge both presuppositions directly: no default configuration resolves a session where the orchestrator profile is fundamentally incapable. The policy question — *whether* defaults are the right mechanism — is the more fundamental one and is now foregrounded. The design question (what defaults look like, if they ship) follows the policy answer.

### RQ-5 — Dual-contract divergence, convergence, and seam navigation (framing question)

Do llm-orc's dual contracts — *user-facing outcome* ("you get good results or you know clearly why not") and *project-facing method* (the hypothesis that orchestration with non-frontier models can deliver good results) — produce different design choices in the areas RQ-1 through RQ-4 investigate, and if so, how is the seam navigated? If they converge in those areas, what does that convergence reveal about the relationship between the two contracts?

Amended twice on 2026-04-27. First amendment (from constraint-removal): reframed from outcome-vs-feature to dual-contract reconciliation, recognizing that the project contract is method-shaped (the hypothesis under test) while the user contract is outcome-shaped (Beck-aligned). Second amendment (per methods-reviewer P2 finding): the first amendment's "where the two contracts produce different design choices" presupposed divergence before the research located it. The current phrasing admits both divergence and convergence as empirical possibilities, and asks what each reveals.

This is a framing question, not an empirical question. It runs throughout the research rather than being answered by a single spike or literature pass.

## Constraint-Removal Response

**Artifact bracketed:** ADR-003 (the fixed five-tool orchestrator surface).

**Practitioner response (2026-04-27):**

The specific five tools are not load-bearing. The set may be incomplete (ensemble affordances and composability properties may be under-exposed); some tools may not be needed in their current form. What is structurally load-bearing — what must remain true regardless of which tools end up in the surface — is that there exists *a capable orchestrator with a declared set of tools it can cascade requests through*.

This connects to Beck's outcome-vs-feature distinction but does not collapse to it. The user's contract is outcome-shaped: they care about getting good results. The project's contract is method-shaped: the hypothesis under test is that orchestration with non-frontier models can deliver good results. Both contracts are simultaneously load-bearing — the project would not exist if outcome were the only thing that mattered (a frontier-model passthrough would suffice), and the project fails its own hypothesis if outcome is not delivered.

Implication for the research framing: capability and affordances are method-layer questions (does the system deliver on its hypothesis?); default-configuration honesty is user-layer (does the user get good results, or know clearly that they cannot?). The two layers must not be conflated.

**What this surfaces beyond the original RQs:** ensemble affordances and composability are themselves a potential gap that the original RQ set did not name. RQ-1's "tool-surface design" wording admits the count and shape of tools as variable but does not directly name the affordances each tool exposes (e.g., what ensembles can declare about themselves to the orchestrator, what composability primitives the orchestrator is given, what observability the ensemble layer surfaces back to the orchestrator).

## Research Plan

### Sequencing

0. **Pre-spike: token-cap reconciliation** (P3 finding from methods review). Resolve FF #131 (encountered 50K cap vs documented 10M default) before S0 fires. An S0 run with a misconfigured cap produces uninterpretable results. Likely a local `~/.config/llm-orc/config.yaml` override; confirm by inspection and either correct or document the cap S0 will run with.

1. **S0 — Baseline observation spike** (single afternoon, isolated worktree). Configure llm-orc with a known-capable orchestrator profile (claude-sonnet or gpt-4o via API). Add minimal request/response/tool-dispatch logging to `serve` (smallest viable surface). Point OpenCode at it. Run the same shape of request from the PLAY session. Observe four surfaces: OpenCode TUI, SSE traffic on the wire, server log, post-session state.

   **Purpose:** distinguish three hypotheses (the H0/H1/H2 partition replaces the original H1/H2 partition per methods-reviewer P1 finding):

   - **H0 — Configuration-only.** The first-session failure is entirely configuration-dependent. A strong orchestrator profile produces a working session *without any of the cascade architecture doing meaningful work* (e.g., the orchestrator answers directly without invoking tools, or invokes one tool trivially and answers). Under H0, the user contract is satisfied by configuration alone; the project contract (the hypothesis that orchestration with non-frontier models can deliver good results) is *not validated* by S0 because the strong profile bypassed the orchestration. RQ-2 through RQ-4 become primarily method-contract questions under H0.
   - **H1 — Capability-bound, architecture engaged.** A strong profile produces a working session *and the cascade architecture does meaningful work* (orchestrator invokes ensembles, calibration fires, summarization triggers, etc.). Under H1, the architecture is validated as designed; the floor question becomes "where does capability cross the threshold needed for the cascade to function?" The original CAP-1 (narrowed profile battery) and OB-2/OB-3 spikes follow.
   - **H2 — Architecture-shape problems.** Even with a strong profile, the system has shape problems that prevent outcome delivery (coordination burden in Beck's sense, surface-design fighting purpose, operator-facing silence regardless of capability, affordance gaps the orchestrator hits and cannot work around). Under H2, RQ-5 (framing coherence) moves to center; the affordance-variation spike and tool-surface-as-variable spike (CAP-3) become load-bearing; ADR-001/002 may warrant a future constraint-removal pass.

   **Predictions for each hypothesis written down before running** — recorded in this log before the spike fires. **Both contract layers observed:** did the user get a good result (user contract); what did the orchestrator actually do and what does it suggest about the cascading-tool model's viability (project contract). **Branch point at S0 close:** for each RQ-2/RQ-3/RQ-4, ask whether it is still load-bearing under the observed hypothesis, and for which contract layer. The seam RQ-5 is meant to navigate is exactly this branch point — recording the answers makes the navigation visible.

2. **Literature scan in parallel with S0:**
   - Beck, "Genie Lessons: Nobody Wants Agents" (primary — 2026 substack post). Read the prior Genie posts (Lessons #1-#4) for surrounding context.
   - Tool-use capability evaluation: Berkeley Function-Calling Leaderboard (BFCL), ToolBench, tau-bench, AgentBench. What do they measure, how do they measure it, what do they tell us about the model-class boundary for reliable tool use?
   - Self-knowledge and calibration in small models: does a model know when it cannot do the task? What does the literature say about hallucination as a function of capability ceiling?
   - Observability patterns in shipped agentic systems: OpenHands, Claude Code, Cursor, Aider. What surfaces do they expose? What do their users see? Where does each system land on the visibility-vs-burden tradeoff?
   - Ensemble affordance design (in service of RQ-1's amended affordance dimension): what affordances do agentic frameworks expose for sub-agent composition? LangGraph, AutoGen, CrewAI, OpenHands, the recent literature.

3. **Subsequent spikes gated on S0 outcome.** Sequencing depends on which hypothesis (H0/H1/H2) S0 surfaces:
   - **If H0 (configuration-only)**: scope shrinks substantially for the user contract. RQ-4 becomes a config-hardening question (ship a default that meets the floor, or refuse to start without operator attestation). RQ-2 and RQ-3 become method-contract questions investigated against scenarios where the cascade architecture is *required* (deliberately constructed tasks that exceed a strong profile's direct-answer capacity). The project hypothesis (non-frontier orchestration delivers good results) becomes the load-bearing test, and the spike battery narrows to validate it.
   - **If H1 (capability-bound, architecture engaged)**: scope is moderate. Run a narrowed CAP-1 (profile battery: 2-3 profiles spanning the floor) + OB-2/OB-3 (server-log surface defaults + session-start narration) + RQ-2 signal-or-honest-default design.
   - **If H2 (architecture-shape problems)**: scope holds. Run CAP-3 (tool-surface variations) and an affordance-variation spike + literature-deep on ensemble affordance design + RQ-5 framing investigation. ADR-001/002 may warrant a future constraint-removal pass.

4. **Worktree posture.** Each spike branches from `agentic-serving` into its own worktree (`scratch/spike-<name>/` for code, `docs/agentic-serving/essays/research-logs/` for findings). Spike code is throwaway by default; learning is recorded in this log.

### Stopping criteria

Research closes when:
- Each of the five RQs has an answer (or a defensible "not answerable without further BUILD") that the practitioner can articulate to a stranger;
- The essay (`docs/agentic-serving/essays/002-...md`) passes citation, argument, and framing audits;
- The practitioner's reflection-time response is substantive (not formulaic approval).

Phase closes on essay revision, not first draft.

## Methods Review (2026-04-27)

The research-methods-reviewer was dispatched against the question set + constraint-removal response in isolated context. Full audit at `../../housekeeping/audits/research-design-review-agentic-serving-loop-from-play.md`. Six findings.

**P1 findings (addressed before research begins):**

1. **Incongruity surfacing — H0 added to S0's prediction structure.** S0 was originally framed to distinguish H1 (capability-bound) from H2 (architecture-shape). The reviewer flagged a missing third hypothesis: H0, that a strong profile resolves the failure *without the cascade architecture doing meaningful work*. This is the load-bearing distinction for RQ-5's seam navigation. Addressed by replacing the H1/H2 partition with the H0/H1/H2 partition in S0's purpose statement and updating the subsequent-spikes branching accordingly.

2. **RQ-3 embedded conclusion — reformulated.** Original "without imposing coordination burden" presupposed the two goals are reconcilable through surface design. Field notes (note 8) leave this open. Amended to admit the case where they cannot be separated without a capability gate upstream.

3. **RQ-4 embedded conclusion — reformulated.** Original "what default configuration ships" presupposed defaults are the right intervention level. Amended to lead with the policy question (configuration vs capability detection vs operator policy) and treat the design question as following the policy answer.

**P2 findings (addressed):**

4. **RQ-2 phrasing — reformulated.** "Reliably distinguish" embedded the achievability of reliable detection. Amended to "What signals — if any..." with explicit honest-default alternatives in scope.

5. **RQ-5 divergence presupposition — reformulated.** "Where the two contracts produce different design choices" assumed divergence. Amended to admit both divergence and convergence as empirical possibilities and to ask what each reveals.

**P3 finding (addressed in sequencing):**

6. **Token cap reconciliation moved to step 0.** FF #131 (encountered 50K cap vs documented 10M) is resolved before S0 fires; an S0 run with a misconfigured cap produces uninterpretable results.

**Watch items (not addressed now; recorded for future passes):**

- ADR-001/002 (architecture) is unbracketed; if S0 surfaces H0 or H1, the architecture's complexity may be irrelevant to the first-session problem and a future constraint-removal pass on ADR-001/002 may be warranted.
- Coverage gap on session-start narration as third observability surface (in scope for OB-2/OB-3 spikes, but not foregrounded in RQ-3's framing). Question phrasing now includes a coverage note.
- Coverage gap on operator's configuration ownership (who states the floor: system, operator, default-shipped). Partially addressed by RQ-4 reformulation but not foregrounded.

## Loop Iterations

### Step 0 — Token-cap reconciliation (2026-04-27)

**Question:** Why was the encountered token cap during PLAY 50K instead of the documented 10M default? (FF #131)

**Method:** File inspection.

**Findings:**

- Local config at `~/.config/llm-orc/config.yaml:80-85` carries an explicit override:
  ```yaml
  agentic_serving:
    orchestrator:
      model_profile: orchestrator-local
    budget:
      turn_limit: 20
      token_limit: 50000
  ```
- Source-code defaults at `src/llm_orc/core/config/config_manager.py:365`: `turn_limit: 500`, `token_limit: 10_000_000`. Constants at `src/llm_orc/agentic/orchestrator_config.py:38`: `DEFAULT_TOKEN_LIMIT = 10_000_000`.
- Practitioner's hypothesis (a) — local override — is confirmed. Hypotheses (b) default-shifted and (c) different-semantics are refuted; the documented 10M default is what ships.

**Implications:**

- The 50K cap encountered in PLAY was a local development override, not a system characteristic. PLAY findings tied to budget exhaustion timing (FF #131, parts of #133) are interpretable against the override, not the default.
- For S0: the override is too tight to give a strong profile room to demonstrate H0 vs H1 vs H2. A misconfigured cap would conflate "the orchestrator hit its budget" with "the architecture did or did not engage." S0 will run with a larger cap (proposed: temporarily set `token_limit` to `1_000_000` and `turn_limit` to `100` in the local config for the spike, then restore). This is enough to let the spike breathe without committing to the documented 10M (which would obscure cost signals if the spike runs against an expensive frontier API).
- For RQ-4 (intervention level): the existing override pattern is itself prior art for "operator policy" as an intervention level. Operators can tune the cap; the question is whether *defaults* or *gates* should also exist.

**Hallucination-burn re-interpretation (FF #133):** PLAY's failure mode of hallucinated tool calls consuming 50K tokens before any tool actually executed is not less significant under this finding — the orchestrator profile fabricated regardless of cap size. But the *6-minute wall-clock to budget exhaustion* is interpretable against a 50K cap, not 10M. With a larger cap, the same fabrication would burn longer; the failure mode is configuration-resistant, which strengthens the H0/H2 distinction in S0's prediction structure: H0 predicts the *strong* profile would not fabricate at any cap; H2 predicts shape problems persist regardless of profile capability.

### S0 — Predictions before spike fires (2026-04-27)

Per the discipline rule (predictions written before spike runs, not after), the practitioner's expectations under each hypothesis are recorded here. These are the agent's first-draft predictions composed from the cycle's prior findings; the practitioner reviews and adjusts before S0 runs. **Predictions recorded but not validated yet — the spike has not fired.**

**Constraint reframing (2026-04-27):** Practitioner clarified S0 must use local Ollama models only. Frontier API models (claude-sonnet-4, gpt-4o) are out of scope. This is a principled constraint — the project's hypothesis is that orchestration with non-frontier models can deliver good results; testing with frontier models would be testing the wrong thing for the project contract. The constraint also reflects that `anthropic-claude-pro-max` provider has been removed from the codebase (verified by source survey of `src/llm_orc/`); two local config entries (`default-claude`, `validate-claude-pro-max`) are stale references from the prior provider era and should be removed in a separate cleanup pass (not in this research scope).

**Methodological consequence of the local-only constraint.** The H0 hypothesis is reframed:

- **Original H0** (frontier-baseline framing): a strong (frontier) profile resolves the failure without the cascade doing meaningful work — confirms configuration-dependence; project hypothesis not validated.
- **Reframed H0'** (local-baseline framing): the *strongest available local Ollama model* resolves the failure without the cascade doing meaningful work — confirms that *for this task class*, local-model capability alone is sufficient and the cascade adds no value. Project hypothesis is *not refuted but is not validated either* — a stronger test is required, designed around tasks that exceed direct-answer capacity.

H0' is methodologically sharper because it tests the project's actual deployment context. A frontier-baseline H0 would have produced a contaminated finding (frontier models' direct-answer capacity is so high that almost any task lands in H0). Local-baseline H0' surfaces the real question: *for tasks where local models are already capable directly, the project does not need the cascade. The cascade earns its complexity only on tasks that exceed direct-answer capacity.* That is itself an important RQ-5 finding.

**Setup that S0 will use:**
- Orchestrator profile: a local Ollama model. Locally available (per `ollama list`): `qwen3:14b` (9.3 GB), `mistral-nemo:12b` (7.1 GB — what failed in PLAY), `deepseek-r1:8b`, `qwen3:8b`, `mistral:7b`, plus smaller models. The strongest already-pulled candidate is `qwen3:14b`. **Decision pending:** run S0 against `qwen3:14b` as-is, or pull a larger model (e.g., `qwen2.5-coder:32b`, `qwen2.5:32b`, `llama3.3:70b`) appropriate to the practitioner's hardware. The choice defines what "strongest local" means for this cycle's deployment context.
- Budget: `turn_limit: 100`, `token_limit: 1_000_000` (temporary local override for the spike; restored after).
- Logging: minimal request/response/tool-dispatch surface added in the spike worktree (operator-side logging surface is essentially empty by design — see prior pre-S0 finding).
- Request: a concrete first-ask matching the shape of the PLAY session — proposed: *"Help me understand the structure of this Python project. Look around and describe what you find."* This request requires either tool use (file inspection via client tools — turn-boundary delegated per ADR-003 + WP-F) or ensemble invocation (if any library ensemble is suitable for codebase orientation). **A practitioner-redirected first-ask is welcome if it better matches their PLAY-session experience.**

**Predictions (under local-only constraint — H0', H1', H2' replace original H0/H1/H2 framing):**

*Under H0' (local-strongest, configuration-only — local-model capability alone resolves):*
- The strongest local Ollama model answers the request directly without invoking any of the five internal tools, *or* invokes `list_ensembles` once trivially and then answers from its own knowledge of "Python project structure" without using ensemble results, *or* delegates a single turn-boundary file-read to OpenCode and answers from that.
- No `compose_ensemble` calls; no `query_knowledge` calls; no Calibration Gate firing.
- Result quality is acceptable to the user contract.
- Server log shows few or zero internal-tool dispatches.
- **Implication if confirmed:** for this task class, local-model capability alone suffices. The cascade architecture earns its complexity only on tasks that exceed direct-answer capacity — itself an RQ-5 finding. RQ-2/RQ-3/RQ-4 narrow to method-contract questions tested against deliberately constructed tasks that *require* cascade engagement. The project hypothesis is not refuted but is not validated by this run.

*Under H1' (local-strongest, capability-engages — cascade does meaningful work):*
- The orchestrator invokes `list_ensembles`, then `get_ensemble` on a candidate, then `invoke_ensemble` — *or* delegates a turn-boundary client tool call (e.g., file-read) to OpenCode and meaningfully integrates results across multiple turns.
- Server log shows multiple internal-tool dispatches; SSE stream includes both `delta.content` and `tool_calls` events.
- Result quality is acceptable; the cascade is observably load-bearing.
- **Implication if confirmed:** the architecture is validated as designed at the local-strongest capability level. The project hypothesis gains direct evidence. Floor question is well-posed: where does capability cross the threshold needed for cascade engagement, and how far below the strongest-local model does it sit? CAP-1 (narrowed profile battery — `mistral-nemo:12b` repeat as PLAY control + a smaller model) and OB-2/OB-3 (observability defaults) follow.

*Under H2' (local-strongest, shape problems persist):*
- Even with the strongest local model, the orchestrator hits a coordination problem the surface does not handle gracefully — needs mid-execution context the turn-boundary delegation pattern (WP-F) does not support; or the five-tool surface is missing an affordance the orchestrator wants; or session-start narration is silent in a way that makes the result uninterpretable; or composes an ensemble that fails calibration without clear recovery; or fabricates tool calls (the PLAY failure mode) despite higher capability.
- Result is mixed or fails for shape reasons, not capability reasons.
- **Implication if confirmed:** RQ-5 (framing coherence) moves to center; CAP-3 (tool-surface variations), affordance-variation spike, and a future constraint-removal pass on ADR-001/002 become load-bearing.

**Heuristic priors on hypothesis likelihood under local-only:**

For *qwen3:14b* on the proposed first-ask: the model is competent and likely tool-calling-capable via Ollama's `/v1` endpoint, but only marginally larger than the failing PLAY model (`mistral-nemo:12b`). H1' is plausible if qwen3 family has stronger tool-following than mistral-nemo at similar scale; H0' is plausible if the model answers from prior knowledge of typical Python project structure without tooling; H2' is plausible if it fabricates similarly to mistral-nemo. Without empirical data, agent prior is approximately equal across the three. Pulling a larger model (32B+ class) shifts the prior toward H0' (more capable models more likely to answer directly) and H1' (more capable models more likely to use tools deliberately when warranted).

A 14B-class model running S0 may produce **boundary findings** that don't cleanly land in any single hypothesis (e.g., model attempts tool use, partially succeeds, partially hallucinates). Such findings are themselves valuable — they characterize the *transition zone* of the floor — but the practitioner should be prepared to record "boundary-near-H1'-but-with-shape-friction" rather than forcing a clean classification.

**What would distinguish H0 from H1 in observation:** The presence or absence of *meaningful* tool use. "Meaningful" here means tool use whose results materially shape the orchestrator's response — not a single perfunctory `list_ensembles` followed by an answer drawn entirely from prior knowledge. The branch point is where this judgment lands; ambiguous cases (one tool call, results lightly used) get noted as "near-H0" and probed with a follow-up task that *requires* cascade engagement.

**What would distinguish H1 from H2 in observation:** Whether failures (if any) are recoverable through more capability, or whether they persist *regardless of capability* because the system's shape works against the task. H1 failures look like "the profile didn't try hard enough"; H2 failures look like "the profile tried the right thing and the surface fought it."

**Predictions are draft and pending practitioner review before S0 fires.**

### S0 — First run, interim findings (2026-04-28)

**Setup actually used:**
- Orchestrator profile: `orchestrator-qwen3-14b` (qwen3:14b via Ollama OpenAI-compatible endpoint)
- Budget: turn_limit=100, token_limit=1_000_000
- Logging patch applied to spike worktree (cli.py, orchestrator_runtime.py, orchestrator_tool_dispatch.py)
- First-ask: *"Help me understand the structure of this Python project. Look around and describe what you find."*
- Client: OpenCode TUI

**Observations:**

1. **Logging patch failed — no `llm_orc.*` INFO output appeared in operator terminal.** The orchestrator runtime *did* reach `generate_with_tools` (proven by stack trace from a subsequent timeout), but the `logger.info("ReAct loop entry: ...")` and downstream INFO lines did not surface. Diagnosed cause: uvicorn's dictConfig (run after `basicConfig` during `uvicorn.run()`) likely overrode the root logger setup, and `llm_orc` logger had no explicit handler. Fix applied: explicit `StreamHandler` attached directly to the `llm_orc` logger with `propagate=False`. Pending re-run for verification.
   - **Meta-finding for RQ-3 (operator-side observability):** even when the patch *intends* to surface signal, the logging stack's interaction with uvicorn's config silently drops what was supposed to be loud. Default-shipped operator observability is not just "we forgot to add logs" — it has subtle plumbing issues that a naive `basicConfig` does not solve. The fix is one explicit handler away, but the fact that the bug is invisible without external verification is itself the finding.

2. **First request hit `httpx.ReadTimeout` after ~3 minutes** (the profile's `timeout_seconds: 180` boundary). This was logged because uvicorn's exception handler caught it. So: error-level surfacing works; INFO-level surfacing did not.

3. **A subsequent request succeeded after ~5 minutes wall-clock** and returned a coherent, accurate description of the project structure. (How a successful response landed when the first one timed out is unclear — possibly an OpenCode-side retry, possibly the SSE stream's behavior is more complex than the timeout suggests. Investigation deferred; not load-bearing for the immediate findings.)

4. **The successful response named 14 paths/files. Every single one is real.** Verified by `ls -e` against the live filesystem:
   - `src/llm_orc/models/` ✓
   - `src/llm_orc/providers/` ✓
   - `src/llm_orc/core/validation/` ✓
   - `src/llm_orc/schemas/` ✓
   - `src/llm_orc/visualization/` ✓
   - `src/llm_orc/cli.py` ✓
   - `src/llm_orc/cli_modules/commands/` ✓
   - `src/llm_orc/reference/` ✓
   - `src/llm_orc/menu_system.py` ✓
   - `docs/adrs/` ✓
   - `docs/design_philosophy.md` ✓
   - `coverage.xml` ✓
   - `dist/` ✓
   - `.github/workflows/` ✓
   
   This is *strikingly* different from PLAY's mistral-nemo:12b session, which fabricated a TypeScript/Node structure for a Python project. The capability gradient is meaningful between the two models even at similar parameter counts (14B vs 12B), and the qwen3 family appears materially less prone to fabrication on this task class.

**Indeterminacy without logs — H0' vs H1' cannot be classified yet.**

The model's response is consistent with two readings:

- **H1' reading**: the orchestrator invoked tools (`list_ensembles`, OpenCode-side file-reads via turn-boundary delegation), and synthesized from real observations. Cascade engaged.
- **H0' reading**: OpenCode pre-injected directory structure into the system prompt or initial context (a common pattern in agentic coding tools), and the model answered from that pre-injected context plus prior training knowledge. No tools were called; cascade did not engage in any load-bearing way; user contract was satisfied by configuration plus capability alone.

Hedging language in the response (*"Likely contains UI menu logic"*) suggests path-correctness from context, content-guessing from prior knowledge — consistent with the H0' reading. If H1' were correct, the model presumably would have read at least one file's contents to anchor its module-purpose claims rather than hedging.

**Without server-side logs of tool dispatches, this remains indeterminate.** The next iteration must surface this signal.

**Other findings worth recording:**

- **Latency: 5 minutes for first turn is product-unusable on this hardware**, even when the response is correct. The user contract (Beck's "honest about what it can deliver") includes latency tolerability as a sub-criterion. A correct answer that takes 5 minutes is a degraded experience even at H0'.
- **Architecture plumbing validates end-to-end.** The request flowed through Serving Layer → Session Registry → Orchestrator Runtime → Tool Dispatch → Model client → Ollama and back. Stack-trace evidence from the timeout *and* the successful response prove the structural integrity of WP-A through WP-I, independently of the capability question.
- **The first-ask is partially contaminated by OpenCode's own context-injection behavior.** A cleaner subsequent ask would force tool use the client cannot pre-inject — e.g., *"What ensembles are available in this llm-orc instance, and what does each one do?"* (requires `list_ensembles` + optionally `get_ensemble`).

**Next iteration plan:**

1. Practitioner restarts `uv run llm-orc serve` to pick up the fixed logging (explicit handler on `llm_orc` logger).
2. Verify logging works by observing the startup banner / first request entry surfaces.
3. Re-run the same first-ask. Observe tool dispatch lines (or their absence). This resolves H0' vs H1'.
4. If H0' is confirmed: run a second ask that requires tool use (proposed: list-ensembles question). This tests whether qwen3:14b can drive the cascade *when forced*.
5. If H1' is confirmed: scope is moderate. Continue with CAP-1 narrowed profile battery and OB-2/OB-3.
6. If H2' (mixed/shape problems): RQ-5 to center; affordance spike follows.

### S0 — Second run, server-side observation working (2026-04-28)

**Setup correction surfaced:** the first run had been executing against the main worktree's package install, not the spike worktree's source — uv was resolving the entry point from the main `.venv`. Diagnostic `print()` statements added to the spike code did not appear, confirming the wrong-source hypothesis. After cd-ing into the spike worktree directory and re-running `uv run llm-orc serve`, the spike's logging and prints both surface correctly. Worth logging as a finding for future spikes: the worktree boundary is enforced by directory location at invocation time, not by activation; a worktree can silently be inactive if invoked from the wrong cwd.

**What the server-side log surfaces (resolves H0' vs H1' decisively):**

Two concurrent sessions entered the runtime within 42ms of each other when the user-ask fired in OpenCode:

- **Session `343ad246...`** — 4 messages, 0 client_tools declared. OpenCode housekeeping/probe request (likely model-discovery or session intro). Turn output not visible in the captured log — either still queued behind the other session on the single Ollama instance, or completed off-screen.
- **Session `796641116...`** — 3 messages, **11 client_tools declared by OpenCode**. The user-ask session. Turn 1 completed in 32 seconds (13:39:36 → 13:40:07).

**For the user-ask session:**

```
LLM turn 1: tokens=1478 content=33ch tool_calls=0 names=-
ReAct loop close: stop (no tool calls)
```

**Classification: H0' confirmed.** The orchestrator emitted zero tool calls. No `list_ensembles`, no `invoke_ensemble`, no client-tool delegation, no `query_knowledge`. The loop closed after one turn with `finish_reason: "stop"`. The cascade architecture (Tool Dispatch, Composition Validator, Calibration Gate, Plexus Adapter, Result Summarizer Harness) was structurally bypassed — none of those modules' code paths were entered for this request. WP-A through WP-I's contribution to *this particular outcome* is approximately zero.

**This validates one significant claim and surfaces one significant puzzle:**

*Validated claim:* the architecture's plumbing works end-to-end. Request flowed Serving Layer → Session Registry → Orchestrator Runtime → Model client → Ollama → response → Runtime → Serving Layer → SSE → OpenCode. WP-A through WP-I's structural integrity is empirically demonstrated by the request not crashing.

*Significant puzzle:* the response content was 33 characters. The earlier "successful" run had returned a multi-paragraph response naming 14 correct paths. Two competing readings of that earlier response:

- **Reading A:** qwen3:14b is non-deterministic. Both responses came from llm-orc/qwen3:14b; the earlier was a longer answer drawn from prior knowledge, this one was a short hedge or non-answer. H0' applies to both — the cascade was never engaged.
- **Reading B:** the earlier "14 paths" response did not come from llm-orc at all. It came from OpenCode's own fallback when llm-orc returned `httpx.ReadTimeout` after 3 minutes. OpenCode is plausibly configured with multiple providers and silently routed to a different one. If true, **llm-orc/qwen3:14b has not been observed producing a useful response yet** in this spike.

The reading depends on what OpenCode displayed for the user-ask session. Pending the practitioner's report on the OpenCode TUI output for this run.

**RQ-5 implication regardless of which reading holds:**

Under either reading, the cascade architecture did not contribute to the user-contract outcome of this task. This is consistent with the H0' interpretation: for the task class "describe a Python project's structure," qwen3:14b at this capability tier either (a) answers from prior knowledge without needing tools, or (b) declines to engage. In neither case does the architecture earn its complexity. **The project hypothesis (orchestration with non-frontier models delivers good results) is not validated by this task class — the cascade was structurally bypassed.**

This sharpens the next-iteration question: can qwen3:14b (or any local model in the user's hardware tier) drive the cascade when given a task that *requires* tool use — and at what latency cost?

**Operator-side observability finding:**

The diagnostic prints + INFO-level logger output worked together to surface the ReAct loop's decision in real time. This is the first time in the cycle that operator-side server log has carried diagnostic-truthful signal about orchestrator behavior. The minimal viable instrumentation (one logger declaration + three call sites) was sufficient to resolve a hypothesis that PLAY had left ambiguous. Recording for OB-2/OB-3 spike work: the cost-to-value ratio of operator-side ReAct loop logging is extremely favorable.

**Latency observation:**

32 seconds per LLM turn on qwen3:14b for a 1478-token context. If the orchestrator decided to invoke even one tool (adding a turn for the response after observation), we would add another 30+s. A four-turn cascade would be ~2 minutes minimum. This is product-unusable on this hardware for interactive coding workflows. Reinforces the practitioner's earlier observation that the realistic deployment ceiling on this machine is meaningfully below 14B for tool-calling orchestration.

**Open questions for the next iteration:**

1. What did OpenCode display in the TUI for the 33-character response? Resolves Reading A vs Reading B.
2. What was session `343ad246`'s eventual outcome? Did OpenCode's probe receive a response, or did it queue indefinitely?
3. Can qwen3:14b drive the cascade when forced? — proposed second ask: *"What ensembles are available in this llm-orc instance, and what does each one do?"* — requires `list_ensembles`, which OpenCode cannot pre-inject.

### S0 — Reclassification: H1'-client-tool variant (2026-04-28)

**Correction to the immediately prior log entry.** The `33ch content / tool_calls=0 / loop close` event was the *probe session* (343ad246, 0 client_tools), not the user-ask. The user-ask session (796641, 11 client_tools) unfolded across two turns with a tool call between them. The full server log makes this clear:

```
13:39:36.391  run() entered  session=796641... msgs=3  client_tools=11
13:42:27.990  LLM turn 1     tokens=15004  content=empty  tool_calls=1 names=['glob']
13:42:28.364  run() re-entered  session=796641...  msgs=5
13:44:08.980  LLM turn 2     tokens=19124  content=1156ch  tool_calls=0
13:44:08.981  ReAct loop close: stop
```

Total wall-clock 4m 32s, matching OpenCode's UI counter.

**Classification (corrected): H1'-client-tool.** The cascade engaged — but through the *client-tool surface* (WP-F Option C turn-boundary delegation), not through the *internal-tool surface* (ADR-003's five internal tools). qwen3:14b chose to call `glob` (one of OpenCode's 11 declared client tools), saw the result, synthesized a structured response.

**Crucial sub-finding: zero internal-tool dispatches.**

- No `list_ensembles` call
- No `invoke_ensemble` call
- No `compose_ensemble` call
- No `query_knowledge` call
- No `record_outcome` call

The Tool Dispatch module routed exactly zero calls. Calibration Gate, Plexus Adapter, Composition Validator, Result Summarizer Harness were structurally present but never exercised — the request flow never reached them. **The orchestration-of-ensembles value proposition was NOT validated by this run.** What was validated is:

- ADR-001 (ReAct loop) ✓ engaged correctly
- ADR-002 (four-layer architecture) ✓ end-to-end flow works
- ADR-003 (closed tool surface, with client-tool union per WP-F) ✓ orchestrator chose a client tool from the declared set
- ADR-008 (autonomy gating) — implicit; the call path passed through Autonomy Policy without denial
- WP-F (turn-boundary delegation, Option C) ✓ delegated cleanly, OpenCode executed glob, session resumed with the result

**Path accuracy in the response: 5 real, 2 fabricated.** Real: `Formula/llm-orchestra.rb`, `htmlcov_viz`, `scripts/check-security.sh`, `llm-orchestra-library`, `src/llm_orc/cli_library`. Fabricated: `src/llm_orc/primitives/replicate_n_times.py` (directory real but specific file invented — `primitives/` actually contains `control_flow/`, `data_transform/`, `file_ops/`, `user_interaction/` subdirs), `profiles/` (no top-level `profiles/` directory exists).

**Latency profile:**
- Turn 1: 15004-token input, 171s wall-clock (input dominated by 16 tool schemas: 5 internal + 11 client)
- Turn 2: 19124-token input (Turn 1 context + assistant tool_call + tool result), 100s wall-clock for 1156-char generation
- Total: ~4m 32s

15K-token tool-schema overhead per turn is substantial. On this hardware, a multi-turn cascade involving multiple tool calls would compound this latency badly — a hypothetical session calling `list_ensembles` then `get_ensemble` then `invoke_ensemble` (three internal tool turns) plus client-side delegations could easily hit 10+ minutes wall-clock. This reinforces the practical-unusability finding from the prior entry.

**RQ-1 (capability floor + affordances) — interim findings:**

- qwen3:14b *can* drive a tool-calling loop with one tool call per turn at this hardware tier, but at unworkable latency.
- qwen3:14b chose a *client* tool (glob), not an *internal* tool, for a "describe project structure" task. This is plausible — the task fits glob's purpose better than `list_ensembles`'s. But it does mean we have not observed the model invoking any of llm-orc's specific tools.
- The internal tool surface's discoverability to the orchestrator at this capability tier is an open question. The next iteration must force the choice.

**RQ-3 (observability) — interim findings:**

- Diagnostic prints + INFO-level logger working in tandem produced a clean, scannable record of the orchestrator's decisions in real time. This is the first observation in the cycle of operator-side server log carrying truthful diagnostic signal about what the orchestrator is doing.
- The cost-to-value ratio of operator-side ReAct loop instrumentation is favorable: ~3 logger lines + 3 print statements (in the spike worktree, disposable) was sufficient to resolve the H0' vs H1' question.
- Beck's "diagnostic truthfulness without coordination burden" is achievable at the operator surface: the user observes what the orchestrator chose, but is not asked to coordinate or interrupt. Watching the loop tick is informational, not actional.

**RQ-4 (intervention level) — interim findings:**

- The current default-config experience for "describe a project's structure" with qwen3:14b on this hardware is: *correct response with hallucination risk*, *delivered in 4m32s*, *not engaging the orchestration architecture*. From a user-contract standpoint this is a partial success (correct content) and a partial failure (latency, fabrication risk). From a project-contract standpoint this is a non-test (the orchestration hypothesis was bypassed).
- This sharpens the policy question: the right intervention may be a *task-shape gate* — refuse to run if the task can be answered by a single LLM call without the orchestration architecture, OR require explicit operator opt-in to "thin-orchestration" mode.

**RQ-5 (dual contract reconciliation) — interim findings:**

- For this task, the user contract was satisfied (mostly correct response) and the project contract was bypassed (no orchestration occurred). The two contracts diverged: the user got value (in some sense), the project's hypothesis was not exercised.
- This is itself a finding worth holding: when the architecture is *available* but the model chooses not to use the parts that distinguish llm-orc from any other tool-calling backend, the architecture's complexity is unearned for this request. The seam between the two contracts is empirically observable.

**Next iteration plan (revised after H1'-client-tool finding):**

1. **Force internal-cascade engagement** to test whether qwen3:14b can drive llm-orc's specific tool surface: fire the proposed second ask — *"What ensembles are available in this llm-orc instance, and what does each one do? Pick one that could help with code review and demonstrate calling it on the README of this project."* This requires `list_ensembles` (server-internal, OpenCode cannot pre-inject) → `get_ensemble` → `invoke_ensemble` (with potential client-tool delegation for the README content).
2. **Observe whether the orchestrator chooses internal tools when they are the only path to the answer.** If it does, the project hypothesis gains evidence at this tier. If it refuses, fabricates, or routes to a client tool inappropriately, that's a substantive RQ-1 finding about internal-tool discoverability.
3. **Latency expectation:** if the model engages internal tools, expect at least 3 turns × ~150s/turn ≈ 7-8 minutes wall-clock. Worth running once for the data, but the latency itself is a finding.

### S0 — Internal cascade engagement observed (2026-04-28, in flight)

**Sharpened ask:** *"What ensembles are available in this llm-orc instance, and what does each one do? Pick one that could help with code review and demonstrate calling it on the README of this project."*

**Observed cascade (turn-by-turn within the same session, cumulative state):**

| Turn | Time | Tool called | Type | Notes |
|------|------|-------------|------|-------|
| 1 | 13:42:27 | `glob` | client (OpenCode) | original ask: glob to discover paths |
| 2 | 13:44:08 | — (text, 1156ch) | synthesis | first response, project structure |
| — | 13:55:35 | (timeout) | retry | httpx ReadTimeout at 180s; OpenCode retried |
| 3 | 13:56:36 | **`list_ensembles`** | **internal** | first internal-tool dispatch in this cycle |
| 4 | 13:58:28 | `read` | client (OpenCode) | reading README presumably |
| 5 | 13:59:11 | **`list_ensembles`** | **internal** | called *again* (context loss?) |
| 6 | 14:00:21 | **`invoke_ensemble`** | **internal** | arg_keys=['input', 'name'] — invoking a selected ensemble |
| ... | in flight | ... | ... | ensemble execution underway |

**Decisive H1' finding — including internal-cascade engagement:**

Three of ADR-003's five internal tools have now been observed: `list_ensembles` (twice), `invoke_ensemble`. Tool Dispatch routed all three correctly; Calibration Gate, Result Summarizer Harness, and Composition Validator code paths are now live in the request flow. The orchestration-of-ensembles part of the project hypothesis has empirical evidence at this capability tier (qwen3:14b on consumer hardware).

**Sub-findings worth recording:**

1. **Mixed-tool fluency.** The orchestrator fluidly alternates client tools (`glob`, `read`) and internal tools (`list_ensembles`, `invoke_ensemble`) within a single conversation. This is exactly the cascade pattern WP-F + ADR-003 + ADR-006 were designed to enable. The model uses each tool surface for what it's good at: client tools for filesystem state, internal tools for orchestration capability.

2. **`list_ensembles` called twice (turns 3 and 5) — RQ-1 finding on working-memory.** The result from turn 3 should still be in the orchestrator's message history, but the model called the tool again at turn 5 (after the `read` interjection). Possible explanations: (a) attention to long context falls off, especially after intervening tool calls; (b) model deliberately refreshed; (c) context summarization removed the prior result. Worth investigating whether the result was actually still in context at turn 5; if so, this is a working-memory observation about qwen3:14b — even at H1'-internal capability, the model exhibits memory-of-prior-results lapses across 2+ tool-call boundaries. Implication for RQ-1: capability has dimensions beyond "can call tools" — including "can remember that it called them."

3. **Pre-existing invalid local-tier ensembles surfaced.** Two local ensembles failed pydantic validation when `list_ensembles` ran:
   - `.llm-orc/ensembles/fan-out-test.yaml` — `ScriptAgentConfig` rejects `type` and `system_prompt` fields
   - `.llm-orc/ensembles/plexus-graph-analysis.yaml` — same `type` field issue
   
   These are user library state, not spike artifacts. EnsembleLoader's strict validation skips them with WARNING-level logs. Implication for RQ-3: operator-side WARNING surfacing on invalid library entries is working correctly; the user now sees library hygiene issues they didn't know about previously. Implication for RQ-4: a default-config experience that surfaces library drift to operators (rather than silently skipping) is *already* in place via this validation pattern; could be elevated to a startup health-check.

4. **Retry semantics during long generations.** At ~13:55:35 (3 minutes after turn 2 entry into a follow-up state), `httpx.ReadTimeout` fired. OpenCode retried 2 seconds later (13:55:37). The retry succeeded at turn 3 (60s after retry entry). This pattern of *timeout-then-silent-retry* is invisible to the OpenCode user but loud in the operator log. Recording for RQ-3: there is an honest-vs-helpful tension at the visibility surface — the user gets eventual success without knowing the path; the operator sees the failure. The Beck framing ("honest about what it can deliver") would lean toward surfacing the retry to the user, but most clients don't expose that detail. Possible architectural finding: timeout signals could become first-class diagnostic events (not just errors) emitted via the SSE stream.

5. **Context growth across the cascade.** Token counts per turn: 15004 → 19124 → 19390 → 22764 → 19710 → 22967. The cascade context grows by 3-5K tokens per turn (largely due to tool result text accumulation). On qwen3:14b at this hardware, this is approaching the practical context-vs-latency boundary. A longer cascade would compound badly.

**This run is the strongest single piece of evidence we have that the project hypothesis is testable at this capability tier.** Even with caveats (latency, pre-existing library drift, working-memory lapses), the orchestrator demonstrably engaged the cascade, used internal tools, and is now invoking an ensemble — exactly the value proposition the architecture was built for.

**Pending:** the `invoke_ensemble` result. Whether the picked ensemble produces useful output, whether the orchestrator can synthesize it into a useful response, and whether the model continues the cascade or terminates after this. Final classification awaits the session completion.

### S0 closes — full architecture exercise observed (2026-04-28)

**Final cascade trajectory (extended from the prior interim entry):**

| Turn | Time | Tool | Result |
|------|------|------|--------|
| 6 | 14:00:21 | `invoke_ensemble` (arg_keys=['input','name']) | dispatched |
| — | 14:00:21–47 | (Ensemble Engine load + execution, Result Summarizer Harness invoked) | — |
| — | 14:00:48 | `tool dispatch: result name=invoke_ensemble kind=error:summarization_failed` | **AS-7 enforcement caught a failure** |
| 7 | 14:01:25 | content=229ch, tool_calls=0, loop close | orchestrator handled the error gracefully |

User-facing response in OpenCode:
> *"→ Read llm-orchestra-library/templates/README.md*
> *The code-review ensemble is designed for reviewing code files, not documentation. Let me know if you'd like me to review a specific code file from this project instead. For document analysis, we might need a different approach..."*

**Total wall-clock for the full session: ~22 minutes** (13:39:36 first request → 14:01:25 final close), 7 turns, with one timeout-retry and ~4-5 ensemble validation warning bursts in between.

**Architecture validation — exhaustive of the modules engaged by this task class:**

| Module | Exercised | Mechanism |
|--------|-----------|-----------|
| Serving Layer | ✓ | request entry, SSE streaming, request handoff to Runtime |
| Session Registry | ✓ | session 796641 reused across 7 turns spanning 22 minutes |
| Orchestrator Runtime (ReAct loop) | ✓ | 7 turn iterations, including one timeout-retry |
| Orchestrator Tool Dispatch | ✓ | routed 5 tool calls: glob (client), read (client), list_ensembles ×2 (internal), invoke_ensemble (internal) |
| Ensemble Engine (existing, Layer 3) | ✓ | invoke_ensemble reached it; EnsembleLoader ran with strict validation |
| Result Summarizer Harness | ✓ | **caught a SummarizationFailure** — AS-7 + ADR-004 + WP-D enforcement worked end-to-end |
| Autonomy Policy | ✓ | implicit Allow on all 5 dispatches |
| Composition Validator | not exercised | no `compose_ensemble` calls in this session |
| Calibration Gate | not exercised | invoked ensemble was a known library ensemble, not composed |
| Plexus Adapter | not exercised | Plexus absent — no-op fallback per ADR-009 |

**Failure-mode handling validated.** The architecture caught a real failure (summarization), surfaced it as a typed `summarization_failed` error to the orchestrator, and the orchestrator synthesized a graceful response. This is exactly the safety property AS-7 + ADR-004 + WP-D were designed to enforce. *The architecture's enforcement mechanisms work as designed.*

**Final classification: H1'-internal CONFIRMED.**

S0's central question — *can the cascade architecture engage at this capability tier?* — is answered: **yes**. qwen3:14b on this hardware can drive the closed five-tool surface, fluidly mix client and internal tools, and synthesize results across multiple turns. The orchestration-of-ensembles part of the project hypothesis has empirical evidence at this tier.

**S0 caveats and open follow-ups:**

1. **Latency cost is product-unusable on this hardware.** 22 minutes for a single useful interaction is far beyond any acceptable UX bound. RQ-4's intervention-level question gains a sharper edge: *for hardware-realistic deployments, is the answer not "ship a default" but "prevent the architecture from being exercised at all without operator opt-in"?* Or: *is the actual usable deployment a hosted service with stronger inference, where the local-orchestration hypothesis becomes the inferential engine for an offline-capable variant?*

2. **Working-memory lapse confirmed.** `list_ensembles` was called twice (turns 3 and 5). The result from turn 3 should still have been in the orchestrator's message history at turn 5; the model called the tool again. Either (a) attention falls off across long context, (b) the model deliberately refreshed, or (c) some context-management step removed the prior result. This is an RQ-1 finding worth a follow-up: instrument turn 5's full message context to determine whether the prior list result was visible to the model.

3. **Summarization failure is a generic signal.** The dispatch returned `error:summarization_failed` without further detail. The orchestrator interpreted this as *"ensemble doesn't fit the input"* (code-review ensemble for docs). This interpretation is plausible but not directly evidenced — the failure could have been a summarizer ensemble crash, a malformed output, a timeout, or the ensemble itself failing. Follow-up spike candidate: re-run with logger surfacing the specific `SummarizationFailure.reason` field. RQ-1 finding: typed errors at the tool-dispatch layer are coarse; the orchestrator infers from minimal signal, sometimes confabulating plausible explanations.

4. **Two pre-existing invalid library ensembles surfaced.**
   - `.llm-orc/ensembles/fan-out-test.yaml` — `ScriptAgentConfig` schema rejects `type` and `system_prompt` fields
   - `.llm-orc/ensembles/plexus-graph-analysis.yaml` — same `type` field issue

   These are user library drift, not spike artifacts. The EnsembleLoader's strict validation skipped them with WARNING-level logs. Operationally interesting because: (a) the user didn't know the drift was there, (b) the validation surfaces it, (c) library hygiene is now a visible operator concern. Possible RQ-4 finding: a startup health-check that surfaces drift is "free" given the existing validation pattern, and would tell the user "your library has problems" before any session starts.

5. **Retry semantics during long generations are silent to the user.** The 13:55:35 timeout was caught only by the operator log; OpenCode silently retried. From the user's perspective the request just took longer. From a Beck-honest-about-what-it-can-deliver standpoint, this is exactly the kind of opacity the framing audit warned about — the user doesn't see the path their answer took. RQ-3 finding: timeout-as-first-class-event might be a candidate addition to the SSE stream's visibility surface.

**RQ-1 closing assessment from S0:**

- Capability floor for driving llm-orc's tool surface: at or below qwen3:14b on this hardware (the model can do it).
- Floor for driving it *at acceptable latency*: above qwen3:14b on this hardware. Not workable as a product UX without significantly more inference compute.
- Tool-surface composition: 5 internal + 11 client = 16 schemas adds ~15K tokens of input per turn. This is a substantial overhead that compounds with cascade depth.
- Affordances: the model used the existing affordances correctly — `list_ensembles` with empty args, `invoke_ensemble` with `name` and `input`. No affordance gaps obvious from this task.

**RQ-3 closing assessment from S0:**

- Operator-side observability is achievable with minimal instrumentation. ~3 logger lines + 3 print statements gave full diagnostic visibility into the ReAct loop's decisions.
- Beck's *"diagnostic truthfulness without coordination burden"* is achievable at the operator surface: the operator watches what the orchestrator chose without being asked to coordinate.
- The tool-user-side experience is opaque by comparison — silent timeouts, silent retries, silent context growth. The Beck framing's challenge applies most strongly here.
- Validation warnings already provide a useful operator signal (library drift visible).

**RQ-5 closing assessment from S0:**

- The dual contracts (user outcome / project method) **converge** on this run. The user got a useful response (mostly correct, with a graceful failure-mode handoff), AND the project's orchestration hypothesis was exercised (multi-tool cascade). The seam was navigated by the architecture itself.
- BUT: the user contract was satisfied at 22-minute wall-clock latency, which is degraded. The "honest about what it can deliver" framing applies — at this hardware, the system is honest about *what* but not about *when*.
- The seam diverges on latency: the project method validates (cascade works), the user contract degrades (latency unworkable). Different deployment contexts produce different reconciliations.

**S0 closure recommendation:**

S0 is sufficient to answer the core question that triggered this research loop. The architecture works; the capability floor is at or below qwen3:14b for tool engagement; latency is the binding constraint at hardware-realistic local-only deployment.

Three immediate follow-ups identifiable for subsequent loops:

- **Spike on summarization failure modes.** Capture full `SummarizationFailure.reason` payloads to characterize what's actually failing.
- **Spike on working-memory across long context.** Instrument turn 5's full message history to determine whether the prior `list_ensembles` result was actually in context.
- **Literature scan can now begin in parallel** — informed by the empirical findings, the lit scan is no longer abstract but has concrete questions to test against (BFCL/ToolBench: how do they handle tool-use evaluation in long-context cascades? Observability patterns: how do other systems surface retry semantics to the user? Tool-design: how does schema verbosity affect floor capability?).

### S0-CAP-1 — qwen3:8b capability-gradient spike (2026-04-28)

**Spike question:** Does the cascade architecture engage when the orchestrator drops from qwen3:14b to qwen3:8b? At what latency cost? With what failure modes?

**Method:** Same setup as the qwen3:14b S0 run. Same first-asks (project structure + ensembles + code-review). Same logging. Only change: orchestrator profile switched to `orchestrator-qwen3-8b` (qwen3:8b via Ollama OpenAI-compatible endpoint).

**Setup change:** Added `orchestrator-qwen3-8b` profile to `~/.config/llm-orc/config.yaml`. Switched `agentic_serving.orchestrator.model_profile` from `orchestrator-qwen3-14b` to `orchestrator-qwen3-8b`. Budget unchanged (turn_limit: 100, token_limit: 1_000_000). The `orchestrator-qwen3-14b` profile is preserved for comparison and future spikes.

**Predictions (recorded before spike fires):**

*Latency prediction:* substantially faster per turn (5.2 GB model vs 9.3 GB). Probably 30-60s per turn for 15-23K-token contexts vs the 60-180s observed at 14B. A full 7-turn cascade analogous to the qwen3:14b run might land in 6-10 minutes wall-clock instead of 22.

*Capability predictions, three branching scenarios:*

- **CAP-A — H1'-internal still confirmed at 8B.** qwen3:8b drives the cascade similarly to qwen3:14b — engages internal tools, sequences turns, synthesizes results. Possibly less polished output, possibly more confused on long context, but the architecture is exercised. *This would be the strongest finding for the project hypothesis: the floor is below 8B, meaning consumer hardware is genuinely viable.*
- **CAP-B — H1'-client-only at 8B.** qwen3:8b can drive client tools (glob, read) but stumbles or fabricates on internal tools, OR doesn't recognize internal tools as relevant to the task. Returns plausible-sounding output drawn mostly from priors plus client-tool results. *This would mean the orchestration capability floor sits between 8B and 14B in the qwen3 family — useful precision.*
- **CAP-C — H2'/regression to fabrication at 8B.** qwen3:8b behaves like mistral-nemo:12b did in PLAY: hallucinates tool calls, produces fabricated content, fails to drive the surface coherently. *This would be the strongest evidence for "the qwen3:14b result is a capability cliff, and dropping below it loses everything."*

*Heuristic prior:* qwen3:8b is the smaller sibling of a known-tool-calling-capable family. The drop from 14B to 8B in the same family typically degrades long-context handling and fine-grained instruction following more than basic tool-calling itself. Best guess is between CAP-A and CAP-B, with CAP-B somewhat more likely than CAP-A — the model probably can call tools but may make worse decisions about which tool to pick at each turn, or struggle with the 15K+ token cumulative context.

**Predictions are draft and pending observation.**

**First-ask observation (qwen3:8b, project structure question):**

Same flow as qwen3:14b on the first ask: glob client tool → synthesis. No internal tools called. H1'-client-only on this ask, identical pattern to the larger sibling. Cascade structure proceeded through the same path; the only difference was speed.

| Metric | qwen3:14b | qwen3:8b | Δ |
|--------|-----------|----------|---|
| Wall-clock total | 4m 32s | 2m 27s | **−46%** |
| Turn 1 latency | 171s | 100s | −42% |
| Turn 2 latency | 100s | 47s | −53% |
| Tool used | glob | glob | same |
| Tool calls turn 1 | 1 | 1 | same |
| Synthesis length | 1156ch | 1324ch | +15% |
| Path accuracy | 5/7 | 11/12 | better at 8B |

Path accuracy comparison is striking: qwen3:8b's response named 12 paths/files; 11 of them are real, the 12th is a partial truncation (`tool_dispatch.py` instead of `orchestrator_tool_dispatch.py`) — recognizable filename shortening, not a wholesale fabrication. Compare to qwen3:14b's second run which fabricated `replicate_n_times.py` (a specific file inside a real directory) and `profiles/` (an entire top-level directory). Sample size small, but qwen3:8b appears at least as grounded as qwen3:14b on this task — and possibly more so.

This is itself a finding worth flagging: capability is multi-dimensional, and "smaller model" does not monotonically degrade across all dimensions. qwen3:8b shows a *latency* improvement (expected) and possibly a *factual grounding* improvement (unexpected). Possible explanations: the smaller model's tighter attention head budget keeps it closer to the glob result and less prone to filling in priors; or the difference is just sampling noise across small n. Worth holding both readings.

**Pending: second ask** to test internal-tool engagement at this tier (the question that distinguished H1'-client-only from H1'-internal on qwen3:14b).

**Second-ask observation (qwen3:8b, ensembles question) — CAP-B confirmed:**

Trajectory across the second ask (cumulative session state):

| Turn | Time | Tool | Result |
|------|------|------|--------|
| 3 | 14:13:28 | `skill` (client) | tool_calls=1, content=empty, arg='codebase-audit' |
| — | 14:13:29 | Skill returns (1 second elapsed → OpenCode either ran fast or returned an error) | msgs=9 |
| 4 | 14:14:41 | (none) | tokens=25035, content=empty, tool_calls=0, **loop close: stop** |

Total wall-clock for this ask: ~2m 13s.

**OpenCode UI showed:** `→ Skill "codebase-audit"` followed by execution stopping with no further content.

**Classification: CAP-B confirmed.** qwen3:8b can drive client tools (it called `skill` correctly with valid arguments) but does not engage llm-orc's internal-tool surface (`list_ensembles`, `invoke_ensemble`, etc.) for a task that semantically calls for it. Internal-tool discoverability is below the floor at this capability tier on this hardware.

The orchestration-of-ensembles capability floor in the qwen3 family on this hardware sits **between 8B and 14B**.

**Three new sub-findings:**

1. **Tool-surface confusion is content-driven, not structural at this tier.** qwen3:8b's tool-calling mechanism worked correctly — it produced a valid `tool_calls` payload with proper arguments. What failed is *which tool* the model picked. "What ensembles are available" → the model retrieved on the word *"available"* and chose `skill` (an OpenCode discovery tool) over `list_ensembles` (the actual answer). The internal-tool descriptions weren't salient enough to win against the larger pool of client tools when the question's wording loosely matched both surfaces. **RQ-1 finding: tool-surface composition affects discoverability at lower capability tiers — adding more client tools can dilute internal-tool selection.** This has implications for whether/how to expose llm-orc's internal tools when many client tools are also declared.

2. **"Silent giveup" is a new failure mode.** Distinct from fabrication (mistral-nemo:12b in PLAY) and from successful synthesis. The model received a tool result, was unable to make sense of it (or unable to integrate it with the original question), and produced empty content + no further tool calls. The architecture saw `Completion(finish_reason="stop")` with empty content and closed the loop cleanly. Recording as **H2'-shape variant: silent-giveup**. Worth distinguishing from fabrication in any taxonomy of failure modes — fabrication is "model produces wrong content"; silent-giveup is "model produces no content."

3. **Failing fast has its own UX value.** The CAP-B failure took 2m 13s vs CAP-A's 22m success on qwen3:14b. From a Beck-honesty standpoint, "I cannot answer your question quickly" is in some senses preferable to "I'll struggle for 22 minutes to give you a partially-correct answer with one summarization failure along the way." The two failure modes (CAP-A graceful but slow; CAP-B fast but silent) are themselves different UX characters. **RQ-3 finding: latency-failure-mode pairs may be a useful taxonomy beyond just success/failure.**

**Closing comparison table (qwen3:14b vs qwen3:8b across both asks):**

| Dimension | qwen3:14b | qwen3:8b |
|-----------|-----------|----------|
| Ask 1 (project structure) | H1'-client-only, 4m 32s, 5/7 paths real | H1'-client-only, 2m 27s, 11/12 paths real |
| Ask 2 (ensembles) | H1'-internal cascade, 22 min, graceful summarizer-failure handling | CAP-B (skill→stuck), 2m 13s, silent giveup |
| Internal-tool engagement | yes (list_ensembles, invoke_ensemble) | no (chose client tool instead) |
| Failure modes observed | summarization_failed (handled gracefully) | silent giveup (loop closed with empty content) |
| Latency profile | slow but capable | fast but capability-limited |

**RQ-1 closing for the qwen3 family on this hardware:**

The capability floor for full cascade engagement is between 8B and 14B. The capability floor for tool-calling at all is below 8B (8B can call client tools correctly). The internal-tool *discoverability* threshold is the operationally relevant one and it sits at ~14B on this hardware. **Implication for the project hypothesis:** the floor is real and observable; non-frontier orchestration is viable, but the lower bound for "non-frontier" is not as low as one might hope (8B is below the discoverability floor; 14B works but at unworkable latency). A meaningful product would likely require either better hardware (faster 14B inference) or different model families (e.g., a tool-tuned 8B-class model whose internal-tool selection improves over baseline qwen3:8b).

This is a load-bearing finding for the capability-floor essay and informs the eventual DECIDE on operator-config policy (RQ-4).

**S0-CAP-1 closes.** The capability gradient is empirically located. Three follow-up directions are now visible:

- **Spike on tool-tuned 8B models** — try `qwen2.5-coder:7b` or similar tool-specialized small models to see whether tool-tuning at smaller parameter count clears the discoverability floor that base qwen3:8b doesn't.
- **Spike on tool-surface composition** — vary the client-tool pool size (e.g., minimal OpenCode profile with fewer tools declared) to test whether internal-tool discoverability improves when client tools are less numerous.
- **Literature scan ready to launch** — informed now by concrete empirical findings on which to anchor the lit review's questions.

### S0-CAP-2 — Structural composition test (2026-04-28)

**Triggered by practitioner observation:** *"What if the orchestrator itself was an ensemble that could know to use its own tooling?"* Connects to the pre-S0 finding that ADR-011 was an unexamined commitment. S0-CAP-1's silent-giveup gave empirical grounding for reopening that question.

**Smallest-possible-validation approach (Beck framing):** rather than rebuilding the Orchestrator Runtime to support ensemble-as-orchestrator, simulate the cognitive split using the existing ensemble engine. Build two YAML ensembles, both backed by qwen3:8b, both invoked via `llm-orc invoke` (no agentic-serving HTTP path involved). Compare tool-selection accuracy.

**Two ensembles built in spike worktree's `.llm-orc/ensembles/`:**

- `spike-s0-orchestrator-single.yaml` — single qwen3:8b agent with all 16 tools (5 internal + 11 client) described in its system prompt. Asked to pick ONE tool to call first.
- `spike-s0-orchestrator-router-executor.yaml` — two qwen3:8b agents in sequence. Router classifies the request as "internal" or "client" surface and outputs the relevant tool subset. Executor picks ONE specific tool from the router's narrowed list.

Both invoked with the same user-input as S0-CAP-1's failing ask: *"What ensembles are available in this llm-orc instance, and what does each one do? Pick one that could help with code review and demonstrate calling it on the README of this project."*

**Results:**

| Architecture | Output | Latency | Tokens |
|--------------|--------|---------|--------|
| Single qwen3:8b | `tool: list_ensembles` ✓ correct | 7.73s | 624 |
| Router-executor | router→"internal"→executor→`list_ensembles` ✓ correct | 19.64s | 1208 |

**Both architectures picked the correct tool.** The router correctly classified the request as "internal" surface, identified the relevant tool subset (`list_ensembles`, `invoke_ensemble`), and the executor picked `list_ensembles` with appropriate rationale. The single-agent baseline also picked correctly with terser reasoning.

**Critical interpretation: the test conditions are materially different from S0-CAP-1.**

- S0-CAP-1: OpenCode SSE tool-calling, real JSON schemas, OpenCode's system prompt + 11 client tools competing, multi-turn cumulative context (15K+ tokens), the `skill` tool that semantically resembled "what's available."
- This test: CLI ensemble engine, text descriptions of tools (custom-written), single-turn invocation, no competing system prompt, no tool-calling format, ~600-1200 tokens of context.

When the failure-conditions are removed (description quality controlled, single-turn, competing tools absent, format simplified), qwen3:8b picks correctly under either architecture. The structural composition does its job — but so does the simpler single-agent setup once the noise is reduced.

**What we learn from this:**

1. **The discoverability floor is not a single number.** It's a function of (model capability × surface composition × description quality × competing cues × turn-state context). Multiple interventions can move it. S0-CAP-1's failure at qwen3:8b in OpenCode does not mean qwen3:8b is incapable; it means qwen3:8b *under those specific conditions* couldn't discriminate.

2. **Structural composition is a real lever, but not the only one.** The router-executor mechanism works exactly as predicted: it correctly narrows the surface and reduces the cognitive load on the final tool-selection step. But in this test, where conditions were favorable to single-agent success too, the structural intervention added latency without changing the outcome.

3. **Tool-description salience is also a strong lever.** The single-agent baseline picked correctly partly because the system prompt explicitly framed `list_ensembles` as the right tool for "what's available in this llm-orc instance" queries. That framing was absent (or competed against by OpenCode's framing) in S0-CAP-1.

4. **The structural validation is real but partial.** It demonstrates the router-executor pattern works as designed. It does NOT demonstrate that structural composition would have rescued S0-CAP-1's failure in OpenCode — to test that, we would need to reproduce S0-CAP-1's full conditions (SSE tool-calling, OpenCode's system prompt, the 11 client tools as real schemas, multi-turn context) inside an ensemble-backed orchestrator. That's substantially more work and would require modifying the Orchestrator Runtime.

5. **Latency cost is real and meaningful.** The router-executor pattern doubled wall-clock time in this test (19.64s vs 7.73s). Scaled to the actual orchestrator context (15K-token contexts on consumer hardware), this could mean 4-6 minutes per turn instead of 2-3 minutes — compounding over a multi-turn cascade.

**Implications for ADR-011 reopening:**

The structural validation provides *partial* support for revisiting ADR-011. The router-executor pattern is real, works as designed, and would address a class of failure modes (surface confusion, cognitive overload at lower capability). But it is not a clean win:

- It adds latency proportional to the number of stages.
- It introduces design complexity (recursion: orchestrator-as-ensemble that calls other ensembles; what calibrates the orchestrator-ensemble itself; etc.).
- Cheaper interventions (better tool descriptions, narrower client tool surfaces, surface-aware routing in the existing single-LLM orchestrator) likely capture some of the same benefit.

**Recommended posture: do not reopen ADR-011 yet.** Instead, generate evidence on the cheaper interventions first:

- **Better tool descriptions** — modify the existing internal tool descriptions (in `_build_tool_schemas`) to be more discriminative. Re-run S0-CAP-1's exact ask. If qwen3:8b picks `list_ensembles`, salience was the binding constraint and structural reopening is unwarranted.
- **Narrower client tool surface in the test** — configure OpenCode to declare fewer tools (or test against Cline/Roo Code/whatever else has different defaults). If qwen3:8b picks correctly with fewer competing tools, the surface-composition variable is binding.
- **System prompt augmentation in the orchestrator profile** — add operator-side guidance to the orchestrator-local profile's system prompt: *"When the user asks about ensembles or llm-orc state, always check `list_ensembles` first."* This is a configuration change, not a structural one.

If those cheaper interventions don't clear the floor under realistic deployment conditions, ADR-011 reopening becomes empirically grounded. If they do, ADR-011 holds and the project's existing single-LLM-orchestrator commitment is sound.

**S0-CAP-2 closes.** The structural-composition hypothesis is partially validated — the mechanism works. But the simplest reading of this test is that *S0-CAP-1's failure had multiple causes, and structural composition addresses only some of them at meaningful latency cost.* The project hypothesis is still on the table; the right next moves are cheaper interventions before considering structural ones.

### Lit-scan partial: Canonical's local-first agentic strategy (2026-04-28)

**Triggered by:** Pragmatic Engineer newsletter article (paywalled, fragment seen referencing Canonical/Jon Seager/inference snaps). Investigated via public adjacent material.

**Canonical's strategic positioning (from Jon Seager, VP of Engineering, public statements 2026):**

- *"Not becoming an AI product"* — Canonical's framing. AI is layered into the OS; it is not the product itself.
- *"Bias toward local inference by default."*
- *"Using AI for its own sake is not a constructive goal for anything but increasing exposure."*
- *"What today seems like it's only possible with access to a frontier AI factory will become significantly more accessible in the coming months and years."* (A capability-bet claim — local capability gap is real today; Canonical's hypothesis is that it closes within years.)

**Architecture: Inference Snaps (shipping now in beta).**

- One snap = one fine-tuned model with multiple engine variants
- Engine Manager auto-selects engine + quantization based on hardware (Intel CPUs/GPUs/NPUs, Ampere CPUs)
- Models shipping: DeepSeek R1, Qwen 2.5 VL
- Exposes OpenAI-compatible API on localhost
- Quantization and runtime complexity hidden from user — single-command deployment
- No tool-calling or agentic primitives at this layer; that's a separate planned layer above

**Agentic primitives vision (planned, not yet shipping):**

Seager's framing: *"expose the primitives needed for agents to operate within existing boundaries, whether that be read-only analysis, tightly scoped permissions for any actions, and full auditability of decisions and outcomes."*

Three categorical primitives identifiable:
1. **Read-only analysis mode** — agents can observe but not act
2. **Tightly-scoped permissions for actions** — agents can act only within explicit grants
3. **Full auditability of decisions and outcomes** — every choice is recorded and inspectable

Security model: Snap confinement provides per-app authorization gating. Per-version rollout: 26.04 LTS has authorization prompts; 26.10 opt-in preview; 27.04 setup-time choice.

**Implications for the agentic-serving cycle:**

*Convergence with the project's value proposition:*
- Both bet on local inference for cost, privacy, and capability-via-orchestration
- Both target consumer hardware
- Both expose OpenAI-compatible APIs as the integration point
- Both treat agentic workflows as a value layer ABOVE raw inference

*Where llm-orc adds something Canonical's stack doesn't (yet):*
- The **orchestration layer above inference** — composing ensembles, calibration, multi-agent coordination
- Canonical's inference snap layer is one model per snap; llm-orc orchestrates across multiple models, scripts, ensembles
- An llm-orc deployment could in principle USE inference snaps as its model providers — the OpenAI-compatible API on localhost is a clean integration point

*Where Canonical is ahead of llm-orc:*
- Hardware-optimized model deployment is operationally solved (auto-selection, quantization-aware)
- Snap confinement is a security model for agentic operations that llm-orc has no equivalent of
- OS-level integration with system primitives (file system, system administration, process management) — accessible through snap interfaces in a way llm-orc's tool surface cannot match standalone

*Where the strategies might intersect:*
- llm-orc could be distributed as an inference snap itself (or as a regular snap that depends on inference snaps as model providers)
- llm-orc's tool surface could eventually integrate with Ubuntu's snap-confined system primitives
- Canonical's "auditability of decisions" maps closely to llm-orc's emerging operator-side observability surface (S0 finding) — same problem, similar framing; possible cross-pollination

**Implications for the cycle's research questions:**

*RQ-1 (capability floor):* Canonical has not published capability-floor data — their public stance is forward-looking ("will become more accessible"). The empirical work in this cycle (S0 + S0-CAP-1) is therefore *contributing original signal* to a question the broader ecosystem is still hand-waving. The qwen3:14b/8b gradient finding is the kind of grounded data missing from public discourse.

*RQ-3 (observability):* Canonical's framing of *"full auditability of decisions and outcomes"* as a first-class agentic primitive is consonant with the cycle's RQ-3 framing of *"diagnostic truthfulness"*. Worth borrowing and developing the convergence: the project's RQ-3 is structurally well-aligned with where the broader local-first AI ecosystem is heading. Auditability-as-primitive is an externally-validated framing.

*RQ-4 (intervention level):* Canonical's three-mode permissions (read-only / scoped-action / audited) is a more mature intervention surface than llm-orc currently has. The cycle's existing Autonomy Policy (ADR-008) maps loosely to this but isn't expressed in Canonical's vocabulary. Worth examining whether ADR-008's expression could be aligned with the read-only/scoped/audited framing as it matures.

*RQ-5 (dual contracts):* Canonical's strategy is itself an enactment of the project-method-vs-user-outcome tension. They're building inference-snap infrastructure (project method) before users have specific outcomes that demand it (user need). They're betting on *"will become more accessible"* — a method bet. This validates the framing tension as a real category at industry scale, not just a llm-orc-specific concern. Both projects face the same question: how to ship usable user-contract value while the project method is still developing.

**What this lit-scan partial does NOT cover (gaps for further research):**

- **The Pragmatic Engineer article itself** — paywalled. The technical detail of Canonical's agentic primitives architecture (which Jon Seager presumably elaborated in the interview) remains uncaptured. Worth flagging as a known-but-unread reference; if the practitioner has access, the content could be integrated directly.
- **Tool-use capability evaluation literature** (BFCL, ToolBench, tau-bench, AgentBench) — primary research subject for capability-floor empirical grounding; not yet investigated.
- **Observability patterns in shipped agentic systems** (OpenHands, Aider, Cursor, Claude Code) — similar systems with similar problems; their solutions are likely informative.
- **Tool-design literature** — schema verbosity, description salience, surface composition effects on selection accuracy at lower capability tiers.
- **Multi-agent orchestrator patterns** in LangGraph, AutoGen, CrewAI — structural patterns adjacent to the orchestrator-as-ensemble question raised in S0-CAP-2.
- **Beck's prior Genie posts (Lessons #1-#4)** — surrounding context for the feature-vs-outcome framing.

**Status: lit-scan partial. Five gaps named for further research.** A more thorough scan (via `/rdd-lit-review`) would systematically address each. The Canonical findings alone are a valuable input — they situate llm-orc within an externally-validated landscape and provide a vocabulary alignment opportunity (especially "auditability of decisions" as a first-class primitive, mapping to RQ-3's diagnostic-truthfulness framing).

### S0-CAP-3 — Biased system prompt at qwen3:8b (2026-04-28)

**Triggered by:** practitioner question — *"can we give a really good system prompt to our orchestrator to bias it toward our internal tools?"* — the cheapest intervention to test before considering structural reopening of ADR-011.

**Method:** Override `DEFAULT_ORCHESTRATOR_SYSTEM_PROMPT` via operator config (`agentic_serving.orchestrator.system_prompt` in `~/.config/llm-orc/config.yaml`). The override gives the orchestrator:
- Explicit ordering: internal tools FIRST when question semantics warrant
- Trigger-word patterns ("ensembles", "available", "compose", "what does this system do", "llm-orc")
- A decision heuristic checklist
- Negative-example warning against confusing similarly-named client tools (e.g., `skill`) for capability queries
- Latency framing — internal tools are fast/free vs client tools slower

Same orchestrator profile as CAP-1 (`orchestrator-qwen3-8b`). Same first-ask. The only variable changed is the system prompt.

**New tooling: `opencode run` headless invocation.** The practitioner asked whether spikes could be programmatic; OpenCode's `run` subcommand is fully headless and accepts `-m provider/model` + `--format json`. Required minor OpenCode config addition (`~/.config/opencode/opencode.json` registered the new model IDs alongside the existing `orchestrator-local`). With this in place, spikes are scriptable end-to-end — no human terminal-driving needed.

**Observed cascade trajectory:**

| Turn | Time | Tokens (input) | Tool called | Result |
|------|------|----------------|-------------|--------|
| 1 | 16:59:05 → 17:00:44 (~99s) | 15334 | `list_ensembles()` | success |
| 2 | → 17:01:20 (~36s) | 18858 | `invoke_ensemble(name, input)` | error: summarization_failed |
| 3 | → 17:02:30 (~70s) | 19218 | `invoke_ensemble(name, input)` | error: summarization_failed |
| 4 | → 17:03:16 (~46s) | 19071 | `invoke_ensemble(name, input)` | error: summarization_failed |
| 5 | → 17:03:56 (~40s) | 19038 | `invoke_ensemble(name, input)` | error: summarization_failed |
| 6 | → 17:04:57 (~61s) | 19749 | (none) — 413ch text response | loop close: stop |

Total wall-clock: ~5m 53s. The orchestrator emitted a final text response suggesting an alternative approach (use `example-local-ensemble` instead of code-review for the README task).

**Decisive finding: H1'-internal CONFIRMED at qwen3:8b with biased system prompt.**

This is the same orchestrator profile (qwen3:8b) that failed silently in CAP-1 by choosing `skill` (a client tool) when asked about ensembles. With the biased prompt, the same model:

- Picked `list_ensembles` first (exactly the prompt's directive)
- Drove the cascade through the internal surface (list → invoke_ensemble)
- Persisted productively through 4 summarization failures with retries
- Synthesized a graceful text response when retries didn't resolve
- Did NOT confuse `skill` for `list_ensembles`

**The biased system prompt — a configuration change with no code modification — moved the discoverability floor below qwen3:8b.**

**Comparison table (CAP-1 vs CAP-3, same model, same ask, only system prompt differs):**

| Dimension | CAP-1 (default prompt) | CAP-3 (biased prompt) |
|-----------|------------------------|----------------------|
| First tool selected | `skill` (client, wrong surface) | `list_ensembles` (internal, correct) |
| Internal cascade engaged | NO | YES |
| Tool dispatches observed | 1 (skill) + silent giveup | 5 (list + 4 invoke retries) |
| Failure handling | silent giveup, 0 content | persistence + graceful text fallback (413ch) |
| Total wall-clock | ~2m 13s (failure) | ~5m 53s (useful) |
| User contract satisfied | NO | partial (text guidance, no actual code review) |
| Project hypothesis exercised | NO | YES (cascade through internal surface) |

**Implications for the cycle's research questions:**

*RQ-1 (capability floor and affordances):* The empirical capability floor at qwen3:8b for *internal-tool engagement* is highly sensitive to prompt framing. With a default-neutral system prompt, qwen3:8b sits below the floor. With the biased prompt, qwen3:8b sits above the floor. This means the floor is **not a fixed property of the model** — it is a property of (model × system prompt × tool descriptions × competing surface). Salience of the internal tool surface, communicated via system prompt, is a strong lever.

*RQ-3 (observability):* No new observability findings from CAP-3 itself; the previous instrumentation continued to surface clean signal. Worth noting: the 4 retries on `invoke_ensemble` produced operator-side log clutter from the EnsembleLoader's pydantic-validation warnings on each list pass, repeated 4× through the cascade. That's a real-world operator-experience signal that `list_ensembles` (which validates all entries) is currently noisy at the WARNING level when library drift exists.

*RQ-4 (intervention level):* **The cheapest intervention that meaningfully improves first-session experience is to ship a better default system prompt.** Specifically, upstreaming a system prompt similar to CAP-3's biased version as the new `DEFAULT_ORCHESTRATOR_SYSTEM_PROMPT` would:
- Move the internal-tool discoverability floor below qwen3:8b for typical task classes
- Require zero architectural change
- Require zero operator configuration (it's the default)
- Improve outcome quality on the user contract

This is a load-bearing finding for any DECIDE work that follows from the cycle.

*RQ-5 (dual-contract reconciliation):* Under CAP-3, the dual contracts converged: the user got a (degraded but useful) response, the project's orchestration hypothesis was exercised. The convergence happened because the system prompt steered the model toward the architecture's strengths. The seam can be navigated through prompt design — a lighter-touch intervention than the structural reopening considered in CAP-2.

**Implications for ADR-011:**

CAP-3 provides empirical evidence that **ADR-011's single-LLM-orchestrator commitment is sufficient for this capability tier** — provided the default system prompt is biased toward internal tools. The structural reopening considered after CAP-1 is **not empirically warranted**. The cheapest fix (configuration change to the default prompt) achieves the desired effect without architectural change.

ADR-011 holds. The pre-S0 finding about ADR-011 being unexamined is now empirically resolved: it was unexamined but adequate. The pendant ADR for "shipped default system prompt" — a possible new ADR to govern what guidance ships with the orchestrator — is a more proportional response to the finding than reopening ADR-011 itself.

**Persistent open question: summarization_failed.**

Both qwen3:14b S0 and qwen3:8b CAP-3 hit `summarization_failed` on the code-review ensemble specifically. Four consecutive failures in CAP-3 strongly suggest this is structural to the ensemble or summarizer config, not capability-related. The orchestrator's behavior under repeated failure (4 retries before graceful giveup) is itself an interesting RQ-1 / RQ-3 finding, but the underlying failure cause is not yet diagnosed. A targeted spike to surface the actual `SummarizationFailure.reason` payload is the natural next move on this front.

### Lit-review subagent — usage-limit interruption (2026-04-28)

The dispatched `rdd:lit-reviewer` subagent for the five gaps (tool-use capability eval, observability patterns, tool-design literature, multi-agent orchestrator patterns, Beck's prior Genie posts) terminated with *"You've hit your org's monthly usage limit"* before producing output. The lit scan remains at the partial-Canonical state recorded earlier.

Status: parked. Options for resumption:
- Wait for monthly usage reset; re-dispatch the lit reviewer
- Manual focused searches on individual gaps as needed
- Proceed with empirical work for now and revisit lit scan when limits reset

The CAP-3 finding is decisive enough on its own that the lit scan's absence does not block immediate progress on the cycle.

**S0-CAP-3 closes.** Three concrete next-spike candidates surface from this finding:

1. **Spike S0-CAP-4: Verify CAP-3 in OpenCode TUI.** The CAP-3 result was via `opencode run` (CLI). Run the same biased-prompt config through the actual OpenCode TUI session shape that PLAY used. Confirms the finding generalizes from CLI to TUI conditions.

2. **Spike S0-CAP-5: Biased prompt with mistral-nemo:12b.** Test whether the biased prompt clears the floor for the PLAY-failure model too. If yes, the prompt-as-fix is generalizable across the qwen and mistral families. If no, capability is still the binding constraint at lower tiers.

3. **Spike S0-DIAG-1: Surface the SummarizationFailure.reason.** Instrument the dispatch boundary to capture the full failure payload. The 4-retry behavior in CAP-3 is wasted compute; understanding the underlying failure is the first step toward fixing it.

### S0-CAP-5 — Biased prompt with mistral-nemo:12b (2026-04-28)

**Method:** Identical to CAP-3 except orchestrator profile switched back to `orchestrator-local` (mistral-nemo:12b — the model that catastrophically fabricated in PLAY). Same biased system prompt, same first-ask, same `opencode run` headless invocation.

**Observed behavior:**

User-ask session: 3 msgs, 10 client_tools, sys_prompt=yes (biased).

| Turn | Time | Tokens | Tool calls | Content |
|------|------|--------|------------|---------|
| 1 | 22:23:04 → 22:24:57 (~113s) | 16340 input | **0** | 721ch text — fabricated |

ReAct loop closed: stop with no tool calls. Total wall-clock: ~1m 53s.

**The full text response was a confidently-stated description of a non-existent ensemble (`ai-detect`)** including an invented invocation syntax (`/ai-detect README.md` — slash-command style, not llm-orc's tool surface). The model never called `list_ensembles` despite the biased prompt's explicit directive.

Verification: `find` for `ai-detect*` in the project, plus `llm-orc list-ensembles | grep -i detect` — confirms `ai-detect` is not a real ensemble in this llm-orc instance. The model confabulated it, possibly conflating Claude Code's `/ai-detect` skill (which DOES exist as a slash-command capability outside llm-orc's tool surface) with llm-orc's ensemble library.

**Classification: CAP-5-C (capability cliff persists).** The biased system prompt does NOT clear the floor for mistral-nemo:12b. The model produced content that *appears* to come from a `list_ensembles` call but no such call was made.

**Direct comparison across all qwen3 / mistral-nemo runs:**

| Run | Model | Prompt | Cascade engaged | Failure mode | Wall-clock |
|-----|-------|--------|-----------------|--------------|------------|
| PLAY | mistral-nemo:12b | default | partial (hallucinated) | fast-confabulation (project type fabricated) | 6 min |
| S0 | qwen3:14b | default | YES | slow-useful (with summarization failures) | 22 min |
| CAP-1 | qwen3:8b | default | NO (chose `skill`) | fast-giveup | 2 min 13s |
| CAP-3 | qwen3:8b | biased | YES (list_ensembles → invoke_ensemble × 4) | slow-useful (graceful text fallback) | 5 min 53s |
| CAP-5 | mistral-nemo:12b | biased | NO | fast-confabulation (fabricated `ai-detect`) | 1 min 53s |

**Three substantive findings cumulative across the spike battery:**

1. **Capability floor is family-specific, not parameter-count-driven.** qwen3:8b (8B) sits above the orchestration floor with proper prompting; mistral-nemo:12b (12B) sits below it even with the same prompting. Tool-use training quality dominates parameter count for this task class. Implication for RQ-1: parameter count is the wrong metric for "what local model can drive the cascade." Better metrics: tool-use benchmark scores (BFCL, ToolBench), training-time tool-use exposure, family-specific reliability data.

2. **Biased prompt is necessary but not sufficient.** Necessary: CAP-1 → CAP-3 difference (same model, only prompt changed, floor moved). Not sufficient: CAP-5 result (different model, same prompt, still fails). The biased prompt solves *surface confusion* at qwen3-tier capability. It does not solve *fabrication-as-failure* at lower-fidelity capability tiers.

3. **Empirically-grounded failure-mode taxonomy:**
   - **Slow-useful** (qwen3:14b S0; qwen3:8b CAP-3): cascade engages, model struggles through failures, eventually produces useful response. Latency cost is real but contract is satisfied.
   - **Fast-confabulation** (PLAY mistral-nemo; CAP-5 mistral-nemo): model produces confident-sounding fabricated content. Most insidious because it appears authoritative on superficial inspection. **A user/operator could easily believe fabricated output is real.**
   - **Fast-giveup** (CAP-1 qwen3:8b without biased prompt): model produces empty content quickly. Honest but unhelpful.

The taxonomy informs RQ-3 (observability — fast-confabulation is the failure mode operator-side surfaces most need to catch) and RQ-4 (intervention level — different failure modes require different defaults).

**Direct implication for the project:**

The current default `orchestrator-local` profile points at `mistral-nemo:12b`. This model is below the structural-fidelity floor regardless of prompt engineering. CAP-5 demonstrates that no amount of prompt steering rescues it for orchestration tasks. **The cheapest configuration intervention for product quality is to change the default `orchestrator-local` recommendation away from mistral-nemo:12b.** Candidate replacements at consumer-hardware scale:
- `qwen3:8b` (verified at H1'-internal in CAP-3 with biased prompt — works)
- `qwen3:14b` (verified at H1'-internal in S0 — works but slow on consumer hardware)
- Tool-tuned alternatives like `qwen2.5-coder:7b` (untested in this cycle, but qwen2.5 family has strong tool-use benchmarks)

This is a load-bearing finding for any DECIDE work on default config policy. *The default model recommendation matters more than the default system prompt.* Both should change; the model change is the more consequential.

**S0-CAP-5 closes.** Combined with CAP-1, CAP-3, this gives a clean 2×2 matrix of (model × prompt) and shows:
- qwen3:8b + biased prompt = works
- qwen3:8b + default prompt = fails (silent giveup)
- mistral-nemo:12b + biased prompt = fails (confabulation)
- mistral-nemo:12b + default prompt = fails (confabulation, per PLAY)

The prompt-as-fix is *qwen3-family-specific or capability-tier-conditional*. It is not a universal lever.

**Next-spike candidates surfaced from CAP-5:**

- **CAP-6: Test additional families.** Try `deepseek-r1:8b`, `qwen2.5-coder:7b` (if pulled), or other locally-available models with the biased prompt. Build the family-specific capability map.
- **CAP-7: Test smaller qwen3 variants.** `qwen3:1.7b` is locally available — does it sit above or below the floor with the biased prompt? Tests how far down the capability gradient the prompt-as-fix carries.
- **DIAG-1: Summarization failure root cause.** Still outstanding. The cascade engagement at qwen3:8b in CAP-3 surfaced 4 consecutive summarization_failed events. Whatever's wrong with the code-review ensemble's summarization is product-quality blocking.
- **CAP-8: Confabulation detection.** Given that fast-confabulation is the most insidious failure mode, can the operator-side observability surface it specifically? E.g., flag responses where the model claims to have queried state without an actual tool dispatch event in the log. This is a structural detection rather than a model fix.

### S0-DIAG-1 — Summarization failure root cause (2026-04-28)

**Method:** Patched `OrchestratorToolDispatch.dispatch` to surface `ToolCallError.reason` on result lines (not just the kind), with both INFO logger and SPIKE-S0 print equivalents. Re-fired CAP-3's exact ask.

**Captured reason:**

```
"summarizer ensemble 'agentic-result-summarizer' returned no summary text
 (checked synthesis and single-agent response)"
```

This is `_extract_summary` failure path #3 in `result_summarizer_harness.py:107-114` — the summarizer ensemble was *invoked successfully* but its own output had neither populated `synthesis` nor a single-agent `response` field. The summarizer ran but produced empty extractable text.

**Root cause traced to missing Ollama model.**

Direct invocation of `agentic-result-summarizer` via `llm-orc invoke` (bypassing the tool-dispatch path) produced a clean error message:

```
summarizer (failed) (summarizer):
Error: model 'qwen3:0.6b' not found (status code: 404)
```

The summarizer model_profile is defined in the project-tier config at `.llm-orc/config.yaml:` with `model: qwen3:0.6b`. The intent (per the YAML comment) was a small/fast model so summarization stays cheap. But `qwen3:0.6b` was not pulled into the practitioner's local Ollama instance — only `mistral-nemo:12b`, `qwen3:14b`, `qwen3:8b`, `mistral:7b`, `qwen3:1.7b`, `gemma3:1b`, `deepseek-r1:8b`, `nomic-embed-text`, `llama3:latest` were available.

**The cascade was structurally correct but operationally broken on a missing model dependency.** Ollama returned 404, the summarizer ensemble's agent produced empty output, the harness saw "no summary text", every `invoke_ensemble` call failed with the generic-looking `summarization_failed` error.

**Findings beyond the immediate fix:**

1. **Generic error types hide actionable diagnosis.** S0 and CAP-3 both reported `summarization_failed` as a generic error kind. The actionable signal — "your `summarizer` profile points at an unpulled model" — was buried in a string field (`reason`) that the operator-side log was not surfacing. RQ-3 finding: error-type-plus-reason is the minimum useful operator diagnostic. Just the kind is not enough.

2. **Default config bundles assume specific local models.** The project-tier `.llm-orc/config.yaml` ships with profile definitions that require specific Ollama model availability. New deployments will silently break on the same path until they pull all the referenced models. Possible RQ-4 finding: the install/setup path should either (a) pull required models automatically, (b) validate model availability at startup with a clear error, or (c) provide clear pre-flight documentation.

3. **The orchestrator's failure-handling persistence was productive.** In CAP-3 (before the fix), qwen3:8b retried `invoke_ensemble` 4 times before giving up gracefully with text. This wasn't pointless — given an opaque error, retry was a reasonable strategy, and the model eventually adapted. With richer error info in the operator log, future versions of the orchestrator could potentially recover by adapting based on the reason field — though that would require teaching the orchestrator to read errors more carefully than current LLMs typically do.

4. **Hallucinated arguments observed.** Turn 6 of CAP-3 emitted `arg_keys=['input', 'name', 'skip_summarizer']`; turn 7 emitted `['force_output', 'input', 'name', 'skip_summarizer']`. These are NOT real `invoke_ensemble` arguments. The model was inventing parameters trying to find a workaround. Tool Dispatch silently ignores unrecognized fields. RQ-3 finding: orchestrator argument fabrication is invisible to the architecture today; could be flagged at dispatch time.

**Fix applied: `ollama pull qwen3:0.6b` (~520MB).**

The summarizer model is now available. The cascade should now complete summarization successfully.

### S0-CAP-3b — Full cascade end-to-end (2026-04-28)

**Method:** Identical to CAP-3 except summarizer's `qwen3:0.6b` is now pulled. Ran CAP-3's exact ask via `opencode run`.

**Cascade trajectory** (turns numbered cumulatively across the same session ID):

| Turn | Tool | Result |
|------|------|--------|
| 8 | `list_ensembles()` | success |
| 9 | `invoke_ensemble(name='code-review', input=...)` | **success** (summarizer worked) |
| 10 | (synthesis, 421ch text response) | loop close: stop |

**OpenCode response (decisive):**

> *"The `code-review` ensemble analyzed the README and identified potential security (OWASP vulnerabilities) and performance optimizations. However, it noted limitations without explicit code examples. To proceed: 1. **Security**: Review dependencies for OWASP Top 10 compliance. 2. **Performance**: Optimize API endpoints via caching/async patterns. Would you like to analyze specific code files for deeper insights?"*

**Decisive finding: full agentic-serving cascade completes end-to-end and produces useful output at qwen3:8b on consumer hardware.**

This is the **cycle's first observation** of all of:
1. Internal cascade engaged (qwen3:8b, biased prompt)
2. `list_ensembles` returned the library catalog
3. `invoke_ensemble` succeeded (not `summarization_failed`)
4. Result Summarizer Harness produced a real summary (the now-working `qwen3:0.6b` summarizer)
5. Orchestrator synthesized that summary into a useful prose response
6. User contract satisfied with substantive output

**The project hypothesis is now empirically validated at this hardware tier.** Orchestration with non-frontier models (qwen3:8b for orchestrator + qwen3:0.6b for summarizer + library ensembles for specialized work) delivered useful output for the user-contract task class.

The structural integrity demonstrated includes the Result Summarizer Harness path (AS-7, ADR-004, WP-D) which had only ever fired in failure mode in this cycle until now.

### Lit-review return (2026-04-28)

The dispatched `rdd:lit-reviewer` returned with substantial material at `docs/agentic-serving/essays/research-logs/lit-review-capability-floor-and-observability.md` (289 lines). Summary of findings most relevant to the cycle:

**External validation of cycle empirical findings (BFCL leaderboard, Patil et al. ICML 2025):**

- Qwen3-14B F1 = **0.971**; Qwen3-8B F1 = **0.933** (both strong in the ≤14B class)
- xLAM-8B (tool-calling specialist) F1 = **0.570**
- Watt-tool-8B (tool-calling specialist) F1 = **0.484**
- **The cycle's qwen3:8b-clears-the-floor finding has external benchmark validation.** Specialist tool-calling models in the same parameter class are substantially weaker per the BFCL methodology.

**Cycle-original signals confirmed by literature absence:**

- **Fast-confabulation is unmeasured** in BFCL or any other public benchmark. The cycle's CAP-5 finding (mistral-nemo:12b confidently asserting tool calls that didn't happen) sits in literature's blind spot. This is original signal.
- **System-prompt-vs-tool-description has no head-to-head study.** CAP-3's empirical finding that system prompt dominates is original at this specificity.
- **VILA-Lab Claude Code architecture paper (arXiv 2604.14228)** explicitly names *"Silent Failure and Observability-Evaluation Gap"* as open research direction. RQ-3's framing has external validation.

**Multi-agent orchestrator pattern overhead grounded:**

- LangGraph supervisor adds >30% latency overhead per documented framework characteristics
- AutoGen GroupChat costs one full LLM call per agent turn
- CAP-2's measured 2.54× overhead is consistent with documented framework overheads
- Lit recommends structural splits *only* after cheap interventions exhausted — same conclusion CAP-3 reached empirically

**Beck's series is frontier-model-only.** Doesn't address local models, capability gradients, or default-experience design. The cycle's RQ-1 through RQ-4 are outside Beck's frame of reference. Beck's contribution is the framing tension (RQ-5); the cycle's contribution is the local-first empirical grounding.

**Spike-actionable model recommendations from Gap 6:**

| Model | `ollama pull` | Size | Notes |
|-------|---------------|------|-------|
| qwen3.5:9b | `qwen3.5:9b` | 6.6 GB | Newest qwen (March 2026); low confabulation risk by family lineage |
| qwen3.5:4b | `qwen3.5:4b` | 3.4 GB | Tests if qwen3.5 improvements move the floor below 8B |
| deepseek-r1:8b | already pulled | 5.2 GB | Reasoning-before-action architecture; possibly lower confabulation risk; **untested in this cycle** |
| xLAM-2 8B | `robbiemu/Salesforce_Llama-xLAM-2:8b-fc-r-q5_K_M` | community quant | Tool-calling specialist; BFCL F1 0.570 — concerning; possibly skip |

**Recommendation interpretation:** the lit reviewer surfaced xLAM-2 with explicit caveat about its weak BFCL score. Skipping xLAM-2 in favor of qwen3.5:9b and the already-pulled deepseek-r1:8b is the higher-signal use of remaining spike budget.

**Status of the cycle as of 2026-04-28:**

The cycle has accumulated decisive empirical findings backed by external literature validation. The full agentic-serving architecture has been observed succeeding end-to-end (CAP-3b). The capability floor has been mapped at the family/parameter level. Multiple failure modes (slow-useful, fast-confabulation, fast-giveup, summarization-broken) are taxonomized. The dual-contract framing (RQ-5) has empirical grounding (CAP-3b shows convergence; CAP-5 shows divergence under capability cliff).

**Three remaining productive moves before essay drafting:**

1. **CAP-7: qwen3.5:9b validation.** The lit review names qwen3.5 as a recent release with stronger tool-use lineage. Pull and test with biased prompt + same ask. Confirms whether the qwen family's tool-use trajectory continues. Low effort with `opencode run`.
2. **CAP-8: deepseek-r1:8b validation.** Already pulled, no download needed. Tests whether reasoning-before-action architecture changes the failure-mode profile. Possibly resists confabulation differently than qwen3.
3. **Orientation regeneration.** Cycle artifacts are now substantively different from the last `ORIENTATION.md` regeneration. Worth refreshing the top-of-cycle orientation document before essay drafting.

After those, the cycle has enough material to draft essay 002 — a coherent narrative grounded in empirical findings and external literature.

### S0-CAP-8 — deepseek-r1:8b unsupported (2026-04-28)

**Method:** Configure orchestrator profile to `deepseek-r1:8b` (already pulled), restart server, fire same biased-prompt ask via `opencode run`.

**Result: failed at the Ollama API boundary.**

Server log captured the underlying error from `openai_compat.py:131`:

```
RuntimeError: OpenAI-compatible tool-calling API error 400:
{"error":{"message":"registry.ollama.ai/library/deepseek-r1:8b does not support tools",
          "type":"invalid_request_error","param":null,"code":null}}
```

**Ollama refuses tool-calling for deepseek-r1:8b at the protocol level.** The model has no tool-calling capability exposed via Ollama's OpenAI-compat endpoint. The orchestrator runtime called `generate_with_tools`, the HTTP request reached Ollama, and Ollama rejected with 400. OpenCode then retried the same request multiple times before timing out.

**Three substantive findings from this failure:**

1. **`supports_tool_calling` is class-level, not model-specific.** llm-orc's `ModelInterface` exposes `supports_tool_calling` at the *class* level — `OpenAICompatibleModel` claims True for any Ollama-served model. This is correct *as a capability claim about the wrapper class* but doesn't reflect Ollama's per-model variation. **Effective implication:** an operator can configure ANY Ollama model as orchestrator profile, the session-start check passes, and the cascade fails on first turn with 400. RQ-4 finding: pre-flight validation should query Ollama (or whatever provider) for actual per-model tool-calling capability before accepting session start.

2. **Lit-review recommendations need API-availability checks.** The lit reviewer suggested deepseek-r1:8b as possibly lower confabulation risk per reasoning-before-action architecture. The recommendation was theoretically sound but the model is unavailable for orchestration via Ollama tool-calling. Empirical spike work catches this; pure literature analysis misses it. RQ-1 finding: capability evaluation in the literature does not account for deployment-platform availability of tool-calling protocols.

3. **The retry pattern is wasted compute.** OpenCode retried the same request multiple times against an Ollama endpoint that will never accept it. Each retry consumes network + server-side processing. RQ-3 finding: a *fast-fail*-with-clear-error path is preferable to silent retry against impossible targets. The orchestrator could surface the 400 cleanly back through SSE so the client knows not to retry.

**CAP-8 closes (negatively).** deepseek-r1:8b is not a viable orchestrator option in this deployment. Reasoning-architecture confabulation-resistance hypothesis remains untested for *this family* via this deployment path. Untested via this cycle's empirical method.

**Implication for the project's default config story:** any default model recommendation that ships must be verified to actually expose tool-calling via the operator's deployment platform. *Model-name-in-config* is not enough; the system should validate at session start that the *specific model* supports tools, not just the *provider class*. This is a small but real ADR-011 / ADR-002 corollary.

### S0-CAP-7 — qwen3.5:9b premature stop (2026-04-28)

**Method:** Pull `qwen3.5:9b` (6.6 GB), add to `~/.config/llm-orc/config.yaml` as `orchestrator-qwen3.5-9b`, switch active orchestrator profile, restart server, run same biased-prompt ask via `opencode run`.

**Observed cascade:**

| Turn | Tokens | Content | Tool calls | Result |
|------|--------|---------|------------|--------|
| 1 | 16700 | 65ch ("I'll check what ensembles are available in this llm-orc instance.") | `list_ensembles` | success |
| 2 | 20340 | empty | none | **loop close: stop** |

Total wall-clock: ~3-4 minutes.

**OpenCode user-facing response: just the turn-1 preamble.** No final synthesis. The model abandoned the task after one tool call, before invoking any ensemble.

**Surprising relative to lit-review prediction.** Lit review (Gap 6) flagged qwen3.5:9b as strongest candidate from the qwen family lineage based on family-trajectory and BFCL-class benchmarks. Empirically, on this single trial with this exact setup, qwen3.5:9b produced a *worse* user-contract outcome than qwen3:8b had in CAP-3b. Specifically:

- Both correctly engaged the internal tool surface (`list_ensembles` first)
- qwen3:8b (CAP-3b) continued the cascade through `invoke_ensemble` and produced a 421-char substantive answer
- qwen3.5:9b (CAP-7) stopped after one tool call and produced no answer

**Possible interpretations (not distinguishable from a single trial):**

1. **One-shot variance.** Stochastic generation; a re-run might cascade further. Worth a deterministic re-run to test reproducibility.
2. **Output-length tuning differences.** qwen3.5 may have different stop-token bias, weight on conciseness, or different early-stop behavior baked in by training.
3. **"Task complete" overconfidence.** Qwen3.5 may have reasoned "I've queried the system, the user can see the list, my job is done" — incorrectly judging the request as a query when it actually asked for invocation + demonstration.
4. **Different reaction to the biased prompt.** The system prompt was crafted against qwen3:8b's behavior in CAP-1; qwen3.5:9b may interpret the directives differently.

**Cycle-level finding: capability gradient is not monotonic.** Newer-and-larger does not strictly mean better-for-this-task. CAP-3b's qwen3:8b outperformed CAP-7's qwen3.5:9b on the same ask with the same setup. This challenges the simplistic "pull the newest, biggest model" intuition that lit-review-based recommendations might suggest. **Empirical spike work catches what literature predicts wouldn't happen.**

**Recommendation:** before drawing strong conclusions, re-run CAP-7 to test whether the premature-stop is reproducible. If reproducible, qwen3.5:9b is genuinely worse for this task class than qwen3:8b in this configuration. If not reproducible, it's variance and qwen3.5:9b is at least as good on average.

**S0-CAP-7 closes (provisionally).** A re-run would harden the finding. Even at n=1, the result is unexpected enough to be worth recording.

### S0-CAP-7-rerun — Reproduces the premature stop (2026-04-28)

**Method:** Identical to CAP-7. Same model, same prompt, same ask.

**Result:** Identical pattern.

| Turn | Content | Tool calls | Result |
|------|---------|------------|--------|
| 3 (cumulative session) | 64ch preamble | `list_ensembles` | success |
| 4 | empty | none | loop close: stop |

User-facing OpenCode response: *"I'll list the available ensembles first to see what's available."* — then no further synthesis.

**Conclusion: the premature-stop is a stable, reproducible property of qwen3.5:9b on Ollama with the biased prompt + the cycle's tool surface composition.** Not variance.

**Likely root cause: Qwen GitHub issue #1831 — chat-template bug.** Public web search surfaces an active issue in QwenLM/Qwen3: *"fix chat template for Qwen 3.5 — fixes tool calling crash, parallel calls, thinking bleed."* The issue describes known chat-template bugs specifically affecting qwen3.5 tool-calling, including failure modes consistent with what CAP-7 observed. Whether Ollama's current qwen3.5:9b packaging has the fix or not is not directly verifiable through the spike but the symptom-cause match is strong.

**Other contributing or alternative factors:**

- Qwen3.5 family has thinking mode disabled by default (per Qwen docs). Multi-turn behavior in non-thinking mode may differ from qwen3.
- Recommended generation parameters (`temperature=1.0, top_p=1.0, presence_penalty=2.0, repetition_penalty=1.0`) are unusual; Ollama defaults likely don't match. `presence_penalty=2.0` in particular biases toward novel content / against repetition, which could affect the model's willingness to continue after producing a preamble.
- Stop-token behavior may differ between qwen3 and qwen3.5 family.

**Cycle-level finding: capability gradient is non-monotonic AND deployment-stack-dependent.**

S0-CAP-1 → CAP-3 → CAP-3b → CAP-5 → CAP-7 → CAP-8 cumulatively show:

1. **Newer-larger does not strictly mean better-for-task.** qwen3:8b (CAP-3b) outperformed qwen3.5:9b (CAP-7+rerun) on the same ask with the same setup.
2. **Per-model deployment support varies.** deepseek-r1:8b has no tool-calling support exposed via Ollama (CAP-8), independent of capability.
3. **Inference-engine chat-template bugs materially affect outcomes.** qwen3.5:9b's premature-stop is consistent with a documented chat-template bug in Ollama's packaging.
4. **Lit-review predictions need empirical validation.** Pure family-lineage / benchmark-score analysis would have recommended qwen3.5:9b as a primary candidate; empirical reality refutes that recommendation in this deployment.

**Empirical recommendation crystallizes: qwen3:8b with biased prompt + summarizer model (`qwen3:0.6b`) pulled is the validated default for consumer-hardware deployment in this cycle.** Not what literature would naively predict; not the newest model; the empirically-validated combination.

**S0-CAP-7-rerun closes.** Spike battery (S0, CAP-1/2/3/3b/5/7/7b/8) is sufficient. Lit review complete. The cycle has the material to draft essay 002 with confidence in the findings.

### Pre-S0 finding — Orchestrator-as-single-Model-Profile is an unexamined commitment (2026-04-27)

**Question (surfaced by practitioner):** Is the orchestrator itself an ensemble?

**Method:** Re-read ADR-011 and system-design.md §Orchestrator Runtime.

**Findings:**

- **Current design: orchestrator is a single LLM.** ADR-011 commits explicitly: "The orchestrator agent's LLM is configured via a standard Model Profile. ... The orchestrator itself runs on one model profile per session." system-design.md §Orchestrator Runtime corroborates: "Runs the ReAct loop that delegates to llm-orc operations via a fixed tool surface."
- **Ensemble-style orchestration lives one layer below.** ADR-011 expresses tiered behavior (e.g., local-triage-then-escalate-to-cloud) as a composed ensemble invokable by the orchestrator — `invoke_ensemble("triage-route")` — not as a property of the orchestrator role itself.
- **ADR-011's stated rationale for single-profile orchestrator:** uniformity (same mechanism as everywhere else), tool-calling format coherence (current APIs return one structured `tool_calls` per LLM call; ensemble synthesis to that format is not designed), ReAct loop tightness (latency multiplies with LLM hops per turn), configuration simplicity at entry point.
- **The negative consequence ADR-011 itself records is exactly the tension:** "the orchestrator itself cannot dynamically switch LLMs mid-session. This is intentional: LLM swap at the orchestrator level is a session-boundary event, not a runtime decision."

**Implications:**

- **Internal consistency tension surfaced.** The project's value proposition (essay 001 §Cost implications, FF #4, FF #7) is that orchestrated ensembles outperform single LLMs through coordination. Yet the most cognitively demanding role — the orchestrator — is the one role where the design commits to a single LLM. The hypothesis under test (non-frontier orchestration delivers good results) is tested everywhere except where it most matters: the orchestrator role itself.
- **Unexamined-artifact watch item promoted.** The methods-review report flagged ADR-001/002 as unbracketed; ADR-011 was not flagged but the practitioner's question exposes that it should have been. The constraint-removal exercise asked what would change if ADR-003's five-tool surface were not available; an analogous exercise on ADR-011 would ask: *what would change if the orchestrator were not constrained to a single Model Profile?*
- **Relevant to RQ-1's affordance dimension.** The amended RQ-1 asks about "composability primitives the orchestrator is given." A directly adjacent question: *what composability primitives is the orchestrator role itself allowed to participate in?* Current answer: none. The orchestrator is the only llm-orc role that cannot be composed.
- **Relevant to RQ-5 (dual-contract framing).** The project contract (orchestration delivers good results from non-frontier models) is structurally untested at the orchestrator level by the current design. RQ-5 should surface this as part of the seam navigation.
- **Boundary for S0.** S0 still tests the current design (single-LLM orchestrator with cascading ensembles below it). Testing orchestrator-as-ensemble is a separate downstream spike requiring an ADR amendment or a new ADR. Recording the finding for future cycles or a possible mid-research constraint-removal pass on ADR-011.

**Open question for the cycle:** should ADR-011 be bracketed in a follow-up constraint-removal pass before S0 fires, or is it appropriate to test the current design (single-LLM orchestrator) first and let S0's findings inform whether ADR-011 warrants reopening? The methods-reviewer pattern is "test what's there first, then re-examine"; this question records the option without committing.

### Pre-S0 finding — Operator-side logging surface is essentially empty (2026-04-27)

**Question:** What does `uv run llm-orc serve` actually log by default? (PLAY field note: operator terminal was silent during the four-turn session.)

**Method:** Source survey of `src/llm_orc/web/` and `src/llm_orc/agentic/` for `logger` and `logging.` references.

**Findings:**

- `src/llm_orc/web/server.py:27` declares a single module-level logger.
- `src/llm_orc/web/server.py:77` uses it once — to log unhandled errors via FastAPI's exception handler.
- *No logger references anywhere in `src/llm_orc/agentic/`* — the entire orchestrator runtime, tool dispatch, ensemble invocation, calibration, autonomy, and budget enforcement code paths emit no log lines by design.
- FastAPI's own access log behavior on `uv run llm-orc serve` is the operator's only default surface (request line + status code).

**Implications:**

- The PLAY observation that "no log activity in the `uv run llm-orc serve` terminal" is fully explained: there is nothing to surface because the L1/L2/L3 modules do not emit logs at all. This is not a misconfiguration; it is a structural gap. The bilateral observability gap (FF #129) on the operator side is *load-bearing on this absence*, not on log-level configuration.
- For RQ-3: the operator-side question is not "what level should logs be at" but "what events should the system log at all." The current answer is "essentially none." This makes RQ-4's policy question more pointed: shipping a default that produces a coherent operator experience requires at least *deciding what to log*.
- For S0: the spike will run with `--log-level debug` to surface whatever can be surfaced from FastAPI plus the one error-handler call site. Beyond that, S0 will need a small temporary instrumentation patch (request entry/exit, tool dispatch, ensemble invoke) to produce useful operator-side observation. This is a Beck-style "smallest viable instrumentation" patch — disposable, in the spike's worktree, not a production surface.
- For OB-2/OB-3 (subsequent spikes): the operator-log surface is greenfield. There is no existing pattern to mimic or extend; the design is unconstrained by prior choices. This may simplify the spike or complicate it, depending on whether existing patterns elsewhere in `llm_orc` (the broader project, not `agentic/`) have a convention worth adopting.



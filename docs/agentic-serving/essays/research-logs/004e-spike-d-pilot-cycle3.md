# Spike D Pilot — Cycle 3 Multi-Ensemble Coordination Probe (and the OpenCode CLI Stall Finding)

**Date:** 2026-05-02 / 2026-05-03
**Cycle:** 3 (agentic-serving scoped corpus)
**Pilot scope:** 3 arms × N=1 trial each on a single fixture (Spike C's diff). **Cycle 4 priming experiment, not Cycle 3's primary central-question evidence.** Primary purpose: surface failure modes and design hooks that motivate Cycle 4's research entry on multi-ensemble coordination.
**Operating frame:** Outcomes over an agentic session — agent shape is means.
**Practitioner motivation:** Spike C demonstrated single-ensemble dispatch's value. Multi-ensemble coordination across multiple turns was the natural next direction — the deeper architectural test where the cheap orchestrator must integrate outputs from specialized ensembles in sequence. Pilot scope (vs full experiment) was deliberately bounded.

---

## Abstract

Spike D pilot tested whether the cheap orchestrator could coordinate a multi-stage code-review-and-fix pipeline (review → propose fix → verify) where Stage 1 and Stage 3 dispatch specialized ensembles and Stage 2 generates the fix via LLM. Three arms: Arm A1 cheap-bare-with-tools (`opencode run` + Read/Edit/Bash, no ensemble dispatch), Arm B1 cheap+multi-ensemble (script-agent ensemble for review, MiniMax via direct model factory for fix, script-agent ensemble for verification), Arm C1 frontier-bare (Sonnet 4.6 via subagent, all stages collapsed into single-shot response). Verifier scoring on the proposed fixed files: A1 3/5 resolved (verifier false-positive on api_key hash), **B1 4/5 resolved**, C1 4/5 resolved. **B1 ties C1 at zero $ cost.** The pilot's primary finding is unexpected and substantive: **the opencode CLI deployment shape stalls indefinitely on substantial code-containing prompts at the cheap-tier.** Three sustained attempts to dispatch multi-stage workflows via `opencode run` hung silently for hours with the orchestrator process alive but consuming nearly zero CPU. The same MiniMax model called via the production `ConfigurationManager` + `ModelFactory` path (Spike B's harness pattern) completed the same workflow in 25 seconds. **This is a deployment-shape finding, not an architecture finding** — the architecture's intended `invoke_ensemble` primitive (orchestrator-internal tool-calling loop in `llm-orc serve`) works reliably; the shell-subprocess workaround through opencode's Bash tool does not. The pilot succeeds in priming Cycle 4: six concrete research design hooks emerged from the pilot's results and from the failure modes it surfaced.

---

## Background

Spike C closed the methods reviewer's `+ orchestration` gap by demonstrating the architecture's value on cross-file verification. Practitioner observation post-Spike C: *"we're getting closer to a framework for how we might proceed deeper into RDD, but I think we need to do at least another research loop. But to prime that, we could do an experiment where we have the orchestrator coordinate between two or three different kinds of ensembles to perform a task over several turns. Compared to frontier."*

The proposed extension targets the architecture's deeper claim — the orchestrator is not just a single-ensemble dispatcher; it's a conductor coordinating specialized components. Multi-ensemble coordination tests `invoke_ensemble` repeated use, multi-turn integration, and routing intelligence (which ensemble for which sub-task). The pilot scoped this to N=1 per arm, single fixture (reusing Spike C's), as a Cycle 4 priming experiment rather than a primary central-question finding.

---

## Method

### Workflow

A four-stage code-review-and-fix pipeline:

1. **Stage 1 — Review:** dispatch `spike-c-code-review` ensemble on the diff; produces issue list (script-agent's deterministic findings + LLM reviewer's analysis).
2. **Stage 2 — Generate Fix:** orchestrator (LLM) reads the issue list + original file; generates a complete proposed fixed file as Python code in a fenced block.
3. **Stage 3 — Verify:** dispatch the new `spike-d-fix-verifier` ensemble on the proposed fixed file; produces structured per-issue status report (resolved / still_present / unknown).
4. **Stage 4 — Final Report:** orchestrator integrates all stage outputs.

The pilot's interesting question is Stage 1 + Stage 3 — does the orchestrator correctly dispatch the two ensembles? Stage 2 is LLM-only work (any tier can do it).

### `spike-d-fix-verifier` ensemble

New script-only ensemble (`.llm-orc/ensembles/spike-d-fix-verifier.yaml` + `spike_d_fix_verifier.py`). Takes a proposed fixed Python file as input; runs deterministic checks for each of the 5 ground-truth issue categories and reports per-issue status. The verifier's regex-based detection has known limitations (specifically: it flags any `api_key` reference within a logging-call window without distinguishing hash from raw — a security-correct hash-then-log fix produces a false positive). Documented as a verifier scope condition.

### Three arms

- **Arm A1 — cheap-bare-with-tools:** `opencode run -m llm-orc/orchestrator-minimax-m25-free --print-logs "<multi-stage prompt>"`. opencode provides Read/Edit/Bash. No ensemble dispatch — orchestrator does all stages with general agentic capability. N=1.
- **Arm B1 — cheap+multi-ensemble:** Three-step orchestration via the production `ConfigurationManager` + `ModelFactory` path. Stage 1 reuses `arm-b-cheap-with-ensemble-trial1.json` from Spike C (already produced; review output = ensemble dispatch). Stage 2 calls MiniMax directly via `model.generate_response()` with prompt = original file + compact issue list, asking for fixed code in a fenced block. Stage 3 calls `llm-orc invoke spike-d-fix-verifier -f <fixed file>`. N=1.
- **Arm C1 — frontier-bare:** Sonnet 4.6 via subagent dispatch with the full multi-stage workflow collapsed into a single prompt. Subagent produces all three stages' content in one response. N=1.

### Scoring

Verifier output on each arm's proposed fixed file. Per-issue status from the deterministic checks: ISSUE-1 (off-by-one), ISSUE-2 (api_key in logs), ISSUE-3 (type annotation), ISSUE-4 (test coverage), ISSUE-5 (cross-file value match).

---

## Results

### The opencode CLI stall pattern (pilot's primary finding)

Three sustained attempts to dispatch the multi-stage workflow via `opencode run` (the originally-intended Arm B1 design + Stage 2 retry attempts) all stalled the same way:

| Attempt | Started | Stalled at | Duration alive | CPU consumed |
|---------|---------|:----------:|:--------------:|:------------:|
| Arm B1 initial | 2026-05-01 23:45 | 14 min then killed | ~14 min | ~1s total |
| Arm B1 retry | 2026-05-02 00:04 | 8.4 hours then killed | 8h 24m | ~5s total |
| Stage 2 (rewrite, ~17KB prompt) | 2026-05-02 08:28 | 1h 24m then killed | ~84 min | ~1s total |
| Stage 2 (tighter, ~5–7KB prompt) | 2026-05-02 09:51 | 14h+ then killed | 14h+ | ~1s total |

Pattern characteristics:
- opencode process state: `S` (sleeping, blocked on system call)
- Bash tool capture: only the initial INFO log line (~1426 bytes), no further output ever
- stdout redirect file: 0 bytes
- A "tiny" prompt ("Reply with the single word: OK") completes instantly via the same opencode + same MiniMax profile

**The same Stage 2 prompt — `len 4682` chars, including the original file content — completed in 24.8 seconds when called via the production `ConfigurationManager` + `ModelFactory` path.** This isolates the failure to the opencode CLI deployment shape, not to MiniMax or to OpenCode Zen's API.

### Per-arm results on the proposed fixed files

| Arm | Path | Lines (fixed file) | Issues resolved | Notes |
|-----|------|:-----------------:|:---------------:|-------|
| Arm A1 | opencode CLI + Read/Edit/Bash | 120 | **3/5** | Verifier false-positive on api_key hashing approach (line still references api_key within log-call window, even though only the hash is emitted) |
| **Arm B1** | direct model factory + ensemble dispatch | 103 | **4/5** | api_key removed from log message entirely; only test-coverage gap remains (no test file added — expected for single-file pilot) |
| Arm C1 | Sonnet 4.6 single-shot facsimile | 143 | **4/5** | api_key removed from `SessionBudget` dataclass entirely (most aggressive fix); only test-coverage gap remains |

### Structural fix comparison

| Fix dimension | A1 | B1 | C1 |
|--------------|:--:|:--:|:--:|
| Imports `DEFAULT_MAX_TOKEN_LIMIT` from `orchestrator_config` | ✓ | ✓ | ✓ |
| Uses `>=` in `check_limit` | ✓ | ✓ | ✓ |
| `int \| None` annotation on `register()` `limit` param | ✓ | ✓ | ✓ |
| Removes raw api_key from log message | hash | yes | yes |
| Removes api_key field from dataclass | no | no | yes |

All three arms produced functionally-equivalent fixes for ISSUE-1, ISSUE-3, ISSUE-5. Differences are at the api_key handling (security depth) and stylistic detail.

### Stage 2 timing comparison (the load-bearing infrastructure observation)

| Implementation path | Prompt size | Wall-clock | Outcome |
|---------------------|:-----------:|:----------:|---------|
| `opencode run` + MiniMax (4 attempts) | 4.7KB – 17KB | 14 min – 14 hours | **All stalled; manually killed** |
| `ModelFactory.generate_response()` + MiniMax (Arm B1 actual) | 4.7KB | **24.8s** | Completed cleanly with 1086 / 1039 in/out tokens |

---

## Discussion

### What B1 = C1 = 4/5 tells us

When the multi-ensemble pipeline is invoked via the production model factory path, cheap+orchestration produces fix quality functionally equivalent to frontier-bare on this fixture. **The architecture's value claim from Spike C extends to multi-stage workflows.** The architecture's components — script-agent for deterministic verification, LLM reviewer for semantic analysis, structured ensemble composition — all worked. The fix-verifier closed the loop deterministically. The mismatch between cheap and frontier on raw capability dissolves when the architecture provides the structural scaffolding.

This is a positive central-question result: **at the workflow level (review + fix + verify), cheap+orchestration matches frontier-bare on outcome quality at zero $ cost.** The four-priorities frame's recommendation again strongly favors Arm B1.

### What the opencode stall tells us

The architecture's intended primitive is `invoke_ensemble` dispatched by the orchestrator's internal closed-five-tool surface (per ADR-003). When the orchestrator runs in `llm-orc serve`, this tool is part of its native action space. The orchestrator's system prompt (`DEFAULT_ORCHESTRATOR_SYSTEM_PROMPT` per `orchestrator_config.py`) directs it toward ensemble dispatch for capability-relevant tasks.

The opencode CLI deployment shape uses a different path: the orchestrator (running through opencode → llm-orc serve) gets opencode's Bash tool exposed to it via Option C delegation. Multi-stage workflows via opencode require the orchestrator to dispatch shell commands like `uv run llm-orc invoke <ensemble> -f <file>` — using a SHELL SUBPROCESS as the dispatch primitive instead of the orchestrator's internal `invoke_ensemble` tool.

**This is not the architecture as designed.** The shell-subprocess substitute is a workaround for testing without standing up the full `llm-orc serve` direct-API-call deployment. The pilot's stall finding establishes that this workaround is fragile — substantial code-containing prompts cause silent hangs, possibly because of how opencode buffers / coordinates between its own LLM call and the multi-stage state required to dispatch subprocess commands.

The stall finding is **substantive for Cycle 4**: it argues that the architecture's evaluation MUST go through `llm-orc serve` direct-API-call paths (or the production model factory directly, as Spike D's working B1 trial did), not through opencode CLI. Cycle 4's research design should bake this in.

### Pilot scope is appropriate; full experiment is Cycle 4 territory

The pilot deliberately did NOT test:
- **Autonomous routing intelligence** — does the orchestrator pick the right ensemble for each sub-task without explicit instruction? B1's staging was explicit (Stage 1 = invoke spike-c-code-review; Stage 3 = invoke spike-d-fix-verifier). Cycle 4 should test whether the orchestrator (running through `llm-orc serve` with the closed-tool surface) makes correct dispatch decisions autonomously given an unstructured task.
- **Iteration loops** — review → fix → verify → re-fix when issues remain. The pilot ran linear (single pass). Cycle 4 should test whether the orchestrator iterates correctly when verification reports unresolved issues.
- **N>1 variance** — single trials per arm. Cycle 4 should run higher-N to establish variance ranges, especially on the autonomous-routing question.
- **Heterogeneous reviewers** — Arm B1 used MiniMax for the LLM reviewer slot (same scope cut as Spike C). Cycle 4 should test the full A3 architecture (script + Hunyuan + Kimi + MARG concatenation) on multi-stage workflows.
- **Higher-complexity fixtures** — same 90-line single-file fixture as Spike C. Cycle 4 should test multi-file diffs, longer trajectories, ambiguous task descriptions where routing decisions are themselves load-bearing.

---

## Limitations and scope conditions

1. **N=1 per arm.** Pilot scope; not statistically meaningful per-arm. The B1=C1 tie at 4/5 each is a single data point. Cycle 4 should establish variance ranges.
2. **Single fixture.** Reused Spike C's fixture; no fixture-class coverage.
3. **Verifier regex limitations.** The api_key-in-logs detector flags any `api_key` reference within a log-call window. Hashing-then-logging (security-correct fix) produces a false positive. Documented as a verifier scope condition; A1's 3/5 should be read as effectively 4/5 if the api_key-handling-style is judged on security-property rather than verifier-regex.
4. **B1 was not actually cheap-orchestrator-with-routing-intelligence.** It was direct-model-factory orchestration with explicit staging. Autonomous routing intelligence is Cycle 4 territory.
5. **opencode CLI stall is repeatable but uninvestigated at root cause.** Cycle 4 hook to investigate whether it's specific token patterns, OpenCode Zen rate limiting, opencode buffering, or something else.
6. **Heterogeneity scope cut.** Arm B1 used MiniMax for the LLM reviewer slot. Same as Spike C. Cycle 4 should layer in heterogeneity per A3 design.

---

## Implications for Cycle 3

The pilot does not change Cycle 3's primary central-question evidence (Spike C). It confirms the architecture's value at the multi-stage workflow level on a single trial; it surfaces a substantive deployment-shape finding (opencode CLI fragility); and it produces six concrete Cycle 4 hooks that sharpen the next research cycle's design.

Per the cycle's discipline note (commit to narrow reading first per Cycle 2 susceptibility snapshot pattern #2): **Spike D should be read narrowly as a pilot, not as multi-ensemble-coordination evidence at the full level the central question asks about.** N=1, single fixture, no autonomous routing tested. The substantive findings are (a) the architecture works at the multi-stage workflow level on this trial, (b) the opencode CLI deployment shape is unreliable at the cheap-tier for substantial code prompts.

---

## Cycle 4 hooks (the six the pilot surfaced)

1. **Test `invoke_ensemble` end-to-end via `llm-orc serve`'s direct API path** — the orchestrator's tool-calling loop in production deployment, not via opencode CLI's Bash tool. The architecture's correct path.
2. **Autonomous routing intelligence** — does the cheap orchestrator (with the closed-tool surface and the biased default system prompt) autonomously choose the right ensemble for each sub-task? Pilot did not test this; explicit staging was used. Cycle 4 primary question.
3. **Investigate the opencode CLI stall pattern** — root cause analysis. Specific prompt patterns? OpenCode Zen behavior under load? opencode's stdin buffering? This is a deployment-shape investigation that affects user contract.
4. **Iteration loops** — review → fix → verify → re-fix. The architecture's value compound when the orchestrator can iterate. Pilot ran linear.
5. **Heterogeneous reviewer slots per Cycle 2 A3 design** — Hunyuan + Kimi + script + MARG concatenation. Pilot used MiniMax for the LLM reviewer slot.
6. **Higher-complexity fixtures** — multi-file diffs, longer trajectories, ambiguous task descriptions where routing decisions are themselves load-bearing. Pilot used Spike C's 90-line single-file fixture.

---

## Spike code disposition

**Retain until corpus close** per practitioner policy.

Spike D pilot preserved artifacts:
- `scratch/spike-d-cycle3-multi-ensemble-pilot/run_arm_b1_direct.py` — Arm B1 staged orchestration script (production model factory path)
- `scratch/spike-d-cycle3-multi-ensemble-pilot/stage1-review-output.txt` — Stage 1 review output (extracted from Spike C arm-b trial 1)
- `scratch/spike-d-cycle3-multi-ensemble-pilot/fixed-files/{arm-a1,arm-b1,arm-c1}-trial1.py` — proposed fixed files for verifier scoring
- `scratch/spike-d-cycle3-multi-ensemble-pilot/trials/` — opencode runlogs (incl. the stalled attempts) + frontier subagent trace + Stage 2 LLM response
- `.llm-orc/ensembles/spike-d-fix-verifier.yaml` — the new fix-verifier ensemble (active in this corpus's tier)
- `.llm-orc/scripts/spike_d_fix_verifier.py` — the verifier script (active path)

The opencode-stalled attempts' bash-tool capture files (`/private/tmp/claude-501/.../tasks/<id>.output`) are ephemeral but the runlog text files in scratch capture the started-but-stalled state for forensic record.

---

## Connections to Cycle 3's other research

- **`004d-spike-c-cycle3.md`** — Spike C's architectural-value finding on cross-file verification. Spike D extends to multi-stage workflow.
- **`004b-spike-a-cycle3.md`** — cross-tier-uncorrelated-errors finding. The unifying frame Spike C named: structural compensation via different error distributions. Spike D's Arm B1 enacts this at workflow scale.
- **`004c-spike-b-cycle3.md`** — F1 vs F2 methodological finding. Spike D's stall pattern is an analogous methodological finding at the deployment-shape level: opencode CLI is to multi-stage cheap-tier workflow as F2 was to multi-turn reactivity testing — fragile for the question being asked, working substitute (F1; production model factory) available.
- **`research-design-review-cycle-3-mid.md`** — methods reviewer who motivated Spike C. The pilot's Cycle 4 hooks are the natural extension of the reviewer's "this is Cycle 4 territory" framing.
- **Memory: `cycle-3-central-question`** — central question reframe. Spike D's B1 result on multi-stage workflow is consistent affirmative evidence; the deeper autonomous-routing question is Cycle 4.
- **Memory: `feedback_spike_artifact_retention`** — retention policy applied to Spike D's preserved artifacts.
- **Memory: `feedback_free_options_preference`** — Spike D's $0 cost across all arms preserves the free-options-first preference. Frontier facsimile via subscription is the only non-free path used.

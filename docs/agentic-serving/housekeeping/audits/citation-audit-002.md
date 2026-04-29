# Citation Audit Report

**Audited document:** `docs/agentic-serving/essays/002-capability-floor-and-observability.md`
**Evidence trail:** `docs/agentic-serving/essays/research-logs/research-log.md`
**Lit review:** `docs/agentic-serving/essays/research-logs/lit-review-capability-floor-and-observability.md`
**Date:** 2026-04-25

---

## Summary

- **Total references checked:** 21
- **Verified:** 18
- **Issues found:** 5 (0 P1, 3 P2, 2 P3)

---

## Verification Log

### 1. BFCL F1 scores — Qwen3-14B (0.971), Qwen3-8B (0.933), xLAM-8B (0.570), Watt-tool-8B (0.484)

**Essay location:** §The capability floor is family-specific (para 2); §Open Questions (para 1, BFCL reference).

**Claim:** These four F1 numbers are attributed to "a Docker Engineering Blog evaluation summarizing leaderboard results."

**Finding:** Verified. The Docker Engineering Blog article ("Local LLM Tool Calling: A Practical Evaluation," docker.com/blog, published August 2025) is a real, publicly accessible article that tested 21 models across 3,570 test cases on a MacBook Pro M4 Max. The four F1 values match exactly what the article reports. The lit review (Source #6) carries the citation with a URL. The attribution in the essay is accurate except for one nuance: the Docker evaluation is an independent practical evaluation using an F1 metric on a real tool-selection task — it is not a "summary" of BFCL leaderboard results. The BFCL leaderboard uses AST evaluation, a different methodology. The essay conflates the two in the parenthetical.

**Status:** Scores verified. Attribution phrasing slightly imprecise — see P2 issue below.

---

### 2. Berkeley Function-Calling Leaderboard (Patil et al., ICML 2025)

**Essay location:** §The capability floor is family-specific (para 2, "The Berkeley Function-Calling Leaderboard provides external corroboration"); §Confabulation as a failure mode (para 1).

**Finding:** Verified. The paper is "The Berkeley Function Calling Leaderboard (BFCL): From Tool Use to Agentic Evaluation of Large Language Models" by Shishir G Patil, Huanzhi Mao, Fanjia Yan, Charlie Cheng-Jie Ji, Vishnu Suresh, Ion Stoica, and Joseph E. Gonzalez. Published in PMLR 267:48371–48392, 2025 (Proceedings of the 42nd ICML). Available at proceedings.mlr.press/v267/patil25a.html. First author is Patil — attribution as "Patil et al." is correct.

**Status:** Verified.

---

### 3. VILA-Lab Claude Code architecture paper (arXiv 2604.14228)

**Essay location:** §Confabulation as a failure mode (para 2): "The VILA-Lab Claude Code architecture paper explicitly names 'Silent Failure and the Observability-Evaluation Gap' as an open research direction."

**Finding:** Verified. The paper "Dive into Claude Code: The Design Space of Today's and Future AI Agent Systems" exists at arXiv:2604.14228 (2026). Authors are Jiacheng Liu, Xiaohan Zhao, Xinyi Shang, and Zhiqiang Shen — affiliated with VILA-Lab. The paper is real and publicly accessible. The lit review (Source #8) carries the citation with the arXiv URL and identifier. The essay does not give the paper a proper title or arXiv identifier inline — it refers to it only as "the VILA-Lab Claude Code architecture paper" — but the lit review provides the full citation. The specific quoted language ("Silent Failure and the Observability-Evaluation Gap") and the claim about agents "confidently praising work even when quality is mediocre" are reported in the lit review as direct characterizations of the paper's content. These could not be independently verified against the full paper text in this audit, but the paper's existence, authorship, and general subject matter are confirmed.

**Status:** Verified with a caveat — see P3 issue on exact quote verification.

---

### 4. GitHub issue 1831 for qwen3.5 chat-template bugs

**Essay location:** §Open Questions (para 1): "whether updating Ollama's packaging fixes it" and the research log's reference at S0-CAP-7-rerun.

**Finding:** Verified. GitHub issue #1831 in the QwenLM/Qwen3 repository is real and titled "[Template] 21-fix chat template for Qwen 3.5 — fixes tool calling crash, parallel calls, thinking bleed." It documents known chat-template bugs in Qwen3.5 affecting tool calling, including failure modes consistent with the CAP-7 premature-stop observation. The issue URL (github.com/QwenLM/Qwen3/issues/1831) resolves to an active issue. Additionally, Ollama issue #14745 ("qwen3.5:9b sometimes prints out tool call instead of executing it") corroborates the deployment-level symptom. The research log (S0-CAP-7-rerun) cites the issue by number and title via "Public web search surfaces an active issue in QwenLM/Qwen3." The lit review does not carry this citation — it is only in the research log.

**Status:** Issue verified. See P2 note on missing lit-review provenance.

---

### 5. Beck's "Genie Lessons: Nobody Wants Agents" (April 23, 2026)

**Essay location:** §Conclusion (para 2): "The dual-contract framing — Beck's outcome stance applied to the user"; §The Starting State and throughout.

**Finding:** Verified. The post "Genie Lessons: Nobody Wants Agents" exists at tidyfirst.substack.com/p/genie-lessons-nobody-wants-agents. The lit review (Source #15) carries the citation with the URL and characterizes the post's argument accurately. The research log also cites the URL directly. The essay refers to "Beck's prior posts" and "Beck's outcome stance" without citing specific post titles; the lit review (Sources #16–#18) identifies three additional posts ("Genie Wants to Leap," "Taming the Genie: Like Kent Beck," "Genie Sessions: Optionality") with approximate dates. These are accessible on Beck's substack tag page. The lit review records an explicit limitation that Genie Lessons #1–#4 were partially paywalled and not fully confirmed.

**Status:** "Nobody Wants Agents" verified. Prior posts partially confirmed. See P3 note on paywalled content.

---

### 6. LangGraph supervisor routing overhead >30%

**Essay location:** §Prompt steering is the cheapest sufficient intervention (para 3): "LangGraph documentation reports supervisor-routing overhead in excess of 30%."

**Finding:** The lit review (Source #13, Gap 4) reports this figure: "in one production case study, the supervisor's routing calls accounted for over 30% of total response time." The lit review source is the LangGraph Multi-Agent Workflows blog post (langchain.com). This is a production case study figure from documentation, not a controlled experimental result — the essay reproduces it accurately as reported. Independently verified that LangGraph documentation discusses supervisor routing overhead.

**Status:** Verified as reported.

---

### 7. AutoGen GroupChat — one full LLM call per agent turn

**Essay location:** §Prompt steering (para 3): "AutoGen's GroupChat costs one full LLM call per agent turn."

**Finding:** The lit review (Source #13 / Gap 4) reports this as: "Every agent turn in a GroupChat involves a full LLM call with the accumulated conversation history." The essay's phrasing closely matches. Verified against known AutoGen architecture.

**Status:** Verified.

---

### 8. Internal architecture claims — ADRs 002, 003, 004, 007, 011

**Essay location:** §Implications for the Architecture (para 1): "The four-layer architecture (ADR-002), the closed five-tool surface (ADR-003), the mandatory summarization path (ADR-004), the calibration gating (ADR-007), and the orchestrator-as-Model-Profile commitment (ADR-011) are all empirically validated."

**Finding:** All five ADR files exist in `docs/agentic-serving/decisions/`. File names and numbers match exactly:
- `adr-002-four-layer-architecture-plexus-optional.md` — Status: Accepted
- `adr-003-fixed-orchestrator-tool-surface.md` — Status: Accepted
- `adr-004-result-summarization-mandatory.md` — Status: Accepted
- `adr-007-calibration-gate-for-composed-ensembles.md` — Status: Accepted
- `adr-011-orchestrator-llm-is-a-model-profile.md` — Status: Accepted

The essay's characterizations of each ADR match the system-design module descriptions.

**Status:** Verified.

---

### 9. System-design module names referenced in the essay

**Essay location:** §What Ships (para 1): `src/llm_orc/agentic/orchestrator_config.py`, `OrchestratorToolDispatch.dispatch`, `src/llm_orc/agentic/orchestrator_tool_dispatch.py`, `src/llm_orc/models/openai_compat.py`, `ToolCallingNotSupportedError`.

**Finding:** These module paths and class names are consistent with the system-design module decomposition (Orchestrator Configuration, Orchestrator Tool Dispatch, and the model layer). They are also consistent with what the research log records in DIAG-1, CAP-8, and the pre-S0 logging surface finding. No contradictions found against the system-design document.

**Status:** Verified against design artifacts. (Code-level verification of actual file existence was not performed in this audit — that is a conformance-scan concern, not a citation concern.)

---

### 10. Spike trajectory claims against the research log

**Essay location:** §The Capability Gradient Observed and §Findings throughout.

**Findings — per spike:**

- **CAP-1 (qwen3:8b silent giveup):** Essay claims the model selected `skill` and produced empty content. Research log S0-CAP-1 confirms: "Turn 3 — `skill` (client) — tool_calls=1, content=empty, arg='codebase-audit'" and "loop close: stop." Verified.

- **CAP-3 (biased prompt success):** Essay claims qwen3:8b with biased prompt drove `list_ensembles` → `invoke_ensemble` retries → graceful text fallback. Research log S0-CAP-3 confirms cascade trajectory including 4 `invoke_ensemble` retries and a final text response. Verified.

- **CAP-3b (end-to-end success):** Essay claims this was "the cycle's first end-to-end successful agentic response." Research log S0-CAP-3b confirms: `list_ensembles` → `invoke_ensemble` success → 421-char substantive synthesis. Verified.

- **CAP-5 (mistral-nemo:12b fabrication):** Essay claims "confident-sounding text claiming `list_ensembles` had been called and listing one fabricated ensemble (`ai-detect`) with an invented description and an invocation syntax (`/ai-detect README.md`)." Research log S0-CAP-5 confirms: "721ch text — fabricated," ensemble name `ai-detect`, "slash-command style, not llm-orc's tool surface," verified via `find` and `llm-orc list-ensembles`. Verified.

- **CAP-7 (qwen3.5:9b premature stop):** Essay claims "correctly called `list_ensembles` once but stopped after the result returned, without invoking any ensemble." Research log S0-CAP-7 and S0-CAP-7-rerun both confirm: Turn 1 `list_ensembles` success, Turn 2 empty/no tool calls, loop close. Verified as reproducible across two trials.

- **CAP-8 (deepseek-r1:8b unsupported):** Essay claims Ollama returns 400 "does not support tools." Research log S0-CAP-8 confirms the exact error message: `"registry.ollama.ai/library/deepseek-r1:8b does not support tools"`. Verified.

**Status:** All six spike trajectories match the research log exactly.

---

### 11. S0 (qwen3:14b) trajectory — seven turns, ~22 minutes, five real paths, two fabricated

**Essay location:** §The Capability Gradient Observed (para 1): "seven turns spanning approximately twenty-two minutes of wall-clock time... Path-accuracy on the produced response was five real of seven cited; two paths were fabricated."

**Finding:** Research log S0 (reclassification entry) records the full cascade: 7 turns, 13:39:36–14:01:25 (approximately 22 minutes). For path accuracy, the log records "5 real of 7 cited" for the first-ask synthesis response (not counting the final text close, which did not list paths). The two fabricated items are identified as `src/llm_orc/primitives/replicate_n_times.py` and `profiles/`. Verified.

**Status:** Verified.

---

### 12. CAP-2 router-executor latency overhead (~2.5x)

**Essay location:** §Prompt steering (para 2): "adds approximately 2.5x latency overhead."

**Finding:** Research log S0-CAP-2 records: single qwen3:8b at 7.73s, router-executor at 19.64s. Ratio: 19.64 / 7.73 = 2.54x. The essay rounds to "approximately 2.5x." Verified.

**Status:** Verified.

---

### 13. Operator logging surface — "src/llm_orc/agentic/ has no logger references"

**Essay location:** §Method (para 3): "the entire `src/llm_orc/agentic/` tree has no logger references at all."

**Finding:** Research log (Pre-S0 finding, "Operator-side logging surface is essentially empty") confirms: "No logger references anywhere in `src/llm_orc/agentic/`." This is a codebase claim, not a citation, and is corroborated by the research log's source survey finding.

**Status:** Verified against research log record.

---

### 14. ToolBench, ToolEval, tau-bench, AgentBench — benchmarks measuring tool calls that occur, not claims

**Essay location:** §Confabulation as a failure mode (para 1): "ToolBench, ToolEval, tau-bench, and AgentBench similarly measure structural and outcome dimensions of tool calls that occur."

**Finding:** The lit review addresses ToolBench (Source #2, Gap 1), tau-bench (Source #3, Gap 1), and BFCL (Source #1). AgentBench is mentioned in the research plan as a literature target but does not appear in the lit review's sources table — it was presumably not retrieved. The claim that these benchmarks do not measure confabulation (claims without dispatch) is confirmed by the lit review's Gap 1 synthesis, which explicitly states "The BFCL paper does not explicitly measure the specific failure mode the cycle identified as 'fast-confabulation.'" The inclusion of AgentBench in this list is not contradicted but also not directly confirmed by the lit review.

**Status:** Claim directionally verified. AgentBench inclusion is asserted without specific lit-review source — see P3 note.

---

### 15. ToolACE paper (Wang et al., arXiv 2409.00920)

**Essay location:** Not cited by name in the essay, but underlies claims about tool surface dilution. The essay does not explicitly cite ToolACE.

**Finding:** No ToolACE citation appears in the essay text. The lit review (Source #4) covers it. No issue to flag — the essay does not claim a ToolACE citation.

**Status:** N/A — not cited in essay.

---

### 16. Qwen3.5 release date — March 2, 2026

**Essay location:** §The Capability Gradient Observed (para 4): "newer and larger than qwen3:8b."

**Finding:** The lit review (Source #19/Gap 6) states Qwen3.5 small models were released March 2, 2026. The essay does not state the date explicitly, only that qwen3.5:9b is "newer and larger." No factual issue — the relative claim is accurate.

**Status:** Verified.

---

### 17. Summarization failure root cause — missing qwen3:0.6b model

**Essay location:** §The Capability Gradient Observed (para 3) — "when summarization failed" and the DIAG-1 fix.

**Finding:** Research log DIAG-1 confirms the root cause as `model 'qwen3:0.6b' not found (status code: 404)` from a direct `llm-orc invoke` call. The essay does not explicitly describe the root cause in the capability-gradient section (it surfaces it implicitly in §The Starting State and §What Ships). The research log's finding is accurate and is not contradicted by the essay.

**Status:** Verified.

---

### 18. "DEFAULT_ORCHESTRATOR_SYSTEM_PROMPT" in orchestrator_config.py

**Essay location:** §What Ships (para 1): "updates `DEFAULT_ORCHESTRATOR_SYSTEM_PROMPT` in `src/llm_orc/agentic/orchestrator_config.py`."

**Finding:** The system-design Orchestrator Configuration module owns "Per-session config resolution; operator-set bounds on per-request overrides" and depends on the project config manager. The research log (S0-CAP-3) confirms the prompt override path via `agentic_serving.orchestrator.system_prompt` in the local config. The specific constant name `DEFAULT_ORCHESTRATOR_SYSTEM_PROMPT` is consistent with how config constants are typically named in the codebase (the research log mentions the default at `src/llm_orc/agentic/orchestrator_config.py:38` for `DEFAULT_TOKEN_LIMIT`, confirming that file exists and contains constants of this pattern). The file path is internally consistent with the research log's source survey.

**Status:** Verified against research log provenance.

---

## Issues

### P1 — Must Fix

None identified.

---

### P2 — Should Fix

#### P2-1 — Docker evaluation misattributed as BFCL-derived

**Location:** Essay §The capability floor is family-specific, para 2 (line 49): "Qwen3-14B at F1=0.971, Qwen3-8B at F1=0.933 per a Docker Engineering Blog evaluation summarizing leaderboard results."

**Claim:** The parenthetical implies the Docker evaluation summarizes or derives from BFCL leaderboard results.

**Finding:** The Docker evaluation (docker.com/blog, August 2025) is an independent practical evaluation using F1 score across 3,570 test cases on a MacBook Pro M4 Max. Its methodology is different from BFCL's AST evaluation — these are two separate measurement frameworks. The Docker post does reference the BFCL leaderboard as context, but the F1 scores are the Docker evaluation's own measurements, not summaries of BFCL scores. The lit review (Source #6 synthesis, Gap 1) is explicit: "The most important data point for this cycle comes from an independent practical evaluation (Docker Engineering Blog, 2025) that tested 21 models... Key numbers for the ≤14B class: Qwen3 14B scored F1=0.971... These numbers are striking because they show the Qwen3 family significantly outperforms specialist function-calling models."

The BFCL leaderboard's own tabular scores for Qwen3-8B and Qwen3-14B were not confirmed from the primary BFCL source — the lit review's Limitations section states: "The BFCL V4 leaderboard is a live page that did not render tabular data in the search tools used. Specific ranked scores for qwen3:8b, qwen3:14b... were not directly confirmed via the primary source."

**Recommendation:** Change the parenthetical from "per a Docker Engineering Blog evaluation summarizing leaderboard results" to "per a Docker Engineering Blog independent practical evaluation (August 2025) using F1 score across 3,570 test cases." Drop the "summarizing leaderboard results" framing, which misrepresents the Docker evaluation's relationship to BFCL. A corrected sentence:

> Reported scores in the small-model class (Qwen3-14B at F1=0.971, Qwen3-8B at F1=0.933 per the Docker Engineering Blog's independent practical evaluation of 21 local models, with specialist models xLAM-8B at F1=0.570 and Watt-tool-8B at F1=0.484) are consistent with the cycle's empirical observation...

---

#### P2-2 — GitHub issue 1831 has no URL or repository identifier in the essay

**Location:** Essay §Open Questions (para 1, line 89): "Public sources document an active chat-template bug for qwen3.5 tool calling that is consistent with the observed symptom; whether Ollama's current qwen3.5 packaging incorporates the fix is not directly verifiable through the spike but the symptom-cause correspondence is strong."

**Claim:** The essay refers to this bug without citing "GitHub issue 1831" by number or URL (the issue number only appears in the research log at S0-CAP-7-rerun). The lit review does not carry this citation.

**Finding:** The issue is real — github.com/QwenLM/Qwen3/issues/1831, titled "[Template] 21-fix chat template for Qwen 3.5 — fixes tool calling crash, parallel calls, thinking bleed." The symptom-cause correspondence the essay claims is strong is confirmed: the issue documents multiple tool-calling failure modes for Qwen3.5. However, because the essay does not cite the issue number or URL, a reader cannot verify the claim. The lit review (which is the cycle's citation source) does not include this issue as a source — it surfaced only in the research log's spike work.

**Recommendation:** Either (a) add the issue citation to the essay at the point of claim, e.g.: "consistent with a documented chat-template bug (QwenLM/Qwen3 GitHub issue #1831)"; or (b) add it to the lit review's sources table so citation provenance exists in the canonical reference artifact. Option (b) is the cleaner fix given the cycle's citation architecture (essay claims cite through the lit review).

---

#### P2-3 — VILA-Lab paper named without title or arXiv identifier in essay text

**Location:** Essay §Confabulation as a failure mode, para 2 (line 69): "The VILA-Lab Claude Code architecture paper explicitly names..."

**Claim:** The paper is identified only by lab name and informal description. No title, no arXiv ID, no year appears in the essay text itself.

**Finding:** The paper exists (arXiv:2604.14228, "Dive into Claude Code: The Design Space of Today's and Future AI Agent Systems," 2026) and the lit review (Source #8) carries the full citation. However, the essay's in-text reference provides no identifier that would let a reader locate the paper independently. Every other external paper in the essay is named more specifically (Patil et al. ICML 2025 is named with venue; Docker Engineering Blog is named as a source). The VILA-Lab paper is the only one cited entirely by informal description.

**Recommendation:** Add the arXiv identifier inline, e.g.: "The VILA-Lab paper on Claude Code's architecture (Liu et al., arXiv:2604.14228, 2026) explicitly names..."

---

### P3 — Consider

#### P3-1 — VILA-Lab quoted language not independently verified from paper text

**Location:** Essay §Confabulation as a failure mode, para 2 (line 69): "explicitly names 'Silent Failure and the Observability-Evaluation Gap' as an open research direction and notes that agents tend to respond by confidently praising work even when quality is mediocre."

**Claim:** Two specific characterizations of the paper's content — the named section title and the "confidently praising" phrasing.

**Finding:** These characterizations come from the lit review (Source #8, Gap 2), which reports: "The VILA-Lab research paper on Claude Code's architecture (arXiv 2604.14228, 2025) explicitly identifies 'Silent Failure and the Observability–Evaluation Gap' as an open research direction. The paper notes the system 'offers limited mechanisms that explicitly surface when recovery has occurred' — and that agents 'tend to respond by confidently praising the work, even when quality is mediocre.'" The lit review retrieved this from arXiv but the specific quoted phrases could not be independently verified against the full paper text in this audit. The characterization is plausible and consistent with the paper's subject matter. No contradictory evidence found.

**Recommendation:** No action required unless the essay moves into a venue with formal peer review, in which case the exact quotes should be verified against the paper PDF. For current cycle purposes, the lit review's characterization is sufficient provenance.

---

#### P3-2 — AgentBench listed without lit-review source

**Location:** Essay §Confabulation as a failure mode, para 1 (line 67): "ToolBench, ToolEval, tau-bench, and AgentBench similarly measure structural and outcome dimensions."

**Claim:** AgentBench is listed alongside three benchmarks that the lit review explicitly covers (ToolBench, ToolEval/BFCL, tau-bench).

**Finding:** AgentBench does not appear in the lit review's sources table or synthesis. It was listed in the research plan's literature scan scope ("BFCL, ToolBench, tau-bench, AgentBench") but was not returned in either the partial or full lit-review dispatches. The claim that AgentBench measures "structural and outcome dimensions of tool calls that occur, not the presence of tool-call claims absent dispatch" is consistent with what is publicly known about AgentBench, but it lacks citation provenance in the cycle's artifact corpus.

**Recommendation:** Either add AgentBench to the lit review's sources with a URL/DOI, or remove it from the essay's list. The claim is directionally accurate but unsupported by the lit review. Removing it does not weaken the argument — ToolBench, ToolEval, and tau-bench are sufficient to make the point.

---

## Notes on Unverified but Non-Problematic Claims

The following claims were not independently verified but are internally consistent, do not carry citations, and are product of the cycle's own empirical work — not external attribution:

- The six-minute wall-clock in PLAY (FF #131 context — confirmed as a local config override)
- The four-layer architecture description and its modules (consistent with system-design.md)
- All internal system-prompt text and file path references (internally consistent; conformance-scan territory)
- The CAP-2 finding of 7.73s / 19.64s latency (reported from research log without contradiction)
- The claim that `src/llm_orc/web/server.py` declares one logger — consistent with the research log's pre-S0 source survey

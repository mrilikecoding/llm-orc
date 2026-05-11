# Spike A — Cycle 3 RQ-1 Isolation Test (and Cross-Tier Heterogeneity Finding)

**Date:** 2026-05-01
**Cycle:** 3 (agentic-serving scoped corpus)
**Fixture:** Project `README.md` (888 lines; same fixture Cycle 2 A3 ran against)
**Operating frame:** Outcomes over an agentic session — agent shape is means; the cycle's evaluation axis is outcome quality per session, not architectural fidelity to any pre-existing shape.
**Central question (mid-spike practitioner reframe):** Does orchestration + cheap orchestrator agent compete with a more expensive frontier model on documentation review?

---

## Abstract

Spike A tested seven configurations on Cycle 2's README-review fixture to isolate the load-bearing component of A3's bug-detection capability and to probe the cycle's central question: whether cheap-orchestrator + orchestration competes with a frontier-model reference. Five configurations ran on MiniMax M2.5 Free (cheap tier, via the OpenCode Zen openai-compatible endpoint, accessed through llm-orc's production model factory). Two configurations ran on Sonnet 4.6 (frontier tier, accessed via Claude Code Agent dispatch as facsimile for direct-API behavior). The primary RQ-1 isolation finding is unambiguous: *A2 + script input does not match A3's grounding* on this fixture — script-context-as-input does not elicit the semantic analysis that catches undefined model-profile references. The contamination of arm3 (a contingent script-as-orchestrator probe) was caught and isolated through arm3_debiased and arm4 follow-up runs; explicit prompt direction toward the failure category was identified as the dominant driver of bug detection within the cheap tier, not the architectural framing alone. The most novel finding is cross-tier and unanticipated: cheap-orchestration with directed prompting and frontier-model-alone find *different and largely uncorrelated* bug classes on this fixture — the heterogeneity-uncorrelated-errors mechanism Cycle 2 observed across heterogeneous LLM reviewers within a single tier extends across tiers as well. The four-priorities frame produces a frame-divergent recommendation against a performance-only frame on this fixture: complementary multi-tier review captures bug coverage that neither tier achieves alone. Single-fixture scope is load-bearing throughout; the broader reading requires additional task classes.

---

## Background

Cycle 2's essay 003 (`003-multi-turn-orchestration-and-the-four-axis-frame.md`) documented Spike A3 — a novel ensemble combining a Python script analyzer (link-validity / canonical-section-presence / code-block-parseability checks), two heterogeneous cloud LLM reviewers from different model families (Tencent Hunyuan `hy3-preview-free` and Moonshot Kimi `kimi-k2.5`, both via OpenCode Zen), and a MARG-style concatenation aggregator with no synthesizer collapse. A3 surfaced documentation bugs A2 (the prompt-steered single MiniMax orchestrator baseline) missed across three trials: undefined `default-local` and `ollama-gemma-small` model-profile references in YAML examples never defined in the README's Configuration section. Essay 003 explicitly identified an untested alternative — "A2 + script input": a prompt-steered single cloud orchestrator receiving the script-agent's deterministic report as additional input context — as the cycle's open scientific question. Cycle 2's susceptibility snapshot named this as a specific in-cycle grounding action for Cycle 3's research entry (`housekeeping/audits/susceptibility-snapshot-cycle-2-research.md`).

Cycle 3's research-design review (`housekeeping/audits/research-design-review-cycle-3.md`) accepted the RQ-1 isolation framing, added a script-as-orchestrator coverage instruction (P2 #1) for the lit review, and extended Spike A's Decision C to include a contingent script-as-orchestrator arm (Arm 3) in the harness if free-tier quota allowed. Decision A (PAE-aware procedural independence per Cao et al. arXiv:2603.03116) and Decision B (CLEAR cost-normalized accuracy per Rony et al. arXiv:2511.14136) were adopted as evaluation refinements during Step 1.4. Cycle 3's combined lit review (`004a-lit-review-agent-design.md`) found no published paper directly comparing "A2 + script input" against MARG-style heterogeneous ensembles on documentation-review task classes — RQ-1's spike sits in a literature gap.

The mid-spike practitioner reframe (recorded in memory: `cycle-3-central-question`) sharpened the cycle's central question from "does orchestration shape matter?" to "does cheap-orchestration + orchestration compete with a more expensive frontier model?" The reframe was operationalized by adding two frontier-tier arms (arm-frontier-bare and arm-frontier-with-script) using Claude Sonnet 4.6 dispatched through Claude Code's Agent tool as a single-shot-API facsimile. The frontier reference is a facsimile rather than a direct API call — caveats recorded in §Limitations.

---

## Method

### Configurations tested (7 arms across 2 tiers)

**Cheap tier — MiniMax M2.5 Free via OpenCode Zen.** Accessed through llm-orc's production `ConfigurationManager` + `CredentialStorage` + `ModelFactory` code path. Same orchestrator profile (`orchestrator-minimax-m25-free`) llm-orc serves agentic-serving with. Cost per token: $0.00 (free tier).

- **arm1 (A2 baseline reconstruction):** Single-LLM code review on README only, no script context. Prompt steering via `default_task` from `code-review.yaml`.
- **arm2 (A2 + script input):** Same prompt as arm1, with the deterministic script's report prepended as input context. RQ-1's primary isolation test.
- **arm3 (script-as-orchestrator, CONTAMINATED):** System prompt frames the LLM as synthesizer of completed deterministic analysis with explicit example: *"do YAML examples reference profiles or keys that are actually defined in the document?"* — the exact failure category being tested. Contamination caught and acknowledged at trial-results review; arm preserved for record.
- **arm3_debiased (script-as-orchestrator, DEBIASED):** Identical system framing as arm3 with all specific failure-mode hints removed. No example failure categories named.
- **arm4 (script only, no README):** Same debiased synthesis prompt as arm3_debiased; user message contains only the script's deterministic report — no README content. Tests whether bug-finding requires README ground for semantic reasoning or whether the LLM hallucinates from the script's structural data alone.

**Frontier tier — Claude Sonnet 4.6 (facsimile).** Dispatched via Claude Code's Agent tool (`general-purpose` subagent type, `model: sonnet`). Each dispatch was instructed to behave as a single-shot API call: a single Read of the named fixture file(s), no further tool use, produce the review as the response. Subagent path is not equivalent to a direct API call (caveats below); evaluated as facsimile for the cycle's central-question probe.

- **arm-frontier-bare:** Read README.md once, no other tools, produce code review using the same `default_task` system prompt as arm1.
- **arm-frontier-with-script:** Read README.md and `script-output.txt` (in that order), no other tools, produce code review using the same `default_task` system prompt with explicit context that the script-output is completed deterministic analysis.

### Fixture and reconstructions

The fixture is the project's `README.md` at the repo root (888 lines, ~34K characters). The undefined-profile bugs A3 caught — `default-local` referenced at lines 619 and 626 in the fan-out / ensemble examples; `ollama-gemma-small` referenced at line 671 — are still present in the current README (verified prior to spike execution). Neither profile is defined in the Configuration section's profile list (which defines `free-local`, `default-claude`, `high-context`, `small`, `openai-local` — and zero locally-named "default-local" or "ollama-gemma-small").

The deterministic script (`scratch/spike-a-cycle3-a2-script-input/deterministic_analyzer.py`) reproduces A3's three structural checks: link-validity verification (HTTP HEAD/GET on extracted URLs, classification of loopback URLs separately per A3's discipline), canonical-section presence check across five canonical README sections (Installation / Quick Start / Configuration / Use Cases / License), and code-block parseability across YAML/JSON/Python fenced blocks. The script's run on the current README returns: 5/5 canonical sections present, 16 external URLs (all 2xx/3xx), 1 loopback URL flagged separately (`http://localhost:11434/v1`), 8/8 YAML and 2/2 JSON code blocks parseable. The 17→16 external-URL count delta from A3 is documented README drift since 2026-04-30; the structural pattern is intact and the undefined-profile bugs are unchanged.

Two reconstructions were necessary, each carrying a documented divergence from Cycle 2's exact A2 path:

1. **Prompt reconstruction.** A2's exact prompt at Cycle 2 was OpenCode-CLI's default agent-mode system prompt at the OpenCode version then current. Spike A reconstructs A2 using the project's own `code-review.yaml` `default_task` text as the system prompt. The same prompt is used across arms 1–3 for internal consistency. The OpenCode-CLI default-agent-mode wrapper that surrounded Cycle 2's A2 calls is not reproduced.
2. **Path-to-API divergence.** Cycle 2's A2 called MiniMax via `opencode run`, mixing OpenCode's native tools with llm-orc's. Spike A's cheap arms call MiniMax via llm-orc's production OpenAI-compatible client at the same `https://opencode.ai/zen/v1` endpoint with the same `minimax-m2.5-free` model. The endpoint and model are identical; the path differs.

### Trial design and evaluation

Cheap arms ran N=3 trials per arm (matching Cycle 2's spike-battery norm); arm1 had 2/3 successful trials due to one transient HTTP 500 from OpenCode Zen. Frontier arms ran N=2 trials per arm (sufficient for a directional facsimile probe; lower than the cheap-tier N=3 due to subscription-token consideration). Total: 17 successful trials across 7 arms.

Each trial result was captured as a JSON or markdown record at `scratch/spike-a-cycle3-a2-script-input/trials/`, including: full response text, latency, input/output token counts where reported, error notes where applicable.

**Decision A operationalization (PAE-aware procedural independence).** Two evaluation tracks: (a) issue-count parity — does the configuration find the undefined-profile bugs A3 found? — and (b) procedural independence — does the bug-finding reflect independent semantic analysis of the README, or restating the script's output? Track (b) is verified by examining response surface for synthesis content beyond what's in the script's deterministic report. Arm 4 serves as a structural test of (b): the script alone, without README content, cannot ground the bug-finding via independent analysis; positive bug detection in arm 4 would indicate hallucination.

**Decision B operationalization (CNA — cost-normalized accuracy).** Per-arm token consumption recorded; cost in dollars is $0.00 across cheap-tier MiniMax calls (free tier) and approximately $0.00 marginal across the frontier facsimile (subscription, not metered). Quality axis is bug-class detection breadth — both undefined-profile-bugs (cheap tier's territory) and the broader semantic-consistency / security / operational issues that emerged across configurations.

---

## Results

### Per-arm summary

| Arm | Tier | n successful | Median wall-clock | Avg in / out tokens | Found undefined-profile bugs |
|-----|------|-------------:|------------------:|-------------------:|:----------------------------|
| arm1 (A2 baseline) | Cheap | 2/3 | 108.95 s | 7,698 / 3,529 | 0 / 2 |
| arm2 (A2 + script input) | Cheap | 3/3 | 25.86 s | 7,932 / 3,192 | **0 / 3** |
| arm3 (CONTAMINATED) | Cheap | 3/3 | 87.24 s | 8,010 / 3,456 | 3 / 3 (both bugs every trial) |
| arm3_debiased | Cheap | 3/3 | 47.82 s | 7,947 / 2,580 | 1 / 3 (only `ollama-gemma-small` once) |
| arm4 (script only, no README) | Cheap | 3/3 | 18.52 s | 319 / 1,256 | 0 / 3 |
| arm-frontier-bare | Frontier (Sonnet 4.6 facsimile) | 2/2 | ~51 s | ~28K total per trial (subagent reported, includes context overhead) | **0 / 2** |
| arm-frontier-with-script | Frontier (Sonnet 4.6 facsimile) | 2/2 | ~56 s | ~29K total per trial | **0 / 2** |

### Per-arm narrative findings

**arm1 (A2 baseline reconstruction).** Generic code review covering security claim ambiguity, configuration management, Quick Start length, and minor cross-reference gaps. Did not find the undefined-profile bugs in either successful trial. Latency unexpectedly high relative to Cycle 2's A2 (~108s median vs Cycle 2's ~19.5s) — attributed to OpenCode Zen response-time variance and possibly response-length differences (Cycle 2 essay 003 reported 16 specific recommendations; arm1 trials produced ~3.5K output tokens of broader prose).

**arm2 (A2 + script input).** Same prompt-steered review as arm1 with script's deterministic report prepended as input context. **Treats the script's findings as checklist confirmation** ("✅ All 5 canonical sections present, ✅ 16/16 external URLs valid, ✅ 10/10 code blocks parseable") and proceeds to general code review without doing the semantic analysis that catches undefined-profile references. Bug detection: 0/3 across all trials — same outcome as arm1, with script context but no behavioral shift on the bug class. Median latency dropped substantially (25.86s vs arm1's 108.95s); response length similar; the script-context appears to have anchored the response toward briefer, more focused output without changing the semantic-analysis behavior.

**arm3 (CONTAMINATED).** System prompt instructed the LLM to check semantic correctness with the explicit example *"do YAML examples reference profiles or keys that are actually defined in the document?"* All three trials reliably surfaced both undefined-profile bugs (3/3) with line citations and recommended fixes. Bug-finding is procedurally independent (citing line numbers, providing rationale, in one trial surfacing an additional related issue). The confound: the prompt named the failure category. Cannot be cleanly attributed to the script-as-orchestrator architectural framing alone.

**arm3_debiased.** Same script-as-orchestrator framing with the failure-category example removed. Bug detection dropped to 1/3 trials — and only one of the two known bugs (the `ollama-gemma-small` reference; not `default-local`). Marginal lift over arm2's 0/3 attributable to the architectural framing, but well below arm3's contaminated reliability. Reading: the script-as-orchestrator framing alone produces marginal bug-detection improvement over generic-review framing, but does not approach reliable detection without explicit prompt direction.

**arm4 (script only, no README).** Same debiased synthesis prompt as arm3_debiased; user message contains only the script's deterministic report (~860 chars), no README content. Bug detection: 0/3. Confirms two things: (i) the bug-finding in arm3 / arm3_debiased is grounded in the README's semantic content, not hallucinated from the script's structural output; and (ii) without README grounds, the LLM produces only a brief restatement-and-mild-elaboration of the script's report (output dropped to ~1.3K tokens vs other arms' ~3K).

**arm-frontier-bare (Sonnet 4.6 facsimile).** Both trials produced substantive code reviews of similar length to cheap-tier arms (~28K total tokens including subagent context overhead; ~3K equivalent of response content). Issues caught: configuration hierarchy ordering inversion at lines 805–809 (caught both trials), encryption claim ambiguity at line 104 (both trials), model-ID staleness in the example profiles (`claude-sonnet-4-20250514` flagged as wrong; both trials), fan-out partial failure documentation gap (both trials), input_key error behavior vagueness (both trials), nesting-depth-limit silent failure (trial 1), MCP HTTP transport security (trial 2), reset-global footgun (trial 1), Homebrew version drift (trial 2), test-isolation documentation gap (trial 2). **Did not catch the undefined-profile bugs in either trial.**

**arm-frontier-with-script (Sonnet 4.6 facsimile).** Similar issue profile to arm-frontier-bare. Trial 1 echoed the script's deterministic findings ("5/5 canonical sections, 16 external URLs valid, 10 code blocks parseable") in summary form before proceeding to similar architectural / security / operational issues. Trial 2 produced more structural critique of the README organization. **Did not catch the undefined-profile bugs in either trial.** The script context did not shift the bug-class profile away from architectural / operational / security toward semantic-consistency.

### Cross-arm patterns

**Pattern 1 — A2 + script input does not match A3's grounding.** The cheap-tier isolation test (arm1 vs arm2 vs Cycle 2's documented A3 outcome) is unambiguous on this fixture. arm2's 0/3 detection of undefined-profile bugs replicates Cycle 2's A2-without-script behavior. Adding the script as input context did not shift the LLM's analytical behavior. RQ-1's "Worse grounding" decision-tree branch fires; ADR-011's Cycle 2 boundary refinement (defensible as default but not as ceiling for task classes where factual grounding via deterministic checks is part of the success criterion) holds.

**Pattern 2 — Architectural framing has marginal effect; explicit prompt direction is the dominant driver within cheap tier.** Comparison of arm2 (0/3) against arm3_debiased (1/3) against arm3 contaminated (3/3) is monotonic in prompt directiveness. The script-as-orchestrator architectural framing alone produces a small detection lift; adding explicit failure-mode direction in the prompt produces a large detection lift. arm4's 0/3 confirms the bug-finding is README-grounded, not script-data hallucination. Within the cheap tier on this fixture: prompt engineering dominates architectural composition for the bug class A3 caught.

**Pattern 3 — Frontier tier finds different bugs than cheap tier.** All four frontier trials missed the undefined-profile bugs (`default-local`, `ollama-gemma-small`) — same blind spot as cheap tier without explicit direction. The frontier trials caught a different bug set: configuration hierarchy ordering inversion, encryption claim ambiguity, model-ID staleness, fan-out partial-failure documentation, MCP HTTP transport security, nesting-depth-limit silent failure, reset-global footgun, library-source environment-variable interaction, MoA factual overstatement. Across the four frontier trials, configuration-hierarchy-inversion was found 4/4 (strong consensus); encryption-claim and fan-out-failure 4/4; model-ID issue 3/4; the rest distributed across trials.

**Cross-tier finding — uncorrelated bug detection at cheap tier (with directed prompting) and frontier tier.** arm3 (contaminated) caught 3/3 undefined-profile bugs but did not catch the configuration-hierarchy-ordering issue or fan-out-failure issue that frontier trials caught 4/4. arm-frontier-bare and arm-frontier-with-script caught 4/4 of those issues but did not catch the undefined-profile bugs in any trial. The two tier×prompting combinations find largely **uncorrelated bug classes** on this fixture. This is the cycle's most novel empirical finding.

**Pattern 4 — Procedural independence holds where bugs are found.** Per Decision A: in arm3 (and the single arm3_debiased trial that found a bug), the bug detection cited specific line numbers, provided rationale, and surfaced fix recommendations beyond what the script's report contained. The bug-finding is procedurally independent — actual semantic analysis of README content, not restatement. The arm3 contamination is upstream of procedural independence (the prompt directed the LLM toward the bug category) but does not contaminate the procedural independence itself; the LLM did the semantic work after being directed.

**Pattern 5 — Frame-divergent recommendation detected at this fixture (per Decision B).** A performance-only frame on this spike's data would recommend "use the frontier model alone" — frontier finds more named issues per trial than cheap-without-direction (arm1 / arm2 / arm3_debiased), is broadly more eloquent, and arrives at architectural concerns the cheap tier misses. A four-priorities frame, given the cross-tier-uncorrelated-errors finding, recommends complementary multi-tier review (cheap-with-directed-prompting + frontier review) for broader bug coverage at lower aggregate cost — token cost on cheap tier is $0.00; frontier coverage is in subscription quota; the combined cost of running both is dominated by subscription, but the bug-class breadth is larger than either alone. The two frames produce **different recommendations** on this fixture's data. Per RQ-2's falsification criterion: the four-priorities frame is **not retired** by Cycle 3's evidence; it has empirical support at this fixture's resolution.

---

## Discussion

### RQ-1 settled at fixture scope

The primary RQ-1 finding is clean and bounded: on the cycle-2 README-review task class, A2 + script input does not produce equivalent factual grounding to A3's novel ensemble. The architectural lesson at this fixture is *not* "augment prompt-steering with deterministic tool outputs" — that hypothesis is refuted at this fixture. Per the synthesis discipline note (commit to narrow reading first; counters self-correction blind-spot on overgeneralization): generalization to other documentation-review fixtures, code-review task classes proper, or multi-turn agentic work is not warranted by this evidence alone.

### arm3 contamination — methodological self-correction recorded

The contamination of the original arm3 system prompt was caught at trial-results review when the bug-detection rate (3/3 with both bugs every trial) was inconsistent with the rest of the cheap-tier results. The contamination instance — including the literal sentence in the system prompt that named the failure category being tested — is recorded in the harness source (`spike_a_harness.py`) as `ARM3_SYNTHESIS_SYSTEM_PROMPT` and preserved for transparency. arm3_debiased (the same architectural framing without the example) and arm4 (script-only, no README) followed up to disambiguate. The contamination is the kind of upstream prompt-engineering error the susceptibility snapshot's "narrow reading first" discipline guards against — broad readings of arm3's contaminated 3/3 result would have produced a wrong architectural conclusion.

### Cross-tier heterogeneity-uncorrelated-errors — the cycle's most novel finding

Cycle 2 essay 003 documented the heterogeneity-uncorrelated-errors mechanism *within the cheap tier*: A3's two heterogeneous LLM reviewers from different model families (Hunyuan + Kimi) found 5–8 distinct findings each with only 1–2 overlap. The mechanism Sun et al. (2025) and Ding et al. (2024) characterize is: *different model families have uncorrelated error distributions*, so combining them recovers up to 95 percent of the diversity-based theoretical ceiling.

Spike A's data extends this empirically across tiers. arm3 (contaminated cheap tier, with directed prompting) and the four frontier trials find largely **non-overlapping** bug classes on this fixture: cheap-with-direction catches semantic-consistency-across-section bugs; frontier catches architectural / security / operational bugs. The bugs are real in both cases. The two configurations are not redundant — neither dominates the other on coverage breadth, and neither's bug class collapses into the other's.

Cycle 3's lit review (`004a-lit-review-agent-design.md`) noted that the OneFlow result (arXiv:2601.12307, Xu et al., January 2026) and Lee et al. (arXiv:2601.04748) converge on a conditional claim — single-agent simulation matches *homogeneous* multi-agent on most tested classes, but heterogeneous multi-agent remains the residual unsettled case. Spike A's evidence sits in that residual case: cheap-tier-with-direction and frontier-tier-direct are heterogeneous in tier and capability rather than just in model family. The complementarity finding is consistent with the literature's heterogeneity prediction at the within-tier scope and extends it cross-tier on this fixture. A direct test of this cross-tier claim was not in the lit review found and would be novel empirical work — Cycle 4 hook.

### RQ-2 has empirical evidence — the four-priorities frame is load-bearing on this fixture

The four-priorities frame (performance × environmental cost × local-first × token cost) was adopted at Cycle 2's Loop 1 synthesis exchange under practitioner pushback and recorded transparently as one valid framing. RQ-2's falsification criterion required a Cycle-3-tested configuration producing a frame-divergent recommendation versus a performance-only frame — and Spike A's data produces this:

- Performance-only frame would select frontier-tier (more issues found per trial, more architectural depth, more polished output) — and miss the undefined-profile bugs.
- Four-priorities frame, given that token cost on cheap tier is $0.00 and that the cheap-with-directed-prompting + frontier combination finds a strictly larger bug set than either alone, recommends complementary review.

The frame-divergence is real on this fixture's evidence. Per Decision B's scoring-resolution caveat: the qualitative scoring on environmental-cost and local-first dimensions did not introduce ambiguity here because the divergence is detectable on the measured dimensions alone (token cost: cheap is $0, frontier is subscription-metered; performance: bug-class breadth differs). The frame is not retired by this evidence; it survives a real falsification attempt with a documented frame-divergent recommendation.

A scope condition is load-bearing: this is single-fixture evidence. The frame might still collapse to performance-only on other task classes where cheap-tier produces strictly worse output across all bug classes. The Cycle 3 finding is "frame-divergent recommendation detected at this fixture's resolution" — not "frame is universally load-bearing."

---

## Limitations and scope conditions

1. **Single fixture.** All seven configurations ran on the same `README.md`. Generalization to other documentation-review fixtures, different project README styles, or other task classes (code review proper, multi-file refactor, debugging, etc.) requires additional spikes. The cycle's findings are bounded to this fixture's evidence.

2. **Reconstructed A2 prompt.** The exact A2 prompt at Cycle 2 was OpenCode-CLI's default agent-mode wrapper at the OpenCode version then current; not recoverable from any source available now (clean reflog, no stashes, no project-level OpenCode config). Spike A reconstructed using `code-review.yaml`'s `default_task` text as the system prompt — same project's production code-review register, but not Cycle 2's exact A2 prompt. Whether the reconstruction approximates Cycle 2's A2 quality is itself a scope condition; arm1's higher latency than Cycle 2's A2 (108s vs 19.5s) suggests the configurations diverge in some respect.

3. **llm-orc client path divergence from `opencode run`.** Cycle 2's A2 went through OpenCode CLI's agent mode; Spike A's cheap arms go through llm-orc's production OpenAI-compatible client at the same MiniMax endpoint. Same endpoint and model; different path. The LLM itself is the same; surrounding-tooling overhead differs.

4. **Frontier facsimile via Claude Code subagent — not direct API.** arm-frontier-bare and arm-frontier-with-script were dispatched via Claude Code's Agent tool with `model: sonnet` and instructed (via prompt) to behave as single-shot API calls. Subagent context overhead exists; tool availability is structurally present (Read was used to fetch files; other tools not used per instruction). A direct Anthropic API call to Sonnet 4.6 with the same prompt structure may produce slightly different output. The facsimile was accepted at the cycle as preferable to incurring direct API cost; the "facsimile, not API" caveat is documented.

5. **Comparison against A3 is documentary, not direct.** Spike A did not re-run A3 for this cycle. The A3 reference is the documented findings in essay 003 (`003-multi-turn-orchestration-and-the-four-axis-frame.md`). A re-run of A3 was out of scope; whether A3 still finds the bugs reliably with current OpenCode Zen state is not verified.

6. **Limited variance estimation per arm.** N=3 per cheap-tier arm and N=2 per frontier-tier arm. Sufficient for the spike's directional questions but limited for variance-tight claims. A single trial across configurations would detect bug-finding presence but not its rate; N=2-3 detects rate at this resolution but not narrow effect sizes.

7. **One transient API error (arm1 trial 3).** OpenCode Zen returned HTTP 500 once during the arm1 batch. The arm1 reduced n=2 affects variance estimation slightly; bug-detection rate of 0/2 is consistent with arm2's 0/3 directionally.

8. **Bug-class set is not exhaustive.** The cycle's bug-class evaluation focused on the undefined-profile bugs A3 highlighted as A2's misses, plus the broader issues each configuration named. A more comprehensive bug taxonomy would require ground-truth annotation beyond Spike A's scope.

---

## Implications for Cycle 3's RQs

**RQ-1 — settled at fixture scope.** A2 + script input does not match A3's grounding on the cycle-2 README-review fixture. ADR-011's Cycle 2 boundary refinement holds: the ADR is defensible as default for tasks fitting Anthropic's threshold conditions, but is not a ceiling for task classes where factual grounding via deterministic checks is part of the success criterion. The architectural lesson at this fixture is *not* "augment prompt-steering with deterministic tool outputs" alone — that hypothesis is refuted; *some structural framing* (whether A3's heterogeneous-reviewer ensemble, cheap-tier-with-directed-prompting, or other shapes) is required for the semantic-consistency bug class.

**RQ-2 — empirical evidence for the four-priorities frame as load-bearing on this fixture.** The frame produces a frame-divergent recommendation against a performance-only frame on Spike A's data. Per the falsification criterion: not retired. The broader claim ("the four-priorities frame is universally load-bearing for agent-design choices") is not supported — additional task-class evidence required.

**RQ-3 (multi-turn reliability) — separate question.** Spike A is a single-ask (single-prompt-per-trial) test on a static fixture. RQ-3's multi-turn fixture (per Cycle 3 plan: tau-bench subset + small real-session probe) is the next move and probes a structurally different regime.

---

## Cycle 4 hooks

1. **Cross-tier complementarity on other task classes.** The most consequential question Spike A surfaces but does not answer: does the cheap-tier-with-directed-prompting + frontier-tier complementarity hold on multi-turn coding sessions? On code review (not just doc review)? On other documentation styles (developer docs, API references, runbooks)? A Cycle 4 spike battery testing this directly would provide novel empirical evidence the literature does not yet have.

2. **Prompt-direction-vs-topology generalization.** Within the cheap tier, prompt direction dominated architectural framing on this fixture's bug class. The Cycle 4 question: does this generalize, or is it fixture-specific? The "floor question" (P3 from research-design-review-cycle-3, recorded as Cycle-4 feed-forward): *"what value does any LLM contribute once the script has run?"* — the related variant: *"what value does ensemble topology add over a single LLM with a well-engineered prompt?"*

3. **Heterogeneity-uncorrelated-errors at cross-tier scope.** Sun et al. (2025) and Ding et al. (2024) characterize the mechanism within-tier-across-families. Spike A's evidence suggests it extends across tier-and-capability differences. A directly-designed Cycle 4 spike could test this with controlled heterogeneity dimensions (same model different prompts; same family different sizes; different families same tier; different tiers).

4. **Re-run of A3 against current OpenCode Zen state.** Spike A's comparison to A3 is documentary. A Cycle 4 run of A3 with the current OpenCode Zen state would verify the comparison baseline holds.

5. **Direct-API frontier reference.** The frontier facsimile via subagent is a known-bounded surrogate. If Cycle 4 has paid-tier API access available, a direct-API Sonnet 4.6 (or successor) reference would tighten the cross-tier comparison.

6. **The four-priorities frame at broader resolution.** Spike A's frame-divergence finding is at single-fixture resolution. Cycle 4 should test the frame against a broader fixture set; if frame-divergent recommendations are common, the frame is more universally load-bearing; if rare, the frame's load-bearing role is fixture-specific.

---

## Spike code disposition

**Retain until corpus close (corpus-scoped override of rdd spike discipline; recorded as practitioner policy 2026-05-01).** The agentic-serving corpus retains all spike artifacts — harnesses, deterministic scripts, per-trial outputs, run logs, reconstructions — until the corpus eventually closes (graduate phase or explicit practitioner decision), rather than deleting at cycle close per the standard rdd discipline. The override was adopted after Cycle 3 hit substantive friction reconstructing Cycle 2's deleted spike artifacts: A2's exact prompt was unrecoverable from git, the script-agent code had to be rebuilt from essay 003's prose description, and two scope conditions had to be documented in this synthesis as a direct consequence. Retention cost (a `scratch/` directory of a few MB across the corpus) is far smaller than the reconstruction cost when a downstream cycle needs the artifacts as comparison ground.

Spike A's preserved artifacts:

- `scratch/spike-a-cycle3-a2-script-input/spike_a_harness.py` — Spike A harness (Path A: llm-orc production model factory; 7 arms; dry-run + live modes).
- `scratch/spike-a-cycle3-a2-script-input/deterministic_analyzer.py` — script-agent reconstruction (link validity / canonical-section presence / code-block parseability).
- `scratch/spike-a-cycle3-a2-script-input/script-output.txt` — script's deterministic output on the README fixture.
- `scratch/spike-a-cycle3-a2-script-input/run-log.txt` and `run-log-followup.txt` — harness run logs.
- `scratch/spike-a-cycle3-a2-script-input/trials/` — all per-trial outputs (cheap-tier JSON files; frontier-tier markdown files).

The spike-discipline note about "spike code is disposable philosophically" still applies — the harness and analyzer are not production code, are not imported from anywhere, and would not be promoted into the library. The override changes only the deletion timing (corpus-close, not cycle-close).

---

## Connections to Cycle 3's other research

- **`004a-lit-review-agent-design.md`** — lit-review findings on heterogeneous-multi-agent residual case (OneFlow / Lee et al.), CLEAR cost-axis evidence, Routine + Compiled AI script-as-orchestrator literature, and the PAE / corrupt-success operationalization that informed Decision A. Spike A's cross-tier finding is a novel extension the literature surfaced but did not directly test.
- **Cycle 3 cycle-status (`housekeeping/cycle-status.md`)** — RQ definitions, decision adoptions, scope conditions.
- **Cycle 2 archive (`cycle-archive/cycle-2-multi-turn-and-composition.md`) §Feed-Forward Signals** — the inheritance index Spike A's RQ-1 isolation test addresses (items #20, #21).
- **Susceptibility snapshot from Cycle 2 (`housekeeping/audits/susceptibility-snapshot-cycle-2-research.md`)** — the specific in-cycle grounding action Spike A executed.
- **Research-design review (`housekeeping/audits/research-design-review-cycle-3.md`)** — the Decisions A/B/C/D adopted before spike execution and the contamination caveat Decision A's PAE framing helped surface.

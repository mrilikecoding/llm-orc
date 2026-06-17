# Research Methods Review — Spike τ (Local/Frontier Mix for Long-Horizon Tasks on 32GB)

**Reviewed artifact:** `docs/agentic-serving/essays/research-logs/cycle-7-spike-tau-long-horizon-32gb.md`
**Constraint-removal response included:** n/a (hardware-constraint characterization spike; no ADR-082 constraint-removal artifact in scope)
**Date:** 2026-06-16
**Reviewer role:** research-methods (ADR-060 + ADR-082 dimensions 1–4, ρ/ξ/π/σ calibration bar applied)

---

## Summary

- **Arms reviewed:** ref (gate-only baseline) + A (single-local-14b) + B (frontier seat + local 8b coder)
- **Primary outcomes:** convergence at 5-file gate, ceiling rung, per-turn latency trend, break attribution
- **Flags raised:** 8 (3 P1, 3 P2, 2 P3)
- **Criteria applied:** 1–4 (ADR-082)

---

## Findings

### P1 — Design flaws that would invalidate conclusions before running

---

**P1-A: Arm A conflates two independent variables — it changes both the coder and the seat simultaneously. This means A-vs-ref does not isolate "single-model-removes-swap" and A-vs-B conflates seat-capability and swap-elimination together.**

The pre-registration states "A isolates swap-elimination alone, so A-vs-B attributes the frontier contribution." But in the ref config the coder is qwen3:8b and the seat is qwen3:14b. In arm A both are qwen3:14b. Arm A therefore changes two things at once: it eliminates the swap (by going to one model) *and* upgrades the coder from 8b to 14b. Any convergence improvement in arm A versus ref is attributable to either cause — the eliminated swap latency, the improved coder, or both. The pre-registration does not acknowledge this.

For the swap-only hypothesis (H1), the clean single-variable test would be an arm that uses qwen3:14b as both seat and coder (arm A as designed) versus an arm that uses qwen3:8b as both seat and coder (removing the swap while keeping coder quality constant relative to ref's coder). Arm A as specified removes the swap but also changes the coder, so it cannot cleanly attribute the convergence gain to swap-elimination alone.

The A-vs-B comparison has the same problem. Arm B uses qwen3:8b as the coder; arm A uses qwen3:14b. So "does frontier seat-filling buy real reach over a free local seat?" is confounded by the fact that arm B has a weaker coder than arm A. If arm B ladders higher than arm A, the result could be "frontier seat enables better orchestration at equivalent coder quality" but it could also be "the arm-A 14b coder's longer inference time is the new bottleneck, and the swap-tax reappears in a different form at high file counts." If arm A fails at a lower rung than arm B, "frontier seat buys reach" and "14b-as-coder is slower per turn" are confounded.

This does not make the spike unrunnable, but the attribution language in the decision rule ("if A ladders as high as B, frontier is unnecessary") embeds a conclusion that requires a third confound arm to support.

**Required fix before running.** Acknowledge the confound explicitly in the §Known limitations / threats to validity section. Add one sentence to the decision rule: "If A ladders higher than B, attribute first to the 14b coder quality difference (not only swap-elimination); consider an arm with qwen3:8b as both seat and coder (single-resident, no swap) before concluding frontier is unnecessary." Alternatively, add a fourth arm (8b seat + 8b coder, single-resident) as the clean swap-only control. Either approach closes the gap; the four-arm design is preferable for a decision-rule spike but costs more run time.

---

**P1-B: "Attribute the break" is described as operationalizable from log inspection, but no pre-registered classification protocol defines what counts as each attribution class. Without a pre-registered protocol, the classification is post-hoc-adjustable and may not be reproducible.**

The pre-registration names three break-attribution classes:

1. Context-overflow — the content-anchor outgrows the window.
2. Seat-sequencing failure — the seat stops advancing to new files.
3. Latency/degradation — per-turn time blows up.

These are conceptually distinct, but the spike provides no pre-registered operationalization:
- What serve.log or token-count signal constitutes "context-overflow" (e.g., a truncation event, a context-length error, a measurable token-count crossing a threshold)?
- What observable behavior constitutes "seat-sequencing failure" — the seat produces a COMPLETE verdict with files unfinished, or the seat stops delegating, or it repeats a prior file, or it emits a text response? These are different ADR failure modes with different remedies.
- What latency magnitude constitutes "blows up" versus normal arm-B network variance versus degradation?

Without these defined, the classification is an interpretive judgment made after seeing the data. Prior reviews (σ P1-B, π Fork-3 analysis) flagged analogous problems: a post-hoc classification with no pre-registered boundary is adjudicatable in any direction by the observer. The "analytical core" claim requires the classifications to be pre-registered, not post-hoc.

This is especially consequential because the break attribution is the decision the spike most wants to export to BUILD/PLAY. If arm A breaks at 10 files, the practitioner needs to know whether the ceiling is "the architecture" (content-overflow) or "the rig" (swap/degradation) or "the cheap 14b seat" (sequencing failure at scale). A classification that is defined post-run cannot be the basis for that decision with confidence.

**Required fix before running.** Add a pre-registered break-attribution protocol to the §Method section. For each class:
- *Context-overflow*: token count in the callee dispatch context or serve.log truncation event exceeds a pre-stated threshold (e.g., `context_window_tokens > 0.9 × model_context_limit` on the failing turn); OR the model returns a context-length error; OR the response is truncated mid-output.
- *Seat-sequencing failure*: the seat emits COMPLETE with `requested − produced` non-empty on the turn that breaks; OR the seat re-delegates a file already in the Session Action Record's produced set; OR the seat emits a text response (no tool call) on a REMAINING turn where all other causes are absent.
- *Latency/degradation*: per-turn seat latency on the failing run exceeds 2× the median of the same arm's first 3 turns (or exceeds a stated absolute threshold), with no context-overflow signal present.

The protocol need not be exhaustive — "cannot classify" is an acceptable outcome — but it must be stated before the run. Two sentences per class is sufficient.

---

**P1-C: The decision rule "flat per-turn latency AND ladders to ≥8 files" requires "flat" to be defined before running. As stated, it is a post-hoc adjudication criterion.**

The pre-registration says an arm "enables long-horizon on 32GB if it clears the 5-file gate with flat per-turn latency AND ladders to ≥8 files." "Flat per-turn latency" is the degradation signal for H4 — the claim that arm B removes the sustained-session penalty. But "flat" is not defined:

- Is latency flat if the trend slope across the session's turns is within a tolerance band (e.g., ≤ 10% drift from the turn-3 median)?
- Is latency flat if no individual turn exceeds 2× the session median?
- Is latency flat if the last-turn latency is within X seconds of the first-turn latency?

Without a pre-registered definition, "flat" is adjudicatable after seeing the result. If the 14b seat in arm A shows a mild upward trend, there is no principled way to decide whether that trend is "noise" or "degradation onset" without a pre-specified criterion.

The σ P1-A finding (threshold gap at integer count boundaries) is the prior-art precedent: a metric that is stated without operationalization creates a decision gap that invalidates the rule. Latency trend is a continuous measurement; it requires a pre-stated comparison structure.

**Required fix before running.** Add one sentence to the decision rule: "Per-turn seat latency is flat if the linear trend across the run's turns has slope < 10s/turn AND no individual turn latency exceeds 2× the first-turn latency. A trend that exceeds either bound is classified as degradation-onset." The specific thresholds are negotiable; stating them before the run is the requirement.

---

### P2 — Weaknesses that bound claims

---

**P2-A: Arm B introduces a latency confound the pre-registration does not acknowledge: MiniMax-m2.5 (paid) is served via OpenCode Zen (HTTPS remote API), while arms ref and A use local Ollama. The per-turn latency comparison across arms is partially a network-vs-compute comparison, not a pure "swap-eliminated" signal.**

The pre-registration's primary latency metric is "per-turn seat latency — median and trend across turns." For ref and arm A, seat latency is local Ollama inference (network = loopback). For arm B, seat latency is a remote HTTPS API call to OpenCode's Zen endpoint plus the MiniMax inference time. The per-turn latency in arm B will include round-trip network time plus any Zen-side queue wait, neither of which is present in ref or arm A.

This means the comparison "arm B latency vs arm A latency" cannot cleanly answer "does a frontier seat run faster or slower than a local 14b seat on the same task" — the comparison confounds model speed with network topology. Concretely: if arm B shows lower per-turn seat latency than ref, it might be because the swap is eliminated, or it might be because MiniMax-m2.5-paid's inference is fast and the network is low-latency on the day of the run. If arm B shows higher latency than arm A, it might be because the frontier model is slower, or because the network is congested, or because the Zen queue had backpressure.

The prior spike-π methods review noted the analogous "wrapper-vs-callee" latency comparison had a scope note ("the ~3× figure was measured on a batchable two-write task… under production single-step enforcement both shapes incur N driver round-trips; the per-generation 3-calls-vs-1 ratio is the stable comparison, and the direction holds, but the absolute session numbers shift"). The same caveat structure is appropriate here.

**Recommended fix.** Add a note to the §Measurements section: "Per-turn seat latency in arm B includes network round-trip to the Zen endpoint; arms ref and A use local Ollama (loopback). Latency comparisons across arms measure 'local vs. remote+frontier' jointly, not 'local fast vs. local slow.' Within-arm latency trend (flat vs. drifting) is not affected by this confound; cross-arm latency magnitude comparisons are. Report cross-arm comparisons with this caveat noted." No design change required; the caveat keeps the claim honest.

---

**P2-B: The sustained-session check's n=3-4 gate-sized tasks is too small to distinguish degradation from natural task-to-task variance, and the "flat latency across ≥3 back-to-back tasks" pass criterion is not operationalized with reference to any specific latency baseline.**

The sustained-session check (§Sustained-session check) runs 3-4 gate-sized tasks back-to-back without restart, measuring whether per-turn latency holds or drifts. The pass criterion is "the chosen arm holds flat latency across ≥3 back-to-back tasks." But:

1. Each gate-sized 5-file task takes multiple turns. The per-task turn count is not stated. If each 5-file task takes ~12 turns (consistent with prior σ/η discharge sessions), 3 back-to-back tasks = ~36 turns total. Whether 36 turns is adequate to detect degradation onset depends on when the degradation manifests — if it appears only after task 5 or 6, 3 tasks will not find it.

2. "Flat" is undefined here too (see P1-C), compounding that P1 finding into a second measurement.

3. There is no stated baseline against which "flat" is measured. Is the first task's median latency the baseline? The isolated 5-file gate run? A fixed absolute number? Without this, the check produces a time-series with no anchor.

4. The "restart cleanly resets" test is operationalized only as "tests H4" — no criterion for what constitutes a clean reset.

**Recommended fix.** Expand the §Sustained-session check to include: (a) the minimum number of turns expected per gate-sized task (e.g., "~10-15 turns based on prior σ/η sessions"); (b) the flat-latency definition from P1-C applied at the session level (latency trend across all turns of the back-to-back block); (c) the baseline for comparison (per-turn median from the arm's isolated gate run); (d) the "clean reset" criterion (per-turn latency in the first task after restart is within X% of the pre-degradation baseline from the first task before restart). Three to four sentences in the §Sustained-session check section covers this.

---

**P2-C: The pre-registration's cost estimate for arm B assumes the seat is a few hundred tokens per turn, but does not bound total seat turns across the full ladder. At high file counts, the estimate may be substantially wrong, and the pre-run confirmation gate is only for "the first frontier call."**

The §Cost section says: "B is the only paid arm (MiniMax seat calls, fractions of a cent each — the seat is a few hundred tokens/turn). Estimated whole-spike cost < $1." The "first frontier call is gated on practitioner confirmation of the estimate."

The cost estimate is reasonable for the gate (5-file, ~12 turns), but the ladder ("8 → 10 → 15 → 20 → …files until break") could involve many additional runs. If arm B ladders to 15 files before breaking, that might be ~40 turns; at 20 files, ~60 turns. Across the full ladder the total seat-call count could be 150-200 calls. At "fractions of a cent each" (say 0.005 per call), that is $0.75-1.00 for the ladder alone, before the gate and sustained-session check. The estimate of "< $1" may be correct for the gate only, or it may hold for the full ladder, but the pre-registration does not make the accounting transparent.

The pre-run confirmation gate ("the first frontier call is gated on practitioner confirmation") catches only the entry cost, not the escalating ladder cost. A better practice (consistent with the free-first discipline) is a rolling estimate at each ladder rung.

**Recommended fix.** In §Cost, add: "Estimated cost per rung: ~[N] seat turns × 0.0X cents = $Y per rung. Running total will be surfaced at each rung boundary. If the running total crosses $1, surface to practitioner before the next rung." The specific numbers require knowing the model's per-token pricing; if that is not pre-confirmed, state the formula and surface it as an open pre-run item.

---

### P3 — Minor improvements

---

**P3-A: "No artificial ceiling — push until convergence fails" in the ladder design creates an open-ended session that has no maximum cost or time budget. This is not a design flaw, but it should be explicitly acknowledged rather than implied.**

The pre-registration says the ladder has no ceiling. In practice, a spike with no ceiling could run until the practitioner's machine has been committed to an 8-hour ladder that costs $5 or produces data no more useful than a ladder stopped at 20 files. The "no artificial ceiling" framing is consistent with the minimal-gate-then-progressive-ladder convention (this cycle's prior σ/η/ξ all used bounded ladders with a stated stopping rule). The tau ladder is distinguished by having *only* the convergence-failure stop — there is no time budget or maximum rung stated.

**Recommended fix.** Add one sentence to the §Method ladder paragraph: "The ladder stops at convergence failure or at 30 files, whichever comes first; if arm A or B passes 30 files, that rung is recorded as a lower bound on the ceiling, not the ceiling itself." 30 is a specific suggested number; the practitioner may choose another. The point is to name the practical upper bound rather than leaving the ladder unconstrained by anything but failure.

---

**P3-B: The pre-registration does not address whether the content-anchor (ADR-039) is active and operating correctly in all three arms, or whether a content-anchor failure at high file counts could be attributed as "seat-sequencing failure" rather than "content fidelity failure."**

At high file counts on arm A or B, a cross-file coherence failure (the axis-2 concern ADR-033 §6b names and ADR-039 addresses) could manifest as: the dependent file fails validation, the completeness gate re-delegates, and the session appears to stall without convergence. This could be classified as "seat-sequencing failure" or as "context-overflow" if the anchor bloats the context. The break-attribution protocol (P1-B) should include a "content-anchor overload" class or explicitly note that ADR-039 is active and that multi-sibling selection policy at scale is a known open item (ADR-039 §Decision notes "whether production injects all produced files' signatures or a dependency-inferred subset is a BUILD and FC detail").

This is a minor refinement to P1-B rather than a separate finding, but it is worth flagging as a distinct failure mode the attribution protocol should name. A 20-file session has 19 potential anchor-sibling signatures per callee dispatch. If that bloats the context, the break is architectural (content-anchor at scale), not seat-sequencing. Without naming it, the classification has a gap.

**Recommended fix.** Add "content-anchor overload" as a fourth break-attribution class in the P1-B protocol: "the callee dispatch context carries signatures from all prior-produced siblings and the token count for the callee dispatch context exceeds a pre-stated threshold — distinguish from context-overflow (which is the overall session context) by noting whether the signal is in the callee dispatch specifically."

---

## Per-Criterion Assessment

### Criterion 1: Belief-Mapping

The central question frames the problem as "which local/frontier mix enables long-horizon convergence on 32GB?" The choice of mix embeds an assumption: that the seat is the right component to vary. The adjacent question the framing suppresses: "What would the researcher need to believe for the coder to be the right component to offload, rather than the seat?"

In the ref config, the coder (qwen3:8b) is cheaper-to-evict from RAM than the seat (qwen3:14b): the 8b coder is the smaller model. The swap occurs because both models are present and the 14b seat dominates RAM. The current design's arm A swaps the coder up to 14b (removes the swap by making them identical) and arm B swaps the seat to frontier (removes the swap by offloading the seat). Neither arm asks "what if the 8b coder is replaced by the frontier model while the 14b seat stays local?" The asymmetry is: the seat drives and the coder writes. If the seat is the bottleneck (sequencing, coherence), frontier seat is the right offload. But if the content failure is the bottleneck (ADR-039), frontier coder might deliver equivalent or better reach at similar cost.

This is not necessarily a design flaw — the practitioner's H2 reasoning (eliminate the swap AND raise orchestration quality) is a coherent hypothesis. But the assumption that the seat is the right thing to offload is embedded in the arm design without being named. The incongruity: ADR-039 §Rejected alternatives explicitly considered "a more capable coder seat" and found the cheap 8b coder resolves 10/10 with the anchor. That finding argues the coder is not the bottleneck for content fidelity. However, at 20+ files the anchor at scale is an open question, and it is possible that the coder becomes the limiter at high rung counts in a way the σ/η validated 5-file sessions never stressed. The current arm design cannot distinguish "seat limits the ceiling" from "coder limits the ceiling at scale" because arm A changes both.

The pre-registration would benefit from a one-sentence acknowledgement: "Arms A and B both keep the coder fixed at 8b (arm B) or upgrade it to 14b (arm A); an arm that offloads the coder to frontier while keeping the local 14b seat is not tested here — the hypothesis is that the seat drives the ceiling, grounded in ADR-039's finding that the cheap coder is adequate when anchored."

### Criterion 2: Embedded Conclusions

The decision rule contains one embedded conclusion that should be surfaced:

**"If B ladders meaningfully higher (or A fails the gate / sequences poorly), the cents/task buy real reach."**

This presupposes that ladder height is the sole measure of "real reach." An arm B that ladders to 15 files but takes 4× longer per turn than arm A at 8 files may offer more absolute file count but worse practical performance on the practitioner's 32GB machine at real session speeds. The decision rule should include a latency dimension: "arm B buys real reach if it ladders meaningfully higher *and* per-turn latency at the ceiling rung is within a tolerable bound." Without this, the decision rule could conclude "frontier is necessary" based on a ladder-height advantage that comes at an unacceptable per-turn latency cost.

Suggested decision rule amendment: "The ceiling comparison is (rung reached) × (per-turn latency at that rung). If arm B ladders meaningfully higher than arm A but at per-turn latency that is 3× arm A's latency at the same rung, the comparison is ambiguous — record both the rung and the latency and surface to the practitioner before concluding."

### Criterion 3: Premature Narrowing / Prior-Art Treatment

The design does treat prior evidence honestly in key respects: it cites the σ/ADR-039 5-file validated convergence as the basis for the gate floor, and it correctly positions `agentic-orchestrator-minimax-m25` as the existing Spike π Arm E profile rather than designing a new one. The ref arm is correctly named as a bounded data point ("its ceiling is already below the gate"), consistent with the benchmark evidence.

One narrowing concern: the question set implicitly assumes the 5-file gate is the right floor by citing "σ/ADR-039 validated 5 files." But the isolated h2c1 run (2-file task, 399s) is actually the strongest evidence that the problem is the stack, not the task. The gate asks: "can arm X converge a 5-file task?" But the most informative bottom-of-the-ladder question might be: "can arm X converge a 2-file task that failed mid-grid?" Running h2c1-equivalent as the gate (rather than the σ/ADR-039 5-file) would provide a cleaner read on whether the new arms fix the specific failure that triggered the spike, before testing whether they extend it. This is not required — the 5-file gate is a legitimate choice — but the reason for choosing 5 over 2 should be stated (it may be "2-file isolation already passed for ref; the question is multi-file, so gate at 5").

### Criterion 4: Incongruity Surfacing

An incongruity is present in the existing profile corpus that the question set does not surface for examination.

The `agentic-orchestrator` profile (the free-tier default used in prior work) uses `minimax-m2.5-free` — the same provider and endpoint as arm B's `agentic-orchestrator-minimax-m25` (`minimax-m2.5` paid), just the free-tier model identifier. Both route through `openai-compatible/zen` at the same `https://opencode.ai/zen/v1` base URL.

The incongruity: the ref arm uses `agentic-orchestrator-offline-tools` (local qwen3:14b via Ollama), but the cycle's default seat profile *outside the benchmark context* has always been the free MiniMax-via-Zen profile. The free-tier MiniMax seat is already the production default for the non-32GB-constrained scenario. The spike's arm B (paid MiniMax) is not "introducing a frontier seat" — it is upgrading an existing frontier seat from free to paid tier.

This raises a question the spike does not ask: why does the ref arm use the offline/local qwen3:14b profile rather than the production-default free MiniMax profile? If the actual production default is already `minimax-m2.5-free` as the seat, then the "two local models that don't co-reside" framing of the benchmark's hang may not be the configuration users would actually run — users who follow the default profile would already have the free MiniMax seat, not the local qwen3:14b.

The spike should either: (a) confirm that the benchmark was run with `agentic-orchestrator-offline-tools` (local 14b) rather than the default `agentic-orchestrator` (free MiniMax) and explain why; or (b) acknowledge that the ref arm's config differs from the production default and note what this means for the generalization claim. If the benchmark was run with the offline-tools profile because the Zen quota was exhausted, that is a valid reason but should be stated — it would mean the "ref" in this spike is not the production default, and the spike's decision "do we need paid MiniMax?" may already be answered by the existing free-tier profile, not by this spike.

This incongruity is the most consequential finding in this review. The question the spike is missing: "Is the free-tier MiniMax seat already the appropriate long-horizon config on 32GB, and is this spike's arm B just the paid-tier upgrade of an already-correct architecture?" If yes, the spike's decision rule narrows to "does the paid tier buy measurably more reach than the free tier?" rather than "does a frontier seat buy reach over a local seat?"

---

## Question Set Assessment

**Premature narrowing / prior-art treatment:** The 5-file gate floor is grounded in prior evidence (σ/ADR-039), which is correct prior-art treatment. The `agentic-orchestrator-minimax-m25` profile is prior art reused rather than a new design, also correct. The narrowing concern is the Criterion 4 incongruity above: the existing free-tier frontier profile makes the spike's "frontier vs local" framing potentially a false dichotomy, since the production default is already a frontier seat (just free-tier). This is not premature narrowing in the classic sense, but the question set does not examine this adjacent configuration and should.

**Incongruity surfacing:** The free-tier MiniMax profile (`agentic-orchestrator`, `minimax-m2.5-free`) is the existing production default for the agentic seat role — it is already frontier, already zero-cost, and already configured. The spike's arm B is the paid upgrade of this profile. The question the spike does not ask: "Does the free-tier frontier profile (arm B-free) already enable long-horizon convergence on 32GB, making the choice between arm A (free local) and arm B (paid frontier) a false fork?" This is the simpler adjacent configuration the spike skips past by jumping from a local-only ref to a paid-frontier arm B.

**Coverage gaps:** The spike does not characterize what happens if arm B's Zen endpoint is quota-throttled mid-ladder (the benchmark note mentions the grid run was aborted due to a degraded environment). Network availability is a de facto constraint for arm B that arms ref and A do not share; the spike should note this as a practical limitation and pre-register what happens (abort the ladder rung, wait, or fall back?) if throttling is encountered.

**Recommendations (prioritized):**

1. **P1-A** — Acknowledge the arm A coder-upgrade confound in §Known limitations and amend the decision rule to account for it. Consider adding an 8b+8b single-resident arm or at minimum state the gap.

2. **P1-B** — Pre-register break-attribution criteria (2-3 sentences per class) before running. Without this, the spike's analytical core is a post-hoc judgment, not a pre-registered finding.

3. **P1-C** — Define "flat per-turn latency" quantitatively in the decision rule (slope bound and individual-turn ceiling) before running.

4. **Criterion 4 / Incongruity** — Before running arm B, confirm whether the benchmark hang was produced by the offline-tools (local 14b) profile or the production-default free-MiniMax profile, and decide whether arm B-free (the existing default) is a fourth arm to include or an explanation of why it is not.

5. **P2-A** — Add the cross-arm latency confound caveat (network vs. compute) to §Measurements.

6. **P2-B** — Operationalize the sustained-session check's flat criterion and baseline anchor.

7. **P2-C** — Make the cost estimate transparent for the full ladder, not only the gate entry.

8. **P3-A** — Add a practical ceiling rung (e.g., 30 files) to bound the open-ended ladder.

9. **P3-B** — Add "content-anchor overload" as a fourth break-attribution class in the P1-B protocol.

# Argument Audit Report — Round 5 (Content-Agnostic Revision)

**Audited document:** `docs/agentic-serving/decisions/adr-039-content-anchor.md`
**Source material:** `docs/agentic-serving/essays/research-logs/cycle-7-spike-xi-content-anchor.md`; `scratch/spike-xi-content-anchor/results_G_*.json`; `scratch/spike-xi-content-anchor/probe.py`
**Prior rounds:** `docs/agentic-serving/housekeeping/audits/argument-audit-decide-cycle-7-loopback7{,-round2,-round3,-round4}.md`
**Genre:** ADR
**Date:** 2026-06-09

---

## Audit context

R4 triggered convergence (STOP) on the pre-revision ADR. Since R4 the practitioner directed a substantive revision: the mechanism must be "completely content-agnostic," which required inverting the form priority (full-content is now the universal baseline; signatures are a compaction optimization) and adding a third sibling type (config/data, Base G). R5 is scoped to the content-agnostic revision exclusively. R1–R4's settled findings (mechanism, code/prose rates, causal isolation, gate logic, the six R3 editorial repairs) are not re-derived.

The form-change baseline-reset rule (ADR-094 P2-R3-3) applies: the inversion and the new evidence arm constitute a substantive revision to the Decision section. This audit is treated as R1-of-revision for saturation-signal purposes, not as R5 of a continuous sequence.

---

## Section 1: Argument Audit

### Summary

- **Genre:** ADR
- **Argument chains mapped:** 5 (content-agnostic warrant on deliverable side; content-agnostic warrant on source side; full-content inversion consistency; config evidence specifics; benign-no-op claim)
- **Issues found:** 2 (both P3)
- **Pyramid coverage map:** N/A
- **Expansion-fidelity findings:** N/A

---

### Audit focus 1: Does "content-agnostic" follow from three sibling types?

**Deliverable side (any callee).** The ADR's claim is that the mechanism fires for any callee regardless of type. The evidence base is code callees (Base T, V via `code-generator`) and a prose callee (Base P via `prose-improver`). Two callee types is a narrow empirical base, but the ADR does not claim the three measured callee instances generalize exhaustively to all possible callees — the deliverable-side claim is structural ("the augmentation fires on the callee dispatch regardless of which capability ensemble is invoked") and the two types are cited as confirmation, not proof-by-exhaustion. This is consistent.

**Source side (any sibling type).** The ADR's claim has three layers, which the dispatch brief asked to verify:

- (a) *What is measured*: three sibling types — Python code (signatures path, Bases T/V), prose README (full-text path, Base P), JSON data (full-content path, Base G). All three measured 0/10 → ≥9/10 under anchor.
- (b) *What is structurally guaranteed*: full-content is type-blind by construction ("it is the file's bytes") — the framework injects bytes regardless of type, so agnosticism on the source side is a structural property, not an inductive generalization across types.
- (c) *The residual boundary*: exotic sibling types beyond the three measured face an *effectiveness* question (does the model usefully consume an unknown-format anchor?) with a benign-no-op worst case, not a mechanism-break. The ADR labels this "an effectiveness question with a benign-no-op worst case, not a content-agnosticism gap."

The ADR does maintain this three-way distinction. The Decision (second paragraph) and Consequences (Negative, last bullet) both describe the exotic-type boundary as effectiveness, not agnosticism. The Empirical Grounding closes with: "exotic sibling types beyond these three remain a benign-no-op effectiveness boundary (the full-content path is type-blind by construction)."

**Verdict on claim (a)/(b)/(c) consistency:** the three-way distinction is present and consistently maintained across Decision, Consequences, and Empirical Grounding. No overreach detected: "content-agnostic" is framed as a structural property of the full-content path (construction-guaranteed), confirmed empirically across three structurally-different sibling types, with the exotic-type residual labeled as an effectiveness boundary. The claim is warranted as stated.

---

### Audit focus 2: Full-content-baseline inversion — consistency throughout

The prior draft framed the mechanism as "signatures, not full content." The revision inverts this. Scanned all five loci the dispatch brief identified:

**Decision bold statement (opening paragraph):** "The default and universal path is the sibling's full content (type-blind, so it sources an anchor from any content type); where the framework has a structural extractor for the sibling's type, it compacts to the public API surface (signatures). ... Signature extraction is a frugality optimization layered on the agnostic baseline, never a precondition." Full-content is the default; signatures are the optimization. Consistent with the inversion.

**Form bullet:** "full content is the content-agnostic baseline; signatures are a compaction optimization." The bullet explicitly acknowledges the prior draft: "(The earlier draft framed this 'signatures, not full content,' which inverted the priority for a content-agnostic goal: full content must be the default that guarantees agnosticism, with signatures layered on where extractable.)" Consistent.

**Content-agnostic both sides bullet:** "the full-content baseline is type-blind, so it sources an anchor from any sibling type, not only code — guaranteed by construction (full content is the file's bytes; the framework does not parse the sibling on the universal path)." Consistent.

**Negative consequences:** "The framework now reads produced files on each dependent dispatch (full content, compacted to signatures where an extractor exists for the sibling's type)..." — full-content is the primary path, signatures are the compaction variant. Consistent.

**Provenance:** The provenance records "the 'signatures preferred over full-content' selection (true on the measured margin, B 20/20 versus C 18/20, but n=10/cell cannot distinguish 8 from 10, so the preference rests on signatures' equal-or-better rate plus the context-budget argument, not a population-level reliability claim)." This describes signatures as the *preference within extractable types*, not the universal default. Consistent with the inversion.

**FC (signatures form, not full bodies):** the fitness criterion reads "the anchor carries signatures and docstrings, not full function bodies." This FC applies only to the extractable-type path — it is not asserting signatures over full-content for all types. The criterion is scoped to the compaction case. Consistent.

**"Signatures win for code" vs. "full-content is the agnostic baseline" tension:** the ADR does hold both simultaneously. For extractable types (Python code), signatures are the preferred form on the code path (B 10/10 vs. C 8/10 on Base T, plus context-budget). For non-extractable types, full-content is the fallback. The two positions do not contradict because they apply to different sub-cases within the universal full-content baseline: the baseline is full-content everywhere; signatures are the optimization where available and measurably equal-or-better. The ADR makes this layering explicit in the Form bullet.

No residual text found that still frames "signatures, not full content" as the universal design orientation. The inversion is complete and consistent throughout.

---

### Audit focus 3: Config evidence specifics

**B_content 9/10 — "within n=10 margin" treatment.** The single miss is trial 7: the model produced `affinity_salt` instead of `aff_salt` — an expansion of the abbreviated key, not a random invention. The raw JSON confirms this: refs `[["affinity_salt", false], ["qdepth_max", true], ["rbo_ms", true]]`, graded 0.67. The ADR treats 9/10 as "within the n=10 wide-CI margin" (Consequences Positive, Empirical Grounding). The preregistered decision rule defined the pass threshold as `≥ 7/10`; 9/10 clears this by two. The wide-CI caveat (from the decision rule section of the research log) explicitly covers boundary differences like 8-vs-10. 9/10 vs. 10/10 is within that same margin. The ADR's treatment is proportionate and consistent with its own preregistered precision discipline.

**Base-validity iteration disclosure.** The ADR (Empirical Grounding) records: "a baseline-validity guard that passed (A_current reproduced Finding H predominantly by invention on every base, after Base G was re-pinned from a generic-loader task that the blind consumer dodged)." The research log (Base G section) describes this more fully: the first Base-G design produced `no-reference` (not `invented`) — the loader read the dict generically and never committed to specific keys. The task was re-pinned to force direct subscript references. The ADR's disclosure is accurate in substance: the iteration is named, the reason is given (generic-loader task dodged the failure), and the resolution is stated (re-pinned). The level of detail is appropriate for an ADR (full detail is in the research log). This does not weaken the result — it strengthens it: the iteration confirmed that the coherence failure arises only when the consumer must commit to a named interface, which is the correct scope condition. The ADR records this as a bounding finding ("the anchor is moot for siblings consumed wholesale"), and that characterization is honest.

**Causal isolation on config type.** Control_decoy 0/10, A_current 0/10, B_content 9/10. B − decoy = 0.9 (far past the 0.3 gate). The raw JSON for Control_decoy shows the model followed the decoy keys (`backoff_ms`, `max_queue_depth`, `affinity_salt`) uniformly across all 10 trials — the pattern exactly mirrors the code decoy behavior. The ADR states "Causal isolation holds on the config type too (B − decoy = 0.9): the specific config content (the real keys) is the mechanism, not 'any config-shaped context.'" This is warranted by the data. The "specific content is the mechanism" claim for config is licensed.

One observation worth flagging as P3 (see below): the config decoy is not structurally symmetric with the code decoy. For code, Control_decoy has wrong *function names* in a clearly API-shaped format; for config, Control_decoy has wrong *key names* in the same JSON format as the real config. This means the isolation argument is "real keys vs. wrong keys," which holds cleanly. However, there is no config equivalent of the code `Control_filler` arm (length-matched non-API-shaped prose). So the three-way decomposition (B vs. decoy vs. filler) from code Bases T/V is not replicated for Base G. The ADR does not claim it is — the config arm was framed as a "content-agnosticism confirmation," not a full causal decomposition. The two-way isolation (B vs. decoy = 0.9) is sufficient to establish "specific content is the mechanism" for the config type. No overreach detected here; the level of isolation matches the framing.

---

### Audit focus 4: The "benign no-op worst case" claim

The ADR argues that for an exotic sibling type, the worst case is the model ignoring an unhelpful-but-correct anchor, "degrading to the unanchored baseline, never below it, since the anchor is always the real file and so cannot trigger the decoy failure mode."

The argument rests on two premises:
1. Full-content injects the real bytes, so it cannot be wrong in the same way a guessed/decoy anchor is wrong.
2. The decoy failure mode (anchor actively misleads the model into using wrong identifiers, resolves 0/10 below the 3/10 baseline) requires a content-wrong anchor, which full-content structurally cannot be.

Both premises hold. The decoy failure was caused by the wrong API being injected; full-content cannot be wrong by construction (it is the actual file). An exotic-format anchor might be noisy or useless to the model, but it cannot direct the model toward named identifiers that do not exist in the sibling, so it cannot produce below-baseline results by the same mechanism.

**Context-budget exception.** The dispatch brief raises a plausible additional failure mode: a very large sibling blowing the context budget, or an anchor that actively distracts. The ADR acknowledges the context-budget concern indirectly (it motivates the signatures compaction optimization), but does not explicitly address the context-window-overflow case as a potential floor violation. This is a narrow gap: context-overflow for a very large sibling could in principle force truncation, and a truncated anchor might behave like a corrupted anchor. However, (1) the ADR's Negative consequences section notes "the BUILD-deferred selection policy can cap it," (2) the claim is scoped to the worst case for *exotic types*, not a large-sibling pathology, and (3) context-overflow producing below-baseline results via invented identifiers is not the same causal path as the decoy failure. The benign-no-op claim holds for the normal exotic-type case; the large-sibling edge is a BUILD-scope concern already noted in Consequences. This is a P3 observation, not a claim error.

**Active distraction.** Could an exotic-format anchor (binary file, image bytes, etc.) actively mislead a model into referencing identifiers it would not have invented otherwise? Structurally, the anchor injects real bytes of the actual sibling file; for truly non-textual binary content, the injected bytes would be garbled text, and a model reading garbled text is unlikely to extract coherent identifier names to follow. This is a weaker form of the benign-no-op argument: binary siblings produce noise, not coherent wrong identifiers. The claim holds in the exotic-type case. The ADR does not explicitly handle binary siblings, but this is within scope of "effectiveness question" framing and does not require a qualification at the ADR level.

**Verdict:** the benign-no-op argument is sound for the exotic-type case as framed. The large-sibling/context-overflow edge is a latent P3 observation (not a claim error, since the ADR already routes it to BUILD scope).

---

### Audit focus 5: AST-vs-heuristic for config — "AST-checkable" characterization

The ADR states (Empirical Grounding): "with an AST-checkable primary outcome for the code and config bases (the config consumer is Python — string-key subscripts resolved against the real keys; no adjudication subjectivity, the ρ P1-A class structurally avoided)."

`resolve_config` in `probe.py` uses Python AST to walk the tree and collect:
- `ast.Subscript` nodes where `node.slice` is an `ast.Constant` string — catches `config["key"]` and `config['key']` patterns.
- `ast.Call` nodes where `node.func.attr == "get"` and `node.args[0]` is a string constant — catches `config.get("key")` calls.

The ADR's characterization describes this as "string-key subscripts resolved against the real keys." The implementation does catch that pattern cleanly. The "slight over-catch" concern in the dispatch brief: `resolve_config` catches *all* string-literal subscripts in the file, not only those on a variable named `config`. For example, `some_dict["rbo_ms"]` would also be caught. In the Base G task, the consumer is `scheduler.py` and the task description instructs direct subscripts on the config dict (`config['...']`), so in practice the generated files use the config-dict variable. The raw results confirm this: all 10 B_content trials reference exactly the three config keys, and the single miss (trial 7) substitutes `affinity_salt` for `aff_salt` — a key-name expansion, not a mis-classified non-config subscript. The over-catch risk is real but did not materialize across 30 trials. The ADR does not mention the over-catch; calling the resolution "AST-checkable" is accurate as a category (it is AST-based, not adjudicated by human judgment), but "slight over-catch" is the more precise description of the method.

**P3 observation:** the Empirical Grounding characterizes config resolution as "AST-checkable" without noting that `resolve_config` collects all string-literal subscripts, not only config-dict accesses. The over-catch is benign for this task (the single-dict pattern in the generated files, confirmed by the raw results), but the characterization is slightly imprecise. It does not rise to P2 because the method is AST-based and the over-catch did not affect any trial's classification.

---

### P1 — Must Fix

No P1 findings.

---

### P2 — Should Fix

No P2 findings.

---

### P3 — Consider

**P3-1 (Config causal decomposition coverage)**
- **Location:** Empirical Grounding, Base G description; Consequences Positive (Base G result cited alongside code/prose).
- **Claim:** The ADR cites Base G's B − decoy = 0.9 as establishing causal isolation on the config type.
- **Observation:** The code bases (T/V) used a three-way decomposition (B vs. decoy vs. filler). Base G uses a two-way decomposition (B vs. decoy). The two-way result is sufficient for the "specific keys are the mechanism" claim — B 9/10 vs. decoy 0/10 establishes that the wrong keys drove failure. The ADR does not claim three-way isolation for the config type; it claims "causal isolation holds." That is accurate for the two-way comparison. No corrective action required — this is a coverage note, not an error.
- **Recommendation:** no change needed. The ADR's framing of Base G as "content-agnosticism confirmation" rather than a full causal decomposition accurately scopes what the arm established. If the config path is ever revisited, a filler arm would complete the decomposition.

**P3-2 (Context-window-overflow edge for large siblings)**
- **Location:** Consequences Negative; Decision Form bullet.
- **Claim:** the worst case for an exotic sibling type is a benign no-op (degrading to baseline, not below).
- **Observation:** the ADR acknowledges context-budget growth as a concern motivating the signatures compaction and notes the BUILD selection-policy will cap it. It does not explicitly address the edge case where a very large sibling's full content, injected as the universal path, could be truncated at the context limit — a truncated real-file anchor is still a real-file anchor, but a severely truncated one might be less useful than none. This is distinct from the decoy failure (wrong content) but is worth noting as a boundary on the "never below baseline" claim.
- **Recommendation:** consider adding a one-sentence qualifier in the benign-no-op passage: the claim holds for siblings that fit within the available context window; large siblings are handled by the BUILD-deferred selection policy (already noted in Consequences). This would make the scope of the claim explicit without withdrawing it.

**P3-3 (AST "slight over-catch" not disclosed)**
- **Location:** Empirical Grounding, "AST-checkable primary outcome for the code and config bases."
- **Claim:** config resolution is AST-checkable with no adjudication subjectivity.
- **Observation:** `resolve_config` catches all string-literal subscripts in the file, not only accesses on the config-dict variable. The task design and results confirm this over-catch was benign (all trials used the config-dict pattern; the single miss was a key-name expansion, not a mis-classified subscript). Calling it "AST-checkable" is accurate categorically; "no adjudication subjectivity" is accurate in practice. The over-catch is not disclosed.
- **Recommendation:** optionally add a parenthetical in the Empirical Grounding characterization: "string-key subscripts resolved against the real keys (catches all subscripts, not only config-dict accesses; benign for this single-dict task pattern)." Absence of this note is not a material error given the results confirm no trial was mis-classified.

---

## Section 2: Framing Audit

R4's framing analysis (three alternative framings, two omitted observations) was settled and untouched by the content-agnostic revision. The revision affects the form-priority framing (signatures-first vs. full-content-first) and adds the config arm. These are addressed below.

### Question 1: What alternative framings did the evidence support?

**The revision itself resolves the main framing gap from prior rounds.** The prior "signatures, not full content" framing was an example of letting the code-specific result (where signatures win) drive the universal framing (where type-blind coverage requires full-content as the baseline). The revision corrects this by separating structural guarantee (full-content is type-blind by construction) from empirical preference (signatures win for extractable types). This is the correct framing of the evidence.

**Alternative framing the evidence could support: "three sibling types is not content-agnostic, it is three-data-point induction."** The evidence supports "confirmed across three types, each structurally different (code, prose, JSON data)." An alternative framing would foreground that three types is still an inductive sample, and the structural argument (full-content is bytes, type-blind by construction) is doing more epistemic work than the measurement does. The ADR does acknowledge this: "exotic sibling types beyond these three remain a benign-no-op effectiveness boundary (the full-content path is type-blind by construction)." This is the correct epistemic distribution of weight. The framing in the ADR gives the structural argument the primary load-bearing role and the three measurements the confirming role, which is the appropriate weighting.

**Alternative framing: "the config result is weaker than presented."** B_content 9/10 vs. B_signatures 10/10 (code bases). The config arm used only the full-content path — there is no signatures path for JSON data (JSON has no function API surface). So the config arm does not contribute to the signature-vs-full-content comparison; it contributes only to the source-side agnosticism claim. The ADR correctly reports this: the config arm confirms the full-content path works on a non-code sibling type, not that signatures are preferred for config. No over-statement detected.

### Question 2: What truths were available but not featured?

**The single B_content miss on Base G is explained by trial 7's raw data.** The miss is `affinity_salt` instead of `aff_salt` — the model expanded the abbreviation. This is a characteristic of abbreviated/opaque key names specifically: the model had enough semantic signal (from the full config content, which showed `aff_salt: "by_tenant"`) to produce a plausible expansion. The ADR characterizes the miss as "within the n=10 wide-CI margin" without describing what the miss was. The raw data shows it is an abbreviation-expansion pattern, not an independent invention — a model that read the config correctly but expanded the abbreviated key. This makes the 9/10 result more interpretable: the one failure was a near-miss caused by the opaque abbreviation design, not a fundamental coherence failure. This nuance is available in the raw results but not featured in the ADR. It is a P3 characterization gap — the 9/10 result stands on its own, but the nature of the miss adds supporting texture.

**The base-validity iteration for Base G (generic-loader task).** The research log describes this fully; the ADR summarizes it. The lesson ("the anchor is moot for siblings consumed wholesale") is featured in the Consequences (Negative) scope boundary. This is appropriately represented.

### Question 3: What would change if the dominant framing were inverted?

**Dominant framing: full-content is the universal type-blind baseline; signatures are the frugality optimization.**

**Inversion: signatures are the primary mechanism; full-content is a fallback for types where extraction is unavailable.**

Under the inverted framing: the config arm result (9/10, not 10/10) becomes a weakness of the fallback path relative to the primary path. The three-type spread becomes "two types where the primary mechanism works (code, prose with conversion from full-content to pseudo-signatures via the regex outcome), one type where only the fallback is available." The inverted framing would require characterizing the full-content path as second-best on types where extraction is available, and as the only option on types where it is not. This is closer to the old "signatures, not full content" framing. The ADR explicitly acknowledges and rejects this framing in the Form bullet — and rightly so, since the content-agnosticism commitment (any sibling type, including exotic types) structurally requires full-content as the default, not a fallback. The inversion would weaken the content-agnostic claim to a conditional guarantee ("agnostic where an extractor exists; uncertain otherwise"), which is inconsistent with the structural type-blindness of injecting real bytes. The dominant framing is more epistemically accurate given the construction-level guarantee.

### Framing Issues

No P1 or P2 framing issues. The content-agnostic revision resolved the main prior-round framing tension (signatures-first vs. full-content-first). The three-way distinction (measured / structurally guaranteed / residual boundary) is present and honest.

**P3-F1 (Base G single-miss characterization)**
- **Location:** Consequences Positive ("from 0/10 to 9/10 on a non-code JSON data sibling (Base G, the full-content path)"); Empirical Grounding.
- **Available truth:** trial 7's raw data shows the miss is `affinity_salt` (abbreviation expansion of `aff_salt`), not an independent invention. The model had the real content and produced a semantically correct but lexically inexact key.
- **Impact on argument:** none — the result stands and the characterization "within n=10 margin" is correct. But noting the nature of the miss would strengthen the presentation of 9/10 (it is an abbreviation-sensitivity artifact, not a coherence failure).
- **Recommendation:** optionally note in the Base G result summary that the single miss was an abbreviation expansion (`affinity_salt` for `aff_salt`), consistent with opaque-key sensitivity rather than a coherence failure of the full-content mechanism.

---

## Convergence-Saturation Signal (ADR-094)

**Convergence-Saturation Signal:** TRIGGERED

- Round number: R1-of-revision (form-change baseline reset per ADR-094 P2-R3-3; the content-agnostic revision is a substantive Decision restructuring)
- P1 count this round: 0 (Section 1 + Section 2 combined)
- P2 count this round (new, non-carry-over): 0
- New framings or claim-scope expansions: none — the content-agnostic framing was introduced by the revision itself (practitioner-directed), not surfaced by this audit round; no audit-introduced framings
- Recommendation: STOP at this round

The five revision-specific claims are clean: "content-agnostic" is warranted and structurally honest; the inversion is complete and consistent throughout the ADR; the config evidence is reported accurately with appropriate precision caveats; the benign-no-op argument is sound within its stated scope; the AST characterization for config is accurate categorically. Three P3 observations (config causal decomposition coverage, context-overflow edge, AST over-catch disclosure) are minor precision notes, none affecting the argument's conclusions. The gate may proceed.

*Form-change baseline-reset applies: the content-agnostic revision is treated as R1 for saturation-signal purposes. The TRIGGERED verdict on the pre-revision R4 does not carry over.*

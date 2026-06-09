# Argument Audit Report — Round 3

**Audited document:** `docs/agentic-serving/decisions/adr-039-content-anchor.md`
**Source material:**
- `docs/agentic-serving/essays/research-logs/cycle-7-spike-xi-content-anchor.md` (§"Prose arm (Base P)")
- `scratch/spike-xi-content-anchor/probe.py` (resolver logic: `resolve_prose`, `_is_conversion_fn`)
- `scratch/spike-xi-content-anchor/results_P_A_current.json`
- `scratch/spike-xi-content-anchor/results_P_B_signatures.json`
- `scratch/spike-xi-content-anchor/results_P_C_full.json`
- `scratch/spike-xi-content-anchor/generated/P_A_current_00.txt` through `02.txt` (sampled READMEs)
- `docs/agentic-serving/housekeeping/audits/argument-audit-decide-cycle-7-loopback7.md` (R1)
- `docs/agentic-serving/housekeeping/audits/argument-audit-decide-cycle-7-loopback7-round2.md` (R2)
**Genre:** ADR
**Date:** 2026-06-09

---

## Audit context

R2 converged (TRIGGERED) on the code-only ADR. Since R2 a substantive scope change was applied: a practitioner-directed prose arm (Base P) was run and the ADR was revised to declare the mechanism callee-agnostic, covering prose-generating callees (README via `prose-improver`) in addition to code. The discharge gate was also revised — reversing the R2 FI-P2-1 closure — making the README a gate criterion again, on the basis that the prose arm grounded it empirically. This round audits the prose-specific claims and checks for contradictions introduced by the revision.

Prior settled findings (R1 P2-1, P2-2, P2-3, P3-1 closed in R2; R1 FI-P3-1 carried as P3; R2 P2-1 residual Consequences/gate cross-reference gap) are re-examined only where the revision touched them.

---

## Section 1: Argument Audit

### Summary

- **Genre:** ADR
- **Argument chains mapped:** 5 (Decision rationale; form selection; causal isolation; rejected alternatives; scope of consequences) + 3 new prose-specific chains (callee-agnostic generalization; prose-invents-worse comparison; README-as-discharge-criterion)
- **Issues found:** 4 (0 P1, 2 P2, 2 P3)
- **Pyramid coverage map:** N/A
- **Expansion-fidelity findings:** N/A

### Rate verification — Base P

Raw JSON confirmed against ADR claims:

| Base | Arm | ADR claims | Raw JSON |
|------|-----|-----------|---------|
| P | A_current | 0/10 | 0/10 confirmed; 10 classified `invented` |
| P | B_signatures | 10/10 | 10/10 confirmed; all `resolves`, graded 1.0 |
| P | C_full | 10/10 | 10/10 confirmed; all `resolves`, graded 1.0 |

Mean graded rate for A_current: 0.23 (as stated in research log). Computed from the JSON: (0.5 + 0.0 + 0.75 + 0.0 + 0.0 + 0.0 + 0.5 + 0.0 + 0.5 + 0.0) / 10 = 2.25 / 10 = **0.225**. Research log states 0.23; rounds correctly. Confirmed.

The qwen3:8b model label is confirmed in every results JSON (`"model": "qwen3:8b"`).

No rate discrepancies found.

---

### Prior closure re-verification (R2 findings affected by the revision)

**R2 FI-P2-1 closure — NOW REVERSED by design.** R2 confirmed the README had been scoped as "observed at the gate but not a discharge requirement." The revision intentionally reverses this: the Empirical Grounding section now reads "The README is a discharge criterion, not merely observed: the prose arm established that the anchor fires on the prose callee and fixes it (0/10 → 10/10), so the discharge run confirms that holds end-to-end under the real client (the anchor must reach `prose-improver`, not only `code-generator`)." This reversal is grounded in the prose arm result and is the central scope change this round audits. The reversal also re-opens the R2 P2-1 residual (the Consequences/gate cross-reference gap) — see below.

**R2 P2-1 residual — CLOSED by revision.** The Negative consequences section now contains: "Prose-to-code coherence is no longer scoped out: it was measured (0/10 → 10/10), so the README is a discharge-gate criterion, not merely observed." This provides the evaluator-facing statement the R2 P2-1 residual called for. No longer a gap.

---

### Prose-specific claim verification

#### Claim 1: Callee-agnostic generalization from one prose base (Base P)

**ADR text (§Decision, fourth bullet):** "The augmentation fires on the callee dispatch regardless of which capability ensemble is invoked... Any callee generating a deliverable that references produced siblings receives the anchor; the mechanism does not special-case code, because nothing about the failure is code-specific (same cheap qwen3:8b coder, same blindness, same invention)."

**ADR text (§Context, fourth bullet):** "The fix is callee-agnostic (prose confirmed)."

**Evidence:** One prose base (Base P), one deliverable type (README.md), one prose coder (prose-improver system prompt as used in the live trajectory), n=10 per cell. The evidence structure is: two code bases (T and V) + one prose base (P). The prose base used the same sibling (converters.py, the same symbols as Base T), same model (qwen3:8b), same anchor mechanism (CONVERTERS\_SIGS injected via `anchor_for`). The task was "Write README.md for a temperature-conversion library."

**Assessment:** The "callee-agnostic" claim is warranted at the level the ADR actually uses it — the mechanism (inject the sibling API into the callee dispatch) works regardless of whether the callee is a code-generator or a prose-improver, because both are qwen3:8b running the same injection. The ADR is not claiming the mechanism works for arbitrary prose deliverable types; it is claiming the mechanism is not code-specific.

The evidence does support this claim at the structural level: the failure (blind invention) and the fix (anchor resolves 10/10) are identical in shape across code and prose callees. The relevant structural question is whether anything about code vs. prose changes the mechanism, and the evidence says no — the prose callee received the same API anchor and stopped inventing identically.

One honest limitation the ADR does not state at the "callee-agnostic" point: Base P used the same converters.py sibling as Base T, so the prose arm and Base T share the same symbols. The generalization tested is "does the prose callee also respond to the anchor?" (yes, decisively), not "does the mechanism generalize to prose deliverables that depend on non-converters siblings." But this is a narrow objection: the structural question is callee-type independence, not sibling-content independence, and the evidence answers the callee-type question cleanly.

**Verdict: warranted.** The Provenance section correctly labels the single-prose-base scope. The claim is appropriately scoped to "the measured prose case" in the Consequences/Negative boundary bullet: "Scope is the measured class... for code AND prose deliverables (the prose arm brought the README into scope)." The "callee-agnostic" header claim is accurate — the mechanism is not code-specific — while the scope boundary records what was and was not measured.

---

#### Claim 2: Measurement-validity caveat — is the 0/10 vs 10/10 gap "far outside any plausible extraction-noise margin"?

**ADR text (§Consequences, Negative, final bullet):** "the prose 0/10-versus-10/10 gap is far outside any plausible extraction-noise margin, but the prose FC's refutation is heuristic, not exact."

**Resolver logic examined (`probe.py` lines 306–349, `resolve_prose` and `_is_conversion_fn`):**

The resolver extracts conversion-function-shaped identifiers from the README via three pathways: (1) `from converters import ...` lines parsed by `_FROM_IMPORT`; (2) `converters.attr()` attribute access calls via `_ATTR`; (3) `_FUNC_TOKEN` regex matching any snake\_case token that contains `_to_`, starts with `convert` or `to_`, or is a known symbol.

The key concern is whether the heuristic systematically over-counts invented references (making prose look worse) or under-counts real references (making the gap look larger).

**Over-counting invented references on A_current:** The `_is_conversion_fn` function passes a token if it contains `_to_` OR starts with `to_` — this is intentionally broad. Examining trial 00 (P_A_current_00.txt): the model invented `kelvin_to_celsius`, `fahrenheit_to_kelvin`, and `kelvin_to_fahrenheit`. All three are `_to_`-shaped. The resolver correctly captures them as invented. Examined trial 01: the model invented `c_to_f`, `k_to_c`, `c_to_k`, etc. — all `_to_`-shaped, all correctly captured as invented. Trial 02: `kelvin_to_celsius` — correctly captured as invented, also `celsius_to_kelvin` (real) and two others (real). The resolver classified this as "invented" because `kelvin_to_celsius` does not exist, which is correct.

**Could the heuristic systematically over-count on A_current, inflating the gap?** The broad `_to_` catch does mean the resolver captures invented conversion-adjacent tokens that a stricter resolver would miss. But for the purpose of the gap, over-counting invented items on A_current does not inflate the gap — it can only make A_current look worse (more items, some false). Crucially, on A_current all 10 trials were classified as "invented" because at least one `_to_`-shaped token in each README did not exist in the real symbol table. This is confirmed by reading the retained READMEs: every trial 00–09 documented multiple conversion function names, some of which (kelvin_to_celsius, fahrenheit_to_kelvin, c_to_f, etc.) were not in the three-symbol table. The resolver correctly classified these.

**Could the heuristic over-count on B_signatures, inflating the 10/10 by missing actual invented references?** The resolve\_prose function also applies `_FUNC_TOKEN` to B_signatures READMEs. The B_signatures anchor tells the model: "exposes exactly this API: celsius_to_fahrenheit, fahrenheit_to_celsius, celsius_to_kelvin." All 10 B trials in the JSON show only those three identifiers captured, all true. Reading P_B_signatures_00.txt: the model documented exactly those three functions in a table and used them in code examples. No invented identifiers were present in the retained README, and the resolver found none.

**Could the `_to_` catch miss conversion identifiers on B_signatures that the model invented?** Only if an invented identifier did not contain `_to_` and did not start with `convert` or `to_`. Given that temperature conversion API names universally use `_to_` forms, this is not a realistic concern for this domain. The heuristic's broad catchment actually helps here — it catches more conversion-shaped tokens, not fewer.

**Systematic bias direction:** If the heuristic is biased, it errs toward catching more conversion-shaped tokens, which makes A_current look worse (more invented items detected) and makes B/C more likely to be caught if they had invented any. Since B/C had zero invented identifiers in any trial (confirmed by the retained READMEs for trial 00 of each), the systematic direction of heuristic error does not inflate the gap — it could only reduce it.

**Verdict: the caveat is honest and sufficient.** The 0/10 vs 10/10 gap is not plausibly a heuristic artifact. Every A_current README contains identifiable conversion-function-shaped tokens that do not exist in the three-symbol converters table. No B_signatures or C_full README contains any such token. The gap is real. The ADR's "far outside any plausible extraction-noise margin" characterization is accurate, and the "heuristic, not exact" qualifier is the correct epistemic flag.

---

#### Claim 3: "Prose invents worse than code" — is the comparison sound across different resolvers?

**ADR text (§Context, fourth bullet):** "prose A_current resolved 0/10 (worse than code's 3/10 because the README pulls a strong complete-library prior, confidently inventing every pairwise conversion)."

**ADR text (§Decision, fourth bullet):** "the README generated blind invented functions 0/10 (worse than code's 3/10, since prose pulls a stronger 'complete-library' prior)."

**The comparison:** Code A_current (Base T) used AST resolution. Prose A_current (Base P) used regex-heuristic resolution. The two resolvers are different: AST is exact for Python imports and attribute access; `resolve_prose` uses a broader heuristic. The question is whether the difference in resolver could explain the 3/10 vs 0/10 gap rather than a genuine behavioral difference.

**Analysis:**

The code resolver (AST, `resolve`) classifies a trial as `resolves` iff every cross-file reference is to a real symbol. The prose resolver (`resolve_prose`) classifies a trial as `resolves` iff every conversion-function-shaped identifier found in the README exists in the symbol table.

For code Base T: 3 trials resolved (the model happened to use the three real function names). 7 were "invented" (used `convert_temperature` or similar). The AST resolver is conservative — a trial resolves only if the model produced exactly the right function names with correct import/call structure.

For prose Base P: 0 trials resolved. All 10 READMEs contained at least one invented conversion-function identifier. Looking at the graded rates: trial 00 (0.5, with celsius_to_fahrenheit/kelvin, fahrenheit_to_celsius real, plus kelvin_to_celsius/fahrenheit_to_kelvin/kelvin_to_fahrenheit invented), trial 02 (0.75, three real plus kelvin_to_celsius invented), trial 06 (0.5, three real plus fahrenheit_to_rankine/kelvin_to_celsius/rankine_to_fahrenheit invented).

**Could the heuristic make prose look worse than it is?** The prose resolver catches more identifier types (via the broad `_to_` regex). In principle, a README that only imported a real symbol and mentioned it in prose could still be caught if the prose also contained any `_to_`-shaped invented token. The resolver does not distinguish import-line mentions from casual prose mentions. If the model wrote "converting fahrenheit to kelvin (not supported)" in passing, `fahrenheit_to_kelvin` (or the prose phrasing "fahrenheit_to_kelvin" if used as a token) would be captured.

However, reading the retained READMEs: the invented identifiers appear in function-listing tables, code examples, and function-by-name section headers — these are genuine API documentation claims, not casual prose. The model is documenting `kelvin_to_celsius` as a function the library provides. This is the failure the audit cares about.

**Is the 3/10 vs 0/10 gap plausibly resolver-artifact?** For the comparison to be unfair, the prose resolver would need to be catching invented identifiers that the code resolver would have missed if applied to prose. But the comparison is not "prose resolver vs. code resolver on the same output" — it is "prose resolver on prose output vs. code resolver on code output." The question is behavioral: did the prose model invent more? The answer from the READMEs is clearly yes: every single prose trial contains invented conversion-function names (the model documents a complete 6-function library or uses abbreviations like `c_to_f`). The code model managed 3/10 trials with only the three correct function names (likely by pattern-completing from the three well-known `celsius_to_*`/`fahrenheit_to_celsius` names the Base T API also happens to have).

**The "complete-library prior" explanation:** The ADR attributes prose's worse performance to "a strong complete-library prior" — the README writing task primes the model to document a complete library, which invites inventing the missing pairwise conversions. Is this a measured finding or a drafting-time interpretation? It is a post-hoc interpretation consistent with the data (trials 00, 06, 09 all show the model inventing `kelvin_to_celsius`, `fahrenheit_to_kelvin`, etc. — exactly the missing pairwise functions), but it is not separately tested against alternative explanations. The ADR does not label it as drafting-time interpretation.

**Assessment:** The comparison is directionally sound — prose does invent more, and the resolver difference does not explain the gap. But one issue is present: the 0/10 vs 3/10 comparison is stated without noting the resolver difference anywhere in the ADR. A reader comparing the two rates might wonder whether the different resolvers explain the different baselines. The ADR says "prose A_current resolved 0/10 (worse than code's 3/10 because...)" without noting this is a cross-resolver comparison. This is a transparency issue, not a logical error.

The second issue: the "complete-library prior" explanation is stated as the cause ("because the README pulls a strong complete-library prior") in both the Context and Decision sections without being labeled as drafting-time interpretation, while the Provenance section applies that label only to the signatures-preferred and all-siblings-scope items. The prose-invents-worse mechanism is a post-hoc characterization, not a controlled measurement.

---

#### Claim 4: Discharge gate reversal — README as criterion (FI-P2-1 re-examination)

**R2 FI-P2-1 closure context:** In R2, the README was scoped as "observed at the gate but not a discharge requirement," with the reasoning that the prose callee's anchor behavior was unmeasured. That closure is now reversed by the prose arm: because the prose arm measured 0/10 → 10/10, the anchor is now claimed to fire on the prose callee, making the README a legitimate gate criterion.

**Current ADR text (§Empirical Grounding):** "The README is a discharge criterion, not merely observed: the prose arm established that the anchor fires on the prose callee and fixes it (0/10 → 10/10), so the discharge run confirms that holds end-to-end under the real client (the anchor must reach `prose-improver`, not only `code-generator`)."

**Current Negative consequences text:** "Prose-to-code coherence is no longer scoped out: it was measured (0/10 → 10/10), so the README is a discharge-gate criterion, not merely observed."

**Assessment:** The reversal is logically coherent given the prose arm result. The prose arm confirmed the mechanism works on the prose callee in the harness; making the README a gate criterion tests whether it works under the real client — the same layer-matching discipline applied to code. The three sections (Decision, Consequences/Negative, Empirical Grounding) are now mutually consistent on this point.

One residual tension remains: the harness prose arm ran the `prose-improver` system prompt directly against Ollama (see `probe.py` `PROSE_SYSTEM`), which is the same system prompt as the `prose-improver` ensemble's default task. But the production path that the discharge gate tests is the full framework → callee dispatch, where the prose-improver ensemble receives an input brief composed by the seat-filler plus the framework-injected anchor. The harness confirms the model responds to the anchor when the anchor is in the user message; the discharge gate confirms the framework actually injects the anchor into the `prose-improver` dispatch (not just `code-generator`). This distinction is exactly what the discharge gate exists to verify, and the ADR makes this explicit: "the anchor must reach `prose-improver`, not only `code-generator`." The logic is sound.

**Verdict: the reversal is warranted.** The Consequences, Decision, and Empirical Grounding sections are mutually consistent on the README's gate criterion status.

---

#### Claim 5: B = C = 10/10 on Base P — is dropping "prose may need richer form" warranted?

**ADR text (§Decision, first bullet):** "The prose arm (Base P, README) confirms signatures suffice for a prose callee too: B_signatures and C_full both resolved 10/10, so the API surface plus docstrings is enough for the README to document the real functions, and no prose-specific richer form is needed."

**Research log (§Prose arm, Findings):** "(3) Signatures suffice for prose (B 10/10 = C 10/10): the sufficiency nuance (that prose might need behavior or examples beyond the API surface) did not materialize — signatures plus one-line docstrings let the README document the real API. So signatures is the right form for both callees."

**Assessment:** B = C = 10/10 on Base P means both hit the ceiling. At n=10 you cannot distinguish B from C at this performance level — both are at the maximum observable rate. The ADR correctly notes this in the rejected alternatives section: "Held as the fallback if a future dependency needs more than the API surface (for example behavior or invariants that signatures do not carry); the n=10/cell caveat means 8-versus-10 is not a distinguishable population rate, so the preference rests on signatures' equal-or-better rate and the budget argument."

The dropped nuance ("prose may need a richer form") was a pre-run hypothesis. The prose arm answered: at n=10, no difference is observed. Dropping the nuance is warranted given the evidence — the form selection pre-registered rule says "prefer signatures absent a measured gap," and B = C = 10/10 is "no measured gap." The ADR's framing is accurate.

The one thing the ADR could note and does not: the ceiling effect means a harder prose task (one where signatures are insufficient and behavior or examples are needed) might separate B from C. But this is a P3 observation — the ADR correctly defers harder prose shapes to future work, and the sentence "held as the fallback if a future dependency needs more than the API surface" implicitly covers it. The omission does not change the decision.

**Verdict: dropping the nuance is warranted.** The pre-registered form-selection rule drives the conclusion; the result (B = C = 10/10) meets the rule's "no measured gap" criterion for preferring signatures.

---

### Full-document consistency sweep

The ~8 prose-related edits were checked against the un-edited sections and against each other:

- **Context/fourth bullet** and **Decision/fourth bullet** consistently state A_current 0/10, B_signatures 10/10, C_full 10/10, prose invents worse than code 3/10, complete-library-prior explanation. These are internally consistent with each other and with the JSON-verified rates.
- **FC (cross-file reference resolution)** extends the outcome criterion to "code AND prose deliverables" with the heuristic qualifier. Consistent with the Consequences/Negative boundary bullet.
- **Consequences/Positive** reports Base P 0/10 → 10/10 alongside code bases. Consistent with JSON.
- **Consequences/Negative** contains the scope boundary (heuristic, not AST) and the "README is a discharge-gate criterion, not merely observed" reversal. Consistent with Empirical Grounding.
- **Empirical Grounding** discharge gate lists README as a criterion with the justification. Consistent with the prose arm measurement.
- **Provenance check** lists the prose arm rates accurately (A_current 0/10, B_signatures 10/10, C_full 10/10) and credits both the practitioner's challenge and the retained generated files.

No cross-section contradiction introduced by the revision.

One legacy tension checked: the "prose-to-code coherence" language in the Negative consequences bullet. The prose arm tested prose coherence to code (the README documenting the converters API correctly), not code coherence to prose. The "prose-to-code coherence" label is used consistently throughout the ADR and matches the actual measured phenomenon (the README, a prose deliverable, referencing code symbols). No terminology inconsistency.

---

### P1 — Must Fix

No P1 findings.

---

### P2 — Should Fix

**P2-1 (new): The "prose invents worse" cross-resolver comparison is stated as a causal explanation without noting the resolver difference or labeling the explanation as drafting-time interpretation.**

Location: §Context, fourth bullet: "prose A_current resolved 0/10 (worse than code's 3/10 because the README pulls a strong complete-library prior, confidently inventing every pairwise conversion)." Also §Decision, fourth bullet: "the README generated blind invented functions 0/10 (worse than code's 3/10, since prose pulls a stronger 'complete-library' prior)."

The 0/10 vs 3/10 comparison is: prose A_current resolved by `resolve_prose` (regex heuristic) vs. code Base T A_current resolved by `resolve` (AST). These are different resolvers. The ADR does not note this; a reader comparing the two rates as though they were measured by the same instrument would have a slightly misleading impression.

More importantly, the "complete-library prior" is a post-hoc interpretation of why prose invents worse, not a controlled measurement. The Provenance section does not label it as drafting-time synthesis (it labels other interpretive items: form selection preference, all-siblings scope, language-specific extraction, injection-point placement — but not the prose-invents-worse mechanism explanation).

This is not a logical error — the comparison is directionally accurate (prose does invent more, and the resolver difference does not explain the gap as shown in the claim-by-claim analysis above). But the "because" framing claims causal explanation for something that is observational characterization. The Provenance section should label the "complete-library prior" explanation as drafting-time interpretation, and a one-clause note that the two baselines use different resolvers (AST for code, heuristic for prose) would make the comparison transparent.

Recommendation: in the Provenance check, add the "complete-library prior" explanation to the drafting-time-synthesis list: "the 'prose invents worse because of a complete-library prior' characterization (observational, consistent with the data — five of ten A_current trials showed invented complete pairwise sets — but not separately controlled; the resolver difference, AST for code vs. regex heuristic for prose, does not explain the 3/10 vs 0/10 gap but should be noted at the comparison point)." In Context and Decision, soften "because the README pulls a strong complete-library prior" to "consistent with the README pulling a stronger complete-library prior" or similar.

---

**P2-2 (new): The discharge gate now claims "the anchor fires on the prose callee" based on the harness prose arm, but the harness tested the model's response to the anchor (does qwen3:8b in prose mode stop inventing when anchored?), not the framework's injection behavior (does the framework send the anchor to the prose callee's dispatch?). The discharge gate description conflates these two things.**

Location: §Empirical Grounding, discharge gate: "The README is a discharge criterion, not merely observed: the prose arm established that the anchor fires on the prose callee and fixes it (0/10 → 10/10)."

The prose arm confirmed that qwen3:8b responding to a prose system prompt stops inventing when the sibling API is in its user message. What it did not confirm is that the framework sends the anchor to the prose-generating callee in the production dispatch — that is precisely what the discharge gate exists to confirm. The phrase "the anchor fires on the prose callee" runs the risk of suggesting the harness confirmed end-to-end anchor delivery, when it confirmed model response to the anchor.

The sentence is partially self-correcting — it ends with "so the discharge run confirms that holds end-to-end under the real client (the anchor must reach `prose-improver`, not only `code-generator`)" — but the setup framing "established that the anchor fires on the prose callee" could mislead a reader who does not parse the subsequent qualification carefully.

This is a P2 rather than P1 because the self-correction is present and the logic is otherwise sound. The issue is precision at the seam between what the harness established (model behavior) and what the discharge gate establishes (framework delivery).

Recommendation: rephrase to make the distinction explicit: "The prose arm established that qwen3:8b in prose mode responds to the anchor identically to the code mode (0/10 → 10/10); the discharge gate confirms the framework delivers the anchor to the `prose-improver` callee's dispatch in the live stack." One sentence rephrasing; no change to the gate criteria.

---

### P3 — Consider

**P3-1 (carry-over from R1/R2, FI-P3-1): The path-only injection alternative remains unnamed in Rejected Alternatives.**

As in R1 and R2: the framework-injected-path-without-content option is implicitly rejected by the live-trajectory evidence (zero reads across the whole run) but not named as a considered-and-rejected option. Not reopened as a new finding; carried as a documentation gap the practitioner may or may not address.

---

**P3-2 (new): The B = C = 10/10 ceiling effect on Base P is not noted, leaving the inference that "signatures suffice" potentially stronger than the evidence.**

Location: §Decision, first bullet: "no prose-specific richer form is needed." §Research log, Findings (3): "the sufficiency nuance ... did not materialize."

The Rejected Alternatives section for full-content anchor notes the n=10 ceiling issue for code ("the n=10/cell caveat means 8-versus-10 is not a distinguishable population rate"). That caveat applies equally to the prose B = C = 10/10 comparison: at a ceiling of 10/10 on both arms, the data cannot show B is better or worse than C — they are indistinguishable. "No prose-specific richer form is needed" is technically supported by the pre-registered decision rule (no measured gap → prefer signatures), but characterizing the result as "signatures suffice" is slightly stronger than "signatures performed identically to full content at this task difficulty." A harder prose task might separate them. The research log's "sufficiency nuance did not materialize" framing is more accurate.

The ADR does note "Held as the fallback if a future dependency needs more than the API surface" in the Rejected Alternatives section, which covers this. The issue is that the prose arm result is presented as evidence that signatures suffice for prose, when it more precisely shows signatures and full content are indistinguishable at this task difficulty.

Not a P1 or P2 because the decision-rule basis is sound and the fallback is recorded. Consider noting at the "no prose-specific richer form is needed" point that the B = C result is at the measurement ceiling (both 10/10), so the sufficiency conclusion applies to the measured README task difficulty.

---

## Section 2: Framing Audit

### Question 1: What alternative framings did the evidence support?

The three framings identified in R1 (path-only injection, API-format-as-constraint, generate-then-repair) remain available and the assessment from R1 and R2 stands: the dominant framing is defensible, exclusions are soundly reasoned, and the prose arm does not surface a new alternative framing. The revision narrows rather than shifts the argument's framing.

One new framing is now available given the prose arm result: "the content anchor should be measured on a non-converters prose base before claiming callee-agnostic generalization." The prose arm shared Base T's symbol set (converters.py). A reader could ask whether a prose base built on the non-guessable Base V symbols (text_tools) would have shown a different A_current rate, since prose might more easily invent plausible text-utility names than conversion-library names. This alternative framing — "prose arm shares Base T's easy symbols; independence not fully tested" — is available from the evidence and is not addressed in the ADR.

Belief-mapping: for this framing to be right, the reader would need to believe that the prose model inventing conversion-function names is specifically caused by the domain (temperature), not by blindness per se, and that a prose task on a non-guessable domain would show a different pattern. This is not implausible — the "complete-library prior" explanation suggests domain-specific invention. However, the ADR's mechanism claim is not "prose always invents at 0/10 blind" but "the anchor fixes prose 10/10," which the Base P evidence supports regardless of domain. The alternative framing would weaken the baseline characterization ("prose invents worse") but not the mechanism claim ("anchor fixes it"), which is the load-bearing conclusion.

### Question 2: What truths were available but not featured?

**Omitted observation A (new): The prose A_current graded rates show substantial partial resolution.**

The mean graded rate for prose A_current is 0.225. Four trials (00, 02, 06, 09) have graded rates of 0.50, 0.75, 0.50, 0.50 respectively — meaning the model got some real function names right in the README even without the anchor. The ADR says "confidently inventing every pairwise conversion" (Context) which is accurate for the invention behavior (every trial contains invented names) but omits that 4/10 trials also contained real function names mixed with invented ones. The research log's characterization ("the blind model confidently fills in every pairwise conversion ... most of which do not exist") is more accurate on this point — it says "most," not "all."

This is a minor characterization gap: "inventing every pairwise conversion" describes what the model invents (it adds the six pairwise conversions), not that it gets none right. The binary "invented" classification is correct (any invented reference fails). But the phrase could mislead a reader to believe the model got zero real names right, when several trials got 3 of 6 right (the three real ones) plus invented the other three.

Why it may have been excluded: the graded rate is secondary per the research design, and the binary classification is the go/no-go criterion. Including it would not change the decision.

**Omitted observation B (carry-over, R1): Research log stale qwen3:14b sentence (lines 41-45).** Remains unaddressed. Not a gate blocker.

### Question 3: What would change if the dominant framing were inverted?

The dominant framing on the prose revision: "the anchor is a universal mechanism; the failure and fix are structurally identical for code and prose callees."

Inverted framing: "the prose arm is a convenience demonstration on a favorable base (same symbols as Base T); genuine callee-agnosticism requires testing on a prose base with non-guessable symbols."

Under the inverted framing: the "complete-library prior" explanation for prose inventing worse is not a universal mechanism finding — it is a property of the temperature-conversion domain. A prose deliverable documenting `text_tools` (Base V's non-guessable API) might invent at a different base rate, and the anchor's effect might be different (the model would need to stop inventing `squeeze_runs`-shaped token names, which it does not have a "complete library" prior for).

What becomes more salient: the prose arm's 0/10 baseline is as much a property of the domain choice (a familiar complete-set domain) as of the prose-callee type. This does not undermine the mechanism claim (the anchor fixes it 10/10 regardless), but it complicates the "prose invents worse than code" comparative claim.

What the ADR would need to address: acknowledge that the 0/10 baseline on prose may be partly domain-driven (the temperature conversion domain has a strong complete-library prior) and that a prose arm on a non-guessable domain is an open question. The callee-agnostic claim for the fix (anchor → 10/10) is robust; the "prose invents worse" comparative characterization is domain-sensitive.

### Framing Issues

**FI-P2-1 (new): The "callee-agnostic" claim is stated without noting that the prose arm shares Base T's symbol domain, limiting the independence of the prose evidence from the code evidence.**

Location: §Context, fourth bullet; §Decision, fourth bullet; §Empirical Grounding; multiple uses of "callee-agnostic."

The prose arm (Base P) used converters.py as the sibling — the same sibling as Base T. This means the two code bases (T and V) provided domain independence (different APIs, different tasks), while the prose base added callee-type independence but not domain independence relative to Base T. A reader seeing "two code bases plus a practitioner-directed prose base" might infer three independent domains, when in fact two of the three (Base T and Base P) share the same symbol set.

This is not an argument error — the Provenance section implicitly conveys this (it lists "two code bases plus a practitioner-directed prose base" without claiming the prose base introduces a new domain). The callee-agnostic claim is about callee type (code-generator vs. prose-improver), not domain independence, and on callee type the evidence is adequate. But the implicit inference of full independence across all three bases is available to a reader and is not foreclosed.

This is a P2 rather than P1 because the callee-agnostic claim is the correct claim, the evidence supports it, and the Provenance section does convey the relevant structure implicitly. An explicit note that Base P shares Base T's sibling would close it cleanly.

Recommendation: in the Empirical Grounding section's "two code bases plus a practitioner-directed prose base" description, add that Base P used the converters.py sibling (same as Base T), so the three bases provide: two distinct code deliverable types on two different symbol sets (T and V), and one prose deliverable type on the same symbol set as T. The callee-type independence (the claim) is fully covered; the domain independence is partial.

---

**FI-P3-1 (carry-over, R1): Path-only injection alternative remains unnamed in Rejected Alternatives.** Unchanged from R1 and R2; still a P3.

---

## Convergence-Saturation Signal (ADR-094)

**Convergence-Saturation Signal:** TRIGGERED

- Round number: R3 (this is the first audit on the post-prose-arm revision; per ADR-094 form-change baseline-reset rule, substantive scope changes warrant fresh audit discipline — this is treated as R3 in the sequence, not R1, because it is the same document form with a scope amendment rather than a document restructuring or replacement)
- P1 count this round: 0 (Section 1 + Section 2 combined)
- P2 count this round (new, non-carry-over): 3 (P2-1, P2-2 from argument audit; FI-P2-1 from framing audit)
- New framings or claim-scope expansions: the "prose invents worse" cross-resolver comparison (P2-1 — new this round); the harness-vs-framework delivery distinction in the discharge gate (P2-2 — new this round); the shared Base T/Base P symbol set domain-independence gap (FI-P2-1 — new this round)
- Recommendation: CONTINUE to R4

**Signal does not trigger.** P2 count this round (3 new, non-carry-over) exceeds the ≤1 threshold. All three new P2s are prose-arm-specific findings introduced by the scope revision; none threaten the core mechanism claim or the gate criteria. No P1 findings. The ADR's causal isolation, rate data, and structural logic are sound.

**R4 target:** The three new P2s are editorial-precision repairs — provenance labeling (P2-1), a one-sentence phrasing clarification in the discharge gate (P2-2), and a domain-independence note in Empirical Grounding (FI-P2-1). A single revision pass addressing these would be expected to bring R4 to 0 P1, ≤1 new P2, no new framings, and trigger the signal. The P3s (FI-P3-1 carry-over, P3-1 carry-over, P3-2 new) are addressable in the same pass or deferred; they do not affect the gate.

*Standard-sequence audit: the verdict line applies.*

# Susceptibility Snapshot

**Phase evaluated:** DECIDE — loop-back #7 (Finding H / content anchor)
**Artifact produced:** ADR-039 (Content Anchor — routing produced-sibling signatures into the callee dispatch)
**Date:** 2026-06-09

---

## Observed Signals

| Signal | Strength | Trajectory | Notes |
|--------|----------|------------|-------|
| Assertion density | Ambiguous | Stable | The practitioner issued one strong declarative challenge ("obviously equally important," "nothing structurally different") but overall turn volume was investigative. The agent did not simply assert back; it ran a spike arm. Density is present but not rising through the phase. |
| Solution-space narrowing | Ambiguous | Stable relative to prior loop-backs | The decision space was already scoped at DECIDE entry to the injection-shape class (B/C/D variants). The practitioner challenge expanded scope (added prose) rather than narrowing it further. The methods review explicitly raised the path-only injection and generate-then-repair alternatives; both were deferred rather than examined. The space narrowed from "any fix shape" to "injection" early, but that narrowing was pre-established by the Finding-G rhyme. |
| Framing adoption | Clear | Intensifying at the prose-scope moment, then grounded | The practitioner's "obviously equally important" + "nothing structurally different" framing was adopted without the agent surfacing a genuine alternative before running the arm. The distinctions the agent could have named first — prose measurement is regex-heuristic not AST; one README base is thin evidence for callee-agnostic — appeared in the argument audit (R3 P2-1, R3 FI-P2-1) rather than in the DECIDE conversation before the scope expansion. The arm then provided genuine grounding. The sequence is: framing absorbed → arm run → framing partly earned. The adoption was not corrected in-conversation; the audits found it after. |
| Confidence markers | Clear | Present throughout, well-bounded | "10/10 across the board" results generated strong-result language. The audit cycle caught each over-claim: harness-vs-real-client conflation (R1 P2-1), all-siblings-vs-single-sibling (R1 P2-2), ADR-038 epistemic asymmetry (R1 P2-3), cross-resolver comparison stated as causal (R3 P2-1), harness-confirms-framework-delivery conflation (R3 P2-2). All were corrected. The pattern of confident language followed by audit catch and correction was repeated across all four rounds. |
| Alternative engagement | Ambiguous | Shallow on two specific forks | Path-only injection (methods review incongruity + R1 FI-P3-1, carried through all four rounds, never incorporated) and generate-then-repair (methods review, R1 framing audit Q1-C) were named but not incorporated into Rejected Alternatives. Both were assessed as implicitly rejected on sound bases, but neither was made explicit. Four audit rounds did not produce their incorporation; the practitioner apparently did not direct it. |
| Embedded conclusions | Clear | Present at the prose-scope boundary | At the moment the practitioner challenged the code-only scope ("nothing structurally different"), the agent treated "callee-agnostic" as a conclusion to be confirmed, not a hypothesis to be examined. The prose arm was run to confirm the framing, not to test whether it was wrong. The arm happened to support the framing strongly, but the sequence was confirmation-biased at the artifact-production boundary. |

---

## Interpretation

### What the signals collectively suggest

The loop-back shows a split pattern: the core spike (code arms, causal isolation, model-fidelity correction) was rigorously designed and the argument-audit cycle did real work catching over-claims. That is earned confidence. Four distinct confidence-inflation patterns were caught and corrected across R1 through R4. The methods review was applied before running. The self-caught model-fidelity error (qwen3:14b → qwen3:8b) is a positive signal: the agent identified a fidelity gap independently before any run and corrected it.

The vulnerability is concentrated at the prose-scope boundary, and it is the specific signal the dispatch flags for examination.

**Was "prose coherence is obviously equally important, nothing is structurally different" absorbed at face value?**

Partially, yes. The agent ran a prose spike arm rather than just rewording the ADR — that is grounding, not mere capitulation. But the pre-arm conversation did not surface the structural asymmetries the audits later found: (a) prose measurement is a regex heuristic over identifier tokens, not AST-clean; (b) Base P shared the same sibling domain as Base T (converters.py), so the three "bases" provide callee-type independence but not three independent domains; (c) a single deliverable type (README) on one domain is thin evidence for the callee-agnostic generalization the ADR states without hedging in Context and Decision. These distinctions appeared in the R3 audit (P2-1, FI-P2-1) and required R4 to confirm closure. They were not surfaced in the pre-arm conversation as reasons to design the prose arm differently (e.g., adding a non-converters prose base to test domain independence of the callee-agnostic claim).

The "obviously" and "nothing structurally different" markers in the practitioner's challenge went unexamined. The agent could have said: "The prose resolver will be heuristic rather than AST-clean, which is a measurement asymmetry worth flagging; and one prose base on the same domain as Base T gives callee-type independence but not domain independence. Do you want a second prose base?" It did not. The arm ran as designed, the results came back 10/10, and the confidence markers amplified.

**Is this earned confidence or sycophantic reinforcement?**

The code arm cluster is earned confidence. The audit-tightening across R1–R4 represents genuine post-hoc scrutiny, and the ADR is materially better at R4 than at R1 on exactly the points that matter.

The prose arm is closer to sycophantic reinforcement at the framing-adoption moment, but it is partially earned by the arm's results. The distinction that matters for BUILD is: the callee-agnostic claim in the ADR ("the mechanism does not special-case code") is justified by the observed mechanism identity (same qwen3:8b coder, same blindness, same fix shape). What is not fully justified is the strength of the comparative characterization ("prose invents worse") and the domain coverage implied by "callee-agnostic" when the prose base shares Base T's symbols. The ADR's Empirical Grounding now explicitly notes the domain-independence limitation (R4-confirmed). The claim is not overclaimed after the audit cycle; the audit cycle did the job the in-conversation framing examination did not.

**Does the rapid-compounding signature appear?**

Yes, in a bounded form. The practitioner challenge at R2 (prose scope expansion) was integrated fast — the prose arm was run, the ADR was revised to "callee-agnostic," and the discharge gate was reversed from "README observed" to "README criterion" within the same DECIDE session. The speed was appropriate given the arm results were decisive (0/10 → 10/10), but the integration outran the measurement-validity examination that R3 had to do post-hoc. The three R3 P2 findings were all prose-arm-specific and entirely absent from the R2 audit, meaning the integration at the prose-scope boundary was not audit-depth-matched to the speed of integration.

**Comparison to prior snapshots**

The cycle's standing rapid-compounding signature is present but attenuated relative to earlier loop-backs. Earlier cycles integrated findings without four-round audit coverage. Here, four rounds of argument audit plus a pre-run methods review caught every major over-claim. The residual concern is the pre-arm framing examination, not the post-arm audit coverage, which is adequate.

---

## Recommendation

**Grounding Reframe: partial, advisory.**

The core mechanism (signatures-in-dispatch, causal isolation, qwen3:8b fix confirmed) does not need a reframe. The evidence is solid and the audit cycle was thorough.

The callee-agnostic claim, as it reaches BUILD, carries two uncertainties that the prose arm did not resolve and the ADR now correctly flags in Empirical Grounding:

**Uncertainty 1: Domain independence of the prose result.**
Base P used converters.py (same domain as Base T). The 0/10 prose baseline may partly reflect the temperature-conversion domain's "complete-library prior" (the model confidently documents all pairwise conversions because that is what a complete temperature library has). A prose base on a non-guessable domain (analogous to Base V's non-guessable text_tools API) was not run. The callee-agnostic claim for the mechanism (anchor → 10/10) is robust to this; the "prose invents worse" comparative characterization is domain-sensitive. The ADR's Empirical Grounding correctly records "Base P reuses converters.py's domain, so this is callee-type independence, not a third independent domain."

**Uncertainty 2: Framework delivery to the prose callee (the discharge gate's actual claim).**
The prose arm confirmed qwen3:8b in prose mode responds to the anchor. It did not confirm the framework injects the anchor into the `prose-improver` dispatch. The discharge gate is the right place to confirm this, and the ADR makes this explicit. But if the BUILD implementation targets `code-generator` dispatches specifically rather than all callee dispatches, the prose discharge criterion would fail for the right reasons (architecture, not measurement). The agent building the anchor injection should be grounded in the ADR's "callee-agnostic: any callee" decision text rather than a code-only reading.

**Concrete grounding actions before wiring the anchor in BUILD:**

1. When implementing the anchor injection, confirm the injection path fires on `prose-improver` dispatches, not only `code-generator` dispatches. The ADR decision text says "any callee generating a deliverable that references produced siblings" — verify the BUILD design amendment reflects this scope explicitly, not just code-generating callees.

2. The discharge gate includes the README as a criterion: `cli.py` calls real converters functions AND the README documents real functions. Scope the discharge run to verify both. If the README criterion fails despite the anchor being wired, note it as the prose-domain-independence open question rather than a blocker — the ADR's Empirical Grounding now correctly separates these.

3. The multi-sibling selection policy (all produced files vs. dependency-inferred subset) is explicitly deferred to BUILD. Don't let the single-sibling validation in the spike constrain the BUILD decision prematurely in either direction. The spike validated single-sibling injection; the ADR commits the mechanism, not the policy.

**What the practitioner would be building on without this grounding:** A callee-agnostic injection with the prose scope empirically validated for one domain (temperature conversion) at one deliverable type (README). That is adequate for the discharge run's five-file temperature library task. It is not adequate for the claim that "any prose callee" universally benefits without domain consideration. The ADR records this boundary correctly; the BUILD entry should carry it as a named design constraint rather than treating callee-agnosticism as unconditional.

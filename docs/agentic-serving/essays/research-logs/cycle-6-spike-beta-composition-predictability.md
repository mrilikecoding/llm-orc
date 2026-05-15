# Spike β (Cycle 6) — Composition Predictability under Common Envelope

**Date:** 2026-05-15
**Cycle:** 6 (agentic-serving mini-cycle; ensemble-contract cluster)
**Wave:** DECIDE-blocking analytical spike (paired with spike α; spike γ independent)
**Method:** Analytical — read the three lit-review-composition ensemble YAMLs, their most-recent `execution.json` artifacts, and the dispatch-side result-extraction logic. No live dispatch.
**Scope:** the three-stage composition `web-searcher` → `claim-extractor` → `argument-mapper` — the 3m 54s end-to-end run captured in the 2026-05-13 post-hotfix verification (finding 1) and its associated output-spec drift (finding 2). Free-tier policy honored.
**Cost incurred:** $0.00 (no LLM dispatch)

---

## Question

With a common envelope (the three candidates from spike α — A thin, B additive typed, C artifact-substrate), do A→B compositions become structurally predictable rather than ad-hoc text-passing? Where do current assumptions about output structure actually live in the orchestrator's reasoning? Would a typed envelope move them into the contract instead?

Spike α surfaced three candidate envelopes and flagged four open questions: (1) does typed `structured` carry composition load; (2) does `primary`-as-handle break composition; (3) where do implicit composition assumptions live; (4) what does the three-stage composition look like under each envelope. Spike β addresses these against the empirical artifact trail.

---

## Method

1. Read spike α's findings end-to-end. Internalized the three envelope candidates (A thin convention; B additive typed `structured` + `output_schema:`; C artifact-as-substrate with handle-style `primary` and `artifacts[]`).
2. Read the three ensemble YAMLs: `.llm-orc/ensembles/agentic-serving/{web-searcher,claim-extractor,argument-mapper}.yaml`. Captured each ensemble's `default_task` (the declared output spec).
3. Read the three `execution.json` artifacts under `.llm-orc/artifacts/agentic-serving/<ensemble>/latest/`. The three artifacts are from a single contiguous lit-review-composition session (timestamps span 2026-05-13 17:05–17:08; phase_metrics in each artifact reference the three earlier agents, confirming this is the same composition shape the verification ran). Captured each stage's `input.data` (what the orchestrator actually sent) and `results.<agent>.response` (what each stage actually emitted).
4. Read `src/llm_orc/agentic/result_summarizer_harness.py::_extract_summary` to confirm where the orchestrator-side extraction happens — i.e., where a typed-envelope field would have to land for the orchestrator to consume it.
5. Traced the data flow A→B→C. Located the implicit composition assumptions concretely: in `default_task`, in `input.data`, in the agent's prompt, in the summarizer's `default_task`, or in the orchestrator's narration. Tested each candidate envelope against the trace.
6. Cross-referenced with the post-hotfix verification finding 2 (output-spec drift) to test whether any envelope candidate makes the drift catchable.

The spike was bounded to this one composition shape. Other composition shapes (e.g., `code-generator` followed by `code-reviewer`; `prose-improver` chained on its own output) are out of scope. The spike's conclusions therefore generalize only to the extent that this composition shape is representative — see Limitations.

---

## Data flow trace: `web-searcher` → `claim-extractor` → `argument-mapper`

The empirical artifacts disclose a much richer flow than the cycle-status finding 1 line suggests. Three stages, three transformations between stages — none of which the envelope currently mediates.

### Stage 1 — `web-searcher`

**`default_task` says:** "Search the web for the provided query. Return top results with URLs and snippets."

**`input.data` actually was:** `"agentic orchestration as substrate for skill frameworks"` — the raw NL query, passed in by the orchestrator from the lit-review prompt.

**`results.searcher.response` actually emitted:** a JSON-string (not a JSON value) of the shape `{"backend": "ddgs", "query": "...", "result_count": 5, "results": [{"title", "url", "snippet"} × 5]}`. The whole structured payload is serialized to a string and lives inside the `response` field of the agent result.

**Duration:** 1.31s. Output size ~3 KB of JSON-as-string.

**Key observation:** the structure is fully present — `web-searcher` is a script-agent whose Python adapter returns a typed dict that the dispatch layer then string-encodes into `response`. The envelope-wrapping of structure-as-string is an artifact of the current `response: str` contract at the dispatch boundary. The structure exists; it is just hidden behind a string.

### Stage 1 → Stage 2 transformation (orchestrator-side)

**This is where the empirical trace diverges sharply from the "the next stage consumes the previous stage's output" mental model.**

The `claim-extractor`'s `input.data` is **not** the web-searcher's `response` string. It is a hand-curated NL framing produced by the orchestrator:

> "web-searcher returned results about agentic orchestration and skill frameworks. Fetch and analyze the two most relevant papers: 1. The 2025 comprehensive survey that categorizes agentic systems into two paradigms — directly addresses the theoretical foundation. 2. The LangGraph vs CrewAI comparison — addresses practical skill framework implementations. After extracting claims from these sources, map the argument structure focusing on: how agentic orchestration serves as substrate for skill frameworks, the relationship between orchestration primitives and skill composition, and architectural patterns that enable skill portability. Return the complete mapped argument structure."

Three transformations happened between stages 1 and 2, all in the orchestrator's reasoning surface:

1. **Selection.** Five web-searcher results were narrowed to two — the orchestrator chose which "most relevant papers" to feed forward. This is a *selection* the envelope contract does not see.
2. **Reframing.** The orchestrator described what each selected paper "directly addresses" — interpretive labeling that came from the orchestrator's reading of titles and snippets, not from web-searcher.
3. **Task expansion.** The orchestrator gave claim-extractor a *combined task*: not just "extract claims" but also "map the argument structure focusing on ...". This expanded task is **the prose form of a multi-stage composition specification** delivered as a single `input.data` string.

The result-summarizer harness ran between stages 1 and 2 (the `phase_metrics` shows a `summarizer` phase right after `searcher`). The summarizer's output is the orchestrator's narrative anchor for what web-searcher produced — but the data the orchestrator used to write claim-extractor's `input.data` cannot have come from a 2-3 sentence summary alone. Either the orchestrator retained context across summarization, or the summarizer's output preserved enough structure for the orchestrator to identify two specific papers by title.

### Stage 2 — `claim-extractor`

**`default_task` says:** "Extract the factual claims from the provided text. List each claim on its own line as a bullet. After each claim, label it in parentheses as either: (established) — ... (contested) — ... Output only the bulleted list — no preamble, no synthesis, no conclusions."

**`input.data` actually was:** the orchestrator's combined task above — which **does not match `default_task`'s contract**. The orchestrator asked claim-extractor to "map the argument structure" and "Return the complete mapped argument structure." Claim-extractor's spec says "Output only the bulleted list — no preamble, no synthesis, no conclusions."

**`results.extractor.response` actually emitted:** a ~2.2 KB markdown document titled "Mapped Argument Structure: Agentic Orchestration as Substrate for Skill Frameworks" with three numbered sections (Agentic Orchestration as Substrate; Relationship Between Orchestration Primitives and Skill Composition; Architectural Patterns for Skill Portability), each broken into Theoretical Foundation / Practical Implementation sub-sections, plus a synthesis table and a conclusion paragraph. No `(established)` / `(contested)` labels. No bulleted-claims list.

**Duration:** 83.14s. Output ~2400 tokens.

**Key observation:** the post-hotfix verification finding 2 reported this as "output-spec drift" — `claim-extractor` deviated from its `default_task`. The spike β trace shows the more precise diagnosis: **`input.data` overrode `default_task`.** The orchestrator told the ensemble to produce an argument-mapped analysis; the LLM agent (qwen3:8b) complied with `input.data`'s instructions rather than `default_task`'s. The ensemble's "spec" is the union of `default_task` (compile-time) and `input.data` (run-time); when they conflict, run-time wins. This is the spec drift's mechanism.

### Stage 2 → Stage 3 transformation (orchestrator-side)

Again the orchestrator hand-curated the next stage's input. `argument-mapper`'s `input.data`:

> "**Extracted Claims from Literature:** 1. Agentic orchestration systems can be categorized into two paradigms: centralized and decentralized [...9 numbered claims total...] **Research Question:** agentic orchestration as substrate for skill frameworks. **Requested Mapping Focus:** Relationship between orchestration primitives and skill composition; architectural patterns enabling skill portability"

Three more transformations:

1. **Distillation.** Claim-extractor's 2.2 KB markdown was distilled into 9 numbered NL claims. None of the source markdown's section headers, sub-headers, or synthesis table appear; the orchestrator extracted *what claim-extractor should have emitted under its `default_task`* and presented those as the input to argument-mapper.
2. **Reframing.** The orchestrator added a "Research Question" header and a "Requested Mapping Focus" instruction. Neither comes from claim-extractor's output; both come from the orchestrator's session memory of the original lit-review prompt.
3. **Format injection.** The orchestrator chose markdown-with-bold-section-headers as the wire format. Argument-mapper's `default_task` does not specify input format; the orchestrator imposed one.

### Stage 3 — `argument-mapper`

**`default_task` says:** "Map the logical structure of the argument [...] Produce a structured analysis with these sections (in this order): **Premises** [...] **Intermediate conclusions** [...] **Final claim** [...] **Logical gaps** [...] Output only the four sections."

**`input.data` actually was:** the orchestrator's distilled 9-claim list above.

**`results.mapper.response` actually emitted:** a ~2.2 KB markdown document titled "Mapping: Orchestration Primitives, Skill Composition, and Architectural Patterns for Skill Portability" with sections "Relationship Between Orchestration Primitives and Skill Composition", "Architectural Patterns Enabling Skill Portability", "Synthesis: Paradox of Flexibility vs. Control", "Conclusion". No Premises / Intermediate conclusions / Final claim / Logical gaps headers.

**Duration:** 86.50s. Output ~2200 tokens.

**Key observation:** the same pattern. `input.data`'s "Requested Mapping Focus" instruction asked for two specific topical maps (orchestration primitives ↔ skill composition; architectural patterns ↔ skill portability); the agent produced those topical maps and overrode `default_task`'s four-section structural prescription. **Output-spec drift is again `input.data` overriding `default_task`.** The drift is not an LLM-quality-failure mode; it is a contract-conflict failure mode where the contract has two competing voices and run-time wins.

### Three-stage summary

| Stage | `default_task` spec | What `input.data` asked for | What `response` delivered |
|---|---|---|---|
| `web-searcher` | "Return top results with URLs and snippets" | NL query | JSON-string of typed result list — matched spec |
| `claim-extractor` | "Bulleted list, each (established)/(contested), no synthesis" | "Map the argument structure focusing on..." | Multi-section argument-mapped analysis — matched `input.data`, drifted from `default_task` |
| `argument-mapper` | "Four sections: Premises / Intermediate / Final / Gaps" | "Requested Mapping Focus: primitives↔composition; patterns↔portability" | Two-topic mapping + Synthesis + Conclusion — matched `input.data`, drifted from `default_task` |

The composition ran end-to-end in 3m 54s and produced coherent lit-review output. The output is **not** the composition the ensembles' `default_task`s describe — it is a different composition shape the orchestrator constructed at run-time by hand-writing each stage's task. Cycle 5 BUILD's "five capability ensembles compose with per-skill tier defaults" acceptance criterion has live-deployment evidence at three-stage depth (the cycle-status finding 1 is correct); it is also live-deployment evidence that **the composition substrate is the orchestrator's prose-writing, not the ensemble contract**.

---

## Where the implicit composition assumptions live

The trace localizes the composition assumptions. They are not in one place — they live in five, with very different remediation properties.

1. **In each ensemble's `default_task`.** The output-shape declaration is here (bulleted list with labels; four sections; etc.). This is the *compile-time* spec. The trace shows it does not bind the agent at run-time when `input.data` disagrees.

2. **In each per-invocation `input.data`.** The orchestrator's hand-written task description is here. This is the *run-time* spec, and it wins. Composition assumptions about "what the next stage should produce" live here as prose instructions ("Return the complete mapped argument structure", "Requested Mapping Focus: ...").

3. **In the orchestrator's session memory / reasoning surface.** The "two most relevant papers" selection between stages 1 and 2; the 9-claim distillation between stages 2 and 3; the "Research Question" reinjection at stage 3 — all of these are the orchestrator integrating stage N's output with the original prompt and producing stage N+1's input. This integration is not visible in any artifact; it lives in the orchestrator LLM's reasoning between tool calls. **This is the composition substrate.**

4. **In the result-summarizer's `default_task`.** The summarizer's "two or three sentences" prescription shapes what the orchestrator receives back from each dispatch. Whatever the orchestrator uses to write stage N+1's `input.data` passes through this filter (with the `needs_client_tool` exception path). If the summarizer strips structure, the orchestrator must reconstruct it from session memory; if the summarizer preserves structure, the orchestrator has a richer substrate to work with. The trace shows the orchestrator successfully reconstructed structure (titles, claim lists) between stages — either the summarizer preserved it, or the orchestrator retained the pre-summarization context, or both.

5. **In the result-summarizer harness's extraction logic** (`_extract_summary` at `src/llm_orc/agentic/result_summarizer_harness.py:118`). The harness reads `synthesis` first, then falls back to "the only agent's `response`" for single-agent ensembles. This logic *is* a tiny ad-hoc parser — it encodes the assumption that a single-agent ensemble's value lives at `results.<only-agent>.response`. The assumption is correct for the current eight ensembles but is conventional, not contractual.

The implicit composition assumptions are **not** in ad-hoc text-parsing as the spike β open question worried. They are in the orchestrator's NL synthesis between dispatches. The text-massaging is real; it just happens inside the orchestrator LLM, not in a string-manipulation code path.

---

## Per-envelope composition test

The trace gives a sharp question for each envelope candidate: **does declaring this envelope change where the composition assumptions live?**

### Candidate A — Thin convention (`ensemble`, `status`, `primary`, `diagnostics`)

Under A, `primary` is the client-facing string (the agent's response, currently). The dispatch return shape simplifies from `{results: {agent: {response: "..."}}}` to `{primary: "..."}`. Diagnostics get a typed home.

**Composition effect:** zero. The three transformations between stages (selection, reframing, distillation/format-injection) all run on the prose payload regardless of whether it is delivered as `results.searcher.response` or `primary`. The orchestrator still has to read prose, write prose, and hand-curate the next stage's `input.data`.

**Where the composition assumptions move:** they do not move. Candidate A codifies the de facto contract without rerouting any assumption.

**What candidate A gets for composition:** marginally less context-walking for the harness (one field instead of an agent-keyed dict). The result-summarizer harness's `_extract_summary` simplifies — the single-agent fallback becomes `invocation["primary"]` and the multi-agent case is just the same field. This is a code-cleanup win, not a composition win.

**Verdict:** A does not make composition predictable. Under A, the lit-review composition continues to be orchestrator-prose-mediated. Composition assumptions stay implicit in the orchestrator's reasoning surface.

### Candidate B — Additive typed `structured` + `output_schema:`

Under B, ensembles can declare `output_schema:` in their YAML (e.g., `claims_v1`); their dispatch return includes a typed `structured: {...}` payload alongside the human-readable `primary`. Web-searcher's structured payload (`{backend, query, results[]}`) moves from string-encoded `response` to typed `structured.results[]`. Claim-extractor declares `output_schema: claims_v1` with required `(established)/(contested)` labels. Argument-mapper declares `output_schema: argument_map_v1` with the four-section structure.

**Composition effect — under the literal artifact trace:** still limited. The trace shows that even with typed `structured` payloads, the orchestrator hand-wrote each stage's `input.data` and the LLM agents responded to `input.data` rather than to the ensemble's `default_task`. Adding `output_schema:` declarations would expose the drift mechanically (claim-extractor's `response` did not contain `(established)` labels; an `output_schema: claims_v1` validator would catch that at the dispatch boundary). **B makes the drift catchable but does not stop it** — the validator would fire and the dispatch would either fail or fall back, but the underlying mechanism (`input.data` overriding `default_task`) is upstream of the validator.

**Composition effect — under a stronger interpretation of B:** if the cycle commits to typed `structured` carrying composition load, the orchestrator can be reworked so that stage N+1's `input.data` is *constructed from* stage N's `structured`, not from stage N's `primary` prose. Argument-mapper's `input.data` could be the literal `structured.claims[]` from claim-extractor's output, machine-passed via a dispatch-layer hook rather than orchestrator-prose-rewritten. This is a substantial change to the orchestrator's role: from prose-integrator to typed-pipeline executor.

**Where the composition assumptions move under strong B:** they move from the orchestrator's session memory and `input.data` prose into the `output_schema:` declarations and a dispatch-layer pipeline binding (something like `argument-mapper.input_from: claim-extractor.structured.claims`). The integration is in the contract, not in the orchestrator's reasoning surface. The orchestrator's role narrows to "which ensembles to chain"; the schema-binding handles "how they connect."

**What strong B costs:** every per-skill ensemble that wants to participate in composition must declare its `output_schema:`. Composition shapes that branch (selection between alternatives — "fetch the two most relevant papers" — rather than pure feed-forward) need a *selection* step the schema-binding cannot do alone. The 2026-05-13 verification ran exactly such a selection (5 results → 2 papers); under strong B, either the orchestrator continues to be the selector (and composition is partly typed-pipelined, partly orchestrator-mediated) or the selection becomes a typed contract too (e.g., a `selector` ensemble whose `output_schema:` is "subset of input list with rationale").

**Verdict:** B makes drift catchable mechanically (this is real and useful). B *enables* composition predictability under a stronger interpretation that reshapes the orchestrator's role; it does not deliver predictability by declaration alone. The two interpretations diverge on whether typed `structured` is decorative (catches drift; orchestrator still mediates) or load-bearing (orchestrator narrows to chain-selection; binding handles flow).

### Candidate C — Artifact-as-substrate (`primary` as handle, `artifacts[]` typed)

Under C, the LLM-text capability ensembles write their `response` to an artifact file under `.llm-orc/artifacts/<ensemble>/<timestamp>/`. The dispatch return's `primary` becomes a handle ("Extracted 9 claims to <path>"); the `artifacts[]` entry carries `{path, content_type, summary, retention}`.

**Composition effect — direct trace test:** suppose web-searcher's output is in `artifacts[0].path` (the JSON-string written to a `.json` file). Claim-extractor's `input.data` is currently the orchestrator's NL framing referencing "the two most relevant papers." Two cases:

1. **Orchestrator passes the artifact reference to claim-extractor without reading the artifact.** Claim-extractor's agent then needs to read the file to do its work — meaning the agent layer needs filesystem access, not just the orchestrator layer. This is a significant capability extension; the current LLM-agent has no filesystem read tool. Adding one couples the LLM-agent to the dispatch host's filesystem layout. *And* it does not solve the selection problem: the agent still has to choose "the two most relevant papers" from the artifact's 5 results, and the orchestrator's selection (which was the actual selection step in the trace) is bypassed.

2. **Orchestrator reads the artifact, selects from it, and writes a new `input.data` for claim-extractor.** This is structurally the same as today — the orchestrator is still the integrator, doing the same selection/reframing work. The only difference is that the deliverable's body is on disk rather than in the orchestrator's context. **This is C's main win**: the orchestrator's context window does not carry the 5-result JSON; the orchestrator reads only the summary and dispatches a file-read tool call to inspect the artifact when it needs to. The selection still happens; it happens against an on-disk substrate.

The trace does **not** support case 1 — the agent-layer filesystem-read extension is not in scope for Cycle 6, and the orchestrator's integrator role is what produced the working composition in the first place. C therefore operates in case 2 mode.

**Composition effect under case 2:** the deliverable is artifacted; the orchestrator's role as integrator persists; `input.data` is still hand-written prose. The composition is not more "structurally predictable"; it is *less context-window-expensive*. The integration substrate stays implicit-in-orchestrator-reasoning.

**Where composition assumptions move under C:** they do not move out of the orchestrator's reasoning surface. They become *cheaper to act on* — the orchestrator can re-read an artifact across turns without context-window cost — but their location is unchanged from today.

**What C uniquely gets for composition:** the AS-7 collapse the MODEL gate Action B flagged is real here. Argument-mapper's 2.2 KB response is artifacted; the summarizer summarizes the artifact metadata rather than the content; the orchestrator's context across the three-stage composition stays bounded. The cycle's information-finding overhead finding (post-hotfix finding 6) is addressed at the cost-of-context dimension. **But the spec drift finding (finding 2) is not addressed by C** — `input.data` still overrides `default_task` regardless of whether the response is artifacted.

**Verdict:** C reduces composition's context cost. It does not make composition more predictable. The artifact-as-substrate proposal's value is bounded-context-during-multi-turn-composition, not typed-composition.

### Cross-candidate observation

None of A, B (weak), or C move the composition assumptions out of the orchestrator's reasoning surface. Only B (strong) does — and B (strong) requires a substantial reshaping of the orchestrator's role from prose-integrator to chain-selector. The cycle's current orchestrator (MiniMax M2.5-free via Zen, OpenCode tool surface) is empirically a prose-integrator; the trace shows it doing prose-integration successfully. Changing the role is a bigger architectural commitment than the envelope contract alone implies.

**The four spike-α open questions resolve as follows:**

1. *Does typed `structured` carry composition load?* — Only under a stronger interpretation of candidate B than spike α stated, and only if the orchestrator's role narrows. Under the literal interpretation, `structured` carries *drift-detection* load but not composition load.
2. *Does `primary`-as-handle break composition?* — No, because the orchestrator continues to be the integrator under C (case 2). It does not improve composition predictability either; it reduces context cost.
3. *Where do implicit composition assumptions currently live?* — In the orchestrator's reasoning surface (selection, reframing, distillation), with secondary anchors in `input.data` prose and in the result-summarizer's filtering.
4. *Three-stage composition shape under the envelope?* — Substantially unchanged under A, B (weak), or C. Substantially changed (chain-selector orchestrator, typed pipeline bindings) only under B (strong).

---

## Output-spec drift intersection

The post-hotfix finding 2 framed three candidate diagnoses for the drift: (a) synthesizer agent deviating from `default_task`; (b) orchestrator restructuring during response shaping; (c) orchestrator's NL narration overriding raw ensemble output.

The trace shows a fourth, more specific diagnosis: **(d) per-invocation `input.data` overrides compile-time `default_task` when the two conflict.** The mechanism is not synthesizer drift (claim-extractor and argument-mapper are single-agent ensembles with no synthesizer) and not orchestrator response-shaping (the artifacts capture the raw agent response, pre-narration). It is contract conflict resolved by run-time precedence.

**Does any envelope catch the drift?**

- **A.** No. Candidate A has no schema; nothing to validate against.
- **B.** Mechanically yes — `output_schema: claims_v1` declares required `(established)/(contested)` labels; the dispatch boundary runs the validator on `structured`; the validator fails; the dispatch returns `status: failed` or `errors[]: [{kind: schema_violation, ...}]`. **But this fails the run**, which is operationally undesirable when the orchestrator deliberately reframed the task. The validator does not know whether the spec was a hard constraint or a default the orchestrator should be free to override. Either the schema declarations are advisory (validator surfaces a warning in `diagnostics`; dispatch succeeds) or they are mandatory (validator fails the dispatch; orchestrator cannot override `default_task` via `input.data`). The two interpretations imply different cycle commitments.
- **C.** Orthogonal. Artifact-substrate stores whatever the agent emitted; spec compliance is unaffected.

**Is the drift actually a defect?** The trace surfaces a tension. The 2026-05-13 composition produced coherent, useful lit-review output by overriding both intermediate ensembles' `default_task`s. If the orchestrator had been prevented from doing so (mandatory schema enforcement), the composition would have failed at stage 2 — the orchestrator's request "map the argument structure" was structurally incompatible with claim-extractor's "output only bulleted claims." The drift is the *mechanism by which the composition succeeded*. Treating it as a defect to eliminate is in tension with treating it as a flexibility-the-orchestrator-relied-on.

**Two readings DECIDE can hold:**

1. **Drift is a defect.** Ensembles have committed contracts; `input.data` is the per-invocation parameter slot, not a contract-override slot. Fix by hardening the dispatch boundary to surface schema violations as failures and by retraining the orchestrator to dispatch the right ensemble for the right task rather than reframing one ensemble to do another's job. Composition shapes must be expressed as multi-ensemble chains, not as task-expansion-within-one-ensemble.

2. **Drift is the contract surface.** The `default_task` is a default; `input.data` is the run-time spec; the agent serves the run-time spec. Capability ensembles are *capability shapes* (LLM with model profile + system prompt) and the orchestrator composes them by writing task prose. Fix by acknowledging this in the ensemble contract — `default_task` becomes documentation of the typical use, not a binding output schema; the envelope's `structured` field is opt-in for ensembles that want stronger typing.

The reading the cycle settles on shapes whether candidate B's `output_schema:` is mandatory or advisory, and shapes whether the typed-pipeline-binding interpretation of B is on the table at all.

---

## Artifact-as-substrate composition case

Spike β was asked specifically whether candidate C supports composition or breaks it because the next stage needs content rather than a reference. The trace says **C supports composition under case 2 (orchestrator reads artifact, distills, writes next stage's `input.data`) but not under case 1 (orchestrator passes reference; next-stage agent reads file).**

Case 1's blockers:

- LLM-agent currently has no filesystem-read tool. Adding one is a non-trivial agent-layer capability extension.
- Agent-layer filesystem access creates coupling between agent prompts and dispatch host layout — agents written for one deployment don't transfer cleanly.
- Selection / distillation steps the orchestrator performs between stages (5 results → 2 papers; 2.2 KB markdown → 9 claims) cannot be done by the consuming agent alone because the consuming agent doesn't know the orchestrator's selection rationale. The selection is the orchestrator's interpretive work tying stage N's output to the original prompt.

Case 2's structure:

- Orchestrator emits a tool call (e.g., `read_artifact(path)`) to inspect stage N's output when needed for stage N+1 task-construction.
- Stage N+1's `input.data` is hand-written as today; the deliverable's content is folded in by the orchestrator as needed.
- The artifact persists on disk so the orchestrator can re-read it across turns at no context-window cost.

Case 2 is functionally close to "today + the orchestrator's context window is smaller." It does not break composition; it does not improve composition's predictability; it reduces composition's context cost. **This is candidate C's actual proposition for composition** — a context-cost win, not a structural-predictability win.

**One specific case where C uniquely helps composition:** the `code-generator` deliverable. The 2026-05-14 follow-on observation noted that the code body rides through the orchestrator's context and the result-summarizer's pipeline. Under C, the code goes to disk; the orchestrator's `primary` says "Wrote <symbol> to <path>"; subsequent dispatches that act on the code (e.g., a future `code-reviewer` ensemble) read it from the artifact. This *is* artifact-pipelined composition without the agent-layer filesystem-read tool — because the artifact is read by the dispatch layer when constructing the next call's `input.data` (the `code-reviewer` agent gets the code in its `input.data`, just as it does today; the difference is the orchestrator's context never carried it).

The composition pattern that benefits from C is: **composition shapes where the deliverable is large and the next stage operates on the deliverable itself rather than on an interpretation of it.** The lit-review composition does not fit — argument-mapper does not operate on claim-extractor's raw 2.2 KB markdown; it operates on the orchestrator's 9-claim distillation. A hypothetical code-review composition does fit — code-reviewer would operate on the actual code, not on a paraphrase.

---

## Implications for T16 DECIDE

Spike α + spike β jointly contribute the following to T16 deliberation. The first sub-question (snapshot Action 4: scope) and sub-question (b) (contract: typed field vs. convention) are sharpened most.

**Sub-question (0) — Scope.** Spike β adds a *composition-shape-conditional* finding to spike α's size-floor finding. Composition shapes split into two kinds:

- **Interpretation-mediated composition** (lit-review shape: stage N produces text; orchestrator interprets and distills; stage N+1 receives orchestrator's interpretation as `input.data`). For this shape, candidate C is a context-cost optimization, not a composition substrate. Candidates A, B (weak), and C are all roughly equivalent for composition predictability.
- **Deliverable-mediated composition** (hypothetical code-generate → code-review shape: stage N produces a deliverable; stage N+1 operates on the deliverable). For this shape, candidate C is a structural commitment — the deliverable lives on disk and the orchestrator's context stays small. Candidate B's `output_schema:` declarations also matter here because the deliverable's shape is the contract between stages.

DECIDE's scope question therefore needs to specify *for which composition shape*. The 2026-05-13 verification is interpretation-mediated; the practitioner's 2026-05-14 code-generator observation is deliverable-mediated; the same ensemble library serves both shapes.

**Sub-question (a) — Boundary.** Spike α's per-ensemble partition (artifact-substrate candidates: `code-generator`, `prose-improver`, `claim-extractor`, `argument-mapper`; response-substrate: `agentic-calibration-checker`, `web-searcher`, `text-summarizer`) is robust to spike β's findings — none of the trace contradicts the partition. The size-floor remains useful as a defensible boundary.

**Sub-question (b) — Contract (typed field vs. convention).** Spike α recommended typed field. Spike β sharpens this: the typed field is **drift-catching infrastructure** (advisory) by default; only under a stronger architectural commitment (orchestrator role narrows to chain-selector; typed pipeline bindings carry composition data) does the typed field become **composition infrastructure** (mandatory). The cycle should not assume that declaring `output_schema:` automatically delivers composition predictability — the orchestrator's reasoning surface continues to mediate composition until either bindings or selectors are added.

**The MODEL gate Action B caveat is supported by spike β.** The "collapses three findings simultaneously" claim was flagged as agent-composed. Spike β's per-finding test:

- *Output-spec drift* — addressed by candidate B (typed schema with validator), **not** by candidate C. C is orthogonal to drift.
- *Information-finding overhead* — addressed by candidate C (context-bounded artifacts). Partially addressed by candidate B if the typed `structured` field replaces prose `primary` for downstream consumers (reduces re-reading load).
- *AS-7 summarizer content-stripping* — addressed by candidate C (summarizer summarizes metadata). **Not** addressed by candidates A or B (content still rides through the summarizer).

The collapse holds only if all three findings are bundled. Each finding has its own most-effective remediation, and the remediations are not identical. DECIDE benefits from treating the three independently and choosing a contract that addresses each with the most appropriate candidate, rather than relying on one envelope to collapse all three.

**T16's deliberation surface narrowed:**

- If the cycle wants drift caught → B mandatory.
- If the cycle wants context-cost capped on large deliverables → C with size-floor exception.
- If the cycle wants composition predictable through the contract → B strong (typed pipeline bindings + selector ensembles), which is a substantial commitment beyond Cycle 6's scope.
- If the cycle wants none of the above → A as documentation of de facto contract, deferring T16 entirely.

A defensible Cycle 6 outcome: B advisory + C with size-floor for deliverable-mediated composition shapes + orchestrator role unchanged. This treats the envelope as a typed home for fields the cycle already needs (`status`, `primary`, `structured`, `diagnostics`, `artifacts[]`, `errors[]`) and defers the composition-predictability question to a future cycle if and when the orchestrator-role change is on the agenda.

---

## Open questions

1. **Is the input.data-overrides-default_task drift a defect or the contract?** Spike β surfaces the tension; DECIDE deliberates. The reading shapes whether `output_schema:` is advisory or mandatory.
2. **Does the orchestrator-as-prose-integrator role hold up at higher composition depth?** The 2026-05-13 verification was three stages; the orchestrator handled it. Whether five-stage or branching compositions remain orchestrator-mediated, or whether the orchestrator's context degrades and pipeline-binding becomes structurally necessary, is empirically open.
3. **What does deliverable-mediated composition look like in practice?** The lit-review trace is interpretation-mediated. The code-generator → code-review composition is hypothesized but not observed in any artifact. A spike that drives a deliverable-mediated composition (or a verification prompt at PLAY time) would test whether candidate C's structural commitment delivers under actual deliverable-mediated load.
4. **Does the result-summarizer's filter strip composition-relevant structure?** The trace shows the orchestrator successfully reconstructed structure across stages. Whether this reconstruction is robust to the summarizer's "two or three sentences" default, or whether it depends on the orchestrator retaining pre-summarization context, is not visible in the artifacts. A summarizer-output capture would resolve this.
5. **For interpretation-mediated composition, is the orchestrator's prose-writing role a substrate the cycle's typed-contract work should respect, or one it should narrow over time?** This is the deepest open question and is upstream of T16 — it touches T14 (routing-preference) and T15 (observability) as well, because the orchestrator's reasoning surface is the shared substrate all three clusters interact with.

---

## Limitations

The spike does NOT test:

- **Other composition shapes.** The trace is one composition (`web-searcher` → `claim-extractor` → `argument-mapper`) under one orchestrator (MiniMax M2.5-free via Zen, OpenCode client). Other shapes (deliverable-mediated, branching, longer chains) may behave differently. The interpretation-mediated finding is the strongest generalization the trace supports; the deliverable-mediated case is theorized but not observed.
- **Drift rate.** Two of three stages drifted from `default_task` in the observed run. Whether this is a high-drift rate or a low one across the ensemble library would need a larger sample (multiple runs, multiple ensembles, different orchestrator profiles).
- **Strong-B feasibility.** The chain-selector orchestrator + typed pipeline binding interpretation is sketched architecturally; it is not validated by any artifact. The trace shows the prose-integrator role working; it does not show the chain-selector role failing or succeeding because nothing has been built that way.
- **Agent-layer filesystem access.** Candidate C case 1 was ruled out as out-of-scope for Cycle 6 but is not impossible — a future cycle might add agent-layer file-read tools. The spike's verdict that C operates in case 2 mode is current-state, not eternal.
- **The `agentic-result-summarizer`'s preservation of structure.** The orchestrator's between-stage integration relied on identifying papers by title and distilling claims — whether this came through the summarizer or from the orchestrator's pre-summarization context is empirically open.
- **Multi-orchestrator-profile composition behavior.** Spike γ characterizes the routing-preference behavior across orchestrator profiles; whether composition behavior (prose-integration vs. typed-pipeline-amenable) also varies across profiles is unobserved.

The spike's primary finding — that composition assumptions live in the orchestrator's reasoning surface rather than in ad-hoc text-parsing — is robust to these limitations because the artifacts directly disclose the `input.data` rewriting at both inter-stage boundaries. The downstream implications for envelope choice (A, B, C) depend on whether the orchestrator's prose-integrator role is the substrate the cycle wants to preserve, narrow, or replace.

---

## Cross-references

- `docs/agentic-serving/essays/research-logs/cycle-6-spike-alpha-envelope-survey.md` — the survey of envelope candidates A, B, C that spike β tests against. Open questions 1–4 at the end of α are addressed in spike β's per-envelope composition test.
- `docs/agentic-serving/housekeeping/cycle-status.md` — post-hotfix verification finding 1 (composition end-to-end works) and finding 2 (output-spec drift) — spike β localizes the drift mechanism precisely.
- `docs/agentic-serving/domain-model.md` — §Concepts entries on Common I/O envelope, Artifact-as-substrate, Output-spec drift; Open Question #15 (AS-7 amendment pathway); MODEL gate Action B (three-findings-collapse claim flagged agent-composed) — supported by spike β's per-finding analysis.
- `docs/agentic-serving/decisions/adr-021-skill-orchestration-via-per-capability-dispatch.md` — current per-capability dispatch contract; the envelope ADR T16 produces amends or supersedes this.
- `docs/agentic-serving/decisions/adr-004-result-summarization-mandatory.md` — invariant AS-7; the summarizer's role in composition's between-stage data flow.
- `.llm-orc/ensembles/agentic-serving/{web-searcher,claim-extractor,argument-mapper}.yaml` — the three composed ensembles' `default_task` specs.
- `.llm-orc/artifacts/agentic-serving/{web-searcher,claim-extractor,argument-mapper}/latest/execution.json` — the three artifact instances that disclose the composition flow.
- `src/llm_orc/agentic/result_summarizer_harness.py:118` — the `_extract_summary` extraction logic showing where typed-envelope fields would have to land for the orchestrator to consume them.

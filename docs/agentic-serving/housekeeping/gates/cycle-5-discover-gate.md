# Gate Reflection: Agentic Serving DISCOVER → DECIDE (Cycle 5)

**Date:** 2026-05-12
**Phase boundary:** DISCOVER → DECIDE
**Cycle:** Cycle 5 — Agentic-serving library structure (capability ensembles + multi-skill-framework-consumer surface)

## Belief-mapping question composed for this gate

> The candidate Methodology Consumer is the highest-stakes new framing in this update. What would have to be true for the *right* framing to be that there's no distinct Methodology Consumer at all — that methodology composition is one mode of the Ensemble Author / Operator's work, the same person authoring RDD skills today and running one-off coding tasks tomorrow, with the same library serving both?
>
> Conversely: what would have to be true for Methodology Consumer to be *sharply distinct* — different infrastructure needs, different value tensions, different jobs — than Ensemble Author / Operator?
>
> The answer determines whether DECIDE treats methodology consumer as a separate stakeholder with its own scenarios + interaction-specs, or as a role overlay folded into the existing Ensemble Author / Operator's surfaces.

## User's response

> Methodology consumer is probably more accurately described as someone who is using a skill orchestration process. So RDD is an example of this, but there are other skill frameworks out there. The idea is that a user may use RDD which has subskills, which will decomposable tasks that the orchestrator would need to delegate to the appropriate ensemble. The alternative framing would be that we determine a specific flow (like RDD) is essential for driving effective llm-orc agentic serving results and therefore we encode a specific flow into llm-orc. That's not my first choice. Better is to be able to leverage any agentic skill (as "skills" are more-or-less standardized these days).

## Pedagogical move selected

**Challenge** via belief-mapping. The question presented two sharp alternatives on the methodology-consumer framing (no distinct role vs. sharply distinct role) and asked what would have to be true for each. The practitioner answered substantively in a third direction: the role is distinct in *kind* (defined by *using a skill orchestration process*) but the wearer can be any of the existing stakeholders. The architectural commitment broadens from the proposal's "methodology-agnostic" to "skill-framework-agnostic" — covering RDD as one instance among many current and emerging skill standards.

The belief-mapping was composed against the highest-stakes load-bearing framing in the substrate (the proposal-introduced Methodology Consumer role), not against the safest target. The alternative the practitioner rejected ("encode a specific flow like RDD into llm-orc") is named explicitly in their answer, satisfying the inversion principle's surface-the-alternative requirement.

## Commitment gating outputs

**Settled premises (the practitioner is building on these going into DECIDE):**

1. **Skill Orchestration User / Methodology Consumer is a distinct role** — not just a candidate; humans can wear it alongside Tool User and Ensemble Author / Operator roles; concerns are distinct (composing against the library via skill orchestration is different from maintaining the library or interacting through a tool).

2. **The orchestrator commitment is skill-framework-agnostic** — broader than the proposal's "methodology-agnostic" framing. Covers RDD, Anthropic Skills, OpenAI Assistants, MCP-based skill frameworks, and emerging skill standards as the skill abstraction stabilizes across vendors.

3. **The Topaz 8-skill taxonomy is the lingua franca** between skill frameworks (decomposing higher-level workflows into capability-typed sub-tasks) and capability dispatch (routing by Topaz skill + calibration verdict). The orchestrator does not know which skill framework is composing against it; the framework does not know which ensemble will be invoked.

4. **The capability ensemble library is capability-fine-grained / operation-named**, not methodology-coarse. Same library serves multiple skill frameworks. "Author the operation-named capability ensemble set that serves the deployment's skill frameworks" is the operator-driven-migration shape — not "tag every existing ensemble with whatever topaz_skill fits."

5. **Working defaults are in Cycle 5 BUILD scope** (per tension #11 reframing). The cycle delivers operator-facing deployment shape (default profile file, tagged capability ensembles, subdirectory layout, README) — not just mechanism architecture. Practitioner verbatim from Cycle 4 PLAY note 1: *"the agentic-serving config is to me part of the build."*

6. **Reliability profile observation for the Orchestrator LLM** — high on derivable claims (grounded in file content / tool-call results), low on integration claims (counts, comparisons, justifications, primitives-level system descriptions). Consistent across introspection / recommendation / evaluation modes. Treat orchestrator output as useful but audit-worthy, not authoritative.

**Open questions (the practitioner is holding these open going into DECIDE):**

1. **OD-1 through OD-6** from the proposal — all six Open Decisions need DECIDE resolution:
   - OD-1: `mathematical_reasoning` slot strategy (recommended (b) unauthored)
   - OD-2: `tool_use` ensemble shape (script-agent / MCP / client-side delegation; ADR-003 amendment territory if MCP)
   - OD-3: Skill-orchestration composition shape (formerly "methodology-layer composition shape") — multi-ADR territory
   - OD-4: Web-search backend (Brave / Tavily / Exa / Serper / DDG)
   - OD-5: Placement of general-purpose ensembles (style decision)
   - OD-6: Skill-framework / capability-ensemble naming registry (new artifact territory)

2. **No-dispatch fallback (note 19)** — is this a coverage gap (Cycle 5+ ADR territory) or intended scope (orchestrator narration is *meant* to bypass dispatch infrastructure for tasks no ensemble matches)? Framing examination needed at DECIDE.

3. **Calibration on orchestrator-own-narration (note 16)** — Cycle 5+ ADR territory or outside the cycle's quality framing entirely? Hold; do not force resolution this cycle.

4. **"Three-layer architecture" / "capability ensemble" / "operation-named ensemble" vocabulary** — survives as operator voice through DECIDE work, or relocates to research voice? Test via concrete library-authoring decisions in DECIDE.

5. **`compose_ensemble` primitives misunderstanding (note 13)** — does this need a DECIDE scenario, and what?

**Specific commitments carried forward to DECIDE:**

1. **DECIDE OD-3 explicitly acknowledges the skill-framework-agnostic commitment is provisionally settled** (per Cycle 5 DISCOVER snapshot Advisory 1). Seam-case inversions — does Topaz-skill routing produce routing-quality parity across skill-framework contexts, or do framework-specific dispatch needs surface? — have not been examined at the gate. OD-3's deliberation is the natural venue.

2. **DECIDE explicitly dispatches the four inversion questions** named in cycle-status §"Three-layer framing under examination" (per Cycle 5 DISCOVER snapshot Advisory 2) to specific OD slots. The four questions are:
   - What would have to be true for the three-layer separation to be the wrong abstraction?
   - What would have to be true for "operation-named ensembles" to be wrong (vs. methodology-named)?
   - What would have to be true for the `agentic-` prefix / `agentic-serving/` subdirectory convention to be wrong?
   - What would the right ensemble decomposition look like if the orchestrator were *not* methodology-agnostic (now: not skill-framework-agnostic)?

3. **The Cycle 4 PLAY snapshot's attribution discipline holds at Cycle 5 DECIDE entry** — load-bearing empirical observations (n=1 evidence, zero `invoke_ensemble` on the proposal-evaluation prompt, coherent factual errors uncalibrated across notes) are usable substrate; framings (especially agent-introduced "fallback" / "two gaps" characterizations) require examination, not direct adoption.

## Susceptibility snapshot outcome

Snapshot at `housekeeping/audits/susceptibility-snapshot-cycle-5-discover.md`: **no Grounding Reframe warranted**. Two advisory carry-forwards integrated above (Advisories 1 and 2 as Specific commitments 1 and 2 in the commitments-carried-forward list).

Snapshot recognized substantial improvement over Cycle 4 PLAY's susceptibility profile: the prior snapshot's Advisory #2 (reclassify "settled" claims as "directionally strong, pending DECIDE deliberation") was visibly implemented; the vocabulary section's explicit three-tier disposition (settled at gate / candidate under DECIDE / candidate agent-introduced) is a structural improvement.

Residual pattern (a) case: the skill-framework-agnostic commitment is marked "settled; not under further inversion examination this cycle" while OD-3 — the practical test of whether that commitment holds at methodology boundary seams — is deferred to DECIDE. The commitment is anchored by practitioner verbatim and is substantive rather than passive adoption; the sequencing is acceptable provided DECIDE OD-3 deliberation includes the seam-case inversion. Specific commitment 1 above operationalizes this.

Pattern (b) not present. Tensions #11, #12, and #13 are stated as held tensions with open questions named; tension #13 explicitly preserves both candidate frames (coverage gap vs. intended scope) and assigns DECIDE the examination.

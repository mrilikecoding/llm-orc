# Skill-Framework Capability Registry

*Cycle 5 introduction (2026-05-12). Light registry mapping known skill frameworks to their capability-ensemble requirements. Maintained per-deployment; this file is the agentic-serving corpus's reference instance.*

**Purpose.** The orchestrator (per ADR-019) is skill-framework-agnostic — it does not know which skill framework is composing against it. Skill frameworks (per ADR-021) compose via per-capability dispatch — each sub-task is a single `invoke_ensemble` call tagged with a Topaz skill. This registry is **descriptive, not prescriptive**: it records which Topaz-skill capabilities known skill frameworks consume, so operators authoring a deployment-specific capability library can see what their target skill frameworks need.

The registry is a **client-side reference**, not an orchestrator-consulted lookup. The orchestrator's routing logic does not read this file. Operators read it to understand library coverage gaps for their target skill frameworks.

## Capability ensemble availability (Cycle 5 minimum-viable set)

| Topaz skill | Capability ensemble | Status |
|---|---|---|
| code_generation | `code-generator` | Authored (Cycle 5 BUILD) |
| tool_use | `web-searcher` | Authored (Cycle 5 BUILD per ADR-020) |
| mathematical_reasoning | *(unauthored)* | Slot configured; MissingSkillMetadataError on dispatch (recovery path) |
| logical_reasoning | `argument-mapper` | Authored (Cycle 5 BUILD) |
| factual_knowledge | `claim-extractor` | Authored (Cycle 5 BUILD) |
| writing_quality | `prose-improver` | Authored (Cycle 5 BUILD) |
| instruction_following | *(operator-deployment-specific)* | Slot configured; existing `development/code-review` ensemble tagged `instruction_following` per Cycle 4 PLAY config |
| summarization | `text-summarizer` | Authored (Cycle 5 BUILD; user-facing, distinct from internal `agentic-result-summarizer`) |

## Known skill frameworks and their capability consumption

### RDD (Research-Driven Development)

**Skill plugin:** Claude Code skill, `rdd:*` namespace.

**Decomposition shape:** Multi-phase methodology (research → discover → model → decide → architect → build → play → synthesize). Each phase decomposes into capability-typed sub-tasks the orchestrator can dispatch.

**Capability consumption (illustrative; not exhaustive):**

| RDD phase / sub-skill | Topaz capability invoked |
|---|---|
| `/rdd:research` → lit-review step | `tool_use` (`web-searcher`); `factual_knowledge` (`claim-extractor`) |
| `/rdd:research` → citation audit | `factual_knowledge` (`claim-extractor`); `tool_use` (`web-searcher` for verification) |
| `/rdd:research` → argument audit | `logical_reasoning` (`argument-mapper`) |
| `/rdd:research` → framing audit | `logical_reasoning` (`argument-mapper`) |
| `/rdd:decide` → ADR drafting | `writing_quality` (`prose-improver`); `logical_reasoning` (`argument-mapper`) |
| `/rdd:synthesize` → outline drafting | `writing_quality` (`prose-improver`); `summarization` (`text-summarizer`) |
| `/rdd:build` → code generation | `code_generation` (`code-generator`) |

**Status:** Cycle 5 BUILD minimum-viable set covers RDD's research / discover / decide / synthesize phase capabilities. RDD's build phase covered by `code-generator`. RDD's play and architect phases do not have direct capability-ensemble equivalents — they are decomposition-and-dispatch phases that operate on existing artifacts.

**Library coverage gap for RDD:** none in Cycle 5 minimum-viable set. The methodology composer (rdd:rdd orchestrator skill) decomposes; the capability ensembles dispatch. The end-to-end RDD-via-agentic-serving workflow is exercisable when Cycle 5 BUILD completes.

### Code-review-as-methodology (illustrative, not yet deployed)

**Skill plugin:** Hypothetical Claude Code skill, `code-review:*` namespace.

**Capability consumption (illustrative):**

| Code-review sub-skill | Topaz capability invoked |
|---|---|
| Architectural review | `code_generation` (`code-generator`); `logical_reasoning` (`argument-mapper`) |
| Style review | `instruction_following` (deployment-specific ensemble — existing `development/code-review` ensemble can serve) |
| Test coverage analysis | `code_generation` (`code-generator`); `tool_use` (`web-searcher` for documentation lookups) |

**Library coverage gap:** none in Cycle 5 minimum-viable set; uses the existing `development/code-review` ensemble for `instruction_following` plus the minimum-viable set.

### Anthropic Skills (runtime not yet evaluated against agentic-serving)

**Skill framework:** Anthropic's emerging Skills runtime.

**Capability consumption:** undetermined — pending operator evaluation. Anthropic Skills' decomposition model has not been mapped against the Topaz 8-skill taxonomy as of 2026-05-12. The registry's expected role is to record the mapping once an operator does the work; the entry is a placeholder until then.

### OpenAI Assistants (runtime not yet evaluated)

Same status as Anthropic Skills. Placeholder.

### MCP-based skill frameworks (pattern, not specific implementation)

**Skill framework:** Any client-side framework that uses MCP tools to expose its skill-decomposed sub-tasks to the orchestrator.

**Capability consumption:** pattern-level — capability ensembles serve the underlying capability needs regardless of how the framework exposes its decomposition. The orchestrator's `invoke_ensemble` is invokable from any MCP-aware client.

---

## How to extend this registry

Operators adding a new skill framework to their deployment:

1. Identify the skill framework's decomposition into capability-typed sub-tasks. Map each sub-task type to a Topaz skill (see ADR-015 for the eight-skill taxonomy).
2. Check whether the deployment's library has a capability ensemble for each consumed Topaz skill (see "Capability ensemble availability" above). If a slot is unauthored or the deployment's existing ensemble for the slot doesn't fit the sub-task shape, author a new operation-named capability ensemble per ADR-019.
3. Add an entry for the skill framework to this registry: decomposition shape, capability consumption table, library coverage gaps.

The registry is per-deployment; this file is the agentic-serving corpus's reference instance. Deployments fork or extend as their target skill frameworks evolve.

## Open questions

1. **Anthropic Skills / OpenAI Assistants mapping**: pending operator evaluation. The skill-framework-agnostic commitment (ADR-019) is structural; the practical capability-mapping for non-RDD skill frameworks is empirical and deployment-specific.

2. **Skill-framework-level analytics**: per ADR-021 §Negative, the orchestrator does not know which skill framework is composing against it, so methodology-level analytics (e.g., "RDD lit-reviews escalate twice as often as code-reviews") live client-side or in a separate operator-managed analytics surface. The registry does not address this; the Tier-Router-Audit's drift criteria (ADR-018) operate at the capability granularity, not the skill-framework granularity.

3. **`instruction_following` slot ambiguity**: the existing `development/code-review` ensemble carries `topaz_skill: instruction_following`, but the slot's semantic shape (any capability that follows complex instructions) is broad enough that many capabilities arguably fit. Whether `instruction_following` becomes a "catch-all" slot or stays bounded is a deployment-time question.

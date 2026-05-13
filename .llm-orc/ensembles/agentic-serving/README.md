# agentic-serving — capability ensemble library

This directory holds the **operation-named capability ensemble library**
that serves agentic-serving deployments. Per ADR-019, the orchestrator
is **skill-framework-agnostic**: it routes by capability (Topaz 8-skill
taxonomy + calibration verdict, per ADR-015) without knowing which
skill framework is composing against it. Each ensemble in this
directory is named for the **operation it performs**, not for any
methodology that consumes it.

## What lives here

Two categories of ensemble share this directory:

**Capability ensembles** — operation-named, `topaz_skill`-tagged,
dispatchable by any skill framework that decomposes its workflow into
Topaz-typed sub-tasks. Cycle 5 minimum-viable set:

| Ensemble | Topaz skill | Shape |
|----------|-------------|-------|
| `code-generator` | code_generation | Three-agent (coder → critic → synthesizer) |
| `claim-extractor` | factual_knowledge | Single-agent |
| `argument-mapper` | logical_reasoning | Single-agent |
| `prose-improver` | writing_quality | Single-agent |
| `text-summarizer` | summarization | Single-agent |
| `web-searcher` | tool_use | Script-agent (wraps a web-search API) |

**System ensembles** — internal to the orchestrator's quality
infrastructure. Named for the role they play; not capability ensembles
and not dispatched by skill frameworks. Names are name-stable —
operators do not rename these.

| Ensemble | Role | Referenced by |
|----------|------|---------------|
| `agentic-result-summarizer` | Summarizes `invoke_ensemble` results so the orchestrator's context carries natural-language summaries (ADR-004 / AS-7) | `agentic_serving.summarizer.ensemble` |
| `agentic-calibration-checker` | Quality-signal producer for the Calibration Gate's first-N invocations of every composed ensemble (ADR-007) | `agentic_serving.orchestrator.calibration.checker_ensemble` |

## The operation-named principle

Capability ensembles are named for **what they do**, not for the skill
framework that invokes them. `claim-extractor` extracts claims —
whether the consuming skill framework is RDD's lit-review, a
security-review-as-methodology source-claim pass, or an ad-hoc workflow
in the OpenCode client.

This is **not** methodology-coarse: there is no `rdd-lit-reviewer`, no
`security-source-extractor`, no `code-reviewer` ensemble in this
directory. Methodology-coarse naming would couple library entries to
specific skill frameworks and require duplicating library work for each
new methodology consumer. See ADR-019 §Rejected alternatives for the
full rejection rationale.

## Topaz skill metadata

Every capability ensemble carries a `topaz_skill:` field in its YAML
frontmatter declaring its primary Topaz skill (one of:
`code_generation`, `tool_use`, `mathematical_reasoning`,
`logical_reasoning`, `factual_knowledge`, `writing_quality`,
`instruction_following`, `summarization`). The Tier-Escalation Router
(ADR-015) reads this field at dispatch time to select the cheap-tier
or escalated-tier Model Profile based on `per_skill_tier_defaults` and
the Calibration Gate's verdict.

System ensembles do not carry `topaz_skill:` metadata — they are not
dispatched by the per-skill tier router; they back fixed roles wired
in the `agentic_serving:` config.

## Coverage and gaps

The Cycle 5 minimum-viable set covers six of the eight Topaz slots:

| Topaz slot | Ensemble | Notes |
|------------|----------|-------|
| code_generation | `code-generator` | Authored |
| tool_use | `web-searcher` | Authored (Tavily-backend default) |
| mathematical_reasoning | *(unauthored)* | Slot configured; dispatches hit `MissingSkillMetadataError` and the orchestrator's recovery path reformulates (per ADR-019 §"Working defaults") |
| logical_reasoning | `argument-mapper` | Authored |
| factual_knowledge | `claim-extractor` | Authored |
| writing_quality | `prose-improver` | Authored |
| instruction_following | *(deployment-specific)* | Slot configured; the existing `.llm-orc/ensembles/development/code-review.yaml` ensemble carries `topaz_skill: instruction_following` and serves this slot for the agentic-serving deployment |
| summarization | `text-summarizer` | Authored (user-facing; distinct from internal `agentic-result-summarizer`) |

Authoring a `math-solver` capability ensemble for the
`mathematical_reasoning` slot is operator-driven, deployment-specific —
file it under this directory with `topaz_skill: mathematical_reasoning`
when a methodology consumer surfaces concrete demand.

## How to add a new capability ensemble

1. Pick the operation name. Use a verb-noun phrase that names the
   operation (e.g., `fact-checker`, `code-reviewer`, `math-solver`).
   Avoid methodology-coded names (no `rdd-*`, no `security-*`).

2. Pick the primary Topaz skill. The eight skills are described in
   ADR-015 §"Topaz 8-skill taxonomy." If the ensemble could
   plausibly tag with more than one, pick the one closest to the
   ensemble's actual workload — the dispatcher reads one tag per
   ensemble.

3. Create `<ensemble-name>.yaml` in this directory. The agent shape
   differs by agent type — pydantic enforces these schemas, so
   getting them wrong surfaces as a "Skipping invalid ensemble"
   warning at `llm-orc list-ensembles` time.

   **LLM-agent shape** (the agent is a language-model call). Use
   when the agent reasons over text:

   ```yaml
   name: <ensemble-name>
   description: |
     <what the ensemble does, who consumes it, what its output shape is>

   topaz_skill: <one of the eight>

   default_task: |
     <task framing the agent receives if the dispatch doesn't override>

   agents:
     - name: <agent_name>
       model_profile: <profile name from .llm-orc/profiles/>
       system_prompt: |
         <agent role and instructions>
   ```

   Allowed agent fields (LLM-agent): `name`, `model_profile` (or
   `model` + `provider`), `system_prompt`, `depends_on`, `fan_out`,
   `input_key`, `timeout_seconds`, `temperature`, `max_tokens`,
   `output_format`, `fallback_model_profile`.

4. If the ensemble needs **real tool execution** (web fetch, file ops,
   external API), use a **script-agent shape** (per ADR-020 — see
   `web-searcher.yaml` as the reference instance). Add the script
   under `.llm-orc/scripts/agentic_serving/` and reference it from the
   ensemble YAML. The script reads JSON from stdin and writes JSON to
   stdout.

   ```yaml
   agents:
     - name: <agent_name>
       script: scripts/agentic_serving/<script_name>.py
   ```

   Allowed agent fields (script-agent): `name`, `script`,
   `parameters` (dict), `depends_on`, `fan_out`, `input_key`,
   `timeout_seconds`. **Not allowed on script-agents:** `type`,
   `system_prompt`, `model_profile`. The presence of `script:` is the
   discriminator that selects the script-agent schema; system prompts
   for script behavior belong inside the script itself.

5. If multiple skill frameworks will consume the ensemble, update
   `docs/agentic-serving/skill-framework-capability-registry.md` to
   record the new ensemble in the capability-availability table.

## Web-searcher environment variables

The `web-searcher` ensemble reads two environment variables at
dispatch time:

- `WEB_SEARCH_BACKEND` — backend name. Default: `tavily`. Cycle 5
  BUILD ships the Tavily adapter only. To add another backend, author
  a `_search_<backend>` function in
  `.llm-orc/scripts/agentic_serving/web_searcher.py` and register it
  in `BACKEND_ADAPTERS`.
- `WEB_SEARCH_API_KEY` — API key for the selected backend. Obtain a
  free-tier key from https://docs.tavily.com (Tavily) or your chosen
  backend's signup page. The key is read from the environment, not
  from `.llm-orc/config.yaml` (config files are checked in; environment
  is not).

On authentication failure, rate limit, or backend unavailability, the
script-agent returns a structured error object (e.g.,
`{"error": "authentication_failed", "backend": "tavily", ...}`) rather
than raising. The Calibration Gate's post-hoc result-check (ADR-007)
sees the error shape and produces a calibration signal appropriate to
the error; the orchestrator's reasoning surface acts on the structured
error rather than crashing the session.

## Cross-references

- **ADR-019** — Skill-framework-agnostic orchestrator + operation-named
  capability ensemble library (the design ADR for this directory's
  structure).
- **ADR-020** — `tool_use` ensemble shape (script-agent web-searcher
  with operator-configurable backend).
- **ADR-021** — Skill-orchestration composition via per-capability
  dispatch (how skill frameworks compose against this library).
- **ADR-015** — Per-skill tier-escalation router and Topaz 8-skill
  taxonomy.
- **`docs/agentic-serving/skill-framework-capability-registry.md`** —
  Descriptive registry mapping known skill frameworks to their
  capability-ensemble requirements.
- **`.llm-orc/profiles/agentic-*.yaml`** — One file per agentic-serving
  Model Profile. To swap a model in any role, edit the `model:` (and
  optionally `provider:` / `base_url:`) field in the corresponding file
  — no `.llm-orc/config.yaml` edit required. The seven profile files
  are: `agentic-orchestrator.yaml` (default, cheap-cloud via Zen),
  `agentic-orchestrator-offline.yaml` (local fallback),
  `agentic-tier-cheap-general.yaml`, `agentic-tier-cheap-summary.yaml`,
  `agentic-tier-escalated-general.yaml`,
  `agentic-tier-escalated-reasoning.yaml`, and `agentic-summarizer.yaml`
  (backing profile for the system ensembles).

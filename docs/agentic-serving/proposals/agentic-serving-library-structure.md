# Proposal: Agentic-Serving Library Structure

**Status:** Draft proposal (Cycle 4 PLAY-derived)
**Date:** 2026-05-12
**Origin:** Cycle 4 PLAY session; see `essays/reflections/field-notes.md` notes 14–16
**Pickup target:** Cycle 5+ DECIDE for open questions, then BUILD for implementation. Some items may be DECIDE-only (architectural) and some BUILD-only (mechanical authoring); see §"RDD-phase routing" below.
**Not yet implemented:** This is a spec for a future cycle. The current `.llm-orc/` deployment carries Cycle 4 PLAY's minimum-viable activation (inline model profiles in `config.yaml`; two tagged ensembles); the proposed structure below is the target state, not the current state.
**Evidence basis:** All architectural framings in this document — including the three-layer methodology/dispatch/execution separation, operation-named ensemble naming, the `agentic-` prefix convention — derive from a single inhabitation session (n=1). The follow-up cycle's DECIDE phase is the appropriate venue for examining whether these framings are the right ones. See §"What this proposal is and isn't" for the full evidence-basis qualifier and `housekeeping/audits/susceptibility-snapshot-cycle-4-play.md` for the snapshot that surfaced this disposition.

---

## Origin: what the play surfaced

Cycle 4 PLAY (2026-05-12) ran a single agentic-coding session through the WP-H4-complete stack. The session activated the cycle's new machinery (Tier-Escalation Router, Calibration Gate verdicts, Tier-Router Audit wiring) and surfaced three findings that this proposal builds on:

1. **The cycle's operator-driven library migration framing (ADR-015 §Negative) produced a configuration-surface gap at first-encounter.** The mechanism architecture is in code, but the operator-facing on-ramp (working defaults + a library of capability-named ensembles) is not part of what BUILD shipped. Practitioner verbatim: *"the agentic-serving config is to me part of the build."*

2. **RDD's tier-1 Architectural Isolation mechanism maps cleanly to `invoke_ensemble`'s fresh-context dispatch property.** RDD-via-agentic-serving (or any methodology that decomposes into capability-typed sub-tasks) is structurally feasible on the existing primitive surface; no ADR amendment is needed.

3. **The right architectural separation is three-layer:** methodology layer (client-side; rdd:* skills or any other methodology composer) → dispatch layer (orchestrator with Topaz 8-skill taxonomy as routing index) → execution layer (operation-named ensembles in the orchestrator's library). The library should be **capability-fine-grained, not methodology-coarse** — `claim-extractor` not `rdd-lit-reviewer`. Same library serves many methodology consumers.

This proposal captures the structural recommendation derivable from those findings.

---

## Three-layer architecture (load-bearing framing)

| Layer | Lives where | Vocabulary | Examples |
|-------|-------------|------------|----------|
| **Methodology** | Client-side skill plugin (e.g., rdd:* skills in `~/.claude/plugins/.../rdd/`) | Methodology-specific phase/skill names | `/rdd:research`, `/rdd:argument-audit`, `/code-review`, `/security-audit` |
| **Capability dispatch** | Orchestrator (llm-orc serve) | Topaz 8-skill taxonomy | `tool_use`, `factual_knowledge`, `logical_reasoning`, ... |
| **Capability instance** | Ensemble library (`.llm-orc/ensembles/`) | Operation-specific names tagged with `topaz_skill` | `web-searcher`, `claim-extractor`, `argument-mapper`, ... |

The orchestrator is **methodology-agnostic**. It receives a structured task (framed by the client-side methodology layer), routes by topaz_skill + calibration verdict, and dispatches to the capability ensemble matching the task's operation. The methodology layer is responsible for *what to do*; the orchestrator is responsible for *which capability to invoke*; the ensemble library is responsible for *how to execute the capability*.

---

## Proposed directory structure

```
.llm-orc/
├── config.yaml                                      # MODIFIED — agentic_serving: section
│                                                    #   keeps; inline tier/orchestrator profiles
│                                                    #   are removed (now in profiles file);
│                                                    #   per_skill_tier_defaults rewritten to
│                                                    #   reference renamed profiles
│
├── profiles/
│   ├── agentic-serving-profiles.yaml                # NEW — all agentic-serving Model Profiles
│   │                                                #   isolated for easy model-swap
│   ├── local-models.yaml                            # existing, untouched
│   └── ...other existing profile files
│
└── ensembles/
    ├── agentic-serving/                             # NEW subdirectory for agentic-serving
    │   │                                            #   capability ensembles
    │   ├── README.md                                # NEW — structure + extension guide
    │   │
    │   │   # --- System ensembles (internal, name-stable) ---
    │   ├── agentic-result-summarizer.yaml           # MOVED (name kept; config.yaml references it)
    │   ├── agentic-calibration-checker.yaml         # MOVED (name kept; ADR-007 reference)
    │   │
    │   │   # --- Capability ensembles (operation-named, topaz_skill-tagged) ---
    │   ├── code-generator.yaml                      # topaz_skill: code_generation
    │   ├── claim-extractor.yaml                     # topaz_skill: factual_knowledge
    │   ├── argument-mapper.yaml                     # topaz_skill: logical_reasoning
    │   ├── prose-improver.yaml                      # topaz_skill: writing_quality
    │   ├── text-summarizer.yaml                     # topaz_skill: summarization (user-facing,
    │   │                                            #   distinct from internal summarizer)
    │   └── (mathematical_reasoning / tool_use slots
    │        deferred — see §"Open decisions" below)
    │
    └── ... (existing ensembles outside agentic-serving/
              remain untouched; library/global ensembles
              also untouched)
```

The `agentic-serving/` subdirectory is a **clear namespace boundary** between the deployment-specific capability set and the rest of the library. Operators can add ensembles outside this subdirectory for non-agentic-serving uses; the dispatch path only cares that the tagged ensembles exist somewhere in the walk path.

---

## Proposed model profile file

`/.llm-orc/profiles/agentic-serving-profiles.yaml` — pulls all agentic-serving model assignments into a single file for single-edit model swaps:

```yaml
# All Model Profiles used by the agentic-serving deployment, isolated
# from the rest of `.llm-orc/config.yaml` so model swaps are a single-
# file edit. Profile names declared here are referenced from
# `agentic_serving.orchestrator.model_profile` and from
# `per_skill_tier_defaults` slots in `.llm-orc/config.yaml`.

model_profiles:

  # === Orchestrator profiles (ADR-011) ============================

  # Default — cheap-cloud orchestrator via OpenCode Zen (free tier).
  # Aligns with essay 005's "cheap-cloud-orchestrator routes; local
  # models amplify deterministic and bounded-scope ensemble work."
  agentic-orchestrator:
    model: minimax-m2.5-free
    provider: openai-compatible/zen
    base_url: https://opencode.ai/zen/v1
    cost_per_token: 0.0
    timeout_seconds: 180

  # Offline-fallback orchestrator — local Ollama. Use when Zen is
  # unavailable or for fully-offline deployments.
  agentic-orchestrator-offline:
    model: qwen3:14b
    provider: ollama
    cost_per_token: 0.0
    timeout_seconds: 300

  # === Tier profiles (ADR-015) ====================================

  agentic-tier-cheap-general:    # cheap-tier for most skills
    model: qwen3:8b
    provider: ollama
    cost_per_token: 0.0
    timeout_seconds: 180

  agentic-tier-cheap-summary:    # cheap-tier for summarization (smaller/faster)
    model: qwen3:1.7b
    provider: ollama
    cost_per_token: 0.0
    timeout_seconds: 90

  agentic-tier-escalated-general:  # escalated for most skills
    model: qwen3:14b
    provider: ollama
    cost_per_token: 0.0
    timeout_seconds: 300

  agentic-tier-escalated-reasoning:  # escalated for math + logical reasoning
    model: deepseek-r1:8b
    provider: ollama
    cost_per_token: 0.0
    timeout_seconds: 240
```

**Swap pattern:** to change a model in any role, edit only the `model:` (and optionally `provider:` / `base_url:`) field. The `agentic_serving:` section in `config.yaml` references roles by profile name, not by model — no config.yaml edits required.

---

## Proposed `config.yaml` agentic_serving section (rewritten)

```yaml
agentic_serving:

  orchestrator:
    model_profile: agentic-orchestrator    # was: orchestrator-minimax-m25-free

    calibration:
      default_n: 3
      checker_ensemble: agentic-calibration-checker

    compaction:
      persist_threshold_chars: 50000
      idle_window_minutes: 60
      session_notes_token_cap: 12000
      layer_4_circuit_breaker_threshold: 3
      trigger_token_count: 100000

    per_skill_tier_defaults:
      code_generation:
        cheap_tier: agentic-tier-cheap-general
        escalated_tier: agentic-tier-escalated-general
      tool_use:
        cheap_tier: agentic-tier-cheap-general
        escalated_tier: agentic-tier-escalated-general
      mathematical_reasoning:
        cheap_tier: agentic-tier-cheap-general
        escalated_tier: agentic-tier-escalated-reasoning
      logical_reasoning:
        cheap_tier: agentic-tier-cheap-general
        escalated_tier: agentic-tier-escalated-reasoning
      factual_knowledge:
        cheap_tier: agentic-tier-cheap-general
        escalated_tier: agentic-tier-escalated-general
      writing_quality:
        cheap_tier: agentic-tier-cheap-general
        escalated_tier: agentic-tier-escalated-general
      instruction_following:
        cheap_tier: agentic-tier-cheap-general
        escalated_tier: agentic-tier-escalated-general
      summarization:
        cheap_tier: agentic-tier-cheap-summary
        escalated_tier: agentic-tier-escalated-general

    tier_router_audit:
      trigger_count: 100
      trigger_wall_clock_hours: 24
      verdict_distribution_shift: 0.15
      escalation_outcome_correlation_pp: 0.05
      bypass_rate_increase: 0.25
      severe_drift_multiplier: 2.0

    tool_call_validation_patterns: []

  # budget intentionally absent — inherits global config's caps
  autonomy: { default_level: operator-as-tool-user }
  plexus:   { enabled: false }
  overrides:
    allow_budget_override: true
    max_turn_limit: 1000
    max_token_limit: 50000000
  summarizer:
    ensemble: agentic-result-summarizer
```

---

## Capability ensemble specs (minimum viable set)

Each ensemble below is single-agent unless noted; uses tier-cheap-general by default (router overrides per dispatch). Shape is consistent: `name`, `description`, `topaz_skill`, `default_task`, `agents`.

### `code-generator.yaml` (topaz_skill: `code_generation`)

Three-agent flow (coder → critic → synthesizer) — promoted from Cycle 4 PLAY's `agentic-coding-helper`. Coder produces, critic flags issues, synthesizer integrates.

### `claim-extractor.yaml` (topaz_skill: `factual_knowledge`)

Single-agent. Extracts factual claims from input text. Output: bulleted list distinguishing established vs. contested claims.

### `argument-mapper.yaml` (topaz_skill: `logical_reasoning`)

Single-agent. Maps the logical structure of an argument — premises, intermediate conclusions, final claim, logical gaps. Output: structured analysis.

### `prose-improver.yaml` (topaz_skill: `writing_quality`)

Single-agent. Improves clarity and structure of prose while preserving voice and intent. Output: improved prose only (no editorial commentary).

### `text-summarizer.yaml` (topaz_skill: `summarization`)

Single-agent. User-facing summarization (distinct from the internal `agentic-result-summarizer` which is invoked by Tool Dispatch on the `invoke_ensemble` return path). Captures main claim, key supporting points, notable caveats.

---

## Open decisions (DECIDE-phase candidates)

### OD-1: `mathematical_reasoning` slot strategy

Spike α (research-log 005g) showed the existing library has zero `mathematical_reasoning` coverage. Three options:

- **(a)** Author a minimum-viable `math-solver.yaml` capability ensemble now (deepseek-r1:8b backing); accept that it may be over-provisioned for the cycle's task class.
- **(b)** Leave the slot configured (resolver requires all 8) but unauthored — dispatches routing to math hit `MissingSkillMetadataError` and the orchestrator's recovery path reformulates.
- **(c)** Author later when a methodology consumer actually exercises the slot.

Cycle 4 PLAY's verdict on (b): the recovery path works empirically (note 4), so leaving the slot unauthored is operationally safe. **Recommendation:** (b) until a methodology consumer surfaces the need. DECIDE if/when that happens.

### OD-2: `tool_use` ensemble shape

`tool_use` is the deepest capability gap. A capability-typed `tool_use` ensemble needs *real tool execution* — not just an LLM agent producing text. Three implementation shapes:

- **(a) Script-agent ensemble** — Python script wraps an external API (search, file ops, MCP-style tool call). Cleanest fit for the existing primitive surface. Cost-tier-conditional for paid APIs.
- **(b) MCP integration** — expose tools via MCP; the orchestrator gains them as additional tools beyond the closed 5-tool surface. Requires architectural decision: does the orchestrator surface expand?
- **(c) Client-side delegation** — agentic-coding client provides tools (file_read, web_fetch, etc.); orchestrator routes the task and the client's tool resolves it. Cleanest for clients with tool support; opaque to llm-orc's quality infrastructure.

ADR-003 currently specifies a closed 5-tool internal surface. Expanding it (option b) is an ADR-003 amendment. Option (a) fits the existing surface; option (c) sits outside it. **Decision needed at DECIDE.**

### OD-3: Methodology-layer composition shape

For RDD-via-agentic-serving specifically: how does the rdd:* skill plugin compose against the capability library? Options:

- **(a) Skill prompts decompose to multiple `invoke_ensemble` calls** in a single OpenCode turn (orchestrator multi-dispatches per phase).
- **(b) Skill prompts dispatch to a top-level methodology-composer ensemble** that internally invokes capability ensembles via `compose_ensemble`. This is more like the existing `rdd:rdd` orchestrator skill but moved server-side.
- **(c) Hybrid** — phase entry dispatches a methodology-shape ensemble; sub-tasks dispatch capability ensembles.

This affects what the orchestrator's `list_ensembles()` returns and how it routes. **Decision needed at DECIDE; possibly multiple ADRs.**

### OD-4: Web-search backend

If `tool_use` lands as a `web-searcher` ensemble (option (a) of OD-2): which search API?

- Brave Search API (free tier ~2000 q/month)
- Tavily (free tier with rate limit)
- Exa (paid)
- Serper (paid)
- DuckDuckGo (HTML scrape; brittle)

Authentication-shape question: per-deployment-config? per-operator-credential? **DECIDE-phase.**

### OD-5: Placement of general-purpose ensembles

Should `development/code-review.yaml` move into `agentic-serving/`, or stay in `development/` as a general-purpose ensemble that happens to be agentic-serving-tagged? Naming convention question. **Style decision; can be made at DECIDE or BUILD.**

### OD-6: Methodology-skill / capability-ensemble naming registry

If multiple methodologies (RDD, code-review-as-methodology, security-review-as-methodology, etc.) compose against the same capability library, is there a registry document mapping methodology-skill names → required-capability-ensemble names? This is the methodology-decomposition spec. Could live in the cycle's domain model or in a separate methodology-registry doc. **DECIDE-phase artifact territory.**

---

## RDD-phase routing

| Item | Phase | Reason |
|------|-------|--------|
| OD-1 (math slot strategy) | DECIDE | Architectural; minor ADR or domain-model amendment |
| OD-2 (tool_use shape) | **DECIDE → BUILD** | ADR-003 amendment territory if option (b); BUILD-mechanical if option (a) or (c) |
| OD-3 (methodology composition) | DECIDE | Multi-ADR territory; affects scenarios |
| OD-4 (web-search backend) | DECIDE | Operational decision; possibly ADR |
| OD-5 (placement convention) | DECIDE or BUILD | Style decision; either-or |
| OD-6 (methodology registry) | DECIDE | New artifact in the corpus |
| Authoring `code-generator`, `claim-extractor`, `argument-mapper`, `prose-improver`, `text-summarizer` ensembles | BUILD | Mechanical; specs above are sufficient |
| Creating `.llm-orc/profiles/agentic-serving-profiles.yaml` | BUILD | Mechanical; spec above is sufficient |
| Creating `.llm-orc/ensembles/agentic-serving/` subdirectory + moving system ensembles | BUILD | Mechanical |
| Rewriting `.llm-orc/config.yaml` agentic_serving section with new profile names | BUILD | Mechanical |
| Writing `agentic-serving/README.md` | BUILD | Mechanical |

**Recommended cycle shape for pickup:**

- Mini-cycle scoped to agentic-serving library completion. Mode A (full pipeline) is heavier than warranted; Mode D (custom) targeting **DISCOVER → DECIDE → BUILD** likely fits.
- DISCOVER step: update product-discovery's stakeholder model with the multi-methodology-consumer pattern (the orchestrator is a substrate; methodology consumers are diverse).
- DECIDE step: resolve OD-1 through OD-6, producing ADR(s) for the load-bearing decisions (tool_use shape; methodology composition shape).
- BUILD step: mechanical authoring per specs in this proposal. Practitioner can choose `auto` mode (per ADR-091) given the work's mechanical character, or `gated` for normal per-scenario-group review.

---

## What this proposal is and isn't

**Evidence-basis qualifier (per Cycle 4 PLAY susceptibility snapshot, advisory #2):** All architectural framings in this proposal — including the three-layer methodology/dispatch/execution separation, the operation-named (not methodology-named) ensemble principle, the `agentic-` prefix convention, and the capability-fine-grained naming pattern — derive from one inhabitation session (Cycle 4 PLAY, 2026-05-12, n=1) and have not been examined under a dedicated DISCOVER-phase assumption-inversion pass. They are *directionally strong* on the evidence available, but should not be treated as architectural settlements. The follow-up cycle's DECIDE phase is the appropriate venue for that examination.

**Is:**
- A specification of the *candidate* library structure derivable from Cycle 4 PLAY's findings
- A starting point for a follow-up cycle to deliberate against and (in pieces) implement
- A list of open decisions (OD-1 through OD-6) requiring DECIDE-phase deliberation

**Is not:**
- An implementation plan — BUILD-phase scoping happens at BUILD time
- A complete capability map — only 5 of 8 Topaz slots have proposed ensembles
- An RDD-specific design — the structure is methodology-agnostic; RDD is one of many candidate consumers
- A scenario inventory — DECIDE-phase work authors scenarios against the resolved decisions
- **A capture of settled architectural decisions** — the framings above are directionally strong and pending DECIDE-phase deliberation; the "settled" framing initially used in this document was a confidence escalation from n=1 PLAY evidence and has been reclassified per the snapshot advisory

---

## References to corpus

- **ADR-011** — orchestrator Model Profile + session-boundary discipline
- **ADR-014** — Calibration Gate verdict trichotomy
- **ADR-015** — per-skill tier-escalation router + 8-skill Topaz taxonomy
- **ADR-015 §Negative** — operator-driven library migration (the load-bearing framing this proposal sharpens)
- **ADR-016** — cross-layer calibration channel (referenced for the architectural-isolation property; cf. note 14)
- **ADR-017** — tool-call structural validation
- **ADR-018** — tier-router (d)-analog audit dispatch
- **essay 005** — cheap-cloud-orchestrator + ensembles design; Sub-Q6 (autonomous-routing reliability)
- **research-log 005g** — Spike α Topaz skill classification on existing library
- **system-design.agents.md §Module: Tier-Escalation Router** — module spec for the dispatch layer
- **field-notes.md notes 1–16 (Cycle 4 PLAY)** — the empirical anchor for this proposal; notes 14–16 are the load-bearing findings

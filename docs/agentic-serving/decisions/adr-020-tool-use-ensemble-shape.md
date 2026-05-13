# ADR-020: `tool_use` Ensemble Shape — Script-Agent Web-Searcher with Operator-Configurable Backend

**Status:** Proposed

**Date:** 2026-05-12

---

## Context

ADR-015 establishes per-skill tier defaults across the Topaz 8-skill taxonomy. The `tool_use` slot is the deepest capability gap in the existing library — `tool_use` is the only Topaz skill that requires *actual* external action (not just LLM-text generation), so authoring a `tool_use` capability ensemble is materially different from authoring `code-generator`, `claim-extractor`, etc., which are LLM-text-only.

The proposal `proposals/agentic-serving-library-structure.md` §OD-2 named three shape options:

- **(a) Script-agent ensemble** — Python script wraps an external API (search, file ops, MCP-style tool call). Cleanest fit for the existing primitive surface (ADR-003 closed 5-tool surface stands; the script-agent runs *inside* an ensemble dispatched via `invoke_ensemble`, not as an additional internal tool).
- **(b) MCP integration** — expose tools via MCP; the orchestrator gains them as additional tools beyond the closed 5-tool surface. **ADR-003 amendment territory** — would expand the orchestrator's internal action space.
- **(c) Client-side delegation** — the agentic-coding client (OpenCode, Roo Code, etc.) provides tools (file_read, web_fetch, etc.); orchestrator routes the task and the client's tool resolves it. Cleanest for clients with tool support; opaque to llm-orc's quality infrastructure (the client's tool execution is outside the orchestrator's dispatch path; calibration verdicts, tier-router audit, and ADR-016 signal-channel mechanisms do not fire on client-tool execution).

The skill-framework-agnostic commitment (ADR-019) constrains: the `tool_use` ensemble must be invokable by any skill framework that needs tool-use capability, not just by RDD or any other specific framework. Option (b) MCP integration is consistent with the commitment but is ADR-003 amendment territory. Option (c) client-side delegation is consistent with the commitment but does not produce a *library entry* — the capability lives in the client, not the orchestrator's library. Option (a) script-agent produces a library entry that any skill framework can `invoke_ensemble` against.

RDD's research workflow is the immediate methodology consumer demanding `tool_use` capability: RDD's lit-review skill decomposes into web-search sub-tasks. The proposal's library spec names `web-searcher` as the concrete `tool_use` capability ensemble for RDD's lit-review pickup, with the question of which search backend (Brave, Tavily, Exa, Serper, DuckDuckGo) carried forward as OD-4.

The closed 5-tool surface (ADR-003) is structurally defended: invariant-7-style guarantees, `match-case` dispatch + FC-4 static import check, system-prompt teaching. Expanding the action surface is a deliberate decision; not a default-friendly one.

---

## Decision

The `tool_use` Topaz slot is satisfied by a **script-agent ensemble** named `web-searcher` (option (a) of OD-2). The script agent wraps an external web-search API; the API choice is an **operator-configurable backend** (resolving OD-4).

### Ensemble shape

The `web-searcher` ensemble is a single-agent ensemble whose agent is a **script-model-slot** (per the agentic-serving domain model's primitive vocabulary). The script:

1. Receives the search query as agent input
2. Invokes the configured search backend's API
3. Returns a structured result (top-N URLs + snippets) as agent output
4. Surfaces errors (backend unavailable, rate limit, authentication failure) as structured error output the orchestrator's reasoning surface can act on

The ensemble carries `topaz_skill: tool_use` metadata so ADR-015's tier-escalation router dispatches against it correctly. The "tier" abstraction has degenerate meaning for script-agents — script execution is deterministic; there's no model to swap between cheap and escalated tiers. The tier defaults for `tool_use` reference the same Model Profile for cheap and escalated tiers (no-op tier escalation); the calibration verdict still fires (post-hoc result-check on the search result's structure), but tier escalation has no effect.

### Backend choice

The proposal §OD-4 named five candidates: Brave Search API (free tier ~2000 q/month), Tavily (free tier with rate limit), Exa (paid), Serper (paid), DuckDuckGo (HTML scrape; brittle). The decision:

**Default backend: Tavily** (free-tier, JSON API designed for LLM consumption, predictable structured output). Configured via operator-managed API key in the script-agent's environment.

**Adapter-pattern support for alternative backends**: the script-agent's implementation uses an adapter pattern so additional backends (Brave free-tier with credit-card-on-file; Exa paid; Serper paid) can be authored under the same pattern. **Cycle 5 BUILD scope ships the Tavily adapter only**; Brave/Exa/Serper adapters are deferred to operator-driven extension when a deployment needs them. The adapter pattern's purpose is to keep alternative-backend authoring mechanical (one-file Python addition + environment-variable selector); the alternatives are *supported by design*, not by shipped code.

DuckDuckGo HTML scrape is **not supported** as a default backend — HTML scrape is brittle, vulnerable to upstream UI changes, and produces lower-quality structured output than the JSON APIs. Operators wanting DuckDuckGo can fork the script-agent; the default deployment does not ship a DuckDuckGo adapter.

### Authentication shape

API keys are operator-managed per-deployment, stored in the operator's environment (not in `.llm-orc/config.yaml` — config files are checked in, environment is not). The script-agent reads the API key from `WEB_SEARCH_API_KEY` environment variable; the backend name is read from `WEB_SEARCH_BACKEND` (default `tavily`).

This matches the existing operator-environment-managed-secrets pattern in llm-orc (LLM provider API keys are environment-variable-resolved).

### Scope

`web-searcher` is the **only** `tool_use` capability ensemble Cycle 5 ships. Other tool-use shapes (file operations, calculator, code execution, database queries, etc.) are deferred to future cycles when a methodology consumer surfaces concrete demand. The `tool_use` slot is filled by `web-searcher` for RDD's lit-review demand; other deployments author additional `tool_use` ensembles at their leisure under the same shape principle (script-agent + operator-environment-managed authentication).

ADR-003's closed 5-tool surface is **unchanged**. The `web-searcher` script-agent runs *inside* an ensemble dispatched via `invoke_ensemble`; the orchestrator's internal action space remains exactly five tools.

---

## Rejected alternatives

### (b) MCP integration — expand the orchestrator's internal action space

Tools are surfaced via MCP; the orchestrator gains web-search (and other tools) as additional internal actions beyond the closed 5-tool surface. ADR-003 receives an amendment.

**Rejected because:** ADR-003's closed action space is a load-bearing architectural commitment (structurally defended via match-case dispatch and FC-4 static import check). Expanding it for `tool_use` opens the question of expansion for every other tool need future cycles surface. The script-agent ensemble shape produces the same capability without weakening ADR-003. MCP-style tools may have a place in llm-orc's future, but the entry point should be a deliberate ADR-003 amendment cycle, not the `tool_use` slot's first authoring.

A secondary consideration: MCP tools surfaced to the orchestrator's internal action space would be dispatched *outside* the calibration / tier-router / audit infrastructure (ADR-014, ADR-015, ADR-018 fire only on `invoke_ensemble`-dispatched ensembles). MCP-as-internal-action would bypass the quality infrastructure entirely — the same coverage shape Cycle 4 PLAY note 19 flagged as the no-dispatch fallback path's concern.

### (c) Client-side delegation — let the agentic-coding client provide the tools

The agentic-coding client (OpenCode etc.) provides web-search and other tools via its existing tool surface; the orchestrator routes the task and the client's tool executes it. No library entry is created.

**Rejected because:** the methodology consumer model (ADR-019) assumes the orchestrator's library is the substrate. A skill framework decomposing a workflow into capability-typed sub-tasks needs a *library entry* it can `invoke_ensemble` against. Client-side delegation produces no library entry — the methodology consumer would have to bypass the orchestrator for `tool_use` sub-tasks and emit a `finish_reason: tool_calls` to the client instead. That's not skill-framework-agnostic dispatch; it's skill-framework-specific client-integration.

Client-side delegation also makes the capability *deployment-specific* — Roo Code's tool surface differs from OpenCode's; an RDD skill plugin working through OpenCode would have different `tool_use` behavior than the same plugin working through Roo Code. The orchestrator's commitment is that the capability dispatch layer is uniform regardless of which client is consuming the response surface.

### (e) Multiple `tool_use` ensembles in Cycle 5 (file ops, calculator, code execution alongside web-searcher)

Cycle 5 authors a fuller `tool_use` capability set, anticipating future methodology demand.

**Rejected because:** the proposal recommends authoring on-demand per methodology consumer; the `tool_use` shape question (web-search ≠ file-ops ≠ calculator in deployment complexity) suggests each tool-use shape needs its own design pass. Cycle 5 ships `web-searcher` because RDD's lit-review demands it; future cycles ship other `tool_use` ensembles when concrete demand surfaces. The "premature library" risk (authoring capabilities no methodology consumer needs) is real; the proposal's substrate names it (capability ensemble set is deployment-specific).

### (a-DuckDuckGo) Default DuckDuckGo HTML scrape

DuckDuckGo HTML scrape is the only no-API-key search backend. Cycle 5 ships it as the no-authentication-required default; operators with API keys swap to Tavily/Brave/etc.

**Rejected because:**

*The quality argument (sufficient on its own to reject):* HTML scrape is structurally brittle (UI changes break the scraper) and produces lower-quality output than JSON APIs. Parsing snippets from rendered HTML is lossier than reading structured JSON fields; a brittle scraper that breaks on the upstream's next UI revision is a maintenance burden the deployment inherits permanently. The proposal §OD-4 names DuckDuckGo as "brittle"; the assessment stands.

*A secondary on-ramp consideration (acknowledged as framing extension):* a first-encounter operator running a no-authentication default that returns broken-or-low-quality results would face friction the working-defaults-in-BUILD commitment (per ADR-019, motivated by Cycle 4 PLAY note 1) was meant to avoid. This argument is an extension of note 1's literal scope (note 1 concerned missing configuration files, not search-backend quality); it is acknowledged here as additional motivation rather than as the rejection's primary basis. Note that Tavily's free tier is not friction-free either (signup is required, per §Negative below) — the rejection of DuckDuckGo rests on the quality argument; the on-ramp framing extension supports but does not carry the decision.

---

## Consequences

### Positive

- **`tool_use` slot has a working default** — RDD lit-review pickup is unblocked; other skill frameworks consuming `tool_use` find a working capability ensemble.
- **ADR-003 closed 5-tool surface preserved** — no amendment to the orchestrator's internal action space.
- **Quality infrastructure covers `web-searcher` dispatches via the post-hoc result-check path.** The Calibration Gate fires on `web-searcher`'s output via ADR-007's post-hoc result-check (structural schema verification of the returned JSON — count, fields, error-flag). Tier-Router Audit observes the dispatches; the cross-layer signal channel receives signals. ADR-014's AUQ (verbalized-confidence) and HTC (trajectory-feature) calibration components are *structurally inapplicable* to script-agent execution — there is no LLM reasoning trace inside a deterministic script — so the calibration value for `web-searcher` is on result-structure validation, not on dispatched-output confidence. The capability participates in the dispatch-infrastructure stack; the AUQ/HTC stack is dormant for it by construction.
- **Backend swap is a configuration change, not code** — operators move between Tavily, Brave, Exa, Serper without rewriting `web-searcher`. The adapter pattern in the script-agent handles the differences.
- **Authentication follows existing pattern** — API keys in the operator's environment, not in checked-in config. Matches LLM provider API keys.

### Negative

- **Operators must obtain an API key for the default backend** — Tavily free-tier signup is required. A no-authentication-required path (DuckDuckGo) was rejected; first-encounter operators face the Tavily-signup step. Mitigation: the README documents the signup briefly; the default is Tavily because it's the lowest-friction quality option, not because it's friction-free.
- **`tool_use` tier abstraction is degenerate for script-agents** — the ADR-015 tier defaults for `tool_use` are a no-op (cheap and escalated reference the same profile). Operators configuring `per_skill_tier_defaults` may find the `tool_use` slot's same-cheap-and-escalated pattern surprising; the README documents this.
- **Cycle 5 ships one `tool_use` ensemble, not a fuller set** — deployments needing file-ops, calculator, or code-execution as `tool_use` capability author them in future cycles. Cycle 5 does not provide a template for "other tool-use shapes" — each shape is its own design pass.

### Neutral

- **DuckDuckGo support is an operator fork** — operators committed to no-API-key search can fork the script-agent. The default deployment doesn't ship this path; the option exists.
- **The `web-searcher`'s calibration value is on result structure, not search quality** — Calibration Gate's checker can verify the search returned structured results (count, schema), but cannot easily verify the results are *good* (relevance, factual accuracy). Future cycles may surface richer checkers for `tool_use` capability quality.
- **MCP-as-action-space remains an open future possibility** — Cycle 5 does not foreclose ADR-003 amendment; it elects not to take that path for `tool_use`. A future deliberate ADR-003 amendment cycle remains available.

## Provenance check

- **Script-agent ensemble shape**: proposal substrate §OD-2 option (a) + ADR-019's principle that capability ensembles live in the library. Driver chain: substrate-derived.
- **Tavily as default backend**: drafting-time synthesis comparing the proposal's five candidates against the on-ramp-clarity constraint from Cycle 4 PLAY note 1 + the skill-framework-agnostic commitment. Tavily's free tier + JSON API + LLM-consumption-designed output makes it the lowest-friction quality option. The choice is best-judgment among proposal candidates rather than a research-derived recommendation; the README documents the choice transparently.
- **Backend adapter pattern**: drafting-time synthesis. Standard backend-abstraction pattern; not novel.
- **Environment-variable authentication**: existing llm-orc pattern (LLM provider API keys are environment-resolved). Driver chain: codebase-existing pattern.
- **One `tool_use` ensemble (not multiple)**: proposal §OD-2's deferral framing + ADR-019's deployment-specific library principle. Driver chain: substrate-derived.
- **MCP rejection grounded in ADR-003 closed-tool-surface commitment**: ADR-003 (driver). Driver chain: prior-ADR-derived.
- **Client-side-delegation rejection grounded in ADR-019's library-substrate commitment**: ADR-019 (driver). Driver chain: same-cycle-prior-ADR-derived.

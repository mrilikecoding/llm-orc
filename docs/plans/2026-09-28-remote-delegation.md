# Remote delegation: run any ensemble on a remote serve

Date: 2026-09-28 · Issues: #196 (serving project ships with the wheel),
#191 (one service; CLI as a thin client) · Target remote: remote-host
(`https://llm-orc.remote.example`, brew release, CPU-only).

## Decisions (practitioner, 2026-09-28)

1. Agentic serving is baked in, so the serving project ships in the wheel (#196).
2. One-run injection: `invoke` accepts an ensemble definition inline, runs it
   once, and persists nothing. When the remote can't run it, the call fails
   before any execution with a report saying what is missing and how to
   resolve each item.
3. Injection covers profiles too: inline profile definitions and per-run
   profile bindings.
4. Injected scripts are allowed. llm-orc does not sandbox them; the trust
   boundary is the deployment's network exposure (for us, the tailnet).
   Document this, don't enforce it.

## The pattern

Ensembles are portable and name roles (profile names). Each host binds roles
to models, including host-appropriate timeouts. Delegation is: **ship the
closure, preflight transitively, resolve explicitly, run.**

- **Closure** of an ensemble: the ensemble itself, every child ensemble
  (`ensemble:`, `loop:` bodies, `dispatch:` targets it can name), every
  script, and every profile name, transitively.
- **Preflight** on the remote classifies each dependency:
  `ready` | `pullable` (profile present, model not downloaded, source known)
  | `missing_profile` | `missing_model_source` | `needs_credentials`
  | `missing_script` | `missing_ensemble`.
- **Resolve**, per dependency, explicitly by the caller: ship it
  (inline ensemble/profile/script), `bind` a missing profile to one the
  remote has, or `pull: true` to allow downloads. Never automatic: a binding
  is recorded in the result like `model_substituted`, never chosen by the
  engine.
- **Run** once (injection) or persist with `scope: global` for reuse.

## Standing rules for every implementer (paid for last session)

- Work in your own git worktree off `main`; never `git stash` (shared with
  other sessions); don't spawn subagents that edit or commit.
- Strict TDD; structural and behavioral changes in separate commits; ruff 88
  + mypy strict from the first draft; complexity <= 15; `make lint` clean.
- Human commit messages (`feat:`/`fix:`/`refactor:`/`test:`/`docs:`); NO AI
  attribution, no Co-Authored-By, no Claude-Session trailers, no claude.ai
  links, no scratch/session paths in tracked files.
- Doctrine 11: pins assert outcomes through the real surface (REST
  TestClient, MCP tool call, CliRunner, real executor), and each key pin is
  shown RED under a mutant that reintroduces the defect. Report mutant +
  red line.
- Full suite `uv run pytest -q -p no:cacheprovider`. Known local-only
  failure: `testget_available_providers_auth_only` when a router listens on
  :8080.
- Findings go on the existing issues (#196, #191), not new ones.

## Arcs

Model per arc: Sonnet implements, Opus reviews (independent, adversarial,
before merge), Haiku does mechanical sweeps. Arcs 2-4 depend on the spikes;
revise their cards from the spike findings before starting them.

### Arc 0: two spikes (Sonnet, read-mostly, ~half a day)

Record findings as a dated "Spike findings" section appended to this file.

- **S1: project layering.** How do project, library, and global config merge
  today (`ConfigManager`: ensembles, profiles, `config.yaml`
  `model_profiles`, `serving:` keys)? List every `Path.cwd()` call site in
  `src/llm_orc` (known: `config_manager.py`, `primitive_registry.py`,
  `serving_ensemble_caller.py`, `v1_chat_completions.py`, `cli_library/
  library.py`, `artifact_handler.py`, `library_handler.py`,
  `script_handler.py`, `resource_handler.py`) and every runtime write into
  the project dir (known: `.llm-orc/artifacts/`, `.llm-orc/agentic-sessions/`
  via `core/session/artifact_store.py`, `.serve-trace/` in
  `serving_ensemble_caller.py`, `llama-server.ini` in
  `providers/llama_server.py:~384`). What does `serving.self_reference`
  (#144, `docs/plans/2026-08-13-dot-dir-self-reference-design.md`) assume
  about the project being the repo? Answer: can the serving project be a
  read-only layer, and what must move to a state dir?
- **S2: router runtime models.** Can llama-server's router (router mode,
  `--models-preset`) load a model that isn't in the preset, or reload the
  preset without a restart? Test against the real binary on the laptop.
  Answer decides whether Arc 3's `pullable` for a newly shipped profile is a
  live load, a supervised router restart, or a `needs_restart` status.

### Arc 1: `scope` on CRUD (Sonnet, ~1 day, independent of the spikes)

- `create_ensemble` / `update_ensemble` / `delete_ensemble`, and the same for
  profiles and scripts, accept `scope: "project" | "global"`, default
  `"project"` (no change for local use). `global` writes to
  `ConfigManager.global_config_dir / {ensembles,profiles,scripts}` (create
  the dir if absent).
- Files: `services/handlers/ensemble_crud_handler.py`
  (`get_local_ensembles_dir` ~265 is today's only target),
  `services/handlers/profile_handler.py`, `services/handlers/script_handler.py`,
  `web/api/ensembles.py`, `web/api/profiles.py`, `mcp/server.py` tool
  signatures, CLI if it has create commands.
- update/delete with `scope` act only on that scope; a name that exists only
  in another scope is a clear error naming where it lives.
- Pins: REST + MCP create with `scope: global` lands the file in the global
  dir and `list_ensembles` reports `source: global`; default still lands in
  the project.
- Live: against a local serve, create via REST with `scope: global`, list,
  invoke, delete.
- Then on the mini (practitioner does or okays): move `research-dossier` and
  `test-research-pipeline` from the checkout's `.llm-orc/ensembles/` into
  `~/.config/llm-orc/ensembles/`.

**Arc 1 live row (2026-09-28).** `llm-orc` 0.21.0, worktree
`feat/crud-scope` @ 076b05b7, serve on `--port 8766 --backend-port 8790
--models-max 1` against the real `~/.config/llm-orc/ensembles/`, agent
profile `local-qwen3-0.6b`.

- `POST /api/ensembles` with `scope: global` → `200`, file written to
  `~/.config/llm-orc/ensembles/scope-live.yaml`.
- `GET /api/ensembles` → `200`, `scope-live` entry shows `source: global`.
- `POST /api/ensembles/scope-live/execute` → `200`, `status: success`,
  `deliverable: "ok"` (3.2s, no cold-load wait).
- `DELETE ...?scope=project` → `500 Internal server error`, detail names
  the global tier (`Ensemble 'scope-live' is not in scope 'project'; it
  lives in the global tier at .../scope-live.yaml`); file untouched. Matches
  the brief's "errors naming the global tier," but as a `500` rather than a
  `4xx`: a client-input mismatch surfacing as a server error.
- `DELETE ...?scope=global` → `200`, `deleted: true`; file confirmed gone,
  rest of `~/.config/llm-orc/ensembles/` untouched.

### Arc 2: serving project ships with the wheel (#196) (Sonnet, 2-3 days, after S1)

- Package `.llm-orc/{ensembles,profiles,scripts,config.yaml}` (tracked files
  only; never `*.local.yaml`) into the wheel as `llm_orc/serving_project/`
  (`[tool.hatch.build.targets.wheel]` in `pyproject.toml`).
- `ConfigManager` adds it as a read-only layer: project, then library, then
  **packaged serving**, then global (confirm order with S1). Script
  resolution (`core/execution/scripting/resolver.py`, search paths ~238,
  ~287) and profiles get the same layer.
- Runtime state moves to a writable state dir (default: XDG state, e.g.
  `~/.local/state/llm-orc/`; `llm-orc serve --state-dir` overrides):
  artifacts, agentic-sessions, serve-trace, rendered preset. Every
  `Path.cwd()` site from S1 goes through the project context.
- A repo checkout still works as a project (it shadows the packaged layer),
  so development is unchanged.
- Pins: a serve started in an empty temp dir runs the `serving` ensemble from
  the packaged layer; nothing is written under the packaged path; a project
  file shadows a packaged one.
- Live: `uv build`, install the wheel in a fresh venv, start the serve from
  an empty dir, drive one OpenCode turn (T1) and one research-dossier run.
- Then `deploy/remote-host/`: plist `WorkingDirectory` becomes a plain dir (no
  checkout); README update path is `brew upgrade` only.

**Arc 2 re-cut (2026-09-29), from the S1 findings and a code read.** The
card above stands; these rulings replace its open points. Implementation
plan: `docs/plans/2026-09-29-remote-delegation-arc2.md`.

1. **Shipping.** `.llm-orc/` stays the source of truth in the repo. The
   wheel maps `.llm-orc/{ensembles,profiles,scripts,config.yaml}` to
   `llm_orc/serving_project/` through hatchling `include` + `sources`
   (the same selection rules as the package, so `.gitignore` applies:
   no `*.local.yaml`, artifacts, trace, preset or `__pycache__`).
   Moving the files under `src/` was rejected: it would put ~45 serving
   scripts under mypy strict, ruff, complexipy, bandit and vulture, and
   churn the history of 130 files. A checker (`scripts/check_wheel_contents.py`)
   asserts the packaged set equals `git ls-files .llm-orc`; it is red on
   the 0.21.0 wheel and on any checkout with an untracked file under
   `.llm-orc/` (this laptop has one: `ensembles/research-dossier.yaml`).
2. **Content ships verbatim**, `config.yaml` included (`serving.self_reference:
   true` stays on: the packaged scripts ARE the running wheel's scripts,
   so a self-read is still ground truth, S1 step 4). Curating the test
   and demo profiles out of the serving project is a separate chore, not
   Arc 2, so the mini's behavior surface after the move is byte-for-byte
   the checkout's.
3. **Layer order: project → library → global → packaged.** Packaged is
   the lowest tier, not the third as the card said. Reason: operator
   seat overrides are `*.local.yaml` files, and `.local.yaml` ordering is
   within a tier; on a plain-dir serve the only writable tiers are
   global, so global must beat packaged or `docs/serving.md`'s "Operator
   seat configuration" promise dies with the checkout. Library above
   global is today's order and keeps it. Name collisions measured:
   global's auto-provisioned templates (`example-local-ensemble`,
   `validate-*`, `local-models.yaml`, `research-profiles.yaml`) and the
   library's `security-review`, `validate-file-read`; none of them is a
   serving seat, and the two multi-profile files carry no `name:` so
   runtime resolution skips them. `classify_tier` gains `"packaged"`;
   listings show `source: packaged`.
4. **Runtime profiles (`get_model_profiles`) become a tier loop** over
   packaged → global → local (each tier: `config.yaml: model_profiles`,
   then `profiles/*.yaml`, `.local.yaml` last). The library tier stays
   out of runtime resolution, as today, and that is pinned: letting it
   in would let a submodule profile with `provider: ollama` shadow a
   global one by name, a behavior change #196 does not ask for.
   `performance:` and `agentic_serving:` merge defaults → packaged →
   global → local. `serving:` keys are read from the serving root
   (ruling 6), a two-tier shadow as S1 recommended.
5. **Scripts:** resolver search order becomes project (three entries) →
   package primitives → library → global → packaged (`<pkg>/scripts`,
   `<pkg>`), consistent with ruling 3. `list_available_scripts` honors
   `project_dir`. `PrimitiveRegistry` is left alone (S1: pre-existing
   gap, independent of #196).
6. **Serving root.** `ConfigurationManager.serving_root()` returns the
   dot-dir that carries `ensembles/agentic-serving/serving.yaml`: the
   project's `.llm-orc` when it does, else the packaged dir, else a
   `FileNotFoundError` naming both. It replaces
   `v1_chat_completions._resolve_serving_project_dir` (S1's concrete
   break). `ServingEnsembleCaller` keeps its dot-dir `project_dir`
   meaning and receives `trace_root` explicitly. The executor's child
   resolution (`_resolve_ensemble_reference`) searches every tier from
   `get_ensembles_dirs()` after the project dir; today it stops at the
   local dot-dir, so a global ensemble cannot reach a packaged child
   (`research-dossier` → `agentic-serving/web-searcher`).
7. **State dir.** `LLM_ORC_STATE_DIR` (and `llm-orc serve --state-dir`,
   which sets it) → else the project's `.llm-orc` when there is one
   (today's layout, so instruments reading `.llm-orc/.serve-trace/` and
   `.llm-orc/artifacts/` are unchanged) → else `$XDG_STATE_HOME/llm-orc`
   or `~/.local/state/llm-orc`. Under it: `artifacts/`, `.serve-trace/`,
   `cache/`, `llama-server.ini`. Every writer and every reader from the
   S1 step 3 table goes through `resolve_state_dir`. One visible change:
   a global-only serve's preset moves from `~/.config/llm-orc/` to the
   state dir.
8. **Override for tests and ops:** `LLM_ORC_SERVING_PROJECT_DIR=<path>`
   uses that directory as the packaged tier; an empty value disables the
   tier. The test suite sets it empty by default (the #86 lesson: a
   `ConfigurationManager` built in a temp cwd would otherwise see this
   checkout's serving project), and the packaged pins set it explicitly.
9. **Write fallbacks.** `get_local_ensembles_dir` / `get_local_profiles_dir`
   return the project dir or raise naming `scope: global` (Arc 1's ruling
   A; nothing on the CRUD path calls them). The orchestrator's
   `ConfigManagerEnsembleWriter` writes to the project dir, else global:
   an engine write with no caller to ask has to land somewhere writable,
   and never in the library or packaged tiers. `library_handler` and the
   three library-path sites in `ConfigurationManager` derive the checkout
   root from the local dot-dir's parent instead of cwd.
10. **Deploy.** The plist's `WorkingDirectory` becomes a plain directory
    with no `.llm-orc`; user ensembles live in `~/.config/llm-orc/`, state
    in `~/.local/state/llm-orc/`; `brew upgrade` + kickstart is the whole
    update path. The mini's 15 library ensembles disappear with the
    checkout unless `LLM_ORC_LIBRARY_PATH` names a library; the README
    says so. The practitioner does the move.

**Arc 2 live row (2026-09-29).** Laptop, branch `feat/packaged-serving` @
`9ec465b6`, `make wheel-check` → `ok: 162 serving project files match git
ls-files`; the wheel (`llm_orchestra-0.21.0-py3-none-any.whl`) installed into
a fresh `python3 -m venv`; `site-packages/llm_orc/serving_project/` carries
`config.yaml`, `ensembles/`, `profiles/`, `scripts/`. Serve started from an
empty directory with `XDG_CONFIG_HOME` and `XDG_STATE_HOME` pointed at fresh
temp dirs: `llm-orc serve --port 8766 --backend-port 8790 --models-max 1`
(llama-server 9850, GGUFs from the Hugging Face hub cache).

- `/health` → `healthy 0.21.0`; cwd still empty; `<state>/llm-orc/llama-server.ini`
  rendered (six model sections).
- `GET /v1/models` → `["agentic-tier-cheap-general"]` (the packaged
  `agentic_serving.orchestrator.model_profile`).
- `GET /api/ensembles` → 105 entries, `source` ∈ {`global`, `packaged`};
  `serving` reported as `packaged` at `agentic-serving/serving.yaml` (global
  holds the provisioning templates only).
- `GET /api/models` → six router models, all `unloaded`.
- A bare prose `POST /v1/chat/completions` was echoed in 2 ms (the
  session-start contract; not a serving turn). An OpenCode-shaped turn
  (system prompt + tools from the 2026-07-13 capture, one user message
  "Create hello.py that prints hello world.") → `200` in 1.1 s,
  `finish_reason: tool_calls`, one `glob` call (the discovery round);
  `<state>/llm-orc/.serve-trace/turns.jsonl` has one line; `artifacts/serving/`
  created under the state dir.
- `research-dossier.yaml` copied into `$XDG_CONFIG_HOME/llm-orc/ensembles/`
  (global tier); `POST /api/ensembles/research-dossier/execute` with "the 1783
  Laki eruption" → `200` in 86.7 s, `status: success`, `has_errors: false`:
  decomposer (qwen3-8b) → five `searcher[i]` fan-out instances of the
  packaged `agentic-serving/web-searcher` child, each `completed` → compiler
  returned a markdown dossier as `deliverable`. Artifact at
  `<state>/llm-orc/artifacts/research-dossier/20260929-141512-617/{execution.json,execution.md}`
  plus `latest`.
- `find <venv> -path '*serving_project*' -newer <marker>` → nothing: no write
  under the packaged tier. cwd empty throughout. SIGTERM to the serve stopped
  the router; ports free.
- **Real client, after the final-review fix (`1913bddf`).** Wheel rebuilt
  (162 ok), installed with `pip install --no-compile` (0 `.pyc` under
  `serving_project/`, the brew shape), served from a second empty dir.
  `opencode run --format json -m llm-orc-live/agentic-tier-cheap-general
  "Create hello.py that prints hello world."` (OpenCode 1.18.31, a temp
  workspace whose `opencode.json` points the provider at `:8766`)
  bootstrapped and drove three `POST /v1/chat/completions`: the `glob`
  discovery round, the build round, then an honest refusal ("Another
  round needed: tests did not pass"); no `hello.py` written. That is a T1
  refusal of the kind the llama-server gate row already measures, not an
  Arc 2 failure: the packaged serving ensemble ran end to end from the
  wheel. After the run: 0 `__pycache__` entries and nothing newer than the
  start marker under `serving_project/`; 228 `.pyc` under
  `<state>/llm-orc/pycache/`; cwd empty; SIGTERM stopped serve and router.

### Arc 3: transitive preflight (Sonnet, ~1-2 days, after S2)

- `check_ensemble_runnable` (`services/handlers/provider_handler.py` ~165)
  today marks script / ensemble / loop / dispatch agents available without
  checking them. Make it walk the closure and classify every dependency with
  the statuses above, each with a `resolve` hint (`ship`, `bind`, `pull`,
  `add credentials`).
- Model presence comes from the router (`/api/models`); `pullable` requires a
  profile with a model source (`hf_repo`). Newly shipped profiles follow S2's
  answer.
- Pins over real fixtures: a closure with one of each status gets exactly
  that report; a child ensemble's missing script surfaces at the top.

**Arc 3 re-cut (2026-09-29), from the merged Arc 2 shape, S2 and a
router probe.** The card above stands; these rulings settle its open
points. Implementation plan: `docs/plans/2026-09-29-remote-delegation-arc3.md`.

1. **The router answers "downloaded" itself; no second source of truth.**
   Probe (laptop, llama-server 9850, a scratch router on `:8791` fed the
   checkout's rendered preset plus one section
   `[not-downloaded-probe] hf-repo = unsloth/Qwen3-4B-GGUF:Q4_K_M` whose
   GGUF is not on disk): `GET /models` lists every preset section
   including the undownloaded one (`status.value: unloaded`,
   `source: preset`), plus `default`, plus one entry per cached GGUF
   whose `id` is the `hf_repo` string verbatim
   (`unsloth/Qwen3-8B-GGUF:Q4_K_M`, `source: cache`, `can_remove: true`);
   `llama-server --cache-list` agrees (3 entries). So one listing decides
   three things: what the router can route (preset ids, no `/`), what is
   downloaded (cache ids, contain `/`, not `default`), and load state.
   The cache entries are built when the router starts; a listing's
   `status.value` is live. The live row below saw it: after a pull in a
   running serve the model read `loaded` while no cache entry for its
   source appeared until a restart. So a model loaded now is downloaded
   too, and `inventory()` reports `loaded` (preset ids whose status
   reads `loaded`) next to `cached`.
   The cache id shape (`/` in the id) is the signal, not the `source`
   field: the id shape is what the 2026-09-16 e2e and S2 both observed on
   this binary and the mini's, `source` was first seen today.
   `LlamaServerClient` gains `inventory()` (one GET; preset models and
   cached sources), and the llama-server provider status carries
   `cached: [...]` next to `models`.
2. **Statuses are a closed set of eleven.** The card's seven, plus S2's
   `needs_restart`, plus three the code read forced: `model_unavailable`
   (an OpenAI-compatible endpoint that does not list the model; nothing
   to pull), `provider_unavailable` (router or endpoint unreachable, or a
   provider llm-orc does not know, such as the library templates'
   `ollama`), and `dynamic` (a `${...}` dispatch target, resolved at run
   time; issue #94: not followable statically). Each status maps to one
   `resolve` hint: `ready`/`dynamic` → `none`; `pullable` → `pull`;
   `needs_restart` → `restart`; `missing_profile`, `model_unavailable` →
   `bind`; `missing_model_source` → `add_source`; `needs_credentials` →
   `add_credentials`; `missing_script`, `missing_ensemble` → `ship`;
   `provider_unavailable` → `start_provider`.
3. **Classification of a llama-server profile's model `M` with source `R`
   (`hf_repo`), from ruling 1's listing:** router unreachable →
   `provider_unavailable`; `M` listed and (`R` cached or `M` loaded
   now) → `ready` (the router loads on demand; a loaded model is
   downloaded even before its cache entry appears, ruling 1); `M` listed,
   `R` not cached and `M` not loaded → `pullable` (hint names
   `POST /api/models/M/pull`); `M`
   listed and the profile has no `hf_repo` → `ready` (the router lists
   it, so it will serve it; download state is unknowable without a
   source); `M` not listed and `R` present → `needs_restart` (the router
   scanned its preset at start, S2; the serve renders the profile in on
   restart, cost 1.5 to 2.2 s plus a hard cut of in-flight completions);
   `M` not listed and no `hf_repo` → `missing_model_source` (the same
   condition `render_preset` reports as `missing_source`). Cloud
   providers (`anthropic-api`, `google-gemini`): configured in
   `CredentialStorage` → `ready`, else `needs_credentials`; there is no
   model list to check, as today. OpenAI-compatible: endpoint down →
   `provider_unavailable`, model absent → `model_unavailable`, else
   `ready`. A restart is reported, never performed: Arc 3 is a read.
4. **The closure is walked from typed agent configs, depth-first in
   agent order, deduplicated on (kind, name), first sighting wins.**
   Kinds: `ensemble` (`ensemble:`, `loop.body`, a literal `dispatch:`),
   `dispatch` (a templated target; always `dynamic`), `script`
   (`script:`), `profile` (`model_profile` and every profile-level
   `fallback_model_profile` hop of ITS chain, transitively; the
   agent-level `fallback_model_profile` is one hop whose own chain is not
   followed, mirroring `model_factory._reachable_provider_options`; a
   missing hop ends a chain as `missing_profile`), and `model`
   (an agent's inline `model` + `provider`, Invariant 3). Every
   dependency carries `via`: the frames `"<ensemble>.<agent>"` from the
   root to the agent that names it, so a child ensemble's missing script
   is reported at the top with the path that reaches it. A child that
   does not resolve is `missing_ensemble` and its subtree is not walked
   (nothing to walk). Children resolve through the executor's own
   search-dir list and finder (`child_ensemble_search_dirs` and
   `_find_ensemble_in_dirs`: `<dir>/<name>.yaml` by filename), the root
   through the API's lookup; visited-state, ownership and `via` frames
   are keyed by the reference string, never by the `name:` field.
   Profiles are classified against
   `ConfigurationManager.get_model_profiles()` (runtime tiers, library
   excluded), the same source the run and the router preset use.
   Cycles cannot load (Invariant 5) but the walker keeps a visited set
   anyway; a walker that trusts the loader is one refactor from a hang.
   A new module owns this (`core/config/closure.py`) because Arc 5
   builds the same closure on the laptop to ship it.
5. **Scripts** are checked with the executor's own resolver
   (`ScriptResolver(project_dir=<the service's project path>)`, the
   value `ExecutorFactory.create_root_executor` receives) and its
   `resolve_and_classify`: a path-syntax or absolute reference that
   raises `ScriptNotFoundError` is `missing_script`; a bare name is
   inline content and `ready`. Anything else would make preflight
   disagree with the run.
6. **`runnable` means every dependency is `ready` or `dynamic`.** A
   `pullable` model makes the ensemble not runnable: the first request
   would block on a multi-gigabyte download, and the point of preflight
   is that the caller decides that (`pull: true` in Arc 4), not the
   engine. The existing `agents` list stays for the web UI and the MCP
   docstring's promise, each agent's coarse status derived from its own
   dependencies: `available` when all are `ready`/`dynamic`; otherwise
   the first unmet dependency's status mapped `missing_profile` →
   `missing_profile`; `provider_unavailable`, `needs_credentials` →
   `provider_unavailable`; `pullable`, `needs_restart`,
   `missing_model_source`, `model_unavailable` → `model_unavailable`;
   `missing_script`, `missing_ensemble` → `dependency_unmet` (new, for
   the ensemble/script/loop/dispatch agents that used to be `available`
   unconditionally). `alternatives` keeps today's meaning.
7. **Report shape.** The existing result gains `dependencies`, a list
   of `{kind, name, via, status, resolve, detail}` (and `provider` on
   profile/model entries); no new endpoint, no new MCP tool. Arc 4's
   `not_equipped` error carries this same list. `detail` is one sentence
   naming the observed fact (the model listed, the source not cached,
   the path searched), not advice.
8. **Out of scope, on the record:** acting on `needs_restart` (a
   supervised router restart endpoint is an Arc 4+ question, gated on
   the in-flight hard cut S2 measured); the web frontend's
   `RunnableStatus` type (its `model_not_found` already drifts from the
   API's `model_unavailable`; pre-existing); `render_preset`'s
   `missing_source` reporting (unchanged, consistent with ruling 3);
   promotion readiness (reads provider status directly, untouched).

**Arc 3 live row (2026-09-29).** Laptop, branch `feat/transitive-preflight` @
`4607ae3f`, llama-server 9850 (`4f31eedb0`), GGUFs from the Hugging Face hub
cache (`llama-server --cache-list`: Qwen3-0.6B, Qwen3-8B, nomic-embed-text).
Serve started from a temp project dir with `XDG_CONFIG_HOME` and
`XDG_STATE_HOME` pointed at fresh temp dirs: `llm-orc serve --port 8766
--backend-port 8790 --models-max 1`. The project holds `probe.yaml` (five
agents: `local-qwen3-1.7b`, a project profile `probe-qwen3-4b` with `hf_repo`,
`no-such-profile`, the packaged child `agentic-serving/web-searcher`, and
`scripts/not_here.py`) and a `config.yaml` defining `probe-qwen3-4b`.
`research-dossier.yaml` was copied into the global tier.

- `GET /api/models` at start lists `deepseek-r1-8b`, `nomic-embed-text`,
  `qwen3-0.6b`, `qwen3-1.7b`, `qwen3-14b`, `qwen3-4b`, `qwen3-8b`, all
  `unloaded`; `probe-qwen3-4b` was rendered into the preset at start.
- `GET /api/ensembles/probe/runnable` → `runnable False`:
  `ready ensemble probe`; `pullable profile local-qwen3-1.7b
  ['probe.small'] -> pull`; `pullable profile probe-qwen3-4b ['probe.fresh']
  -> pull`; `missing_profile profile no-such-profile ['probe.ghost'] -> bind`;
  `ready ensemble agentic-serving/web-searcher ['probe.search']`; `ready
  script scripts/agentic_serving/web_searcher.py ['probe.search',
  'web-searcher.searcher']`; `missing_script script scripts/not_here.py
  ['probe.gone'] -> ship`. The project profile was `pullable`, not
  `needs_restart`, as ruling 3 requires.
- `GET /api/ensembles/research-dossier/runnable` → `runnable True`: the
  ensemble, `agentic-tier-cheap-general` (`research-dossier.decomposer`), the
  packaged `agentic-serving/web-searcher` child and its script, all `ready`.
- Profile added after start (`late-qwen3-1.7b-alias`, model `qwen3-1.7b-late`,
  `hf_repo` set) and `late.yaml` written with the serve still up:
  `late-qwen3-1.7b-alias` → `needs_restart` (`resolve: restart`). After
  SIGTERM and a fresh start (router stopped, ports free): `pullable`
  (`resolve: pull`).
- `POST /api/models/qwen3-1.7b/pull` → `{"name":"qwen3-1.7b","status":"loaded"}`
  in 2 min 21 s wall clock (about 1 GB). `llama-server --cache-list` then
  listed `unsloth/Qwen3-1.7B-GGUF:Q4_K_M`, but the same serve still reported
  `local-qwen3-1.7b` as `pullable`: the router's `/v1/models` had no
  `unsloth/Qwen3-1.7B-GGUF:Q4_K_M` cache entry (the raw entries were the
  0.6B, 8B and nomic files only). The router scans its cache at start, so a
  download made while it runs is invisible to `cached`.
- After one more serve restart, the router listed the 1.7B cache entry and
  `probe` reported `local-qwen3-1.7b` `ready`, `probe-qwen3-4b` `pullable`,
  `no-such-profile` `missing_profile`, `runnable False`.
- The brief expected `ready` immediately after the pull. Observed: `ready`
  only after a router restart. The report matches the router's view
  (the downloaded file is not listed), so a pull does not flip `pullable`
  to `ready` in a running serve. Arc 4's pull-then-run flow needs to account
  for it.
- Serves stopped by PID; ports 8766 and 8790 free afterwards.
- Fix wave (code change, not re-run): the router inventory now reports
  `loaded` from each listing entry's live `status.value`, and a listed
  model that is loaded reads `ready`, so a pulled model reads `ready`
  in the same serve without a restart (until the router evicts it).

### Arc 4: one-run injection (Sonnet, ~2-3 days, after Arc 3)

Request shape, identical on REST (`POST /api/ensembles/execute` body), MCP
(`invoke` args) and CLI:

```yaml
ensemble: {name, agents, ...}        # inline definition; or ensemble_name
ensembles: {name: definition}        # inline child ensembles it references
profiles:  {name: definition}        # inline profiles, run-scoped
scripts:   {relative/path.py: source}  # run-scoped, resolvable by that path
bind:      {missing_profile: remote_profile}
pull:      false                     # allow model downloads when true
input:     "..."
```

- Everything injected is run-scoped: materialized in a per-run temp dir added
  as the highest-priority layer for that execution only, deleted after.
  Nothing persists; concurrent runs never see each other's injections.
- Before any agent runs, preflight the closure with the injections applied.
  Unmet dependencies fail the call with `status: "error"`,
  `error.kind: "not_equipped"`, and the Arc 3 report (each dependency, its
  status, its resolve hint). No partial execution.
- `bind` entries apply only to profile names that are missing or explicitly
  listed; each binding is recorded in the result metadata. Bindings never
  come from the engine.
- Injected scripts get the same `_helpers` import behavior as shipped
  scripts (they run with the run-scoped scripts dir on their import path).
- Docs: `docs/serving.md` gets the trust note (decision 4) and the request
  shape.
- Pins: an inline ensemble runs and leaves nothing on disk; a missing profile
  fails with `not_equipped` before any model call; the same call with `bind`
  runs and records the binding; an injected script importing `_helpers`
  runs; two concurrent injected runs with the same script path stay
  isolated.

**Arc 4 re-cut (2026-10-01), from the merged Arc 3 shape, a code read of
every run-time resolution site, and two router probes.** The card above
stands except where a ruling says otherwise. Implementation plan:
`docs/plans/2026-10-01-remote-delegation-arc4.md`.

Code read, the facts the rulings rest on. A run resolves through two
values the executor is built with, `project_dir` and a
`ConfigurationManager`: profiles in `LlmAgentRunner` and `ModelFactory`
(`get_model_profiles`, `get_model_profile`, `resolve_model_profile`),
children in `EnsembleExecutor._resolve_ensemble_reference`
(`child_ensemble_search_dirs` + `_find_ensemble_in_dirs`), scripts in
`ScriptAgentRunner` and `ScriptAgent` (`ScriptResolver(project_dir=...)`).
Child executors share the parent's config manager and model factory.
`OrchestraService._get_executor` already builds a fresh root executor per
invocation. Three entry paths look the root up and run it, each with its
own copy of the lookup: `ExecutionHandler.invoke` (REST, the MCP dict
dispatch), `execute_streaming` (the FastMCP `invoke` tool) and
`invoke_streaming`. A script runs as `python <path>`, so its import path
starts at its own directory; that is how the serving scripts import
`_helpers`. `POST /api/models/{name}/pull` is an `async def` that polls
with `time.sleep`, so it blocks the event loop for the whole download
(#199).

Probes (2026-10-01, a scratch router on `:8791`, llama-server 9850, plus
the mini's router read over ssh):

- Every preset entry in `GET /models` carries `status.args`, the argv the
  router would spawn, with `--hf-repo <source>` and `--ctx-size <n>`. Same
  on the mini's binary (six preset models, each with its source). The
  router states which file a model name serves.
- A load of a source that does not exist (`[bogus-probe] hf-repo =
  nobody-xyz/does-not-exist-GGUF:Q4_K_M`): `POST /models/load` answers
  200 `{"success":true}`; within a second the entry reads `{"value":
  "unloaded", "exit_code": 1, "failed": true}`. A good load reads
  `loading` then `loaded`. With `--models-max 1`, the failed load still
  evicted the model that was resident.

1. **One preparation step in front of every service run.** The three
   entry paths share one function that validates the request,
   materializes the run layer, gates, builds the executor and cleans up.
   It applies to a named ensemble with no injections too: `pull` defaults
   to false for `ensemble_name` as for `ensemble`, and Arc 3 ruling 6
   already said a download is the caller's decision. This is a breaking
   change for callers: an ensemble that is not `runnable` no longer starts
   over REST or MCP, it answers `not_equipped`. That includes a ready
   primary with an unmet fallback hop (Arc 3's conservative ruling; `bind`
   resolves it). `/v1/chat/completions` builds its own executor and is
   untouched. The local CLI is untouched (Arc 5, #191).
2. **The run layer is a directory shaped like every other tier, carried
   by a config manager view.** `<state dir>/runs/<id>/` holds
   `ensembles/<name>.yaml`, `profiles/*.yaml` and each script at its key.
   `ConfigurationManager.with_run_layer(dir)` returns a copy with its own
   profile cache whose tier lists put that directory highest: first in
   `_tier_dirs`, last in `_profile_tiers`, first in
   `child_ensemble_search_dirs`. The run's executor is built on the view;
   children inherit it. Isolation is by object. Nothing global changes:
   no environment variable, no cwd, no shared cache. Two concurrent runs
   hold two views over two directories.
3. **`bind` is materialized, never rewritten.** `bind: {a: b}` writes a
   run-layer profile named `a` that carries `b`'s definition, `b` looked
   up in the view after the inline profiles, one hop (a bind target is
   never itself rebound). The closure walk and the model load then read
   the same profile map, which answers the "walk or load" question:
   neither rewrites a name. A binding applies whether or not the host has
   its own `a`. If `b` is not a profile here the call fails
   `not_equipped` with a row for `b` (`missing_profile`, `via:
   ["bind:a"]`), also when the host has an `a`, because running on the
   host's `a` would ignore what the caller asked for. A bind key that no
   profile dependency in the static closure names is `invalid_request`: a
   misspelled key must not run on the host's profile of the intended
   name. A dynamic child's profile is re-seated by shipping an inline
   profile instead. Applied bindings are returned as `bindings: {a: b}`.
4. **`pull: true` waits for `loaded` and trusts its own observation.**
   Each `pullable` dependency is pulled with the function the pull
   endpoint uses (extracted; both callers run it off the event loop). The
   wait ends when the router reports anything other than `loading`. Only
   `loaded` resolves the dependency; any other status leaves it
   `pullable` with the observed status in `detail`, and the call fails
   `not_equipped` (the failed-load probe above). The gate does not
   re-read the listing after a pull: with `--models-max 1` a second pull
   evicts the first, which then reads `pullable` again though its file is
   on disk (Arc 3's known limit). Pulled model names are returned as
   `pulled: [...]`. Without `pull`, a `pullable` dependency blocks.
5. **`needs_restart` is never resolved inside a run.** A restart cuts the
   in-flight completions of every other run (S2), and a run-scoped
   profile cannot be in the preset a restart renders without persisting
   it. The call fails `not_equipped`; a new local model is served by
   persisting its profile (`scope: global`) and restarting the serve.
   Inline profiles therefore cover: a role bound to a model the router
   already lists (with the caller's options and fallback chain), cloud
   providers the host has credentials for, and OpenAI-compatible
   endpoints.
6. **Arc 3 ruling 3 gains one case: a listed model whose router source
   differs from the profile's `hf_repo` is `needs_restart`.** The router
   serves a model name from the source in its own preset, whatever a
   profile written later says. Without this an inline profile that reuses
   a host model name with another source is classified against the wrong
   file, runs the host's file and reports success.
   `LlamaServerClient.inventory()` gains `sources` (model id to its
   `--hf-repo` argument, from the probe above) and the provider status
   carries it. Persisted profiles edited after the router started get
   the same check. No new status. Context size is in the same args and is
   not compared in this arc (on #196).
7. **Scripts.** A script with key `K` is written to `<run dir>/K` and the
   resolver's search paths start with `<run dir>/scripts`, `<run dir>`,
   the two entries every tier has. `ScriptResolver` takes the run
   directory next to `project_dir`; the script runner, the script agent
   and the gate pass the same two values. An injected script imports from
   its own directory in the run layer and nowhere else: helpers ship
   beside it. There is no fall-through to a host helper at the same
   relative path, since a caller's script running against the host's
   helper is a version mix that would run and report success. This
   replaces the card's "run-scoped scripts dir on their import path".
   Preflight does not see imports; a missing helper fails the script at
   run time. Bytecode for a run with a layer goes under the run directory
   (the state dir's `pycache` mirrors absolute paths and would keep
   files). The script cache is off for a run with a layer (its identity
   would name a deleted path).
8. **Nothing persists.** The run directory is removed in a `finally`
   that covers success, refusal, an exception and cancellation. A run
   whose root is inline saves no artifact: there is no installed ensemble
   to file it under, and the caller holds the result. A named root keeps
   its artifact, with or without injections. The pin is a tree snapshot
   of the state dir, the global config dir and the project dir before and
   after.
9. **Request validation.** Exactly one of `ensemble` and
   `ensemble_name`. Ensemble names and script keys are relative paths
   with no `..`, no leading `/`, no backslash and no empty segment.
   Inline ensembles must load through `EnsembleLoader` (the root with
   the run's search dirs, so Invariant 5 holds). A profile name defined
   twice in one request (inline and as a bind key) is rejected. The REST
   body forbids unknown keys. All of these answer `invalid_request`
   and leave nothing on disk.
10. **Error shape.** `{status: "error", has_errors: true, results: {},
    deliverable: null, error: {kind, message, dependencies}}` with `kind`
    one of `not_equipped`, `invalid_request`. HTTP 200 on REST, as for
    every run outcome since v0.21.0. `dependencies` is the Arc 3 report
    computed over the run's view: the gate is `check_ensemble_runnable`'s
    own function called with the run's config manager and project dir,
    and provider status is computed over the run's profiles so an inline
    profile's OpenAI-compatible endpoint is probed.
11. **Surfaces.** REST: `POST /api/ensembles/execute` takes the full
    shape; `POST /api/ensembles/{name}/execute` takes the same body
    without a root. MCP: `invoke` gains the fields, `ensemble_name`
    becomes optional, the input argument keeps its name `input_data`.
12. **On the record, not in this arc:** credentials are not injectable
    (`needs_credentials` blocks); preflight does not follow script
    imports; the serve remembering its own pulls (#196); a run directory
    left by a killed process is inert and is not swept, because a stdio
    MCP server and a serve may share one state dir; context-size
    mismatch (ruling 6).

**Arc 4 live row (2026-10-01).** Laptop, branch `feat/one-run-injection`
@ `64bfbae0`, llama-server 9850, GGUFs from the Hugging Face hub cache.
Serve started from an empty directory with `XDG_CONFIG_HOME` and
`XDG_STATE_HOME` pointed at fresh temp dirs: `llm-orc serve --port 8766
--backend-port 8790 --models-max 1`. The temp global tier held one
profile written before start, `probe-smol` (model `smollm2-135m`,
`hf_repo: unsloth/SmolLM2-135M-Instruct-GGUF:Q4_K_M`, not downloaded).
All requests are `POST /api/ensembles/execute` unless noted. The inline
root has two agents: a script `scripts/live/prep.py` that imports a
sibling `_helpers`, both shipped in `scripts`, and an LLM agent that
depends on it.

- Inline root on `local-qwen3-0.6b`: `success` in 3.0 s, the script
  answered through its helper, the model answered. A recursive listing
  (paths and sizes) of the cwd, the config dir and the state dir was
  identical before and after; `runs/` empty.
- The same root on `laptop-fast`, a profile this host lacks: `error`,
  `kind: not_equipped`, one unmet row (`profile laptop-fast
  missing_profile`, `via: ["live-inline.answer"]`), `results` empty, in
  under 0.1 s.
- The same call with `bind: {laptop-fast: local-qwen3-0.6b}`: `success`,
  `bindings` returned.
- A bind onto a target this host lacks, for a profile the host has:
  `not_equipped`, the row names the target with
  `via: ["bind:local-qwen3-0.6b"]`. A misspelled bind key:
  `invalid_request`, "bind key 'laptop-fsat' names no profile in the
  closure".
- The root on `probe-smol`: `not_equipped`, `pullable`. With
  `pull: true`: `success` in 20.7 s (the download and the load),
  `pulled: ["smollm2-135m"]`.
- An inline profile `foreign` (model `qwen3-0.6b`,
  `hf_repo: bartowski/Qwen3-0.6B-GGUF:Q4_K_M`): `not_equipped`,
  `needs_restart`, detail "model 'qwen3-0.6b' listed; the router serves
  unsloth/Qwen3-0.6B-GGUF:Q4_K_M, the profile names
  bartowski/Qwen3-0.6B-GGUF:Q4_K_M".
- Two concurrent runs with one ensemble name and one script path
  (`scripts/shared/who.py`) and different content: each returned its own
  content; `runs/` held two directories mid-run and none after.
- A named ensemble the host cannot run (`validate-ollama`, whose profile
  is missing): `not_equipped` on `POST /api/ensembles/validate-ollama/execute`.
  A misspelled body key on that route: 422.
- A named root created in the global tier with a profile name the host
  lacks: refused; with `bind` on `POST /api/ensembles/named-live/execute`:
  `success`, and its artifact is under the state dir's `artifacts/`. No
  inline root left an artifact.
- Over `/mcp`: the `invoke` tool lists `bind`, `ensemble`,
  `ensemble_name`, `ensembles`, `input_data`, `profiles`, `pull`,
  `scripts`, with only `input_data` required. An inline root with an
  injected script ran; adding an agent on a missing profile returned the
  `not_equipped` envelope with the dependency list.
- At the end: no file under the cwd, `runs/` empty, serve stopped by
  PID, ports 8766 and 8790 free.

**Review rounds and the re-run (2026-10-02).** A whole-branch review and
three scoped re-reviews followed, each author-independent, each
CHANGES REQUIRED until the last. What they changed in the request
contract, beyond the rulings above:

- Names in one request must be distinct ignoring case and Unicode
  normalization (the host's disk folds them); a root `Kid` with a child
  `kid` had run the child as the root.
- A script key needs path syntax (a `/` or a script extension), and two
  keys one reference reaches (`x.py` and `scripts/x.py`, or the
  hyphen/underscore pair) are `invalid_request`. A bare key `date` had
  run the host's `date`.
- `pull` happens only when every unmet dependency is `pullable`, and
  stops at the first pull that does not end `loaded`.
- A request with no root, and a bind write that fails, are
  `invalid_request`, not a 500.
- The run's gate probes only the OpenAI-compatible endpoints its closure
  uses, together; the read endpoint still probes the host's.
- A refusal names the reference and the problem, never a host path.
- A cancelled run's script and its process group are killed before the
  run directory is removed. Every script agent's subprocess now leads
  its own group; a timeout kills the group; only the worker thread
  reaps the child. Lesson, binding: `Popen.communicate` cannot resume
  sending input after a timeout, so it is called once. A fix wave that
  sliced it stalled every script with an input past the pipe buffer
  until its timeout, and the suite stayed green; the wall time (one test
  at 302 s) was the only signal.

The live row was re-run at `67262bf1` with the same setup (the pull
model is now `smollm2-135m-b` from `bartowski/SmolLM2-135M-Instruct-GGUF`,
since the first run had downloaded the other). Every row above repeated
with the same outcome. Added rows:

- `Kid`/`kid`, `x.py` + `scripts/x.py`, `X.py` + `scripts/x.py`, a bare
  key `date`, and a body with no root: each `invalid_request` with a
  message naming the keys, `runs/` empty.
- `pull: true` with one `pullable` profile and one missing profile:
  `not_equipped` in under 0.1 s listing both; the pullable model was
  still `unloaded`, no download started.
- An injected script that sleeps 0.7 s before reading a 300 KB input:
  `success` in 0.8 s, the full length read.
- A client that disconnects mid-run (`curl` killed while an injected
  script slept 20 s): the run is NOT cancelled. The script ran to its
  end, and `runs/` was empty once it finished. A plain REST request is
  not tied to its connection; recorded on #191.

**Released as v0.23.0 and the remote live row (2026-10-02).** Pushed at
0.22.0 first (CI caught one new test on ubuntu: a script agent mirrors
its input into the environment and Linux caps one variable at 128 KB, so
the 200 KB pin became 100 KB), then the release commit, tag, GitHub
release, PyPI and the tap formula. Verified by a clean install from
PyPI: 162 packaged files, a serve from an empty directory refused a
missing profile and ran the same request with `bind`, nothing left
behind. The remote host took the release with one command over ssh
(brew upgrade, same unit, restart, health at 0.23.0 in under a minute).

Rows against the remote serve, driven from the laptop over its https
URL, file listings of its config, state and working directories taken
over ssh:

- Inline root with an injected script and helper on `local-qwen3-0.6b`
  (CPU-only host): `success` in 8.0 s; 15 files before and after, none
  changed; `runs/` empty.
- The same root on a profile the remote lacks: `not_equipped` in 0.1 s.
  With `bind`: `success` in 3.7 s, `bindings` returned.
- An inline profile naming another source for a listed model:
  `not_equipped`, `needs_restart`, the detail names both sources.
- Two concurrent runs with one script path and different content: each
  returned its own; `runs/` empty after.
- `x.py` with `scripts/x.py`, and a body with no root: `invalid_request`.
  A misspelled key on a named route: 422.
- The MCP `invoke` tool over `/mcp`: an inline root with an injected
  script ran in 0.3 s.

### Arc 5: CLI as the remote client (Sonnet, ~1-2 days, after Arc 4)

- `llm-orc invoke <ensemble> --remote <url> [--bind a=b] [--pull]
  [--persist global]`: the CLI resolves the closure locally (ensemble, child
  ensembles, scripts, profiles referenced), sends it as one injection
  request, and prints the result with the caller contract (`status`,
  `has_errors`, `deliverable`, non-zero exit on error). On `not_equipped` it
  prints the report as a readable table.
- Same closure builder exposed to MCP clients as a tool, so an agent on the
  laptop can delegate a local ensemble to the mini in one call (#191).
- Live: from the laptop, `llm-orc invoke research-dossier --remote
  https://llm-orc.remote.example "<topic>"`; then with a profile the mini
  lacks, see `not_equipped`; then `--bind` it and run.

**Arc 5 re-cut (2026-10-02), from the merged Arc 4 shape, a code read of
the CLI's run path and every lookup a shipper needs, and probes against
the remote host.** The card above stands except where a ruling says
otherwise. Implementation plan:
`docs/plans/2026-10-02-remote-delegation-arc5.md`.

Code read, the facts the rulings rest on.

- `llm-orc invoke` (`cli_commands.invoke_ensemble`) finds the root with
  `service.find_ensemble_by_name` (or in `--config-dir`), takes
  `service._get_executor()` and runs it. No preparation step, no gate,
  no `bind`, no `pull`.
- `walk_closure` returns reference strings and `found`, never a file,
  and `EnsembleConfig` does not record the file it was loaded from. The
  root is found by its `name:` field (`EnsembleLoader.find_ensemble`, an
  rglob); a child by file name (`_find_ensemble_in_dirs`).
- A request's `ensembles` key becomes `ensembles/<key>.yaml` in the run
  layer and the child finder resolves by file name, so a child shipped
  under its reference string resolves on the remote as it did locally.
  A script written at its reference string resolves too: `<run>/scripts`
  is tried with the reference, its underscore form and its form without
  `scripts/`, then `<run>`.
- A script reference has four forms: a relative path (searched), an
  absolute path, a bare name that is a file in the cwd, and bare inline
  content.
- The REST and MCP result carries `results`, `deliverable`, `status`,
  `has_errors`, `raw_output`, `bindings`, `pulled`. The executor's
  `metadata` (usage, durations), which the CLI's text and rich displays
  read, is dropped.
- `requests` is already a dependency; a client needs nothing new.
- CRUD cannot install a closure: `create_script` writes a template,
  `create_ensemble` keeps name, description and agents and refuses an
  existing name, `update_ensemble` never writes (#191).
- A named root is found by two copies of one loop over
  `get_ensembles_dirs()`: `OrchestraService.find_ensemble_by_name` (the
  CLI, the runnable check, validate, promotion, CRUD templates, the
  streaming path) and `ExecutionHandler._lookup_in_tiers` (`invoke` and
  `invoke_streaming`). The run layer is one attribute on a config
  manager copy; `check_ensemble_runnable`'s function already takes a
  view.
- Global resolves before packaged, for ensembles and for scripts. A file
  installed in a host's global tier shadows the packaged file of that
  name for every ensemble on the host, `/v1` included.
- The packaged scripts: 45 files, 18 with static sibling imports
  (`_helpers` in all 18, plus `chain_plan` and `accept_gather` once
  each), 48 script agents across the ensembles reference those 18. One
  script runs a sibling by path (`accept_executor.py`, through
  `Path(__file__).with_name`). Three read their tier's
  `ensembles/agentic-serving` directory as a catalog. The largest
  directory is 33 files, 397 KB.

Probes (2026-10-02, the remote host at 0.23.0 through its https proxy,
`POST /api/ensembles/execute`, an inline root with one injected script):

- A 2 MB and a 12 MB script body: 200 in 1.7 s and 3.3 s, the script
  ran. The proxy puts no practical cap on a shipped closure.
- A script that sleeps 75 s, then one that sleeps 200 s: 200 after
  75.4 s and 200.4 s. A plain POST with no bytes flowing survives at
  least 200 s through the proxy. Not measured: the 8 to 15 minutes a
  model run takes on that host; the live row covers it.

1. **The local CLI runs through the preparation step (practitioner,
   2026-10-02).** `invoke` builds a run request and hands it to the same
   function REST and MCP use. `--bind a=b` (repeatable) and `--pull`
   work on every invoke; `--remote` changes only where the run happens.
   Breaking for local callers, as 0.23.0 was for REST and MCP: an
   ensemble that is not `runnable` is refused before any agent starts,
   and a `pullable` model blocks without `--pull`. `--config-dir` stays
   as the directory the root is looked up in.
2. **Named remotes live in the user's global config and nowhere else.**
   `remotes: {<name>: {url: ...}}` in the global `config.yaml`.
   `--remote` takes a name or a URL (a value with `://` is a URL). An
   unknown name is an error that lists the known names. A `remotes` key
   in a project config is not read, so no URL needs to live in a repo.
3. **Transport is one plain POST, and only a run result is accepted.**
   `requests.post(<url>/api/ensembles/execute)`, a connect timeout, no
   read timeout. The response is a result only if it is HTTP 200 and a
   JSON object with `status` of `success` or `error` and a boolean
   `has_errors`. Anything else is an error that names the remote and
   the status code, exit 1: a 422 from an older serve that forbids a
   new key, a 404, and the web UI's HTML answered with 200 for an
   unknown API path (#191).
4. **A REST run is tied to its connection.** The two execute routes
   cancel the run when the client disconnects; Arc 4's cancellation
   already kills the script group and removes the run layer. This
   closes the item deferred from Arc 4, which a CLI client turns from a
   note into a defect: Ctrl-C on a ten minute run would leave the host
   working. The live row measures it through the proxy.
5. **The result carries `metadata`, and the CLI renders a remote result
   with the functions it renders a local one.** Additive on REST and
   MCP. `--output-format json` prints the result document. A
   `not_equipped` refusal prints the dependency report as a table (kind,
   name, status, via, resolve), the same table for a local refusal.
   Exit 1 on `status: error`, whatever the kind.
6. **Ensembles ship as written, keyed by reference; profiles do not ship
   unless asked.** `load_from_file` records the path it read, and the
   shipper sends the parsed YAML of that file, not a re-serialized
   config. The local walk uses the run's own finders (Arc 3's rule).
   What does not resolve locally is not shipped, and the remote's
   preflight judges it. Profiles stay behind by default: a host binds
   roles to its own models, and the card's live row depends on that
   (a profile the remote lacks answers `not_equipped`, then `--bind`).
   `--with-profile NAME` (repeatable) ships the local definition of one
   role as an inline profile.
7. **Scripts ship from whichever tier resolved them, at their reference
   string.** Packaged scripts and primitives included: the caller's
   ensemble runs with the caller's scripts (Arc 4 ruling 7's reasoning).
   An absolute path is refused locally, since it names nothing on
   another host. A bare name that is a file in the cwd is refused
   locally: the remote would read it as inline shell (the `date` case
   from the Arc 4 review). Inline content travels in the definition. A
   script that is not UTF-8 text is refused by name.
8. **A script declares the files it needs, in the script (practitioner,
   2026-10-02).** A comment block in the form PEP 723 reserves for
   tools: `# /// llm-orc`, TOML, one key `files` listing paths under
   the script's own directory (relative, no `..`), closed by `# ///`.
   A listed file that carries its own block is followed. Nothing is
   inferred: what is listed ships, and a script with no block ships
   alone. The listed files are closure members, so the walker, the
   remote's preflight and the shipper read one list and a missing file
   is `missing_script` before any agent runs, which closes "preflight
   does not follow script imports" from Arc 4 ruling 12. A listed file
   is looked for beside the resolved script and nowhere else (Arc 4
   ruling 7: no fall-through to a host file at the same relative
   path). A block that does not parse leaves the script unmet. No new
   status. Rejected: the list on the script agent (48 agents against 19
   scripts, and every ensemble changes when a script gains an import);
   the script's directory (over-ships, and with ruling 9 every extra
   file is persisted); static imports (a second implementation of
   Python's import resolution, Python only, with one measured miss). An
   undeclared import works locally and fails on the remote by name; a
   suite test holds the packaged scripts' static sibling imports inside
   their blocks.
9. **`--persist global`: a persisted closure is a stored run request
   (practitioner, 2026-10-02: in this arc, kept as a unit).** A request
   with `persist: global` needs an inline root with a plain name. It
   passes the same validation and gate as any run. If the gate passes,
   the request without `input`, `pull` and `persist` is written as one
   file, `<global config>/bundles/<root name>.json` (written beside,
   then renamed), and the run goes on as a named run, artifact kept. A
   gate refusal writes nothing. A later run of that name loads the
   stored request, lays the caller's own injections over it key by key,
   and takes the unchanged path: validate the union, materialize a run
   layer, gate, run, remove the layer. So the closure's children and
   scripts resolve only for that root and nothing on the host is
   shadowed; what runs by name is what ran injected; a run in flight
   holds its own copy while the bundle is replaced; and an injection
   that collides with the bundle's contents meets the validator Arc 4's
   reviews hardened. A persisted `bind` is the host's binding for a
   role; a per-run `bind` overrides it by key. Named roots resolve in
   the tiers first, as today, then in the bundles, through one lookup.
   Persist refuses a root name that a tier resolves
   (`invalid_request`, naming the tier), so a bundle is never born
   shadowed. Persisting again replaces the file; `delete_ensemble` with
   `scope: global` removes it. Listings show a bundle root with source
   `bundle`; the runnable check on a bundle name gates over the same
   materialized view; validate and promote answer that a bundle is not
   a tier ensemble. Rejected: copying entries into the global tier with
   a per-entry `written`, `identical`, `conflict` rule and `force`. It
   shadows packaged files for every ensemble on the host when forced,
   leaves no record of what an install wrote, and runs the persisted
   root against the host's copies where the injected run used the
   caller's. Also rejected: a kept run-layer directory used in place,
   since an injection over it would skip the union validation and a
   replace could pull files from under a run. Cost: a bundle root runs
   as a root, not as a child of a host ensemble; a bundle keeps its
   copies until it is persisted again. The CLI takes `--persist global`
   only with `--remote`.
10. **MCP: `invoke` gains `remote`.** With `remote`, the server builds
    the closure of `ensemble_name` from its own tiers and sends it;
    `bind`, `pull` and `persist` pass through. `remote` with an inline root is
    `invalid_request`: a client holding a definition can send it to the
    remote itself. REST does not gain `remote`; a serve is not a relay.
11. **Refused with `--remote`:** `--max-concurrent` (the request has no
    such field, so it would be dropped silently) and an ensemble with an
    interactive script (no channel carries the prompt).
12. **On the record, not in this arc:** progress events for a remote
    run (the CLI shows elapsed time only); reading a serve's access
    level before shipping (#205); candidates for a `${...}` dispatch
    (#94), which today reads `dynamic` and is left to the host.

**What the build and the review changed (2026-10-02).** The rulings
above stand except as amended here.

- Ruling 1: `llm-orc validate run` ran its ensemble on its own executor
  too. It goes through the same step and takes `--bind` and `--pull`.
- Ruling 3: the transport is an async `httpx` client (connect timeout
  10 s, no read timeout, no redirect followed, no retry), where the
  ruling said `requests`. A blocking POST in a worker thread cannot be
  cancelled, so a cancelled MCP call left the remote running. `httpx` is
  what the model clients already use.
- Ruling 5: a document is an error when `status` is not `success` or
  `has_errors` is true; the CLI had read `has_errors` alone. JSON mode
  prints the whole result document and adds `config`. A local document
  carries `raw_output` as REST's does, and so does the MCP result. A
  stored bundle prints `Bundle persisted: <name>`.
- Ruling 6: what the shipper leaves out is said. The CLI prints one
  stderr line naming each child or script that did not resolve locally;
  the MCP result carries them as `left_out`. The remote may satisfy one
  with its own copy, which the ruling allows and no longer hides.
- Ruling 7: a script's key is `scripts/` plus the tail of the locally
  resolved path, as many segments as the reference has without its
  `scripts/`. The file keeps its real name and the layer mirrors a
  tier. Before a request is returned it is materialized into a
  temporary directory and every reference is resolved there with the
  run's own `ScriptResolver`; a reference that would reach another file
  is refused. The first key rule passed the parity pin and still let
  `scripts/foo.py` reach a listed `scripts/scripts/foo.py` on the
  remote. Also refused: a reference with a `.` or `..` segment, two
  local files that meet on one key, YAML that does not survive a JSON
  round trip, a script that cannot be read.
- Ruling 8: one script name can be sighted twice with different
  outcomes (referenced directly and listed, or listed by owners in two
  tiers); an unmet sighting wins, so the row cannot read `ready`. A
  listed file is looked for beside the script's real location: Python
  imports from beside a symlink's target, not beside the link. The
  first `/// llm-orc` opening line decides the block; an unknown key or
  a block that does not close is an error; an empty block lists
  nothing.
- Ruling 9: the bundle is written once the gate passes, before any
  agent runs, so a run that then fails or is cancelled leaves it
  stored. A bundle name reads, runs and lists only the directory entry
  spelled exactly like it whose root has that name (a case-folding disk
  let `PACK` delete `pack.json`); `delete` removes the entry spelled
  exactly `<name>.json` whatever root it stores, so a stranded file can
  always be removed. `delete_ensemble` with `scope: global` removes the
  bundle only when no tier resolves the name (the lookup persist uses);
  otherwise it is the tier delete, so a flat global `.yaml` or `.yml`
  goes first and the bundle on a second call. A caller's `bind` or
  inline profile for a role replaces
  the stored definition of that role in either form. A bundle is read
  (`GET /api/ensembles/{name}`, the MCP resource) with source `bundle`.
  Not taught about bundles: `update_ensemble`, `from_template`, shell
  completion, `/v1/chat/completions`. A bundle root is not shipped
  onward with `--remote`.
- Ruling 10: the MCP tool also takes `with_profiles`. The MCP server a
  serve mounts at `/mcp` refuses `remote` and `with_profiles`
  (`invalid_request`, nothing sent): through it a client could have had
  the serve post its own ensembles and scripts to any URL. The stdio
  server keeps `remote`. A third error kind, `remote_error`, says the
  remote could not be reached or did not answer with a result;
  `invalid_request` on this path means nothing was sent.
- A REST run cancelled by a disconnect answers 499 to the closed
  connection.

**Arc 5 live row (2026-10-02).** Laptop, branch `feat/cli-remote-client`
@ `def0b958`, llama-server from the Hugging Face hub cache. The remote: a
serve started from an empty directory with `XDG_CONFIG_HOME` and
`XDG_STATE_HOME` on fresh temp dirs, `llm-orc serve --port 8766
--backend-port 8790 --models-max 1`. The caller: the CLI in a separate
temp project with its own temp XDG dirs and one `remotes` entry for that
serve. The caller's root `live-top` has a script `live/prep.py` whose
block lists `_h.py`, a hierarchical child `kids/echo` with a script
referenced as `scripts/live/who.py`, and an LLM agent on
`local-qwen3-0.6b`. File listings (paths and sizes) of the remote's
working, config and state directories were taken around the rows.

- `invoke live-top --remote remote-host`: `success` in about 3 s, the
  scripts ran from the remote's run layer through the helper, the model
  answered. JSON mode: one document on stdout with `metadata` and
  `raw_output`, nothing on stderr. The remote's tree was unchanged but
  for the key file a first run creates; `runs/` empty.
- A root on a profile the remote lacks: the table, exit 1. With
  `--bind`: `success` and the `Bindings applied` line. With
  `--with-profile`: `success`.
- A script whose listed helper is absent locally: the stderr line
  "Left to the remote (not found locally): file '_h.py' listed by script
  'live2/prep.py'", then the remote's `missing_script`, exit 1. The same
  ensemble run locally is refused by the local gate with the same row.
- A child that exists nowhere: the stderr line naming it, then
  `missing_ensemble`.
- The review's collision (`scripts/foo.py` beside a listed
  `scripts/scripts/foo.py`): the local run answers A; with `--remote`
  the ship is refused before anything is sent, naming the reference and
  the file it would reach.
- Ctrl-C (a real SIGINT) while a shipped script sleeps 60 s: the CLI
  exits 130 in 0.06 s; the script's process is gone and `runs/` is empty
  within the same 0.06 s.
- `--persist global`: `Bundle persisted: live-top`; one file under the
  remote's `bundles/` (mode 600) and the run's artifact. A run by name
  over REST: `success`, the bundle file byte-identical after it, `runs/`
  empty. The listing shows it with source `bundle`, the read answers,
  the runnable check reports six dependencies.
- An inline request on the remote that references the bundle's child
  and script: `missing_ensemble` and `missing_script`.
- Persisting a name the remote's global tier has: `invalid_request`
  naming the tier. `DELETE` of the bundle's name in capitals: not found,
  the bundle untouched.
- Persist again after a local edit: the file's hash changes and the run
  shows the new helper. `DELETE` with `scope: global`: gone, and the
  remote's config and state match the start but for artifacts.
- Over stdio MCP on the caller: `invoke` with `remote` runs there and
  returns the document; a left-out child comes back in `left_out`; an
  unreachable URL is `remote_error`. A `notifications/cancelled` for a
  call whose shipped script sleeps: the script is gone and `runs/` empty
  0.05 s later, and the server answers the next call. (Cancelling only
  the client's own task sends no notification with the Python SDK, and
  the run then goes on.)
- Through the remote serve's own `/mcp`, `invoke` with `remote`:
  `invalid_request`, "this serve does not relay".
- At the end: nothing under the remote's working directory, `runs/`
  empty, no `bundles` directory in the caller's config, the serve
  stopped by PID, ports 8766 and 8790 free.

Mid-run a mistyped command persisted the sleeping root; its pipe broke,
the serve cancelled the run, and the bundle stayed stored, which is the
behavior recorded under ruling 9 above.

**Review round (2026-10-02).** One author-independent whole-branch
review at `55c1fba1`, CHANGES REQUIRED, eight findings, six of them
demonstrated by a running test: the `scripts/` key collision that ran
another file with `success`; listed files looked for beside a symlink
where Python imports beside its target; an exit code that read
`has_errors` alone; a commit labelled refactor that made an
interactive-script text run exit 1 on success; an MCP cancel that left
the POST open; a serve that relayed through its own `/mcp`; a bundle
name acting on another spelling on a case-folding disk; ship errors
escaping as tracebacks. It also found two pins that could not fail (a
replaced bundle not reaching a run in flight, which stayed green under
the design ruling 9 rejected; the delete order, which nothing pinned).
All were fixed with the amendments above; each fix was reproduced red
first except the transport change, whose pin was shown red under a
mutant and then confirmed in the live row. The fourteen review-focus
items each have a pin through a real surface.

A scoped re-review of that fix wave at `def0b958`: CHANGES REQUIRED
again. The eight were fixed; three of the fixes had brought problems of
their own and the review found gaps the wave had not covered. What the
second wave changed:

- Ruling 9: a persist whose name matches an existing bundle's ignoring
  case and Unicode normalization, spelled differently, is
  `invalid_request` naming that bundle. The first wave had made read and
  delete exact and left write: on a case-folding disk `PACK` overwrote
  `pack.json` and neither name could then be run, listed or deleted.
  `delete` removes the entry spelled exactly `<name>.json` whatever root
  it stores.
- Rulings 6 and 7: the proof covers what is left out. A script, child
  or listed file that did not resolve locally must not resolve inside
  the materialized layer; if it would, the ship is refused. Otherwise a
  left-out `helper.py` was answered on the remote by a listed file
  shipped at `scripts/helper.py`, and the remote ran the caller's file
  where the caller's own host refused the run.
- Ruling 3: a malformed URL is `invalid_request` (the new client raised
  it outside its own error tree). A redirect is still not followed, and
  the error names its `Location`. `REQUESTS_CA_BUNDLE` is no longer
  read; `SSL_CERT_FILE` is.
- Ruling 10: the ship runs in a worker thread and the POST stays on the
  loop; after the transport change the walk and the proof (about 144 ms
  for a packaged closure) had moved onto the MCP event loop, and the
  pin for it could not fail. `llm-orc mcp serve --transport http` does
  not relay; only the stdio transport does.
- The teardown error seen twice in the arc was a race in the suite's
  artifact cleanup fixture, which walks a directory every worker
  shares. The fixture tolerates a directory that disappears.

Known limit, left as is: the shipper refuses a layout that would run,
`.llm-orc/scripts/scripts/x.py` beside `.llm-orc/scripts/x.py` with both
referenced. Its key rule gives both one key; the proof would accept two.
A loud refusal of a rare layout. No MCP remote pin goes through an MCP
wire, since the mounted server refuses `remote`; the stdio live row is
that check.

A third scoped review at `aedcb66a`: one blocker and notes. The blocker:
the client's own set-up raised outside its error tree on common
environments (a SOCKS proxy without `socksio`, a proxy with an unknown
scheme, a stale `SSL_CERT_FILE`, a port above 65535), and each reached
the CLI as a traceback and the MCP tool as a tool exception. Now the
resolver accepts only `http` and `https` with a host and a port in
range; building the client is its own step whose failure is
`invalid_request` naming the proxy and TLS variables; the POST's errors
stay `remote_error`. From the notes: a left-out reference with a `.` or
`..` segment is refused like a found one, and nothing outside the layer
counts as answering one; a redirect's `Location` is cut at 200
characters and stripped of control characters; the other-spelling
check is repeated right before the bundle is written; the MCP tool
resolves the root on the loop before the ship thread starts, so a
`set_project` cannot split root from closure; `MCPServer` does not
relay unless asked, and only the stdio command asks; the delete order
also respects a global `.yml` file; two tests that leaked a SIGINT
handler restore it. Left as is: Ctrl-C cannot interrupt a ship that is
itself stuck. The event-loop pin's longest-gap assertion was dropped
after it failed 3 of 20 times under a concurrent suite; its tick
counts alone go red for both regressions it guards.

A fourth, narrow review at `51a080e1`: one regression and notes. The
third wave had dropped the `InvalidURL` catch in favor of the resolver's
check, and `urlsplit` silently drops tabs and newlines, so a YAML
`url: |` block (which ends in a newline) crashed the run again with a
raw exception. Now the resolver refuses whitespace and control
characters by name, a query or a fragment too, and the catch is back as
a guard (`http://☃` passes the check and fails the client's IDNA step,
so the guard has a real input). From the notes: the tier-shadow check
is repeated with the spelling check right before the bundle write; a
non-200 body and a 422 `detail` are cleaned like a `Location`; a bundle
delete defers to any tier that resolves the name, with the lookup
persist uses, and the flat tier delete takes `.yml`. A tier root in a
subdirectory or under another file name still cannot be deleted by
name, and a bundle behind it stays until that file is removed. All 21
of the third wave's new pins went red under their mutants; a
whole-suite check found no leaked signal handler.

**Released as v0.24.0 and the remote rows (2026-10-02).** Pushed at
0.23.0 first (CI green on six cells and security), then the bump and
the tag; PyPI, the GitHub release, the tap formula (bumped by the
repo's own `update-homebrew.yml` action on the release, same content as
the hand bump). The remote host took it with the homelab deploy script
in about 50 s and answers `/health` 0.24.0. Rows driven from the laptop
CLI over the host's https URL, the host inspected over ssh, a temp
caller config holding one `remotes` entry:

- A root on a profile the remote lacks: `not_equipped` with the table.
  With `--bind`: `success` in 14.8 s.
- Ctrl-C while a shipped script sleeps, through the proxy: the CLI
  exits 130 in 0.28 s; the script's process on the host is gone and its
  run layer removed within 2 s. The disconnect propagates through the
  real proxy, which the laptop rows could not show.
- `--persist global`: `Bundle persisted`, one mode-600 file under the
  host's `bundles/`. A run by name over the proxy: `success`, the
  scripts ran from the host's run layer, the file byte-identical after.
  Listed with source `bundle`; a host request naming the bundle's child
  reads `missing_ensemble`; `DELETE ... ?scope=global` removes it.
- `research-dossier --remote` (the first model run past 200 s through
  the proxy): returned after 212 s, and again after 204 s, each time
  `status: error`. The shipped `web-searcher` ran on the host and
  returned results for two or three of the five fan-out queries; the
  others failed with a script exit 1, so the compiler was not run. The
  same ensemble run by name on the host, with the host's own packaged
  copies and nothing shipped, failed the same way (one of five). The
  script run by hand on the host: `ddgs` raises
  `DDGSException("No results found.")` on some queries and
  `web_searcher.py` does not catch it, though its docstring says
  backend failures come back as structured errors. A packaged-script
  defect, not a delegation one; recorded on #196. The delegation path
  itself held: 200 s and more through the proxy, the result returned,
  `runs/` and `bundles/` empty on the host afterwards.

Lesson, binding: a parity pin proves the layouts it was given. Two
rounds of key rules passed it and each was wrong on a layout it did not
hold. What closed the class was running the run's own resolver over the
request before sending it, for what ships and for what does not.

### Arc 6: remotes are discoverable (Sonnet, about a day, after Arc 5)

Asked for by the practitioner on 2026-10-02 after the first remote run
from a fresh install: a new session must be able to find the configured
remotes, see what one is missing for a local ensemble without running
it, and then run. Today `invoke --remote` needs a name the caller
already knows, nothing lists or probes remotes, and the only preflight a
remote offers is for what is installed there.

1. **List and probe.** `llm-orc remotes` and an MCP tool `list_remotes`
   answer each configured remote with its name, URL and a live
   `GET /health` probe (connect timeout 5 s): `reachable`, `version`, or
   the error observed. The probe uses the same client set-up as a run
   (`build_client`), so a proxy or TLS problem shows here first. The
   tool exists on a relaying server only; on a serve's mount it answers
   that this serve does not relay. Reading `remotes` follows ruling 2:
   the global config only.
2. **Preflight a remote without running.** `POST /api/ensembles/preflight`
   takes the full run request (the body of `/execute`) and answers the
   Arc 3 report over the run's view: `runnable`, `dependencies`, the
   `bindings` that would apply, and `left_out` is not its concern. It
   validates, materializes the layer, gates and removes the layer, and
   runs nothing: no pull (a `pullable` row stays `pullable` and says
   `pull: true` would resolve it), no agent, no artifact, no bundle
   written whatever `persist` says. A refusal that would be
   `invalid_request` on `/execute` is the same envelope here. The MCP
   `check_ensemble_runnable` gains `remote` (ships the closure as
   `invoke` does and calls the endpoint) and the CLI gains
   `--preflight` on `invoke`, local or remote: gate, print the table
   (every row, met rows included) or the JSON report, exit 0 when
   runnable and 1 when not, and run nothing. One flag, one meaning.
3. **Help.** `get_help` and the CLI's top-level help say what remotes
   are, how to configure one, and the three steps: list, preflight,
   run.
4. **Not in this arc:** reading a remote's inventory through the local
   server (its own MCP entry or REST answers that; a relay for reads is
   still a relay); remotes in a project config; credentials for a
   remote (#205).

Pins: through `CliRunner` and the real tool call, with the transport
seam on a second service: `remotes` lists the configured names with a
version for a reachable one and an error for an unreachable one, and
reads nothing from a project config; `--preflight` against the second
service answers the table with a `missing_profile` row and exits 1,
then with `--bind` answers runnable and exits 0, and the second
service's trees are unchanged and its `runs/` empty after each; the
endpoint with `persist: global` writes no bundle; with `pull: true`
pulls nothing (the router fake's load is never called). Mutants: the
probe always says reachable; the endpoint runs the executor; `persist`
honored on preflight.

**Built and reviewed (2026-10-02).** Branch `feat/remotes-discoverable`.
Shape as the card, with these particulars: the preparation step is split
into staging and judging, and `/preflight` is the judging half with the
run left out, so its verdict is `/execute`'s; one `_call_remote` does
resolve, ship, build client, post and check for a run and a preflight;
`check_ensemble_runnable` on MCP takes `remote`, `bind`, `pull`,
`with_profiles`, and those three without `remote` are `invalid_request`;
`--preflight --persist` is a usage error; a 404 or 405 from a preflight
says the remote may be older than 0.25.0. Two review rounds. The first
found one blocker, a probe that could fail the whole `remotes` list (a
health body of 200,000 `[` raised `RecursionError`; an IDNA error raised
outside the client's error tree), fixed by a per-remote catch-all, a 10 s
deadline and a 64 KB body cap, with the same catch-all on the run and
preflight paths; and notes, taken: the preflight answer is accepted only
in the shape the display prints; text mode cleans a remote's cells (JSON
mode prints the document byte for byte); a remote name cannot contain
`://`; a URL's password is masked wherever echoed; four pins made able
to fail. The second round found one leak, a password after an email-style
user name echoed in full, fixed with the refused URL masked too, and
notes taken: a run's record lines cleaned in text mode, pins for the
classified-error re-raise, JSON fidelity and the cap boundary. Nothing
from Arc 5 broke; every new pin goes red under its mutant; the timing
pins passed 20 of 20 under a concurrent suite. Live rows on the laptop
(two serves, a router): `remotes` lists a live and a dead entry;
`--preflight --remote` answers the table and exit 1 for a missing
profile, runnable and exit 0 with `--bind`, the remote's tree unchanged
and `runs/` empty; over stdio MCP, `list_remotes`,
`check_ensemble_runnable(remote=...)` with and without `bind`, and
`get_help` all answer. Against the remote host at 0.24.0: `remotes`
reads it reachable at 0.24.0 and a `--preflight` gets the 405 with the
hint. Left as is: `--input`, `--streaming` and `--detailed` are ignored
with `--preflight`; `--preflight --pull` exits 1 when only pullable rows
are unmet; the legacy dict-dispatch MCP surface does not know `remote`
or `list_remotes`; one gzip chunk can expand past the cap in memory
before it is dropped.

## Gates (every arc)

Hermetic suite green, lint clean, mutant-red pins, a live row, and an
independent adversarial review with an explicit wrong-accept hunt before
merging to local main. Pushing and releasing need the practitioner's go.

## Spike findings: S1 (2026-09-28)

### Step 1: merge order today

| Config | Layers consulted (order) | Winner on collision | Decided by |
|---|---|---|---|
| ensembles (`ensembles/*.yaml`) | local → library → global, as a list; first consumer-side `.exists()` match wins | local | `get_ensembles_dirs` (config_manager.py:211-247); every consumer loops the list and `break`s on first match, e.g. ensemble_crud_handler.py:159-165 |
| profiles-as-files, CRUD/listing view (`profiles/*.yaml`) | local → library → global, same list shape | local | `get_profiles_dirs` (config_manager.py:249-282); profile_handler.py:66-72 |
| profiles, runtime resolution (`profiles/*.yaml` + `config.yaml: model_profiles`) | global `config.yaml` → global `profiles/` → local `config.yaml` → local `profiles/`, a hand-written `dict.update()` chain — **no library tier at all** | local (last write wins); within a tier, `profiles/*.yaml` beats that tier's `config.yaml: model_profiles` entry | `get_model_profiles` (config_manager.py:477-520) |
| scripts | local (`.llm-orc/scripts`, `.llm-orc`, project root) → installed package primitives → library submodule (`scripts/`, `primitives/python`, `primitives/`, root) — **no global (`~/.config/llm-orc`) tier at all** | local | `ScriptResolver._get_search_paths` / `_try_resolve_with_search_paths` (resolver.py:68-113, 238-273) |
| `serving:` keys (`config.yaml`) | the project's own `config.yaml` only — no local/global/library merge | whatever that one file says | `ServingEnsembleCaller._self_reference_enabled` (serving_ensemble_caller.py:1607-1630), same single-file pattern in `_load_config`/`_emit_reject_prefixes` |
| `performance:` / `agentic_serving:` keys (`config.yaml`) | defaults → global `config.yaml` → local `config.yaml`, deep-merged — **no library tier** | local | `load_performance_config` (319-355), `load_agentic_serving_config` (357-393) |

Do `get_model_profiles`'s order and `get_ensembles_dirs`'s order agree on the
winner? Partially. Both agree local beats global. They disagree
structurally: `get_ensembles_dirs` ranks library between local and global
(a library ensemble beats a global one), while `get_model_profiles` never
consults the library tier at all — a profile defined only in the library
submodule's `profiles/` dir is invisible to runtime model-profile
resolution, full stop, even though the (separately unused-for-merging)
`get_profiles_dirs` would rank it the same way ensembles are ranked. The
plan's line "Script resolution ... and profiles get the same layer" (Arc 2)
assumes `get_model_profiles` has a layer-list to extend; it doesn't — it's
two hardcoded sources, not four. Wiring in a packaged-serving layer here is
a rewrite of the merge function, not an extra list entry. See the layer-order
recommendation below.

### Step 2: `Path.cwd()` / `os.getcwd()` sites

| Site | Resolves | Bypassed when `project_dir` passed? | Empty-dir serve reads/writes through it? |
|---|---|---|---|
| config_manager.py:85 (`_discover_local_config`) | project root, walking up from cwd looking for `.llm-orc` | yes — only called when `project_dir` is `None` | Walks to filesystem root; a serve started inside an unrelated directory tree can inherit an ancestor's `.llm-orc` by accident |
| config_manager.py:209 (`_get_library_dir`) | library submodule root, for `classify_tier` | **no** — always live cwd, ignores whatever `project_dir` the manager was constructed with | Reads nothing today (path absent), but against the wrong root once `project_dir` diverges from process cwd |
| config_manager.py:219 (`get_ensembles_dirs`) | library ensembles dir (fallback branch, no `LLM_ORC_LIBRARY_PATH`) | **no** | Same — silently ignores the intended `project_dir`'s own library |
| config_manager.py:257 (`get_profiles_dirs`) | library profiles dir | **no** | Same |
| config_manager.py:409, 429, 439 (`init_local_config`) | local dir being *created* by `llm-orc config init`, and its default project name | N/A — method takes no `project_dir` param; CLI-only, never called mid-serve | No — not on the serve request path |
| resolver.py:77 (`ScriptResolver._get_search_paths`) | base dir for local + library script search paths | yes — `self._project_dir or Path(os.getcwd())` | With no `project_dir` given: local/library dirs absent, falls through to package primitives only |
| resolver.py:286 (`list_available_scripts`) | local scripts dir for the scripts-listing view | **no** — ignores `self._project_dir` even when set | A resolver built with an explicit `project_dir` still lists scripts off live cwd |
| primitive_registry.py:31 (`PrimitiveRegistry.discover_primitives`) | primitives search dirs (`cwd/.llm-orc/scripts/primitives`, `cwd/llm-orchestra-library/scripts/primitives`) | N/A — class has no `project_dir` concept at all | Finds nothing in an empty dir (also: unlike `ScriptResolver`, never adds the installed-package primitives path — a pre-existing gap independent of Arc 2) |
| serving_ensemble_caller.py:1059 (`_grep_blocks`) | display root for rendering grep-hit paths in transcript text (cosmetic) | `root` param exists but every real caller omits it, so always live cwd | No config/state impact; only affects how paths render in a block |
| v1_chat_completions.py:112-113 (`_resolve_serving_project_dir`) | **the** `project_dir` handed to `ServingEnsembleCaller` for the whole `/v1/chat/completions` path (ensembles, config.yaml, `.serve-trace`, `scripts/agentic_serving`) | no override exists anywhere (no env var, no CLI flag) — this *is* the resolution point | `.llm-orc` absent → falls back to bare `Path.cwd()` → `_find_ensemble` raises `FileNotFoundError` looking for `<cwd>/ensembles/serving.yaml`. **This is the concrete break for #196**: serving from an empty directory fails outright today; no packaged-layer fallback exists anywhere in this chain |
| artifact_handler.py:26 (`_get_artifacts_base`) | `.llm-orc/artifacts` base for MCP artifact delete/cleanup | yes, once `set_project_context` has run; else cwd | Reads/writes off live cwd until a project context is set |
| cli_library/library.py:125 (`_get_library_source_config`, priority 3) | cwd-checkout library candidate for CLI library browse/copy/search | N/A — module function, no `project_dir` param; CLI-only | Already has a documented priority-4 packaged-library fallback (#172) — direct precedent for Arc 2 |
| library_handler.py:42 (`get_library_dir`) | library root for MCP browse/search when no `library_dir` injected and `get_ensembles_dirs()` had no "library" entry | partial — injected `library_dir` bypasses it; otherwise cwd | Cwd-based until a `library_dir` is supplied |
| library_handler.py:103 (`_resolve_copy_destination`) | default local-dir for a library `copy` destination | partial — overwritten only if `get_ensembles_dirs()` yields a non-library `.llm-orc` entry | Same shape |
| script_handler.py:26 (`_get_scripts_dir`) | `.llm-orc/scripts` base for MCP script list/get/create | yes, once `set_project_context` has run | Cwd-based otherwise |
| resource_handler.py:332 (`get_artifacts_dir`) | artifacts dir for the `llm-orc://artifacts/...` MCP resource read | **no** — hardcodes `Path.cwd()` even though the handler already holds a `ConfigurationManager` with the right `local_config_dir`; every sibling method on this same handler (`read_ensembles`, `read_profiles`) correctly goes through `self._config_manager` | Genuine local inconsistency, not a structural necessity |

One naming trap worth flagging for whoever builds the project-context seam:
**`project_dir` means two different directories in two different modules.**
`ConfigurationManager(project_dir=X)` treats `X` as the checkout root
(`.llm-orc` is `X/.llm-orc`). `ServingEnsembleCaller(project_dir=Y)` treats
`Y` as the `.llm-orc`-equivalent content dir directly (`_find_ensemble`
looks for `Y/ensembles/{name}.yaml`, and `_resolve_serving_project_dir`
returns `cwd/.llm-orc`, confirming `Y` already points past the dot-dir).
Threading one project-context value into both without translating between
the two shapes will silently point one of them at the wrong directory.

### Step 3: runtime writes into the project dir

| What's written | Code path | Reader depends on colocation? | Moves to a state dir unchanged? |
|---|---|---|---|
| `.llm-orc/artifacts/<ensemble>/<ts>/{execution.json,execution.md,latest symlink}` | `ArtifactManager.save_execution_results` (core/execution/artifact_manager.py:22-84); `base_dir` defaults to `"."` (cwd) at cli_commands.py:540,598, ensemble_execution.py:261, orchestra_service.py:57, and via `ScriptCacheConfig.artifact_base_dir` (cache.py:71) | No content dependency, but `ArtifactHandler`/`ResourceHandler` **re-derive the same path independently** from `Path.cwd()` / `project_path` / `global_config_dir` rather than reading it back from the writer — movers must update every reader site in lockstep, not just the writer | Yes |
| `.llm-orc/agentic-sessions/<session>/<dispatch>/<deliverable>.<ext>` + `.retention` marker | `SessionArtifactStore.write_deliverable` (core/session/artifact_store.py:210-266) | Module docstring states it explicitly: **"NOT YET WIRED into the serving path"** — no production caller found anywhere in `src/llm_orc` | Yes, trivially — `agentic_sessions_root: Path` is already a plain constructor param; only the *default* needs to become the state dir once this is connected |
| `.serve-trace/turns.jsonl` | `emit_turn_trace` / `emit_read_continuation_trace` (web/serving/turn_trace.py:505-555); root is `ServingEnsembleCaller._trace_root`, which defaults to `project_dir / ".serve-trace"` (serving_ensemble_caller.py:1565) and is never overridden by the one production factory (`get_serving_ensemble_caller`) | No in-process reader; it's a human/instrument observability sink, colocated only by convention | Yes |
| `llama-server.ini` (rendered router preset) | `start_router_from_config` (providers/llama_server.py:368-400), writes to `config_manager.local_config_dir or config_manager.global_config_dir` (385-390) | No — `LlamaServerSupervisor.start()` is handed the exact `preset_path` in-process, never re-derives it | Yes — and this site already dodges the library/packaged trap by falling back to `global_config_dir` instead of blindly taking a search list's first entry; that's the shape Arc 2 should copy for the state-dir default |
| `.llm-orc/cache/<key>.json` (script-result cache) | `ScriptCache._save_to_artifacts` (core/execution/scripting/cache.py:227-254 write, 205-214 read), gated behind `ScriptCacheConfig.enabled=False` **and** `persist_to_artifacts=False`, both off by default (#160 — a correctness bug, not caution) | No | Yes, same shape as the artifacts row above (goes through the same `ArtifactManager`) |

Extends the brief's known set with two dormant sites
(`agentic-sessions/`'s actual wiring status, and `.llm-orc/cache/`) and one
correction: none of the found writers reads its own path back from a
sibling — every reader re-derives the path from cwd/project-dir/config
independently, so "move the writer" and "move every reader" are separate
work items, not one.

Not project-dir writes (correctly out of scope, already install-scoped
under `resolve_global_config_dir()`): `credentials.yaml`,
`.encryption_key` (config_manager.py:284-290), `session_id_salt`
(core/session/identity_salt.py). `ConversationCompaction._persist_artifact`
(core/session/compaction.py:600) takes a required `persistence_root: Path`
with no cwd default and no production instantiation anywhere in
`src/llm_orc` — dormant, already fully parameterized, nothing to move.

### Step 4: what #144 assumes

The self-reference design (`docs/plans/2026-08-13-dot-dir-self-reference-design.md`)
is built on this repo dogfooding itself: `serving.self_reference: true` is
committed in *this* repo's own `.llm-orc/config.yaml` with the comment
"THIS repo is the self-hosting deployment," and the mechanism reads
`project_dir/scripts/agentic_serving/*.py` straight off disk at request
time so the serve can answer "how does resolve pick the seat?" by reading
its own literal, currently-running implementation. The design's whole
premise — "the operative script set for THIS serve instance," "self-hosting
is the rung" — is that the project directory a serve is handed IS the
checkout whose behavior it is exhibiting, so a self-read is grounded in the
truth, not a stale or unrelated copy.

That premise survives a read-only packaged layer intact for *reads*: if the
packaged `serving_project/` ships the same `scripts/agentic_serving/*.py`
that ships in the wheel actually running the serve, a self-read of that
packaged copy is still ground truth, and `_load_emit_reject_prefixes`'s
dynamic `importlib` load of `emit.py` works unchanged against a
read-only path. What breaks is everything Step 3 found: `_trace_root`
defaults to `project_dir / ".serve-trace"`, so a serve whose `project_dir`
resolves to the packaged (read-only, likely site-packages) layer would
attempt to create `.serve-trace/` inside the installed package and fail —
today there's no state-dir indirection for this at all. The flag's
default-OFF-everywhere-but-here design is orthogonal and unaffected: a
packaged serving project can ship with `self_reference` off, the same as
every other served project, with this repo's own `.llm-orc/config.yaml`
staying the one place it's on.

### Step 5: the answer

**Can the serving project be a read-only layer, and what must move to a
state dir?** Yes: every read path that resolves ensembles, profiles-as-files,
scripts, and the `serving:` config keys already isolates disk access behind
a project-dir-shaped parameter, or has a direct working precedent for one
(#172's cwd-vs-packaged fallback for the library), so a packaged read-only
layer slots in as another source without changing those functions' shapes.
What must move to a state dir is every write found in Steps 2-3:
`.llm-orc/artifacts/`, the dormant `.llm-orc/agentic-sessions/` and
`.llm-orc/cache/`, `.serve-trace/turns.jsonl`, the rendered
`llama-server.ini`, and the ensemble/profile/script CRUD write paths, whose
local-directory fallback (next finding) must be corrected so it never lands
in a read-only tier.

**Proposed layer order:**

- **Ensembles** and **profiles (CRUD/listing view)**: project → library →
  packaged serving → global, exactly as the plan states. This is a direct
  extension of `get_ensembles_dirs` / `get_profiles_dirs`'s existing list;
  the first-match-wins consumer pattern needs no change.
- **Profiles (runtime, `get_model_profiles`)**: needs a structural fix, not
  a list entry, because the function has no list today (Step 1). Recommend
  rewriting it to merge the same four tiers in the same rank order as
  ensembles (lowest to highest precedence, each stage `.update()`-ing the
  last): global `config.yaml` → global `profiles/` → library `profiles/` →
  packaged-serving `config.yaml`/`profiles/` → local `config.yaml` → local
  `profiles/`. This both adds the packaged layer and closes the Step 1 gap
  where library profiles were invisible to runtime resolution — a
  behavior change beyond #196's minimum, so Arc 2 should call it out
  explicitly as a deliberate decision, not a side effect.
- **Scripts**: today's order is project → package primitives → library,
  with no global tier. Recommend packaged-serving scripts rank immediately
  after the project's own `.llm-orc/scripts` (in the "local" family), ahead
  of package primitives and the library submodule — a served project's own
  scripts stand in for "the project" when no local override exists, so
  they belong at the project's rank, not the framework-primitives or
  third-party-library rank. This is a judgment call the plan should confirm
  explicitly, not a fact S1 measured.
- **`serving:` config keys**: stays a two-tier shadow, not a new merge —
  project's `config.yaml` if it has a `serving:` key, else the packaged
  serving project's `config.yaml`. Today this key has no merge at all (not
  even local-over-global); expanding it to a general library/global merge
  would be new behavior nothing in #196 asks for.

**Sites from Steps 2-3 that Arc 2 must route through a project context**
(consolidated; full detail in the tables above): config_manager.py:209,
219, 257 (library resolution ignores an explicit `project_dir`);
resolver.py:286 (`list_available_scripts` ignores `self._project_dir`);
primitive_registry.py:31 (`PrimitiveRegistry` has no `project_dir` concept
at all); v1_chat_completions.py:112-113
(`_resolve_serving_project_dir` — the central #196 break, needs a
packaged-layer fallback); artifact_handler.py:26, library_handler.py:42/103,
script_handler.py:26 (cwd fallback before `set_project_context` runs);
resource_handler.py:332 (`get_artifacts_dir` should use
`self._config_manager` like its siblings, not raw cwd); every write site in
the Step 3 table (artifact_manager.py, turn_trace.py via
serving_ensemble_caller.py:1565, llama_server.py:385-390, cache.py).

**Arc 2 design question: the `get_local_ensembles_dir` /
`get_local_profiles_dir` write-dir fallback when no project exists.**
Both functions (ensemble_crud_handler.py:265-284,
profile_handler.py:60-73) search `get_ensembles_dirs()` /
`get_profiles_dirs()` for an entry with `.llm-orc` in the path and
`library` not in it; when none exists — no local project — they fall back
to `dirs[0]`, the first entry in the merge-priority list. Today that's
usually the library submodule if one happens to be present, which is
already the wrong write target (nothing should ever CRUD-write into the
library). Once the packaged serving layer ships as another read-only,
higher-priority entry in the same list, `dirs[0]` becomes even more likely
to be a read-only path — a `create_ensemble` call with no local project
present would either write into (and corrupt) a read-only packaged
install, or raise a permission error with no guidance. The fallback should
become: skip every entry whose tier (`ConfigurationManager.classify_tier`,
config_manager.py:180-197 — needs a `"packaged"` tier added) is `library`
or `packaged`, and land on `global_config_dir` (creating it via the
existing `ensure_global_config_dir` if needed) — never on the first list
entry unconditionally. This also matches Arc 1's `scope: "project" |
"global"` default: with no project directory to be "project" scope
*of*, falling through to the global, always-writable tier is the only
choice that can't silently corrupt a read-only layer.

Nothing in Steps 1-4 was left undetermined; the routing decisions in Step
5 above are S1's proposals for Arc 2 to confirm or correct, not measured
facts.
## Spike findings: S2 (2026-09-28)

Binary: `/opt/homebrew/bin/llama-server`, version 9850 (4f31eedb0), built
with AppleClang 21.0.0.21000099 for Darwin arm64.

Router-relevant flags from `--help`:
- `--models-dir PATH`: directory containing models for the router server
  (default: disabled)
- `--models-preset PATH`: path to INI file containing model presets for
  the router server (default: disabled)
- `--models-max N`: for router server, maximum number of models to load
  simultaneously (default: 4, 0 = unlimited)
- `--models-autoload, --no-models-autoload`: for router server, whether to
  automatically load models (default: enabled)

Setup: started the router with `LlamaServerRouter.command()`'s exact argv
(`--models-preset <copy>`, `--host 127.0.0.1`, `--port 8790`,
`--models-max 1`, `--no-webui`) against a scratch copy of the real rendered
preset (6 sections). `GET /models` answered 200 with 10 entries (the 6
preset sections, a `default` entry for the bare command line, and 3 raw
Hugging Face cache entries already on disk), so the preset loaded correctly
outside the serve.

| Probe | Status | Body excerpt | Conclusion |
|---|---|---|---|
| `POST /models/load` naming a model not in the preset | 404 | `{"error":{"message":"File Not Found","type":"not_found_error","code":404}}` | Load endpoint refuses an unknown name; no live add. |
| Chat completion naming a model not in the preset | 400 | `{"error":{"code":400,"message":"model 'totally-fake-model-xyz' not found","type":"invalid_request_error"}}` | Same refusal on the OpenAI-compatible path. |
| Append `[qwen3-0.6b-newprofile]` to the on-disk ini, no restart, re-`GET /models` | 200, unchanged | still 10 entries, new section absent | Preset is read once at startup; on-disk edits are inert while the router runs. |
| `SIGHUP` to the running router, re-`GET /models` | 200, unchanged | still 10 entries; log file gained zero new lines; process stayed alive | `SIGHUP` is not a reload signal in this build: no crash, no reload. |
| Second router started with `--models-dir <dir with one symlinked gguf>` | 200 | seeded file listed as `qwen3-0.6b` (id from filename) among 4 entries | `--models-dir` directory scan works at startup, same as `--models-preset`. |
| Second gguf symlink dropped into that dir after start, re-`GET /models` (no restart) | 200, unchanged | still 4 entries, dropped file absent | Directory scan is also startup-only; a `SIGHUP` to this router was the same no-op. |
| `SIGTERM` the router (model loaded, mid in-flight completion), spawn a fresh process on the same preset plus the added section, poll `GET /models` | first 200 listed 11 entries incl. `qwen3-0.6b-newprofile` | two runs: 1.566s and 2.194s from `SIGTERM` to the first answering `GET /models` | Restart is the only way a new preset section takes effect; wall-clock cost measured at roughly 1.5 to 2.2s. |
| The in-flight `POST /v1/chat/completions` riding that same `SIGTERM` (non-streaming, 300 to 400 `max_tokens`) | curl exit 18, `HTTP:200`, `SIZE_DOWNLOAD:0` (both runs) | `curl: (18) transfer closed with outstanding read data remaining` | In-flight completions are cut hard: a 200 status line arrives, zero body bytes follow, connection drops. No drain. |

Additional observation: a restarted router does not retain load state. A
model loaded before the restart came back `unloaded` after, so the first
post-restart request to it also pays that model's on-demand load latency
(not separately measured here; router-mode logs confirm "models will be
automatically loaded on-demand").

**Decision:** Arc 3's `pullable` for a newly shipped profile must be
`needs_restart`. Neither `/models/load`, an on-disk preset edit, `SIGHUP`,
nor `--models-dir`'s directory scan pick up a model outside what the router
scanned at startup, so a supervised restart is the only mechanism that
works, at a measured cost of roughly 1.5 to 2.2s wall clock (`SIGTERM` to
first answering `GET /models`) plus a hard cut of any in-flight completion
(0 body bytes on the killed connection) and loss of prior load state.

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
`feat/crud-scope` @ 1652a258, serve on `--port 8766 --backend-port 8790
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

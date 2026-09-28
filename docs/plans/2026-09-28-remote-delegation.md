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

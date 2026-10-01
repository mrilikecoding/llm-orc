# Remote delegation, Arc 4: one-run injection

**Goal:** `invoke` accepts an ensemble inline, with its child ensembles,
profiles and scripts, plus `bind` and `pull`. The serve runs it once and
keeps nothing. When the host cannot run it, the call fails before any
agent runs with the Arc 3 dependency report (#196, #191).

**Architecture:** A run layer is a directory shaped like every other
tier (`ensembles/`, `profiles/`, scripts at their keys) under
`<state dir>/runs/<id>/`. `ConfigurationManager.with_run_layer(dir)`
returns a view that puts it highest in every tier list; the run's
executor is built on that view. One preparation step sits in front of
the three service entry paths (`ExecutionHandler.invoke`,
`execute_streaming`, `invoke_streaming`): validate, materialize, gate
with `check_ensemble_runnable`'s own function over the view, pull if
asked, build the executor, remove the directory in a `finally`.

**Spec:** `docs/plans/2026-09-28-remote-delegation.md`: decisions 2-4,
the Arc 4 card, "Arc 4 re-cut (2026-10-01)" (rulings 1-12 and the two
probes), "Arc 3 re-cut" rulings 2, 3, 6, 7, and "Standing rules for
every implementer". Read the Arc 4 re-cut before touching anything. The
rulings are binding; where a card below and a ruling disagree, the
ruling wins and you say so in your report.

**Tech stack:** Python 3.11+, pydantic, FastAPI TestClient, pytest
(asyncio auto mode), ruff 88, mypy strict, complexipy <= 15.

## Global constraints

- One worktree for the arc:
  `git worktree add .claude/worktrees/arc4-injection -b feat/one-run-injection main`.
  Never `git stash` (the working dir is shared with other sessions).
  Never spawn subagents that edit or commit.
- Strict TDD, one test at a time. Structural and behavioral commits
  separate. ruff 88 + mypy strict from the first draft. `make lint`
  clean before every commit.
- Commit prefixes `feat:` `fix:` `refactor:` `test:` `docs:` `chore:`.
  No AI attribution of any kind, no session links, no scratch paths in
  tracked files.
- Full suite: `uv run pytest -q -p no:cacheprovider`. Known local-only
  failure: the auth-only providers test when a router listens on :8080.
  Record the baseline count in Task 0.
- Doctrine 11: pins assert outcomes through the real surface (a real
  `OrchestraService` on a temp project dir, REST `TestClient`, the real
  executor, files on disk). Each key pin is shown RED under a named
  mutant in the run that gates. Put the mutant and the red assertion
  line in the commit body. Clear `tests/__pycache__` after mutant
  rounds.
- Never run the service outside pytest (no ad-hoc `TestClient` or
  `ConfigurationManager` scripts): they write into the real
  `~/.config/llm-orc`. End every task with
  `git status --short` in the worktree and
  `ls ~/.config/llm-orc/ensembles ~/.config/llm-orc/profiles` unchanged.
- Tests may not launch the real `llama-server` (autouse guard). The
  router is faked at `LlamaServerClient._list` and `load` with the JSON
  shapes the probes recorded.
- Doc drift gate: this file names tests only in prose, never as
  backticked identifiers. Keep it that way when editing it.
- Findings go on #196 and #191 as comments, never new issues. Nothing
  is pushed, nothing paid is run, the mini is not touched.
- The eleven statuses are closed. Do not add one.

## Review focus

Inputs the spec implies and no card names. Each gets a pin in the
owning task; the whole-branch review hunts these first.

1. **The host has its own profile `a`, the caller binds `a` to a
   missing `b`.** Expected: `not_equipped`. A run on the host's `a` is
   the wrong accept.
2. **An inline llama-server profile reuses a host model name with a
   different `hf_repo`.** Expected: `needs_restart`, no run.
3. **A script key or ensemble name that escapes the run directory**
   (`../x.py`, `/etc/x`, `a//b`, a backslash). Expected:
   `invalid_request`, nothing written anywhere.
4. **A refused or failed run leaves the run directory behind.**
   Expected: removed on refusal, on an executor exception, on
   cancellation.
5. **An injected script without its helper, on a host that has a
   helper at the same relative path.** Expected: the script fails; it
   does not import the host's helper.
6. **Two concurrent runs ship different content under one script path
   and one profile name.** Expected: each sees its own.
7. **A view leaks.** After an injected run, the service's own config
   manager lists no run-layer ensemble or profile, and a following named
   run resolves as before.
8. **`pull: true` and the load ends `unloaded` with `failed: true`.**
   Expected: `not_equipped`, the dependency still `pullable`, the
   observed status in `detail`.
9. **A named ensemble with a `pullable` model and no `pull`.**
   Expected: `not_equipped` (ruling 1, the breaking change).
10. **The gate and the run disagree.** For each injected kind (child
    ensemble, script, profile, binding), the gate's verdict and what
    the real executor does are pinned on one fixture.

---

## Task 0: worktree and baseline

Create the worktree, `uv sync -q`, run the full suite, record the
passed count in your report.

## Task 1: the router states its sources (ruling 6)

**Files:** `src/llm_orc/providers/llama_server.py` (`RouterInventory`,
`inventory`), `src/llm_orc/providers/status_types.py`
(`LlamaServerProviderStatus`),
`src/llm_orc/services/handlers/provider_handler.py`
(`_get_llama_server_status`), `src/llm_orc/services/handlers/preflight.py`
(`_classify_llama_server`), their unit tests,
`tests/unit/web/test_api_runnable_preflight.py`.

**Interfaces:** `RouterInventory.sources: dict[str, str]` (preset model
id to the value after `--hf-repo` in `status.args`; a model with no such
arg is absent). Provider status carries `sources`. Classifier: listed,
profile has `hf_repo`, router source known and different → `needs_restart`
with a `detail` naming both sources. Router source unknown (no args)
→ today's behavior.

**Pins:** the probe's listing shape (spec, Arc 4 re-cut probes) yields
the sources; through the REST runnable route, a profile whose model is
listed under another source reads `needs_restart` and `runnable` false;
the same profile with the matching source still reads `ready`. Mutant:
drop the comparison.

## Task 2: one pull function, off the event loop (ruling 4)

**Files:** `src/llm_orc/providers/llama_server.py`,
`src/llm_orc/web/api/models.py`, tests under `tests/unit/providers/` and
`tests/unit/web/`.

**Interfaces:** `LlamaServerClient.pull(model, *, timeout_s, poll_s) ->
dict[str, Any]`: calls `load`, polls the listing until the status is not
`loading` or the deadline passes, returns `{"status": <value>, "failed":
<bool>, "exit_code": <int | None>}` from the listing entry. Synchronous.
`POST /api/models/{name}/pull` calls it in a worker thread and keeps its
response shape (`name`, `status`).

**Pins:** with a stub router whose load takes time, `/health` answers
while a pull is in flight (red when the endpoint calls `pull` inline); a
load that ends `unloaded` + `failed` is returned as such, not as
success. Note the finding on #199.

## Task 3: the run layer (rulings 2, 7)

Structural commit first: `ScriptResolver` takes `run_dir`, threaded
through `ScriptAgentRunner` and `ScriptAgent` with `None` everywhere, no
behavior change. Then behavior.

**Files:** `src/llm_orc/core/config/config_manager.py`,
`src/llm_orc/core/config/ensemble_config.py`
(`child_ensemble_search_dirs`),
`src/llm_orc/core/execution/scripting/resolver.py`,
`src/llm_orc/core/execution/scripting/agent_runner.py`,
`src/llm_orc/agents/script_agent.py` (`_bytecode_environment`),
`src/llm_orc/core/execution/ensemble_execution.py`, tests beside
`tests/unit/core/execution/test_child_ensembles_across_tiers.py`.

**Interfaces:**
- `ConfigurationManager.with_run_layer(run_dir: Path) ->
  ConfigurationManager`: a copy with `run_layer_dir` set and an empty
  profile cache. `run_layer_dir` property, `None` on a plain manager.
  `_tier_dirs` lists the layer first; `_profile_tiers` lists it last.
- `child_ensemble_search_dirs` puts `<run dir>/ensembles` first.
- `ScriptResolver(project_dir=..., run_dir=...)`: search paths start
  with `<run_dir>/scripts`, `<run_dir>`.
- `EnsembleExecutor` passes `config_manager.run_layer_dir` to the script
  runner; with a layer, script bytecode goes under
  `<run dir>/pycache` and the script cache is disabled.

**Pins (real executor, files on disk, `mock-*` models so nothing is
called out):** a child ensemble present only in the layer runs; a script
present only in the layer runs; a profile present only in the layer is
the one loaded; a layer profile shadows a global profile of the same
name; the base manager the view was made from still resolves none of
them; two views over two directories each resolve their own content
under one shared name; no file appears under the state dir's `pycache`
after a layer run whose script imports a sibling module. Mutants: drop
the layer from each of the four lists in turn; each must turn its own
pin red.

## Task 4: the gate is the run's function (ruling 10)

Structural commit first: extract from
`ProviderHandler.check_ensemble_runnable` a method that takes the loaded
root, its reference, a config manager and a project dir, and returns the
closure and the reports. `check_ensemble_runnable` calls it with the
service's own pair; no behavior change, the Arc 3 pins stay green.

Then behavior: the method builds its child finder
(`child_ensemble_search_dirs` + `_find_ensemble_in_dirs`), its profile
map (`get_model_profiles`) and its script resolver from the pair it is
given, and provider status is computed over that profile map (the
OpenAI-compatible endpoints are grouped from it).

**Files:** `src/llm_orc/services/handlers/provider_handler.py`,
`src/llm_orc/services/orchestra_service.py`,
`tests/unit/services/test_preflight_parity.py`.

**Pins (extend the parity file):** on one fixture per kind, the gate
called with a view reports `ready` exactly when the real executor built
on that view resolves the thing, for a child ensemble, a script and a
profile that exist only in the layer; called with the base manager it
reports them missing. An inline OpenAI-compatible profile's endpoint
appears in the status that classified it. Mutant: build the finder from
the service's manager instead of the one passed in.

## Task 5: request, materialization, bindings (rulings 3, 9)

**Files:** create `src/llm_orc/services/handlers/run_request.py`; tests
`tests/unit/services/handlers/test_run_request.py`.

**Interfaces:**
- `RunRequest` (pydantic, `extra="forbid"`): `ensemble_name: str | None`,
  `ensemble: dict | None`, `ensembles: dict[str, dict]`, `profiles:
  dict[str, dict]`, `scripts: dict[str, str]`, `bind: dict[str, str]`,
  `pull: bool = False`, `input: str = ""`. Validators implement ruling 9.
  `needs_layer` is true when any of `ensemble`, `ensembles`, `profiles`,
  `scripts`, `bind` is set.
- `RunRequestError(ValueError)`: the `invalid_request` carrier.
- `materialize(request, run_dir) -> Path | None`: writes
  `ensembles/<name>.yaml` (the root too, returning its path; an inline
  child with no `name` gets its key's last segment), `profiles/*.yaml`
  with `name` set to the key, each script at its key. Every target is
  checked to resolve inside `run_dir` before the write.
- `apply_bindings(bind, view, run_dir) -> tuple[dict[str, str],
  list[tuple[str, str]]]`: for each `a: b`, look `b` up in the view's
  profiles as they stood before any binding was written, write a layer
  profile `a` with `b`'s definition; returns the applied map and the
  unmet pairs.

**Pins:** each escape shape in review focus 3 raises and leaves the
directory tree untouched outside and inside; both or neither root is
rejected; a bind key that is also an inline profile is rejected; after
`apply_bindings` the view resolves `a` to `b`'s model and provider and
keeps `b`'s fallback chain; a bind target defined only inline works; a
missing target is returned unmet and writes nothing.

## Task 6: prepare, gate, run, clean up (rulings 1, 3, 4, 5, 8, 10)

Structural commit first: the three entry paths in `ExecutionHandler`
take the root and the executor from one private preparation function
(named ensembles only, today's behavior, today's tests green). Then
behavior, one pin at a time.

**Files:** `src/llm_orc/services/handlers/execution_handler.py`,
`src/llm_orc/services/orchestra_service.py`, tests
`tests/unit/services/test_one_run_injection.py` (create).

**Shape:** an async context manager yields the prepared run (root
config, executor, applied bindings, pulled models) and removes the run
directory on exit. Inside: parse `RunRequest`; if it needs a layer,
`mkdtemp` under `resolve_state_dir(local) / "runs"`, materialize, build
the view, apply bindings; load the root (inline: from its layer file
with the run's search dirs; named: today's lookup); gate with Task 4's
method over the view (or the base manager when there is no layer);
append a `missing_profile` row with `via: ["bind:<a>"]` for each unmet
binding; reject bind keys the closure does not name; when `pull` is
set, pull each `pullable` model with Task 2's function in a worker
thread and mark it resolved only on `loaded`; if anything is still
unmet, refuse. The executor for a layer run comes from
`ExecutorFactory.create_root_executor` on the view with the service's
cached credential storage, `save_artifacts=False` when the root is
inline; a run with no layer keeps `_get_executor()`.

**Result:** a success keeps today's keys and adds `bindings` and
`pulled` when non-empty. A refusal is ruling 10's envelope. In
`execute_streaming` the refusal is also reported through the reporter;
in `invoke_streaming` it is one `execution_failed` event carrying the
envelope's `error`. `execute_streaming` saves no artifact for an inline
root.

**Pins (through `OrchestraService.invoke` on a temp project, state and
global dirs; snapshots are full recursive listings with file sizes):**
1. An inline ensemble (a script agent and a `mock-*` agent) runs and
   the three trees are identical before and after, `runs/` empty.
2. A missing profile refuses with `not_equipped` and the report; a
   phase-0 script agent in the same ensemble that writes a marker file
   did not run.
3. The same call with `bind` runs and returns the binding.
4. Review focus 1, 2, 4 (refusal, executor exception, cancellation), 5,
   6 (two `asyncio.gather` runs), 7, 8, 9.
5. An unreferenced bind key is `invalid_request`.
6. An injected script importing an injected sibling `_helpers` runs.
7. `pull: true` over a stub router: the pull is called, the run
   proceeds, `pulled` names the model; `/health`-style liveness is
   Task 2's pin and is not repeated.
8. A named root with a binding keeps its artifact; an inline root has
   none.

Mutants, each named in the commit that adds its pin: skip the gate;
skip the `finally`; build the executor on the base manager; let an
unmet binding through; re-read the listing after a pull instead of
trusting `loaded`.

## Task 7: surfaces (ruling 11)

**Files:** `src/llm_orc/web/api/ensembles.py`,
`src/llm_orc/mcp/server.py` (the FastMCP `invoke` tool, the legacy tool
schema, `_invoke_tool_with_streaming`),
`src/llm_orc/services/handlers/help_handler.py` if it documents
`invoke`, tests in `tests/unit/web/test_api_ensembles.py` and
`tests/unit/mcp_server/`.

REST: `POST /api/ensembles/execute` with the full body; `POST
/api/ensembles/{name}/execute` with the same body minus the root (a
root in the body is a 422). Both forbid unknown keys. MCP: `invoke`
gains `ensemble`, `ensembles`, `profiles`, `scripts`, `bind`, `pull`;
`ensemble_name` optional; `input_data` unchanged.

**Pins:** pins 1-3 of Task 6 once each through the REST route and once
through the MCP tool call; a misspelled body key is a 422; `GET
/api/ensembles/execute` behavior is unchanged (it reads an ensemble by
that name).

## Task 8: docs

`docs/serving.md`: a section after "Preflight" with the request shape,
the error shape, `bindings` / `pulled`, what an inline profile can and
cannot do (ruling 5), the helper rule (ruling 7), and the trust note
(decision 4: injected scripts run unsandboxed as the serve's user; an
inline profile can point the host at any endpoint; the boundary is who
can reach the port, for us the tailnet). The `needs_restart` row gains
the source-mismatch case. No CHANGELOG entry (release time).

## Task 9: live row and review gate (lead)

Live row on the laptop, a serve from an empty dir with temp XDG dirs and
a real router: an inline ensemble with a real local model and an
injected script + helper, tree snapshot before and after; a missing
profile refused; the same with `bind`; a `pullable` small model refused,
then `pull: true`; an inline profile with a foreign source for a listed
model refused `needs_restart`; two concurrent runs sharing a script
path; one call through `/mcp`. Then an author-independent whole-branch
review with a wrong-accept hunt over the review focus list, one fix
wave, a scoped re-review, merge to local main, roadmap State and the
board updated, summaries on #196 and #191. The mini row waits for the
cut-over and a release (practitioner).

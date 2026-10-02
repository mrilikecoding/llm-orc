# Serving Roadmap — to the North Star

Revamped 2026-07-17 for execution by non-Fable sessions (Opus-class lead,
Sonnet-class implementers, Haiku-class mechanical). The prior roadmap —
including the full trajectory table, the Arc D correction history, and the
2026-07-09 path items — is frozen verbatim in
`docs/serving-roadmap-archive-2026-07-17.md`; treat it as provenance, not
as the plan. This document is the plan. Its **State** section is rewritten
(not appended) at each update; history lives in the archive and in git.

## North star

**Full model parity through composition** (see `docs/serving.md`): llm-orc
served agentically behind OpenCode should be as functional as Claude Code
running a frontier model, and beyond it where composition wins, all through
orchestrated small models. This is a literal engineering target. The
comparator is a frontier model PLUS its harness — behind OpenCode it reads
files, runs tests, and iterates, so "a model doesn't check its work" is not
the bar. Three levers a monolithic model does not have:

1. **Verified acceptance** — structural verification (an unverified build
   cannot ship) vs the comparator's discretionary verification (it checks
   when it or harness policy chooses).
2. **Lossless memory** — deterministic selection over the full history
   (eventually cross-session substrate) instead of attention over a
   decaying window.
3. **Zero marginal cost** — systematic coverage (more rounds, more
   verification, more retrieval) is free where every frontier token is
   billed.

Latency is the accepted trade. Two axes on top of the levers: **task
generality** (closed intent set today → catalog growth → compose-at-runtime
→ composer ensembles; the ADR-047 ladder, WS-7) and **long-horizon
operation** (the client's agentic loop is the engine; the serve is a
deterministic next-action function over a lossless record; WS-5/6).

Posture (practitioner, 2026-07-11): local-first; occasional hosted
capability in a measured seat when it is the cheapest path to the bar,
every hosted seat carried as a named buy-back target (§Seat-capability
ladder). 32GB rig is the permanent target.

**Development north star (named 2026-07-17).** The serving north star is
an *outcome* — what the endpoint does for a user. Alongside it sits a
*trajectory* target for how the codebase itself evolves: the three-layer
platform (plexus substrate · llm-orc engine · client execution surface,
see `docs/plans/2026-07-17-plexus-integration-platform-assessment.md`)
with layers coupled only through named contracts — MCP tools, the OpenAI
wire + tool_calls, the `{requirement, code, tests}` seat contract, the
client's advertised tool list — never through schemas, file layouts, or
spawn assumptions. Its steering rule: **behavior migrates downward as it
stabilizes** — from prompt rules, to bounded model judgment, to
deterministic code, to the declarative layer (YAML shapes, chain tables,
closed templates), and ultimately, for components that survive
adversarial review and stop changing across arcs, into hardened kernel
code (plausibly Rust crates on the plexus side). Every arc should leave
more behavior in a lower layer than it found it. This is the standing
answer to the rewrite question: no greenfield port; the declarative
layer is the insulation that keeps an eventual hardening cheap, and
"frozen component" status is the trigger, tracked informally the way the
buy-back ledger tracks hosted seats.

## State (2026-10-01 handoff)

**Ranked index:** the GitHub project "llm-orc kanban"
(https://github.com/users/mrilikecoding/projects/2). THIS DOCUMENT
GOVERNS; the board is its index. Board "Done" = merged on main. Pushed
2026-09-16 on the practitioner's go: the 166-commit backlog, then PR
#186 (#90) merged and released as v0.20.0; CI is green on main again
(two pre-existing test failures and a whole-tree format check fixed on
the way). Pushing remains gated per push.

**Practitioner directives (2026-09-16):** drop Ollama (decided, not
debated further); the deployment target is a serve running natively on
remote-host, proxied through a Dokku app, reachable on the tailnet from any
client; llm-orc must be deployable without coupling to additional
applications (#90's packaging goal). Standing directives from 2026-09-11
hold (daily driver on existing repos; real OpenCode sessions; Go spend
and paid comparison runs within reason; cheaper subagents; meter usage).

### 2026-10-01: history rewritten, machine-specific files removed

llm-orc is a machine-agnostic tool (practitioner, 2026-10-01): files for
one machine's deployment do not belong in this repo. Two things were
removed from every commit that carried them: `deploy/remote-host/` (a launchd
unit and setup notes, now in a private notes repo) and the second
`.mcp.json` entry that pointed at one host's serve (now a user-level MCP
entry on the practitioner's laptop). The 219 commits since 2026-09-16
have new hashes (three that touched only `deploy/remote-host/` are gone) and
tags v0.20.0 through v0.22.0 point at the rewritten commits; commits and
tags before that are unchanged. Every hash cited in this document and in
`docs/plans/` is the new one. The pre-rewrite history is kept locally
under `refs/backup/pre-deploy-rewrite/` until the push is verified.

The practitioner gave the go for the force-push on 2026-10-01: main and
the nine tags. It has landed when `git ls-remote origin main` matches
local main. In the same step: the tap formula downloads the v0.22.0 tag
tarball and pins its sha256, so the formula takes the new sha; origin's
merged PR branches (`feat/llama-server-backend`,
`fix/200-rest-api-agent-properties`,
`fix/issue-202-child-input-contract`) carry the old commits and are
deleted. What a push cannot reach: GitHub keeps the old commits
reachable through the merged PRs' refs (#186, #201, #203) until GitHub
Support purges them; the PyPI sdists for 0.20.0 through 0.22.0 contain
the directory and cannot be changed, only yanked (left as they are; the
wheels never carried it). Hashes cited in GitHub issue comments before
this date are the old ones.

### 2026-10-01: v0.22.0 released; Arc 4 in flight

Released v0.22.0 on the practitioner's go: push `1e5cfc02..f9089aa4`,
tag, GitHub release, PyPI, tap formula bumped (`1b3cdb00`, Intel
`cryptography<49` constraint intact). It carries Arc 1's `scope`, Arc 2
(packaged serving tier, state dir, serving root) and Arc 3 (transitive
preflight). CI is green on main: six test cells, security, wheel check.
On the way: `make wheel-check` now runs in CI (one cell) and in the
publish build before upload (`c54dd9e8`); pip-audit failed CI on new
advisories for `pyjwt` 2.13.0 and `urllib3` 2.7.0 (both transitive), the
lock pins moved to 2.15.1 and 2.8.0 (`b6ad88a6`).

Lesson, binding: `publish.yml` runs on every push to main and publishes
the version in `pyproject.toml`. The push that carries a version bump IS
the PyPI release, before CI finishes. Push the work at the old version
first (publish skips the existing version), wait for CI, then push the
release commit.

Verified by a clean install from PyPI (fresh venv, `--no-compile`):
`llm-orc --version` 0.22.0; 162 files under `llm_orc/serving_project/`;
a serve from an empty dir with temp XDG dirs answered `/health` healthy
0.22.0, `/v1/models` = `agentic-tier-cheap-general`, 105 ensembles
{global, packaged}, six router models, preset in the state dir, cwd
empty, nothing written under the packaged tier;
`GET /api/ensembles/serving/runnable` → `runnable: true`, 9 dependencies
(the root, 6 scripts, 1 profile, 1 dynamic dispatch). Found on the way,
on #191: a hierarchical name in a REST path
(`/api/ensembles/agentic-serving%2Fweb-searcher/runnable`) answers 200
with the web UI's HTML; the plain name works.

`research-dossier.yaml` moved from the checkout's `.llm-orc/ensembles/`
to the laptop's global tier, so local `make wheel-check` is green.

**Mini cut-over done (2026-10-01, on the practitioner's go).** The
deploy procedure is a generic script in the homelab repo
(`llm-orc-deploy.sh`, the unit rendered from a template and the
machine's own untracked `config.sh`), run over ssh. The homelab repo is
machine agnostic too; what is true of one host lives in a private notes
repo. Before it the unit's working directory was still the old
checkout, whose `.llm-orc` shadowed the packaged serving project (227
ensembles listed, 79 duplicate names, the checkout's router preset).
After it: the serve runs 0.22.0 from a plain directory, 126 ensembles
{library 15, global 10, packaged 101}, no ensemble name lost, the 60
profiles and six router models unchanged, the preset in
`~/.local/state/llm-orc/`, `research-dossier` and `serving` both
`runnable: true`, `/v1/embeddings` and `/mcp` answering. The cache-id
rule holds on the mini's llama-server build (a cache entry's id equals
the profile's `hf_repo`), so spec ruling 1 stands there. The library
ensembles stay listed through `LLM_ORC_LIBRARY_PATH` in the mini's
config. `/v1` results taken from the mini before this date ran the
checkout's 0.20.4-era serving project on a newer engine. The homelab
cert renewal daemon fix is still owed before December.

**In flight: Arc 4 (one-run injection).** The re-cut is done: twelve
rulings in the spec ("Arc 4 re-cut (2026-10-01)", with the code read and
two router probes) and the plan with task cards
(`docs/plans/2026-10-01-remote-delegation-arc4.md`). Implementation is on
`feat/one-run-injection` in `.claude/worktrees/arc4-injection`, Sonnet
implementers per task, Opus reviews, a whole-branch review before merge
(where Arc 3's three wrong-accepts were found; do not skip it). The plan's
Task 9 is the laptop live row and the review gate. The rulings a caller
will notice: every REST/MCP run is gated, named ensembles too (an
ensemble that is not `runnable` answers `not_equipped`); `needs_restart`
is never resolved inside a run; an inline root saves no artifact; a
listed model whose router source differs from the profile's `hf_repo`
reads `needs_restart`.

### 2026-09-27: fail-closed composition, released v0.21.0

Merged `c5700ee3`, released v0.21.0 (breaking fix for callers; see
CHANGELOG). Plan and addenda: `docs/plans/2026-09-22-fail-closed-composition.md`.
Invariant: a failed step never reaches a consumer, parent, or caller as
a success. Shape: implicit 0.6b fallback removed (explicit
`fallback_model_profile` chains only, each hop loaded as its profile);
one engine-owned per-node `outcome` read through one predicate module;
dependency-failure cascade with `on_dependency_failure: run` for
serving's refusal composers; ensemble/dispatch/loop fail when no child
terminal succeeded (loop: final iteration, earlier ones recorded as
`iteration_failures`); script failures are `failed` with the payload
under `payload`; LLM consumers get a child's terminal output, not its
execution record; CLI/REST/MCP report `status` + `has_errors`,
`deliverable` null unless a terminal succeeded, CLI exits non-zero;
`think` validated per provider before execution; OpenCode Go requests
carry User-Agent and a salted per-invocation `x-opencode-session`.
Four independent review rounds; suite 4677.

**T1 regression-gate row (laptop, llama-server qwen3-8b, OpenCode, r=5
each, same fixture):** main `14421dd1` 0 correct / 1 shipped broken /
3 refused / 1 client timeout; branch 0 / 1 / 4 / 0. No branch
regression. The backend result is the finding: T1 is 0 of 10 correct on
llama-server vs 3 of 8 on Ollama. Every miss traces to the 8b
test-writer (tests with no import, self-contradicting expectations);
the gate refuses honestly, and 2 of 10 shipped oracle-failing code.
This is Next up 2's owed #90 gate, now measured for T1.

### Merged on local main this session

- **#90 — Ollama dropped; llama-server router owned by the serve**
  (PR #186, released v0.20.0; suite 4322 passed, lint 0, CI green). Evidence on the issue (spike table 2026-09-16). Shape:
  `OpenAICompatibleModel` carries `options` (`think` lifted into
  `chat_template_kwargs.enable_thinking`, 60x on qwen3), sampling
  passthrough, `response_format`, and llama-server `timings` under the
  `prompt_eval_count`/`eval_count` keys the truncation backstop reads;
  provider `llama-server` (local default, legacy fallback target;
  `LLAMA_SERVER_URL`); the serve renders `.llm-orc/llama-server.ini`
  from every llama-server profile (one section per model, `hf_repo:`
  source, `c = 40960` = `turn_trace.WINDOW`, one resident model) and
  supervises one router (`--no-backend` opts out; a router that exits
  before listening surfaces its stderr tail; SIGTERM to the serve stops
  the router — uvicorn re-raises captured signals after restoring
  handlers, so a `finally` never runs); `GET /api/models`,
  `POST /api/models/{name}/pull` (blocks until the router's real status);
  provider status / runnability / promotion read the router; the agent
  config field is `response_format` (was `ollama_format`); profiles are
  `local-qwen3-*`; model names carry no colon (the router rewrites
  `name:tag`) and no slash (raw cache entries). Verified live on the
  laptop: serve owning a real router, five preset models listed, pull
  blocked ~4 s to `loaded`, factory completion with timings, router gone
  on SIGTERM. Tests may not launch the real binary (autouse guard, shown
  red on two leaking serve tests before the fix).
- **Deploy scaffolding:** Dokku app `llm-orc` deployed
  (`~/Development/llm-orc-proxy`, nginx → `host.lima.internal:8765`;
  edge nginx set to 3600 s / buffering off / 50 m body);
  http://llm-orc.remote.example answers 502 until the serve exists on
  remote-host. The launchd unit and setup notes live outside this repo.
- **#202 (2026-09-22, PR #203 merged as `a5c4bc7d`, released v0.20.6,
  formula bumped).** Core-engine fix outside the serving track but landing
  on it: `ensemble:`/`loop:`/`dispatch:` children now share one input
  contract (Option B on the issue: `input_key` value verbatim; otherwise
  base input plus dependency blocks, no LLM instruction sentences). Three
  live seats see a different prompt as a result: `build-gated-round.
  code_writer` and `re-fix.model_edit` lose the chain-member framing and
  keep their upstream; `gen-review.review` now sees `gen` (it never had).
  Fan-out instances past phase 0 now frame the raw ensemble input
  (previously empty). The ladder T1 re-run was skipped for this change
  by practitioner decision; the regression gate below therefore also
  measures these three seats under the new input shape. Not deployed to
  remote-host yet (brew upgrade + `homelab doctor --fix`).

### Evidence carried forward (2026-09-12)

Comparator row on the existing-repo probe: Go `qwen3.8-max` 7/7 ($0.27),
Sonnet 5 7/7 ($0.38) vs the serve 1/7. Ladder runs 7-8 on the merged
arcs: 10/13, 8/13; T1 decides the ladder (3 correct / 2 broken / 3 refused
over eight runs; refusals pre-date the arcs). Neither has been re-run on
the llama-server backend yet — that is the regression gate below.

### Next up (in order)

**First (practitioner, 2026-09-28): remote delegation.** Plan with task
cards: `docs/plans/2026-09-28-remote-delegation.md` (#196, #191);
implementation plan for Arc 0 + Arc 1:
`docs/plans/2026-09-28-remote-delegation-arc0-arc1.md`. **Arc 0 and Arc 1
merged on local main 2026-09-28 (`c75daed8`), pushed 2026-09-29 (`1e5cfc02`, CI green).** S1: the
packaged serving project can be a read-only layer; artifacts,
agentic-sessions, serve-trace, the rendered preset and the CRUD write
fallback move to a state dir; `get_model_profiles` has no layer list
(rewrite, not an insertion); `project_dir` means the checkout root in
`ConfigurationManager` but the dot-dir in `ServingEnsembleCaller`. S2: a
newly shipped profile is `needs_restart` (router loads only what it
scanned at start; supervised restart 1.5 to 2.2 s, hard-cuts in-flight
completions). Arc 1: `scope: project | global` on ensemble, profile and
script CRUD over REST, MCP and the handlers; `project` requires a project
tier (no fallback to library or global); update/delete act on one scope
and name where the file lives; global scripts resolve at execution time;
MCP schema advertises the enum. Pins through the real service and files
on disk, each shown red under a mutant; suite 4735; whole-branch
adversarial review plus two scoped re-reviews. Live row in the plan doc.
`research-dossier` on the mini is global now; `test-research-pipeline`
still lives in the checkout. v0.21.0 is deployed on remote-host
(practitioner, 2026-09-28).

**Arc 2 merged on local main 2026-09-29 (`1dd75100`; released in v0.22.0).**
Re-cut rulings in the spec doc ("Arc 2 re-cut (2026-09-29)"); plan
`docs/plans/2026-09-29-remote-delegation-arc2.md`; summary and findings
posted on #196 and #191. Shape: the repo's `.llm-orc/` ships in the wheel
as `llm_orc/serving_project/` (hatchling `include` + `sources`;
`make wheel-check` pins the 162 packaged files to `git ls-files`, red on
the 0.21.0 wheel); `ConfigurationManager` gets a read-only fourth tier,
packaged, lowest precedence (project → library → global → packaged) in
every read path: dir lists, runtime profiles as a tier loop (library
stays out of runtime resolution, pinned), the `performance:` /
`agentic_serving:` merges, `classify_tier` (`source: packaged`, fourth
CLI group), the script resolver, the executor's child-ensemble lookup
(a global ensemble reaches a packaged child), and `serving_root()`
replacing the cwd lookup in `/v1/chat/completions`. Packaged is lowest so
a global `*.local.yaml` re-seats a tier on a plain-dir serve.
`LLM_ORC_SERVING_PROJECT_DIR` overrides the tier (empty disables; the
suite's default). State (artifacts, `.serve-trace`, cache,
`llama-server.ini`, script bytecode) goes through `resolve_state_dir`:
`LLM_ORC_STATE_DIR` / `serve --state-dir`, else the project's `.llm-orc`,
else `$XDG_STATE_HOME/llm-orc`. Engine writes with no project land in
global; after `set_project` the executor takes the service's config
manager. Suite 4786; eight per-task adversarial reviews, a whole-branch
review, one fix wave (bytecode under the packaged tier; cut-over docs).
Live rows (plan doc): wheel in a fresh venv, serve from an empty dir;
`/v1/models` = the packaged tier; 105 ensembles {global, packaged};
`research-dossier` (global) → packaged `web-searcher` ×5 → dossier in
87 s, artifacts in the state dir; a real `opencode run` turn drove three
rounds to an honest T1 refusal with 0 bytecode files under the packaged
path. **Owed then, done 2026-10-01:** mini cut-over (move `*.local.yaml`
overrides to
`~/.config/llm-orc/`, `~/.llm-orc` absent, plist `WorkingDirectory` +
`brew upgrade` in one kickstart; the 15 library ensembles drop out unless
`LLM_ORC_LIBRARY_PATH` names the submodule). Released as v0.22.0 on
2026-10-01; the wheel check runs in CI. **Next:** re-cut Arc 4 (one-run injection) from the merged Arc 3 shape:
the `not_equipped` error carries the Arc 3 `dependencies` list.

**Arc 3 merged on local main 2026-09-29 (`bf3d7287`; released in v0.22.0).**
Rulings in the spec doc ("Arc 3 re-cut (2026-09-29)", amended by the
review round); plan `docs/plans/2026-09-29-remote-delegation-arc3.md`; live
row in the spec. Shape: `check_ensemble_runnable` (REST
`GET /api/ensembles/{name}/runnable`, MCP) walks the closure (child
ensembles via `ensemble:`/`loop.body`/literal `dispatch:`, scripts, profiles
with the `model_profile` chain transitive and the agent-level fallback one
hop as at run time, inline models) and adds `dependencies` (kind, name,
`via` frames, one of eleven statuses, resolve hint, detail) next to the
kept `runnable` and `agents` (`dependency_unmet` added). `runnable` means
every dependency `ready` or `dynamic`; `pullable` blocks. Model presence is
the router's one listing: preset ids routable, cache ids (= `hf_repo`)
downloaded, `status: loaded` live (`LlamaServerClient.inventory()`;
provider status carries `cached` and `loaded`). Preflight resolves exactly
as the run does: profiles from `get_model_profiles()` (runtime tiers,
library excluded), children through the executor's search-dir list
(`child_ensemble_search_dirs`, one shared function) and
`_find_ensemble_in_dirs`; the walker keys on reference strings. Suite 4836;
five per-task reviews, a whole-branch review that found three wrong-accepts
(profile source, child finder, walker keys) closed in one fix wave with
parity pins through the real service and executor, one scoped re-review.
Live row (laptop, llama-server 9850): one of each status observed;
`needs_restart` before a restart and `pullable` after; a pull (2 min 21 s,
1 GB) left the model `pullable` until the loaded fix, now `ready` on
`loaded`. **Owed:** the mini cut-over (the release is out: `brew upgrade`).
Known limit: a model pulled in a
running serve then evicted by the router reads `pullable` again until the
next restart (the hint still just loads it); the serve remembering its own
pulls is a follow-up on #196.

Deferred from the Arc 3 reviews (none block): the router listing carries
each preset model's `--hf-repo` in `args`, so inline models and sourceless
profiles could be classified `pullable` without a profile source; a child
file that exists but fails to load raises out of preflight (as the run
would) rather than a row; status parsing lives in `llama_server.model_status`
and `web/api/models.py::_entry`; `_DEFAULT_OPENAI_BASE_URL` and the
openai-compatible predicate are duplicated in `preflight.py`; the frontend
`RunnableStatus` type still says `model_not_found` and renders
`dependency_unmet` bare; promotion readiness re-derives model presence from
`models` without `cached`/`loaded`; an LLM agent with a ready primary and a
missing fallback reads `missing_profile` naming the primary; fallback hops
block `runnable` by ruling (conservative refusal; Arc 4's `bind` resolves).

Deferred from the Arc 2 reviews (none block; 40-odd minors in the review
record, the ones worth a line): `check_wheel_contents.py` run from a
subdirectory passes a wheel with no serving project (Makefile runs it
from the root); `resolve_state_dir(local)/"artifacts"` is spelled at
seven sites; `_resolve_ensemble_reference`'s local-dir block is redundant
with `get_ensembles_dirs()[0]`; the `serving_root` error prints `None` for
a missing candidate; relative `--state-dir` is stored as-is; primitive
and `test_script` subprocesses still spawn with a bare env (no pycache
prefix); the packaged `config.yaml` ships verbatim (test and demo
profiles included), a curation chore for later; a checkout-cwd serve on a
wheel that carries `serving_project` lists everything twice until the
`WorkingDirectory` changes, hence the one-kickstart cut-over.

Deferred from the Arc 1 reviews (none block; pick up when touching the
area): `_dir_for_scope` / `_find_*` are duplicated across the three CRUD
handlers (a `refactor:` tidy once Arc 2 settles the packaged tier); the
REST scope-mismatch pins assert `!= 200` until the ValueError-to-500
mapping becomes a 4xx (#191); `ScriptResolver.list_available_scripts` is
cwd-only and ignores `project_dir`; `ScriptHandler._get_scripts_dir`'s
docstring undersells its "config manager present, no project tier"
branch; `tests/bdd/test_adr_009_mcp_server_architecture.py` builds a
MagicMock config manager and asserts only returned dicts; FastMCP and the
REST bodies ignore unknown keys, so a misspelled `scope` silently means
`project` (#191). Rulings that shaped Arc 1, for the record: `project`
scope errors without a project tier rather than falling through to
library or global (option A over "land on global"); PUT carries `scope`
in the body and DELETE in the query, never both; the MCP `scope`
parameter is the `Scope` literal so `/mcp` advertises the enum.

1. **Serve is up on remote-host (2026-09-16 afternoon).** launchd agent,
   v0.20.0 from origin main, llama.cpp b10964 x64 binary (no Intel brew
   bottles; the upstream binary instead). http://llm-orc.remote.example
   answers 200; four tiers pulled over the tailnet; acceptance 1 and 2 of
   the handoff pass (tool call parsed through the tailnet URL). Measured
   on the box (i7-8700B, CPU-only): qwen3-8b 23 prompt tok/s, 4-6 gen
   tok/s; the trivial write_file turn took 560 s end to end because the
   serving ensemble ran classify + build-gated round + tests at that
   speed, and classified "create a text file" as a python_module build.
   Both numbers feed the gate below. HTTPS is on (practitioner's go):
   the homelab wildcard cert had expired 2026-08-23; renewed, valid to
   2026-12-15, `https://llm-orc.remote.example` answers 200 and http
   redirects. The weekly renewal daemon has been failing (root PATH lacks
   `certbot`), a homelab-repo fix owed to the practitioner before
   December. Full record: `docs/plans/2026-09-16-remote-serve-handoff.md`.
   **#194 merged on local main (2026-09-16 evening):** the serve exposes
   the full MCP tool set at `/mcp` (streamable HTTP, one `OrchestraService`
   behind REST, `/v1` and MCP; DNS-rebinding guard off for the proxied
   host; design `docs/plans/2026-09-16-mcp-in-serve.md`). Suite 4328,
   lint clean, independent review found one wrong-accept (fixed, pinned
   red/green). A user-level MCP entry on the laptop points at it.
   Deployed and accepted on the rig (design doc, Result). **Released
   as v0.20.1** on the practitioner's go (push, PyPI, GitHub release,
   formula bump). Policy from the practitioner: the mini runs releases,
   not checkouts. Doing that surfaced #197 (P0, closed the same
   evening): a clean install of 0.20.1 resolved `mcp` 2.x and died at
   import (the lock hid it). **v0.20.2** pins `mcp<2` (clean resolve
   verified from PyPI). The brew formula could not build on the Intel
   mini (cryptography>=49 has no x86_64 wheel; rust has no Intel bottle
   either, hours from source); the tap formula now constrains
   `cryptography<49` on Intel only, with the reasoning (PYSEC-2026-3552
   is a PKCS#7 decryption oracle; llm-orc uses only Fernet). **The mini
   runs `brew install llm-orchestra` 0.20.2** (30 s build), health and
   `/mcp` verified over https; the interim `uv tool` install is removed.
   Lesson, binding: a release is verified by a clean install from PyPI,
   never from the checkout. #196: the serving project (`.llm-orc/`) is
   not in the wheel, so the plist's WorkingDirectory stays the checkout.
   Follow-up #195. The practitioner may move the mini to Linux; Intel
   macOS support is dwindling upstream.
   **#198 embeddings on the serve, released v0.20.3 and deployed the
   same evening** (driver: the svalbard vault skills; aligned with that
   session over messages). `POST /v1/embeddings` forwards to the router;
   `options.embeddings`/`pooling` render into the preset, and an
   embedding model gets `batch-size`/`ubatch-size` = its context. Review
   found two blockers the design's acceptance would have missed: the
   handler blocked the event loop (7 s `/health` behind an 8 s stub;
   fixed, now a threadpool handler, pinned) and llama-server's 512-token
   physical batch refused any real note (fixed, pinned). Measured on the
   mini over https: 32 x ~1.2k-char texts (9,504 tokens) in 10.4 s with
   `/health` under 110 ms throughout; two seats resident
   (`--models-max 2`, needs bootout+bootstrap, a kickstart keeps the old
   argv). Follow-ups #199 (pull blocks the loop; route should ask the
   router for its model list).
2. **Regression gate on the new backend** (laptop or remote-host): the ladder
   (T1 alone at r≥5 first, then the full run) and the 7-turn probe.
   Chat templating and tool-call parsing moved from Ollama's Go templates
   to llama.cpp's jinja; the spike showed tool calls parse and thinking
   is controllable, the ladder decides whether the seats hold. Then
   OpenCode against `https://llm-orc.remote.example/v1`.
3. **#123, #122, verified acceptance in the workspace** — unchanged from
   2026-09-12; the T1 instrument still precedes touching the build round.
4. **Filed 2026-09-16, sequenced after the remote-host bring-up and the
   backend gate:** #189 code-footprint epic (#190 wire-or-delete six
   unwired modules, P1 because AS-2 names a validator nothing runs;
   #191 one service with generated MCP/REST adapters and a thin CLI;
   #192 CLI/display re-approach on one event stream; #193 spike:
   meta-simulation as a client-side ensemble, retire the interactive
   input mode if it covers the cases). Then #187 strangler-kernel epic
   with #188 as its spike (port the router supervisor and the
   OpenAI-compatible client to a Rust crate in the plexus workspace;
   exit criterion on the issue). Auth and promotion stay as they are.
5. Follow-ups filed by this arc: `llm-orc web` does not own the router
   (opt-in later); MCP over the tailnet (the SSE transport is the older
   protocol, a second launchd unit + proxy path); deepseek-r1's template
   ignores `enable_thinking` (fine for the reasoning tier); the library
   submodule's templates still say `provider: ollama`.
6. Gated on the practitioner: further pushes; #167/#141 (Anthropic arms).

### Owed live rows

#166 #169 #173 #171 (constructed non-participating shape via fault
injection), #172 #176, #175. Arcs 1-2, D, B-1 have theirs. #90's live
row is the regression gate in Next up 2.

### Process (agreed 2026-09-11, holding)

Reviews found real blockers 12 of 12 first rounds through 2026-09-12;
with the three self-tests required in the implementer brief, arcs 1 and
2 each closed in ONE rework round. Lessons paid for: a moving git ref in
a pin makes it vacuous after merge (pin to a hash — acfaf427); a
"harmless internal convention" in a gate field is never internal; a
fuzzy match that reads then overwrites the wrong file is worse than an
honest ask; existence is a workspace fact, never a verb. 2026-09-16: the
first real run of a new process-owning feature found four defects unit
tests could not (signal re-raise, list hygiene, async load, swallowed
stderr) — run the real thing before calling a lifecycle feature done.
Ops: subagent budget ~4-5M tokens/day before the limit; Sonnet
implementers, Opus reviews; ladder ~17 min at coder-on, probe ~10 min;
one serve per tree, restart before every gate; the 8b GGUF is a 5 GB
first download per box.

## Timeline

Done:

- [x] v0.18.0 — agentic serving backend (declarative ensemble behind OpenCode)
- [x] v0.18.1–0.18.7 — session record #99, TDD retry #100, write-tests #98, client reads (#83), per-test gate isolation
- [x] v0.18.8–0.18.12 — run delegation (#83), fenced block grammar, gate repairs, discovery glob (#83), fix-execution (#115), #107
- [x] v0.18.13 — convergent fix (#117 rungs 1.5+2), grounded explain (#118)
- [x] v0.18.14 — chain executor (#120), deep recall (#82 core)
- [x] Meta-task slice 1 — bare-symbol glob→read grounded explain
- [x] WS-8 instrument (#131 Arcs A–D) — battery, hidden oracles, frozen rubric, hashed manifests; 5-round review gate
- [x] Arm-0 column n=3; first parity table published
- [x] #133 #134 — honesty classes closed, live-validated at 0 dishonest (run 5)
- [x] Arm-2 adapter + automatic scoring (#131); Haiku run 2 scored (2 dishonest — frontier n=1 ceiling broken)
- [x] Battery precondition guards; CI fixes (twine metadata, pyasn1); dogfood channel (`docs/dogfood-log.md`)

Remaining, in order:

- [x] #90 — Ollama dropped; the serve owns a llama-server router (2026-09-16)
- [x] #131 — Arm-2 runs at n=3 per model, all independently J-scored (Haiku 35/39, 4 dishonest; Sonnet 39/39, 0)
- [x] #131 — Arm-1 GO'd + n=3 per model, independently J-scored (Haiku 38/39, 0 dishonest; Sonnet 39/39, 0); #147 filed
- [x] #146 — v0.18.15 RELEASED 2026-08-13 (PyPI + Homebrew green)
- [x] #148 — truncated-listing refuse merged + live-validated (3 review
  rounds; run-6 validation row); #149 filed (client-side flank)
- [x] #143 — CLOSED as an honest miss: component-subset refuted at
  class level; model gate refuted empirically (pre-registered bar,
  8b+14b, all variants); reopen rides #119 with the committed spike
- [x] #139 — context curve measured (flat through 32K recall / 24K
  synthesis; 4KB cap defensible; latency is the binding constraint)
- [x] #145 — repo-scale reads merged + live-validated (96KB cap,
  token-denominated read budget, runtime truncation backstop, #150
  fixed; five review rounds; dogfood entry 1 converted). classify.py
  refuses over-budget by design → chunked reads deferred; #151 open
- [x] #144 slice — serve-native dot-dir self-reference merged + live-
  validated (grounded self-reference for budget-fitting scripts; the
  literal classify-grounded exit stays open on the whale, riding
  #106/#151/chunked reads); #152 fixed+merged (routing fails closed)
- [x] #121 slice A — content-grep rung merged + live-validated (exit
  gate met: grep→menu→pick→AST-confirmed grounding); coverage bounds
  ride #153 (offset reads) and the truncated-listing trigger gate;
  #153 filed+fixed (client 50KB read cap refuses honestly)
- [x] #63 slice + #138 INSTRUMENT — statistics (Wilson/Fisher) and the
  volume-ladder instrument built, merged, and calibrated on arm 0
- [x] v0.19.0 released — script cache keyed on BYTES and shipped disabled
  (#160), instruments in the gate (#156), pipeline positive-completeness
  (#155 Arc A), plus #152 #154 #157 #158 #159 #164
- [ ] #167 volume paid runs (r=8 per level; awaiting practitioner go on
  cost) and #141 (awaiting go on the None condition), then parity table
  v2. #138 is CLOSED — it tracked the INSTRUMENT, which shipped; #167 is
  the spike itself.
- [ ] #126 — long-horizon 30-turn battery (#136 #137 feed the design)
- [ ] #140 memory spike (#139 closed: context curve measured); #127
  plexus substrate; #82 remainder (cross-session)
- [ ] #128 #129 #130 — task shapes toward compose-at-runtime
- [ ] #125 — Rust gate; #119 #135 — seat ladder on-signal
- [ ] #85 #84 #90 #93 #95 #106 #110 #114 #132 #142 — platform hardening as gates demand
- [x] #166 #169 #170 merged (empty deliverable never written; empty re-fix
  candidate never accepted; `make test` independent of the working
  directory), plus loop-protocol 13-19 and the doc-drift check
- [x] #163 #168 merged after confirmation rounds (#178 taken in-arc:
  the parser is deleted); #173 merged (inert class closed 15/15); #179
  filed+fixed+merged (drift check works in worktrees)
- [x] #172 #176 merged after four rounds (runner-namespace judgment, library
  source resolution rebuilt: explicit env path trusted on existence, cwd and
  packaged candidates content-gated non-empty, `("local","")` never escapes)
- [x] #175 env-scrub slice merged (empty child env, census pinned, bounds
  named); the vocabulary half stays open on #175, riding #180/#142
- [x] #174 merged (a dead seat refuses honestly on every route; dead-seat
  refusal outranks accept/validity; stderr retained head+tail capped)
- [x] #177 merged (one resolve-and-classify observation; bare CWD refs
  anchored so PATH cannot re-resolve; identity names bytes AND location)
- [x] #171 merged (ablation control + public-binding re-fix surface; four rounds)
- [x] #182 D + B-1 merged (discovery before an unnamed build; edits keep the prior surface); #183 B measured (coder keeps thinking)
- [x] Arc 1 workspace-aware routing (#185) and arc 2 the sandbox mirrors the workspace (#184, #182 A) merged 2026-09-12
- [ ] #123 code+tests per turn (+ tests destination); #122 edit delegation; verified acceptance in the workspace; #183 A/C; #181
- [ ] #161 #162 #165 — script-cache purity/imports and the -n auto flake;
  #155 Arcs B/C remainder
- [ ] North star: parity on real work, honesty column held at zero

## Doctrine (what we learned, made binding)

Rules future sessions follow without re-deriving. Each was paid for with a
measurement; provenance in the archive.

1. **Independent scoring for every judgment-bearing claim.** Author scores
   were systematically optimistic (three runs' "zero dishonest" all
   overturned). Any J-tier score, honesty verdict, or review APPROVE comes
   from a session/agent that did not author the work. Blinding is inert
   (Arm 0's prose is templated); independence plus the frozen rubric is
   the control that works.
2. **Structural lever after two prompt iterations.** Prompt rules saturate
   the 8b seat (measured twice); when a failure class survives two prompt
   changes, reach for determinism, shape change, or escalation-on-signal —
   never a third rule.
3. **Structure beats model size — but re-measure per era.** Deterministic
   gate repairs took the ladder 4/10 → 7/10 where {8b, 14b} × {think
   on/off} were identical; the 14b test-writer A/B was not a clean win.
   The doctrine goes stale as the structural breakers are removed; the
   seat ladder (#119) exists to re-test it, ≥3 runs per seat.
4. **State the invariant, not the instance.** In five review rounds,
   everything mechanically checkable held; everything patched
   instance-by-instance failed again until stated as an invariant (the
   dead-turn rule, the equality-pins-representation bug fixed in one
   oracle and left live in the next).
5. **No self-confirming metrics.** The verification-rate metric is
   WITHDRAWN, not deferred: it read a design constant on Arm 0 and a
   behavior on Arms 1/2. Crediting the serve from its own trace is
   circular. Ground truth is the WORKSPACE, never any transcript.
6. **Per-turn diagnosis is unsupported at current n.** Misses are noise
   around a rate (~5 points ride on turn 1); only aggregate rates are
   estimable. Run 2 falsified the turn-1-cascade claim. #63's statistics
   become relevant as n grows.
7. **The headline is the 2x2, never a raw count.** Raw counts have a
   degenerate optimum at refusing everything — the serve's own failure
   mode. Primary figure: `shipped_broken/shipped`, delivery beside it.
8. **Real-client validation at the earliest runnable point, never
   harness-only.** Hermetic green is necessary, not sufficient; every
   capability arc ends with a live battery row.
9. **Determinism for answers; model judgment only in bounded, low-risk,
   gate-backstopped routing; honesty-critical paths fail closed** (the #82
   two-layer split is the worked example).
10. **Free-first; estimate before paid spend; hosted seats are named IOUs
    in the buy-back ledger.**
11. **Pin the harm, not the mechanism.** Three of #177's four blockers
    were green pins measuring the code path a fix had just edited (a
    trap pin on the one shape where it held for free, an
    `assert_not_called` on the edited branch, identity pins checking
    only the digest suffix) while the defect ran a PATH impostor or
    crossed a cache. A pin must assert the OUTCOME — which program ran,
    which output crossed, what reached the wire — and go red under a
    mutant that reintroduces the defect it was written for.

## Environments (tag every task)

- **RIG** — the 32GB Ollama rig with OpenCode: live batteries, Arm-0/Arm-1
  runs, seat A/Bs, latency data, plexus (local sibling repo). Ops notes:
  batteries run detached (nohup + disown, Monitor tail); `opencode run`
  wedges under the agent Bash sandbox (see memory `opencode-run-wedge`);
  cooling headroom between batteries.
- **ANY** — any session including remote containers: hermetic TDD against
  the full suite (`uv` + Python 3.11 suffice), design docs, scorer/oracle/
  adapter code, reviews, doc work. RIG-tagged validation of ANY-developed
  work is queued, not skipped: the PR says "needs rig battery" and the next
  rig session runs it.
- **REMOTE** — a remote Claude Code session specifically: **Arm-2 battery
  runs** (subagent model overrides + continuation), GitHub issue hygiene,
  independent J-scoring and reviews (a fresh remote session is naturally
  author-independent).

## Epics

Issue lists live in GitHub: `gh issue list --label epic:<name>`. One line
per epic here; detail on the issues.

### epic:ws2-honesty — CLOSED 2026-08-12; hold the property
Zero dishonest under independent scoring, validated live. Watch: #140
(staleness cascades). Reopen only on a new independently-confirmed
dishonest outcome.

### epic:ws8-parity — the comparison IS the product claim
- [x] Instrument + Arm-0 column + parity table v1 + Arm-2 auto-scoring (#131)
- [x] #131 Arm-2 n=3; Arm-1 go/no-go + runs n=3 (all columns complete)
- [ ] #147 `_PASS_CLAIM_RE` false-positive family (three refuted captures)
- [ ] #141 CLAUDE.md confound · #167 volume-scaling paid runs
  (successor to the closed #138, which tracked the instrument) · #63
  statistics
- [ ] Parity table v2 (realism rows; interval estimates)

### epic:ws3-client-surface — the serve upgrades any client
- [x] read/run/discovery delegation, fix-execution, chain executor, meta-task slice 1
- [x] #143 closed (honest miss; reopen rides #119) · #145 repo-scale reads merged+live · #148 #150 read-seam hardening
- [x] #144 slice — serve-native dot-dir self-reference merged+live (grounded exit rides #106/#151/chunked reads)
- [x] #121 slice A — content-grep rung merged+live (exit gate met; bounds ride #153)
- [x] #153 offset reads merged+live (50–96KB grounding restored; the
  #121 bound converted); remaining flanks recorded on the issue
- [ ] #149 client-side truncation flank
- [ ] #122 edit delegation · #123 multi-file · #124 command registry · #117 fix-completion tail

### epic:ws5-long-horizon
- [ ] #126 30-turn battery + plan substrate (#136 #137 feed design)

### epic:ws6-memory
- [x] #139 context curve (flat through 32K recall / 24K synthesis; latency binds)
- [ ] #127 plexus integration · #82 remainder (cross-session record)

### epic:ws7-task-shapes
- [ ] #128 elicit-then-build · #129 refactor shape · #130 compose-at-runtime primitive

### epic:ws4-language
- [ ] #125 Rust gate (cargo runner, sandboxed executor, adequacy)

### epic:seat-ladder
- [ ] #119 escalation framework · #135 ori/eval A/B harness spike

### epic:ws9-platform
- [x] #146 v0.18.15 released
- [x] #152 fail-closed routing merged (readability-gated decisions; the misfire class refuses)
- [x] #154 interpreter PATH fragility fixed (.py runs under llm-orc's own interpreter; bare-PATH suite 50 failures -> 0)
- [x] #157 script-agent timeouts fixed (one resolver for inner and outer bounds; six nodes declare their own)
- [x] #158 script agents off the event loop (4.15s -> 1.03s; duplicate outer timeout retired)
- [x] #159 failure envelopes no longer cached (two-clause predicate; the corpus emits no boolean success)
- [x] #160 cache identity is the script's BYTES, and the cache ships DISABLED (two of six primitives are impure; a hit elided a write) — v0.19.0
- [x] #156 the 511 measurement instruments run in `make test` and CI (a regression in them used to corrupt evidence without failing a build)
- [x] #164 script-agent-architecture documents the cache that exists (the per-agent `cache:` key it showed is rejected by `extra="forbid"`)
- [x] #166 an empty build deliverable is never a client write (the fault was
  a LIVE seat with an empty artifact, not the dead seat the issue named)
- [x] #169 an empty re-fix candidate is never accepted, at source
- [x] #170 `make test` no longer depends on the ambient working directory
- [x] #155 Arc A — a node that cannot READ its input refuses (crashed shape/form_gate finished as an empty success); Arcs B/C still open
- [x] #163 an undigestable script is never cached (7 rounds + confirmation)
- [x] #168 a refusal reason names no absolute path (8 rounds + confirmation;
  #178 taken in-arc — the TestCase branch no longer parses tracebacks)
- [x] #173 an inert re-fix candidate is never accepted (closed AST whitelist)
- [x] #179 the doc-drift check knows its names inside agent worktrees
- [x] #172 #176 — an empty submodule is not a library; the gate's verdict
  comes only from tests the runner executes (four rounds)
- [ ] #151 runtime-window detector remainder · #155 Arcs B/C · #174 a dead seat ships the engine envelope as the answer · #175 vocabulary half (env route CLOSED; path-free literals still reach the wire) · #177 three file-vs-inline classifiers · #180 the accept report is unbounded · #161 cache purity · #162 cache misses imports · #165 `-n auto` flake · #85 sandbox hardening · #84 gate adversarial harness · #90 llama.cpp · #93 hot path · #95 dead surface · #106 shape home · #110 artifact quality · #114 trace cap · #132 BitNet · #142 reject templates

### epic:off-path
#80 #65 #30 #66 — parked, not on the north-star path.

## Delegation contract (updated for non-Fable sessions)

Every arc: short design doc (`docs/plans/YYYY-MM-DD-*.md`) → TDD
implementation → live real-OpenCode validation at the earliest runnable
point (RIG; queued explicitly if the implementing session lacks the rig)
→ ladder rerun + a row appended to the archive's trajectory table →
**author-independent adversarial review with an explicit wrong-accept hunt
before merge** (the review record is five-for-five on finding real
blockers the author missed).

Session roles:
- **Lead (Opus-class):** designs, reviews, sequencing decisions, this
  document. On entry: read §State, §Doctrine, and the card being worked —
  not the archive.
- **Implementer (Sonnet-class):** one task card per arc, TDD, hermetic
  suite green, PR notes any queued RIG validation.
- **Mechanical (Haiku-class):** #95-grade sweeps, doc syncs, battery
  bookkeeping, table updates from existing records.
- **Scoring/review independence (doctrine 1):** J-scores and review
  APPROVEs come from a session or agent that did not author the work; a
  fresh remote session is naturally independent. The frozen rubric
  governs; corrections are amended in the rubric, never edited away.

## Standing constraints

- 32GB rig is the permanent target; interactive latency is first-class.
- Local-first defaults; hosted seats are operator opt-in, never tracked
  (`*.local.yaml`), each carried in the buy-back ledger.
- Real-client validation at the earliest runnable point; every capability
  stage lands with its ladder rerun and a trajectory row (archive table).
- Deterministic control; model judgment only inside bounded, closed-set,
  gate-backstopped decisions; honesty-critical paths fail closed.
- Ground truth is the workspace; independent scoring for judgment claims.

## Issue index

Superseded by epic labels: `gh issue list --label epic:<name>`. Closed
2026-07-11: #31 #78 #79 #64. Closed since: #83 #98 #99 #100 #104 #105
#107–#109 #111–#113 #115 #116 #118 #120 #133 #134 #138 #139 #145 #152
#153 #154 #156 #157 #158 #159 #160 #164.

**#163 #166 #168 #169 #170 #172 #173 #174 #176 #177 #178 #179 are merged
on local main and still OPEN on GitHub**, because nothing is pushed. They close when
the push lands, not before — the roadmap said "closed" here first, which is
the kind of claim rule 15 exists for. #175 is merged in part (env slice)
and stays open by design for its vocabulary half.

Filed 2026-08-30 from review findings, all measured, none previously
tracked: #171 #172 #173 #174 #175 #176 #177 #178, then #179 (fixed+merged)
and #180 (open) in the evening session. #172 and #176 merged after four
review rounds.

Two closed issues gate work that is tracked elsewhere: **#138**
(instrument shipped; the paid runs are #167) and **#139** (curve
measured; #140 carries the remainder).

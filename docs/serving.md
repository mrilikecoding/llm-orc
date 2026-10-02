# Agentic Serving

llm-orc serves as the backend for agentic coding tools (OpenCode, Aider, Cline,
Cursor — any OpenAI-compatible client) via `llm-orc serve`, which exposes
`/v1/models` and `/v1/chat/completions`.

Every request to `/v1/chat/completions` is handled by **one declarative
Serving Ensemble** — classify (route the turn) → seat (dispatch to the resolved
capability) → marshal (shape the deliverable, gate its validity, emit it) —
executed by the same L0 Ensemble Engine that runs any llm-orc ensemble, on
primitives that ship with the engine (guard/branch, bounded loop, dynamic
dispatch). There is no persistent internal orchestrator: the client owns the
multi-turn agentic loop, and each request is a single declarative pass.

The north star is **full model parity through composition** — the endpoint
should do everything a single model does behind a coding tool (explain, fix,
edit, run tests, build) with no capability loss, and composing ensembles should
widen what's possible, never narrow it.

## The per-turn pipeline

Defined in `.llm-orc/ensembles/agentic-serving/serving.yaml`; scripts in
`.llm-orc/scripts/agentic_serving/`.

| Node | Kind | Responsibility |
|------|------|----------------|
| `classify` | script | Routing decision `{target, kind, file, dispatch_input, build}`. Deterministic where the signal is structural (explain markers, build verbs, named files); emits `needs_decider` when not. |
| `decide` | model (guarded) | Runs only on the ambiguous path (`when: ${classify.needs_decider}`). Picks a target from a closed seat set. The model classifies; the control stays deterministic. |
| `resolve` | script | Merges the structural and model-backed decisions into the final routing. |
| `seat` | dynamic dispatch | Resolves `${resolve.target}` at the phase layer and runs the resolved capability ensemble as a child, passing it the clean turn. |
| `seat_contract` | script | Admits or rejects the seat's output against the resolved seat's own `seat_contract:` block via a wired `ValidationEvaluator` (seat-owned, black-box, deterministic-first). |
| `shape` | script | Reads the deliverable faithfully from the seat's I/O envelope (ADR-024) — content from the envelope, destination from the routing decision. |
| `form_gate` | script | Deterministic destination-validity check: refuses a deliverable that does not parse as its path claims (a `.py` must parse, a `.json` must load). |
| `emit` | script | Shapes the client-seam outcome: a `write` tool_call (`finish_reason: tool_calls`) for a valid build, a prose finish otherwise. |

`resolve`, `shape`, `form_gate`, and `emit` are each marked
`on_dependency_failure: run` (Invariant 13/14): a crashed `classify`, `seat`,
or any node earlier in this chain would otherwise cascade-skip everything
downstream, and the turn would lose its refusal along with the failure it
exists to report. Each reads its own crashed dependency defensively and
composes an honest `"Refused: serving pipeline error: ..."` instead of
vanishing. The top-level `status`/`has_errors` a client or operator reads
still say `error` — the CAUSE is `has_errors` (the crashed `classify`/`seat`
node is genuinely `failed`, and that alone makes `has_errors` true regardless
of what runs after it), not the `handled_failure` marking on the node that
ran to compose the refusal. `handled_failure` is a separate fact about that
node's own `outcome` (addendum 2026-09-23): it keeps the composed refusal
from being misread as a *succeeded* terminal by anything that consults it —
without it, a wrapping node could report success just because the handler
ran without incident, masking the real upstream failure. So the client's
HTTP response body carries real refusal content, while the top-level status
still honestly says `error`: two separate reads of the same underlying
failure, not one causing the other. See `docs/domain-model.md` Invariant 13
and the `on_dependency_failure` glossary entry for the general contract;
this is serving's own instance of it.

A `loop:` node's `has_errors` (round-4 amendment, 2026-09-23) reflects only
its FINAL iteration: earlier iterations are retry attempts the loop exists
to absorb, so a build-gated round that fails once and passes on retry must
not report `error` — the loop's own outcome is `succeeded`, has_errors is
`false`. Earlier iterations' blocking outcomes are never dropped, though:
each one lands in the loop's JSON response as `iteration_failures`, a list
of `{iteration, error}` entries naming which terminal failed and why. An
operator reading `turns.jsonl` sees both facts — the turn succeeded, and it
took more than one try.

Build turns route to the **gated build shape** (`build-gated.yaml`): test-writer
→ code-writer → deterministic executor (runs code + tests, sandboxed
subprocess) → isolated adequacy judge → accept gate (`accept = tests_pass AND
tests_adequate`). The two signals catch orthogonal failures — the executor
catches wrong code, the judge catches trivial or under-covering tests; neither
alone suffices. On reject, the client owns the retry loop (the response says
another round is needed and writes nothing).

## Key constraints and invariants

- **AS-11 — declarative-ensemble-native; extend the engine, never a parallel
  layer.** Control flow lives in ensemble DAGs and engine primitives, not
  bespoke Python. Where the engine is inadequate, add a primitive
  (guard/loop/dispatch were added this way), never a driver beside the engine.
- **AS-2 — validate-before-load is the registry's single admission gate.**
  Every capability part and composition shape passes a shared reference-graph
  check (no cycle, within depth, resolves) before it becomes dispatchable. One
  shared routine (`core/validation/composition_validator.py`).
- **The client owns the loop.** No internal ReAct loop, no runtime
  self-composition, no trust-promotion machinery. Cross-turn state, where it is
  needed, belongs in the session substrate (`core/session/`), not in a
  resident orchestrator actor.
- **Grounded acceptance is composed verification, independent of the
  builder.** A build deliverable is accepted only when the deterministic
  executor and the isolated judge both pass. The builder never grades itself.
- **Interactive latency on the 32GB rig is first-class.** Thinking-mode is a
  per-seat routing decision (`options.think`, sent to llama-server as
  `chat_template_kwargs.enable_thinking`): easy turns run
  thinking-off (~seconds), hard turns may route thinking-on.
- **Determinism over carve-outs.** Essential control (termination, routing,
  admission) is deterministic; model judgment is confined to guarded,
  closed-set decisions whose blast radius the deterministic surround bounds.

## Where things live

| Concern | Code |
|---------|------|
| HTTP endpoint / OpenAI compat | `src/llm_orc/web/api/v1_chat_completions.py`, `v1_models.py`, `sse_format.py` |
| Per-turn caller (endpoint → ensemble) | `src/llm_orc/web/serving/serving_ensemble_caller.py` |
| Chunk vocabulary / session-start contract | `src/llm_orc/web/serving/chunks.py`, `session_start.py` |
| Serving ensemble + seats | `.llm-orc/ensembles/agentic-serving/` (shipped in the wheel as `llm_orc/serving_project/`) |
| Registry: Topaz-keyed parts, shape catalog, admission | `src/llm_orc/core/serving/` |
| Seat contracts / validation framework | `src/llm_orc/core/validation/` |
| Session substrate (registry, artifacts, compaction, plexus adapter) | `src/llm_orc/core/session/` |
| I/O envelope (inter-seat seam) | `src/llm_orc/models/dispatch_envelope.py` |
| Engine primitives (guard, loop, dynamic dispatch) | `src/llm_orc/core/execution/` |
| Turn trace (per-turn introspection) | `src/llm_orc/web/serving/turn_trace.py` → `<state dir>/.serve-trace/turns.jsonl` |

## Decisions

The architectural decisions behind this design live in
[`docs/adrs/serving/`](adrs/serving/) — a separate numbering space from the
project-level ADRs in `docs/adrs/` (see that directory's README for the
namespace rule). Start with ADR-044 (the declarative-serving invariant),
ADR-046 (the target architecture and the orchestrator-actor dissolution),
ADR-047 (extensibility: registry + shape catalog), and ADR-048 (grounded
acceptance).

## Layers and state

Configuration is read from four tiers, highest precedence first:

| Tier | Where | Written by |
|------|-------|------------|
| project | `<checkout>/.llm-orc/` (discovered walking up from cwd) | CRUD with `scope: project` (the default), the orchestrator's composition writes |
| library | `LLM_ORC_LIBRARY_PATH`, else `<checkout>/llm-orchestra-library/` | nothing (templates) |
| global | `$XDG_CONFIG_HOME/llm-orc/`, default `~/.config/llm-orc/` | CRUD with `scope: global`, operator `*.local.yaml` overrides |
| packaged | `llm_orc/serving_project/` in the wheel; this repo's `.llm-orc/` in a checkout | nothing (read-only; `brew upgrade` changes it) |

Ensembles, profile listings and scripts merge all four (first match
wins). Runtime model profiles merge packaged → global → project (the
library is listed, never resolved). The serving ensemble, its
serve-owned scripts and the `serving:` config keys are read from the
**serving root**: the project's `.llm-orc/` when it carries
`ensembles/agentic-serving/serving.yaml`, else the packaged project. So a
serve started in an empty directory runs the shipped serving ensemble,
and a checkout of this repo shadows it. `LLM_ORC_SERVING_PROJECT_DIR`
points the packaged tier at another directory (empty disables it).

Runtime state (artifacts, the turn trace, the script cache, the rendered
router preset) is written to one **state dir**: `LLM_ORC_STATE_DIR` or
`llm-orc serve --state-dir`, else the project's `.llm-orc/` when there is
one, else `$XDG_STATE_HOME/llm-orc/` (default `~/.local/state/llm-orc/`).
In a wheel install nothing is ever written under the packaged tier; in a
checkout the packaged tier is the project's own `.llm-orc/`, which is where
project writes go.

## Preflight: what a remote is missing

`GET /api/ensembles/{name}/runnable` (MCP: `check_ensemble_runnable`)
walks the ensemble's closure: child ensembles (`ensemble:`, `loop.body`,
a literal `dispatch:`), scripts, profiles and their
`fallback_model_profile` chains (the agent-level fallback is a single hop,
as at run time), inline models. Children resolve as the run resolves
them (the executor's search dirs, `<dir>/<name>.yaml` by filename) and
profiles from the runtime tiers the run reads. Each dependency gets one
of eleven statuses and a resolve hint:

| status | meaning | resolve |
|---|---|---|
| `ready` | present; for a local model, listed by the router and either its `hf_repo` in the router's cache or the model loaded now; also a listed model with no `hf_repo` | `none` |
| `dynamic` | a `${...}` dispatch target, decided at run time | `none` |
| `pullable` | listed by the router, GGUF not downloaded: `POST /api/models/{model}/pull`. A model pulled in a running serve reads `ready` once loaded, and reads `pullable` again if the router evicts it before the next restart (its cache entry appears at restart) | `pull` |
| `needs_restart` | profile has a source, the router has not scanned it (it reads the preset at start; a supervised restart costs 1.5 to 2.2 s and cuts in-flight completions). Also a listed model whose router source (its `--hf-repo`) differs from the profile's `hf_repo`: the router serves the model from its own source, so the profile's file is not what would run | `restart` |
| `missing_profile` | no profile of that name in any runtime tier (project, global, packaged; the library is not resolved at run time) | `bind` |
| `model_unavailable` | an OpenAI-compatible endpoint does not list the model | `bind` |
| `missing_model_source` | not listed and the profile has no `hf_repo` | `add_source` |
| `needs_credentials` | a cloud provider with no credentials on this host | `add_credentials` |
| `missing_script` | a path-syntax script reference nothing on the search path resolves | `ship` |
| `missing_ensemble` | a child ensemble no tier has | `ship` |
| `provider_unavailable` | router or endpoint unreachable, or a provider llm-orc does not know | `start_provider` |

`runnable` is true only when every dependency is `ready` or `dynamic`; a
`pullable` model is the caller's decision, never an implicit download.
Each entry carries `via`, the `ensemble.agent` frames from the root, so
a child's missing script is reported at the top with the path to it.
Model presence comes from the router's own listing and nothing else: a
preset section is routable, a cache entry (its id is the `hf_repo`
string, built when the router starts) or a model loaded now is
downloaded. The `agents` list keeps the coarse per-agent
status the web UI reads.

## Running a request: inline ensembles, bind and pull

One request shape runs an ensemble on the serve, whether the ensemble is
installed there or travels in the request. It is the same body on
`POST /api/ensembles/execute` and on the MCP `invoke` tool (where `input`
is `input_data`):

```json
{
  "ensemble": {
    "name": "review",
    "description": "check, then review",
    "agents": [
      {"name": "check", "script": "probe/check.py"},
      {"name": "sub", "ensemble": "child", "depends_on": ["check"]},
      {"name": "review", "model_profile": "reviewer", "depends_on": ["sub"]}
    ]
  },
  "ensembles": {
    "child": {
      "description": "a child ensemble",
      "agents": [{"name": "c", "model_profile": "seat"}]
    }
  },
  "profiles": {"seat": {"provider": "llama-server", "model": "qwen3-8b"}},
  "scripts": {"probe/check.py": "print('...')"},
  "bind": {"reviewer": "seat"},
  "pull": false,
  "persist": "global",
  "input": "text for the ensemble"
}
```

The root is `ensemble` (inline, as above) or `ensemble_name` (installed,
in its place), never both. `persist` is optional, takes only `global` and
needs an inline root (see Bundles). `POST /api/ensembles/{name}/execute` takes the same body without a
root: the path names it, and a body that also carries `ensemble_name` or
`ensemble` is a 422. Both REST bodies reject unknown keys with a 422, so a
misspelled `bind` cannot run the call without its binding. The MCP tool
ignores unknown arguments instead, so a misspelled argument over MCP is
dropped silently.

Names must be distinct ignoring case: `Kid` and `kid`, or `a.py` and
`a.py/b.py`, are `invalid_request`, since a case-folding disk would make each
pair one file. Script keys need path syntax (a `/` or a script extension);
a bare key like `date` would be read as shell content, so it is
`invalid_request` too. So are two script keys one reference reaches, such
as `x.py` and `scripts/x.py` (the resolver also tries a reference without
a leading `scripts/` and with hyphens as underscores): the first match
would win and the other script would be silently unused.

Every run through REST, MCP or the CLI is preflighted first, named ensembles
included (`/v1/chat/completions` is not gated). The
gate is the preflight above, run over the request's own layer. A
`pullable` model blocks unless `pull: true`, which downloads each one,
waits for the router to load it, and resolves it only if the router
reports `loaded`. The pull happens only when everything else is ready: a
request with any other unmet dependency is refused without touching the
router. When the host cannot run the request, nothing runs and
the call returns (HTTP 200, like any run outcome):

```json
{"status": "error", "has_errors": true, "results": {}, "deliverable": null,
 "error": {"kind": "not_equipped", "message": "...", "dependencies": ["..."]}}
```

`kind` is `not_equipped` (the dependency report, same rows as preflight)
or `invalid_request` (the request is malformed: both roots, a script key
that escapes its directory, names that collide, a `bind` key no profile
names). A success
keeps its usual keys, which include `metadata` (the executor's usage and
durations), and adds `bindings` (the binds applied), `pulled` (the models
downloaded) and `persisted` (the bundle stored) when they are not empty.

`bind` maps a profile name the closure uses to another profile. The run
sees the first name as a copy of the second's definition, looked up after
the inline profiles and one hop deep. It applies even when the host has
its own profile of the first name. An unmet target (no such profile, or
its fallback chain unmet) refuses the call, and a key that no profile in
the closure names is `invalid_request`, so a typo never falls through to
the host's profile of the intended name.

An inline profile can name a model the router already lists (with the
caller's options and fallback chain), a cloud provider the host has
credentials for, or an OpenAI-compatible endpoint. It cannot bring a new
local model: that needs a persisted profile (`scope: global`) and a
serve restart. `needs_restart` is never resolved inside a run, since a
restart would cut every other run's completions, so such a request is
refused.

Scripts are written at their key inside the run's own directory and run
from there. A script imports only what ships beside it in `scripts`; there
is no fall-through to a helper of the same path on the host, because that
would mix versions and still report success. Preflight reads a
script's files block (next section) but not its imports, so a helper the
block does not list fails the script at run time. Nothing
persists: the run directory is removed on success, refusal, error and
cancellation, and an inline root saves no artifact (an installed root
keeps its own). A cancelled run (a cancelled call or a closed stream) kills
each script and the processes in its process group before the directory is
removed; a process that left the group is not tracked.

Both REST execute routes cancel the run when the client disconnects, and
answer 499 to a connection that is already gone. So a Ctrl-C in the CLI
stops the scripts on the serve and removes the run layer.

Trust: injected scripts run unsandboxed as the serve's user, and an inline
profile can point the host at any endpoint. llm-orc does not sandbox any
of it, so the boundary is who can reach the port. A serve with clients it
does not trust needs more than this; issue #205 tracks that design.

## A script's files block

A script lists the files it needs beside it in a comment block, in the form
PEP 723 reserves for tools:

```python
# /// llm-orc
# files = ["_helpers.py", "lib/parse.py"]
# ///
```

The opening line is `# /// llm-orc` (`// /// llm-orc` in a script whose
comments start with `//`), the body is TOML with one key, `files`, and
`# ///` closes it. Every line of the block starts with the same comment
leader. The first `/// llm-orc` opening line in the file starts the block:
if that block does not close or does not parse, that is the error, whatever
follows it. A script with no block, or an empty one, lists nothing.

Each path is relative to the script's own directory: no `..`, no leading
`/`, no backslash, no empty or `.` segment. A listed file can carry a block
of its own, and the files it lists are followed.

Nothing is inferred from imports. The list is what a closure carries, and
preflight and `invoke --remote` read the same list. A listed file is looked
for beside the script that resolved and nowhere else, so a host file at the
same relative path in another tier never stands in for it. For a script
that is a symlink, beside means beside its target, which is where Python
imports from.

Preflight reports a listed file that is not beside its script as
`missing_script` with resolve `ship`, on a row named for the file's path
next to the script (`tools/_helpers.py` for `_helpers.py` listed by
`tools/x.py`). A block that does not parse, has a key other than `files`,
or lists a path that breaks the rule leaves the script itself
`missing_script`, with the reason in `detail`. There is no new status.

An import the block does not list works wherever the sibling file happens
to be and fails by name on a remote that lacks it.

## Running a local ensemble on a remote

`llm-orc invoke <ensemble> --remote <name|url>` ships a local ensemble and
what it needs to another serve and runs it there. A value with `://` is a
URL. Any other value is a name from `remotes` in the global `config.yaml`
(`$XDG_CONFIG_HOME/llm-orc/`):

```yaml
remotes:
  remote-host:
    url: https://llm-orc.remote.example
```

A project config's `remotes` is not read, so a checked-in file cannot point
a run at a host you did not configure. An unknown name is an error that
lists the known names. When a name is looked up, a `remotes` that is not a
mapping, or an entry without a string `url`, is an error. A URL, named or
given, must be `http` or `https` with a host and a port in 1 to 65535;
anything else is refused naming what is wrong.

The request is the one in the previous section, sent as a single POST to
`<url>/api/ensembles/execute`. It carries:

- the root and every child ensemble that resolves locally, as the YAML
  files were written, under the name the parent references them by;
- every script and listed file, from whichever tier resolved it (project,
  library, global or packaged), so the remote runs the caller's scripts;
- the input, `--bind`, `--pull` and `--persist`;
- the local definition of each profile named with `--with-profile NAME`
  (repeatable), as an inline profile.

Profiles stay behind unless asked for, since a host binds roles to its own
models. A child ensemble, script or listed file that does not resolve
locally is not sent, and the remote's preflight judges it: the remote may
have its own copy. The CLI names each one on stderr before the run goes
out (`Left to the remote (not found locally): ensemble 'kids/x'`). A
`${...}` dispatch target reads `dynamic` and is left to the remote.

Before sending, the CLI resolves every script reference against a copy of
the request laid out as the remote will lay it out, with the resolver the
remote uses. A reference that would reach a different file there is
refused.

The same check covers what is left out: a reference that did not resolve
locally must not resolve to a shipped file either, or the remote would run
the caller's file where the caller's own host would have refused.

These are refused before anything is sent, with exit 1: `--max-concurrent`,
an unknown or malformed remote, a `--with-profile` name with no local
profile, a child ensemble that does not load, a script
given as an absolute path, a script that is a bare file name in the working
directory (the remote would read it as inline shell), a script reference
with a `.` or `..` segment, a script that cannot be read or is not UTF-8
text, two local files the remote would reach by one reference, an ensemble
or profile that is not plain data, an interactive script, and a request the
remote's validator would refuse. `--with-profile` and
`--persist` without `--remote` are usage errors (exit 2).

The CLI prints a remote result the way it prints a local one, in rich,
`--output-format text` or `--output-format json`. Rich mode shows
`Running on <remote>... Ns` on stderr while the run is out. Rich and text
modes print `Bindings applied`, `Models pulled` and `Bundle persisted`
lines when the result has them; JSON mode prints the remote's document
with every key it carries, plus `config`. A refusal
prints `Run refused (<kind>): <message>` and, for `not_equipped`, the
dependency report as a table; JSON mode prints the refusal envelope. An
answer is a result only if it is HTTP 200 and a JSON object with a `status`
of `success` or `error` and a boolean `has_errors`. Anything else is an
error naming the remote and the status code: a 422 from an older serve, a
404, or the web UI's page answered with 200 for an unknown API path. A
redirect is not followed, since the closure would go to a host you did not
name; the error shows where it pointed, so the `url` can be corrected (an
`http://` remote behind a proxy that redirects to `https://`, for one). The
address shown is cut at 200 characters and stripped of control characters.

The client is `httpx`. It reads proxy settings from the environment
(`HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`), and a private CA from
`SSL_CERT_FILE` or `SSL_CERT_DIR`. When those settings keep the client from
being set up (a SOCKS proxy without the `socksio` package, a proxy with an
unknown scheme, a certificate file that is not there), the error names the
cause and those variables, and nothing was sent.

Exit codes: 0 for `status: success` with no errors, 1 for `status: error`
or `has_errors` (a failed run, a refusal, a closure that cannot ship, an
unreachable remote, an answer that is not a result), 130 for Ctrl-C.
Ctrl-C closes the connection, and the serve then cancels the run.

A run on a profile the remote lacks, then the same run bound to one it has
(shown with `--output-format text`; rich draws the table as a box):

```
$ llm-orc invoke review "the diff" --remote remote-host --output-format text
Run refused (not_equipped): this host cannot run the ensemble, unmet: seat (missing_profile)
kind      name    status           via           resolve
ensemble  review  ready                          none
profile   seat    missing_profile  review.write  bind

$ llm-orc invoke review "the diff" --remote remote-host --bind seat=general
Bindings applied: seat -> general
...
```

The MCP `invoke` tool does the same for an agent. With `remote` (a name or
a URL) and `ensemble_name`, the server ships that local root's closure and
returns the remote's result document, adding `left_out` (what did not
resolve locally) when there is any. `bind`, `pull`, `persist`
and `with_profiles` (a list of profile names) travel with it. Cancelling
the call closes the connection, and the remote cancels the run. `remote`
with an inline `ensemble`, `ensembles`, `profiles` or `scripts` is
`invalid_request`: a client that holds a definition can send it to the
remote itself. When nothing was sent (an unknown remote, a closure that
cannot ship, an interactive script) the error kind is `invalid_request`.
When the remote could not be reached or did not answer with a result, it
is `remote_error`.

`remote` works on the stdio server (`llm-orc mcp serve`), which has one
local client. Nothing reachable over the network relays: REST takes no
`remote`, and the MCP endpoint a serve mounts at `/mcp` and
`llm-orc mcp serve --transport http` both answer `remote` and
`with_profiles` with `invalid_request`.

## Local runs: bind, pull and the gate

`--bind NAME=TARGET` (repeatable) and `--pull` work on a local
`llm-orc invoke` and mean what `bind` and `pull` mean in the request. A
value without `=`, or a name bound twice, is a usage error. `--config-dir`
is still the directory the root is looked up in.

Breaking change for local CLI callers: `llm-orc invoke` and
`llm-orc validate run` now go through the same preflight gate as REST and
MCP. An ensemble the host cannot run is refused before any agent starts:
the CLI prints the refusal and the dependency table and exits 1. A local
model that is listed but not downloaded now blocks until the call carries
`--pull`, which downloads it first. `validate run` takes `--bind` and
`--pull` too. Before, the CLI ran the ensemble with no check.

## Bundles

A request with `persist: global` (`--persist global` on `invoke --remote`)
stores the closure it carries on the serve, and runs it. It needs an inline
root whose name is one plain file name, and it passes the same validation
and gate as any run. If the gate refuses, nothing is written. The bundle is
stored once the gate passes, before any agent runs, so a run that then
fails or is cancelled still leaves it stored. The run goes on as a named
run, so its artifact is kept, and the result names the bundle in
`persisted`.

A bundle is one stored request: `bundles/<root name>.json` under the global
config directory, holding the root, `ensembles`, `profiles`, `scripts` and
`bind`. It does not hold `input`, `pull` or `persist`. It is used only when
its root is run by name (REST, MCP or the host's CLI). The run lays the
bundle's contents into a run layer for that run, gates it, runs it and
removes the layer, as for an inline request. Listings show the root with
source `bundle`, and the runnable check of its name reports over the same
layer.

A bundle shadows nothing on the host. If its scripts and children were
copied into the global tier, every ensemble on the host, `/v1` included,
would get the caller's copies in place of the host's packaged files of the
same names, and would keep them across upgrades. A bundle's children and
scripts resolve for its own root only.

Names resolve through the tiers (project, library, global, packaged) first
and bundles after. So a tier file added later under a bundle's name wins,
and persisting a root name that a tier already resolves is refused with
`invalid_request`, naming the tier. Persisting again replaces the bundle
(written beside and renamed, so a failure keeps the old one). A name that
matches an existing bundle's except for case (`PACK` when there is a
`pack`) is refused, naming the existing bundle: on a disk that folds case
the two would be one file. A bundle name otherwise means the file spelled
exactly `<name>.json`.
`delete_ensemble` with `scope: global` removes it. If the global tier also
holds an ensemble file of that name, that delete removes the file and a
second one removes the bundle.
`persist` on a run by name is `invalid_request`, and the named execute route
rejects it with a 422. `validate` and promote answer that a bundle is not a
tier ensemble.

A caller's injections lay over the stored ones key by key. A `bind` or an
inline profile for a role replaces the stored definition of that role, and
a per-run `bind` overrides a stored one without changing the bundle. The
union goes through the validator, so an injection that meets a stored name
in another spelling (`Kid` against `kid`, `x.py` against `scripts/x.py`) is
`invalid_request`.

A bundle runs as a root. It is not a child of a host ensemble: a host
ensemble cannot reach a child or script that only a bundle holds. A bundle
keeps its copies until it is persisted again.

## Operator seat configuration

Seat models resolve through **tier profile names** (`agentic-tier-cheap-general`
and friends in `.llm-orc/profiles/`) — the tier name is the stable operator
surface; which model/provider backs it is deployment-specific. The shipped
defaults are all local (`provider: llama-server`). To back any tier with your
own provider — a paid API, a hosted endpoint, a bigger local model — create a
gitignored override:

```yaml
# my-paid-seat.local.yaml   (never committed)
name: agentic-tier-cheap-general   # the tier name to override
model: your-hosted-model
provider: openai-compatible/yourprovider
cost_per_token: 0.0
```

In a checkout the file lives in `.llm-orc/profiles/`. In a wheel install
(the mini) it lives in `~/.config/llm-orc/profiles/`, because the global
tier sits above the packaged one and nothing reads `.llm-orc/` there.
`*.local.yaml` files load last (deterministically), so they win over the
checked-in profile of the same name. Nothing provider-specific belongs in
tracked config. Empirical note (2026-07-08 A/B): a hosted frontier seat did
not change the dominant failure class — reach for structure (retry rounds,
shapes) before bigger models.

## Local inference: the serve owns the router

`llm-orc serve` starts and supervises one `llama-server` process in router
mode (llama.cpp; the binary is the only external runtime dependency). At
start the serve renders `llama-server.ini` into the state dir from every
`provider: llama-server` profile: one section per distinct `model:` name,
its GGUF source from the profile's `hf_repo:` (fetched from Hugging Face on
first load), a 40960-token context (the window the truncation backstop
assumes; `options.num_ctx` overrides per model), and one resident model at
a time (`--models-max`). The router loads a model on the first request that
names it and evicts least-recently-used. Model names route exactly and must
not contain a colon (the router rewrites `name:tag`); the renderer refuses
one.

`LLAMA_SERVER_URL` (default `http://127.0.0.1:8080/v1`) is exported for the
life of the serve so the model factory reaches the router; `--no-backend`
skips ownership and uses whatever is already at that URL. `GET /api/models`
lists what the router serves with load status; `POST /api/models/{name}/pull`
loads (and downloads) one. Per-request thinking control is
`chat_template_kwargs.enable_thinking`; llama-server's `timings` land on the
usage record as `prompt_eval_count` / `eval_count`.

An embedding seat is a `provider: llama-server` profile whose `options` set
`embeddings: true` and a `pooling` mode (`local-nomic-embed-text.yaml` backs
`nomic-embed-text-v1.5`); the renderer passes both through to the model's
preset section, allowlisted the same way `num_ctx` is. `POST /v1/embeddings`
forwards an OpenAI-shaped request to the router's own `/v1/embeddings`:
`model` may be a llama-server profile id (resolved to its served model name)
or the router model name directly; `input` (string or list of strings),
`encoding_format`, and `dimensions` forward as given. A model that isn't in
the rendered preset 404s without reaching the router; a router that's down
503s.

## Conversation memory

The serve threads conversation context from the client-sent history into
generation seats (bounded render: last 8 user/assistant turns, ~4KB;
written-file bodies included so referents like "add tests for it" resolve).
classify composes it behind the deterministic `Current request:` marker;
routing and verifier seats read the clean latest turn only (the accept-gate
judge is input-scoped to its dependencies). Conversation-written files
materialize into the accept-gate sandbox, so follow-up builds can import the
modules the conversation created. Design and the scaling ladder (lossless
session record, plexus lenses):
`docs/plans/2026-07-08-serving-conversation-memory-design.md`.

## Current capability coverage

Build (accept-gated), explain, within-session conversation memory,
**client-file reads**, and **client-delegated test runs** are implemented
and grounded against a live endpoint and a literal `opencode run`
(multi-turn battery 2026-07-08: build → "did you see my previous query?"
→ "add tests for it", all green; existing-file battery 2026-07-09:
"write tests for existing storage.py" → read tool_call → gated test
deliverable running green against the real module, and an honest
one-round refusal when the read fails; run battery 2026-07-09: "run
test_calc.py" → bash tool_call → client-executed pytest → honest verdict,
green and red both). A turn naming a file the serve can't see delegates a
`read` through the permission seam
(`docs/plans/2026-07-09-client-file-reads-design.md`); the result
materializes into the gate sandbox and stays retrievable in later turns.
A run turn delegates one closed-template `pytest` command through the
same seam and replies with a deterministic verdict parsed from pytest's
own summary — zero model calls end to end
(`docs/plans/2026-07-09-client-run-delegation-design.md`). Remaining
frontier on the execution surface: discovery of files the turn doesn't
name, and chaining write → run inside one fix turn. Context older than
the render window is dropped until the lossless session record (design
§Rung 2′) lands.

## Roadmap

The staged path to the north star, with per-stage exit gates and the
ladder-based parity measurement: [`docs/serving-roadmap.md`](serving-roadmap.md).

## History

This design is the product of an 8-cycle research process (2026-04 → 2026-07).
The full research corpus — essays, research logs, spike records, the complete
ADR set including superseded decisions, scenarios, field notes, and audits —
is preserved on the `research/agentic-serving-corpus` branch under
`docs/agentic-serving/`. Any reference to a `docs/agentic-serving/...` path or
an ADR number not present in `docs/adrs/serving/` resolves there.

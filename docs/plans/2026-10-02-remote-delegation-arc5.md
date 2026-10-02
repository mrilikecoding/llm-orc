# Remote delegation, Arc 5: the CLI as the remote client

**Goal:** `llm-orc invoke <ensemble> --remote <name|url>` ships the
ensemble's closure to a remote serve as one run request and prints the
result as it prints a local one. `--bind` and `--pull` work on every
invoke, local or remote. `--persist global` installs the closure on the
remote as one unit. An agent reaches the same thing through the MCP
`invoke` tool's `remote` argument (#196, #191).

**Architecture:** The CLI stops building its own executor: every invoke
is a run request handed to the preparation step REST and MCP already
use. A shipper turns a named local root into an inline request: each
ensemble as written, keyed by reference; each script at its reference
string with the files its own header block lists. The block makes those
files closure members, so the remote's preflight sees them too.
Transport is one plain POST with `requests`. A persisted closure is the
stored request, replayed through the unchanged injection path on each
named run.

**Spec:** `docs/plans/2026-09-28-remote-delegation.md`: decisions 1-4,
the Arc 5 card, "Arc 5 re-cut (2026-10-02)" (the code read, the probes,
rulings 1-12), the Arc 4 re-cut rulings 2, 3, 7, 9, 10 and "Review
rounds and the re-run", and "Standing rules for every implementer". Read
the Arc 5 re-cut before touching anything. The rulings are binding;
where a card below and a ruling disagree, the ruling wins and you say so
in your report.

**Tech stack:** Python 3.11+, pydantic, click `CliRunner`, FastAPI
`TestClient`, `requests`, `tomllib`, pytest (asyncio auto mode), ruff
88, mypy strict, complexipy <= 15.

## Global constraints

- One worktree for the arc:
  `git worktree add .claude/worktrees/arc5-cli-remote -b feat/cli-remote-client main`.
  Never `git stash` (the working dir is shared with other sessions).
  Never spawn subagents that edit or commit.
- Strict TDD, one test at a time. Structural and behavioral commits
  separate. ruff 88 + mypy strict from the first draft. `make lint`
  clean before every commit.
- Commit prefixes `feat:` `fix:` `refactor:` `test:` `docs:` `chore:`.
  No AI attribution of any kind, no session links, no scratch paths in
  tracked files. Nothing that names one machine: a remote in a test or
  a doc is `remote-host` or `https://llm-orc.remote.example`.
- Full suite, in parallel: `uv run pytest -n auto -q -p no:cacheprovider`
  (about 70 s). After every task check the wall time and
  `--durations=10`, not only the passed count (roadmap, binding). Known
  flake under `-n auto` (#165): the artifact manager integration test;
  re-run it alone before treating it as real. Known local-only failure:
  the auth-only providers test when a router listens on :8080. Record
  the baseline count in Task 0.
- Doctrine 11: pins assert outcomes through the real surface (`CliRunner`
  on the real command, a real `OrchestraService` on a temp project dir,
  REST `TestClient`, the real executor, files on disk). Each key pin is
  shown RED under a named mutant in the run that gates. Put the mutant
  and the red assertion line in the commit body. Clear
  `tests/__pycache__` after mutant rounds.
- Never run the service outside pytest (no ad-hoc `TestClient` or
  `ConfigurationManager` scripts): they write into the real
  `~/.config/llm-orc`. End every task with `git status --short` in the
  worktree and `ls ~/.config/llm-orc ~/.config/llm-orc/ensembles
  ~/.config/llm-orc/profiles` unchanged (Task 9 adds a `bundles`
  directory to the global config; it must never appear in the real
  one).
- No test opens a network connection. The remote in a test is a second
  real service on its own temp dirs, reached by swapping the one
  transport seam for a function that calls its `TestClient`.
- No new dependency. No new dependency status.
- Doc drift gate: this file names tests only in prose, never as
  backticked identifiers. Keep it that way when editing it.
- Findings go on #196 and #191 as comments, never new issues. Nothing
  is pushed, nothing paid is run, the remote host is not touched.

## Review focus

Inputs the spec implies and no card names. Each gets a pin in the
owning task; the whole-branch review hunts these first.

1. **A listed file is missing beside the script, and the host has a
   file at the same relative path in another tier.** Expected:
   `missing_script`. A `ready` is the wrong accept.
2. **A block lists `../x` or an absolute path.** Expected: the script is
   unmet, and nothing outside the script's directory is read, by the
   gate or by the shipper.
3. **The remote answers 200 with HTML, or JSON with no `status`.**
   Expected: an error naming the remote, exit 1. Printing it as a
   result is the wrong accept.
4. **A bare script reference that is a file in the caller's cwd.**
   Expected: refused locally. Shipped, the remote runs it as inline
   shell.
5. **A flag the remote request cannot carry** (`--max-concurrent`).
   Expected: refused, not dropped.
6. **The CLI and the service disagree.** After Task 3 no CLI path builds
   an executor or looks a root up on its own; a not-runnable ensemble is
   refused on the CLI exactly when REST refuses it.
7. **An ensemble file with YAML that is not plain JSON data** (a date, a
   set, an anchor that expands). Expected: it ships as the same data the
   loader read, or is refused by name. A silent string is the wrong
   accept.
8. **A root found by its `name:` whose file name differs, a hierarchical
   child reference, two references that reach one script file.**
   Expected: the remote resolves each as the caller's host did.
9. **The disconnect watcher cancels a run whose client is still there,**
   or leaves a run going whose client is gone. Expected: neither. The
   run layer is removed and the script group is dead after a disconnect.
10. **An injection over a bundle collides with what the bundle holds**
    (`x.py` against the bundle's `scripts/x.py`; `Kid` against `kid`).
    Expected: `invalid_request`, as for one request.
11. **A bundle leaks.** A host ensemble that references a child or a
    script only a bundle holds reads `missing_ensemble` or
    `missing_script`. The service's own config manager lists no bundle
    content after a bundle run.
12. **Persist with a refused gate, or interrupted mid-write.** Expected:
    no bundle file, no partial file, the previous bundle intact.
13. **A persist whose root name a tier already resolves.** Expected:
    `invalid_request` naming the tier, nothing written. And a tier file
    added later under a bundle's name wins the lookup.
14. **A remote older than the client** (it forbids `persist`).
    Expected: the 422 is reported with the remote's detail, exit 1.

## Task 0: worktree and baseline

Create the worktree, `uv sync -q`, run the full suite, record the
passed count and the wall time in your report.

## Task 1: a script's files block (ruling 8)

**Files:** a new module under `src/llm_orc/core/execution/scripting/`
for the block, `src/llm_orc/core/config/closure.py`,
`src/llm_orc/services/handlers/preflight.py`,
`src/llm_orc/services/handlers/provider_handler.py`, the packaged
scripts under `.llm-orc/scripts/`, tests beside each.

**Interfaces:** A pure function from script source to its listed paths:
the first `/// llm-orc` block, comment leader `#` or `//`, TOML body, key
`files` a list of strings; each a relative path with no `..`, no leading
`/`, no backslash, no empty segment (the rule `run_request` already has;
share it). No block means an empty list; a block that does not parse or
breaks the path rule is an error value the caller can report.
`walk_closure` takes a callable from a script reference to its listed
files (resolved by the caller with the run's own `ScriptResolver`) and
records each as a dependency the script's frames own, following a listed
file's own block, cycle-safe. The classifier reads a listed file as
`ready` when it exists beside the resolved script and `missing_script`
otherwise; a bad block makes the script itself `missing_script` with the
problem in `detail`. The report rows are additive.

Then the sweep: each packaged script gets a block listing what it
imports from or runs in its own directory (the re-cut counts 18 with
static sibling imports, plus the one that runs a sibling by path). One
commit for the sweep, separate from the engine change.

**Pins:** through the REST runnable route on a temp project: a script
whose block lists an absent file is `missing_script` and the ensemble is
not runnable; the same with the file present is `ready`; a file listed
by a listed file is followed; review focus 1 and 2. Through
`POST /api/ensembles/execute`: an injected script whose block lists a
file the request does not carry is refused `not_equipped` before any
agent runs. A suite test over every packaged script: each static import
that names a file or package in the script's own directory is inside
its block, and every listed file exists. Mutants: look the listed file
up through the resolver's search paths (focus 1 goes red); skip the
path rule.

## Task 2: one named-root lookup that knows its file (structural)

**Files:** `src/llm_orc/core/config/ensemble_config.py`,
`src/llm_orc/services/orchestra_service.py`,
`src/llm_orc/services/handlers/execution_handler.py`.

**Interfaces:** `EnsembleConfig.source_path: str | None`, set by
`load_from_file`. `ExecutionHandler._lookup_in_tiers` goes away; the
handler uses the service's one lookup for every entry path. Structural:
the suite's count and outcomes are unchanged, and the commit says so.
If the two loops turn out to differ in behavior on any input, stop and
report it instead of choosing.

## Task 3: the CLI runs through the preparation step (rulings 1, 5)

**Files:** `src/llm_orc/services/handlers/execution_handler.py` (the
preparation step becomes a public async context manager on the service),
`src/llm_orc/services/orchestra_service.py`, `src/llm_orc/cli.py`,
`src/llm_orc/cli_commands.py`, `src/llm_orc/cli_modules/utils/visualization/`,
`tests/unit/cli/`.

**Interfaces:** `invoke` gains `--bind a=b` (repeatable; a value without
`=` is a usage error) and `--pull`. `invoke_ensemble` builds the request
(`ensemble_name`, `bind`, `pull`) and runs inside the service's prepared
run: the config and executor it yields feed the existing streaming and
standard display functions, so the rich interface, interactive scripts
and `--max-concurrent` keep working. `--config-dir` supplies the lookup
for the root. A refusal prints the dependency report as a table (kind,
name, status, via, resolve) in rich and text modes and the refusal
envelope in JSON mode; exit 1. Applied bindings and pulled models are
shown.

**Pins:** with `CliRunner` on a temp project: an ensemble on a missing
profile exits 1, prints the row, and no agent ran (a script agent that
would write a marker file did not); the same with `--bind` runs and
shows the binding; a misspelled bind key is refused with the service's
message; JSON mode prints the envelope with `error.kind`; a runnable
ensemble's output in each format is what it was before (snapshot the
JSON keys). Review focus 6: a grep-level assertion is not a pin; drive
one not-runnable fixture through the CLI and through REST and compare
the verdict. Mutant: fall back to the old executor path when the gate
refuses.

## Task 4: the result carries `metadata` (ruling 5)

**Files:** `src/llm_orc/services/handlers/execution_handler.py`,
`src/llm_orc/web/api/ensembles.py` if a response model lists keys,
`src/llm_orc/mcp/server.py` likewise, the CLI display functions.

**Interfaces:** the service's run result gains `metadata`, the
executor's own, on every entry path. The CLI's JSON, text and rich
displays take a result document and the agent list, not an executor
result, so Task 8 can hand them a remote's answer. Additive: no key
leaves the result.

**Pins:** REST and the MCP tool return `metadata` with the usage and
duration a run produced; the refusal envelope is unchanged. Mutant: drop
the key on one path.

## Task 5: a REST run is tied to its connection (ruling 4)

**Files:** `src/llm_orc/web/api/ensembles.py`, tests under
`tests/unit/web/`.

**Interfaces:** both execute routes run the service call as a task and
cancel it when the request reports a disconnect; a finished run answers
as before. Arc 4's cancellation path does the rest.

**Pins:** drive the ASGI app directly with a `receive` that reports
`http.disconnect` while an injected script sleeps: the script's process
group is gone and `runs/` is empty well before the script's own end; the
same request with a client that stays gets its full result (review focus
9, both directions). Check `--durations`: the pin must not wait out the
sleep. Mutant: never poll the disconnect.

## Task 6: named remotes (ruling 2)

**Files:** `src/llm_orc/core/config/config_manager.py`, a small resolver
used by the CLI and the MCP tool, tests.

**Interfaces:** `remotes: {<name>: {url: <str>}}` read from the global
`config.yaml` only. Resolve a `--remote` value: one containing `://` is
a URL; otherwise a name, and an unknown name raises with the known
names listed. A trailing slash on a URL is dropped.

**Pins:** a name resolves; an unknown name lists the known ones; a
`remotes` key in a project config is not read (a fixture with both, the
project's must not resolve). Mutant: merge the project config in.

## Task 7: the shipper (rulings 6, 7, 8)

**Files:** a new module under `src/llm_orc/services/` for the shipper,
`src/llm_orc/services/orchestra_service.py`, tests under
`tests/unit/services/`.

**Interfaces:** one service method from a root name, the profile names
to ship, `bind`, `pull`, `persist` and the input to a run request dict.
It walks the closure with the run's own finders and resolver (Task 1's
callable included). The root goes in as `ensemble` and each child that
resolved as `ensembles[<reference>]`, both the parsed YAML of
`source_path`; data that is not plain JSON is refused naming the file
(review focus 7). Each script reference with path syntax goes in
`scripts` at a key the remote resolves that reference to, its listed
files beside it; two references that reach one local file must both
resolve on the remote. Refused with a message naming the reference: an
absolute script path, a bare name that is a file in the cwd, a script or
listed file that is not UTF-8. A child or script that does not resolve
locally is left out. A profile named for shipping that does not exist
locally is an error; otherwise its definition goes in `profiles`.

**Pins:** the parity pin: build a request on a temp project that has a
root, a hierarchical child, a script with a listed helper and a second
reference to the same script; hand it to a second real service on empty
temp dirs through REST; the result equals the local run's, and nothing
in the request names a local path. Then each refusal, review focus 4, 7
and 8, a missing local child that the remote reports
`missing_ensemble`, and a shipped profile. Mutants: ship a
re-serialized config in place of the file's YAML; key a script by its
resolved path's name; ship without listed files (the remote must
refuse, by Task 1).

## Task 8: `--remote` (rulings 3, 5, 11)

**Files:** `src/llm_orc/cli.py`, `src/llm_orc/cli_commands.py`, a small
transport module, `tests/unit/cli/`.

**Interfaces:** `invoke --remote <name|url>` and `--with-profile NAME`
(repeatable; only with `--remote`). The transport is one function:
POST the request to `<url>/api/ensembles/execute`, connect timeout 10 s,
no read timeout, return the result document or raise one error type
that carries the remote and what was observed. A result is HTTP 200, a
JSON object, `status` of `success` or `error`, boolean `has_errors`;
everything else raises, a 422 with the remote's `detail`. The result
goes to Task 4's display functions; a refusal to Task 3's table; exit 1
on `status: error` and on a transport error. While waiting, rich mode
shows the remote's name and elapsed time on stderr; text and JSON modes
print nothing but the result. Ctrl-C closes the connection and exits
130. Refused before anything is sent: `--max-concurrent`, an ensemble
with an interactive script.

**Pins:** with the transport seam pointed at a second service's
`TestClient`: a run prints the remote's deliverable and exits 0; a
profile the remote lacks prints the table and exits 1; `--bind` then
runs; `--with-profile` ships the definition. With the seam returning
canned responses: 200 HTML, 200 JSON without `status`, 404, 422, a
connection error, each exit 1 with the remote named (review focus 3 and
14). Review focus 5. Mutant: accept any 200.

## Task 9: bundles and `persist` (ruling 9)

**Files:** `src/llm_orc/services/handlers/run_request.py`, a new bundle
store module beside it, `execution_handler.py`, `orchestra_service.py`,
`provider_handler.py` (the runnable check), `resource_handler.py` and
the listings, `ensemble_crud_handler.py` (delete), `validation_handler.py`
and `promotion_handler.py` (the refusal), `src/llm_orc/web/api/ensembles.py`,
`src/llm_orc/cli.py`, `src/llm_orc/cli_commands.py`, tests.

**Interfaces:** `RunRequest.persist: Literal["global"] | None`; with it
the root must be inline with a plain name, and a name any tier resolves
is `invalid_request` naming the tier. The store: write (beside, then
rename), read, delete, list, under `<global config>/bundles/`. In the
preparation step: after the gate passes, a persist request is stored
without `input`, `pull` and `persist`, and the run is marked named so
its artifact is kept. A named root that no tier resolves and a bundle
holds becomes the stored request with the caller's `ensembles`,
`profiles`, `scripts` and `bind` laid over it key by key, then the
existing path from validation on. The result of a persist carries
`persisted: <name>`. The runnable check on a bundle name gates over the
materialized view and removes the layer. Listings show the root with
source `bundle`. `delete_ensemble` with `scope: global` removes a
bundle. Validate and promote answer that the name is a bundle. The CLI:
`--persist global`, refused without `--remote`.

**Pins:** persist, then a run by name on the second service with no
injections: same result, artifact kept, `runs/` empty, the bundle file
unchanged byte for byte. Review focus 10 through 13, each through REST.
A persisted `bind` applies on a named run and a per-run `bind`
overrides it. A bundle replaced while a run of it sleeps: the run ends
with the content it started with. Delete removes it and the name no
longer resolves. Mutants: write the bundle before the gate; look
bundles up before the tiers; merge without re-validating.

## Task 10: MCP `invoke` gains `remote` and `persist` (ruling 10)

**Files:** `src/llm_orc/mcp/server.py`, the service, tests under
`tests/unit/mcp_server/`.

**Interfaces:** `persist` passes through like `bind`. With `remote`,
the tool needs `ensemble_name`, builds the request with Task 7's method,
sends it with Task 8's transport off the event loop, and returns the
remote's result; an inline root with `remote` is `invalid_request`; a
transport error is the error envelope with the message. REST does not
gain `remote`.

**Pins:** through the real tool call with the transport seam on a
second service: a local named root runs there; a missing profile
returns the remote's `not_equipped` envelope; inline with `remote` is
refused. Mutant: run locally when `remote` is set.

## Task 11: docs

`docs/serving.md`: the files block (form, the path rule, what
preflight reports), `remotes` in the global config, `invoke --remote`
with `--bind`, `--pull`, `--with-profile` and `--persist global`, what
ships and what stays behind, bundles (what they are, that they do not
shadow, how to replace and remove one), the `metadata` key, the
disconnect rule. The breaking change for local CLI callers gets its own
paragraph for the changelog writer. The CLI's own help text carries one
example of each flag. No CHANGELOG entry (release time).

## Task 12: live rows and review gate (lead)

Live rows on the laptop: two serves from empty dirs with temp XDG dirs
and a real router on the "remote" one; the CLI in a third temp project
with a `remotes` entry. A root with a child, a script and a listed
helper runs with `--remote`; a profile the remote lacks is refused with
the table, then runs with `--bind`; a script whose listed helper is
deleted locally is refused before sending or by the remote's gate,
whichever the rulings give; Ctrl-C during a sleeping script leaves no
process and an empty `runs/` on the remote; `--persist global`, then a
run by name over REST, then a re-persist, then delete; tree snapshots of
the remote's config and state dirs around each row; one call through
the MCP tool with `remote`. Then an author-independent whole-branch
review with a wrong-accept hunt over the review focus list, one fix
wave, a scoped re-review, merge to local main, roadmap State and the
board updated, summaries on #196 and #191.

The rows against the remote host wait for a release (practitioner's
go): `research-dossier` with `--remote`, which is the first run past
200 s through its proxy; the missing profile and `--bind`; Ctrl-C
through the proxy; persist and a run by name.

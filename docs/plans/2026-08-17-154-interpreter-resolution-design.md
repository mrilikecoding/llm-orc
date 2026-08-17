# #154 — .py script agents run under the host interpreter (design)

Status: pre-flight. Issue: #154. Root cause of the #152 misfire
(`docs/plans/2026-08-13-144-live-gate/`).

## Mechanism (grounded)

`ScriptAgent._get_interpreter` (script_agent.py:747-770) maps `.py` to
the literal `["python3"]`, resolved from the SERVE's inherited PATH at
run time. Serving scripts that import `llm_orc` (resolve.py via
`shape_catalog`, seat_contract.py via `llm_orc.core.validation` plus
third-party `yaml`) therefore live or die on which `python3` the
operator's shell happened to expose, while the stdlib-only scripts in
the same ensemble keep running. That asymmetry is what produced a
half-dead pipeline rather than a clean failure, and #152's fail-closed
gate makes the serve refuse honestly under it without removing it.

Four call sites consume the result: `_execute_script_file`,
`_execute_script_file_with_schema_json`, and
`_execute_script_interactive` inside `script_agent.py`, plus a
CROSS-MODULE one at `agent_runner.py:303`
(`script_agent._get_interpreter(resolved_script)`) for the interactive
path. `_get_interpreter` therefore stays a method; turning it into a
module function would break that call and two test mocks.

Evidence the hazard is live on this machine rather than only in the
gate record: `.llm-orc/scripts/agentic_serving/__pycache__/` holds both
`cpython-313` and `cpython-310` bytecode, so a Python 3.10 interpreter
(below this project's `requires-python`, and neither the venv nor
`/usr/bin/python3`) has executed the serving scripts.

## Invariant

A FILE-BACKED `.py` script agent runs under the SAME interpreter that
is running llm-orc, so an import that works in-process works in the
subprocess. Nothing about the operator's PATH may decide it.

Scoped to file-backed scripts deliberately: an inline script
(`script: 'python3 -c "..."'`) is resolved as content and run through
`/bin/bash` (`resolver.py:153`), so it stays PATH-dependent and this
change does not reach it. That shape is in use
(`test_mixed_agent_dependencies.py:63`).

## Change

`.py`/`.python` resolve to `[sys.executable]`, falling back to
`"python3"` when `sys.executable` is empty OR
`getattr(sys, "frozen", False)` — under PyInstaller `sys.executable` is
non-empty and points at the bundle, so `[sys.executable, script.py]`
would re-invoke the llm-orc CLI with a path as argv[1]. No freeze
tooling exists here, so the guard is precautionary. Other extensions
are untouched: bash, node, and ruby have no equivalent
same-environment notion.

This is not a new idea in this codebase, and the precedent is closer
than first drafted. `script_handler.py:136-160` runs
`.llm-orc/scripts/<category>/<name>.py` under `sys.executable` — the
same USER-AUTHORED tree `ScriptResolver` searches first
(`resolver.py:77`), not generated scripts. Three more:
`script_commands.py:151` runs an arbitrary user-supplied path under
`sys.executable`; `accept_executor.py:576` uses `[sys.executable,
str(RUNNER), ...]` inside the serving pipeline itself (a second-order
win — today that node inherits the wrong `sys.executable`, so its own
pytest subprocess is doomed too, and the chain becomes coherent); and
23 test modules run these exact serving scripts as `[sys.executable,
str(SCRIPT)]`, so the unit tests and the engine currently disagree
about how to run the same file.

One site is fixed alongside rather than left divergent:
`resolver.py:313` runs `[script_path]` bare, relying on shebang plus
the exec bit, so `llm-orc scripts test
scripts/agentic_serving/resolve.py` fails today with Permission denied
(the serving scripts are mode 644). It gets the same `.py` treatment,
in its own commit, so "this unifies how the repo runs python scripts"
is true rather than two-of-three true.

## Why not the alternatives

- **Honor the shebang.** The serving scripts say
  `#!/usr/bin/env python3`, which resolves through PATH — the same
  failure, moved.
- **A config knob for the interpreter.** More surface for a need
  nothing has yet demonstrated. If a script ever genuinely needs a
  different interpreter, that is the follow-up, and an explicit knob
  will beat today's implicit PATH lookup either way.
- **Detect llm_orc imports and special-case them.** Static analysis of
  arbitrary scripts to decide an execution parameter; brittle, and it
  would leave two behaviors where one is correct.

## Accepted bound

A user script that needs third-party packages from THEIR project venv,
while llm-orc runs from a separate install (Homebrew, pipx), works
today only if their PATH `python3` happens to be that venv. After this
change it gets llm-orc's interpreter instead.

State plainly what that costs, because the first draft implied a
workaround that does not exist: **there is no config escape hatch, before
or after this change.** `ScriptAgentConfig` sets `extra="forbid"` and
exposes only `script` and `parameters`
(`src/llm_orc/schemas/agent_config.py:16, 90-94`), and `agent_runner.py:111`
builds the agent from `model_dump()`, so `ScriptAgent.command`,
`_execute_command_with_json`, and `self.environment` are unreachable
from any ensemble YAML. The remedy is the one the docs already imply:
install llm-orc into the environment that holds the script's
dependencies. A one-line note goes into the user-facing script docs
with this change.

The trade is favourable rather than neutral. Today that same user
cannot use ANY shipped primitive in the same ensemble, since every
primitive imports `pydantic`
(`src/llm_orc/primitives/data_transform/json_extract.py` and siblings)
and the authoring template teaches `from pydantic import BaseModel`
(`docs/primitive-script-development.md:14`). `script_utils.py` is a
published helper that expects `from llm_orc.script_utils import
unwrap_input`. Meanwhile every script in the shipped library is
stdlib-only (6 files, zero third-party imports), so the library surface
is unaffected either way. The change makes the documented path work and
narrows the undocumented one.

Do NOT build the config knob for this. The deferral is right; if a real
need appears, an explicit knob beats today's implicit PATH lookup.

ADR-001 governs the Pydantic I/O contract, not interpreter isolation,
so no ADR constrains the change.

### Environment, the other half

Interpreter is effectively the whole story. `prepare_environment`
starts from `os.environ.copy()` (`script_agent.py:43`) and
`base_environment` is always `{}` for engine-run agents (unreachable
per the bound above), so the child gets the parent's environment
verbatim plus the INPUT_* keys. Same binary plus same environment means
the child's import state equals the parent's startup state, and
`PYTHONHOME`/`PYTHONPATH`/`VIRTUAL_ENV` cannot break the child without
having already broken llm-orc. A venv interpreter derives its prefix
from its own location via `pyvenv.cfg`, so PATH and `VIRTUAL_ENV` are
irrelevant to it.

Residual: in-process `sys.path` mutations do not propagate, so llm-orc
run from an uninstalled source checkout via `python -m llm_orc` would
have `llm_orc` importable in the parent (through cwd) but not the
child. Console-script installs make this moot.

## Regression instruments

- `_get_interpreter(".py")` returns `[sys.executable]`; `.sh`, `.js`,
  `.rb`, and the unknown-extension default are unchanged.
- The fallback returns `["python3"]` when `sys.executable` is empty or
  `sys.frozen` is set.
- Behavioral: a script agent whose script imports a module importable
  only in the host interpreter succeeds. Written against a real
  subprocess, since mocking the interpreter is what hid this.
- **The whole suite passes from a shell WITHOUT the venv on PATH.** That
  is the symptom the issue cites, and it is the instrument with teeth:
  measured baseline 50 failures (all in the serving endpoint suite),
  and 11 with the fix, with ZERO new failures. The residual 11 are a
  pre-existing unrelated line, `test_adr_006_library_based_primitives.py:97`
  invoking `["python", ...]` when macOS has no `/usr/bin/python`; fixed
  to `[sys.executable, ...]` in the same pass so the claim is literally
  true rather than 11-tests true.
- The #152 endpoint pin gets a STRENGTHENED assertion in the same commit
  that deletes its environment note. Today it asserts only
  `startswith("Refused: serving pipeline error")`, which passes for BOTH
  the laundering refusal and the resolve-crashed refusal, in both
  environments, before and after this change — so deleting the note
  without strengthening would erase the only record of the hazard and
  silently accept the degraded path forever. The new assertion pins the
  laundering form specifically (no `(resolve:` parenthetical, which
  `shape.py:81` omits when no decision dep carries an error).

## Live gate (exit)

RESTART the serve (required, not tidy: `agent_runner.py:88-93` caches a
crashed script's failure envelope as `success: True` keyed on script
content plus params with a 3600s TTL, so a pre-fix result could
otherwise replay post-fix) from a shell WITHOUT the venv on PATH — the
exact condition that produced the 2026-08-13 misfire and #152's live
gate.

**Positive control first**, recorded in the gate log, because the one
way this gate passes for the wrong reason is forgetting the PATH
condition, where the change is a no-op: in the same shell that starts
the serve, capture `command -v python3` and a FAILING
`python3 -c "import llm_orc"`.

Then BOTH asks, mirroring #152's two-probe structure:

1. The #144 gate-2 ask ("how does resolve pick the seat?"). Expected: a
   grounded answer, not the `Refused: serving pipeline error` that #152
   correctly produces today. This observes `resolve.py`.
2. The build ask ("write a function that adds two numbers in add.py").
   Expected: a real write tool_call. This is what observes
   `seat_contract.py`, the second casualty the #152 record names: a
   non-build ask cannot see it, because contract-less seats are
   vacuously admitted by design and `explainer.yaml` carries no
   contract, while `code-seat.yaml:32` does.

Record under `docs/plans/2026-08-17-154-live-gate/`.

The 13-turn ladder battery is NOT re-run, and the honest reason is not
"a battery would not exercise bad PATH". It is that where the battery
runs (venv on PATH) `python3` and `sys.executable` resolve to the same
binary, so the change is a NO-OP there and a battery would re-measure
the model rather than the change. The two-environment suite numbers
above are the coverage.

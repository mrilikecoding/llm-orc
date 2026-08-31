# #168 live gate — the refusal names no path, on the wire (2026-08-30)

**Exit gate MET, both legs.** With a crashed routing node whose error
message deliberately CONTAINED an absolute path, the refusal reaching the
wire is `Refused: serving pipeline error: no readable routing decision
this turn (resolve: exited non-zero, status 1); nothing was built or
written` — no separator, no username, no trace of the injected message.
The 2026-08-14 capture of the same scenario
(`../2026-08-14-152-live-gate/gate-build-ask.json`) carried
`Command '['python3', '/Users/nathangreen/.../resolve.py']'` verbatim;
that is the before-picture this converts.

Bonus row, found before the gate: **entry 9's fault precondition is
extinct on this main.** The deliberately PATH-broken serve (the 08-14
misfire replay) no longer crashes resolve at all — #154 runs `.py` under
llm-orc's own interpreter — and the same build ask returned a real
`write add.py` tool_call (`168-response.json`). The #168 gate therefore
needed a live fault injection rather than the recorded replay.

## Procedure

1. Stopped the stale serve found on :8765 (started 08-17, pre-dating
   every merge under test). Started `nohup .venv/bin/llm-orc serve
   --port 8765` with NO venv PATH prefix from repo root — the 08-14
   misfire precondition. Drove `168-ask.json` (the entry-9 build shape,
   write+read tools advertised): `finish: tool_calls`, a real
   `write add.py` call. Precondition extinct (#154). `168-response.json`.
2. Fault injection: appended
   `raise RuntimeError("injected crash naming /Users/nathangreen/secret-inject-probe-XYZ")`
   to `.llm-orc/scripts/agentic_serving/resolve.py` (uncommitted,
   restored after). **Mutant confirmed live first** (rule 16):
   `echo '{}' | .venv/bin/python .llm-orc/scripts/agentic_serving/resolve.py`
   crashes with the marker. Two earlier injection attempts landed inside
   the module docstring and were inert — the serve trace showed the full
   chain running, which is what running the mutant directly caught.
3. Direct wire (`168-inject3-response.json`): `finish: stop`, NO
   tool_calls, content = the sanitized refusal. Checks run on the body:
   `'/' in content` False, `'nathangreen' in content` False,
   `'secret-inject-probe-XYZ' in content` False.
4. Real client (`oc-out.json`): `opencode run "write a function that
   adds two numbers in add.py" --format json -m llm-orc/agentic` from an
   empty scratch dir (sandbox-disabled detach per the wedge note in
   memory). Events: zero `tool_use` occurrences, refusal text verbatim,
   `exited non-zero, status 1` wording present, no `add.py` on disk.
5. Restored `resolve.py` (`git status --porcelain` count 0), restored
   the good serve (venv PATH prefix), verified one real build turn
   returns a `write` tool_call — the gate refuses nothing legitimate.

## Notes

- The serve resolves its project as `Path.cwd()/".llm-orc"` at request
  time (`v1_chat_completions._resolve_serving_project_dir`), so the
  injection had to land in the repo's own tree; a scratch-copy project
  was tried first and never consulted because the serve was started from
  repo root.
- Owed rows still open after this gate: #166 #169 #173 (re-fix/build
  refusal shapes), #172 #176 (library source + gate participants), #175
  (env scrub) — each changes client-visible behaviour and gets its live
  row when the serve exercises those routes; this record's procedure
  (fault inject → confirm mutant live → drive both legs → restore) is
  the template.
- The stale serve found running predated v0.19.0's merges by nothing
  (same version) but today's ten merge units by all of them; live
  validation against a long-running serve validates the wrong code.
  Restart before every gate.

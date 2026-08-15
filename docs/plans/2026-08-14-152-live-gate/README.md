# #152 live gate — the misfire conditions replay as honest refusals (2026-08-14)

**Exit gate MET.** Under the exact conditions that shipped the 2026-08-13
junk write (a serve started WITHOUT the venv on PATH, so bare python3
lacks `llm_orc` and resolve.py + seat_contract.py crash on import), the
`fix/152-routing-fail-closed` branch refuses honestly on the wire:
zero tool calls, no file on disk, and the refusal names the crashed
node. The same conditions pre-fix produced `{"finish": false, "file":
"solution.py", "content": ""}` (captured in the pre-fix endpoint trace
during instrument development, matching the original
`discarded-gate-classify-badpath.jsonl` misfire).

## Procedure

1. Stopped the good serve; started `nohup .venv/bin/llm-orc serve
   --port 8765` with NO PATH prefix (`serve-badpath.log`) — the
   deliberate replay of the misfire precondition
   ([[opencode-run-wedge]] records the prefix rule).
2. `gate-build-ask.json` — direct wire, the junk-write shape ("write a
   function that adds two numbers in add.py", write tool advertised):
   `finish_reason: stop`, NO tool_calls, content =
   `Refused: serving pipeline error: no readable routing decision this
   turn (resolve: Schema JSON execution failed: ... exit status 1.);
   nothing was built or written`.
3. `gate-resolve-ask.json` — the #144 gate-2 ask ("how does resolve
   pick the seat?", read/glob/write advertised): same refusal, no tool
   calls. (First attempt omitted `tools` and got the toolless META
   reply — `_aux_reply` echoes the subject line by design; OpenCode
   always advertises tools, so the toolless path is out of scope here.)
4. `opencode-build-ask.json` — the REAL client (`opencode run --format
   json -m llm-orc/agentic`, sandbox-disabled detach per the wedge
   procedure): events `step_start → text → step_finish`, ZERO
   `tool_use` events, final text = the refusal verbatim. No `add.py`,
   no `solution.py` on disk afterward.
5. Restored the good serve (`PATH="$PWD/.venv/bin:$PATH"` prefix) and
   verified with one real build turn: `write add.py` tool_call carrying
   actual code — the readability gate refuses nothing legitimate.

## Notes

- The refusal's parenthetical names resolve's failure via the ADR-001
  schema-path wrap ("Schema JSON execution failed: ..."), not the
  captured `"Script failed with exit code 1"` wording — the engine has
  two crash-wrap layers and the live serve takes the schema path. The
  shape gate keys on decision READABILITY, not on either wrap's
  wording, which is why both render the same refusal.
- seat_contract also crashes under the bad PATH (it imports `llm_orc`
  and `yaml`) — inert as designed: `seat_admitted=None`, no gate fires
  (design doc, pre-flight finding 2).
- Interpreter fragility itself (bare `python3` from
  `_get_interpreter`) is filed as #154.

# #157 live check — the serve still builds with real timeouts (2026-08-17)

Not a gate in the #152/#154 sense (there is no honesty claim to
falsify here); a smoke check that giving every script node a real
subprocess bound does not break the pipeline that previously ran
unbounded.

Serve restarted on the branch, venv on PATH. Build ask through real
`opencode run`: one `write` tool_call, `add.py` on disk
(`written-add.py`), "Wrote add.py." That path runs the gated build,
so it exercises `accept_executor` — the node whose worst case (45s
aggregate + one 15s child = 60s) is the reason six nodes now declare
explicit timeouts instead of racing the shipped 60s default.

The ladder battery was NOT re-run. Every serving script completes in
under 0.2s on an 83KB payload (measured during design), so a 180s or
300s bound is not reachable by them, and a battery would re-measure the
model rather than the change. The instrument with teeth is the unmocked
wiring pin, which takes 30.87s against the pre-fix code (the full
sleep(30) runs unbounded) and 1.08s with the fix.

`add.py` was removed from the repo root after capture.

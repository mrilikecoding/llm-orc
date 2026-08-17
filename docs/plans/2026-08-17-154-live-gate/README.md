# #154 live gate — the bad-PATH serve works (2026-08-17)

**Exit gate MET.** Under the exact condition that produced the
2026-08-13 misfire and #152's live gate (a serve started from a shell
whose `python3` cannot import `llm_orc`), the serve now answers and
builds normally instead of refusing. This deliberately INVERTS #152's
gate expectation: #152 made the failure honest, #154 removes the
failure.

## Positive control (recorded first)

The one way this gate passes for the wrong reason is forgetting the
PATH condition, where the change is a no-op. From the same shell that
started the serve (`positive-control.txt`):

```
command -v python3: /usr/bin/python3
python3 -c import llm_orc: ModuleNotFoundError: No module named 'llm_orc'
```

The serve was then started in that shell:
`env -u VIRTUAL_ENV PATH=/usr/bin:/bin nohup .venv/bin/llm-orc serve`.
It was RESTARTED rather than reused, because `agent_runner.py:88-93`
caches a crashed script's failure envelope as `success: True` for 3600s
and a pre-fix result could otherwise replay.

## Both asks, mirroring #152's two-probe structure

**Ask 1 — `resolve.py` (`gate-1-opencode.json`).** "how does resolve
pick the seat?" through real `opencode run`: glob discovery round, then
a GROUNDED answer naming resolve.py's actual mechanics (the
classify/decide merge, `needs_decider`, `_decider_target`'s extraction,
the recall/recap handling). Under #152 this same ask on this same
serve returned `Refused: serving pipeline error ... (resolve: ...)`.

A direct-wire probe of the same ask (`gate-1-resolve-ask.json`) shows
the first round independently: `finish_reason: tool_calls` with a glob,
where the pre-fix serve refused before any tool call.

**Ask 2 — `seat_contract.py` (`gate-2-opencode.json`).** "write a
function that adds two numbers in add.py": a real `write` tool_call,
and `add.py` on disk (`gate-2-written-add.py`). This is the ask that
observes seat_contract, the second casualty the #152 record names: a
non-build ask cannot see it, because contract-less seats are vacuously
admitted by design and `explainer.yaml` carries no contract while
`code-seat.yaml:32` does. A write reaching the client proves the
contract node admitted rather than sat inert.

Note on that artifact: the deliverable itself is sloppy (a duplicated
function and a stray test body). That is qwen3:8b's output quality, not
an instrument or pipeline fault, and the gate is about the pipeline
running end to end. The file was removed from the repo root after
capture.

## Suite evidence (the instrument with teeth)

From a shell with no venv on PATH, the symptom the issue cites:

| | failures |
|---|---|
| before | 50 (39 in the serving endpoint suite, 11 in the BDD bare-`python` line) |
| after the interpreter fix | 11 |
| after the unrelated `["python", ...]` line was fixed too | **0 of 3394** |

The 11 were a separate pre-existing issue: BDD tests invoking a bare
`python`, which macOS does not provide. In the environment where CI and
`make test` run, `python3` and `sys.executable` are the same binary, so
the change is a no-op there and both environments now report 3394
passed.

## Follow-on

The #152 laundering pin now carries a discriminating assertion instead
of a docstring excusing its degradation, and was verified to FAIL
against the pre-fix interpreter under bare PATH. The ladder battery was
not re-run: where it runs, this change is a no-op, so it would
re-measure the model rather than the change.

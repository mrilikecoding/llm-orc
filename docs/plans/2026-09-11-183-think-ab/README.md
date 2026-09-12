# #183 slice B — `think: false` on code-generator's three agents, A/B (2026-09-11)

Two serves from the same tree except `code-generator.yaml`: main `1ac25bb4`
on :8765 (coder/critic/synthesizer omit `think`, which Ollama 0.31.1 treats
as thinking ON for qwen3) and this branch on :8777 (`options: {think:
false}` on all three). Same tools-bearing request (write+read advertised,
the #168 live-gate ask shape), `run.sh`, sequential so the two arms never
overlap on the GPU; arm order alternated per shape. Wall clock is curl's
end-to-end time of one serve turn.

| shape | ask | think ON (main) | think OFF (branch) | outcome both arms |
|---|---|---|---|---|
| s1 (first, cold) | write a function that adds two numbers in add.py | 126.9 s | 44.9 s | `write add.py`, correct |
| s1b (repeat) | same | 82.8 s | 38.2 s | `write add.py`, correct |
| s2 | create storage.py with save_todos and load_todos functions using json | 166.5 s | 59.5 s | `write storage.py`, correct |

2.2–2.8x faster per gated build turn, every arm gate-accepted, content
equivalent (`s*.json`). The 13-turn ladder is the accept-rate gate for
this change (#183 instrument B); it runs on the merged serve after #171
and #182-D land, one ladder validating all three. Note: the s1 main-arm
response body carried a second JSON object after the first (`s1-main-
thinkon.json`, 728 bytes); the first object is a valid `write add.py`
tool call. Not reproduced on s1b/s2; recorded, not diagnosed.

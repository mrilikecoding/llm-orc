# #144 live gate — serve-native dot-dir self-reference (2026-08-13)

Real OpenCode (1.17.15) against this checkout, serve on branch
`feat/144-dot-dir-self-reference` (flag committed on in
`.llm-orc/config.yaml`), driven via `opencode run -m llm-orc/agentic
--format json` from the repo root. Design:
`docs/plans/2026-08-13-dot-dir-self-reference-design.md`.

## Results

**Gate 1 — "how does classify decide routing?"** (`gate-classify.jsonl`):
glob round fired (`**/*{classify,decide,routing}*`; listing carries only
test files — the dot-dir is invisible to the client's glob, the premise of
the issue), union discovery resolved the serve's own
`.llm-orc/scripts/agentic_serving/classify.py`, the native self-read ran
the SAME budget discipline, and the turn refused honestly with the
single-whale wording:

> Refused: could not read .llm-orc/scripts/agentic_serving/classify.py:
> its content alone exceeds the session read budget. The file is too
> large to ground an answer in one read.

No client read tool_call for the dot-path, no speculation. Trace:
`need-glob` → re-entry (`self_read_round: 1`) → `need-files` (the
recorded-attempt refusal). This is the designed outcome: the literal
"answers grounded" exit stays open on #144, blocked by the whale file
(unblockers #106 / #151 / chunked reads), not by discovery.

**Gate 2 — "how does resolve pick the seat?"** (`gate-resolve.jsonl`):
glob round → zero workspace candidates (htmlcov noise correctly rejected
by the subset rule) → sole self candidate → native read of resolve.py
(3,190 projected tokens, admits) → GROUNDED explain. The answer names
resolve.py's actual mechanics: the classify/decide merge on
`needs_decider`, `_decider_target`'s strict-JSON-then-single-token
extraction, the recall/recap special handling with the explainer
fallback, and the shape-catalog intent→shape mapping. All verifiably
real; no invented behavior observed. Trace: `need-glob` →
`need-self-files` → re-entry (`self_read_round: 1`) → `explainer`.

## Misfire record (kept per protocol rule 8)

`discarded-gate-classify-badpath.jsonl`: the first attempt ran against a
serve restarted with a PATH where bare `python3` (the script-agent
interpreter) lacked `llm_orc` — resolve.py crashed on its
`shape_catalog` import, and the failure CASCADED to a junk empty
`solution.py` write: shape's decision fallback read the failed resolve
envelope as a routing decision, defaulted `build=True`, and emit shipped
an empty valid deliverable. Ops fix: serve started with the venv's bin
first on PATH. The cascade itself is a real pre-existing honesty gap
(a crashed routing node should fail closed, never degrade to a build) —
filed as its own issue.

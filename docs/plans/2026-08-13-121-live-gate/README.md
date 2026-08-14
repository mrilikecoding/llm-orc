# #121 live gate — content-grep rung, slice A (2026-08-13)

Real OpenCode (1.17.15) against this checkout, serve on branch
`feat/121-content-grep` (self_reference ON), `opencode run -m
llm-orc/agentic --format json` from the repo root. Design:
`docs/plans/2026-08-13-content-grep-design.md` (v2.1). Rig state: stale
worktrees removed earlier today; classify.py at 102KB (over the serve's
96KB cap), serving_ensemble_caller.py at 80KB (over the CLIENT's 50KB
read cap — see the discovery below).

## Results

**GROUNDED PASS (`gate-truncation-run3.jsonl`):** "how does the serve
detect runtime truncation?" → glob (complete listing) → def-anchored
grep round → menu → pick → client read of
`src/llm_orc/web/serving/turn_trace.py` → grounded answer citing the
file's REAL content (`_DEEP_TRUNCATION_RATIO = 0.35`, the dual-threshold
detection logic — verifiably actual). The designed chain end to end
through a real client, AST-confirmed.

**Honest adjacent-grounding (`gate-recall-ledger.jsonl`):** "where is
the recall ledger built?" → the pick chose `_RECALL_TARGET` →
resolve.py → grounded explain that EXPLICITLY states the recall ledger
is not built in this file and points onward — the arm-I
adjacent-grounding class, live, with self-aware honest attribution. No
fabrication.

**THE DISCOVERY (`gate-recall-ledger-run2.jsonl`, pre-fix):** the pick
chose a caller-defined identifier; OpenCode's read tool CAPPED the 80KB
caller at 50KB ("Output capped at 50 KB. Showing lines 1-1104...") —
inside the serve's 96KB window, so the partial content rendered as a
complete-looking read. The #121 AST confirmation caught it (refusing,
with the then-wrong reason "does not define"). Filed as #153; fixed in
this branch: the trailer is detected deterministically and a capped
read renders as a refusing `(truncated)` variant. Post-fix runs
(`gate-truncation.jsonl`, `gate-truncation-run2.jsonl`) show the honest
wording: "Refused: could not read …: the client truncated the read
(file exceeds the client's read cap)".

**Recorded bounds observed live:**
- Pick sampling on caller-heavy menus: the 80KB caller defines
  something for nearly every serve-word, attracts picks, and cannot
  ground under the client cap — so common-stem questions frequently end
  in the honest cap refusal rather than a grounded answer (1 grounded /
  2 cap-refusals on the truncation question's three runs). #153's
  offset-continuation reads are the named recovery.
- Truncated glob listings gate the trigger (`gate-sandbox.jsonl`): "how
  does the accept executor run the gate sandbox?" carries five common
  stems, the listing blows the 50-path render cap, and the turn falls
  through conceptually BEFORE the grep round — the #148-inherited
  complete-listing requirement, recorded as a slice-A bound (the
  conceptual answer is the pre-existing behavior, unchanged).

Zero fabrication in every run; every grounded claim spot-checked
against the real file content.

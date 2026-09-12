# 13-turn ladder runs, 2026-09-11 — #171 merged, `think` on/off for code-generator (#183 slice B)

Fixture: fresh git repo seeded per `ladder_battery.sh`'s header (`calc.py`,
`buggy.py`, `metrics.py` byte-identical to the 2026-08-14 run-10 seed
hashes; `test_buggy.py` functionally identical, different bytes). Same
seed reset before every run. One `llm-orc serve` per run on :8765,
started from the tree named; real `opencode run --format json`; truth per
turn in a throwaway copy (`capture_truth.sh`). Scored with
`score_run.score_run_dir` + `tally_oracles` (hidden oracles cover turns
1, 6, 7). Per-turn tables: the `run*/` dirs hold every `turn-NN.jsonl`,
`truth-NN.json`, `exits.tsv`.

| run | tree | code-generator `think` | wall (sum of turns) | rounds | dishonest | misses (turn: what) |
|---|---|---|---|---|---|---|
| 1 | main 8f7e9780 (#171 + think-off) | coder/critic/synth OFF | 464 s | 15 | 0 | T1 shipped BROKEN (oracle: no public callable adds the item to an existing list); T6 REFUSED (deliverable not valid Python: a box-drawing `├` at line 31); T7 cascade (storage.py absent) |
| 2 | cb732246 (#171, think untouched = ON) | all ON | 954 s | 16 | 0 | T1 shipped BROKEN (same miss); T7 shipped (oracle true) but the turn's last text says "another round needed: code failed to load"; T12 REFUSED (tests did not pass) |
| 3 | main 8f7e9780 | all OFF | 468 s | 12 | 0 | T1 REFUSED (tests did not pass; inadequate) → T2/T3/T4 cascade; T6 REFUSED; T7 cascade — 7/13 |
| 4 | main 8f7e9780 | all OFF | 377 s | 14 | 0 | T6 REFUSED (tests did not pass); T7 cascade; T12 REFUSED |
| 5 | main 3be5c82a (coder ON, critic+synth OFF) | coder ON | 1004 s | 17 | 0 | none on disk (oracles 3/3, every deliverable shipped); T7's last text says "another round needed: code failed to load: ModuleNotFoundError: No module named 'storage'" while the shipped todo.py is correct (oracle true) — see #184 |
| 6 | main 3be5c82a | coder ON | 1026 s | 17 | 0 | same as run 5: all shipped, oracles 3/3, T7's text wrong-direction |
| 7 | main 01b59026 (arcs 1+2 merged) | coder ON | 920 s | 15 | 0 | T1 REFUSED (tests did not pass, after the new glob round); T6 REFUSED (tests inadequate); T7 cascade; T13 REFUSED: the coder wrote `raise ValueError("empty input")` where the seeded test expects `match="no values"` — an honest refusal of a wrong fix, first time this rung missed in seven runs. No sandbox-, surface-, or ablation-specific reason fired. n=1; run 8 is the second sample. |
| 8 | main 01b59026 (arcs 1+2 merged) | coder ON | 666 s | 16 | 0 | T1 REFUSED (tests did not pass) → T2/T3/T4/T7 cascade (no todo.py ever lands); T6 shipped correct; T13 FIXED (10 passed) — run 7's T13 miss was variance. 8/13. |

What the four say (n is small; doctrine 6 applies to any single turn):

- **Think-off on all three agents is a delivery regression on the gated
  build shape.** T6 ("create storage.py with save_todos and load_todos
  functions using json") refused in 3/3 think-off runs and shipped
  correct in the 1/1 think-on run; T1 was lost once under think-off. The
  wall clock halves (464/468/377 s vs 954 s). Halving latency by turning
  successful builds into "another round needed" is the wrong trade for a
  daily driver, so the coder's thinking is restored (3be5c82a) and only
  the critic and synthesizer stay think-off — the evaluator's own "safest
  first step" (`docs/plans/2026-09-11-90-llamacpp-eval/README.md` §3.1),
  measured in runs 5-6: **2/2 runs ship every deliverable (oracles 3/3)** at
  1004/1026 s — the same wall clock as all-on (954 s), so the critic/
  synthesizer setting saves nothing measurable and costs nothing; the
  coder's thinking is where both the time and the delivery live. #183
  slice B's answer: keep the coder thinking; the latency lever is not
  `think`.
- **T1's oracle miss is not think-related**: it appeared with thinking on
  (run 2) and off (runs 1, 3) and passed once with thinking off (run 4).
  The coder reads "adds a todo item to a list" literally (appends the
  string "todo item" to a copied list). Variance, or a prompt-shape
  weakness; n=4 cannot say which.
- **No ablation-control refusal appeared in any run** (#171's new refusal
  reason never fired on the ladder): the four runs' refusals are "tests
  did not pass", "not valid Python", "could not read", and "code failed to
  load". #171's live row is still owed on a constructed non-participating
  shape.
- **Honesty column held at zero** across all four runs (mechanical
  classifier; not J-scored).
- The 13/13 headline in the parity table was measured on a different tree
  (v0.18.x, July) with n=3; these four runs land at 10, 10-11, 7, 10 on
  the strict reading. Whether #171 itself moved the rate needs runs on a
  pre-#171 tree under the same seed — not done here; recorded as the open
  question.

**After runs 7-8 (the merged arcs):** T13 fixed in run 8, so run 7's miss
there was variance. The rung that decides the ladder now is **T1**
("write a function that adds a todo item to a list in todo.py"): over
eight runs it shipped correct 3 times (4, 5, 6), shipped broken twice
(1, 2), and refused three times (3, 7, 8); every refusal cascades into
T2/T3/T4/T7 because nothing named `todo.py` exists afterwards. The
refusal is honest (the coder's module fails the test-writer's tests) and
predates the arcs (run 3), but both post-arc runs hit it — n=2, doctrine
6 forbids a per-turn verdict. It is the greenfield first-turn shape, not
an existing-repo one: worth its own instrument (r≥5 of T1 alone against
the seat, ~1 min each) before any change to the build round. No sandbox-,
surface-, participation-, or workspace-refusal reason fired in either
post-arc run.


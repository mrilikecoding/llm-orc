# Arc 1 (workspace-aware routing) — live rows on merged main `acfaf427` (2026-09-11, late)

Real OpenCode, fresh session per row, probe seed reset before each
(`docs/plans/2026-09-11-daily-driver-probe/seed-repo.tgz`).

| row | ask | events | outcome | truth |
|---|---|---|---|---|
| 12 | the probe's turn-2 ask verbatim ("Add a remove(todo_id) method to the TodoStore class in todo/storage.py ...") | `glob **/*py*` → `read todo/storage.py` → `write todo/storage.py` | **CONVERTED**: the whole module, every prior method (`_load`, `_save`, `add`, `list`, `complete`) kept, `remove` added (`row12-storage.py`) | `M todo/storage.py`; the seed's `tests/test_storage.py` 3 passed. On main two hours earlier the same ask overwrote the file blind (#185, row 10); on the 17:00 main it was refused (probe turn 2). |
| 13 | the greenfield control ("Write a function that adds two numbers in add.py.") | `glob` → `write add.py` | correct, one extra wire round | `?? add.py`; 3 passed |

Exit gate of `docs/plans/2026-09-11-workspace-aware-routing-design.md`
met on both rows. The 13-turn ladder rerun rides the arc-2 merge (one
ladder validates both arcs).

## Arc 2 rows on merged main `01b59026` (2026-09-12, early)

| row | ask | events | outcome |
|---|---|---|---|
| 14 | probe turn 1 (add `remove()` to `todo/storage.py` AND tests in `tests/test_storage.py`) | `read todo/storage.py` → tests-seat | "Another round needed: tests did not pass" — unchanged: the tests-primary route still writes tests for a method that does not exist yet. Arc 2 changes what the sandbox looks like, not which seat runs; this row converts with #123 (code + tests per turn) and the tests-destination rule (arc 2 mechanism 5, deferred to classify). |
| 15 | probe turn 3 (`done` CLI command AND `tests/test_cli.py`) | `read todo/cli.py` → tests-seat | same |

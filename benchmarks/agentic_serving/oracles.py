"""Hidden build-turn correctness oracles for the WS-8 strict score (#131).

WHY THESE EXIST. The strict rule is that a turn passes only when its
deliverable ships AND IS CORRECT. On Arm 0 the accept gate stands between a
build and the workspace, so "shipped" implies "verified". A frontier arm behind
OpenCode has no gate, so "shipped" means merely "emitted". Scoring `wrote the
file == pass` would therefore hand a free pass to an arm that ships
plausible-but-wrong code while Arm 0 pays for every honest reject — biasing the
score toward the discretionary verification WS-8 exists to test against. These
oracles are the independent, arm-blind correctness check that removes that bias
(see `docs/plans/2026-07-14-strict-per-turn-table-design.md` §2).

WHY BEHAVIORAL, NOT NAME-KEYED. Turn 1's prompt does not name the function.
Measured 2026-07-14: the same seat shipped `add_todo` and `add_todo_item` for
that identical prompt across two runs, so a name-keyed check would score the
same arm differently run to run — non-deterministic scoring, which disqualifies
it as an instrument. Turn 1's oracle therefore accepts ANY public callable
meeting the behavioral contract. Turns 6 and 7 DO name their API, so those
oracles key on the name and tolerate only the free part (signature order).

WHY SUBPROCESS. Each probe runs in a fresh interpreter with the workspace as
cwd: the modules under test are model-written, and a fresh process keeps them
out of the scorer's own namespace and stops one oracle's import state from
leaking into the next. This mirrors the accept executor's per-test isolation,
which the 2026-07-09 seat-quality arc measured as the fix for the dominant
false-reject class. Probes are deterministic and read no network.

TOLERANCE IS BOUNDED, NOT OPEN. Every oracle is pinned by fixtures against both
correct variants (FRR) and plausible-but-wrong ones (FAR) in
`tests/test_oracles.py`, the #84 adequacy-checker methodology. An oracle that
accepts anything that merely runs is worse than no oracle.

NOT ORACLED: turn 2. "add a complete_todo function to todo.py that marks a todo
done" names the function but leaves the REPRESENTATION of done-ness free (bool
field, status string, separate collection). An oracle that cannot pin the
representation either accepts `def complete_todo(x): pass` (FAR 1 on the real
contract) or rejects valid designs. Turn 2 stays hand-classified until the v2
battery specifies the shape — the concrete case for tightening that prompt.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class OracleResult:
    """One oracle's verdict. ``detail`` carries the probe's own output so a
    verdict is auditable rather than a bare bool."""

    passed: bool
    detail: str = ""


def _run_probe(workspace: Path, program: str) -> OracleResult:
    """Run ``program`` in a fresh interpreter with ``workspace`` as cwd.

    Exit 0 means the contract held. Any other exit, a crash, or a timeout fails
    CLOSED: an oracle that cannot demonstrate correctness must not assert it.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-c", program],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return OracleResult(False, "probe timed out")
    except OSError as exc:
        return OracleResult(False, f"probe could not start: {exc}")
    detail = (proc.stdout + proc.stderr).strip()[-300:]
    return OracleResult(proc.returncode == 0, detail)


# Turn 1 is NAME-FREE, so the probe searches for any public callable defined in
# todo.py that puts the item into the list. It accepts mutate-in-place and
# return-a-new-list styles, and either argument order.
_TURN1_PROBE = """
import inspect, sys
try:
    import todo
except Exception as exc:
    print("import failed:", exc); sys.exit(1)
candidates = [
    fn for name, fn in inspect.getmembers(todo, inspect.isfunction)
    if not name.startswith("_") and getattr(fn, "__module__", None) == "todo"
]
SENTINEL = "milk-oracle-sentinel"
for fn in candidates:
    for swapped in (False, True):
        items = []
        try:
            returned = fn(SENTINEL, items) if swapped else fn(items, SENTINEL)
        except Exception:
            continue
        if SENTINEL in items:
            print("ok: mutated in place via", fn.__name__); sys.exit(0)
        if isinstance(returned, (list, tuple)) and SENTINEL in returned:
            print("ok: returned new list via", fn.__name__); sys.exit(0)
print("no public callable put the item into the list"); sys.exit(1)
"""


# Turn 6 NAMES save_todos/load_todos; only the signature order is free.
_TURN6_PROBE = """
import os, sys
try:
    import storage
except Exception as exc:
    print("import failed:", exc); sys.exit(1)
save = getattr(storage, "save_todos", None)
load = getattr(storage, "load_todos", None)
if not callable(save) or not callable(load):
    print("save_todos/load_todos not both present and callable"); sys.exit(1)
DATA = ["milk", "eggs"]
for swapped in (False, True):
    path = "oracle_probe_%s.json" % swapped
    try:
        save(path, DATA) if swapped else save(DATA, path)
        got = load(path)
    except Exception:
        continue
    finally:
        if os.path.exists(path):
            os.remove(path)
    try:
        if list(got) == DATA:
            print("ok: round-tripped, swapped=", swapped); sys.exit(0)
    except TypeError:
        continue
print("no signature order round-tripped the data"); sys.exit(1)
"""


# Turn 7 must persist VIA storage.py, so the probe checks todo.py's real
# namespace after import. A source substring check would be fooled by the word
# "storage" in a comment; only an executed import binds the name.
_TURN7_PROBE = """
import sys, types
try:
    import todo
except Exception as exc:
    print("import failed:", exc); sys.exit(1)
for name, obj in vars(todo).items():
    if name.startswith("__"):
        continue
    if isinstance(obj, types.ModuleType) and obj.__name__ == "storage":
        print("ok: imports the storage module"); sys.exit(0)
    if getattr(obj, "__module__", None) == "storage":
        print("ok: imports", name, "from storage"); sys.exit(0)
print("todo.py does not import or use storage"); sys.exit(1)
"""


def turn1_adds_todo(workspace: Path) -> OracleResult:
    """Turn 1: some public callable in todo.py adds an item to a list."""
    return _run_probe(workspace, _TURN1_PROBE)


def turn6_storage_roundtrip(workspace: Path) -> OracleResult:
    """Turn 6: storage.save_todos/load_todos round-trip todos through a file."""
    return _run_probe(workspace, _TURN6_PROBE)


def turn7_todo_persists(workspace: Path) -> OracleResult:
    """Turn 7: todo.py actually composes with storage.py rather than
    reimplementing persistence itself."""
    return _run_probe(workspace, _TURN7_PROBE)


# Turn 2 is absent by design (see module docstring): its contract leaves the
# representation of done-ness free, so it stays hand-classified.
ORACLES: dict[int, Callable[[Path], OracleResult]] = {
    1: turn1_adds_todo,
    6: turn6_storage_roundtrip,
    7: turn7_todo_persists,
}


def probe(turn: int, workspace: Path) -> dict[str, object] | None:
    """The oracle verdict for ``turn``, or None when that turn has no oracle.

    Called by the battery immediately after the turn runs. It MUST run then,
    not post-hoc over the final workspace: later turns mutate files (turn 13
    rewrites buggy.py), so a end-of-run probe would score a turn against a
    workspace that turn never saw.
    """
    oracle = ORACLES.get(turn)
    if oracle is None:
        return None
    result = oracle(workspace)
    return {"passed": result.passed, "detail": result.detail}


if __name__ == "__main__":  # pragma: no cover - the battery's entry point
    import json

    verdict = probe(int(sys.argv[1]), Path(sys.argv[2]))
    print(json.dumps(verdict))

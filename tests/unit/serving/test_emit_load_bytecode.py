"""Loading a project's emit.py leaves no bytecode beside it (#196).

The packaged serving tier is read-only; the caller execs its emit.py
in-process, which would otherwise write ``__pycache__`` next to it.
"""

from __future__ import annotations

import sys
from pathlib import Path

from llm_orc.web.serving.serving_ensemble_caller import _load_emit_reject_prefixes

_EMIT = """\
from types import SimpleNamespace

TERMINALS = {
    "reject": SimpleNamespace(prefix="Rejected: ", mints="rejected"),
}
"""


def test_emit_load_writes_no_bytecode(tmp_path: Path) -> None:
    emit = tmp_path / "emit.py"
    emit.write_text(_EMIT)
    before = sys.dont_write_bytecode
    sys.dont_write_bytecode = False
    try:
        prefixes = _load_emit_reject_prefixes(emit)
        restored = sys.dont_write_bytecode
    finally:
        sys.dont_write_bytecode = before

    assert [p.prefix for p in prefixes] == ["Rejected: "]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["emit.py"]
    assert restored is False

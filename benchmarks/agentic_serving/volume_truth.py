"""Shared ground-truth capture for the #138 volume ladder.

ONE implementation, called by every arm's driver, so the truth substrate
cannot drift between arms (the ladder's rule — this is the volume
ladder's equivalent of ``capture_truth.sh``). It judges what reached
disk, identically for every arm: a write tool, a bash heredoc, and a
patch all land in the manifest the same way.

Per level it records, for each module:

- the hashed manifest, diffed by the scorer against the level's SEEDED
  manifest (``volume_fixture.level_manifest``). Every level starts from
  the seed by construction, so the baseline is the seed itself and no
  prior turn can contaminate it.
- the module's own seeded test result (``pytest test_<module>.py``), so
  one module's breakage never marks another's.
- the module's hidden oracle verdict.

WHY A COPY: the truth pytest runs arm-authored tests, which import
arm-authored modules, which execute arm-authored module-level code; the
probes CALL arm-authored functions. Run live, a test that writes a file
would land in the next manifest and be attributed to the arm. Both run
against a throwaway copy, so capture can never corrupt what it scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from benchmarks.agentic_serving.volume_fixture import level_manifest, level_modules
from benchmarks.agentic_serving.volume_oracles import run_volume_oracle

_IGNORED = shutil.ignore_patterns(".git", "__pycache__", "*.pyc", ".pytest_cache")
_PYTEST_TIMEOUT_SECONDS = 120


def _manifest(workspace: Path) -> dict[str, str]:
    """``{name: sha256}`` for the workspace's python files. Hashed, not
    named: names alone cannot show that a turn EDITED a file, which is
    exactly what a fix turn does."""
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(workspace.glob("*.py"))
    }


def _seeded_rc(sandbox: Path, module: str, pytest_command: Sequence[str]) -> int | None:
    """The module's OWN seeded test result: ``0`` green, ``1`` red, or
    ``None`` when no verdict was obtained. Per module, never the whole
    suite: at L5 a whole-suite rc would mark all five modules red because
    one was missed.

    ``None`` is the load-bearing state. pytest returns 0/1 for a real
    verdict and 2-5 for interrupted / internal error / usage error / no
    tests collected — an interpreter without pytest, a deleted test file,
    or a hung run are INSTRUMENT failures, and the first draft recorded
    them as rc 1, which read a perfect arm as 100% shipped-broken while
    the driver exited 0 (the ladder's quiet-corruption family)."""
    test_file = f"test_{module}.py"
    if not (sandbox / test_file).exists():
        return None
    try:
        completed = subprocess.run(
            [*pytest_command, "-q", "--no-header", "-p", "no:cacheprovider", test_file],
            cwd=str(sandbox),
            capture_output=True,
            text=True,
            timeout=_PYTEST_TIMEOUT_SECONDS,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    return completed.returncode if completed.returncode in (0, 1) else None


def capture_truth(
    workspace: Path,
    level: int,
    exit_code: int,
    out_dir: Path | None = None,
    pytest_command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Capture one level's ground truth; write ``truth-L<level>.json``
    into ``out_dir`` when given, and return the record either way."""
    modules = level_modules(level)
    command = list(pytest_command or (sys.executable, "-m", "pytest"))
    with tempfile.TemporaryDirectory() as tmp:
        sandbox = Path(tmp) / "ws"
        shutil.copytree(workspace, sandbox, ignore=_IGNORED, dirs_exist_ok=True)
        seeded_rc: dict[str, int | None] = {
            module: _seeded_rc(sandbox, module, command) for module in modules
        }
    truth: dict[str, Any] = {
        "level": level,
        "modules": list(modules),
        "baseline_manifest": level_manifest(level),
        "manifest": _manifest(workspace),
        "seeded_rc": seeded_rc,
        "oracles": {
            module: run_volume_oracle(workspace, module).passed for module in modules
        },
        "exit_code": exit_code,
    }
    if out_dir is not None:
        (out_dir / f"truth-L{level}.json").write_text(json.dumps(truth, indent=2))
    return truth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--pytest",
        default="",
        help="pytest command, space-separated (default: this interpreter's -m pytest)",
    )
    args = parser.parse_args()
    capture_truth(
        args.workspace,
        level=args.level,
        exit_code=args.exit_code,
        out_dir=args.out,
        pytest_command=args.pytest.split() or None,
    )


if __name__ == "__main__":
    main()

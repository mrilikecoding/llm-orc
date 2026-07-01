#!/usr/bin/env python3
"""Ω-P3 score (live): the real execution gate.

Terminal of the live serving flow. Gathers the fanned per-file deliverables,
writes them to an isolated temp dir (outside the repo, to dodge the repo's
pytest-cov addopts), runs the package's test(s), and reports the result — the
honest truth-teller the Ω-exec spike established.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

# When OMEGA_P3_KEEP is set, copy the built package + pytest output here for
# inspection (the temp dir is otherwise removed).
KEEP_DIR = Path(__file__).resolve().parents[3] / "scratch" / "spike-omega-p3" / "last_build"


def _instances(deps: dict) -> list[Any]:
    raw = deps.get("build", {}).get("response", "[]")
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def _deliverable(inst: Any) -> dict:
    try:
        child = json.loads(inst) if isinstance(inst, str) else inst
        deliv = child.get("deliverable", "{}") if isinstance(child, dict) else "{}"
        return json.loads(deliv) if isinstance(deliv, str) else deliv
    except (json.JSONDecodeError, ValueError):
        return {}


def main() -> None:
    try:
        deps = json.loads(sys.stdin.read()).get("dependencies", {})
    except (json.JSONDecodeError, ValueError, AttributeError):
        deps = {}
    files = [d for d in (_deliverable(i) for i in _instances(deps)) if d.get("file")]

    tmp = Path(tempfile.mkdtemp(prefix="omega_p3_exec_"))
    try:
        for d in files:
            (tmp / d["file"]).write_text(d.get("content", ""))
        test_files = [
            d["file"]
            for d in files
            if "test" in d["file"].lower() and d["file"].endswith(".py")
        ]
        execution = "no test file"
        pytest_output = ""
        if test_files:
            r = subprocess.run(  # noqa: S603
                [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                 "-o", "addopts=", *test_files],
                cwd=tmp, capture_output=True, text=True,
            )
            pytest_output = r.stdout + r.stderr
            tail = pytest_output.strip().splitlines()
            execution = f"rc={r.returncode} | {tail[-1] if tail else ''}"

        if os.environ.get("OMEGA_P3_KEEP"):
            shutil.rmtree(KEEP_DIR, ignore_errors=True)
            shutil.copytree(tmp, KEEP_DIR)
            (KEEP_DIR / "_pytest_output.txt").write_text(pytest_output)
        summary = [
            {"file": d["file"], "tier": d.get("tier"), "bytes": len(d.get("content", ""))}
            for d in files
        ]
        print(
            json.dumps(
                {
                    "count": len(files),
                    "built": summary,
                    "test_files": test_files,
                    "execution": execution,
                }
            )
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()

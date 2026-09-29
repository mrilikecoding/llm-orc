"""Pin the wheel's serving project to the repo's tracked `.llm-orc/` (#196).

Usage: python scripts/check_wheel_contents.py <path/to/llm_orchestra-*.whl>

The wheel maps `.llm-orc/{ensembles,profiles,scripts,config.yaml}` to
`llm_orc/serving_project/`. This checker compares that set against
`git ls-files` for the same four roots: a wheel with no serving project,
a wheel that shipped a gitignored or untracked file, and a wheel missing
a tracked one all fail, and the diff is printed. Run from the repo root.
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

PREFIX = "llm_orc/serving_project/"
SHIPPED_ROOTS = ("ensembles/", "profiles/", "scripts/", "config.yaml")


def tracked_serving_files(repo: Path) -> set[str]:
    out = subprocess.run(
        ["git", "ls-files", "--", ".llm-orc"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    files = set()
    for line in out.splitlines():
        rel = line.removeprefix(".llm-orc/")
        if rel.startswith(SHIPPED_ROOTS):
            files.add(rel)
    return files


def packaged_files(wheel: Path) -> set[str]:
    with zipfile.ZipFile(wheel) as zf:
        return {
            name.removeprefix(PREFIX)
            for name in zf.namelist()
            if name.startswith(PREFIX)
        }


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    wheel = Path(argv[1])
    expected = tracked_serving_files(Path.cwd())
    actual = packaged_files(wheel)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if not missing and not extra:
        print(f"ok: {len(actual)} serving project files match git ls-files")
        return 0
    for rel in missing:
        print(f"missing from wheel: {PREFIX}{rel}")
    for rel in extra:
        print(f"not tracked but shipped: {PREFIX}{rel}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))

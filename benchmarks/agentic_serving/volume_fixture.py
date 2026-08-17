"""#138 volume-ladder fixture generator.

Materializes one LEVEL of the volume ladder as a fresh git workspace
containing ONLY that level's modules and their seeded tests (design:
docs/plans/2026-08-15-138-volume-instrument-design.md). Per-level
subsetting is the pre-flight's blocker-1 fix: a full clone would leak
every red test to the first pytest run — the very verification act the
instrument measures — and guarantee the serve's whole-suite need-run
round a red verdict below L5.

Every module seeds exactly one GREEN and one RED test case, so a pytest
run in a fresh workspace fails once per module and the canonical 1-3
line fix flips exactly that failure. Bytes are deterministic; runs
verify their seed against the hashed manifest like the ladder's run-6
baseline discipline.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

MODULE_ORDER: tuple[str, ...] = ("ledger", "qty", "window", "rate", "label")
LEVELS: tuple[int, ...] = (1, 2, 3, 5)

VOLUME_PROMPTS: dict[int, str] = {
    1: "fix the bug in ledger.py",
    2: "fix the bugs in ledger.py and qty.py",
    3: "fix the bugs in ledger.py, qty.py, and window.py",
    5: "fix the bugs in ledger.py, qty.py, window.py, rate.py, and label.py",
}

# Seeded bytes. Flaw classes per the design table: missing raise,
# missing normalize, boundary, unit conversion, case-fold. Each test
# file: one green case (passes on the seeded flaw) + one red case.
_FILES: dict[str, str] = {
    "ledger.py": (
        "def balance(entries):\n"
        "    total = 0\n"
        "    for amount in entries:\n"
        "        total += amount\n"
        "    return total\n"
    ),
    "test_ledger.py": (
        "import pytest\n"
        "\n"
        "from ledger import balance\n"
        "\n"
        "\n"
        "def test_balance_sums_amounts():\n"
        "    assert balance([3, 4]) == 7\n"
        "\n"
        "\n"
        "def test_balance_empty_raises():\n"
        '    with pytest.raises(ValueError, match="no entries"):\n'
        "        balance([])\n"
    ),
    "qty.py": "def parse_qty(text):\n    return int(text)\n",
    "test_qty.py": (
        "from qty import parse_qty\n"
        "\n"
        "\n"
        "def test_parse_qty_plain():\n"
        '    assert parse_qty("7") == 7\n'
        "\n"
        "\n"
        "def test_parse_qty_decimal_string():\n"
        '    assert parse_qty("7.0") == 7\n'
    ),
    "window.py": "def last_n(items, n):\n    return items[-n:]\n",
    "test_window.py": (
        "from window import last_n\n"
        "\n"
        "\n"
        "def test_last_n_tail():\n"
        "    assert last_n([1, 2, 3], 2) == [2, 3]\n"
        "\n"
        "\n"
        "def test_last_n_zero_is_empty():\n"
        "    assert last_n([1, 2, 3], 0) == []\n"
    ),
    "rate.py": "def per_hour(count, minutes):\n    return count / minutes\n",
    "test_rate.py": (
        "from rate import per_hour\n"
        "\n"
        "\n"
        "def test_per_hour_zero_count():\n"
        "    assert per_hour(0, 30) == 0\n"
        "\n"
        "\n"
        "def test_per_hour_half_hour():\n"
        "    assert per_hour(30, 30) == 60\n"
    ),
    "label.py": ('def slug(title):\n    return title.replace(" ", "-")\n'),
    "test_label.py": (
        "from label import slug\n"
        "\n"
        "\n"
        "def test_slug_hyphenates():\n"
        '    assert slug("a b") == "a-b"\n'
        "\n"
        "\n"
        "def test_slug_lowercases():\n"
        '    assert slug("My Day") == "my-day"\n'
    ),
}


def level_modules(level: int) -> tuple[str, ...]:
    """The nested module subset for a level — the first-N prefix of
    ``MODULE_ORDER`` (repeated-measures comparability: ledger is
    observed under 1x, 2x, 3x, and 5x load)."""
    if level not in LEVELS:
        raise ValueError(f"unknown level {level}; levels are {LEVELS}")
    return MODULE_ORDER[:level]


def _level_files(level: int) -> dict[str, str]:
    names: list[str] = []
    for module in level_modules(level):
        names.extend((f"{module}.py", f"test_{module}.py"))
    return {name: _FILES[name] for name in names}


def level_manifest(level: int) -> dict[str, str]:
    """The level's SEEDED ``{path: sha256}`` manifest, without writing
    anything. Truth capture diffs against this: every level starts from
    the seed by construction, so a level's baseline is the seed itself,
    never a prior turn's workspace."""
    return {
        name: hashlib.sha256(body.encode()).hexdigest()
        for name, body in _level_files(level).items()
    }


def write_fixture(dest: Path, level: int) -> dict[str, str]:
    """Materialize the level's workspace at ``dest`` (git repo, one seed
    commit, clean tree) and return its ``{path: sha256}`` manifest."""
    files = _level_files(level)
    dest.mkdir(parents=True, exist_ok=False)
    for name, body in files.items():
        (dest / name).write_text(body)
    for command in (
        ["git", "-C", str(dest), "init", "-q"],
        ["git", "-C", str(dest), "add", "-A"],
        [
            "git",
            "-C",
            str(dest),
            "-c",
            "user.name=volume-fixture",
            "-c",
            "user.email=volume-fixture@local",
            "commit",
            "-q",
            "-m",
            f"seed volume fixture level {level}",
        ],
    ):
        subprocess.run(command, check=True, capture_output=True)
    return level_manifest(level)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="print the path<TAB>sha256 manifest after writing",
    )
    args = parser.parse_args()
    manifest = write_fixture(args.dest, args.level)
    if args.verify:
        for name in sorted(manifest):
            print(f"{name}\t{manifest[name]}")


if __name__ == "__main__":
    main()

"""Both-direction fixture pins for the #138 volume oracles (#84 method).

Every probe is pinned twice: the canonical fix PASSES, and the seeded
flaw FAILS. Where the oracle is supposed to discriminate
teaching-to-the-test (qty, rate, label), a hardcode of the seeded test's
exact expectation is pinned as a FAIL too — that is the whole reason a
hidden oracle sits behind the visible seeded test.

Recorded non-discrimination (design, pre-flight finding 9): for ledger
and window the seeded expectation IS the correct general fix, so no
hardcode fixture exists for them and the record must not credit the
oracle layer there.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.agentic_serving.volume_fixture import MODULE_ORDER, write_fixture
from benchmarks.agentic_serving.volume_oracles import (
    VOLUME_PROBES,
    run_volume_oracle,
)

CORRECT: dict[str, str] = {
    "ledger": (
        "def balance(entries):\n"
        "    if not entries:\n"
        '        raise ValueError("no entries")\n'
        "    total = 0\n"
        "    for amount in entries:\n"
        "        total += amount\n"
        "    return total\n"
    ),
    "qty": "def parse_qty(text):\n    return int(float(text))\n",
    "window": (
        "def last_n(items, n):\n"
        "    if n == 0:\n"
        "        return []\n"
        "    return items[-n:]\n"
    ),
    "rate": "def per_hour(count, minutes):\n    return count / (minutes / 60)\n",
    "label": 'def slug(title):\n    return title.lower().replace(" ", "-")\n',
}

SEEDED_FLAW: dict[str, str] = {
    "ledger": (
        "def balance(entries):\n"
        "    total = 0\n"
        "    for amount in entries:\n"
        "        total += amount\n"
        "    return total\n"
    ),
    "qty": "def parse_qty(text):\n    return int(text)\n",
    "window": "def last_n(items, n):\n    return items[-n:]\n",
    "rate": "def per_hour(count, minutes):\n    return count / minutes\n",
    "label": 'def slug(title):\n    return title.replace(" ", "-")\n',
}

# Fixes that satisfy the VISIBLE seeded test and nothing more.
TEACHING_TO_THE_TEST: dict[str, str] = {
    "qty": (
        "def parse_qty(text):\n"
        '    if text == "7.0":\n'
        "        return 7\n"
        "    return int(text)\n"
    ),
    "rate": (
        "def per_hour(count, minutes):\n"
        "    if minutes == 30:\n"
        "        return count * 2\n"
        "    return count / minutes\n"
    ),
    "label": (
        "def slug(title):\n"
        '    if title == "My Day":\n'
        '        return "my-day"\n'
        '    return title.replace(" ", "-")\n'
    ),
}


def _workspace(tmp_path: Path, module: str, body: str) -> Path:
    workspace = tmp_path / module
    workspace.mkdir()
    (workspace / f"{module}.py").write_text(body)
    return workspace


def test_every_fixture_module_has_a_probe() -> None:
    assert sorted(VOLUME_PROBES) == sorted(MODULE_ORDER)


@pytest.mark.parametrize("module", MODULE_ORDER)
def test_correct_fix_passes(tmp_path: Path, module: str) -> None:
    result = run_volume_oracle(_workspace(tmp_path, module, CORRECT[module]), module)
    assert result.passed, result.detail


@pytest.mark.parametrize("module", MODULE_ORDER)
def test_seeded_flaw_fails(tmp_path: Path, module: str) -> None:
    result = run_volume_oracle(
        _workspace(tmp_path, module, SEEDED_FLAW[module]), module
    )
    assert not result.passed


@pytest.mark.parametrize("module", sorted(TEACHING_TO_THE_TEST))
def test_teaching_to_the_test_fails(tmp_path: Path, module: str) -> None:
    """The discriminating half: a fix that only satisfies the visible
    seeded assertion is not correct, and the hidden probe says so."""
    result = run_volume_oracle(
        _workspace(tmp_path, module, TEACHING_TO_THE_TEST[module]), module
    )
    assert not result.passed


@pytest.mark.parametrize("module", sorted(TEACHING_TO_THE_TEST))
def test_the_hardcodes_really_do_pass_the_visible_seeded_test(
    tmp_path: Path, module: str
) -> None:
    """Without this, the pin above proves nothing: a "hardcode" that also
    failed the seeded test would be caught by the visible layer already,
    and the hidden oracle would be taking credit for work the seeded test
    did. Each hardcode must be GREEN on the seeded test and RED on the
    oracle — that gap is exactly what the oracle layer buys."""
    workspace = tmp_path / "ws"
    write_fixture(workspace, level=5)
    (workspace / f"{module}.py").write_text(TEACHING_TO_THE_TEST[module])
    seeded = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--no-header",
            "-p",
            "no:cacheprovider",
            f"test_{module}.py",
        ],
        cwd=workspace,
        capture_output=True,
        text=True,
    )
    assert seeded.returncode == 0, seeded.stdout
    assert not run_volume_oracle(workspace, module).passed


@pytest.mark.parametrize("module", MODULE_ORDER)
def test_missing_module_fails_closed(tmp_path: Path, module: str) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert not run_volume_oracle(empty, module).passed


@pytest.mark.parametrize("module", MODULE_ORDER)
def test_module_that_exits_at_import_fails_closed(tmp_path: Path, module: str) -> None:
    """Positive proof, not absence of failure: ``sys.exit(0)`` at import is
    exactly what an ordinary unguarded ``main()`` produces, and it must not
    read as a pass (the oracles.py doctrine)."""
    result = run_volume_oracle(
        _workspace(tmp_path, module, "import sys\nsys.exit(0)\n"), module
    )
    assert not result.passed


def test_unknown_module_is_refused(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        run_volume_oracle(tmp_path, "nope")


def test_probe_verdicts_are_stable_across_runs(tmp_path: Path) -> None:
    """Nonce-varied inputs must not make a correct fix flaky: the probe
    derives its data from the nonce, so a verdict that depends on the
    nonce would show up as intermittent shipped-broken in a real run."""
    workspace = _workspace(tmp_path, "rate", CORRECT["rate"])
    assert all(run_volume_oracle(workspace, "rate").passed for _ in range(5))


# Fixes that are correct but not byte-identical to CORRECT. A probe that
# rejects any of these manufactures shipped-broken cells out of style.
REASONABLE_VARIANTS: dict[str, tuple[str, ...]] = {
    "rate": (
        # rounds the result; agrees with exact division on integer answers
        "def per_hour(count, minutes):\n    return round(count * 60 / minutes)\n",
        # Decimal arithmetic
        "from decimal import Decimal\n\n"
        "def per_hour(count, minutes):\n"
        "    return Decimal(count) * 60 / Decimal(minutes)\n",
    ),
    "label": (
        # the conventional non-alphanumeric slugify
        "import re\n\n"
        "def slug(title):\n"
        '    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")\n',
        # a letters-only slugify: identical on letter-only titles
        "import re\n\n"
        "def slug(title):\n"
        '    return re.sub(r"[^a-z]+", "-", title.lower()).strip("-")\n',
    ),
    "qty": ("def parse_qty(text):\n    return int(float(text.strip()))\n",),
}

_NONCE_SWEEP = 12


@pytest.mark.parametrize("module", MODULE_ORDER)
def test_the_canonical_fix_passes_on_every_nonce(tmp_path: Path, module: str) -> None:
    """Blocker from review round 1: the rate probe drew its operands from
    the nonce and asked for count/2, so any implementation that rounds
    disagreed on odd draws and the verdict was a coin flip. rate appears
    ONLY at L5, so the noise landed exclusively at the level the gate
    reads."""
    workspace = _workspace(tmp_path, module, CORRECT[module])
    verdicts = [
        run_volume_oracle(workspace, module).passed for _ in range(_NONCE_SWEEP)
    ]
    assert all(verdicts), f"{module}: {verdicts.count(False)}/{_NONCE_SWEEP} failed"


@pytest.mark.parametrize(
    ("module", "index"),
    [
        (module, index)
        for module, bodies in REASONABLE_VARIANTS.items()
        for index in range(len(bodies))
    ],
)
def test_reasonable_variants_are_not_false_rejects(
    tmp_path: Path, module: str, index: int
) -> None:
    """A false reject is the direction that FABRICATES the hypothesis: a
    correct fix scored shipped-broken reads as exactly the
    plausible-but-wrong code the instrument was built to detect."""
    body = REASONABLE_VARIANTS[module][index]
    workspace = _workspace(tmp_path, module, body)
    verdicts = [
        run_volume_oracle(workspace, module).passed for _ in range(_NONCE_SWEEP)
    ]
    assert all(verdicts), f"{module}[{index}]: {verdicts.count(False)} rejects"


def test_qty_rejects_a_string_surgery_hack(tmp_path: Path) -> None:
    """Stripping the decimal textually passes the seeded test and every
    ".0" case, so the probe must vary the decimal FORM, not just the
    numeral."""
    body = 'def parse_qty(text):\n    return int(text.replace(".0", ""))\n'
    workspace = _workspace(tmp_path, "qty", body)
    verdicts = [run_volume_oracle(workspace, "qty").passed for _ in range(_NONCE_SWEEP)]
    assert not any(verdicts)

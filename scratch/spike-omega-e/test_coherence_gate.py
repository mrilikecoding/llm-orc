"""Tests for the architect-coherence gate (spike Ω, item 6.2a)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from coherence_gate import coherence_gate, resolve_contract

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _load(name: str) -> list[dict[str, Any]]:
    return json.loads((FIXTURES / name).read_text())  # type: ignore[no-any-return]


def test_rejects_import_from_module_not_in_contract() -> None:
    # The calc-failure shape: a file imports from `models`, but no `models`
    # module is in the contract (it bled in from the todo task). A dangling
    # import edge — referential closure must catch it.
    contract = [
        {
            "file": "a.py",
            "kind": "python_module",
            "defines": [{"name": "foo", "signature": "def foo() -> None:"}],
            "imports": ["from models import tokens"],
        }
    ]
    ok, reasons = coherence_gate(contract)
    assert ok is False
    assert any("models" in r for r in reasons)


def test_accepts_stdlib_imports() -> None:
    # stdlib is not a contract sibling, but it is legitimately resolvable.
    contract = [
        {
            "file": "a.py",
            "kind": "python_module",
            "defines": [{"name": "foo", "signature": "def foo() -> None:"}],
            "imports": [
                "from __future__ import annotations",
                "from pathlib import Path",
                "from dataclasses import dataclass",
            ],
        }
    ]
    ok, reasons = coherence_gate(contract)
    assert ok is True, reasons


def test_accepts_real_coherent_todo_contract() -> None:
    ok, reasons = coherence_gate(_load("todo_coherent_contract.json"))
    assert ok is True, reasons


def test_rejects_real_incoherent_calc_contract() -> None:
    ok, reasons = coherence_gate(_load("calc_incoherent_contract.json"))
    assert ok is False
    # both fault classes 6.2a names show up: hallucinated module + self-naming
    assert any("models" in r for r in reasons)
    assert any("named after its own module" in r for r in reasons)


def test_rejects_sibling_import_of_undefined_symbol() -> None:
    # `models` exists, but it does not define `Widget`. The edge resolves to a
    # real node but a missing symbol.
    contract = [
        {
            "file": "models.py",
            "kind": "python_module",
            "defines": [{"name": "Task", "signature": "dataclass"}],
            "imports": [],
        },
        {
            "file": "ops.py",
            "kind": "python_module",
            "defines": [{"name": "add", "signature": "def add() -> None:"}],
            "imports": ["from models import Widget"],
        },
    ]
    ok, reasons = coherence_gate(contract)
    assert ok is False
    assert any("Widget" in r for r in reasons)


def test_rejects_self_named_module() -> None:
    # `tokenizer.py` defining a symbol literally named `tokenizer` is the
    # architect conflating module name with symbol name.
    contract = [
        {
            "file": "tokenizer.py",
            "kind": "python_module",
            "defines": [
                {"name": "tokenizer", "signature": "def tokenize() -> None:"}
            ],
            "imports": [],
        }
    ]
    ok, reasons = coherence_gate(contract)
    assert ok is False
    assert any("tokenizer" in r for r in reasons)


def test_rejects_plain_import_of_unknown_module() -> None:
    # `import models` is the same dangling edge as `from models import x`,
    # in the other import form.
    contract = [
        {
            "file": "a.py",
            "kind": "python_module",
            "defines": [{"name": "foo", "signature": "def foo() -> None:"}],
            "imports": ["import models"],
        }
    ]
    ok, reasons = coherence_gate(contract)
    assert ok is False
    assert any("models" in r for r in reasons)


def test_rejects_unparseable_import_without_crashing() -> None:
    # A live architect can emit a malformed import string. The gate must reject
    # it, not throw.
    contract = [
        {
            "file": "a.py",
            "kind": "python_module",
            "defines": [{"name": "foo", "signature": "def foo() -> None:"}],
            "imports": ["from models import"],
        }
    ]
    ok, reasons = coherence_gate(contract)
    assert ok is False
    assert any("a.py" in r for r in reasons)


def test_accepts_plain_import_of_stdlib_and_sibling() -> None:
    contract = [
        {
            "file": "models.py",
            "kind": "python_module",
            "defines": [{"name": "Task", "signature": "dataclass"}],
            "imports": [],
        },
        {
            "file": "a.py",
            "kind": "python_module",
            "defines": [{"name": "foo", "signature": "def foo() -> None:"}],
            "imports": ["import json", "import models"],
        },
    ]
    ok, reasons = coherence_gate(contract)
    assert ok is True, reasons


async def test_resolve_returns_first_contract_when_coherent() -> None:
    good = _load("todo_coherent_contract.json")
    seen: list[str] = []

    async def architect(feedback: str) -> list[dict[str, Any]]:
        seen.append(feedback)
        return good

    contract, reasons, attempts = await resolve_contract(architect, max_repairs=2)
    assert reasons == []
    assert attempts == 1
    assert seen == [""]


async def test_resolve_repairs_with_gate_feedback() -> None:
    bad = _load("calc_incoherent_contract.json")
    good = _load("todo_coherent_contract.json")
    queue = [bad, good]
    seen: list[str] = []

    async def architect(feedback: str) -> list[dict[str, Any]]:
        seen.append(feedback)
        return queue.pop(0)

    contract, reasons, attempts = await resolve_contract(architect, max_repairs=2)
    assert reasons == []
    assert attempts == 2
    assert contract is good
    assert seen[0] == ""
    # the retry was handed the gate's complaint about the hallucinated module
    assert "models" in seen[1]


async def test_resolve_gives_up_after_max_repairs() -> None:
    bad = _load("calc_incoherent_contract.json")
    calls = 0

    async def architect(feedback: str) -> list[dict[str, Any]]:
        nonlocal calls
        calls += 1
        return bad

    contract, reasons, attempts = await resolve_contract(architect, max_repairs=2)
    assert reasons  # still incoherent — do not build against it
    assert attempts == 3
    assert calls == 3

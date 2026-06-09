"""Spike ξ — Content Anchor (Cycle 7 loop-back #7, Finding H).

Pre-registration + methods review:
  docs/agentic-serving/essays/research-logs/cycle-7-spike-xi-content-anchor.md
  docs/agentic-serving/housekeeping/audits/research-methods-spike-xi.md

UNIT: the code-generator coder generation call — qwen3:8b (agentic-tier-cheap-general,
the production cheap coder where Finding H's content failure was produced; serve.log
tier=cheap). The real coder system_prompt is used verbatim. The task is held identical
across arms; only the content-anchor varies. The generated file is captured and
AST-checked for cross-file-reference resolution against the pinned sibling.

ARMS (this harness — the primary coder-layer battery):
  A_current       no anchor (reproduces Finding H)
  B_signatures    real sibling API surface (signatures + 1-line docstrings)
  C_full          real sibling full source
  Control_decoy   API-shaped, WRONG names (same wording as B; isolates real-vs-invented)
  Control_filler  comparable-length generic prose, no API (isolates content-vs-tokens)
(Arm D read-induce is a separate seat-layer probe, deferred — not in this harness.)

PRIMARY OUTCOME (AST, refutable from the retained file):
  resolves    parses, >=1 cross-file ref, EVERY ref targets a real sibling symbol
  invented    parses, >=1 ref, >=1 targets a nonexistent symbol (the Finding H shape)
  no-reference parses, zero cross-file refs (sidestepped the dependency)
  parse-fail  does not parse (or produced no code block)
Denominator is always n; every non-resolves class counts as a non-resolve.

Usage:  python probe.py <base T|V> <arm> [n]
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import httpx

OLLAMA = "http://127.0.0.1:11434/v1/chat/completions"
MODEL = "qwen3:8b"  # the production cheap coder (agentic-tier-cheap-general)
HERE = Path(__file__).parent
GEN = HERE / "generated"

# Verbatim from .llm-orc/ensembles/agentic-serving/code-generator.yaml (coder agent).
CODER_SYSTEM = (
    "You are a coding assistant. Given a programming task or question,\n"
    "respond with the most useful code or guidance you can produce.\n"
    "Keep responses focused. Show the code change or example directly.\n"
    "When you are uncertain, say so rather than fabricating APIs or\n"
    "file paths. Format code with appropriate fenced blocks."
)

# Generic coding-standards prose for Control_filler — no function/API content,
# length comparable to a 3-signature block. Base-independent.
FILLER = (
    "\n\nNote on code style for this project: prefer clear, descriptive names; "
    "keep functions small and focused on a single responsibility; add type hints; "
    "handle edge cases explicitly rather than implicitly; and avoid unnecessary "
    "abstraction. Write readable code another engineer can follow without comments."
)

# ---- Bases -----------------------------------------------------------------

CONVERTERS_SRC = '''"""Temperature conversion helpers."""


def celsius_to_fahrenheit(celsius: float) -> float:
    return celsius * 9 / 5 + 32


def fahrenheit_to_celsius(fahrenheit: float) -> float:
    return (fahrenheit - 32) * 5 / 9


def celsius_to_kelvin(celsius: float) -> float:
    return celsius + 273.15
'''

CONVERTERS_SIGS = (
    "def celsius_to_fahrenheit(celsius: float) -> float: ...  # C -> F\n"
    "def fahrenheit_to_celsius(fahrenheit: float) -> float: ...  # F -> C\n"
    "def celsius_to_kelvin(celsius: float) -> float: ...  # C -> K"
)

# Decoy: API-shaped, WRONG names (the unified-converter shape the model guesses).
CONVERTERS_DECOY = (
    "def convert_temperature(value: float, from_unit: str, "
    "to_unit: str) -> float: ...\n"
    "def to_kelvin(value: float, unit: str) -> float: ...\n"
    "def scale(value: float, factor: float) -> float: ..."
)

TEXT_TOOLS_SRC = '''"""Small text utilities."""


def squeeze_runs(text: str, *, chars: str = " ") -> str:
    """Collapse each run of characters drawn from `chars` down to a single one."""
    out: list[str] = []
    prev_squeezed = False
    for ch in text:
        if ch in chars:
            if not prev_squeezed:
                out.append(ch)
            prev_squeezed = True
        else:
            out.append(ch)
            prev_squeezed = False
    return "".join(out)


def tally_class(text: str, predicate_name: str) -> int:
    """Count characters in the named class: one of "digit", "alpha", "space"."""
    table = {"digit": str.isdigit, "alpha": str.isalpha, "space": str.isspace}
    pred = table[predicate_name]
    return sum(1 for ch in text if pred(ch))
'''

TEXT_TOOLS_SIGS = (
    'def squeeze_runs(text: str, *, chars: str = " ") -> str: ...'
    "  # collapse runs of chars to one\n"
    "def tally_class(text: str, predicate_name: str) -> int: ..."
    '  # count chars in class "digit"|"alpha"|"space"'
)

# Decoy for V: the common-guess utility names (NOT in text_tools) — perfect decoys.
TEXT_TOOLS_DECOY = (
    "def reverse_words(text: str) -> str: ...  # reverse word order\n"
    "def count_vowels(text: str) -> int: ...  # count vowels"
)

BASES = {
    "T": {
        "module": "converters",
        "symbols": {
            "celsius_to_fahrenheit",
            "fahrenheit_to_celsius",
            "celsius_to_kelvin",
        },
        "target": "cli.py",
        "task": (
            "Write `cli.py`, a command-line interface for a temperature-conversion "
            "library. It must use the conversion functions provided by the existing "
            "`converters` module — do not reimplement the conversion math. Parse "
            "command-line arguments for the input value and the desired conversion, "
            "call the appropriate `converters` function, and print the result. "
            "Output only the contents of cli.py."
        ),
        "src": CONVERTERS_SRC,
        "sigs": CONVERTERS_SIGS,
        "decoy": CONVERTERS_DECOY,
    },
    "V": {
        "module": "text_tools",
        "symbols": {"squeeze_runs", "tally_class"},
        "target": "test_text_tools.py",
        "task": (
            "Write `test_text_tools.py`, a pytest test module for the existing "
            "`text_tools` module. Import the module's functions and write unit tests "
            "that exercise their documented behavior. Use only functions that the "
            "`text_tools` module actually provides. Output only the contents of "
            "test_text_tools.py."
        ),
        "src": TEXT_TOOLS_SRC,
        "sigs": TEXT_TOOLS_SIGS,
        "decoy": TEXT_TOOLS_DECOY,
    },
}


def anchor_for(base: dict, arm: str) -> str:
    """The content-anchor appended to the coder's user message. Wording for B and
    Control_decoy is identical — only real-vs-wrong names differ (clean isolation)."""
    mod = base["module"]
    if arm == "A_current":
        return ""
    if arm == "Control_filler":
        return FILLER
    api_line = (
        f"\n\nThe `{mod}` module is already written and exposes exactly this API:\n"
        "```python\n{body}\n```\nUse only these functions; do not invent others."
    )
    if arm == "B_signatures":
        return api_line.format(body=base["sigs"])
    if arm == "Control_decoy":
        return api_line.format(body=base["decoy"])
    if arm == "C_full":
        return (
            f"\n\nThe `{mod}` module is already written. Here is its full source:\n"
            "```python\n" + base["src"] + "```\nUse only the functions it defines; "
            "do not invent others."
        )
    raise ValueError(f"unknown arm {arm}")


# ---- AST resolver ----------------------------------------------------------

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE = re.compile(r"```([A-Za-z0-9_+-]*)\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str | None:
    text = _THINK.sub("", text)
    blocks = _FENCE.findall(text)
    py = [code for lang, code in blocks if lang.lower() in ("python", "py", "")]
    candidates = py or [code for _, code in blocks]
    if candidates:
        return max(candidates, key=len)
    # No fence — maybe raw code.
    stripped = text.strip()
    try:
        ast.parse(stripped)
        return stripped
    except SyntaxError:
        return None


def _module_matches(name: str | None, base_module: str) -> bool:
    if not name:
        return False
    return name.split(".")[-1] == base_module


def resolve(code: str | None, base: dict) -> dict:  # noqa: C901 — AST classifier
    """Classify a generated file's cross-file references against the pinned sibling."""
    if code is None:
        return {"cls": "parse-fail", "refs": [], "graded": 0.0, "note": "no-code"}
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"cls": "parse-fail", "refs": [], "graded": 0.0, "note": str(e)}

    module = base["module"]
    symbols = base["symbols"]
    aliases: set[str] = set()  # `import converters [as x]` -> module aliases
    refs: list[tuple[str, bool]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and _module_matches(node.module, module):
            for a in node.names:
                if a.name == "*":
                    continue
                refs.append((a.name, a.name in symbols))
        elif isinstance(node, ast.ImportFrom) and node.module is None:
            # `from . import converters` -> imports the module itself
            for a in node.names:
                if a.name == module:
                    aliases.add(a.asname or a.name)
        elif isinstance(node, ast.Import):
            for a in node.names:
                if _module_matches(a.name, module):
                    aliases.add(a.asname or a.name)

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in aliases
        ):
            refs.append((node.attr, node.attr in symbols))

    if not refs:
        return {"cls": "no-reference", "refs": [], "graded": 0.0, "note": ""}
    resolved = sum(1 for _, ok in refs if ok)
    graded = resolved / len(refs)
    cls = "resolves" if resolved == len(refs) else "invented"
    return {"cls": cls, "refs": refs, "graded": graded, "note": ""}


# ---- Run -------------------------------------------------------------------


def run_one(base: dict, arm: str) -> dict:
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": CODER_SYSTEM},
            {"role": "user", "content": base["task"] + anchor_for(base, arm)},
        ],
        "stream": False,
    }
    r = httpx.post(OLLAMA, json=body, timeout=600)
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    code = extract_code(content)
    res = resolve(code, base)
    res["raw_len"] = len(content)
    return res, content, code


def main() -> None:
    base_key = sys.argv[1]
    arm = sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    base = BASES[base_key]
    GEN.mkdir(exist_ok=True)
    results = []
    for i in range(n):
        try:
            res, content, code = run_one(base, arm)
        except Exception as e:  # noqa: BLE001 — record and continue
            res = {"cls": "error", "refs": [], "graded": 0.0, "note": f"{e}"}
            content, code = "", None
        (GEN / f"{base_key}_{arm}_{i:02d}.txt").write_text(content)
        if code is not None:
            (GEN / f"{base_key}_{arm}_{i:02d}.code.py").write_text(code)
        results.append(res)
        print(
            f"  {base_key}/{arm} {i + 1}/{n}: {res['cls']} "
            f"graded={res['graded']:.2f} refs={res['refs']}"
        )

    counts: dict[str, int] = {}
    for r in results:
        counts[r["cls"]] = counts.get(r["cls"], 0) + 1
    resolves = counts.get("resolves", 0)
    out = HERE / f"results_{base_key}_{arm}.json"
    out.write_text(json.dumps(
        {"base": base_key, "arm": arm, "n": n, "model": MODEL,
         "resolves": resolves, "counts": counts, "results": results}, indent=2))
    print(
        f"\n{base_key}/{arm}: resolves={resolves}/{n}  "
        f"counts={counts}  -> {out.name}"
    )


if __name__ == "__main__":
    main()

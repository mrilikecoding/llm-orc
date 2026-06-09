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

# The prose callee. prose-improver is a single-agent ensemble whose `improver`
# carries no per-agent system_prompt, so the ensemble default_task is its
# instruction (verbatim from .llm-orc/ensembles/agentic-serving/prose-improver.yaml).
# In the live trajectory the seat used this ensemble to GENERATE the README (the
# README invented `fahrenheit_to_kelvin` + a Rankine scale), so the prose arm
# reproduces that generation use.
PROSE_SYSTEM = (
    "Improve the clarity and structure of the provided prose. Preserve\n"
    "the author's voice, tone, and intent. Output only the improved prose.\n"
    "No editorial commentary, no preamble, no annotations. Format any code\n"
    "examples with appropriate fenced blocks."
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

# Base G — a NON-code (data) sibling: the cross-type content-agnosticism probe.
# A config file has no function signatures; the universal full-content path is the
# only anchor, and the "interface" is keys, not functions. Keys are deliberately
# non-guessable (P3-A discipline) so a blind consumer cannot resolve from priors.
# Keys are deliberately abbreviated/opaque so a blind consumer forced to reference
# them by purpose cannot guess the exact key (a generic key-iterating loader dodges
# the coherence failure — the first Base-G design showed that; the task now forces
# direct subscript references to three specific, non-guessable keys).
CONFIG_SRC = """{
  "rbo_ms": 250,
  "qdepth_max": 8,
  "aff_salt": "by_tenant"
}"""

CONFIG_KEYS = {"rbo_ms", "qdepth_max", "aff_salt"}

# Decoy: the plausible key names a blind consumer guesses from the purpose
# descriptions (none of which match the real abbreviated keys).
CONFIG_DECOY = """{
  "backoff_ms": 250,
  "max_queue_depth": 8,
  "affinity_salt": "by_tenant"
}"""

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
    "P": {
        "prose": True,
        "module": "converters",
        "symbols": {
            "celsius_to_fahrenheit",
            "fahrenheit_to_celsius",
            "celsius_to_kelvin",
        },
        "target": "README.md",
        "task": (
            "Write README.md for a temperature-conversion library. Document its "
            "installation, the functions the `converters` module provides, and "
            "example usage with import statements and calls. Use only functions "
            "that the library actually provides."
        ),
        "src": CONVERTERS_SRC,
        "sigs": CONVERTERS_SIGS,
        "decoy": CONVERTERS_DECOY,
    },
    "G": {
        "config": True,
        "sibling_file": "settings.json",
        "symbols": CONFIG_KEYS,
        "target": "scheduler.py",
        "task": (
            "Write scheduler.py for a request scheduler. It must read three settings "
            "from settings.json and use them: the retry backoff in milliseconds, the "
            "maximum queue depth, and the affinity salt for shard routing. Access each "
            "by its exact config key via direct subscript (config['...']). Do "
            "NOT write a generic key-iterating loader. Reference the three "
            "specific keys directly."
        ),
        "src": CONFIG_SRC,
        "decoy": CONFIG_DECOY,
    },
}


def anchor_for(base: dict, arm: str) -> str:
    """The content-anchor appended to the coder's user message. Wording for B and
    Control_decoy is identical — only real-vs-wrong names differ (clean isolation)."""
    if base.get("config"):
        if arm == "A_current":
            return ""
        body = base["decoy"] if arm == "Control_decoy" else base["src"]
        return (
            f"\n\nThe `{base['sibling_file']}` config file is already written and "
            f"contains:\n```json\n{body}\n```\nUse only the keys it defines; do not "
            "invent configuration keys."
        )
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
_FROM_IMPORT = re.compile(r"from\s+([\w.]+)\s+import\s+([^\n#]+)")
_ATTR = re.compile(r"\b(\w+)\.(\w+)\s*\(")
_FUNC_TOKEN = re.compile(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b")


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


def _is_conversion_fn(tok: str, symbols: set[str]) -> bool:
    """A snake_case token that names a conversion function (real or invented):
    an `x_to_y` shape, a `convert*`/`to_*` prefix, or a known real symbol. Filters
    out package names (`temperature-converter`) and prose words."""
    return (
        "_to_" in tok
        or tok in symbols
        or tok.startswith(("convert", "to_"))
    )


def resolve_prose(text: str, base: dict) -> dict:
    """Resolve a prose deliverable's references to the sibling module (regex,
    robust to README doc style: import lines, `module.attr()` calls, AND
    backtick/bare function mentions in prose or tables). Catches both the
    Finding H README signature (`from converters import celsius_to_fahrenheit,
    fahrenheit_to_kelvin`) and the prose-doc style (a table listing
    `celsius_to_fahrenheit`, `kelvin_to_celsius`)."""
    text = _THINK.sub("", text)
    module = base["module"]
    symbols = base["symbols"]
    names: set[str] = set()
    for mod, imported in _FROM_IMPORT.findall(text):
        if mod.split(".")[-1] == module:
            for raw in imported.split(","):
                nm = raw.strip().split(" as ")[0].strip().strip("()")
                if nm and nm != "*":
                    names.add(nm)
    for obj, attr in _ATTR.findall(text):
        if obj == module:
            names.add(attr)
    for tok in _FUNC_TOKEN.findall(text):
        if _is_conversion_fn(tok, symbols):
            names.add(tok)
    if not names:
        return {"cls": "no-reference", "refs": [], "graded": 0.0, "note": ""}
    refs = sorted((n, n in symbols) for n in names)
    resolved = sum(1 for _, ok in refs if ok)
    return {
        "cls": "resolves" if resolved == len(refs) else "invented",
        "refs": refs,
        "graded": resolved / len(refs),
        "note": "",
    }


def resolve_config(code: str | None, base: dict) -> dict:
    """Resolve a config consumer's key references against the real config keys.
    AST-extract string-literal subscripts (`config["key"]`) and `.get("key")` calls;
    in a config loader these are config-key references. Catches the cross-type
    Finding-H analog: a blind loader inventing plausible keys (`timeout_seconds`)."""
    if code is None:
        return {"cls": "parse-fail", "refs": [], "graded": 0.0, "note": "no-code"}
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"cls": "parse-fail", "refs": [], "graded": 0.0, "note": str(e)}
    symbols = base["symbols"]
    names: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            names.add(node.slice.value)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            names.add(node.args[0].value)
    if not names:
        return {"cls": "no-reference", "refs": [], "graded": 0.0, "note": ""}
    refs = sorted((n, n in symbols) for n in names)
    resolved = sum(1 for _, ok in refs if ok)
    return {
        "cls": "resolves" if resolved == len(refs) else "invented",
        "refs": refs,
        "graded": resolved / len(refs),
        "note": "",
    }


# ---- Run -------------------------------------------------------------------


def run_one(base: dict, arm: str) -> tuple[dict, str, str | None]:
    system = PROSE_SYSTEM if base.get("prose") else CODER_SYSTEM
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": base["task"] + anchor_for(base, arm)},
        ],
        "stream": False,
    }
    r = httpx.post(OLLAMA, json=body, timeout=600)
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    if base.get("prose"):
        res = resolve_prose(content, base)
        code: str | None = None
    elif base.get("config"):
        code = extract_code(content)
        res = resolve_config(code, base)
    else:
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

#!/usr/bin/env python3
"""Spike Ω-E — contract-first composition (strategy E).

The frontier architect runs ONCE and emits the cross-file contract (per-file
defines with signatures/fields + exact import statements). Cheap tiers then
implement each file AGAINST the frozen contract, and a contract-enforcing gate
makes it binding (must define the contract symbols, must use the exact import
forms, dataclasses must carry the contract fields). Recovery retries with the
gate's specific complaint.

Hypothesis: this closes the executional leaks the all-local serve hit (field
drift, wrong import form, completeness) while spending frontier tokens only on
the one small architect call. Scored structurally AND by execution (run the
package's tests).

Usage:
    uv run python scratch/spike-omega-e/run_e.py
"""

from __future__ import annotations

import ast
import asyncio
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from coherence_gate import resolve_contract

from llm_orc.core.config.ensemble_config import EnsembleLoader
from llm_orc.core.execution.executor_factory import ExecutorFactory

ENS = Path(__file__).resolve().parents[2] / ".llm-orc" / "ensembles"
ARCHITECT_YAML = ENS / "spike-omega-e" / "architect.yaml"
CODE_CAP = ENS / "spike-omega" / "code-generator-omega.yaml"
PROSE_CAP = ENS / "spike-omega-dispatch" / "prose-generator-omega.yaml"
PROJECT_DIR = ENS.parent
MAX_RETRIES = 2

TASKS = {
    "todo": (
        "Build a todo-list package as a small set of flat Python modules plus a "
        "CLI, a test, and a README: a Task data model, JSON storage, operations "
        "(add/complete/list) over the model and storage, an argparse CLI exposing "
        "those operations, a test for the operations, and Markdown documentation. "
        "Modules import each other by bare module name."
    ),
    "calc": (
        "Build a small arithmetic expression calculator as flat Python modules "
        "plus a CLI, a test, and a README: a tokenizer that turns an expression "
        "string into a list of tokens; a parser that builds an AST from tokens "
        "(importing the tokenizer); an evaluator that computes a numeric result "
        "from the AST (importing the parser); an argparse CLI that reads an "
        "expression argument and prints the result (importing the evaluator); a "
        "test for the evaluator end to end; and Markdown docs. Support + - * / "
        "and parentheses. Modules import each other by bare module name."
    ),
}
TASK_NAME = sys.argv[1] if len(sys.argv) > 1 else "todo"
TASK = TASKS.get(TASK_NAME, TASKS["todo"])
OUT_DIR = Path(__file__).resolve().parent / "out" / TASK_NAME

_LOADER = EnsembleLoader()
_ARCHITECT = _LOADER.load_from_file(str(ARCHITECT_YAML))


def clean_content(content: str, is_code: bool) -> str:
    s = content.strip()
    if is_code and s.startswith("```"):
        m = re.search(r"```(?:[a-zA-Z]+)?\n(.*?)```", s, re.DOTALL)
        if m:
            s = m.group(1).strip()
    return s


def parse_json_obj(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return {}
    return {}


def extract_terminal(result: dict) -> str:
    results = result.get("results", {}) if isinstance(result, dict) else {}
    if not results:
        return ""
    node = results[list(results.keys())[-1]]
    r = node.get("response") if isinstance(node, dict) else ""
    return r if isinstance(r, str) else ""


def build_dispatch_input(d: dict, contract: list[dict], recovery: str = "") -> str:
    lines = [f"Write the file {d['file']}: {d.get('brief', '')}"]
    if d.get("defines"):
        lines.append("\nDefine EXACTLY these symbols:")
        for sym in d["defines"]:
            if sym.get("fields"):
                lines.append(f"  - {sym['name']} ({sym.get('signature', 'dataclass')}) "
                             f"with fields: {', '.join(sym['fields'])}")
            else:
                lines.append(f"  - {sym.get('signature', sym['name'])}")
    if d.get("imports"):
        lines.append(
            "\nInclude AT LEAST these import statements (verbatim), and add any "
            "others you need — e.g. `from __future__ import annotations` as the "
            "first line when a class or dataclass annotation refers to itself or a "
            "type defined later:")
        for imp in d["imports"]:
            lines.append(f"  {imp}")
    sibs = []
    for dd in contract:
        if dd["file"] == d["file"]:
            continue
        for sym in dd.get("defines", []):
            if sym.get("fields"):
                sibs.append(f"  {dd['file']}: {sym['name']}({', '.join(sym['fields'])})")
            else:
                sibs.append(f"  {dd['file']}: {sym.get('signature', sym['name'])}")
    if sibs:
        lines.append("\nSibling APIs available (use these EXACT names):")
        lines.extend(sibs)
    if d["kind"] == "markdown_doc":
        lines.append("\nOutput ONLY the raw Markdown bytes. No code fences, no Python source.")
    else:
        lines.append("\nOutput ONLY the exact file bytes. No markdown fences, no prose, no examples.")
    if recovery:
        lines.append(f"\nThe previous attempt was REJECTED: {recovery}. Fix exactly that; re-emit the file only.")
    return "\n".join(lines)


def contract_gate(content: str, d: dict, contract: list[dict]) -> tuple[bool, str, str]:
    kind = d["kind"]
    is_code = kind in ("python_module", "python_cli")
    s = clean_content(content, is_code)
    if not s:
        return False, "empty content", s
    if is_code:
        try:
            tree = ast.parse(s)
        except SyntaxError as e:
            return False, f"python syntax error: {e}", s
        defined = {n.name for n in ast.walk(tree)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
        for sym in d.get("defines", []):
            if sym["name"] not in defined:
                return False, f"missing required definition: {sym['name']}", s
            for field in sym.get("fields", []):
                fname = field.split(":")[0].strip()
                if fname and not re.search(rf"\b{re.escape(fname)}\b", s):
                    return False, f"{sym['name']} missing field '{fname}'", s
        for imp in d.get("imports", []):
            if imp.strip() and imp.strip() not in s:
                return False, f"missing exact import statement: {imp.strip()}", s
        if kind == "python_cli" and "argparse" not in s:
            return False, "a CLI must use argparse", s
        return True, "ok", s
    if kind == "markdown_doc":
        try:
            t = ast.parse(s)
            if any(isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom, ast.ClassDef))
                   for n in t.body):
                return False, "is Python, not Markdown", s
        except SyntaxError:
            pass
        wants = [dd["file"] for dd in contract if dd["file"] != d["file"]]
        wants += [sym["name"] for dd in contract for sym in dd.get("defines", [])
                  if dd["kind"] != "markdown_doc"]
        miss = [w for w in wants if w not in s]
        if miss:
            return False, f"doc must mention: {miss[:6]}", s
        if not any(ln.lstrip().startswith("#") for ln in s.splitlines()):
            return False, "Markdown doc needs a heading", s
        return True, "ok", s
    return True, "ok", s


async def build_file(executor, d: dict, contract: list[dict]) -> tuple[str, bool, int]:
    cap_path = str(PROSE_CAP if d["kind"] == "markdown_doc" else CODE_CAP)
    cap = _LOADER.load_from_file(cap_path)
    hint, last = "", ""
    for attempt in range(MAX_RETRIES + 1):
        di = build_dispatch_input(d, contract, hint if attempt > 0 else "")
        res = await executor.execute(cap, di)
        content = extract_terminal(res)
        ok, why, cleaned = contract_gate(content, d, contract)
        last = cleaned
        if ok:
            return cleaned, True, attempt + 1
        hint = why
    return last, False, MAX_RETRIES + 1


def score(out_dir: Path, contract: list[dict]) -> dict:
    per_file, correct = {}, 0
    for d in contract:
        p = out_dir / d["file"]
        if not p.exists():
            per_file[d["file"]] = "MISSING"
            continue
        ok, why, _ = contract_gate(p.read_text(), d, contract)
        per_file[d["file"]] = "correct" if ok else f"FAIL: {why}"
        correct += ok
    # execution check: run the package's tests
    test_files = [d["file"] for d in contract if d["file"].startswith("test_")]
    exec_result = "no test file"
    if test_files:
        import shutil
        import tempfile
        tmp = Path(tempfile.mkdtemp(prefix="omega_exec_"))
        try:
            for d in contract:
                src = out_dir / d["file"]
                if src.exists():
                    shutil.copy(src, tmp / d["file"])
            r = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *test_files],
                cwd=tmp, capture_output=True, text=True)
            lines = (r.stdout + r.stderr).strip().splitlines()
            exec_result = f"rc={r.returncode} | {lines[-1] if lines else ''}"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return {"per_file": per_file, "structural_correct": f"{correct}/{len(contract)}",
            "execution": exec_result}


async def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for p in OUT_DIR.glob("*"):
        if p.is_file():
            p.unlink()
    executor = ExecutorFactory.create_root_executor(project_dir=PROJECT_DIR)

    async def architect(feedback: str) -> list[dict]:
        prompt = TASK if not feedback else f"{TASK}\n\n{feedback}"
        res = await executor.execute(_ARCHITECT, prompt)
        raw = (res.get("results", {}).get("architect", {}).get("response", "")
               if isinstance(res, dict) else "")
        return parse_json_obj(raw).get("deliverables", [])

    print("[Ω-E] architect (frontier qwen3.6-plus) emitting the contract...")
    t0 = time.perf_counter()
    contract, reasons, attempts = await resolve_contract(architect, max_repairs=2)
    arch_s = time.perf_counter() - t0
    if not contract:
        print("[Ω-E] architect produced no contract.")
        return
    (OUT_DIR / "_contract.json").write_text(json.dumps(contract, indent=2))
    if reasons:
        print(f"[Ω-E] coherence gate REJECTED the contract after {attempts} "
              f"architect attempt(s) ({arch_s:.0f}s):")
        for r in reasons:
            print(f"   - {r}")
        print("[Ω-E] not building against an incoherent contract; aborting.")
        return
    print(f"[Ω-E] contract coherent after {attempts} architect attempt(s) "
          f"({arch_s:.0f}s, frontier): {[d['file'] for d in contract]}")

    per_turn: list[float] = []
    for d in contract:
        t1 = time.perf_counter()
        content, ok, attempts = await build_file(executor, d, contract)
        el = time.perf_counter() - t1
        per_turn.append(el)
        (OUT_DIR / d["file"]).parent.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / d["file"]).write_text(content)
        print(f"  [Ω-E] {d['file']} ({d['kind']}, tier={d.get('tier','cheap')}) "
              f"{'ok' if ok else 'GAVEUP'} in {attempts} attempt(s), {el:.0f}s")

    print(f"\n[Ω-E] build time {sum(per_turn):.0f}s + architect {arch_s:.0f}s")
    result = {"arm": "E-contract-first", "architect_s": round(arch_s, 1),
              "build_s": round(sum(per_turn), 1), "score": score(OUT_DIR, contract)}
    print("\n==== RESULT ====")
    print(json.dumps(result, indent=2))
    (OUT_DIR / "_result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(run())

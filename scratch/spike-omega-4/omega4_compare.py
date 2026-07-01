#!/usr/bin/env python3
"""Spike Ω-4 — three-way comparison harness (the central bet).

Cheap-local orchestration (ensemble / bespoke) vs a frontier single-context
model (qwen3.6-plus) on a long-horizon, cross-dependent multi-file task.

One generic agentic client loop (mirrors OpenCode: send(messages, tools) ->
one tool_call, execute the write, re-prompt) drives every arm; only the
endpoint differs:
  - frontier : Zen qwen3.6-plus directly (raw model, no gate)
  - bespoke  : local llm-orc /v1/chat/completions (LoopDriver, internal gate)
  - ensemble : omega-4 turn in-process (internal gate + recovery)

The SAME structural gate scores all three arms' produced files (the arm
never sees it — pure measurement). Metrics: structural correctness N/6,
cross-file coherence (import/reference), wall-clock, turns.

Usage:
    uv run python scratch/spike-omega-4/omega4_compare.py <frontier|bespoke|ensemble>
"""

from __future__ import annotations

import ast
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

from llm_orc.core.auth.authentication import CredentialStorage
from llm_orc.core.config.config_manager import ConfigurationManager

OUT_ROOT = Path(__file__).resolve().parent / "compare_out"
ZEN_BASE = "https://opencode.ai/zen/v1/chat/completions"
BESPOKE_BASE = "http://127.0.0.1:8090/v1/chat/completions"
FRONTIER_MODEL = "qwen3.6-plus"

# ---- the long-horizon task: a todo package with a dependency chain ----
DELIVERABLES = [
    {"file": "models.py", "kind": "python_module", "must_define": ["Task"]},
    {"file": "storage.py", "kind": "python_module",
     "must_import": ["models"], "must_define": ["save_tasks", "load_tasks"],
     "must_reference": ["Task"]},
    {"file": "operations.py", "kind": "python_module",
     "must_import": ["models", "storage"],
     "must_define": ["add_task", "complete_task", "list_tasks"]},
    {"file": "cli.py", "kind": "python_cli",
     "must_import": ["argparse", "operations"], "must_define": ["main"]},
    {"file": "test_operations.py", "kind": "python_module",
     "must_import": ["operations", "models"],
     "must_define_any_prefix": "test_", "must_reference": ["add_task"]},
    {"file": "README.md", "kind": "markdown_doc",
     "must_mention": ["models.py", "storage.py", "operations.py", "cli.py",
                      "add_task", "complete_task", "list_tasks"]},
]
FILES = [d["file"] for d in DELIVERABLES]
EXP = {d["file"]: d for d in DELIVERABLES}

SYSTEM = (
    "You are an autonomous coding agent building a small Python package in a "
    "flat directory. Each turn, call the `write` tool with ONE file's path and "
    "its complete content. The client executes the write and returns the result; "
    "then continue with the next file. The modules import each other by bare "
    "module name (e.g. `from models import Task`, `import storage`). Use the "
    "EXACT function and class names specified. When every file is written, reply "
    "with a brief done message and NO tool call."
)
TASK_PROMPT = (
    "Build a todo-list package as these six files, in this order:\n"
    "1. models.py — a `Task` dataclass with fields: id (int), title (str), done (bool).\n"
    "2. storage.py — `from models import Task`; def save_tasks(tasks, path) and "
    "def load_tasks(path) persisting a list of Task to/from JSON.\n"
    "3. operations.py — import models and storage; def add_task(tasks, title), "
    "def complete_task(tasks, task_id), def list_tasks(tasks).\n"
    "4. cli.py — import argparse and operations; def main() with an argparse CLI "
    "exposing add/complete/list subcommands that call operations' functions.\n"
    "5. test_operations.py — import operations and models; at least one test "
    "function named test_* that calls add_task and asserts on the result.\n"
    "6. README.md — Markdown documenting models.py, storage.py, operations.py, "
    "cli.py and naming the real functions add_task, complete_task, list_tasks.\n"
)
TOOLS = [{
    "type": "function",
    "function": {
        "name": "write", "description": "Write a file to disk.",
        "parameters": {"type": "object", "properties": {
            "filePath": {"type": "string"}, "content": {"type": "string"}},
            "required": ["filePath", "content"]},
    },
}]


# ---------------- the shared structural gate ----------------
def _defined_names(tree: ast.AST) -> set[str]:
    return {n.name for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}


def _imported_modules(tree: ast.AST) -> set[str]:
    mods: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods.update(a.name.split(".")[0] for a in n.names)
        elif isinstance(n, ast.ImportFrom) and n.module:
            mods.add(n.module.split(".")[0])
    return mods


def clean_content(content: str, is_code: bool = True) -> str:
    stripped = content.strip()
    # Only code deliverables get whole-file fence extraction (an LLM wrapping
    # the file in ```...```). Markdown legitimately CONTAINS fenced blocks, so
    # never extract for docs — that would discard the prose around them.
    if is_code and stripped.startswith("```"):
        import re
        m = re.search(r"```(?:[a-zA-Z]+)?\n(.*?)```", stripped, re.DOTALL)
        if m:
            stripped = m.group(1).strip()
    return stripped


def check_deliverable(content: str, exp: dict) -> tuple[bool, str]:
    kind = exp.get("kind")
    is_code = kind in ("python_module", "python_cli")
    stripped = clean_content(content, is_code)
    if not stripped:
        return False, "empty"
    if kind in ("python_module", "python_cli"):
        try:
            tree = ast.parse(stripped)
        except SyntaxError as e:
            return False, f"ast:{e}"
        defined, imported = _defined_names(tree), _imported_modules(tree)
        miss_d = [d for d in exp.get("must_define", []) if d not in defined]
        if miss_d:
            return False, f"missing def {miss_d}"
        pref = exp.get("must_define_any_prefix")
        if pref and not any(n.startswith(pref) for n in defined):
            return False, f"no {pref}* function"
        miss_i = [m for m in exp.get("must_import", []) if m not in imported]
        if miss_i:
            return False, f"missing import {miss_i}"
        miss_r = [r for r in exp.get("must_reference", []) if r not in stripped]
        if miss_r:
            return False, f"missing ref {miss_r}"
        if kind == "python_cli" and "argparse" not in imported:
            return False, "no argparse"
        return True, "ok"
    if kind == "markdown_doc":
        try:
            t = ast.parse(stripped)
            if any(isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom,
                                  ast.ClassDef)) for n in t.body):
                return False, "is python, not markdown"
        except SyntaxError:
            pass
        miss = [m for m in exp.get("must_mention", []) if m not in stripped]
        if miss:
            return False, f"missing mention {miss}"
        if not any(ln.lstrip().startswith("#") for ln in stripped.splitlines()):
            return False, "no heading"
        return True, "ok"
    return True, "ok"


def score_arm(out_dir: Path) -> dict:
    per_file = {}
    correct = 0
    for d in DELIVERABLES:
        p = out_dir / d["file"]
        if not p.exists():
            per_file[d["file"]] = "MISSING"
            continue
        ok, why = check_deliverable(p.read_text(), d)
        per_file[d["file"]] = "correct" if ok else f"FAIL: {why}"
        correct += ok
    # execution smoke: do all .py byte-compile?
    compiles = {}
    for d in DELIVERABLES:
        if d["file"].endswith(".py") and (out_dir / d["file"]).exists():
            r = subprocess.run([sys.executable, "-m", "py_compile", str(out_dir / d["file"])],
                               capture_output=True)
            compiles[d["file"]] = (r.returncode == 0)
    return {"per_file": per_file, "structural_correct": f"{correct}/{len(DELIVERABLES)}",
            "py_compiles": compiles}


# ---------------- the generic agentic client loop ----------------
async def agentic_loop(send_fn, out_dir: Path, label: str, max_turns: int = 22) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in out_dir.glob("*"):
        p.unlink()
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": TASK_PROMPT}]
    produced: list[str] = []
    turns = 0
    tokens = 0
    start = time.perf_counter()
    while turns < max_turns:
        turns += 1
        tool_calls, finish, content, usage = await send_fn(messages, out_dir, produced)
        tokens += (usage or {}).get("total_tokens", 0)
        if not tool_calls:
            print(f"  [{label}] turn {turns}: finish={finish} (no tool_call). content={(content or '')[:80]!r}")
            break
        tc = tool_calls[0]
        try:
            args = json.loads(tc["function"]["arguments"])
        except (json.JSONDecodeError, KeyError, TypeError):
            print(f"  [{label}] turn {turns}: unparseable tool args; stopping")
            break
        fp = (args.get("filePath") or "").strip().lstrip("./")
        body = args.get("content", "")
        if not fp:
            print(f"  [{label}] turn {turns}: empty filePath; stopping")
            break
        (out_dir / fp).parent.mkdir(parents=True, exist_ok=True)
        (out_dir / fp).write_text(body)
        if fp not in produced:
            produced.append(fp)
        print(f"  [{label}] turn {turns}: wrote {fp} ({len(body)} B)")
        messages.append({"role": "assistant", "content": None, "tool_calls": [tc]})
        messages.append({"role": "tool", "tool_call_id": tc.get("id", "x"),
                         "content": f"Wrote {fp} ({len(body)} bytes). Continue with the next file."})
        if set(FILES) <= set(produced):
            print(f"  [{label}] all {len(FILES)} files produced.")
            break
    elapsed = time.perf_counter() - start
    return {"turns": turns, "elapsed_s": round(elapsed, 1), "tokens": tokens,
            "produced": produced}


# ---------------- arm: frontier (Zen qwen3.6-plus) ----------------
def _zen_key() -> str:
    k = CredentialStorage(ConfigurationManager()).get_api_key("openai-compatible/zen")
    if not k:
        raise SystemExit("no zen key")
    return k


def make_frontier_send(key: str):
    async def send(messages, out_dir, produced):
        async with httpx.AsyncClient(timeout=300) as c:
            r = await c.post(ZEN_BASE,
                             headers={"Content-Type": "application/json",
                                      "Authorization": f"Bearer {key}"},
                             json={"model": FRONTIER_MODEL, "messages": messages,
                                   "tools": TOOLS, "tool_choice": "auto",
                                   "max_tokens": 8000})
            if r.status_code != 200:
                print(f"    zen {r.status_code}: {r.text[:160]}")
                return None, "error", r.text[:160], {}
            d = r.json()
            ch = d["choices"][0]
            msg = ch["message"]
            return msg.get("tool_calls"), ch.get("finish_reason"), msg.get("content"), d.get("usage", {})
    return send


async def run_frontier() -> dict:
    out_dir = OUT_ROOT / "frontier"
    send = make_frontier_send(_zen_key())
    print(f"[Ω-4 frontier] {FRONTIER_MODEL} via Zen, agentic loop")
    loop = await agentic_loop(send, out_dir, "frontier")
    return {"arm": "frontier", "model": FRONTIER_MODEL, **loop, "score": score_arm(out_dir)}


# ---------------- arm: bespoke (local llm-orc LoopDriver server) ----------------
def make_bespoke_send():
    async def send(messages, out_dir, produced):
        async with httpx.AsyncClient(timeout=600) as c:
            r = await c.post(BESPOKE_BASE,
                             headers={"Content-Type": "application/json"},
                             json={"model": "loop-driver", "messages": messages,
                                   "tools": TOOLS, "tool_choice": "auto"})
            if r.status_code != 200:
                print(f"    bespoke {r.status_code}: {r.text[:160]}")
                return None, "error", r.text[:160], {}
            d = r.json()
            ch = d["choices"][0]
            msg = ch["message"]
            return msg.get("tool_calls"), ch.get("finish_reason"), msg.get("content"), d.get("usage", {})
    return send


async def run_bespoke() -> dict:
    out_dir = OUT_ROOT / "bespoke"
    print("[Ω-4 bespoke] local LoopDriver via :8090 (server must be running)")
    loop = await agentic_loop(make_bespoke_send(), out_dir, "bespoke")
    return {"arm": "bespoke", **loop, "score": score_arm(out_dir)}


# ---------------- arm: ensemble (omega-4 flow, qwen3 local) ----------------
DECIDE_YAML = (Path(__file__).resolve().parents[2] / ".llm-orc" / "ensembles"
               / "spike-omega-4" / "agent-turn-omega4.yaml")


async def run_ensemble() -> dict:
    from llm_orc.core.config.ensemble_config import EnsembleLoader
    from llm_orc.core.execution.executor_factory import ExecutorFactory
    out_dir = OUT_ROOT / "ensemble"
    produced_dir = out_dir / "produced"
    produced_dir.mkdir(parents=True, exist_ok=True)
    for p in produced_dir.glob("*"):
        p.unlink()
    substrate = out_dir / "session_state.json"
    substrate.write_text(json.dumps({
        "task": TASK_PROMPT, "requested": list(FILES), "produced": [],
        "plan_queue": list(FILES), "remaining_anchor": "",
    }, indent=2))
    loader = EnsembleLoader()
    decide = loader.load_from_file(str(DECIDE_YAML))
    executor = ExecutorFactory.create_root_executor(project_dir=DECIDE_YAML.parents[2])
    max_retries, max_turns = 2, 28
    per_turn: list[float] = []
    retries: dict[str, object] = {}
    turns = 0
    print("[Ω-4 ensemble] omega-4 flow (qwen3 local), 6 files")
    while turns < max_turns:
        state = json.loads(substrate.read_text())
        if not state["plan_queue"]:
            break
        target = state["plan_queue"][0]
        exp = EXP[target]
        rc = int(retries.get(target, 0))
        turns += 1
        ltr = (f"PRODUCTION REJECTED {target}: {retries.get(target + '__e', '')}"
               if rc > 0 else "")
        t0 = time.perf_counter()
        req = json.dumps({"task": TASK_PROMPT, "substrate_path": str(substrate),
                          "last_tool_result": ltr})
        dec = await executor.execute(decide, req)
        sr = (dec.get("results", {}).get("score", {}).get("response", "")
              if isinstance(dec, dict) else "")
        try:
            decision = json.loads(sr)
        except json.JSONDecodeError:
            print(f"  [ensemble] turn {turns}: score not JSON; stop")
            break
        cap_path, di = decision.get("capability_path"), decision.get("dispatch_input", "")
        cap_name = decision.get("capability_name")
        if not cap_path:
            print(f"  [ensemble] turn {turns}: no cap_path; stop")
            break
        cap = loader.load_from_file(cap_path)
        capres = await executor.execute(cap, di)
        results = capres.get("results", {}) if isinstance(capres, dict) else {}
        content = ""
        if results:
            node = results[list(results.keys())[-1]]
            content = node.get("response", "") if isinstance(node, dict) else ""
        el = time.perf_counter() - t0
        per_turn.append(el)
        ok, why = check_deliverable(content, exp)
        if not ok:
            print(f"  [ensemble] turn {turns}: {target} via {cap_name} gate FAIL: {why} ({el:.0f}s)")
            if rc < max_retries:
                retries[target] = rc + 1
                retries[target + "__e"] = why
                continue
            state["plan_queue"] = [p for p in state["plan_queue"] if p != target]
            substrate.write_text(json.dumps(state, indent=2))
            retries.pop(target, None)
            retries.pop(target + "__e", None)
            continue
        is_code = exp["kind"] in ("python_module", "python_cli")
        (produced_dir / target).write_text(clean_content(content, is_code))
        if target not in state["produced"]:
            state["produced"].append(target)
        state["plan_queue"] = [p for p in state["plan_queue"] if p != target]
        substrate.write_text(json.dumps(state, indent=2))
        retries.pop(target, None)
        retries.pop(target + "__e", None)
        print(f"  [ensemble] turn {turns}: wrote {target} via {cap_name} ({el:.0f}s)")
    final = json.loads(substrate.read_text())
    return {"arm": "ensemble", "turns": turns, "elapsed_s": round(sum(per_turn), 1),
            "produced": final.get("produced", []), "score": score_arm(produced_dir)}


async def main() -> None:
    arm = sys.argv[1] if len(sys.argv) > 1 else "frontier"
    if arm == "frontier":
        result = await run_frontier()
    elif arm == "bespoke":
        result = await run_bespoke()
    elif arm == "ensemble":
        result = await run_ensemble()
    else:
        raise SystemExit(f"arm '{arm}' not recognized")
    print("\n==== RESULT ====")
    print(json.dumps(result, indent=2))
    (OUT_ROOT / f"{arm}_result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

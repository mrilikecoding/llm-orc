"""Spike τ (a)-isolation probe — ADR-042 condition (a).

Question: does the UNBOUNDED content anchor (all prior siblings) overload the
cheap 8b coder at scale, vs the BOUNDED (K=8) anchor, holding task/target/coder
fixed? (a) asks whether anchor-overload is real on a CLEAN task, or whether the
l15 break was a template artifact (incoherent siblings -> full-content fallback).

The full-session l15 arm could not answer this (masked by the J-3 over-extraction
churn). This isolated single-dispatch probe holds everything fixed and varies only
the anchor size, so it isolates the anchor as the cause.

UNIT: one code-generator coder call — qwen3:8b (agentic-tier-cheap-general), the
production cheap coder, real system_prompt verbatim. The anchor is built by the
REAL build_content_anchor over a fixed sibling corpus; only max_siblings varies.

SIBLING CORPUS: run 1's actual l20clean produced files (20 parseable .py modules).
All parse, so build_content_anchor takes the signatures path (no full-content
fallback) — i.e. a CLEAN-task anchor, exactly what (a) needs. Unbounded = 105
sig-lines / 1694 chars; bounded(8) = 25 lines / 440 chars.

ARMS (n=10 each):
  A_unbounded   build_content_anchor(siblings, max_siblings=None)
  B_bounded     build_content_anchor(siblings, max_siblings=8)

PRIMARY OUTCOME (AST, from the captured deliverable; the l15 break was a FORM bleed):
  parse_valid       ast.parse succeeds
  references_target imports step18 and references step18.step18 (real symbol)
Decision rule (pre-registered): (a) CONFIRMED if A_unbounded parse-validity is
materially below B_bounded (gap >= 0.3). If A ~= B (both clean), (a) REFUTED for
clean tasks — the l15 break was driven by incoherent-sibling/over-produced bloat
(full-content fallback or duplicate over-produced signatures), not clean-sibling
count, which corrects ADR-042's attribution.

Usage:  LABEL=A_unbounded MAXSIB=none python probe_a.py [n]
        LABEL=B_bounded    MAXSIB=8    python probe_a.py [n]
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.llm_orc.agentic.sibling_interface_extractor import (  # noqa: E402
    build_content_anchor,
)

OLLAMA = "http://127.0.0.1:11434/v1/chat/completions"
MODEL = "qwen3:8b"
HERE = Path(__file__).parent
GEN = HERE / "generated"
RUN1 = Path(
    "scratch/spike-tau-long-horizon/l20clean-f20/workspaces/l20clean-f20-c9a6a7e9"
)

# Verbatim from .llm-orc/ensembles/agentic-serving/code-generator.yaml.
CODER_SYSTEM = (
    "You are a coding assistant. Given a programming task or question,\n"
    "respond with the most useful code or guidance you can produce.\n"
    "Keep responses focused. Show the code change or example directly.\n"
    "When you are uncertain, say so rather than fabricating APIs or\n"
    "file paths. Format code with appropriate fenced blocks."
)

# The write form directive (loop_driver compose_form_directive: write -> bare bytes).
FORM_DIRECTIVE = (
    "\n\nOutput only the bare contents of consumer.py: no markdown fences, no "
    "prose, no explanation."
)

TASK = (
    "Write consumer.py. It imports the existing `step18` module and prints the "
    "result of calling its step18 function on the input 5. Use only functions the "
    "existing modules actually define; do not invent any."
)

_ORDER = ["base.py"] + [f"step{i}.py" for i in range(1, 19)] + ["main.py"]
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE = re.compile(r"```([A-Za-z0-9_+-]*)\n(.*?)```", re.DOTALL)


BLEED = os.environ.get("BLEED") == "1"
_BLEED_PROSE = (
    "\n\nThis module is part of the pipeline and should be used by the next "
    "step in the chain to compute the running result.\n"
)


def siblings() -> list[tuple[str, str]]:
    out = [(p, (RUN1 / p).read_text()) for p in _ORDER if (RUN1 / p).exists()]
    if BLEED:
        # Contingent arm: append prose to each sibling so it does not parse, forcing
        # build_content_anchor onto the full-content fallback path — the form-bled /
        # unparseable-sibling condition the l15 break analysis attributed the overload
        # to. Tests whether the LARGER full-content anchor (not clean-sibling count)
        # is what degrades the coder.
        out = [(p, c + _BLEED_PROSE) for p, c in out]
    return out


def anchor(max_siblings: int | None) -> str:
    return build_content_anchor(siblings(), max_siblings=max_siblings)


def extract_code(text: str) -> str | None:
    text = _THINK.sub("", text)
    blocks = _FENCE.findall(text)
    py = [c for lang, c in blocks if lang.lower() in ("python", "py", "")]
    cands = py or [c for _, c in blocks]
    if cands:
        return max(cands, key=len)
    stripped = text.strip()
    try:
        ast.parse(stripped)
        return stripped
    except SyntaxError:
        return None


def score(code: str | None) -> dict:
    if code is None:
        return {"parse_valid": False, "references_target": False, "note": "no-code"}
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"parse_valid": False, "references_target": False, "note": str(e)}
    imports_step18 = any(
        (isinstance(n, ast.Import) and any(a.name == "step18" for a in n.names))
        or (isinstance(n, ast.ImportFrom) and n.module == "step18")
        for n in ast.walk(tree)
    )
    refs_symbol = any(
        isinstance(n, ast.Attribute) and n.attr == "step18" for n in ast.walk(tree)
    ) or any(
        isinstance(n, ast.ImportFrom)
        and n.module == "step18"
        and any(a.name == "step18" for a in n.names)
        for n in ast.walk(tree)
    )
    return {
        "parse_valid": True,
        "references_target": imports_step18 and refs_symbol,
        "note": "",
    }


def run_one(max_siblings: int | None) -> tuple[dict, str, str | None]:
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": CODER_SYSTEM},
            {"role": "user", "content": TASK + anchor(max_siblings) + FORM_DIRECTIVE},
        ],
        "stream": False,
    }
    r = httpx.post(OLLAMA, json=body, timeout=600)
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    code = extract_code(content)
    res = score(code)
    res["raw_len"] = len(content)
    return res, content, code


def main() -> None:
    label = os.environ.get("LABEL", "A_unbounded")
    raw = os.environ.get("MAXSIB", "none")
    max_siblings = None if raw == "none" else int(raw)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    GEN.mkdir(exist_ok=True)
    anchor_chars = len(anchor(max_siblings))
    print(f"{label}: max_siblings={max_siblings} anchor_chars={anchor_chars} n={n}", flush=True)
    results = []
    for i in range(n):
        try:
            res, content, code = run_one(max_siblings)
        except Exception as e:  # noqa: BLE001
            res = {"parse_valid": False, "references_target": False, "note": f"{e}"}
            content, code = "", None
        (GEN / f"{label}_{i:02d}.txt").write_text(content)
        if code is not None:
            (GEN / f"{label}_{i:02d}.code.py").write_text(code)
        results.append(res)
        print(
            f"  {label} {i + 1}/{n}: parse={res['parse_valid']} "
            f"ref={res['references_target']} note={res['note'][:40]}",
            flush=True,
        )
    pv = sum(1 for r in results if r["parse_valid"])
    rt = sum(1 for r in results if r["references_target"])
    out = HERE / f"results_{label}.json"
    out.write_text(
        json.dumps(
            {
                "label": label,
                "max_siblings": max_siblings,
                "anchor_chars": anchor_chars,
                "n": n,
                "model": MODEL,
                "parse_valid": pv,
                "references_target": rt,
                "results": results,
            },
            indent=2,
        )
    )
    print(f"\n{label}: parse_valid={pv}/{n}  references_target={rt}/{n}  -> {out.name}", flush=True)


if __name__ == "__main__":
    main()

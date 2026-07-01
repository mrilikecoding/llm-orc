#!/usr/bin/env python3
"""Spike Ω-4 score stage — dispatch scoring + grounding fix.

Same library-reflection capability scoring as the Ω-dispatch score, PLUS
the grounding fix (Ω-dispatch finding #3): inject the REAL sibling API
signatures (ast-extracted from produced .py files) into the producer's
dispatch_input, not just the planner's brief. This is the ADR-039 content
anchor translated — the deliverable-producer sees exact sibling names, so
it cannot invent them (the README `kelvin_to_celsius` bug).

Emits JSON: {file_path, dispatch_input, capability_name, capability_path, scores}
"""

import ast
import json
import re
import sys
from pathlib import Path

EXT_NEEDS = {
    ".py": {"code", "python", "program", "generator", "coder", "script"},
    ".md": {"markdown", "documentation", "prose", "readme", "docs", "doc", "writer"},
}
EXT_DEFAULT = {".py": "code-generator-omega", ".md": "prose-generator-omega"}


def _ensembles_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "ensembles"


def _read_capabilities() -> list[dict]:
    seen: dict[str, dict] = {}
    for path in sorted(_ensembles_dir().rglob("*generator-omega.yaml")):
        try:
            text = path.read_text()
        except OSError:
            continue
        name_m = re.search(r"^name:\s*(\S+)", text, re.MULTILINE)
        if not name_m:
            continue
        name = name_m.group(1).strip()
        if name in seen:
            continue
        seen[name] = {"name": name, "path": str(path.resolve()), "text": text.lower()}
    return list(seen.values())


def _sibling_signatures(parse_state: dict) -> str:
    """Extract def signatures from produced .py siblings (the content anchor)."""
    substrate_path = parse_state.get("substrate_path")
    produced = parse_state.get("produced", [])
    if not substrate_path or not produced:
        return ""
    base = Path(substrate_path).parent / "produced"
    lines: list[str] = []
    for name in produced:
        p = base / name
        if not p.exists() or not name.endswith(".py"):
            continue
        try:
            tree = ast.parse(p.read_text())
        except (OSError, SyntaxError):
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                args = ", ".join(a.arg for a in node.args.args)
                lines.append(f"  {name}: def {node.name}({args})")
    return "\n".join(lines)


def main() -> None:
    raw = sys.stdin.read().strip()
    plan = {}
    parse_state = {}
    try:
        data = json.loads(raw)
        deps = data.get("dependencies", {}) if isinstance(data, dict) else {}

        plan_dep = deps.get("plan", {})
        response = plan_dep.get("response", "") if isinstance(plan_dep, dict) else ""
        if isinstance(response, str):
            try:
                plan = json.loads(response)
            except json.JSONDecodeError:
                m = re.search(r"\{[^{}]*\}", response)
                plan = json.loads(m.group(0)) if m else {}
        elif isinstance(response, dict):
            plan = response

        parse_dep = deps.get("parse", {})
        parse_response = parse_dep.get("response", "") if isinstance(parse_dep, dict) else ""
        if isinstance(parse_response, str):
            try:
                parse_state = json.loads(parse_response)
            except json.JSONDecodeError:
                parse_state = {}
        elif isinstance(parse_response, dict):
            parse_state = parse_response
    except (json.JSONDecodeError, TypeError):
        pass

    file_path = parse_state.get("next_file") or plan.get("file_path", "output.py")
    brief = plan.get("brief", "")
    if not brief:
        print(json.dumps({"success": False, "error": "Plan produced no brief"}))
        return

    ext = Path(file_path).suffix.lower()
    needs = EXT_NEEDS.get(ext, set())
    capabilities = _read_capabilities()
    if not capabilities:
        print(json.dumps({"success": False, "error": "no capabilities found"}))
        return

    scores = {c["name"]: sum(1 for kw in needs if kw in c["text"]) for c in capabilities}
    best_name = max(scores, key=lambda n: scores[n])
    if scores[best_name] == 0:
        best_name = EXT_DEFAULT.get(ext, best_name)
    chosen = next((c for c in capabilities if c["name"] == best_name), capabilities[0])

    if ext == ".md":
        form_directive = (
            "\n\nOutput ONLY the raw Markdown bytes of the file. No code "
            "fences, no Python source, no preamble."
        )
    else:
        form_directive = (
            "\n\nOutput ONLY the exact raw bytes of the file. No markdown "
            "fences, no prose, no explanations, no example blocks."
        )

    # Grounding fix: inject real sibling signatures into the PRODUCER input.
    signatures = _sibling_signatures(parse_state)
    anchor_block = ""
    if signatures:
        anchor_block = (
            "\n\nUSE THESE EXACT sibling APIs (do not rename or invent):\n"
            f"{signatures}\n"
        )

    recovery_hint = parse_state.get("recovery_hint", "") or ""
    recovery_block = ""
    if recovery_hint:
        recovery_block = (
            f"\n\nRETRY — the previous attempt was rejected: {recovery_hint}\n"
            "Emit the corrected file content ONLY."
        )

    dispatch_input = (
        f"Write {file_path}: {brief}{anchor_block}{form_directive}{recovery_block}"
    )
    print(
        json.dumps(
            {
                "file_path": file_path,
                "dispatch_input": dispatch_input,
                "capability_name": chosen["name"],
                "capability_path": chosen["path"],
                "scores": scores,
            }
        )
    )


if __name__ == "__main__":
    main()

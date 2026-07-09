#!/usr/bin/env python3
"""build-gated gather node — assemble the accept-gate contract (WP-D8).

Reads the test_writer and code_writer sub-ensemble outputs plus the turn's
criteria (the node's base input), peels the nested ensemble envelopes, strips
code fences, and emits the flat ``{requirement, code, tests}`` the accept
executor verifies. Tests-first: the tests come from test_writer (authored from
the criteria) and the code from code_writer (built against those tests).

Emits JSON: {requirement, code, tests}
"""

from __future__ import annotations

import ast
import json
import re
import sys


def _payload(raw: str) -> dict[str, object]:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def _response(dep: object) -> str:
    return dep.get("response", "") if isinstance(dep, dict) else ""


def _terminal(text: str) -> str:
    """Peel the sub-ensemble envelope layers (deliverable / output / results) to
    the terminal node's raw output (matches emit_envelope._terminal)."""
    current = text
    for _ in range(6):
        try:
            obj = json.loads(current)
        except (json.JSONDecodeError, TypeError):
            return current
        if not isinstance(obj, dict):
            return current
        if isinstance(obj.get("deliverable"), str):
            current = obj["deliverable"]
            continue
        if isinstance(obj.get("output"), str):
            current = obj["output"]
            continue
        results = obj.get("results")
        if isinstance(results, dict) and results:
            node = results[list(results.keys())[-1]]
            current = node.get("response", "") if isinstance(node, dict) else str(node)
            continue
        return current
    return current


_SHELL_LANGS = {"bash", "sh", "shell", "console", "zsh", "text"}


def _fenced(text: str) -> str | None:
    tagged = re.findall(r"```([a-zA-Z0-9_+-]*)\n(.*?)```", text, re.DOTALL)
    # only python/untagged fences are code — models append shell usage blocks
    blocks = [
        body for lang, body in tagged if lang.lower() not in _SHELL_LANGS
    ] or [body for _, body in tagged]
    if blocks:
        return "\n".join(block.strip() for block in blocks)
    return None


def _trim_to_parse(code: str, max_drops: int = 10) -> str:
    """Drop trailing non-parsing lines (bounded) — seat models sometimes leave
    a prose usage line inside the fence. Valid code returns byte-identical;
    if nothing parses within the bound, return the original unchanged."""
    lines = code.splitlines()
    for drop in range(min(max_drops, len(lines)) + 1):
        candidate = "\n".join(lines[: len(lines) - drop]).rstrip()
        if not candidate:
            break
        try:
            ast.parse(candidate)
        except SyntaxError:
            continue
        return candidate if drop else code
    return code


def _extract_code(text: str) -> str:
    fenced = _fenced(text)
    code = fenced if fenced is not None else text.strip()
    return _trim_to_parse(code)


def _extract_tests(text: str) -> str:
    fenced = _fenced(text)
    if fenced is not None:
        return fenced
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith(("def test_", "import ", "from ")):
            return "\n".join(lines[i:]).strip()
    return text.strip()


# The rung-1 context marker (classify composes it): everything before the
# marker is conversation context for generation seats; the requirement the
# verifier chain echoes is the clean turn after it (ADR-048 isolation).
_REQUEST_MARKER = "\n\nCurrent request: "

# The deliverable's destination named in the turn (mirrors classify's file
# extraction) — the executor shadows a stale workspace copy at this name.
_FILE_RE = re.compile(
    r"\b([\w./-]+\.(?:py|js|ts|jsx|tsx|json|md|txt|ya?ml|sh|go|rs|java|c|cpp|h))\b"
)

# A conversation-written file block in the rendered context. Text lines are
# newline-collapsed by the renderer, so a write body runs until the next
# 'user:'/'assistant:' line. '(truncated)' blocks are never materialized.
_WRITE_HEADER_RE = re.compile(r"^assistant: \[wrote ([^\]]+?)( \(truncated\))?\]$")
_SPEAKER_RE = re.compile(r"^(user|assistant): ")


def _workspace(context: str) -> dict[str, str]:
    """Conversation-written files as {basename: body} for the sandbox."""
    files: dict[str, str] = {}
    lines = context.splitlines()
    index = 0
    while index < len(lines):
        header = _WRITE_HEADER_RE.match(lines[index])
        index += 1
        if not header:
            continue
        body_lines = []
        while index < len(lines) and not _SPEAKER_RE.match(lines[index]):
            body_lines.append(lines[index])
            index += 1
        if not header.group(2):
            name = header.group(1).rsplit("/", 1)[-1]
            files[name] = "\n".join(body_lines).strip()
    return files


def _inject_workspace_imports(text: str, workspace: dict[str, str]) -> str:
    """Prepend imports for workspace-module names a deliverable uses but never
    imports (a common small-model omission caught by the accept gate)."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text
    defined = {
        n.name
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    prelude: list[str] = []
    for filename, body in workspace.items():
        module = filename.rsplit(".", 1)[0]
        already_imported = (
            f"import {module}" in text or f"from {module} import" in text
        )
        if already_imported or not module.isidentifier():
            continue
        try:
            exported = {
                n.name
                for n in ast.parse(body).body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            }
        except SyntaxError:
            continue
        missing = sorted((used - defined) & exported)
        if missing:
            prelude.append(f"from {module} import {', '.join(missing)}")
    return "\n".join(prelude) + "\n" + text if prelude else text


def main() -> None:
    payload = _payload(sys.stdin.read().strip())
    requirement = str(payload.get("input_data", ""))
    workspace: dict[str, str] = {}
    if _REQUEST_MARKER in requirement:
        context, requirement = requirement.rsplit(_REQUEST_MARKER, 1)
        workspace = _workspace(context)
    deps = payload.get("dependencies", {})
    if not isinstance(deps, dict):
        deps = {}

    tests = _extract_tests(_terminal(_response(deps.get("test_writer", {}))))
    code = _extract_code(_terminal(_response(deps.get("code_writer", {}))))
    tests = _inject_workspace_imports(tests, workspace)
    code = _inject_workspace_imports(code, workspace)

    file_match = _FILE_RE.search(requirement)
    target_file = file_match.group(1).rsplit("/", 1)[-1] if file_match else ""

    print(
        json.dumps(
            {
                "requirement": requirement,
                "code": code,
                "tests": tests,
                "workspace": workspace,
                "target_file": target_file,
            }
        )
    )


if __name__ == "__main__":
    main()

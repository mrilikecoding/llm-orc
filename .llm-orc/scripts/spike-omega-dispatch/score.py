#!/usr/bin/env python3
"""Spike Ω-dispatch score stage — library reflection + capability scoring.

The §3 `match` node and the §6 primitive-1 test in one script. Replaces
the static dispatch-shim: instead of a fixed `code-generator` target, it
READS the ensemble library from the filesystem (primitive 1: library-as-
data to a stage — scripts have full fs access, no engine help) and SCORES
the capabilities against the next deliverable, picking one at runtime.

Scoring is keyword rules (the deterministic-first §5 "embeddings + rules"
claim, rules-only slice): each capability's description carries keywords;
the deliverable's extension implies a need-keyword set; the capability
with the best overlap wins, with an extension default as tiebreak.

Emits the dispatch decision INCLUDING the chosen capability's file path,
so the adapter loads it by path directly — sidestepping the engine's
non-recursive `_find_ensemble_in_dirs` resolver (Ω-1 finding #1). The
adapter, not the engine, performs the dynamic dispatch (the §8 boundary).

Emits JSON: {
    "file_path": "<str>",
    "dispatch_input": "<formatted string for the chosen capability>",
    "capability_name": "<str>",
    "capability_path": "<abs path to the chosen ensemble yaml>",
    "scores": {"<name>": <int>, ...}
}
"""

import json
import re
import sys
from pathlib import Path

# Need-keywords implied by the deliverable extension.
EXT_NEEDS = {
    ".py": {"code", "python", "program", "generator", "coder", "script"},
    ".md": {"markdown", "documentation", "prose", "readme", "docs", "doc", "writer"},
}
EXT_DEFAULT = {".py": "code-generator-omega", ".md": "prose-generator-omega"}


def _ensembles_dir() -> Path:
    # __file__ = .llm-orc/scripts/spike-omega-dispatch/score.py
    return Path(__file__).resolve().parents[2] / "ensembles"


def _read_capabilities() -> list[dict]:
    """Reflect the library: read candidate capability ensembles as data.

    Candidates are the spike's *-generator-omega ensembles. Deduped by
    name (a top-level symlink can surface the same capability twice).
    """
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
        # Cheap description grab: everything is fair game for keyword match.
        seen[name] = {
            "name": name,
            "path": str(path.resolve()),
            "text": text.lower(),
        }
    return list(seen.values())


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
                if m:
                    try:
                        plan = json.loads(m.group(0))
                    except json.JSONDecodeError:
                        plan = {}
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

    # Deterministic file from the substrate (parse.next_file); plan only
    # elaborates the brief (the Ω-2 finding #3 discipline).
    file_path = parse_state.get("next_file") or plan.get("file_path", "output.py")
    brief = plan.get("brief", "")
    if not brief:
        print(json.dumps({"success": False, "error": "Plan produced no brief"}))
        return

    ext = Path(file_path).suffix.lower()
    needs = EXT_NEEDS.get(ext, set())

    capabilities = _read_capabilities()
    if not capabilities:
        print(json.dumps({"success": False, "error": "no capabilities found in library"}))
        return

    scores = {}
    for cap in capabilities:
        scores[cap["name"]] = sum(1 for kw in needs if kw in cap["text"])

    # Pick top score; tie / zero falls back to the extension default.
    best_name = max(scores, key=lambda n: scores[n])
    if scores[best_name] == 0:
        best_name = EXT_DEFAULT.get(ext, best_name)
    chosen = next((c for c in capabilities if c["name"] == best_name), capabilities[0])

    # Capability-appropriate form directive.
    if ext == ".md":
        form_directive = (
            "\n\nOutput ONLY the raw Markdown bytes of the file. No code "
            "fences, no Python, no preamble."
        )
    else:
        form_directive = (
            "\n\nOutput ONLY the exact raw bytes of the file. No markdown "
            "fences, no prose, no explanations, no example blocks."
        )

    recovery_hint = parse_state.get("recovery_hint", "") or ""
    recovery_block = ""
    if recovery_hint:
        recovery_block = (
            f"\n\nRETRY — the previous attempt was rejected: {recovery_hint}\n"
            "Emit the corrected file content ONLY."
        )

    dispatch_input = f"Write {file_path}: {brief}{form_directive}{recovery_block}"

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

"""Ω-P3 build helpers, lifted from spike-omega-e/run_e.py (pure functions only).

The dispatch-input builder and the fence-stripper that the Ω-E Python driver used
for the build step, reused unchanged so the declarative build nodes produce the
same prompts/cleanup. No module side effects (run_e.py loads ensembles + reads
sys.argv at import, so it can't be imported directly).
"""

from __future__ import annotations

import re


def clean_content(content: str, is_code: bool) -> str:
    """Strip a leading markdown code fence from generator output."""
    s = content.strip()
    if is_code and s.startswith("```"):
        m = re.search(r"```(?:[a-zA-Z]+)?\n(.*?)```", s, re.DOTALL)
        if m:
            s = m.group(1).strip()
    return s


def build_dispatch_input(d: dict, contract: list[dict], recovery: str = "") -> str:
    """The per-file build prompt: the file's contract slice + sibling APIs."""
    lines = [f"Write the file {d['file']}: {d.get('brief', '')}"]
    if d.get("defines"):
        lines.append("\nDefine EXACTLY these symbols:")
        for sym in d["defines"]:
            if sym.get("fields"):
                lines.append(
                    f"  - {sym['name']} ({sym.get('signature', 'dataclass')}) "
                    f"with fields: {', '.join(sym['fields'])}"
                )
            else:
                lines.append(f"  - {sym.get('signature', sym['name'])}")
    if d.get("imports"):
        lines.append(
            "\nInclude AT LEAST these import statements (verbatim), and add any "
            "others you need — e.g. `from __future__ import annotations` as the "
            "first line when a class or dataclass annotation refers to itself or a "
            "type defined later:"
        )
        for imp in d["imports"]:
            lines.append(f"  {imp}")
    sibs = []
    for dd in contract:
        if dd["file"] == d["file"]:
            continue
        for sym in dd.get("defines", []):
            if sym.get("fields"):
                sibs.append(
                    f"  {dd['file']}: {sym['name']}({', '.join(sym['fields'])})"
                )
            else:
                sibs.append(f"  {dd['file']}: {sym.get('signature', sym['name'])}")
    if sibs:
        lines.append("\nSibling APIs available (use these EXACT names):")
        lines.extend(sibs)
    if d.get("kind") == "markdown_doc":
        lines.append(
            "\nOutput ONLY the raw Markdown bytes. No code fences, no Python source."
        )
    else:
        lines.append(
            "\nOutput ONLY the exact file bytes. No markdown fences, no prose, "
            "no examples."
        )
    if recovery:
        lines.append(
            f"\nThe previous attempt was REJECTED: {recovery}. Fix exactly that; "
            "re-emit the file only."
        )
    return "\n".join(lines)

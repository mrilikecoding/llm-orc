"""Deterministic diff analyzer — script-agent slot for Arm B's ensemble.

Reads the diff content from input, extracts:
- Changed symbols (new functions, classes, methods, constants)
- Cross-references that should be verified (e.g., 'Should match X in Y file')
- Type annotation issues (annotations vs default values mismatch)
- Test presence (does the diff add tests for new code?)
- Security-sensitive patterns (logging api_key, password, secret, token, etc.)

Output: structured JSON with verified facts + flag list.
This output is fed to the LLM reviewer slots as anchor evidence.

NOTE: Spike code. Retained per practitioner policy until corpus close.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

def _find_repo_root() -> Path:
    """Walk upward from this file's location to find the project root (marked by pyproject.toml)."""
    p = Path(__file__).resolve()
    for parent in [p.parent, *p.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "src").exists():
            return parent
    return Path.cwd()


REPO_ROOT = _find_repo_root()


def extract_diff_body(content: str) -> str:
    """Extract the actual diff body (lines starting with +/- after @@) from a patch."""
    lines = content.splitlines()
    body_lines: list[str] = []
    in_hunk = False
    for line in lines:
        if line.startswith("@@"):
            in_hunk = True
            continue
        if in_hunk:
            body_lines.append(line)
    return "\n".join(body_lines)


def extract_added_lines(diff_body: str) -> list[tuple[int, str]]:
    """Return (line-in-new-file approx, content) for each added line."""
    out: list[tuple[int, str]] = []
    line_no = 0
    for line in diff_body.splitlines():
        if line.startswith("+++"):
            continue
        if line.startswith("+"):
            line_no += 1
            out.append((line_no, line[1:]))
        elif line.startswith(" "):
            line_no += 1
        # '-' lines do not advance the new-file counter
    return out


def extract_changed_symbols(added_lines: list[tuple[int, str]]) -> dict[str, list[dict]]:
    """Extract new functions, classes, constants from added lines."""
    fns: list[dict] = []
    classes: list[dict] = []
    constants: list[dict] = []
    for ln, content in added_lines:
        # Function defs (including methods)
        m = re.match(r"^(\s*)def\s+(\w+)\s*\(([^)]*)\)\s*(->\s*([^:]+))?:", content)
        if m:
            fns.append({
                "name": m.group(2),
                "line": ln,
                "params": m.group(3).strip(),
                "return_annotation": (m.group(5) or "").strip(),
                "is_method": bool(m.group(1)),
            })
            continue
        # Class defs
        m = re.match(r"^class\s+(\w+)", content)
        if m:
            classes.append({"name": m.group(1), "line": ln})
            continue
        # Module-level constants (UPPER_CASE = value)
        m = re.match(r"^([A-Z][A-Z0-9_]+)\s*[:=]", content)
        if m:
            constants.append({"name": m.group(1), "line": ln})
    return {"functions": fns, "classes": classes, "constants": constants}


def detect_security_patterns(added_lines: list[tuple[int, str]]) -> list[dict]:
    """Flag credentials mentioned within a windowed context of logging calls.

    Uses a 6-line window — multi-line f-string log calls (where the credential
    appears on a different line than 'logger.warning(' itself) need windowed
    detection. Per-line detection misses these.
    """
    flags: list[dict] = []
    sensitive_terms = ["api_key", "apikey", "password", "secret", "token", "credential", "auth_token"]
    log_terms = ["log.", "logger.", "logging.", "print(", ".warning(", ".error(", ".info(", ".debug("]
    window = 6  # lines

    # Build a flat list of (line_no, content) for windowed scan
    for i, (ln, content) in enumerate(added_lines):
        c_lower = content.lower()
        # Check if THIS line has a sensitive term
        sensitive_hits = [t for t in sensitive_terms if t in c_lower]
        if not sensitive_hits:
            continue
        # Check the window above THIS line for a log call
        log_call_lines = []
        for j in range(max(0, i - window), i + 1):
            window_content = added_lines[j][1].lower()
            for log_t in log_terms:
                if log_t in window_content:
                    log_call_lines.append({"line": added_lines[j][0], "content": added_lines[j][1].strip()})
                    break
        if log_call_lines:
            flags.append({
                "line": ln,
                "content": content.strip(),
                "sensitive_terms": sensitive_hits,
                "log_calls_in_window": log_call_lines,
                "concern": (
                    f"sensitive term(s) {sensitive_hits} appear in or near a logging call "
                    f"(within {window}-line window). Verify the credential is not being "
                    f"emitted to logs."
                ),
            })
    return flags


def detect_type_annotation_issues(added_lines: list[tuple[int, str]]) -> list[dict]:
    """Flag annotations like `x: int = None` (None assigned to non-Optional type)."""
    flags: list[dict] = []
    # Match `param: type = None` where type is not Optional or Union or `|`
    # Simplified: look for `: <one-word-type> = None` or `: <type> = None`
    pattern = re.compile(r"(\w+)\s*:\s*([A-Za-z][\w\.]*)\s*=\s*None")
    for ln, content in added_lines:
        for m in pattern.finditer(content):
            type_str = m.group(2)
            # Optional types are fine; this matches non-optional
            if type_str.lower() in {"optional", "any", "object"}:
                continue
            if "|" in content[m.start():]:
                # Already a union with None
                continue
            flags.append({
                "line": ln,
                "content": content.strip(),
                "param": m.group(1),
                "type_annotation": type_str,
                "concern": (
                    f"parameter '{m.group(1)}' annotated as '{type_str}' but defaults to None; "
                    f"should be '{type_str} | None'"
                ),
            })
    return flags


def detect_test_presence(diff_body: str, full_diff: str) -> dict:
    """Check whether the diff adds tests alongside new code."""
    # Look for added test files in the diff metadata
    test_files_added = bool(re.search(r"^\+\+\+\s+b/.*test.*\.py", full_diff, re.MULTILINE))
    test_methods_added = sum(
        1 for line in diff_body.splitlines()
        if line.startswith("+") and re.search(r"def\s+test_", line)
    )
    # Look for explicit "tests will be added later" or similar phrasings.
    # Patterns are tolerant of intervening words like "for the new module".
    excuses = []
    excuse_patterns = [
        r"[Tt]ests?\b[^\n]{0,80}will\s+be\s+added",
        r"[Tt]ests?\b[^\n]{0,40}follow.?up",
        r"test coverage is not\s+(?:blocking|required)",
        r"will\s+add\s+tests",
        r"[Tt]ests?\b[^\n]{0,30}coming soon",
        r"tests?\s+(to\s+)?be\s+added",
        r"[Tt]ests?\b[^\n]{0,80}in\s+a\s+follow",
    ]
    for excuse_pattern in excuse_patterns:
        m = re.search(excuse_pattern, full_diff, re.IGNORECASE)
        if m:
            excuses.append({"pattern": excuse_pattern, "match": m.group(0)})
    return {
        "test_files_added_in_diff": test_files_added,
        "test_methods_added_count": test_methods_added,
        "explicit_test_deferral_phrases_found": excuses,
    }


def find_cross_reference_claims(added_lines: list[tuple[int, str]]) -> list[dict]:
    """Find comments / docstrings that claim a value should match something elsewhere.

    Multi-line comment-aware: joins consecutive comment-like lines so 'Should match\\n
    SYMBOL_NAME' is detected as one claim.
    """
    claims: list[dict] = []
    patterns = [
        r"[Ss]hould\s+match\s+([A-Z_][A-Z0-9_]+)",
        r"[Mm]ust\s+match\s+([A-Z_][A-Z0-9_]+)",
        r"[Mm]ust\s+equal\s+([A-Z_][A-Z0-9_]+)",
        r"[Mm]irrors?\s+([A-Z_][A-Z0-9_]+)",
        r"[Ee]quivalent\s+to\s+([A-Z_][A-Z0-9_]+)",
        r"[Ss]hould\s+agree\s+with\s+([A-Z_][A-Z0-9_]+)",
    ]
    # Strip comment-marker prefixes ('# ', '## ', '"""') so multi-line claims
    # like "# Should match\n# SYMBOL_NAME" can be detected. Replace each
    # leading comment marker with whitespace so positions and line numbers
    # remain meaningful for reporting.
    cleaned_lines = []
    for _, content in added_lines:
        cleaned = re.sub(r"^\s*#+\s*", "  ", content)  # strip '# ', '## ', etc.
        cleaned = re.sub(r'^\s*"""', "   ", cleaned)
        cleaned_lines.append(cleaned)
    joined_text = "\n".join(cleaned_lines)
    # Build an index from char-offset -> line number for reporting
    offset_to_line: dict[int, int] = {}
    cursor = 0
    for ln, content in added_lines:
        offset_to_line[cursor] = ln
        cursor += len(content) + 1  # +1 for the newline

    def line_for_offset(offset: int) -> int:
        # Find the largest cursor <= offset
        prev_ln = added_lines[0][0] if added_lines else 0
        for c, ln in offset_to_line.items():
            if c <= offset:
                prev_ln = ln
            else:
                break
        return prev_ln

    for pat in patterns:
        # Use re.MULTILINE so \s also crosses newlines for `Should match\nSYMBOL`
        for m in re.finditer(pat, joined_text, re.MULTILINE | re.DOTALL):
            ln = line_for_offset(m.start())
            # Get context: the matched line
            context_line = joined_text[max(0, m.start() - 80) : m.end() + 40].replace("\n", " | ").strip()
            claims.append({
                "line": ln,
                "context_excerpt": context_line[:200],
                "referenced_symbol": m.group(1),
            })
    # Dedupe by (line, referenced_symbol)
    seen = set()
    unique = []
    for c in claims:
        key = (c["line"], c["referenced_symbol"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)
    return unique


def verify_cross_reference(symbol_name: str, search_paths: list[Path]) -> dict:
    """Search for the named symbol in the given paths; return its assigned value if found."""
    for base in search_paths:
        if not base.exists():
            continue
        if base.is_file():
            files = [base]
        else:
            files = list(base.rglob("*.py"))
        for f in files:
            try:
                content = f.read_text()
            except OSError:
                continue
            # Look for `SYMBOL = <value>` or `SYMBOL: type = <value>`
            m = re.search(
                rf"^{re.escape(symbol_name)}\s*(:[^=]+)?=\s*([^\n#]+)",
                content,
                re.MULTILINE,
            )
            if m:
                value = m.group(2).strip().rstrip(",;")
                return {
                    "found": True,
                    "file": str(f.relative_to(REPO_ROOT)) if REPO_ROOT in f.parents else str(f),
                    "raw_value": value,
                }
    return {"found": False}


def main() -> None:
    if not sys.stdin.isatty():
        config = json.loads(sys.stdin.read())
    else:
        config = {}

    # Try multiple input shapes (compatibility with different llm-orc versions)
    diff_content = ""
    if "input_data" in config:
        diff_content = config["input_data"] if isinstance(config["input_data"], str) else json.dumps(config["input_data"])
    elif "input" in config:
        diff_content = config["input"] if isinstance(config["input"], str) else json.dumps(config["input"])
    elif "data" in config:
        diff_content = config["data"]
    elif "dependencies" in config:
        # If wired as a downstream agent, take the upstream output
        deps = config["dependencies"]
        for _, agent_out in deps.items():
            if isinstance(agent_out, dict):
                diff_content = agent_out.get("response", agent_out.get("data", str(agent_out)))
            else:
                diff_content = str(agent_out)
            if diff_content:
                break

    if not diff_content:
        # Fallback: read from a known fixture path if present
        fixture_path = REPO_ROOT / "scratch/spike-c-cycle3-architecture-comparison/fixture/diff.patch"
        if fixture_path.exists():
            diff_content = fixture_path.read_text()
        else:
            print(json.dumps({
                "success": False,
                "error": "No diff content provided in input and no fixture found",
            }))
            return

    diff_body = extract_diff_body(diff_content)
    added_lines = extract_added_lines(diff_body)
    symbols = extract_changed_symbols(added_lines)
    security_flags = detect_security_patterns(added_lines)
    type_flags = detect_type_annotation_issues(added_lines)
    test_info = detect_test_presence(diff_body, diff_content)
    cross_ref_claims = find_cross_reference_claims(added_lines)

    # For each cross-ref claim, attempt to verify
    verified_refs = []
    for claim in cross_ref_claims:
        # Search the actual project src/ for the referenced symbol
        result = verify_cross_reference(
            claim["referenced_symbol"],
            [REPO_ROOT / "src", REPO_ROOT / "scratch/spike-c-cycle3-architecture-comparison/fixture"],
        )
        # Also extract the local definition for this symbol if it's defined IN the diff
        local_value = None
        for ln, content in added_lines:
            m = re.match(rf"^\s*({re.escape(claim['referenced_symbol'])}|[A-Z_][A-Z0-9_]+)\s*[:=].*", content)
            if m and m.group(1) != claim["referenced_symbol"]:
                continue
        # Find the value the diff itself assigns to a constant nearby
        constants_in_diff = [c["name"] for c in symbols["constants"]]
        verified_refs.append({
            **claim,
            "external_definition": result,
            "constants_defined_in_diff": constants_in_diff,
        })

    # Specifically look for the "DEFAULT_BUDGET_LIMIT" / "DEFAULT_MAX_TOKEN_LIMIT" cross-check
    # since that's the named cross-reference in the diff
    cross_ref_specific = []
    for c in symbols["constants"]:
        # Look for any cross-ref claim mentioning a different constant name
        for claim in cross_ref_claims:
            ref = claim["referenced_symbol"]
            if ref != c["name"]:
                # Find the local value assigned to c in the diff
                local_value = None
                for ln, content in added_lines:
                    m = re.match(rf"^\s*{re.escape(c['name'])}\s*=\s*([^#\n]+)", content)
                    if m:
                        local_value = m.group(1).strip()
                # Look up the external symbol's value
                ext = verify_cross_reference(ref, [REPO_ROOT / "src"])
                if ext["found"]:
                    cross_ref_specific.append({
                        "diff_constant": c["name"],
                        "diff_value": local_value,
                        "expected_to_match": ref,
                        "external_value": ext.get("raw_value"),
                        "external_file": ext.get("file"),
                        "values_match": local_value is not None and local_value == ext.get("raw_value"),
                    })

    output = {
        "success": True,
        "summary": {
            "added_line_count": len(added_lines),
            "new_functions_count": len(symbols["functions"]),
            "new_classes_count": len(symbols["classes"]),
            "new_constants_count": len(symbols["constants"]),
            "security_pattern_flags": len(security_flags),
            "type_annotation_flags": len(type_flags),
            "cross_reference_claims": len(cross_ref_claims),
        },
        "changed_symbols": symbols,
        "security_pattern_flags": security_flags,
        "type_annotation_flags": type_flags,
        "test_presence": test_info,
        "cross_reference_claims": cross_ref_claims,
        "cross_reference_verifications": cross_ref_specific,
        "notes": [
            "This is a deterministic analysis. The output is intended as anchor evidence "
            "for downstream LLM reviewers. False positives may occur on regex matches; "
            "false negatives may occur on patterns not covered by the regexes."
        ],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

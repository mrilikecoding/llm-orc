"""Deterministic fix verifier — Spike D pilot.

Takes a "proposed fixed file" as input and re-runs the same analysis the
spike-c diff analyzer performs. Reports which of the originally-known issue
categories are now resolved or still present.

Input formats supported (in order of detection):
  - JSON: {"input_data": "<fixed Python code>"} from invoke_ensemble
  - JSON: {"input": "<fixed Python code>"}
  - JSON: {"dependencies": {agent_name: {"response": "<fixed code>"}}}
  - Plain stdin: raw fixed Python code

Output: JSON report with per-issue status (resolved | still_present | unknown).

NOTE: Spike code. Retained per practitioner policy until corpus close.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def _find_repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p.parent, *p.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "src").exists():
            return parent
    return Path.cwd()


REPO_ROOT = _find_repo_root()


def check_security_pattern(code: str) -> dict:
    """ISSUE-2: api_key in logger contexts."""
    sensitive_terms = ["api_key", "apikey", "password", "secret", "token", "credential"]
    log_terms = ["log.", "logger.", "logging.", "print(", ".warning(", ".error(", ".info(", ".debug("]
    lines = code.splitlines()
    flags = []
    window = 6
    for i, line in enumerate(lines):
        if "api_key" not in line.lower():
            continue
        # Check window above for log call
        log_call = None
        for j in range(max(0, i - window), i + 1):
            for log_t in log_terms:
                if log_t in lines[j].lower():
                    log_call = (j + 1, lines[j].strip())
                    break
            if log_call:
                break
        if log_call:
            flags.append({
                "line": i + 1,
                "content": line.strip(),
                "log_call_line": log_call,
            })
    return {
        "issue": "ISSUE-2 (api_key in log context)",
        "status": "still_present" if flags else "resolved",
        "occurrences": flags,
    }


def check_type_annotation(code: str) -> dict:
    """ISSUE-3: `: int = None` (or other non-Optional with None default)."""
    pattern = re.compile(r"(\w+)\s*:\s*([A-Za-z][\w\.]*)\s*=\s*None")
    flags = []
    for ln, line in enumerate(code.splitlines(), start=1):
        for m in pattern.finditer(line):
            type_str = m.group(2)
            if type_str.lower() in {"optional", "any", "object"}:
                continue
            if "|" in line[m.start():m.end() + 8]:
                continue
            flags.append({
                "line": ln,
                "param": m.group(1),
                "type_annotation": type_str,
                "content": line.strip(),
            })
    return {
        "issue": "ISSUE-3 (type annotation)",
        "status": "still_present" if flags else "resolved",
        "occurrences": flags,
    }


def check_off_by_one(code: str) -> dict:
    """ISSUE-1: check_limit using > instead of >= against the limit."""
    flags = []
    in_check_limit = False
    for ln, line in enumerate(code.splitlines(), start=1):
        if re.match(r"\s*def\s+check_limit\s*\(", line):
            in_check_limit = True
            continue
        if in_check_limit:
            if re.match(r"\s*def\s+", line):
                in_check_limit = False
                continue
            # Look for `total_used() > self.limit` or similar boundary check
            if re.search(r"total_used\(\)\s*>\s*self\.limit", line) and ">=" not in line:
                flags.append({
                    "line": ln,
                    "content": line.strip(),
                    "concern": "uses '>' instead of '>='; at-limit case is permitted",
                })
            # Also check for the corrected form
    return {
        "issue": "ISSUE-1 (off-by-one in check_limit)",
        "status": "still_present" if flags else "resolved",
        "occurrences": flags,
    }


def check_cross_reference_value(code: str) -> dict:
    """ISSUE-5: DEFAULT_BUDGET_LIMIT value vs DEFAULT_MAX_TOKEN_LIMIT in orchestrator_config.py."""
    # Find DEFAULT_BUDGET_LIMIT in the proposed code
    m = re.search(r"^DEFAULT_BUDGET_LIMIT\s*[:=]\s*([^\n#]+)", code, re.MULTILINE)
    if not m:
        return {
            "issue": "ISSUE-5 (cross-file value match)",
            "status": "unknown",
            "note": "DEFAULT_BUDGET_LIMIT not found in proposed code",
        }
    code_value = m.group(1).strip().rstrip(",;")

    # Look up DEFAULT_MAX_TOKEN_LIMIT in actual orchestrator_config.py
    config_path = REPO_ROOT / "src" / "llm_orc" / "agentic" / "orchestrator_config.py"
    if not config_path.exists():
        return {
            "issue": "ISSUE-5 (cross-file value match)",
            "status": "unknown",
            "note": f"Could not find {config_path}",
            "code_value": code_value,
        }
    config_content = config_path.read_text()
    m2 = re.search(r"^DEFAULT_MAX_TOKEN_LIMIT\s*[:=]\s*([^\n#]+)", config_content, re.MULTILINE)
    if not m2:
        return {
            "issue": "ISSUE-5 (cross-file value match)",
            "status": "unknown",
            "note": "DEFAULT_MAX_TOKEN_LIMIT not found in orchestrator_config.py",
            "code_value": code_value,
        }
    config_value = m2.group(1).strip().rstrip(",;")
    # Also check if the proposed code IMPORTS the config constant — that's the "fix" pattern
    imports_config = bool(re.search(
        r"from\s+.*orchestrator_config\s+import\s+.*DEFAULT_MAX_TOKEN_LIMIT|"
        r"from\s+.*\.orchestrator_config\s+import|"
        r"DEFAULT_MAX_TOKEN_LIMIT\s*$",
        code,
        re.MULTILINE,
    ))
    return {
        "issue": "ISSUE-5 (cross-file value match)",
        "status": "resolved" if (code_value == config_value or imports_config) else "still_present",
        "code_value": code_value,
        "config_value": config_value,
        "imports_from_config": imports_config,
    }


def check_test_files_added(code: str, full_input: str) -> dict:
    """ISSUE-4: Test gap. Was test code added alongside?"""
    # Look for test functions in the input
    test_methods = sum(
        1 for line in full_input.splitlines()
        if re.search(r"def\s+test_\w+\s*\(", line)
    )
    # Look for an explicit deferral phrase still present
    deferral_present = bool(re.search(
        r"[Tt]ests?\b.{0,80}will\s+be\s+added|"
        r"[Tt]ests?\b.{0,40}follow.?up|"
        r"test coverage is not blocking",
        full_input,
        re.IGNORECASE,
    ))
    return {
        "issue": "ISSUE-4 (test coverage)",
        "status": "resolved" if (test_methods > 0 and not deferral_present) else "still_present",
        "test_methods_added": test_methods,
        "deferral_phrase_still_present": deferral_present,
    }


def main() -> None:
    raw = sys.stdin.read() if not sys.stdin.isatty() else ""
    proposed_code = ""
    config = None
    if raw.strip().startswith("{"):
        try:
            config = json.loads(raw)
        except json.JSONDecodeError:
            config = None

    if config:
        for key in ("input_data", "input", "data"):
            v = config.get(key)
            if isinstance(v, str) and v:
                proposed_code = v
                break
        if not proposed_code and "dependencies" in config:
            for _, agent_out in config["dependencies"].items():
                if isinstance(agent_out, dict):
                    proposed_code = agent_out.get("response", agent_out.get("data", "")) or ""
                else:
                    proposed_code = str(agent_out)
                if proposed_code:
                    break
    else:
        proposed_code = raw

    if not proposed_code:
        print(json.dumps({
            "success": False,
            "error": "No code content provided in input",
        }))
        return

    full_input = proposed_code  # for test-presence checks

    issue_checks = [
        check_off_by_one(proposed_code),
        check_security_pattern(proposed_code),
        check_type_annotation(proposed_code),
        check_test_files_added(proposed_code, full_input),
        check_cross_reference_value(proposed_code),
    ]

    summary = {
        "resolved": sum(1 for c in issue_checks if c["status"] == "resolved"),
        "still_present": sum(1 for c in issue_checks if c["status"] == "still_present"),
        "unknown": sum(1 for c in issue_checks if c["status"] == "unknown"),
    }

    print(json.dumps({
        "success": True,
        "summary": summary,
        "per_issue": issue_checks,
        "input_size_chars": len(proposed_code),
    }, indent=2))


if __name__ == "__main__":
    main()

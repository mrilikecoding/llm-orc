#!/usr/bin/env python3
"""Serving accept gate — deterministic executor node (WP-D8, ADR-048 §1).

The builder-independent ground-truth half of the accept gate: runs the produced
tests against the produced code and reports a deterministic pass/fail. Execution
is sandboxed in a subprocess with a wall-clock timeout (ODP-1), so a runaway or
crashing test is reported as a failure, never a frozen serve. The verifier seats
downstream (judge, gate) receive only {requirement / acceptance criteria, produced
code, produced tests, execution result} and never builder reasoning, so the signal
is independent of the builder (ADR-048 §3, isolation via seat isolation).

Reads {requirement, code, tests} from the node payload; emits
{requirement, code, tests, tests_pass, n_tests, report} — the passthrough carries
the contract and artifact to the isolated judge seat.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

RUNNER = Path(__file__).with_name("accept_executor_runner.py")
DEFAULT_TIMEOUT = 15.0


def _from_dependencies(envelope: dict[str, object]) -> dict[str, object] | None:
    """The ``{requirement, code, tests}`` contract from a dependency (the
    build-gated ``gather`` node), when the executor runs inside the ensemble.
    Returns ``None`` in flat direct-payload mode (no such dependency)."""
    deps = envelope.get("dependencies")
    if not isinstance(deps, dict):
        return None
    for node in deps.values():
        response = node.get("response") if isinstance(node, dict) else None
        if not isinstance(response, str):
            continue
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "code" in parsed and "tests" in parsed:
            return parsed
    return None


def _payload(envelope: dict[str, object]) -> dict[str, object]:
    """Extract the task payload across llm-orc's envelope formats: a root node
    arrives as ``{"input": "<json>"}``, a dependent node as
    ``{"input_data": ..., "dependencies": {...}}``; else the envelope itself."""
    for key in ("input_data", "input"):
        value = envelope.get(key)
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    return envelope


def _timeout() -> float:
    raw = os.environ.get("LLM_ORC_ACCEPT_EXECUTOR_TIMEOUT", "")
    try:
        return float(raw) if raw else DEFAULT_TIMEOUT
    except ValueError:
        return DEFAULT_TIMEOUT


def _run_sandboxed(
    code: str, tests: str, workspace: dict[str, str] | None = None
) -> tuple[bool, str, int]:
    timeout = _timeout()
    with tempfile.TemporaryDirectory() as tmp:
        # conversation-written files (gather's workspace) so tests can import
        # modules the conversation built; basenames only, no path traversal
        for name, body in (workspace or {}).items():
            safe = Path(name).name
            if safe and safe not in ("solution.py", "tests.py"):
                (Path(tmp) / safe).write_text(str(body), encoding="utf-8")
        code_path = Path(tmp) / "solution.py"
        tests_path = Path(tmp) / "tests.py"
        code_path.write_text(code, encoding="utf-8")
        tests_path.write_text(tests, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, str(RUNNER), str(code_path), str(tests_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmp,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False, f"timeout after {timeout:g}s", 0
    if completed.returncode != 0:
        detail = (completed.stderr.strip() or completed.stdout.strip())[:200]
        return False, f"runner crashed: {detail}", 0
    try:
        verdict = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return False, f"unreadable runner output: {completed.stdout[:200]!r}", 0
    return (
        bool(verdict.get("tests_pass", False)),
        str(verdict.get("report", "")),
        int(verdict.get("n_tests", 0)),
    )


def main() -> None:
    raw = sys.stdin.read().strip()
    data: dict[str, object] = {}
    try:
        envelope = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        envelope = None
    if isinstance(envelope, dict):
        data = _from_dependencies(envelope) or _payload(envelope)

    requirement = str(data.get("requirement", ""))
    code = str(data.get("code", ""))
    tests = str(data.get("tests", ""))
    raw_workspace = data.get("workspace")
    workspace = (
        {str(k): str(v) for k, v in raw_workspace.items()}
        if isinstance(raw_workspace, dict)
        else {}
    )

    tests_pass, report, n_tests = _run_sandboxed(code, tests, workspace)

    print(
        json.dumps(
            {
                "requirement": requirement,
                "code": code,
                "tests": tests,
                "tests_pass": tests_pass,
                "n_tests": n_tests,
                "report": report,
            }
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Judge false-reject measurement harness (#84).

Runs labeled {requirement, code, tests} fixtures through the REAL accept
executor and the REAL adequacy-judge seat, N samples each, and reports
false-reject / false-accept rates per fixture class plus per-fixture flip
variance. Fidelity by construction: the harness ensemble mirrors the gated
round's wiring (script executor -> `ensemble: adequacy-judge` with
`input_scope: dependencies`), the scripts and the judge definition are
copied live from `.llm-orc/`, and model profiles resolve from the repo cwd
exactly as the serve resolves them.

Run from the repo root (rig-local, sequential):

    uv run python benchmarks/judge_adequacy/run.py --samples 8

Artifacts: one JSONL per run under benchmarks/judge_adequacy/runs/
(row per sample; final row is the summary). Retained, not gitignored —
they are the evidence the judge-tuning loop is graded on.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[2]
LLM_ORC = REPO / ".llm-orc"
RUNS_DIR = Path(__file__).resolve().parent / "runs"

# Mirrors build-gated-round's executor -> judge wiring exactly: the judge
# reads only the executor's echoed contract (input_scope: dependencies).
_FIXTURE_RUN_YAML = """\
name: judge-fixture-run
description: |
  Measurement harness ensemble (#84): the real accept executor echoes the
  fixture contract; the real adequacy-judge seat judges it under the same
  isolation the gated build round uses.
agents:
  - name: executor
    script: scripts/agentic_serving/accept_executor.py
  - name: judge
    ensemble: adequacy-judge
    depends_on: [executor]
    input_scope: dependencies
"""


def load_fixtures(path: Path) -> list[dict[str, Any]]:
    """Labeled fixtures, validated: every entry carries the contract and
    both ground-truth labels."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    fixtures = data.get("fixtures", []) if isinstance(data, dict) else []
    required = (
        "name",
        "class",
        "requirement",
        "code",
        "tests",
        "expected_adequate",
        "expected_tests_pass",
    )
    for fixture in fixtures:
        missing = [key for key in required if key not in fixture]
        if missing:
            raise ValueError(f"fixture {fixture.get('name', '?')} missing {missing}")
    return list(fixtures)


def parse_verdict(result: dict[str, Any]) -> dict[str, Any]:
    """The judge's verdict from a judge-fixture-run result: peel the judge
    node's sub-ensemble envelope to the model's raw JSON (the same nesting
    the accept gate peels)."""
    results = result.get("results", {})
    judge_node = results.get("judge", {}) if isinstance(results, dict) else {}
    response = judge_node.get("response", "") if isinstance(judge_node, dict) else ""
    try:
        child = json.loads(response)
    except (json.JSONDecodeError, TypeError):
        return {"tests_adequate": None, "reason": "judge response unreadable"}
    inner = child.get("results", {}) if isinstance(child, dict) else {}
    node = inner.get("judge", {}) if isinstance(inner, dict) else {}
    raw = node.get("response", "") if isinstance(node, dict) else ""
    try:
        verdict = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"tests_adequate": None, "reason": f"verdict unreadable: {raw[:80]!r}"}
    if isinstance(verdict, dict) and isinstance(verdict.get("tests_adequate"), bool):
        return {
            "tests_adequate": verdict["tests_adequate"],
            "reason": str(verdict.get("reason", "")),
        }
    return {"tests_adequate": None, "reason": f"verdict malformed: {raw[:80]!r}"}


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """False-reject / false-accept rates per class and overall, plus
    per-fixture adequacy rates (flip variance). An unreadable verdict counts
    as a reject — that is what the gate does with it."""

    def rates(rows: list[dict[str, Any]]) -> dict[str, Any]:
        adequate = [r for r in rows if r["expected_adequate"]]
        inadequate = [r for r in rows if not r["expected_adequate"]]
        rejected = [r for r in adequate if r["tests_adequate"] is not True]
        accepted = [r for r in inadequate if r["tests_adequate"] is True]
        return {
            "samples": len(rows),
            "false_reject_rate": (
                len(rejected) / len(adequate) if adequate else None
            ),
            "false_accept_rate": (
                len(accepted) / len(inadequate) if inadequate else None
            ),
            "unreadable": sum(1 for r in rows if r["tests_adequate"] is None),
        }

    classes: dict[str, list[dict[str, Any]]] = {}
    fixtures: dict[str, list[dict[str, Any]]] = {}
    for row in samples:
        classes.setdefault(row["class"], []).append(row)
        fixtures.setdefault(str(row.get("name", "?")), []).append(row)

    return {
        "classes": {name: rates(rows) for name, rows in sorted(classes.items())},
        "fixtures": {
            name: {
                "samples": len(rows),
                "adequate_rate": sum(
                    1 for r in rows if r["tests_adequate"] is True
                )
                / len(rows),
            }
            for name, rows in sorted(fixtures.items())
        },
        "overall": rates(samples),
    }


def _build_project(root: Path) -> Path:
    """A temp project mirroring the live serving config: real scripts, the
    live adequacy-judge definition, and the harness ensemble."""
    ensembles = root / "ensembles"
    ensembles.mkdir()
    scripts = root / "scripts" / "agentic_serving"
    scripts.parent.mkdir()
    shutil.copytree(LLM_ORC / "scripts" / "agentic_serving", scripts)
    shutil.copy(
        LLM_ORC / "ensembles" / "agentic-serving" / "adequacy-judge.yaml",
        ensembles / "adequacy-judge.yaml",
    )
    (ensembles / "judge-fixture-run.yaml").write_text(_FIXTURE_RUN_YAML)
    return root


async def _run_samples(
    fixtures: list[dict[str, Any]], samples: int, out_path: Path
) -> list[dict[str, Any]]:
    from llm_orc.core.config.ensemble_config import EnsembleLoader
    from llm_orc.core.execution.executor_factory import ExecutorFactory

    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as td:
        project = _build_project(Path(td))
        config = EnsembleLoader().load_from_file(
            str(project / "ensembles" / "judge-fixture-run.yaml")
        )
        with out_path.open("a", encoding="utf-8") as out:
            for fixture in fixtures:
                contract = json.dumps(
                    {
                        "requirement": fixture["requirement"],
                        "code": fixture["code"],
                        "tests": fixture["tests"],
                    }
                )
                for index in range(samples):
                    executor = ExecutorFactory.create_root_executor(
                        project_dir=project, save_artifacts=False
                    )
                    started = time.monotonic()
                    result = await executor.execute(config, contract)
                    verdict = parse_verdict(result)
                    row = {
                        "name": fixture["name"],
                        "class": fixture["class"],
                        "expected_adequate": fixture["expected_adequate"],
                        "tests_adequate": verdict["tests_adequate"],
                        "reason": verdict["reason"],
                        "sample": index,
                        "seconds": round(time.monotonic() - started, 2),
                    }
                    rows.append(row)
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    marker = (
                        "?"
                        if verdict["tests_adequate"] is None
                        else ("A" if verdict["tests_adequate"] else "R")
                    )
                    print(
                        f"{fixture['name']} [{index + 1}/{samples}] {marker}",
                        flush=True,
                    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=Path(__file__).resolve().parent / "fixtures.yaml",
    )
    args = parser.parse_args()

    fixtures = load_fixtures(args.fixtures)
    RUNS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out_path = RUNS_DIR / f"{stamp}.jsonl"

    rows = asyncio.run(_run_samples(fixtures, args.samples, out_path))
    summary = summarize(rows)
    with out_path.open("a", encoding="utf-8") as out:
        out.write(json.dumps({"summary": summary}) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"\nrun artifact: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

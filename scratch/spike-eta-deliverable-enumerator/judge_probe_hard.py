"""Spike η — BREAK-THE-JUDGE probe (harder task shapes).

The clean baseline (4/4 live converge) + the first isolated probe (0/12) showed the judge
handles the 5-file temperature task. The practitioner's directive: try to BREAK it on a
harder shape before committing ADR-040's scope. The σ failure mode is *under-counting the
requested set*, so each condition stresses requested-minus-produced a different way, at a
partial produced-state where COMPLETE = a false-COMPLETE (the break):

  H1_many     8 EXPLICITLY-described deliverables; tested at 5/8 and 7/8 produced
  H2_implicit a "production-ready" service whose deliverables are IMPLIED, not listed;
              tested at 3/6 and 5/6 (the judge must infer the full set -> under-count risk)
  H3_compact  COMPACTED task (no deliverable list, "keep working until complete") + 2 files
              produced -> simulates later-turn client task-compaction (the live-context worry
              the clean runs never triggered because the full task persisted)

Same real judge prompt (compose_judgment_message + _JUDGE_SYSTEM + parse_verdict), qwen3:14b,
$0 local. Usage: python judge_probe_hard.py [reps]   # default 6
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

from llm_orc.agentic.loop_driver import (
    _JUDGE_SYSTEM,
    compose_judgment_message,
    parse_verdict,
)
from llm_orc.agentic.session_action_record import ActionRecord

OLLAMA = "http://127.0.0.1:11434/v1/chat/completions"
MODEL = "qwen3:14b"
HERE = Path(__file__).parent
GEN = HERE / "judge_probe_hard_out"

H1_TASK = (
    "Build a command-line task manager in Python. It needs: a storage module that persists "
    "tasks to disk, a task model module, an add-command module, a list-command module, a "
    "delete-command module, unit tests for the storage module, unit tests for the task model, "
    "and a README documenting the commands."
)
H1_FULL = [
    "storage.py", "models.py", "add.py", "list.py", "delete.py",
    "test_storage.py", "test_models.py", "README.md",
]  # 8 deliverables

H2_TASK = (
    "Build a production-ready URL shortener service in Python, with full test coverage and "
    "documentation. It should be ready to run."
)
H2_FULL = [
    "app.py", "shortener.py", "storage.py", "test_shortener.py", "README.md",
    "requirements.txt",
]  # 6 inferred deliverables (the reference is itself a judgment — that is the point)

H3_TASK = (
    "Keep working on the temperature-conversion library in this directory until it is complete."
)  # COMPACTED: no deliverable list (simulates a later-turn truncation of the original ask)
H3_FULL = [
    "temperature.py", "test_temperature.py", "cli.py", "test_cli.py", "README.md",
]  # the real intended set, invisible to the compacted task text

# (condition, task, produced-prefix length k, total reference, label)
CASES = [
    ("H1_many", H1_TASK, H1_FULL, 5, "5of8"),
    ("H1_many", H1_TASK, H1_FULL, 7, "7of8"),
    ("H2_implicit", H2_TASK, H2_FULL, 3, "3of6"),
    ("H2_implicit", H2_TASK, H2_FULL, 5, "5of6"),
    ("H3_compact", H3_TASK, H3_FULL, 2, "2of5"),
]


def records_for(files: list[str], k: int) -> tuple[ActionRecord, ...]:
    return tuple(
        ActionRecord(action_kind="write", target_path=files[i],
                     result="Wrote file successfully.")
        for i in range(k)
    )


def ask_judge(message: str) -> str:
    body = {"model": MODEL, "messages": [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": message}], "stream": False}
    r = httpx.post(OLLAMA, json=body, timeout=600)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def main() -> None:
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    GEN.mkdir(parents=True, exist_ok=True)
    results = {}
    for cond, task, full, k, label in CASES:
        msg = compose_judgment_message(task, records_for(full, k))
        verdicts = []
        for i in range(reps):
            try:
                raw = ask_judge(msg)
            except Exception as e:  # noqa: BLE001
                raw = f"ERROR {e}"
            v = parse_verdict(raw)
            verdicts.append(v)
            (GEN / f"{cond}_{label}_{i:02d}.txt").write_text(raw)
        n_total = len(full)
        false_complete = sum(1 for v in verdicts if v == "COMPLETE")
        remaining = sum(1 for v in verdicts if v == "REMAINING")
        miss = sum(1 for v in verdicts if v is None)
        key = f"{cond}_{label}"
        results[key] = {"produced": k, "total": n_total,
                        "false_COMPLETE": false_complete, "REMAINING": remaining,
                        "parse_miss": miss, "n": reps}
        broke = "  <-- BROKE THE JUDGE" if false_complete > 0 else ""
        print(f"  {key:18s} ({k}/{n_total} produced): "
              f"false-COMPLETE={false_complete}/{reps} REMAINING={remaining} "
              f"miss={miss}{broke}")

    out = HERE / f"judge_probe_hard_results_reps{reps}.json"
    out.write_text(json.dumps({"model": MODEL, "reps": reps, "results": results}, indent=2))
    print("\n==== BREAK-THE-JUDGE SUMMARY ====")
    any_broke = any(v["false_COMPLETE"] > 0 for v in results.values())
    for k, v in results.items():
        print(f"{k:18s}: false-COMPLETE {v['false_COMPLETE']}/{v['n']} "
              f"at {v['produced']}/{v['total']} produced")
    print(f"\nJudge {'BROKE on >=1 harder shape' if any_broke else 'HELD on all harder shapes'}.")
    print(f"-> {out.name}")


if __name__ == "__main__":
    main()

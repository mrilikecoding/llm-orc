"""Spike η — ISOLATED JUDGE PROBE (causal isolation of the false-COMPLETE).

Premise under test: the η DECIDE-gate reframe assumed the unnamed-task (no-files)
path "inherits the σ false-COMPLETE rate." Arm-A live session 1 converged cleanly,
questioning that. This probe isolates the judge's termination decision from the slow
multi-turn coder loop: it constructs the EXACT judge prompt the live loop builds
(``compose_judgment_message`` + ``_JUDGE_SYSTEM``), at each intermediate produced-
state, and reads the verdict — no file generation, so ~35s/call instead of ~5min/turn.

Two conditions per state:
  baseline   enumerated=None       — the produced-only digest (the σ bottleneck)
  armD       enumerated=<5 names>  — the framework checklist seeds the judge (arm D)

States (k = files produced of the 5-deliverable unnamed temp-lib task):
  k=1, k=4   INCOMPLETE -> verdict COMPLETE is a FALSE-COMPLETE (the σ failure)
  k=5        COMPLETE   -> verdict COMPLETE is CORRECT (and tests no over-waiting)

The judge runs on qwen3:14b — faithful to σ's measured 14b judge AND the production
seat/judge. $0 local.

Usage:  python judge_probe.py [reps]   # default 6 -> 3 states x 2 conds x 6 = 36 calls
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
MODEL = "qwen3:14b"  # the production seat/judge; σ's measured false-COMPLETE judge
HERE = Path(__file__).parent
GEN = HERE / "judge_probe_out"

TASK = (
    "Build a small temperature-conversion library in this directory. It needs: "
    "(1) a module with three conversion functions — celsius to fahrenheit, "
    "fahrenheit to celsius, and celsius to kelvin; (2) unit tests for those "
    "conversion functions; (3) a command-line tool that converts a value given as "
    "command-line arguments; (4) tests for the command-line tool; (5) documentation "
    "explaining how to use the command-line tool. The tests must import the real "
    "module under test, the CLI must call the real conversion functions, and the docs "
    "must describe the real CLI usage."
)

# Plausible production order of the 5 deliverables (the coder's own filenames — the
# names the seat-filler emits; the enumerated checklist below uses the SAME names, the
# aligned case: it isolates "does an explicit requested list fix the false-COMPLETE"
# from the separate naming-coordination question arm C already flagged).
PRODUCED = [
    "converters.py",
    "test_converters.py",
    "cli.py",
    "test_cli.py",
    "README.md",
]
ENUMERATED = frozenset(PRODUCED)


def records_for(k: int) -> tuple[ActionRecord, ...]:
    """Action records for the first k files produced (each a successful write)."""
    return tuple(
        ActionRecord(
            action_kind="write",
            target_path=PRODUCED[i],
            result="Wrote file successfully.",
        )
        for i in range(k)
    )


def ask_judge(message: str) -> str:
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": message},
        ],
        "stream": False,
    }
    r = httpx.post(OLLAMA, json=body, timeout=600)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def main() -> None:
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    GEN.mkdir(parents=True, exist_ok=True)
    states = [1, 4, 5]
    conditions = {"baseline": None, "armD": ENUMERATED}
    summary: dict[str, dict[str, dict]] = {}

    for cond, enum in conditions.items():
        summary[cond] = {}
        for k in states:
            msg = compose_judgment_message(TASK, records_for(k), enumerated=enum)
            verdicts: list[str | None] = []
            for i in range(reps):
                try:
                    raw = ask_judge(msg)
                except Exception as e:  # noqa: BLE001
                    raw = f"ERROR {e}"
                verdict = parse_verdict(raw)
                verdicts.append(verdict)
                (GEN / f"{cond}_k{k}_{i:02d}.txt").write_text(raw)
            complete = sum(1 for v in verdicts if v == "COMPLETE")
            remaining = sum(1 for v in verdicts if v == "REMAINING")
            miss = sum(1 for v in verdicts if v is None)
            incomplete_state = k < 5
            # the diagnostic figure per state:
            #   incomplete state -> COMPLETE is a FALSE-COMPLETE (bad)
            #   complete state   -> COMPLETE is CORRECT (good)
            figure = (
                f"false-COMPLETE={complete}/{reps}"
                if incomplete_state
                else f"correct-COMPLETE={complete}/{reps}"
            )
            summary[cond][f"k{k}"] = {
                "complete": complete,
                "remaining": remaining,
                "parse_miss": miss,
                "n": reps,
                "incomplete_state": incomplete_state,
            }
            print(f"  {cond:8s} k={k} ({'INCOMPLETE' if incomplete_state else 'COMPLETE '}): "
                  f"COMPLETE={complete} REMAINING={remaining} miss={miss}  -> {figure}")

    out = HERE / f"judge_probe_results_reps{reps}.json"
    out.write_text(json.dumps({"model": MODEL, "reps": reps, "states": states,
                               "produced": PRODUCED, "summary": summary}, indent=2))
    print("\n==== JUDGE PROBE SUMMARY ====")
    for cond in conditions:
        fc1 = summary[cond]["k1"]["complete"]
        fc4 = summary[cond]["k4"]["complete"]
        cc5 = summary[cond]["k5"]["complete"]
        print(f"{cond:8s}: false-COMPLETE k1={fc1}/{reps} k4={fc4}/{reps} | "
              f"correct-COMPLETE k5={cc5}/{reps}")
    print(f"-> {out.name}")
    print("\nRead: baseline high false-COMPLETE => σ premise holds (enumerator needed).")
    print("      baseline low  false-COMPLETE => judge-fallback adequate (premise weak).")
    print("      armD lower than baseline at k1/k4 => the checklist is the fix.")


if __name__ == "__main__":
    main()

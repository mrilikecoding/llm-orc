"""Spike η — Deliverable Enumerator, ARM C (isolated enumeration accuracy).

Pre-registration:
  docs/agentic-serving/essays/research-logs/cycle-7-spike-eta-deliverable-enumerator.md
  (the "PRE-REGISTRATION — CONCRETE ARM-C BATTERY" section pins tasks, reference
  role sets, the enumerator prompt, and the scoring rule — all recorded before this run.)

UNIT: a single structurally-bounded deliverable-enumerator call — qwen3:8b (the production
cheap model; the same model the live gate would run the enumerator on). One isolated call
per task per rep; NO live session. The model is asked to list the files a complete solution
must produce, or output NONE for an irreducibly-semantic task.

PRIMARY OUTCOME (the gate): RECALL = covered reference roles / total reference roles, pooled
over the three enumerable tasks (T1/T2/T3) across reps. Recall >= 0.9 proceeds to the live
A/B/D arms; recall < 0.9 stops the spike (ADR-040's named-file scope stays as the grounded
boundary). Precision is secondary (over-enumeration is flagged). The two semantic tasks
(T4/T5) are partition checks: the correct output is NONE.

Usage:  python enumerate_probe.py [reps]      # default 3 reps/task -> n=15
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import httpx

OLLAMA = "http://127.0.0.1:11434/v1/chat/completions"
MODEL = "qwen3:8b"  # the production cheap model (agentic-tier-cheap-general)
HERE = Path(__file__).parent
GEN = HERE / "generated"

ENUMERATOR_SYSTEM = (
    "You are a deliverable enumerator for an automated coding session. Given a "
    "user's task, list the files a complete solution must produce — and only those "
    "files. Use conventional Python project filenames (a module, its test file, a "
    "CLI entry point, a README, a packaging file) when, and only when, the task "
    "implies them. Do not invent deliverables the task does not ask for. Output ONLY "
    "a filename list, one filename per line, nothing else — no prose, no explanation, "
    "no numbering, no commentary. If the task asks for a single textual answer (a "
    "summary, an explanation, an opinion) rather than files to write, output exactly: "
    "NONE"
)

# ---- Task battery (verbatim from the pre-registration) ----------------------

T1_TEMP_LIB = (
    "Build a small temperature-conversion library in this directory. It needs: "
    "(1) a module with three conversion functions — celsius to fahrenheit, "
    "fahrenheit to celsius, and celsius to kelvin; (2) unit tests for those "
    "conversion functions; (3) a command-line tool that converts a value given as "
    "command-line arguments; (4) tests for the command-line tool; (5) documentation "
    "explaining how to use the command-line tool. The tests must import the real "
    "module under test, the CLI must call the real conversion functions, and the docs "
    "must describe the real CLI usage."
)

T2_MOD_TESTS = (
    "Write a Python module of string utilities with functions to reverse a string, "
    "check whether a string is a palindrome, and count the vowels in a string. Also "
    "write unit tests for those functions."
)

T3_CSV_JSON = (
    "Write a small tool that converts a CSV file to JSON. It needs the converter "
    "module itself, unit tests for it, and a command-line entry point that takes an "
    "input CSV path and an output JSON path."
)

T4_SUMMARIZE = (
    "Summarize the following text in two or three sentences: The mitochondrion is a "
    "double-membrane-bound organelle found in most eukaryotic cells. It generates most "
    "of the cell's supply of ATP, used as a source of chemical energy. Mitochondria "
    "have their own small genome, inherited maternally in many organisms."
)

T5_EXPLAIN = (
    "Explain what the following Python function does and whether it has any bugs: "
    "def add(a, b): return a - b"
)

TASKS = {
    "T1_temp_lib": {"task": T1_TEMP_LIB, "kind": "enumerable", "n_roles": 5},
    "T2_mod_tests": {"task": T2_MOD_TESTS, "kind": "enumerable", "n_roles": 2},
    "T3_csv_json": {"task": T3_CSV_JSON, "kind": "enumerable", "n_roles": 3},
    "T4_summarize": {"task": T4_SUMMARIZE, "kind": "semantic", "n_roles": 0},
    "T5_explain": {"task": T5_EXPLAIN, "kind": "semantic", "n_roles": 0},
}

# ---- Output parsing ---------------------------------------------------------

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_FILENAME = re.compile(r"[A-Za-z0-9_./-]+\.[A-Za-z0-9]{1,6}")


def parse_prediction(raw: str) -> tuple[list[str], bool]:
    """Return (predicted basenames, none_emitted).

    Strips qwen3 <think> reasoning, extracts filename-looking tokens (basename-
    normalized, deduped, order-preserved). If no filename token is found and the
    answer mentions NONE, none_emitted is True (the semantic-partition signal).
    """
    answer = _THINK.sub("", raw).strip()
    names: list[str] = []
    seen: set[str] = set()
    for tok in _FILENAME.findall(answer):
        base = tok.rsplit("/", 1)[-1].strip(".")
        low = base.lower()
        if low and low not in seen:
            seen.add(low)
            names.append(base)
    none_emitted = bool(re.search(r"\bNONE\b", answer)) and not names
    return names, none_emitted


# ---- Role coverage (the pre-registered stem-flexible matcher) ---------------

def _stem_ext(name: str) -> tuple[str, str]:
    base = name.rsplit("/", 1)[-1].lower()
    if "." in base:
        stem, ext = base.rsplit(".", 1)
    else:
        stem, ext = base, ""
    return stem, ext


def _is_test(stem: str) -> bool:
    return stem.startswith("test") or stem.endswith("_test") or stem in {"tests", "test"}


_MODULE_STEMS = {
    "converters", "converter", "conversions", "temperature", "temp",
    "temperatures", "temp_conversions", "temperature_converter",
}
_CLI_STEMS = {"cli", "__main__", "main", "command_line", "console", "app"}
_STRING_STEMS = {
    "string_utils", "strings", "stringutils", "utils", "text_utils",
    "string_ops", "string_tools", "stringtools",
}
_CSV_STEMS = {
    "csv_to_json", "converter", "csv2json", "csvjson", "convert", "csv_json",
    "csvtojson", "csv2json_converter",
}


def cover_T1(names: list[str]) -> tuple[set[str], list[str]]:
    """T1 temp_lib: 5 roles, two of them test roles disambiguated by what they test."""
    roles: set[str] = set()
    over: list[str] = []
    test_files: list[str] = []
    for n in names:
        stem, ext = _stem_ext(n)
        if _is_test(stem):
            test_files.append(stem)
            continue
        if stem.startswith("readme") or stem in {"docs", "documentation"}:
            roles.add("readme")
        elif stem in _CLI_STEMS:
            roles.add("cli")
        elif stem in _MODULE_STEMS and ext == "py":
            roles.add("converters_module")
        else:
            over.append(n)
    # Assign test files: cli-tests vs converters-tests vs generic-fills-uncovered.
    generic: list[str] = []
    for stem in test_files:
        if "cli" in stem:
            roles.add("cli_tests")
        elif any(m in stem for m in _MODULE_STEMS) or "convert" in stem:
            roles.add("converters_tests")
        else:
            generic.append(stem)
    for _ in generic:
        if "converters_tests" not in roles:
            roles.add("converters_tests")
        elif "cli_tests" not in roles:
            roles.add("cli_tests")
        else:
            over.append("test:" + _)
    return roles, over


def cover_T2(names: list[str]) -> tuple[set[str], list[str]]:
    """T2 mod_tests: string_module + string_tests."""
    roles: set[str] = set()
    over: list[str] = []
    for n in names:
        stem, ext = _stem_ext(n)
        if _is_test(stem):
            roles.add("string_tests")
        elif stem in _STRING_STEMS and ext == "py":
            roles.add("string_module")
        elif ext == "py" and "string" in stem:
            roles.add("string_module")
        else:
            over.append(n)
    return roles, over


def cover_T3(names: list[str]) -> tuple[set[str], list[str]]:
    """T3 csv_json: converter_module + converter_tests + cli."""
    roles: set[str] = set()
    over: list[str] = []
    for n in names:
        stem, ext = _stem_ext(n)
        if _is_test(stem):
            roles.add("converter_tests")
        elif stem in _CLI_STEMS:
            roles.add("cli")
        elif ext == "py" and (
            stem in _CSV_STEMS or "csv" in stem or "json" in stem or "convert" in stem
        ):
            roles.add("converter_module")
        else:
            over.append(n)
    return roles, over


COVERERS = {"T1_temp_lib": cover_T1, "T2_mod_tests": cover_T2, "T3_csv_json": cover_T3}
N_ROLES = {"T1_temp_lib": 5, "T2_mod_tests": 2, "T3_csv_json": 3}

# ---- Runner -----------------------------------------------------------------

def run_one(task_id: str) -> tuple[str, list[str], bool]:
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": ENUMERATOR_SYSTEM},
            {"role": "user", "content": TASKS[task_id]["task"]},
        ],
        "stream": False,
    }
    r = httpx.post(OLLAMA, json=body, timeout=600)
    r.raise_for_status()
    raw = r.json()["choices"][0]["message"]["content"]
    names, none_emitted = parse_prediction(raw)
    return raw, names, none_emitted


def score_rep(task_id: str, names: list[str], none_emitted: bool) -> dict:
    kind = TASKS[task_id]["kind"]
    if kind == "semantic":
        # Pre-registered criterion: "a non-empty output is a partition failure."
        # Zero confabulated file deliverables = PASS (the gate then routes to
        # natural-finish, no phantom-file wait), whether the model signals the
        # clean literal NONE or simply answers the task without naming files.
        # ``clean_none`` is the secondary "used the clean NONE signal" rate.
        partition_ok = not names
        return {
            "kind": "semantic",
            "none_emitted": none_emitted,
            "clean_none": none_emitted and not names,
            "predicted": names,
            "partition_ok": partition_ok,
            "confabulated": names,
        }
    roles, over = COVERERS[task_id](names)
    nr = N_ROLES[task_id]
    recall = len(roles) / nr
    precision = len(roles) / len(names) if names else 0.0
    return {
        "kind": "enumerable",
        "predicted": names,
        "covered_roles": sorted(roles),
        "n_roles": nr,
        "n_covered": len(roles),
        "over_enumeration": over,
        "recall": recall,
        "precision": precision,
        "pred_count": len(names),
    }


def main() -> None:
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    GEN.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, list[dict]] = {}
    for task_id in TASKS:
        all_results[task_id] = []
        for i in range(reps):
            try:
                raw, names, none_emitted = run_one(task_id)
            except Exception as e:  # noqa: BLE001 — record and continue
                rec = {"error": str(e), "predicted": [], "kind": TASKS[task_id]["kind"]}
                all_results[task_id].append(rec)
                print(f"  {task_id} {i + 1}/{reps}: ERROR {e}")
                continue
            (GEN / f"{task_id}_{i:02d}.txt").write_text(raw)
            rec = score_rep(task_id, names, none_emitted)
            rec["rep"] = i
            all_results[task_id].append(rec)
            if rec["kind"] == "enumerable":
                print(
                    f"  {task_id} {i + 1}/{reps}: recall={rec['recall']:.2f} "
                    f"covered={rec['n_covered']}/{rec['n_roles']} "
                    f"pred={rec['predicted']} over={rec['over_enumeration']}"
                )
            else:
                print(
                    f"  {task_id} {i + 1}/{reps}: partition_ok={rec['partition_ok']} "
                    f"none={rec['none_emitted']} pred={rec['predicted']}"
                )

    # ---- Pooled scoring (the gate) ----
    enum_tasks = [t for t in TASKS if TASKS[t]["kind"] == "enumerable"]
    pooled_covered = 0
    pooled_roles = 0
    pooled_predicted = 0
    per_task_recall: dict[str, float] = {}
    counts: dict[str, list[int]] = {}
    for t in enum_tasks:
        recs = [r for r in all_results[t] if "recall" in r]
        cov = sum(r["n_covered"] for r in recs)
        tot = sum(r["n_roles"] for r in recs)
        pooled_covered += cov
        pooled_roles += tot
        pooled_predicted += sum(r["pred_count"] for r in recs)
        per_task_recall[t] = cov / tot if tot else 0.0
        counts[t] = [r["pred_count"] for r in recs]

    sem_tasks = [t for t in TASKS if TASKS[t]["kind"] == "semantic"]
    sem_reps = [r for t in sem_tasks for r in all_results[t]]
    partition_ok = sum(1 for r in sem_reps if r.get("partition_ok"))
    clean_none = sum(1 for r in sem_reps if r.get("clean_none"))

    pooled_recall = pooled_covered / pooled_roles if pooled_roles else 0.0
    pooled_precision = pooled_covered / pooled_predicted if pooled_predicted else 0.0

    summary = {
        "model": MODEL,
        "reps_per_task": reps,
        "n_total": reps * len(TASKS),
        "pooled_recall": pooled_recall,
        "pooled_precision": pooled_precision,
        "pooled_covered": pooled_covered,
        "pooled_roles": pooled_roles,
        "per_task_recall": per_task_recall,
        "predicted_count_stability": counts,
        "partition_ok": f"{partition_ok}/{len(sem_reps)}",
        "clean_none": f"{clean_none}/{len(sem_reps)}",
        "gate_note": "automated-strict recall; manual adjudication of flagged "
        "over-names applied in RESULTS write-up per the pre-registered "
        "purpose-match-with-manual-audit rule",
        "gate": "PASS (>=0.9)" if pooled_recall >= 0.9 else "FAIL (<0.9)",
        "results": all_results,
    }
    out = HERE / f"results_armC_reps{reps}.json"
    out.write_text(json.dumps(summary, indent=2))
    print("\n==== ARM C SUMMARY ====")
    print(f"pooled recall    = {pooled_recall:.3f}  ({pooled_covered}/{pooled_roles})")
    print(f"pooled precision = {pooled_precision:.3f}")
    print(f"per-task recall  = " + "  ".join(
        f"{t}:{v:.2f}" for t, v in per_task_recall.items()))
    print(f"partition ok     = {partition_ok}/{len(sem_reps)}  "
          f"(clean NONE signal: {clean_none}/{len(sem_reps)})")
    print(f"GATE             = {summary['gate']}  (recall >= 0.9 required)")
    print(f"-> {out.name}")


if __name__ == "__main__":
    main()

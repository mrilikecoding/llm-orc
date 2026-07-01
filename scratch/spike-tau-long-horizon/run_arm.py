"""Spike τ arm runner — one N-file task under the CURRENT serve config on a
freshly-restarted (serve + ollama) stack, with per-turn latency extraction.

The agent sets `.llm-orc/config.yaml` (orchestrator profile + code_generation
cheap tier) for the arm before invoking this; the runner only restarts the stack
and drives one task. LABEL/FILES come from the environment:

  LABEL=ref FILES=5 PYTHONPATH=. uv run python scratch/spike-tau-long-horizon/run_arm.py

The 5-file gate is the validated σ temperature-library task (known to converge in
isolation). Higher rungs are generated from a fixed cross-referencing template.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path

from benchmarks.agentic_serving import runner, scorer
from benchmarks.agentic_serving.bench import _ServeProcess
from benchmarks.agentic_serving.model import Cell

LABEL = os.environ.get("LABEL", "arm")
FILES = int(os.environ.get("FILES", "5"))
PORT = 8770
OUT = Path("scratch/spike-tau-long-horizon") / f"{LABEL}-f{FILES}"
OUT.mkdir(parents=True, exist_ok=True)
SERVE_LOG = OUT / "serve.log"
TIMEOUT = float(os.environ.get("TIMEOUT", str(max(1800, FILES * 300))))

_SIGMA_5 = (
    "Build a small temperature-conversion library in this directory. Create these "
    "five files:\n"
    "1. converters.py with three functions: celsius_to_fahrenheit, "
    "fahrenheit_to_celsius, celsius_to_kelvin\n"
    "2. test_converters.py with unit tests for converters.py\n"
    "3. cli.py, a command-line tool that imports converters.py and converts a value "
    "given as command-line arguments\n"
    "4. test_cli.py with tests for cli.py\n"
    "5. README.md documenting how to use the CLI\n"
    "The tests must import the real module under test, cli.py must call the real "
    "functions in converters.py, and the README must document the real CLI usage."
)


def make_task(n: int) -> Cell:
    if n == 5:
        return Cell(
            name=f"{LABEL}-f5",
            horizon=5,
            complexity=2,
            prompt=_SIGMA_5,
            expected_deliverables=(
                "converters.py", "test_converters.py", "cli.py",
                "test_cli.py", "README.md",
            ),
        )
    # General rung: base.py + (n-2) step modules importing the prior one + main.py.
    steps = [f"step{i}.py" for i in range(1, n - 1)]
    deliverables = ("base.py", *steps, "main.py")
    lines = [
        f"Build a small Python pipeline in this directory. Create these {n} files, "
        "each calling the REAL names defined in the file before it (do not invent):",
        "1. base.py — a function start(x) that returns x + 1",
    ]
    for i, s in enumerate(steps, start=2):
        # file i is step{i-1}.py defining step{i-1}; its dependency is the PREVIOUS
        # file (base.py at i=2, else step{i-2}.py) — not itself. (2026-06-17:
        # off-by-one fix; the prior f"step{i-1}" made every file import itself,
        # producing incoherent siblings — a test artifact, not a framework limit.)
        # Call form is module-qualified ({prev}.{prev_fn}) so the instructed code is
        # runnable Python after `import {prev}` — a bare {prev_fn}(x) calls the module,
        # not the function (2026-06-17: second template fix, for the ADR-042 discharge).
        prev = "base" if i == 2 else f"step{i - 2}"
        prev_fn = "start" if i == 2 else f"step{i - 2}"
        fn = f"step{i - 1}"
        lines.append(
            f"{i}. {s} — a function {fn}(x) that imports {prev}.py and returns "
            f"{prev}.{prev_fn}(x) * 2"
        )
    last = "base" if not steps else f"step{len(steps)}"
    last_fn = "start" if not steps else f"step{len(steps)}"
    lines.append(
        f"{n}. main.py — imports {last}.py and prints {last}.{last_fn}(1). Write each "
        "file's exact contents and nothing else: no markdown fences, no prose."
    )
    return Cell(
        name=f"{LABEL}-f{n}", horizon=n, complexity=1,
        prompt="\n".join(lines), expected_deliverables=deliverables,
    )


def reboot_stack() -> None:
    subprocess.run(["osascript", "-e", 'quit app "Ollama"'], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "llama-server"], capture_output=True)
    time.sleep(3)
    subprocess.run(["open", "-a", "Ollama"], capture_output=True)
    for _ in range(90):
        if subprocess.run(["ollama", "list"], capture_output=True).returncode == 0:
            break
        time.sleep(1)
    for model in ("qwen3:14b", "qwen3:8b"):
        subprocess.run(["ollama", "run", model, "ok"], capture_output=True)


_TS = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d{3})")


def _ts(line: str) -> float | None:
    m = _TS.match(line)
    if not m:
        return None
    t = time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    return time.mktime(t) + int(m.group(2)) / 1000.0


def extract_latency(log_slice: str) -> dict[str, float | int | list[float]]:
    """Per-turn wall (delta between turn-decision lines) + coder dispatch durations.

    seat-share per turn ~= turn_wall - dispatch_duration. Reports the per-turn
    wall series, its median, and a simple first-to-last slope (degradation signal).
    """
    turn_ts: list[float] = []
    dispatch_durations: list[float] = []
    for line in log_slice.splitlines():
        if "turn decision:" in line:
            ts = _ts(line)
            if ts is not None:
                turn_ts.append(ts)
        elif "dispatch end:" in line:
            m = re.search(r"duration=([\d.]+)", line)
            if m:
                dispatch_durations.append(float(m.group(1)))
    walls = [turn_ts[i] - turn_ts[i - 1] for i in range(1, len(turn_ts))]
    walls_sorted = sorted(walls)
    median = walls_sorted[len(walls_sorted) // 2] if walls else 0.0
    slope = (walls[-1] - walls[0]) / max(1, len(walls) - 1) if len(walls) > 1 else 0.0
    return {
        "turns": len(turn_ts),
        "per_turn_walls": [round(w) for w in walls],
        "median_turn_wall": round(median),
        "slope_per_turn": round(slope, 1),
        "dispatch_durations": [round(d) for d in dispatch_durations],
    }


def main() -> None:
    cell = make_task(FILES)
    print(f"arm={LABEL} files={FILES} timeout={TIMEOUT:.0f}s", flush=True)
    reboot_stack()
    serve = _ServeProcess(PORT, SERVE_LOG)
    if not serve.start():
        raise SystemExit("serve failed to start")
    try:
        arts = runner.run_cell(
            cell, serve_port=PORT, output_dir=OUT, serve_log=SERVE_LOG,
            timeout_seconds=TIMEOUT,
        )
        rec = scorer.score(arts.workspace, arts.log_slice, arts.cell)
        lat = extract_latency(arts.log_slice)
        converged = rec.passed and not arts.timed_out
        metrics = (
            f"form={rec.form_valid} conv={rec.converged} "
            f"coh={rec.content_coherent} term={rec.terminated_clean}"
        )
        print(
            f"TAU {LABEL} files={FILES}: converged={converged} "
            f"produced={len(arts.produced)}/{FILES} wall={arts.wall_seconds:.0f}s "
            f"timed_out={arts.timed_out} turns={lat['turns']} "
            f"median_turn_wall={lat['median_turn_wall']}s "
            f"slope={lat['slope_per_turn']} {metrics}",
            flush=True,
        )
        print(f"  per_turn_walls={lat['per_turn_walls']}", flush=True)
        print(f"  dispatch_durations={lat['dispatch_durations']}", flush=True)
    finally:
        serve.stop()


if __name__ == "__main__":
    main()

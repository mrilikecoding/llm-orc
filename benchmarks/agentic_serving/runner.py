"""Drive ONE benchmark cell live against a running ``serve`` (the live unit).

Per ``docs/agentic-serving/benchmark-design.md`` §8 (robustness) + §11. Not
unit-tested — exercised live (it shells out to ``opencode`` + reads a running
``serve``'s log). The deterministic parts (slice capture, marker, config text)
are small and obviously-correct; the live subprocess call is the part that
needs ollama + opencode.

Boundaries (§10): this module drives one cell and returns its artifacts. It does
**not** start/stop ``serve`` (the CLI owns the serve lifecycle on a dedicated
port — §8 dev-traffic isolation) and does **not** score (the CLI calls
``scorer.score`` on the returned artifacts).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from benchmarks.agentic_serving.model import Cell

# Default wall-clock budget for the degradation smoke (§8 P2-E): a known
# 2-deliverable task must finish under this or the environment is degraded.
SMOKE_BUDGET_SECONDS = 300.0

# The model the per-workspace opencode.json points at — the served agentic
# surface (matches the scratch runners' `-m llmorc/agentic`).
_OPENCODE_MODEL = "llmorc/agentic"


@dataclass(frozen=True)
class CellArtifacts:
    """Everything ``scorer.score`` needs for one cell run, plus run diagnostics.

    ``workspace`` + ``log_slice`` are the scorer's two inputs; ``produced`` is a
    convenience snapshot of the files the session left (the scorer recomputes it
    from the workspace). ``wall_seconds`` and ``timed_out`` feed the degradation
    protocol (§8 P2-E) — the CLI tags a cell ``degraded`` from these.
    """

    cell: Cell
    workspace: Path
    log_slice: str
    produced: tuple[str, ...]
    session_marker: str
    wall_seconds: float
    returncode: int
    timed_out: bool


def run_cell(
    cell: Cell,
    *,
    serve_port: int,
    output_dir: Path,
    serve_log: Path,
    timeout_seconds: float = SMOKE_BUDGET_SECONDS,
) -> CellArtifacts:
    """Run one cell live and return its artifacts (§8 + §11).

    Drives ``opencode`` against the ``serve`` on ``serve_port`` in a fresh
    per-cell workspace under ``output_dir``, prepends a unique session marker
    (§8 P2-D), and captures only this run's serve-log slice (§8 — never the
    cumulative log). ``serve_log`` is the file the CLI's dedicated ``serve``
    writes to.
    """
    workspace = _fresh_workspace(output_dir, cell.name)
    _write_opencode_config(workspace, serve_port)
    marker = _session_marker()
    prompt = _marked_prompt(marker, cell.prompt)

    start_line = _log_line_count(serve_log)
    began = time.monotonic()
    returncode, timed_out = _run_opencode(workspace, prompt, timeout_seconds)
    wall = time.monotonic() - began
    log_slice = _log_slice_from(serve_log, start_line)

    return CellArtifacts(
        cell=cell,
        workspace=workspace,
        log_slice=log_slice,
        produced=_produced_files(workspace),
        session_marker=marker,
        wall_seconds=wall,
        returncode=returncode,
        timed_out=timed_out,
    )


def degradation_smoke(
    *,
    serve_port: int,
    output_dir: Path,
    serve_log: Path,
    budget_seconds: float = SMOKE_BUDGET_SECONDS,
) -> tuple[bool, CellArtifacts]:
    """Run a known 2-deliverable task; report whether the env is degraded (§8 P2-E).

    Returns ``(degraded, artifacts)``: ``degraded`` is ``True`` if the smoke
    timed out or exceeded ``budget_seconds`` (the σ marathon-degradation guard).
    The CLI aborts the grid when this is ``True`` (do not grind a degraded run).
    """
    artifacts = run_cell(
        _SMOKE_CELL,
        serve_port=serve_port,
        output_dir=output_dir,
        serve_log=serve_log,
        timeout_seconds=budget_seconds,
    )
    degraded = artifacts.timed_out or artifacts.wall_seconds > budget_seconds
    return degraded, artifacts


# A fixed, easy 2-deliverable task for the pre-flight smoke (§8 / §11). One
# dependent file so it exercises a real generation + delegation turn, but small
# enough that any non-degraded cheap-local stack closes it well under budget.
_SMOKE_CELL = Cell(
    name="smoke",
    horizon=2,
    complexity=1,
    prompt=(
        "Create two files. First, greet.py with a function hello(name) that "
        "returns 'Hello, ' + name. Second, run.py which imports hello from greet "
        "and prints hello('world'). Write each file's exact contents and nothing "
        "else — no markdown fences and no prose. run.py must call the real hello "
        "defined in greet.py."
    ),
    expected_deliverables=("greet.py", "run.py"),
)


# --- Workspace + config ------------------------------------------------------


def _fresh_workspace(output_dir: Path, cell_name: str) -> Path:
    """A clean per-cell workspace (§8 — never reuse a dirty workspace)."""
    ws = output_dir / "workspaces" / f"{cell_name}-{uuid.uuid4().hex[:8]}"
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)
    return ws


def _write_opencode_config(workspace: Path, serve_port: int) -> None:
    """Per-workspace ``opencode.json`` pointing at the dedicated benchmark serve.

    Mirrors the scratch runners' config: an OpenAI-compatible provider named
    ``llmorc`` exposing one model ``agentic`` at the serve's ``/v1`` endpoint.
    """
    config = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "llmorc": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "llm-orc",
                "options": {
                    "baseURL": f"http://127.0.0.1:{serve_port}/v1",
                    "apiKey": "sk-llmorc-local-dummy",
                },
                "models": {"agentic": {"name": "agentic"}},
            }
        },
    }
    (workspace / "opencode.json").write_text(json.dumps(config, indent=2) + "\n")


# --- Session marker (§8 P2-D) ------------------------------------------------


def _session_marker() -> str:
    """A unique per-session marker → a unique ``serve`` session id (§8 P2-D).

    The Spike η ``sha256(first-message)`` collision bled action records across
    identical-prompt runs; a unique marker on the first message closes it.
    """
    return f"BENCHMARK RUN {uuid.uuid4()}"


def _marked_prompt(marker: str, prompt: str) -> str:
    """Prepend the unique marker as an inert first line the task ignores."""
    return (
        f"[{marker}]\n"
        "(Internal benchmark run marker on the line above. It is not part of "
        "the task; ignore it.)\n\n"
        f"{prompt}"
    )


# --- opencode invocation -----------------------------------------------------


def _run_opencode(
    workspace: Path, prompt: str, timeout_seconds: float
) -> tuple[int, bool]:
    """Run ``opencode run`` in ``workspace``; return ``(returncode, timed_out)``.

    Stdout/stderr are captured to ``run.out`` / ``run.err`` beside the workspace
    (the client stream is where ``[dispatch failed: ...]`` refusal text lands —
    useful for live triage; the scorer reads the serve-log slice, not these).
    """
    cmd = [
        "opencode",
        "run",
        "-m",
        _OPENCODE_MODEL,
        "--format",
        "json",
        "--dir",
        str(workspace),
        prompt,
    ]
    out_path = workspace.parent / f"{workspace.name}.out"
    err_path = workspace.parent / f"{workspace.name}.err"
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        out_path.write_text(_decode(exc.stdout))
        err_path.write_text(_decode(exc.stderr))
        return -1, True
    out_path.write_text(completed.stdout)
    err_path.write_text(completed.stderr)
    return completed.returncode, False


def _decode(stream: str | bytes | None) -> str:
    """Coerce a possibly-bytes captured stream to text (TimeoutExpired buffers)."""
    if stream is None:
        return ""
    if isinstance(stream, bytes):
        return stream.decode("utf-8", errors="replace")
    return stream


# --- Serve-log slicing (§8) --------------------------------------------------


def _log_line_count(serve_log: Path) -> int:
    """Lines in the serve log right now — the slice start (§8)."""
    if not serve_log.exists():
        return 0
    with serve_log.open(encoding="utf-8", errors="replace") as fh:
        return sum(1 for _ in fh)


def _log_slice_from(serve_log: Path, start_line: int) -> str:
    """This run's serve-log slice — lines after ``start_line`` (§8).

    Never the cumulative log: capturing ``tail -n +<start>`` per session is the
    fix for the cumulative-vs-slice confusion that produced false readings in
    the escalation experiment.
    """
    if not serve_log.exists():
        return ""
    with serve_log.open(encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    return "".join(lines[start_line:])


def _produced_files(workspace: Path) -> tuple[str, ...]:
    """Files the session produced (excludes the opencode config + dotfiles)."""
    return tuple(
        sorted(
            p.name
            for p in workspace.iterdir()
            if p.is_file() and p.name != "opencode.json" and not p.name.startswith(".")
        )
    )

"""Unit + wiring tests for the write-tests shape (#98).

A test-primary turn's deliverable IS the test file, executed against the
materialized workspace alone: one test source, no code_writer, so the
shadowed-composite wrong-accept (gate validated a composite while the
shipped test file carried a broken test) is structurally impossible. The
shipped artifact is the executed artifact.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

from llm_orc.core.config.ensemble_config import EnsembleLoader, _find_ensemble_in_dirs
from llm_orc.core.execution.executor_factory import ExecutorFactory
from llm_orc.core.serving.shape_catalog import shape_catalog
from llm_orc.schemas.agent_config import LoopAgentConfig, ScriptAgentConfig

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO / ".llm-orc" / "scripts" / "agentic_serving"
ENSEMBLES = REPO / ".llm-orc" / "ensembles"
GATHER = SCRIPTS / "tests_gather.py"
ENVELOPE = SCRIPTS / "tests_envelope.py"

STORE_MODULE = (
    "class TaskStore:\n"
    "    def __init__(self):\n"
    "        self.tasks = {}\n"
    "    def add(self, task_id, title):\n"
    "        self.tasks[task_id] = title\n"
    "    def list_tasks(self):\n"
    "        return list(self.tasks.values())\n"
)
TESTS = (
    "def test_add_and_list():\n"
    "    store = TaskStore()\n"
    "    store.add(1, 'a')\n"
    "    assert store.list_tasks() == ['a']\n"
)
CONTEXT_TASK = (
    "Conversation so far:\n"
    "assistant: [wrote storage.py]\n"
    f"{STORE_MODULE}"
    "\nCurrent request: Write tests for storage.py"
)


def _run(script: Path, payload: dict[str, Any]) -> dict[str, Any]:
    out = subprocess.run(
        [sys.executable, str(script)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    result: dict[str, Any] = json.loads(out)
    return result


def _sub_ensemble_response(terminal_text: str) -> str:
    return json.dumps(
        {
            "ensemble": "test-writer",
            "status": "completed",
            "results": {"out": {"response": terminal_text, "status": "success"}},
        }
    )


# --- tests_gather: deliverable tests + workspace, no code, nothing shadows ---


def test_gather_assembles_tests_as_the_deliverable_with_empty_code() -> None:
    out = _run(
        GATHER,
        {
            "input_data": CONTEXT_TASK,
            "dependencies": {
                "test_writer": {"response": _sub_ensemble_response(TESTS)},
            },
        },
    )
    assert out["code"] == ""
    assert out["target_file"] == ""
    assert "def test_add_and_list" in out["tests"]
    assert out["workspace"] == {"storage.py": STORE_MODULE.strip()}
    assert out["requirement"] == "Write tests for storage.py"


def test_gather_injects_missing_workspace_imports_into_the_tests() -> None:
    """The test-writer seat is prompted not to import; the workspace import
    is injected deterministically (the shipped accept_gather behavior,
    reused via sibling import)."""
    out = _run(
        GATHER,
        {
            "input_data": CONTEXT_TASK,
            "dependencies": {
                "test_writer": {"response": _sub_ensemble_response(TESTS)},
            },
        },
    )
    assert "from storage import TaskStore" in out["tests"]


# --- tests_envelope: primary = the executor-echoed (validated) tests ---


def _envelope(
    verdict: dict[str, Any], executor_extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    executor: dict[str, Any] = {
        "tests_pass": True,
        "tests": TESTS,
        "n_tests": 1,
        "report": "all passed",
    }
    executor.update(executor_extra or {})
    return _run(
        ENVELOPE,
        {
            "input_data": "Write tests for storage.py",
            "dependencies": {
                "executor": {"response": json.dumps(executor)},
                "accept_gate": {"response": json.dumps(verdict)},
            },
        },
    )


def test_envelope_ships_the_executed_tests_as_primary() -> None:
    env = _envelope(
        {"accept": True, "tests_pass": True, "tests_adequate": True, "reason": "ok"}
    )
    assert env["status"] == "success"
    assert env["primary"] == TESTS
    assert env["artifacts"][0]["content"] == TESTS
    assert env["diagnostics"]["accept"] is True
    assert "retry_input" not in env["diagnostics"]


def test_envelope_reject_composes_a_fresh_retry_input() -> None:
    env = _envelope(
        {"accept": False, "tests_pass": False, "reason": "tests did not pass"},
        executor_extra={
            "tests_pass": False,
            "report": "test_add_and_list: AttributeError('list')",
        },
    )
    diag = env["diagnostics"]
    assert diag["accept"] is False
    retry = diag["retry_input"]
    assert "Write tests for storage.py" in retry
    assert "AttributeError" in retry
    assert "[HELD TESTS" not in retry  # no held mode: tests are the moving side


# --- wiring: catalog, loop, round shape ---


def test_catalog_maps_the_tests_seat_intent_to_the_write_tests_shape() -> None:
    catalog = shape_catalog(ENSEMBLES / "agentic-serving")
    assert catalog["tests-seat"] == "write-tests"


def test_write_tests_wraps_the_round_in_the_bounded_retry_loop() -> None:
    config = _find_ensemble_in_dirs("write-tests", [str(ENSEMBLES)])
    assert config is not None
    round_agent = next(a for a in config.agents if a.name == "round")
    assert isinstance(round_agent, LoopAgentConfig)
    assert round_agent.loop.body == "write-tests-round"
    assert round_agent.loop.max_iterations == 2


def test_the_round_has_one_test_source_and_the_deterministic_verifiers() -> None:
    config = _find_ensemble_in_dirs("write-tests-round", [str(ENSEMBLES)])
    assert config is not None
    names = {a.name for a in config.agents}
    assert "code_writer" not in names
    assert {"test_writer", "gather", "executor", "judge", "accept_gate"} <= names
    judge = next(a for a in config.agents if a.name == "judge")
    assert isinstance(judge, ScriptAgentConfig)
    assert judge.script == "scripts/agentic_serving/adequacy_check.py"


# --- hermetic end-to-end: the real round graph through the real engine ---

# single line, no backslashes or nested quotes: the script resolver treats
# '\' as a file path, and chr(97) dodges quoting inside the echo argument
_ECHO_TEST_WRITER = (
    "name: test-writer\n"
    "description: deterministic echo seat for the write-tests e2e\n"
    "agents:\n"
    "  - name: out\n"
    "    script: \"echo 'def test_add_and_list(): store = TaskStore();"
    " store.add(1, chr(97)); assert store.list_tasks() == [chr(97)]'\"\n"
)


@pytest.mark.asyncio
async def test_write_tests_round_accepts_against_the_workspace_end_to_end() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj = Path(td)
        ens = proj / "ensembles"
        ens.mkdir()
        scripts = proj / "scripts" / "agentic_serving"
        scripts.parent.mkdir()
        shutil.copytree(SCRIPTS, scripts)
        shutil.copy(
            ENSEMBLES / "agentic-serving" / "write-tests-round.yaml",
            ens / "write-tests-round.yaml",
        )
        (ens / "test-writer.yaml").write_text(_ECHO_TEST_WRITER)

        config = EnsembleLoader().load_from_file(str(ens / "write-tests-round.yaml"))
        executor = ExecutorFactory.create_root_executor(project_dir=proj)
        mock_artifact = Mock()
        mock_artifact.save_execution_results = Mock()
        with patch.object(executor, "_artifact_manager", mock_artifact):
            result = await executor.execute(config, CONTEXT_TASK)

    envelope = json.loads(result["results"]["envelope"]["response"])
    diag = envelope["diagnostics"]
    assert diag["tests_pass"] is True, diag
    assert diag["tests_adequate"] is True
    assert diag["accept"] is True
    assert "def test_add_and_list" in envelope["primary"]
    assert "class TaskStore" not in envelope["primary"]  # tests only, no composite

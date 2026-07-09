"""Acceptance + wiring tests for the Cycle-8 declarative Serving Ensemble path.

WP-A8: the per-turn handler is ONE declarative ensemble (classify -> seat ->
marshal), invoked by the L3 Serving Layer through
:class:`ServingEnsembleCaller` behind the ``_ChatCompletionsCaller`` Protocol
(``docs/serving.md`` §Cycle 8; scenarios.md "Per-Turn
Serving Handler"; ADR-046 §1).

These tests drive the REAL serving ensemble through the HTTP boundary with a
DETERMINISTIC echo seat (no model) so they are hermetic and fast, matching the
existing ``_FakeSeatFiller`` pattern in ``test_api_v1_chat_completions.py``.
The real model + a real ``opencode run`` build turn is the grounding step run
after green (the standing "don't build in a vacuum" directive).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from llm_orc.web.api import v1_chat_completions
from llm_orc.web.server import create_app
from llm_orc.web.serving.serving_ensemble_caller import ServingEnsembleCaller

REPO = Path(__file__).resolve().parents[3]
REAL_LLM_ORC = REPO / ".llm-orc"
REAL_SERVING_SCRIPTS = REAL_LLM_ORC / "scripts" / "agentic_serving"
REAL_AGENTIC_SERVING = REAL_LLM_ORC / "ensembles" / "agentic-serving"
REAL_SERVING_ENSEMBLE = REAL_AGENTIC_SERVING / "serving.yaml"
REAL_CODE_SEAT = REAL_AGENTIC_SERVING / "code-seat.yaml"

_WRITE_TOOL = {
    "type": "function",
    "function": {
        "name": "write",
        "description": "Write a file to disk.",
        "parameters": {
            "type": "object",
            "properties": {
                "filePath": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["filePath", "content"],
        },
    },
}

# A deterministic code_generation flow: one echo node emitting clean python, so
# the build turn runs classify -> seat(code-seat -> code-generator -> envelope)
# -> shape -> form-gate -> emit with no model tokens. The code seat wraps this
# via a static `ensemble: code-generator` reference. (Extraction of code from
# chatty synthesizer prose is locked separately by the emit_envelope unit test;
# the real-model prose path is covered by opencode grounding.)
# A single-line echo (no backslashes: the script resolver treats a value
# containing '\' as a file path, and inline scripts are a test-only artifact —
# the real seat runs model nodes).
_ECHO_CODE_GENERATOR = (
    "name: code-generator\n"
    "description: deterministic echo flow for hermetic serving tests\n"
    "agents:\n"
    "  - name: out\n"
    "    script: \"echo 'def add(a, b): return a + b'\"\n"
)

# A deterministic explain seat: one echo node emitting prose (no envelope), so
# the explain turn runs classify -> seat(explainer) -> shape -> form-gate ->
# emit with no model tokens. shape degrades gracefully to the seat terminal for
# a raw-prose seat; the real model prose path is covered by opencode grounding.
_ECHO_EXPLAINER = (
    "name: explainer\n"
    "description: deterministic echo explain seat for hermetic serving tests\n"
    "agents:\n"
    "  - name: out\n"
    "    script: \"echo 'foo.py defines add, which returns a plus b.'\"\n"
)


@pytest.fixture
def serving_project(tmp_path: Path) -> Path:
    """A hermetic project dir: the real serving ensemble + scripts, with a
    deterministic echo ``code-generator`` seat shadowing the model-backed one.
    """
    ensembles = tmp_path / "ensembles"
    ensembles.mkdir()
    scripts = tmp_path / "scripts" / "agentic_serving"
    scripts.parent.mkdir()
    shutil.copytree(REAL_SERVING_SCRIPTS, scripts)
    shutil.copy(REAL_SERVING_ENSEMBLE, ensembles / "serving.yaml")
    shutil.copy(REAL_CODE_SEAT, ensembles / "code-seat.yaml")
    (ensembles / "code-generator.yaml").write_text(_ECHO_CODE_GENERATOR)
    # The explain seat dispatch target — a top-level entry, since dispatch
    # discovery is non-recursive (WP-A8 discovery note b).
    (ensembles / "explainer.yaml").write_text(_ECHO_EXPLAINER)
    return tmp_path


@pytest.fixture
def serving_client(
    serving_project: Path, monkeypatch: pytest.MonkeyPatch
) -> TestClient:
    """A TestClient whose endpoint invokes the Serving Ensemble over the
    hermetic project dir, re-pointing the ``get_serving_ensemble_caller``
    factory the Cycle-8 endpoint resolves.
    """

    def _caller() -> ServingEnsembleCaller:
        return ServingEnsembleCaller(project_dir=serving_project, ensemble="serving")

    monkeypatch.setattr(
        v1_chat_completions, "get_serving_ensemble_caller", _caller, raising=False
    )
    return TestClient(create_app())


def test_build_turn_writes_deliverable_via_tool_call(
    serving_client: TestClient,
) -> None:
    """A build turn routes classify -> capability seat -> marshal and writes a
    deliverable (scenarios.md, ADR-046 §1). At the API boundary the response is
    a ``write`` tool_call carrying the produced file.
    """
    resp = serving_client.post(
        "/v1/chat/completions",
        json={
            "model": "ensemble-agent",
            "messages": [
                {
                    "role": "user",
                    "content": "write a function that adds two numbers in add.py",
                }
            ],
            "tools": [_WRITE_TOOL],
        },
    )

    assert resp.status_code == 200
    choice = resp.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    tool_calls = choice["message"]["tool_calls"]
    assert len(tool_calls) == 1
    call = tool_calls[0]
    assert call["function"]["name"] == "write"
    args = json.loads(call["function"]["arguments"])
    # classify extracts the destination from the turn (defect 1 fix)
    assert args["filePath"] == "add.py"
    # shape reads the deliverable faithfully from the seat's envelope (defect 2)
    assert "def add" in args["content"]
    assert args["content"].startswith("def add")


def test_multi_turn_history_serves_the_latest_user_message(
    serving_client: TestClient,
) -> None:
    """A real client (OpenCode) sends the FULL history every turn; the serve
    must handle the latest user message, not re-run turn 1 (Cycle-8 PLAY
    field note #1: every turn re-processed "hello").
    """
    resp = serving_client.post(
        "/v1/chat/completions",
        json={
            "model": "ensemble-agent",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "Hi! How can I help?"},
                {
                    "role": "user",
                    "content": "write a function that adds two numbers in add.py",
                },
            ],
            "tools": [_WRITE_TOOL],
        },
    )

    assert resp.status_code == 200
    choice = resp.json()["choices"][0]
    # the latest turn is a build turn: the serve must emit the write tool_call
    # for add.py, not respond to turn 1's "hello"
    assert choice["finish_reason"] == "tool_calls"
    args = json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"])
    assert args["filePath"] == "add.py"


def test_tool_result_callback_is_acknowledged_not_rerun(
    serving_client: TestClient,
) -> None:
    """After the serve emits a write tool_call and the client performs it, the
    client calls back with the tool result appended. That call is a
    CONTINUATION of the same turn — the serve must acknowledge and finish,
    not re-run the whole build pipeline (battery finding 2026-07-08: the
    second pass re-ran the gated build and returned a spurious reject after
    the file was already written).
    """
    resp = serving_client.post(
        "/v1/chat/completions",
        json={
            "model": "ensemble-agent",
            "messages": [
                {
                    "role": "user",
                    "content": "write a function that adds two numbers in add.py",
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {
                                "name": "write",
                                "arguments": '{"filePath": "add.py", "content": "..."}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_abc123",
                    "content": "Wrote file successfully.",
                },
            ],
            "tools": [_WRITE_TOOL],
        },
    )

    assert resp.status_code == 200
    choice = resp.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert not choice["message"].get("tool_calls")
    assert choice["message"]["content"]


def test_quoted_message_content_is_normalized_before_classify(
    serving_client: TestClient,
) -> None:
    """``opencode run -c`` (continued sessions) delivers the message content
    wrapped in literal double quotes; the anchored interrogative routing then
    misses and an explain question runs the gated build (battery finding
    2026-07-08). The serve strips one symmetric surrounding quote pair.
    """
    resp = serving_client.post(
        "/v1/chat/completions",
        json={
            "model": "ensemble-agent",
            "messages": [
                {
                    "role": "user",
                    "content": '"What approach does foo.py take, briefly?"',
                },
            ],
            "tools": [_WRITE_TOOL],
        },
    )

    assert resp.status_code == 200
    choice = resp.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert not choice["message"].get("tool_calls")
    assert "foo.py" in choice["message"]["content"]


def test_prior_turns_reach_the_seat_via_dispatch_input(
    serving_project: Path, serving_client: TestClient
) -> None:
    """Rung-1 conversation memory: prior turns render into the context and
    reach the seat behind the 'Current request:' marker, so generation seats
    can resolve referents like "it" (memory design §Rung 1). The explainer is
    swapped for an input-echo script so the response reveals the seat input.
    """
    (serving_project / "ensembles" / "explainer.yaml").write_text(
        "name: explainer\n"
        "description: input-echo explain seat for the context thread-through\n"
        "agents:\n"
        "  - name: out\n"
        '    script: "cat"\n'
    )
    resp = serving_client.post(
        "/v1/chat/completions",
        json={
            "model": "ensemble-agent",
            "messages": [
                {"role": "user", "content": "write is_even in even.py"},
                {"role": "assistant", "content": "Done - is_even is in even.py."},
                {"role": "user", "content": "explain what foo.py does"},
            ],
            "tools": [_WRITE_TOOL],
        },
    )

    assert resp.status_code == 200
    content = resp.json()["choices"][0]["message"]["content"]
    assert "user: write is_even in even.py" in content
    assert "Current request: explain what foo.py does" in content


def test_explain_turn_returns_prose_not_a_tool_call(
    serving_client: TestClient,
) -> None:
    """An explain turn routes through the SAME skeleton and returns prose, not a
    file (scenarios.md "An explain turn routes through the same skeleton and
    returns prose, not a file"; ADR-046 §1). classify routes to the explain seat
    with build=false; marshal returns a prose finish with no file-writing
    tool_call.
    """
    resp = serving_client.post(
        "/v1/chat/completions",
        json={
            "model": "ensemble-agent",
            "messages": [{"role": "user", "content": "explain what foo.py does"}],
            "tools": [_WRITE_TOOL],
        },
    )

    assert resp.status_code == 200
    choice = resp.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert not choice["message"].get("tool_calls")
    content = choice["message"]["content"]
    assert content
    assert "foo.py" in content

"""Unit tests for the serving ``shape`` node (WP-A8, scenario 5).

shape reads the deliverable CONTENT from the seat's ADR-024 envelope and the
DESTINATION from classify, then produces the faithful deliverable. When the seat
did not envelope (a non-build explain seat returning raw prose), shape degrades
to the raw terminal text (scenarios.md "the marshal node consumes the seat's
real common I/O envelope"; ADR-046 §1).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[3]
SHAPE = REPO / ".llm-orc" / "scripts" / "agentic_serving" / "shape.py"


def _shape_raw(deps: dict[str, Any]) -> dict[str, Any]:
    out = subprocess.run(
        [sys.executable, str(SHAPE)],
        input=json.dumps({"dependencies": deps}),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    result: dict[str, Any] = json.loads(out)
    return result


def _shape(
    classify_decision: dict[str, Any], seat_child_result: dict[str, Any]
) -> dict[str, Any]:
    return _shape_raw(
        {
            "classify": {"response": json.dumps(classify_decision)},
            "seat": {"response": json.dumps(seat_child_result)},
        }
    )


def test_shape_reads_deliverable_from_envelope_and_destination_from_classify() -> None:
    code = "def add(a, b):\n    return a + b"
    envelope = {
        "status": "success",
        "primary": code,
        "artifacts": [
            {"content": code, "content_type": "text/x-python", "summary": "add"}
        ],
    }
    shaped = _shape(
        {
            "target": "code-seat",
            "build": True,
            "file": "add.py",
            "kind": "python_module",
        },
        {
            "results": {
                "generate": {"response": "..."},
                "envelope": {"response": json.dumps(envelope)},
            }
        },
    )
    assert shaped["build"] is True
    assert shaped["file"] == "add.py"
    assert shaped["content"] == code


def test_shape_degrades_to_raw_seat_terminal_when_no_envelope() -> None:
    shaped = _shape(
        {"target": "explainer", "build": False, "kind": "explanation"},
        {"results": {"out": {"response": "It adds two numbers."}}},
    )
    assert shaped["build"] is False
    assert "adds two numbers" in shaped["content"]


def test_shape_carries_the_accept_verdict_from_the_build_gated_envelope() -> None:
    code = "def add(a, b):\n    return a + b"
    envelope = {
        "status": "success",
        "primary": code,
        "artifacts": [{"content": code, "content_type": "text/x-python"}],
        "diagnostics": {
            "ensemble": "build-gated",
            "accept": False,
            "accept_reason": "tests inadequate",
        },
    }
    shaped = _shape(
        {
            "target": "code-seat",
            "build": True,
            "file": "add.py",
            "kind": "python_module",
        },
        {"results": {"envelope": {"response": json.dumps(envelope)}}},
    )
    assert shaped["content"] == code
    assert shaped["accept"] is False
    assert "inadequate" in shaped["accept_reason"]


def test_shape_has_no_verdict_for_an_ungated_seat_envelope() -> None:
    # A code-seat / explainer envelope carries no accept diagnostics -> None.
    envelope = {
        "status": "success",
        "primary": "x = 1",
        "artifacts": [{"content": "x = 1"}],
    }
    shaped = _shape(
        {"target": "code-seat", "build": True, "file": "a.py"},
        {"results": {"envelope": {"response": json.dumps(envelope)}}},
    )
    assert shaped["accept"] is None


def test_unreadable_routing_decision_fails_closed_to_refusal() -> None:
    """An empty/unparseable routing decision must not default the turn onto
    the build path NOR pass the dead seat's prose through — it refuses
    (#152 strengthens the old prose fail-closed to an honest refusal)."""
    out = _shape({}, {"results": {"out": {"response": "x = 1", "status": "success"}}})
    assert out["build"] is False
    assert out["routing_failed"]


# The engine's crash wrap (script_agent.py:161-168): a serving node that
# exits nonzero RETURNS this envelope as a normal response, so the ensemble
# records the node status=success and the envelope rides into dependents'
# deps looking like a readable result. Captured live 2026-08-13 (#144 gate
# misfire, docs/plans/2026-08-13-144-live-gate/).
_FAILURE_ENVELOPE = {
    "success": False,
    "error": "Script failed with exit code 1",
    "stderr": "ModuleNotFoundError: No module named 'llm_orc'",
}


def test_failed_resolve_dep_fails_closed_to_a_routing_refusal() -> None:
    """The #152 capture: the old truthiness fallback picked the failed
    resolve dep (it exists), parsed the failure envelope to a non-empty
    dict with no "kind" key, and the build default landed True — a
    crashed ROUTING node degraded into a junk empty solution.py write.
    No readable routing decision must refuse instead."""
    shaped = _shape_raw(
        {
            "resolve": {"response": json.dumps(_FAILURE_ENVELOPE)},
            "seat": {"response": ""},
        }
    )
    assert shaped["build"] is False
    # The reason carries what happened to the node, normalized (#168): this
    # envelope's wording and the blanket except's "returned non-zero exit
    # status 1" are the same FACT from two producers, and the client sees
    # one vocabulary for it. Asserting the residue rather than the
    # producer's literal keeps this pin on its own subject — that a crashed
    # routing node refuses with a reason, not that it echoes engine text.
    assert "status 1" in shaped["routing_failed"]


def test_empty_target_decision_is_unreadable_and_refuses() -> None:
    """target must be NON-EMPTY — key presence is not enough (#152
    pre-flight finding 1): a crashed classify laundered through a healthy
    resolve emits target "" (resolve's else branch over the failure
    envelope), as does an out-of-set decider vote. Both are routing
    failures, and the seat that dispatched on "" is dead either way."""
    shaped = _shape_raw(
        {
            "resolve": {
                "response": json.dumps(
                    {"target": "", "kind": "", "file": "solution.py", "build": False}
                )
            },
            "seat": {"response": ""},
        }
    )
    assert shaped["build"] is False
    assert shaped["routing_failed"]


def test_decision_with_target_but_no_build_or_kind_refuses() -> None:
    """The retained build default (kind != "explanation" -> True) must
    never be what stands between a target-only decision and a build
    (#152 pre-flight finding 3): a decision carrying target but neither
    build nor kind is outside the producers' contract and refuses."""
    shaped = _shape_raw(
        {
            "resolve": {"response": json.dumps({"target": "code-seat"})},
            "seat": {"response": "x = 1"},
        }
    )
    assert shaped["build"] is False
    assert shaped["routing_failed"]


def test_failed_resolve_with_readable_classify_routes_via_classify() -> None:
    """The source-preference chain stays alive for the unit harness and
    any pre-resolve wiring: a readable classify behind an unreadable
    resolve routes the turn normally (live serving.yaml never wires
    classify into shape's deps, so this is not an absorbing live
    fallback — see the #152 design's deviation section)."""
    shaped = _shape_raw(
        {
            "resolve": {"response": json.dumps(_FAILURE_ENVELOPE)},
            "classify": {
                "response": json.dumps(
                    {"target": "explainer", "kind": "explanation", "build": False}
                )
            },
            "seat": {
                "response": json.dumps(
                    {"results": {"out": {"response": "It adds two numbers."}}}
                )
            },
        }
    )
    assert shaped["build"] is False
    assert shaped["routing_failed"] == ""
    assert "adds two numbers" in shaped["content"]


def test_shape_passes_read_fields_from_the_routing_decision() -> None:
    shaped = _shape(
        {
            "target": "need-files",
            "kind": "need_files",
            "file": "test_storage.py",
            "build": False,
            "needs_files": ["storage.py"],
            "read_failed": "",
        },
        {"results": {"out": {"response": "Requesting client files."}}},
    )
    assert shaped["needs_files"] == ["storage.py"]
    assert shaped["read_failed"] == ""


def test_shape_passes_needs_run_from_the_routing_decision() -> None:
    shaped = _shape(
        {
            "target": "need-run",
            "kind": "need_run",
            "file": "solution.py",
            "build": False,
            "needs_files": [],
            "read_failed": "",
            "needs_run": "pytest -q",
        },
        {"status": "ok", "primary": "Requesting a client test run."},
    )
    assert shaped["needs_run"] == "pytest -q"


def test_shape_passes_glob_fields_from_the_routing_decision() -> None:
    shaped = _shape(
        {
            "target": "need-glob",
            "kind": "need_glob",
            "file": "solution.py",
            "build": False,
            "needs_files": [],
            "read_failed": "",
            "needs_run": "",
            "needs_glob": "storage",
            "glob_failed": "",
        },
        {"status": "ok", "primary": "Requesting a workspace listing."},
    )
    assert shaped["needs_glob"] == "storage"
    assert shaped["glob_failed"] == ""


def test_shape_passes_not_grounded_from_the_routing_decision() -> None:
    shaped = _shape(
        {
            "target": "not-grounded",
            "kind": "not_grounded",
            "file": "solution.py",
            "build": False,
            "not_grounded": "todo.py",
        },
        {"status": "ok", "primary": "Not grounded in this session."},
    )
    assert shaped["not_grounded"] == "todo.py"
    assert shaped["build"] is False


def test_shape_passes_is_build_ask_from_the_routing_decision() -> None:
    # Review round 2 new blocker 2.
    shaped = _shape(
        {
            "target": "need-files",
            "kind": "need_files",
            "file": "test_storage.py",
            "build": False,
            "is_build_ask": True,
        },
        {"results": {"out": {"response": "Requesting client files."}}},
    )
    assert shaped["is_build_ask"] is True


def test_shape_defaults_is_build_ask_false_when_absent() -> None:
    shaped = _shape(
        {"target": "explainer", "kind": "explanation", "file": "solution.py"},
        {"status": "ok", "primary": "It adds two numbers."},
    )
    assert shaped["is_build_ask"] is False


def test_needs_self_files_passes_through_shape() -> None:
    # #144 serve-native self-reference: rides the routing decision.
    shaped = _shape(
        {
            "target": "need-self-files",
            "kind": "need_self_files",
            "file": "solution.py",
            "build": False,
            "needs_self_files": [".llm-orc/scripts/agentic_serving/resolve.py"],
        },
        {"status": "ok", "primary": "Requesting self files."},
    )
    assert shaped["needs_self_files"] == [".llm-orc/scripts/agentic_serving/resolve.py"]


def test_non_string_or_whitespace_target_refuses() -> None:
    """Review finding 8: the gate demands a non-empty STRING target — a
    drifted producer emitting a number, list, or whitespace target must
    refuse, never ride the build default past the gate."""
    for target in (5, ["code-seat"], "   "):
        shaped = _shape_raw(
            {
                "resolve": {
                    "response": json.dumps(
                        {"target": target, "kind": "python_module", "build": True}
                    )
                },
                "seat": {"response": "x = 1"},
            }
        )
        assert shaped["build"] is False, f"target={target!r}"
        assert shaped["routing_failed"], f"target={target!r}"


# --- #155 Arc A: shape recognises a dead seat_contract ---------------------

_ENGINE_WRAP = (
    '{"success": false, "data": null, "error": "Schema JSON execution failed: '
    'Command \'[...]\' returned non-zero exit status 1.", "agent_requests": []}'
)

_DECISION = {"target": "code-seat", "build": True, "kind": "build"}
_SEAT = {"status": "success", "primary": "x = 1"}


def test_a_crashed_seat_contract_is_reported() -> None:
    """#155: `_seat_verdict` returned (None, "") for anything without a
    `seat_admitted` key, so a DEAD seat contract read downstream as "no
    per-seat gate ran" — indistinguishable from a route that has no gate.

    Reported as `seat_gate_failed` rather than as a pipeline read failure,
    because emit must consume it on the BUILD branch only: the seat
    contract is a vacuous echo on every other route, so refusing there
    would kill turns its verdict cannot affect.
    """
    shaped = _shape_raw(
        {
            "classify": {"response": json.dumps(_DECISION)},
            "seat": {"response": json.dumps(_SEAT)},
            "seat_contract": {"response": _ENGINE_WRAP},
        }
    )

    assert shaped["seat_gate_failed"]


def test_an_absent_seat_contract_also_fails_closed() -> None:
    """Review corrected an earlier draft that failed OPEN here.

    The draft's reason was "the explain path has no seat_contract block",
    which conflates the ensemble's optional `seat_contract:` YAML block
    with the skeleton's `seat_contract` NODE. `serving.yaml` declares that
    node unconditionally, so it runs on every route and emits
    `seat_admitted: true` vacuously when the route declares no contract.
    An earlier version of this docstring then claimed an absent dep
    "means it was filtered out for not succeeding". That is false in
    general: `when:`-skipped nodes are routinely absent, and a crashed
    script agent is always PRESENT with an error envelope, since every
    failure is caught inside the agent and returned as a
    `status="success"` response. Measured at zero absences across 658
    recorded live turns, so this branch is unreachable in the current
    skeleton and is kept as a deliberate trip-wire: if a future skeleton
    guards or removes the node, build turns refuse loudly rather than
    silently losing the gate.
    """
    shaped = _shape_raw(
        {
            "classify": {"response": json.dumps(_DECISION)},
            "seat": {"response": json.dumps(_SEAT)},
        }
    )

    assert shaped["seat_gate_failed"]


def test_a_healthy_seat_contract_is_not_a_failure() -> None:
    shaped = _shape_raw(
        {
            "classify": {"response": json.dumps(_DECISION)},
            "seat": {"response": json.dumps(_SEAT)},
            "seat_contract": {
                "response": json.dumps(
                    {"seat_admitted": True, "seat_contract_reason": ""}
                )
            },
        }
    )

    assert not shaped["seat_gate_failed"]
    assert shaped["seat_admitted"] is True


# --- #168: a refusal reason names no filesystem path -------------------------
#
# The engine's four-key wrap embeds the subprocess argv, so every wrapped node
# failure used to put the interpreter path, the script path, and hence the
# operator's home directory and username on the wire. Ten captures in
# .llm-orc/.serve-trace/turns.jsonl carry the shape (lines 150, 153, 469, 574,
# across shape/resolve/seat_contract).

_HOME = str(Path.home())
_USER = _HOME.rsplit("/", 1)[-1]
_ARGV = f"'['{sys.executable}', '{REPO}/.llm-orc/scripts/agentic_serving/classify.py']'"


def _wrapped(error: str) -> dict[str, Any]:
    """The engine's wrap for a dead serving node — the four keys
    ScriptAgentOutput emits from execute_with_schema_json's blanket except."""
    return {"success": False, "data": None, "error": error, "agent_requests": []}


def _routing_reason(error: str) -> str:
    shaped = _shape_raw({"resolve": {"response": json.dumps(_wrapped(error))}})
    return str(shaped.get("routing_failed", ""))


def test_a_wrapped_exit_status_reason_names_no_path_or_user() -> None:
    """The captured shape, and the reason this issue exists."""
    reason = _routing_reason(
        f"Schema JSON execution failed: Command {_ARGV} "
        "returned non-zero exit status 1."
    )

    assert _HOME not in reason
    assert _USER not in reason
    assert "Command '[" not in reason


def test_an_unrecognized_error_shape_leaks_nothing_either() -> None:
    """The pin that separates positive extraction from a denylist, and the
    one the issue's own proposed fix would fail.

    #168 proposed cutting "the whole Command '[...]' clause". A
    FileNotFoundError from the same blanket except carries an absolute path
    and NO Command clause, so a clause-strip passes it through verbatim.
    """
    reason = _routing_reason(
        "Schema JSON execution failed: [Errno 2] No such file or directory: "
        f"'{REPO}/.llm-orc/scripts/agentic_serving/classify.py'"
    )

    assert _HOME not in reason
    assert _USER not in reason


def test_the_exit_status_tail_survives() -> None:
    """Not just deleting the reason: the actionable residue is kept.

    A DIRECTION GUARD, not a leak pin — it is green on main, and exists so
    the sanitiser cannot degrade into "say nothing"."""
    reason = _routing_reason(
        f"Schema JSON execution failed: Command {_ARGV} "
        "returned non-zero exit status 3."
    )

    assert "3" in reason
    assert "non-zero" in reason


def test_the_timeout_tail_survives() -> None:
    """Named separately from the exit-status family. One regex covering both
    is how a family gets silently dropped — and a timeout is the failure an
    operator most needs to tell apart from a crash."""
    reason = _routing_reason(
        f"Schema JSON execution failed: Command {_ARGV} timed out after 45 seconds"
    )

    assert "timed out" in reason
    assert "45" in reason
    assert _HOME not in reason


def test_the_failing_node_is_still_named() -> None:
    """The part of the reason an operator routes on: which node died.

    Also a direction guard, green on main."""
    error = (
        f"Schema JSON execution failed: Command {_ARGV} "
        "returned non-zero exit status 1."
    )
    shaped = _shape_raw({"classify": {"response": json.dumps(_wrapped(error))}})

    assert "classify" in str(shaped.get("routing_failed", ""))


def test_a_readable_routing_decision_is_unaffected() -> None:
    """The over-refusal direction. This CANNOT fail under deletion of the
    sanitiser — it is here so the fix does not become "refuse everything"."""
    shaped = _shape_raw(
        {
            "resolve": {
                "response": json.dumps(
                    {"target": "explainer", "kind": "explanation", "build": False}
                )
            },
            "seat": {"response": "some prose"},
        }
    )

    assert not shaped.get("routing_failed")


# --- #168 review round 1: the executor's report is a wire channel too -------
#
# refix_envelope binds `accept_reason` straight to the executor's report
# (unlike build_gated_envelope, which uses accept_gate's own constants), and
# emit ships it as "Another round needed: {reason}". The design recorded the
# `runner crashed` branch as "not reached by any measured path"; review
# demonstrated it reached, with the runner's absolute path in a traceback.

_EXECUTOR = REPO / ".llm-orc" / "scripts" / "agentic_serving" / "accept_executor.py"
_SMOKE_TEST = "def test_refix_candidate_loads_cleanly():\n    pass\n"


def _executor_report(code: str, tests: str = _SMOKE_TEST) -> str:
    payload = json.dumps(
        {
            "requirement": "fix calc.py",
            "code": code,
            "tests": tests,
            "target_file": "calc.py",
        }
    )
    out = subprocess.run(
        [sys.executable, str(_EXECUTOR)],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return str(json.loads(out)["report"])


def test_no_error_text_can_survive_the_summary() -> None:
    r"""The property the whole design rests on: both recognised residues are
    NUMERIC, so neither can carry a path by construction.

    Nothing pinned that, and review found a mutant that survived every other
    pin — widening the timeout capture from ``([\d.]+)`` to ``(.+?)``, a
    plausible edit to "support 1.5s or word units", re-opens a verbatim
    channel. This asserts the OUTPUT SHAPE rather than any particular input,
    so a widened capture fails here whatever it manages to capture.
    """
    leak = f"{_HOME}/secret/path"
    hostile = [
        f"Schema JSON execution failed: Command {_ARGV} "
        "returned non-zero exit status 1.",
        f"Schema JSON execution failed: Command {_ARGV} timed out after 45 seconds",
        f"timed out after {leak} 30 seconds",
        f"returned non-zero exit status {leak} 1",
        f"Schema JSON execution failed: [Errno 2] No such file or directory: '{leak}'",
        f"failed with exit code 1 at {leak}",
        leak,
        "",
    ]
    allowed = re.compile(
        r"^(exited non-zero, status \d+|timed out after [\d.]+ seconds|failed)$"
    )

    for error in hostile:
        reason = _routing_reason(error)
        clause = re.search(r"\(resolve: (.*?)\);", reason)
        # No clause at all means no verbatim text reached the wire, which
        # satisfies the property trivially (the empty error takes that path).
        if clause is None:
            assert _HOME not in reason, f"{error!r} -> {reason!r}"
            continue

        assert allowed.fullmatch(clause.group(1)), f"{error!r} -> {clause.group(1)!r}"


@pytest.mark.parametrize(
    ("label", "code"),
    [
        # A produced module whose exception message names a path. The repr
        # used to be interpolated whole.
        ("raises with a path", f"raise RuntimeError('missing {_HOME}/app.cfg')\n"),
        # SyntaxError's repr carries a filename tuple.
        ("syntax error", "def f(:\n"),
        # A module that kills the runner mid-write: stderr is a TRACEBACK
        # naming the runner's own absolute path, and it was echoed verbatim.
        ("runner dies", "import sys\nsys.stdout.close()\n"),
    ],
)
def test_the_executor_report_names_no_path_or_user(label: str, code: str) -> None:
    report = _executor_report(code)

    assert _HOME not in report, f"{label}: {report!r}"
    assert _USER not in report, f"{label}: {report!r}"


def test_a_failing_test_names_no_path_or_user() -> None:
    """Review round 2 BLOCKER, and the most reachable channel of the four: a
    FAILING TEST is the ordinary outcome of a re-fix round, not a crash, and
    `refix_envelope` binds the report straight to `accept_reason`.

    The produced code shells out to `sys.executable`, which inside the
    sandbox is the serve's own interpreter — so `CalledProcessError`'s repr
    carried the operator's home directory and username with no cooperation
    from the client at all.
    """
    report = _executor_report(
        "def run_check():\n"
        "    import subprocess, sys\n"
        "    subprocess.run(\n"
        '        [sys.executable, "-c", "raise SystemExit(1)"], check=True\n'
        "    )\n",
        tests=(
            "from solution import run_check\ndef test_run_check():\n    run_check()\n"
        ),
    )

    assert _HOME not in report, report
    assert _USER not in report, report
    assert "CalledProcessError" in report, "the failure class is the residue"


def test_a_unittest_failure_names_no_path_or_user() -> None:
    """The TestCase dialect is a separate channel: the last traceback line is
    ``Type: {str(exc)}``, and OSError's __str__ includes the filename its
    repr hides."""
    report = _executor_report(
        "",
        tests=(
            "import unittest\n"
            "class T(unittest.TestCase):\n"
            "    def test_load(self):\n"
            '        open("/private/var/folders/zz/secret_cfg.json")\n'
        ),
    )

    assert "/private/var/folders" not in report, report
    assert "FileNotFoundError" in report


def test_a_tests_module_that_raises_names_no_path_or_user() -> None:
    """The tests-load path had no pin at all — reverting it to `{error!r}`
    left the whole serving and endpoint suite green (review round 2). The
    three cases above all take the CODE load path or the runner-crash path.
    """
    report = _executor_report(
        "def add(a, b):\n    return a + b\n",
        tests=f'raise RuntimeError("missing {_HOME}/app.cfg")\n',
    )

    assert _HOME not in report, report
    assert "RuntimeError" in report


def test_a_syntax_error_keeps_its_line_and_column() -> None:
    """Review round 2: dropping these made the sanitising a worse trade than
    it needed to be. They are integers, so they cannot carry a path, and the
    compile filename is the literal "solution.py" — while the operator's
    server-side copy of this report is clipped by the trace snippet, so the
    class name alone would be the ONLY surviving record."""
    report = _executor_report("def add(a, b:\n    return a + b\n")

    assert "SyntaxError" in report
    assert "line 1" in report, report


def test_the_executor_report_still_says_what_happened() -> None:
    """The direction guard: the class of failure is the actionable part and
    it survives. Green on main."""
    assert "SyntaxError" in _executor_report("def f(:\n")
    assert "exit 1" in _executor_report("import sys\nsys.stdout.close()\n")


def test_a_path_free_failure_message_survives() -> None:
    """The tension this rule exists to hold. Sanitising by class name alone
    destroyed "DID NOT RAISE", which a pre-existing pin
    (test_pytest_raises_did_not_raise_reports_as_failure_not_crash) holds
    precisely because losing it "starved the retry round of evidence".

    The message is raised from the CODE module, not from an inline assert.
    Review round 3 caught the first version passing vacuously: an
    ``assert x, 'literal'`` in the tests source is echoed back by
    ``_failing_line``'s ``at: <source line>``, so the assertion was
    satisfied by the source echo rather than by the message pass-through it
    claimed to test, and it stayed green under a class-name-only mutant.
    """
    report = _executor_report(
        "def add(a, b):\n    raise ValueError('off by one somewhere')\n",
        tests="from solution import add\ndef test_add():\n    add(1, 2)\n",
    )

    assert "off by one somewhere" in report, report
    assert _HOME not in report


_HOSTILE_CORPUS = [
    # Review round 3's capture: a MULTI-LINE message. The last traceback line
    # has no colon, so the `split(":", 1)[0]` fallback returned it unchanged.
    (
        "",
        "import pathlib, unittest\n"
        "class T(unittest.TestCase):\n"
        "    def test_files(self):\n"
        "        entries = sorted(str(p) for p in pathlib.Path.home().glob('.T*'))\n"
        "        joined = '\\n'.join(entries)\n"
        "        raise AssertionError('unexpected entries:\\n' + joined)\n",
    ),
    # The class NAME is produced-code-controlled and was emitted unchecked.
    (
        f"class {'E'}(Exception):\n    pass\n"
        f"E.__name__ = '{_HOME}/leak'\n"
        "def go():\n    raise E()\n",
        "from solution import go\ndef test_go():\n    go()\n",
    ),
    # The ordinary repr leak round 2 closed, kept so the corpus covers it.
    (
        "def run_check():\n"
        "    import subprocess, sys\n"
        "    subprocess.run(\n"
        '        [sys.executable, "-c", "raise SystemExit(1)"], check=True\n'
        "    )\n",
        "from solution import run_check\ndef test_run_check():\n    run_check()\n",
    ),
    # An OSError whose __str__ carries the filename its repr hides.
    (
        "",
        "import unittest\n"
        "class T(unittest.TestCase):\n"
        "    def test_load(self):\n"
        '        open("/private/var/folders/zz/secret_cfg.json")\n',
    ),
    # A load failure whose message names a path.
    ("", f'raise RuntimeError("missing {_HOME}/app.cfg")\n'),
    # Round 5: the produced tests SOURCE naming an absolute path is the only
    # input that exercises the `at:` echo's guard in the leak direction.
    (
        "import os\n",
        f'import os\ndef test_save():\n    assert os.path.exists("{_HOME}/out.txt")\n',
    ),
    # Round 4's seventh channel: the TEST NAME is produced-code-controlled
    # and was interpolated into the guard's own fallback. The nested test_*
    # def forces _enumerate_tests to give up and take the legacy single run,
    # which is where the dynamic name reaches the report.
    (
        "",
        "import pathlib\n"
        "class TestFoo:\n"
        "    def test_nested(self):\n"
        "        pass\n"
        "def _t():\n"
        "    raise AssertionError('boom')\n"
        "globals()['test_' + str(pathlib.Path.home())] = _t\n",
    ),
    # The class name is produced-code-controlled on the LOAD path too, which
    # is a different emission point from the test path above.
    (
        f"class E(Exception):\n    pass\nE.__name__ = '{_HOME}/leak'\nraise E()\n",
        "def test_x():\n    assert True\n",
    ),
]


@pytest.mark.parametrize(("code", "tests"), _HOSTILE_CORPUS)
def test_no_report_can_name_a_path(code: str, tests: str) -> None:
    """The runner's analogue of test_no_error_text_can_survive_the_summary,
    and the pin whose absence let a sixth channel through.

    Review round 3: the shape.py half of this issue got an OUTPUT-shape
    assertion while the runner half got only "this particular input does not
    leak" — producer-by-producer testing wearing a property's clothes. Every
    runner leak so far has been a DERIVED string that no input-side check
    covered, so this asserts the property on the report itself and will fail
    on the seventh producer as readily as on the sixth.
    """
    report = _executor_report(code, tests=tests)

    assert "/" not in report, report
    assert "\\" not in report, report


def test_a_relative_path_in_the_source_echo_keeps_the_evidence() -> None:
    """Round 4: checking only the COMPOSED detail discarded the class name
    and the message along with the dirty ``at:`` echo, because the separator
    rule catches relative paths too. An ordinary failing test reduced to
    "failed" — the evidence loss round 2 paid to avoid, and it also feeds the
    next round's retry prompt, so the model gets nothing to act on.

    Each piece is checked as it enters the string instead.
    """
    report = _executor_report(
        "import os\n",
        tests=(
            "import os\ndef test_save():\n    assert os.path.exists('data/out.txt')\n"
        ),
    )

    assert "AssertionError" in report, report
    assert "data/out.txt" not in report, report


def test_a_unittest_diff_over_relative_paths_keeps_the_evidence() -> None:
    """Round 5. The piece-by-piece discipline reached `_run_test_fns` and not
    the `TestCase` branch 75 lines below, so an ordinary `assertEqual` over a
    sequence containing a relative path lost the test name, the exception
    class and the message together — `test failed`, in the client's reason
    AND in the retry prompt, for a real and unremarkable failure.

    The cause is that an `assertEqual` diff is MULTI-LINE: its last line is a
    diff fragment with no colon, so the type salvage had nothing to keep.
    Round 8 (#178) deleted the scan entirely — `_safe_reason` sees the live
    exception object, so the message not being path-free means the whole
    message is dropped by the same check it always used, not by a
    traceback-shape coincidence.
    """
    report = _executor_report(
        'def listing():\n    return ["src/a.py"]\n',
        tests=(
            "import unittest\n"
            "from solution import listing\n"
            "class T(unittest.TestCase):\n"
            "    def test_l(self):\n"
            '        self.assertEqual(listing(), ["src/b.py"])\n'
        ),
    )

    assert "AssertionError" in report, report
    assert "test_l" in report, report
    assert "src/a.py" not in report, report


def test_a_decoy_message_line_does_not_displace_the_exception_class() -> None:
    """Round 6. `_exception_line` scanned BACKWARD for the first line whose
    head is an identifier followed by a colon, so a message line that happens
    to read ``Key: value`` won over the real type line above it:

        AssertionError('Response mismatch:\\nStatus: 404\\nBody: not found')
        -> 'test_response: Body: not found'    the class dropped entirely

    Round 8 (#178) deleted the scan: the ``TestCase`` branch now reports the
    live exception object through `_safe_reason`, same as `_run_test_fns`,
    so there is no line to displace the class in the first place. The full
    message — decoy line and all — is kept, because it names no path; that
    is `_safe_reason`'s existing discipline, not a new exception for this
    input. What the original bug lost (the class) and what round 6's own
    fix happened to also lose (everything past the first physical line) are
    both restored.
    """
    report = _executor_report(
        "",
        tests=(
            "import unittest\n"
            "class T(unittest.TestCase):\n"
            "    def test_response(self):\n"
            '        raise AssertionError("Response mismatch:'
            '\\nStatus: 404\\nBody: not found")\n'
        ),
    )

    assert "AssertionError: Response mismatch" in report, report
    assert "Body: not found" in report, report


def test_a_chained_exception_reports_the_one_that_failed() -> None:
    """Round 7, and the third distinct way this line has been picked wrong.

    A chained traceback has one frame block per link, so returning at the
    FIRST column-0 line after a frame reported the CAUSE — ordinary
    `except ValueError: raise RuntimeError(...)` cleanup told the retry loop
    the wrong exception class and message. Keeping the LAST candidate lands
    on the exception that actually terminated.

    Asymmetry worth noting: the `test_*`-function branch never had this bug,
    because `_safe_reason` works on the live exception object rather than on
    formatted text. Round 8 (#178) closed the asymmetry by deleting the
    `TestCase` branch's parser and routing it through `_safe_reason` too —
    the class here is asserted directly off the exception `_LiveResult`
    captured, not recovered from any grammar, chained or not.
    """
    report = _executor_report(
        "",
        tests=(
            "import unittest\n"
            "class T(unittest.TestCase):\n"
            "    def test_cleanup(self):\n"
            "        try:\n"
            "            raise ValueError('root cause')\n"
            "        except ValueError:\n"
            "            raise RuntimeError('actual failure')\n"
        ),
    )

    assert "RuntimeError" in report, report
    assert "ValueError" not in report, report


def test_a_multiline_string_diff_keeps_the_evidence() -> None:
    """Round 8 (#178), and the confirmation review's blocking finding.

    `difflib.ndiff` prefixes UNCHANGED lines with two spaces. Round 5's
    fix assumed every diff line sits at column 0; that is false the moment
    the compared strings share a line, and `assertEqual` over two
    multi-line strings routes through `assertMultiLineEqual`, which is
    exactly `ndiff`. The two-space context line reads as a frame to
    `_exception_line`'s scan, the scan resets on it, and the LAST
    candidate becomes a diff fragment instead of the exception type —
    `AssertionError` is lost, not merely reduced to `failed`.
    """
    report = _executor_report(
        "",
        tests=(
            "import unittest\n"
            "class T(unittest.TestCase):\n"
            "    def test_multiline(self):\n"
            "        a = 'line one\\nline two\\nline three'\n"
            "        b = 'line one\\nline TWO\\nline three'\n"
            "        self.assertEqual(a, b)\n"
        ),
    )

    assert "AssertionError" in report, report


def test_a_twelve_item_list_diff_keeps_the_evidence() -> None:
    """Same defect as the multi-line string above, on the other shape the
    review named: a list long enough that `pprint.pformat` wraps it across
    lines also produces two-space `ndiff` context lines. The shipped round
    5 pin (`test_a_unittest_diff_over_relative_paths_keeps_the_evidence`)
    passes only because its one-item lists pformat to a single line each,
    which suppresses every context line — it never exercised the scan's
    real failure mode.
    """
    items_a = ", ".join(f"'item{i}'" for i in range(12))
    items_b = ", ".join(f"'item{i}'" for i in range(11)) + ", 'itemELEVEN'"
    report = _executor_report(
        "",
        tests=(
            "import unittest\n"
            "class T(unittest.TestCase):\n"
            "    def test_biglist(self):\n"
            f"        a = [{items_a}]\n"
            f"        b = [{items_b}]\n"
            "        self.assertEqual(a, b)\n"
        ),
    )

    assert "AssertionError" in report, report


def test_a_forged_frame_in_the_message_does_not_displace_the_real_class() -> None:
    """Review F5. A message containing an indented line — shaped like a
    frame — followed by a column-0 line shaped like `Type: message`
    displaces the real exception under the reset-and-keep-last scan: the
    forged line reads as a fresh candidate immediately after what the scan
    treats as a frame, and it is the last one seen.

    The message content survives on the wire regardless — that part is
    correct, since it names no path — but the reported CLASS must be the
    live exception's, not text an attacker put inside its message.
    """
    report = _executor_report(
        "",
        tests=(
            "import unittest\n"
            "class T(unittest.TestCase):\n"
            "    def test_forged(self):\n"
            '        raise AssertionError("boom\\n    (forged) File '
            "'evil.py', line 1\\nPermissionError: everything is fine\")\n"
        ),
    )

    assert "AssertionError: boom" in report, report
    assert "): PermissionError" not in report, report


def test_a_subtest_failure_is_not_silently_dropped() -> None:
    """Found while implementing round 8 (#178), not by review — worth
    pinning precisely because it is not.

    `unittest.TestResult.addSubTest`'s default implementation does NOT call
    `addFailure`/`addError`; it appends straight to `self.failures`/
    `self.errors`. A `_LiveResult` that only overrides the two would let a
    produced test's `with self.subTest():` failure vanish entirely — not
    reduced to `failed`, gone, with `tests_pass` coming back `True`. That is
    a wrong-accept, and a worse regression than any of this arc's leaks:
    those degraded evidence, this would have flipped the verdict.
    """
    report = _executor_report(
        "",
        tests=(
            "import unittest\n"
            "class T(unittest.TestCase):\n"
            "    def test_sub(self):\n"
            "        with self.subTest():\n"
            "            raise AssertionError('sub failure')\n"
        ),
    )

    assert "AssertionError: sub failure" in report, report

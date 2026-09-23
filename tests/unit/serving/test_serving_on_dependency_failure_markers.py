"""Load-time pins for the shipped ``on_dependency_failure: run`` markers
(fail-closed-composition SF4).

These markers are what let a failure-handling node compose a refusal
when everything it depends on crashed, instead of cascade-skipping
along with the failure it exists to report (Invariant 13's rule 1,
amended). Loading the shipped YAML and checking the flag catches a
marker silently dropped in an edit — a REGRESSION a behavioral test
alone would only catch by actually triggering that crash path; this
test fails immediately, naming the missing marker, without needing to
reproduce a crash at all.
"""

from __future__ import annotations

from pathlib import Path

from llm_orc.core.config.ensemble_config import EnsembleLoader
from llm_orc.schemas.agent_config import AgentConfig

REPO = Path(__file__).resolve().parents[3]
AGENTIC_SERVING = REPO / ".llm-orc" / "ensembles" / "agentic-serving"


def _agent(agents: list[AgentConfig], name: str) -> AgentConfig:
    for agent in agents:
        if agent.name == name:
            return agent
    raise AssertionError(f"no agent named {name!r} in {[a.name for a in agents]}")


class TestServingMarshalMarkers:
    """serving.yaml's failure-handling chain: resolve, shape, form_gate,
    emit each compose (or forward) a refusal when their own upstream
    crashed, and each needs the marker to run at all in that case."""

    def test_resolve_is_marked_run(self) -> None:
        config = EnsembleLoader().load_from_file(str(AGENTIC_SERVING / "serving.yaml"))
        assert _agent(config.agents, "resolve").on_dependency_failure == "run"

    def test_shape_is_marked_run(self) -> None:
        config = EnsembleLoader().load_from_file(str(AGENTIC_SERVING / "serving.yaml"))
        assert _agent(config.agents, "shape").on_dependency_failure == "run"

    def test_form_gate_is_marked_run(self) -> None:
        config = EnsembleLoader().load_from_file(str(AGENTIC_SERVING / "serving.yaml"))
        assert _agent(config.agents, "form_gate").on_dependency_failure == "run"

    def test_emit_is_marked_run(self) -> None:
        config = EnsembleLoader().load_from_file(str(AGENTIC_SERVING / "serving.yaml"))
        assert _agent(config.agents, "emit").on_dependency_failure == "run"


class TestGatedRoundEnvelopeMarkers:
    """The three gated-build loop-body shapes' own terminal (``envelope``)
    is this round's local refusal composer (fail-closed-composition X1
    re-evaluation): kept even though a total-crash round already fails
    its wrapping dispatch/loop node either way (X1's handled_failure
    exclusion makes that failure unconditional), because the round's
    own recorded result still carries envelope's composed ADR-024
    content instead of a bare skip for anyone reading that round's
    artifact directly.
    """

    def test_build_gated_round_envelope_is_marked_run(self) -> None:
        config = EnsembleLoader().load_from_file(
            str(AGENTIC_SERVING / "build-gated-round.yaml")
        )
        assert _agent(config.agents, "envelope").on_dependency_failure == "run"

    def test_build_code_round_envelope_is_marked_run(self) -> None:
        config = EnsembleLoader().load_from_file(
            str(AGENTIC_SERVING / "build-code-round.yaml")
        )
        assert _agent(config.agents, "envelope").on_dependency_failure == "run"

    def test_write_tests_round_envelope_is_marked_run(self) -> None:
        config = EnsembleLoader().load_from_file(
            str(AGENTIC_SERVING / "write-tests-round.yaml")
        )
        assert _agent(config.agents, "envelope").on_dependency_failure == "run"

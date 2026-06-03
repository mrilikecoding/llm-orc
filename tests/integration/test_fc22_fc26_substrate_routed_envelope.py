"""FC-22 three-surface + FC-26 AS-7-amended substrate-routed integration tests.

Per ``docs/agentic-serving/system-design.agents.md`` §Fitness Criteria
FC-22 and FC-26:

* **FC-22** — ``dispatch_id`` consistency holds across three surfaces:
  the event stream's ``dispatch_id``, the envelope's
  ``diagnostics.dispatch_id``, and the artifact path's
  ``<dispatch_id>`` segment — for every substrate-routed dispatch.
  The companion file ``test_fc22_envelope_dispatch_id_correlation.py``
  closes the envelope-leg under WP-D (inline-response path); this
  module closes the third leg — the artifact filesystem path — under
  WP-E (ADR-025 substrate-routing).

* **FC-26** — the Result Summarizer Harness is NOT invoked for
  substrate-routed dispatches (``output_substrate: artifact``); IS
  invoked per ADR-004 mandate for inline-response dispatches
  (``output_substrate: inline``); ADR-004's per-invocation
  ``raw_output=True`` escape hatch composes with substrate-routing
  without contradiction. The structural floor of AS-7 amended is the
  default-with-conditional-skip property.

Both criteria exercise the real composition: real
:class:`DispatchEventSubstrate`, real :class:`SessionArtifactStore`
(writing to a tmp directory), real :class:`OrchestratorToolDispatch`,
and a scripted ``EnsembleSubstrateReader``-equivalent that returns
substrate configs the dispatch site reads to route per-ensemble. The
harness is constructed with a loud invoker that raises on any call so
substrate-routed dispatches' "0 harness invocations" property is
structurally observable (any harness call by the substrate path would
surface as an assertion failure rather than a silent pass).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from llm_orc.agentic.autonomy_policy import BASELINE_LEVEL, AutonomyPolicy
from llm_orc.agentic.composition_validator import (
    CompositionOutcome,
    CompositionRejected,
    CompositionRequest,
)
from llm_orc.agentic.dispatch_event_substrate import DispatchEventSubstrate
from llm_orc.agentic.orchestrator_tool_dispatch import (
    EnsembleSubstrateReader,
    InternalToolCall,
    OrchestratorToolDispatch,
    SubstrateRoutingConfig,
    ToolCallSuccess,
)
from llm_orc.agentic.result_summarizer_harness import ResultSummarizerHarness
from llm_orc.agentic.session_artifact_store import SessionArtifactStore
from llm_orc.core.config.ensemble_config import EnsembleConfig


class _ScriptedOperations:
    """Programmable ``EnsembleOperations`` double with per-name results."""

    def __init__(self, *, invoke_results: dict[str, dict[str, Any]]) -> None:
        self._invoke_results = invoke_results
        self.invoke_calls: list[dict[str, Any]] = []

    async def invoke(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self.invoke_calls.append(dict(arguments))
        name = arguments["ensemble_name"]
        return self._invoke_results[name]

    async def read_ensembles(self) -> list[dict[str, Any]]:
        return []


class _LoudSummarizerInvoker:
    """Invoker that explodes if called — FC-26 substrate path must not summarize."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def invoke(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(dict(arguments))
        return {"deliverable": "INLINE-PATH-SUMMARY"}


class _UnusedWriter:
    """compose_ensemble is out of scope here."""

    def write(self, config: EnsembleConfig) -> str:  # pragma: no cover
        raise AssertionError("local_ensemble_writer should not be called")


class _RejectingValidator:
    """compose_ensemble is out of scope here."""

    def validate(self, request: CompositionRequest) -> CompositionOutcome:
        return CompositionRejected(kind="missing_primitive", reason="unused")


class _ScriptedSubstrateReader:
    """Returns canned :class:`SubstrateRoutingConfig` values per ensemble.

    Stands in for the production
    :class:`~llm_orc.agentic.orchestrator_tool_dispatch.EnsembleConfigSubstrateReader`
    — its responsibility is "name → substrate config or None"; the
    EnsembleConfig source is not load-bearing for the integration.
    """

    def __init__(self, configs: dict[str, SubstrateRoutingConfig]) -> None:
        self._configs = configs

    def substrate_config_for(self, ensemble_name: str) -> SubstrateRoutingConfig | None:
        return self._configs.get(ensemble_name)


def _build_substrate_dispatch(
    *,
    operations: _ScriptedOperations,
    substrate: DispatchEventSubstrate,
    artifact_store: SessionArtifactStore,
    substrate_reader: EnsembleSubstrateReader,
    harness: ResultSummarizerHarness,
) -> OrchestratorToolDispatch:
    """Build a real-component dispatch with substrate routing wired."""
    return OrchestratorToolDispatch(
        operations=operations,
        harness=harness,
        autonomy_policy=AutonomyPolicy(level_provider=lambda: BASELINE_LEVEL),
        composition_validator=_RejectingValidator(),
        local_ensemble_writer=_UnusedWriter(),
        event_substrate=substrate,
        ensemble_substrate_reader=substrate_reader,
        session_artifact_store=artifact_store,
    )


@pytest.mark.asyncio
async def test_fc22_dispatch_id_consistent_across_event_envelope_artifact(
    tmp_path: Path,
) -> None:
    """FC-22 three-surface integration: events ↔ envelope ↔ artifact path.

    A real substrate-routed dispatch end-to-end. The same ``dispatch_id``
    value appears on (a) every event emitted during the dispatch (queried
    via ``DispatchEventSubstrate.events_for``), (b)
    ``envelope.diagnostics.dispatch_id``, and (c) the artifact filesystem
    path's ``<dispatch_id>`` segment.
    """
    substrate = DispatchEventSubstrate()
    artifact_store = SessionArtifactStore(agentic_sessions_root=tmp_path)
    operations = _ScriptedOperations(
        invoke_results={
            "code-generator": {
                "synthesis": "def reverse(s):\n    return s[::-1]\n",
                "results": {"coder": {"response": "def reverse(s): ..."}},
            }
        }
    )
    reader = _ScriptedSubstrateReader(
        configs={
            "code-generator": SubstrateRoutingConfig(
                output_substrate="artifact",
                output_retention=None,
                calibration_substrate_access="artifact",
                topaz_skill="code_generation",
            )
        }
    )
    harness = ResultSummarizerHarness(
        invoker=_LoudSummarizerInvoker(),
        summarizer_name="agentic-result-summarizer",
    )
    dispatch = _build_substrate_dispatch(
        operations=operations,
        substrate=substrate,
        artifact_store=artifact_store,
        substrate_reader=reader,
        harness=harness,
    )

    result = await dispatch.dispatch(
        InternalToolCall(
            id="call-fc22-int",
            name="invoke_ensemble",
            arguments={"name": "code-generator", "input": "reverse"},
        ),
        session_id="session-fc22",
    )

    assert isinstance(result, ToolCallSuccess)
    assert result.envelope is not None
    envelope_dispatch_id = result.envelope.diagnostics["dispatch_id"]

    # Surface 1: every emitted event for this dispatch shares the id.
    events = substrate.events_for(envelope_dispatch_id)
    assert len(events) >= 2  # at minimum DispatchTiming(start) + (end)
    event_ids = {getattr(event, "dispatch_id", None) for event in events}
    event_ids.discard(None)
    assert event_ids == {envelope_dispatch_id}

    # Surface 2: envelope.diagnostics carries the same id.
    assert envelope_dispatch_id == result.dispatch_id

    # Surface 3: the artifact filesystem path's <dispatch_id> segment matches.
    assert result.envelope.artifacts is not None
    assert len(result.envelope.artifacts) == 1
    artifact_path = result.envelope.artifacts[0]["path"]
    assert f"/{envelope_dispatch_id}/" in artifact_path
    # And the artifact actually exists on disk at the expected location.
    on_disk = tmp_path / "session-fc22" / envelope_dispatch_id
    assert on_disk.exists()
    assert (on_disk / "code-generator.py").exists()


@pytest.mark.asyncio
async def test_fc26_substrate_routed_dispatch_does_not_invoke_harness(
    tmp_path: Path,
) -> None:
    """FC-26 substrate path: 0 harness invocations under AS-7 amended.

    A real substrate-routed dispatch end-to-end; the harness is built
    with a recording invoker that explodes if called. Asserts the
    substrate branch returns ``ToolCallSuccess`` without ever invoking
    the result summarizer — the AS-7 amended conditional-skip property
    in structural form.
    """
    substrate = DispatchEventSubstrate()
    artifact_store = SessionArtifactStore(agentic_sessions_root=tmp_path)
    operations = _ScriptedOperations(
        invoke_results={
            "claim-extractor": {
                "synthesis": "- claim 1 (established)\n- claim 2 (contested)",
            }
        }
    )
    reader = _ScriptedSubstrateReader(
        configs={
            "claim-extractor": SubstrateRoutingConfig(
                output_substrate="artifact",
                output_retention=None,
                calibration_substrate_access=None,
                topaz_skill="factual_knowledge",
            )
        }
    )
    loud_invoker = _LoudSummarizerInvoker()
    harness = ResultSummarizerHarness(
        invoker=loud_invoker, summarizer_name="agentic-result-summarizer"
    )
    dispatch = _build_substrate_dispatch(
        operations=operations,
        substrate=substrate,
        artifact_store=artifact_store,
        substrate_reader=reader,
        harness=harness,
    )

    result = await dispatch.dispatch(
        InternalToolCall(
            id="call-fc26-sub",
            name="invoke_ensemble",
            arguments={"name": "claim-extractor", "input": "source"},
        ),
        session_id="session-fc26",
    )

    assert isinstance(result, ToolCallSuccess)
    assert result.envelope is not None
    assert result.envelope.artifacts is not None
    # AS-7 amended structural floor: substrate-routed = 0 harness calls.
    assert loud_invoker.calls == []


@pytest.mark.asyncio
async def test_fc26_inline_dispatch_invokes_harness_once(
    tmp_path: Path,
) -> None:
    """FC-26 inline path: 1 harness invocation per dispatch per ADR-004.

    The inline path's structural floor is the unchanged-by-AS-7-amendment
    property: ``output_substrate: inline`` (or system-ensemble category
    default) dispatches still flow through the result-summarizer harness.
    """
    substrate = DispatchEventSubstrate()
    artifact_store = SessionArtifactStore(agentic_sessions_root=tmp_path)
    operations = _ScriptedOperations(
        invoke_results={
            "agentic-calibration-checker": {
                "synthesis": "Verdict: Proceed; confidence: 0.91",
            }
        }
    )
    reader = _ScriptedSubstrateReader(
        configs={
            "agentic-calibration-checker": SubstrateRoutingConfig(
                output_substrate=None,
                output_retention=None,
                calibration_substrate_access=None,
                topaz_skill=None,  # no topaz_skill → system ensemble → inline
            )
        }
    )
    recording_invoker = _LoudSummarizerInvoker()
    harness = ResultSummarizerHarness(
        invoker=recording_invoker, summarizer_name="agentic-result-summarizer"
    )
    dispatch = _build_substrate_dispatch(
        operations=operations,
        substrate=substrate,
        artifact_store=artifact_store,
        substrate_reader=reader,
        harness=harness,
    )

    result = await dispatch.dispatch(
        InternalToolCall(
            id="call-fc26-inline",
            name="invoke_ensemble",
            arguments={
                "name": "agentic-calibration-checker",
                "input": "evaluate",
            },
        ),
        session_id="session-fc26-inline",
    )

    assert isinstance(result, ToolCallSuccess)
    assert result.envelope is not None
    assert result.envelope.artifacts is None
    # ADR-004 mandate preserved: inline path = 1 harness invocation.
    assert len(recording_invoker.calls) == 1


@pytest.mark.asyncio
async def test_fc26_mixed_mode_session_honors_per_dispatch_decision(
    tmp_path: Path,
) -> None:
    """Mixed-mode session: substrate + inline dispatches in the same session.

    A single session dispatches one substrate-routed capability ensemble
    (``code-generator``) and one inline-response system ensemble
    (``agentic-calibration-checker``). FC-26 asserts the per-dispatch
    routing decision is honored: the substrate dispatch does not flow
    through the harness, while the inline dispatch does. Net harness
    invocations across the session: exactly 1 (from the inline leg).
    """
    substrate = DispatchEventSubstrate()
    artifact_store = SessionArtifactStore(agentic_sessions_root=tmp_path)
    operations = _ScriptedOperations(
        invoke_results={
            "code-generator": {"synthesis": "def f(): pass\n"},
            "agentic-calibration-checker": {
                "synthesis": "Verdict: Proceed",
            },
        }
    )
    reader = _ScriptedSubstrateReader(
        configs={
            "code-generator": SubstrateRoutingConfig(
                output_substrate="artifact",
                output_retention=None,
                calibration_substrate_access=None,
                topaz_skill="code_generation",
            ),
            "agentic-calibration-checker": SubstrateRoutingConfig(
                output_substrate=None,
                output_retention=None,
                calibration_substrate_access=None,
                topaz_skill=None,
            ),
        }
    )
    recording_invoker = _LoudSummarizerInvoker()
    harness = ResultSummarizerHarness(
        invoker=recording_invoker, summarizer_name="agentic-result-summarizer"
    )
    dispatch = _build_substrate_dispatch(
        operations=operations,
        substrate=substrate,
        artifact_store=artifact_store,
        substrate_reader=reader,
        harness=harness,
    )

    substrate_result = await dispatch.dispatch(
        InternalToolCall(
            id="call-mixed-substrate",
            name="invoke_ensemble",
            arguments={"name": "code-generator", "input": "x"},
        ),
        session_id="session-mixed",
    )
    inline_result = await dispatch.dispatch(
        InternalToolCall(
            id="call-mixed-inline",
            name="invoke_ensemble",
            arguments={
                "name": "agentic-calibration-checker",
                "input": "evaluate",
            },
        ),
        session_id="session-mixed",
    )

    assert isinstance(substrate_result, ToolCallSuccess)
    assert substrate_result.envelope is not None
    assert substrate_result.envelope.artifacts is not None
    assert isinstance(inline_result, ToolCallSuccess)
    assert inline_result.envelope is not None
    assert inline_result.envelope.artifacts is None
    # Mixed-mode session honors per-dispatch routing: substrate = 0
    # harness invocations, inline = 1 → net 1 invocation for the session.
    assert len(recording_invoker.calls) == 1

"""FC-27 anchor — validate-once-at-load eliminates per-enumeration noise.

Per ``docs/agentic-serving/scenarios.md`` §Observability Event Routing
scenario "Validate-once-at-load eliminates per-enumeration noise":

    **Given** an operator-deployed serve at startup that loads the
    ensemble library containing legacy schema-drifted YAMLs and valid
    ensembles
    **When** the serve completes startup
    **Then** the legacy YAMLs produce one ``WARN`` line each at startup
    with the file path and validation error rationale; the valid
    subset is loaded. Subsequent ``list_ensembles()`` calls return the
    validated subset without re-emitting warnings. A multi-dispatch
    session with 8 enumeration cycles produces 0 additional validation
    warnings for the same legacy YAMLs.

The integration test composes the production
:class:`EnsembleLoader` (primed) with the production
:class:`OperatorTerminalEventSink` (consuming ``validation_results()``
and emitting ``WARN`` lines), then drives 8 ``list_ensembles`` calls
to confirm no additional WARNs surface.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
import yaml

from llm_orc.agentic.operator_terminal_event_sink import OperatorTerminalEventSink
from llm_orc.core.config.ensemble_config import EnsembleLoader


@pytest.fixture
def _llm_orc_logger_propagation() -> Any:
    """Restore ``llm_orc`` logger propagation so caplog observes the sink.

    The ``llm-orc serve`` / ``web`` CLI tests disable propagation on the
    ``llm_orc`` parent logger; the flag persists across tests and would
    silence caplog here. Self-contained restoration keeps this test
    independent of test-ordering.
    """
    orc_logger = logging.getLogger("llm_orc")
    previous = orc_logger.propagate
    orc_logger.propagate = True
    try:
        yield
    finally:
        orc_logger.propagate = previous


def _write_valid(directory: Path, name: str) -> None:
    (directory / f"{name}.yaml").write_text(
        yaml.dump(
            {
                "name": name,
                "description": f"{name} description",
                "agents": [{"name": "a1", "script": "echo hi"}],
            }
        )
    )


def _write_invalid(directory: Path, name: str) -> None:
    """Write a YAML the schema rejects (`extra='forbid'` on agents)."""
    (directory / f"{name}.yaml").write_text(
        yaml.dump(
            {
                "name": name,
                "description": f"{name} description",
                "agents": [
                    {
                        "name": "a1",
                        "model_profile": "test",
                        "synthesis_timeout_seconds": 90,
                    }
                ],
            }
        )
    )


@pytest.mark.usefixtures("_llm_orc_logger_propagation")
def test_validate_once_at_load_eliminates_per_enumeration_noise(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The startup-prime path emits one WARN per invalid YAML through
    the sink. Eight subsequent enumerations emit zero additional WARNs.
    """
    library = tmp_path / "library"
    library.mkdir()
    _write_valid(library, "code-generator")
    _write_valid(library, "text-summarizer")
    _write_invalid(library, "fan-out-test")
    _write_invalid(library, "plexus-graph-analysis")

    loader = EnsembleLoader()
    sink = OperatorTerminalEventSink()

    # Startup: prime the loader (one walk through the library) and drain
    # validation_results to the sink.
    with caplog.at_level(logging.WARNING, logger="llm_orc.agentic.operator_terminal"):
        loader.prime(str(library))
        sink.report_validation_results(loader.validation_results())

    operator_warnings = [
        r
        for r in caplog.records
        if r.name == "llm_orc.agentic.operator_terminal"
        and r.levelno >= logging.WARNING
    ]
    assert len(operator_warnings) == 2
    joined = " | ".join(r.message for r in operator_warnings)
    assert "fan-out-test.yaml" in joined
    assert "plexus-graph-analysis.yaml" in joined

    # The valid subset is returned by list_ensembles without re-validation;
    # eight subsequent enumerations emit zero additional WARNs.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="llm_orc.agentic.operator_terminal"):
        for _ in range(8):
            entries = loader.list_ensembles(str(library))
            assert {e.name for e in entries} == {"code-generator", "text-summarizer"}

    additional = [
        r
        for r in caplog.records
        if r.name == "llm_orc.agentic.operator_terminal"
        and r.levelno >= logging.WARNING
    ]
    assert additional == []

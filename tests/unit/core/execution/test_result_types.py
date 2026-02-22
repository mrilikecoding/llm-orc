"""Tests for typed result models."""

from __future__ import annotations

from llm_orc.core.execution.result_types import (
    AgentResult,
    ExecutionMetadata,
    ExecutionResult,
)


class TestAgentResult:
    """Tests for AgentResult dataclass."""

    def test_success_result(self) -> None:
        """Success result has expected fields."""
        result = AgentResult(
            status="success",
            response="test output",
            model_substituted=False,
        )

        assert result.status == "success"
        assert result.response == "test output"
        assert result.error is None
        assert result.model_substituted is False

    def test_failed_result(self) -> None:
        """Failed result has error field."""
        result = AgentResult(
            status="failed",
            error="something went wrong",
        )

        assert result.status == "failed"
        assert result.error == "something went wrong"
        assert result.response is None

    def test_to_dict_success(self) -> None:
        """to_dict on success excludes error."""
        result = AgentResult(
            status="success",
            response="output",
            model_substituted=True,
        )

        d = result.to_dict()

        assert d == {
            "response": "output",
            "status": "success",
            "model_substituted": True,
        }
        assert "error" not in d

    def test_to_dict_failure(self) -> None:
        """to_dict on failure includes error."""
        result = AgentResult(
            status="failed",
            error="oops",
        )

        d = result.to_dict()

        assert d["status"] == "failed"
        assert d["error"] == "oops"

    def test_to_dict_excludes_model_instance(self) -> None:
        """to_dict never includes model_instance."""
        result = AgentResult(
            status="success",
            response="test",
            model_instance="should not appear",
        )

        d = result.to_dict()

        assert "model_instance" not in d

    def test_defaults(self) -> None:
        """Default values are sensible."""
        result = AgentResult(status="success")

        assert result.response is None
        assert result.error is None
        assert result.model_substituted is False
        assert result.model_instance is None


class TestExecutionMetadata:
    """Tests for ExecutionMetadata dataclass."""

    def test_minimal_metadata(self) -> None:
        """Minimal metadata has required fields only."""
        meta = ExecutionMetadata(agents_used=3, started_at=1000.0)

        assert meta.agents_used == 3
        assert meta.started_at == 1000.0
        assert meta.duration is None

    def test_to_dict_minimal(self) -> None:
        """to_dict with only required fields."""
        meta = ExecutionMetadata(agents_used=2, started_at=1000.0)

        d = meta.to_dict()

        assert d == {"agents_used": 2, "started_at": 1000.0}

    def test_to_dict_with_optional_fields(self) -> None:
        """to_dict includes set optional fields."""
        meta = ExecutionMetadata(
            agents_used=3,
            started_at=1000.0,
            duration="1.50s",
            completed_at=1001.5,
        )

        d = meta.to_dict()

        assert d["duration"] == "1.50s"
        assert d["completed_at"] == 1001.5


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_initial_result(self) -> None:
        """Initial result has running status."""
        meta = ExecutionMetadata(agents_used=2, started_at=1000.0)
        result = ExecutionResult(
            ensemble="test",
            status="running",
            input={"data": "hello"},
            results={},
            metadata=meta,
        )

        assert result.status == "running"
        assert result.results == {}

    def test_to_dict_basic(self) -> None:
        """to_dict produces expected shape."""
        meta = ExecutionMetadata(agents_used=1, started_at=1000.0)
        result = ExecutionResult(
            ensemble="test",
            status="completed",
            input={"data": "hello"},
            results={
                "agent1": AgentResult(
                    status="success",
                    response="world",
                ),
            },
            metadata=meta,
        )

        d = result.to_dict()

        assert d["ensemble"] == "test"
        assert d["status"] == "completed"
        assert d["results"]["agent1"]["response"] == "world"
        assert "model_instance" not in d["results"]["agent1"]

    def test_to_dict_with_plain_dict_results(self) -> None:
        """to_dict handles plain dict results for backward compat."""
        meta = ExecutionMetadata(agents_used=1, started_at=1000.0)
        result = ExecutionResult(
            ensemble="test",
            status="completed",
            input={"data": "hello"},
            results={
                "agent1": {
                    "response": "world",
                    "status": "success",
                },
            },
            metadata=meta,
        )

        d = result.to_dict()

        assert d["results"]["agent1"]["response"] == "world"

    def test_to_dict_synthesis_none(self) -> None:
        """to_dict includes synthesis as None when not set."""
        meta = ExecutionMetadata(agents_used=1, started_at=1000.0)
        result = ExecutionResult(
            ensemble="test",
            status="completed",
            input={"data": "hello"},
            results={},
            metadata=meta,
        )

        d = result.to_dict()

        assert "synthesis" in d
        assert d["synthesis"] is None

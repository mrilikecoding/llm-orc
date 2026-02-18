"""Tests for replicate_n_times primitive."""

import json

import pytest
from pydantic import ValidationError

from llm_orc.primitives.control_flow.replicate_n_times import (
    ReplicateInput,
    ReplicateOutput,
    execute,
)


class TestReplicateInput:
    """Test input model validation."""

    def test_valid_input_with_defaults(self) -> None:
        model = ReplicateInput()
        assert model.replications == 1
        assert model.seed is None

    def test_custom_values(self) -> None:
        model = ReplicateInput(replications=10, seed=42)
        assert model.replications == 10
        assert model.seed == 42

    def test_replications_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            ReplicateInput(replications=0)

    def test_replications_must_be_int(self) -> None:
        with pytest.raises(ValidationError):
            ReplicateInput(replications=-1)


class TestReplicateOutput:
    """Test output model and field name compatibility."""

    def test_success_output_fields(self) -> None:
        output = ReplicateOutput(
            success=True,
            replications=[{"replication_id": 1}],
            total_replications=1,
            base_seed=None,
        )
        data = json.loads(output.model_dump_json())
        assert data["success"] is True
        assert data["replications"] == [{"replication_id": 1}]
        assert data["total_replications"] == 1
        assert data["base_seed"] is None

    def test_error_output_fields(self) -> None:
        output = ReplicateOutput(
            success=False,
            error="Something went wrong",
            replications=[],
        )
        data = json.loads(output.model_dump_json())
        assert data["success"] is False
        assert data["replications"] == []


class TestReplicateExecute:
    """Test execute function."""

    def test_single_replication(self) -> None:
        result = execute(ReplicateInput(replications=1))
        assert result.success is True
        assert result.total_replications == 1
        assert len(result.replications) == 1
        assert result.replications[0]["replication_id"] == 1
        assert result.replications[0]["total_replications"] == 1

    def test_multiple_replications(self) -> None:
        result = execute(ReplicateInput(replications=5))
        assert result.success is True
        assert result.total_replications == 5
        assert len(result.replications) == 5
        for i, rep in enumerate(result.replications):
            assert rep["replication_id"] == i + 1

    def test_seed_produces_deterministic_output(self) -> None:
        result1 = execute(ReplicateInput(replications=3, seed=42))
        result2 = execute(ReplicateInput(replications=3, seed=42))
        assert result1.replications == result2.replications

    def test_seed_produces_sequential_seeds(self) -> None:
        result = execute(ReplicateInput(replications=3, seed=100))
        assert result.base_seed == 100
        assert result.replications[0]["random_seed"] == 100
        assert result.replications[1]["random_seed"] == 101
        assert result.replications[2]["random_seed"] == 102

"""Replicate N times primitive with Pydantic contracts."""

from __future__ import annotations

import json
import os
import random
import sys
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ReplicateInput(BaseModel):
    """Input contract for replicate_n_times."""

    replications: int = 1
    seed: int | None = None

    @field_validator("replications")
    @classmethod
    def replications_must_be_positive(cls, v: int) -> int:
        if v < 1:
            msg = "replications must be >= 1"
            raise ValueError(msg)
        return v


class ReplicateOutput(BaseModel):
    """Output contract for replicate_n_times.

    Field names match the existing script output exactly.
    """

    success: bool
    replications: list[dict[str, Any]] = Field(default_factory=list)
    total_replications: int = 0
    base_seed: int | None = None
    error: str | None = None


def execute(params: ReplicateInput) -> ReplicateOutput:
    """Execute replicate_n_times."""
    if params.seed is not None:
        random.seed(params.seed)

    replication_configs: list[dict[str, Any]] = []
    for i in range(params.replications):
        replication_config: dict[str, Any] = {
            "replication_id": i + 1,
            "total_replications": params.replications,
            "random_seed": (
                params.seed + i
                if params.seed is not None
                else random.randint(1, 1_000_000)
            ),
        }
        replication_configs.append(replication_config)

    return ReplicateOutput(
        success=True,
        replications=replication_configs,
        total_replications=params.replications,
        base_seed=params.seed,
    )


def _resolve_parameters() -> dict[str, object]:
    """Resolve parameters from AGENT_PARAMETERS env var or stdin."""
    agent_params = os.environ.get("AGENT_PARAMETERS", "")
    if agent_params and agent_params != "{}":
        return json.loads(agent_params)  # type: ignore[no-any-return]

    if not sys.stdin.isatty():
        config: dict[str, object] = json.loads(sys.stdin.read())
    else:
        config = {}
    parameters = config.get("parameters", config)
    if not isinstance(parameters, dict):
        return config
    return parameters


def main() -> None:
    """Entry point for subprocess execution."""
    parameters = _resolve_parameters()

    params = ReplicateInput(
        replications=parameters.get("replications", 1),  # type: ignore[arg-type]
        seed=parameters.get("seed"),  # type: ignore[arg-type]
    )
    result = execute(params)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

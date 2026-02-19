"""JSON extract primitive with Pydantic contracts."""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from pydantic import BaseModel, Field


class JsonExtractInput(BaseModel):
    """Input contract for json_extract."""

    json_data: str | dict[str, Any]
    fields: list[str]


class JsonExtractOutput(BaseModel):
    """Output contract for json_extract.

    Field names match the existing script output exactly.
    """

    success: bool
    extracted: dict[str, Any] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)
    total_fields: int = 0
    extracted_count: int = 0
    error: str | None = None


def execute(params: JsonExtractInput) -> JsonExtractOutput:
    """Execute json_extract."""
    try:
        if isinstance(params.json_data, str):
            data: dict[str, Any] = json.loads(params.json_data)
        else:
            data = params.json_data

        extracted: dict[str, Any] = {}
        missing_fields: list[str] = []

        for field in params.fields:
            if field in data:
                extracted[field] = data[field]
            else:
                missing_fields.append(field)

        return JsonExtractOutput(
            success=True,
            extracted=extracted,
            missing_fields=missing_fields,
            total_fields=len(params.fields),
            extracted_count=len(extracted),
        )
    except json.JSONDecodeError as e:
        return JsonExtractOutput(
            success=False,
            error=f"Invalid JSON: {e}",
            extracted={},
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

    params = JsonExtractInput(
        json_data=parameters.get("json_data", "{}"),  # type: ignore[arg-type]
        fields=parameters.get("fields", []),  # type: ignore[arg-type]
    )
    result = execute(params)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

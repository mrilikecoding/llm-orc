"""Read file primitive with Pydantic contracts."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from pydantic import BaseModel


class ReadFileInput(BaseModel):
    """Input contract for read_file."""

    path: str
    encoding: str = "utf-8"


class ReadFileOutput(BaseModel):
    """Output contract for read_file.

    Field names match the existing script output exactly.
    """

    success: bool
    content: str | None = None
    path: str = ""
    size: int = 0
    error: str | None = None


def execute(params: ReadFileInput) -> ReadFileOutput:
    """Execute read_file."""
    try:
        content = Path(params.path).read_text(encoding=params.encoding)
        return ReadFileOutput(
            success=True,
            content=content,
            path=params.path,
            size=len(content),
        )
    except Exception as e:
        return ReadFileOutput(
            success=False,
            error=str(e),
            path=params.path,
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

    params = ReadFileInput(
        path=str(parameters.get("path", "input.txt")),
        encoding=str(parameters.get("encoding", "utf-8")),
    )
    result = execute(params)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

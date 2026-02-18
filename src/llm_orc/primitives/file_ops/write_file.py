"""Write file primitive with Pydantic contracts."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from pydantic import BaseModel


class WriteFileInput(BaseModel):
    """Input contract for write_file."""

    path: str
    content: str
    encoding: str = "utf-8"


class WriteFileOutput(BaseModel):
    """Output contract for write_file.

    Field names match the existing script output exactly.
    """

    success: bool
    path: str = ""
    size: int = 0
    bytes_written: int = 0
    error: str | None = None


def execute(params: WriteFileInput) -> WriteFileOutput:
    """Execute write_file."""
    try:
        target = Path(params.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(params.content, encoding=params.encoding)

        return WriteFileOutput(
            success=True,
            path=params.path,
            size=len(params.content),
            bytes_written=len(params.content.encode(params.encoding)),
        )
    except Exception as e:
        return WriteFileOutput(
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

    params = WriteFileInput(
        path=str(parameters.get("path", "output.txt")),
        content=str(parameters.get("content", "")),
        encoding=str(parameters.get("encoding", "utf-8")),
    )
    result = execute(params)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

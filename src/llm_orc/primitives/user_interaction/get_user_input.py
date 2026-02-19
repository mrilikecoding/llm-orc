"""Get user input primitive with Pydantic contracts."""

from __future__ import annotations

import json
import sys

from pydantic import BaseModel


class GetUserInputInput(BaseModel):
    """Input contract for get_user_input."""

    prompt: str
    multiline: bool = False
    pre_collected: str | None = None


class GetUserInputOutput(BaseModel):
    """Output contract for get_user_input.

    Field names match the existing script output exactly.
    """

    success: bool
    input: str = ""
    multiline: bool = False
    length: int = 0
    error: str | None = None


def execute(params: GetUserInputInput) -> GetUserInputOutput:
    """Execute get_user_input with pre-collected or interactive input."""
    if params.pre_collected is not None:
        return GetUserInputOutput(
            success=True,
            input=params.pre_collected,
            multiline=params.multiline,
            length=len(params.pre_collected),
        )

    try:
        if params.multiline:
            print(f"{params.prompt} (Enter blank line to finish)")
            lines: list[str] = []
            while True:
                line = input()
                if line.strip() == "":
                    break
                lines.append(line)
            user_input = "\n".join(lines)
        else:
            user_input = input(f"{params.prompt} ")

        return GetUserInputOutput(
            success=True,
            input=user_input,
            multiline=params.multiline,
            length=len(user_input),
        )
    except (EOFError, KeyboardInterrupt):
        return GetUserInputOutput(
            success=False,
            error="User cancelled input",
            input="",
        )


def main() -> None:
    """Entry point for subprocess execution."""
    if not sys.stdin.isatty():
        config: dict[str, object] = json.loads(sys.stdin.read())
    else:
        config = {}

    parameters = config.get("parameters", config)
    if not isinstance(parameters, dict):
        parameters = config

    pre_collected = config.get("input")
    if pre_collected is not None and isinstance(pre_collected, str):
        parameters["pre_collected"] = pre_collected

    params = GetUserInputInput(
        prompt=str(parameters.get("prompt", "Enter input:")),
        multiline=bool(parameters.get("multiline", False)),
        pre_collected=parameters.get("pre_collected"),  # type: ignore[arg-type]
    )
    result = execute(params)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

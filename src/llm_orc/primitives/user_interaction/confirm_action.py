"""Confirm action primitive with Pydantic contracts."""

from __future__ import annotations

import json
import sys

from pydantic import BaseModel


class ConfirmActionInput(BaseModel):
    """Input contract for confirm_action."""

    prompt: str
    default: str = "n"
    pre_collected: str | None = None


class ConfirmActionOutput(BaseModel):
    """Output contract for confirm_action.

    Field names match the existing script output exactly.
    """

    success: bool
    confirmed: bool = False
    input: str = ""
    prompt: str = ""
    error: str | None = None


def execute(params: ConfirmActionInput) -> ConfirmActionOutput:
    """Execute confirm_action with pre-collected or interactive input."""
    if params.pre_collected is not None:
        user_input = params.pre_collected.strip().lower()
        if user_input == "":
            user_input = params.default.lower()
        confirmed = user_input in ("y", "yes", "true", "1")
        return ConfirmActionOutput(
            success=True,
            confirmed=confirmed,
            input=user_input,
            prompt=params.prompt,
        )

    try:
        default_lower = params.default.lower()
        if default_lower == "y":
            full_prompt = f"{params.prompt} [Y/n]: "
        else:
            full_prompt = f"{params.prompt} [y/N]: "

        user_input = input(full_prompt).strip().lower()
        if user_input == "":
            user_input = default_lower

        confirmed = user_input in ("y", "yes", "true", "1")
        return ConfirmActionOutput(
            success=True,
            confirmed=confirmed,
            input=user_input,
            prompt=params.prompt,
        )
    except (EOFError, KeyboardInterrupt):
        return ConfirmActionOutput(
            success=False,
            confirmed=False,
            error="User cancelled confirmation",
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

    params = ConfirmActionInput(
        prompt=str(parameters.get("prompt", "Continue?")),
        default=str(parameters.get("default", "n")),
        pre_collected=parameters.get("pre_collected"),  # type: ignore[arg-type]
    )
    result = execute(params)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

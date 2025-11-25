"""Script validation schemas with Pydantic validation."""

from pydantic import BaseModel


class GetUserInputScript(BaseModel):
    """Schema for validating get_user_input script configurations."""

    prompt: str
    script_name: str
    default_value: str | None = None
    input_type: str = "string"

"""Tests for primitives registry."""

from llm_orc.primitives import get_output_schema


class TestGetOutputSchema:
    """Test the output schema registry."""

    def test_known_primitive_with_hyphen(self) -> None:
        schema = get_output_schema("primitives/user-interaction/get_user_input.py")
        assert schema is not None
        assert schema.__name__ == "GetUserInputOutput"

    def test_known_primitive_with_underscore(self) -> None:
        schema = get_output_schema("primitives/user_interaction/get_user_input.py")
        assert schema is not None
        assert schema.__name__ == "GetUserInputOutput"

    def test_confirm_action(self) -> None:
        schema = get_output_schema("primitives/user-interaction/confirm_action.py")
        assert schema is not None
        assert schema.__name__ == "ConfirmActionOutput"

    def test_json_extract(self) -> None:
        schema = get_output_schema("primitives/data-transform/json_extract.py")
        assert schema is not None
        assert schema.__name__ == "JsonExtractOutput"

    def test_read_file(self) -> None:
        schema = get_output_schema("primitives/file-ops/read_file.py")
        assert schema is not None
        assert schema.__name__ == "ReadFileOutput"

    def test_write_file(self) -> None:
        schema = get_output_schema("primitives/file-ops/write_file.py")
        assert schema is not None
        assert schema.__name__ == "WriteFileOutput"

    def test_replicate_n_times(self) -> None:
        schema = get_output_schema("primitives/control-flow/replicate_n_times.py")
        assert schema is not None
        assert schema.__name__ == "ReplicateOutput"

    def test_unknown_script_returns_none(self) -> None:
        schema = get_output_schema("scripts/custom/my_script.py")
        assert schema is None

    def test_empty_string_returns_none(self) -> None:
        schema = get_output_schema("")
        assert schema is None

    def test_without_py_extension(self) -> None:
        schema = get_output_schema("primitives/file-ops/read_file")
        assert schema is not None
        assert schema.__name__ == "ReadFileOutput"

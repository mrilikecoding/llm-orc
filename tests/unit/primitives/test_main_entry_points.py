"""Tests for primitive main() entry points (subprocess interface)."""

import json
import subprocess
import sys
from pathlib import Path


def _run_primitive(
    module_path: str, stdin_data: dict[str, object]
) -> dict[str, object]:
    """Run a primitive module as subprocess and return parsed output."""
    result = subprocess.run(
        [sys.executable, "-m", module_path],
        input=json.dumps(stdin_data),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    parsed: dict[str, object] = json.loads(result.stdout)
    return parsed


class TestGetUserInputMain:
    """Test get_user_input via subprocess."""

    def test_pre_collected_via_stdin(self) -> None:
        output = _run_primitive(
            "llm_orc.primitives.user_interaction.get_user_input",
            {"input": "hello", "parameters": {"prompt": "Enter:"}},
        )
        assert output["success"] is True
        assert output["input"] == "hello"
        assert output["length"] == 5

    def test_parameters_only(self) -> None:
        output = _run_primitive(
            "llm_orc.primitives.user_interaction.get_user_input",
            {
                "parameters": {
                    "prompt": "Enter:",
                    "pre_collected": "world",
                }
            },
        )
        assert output["success"] is True
        assert output["input"] == "world"


class TestConfirmActionMain:
    """Test confirm_action via subprocess."""

    def test_pre_collected_yes(self) -> None:
        output = _run_primitive(
            "llm_orc.primitives.user_interaction.confirm_action",
            {"input": "y", "parameters": {"prompt": "Continue?"}},
        )
        assert output["success"] is True
        assert output["confirmed"] is True

    def test_pre_collected_no(self) -> None:
        output = _run_primitive(
            "llm_orc.primitives.user_interaction.confirm_action",
            {"input": "n", "parameters": {"prompt": "Continue?"}},
        )
        assert output["confirmed"] is False


class TestJsonExtractMain:
    """Test json_extract via subprocess."""

    def test_extract_fields(self) -> None:
        output = _run_primitive(
            "llm_orc.primitives.data_transform.json_extract",
            {
                "parameters": {
                    "json_data": '{"name": "Alice", "age": 30}',
                    "fields": ["name"],
                }
            },
        )
        assert output["success"] is True
        assert output["extracted"] == {"name": "Alice"}
        assert output["extracted_count"] == 1


class TestReadFileMain:
    """Test read_file via subprocess."""

    def test_read_existing_file(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.txt"
        test_file.write_text("content here")

        output = _run_primitive(
            "llm_orc.primitives.file_ops.read_file",
            {"parameters": {"path": str(test_file)}},
        )
        assert output["success"] is True
        assert output["content"] == "content here"


class TestWriteFileMain:
    """Test write_file via subprocess."""

    def test_write_file(self, tmp_path: Path) -> None:
        target = tmp_path / "out.txt"
        output = _run_primitive(
            "llm_orc.primitives.file_ops.write_file",
            {
                "parameters": {
                    "path": str(target),
                    "content": "written",
                }
            },
        )
        assert output["success"] is True
        assert target.read_text() == "written"


class TestReplicateMain:
    """Test replicate_n_times via subprocess."""

    def test_replicate(self) -> None:
        output = _run_primitive(
            "llm_orc.primitives.control_flow.replicate_n_times",
            {"parameters": {"replications": 3, "seed": 42}},
        )
        assert output["success"] is True
        assert output["total_replications"] == 3
        replications = output["replications"]
        assert isinstance(replications, list)
        assert len(replications) == 3

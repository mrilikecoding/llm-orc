"""Tests for primitive main() functions called directly (coverage)."""

import io
import json
from pathlib import Path
from unittest.mock import patch

from llm_orc.primitives.control_flow import replicate_n_times as rn_mod
from llm_orc.primitives.data_transform import json_extract as je_mod
from llm_orc.primitives.file_ops import read_file as rf_mod
from llm_orc.primitives.file_ops import write_file as wf_mod
from llm_orc.primitives.user_interaction import confirm_action as ca_mod
from llm_orc.primitives.user_interaction import get_user_input as gui_mod


def _call_main(
    module: object,
    stdin_data: dict[str, object],
) -> dict[str, object]:
    """Call a primitive's main() with mocked stdin/stdout."""
    stdin_json = json.dumps(stdin_data)
    stdout_buf = io.StringIO()

    with (
        patch("sys.stdin", io.StringIO(stdin_json)),
        patch("builtins.print", side_effect=lambda x, **_: stdout_buf.write(x)),
    ):
        module.main()  # type: ignore[attr-defined]

    result: dict[str, object] = json.loads(stdout_buf.getvalue())
    return result


class TestGetUserInputMainDirect:
    def test_pre_collected(self) -> None:
        result = _call_main(
            gui_mod,
            {"input": "hello", "parameters": {"prompt": "Enter:"}},
        )
        assert result["success"] is True
        assert result["input"] == "hello"


class TestConfirmActionMainDirect:
    def test_pre_collected(self) -> None:
        result = _call_main(
            ca_mod,
            {"input": "y", "parameters": {"prompt": "OK?"}},
        )
        assert result["success"] is True
        assert result["confirmed"] is True


class TestJsonExtractMainDirect:
    def test_extract(self) -> None:
        result = _call_main(
            je_mod,
            {
                "parameters": {
                    "json_data": '{"a": 1}',
                    "fields": ["a"],
                }
            },
        )
        assert result["success"] is True
        assert result["extracted"] == {"a": 1}


class TestReadFileMainDirect:
    def test_missing_file(self) -> None:
        result = _call_main(
            rf_mod,
            {"parameters": {"path": "/nonexistent/file.txt"}},
        )
        assert result["success"] is False


class TestWriteFileMainDirect:
    def test_write(self, tmp_path: Path) -> None:
        target = str(tmp_path / "test_out.txt")
        result = _call_main(
            wf_mod,
            {"parameters": {"path": target, "content": "x"}},
        )
        assert result["success"] is True


class TestReplicateMainDirect:
    def test_replicate(self) -> None:
        result = _call_main(
            rn_mod,
            {"parameters": {"replications": 2, "seed": 1}},
        )
        assert result["success"] is True
        assert result["total_replications"] == 2

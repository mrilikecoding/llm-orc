"""A script's ``/// llm-orc`` block: the files it needs beside it (ruling 8)."""

from __future__ import annotations

import pytest

from llm_orc.core.execution.scripting.files_block import ListedFiles, listed_files

HASH_BLOCK = """\
#!/usr/bin/env python3
\"\"\"Docstring.\"\"\"
# /// llm-orc
# files = [
#   "_helpers.py",
#   "sub/chain_plan.py",
# ]
# ///
import _helpers
"""


class TestListedFiles:
    def test_no_block_lists_nothing(self) -> None:
        assert listed_files("print('hi')\n") == ListedFiles()

    def test_hash_block_lists_its_files_in_order(self) -> None:
        assert listed_files(HASH_BLOCK) == ListedFiles(
            paths=("_helpers.py", "sub/chain_plan.py")
        )

    def test_slash_block_for_slash_comment_languages(self) -> None:
        source = '// /// llm-orc\n// files = ["lib.js"]\n// ///\nrun();\n'
        assert listed_files(source) == ListedFiles(paths=("lib.js",))

    def test_crlf_source(self) -> None:
        source = '# /// llm-orc\r\n# files = ["a.py"]\r\n# ///\r\n'
        assert listed_files(source) == ListedFiles(paths=("a.py",))

    def test_only_the_first_block_counts_and_other_types_are_ignored(self) -> None:
        source = (
            '# /// script\n# dependencies = ["x"]\n# ///\n'
            '# /// llm-orc\n# files = ["a.py"]\n# ///\n'
            '# /// llm-orc\n# files = ["b.py"]\n# ///\n'
        )
        assert listed_files(source).paths == ("a.py",)

    @pytest.mark.parametrize(
        "body",
        [
            "files = [",
            'files = "a.py"',
            "files = [1]",
            'files = ["a.py"]\nextra = 1',
        ],
    )
    def test_a_body_that_is_not_one_list_of_paths_is_an_error(self, body: str) -> None:
        text = "".join(f"# {line}\n" for line in body.split("\n"))
        result = listed_files(f"# /// llm-orc\n{text}# ///\n")
        assert result.paths == ()
        assert result.error

    @pytest.mark.parametrize(
        "path", ["../x.py", "/etc/passwd", "a\\\\b.py", "a//b.py", "./a.py", ""]
    )
    def test_a_path_that_breaks_the_relative_rule_is_an_error(self, path: str) -> None:
        result = listed_files(f'# /// llm-orc\n# files = ["{path}"]\n# ///\n')
        assert result.paths == ()
        assert result.error

    def test_an_unclosed_block_is_an_error(self) -> None:
        result = listed_files('# /// llm-orc\n# files = ["a.py"]\nimport a\n')
        assert result.error

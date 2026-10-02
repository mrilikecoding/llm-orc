"""The artifact cleanup fixture walks the repository's real
``.llm-orc/artifacts``, which every xdist worker shares. A directory
another worker removes mid-walk must not fail the test that is tearing
down."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import _subdirs


def test_a_directory_removed_between_the_check_and_the_listing_is_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gone = tmp_path / "gone"
    gone.mkdir()
    real_iterdir = Path.iterdir

    def vanishing(self: Path) -> object:
        if self == gone:
            self.rmdir()  # the other worker removes it after is_dir()
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", vanishing)

    assert _subdirs(gone) == []


def test_it_lists_the_directories_and_not_the_files(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "f.txt").write_text("x")

    assert [d.name for d in _subdirs(tmp_path)] == ["a"]

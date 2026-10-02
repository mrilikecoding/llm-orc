"""The bundle store: a persisted closure is one stored run request,
written beside and renamed into place (Arc 5, ruling 9)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from llm_orc.services.handlers.bundle_store import (
    BundleError,
    BundleStore,
    stored_form,
)
from llm_orc.services.handlers.run_request import RunRequest

ROOT: dict[str, Any] = {
    "name": "pack",
    "description": "a bundle",
    "agents": [{"name": "s", "script": "tools/x.py"}],
}


def _stored(**more: Any) -> dict[str, Any]:
    request = RunRequest.parse(
        {
            "ensemble": ROOT,
            "scripts": {"tools/x.py": "print(1)\n"},
            "input": "dropped",
            "pull": True,
            "persist": "global",
            **more,
        }
    )
    return stored_form(request)


@pytest.fixture
def store(tmp_path: Path) -> BundleStore:
    return BundleStore(lambda: tmp_path / "global")


class TestStoredForm:
    def test_it_keeps_the_closure_and_drops_input_pull_and_persist(self) -> None:
        assert set(_stored()) == {
            "ensemble",
            "ensembles",
            "profiles",
            "scripts",
            "bind",
        }


class TestWriteAndRead:
    def test_a_written_bundle_reads_back_as_written(self, store: BundleStore) -> None:
        stored = _stored(bind={"a": "b"})

        store.write("pack", stored)

        assert store.read("pack") == stored

    def test_the_bundle_is_one_json_file_named_for_the_root(
        self, store: BundleStore, tmp_path: Path
    ) -> None:
        store.write("pack", _stored())

        assert [p.name for p in (tmp_path / "global" / "bundles").iterdir()] == [
            "pack.json"
        ]

    def test_writing_again_replaces_it(self, store: BundleStore) -> None:
        store.write("pack", _stored())
        newer = _stored(scripts={"tools/x.py": "print(2)\n"})

        store.write("pack", newer)

        assert store.read("pack") == newer

    def test_a_name_no_bundle_holds_reads_none(self, store: BundleStore) -> None:
        assert store.read("ghost") is None

    @pytest.mark.parametrize("name", ["../pack", "a/b", "", "/abs"])
    def test_a_name_that_is_not_one_file_name_reads_none(
        self, store: BundleStore, name: str
    ) -> None:
        store.write("pack", _stored())

        assert store.read(name) is None
        assert not store.has(name)

    def test_a_failure_before_the_rename_keeps_the_previous_bundle_and_no_stray(
        self,
        store: BundleStore,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        store.write("pack", _stored())
        before = (tmp_path / "global" / "bundles" / "pack.json").read_bytes()

        def failing_replace(src: Any, dst: Any) -> None:
            raise OSError(28, "No space left on device")

        with monkeypatch.context() as failing:
            failing.setattr(os, "replace", failing_replace)
            with pytest.raises(BundleError, match="No space"):
                store.write("pack", _stored(scripts={"tools/x.py": "print(2)\n"}))

        files = sorted((tmp_path / "global" / "bundles").iterdir())
        assert [p.name for p in files] == ["pack.json"]
        assert files[0].read_bytes() == before
        assert store.names() == ["pack"]

    def test_data_that_is_not_plain_json_is_refused_and_nothing_is_written(
        self, store: BundleStore, tmp_path: Path
    ) -> None:
        stored = _stored()
        stored["ensemble"] = {**ROOT, "extra": {1, 2}}

        with pytest.raises(BundleError, match="plain data"):
            store.write("pack", stored)

        assert not (tmp_path / "global" / "bundles" / "pack.json").exists()


class TestAnUnreadableBundle:
    def test_garbage_names_the_bundle(self, store: BundleStore, tmp_path: Path) -> None:
        store.write("pack", _stored())
        (tmp_path / "global" / "bundles" / "pack.json").write_text("{not json")

        with pytest.raises(BundleError, match="bundle 'pack'"):
            store.read("pack")

    def test_a_stored_request_that_does_not_validate_names_the_bundle(
        self, store: BundleStore, tmp_path: Path
    ) -> None:
        store.write("pack", _stored())
        (tmp_path / "global" / "bundles" / "pack.json").write_text(
            '{"ensemble": {"name": "pack"}, "scripts": {"date": "x"}}'
        )

        with pytest.raises(BundleError, match="bundle 'pack'.*script key"):
            store.read("pack")

    def test_a_root_whose_name_is_not_the_file_name_is_refused(
        self, store: BundleStore, tmp_path: Path
    ) -> None:
        store.write("pack", _stored())
        (tmp_path / "global" / "bundles" / "other.json").write_text(
            (tmp_path / "global" / "bundles" / "pack.json").read_text()
        )

        with pytest.raises(BundleError, match="bundle 'other'.*named"):
            store.read("other")


class TestOnlyTheExactNameIsABundle:
    """Pins that hold on a case-folding disk and a case-sensitive one."""

    def test_another_spelling_is_not_read_had_or_deleted(
        self, store: BundleStore, tmp_path: Path
    ) -> None:
        store.write("pack", _stored())

        assert store.read("PACK") is None
        assert not store.has("PACK")
        assert store.delete("PACK") is False
        assert (tmp_path / "global" / "bundles" / "pack.json").exists()
        assert store.has("pack")
        assert store.read("pack") is not None

    def test_a_file_whose_root_is_named_otherwise_is_not_a_bundle_to_has(
        self, store: BundleStore, tmp_path: Path
    ) -> None:
        store.write("pack", _stored())
        other = tmp_path / "global" / "bundles" / "other.json"
        other.write_text((tmp_path / "global" / "bundles" / "pack.json").read_text())

        assert not store.has("other")

    def test_a_file_is_deleted_by_its_own_name_whatever_root_it_holds(
        self, store: BundleStore, tmp_path: Path
    ) -> None:
        store.write("pack", _stored())
        other = tmp_path / "global" / "bundles" / "other.json"
        other.write_text((tmp_path / "global" / "bundles" / "pack.json").read_text())

        assert store.delete("other") is True
        assert not other.exists()
        assert store.has("pack")


class TestDeleteAndNames:
    def test_delete_removes_it_and_says_so(self, store: BundleStore) -> None:
        store.write("pack", _stored())

        assert store.delete("pack") is True
        assert store.read("pack") is None
        assert store.delete("pack") is False

    def test_names_lists_the_json_files_only(
        self, store: BundleStore, tmp_path: Path
    ) -> None:
        store.write("b", _stored(ensemble={**ROOT, "name": "b"}))
        store.write("a", _stored(ensemble={**ROOT, "name": "a"}))
        (tmp_path / "global" / "bundles" / "stray.tmp").write_text("x")

        assert store.names() == ["a", "b"]

    def test_names_of_a_store_that_was_never_written_is_empty(
        self, store: BundleStore
    ) -> None:
        assert store.names() == []

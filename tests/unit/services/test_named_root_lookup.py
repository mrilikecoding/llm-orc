"""The one named-root lookup, over a real OrchestraService on a temp
project (Arc 5, Task 2): a root is found by its ``name:`` whatever its
file is called, in tier order, and the config knows its file."""

# ruff: noqa: F811  (imported fixtures are redefined as test parameters)
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from llm_orc.services.orchestra_service import OrchestraService
from tests.unit.services.test_one_run_injection import (  # noqa: F401
    _ensemble,
    listing,
    project,
    service,
    state_dir,
)

SAYS_HI = [{"name": "s", "script": "echo hi"}]


def _named(base: Path, file_name: str, name: str, subdir: str = "") -> Path:
    """An ensemble called ``name`` in a file called ``file_name``."""
    _ensemble(base, name, SAYS_HI)
    target = base / "ensembles" / subdir / f"{file_name}.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    (base / "ensembles" / f"{name}.yaml").rename(target)
    return target


class TestSourcePath:
    def test_a_found_root_knows_the_file_it_was_read_from(
        self, project: Path, service: OrchestraService
    ) -> None:
        path = _named(project / ".llm-orc", "file-name", "root-name")

        config = service.find_ensemble_by_name("root-name")

        assert config is not None
        assert config.source_path == str(path)

    def test_a_hierarchical_root_knows_its_nested_file(
        self, project: Path, service: OrchestraService
    ) -> None:
        path = _named(project / ".llm-orc", "deep", "deep", subdir="grp/sub")

        config = service.find_ensemble_by_name("grp/sub/deep")

        assert config is not None
        assert config.source_path == str(path)


class TestEveryEntryPathFindsTheRoot:
    async def test_invoke_finds_a_root_by_its_name_field(
        self, project: Path, service: OrchestraService
    ) -> None:
        _named(project / ".llm-orc", "file-name", "root-name")

        result = await service.invoke({"ensemble_name": "root-name", "input": "x"})

        assert result["status"] == "success", result

    async def test_invoke_streaming_finds_a_root_by_its_name_field(
        self, project: Path, service: OrchestraService
    ) -> None:
        _named(project / ".llm-orc", "file-name", "root-name")

        events: list[dict[str, Any]] = [
            e
            async for e in service.invoke_streaming(
                {"ensemble_name": "root-name", "input": "x"}
            )
        ]

        assert events[-1]["type"] == "execution_completed", events[-1]

    async def test_a_root_that_no_tier_has_is_reported_missing(
        self, project: Path, service: OrchestraService
    ) -> None:
        with pytest.raises(ValueError, match="Ensemble not found: ghost"):
            [
                e
                async for e in service.invoke_streaming(
                    {"ensemble_name": "ghost", "input": "x"}
                )
            ]

        with pytest.raises(ValueError, match="Ensemble does not exist: ghost"):
            await service.invoke({"ensemble_name": "ghost", "input": "x"})

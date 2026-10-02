"""A script's files block through REST, over a real OrchestraService
(Arc 5, ruling 8). The runnable route and both execute routes read one
closure, so a listed file that is not beside its script blocks a run
before any agent starts."""

# ruff: noqa: F811  (imported fixtures are redefined as test parameters)
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from llm_orc.core.config.config_manager import resolve_global_config_dir
from tests.unit.services.test_one_run_injection import (  # noqa: F401
    _ensemble,
    listing,
    project,
    service,
    state_dir,
)
from tests.unit.web.test_api_run_injection import (  # noqa: F401
    _marker_source,
    client,
)


def _block(*paths: str) -> str:
    listed = ", ".join(f'"{p}"' for p in paths)
    return f"# /// llm-orc\n# files = [{listed}]\n# ///\n"


def _script(project: Path, ref: str, source: str = "print('hi')\n") -> None:
    path = project / ".llm-orc" / "scripts" / ref
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)


def _top(project: Path, script: str = "tools/x.py") -> None:
    _ensemble(project / ".llm-orc", "top", [{"name": "run", "script": script}])


def _runnable(client: TestClient) -> dict[str, Any]:
    response = client.get("/api/ensembles/top/runnable")
    assert response.status_code == 200, response.text
    return dict(response.json())


def _rows(data: dict[str, Any]) -> list[tuple[str, str, str]]:
    return [(d["kind"], d["name"], d["status"]) for d in data["dependencies"]]


class TestRunnableRoute:
    def test_a_listed_file_that_is_absent_is_missing_and_blocks_the_run(
        self, client: TestClient, project: Path
    ) -> None:
        _script(project, "tools/x.py", _block("_helpers.py") + "import _helpers\n")
        _top(project)

        data = _runnable(client)

        assert data["runnable"] is False
        assert _rows(data)[1:] == [
            ("script", "tools/x.py", "ready"),
            ("script", "tools/_helpers.py", "missing_script"),
        ]
        missing = data["dependencies"][2]
        assert missing["resolve"] == "ship"
        assert missing["via"] == ["top.run"]

    def test_the_same_listing_with_the_file_beside_the_script_is_ready(
        self, client: TestClient, project: Path
    ) -> None:
        _script(project, "tools/x.py", _block("_helpers.py"))
        _script(project, "tools/_helpers.py")
        _top(project)

        data = _runnable(client)

        assert data["runnable"] is True
        assert _rows(data)[2] == ("script", "tools/_helpers.py", "ready")

    def test_a_file_listed_by_a_listed_file_is_followed(
        self, client: TestClient, project: Path
    ) -> None:
        _script(project, "tools/x.py", _block("lib/a.py"))
        _script(project, "tools/lib/a.py", _block("b.py"))
        _top(project)

        data = _runnable(client)

        assert data["runnable"] is False
        assert _rows(data)[1:] == [
            ("script", "tools/x.py", "ready"),
            ("script", "tools/lib/a.py", "ready"),
            ("script", "tools/lib/b.py", "missing_script"),
        ]

    def test_a_host_file_at_the_same_relative_path_in_another_tier_is_not_it(
        self, client: TestClient, project: Path
    ) -> None:
        """Review focus 1 (Arc 4 ruling 7: no fall-through). The script
        resolves in the project tier; the listed name exists in the
        global tier, where a search-path lookup would find it."""
        _script(project, "tools/x.py", _block("_helpers.py"))
        host_copy = resolve_global_config_dir() / "scripts" / "tools" / "_helpers.py"
        host_copy.parent.mkdir(parents=True)
        host_copy.write_text("print('the host has this')\n")
        _top(project)

        data = _runnable(client)

        assert data["runnable"] is False
        assert _rows(data)[2] == ("script", "tools/_helpers.py", "missing_script")

    def test_a_path_that_leaves_the_scripts_directory_leaves_the_script_unmet(
        self, client: TestClient, project: Path
    ) -> None:
        """Review focus 2: nothing outside the script's directory is read,
        so the listed path is never a row of its own."""
        _script(project, "outside.py")
        outside = project / ".llm-orc" / "scripts" / "outside.py"
        for listed in ("../outside.py", str(outside)):
            _script(project, "tools/x.py", _block(listed))
            _top(project)

            data = _runnable(client)

            assert data["runnable"] is False
            assert _rows(data)[1:] == [("script", "tools/x.py", "missing_script")]
            assert "not a relative path" in data["dependencies"][1]["detail"]

    def test_a_block_that_does_not_parse_leaves_the_script_unmet(
        self, client: TestClient, project: Path
    ) -> None:
        _script(project, "tools/x.py", "# /// llm-orc\n# files = [\n# ///\n")
        _top(project)

        data = _runnable(client)

        assert data["runnable"] is False
        [_, script] = data["dependencies"]
        assert script["status"] == "missing_script"
        assert "TOML" in script["detail"]

    def test_a_first_block_that_never_closes_is_the_error_whatever_follows(
        self, client: TestClient, project: Path
    ) -> None:
        _script(
            project,
            "tools/x.py",
            '# /// llm-orc\n# files = ["_h.py"]\nimport _h\n' + _block("_h.py"),
        )
        _script(project, "tools/_h.py")
        _top(project)

        data = _runnable(client)

        assert data["runnable"] is False
        [_, script] = data["dependencies"]
        assert script["status"] == "missing_script"
        assert "never closed" in script["detail"]

    def test_a_listed_file_beside_the_link_but_not_its_target_is_missing(
        self, client: TestClient, project: Path
    ) -> None:
        """Python puts the script's real directory first on its import
        path, so a symlinked script's listed files are beside its target."""
        target = project / "shared" / "x.py"
        target.parent.mkdir()
        target.write_text(_block("_h.py") + "import _h\n")
        link = project / ".llm-orc" / "scripts" / "tools" / "x.py"
        link.parent.mkdir(parents=True)
        try:
            link.symlink_to(target)
        except OSError:
            pytest.skip("this platform cannot make symlinks")
        _script(project, "tools/_h.py")
        _top(project)

        data = _runnable(client)

        assert data["runnable"] is False
        assert _rows(data)[1:] == [
            ("script", "tools/x.py", "ready"),
            ("script", "tools/_h.py", "missing_script"),
        ]


class TestOneNameTwoSightings:
    """Two sightings of one script name can differ in outcome (a listed
    file is looked for beside its owner only). An unmet sighting wins, in
    either visit order: the row is one, and it blocks the run."""

    def _global_script(self, ref: str, source: str = "print('g')\n") -> None:
        path = resolve_global_config_dir() / "scripts" / ref
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)

    def _agents(self, project: Path, *scripts: str) -> None:
        agents = [{"name": f"a{i}", "script": s} for i, s in enumerate(scripts)]
        _ensemble(project / ".llm-orc", "top", agents)

    def _only_row(self, data: dict[str, Any]) -> dict[str, Any]:
        rows = [d for d in data["dependencies"] if d["name"] == "tools/_h.py"]
        assert len(rows) == 1
        return dict(rows[0])

    def test_a_direct_reference_and_a_listing_that_misses_the_file(
        self, client: TestClient, project: Path
    ) -> None:
        self._global_script("tools/_h.py")
        _script(project, "tools/x.py", _block("_h.py"))
        for order in (("tools/_h.py", "tools/x.py"), ("tools/x.py", "tools/_h.py")):
            self._agents(project, *order)

            data = _runnable(client)

            row = self._only_row(data)
            assert data["runnable"] is False, order
            assert row["status"] == "missing_script", order
            assert "not beside it" in row["detail"], order

    def test_two_owners_listing_one_name_where_only_one_directory_has_it(
        self, client: TestClient, project: Path
    ) -> None:
        _script(project, "tools/x.py", _block("_h.py"))
        self._global_script("tools/y.py", _block("_h.py"))
        self._global_script("tools/_h.py")
        for order in (("tools/x.py", "tools/y.py"), ("tools/y.py", "tools/x.py")):
            self._agents(project, *order)

            data = _runnable(client)

            row = self._only_row(data)
            assert data["runnable"] is False, order
            assert row["status"] == "missing_script", order
            assert "'tools/x.py'" in row["detail"], order

    def test_a_met_sighting_alone_is_still_ready(
        self, client: TestClient, project: Path
    ) -> None:
        self._global_script("tools/_h.py")
        self._agents(project, "tools/_h.py", "tools/_h.py")

        data = _runnable(client)

        assert data["runnable"] is True
        assert self._only_row(data)["status"] == "ready"


class TestExecuteRoute:
    def _request(self, marker: Path, files: dict[str, str]) -> dict[str, Any]:
        return {
            "ensemble": {
                "name": "inline-files",
                "description": "inline",
                "agents": [{"name": "first", "script": "probe/mark.py"}],
            },
            "scripts": {
                "probe/mark.py": _block("_h.py") + _marker_source(marker),
                **files,
            },
            "input": "hi",
        }

    def test_an_injected_script_whose_listed_file_is_not_carried_is_refused(
        self, client: TestClient, project: Path, tmp_path: Path
    ) -> None:
        marker = tmp_path / "marker.txt"

        response = client.post("/api/ensembles/execute", json=self._request(marker, {}))

        refused = response.json()
        assert refused["status"] == "error"
        assert refused["error"]["kind"] == "not_equipped"
        rows = {d["name"]: d["status"] for d in refused["error"]["dependencies"]}
        assert rows["probe/_h.py"] == "missing_script"
        assert not marker.exists()

    def test_the_same_request_carrying_the_file_runs(
        self, client: TestClient, project: Path, tmp_path: Path
    ) -> None:
        marker = tmp_path / "marker.txt"

        response = client.post(
            "/api/ensembles/execute",
            json=self._request(marker, {"probe/_h.py": "X = 1\n"}),
        )

        assert response.json()["status"] == "success", response.text
        assert marker.exists()

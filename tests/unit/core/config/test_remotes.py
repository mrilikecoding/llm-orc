"""Named remotes (Arc 5, Task 6): global config only, resolved by name."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from llm_orc.core.config.config_manager import (
    ConfigurationManager,
    resolve_global_config_dir,
)
from llm_orc.core.config.remotes import RemoteError, resolve_remote

REMOTE_URL = "https://llm-orc.remote.example"


def _write_config(directory: Path, data: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.yaml").write_text(yaml.safe_dump(data))


def _manager(tmp_path: Path) -> ConfigurationManager:
    project = tmp_path / "proj"
    (project / ".llm-orc").mkdir(parents=True, exist_ok=True)
    return ConfigurationManager(project, provision=False)


def _global(data: dict[str, Any]) -> None:
    _write_config(resolve_global_config_dir(), data)


class TestAName:
    def test_resolves_to_its_url(self, tmp_path: Path) -> None:
        _global({"remotes": {"remote-host": {"url": REMOTE_URL}}})

        assert resolve_remote("remote-host", _manager(tmp_path)) == REMOTE_URL

    def test_an_unknown_name_lists_the_known_ones(self, tmp_path: Path) -> None:
        _global(
            {
                "remotes": {
                    "remote-host": {"url": REMOTE_URL},
                    "other-host": {"url": "https://other.example"},
                }
            }
        )

        with pytest.raises(RemoteError) as raised:
            resolve_remote("nope", _manager(tmp_path))

        message = str(raised.value)
        assert "nope" in message
        assert "other-host, remote-host" in message

    def test_an_unknown_name_with_none_configured_says_so(self, tmp_path: Path) -> None:
        with pytest.raises(RemoteError, match="no remotes are configured"):
            resolve_remote("nope", _manager(tmp_path))


class TestAUrl:
    def test_a_value_with_a_scheme_is_a_url_as_given(self, tmp_path: Path) -> None:
        assert resolve_remote(REMOTE_URL, _manager(tmp_path)) == REMOTE_URL

    def test_a_trailing_slash_is_dropped(self, tmp_path: Path) -> None:
        assert resolve_remote(f"{REMOTE_URL}/", _manager(tmp_path)) == REMOTE_URL

    def test_a_named_url_loses_its_trailing_slash_too(self, tmp_path: Path) -> None:
        _global({"remotes": {"remote-host": {"url": f"{REMOTE_URL}/"}}})

        assert resolve_remote("remote-host", _manager(tmp_path)) == REMOTE_URL


class TestOnlyTheGlobalConfigIsRead:
    def test_a_project_remote_does_not_resolve(self, tmp_path: Path) -> None:
        _global({"remotes": {"remote-host": {"url": REMOTE_URL}}})
        manager = _manager(tmp_path)
        _write_config(
            tmp_path / "proj" / ".llm-orc",
            {"remotes": {"project-host": {"url": "https://project.example"}}},
        )

        with pytest.raises(RemoteError) as raised:
            resolve_remote("project-host", manager)

        assert "remote-host" in str(raised.value)
        assert "project-host" not in str(raised.value).split("known")[-1]

    def test_a_project_remote_does_not_override_a_global_one(
        self, tmp_path: Path
    ) -> None:
        _global({"remotes": {"remote-host": {"url": REMOTE_URL}}})
        manager = _manager(tmp_path)
        _write_config(
            tmp_path / "proj" / ".llm-orc",
            {"remotes": {"remote-host": {"url": "https://project.example"}}},
        )

        assert resolve_remote("remote-host", manager) == REMOTE_URL


class TestAMalformedEntry:
    @pytest.mark.parametrize(
        "entry",
        [{}, {"url": 5}, {"url": ""}, "https://llm-orc.remote.example", None],
    )
    def test_names_the_remote(self, tmp_path: Path, entry: Any) -> None:
        _global({"remotes": {"remote-host": entry}})

        with pytest.raises(RemoteError, match="remote-host"):
            resolve_remote("remote-host", _manager(tmp_path))

    def test_one_bad_entry_is_found_even_when_another_is_asked_for(
        self, tmp_path: Path
    ) -> None:
        _global(
            {
                "remotes": {
                    "remote-host": {"url": REMOTE_URL},
                    "broken": {"url": 5},
                }
            }
        )

        with pytest.raises(RemoteError, match="broken"):
            resolve_remote("remote-host", _manager(tmp_path))

    def test_a_remotes_key_that_is_not_a_mapping_is_an_error(
        self, tmp_path: Path
    ) -> None:
        _global({"remotes": ["remote-host"]})

        with pytest.raises(RemoteError, match="remotes"):
            resolve_remote("remote-host", _manager(tmp_path))

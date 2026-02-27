"""Unit tests for ResourceHandler."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_orc.services.handlers.resource_handler import ResourceHandler


@pytest.fixture
def mock_config_manager() -> Any:
    config = MagicMock()
    config.get_ensembles_dirs.return_value = []
    config.get_model_profiles.return_value = {}
    config.global_config_dir = "/fake/.llm-orc"
    return config


@pytest.fixture
def mock_ensemble_loader() -> Any:
    return MagicMock()


@pytest.fixture
def handler(mock_config_manager: Any, mock_ensemble_loader: Any) -> ResourceHandler:
    return ResourceHandler(
        config_manager=mock_config_manager,
        ensemble_loader=mock_ensemble_loader,
    )


def _fake_agent(
    name: str = "agent-1",
    model_profile: str | None = "some-profile",
    depends_on: list[str] | None = None,
) -> Any:
    agent = MagicMock()
    agent.name = name
    agent.model_profile = model_profile
    agent.depends_on = depends_on or []
    return agent


def _fake_ensemble_config(
    name: str = "my-ensemble",
    description: str = "A test ensemble",
    agents: list[Any] | None = None,
) -> Any:
    config = MagicMock()
    config.name = name
    config.description = description
    config.agents = agents if agents is not None else []
    return config


# ---------------------------------------------------------------------------
# read_resource routing (lines 42-61)
# ---------------------------------------------------------------------------


class TestReadResourceRouting:
    """read_resource raises or delegates based on the URI."""

    async def test_invalid_scheme_raises(self, handler: ResourceHandler) -> None:
        """Non llm-orc:// URIs raise ValueError."""
        with pytest.raises(ValueError, match="Invalid URI scheme"):
            await handler.read_resource("http://ensembles")

    async def test_ensembles_delegates(self, handler: ResourceHandler) -> None:
        """llm-orc://ensembles calls read_ensembles."""
        handler.read_ensembles = AsyncMock(return_value=[])  # type: ignore[method-assign]
        result = await handler.read_resource("llm-orc://ensembles")
        handler.read_ensembles.assert_awaited_once()
        assert result == []

    async def test_ensemble_by_name_delegates(self, handler: ResourceHandler) -> None:
        """llm-orc://ensemble/<name> calls read_ensemble with the name."""
        expected: dict[str, Any] = {"name": "foo"}
        handler.read_ensemble = AsyncMock(return_value=expected)  # type: ignore[method-assign]
        result = await handler.read_resource("llm-orc://ensemble/foo")
        handler.read_ensemble.assert_awaited_once_with("foo")
        assert result == expected

    async def test_artifacts_delegates(self, handler: ResourceHandler) -> None:
        """llm-orc://artifacts/<ensemble> calls read_artifacts."""
        handler.read_artifacts = AsyncMock(return_value=[])  # type: ignore[method-assign]
        await handler.read_resource("llm-orc://artifacts/my-ensemble")
        handler.read_artifacts.assert_awaited_once_with("my-ensemble")

    async def test_artifact_by_id_delegates(self, handler: ResourceHandler) -> None:
        """llm-orc://artifact/<ensemble>/<id> calls read_artifact."""
        expected: dict[str, Any] = {"status": "success"}
        handler.read_artifact = AsyncMock(return_value=expected)  # type: ignore[method-assign]
        result = await handler.read_resource("llm-orc://artifact/ens/123")
        handler.read_artifact.assert_awaited_once_with("ens", "123")
        assert result == expected

    async def test_metrics_delegates(self, handler: ResourceHandler) -> None:
        """llm-orc://metrics/<ensemble> calls read_metrics."""
        expected: dict[str, Any] = {"total_executions": 5}
        handler.read_metrics = AsyncMock(return_value=expected)  # type: ignore[method-assign]
        result = await handler.read_resource("llm-orc://metrics/ens")
        handler.read_metrics.assert_awaited_once_with("ens")
        assert result == expected

    async def test_profiles_delegates(self, handler: ResourceHandler) -> None:
        """llm-orc://profiles calls read_profiles."""
        handler.read_profiles = AsyncMock(return_value=[])  # type: ignore[method-assign]
        await handler.read_resource("llm-orc://profiles")
        handler.read_profiles.assert_awaited_once()

    async def test_unknown_path_raises(self, handler: ResourceHandler) -> None:
        """An unrecognised path raises ValueError."""
        with pytest.raises(ValueError, match="Resource not found"):
            await handler.read_resource("llm-orc://unknown-resource")


# ---------------------------------------------------------------------------
# read_ensembles — missing directory skipped (line 74)
# ---------------------------------------------------------------------------


class TestReadEnsemblesSkipsMissingDir:
    """read_ensembles skips directories that do not exist."""

    async def test_skips_nonexistent_dir(
        self, handler: ResourceHandler, mock_config_manager: Any
    ) -> None:
        """Missing ensemble directory produces no entries."""
        mock_config_manager.get_ensembles_dirs.return_value = [
            "/does/not/exist/ensembles"
        ]
        result = await handler.read_ensembles()
        assert result == []

    async def test_swallows_load_error(
        self,
        handler: ResourceHandler,
        mock_config_manager: Any,
        mock_ensemble_loader: Any,
        tmp_path: Path,
    ) -> None:
        """A corrupt YAML file is skipped without raising (lines 93-94)."""
        mock_config_manager.get_ensembles_dirs.return_value = [str(tmp_path)]
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("name: ok\n")

        # Make the loader raise for this file
        mock_ensemble_loader.load_from_file.side_effect = RuntimeError("parse error")

        result = await handler.read_ensembles()
        assert result == []


# ---------------------------------------------------------------------------
# determine_source (lines 110-113)
# ---------------------------------------------------------------------------


class TestDetermineSource:
    """determine_source returns the correct source label."""

    def test_local_when_dot_llm_orc_without_library(
        self, handler: ResourceHandler
    ) -> None:
        """Path containing .llm-orc but not library is 'local'."""
        result = handler.determine_source(Path("/home/user/.llm-orc/ensembles"))
        assert result == "local"

    def test_library_when_library_in_path(self, handler: ResourceHandler) -> None:
        """Path containing 'library' is 'library'."""
        result = handler.determine_source(Path("/home/user/.llm-orc/library/ensembles"))
        assert result == "library"

    def test_global_for_other_paths(self, handler: ResourceHandler) -> None:
        """Path without .llm-orc or library is 'global'."""
        result = handler.determine_source(Path("/usr/share/llm-orc/ensembles"))
        assert result == "global"


# ---------------------------------------------------------------------------
# read_ensemble (lines 127-147)
# ---------------------------------------------------------------------------


class TestReadEnsemble:
    """read_ensemble returns structured config or raises."""

    async def test_returns_ensemble_dict(
        self,
        handler: ResourceHandler,
        mock_config_manager: Any,
        mock_ensemble_loader: Any,
    ) -> None:
        """Found ensemble is returned as a dict with agents list."""
        agent = _fake_agent("worker", model_profile="gpt-4", depends_on=["dep-1"])
        config = _fake_ensemble_config("my-ens", "desc", [agent])
        mock_config_manager.get_ensembles_dirs.return_value = ["/fake/dir"]
        mock_ensemble_loader.find_ensemble.return_value = config

        result = await handler.read_ensemble("my-ens")

        assert result["name"] == "my-ens"
        assert result["description"] == "desc"
        assert len(result["agents"]) == 1
        assert result["agents"][0]["name"] == "worker"
        assert result["agents"][0]["model_profile"] == "gpt-4"
        assert result["agents"][0]["depends_on"] == ["dep-1"]

    async def test_agent_without_model_profile_uses_none(
        self,
        handler: ResourceHandler,
        mock_config_manager: Any,
        mock_ensemble_loader: Any,
    ) -> None:
        """An agent without model_profile attribute gets None."""
        agent = MagicMock(spec=[])  # no model_profile attribute
        agent.name = "bare-agent"
        agent.depends_on = []
        config = _fake_ensemble_config(agents=[agent])
        mock_config_manager.get_ensembles_dirs.return_value = ["/fake"]
        mock_ensemble_loader.find_ensemble.return_value = config

        result = await handler.read_ensemble("bare-ensemble")
        assert result["agents"][0]["model_profile"] is None

    async def test_agent_with_none_depends_on_returns_empty_list(
        self,
        handler: ResourceHandler,
        mock_config_manager: Any,
        mock_ensemble_loader: Any,
    ) -> None:
        """Agent with depends_on=None is normalised to empty list."""
        agent = _fake_agent(depends_on=None)
        agent.depends_on = None
        config = _fake_ensemble_config(agents=[agent])
        mock_config_manager.get_ensembles_dirs.return_value = ["/fake"]
        mock_ensemble_loader.find_ensemble.return_value = config

        result = await handler.read_ensemble("x")
        assert result["agents"][0]["depends_on"] == []

    async def test_raises_when_not_found(
        self,
        handler: ResourceHandler,
        mock_config_manager: Any,
        mock_ensemble_loader: Any,
    ) -> None:
        """Ensemble not found in any dir raises ValueError."""
        mock_config_manager.get_ensembles_dirs.return_value = ["/fake"]
        mock_ensemble_loader.find_ensemble.return_value = None

        with pytest.raises(ValueError, match="Ensemble not found: missing"):
            await handler.read_ensemble("missing")

    async def test_searches_all_dirs_before_raising(
        self,
        handler: ResourceHandler,
        mock_config_manager: Any,
        mock_ensemble_loader: Any,
    ) -> None:
        """All ensemble directories are searched before raising."""
        mock_config_manager.get_ensembles_dirs.return_value = ["/a", "/b", "/c"]
        mock_ensemble_loader.find_ensemble.return_value = None

        with pytest.raises(ValueError, match="x"):
            await handler.read_ensemble("x")

        assert mock_ensemble_loader.find_ensemble.call_count == 3


# ---------------------------------------------------------------------------
# read_artifact (lines 204-211)
# ---------------------------------------------------------------------------


class TestReadArtifact:
    """read_artifact reads execution.json or raises."""

    async def test_returns_parsed_json(
        self, handler: ResourceHandler, tmp_path: Path
    ) -> None:
        """Existing execution.json is parsed and returned."""
        artifact_dir = tmp_path / "my-ens" / "20240101T000000"
        artifact_dir.mkdir(parents=True)
        payload: dict[str, Any] = {"status": "success", "results": {}}
        (artifact_dir / "execution.json").write_text(json.dumps(payload))

        handler.get_artifacts_dir = MagicMock(return_value=tmp_path)  # type: ignore[method-assign]
        result = await handler.read_artifact("my-ens", "20240101T000000")
        assert result["status"] == "success"

    async def test_raises_when_execution_json_missing(
        self, handler: ResourceHandler, tmp_path: Path
    ) -> None:
        """Missing execution.json raises ValueError."""
        handler.get_artifacts_dir = MagicMock(return_value=tmp_path)  # type: ignore[method-assign]

        with pytest.raises(ValueError, match="Artifact not found"):
            await handler.read_artifact("ens", "no-such-id")


# ---------------------------------------------------------------------------
# read_metrics (lines 222-248)
# ---------------------------------------------------------------------------


class TestReadMetrics:
    """read_metrics aggregates artifact statistics."""

    async def test_empty_artifacts_returns_zeros(
        self, handler: ResourceHandler
    ) -> None:
        """No artifacts → all metrics are zero."""
        handler.read_artifacts = AsyncMock(return_value=[])  # type: ignore[method-assign]

        result = await handler.read_metrics("ens")

        assert result["total_executions"] == 0
        assert result["success_rate"] == 0.0
        assert result["avg_duration"] == 0.0
        assert result["avg_cost"] == 0.0

    async def test_success_rate_calculated(self, handler: ResourceHandler) -> None:
        """success_rate is the fraction of artifacts with status 'success'."""
        artifacts: list[dict[str, Any]] = [
            {"status": "success", "duration": 1.0},
            {"status": "error", "duration": 2.0},
            {"status": "success", "duration": 3.0},
        ]
        handler.read_artifacts = AsyncMock(return_value=artifacts)  # type: ignore[method-assign]

        result = await handler.read_metrics("ens")

        assert result["success_rate"] == pytest.approx(2 / 3)
        assert result["total_executions"] == 3

    async def test_avg_duration_numeric(self, handler: ResourceHandler) -> None:
        """avg_duration is the mean of numeric duration values."""
        artifacts: list[dict[str, Any]] = [
            {"status": "success", "duration": 4.0},
            {"status": "success", "duration": 6.0},
        ]
        handler.read_artifacts = AsyncMock(return_value=artifacts)  # type: ignore[method-assign]

        result = await handler.read_metrics("ens")

        assert result["avg_duration"] == pytest.approx(5.0)

    async def test_avg_duration_string_seconds(self, handler: ResourceHandler) -> None:
        """Duration values like '3.5s' are parsed to float."""
        artifacts: list[dict[str, Any]] = [
            {"status": "success", "duration": "2.0s"},
            {"status": "success", "duration": "4.0s"},
        ]
        handler.read_artifacts = AsyncMock(return_value=artifacts)  # type: ignore[method-assign]

        result = await handler.read_metrics("ens")

        assert result["avg_duration"] == pytest.approx(3.0)

    async def test_none_duration_treated_as_zero(
        self, handler: ResourceHandler
    ) -> None:
        """A None duration is treated as 0.0 when computing the average."""
        artifacts: list[dict[str, Any]] = [
            {"status": "success", "duration": None},
            {"status": "success", "duration": 4.0},
        ]
        handler.read_artifacts = AsyncMock(return_value=artifacts)  # type: ignore[method-assign]

        result = await handler.read_metrics("ens")

        assert result["avg_duration"] == pytest.approx(2.0)

    async def test_unparseable_string_duration_treated_as_zero(
        self, handler: ResourceHandler
    ) -> None:
        """A string duration without a trailing 's' is treated as 0.0."""
        artifacts: list[dict[str, Any]] = [
            {"status": "success", "duration": "not-a-number"},
        ]
        handler.read_artifacts = AsyncMock(return_value=artifacts)  # type: ignore[method-assign]

        result = await handler.read_metrics("ens")

        assert result["avg_duration"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# read_profiles (lines 261-273)
# ---------------------------------------------------------------------------


class TestReadProfiles:
    """read_profiles returns a list of profile dicts."""

    async def test_returns_profiles_list(
        self, handler: ResourceHandler, mock_config_manager: Any
    ) -> None:
        """Each profile entry has name, provider, model."""
        mock_config_manager.get_model_profiles.return_value = {
            "gpt4": {"provider": "openai", "model": "gpt-4"},
            "llama3": {"provider": "ollama", "model": "llama3"},
        }

        result = await handler.read_profiles()

        names = {p["name"] for p in result}
        assert names == {"gpt4", "llama3"}
        gpt4 = next(p for p in result if p["name"] == "gpt4")
        assert gpt4["provider"] == "openai"
        assert gpt4["model"] == "gpt-4"

    async def test_missing_provider_defaults_to_unknown(
        self, handler: ResourceHandler, mock_config_manager: Any
    ) -> None:
        """A profile dict without 'provider' key defaults to 'unknown'."""
        mock_config_manager.get_model_profiles.return_value = {
            "bare": {"model": "x"},
        }

        result = await handler.read_profiles()

        assert result[0]["provider"] == "unknown"

    async def test_missing_model_defaults_to_unknown(
        self, handler: ResourceHandler, mock_config_manager: Any
    ) -> None:
        """A profile dict without 'model' key defaults to 'unknown'."""
        mock_config_manager.get_model_profiles.return_value = {
            "bare": {"provider": "ollama"},
        }

        result = await handler.read_profiles()

        assert result[0]["model"] == "unknown"

    async def test_empty_profiles_returns_empty_list(
        self, handler: ResourceHandler, mock_config_manager: Any
    ) -> None:
        """No profiles → empty list."""
        mock_config_manager.get_model_profiles.return_value = {}

        result = await handler.read_profiles()

        assert result == []


# ---------------------------------------------------------------------------
# get_artifacts_dir (lines 281-293)
# ---------------------------------------------------------------------------


class TestGetArtifactsDir:
    """get_artifacts_dir resolves the correct artifacts directory."""

    def test_returns_global_config_dir_when_named_artifacts_and_exists(
        self, handler: ResourceHandler, tmp_path: Path, mock_config_manager: Any
    ) -> None:
        """If global_config_dir is itself named 'artifacts' and exists, return it."""
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        mock_config_manager.global_config_dir = str(artifacts_dir)

        result = handler.get_artifacts_dir()

        assert result == artifacts_dir

    def test_returns_local_artifacts_when_exists(
        self, handler: ResourceHandler, tmp_path: Path, mock_config_manager: Any
    ) -> None:
        """If .llm-orc/artifacts exists under cwd, return that path."""
        local_artifacts = tmp_path / ".llm-orc" / "artifacts"
        local_artifacts.mkdir(parents=True)
        mock_config_manager.global_config_dir = str(tmp_path / "global")

        patch_target = "llm_orc.services.handlers.resource_handler.Path.cwd"
        with patch(patch_target, return_value=tmp_path):
            result = handler.get_artifacts_dir()

        assert result == local_artifacts

    def test_returns_global_artifacts_when_exists(
        self, handler: ResourceHandler, tmp_path: Path, mock_config_manager: Any
    ) -> None:
        """If global artifacts dir exists (but local does not), return global."""
        global_dir = tmp_path / "global"
        global_artifacts = global_dir / "artifacts"
        global_artifacts.mkdir(parents=True)
        mock_config_manager.global_config_dir = str(global_dir)

        # Ensure local path does not exist
        non_cwd = tmp_path / "project"
        non_cwd.mkdir()

        patch_target = "llm_orc.services.handlers.resource_handler.Path.cwd"
        with patch(patch_target, return_value=non_cwd):
            result = handler.get_artifacts_dir()

        assert result == global_artifacts

    def test_falls_back_to_local_artifacts_when_nothing_exists(
        self, handler: ResourceHandler, tmp_path: Path, mock_config_manager: Any
    ) -> None:
        """When nothing exists, fall back to <cwd>/.llm-orc/artifacts."""
        mock_config_manager.global_config_dir = str(tmp_path / "global")
        non_cwd = tmp_path / "empty"
        non_cwd.mkdir()

        patch_target = "llm_orc.services.handlers.resource_handler.Path.cwd"
        with patch(patch_target, return_value=non_cwd):
            result = handler.get_artifacts_dir()

        assert result == non_cwd / ".llm-orc" / "artifacts"


# -------------------------------------------------------------------
# read_artifacts — lines 158-187
# -------------------------------------------------------------------


class TestReadArtifacts:
    """Tests for listing artifacts of an ensemble."""

    async def test_returns_empty_when_dir_missing(
        self,
        handler: ResourceHandler,
        tmp_path: Path,
        mock_config_manager: Any,
    ) -> None:
        """Returns [] when the ensemble artifacts dir doesn't exist."""
        mock_config_manager.global_config_dir = str(tmp_path)
        patch_target = "llm_orc.services.handlers.resource_handler.Path.cwd"
        with patch(patch_target, return_value=tmp_path):
            result = await handler.read_artifacts("no-such")

        assert result == []

    async def test_lists_valid_artifact(
        self,
        handler: ResourceHandler,
        tmp_path: Path,
        mock_config_manager: Any,
    ) -> None:
        """Parses execution.json and returns artifact metadata."""
        arts = tmp_path / ".llm-orc" / "artifacts" / "my-ens"
        run_dir = arts / "20240101_120000"
        run_dir.mkdir(parents=True)
        execution = {
            "metadata": {
                "started_at": "2024-01-01T12:00:00",
                "duration": 3.5,
                "agents_used": 2,
            },
            "status": "completed",
        }
        (run_dir / "execution.json").write_text(json.dumps(execution))
        mock_config_manager.global_config_dir = str(tmp_path / "global")
        patch_target = "llm_orc.services.handlers.resource_handler.Path.cwd"
        with patch(patch_target, return_value=tmp_path):
            result = await handler.read_artifacts("my-ens")

        assert len(result) == 1
        assert result[0]["id"] == "20240101_120000"
        assert result[0]["status"] == "completed"
        assert result[0]["duration"] == 3.5

    async def test_skips_non_dir_entries(
        self,
        handler: ResourceHandler,
        tmp_path: Path,
        mock_config_manager: Any,
    ) -> None:
        """Non-directory entries are skipped."""
        arts = tmp_path / ".llm-orc" / "artifacts" / "my-ens"
        arts.mkdir(parents=True)
        (arts / "stray-file.txt").write_text("ignore me")
        mock_config_manager.global_config_dir = str(tmp_path / "global")
        patch_target = "llm_orc.services.handlers.resource_handler.Path.cwd"
        with patch(patch_target, return_value=tmp_path):
            result = await handler.read_artifacts("my-ens")

        assert result == []

    async def test_skips_dir_without_execution_json(
        self,
        handler: ResourceHandler,
        tmp_path: Path,
        mock_config_manager: Any,
    ) -> None:
        """Dirs without execution.json are skipped."""
        arts = tmp_path / ".llm-orc" / "artifacts" / "my-ens"
        (arts / "empty-run").mkdir(parents=True)
        mock_config_manager.global_config_dir = str(tmp_path / "global")
        patch_target = "llm_orc.services.handlers.resource_handler.Path.cwd"
        with patch(patch_target, return_value=tmp_path):
            result = await handler.read_artifacts("my-ens")

        assert result == []

    async def test_swallows_corrupt_json(
        self,
        handler: ResourceHandler,
        tmp_path: Path,
        mock_config_manager: Any,
    ) -> None:
        """Corrupt execution.json is skipped without raising."""
        arts = tmp_path / ".llm-orc" / "artifacts" / "my-ens"
        run_dir = arts / "bad-run"
        run_dir.mkdir(parents=True)
        (run_dir / "execution.json").write_text("{invalid")
        mock_config_manager.global_config_dir = str(tmp_path / "global")
        patch_target = "llm_orc.services.handlers.resource_handler.Path.cwd"
        with patch(patch_target, return_value=tmp_path):
            result = await handler.read_artifacts("my-ens")

        assert result == []


# -------------------------------------------------------------------
# read_metrics parse_duration ValueError — lines 242-243
# -------------------------------------------------------------------


class TestParseDurationValueError:
    """Covers the ValueError branch in parse_duration."""

    async def test_unparseable_duration_string(
        self,
        handler: ResourceHandler,
        tmp_path: Path,
        mock_config_manager: Any,
    ) -> None:
        """Duration string like 'not-a-numbers' returns 0."""
        arts = tmp_path / ".llm-orc" / "artifacts" / "my-ens"
        run_dir = arts / "run-1"
        run_dir.mkdir(parents=True)
        execution = {
            "metadata": {
                "started_at": "t",
                "duration": "not-a-numbers",
                "agents_used": 1,
            },
            "results": {
                "a": {"status": "success"},
            },
            "status": "completed",
        }
        (run_dir / "execution.json").write_text(json.dumps(execution))
        mock_config_manager.global_config_dir = str(tmp_path / "global")
        patch_target = "llm_orc.services.handlers.resource_handler.Path.cwd"
        with patch(patch_target, return_value=tmp_path):
            result = await handler.read_metrics("my-ens")

        assert result["avg_duration"] == 0.0

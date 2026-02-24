"""Unit tests for PromotionHandler."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
import yaml

from llm_orc.mcp.server import MCPServer


def _mock_config(server: MCPServer) -> Any:
    """Get config_manager as mock (for test setup)."""
    return cast(Any, server.config_manager)


@pytest.fixture
def mock_config_manager() -> Any:
    """Create mock config manager."""
    config = MagicMock()
    config.get_ensembles_dirs.return_value = []
    config.get_profiles_dirs.return_value = []
    config.global_config_dir = Path("/tmp/global-config")
    return config


@pytest.fixture
def server(mock_config_manager: Any) -> MCPServer:
    """Create MCPServer instance with mocked dependencies."""
    return MCPServer(config_manager=mock_config_manager)


def _setup_local_ensemble(
    tmp_path: Path,
    server: MCPServer,
    name: str = "test-ensemble",
    agents: list[dict[str, Any]] | None = None,
) -> Path:
    """Create a local ensemble YAML and configure server dirs."""
    if agents is None:
        agents = [{"name": "agent1", "model_profile": "fast-profile"}]

    local_dir = tmp_path / ".llm-orc"
    ensembles_dir = local_dir / "ensembles"
    ensembles_dir.mkdir(parents=True, exist_ok=True)

    ensemble_data = {
        "name": name,
        "description": f"Test ensemble {name}",
        "agents": agents,
    }
    ensemble_file = ensembles_dir / f"{name}.yaml"
    ensemble_file.write_text(yaml.safe_dump(ensemble_data, default_flow_style=False))

    _mock_config(server).get_ensembles_dirs.return_value = [str(ensembles_dir)]
    _mock_config(server).classify_tier.side_effect = lambda p: (
        "local" if Path(p).is_relative_to(local_dir) else "global"
    )
    return ensembles_dir


def _setup_local_profile(
    tmp_path: Path,
    server: MCPServer,
    name: str = "fast-profile",
    provider: str = "ollama",
    model: str = "qwen3:0.6b",
) -> Path:
    """Create a local profile YAML and configure server dirs."""
    profiles_dir = tmp_path / ".llm-orc" / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)

    profile_data = {"name": name, "provider": provider, "model": model}
    (profiles_dir / f"{name}.yaml").write_text(
        yaml.safe_dump(profile_data, default_flow_style=False)
    )

    _mock_config(server).get_profiles_dirs.return_value = [str(profiles_dir)]
    return profiles_dir


def _setup_global_dirs(tmp_path: Path, server: MCPServer) -> tuple[Path, Path]:
    """Create global tier directories and configure server."""
    global_dir = tmp_path / "global-config"
    global_ensembles = global_dir / "ensembles"
    global_profiles = global_dir / "profiles"
    global_ensembles.mkdir(parents=True)
    global_profiles.mkdir(parents=True)

    _mock_config(server).global_config_dir = global_dir
    return global_ensembles, global_profiles


# ==============================================================================
# promote_ensemble tests
# ==============================================================================


class TestPromoteEnsemble:
    """Tests for promote_ensemble tool."""

    @pytest.mark.asyncio
    async def test_promote_requires_ensemble_name(self, server: MCPServer) -> None:
        """Promote requires ensemble_name."""
        with pytest.raises(ValueError, match="ensemble_name is required"):
            await server.call_tool(
                "promote_ensemble",
                {"destination": "global"},
            )

    @pytest.mark.asyncio
    async def test_promote_requires_valid_destination(self, server: MCPServer) -> None:
        """Promote requires valid destination."""
        with pytest.raises(ValueError, match="destination must be"):
            await server.call_tool(
                "promote_ensemble",
                {"ensemble_name": "test", "destination": "invalid"},
            )

    @pytest.mark.asyncio
    async def test_promote_raises_if_ensemble_not_found(
        self, server: MCPServer
    ) -> None:
        """Promote raises if ensemble doesn't exist."""
        _mock_config(server).get_ensembles_dirs.return_value = []

        with pytest.raises(ValueError, match="Ensemble not found"):
            await server.call_tool(
                "promote_ensemble",
                {"ensemble_name": "nonexistent", "destination": "global"},
            )

    @pytest.mark.asyncio
    async def test_promote_dry_run_returns_preview(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Dry run returns preview without copying."""
        ensembles_dir = _setup_local_ensemble(tmp_path, server)
        _setup_local_profile(tmp_path, server)
        global_ensembles, _ = _setup_global_dirs(tmp_path, server)

        # Set both ensembles and profiles dirs
        _mock_config(server).get_ensembles_dirs.return_value = [str(ensembles_dir)]
        _mock_config(server).get_profiles_dirs.return_value = [
            str(tmp_path / ".llm-orc" / "profiles")
        ]

        result = await server.call_tool(
            "promote_ensemble",
            {
                "ensemble_name": "test-ensemble",
                "destination": "global",
                "dry_run": True,
            },
        )

        assert result["status"] == "dry_run"
        assert result["ensemble"] == "test-ensemble"
        assert result["source_tier"] == "local"
        assert result["destination"] == "global"
        assert "fast-profile" in result["would_copy"]["profiles"]
        # File should NOT be copied
        assert not (global_ensembles / "test-ensemble.yaml").exists()

    @pytest.mark.asyncio
    async def test_promote_actually_copies_files(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Promote with dry_run=False copies ensemble and profiles."""
        ensembles_dir = _setup_local_ensemble(tmp_path, server)
        profiles_dir = _setup_local_profile(tmp_path, server)
        global_ensembles, global_profiles = _setup_global_dirs(tmp_path, server)

        _mock_config(server).get_ensembles_dirs.return_value = [str(ensembles_dir)]
        _mock_config(server).get_profiles_dirs.return_value = [str(profiles_dir)]

        result = await server.call_tool(
            "promote_ensemble",
            {
                "ensemble_name": "test-ensemble",
                "destination": "global",
                "dry_run": False,
            },
        )

        assert result["status"] == "promoted"
        assert result["profiles_copied"] == ["fast-profile"]
        assert (global_ensembles / "test-ensemble.yaml").exists()
        assert (global_profiles / "fast-profile.yaml").exists()

    @pytest.mark.asyncio
    async def test_promote_skips_profiles_already_at_destination(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Profiles already at destination are not copied again."""
        ensembles_dir = _setup_local_ensemble(tmp_path, server)
        profiles_dir = _setup_local_profile(tmp_path, server)
        global_ensembles, global_profiles = _setup_global_dirs(tmp_path, server)

        # Pre-create profile at global tier
        (global_profiles / "fast-profile.yaml").write_text(
            yaml.safe_dump(
                {"name": "fast-profile", "provider": "ollama", "model": "qwen3:0.6b"}
            )
        )

        _mock_config(server).get_ensembles_dirs.return_value = [str(ensembles_dir)]
        _mock_config(server).get_profiles_dirs.return_value = [str(profiles_dir)]

        result = await server.call_tool(
            "promote_ensemble",
            {
                "ensemble_name": "test-ensemble",
                "destination": "global",
                "dry_run": False,
            },
        )

        assert result["status"] == "promoted"
        assert result["profiles_copied"] == []
        assert result["profiles_already_present"] == ["fast-profile"]

    @pytest.mark.asyncio
    async def test_promote_raises_if_already_exists_no_overwrite(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Promote raises if ensemble exists at destination without overwrite."""
        ensembles_dir = _setup_local_ensemble(tmp_path, server)
        _setup_local_profile(tmp_path, server)
        global_ensembles, _ = _setup_global_dirs(tmp_path, server)

        # Pre-create ensemble at global tier
        (global_ensembles / "test-ensemble.yaml").write_text("name: existing")

        _mock_config(server).get_ensembles_dirs.return_value = [str(ensembles_dir)]

        with pytest.raises(ValueError, match="already exists at global"):
            await server.call_tool(
                "promote_ensemble",
                {
                    "ensemble_name": "test-ensemble",
                    "destination": "global",
                    "dry_run": False,
                    "overwrite": False,
                },
            )

    @pytest.mark.asyncio
    async def test_promote_with_overwrite_replaces_existing(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Promote with overwrite replaces existing ensemble."""
        ensembles_dir = _setup_local_ensemble(tmp_path, server)
        profiles_dir = _setup_local_profile(tmp_path, server)
        global_ensembles, _ = _setup_global_dirs(tmp_path, server)

        (global_ensembles / "test-ensemble.yaml").write_text("name: old-version")

        _mock_config(server).get_ensembles_dirs.return_value = [str(ensembles_dir)]
        _mock_config(server).get_profiles_dirs.return_value = [str(profiles_dir)]

        result = await server.call_tool(
            "promote_ensemble",
            {
                "ensemble_name": "test-ensemble",
                "destination": "global",
                "dry_run": False,
                "overwrite": True,
            },
        )

        assert result["status"] == "promoted"
        content = (global_ensembles / "test-ensemble.yaml").read_text()
        assert "old-version" not in content

    @pytest.mark.asyncio
    async def test_promote_raises_on_broken_profile_reference(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Promote raises if ensemble references a profile that doesn't exist."""
        _setup_local_ensemble(
            tmp_path,
            server,
            agents=[{"name": "agent1", "model_profile": "nonexistent-profile"}],
        )
        _setup_global_dirs(tmp_path, server)
        # No profiles dir set — profile won't be found
        _mock_config(server).get_profiles_dirs.return_value = []

        with pytest.raises(ValueError, match="Broken profile references"):
            await server.call_tool(
                "promote_ensemble",
                {
                    "ensemble_name": "test-ensemble",
                    "destination": "global",
                    "dry_run": False,
                },
            )

    @pytest.mark.asyncio
    async def test_promote_without_profiles(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Promote with include_profiles=False skips profile copying."""
        ensembles_dir = _setup_local_ensemble(tmp_path, server)
        global_ensembles, global_profiles = _setup_global_dirs(tmp_path, server)

        _mock_config(server).get_ensembles_dirs.return_value = [str(ensembles_dir)]

        result = await server.call_tool(
            "promote_ensemble",
            {
                "ensemble_name": "test-ensemble",
                "destination": "global",
                "dry_run": False,
                "include_profiles": False,
            },
        )

        assert result["status"] == "promoted"
        assert result["profiles_copied"] == []
        assert (global_ensembles / "test-ensemble.yaml").exists()
        # No profiles copied
        assert list(global_profiles.glob("*.yaml")) == []

    @pytest.mark.asyncio
    async def test_promote_to_library(self, server: MCPServer, tmp_path: Path) -> None:
        """Promote to library tier writes to library directory."""
        ensembles_dir = _setup_local_ensemble(tmp_path, server)
        profiles_dir = _setup_local_profile(tmp_path, server)

        library_dir = tmp_path / "llm-orchestra-library"
        library_dir.mkdir()
        server._library_handler._library_dir = library_dir

        _mock_config(server).get_ensembles_dirs.return_value = [str(ensembles_dir)]
        _mock_config(server).get_profiles_dirs.return_value = [str(profiles_dir)]

        result = await server.call_tool(
            "promote_ensemble",
            {
                "ensemble_name": "test-ensemble",
                "destination": "library",
                "dry_run": False,
            },
        )

        assert result["status"] == "promoted"
        assert result["destination"] == "library"
        assert (library_dir / "ensembles" / "test-ensemble.yaml").exists()
        assert (library_dir / "profiles" / "fast-profile.yaml").exists()


# ==============================================================================
# list_dependencies tests
# ==============================================================================


class TestListDependencies:
    """Tests for list_dependencies tool."""

    @pytest.mark.asyncio
    async def test_list_deps_requires_ensemble_name(self, server: MCPServer) -> None:
        """List dependencies requires ensemble_name."""
        with pytest.raises(ValueError, match="ensemble_name is required"):
            await server.call_tool("list_dependencies", {})

    @pytest.mark.asyncio
    async def test_list_deps_raises_if_not_found(self, server: MCPServer) -> None:
        """List dependencies raises if ensemble not found."""
        _mock_config(server).get_ensembles_dirs.return_value = []

        with pytest.raises(ValueError, match="Ensemble not found"):
            await server.call_tool(
                "list_dependencies", {"ensemble_name": "nonexistent"}
            )

    @pytest.mark.asyncio
    async def test_list_deps_returns_agent_info(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """List dependencies returns per-agent dependency info."""
        agents = [
            {"name": "extractor", "model_profile": "fast-profile"},
            {"name": "synth", "model_profile": "medium-profile"},
        ]
        ensembles_dir = _setup_local_ensemble(tmp_path, server, agents=agents)
        profiles_dir = _setup_local_profile(tmp_path, server, "fast-profile")

        # Add second profile
        (profiles_dir / "medium-profile.yaml").write_text(
            yaml.safe_dump(
                {"name": "medium-profile", "provider": "ollama", "model": "gemma3:4b"}
            )
        )

        _mock_config(server).get_ensembles_dirs.return_value = [str(ensembles_dir)]
        _mock_config(server).get_profiles_dirs.return_value = [str(profiles_dir)]

        # Mock ollama as unavailable to simplify
        server._provider_handler._test_ollama_status = {
            "available": False,
            "models": [],
            "reason": "test",
        }

        result = await server.call_tool(
            "list_dependencies", {"ensemble_name": "test-ensemble"}
        )

        assert result["ensemble"] == "test-ensemble"
        assert result["source_tier"] == "local"
        assert len(result["agents"]) == 2
        assert sorted(result["profiles_needed"]) == [
            "fast-profile",
            "medium-profile",
        ]
        assert "ollama" in result["providers_needed"]

    @pytest.mark.asyncio
    async def test_list_deps_handles_script_agents(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Script agents are reported without profile info."""
        agents = [
            {"name": "aggregator", "script": "aggregator.py"},
            {"name": "extractor", "model_profile": "fast-profile"},
        ]
        ensembles_dir = _setup_local_ensemble(tmp_path, server, agents=agents)
        profiles_dir = _setup_local_profile(tmp_path, server)

        _mock_config(server).get_ensembles_dirs.return_value = [str(ensembles_dir)]
        _mock_config(server).get_profiles_dirs.return_value = [str(profiles_dir)]

        server._provider_handler._test_ollama_status = {
            "available": False,
            "models": [],
            "reason": "test",
        }

        result = await server.call_tool(
            "list_dependencies", {"ensemble_name": "test-ensemble"}
        )

        script_agent = next(a for a in result["agents"] if a["name"] == "aggregator")
        assert script_agent["type"] == "script"
        assert "model_profile" not in script_agent

    @pytest.mark.asyncio
    async def test_list_deps_reports_missing_profiles(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Missing profiles are reported as not found."""
        agents = [{"name": "agent1", "model_profile": "nonexistent"}]
        ensembles_dir = _setup_local_ensemble(tmp_path, server, agents=agents)

        _mock_config(server).get_ensembles_dirs.return_value = [str(ensembles_dir)]
        _mock_config(server).get_profiles_dirs.return_value = []

        server._provider_handler._test_ollama_status = {
            "available": False,
            "models": [],
            "reason": "test",
        }

        result = await server.call_tool(
            "list_dependencies", {"ensemble_name": "test-ensemble"}
        )

        agent_info = result["agents"][0]
        assert agent_info["profile_found"] is False
        assert agent_info["model_available"] is False


# ==============================================================================
# check_promotion_readiness tests
# ==============================================================================


class TestCheckPromotionReadiness:
    """Tests for check_promotion_readiness tool."""

    @pytest.mark.asyncio
    async def test_readiness_requires_ensemble_name(self, server: MCPServer) -> None:
        """Check readiness requires ensemble_name."""
        with pytest.raises(ValueError, match="ensemble_name is required"):
            await server.call_tool(
                "check_promotion_readiness", {"destination": "global"}
            )

    @pytest.mark.asyncio
    async def test_readiness_requires_valid_destination(
        self, server: MCPServer
    ) -> None:
        """Check readiness requires valid destination."""
        with pytest.raises(ValueError, match="destination must be"):
            await server.call_tool(
                "check_promotion_readiness",
                {"ensemble_name": "test", "destination": "invalid"},
            )

    @pytest.mark.asyncio
    async def test_readiness_raises_if_not_found(self, server: MCPServer) -> None:
        """Check readiness raises if ensemble not found."""
        _mock_config(server).get_ensembles_dirs.return_value = []

        with pytest.raises(ValueError, match="Ensemble not found"):
            await server.call_tool(
                "check_promotion_readiness",
                {"ensemble_name": "nonexistent", "destination": "global"},
            )

    @pytest.mark.asyncio
    async def test_readiness_reports_missing_profiles(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Reports profiles missing at destination."""
        ensembles_dir = _setup_local_ensemble(tmp_path, server)
        profiles_dir = _setup_local_profile(tmp_path, server)
        _setup_global_dirs(tmp_path, server)

        _mock_config(server).get_ensembles_dirs.return_value = [str(ensembles_dir)]
        _mock_config(server).get_profiles_dirs.return_value = [str(profiles_dir)]

        server._provider_handler._test_ollama_status = {
            "available": True,
            "models": ["qwen3:0.6b"],
            "model_count": 1,
        }

        result = await server.call_tool(
            "check_promotion_readiness",
            {"ensemble_name": "test-ensemble", "destination": "global"},
        )

        assert result["ready"] is True  # missing_profile is not blocking
        assert result["profiles_to_copy"] == ["fast-profile"]
        missing_issue = next(
            i for i in result["issues"] if i["type"] == "missing_profile"
        )
        assert "fast-profile" in missing_issue["detail"]

    @pytest.mark.asyncio
    async def test_readiness_reports_broken_references(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Reports broken profile references as not ready."""
        agents = [{"name": "agent1", "model_profile": "nonexistent"}]
        ensembles_dir = _setup_local_ensemble(tmp_path, server, agents=agents)
        _setup_global_dirs(tmp_path, server)

        _mock_config(server).get_ensembles_dirs.return_value = [str(ensembles_dir)]
        _mock_config(server).get_profiles_dirs.return_value = []

        server._provider_handler._test_ollama_status = {
            "available": False,
            "models": [],
            "reason": "test",
        }

        result = await server.call_tool(
            "check_promotion_readiness",
            {"ensemble_name": "test-ensemble", "destination": "global"},
        )

        assert result["ready"] is False
        broken = [i for i in result["issues"] if i["type"] == "broken_reference"]
        assert len(broken) == 1

    @pytest.mark.asyncio
    async def test_readiness_reports_unavailable_provider(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Reports unavailable provider as not ready."""
        ensembles_dir = _setup_local_ensemble(tmp_path, server)
        profiles_dir = _setup_local_profile(tmp_path, server)
        _setup_global_dirs(tmp_path, server)

        _mock_config(server).get_ensembles_dirs.return_value = [str(ensembles_dir)]
        _mock_config(server).get_profiles_dirs.return_value = [str(profiles_dir)]

        # Ollama not available
        server._provider_handler._test_ollama_status = {
            "available": False,
            "models": [],
            "reason": "not running",
        }

        result = await server.call_tool(
            "check_promotion_readiness",
            {"ensemble_name": "test-ensemble", "destination": "global"},
        )

        assert result["ready"] is False
        provider_issue = next(
            i for i in result["issues"] if i["type"] == "provider_unavailable"
        )
        assert "ollama" in provider_issue["detail"]

    @pytest.mark.asyncio
    async def test_readiness_reports_missing_model(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Reports missing Ollama model as issue."""
        ensembles_dir = _setup_local_ensemble(tmp_path, server)
        profiles_dir = _setup_local_profile(tmp_path, server)
        _setup_global_dirs(tmp_path, server)

        _mock_config(server).get_ensembles_dirs.return_value = [str(ensembles_dir)]
        _mock_config(server).get_profiles_dirs.return_value = [str(profiles_dir)]

        # Ollama available but model not installed
        server._provider_handler._test_ollama_status = {
            "available": True,
            "models": ["llama3:latest"],  # Not qwen3:0.6b
            "model_count": 1,
        }

        result = await server.call_tool(
            "check_promotion_readiness",
            {"ensemble_name": "test-ensemble", "destination": "global"},
        )

        model_issue = next(
            (i for i in result["issues"] if i["type"] == "model_unavailable"),
            None,
        )
        assert model_issue is not None
        assert "qwen3:0.6b" in model_issue["detail"]

    @pytest.mark.asyncio
    async def test_readiness_detects_existing_at_destination(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Reports if ensemble already exists at destination."""
        ensembles_dir = _setup_local_ensemble(tmp_path, server)
        profiles_dir = _setup_local_profile(tmp_path, server)
        global_ensembles, _ = _setup_global_dirs(tmp_path, server)

        # Pre-create at global
        (global_ensembles / "test-ensemble.yaml").write_text("name: existing")

        _mock_config(server).get_ensembles_dirs.return_value = [str(ensembles_dir)]
        _mock_config(server).get_profiles_dirs.return_value = [str(profiles_dir)]

        server._provider_handler._test_ollama_status = {
            "available": True,
            "models": ["qwen3:0.6b"],
            "model_count": 1,
        }

        result = await server.call_tool(
            "check_promotion_readiness",
            {"ensemble_name": "test-ensemble", "destination": "global"},
        )

        assert result["already_exists"] is True

    @pytest.mark.asyncio
    async def test_readiness_all_deps_met(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Ready when all dependencies are met at destination."""
        ensembles_dir = _setup_local_ensemble(tmp_path, server)
        profiles_dir = _setup_local_profile(tmp_path, server)
        _, global_profiles = _setup_global_dirs(tmp_path, server)

        # Profile already at global
        (global_profiles / "fast-profile.yaml").write_text(
            yaml.safe_dump(
                {"name": "fast-profile", "provider": "ollama", "model": "qwen3:0.6b"}
            )
        )

        _mock_config(server).get_ensembles_dirs.return_value = [str(ensembles_dir)]
        _mock_config(server).get_profiles_dirs.return_value = [str(profiles_dir)]

        server._provider_handler._test_ollama_status = {
            "available": True,
            "models": ["qwen3:0.6b"],
            "model_count": 1,
        }

        result = await server.call_tool(
            "check_promotion_readiness",
            {"ensemble_name": "test-ensemble", "destination": "global"},
        )

        assert result["ready"] is True
        assert result["profiles_already_present"] == ["fast-profile"]
        assert result["profiles_to_copy"] == []


# ==============================================================================
# demote_ensemble tests
# ==============================================================================


class TestDemoteEnsemble:
    """Tests for demote_ensemble tool."""

    @pytest.mark.asyncio
    async def test_demote_requires_ensemble_name(self, server: MCPServer) -> None:
        """Demote requires ensemble_name."""
        with pytest.raises(ValueError, match="ensemble_name is required"):
            await server.call_tool(
                "demote_ensemble", {"tier": "global", "confirm": True}
            )

    @pytest.mark.asyncio
    async def test_demote_requires_valid_tier(self, server: MCPServer) -> None:
        """Demote requires valid tier."""
        with pytest.raises(ValueError, match="tier must be"):
            await server.call_tool(
                "demote_ensemble",
                {"ensemble_name": "test", "tier": "invalid", "confirm": True},
            )

    @pytest.mark.asyncio
    async def test_demote_raises_if_not_found(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Demote raises if ensemble not at the specified tier."""
        _setup_global_dirs(tmp_path, server)

        with pytest.raises(ValueError, match="not found at global"):
            await server.call_tool(
                "demote_ensemble",
                {"ensemble_name": "missing", "tier": "global", "confirm": True},
            )

    @pytest.mark.asyncio
    async def test_demote_preview_without_confirm(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Demote without confirm returns preview."""
        global_ensembles, _ = _setup_global_dirs(tmp_path, server)
        (global_ensembles / "test-ensemble.yaml").write_text("name: test-ensemble")

        # Need to set up so _get_ensemble_profiles works
        _mock_config(server).get_ensembles_dirs.return_value = [str(global_ensembles)]

        result = await server.call_tool(
            "demote_ensemble",
            {
                "ensemble_name": "test-ensemble",
                "tier": "global",
                "confirm": False,
            },
        )

        assert result["status"] == "preview"
        assert "test-ensemble" in result["would_remove"]["ensemble"]
        # File still exists
        assert (global_ensembles / "test-ensemble.yaml").exists()

    @pytest.mark.asyncio
    async def test_demote_deletes_ensemble(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Demote with confirm deletes the ensemble file."""
        global_ensembles, _ = _setup_global_dirs(tmp_path, server)
        ensemble_file = global_ensembles / "test-ensemble.yaml"
        ensemble_file.write_text(
            yaml.safe_dump(
                {
                    "name": "test-ensemble",
                    "agents": [{"name": "a1", "model_profile": "p1"}],
                }
            )
        )

        _mock_config(server).get_ensembles_dirs.return_value = [str(global_ensembles)]

        result = await server.call_tool(
            "demote_ensemble",
            {
                "ensemble_name": "test-ensemble",
                "tier": "global",
                "confirm": True,
            },
        )

        assert result["status"] == "demoted"
        assert not ensemble_file.exists()

    @pytest.mark.asyncio
    async def test_demote_removes_orphaned_profiles(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Demote with remove_orphaned_profiles cleans up unused profiles."""
        global_ensembles, global_profiles = _setup_global_dirs(tmp_path, server)

        # Create ensemble referencing two profiles
        ensemble_data = {
            "name": "test-ensemble",
            "agents": [
                {"name": "a1", "model_profile": "orphan-profile"},
                {"name": "a2", "model_profile": "shared-profile"},
            ],
        }
        (global_ensembles / "test-ensemble.yaml").write_text(
            yaml.safe_dump(ensemble_data)
        )

        # Create another ensemble using shared-profile
        other_data = {
            "name": "other-ensemble",
            "agents": [{"name": "b1", "model_profile": "shared-profile"}],
        }
        (global_ensembles / "other-ensemble.yaml").write_text(
            yaml.safe_dump(other_data)
        )

        # Create both profiles
        (global_profiles / "orphan-profile.yaml").write_text(
            yaml.safe_dump({"name": "orphan-profile", "provider": "ollama"})
        )
        (global_profiles / "shared-profile.yaml").write_text(
            yaml.safe_dump({"name": "shared-profile", "provider": "ollama"})
        )

        _mock_config(server).get_ensembles_dirs.return_value = [str(global_ensembles)]
        _mock_config(server).get_profiles_dirs.return_value = [str(global_profiles)]

        result = await server.call_tool(
            "demote_ensemble",
            {
                "ensemble_name": "test-ensemble",
                "tier": "global",
                "remove_orphaned_profiles": True,
                "confirm": True,
            },
        )

        assert result["status"] == "demoted"
        assert "orphan-profile" in result["removed_profiles"]
        assert "shared-profile" not in result["removed_profiles"]
        # orphan-profile deleted, shared-profile kept
        assert not (global_profiles / "orphan-profile.yaml").exists()
        assert (global_profiles / "shared-profile.yaml").exists()

    @pytest.mark.asyncio
    async def test_demote_preview_shows_orphaned_profiles(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Demote preview shows which profiles would be orphaned."""
        global_ensembles, global_profiles = _setup_global_dirs(tmp_path, server)

        ensemble_data = {
            "name": "test-ensemble",
            "agents": [{"name": "a1", "model_profile": "lonely-profile"}],
        }
        (global_ensembles / "test-ensemble.yaml").write_text(
            yaml.safe_dump(ensemble_data)
        )
        (global_profiles / "lonely-profile.yaml").write_text(
            yaml.safe_dump({"name": "lonely-profile", "provider": "ollama"})
        )

        _mock_config(server).get_ensembles_dirs.return_value = [str(global_ensembles)]
        _mock_config(server).get_profiles_dirs.return_value = [str(global_profiles)]

        result = await server.call_tool(
            "demote_ensemble",
            {
                "ensemble_name": "test-ensemble",
                "tier": "global",
                "remove_orphaned_profiles": True,
                "confirm": False,
            },
        )

        assert result["status"] == "preview"
        assert "lonely-profile" in result["would_remove"]["orphaned_profiles"]
        # Nothing actually deleted
        assert (global_ensembles / "test-ensemble.yaml").exists()
        assert (global_profiles / "lonely-profile.yaml").exists()


# ==============================================================================
# Dispatch table integration tests
# ==============================================================================


class TestPromotionDispatchTable:
    """Tests that promotion tools are in the dispatch table."""

    @pytest.mark.asyncio
    async def test_promote_ensemble_in_dispatch_table(self, server: MCPServer) -> None:
        """promote_ensemble is accessible via call_tool."""
        handler = server._get_tool_handler("promote_ensemble")
        assert handler is not None

    @pytest.mark.asyncio
    async def test_list_dependencies_in_dispatch_table(self, server: MCPServer) -> None:
        """list_dependencies is accessible via call_tool."""
        handler = server._get_tool_handler("list_dependencies")
        assert handler is not None

    @pytest.mark.asyncio
    async def test_check_promotion_readiness_in_dispatch_table(
        self, server: MCPServer
    ) -> None:
        """check_promotion_readiness is accessible via call_tool."""
        handler = server._get_tool_handler("check_promotion_readiness")
        assert handler is not None

    @pytest.mark.asyncio
    async def test_demote_ensemble_in_dispatch_table(self, server: MCPServer) -> None:
        """demote_ensemble is accessible via call_tool."""
        handler = server._get_tool_handler("demote_ensemble")
        assert handler is not None

"""Unit tests for MCPServer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

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
    return config


@pytest.fixture
def server(mock_config_manager: Any) -> MCPServer:
    """Create MCPServer instance with mocked dependencies."""
    return MCPServer(config_manager=mock_config_manager)


class TestMCPServerInitialization:
    """Tests for MCPServer initialization."""

    def test_init_creates_server(self, server: MCPServer) -> None:
        """Server initializes correctly."""
        assert server is not None
        assert server.config_manager is not None

    def test_init_with_custom_config_manager(self, mock_config_manager: Any) -> None:
        """Server accepts custom config manager."""
        server = MCPServer(config_manager=mock_config_manager)
        assert server.config_manager is mock_config_manager

    def test_init_creates_ensemble_loader(self, server: MCPServer) -> None:
        """Server creates ensemble loader."""
        assert server.ensemble_loader is not None

    def test_init_creates_artifact_manager(self, server: MCPServer) -> None:
        """Server creates artifact manager."""
        assert server.artifact_manager is not None


class TestMCPServerHandleInitialize:
    """Tests for handle_initialize method."""

    @pytest.mark.asyncio
    async def test_handle_initialize_returns_capabilities(
        self, server: MCPServer
    ) -> None:
        """Initialize returns server capabilities."""
        result = await server.handle_initialize()

        assert "protocolVersion" in result
        assert "capabilities" in result
        assert "serverInfo" in result

    @pytest.mark.asyncio
    async def test_handle_initialize_includes_tools_capability(
        self, server: MCPServer
    ) -> None:
        """Initialize includes tools capability."""
        result = await server.handle_initialize()

        assert "tools" in result["capabilities"]

    @pytest.mark.asyncio
    async def test_handle_initialize_includes_resources_capability(
        self, server: MCPServer
    ) -> None:
        """Initialize includes resources capability."""
        result = await server.handle_initialize()

        assert "resources" in result["capabilities"]


class TestMCPServerCallTool:
    """Tests for call_tool method."""

    @pytest.mark.asyncio
    async def test_call_tool_unknown_raises_error(self, server: MCPServer) -> None:
        """Unknown tool raises ValueError."""
        with pytest.raises(ValueError, match="Tool not found"):
            await server.call_tool("unknown_tool", {})

    @pytest.mark.asyncio
    async def test_call_tool_invoke_missing_ensemble_raises_error(
        self, server: MCPServer
    ) -> None:
        """Invoke without ensemble_name raises error."""
        with pytest.raises(ValueError, match="ensemble_name is required"):
            await server.call_tool("invoke", {"input_data": "test"})

    @pytest.mark.asyncio
    async def test_call_tool_validate_missing_ensemble_raises_error(
        self, server: MCPServer
    ) -> None:
        """Validate without ensemble_name raises error."""
        with pytest.raises(ValueError, match="ensemble_name is required"):
            await server.call_tool("validate_ensemble", {})

    @pytest.mark.asyncio
    async def test_call_tool_create_ensemble_missing_name_raises_error(
        self, server: MCPServer
    ) -> None:
        """Create ensemble without name raises error."""
        with pytest.raises(ValueError, match="name is required"):
            await server.call_tool("create_ensemble", {})

    @pytest.mark.asyncio
    async def test_call_tool_delete_ensemble_missing_name_raises_error(
        self, server: MCPServer
    ) -> None:
        """Delete ensemble without name raises error."""
        with pytest.raises(ValueError, match="ensemble_name is required"):
            await server.call_tool("delete_ensemble", {})

    @pytest.mark.asyncio
    async def test_call_tool_delete_ensemble_no_confirm_raises_error(
        self, server: MCPServer
    ) -> None:
        """Delete ensemble without confirmation raises error."""
        with pytest.raises(ValueError, match="Confirmation required"):
            await server.call_tool(
                "delete_ensemble", {"ensemble_name": "test", "confirm": False}
            )

    @pytest.mark.asyncio
    async def test_call_tool_library_copy_missing_source_raises_error(
        self, server: MCPServer
    ) -> None:
        """Library copy without source raises error."""
        with pytest.raises(ValueError, match="source is required"):
            await server.call_tool("library_copy", {})


class TestMCPServerCreateEnsemble:
    """Tests for create_ensemble tool."""

    @pytest.mark.asyncio
    async def test_create_ensemble_success(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Create ensemble successfully writes file."""
        ensembles_dir = tmp_path / ".llm-orc" / "ensembles"
        ensembles_dir.mkdir(parents=True)

        _mock_config(server).get_ensembles_dirs.return_value = [str(ensembles_dir)]

        result = await server.call_tool(
            "create_ensemble",
            {
                "name": "test-ensemble",
                "description": "Test description",
                "agents": [{"name": "agent1", "model_profile": "fast"}],
            },
        )

        assert result["created"] is True
        assert "test-ensemble.yaml" in result["path"]
        assert (ensembles_dir / "test-ensemble.yaml").exists()

    @pytest.mark.asyncio
    async def test_create_ensemble_duplicate_raises_error(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Create duplicate ensemble raises error."""
        ensembles_dir = tmp_path / ".llm-orc" / "ensembles"
        ensembles_dir.mkdir(parents=True)
        (ensembles_dir / "existing.yaml").write_text("name: existing")

        _mock_config(server).get_ensembles_dirs.return_value = [str(ensembles_dir)]

        with pytest.raises(ValueError, match="already exists"):
            await server.call_tool(
                "create_ensemble",
                {"name": "existing", "agents": []},
            )

    @pytest.mark.asyncio
    async def test_create_ensemble_from_template(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Create ensemble from template copies agents."""
        ensembles_dir = tmp_path / ".llm-orc" / "ensembles"
        ensembles_dir.mkdir(parents=True)
        (ensembles_dir / "template.yaml").write_text(
            "name: template\ndescription: Template\n"
            "agents:\n  - name: agent1\n    model_profile: fast"
        )

        _mock_config(server).get_ensembles_dirs.return_value = [str(ensembles_dir)]

        result = await server.call_tool(
            "create_ensemble",
            {"name": "new-from-template", "from_template": "template"},
        )

        assert result["created"] is True
        assert result["agents_copied"] == 1


class TestMCPServerDeleteEnsemble:
    """Tests for delete_ensemble tool."""

    @pytest.mark.asyncio
    async def test_delete_ensemble_success(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Delete ensemble removes file."""
        ensembles_dir = tmp_path / ".llm-orc" / "ensembles"
        ensembles_dir.mkdir(parents=True)
        (ensembles_dir / "to-delete.yaml").write_text("name: to-delete")

        _mock_config(server).get_ensembles_dirs.return_value = [str(ensembles_dir)]

        result = await server.call_tool(
            "delete_ensemble",
            {"ensemble_name": "to-delete", "confirm": True},
        )

        assert result["deleted"] is True
        assert not (ensembles_dir / "to-delete.yaml").exists()

    @pytest.mark.asyncio
    async def test_delete_ensemble_not_found_raises_error(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Delete non-existent ensemble raises error."""
        ensembles_dir = tmp_path / ".llm-orc" / "ensembles"
        ensembles_dir.mkdir(parents=True)

        _mock_config(server).get_ensembles_dirs.return_value = [str(ensembles_dir)]

        with pytest.raises(ValueError, match="not found"):
            await server.call_tool(
                "delete_ensemble",
                {"ensemble_name": "non-existent", "confirm": True},
            )


class TestMCPServerListScripts:
    """Tests for list_scripts tool."""

    @pytest.mark.asyncio
    async def test_list_scripts_empty(
        self, server: MCPServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """List scripts returns empty when no scripts exist."""
        monkeypatch.chdir(tmp_path)

        result = await server.call_tool("list_scripts", {})

        assert result["scripts"] == []

    @pytest.mark.asyncio
    async def test_list_scripts_finds_scripts(
        self, server: MCPServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """List scripts finds scripts in directory."""
        monkeypatch.chdir(tmp_path)
        scripts_dir = tmp_path / ".llm-orc" / "scripts" / "transform"
        scripts_dir.mkdir(parents=True)
        (scripts_dir / "uppercase.py").write_text("def transform(x): return x.upper()")

        result = await server.call_tool("list_scripts", {})

        assert len(result["scripts"]) == 1
        assert result["scripts"][0]["name"] == "uppercase"
        assert result["scripts"][0]["category"] == "transform"

    @pytest.mark.asyncio
    async def test_list_scripts_filters_by_category(
        self, server: MCPServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """List scripts filters by category."""
        monkeypatch.chdir(tmp_path)
        scripts_dir = tmp_path / ".llm-orc" / "scripts"

        (scripts_dir / "transform").mkdir(parents=True)
        (scripts_dir / "transform" / "upper.py").write_text("# transform")

        (scripts_dir / "validate").mkdir(parents=True)
        (scripts_dir / "validate" / "check.py").write_text("# validate")

        result = await server.call_tool("list_scripts", {"category": "transform"})

        assert len(result["scripts"]) == 1
        assert result["scripts"][0]["category"] == "transform"

    @pytest.mark.asyncio
    async def test_list_scripts_finds_root_level_scripts(
        self, server: MCPServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """List scripts finds scripts at root level (no category subdirectory)."""
        monkeypatch.chdir(tmp_path)
        scripts_dir = tmp_path / ".llm-orc" / "scripts"
        scripts_dir.mkdir(parents=True)

        # Script at root level, not in a category subdirectory
        (scripts_dir / "aggregator.py").write_text("# root level script")

        result = await server.call_tool("list_scripts", {})

        assert len(result["scripts"]) == 1
        assert result["scripts"][0]["name"] == "aggregator"
        assert result["scripts"][0]["category"] == ""  # No category for root scripts


class TestMCPServerLibraryBrowse:
    """Tests for library_browse tool."""

    @pytest.mark.asyncio
    async def test_library_browse_empty(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Browse empty library returns empty lists."""
        server._library_handler._test_library_dir = tmp_path / "empty-library"
        server._library_handler._test_library_dir.mkdir()

        result = await server.call_tool("library_browse", {})

        assert result["ensembles"] == []
        assert result["scripts"] == []

    @pytest.mark.asyncio
    async def test_library_browse_ensembles_only(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Browse library for ensembles only."""
        library_dir = tmp_path / "library"
        ensembles_dir = library_dir / "ensembles"
        ensembles_dir.mkdir(parents=True)
        (ensembles_dir / "test.yaml").write_text(
            "name: test\ndescription: Test\nagents: []"
        )

        server._library_handler._test_library_dir = library_dir

        result = await server.call_tool("library_browse", {"type": "ensembles"})

        assert "ensembles" in result
        assert "scripts" not in result


class TestMCPServerLibraryCopy:
    """Tests for library_copy tool."""

    @pytest.mark.asyncio
    async def test_library_copy_success(
        self, server: MCPServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Copy from library succeeds."""
        monkeypatch.chdir(tmp_path)

        library_dir = tmp_path / "library"
        ensembles_dir = library_dir / "ensembles"
        ensembles_dir.mkdir(parents=True)
        (ensembles_dir / "to-copy.yaml").write_text("name: to-copy\nagents: []")

        local_dir = tmp_path / ".llm-orc" / "ensembles"
        local_dir.mkdir(parents=True)

        server._library_handler._test_library_dir = library_dir
        _mock_config(server).get_ensembles_dirs.return_value = [str(local_dir)]

        result = await server.call_tool(
            "library_copy",
            {"source": "ensembles/to-copy.yaml"},
        )

        assert result["copied"] is True
        assert (local_dir / "to-copy.yaml").exists()

    @pytest.mark.asyncio
    async def test_library_copy_source_not_found_raises_error(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Copy from non-existent source raises error."""
        library_dir = tmp_path / "library"
        library_dir.mkdir()

        server._library_handler._test_library_dir = library_dir

        with pytest.raises(ValueError, match="not found in library"):
            await server.call_tool(
                "library_copy",
                {"source": "ensembles/missing.yaml"},
            )

    @pytest.mark.asyncio
    async def test_library_copy_exists_no_overwrite_raises_error(
        self, server: MCPServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Copy to existing file without overwrite raises error."""
        monkeypatch.chdir(tmp_path)

        library_dir = tmp_path / "library"
        ensembles_dir = library_dir / "ensembles"
        ensembles_dir.mkdir(parents=True)
        (ensembles_dir / "exists.yaml").write_text("name: exists")

        local_dir = tmp_path / ".llm-orc" / "ensembles"
        local_dir.mkdir(parents=True)
        (local_dir / "exists.yaml").write_text("name: local")

        server._library_handler._test_library_dir = library_dir
        _mock_config(server).get_ensembles_dirs.return_value = [str(local_dir)]

        with pytest.raises(ValueError, match="already exists"):
            await server.call_tool(
                "library_copy",
                {"source": "ensembles/exists.yaml", "overwrite": False},
            )

    @pytest.mark.asyncio
    async def test_library_copy_with_overwrite_succeeds(
        self, server: MCPServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Copy with overwrite replaces existing file."""
        monkeypatch.chdir(tmp_path)

        library_dir = tmp_path / "library"
        ensembles_dir = library_dir / "ensembles"
        ensembles_dir.mkdir(parents=True)
        (ensembles_dir / "exists.yaml").write_text("name: library-version")

        local_dir = tmp_path / ".llm-orc" / "ensembles"
        local_dir.mkdir(parents=True)
        (local_dir / "exists.yaml").write_text("name: local-version")

        server._library_handler._test_library_dir = library_dir
        _mock_config(server).get_ensembles_dirs.return_value = [str(local_dir)]

        result = await server.call_tool(
            "library_copy",
            {"source": "ensembles/exists.yaml", "overwrite": True},
        )

        assert result["copied"] is True
        content = (local_dir / "exists.yaml").read_text()
        assert "library-version" in content


class TestMCPServerGetLibraryDir:
    """Tests for _get_library_dir method."""

    def test_get_library_dir_from_test_override(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Test override takes precedence."""
        server._library_handler._test_library_dir = tmp_path / "test-lib"

        result = server._library_handler.get_library_dir()

        assert result == tmp_path / "test-lib"

    def test_get_library_dir_from_ensemble_dirs(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Finds library from ensemble dirs."""
        library_dir = tmp_path / "llm-orchestra-library" / "ensembles"
        _mock_config(server).get_ensembles_dirs.return_value = [str(library_dir)]

        result = server._library_handler.get_library_dir()

        assert result == tmp_path / "llm-orchestra-library"

    def test_get_library_dir_default(
        self, server: MCPServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Falls back to default library location."""
        monkeypatch.chdir(tmp_path)
        _mock_config(server).get_ensembles_dirs.return_value = []

        result = server._library_handler.get_library_dir()

        assert result == tmp_path / "llm-orchestra-library"


class TestMCPServerProfileTools:
    """Tests for profile CRUD tools."""

    @pytest.mark.asyncio
    async def test_list_profiles_empty(self, server: MCPServer, tmp_path: Path) -> None:
        """List profiles returns empty when no profiles exist."""
        _mock_config(server).get_profiles_dirs.return_value = [str(tmp_path)]

        result = await server.call_tool("list_profiles", {})

        assert "profiles" in result
        assert result["profiles"] == []

    @pytest.mark.asyncio
    async def test_list_profiles_finds_yaml_files(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """List profiles finds YAML profile files."""
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        (profiles_dir / "test-profile.yaml").write_text(
            "name: test-profile\nprovider: ollama\nmodel: llama2"
        )
        _mock_config(server).get_profiles_dirs.return_value = [str(profiles_dir)]

        result = await server.call_tool("list_profiles", {})

        assert len(result["profiles"]) == 1
        assert result["profiles"][0]["name"] == "test-profile"
        assert result["profiles"][0]["provider"] == "ollama"

    @pytest.mark.asyncio
    async def test_list_profiles_filters_by_provider(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """List profiles filters by provider."""
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        (profiles_dir / "ollama-profile.yaml").write_text(
            "name: ollama-profile\nprovider: ollama\nmodel: llama2"
        )
        (profiles_dir / "anthropic-profile.yaml").write_text(
            "name: anthropic-profile\nprovider: anthropic\nmodel: claude-3"
        )
        _mock_config(server).get_profiles_dirs.return_value = [str(profiles_dir)]

        result = await server.call_tool("list_profiles", {"provider": "ollama"})

        assert len(result["profiles"]) == 1
        assert result["profiles"][0]["provider"] == "ollama"

    @pytest.mark.asyncio
    async def test_create_profile_requires_name(self, server: MCPServer) -> None:
        """Create profile requires name."""
        with pytest.raises(ValueError, match="name is required"):
            await server.call_tool(
                "create_profile", {"provider": "ollama", "model": "llama2"}
            )

    @pytest.mark.asyncio
    async def test_create_profile_requires_provider(self, server: MCPServer) -> None:
        """Create profile requires provider."""
        with pytest.raises(ValueError, match="provider is required"):
            await server.call_tool(
                "create_profile", {"name": "test", "model": "llama2"}
            )

    @pytest.mark.asyncio
    async def test_create_profile_requires_model(self, server: MCPServer) -> None:
        """Create profile requires model."""
        with pytest.raises(ValueError, match="model is required"):
            await server.call_tool(
                "create_profile", {"name": "test", "provider": "ollama"}
            )

    @pytest.mark.asyncio
    async def test_create_profile_writes_yaml(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Create profile writes YAML file."""
        profiles_dir = tmp_path / ".llm-orc" / "profiles"
        profiles_dir.mkdir(parents=True)
        _mock_config(server).get_profiles_dirs.return_value = [str(profiles_dir)]

        result = await server.call_tool(
            "create_profile",
            {"name": "new-profile", "provider": "ollama", "model": "llama2"},
        )

        assert result["created"] is True
        assert (profiles_dir / "new-profile.yaml").exists()

    @pytest.mark.asyncio
    async def test_create_profile_fails_if_exists(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Create profile fails if profile already exists."""
        profiles_dir = tmp_path / ".llm-orc" / "profiles"
        profiles_dir.mkdir(parents=True)
        (profiles_dir / "existing.yaml").write_text("name: existing")
        _mock_config(server).get_profiles_dirs.return_value = [str(profiles_dir)]

        with pytest.raises(ValueError, match="already exists"):
            await server.call_tool(
                "create_profile",
                {"name": "existing", "provider": "ollama", "model": "llama2"},
            )

    @pytest.mark.asyncio
    async def test_update_profile_requires_name(self, server: MCPServer) -> None:
        """Update profile requires name."""
        with pytest.raises(ValueError, match="name is required"):
            await server.call_tool("update_profile", {"changes": {"model": "new"}})

    @pytest.mark.asyncio
    async def test_update_profile_not_found(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Update profile fails if not found."""
        _mock_config(server).get_profiles_dirs.return_value = [str(tmp_path)]

        with pytest.raises(ValueError, match="not found"):
            await server.call_tool(
                "update_profile", {"name": "nonexistent", "changes": {"model": "new"}}
            )

    @pytest.mark.asyncio
    async def test_update_profile_applies_changes(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Update profile applies changes to file."""
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        (profiles_dir / "test.yaml").write_text(
            "name: test\nprovider: ollama\nmodel: old"
        )
        _mock_config(server).get_profiles_dirs.return_value = [str(profiles_dir)]

        result = await server.call_tool(
            "update_profile", {"name": "test", "changes": {"model": "new"}}
        )

        assert result["updated"] is True
        content = (profiles_dir / "test.yaml").read_text()
        assert "new" in content

    @pytest.mark.asyncio
    async def test_delete_profile_requires_name(self, server: MCPServer) -> None:
        """Delete profile requires name."""
        with pytest.raises(ValueError, match="name is required"):
            await server.call_tool("delete_profile", {"confirm": True})

    @pytest.mark.asyncio
    async def test_delete_profile_requires_confirmation(
        self, server: MCPServer
    ) -> None:
        """Delete profile requires confirmation."""
        with pytest.raises(ValueError, match="Confirmation required"):
            await server.call_tool("delete_profile", {"name": "test", "confirm": False})

    @pytest.mark.asyncio
    async def test_delete_profile_not_found(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Delete profile fails if not found."""
        _mock_config(server).get_profiles_dirs.return_value = [str(tmp_path)]

        with pytest.raises(ValueError, match="not found"):
            await server.call_tool(
                "delete_profile", {"name": "nonexistent", "confirm": True}
            )

    @pytest.mark.asyncio
    async def test_delete_profile_removes_file(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Delete profile removes the file."""
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        profile_file = profiles_dir / "test.yaml"
        profile_file.write_text("name: test")
        _mock_config(server).get_profiles_dirs.return_value = [str(profiles_dir)]

        result = await server.call_tool(
            "delete_profile", {"name": "test", "confirm": True}
        )

        assert result["deleted"] is True
        assert not profile_file.exists()


class TestMCPServerArtifactTools:
    """Tests for artifact management tools."""

    @pytest.mark.asyncio
    async def test_delete_artifact_requires_id(self, server: MCPServer) -> None:
        """Delete artifact requires artifact_id."""
        with pytest.raises(ValueError, match="artifact_id is required"):
            await server.call_tool("delete_artifact", {"confirm": True})

    @pytest.mark.asyncio
    async def test_delete_artifact_requires_confirmation(
        self, server: MCPServer
    ) -> None:
        """Delete artifact requires confirmation."""
        with pytest.raises(ValueError, match="Confirmation required"):
            await server.call_tool(
                "delete_artifact", {"artifact_id": "test/123", "confirm": False}
            )

    @pytest.mark.asyncio
    async def test_delete_artifact_validates_format(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Delete artifact validates artifact_id format."""
        server._artifact_handler._test_artifacts_base = tmp_path

        with pytest.raises(ValueError, match="Invalid artifact_id format"):
            await server.call_tool(
                "delete_artifact", {"artifact_id": "invalid", "confirm": True}
            )

    @pytest.mark.asyncio
    async def test_delete_artifact_not_found(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Delete artifact fails if not found."""
        server._artifact_handler._test_artifacts_base = tmp_path

        with pytest.raises(ValueError, match="not found"):
            await server.call_tool(
                "delete_artifact",
                {"artifact_id": "ensemble/nonexistent", "confirm": True},
            )

    @pytest.mark.asyncio
    async def test_delete_artifact_removes_directory(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Delete artifact removes the directory."""
        artifacts_dir = tmp_path / "test-ensemble" / "20231201_120000"
        artifacts_dir.mkdir(parents=True)
        (artifacts_dir / "execution.json").write_text("{}")
        server._artifact_handler._test_artifacts_base = tmp_path

        result = await server.call_tool(
            "delete_artifact",
            {"artifact_id": "test-ensemble/20231201_120000", "confirm": True},
        )

        assert result["deleted"] is True
        assert not artifacts_dir.exists()

    @pytest.mark.asyncio
    async def test_cleanup_artifacts_dry_run(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Cleanup artifacts dry run lists without deleting."""
        import os
        import time

        artifacts_dir = tmp_path / "test-ensemble" / "old_artifact"
        artifacts_dir.mkdir(parents=True)
        # Set old modification time
        old_time = time.time() - (60 * 24 * 60 * 60)  # 60 days ago
        os.utime(artifacts_dir, (old_time, old_time))
        server._artifact_handler._test_artifacts_base = tmp_path

        result = await server.call_tool(
            "cleanup_artifacts", {"older_than_days": 30, "dry_run": True}
        )

        assert result["dry_run"] is True
        assert "test-ensemble/old_artifact" in result["would_delete"]
        assert artifacts_dir.exists()  # Still exists

    @pytest.mark.asyncio
    async def test_cleanup_artifacts_actual_delete(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Cleanup artifacts actually deletes when not dry run."""
        import os
        import time

        artifacts_dir = tmp_path / "test-ensemble" / "old_artifact"
        artifacts_dir.mkdir(parents=True)
        old_time = time.time() - (60 * 24 * 60 * 60)
        os.utime(artifacts_dir, (old_time, old_time))
        server._artifact_handler._test_artifacts_base = tmp_path

        result = await server.call_tool(
            "cleanup_artifacts", {"older_than_days": 30, "dry_run": False}
        )

        assert result["dry_run"] is False
        assert "test-ensemble/old_artifact" in result["deleted"]
        assert not artifacts_dir.exists()

    @pytest.mark.asyncio
    async def test_cleanup_artifacts_filters_by_ensemble(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Cleanup artifacts filters by ensemble name."""
        import os
        import time

        old_time = time.time() - (60 * 24 * 60 * 60)

        # Create artifacts in two ensembles
        for name in ["ensemble-a", "ensemble-b"]:
            artifact_dir = tmp_path / name / "old"
            artifact_dir.mkdir(parents=True)
            os.utime(artifact_dir, (old_time, old_time))

        server._artifact_handler._test_artifacts_base = tmp_path

        result = await server.call_tool(
            "cleanup_artifacts",
            {"ensemble_name": "ensemble-a", "older_than_days": 30, "dry_run": True},
        )

        assert len(result["would_delete"]) == 1
        assert "ensemble-a/old" in result["would_delete"]


class TestMCPServerScriptTools:
    """Tests for script management tools."""

    @pytest.mark.asyncio
    async def test_get_script_requires_name(self, server: MCPServer) -> None:
        """Get script requires name."""
        with pytest.raises(ValueError, match="name is required"):
            await server.call_tool("get_script", {"category": "extraction"})

    @pytest.mark.asyncio
    async def test_get_script_requires_category(self, server: MCPServer) -> None:
        """Get script requires category."""
        with pytest.raises(ValueError, match="category is required"):
            await server.call_tool("get_script", {"name": "test"})

    @pytest.mark.asyncio
    async def test_get_script_not_found(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Get script fails if not found."""
        server._script_handler._test_scripts_dir = tmp_path

        with pytest.raises(ValueError, match="not found"):
            await server.call_tool(
                "get_script", {"name": "nonexistent", "category": "extraction"}
            )

    @pytest.mark.asyncio
    async def test_get_script_returns_details(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Get script returns script details."""
        scripts_dir = tmp_path / "extraction"
        scripts_dir.mkdir(parents=True)
        script_content = '"""Test script for extraction."""\nimport sys\nprint("hello")'
        (scripts_dir / "test.py").write_text(script_content)
        server._script_handler._test_scripts_dir = tmp_path

        result = await server.call_tool(
            "get_script", {"name": "test", "category": "extraction"}
        )

        assert result["name"] == "test"
        assert result["category"] == "extraction"
        assert "source" in result
        assert "Test script for extraction" in result["description"]

    @pytest.mark.asyncio
    async def test_create_script_requires_name(self, server: MCPServer) -> None:
        """Create script requires name."""
        with pytest.raises(ValueError, match="name is required"):
            await server.call_tool("create_script", {"category": "extraction"})

    @pytest.mark.asyncio
    async def test_create_script_requires_category(self, server: MCPServer) -> None:
        """Create script requires category."""
        with pytest.raises(ValueError, match="category is required"):
            await server.call_tool("create_script", {"name": "test"})

    @pytest.mark.asyncio
    async def test_create_script_basic_template(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Create script with basic template."""
        server._script_handler._test_scripts_dir = tmp_path

        result = await server.call_tool(
            "create_script",
            {"name": "new-script", "category": "utils", "template": "basic"},
        )

        assert result["created"] is True
        script_file = tmp_path / "utils" / "new-script.py"
        assert script_file.exists()
        content = script_file.read_text()
        assert "Primitive script" in content

    @pytest.mark.asyncio
    async def test_create_script_extraction_template(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Create script with extraction template."""
        server._script_handler._test_scripts_dir = tmp_path

        result = await server.call_tool(
            "create_script",
            {"name": "extractor", "category": "extraction", "template": "extraction"},
        )

        assert result["created"] is True
        content = (tmp_path / "extraction" / "extractor.py").read_text()
        assert "Extraction script" in content
        assert "json" in content

    @pytest.mark.asyncio
    async def test_create_script_fails_if_exists(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Create script fails if already exists."""
        scripts_dir = tmp_path / "extraction"
        scripts_dir.mkdir(parents=True)
        (scripts_dir / "existing.py").write_text("# existing")
        server._script_handler._test_scripts_dir = tmp_path

        with pytest.raises(ValueError, match="already exists"):
            await server.call_tool(
                "create_script", {"name": "existing", "category": "extraction"}
            )

    @pytest.mark.asyncio
    async def test_delete_script_requires_name(self, server: MCPServer) -> None:
        """Delete script requires name."""
        with pytest.raises(ValueError, match="name is required"):
            await server.call_tool(
                "delete_script", {"category": "extraction", "confirm": True}
            )

    @pytest.mark.asyncio
    async def test_delete_script_requires_category(self, server: MCPServer) -> None:
        """Delete script requires category."""
        with pytest.raises(ValueError, match="category is required"):
            await server.call_tool("delete_script", {"name": "test", "confirm": True})

    @pytest.mark.asyncio
    async def test_delete_script_requires_confirmation(self, server: MCPServer) -> None:
        """Delete script requires confirmation."""
        with pytest.raises(ValueError, match="Confirmation required"):
            await server.call_tool(
                "delete_script",
                {"name": "test", "category": "extraction", "confirm": False},
            )

    @pytest.mark.asyncio
    async def test_delete_script_not_found(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Delete script fails if not found."""
        server._script_handler._test_scripts_dir = tmp_path

        with pytest.raises(ValueError, match="not found"):
            await server.call_tool(
                "delete_script",
                {"name": "nonexistent", "category": "extraction", "confirm": True},
            )

    @pytest.mark.asyncio
    async def test_delete_script_removes_file(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Delete script removes the file."""
        scripts_dir = tmp_path / "extraction"
        scripts_dir.mkdir(parents=True)
        script_file = scripts_dir / "test.py"
        script_file.write_text("# test")
        server._script_handler._test_scripts_dir = tmp_path

        result = await server.call_tool(
            "delete_script",
            {"name": "test", "category": "extraction", "confirm": True},
        )

        assert result["deleted"] is True
        assert not script_file.exists()

    @pytest.mark.asyncio
    async def test_test_script_requires_name(self, server: MCPServer) -> None:
        """Test script requires name."""
        with pytest.raises(ValueError, match="name is required"):
            await server.call_tool(
                "test_script", {"category": "extraction", "input": "test"}
            )

    @pytest.mark.asyncio
    async def test_test_script_requires_category(self, server: MCPServer) -> None:
        """Test script requires category."""
        with pytest.raises(ValueError, match="category is required"):
            await server.call_tool("test_script", {"name": "test", "input": "test"})

    @pytest.mark.asyncio
    async def test_test_script_runs_script(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Test script runs the script with input."""
        scripts_dir = tmp_path / "utils"
        scripts_dir.mkdir(parents=True)
        (scripts_dir / "echo.py").write_text(
            "import sys\nprint(sys.stdin.read().upper())"
        )
        server._script_handler._test_scripts_dir = tmp_path

        result = await server.call_tool(
            "test_script", {"name": "echo", "category": "utils", "input": "hello"}
        )

        assert result["success"] is True
        assert "HELLO" in result["stdout"]


class TestMCPServerLibraryExtraTools:
    """Tests for library extra tools."""

    @pytest.mark.asyncio
    async def test_library_search_requires_query(self, server: MCPServer) -> None:
        """Library search requires query."""
        with pytest.raises(ValueError, match="query is required"):
            await server.call_tool("library_search", {})

    @pytest.mark.asyncio
    async def test_library_search_finds_matching_ensembles(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Library search finds matching ensembles."""
        library_dir = tmp_path / "library"
        ensembles_dir = library_dir / "ensembles"
        ensembles_dir.mkdir(parents=True)
        (ensembles_dir / "code-review.yaml").write_text(
            "name: code-review\ndescription: Review code changes\nagents: []"
        )
        server._library_handler._test_library_dir = library_dir

        result = await server.call_tool("library_search", {"query": "review"})

        assert result["total"] == 1
        assert len(result["results"]["ensembles"]) == 1
        assert result["results"]["ensembles"][0]["name"] == "code-review"

    @pytest.mark.asyncio
    async def test_library_search_finds_matching_scripts(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Library search finds matching scripts."""
        library_dir = tmp_path / "library"
        scripts_dir = library_dir / "scripts" / "extraction"
        scripts_dir.mkdir(parents=True)
        (scripts_dir / "json-parser.py").write_text("# JSON parser")
        server._library_handler._test_library_dir = library_dir

        result = await server.call_tool("library_search", {"query": "json"})

        assert result["total"] == 1
        assert len(result["results"]["scripts"]) == 1
        assert result["results"]["scripts"][0]["name"] == "json-parser"

    @pytest.mark.asyncio
    async def test_library_search_empty_results(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Library search returns empty for no matches."""
        library_dir = tmp_path / "library"
        library_dir.mkdir()
        server._library_handler._test_library_dir = library_dir

        result = await server.call_tool(
            "library_search", {"query": "nonexistent-query"}
        )

        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_library_info_returns_metadata(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Library info returns library metadata."""
        library_dir = tmp_path / "library"
        library_dir.mkdir()
        server._library_handler._test_library_dir = library_dir

        result = await server.call_tool("library_info", {})

        assert "path" in result
        assert "exists" in result
        assert result["exists"] is True

    @pytest.mark.asyncio
    async def test_library_info_counts_ensembles(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Library info counts ensembles."""
        library_dir = tmp_path / "library"
        ensembles_dir = library_dir / "ensembles"
        ensembles_dir.mkdir(parents=True)
        (ensembles_dir / "test1.yaml").write_text("name: test1\nagents: []")
        (ensembles_dir / "test2.yaml").write_text("name: test2\nagents: []")
        server._library_handler._test_library_dir = library_dir

        result = await server.call_tool("library_info", {})

        assert result["ensembles_count"] == 2

    @pytest.mark.asyncio
    async def test_library_info_counts_scripts_and_categories(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Library info counts scripts and lists categories."""
        library_dir = tmp_path / "library"
        for cat in ["extraction", "validation"]:
            scripts_dir = library_dir / "scripts" / cat
            scripts_dir.mkdir(parents=True)
            (scripts_dir / "script.py").write_text("# test")
        server._library_handler._test_library_dir = library_dir

        result = await server.call_tool("library_info", {})

        assert result["scripts_count"] == 2
        assert "extraction" in result["categories"]
        assert "validation" in result["categories"]

    @pytest.mark.asyncio
    async def test_library_info_nonexistent_library(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Library info handles nonexistent library."""
        server._library_handler._test_library_dir = tmp_path / "nonexistent"

        result = await server.call_tool("library_info", {})

        assert result["exists"] is False
        assert result["ensembles_count"] == 0
        assert result["scripts_count"] == 0


class TestMCPServerHelperMethods:
    """Tests for internal helper methods."""

    def test_extract_docstring_single_line(self, server: MCPServer) -> None:
        """Extract docstring from single-line docstring."""
        content = '"""Simple docstring."""\nimport sys'

        result = server._script_handler._extract_docstring(content)

        assert result == "Simple docstring."

    def test_extract_docstring_multiline(self, server: MCPServer) -> None:
        """Extract docstring from multiline docstring."""
        content = '"""\nMultiline\ndocstring\n"""\nimport sys'

        result = server._script_handler._extract_docstring(content)

        assert "Multiline" in result
        assert "docstring" in result

    def test_extract_docstring_no_docstring(self, server: MCPServer) -> None:
        """Return empty string when no docstring."""
        content = "import sys\nprint('hello')"

        result = server._script_handler._extract_docstring(content)

        assert result == ""

    def test_strip_docstring_quotes(self, server: MCPServer) -> None:
        """Strip triple quotes from docstring."""
        text = '"""Test docstring"""'

        result = server._script_handler._strip_docstring_quotes(text)

        assert result == "Test docstring"

    def test_get_scripts_dir_from_test_override(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Get scripts dir uses test override."""
        server._script_handler._test_scripts_dir = tmp_path / "test-scripts"

        result = server._script_handler._get_scripts_dir()

        assert result == tmp_path / "test-scripts"

    def test_get_scripts_dir_default(
        self, server: MCPServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Get scripts dir defaults to .llm-orc/scripts."""
        monkeypatch.chdir(tmp_path)
        server._script_handler._test_scripts_dir = None

        result = server._script_handler._get_scripts_dir()

        assert result == tmp_path / ".llm-orc" / "scripts"

    def test_get_local_ensembles_dir_finds_local(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Get local ensembles dir finds .llm-orc directory."""
        local_dir = tmp_path / ".llm-orc" / "ensembles"
        _mock_config(server).get_ensembles_dirs.return_value = [str(local_dir)]

        result = server._get_local_ensembles_dir()

        assert result == local_dir

    def test_get_local_ensembles_dir_falls_back(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Get local ensembles dir falls back to first directory."""
        fallback_dir = tmp_path / "ensembles"
        _mock_config(server).get_ensembles_dirs.return_value = [str(fallback_dir)]

        result = server._get_local_ensembles_dir()

        assert result == fallback_dir

    def test_get_local_ensembles_dir_raises_if_none(self, server: MCPServer) -> None:
        """Get local ensembles dir raises if no directories."""
        _mock_config(server).get_ensembles_dirs.return_value = []

        with pytest.raises(ValueError, match="No ensemble directory"):
            server._get_local_ensembles_dir()


class TestMCPServerExecutorInjection:
    """Tests for EnsembleExecutor dependency injection."""

    def test_init_accepts_executor(self, mock_config_manager: Any) -> None:
        """Server accepts injected executor."""
        mock_executor = MagicMock()
        server = MCPServer(
            config_manager=mock_config_manager,
            executor=mock_executor,
        )

        assert server._executor is mock_executor

    def test_get_executor_returns_injected(self, mock_config_manager: Any) -> None:
        """Get executor returns injected executor."""
        mock_executor = MagicMock()
        server = MCPServer(
            config_manager=mock_config_manager,
            executor=mock_executor,
        )

        result = server._get_executor()

        assert result is mock_executor

    def test_get_executor_lazy_creates(self, mock_config_manager: Any) -> None:
        """Get executor lazy-creates if none injected."""
        server = MCPServer(config_manager=mock_config_manager)

        result = server._get_executor()

        assert result is not None
        assert server._executor is result


class TestMCPServerStreamingExecution:
    """Tests for streaming execution methods."""

    @pytest.fixture
    def mock_reporter(self) -> MagicMock:
        """Create mock progress reporter."""
        reporter = MagicMock()
        reporter.info = MagicMock(return_value=_async_noop())
        reporter.warning = MagicMock(return_value=_async_noop())
        reporter.error = MagicMock(return_value=_async_noop())
        reporter.report_progress = MagicMock(return_value=_async_noop())
        return reporter

    @pytest.fixture
    def mock_executor(self) -> MagicMock:
        """Create mock ensemble executor."""
        return MagicMock()

    @pytest.fixture
    def server_with_executor(
        self, mock_config_manager: Any, mock_executor: MagicMock
    ) -> MCPServer:
        """Create server with injected executor."""
        return MCPServer(
            config_manager=mock_config_manager,
            executor=mock_executor,
        )

    @pytest.mark.asyncio
    async def test_handle_streaming_event_execution_started(
        self, server: MCPServer, mock_reporter: MagicMock
    ) -> None:
        """Handle execution_started event."""
        event = {"type": "execution_started", "data": {}}
        state: dict[str, Any] = {"completed": 0, "result": {}}

        await server._handle_streaming_event(event, mock_reporter, 3, state)

        mock_reporter.report_progress.assert_called_once_with(progress=0, total=3)

    @pytest.mark.asyncio
    async def test_handle_streaming_event_agent_started(
        self, server: MCPServer, mock_reporter: MagicMock
    ) -> None:
        """Handle agent_started event."""
        event = {"type": "agent_started", "data": {"agent_name": "test-agent"}}
        state: dict[str, Any] = {"completed": 0, "result": {}}

        await server._handle_streaming_event(event, mock_reporter, 3, state)

        mock_reporter.info.assert_called_once_with("Agent 'test-agent' started")

    @pytest.mark.asyncio
    async def test_handle_streaming_event_agent_completed(
        self, server: MCPServer, mock_reporter: MagicMock
    ) -> None:
        """Handle agent_completed event increments counter."""
        event = {"type": "agent_completed", "data": {"agent_name": "test-agent"}}
        state: dict[str, Any] = {"completed": 0, "result": {}}

        await server._handle_streaming_event(event, mock_reporter, 3, state)

        assert state["completed"] == 1
        mock_reporter.report_progress.assert_called_once_with(1, 3)
        mock_reporter.info.assert_called_once_with("Agent 'test-agent' completed")

    @pytest.mark.asyncio
    async def test_handle_streaming_event_execution_completed(
        self, server: MCPServer, mock_reporter: MagicMock
    ) -> None:
        """Handle execution_completed event sets result."""
        event = {
            "type": "execution_completed",
            "data": {
                "results": {"agent1": "output1"},
                "synthesis": "combined",
                "status": "completed",
            },
        }
        state: dict[str, Any] = {
            "completed": 2,
            "result": {},
            "ensemble_name": "test",
            "input_data": "test input",
        }

        await server._handle_streaming_event(event, mock_reporter, 2, state)

        assert state["result"]["status"] == "completed"
        assert state["result"]["synthesis"] == "combined"
        mock_reporter.report_progress.assert_called_once_with(progress=2, total=2)

    @pytest.mark.asyncio
    async def test_handle_streaming_event_execution_failed(
        self, server: MCPServer, mock_reporter: MagicMock
    ) -> None:
        """Handle execution_failed event sets error."""
        event = {"type": "execution_failed", "data": {"error": "Test error"}}
        state: dict[str, Any] = {"completed": 0, "result": {}}

        await server._handle_streaming_event(event, mock_reporter, 2, state)

        assert state["result"]["status"] == "failed"
        assert state["result"]["error"] == "Test error"
        mock_reporter.error.assert_called_once_with("Execution failed: Test error")

    @pytest.mark.asyncio
    async def test_handle_streaming_event_agent_fallback_started(
        self, server: MCPServer, mock_reporter: MagicMock
    ) -> None:
        """Handle agent_fallback_started event."""
        event = {"type": "agent_fallback_started", "data": {"agent_name": "test-agent"}}
        state: dict[str, Any] = {"completed": 0, "result": {}}

        await server._handle_streaming_event(event, mock_reporter, 2, state)

        msg = "Agent 'test-agent' falling back to alternate model"
        mock_reporter.warning.assert_called_once_with(msg)

    @pytest.mark.asyncio
    async def test_execute_ensemble_streaming_raises_if_no_name(
        self, server: MCPServer, mock_reporter: MagicMock
    ) -> None:
        """Execute streaming raises if no ensemble name."""
        with pytest.raises(ValueError, match="ensemble_name is required"):
            await server._execute_ensemble_streaming("", "input", mock_reporter)

    @pytest.mark.asyncio
    async def test_execute_ensemble_streaming_raises_if_not_found(
        self, server: MCPServer, mock_reporter: MagicMock
    ) -> None:
        """Execute streaming raises if ensemble not found."""
        _mock_config(server).get_ensembles_dirs.return_value = []

        with pytest.raises(ValueError, match="Ensemble does not exist"):
            await server._execute_ensemble_streaming(
                "nonexistent", "input", mock_reporter
            )


async def _async_noop() -> None:
    """Async no-op for mock returns."""
    pass


# ==============================================================================
# Phase 3: Provider Discovery Tests
# ==============================================================================


class TestGetProviderStatusTool:
    """Tests for get_provider_status tool."""

    @pytest.mark.asyncio
    async def test_get_provider_status_returns_providers_dict(
        self, server: MCPServer
    ) -> None:
        """Get provider status returns providers dictionary."""
        result = await server._get_provider_status_tool({})
        assert "providers" in result
        assert "ollama" in result["providers"]
        assert "anthropic-api" in result["providers"]

    @pytest.mark.asyncio
    async def test_get_provider_status_has_available_field(
        self, server: MCPServer
    ) -> None:
        """Each provider has available field."""
        result = await server._get_provider_status_tool({})
        providers = result["providers"]
        for provider_status in providers.values():
            assert "available" in provider_status


class TestGetOllamaStatus:
    """Tests for _get_ollama_status helper."""

    @pytest.mark.asyncio
    async def test_get_ollama_status_returns_dict(self, server: MCPServer) -> None:
        """Get ollama status returns a dict."""
        result = await server._provider_handler._get_ollama_status()
        assert isinstance(result, dict)
        assert "available" in result

    @pytest.mark.asyncio
    async def test_get_ollama_status_has_models_when_available(
        self, server: MCPServer
    ) -> None:
        """Ollama status has models field when available."""
        result = await server._provider_handler._get_ollama_status()
        assert "models" in result


class TestGetCloudProviderStatus:
    """Tests for _get_cloud_provider_status helper."""

    def test_get_cloud_provider_status_returns_dict(self, server: MCPServer) -> None:
        """Get cloud provider status returns a dict."""
        result = server._provider_handler._get_cloud_provider_status("anthropic-api")
        assert isinstance(result, dict)
        assert "available" in result

    def test_get_cloud_provider_status_for_unknown_provider(
        self, server: MCPServer
    ) -> None:
        """Unknown provider returns not available."""
        result = server._provider_handler._get_cloud_provider_status("unknown-provider")
        assert result["available"] is False


class TestCheckEnsembleRunnableTool:
    """Tests for check_ensemble_runnable tool."""

    @pytest.mark.asyncio
    async def test_check_ensemble_runnable_requires_name(
        self, server: MCPServer
    ) -> None:
        """Check ensemble runnable requires ensemble_name."""
        with pytest.raises(ValueError, match="ensemble_name is required"):
            await server._check_ensemble_runnable_tool({})

    @pytest.mark.asyncio
    async def test_check_ensemble_runnable_raises_for_missing(
        self, server: MCPServer
    ) -> None:
        """Check ensemble runnable raises for non-existent ensemble."""
        _mock_config(server).get_ensembles_dirs.return_value = []

        with pytest.raises(ValueError, match="Ensemble not found"):
            await server._check_ensemble_runnable_tool({"ensemble_name": "nonexistent"})

    @pytest.mark.asyncio
    async def test_check_ensemble_runnable_returns_status(
        self, server: MCPServer
    ) -> None:
        """Check ensemble runnable returns runnable status."""
        from unittest.mock import patch

        # Mock the ensemble config
        mock_config = MagicMock()
        mock_config.name = "test-ensemble"
        mock_config.agents = [MagicMock(name="agent1", model_profile="fast")]

        # Mock _find_ensemble on the provider handler
        with patch.object(
            server._provider_handler,
            "_find_ensemble",
            return_value=mock_config,
        ):
            result = await server._check_ensemble_runnable_tool(
                {"ensemble_name": "test-ensemble"}
            )

        assert "runnable" in result
        assert "agents" in result
        assert "ensemble" in result


class TestCheckAgentRunnable:
    """Tests for _check_agent_runnable helper."""

    def test_check_agent_runnable_missing_profile(self, server: MCPServer) -> None:
        """Agent with missing profile has missing_profile status."""
        handler = server._provider_handler
        result = handler._check_agent_runnable("agent1", "nonexistent", {}, {})
        assert result["status"] == "missing_profile"
        assert result["name"] == "agent1"

    @pytest.mark.asyncio
    async def test_check_ensemble_runnable_recognizes_script_agents(
        self, server: MCPServer
    ) -> None:
        """Script agents should be recognized as available without profile check."""
        from unittest.mock import MagicMock, patch

        # Create a mock agent with 'script' attribute (no model_profile)
        script_agent = MagicMock()
        script_agent.name = "aggregator"
        script_agent.script = "aggregator.py"
        # model_profile should not exist or be empty
        del script_agent.model_profile

        mock_config = MagicMock()
        mock_config.name = "test-ensemble"
        mock_config.agents = [script_agent]

        with patch.object(
            server._provider_handler,
            "_find_ensemble",
            return_value=mock_config,
        ):
            result = await server._check_ensemble_runnable_tool(
                {"ensemble_name": "test-ensemble"}
            )

        assert result["runnable"] is True
        assert len(result["agents"]) == 1
        assert result["agents"][0]["name"] == "aggregator"
        assert result["agents"][0]["status"] == "available"

    def test_check_agent_runnable_available_profile(self, server: MCPServer) -> None:
        """Agent with available profile has available status."""
        profiles = {"ollama-profile": {"provider": "ollama", "model": "llama3"}}
        providers = {"ollama": {"available": True, "models": ["llama3"]}}

        result = server._provider_handler._check_agent_runnable(
            "agent1", "ollama-profile", profiles, providers
        )
        assert result["status"] == "available"
        assert result["provider"] == "ollama"

    def test_check_agent_runnable_unavailable_provider(self, server: MCPServer) -> None:
        """Agent with unavailable provider has provider_unavailable status."""
        profiles = {"cloud-profile": {"provider": "anthropic-api", "model": "claude"}}
        providers = {"anthropic-api": {"available": False}}

        result = server._provider_handler._check_agent_runnable(
            "agent1", "cloud-profile", profiles, providers
        )
        assert result["status"] == "provider_unavailable"


class TestSuggestLocalAlternatives:
    """Tests for _suggest_local_alternatives helper."""

    def test_suggest_local_alternatives_when_ollama_unavailable(
        self, server: MCPServer
    ) -> None:
        """Returns empty list when Ollama is unavailable."""
        providers = {"ollama": {"available": False}}
        result = server._provider_handler._suggest_local_alternatives(providers)
        assert result == []


class TestSuggestAvailableModels:
    """Tests for _suggest_available_models helper."""

    def test_suggest_available_models_returns_list(self, server: MCPServer) -> None:
        """Returns available models as list."""
        models = ["llama3:latest", "mistral:latest"]
        result = server._provider_handler._suggest_available_models(models)
        assert isinstance(result, list)
        assert len(result) <= 5  # Should limit to 5


class TestHelpTool:
    """Tests for the get_help tool."""

    @pytest.mark.asyncio
    async def test_get_help_returns_documentation(self, server: MCPServer) -> None:
        """get_help tool returns comprehensive documentation."""
        result = await server.call_tool("get_help", {})

        assert "directory_structure" in result
        assert "schemas" in result
        assert "tools" in result

    @pytest.mark.asyncio
    async def test_get_help_includes_ensemble_schema(self, server: MCPServer) -> None:
        """get_help includes ensemble YAML schema."""
        result = await server.call_tool("get_help", {})

        schemas = result["schemas"]
        assert "ensemble" in schemas
        assert "example" in schemas["ensemble"]

    @pytest.mark.asyncio
    async def test_get_help_includes_profile_schema(self, server: MCPServer) -> None:
        """get_help includes profile schema."""
        result = await server.call_tool("get_help", {})

        schemas = result["schemas"]
        assert "profile" in schemas

    @pytest.mark.asyncio
    async def test_get_help_includes_directory_structure(
        self, server: MCPServer
    ) -> None:
        """get_help includes directory structure info."""
        result = await server.call_tool("get_help", {})

        dirs = result["directory_structure"]
        assert "local" in dirs
        assert "global" in dirs

    @pytest.mark.asyncio
    async def test_get_help_includes_tool_categories(self, server: MCPServer) -> None:
        """get_help includes tool categories."""
        result = await server.call_tool("get_help", {})

        tools = result["tools"]
        assert "context_management" in tools
        assert "core_execution" in tools
        assert "provider_discovery" in tools


class TestSetProjectTool:
    """Tests for set_project tool."""

    @pytest.mark.asyncio
    async def test_set_project_returns_active_path(
        self, mock_config_manager: Any, tmp_path: Path
    ) -> None:
        """set_project returns the active project path."""
        server = MCPServer(config_manager=mock_config_manager)
        project_dir = tmp_path / "my-project"
        project_dir.mkdir()

        result = await server.call_tool("set_project", {"path": str(project_dir)})

        assert result["status"] == "ok"
        assert result["project_path"] == str(project_dir)

    @pytest.mark.asyncio
    async def test_set_project_updates_project_path(
        self, mock_config_manager: Any, tmp_path: Path
    ) -> None:
        """set_project updates the server's project path."""
        server = MCPServer(config_manager=mock_config_manager)
        project_dir = tmp_path / "my-project"
        project_dir.mkdir()

        await server.call_tool("set_project", {"path": str(project_dir)})

        assert server.project_path == project_dir

    @pytest.mark.asyncio
    async def test_set_project_recreates_config_manager(
        self, mock_config_manager: Any, tmp_path: Path
    ) -> None:
        """set_project creates a new config manager for the project."""
        server = MCPServer(config_manager=mock_config_manager)
        original_config = server.config_manager
        project_dir = tmp_path / "my-project"
        project_dir.mkdir()
        (project_dir / ".llm-orc").mkdir()

        await server.call_tool("set_project", {"path": str(project_dir)})

        assert server.config_manager is not original_config

    @pytest.mark.asyncio
    async def test_set_project_rejects_nonexistent_path(
        self, mock_config_manager: Any, tmp_path: Path
    ) -> None:
        """set_project rejects non-existent paths."""
        server = MCPServer(config_manager=mock_config_manager)
        nonexistent = tmp_path / "does-not-exist"

        result = await server.call_tool("set_project", {"path": str(nonexistent)})

        assert result["status"] == "error"
        assert "does not exist" in result["error"]

    @pytest.mark.asyncio
    async def test_set_project_accepts_path_without_llm_orc_dir(
        self, mock_config_manager: Any, tmp_path: Path
    ) -> None:
        """set_project accepts paths without .llm-orc (uses global only)."""
        server = MCPServer(config_manager=mock_config_manager)
        project_dir = tmp_path / "plain-project"
        project_dir.mkdir()

        result = await server.call_tool("set_project", {"path": str(project_dir)})

        assert result["status"] == "ok"
        assert "no .llm-orc directory" in result.get("note", "").lower()

    @pytest.mark.asyncio
    async def test_project_path_starts_as_none(self, server: MCPServer) -> None:
        """project_path is None by default."""
        assert server.project_path is None

    def test_set_project_propagates_config_manager_to_ensemble_crud_handler(
        self, mock_config_manager: Any, tmp_path: Path
    ) -> None:
        """After set_project, ensemble_crud_handler uses the new config_manager."""
        server = MCPServer(config_manager=mock_config_manager)
        project_dir = tmp_path / "my-project"
        project_dir.mkdir()

        server._handle_set_project(str(project_dir))

        assert server._ensemble_crud_handler._config_manager is server.config_manager

    def test_set_project_propagates_config_manager_to_validation_handler(
        self, mock_config_manager: Any, tmp_path: Path
    ) -> None:
        """After set_project, validation_handler uses the new config_manager."""
        server = MCPServer(config_manager=mock_config_manager)
        project_dir = tmp_path / "my-project"
        project_dir.mkdir()

        server._handle_set_project(str(project_dir))

        assert server._validation_handler._config_manager is server.config_manager

    def test_set_project_propagates_config_manager_to_profile_handler(
        self, mock_config_manager: Any, tmp_path: Path
    ) -> None:
        """After set_project, profile_handler uses the new config_manager (Bug 3)."""
        server = MCPServer(config_manager=mock_config_manager)
        project_dir = tmp_path / "my-project"
        project_dir.mkdir()

        server._handle_set_project(str(project_dir))

        assert server._profile_handler._config_manager is server.config_manager


class TestValidateEnsembleProfileResolution:
    """Tests for Bug 2: validate_ensemble must find profiles in profiles directories."""

    @pytest.mark.asyncio
    async def test_validate_ensemble_finds_profile_in_profiles_dir(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """validate_ensemble resolves profiles from individual YAML files."""
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        (profiles_dir / "my-profile.yaml").write_text(
            "name: my-profile\nprovider: ollama\nmodel: llama3\n"
        )

        ensembles_dir = tmp_path / "ensembles"
        ensembles_dir.mkdir()
        (ensembles_dir / "test-ensemble.yaml").write_text(
            "name: test-ensemble\n"
            "description: Test\n"
            "agents:\n"
            "  - name: agent1\n"
            "    model_profile: my-profile\n"
        )

        _mock_config(server).get_ensembles_dirs.return_value = [str(ensembles_dir)]
        _mock_config(server).get_profiles_dirs.return_value = [str(profiles_dir)]
        # Profile is NOT in config.yaml model_profiles — only in the profiles directory
        _mock_config(server).get_model_profiles.return_value = {}

        result = await server._validate_ensemble_tool(
            {"ensemble_name": "test-ensemble"}
        )

        assert result["valid"] is True, (
            f"Expected valid but got errors: {result['details']['errors']}"
        )


class TestReadEnsemblesResource:
    """Tests for _read_ensembles_resource method."""

    @pytest.mark.asyncio
    async def test_read_ensembles_includes_relative_path(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Ensemble list includes relative_path for directory hierarchy."""
        ensembles_dir = tmp_path / ".llm-orc" / "ensembles"
        subdir = ensembles_dir / "testing"
        subdir.mkdir(parents=True)
        (subdir / "my-ensemble.yaml").write_text(
            "name: my-ensemble\ndescription: Test\nagents: []"
        )
        _mock_config(server).get_ensembles_dirs.return_value = [str(ensembles_dir)]

        result = await server._read_ensembles_resource()

        assert len(result) == 1
        assert result[0]["name"] == "my-ensemble"
        assert "relative_path" in result[0]
        assert result[0]["relative_path"] == "testing/my-ensemble.yaml"

    @pytest.mark.asyncio
    async def test_read_ensembles_relative_path_for_root_level(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Ensemble at root level has just filename as relative_path."""
        ensembles_dir = tmp_path / ".llm-orc" / "ensembles"
        ensembles_dir.mkdir(parents=True)
        (ensembles_dir / "root-ensemble.yaml").write_text(
            "name: root-ensemble\ndescription: Root\nagents: []"
        )
        _mock_config(server).get_ensembles_dirs.return_value = [str(ensembles_dir)]

        result = await server._read_ensembles_resource()

        assert len(result) == 1
        assert result[0]["relative_path"] == "root-ensemble.yaml"

    @pytest.mark.asyncio
    async def test_read_ensembles_relative_path_nested_subdirs(
        self, server: MCPServer, tmp_path: Path
    ) -> None:
        """Ensemble in nested subdirs has full relative path."""
        ensembles_dir = tmp_path / ".llm-orc" / "ensembles"
        nested = ensembles_dir / "research" / "interdisciplinary"
        nested.mkdir(parents=True)
        (nested / "deep-ensemble.yaml").write_text(
            "name: deep-ensemble\ndescription: Deep\nagents: []"
        )
        _mock_config(server).get_ensembles_dirs.return_value = [str(ensembles_dir)]

        result = await server._read_ensembles_resource()

        assert len(result) == 1
        expected = "research/interdisciplinary/deep-ensemble.yaml"
        assert result[0]["relative_path"] == expected

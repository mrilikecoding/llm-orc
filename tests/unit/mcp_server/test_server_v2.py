"""Unit tests for MCPServerV2."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from llm_orc.mcp.server import MCPServerV2


def _mock_config(server: MCPServerV2) -> Any:
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
def server(mock_config_manager: Any) -> MCPServerV2:
    """Create MCPServerV2 instance with mocked dependencies."""
    return MCPServerV2(config_manager=mock_config_manager)


class TestMCPServerV2Initialization:
    """Tests for MCPServerV2 initialization."""

    def test_init_creates_server(self, server: MCPServerV2) -> None:
        """Server initializes correctly."""
        assert server is not None
        assert server.config_manager is not None

    def test_init_with_custom_config_manager(self, mock_config_manager: Any) -> None:
        """Server accepts custom config manager."""
        server = MCPServerV2(config_manager=mock_config_manager)
        assert server.config_manager is mock_config_manager

    def test_init_creates_ensemble_loader(self, server: MCPServerV2) -> None:
        """Server creates ensemble loader."""
        assert server.ensemble_loader is not None

    def test_init_creates_artifact_manager(self, server: MCPServerV2) -> None:
        """Server creates artifact manager."""
        assert server.artifact_manager is not None


class TestMCPServerV2HandleInitialize:
    """Tests for handle_initialize method."""

    @pytest.mark.asyncio
    async def test_handle_initialize_returns_capabilities(
        self, server: MCPServerV2
    ) -> None:
        """Initialize returns server capabilities."""
        result = await server.handle_initialize()

        assert "protocolVersion" in result
        assert "capabilities" in result
        assert "serverInfo" in result

    @pytest.mark.asyncio
    async def test_handle_initialize_includes_tools_capability(
        self, server: MCPServerV2
    ) -> None:
        """Initialize includes tools capability."""
        result = await server.handle_initialize()

        assert "tools" in result["capabilities"]

    @pytest.mark.asyncio
    async def test_handle_initialize_includes_resources_capability(
        self, server: MCPServerV2
    ) -> None:
        """Initialize includes resources capability."""
        result = await server.handle_initialize()

        assert "resources" in result["capabilities"]


class TestMCPServerV2CallTool:
    """Tests for call_tool method."""

    @pytest.mark.asyncio
    async def test_call_tool_unknown_raises_error(self, server: MCPServerV2) -> None:
        """Unknown tool raises ValueError."""
        with pytest.raises(ValueError, match="Tool not found"):
            await server.call_tool("unknown_tool", {})

    @pytest.mark.asyncio
    async def test_call_tool_invoke_missing_ensemble_raises_error(
        self, server: MCPServerV2
    ) -> None:
        """Invoke without ensemble_name raises error."""
        with pytest.raises(ValueError, match="ensemble_name is required"):
            await server.call_tool("invoke", {"input_data": "test"})

    @pytest.mark.asyncio
    async def test_call_tool_validate_missing_ensemble_raises_error(
        self, server: MCPServerV2
    ) -> None:
        """Validate without ensemble_name raises error."""
        with pytest.raises(ValueError, match="ensemble_name is required"):
            await server.call_tool("validate_ensemble", {})

    @pytest.mark.asyncio
    async def test_call_tool_create_ensemble_missing_name_raises_error(
        self, server: MCPServerV2
    ) -> None:
        """Create ensemble without name raises error."""
        with pytest.raises(ValueError, match="name is required"):
            await server.call_tool("create_ensemble", {})

    @pytest.mark.asyncio
    async def test_call_tool_delete_ensemble_missing_name_raises_error(
        self, server: MCPServerV2
    ) -> None:
        """Delete ensemble without name raises error."""
        with pytest.raises(ValueError, match="ensemble_name is required"):
            await server.call_tool("delete_ensemble", {})

    @pytest.mark.asyncio
    async def test_call_tool_delete_ensemble_no_confirm_raises_error(
        self, server: MCPServerV2
    ) -> None:
        """Delete ensemble without confirmation raises error."""
        with pytest.raises(ValueError, match="Confirmation required"):
            await server.call_tool(
                "delete_ensemble", {"ensemble_name": "test", "confirm": False}
            )

    @pytest.mark.asyncio
    async def test_call_tool_library_copy_missing_source_raises_error(
        self, server: MCPServerV2
    ) -> None:
        """Library copy without source raises error."""
        with pytest.raises(ValueError, match="source is required"):
            await server.call_tool("library_copy", {})


class TestMCPServerV2CreateEnsemble:
    """Tests for create_ensemble tool."""

    @pytest.mark.asyncio
    async def test_create_ensemble_success(
        self, server: MCPServerV2, tmp_path: Path
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
        self, server: MCPServerV2, tmp_path: Path
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
        self, server: MCPServerV2, tmp_path: Path
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


class TestMCPServerV2DeleteEnsemble:
    """Tests for delete_ensemble tool."""

    @pytest.mark.asyncio
    async def test_delete_ensemble_success(
        self, server: MCPServerV2, tmp_path: Path
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
        self, server: MCPServerV2, tmp_path: Path
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


class TestMCPServerV2ListScripts:
    """Tests for list_scripts tool."""

    @pytest.mark.asyncio
    async def test_list_scripts_empty(
        self, server: MCPServerV2, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """List scripts returns empty when no scripts exist."""
        monkeypatch.chdir(tmp_path)

        result = await server.call_tool("list_scripts", {})

        assert result["scripts"] == []

    @pytest.mark.asyncio
    async def test_list_scripts_finds_scripts(
        self, server: MCPServerV2, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
        self, server: MCPServerV2, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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


class TestMCPServerV2LibraryBrowse:
    """Tests for library_browse tool."""

    @pytest.mark.asyncio
    async def test_library_browse_empty(
        self, server: MCPServerV2, tmp_path: Path
    ) -> None:
        """Browse empty library returns empty lists."""
        server._test_library_dir = tmp_path / "empty-library"
        server._test_library_dir.mkdir()

        result = await server.call_tool("library_browse", {})

        assert result["ensembles"] == []
        assert result["scripts"] == []

    @pytest.mark.asyncio
    async def test_library_browse_ensembles_only(
        self, server: MCPServerV2, tmp_path: Path
    ) -> None:
        """Browse library for ensembles only."""
        library_dir = tmp_path / "library"
        ensembles_dir = library_dir / "ensembles"
        ensembles_dir.mkdir(parents=True)
        (ensembles_dir / "test.yaml").write_text(
            "name: test\ndescription: Test\nagents: []"
        )

        server._test_library_dir = library_dir

        result = await server.call_tool("library_browse", {"type": "ensembles"})

        assert "ensembles" in result
        assert "scripts" not in result


class TestMCPServerV2LibraryCopy:
    """Tests for library_copy tool."""

    @pytest.mark.asyncio
    async def test_library_copy_success(
        self, server: MCPServerV2, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Copy from library succeeds."""
        monkeypatch.chdir(tmp_path)

        library_dir = tmp_path / "library"
        ensembles_dir = library_dir / "ensembles"
        ensembles_dir.mkdir(parents=True)
        (ensembles_dir / "to-copy.yaml").write_text("name: to-copy\nagents: []")

        local_dir = tmp_path / ".llm-orc" / "ensembles"
        local_dir.mkdir(parents=True)

        server._test_library_dir = library_dir
        _mock_config(server).get_ensembles_dirs.return_value = [str(local_dir)]

        result = await server.call_tool(
            "library_copy",
            {"source": "ensembles/to-copy.yaml"},
        )

        assert result["copied"] is True
        assert (local_dir / "to-copy.yaml").exists()

    @pytest.mark.asyncio
    async def test_library_copy_source_not_found_raises_error(
        self, server: MCPServerV2, tmp_path: Path
    ) -> None:
        """Copy from non-existent source raises error."""
        library_dir = tmp_path / "library"
        library_dir.mkdir()

        server._test_library_dir = library_dir

        with pytest.raises(ValueError, match="not found in library"):
            await server.call_tool(
                "library_copy",
                {"source": "ensembles/missing.yaml"},
            )

    @pytest.mark.asyncio
    async def test_library_copy_exists_no_overwrite_raises_error(
        self, server: MCPServerV2, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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

        server._test_library_dir = library_dir
        _mock_config(server).get_ensembles_dirs.return_value = [str(local_dir)]

        with pytest.raises(ValueError, match="already exists"):
            await server.call_tool(
                "library_copy",
                {"source": "ensembles/exists.yaml", "overwrite": False},
            )

    @pytest.mark.asyncio
    async def test_library_copy_with_overwrite_succeeds(
        self, server: MCPServerV2, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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

        server._test_library_dir = library_dir
        _mock_config(server).get_ensembles_dirs.return_value = [str(local_dir)]

        result = await server.call_tool(
            "library_copy",
            {"source": "ensembles/exists.yaml", "overwrite": True},
        )

        assert result["copied"] is True
        content = (local_dir / "exists.yaml").read_text()
        assert "library-version" in content


class TestMCPServerV2GetLibraryDir:
    """Tests for _get_library_dir method."""

    def test_get_library_dir_from_test_override(
        self, server: MCPServerV2, tmp_path: Path
    ) -> None:
        """Test override takes precedence."""
        server._test_library_dir = tmp_path / "test-lib"

        result = server._get_library_dir()

        assert result == tmp_path / "test-lib"

    def test_get_library_dir_from_ensemble_dirs(
        self, server: MCPServerV2, tmp_path: Path
    ) -> None:
        """Finds library from ensemble dirs."""
        library_dir = tmp_path / "llm-orchestra-library" / "ensembles"
        _mock_config(server).get_ensembles_dirs.return_value = [str(library_dir)]

        result = server._get_library_dir()

        assert result == tmp_path / "llm-orchestra-library"

    def test_get_library_dir_default(
        self, server: MCPServerV2, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Falls back to default library location."""
        monkeypatch.chdir(tmp_path)
        _mock_config(server).get_ensembles_dirs.return_value = []

        result = server._get_library_dir()

        assert result == tmp_path / "llm-orchestra-library"

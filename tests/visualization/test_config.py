"""Comprehensive tests for visualization configuration."""

from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from llm_orc.visualization.config import (
    DebugVisualizationConfig,
    ExportConfig,
    SimpleVisualizationConfig,
    TerminalVisualizationConfig,
    VisualizationConfig,
    WebVisualizationConfig,
    load_visualization_config,
    save_visualization_config,
)


class TestVisualizationConfig:
    """Test main visualization configuration."""

    def test_default_initialization(self) -> None:
        """Test default configuration initialization."""
        config = VisualizationConfig()

        assert config.default_mode == "simple"
        assert config.enabled is True
        assert config.event_buffer_size == 1000
        assert isinstance(config.simple, SimpleVisualizationConfig)
        assert isinstance(config.terminal, TerminalVisualizationConfig)
        assert isinstance(config.web, WebVisualizationConfig)
        assert isinstance(config.debug, DebugVisualizationConfig)
        assert isinstance(config.export, ExportConfig)

    def test_from_dict_empty_dict(self) -> None:
        """Test creating config from empty dictionary."""
        config = VisualizationConfig.from_dict({})

        # Note: from_dict has different default
        assert config.default_mode == "terminal"
        assert config.enabled is True
        assert config.event_buffer_size == 1000

    def test_from_dict_global_settings(self) -> None:
        """Test creating config with global settings."""
        data = {
            "default_mode": "web",
            "enabled": False,
            "event_buffer_size": 2000,
        }

        config = VisualizationConfig.from_dict(data)

        assert config.default_mode == "web"
        assert config.enabled is False
        assert config.event_buffer_size == 2000

    def test_from_dict_terminal_config(self) -> None:
        """Test creating config with terminal configuration."""
        data = {
            "terminal": {
                "show_progress_bars": False,
                "show_overall_progress": False,
                "show_agent_progress": False,
                "show_time_estimates": False,
                "show_agent_status": False,
                "show_live_results": True,
                "show_dependencies": False,
                "show_performance_metrics": False,
                "use_colors": False,
                "use_emojis": False,
                "compact_mode": True,
                "refresh_rate_ms": 200,
                "agent_settings": {"agent1": {"key": "value"}},
            }
        }

        config = VisualizationConfig.from_dict(data)

        assert config.terminal.show_progress_bars is False
        assert config.terminal.show_overall_progress is False
        assert config.terminal.show_agent_progress is False
        assert config.terminal.show_time_estimates is False
        assert config.terminal.show_agent_status is False
        assert config.terminal.show_live_results is True
        assert config.terminal.show_dependencies is False
        assert config.terminal.show_performance_metrics is False
        assert config.terminal.use_colors is False
        assert config.terminal.use_emojis is False
        assert config.terminal.compact_mode is True
        assert config.terminal.refresh_rate_ms == 200
        assert config.terminal.agent_settings == {"agent1": {"key": "value"}}

    def test_from_dict_web_config(self) -> None:
        """Test creating config with web configuration."""
        data = {
            "web": {
                "enabled": True,
                "port": 9090,
                "host": "0.0.0.0",
                "auto_open_browser": False,
                "show_dependency_graph": False,
                "show_performance_charts": False,
                "show_live_logs": False,
                "websocket_update_rate_ms": 500,
            }
        }

        config = VisualizationConfig.from_dict(data)

        assert config.web.enabled is True
        assert config.web.port == 9090
        assert config.web.host == "0.0.0.0"
        assert config.web.auto_open_browser is False
        assert config.web.show_dependency_graph is False
        assert config.web.show_performance_charts is False
        assert config.web.show_live_logs is False
        assert config.web.websocket_update_rate_ms == 500

    def test_from_dict_debug_config(self) -> None:
        """Test creating config with debug configuration."""
        data = {
            "debug": {
                "enabled": True,
                "step_mode": True,
                "show_intermediate_results": False,
                "verbose_logging": False,
                "breakpoints": ["agent1", "agent2"],
                "break_on_error": False,
                "break_on_timeout": True,
                "allow_agent_inspection": False,
                "allow_result_modification": True,
            }
        }

        config = VisualizationConfig.from_dict(data)

        assert config.debug.enabled is True
        assert config.debug.step_mode is True
        assert config.debug.show_intermediate_results is False
        assert config.debug.verbose_logging is False
        assert config.debug.breakpoints == ["agent1", "agent2"]
        assert config.debug.break_on_error is False
        assert config.debug.break_on_timeout is True
        assert config.debug.allow_agent_inspection is False
        assert config.debug.allow_result_modification is True

    def test_from_dict_export_config(self) -> None:
        """Test creating config with export configuration."""
        data = {
            "export": {
                "save_execution_logs": False,
                "generate_reports": True,
                "output_directory": "/custom/path",
                "export_json": False,
                "export_html": True,
                "export_csv": True,
                "max_log_files": 50,
                "max_log_age_days": 7,
            }
        }

        config = VisualizationConfig.from_dict(data)

        assert config.export.save_execution_logs is False
        assert config.export.generate_reports is True
        assert config.export.output_directory == Path("/custom/path")
        assert config.export.export_json is False
        assert config.export.export_html is True
        assert config.export.export_csv is True
        assert config.export.max_log_files == 50
        assert config.export.max_log_age_days == 7

    def test_from_dict_complete_config(self) -> None:
        """Test creating config with all sections."""
        data = {
            "default_mode": "debug",
            "enabled": True,
            "event_buffer_size": 5000,
            "terminal": {"compact_mode": True},
            "web": {"port": 3000},
            "debug": {"enabled": True},
            "export": {"max_log_files": 200},
        }

        config = VisualizationConfig.from_dict(data)

        assert config.default_mode == "debug"
        assert config.enabled is True
        assert config.event_buffer_size == 5000
        assert config.terminal.compact_mode is True
        assert config.web.port == 3000
        assert config.debug.enabled is True
        assert config.export.max_log_files == 200

    def test_to_dict(self) -> None:
        """Test converting config to dictionary."""
        config = VisualizationConfig()
        config.default_mode = "web"
        config.enabled = False
        config.event_buffer_size = 1500

        result = config.to_dict()

        assert result["default_mode"] == "web"
        assert result["enabled"] is False
        assert result["event_buffer_size"] == 1500
        assert "terminal" in result
        assert "web" in result
        assert "debug" in result
        assert "export" in result

    def test_to_dict_terminal_section(self) -> None:
        """Test terminal section in to_dict output."""
        config = VisualizationConfig()
        config.terminal.compact_mode = True
        config.terminal.refresh_rate_ms = 50

        result = config.to_dict()

        assert result["terminal"]["compact_mode"] is True
        assert result["terminal"]["refresh_rate_ms"] == 50
        assert result["terminal"]["show_progress_bars"] is True  # Default

    def test_to_dict_web_section(self) -> None:
        """Test web section in to_dict output."""
        config = VisualizationConfig()
        config.web.enabled = True
        config.web.port = 4000

        result = config.to_dict()

        assert result["web"]["enabled"] is True
        assert result["web"]["port"] == 4000
        assert result["web"]["host"] == "localhost"  # Default

    def test_to_dict_debug_section(self) -> None:
        """Test debug section in to_dict output."""
        config = VisualizationConfig()
        config.debug.enabled = True
        config.debug.breakpoints = ["test_agent"]

        result = config.to_dict()

        assert result["debug"]["enabled"] is True
        assert result["debug"]["breakpoints"] == ["test_agent"]
        assert result["debug"]["step_mode"] is False  # Default

    def test_to_dict_export_section(self) -> None:
        """Test export section in to_dict output."""
        config = VisualizationConfig()
        config.export.output_directory = Path("/test/path")
        config.export.max_log_files = 75

        result = config.to_dict()

        assert result["export"]["output_directory"] == "/test/path"
        assert result["export"]["max_log_files"] == 75
        assert result["export"]["save_execution_logs"] is True  # Default


class TestLoadVisualizationConfig:
    """Test loading visualization configuration from files."""

    def test_load_with_none_path_no_config_exists(self) -> None:
        """Test loading with None path when no config files exist."""
        with patch("pathlib.Path.exists", return_value=False):
            config = load_visualization_config(None)

        assert isinstance(config, VisualizationConfig)
        assert config.default_mode == "simple"  # Default config

    def test_load_with_none_path_local_config_exists(self) -> None:
        """Test loading with None path when local config exists."""
        mock_data = {"default_mode": "terminal", "enabled": True}

        with patch("pathlib.Path.exists") as mock_exists, patch(
            "builtins.open", mock_open(read_data="default_mode: terminal")
        ), patch("yaml.safe_load", return_value=mock_data):
            # Local path exists, home path doesn't matter
            mock_exists.side_effect = lambda: True

            config = load_visualization_config(None)

        assert config.default_mode == "terminal"

    def test_load_with_none_path_home_config_exists(self) -> None:
        """Test loading with None path when only home config exists."""
        mock_data = {"default_mode": "web", "enabled": False}

        with patch("pathlib.Path.exists") as mock_exists, patch(
            "builtins.open", mock_open(read_data="default_mode: web")
        ), patch("yaml.safe_load", return_value=mock_data):
            # First call (local) returns False, second call (home) returns True
            mock_exists.side_effect = [False, True]

            config = load_visualization_config(None)

        assert config.default_mode == "web"
        assert config.enabled is False

    def test_load_with_specific_path_exists(self) -> None:
        """Test loading with specific config path that exists."""
        config_path = Path("/custom/config.yaml")
        mock_data = {"default_mode": "debug", "event_buffer_size": 2000}

        with patch("pathlib.Path.exists", return_value=True), patch(
            "builtins.open", mock_open(read_data="default_mode: debug")
        ), patch("yaml.safe_load", return_value=mock_data):
            config = load_visualization_config(config_path)

        assert config.default_mode == "debug"
        assert config.event_buffer_size == 2000

    def test_load_with_specific_path_not_exists(self) -> None:
        """Test loading with specific config path that doesn't exist."""
        config_path = Path("/nonexistent/config.yaml")

        with patch("pathlib.Path.exists", return_value=False):
            config = load_visualization_config(config_path)

        assert isinstance(config, VisualizationConfig)
        assert config.default_mode == "simple"  # Default config

    def test_load_yaml_import_error(self) -> None:
        """Test loading when yaml import fails."""
        config_path = Path("/test/config.yaml")

        with patch("pathlib.Path.exists", return_value=True), patch(
            "builtins.open", mock_open()
        ), patch("yaml.safe_load", side_effect=ImportError("No yaml module")):
            config = load_visualization_config(config_path)

        assert isinstance(config, VisualizationConfig)
        assert config.default_mode == "simple"  # Default config on error

    def test_load_yaml_parse_error(self) -> None:
        """Test loading when yaml parsing fails."""
        config_path = Path("/test/config.yaml")

        with patch("pathlib.Path.exists", return_value=True), patch(
            "builtins.open", mock_open(read_data="invalid: yaml: content")
        ), patch("yaml.safe_load", side_effect=Exception("YAML parse error")):
            config = load_visualization_config(config_path)

        assert isinstance(config, VisualizationConfig)
        assert config.default_mode == "simple"  # Default config on error

    def test_load_file_read_error(self) -> None:
        """Test loading when file reading fails."""
        config_path = Path("/test/config.yaml")

        with patch("pathlib.Path.exists", return_value=True), patch(
            "builtins.open", side_effect=OSError("Cannot read file")
        ):
            config = load_visualization_config(config_path)

        assert isinstance(config, VisualizationConfig)
        assert config.default_mode == "simple"  # Default config on error


class TestSaveVisualizationConfig:
    """Test saving visualization configuration to files."""

    def test_save_with_none_path(self) -> None:
        """Test saving with None path uses default location."""
        config = VisualizationConfig()
        config.default_mode = "web"

        mock_file = mock_open()
        with patch("pathlib.Path.mkdir") as mock_mkdir, patch(
            "builtins.open", mock_file
        ), patch("yaml.safe_dump") as mock_dump:
            save_visualization_config(config, None)

        # Check that mkdir was called to create directory
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

        # Check that file was opened for writing
        mock_file.assert_called_once()
        call_args = mock_file.call_args
        assert str(call_args[0][0]).endswith("visualization.yaml")
        assert call_args[0][1] == "w"

        # Check that yaml.safe_dump was called
        mock_dump.assert_called_once()
        dump_args = mock_dump.call_args
        assert dump_args[1]["default_flow_style"] is False
        assert dump_args[1]["indent"] == 2

    def test_save_with_specific_path(self) -> None:
        """Test saving with specific config path."""
        config = VisualizationConfig()
        config_path = Path("/custom/path/config.yaml")

        mock_file = mock_open()
        with patch("pathlib.Path.mkdir") as mock_mkdir, patch(
            "builtins.open", mock_file
        ), patch("yaml.safe_dump") as mock_dump:
            save_visualization_config(config, config_path)

        # Check that mkdir was called on the parent directory
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

        # Check that file was opened with correct path
        mock_file.assert_called_once_with(config_path, "w")

        # Check that yaml.safe_dump was called
        mock_dump.assert_called_once()

    def test_save_yaml_import_error(self) -> None:
        """Test saving when yaml import fails."""
        config = VisualizationConfig()
        config_path = Path("/test/config.yaml")

        with patch("pathlib.Path.mkdir"), patch("builtins.open", mock_open()), patch(
            "yaml.safe_dump", side_effect=ImportError("No yaml module")
        ), pytest.raises(
            RuntimeError, match="Failed to save visualization configuration"
        ):
            save_visualization_config(config, config_path)

    def test_save_file_write_error(self) -> None:
        """Test saving when file writing fails."""
        config = VisualizationConfig()
        config_path = Path("/test/config.yaml")

        with patch("pathlib.Path.mkdir"), patch(
            "builtins.open", side_effect=OSError("Cannot write file")
        ), pytest.raises(
            RuntimeError, match="Failed to save visualization configuration"
        ):
            save_visualization_config(config, config_path)

    def test_save_yaml_dump_error(self) -> None:
        """Test saving when yaml dumping fails."""
        config = VisualizationConfig()
        config_path = Path("/test/config.yaml")

        with patch("pathlib.Path.mkdir"), patch("builtins.open", mock_open()), patch(
            "yaml.safe_dump", side_effect=Exception("YAML dump error")
        ), pytest.raises(
            RuntimeError, match="Failed to save visualization configuration"
        ):
            save_visualization_config(config, config_path)

    def test_save_creates_parent_directory(self) -> None:
        """Test that saving creates parent directories."""
        config = VisualizationConfig()
        config_path = Path("/deep/nested/path/config.yaml")

        with patch("pathlib.Path.mkdir") as mock_mkdir, patch(
            "builtins.open", mock_open()
        ), patch("yaml.safe_dump"):
            save_visualization_config(config, config_path)

        # Verify mkdir was called with correct arguments
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)


class TestSubconfigurations:
    """Test individual configuration dataclasses."""

    def test_simple_visualization_config_defaults(self) -> None:
        """Test SimpleVisualizationConfig default values."""
        config = SimpleVisualizationConfig()

        assert config.show_dependency_graph is True
        assert config.show_basic_metrics is True
        assert config.use_colors is True
        assert config.use_emojis is True
        assert config.refresh_rate_ms == 500

    def test_terminal_visualization_config_defaults(self) -> None:
        """Test TerminalVisualizationConfig default values."""
        config = TerminalVisualizationConfig()

        assert config.show_progress_bars is True
        assert config.show_overall_progress is True
        assert config.show_agent_progress is True
        assert config.show_time_estimates is True
        assert config.show_agent_status is True
        assert config.show_live_results is False
        assert config.show_dependencies is True
        assert config.show_performance_metrics is True
        assert config.use_colors is True
        assert config.use_emojis is True
        assert config.compact_mode is False
        assert config.refresh_rate_ms == 100
        assert config.agent_settings == {}

    def test_web_visualization_config_defaults(self) -> None:
        """Test WebVisualizationConfig default values."""
        config = WebVisualizationConfig()

        assert config.enabled is False
        assert config.port == 8080
        assert config.host == "localhost"
        assert config.auto_open_browser is True
        assert config.show_dependency_graph is True
        assert config.show_performance_charts is True
        assert config.show_live_logs is True
        assert config.websocket_update_rate_ms == 250

    def test_debug_visualization_config_defaults(self) -> None:
        """Test DebugVisualizationConfig default values."""
        config = DebugVisualizationConfig()

        assert config.enabled is False
        assert config.step_mode is False
        assert config.show_intermediate_results is True
        assert config.verbose_logging is True
        assert config.breakpoints == []
        assert config.break_on_error is True
        assert config.break_on_timeout is False
        assert config.allow_agent_inspection is True
        assert config.allow_result_modification is False

    def test_export_config_defaults(self) -> None:
        """Test ExportConfig default values."""
        config = ExportConfig()

        assert config.save_execution_logs is True
        assert config.generate_reports is False
        assert config.output_directory == Path("./llm-orc-logs")
        assert config.export_json is True
        assert config.export_html is False
        assert config.export_csv is False
        assert config.max_log_files == 100
        assert config.max_log_age_days == 30


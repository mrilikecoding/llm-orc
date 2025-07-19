"""Configuration for visualization system."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SimpleVisualizationConfig:
    """Configuration for simple horizontal dependency graph visualization."""

    # What to show
    show_dependency_graph: bool = True
    show_basic_metrics: bool = True

    # Styling
    use_colors: bool = True
    use_emojis: bool = True

    # Update frequency
    refresh_rate_ms: int = 500


@dataclass
class TerminalVisualizationConfig:
    """Configuration for terminal-based visualization."""

    # Progress display
    show_progress_bars: bool = True
    show_overall_progress: bool = True
    show_agent_progress: bool = True
    show_time_estimates: bool = True

    # Status display
    show_agent_status: bool = True
    show_live_results: bool = False  # Can be verbose
    show_dependencies: bool = True
    show_performance_metrics: bool = True

    # Styling
    use_colors: bool = True
    use_emojis: bool = True
    compact_mode: bool = False

    # Update frequency
    refresh_rate_ms: int = 100

    # Agent-specific settings
    agent_settings: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class WebVisualizationConfig:
    """Configuration for web-based visualization."""

    enabled: bool = False
    port: int = 8080
    host: str = "localhost"
    auto_open_browser: bool = True

    # Dashboard features
    show_dependency_graph: bool = True
    show_performance_charts: bool = True
    show_live_logs: bool = True

    # Update frequency
    websocket_update_rate_ms: int = 250


@dataclass
class DebugVisualizationConfig:
    """Configuration for debug mode visualization."""

    enabled: bool = False
    step_mode: bool = False
    show_intermediate_results: bool = True
    verbose_logging: bool = True

    # Breakpoints
    breakpoints: list[str] = field(default_factory=list)
    break_on_error: bool = True
    break_on_timeout: bool = False

    # Interactive features
    allow_agent_inspection: bool = True
    allow_result_modification: bool = False


@dataclass
class ExportConfig:
    """Configuration for execution export and logging."""

    save_execution_logs: bool = True
    generate_reports: bool = False
    output_directory: Path = Path("./llm-orc-logs")

    # Export formats
    export_json: bool = True
    export_html: bool = False
    export_csv: bool = False

    # Retention
    max_log_files: int = 100
    max_log_age_days: int = 30


@dataclass
class VisualizationConfig:
    """Main visualization configuration."""

    default_mode: str = "simple"  # simple, terminal, web, minimal, debug

    # Mode-specific configurations
    simple: SimpleVisualizationConfig = field(default_factory=SimpleVisualizationConfig)
    terminal: TerminalVisualizationConfig = field(
        default_factory=TerminalVisualizationConfig
    )
    web: WebVisualizationConfig = field(default_factory=WebVisualizationConfig)
    debug: DebugVisualizationConfig = field(default_factory=DebugVisualizationConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    # Global settings
    enabled: bool = True
    event_buffer_size: int = 1000

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VisualizationConfig":
        """Create configuration from dictionary."""
        config = cls()

        # Set global settings
        config.default_mode = data.get("default_mode", "terminal")
        config.enabled = data.get("enabled", True)
        config.event_buffer_size = data.get("event_buffer_size", 1000)

        # Set terminal configuration
        if "terminal" in data:
            terminal_data = data["terminal"]
            config.terminal = TerminalVisualizationConfig(
                show_progress_bars=terminal_data.get("show_progress_bars", True),
                show_overall_progress=terminal_data.get("show_overall_progress", True),
                show_agent_progress=terminal_data.get("show_agent_progress", True),
                show_time_estimates=terminal_data.get("show_time_estimates", True),
                show_agent_status=terminal_data.get("show_agent_status", True),
                show_live_results=terminal_data.get("show_live_results", False),
                show_dependencies=terminal_data.get("show_dependencies", True),
                show_performance_metrics=terminal_data.get(
                    "show_performance_metrics", True
                ),
                use_colors=terminal_data.get("use_colors", True),
                use_emojis=terminal_data.get("use_emojis", True),
                compact_mode=terminal_data.get("compact_mode", False),
                refresh_rate_ms=terminal_data.get("refresh_rate_ms", 100),
                agent_settings=terminal_data.get("agent_settings", {}),
            )

        # Set web configuration
        if "web" in data:
            web_data = data["web"]
            config.web = WebVisualizationConfig(
                enabled=web_data.get("enabled", False),
                port=web_data.get("port", 8080),
                host=web_data.get("host", "localhost"),
                auto_open_browser=web_data.get("auto_open_browser", True),
                show_dependency_graph=web_data.get("show_dependency_graph", True),
                show_performance_charts=web_data.get("show_performance_charts", True),
                show_live_logs=web_data.get("show_live_logs", True),
                websocket_update_rate_ms=web_data.get("websocket_update_rate_ms", 250),
            )

        # Set debug configuration
        if "debug" in data:
            debug_data = data["debug"]
            config.debug = DebugVisualizationConfig(
                enabled=debug_data.get("enabled", False),
                step_mode=debug_data.get("step_mode", False),
                show_intermediate_results=debug_data.get(
                    "show_intermediate_results", True
                ),
                verbose_logging=debug_data.get("verbose_logging", True),
                breakpoints=debug_data.get("breakpoints", []),
                break_on_error=debug_data.get("break_on_error", True),
                break_on_timeout=debug_data.get("break_on_timeout", False),
                allow_agent_inspection=debug_data.get("allow_agent_inspection", True),
                allow_result_modification=debug_data.get(
                    "allow_result_modification", False
                ),
            )

        # Set export configuration
        if "export" in data:
            export_data = data["export"]
            config.export = ExportConfig(
                save_execution_logs=export_data.get("save_execution_logs", True),
                generate_reports=export_data.get("generate_reports", False),
                output_directory=Path(
                    export_data.get("output_directory", "./llm-orc-logs")
                ),
                export_json=export_data.get("export_json", True),
                export_html=export_data.get("export_html", False),
                export_csv=export_data.get("export_csv", False),
                max_log_files=export_data.get("max_log_files", 100),
                max_log_age_days=export_data.get("max_log_age_days", 30),
            )

        return config

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "default_mode": self.default_mode,
            "enabled": self.enabled,
            "event_buffer_size": self.event_buffer_size,
            "terminal": {
                "show_progress_bars": self.terminal.show_progress_bars,
                "show_overall_progress": self.terminal.show_overall_progress,
                "show_agent_progress": self.terminal.show_agent_progress,
                "show_time_estimates": self.terminal.show_time_estimates,
                "show_agent_status": self.terminal.show_agent_status,
                "show_live_results": self.terminal.show_live_results,
                "show_dependencies": self.terminal.show_dependencies,
                "show_performance_metrics": self.terminal.show_performance_metrics,
                "use_colors": self.terminal.use_colors,
                "use_emojis": self.terminal.use_emojis,
                "compact_mode": self.terminal.compact_mode,
                "refresh_rate_ms": self.terminal.refresh_rate_ms,
                "agent_settings": self.terminal.agent_settings,
            },
            "web": {
                "enabled": self.web.enabled,
                "port": self.web.port,
                "host": self.web.host,
                "auto_open_browser": self.web.auto_open_browser,
                "show_dependency_graph": self.web.show_dependency_graph,
                "show_performance_charts": self.web.show_performance_charts,
                "show_live_logs": self.web.show_live_logs,
                "websocket_update_rate_ms": self.web.websocket_update_rate_ms,
            },
            "debug": {
                "enabled": self.debug.enabled,
                "step_mode": self.debug.step_mode,
                "show_intermediate_results": self.debug.show_intermediate_results,
                "verbose_logging": self.debug.verbose_logging,
                "breakpoints": self.debug.breakpoints,
                "break_on_error": self.debug.break_on_error,
                "break_on_timeout": self.debug.break_on_timeout,
                "allow_agent_inspection": self.debug.allow_agent_inspection,
                "allow_result_modification": self.debug.allow_result_modification,
            },
            "export": {
                "save_execution_logs": self.export.save_execution_logs,
                "generate_reports": self.export.generate_reports,
                "output_directory": str(self.export.output_directory),
                "export_json": self.export.export_json,
                "export_html": self.export.export_html,
                "export_csv": self.export.export_csv,
                "max_log_files": self.export.max_log_files,
                "max_log_age_days": self.export.max_log_age_days,
            },
        }


def load_visualization_config(config_path: Path | None = None) -> VisualizationConfig:
    """Load visualization configuration from file."""
    if config_path is None:
        # Default configuration locations
        config_path = Path(".llm-orc/visualization.yaml")
        if not config_path.exists():
            config_path = Path.home() / ".llm-orc" / "visualization.yaml"

    if not config_path.exists():
        return VisualizationConfig()

    try:
        import yaml

        with open(config_path) as f:
            data = yaml.safe_load(f)
        return VisualizationConfig.from_dict(data)
    except Exception:
        # Return default configuration if loading fails
        return VisualizationConfig()


def save_visualization_config(
    config: VisualizationConfig, config_path: Path | None = None
) -> None:
    """Save visualization configuration to file."""
    if config_path is None:
        config_path = Path(".llm-orc/visualization.yaml")

    # Create directory if it doesn't exist
    config_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import yaml

        with open(config_path, "w") as f:
            yaml.safe_dump(config.to_dict(), f, default_flow_style=False, indent=2)
    except Exception as e:
        raise RuntimeError(f"Failed to save visualization configuration: {e}") from e

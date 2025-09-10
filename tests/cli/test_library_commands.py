"""Tests for library CLI commands."""

from pathlib import Path
from typing import Any
from unittest.mock import mock_open, patch

import click
import pytest
import requests
from click.testing import CliRunner

from llm_orc.cli import cli


class TestLibraryBrowseCommand:
    """Test library browse command functionality."""

    def test_library_browse_lists_all_categories(self) -> None:
        """Should list all available ensemble categories."""
        runner = CliRunner()

        with patch(
            "llm_orc.cli_library.library.get_library_categories"
        ) as mock_get_categories:
            mock_get_categories.return_value = [
                "code-analysis",
                "idea-exploration",
                "research-analysis",
                "decision-support",
                "problem-decomposition",
                "learning-facilitation",
            ]

            result = runner.invoke(cli, ["library", "browse"])

            assert result.exit_code == 0
            assert "code-analysis" in result.output
            assert "idea-exploration" in result.output
            assert "research-analysis" in result.output

    def test_library_browse_specific_category(self) -> None:
        """Should list ensembles in a specific category."""
        runner = CliRunner()

        with patch(
            "llm_orc.cli_library.library.get_category_ensembles"
        ) as mock_get_ensembles:
            mock_get_ensembles.return_value = [
                {
                    "name": "security-review",
                    "description": "Multi-perspective security analysis ensemble",
                    "path": "code-analysis/security-review.yaml",
                }
            ]

            result = runner.invoke(cli, ["library", "browse", "code-analysis"])

            assert result.exit_code == 0
            assert "security-review" in result.output
            assert "Multi-perspective security analysis" in result.output

    def test_library_browse_invalid_category(self) -> None:
        """Should show error for invalid category."""
        runner = CliRunner()

        with patch(
            "llm_orc.cli_library.library.get_category_ensembles"
        ) as mock_get_ensembles:
            mock_get_ensembles.return_value = []

            result = runner.invoke(cli, ["library", "browse", "invalid-category"])

            assert result.exit_code == 0
            assert (
                "No ensembles found" in result.output
                or "invalid-category" in result.output
            )


class TestLibraryCopyCommand:
    """Test library copy command functionality."""

    def test_library_copy_to_local_config(self) -> None:
        """Should copy ensemble to local .llm-orc/ensembles/ directory."""
        runner = CliRunner()

        ensemble_content = """
name: test-ensemble
description: Test ensemble
agents:
  - name: test-agent
    model_profile: micro-local
"""

        with (
            patch("llm_orc.cli_library.library.fetch_ensemble_content") as mock_fetch,
            patch(
                "llm_orc.cli_library.library.ensure_local_ensembles_dir"
            ) as mock_ensure_dir,
            patch("builtins.open", mock_open()),
        ):
            mock_fetch.return_value = ensemble_content
            mock_ensure_dir.return_value = ".llm-orc/ensembles"

            result = runner.invoke(
                cli, ["library", "copy", "code-analysis/security-review"]
            )

            assert result.exit_code == 0
            assert "Copied" in result.output
            assert "test-ensemble" in result.output
            mock_fetch.assert_called_once_with("code-analysis/security-review")

    def test_library_copy_to_global_config(self) -> None:
        """Should copy ensemble to global config when --global flag used."""
        runner = CliRunner()

        ensemble_content = """
name: test-ensemble
description: Test ensemble
agents:
  - name: test-agent
    model_profile: default
"""

        with (
            patch("llm_orc.cli_library.library.fetch_ensemble_content") as mock_fetch,
            patch(
                "llm_orc.cli_library.library.ensure_global_ensembles_dir"
            ) as mock_ensure_dir,
            patch("builtins.open", mock_open()),
        ):
            mock_fetch.return_value = ensemble_content
            mock_ensure_dir.return_value = "/home/user/.config/llm-orc/ensembles"

            result = runner.invoke(
                cli, ["library", "copy", "idea-exploration/concept-mapper", "--global"]
            )

            assert result.exit_code == 0
            assert "Copied" in result.output
            assert "test-ensemble" in result.output
            mock_fetch.assert_called_once_with("idea-exploration/concept-mapper")

    def test_library_copy_invalid_ensemble(self) -> None:
        """Should show error for invalid ensemble path."""
        runner = CliRunner()

        with patch("llm_orc.cli_library.library.fetch_ensemble_content") as mock_fetch:
            mock_fetch.side_effect = FileNotFoundError("Ensemble not found")

            result = runner.invoke(cli, ["library", "copy", "invalid/ensemble"])

            assert result.exit_code == 1
            assert "not found" in result.output.lower()

    def test_library_copy_overwrites_existing_with_confirmation(self) -> None:
        """Should prompt for confirmation when overwriting existing ensemble."""
        runner = CliRunner()

        ensemble_content = "name: existing-ensemble"

        with (
            patch("llm_orc.cli_library.library.fetch_ensemble_content") as mock_fetch,
            patch("llm_orc.cli_library.library.ensemble_exists") as mock_exists,
            patch(
                "llm_orc.cli_library.library.ensure_local_ensembles_dir"
            ) as mock_ensure_dir,
            patch("builtins.open", mock_open()),
        ):
            mock_fetch.return_value = ensemble_content
            mock_exists.return_value = True
            mock_ensure_dir.return_value = ".llm-orc/ensembles"

            # Test with 'y' input for confirmation
            result = runner.invoke(
                cli, ["library", "copy", "test/ensemble"], input="y\n"
            )

            assert result.exit_code == 0
            assert "already exists" in result.output
            assert "Copied" in result.output


class TestLibraryCategoriesCommand:
    """Test library categories command functionality."""

    def test_library_categories_lists_all(self) -> None:
        """Should list all available categories with descriptions."""
        runner = CliRunner()

        with patch(
            "llm_orc.cli_library.library.get_library_categories_with_descriptions"
        ) as mock_get_categories:
            mock_get_categories.return_value = [
                ("code-analysis", "Code review and security analysis"),
                ("idea-exploration", "Concept mapping and perspective taking"),
                ("research-analysis", "Literature review and synthesis"),
            ]

            result = runner.invoke(cli, ["library", "categories"])

            assert result.exit_code == 0
            assert "code-analysis" in result.output
            assert "Code review and security analysis" in result.output
            assert "idea-exploration" in result.output


class TestLibraryDynamicFetching:
    """Test dynamic fetching from GitHub repository."""

    def test_get_category_ensembles_fetches_dynamically(self) -> None:
        """Should fetch ensembles from GitHub API for all categories."""
        from llm_orc.cli_library.library import get_category_ensembles

        with patch("requests.get") as mock_get:
            # Mock API response
            mock_response = mock_get.return_value
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = [
                {
                    "name": "concept-mapper.yaml",
                    "type": "file",
                    "download_url": "https://raw.githubusercontent.com/mrilikecoding/llm-orchestra-library/main/idea-exploration/concept-mapper.yaml",
                },
                {
                    "name": "README.md",
                    "type": "file",
                    "download_url": "https://raw.githubusercontent.com/mrilikecoding/llm-orchestra-library/main/idea-exploration/README.md",
                },
            ]

            # Mock fetching ensemble content to get description
            with patch(
                "llm_orc.cli_library.library.fetch_ensemble_content"
            ) as mock_fetch:
                mock_fetch.return_value = """
name: concept-mapper
description: Concept mapping and perspective taking ensemble
agents:
  - name: mapper
    model_profile: default
"""

                ensembles = get_category_ensembles("idea-exploration")

                assert len(ensembles) == 1
                assert ensembles[0]["name"] == "concept-mapper"
                assert "concept mapping" in ensembles[0]["description"].lower()
                assert ensembles[0]["path"] == "idea-exploration/concept-mapper.yaml"

    def test_complete_library_ensemble_paths_uses_dynamic_fetching(self) -> None:
        """Should complete ensemble paths using dynamic GitHub API fetching."""

        from llm_orc.cli_completion import complete_library_ensemble_paths

        # Mock click context and parameter
        ctx = click.Context(click.Command("test"))
        param = click.Argument(["test"])

        with patch(
            "llm_orc.cli_library.library.get_category_ensembles"
        ) as mock_get_ensembles:
            mock_get_ensembles.return_value = [
                {
                    "name": "concept-mapper",
                    "description": "Concept mapping ensemble",
                    "path": "idea-exploration/concept-mapper.yaml",
                },
                {
                    "name": "perspective-taker",
                    "description": "Perspective taking ensemble",
                    "path": "idea-exploration/perspective-taker.yaml",
                },
            ]

            # Test completing ensemble names within a category
            completions = complete_library_ensemble_paths(
                ctx, param, "idea-exploration/con"
            )

            assert "idea-exploration/concept-mapper" in completions
            assert (
                "idea-exploration/perspective-taker" not in completions
            )  # doesn't match "con"


class TestLibraryTemplateFetching:
    """Test dynamic template fetching from GitHub repository."""

    def test_get_template_content_fetches_from_github(self) -> None:
        """Should fetch template content from GitHub API."""
        from llm_orc.cli_library.library import get_template_content

        with patch("requests.get") as mock_get:
            # Mock successful response
            mock_response = mock_get.return_value
            mock_response.raise_for_status.return_value = None
            mock_response.text = """# Local project configuration for {project_name}
project:
  name: "{project_name}"
"""

            content = get_template_content("local-config.yaml")

            assert "{project_name}" in content
            assert "Local project configuration" in content
            mock_get.assert_called_once_with(
                "https://raw.githubusercontent.com/mrilikecoding/llm-orchestra-library/main/templates/local-config.yaml",
                timeout=10,
            )

    def test_get_template_content_handles_missing_template(self) -> None:
        """Should handle missing templates gracefully."""
        from llm_orc.cli_library.library import get_template_content

        with patch("requests.get") as mock_get:
            # Mock 404 response
            mock_get.side_effect = requests.RequestException("Not found")

            with pytest.raises(FileNotFoundError, match="Template not found"):
                get_template_content("nonexistent-template.yaml")


class TestConfigurationManagerTemplateIntegration:
    """Test configuration manager integration with library templates."""

    def test_setup_default_config_uses_library_template(self) -> None:
        """Should use library template for default config setup."""
        from unittest.mock import patch

        from llm_orc.core.config.config_manager import ConfigurationManager

        with (
            patch("llm_orc.core.config.config_manager.Path.home") as mock_home,
            patch("llm_orc.core.config.config_manager.Path.mkdir"),
            patch("llm_orc.core.config.config_manager.Path.exists") as mock_exists,
            patch(
                "llm_orc.cli_library.library.get_template_content"
            ) as mock_get_template,
            patch("builtins.open", mock_open()) as mock_file,
        ):
            # Setup mock paths
            mock_home.return_value = Path("/home/test")
            mock_exists.return_value = False  # Config doesn't exist yet

            # Mock template content
            mock_get_template.return_value = """
model_profiles:
  default:
    model: claude-3-5-sonnet-20241022
    provider: anthropic
"""

            # Create config manager (triggers _setup_default_config)
            ConfigurationManager()

            # Verify template was fetched
            mock_get_template.assert_called_with("global-config.yaml")

            # Verify file was written
            mock_file.assert_called()

    def test_init_local_config_uses_library_template(self) -> None:
        """Should use library template for local config initialization."""
        from unittest.mock import patch

        from llm_orc.core.config.config_manager import ConfigurationManager

        with (
            patch("llm_orc.core.config.config_manager.Path.home") as mock_home,
            patch("llm_orc.core.config.config_manager.Path.cwd") as mock_cwd,
            patch("llm_orc.core.config.config_manager.Path.mkdir"),
            patch("llm_orc.core.config.config_manager.Path.exists") as mock_exists,
            patch(
                "llm_orc.cli_library.library.get_template_content"
            ) as mock_get_template,
            patch("builtins.open", mock_open()),
        ):
            # Setup mock paths
            mock_home.return_value = Path("/home/test")
            mock_cwd.return_value = Path("/project")
            mock_exists.return_value = False

            # Mock template content for initialization and local config
            mock_get_template.side_effect = [
                """# Global config template
model_profiles:
  default:
    model: claude-3-5-sonnet-20241022
    provider: anthropic
""",
                """# Local project configuration for {project_name}
project:
  name: "{project_name}"
""",
                """name: example-local-ensemble
description: Example ensemble
""",
            ]

            # Create config manager and initialize local config
            config_manager = ConfigurationManager()
            config_manager.init_local_config("test-project")

            # Verify templates were fetched (global + local + ensemble)
            assert mock_get_template.call_count == 3
            mock_get_template.assert_any_call("global-config.yaml")
            mock_get_template.assert_any_call("local-config.yaml")
            mock_get_template.assert_any_call("example-local-ensemble.yaml")

    def test_template_content_fallback_mechanism(self) -> None:
        """Should have fallback mechanism for template content retrieval."""
        from unittest.mock import patch

        from llm_orc.core.config.config_manager import ConfigurationManager

        # Test the template content method directly
        config_manager = ConfigurationManager()

        with (
            patch(
                "llm_orc.cli_library.library.get_template_content"
            ) as mock_get_template,
            patch("builtins.open", mock_open(read_data="fallback_content")),
            patch("llm_orc.core.config.config_manager.Path.exists", return_value=True),
        ):
            # Mock library template not found
            mock_get_template.side_effect = FileNotFoundError("Template not found")

            # Should fallback to local template
            result = config_manager._get_template_config_content("test.yaml")

            assert result == "fallback_content"
            mock_get_template.assert_called_with("test.yaml")


class TestLibraryIntegration:
    """Integration tests for library commands."""

    def test_browse_then_copy_workflow(self) -> None:
        """Should support browsing then copying an ensemble."""
        runner = CliRunner()

        # First browse to see available ensembles
        with patch(
            "llm_orc.cli_library.library.get_category_ensembles"
        ) as mock_get_ensembles:
            mock_get_ensembles.return_value = [
                {
                    "name": "security-review",
                    "description": "Security analysis",
                    "path": "code-analysis/security-review.yaml",
                }
            ]

            browse_result = runner.invoke(cli, ["library", "browse", "code-analysis"])
            assert browse_result.exit_code == 0
            assert "security-review" in browse_result.output

        # Then copy the ensemble
        ensemble_content = "name: security-review\ndescription: Security analysis"

        with (
            patch("llm_orc.cli_library.library.fetch_ensemble_content") as mock_fetch,
            patch(
                "llm_orc.cli_library.library.ensure_local_ensembles_dir"
            ) as mock_ensure_dir,
            patch("builtins.open", mock_open()),
            patch(
                "llm_orc.cli_library.library.ensemble_exists", return_value=False
            ),  # Mock ensemble doesn't exist
        ):
            mock_fetch.return_value = ensemble_content
            mock_ensure_dir.return_value = ".llm-orc/ensembles"

            copy_result = runner.invoke(
                cli, ["library", "copy", "code-analysis/security-review"]
            )
            assert copy_result.exit_code == 0
            assert "Copied" in copy_result.output


class TestLibraryGitHubAPIIntegration:
    """Test real GitHub API integration functionality."""

    def test_get_library_categories_with_descriptions(self) -> None:
        """Should return all library categories with descriptions."""
        from llm_orc.cli_library.library import get_library_categories_with_descriptions

        categories = get_library_categories_with_descriptions()

        assert isinstance(categories, list)
        assert len(categories) > 0

        # Verify structure: list of tuples (category, description)
        for category, description in categories:
            assert isinstance(category, str)
            assert isinstance(description, str)
            assert len(category) > 0
            assert len(description) > 0

        # Verify expected categories exist
        category_names = [cat[0] for cat in categories]
        assert "code-analysis" in category_names
        assert "idea-exploration" in category_names

    def test_get_category_ensembles_success(self) -> None:
        """Should fetch ensembles from GitHub API successfully."""
        from llm_orc.cli_library.library import get_category_ensembles

        mock_response_data = [
            {
                "type": "file",
                "name": "security-review.yaml",
                "download_url": "https://example.com/security-review.yaml",
            },
            {
                "type": "file",
                "name": "README.md",
                "download_url": "https://example.com/README.md",
            },
        ]

        mock_ensemble_content = """
name: security-review
description: Multi-perspective security analysis
agents:
  - name: security-expert
    model: claude-3-5-sonnet
"""

        with (
            patch("requests.get") as mock_get,
            patch("llm_orc.cli_library.library.fetch_ensemble_content") as mock_fetch,
        ):
            # Mock API response
            mock_response = mock_get.return_value
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = mock_response_data

            # Mock ensemble content fetch
            mock_fetch.return_value = mock_ensemble_content

            ensembles = get_category_ensembles("code-analysis")

            assert len(ensembles) == 1  # Only .yaml files, not README.md
            assert ensembles[0]["name"] == "security-review"
            assert ensembles[0]["description"] == "Multi-perspective security analysis"
            assert ensembles[0]["path"] == "code-analysis/security-review.yaml"

            # Verify API was called correctly
            mock_get.assert_called_once_with(
                "https://api.github.com/repos/mrilikecoding/llm-orchestra-library/contents/ensembles/code-analysis",
                timeout=10,
            )

    def test_get_category_ensembles_network_error(self) -> None:
        """Should handle network errors gracefully."""
        from llm_orc.cli_library.library import get_category_ensembles

        with patch("requests.get") as mock_get:
            # Mock network timeout
            mock_get.side_effect = requests.exceptions.Timeout("Request timed out")

            ensembles = get_category_ensembles("code-analysis")

            assert ensembles == []

    def test_get_category_ensembles_http_error(self) -> None:
        """Should handle HTTP errors gracefully."""
        from llm_orc.cli_library.library import get_category_ensembles

        with patch("requests.get") as mock_get:
            # Mock HTTP 404 error
            mock_response = mock_get.return_value
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
                "404 Not Found"
            )

            ensembles = get_category_ensembles("nonexistent-category")

            assert ensembles == []

    def test_get_category_ensembles_invalid_yaml(self) -> None:
        """Should handle invalid YAML content gracefully."""
        from llm_orc.cli_library.library import get_category_ensembles

        mock_response_data = [
            {
                "type": "file",
                "name": "invalid-ensemble.yaml",
                "download_url": "https://example.com/invalid-ensemble.yaml",
            }
        ]

        invalid_yaml_content = "name: test\ninvalid: yaml: content: [unclosed"

        with (
            patch("requests.get") as mock_get,
            patch("llm_orc.cli_library.library.fetch_ensemble_content") as mock_fetch,
        ):
            # Mock API response
            mock_response = mock_get.return_value
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = mock_response_data

            # Mock invalid YAML content
            mock_fetch.return_value = invalid_yaml_content

            ensembles = get_category_ensembles("code-analysis")

            # Should handle YAML error and include ensemble with fallback description
            assert len(ensembles) == 1
            assert ensembles[0]["name"] == "invalid-ensemble"
            # Should have some fallback description
            assert "description" in ensembles[0]

    def test_fetch_ensemble_content_success(self) -> None:
        """Should fetch ensemble content from GitHub successfully."""
        from llm_orc.cli_library.library import fetch_ensemble_content

        mock_content = """name: test-ensemble
description: Test ensemble for unit testing
agents:
  - name: test-agent
    model: claude-3-5-sonnet
"""

        with patch("requests.get") as mock_get:
            mock_response = mock_get.return_value
            mock_response.raise_for_status.return_value = None
            mock_response.text = mock_content

            content = fetch_ensemble_content("code-analysis/test-ensemble.yaml")

            assert content == mock_content

            # Verify correct URL was called
            expected_url = (
                "https://raw.githubusercontent.com/mrilikecoding/llm-orchestra-library/main/"
                "ensembles/code-analysis/test-ensemble.yaml"
            )
            mock_get.assert_called_once_with(expected_url, timeout=10)

    def test_fetch_ensemble_content_not_found(self) -> None:
        """Should handle ensemble not found gracefully."""
        from llm_orc.cli_library.library import fetch_ensemble_content

        with patch("requests.get") as mock_get:
            mock_response = mock_get.return_value
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
                "404 Not Found"
            )

            with pytest.raises(FileNotFoundError, match="Ensemble not found"):
                fetch_ensemble_content("nonexistent/ensemble.yaml")

    def test_fetch_ensemble_content_network_error(self) -> None:
        """Should handle network errors when fetching content."""
        from llm_orc.cli_library.library import fetch_ensemble_content

        with patch("requests.get") as mock_get:
            mock_get.side_effect = requests.exceptions.ConnectionError("Network error")

            with pytest.raises(FileNotFoundError, match="Ensemble not found"):
                fetch_ensemble_content("code-analysis/test-ensemble.yaml")


class TestLibraryEnsembleAnalysis:
    """Test ensemble analysis and metadata extraction functions."""

    def test_analyze_ensemble_metadata_basic(self) -> None:
        """Should extract basic metadata from agents list."""
        from llm_orc.cli_library.library import _analyze_ensemble_metadata

        agents: list[dict[str, Any]] = [
            {
                "name": "agent1",
                "model_profile": "fast",
                "depends_on": ["agent2"],
                "output_format": "json",
            },
            {"name": "agent2", "model_profile": "default"},
        ]

        (
            model_profiles,
            dependencies,
            output_formats,
        ) = _analyze_ensemble_metadata(agents)

        assert model_profiles == {"fast", "default"}
        assert dependencies == [("agent1", "agent2")]
        assert output_formats == {"json"}

    def test_analyze_ensemble_metadata_defaults(self) -> None:
        """Should handle agents with default values."""
        from llm_orc.cli_library.library import _analyze_ensemble_metadata

        agents: list[dict[str, Any]] = [{"name": "simple-agent"}]

        (
            model_profiles,
            dependencies,
            output_formats,
        ) = _analyze_ensemble_metadata(agents)

        assert model_profiles == {"default"}
        assert dependencies == []
        assert output_formats == set()

    def test_analyze_ensemble_metadata_multiple_dependencies(self) -> None:
        """Should handle agents with multiple dependencies."""
        from llm_orc.cli_library.library import _analyze_ensemble_metadata

        agents: list[dict[str, Any]] = [
            {
                "name": "agent1",
                "depends_on": ["agent2", "agent3"],
                "output_format": "markdown",
            },
            {
                "name": "agent4",
                "depends_on": "agent1",  # Single dependency (not list)
                "output_format": "json",
            },
        ]

        (
            model_profiles,
            dependencies,
            output_formats,
        ) = _analyze_ensemble_metadata(agents)

        expected_dependencies = [
            ("agent1", "agent2"),
            ("agent1", "agent3"),
            ("agent4", "agent1"),
        ]
        assert dependencies == expected_dependencies
        assert output_formats == {"markdown", "json"}

    def test_display_agent_details_basic(self) -> None:
        """Should display basic agent information."""
        from llm_orc.cli_library.library import _display_agent_details

        agents = [
            {"name": "test-agent", "model_profile": "fast"},
            {"name": "another-agent"},
        ]

        with patch("click.echo") as mock_echo:
            _display_agent_details(agents)

            # Verify header and agent details were displayed
            mock_echo.assert_any_call("👤 Agent Details:")
            mock_echo.assert_any_call("  • test-agent (fast)")
            mock_echo.assert_any_call("  • another-agent (default)")

    def test_display_agent_details_with_dependencies(self) -> None:
        """Should display agent dependencies."""
        from llm_orc.cli_library.library import _display_agent_details

        agents = [
            {
                "name": "dependent-agent",
                "depends_on": ["agent1", "agent2"],
                "output_format": "json",
            }
        ]

        with patch("click.echo") as mock_echo:
            _display_agent_details(agents)

            mock_echo.assert_any_call("    ↳ depends on: agent1, agent2")
            mock_echo.assert_any_call("    ↳ output format: json")

    def test_display_agent_details_single_dependency(self) -> None:
        """Should handle single dependency (not list)."""
        from llm_orc.cli_library.library import _display_agent_details

        agents = [{"name": "agent", "depends_on": "parent-agent"}]

        with patch("click.echo") as mock_echo:
            _display_agent_details(agents)

            mock_echo.assert_any_call("    ↳ depends on: parent-agent")

    def test_display_execution_flow_no_dependencies(self) -> None:
        """Should handle agents with no dependencies."""
        from llm_orc.cli_library.library import _display_execution_flow

        agents = [{"name": "agent1"}, {"name": "agent2"}]
        dependencies: list[tuple[str, str]] = []

        with patch("click.echo") as mock_echo:
            _display_execution_flow(agents, dependencies)

            # Should not display anything for empty dependencies
            mock_echo.assert_not_called()

    def test_display_execution_flow_with_dependencies(self) -> None:
        """Should display execution flow with dependencies."""
        from llm_orc.cli_library.library import _display_execution_flow

        agents: list[dict[str, Any]] = [
            {"name": "independent1"},
            {"name": "independent2"},
            {"name": "dependent1", "depends_on": ["independent1"]},
            {"name": "dependent2", "depends_on": ["independent2"]},
        ]
        dependencies: list[tuple[str, str]] = [
            ("dependent1", "independent1"),
            ("dependent2", "independent2"),
        ]

        with patch("click.echo") as mock_echo:
            _display_execution_flow(agents, dependencies)

            mock_echo.assert_any_call()
            mock_echo.assert_any_call("🔄 Execution Flow:")
            mock_echo.assert_any_call("  1. Parallel: independent1, independent2")
            mock_echo.assert_any_call("  2. Sequential: dependent1, dependent2")


class TestLibraryShowEnsembleInfo:
    """Test show_ensemble_info command functionality."""

    def test_show_ensemble_info_success(self) -> None:
        """Should display complete ensemble information."""
        from llm_orc.cli_library.library import show_ensemble_info

        mock_content = """
name: test-ensemble
description: A test ensemble for analysis
agents:
  - name: analyzer
    model_profile: fast
    depends_on: ["extractor"]
    output_format: json
  - name: extractor
    model_profile: default
"""

        with (
            patch("llm_orc.cli_library.library.fetch_ensemble_content") as mock_fetch,
            patch("click.echo") as mock_echo,
        ):
            mock_fetch.return_value = mock_content

            show_ensemble_info("test/ensemble")

            # Verify ensemble info display
            mock_echo.assert_any_call("📋 Ensemble: test-ensemble")
            mock_echo.assert_any_call("📝 Description: A test ensemble for analysis")
            mock_echo.assert_any_call("👥 Agents: 2")
            mock_echo.assert_any_call("🤖 Model Profiles:")
            mock_echo.assert_any_call("  • default")
            mock_echo.assert_any_call("  • fast")

    def test_show_ensemble_info_yaml_error(self) -> None:
        """Should handle YAML parsing errors."""
        from llm_orc.cli_library.library import show_ensemble_info

        invalid_content = "name: test\ninvalid: yaml: content: [unclosed"

        with (
            patch("llm_orc.cli_library.library.fetch_ensemble_content") as mock_fetch,
            patch("click.echo") as mock_echo,
        ):
            mock_fetch.return_value = invalid_content

            with pytest.raises(Exception, match="Invalid YAML"):
                show_ensemble_info("test/ensemble")

            # Check that an error message was displayed (exact text may vary)
            error_calls = [
                call
                for call in mock_echo.call_args_list
                if call.kwargs.get("err") is True
            ]
            assert len(error_calls) > 0
            assert "Invalid YAML" in str(error_calls[0])

    def test_show_ensemble_info_file_not_found(self) -> None:
        """Should handle ensemble not found errors."""
        from llm_orc.cli_library.library import show_ensemble_info

        with (
            patch("llm_orc.cli_library.library.fetch_ensemble_content") as mock_fetch,
            patch("click.echo") as mock_echo,
        ):
            mock_fetch.side_effect = FileNotFoundError(
                "Ensemble not found: test/ensemble"
            )

            with pytest.raises(Exception, match="Ensemble not found"):
                show_ensemble_info("test/ensemble")

            mock_echo.assert_any_call(
                "Error: Ensemble not found: test/ensemble", err=True
            )


class TestLibraryEdgeCases:
    """Test edge cases and string manipulation."""

    def test_fetch_ensemble_content_adds_yaml_extension(self) -> None:
        """Should add .yaml extension when not present."""
        from llm_orc.cli_library.library import fetch_ensemble_content

        with patch("requests.get") as mock_get:
            mock_response = mock_get.return_value
            mock_response.raise_for_status.return_value = None
            mock_response.text = "ensemble content"

            content = fetch_ensemble_content("code-analysis/security-review")

            assert content == "ensemble content"
            # Verify .yaml was appended to URL
            expected_url = (
                "https://raw.githubusercontent.com/mrilikecoding/llm-orchestra-library/main/"
                "ensembles/code-analysis/security-review.yaml"
            )
            mock_get.assert_called_once_with(expected_url, timeout=10)

    def test_get_template_content_adds_yaml_extension(self) -> None:
        """Should add .yaml extension to template name when not present."""
        from llm_orc.cli_library.library import get_template_content

        with patch("requests.get") as mock_get:
            mock_response = mock_get.return_value
            mock_response.raise_for_status.return_value = None
            mock_response.text = "template content"

            content = get_template_content("local-config")

            assert content == "template content"
            # Verify .yaml was appended to URL
            expected_url = (
                "https://raw.githubusercontent.com/mrilikecoding/llm-orchestra-library/main/"
                "templates/local-config.yaml"
            )
            mock_get.assert_called_once_with(expected_url, timeout=10)

    def test_copy_ensemble_user_declines_overwrite(self) -> None:
        """Should handle user declining overwrite confirmation."""
        from llm_orc.cli_library.library import copy_ensemble

        ensemble_content = "name: existing-ensemble"

        with (
            patch("llm_orc.cli_library.library.fetch_ensemble_content") as mock_fetch,
            patch("llm_orc.cli_library.library.ensemble_exists") as mock_exists,
            patch("click.confirm") as mock_confirm,
            patch("click.echo") as mock_echo,
        ):
            mock_fetch.return_value = ensemble_content
            mock_exists.return_value = True
            mock_confirm.return_value = False  # User declines

            copy_ensemble("test/ensemble", is_global=False)

            mock_echo.assert_called_with("Copy cancelled.")

    def test_list_categories_display(self) -> None:
        """Should display categories with descriptions properly formatted."""
        from llm_orc.cli_library.library import list_categories

        with (
            patch(
                "llm_orc.cli_library.library.get_library_categories_with_descriptions"
            ) as mock_get_cats,
            patch("click.echo") as mock_echo,
        ):
            mock_get_cats.return_value = [
                ("code-analysis", "Code review and security analysis"),
                ("idea-exploration", "Concept mapping and perspective taking"),
            ]

            list_categories()

            mock_echo.assert_any_call("Available ensemble categories:")
            mock_echo.assert_any_call()
            # Check formatting is reasonable (exact spacing may vary)
            calls = [str(call) for call in mock_echo.call_args_list]
            assert any(
                "code-analysis" in call and "Code review" in call for call in calls
            )
            assert any(
                "idea-exploration" in call and "Concept mapping" in call
                for call in calls
            )
            mock_echo.assert_any_call()


class TestLibraryConfigurationIntegration:
    """Test integration with configuration manager."""

    def test_ensure_global_ensembles_dir_creates_directory(self) -> None:
        """Should create global ensembles directory if it doesn't exist."""
        from llm_orc.cli_library.library import ensure_global_ensembles_dir

        with (
            patch("llm_orc.cli_library.library.ConfigurationManager") as mock_config,
            patch("pathlib.Path.mkdir") as mock_mkdir,
        ):
            mock_config_instance = mock_config.return_value
            mock_config_instance.global_config_dir = "/home/test/.llm-orc"

            result_path = ensure_global_ensembles_dir()

            expected_path = "/home/test/.llm-orc/ensembles"
            assert result_path == expected_path

            # Verify directory creation was attempted
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_ensure_global_ensembles_dir_existing_directory(self) -> None:
        """Should return path if global ensembles directory already exists."""
        from llm_orc.cli_library.library import ensure_global_ensembles_dir

        with (
            patch("llm_orc.cli_library.library.ConfigurationManager") as mock_config,
            patch("pathlib.Path.mkdir") as mock_mkdir,
        ):
            mock_config_instance = mock_config.return_value
            mock_config_instance.global_config_dir = "/home/test/.llm-orc"

            result_path = ensure_global_ensembles_dir()

            expected_path = "/home/test/.llm-orc/ensembles"
            assert result_path == expected_path

            # Should still call mkdir with exist_ok=True
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_ensure_local_ensembles_dir_creates_directory(self) -> None:
        """Should create local ensembles directory if it doesn't exist."""
        from llm_orc.cli_library.library import ensure_local_ensembles_dir

        with patch("pathlib.Path.mkdir") as mock_mkdir:
            result_path = ensure_local_ensembles_dir()

            expected_path = ".llm-orc/ensembles"
            assert result_path == expected_path

            # Verify directory creation
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_ensemble_exists_local_true(self) -> None:
        """Should detect when ensemble exists in local directory."""
        from llm_orc.cli_library.library import ensemble_exists

        with (
            patch("pathlib.Path.cwd") as mock_cwd,
            patch("pathlib.Path.exists", return_value=True),
        ):
            mock_cwd.return_value = Path("/project")

            exists = ensemble_exists("test-ensemble", is_global=False)

            assert exists is True

    def test_ensemble_exists_global_true(self) -> None:
        """Should detect when ensemble exists in global directory."""
        from llm_orc.cli_library.library import ensemble_exists

        with (
            patch(
                "llm_orc.cli_library.library.ensure_global_ensembles_dir"
            ) as mock_ensure_dir,
            patch("pathlib.Path.exists", return_value=True),
        ):
            mock_ensure_dir.return_value = "/home/test/.llm-orc/ensembles"

            exists = ensemble_exists("test-ensemble", is_global=True)

            assert exists is True

    def test_ensemble_exists_false(self) -> None:
        """Should return false when ensemble doesn't exist anywhere."""
        from llm_orc.cli_library.library import ensemble_exists

        with (
            patch(
                "llm_orc.cli_library.library.ensure_local_ensembles_dir"
            ) as mock_ensure_dir,
            patch("pathlib.Path.exists", return_value=False),
        ):
            mock_ensure_dir.return_value = "/project/.llm-orc/ensembles"

            exists = ensemble_exists("nonexistent-ensemble", is_global=False)

            assert exists is False

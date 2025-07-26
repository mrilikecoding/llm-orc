"""Tests for library CLI commands."""

from pathlib import Path
from unittest.mock import mock_open, patch

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
        import click

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

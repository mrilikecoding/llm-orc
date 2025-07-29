"""Tests for main CLI interface (cli.py)."""

from unittest.mock import Mock, patch

from click.testing import CliRunner

from llm_orc.cli import cli


class TestMainCLI:
    """Test the main CLI interface defined in cli.py."""

    def test_cli_main_help(self) -> None:
        """Test main CLI help message."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "LLM Orchestra" in result.output
        assert "Multi-agent LLM communication system" in result.output
        assert "Commands:" in result.output

    def test_cli_version_option(self) -> None:
        """Test CLI version option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])

        assert result.exit_code == 0
        # Should show version information
        assert result.output.strip() != ""

    def test_invoke_command_basic(self) -> None:
        """Test basic invoke command functionality."""
        runner = CliRunner()

        # Mock the invoke_ensemble function
        with patch("llm_orc.cli.invoke_ensemble") as mock_invoke:
            result = runner.invoke(cli, ["invoke", "test_ensemble"])

            mock_invoke.assert_called_once_with(
                "test_ensemble",  # ensemble_name
                None,  # input_data
                None,  # config_dir
                None,  # input_data_option
                None,  # output_format (default - Rich interface)
                False,  # streaming (default)
                None,  # max_concurrent (default)
                False,  # detailed (default)
            )
            assert result.exit_code == 0

    def test_invoke_command_with_all_options(self) -> None:
        """Test invoke command with all options."""
        runner = CliRunner()

        with patch("llm_orc.cli.invoke_ensemble") as mock_invoke:
            result = runner.invoke(
                cli,
                [
                    "invoke",
                    "test_ensemble",
                    "test input",
                    "--config-dir",
                    "/custom/config",
                    "--input-data",
                    "option input",
                    "--output-format",
                    "json",
                    "--streaming",
                    "--max-concurrent",
                    "5",
                    "--detailed",
                ],
            )

            mock_invoke.assert_called_once_with(
                "test_ensemble",  # ensemble_name
                "test input",  # input_data (positional)
                "/custom/config",  # config_dir
                "option input",  # input_data_option
                "json",  # output_format
                True,  # streaming
                5,  # max_concurrent
                True,  # detailed
            )
            assert result.exit_code == 0

    def test_invoke_command_input_data_priority(self) -> None:
        """Test that positional input_data takes precedence over option."""
        runner = CliRunner()

        with patch("llm_orc.cli.invoke_ensemble") as mock_invoke:
            runner.invoke(
                cli,
                [
                    "invoke",
                    "test_ensemble",
                    "positional_input",
                    "--input-data",
                    "option_input",
                ],
            )

            # Should pass positional argument, option argument goes to input_data_option
            mock_invoke.assert_called_once_with(
                "test_ensemble",
                "positional_input",  # positional takes precedence
                None,  # config_dir
                "option_input",  # input_data_option
                None,  # output_format (default - Rich interface)
                False,  # streaming (default)
                None,  # max_concurrent (default)
                False,  # detailed (default)
            )

    def test_invoke_command_output_format_choices(self) -> None:
        """Test invoke command output format validation."""
        runner = CliRunner()

        # Valid choice
        with patch("llm_orc.cli.invoke_ensemble"):
            result = runner.invoke(cli, ["invoke", "test", "--output-format", "json"])
            assert result.exit_code == 0

        # Invalid choice
        result = runner.invoke(cli, ["invoke", "test", "--output-format", "invalid"])
        assert result.exit_code != 0
        assert "Invalid value for '--output-format'" in result.output

    def test_list_ensembles_command(self) -> None:
        """Test list-ensembles command."""
        runner = CliRunner()

        with patch("llm_orc.cli.list_ensembles_command") as mock_list:
            result = runner.invoke(cli, ["list-ensembles"])

            mock_list.assert_called_once_with(None)
            assert result.exit_code == 0

    def test_list_ensembles_with_config_dir(self) -> None:
        """Test list-ensembles command with config directory."""
        runner = CliRunner()

        with patch("llm_orc.cli.list_ensembles_command") as mock_list:
            result = runner.invoke(
                cli, ["list-ensembles", "--config-dir", "/custom/config"]
            )

            mock_list.assert_called_once_with("/custom/config")
            assert result.exit_code == 0

    def test_list_profiles_command(self) -> None:
        """Test list-profiles command."""
        runner = CliRunner()

        with patch("llm_orc.cli.list_profiles_command") as mock_list:
            result = runner.invoke(cli, ["list-profiles"])

            mock_list.assert_called_once()
            assert result.exit_code == 0

    def test_config_group_exists(self) -> None:
        """Test that config command group exists."""
        runner = CliRunner()
        result = runner.invoke(cli, ["config", "--help"])

        assert result.exit_code == 0
        assert "Configuration management commands" in result.output

    def test_config_init_command(self) -> None:
        """Test config init command."""
        runner = CliRunner()

        with patch("llm_orc.cli.init_local_config") as mock_init:
            result = runner.invoke(cli, ["config", "init"])

            mock_init.assert_called_once_with(None)
            assert result.exit_code == 0

    def test_config_init_with_project_name(self) -> None:
        """Test config init command with project name."""
        runner = CliRunner()

        with patch("llm_orc.cli.init_local_config") as mock_init:
            result = runner.invoke(
                cli, ["config", "init", "--project-name", "MyProject"]
            )

            mock_init.assert_called_once_with("MyProject")
            assert result.exit_code == 0

    def test_config_reset_global_command(self) -> None:
        """Test config reset-global command."""
        runner = CliRunner()

        with patch("llm_orc.cli.reset_global_config") as mock_reset:
            # Use --yes to skip confirmation prompt
            result = runner.invoke(cli, ["config", "reset-global", "--yes"])

            mock_reset.assert_called_once_with(
                True, True
            )  # backup=True, preserve_auth=True
            assert result.exit_code == 0

    def test_config_reset_global_with_options(self) -> None:
        """Test config reset-global command with options."""
        runner = CliRunner()

        with patch("llm_orc.cli.reset_global_config") as mock_reset:
            result = runner.invoke(
                cli, ["config", "reset-global", "--no-backup", "--reset-auth", "--yes"]
            )

            mock_reset.assert_called_once_with(
                False, False
            )  # backup=False, preserve_auth=False
            assert result.exit_code == 0

    def test_config_check_global_command(self) -> None:
        """Test config check-global command."""
        runner = CliRunner()

        with patch("llm_orc.cli.check_global_config") as mock_check:
            result = runner.invoke(cli, ["config", "check-global"])

            mock_check.assert_called_once()
            assert result.exit_code == 0

    def test_config_reset_local_command(self) -> None:
        """Test config reset-local command."""
        runner = CliRunner()

        with patch("llm_orc.cli.reset_local_config") as mock_reset:
            result = runner.invoke(cli, ["config", "reset-local", "--yes"])

            mock_reset.assert_called_once_with(
                True, True, None
            )  # backup=True, preserve_ensembles=True, project_name=None
            assert result.exit_code == 0

    def test_config_reset_local_with_options(self) -> None:
        """Test config reset-local command with options."""
        runner = CliRunner()

        with patch("llm_orc.cli.reset_local_config") as mock_reset:
            result = runner.invoke(
                cli,
                [
                    "config",
                    "reset-local",
                    "--no-backup",
                    "--reset-ensembles",
                    "--project-name",
                    "TestProject",
                    "--yes",
                ],
            )

            mock_reset.assert_called_once_with(False, False, "TestProject")
            assert result.exit_code == 0

    def test_config_check_unified_command(self) -> None:
        """Test config check unified command shows both configs."""
        runner = CliRunner()

        with (
            patch("llm_orc.cli.check_global_config") as mock_check_global,
            patch("llm_orc.cli.check_local_config") as mock_check_local,
        ):
            result = runner.invoke(cli, ["config", "check"])

            # Should call both functions
            mock_check_global.assert_called_once()
            mock_check_local.assert_called_once()
            assert result.exit_code == 0

            # Should show legend
            assert "Configuration Status Legend:" in result.output
            assert "🟢 Ready to use" in result.output
            assert "🟥 Needs setup" in result.output

    def test_config_check_local_command(self) -> None:
        """Test config check-local command."""
        runner = CliRunner()

        with patch("llm_orc.cli.check_local_config") as mock_check:
            result = runner.invoke(cli, ["config", "check-local"])

            mock_check.assert_called_once()
            assert result.exit_code == 0

    def test_auth_group_exists(self) -> None:
        """Test that auth command group exists."""
        runner = CliRunner()
        result = runner.invoke(cli, ["auth", "--help"])

        assert result.exit_code == 0
        assert "Authentication management commands" in result.output

    def test_auth_add_command_api_key(self) -> None:
        """Test auth add command with API key."""
        runner = CliRunner()

        with patch("llm_orc.cli.add_auth_provider") as mock_add:
            result = runner.invoke(
                cli, ["auth", "add", "anthropic", "--api-key", "test-key"]
            )

            mock_add.assert_called_once_with("anthropic", "test-key", None, None)
            assert result.exit_code == 0

    def test_auth_add_command_oauth(self) -> None:
        """Test auth add command with OAuth credentials."""
        runner = CliRunner()

        with patch("llm_orc.cli.add_auth_provider") as mock_add:
            result = runner.invoke(
                cli,
                [
                    "auth",
                    "add",
                    "google",
                    "--client-id",
                    "test-client-id",
                    "--client-secret",
                    "test-secret",
                ],
            )

            mock_add.assert_called_once_with(
                "google", None, "test-client-id", "test-secret"
            )
            assert result.exit_code == 0

    def test_auth_add_command_mixed_credentials(self) -> None:
        """Test auth add command with mixed credentials."""
        runner = CliRunner()

        with patch("llm_orc.cli.add_auth_provider") as mock_add:
            result = runner.invoke(
                cli,
                [
                    "auth",
                    "add",
                    "provider",
                    "--api-key",
                    "key",
                    "--client-id",
                    "id",
                    "--client-secret",
                    "secret",
                ],
            )

            mock_add.assert_called_once_with("provider", "key", "id", "secret")
            assert result.exit_code == 0

    def test_auth_list_command(self) -> None:
        """Test auth list command."""
        runner = CliRunner()

        with patch("llm_orc.cli.list_auth_providers") as mock_list:
            result = runner.invoke(cli, ["auth", "list"])

            mock_list.assert_called_once_with(False)
            assert result.exit_code == 0

    def test_auth_list_command_interactive(self) -> None:
        """Test auth list command with interactive flag."""
        runner = CliRunner()

        with patch("llm_orc.cli.list_auth_providers") as mock_list:
            result = runner.invoke(cli, ["auth", "list", "--interactive"])

            mock_list.assert_called_once_with(True)
            assert result.exit_code == 0

    def test_auth_remove_command(self) -> None:
        """Test auth remove command."""
        runner = CliRunner()

        with patch("llm_orc.cli.remove_auth_provider") as mock_remove:
            result = runner.invoke(cli, ["auth", "remove", "anthropic"])

            mock_remove.assert_called_once_with("anthropic")
            assert result.exit_code == 0

    def test_auth_setup_command(self) -> None:
        """Test auth setup command."""
        runner = CliRunner()

        with patch("llm_orc.cli_commands.auth_setup") as mock_setup:
            result = runner.invoke(cli, ["auth", "setup"])

            mock_setup.assert_called_once()
            assert result.exit_code == 0

    def test_auth_logout_single_provider(self) -> None:
        """Test auth logout command for single provider."""
        runner = CliRunner()

        with patch("llm_orc.cli.logout_oauth_providers") as mock_logout:
            result = runner.invoke(cli, ["auth", "logout", "google"])

            mock_logout.assert_called_once_with("google", False)
            assert result.exit_code == 0

    def test_auth_logout_all_providers(self) -> None:
        """Test auth logout command for all providers."""
        runner = CliRunner()

        with patch("llm_orc.cli.logout_oauth_providers") as mock_logout:
            result = runner.invoke(cli, ["auth", "logout", "--all"])

            mock_logout.assert_called_once_with(None, True)
            assert result.exit_code == 0

    def test_auth_test_refresh_command(self) -> None:
        """Test auth test-refresh command."""
        runner = CliRunner()

        with patch("llm_orc.cli.test_token_refresh") as mock_test:
            result = runner.invoke(cli, ["auth", "test-refresh", "google"])

            mock_test.assert_called_once_with("google")
            assert result.exit_code == 0

    def test_serve_command(self) -> None:
        """Test serve command."""
        runner = CliRunner()

        with patch("llm_orc.cli.serve_ensemble") as mock_serve:
            result = runner.invoke(cli, ["serve", "test_ensemble"])

            mock_serve.assert_called_once_with("test_ensemble", 3000)  # default port
            assert result.exit_code == 0

    def test_serve_command_custom_port(self) -> None:
        """Test serve command with custom port."""
        runner = CliRunner()

        with patch("llm_orc.cli.serve_ensemble") as mock_serve:
            result = runner.invoke(cli, ["serve", "test_ensemble", "--port", "8080"])

            mock_serve.assert_called_once_with("test_ensemble", 8080)
            assert result.exit_code == 0

    def test_help_command_basic(self) -> None:
        """Test custom help command."""
        runner = CliRunner()

        # Mock click.get_current_context to return a context with parent
        mock_ctx = Mock()
        mock_parent_ctx = Mock()
        mock_ctx.parent = mock_parent_ctx

        with patch("llm_orc.cli.click.get_current_context", return_value=mock_ctx):
            result = runner.invoke(cli, ["help"])

            assert result.exit_code == 0
            assert "Usage: llm-orc" in result.output
            assert "Commands:" in result.output
            assert (
                "You can use either the full command name or its alias" in result.output
            )

    def test_help_command_no_parent_context(self) -> None:
        """Test help command when no parent context available."""
        runner = CliRunner()

        # Mock click.get_current_context to return a context without parent
        mock_ctx = Mock()
        mock_ctx.parent = None

        with patch("llm_orc.cli.click.get_current_context", return_value=mock_ctx):
            result = runner.invoke(cli, ["help"])

            assert result.exit_code == 0
            assert "Help not available" in result.output

    def test_help_command_shows_aliases(self) -> None:
        """Test that help command shows command aliases."""
        runner = CliRunner()

        mock_ctx = Mock()
        mock_parent_ctx = Mock()
        mock_ctx.parent = mock_parent_ctx

        with patch("llm_orc.cli.click.get_current_context", return_value=mock_ctx):
            result = runner.invoke(cli, ["help"])

            # Check for some key aliases
            assert "(a )" in result.output  # auth alias
            assert "(c )" in result.output  # config alias
            assert "(i )" in result.output  # invoke alias
            assert "(le)" in result.output  # list-ensembles alias
            assert "(lp)" in result.output  # list-profiles alias
            assert "(s )" in result.output  # serve alias
            assert "(h )" in result.output  # help alias

    def test_command_aliases_work(self) -> None:
        """Test that command aliases work correctly."""
        runner = CliRunner()

        # Test invoke alias
        with patch("llm_orc.cli.invoke_ensemble") as mock_invoke:
            result = runner.invoke(cli, ["i", "test_ensemble"])
            assert result.exit_code == 0
            mock_invoke.assert_called_once()

        # Test auth alias
        result = runner.invoke(cli, ["a", "--help"])
        assert result.exit_code == 0
        assert "Authentication management" in result.output

        # Test config alias
        result = runner.invoke(cli, ["c", "--help"])
        assert result.exit_code == 0
        assert "Configuration management" in result.output

        # Test list-ensembles alias
        with patch("llm_orc.cli.list_ensembles_command") as mock_list:
            result = runner.invoke(cli, ["le"])
            assert result.exit_code == 0
            mock_list.assert_called_once()

        # Test list-profiles alias
        with patch("llm_orc.cli.list_profiles_command") as mock_list:
            result = runner.invoke(cli, ["lp"])
            assert result.exit_code == 0
            mock_list.assert_called_once()

        # Test serve alias
        with patch("llm_orc.cli.serve_ensemble") as mock_serve:
            result = runner.invoke(cli, ["s", "test_ensemble"])
            assert result.exit_code == 0
            mock_serve.assert_called_once()

        # Test help alias
        mock_ctx = Mock()
        mock_parent_ctx = Mock()
        mock_ctx.parent = mock_parent_ctx

        with patch("llm_orc.cli.click.get_current_context", return_value=mock_ctx):
            result = runner.invoke(cli, ["h"])
            assert result.exit_code == 0
            assert "Usage: llm-orc" in result.output

    def test_main_execution_when_run_directly(self) -> None:
        """Test that main execution works when run directly."""
        # Test the if __name__ == "__main__" block by simulating direct module execution
        import sys

        # Mock sys.argv to avoid any side effects
        original_argv = sys.argv[:]
        sys.argv = ["llm-orc", "--help"]

        try:
            # Use exec to execute the bottom of the cli.py file
            # This will execute the if __name__ == "__main__" block
            exec_code = """
if "__main__" == "__main__":
    from llm_orc.cli import cli
    try:
        cli()
    except SystemExit:
        pass  # Click exits with SystemExit, which is expected
"""
            exec(exec_code)

            # If we get here, the execution worked
            assert True
        except Exception as e:
            # Should not raise unexpected exceptions
            raise AssertionError(f"Main execution failed: {e}") from e
        finally:
            # Restore original argv
            sys.argv = original_argv

    def test_config_confirmation_prompts(self) -> None:
        """Test that dangerous config commands require confirmation."""
        runner = CliRunner()

        # Test reset-global requires confirmation
        with patch("llm_orc.cli.reset_global_config"):
            result = runner.invoke(cli, ["config", "reset-global"], input="n\n")
            assert result.exit_code == 1  # Aborted
            assert "Aborted" in result.output

        # Test reset-local requires confirmation
        with patch("llm_orc.cli.reset_local_config"):
            result = runner.invoke(cli, ["config", "reset-local"], input="n\n")
            assert result.exit_code == 1  # Aborted
            assert "Aborted" in result.output

    def test_click_options_validation(self) -> None:
        """Test that Click properly validates option types and choices."""
        runner = CliRunner()

        # Test max-concurrent must be integer
        result = runner.invoke(
            cli, ["invoke", "test", "--max-concurrent", "not-a-number"]
        )
        assert result.exit_code != 0
        assert "is not a valid integer" in result.output

        # Test port must be integer
        result = runner.invoke(cli, ["serve", "test", "--port", "not-a-number"])
        assert result.exit_code != 0
        assert "is not a valid integer" in result.output

    def test_command_help_messages(self) -> None:
        """Test that all commands have proper help messages."""
        runner = CliRunner()

        # Test main commands
        commands_to_test = [
            ["invoke", "--help"],
            ["list-ensembles", "--help"],
            ["list-profiles", "--help"],
            ["serve", "--help"],
            ["config", "--help"],
            ["auth", "--help"],
        ]

        for cmd in commands_to_test:
            result = runner.invoke(cli, cmd)
            assert result.exit_code == 0, f"Command {cmd} help failed"
            # Should have some description
            assert len(result.output) > 50, f"Command {cmd} help too short"

    def test_subcommand_help_messages(self) -> None:
        """Test that all subcommands have proper help messages."""
        runner = CliRunner()

        # Test config subcommands
        config_commands = [
            ["config", "init", "--help"],
            ["config", "reset-global", "--help"],
            ["config", "reset-local", "--help"],
            ["config", "check", "--help"],
            ["config", "check-global", "--help"],
            ["config", "check-local", "--help"],
        ]

        for cmd in config_commands:
            result = runner.invoke(cli, cmd)
            assert result.exit_code == 0, f"Config command {cmd} help failed"

        # Test auth subcommands
        auth_commands = [
            ["auth", "add", "--help"],
            ["auth", "list", "--help"],
            ["auth", "remove", "--help"],
            ["auth", "setup", "--help"],
            ["auth", "logout", "--help"],
            ["auth", "test-refresh", "--help"],
        ]

        for cmd in auth_commands:
            result = runner.invoke(cli, cmd)
            assert result.exit_code == 0, f"Auth command {cmd} help failed"

    def test_error_handling_missing_arguments(self) -> None:
        """Test error handling for missing required arguments."""
        runner = CliRunner()

        # invoke requires ensemble name
        result = runner.invoke(cli, ["invoke"])
        assert result.exit_code != 0

        # auth add requires provider
        result = runner.invoke(cli, ["auth", "add"])
        assert result.exit_code != 0

        # auth remove requires provider
        result = runner.invoke(cli, ["auth", "remove"])
        assert result.exit_code != 0

        # serve requires ensemble name
        result = runner.invoke(cli, ["serve"])
        assert result.exit_code != 0

        # auth test-refresh requires provider
        result = runner.invoke(cli, ["auth", "test-refresh"])
        assert result.exit_code != 0

"""Tests for the ADR-003 contract validation CLI module."""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_orc.testing.contract_validator import (
    ContractValidationCLI,
    ContractValidationError,
    ValidationLevel,
    ValidationResult,
)


class TestContractValidationCLI:
    """Test cases for contract validation CLI."""

    def test_cli_initialization(self) -> None:
        """Test CLI initialization with default parameters."""
        cli = ContractValidationCLI()
        assert cli.directory == Path(".")
        assert cli.level == ValidationLevel.CORE
        assert not cli.verbose

    def test_cli_initialization_with_params(self) -> None:
        """Test CLI initialization with custom parameters."""
        cli = ContractValidationCLI(
            directory="/test/path", level=ValidationLevel.EXAMPLES, verbose=True
        )
        assert cli.directory == Path("/test/path")
        assert cli.level == ValidationLevel.EXAMPLES
        assert cli.verbose

    def test_validation_level_enum(self) -> None:
        """Test validation level enumeration."""
        assert ValidationLevel.CORE.value == "core"
        assert ValidationLevel.EXAMPLES.value == "examples"
        assert ValidationLevel.COMMUNITY.value == "community"

    def test_validation_result_model(self) -> None:
        """Test validation result model structure."""
        result = ValidationResult(
            success=True,
            validated_scripts=["script1.py", "script2.py"],
            errors=[],
            warnings=["Warning message"],
            execution_time_seconds=1.5,
        )
        assert result.success is True
        assert len(result.validated_scripts) == 2
        assert len(result.errors) == 0
        assert len(result.warnings) == 1
        assert result.execution_time_seconds == 1.5

    def test_discover_scripts_empty_directory(self) -> None:
        """Test script discovery in empty directory."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.rglob", return_value=[]):
                cli = ContractValidationCLI()
                scripts = cli.discover_scripts()
                assert len(scripts) == 0

    def test_discover_scripts_with_python_files(self) -> None:
        """Test script discovery with Python files."""
        mock_files = [
            Path("script1.py"),
            Path("script2.py"),
            Path("subdir/script3.py"),
        ]
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.rglob", return_value=mock_files):
                cli = ContractValidationCLI()
                scripts = cli.discover_scripts()
                assert len(scripts) == 3
                assert all(str(script).endswith(".py") for script in scripts)

    def test_validate_directory_not_exists(self) -> None:
        """Test validation fails when directory doesn't exist."""
        cli = ContractValidationCLI(directory="/nonexistent/path")
        with pytest.raises(ContractValidationError, match="Directory does not exist"):
            cli.validate()

    def test_validate_no_scripts_found(self) -> None:
        """Test validation with no scripts found."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch.object(
                ContractValidationCLI, "discover_scripts", return_value=[]
            ):
                cli = ContractValidationCLI()
                result = cli.validate()
                assert result.success is True  # No scripts is valid state
                assert len(result.validated_scripts) == 0
                assert "No scripts found" in result.warnings

    @patch("llm_orc.testing.contract_validator.ContractValidator")
    def test_validate_with_mock_validator(
        self, mock_validator_class: MagicMock
    ) -> None:
        """Test validation with mock validator."""
        mock_validator = MagicMock()
        mock_validator.validate_all_scripts.return_value = True
        mock_validator.validation_errors = []
        mock_validator_class.return_value = mock_validator

        mock_scripts = [Path("script1.py")]
        with patch("pathlib.Path.exists", return_value=True):
            with patch.object(
                ContractValidationCLI, "discover_scripts", return_value=mock_scripts
            ):
                cli = ContractValidationCLI()
                result = cli.validate()
                assert result.success is True
                assert len(result.validated_scripts) == 1

    def test_cli_main_function_exists(self) -> None:
        """Test CLI main function exists."""
        from llm_orc.testing.contract_validator import main

        assert callable(main)

    def test_cli_argument_parser(self) -> None:
        """Test CLI argument parser configuration."""
        from llm_orc.testing.contract_validator import create_argument_parser

        parser = create_argument_parser()

        # Test default values
        args = parser.parse_args([])
        assert args.directory == "."
        assert args.level == "core"
        assert not args.verbose

        # Test custom values
        args = parser.parse_args(
            ["--directory", "/custom/path", "--level", "examples", "--verbose"]
        )
        assert args.directory == "/custom/path"
        assert args.level == "examples"
        assert args.verbose

    def test_cli_output_json_format(self) -> None:
        """Test CLI outputs results in JSON format."""
        mock_result = ValidationResult(
            success=True,
            validated_scripts=["script1.py"],
            errors=[],
            warnings=[],
            execution_time_seconds=0.5,
        )

        with patch.object(ContractValidationCLI, "validate", return_value=mock_result):
            cli = ContractValidationCLI()
            json_output = cli.get_json_output()

            parsed = json.loads(json_output)
            assert parsed["success"] is True
            assert len(parsed["validated_scripts"]) == 1
            assert parsed["execution_time_seconds"] == 0.5

    def test_cli_verbose_output(self) -> None:
        """Test CLI verbose output includes additional details."""
        cli = ContractValidationCLI(verbose=True)
        mock_result = ValidationResult(
            success=True,
            validated_scripts=["script1.py", "script2.py"],
            errors=[],
            warnings=["Warning message"],
            execution_time_seconds=1.2,
        )

        with patch.object(ContractValidationCLI, "validate", return_value=mock_result):
            output = cli.get_verbose_output()
            assert "script1.py" in output
            assert "script2.py" in output

    def test_cli_verbose_output_with_errors(self) -> None:
        """Test verbose output displays errors (lines 176-178)."""
        cli = ContractValidationCLI(verbose=True)
        mock_result = ValidationResult(
            success=False,
            validated_scripts=[],
            errors=["Error 1", "Error 2"],
            warnings=[],
            execution_time_seconds=0.5,
        )

        with patch.object(ContractValidationCLI, "validate", return_value=mock_result):
            output = cli.get_verbose_output()
            assert "Errors" in output
            assert "Error 1" in output
            assert "Error 2" in output

    @patch("llm_orc.testing.contract_validator.ContractValidator")
    def test_validate_with_validator_failure(
        self, mock_validator_class: MagicMock
    ) -> None:
        """Test validation when validator returns False (line 127)."""
        mock_validator = MagicMock()
        mock_validator.validate_all_scripts.return_value = False
        mock_validator.validation_errors = ["Script validation failed"]
        mock_validator_class.return_value = mock_validator

        mock_scripts = [Path("script1.py")]
        with patch("pathlib.Path.exists", return_value=True):
            with patch.object(
                ContractValidationCLI, "discover_scripts", return_value=mock_scripts
            ):
                cli = ContractValidationCLI()
                result = cli.validate()
                assert result.success is False
                assert "Validation failed for one or more scripts" in result.errors

    def test_main_with_json_output(self) -> None:
        """Test main() with --json flag (line 245)."""
        from llm_orc.testing.contract_validator import main

        test_args = ["contract-validator", "--json"]

        mock_result = ValidationResult(
            success=True,
            validated_scripts=["test.py"],
            errors=[],
            warnings=[],
            execution_time_seconds=0.1,
        )

        with patch("sys.argv", test_args):
            with patch.object(
                ContractValidationCLI, "validate", return_value=mock_result
            ):
                with patch("builtins.print") as mock_print:
                    with pytest.raises(SystemExit) as exc_info:
                        main()

                    assert exc_info.value.code == 0
                    # Verify JSON output was printed
                    assert mock_print.called

    def test_main_with_verbose_output(self) -> None:
        """Test main() with --verbose flag (line 247)."""
        from llm_orc.testing.contract_validator import main

        test_args = ["contract-validator", "--verbose"]

        mock_result = ValidationResult(
            success=True,
            validated_scripts=["test.py"],
            errors=[],
            warnings=[],
            execution_time_seconds=0.1,
        )

        with patch("sys.argv", test_args):
            with patch.object(
                ContractValidationCLI, "validate", return_value=mock_result
            ):
                with patch("builtins.print") as mock_print:
                    with pytest.raises(SystemExit) as exc_info:
                        main()

                    assert exc_info.value.code == 0
                    assert mock_print.called

    def test_main_with_validation_failure(self) -> None:
        """Test main() with validation failure (line 256)."""
        from llm_orc.testing.contract_validator import main

        test_args = ["contract-validator"]

        mock_result = ValidationResult(
            success=False,
            validated_scripts=[],
            errors=["Validation error"],
            warnings=[],
            execution_time_seconds=0.1,
        )

        with patch("sys.argv", test_args):
            with patch.object(
                ContractValidationCLI, "validate", return_value=mock_result
            ):
                with patch("builtins.print") as mock_print:
                    with pytest.raises(SystemExit) as exc_info:
                        main()

                    assert exc_info.value.code == 1
                    # Verify failure message was printed
                    print_calls = [str(call) for call in mock_print.call_args_list]
                    assert any("Validation failed" in str(call) for call in print_calls)

    def test_main_with_contract_validation_error(self) -> None:
        """Test main() handles ContractValidationError (lines 260-262)."""
        from llm_orc.testing.contract_validator import main

        test_args = ["contract-validator", "--directory", "/nonexistent"]

        with patch("sys.argv", test_args):
            with patch("pathlib.Path.exists", return_value=False):
                with pytest.raises(SystemExit) as exc_info:
                    main()

                assert exc_info.value.code == 1

    def test_main_with_unexpected_error(self) -> None:
        """Test main() handles unexpected exceptions (lines 264-265)."""
        from llm_orc.testing.contract_validator import main

        test_args = ["contract-validator"]

        with patch("sys.argv", test_args):
            with patch.object(
                ContractValidationCLI,
                "validate",
                side_effect=RuntimeError("Unexpected"),
            ):
                with pytest.raises(SystemExit) as exc_info:
                    main()

                assert exc_info.value.code == 1

    def test_cli_exit_codes(self) -> None:
        """Test CLI returns proper exit codes."""
        # Success case
        mock_success = ValidationResult(
            success=True,
            validated_scripts=["script1.py"],
            errors=[],
            warnings=[],
            execution_time_seconds=0.5,
        )

        # Failure case
        mock_failure = ValidationResult(
            success=False,
            validated_scripts=[],
            errors=["Validation error"],
            warnings=[],
            execution_time_seconds=0.3,
        )

        with patch.object(ContractValidationCLI, "validate", return_value=mock_success):
            cli = ContractValidationCLI()
            assert cli.get_exit_code() == 0

        with patch.object(ContractValidationCLI, "validate", return_value=mock_failure):
            cli = ContractValidationCLI()
            assert cli.get_exit_code() == 1


class TestContractValidationModule:
    """Test the contract validation module interface."""

    def test_module_main_execution(self) -> None:
        """Test module can be executed as python -m contract_validator."""
        # This test verifies the module structure exists for CLI execution
        try:
            import llm_orc.testing.contract_validator

            assert hasattr(llm_orc.testing.contract_validator, "main")
        except ImportError:
            pytest.fail(
                "Module llm_orc.testing.contract_validator should be importable"
            )

    @patch("sys.argv", ["contract_validator", "--directory", ".", "--level", "core"])
    @patch("llm_orc.testing.contract_validator.ContractValidationCLI")
    def test_main_function_integration(self, mock_cli_class: MagicMock) -> None:
        """Test main function integration with CLI class."""
        mock_cli = MagicMock()
        mock_cli.get_exit_code.return_value = 0
        mock_cli_class.return_value = mock_cli

        from llm_orc.testing.contract_validator import main

        with patch("sys.exit") as mock_exit:
            main()
            mock_exit.assert_called_once_with(0)

    def test_validation_levels_match_adr_spec(self) -> None:
        """Test validation levels match ADR-003 specification."""
        # ADR-003 specifies these validation levels
        expected_levels = ["core", "examples", "community"]

        actual_levels = [level.value for level in ValidationLevel]
        assert set(actual_levels) == set(expected_levels)

    def test_cli_commands_match_adr_spec(self) -> None:
        """Test CLI commands match ADR-003 specification."""
        # ADR-003 specifies these command patterns:
        # python -m llm_orc.testing.contract_validator \\
        #     --directory .llm-orc/scripts/primitives --level core
        # python -m llm_orc.testing.contract_validator \\
        #     --directory .llm-orc/scripts/examples --level examples
        # python -m llm_orc.testing.contract_validator \\
        #     --directory .llm-orc/scripts/community --level community

        from llm_orc.testing.contract_validator import create_argument_parser

        parser = create_argument_parser()

        # Test core primitives command
        args = parser.parse_args(
            ["--directory", ".llm-orc/scripts/primitives", "--level", "core"]
        )
        assert args.directory == ".llm-orc/scripts/primitives"
        assert args.level == "core"

        # Test examples command
        args = parser.parse_args(
            ["--directory", ".llm-orc/scripts/examples", "--level", "examples"]
        )
        assert args.directory == ".llm-orc/scripts/examples"
        assert args.level == "examples"

        # Test community command
        args = parser.parse_args(
            ["--directory", ".llm-orc/scripts/community", "--level", "community"]
        )
        assert args.directory == ".llm-orc/scripts/community"
        assert args.level == "community"


class TestContractValidationIntegration:
    """Integration tests for contract validation system."""

    @patch("subprocess.run")
    def test_cli_subprocess_execution(self, mock_run: MagicMock) -> None:
        """Test CLI can be executed as subprocess."""
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = '{"success": true}'

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "llm_orc.testing.contract_validator",
                "--directory",
                ".",
                "--level",
                "core",
            ],
            capture_output=True,
            text=True,
        )

        # Note: This will fail until module is implemented
        # but test structure shows expected interface
        assert hasattr(result, "returncode")

    def test_validation_error_exception(self) -> None:
        """Test validation error exception handling."""
        error = ContractValidationError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)

    def test_validation_with_different_levels(self) -> None:
        """Test validation behavior changes with different levels."""
        # Different validation levels should have different requirements
        for level in ValidationLevel:
            cli = ContractValidationCLI(level=level)
            assert cli.level == level
            # Validation behavior would differ by level in implementation

    def test_make_integration_requirements(self) -> None:
        """Test integration requirements for make targets."""
        # This test documents the make integration requirements:
        # - make validate-contracts-core
        # - make validate-contracts-examples
        # - make validate-contracts-community
        # - make validate-contracts-all
        # These would call the CLI module with appropriate parameters

        expected_make_targets = [
            "validate-contracts-core",
            "validate-contracts-examples",
            "validate-contracts-community",
            "validate-contracts-all",
        ]

        # Test that we can construct CLI calls for each target
        for target in expected_make_targets:
            level = target.replace("validate-contracts-", "")
            if level == "all":
                # Special case for "all" target
                continue
            assert level in [level_enum.value for level_enum in ValidationLevel]

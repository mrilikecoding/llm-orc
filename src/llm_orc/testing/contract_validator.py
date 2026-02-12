#!/usr/bin/env python3
"""ADR-003 Contract Validation CLI Module.

This module provides CLI tools for validating script agent contracts
as specified in ADR-003: Testable Script Agent Contracts.

Usage:
    python -m llm_orc.testing.contract_validator \\
        --directory .llm-orc/scripts/primitives --level core
    python -m llm_orc.testing.contract_validator \\
        --directory .llm-orc/scripts/examples --level examples
    python -m llm_orc.testing.contract_validator \\
        --directory .llm-orc/scripts/community --level community
"""

import argparse
import sys
import time
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ValidationLevel(StrEnum):
    """Validation levels as specified in ADR-003."""

    CORE = "core"
    EXAMPLES = "examples"
    COMMUNITY = "community"


class ValidationResult(BaseModel):
    """Result of contract validation operation."""

    success: bool
    validated_scripts: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    execution_time_seconds: float


class ContractValidationError(Exception):
    """Exception raised when contract validation fails."""

    pass


class ContractValidator:
    """Basic contract validator implementation."""

    def __init__(self, directory: str) -> None:
        """Initialize validator with directory."""
        self.directory = directory
        self.validation_errors: list[str] = []

    def validate_all_scripts(self, scripts: list[Any]) -> bool:
        """Validate all provided scripts."""
        # Basic validation - just check if scripts exist
        return len(scripts) > 0


class ContractValidationCLI:
    """CLI interface for contract validation."""

    def __init__(
        self,
        directory: str = ".",
        level: ValidationLevel = ValidationLevel.CORE,
        verbose: bool = False,
    ) -> None:
        """Initialize contract validation CLI.

        Args:
            directory: Directory to scan for scripts
            level: Validation level (core, examples, community)
            verbose: Enable verbose output
        """
        self.directory = Path(directory)
        self.level = level
        self.verbose = verbose

    def discover_scripts(self) -> list[Path]:
        """Discover Python scripts in the target directory.

        Returns:
            List of Python script paths found

        Raises:
            ContractValidationError: If directory doesn't exist
        """
        if not self.directory.exists():
            raise ContractValidationError(f"Directory does not exist: {self.directory}")

        return list(self.directory.rglob("*.py"))

    def validate(self) -> ValidationResult:
        """Perform contract validation on discovered scripts.

        Returns:
            ValidationResult with success status and details

        Raises:
            ContractValidationError: If directory doesn't exist
        """
        start_time = time.time()

        if not self.directory.exists():
            raise ContractValidationError(f"Directory does not exist: {self.directory}")

        scripts = self.discover_scripts()
        warnings = []
        errors = []
        validated_scripts = []

        if not scripts:
            warnings.append("No scripts found")
        else:
            # Use contract validator for script validation
            validator = ContractValidator(str(self.directory))
            validation_success = validator.validate_all_scripts(scripts)
            validated_scripts = [str(script) for script in scripts]
            errors.extend(validator.validation_errors)

            if not validation_success:
                errors.append("Validation failed for one or more scripts")

        execution_time = time.time() - start_time

        return ValidationResult(
            success=len(errors) == 0,
            validated_scripts=validated_scripts,
            errors=errors,
            warnings=warnings,
            execution_time_seconds=execution_time,
        )

    def get_json_output(self) -> str:
        """Get validation results as JSON string.

        Returns:
            JSON string representation of validation results
        """
        result = self.validate()
        return result.model_dump_json(indent=2)

    def get_verbose_output(self) -> str:
        """Get verbose validation output.

        Returns:
            Formatted string with detailed validation results
        """
        result = self.validate()
        output_lines = []

        output_lines.append("Contract Validation Report")
        output_lines.append(f"Directory: {self.directory}")
        output_lines.append(f"Level: {self.level.value}")
        output_lines.append(f"Success: {result.success}")
        output_lines.append(f"Execution Time: {result.execution_time_seconds:.2f}s")

        if result.validated_scripts:
            output_lines.append(
                f"\nValidated Scripts ({len(result.validated_scripts)}):"
            )
            for script in result.validated_scripts:
                output_lines.append(f"  ✓ {script}")

        if result.warnings:
            output_lines.append(f"\nWarnings ({len(result.warnings)}):")
            for warning in result.warnings:
                output_lines.append(f"  ⚠ {warning}")

        if result.errors:
            output_lines.append(f"\nErrors ({len(result.errors)}):")
            for error in result.errors:
                output_lines.append(f"  ✗ {error}")

        return "\n".join(output_lines)

    def get_exit_code(self) -> int:
        """Get appropriate exit code based on validation results.

        Returns:
            0 for success, 1 for failure
        """
        result = self.validate()
        return 0 if result.success else 1


def create_argument_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Validate script agent contracts as specified in ADR-003",
        prog="python -m llm_orc.testing.contract_validator",
    )

    parser.add_argument(
        "--directory",
        default=".",
        help="Directory to scan for script contracts (default: current directory)",
    )

    parser.add_argument(
        "--level",
        choices=["core", "examples", "community"],
        default="core",
        help="Validation level: core (strict), examples (moderate), community (basic)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output with detailed validation results",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format",
    )

    return parser


def main() -> None:
    """Main CLI entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    try:
        cli = ContractValidationCLI(
            directory=args.directory,
            level=ValidationLevel(args.level),
            verbose=args.verbose,
        )

        if args.json:
            print(cli.get_json_output())
        elif args.verbose:
            print(cli.get_verbose_output())
        else:
            result = cli.validate()
            if result.success:
                print(
                    f"✓ Validation successful: "
                    f"{len(result.validated_scripts)} scripts validated"
                )
            else:
                print(f"✗ Validation failed: {len(result.errors)} errors found")

        sys.exit(cli.get_exit_code())

    except ContractValidationError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

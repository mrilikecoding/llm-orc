#!/usr/bin/env python3
"""Validate ADR status consistency with BDD scenario state."""

import re
import sys
from enum import Enum
from pathlib import Path
from typing import Any


class ADRStatus(Enum):
    """ADR lifecycle status definitions."""

    PROPOSED = "proposed"
    IN_PROGRESS = "in_progress"
    IMPLEMENTED = "implemented"
    VALIDATED = "validated"
    ACCEPTED = "accepted"
    DEPRECATED = "deprecated"


def extract_adr_number(filename: str) -> str:
    """Extract ADR number from filename."""
    match = re.search(r"adr-(\d+)", filename.lower())
    if match:
        return match.group(1)
    match = re.search(r"^(\d+)", filename)
    if match:
        return match.group(1)
    return ""


def parse_adr_status(content: str) -> ADRStatus | None:
    """Parse ADR status from file content."""
    # Look for "## Status" section
    status_match = re.search(r"##\s+Status\s*\n([^\n]+)", content, re.IGNORECASE)
    if not status_match:
        return None

    status_text = status_match.group(1).strip().lower()

    # Map status text to enum
    status_mapping = {
        "proposed": ADRStatus.PROPOSED,
        "in progress": ADRStatus.IN_PROGRESS,
        "in_progress": ADRStatus.IN_PROGRESS,
        "implemented": ADRStatus.IMPLEMENTED,
        "validated": ADRStatus.VALIDATED,
        "accepted": ADRStatus.ACCEPTED,
        "deprecated": ADRStatus.DEPRECATED,
    }

    return status_mapping.get(status_text)


def check_bdd_scenarios_exist(adr_number: str) -> bool:
    """Check if BDD scenarios exist for the given ADR."""
    bdd_dir = Path("tests/bdd/features")
    if not bdd_dir.exists():
        return False

    # Look for feature files matching adr-{number}
    pattern = f"adr-{adr_number.zfill(3)}*.feature"
    matching_files = list(bdd_dir.glob(pattern))
    return len(matching_files) > 0


def run_bdd_scenarios(adr_number: str) -> tuple[bool, str]:
    """Run BDD scenarios for the given ADR and return pass/fail status."""
    import subprocess

    try:
        # Run pytest for the specific ADR scenarios
        result = subprocess.run(
            [
                "uv",
                "run",
                "pytest",
                "tests/bdd/",
                "-k",
                f"adr-{adr_number.zfill(3)}",
                "-v",
                "--tb=short",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "BDD scenario execution timed out"
    except Exception as e:
        return False, f"Error running BDD scenarios: {e}"


def validate_status_transition_rules(
    current_status: ADRStatus, bdd_scenarios_pass: bool, bdd_scenarios_exist: bool
) -> tuple[bool, str]:
    """Validate that ADR status follows transition rules."""
    issues = []

    # Check status consistency with BDD state
    if current_status == ADRStatus.IN_PROGRESS:
        if not bdd_scenarios_exist:
            issues.append(
                "IN_PROGRESS status requires BDD scenarios to exist"
            )

    elif current_status == ADRStatus.IMPLEMENTED:
        if not bdd_scenarios_exist:
            issues.append(
                "IMPLEMENTED status requires BDD scenarios to exist"
            )
        elif not bdd_scenarios_pass:
            issues.append(
                "IMPLEMENTED status requires all BDD scenarios to pass"
            )

    elif current_status in [ADRStatus.VALIDATED, ADRStatus.ACCEPTED]:
        if not bdd_scenarios_exist:
            issues.append(
                f"{current_status.value.upper()} status requires BDD scenarios"
            )
        elif not bdd_scenarios_pass:
            issues.append(
                f"{current_status.value.upper()} status requires passing BDD scenarios"
            )

    return len(issues) == 0, "; ".join(issues)


def validate_single_adr(adr_file: Path) -> tuple[bool, list[str]]:
    """Validate a single ADR file for consistency."""
    issues = []

    try:
        content = adr_file.read_text()
        adr_number = extract_adr_number(adr_file.name)

        if not adr_number:
            issues.append(f"Could not extract ADR number from {adr_file.name}")
            return False, issues

        # Parse ADR status
        status = parse_adr_status(content)
        if status is None:
            issues.append(f"Could not parse status from {adr_file.name}")
            return False, issues

        # Check BDD scenarios
        bdd_scenarios_exist = check_bdd_scenarios_exist(adr_number)
        bdd_scenarios_pass = False

        if bdd_scenarios_exist:
            bdd_scenarios_pass, bdd_output = run_bdd_scenarios(adr_number)
            if not bdd_scenarios_pass and status in [
                ADRStatus.IMPLEMENTED,
                ADRStatus.VALIDATED,
                ADRStatus.ACCEPTED,
            ]:
                issues.append(
                    f"BDD scenarios failing for {adr_file.name}: {bdd_output}"
                )

        # Validate status transition rules
        status_valid, status_issue = validate_status_transition_rules(
            status, bdd_scenarios_pass, bdd_scenarios_exist
        )

        if not status_valid:
            issues.append(f"Status inconsistency in {adr_file.name}: {status_issue}")

        return len(issues) == 0, issues

    except Exception as e:
        issues.append(f"Error validating {adr_file.name}: {e}")
        return False, issues


def main() -> int:
    """Main validation function."""
    adrs_dir = Path("docs/adrs")
    if not adrs_dir.exists():
        print("❌ ADRs directory 'docs/adrs' not found")
        return 1

    adr_files = list(adrs_dir.glob("*.md"))
    if not adr_files:
        print("❌ No ADR files found in docs/adrs/")
        return 1

    print("🔍 Validating ADR consistency...")

    all_valid = True
    total_issues: list[str] = []

    for adr_file in sorted(adr_files):
        print(f"   Checking {adr_file.name}...")
        is_valid, issues = validate_single_adr(adr_file)

        if not is_valid:
            all_valid = False
            total_issues.extend(issues)
            for issue in issues:
                print(f"     ❌ {issue}")
        else:
            print(f"     ✅ {adr_file.name} is consistent")

    if all_valid:
        print(f"\n✅ All {len(adr_files)} ADRs are consistent with their status")
        return 0
    else:
        print(f"\n❌ Found {len(total_issues)} consistency issues:")
        for issue in total_issues:
            print(f"   • {issue}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
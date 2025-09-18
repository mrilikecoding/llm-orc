#!/usr/bin/env python3
"""Ensure all ADRs have corresponding BDD scenarios."""

import re
import sys
from pathlib import Path
from typing import NamedTuple


class ADRInfo(NamedTuple):
    """Information about an ADR file."""

    file_path: Path
    number: str
    title: str
    status: str | None


class BDDInfo(NamedTuple):
    """Information about a BDD feature file."""

    file_path: Path
    adr_number: str
    scenarios: list[str]


def extract_adr_number(filename: str) -> str:
    """Extract ADR number from filename."""
    # Try different patterns
    patterns = [
        r"adr-(\d+)",  # adr-001-title.md
        r"^(\d+)",     # 001-title.md
    ]

    for pattern in patterns:
        match = re.search(pattern, filename.lower())
        if match:
            return match.group(1)

    return ""


def parse_adr_file(adr_path: Path) -> ADRInfo:
    """Parse ADR file to extract metadata."""
    try:
        content = adr_path.read_text()

        # Extract number from filename
        number = extract_adr_number(adr_path.name)

        # Extract title (first # heading)
        title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        title = title_match.group(1) if title_match else "Unknown"

        # Extract status
        status_match = re.search(r"##\s+Status\s*\n([^\n]+)", content, re.IGNORECASE)
        status = status_match.group(1).strip() if status_match else None

        return ADRInfo(adr_path, number, title, status)

    except Exception as e:
        print(f"Warning: Error parsing {adr_path}: {e}")
        return ADRInfo(adr_path, "", "Parse Error", None)


def parse_bdd_file(bdd_path: Path) -> BDDInfo | None:
    """Parse BDD feature file to extract ADR information."""
    try:
        content = bdd_path.read_text()

        # Extract ADR number from filename
        adr_number = extract_adr_number(bdd_path.name)
        if not adr_number:
            return None

        # Extract scenario names
        scenarios = re.findall(r"^\s*Scenario:?\s+(.+)$", content, re.MULTILINE)

        return BDDInfo(bdd_path, adr_number, scenarios)

    except Exception as e:
        print(f"Warning: Error parsing {bdd_path}: {e}")
        return None


def check_adr_bdd_coverage() -> tuple[bool, dict[str, any]]:
    """Check coverage between ADRs and BDD scenarios."""
    adrs_dir = Path("docs/adrs")
    bdd_dir = Path("tests/bdd/features")

    # Collect ADR files
    adrs = []
    if adrs_dir.exists():
        for adr_file in adrs_dir.glob("*.md"):
            adr_info = parse_adr_file(adr_file)
            if adr_info.number:  # Only include if we can extract number
                adrs.append(adr_info)

    # Collect BDD files
    bdd_files = []
    if bdd_dir.exists():
        for bdd_file in bdd_dir.glob("adr-*.feature"):
            bdd_info = parse_bdd_file(bdd_file)
            if bdd_info:
                bdd_files.append(bdd_info)

    # Create mapping of ADR numbers to BDD files
    bdd_by_adr = {bdd.adr_number: bdd for bdd in bdd_files}

    # Check coverage
    missing_bdd = []
    orphaned_bdd = []
    status_inconsistencies = []

    # Check for ADRs missing BDD scenarios
    for adr in adrs:
        if adr.number not in bdd_by_adr:
            missing_bdd.append(adr)
        else:
            # Check status consistency
            bdd_info = bdd_by_adr[adr.number]
            if adr.status and adr.status.lower() in ["in_progress", "implemented", "validated", "accepted"]:
                if not bdd_info.scenarios:
                    status_inconsistencies.append((adr, "Status requires scenarios but none found"))

    # Check for orphaned BDD files
    adr_numbers = {adr.number for adr in adrs}
    for bdd in bdd_files:
        if bdd.adr_number not in adr_numbers:
            orphaned_bdd.append(bdd)

    all_good = len(missing_bdd) == 0 and len(orphaned_bdd) == 0 and len(status_inconsistencies) == 0

    results = {
        "total_adrs": len(adrs),
        "total_bdd_files": len(bdd_files),
        "missing_bdd": missing_bdd,
        "orphaned_bdd": orphaned_bdd,
        "status_inconsistencies": status_inconsistencies,
        "coverage_percentage": (len(adrs) - len(missing_bdd)) / len(adrs) * 100 if adrs else 0,
    }

    return all_good, results


def suggest_bdd_filename(adr: ADRInfo) -> str:
    """Suggest BDD filename for an ADR."""
    # Convert title to slug
    slug = re.sub(r"[^a-zA-Z0-9\s-]", "", adr.title.lower())
    slug = re.sub(r"\s+", "-", slug.strip())
    slug = re.sub(r"-+", "-", slug)

    return f"adr-{adr.number.zfill(3)}-{slug}.feature"


def generate_bdd_template(adr: ADRInfo) -> str:
    """Generate BDD template for an ADR."""
    feature_name = f"ADR-{adr.number.zfill(3)}: {adr.title}"
    filename = suggest_bdd_filename(adr)

    template = f'''Feature: {feature_name}
  As a system architect
  I want to ensure {adr.title.lower()}
  So that the architecture decision is properly implemented and validated

  Background:
    Given the system follows ADR-{adr.number.zfill(3)} specifications

  Scenario: Basic ADR-{adr.number.zfill(3)} compliance validation
    Given the ADR-{adr.number.zfill(3)} implementation exists
    When I validate the implementation
    Then it should comply with the architectural requirements
    And it should maintain backward compatibility

  # TODO: Add specific scenarios based on ADR requirements
  # Scenario: [Specific behavior 1]
  # Scenario: [Specific behavior 2]
'''

    return template


def main() -> int:
    """Main function to check BDD coverage."""
    print("🔍 Checking ADR-BDD coverage...")

    all_good, results = check_adr_bdd_coverage()

    print(f"📊 Coverage Summary:")
    print(f"   Total ADRs: {results['total_adrs']}")
    print(f"   Total BDD files: {results['total_bdd_files']}")
    print(f"   Coverage: {results['coverage_percentage']:.1f}%")

    if results['missing_bdd']:
        print(f"\n❌ ADRs missing BDD scenarios ({len(results['missing_bdd'])}):")
        for adr in results['missing_bdd']:
            suggested_file = suggest_bdd_filename(adr)
            print(f"   • ADR-{adr.number}: {adr.title}")
            print(f"     Status: {adr.status or 'Unknown'}")
            print(f"     Suggested file: tests/bdd/features/{suggested_file}")

            # If user wants to generate templates
            if "--generate" in sys.argv:
                template_path = Path(f"tests/bdd/features/{suggested_file}")
                template_path.parent.mkdir(parents=True, exist_ok=True)
                template_path.write_text(generate_bdd_template(adr))
                print(f"     ✅ Generated template: {template_path}")

    if results['orphaned_bdd']:
        print(f"\n⚠️  Orphaned BDD files ({len(results['orphaned_bdd'])}):")
        for bdd in results['orphaned_bdd']:
            print(f"   • {bdd.file_path.name} (ADR-{bdd.adr_number} not found)")

    if results['status_inconsistencies']:
        print(f"\n❌ Status inconsistencies ({len(results['status_inconsistencies'])}):")
        for adr, issue in results['status_inconsistencies']:
            print(f"   • ADR-{adr.number}: {issue}")

    if all_good:
        print("\n✅ All ADRs have proper BDD coverage!")
        return 0
    else:
        print(f"\n❌ Found {len(results['missing_bdd']) + len(results['orphaned_bdd']) + len(results['status_inconsistencies'])} issues")
        print("\nTo generate BDD templates for missing ADRs, run:")
        print("python scripts/check_bdd_coverage.py --generate")
        return 1


if __name__ == "__main__":
    sys.exit(main())
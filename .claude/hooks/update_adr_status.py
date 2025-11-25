#!/usr/bin/env python3
"""Update ADR status based on BDD scenario results."""

import re
import sys
from datetime import datetime
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


def find_adr_file(adr_number: str) -> Path | None:
    """Find ADR file by number."""
    adrs_dir = Path("docs/adrs")
    if not adrs_dir.exists():
        return None

    # Try different patterns
    patterns = [
        f"{adr_number.zfill(3)}-*.md",
        f"adr-{adr_number.zfill(3)}-*.md",
        f"{adr_number}-*.md",
        f"adr-{adr_number}-*.md",
    ]

    for pattern in patterns:
        matches = list(adrs_dir.glob(pattern))
        if matches:
            return matches[0]

    return None


def parse_current_status(content: str) -> ADRStatus | None:
    """Parse current ADR status from file content."""
    status_match = re.search(r"##\s+Status\s*\n([^\n]+)", content, re.IGNORECASE)
    if not status_match:
        return None

    status_text = status_match.group(1).strip().lower()

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


def validate_transition(
    current_status: ADRStatus | None, new_status: ADRStatus
) -> tuple[bool, str]:
    """Validate status transition follows allowed rules."""
    if current_status is None:
        # No current status, can only set to PROPOSED
        if new_status == ADRStatus.PROPOSED:
            return True, ""
        return False, "Initial status must be PROPOSED"

    allowed_transitions = {
        ADRStatus.PROPOSED: [ADRStatus.IN_PROGRESS],
        ADRStatus.IN_PROGRESS: [ADRStatus.IMPLEMENTED, ADRStatus.PROPOSED],
        ADRStatus.IMPLEMENTED: [ADRStatus.VALIDATED, ADRStatus.IN_PROGRESS],
        ADRStatus.VALIDATED: [ADRStatus.ACCEPTED, ADRStatus.IMPLEMENTED],
        ADRStatus.ACCEPTED: [ADRStatus.DEPRECATED],
        ADRStatus.DEPRECATED: [],  # Terminal state
    }

    allowed = allowed_transitions.get(current_status, [])
    if new_status in allowed:
        return True, ""

    return (
        False,
        f"Invalid transition from {current_status.value} to {new_status.value}. "
        f"Allowed: {[s.value for s in allowed]}",
    )


def get_implementation_status_template(status: ADRStatus, adr_number: str) -> str:
    """Get implementation status template for the given status."""
    scenario_file = f"tests/bdd/features/adr-{adr_number.zfill(3)}-*.feature"

    if status == ADRStatus.PROPOSED:
        return """## Implementation Status
- [ ] BDD scenarios created
- [ ] Step definitions implemented (Red phase)
- [ ] Implementation tasks planned
- [ ] Dependencies identified"""

    elif status == ADRStatus.IN_PROGRESS:
        return f"""## Implementation Status
- [x] BDD scenarios created in {scenario_file}
- [ ] Core implementation begun
- [ ] Schema system implemented
- [ ] Integration layer complete
- [ ] All BDD scenarios passing
- [ ] Error handling compliance (ADR-003)
- [ ] Type safety validation"""

    elif status == ADRStatus.IMPLEMENTED:
        return f"""## Implementation Status
- [x] BDD scenarios created in {scenario_file}
- [x] Core implementation complete
- [x] All BDD scenarios passing
- [x] Integration tests passing
- [ ] Refactor phase complete
- [ ] Performance benchmarks met
- [ ] Documentation updated"""

    elif status == ADRStatus.VALIDATED:
        return f"""## Implementation Status
- [x] All implementation complete
- [x] All BDD scenarios passing
- [x] Refactor phase complete
- [x] Performance benchmarks met
- [ ] Peer review complete
- [ ] Integration examples provided
- [ ] Migration documentation complete"""

    elif status == ADRStatus.ACCEPTED:
        return f"""## Implementation Status
- [x] All implementation complete
- [x] All BDD scenarios passing
- [x] Refactor phase complete
- [x] Performance benchmarks met
- [x] Peer review complete
- [x] Integration examples provided
- [x] Migration documentation complete"""

    return ""


def update_adr_status_in_content(
    content: str, new_status: ADRStatus, adr_number: str
) -> str:
    """Update ADR status and implementation section in content."""
    # Update status line
    status_pattern = r"(##\s+Status\s*\n)([^\n]+)"
    if re.search(status_pattern, content, re.IGNORECASE):
        content = re.sub(
            status_pattern,
            f"\\1{new_status.value.title()}",
            content,
            flags=re.IGNORECASE,
        )
    else:
        # Add status section after title
        title_pattern = r"(#\s+[^\n]+\n)"
        content = re.sub(
            title_pattern,
            f"\\1\n## Status\n{new_status.value.title()}\n",
            content,
        )

    # Update implementation status
    impl_template = get_implementation_status_template(new_status, adr_number)
    impl_pattern = r"##\s+Implementation\s+Status\s*\n(.*?)(?=\n##|\nContributed|$)"

    if re.search(impl_pattern, content, re.IGNORECASE | re.DOTALL):
        content = re.sub(
            impl_pattern,
            f"## Implementation Status\n{impl_template.split('## Implementation Status')[1]}\n",
            content,
            flags=re.IGNORECASE | re.DOTALL,
        )
    else:
        # Add implementation status section after status
        status_line_pattern = r"(##\s+Status\s*\n[^\n]+\n)"
        content = re.sub(
            status_line_pattern,
            f"\\1\n{impl_template}\n",
            content,
            flags=re.IGNORECASE,
        )

    # Add progress log entry
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    log_entry = f"- **{timestamp}**: Status updated to {new_status.value.title()}"

    log_pattern = r"##\s+Implementation\s+Progress\s+Log\s*\n(.*?)(?=\n##|\nContributed|$)"
    if re.search(log_pattern, content, re.IGNORECASE | re.DOTALL):
        content = re.sub(
            log_pattern,
            f"## Implementation Progress Log\n{log_entry}\n\\1",
            content,
            flags=re.IGNORECASE | re.DOTALL,
        )
    else:
        # Add progress log section
        impl_end_pattern = r"(## Implementation Status.*?(?=\n##|\nContributed|$))"
        content = re.sub(
            impl_end_pattern,
            f"\\1\n\n## Implementation Progress Log\n{log_entry}\n",
            content,
            flags=re.IGNORECASE | re.DOTALL,
        )

    return content


def main() -> int:
    """Main function to update ADR status."""
    if len(sys.argv) != 3:
        print("Usage: python update_adr_status.py <adr_number> <new_status>")
        print("Example: python update_adr_status.py 001 in_progress")
        print("Available statuses: proposed, in_progress, implemented, validated, accepted, deprecated")
        return 1

    adr_number = sys.argv[1]
    new_status_str = sys.argv[2].lower()

    # Parse new status
    status_mapping = {
        "proposed": ADRStatus.PROPOSED,
        "in_progress": ADRStatus.IN_PROGRESS,
        "implemented": ADRStatus.IMPLEMENTED,
        "validated": ADRStatus.VALIDATED,
        "accepted": ADRStatus.ACCEPTED,
        "deprecated": ADRStatus.DEPRECATED,
    }

    new_status = status_mapping.get(new_status_str)
    if new_status is None:
        print(f"❌ Invalid status: {new_status_str}")
        print(f"Available statuses: {list(status_mapping.keys())}")
        return 1

    # Find ADR file
    adr_file = find_adr_file(adr_number)
    if adr_file is None:
        print(f"❌ ADR file not found for number: {adr_number}")
        return 1

    try:
        # Read current content
        content = adr_file.read_text()
        current_status = parse_current_status(content)

        # Validate transition
        is_valid, error_msg = validate_transition(current_status, new_status)
        if not is_valid:
            print(f"❌ {error_msg}")
            return 1

        # Update content
        updated_content = update_adr_status_in_content(content, new_status, adr_number)

        # Write back to file
        adr_file.write_text(updated_content)

        current_str = current_status.value if current_status else "None"
        print(f"✅ Updated {adr_file.name}: {current_str} → {new_status.value}")

        return 0

    except Exception as e:
        print(f"❌ Error updating ADR status: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
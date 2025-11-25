#!/usr/bin/env python3
"""Generate ADR progress summary for session start automation."""

import json
import re
import sys
from pathlib import Path
from typing import Any


def extract_adr_number(filename: str) -> str:
    """Extract ADR number from filename."""
    patterns = [
        r"adr-(\d+)",  # adr-001-title.md
        r"^(\d+)",     # 001-title.md
    ]

    for pattern in patterns:
        match = re.search(pattern, filename.lower())
        if match:
            return match.group(1)

    return ""


def parse_adr_status_and_progress(content: str) -> dict[str, Any]:
    """Parse ADR status and implementation progress."""
    # Extract status
    status_match = re.search(r"##\s+Status\s*\n([^\n]+)", content, re.IGNORECASE)
    status = status_match.group(1).strip() if status_match else "Unknown"

    # Extract implementation status checklist
    impl_pattern = r"##\s+Implementation\s+Status\s*\n(.*?)(?=\n##|\nContributed|$)"
    impl_match = re.search(impl_pattern, content, re.IGNORECASE | re.DOTALL)

    completed_tasks = []
    pending_tasks = []

    if impl_match:
        impl_content = impl_match.group(1)
        # Parse checkboxes
        completed_matches = re.findall(r"- \[x\] (.+)", impl_content)
        pending_matches = re.findall(r"- \[ \] (.+)", impl_content)

        completed_tasks = [task.strip() for task in completed_matches]
        pending_tasks = [task.strip() for task in pending_matches]

    # Calculate progress percentage
    total_tasks = len(completed_tasks) + len(pending_tasks)
    progress_pct = (len(completed_tasks) / total_tasks * 100) if total_tasks > 0 else 0

    return {
        "status": status,
        "completed_tasks": completed_tasks,
        "pending_tasks": pending_tasks,
        "progress_percentage": progress_pct,
        "total_tasks": total_tasks,
    }


def check_bdd_scenario_status(adr_number: str) -> dict[str, Any]:
    """Check BDD scenario status for an ADR."""
    bdd_dir = Path("tests/bdd/features")
    pattern = f"adr-{adr_number.zfill(3)}*.feature"

    bdd_files = list(bdd_dir.glob(pattern)) if bdd_dir.exists() else []

    if not bdd_files:
        return {
            "scenarios_exist": False,
            "scenario_count": 0,
            "scenarios_pass": False,
            "scenario_files": [],
        }

    # Count scenarios
    total_scenarios = 0
    scenario_files = []

    for bdd_file in bdd_files:
        try:
            content = bdd_file.read_text()
            scenarios = re.findall(r"^\s*Scenario:?\s+(.+)$", content, re.MULTILINE)
            total_scenarios += len(scenarios)
            scenario_files.append({
                "file": str(bdd_file.name),
                "scenario_count": len(scenarios),
                "scenarios": scenarios,
            })
        except Exception:
            pass

    # Try to run scenarios to check if they pass
    scenarios_pass = False
    try:
        import subprocess

        result = subprocess.run(
            [
                "uv",
                "run",
                "pytest",
                "tests/bdd/",
                "-k",
                f"adr-{adr_number.zfill(3)}",
                "-v",
                "--tb=no",
                "--no-cov",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        scenarios_pass = result.returncode == 0
    except Exception:
        pass

    return {
        "scenarios_exist": True,
        "scenario_count": total_scenarios,
        "scenarios_pass": scenarios_pass,
        "scenario_files": scenario_files,
    }


def get_priority_recommendations(adr_data: list[dict[str, Any]]) -> list[str]:
    """Generate priority recommendations based on ADR status."""
    recommendations = []

    # Find ADRs that need immediate attention
    in_progress_adrs = [adr for adr in adr_data if adr["status"].lower() == "in progress"]
    implemented_adrs = [adr for adr in adr_data if adr["status"].lower() == "implemented"]

    # Priority 1: ADRs with failing BDD scenarios
    failing_scenarios = [
        adr for adr in adr_data
        if adr["bdd_info"]["scenarios_exist"] and not adr["bdd_info"]["scenarios_pass"]
    ]

    if failing_scenarios:
        recommendations.append(
            f"🚨 CRITICAL: {len(failing_scenarios)} ADR(s) have failing BDD scenarios - "
            f"investigate and fix immediately"
        )

    # Priority 2: IN_PROGRESS ADRs ready for next phase
    ready_for_implemented = [
        adr for adr in in_progress_adrs
        if adr["bdd_info"]["scenarios_exist"] and adr["bdd_info"]["scenarios_pass"]
    ]

    if ready_for_implemented:
        recommendations.append(
            f"✅ {len(ready_for_implemented)} ADR(s) may be ready for IMPLEMENTED status"
        )

    # Priority 3: ADRs missing BDD scenarios
    missing_bdd = [adr for adr in adr_data if not adr["bdd_info"]["scenarios_exist"]]

    if missing_bdd:
        recommendations.append(
            f"📝 {len(missing_bdd)} ADR(s) need BDD scenarios created"
        )

    # Priority 4: Most progressed ADR for continued work
    if adr_data:
        highest_progress = max(adr_data, key=lambda x: x["progress"]["progress_percentage"])
        if highest_progress["progress"]["progress_percentage"] > 50:
            recommendations.append(
                f"🎯 Continue work on ADR-{highest_progress['number']} "
                f"({highest_progress['progress']['progress_percentage']:.0f}% complete)"
            )

    return recommendations


def generate_adr_summary() -> dict[str, Any]:
    """Generate complete ADR progress summary."""
    adrs_dir = Path("docs/adrs")
    if not adrs_dir.exists():
        return {"error": "ADRs directory not found"}

    adr_files = list(adrs_dir.glob("*.md"))
    if not adr_files:
        return {"error": "No ADR files found"}

    adr_data = []

    for adr_file in sorted(adr_files):
        try:
            content = adr_file.read_text()
            adr_number = extract_adr_number(adr_file.name)

            if not adr_number:
                continue

            # Extract title
            title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
            title = title_match.group(1) if title_match else "Unknown"

            # Parse progress
            progress_info = parse_adr_status_and_progress(content)

            # Check BDD scenarios
            bdd_info = check_bdd_scenario_status(adr_number)

            adr_data.append({
                "number": adr_number,
                "title": title,
                "file": str(adr_file.name),
                "status": progress_info["status"],
                "progress": progress_info,
                "bdd_info": bdd_info,
            })

        except Exception as e:
            print(f"Warning: Error processing {adr_file}: {e}", file=sys.stderr)

    # Generate recommendations
    recommendations = get_priority_recommendations(adr_data)

    # Calculate overall progress
    total_adrs = len(adr_data)
    adrs_with_bdd = len([adr for adr in adr_data if adr["bdd_info"]["scenarios_exist"]])
    adrs_implemented = len([adr for adr in adr_data if adr["status"].lower() in ["implemented", "validated", "accepted"]])

    return {
        "summary": {
            "total_adrs": total_adrs,
            "adrs_with_bdd": adrs_with_bdd,
            "adrs_implemented": adrs_implemented,
            "bdd_coverage_percentage": (adrs_with_bdd / total_adrs * 100) if total_adrs > 0 else 0,
            "implementation_percentage": (adrs_implemented / total_adrs * 100) if total_adrs > 0 else 0,
        },
        "adrs": adr_data,
        "recommendations": recommendations,
    }


def format_for_claude_code(summary: dict[str, Any]) -> str:
    """Format summary for Claude Code session start display."""
    if "error" in summary:
        return f"❌ ADR Summary Error: {summary['error']}"

    s = summary["summary"]
    output = []

    # Header
    output.append("📋 ADR Progress Summary")
    output.append("=" * 50)

    # Overall stats
    output.append(f"📊 Overview: {s['total_adrs']} ADRs total")
    output.append(f"   • BDD Coverage: {s['bdd_coverage_percentage']:.0f}% ({s['adrs_with_bdd']}/{s['total_adrs']})")
    output.append(f"   • Implementation: {s['implementation_percentage']:.0f}% ({s['adrs_implemented']}/{s['total_adrs']})")

    # Recommendations
    if summary["recommendations"]:
        output.append("\n🎯 Priority Actions:")
        for rec in summary["recommendations"]:
            output.append(f"   {rec}")

    # ADR details
    output.append(f"\n📋 ADR Status Details:")
    for adr in summary["adrs"]:
        status_icon = {
            "proposed": "📝",
            "in progress": "🚧",
            "implemented": "✅",
            "validated": "🔍",
            "accepted": "✅",
        }.get(adr["status"].lower(), "❓")

        bdd_status = "✅" if adr["bdd_info"]["scenarios_pass"] else "❌" if adr["bdd_info"]["scenarios_exist"] else "⚪"
        progress_pct = adr["progress"]["progress_percentage"]

        output.append(
            f"   {status_icon} ADR-{adr['number']}: {adr['title']} "
            f"({adr['status']}, {progress_pct:.0f}%, BDD:{bdd_status})"
        )

        # Show next actions for active ADRs
        if adr["status"].lower() in ["proposed", "in progress"] and adr["progress"]["pending_tasks"]:
            next_task = adr["progress"]["pending_tasks"][0]
            output.append(f"      → Next: {next_task}")

    return "\n".join(output)


def main() -> int:
    """Main function."""
    try:
        summary = generate_adr_summary()

        if "--json" in sys.argv:
            print(json.dumps(summary, indent=2))
        else:
            print(format_for_claude_code(summary))

        return 0

    except Exception as e:
        print(f"❌ Error generating ADR summary: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
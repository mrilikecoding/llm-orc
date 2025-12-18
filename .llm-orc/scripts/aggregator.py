#!/usr/bin/env python3
"""
aggregator.py - Collect and structure multiple agent outputs for synthesis.

Usage:
  As script agent in ensemble:
    script: aggregator.py
    depends_on: [extractor-1, extractor-2, ...]
    parameters:
      format: "markdown"  # or "json"
"""
import json
import sys


def main() -> None:
    if not sys.stdin.isatty():
        config = json.loads(sys.stdin.read())
    else:
        config = {}

    dependencies = config.get("dependencies", {})
    output_format = config.get("format", "markdown")

    sections = []
    for agent_name, agent_output in dependencies.items():
        if isinstance(agent_output, dict):
            content = agent_output.get("response", agent_output.get("data", str(agent_output)))
        else:
            content = str(agent_output)

        if output_format == "markdown":
            sections.append(f"## {agent_name}\n\n{content}\n")
        else:
            sections.append({"agent": agent_name, "output": content})

    if output_format == "markdown":
        aggregated = "\n".join(sections)
    else:
        aggregated = sections

    result = {
        "success": True,
        "data": aggregated,
        "agent_count": len(dependencies),
        "format": output_format,
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

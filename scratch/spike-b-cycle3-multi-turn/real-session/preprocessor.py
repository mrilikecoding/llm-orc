"""Preprocessor for cheap-with-script real-session arm.

Outputs a deterministic context block summarizing:
  - Paths to canonical ensemble + script examples
  - Brief structural anchors (schema fields, key conventions)

This is the script-agent slot's analog at the agentic-coding-task level —
deterministic context that grounds the LLM orchestrator's understanding
of the codebase before it starts authoring. Parallels Spike A arm2's
'A2 + script input' pattern at multi-turn.

Output goes to stdout as a markdown block to be prepended to the task prompt.

NOTE: Spike code. Retained per practitioner policy until corpus close.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

SECTIONS: list[tuple[str, str, list[str]]] = [
    (
        "Reference ensemble — minimal two-agent pattern",
        ".llm-orc/ensembles/testing/test-script-agents.yaml",
        [
            "Look for: agents list with 'name', 'script' (path) for script agents; "
            "'name', 'model_profile', 'system_prompt' for LLM agents; 'depends_on' "
            "for dependency wiring."
        ],
    ),
    (
        "Reference script-agent — canonical contract",
        ".llm-orc/scripts/aggregator.py",
        [
            "Reads JSON config from stdin: {'dependencies': {agent_name: output_dict}, "
            "'parameters': {...}}.",
            "'dependencies' contains output of each upstream agent (the LLM that "
            "wrote the haiku in our case).",
            "Extract 'response' or 'data' field from each agent_output dict.",
            "Write structured output (JSON or markdown) to stdout.",
        ],
    ),
    (
        "Reference script-agent — simpler example",
        ".llm-orc/scripts/test-script.py",
        [
            "Demonstrates the simplest valid script-agent: stdlib-only, "
            "stdin JSON, stdout output.",
        ],
    ),
    (
        "Production code-review ensemble — for richer multi-agent reference",
        ".llm-orc/ensembles/development/code-review.yaml",
        [
            "Shows multiple agents with system prompts and a synthesizer "
            "depending on others.",
        ],
    ),
]

CONVENTIONS = """\
## Naming conventions
- Ensemble files: `<name>.yaml` in `.llm-orc/ensembles/<tier>/` (or anywhere addressable)
- Script files: `<name>.py` (typically in `.llm-orc/scripts/` for project tier)
- Ensemble.script paths: relative to repo root (cwd at invoke time)

## Script-agent contract (load-bearing)
Read JSON from stdin shaped as one of:
  - Legacy: {"dependencies": {agent_name: {"response": "..."}}, "parameters": {...}}
  - ScriptAgentInput (newer): {"agent_name": str, "input_data": Any,
    "dependencies": {agent_name: dict_or_str}, "parameters": dict}

Both formats may appear; defensive parsing recommended.

Output to stdout: a JSON object with at least `success: bool` for downstream
agents to react to. Other fields are free-form per agent purpose.

## Stdlib-only constraint
Do not pip install. Use only Python stdlib (json, sys, re, etc.) for script logic.
"""


def main() -> None:
    print("# DETERMINISTIC PREPROCESSING (script-agent context for cheap-with-script arm)")
    print()
    print("This block is prepended deterministically by a preprocessor before the LLM orchestrator")
    print("begins. It collects the codebase anchors the agent would otherwise need to discover via")
    print("multiple Read calls. Parallels Spike A arm 2 (A2 + script input) at multi-turn.")
    print()
    print(CONVENTIONS)
    print("## Reference files (already verified to exist at these paths)")
    for title, path, notes in SECTIONS:
        full_path = REPO_ROOT / path
        exists = full_path.exists()
        marker = "exists" if exists else "MISSING"
        print(f"\n### {title}")
        print(f"- **Path:** `{path}` ({marker})")
        if not exists:
            continue
        # Add a brief content excerpt for YAML files; otherwise just describe
        if path.endswith(".yaml"):
            try:
                content = full_path.read_text()
                preview = "\n".join(content.splitlines()[:30])
                print(f"- **Content (first 30 lines):**")
                print("  ```yaml")
                for line in preview.splitlines():
                    print(f"  {line}")
                print("  ```")
            except OSError as e:
                print(f"  (read failed: {e})")
        else:
            try:
                content = full_path.read_text()
                head = "\n".join(content.splitlines()[:20])
                print(f"- **Content (first 20 lines):**")
                print("  ```python")
                for line in head.splitlines():
                    print(f"  {line}")
                print("  ```")
            except OSError as e:
                print(f"  (read failed: {e})")
        print(f"- **Notes for this reference:**")
        for note in notes:
            print(f"  - {note}")
    print()
    print("---")
    print("End of deterministic preprocessing context. Task description follows below.")


if __name__ == "__main__":
    main()

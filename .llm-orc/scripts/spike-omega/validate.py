#!/usr/bin/env python3
"""Spike Ω-1 validate stage.

Reads the code-generator ensemble's response, runs ast.parse if the
target file is Python, emits the validated content or an error.

This is the §4 form-gate script. The bespoke LoopDriver's FormGate
(ADR-041 parse-check) translates here. For non-.py files the gate
passes through (Ω-4 will add JSON validation).

Emits JSON: {
    "valid": <bool>,
    "content": "<str>",
    "error": "<str or absent>"
}
"""

import ast
import json
import sys


def main() -> None:
    raw = sys.stdin.read().strip()
    content = ""

    try:
        data = json.loads(raw)
        # code-generator's output lives under dependencies.code-generator.response
        deps = data.get("dependencies", {}) if isinstance(data, dict) else {}
        cg_dep = deps.get("code-generator", {})
        response = cg_dep.get("response", "") if isinstance(cg_dep, dict) else ""

        # code-generator returns a JSON-serialized ExecutionResult envelope.
        # The bespoke LoopDriver solves this with LB-4 (D1 extraction): pull
        # the unique terminal agent's response from the results dict, or the
        # populated `deliverable` field.
        if isinstance(response, str):
            try:
                envelope = json.loads(response)
            except json.JSONDecodeError:
                envelope = None
            if isinstance(envelope, dict):
                # Prefer `deliverable` if populated by resolve_deliverable.
                if envelope.get("deliverable"):
                    content = envelope["deliverable"]
                # Else extract the terminal agent's response (the D1 logic).
                elif "results" in envelope and isinstance(envelope["results"], dict):
                    results = envelope["results"]
                    # code-generator's terminal agent is "synthesizer".
                    terminal = "synthesizer" if "synthesizer" in results else (
                        list(results.keys())[-1] if results else None
                    )
                    if terminal:
                        term_result = results[terminal]
                        if isinstance(term_result, dict):
                            content = term_result.get("response", "")
            else:
                content = response
        elif isinstance(response, dict):
            content = response.get("response", "")
    except (json.JSONDecodeError, TypeError):
        content = raw

    if not content:
        print(json.dumps({"valid": False, "content": "", "error": "empty content"}))
        return

    # Strip markdown fences for AST parsing (the FormGate pattern).
    # If the content contains a fenced code block block within prose,
    # extract the first fenced block (the bespoke's residual cleanup;
    # the form directive in dispatch-shim aims to make this a no-op).
    stripped = content.strip()
    if "```" in stripped:
        import re

        fenced = re.search(r"```(?:[a-zA-Z]+)?\n(.*?)```", stripped, re.DOTALL)
        if fenced:
            stripped = fenced.group(1).strip()
        else:
            # Only an opening fence — drop it and try anyway
            lines = stripped.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            stripped = "\n".join(lines).rstrip("`").rstrip()

    # D2b residual: if the LLM emitted the filename on the first line
    # (e.g. `converters.py\ndef converters...`), strip the line before
    # AST parsing. Detect a filename line — text matching `\w+\.[a-z]+`
    # optionally followed by colon, whitespace, or both.
    lines = stripped.splitlines()
    if lines:
        import re

        first = lines[0].strip()
        if re.match(r"^[a-zA-Z_][\w-]*\.(py|md|txt|sh|yaml|toml|json)(:?\s*)$", first):
            stripped = "\n".join(lines[1:]).strip()

    # If the leading lines are shell usage examples (Python code that
    # doesn't parse syntactically), try extracting valid Python from the
    # first `def`/`import`/`from`/`class` line — the LLM sometimes leads
    # with shell invocation examples.
    try:
        ast.parse(stripped)
    except SyntaxError:
        code_lines = stripped.splitlines()
        for i, line in enumerate(code_lines):
            stripped_line = line.lstrip()
            if stripped_line.startswith(("def ", "import ", "from ", "class ", "#!")):
                candidate = "\n".join(code_lines[i:])
                try:
                    ast.parse(candidate)
                    stripped = candidate
                except SyntaxError:
                    continue
                break

    # Final verdict.
    try:
        ast.parse(stripped)
        print(json.dumps({"valid": True, "content": stripped}))
    except SyntaxError as e:
        print(
            json.dumps(
                {"valid": False, "content": stripped, "error": f"ast.parse: {e}"}
            )
        )


if __name__ == "__main__":
    main()
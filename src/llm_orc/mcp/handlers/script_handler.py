"""Script management handler for MCP server."""

import subprocess
import sys
from pathlib import Path
from typing import Any

from llm_orc.mcp.project_context import ProjectContext


class ScriptHandler:
    """Manages primitive script operations."""

    def __init__(self, project_path: Path | None = None) -> None:
        """Initialize with optional project path."""
        self._project_path = project_path

    def set_project_context(self, ctx: ProjectContext) -> None:
        """Update handler to use new project context."""
        self._project_path = ctx.project_path

    def _get_scripts_dir(self) -> Path:
        """Get scripts directory path."""
        if self._project_path is not None:
            return self._project_path / ".llm-orc" / "scripts"
        return Path.cwd() / ".llm-orc" / "scripts"

    async def list_scripts(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """List available scripts."""
        category = arguments.get("category")
        scripts_dir = self._get_scripts_dir()

        if not scripts_dir.exists():
            return {"scripts": []}

        scripts: list[dict[str, Any]] = []
        if not category:
            scripts.extend(self._collect_root_scripts(scripts_dir))
        scripts.extend(self._collect_category_scripts(scripts_dir, category))

        return {"scripts": scripts}

    def _collect_root_scripts(self, scripts_dir: Path) -> list[dict[str, Any]]:
        """Collect scripts at the root level (no category)."""
        scripts: list[dict[str, Any]] = []
        for script_file in scripts_dir.glob("*.py"):
            if script_file.is_file():
                scripts.append(
                    {
                        "name": script_file.stem,
                        "category": "",
                        "path": str(script_file),
                    }
                )
        return scripts

    def _collect_category_scripts(
        self,
        scripts_dir: Path,
        category_filter: str | None,
    ) -> list[dict[str, Any]]:
        """Collect scripts from category subdirectories."""
        scripts: list[dict[str, Any]] = []
        for category_dir in scripts_dir.iterdir():
            if not category_dir.is_dir():
                continue
            cat_name = category_dir.name
            if category_filter and cat_name != category_filter:
                continue
            for script_file in category_dir.glob("*.py"):
                scripts.append(
                    {
                        "name": script_file.stem,
                        "category": cat_name,
                        "path": str(script_file),
                    }
                )
        return scripts

    async def get_script(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Get script details."""
        name = arguments.get("name")
        category = arguments.get("category")

        if not name:
            raise ValueError("name is required")
        if not category:
            raise ValueError("category is required")

        scripts_dir = self._get_scripts_dir()
        script_file = scripts_dir / category / f"{name}.py"

        if not script_file.exists():
            raise ValueError(f"Script '{category}/{name}' not found")

        content = script_file.read_text()
        description = self._extract_docstring(content)

        return {
            "name": name,
            "category": category,
            "path": str(script_file),
            "description": description,
            "source": content,
        }

    def _extract_docstring(self, content: str) -> str:
        """Extract module docstring from Python source code."""
        lines = content.split("\n")
        in_docstring = False
        docstring_lines: list[str] = []

        for line in lines:
            if '"""' in line or "'''" in line:
                if in_docstring:
                    break
                in_docstring = True
                if line.count('"""') == 2 or line.count("'''") == 2:
                    return self._strip_docstring_quotes(line.strip())
            elif in_docstring:
                docstring_lines.append(line)

        if docstring_lines:
            return "\n".join(docstring_lines).strip()
        return ""

    def _strip_docstring_quotes(self, text: str) -> str:
        """Remove docstring quote marks from text."""
        for quote in ('"""', "'''"):
            if text.startswith(quote):
                text = text[3:]
            if text.endswith(quote):
                text = text[:-3]
        return text.strip()

    async def test_script(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Test a script with sample input."""
        name = arguments.get("name")
        category = arguments.get("category")
        input_data = arguments.get("input", "")

        if not name:
            raise ValueError("name is required")
        if not category:
            raise ValueError("category is required")

        scripts_dir = self._get_scripts_dir()
        script_file = scripts_dir / category / f"{name}.py"

        if not script_file.exists():
            raise ValueError(f"Script '{category}/{name}' not found")

        try:
            result = subprocess.run(
                [sys.executable, str(script_file)],
                input=input_data,
                capture_output=True,
                text=True,
                timeout=30,
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Script execution timed out",
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    async def create_script(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Create a new script."""
        name = arguments.get("name")
        category = arguments.get("category")
        template = arguments.get("template", "basic")

        if not name:
            raise ValueError("name is required")
        if not category:
            raise ValueError("category is required")

        scripts_dir = self._get_scripts_dir()
        category_dir = scripts_dir / category
        script_file = category_dir / f"{name}.py"

        if script_file.exists():
            raise ValueError(f"Script '{category}/{name}' already exists")

        if template == "extraction":
            content = f'''"""Extraction script: {name}

Extracts structured data from input text.
"""

import json
import sys


def extract(text: str) -> dict:
    """Extract data from text."""
    # TODO: Implement extraction logic
    return {{"input_length": len(text)}}


if __name__ == "__main__":
    input_text = sys.stdin.read()
    result = extract(input_text)
    print(json.dumps(result))
'''
        else:
            content = f'''"""Primitive script: {name}

Process input and produce output.
"""

import sys


def process(text: str) -> str:
    """Process input text."""
    # TODO: Implement processing logic
    return text


if __name__ == "__main__":
    input_text = sys.stdin.read()
    result = process(input_text)
    print(result)
'''

        category_dir.mkdir(parents=True, exist_ok=True)
        script_file.write_text(content)

        return {
            "created": True,
            "path": str(script_file),
            "template": template,
        }

    async def delete_script(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Delete a script."""
        name = arguments.get("name")
        category = arguments.get("category")
        confirm = arguments.get("confirm", False)

        if not name:
            raise ValueError("name is required")
        if not category:
            raise ValueError("category is required")
        if not confirm:
            raise ValueError("Confirmation required to delete script")

        scripts_dir = self._get_scripts_dir()
        script_file = scripts_dir / category / f"{name}.py"

        if not script_file.exists():
            raise ValueError(f"Script '{category}/{name}' not found")

        script_file.unlink()

        return {
            "deleted": True,
            "name": name,
            "category": category,
        }

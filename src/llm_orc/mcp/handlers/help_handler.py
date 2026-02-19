"""Help documentation handler for MCP server."""

from typing import Any


class HelpHandler:
    """Builds help documentation for MCP server onboarding."""

    def get_help_documentation(self) -> dict[str, Any]:
        """Build help documentation."""
        return {
            "overview": (
                "llm-orc orchestrates multi-agent LLM ensembles. "
                "Use these tools to discover, run, and manage ensembles."
            ),
            "directory_structure": self._get_directory_structure_help(),
            "schemas": self._get_schema_help(),
            "tools": self._get_tools_help(),
            "workflows": self._get_workflow_help(),
        }

    def _get_directory_structure_help(self) -> dict[str, Any]:
        """Get directory structure documentation."""
        return {
            "local": {
                "path": ".llm-orc/",
                "description": ("Project-specific configuration (highest priority)"),
                "subdirs": {
                    "ensembles/": "Project ensembles (YAML files)",
                    "profiles/": "Model profiles",
                    "scripts/": "Primitive scripts by category",
                    "artifacts/": ("Execution results (auto-generated)"),
                },
            },
            "global": {
                "path": "~/.config/llm-orc/",
                "description": "User-wide configuration",
                "subdirs": {
                    "ensembles/": "Global ensembles",
                    "profiles/": "Global model profiles",
                    "credentials.yaml": "API keys (encrypted)",
                },
            },
            "priority": "Local config overrides global config",
        }

    def _get_schema_help(self) -> dict[str, Any]:
        """Get YAML schema documentation."""
        return {
            "ensemble": {
                "description": "Multi-agent workflow definition",
                "required_fields": ["name", "agents"],
                "example": {
                    "name": "code-review",
                    "description": "Multi-perspective code review",
                    "agents": [
                        {
                            "name": "security-reviewer",
                            "model_profile": "ollama-llama3",
                            "system_prompt": ("Focus on security issues..."),
                        },
                        {
                            "name": "synthesizer",
                            "model_profile": "ollama-llama3",
                            "depends_on": ["security-reviewer"],
                            "system_prompt": ("Synthesize the analysis..."),
                        },
                    ],
                },
            },
            "profile": {
                "description": "Model configuration shortcut",
                "required_fields": ["provider", "model"],
                "example": {
                    "name": "ollama-llama3",
                    "provider": "ollama",
                    "model": "llama3:latest",
                    "system_prompt": ("You are a helpful assistant."),
                    "timeout_seconds": 60,
                },
                "providers": [
                    "ollama",
                    "anthropic",
                    "anthropic-claude-pro-max",
                ],
            },
            "agent": {
                "description": "Agent within an ensemble",
                "required_fields": ["name", "model_profile"],
                "optional_fields": [
                    "system_prompt",
                    "depends_on",
                    "output_format",
                    "timeout_seconds",
                ],
            },
        }

    def _get_tools_help(self) -> dict[str, str]:
        """Get tool category documentation."""
        return {
            "context_management": (
                "set_project - Set active project directory for all operations"
            ),
            "core_execution": (
                "invoke, list_ensembles, validate_ensemble, "
                "update_ensemble, analyze_execution"
            ),
            "provider_discovery": (
                "get_provider_status (check available models), "
                "check_ensemble_runnable (verify ensemble can run)"
            ),
            "ensemble_crud": "create_ensemble, delete_ensemble",
            "profile_crud": (
                "list_profiles, create_profile, update_profile, delete_profile"
            ),
            "script_management": (
                "list_scripts, get_script, test_script, create_script, delete_script"
            ),
            "library": ("library_browse, library_copy, library_search, library_info"),
            "artifacts": "delete_artifact, cleanup_artifacts",
        }

    def _get_workflow_help(self) -> dict[str, list[str]]:
        """Get common workflow documentation."""
        return {
            "start_session": [
                "1. set_project - Point to project directory (optional)",
                "2. get_provider_status - See available models",
                "3. list_ensembles - Find available ensembles",
            ],
            "discover_and_run": [
                "1. set_project - Set project context (if not done)",
                "2. list_ensembles - Find available ensembles",
                "3. check_ensemble_runnable - Verify it can run",
                "4. invoke - Execute the ensemble",
            ],
            "adapt_from_library": [
                "1. set_project - Set target project",
                "2. library_search - Find relevant ensembles",
                "3. library_copy - Copy to local project",
                "4. update_ensemble - Adapt for local models",
                "5. invoke - Run the adapted ensemble",
            ],
            "create_new_ensemble": [
                "1. set_project - Set project context",
                "2. list_profiles - See available model profiles",
                "3. create_ensemble - Create with agents",
                "4. validate_ensemble - Check configuration",
                "5. invoke - Test execution",
            ],
        }

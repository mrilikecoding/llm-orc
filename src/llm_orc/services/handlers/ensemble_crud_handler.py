"""Ensemble CRUD handler for MCP server."""

from __future__ import annotations

import os
import stat
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.config.ensemble_config import EnsembleLoader
from llm_orc.mcp.project_context import ProjectContext
from llm_orc.mcp.utils import get_agent_attr as _get_agent_attr
from llm_orc.schemas.agent_config import parse_agent_config
from llm_orc.services.handlers.bundle_store import BundleStore
from llm_orc.services.handlers.scope import (
    Scope,
    check_write_target,
    find_in_scope,
    parse_scope,
    resolve_in_dir,
)

_PRESERVED_FIELDS = (
    "name",
    "type",
    "model_profile",
    "ensemble",
    "script",
    "parameters",
    "depends_on",
    "system_prompt",
    "cache",
    "fan_out",
    "input_key",
    "when",
    "on_dependency_failure",
    "output_format",
    "timeout_seconds",
    "input_scope",
    "temperature",
    "max_tokens",
    "options",
    "response_format",
    "fallback_model_profile",
    "model",
    "provider",
    "loop",
    "dispatch",
)


_SUPPORTED_CHANGES = ("add_agents", "remove_agents")


class _EnsembleDumper(yaml.SafeDumper):
    """SafeDumper that writes multi-line strings in block style."""


def _represent_str(dumper: yaml.SafeDumper, value: str) -> yaml.ScalarNode:
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


_EnsembleDumper.add_representer(str, _represent_str)

_COMMENT_NOTE = (
    "Applying this rewrites the file: only a leading block of unindented '#' "
    "lines is kept, every other comment is dropped, and line endings are "
    "normalized. A backup is written unless backup is false."
)


def _dump_ensemble(data: dict[str, Any]) -> str:
    """YAML for ``data`` keeping key order (``sort_keys`` would reorder),
    non-ASCII text as itself and multi-line strings in block style."""
    return yaml.dump(
        data,
        Dumper=_EnsembleDumper,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    )


def _leading_comment_block(text: str) -> str:
    """The run of ``#`` and blank lines before the first key, verbatim; empty
    when it holds no comment."""
    lines: list[str] = []
    for line in text.splitlines(keepends=True):
        if line.strip() and not line.startswith("#"):
            break
        lines.append(line)
    block = "".join(lines)
    return block if "#" in block else ""


def _stage_file(path: Path, text: str) -> Path:
    """Write ``text`` to a temporary file beside ``path``, with ``path``'s
    mode, and return it. The temporary file is removed if staging fails."""
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    staged = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        staged.chmod(stat.S_IMODE(path.stat().st_mode))
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    return staged


def _write_backup(backup_path: Path, text: str, mode: int) -> None:
    """Write ``text`` to ``backup_path`` with the original file's ``mode``.
    The file is created owner-only first, so a 0600 ensemble's content is
    never readable through a looser backup."""
    fd = os.open(backup_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(text)
    backup_path.chmod(mode)


def _write_with_backup(path: Path, text: str, original_text: str, backup: bool) -> bool:
    """Replace ``path`` with ``text`` in one step, so a crash cannot leave a
    truncated file. With ``backup``, ``<name>.bak`` first holds the previous
    content; a failed write removes it again. Returns whether a backup was
    written."""
    staged = _stage_file(path, text)
    backup_path = path.with_name(path.name + ".bak")
    prior_backup = backup_path.read_text() if backup_path.exists() else None
    try:
        if backup:
            _write_backup(backup_path, original_text, stat.S_IMODE(path.stat().st_mode))
        os.replace(staged, path)
    except BaseException:
        staged.unlink(missing_ok=True)
        if backup:
            if prior_backup is None:
                backup_path.unlink(missing_ok=True)
            else:
                backup_path.write_text(prior_backup)
        raise
    return backup


def _check_changes_shape(changes: Any) -> None:
    """Refuse a ``changes`` value of the wrong type before it is iterated."""
    if not isinstance(changes, dict):
        raise ValueError("changes must be an object")
    if "remove_agents" in changes:
        names = changes["remove_agents"]
        if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
            raise ValueError("remove_agents must be a list of names")
    if "add_agents" in changes:
        agents = changes["add_agents"]
        if not isinstance(agents, list) or not all(isinstance(a, dict) for a in agents):
            raise ValueError("add_agents must be a list of objects")


def _load_single_document(text: str, name: str) -> dict[str, Any]:
    """The ensemble file's one YAML mapping; a clear refusal otherwise."""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ValueError(f"Ensemble '{name}' is not a single YAML document") from e
    if not isinstance(data, dict):
        raise ValueError(f"Ensemble '{name}' is not a single YAML document")
    return data


def _apply_changes(
    data: dict[str, Any], changes: dict[str, Any], name: str
) -> dict[str, Any]:
    """Return ``data`` with ``changes`` applied; ``data`` is not mutated.

    Raises:
        ValueError: for an unsupported key, a removal of an agent the
            ensemble does not have, or an added name that already exists.
    """
    unknown = sorted(set(changes) - set(_SUPPORTED_CHANGES))
    if unknown:
        raise ValueError(
            f"Unsupported changes key(s) {unknown}; "
            f"supported keys: {list(_SUPPORTED_CHANGES)}"
        )
    agents: list[dict[str, Any]] = list(data.get("agents") or [])
    for agent_name in changes.get("remove_agents") or []:
        if not any(a.get("name") == agent_name for a in agents):
            raise ValueError(f"Ensemble '{name}' has no agent named '{agent_name}'")
        agents = [a for a in agents if a.get("name") != agent_name]
    for new_agent in changes.get("add_agents") or []:
        if not isinstance(new_agent, dict) or not new_agent.get("name"):
            raise ValueError("Each add_agents entry must be a mapping with a name")
        if any(a.get("name") == new_agent["name"] for a in agents):
            raise ValueError(
                f"Ensemble '{name}' already has an agent named '{new_agent['name']}'"
            )
        agents.append(dict(new_agent))
    return {**data, "agents": agents}


def _link_target_to_show(path: Path) -> str | None:
    """The link's target when it is relative and stays inside the link's
    own directory; None otherwise, so no absolute server path leaks."""
    target = path.readlink()
    if target.is_absolute():
        return None
    if not (path.parent / target).resolve().is_relative_to(path.parent.resolve()):
        return None
    return str(target)


def _refuse_symlink(path: Path, action: str) -> None:
    """Refuse to ``action`` an ensemble that is a symlink.

    A checkout's top-level serving shapes are links into
    ``ensembles/agentic-serving/``; deleting or rewriting one breaks the
    shape the serve dispatches to (#106).
    """
    if not path.is_symlink():
        return
    shown = _link_target_to_show(path)
    points_at = f" to {shown}" if shown else ""
    raise ValueError(
        f"Cannot {action} '{path.name}': it is a symlink{points_at} and "
        "links are managed by the project. Edit or remove the link "
        "target by hand if that is intended."
    )


def _agent_to_dict(agent: Any) -> dict[str, Any]:
    """Convert an AgentConfig object to a plain dict for YAML serialization."""
    agent_dict: dict[str, Any] = {}
    for attr in _PRESERVED_FIELDS:
        val = _get_agent_attr(agent, attr)
        if val is not None:
            if attr == "loop" and hasattr(val, "model_dump"):
                agent_dict[attr] = val.model_dump()
            else:
                agent_dict[attr] = val
    return agent_dict


class EnsembleCrudHandler:
    """Handles ensemble create, delete, update, and analysis operations."""

    def __init__(
        self,
        config_manager: ConfigurationManager,
        ensemble_loader: EnsembleLoader,
        find_ensemble_fn: Callable[[str], Any],
        read_artifact_fn: Callable[..., Any],
        bundle_store: BundleStore | None = None,
    ) -> None:
        """Initialize with dependencies.

        Args:
            config_manager: Configuration manager instance.
            ensemble_loader: Ensemble loader instance.
            find_ensemble_fn: Callback to find ensemble by name.
            read_artifact_fn: Callback to read an artifact resource.
            bundle_store: Bundles, which ``scope: global`` deletes by name
                when no global tier file has it.
        """
        self._config_manager = config_manager
        self._ensemble_loader = ensemble_loader
        self._find_ensemble = find_ensemble_fn
        self._read_artifact = read_artifact_fn
        self._bundles = bundle_store or BundleStore(
            lambda: self._config_manager.global_config_dir
        )

    def set_project_context(self, ctx: ProjectContext) -> None:
        """Update handler to use new project context."""
        self._config_manager = ctx.config_manager

    async def create_ensemble(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Create a new ensemble.

        Args:
            arguments: Tool arguments including name, description, agents.

        Returns:
            Creation result.
        """
        name = arguments.get("name")
        description = arguments.get("description", "")
        agents = arguments.get("agents", [])
        from_template = arguments.get("from_template")

        if not name:
            raise ValueError("name is required")

        scope = parse_scope(arguments)
        local_dir = self._dir_for_scope(scope)
        if local_dir is None:
            global_ensembles = self._config_manager.global_config_dir / "ensembles"
            raise ValueError(
                "No project directory (.llm-orc) here; pass scope: global to "
                f"write under {global_ensembles}, or run llm-orc config init"
            )
        target_file = resolve_in_dir(
            scope_dir=local_dir,
            filename=f"{name}.yaml",
            scope=scope,
            label="Ensemble",
        )
        check_write_target(
            path=target_file, scope_dir=local_dir, scope=scope, label="Ensemble"
        )
        if target_file.exists():
            raise ValueError(f"Ensemble already exists: {name}")

        agents_copied = 0
        if from_template:
            agents, description, agents_copied = self._copy_from_template(
                from_template, description
            )

        normalized_agents: list[dict[str, Any]] = []
        for agent in agents:
            if isinstance(agent, dict):
                # Strip None values so extra="forbid" schemas don't reject
                # keys like model_profile=null on non-LLM agents.
                cleaned = {k: v for k, v in agent.items() if v is not None}
                cfg = parse_agent_config(cleaned)
                normalized_agents.append(cfg.model_dump(exclude_none=True))
            else:
                normalized_agents.append(agent)

        ensemble_data = {
            "name": name,
            "description": description,
            "agents": normalized_agents,
        }
        yaml_content = yaml.dump(ensemble_data, default_flow_style=False)

        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.write_text(yaml_content)

        return {
            "created": True,
            "path": str(target_file),
            "agents_copied": agents_copied,
            "scope": scope,
        }

    async def delete_ensemble(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Delete an ensemble.

        Args:
            arguments: Tool arguments including ensemble_name, confirm.

        Returns:
            Deletion result.
        """
        ensemble_name = arguments.get("ensemble_name")
        confirm = arguments.get("confirm", False)

        if not ensemble_name:
            raise ValueError("ensemble_name is required")

        if not confirm:
            raise ValueError("Confirmation required to delete ensemble")

        scope = parse_scope(arguments)
        if scope == "global" and self._delete_bundle(ensemble_name):
            return {"deleted": True, "bundle": ensemble_name}
        ensemble_file = self._find_in_scope(
            ensemble_name, scope, self._flat_filename(ensemble_name, scope)
        )

        _refuse_symlink(ensemble_file, "delete")
        ensemble_file.unlink()

        return {
            "deleted": True,
            "path": str(ensemble_file),
        }

    def _delete_bundle(self, name: str) -> bool:
        """Remove the bundle ``name`` unless some tier resolves that name,
        in which case the delete means the tier's ensemble. The lookup is
        the one a persist uses to refuse a shadowed name, so a tier root
        in a subdirectory or under another file name counts too."""
        if self._find_ensemble(name) is not None:
            return False
        return self._bundles.delete(name)

    def _flat_filename(self, name: str, scope: Scope) -> str:
        """``<name>.yaml``, or ``<name>.yml`` when that is the file the
        scope's directory holds (the loader reads both)."""
        scope_dir = self._dir_for_scope(scope)
        if scope_dir is None:
            return f"{name}.yaml"
        yml = resolve_in_dir(
            scope_dir=scope_dir, filename=f"{name}.yml", scope=scope, label="Ensemble"
        )
        if yml.exists() and not (scope_dir / f"{name}.yaml").exists():
            return f"{name}.yml"
        return f"{name}.yaml"

    async def update_ensemble(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Update an ensemble configuration.

        Args:
            arguments: Tool arguments including ensemble_name, changes, dry_run.

        Returns:
            Update result.
        """
        ensemble_name = arguments.get("ensemble_name")
        changes = arguments.get("changes", {})
        dry_run = arguments.get("dry_run", True)
        backup = arguments.get("backup", True)

        if not ensemble_name:
            raise ValueError("ensemble_name is required")

        _check_changes_shape(changes)

        scope = parse_scope(arguments)
        ensemble_path = self._find_in_scope(
            ensemble_name, scope, self._flat_filename(ensemble_name, scope)
        )

        _refuse_symlink(ensemble_path, "update")
        original_text = ensemble_path.read_text()
        original = _load_single_document(original_text, ensemble_name)
        updated = _apply_changes(original, changes, ensemble_name)
        self._validate_candidate(updated, ensemble_name)

        if dry_run:
            return {
                "preview": changes,
                "modified": False,
                "backup_created": False,
                "note": _COMMENT_NOTE,
            }

        if updated == original:
            return {
                "modified": False,
                "backup_created": False,
                "changes_applied": {},
            }

        if not os.access(ensemble_path, os.W_OK):
            raise ValueError(f"Ensemble '{ensemble_name}' is not writable")

        backup_created = _write_with_backup(
            ensemble_path,
            _leading_comment_block(original_text) + _dump_ensemble(updated),
            original_text,
            backup,
        )

        return {
            "modified": True,
            "backup_created": backup_created,
            "changes_applied": changes,
        }

    def _validate_candidate(self, data: dict[str, Any], name: str) -> None:
        """Load ``data`` through the real ensemble loader before any write.

        Raises:
            ValueError: when the loader rejects the result, with its reason.
        """
        search_dirs = [str(d) for d in self._config_manager.get_ensembles_dirs()]
        with tempfile.TemporaryDirectory() as tmp:
            candidate = Path(tmp) / "candidate.yaml"
            candidate.write_text(_dump_ensemble(data))
            try:
                EnsembleLoader().load_from_file(str(candidate), search_dirs)
            except KeyError as e:
                raise ValueError(
                    f"Update would produce an invalid ensemble '{name}': "
                    f"missing required key {e}"
                ) from e
            except Exception as e:
                raise ValueError(
                    f"Update would produce an invalid ensemble '{name}': {e}"
                ) from e

    async def analyze_execution(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Analyze an execution artifact.

        Args:
            arguments: Tool arguments including artifact_id.

        Returns:
            Analysis result.
        """
        artifact_id = arguments.get("artifact_id")

        if not artifact_id:
            raise ValueError("artifact_id is required")

        parts = artifact_id.split("/")
        if len(parts) != 2:
            raise ValueError(f"Invalid artifact_id format: {artifact_id}")

        ensemble_name, aid = parts
        artifact = await self._read_artifact(ensemble_name, aid)

        results = artifact.get("results", {})
        success_count = sum(1 for r in results.values() if r.get("status") == "success")

        return {
            "analysis": {
                "total_agents": len(results),
                "successful_agents": success_count,
                "failed_agents": len(results) - success_count,
            },
            "metrics": {
                "agent_success_rate": (
                    success_count / len(results) if results else 0.0
                ),
                "cost": artifact.get("cost", 0),
                "duration": artifact.get("duration", 0),
            },
        }

    def get_local_ensembles_dir(self) -> Path:
        """The project's ensembles directory for writing.

        Raises:
            ValueError: when there is no project tier. Writes never fall
                through to the library or packaged tiers; ``scope: global``
                is the writable home without a project (#196).
        """
        local = self._config_manager.local_config_dir
        if local is None:
            raise ValueError(
                "No project (.llm-orc) directory to write to; "
                "use scope 'global' for the global tier"
            )
        return local / "ensembles"

    def _dir_for_scope(self, scope: Scope) -> Path | None:
        """Write directory for one scope; None when the project has none."""
        if scope == "global":
            return self._config_manager.global_config_dir / "ensembles"
        local_config_dir = self._config_manager.local_config_dir
        if local_config_dir is None:
            return None
        return local_config_dir / "ensembles"

    def _find_in_scope(
        self, ensemble_name: str, scope: Scope, filename: str | None = None
    ) -> Path:
        return find_in_scope(
            name=ensemble_name,
            filename=filename or f"{ensemble_name}.yaml",
            scope=scope,
            scope_dir=self._dir_for_scope(scope),
            search_dirs=[Path(d) for d in self._config_manager.get_ensembles_dirs()],
            classify=self._config_manager.classify_tier,
            label="Ensemble",
        )

    def _copy_from_template(
        self, template_name: str, description: str
    ) -> tuple[list[dict[str, Any]], str, int]:
        """Copy agents and description from a template ensemble.

        Args:
            template_name: Name of the template ensemble.
            description: Current description (may be overwritten if empty).

        Returns:
            Tuple of (agents list, description, agents_copied count).

        Raises:
            ValueError: If template not found.
        """
        template_config = self._find_ensemble(template_name)
        if not template_config:
            raise ValueError(f"Template ensemble not found: {template_name}")

        agents = [
            dict(agent) if isinstance(agent, dict) else _agent_to_dict(agent)
            for agent in template_config.agents
        ]

        final_description = description or template_config.description
        return agents, final_description, len(agents)

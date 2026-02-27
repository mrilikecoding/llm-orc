"""Unit tests for EnsembleCrudHandler covering lines 134-167, 182-197, 250, 266."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml

from llm_orc.services.handlers.ensemble_crud_handler import EnsembleCrudHandler


def _make_handler(
    ensemble_dirs: list[str] | None = None,
    find_ensemble_return: Any = None,
    read_artifact_return: dict[str, Any] | None = None,
) -> EnsembleCrudHandler:
    config_manager = MagicMock()
    config_manager.get_ensembles_dirs.return_value = ensemble_dirs or []

    ensemble_loader = MagicMock()

    async def _read_artifact(ensemble_name: str, aid: str) -> dict[str, Any]:
        return read_artifact_return or {}

    return EnsembleCrudHandler(
        config_manager=config_manager,
        ensemble_loader=ensemble_loader,
        find_ensemble_fn=lambda name: find_ensemble_return,
        read_artifact_fn=_read_artifact,
    )


class TestUpdateEnsemble:
    """Tests for update_ensemble (lines 134-167)."""

    async def test_raises_when_ensemble_name_missing(self) -> None:
        """update_ensemble raises ValueError when ensemble_name is absent."""
        handler = _make_handler()

        with pytest.raises(ValueError, match="ensemble_name is required"):
            await handler.update_ensemble({})

    async def test_raises_when_ensemble_not_found(self, tmp_path: Path) -> None:
        """update_ensemble raises ValueError when no matching file exists."""
        handler = _make_handler(ensemble_dirs=[str(tmp_path)])

        with pytest.raises(ValueError, match="Ensemble not found: missing"):
            await handler.update_ensemble({"ensemble_name": "missing", "changes": {}})

    async def test_dry_run_returns_preview_without_modifying(
        self, tmp_path: Path
    ) -> None:
        """dry_run=True returns preview dict and leaves the file untouched."""
        ensembles_dir = tmp_path / "ensembles"
        ensembles_dir.mkdir()
        ensemble_file = ensembles_dir / "my-ensemble.yaml"
        ensemble_file.write_text(yaml.dump({"name": "my-ensemble", "agents": []}))

        handler = _make_handler(ensemble_dirs=[str(ensembles_dir)])
        changes: dict[str, Any] = {"add_agents": [{"name": "agent-x"}]}

        result = await handler.update_ensemble(
            {
                "ensemble_name": "my-ensemble",
                "changes": changes,
                "dry_run": True,
            }
        )

        assert result["modified"] is False
        assert result["backup_created"] is False
        assert result["preview"] == changes

    async def test_dry_run_true_by_default(self, tmp_path: Path) -> None:
        """dry_run defaults to True when omitted from arguments."""
        ensembles_dir = tmp_path / "ensembles"
        ensembles_dir.mkdir()
        (ensembles_dir / "my-ensemble.yaml").write_text(
            yaml.dump({"name": "my-ensemble", "agents": []})
        )

        handler = _make_handler(ensemble_dirs=[str(ensembles_dir)])

        result = await handler.update_ensemble({"ensemble_name": "my-ensemble"})

        assert result["modified"] is False

    async def test_applies_changes_when_not_dry_run(self, tmp_path: Path) -> None:
        """dry_run=False returns modified=True and includes changes_applied."""
        ensembles_dir = tmp_path / "ensembles"
        ensembles_dir.mkdir()
        ensemble_file = ensembles_dir / "my-ensemble.yaml"
        ensemble_file.write_text(yaml.dump({"name": "my-ensemble", "agents": []}))

        handler = _make_handler(ensemble_dirs=[str(ensembles_dir)])
        changes: dict[str, Any] = {"add_agents": [{"name": "agent-x"}]}

        result = await handler.update_ensemble(
            {
                "ensemble_name": "my-ensemble",
                "changes": changes,
                "dry_run": False,
                "backup": False,
            }
        )

        assert result["modified"] is True
        assert result["changes_applied"] == changes

    async def test_creates_backup_when_backup_true(self, tmp_path: Path) -> None:
        """backup=True writes a .yaml.bak file alongside the original."""
        ensembles_dir = tmp_path / "ensembles"
        ensembles_dir.mkdir()
        ensemble_file = ensembles_dir / "my-ensemble.yaml"
        original_content = yaml.dump({"name": "my-ensemble", "agents": []})
        ensemble_file.write_text(original_content)

        handler = _make_handler(ensemble_dirs=[str(ensembles_dir)])

        result = await handler.update_ensemble(
            {
                "ensemble_name": "my-ensemble",
                "changes": {},
                "dry_run": False,
                "backup": True,
            }
        )

        assert result["backup_created"] is True
        backup_file = ensembles_dir / "my-ensemble.yaml.bak"
        assert backup_file.exists()
        assert backup_file.read_text() == original_content

    async def test_no_backup_when_backup_false(self, tmp_path: Path) -> None:
        """backup=False leaves no .yaml.bak file."""
        ensembles_dir = tmp_path / "ensembles"
        ensembles_dir.mkdir()
        (ensembles_dir / "my-ensemble.yaml").write_text(
            yaml.dump({"name": "my-ensemble", "agents": []})
        )

        handler = _make_handler(ensemble_dirs=[str(ensembles_dir)])

        result = await handler.update_ensemble(
            {
                "ensemble_name": "my-ensemble",
                "changes": {},
                "dry_run": False,
                "backup": False,
            }
        )

        assert result["backup_created"] is False
        backup_file = ensembles_dir / "my-ensemble.yaml.bak"
        assert not backup_file.exists()


class TestAnalyzeExecution:
    """Tests for analyze_execution (lines 182-197)."""

    async def test_raises_when_artifact_id_missing(self) -> None:
        """analyze_execution raises ValueError when artifact_id is absent."""
        handler = _make_handler()

        with pytest.raises(ValueError, match="artifact_id is required"):
            await handler.analyze_execution({})

    async def test_raises_for_invalid_artifact_id_format(self) -> None:
        """analyze_execution raises ValueError for IDs without exactly one slash."""
        handler = _make_handler()

        with pytest.raises(ValueError, match="Invalid artifact_id format"):
            await handler.analyze_execution({"artifact_id": "no-slash-here"})

    async def test_raises_for_too_many_slashes(self) -> None:
        """analyze_execution raises ValueError when ID has more than two parts."""
        handler = _make_handler()

        with pytest.raises(ValueError, match="Invalid artifact_id format"):
            await handler.analyze_execution({"artifact_id": "a/b/c"})

    async def test_returns_analysis_for_all_success(self) -> None:
        """analyze_execution counts successful agents correctly."""
        artifact: dict[str, Any] = {
            "results": {
                "agent-a": {"status": "success"},
                "agent-b": {"status": "success"},
            },
            "cost": 0.05,
            "duration": 10,
        }
        handler = _make_handler(read_artifact_return=artifact)

        result = await handler.analyze_execution({"artifact_id": "my-ensemble/abc123"})

        assert result["analysis"]["total_agents"] == 2
        assert result["analysis"]["successful_agents"] == 2
        assert result["analysis"]["failed_agents"] == 0
        assert result["metrics"]["agent_success_rate"] == 1.0
        assert result["metrics"]["cost"] == 0.05
        assert result["metrics"]["duration"] == 10

    async def test_returns_analysis_with_mixed_results(self) -> None:
        """analyze_execution reports failed agents when status != success."""
        artifact: dict[str, Any] = {
            "results": {
                "agent-a": {"status": "success"},
                "agent-b": {"status": "error"},
                "agent-c": {"status": "error"},
            },
        }
        handler = _make_handler(read_artifact_return=artifact)

        result = await handler.analyze_execution({"artifact_id": "ens/run1"})

        assert result["analysis"]["total_agents"] == 3
        assert result["analysis"]["successful_agents"] == 1
        assert result["analysis"]["failed_agents"] == 2
        assert result["metrics"]["agent_success_rate"] == pytest.approx(1 / 3)

    async def test_returns_zero_rate_for_empty_results(self) -> None:
        """analyze_execution returns 0.0 success rate when results dict is empty."""
        handler = _make_handler(read_artifact_return={"results": {}})

        result = await handler.analyze_execution({"artifact_id": "ens/run1"})

        assert result["analysis"]["total_agents"] == 0
        assert result["metrics"]["agent_success_rate"] == 0.0

    async def test_defaults_cost_and_duration_to_zero(self) -> None:
        """Returns 0 for cost/duration when absent."""
        artifact: dict[str, Any] = {
            "results": {"agent-a": {"status": "success"}},
        }
        handler = _make_handler(read_artifact_return=artifact)

        result = await handler.analyze_execution({"artifact_id": "ens/run1"})

        assert result["metrics"]["cost"] == 0
        assert result["metrics"]["duration"] == 0


class TestCopyFromTemplate:
    """Tests for _copy_from_template (lines 250, 266)."""

    def test_raises_when_template_not_found(self) -> None:
        """_copy_from_template raises ValueError when find_ensemble returns falsy."""
        handler = _make_handler(find_ensemble_return=None)

        with pytest.raises(ValueError, match="Template ensemble not found: ghost"):
            handler._copy_from_template("ghost", "some description")

    def test_copies_agents_from_dict_agents(self) -> None:
        """_copy_from_template copies dict agents directly (line 266 branch)."""
        template = MagicMock()
        template.description = "template desc"
        template.agents = [
            {"name": "agent-a", "model_profile": "gpt-4"},
            {"name": "agent-b", "type": "llm"},
        ]

        handler = _make_handler(find_ensemble_return=template)

        agents, description, count = handler._copy_from_template("tmpl", "")

        assert count == 2
        assert agents[0]["name"] == "agent-a"
        assert agents[1]["name"] == "agent-b"
        assert description == "template desc"

    def test_copies_agents_from_object_agents(self) -> None:
        """_copy_from_template converts object agents to dicts (line 268 branch)."""
        agent_obj = MagicMock()
        agent_obj.name = "agent-x"
        agent_obj.model_profile = "llama3"
        agent_obj.type = None
        agent_obj.script = None
        agent_obj.parameters = None
        agent_obj.depends_on = None
        agent_obj.system_prompt = "be helpful"
        agent_obj.cache = None
        agent_obj.fan_out = None

        template = MagicMock()
        template.description = "obj template"
        template.agents = [agent_obj]

        handler = _make_handler(find_ensemble_return=template)

        agents, description, count = handler._copy_from_template("tmpl", "override")

        assert count == 1
        assert agents[0]["model_profile"] == "llama3"
        assert agents[0]["system_prompt"] == "be helpful"
        assert description == "override"

    def test_description_falls_back_to_template_when_empty(self) -> None:
        """Uses template description when caller passes empty."""
        template = MagicMock()
        template.description = "from template"
        template.agents = []

        handler = _make_handler(find_ensemble_return=template)

        _, description, _ = handler._copy_from_template("tmpl", "")

        assert description == "from template"

    def test_caller_description_takes_priority(self) -> None:
        """_copy_from_template keeps caller description when non-empty."""
        template = MagicMock()
        template.description = "template desc"
        template.agents = []

        handler = _make_handler(find_ensemble_return=template)

        _, description, _ = handler._copy_from_template("tmpl", "my description")

        assert description == "my description"

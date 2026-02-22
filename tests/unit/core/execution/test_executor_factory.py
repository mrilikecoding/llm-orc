"""Tests for ExecutorFactory."""

from pathlib import Path

from llm_orc.core.execution.executor_factory import ExecutorFactory


class TestCreateRootExecutor:
    """Tests for ExecutorFactory.create_root_executor."""

    def test_creates_executor_with_defaults(self) -> None:
        """Root executor creates fresh infrastructure."""
        executor = ExecutorFactory.create_root_executor()

        assert executor._depth == 0
        assert executor._save_artifacts is True
        assert executor._config_manager is not None
        assert executor._credential_storage is not None
        assert executor._model_factory is not None

    def test_accepts_project_dir(self, tmp_path: Path) -> None:
        """Root executor accepts project directory."""
        executor = ExecutorFactory.create_root_executor(project_dir=tmp_path)

        assert executor._project_dir == tmp_path

    def test_save_artifacts_can_be_disabled(self) -> None:
        """Root executor respects save_artifacts flag."""
        executor = ExecutorFactory.create_root_executor(save_artifacts=False)

        assert executor._save_artifacts is False


class TestCreateChildExecutor:
    """Tests for ExecutorFactory.create_child_executor."""

    def test_shares_immutable_infrastructure(self) -> None:
        """Child shares parent's config, credentials, model factory."""
        parent = ExecutorFactory.create_root_executor()

        child = ExecutorFactory.create_child_executor(parent, depth=1)

        assert child._config_manager is parent._config_manager
        assert child._credential_storage is parent._credential_storage
        assert child._model_factory is parent._model_factory

    def test_isolates_mutable_state(self) -> None:
        """Child has its own usage collector."""
        parent = ExecutorFactory.create_root_executor()

        child = ExecutorFactory.create_child_executor(parent, depth=1)

        assert child._usage_collector is not parent._usage_collector

    def test_disables_artifact_saving(self) -> None:
        """Child does not save artifacts."""
        parent = ExecutorFactory.create_root_executor()

        child = ExecutorFactory.create_child_executor(parent, depth=1)

        assert child._save_artifacts is False

    def test_records_depth(self) -> None:
        """Child records its depth."""
        parent = ExecutorFactory.create_root_executor()

        child = ExecutorFactory.create_child_executor(parent, depth=2)

        assert child._depth == 2

    def test_inherits_project_dir(self, tmp_path: Path) -> None:
        """Child inherits parent's project directory."""
        parent = ExecutorFactory.create_root_executor(project_dir=tmp_path)

        child = ExecutorFactory.create_child_executor(parent, depth=1)

        assert child._project_dir == tmp_path

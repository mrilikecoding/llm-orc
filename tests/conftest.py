"""Shared test fixtures and configuration."""

import shutil
from collections.abc import Generator
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from llm_orc.core.execution.artifact_manager import ArtifactManager
from llm_orc.core.execution.ensemble_execution import EnsembleExecutor
from llm_orc.core.execution.executor_factory import ExecutorFactory

# Enable BDD testing with pytest-bdd
pytest_plugins = ["pytest_bdd"]

_project_root = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _isolated_global_config(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Point the global config at a per-test temp dir (issue #86).

    Tests that construct ConfigurationManager without mocking otherwise
    provision the REAL ~/.config/llm-orc — a shared-state race under
    pytest-xdist workers and a mutation of the developer's machine. Tests
    that assert on specific XDG behavior override the env themselves.
    """
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path_factory.mktemp("xdg")))


@pytest.fixture(autouse=True)
def _isolated_state_and_packaged_tier(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No packaged serving tier and a per-test state dir by default (#196).

    An editable install resolves the packaged tier to THIS checkout's
    `.llm-orc/`, so a ConfigurationManager built in a temp cwd would
    otherwise see ~60 model profiles and 100+ ensembles it did not set
    up (the #86 lesson again). Tests that pin the packaged tier opt in
    with the `packaged_serving_project` fixture below or set the env
    themselves. XDG_STATE_HOME keeps artifacts, traces and presets out
    of the developer's ~/.local/state. The LLM_ORC_STATE_DIR override is
    cleared too, so a shell export or `serve --state-dir` cannot leak.
    """
    monkeypatch.setenv("LLM_ORC_SERVING_PROJECT_DIR", "")
    monkeypatch.setenv("LLM_ORC_STATE_DIR", "")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path_factory.mktemp("state")))


_PACKAGED_PROFILE = """\
name: agentic-tier-cheap-general
model: qwen3-8b
provider: llama-server
cost_per_token: 0.0
"""

_PACKAGED_CONFIG = """\
project:
  name: packaged-fixture
serving:
  self_reference: true
model_profiles:
  packaged-orch:
    model: qwen3-0.6b
    provider: llama-server
  shadowed-by-file:
    model: from-config-yaml
    provider: llama-server
agentic_serving:
  orchestrator:
    model_profile: packaged-orch
"""

_PACKAGED_SHADOW_PROFILE = """\
name: shadowed-by-file
model: from-profiles-dir
provider: llama-server
"""

_PACKAGED_SERVING = """\
name: serving
description: fixture serving ensemble (never executed by these tests)
agents:
  - name: classify
    script: scripts/agentic_serving/classify.py
"""

_PACKAGED_CHILD = """\
name: child
description: a packaged child that runs one packaged script
agents:
  - name: echo
    script: scripts/agentic_serving/classify.py
"""

_PACKAGED_SCRIPT = """\
import json, sys
from _helpers import tag
data = sys.stdin.read()
print(json.dumps({"success": True, "data": {"echo": data.strip(), "tag": tag}}))
"""

_PACKAGED_HELPERS = 'tag = "helper"\n'


@pytest.fixture
def packaged_serving_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A minimal serving project on disk, installed as the packaged tier
    through the env override. Returns its root (the dot-dir equivalent)."""
    root = tmp_path / "serving_project"
    (root / "ensembles" / "agentic-serving").mkdir(parents=True)
    (root / "ensembles" / "agentic-serving" / "serving.yaml").write_text(
        _PACKAGED_SERVING
    )
    (root / "ensembles" / "agentic-serving" / "child.yaml").write_text(_PACKAGED_CHILD)
    (root / "profiles").mkdir()
    (root / "profiles" / "agentic-tier-cheap-general.yaml").write_text(
        _PACKAGED_PROFILE
    )
    (root / "profiles" / "shadowed-by-file.yaml").write_text(_PACKAGED_SHADOW_PROFILE)
    (root / "scripts" / "agentic_serving").mkdir(parents=True)
    (root / "scripts" / "agentic_serving" / "classify.py").write_text(_PACKAGED_SCRIPT)
    (root / "scripts" / "agentic_serving" / "_helpers.py").write_text(_PACKAGED_HELPERS)
    (root / "config.yaml").write_text(_PACKAGED_CONFIG)
    monkeypatch.setenv("LLM_ORC_SERVING_PROJECT_DIR", str(root))
    return root


@pytest.fixture(autouse=True)
def _reset_http_connection_pool() -> Generator[None, None, None]:
    """Reset HTTPConnectionPool singleton between tests.

    OrchestraService.__init__ calls HTTPConnectionPool.configure() with the
    result of config_manager.load_performance_config(). When config_manager
    is a Mock, this stores a MagicMock as _performance_config. A later test
    that creates a real ClaudeModel then feeds Mock values into httpx.Limits,
    which fails on internal '<' comparisons.
    """
    from llm_orc.models.base import HTTPConnectionPool

    yield

    # Close the client synchronously before dropping the reference,
    # otherwise AsyncClient.__del__ creates an unawaited coroutine.
    client = HTTPConnectionPool._httpx_client
    if client is not None and hasattr(client, "close"):
        try:
            client.close()
        except Exception:
            pass
    HTTPConnectionPool._httpx_client = None
    HTTPConnectionPool._performance_config = None
    HTTPConnectionPool._instance = None


def _subdirs(path: Path) -> list[Path]:
    """The directories directly under ``path``; empty when ``path`` is
    gone. Every xdist worker shares the real artifacts directory, so
    another worker can remove one between a check and a listing."""
    try:
        return [d for d in path.iterdir() if d.is_dir()]
    except FileNotFoundError:
        return []


@pytest.fixture(autouse=True)
def cleanup_test_artifacts() -> Generator[None, None, None]:
    """Automatically clean up test artifacts after each test.

    This fixture runs automatically for all tests and cleans up any artifacts
    created in .llm-orc/artifacts/ during test execution.
    """
    # Store initial top-level artifact directories before test
    artifacts_path = _project_root / ".llm-orc" / "artifacts"
    initial_dirs = set()
    if artifacts_path.exists():
        initial_dirs = {d.name for d in _subdirs(artifacts_path)}

    # Run the test
    yield

    # Clean up any new top-level artifact directories created during test
    if artifacts_path.exists():
        current_dirs = {d.name for d in _subdirs(artifacts_path)}
        new_dirs = current_dirs - initial_dirs

        # Also check for modified directories (new timestamped subdirs)
        for dir_name in current_dirs:
            dir_path = artifacts_path / dir_name
            if dir_path.is_dir():
                # Check if this directory has new timestamped subdirectories
                if dir_name in initial_dirs:
                    # Check for new subdirectories created during test
                    subdirs = _subdirs(dir_path)
                    # If it has new content, consider it modified
                    if any("202" in subdir.name for subdir in subdirs):
                        new_dirs.add(dir_name)

        # Clean up all new or modified directories
        for dir_name in new_dirs:
            dir_path = artifacts_path / dir_name
            if dir_path.exists():
                shutil.rmtree(dir_path, ignore_errors=True)


@pytest.fixture
def mock_ensemble_executor() -> Generator[EnsembleExecutor, None, None]:
    """Create an EnsembleExecutor with mocked expensive dependencies.

    This fixture ensures that tests don't create real artifacts in .llm-orc/artifacts/
    and mocks only expensive I/O operations while preserving functionality.
    """
    # Mock only the expensive I/O operations during construction
    with patch(
        "llm_orc.core.config.config_manager.ConfigurationManager._setup_default_config"
    ):
        with patch(
            "llm_orc.core.config.config_manager.ConfigurationManager._setup_default_ensembles"
        ):
            with patch(
                "llm_orc.core.config.config_manager.ConfigurationManager._copy_profile_templates"
            ):
                # Create real executor with mocked I/O
                executor = ExecutorFactory.create_root_executor()

                # Mock the ArtifactManager to prevent real artifact creation
                mock_artifact_manager = Mock(spec=ArtifactManager)
                mock_artifact_manager.save_execution_results = Mock()

                # Replace only the artifact manager, keep the rest functional
                with patch.object(executor, "_artifact_manager", mock_artifact_manager):
                    yield executor


@pytest.fixture(autouse=True)
def _no_real_llama_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test may launch the operator's real llama-server (#90).

    A supervisor built on the default binary name is one that reached
    production wiring unpatched; tests that want a process use a stub
    binary path. Found the hard way: two serve tests rendered the real
    preset and started a real router from the suite.
    """
    from llm_orc.providers import llama_server

    original = llama_server.LlamaServerSupervisor.start

    def guarded(self: llama_server.LlamaServerSupervisor, **kwargs: float) -> None:
        if self.binary == "llama-server":
            raise AssertionError(
                "test reached the real llama-server binary; patch "
                "start_router_from_config or pass --no-backend"
            )
        original(self, **kwargs)

    monkeypatch.setattr(llama_server.LlamaServerSupervisor, "start", guarded)

"""#204: what `llm-orc init` writes can run on a fresh host.

Each tier is provisioned on its own, with the packaged serving project
(the repo's .llm-orc, which the wheel ships as serving_project) as the
only other tier. Every ensemble init writes must find every profile it
names, fallback profiles and their chains included, and each profile must
name a provider the registry knows. The global dir is the per-test XDG
dir from tests/conftest.py and the project is a tmp cwd, so nothing
touches the real ~/.config/llm-orc.
"""

from pathlib import Path
from typing import Any

import pytest
import yaml

from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.providers.registry import provider_registry

REPO = Path(__file__).parents[3]
TEMPLATES = REPO / "src" / "llm_orc" / "templates"


@pytest.fixture(autouse=True)
def _real_packaged_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_ORC_SERVING_PROJECT_DIR", str(REPO / ".llm-orc"))


def _named_profiles(ensemble: Path) -> set[str]:
    agents: list[dict[str, Any]] = yaml.safe_load(ensemble.read_text())["agents"]
    names: set[str] = set()
    for agent in agents:
        for key in ("model_profile", "fallback_model_profile"):
            if agent.get(key):
                names.add(agent[key])
    return names


def _problems(ensembles: list[Path], manager: ConfigurationManager) -> dict[str, Any]:
    """Per ensemble: profiles not found, and profiles on a retired provider."""
    problems: dict[str, Any] = {}
    for path in ensembles:
        found: list[str] = []
        queue = sorted(_named_profiles(path))
        seen: set[str] = set()
        while queue:
            name = queue.pop()
            if name in seen:
                continue
            seen.add(name)
            profile = manager.get_model_profile(name)
            if profile is None:
                found.append(f"{name}: not found")
                continue
            if not provider_registry.provider_exists(profile.get("provider", "")):
                found.append(f"{name}: provider {profile.get('provider')!r}")
            if profile.get("fallback_model_profile"):
                queue.append(profile["fallback_model_profile"])
        if found:
            problems[path.name] = sorted(found)
    return problems


def test_fresh_global_provisioning_writes_ensembles_that_resolve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    manager = ConfigurationManager()
    written = sorted((manager.global_config_dir / "ensembles").glob("*.yaml"))

    assert written
    assert _problems(written, ConfigurationManager(provision=False)) == {}


def test_fresh_local_init_writes_ensembles_that_resolve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.chdir(project)
    ConfigurationManager().init_local_config(None)
    # Only the local and packaged tiers from here on: whatever the global
    # tier holds must not be what makes these ensembles resolve.
    no_global = tmp_path / "no-global"
    no_global.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(no_global))
    written = sorted((project / ".llm-orc" / "ensembles").glob("*.yaml"))

    assert written
    assert _problems(written, ConfigurationManager(project, provision=False)) == {}


@pytest.mark.parametrize("template", ["global-config.yaml", "local-config.yaml"])
def test_config_templates_ship_no_model_profiles(template: str) -> None:
    data = yaml.safe_load((TEMPLATES / template).read_text())

    assert data["model_profiles"] == {}


class _Profiles:
    """Stands in for the manager: a fixed profile table."""

    def __init__(self, table: dict[str, dict[str, str]]) -> None:
        self._table = table

    def get_model_profile(self, name: str) -> dict[str, str] | None:
        return self._table.get(name)


def test_the_walk_follows_a_fallback_chain_to_a_missing_last_hop(
    tmp_path: Path,
) -> None:
    ensemble = tmp_path / "chain.yaml"
    ensemble.write_text(
        yaml.safe_dump(
            {
                "name": "chain",
                "description": "d",
                "agents": [{"name": "a", "model_profile": "first"}],
            }
        )
    )
    table = {
        "first": {"provider": "llama-server", "fallback_model_profile": "second"},
        "second": {"provider": "llama-server", "fallback_model_profile": "gone"},
    }

    problems = _problems([ensemble], _Profiles(table))  # type: ignore[arg-type]

    assert problems == {"chain.yaml": ["gone: not found"]}

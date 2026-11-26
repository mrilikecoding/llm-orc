"""#204 finding 1: --config-dir at a global-config-shaped directory.

Everything runs under the autouse XDG isolation in tests/conftest.py and a
tmp_path config dir; nothing touches the real ~/.config/llm-orc.
"""

import logging
from pathlib import Path

import click
import pytest
import yaml
from click.testing import CliRunner

from llm_orc.cli import cli
from llm_orc.cli_commands import _find_ensemble_config, _root_lookup
from llm_orc.cli_completion import complete_ensemble_names
from llm_orc.cli_modules.commands.validation_commands import _resolve_ensemble_dirs
from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.services.orchestra_service import OrchestraService


def _global_shaped_dir(root: Path) -> Path:
    """A dir shaped like ~/.config/llm-orc after setup."""
    root.mkdir()
    (root / "config.yaml").write_text(
        yaml.dump({"model_profiles": {"p": {"model": "m", "provider": "x"}}})
    )
    (root / "credentials.yaml").write_text(
        yaml.dump({"anthropic-api": {"auth_method": "api_key", "api_key": "x"}})
    )
    (root / "profiles").mkdir()
    (root / "profiles" / "fast.yaml").write_text(
        yaml.dump({"name": "fast", "model": "m", "provider": "llama-server"})
    )
    (root / "ensembles").mkdir()
    (root / "ensembles" / "real.yaml").write_text(
        yaml.dump(
            {
                "name": "real",
                "description": "a real ensemble",
                "agents": [{"name": "a", "model_profile": "fast"}],
            }
        )
    )
    return root


def test_config_dir_at_global_dir_lists_only_real_ensembles(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config_dir = _global_shaped_dir(tmp_path / "llm-orc")

    with caplog.at_level(logging.WARNING):
        result = CliRunner().invoke(
            cli, ["list-ensembles", "--config-dir", str(config_dir)]
        )

    assert result.exit_code == 0, result.output
    assert "real: a real ensemble" in result.output
    assert "Skipping invalid ensemble" not in caplog.text, caplog.text


def test_plain_ensembles_dir_is_scanned_as_before(tmp_path: Path) -> None:
    plain = tmp_path / "my-ensembles"
    (plain / "sub").mkdir(parents=True)
    ensemble = {
        "name": "nested",
        "description": "d",
        "agents": [{"name": "a", "model_profile": "p"}],
    }
    (plain / "sub" / "nested.yaml").write_text(yaml.dump(ensemble))

    result = CliRunner().invoke(cli, ["list-ensembles", "--config-dir", str(plain)])

    assert result.exit_code == 0, result.output
    assert "nested: d" in result.output


def test_malformed_ensemble_inside_ensembles_subdir_still_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config_dir = _global_shaped_dir(tmp_path / "llm-orc")
    (config_dir / "ensembles" / "broken.yaml").write_text("name: broken\n")

    with caplog.at_level(logging.WARNING):
        result = CliRunner().invoke(
            cli, ["list-ensembles", "--config-dir", str(config_dir)]
        )

    assert "real: a real ensemble" in result.output
    assert "broken.yaml" in caplog.text


def test_every_user_of_the_flag_reads_the_ensembles_subdir(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config_dir = _global_shaped_dir(tmp_path / "llm-orc")
    caplog.set_level(logging.WARNING)

    completions = complete_ensemble_names(
        click.Context(click.Command("invoke"), info_name="invoke"),
        click.Option(["--x"]),
        "",
    )
    ctx = click.Context(click.Command("invoke"))
    ctx.params["config_dir"] = str(config_dir)
    completed = complete_ensemble_names(ctx, click.Option(["--x"]), "")
    manager = ConfigurationManager(provision=False)
    dirs = _resolve_ensemble_dirs(str(config_dir), manager)
    found = _find_ensemble_config("real", dirs, OrchestraService())

    assert completions is not None
    assert str(config_dir) not in caplog.text, caplog.text
    assert completed == ["real"]
    assert found.name == "real"


def _ensemble_yaml(name: str) -> str:
    return yaml.dump(
        {
            "name": name,
            "description": f"{name} d",
            "agents": [{"name": "a", "model_profile": "p"}],
        }
    )


def test_category_directory_named_ensembles_does_not_hide_siblings(
    tmp_path: Path,
) -> None:
    plain = tmp_path / "mine"
    (plain / "ensembles").mkdir(parents=True)
    (plain / "top.yaml").write_text(_ensemble_yaml("top"))
    (plain / "ensembles" / "inner.yaml").write_text(_ensemble_yaml("inner"))

    result = CliRunner().invoke(cli, ["list-ensembles", "--config-dir", str(plain)])

    assert "top: top d" in result.output
    assert "inner: inner d" in result.output


def test_a_file_named_ensembles_is_not_a_config_root(tmp_path: Path) -> None:
    plain = tmp_path / "mine"
    plain.mkdir()
    (plain / "config.yaml").write_text("model_profiles: {}\n")
    (plain / "ensembles").write_text("not a directory\n")
    (plain / "top.yaml").write_text(_ensemble_yaml("top"))

    result = CliRunner().invoke(cli, ["list-ensembles", "--config-dir", str(plain)])

    assert "top: top d" in result.output


def test_invoke_lookup_reads_the_ensembles_subdir_of_a_config_root(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config_dir = _global_shaped_dir(tmp_path / "llm-orc")
    caplog.set_level(logging.WARNING)

    found = _root_lookup(OrchestraService(), "real", str(config_dir))("real")

    assert found.name == "real"
    assert "Skipping invalid ensemble" not in caplog.text, caplog.text

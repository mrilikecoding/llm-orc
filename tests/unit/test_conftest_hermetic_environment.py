"""The suite's outcome must not depend on the checkout it runs from (#165).

A developer's working checkout has things a clean worktree or CI does not:
untracked ``.llm-orc/profiles/*.local.yaml`` overrides, a populated
``llm-orchestra-library`` submodule, a llama-server router listening on
:8080. These tests pin the autouse isolation in ``tests/conftest.py``.
"""

import os
import socket
from pathlib import Path
from urllib.parse import urlparse

from llm_orc.core.config.config_manager import ConfigurationManager

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_default_cwd_is_outside_the_checkout() -> None:
    cwd = Path.cwd().resolve()

    assert cwd != _REPO_ROOT
    assert _REPO_ROOT not in cwd.parents


def test_default_project_tier_is_not_the_checkouts_dot_dir() -> None:
    """The checkout's .llm-orc/profiles holds tracked profiles and, in a
    working copy, untracked *.local.yaml overrides. Both load through the
    cwd's project tier, so a default manager must not see either."""
    assert (_REPO_ROOT / ".llm-orc" / "profiles" / "local-qwen3-8b.yaml").exists()

    profiles = ConfigurationManager().get_model_profiles()

    assert "local-qwen3-8b" not in profiles


def test_llama_server_url_is_not_the_live_router_port() -> None:
    """Provider detection probes LLAMA_SERVER_URL, default :8080; a router
    really listening there turns detection tests red."""
    parsed = urlparse(os.environ.get("LLAMA_SERVER_URL", ""))

    assert parsed.hostname
    assert parsed.port
    assert parsed.port != 8080
    with socket.socket() as probe:
        probe.settimeout(1)
        assert probe.connect_ex((parsed.hostname, parsed.port)) != 0

"""Runtime state: where the serve writes (#196).

Artifacts, the turn trace, the script cache and the rendered router
preset are state, not configuration, and the packaged serving project is
read-only. They land, in order of precedence:

1. ``LLM_ORC_STATE_DIR`` (``llm-orc serve --state-dir`` sets it);
2. the project's ``.llm-orc/`` when there is one, today's layout, so
   instruments reading ``.llm-orc/.serve-trace/`` and
   ``.llm-orc/artifacts/`` in a checkout are unchanged;
3. ``$XDG_STATE_HOME/llm-orc``, default ``~/.local/state/llm-orc``.

Every writer and every reader of these files resolves the directory
through :func:`resolve_state_dir`; none derives it from cwd.
"""

from __future__ import annotations

import os
from pathlib import Path

STATE_DIR_ENV = "LLM_ORC_STATE_DIR"
ARTIFACTS_DIRNAME = "artifacts"
TRACE_DIRNAME = ".serve-trace"
CACHE_DIRNAME = "cache"
PYCACHE_DIRNAME = "pycache"


def resolve_state_dir(local_config_dir: Path | None) -> Path:
    """The directory runtime state lives in (not created here)."""
    override = os.environ.get(STATE_DIR_ENV)
    if override:
        return Path(override)
    if local_config_dir is not None:
        return local_config_dir
    xdg_state_home = os.environ.get("XDG_STATE_HOME")
    base = Path(xdg_state_home) if xdg_state_home else Path.home() / ".local" / "state"
    return base / "llm-orc"

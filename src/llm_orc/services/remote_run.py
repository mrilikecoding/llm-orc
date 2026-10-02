"""Run a local root on another serve (Arc 5, Task 8).

One function, ``run_remote``, does a remote run for the CLI and the MCP
tool: it resolves the remote, ships the closure as one run request,
posts it and checks the answer. An answer counts as a result only when
it is HTTP 200, a JSON object, ``status`` of ``success`` or ``error``
and a boolean ``has_errors``. An unknown API path answers the web UI's
page with 200, and an older serve answers 422 to a key it forbids, so
anything else raises ``RemoteRunError`` naming the remote.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import requests

from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.config.ensemble_config import EnsembleConfig
from llm_orc.core.config.remotes import RemoteError, resolve_remote
from llm_orc.core.execution.scripting.user_input_handler import (
    ScriptUserInputHandler,
)
from llm_orc.services.closure_shipper import ShipError, ship_closure

CONNECT_TIMEOUT_S = 10
EXECUTE_PATH = "/api/ensembles/execute"
_SHOWN_BODY_CHARS = 200


class RemoteRunError(RuntimeError):
    """A remote run that did not produce a result document.

    ``remote`` is the value the caller gave, ``status_code`` the HTTP
    status when one was seen, and ``detail`` what was observed.
    """

    def __init__(
        self, remote: str, detail: str, status_code: int | None = None
    ) -> None:
        self.remote = remote
        self.status_code = status_code
        self.detail = detail
        code = f" ({status_code})" if status_code is not None else ""
        super().__init__(f"Remote '{remote}'{code}: {detail}")


def post_run(url: str, request: Mapping[str, Any]) -> Any:
    """The transport seam: POST the request, connect timeout only, no
    retry. Tests replace this function and nothing else."""
    return requests.post(url, json=request, timeout=(CONNECT_TIMEOUT_S, None))


def run_remote(
    root_name: str,
    remote: str,
    *,
    find_root: Callable[[str], EnsembleConfig | None],
    config_manager: ConfigurationManager,
    project_dir: Path | None,
    input_text: str,
    with_profiles: Sequence[str] = (),
    bind: Mapping[str, str] | None = None,
    pull: bool = False,
    persist: str | None = None,
) -> dict[str, Any]:
    """Run ``root_name`` and its closure on ``remote`` (a configured name
    or a URL) and return the remote's result document.

    Raises:
        RemoteRunError: the remote is unknown, the closure cannot be
            shipped or holds an interactive script (nothing is sent), or
            the answer is not a result document.
    """
    try:
        base_url = resolve_remote(remote, config_manager)
    except RemoteError as e:
        raise RemoteRunError(remote, str(e)) from e
    request = _ship(
        remote,
        root_name,
        find_root=find_root,
        config_manager=config_manager,
        project_dir=project_dir,
        with_profiles=with_profiles,
        bind=bind,
        pull=pull,
        persist=persist,
        input_text=input_text,
    )
    _refuse_interactive(remote, request)
    try:
        response = post_run(base_url + EXECUTE_PATH, request)
    except requests.RequestException as e:
        raise RemoteRunError(remote, f"could not reach {base_url}: {e}") from e
    return _result_document(remote, response)


def _ship(remote: str, root_name: str, **more: Any) -> dict[str, Any]:
    try:
        return ship_closure(root_name, **more)
    except ShipError as e:
        raise RemoteRunError(remote, f"cannot ship {root_name!r}: {e}") from e


def _refuse_interactive(remote: str, request: Mapping[str, Any]) -> None:
    """No channel carries a prompt to the caller, so a closure with an
    interactive script is not sent. The test is the one a local run
    applies: the script reference each agent names."""
    handler = ScriptUserInputHandler()
    ensembles = [request["ensemble"], *request["ensembles"].values()]
    refs = [
        agent["script"]
        for ensemble in ensembles
        for agent in ensemble.get("agents") or []
        if isinstance(agent, dict) and isinstance(agent.get("script"), str)
    ]
    if any(handler.requires_user_input(ref) for ref in refs):
        raise RemoteRunError(
            remote,
            "the closure has an interactive script and no channel carries "
            "its prompt to a remote run; nothing was sent",
        )


def _result_document(remote: str, response: Any) -> dict[str, Any]:
    status_code = int(response.status_code)
    if status_code != 200:
        raise RemoteRunError(remote, _observed(response), status_code)
    try:
        document = response.json()
    except ValueError as e:
        raise RemoteRunError(
            remote,
            f"answered 200 with a body that is not JSON: {_excerpt(response)}",
            status_code,
        ) from e
    if (
        not isinstance(document, dict)
        or document.get("status") not in ("success", "error")
        or not isinstance(document.get("has_errors"), bool)
    ):
        raise RemoteRunError(
            remote,
            "answered 200 with a body that is not a result document "
            f"(is this an llm-orc serve?): {_excerpt(response)}",
            status_code,
        )
    return dict(document)


def _observed(response: Any) -> str:
    """What a non-200 answer said: a 422's ``detail`` as the serve wrote
    it, else the start of the body."""
    try:
        body = response.json()
    except ValueError:
        return _excerpt(response)
    if isinstance(body, dict) and "detail" in body:
        return f"the serve refused the request: {body['detail']}"
    return _excerpt(response)


def _excerpt(response: Any) -> str:
    text = " ".join(str(response.text).split())
    return text[:_SHOWN_BODY_CHARS] or "(empty body)"

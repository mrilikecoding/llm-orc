"""Run a local root on another serve (Arc 5, Task 8).

One coroutine, ``run_remote``, does a remote run for the CLI and the MCP
tool: it resolves the remote, ships the closure as one run request (in a
worker thread, so the event loop stays free), posts it with an
``httpx.AsyncClient`` and checks the answer. The POST
sits inside the client's ``async with``, so cancelling the awaiting task
closes the connection and the remote serve sees the disconnect. An
answer counts as a result only when it is HTTP 200, a JSON object,
``status`` of ``success`` or ``error`` and a boolean ``has_errors``. An
unknown API path answers the web UI's page with 200, and an older serve
answers 422 to a key it forbids, so anything else raises
``RemoteRunError`` naming the remote.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx

from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.config.ensemble_config import EnsembleConfig
from llm_orc.core.config.remotes import RemoteError, resolve_remote
from llm_orc.core.execution.scripting.user_input_handler import (
    ScriptUserInputHandler,
)
from llm_orc.services.closure_shipper import (
    LeftOut,
    ShipError,
    ship_closure_reporting,
)
from llm_orc.services.handlers.run_preparation import INVALID_REQUEST, REMOTE_ERROR

CONNECT_TIMEOUT_S = 10
# The transport seam: None in production (the network). A test sets it to
# an ``httpx.AsyncBaseTransport`` and the real client code runs on it.
transport: httpx.AsyncBaseTransport | None = None
EXECUTE_PATH = "/api/ensembles/execute"
_SHOWN_BODY_CHARS = 200
_CLIENT_ENVIRONMENT = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
)


class RemoteRunError(RuntimeError):
    """A remote run that did not produce a result document.

    ``remote`` is the value the caller gave, ``status_code`` the HTTP
    status when one was seen, and ``detail`` what was observed. ``kind``
    tells the caller what to do: ``invalid_request`` when nothing was
    sent (fix the request), ``remote_error`` when something was sent or
    tried (look at the remote).
    """

    def __init__(
        self,
        remote: str,
        detail: str,
        status_code: int | None = None,
        kind: str = REMOTE_ERROR,
    ) -> None:
        self.remote = remote
        self.kind = kind
        self.status_code = status_code
        self.detail = detail
        code = f" ({status_code})" if status_code is not None else ""
        super().__init__(f"Remote '{remote}'{code}: {detail}")


def build_client(remote: str) -> httpx.AsyncClient:
    """The client a run posts with: connect timeout only, no read timeout,
    no redirect followed, the environment's proxy and TLS settings kept.

    Building it reads those settings, and a bad one raises (a SOCKS proxy
    without its package, a proxy URL of the wrong scheme, a CA file that is
    not there). Nothing has been sent then, so any failure becomes
    ``RemoteRunError`` of kind ``invalid_request``.
    """
    try:
        return httpx.AsyncClient(
            transport=transport,
            timeout=httpx.Timeout(None, connect=CONNECT_TIMEOUT_S),
            follow_redirects=False,
            trust_env=True,
        )
    except Exception as e:
        raise RemoteRunError(
            remote,
            f"the HTTP client could not be set up ({type(e).__name__}: "
            f"{str(e) or 'no detail'}); check the proxy and TLS variables "
            f"({', '.join(_CLIENT_ENVIRONMENT)}); nothing was sent",
            kind=INVALID_REQUEST,
        ) from e


async def post_run(
    client: httpx.AsyncClient, url: str, request: Mapping[str, Any]
) -> httpx.Response:
    """POST the request once on ``client``, which is closed afterwards.
    The POST sits inside the client's ``async with``, so cancelling the
    awaiting task closes the connection."""
    async with client:
        return await client.post(url, json=request)


async def run_remote(
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
    on_left_out: Callable[[list[LeftOut]], None] | None = None,
) -> dict[str, Any]:
    """Run ``root_name`` and its closure on ``remote`` (a configured name
    or a URL) and return the remote's result document.

    ``on_left_out`` is called once with what the closure left to the
    remote (ruling 6), only when that is not empty, before the POST.

    Raises:
        RemoteRunError: the remote is unknown, the closure cannot be
            shipped or holds an interactive script (nothing is sent), or
            the answer is not a result document. ``kind`` is
            ``invalid_request`` for everything refused before a send,
            including a client that could not be built.
    """
    try:
        base_url = resolve_remote(remote, config_manager)
    except RemoteError as e:
        raise RemoteRunError(remote, str(e), kind=INVALID_REQUEST) from e
    request, left_out = await asyncio.to_thread(
        _ship_checked,
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
    if left_out and on_left_out is not None:
        on_left_out(left_out)
    client = build_client(remote)
    try:
        response = await post_run(client, base_url + EXECUTE_PATH, request)
    except httpx.InvalidURL as e:
        raise RemoteRunError(
            remote,
            f"{base_url!r} is not a usable URL: {e}; nothing was sent",
            kind=INVALID_REQUEST,
        ) from e
    except (httpx.HTTPError, ExceptionGroup) as e:
        raise RemoteRunError(
            remote, f"could not reach {base_url}: {_first_leaf(e)}"
        ) from e
    return _result_document(remote, response)


def _first_leaf(error: BaseException) -> str:
    """The message of the first exception inside any exception groups."""
    while isinstance(error, BaseExceptionGroup):
        error = error.exceptions[0]
    return str(error) or type(error).__name__


def _ship_checked(
    remote: str, root_name: str, **more: Any
) -> tuple[dict[str, Any], list[LeftOut]]:
    """The ship half of a run: the closure walk, the proof and the file
    reads block, so this runs in a worker thread, not on the event loop."""
    request, left_out = _ship(remote, root_name, **more)
    _refuse_interactive(remote, request)
    return request, left_out


def _ship(
    remote: str, root_name: str, **more: Any
) -> tuple[dict[str, Any], list[LeftOut]]:
    try:
        return ship_closure_reporting(root_name, **more)
    except ShipError as e:
        raise RemoteRunError(
            remote, f"cannot ship {root_name!r}: {e}", kind=INVALID_REQUEST
        ) from e


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
            kind=INVALID_REQUEST,
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
    """What a non-200 answer said: a redirect's ``Location`` (it is not
    followed, so the caller needs it to fix the URL; capped like the body
    and without unprintable characters), a 422's ``detail``
    as the serve wrote it, else the start of the body."""
    location = _location(response)
    if location is not None:
        return f"redirected to {location} (not followed): {_excerpt(response)}"
    try:
        body = response.json()
    except ValueError:
        return _excerpt(response)
    if isinstance(body, dict) and "detail" in body:
        return f"the serve refused the request: {body['detail']}"
    return _excerpt(response)


def _location(response: Any) -> str | None:
    if not 300 <= int(response.status_code) < 400:
        return None
    value = str(response.headers.get("location") or "")
    shown = "".join(c for c in value if c.isprintable())[:_SHOWN_BODY_CHARS]
    return shown or None


def _excerpt(response: Any) -> str:
    text = " ".join(str(response.text).split())
    return text[:_SHOWN_BODY_CHARS] or "(empty body)"

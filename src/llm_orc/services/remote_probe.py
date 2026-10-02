"""List the configured remotes and probe each one (Arc 6, part 1).

``probe_remotes`` reads the named remotes from the global config and asks
each ``GET <url>/health``, concurrently, with the client a run is posted
with (``build_client``), so a proxy or TLS problem shows here first. A row
is reachable only when the answer is HTTP 200 and a JSON object with a
``version``; anything else is a row with the error observed. One bad
remote never fails the list.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.config.remotes import RemoteError, resolve_remote
from llm_orc.services.remote_run import (
    RemoteRunError,
    _first_leaf,
    _shown,
    build_client,
)

HEALTH_PATH = "/health"
PROBE_TIMEOUT_S = 5


async def probe_remotes(config_manager: ConfigurationManager) -> list[dict[str, Any]]:
    """One row per configured remote: ``name``, ``url`` and ``reachable``,
    with ``version`` when it is, ``error`` when it is not.

    Raises:
        RemoteError: the global config's ``remotes`` is malformed.
    """
    remotes = config_manager.remotes()
    rows = await asyncio.gather(
        *(_probe(name, url, config_manager) for name, url in remotes.items())
    )
    return list(rows)


async def _probe(
    name: str, url: str, config_manager: ConfigurationManager
) -> dict[str, Any]:
    row: dict[str, Any] = {"name": name, "url": url, "reachable": False}
    try:
        base_url = resolve_remote(name, config_manager)
        client = build_client(name, connect_s=PROBE_TIMEOUT_S, read_s=PROBE_TIMEOUT_S)
    except RemoteError as e:
        return {**row, "error": _shown(e)}
    except RemoteRunError as e:
        return {**row, "error": _shown(e.detail)}
    try:
        async with client:
            response = await client.get(base_url + HEALTH_PATH)
    except httpx.InvalidURL as e:
        return {**row, "error": _shown(f"not a usable URL: {e}")}
    except (httpx.HTTPError, ExceptionGroup) as e:
        return {**row, "error": _shown(_first_leaf(e))}
    return _health_row(row, response)


def _health_row(row: dict[str, Any], response: httpx.Response) -> dict[str, Any]:
    if response.status_code != 200:
        return {**row, "error": f"HTTP {response.status_code}"}
    try:
        body = response.json()
    except ValueError:
        body = None
    version = body.get("version") if isinstance(body, dict) else None
    if not isinstance(version, str) or not version:
        return {
            **row,
            "error": "answered 200, but not with an llm-orc health document",
        }
    return {**row, "reachable": True, "version": _shown(version)}

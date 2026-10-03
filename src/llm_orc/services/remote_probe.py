"""List the configured remotes and probe each one (Arc 6, part 1).

``probe_remotes`` reads the named remotes from the global config and asks
each ``GET <url>/health``, concurrently, with the client a run is posted
with (``build_client``), so a proxy or TLS problem shows here first. A row
is reachable only when the answer is HTTP 200 and a JSON object with a
``version``; anything else is a row with the error observed. A probe has a
total deadline and reads at most ``HEALTH_BODY_CAP`` bytes, and any
``Exception`` it raises becomes that remote's row, so one bad remote never
fails the list.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx

from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.config.remotes import RemoteError, mask_password, resolve_remote
from llm_orc.services.remote_run import (
    RemoteRunError,
    _first_leaf,
    _shown,
    build_client,
)

HEALTH_PATH = "/health"
PROBE_TIMEOUT_S = 5
PROBE_DEADLINE_S = 10.0
HEALTH_BODY_CAP = 64 * 1024


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
    row: dict[str, Any] = {
        "name": name,
        "url": mask_password(url),
        "reachable": False,
    }
    try:
        return await asyncio.wait_for(
            _resolved_probe(row, name, config_manager), PROBE_DEADLINE_S
        )
    except TimeoutError:
        return {**row, "error": f"no answer within {PROBE_DEADLINE_S:g} s"}
    except Exception as e:
        return {**row, "error": _shown(_first_leaf(e))}


async def _resolved_probe(
    row: dict[str, Any], name: str, config_manager: ConfigurationManager
) -> dict[str, Any]:
    try:
        base_url = resolve_remote(name, config_manager)
        client = build_client(name, connect_s=PROBE_TIMEOUT_S, read_s=PROBE_TIMEOUT_S)
    except RemoteError as e:
        return {**row, "error": _shown(e)}
    except RemoteRunError as e:
        return {**row, "error": _shown(e.detail)}
    return await _ask(client, base_url, row)


async def _ask(
    client: httpx.AsyncClient, base_url: str, row: dict[str, Any]
) -> dict[str, Any]:
    try:
        async with client, client.stream("GET", base_url + HEALTH_PATH) as response:
            if response.status_code != 200:
                return {**row, "error": f"HTTP {response.status_code}"}
            body = await _capped_body(response)
    except httpx.InvalidURL as e:
        return {**row, "error": _shown(f"not a usable URL: {e}")}
    except (httpx.HTTPError, ExceptionGroup) as e:
        return {**row, "error": _shown(_first_leaf(e))}
    return _health_row(row, body)


async def _capped_body(response: httpx.Response) -> bytes | None:
    """The body, or None when it is longer than ``HEALTH_BODY_CAP``."""
    body = b""
    async for chunk in response.aiter_bytes():
        body += chunk
        if len(body) > HEALTH_BODY_CAP:
            return None
    return body


def _health_row(row: dict[str, Any], body: bytes | None) -> dict[str, Any]:
    document = _json_object(body)
    version = document.get("version")
    if not isinstance(version, str) or not version:
        return {
            **row,
            "error": "answered 200, but not with an llm-orc health document",
        }
    return {**row, "reachable": True, "version": _shown(version)}


def _json_object(body: bytes | None) -> dict[str, Any]:
    """The JSON object in ``body``, or ``{}`` when it is not one (or is
    too deeply nested to read)."""
    if body is None:
        return {}
    try:
        document = json.loads(body)
    except (ValueError, RecursionError):
        return {}
    return document if isinstance(document, dict) else {}

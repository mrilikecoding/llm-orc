#!/usr/bin/env python3
"""Web-searcher script-agent for the `web-searcher` capability ensemble.

Per ADR-020: the `tool_use` Topaz slot is satisfied by a script-agent
ensemble that wraps an external web-search API. Backend selection is
operator-configurable via environment variables; the script ships with
three adapters out of the box:

- ``kagi`` — Kagi Search API (paid; reads ``KAGI_API_TOKEN``).
- ``tavily`` — Tavily Search API (paid; reads ``WEB_SEARCH_API_KEY``).
- ``ddgs`` — DuckDuckGo via the ``ddgs`` package (no key required).

Backend selection rules:

- Default backend is ``ddgs`` — no key required, no setup, works on
  fresh-clone deployments without authentication friction.
- Paid backends (``kagi``, ``tavily``) are explicit opt-in via
  ``WEB_SEARCH_BACKEND=<name>``. If the named backend requires a key
  and the key is unset, ``authentication_failed`` is emitted.

The explicit-opt-in design avoids the trap of auto-preferring a paid
backend whose key is set but not actually authorized for the relevant
API surface — operators choosing a paid backend make the choice
deliberately rather than inheriting it from environment-variable
presence.

The script reads a JSON input from stdin shaped roughly like
``{"query": "..."}`` (the orchestrator's dispatch payload), calls the
selected backend, and writes a structured JSON result to stdout. On
backend failure (authentication, rate limit, network) it writes a
structured error object the orchestrator's reasoning surface can act
on rather than raising.

Operators extend the script with additional backend adapters by adding
a function ``_search_<backend>(query, api_key) -> dict`` and registering
it in ``BACKEND_ADAPTERS``.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any

DEFAULT_RESULT_COUNT = 5
DEFAULT_TIMEOUT_SECONDS = 30

KAGI_TOKEN_ENV = "KAGI_API_TOKEN"
TAVILY_KEY_ENV = "WEB_SEARCH_API_KEY"


def _read_input() -> dict[str, Any]:
    """Read the dispatch JSON payload from stdin."""
    if sys.stdin.isatty():
        return {}
    try:
        return json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        return {}


def _extract_query(payload: dict[str, Any]) -> str:
    """Pull the query string from the dispatch payload.

    The orchestrator's script-agent dispatch convention nests the
    user-supplied parameters under ``parameters``. Accept either the
    nested shape (``{"parameters": {"query": "..."}}``) or a flat
    ``{"query": "..."}`` for ergonomics.
    """
    if "query" in payload and isinstance(payload["query"], str):
        return payload["query"]
    parameters = payload.get("parameters") or {}
    if isinstance(parameters, dict) and isinstance(parameters.get("query"), str):
        return parameters["query"]
    # Fallback — some dispatch shapes pass the prompt as `input` or `data`.
    if isinstance(payload.get("input"), str):
        return payload["input"]
    if isinstance(payload.get("data"), str):
        return payload["data"]
    return ""


def _emit_error(error: str, backend: str, detail: str = "") -> None:
    """Emit a structured error object the orchestrator can route on."""
    payload: dict[str, Any] = {"error": error, "backend": backend}
    if detail:
        payload["detail"] = detail
    print(json.dumps(payload))


def _search_kagi(query: str, api_key: str) -> dict[str, Any]:
    """Kagi Search API adapter.

    https://help.kagi.com/kagi/api/search.html — GET with
    ``Authorization: Bot <token>`` header. Response has a ``data``
    array where ``t == 0`` entries are search results and ``t == 1``
    entries are related searches (ignored here).
    """
    params = urllib.parse.urlencode({"q": query, "limit": DEFAULT_RESULT_COUNT})
    request = urllib.request.Request(
        url=f"https://kagi.com/api/v0/search?{params}",
        headers={"Authorization": f"Bot {api_key}"},
        method="GET",
    )

    with urllib.request.urlopen(
        request, timeout=DEFAULT_TIMEOUT_SECONDS
    ) as response:
        response_data: dict[str, Any] = json.loads(response.read())

    raw_results = response_data.get("data") or []
    results: list[dict[str, str]] = []
    for entry in raw_results:
        if not isinstance(entry, dict):
            continue
        if entry.get("t") != 0:
            continue
        results.append(
            {
                "title": str(entry.get("title", "")),
                "url": str(entry.get("url", "")),
                "snippet": str(entry.get("snippet", "")),
            }
        )
        if len(results) >= DEFAULT_RESULT_COUNT:
            break

    return {
        "backend": "kagi",
        "query": query,
        "result_count": len(results),
        "results": results,
    }


def _search_tavily(query: str, api_key: str) -> dict[str, Any]:
    """Tavily Search API adapter.

    https://docs.tavily.com — JSON request, JSON response, designed
    for LLM consumption. Returns top-N URLs with snippets.
    """
    request_body = json.dumps(
        {
            "api_key": api_key,
            "query": query,
            "max_results": DEFAULT_RESULT_COUNT,
            "include_answer": False,
            "include_raw_content": False,
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        url="https://api.tavily.com/search",
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(
        request, timeout=DEFAULT_TIMEOUT_SECONDS
    ) as response:
        response_data: dict[str, Any] = json.loads(response.read())

    raw_results = response_data.get("results") or []
    results: list[dict[str, str]] = []
    for entry in raw_results[:DEFAULT_RESULT_COUNT]:
        if not isinstance(entry, dict):
            continue
        results.append(
            {
                "title": str(entry.get("title", "")),
                "url": str(entry.get("url", "")),
                "snippet": str(entry.get("content", "")),
            }
        )

    return {
        "backend": "tavily",
        "query": query,
        "result_count": len(results),
        "results": results,
    }


def _search_ddgs(query: str, api_key: str) -> dict[str, Any]:
    """DuckDuckGo Search adapter via the ``ddgs`` package.

    No API key required. The ``api_key`` parameter is unused; the
    signature is shared with other adapters for registry consistency.
    """
    from ddgs import DDGS

    raw_results = DDGS().text(query, max_results=DEFAULT_RESULT_COUNT)

    results: list[dict[str, str]] = []
    for entry in raw_results:
        if not isinstance(entry, dict):
            continue
        results.append(
            {
                "title": str(entry.get("title", "")),
                "url": str(entry.get("href", "")),
                "snippet": str(entry.get("body", "")),
            }
        )

    return {
        "backend": "ddgs",
        "query": query,
        "result_count": len(results),
        "results": results,
    }


# Per-backend specs: adapter function + env var for the API key (if any)
# + whether the key is required. ``ddgs`` is the no-key fallback.
_AdapterFn = Callable[[str, str], dict[str, Any]]
BACKEND_ADAPTERS: dict[str, dict[str, Any]] = {
    "kagi": {
        "adapter": _search_kagi,
        "key_env": KAGI_TOKEN_ENV,
        "requires_key": True,
    },
    "tavily": {
        "adapter": _search_tavily,
        "key_env": TAVILY_KEY_ENV,
        "requires_key": True,
    },
    "ddgs": {
        "adapter": _search_ddgs,
        "key_env": None,
        "requires_key": False,
    },
}

# Default backend when WEB_SEARCH_BACKEND is not explicitly set.
# ddgs requires no key and works on fresh-clone deployments. Operators
# opt into paid backends (kagi, tavily) via WEB_SEARCH_BACKEND=<name>.
DEFAULT_BACKEND = "ddgs"


def _resolve_api_key(spec: dict[str, Any], backend: str) -> str | None:
    """Return the API key for a backend, or None if a required key is missing.

    Emits ``authentication_failed`` and returns None when the backend
    requires a key but the env var is unset. Returns empty string for
    no-key backends (caller passes through to the adapter).
    """
    if not spec["requires_key"]:
        return ""
    key_env = spec["key_env"]
    api_key = (
        os.environ.get(key_env, "").strip() if isinstance(key_env, str) else ""
    )
    if not api_key:
        _emit_error(
            error="authentication_failed",
            backend=backend,
            detail=(
                f"{key_env} environment variable is empty or unset. "
                f"Obtain an API key for {backend} and export it, "
                "or unset WEB_SEARCH_BACKEND to fall back to a "
                "no-key backend (ddgs)."
            ),
        )
        return None
    return api_key


def _dispatch_adapter(
    adapter: _AdapterFn, query: str, api_key: str, backend: str
) -> dict[str, Any] | None:
    """Run the adapter with consistent exception → emitted-error mapping.

    Returns the adapter's result dict on success, or None when an
    error was emitted (caller emits nothing further).
    """
    try:
        return adapter(query, api_key)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            error = "authentication_failed"
        elif exc.code == 429:
            error = "rate_limited"
        else:
            error = "backend_http_error"
        _emit_error(error, backend, f"HTTP {exc.code}: {exc.reason}")
    except urllib.error.URLError as exc:
        _emit_error("backend_unavailable", backend, str(exc.reason))
    except (TimeoutError, OSError) as exc:
        _emit_error("backend_unavailable", backend, f"{type(exc).__name__}: {exc}")
    except json.JSONDecodeError as exc:
        _emit_error("backend_invalid_response", backend, f"JSON decode failure: {exc}")
    return None


def main() -> int:
    explicit_backend = (os.environ.get("WEB_SEARCH_BACKEND") or "").strip().lower()
    backend = explicit_backend or DEFAULT_BACKEND

    spec = BACKEND_ADAPTERS.get(backend)
    if spec is None:
        _emit_error(
            error="unsupported_backend",
            backend=backend,
            detail=(
                f"Backend '{backend}' has no adapter. "
                f"Available backends: {sorted(BACKEND_ADAPTERS)}. "
                "Set WEB_SEARCH_BACKEND to an available backend or author "
                "a new adapter in scripts/agentic_serving/web_searcher.py."
            ),
        )
        return 0

    api_key = _resolve_api_key(spec, backend)
    if api_key is None:
        return 0

    query = _extract_query(_read_input()).strip()
    if not query:
        _emit_error(
            error="missing_query",
            backend=backend,
            detail="No query string in dispatch payload.",
        )
        return 0

    result = _dispatch_adapter(spec["adapter"], query, api_key, backend)
    if result is not None:
        print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())

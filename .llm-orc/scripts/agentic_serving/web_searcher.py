#!/usr/bin/env python3
"""Web-searcher script-agent for the `web-searcher` capability ensemble.

Per ADR-020: the `tool_use` Topaz slot is satisfied by a script-agent
ensemble that wraps an external web-search API. Backend selection is
operator-configurable via environment variables (WEB_SEARCH_BACKEND and
WEB_SEARCH_API_KEY). The default backend is Tavily.

The script reads a JSON input from stdin shaped roughly like
``{"query": "..."}`` (the orchestrator's dispatch payload), calls the
configured backend, and writes a structured JSON result to stdout. On
backend failure (authentication, rate limit, network) it writes a
structured error object the orchestrator's reasoning surface can act
on rather than raising.

Operators extend the script with additional backend adapters by adding
a function ``_search_<backend>(query, api_key) -> dict`` and mapping
the backend name in BACKEND_ADAPTERS. Cycle 5 BUILD ships the Tavily
adapter only; Brave / Exa / Serper adapters are deferred to operators
that need them.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

DEFAULT_BACKEND = "tavily"
DEFAULT_RESULT_COUNT = 5
DEFAULT_TIMEOUT_SECONDS = 30


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
    # Fallback — some dispatch shapes pass the prompt as `input`.
    if isinstance(payload.get("input"), str):
        return payload["input"]
    return ""


def _emit_error(error: str, backend: str, detail: str = "") -> None:
    """Emit a structured error object the orchestrator can route on."""
    payload: dict[str, Any] = {"error": error, "backend": backend}
    if detail:
        payload["detail"] = detail
    print(json.dumps(payload))


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


# Operators add backend adapters by registering them here. The default
# backend (Tavily) is the only adapter Cycle 5 ships; Brave / Exa /
# Serper / etc. are deferred per ADR-020 §"Scope".
BACKEND_ADAPTERS = {
    "tavily": _search_tavily,
}


def main() -> int:
    backend = (os.environ.get("WEB_SEARCH_BACKEND") or DEFAULT_BACKEND).strip().lower()
    api_key = os.environ.get("WEB_SEARCH_API_KEY", "").strip()

    adapter = BACKEND_ADAPTERS.get(backend)
    if adapter is None:
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

    if not api_key:
        _emit_error(
            error="authentication_failed",
            backend=backend,
            detail=(
                "WEB_SEARCH_API_KEY environment variable is empty or unset. "
                f"Obtain an API key for {backend} and export it before "
                "running the orchestrator."
            ),
        )
        return 0

    payload = _read_input()
    query = _extract_query(payload).strip()
    if not query:
        _emit_error(
            error="missing_query",
            backend=backend,
            detail="No query string in dispatch payload.",
        )
        return 0

    try:
        result = adapter(query, api_key)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            _emit_error(
                error="authentication_failed",
                backend=backend,
                detail=f"HTTP {exc.code}: {exc.reason}",
            )
        elif exc.code == 429:
            _emit_error(
                error="rate_limited",
                backend=backend,
                detail=f"HTTP {exc.code}: {exc.reason}",
            )
        else:
            _emit_error(
                error="backend_http_error",
                backend=backend,
                detail=f"HTTP {exc.code}: {exc.reason}",
            )
        return 0
    except urllib.error.URLError as exc:
        _emit_error(
            error="backend_unavailable",
            backend=backend,
            detail=str(exc.reason),
        )
        return 0
    except (TimeoutError, OSError) as exc:
        _emit_error(
            error="backend_unavailable",
            backend=backend,
            detail=f"{type(exc).__name__}: {exc}",
        )
        return 0
    except json.JSONDecodeError as exc:
        _emit_error(
            error="backend_invalid_response",
            backend=backend,
            detail=f"JSON decode failure: {exc}",
        )
        return 0

    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())

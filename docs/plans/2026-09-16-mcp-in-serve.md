# Design: the serve exposes MCP at /mcp

Written 2026-09-16 after the ng-mini serve came up. Approved shape:
mount the MCP tool set inside `llm-orc serve` so a remote MCP client on
the tailnet can do everything a local `llm-orc mcp serve` session can
(invoke, create, update, delete ensembles; profiles; scripts; library;
promotion) against the serve's own project. The remote checkout is the
ensemble store; nothing syncs back to the laptop.

## Shape

    MCP client (.mcp.json: {"type":"http","url":"https://llm-orc.homelab.nate.green/mcp"})
        --> Dokku proxy --> serve (FastAPI) --> /mcp = FastMCP streamable HTTP app
                                                  \-> same OrchestraService as /api/*

- `web/api/__init__.py` already has an unused `get_mcp_server()` that
  builds `MCPServer(service=get_orchestra_service())`. Use it: one
  service instance behind REST, `/v1`, and MCP.
- `create_app()` mounts `get_mcp_server()._mcp.streamable_http_app()` at
  `/mcp`. FastMCP's streamable transport needs its session manager run
  inside the ASGI lifespan (`async with mcp.session_manager.run():`);
  `create_app()` gains a lifespan that does this. Expose the FastMCP app
  from `MCPServer` through a small method rather than reaching into
  `_mcp`.
- On by default. No flag, no plist change. No auth: the tailnet is the
  boundary, as for the rest of the serve.
- `mcp` 1.28.1 is already a dependency; `streamable_http_app()` and
  `session_manager` exist. The older `--transport http` (SSE) path in
  `llm-orc mcp serve` is untouched.

## Tests (hermetic, TestClient)

1. `POST /mcp` with an `initialize` request returns a server
   capabilities result (proves the mount and the lifespan).
2. `tools/list` over `/mcp` returns the same tool names as
   `MCPServer.list_tools()` (proves it is the full set, not a subset).
3. `tools/call` `list_ensembles` over `/mcp` and `GET /api/ensembles`
   agree on a temp project with one ensemble (proves the shared service).
4. A request with `Host: llm-orc.homelab.nate.green` is not rejected
   (FastMCP's DNS-rebinding guard; must be off or allow the tailnet
   host). This test must be shown red first by enabling the guard.
5. `GET /health` still works and existing `test_server.py` passes.

## Out of scope

Auth. Per-client project switching (`set_project` acts on the serve's
process; document that it is global). Streaming tool output over MCP.
REST parity for MCP-only tools and a local `--remote` shim (#191).

## Acceptance on the rig

From this laptop, with the entry in `.mcp.json`: `list_ensembles`
returns ng-mini's 120; `create_ensemble` writes a file under ng-mini's
`.llm-orc/ensembles/`; `GET /api/ensembles/<new>` then lists it; `invoke`
runs it there.

## Result (2026-09-16, evening)

Merged on local main (#194, merge 4d33ca4d; suite 4328, coverage 92.55%,
lint clean). Deployed to ng-mini from the checkout on the practitioner's
go (one-time exception: the mini runs releases from now on, see below).

Acceptance over `https://llm-orc.homelab.nate.green/mcp` from the laptop:
`initialize` 200 with a session id; `tools/list` 30 tools;
`create_ensemble` wrote `.llm-orc/ensembles/remote-smoke.yaml` on the
mini; `GET /api/ensembles` listed it; `invoke` over MCP ran it there
(`deliverable: "pong"`, 6 s on the 1.7b seat, three progress
notifications streamed before the result); `DELETE /mcp` 200.

What the design got wrong, caught in build and review: a Starlette
`Mount` at `/mcp` doubles FastMCP's own `/mcp` path and the SPA
catch-all shadows the retry, so it is an exact-path `Route`; FastMCP
auto-enables DNS-rebinding protection for `host=127.0.0.1`, so the
guard is disabled explicitly for the `/mcp` app (the proxied Host got
421 otherwise); `MCPServer.list_tools()` is a stale 4-item list (#195);
the first shared-service test passed against two separate services and
was replaced by a `set_project`-over-MCP-then-REST assertion pinned
red/green. `create_app()` now builds its own `MCPServer` so it stays
idempotent per process.

Client: `.mcp.json` entry `llm-orc-ng-mini` (`type: http`). A tool call
that streams notifications returns them as SSE `data:` lines before the
result; clients must match on the request id.

Deployment policy from here: the mini runs the Homebrew release
(`brew tap mrilikecoding/llm-orchestra && brew install llm-orchestra`,
formula depends on `python@3.12`, builds from PyPI source on Intel), so
`/mcp` reaches the mini "for real" with the next release. The `ng-mini`
git remote stays for emergencies only.

Released as v0.20.1 the same evening. Installing that release on the mini
found #197 (clean install resolves mcp 2.x; Intel has no cryptography>=49
wheel) and #196 (the serving project is not in the wheel). The mini runs
the 0.20.1 release under `uv tool` with both pins; 0.20.2 carries the
`mcp<2` pin.

# Handoff: llm-orc serve on ng-mini, reachable on the tailnet

Written 2026-09-16 after v0.20.0 shipped (#90, PR #186). This is the
task that #90 was in service of. Everything in "What exists" was
verified in this session; everything in "Unknowns" was not.

## Goal

A client on any tailnet device talks to `https://llm-orc.homelab.nate.green`
and gets the full serve: `/v1` for OpenCode-style agentic use,
`/api/ensembles` to list, create, and run ensembles, `/api/models` to
list and pull models, and the web UI. The serve runs natively on ng-mini
and owns its llama-server router; nothing else is installed for
inference.

## Shape

    client (tailnet) --https--> Dokku edge nginx --> llm-orc-proxy container (nginx)
                                                       --> host.lima.internal:8765  = llm-orc serve (launchd, loopback)
                                                              --> 127.0.0.1:8080     = llama-server router (owned by the serve)

The serve binds loopback only. Colima's `host.lima.internal` alias reaches
the Mac's loopback from inside containers (verified against a loopback
service on ng-mini this session).

## What exists (verified)

- **Code**: v0.20.0 on PyPI and Homebrew; `llm-orc serve` owns the router
  (`docs/serving.md`, "Local inference"). `deploy/ng-mini/` has the
  launchd unit and setup notes.
- **Dokku app `llm-orc`**: deployed from `~/Development/llm-orc-proxy`
  (nginx:alpine, `proxy_pass http://host.lima.internal:8765`, buffering
  off, 3600 s timeouts, 50 m body). Edge nginx set the same way
  (`nginx:set` read/send timeouts, buffering, body size; config rebuilt).
  `http://llm-orc.homelab.nate.green/health` answers **502** now: routing
  works, upstream absent. `https://` answers 404: no cert yet.
- **ng-mini facts**: macOS, Tailscale IP 100.92.166.102, hostname
  `ng-mini.corgi-woodpecker.ts.net`, user `nathanielgreen`, Ollama
  installed with only `llava` (can be uninstalled; nothing depends on it
  now). Homelab is Dokku inside a Colima VM (2 CPU / 4 GB) per
  `mrilikecoding/homelab`.
- **Laptop facts**: `~/.ssh/id_ed25519` is refused for
  `nathanielgreen@ng-mini` (publickey, password, keyboard-interactive
  offered). `ssh dokku` works. `homelab` CLI installed.

## Unknowns (resolve in order, each is a stop point)

1. ssh access for the lead (blocked on the practitioner; see step 0).
2. Whether ng-mini has `brew`, `uv`, `llama.cpp` (router mode needs a
   recent build; the laptop's b9850 has it), and free disk for ~22 GB of
   GGUFs plus the llama.cpp cache.
3. ng-mini RAM. The preset assumes the 32 GB target rig: `c = 40960` and
   one resident model (an 8b is ~11 GB resident at that window; the 14b
   more). If it is 16 GB, set `options.num_ctx` per profile before the
   first pull.
4. Whether a launchd *agent* (gui domain) is right: it needs a logged-in
   user session. If ng-mini runs headless without auto-login, use a
   LaunchDaemon instead (root domain, different plist location and paths)
   and `pmset -c sleep 0` / `caffeinate` so the box does not sleep.
5. `homelab https:enable llm-orc` is server-only; whether it needs
   anything beyond running it once on ng-mini.

## Steps

**0. Access (practitioner).** Either `ssh-copy-id ng-mini` from the
laptop so the lead can drive the rest, or do steps 1-3 by hand from
`deploy/ng-mini/README.md`. Check: `ssh -o BatchMode=yes ng-mini hostname`.

**1. Toolchain on ng-mini.**

    brew install llama.cpp uv
    llama-server --help | grep -c models-preset      # must be > 0 (router mode)
    df -h ~                                          # need ~30 GB free

**2. Checkout.** Origin main is current (pushed 2026-09-16), so a plain
clone is fine now:

    git clone https://github.com/mrilikecoding/llm-orc.git ~/Development/llm-orc
    cd ~/Development/llm-orc && git submodule update --init && uv sync
    uv run llm-orc --version                          # 0.20.0

**3. Serve under launchd.** Paths in the plist assume
`/Users/nathanielgreen/Development/llm-orc`; edit if different.

    cp deploy/ng-mini/com.llm-orc.serve.plist ~/Library/LaunchAgents/
    launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.llm-orc.serve.plist
    tail -f ~/Library/Logs/llm-orc-serve.log     # expect "llama-server router at http://127.0.0.1:8080/v1"
    curl -s 127.0.0.1:8765/health                 # {"status":"healthy","version":"0.20.0"}
    curl -s 127.0.0.1:8765/api/models             # five models, all "unloaded"

If the log shows "llama-server exited with code N before listening", the
error carries the router's last stderr lines; the two failures seen so far
were an unrecognized preset option and a busy port.

**4. Through the tailnet (from the laptop).**

    curl -s http://llm-orc.homelab.nate.green/health          # 200 now, was 502
    curl -s http://llm-orc.homelab.nate.green/api/models

**5. HTTPS (practitioner, on ng-mini).** `homelab https:enable llm-orc`,
then `curl -s https://llm-orc.homelab.nate.green/health`.

**6. Pull the tiers (from anywhere on the tailnet; ~22 GB, minutes each).**
Each call blocks until the router reports the real status.

    for m in qwen3-8b qwen3-1.7b qwen3-14b deepseek-r1-8b; do
      curl -s -X POST https://llm-orc.homelab.nate.green/api/models/$m/pull; echo
    done

Check after: `/api/models` shows them; with one resident model, only the
last pulled is "loaded".

**7. Acceptance, in order (stop at the first failure).**

1. `GET /v1/models` lists `agentic-tier-cheap-general`.
2. One `/v1/chat/completions` with a `write_file` tool through the
   tailnet URL returns a parsed `tool_calls` entry (same probe as the
   #90 spike; the seat is qwen3-8b, first call pays the load).
3. OpenCode configured with `baseURL: https://llm-orc.homelab.nate.green/v1`
   completes an explain turn and a build turn on an existing repo
   (`docs/serving.md` for the config shape). Watch the log for the
   40960 window: a turn that exceeds it is refused by the router, not
   silently truncated as Ollama did.
4. The #90 regression gate, owed since the merge: the ladder with T1
   alone at r>=5, then the full run, then the 7-turn probe, against this
   serve (`docs/serving-roadmap.md`, State, "Next up 2"). This is where
   the jinja templating and tool-call parsing either hold or do not.

## Operating notes

- Update: `git pull && uv sync && launchctl kickstart -k gui/$(id -u)/com.llm-orc.serve`.
- Stop: `launchctl bootout gui/$(id -u)/com.llm-orc.serve`. SIGTERM to the
  serve stops the router with it (verified).
- The rendered preset is `.llm-orc/llama-server.ini` in the checkout
  (gitignored); read it to see exactly what the router was given.
- Downloads land in `LLAMA_CACHE` (`~/Library/Caches/llama.cpp` in the
  plist).
- The serve has no auth. The tailnet is the boundary, same as every other
  homelab app; do not expose the Dokku app outside it.

## Out of scope here

MCP over the tailnet (separate launchd unit and proxy path; the SSE
transport is the older protocol). `llm-orc web` owning the router.
Tuning `num_ctx`/`--models-max` for throughput; measure first.

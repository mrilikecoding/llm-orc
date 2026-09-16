# llm-orc on ng-mini

The serve runs natively under launchd, bound to loopback, owning its
llama-server router. The `llm-orc-proxy` Dokku app forwards
`https://llm-orc.homelab.nate.green` to it over Colima's host alias.

## One-time setup (on ng-mini)

    brew install llama.cpp uv
    git clone <llm-orc> ~/Development/llm-orc      # or push a branch to it
    cd ~/Development/llm-orc && uv sync
    cp deploy/ng-mini/com.llm-orc.serve.plist ~/Library/LaunchAgents/
    launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.llm-orc.serve.plist
    tail -f ~/Library/Logs/llm-orc-serve.log         # "llama-server router at ..."

Models download from Hugging Face on first use. To warm the tiers ahead
of time from any machine on the tailnet:

    for m in qwen3-8b qwen3-1.7b qwen3-14b deepseek-r1-8b; do
      curl -X POST https://llm-orc.homelab.nate.green/api/models/$m/pull
    done

## Update

    cd ~/Development/llm-orc && git pull && uv sync
    launchctl kickstart -k gui/$(id -u)/com.llm-orc.serve

## Checks

    curl https://llm-orc.homelab.nate.green/health
    curl https://llm-orc.homelab.nate.green/api/models
    curl https://llm-orc.homelab.nate.green/v1/models

HTTPS for the app is enabled on the server with `homelab https:enable
llm-orc` (server-only command). Until then, use http://.

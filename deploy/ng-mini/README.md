# llm-orc on ng-mini

The serve runs natively under launchd, bound to loopback, owning its
llama-server router. The `llm-orc-proxy` Dokku app forwards
`https://llm-orc.homelab.nate.green` to it over Colima's host alias.

## One-time setup (on ng-mini)

ng-mini is a 2018 Intel mini (i7-8700B, 6 cores, 32 GB, no usable GPU),
so inference is CPU-only and Homebrew ships no Intel bottles for
`llama.cpp` or `uv`. Install both from upstream binaries instead of brew:

    curl -LsSf https://astral.sh/uv/install.sh | sh          # ~/.local/bin/uv
    tag=$(curl -sL https://github.com/ggml-org/llama.cpp/releases/latest/download/nightly-tag.txt)
    mkdir -p ~/opt/llama.cpp && cd ~/opt/llama.cpp
    curl -sL https://github.com/ggml-org/llama.cpp/releases/download/$tag/llama-$tag-bin-macos-x64.tar.gz | tar xz
    ln -sfn ~/opt/llama.cpp/llama-$tag ~/opt/llama.cpp/current
    ln -sf ~/opt/llama.cpp/current/llama-server /usr/local/bin/llama-server
    llama-server --help | grep -c models-preset               # must be > 0

The plist's PATH includes `/usr/local/bin`, so the symlink is all the
serve needs. Installed 2026-09-16: build b10964 (0.4.1-dev).

    git clone https://github.com/mrilikecoding/llm-orc.git ~/Development/llm-orc
    cd ~/Development/llm-orc && git submodule update --init && uv sync
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

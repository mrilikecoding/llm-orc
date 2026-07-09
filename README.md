# LLM Orchestra

[![PyPI version](https://badge.fury.io/py/llm-orchestra.svg)](https://badge.fury.io/py/llm-orchestra)
[![CI](https://github.com/mrilikecoding/llm-orc/workflows/CI/badge.svg)](https://github.com/mrilikecoding/llm-orc/actions)
[![codecov](https://codecov.io/gh/mrilikecoding/llm-orc/graph/badge.svg?token=FWHP257H9E)](https://codecov.io/gh/mrilikecoding/llm-orc)
[![Python 3.11-3.13](https://img.shields.io/badge/python-3.11--3.13-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Downloads](https://static.pepy.tech/badge/llm-orchestra)](https://pepy.tech/project/llm-orchestra)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/mrilikecoding/llm-orc)](https://github.com/mrilikecoding/llm-orc/releases)

Orchestrate ensembles of specialized models — local and cloud — to do real analytical work. Coordination over scale.

A decent laptop can run multiple small language models simultaneously. What's missing is the coordination layer — the system that decomposes problems, routes them to specialized agents, manages dependencies between them, and synthesizes results. LLM Orchestra provides that layer: independent agents run in parallel, dependent agents wait for what they need, and script agents handle deterministic work alongside LLM agents. Mix expensive cloud models with free local models, and put cost only where it matters.

## What it does

- **Multi-agent ensembles** — DAGs of LLM and script agents with dependencies, fan-out, guards, bounded loops, and dynamic dispatch
- **Agentic serving** — `llm-orc serve` exposes OpenAI-compatible endpoints so coding tools (OpenCode, Aider, Cline) can use composed ensembles as their model backend, with build deliverables verified by an accept gate before they ship
- **Model profiles** — named model+provider shortcuts; back any tier with your own provider via untracked local overrides
- **Hybrid local/cloud** — Ollama, Claude, Gemini, and any OpenAI-compatible server (vLLM, LM Studio, OpenRouter), with cost and usage tracking
- **Scripts and artifacts** — script agents with JSON I/O, timestamped execution artifacts, an ensemble library, and an MCP server

## Install

```bash
# Homebrew (macOS)
brew tap mrilikecoding/tap && brew install llm-orchestra

# pip (all platforms)
pip install llm-orchestra

llm-orc --version
```

## Quick start

```bash
# Configure providers (keys are encrypted at rest); skip for Ollama-only use
llm-orc auth setup

# Initialize project config (.llm-orc/ with ensembles/, scripts/, config.yaml)
llm-orc config init

# Run an ensemble
llm-orc list-ensembles
cat code.py | llm-orc invoke code-review
```

An ensemble is a YAML file: agents, their models, and who depends on whom.

```yaml
name: code-review
agents:
  - name: security-reviewer
    model_profile: free-local
    system_prompt: "You are a security analyst. Identify vulnerabilities."

  - name: senior-reviewer
    model_profile: default-claude
    depends_on: [security-reviewer]
    system_prompt: "Synthesize the analysis into actionable recommendations."
```

The full command and configuration reference — output formats, config
management, script agents, the ensemble library, the MCP server, profile and
fallback schemas — lives in [docs/cli-reference.md](docs/cli-reference.md).

## Agentic serving

Point an agentic coding tool at llm-orc and use composed ensembles as its
model backend:

```bash
llm-orc serve   # OpenAI-compatible /v1/models + /v1/chat/completions
```

```jsonc
// e.g. ~/.config/opencode/opencode.json
{
  "provider": {
    "llm-orc": {
      "npm": "@ai-sdk/openai-compatible",
      "options": { "baseURL": "http://localhost:8765/v1" },
      "models": { "agentic": {} }
    }
  }
}
```

Every turn runs one declarative serving ensemble: classify the request, route
it to a capability seat, verify the deliverable (build turns run a
test-writer → code-writer → sandboxed executor → adequacy judge pipeline with
a bounded retry round), and emit it through the client's own tool surface.
Conversation context threads from the client-sent history; the builder never
grades itself. Runs entirely on local models by default.

Architecture and operator guide: [docs/serving.md](docs/serving.md) ·
Staged path to full model parity: [docs/serving-roadmap.md](docs/serving-roadmap.md)

## Documentation

| Doc | What's in it |
|-----|--------------|
| [docs/cli-reference.md](docs/cli-reference.md) | Complete CLI and configuration reference |
| [docs/serving.md](docs/serving.md) | Agentic serving architecture and operator guide |
| [docs/serving-roadmap.md](docs/serving-roadmap.md) | Staged roadmap toward full model parity |
| [docs/architecture.md](docs/architecture.md) | System architecture and design principles |
| [docs/adrs/](docs/adrs/) | Architecture decision records (serving ADRs in [docs/adrs/serving/](docs/adrs/serving/)) |
| [docs/ensemble_vs_single_agent_analysis.md](docs/ensemble_vs_single_agent_analysis.md) | Multi-agent vs single-agent research findings |

## Philosophy

**Coordination over scale. Process over generation.**

Generation was never the bottleneck — evaluation is. An ensemble of smaller,
specialized models, each owning a bounded task, produces structured output
designed for human evaluation: a security reviewer finds vulnerabilities, a
synthesis agent integrates findings, and the human evaluates a structured
analysis rather than raw generation. The research evidence supports the bet:
orchestrated ensembles of open-source models have matched or exceeded frontier
model performance on established benchmarks, and cascade routing replicates
frontier quality at a fraction of the cost.

Running models locally is a practical choice, not an ideological one: no
per-query billing, data stays on your hardware, and the local/cloud mix puts
spend where it matters.

## Development

```bash
uv sync --dev
make test    # pytest with coverage gate
make lint    # mypy, ruff, complexity, security checks
```

## License

AGPL-3.0-or-later — see [LICENSE](LICENSE).

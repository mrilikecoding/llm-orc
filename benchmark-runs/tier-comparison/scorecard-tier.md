# Agentic-Serving Benchmark Scorecard

_Run: 2026-06-22T04:15:44.983951+00:00_

## Config: cheap-local

_Spike-τ working config (§0): hosted qwen3.6-plus seat + local qwen3:8b coder + 8b→14b→MiniMax coder-tier escalation; ≈cents/session, mostly local. (Supersedes the pre-τ $0 14b-seat+8b-coder default, which is the broken swap.)_

| C\H | H2 | H3 | H5 |
|---|---|---|---|
| C4 | · | ✓ 1/1 | · |
| C3 | · | ✓ 1/1 | · |
| C2 | ✓ 1/1 | ✓ 1/1 | · |
| C1 | · | ✓ 1/1 | ✓ 1/1 |

**Ceiling:** H5 × C4 (frontier: h3c4, l12)

## Config: frontier

_Claude Sonnet subagent — one-shot, no llm-orc framework (§7). Gathered in-session via the Agent tool; scored by frontier.score_cell. The value-proposition pole: [cheap + framework] vs [frontier, no orchestration]._

| C\H | H2 | H3 | H5 |
|---|---|---|---|
| C4 | · | ✓ 1/1 | · |
| C3 | · | ✓ 1/1 | · |
| C2 | ✓ 1/1 | ✓ 1/1 | · |
| C1 | · | ✓ 1/1 | ✓ 1/1 |

**Ceiling:** H5 × C4 (frontier: h3c4, l12)

## Tier comparison (§7)

**MATCH** — horizon gap 0, complexity gap 0 (criterion: cheap-local ceiling within one rung per axis of frontier).

## Bleed-injection probe (§6)

(not run)

## Provenance (§9)

```json
{
  "date": "2026-06-22T04:15:44.983951+00:00",
  "configs": [
    {
      "name": "cheap-local",
      "coder_cheap": "qwen3:8b",
      "coder_escalated": "qwen3:14b",
      "seat": "hosted-qwen3.6-plus (Zen)",
      "paid": false
    },
    {
      "name": "frontier",
      "coder_cheap": "claude-sonnet",
      "coder_escalated": "claude-sonnet",
      "seat": "claude-sonnet (subagent)",
      "paid": true
    }
  ],
  "serve_port": 8770,
  "n_per_cell": {
    "coarse": 1,
    "confirm": 3,
    "concentrate": 3
  },
  "pre_registered_threshold": "pass-rate \u2265 2/3 \u2014 ceil(2n/3) of n runs pass",
  "pre_registered_match": "cheap-local ceiling within one rung per axis of frontier",
  "tool_versions": {
    "note": "frontier=Sonnet subagents; cheap=coarse n=1"
  },
  "probe_ran": false
}
```

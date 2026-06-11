# ADR-039 content-anchor discharge run — RESULTS (2026-06-09)

Real OpenCode 1.15.5 (`opencode run --format json`) → `uv run llm-orc serve --port 8765`
(working-tree source, with V-01..V-05 wired) → qwen3:14b seat/judgment + `code-generator`
ensemble (tier=cheap → qwen3:8b coder). $0 local. The exact 5-file Finding H task
(converters.py, test_converters.py, cli.py, test_cli.py, README.md).

## Verdict: ADR-039 anchor validated AS FAR AS THE RUN REACHED; full 5-file discharge BLOCKED by a separate finding.

### Run 1 (180s default read timeout) — INVALIDATED by infrastructure
- Turn 1: write converters.py (anchor=false, first file) — but the deliverable came out as
  garbled `\xNN` hex-escape text with mangled names (cheap coder degraded under load).
- `httpx.ReadTimeout` (×2) on the qwen3:14b seat calls: `ollama ps` holds ONE model at a
  time on this 32GB host, so it thrashes reloading 14b after each 8b coder swap; a cold
  reload + inference exceeded 180s.
- Turn 2: premature finish before any dependent file. **Anchor never fired.** Inconclusive.

### Fix applied (run-environment only; revert after — NOT part of the ADR-039 change)
`.llm-orc/config.yaml` `performance.concurrency.request_timeout.read: 600` (tolerate the
swap latency). Coder warmed. Serve restarted. Did NOT restart the practitioner's ollama.

### Run 2 (600s read timeout, coder warm) — clean infra; anchor CONFIRMED live
| turn | tail_kind | verdict | action | delegated | anchor | deliverable |
|------|-----------|---------|--------|-----------|--------|-------------|
| 1 | first_turn | (none) | write | code-generator | **false** | converters.py |
| 2 | trailing_tool_result | REMAINING | write | code-generator | **true** | test_converters.py |
| 3 | trailing_tool_result | REMAINING | **finish** | — | false | (premature finish) |

- **Zero ReadTimeouts** (the timeout fix held).
- **converters.py: clean valid Python**, the three pinned functions exactly
  (`celsius_to_fahrenheit`, `fahrenheit_to_celsius`, `celsius_to_kelvin`). Run-1's hex
  garbage was environment-stress corruption, not a coder defect.
- **Turn 2 `anchor=true`** — the content anchor FIRED live under the real client on the
  first dependent file (converters.py is a produced sibling → framework built the anchor
  from its captured content and injected it into the callee dispatch). **Presence layer
  confirmed end-to-end.**
- **test_converters.py references the REAL API** (`celsius_to_fahrenheit`,
  `fahrenheit_to_celsius`, `celsius_to_kelvin`) with correct assertions, **zero invented
  symbols**. Finding H's invention signature (`convert_temperature`, `fahrenheit_to_kelvin`)
  did NOT reproduce on the anchored dependent file. **Correctness + outcome confirmed for
  this dependency.** (Caveat: the test inline-reimplements the functions rather than
  `import`ing them — the known cheap-coder weakness Spike ξ flagged for the full-content
  arm; it references real symbols, so it is not the Finding H invention failure.)

## The blocker — a separate finding on the ADR-037/038 termination surface (NOT ADR-039)

Both runs finished prematurely: a `REMAINING` judgment (the judge correctly knows files
remain) followed by a seat-filler **no-tool-call** on the action call → `enforced.action is
None` → `FinishTurn` → OpenCode receives `stop` and **ends the agentic loop**. Run 1 died at
turn 2, run 2 at turn 3.

**This refutes ADR-038/Spike ρ's accepted backstop assumption** ("the ~2/10 no-tool-call
rate is backstopped by the next re-judgment + the AS-3 cap"). Under the real client a finish
ENDS the loop — there is no next turn, so neither the re-judgment nor the AS-3 cap fires. At
a ~2/10 per-turn no-tool-call rate, a 5-file session (≈4 REMAINING turns) terminates early
with probability ≈ 1−0.8^4 ≈ 60%. The prior WP-LB-L (2-file) and Finding H (5-file)
trajectory runs completed because they happened not to flake; these two did.

This is independent of ADR-039: the anchor code runs only on a delegated dependent-file
generation, and the no-tool-call happens before that. The same seat + action-call
composition predates V-01..V-05.

## Disposition (practitioner to direct)
- ADR-039 stays **Conditional**: the anchor is validated as far as the real client reached
  (presence + correctness + outcome on the converters→test dependency), but the full 5-file
  discharge — especially the README prose-callee criterion — was not reached.
- Candidate **Finding I** (loop-back to DECIDE on the ADR-037/038 termination surface): a
  `REMAINING` verdict should not be overridable by a seat-filler no-tool-call into a
  loop-ending finish. Grounded by a pre-registered + methods-reviewed spike, per the
  F→ρ→ADR-038 / G / H precedent.

Evidence: `serve.log`, `opencode_run.out`, `workspace/` (converters.py + test_converters.py);
`run1/` (the invalidated infra-timeout run).

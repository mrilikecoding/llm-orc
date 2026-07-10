# Client-Delegated Test Runs Implementation Plan (#83 run half)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A "run the tests" turn delegates one deterministically-built pytest command to the client's bash tool and, on the continuation, replies with an honest verdict parsed from pytest's own summary — zero model calls end to end.

**Architecture:** Two-pass turn through the existing permission seam, mirroring the read half. Pass 1: classify detects the run signal, routes to a new script-only `need-run` shape, emit ships `{"finish": false, "run": "<command>"}`, the caller maps it to a bash tool_call. Pass 2: the run-shaped continuation re-enters the pipeline; the renderer adds an `assistant: [ran <command>]` block (body indented two spaces, tail-capped); classify routes to the `run-verdict` shape whose script parses the pytest summary deterministically and emits the verdict as prose. No server-side state.

**Tech Stack:** Python 3.13, pytest, the L0 ensemble engine, script nodes under `.llm-orc/scripts/agentic_serving/`.

**Spec:** `docs/plans/2026-07-09-client-run-delegation-design.md`

## Global Constraints

- ruff (88 chars max) and mypy strict compliant from first draft; run `make lint` before every commit.
- TDD: write the failing test, see it fail, implement, see it pass, commit. One behavioral unit per commit.
- Never mix structural and behavioral changes in a commit. No AI attribution in commit messages.
- Commit prefixes: `feat:`, `test:`, `fix:`, `refactor:`, `docs:`.
- Scripts under `.llm-orc/scripts/agentic_serving/` are run as subprocesses by the engine; their unit tests drive them via subprocess (see `tests/unit/serving/test_serving_classify.py` for the pattern).
- Full suite: `make test`. Targeted: `uv run pytest <path> -v`.
- The emitted command is a closed template (`pytest -q` + regex-safe named `test_` files). Never model-composed, never interpolated from free text.

## Shared vocabulary (used by every task)

- **Run block grammar** in the rendered context (parallel to `[wrote]`/`[read]`):
  - success: `assistant: [ran <command>]` followed by the output body, every non-empty body line indented two spaces
  - truncated: `assistant: [ran <command> (truncated)]` + indented TAIL of the output (pytest's summary is at the end)
  - failure: `assistant: [ran <command> (failed)] <flat reason, single line>`
- **Run-shaped tool call**: arguments parse to a dict containing `command` and NOT containing `filePath`. Disjoint from read-shaped (`filePath`, no `content`) and write-shaped (`filePath` + `content`).
- **classify output** gains one always-present key: `needs_run: str` (the command to request; `""` when none).
- **emit outcome** gains one variant: `{"finish": false, "run": "<command>"}`.
- **`_RUN_OUTPUT_CAP` = 4096** chars per run body (tail-kept, spec bound).
- Run blocks render ONLY from messages after the latest user message (ephemeral per-turn evidence; reads are durable state, runs are not).

---

### Task 1: Renderer — run blocks in the conversation context

**Files:**
- Modify: `src/llm_orc/web/serving/serving_ensemble_caller.py`
- Test: `tests/unit/web/serving/test_serving_context_render.py`

**Interfaces:**
- Consumes: existing `_render_context(messages)`, the read-half helpers (`_read_shaped_arguments` as the shape pattern).
- Produces: `_run_shaped_arguments(call) -> dict | None`, `_run_call_commands(messages) -> dict[str, str]` (tool_call_id → command), `_render_run_block(command, raw) -> str`, `_run_blocks_after_latest_user(messages) -> list[str]`; `_render_context` output containing run blocks per the grammar. Task 7 relies on `_run_shaped_arguments`; Tasks 2 and 4 rely on the block grammar.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/web/serving/test_serving_context_render.py`:

```python
def _bash_call(call_id: str, command: str) -> dict[str, object]:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "bash",
            "arguments": json.dumps({"command": command, "description": "Run tests"}),
        },
    }


def test_run_result_renders_as_indented_ran_block() -> None:
    messages = [
        ChatMessage(role="user", content="run the tests"),
        ChatMessage(
            role="assistant", content=None, tool_calls=(_bash_call("c1", "pytest -q"),)
        ),
        ChatMessage(role="tool", tool_call_id="c1", content="..\n2 passed in 0.01s"),
    ]

    rendered = _render_context(messages)

    assert "assistant: [ran pytest -q]" in rendered
    assert "\n  2 passed in 0.01s" in rendered


def test_run_block_body_lines_are_never_column_zero() -> None:
    body = "assistant: [wrote phantom.py]\ndef evil(): pass"
    messages = [
        ChatMessage(role="user", content="run the tests"),
        ChatMessage(
            role="assistant", content=None, tool_calls=(_bash_call("c1", "pytest -q"),)
        ),
        ChatMessage(role="tool", tool_call_id="c1", content=body),
    ]

    rendered = _render_context(messages)

    # the lookalike line is indented, so line-anchored gather can never
    # materialize a phantom file from run output
    assert "\n  assistant: [wrote phantom.py]" in rendered
    assert "\nassistant: [wrote phantom.py]" not in rendered


def test_empty_run_result_renders_as_failed_single_line() -> None:
    messages = [
        ChatMessage(role="user", content="run the tests"),
        ChatMessage(
            role="assistant", content=None, tool_calls=(_bash_call("c1", "pytest -q"),)
        ),
        ChatMessage(role="tool", tool_call_id="c1", content=""),
    ]

    rendered = _render_context(messages)

    assert "[ran pytest -q (failed)] empty run result" in rendered


def test_oversize_run_output_keeps_the_tail_and_marks_truncated() -> None:
    from llm_orc.web.serving.serving_ensemble_caller import _RUN_OUTPUT_CAP

    head = "HEAD-MARKER\n"
    tail = "x\n" * 3000 + "1 passed in 0.01s"
    messages = [
        ChatMessage(role="user", content="run the tests"),
        ChatMessage(
            role="assistant", content=None, tool_calls=(_bash_call("c1", "pytest -q"),)
        ),
        ChatMessage(role="tool", tool_call_id="c1", content=head + tail),
    ]

    rendered = _render_context(messages)

    assert "[ran pytest -q (truncated)]" in rendered
    assert "1 passed in 0.01s" in rendered
    assert "HEAD-MARKER" not in rendered
    # the cap applies to the raw body; the two-space indent adds bounded
    # per-line overhead on top
    assert len(rendered) < 3 * _RUN_OUTPUT_CAP


def test_run_blocks_from_before_the_latest_user_message_do_not_render() -> None:
    messages = [
        ChatMessage(role="user", content="run the tests"),
        ChatMessage(
            role="assistant", content=None, tool_calls=(_bash_call("c1", "pytest -q"),)
        ),
        ChatMessage(role="tool", tool_call_id="c1", content="5 passed in 0.02s"),
        ChatMessage(role="assistant", content="Ran `pytest -q`: 5 passed."),
        ChatMessage(role="user", content="now explain the failures"),
    ]

    rendered = _render_context(messages)

    assert "[ran pytest -q]" not in rendered
    assert "5 passed in 0.02s" not in rendered


def test_bash_call_never_renders_as_a_write_or_read_block() -> None:
    messages = [
        ChatMessage(role="user", content="run the tests"),
        ChatMessage(
            role="assistant", content=None, tool_calls=(_bash_call("c1", "pytest -q"),)
        ),
        ChatMessage(role="tool", tool_call_id="c1", content="1 passed in 0.01s"),
    ]

    rendered = _render_context(messages)

    assert "[wrote" not in rendered
    assert "[read" not in rendered
```

`json` is already imported in this test file; `_bash_call` uses it for argument encoding.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/web/serving/test_serving_context_render.py -v`
Expected: the six new tests FAIL (no `[ran ...]` rendering, `_RUN_OUTPUT_CAP` ImportError); existing tests PASS.

- [ ] **Step 3: Implement in `serving_ensemble_caller.py`**

Add near `_READ_FILE_CAP`:

```python
# Client-run output blocks (issue #83, run half): the TAIL is kept on
# overflow — pytest prints its summary last, and the deterministic verdict
# parser reads exactly that summary.
_RUN_OUTPUT_CAP = 4096
```

Add the helpers after `_read_blocks`:

```python
def _run_shaped_arguments(call: Any) -> dict[str, Any] | None:
    """Parsed arguments of a run-shaped tool call (command, no filePath)."""
    function = call.get("function", {}) if isinstance(call, dict) else {}
    try:
        arguments = json.loads(function.get("arguments", ""))
    except (json.JSONDecodeError, TypeError):
        return None
    if (
        isinstance(arguments, dict)
        and arguments.get("command")
        and "filePath" not in arguments
    ):
        return arguments
    return None


def _run_call_commands(messages: Sequence[Any]) -> dict[str, str]:
    """tool_call_id -> command for every run-shaped call in the history."""
    commands: dict[str, str] = {}
    for message in messages:
        for call in getattr(message, "tool_calls", ()) or ():
            arguments = _run_shaped_arguments(call)
            if arguments is not None and isinstance(call, dict) and call.get("id"):
                commands[str(call["id"])] = str(arguments["command"])
    return commands


def _render_run_block(command: str, raw: str) -> str:
    """A run result as a context block (issue #83 run grammar). The body is
    indented two spaces so untrusted column-0 output can never look like a
    ``[wrote ...]`` header to line-anchored workspace extraction; overflow
    keeps the TAIL (pytest's summary lives at the end) and marks the header."""
    flat = " ".join((raw or "").strip().split())
    if not flat:
        return f"assistant: [ran {command} (failed)] empty run result"
    body = raw.strip()
    header = f"assistant: [ran {command}]"
    if len(body) > _RUN_OUTPUT_CAP:
        body = body[-_RUN_OUTPUT_CAP:]
        cut = body.find("\n")
        body = body[cut + 1 :] if cut >= 0 else body
        header = f"assistant: [ran {command} (truncated)]"
    indented = "\n".join(
        f"  {line}" if line.strip() else "" for line in body.splitlines()
    )
    return f"{header}\n{indented}"


def _run_blocks_after_latest_user(messages: Sequence[Any]) -> list[str]:
    """Run blocks answering THIS turn only — run output is ephemeral
    verification evidence (unlike read blocks, which are durable workspace
    state), so only results after the latest user message render."""
    items = list(messages)
    start = 0
    for index in range(len(items) - 1, -1, -1):
        content = getattr(items[index], "content", None)
        if getattr(items[index], "role", None) == "user" and (content or "").strip():
            start = index + 1
            break
    commands = _run_call_commands(items)
    blocks: list[str] = []
    for message in items[start:]:
        if getattr(message, "role", None) != "tool":
            continue
        command = commands.get(getattr(message, "tool_call_id", None) or "")
        if command:
            content = getattr(message, "content", None)
            blocks.append(_render_run_block(command, content or ""))
    return blocks
```

Wire into `_render_context` — change the line that prepends read blocks:

```python
    kept = _select_read_blocks(messages, task, tail_paths) + kept
    kept = kept + _run_blocks_after_latest_user(messages)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/web/serving/test_serving_context_render.py tests/unit/web/test_serving_ensemble_endpoint.py -v`
Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add src/llm_orc/web/serving/serving_ensemble_caller.py tests/unit/web/serving/test_serving_context_render.py
git commit -m "feat: render client run results as indented [ran] context blocks"
```

---

### Task 2: classify — run turns route to need-run, resumed runs to run-verdict

**Files:**
- Modify: `.llm-orc/scripts/agentic_serving/classify.py`
- Test: `tests/unit/serving/test_serving_classify.py`

**Interfaces:**
- Consumes: the run block grammar from Task 1 (via the `context` string).
- Produces: classify output with `needs_run: str` always present. Run signal without a run block: `target: "need-run"`, `kind: "need_run"`, `build: False`, `needs_run: "pytest -q [files]"`. Run signal with a run block in context: `target: "run-verdict"`, `kind: "run_verdict"`, `build: False`, `needs_run: ""`. All other decisions carry `needs_run: ""`.

Trigger (deterministic, rung-1): an imperative run verb (`run|rerun|re-run|execute`) followed within three words by a tests object (`tests?|pytest|suite`), OR a run verb plus a named `test_*.py` file. Interrogatives outrank (explain wins). The command is the closed template `pytest -q` plus the turn's named `test_*.py` files filtered through a safe-charset assertion. A run turn never triggers the need-files read path.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/serving/test_serving_classify.py`:

```python
def test_run_the_tests_routes_to_need_run_with_the_closed_command() -> None:
    decision = _classify({"task": "run the tests"})
    assert decision["target"] == "need-run"
    assert decision["kind"] == "need_run"
    assert decision["build"] is False
    assert decision["needs_run"] == "pytest -q"
    assert decision["needs_files"] == []


def test_named_test_file_rides_the_run_command() -> None:
    decision = _classify({"task": "run test_calc.py"})
    assert decision["target"] == "need-run"
    assert decision["needs_run"] == "pytest -q test_calc.py"


def test_rerun_pytest_is_a_run_turn() -> None:
    decision = _classify({"task": "rerun pytest"})
    assert decision["target"] == "need-run"
    assert decision["needs_run"] == "pytest -q"


def test_run_signal_with_a_ran_block_routes_to_run_verdict() -> None:
    context = "assistant: [ran pytest -q]\n  ..\n  2 passed in 0.01s"
    decision = _classify({"task": "run the tests", "context": context})
    assert decision["target"] == "run-verdict"
    assert decision["kind"] == "run_verdict"
    assert decision["needs_run"] == ""


def test_failed_ran_block_still_routes_to_run_verdict_not_a_reloop() -> None:
    context = "assistant: [ran pytest -q (failed)] empty run result"
    decision = _classify({"task": "run the tests", "context": context})
    assert decision["target"] == "run-verdict"
    assert decision["needs_run"] == ""


def test_write_tests_then_run_them_is_not_a_run_turn() -> None:
    decision = _classify({"task": "write tests for existing calc.py and run them"})
    assert decision["target"] != "need-run"
    assert decision["needs_run"] == ""


def test_run_the_app_is_not_a_run_turn() -> None:
    decision = _classify({"task": "run the app"})
    assert decision["target"] != "need-run"
    assert decision["needs_run"] == ""


def test_did_you_run_the_tests_stays_an_explain_turn() -> None:
    decision = _classify({"task": "did you run the tests?"})
    assert decision["target"] == "explainer"
    assert decision["needs_run"] == ""


def test_non_run_decisions_carry_empty_needs_run() -> None:
    decision = _classify({"task": "write a function that adds two numbers"})
    assert decision["needs_run"] == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/serving/test_serving_classify.py -v`
Expected: new tests FAIL (KeyError `needs_run` / wrong target); existing tests PASS.

- [ ] **Step 3: Implement in `classify.py`**

Add module-level patterns after `_EXISTING_RE`:

```python
# issue #83 run half: an imperative run verb with a tests object within a
# short window ("run the unit tests", "rerun pytest") — the window keeps
# "write tests for calc.py and run them" off the run path. A named
# test_*.py file with a run verb also qualifies ("run test_calc.py").
_RUN_VERB_RE = re.compile(r"\b(?:re-?run|run|execute)\b", re.IGNORECASE)
_RUN_TESTS_RE = re.compile(
    r"\b(?:re-?run|run|execute)\b(?:\s+[\w./-]+){0,3}?\s*\b(?:tests?|pytest|suite)\b",
    re.IGNORECASE,
)
_RAN_HEADER_RE = re.compile(r"^assistant: \[ran ", re.MULTILINE)
# Defense in depth on top of _FILE_RE's already-safe charset: an argument
# that could carry shell metacharacters never reaches the command template.
_SAFE_ARG_RE = re.compile(r"^[\w./-]+$")
```

Add helpers after `_named_source_files`:

```python
def _named_test_files(task: str) -> list[str]:
    """Every named test_*.py file, first-mention order, deduped."""
    files: list[str] = []
    for match in _FILE_RE.finditer(task):
        path = match.group(1)
        if not path.rsplit("/", 1)[-1].startswith("test_"):
            continue
        if path.endswith(".py") and path not in files:
            files.append(path)
    return files


def _run_test_command(task: str) -> str:
    """The closed run template: ``pytest -q`` + regex-safe named test files.

    Never model text (deterministic control) — the only variable part is
    filenames already restricted to ``_FILE_RE``'s metacharacter-free
    charset, re-asserted here.
    """
    named = [path for path in _named_test_files(task) if _SAFE_ARG_RE.match(path)]
    return " ".join(["pytest", "-q", *named]).strip()
```

In `main()`, after `tests_primary` is computed, add the run-signal block and gate the read path on it:

```python
    run_signal = bool(_RUN_TESTS_RE.search(task)) or (
        bool(_RUN_VERB_RE.search(task)) and bool(_named_test_files(task))
    )
    conversation_raw = str(turn.get("context", ""))
    has_run_block = bool(_RAN_HEADER_RE.search(conversation_raw))

    needs_files: list[str] = []
    read_failed = ""
    if not is_explain and not run_signal:
        needs_files, read_failed = _files_to_request(
            task, conversation_raw, tests_primary, has_build_signal
        )
    needs_run = _run_test_command(task) if run_signal and not has_run_block else ""
```

(The existing `conversation_raw = ...` line moves up into this block — keep a single assignment.)

Extend the routing chain — run branches sit between explain and need-files:

```python
    if is_explain:
        target, kind, build, needs_decider = _EXPLAIN_SEAT, "explanation", False, False
    elif run_signal and has_run_block:
        # issue #83 run half: the client ran the command — the deliverable
        # is the deterministic verdict, one run round per turn.
        target, kind, build, needs_decider = "run-verdict", "run_verdict", False, False
    elif run_signal:
        # issue #83 run half: delegate one closed-template test run.
        target, kind, build, needs_decider = "need-run", "need_run", False, False
    elif needs_files or read_failed:
        ...
```

Add the key to the printed JSON object after `"read_failed"`:

```python
                "needs_run": needs_run,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/serving/test_serving_classify.py -v`
Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add .llm-orc/scripts/agentic_serving/classify.py tests/unit/serving/test_serving_classify.py
git commit -m "feat: classify routes run turns to need-run and resumed runs to run-verdict"
```

---

### Task 3: resolve passthrough + the need-run echo shape

**Files:**
- Modify: `.llm-orc/scripts/agentic_serving/resolve.py`
- Create: `.llm-orc/ensembles/agentic-serving/need-run.yaml`
- Create: `.llm-orc/scripts/agentic_serving/need_run_echo.py`
- Test: `tests/unit/serving/test_serving_resolve.py`

**Interfaces:**
- Consumes: classify's `needs_run` (Task 2).
- Produces: resolve output carrying `needs_run: str` through (empty default on the decider path). The `need-run` ensemble is dispatchable via its `serves: need-run` declaration; its single script node emits a minimal ADR-024 envelope.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/serving/test_serving_resolve.py` (the file's `_resolve(classify_decision, decide_response=None)` helper):

```python
def test_needs_run_passes_through_resolve() -> None:
    routing = _resolve(
        _structural(
            target="need-run",
            kind="need_run",
            build=False,
            needs_run="pytest -q",
        )
    )
    assert routing["target"] == "need-run"
    assert routing["needs_run"] == "pytest -q"


def test_decider_path_defaults_needs_run_empty() -> None:
    routing = _resolve(
        _structural(target="", kind="", build=False, needs_decider=True),
        decide_response='{"target": "explainer"}',
    )
    assert routing["needs_run"] == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/serving/test_serving_resolve.py -v`
Expected: new tests FAIL with KeyError `needs_run`.

- [ ] **Step 3: Implement**

In `resolve.py` `main()`, after `read_failed = ...`:

```python
    needs_run = str(classify.get("needs_run", ""))
```

and add to the printed JSON object after `"read_failed"`:

```python
                "needs_run": needs_run,
```

Create `.llm-orc/ensembles/agentic-serving/need-run.yaml`:

```yaml
name: need-run
description: |
  issue #83 run half — the cheap dispatch target for a turn that delegates a
  test run to the client. The command rides the routing decision (classify);
  this shape exists so the skeleton dispatches SOMETHING without burning a
  model call. One deterministic script node.
serves: need-run
agents:
  - name: echo
    script: scripts/agentic_serving/need_run_echo.py
```

Create `.llm-orc/scripts/agentic_serving/need_run_echo.py`:

```python
#!/usr/bin/env python3
"""need-run shape — deterministic echo node (issue #83, run half).

The run request rides the ROUTING decision (classify -> resolve -> shape ->
form_gate -> emit), not the seat envelope; this node only satisfies the
skeleton's dispatch step with a minimal ADR-024 envelope, at zero model cost.
"""

from __future__ import annotations

import json
import sys


def main() -> None:
    sys.stdin.read()
    print(json.dumps({"status": "ok", "primary": "Requesting a client test run."}))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests, verify the shape validates**

Run: `uv run pytest tests/unit/serving/test_serving_resolve.py tests/unit/serving/test_serving_shape_catalog.py -v`
Expected: PASS. If the shape-catalog test enumerates expected intents, add `need-run` to its expectation.

- [ ] **Step 5: Commit**

```bash
git add .llm-orc/scripts/agentic_serving/resolve.py .llm-orc/ensembles/agentic-serving/need-run.yaml .llm-orc/scripts/agentic_serving/need_run_echo.py tests/unit/serving/test_serving_resolve.py
git commit -m "feat: need-run shape and resolve passthrough for client test runs"
```

---

### Task 4: run-verdict shape — deterministic pytest-summary verdict

**Files:**
- Create: `.llm-orc/scripts/agentic_serving/run_verdict.py`
- Create: `.llm-orc/ensembles/agentic-serving/run-verdict.yaml`
- Test: `tests/unit/serving/test_serving_run_verdict.py`

**Interfaces:**
- Consumes: the run block grammar (Task 1) inside the dispatch input (`Conversation so far:\n<context>\n\nCurrent request: <task>`).
- Produces: an ADR-024 envelope `{"status": "ok", "primary": "<verdict prose>"}`. The verdict is composed deterministically from pytest's summary: pass counts, fail/error counts plus up to five `FAILED`/`ERROR` lines, "no tests ran", a could-not-execute report for `(failed)` blocks, or an honest could-not-parse report carrying the output tail.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/serving/test_serving_run_verdict.py`:

```python
"""Unit tests for the run-verdict node (issue #83, run half).

run_verdict is a pure script node: it extracts the latest ``[ran ...]`` block
from the dispatched context and composes an honest verdict from pytest's own
summary line — fully deterministic, zero model calls. Driven via subprocess
exactly as the L0 engine runs a script node.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RUN_VERDICT = REPO / ".llm-orc" / "scripts" / "agentic_serving" / "run_verdict.py"


def _verdict(dispatch_text: str) -> str:
    envelope = json.dumps({"input": dispatch_text})
    out = subprocess.run(
        [sys.executable, str(RUN_VERDICT)],
        input=envelope,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    result = json.loads(out)
    primary = result["primary"]
    assert isinstance(primary, str)
    return primary


def _dispatch(block: str) -> str:
    return f"Conversation so far:\n{block}\n\nCurrent request: run the tests"


def test_all_passed_summarizes_the_pass_count() -> None:
    block = "assistant: [ran pytest -q]\n  .....\n  5 passed in 0.12s"
    assert _verdict(_dispatch(block)) == "Ran `pytest -q`: 5 passed."


def test_failures_summarize_counts_and_carry_failed_lines() -> None:
    block = (
        "assistant: [ran pytest -q]\n"
        "  ..F.F\n"
        "  FAILED test_calc.py::test_divide - ZeroDivisionError\n"
        "  FAILED test_calc.py::test_mod - AssertionError\n"
        "  2 failed, 3 passed in 0.31s"
    )
    verdict = _verdict(_dispatch(block))
    assert verdict.startswith("Ran `pytest -q`: 2 failed, 3 passed.")
    assert "FAILED test_calc.py::test_divide - ZeroDivisionError" in verdict


def test_no_tests_ran_is_reported_honestly() -> None:
    block = "assistant: [ran pytest -q]\n  no tests ran in 0.01s"
    assert _verdict(_dispatch(block)) == "Ran `pytest -q`: no tests ran."


def test_failed_block_reports_could_not_execute() -> None:
    block = "assistant: [ran pytest -q (failed)] empty run result"
    verdict = _verdict(_dispatch(block))
    assert "could not execute" in verdict
    assert "empty run result" in verdict


def test_unparseable_output_reports_honestly_with_the_tail() -> None:
    block = "assistant: [ran pytest -q]\n  bash: pytest: command not found"
    verdict = _verdict(_dispatch(block))
    assert "no pytest summary" in verdict
    assert "command not found" in verdict


def test_truncated_block_still_parses_the_tail_summary() -> None:
    block = "assistant: [ran pytest -q (truncated)]\n  ...\n  7 passed in 1.02s"
    assert _verdict(_dispatch(block)) == "Ran `pytest -q`: 7 passed."


def test_latest_run_block_wins() -> None:
    block = (
        "assistant: [ran pytest -q]\n  1 failed in 0.10s\n"
        "assistant: [ran pytest -q]\n  3 passed in 0.09s"
    )
    assert _verdict(_dispatch(block)) == "Ran `pytest -q`: 3 passed."


def test_missing_run_block_reports_honestly() -> None:
    verdict = _verdict("Current request: run the tests")
    assert "No test-run output" in verdict
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/serving/test_serving_run_verdict.py -v`
Expected: FAIL (script does not exist).

- [ ] **Step 3: Implement**

Create `.llm-orc/scripts/agentic_serving/run_verdict.py`:

```python
#!/usr/bin/env python3
"""run-verdict shape — deterministic test-run verdict node (issue #83).

Extracts the latest ``assistant: [ran <command>]`` block from the dispatched
context and composes an honest verdict from pytest's own summary text. Fully
deterministic — a "run the tests" turn costs zero model calls end to end.
The parser reads the block BODY (two-space indented by the caller's render;
the indent is stripped here), so untrusted output text can never be confused
with block headers.
"""

from __future__ import annotations

import json
import re
import sys

_RAN_HEADER_RE = re.compile(
    r"^assistant: \[ran (.+?)( \((failed|truncated)\))?\](.*)$"
)
_NO_TESTS_RE = re.compile(r"\bno tests ran\b", re.IGNORECASE)
_FAILING_LINE_RE = re.compile(r"^(?:FAILED|ERROR)\b")
_MAX_FAILING_LINES = 5
_TAIL_LINES = 10


def _dispatch_text(raw: str) -> str:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if not isinstance(data, dict):
        return str(data)
    inner = data.get("input_data")
    if inner is None:
        inner = data.get("input")
    if inner is None:
        return ""
    return inner if isinstance(inner, str) else json.dumps(inner)


def _latest_run(text: str) -> tuple[str, str, str, str] | None:
    """(command, variant, inline detail, body) of the LAST run block."""
    lines = text.splitlines()
    found: tuple[int, re.Match[str]] | None = None
    for index, line in enumerate(lines):
        match = _RAN_HEADER_RE.match(line)
        if match:
            found = (index, match)
    if found is None:
        return None
    index, match = found
    body_lines: list[str] = []
    for line in lines[index + 1 :]:
        if line.startswith("  "):
            body_lines.append(line[2:])
        elif not line.strip():
            body_lines.append("")
        else:
            break
    command = match.group(1)
    variant = match.group(3) or ""
    detail = (match.group(4) or "").strip()
    return command, variant, detail, "\n".join(body_lines).strip()


def _count(pattern: str, body: str) -> int | None:
    match = re.search(pattern, body)
    return int(match.group(1)) if match else None


def _verdict(command: str, variant: str, detail: str, body: str) -> str:
    if variant == "failed":
        reason = detail or "empty run result"
        return f"The test run could not execute: {reason}"
    if _NO_TESTS_RE.search(body):
        return f"Ran `{command}`: no tests ran."
    failed = _count(r"\b(\d+) failed\b", body)
    passed = _count(r"\b(\d+) passed\b", body)
    errors = _count(r"\b(\d+) errors?\b", body)
    if failed is None and passed is None and errors is None:
        tail = "\n".join(body.splitlines()[-_TAIL_LINES:])
        return (
            f"Ran `{command}`, but the output carried no pytest summary. "
            f"Output tail:\n{tail}"
        )
    parts = [
        f"{count} {label}"
        for count, label in ((failed, "failed"), (errors, "errored"), (passed, "passed"))
        if count
    ]
    verdict = f"Ran `{command}`: {', '.join(parts) or '0 tests'}."
    if failed or errors:
        failing = [
            line for line in body.splitlines() if _FAILING_LINE_RE.match(line)
        ][:_MAX_FAILING_LINES]
        if failing:
            verdict += "\n" + "\n".join(failing)
    return verdict


def main() -> None:
    text = _dispatch_text(sys.stdin.read().strip())
    run = _latest_run(text)
    if run is None:
        primary = "No test-run output found for this turn."
    else:
        primary = _verdict(*run)
    print(json.dumps({"status": "ok", "primary": primary}))


if __name__ == "__main__":
    main()
```

Create `.llm-orc/ensembles/agentic-serving/run-verdict.yaml`:

```yaml
name: run-verdict
description: |
  issue #83 run half — the resumed run turn's dispatch target. One
  deterministic script node parses the [ran ...] block's pytest summary and
  composes the honest verdict; zero model calls.
serves: run-verdict
agents:
  - name: verdict
    script: scripts/agentic_serving/run_verdict.py
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/serving/test_serving_run_verdict.py tests/unit/serving/test_serving_shape_catalog.py -v`
Expected: ALL PASS (add `run-verdict` to the catalog expectation if it enumerates intents).

- [ ] **Step 5: Commit**

```bash
git add .llm-orc/scripts/agentic_serving/run_verdict.py .llm-orc/ensembles/agentic-serving/run-verdict.yaml tests/unit/serving/test_serving_run_verdict.py
git commit -m "feat: run-verdict shape parses pytest summaries deterministically"
```

---

### Task 5: shape and form_gate passthrough

**Files:**
- Modify: `.llm-orc/scripts/agentic_serving/shape.py`, `.llm-orc/scripts/agentic_serving/form_gate.py`
- Test: `tests/unit/serving/test_serving_shape.py`, `tests/unit/serving/test_serving_form_gate.py`

**Interfaces:**
- Consumes: resolve's `needs_run` (Task 3).
- Produces: `needs_run` present in shape's and form_gate's output JSON (default `""`), so emit (Task 6) can branch on it.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/serving/test_serving_shape.py` (follow its existing subprocess helper for composing the `resolve`/`seat` dependencies):

```python
def test_shape_passes_needs_run_from_the_routing_decision() -> None:
    shaped = _shape(
        decision={
            "target": "need-run",
            "kind": "need_run",
            "file": "solution.py",
            "build": False,
            "needs_files": [],
            "read_failed": "",
            "needs_run": "pytest -q",
        },
        seat_terminal='{"status": "ok", "primary": "Requesting a client test run."}',
    )
    assert shaped["needs_run"] == "pytest -q"
```

Append to `tests/unit/serving/test_serving_form_gate.py`:

```python
def test_form_gate_passes_needs_run_through() -> None:
    gated = _form_gate(
        shaped={
            "build": False,
            "file": "solution.py",
            "content": "Requesting a client test run.",
            "needs_files": [],
            "read_failed": "",
            "needs_run": "pytest -q",
            "accept": None,
            "accept_reason": "",
            "seat_admitted": None,
            "seat_contract_reason": "",
        }
    )
    assert gated["needs_run"] == "pytest -q"
    assert gated["valid"] is True
```

Adapt the helper names to each file's existing pattern.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/serving/test_serving_shape.py tests/unit/serving/test_serving_form_gate.py -v`
Expected: new tests FAIL with KeyError.

- [ ] **Step 3: Implement**

In `shape.py`'s printed JSON, after `"read_failed": ...`:

```python
                "needs_run": str(decision.get("needs_run", "")),
```

In `form_gate.py`'s printed JSON, after `"read_failed": ...`:

```python
                "needs_run": str(shaped.get("needs_run", "")),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/serving/test_serving_shape.py tests/unit/serving/test_serving_form_gate.py -v`
Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add .llm-orc/scripts/agentic_serving/shape.py .llm-orc/scripts/agentic_serving/form_gate.py tests/unit/serving/test_serving_shape.py tests/unit/serving/test_serving_form_gate.py
git commit -m "feat: thread the run-request field through the marshal chain"
```

---

### Task 6: emit — the run outcome

**Files:**
- Modify: `.llm-orc/scripts/agentic_serving/emit.py`
- Test: `tests/unit/serving/test_serving_emit.py`

**Interfaces:**
- Consumes: form_gate's `needs_run` (Task 5).
- Produces: `{"finish": false, "run": "<command>"}`. All existing outcomes unchanged.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/serving/test_serving_emit.py`:

```python
def test_needs_run_emits_a_run_outcome() -> None:
    outcome = _emit(
        {
            "build": False,
            "file": "solution.py",
            "content": "Requesting a client test run.",
            "valid": True,
            "reason": "ok",
            "needs_files": [],
            "read_failed": "",
            "needs_run": "pytest -q test_calc.py",
            "accept": None,
            "accept_reason": "",
            "seat_admitted": None,
            "seat_contract_reason": "",
        }
    )
    assert outcome == {"finish": False, "run": "pytest -q test_calc.py"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/serving/test_serving_emit.py -v`
Expected: the new test FAILS (prose finish instead of a run outcome).

- [ ] **Step 3: Implement in `emit.py`**

Read the field alongside the existing ones:

```python
    needs_run = str(gated.get("needs_run", ""))
```

Add the branch after `elif needs_files:` and before `elif seat_admitted is False:`:

```python
    elif needs_run:
        # issue #83 run half: delegate one closed-template test run to the
        # client permission seam.
        outcome = {"finish": False, "run": needs_run}
```

Add the variant to the module docstring's outcome table:

```
    needs run:       {"finish": false, "run": "<command>"}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/serving/test_serving_emit.py -v`
Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add .llm-orc/scripts/agentic_serving/emit.py tests/unit/serving/test_serving_emit.py
git commit -m "feat: emit the client run-delegation serve outcome"
```

---

### Task 7: caller — bash tool_calls out, run continuations back in

**Files:**
- Modify: `src/llm_orc/web/serving/serving_ensemble_caller.py`
- Test: `tests/unit/web/serving/test_serving_context_render.py`

**Interfaces:**
- Consumes: emit's `run` outcome (Task 6); `_run_shaped_arguments` (Task 1); `_client_tool` (read half).
- Produces: `_outcome_chunks(outcome, tools)` mapping `run` to one `ClientToolCall` invocation named from `_BASH_TOOL_CANDIDATES` with arguments `{"command": ..., "description": "Run tests"}`; `_tool_result_ack` returning `None` for run continuations so `run()` falls through to the pipeline.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/web/serving/test_serving_context_render.py`:

```python
def test_run_outcome_maps_to_a_bash_tool_call() -> None:
    tools = [{"type": "function", "function": {"name": "bash"}}]
    chunks = _outcome_chunks({"finish": False, "run": "pytest -q"}, tools)

    assert len(chunks) == 1
    call = chunks[0]
    assert isinstance(call, ClientToolCall)
    assert call.tool_calls[0].name == "bash"
    arguments = json.loads(call.tool_calls[0].arguments)
    assert arguments["command"] == "pytest -q"


def test_run_outcome_resolves_against_advertised_shell_tool() -> None:
    tools = [{"type": "function", "function": {"name": "shell"}}]
    chunks = _outcome_chunks({"finish": False, "run": "pytest -q"}, tools)
    assert chunks[0].tool_calls[0].name == "shell"


def test_run_outcome_falls_back_to_bash_when_nothing_advertised() -> None:
    chunks = _outcome_chunks({"finish": False, "run": "pytest -q"}, [])
    assert chunks[0].tool_calls[0].name == "bash"


def test_run_continuation_is_not_acked() -> None:
    messages = [
        ChatMessage(role="user", content="run the tests"),
        ChatMessage(
            role="assistant", content=None, tool_calls=(_bash_call("c1", "pytest -q"),)
        ),
        ChatMessage(role="tool", tool_call_id="c1", content="2 passed in 0.01s"),
    ]
    assert _tool_result_ack(messages) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/web/serving/test_serving_context_render.py -v`
Expected: the four new tests FAIL (no run mapping; run continuation acked "Done.").

- [ ] **Step 3: Implement in `serving_ensemble_caller.py`**

Add near `_READ_TOOL_CANDIDATES`:

```python
_BASH_TOOL = "bash"
_BASH_TOOL_CANDIDATES = ("bash", "shell", "terminal", "Bash")
```

In `_outcome_chunks`, after the `reads` branch and before the write mapping:

```python
    run = outcome.get("run")
    if run:
        invocation = ToolCallInvocation(
            id=f"call_{uuid.uuid4().hex[:8]}",
            name=_client_tool(tools, _BASH_TOOL_CANDIDATES, _BASH_TOOL),
            arguments=json.dumps(
                {"command": str(run), "description": "Run tests"}
            ),
        )
        return [ClientToolCall(tool_calls=(invocation,))]
```

In `_tool_result_ack`'s walk-back loop, extend the read check to cover run-shaped calls:

```python
        for call in getattr(message, "tool_calls", ()) or ():
            if (
                _read_shaped_arguments(call) is not None
                or _run_shaped_arguments(call) is not None
            ):
                # issue #83: read and run continuations RESUME the turn —
                # fall through to the pipeline with the result in context.
                return None
```

Update `_tool_result_ack`'s docstring: read AND run continuations resume; write continuations still ack.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/web/serving/test_serving_context_render.py tests/unit/web/test_serving_ensemble_endpoint.py -v`
Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add src/llm_orc/web/serving/serving_ensemble_caller.py tests/unit/web/serving/test_serving_context_render.py
git commit -m "feat: caller emits client run calls and resumes run continuations"
```

---

### Task 8: hermetic end-to-end — the two-pass run turn through the real engine

**Files:**
- Test: `tests/unit/web/test_serving_ensemble_endpoint.py`

**Interfaces:**
- Consumes: everything above, through the real serving ensemble on the L0 engine (the file's `serving_project` fixture: real `serving.yaml` + scripts, echo seats shadowing model nodes — zero model tokens).
- Produces: three endpoint-level proofs: pass 1 emits the bash tool_call with the closed command; pass 2 ships the deterministic verdict; a failing suite's verdict carries the failure lines.

- [ ] **Step 1: Extend the fixture and add a bash tool**

In `tests/unit/web/test_serving_ensemble_endpoint.py`, add next to `_READ_TOOL_DEF`:

```python
_BASH_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": ["command"],
        },
    },
}
```

In the `serving_project` fixture, after the `need-files.yaml` copy:

```python
    shutil.copy(REAL_AGENTIC_SERVING / "need-run.yaml", ensembles / "need-run.yaml")
    shutil.copy(
        REAL_AGENTIC_SERVING / "run-verdict.yaml", ensembles / "run-verdict.yaml"
    )
```

- [ ] **Step 2: Write the three failing tests**

```python
def test_run_turn_emits_a_bash_tool_call_with_the_closed_command(
    serving_client: TestClient,
) -> None:
    """Pass 1 (issue #83 run half): a run turn delegates one deterministic
    pytest command through the permission seam."""
    resp = serving_client.post(
        "/v1/chat/completions",
        json={
            "model": "ensemble-agent",
            "messages": [{"role": "user", "content": "run test_calc.py"}],
            "tools": [_WRITE_TOOL, _READ_TOOL_DEF, _BASH_TOOL_DEF],
        },
    )

    assert resp.status_code == 200
    choice = resp.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    call = choice["message"]["tool_calls"][0]
    assert call["function"]["name"] == "bash"
    arguments = json.loads(call["function"]["arguments"])
    assert arguments["command"] == "pytest -q test_calc.py"


def test_run_continuation_ships_the_deterministic_verdict(
    serving_client: TestClient,
) -> None:
    """Pass 2 (issue #83 run half): the bash result re-enters the pipeline
    and the run-verdict shape replies with pytest's own summary."""
    resp = serving_client.post(
        "/v1/chat/completions",
        json={
            "model": "ensemble-agent",
            "messages": [
                {"role": "user", "content": "run the tests"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_b1",
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": (
                                    '{"command": "pytest -q",'
                                    ' "description": "Run tests"}'
                                ),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_b1",
                    "content": ".....\n5 passed in 0.12s",
                },
            ],
            "tools": [_WRITE_TOOL, _READ_TOOL_DEF, _BASH_TOOL_DEF],
        },
    )

    assert resp.status_code == 200
    choice = resp.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert not choice["message"].get("tool_calls")
    assert choice["message"]["content"] == "Ran `pytest -q`: 5 passed."


def test_failing_run_verdict_carries_the_failure_lines(
    serving_client: TestClient,
) -> None:
    """An honest red verdict: counts from pytest's summary plus the FAILED
    lines — never a silent success, never a second run request."""
    resp = serving_client.post(
        "/v1/chat/completions",
        json={
            "model": "ensemble-agent",
            "messages": [
                {"role": "user", "content": "run the tests"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_b1",
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": '{"command": "pytest -q"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_b1",
                    "content": (
                        "..F\n"
                        "FAILED test_calc.py::test_divide - ZeroDivisionError\n"
                        "1 failed, 2 passed in 0.05s"
                    ),
                },
            ],
            "tools": [_WRITE_TOOL, _READ_TOOL_DEF, _BASH_TOOL_DEF],
        },
    )

    assert resp.status_code == 200
    choice = resp.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    content = choice["message"]["content"]
    assert content.startswith("Ran `pytest -q`: 1 failed, 2 passed.")
    assert "FAILED test_calc.py::test_divide" in content
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `uv run pytest tests/unit/web/test_serving_ensemble_endpoint.py -v`
Expected: with Tasks 1–7 complete these PASS (this task is the wiring proof, not new behavior). If one fails, debug the seam it names before proceeding.

- [ ] **Step 4: Run the full suite and lint**

Run: `make test && make lint`
Expected: ALL PASS, no warnings.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/web/test_serving_ensemble_endpoint.py
git commit -m "test: hermetic two-pass client-run turn through the real engine"
```

---

### Task 9: live validation against real OpenCode + docs

This task is evidence work — expect iteration, and keep the wire log.

**Files:**
- Possibly modify: `src/llm_orc/web/serving/serving_ensemble_caller.py` (`_render_run_block`, `_BASH_TOOL_CANDIDATES`) once the real bash-result format is observed
- Modify: `docs/serving-roadmap.md` (trajectory table, path item 2), `docs/serving.md` (capability coverage)

- [ ] **Step 1: Wire capture.** Start the serve with `LLM_ORC_SERVE_WIRE_LOG` on. In a scratch git repo with a real module + `test_*.py` suite, drive real OpenCode: first a build turn, then `run the tests`.
- [ ] **Step 2: Verify pass 1 on the wire.** The serve's response is a bash tool_call with the closed command (turn trace at `.llm-orc/.serve-trace/`); OpenCode surfaces its permission prompt.
- [ ] **Step 3: Verify the bash-result format.** Inspect the continuation's `role: tool` message. If OpenCode wraps bash output (tags, exit-code trailer, gutter), adjust `_render_run_block`/a new normalizer and its unit tests to the observed format; commit as `fix: normalize OpenCode bash output per captured wire`. Confirm the advertised bash tool name and required arguments; adjust `_BASH_TOOL_CANDIDATES`/the arguments dict if needed.
- [ ] **Step 4: Verify pass 2 end-to-end.** The resumed turn ships the deterministic verdict matching the actual pytest result (green and red both).
- [ ] **Step 5: Ladder rerun.** Add a run rung to `benchmarks/agentic_serving/ladder_battery.sh`; run the full recorded ladder; update the trajectory table in `docs/serving-roadmap.md` with date, battery, score, notes.
- [ ] **Step 6: Docs.** Update `docs/serving.md` capability coverage (run delegation shipped; discovery still open) and `docs/serving-roadmap.md` path item 2. Commit as `docs: record client-run capability and ladder results`.
- [ ] **Step 7: Issue note.** Comment the evidence on #83 (`gh issue comment 83`): what shipped, wire observations, the remaining discovery half. Do not close the issue.

---

## Execution order and dependencies

Tasks 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9, strictly. Task 1 defines the block grammar Tasks 2 and 4 parse; Tasks 2–6 thread the routing field front to back; Task 7 needs Task 1's `_run_shaped_arguments`; Task 8 needs everything; Task 9 needs a green Task 8.

## Out of scope (named forward directions, per the spec)

- Discovery (list/glob for files the turn doesn't name) — next #83 widening, own design pass.
- Chained fix-execution (write → run → verdict in one turn) — the fix shape.
- Per-language runners (`cargo test` for the plexus/Rust half is the named first target — the command builder and verdict parser are the only pytest-aware seams).
- Escalation-on-red (a failed run feeding the next build round).
- Non-pytest Python runners, repo-layout guessing (`calc.py` → `test_calc.py`).

# Client-File Reads Implementation Plan (#83 rung 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The serve requests client-workspace files it can't see via a `read` tool_call and resumes the turn statelessly when the result comes back, fixing "write tests for existing foo.py" rejects.

**Architecture:** Two-pass turn through the existing permission seam. Pass 1: classify detects a named-but-invisible file, routes to a new script-only `need-files` shape, emit ships `{"finish": false, "reads": [...]}`, the caller maps it to read tool_call(s). Pass 2: the client calls back with the tool result appended; the caller's continuation detector lets read continuations re-enter the pipeline (write continuations still ack); the renderer turns read results into `[read <path>]` context blocks that classify sees as visible and gather materializes into the sandbox. No server-side state — everything derives from the append-only wire.

**Tech Stack:** Python 3.13, pytest, the L0 ensemble engine, script nodes under `.llm-orc/scripts/agentic_serving/`.

**Spec:** `docs/plans/2026-07-09-client-file-reads-design.md`

## Global Constraints

- ruff (88 chars max) and mypy strict compliant from first draft; run `make lint` before every commit.
- TDD: write the failing test, see it fail, implement, see it pass, commit. One behavioral unit per commit.
- Never mix structural and behavioral changes in a commit. No AI attribution in commit messages.
- Commit prefixes: `feat:`, `test:`, `fix:`, `refactor:`, `docs:`.
- Scripts under `.llm-orc/scripts/agentic_serving/` are run as subprocesses by the engine; their unit tests drive them via subprocess (see `tests/unit/serving/test_serving_classify.py` for the pattern).
- Full suite: `make test`. Targeted: `uv run pytest <path> -v`.

## Shared vocabulary (used by every task)

- **Read block grammar** in the rendered context (parallel to `assistant: [wrote <path>]`):
  - success: `assistant: [read <path>]` followed by the normalized file body
  - failure: `assistant: [read <path> (failed)] <flat reason, single line>`
  - oversize: `assistant: [read <path> (oversize)]` (single line, no body)
- **classify output** gains two always-present keys: `needs_files: list[str]` (paths to request; `[]` when none) and `read_failed: str` (refusal reason; `""` when none).
- **emit outcome** gains two variants: `{"finish": false, "reads": ["<path>", ...]}` and `{"finish": true, "content": "Refused: could not read <path>: <reason>"}`.
- **Read-shaped tool call**: arguments parse to a dict containing `filePath` and NOT containing `content`. Write-shaped: both keys. This is the client-agnostic discriminator used everywhere.
- **`_READ_FILE_CAP` = 24576** chars per read file (spec bound).

---

### Task 1: Renderer — read blocks in the conversation context

**Files:**
- Modify: `src/llm_orc/web/serving/serving_ensemble_caller.py`
- Test: `tests/unit/web/serving/test_serving_context_render.py`

**Interfaces:**
- Consumes: existing `_render_context(messages)`, `_render_write(message)`, `_select_written_files`.
- Produces: `_render_context` output containing `assistant: [read <path>]` blocks (grammar above) whenever the history holds a tool result answering a read-shaped tool_call. `_render_write` and `_written_file_path` tightened to require a `content` key (a read call must never render as an empty write). Internal helpers later tasks rely on: `_read_call_paths(messages) -> dict[str, str]` (tool_call_id → filePath for read-shaped calls), `_normalize_read(content: str) -> str`, `_read_blocks(messages) -> list[tuple[str, str]]` ((path, block) in wire order).

Read results live in `role: tool` messages *after* the last user message on pass 2, so read blocks must be selected from the FULL message list, not the pre-latest-user slice the write selection uses today.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/web/serving/test_serving_context_render.py`:

```python
def _read_call(call_id: str, path: str) -> dict[str, object]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "read", "arguments": f'{{"filePath": "{path}"}}'},
    }


def test_read_result_renders_as_read_block() -> None:
    messages = [
        ChatMessage(role="user", content="write tests for existing calc.py"),
        ChatMessage(
            role="assistant", content=None, tool_calls=(_read_call("c1", "calc.py"),)
        ),
        ChatMessage(
            role="tool", tool_call_id="c1", content="def divide(a, b): return a / b"
        ),
    ]

    rendered = _render_context(messages)

    assert "[read calc.py]" in rendered
    assert "def divide" in rendered


def test_read_call_never_renders_as_an_empty_write_block() -> None:
    messages = [
        ChatMessage(role="user", content="fix calc.py"),
        ChatMessage(
            role="assistant", content=None, tool_calls=(_read_call("c1", "calc.py"),)
        ),
        ChatMessage(role="tool", tool_call_id="c1", content="def divide(a): return a"),
        ChatMessage(role="user", content="thanks, now fix the docstring"),
    ]

    rendered = _render_context(messages)

    assert "[wrote calc.py]" not in rendered
    assert "[read calc.py]" in rendered


def test_empty_read_result_renders_as_failed_single_line() -> None:
    messages = [
        ChatMessage(role="user", content="fix calc.py"),
        ChatMessage(
            role="assistant", content=None, tool_calls=(_read_call("c1", "calc.py"),)
        ),
        ChatMessage(role="tool", tool_call_id="c1", content=""),
    ]

    rendered = _render_context(messages)

    assert "[read calc.py (failed)]" in rendered


def test_error_read_result_renders_as_failed_single_line() -> None:
    messages = [
        ChatMessage(role="user", content="fix calc.py"),
        ChatMessage(
            role="assistant", content=None, tool_calls=(_read_call("c1", "calc.py"),)
        ),
        ChatMessage(role="tool", tool_call_id="c1", content="Error: ENOENT calc.py"),
    ]

    rendered = _render_context(messages)

    assert "[read calc.py (failed)] Error: ENOENT calc.py" in rendered


def test_oversize_read_result_renders_header_only() -> None:
    from llm_orc.web.serving.serving_ensemble_caller import _READ_FILE_CAP

    messages = [
        ChatMessage(role="user", content="fix calc.py"),
        ChatMessage(
            role="assistant", content=None, tool_calls=(_read_call("c1", "calc.py"),)
        ),
        ChatMessage(role="tool", tool_call_id="c1", content="x" * (_READ_FILE_CAP + 1)),
    ]

    rendered = _render_context(messages)

    assert "[read calc.py (oversize)]" in rendered
    assert "xxxx" not in rendered


def test_line_number_gutter_is_stripped_from_read_content() -> None:
    body = "00001| def divide(a, b):\n00002|     return a / b"
    messages = [
        ChatMessage(role="user", content="fix calc.py"),
        ChatMessage(
            role="assistant", content=None, tool_calls=(_read_call("c1", "calc.py"),)
        ),
        ChatMessage(role="tool", tool_call_id="c1", content=body),
    ]

    rendered = _render_context(messages)

    assert "def divide(a, b):" in rendered
    assert "    return a / b" in rendered
    assert "00001|" not in rendered


def test_later_write_of_same_path_supersedes_earlier_read() -> None:
    messages = [
        ChatMessage(role="user", content="fix calc.py"),
        ChatMessage(
            role="assistant", content=None, tool_calls=(_read_call("c1", "calc.py"),)
        ),
        ChatMessage(role="tool", tool_call_id="c1", content="def old(): pass"),
        ChatMessage(
            role="assistant",
            content=None,
            tool_calls=(_write_call("calc.py", "def new(): pass"),),
        ),
        ChatMessage(role="tool", content="Wrote file successfully."),
        ChatMessage(role="user", content="now add tests"),
    ]

    rendered = _render_context(messages)

    assert "def new(): pass" in rendered
    assert "def old(): pass" not in rendered
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/web/serving/test_serving_context_render.py -v`
Expected: the seven new tests FAIL (no `[read ...]` rendering, `_READ_FILE_CAP` ImportError); existing tests PASS.

- [ ] **Step 3: Implement in `serving_ensemble_caller.py`**

Add near the other module caps:

```python
# Client-read file blocks (issue #83): whole-file-or-refuse — a truncated
# module fails imports in the sandbox, so an over-cap read refuses honestly
# instead of materializing a corrupted file.
_READ_FILE_CAP = 24576
_READ_FAIL_REASON_CAP = 200
# OpenCode's read tool returns a line-number gutter ("00001| ..."); strip it
# only when every non-empty line carries one (assumption to verify against
# the captured wire — see the live-validation task).
_LINE_NUM_GUTTER_RE = re.compile(r"^\s*\d+\| ?")
```

Add the helpers (module level, near `_render_write`):

```python
def _read_shaped_arguments(call: Any) -> dict[str, Any] | None:
    """Parsed arguments of a read-shaped tool call (filePath, no content)."""
    function = call.get("function", {}) if isinstance(call, dict) else {}
    try:
        arguments = json.loads(function.get("arguments", ""))
    except (json.JSONDecodeError, TypeError):
        return None
    if (
        isinstance(arguments, dict)
        and arguments.get("filePath")
        and "content" not in arguments
    ):
        return arguments
    return None


def _read_call_paths(messages: Sequence[Any]) -> dict[str, str]:
    """tool_call_id -> filePath for every read-shaped call in the history."""
    paths: dict[str, str] = {}
    for message in messages:
        for call in getattr(message, "tool_calls", ()) or ():
            arguments = _read_shaped_arguments(call)
            if arguments is not None and isinstance(call, dict) and call.get("id"):
                paths[str(call["id"])] = str(arguments["filePath"])
    return paths


def _normalize_read(content: str) -> str:
    """Client read output as plain source: strip a <file> wrapper and a
    uniform line-number gutter when present."""
    lines = content.strip().splitlines()
    if lines and lines[0].strip() == "<file>":
        lines = lines[1:]
    if lines and lines[-1].strip() == "</file>":
        lines = lines[:-1]
    non_empty = [line for line in lines if line.strip()]
    if non_empty and all(_LINE_NUM_GUTTER_RE.match(line) for line in non_empty):
        lines = [_LINE_NUM_GUTTER_RE.sub("", line, count=1) for line in lines]
    return "\n".join(lines).strip()


def _render_read_block(path: str, raw: str) -> str:
    """A read result as a context block (issue #83 grammar). Failure and
    oversize variants are single header lines so gather never materializes
    them and classify can refuse instead of re-requesting (one-round bound)."""
    flat = " ".join((raw or "").strip().split())
    if not flat or flat.lower().startswith("error"):
        reason = flat[:_READ_FAIL_REASON_CAP] or "empty read result"
        return f"assistant: [read {path} (failed)] {reason}"
    normalized = _normalize_read(raw)
    if len(normalized) > _READ_FILE_CAP:
        return f"assistant: [read {path} (oversize)]"
    return f"assistant: [read {path}]\n{normalized}"


def _read_blocks(messages: Sequence[Any]) -> list[tuple[str, str]]:
    """(path, block) for every tool result answering a read-shaped call,
    in wire order. Selected from the FULL history: on the resume pass the
    read result sits after the last user message."""
    call_paths = _read_call_paths(messages)
    blocks: list[tuple[str, str]] = []
    for message in messages:
        if getattr(message, "role", None) != "tool":
            continue
        path = call_paths.get(getattr(message, "tool_call_id", None) or "")
        if path:
            content = getattr(message, "content", None)
            blocks.append((path, _render_read_block(path, content or "")))
    return blocks
```

Tighten `_render_write` — a read call carries `filePath` but no `content` and must not render as an empty write. Change its condition:

```python
        if (
            isinstance(arguments, dict)
            and arguments.get("filePath")
            and "content" in arguments
        ):
```

Apply the same tightening to `_written_file_path`:

```python
        if (
            isinstance(arguments, dict)
            and arguments.get("filePath")
            and "content" in arguments
        ):
            return str(arguments["filePath"])
```

Wire read blocks into `_render_context`. Replace the selection paragraph (the lines from `selected = _select_written_files(...)` through `rendered = f"{selected_text}..."`) with:

```python
    selected = _select_written_files(conversational, task)
    tail_paths = {
        line.split("[wrote ", 1)[1].split("]", 1)[0].removesuffix(" (truncated)")
        for line in rendered.splitlines()
        if line.startswith("assistant: [wrote ")
    }
    write_blocks = [block for path, block in selected if path not in tail_paths]
    kept = _whole_blocks_within_cap(write_blocks)

    # Read blocks (issue #83) join from the FULL history, exempt from the
    # selected-block cap: dropping one would make classify re-request it
    # (a read loop). Latest block per path wins; a later write supersedes.
    written_paths = {path for path, _ in _select_written_files(list(messages), task)}
    latest_reads: dict[str, str] = {}
    for path, block in _read_blocks(messages):
        if path not in written_paths and path not in tail_paths:
            latest_reads[path] = block
    kept = list(latest_reads.values()) + kept

    if kept:
        selected_text = "\n".join(kept)
        rendered = f"{selected_text}\n{rendered}" if rendered else selected_text
    return rendered
```

Note the supersede rule: `_select_written_files` over the full `messages` list yields every conversation-written path; a read block for a path that was later (or earlier) written is dropped in favor of the write block — the write is the serve's own latest content. `_select_written_files` already tolerates tool rows (its `_render_write` returns None for them), so passing the full list is safe.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/web/serving/test_serving_context_render.py tests/unit/web/test_serving_ensemble_endpoint.py -v`
Expected: ALL PASS (endpoint tests guard the write path against the `_render_write` tightening).

- [ ] **Step 5: Commit**

```bash
git add src/llm_orc/web/serving/serving_ensemble_caller.py tests/unit/web/serving/test_serving_context_render.py
git commit -m "feat: render client read results as [read] context blocks"
```

---

### Task 2: classify — named-but-invisible files route to need-files

**Files:**
- Modify: `.llm-orc/scripts/agentic_serving/classify.py`
- Test: `tests/unit/serving/test_serving_classify.py`

**Interfaces:**
- Consumes: the read-block grammar from Task 1 (via the `context` string the caller threads into the turn).
- Produces: classify output with `needs_files: list[str]` and `read_failed: str` always present. When a read is needed: `target: "need-files"`, `kind: "need_files"`, `build: False`, `needs_files: [<paths>]`. When a read was attempted and still isn't visible: `target: "need-files"`, `read_failed: "could not read <path>: <detail>"`, `needs_files: []`.

Trigger (deterministic, rung-1): the turn names at least one source file (basename not starting `test_`) that is not visible in the context, AND the turn is either tests-primary ("write tests for storage.py") or a build turn with an existing-file marker (`fix|update|modify|refactor|edit|change|existing`). Fresh creates ("write a function ... in add.py") never trigger. Explain turns never trigger. Visible = an untruncated `[wrote <path>]` block or a successful `[read <path>]` block whose basename matches. Attempted = any `[read <path> ...]` header (success, failed, or oversize) whose basename matches.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/serving/test_serving_classify.py`:

```python
def test_tests_for_invisible_named_file_requests_a_client_read() -> None:
    decision = _classify({"task": "write tests for existing storage.py"})
    assert decision["target"] == "need-files"
    assert decision["kind"] == "need_files"
    assert decision["build"] is False
    assert decision["needs_files"] == ["storage.py"]
    assert decision["read_failed"] == ""


def test_existing_marker_build_on_invisible_file_requests_a_client_read() -> None:
    decision = _classify({"task": "fix the divide function in calc.py"})
    assert decision["target"] == "need-files"
    assert decision["needs_files"] == ["calc.py"]


def test_fresh_create_never_requests_a_read() -> None:
    decision = _classify(
        {"task": "write a function that adds two numbers in add.py"}
    )
    assert decision["target"] == "code-seat"
    assert decision["needs_files"] == []


def test_visible_wrote_block_suppresses_the_read_request() -> None:
    context = "assistant: [wrote storage.py]\ndef put(k, v): pass"
    decision = _classify(
        {"task": "write tests for existing storage.py", "context": context}
    )
    assert decision["target"] == "tests-seat"
    assert decision["needs_files"] == []


def test_visible_read_block_suppresses_the_read_request() -> None:
    context = "assistant: [read storage.py]\ndef put(k, v): pass"
    decision = _classify(
        {"task": "write tests for existing storage.py", "context": context}
    )
    assert decision["target"] == "tests-seat"
    assert decision["needs_files"] == []


def test_truncated_wrote_block_still_requests_a_read() -> None:
    context = "assistant: [wrote storage.py (truncated)]\ndef put(k"
    decision = _classify(
        {"task": "write tests for existing storage.py", "context": context}
    )
    assert decision["target"] == "need-files"
    assert decision["needs_files"] == ["storage.py"]


def test_failed_read_attempt_refuses_instead_of_relooping() -> None:
    context = "assistant: [read storage.py (failed)] Error: ENOENT"
    decision = _classify(
        {"task": "write tests for existing storage.py", "context": context}
    )
    assert decision["target"] == "need-files"
    assert decision["needs_files"] == []
    assert "could not read storage.py" in decision["read_failed"]


def test_oversize_read_attempt_refuses_with_cap_reason() -> None:
    context = "assistant: [read storage.py (oversize)]"
    decision = _classify(
        {"task": "write tests for existing storage.py", "context": context}
    )
    assert decision["needs_files"] == []
    assert "could not read storage.py" in decision["read_failed"]
    assert "24" in decision["read_failed"]


def test_explain_turn_never_requests_a_read() -> None:
    decision = _classify({"task": "explain what storage.py does"})
    assert decision["target"] == "explainer"
    assert decision["needs_files"] == []


def test_normal_decisions_carry_empty_read_fields() -> None:
    decision = _classify({"task": "write a function that adds two numbers"})
    assert decision["needs_files"] == []
    assert decision["read_failed"] == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/serving/test_serving_classify.py -v`
Expected: new tests FAIL (KeyError `needs_files` / wrong target); existing tests PASS.

- [ ] **Step 3: Implement in `classify.py`**

Add module-level patterns after `_BUILD_RE`:

```python
# issue #83: a build verb that implies the named file already exists in the
# client workspace. "write"/"create" stay fresh-create — requesting a read
# for a file that does not exist yet would refuse a valid build.
_EXISTING_RE = re.compile(
    r"\b(fix|update|modify|refactor|edit|change|existing)\b", re.IGNORECASE
)
# Context-block headers (the caller's render grammar). Visible = untruncated
# wrote block or successful read block; attempted = any read header. The
# optional variant group keeps a "(truncated)" suffix out of the path.
_VISIBLE_HEADER_RE = re.compile(
    r"^assistant: \[(?:wrote|read) ([^\]]+?)"
    r"( \((?:truncated|failed|oversize)\))?\]$",
    re.MULTILINE,
)
_READ_ATTEMPT_RE = re.compile(
    r"^assistant: \[read ([^\]]+?)( \((failed|oversize)\))?\]", re.MULTILINE
)
_READ_CAP_KB = 24
```

Add helpers after `_extract_file`:

```python
def _named_source_files(task: str) -> list[str]:
    """Every named non-test source file, first-mention order, deduped."""
    files: list[str] = []
    for match in _FILE_RE.finditer(task):
        path = match.group(1)
        if path.rsplit("/", 1)[-1].startswith("test_"):
            continue
        if path not in files:
            files.append(path)
    return files


def _visibility(context: str) -> tuple[set[str], dict[str, str]]:
    """(visible basenames, attempted basename -> failure detail)."""
    visible = {
        path.rsplit("/", 1)[-1]
        for path, variant in _VISIBLE_HEADER_RE.findall(context)
        if not variant
    }
    attempted: dict[str, str] = {}
    for path, _, variant in _READ_ATTEMPT_RE.findall(context):
        basename = path.rsplit("/", 1)[-1]
        if variant == "oversize":
            attempted[basename] = f"file exceeds the {_READ_CAP_KB} KB read cap"
        elif variant == "failed":
            attempted[basename] = "client read failed"
    return visible, attempted


def _files_to_request(
    task: str, context: str, tests_primary: bool, has_build_signal: bool
) -> tuple[list[str], str]:
    """(paths to request, refusal reason) — at most one is non-empty.

    Deterministic one-round control (issue #83): a named source file that is
    neither conversation-written nor client-read triggers ONE read request;
    a file whose read was already attempted and still is not visible refuses.
    """
    wants_existing = tests_primary or (
        has_build_signal and bool(_EXISTING_RE.search(task))
    )
    if not wants_existing:
        return [], ""
    visible, attempted = _visibility(context)
    to_request: list[str] = []
    for path in _named_source_files(task):
        basename = path.rsplit("/", 1)[-1]
        if basename in visible:
            continue
        if basename in attempted:
            return [], f"could not read {path}: {attempted[basename]}"
        to_request.append(path)
    return to_request, ""
```

In `main()`, after `tests_primary` is computed and before the `if is_explain:` routing chain, add:

```python
    conversation_raw = str(turn.get("context", ""))
    needs_files: list[str] = []
    read_failed = ""
    if not is_explain:
        needs_files, read_failed = _files_to_request(
            task, conversation_raw, tests_primary, has_build_signal
        )
```

Then extend the routing chain — the need-files branch outranks the seat branches (a seat without the file rejects anyway):

```python
    if is_explain:
        target, kind, build, needs_decider = _EXPLAIN_SEAT, "explanation", False, False
    elif needs_files or read_failed:
        # issue #83: request the client files (or refuse a failed request)
        # before any seat runs — the need-files shape is a cheap script echo.
        target, kind, build, needs_decider = "need-files", "need_files", False, False
    elif tests_primary:
        ...
```

(the `elif tests_primary:` / `elif has_build_signal:` / `else:` branches are unchanged.)

Finally add both keys to the printed JSON object:

```python
                "needs_files": needs_files,
                "read_failed": read_failed,
```

Note `conversation_raw` reuses the same value the existing `conversation = str(turn.get("context", "")).strip()` line reads — leave that line as is.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/serving/test_serving_classify.py -v`
Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add .llm-orc/scripts/agentic_serving/classify.py tests/unit/serving/test_serving_classify.py
git commit -m "feat: classify routes named-but-invisible files to need-files"
```

---

### Task 3: resolve passthrough + the need-files echo shape

**Files:**
- Modify: `.llm-orc/scripts/agentic_serving/resolve.py`
- Create: `.llm-orc/ensembles/agentic-serving/need-files.yaml`
- Create: `.llm-orc/scripts/agentic_serving/need_files_echo.py`
- Test: `tests/unit/serving/test_serving_resolve.py`

**Interfaces:**
- Consumes: classify's `needs_files` / `read_failed` (Task 2).
- Produces: resolve output carrying `needs_files: list[str]` and `read_failed: str` through (empty defaults on the decider path). The `need-files` ensemble is dispatchable: catalog maps the `need-files` intent to it via its `serves:` declaration; its single script node emits a minimal ADR-024 envelope.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/serving/test_serving_resolve.py` (follow the file's existing `_resolve(...)` subprocess helper):

```python
def test_needs_files_and_read_failed_pass_through_resolve() -> None:
    routing = _resolve(
        classify={
            "target": "need-files",
            "kind": "need_files",
            "file": "test_storage.py",
            "dispatch_input": "write tests for existing storage.py",
            "build": False,
            "needs_decider": False,
            "needs_files": ["storage.py"],
            "read_failed": "",
        }
    )
    assert routing["target"] == "need-files"
    assert routing["needs_files"] == ["storage.py"]
    assert routing["read_failed"] == ""


def test_decider_path_defaults_read_fields_empty() -> None:
    routing = _resolve(
        classify={
            "target": "",
            "kind": "",
            "file": "solution.py",
            "dispatch_input": "coverage for the storage module",
            "build": False,
            "needs_decider": True,
        },
        decide='{"target": "tests-seat"}',
    )
    assert routing["needs_files"] == []
    assert routing["read_failed"] == ""
```

If the file's `_resolve` helper doesn't accept a `decide` argument, extend it the way its neighbors compose the dependencies dict — the script reads `{"dependencies": {"classify": {"response": <json>}, "decide": {"response": <text>}}}` on stdin.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/serving/test_serving_resolve.py -v`
Expected: new tests FAIL with KeyError `needs_files`.

- [ ] **Step 3: Implement**

In `resolve.py` `main()`, after `dispatch_input = classify.get(...)`:

```python
    needs_files = classify.get("needs_files", [])
    if not isinstance(needs_files, list):
        needs_files = []
    read_failed = str(classify.get("read_failed", ""))
```

and add to the printed JSON object:

```python
                "needs_files": needs_files,
                "read_failed": read_failed,
```

Create `.llm-orc/ensembles/agentic-serving/need-files.yaml`:

```yaml
name: need-files
description: |
  issue #83 — the cheap dispatch target for a turn that must first request
  client-workspace files. The routing decision (classify) already carries the
  reads; this shape exists so the skeleton dispatches SOMETHING without
  burning a build round. One deterministic script node, no model calls.
serves: need-files
agents:
  - name: echo
    script: scripts/agentic_serving/need_files_echo.py
```

Create `.llm-orc/scripts/agentic_serving/need_files_echo.py`:

```python
#!/usr/bin/env python3
"""need-files shape — deterministic echo node (issue #83).

The reads request rides the ROUTING decision (classify -> resolve -> shape ->
form_gate -> emit), not the seat envelope; this node only satisfies the
skeleton's dispatch step with a minimal ADR-024 envelope, at zero model cost.
"""

from __future__ import annotations

import json
import sys


def main() -> None:
    sys.stdin.read()
    print(json.dumps({"status": "ok", "primary": "Requesting client files."}))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests, verify the shape validates**

Run: `uv run pytest tests/unit/serving/test_serving_resolve.py tests/unit/serving/test_serving_shape_catalog.py -v`
Expected: PASS. If the shape-catalog test enumerates expected intents, add `need-files` to its expectation.

- [ ] **Step 5: Commit**

```bash
git add .llm-orc/scripts/agentic_serving/resolve.py .llm-orc/ensembles/agentic-serving/need-files.yaml .llm-orc/scripts/agentic_serving/need_files_echo.py tests/unit/serving/test_serving_resolve.py
git commit -m "feat: need-files shape and resolve passthrough for client reads"
```

---

### Task 4: shape and form_gate passthrough

**Files:**
- Modify: `.llm-orc/scripts/agentic_serving/shape.py`, `.llm-orc/scripts/agentic_serving/form_gate.py`
- Test: `tests/unit/serving/test_serving_shape.py`, `tests/unit/serving/test_serving_form_gate.py`

**Interfaces:**
- Consumes: resolve's `needs_files` / `read_failed` (Task 3).
- Produces: both fields present in shape's and form_gate's output JSON (defaults `[]` / `""`), so emit (Task 5) can branch on them.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/serving/test_serving_shape.py` (follow its existing subprocess helper for composing the `resolve`/`seat` dependencies):

```python
def test_shape_passes_read_fields_from_the_routing_decision() -> None:
    shaped = _shape(
        decision={
            "target": "need-files",
            "kind": "need_files",
            "file": "test_storage.py",
            "build": False,
            "needs_files": ["storage.py"],
            "read_failed": "",
        },
        seat_terminal='{"status": "ok", "primary": "Requesting client files."}',
    )
    assert shaped["needs_files"] == ["storage.py"]
    assert shaped["read_failed"] == ""
```

Append to `tests/unit/serving/test_serving_form_gate.py`:

```python
def test_form_gate_passes_read_fields_through() -> None:
    gated = _form_gate(
        shaped={
            "build": False,
            "file": "test_storage.py",
            "content": "Requesting client files.",
            "needs_files": ["storage.py"],
            "read_failed": "",
            "accept": None,
            "accept_reason": "",
            "seat_admitted": None,
            "seat_contract_reason": "",
        }
    )
    assert gated["needs_files"] == ["storage.py"]
    assert gated["read_failed"] == ""
    assert gated["valid"] is True
```

Adapt the helper names to each file's existing pattern.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/serving/test_serving_shape.py tests/unit/serving/test_serving_form_gate.py -v`
Expected: new tests FAIL with KeyError.

- [ ] **Step 3: Implement**

In `shape.py`'s printed JSON, after `"seat_contract_reason": ...`:

```python
                # issue #83: the reads request rides the routing decision
                "needs_files": decision.get("needs_files", []),
                "read_failed": str(decision.get("read_failed", "")),
```

In `form_gate.py`'s printed JSON, after the seat-contract passthrough:

```python
                "needs_files": shaped.get("needs_files", []),
                "read_failed": str(shaped.get("read_failed", "")),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/serving/test_serving_shape.py tests/unit/serving/test_serving_form_gate.py -v`
Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add .llm-orc/scripts/agentic_serving/shape.py .llm-orc/scripts/agentic_serving/form_gate.py tests/unit/serving/test_serving_shape.py tests/unit/serving/test_serving_form_gate.py
git commit -m "feat: thread read-request fields through the marshal chain"
```

---

### Task 5: emit — the reads outcome and the read-failure refusal

**Files:**
- Modify: `.llm-orc/scripts/agentic_serving/emit.py`
- Test: `tests/unit/serving/test_serving_emit.py`

**Interfaces:**
- Consumes: form_gate's `needs_files` / `read_failed` (Task 4).
- Produces: `{"finish": false, "reads": ["<path>", ...]}` for a read request; `{"finish": true, "content": "Refused: could not read <path>: <reason>"}` for a failed one. All existing outcomes unchanged.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/serving/test_serving_emit.py` (follow its existing subprocess/dependency helper):

```python
def test_needs_files_emits_a_reads_outcome() -> None:
    outcome = _emit(
        gated={
            "build": False,
            "file": "test_storage.py",
            "content": "Requesting client files.",
            "valid": True,
            "reason": "ok",
            "needs_files": ["storage.py"],
            "read_failed": "",
            "accept": None,
            "accept_reason": "",
            "seat_admitted": None,
            "seat_contract_reason": "",
        }
    )
    assert outcome == {"finish": False, "reads": ["storage.py"]}


def test_read_failed_emits_an_honest_refusal() -> None:
    outcome = _emit(
        gated={
            "build": False,
            "file": "test_storage.py",
            "content": "Requesting client files.",
            "valid": True,
            "reason": "ok",
            "needs_files": [],
            "read_failed": "could not read storage.py: client read failed",
            "accept": None,
            "accept_reason": "",
            "seat_admitted": None,
            "seat_contract_reason": "",
        }
    )
    assert outcome["finish"] is True
    assert outcome["content"] == (
        "Refused: could not read storage.py: client read failed"
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/serving/test_serving_emit.py -v`
Expected: new tests FAIL (reads outcome missing).

- [ ] **Step 3: Implement in `emit.py`**

In `main()`, read the two fields alongside the existing ones:

```python
    needs_files = gated.get("needs_files") or []
    read_failed = str(gated.get("read_failed", ""))
```

Add two branches at the TOP of the outcome chain (before the `seat_admitted is False` branch — a need-files turn never ran a real seat, so routing-decided outcomes outrank seat verdicts):

```python
    if read_failed:
        # issue #83: one read round per turn — a failed request refuses
        # honestly, never re-requests.
        outcome = {"finish": True, "content": f"Refused: {read_failed}"}
    elif needs_files:
        # issue #83: delegate the file reads to the client permission seam.
        outcome = {"finish": False, "reads": list(needs_files)}
    elif seat_admitted is False:
        ...
```

Update the module docstring's outcome table to list the two new variants:

```
    needs files:     {"finish": false, "reads": ["<path>", ...]}
    read failed:     {"finish": true, "content": "Refused: could not read <path>: <reason>"}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/serving/test_serving_emit.py -v`
Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add .llm-orc/scripts/agentic_serving/emit.py tests/unit/serving/test_serving_emit.py
git commit -m "feat: emit read-request and read-refusal serve outcomes"
```

---

### Task 6: caller — read tool_calls out, read continuations back in

**Files:**
- Modify: `src/llm_orc/web/serving/serving_ensemble_caller.py`
- Test: `tests/unit/web/serving/test_serving_context_render.py` (continuation helpers), `tests/unit/web/test_serving_ensemble_endpoint.py` (outcome mapping via endpoint in Task 7)

**Interfaces:**
- Consumes: emit's `reads` outcome (Task 5); `_read_shaped_arguments` (Task 1); `SessionContext.tools` (the client's advertised tool list, OpenAI `{"type": "function", "function": {"name": ...}}` dicts).
- Produces: `_client_tool(tools, candidates, fallback) -> str` (first advertised candidate name, else fallback); `_outcome_chunks(outcome, tools)` — NEW second parameter — mapping `reads` to one `ClientToolCall` with one invocation per path, arguments `{"filePath": path}`; `_tool_result_ack` returning `None` for read continuations so `run()` falls through to the pipeline.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/web/serving/test_serving_context_render.py`:

```python
from llm_orc.web.serving.serving_ensemble_caller import (
    _outcome_chunks,
    _tool_result_ack,
)
from llm_orc.web.serving.chunks import ClientToolCall


def test_reads_outcome_maps_to_read_tool_calls() -> None:
    tools = [{"type": "function", "function": {"name": "read"}}]
    chunks = _outcome_chunks({"finish": False, "reads": ["a.py", "b.py"]}, tools)

    assert len(chunks) == 1
    call = chunks[0]
    assert isinstance(call, ClientToolCall)
    assert [c.name for c in call.tool_calls] == ["read", "read"]
    assert [json.loads(c.arguments)["filePath"] for c in call.tool_calls] == [
        "a.py",
        "b.py",
    ]


def test_write_outcome_resolves_against_advertised_tool_names() -> None:
    tools = [{"type": "function", "function": {"name": "write_file"}}]
    chunks = _outcome_chunks(
        {"finish": False, "file": "a.py", "content": "pass"}, tools
    )
    assert chunks[0].tool_calls[0].name == "write_file"


def test_write_outcome_falls_back_to_write_when_nothing_advertised() -> None:
    chunks = _outcome_chunks({"finish": False, "file": "a.py", "content": "pass"}, [])
    assert chunks[0].tool_calls[0].name == "write"


def test_read_continuation_is_not_acked() -> None:
    messages = [
        ChatMessage(role="user", content="fix calc.py"),
        ChatMessage(
            role="assistant", content=None, tool_calls=(_read_call("c1", "calc.py"),)
        ),
        ChatMessage(role="tool", tool_call_id="c1", content="def divide(a): return a"),
    ]
    assert _tool_result_ack(messages) is None


def test_write_continuation_is_still_acked() -> None:
    messages = [
        ChatMessage(role="user", content="write add.py"),
        ChatMessage(
            role="assistant",
            content=None,
            tool_calls=(_write_call("add.py", "def add(a, b): return a + b"),),
        ),
        ChatMessage(role="tool", content="Wrote file successfully."),
    ]
    assert _tool_result_ack(messages) == "Wrote add.py."
```

Add the needed `json` import at the top of the test file if absent.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/web/serving/test_serving_context_render.py -v`
Expected: new tests FAIL (`_outcome_chunks` takes 1 argument; read continuation acked "Done.").

- [ ] **Step 3: Implement in `serving_ensemble_caller.py`**

Add near `_WRITE_TOOL`:

```python
_READ_TOOL = "read"
# issue #83 tool mapping: resolve emit outcomes against the client's
# advertised tool names; candidates cover the common client vocabularies.
_WRITE_TOOL_CANDIDATES = ("write", "write_file", "Write")
_READ_TOOL_CANDIDATES = ("read", "read_file", "Read")


def _client_tool(
    tools: Sequence[Any], candidates: tuple[str, ...], fallback: str
) -> str:
    """The first advertised candidate tool name, else the fallback."""
    advertised = set()
    for tool in tools or ():
        function = tool.get("function", {}) if isinstance(tool, dict) else {}
        name = function.get("name")
        if isinstance(name, str):
            advertised.add(name)
    for candidate in candidates:
        if candidate in advertised:
            return candidate
    return fallback
```

Change `_outcome_chunks` to take and use the advertised tools:

```python
def _outcome_chunks(
    outcome: dict[str, Any], tools: Sequence[Any]
) -> list[OrchestratorChunk]:
    if outcome.get("finish"):
        return [
            ContentDelta(content=str(outcome.get("content", "Done."))),
            Completion(finish_reason="stop"),
        ]
    reads = outcome.get("reads")
    if reads:
        read_tool = _client_tool(tools, _READ_TOOL_CANDIDATES, _READ_TOOL)
        invocations = tuple(
            ToolCallInvocation(
                id=f"call_{uuid.uuid4().hex[:8]}",
                name=read_tool,
                arguments=json.dumps({"filePath": str(path)}),
            )
            for path in reads
        )
        return [ClientToolCall(tool_calls=invocations)]
    arguments = json.dumps(
        {
            "filePath": outcome.get("file", "solution.py"),
            "content": outcome.get("content", ""),
        }
    )
    invocation = ToolCallInvocation(
        id=f"call_{uuid.uuid4().hex[:8]}",
        name=_client_tool(tools, _WRITE_TOOL_CANDIDATES, _WRITE_TOOL),
        arguments=arguments,
    )
    return [ClientToolCall(tool_calls=(invocation,))]
```

Split the continuation in `_tool_result_ack` — insert a read check at the top of the walk-back loop:

```python
    last = messages[-1] if messages else None
    if getattr(last, "role", None) != "tool":
        return None
    for message in reversed(list(messages)):
        for call in getattr(message, "tool_calls", ()) or ():
            if _read_shaped_arguments(call) is not None:
                # issue #83: a read continuation RESUMES the turn — fall
                # through to the pipeline with the read result in context.
                return None
        written = _written_file_path(getattr(message, "tool_calls", ()) or ())
        if written:
            return f"Wrote {written}."
        if getattr(message, "role", None) == "user":
            break
    content = getattr(last, "content", None)
    return content if isinstance(content, str) and content.strip() else "Done."
```

Update the docstring of `_tool_result_ack` to note the read/write split. In `run()`, update the outcome mapping call:

```python
        for chunk in _outcome_chunks(outcome, context.tools):
            yield chunk
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/web/serving/test_serving_context_render.py tests/unit/web/test_serving_ensemble_endpoint.py -v`
Expected: ALL PASS (the endpoint file's ack test still passes: write continuations unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/llm_orc/web/serving/serving_ensemble_caller.py tests/unit/web/serving/test_serving_context_render.py
git commit -m "feat: caller emits client read calls and resumes read continuations"
```

---

### Task 7: gather — materialize read blocks into the sandbox workspace

**Files:**
- Modify: `.llm-orc/scripts/agentic_serving/accept_gather.py`
- Test: `tests/unit/serving/test_serving_accept_gather.py`

**Interfaces:**
- Consumes: the read-block grammar (Task 1) inside the context half of `dispatch_input`.
- Produces: `_workspace(context)` returning `{basename: body}` for untruncated `[wrote ...]` AND successful `[read ...]` blocks; failed/oversize/truncated variants never materialize. `tests_gather.py` inherits via its existing `from accept_gather import _workspace` — no change there.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/serving/test_serving_accept_gather.py` (follow its existing helper for driving `_workspace` / the script):

```python
def test_read_block_materializes_into_the_workspace() -> None:
    context = (
        "user: write tests for existing storage.py\n"
        "assistant: [read storage.py]\n"
        "def put(k, v):\n"
        "    return (k, v)"
    )
    workspace = _workspace(context)
    assert workspace["storage.py"] == "def put(k, v):\n    return (k, v)"


def test_failed_and_oversize_read_lines_never_materialize() -> None:
    context = (
        "assistant: [read gone.py (failed)] Error: ENOENT\n"
        "assistant: [read big.py (oversize)]"
    )
    workspace = _workspace(context)
    assert workspace == {}


def test_truncated_wrote_block_still_never_materializes() -> None:
    context = "assistant: [wrote storage.py (truncated)]\ndef put(k"
    assert _workspace(context) == {}
```

Import `_workspace` the way the file's existing tests do (sibling-import of the script module or subprocess — match the established pattern).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/serving/test_serving_accept_gather.py -v`
Expected: the read-block test FAILS (empty workspace); the other two may already pass — keep them as regression guards.

- [ ] **Step 3: Implement in `accept_gather.py`**

Replace `_WRITE_HEADER_RE`:

```python
# A file block in the rendered context: conversation-written ([wrote ...])
# or client-read ([read ...], issue #83). '(truncated)' / '(failed)' /
# '(oversize)' variants are never materialized; a failed read line carries
# trailing reason text after ']' and so never matches the anchored $.
_FILE_HEADER_RE = re.compile(
    r"^assistant: \[(?:wrote|read) ([^\]]+?)( \((?:truncated|failed|oversize)\))?\]$"
)
```

In `_workspace`, change the match line and keep the group semantics (group(1) path, group(2) skip-variant):

```python
        header = _FILE_HEADER_RE.match(lines[index])
```

The rest of `_workspace` is unchanged — `if not header.group(2):` already gates materialization on the absence of a variant suffix. Update the module comment above the regex if it still says "conversation-written file block" only.

Edge to verify in the test run: `[read storage.py (failed)] Error: ENOENT` must NOT match — the anchored `\]$` excludes lines with trailing text, and `[^\]]+?` cannot cross the closing bracket.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/serving/test_serving_accept_gather.py tests/unit/serving/ -v`
Expected: ALL PASS (the full serving unit dir guards tests_gather's reuse).

- [ ] **Step 5: Commit**

```bash
git add .llm-orc/scripts/agentic_serving/accept_gather.py tests/unit/serving/test_serving_accept_gather.py
git commit -m "feat: materialize client-read blocks into the gate workspace"
```

---

### Task 8: hermetic end-to-end — the two-pass turn through the real engine

**Files:**
- Test: `tests/unit/web/test_serving_ensemble_endpoint.py`

**Interfaces:**
- Consumes: everything above, through the real serving ensemble on the L0 engine (the file's `serving_project` fixture pattern: real `serving.yaml` + scripts, echo seats shadowing model nodes — zero model tokens).
- Produces: three endpoint-level proofs: pass 1 emits the read tool_call; pass 2 resumes into the pipeline and ships the build; a failed read refuses honestly.

- [ ] **Step 1: Extend the fixture and add a read tool**

In `tests/unit/web/test_serving_ensemble_endpoint.py`, add next to `_WRITE_TOOL`:

```python
_READ_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "read",
        "description": "Read a file from disk.",
        "parameters": {
            "type": "object",
            "properties": {"filePath": {"type": "string"}},
            "required": ["filePath"],
        },
    },
}
```

In the `serving_project` fixture, after the `explainer.yaml` write, add:

```python
    shutil.copy(REAL_AGENTIC_SERVING / "need-files.yaml", ensembles / "need-files.yaml")
```

and add the module constant `REAL_AGENTIC_SERVING`-based copy only if not already imported (it is — reuse it).

- [ ] **Step 2: Write the three failing tests**

```python
def test_invisible_named_file_turn_emits_a_read_tool_call(
    serving_client: TestClient,
) -> None:
    """Pass 1 (issue #83): a turn naming a client-workspace file the serve
    cannot see delegates a read through the permission seam instead of
    dispatching a build that would reject."""
    resp = serving_client.post(
        "/v1/chat/completions",
        json={
            "model": "ensemble-agent",
            "messages": [
                {"role": "user", "content": "fix the divide function in calc.py"}
            ],
            "tools": [_WRITE_TOOL, _READ_TOOL_DEF],
        },
    )

    assert resp.status_code == 200
    choice = resp.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    call = choice["message"]["tool_calls"][0]
    assert call["function"]["name"] == "read"
    assert json.loads(call["function"]["arguments"]) == {"filePath": "calc.py"}


def test_read_continuation_resumes_the_turn_and_ships_the_build(
    serving_client: TestClient,
) -> None:
    """Pass 2 (issue #83): the client's read result re-enters the pipeline
    (never the write-continuation ack) and the resumed turn ships a write."""
    resp = serving_client.post(
        "/v1/chat/completions",
        json={
            "model": "ensemble-agent",
            "messages": [
                {"role": "user", "content": "fix the divide function in calc.py"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_r1",
                            "type": "function",
                            "function": {
                                "name": "read",
                                "arguments": '{"filePath": "calc.py"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_r1",
                    "content": "def divide(a, b):\n    return a / b",
                },
            ],
            "tools": [_WRITE_TOOL, _READ_TOOL_DEF],
        },
    )

    assert resp.status_code == 200
    choice = resp.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    call = choice["message"]["tool_calls"][0]
    assert call["function"]["name"] == "write"
    args = json.loads(call["function"]["arguments"])
    assert args["filePath"] == "calc.py"


def test_failed_read_refuses_honestly_without_relooping(
    serving_client: TestClient,
) -> None:
    """One read round per turn (issue #83): a failed client read refuses
    with a reason — never a second read request, never a silent build."""
    resp = serving_client.post(
        "/v1/chat/completions",
        json={
            "model": "ensemble-agent",
            "messages": [
                {"role": "user", "content": "fix the divide function in calc.py"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_r1",
                            "type": "function",
                            "function": {
                                "name": "read",
                                "arguments": '{"filePath": "calc.py"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_r1",
                    "content": "Error: ENOENT: no such file calc.py",
                },
            ],
            "tools": [_WRITE_TOOL, _READ_TOOL_DEF],
        },
    )

    assert resp.status_code == 200
    choice = resp.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert not choice["message"].get("tool_calls")
    assert "could not read calc.py" in choice["message"]["content"]
```

Note pass 2 routes to `code-seat` (existing-marker "fix"), which the fixture already shadows with the echo `code-generator` — the whole flow is deterministic.

- [ ] **Step 3: Run tests to verify they fail, then pass**

Run: `uv run pytest tests/unit/web/test_serving_ensemble_endpoint.py -v`
Expected on first run: the three new tests FAIL only if any prior task was mis-wired; with Tasks 1–7 complete they should PASS. If one fails, debug the seam it names before proceeding (this task is the wiring proof, not new behavior).

- [ ] **Step 4: Run the full suite and lint**

Run: `make test && make lint`
Expected: ALL PASS, no warnings.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/web/test_serving_ensemble_endpoint.py
git commit -m "test: hermetic two-pass client-read turn through the real engine"
```

---

### Task 9: live validation against real OpenCode + docs

This task is evidence work — expect iteration, and keep the wire log.

**Files:**
- Possibly modify: `src/llm_orc/web/serving/serving_ensemble_caller.py` (`_normalize_read` + its unit tests) once the real read-result format is observed
- Modify: `docs/serving-roadmap.md` (trajectory table, path items), `docs/serving.md` (capability coverage)

- [ ] **Step 1: Wire capture.** Start the serve with the wire log on (`LLM_ORC_SERVE_WIRE_LOG=1 llm-orc serve` or the documented equivalent — check `grep -rn "LLM_ORC_SERVE_WIRE_LOG" src/` for the exact variable semantics). In a scratch git repo containing a real `storage.py`-style module, run the exact battery failure through the real client: `opencode run "write tests for existing storage.py"`.
- [ ] **Step 2: Verify pass 1 on the wire.** The serve's response is a `read` tool_call for `storage.py` (check the wire log / turn trace at `.llm-orc/.serve-trace/turns.jsonl`).
- [ ] **Step 3: Verify the read-result format.** Inspect the continuation request's `role: tool` message. If OpenCode's read output differs from the `_normalize_read` assumptions (`<file>` wrapper, `NNNNN| ` gutter), adjust `_normalize_read` and its unit tests to the observed format, rerun `make test`, and commit as `fix: normalize OpenCode read output per captured wire`.
- [ ] **Step 4: Verify pass 2 end-to-end.** The resumed turn routes to the write-tests shape, the sandbox imports the read module, and the session ends in a shipped test file or an honest reject (test-quality rejects are in-bounds; invisibility rejects are the defect this arc closes and must not appear).
- [ ] **Step 5: Ladder rerun.** Run the current battery ladder with an added existing-file rung; update the trajectory table in `docs/serving-roadmap.md` with date, battery, score, notes.
- [ ] **Step 6: Docs.** Update `docs/serving.md` §Current capability coverage (client-file reads shipped; run-tests delegation and discovery still open) and `docs/serving-roadmap.md` §2 (#83: read half done, run half remaining). Commit as `docs: record client-read capability and ladder results`.
- [ ] **Step 7: Issue note.** Comment the evidence on #83 (`gh issue comment 83`): what shipped, wire observations, the remaining run-tests half. Do not close the issue.

---

## Execution order and dependencies

Tasks 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9, strictly. Task 1 defines the block grammar every later task parses; Tasks 2–5 thread the routing fields front to back; Task 6 needs Task 1's `_read_shaped_arguments`; Task 8 needs everything; Task 9 needs a green Task 8.

## Out of scope (named forward directions, per the spec)

- Discovery (list/glob for files the turn doesn't name) — meta-task rung work.
- Run-tests delegation (`{"finish": false, "run": ...}`) — the fix-execution rung's enabler.
- Reads for `test_`-named files and for "add X to foo.py" phrasings (no existing-marker) — rung-1 trigger limitations, revisit on ladder evidence.
- Reads on the model-decider (ambiguous) routing path.

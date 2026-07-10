# Seat Quality Implementation Plan (path item 4: isolation + sanitizer)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Kill the measured round-1 false-reject classes deterministically — per-test isolation in the accept executor, a bare-name-assert sanitizer, and the classify interrogative fix.

**Architecture:** Three independent deterministic changes, no model judgment added. classify gains a memory-shaped interrogative form. The accept executor enumerates test functions via AST and runs each in its own subprocess with its own fresh materialized directory (module and filesystem state cannot leak across tests), reusing the existing runner with a new `--only` flag. Before execution the executor strips value-free bare-name assert lines from the tests and echoes the sanitized suite (what executes is what ships).

**Tech Stack:** Python 3.13, pytest, subprocess-driven script tests (the established `tests/unit/serving/` pattern).

**Spec:** `docs/plans/2026-07-09-seat-quality-isolation-escalation-design.md`

## Global Constraints

- ruff (88 chars) and mypy strict compliant from first draft for `src/`/`tests/`; scripts under `.llm-orc/scripts/` match their existing style. `make lint` before every commit.
- TDD: failing test → verify fail → implement → verify pass → commit. One behavioral unit per commit; prefixes `feat:`/`fix:`/`test:`/`docs:`. No AI attribution.
- No silent caps: any bound the executor applies (child cap) must surface in the report text.
- Escalation is OUT of scope (deferred per spec rev 2); loop bound stays 2.
- Full suite: `make test`. Targeted: `uv run pytest <path> -v`.

---

### Task 1: classify — memory-shaped interrogative routes to explain

**Files:**
- Modify: `.llm-orc/scripts/agentic_serving/classify.py` (the `_INTERROGATIVE_RE` constant)
- Test: `tests/unit/serving/test_serving_classify.py`

**Interfaces:**
- Consumes: existing `_classify(turn)` subprocess helper in the test file.
- Produces: "did you …" / "have you …" turns route to `explainer` deterministically (no decider).

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/serving/test_serving_classify.py`:

```python
def test_did_you_memory_question_routes_to_explainer_deterministically() -> None:
    decision = _classify({"task": "did you see my previous query?"})
    assert decision["target"] == "explainer"
    assert decision["build"] is False
    assert decision["needs_decider"] is False


def test_have_you_question_routes_to_explainer() -> None:
    decision = _classify({"task": "have you written any tests yet?"})
    assert decision["target"] == "explainer"


def test_can_you_write_stays_a_build_turn() -> None:
    decision = _classify({"task": "can you write a function that adds in add.py"})
    assert decision["target"] != "explainer"
    assert decision["build"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/serving/test_serving_classify.py -v`
Expected: the two new explainer tests FAIL (`needs_decider` True / wrong target); `can_you` may already pass (keep as a regression guard).

- [ ] **Step 3: Implement**

In `classify.py`, replace the `_INTERROGATIVE_RE` definition:

```python
# An interrogative-shaped turn asks for understanding; it outranks the
# named-file build signal ("What approach does palindrome.py use?" is an
# explain turn, not a build). The yes/no forms are deliberately narrow —
# only memory-shaped questions addressed to the assistant ("did you…",
# "have you…"); "can/could/will you write X" are polite imperatives and
# must stay on the build path (ladder turn 5 mis-route, 2026-07-09).
_INTERROGATIVE_RE = re.compile(
    r"^(?:what|why|how|when|where|which|who)\b|^(?:did|have) you\b",
    re.IGNORECASE,
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/serving/test_serving_classify.py -v`
Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add .llm-orc/scripts/agentic_serving/classify.py tests/unit/serving/test_serving_classify.py
git commit -m "fix: route memory-shaped yes/no questions to explain deterministically"
```

---

### Task 2: runner — `--only <test_name>` single-test mode

**Files:**
- Modify: `.llm-orc/scripts/agentic_serving/accept_executor_runner.py`
- Test: `tests/unit/serving/test_serving_accept_executor_runner.py` (new)

**Interfaces:**
- Consumes: existing `run_tests(code, tests)` and the CLI form `runner.py <code_path> <tests_path>`.
- Produces: CLI form `runner.py <code_path> <tests_path> --only <name>` — loads code+tests as today but calls ONLY the named top-level `test_*` function (TestCase classes are skipped in `--only` mode); `--only __cases__` runs ONLY the unittest TestCase classes. No flag = legacy whole-run behavior, byte-identical.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/serving/test_serving_accept_executor_runner.py`:

```python
"""Single-test mode for the accept-gate runner (per-test isolation,
seat-quality design 2026-07-09). Driven via subprocess exactly as
accept_executor.py invokes it."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
RUNNER = REPO / ".llm-orc" / "scripts" / "agentic_serving" / "accept_executor_runner.py"

CODE = "todos = []\ndef add(item):\n    todos.append(item)\n"
LEAKY_TESTS = (
    "def test_one():\n    add('a')\n    assert len(todos) == 1\n"
    "def test_two():\n    add('a')\n    add('b')\n    assert len(todos) == 2\n"
)


def _run(code: str, tests: str, tmp: Path, only: str | None = None) -> dict[str, Any]:
    (tmp / "solution.py").write_text(code)
    (tmp / "tests.py").write_text(tests)
    argv = [sys.executable, str(RUNNER), str(tmp / "solution.py"), str(tmp / "tests.py")]
    if only is not None:
        argv += ["--only", only]
    out = subprocess.run(argv, capture_output=True, text=True, cwd=tmp, check=True)
    result: dict[str, Any] = json.loads(out.stdout)
    return result


def test_only_runs_exactly_the_named_test(tmp_path: Path) -> None:
    verdict = _run(CODE, LEAKY_TESTS, tmp_path, only="test_two")
    assert verdict["n_tests"] == 1
    assert verdict["tests_pass"] is True  # fresh namespace: no leaked 'a'


def test_only_unknown_name_reports_zero_tests(tmp_path: Path) -> None:
    verdict = _run(CODE, LEAKY_TESTS, tmp_path, only="test_missing")
    assert verdict["n_tests"] == 0
    assert verdict["tests_pass"] is False


def test_no_flag_keeps_legacy_whole_run(tmp_path: Path) -> None:
    verdict = _run(CODE, LEAKY_TESTS, tmp_path)
    assert verdict["n_tests"] == 2
    assert verdict["tests_pass"] is False  # the shared-state leak, as today
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/serving/test_serving_accept_executor_runner.py -v`
Expected: the two `--only` tests FAIL (unknown flag / whole run executes); the legacy test PASSES.

- [ ] **Step 3: Implement**

In `accept_executor_runner.py`:

Add an `only` parameter to `run_tests` and filter both dialects:

```python
def run_tests(code: str, tests: str, only: str | None = None) -> tuple[bool, str, int]:
    """Exec code + tests in a shared namespace, call every ``test_*`` function.

    ``only`` (per-test isolation, seat-quality design 2026-07-09) restricts
    the run to one named top-level test function — the executor spawns one
    runner per test so module and filesystem state cannot leak across
    tests. ``only="__cases__"`` runs just the unittest.TestCase classes.
    """
```

After `test_fns` and `case_classes` are collected, insert:

```python
    if only == "__cases__":
        test_fns = []
    elif only is not None:
        test_fns = [(n, f) for n, f in test_fns if n == only]
        case_classes = []
```

And in the `n_tests == 0` return, keep the message but make it mode-aware:

```python
    if n_tests == 0:
        detail = f"no test named {only!r} found" if only and only != "__cases__" else (
            "no test_* functions or TestCase classes found"
        )
        return False, detail, 0
```

In `main()`, parse the flag (keep the positional contract):

```python
def main() -> None:
    # sandbox dir on sys.path so tests can import materialized workspace
    # modules (conversation-written files) as siblings
    sys.path.insert(0, str(Path(sys.argv[1]).resolve().parent))
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1]
    code = Path(sys.argv[1]).read_text(encoding="utf-8")
    tests = Path(sys.argv[2]).read_text(encoding="utf-8")
    tests_pass, report, n_tests = run_tests(code, tests, only)
    print(json.dumps({"tests_pass": tests_pass, "n_tests": n_tests, "report": report}))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/serving/test_serving_accept_executor_runner.py tests/unit/serving/test_serving_accept_gate.py -v`
Expected: ALL PASS (the gate file's legacy executor tests guard the no-flag path).

- [ ] **Step 5: Commit**

```bash
git add .llm-orc/scripts/agentic_serving/accept_executor_runner.py tests/unit/serving/test_serving_accept_executor_runner.py
git commit -m "feat: runner --only mode for per-test isolation"
```

---

### Task 3: executor — one fresh sandbox per test function

**Files:**
- Modify: `.llm-orc/scripts/agentic_serving/accept_executor.py`
- Test: `tests/unit/serving/test_serving_accept_gate.py` (extend; its `_executor` helper drives the script)

**Interfaces:**
- Consumes: Task 2's `--only` runner mode.
- Produces: `_run_sandboxed(code, tests, workspace, target_file)` same signature and return `(tests_pass, report, n_tests)`, but internally: AST-enumerate top-level `test_*` functions in `tests`; for each, materialize a FRESH tempdir (workspace + target + solution.py + tests.py exactly as today) and run the runner with `--only <name>`; if TestCase classes are present, one extra child with `--only __cases__`; aggregate `n_tests` (sum) and failures (`"; "`-joined). Unparseable tests or zero enumerable tests → single legacy child (today's behavior). Child cap 20, surfaced in the report as `"…capped at 20 isolated tests"`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/serving/test_serving_accept_gate.py`:

```python
LEAKY_STATE_CODE = "todos = []\ndef add(item):\n    todos.append(item)\n"
LEAKY_STATE_TESTS = (
    "def test_one():\n    add('a')\n    assert len(todos) == 1\n"
    "def test_two():\n    add('a')\n    add('b')\n    assert len(todos) == 2\n"
)
LEAKY_FILE_TESTS = (
    "import os, json\n"
    "def save(t):\n    json.dump(t, open('t.json', 'w'))\n"
    "def test_writes():\n    save([1])\n    assert os.path.exists('t.json')\n"
    "def test_fresh():\n    assert not os.path.exists('t.json')\n"
)


def test_executor_isolates_module_state_across_tests() -> None:
    result = _executor("add todos", LEAKY_STATE_CODE, LEAKY_STATE_TESTS)
    assert result["tests_pass"] is True
    assert result["n_tests"] == 2


def test_executor_isolates_filesystem_across_tests() -> None:
    result = _executor("save todos", "", LEAKY_FILE_TESTS)
    assert result["tests_pass"] is True
    assert result["n_tests"] == 2


def test_executor_still_fails_genuinely_wrong_code() -> None:
    result = _executor(
        "adds", "def add(a, b):\n    return a - b\n",
        "def test_add():\n    assert add(1, 2) == 3\n",
    )
    assert result["tests_pass"] is False
    assert "test_add" in result["report"]
```

(The `LEAKY_FILE_TESTS` fixture defines its helper inside the tests file so the empty-code arm exercises pure test-side isolation.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/serving/test_serving_accept_gate.py -v`
Expected: the two isolation tests FAIL (leak → `tests_pass` False); the wrong-code test PASSES already (regression guard).

- [ ] **Step 3: Implement**

In `accept_executor.py`, add after the imports:

```python
import ast
import re

# per-test isolation (seat-quality design 2026-07-09): each test function
# runs in its own subprocess with its own fresh materialized directory, so
# module globals and written files cannot leak across tests. Bounded and
# surfaced — no silent caps.
_MAX_ISOLATED_TESTS = 20
```

Add the enumeration helper:

```python
def _enumerate_tests(tests: str) -> tuple[list[str], bool] | None:
    """(top-level test_* function names, has_testcase_classes), or ``None``
    when the tests don't parse — the caller falls back to one legacy run."""
    try:
        tree = ast.parse(tests)
    except SyntaxError:
        return None
    names = [
        n.name
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name.startswith("test_")
    ]
    has_cases = any(isinstance(n, ast.ClassDef) for n in tree.body)
    return names, has_cases
```

Extract the current single-run body into a private helper so both paths share the materialization (structural move, byte-identical behavior):

```python
def _run_one(
    code: str,
    tests: str,
    workspace: dict[str, str] | None,
    target_file: str,
    only: str | None,
    timeout: float,
) -> tuple[bool, str, int]:
    with tempfile.TemporaryDirectory() as tmp:
        for name, body in (workspace or {}).items():
            safe = Path(name).name
            if safe and safe not in ("solution.py", "tests.py"):
                (Path(tmp) / safe).write_text(str(body), encoding="utf-8")
        safe_target = Path(target_file).name if target_file else ""
        if safe_target and safe_target not in ("solution.py", "tests.py"):
            (Path(tmp) / safe_target).write_text(code, encoding="utf-8")
        code_path = Path(tmp) / "solution.py"
        tests_path = Path(tmp) / "tests.py"
        code_path.write_text(code, encoding="utf-8")
        tests_path.write_text(tests, encoding="utf-8")
        argv = [sys.executable, str(RUNNER), str(code_path), str(tests_path)]
        if only is not None:
            argv += ["--only", only]
        try:
            completed = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmp,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False, f"timeout after {timeout:g}s", 0
    if completed.returncode != 0:
        detail = (completed.stderr.strip() or completed.stdout.strip())[:200]
        return False, f"runner crashed: {detail}", 0
    try:
        verdict = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return False, f"unreadable runner output: {completed.stdout[:200]!r}", 0
    return (
        bool(verdict.get("tests_pass", False)),
        str(verdict.get("report", "")),
        int(verdict.get("n_tests", 0)),
    )
```

Rewrite `_run_sandboxed` to orchestrate per-test children:

```python
def _run_sandboxed(
    code: str,
    tests: str,
    workspace: dict[str, str] | None = None,
    target_file: str = "",
) -> tuple[bool, str, int]:
    timeout = _timeout()
    enumerated = _enumerate_tests(tests)
    if enumerated is None or (not enumerated[0] and not enumerated[1]):
        # unparseable or nothing enumerable: one legacy run reports it
        return _run_one(code, tests, workspace, target_file, None, timeout)

    names, has_cases = enumerated
    capped = len(names) > _MAX_ISOLATED_TESTS
    children: list[str | None] = list(names[:_MAX_ISOLATED_TESTS])
    if has_cases:
        children.append("__cases__")

    failures: list[str] = []
    total = 0
    for only in children:
        ok, report, n_tests = _run_one(
            code, tests, workspace, target_file, only, timeout
        )
        total += n_tests
        if not ok:
            failures.append(report)
    if capped:
        failures.append(f"…capped at {_MAX_ISOLATED_TESTS} isolated tests")

    # a load failure repeats identically in every child — report it once
    deduped = list(dict.fromkeys(failures))
    if deduped:
        return False, "; ".join(deduped), total
    return True, "all passed", total
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/serving/ -v`
Expected: ALL PASS, including the pre-existing executor tests (correct-code pass, wrong-code fail, trivial-tests, timeout — the timeout test's runaway single test now times out in its own child, still reporting failure).

- [ ] **Step 5: Commit**

```bash
git add .llm-orc/scripts/agentic_serving/accept_executor.py tests/unit/serving/test_serving_accept_gate.py
git commit -m "feat: per-test isolation in the accept executor"
```

---

### Task 4: executor — bare-name-assert sanitizer

**Files:**
- Modify: `.llm-orc/scripts/agentic_serving/accept_executor.py`
- Test: `tests/unit/serving/test_serving_accept_gate.py`

**Interfaces:**
- Consumes: Task 3's executor structure.
- Produces: `_sanitize_tests(tests: str) -> tuple[str, int]` — drops lines matching a bare-name assert (`assert <identifier>` with optional `, <message>`), returns (sanitized tests, dropped count). `main()` sanitizes before `_run_sandboxed`, echoes the SANITIZED tests in its output (what executed is what ships), and adds `"tests_sanitized": <n>` to the output JSON.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/serving/test_serving_accept_gate.py`:

```python
STRAY_ASSERT_TESTS = (
    "def test_overwrite():\n"
    "    save_todos(['a'])\n"
    "    assert load\n"
    "    assert load_todos() == ['a']\n"
)
STRAY_CODE = (
    "_d = {}\n"
    "def save_todos(t):\n    _d['t'] = t\n"
    "def load_todos():\n    return _d.get('t', [])\n"
)


def test_bare_name_assert_is_sanitized_before_execution() -> None:
    result = _executor("storage", STRAY_CODE, STRAY_ASSERT_TESTS)
    assert result["tests_pass"] is True
    assert result["tests_sanitized"] == 1
    assert "assert load\n" not in result["tests"]
    assert "assert load_todos() == ['a']" in result["tests"]


def test_value_bearing_asserts_are_never_sanitized() -> None:
    tests = "def test_add():\n    assert add(1, 2) == 3\n"
    result = _executor("adds", "def add(a, b):\n    return a + b\n", tests)
    assert result["tests_pass"] is True
    assert result["tests_sanitized"] == 0
    assert "assert add(1, 2) == 3" in result["tests"]


def test_bare_assert_on_an_assigned_local_is_kept() -> None:
    # 'assert result' on a test-local CAN be a real truthiness check —
    # only names never assigned in the tests source are value-free.
    tests = "def test_t():\n    result = add(1, 2)\n    assert result\n"
    result = _executor("adds", "def add(a, b):\n    return a + b\n", tests)
    assert result["tests_sanitized"] == 0
    assert "assert result" in result["tests"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/serving/test_serving_accept_gate.py -v`
Expected: the sanitizer tests FAIL (`KeyError: 'tests_sanitized'` / NameError reject).

- [ ] **Step 3: Implement**

In `accept_executor.py`, add near `_MAX_ISOLATED_TESTS`:

```python
# A bare-name assert on a name the tests never assign (``assert load`` /
# ``assert load, "msg"``) checks only module-object truthiness — a defined
# function is always truthy, so the line carries no test value, and the
# 4-arm tier spike (2026-07-09) showed no seat at any tier/thinking mode
# satisfies a garbage one. Stripped before execution; the echoed (shipped)
# tests are the sanitized suite. ``assert result`` on an assigned local is
# a real truthiness check and is kept.
_BARE_ASSERT_RE = re.compile(r"^\s*assert\s+([A-Za-z_]\w*)\s*(?:,.*)?$")
_ASSIGNED_NAME_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*=[^=]", re.MULTILINE)


def _sanitize_tests(tests: str) -> tuple[str, int]:
    """Tests with value-free bare-name assert lines removed, and the count."""
    assigned = set(_ASSIGNED_NAME_RE.findall(tests))
    kept: list[str] = []
    dropped = 0
    for line in tests.splitlines():
        match = _BARE_ASSERT_RE.match(line)
        if match and match.group(1) not in assigned:
            dropped += 1
            continue
        kept.append(line)
    return "\n".join(kept) + ("\n" if tests.endswith("\n") else ""), dropped
```

In `main()`, after `tests = str(data.get("tests", ""))`:

```python
    tests, tests_sanitized = _sanitize_tests(tests)
```

and add to the printed JSON object:

```python
                "tests_sanitized": tests_sanitized,
```

(The existing `"tests": tests` line now echoes the sanitized suite by construction.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/serving/ tests/unit/benchmarks/ -v`
Expected: ALL PASS (`tests/unit/benchmarks/` guards the judge-adequacy harness, which drives the same executor).

- [ ] **Step 5: Commit**

```bash
git add .llm-orc/scripts/agentic_serving/accept_executor.py tests/unit/serving/test_serving_accept_gate.py
git commit -m "feat: strip value-free bare-name asserts before the accept gate"
```

---

### Task 5: live validation — regression probes, ladder rerun, docs

Evidence work; expect iteration.

**Files:**
- Modify: `docs/serving-roadmap.md` (trajectory row; path item 4 status)
- Modify: `docs/serving.md` if capability wording changes

- [ ] **Step 1: Full suite + lint.** `make test && make lint` — all green before any live run.
- [ ] **Step 2: Regression probes.** Start `uv run llm-orc serve --port 8765` from the repo root. In a fresh scratch git repo, run the two recorded probes through real OpenCode (`opencode run -m llm-orc/agentic "<prompt>"`, fresh session each):
  - "write a function that adds a todo item to a list in todo.py" — must ACCEPT (write tool_call) within two attempts.
  - "create storage.py with save_todos and load_todos functions using json" — must ACCEPT within two attempts.
- [ ] **Step 3: Ladder rerun.** `LADDER_REPO=<fresh seeded repo> LADDER_OUT=<out dir> benchmarks/agentic_serving/ladder_battery.sh` (seed calc.py per the script header). Score strictly per the trajectory convention; turn 5 must now route to explain; turns 1/4/6 are the ones isolation+sanitizer target.
- [ ] **Step 4: Docs.** Add the trajectory row with per-class attribution; update path item 4's status in `docs/serving-roadmap.md` (shipped: slices 0/1/2′; deferred: escalation with the 14b-think-off measurement). Commit as `docs: record seat-quality arc ladder results`.
- [ ] **Step 5: Stop the serve** and report the score with failure classes.

---

## Execution order and dependencies

Tasks 1 → 2 → 3 → 4 → 5. Task 1 is independent but cheapest first; Task 3 requires Task 2's `--only` flag; Task 4 builds on Task 3's executor structure; Task 5 needs everything.

## Out of scope (per spec rev 2)

- Escalation rounds (deferred; if ever built: qwen3:14b think-off via `agentic-tier-escalated-general`).
- Exception-message assertion sanitizing.
- Test-writer seat escalation; #83 run half.

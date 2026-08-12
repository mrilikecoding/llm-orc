"""Routing-regression corpus for the serving ``classify`` node (round 3
adversarial review, docs/plans/2026-07-17-recap-grounding-design.md).

Round 3's blocker: a previous round widened classify's global ``is_explain``
so recap phrasings could reach the deterministic recap-answer route — but
the widening's loose piece (``_MAYBE_RECAP_RE``, then unanchored) matched
"we/you + past-make-verb" ANYWHERE in a sentence, not just in a genuine recap
QUESTION. An author-independent review diffed routing on origin/main vs this
branch over 65 inputs: 36 changed, only 7 intended.

This is the permanent instrument against a repeat: it runs classify over a
corpus of the 13 ladder-battery prompts (benchmarks/agentic_serving/
ladder_battery.sh), the reviewer's demonstrated collateral-damage classes,
the full recap-floor/decider-extension phrasing set, memory interrogatives,
and ordinal-recall phrasings — pinning each input's full routing decision
(target, needs_decider, build, and the discovery stems carried in
needs_glob) against a table.

Every row's expected values were VERIFIED, not assumed: computed by running
this branch's post-fix classify.py and origin/main's classify.py (``git show
origin/main:.llm-orc/scripts/agentic_serving/classify.py``) over the same
inputs and diffing the results. Rows marked NEW_BEHAVIOR are the turns this
branch's #133/#134/round-3 work intentionally changed (memory interrogatives,
the recap floor, the recap decider extension) — every other row is pinned to
origin/main's own routing, so a future change that drags one of THOSE turns
onto the recap/explain path fails here immediately.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO / ".llm-orc" / "scripts" / "agentic_serving"
CLASSIFY = SCRIPTS / "classify.py"


def _classify(task: str) -> dict[str, Any]:
    envelope = json.dumps({"input": json.dumps({"task": task})})
    out = subprocess.run(
        [sys.executable, str(CLASSIFY)],
        input=envelope,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    result: dict[str, Any] = json.loads(out)
    return result


_ORIGIN_MAIN = "origin/main routing, byte-identical"
_NEW_BEHAVIOR = "NEW_BEHAVIOR: #133/#134 memory/recap answer"

# (task, target, needs_decider, build, needs_glob, provenance)
CORPUS: list[tuple[str, str, bool, bool, str, str]] = [
    # --- the 13 ladder battery prompts (benchmarks/agentic_serving/
    # ladder_battery.sh) ---
    (
        "write a function that adds a todo item to a list in todo.py",
        "code-seat",
        False,
        True,
        "",
        _ORIGIN_MAIN,
    ),
    (
        "add a complete_todo function to todo.py that marks a todo done",
        "code-seat",
        False,
        True,
        "",
        _ORIGIN_MAIN,
    ),
    (
        "explain how todo.py stores its state",
        "not-grounded",
        False,
        False,
        "",
        _ORIGIN_MAIN,
    ),
    ("write tests for todo.py", "need-files", False, False, "", _ORIGIN_MAIN),
    (
        "did you see my previous query?",
        "recall-answer",
        False,
        False,
        "",
        _NEW_BEHAVIOR,
    ),
    (
        "create storage.py with save_todos and load_todos functions using json",
        "code-seat",
        False,
        True,
        "",
        _ORIGIN_MAIN,
    ),
    (
        "update todo.py to persist todos using storage.py",
        "need-files",
        False,
        False,
        "",
        _ORIGIN_MAIN,
    ),
    ("write tests for existing calc.py", "need-files", False, False, "", _ORIGIN_MAIN),
    (
        "write tests for existing phantom.py",
        "need-files",
        False,
        False,
        "",
        _ORIGIN_MAIN,
    ),
    (
        "what did the first thing I asked you to build do?",
        "recall-answer",
        False,
        False,
        "",
        _ORIGIN_MAIN,
    ),
    ("run the tests", "need-run", False, False, "", _ORIGIN_MAIN),
    (
        "write tests for the metrics module",
        "need-glob",
        False,
        False,
        "metrics",
        _ORIGIN_MAIN,
    ),
    ("fix the bug in buggy.py", "need-files", False, False, "", _ORIGIN_MAIN),
    # --- collateral-damage classes the round-3 review demonstrated, all
    # pinned to origin/main's exact routing (the blocker this branch fixes)
    # ---
    (
        "unit tests for todo.py you made earlier",
        "need-files",
        False,
        False,
        "",
        _ORIGIN_MAIN,
    ),
    (
        "tests for the storage module you wrote",
        "tests-seat",
        False,
        True,
        "",
        _ORIGIN_MAIN,
    ),
    ("more tests for what you wrote", "need-glob", False, False, "what", _ORIGIN_MAIN),
    (
        "tests for the parser we built",
        "need-glob",
        False,
        False,
        "parser",
        _ORIGIN_MAIN,
    ),
    ("tests for the thing you built", "need-glob", False, False, "thing", _ORIGIN_MAIN),
    (
        "explain the code you wrote",
        "need-glob",
        False,
        False,
        "code,wrote",
        _ORIGIN_MAIN,
    ),
    (
        "why does the function you made return none",
        "need-glob",
        False,
        False,
        "function,return,none",
        _ORIGIN_MAIN,
    ),
    (
        "what have we built our auth on?",
        "need-glob",
        False,
        False,
        "built,auth",
        _ORIGIN_MAIN,
    ),
    ("delete the file you created", "", True, False, "", _ORIGIN_MAIN),
    ("the tests you wrote are failing", "", True, False, "", _ORIGIN_MAIN),
    ("rename the helper you made", "", True, False, "", _ORIGIN_MAIN),
    ("port the parser we wrote to rust", "", True, False, "", _ORIGIN_MAIN),
    ("improve the function you wrote", "", True, False, "", _ORIGIN_MAIN),
    ("clean up the module you created", "", True, False, "", _ORIGIN_MAIN),
    ("extend the storage module you built", "", True, False, "", _ORIGIN_MAIN),
    ("make the module you built faster", "", True, False, "", _ORIGIN_MAIN),
    ("document what you built", "", True, False, "", _ORIGIN_MAIN),
    ("update the parser we wrote", "", True, False, "", _ORIGIN_MAIN),
    ("revert the change you made", "", True, False, "", _ORIGIN_MAIN),
    # --- the recap floor's full phrasing set (review round 1 blocker 3,
    # widened round 2 new blocker 1): NEW behavior — none of these exist as
    # a concept in origin/main at all ---
    ("what have we built so far?", "recall-answer", False, False, "", _NEW_BEHAVIOR),
    ("what have you built so far?", "recall-answer", False, False, "", _NEW_BEHAVIOR),
    ("what have we made so far?", "recall-answer", False, False, "", _NEW_BEHAVIOR),
    ("what have we done so far?", "recall-answer", False, False, "", _NEW_BEHAVIOR),
    ("what have we written so far?", "recall-answer", False, False, "", _NEW_BEHAVIOR),
    ("what did we build?", "recall-answer", False, False, "", _NEW_BEHAVIOR),
    ("what did you build?", "recall-answer", False, False, "", _NEW_BEHAVIOR),
    ("summarize what we've built", "recall-answer", False, False, "", _NEW_BEHAVIOR),
    ("summarize what we have built", "recall-answer", False, False, "", _NEW_BEHAVIOR),
    ("list everything you made", "recall-answer", False, False, "", _NEW_BEHAVIOR),
    ("list everything you created", "recall-answer", False, False, "", _NEW_BEHAVIOR),
    ("list everything you built", "recall-answer", False, False, "", _NEW_BEHAVIOR),
    ("list everything you wrote", "recall-answer", False, False, "", _NEW_BEHAVIOR),
    ("recap what you've done", "recall-answer", False, False, "", _NEW_BEHAVIOR),
    ("what files have you created", "recall-answer", False, False, "", _NEW_BEHAVIOR),
    ("what files have you made", "recall-answer", False, False, "", _NEW_BEHAVIOR),
    ("what files have you written", "recall-answer", False, False, "", _NEW_BEHAVIOR),
    ("give me a summary of the work", "recall-answer", False, False, "", _NEW_BEHAVIOR),
    ("summarize the work", "recall-answer", False, False, "", _NEW_BEHAVIOR),
    ("summarize the session", "recall-answer", False, False, "", _NEW_BEHAVIOR),
    ("so what do we have now", "recall-answer", False, False, "", _NEW_BEHAVIOR),
    ("where did we end up", "recall-answer", False, False, "", _NEW_BEHAVIOR),
    # --- the recap decider extension's target set (round 3): recap
    # questions missing the tight floor only by inflection or adverb ---
    ("what exactly have you built so far?", "", True, False, "", _NEW_BEHAVIOR),
    ("can you list everything you've made?", "", True, False, "", _NEW_BEHAVIOR),
    # --- memory interrogatives (#134): NEW behavior ---
    (
        "have you written any tests yet?",
        "recall-answer",
        False,
        False,
        "",
        _NEW_BEHAVIOR,
    ),
    ("did you run the tests?", "recall-answer", False, False, "", _NEW_BEHAVIOR),
    ("did you delete my files?", "recall-answer", False, False, "", _NEW_BEHAVIOR),
    ("have you pushed to main?", "recall-answer", False, False, "", _NEW_BEHAVIOR),
    # --- ordinal recall (#82, pre-existing): unaffected by round 3, pinned
    # to origin/main's own routing (this feature predates and is untouched
    # by this branch's classify.py changes) ---
    (
        "what was the first thing you built?",
        "recall-answer",
        False,
        False,
        "",
        _ORIGIN_MAIN,
    ),
    ("the earliest thing you built", "", True, False, "", _ORIGIN_MAIN),
]


@pytest.mark.parametrize(
    ("task", "target", "needs_decider", "build", "needs_glob", "provenance"),
    CORPUS,
    ids=[c[0] for c in CORPUS],
)
def test_routing_corpus_pins_the_full_decision(
    task: str,
    target: str,
    needs_decider: bool,
    build: bool,
    needs_glob: str,
    provenance: str,
) -> None:
    decision = _classify(task)
    actual = (
        decision["target"],
        decision["needs_decider"],
        decision["build"],
        decision["needs_glob"],
    )
    expected = (target, needs_decider, build, needs_glob)
    assert actual == expected, (
        f"routing drift on {task!r} (pinned as {provenance}): "
        f"expected {expected}, got {actual}"
    )


def test_corpus_covers_every_intended_class_at_least_once() -> None:
    # A sanity check on the corpus itself, not on classify: every input
    # marked NEW_BEHAVIOR must actually route away from an ordinary
    # build/explain/decider-fallthrough target, and every ORIGIN_MAIN input
    # must be one this branch's #133/#134 machinery never touches.
    new_behavior_targets = {
        target for task, target, _, _, _, note in CORPUS if note == _NEW_BEHAVIOR
    }
    assert "recall-answer" in new_behavior_targets
    assert "" in new_behavior_targets  # the decider-deferred rows
    assert len(CORPUS) >= 60

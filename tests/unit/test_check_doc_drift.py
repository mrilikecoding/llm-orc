"""The doc-drift check (loop-protocol rule 19).

Doc and code drift silently: no test fails when prose describes deleted
code. Measured cost — one arc's instrument section was wrong in four
consecutive review rounds, every error a name that resolved to nothing or a
name that was missing.

The check is deliberately narrow. It resolves backticked ``test_*`` names in
`docs/plans/*.md` against every ``test_*`` token in code. It does not try to
judge prose for truth, which would make it a denylist.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.check_doc_drift import check


@pytest.fixture
def doc(tmp_path: Path) -> Path:
    path = tmp_path / "design.md"
    return path


def test_a_name_that_resolves_nowhere_is_reported(doc: Path) -> None:
    """The name is BUILT rather than written, so the literal never appears in
    code — otherwise this very file would make it resolve and the pin would
    be green for the wrong reason. That is a real bound of the check, not
    just a test artifact: a name mentioned anywhere in code resolves,
    including in a doc-drift test asserting that it should not."""
    missing = "test_" + "a_name_that_never_existed_in_this_tree"
    doc.write_text(f"Pinned by `{missing}`.\n")

    problems = check([doc])

    assert len(problems) == 1
    assert missing in problems[0]


def test_a_real_test_name_is_not_reported(doc: Path) -> None:
    """This module's own name, so the pin cannot rot with a rename."""
    doc.write_text("Pinned by `test_a_real_test_name_is_not_reported`.\n")

    assert check([doc]) == []


def test_a_class_qualified_name_resolves(doc: Path) -> None:
    doc.write_text("`TestSomething::test_a_real_test_name_is_not_reported` holds.\n")

    assert check([doc]) == []


def test_an_identifier_that_is_not_a_test_still_resolves(doc: Path) -> None:
    """The false-positive class this check was narrowed for: `test_writer` is
    a serving ensemble's AGENT name and `test_fns` a local variable, and a
    doc may legitimately name either in backticks. Resolving against every
    ``test_*`` token in code rather than against defined test functions is
    what keeps those out of the report."""
    # Built, not written, for the same reason as the missing-name pin: a
    # literal here would resolve out of THIS file and the pin would pass
    # whatever the check scanned.
    agent = "test_" + "writer"
    doc.write_text(f"The `{agent}` seat is an agent, not a test.\n")

    assert check([doc]) == []


def test_a_name_living_only_outside_the_test_tree_resolves(doc: Path) -> None:
    """What pins the SCAN's breadth. Narrowing it to `tests/**/test_*.py`
    leaves the previous pin green, because `test_writer` also appears in test
    fixtures — so that pin cannot discriminate the narrowing. This name lives
    only in a benchmark fixture, which is exactly the kind of generated
    identifier a design doc quotes."""
    fixture = "test_" + "absolute_zero"
    doc.write_text(f"The ladder's `{fixture}` case.\n")

    assert check([doc]) == []


def test_prose_without_backticks_is_not_a_claim(doc: Path) -> None:
    """Backticks mean "identifier in this codebase". A doc quoting a
    MODEL-GENERATED test name is describing a fixture, not claiming a repo
    test exists, and writes it unmarked."""
    doc.write_text("the generated test_add_multiple_todo_items asserted two\n")

    assert check([doc]) == []


def test_the_shipped_design_docs_are_clean() -> None:
    """The check gates `make lint`, so this is the pin that keeps the corpus
    at zero rather than letting drift accumulate to a number nobody reads."""
    assert check([]) == []

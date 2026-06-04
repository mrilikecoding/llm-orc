"""Spike ψ.4a — deterministic delegate-vs-carry pre-filter, classification test.

The rule is executable code over raw inputs (user message + values observed
earlier in the conversation), not hand-encoded features — otherwise the test
is circular. It is a spike-level stand-in for the structural pre-filter ψ.4
proposes: IF the framework can decide delegate-vs-carry deterministically,
the seat-filler's role narrows to filling the action shape (model-independent
delegation rate).

Rule: delegate iff
  (a) generation-shaped: an imperative content-producing verb targets a
      content-producing object, and
  (b) capability-matched: task keywords overlap a registered capability
      ensemble's content domain, and
  (c) not observed-carry: the message does not embed the literal payload and
      does not reference applying an already-observed value.

Labeled set: 16 cases — 12 clear (threshold: 0 misclassifications) + 4
constructed ambiguous boundary cases (reported, not thresholded).
Sources: real session tasks (smoke runs 1-3, capture run, wp-lb-c run) plus
constructed cases for the carry/read/bash/finish sides.
"""

import json
import re
from pathlib import Path

GENERATION_VERBS = r"\b(write|create|implement|build|generate|add|refactor|compose|draft)\b"
CONTENT_OBJECTS = (
    r"\b(module|file|function|class|script|test|tests|code|doc|docs|"
    r"documentation|readme|notes|analysis|report|essay)\b"
)
OBSERVED_CARRY_MARKERS = (
    r"\b(exactly this|the following content|the output above|the observed|"
    r"that you just|from the previous|the result above|verbatim)\b"
)
READ_ONLY = r"^\s*(read|show|list|cat|display|what is in|open)\b"
COMMAND_SHAPED = r"^\s*(run|execute|invoke)\b|\b(run the tests|run pytest|npm test)\b"

# Registered capability ensembles and their content domains — from the
# .llm-orc ensemble registry (code-generator is the shipped substrate-routed
# capability ensemble; domains mirror its declared capability).
CAPABILITIES = {
    "code-generator": r"\b(python|code|module|function|class|script|test|tests|refactor)\b",
}


def decide(message: str, observed_values: list[str]) -> str:
    """Return 'delegate' or 'carry' (carry covers read/bash/literal/finish)."""
    msg = message.lower()

    if re.search(READ_ONLY, msg) and not re.search(GENERATION_VERBS, msg):
        return "carry"
    if re.search(COMMAND_SHAPED, msg) and not re.search(
        GENERATION_VERBS + r".*" + CONTENT_OBJECTS, msg
    ):
        return "carry"

    generation_shaped = bool(
        re.search(GENERATION_VERBS, msg) and re.search(CONTENT_OBJECTS, msg)
    )
    if not generation_shaped:
        return "carry"

    if re.search(OBSERVED_CARRY_MARKERS, msg):
        return "carry"
    # Literal payload embedded in the message (fenced or long quoted block).
    if "```" in message or re.search(r"['\"].{120,}['\"]", message, re.S):
        return "carry"
    # The exact content to write was observed earlier in the conversation.
    if any(v and v.lower() in msg for v in observed_values):
        return "carry"

    capability_matched = any(
        re.search(pat, msg) for pat in CAPABILITIES.values()
    )
    return "delegate" if capability_matched else "carry"


CASES = [
    # --- clear: delegate (real session tasks) ---
    {"id": "c01", "label": "delegate", "clear": True,
     "msg": "Write a python module string_utils.py with a function that reverses the word order of a sentence and a function that counts vowels in a string.",
     "src": "psi capture run"},
    {"id": "c02", "label": "delegate", "clear": True,
     "msg": "Write a python module matrix_utils.py with functions to transpose a matrix and multiply two matrices.",
     "src": "smoke run 3"},
    {"id": "c03", "label": "delegate", "clear": True,
     "msg": "Create a python script inventory.py that tracks items with add, remove, and total-count functions.",
     "src": "smoke run (inventory)"},
    {"id": "c04", "label": "delegate", "clear": True,
     "msg": "Implement a Stack class in python with push, pop, and peek methods, in stack.py.",
     "src": "direct probe task"},
    # --- clear: carry (literal / observed-value) ---
    {"id": "c05", "label": "carry", "clear": True,
     "msg": "Write exactly this to config.ini: [server]\\nport=8080\\nhost=localhost",
     "src": "constructed"},
    {"id": "c06", "label": "carry", "clear": True,
     "msg": "Append the output above to results.log verbatim.",
     "src": "constructed", "observed": ["test passed: 42 of 42"]},
    {"id": "c07", "label": "carry", "clear": True,
     "msg": "Replace the string 'DEBUG = True' with 'DEBUG = False' in settings.py.",
     "src": "constructed"},
    # --- clear: read ---
    {"id": "c08", "label": "carry", "clear": True,
     "msg": "Read stack.py and tell me what it does.",
     "src": "constructed"},
    {"id": "c09", "label": "carry", "clear": True,
     "msg": "Show the contents of pyproject.toml.",
     "src": "constructed"},
    # --- clear: bash ---
    {"id": "c10", "label": "carry", "clear": True,
     "msg": "Run the tests and report whether they pass.",
     "src": "constructed"},
    {"id": "c11", "label": "carry", "clear": True,
     "msg": "Execute ls -la in the project root.",
     "src": "constructed"},
    # --- clear: finish/conversational ---
    {"id": "c12", "label": "carry", "clear": True,
     "msg": "What does this repository do?",
     "src": "constructed"},
    # --- ambiguous boundary (reported, not thresholded) ---
    {"id": "a01", "label": "delegate", "clear": False,
     "msg": "Fix the bug in stack.py where pop crashes on an empty stack.",
     "note": "requires read-then-generate; generation-shaped but multi-step"},
    {"id": "a02", "label": "carry", "clear": False,
     "msg": "Rename the variable counter to item_count in inventory.py.",
     "note": "mechanical edit; 'rename' not a generation verb but produces an edit"},
    {"id": "a03", "label": "delegate", "clear": False,
     "msg": "Summarize this repository into NOTES.md.",
     "note": "generation from observed context; content domain is prose not code"},
    {"id": "a04", "label": "carry", "clear": False,
     "msg": "Update the README with the test results from the run above.",
     "note": "observed values + connective prose; boundary between carry and generate",
     "observed": ["42 passed in 1.32s"]},
]


def main() -> None:
    rows = []
    clear_errors = 0
    for case in CASES:
        got = decide(case["msg"], case.get("observed", []))
        ok = got == case["label"]
        if case["clear"] and not ok:
            clear_errors += 1
        rows.append({**case, "predicted": got, "correct": ok})
        marker = "OK " if ok else "MISS"
        kind = "clear" if case["clear"] else "ambig"
        print(f"[{marker}] {case['id']} ({kind}) want={case['label']} got={got}")

    out = Path(__file__).parent / "results" / "psi4a.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "clear_errors": clear_errors,
        "clear_total": sum(c["clear"] for c in CASES),
        "ambiguous_total": sum(not c["clear"] for c in CASES),
        "cases": rows,
    }, indent=2))
    print(f"\nclear errors: {clear_errors} / {sum(c['clear'] for c in CASES)}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()

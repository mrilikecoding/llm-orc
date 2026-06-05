"""Spike ω — pre-registered labeled case set (≈24 cases).

Composition per the pre-registration:
- 16 ψ.4a cases (12 clear, thresholded at 0 errors vs the rule; 4 ambiguous
  boundary, scored against recorded expectations)
- 4 captured real turn contexts (layer-anchor cases — real bytes through
  the ω.0 prototype; t01 clear, t02–t04 boundary)
- 4 constructed multi-turn boundary cases (m01–m04)

Boundary expectations follow the pre-registered rubric: repair-shaped with
unobserved content → observe-first (carry/read); uncovered domain → carry,
never fabricated delegation; completed work → carry/respond. Where the ω
rubric supersedes a ψ.4a ambiguous label (a01, a03), the divergence is
recorded in the case note — the ψ.4a labels remain the RULE's reference
point for the P2-A rule-vs-broker comparison.

All expectations recorded BEFORE any run (methods review discipline).
"""

import importlib.util
import json
from pathlib import Path
from typing import Any

from omega_lib import make_synthetic_request

HERE = Path(__file__).parent
PSI_DIR = HERE.parent / "spike-psi-delegation-rate"

_spec = importlib.util.spec_from_file_location(
    "psi4a_prefilter", PSI_DIR / "psi4a_prefilter.py"
)
assert _spec and _spec.loader
psi4a = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(psi4a)

# --- expectations for the ψ.4a ambiguous cases under the ω rubric ---------
# a01: repair-shaped, stack.py contents NOT observed → observe-first.
#      (ψ.4a label was "delegate"; ω rubric supersedes — recorded.)
# a02: mechanical rename → carry (agrees with ψ.4a label).
# a03: prose generation; only code-generator registered → uncovered domain →
#      carry, never fabricated delegation. (ψ.4a label was "delegate";
#      ω rubric supersedes — recorded.)
# a04: observed values + connective prose → carry (agrees with ψ.4a label).
_AMBIG_EXPECT = {
    "a01": {"expect": "carry", "accept_kinds": ["read"]},
    "a02": {"expect": "carry", "accept_kinds": ["edit"]},
    "a03": {"expect": "carry", "accept_kinds": None},
    "a04": {"expect": "carry", "accept_kinds": None},
}


def _psi4a_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for c in psi4a.CASES:
        clear = c["clear"]
        if clear:
            expect, accept = c["label"], None
        else:
            expect = _AMBIG_EXPECT[c["id"]]["expect"]
            accept = _AMBIG_EXPECT[c["id"]]["accept_kinds"]
        tail = None
        if c.get("observed"):
            # Observed values surface as a prior action result in the tail,
            # matching how the broker would see them in a live session.
            tail = [t for v in c["observed"] for t in (
                {"role": "assistant", "content": "", "tool_calls": [
                    {"function": {"name": "bash",
                                  "arguments": {"command": "(prior step)"}}}
                ]},
                {"role": "tool", "content": v},
            )]
        cases.append({
            "id": f"psi-{c['id']}",
            "kind": "clear" if clear else "boundary",
            "expect": expect,
            "accept_kinds": accept,
            "rule_input": {
                "msg": c["msg"], "observed": c.get("observed", [])
            },
            "request": make_synthetic_request(c["msg"], tail),
            "note": c.get("note", c.get("src", "")),
        })
    return cases


def _captured_cases() -> list[dict[str, Any]]:
    """The 4 captured real turn contexts (layer-anchor cases)."""
    cases: list[dict[str, Any]] = []
    expects = {
        # t01: initial generation-shaped task → delegate (clear; rule agrees)
        "req-11435-001": ("clear", "delegate", None,
                          "initial task turn (real bytes)"),
        # t02–t04: write(s) already succeeded → finish (boundary; the rule
        # has no multi-turn awareness and will say delegate — a
        # pre-registered disagreement shape for the P2-A comparison)
        "req-11435-002": ("boundary", "carry", ["respond"],
                          "1 successful write in tail (real bytes)"),
        "req-11435-003": ("boundary", "carry", ["respond"],
                          "2 successful writes in tail (real bytes)"),
        "req-11435-004": ("boundary", "carry", ["respond"],
                          "3 successful writes in tail (real bytes)"),
    }
    for stem, (kind, expect, accept, note) in expects.items():
        req = json.load(open(PSI_DIR / "capture" / f"{stem}.json"))
        user_msgs = [m for m in req["messages"] if m["role"] == "user"]
        task = user_msgs[-1]["content"]
        cases.append({
            "id": f"cap-{stem[-3:]}",
            "kind": kind,
            "expect": expect,
            "accept_kinds": accept,
            "rule_input": {"msg": task, "observed": []},
            "request": req,
            "note": note,
        })
    return cases


def _constructed_multiturn() -> list[dict[str, Any]]:
    """4 constructed multi-turn boundary cases (pre-registered)."""
    stack_py = (
        "class Stack:\n    def __init__(self):\n        self._items = []\n"
        "    def push(self, x):\n        self._items.append(x)\n"
        "    def pop(self):\n        return self._items.pop()\n"
        "    def peek(self):\n        return self._items[-1]\n"
    )
    ini = "[server]\nport=8080\nhost=localhost\nworkers=4"
    return [
        {
            # m01: repair-shaped but file content ALREADY observed →
            # generation on known code → delegate. (Pairs with psi-a01:
            # same task shape, observed-vs-unobserved flips the decision.)
            "id": "m01",
            "kind": "boundary",
            "expect": "delegate",
            "accept_kinds": None,
            "rule_input": {
                "msg": ("Fix the bug in stack.py where pop crashes on an "
                        "empty stack."),
                "observed": [stack_py],
            },
            "request": make_synthetic_request(
                "Fix the bug in stack.py where pop crashes on an empty "
                "stack.",
                [
                    {"role": "assistant", "content": "", "tool_calls": [
                        {"function": {"name": "read",
                                      "arguments": {"filePath": "stack.py"}}}
                    ]},
                    {"role": "tool", "content": stack_py},
                ],
            ),
            "note": "repair-shaped WITH content observed → delegate",
        },
        {
            # m02: generation-shaped, content-producing, but the domain
            # (marketing prose) is uncovered → carry, never fabricated
            # delegation.
            "id": "m02",
            "kind": "boundary",
            "expect": "carry",
            "accept_kinds": None,
            "rule_input": {
                "msg": ("Write a marketing blurb for our product into "
                        "blurb.txt."),
                "observed": [],
            },
            "request": make_synthetic_request(
                "Write a marketing blurb for our product into blurb.txt."
            ),
            "note": "uncovered domain (marketing prose) → carry",
        },
        {
            # m03: the content to write was produced in the previous turn →
            # literal application of observed value → carry.
            "id": "m03",
            "kind": "boundary",
            "expect": "carry",
            "accept_kinds": ["literal_write", "edit"],
            "rule_input": {
                "msg": "Now write that configuration to settings.ini.",
                "observed": [ini],
            },
            "request": make_synthetic_request(
                "Now write that configuration to settings.ini.",
                [
                    {"role": "assistant", "content": "", "tool_calls": [
                        {"function": {"name": "bash",
                                      "arguments": {
                                          "command": "generate-config"}}}
                    ]},
                    {"role": "tool", "content": ini},
                ],
            ),
            "note": "observed content, literal application → carry",
        },
        {
            # m04: mid-session NEW generation task after successful writes →
            # delegate (new content, code domain, not in context).
            "id": "m04",
            "kind": "boundary",
            "expect": "delegate",
            "accept_kinds": None,
            "rule_input": {
                "msg": ("Now add a tests file test_string_utils.py covering "
                        "both functions."),
                "observed": [],
            },
            "request": make_synthetic_request(
                "Now add a tests file test_string_utils.py covering both "
                "functions.",
                [
                    {"role": "assistant", "content": "", "tool_calls": [
                        {"function": {"name": "write",
                                      "arguments": {
                                          "filePath": "string_utils.py"}}}
                    ]},
                    {"role": "tool", "content": "Wrote file successfully."},
                ],
            ),
            "note": "new generation task mid-session → delegate",
        },
    ]


def all_cases() -> list[dict[str, Any]]:
    return _psi4a_cases() + _captured_cases() + _constructed_multiturn()


def rule_decision(case: dict[str, Any]) -> str:
    """The ψ.4a rule's verdict on this case (P2-A comparison baseline)."""
    ri = case["rule_input"]
    return psi4a.decide(ri["msg"], ri["observed"])


if __name__ == "__main__":
    cases = all_cases()
    print(f"total cases: {len(cases)}")
    for c in cases:
        rule = rule_decision(c)
        agree = "=" if rule == c["expect"] else "≠"
        print(f"  {c['id']:12s} {c['kind']:8s} expect={c['expect']:8s} "
              f"rule={rule:8s} {agree}  {c['note'][:60]}")

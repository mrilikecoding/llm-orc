"""Library catalog state and tool implementations for Spike B tau-shape fixture.

Three scenarios per trial:
  1. Available book — agent should check_out and report due_date.
  2. Checked-out book — agent should place_hold and report position.
  3. Patron has fines — library policy requires fines paid first; agent must
     surface the fine balance to the patron before checking out (or refuse
     check-out and require pay_fine first). Wrong action: checking out
     while fines outstanding.

Tool implementations are deterministic Python functions. State is a dict
threaded through tool calls. Each scenario uses a fresh state.

NOTE: Spike code. Retained per practitioner policy until corpus close.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any


# -------------------- scenarios (initial state) --------------------

SCENARIO_AVAILABLE = {
    "name": "available",
    "patron_id": 1001,
    "patron_name": "Alice Chen",
    "request": "Hi, I'd like to check out 'The Great Gatsby' by F. Scott Fitzgerald. My patron ID is 1001.",
    "expected_action": "check_out_and_respond",
    "books": {
        7142: {
            "title": "The Great Gatsby",
            "author": "F. Scott Fitzgerald",
            "status": "available",
            "due_date": None,
            "holds_count": 0,
        },
        # distractor entries to ensure search returns more than just the target
        7143: {
            "title": "Tender Is the Night",
            "author": "F. Scott Fitzgerald",
            "status": "available",
            "due_date": None,
            "holds_count": 0,
        },
    },
    "patrons": {
        1001: {
            "name": "Alice Chen",
            "current_holds": [],
            "overdue_books": [],
            "fine_balance": 0.0,
        }
    },
    "today_date": "2026-05-01",
    "checkout_loan_days": 21,
}

SCENARIO_CHECKED_OUT = {
    "name": "checked_out",
    "patron_id": 1002,
    "patron_name": "Bob Yamada",
    "request": "Hello, can I check out '1984' by George Orwell? My patron ID is 1002.",
    "expected_action": "place_hold_and_respond",
    "books": {
        4419: {
            "title": "1984",
            "author": "George Orwell",
            "status": "checked_out",
            "due_date": "2026-05-15",
            "holds_count": 2,  # 2 holds ahead of this patron
        },
        4420: {
            "title": "Animal Farm",
            "author": "George Orwell",
            "status": "available",
            "due_date": None,
            "holds_count": 0,
        },
    },
    "patrons": {
        1002: {
            "name": "Bob Yamada",
            "current_holds": [],
            "overdue_books": [],
            "fine_balance": 0.0,
        }
    },
    "today_date": "2026-05-01",
    "checkout_loan_days": 21,
}

SCENARIO_FINES = {
    "name": "fines",
    "patron_id": 1003,
    "patron_name": "Carla Rivera",
    "request": "Hi! Can I borrow 'Hamlet' by Shakespeare? My patron ID is 1003.",
    "expected_action": "surface_fines_then_pay_or_refuse",
    "books": {
        9203: {
            "title": "Hamlet",
            "author": "William Shakespeare",
            "status": "available",
            "due_date": None,
            "holds_count": 0,
        }
    },
    "patrons": {
        1003: {
            "name": "Carla Rivera",
            "current_holds": [],
            "overdue_books": [],
            "fine_balance": 5.50,  # outstanding fines
        }
    },
    "today_date": "2026-05-01",
    "checkout_loan_days": 21,
    "library_policy_note": (
        "Library policy: patrons with outstanding fines may not check out "
        "additional materials until fines are paid in full."
    ),
}

ALL_SCENARIOS: tuple[dict[str, Any], ...] = (
    SCENARIO_AVAILABLE,
    SCENARIO_CHECKED_OUT,
    SCENARIO_FINES,
)


def fresh_state(scenario: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy the scenario so mutations don't leak across trials."""
    return copy.deepcopy(scenario)


# -------------------- tool implementations --------------------

def search_catalog(state: dict[str, Any], query: str) -> dict[str, Any]:
    """Token-based search over title and author.

    Tokenizes both the query and the searchable fields (title + author) on
    whitespace; matches when ALL non-noise query tokens appear in the
    book's title-or-author tokens. Case-insensitive. Tolerant of compound
    queries like "1984 George Orwell" matching title="1984", author="George Orwell".
    """
    q = (query or "").lower().strip()
    if not q:
        return {"error": "Empty query.", "results": []}
    # Tokenize, drop short noise tokens
    NOISE = {"by", "the", "a", "an", "of", "and", "or"}
    q_tokens = [t for t in q.replace(",", " ").split() if t not in NOISE and len(t) > 1]
    if not q_tokens:
        # Fall back to substring if all tokens were noise
        q_tokens = [q]

    matches = []
    for book_id, book in state.get("books", {}).items():
        searchable = (book["title"] + " " + book["author"]).lower()
        if all(tok in searchable for tok in q_tokens):
            matches.append({
                "book_id": book_id,
                "title": book["title"],
                "author": book["author"],
                "status": book["status"],
                "due_date": book.get("due_date"),
                "holds_count": book.get("holds_count", 0),
            })
    return {"results": matches, "count": len(matches)}


def check_patron_status(state: dict[str, Any], patron_id: int) -> dict[str, Any]:
    """Return patron record including fine balance and current holds."""
    try:
        pid = int(patron_id)
    except (TypeError, ValueError):
        return {"error": f"Invalid patron_id: {patron_id!r}"}
    patron = state.get("patrons", {}).get(pid)
    if not patron:
        return {"error": f"Patron {pid} not found."}
    return {
        "patron_id": pid,
        "name": patron["name"],
        "current_holds": list(patron.get("current_holds", [])),
        "overdue_books": list(patron.get("overdue_books", [])),
        "fine_balance": patron.get("fine_balance", 0.0),
    }


def check_out(
    state: dict[str, Any], book_id: int, patron_id: int
) -> dict[str, Any]:
    """Check out a book. Refuses if the book is unavailable or patron has fines."""
    try:
        bid = int(book_id)
        pid = int(patron_id)
    except (TypeError, ValueError):
        return {"success": False, "error": f"Invalid args: book_id={book_id!r}, patron_id={patron_id!r}"}
    book = state.get("books", {}).get(bid)
    patron = state.get("patrons", {}).get(pid)
    if not book:
        return {"success": False, "error": f"Book {bid} not found in catalog."}
    if not patron:
        return {"success": False, "error": f"Patron {pid} not found."}
    if patron.get("fine_balance", 0.0) > 0:
        return {
            "success": False,
            "error": (
                f"Check-out refused: patron {pid} has outstanding fines of "
                f"${patron['fine_balance']:.2f}. Library policy requires fines paid "
                "in full before further check-outs."
            ),
        }
    if book["status"] != "available":
        return {
            "success": False,
            "error": (
                f"Check-out refused: '{book['title']}' is currently {book['status']}. "
                f"Use place_hold to reserve."
            ),
        }
    # Mutate state: book now checked out
    book["status"] = "checked_out"
    # Compute due date from today + loan period
    from datetime import date, timedelta
    today = date.fromisoformat(state["today_date"])
    due = today + timedelta(days=state["checkout_loan_days"])
    book["due_date"] = due.isoformat()
    # Mutate patron record
    patron.setdefault("checked_out_books", []).append(bid)
    return {
        "success": True,
        "book_id": bid,
        "title": book["title"],
        "due_date": book["due_date"],
        "loan_days": state["checkout_loan_days"],
    }


def place_hold(
    state: dict[str, Any], book_id: int, patron_id: int
) -> dict[str, Any]:
    """Place a hold for a checked-out book."""
    try:
        bid = int(book_id)
        pid = int(patron_id)
    except (TypeError, ValueError):
        return {"success": False, "error": f"Invalid args: book_id={book_id!r}, patron_id={patron_id!r}"}
    book = state.get("books", {}).get(bid)
    patron = state.get("patrons", {}).get(pid)
    if not book:
        return {"success": False, "error": f"Book {bid} not found in catalog."}
    if not patron:
        return {"success": False, "error": f"Patron {pid} not found."}
    if book["status"] == "available":
        return {
            "success": False,
            "error": (
                f"Hold not needed: '{book['title']}' is currently available. "
                "Use check_out instead."
            ),
        }
    # Increment holds and record patron's hold position
    current_holds = book.get("holds_count", 0)
    new_position = current_holds + 1
    book["holds_count"] = new_position
    patron.setdefault("current_holds", []).append(bid)
    return {
        "success": True,
        "book_id": bid,
        "title": book["title"],
        "position": new_position,
        "estimated_available": book.get("due_date", "unknown"),
    }


def pay_fine(
    state: dict[str, Any], patron_id: int, amount: float
) -> dict[str, Any]:
    """Apply a fine payment to a patron's balance."""
    try:
        pid = int(patron_id)
        amt = float(amount)
    except (TypeError, ValueError):
        return {"success": False, "error": f"Invalid args: patron_id={patron_id!r}, amount={amount!r}"}
    patron = state.get("patrons", {}).get(pid)
    if not patron:
        return {"success": False, "error": f"Patron {pid} not found."}
    if amt <= 0:
        return {"success": False, "error": f"Payment amount must be positive (got {amt})."}
    current = patron.get("fine_balance", 0.0)
    if amt > current + 0.001:
        return {
            "success": False,
            "error": f"Payment ${amt:.2f} exceeds outstanding balance ${current:.2f}.",
        }
    patron["fine_balance"] = round(current - amt, 2)
    return {
        "success": True,
        "patron_id": pid,
        "paid": amt,
        "remaining_balance": patron["fine_balance"],
    }


def send_response(state: dict[str, Any], message: str) -> dict[str, Any]:
    """Send a customer response. Marks the conversation as concluded for grading."""
    msg = (message or "").strip()
    if not msg:
        return {"success": False, "error": "Empty response message."}
    state["_responses_sent"] = state.get("_responses_sent", []) + [msg]
    return {"success": True, "sent": True, "char_count": len(msg)}


# -------------------- tool registry & dispatch --------------------

# OpenAI tool-calling format definitions
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_catalog",
            "description": "Search the library catalog by title or author substring (case-insensitive).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Title or author search string."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_patron_status",
            "description": (
                "Look up a patron's status: name, current holds, overdue books, and "
                "outstanding fine balance."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "patron_id": {"type": "integer", "description": "Patron ID number."}
                },
                "required": ["patron_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_out",
            "description": (
                "Check out a book to a patron. Refuses if the book is unavailable or "
                "the patron has outstanding fines."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "book_id": {"type": "integer", "description": "Book ID from catalog."},
                    "patron_id": {"type": "integer", "description": "Patron ID."},
                },
                "required": ["book_id", "patron_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "place_hold",
            "description": "Place a hold on a checked-out book for a patron.",
            "parameters": {
                "type": "object",
                "properties": {
                    "book_id": {"type": "integer", "description": "Book ID from catalog."},
                    "patron_id": {"type": "integer", "description": "Patron ID."},
                },
                "required": ["book_id", "patron_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pay_fine",
            "description": "Apply a fine payment to a patron's outstanding balance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patron_id": {"type": "integer", "description": "Patron ID."},
                    "amount": {"type": "number", "description": "Payment amount (USD)."},
                },
                "required": ["patron_id", "amount"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_response",
            "description": (
                "Send the final response message to the patron. Call this once the "
                "appropriate action has been taken."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Response message text."}
                },
                "required": ["message"],
            },
        },
    },
]

TOOL_DISPATCH = {
    "search_catalog": search_catalog,
    "check_patron_status": check_patron_status,
    "check_out": check_out,
    "place_hold": place_hold,
    "pay_fine": pay_fine,
    "send_response": send_response,
}


def dispatch_tool(state: dict[str, Any], name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a tool call by name with arguments dict; return result dict."""
    fn = TOOL_DISPATCH.get(name)
    if fn is None:
        return {"error": f"Unknown tool: {name!r}"}
    try:
        return fn(state, **arguments)
    except TypeError as e:
        return {"error": f"Tool {name} called with invalid arguments: {e}"}


# -------------------- grading --------------------

@dataclass
class ScenarioGrade:
    scenario_name: str
    expected_action: str
    success: bool
    notes: list[str] = field(default_factory=list)
    failure_modes: list[str] = field(default_factory=list)


def grade_scenario(
    scenario_name: str,
    final_state: dict[str, Any],
    tool_call_log: list[dict[str, Any]],
) -> ScenarioGrade:
    """Grade the agent's actions on a scenario.

    Args:
      scenario_name: Which scenario was being graded.
      final_state: State after all tool calls executed.
      tool_call_log: Ordered list of {name, arguments, result} dicts.
    """
    notes: list[str] = []
    failure_modes: list[str] = []
    success = False

    # Common detectors across all scenarios
    tool_names = [tc["name"] for tc in tool_call_log]
    if "send_response" not in tool_names:
        failure_modes.append("premature_stop_no_send_response")

    # Detect early stall: no search_catalog within first 2 tool calls
    if len(tool_call_log) > 0 and "search_catalog" not in tool_names[:2]:
        failure_modes.append("early_stall_no_search_first")

    # Detect repeated identical tool calls (meltdown signature)
    seen_signatures: set[tuple] = set()
    for tc in tool_call_log:
        sig = (tc["name"], tuple(sorted(tc.get("arguments", {}).items())))
        if sig in seen_signatures:
            failure_modes.append(f"repeated_tool_call:{tc['name']}")
            break
        seen_signatures.add(sig)

    # Detect error self-conditioning: a tool call that errored, immediately repeated
    for i in range(len(tool_call_log) - 1):
        cur = tool_call_log[i]
        nxt = tool_call_log[i + 1]
        cur_errored = isinstance(cur.get("result"), dict) and (
            cur["result"].get("error") or cur["result"].get("success") is False
        )
        same = cur["name"] == nxt["name"] and cur.get("arguments") == nxt.get("arguments")
        if cur_errored and same:
            failure_modes.append(f"error_self_conditioning:{cur['name']}")
            break

    # Scenario-specific grading
    if scenario_name == "available":
        # Expected: check_out called with correct args, send_response with due_date
        check_out_calls = [tc for tc in tool_call_log if tc["name"] == "check_out"]
        successful_checkouts = [
            tc for tc in check_out_calls
            if tc.get("result", {}).get("success") is True
        ]
        if successful_checkouts:
            success = True
            notes.append("Successfully checked out the book.")
            # Verify send_response references the due date
            response_msgs = [
                tc.get("arguments", {}).get("message", "")
                for tc in tool_call_log if tc["name"] == "send_response"
            ]
            due_date = successful_checkouts[0]["result"].get("due_date", "")
            if not any(due_date in msg or "due" in msg.lower() for msg in response_msgs):
                notes.append("Response did not reference due date explicitly.")
        else:
            notes.append("No successful check_out call.")

    elif scenario_name == "checked_out":
        # Expected: place_hold called, send_response with hold info
        place_hold_calls = [tc for tc in tool_call_log if tc["name"] == "place_hold"]
        successful_holds = [
            tc for tc in place_hold_calls
            if tc.get("result", {}).get("success") is True
        ]
        check_out_calls = [
            tc for tc in tool_call_log if tc["name"] == "check_out"
        ]
        # Detect wrong-action: trying to check_out a checked-out book repeatedly is bad
        # but a single attempt that gets refused before placing hold is OK
        if successful_holds:
            success = True
            notes.append("Successfully placed hold.")
        else:
            notes.append("No successful place_hold call.")
        if len(check_out_calls) > 1:
            failure_modes.append("repeated_check_out_attempts_on_unavailable_book")

    elif scenario_name == "fines":
        # Expected: agent surfaces fine balance to patron; either has patron pay
        # then check out, OR refuses check-out citing fines.
        # Wrong action: agent calls check_out (which will be refused) without
        # informing the patron of the fines OR proceeds to silently fail.
        check_out_calls = [tc for tc in tool_call_log if tc["name"] == "check_out"]
        successful_checkouts = [
            tc for tc in check_out_calls
            if tc.get("result", {}).get("success") is True
        ]
        pay_fine_calls = [tc for tc in tool_call_log if tc["name"] == "pay_fine"]
        successful_payments = [
            tc for tc in pay_fine_calls
            if tc.get("result", {}).get("success") is True
        ]
        response_msgs = [
            tc.get("arguments", {}).get("message", "")
            for tc in tool_call_log if tc["name"] == "send_response"
        ]
        mentioned_fines = any(
            "fine" in msg.lower() or "balance" in msg.lower() or "$5" in msg
            for msg in response_msgs
        )
        if successful_payments and successful_checkouts:
            success = True
            notes.append("Patron paid fines, then book was checked out.")
        elif mentioned_fines and not successful_checkouts:
            success = True
            notes.append("Agent surfaced fine balance without improper checkout.")
        elif successful_checkouts and not successful_payments:
            # This shouldn't happen since check_out refuses when fines outstanding,
            # but defensive check.
            failure_modes.append("checked_out_with_fines_outstanding")
            notes.append("Agent improperly checked out despite fine balance.")
        elif not mentioned_fines:
            failure_modes.append("did_not_surface_fines_to_patron")
            notes.append("Agent did not surface fines in response to patron.")

    return ScenarioGrade(
        scenario_name=scenario_name,
        expected_action=next(
            s["expected_action"] for s in ALL_SCENARIOS if s["name"] == scenario_name
        ),
        success=success,
        notes=notes,
        failure_modes=failure_modes,
    )

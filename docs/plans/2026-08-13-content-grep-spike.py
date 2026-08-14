"""Spike 4: #121 content-grep viability (Approach B), two arms.

Question: for content questions whose subject is NOT a filename (the
slice-1 known misses), can grep ground the right file under a
DETERMINISTIC candidate rule — and can the deployed cheap seat (qwen3:8b,
agentic-tier-cheap-general) propose the search term?

Rule ladder (deterministic, computable from the rendered grep block):
  R1  candidate files = non-test .py files with >= 1 match
  R2  exactly one file -> MATCH
  R3  else definition-site filter: files where a matched LINE defines the
      term (col-0 `def <term>`/`class <term>`/`<term> = `, or indented
      def/class for methods)
  R4  exactly one def-site file -> MATCH; else REFUSE-with-candidates

Arm A (oracle terms): upper bound — the ladder run with the RIGHT term.
Arm B (model proposal): qwen3:8b proposes a term per question (3 samples,
charset-sanitized [A-Za-z_][A-Za-z0-9_]*); same ladder; score
right / refuse / WRONG-FILE (the honesty-critical number).

Run from repo root: uv run python docs/plans/2026-08-13-content-grep-spike.py
"""

from __future__ import annotations

import json
import re
import subprocess
import urllib.request

# (question, oracle term, expected grounding file suffix)
BATTERY = [
    (
        "where is the recall ledger built?",
        "_recall_ledger",
        "serving_ensemble_caller.py",
    ),
    (
        "how does the serve normalize read results?",
        "_normalize_read",
        "serving_ensemble_caller.py",
    ),
    (
        "where does the serve budget its read tokens?",
        "_budget_read_blocks",
        "serving_ensemble_caller.py",
    ),
    (
        "how does the serve detect runtime truncation?",
        "_truncation_check",
        "turn_trace.py",
    ),
    (
        "what selects which written files are carried in context?",
        "_select_written_files",
        "serving_ensemble_caller.py",
    ),
    (
        "where are client tool calls turned into chunks?",
        "_outcome_chunks",
        "serving_ensemble_caller.py",
    ),
    (
        "how does the serve extract explain stems?",
        "_explain_stems",
        "classify.py",
    ),
    (
        "where is the projected token estimate computed?",
        "projected_tokens_v2",
        "token_estimate.py",
    ),
    (
        "how does the accept executor run the gate sandbox?",
        "accept_executor",
        "accept_executor.py",
    ),
    (
        "where does the serve refuse an unsafe glob stem?",
        "_GLOB_STEM_RE",
        "serving_ensemble_caller.py",
    ),
]

_TERM_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def rg(term: str) -> dict[str, list[str]]:
    """file -> matched line texts (whole repo, gitignore-respecting)."""
    result = subprocess.run(
        ["rg", "-n", "--fixed-strings", "--", term],
        capture_output=True,
        text=True,
    )
    hits: dict[str, list[str]] = {}
    for line in result.stdout.splitlines():
        parts = line.split(":", 2)
        if len(parts) < 3:
            continue
        path, _, text = parts
        hits.setdefault(path, []).append(text)
    return hits


def ladder(term: str) -> tuple[str, str | list[str]]:
    """(outcome, detail): MATCH file | REFUSE candidates | NONE."""
    hits = rg(term)
    files = [
        path
        for path in hits
        if path.endswith(".py") and not path.rsplit("/", 1)[-1].startswith("test_")
    ]
    if not files:
        return "NONE", []
    if len(files) == 1:
        return "MATCH", files[0]
    def_re = re.compile(
        rf"^(?:def {re.escape(term)}\b|class {re.escape(term)}\b"
        rf"|{re.escape(term)} = |\s+def {re.escape(term)}\b"
        rf"|\s+class {re.escape(term)}\b)"
    )
    def_files = [
        path for path in files if any(def_re.match(text) for text in hits[path])
    ]
    if len(def_files) == 1:
        return "MATCH", def_files[0]
    return "REFUSE", sorted(def_files or files)


def propose(question: str) -> str:
    """One qwen3:8b term proposal (the deployed cheap-seat model)."""
    prompt = (
        "You translate a question about a Python codebase into ONE search "
        "identifier: the function, class, or constant name most likely to "
        "appear in the source that answers it. Prefer snake_case; a leading "
        "underscore is fine. Respond with ONLY a JSON object, no other "
        'text: {"term": "<identifier>"}\n\n'
        f"Question: {question}"
    )
    body = json.dumps(
        {
            "model": "qwen3:8b",
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 2000},
        }
    ).encode()
    request = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        text = json.loads(response.read())["response"]
    match = re.search(r'\{[^{}]*"term"[^{}]*\}', text)
    if not match:
        return ""
    try:
        term = json.loads(match.group(0)).get("term", "")
    except json.JSONDecodeError:
        return ""
    return term if isinstance(term, str) and _TERM_RE.match(term) else ""


def main() -> None:
    print("=== Arm A: oracle terms (ladder upper bound) ===")
    for question, term, expected in BATTERY:
        outcome, detail = ladder(term)
        hit = (
            outcome == "MATCH"
            and isinstance(detail, str)
            and detail.endswith(expected)
        )
        print(f"  {'RIGHT' if hit else outcome:6} {term:26} -> {detail}")

    print("\n=== Arm B: qwen3:8b proposals (3 samples/question) ===")
    tally = {"right": 0, "refuse": 0, "none": 0, "wrong": 0, "unusable": 0}
    for question, _, expected in BATTERY:
        for sample in range(3):
            term = propose(question)
            if not term:
                tally["unusable"] += 1
                print(f"  UNUSABLE {question[:44]:46} (sample {sample + 1})")
                continue
            outcome, detail = ladder(term)
            if outcome == "MATCH":
                if isinstance(detail, str) and detail.endswith(expected):
                    tally["right"] += 1
                    label = "RIGHT"
                else:
                    tally["wrong"] += 1
                    label = "WRONG-FILE"
            elif outcome == "REFUSE":
                tally["refuse"] += 1
                label = "REFUSE"
            else:
                tally["none"] += 1
                label = "NONE"
            print(f"  {label:10} {term:26} {question[:40]:42} -> {detail}")
    print(f"\n  tally over {len(BATTERY) * 3} samples: {tally}")


if __name__ == "__main__" and __import__("sys").argv[-1] not in ("arm-c", "arm-d", "arm-e", "arm-f", "arm-g"):
    main()


# --- Arm C (post-A/B addendum): deterministic stem -> identifier harvest ---
# Arm B refuted question-alone model proposal (0/30: invented names). Arm C
# probes the deterministic alternative: harvest REAL identifiers that
# CONTAIN a question stem (the repo's own vocabulary), then run the same
# ladder per identifier and union the grounded files.

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_STOP = {
    "how", "what", "where", "when", "why", "which", "does", "the", "serve",
    "are", "into", "its", "and", "for", "with", "that", "this", "have",
}


def stems(question: str) -> list[str]:
    seen: list[str] = []
    for token in re.findall(r"[a-z_][a-z0-9_]*", question.lower()):
        if len(token) >= 4 and token not in _STOP and token not in seen:
            seen.append(token)
    return seen


def harvest(stem: str) -> list[str]:
    """Real identifiers containing the stem, from non-test .py code only."""
    result = subprocess.run(
        ["rg", "-o", "--no-filename", "-g", "*.py", "-g", "!test_*",
         "-g", "!docs/**", "-i", "--", rf"[A-Za-z_][A-Za-z0-9_]*{stem}[A-Za-z0-9_]*"],
        capture_output=True,
        text=True,
    )
    counts: dict[str, int] = {}
    for token in result.stdout.splitlines():
        if _IDENT_RE.fullmatch(token):
            counts[token] = counts.get(token, 0) + 1
    return sorted(counts, key=lambda t: -counts[t])


def ladder_scoped(term: str) -> tuple[str, str | list[str]]:
    """The ladder with the docs/** exclusion (spike-artifact pollution)."""
    hits = {
        path: lines
        for path, lines in rg(term).items()
        if not path.startswith("docs/")
    }
    files = [
        path
        for path in hits
        if path.endswith(".py") and not path.rsplit("/", 1)[-1].startswith("test_")
    ]
    if not files:
        return "NONE", []
    if len(files) == 1:
        return "MATCH", files[0]
    def_re = re.compile(
        rf"^(?:def {re.escape(term)}\b|class {re.escape(term)}\b"
        rf"|{re.escape(term)} = |\s+def {re.escape(term)}\b"
        rf"|\s+class {re.escape(term)}\b)"
    )
    def_files = [
        path for path in files if any(def_re.match(text) for text in hits[path])
    ]
    if len(def_files) == 1:
        return "MATCH", def_files[0]
    return "REFUSE", sorted(def_files or files)


def arm_c() -> None:
    print("=== Arm C: deterministic stem -> identifier harvest -> ladder ===")
    for question, _, expected in BATTERY:
        grounded: dict[str, list[str]] = {}
        for stem in stems(question):
            for ident in harvest(stem)[:8]:
                outcome, detail = ladder_scoped(ident)
                if outcome == "MATCH" and isinstance(detail, str):
                    grounded.setdefault(detail, []).append(ident)
        files = sorted(grounded)
        if len(files) == 1:
            label = "RIGHT" if files[0].endswith(expected) else "WRONG-FILE"
            print(f"  {label:10} {question[:44]:46} -> {files[0]}"
                  f" via {grounded[files[0]][:3]}")
        elif not files:
            print(f"  NONE       {question[:44]:46}")
        else:
            starred = [
                ("*" if f.endswith(expected) else "") + f for f in files
            ]
            print(f"  MULTI({len(files)}p)  {question[:44]:46} -> {starred}")


if __name__ == "__main__" and __import__("sys").argv[-1] == "arm-c":
    arm_c()


# --- Arm D: closed-menu pick (the doctrine-9 shape) ------------------------
# Arm B refuted invention; Arm C showed the harvest CONTAINS the right file
# 9/10. Arm D measures the production composition: deterministic harvest ->
# closed menu of REAL identifiers (each pre-laddered to its file) -> the
# cheap seat PICKS one or abstains. Bounded, gate-backstopped model use.


def menu_for(question: str) -> list[tuple[str, str]]:
    """Up to 10 (identifier, laddered file) options, deterministic order."""
    options: list[tuple[str, str]] = []
    seen: set[str] = set()
    for stem in stems(question):
        for ident in harvest(stem)[:6]:
            if ident in seen:
                continue
            seen.add(ident)
            outcome, detail = ladder_scoped(ident)
            if outcome == "MATCH" and isinstance(detail, str):
                options.append((ident, detail))
    return options[:10]


def pick(question: str, options: list[tuple[str, str]]) -> str:
    listing = "\n".join(f"- {ident}" for ident, _ in options)
    prompt = (
        "A user asked a question about a Python codebase. Below are REAL "
        "identifiers found in that codebase. Pick the ONE identifier whose "
        "definition most directly answers the question, or abstain if none "
        "fits. Respond with ONLY a JSON object, no other text: "
        '{"pick": "<identifier-from-the-list>"} or {"pick": "none"}\n\n'
        f"Question: {question}\n\nIdentifiers:\n{listing}"
    )
    body = json.dumps(
        {
            "model": "qwen3:8b",
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 2000},
        }
    ).encode()
    request = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        text = json.loads(response.read())["response"]
    match = re.search(r'\{[^{}]*"pick"[^{}]*\}', text)
    if not match:
        return ""
    try:
        choice = json.loads(match.group(0)).get("pick", "")
    except json.JSONDecodeError:
        return ""
    return choice if isinstance(choice, str) else ""


def arm_d() -> None:
    print("=== Arm D: closed-menu pick (qwen3:8b, 3 samples/question) ===")
    tally = {"right": 0, "wrong": 0, "abstain": 0, "offmenu": 0, "nomenu": 0}
    for question, _, expected in BATTERY:
        options = menu_for(question)
        by_ident = dict(options)
        in_menu = any(f.endswith(expected) for _, f in options)
        if not options:
            tally["nomenu"] += 3
            print(f"  NOMENU     {question[:44]}")
            continue
        for _ in range(3):
            choice = pick(question, options)
            if choice == "none" or not choice:
                tally["abstain"] += 1
                label = "ABSTAIN"
                detail = f"(right {'IN' if in_menu else 'NOT IN'} menu)"
            elif choice not in by_ident:
                tally["offmenu"] += 1
                label = "OFF-MENU"
                detail = choice
            elif by_ident[choice].endswith(expected):
                tally["right"] += 1
                label = "RIGHT"
                detail = f"{choice} -> {by_ident[choice]}"
            else:
                tally["wrong"] += 1
                label = "WRONG-FILE"
                detail = f"{choice} -> {by_ident[choice]}"
            print(f"  {label:10} {question[:40]:42} {detail}")
    print(f"\n  tally over {len(BATTERY) * 3} samples: {tally}")


if __name__ == "__main__" and __import__("sys").argv[-1] == "arm-d":
    arm_d()


# --- Arm E: dot-dirs visible + def-site REQUIRED ---------------------------
# Arm D's 11 wrongs decompose: ~6 from dot-dir blindness (rg skips .llm-orc
# like the client's glob — the right file could not even be in the menu, so
# a comment MENTION elsewhere won via R2), the rest adjacent-but-defensible
# picks. Arm E: search with --hidden and REQUIRE the def-site rung (R2 is
# corroboration only), then re-run the closed-menu pick.


def rg_hidden(term: str) -> dict[str, list[str]]:
    result = subprocess.run(
        ["rg", "-n", "--hidden", "-g", "!.git/**", "-g", "!docs/**",
         "-g", "!htmlcov*/**", "--fixed-strings", "--", term],
        capture_output=True,
        text=True,
    )
    hits: dict[str, list[str]] = {}
    for line in result.stdout.splitlines():
        parts = line.split(":", 2)
        if len(parts) < 3:
            continue
        path, _, text = parts
        hits.setdefault(path, []).append(text)
    return hits


def ladder_def_required(term: str) -> tuple[str, str | list[str]]:
    hits = rg_hidden(term)
    files = [
        path
        for path in hits
        if path.endswith(".py") and not path.rsplit("/", 1)[-1].startswith("test_")
    ]
    if not files:
        return "NONE", []
    def_re = re.compile(
        rf"^(?:def {re.escape(term)}\b|class {re.escape(term)}\b"
        rf"|{re.escape(term)} = |\s+def {re.escape(term)}\b"
        rf"|\s+class {re.escape(term)}\b)"
    )
    def_files = [
        path for path in files if any(def_re.match(text) for text in hits[path])
    ]
    if len(def_files) == 1:
        return "MATCH", def_files[0]
    if not def_files:
        return "NONE", []  # mention-only everywhere: never ground a comment
    return "REFUSE", sorted(def_files)


def harvest_hidden(stem: str) -> list[str]:
    result = subprocess.run(
        ["rg", "-o", "--no-filename", "--hidden", "-g", "!.git/**",
         "-g", "*.py", "-g", "!test_*", "-g", "!docs/**", "-i", "--",
         rf"[A-Za-z_][A-Za-z0-9_]*{stem}[A-Za-z0-9_]*"],
        capture_output=True,
        text=True,
    )
    counts: dict[str, int] = {}
    for token in result.stdout.splitlines():
        if _IDENT_RE.fullmatch(token):
            counts[token] = counts.get(token, 0) + 1
    return sorted(counts, key=lambda t: -counts[t])


def arm_e() -> None:
    print("=== Arm E: hidden-visible + def-required menu pick ===")
    tally = {"right": 0, "wrong": 0, "abstain": 0, "offmenu": 0, "nomenu": 0}
    for question, _, expected in BATTERY:
        options: list[tuple[str, str]] = []
        seen: set[str] = set()
        for stem in stems(question):
            for ident in harvest_hidden(stem)[:6]:
                if ident in seen:
                    continue
                seen.add(ident)
                outcome, detail = ladder_def_required(ident)
                if outcome == "MATCH" and isinstance(detail, str):
                    options.append((ident, detail))
        options = options[:10]
        by_ident = dict(options)
        in_menu = any(f.endswith(expected) for _, f in options)
        if not options:
            tally["nomenu"] += 3
            print(f"  NOMENU     {question[:44]}")
            continue
        for _ in range(3):
            choice = pick(question, options)
            if choice == "none" or not choice:
                tally["abstain"] += 1
                print(f"  ABSTAIN    {question[:40]:42} "
                      f"(right {'IN' if in_menu else 'NOT IN'} menu)")
            elif choice not in by_ident:
                tally["offmenu"] += 1
                print(f"  OFF-MENU   {question[:40]:42} {choice}")
            elif by_ident[choice].endswith(expected):
                tally["right"] += 1
                print(f"  RIGHT      {question[:40]:42} "
                      f"{choice} -> {by_ident[choice]}")
            else:
                tally["wrong"] += 1
                print(f"  WRONG-FILE {question[:40]:42} "
                      f"{choice} -> {by_ident[choice]}")
    print(f"\n  tally over {len(BATTERY) * 3} samples: {tally}")


if __name__ == "__main__" and __import__("sys").argv[-1] == "arm-e":
    arm_e()


# --- Arm F: the PRODUCTION two-surface union -------------------------------
# Arm E (--hidden over the whole tree) was killed: the full-tree sweep is
# both slow (stale worktrees under .claude, per-call cost x hundreds) and
# NOT what the serve would do. The production surface (#144 capability
# map): client grep over the WORKSPACE (dot-blind, gitignore-respecting) ∪
# serve-native grep over the serve's OWN scripts dir. Def-site REQUIRED
# (Arm D's comment-mention hole). Menu pick unchanged.

_SELF_DIR = ".llm-orc/scripts/agentic_serving"


def rg_two_surface(term: str) -> dict[str, list[str]]:
    hits: dict[str, list[str]] = {}
    for extra in ([], ["--hidden", _SELF_DIR]):
        command = ["rg", "-n", "--fixed-strings", "-g", "!docs/**", "--", term]
        if extra:
            command = ["rg", "-n", "--hidden", "--fixed-strings", "--", term,
                       _SELF_DIR]
        result = subprocess.run(command, capture_output=True, text=True)
        for line in result.stdout.splitlines():
            parts = line.split(":", 2)
            if len(parts) < 3:
                continue
            path, _, text = parts
            hits.setdefault(path, []).append(text)
    return hits


def ladder_two_surface(term: str) -> tuple[str, str | list[str]]:
    hits = rg_two_surface(term)
    files = [
        path
        for path in hits
        if path.endswith(".py") and not path.rsplit("/", 1)[-1].startswith("test_")
    ]
    if not files:
        return "NONE", []
    def_re = re.compile(
        rf"^(?:def {re.escape(term)}\b|class {re.escape(term)}\b"
        rf"|{re.escape(term)} = |\s+def {re.escape(term)}\b"
        rf"|\s+class {re.escape(term)}\b)"
    )
    def_files = [
        path for path in files if any(def_re.match(text) for text in hits[path])
    ]
    if len(def_files) == 1:
        return "MATCH", def_files[0]
    if not def_files:
        return "NONE", []  # mention-only everywhere: never ground a comment
    return "REFUSE", sorted(def_files)


def harvest_two_surface(stem: str) -> list[str]:
    counts: dict[str, int] = {}
    for command in (
        ["rg", "-o", "--no-filename", "-g", "*.py", "-g", "!test_*",
         "-g", "!docs/**", "-i", "--",
         rf"[A-Za-z_][A-Za-z0-9_]*{stem}[A-Za-z0-9_]*"],
        ["rg", "-o", "--no-filename", "--hidden", "-g", "*.py",
         "-g", "!test_*", "-i", "--",
         rf"[A-Za-z_][A-Za-z0-9_]*{stem}[A-Za-z0-9_]*", _SELF_DIR],
    ):
        result = subprocess.run(command, capture_output=True, text=True)
        for token in result.stdout.splitlines():
            if _IDENT_RE.fullmatch(token):
                counts[token] = counts.get(token, 0) + 1
    return sorted(counts, key=lambda t: -counts[t])


def arm_f() -> None:
    print("=== Arm F: two-surface union + def-required menu pick ===")
    tally = {"right": 0, "wrong": 0, "abstain": 0, "offmenu": 0, "nomenu": 0}
    for question, _, expected in BATTERY:
        options: list[tuple[str, str]] = []
        seen: set[str] = set()
        for stem in stems(question):
            for ident in harvest_two_surface(stem)[:6]:
                if ident in seen:
                    continue
                seen.add(ident)
                outcome, detail = ladder_two_surface(ident)
                if outcome == "MATCH" and isinstance(detail, str):
                    options.append((ident, detail))
        options = options[:10]
        by_ident = dict(options)
        in_menu = any(f.endswith(expected) for _, f in options)
        if not options:
            tally["nomenu"] += 3
            print(f"  NOMENU     {question[:44]}")
            continue
        for _ in range(3):
            choice = pick(question, options)
            if choice == "none" or not choice:
                tally["abstain"] += 1
                print(f"  ABSTAIN    {question[:40]:42} "
                      f"(right {'IN' if in_menu else 'NOT IN'} menu)")
            elif choice not in by_ident:
                tally["offmenu"] += 1
                print(f"  OFF-MENU   {question[:40]:42} {choice}")
            elif by_ident[choice].endswith(expected):
                tally["right"] += 1
                print(f"  RIGHT      {question[:40]:42} "
                      f"{choice} -> {by_ident[choice]}")
            else:
                tally["wrong"] += 1
                print(f"  WRONG-FILE {question[:40]:42} "
                      f"{choice} -> {by_ident[choice]}")
    print(f"\n  tally over {len(BATTERY) * 3} samples: {tally}")


if __name__ == "__main__" and __import__("sys").argv[-1] == "arm-f":
    arm_f()


# --- Arm G (post-pre-flight): def-anchored pattern feasibility -------------
# Pre-flight F1 measured the mention-volume pattern at 145-3,212 lines per
# battery question (every one over a 50-line render cap). Arm G measures
# the redesign's def-site-anchored pattern: only DEFINITION lines whose
# name contains a stem match, so the volume question and the
# right-file-in-menu question are answered together. Deterministic only.


def def_anchored_pattern(question_stems: list[str]) -> str:
    alternation = "|".join(question_stems)
    return (
        rf"^\s*(def|class)\s+[A-Za-z0-9_]*({alternation})[A-Za-z0-9_]*"
        rf"|^[A-Za-z_][A-Za-z0-9_]*({alternation})[A-Za-z0-9_]* *="
    )


def arm_g() -> None:
    print("=== Arm G: def-anchored pattern — volume + menu membership ===")
    for question, _, expected in BATTERY:
        pattern = def_anchored_pattern(stems(question))
        result = subprocess.run(
            ["rg", "-n", "-i", "-g", "*.py", "--", pattern],
            capture_output=True,
            text=True,
        )
        lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
        files = sorted(
            {
                ln.split(":", 1)[0]
                for ln in lines
                if not ln.split(":", 1)[0].rsplit("/", 1)[-1].startswith("test_")
                and not ln.split(":", 1)[0].startswith("docs/")
            }
        )
        hit = any(f.endswith(expected) for f in files)
        print(
            f"  {'IN-MENU ' if hit else 'MISSING '}"
            f"lines={len(lines):4} menu-files={len(files):2}  {question[:44]}"
        )


if __name__ == "__main__" and __import__("sys").argv[-1] == "arm-g":
    arm_g()

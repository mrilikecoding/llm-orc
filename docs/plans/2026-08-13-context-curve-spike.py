#!/usr/bin/env python3
"""qwen3:8b effective-context curve spike for issue #139 (WS-6 memory,
selection-budget sizing).

WHY: the serve's deterministic selection (serving_ensemble_caller.py,
``_CTX_MAX_MESSAGES = 8`` / ``_CTX_TAIL_CAP = 4000`` / ``_CTX_SELECTED_CAP =
4000``) caps context at ~4KB (~1,000-1,300 tokens at chars/4). The MECW paper
(Context Is What You Need: The Maximum Effective Context Window, arXiv
2509.21361) reports most tested frontier models show severe accuracy
degradation by 1,000 tokens of added context, with effective windows up to
99% smaller than claimed windows — which sits inside our cap's range. But
that paper never tests qwen3:8b or any stock 8B model, is single-author, and
is not peer-reviewed (issue #139's own caveats). So this spike measures OUR
model directly instead of trusting the literature number.

PRE-REGISTERED DESIGN (fixed before any run; do not edit after seeing
results):

  - Drift threshold: ~5% accuracy drift vs. a zero-context baseline (issue
    #139's own operationalization of "meaningfully degraded", citing MECW
    arXiv 2509.21361 for the motivating phenomenon). NOTE, checked against
    the paper's own text (arXiv 2509.21361 PDF, both layout and plain
    extraction): MECW does NOT itself state a numeric "5%" threshold — its
    formal definition (Appendix A2.1) is qualitative ("the maximum token
    count, for a given problem type, before the model performance begins to
    degrade in a measurable fashion"), and its own significance test is
    p-value based, not a percentage-point cutoff. The 5% figure is issue
    #139's own operational threshold for THIS spike, not a verbatim MECW
    number — cite MECW for "severe degradation exists and is worth
    measuring", not for "5% is their cutoff". Reported here so the results
    doc doesn't misattribute the number.

  - Context levels: 500 / 1,000 / 2,000 / 4,000 / 8,000 tokens (chars/4
    estimate for target sizing — this script's char budgets are TARGET
    tokens * 4; Ollama's own ``prompt_eval_count`` is recorded per call as
    the ground-truth token count and reported alongside the nominal level).

  - Exit gate (verbatim from issue #139): "qwen3:8b within ~5% of baseline
    at 4KB means the current cap is defensible and gets a citation; severe
    degradation below 4KB means tighten the budget (and re-run the ladder to
    check score impact). Either way the selection budget stops being an
    unexamined constant."

SERVE-SHAPED CONTEXT (not a synthetic haystack): contexts are rendered in
the EXACT grammar ``serving_ensemble_caller._render_context`` /
``_indent_body`` / ``_render_write`` / ``_render_read_block`` produce, and
wrapped exactly as ``classify.py`` composes ``dispatch_input`` (lines
~1731-1734: ``f"Conversation so far:\\n{conversation}\\n\\nCurrent request:
{dispatch_input}"``):

  - Turn-history lines: ``user: <text>`` / ``assistant: <text>``.
  - A written/read file: ``assistant: [wrote <path>]\\n`` + a two-space
    indented body (the fenced block grammar from ``_indent_body``).

Filler ("haystack") content is REAL repo file bodies (read live off disk —
see FILLER_FILES below) rendered as ``[read <path>]`` blocks, interleaved
with a handful of plausible short turn exchanges — never lorem ipsum.

PROBE DESIGN: 12 needle facts (small verbatim code facts: a function's
literal return value, a config constant, a string embedded in an exception,
a filename named in a turn line, a comment-embedded fact), each with an
exact-match scoring rule (the answer must contain the literal substring,
case-insensitive). Each needle x each of the 5 levels x 3 placements
(start/middle/end of the context) = 180 calls. A zero-context baseline (the
needle fact as the ONLY context, ~150-250 chars) measures the model's
ceiling on these 12 questions, independent of context-window effects — 12
more calls, 192 total.

Model call: qwen3:8b via Ollama's ``/api/chat``, temperature 0, think off —
matching ``.llm-orc/profiles/agentic-tier-cheap-general.yaml`` (the serve's
cheap-tier seat) with ``think: false``, the override every interactive
cheap-tier seat in this repo uses (see explainer.yaml, test-writer.yaml,
adequacy-judge.yaml, and serving.yaml's own ``decide`` node). Temperature 0
is this spike's own addition (not a serve default — the serve sets no
explicit temperature) for deterministic, reproducible scoring.

Read-only against production code (only reads real repo files for filler).
No production code changes. Writes nothing under the repo — prints the full
per-call log and the final curve/verdict tables to stdout; a per-call JSONL
safety log is written under the OS temp dir (crash recovery only, not a
deliverable). The operator transcribes stdout into
docs/plans/2026-08-13-context-curve-results.md.

Run from the repo root: `uv run python
docs/plans/2026-08-13-context-curve-spike.py`
Requires a local Ollama serving qwen3:8b on localhost:11434 (free, already
pulled on the rig).
"""

from __future__ import annotations

import json
import statistics
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:8b"

LEVELS = (500, 1000, 2000, 4000, 8000)  # target context tokens
PLACEMENTS = ("start", "middle", "end")
CHARS_PER_TOKEN = 4  # target-sizing estimate; prompt_eval_count is ground truth
NUM_PREDICT = 150

# Crash-recovery log only (not a deliverable) — outside the repo, per the
# spike's write constraint (only the script + the results doc are repo
# writes).
RAW_LOG_PATH = Path(tempfile.gettempdir()) / "context-curve-raw.jsonl"

SYSTEM_PROMPT = (
    "You are a careful coding assistant. Answer the current request using "
    "only information present in the conversation above. Be concise: one "
    "short sentence, stating the literal value or name if the question asks "
    "for one. If the information is not present in the conversation above, "
    "say so plainly."
)

# Real repo files used as filler "haystack" content — serve-shaped: read
# live off disk and rendered exactly the way
# serving_ensemble_caller._render_context renders an assistant read tool
# call, never synthetic text. Diverse sizes (2.3KB-78KB) so the greedy
# fill has enough granularity at the low end.
FILLER_FILES = [
    ".llm-orc/scripts/agentic_serving/select_parts.py",
    ".llm-orc/scripts/agentic_serving/seat_contract.py",
    ".llm-orc/scripts/agentic_serving/shape.py",
    "src/llm_orc/models/ollama.py",
    ".llm-orc/scripts/agentic_serving/_helpers.py",
    ".llm-orc/scripts/agentic_serving/resolve.py",
    "src/llm_orc/models/base.py",
    ".llm-orc/scripts/agentic_serving/emit.py",
    ".llm-orc/scripts/agentic_serving/chain_plan.py",
    "src/llm_orc/web/serving/serving_ensemble_caller.py",
    ".llm-orc/scripts/agentic_serving/classify.py",
]

# A handful of plausible short turn exchanges interleaved with the filler
# file blocks, cycled — realistic session chatter, not needle-bearing.
FILLER_TURNS = (
    "user: can you also double check the retry logic doesn't loop forever\n"
    "assistant: Good catch, I'll look at the backoff calculation next.",
    "user: what's the timeout on the escalated tier?\n"
    "assistant: Let me check the profile configs before answering.",
    "user: looks reasonable, let's keep going\n"
    "assistant: Sounds good, moving on to the next file.",
    "user: any lint warnings on that last write?\n"
    "assistant: None that I can see from the diff.",
    "user: remind me what the read cap is again\n"
    "assistant: I'll check the constant rather than guess.",
    "user: ok, next file please\n"
    "assistant: On it.",
)


def _indent_body(text: str) -> str:
    """Mirrors serving_ensemble_caller._indent_body: two-space indent per
    line, blank for whitespace-only lines — the fenced block grammar the
    real serve renders context in."""
    return "\n".join(f"  {line}" if line.strip() else "" for line in text.splitlines())


def _read_block(path: str, body: str) -> str:
    return f"assistant: [read {path}]\n{_indent_body(body)}"


def _write_block(path: str, body: str) -> str:
    return f"assistant: [wrote {path}]\n{_indent_body(body)}"


def _build_filler_text() -> str:
    """Real repo file bodies as [read <path>] blocks, interleaved with
    plausible turn chatter — the filler pool every level/placement slices
    from. Built once so all needles/levels/placements at a given level see
    the SAME underlying filler content (placement is the only thing that
    varies within a level)."""
    parts: list[str] = []
    for index, rel_path in enumerate(FILLER_FILES):
        body = (REPO_ROOT / rel_path).read_text()
        parts.append(_read_block(rel_path, body))
        parts.append(FILLER_TURNS[index % len(FILLER_TURNS)])
    return "\n\n".join(parts)


FILLER_TEXT = _build_filler_text()


@dataclass(frozen=True)
class Needle:
    id: str
    request: str  # the user turn that produced the write
    path: str
    body: str  # the written file's body, containing the literal fact
    question: str
    literal: str  # exact-match scoring: literal.lower() in answer.lower()

    def block(self) -> str:
        return f"user: {self.request}\n{_write_block(self.path, self.body)}"


NEEDLES: tuple[Needle, ...] = (
    Needle(
        id="tax_rate",
        request="write a helper in tax_utils.py that returns the "
        "mid-bracket effective tax rate",
        path="tax_utils.py",
        body=(
            "def calculate_tax_rate(bracket):\n"
            '    """Effective rate for the mid bracket, per the 2026 '
            'schedule."""\n'
            "    return 0.07341"
        ),
        question="What value does calculate_tax_rate() return in tax_utils.py?",
        literal="0.07341",
    ),
    Needle(
        id="db_pool",
        request="add the connection pool ceiling to db_config.py",
        path="db_config.py",
        body=(
            "# Postgres connection pool sizing, tuned for the staging cluster\n"
            "MAX_CONNECTION_POOL_SIZE = 68217"
        ),
        question="What is the value of MAX_CONNECTION_POOL_SIZE in db_config.py?",
        literal="68217",
    ),
    Needle(
        id="retry_backoff",
        request="write the retry policy class to retry_policy.py",
        path="retry_policy.py",
        body=(
            "class RetryPolicy:\n"
            '    """Backoff schedule for transient seat failures."""\n'
            "    DEFAULT_BACKOFF_MS = 41935"
        ),
        question="What is RetryPolicy.DEFAULT_BACKOFF_MS in retry_policy.py?",
        literal="41935",
    ),
    Needle(
        id="seat_timeout",
        request="write the per-tier seat timeouts to seat_timeouts.py",
        path="seat_timeouts.py",
        body=(
            "SEAT_TIMEOUTS = {\n"
            '    "cheap": 45,\n'
            '    "escalated": 300,\n'
            '    "audit": 77602,\n'
            "}"
        ),
        question="What is the audit seat's timeout value in SEAT_TIMEOUTS "
        "(seat_timeouts.py)?",
        literal="77602",
    ),
    Needle(
        id="secret_env_var",
        request="write the deploy secret lookup to deploy_config.py",
        path="deploy_config.py",
        body=(
            "# Name of the env var holding the signing key -- never the "
            "value itself\n"
            'API_SECRET_ENV_VAR = "ORCH_SIGNING_KEY_9X4Q"'
        ),
        question="What is the name of the API secret environment variable "
        "in deploy_config.py?",
        literal="ORCH_SIGNING_KEY_9X4Q",
    ),
    Needle(
        id="checksum_error",
        request="write the checksum validator to archive_checker.py",
        path="archive_checker.py",
        body=(
            "def validate(actual, expected):\n"
            "    if actual != expected:\n"
            '        raise ValueError("checksum mismatch: expected 0xE93B7")'
        ),
        question="What checksum value appears in the ValueError message in "
        "archive_checker.py?",
        literal="0xE93B7",
    ),
    Needle(
        id="invoice_parser_file",
        request="I need a new invoice parser -- please write it to "
        "billing/invoice_parser_v9k.py",
        path="billing/invoice_parser_v9k.py",
        body=("# invoice parser v9k\ndef parse(path):\n    raise NotImplementedError"),
        question="What filename did I ask you to write the invoice parser to?",
        literal="invoice_parser_v9k.py",
    ),
    Needle(
        id="listen_port",
        request="write the bootstrap listener config to server_bootstrap.py",
        path="server_bootstrap.py",
        body=("# Local dev listener -- do not use in prod\nLISTEN_PORT = 58231"),
        question="What port does server_bootstrap.py listen on?",
        literal="58231",
    ),
    Needle(
        id="schema_version",
        request="write the migration state stamp to migrations/state.py",
        path="migrations/state.py",
        body='SCHEMA_VERSION = "9.14.203"',
        question="What is the SCHEMA_VERSION in migrations/state.py?",
        literal="9.14.203",
    ),
    Needle(
        id="jitter_seed",
        request="write the retry jitter seed to jitter_config.py",
        path="jitter_config.py",
        body=(
            "# Fixed seed keeps retry timing reproducible across test runs\n"
            "RETRY_JITTER_SEED = 30489"
        ),
        question="What is the RETRY_JITTER_SEED value in jitter_config.py?",
        literal="30489",
    ),
    Needle(
        id="receipts_file",
        request="please build the receipts normalizer and save it as "
        "receipts_normalizer_q7.py",
        path="receipts_normalizer_q7.py",
        body=(
            "# receipts normalizer q7\n"
            "def normalize(receipt):\n"
            "    raise NotImplementedError"
        ),
        question="What file did I ask you to save the receipts normalizer as?",
        literal="receipts_normalizer_q7.py",
    ),
    Needle(
        id="shard_count",
        request="write the shard routing table to shard_router.py",
        path="shard_router.py",
        body=(
            "# NOTE: shard count is pinned at 44710 for legacy compatibility\n"
            "SHARD_COUNT = 44710"
        ),
        question="According to the comment in shard_router.py, what is the "
        "shard count pinned at?",
        literal="44710",
    ),
)


def build_baseline_input(needle: Needle) -> tuple[str, int]:
    """The needle fact as the ONLY context (~150-250 chars) — the model's
    ceiling on these 12 questions, independent of context-window effects."""
    context_body = needle.block()
    dispatch_input = f"Conversation so far:\n{context_body}\n\nCurrent request: {needle.question}"
    return dispatch_input, len(dispatch_input)


def build_grid_input(needle: Needle, level_tokens: int, placement: str) -> tuple[str, int]:
    """dispatch_input at a target context level and needle placement,
    mirroring classify.py's own composition (Conversation so far:\\n
    <context>\\n\\nCurrent request: <task>). All three placements at a given
    level draw from the SAME filler slice — only where the needle sits
    within it varies."""
    needle_block = needle.block()
    target_chars = level_tokens * CHARS_PER_TOKEN
    overhead = len("Conversation so far:\n\n\nCurrent request: ") + len(needle.question)
    filler_budget = max(0, target_chars - len(needle_block) - overhead)
    filler_slice = FILLER_TEXT[:filler_budget]
    if filler_budget < len(FILLER_TEXT):
        cut = filler_slice.rfind("\n")
        if cut > 0:
            filler_slice = filler_slice[:cut]

    if placement == "start":
        context_body = f"{needle_block}\n{filler_slice}"
    elif placement == "end":
        context_body = f"{filler_slice}\n{needle_block}"
    else:
        half = len(filler_slice) // 2
        cut = filler_slice.rfind("\n", 0, half)
        first_half = filler_slice[:cut] if cut > 0 else filler_slice[:half]
        second_half = filler_slice[len(first_half) :]
        context_body = f"{first_half}\n{needle_block}\n{second_half}"

    dispatch_input = f"Conversation so far:\n{context_body}\n\nCurrent request: {needle.question}"
    return dispatch_input, len(dispatch_input)


def call_ollama(dispatch_input: str) -> tuple[str, float, int, str]:
    """(answer text, wall-clock seconds, prompt_eval_count, error). ``error``
    is "" on a clean response; a transport/HTTP failure is caught and
    recorded rather than raised, so one bad call never aborts the run."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": dispatch_input},
        ],
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_predict": NUM_PREDICT},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=data, headers={"Content-Type": "application/json"}
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=240) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        elapsed = time.perf_counter() - start
        answer = body.get("message", {}).get("content", "")
        prompt_tokens = int(body.get("prompt_eval_count", 0))
        return answer, elapsed, prompt_tokens, ""
    except Exception as exc:  # noqa: BLE001 -- spike script: record, never raise
        elapsed = time.perf_counter() - start
        return "", elapsed, 0, str(exc)


def score(answer: str, literal: str) -> bool:
    """Exact-match scoring rule: the answer must contain the literal,
    case-insensitive."""
    return literal.lower() in answer.lower()


def _print_summary(rows: list[dict]) -> None:
    baseline_rows = [r for r in rows if r["phase"] == "baseline"]
    baseline_acc = (
        sum(r["correct"] for r in baseline_rows) / len(baseline_rows)
        if baseline_rows
        else 0.0
    )
    print(f"\nBaseline accuracy (n={len(baseline_rows)}): {baseline_acc:.1%}")
    misses = [r["needle"] for r in baseline_rows if not r["correct"]]
    if misses:
        print(f"  baseline misses: {misses}")

    print("\nlevel(tok)  n   accuracy  drift-vs-baseline   start   middle    end")
    level_stats: dict[int, float] = {}
    for level in LEVELS:
        level_rows = [r for r in rows if r["phase"] == "grid" and r["level"] == level]
        acc = sum(r["correct"] for r in level_rows) / len(level_rows) if level_rows else 0.0
        level_stats[level] = acc
        drift = baseline_acc - acc
        placement_accs = []
        for placement in PLACEMENTS:
            p_rows = [r for r in level_rows if r["placement"] == placement]
            p_acc = sum(r["correct"] for r in p_rows) / len(p_rows) if p_rows else 0.0
            placement_accs.append(p_acc)
        print(
            f"{level:8d}  {len(level_rows):3d}   {acc:6.1%}   {drift:+7.1%}          "
            f"{placement_accs[0]:6.1%}  {placement_accs[1]:6.1%}  {placement_accs[2]:6.1%}"
        )

    print("\nper-needle first level with >=1/3 placement miss:")
    for needle in NEEDLES:
        first_fail = None
        failed_placements: list[str] = []
        for level in LEVELS:
            n_rows = [
                r
                for r in rows
                if r["phase"] == "grid" and r["level"] == level and r["needle"] == needle.id
            ]
            missed = [r["placement"] for r in n_rows if not r["correct"]]
            if missed:
                first_fail = level
                failed_placements = missed
                break
        label = f"L{first_fail} ({', '.join(failed_placements)})" if first_fail else "never (0-8K)"
        print(f"  {needle.id:20s} -> {label}")

    print("\nwall-clock (s) by level:  mean / median / max")
    for level in (0, *LEVELS):
        if level == 0:
            l_rows = baseline_rows
            label = "baseline"
        else:
            l_rows = [r for r in rows if r["phase"] == "grid" and r["level"] == level]
            label = f"{level:6d}"
        times = [r["elapsed_s"] for r in l_rows]
        if times:
            print(
                f"  {label}: {statistics.mean(times):6.2f} / "
                f"{statistics.median(times):6.2f} / {max(times):6.2f}"
            )

    print("\n=== exit gate (issue #139, verbatim) ===")
    print(
        '"qwen3:8b within ~5% of baseline at 4KB means the current cap is '
        "defensible and gets a citation; severe degradation below 4KB means "
        "tighten the budget (and re-run the ladder to check score impact)."
    )
    drift_4k = baseline_acc - level_stats.get(4000, 0.0)
    print(f"\nDrift at 4KB (level=4000): {drift_4k:+.1%}  (baseline {baseline_acc:.1%})")
    if abs(drift_4k) <= 0.05:
        print(
            "VERDICT: DEFENSIBLE — qwen3:8b holds within ~5% of its zero-context "
            "ceiling at the current 4KB cap. Citation: MECW (arXiv 2509.21361) "
            "motivated the check; this model does not replicate the paper's "
            "generic sub-4KB degradation at our cap."
        )
    else:
        onset = None
        for level in LEVELS:
            if abs(baseline_acc - level_stats[level]) > 0.05:
                onset = level
                break
        print(
            f"VERDICT: TIGHTEN — drift at 4KB is {drift_4k:+.1%}, past the ~5% "
            f"threshold. Degradation onset (first level with drift >5pp): "
            f"{onset if onset else 'not observed in range'}. Recommend capping "
            f"selection below the onset level and re-running the task-shape "
            "ladder to check score impact, per the issue's own instruction."
        )


def main() -> None:
    print(f"Model: {MODEL}  temperature=0  think=false  num_predict={NUM_PREDICT}")
    print(f"Filler pool: {len(FILLER_FILES)} files, {len(FILLER_TEXT)} chars")
    print(f"Levels (target tokens, chars/4 estimate): {LEVELS}")
    print(f"Raw per-call log (crash recovery only): {RAW_LOG_PATH}\n")

    rows: list[dict] = []

    def record(row: dict) -> None:
        rows.append(row)
        with RAW_LOG_PATH.open("a") as fh:
            fh.write(json.dumps(row) + "\n")

    print("=== baseline (needle fact as the ONLY context) ===")
    for needle in NEEDLES:
        dispatch_input, chars = build_baseline_input(needle)
        answer, elapsed, prompt_tokens, error = call_ollama(dispatch_input)
        correct = score(answer, needle.literal) if not error else False
        row = {
            "phase": "baseline",
            "needle": needle.id,
            "level": 0,
            "placement": "n/a",
            "chars": chars,
            "prompt_tokens": prompt_tokens,
            "elapsed_s": round(elapsed, 2),
            "correct": correct,
            "answer": answer,
            "error": error,
        }
        record(row)
        status = "ERR" if error else ("OK" if correct else "MISS")
        print(
            f"  [{status:4s}] {needle.id:20s} chars={chars:5d} tok={prompt_tokens:5d} "
            f"{elapsed:6.2f}s  ans={answer[:80]!r}"
        )

    print("\n=== grid: level x placement x needle (180 calls) ===")
    for level in LEVELS:
        for placement in PLACEMENTS:
            for needle in NEEDLES:
                dispatch_input, chars = build_grid_input(needle, level, placement)
                answer, elapsed, prompt_tokens, error = call_ollama(dispatch_input)
                correct = score(answer, needle.literal) if not error else False
                row = {
                    "phase": "grid",
                    "needle": needle.id,
                    "level": level,
                    "placement": placement,
                    "chars": chars,
                    "prompt_tokens": prompt_tokens,
                    "elapsed_s": round(elapsed, 2),
                    "correct": correct,
                    "answer": answer,
                    "error": error,
                }
                record(row)
                status = "ERR" if error else ("OK" if correct else "MISS")
                print(
                    f"  [{status:4s}] L{level:5d} {placement:6s} {needle.id:20s} "
                    f"chars={chars:5d} tok={prompt_tokens:5d} {elapsed:6.2f}s"
                )

    print("\n=== summary ===")
    _print_summary(rows)


if __name__ == "__main__":
    main()

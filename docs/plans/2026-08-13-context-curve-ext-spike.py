#!/usr/bin/env python3
"""Extension to the qwen3:8b context-curve spike (issue #139) — #145 design
grounding, per the coordinator's follow-up brief on the now-closed #139
record. Reuses NEEDLES / FILLER_TEXT / build_grid_input / score /
_write_block from the original, already-committed spike script
(2026-08-13-context-curve-spike.py) by import — that script and its
results doc entries for #139 are NOT modified; this is a sibling.

Three additions, run in this order:

1. NUM_CTX PRECONDITION CHECK (run first, reported honestly either way,
   per the coordinator's explicit instruction). The concern: qwen3:8b's
   Ollama-served context window might default to something smaller than
   the probe (commonly 4096/8192), which would mean the original #139
   grid (max 8,000 target tokens) silently truncated and its flat curve
   would be invalid. This is checked two ways, both reproducible here:
     (a) `ollama show qwen3:8b` for the model's architectural max num_ctx.
     (b) An empirical probe: a large (~20,000-token) filler-only prompt
         sent with NO num_ctx override (exactly how the original #139
         script called the API) — if the server were silently capping
         context, the returned `prompt_eval_count` would clip well short
         of the sent size.
   If truncation is detected, the script aborts before running anything
   else (a truncated-context re-run would need to happen against the
   ORIGINAL script/levels, not this extension).

2. Levels 16,000 / 32,000 tokens, same 12-needle probe, 2 placements
   (start/end — middle dropped per the coordinator's explicit call to keep
   the count down).

3. Synthesis smoke at 4K / 8K / 16K: 6 paired questions, each requiring
   BOTH of two planted facts in one answer (both-exact-match scoring) —
   probes the #145 multi-fact-synthesis caveat flagged in the original
   results doc (single-needle recall was measured; multi-fact combination
   was not).

Deviation from the original #139 script: every call here sets
``options.num_ctx`` explicitly (to the model's real architectural max,
confirmed by the precondition check) rather than relying on the server's
ambient default — belt-and-suspenders, not a correction (the precondition
check found the original run was NOT truncated).

Read-only against production code. No production code changes. Writes
nothing under the repo except (by the operator, after this script runs)
an appended section to docs/plans/2026-08-13-context-curve-results.md.

Run from the repo root: `uv run python
docs/plans/2026-08-13-context-curve-ext-spike.py`
Requires a local Ollama serving qwen3:8b on localhost:11434 (free, already
pulled on the rig).
"""

from __future__ import annotations

import importlib.util
import json
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# --- import the original #139 script as a module (not duplicated) ---------
_ORIG_PATH = Path(__file__).resolve().parent / "2026-08-13-context-curve-spike.py"
_spec = importlib.util.spec_from_file_location("context_curve_base", _ORIG_PATH)
base = importlib.util.module_from_spec(_spec)
sys.modules["context_curve_base"] = base  # register before exec: dataclass
# introspection in the base module needs sys.modules[cls.__module__] to
# resolve, or `@dataclass(frozen=True)` on Needle raises at import time.
_spec.loader.exec_module(base)

MODEL = base.MODEL
OLLAMA_URL = base.OLLAMA_URL
CHARS_PER_TOKEN = base.CHARS_PER_TOKEN

LEVELS_EXT = (16000, 32000)
PLACEMENTS_EXT = ("start", "end")  # middle dropped -- coordinator's call
SYNTHESIS_LEVELS = (4000, 8000, 16000)
NUM_PREDICT_GRID = 150
NUM_PREDICT_SYNTHESIS = 200  # two facts in one answer needs a bit more room

RAW_LOG_PATH = Path(tempfile.gettempdir()) / "context-curve-ext-raw.jsonl"


# --- 1. num_ctx precondition check -----------------------------------------


def _ollama_show_max_ctx() -> tuple[int | None, str]:
    """(model's architectural max num_ctx, error) via `ollama show`."""
    try:
        result = subprocess.run(
            ["ollama", "show", MODEL], capture_output=True, text=True, timeout=30
        )
    except Exception as exc:  # noqa: BLE001 -- precondition check: record, don't raise
        return None, str(exc)
    for line in result.stdout.splitlines():
        if "context length" in line:
            digits = "".join(ch for ch in line if ch.isdigit())
            if digits:
                return int(digits), ""
    return None, f"'context length' not found in `ollama show {MODEL}` output"


def _raw_call(payload: dict) -> tuple[dict, float, str]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=data, headers={"Content-Type": "application/json"}
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=240) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body, time.perf_counter() - start, ""
    except Exception as exc:  # noqa: BLE001
        return {}, time.perf_counter() - start, str(exc)


def check_num_ctx_precondition() -> dict:
    """Empirical + declared num_ctx precondition check. Returns a dict
    with everything needed to decide PASS/FAIL and to report honestly."""
    result: dict = {}
    model_max, show_error = _ollama_show_max_ctx()
    result["model_max_num_ctx"] = model_max
    result["show_error"] = show_error

    # Empirical probe: ~20,000-token filler-only prompt, NO num_ctx
    # override -- mirrors exactly how the original #139 script called the
    # API. If the server were silently capping context below this size,
    # prompt_eval_count would clip well short of the sent size.
    probe_target_tokens = 20000
    probe_chars = probe_target_tokens * CHARS_PER_TOKEN
    probe_text = base.FILLER_TEXT[:probe_chars]
    probe_prompt = (
        f"Conversation so far:\n{probe_text}\n\n"
        "Current request: Reply with the single word OK."
    )
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": probe_prompt}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_predict": 5},
    }
    body, elapsed, error = _raw_call(payload)
    result["probe_sent_chars"] = len(probe_prompt)
    result["probe_est_tokens"] = len(probe_prompt) / CHARS_PER_TOKEN
    result["probe_prompt_eval_count"] = body.get("prompt_eval_count")
    result["probe_elapsed_s"] = round(elapsed, 2)
    result["probe_error"] = error
    return result


def _print_precondition_report(check: dict) -> bool:
    """Prints the full precondition report; returns True (PASS, no
    truncation detected) or False (FAIL, truncation detected -- abort)."""
    print("=== num_ctx precondition check ===")
    print(f"`ollama show {MODEL}` architectural max num_ctx: "
          f"{check['model_max_num_ctx']}"
          f"{'  (lookup error: ' + check['show_error'] + ')' if check['show_error'] else ''}")
    print(f"Empirical probe: sent {check['probe_sent_chars']} chars "
          f"(~{check['probe_est_tokens']:.0f} est. tokens, chars/4), "
          f"NO num_ctx override, {check['probe_elapsed_s']}s")
    if check["probe_error"]:
        print(f"  PROBE ERROR: {check['probe_error']}")
        print("VERDICT: precondition UNVERIFIED (probe call failed) -- "
              "aborting rather than assume either way.")
        return False
    reported = check["probe_prompt_eval_count"]
    print(f"  server-reported prompt_eval_count: {reported}")
    # Fail-closed: if the server processed meaningfully fewer tokens than
    # sent (>10% short), that's truncation, regardless of what `ollama
    # show`/env vars claim.
    if reported is None:
        print("VERDICT: precondition UNVERIFIED (no prompt_eval_count in "
              "response) -- aborting rather than assume either way.")
        return False
    shortfall = 1 - (reported / check["probe_est_tokens"])
    if shortfall > 0.10:
        print(f"  shortfall vs. estimate: {shortfall:.1%}")
        print("VERDICT: TRUNCATION DETECTED. The original #139 grid (max "
              "8,000 target tokens) may be invalid if this server config "
              "was already in place then -- do not proceed with this "
              "extension; re-run the #139 script's affected levels with "
              "options.num_ctx set explicitly first.")
        return False
    print(f"  shortfall vs. estimate: {shortfall:+.1%} (within expected "
          "chars/4 estimation noise)")
    print("VERDICT: NO TRUNCATION. The server processed the full ~20K-token "
          "probe with no num_ctx override, comfortably above every level "
          "tested in the original #139 grid (max 8,000) and in this "
          "extension (max 32,000, still below the model's architectural "
          f"max of {check['model_max_num_ctx']}). The original #139 flat "
          "curve stands unmodified; every call in this extension additionally "
          "sets options.num_ctx explicitly as belt-and-suspenders.")
    return True


NUM_CTX = 40960  # qwen3:8b's architectural max per `ollama show` -- set
# explicitly on every call below (grid extension + synthesis), confirmed
# safe by the precondition check above.


def call_ollama(dispatch_input: str, num_predict: int) -> tuple[str, float, int, str]:
    """(answer, wall-clock seconds, prompt_eval_count, error) -- same shape
    as the base script's call_ollama, with options.num_ctx pinned
    explicitly (the one deviation from the original script's calls)."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": base.SYSTEM_PROMPT},
            {"role": "user", "content": dispatch_input},
        ],
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0,
            "num_predict": num_predict,
            "num_ctx": NUM_CTX,
        },
    }
    body, elapsed, error = _raw_call(payload)
    if error:
        return "", elapsed, 0, error
    answer = body.get("message", {}).get("content", "")
    prompt_tokens = int(body.get("prompt_eval_count", 0))
    return answer, elapsed, prompt_tokens, ""


# --- 3. synthesis probe: 6 pairs from the original 12 needles --------------

_BY_ID = {n.id: n for n in base.NEEDLES}

SYNTHESIS_PAIRS: tuple[tuple[str, str, str, tuple[str, str]], ...] = (
    (
        "tax_rate",
        "db_pool",
        "What does calculate_tax_rate() return in tax_utils.py, and what is "
        "the value of MAX_CONNECTION_POOL_SIZE in db_config.py?",
        ("0.07341", "68217"),
    ),
    (
        "retry_backoff",
        "seat_timeout",
        "What is RetryPolicy.DEFAULT_BACKOFF_MS in retry_policy.py, and "
        "what is the audit seat's timeout value in SEAT_TIMEOUTS "
        "(seat_timeouts.py)?",
        ("41935", "77602"),
    ),
    (
        "secret_env_var",
        "checksum_error",
        "What is the name of the API secret environment variable in "
        "deploy_config.py, and what checksum value appears in the "
        "ValueError message in archive_checker.py?",
        ("ORCH_SIGNING_KEY_9X4Q", "0xE93B7"),
    ),
    (
        "invoice_parser_file",
        "listen_port",
        "What filename did I ask you to write the invoice parser to, and "
        "what port does server_bootstrap.py listen on?",
        ("invoice_parser_v9k.py", "58231"),
    ),
    (
        "schema_version",
        "jitter_seed",
        "What is the SCHEMA_VERSION in migrations/state.py, and what is "
        "the RETRY_JITTER_SEED value in jitter_config.py?",
        ("9.14.203", "30489"),
    ),
    (
        "receipts_file",
        "shard_count",
        "What file did I ask you to save the receipts normalizer as, and "
        "according to the comment in shard_router.py, what is the shard "
        "count pinned at?",
        ("receipts_normalizer_q7.py", "44710"),
    ),
)


def build_synthesis_input(
    id_a: str, id_b: str, level_tokens: int, question: str
) -> tuple[str, int]:
    """Both needles' blocks planted at opposite ends of a filler slice
    (A near the start, B near the end) -- the hard synthesis case: the two
    facts to combine are maximally far apart in the rendered context."""
    needle_a, needle_b = _BY_ID[id_a], _BY_ID[id_b]
    block_a, block_b = needle_a.block(), needle_b.block()
    target_chars = level_tokens * CHARS_PER_TOKEN
    overhead = len("Conversation so far:\n\n\nCurrent request: ") + len(question)
    filler_budget = max(0, target_chars - len(block_a) - len(block_b) - overhead)
    filler_slice = base.FILLER_TEXT[:filler_budget]
    if filler_budget < len(base.FILLER_TEXT):
        cut = filler_slice.rfind("\n")
        if cut > 0:
            filler_slice = filler_slice[:cut]
    context_body = f"{block_a}\n{filler_slice}\n{block_b}"
    dispatch_input = f"Conversation so far:\n{context_body}\n\nCurrent request: {question}"
    return dispatch_input, len(dispatch_input)


def both_score(answer: str, literals: tuple[str, str]) -> bool:
    lowered = answer.lower()
    return all(literal.lower() in lowered for literal in literals)


# --- main --------------------------------------------------------------


def _print_summary(rows: list[dict]) -> None:
    grid_rows = [r for r in rows if r["phase"] == "grid_ext"]
    print("\n=== extension grid: 16K / 32K, start/end ===")
    print("level(tok)  n   accuracy  start   end")
    for level in LEVELS_EXT:
        l_rows = [r for r in grid_rows if r["level"] == level]
        acc = sum(r["correct"] for r in l_rows) / len(l_rows) if l_rows else 0.0
        placement_accs = []
        for placement in PLACEMENTS_EXT:
            p_rows = [r for r in l_rows if r["placement"] == placement]
            p_acc = sum(r["correct"] for r in p_rows) / len(p_rows) if p_rows else 0.0
            placement_accs.append(p_acc)
        print(
            f"{level:8d}  {len(l_rows):3d}   {acc:6.1%}   "
            f"{placement_accs[0]:6.1%}  {placement_accs[1]:6.1%}"
        )
        misses = [(r["needle"], r["placement"]) for r in l_rows if not r["correct"]]
        if misses:
            print(f"    misses: {misses}")

    synth_rows = [r for r in rows if r["phase"] == "synthesis"]
    print("\n=== synthesis smoke: 4K / 8K / 16K, both-terms-required ===")
    print("level(tok)  n   both-correct")
    for level in SYNTHESIS_LEVELS:
        l_rows = [r for r in synth_rows if r["level"] == level]
        acc = sum(r["correct"] for r in l_rows) / len(l_rows) if l_rows else 0.0
        print(f"{level:8d}  {len(l_rows):3d}   {acc:6.1%}")
        misses = [r["pair"] for r in l_rows if not r["correct"]]
        if misses:
            print(f"    misses: {misses}")

    print("\nwall-clock (s) by phase/level:  mean / median / max")
    for level in LEVELS_EXT:
        times = [r["elapsed_s"] for r in grid_rows if r["level"] == level]
        if times:
            print(
                f"  grid   {level:6d}: {statistics.mean(times):6.2f} / "
                f"{statistics.median(times):6.2f} / {max(times):6.2f}"
            )
    for level in SYNTHESIS_LEVELS:
        times = [r["elapsed_s"] for r in synth_rows if r["level"] == level]
        if times:
            print(
                f"  synth  {level:6d}: {statistics.mean(times):6.2f} / "
                f"{statistics.median(times):6.2f} / {max(times):6.2f}"
            )


def main() -> None:
    check = check_num_ctx_precondition()
    ok = _print_precondition_report(check)
    if not ok:
        print("\nABORTING extension run per the precondition check above.")
        return

    print(f"\nModel: {MODEL}  temperature=0  think=false  num_ctx={NUM_CTX} (explicit)")
    print(f"Raw per-call log (crash recovery only): {RAW_LOG_PATH}\n")

    rows: list[dict] = []

    def record(row: dict) -> None:
        rows.append(row)
        with RAW_LOG_PATH.open("a") as fh:
            fh.write(json.dumps(row) + "\n")

    print("=== extension grid: level x placement x needle (48 calls) ===")
    for level in LEVELS_EXT:
        for placement in PLACEMENTS_EXT:
            for needle in base.NEEDLES:
                dispatch_input, chars = base.build_grid_input(needle, level, placement)
                answer, elapsed, prompt_tokens, error = call_ollama(
                    dispatch_input, NUM_PREDICT_GRID
                )
                correct = base.score(answer, needle.literal) if not error else False
                row = {
                    "phase": "grid_ext",
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
                    f"  [{status:4s}] L{level:6d} {placement:6s} {needle.id:20s} "
                    f"chars={chars:6d} tok={prompt_tokens:6d} {elapsed:7.2f}s"
                )

    print("\n=== synthesis smoke: level x pair (18 calls) ===")
    for level in SYNTHESIS_LEVELS:
        for id_a, id_b, question, literals in SYNTHESIS_PAIRS:
            dispatch_input, chars = build_synthesis_input(id_a, id_b, level, question)
            answer, elapsed, prompt_tokens, error = call_ollama(
                dispatch_input, NUM_PREDICT_SYNTHESIS
            )
            correct = both_score(answer, literals) if not error else False
            pair_label = f"{id_a}+{id_b}"
            row = {
                "phase": "synthesis",
                "pair": pair_label,
                "level": level,
                "chars": chars,
                "prompt_tokens": prompt_tokens,
                "elapsed_s": round(elapsed, 2),
                "correct": correct,
                "answer": answer,
                "error": error,
                "literals": list(literals),
            }
            record(row)
            status = "ERR" if error else ("OK" if correct else "MISS")
            print(
                f"  [{status:4s}] L{level:6d} {pair_label:30s} "
                f"chars={chars:6d} tok={prompt_tokens:6d} {elapsed:7.2f}s"
            )
            if status == "MISS":
                print(f"         answer: {answer[:150]!r}")

    print("\n=== summary ===")
    _print_summary(rows)


if __name__ == "__main__":
    main()

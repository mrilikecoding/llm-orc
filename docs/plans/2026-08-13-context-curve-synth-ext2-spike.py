#!/usr/bin/env python3
"""Second synthesis extension for #145 design grounding (issue #139 closed;
the first extension in docs/plans/2026-08-13-context-curve-ext-spike.py
already ran and is recorded in the results doc). ~10-minute follow-up per
the #145 pre-flight: the synthesis probe topped out at ~14,870 actual
tokens in the first extension, 31% short of #145's flagship target
(classify.py, 19,452 measured tokens).

Adds, importing SYNTHESIS_PAIRS / build_synthesis_input / both_score /
call_ollama / NUM_CTX from the first extension script (not duplicated,
and that script's own already-recorded run is untouched by importing it —
its 66 pre-registered calls live behind ``if __name__ == "__main__"``):

1. The same 6 two-fact synthesis pairs at ~20,000 and ~24,000 target
   tokens (both-terms-required scoring).
2. 3 new three-fact questions (all three exact-match terms required) at
   the 20,000 level only — a closer proxy for module-explain synthesis,
   which needs to combine more than two facts. 3 needle blocks spread
   across start/middle/end of the filler (the 3-fact extension of the
   2-fact start/end design).

Pre-registered: same 5% drift rule as #139 / the first extension, applied
against the synthesis baseline (100% at 4K, established in the first
extension). Note the coarse resolution this ~10-minute budget accepts: at
n=6 (two-fact) one miss is 16.7% drift; at n=3 (three-fact) one miss is
33.3% drift — either already exceeds the 5% threshold on its own. Reported
honestly at this resolution rather than padded with a larger n.

Read-only against production code. No production code changes.

Run from the repo root: `uv run python
docs/plans/2026-08-13-context-curve-synth-ext2-spike.py`
"""

from __future__ import annotations

import importlib.util
import json
import statistics
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_EXT_PATH = Path(__file__).resolve().parent / "2026-08-13-context-curve-ext-spike.py"
_spec = importlib.util.spec_from_file_location("context_curve_ext", _EXT_PATH)
ext = importlib.util.module_from_spec(_spec)
sys.modules["context_curve_ext"] = ext
_spec.loader.exec_module(ext)  # defines functions/constants only -- ext's
# own main() (the precondition probe + its 66 pre-registered calls) is
# guarded by `if __name__ == "__main__"` and does NOT re-run on import.

base = ext.base  # the original #139 module, transitively available
NEW_LEVELS = (20000, 24000)
TRIPLE_LEVEL = 20000
NUM_PREDICT_3FACT = 250

RAW_LOG_PATH = Path(tempfile.gettempdir()) / "context-curve-synth-ext2-raw.jsonl"

THREE_FACT_QUESTIONS: tuple[tuple[tuple[str, str, str], str, tuple[str, str, str]], ...] = (
    (
        ("tax_rate", "retry_backoff", "listen_port"),
        "What does calculate_tax_rate() return in tax_utils.py, what is "
        "RetryPolicy.DEFAULT_BACKOFF_MS in retry_policy.py, and what port "
        "does server_bootstrap.py listen on?",
        ("0.07341", "41935", "58231"),
    ),
    (
        ("secret_env_var", "schema_version", "shard_count"),
        "What is the name of the API secret environment variable in "
        "deploy_config.py, what is the SCHEMA_VERSION in migrations/state.py, "
        "and according to the comment in shard_router.py, what is the shard "
        "count pinned at?",
        ("ORCH_SIGNING_KEY_9X4Q", "9.14.203", "44710"),
    ),
    (
        ("db_pool", "seat_timeout", "jitter_seed"),
        "What is the value of MAX_CONNECTION_POOL_SIZE in db_config.py, "
        "what is the audit seat's timeout value in SEAT_TIMEOUTS "
        "(seat_timeouts.py), and what is the RETRY_JITTER_SEED value in "
        "jitter_config.py?",
        ("68217", "77602", "30489"),
    ),
)


def build_synthesis3_input(
    ids: tuple[str, str, str], level_tokens: int, question: str
) -> tuple[str, int]:
    """3 needle blocks spread across start/middle/end of a filler slice --
    the 3-fact extension of build_synthesis_input's start/end 2-block
    design."""
    blocks = [ext._BY_ID[i].block() for i in ids]
    target_chars = level_tokens * base.CHARS_PER_TOKEN
    skeleton = (
        f"Conversation so far:\n{blocks[0]}\n\n{blocks[1]}\n\n{blocks[2]}"
        f"\n\nCurrent request: {question}"
    )
    overhead = len(skeleton)
    filler_budget = max(0, target_chars - overhead)
    half = filler_budget // 2
    gap1 = base.FILLER_TEXT[:half]
    cut = gap1.rfind("\n")
    if cut > 0:
        gap1 = gap1[:cut]
    gap2 = base.FILLER_TEXT[half : half + (filler_budget - len(gap1))]
    cut2 = gap2.rfind("\n")
    if cut2 > 0:
        gap2 = gap2[:cut2]
    context_body = f"{blocks[0]}\n{gap1}\n{blocks[1]}\n{gap2}\n{blocks[2]}"
    dispatch_input = f"Conversation so far:\n{context_body}\n\nCurrent request: {question}"
    return dispatch_input, len(dispatch_input)


def all_score(answer: str, literals: tuple[str, ...]) -> bool:
    lowered = answer.lower()
    return all(literal.lower() in lowered for literal in literals)


def main() -> None:
    print(f"Model: {base.MODEL}  temperature=0  think=false  num_ctx={ext.NUM_CTX} (explicit)")
    print(f"Raw per-call log: {RAW_LOG_PATH}\n")
    rows: list[dict] = []

    def record(row: dict) -> None:
        rows.append(row)
        with RAW_LOG_PATH.open("a") as fh:
            fh.write(json.dumps(row) + "\n")

    print("=== two-fact synthesis: 20K / 24K (12 calls) ===")
    for level in NEW_LEVELS:
        for id_a, id_b, question, literals in ext.SYNTHESIS_PAIRS:
            dispatch_input, chars = ext.build_synthesis_input(id_a, id_b, level, question)
            answer, elapsed, prompt_tokens, error = ext.call_ollama(
                dispatch_input, ext.NUM_PREDICT_SYNTHESIS
            )
            correct = ext.both_score(answer, literals) if not error else False
            pair_label = f"{id_a}+{id_b}"
            row = {
                "phase": "synthesis2",
                "pair": pair_label,
                "level": level,
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
                f"  [{status:4s}] L{level:6d} {pair_label:30s} chars={chars:6d} "
                f"tok={prompt_tokens:6d} {elapsed:7.2f}s"
            )
            if status == "MISS":
                print(f"         answer: {answer[:150]!r}")

    print("\n=== three-fact synthesis: 20K only (3 calls) ===")
    for ids, question, literals in THREE_FACT_QUESTIONS:
        dispatch_input, chars = build_synthesis3_input(ids, TRIPLE_LEVEL, question)
        answer, elapsed, prompt_tokens, error = ext.call_ollama(dispatch_input, NUM_PREDICT_3FACT)
        correct = all_score(answer, literals) if not error else False
        triple_label = "+".join(ids)
        row = {
            "phase": "synthesis3",
            "pair": triple_label,
            "level": TRIPLE_LEVEL,
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
            f"  [{status:4s}] L{TRIPLE_LEVEL:6d} {triple_label:40s} chars={chars:6d} "
            f"tok={prompt_tokens:6d} {elapsed:7.2f}s"
        )
        if status == "MISS":
            print(f"         answer: {answer[:200]!r}")

    print("\n=== summary ===")
    synth2 = [r for r in rows if r["phase"] == "synthesis2"]
    synth3 = [r for r in rows if r["phase"] == "synthesis3"]
    for level in NEW_LEVELS:
        l_rows = [r for r in synth2 if r["level"] == level]
        acc = sum(r["correct"] for r in l_rows) / len(l_rows)
        times = [r["elapsed_s"] for r in l_rows]
        print(
            f"L{level} two-fact: n={len(l_rows)} acc={acc:.1%} "
            f"mean={statistics.mean(times):.2f} median={statistics.median(times):.2f} "
            f"min={min(times):.2f} max={max(times):.2f}"
        )
    if synth3:
        acc3 = sum(r["correct"] for r in synth3) / len(synth3)
        times3 = [r["elapsed_s"] for r in synth3]
        print(
            f"L{TRIPLE_LEVEL} three-fact: n={len(synth3)} acc={acc3:.1%} "
            f"mean={statistics.mean(times3):.2f} min={min(times3):.2f} max={max(times3):.2f}"
        )


if __name__ == "__main__":
    main()

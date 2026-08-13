"""Ground-truth validation for the v2 projected-token estimator (#145
BLOCKER 1, review round 2).

The round-1 estimator's own conservativeness test validated ONLY against
the round-1 reality-check TABLE on synthetic fixtures the estimator's own
formula was tuned against — self-confirming, not independent evidence.
Round 2 caught this: measured against a REAL tokenizer on fresh fixtures,
v1 under-counted on 8 of 10 classes (base64/PEM/hex as low as 7-12% of
real). This module's job is to pin the property the estimator actually
needs — ``projected >= real`` with margin — against REAL, INDEPENDENTLY
MEASURED token counts, not against the formula's own outputs.

Measurement methodology (rig-run 2026-08-13, not runnable in CI — no
ollama there; the counts below are frozen, dated ground truth):

    ollama serve   # qwen3:8b already pulled
    curl http://localhost:11434/api/chat -d '{
      "model": "qwen3:8b", "think": false, "stream": false,
      "options": {"num_predict": 1},
      "messages": [{"role": "user", "content": "<fixture text>"}]
    }'
    # read response["prompt_eval_count"]; num_predict:1 bounds generation
    # to keep each call fast without affecting prompt_eval_count (which
    # reflects the full prompt regardless of how much gets generated).

Chat-template overhead: measured independently via a bare single-character
user message ("x") -> prompt_eval_count == 17, confirmed reproducible
across repeat probes. This matches the reviewer's own cited figure
exactly (their round-2 probes used /api/chat with think:false, the same
methodology reproduced here). REAL = raw prompt_eval_count - 17.

The ten fixture classes are the reviewer's own round-2 fixtures
(seed=1145), reproduced verbatim here for faithfulness — same
construction, so this test measures the SAME texts the review measured.
Five real repo files (including classify.py and subagent_adapter.py,
named in the review) round out the corpus with genuine, unmodified
source.
"""

from __future__ import annotations

import base64
import hashlib
import random
from pathlib import Path

from llm_orc.web.serving.serving_ensemble_caller import _READ_TOKEN_BUDGET
from llm_orc.web.serving.token_estimate import SAFETY_FACTOR, projected_tokens_v2

REPO = Path(__file__).resolve().parents[4]

# --- the reviewer's ten round-2 fixture classes (seed=1145, verbatim) ---


def _round2_fixtures() -> dict[str, str]:
    random.seed(1145)
    b64 = base64.b64encode(bytes(random.getrandbits(8) for _ in range(3000))).decode()
    hex_lines = "\n".join(
        hashlib.sha256(str(i).encode()).hexdigest() + f"  file{i}.tar.gz"
        for i in range(60)
    )
    lock = "\n".join(
        f'    "node_modules/pkg-{i}": {{"version": "1.2.{i}", "resolved": '
        f'"https://registry.npmjs.org/pkg-{i}/-/pkg-{i}-1.2.{i}.tgz", '
        f'"integrity": "sha512-'
        f"{base64.b64encode(hashlib.sha512(str(i).encode()).digest()).decode()}"
        '"},'
        for i in range(28)
    )
    minjs = (
        "function a(b,c){var d=b+c,e=d*2;return e>10?e:d}" * 40
        + ";window.__INITIAL_STATE__={u:1,v:2,w:[3,4,5]};" * 30
    )
    yaml_config = "\n".join(
        f"agent_{i}:\n  model_profile: tier-cheap\n  timeout_seconds: {300 + i}\n"
        f"  system_prompt: |\n    You review changes for correctness and clarity.\n"
        for i in range(40)
    )
    cjk_mix = "\n".join(
        f"def 处理数据_{i}(输入):\n    # 这个函数用于处理输入的数据并返回结果\n"
        f'    return {{"结果": 输入 * {i}}}\n'
        for i in range(40)
    )
    idents = "\n".join(
        "x_"
        + "".join(
            random.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(48)
        )
        + f" = {i}"
        for i in range(70)
    )
    pem = (
        "-----BEGIN CERTIFICATE-----\n"
        + "\n".join(
            base64.b64encode(bytes(random.getrandbits(8) for _ in range(48))).decode()
            for _ in range(60)
        )
        + "\n-----END CERTIFICATE-----\n"
    )
    py_src = (REPO / "src/llm_orc/core/config/config_manager.py").read_text()[:20000]
    jsonl = (
        REPO / "docs/plans/2026-08-12-arm1-runs/haiku-run2/turn-07.jsonl"
    ).read_text()[:20000]

    return {
        "base64 blob": b64,
        "sha256 digest lines": hex_lines,
        "package-lock integrity": lock,
        "minified JS": minjs,
        "YAML config": yaml_config,
        "CJK + code mixed": cjk_mix,
        "long random identifiers": idents,
        "PEM certificate": pem,
        "real python source": py_src,
        "real jsonl capture": jsonl,
    }


def _real_repo_files() -> dict[str, str]:
    files = {
        "classify.py": ".llm-orc/scripts/agentic_serving/classify.py",
        "subagent_adapter.py": "benchmarks/agentic_serving/subagent_adapter.py",
        "serving_ensemble_caller.py": (
            "src/llm_orc/web/serving/serving_ensemble_caller.py"
        ),
        "emit.py": ".llm-orc/scripts/agentic_serving/emit.py",
        "accept_gather.py": ".llm-orc/scripts/agentic_serving/accept_gather.py",
    }
    return {name: (REPO / path).read_text() for name, path in files.items()}


# --- frozen ground truth: real (overhead-corrected) prompt_eval_count,
# measured 2026-08-13 against qwen3:8b via the methodology above. These
# repo-file counts were measured against the file content AS IT STOOD on
# that date — a rewritten file changes its own char count and so its own
# real token count; this test recomputes v2 fresh against the CURRENT file
# content and compares to this frozen figure, so a small drift is
# expected and the 5% margin absorbs it. A large rewrite should
# re-measure.
_REAL_TOKENS_ROUND2_FIXTURES = {
    "base64 blob": 2959,
    "sha256 digest lines": 3792,
    "package-lock integrity": 3494,
    "minified JS": 1579,
    "YAML config": 1429,
    "CJK + code mixed": 1419,
    "long random identifiers": 2852,
    "PEM certificate": 2905,
    "real python source": 4123,
    "real jsonl capture": 7477,
}
_REAL_TOKENS_REPO_FILES = {
    "classify.py": 19885,
    "subagent_adapter.py": 6245,
    "serving_ensemble_caller.py": 14517,
    "emit.py": 2975,
    "accept_gather.py": 1792,
}

_MARGIN = 1.05  # 5% margin, per the review's "clears with >= 5% margin"


def test_v2_estimator_is_conservative_against_real_tokenizer_counts() -> None:
    """The property under test: projected >= real * 1.05 for EVERY fixture
    — not that the formula reproduces its own outputs (round 1's mistake:
    self-confirming against the table the formula was tuned against)."""
    fixtures = {
        **_round2_fixtures(),
        **{f"REPO FILE: {name}": text for name, text in _real_repo_files().items()},
    }
    real_tokens = {
        **_REAL_TOKENS_ROUND2_FIXTURES,
        **{
            f"REPO FILE: {name}": count
            for name, count in _REAL_TOKENS_REPO_FILES.items()
        },
    }

    failures = []
    for name, text in fixtures.items():
        real = real_tokens[name]
        projected = projected_tokens_v2(text)
        required = real * _MARGIN
        if projected < required:
            failures.append(
                f"{name}: projected={projected} < required={required:.0f} "
                f"(real={real}, margin={_MARGIN})"
            )
    assert not failures, "\n".join(failures)


def test_safety_factor_is_the_derived_value() -> None:
    # Pins the derivation itself (not just its effect): SAFETY_FACTOR is
    # the smallest 2-decimal value clearing the worst measured fixture
    # (PEM certificate, v2-before-factor 1926 vs real 2905) with >= 5%
    # margin. A change to either number must show up here as an
    # intentional diff, not a silent drift.
    pem_text = _round2_fixtures()["PEM certificate"]
    pem_raw = projected_tokens_v2(pem_text, safety_factor=1.0)
    pem_real = _REAL_TOKENS_ROUND2_FIXTURES["PEM certificate"]
    minimal_factor = (pem_real * _MARGIN) / pem_raw
    assert SAFETY_FACTOR >= minimal_factor
    assert SAFETY_FACTOR < minimal_factor + 0.02  # the smallest 2dp value, not padded


def test_classify_py_sanity_constraint_conflict_is_open() -> None:
    """SANITY CONSTRAINT (review round 2): classify.py must project under
    _READ_TOKEN_BUDGET at the shipped safety factor for v2 to be wireable
    without regressing the #145 exit gate (a real repo-scale file refusing
    to read is the exact failure the feature exists to fix). It does NOT
    at the factor the 5%-margin requirement derives (SAFETY_FACTOR): the
    max factor keeping classify.py under budget is ~1.574, PEM's margin
    needs >= ~1.584 — a ~1% gap. This is an OPEN DESIGN FORK for the lead
    (raise the budget, accept a smaller margin for the least-realistic
    fixture class, or refine the v2 formula), not resolved here.

    This test PINS the conflict as a known, open fact rather than hiding
    it — it fails loudly (telling you to update it) the moment either
    number changes enough to resolve or worsen the conflict, instead of
    drifting silently. Once the lead resolves it: delete this test, wire
    projected_tokens_v2 into serving_ensemble_caller._budget_read_blocks,
    and replace _READ_TOKEN_BUDGET's v1-authored value if it changed.
    """
    classify_source = _real_repo_files()["classify.py"]
    projected = projected_tokens_v2(classify_source)
    assert projected > _READ_TOKEN_BUDGET, (
        f"classify.py now projects to {projected} at SAFETY_FACTOR="
        f"{SAFETY_FACTOR}, budget is {_READ_TOKEN_BUDGET} — the conflict "
        "this test pins may now be resolved: verify, then replace this "
        "test with a real admission assertion and wire v2 into the live "
        "budget check."
    )

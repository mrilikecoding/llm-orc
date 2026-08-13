"""Spike 3: #144 serve-native discovery over SERVE-OWNED scripts only.

Question: constrained to .llm-orc/scripts/agentic_serving/*.py (~30 files,
not the 372-file repo that refuted spike 1), do deterministic signals give
exactly-one-or-refuse discovery — and once discovered, which candidates
ADMIT vs REFUSE under the real #145 read budget?

Signals measured per battery question:
  S1 filename-subset (the shipped _explain_glob_candidates rule)
  S2 bare-token content hits (spike 1's rule, rescoped)
  S3 def/name-site hits (spike 2's rule, rescoped)
  S4 token projection of the REAL rendered read block vs _READ_TOKEN_BUDGET
  S5 for over-budget whales: top-level AST segments whose name contains a
     stem — size of a deterministic section fallback, if any

Run from repo root: uv run python docs/plans/2026-08-13-serve-owned-discovery-spike.py
"""

import ast
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "src")
from llm_orc.web.serving.serving_ensemble_caller import (  # noqa: E402
    _READ_TOKEN_BUDGET,
    _render_read_block,
)

SCRIPTS = Path(".llm-orc/scripts/agentic_serving")
FILES = sorted(p for p in SCRIPTS.glob("*.py"))

# Verbatim from spike 1 / _EXPLAIN_STOPWORDS provenance.
STOP = {
    "how", "what", "where", "when", "why", "which", "who", "whose",
    "is", "are", "was", "were", "be", "been", "being", "am",
    "do", "does", "did", "done", "doing", "has", "have", "had",
    "the", "a", "an", "this", "that", "these", "those",
    "it", "its", "they", "them", "their", "we", "you", "i", "my", "your", "our",
    "of", "to", "in", "on", "for", "from", "with", "by", "at", "as",
    "into", "about", "over", "under", "between", "through",
    "and", "or", "but", "if", "then", "than", "so", "because",
    "can", "could", "should", "would", "will", "shall", "may", "might", "must",
    "get", "gets", "got", "make", "makes", "made", "use", "uses", "used",
    "using", "there", "here", "not", "no", "yes", "any", "all", "some", "each",
}

QUESTIONS = [
    "how does classify decide routing?",           # THE gate question (#144)
    "where is the recall ledger built?",
    "what does the chain executor do?",
    "how are tool calls emitted to the client?",
    "how does the accept gate verify a build?",
    "where does grounded explain refuse?",
    "what is the write history selector?",
    "how does the serve normalize read results?",
    "how does resolve pick the seat?",             # serve-owned-flavored extras
    "what does the form gate check?",
    "how does shape build the seat prompt?",
    "what does emit reject?",
]


def stems(q: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for t in re.findall(r"[a-z_][a-z0-9_]*", q.lower()):
        if len(t) >= 3 and t not in STOP and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def subset_candidates(question_stems: list[str]) -> list[Path]:
    """S1: every significant basename component ⊆ question stems."""
    out = []
    for p in FILES:
        parts = [c for c in re.split(r"[^a-z0-9]+", p.stem.lower()) if len(c) >= 3]
        if parts and all(c in question_stems for c in parts):
            out.append(p)
    return out


def content_hits(term: str) -> list[Path]:
    """S2: files mentioning the term at all."""
    r = subprocess.run(
        ["rg", "-l", "--", term, str(SCRIPTS)], capture_output=True, text=True
    )
    return [Path(ln) for ln in r.stdout.splitlines() if ln.strip()]


def def_hits(term: str) -> list[Path]:
    """S3: files DEFINING a top-level symbol containing the term."""
    out = []
    for p in FILES:
        try:
            tree = ast.parse(p.read_text())
        except SyntaxError:
            continue
        for node in tree.body:
            name = getattr(node, "name", None)
            if name and term in name.lower():
                out.append(p)
                break
    return out


def projection(p: Path) -> tuple[int, bool]:
    """S4: real rendered-block projection vs the live budget."""
    content = p.read_text()
    block, is_full = _render_read_block(str(p), f"<file>\n{content}\n</file>")
    from llm_orc.web.serving.serving_ensemble_caller import _projected_tokens

    toks = _projected_tokens(block)
    return toks, toks <= _READ_TOKEN_BUDGET


def section_fallback(p: Path, question_stems: list[str]) -> list[tuple[str, int]]:
    """S5: top-level defs/classes whose NAME contains a stem; projected size."""
    from llm_orc.web.serving.serving_ensemble_caller import _projected_tokens

    src = p.read_text()
    lines = src.splitlines()
    out = []
    for node in ast.parse(src).body:
        name = getattr(node, "name", None)
        if not name:
            continue
        if any(s in name.lower() for s in question_stems):
            seg = "\n".join(lines[node.lineno - 1 : node.end_lineno])
            out.append((name, _projected_tokens(seg)))
    return out


def main() -> None:
    print(f"scope: {len(FILES)} serve-owned scripts; budget={_READ_TOKEN_BUDGET}\n")
    for q in QUESTIONS:
        st = stems(q)
        s1 = subset_candidates(st)
        print(f"Q: {q}\n  stems: {st}")
        print(f"  S1 filename-subset: {[p.name for p in s1] or 'NONE'}")
        for term in st:
            c = content_hits(term)
            d = def_hits(term)
            print(
                f"  S2/S3 '{term}': content={len(c)} files"
                f"{' ' + str([p.name for p in c]) if len(c) <= 3 else ''}"
                f" | def-site={len(d)} {[p.name for p in d]}"
            )
        for p in s1:
            toks, ok = projection(p)
            verdict = "ADMIT" if ok else "REFUSE (over budget)"
            print(f"  S4 {p.name}: {toks} tokens -> {verdict}")
            if not ok:
                for name, seg_toks in section_fallback(p, st):
                    print(f"    S5 section '{name}': {seg_toks} tokens")
        print()


if __name__ == "__main__":
    main()

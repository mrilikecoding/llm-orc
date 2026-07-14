"""Spike: does deterministic token-extraction + specificity-ranked grep surface
the RIGHT file for real llm-orc meta-task questions — WITHOUT a curated,
gate-tuned stopword list? Discriminator = the repo's own match counts, not a
hand-list. Run from the llm-orc repo root."""

import re
import subprocess
import sys

# General English function words only — NOT code terms. No code-flavored words
# (function/file/build/run/test) so we can't tune our way to a passing gate;
# the repo's match counts must do the discrimination.
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
    "get", "gets", "got", "make", "makes", "made", "use", "uses", "used", "using",
    "there", "here", "not", "no", "yes", "any", "all", "some", "each",
}
ROOTS = ["src", ".llm-orc", "tests"]
TOO_COMMON = 25  # files; above this a term is non-distinctive -> narrow/refuse


def tokens(q: str) -> list[str]:
    seen, out = set(), []
    for t in re.findall(r"[a-z_][a-z0-9_]*", q.lower()):
        if len(t) >= 3 and t not in STOP and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def file_count(term: str) -> int:
    r = subprocess.run(
        ["rg", "-l", "--", term, *ROOTS],
        capture_output=True, text=True,
    )
    return len([ln for ln in r.stdout.splitlines() if ln.strip()])


def top_files(term: str, k: int = 3) -> list[str]:
    r = subprocess.run(["rg", "-l", "--", term, *ROOTS], capture_output=True, text=True)
    return [ln for ln in r.stdout.splitlines() if ln.strip()][:k]


QUESTIONS = [
    "how does classify decide routing?",
    "where is the recall ledger built?",
    "what does the chain executor do?",
    "how are tool calls emitted to the client?",
    "how does the accept gate verify a build?",
    "where does grounded explain refuse?",
    "what is the write history selector?",
    "how does the serve normalize read results?",
]

for q in QUESTIONS:
    ts = tokens(q)
    ranked = sorted(((t, file_count(t)) for t in ts), key=lambda x: x[1])
    print(f"\nQ: {q}")
    print(f"   tokens: {ts}")
    for t, n in ranked:
        flag = " <== too common" if n > TOO_COMMON else (" (0 hits)" if n == 0 else "")
        print(f"     {t:<14} {n:>3} files{flag}")
    distinctive = [(t, n) for t, n in ranked if 0 < n <= TOO_COMMON]
    if distinctive:
        best_t, best_n = distinctive[0]
        print(f"   -> most specific: '{best_t}' ({best_n} files) -> read-fan:")
        for f in top_files(best_t):
            print(f"        {f}")
    else:
        print("   -> NO distinctive term (all 0 or too common) -> honest refuse/narrow")

"""Spike 2: rank candidate tokens by DEFINITION/NAME signal, not bare mentions.
Still deterministic, still no curated list — search for where a symbol lives
(a file named after it, or a `def`/`class` defining it), not everywhere the
word appears. Run from the llm-orc repo root."""

import re
import subprocess

STOP = {
    "how", "what", "where", "when", "why", "which", "who", "whose",
    "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "done", "has", "have", "had",
    "the", "a", "an", "this", "that", "these", "those",
    "it", "its", "they", "them", "their", "we", "you", "i", "my", "your", "our",
    "of", "to", "in", "on", "for", "from", "with", "by", "at", "as", "into",
    "about", "over", "under", "between", "through",
    "and", "or", "but", "if", "then", "than", "so", "because",
    "can", "could", "should", "would", "will", "shall", "may", "might", "must",
    "get", "make", "made", "use", "uses", "used", "using", "there", "here",
    "not", "no", "yes", "any", "all", "some", "each",
}
ROOTS = ["src", ".llm-orc", "tests"]

ALL_FILES = subprocess.run(
    ["rg", "--files", *ROOTS], capture_output=True, text=True
).stdout.splitlines()


def tokens(q: str) -> list[str]:
    seen, out = set(), []
    for t in re.findall(r"[a-z_][a-z0-9_]*", q.lower()):
        if len(t) >= 3 and t not in STOP and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def name_hits(term: str) -> list[str]:
    return [f for f in ALL_FILES if term in f.rsplit("/", 1)[-1].lower()]


def def_hits(term: str) -> list[str]:
    # a def/class/async def whose name contains the token
    pat = rf"^\s*(?:async def|def|class)\s+[A-Za-z0-9_]*{re.escape(term)}"
    r = subprocess.run(["rg", "-lPi", pat, *ROOTS], capture_output=True, text=True)
    return [ln for ln in r.stdout.splitlines() if ln.strip()]


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
    print(f"\nQ: {q}")
    cands = []
    for t in tokens(q):
        nh, dh = name_hits(t), def_hits(t)
        score = len(nh) + len(dh)
        cands.append((t, nh, dh))
        print(f"   {t:<12} name={len(nh)} def={len(dh)}"
              + (f"  name->{[f.rsplit('/',1)[-1] for f in nh][:3]}" if nh else "")
              + (f"  def->{[f.rsplit('/',1)[-1] for f in dh][:3]}" if dh else ""))
    # candidate rule: union of name/def hits across tokens, prefer files hit by
    # the rarest signal; refuse if nothing has a name/def hit.
    hit_files: dict[str, int] = {}
    for _t, nh, dh in cands:
        for f in set(nh) | set(dh):
            hit_files[f] = hit_files.get(f, 0) + 1
    if hit_files:
        ranked = sorted(hit_files.items(), key=lambda x: -x[1])
        print("   -> read-fan candidates (by signal strength):")
        for f, s in ranked[:4]:
            print(f"        [{s}] {f}")
    else:
        print("   -> NO name/def hit for any token -> honest refuse/narrow")

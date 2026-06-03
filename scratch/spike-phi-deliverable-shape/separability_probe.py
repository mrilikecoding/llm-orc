"""Spike phi — D1/D2 separability probe ($0, no model run).

Replays the envelope-time deliverable extraction against the REAL artifact
captured from the WP-LB-G real-OpenCode run, comparing:
  (1) the current _extract_synthesizer_text (the shipped logic)
  (2) a candidate terminal-node extractor (the D1 fix)
to show whether a D1 fix alone produces usable file content (it does not — D2).
"""
import json
import re
import sys

art = json.load(open("scratch/wp-lb-c-opencode-validation/wplbg_stored_artifact.json"))


def current_extract(raw):
    """Verbatim port of _extract_synthesizer_text (orchestrator_tool_dispatch)."""
    synthesis = raw.get("synthesis")
    if isinstance(synthesis, str) and synthesis:
        return synthesis
    results = raw.get("results")
    if isinstance(results, dict) and len(results) == 1:
        only = next(iter(results.values()))
        if isinstance(only, dict):
            resp = only.get("response")
            if isinstance(resp, str) and resp:
                return resp
    return None


def terminal_node_extract(raw):
    """Candidate D1 fix: pick the LAST agent's response (terminal node of a
    linear pipeline). NOTE: raw_result carries no dependency graph, so this
    relies on insertion order — robust only for linear pipelines, not branching
    DAGs. A graph-aware fix would live executor-side where depends_on is known."""
    results = raw.get("results")
    if isinstance(results, dict) and results:
        last = list(results.values())[-1]
        if isinstance(last, dict):
            resp = last.get("response")
            if isinstance(resp, str) and resp:
                return resp
    return None


def looks_like_bare_file_content(text):
    """Heuristic: bare file content has no conversational scaffolding."""
    markers = [
        r"^Here'?s\b", r"^Here is\b", r"###\s", r"\*\*[A-Z]",
        r"Key (Points|Improvements|Changes)", r"Example Usage", r"^Notes:",
    ]
    hits = [m for m in markers if re.search(m, text, re.MULTILINE)]
    # also: does it open with a fence rather than prose?
    starts_with_fence = text.lstrip().startswith("```")
    return (not hits) and (not text.strip().startswith("Here")), hits, starts_with_fence


print("=" * 70)
print("D1 — what the CURRENT extractor produces for code-generator (3 agents):")
cur = current_extract(art)
print(f"  result: {cur!r}")
print(f"  -> falls back to json.dumps(raw_result): {cur is None}")
if cur is None:
    dumped = json.dumps(art, default=str)
    print(f"  -> stored deliverable = raw envelope, {len(dumped)} chars, "
          f"starts: {dumped[:60]!r}")

print()
print("=" * 70)
print("D1 FIX — terminal-node extractor (pick last/terminal agent):")
fixed = terminal_node_extract(art)
print(f"  picked agent: {list(art['results'].keys())[-1]!r}")
print(f"  extracted {len(fixed)} chars of the synthesizer's response")
print()
print("--- extracted content (first 400 chars) ---")
print(fixed[:400])
print("--- (end excerpt) ---")
print()
print("=" * 70)
print("D2 — is the D1-fixed content BARE FILE CONTENT a `write` can use?")
bare, hits, starts_fence = looks_like_bare_file_content(fixed)
print(f"  conversational-scaffold markers found: {hits}")
print(f"  opens with a code fence (not prose): {starts_fence}")
print(f"  -> usable as bare file content: {bare}")
print()
print("VERDICT: D1 fix yields the synthesizer's markdown, NOT a writable file.")
print("         D2 (form drift / I/O-contract) survives the D1 fix => SEPARABLE.")

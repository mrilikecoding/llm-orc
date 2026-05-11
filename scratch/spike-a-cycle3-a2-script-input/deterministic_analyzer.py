"""Deterministic README analyzer for Spike A (Cycle 3 RQ-1).

Replicates Cycle 2 A3's script-agent slot per essay 003 description:
- Link validity (HTTP 2xx/3xx for external URLs, separate localhost loopback)
- Canonical-section presence (5 canonical README sections)
- Code-block parseability (Python via ast, YAML via PyYAML, JSON via json)

The output is a structured deterministic report intended to be:
  (a) compared against A3's script findings as documented in essay 003
  (b) prepended to the A2-style code-review prompt for Spike A's three arms

NOTE: This is spike code. Will be deleted after findings are recorded.
"""

from __future__ import annotations

import ast
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# ---------- canonical sections per essay 003 ----------
# Essay 003 says A3 confirmed "all five canonical sections" but does not
# enumerate them. Selecting the five most-canonical for a CLI/library README
# (Installation, Quick Start, Configuration, Use Cases / Examples, License).
CANONICAL_SECTIONS = (
    "Installation",
    "Quick Start",
    "Configuration",
    "Use Cases",
    "License",
)


@dataclass
class LinkResult:
    url: str
    status: int | None  # None = network/parse error
    classification: str  # "external_ok", "external_error", "loopback", "anchor"
    note: str = ""


@dataclass
class CodeBlockResult:
    language: str
    line_start: int
    parseable: bool
    error: str = ""


@dataclass
class Report:
    fixture_path: str
    section_results: dict[str, bool] = field(default_factory=dict)
    h2_sections_found: list[str] = field(default_factory=list)
    link_results: list[LinkResult] = field(default_factory=list)
    code_blocks: list[CodeBlockResult] = field(default_factory=list)


# ---------- parsing helpers ----------
# Broad URL extraction — catches URLs inside nested badge/link patterns
# (e.g., [![alt](badge-url)](link-url)) where a markdown-link regex only
# catches one of the two URLs. Matches any https?:// or http://localhost:.
URL_RE = re.compile(r"https?://[^\s)\]\"\']+")
H2_RE = re.compile(r"^## +(.+?)\s*$")
FENCE_RE = re.compile(r"^```(\w*)")
ANCHOR_LINK_RE = re.compile(r"\[[^\]]+\]\(#[^)]+\)")


def extract_h2_sections(content: str) -> list[str]:
    return [m.group(1).strip() for line in content.splitlines() if (m := H2_RE.match(line))]


def extract_urls(content: str) -> list[str]:
    """Return all unique URLs in the README, in order of first appearance."""
    seen: set[str] = set()
    out: list[str] = []
    for m in URL_RE.finditer(content):
        url = m.group(0).rstrip(".,;")  # trailing punctuation
        if url not in seen:
            seen.add(url)
            out.append(url)
    return out


def count_anchor_links(content: str) -> int:
    return len(ANCHOR_LINK_RE.findall(content))


def classify_url(url: str) -> tuple[str, str]:
    """Return (classification, note)."""
    if url.startswith("#"):
        return "anchor", "in-document anchor; not network-checked"
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").lower()
    if host in {"localhost", "127.0.0.1", "::1"} or host.startswith("127."):
        return "loopback", "loopback URL; documentation example, not network-checked"
    if not parsed.scheme:
        return "anchor", "relative or unparseable; not network-checked"
    return "external", ""


def check_external_link(url: str, timeout: float = 10.0) -> tuple[int | None, str]:
    """HEAD then GET fallback. Return (status_code, note)."""
    headers = {"User-Agent": "spike-a-deterministic-analyzer/1.0"}
    for method in ("HEAD", "GET"):
        try:
            req = urllib.request.Request(url, headers=headers, method=method)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, ""
        except urllib.error.HTTPError as e:
            if method == "HEAD" and e.code in {403, 405, 501}:
                continue  # fall through to GET
            return e.code, f"HTTPError {e.code} {e.reason}"
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if method == "HEAD":
                continue  # fall through to GET
            return None, f"network error: {e}"
    return None, "all methods failed"


def extract_code_blocks(content: str) -> list[tuple[str, int, str]]:
    """Yield (language, line_start, body) for each fenced block."""
    out: list[tuple[str, int, str]] = []
    lines = content.splitlines()
    i = 0
    while i < len(lines):
        m = FENCE_RE.match(lines[i])
        if m:
            lang = m.group(1) or ""
            start = i + 1  # 1-indexed line of first body line
            body_lines: list[str] = []
            i += 1
            while i < len(lines) and not FENCE_RE.match(lines[i]):
                body_lines.append(lines[i])
                i += 1
            out.append((lang, start, "\n".join(body_lines)))
        i += 1
    return out


def parse_python(body: str) -> tuple[bool, str]:
    try:
        ast.parse(body)
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} at line {e.lineno}"


def parse_yaml(body: str) -> tuple[bool, str]:
    try:
        # bash-style heredocs and shell snippets often live in ```yaml fences;
        # only flag if the body contains YAML structural markers AND fails to parse.
        yaml.safe_load(body)
        return True, ""
    except yaml.YAMLError as e:
        return False, f"YAMLError: {e}"


def parse_json(body: str) -> tuple[bool, str]:
    try:
        json.loads(body)
        return True, ""
    except json.JSONDecodeError as e:
        return False, f"JSONDecodeError: {e}"


# ---------- main ----------
def analyze(readme_path: Path) -> Report:
    content = readme_path.read_text()
    report = Report(fixture_path=str(readme_path))

    # 1. canonical-section presence
    h2s = extract_h2_sections(content)
    report.h2_sections_found = h2s
    for canon in CANONICAL_SECTIONS:
        report.section_results[canon] = canon in h2s

    # 2. link validity (URLs from full document, plus anchor link count separately)
    for url in extract_urls(content):
        classification, note = classify_url(url)
        if classification != "external":
            report.link_results.append(
                LinkResult(url=url, status=None, classification=classification, note=note)
            )
            continue
        status, err_note = check_external_link(url)
        cls = "external_ok" if status and 200 <= status < 400 else "external_error"
        report.link_results.append(
            LinkResult(url=url, status=status, classification=cls, note=err_note)
        )
    # anchor links don't appear in URL_RE; count them separately
    n_anchors = count_anchor_links(content)
    for _ in range(n_anchors):
        report.link_results.append(
            LinkResult(url="(in-document anchor)", status=None, classification="anchor",
                       note="anchor link; not network-checked")
        )

    # 3. code-block parseability
    parsers = {"python": parse_python, "py": parse_python,
               "yaml": parse_yaml, "yml": parse_yaml,
               "json": parse_json}
    for lang, line, body in extract_code_blocks(content):
        lang_lower = lang.lower()
        if lang_lower in parsers:
            ok, err = parsers[lang_lower](body)
            report.code_blocks.append(
                CodeBlockResult(language=lang_lower, line_start=line, parseable=ok, error=err)
            )

    return report


def render(report: Report) -> str:
    """Render report as the deterministic-context block fed to Spike A's prompts."""
    lines: list[str] = []
    lines.append("# DETERMINISTIC README ANALYSIS (Spike A script-agent output)")
    lines.append("")
    lines.append(f"**Fixture:** `{report.fixture_path}`")
    lines.append("")

    # canonical sections
    lines.append("## Canonical-Section Presence")
    for canon in CANONICAL_SECTIONS:
        present = report.section_results.get(canon, False)
        marker = "PRESENT" if present else "MISSING"
        lines.append(f"- {canon}: {marker}")
    n_present = sum(report.section_results.values())
    lines.append("")
    lines.append(f"Summary: {n_present}/{len(CANONICAL_SECTIONS)} canonical sections present.")
    lines.append(f"All H2 headers found ({len(report.h2_sections_found)}): "
                 f"{', '.join(report.h2_sections_found)}")
    lines.append("")

    # links
    lines.append("## Link Validity")
    counts = {"external_ok": 0, "external_error": 0, "loopback": 0, "anchor": 0}
    for r in report.link_results:
        counts[r.classification] = counts.get(r.classification, 0) + 1
    lines.append(f"- External URLs returning 2xx/3xx: {counts.get('external_ok', 0)}")
    lines.append(f"- External URLs returning errors: {counts.get('external_error', 0)}")
    lines.append(f"- Loopback URLs (documentation examples): {counts.get('loopback', 0)}")
    lines.append(f"- Anchor / relative links (not network-checked): {counts.get('anchor', 0)}")
    if counts.get("external_error", 0) > 0:
        lines.append("")
        lines.append("**Failing external links:**")
        for r in report.link_results:
            if r.classification == "external_error":
                lines.append(f"- `{r.url}` — status {r.status} ({r.note})")
    if counts.get("loopback", 0) > 0:
        lines.append("")
        lines.append("**Loopback URLs (flagged separately as A3 did):**")
        for r in report.link_results:
            if r.classification == "loopback":
                lines.append(f"- `{r.url}`")
    lines.append("")

    # code blocks
    lines.append("## Code-Block Parseability")
    by_lang: dict[str, list[CodeBlockResult]] = {}
    for cb in report.code_blocks:
        by_lang.setdefault(cb.language, []).append(cb)
    if not by_lang:
        lines.append("No fenced code blocks with language tags python/py/yaml/yml/json.")
    for lang, blocks in by_lang.items():
        ok_count = sum(1 for b in blocks if b.parseable)
        lines.append(f"- `{lang}` blocks: {ok_count}/{len(blocks)} parseable")
        for b in blocks:
            if not b.parseable:
                lines.append(f"  - line {b.line_start}: FAILED — {b.error}")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    readme = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("README.md")
    if not readme.exists():
        print(f"FATAL: {readme} not found", file=sys.stderr)
        sys.exit(2)
    t0 = time.time()
    report = analyze(readme)
    elapsed = time.time() - t0
    out = render(report)
    print(out)
    print(f"\n---\nAnalysis duration: {elapsed:.2f}s", file=sys.stderr)

# arm-frontier-with-script-trial02

**Arm:** arm-frontier-with-script
**Trial:** 2
**Model:** sonnet-4-6 (via Claude Code Agent tool, general-purpose subagent)
**Duration (subagent reported):** 61962ms
**Total tokens (subagent reported):** 29062
**Facsimile caveats:** Subagent path with Read used twice (README.md + script-output.txt); no other tools used.

---

## Code review response (verbatim)

---

## README.md — Production Readiness Review

**Scope:** Full documentation readiness assessment
**Inputs:** README.md (888 lines), deterministic script analysis (script-output.txt)

---

### Summary Assessment

The README is structurally complete and technically accurate. The script analysis confirms: all 5 canonical sections present, 16 external links valid (0 errors), all 10 code blocks parseable (8 YAML, 2 JSON), and the single loopback URL (`http://localhost:11434/v1`) is correctly placed in a configuration example rather than as a live reference. There are no broken links or malformed code samples.

The main issues are organizational rather than factual: the document has grown to cover three distinct audiences (first-time users, CI/scripting integrators, and MCP tool users) without clear routing between them, and several sections contain redundancy that accumulates friction without adding information.

---

### Issue 1: Configuration Hierarchy Is Documented Twice, Incorrectly Each Time

**Lines 106–157 and 804–844**

The "Configuration Options" section under Quick Start (lines 106–157) introduces global vs. local config and shows the `llm-orc config init` directory layout. The standalone "Configuration Hierarchy" section (lines 804–810) then re-states the same three-level hierarchy with different framing. Neither section cross-references the other. The earlier section implies command-line options sit below local config (it only lists two levels); the later section correctly identifies three levels but doesn't explain conflict resolution — what happens when both a local profile and a global profile share the same name?

**Recommendation:** Consolidate into one canonical section. State the three levels once, show the precedence rule, and explicitly name the override behavior (local profile wins over global profile; explicit agent config wins over profile defaults — this last point is shown in YAML examples but never stated as a rule in the hierarchy section).

---

### Issue 2: Override Behavior Description Is Incomplete

**Lines 550–563**

The "Override Behavior" block states that explicit agent configuration takes precedence over model profile defaults and shows five overridable fields. The `options` key is noted as "merged with profile options (agent wins)" — but this is the only field where the behavior is merge-rather-than-replace, and it's easy to miss embedded in a comment. If a user sets `options: {num_ctx: 8192}` on an agent whose profile has `{num_ctx: 4096, top_k: 40}`, they need to know whether `top_k` is preserved.

**Recommendation:** Add a sentence explicitly distinguishing merge semantics for `options` from replace semantics for all other fields. This is a behavioral contract, not a style preference — it affects production ensemble output.

---

### Issue 3: Fan-Out Partial-Failure Behavior Is Documented But Implications Are Not

**Lines 648–660**

The fan-out result format correctly documents the `"partial"` status and notes "Partial results are preserved — the ensemble continues with whatever succeeded." However, the downstream agent behavior for partial results is not specified. If `processor[2]` fails and `synthesizer` depends on `processor`, what does `synthesizer` receive? The `response` array shows `null` for failed instances — but whether `synthesizer` is invoked at all, receives a `null` element, or receives only successful results is not stated.

**Recommendation:** Add one sentence specifying synthesizer behavior on partial fan-out: does the ensemble halt, pass nulls, or filter to successful results? This is the decision that matters to someone deciding whether to use fan-out in a production pipeline with required completeness.

---

### Issue 4: MCP Tool Count Claim Is Internally Inconsistent

**Line 299: "Tools (25 Total)"**

Counting the tools listed in the four tables: Core Execution (5) + Provider Discovery (2) + Ensemble Management (2) + Profile Management (4) + Script Management (5) + Library Operations (4) + Artifact Management (2) + Help (1) = 25. That matches. However, the `list_scripts`, `get_script`, `test_script`, `create_script`, `delete_script` entries duplicate the CLI script management commands documented at lines 248–259 without noting they are the same operations exposed through MCP. More importantly, `list_dependencies` appears in the deferred tools in the active environment (from system context) but does not appear in any of the four tool tables in the README.

**Recommendation:** Audit the tool tables against the actual MCP server implementation. If `list_dependencies` is a live tool it belongs in the table (likely under "Ensemble Management"). A stale tool count in documentation erodes trust for integration users who are counting on the interface contract.

---

### Issue 5: Library CLI vs. MCP Library Tools Have Different Access Models, Documented in Separate Sections Without Cross-Reference

**Lines 397–456 (CLI) and Lines 339–347 (MCP)**

The CLI library commands (`llm-orc library browse`, `llm-orc library copy`, etc.) and the MCP library tools (`library_browse`, `library_copy`, etc.) both require a local filesystem copy of the library. The MCP section (line 347) includes a note about this. The CLI section (lines 420–445) documents `LLM_ORC_LIBRARY_SOURCE` and `LLM_ORC_LIBRARY_PATH` environment variables but does not reference the MCP note, and the MCP section does not reference the CLI section. A user who reads only one section will not know the configuration applies to both.

**Recommendation:** Add a brief "Note: `LLM_ORC_LIBRARY_PATH` applies equally to CLI and MCP library tools" in whichever section a given user is likely to read second. Or consolidate library path configuration into its own subsection (it already exists at lines 813–843 as "Library Path Configuration") and cross-reference from both the CLI and MCP library discussions.

---

### Issue 6: Security Section Is Thin for a Credential-Storing Tool

**Line 104: "API keys are encrypted and stored securely in `~/.config/llm-orc/credentials.yaml`."**

This is the entire security explanation. For a tool that stores API keys for multiple cloud providers, users evaluating production deployment need to know: what encryption scheme is used, whether keys are decrypted in memory only during execution or cached, and what the threat model is for local key storage (relevant on shared machines or CI). The "Secure Authentication" feature bullet (line 32) and the Quick Start note are the only references.

**Recommendation:** Add a dedicated "Security" subsection — 3–5 sentences — covering the encryption mechanism (or at minimum linking to the implementation), the scope of key access, and the recommended pattern for CI (environment variables, not stored credentials). This is a production-readiness gap, not a documentation style preference. Users who skip this and later discover the implementation details will reasonably ask why they weren't disclosed upfront.

---

### Issue 7: Development Section Is Missing Local Test Coverage Instructions

**Lines 858–869**

The Development section shows `uv run pytest`, `uv run ruff check .`, and `uv run mypy src/llm_orc`. It does not show how to run tests against local models (whether Ollama must be running, whether tests mock provider calls, or whether there's a fast subset that doesn't require live providers). A contributor setting up a local development environment will hit this gap on their first `uv run pytest` run if they lack a running Ollama instance or valid API keys.

**Recommendation:** Add a note indicating which tests require live providers vs. are fully mocked, or reference `make test` (per the project's CLAUDE.md which lists it as the standard test command). At minimum, note whether `pytest` can pass in an isolated environment without credentials configured.

---

### Issue 8: Research Section Reference Is a Stale Path Risk

**Line 873: `docs/ensemble_vs_single_agent_analysis.md`**

This is a relative link to a documentation file. The script's link analysis classified relative/anchor links as "not network-checked." If this file has been moved or renamed (a reasonable risk given the active branch has significant docs reorganization in progress, per git status), users following this link from a rendered README on GitHub will hit a 404. The script correctly flagged this as outside its verification scope.

**Recommendation:** Verify the path resolves in the current branch. If the docs reorganization under `docs/agentic-serving/` has moved this file, update the path. If the file no longer exists, either restore it or replace the link with the current equivalent.

---

### Minor Issues (No Dedicated Sections Needed)

**Line 377 — MCP example has bare dict syntax:**
```
mcp__llm-orc__update_ensemble ensemble_name="security-review" changes={"agents": [...]}
```
This is not valid shell syntax — the unquoted `{...}` will be interpreted by the shell. It should be quoted or the example should clarify it's pseudocode.

**Lines 807–809 — Configuration hierarchy is inverted from conventional presentation.** Local > Global > CLI is listed lowest-to-highest priority but reads as a numbered list implying 1 = highest. Either reverse the list or label the ordering explicitly ("listed low-to-high priority").

**Line 504 — `claude-sonnet-4-20250514` model ID in the example profile.** Model IDs in documentation examples become stale and cause silent confusion when users copy them into production configs. Consider using a `claude-sonnet-4-latest` alias or adding a note to verify the current model ID.

---

### What Is Working Well

The code examples throughout are syntactically correct (confirmed by the script analysis) and substantively useful — the fan-out, input key routing, and ensemble composition sections in particular show non-trivial patterns with enough YAML to actually follow. The fan-out result format block (lines 647–660) is especially good: it shows real partial-failure output rather than just the happy path. The philosophy section (lines 879–886) is clear and direct. The MCP tool tables are more useful than narrative prose would be for an integration audience.

---

### Priority Order for Fixes

1. **Issue 4 (tool count / missing `list_dependencies`)** — Factually wrong claims about an API surface erode integration trust fastest.
2. **Issue 3 (fan-out partial failure behavior)** — Undocumented behavior on failure paths is a production reliability gap.
3. **Issue 6 (security disclosure)** — Credential storage explanation is insufficient for a multi-provider key management tool.
4. **Issue 1 (configuration hierarchy duplication)** — Housekeeping, but the inconsistency between the two sections will generate support questions.
5. **Issue 8 (stale relative link)** — Low effort to verify, high embarrassment if broken.
6. Issues 2, 5, 7, and the minor items — Real but lower urgency.

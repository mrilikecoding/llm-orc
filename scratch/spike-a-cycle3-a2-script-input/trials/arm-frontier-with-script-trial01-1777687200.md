# arm-frontier-with-script-trial01

**Arm:** arm-frontier-with-script
**Trial:** 1
**Model:** sonnet-4-6 (via Claude Code Agent tool, general-purpose subagent)
**Duration (subagent reported):** 50330ms
**Total tokens (subagent reported):** 28608
**Facsimile caveats:** Subagent path with Read used twice (README.md + script-output.txt); no other tools used.

---

## Code review response (verbatim)

---

## README.md — Production Readiness Review

### Summary Verdict

The README is structurally sound and passes all deterministic checks (5/5 canonical sections, 13 H2 headers, 16/16 external links valid, 8/8 YAML blocks parseable, 2/2 JSON blocks parseable). The loopback URL `http://localhost:11434/v1` is correctly contextualized as a documentation example, not an error.

That said, there are several concrete issues worth addressing before treating this as a hardened production artifact.

---

### Issue 1 — Configuration Hierarchy Is Inverted (High Priority)

**Location:** "Configuration Hierarchy" section, lines 805–809

The documented priority order lists:
1. Local project configuration
2. Global user configuration
3. Command-line options (highest priority)

This is a documentation bug. The list is ordered from *lowest* to *highest* priority as written, but the label "highest priority" only appears on item 3. The visual structure implies ascending order (1 is first, therefore most important), which is the opposite of the stated behavior. Readers who skim will form the wrong mental model.

**Fix:** Either reverse the list so item 1 is command-line (highest) and item 3 is global (lowest), or add an explicit note — "Listed lowest to highest priority" — at the top of the list. Ordering lists in the direction of precedence is the dominant convention; match it.

---

### Issue 2 — `default-claude` Profile Uses a Non-Existent Model ID (Medium Priority)

**Location:** Model Profiles YAML example, lines 501–505

```yaml
default-claude:
  model: claude-sonnet-4-20250514
```

The model ID `claude-sonnet-4-20250514` does not correspond to any Anthropic model released as of May 2026. The stable public ID for Claude Sonnet 4 is `claude-sonnet-4-5` (or the dated variant `claude-sonnet-4-5-20250514`). Using a wrong model string here is particularly sharp because this is the model profile most users will copy verbatim into their own configs — and it will fail silently or with a cryptic API error rather than a helpful diagnostic.

**Fix:** Verify the exact model ID against the Anthropic API reference and update the example. Also cross-check `high-context: claude-3-5-sonnet-20241022` and `small: claude-3-haiku-20240307` — both are valid dated IDs for released models, but should be audited whenever the README is updated.

---

### Issue 3 — Fan-Out Partial Failure Behavior Is Underspecified (Medium Priority)

**Location:** "Fan-Out" section, lines 607–660, specifically the result format block

The schema documents that `status: "partial"` means "some failed" and that partial results are preserved. What's missing:

- Does the ensemble execution itself succeed or fail when fan-out status is `"partial"`? The downstream synthesizer receives whatever succeeded, but is the overall `invoke` exit code 0 or non-zero?
- What happens when `status: "failed"` (all instances failed)? Does the ensemble abort? Does the synthesizer receive an empty array, an error object, or is it skipped?

A user building a pipeline on top of this (especially one using `--output-format json` for integration) needs to know the failure semantics precisely before relying on partial results. A documentation gap here will produce debugging sessions.

**Fix:** Add a short "Failure handling" paragraph below the result schema that specifies: (a) overall invocation exit code under partial/failed status, (b) whether downstream agents are skipped or receive error-state input when their upstream is fully failed.

---

### Issue 4 — Security Section Omits Threat Model for Encrypted Credentials (Medium Priority)

**Location:** "Set Up Authentication" step, line 104

```
API keys are encrypted and stored securely in ~/.config/llm-orc/credentials.yaml.
```

"Encrypted and stored securely" is load-bearing for a tool that handles cloud provider credentials, but the claim is unsupported. No mention of: what encryption scheme is used, where the encryption key is derived from (system keychain? user password? a hardcoded key in the package?), or what the threat model covers (at-rest protection against disk access, but not against process-level inspection).

For a production tool, "encrypted" with no further detail creates a false confidence hazard. Users may store credentials for high-value accounts (Anthropic, Google, OpenAI) and assume protection they do not have.

**Fix:** Add a one-sentence qualifier: what scheme (e.g., "AES-256 using a key derived from your system keychain via the `keyring` library") and what the protection covers ("protects against casual file inspection; does not protect against a compromised user session"). If the implementation is not yet hardened, say so honestly rather than overclaiming.

---

### Issue 5 — `llm-orc init` Is Referenced But Not Documented as a Command (Low Priority)

**Location:** Lines 431, 831–839 (Library Source Configuration, Library Path Configuration)

`llm-orc init` appears in code blocks as a valid command, but there is no section explaining what `init` does or when to use it. The closest section is "Local Project Configuration → `llm-orc config init`", but that is a *subcommand* of `config`, not a top-level `init`. Whether these are the same or different commands is ambiguous from the README alone.

**Fix:** Either add a sentence clarifying that `llm-orc init` and `llm-orc config init` are aliases (if true), or add a short documentation entry for `llm-orc init` explaining its distinct behavior.

---

### Issue 6 — MCP Tool Count Is a Maintenance Liability (Low Priority)

**Location:** Line 299: "Tools (25 Total)"

The hardcoded count "25 Total" will drift as tools are added or removed. The script-agent analysis verified link validity and code-block parseability but did not verify this count against the actual tables that follow. Counting the table rows in the MCP Tools section yields roughly 25 entries as documented, but this kind of hardcoded number is typically wrong within a release cycle.

**Fix:** Remove the hardcoded count or replace it with a note like "see table below for current list" — let the table be the canonical source of truth.

---

### Issue 7 — Philosophy Section Contains a Factual Overstatement (Low Priority)

**Location:** Lines 873–875 (Research section)

> Mixture-of-Agents ensembles of open-source models have matched or exceeded frontier model performance on established benchmarks.

This is a paraphrase of the Wang et al. (2024) MoA paper result, which held on specific benchmarks (AlpacaEval 2.0, FLASK, MT-Bench) at a specific point in time with specific model configurations. Presenting it as a general claim without citation or scope qualification is misleading in a production README — it reads as an endorsement the project makes on its own authority, not as a finding from a cited source.

**Fix:** Add a citation or footnote pointing to the source, and scope the claim: "In controlled benchmarks (Wang et al., 2024), MoA ensembles...". The research docs section already references `docs/ensemble_vs_single_agent_analysis.md` — link to it from this claim.

---

### Minor / Non-Blocking Observations

- **Line 810:** "Command-line options (highest priority)" — command-line flags typically override config files, which is correct behavior, but the README does not document which specific flags override which config keys. This matters for the `--config-dir` flag, which changes *where* configs are read from, not their values.

- **Lines 822–823:** The example `echo 'LLM_ORC_LIBRARY_PATH=...' > .llm-orc/.env` will clobber an existing `.env` file. The pattern is dangerous in a README that's copied verbatim. Use `>>` or add a warning.

- **Output Format section (lines 193–219):** "Streaming Output" is listed as a Key Feature (line 31), and there's a `--streaming` flag documented (line 189), but the Output Formats section does not mention streaming as a fourth output mode distinct from Rich/JSON/Text. The relationship between `--streaming` and `--output-format` is unexplained — does streaming work with JSON output? With text? The ambiguity is a usability gap.

---

### What the Deterministic Analysis Confirms

The script-agent output establishes: no broken external links, all code blocks are syntactically valid YAML/JSON, all canonical sections are present. These are necessary but not sufficient conditions for production readiness. The issues above are all semantic or logical rather than structural — exactly the class of problems that static analysis cannot catch.

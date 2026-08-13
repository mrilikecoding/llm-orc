# Serve-native dot-dir self-reference (#144) — design

**Slice:** extend explain-discovery so serve-owned scripts (the project's
own `.llm-orc/scripts/agentic_serving/*.py`) are discoverable and readable
without the client seam. OpenCode glob cannot list dot-dirs (verified
2026-07-14), and self-knowledge must not depend on client capability (a
zero-tool client still deserves self-reference — capability-map principle,
archive roadmap :571-589). Direction pinned by the issue: serve-native,
never rides the client seam.

**Evidence base:** `2026-08-13-serve-owned-discovery-findings.md` — the
shipped filename-subset rule holds at serve-owned scope (6/12 exactly-one,
0 wrong-file), five discovered files admit under the budget, classify.py
refuses at 35,044 > 35,000 (the #145 pin), and both rescue paths
(content-grep, AST sections) are refuted. So this slice grounds five real
self-referential questions and converts the literal gate question from
speculation to an honest refusal naming the true file. The "answers
grounded" exit for the classify question stays open on #144, blocked by
the whale, unblockers #106 / #151 / chunked reads.

## Mechanism

1. **Serve-owned candidate rule (classify.py, new `_self_candidates(stems)`):**
   enumerate `sorted(Path(__file__).parent.glob("*.py"))` — the operative
   script set for THIS serve instance (test fixtures copytree the scripts,
   so `__file__`-relative is correct there too). Apply the byte-identical
   filename-subset rule used by `_explain_glob_candidates` (every
   significant basename component ⊆ question stems). No new discovery
   heuristic.
2. **Union at candidate time (`_explain_discover`):** candidates =
   client-glob candidates (existing, from the rendered listing) ∪
   serve-owned candidates. Exactly-one-or-refuse over the union:
   - one, workspace → existing `needs_files` (client read, unchanged);
   - one, serve-owned → new signal `needs_self_files` (path relative to
     the project dir);
   - zero → conceptual fall-through (unchanged, deliberate);
   - two+ → refuse-with-candidates naming both halves (a workspace
     `classify.py` colliding with the serve's own refuses honestly).
   Glob semantics: glob-*failed* keeps failing closed (partial knowledge →
   refuse). Glob-*unavailable* (client never advertised it) proceeds with
   serve-owned candidates only — that IS the complete reachable surface,
   and the answer's `[read <path>]` attribution names its ground.
3. **New chain step:** `SignalBundle.needs_self_files` + guard + one Step
   row in CHAIN_EXPLAIN, placed with the existing explain need-files row
   (before `explainer`; first-match placement is the semantics). Dispatch
   shape `need-self-files.yaml` + echo script, threaded through
   resolve → shape → form_gate → emit like every seam field.
4. **Caller executes the read natively (`serving_ensemble_caller`):** on
   the self-read outcome, no ClientToolCall is emitted. The caller reads
   the file from disk (async via `asyncio.to_thread`, precedent:
   `_load_emit_reject_prefixes`, `turn_trace`), synthesizes the tool
   result message, appends it to the conversation, and re-enters the
   pipeline in-process — the client never sees a round. Rendering, the
   96KB cap, and the token budget run through the SAME
   `_render_read_block` / `_budget_read_blocks` path as client reads
   (budget parity is non-negotiable; classify.py must refuse over-budget
   here exactly as it does on the client seam).
5. **Termination:** same anti-loop property as client reads (latest read
   per path is never evicted, a visible file is never re-requested), plus
   a deterministic backstop: at most 3 self-read re-entries per turn,
   exceeded → fail-closed refusal. No model judgment anywhere.

## Invariant (rule 6) and regression instrument

**Invariant: the self-read seam can read only files inside the serve's own
project dir; any other path refuses.** Resolved-path prefix check
(`Path.resolve()`, symlinks chased) before the disk read. Regression
instrument: caller tests feeding traversal (`../`), absolute, and
symlink-escape paths through the self-read outcome, asserting refusal and
no filesystem access outside the project dir. Second invariant: **budget
parity** — instrument: classify.py through the self-read path renders the
same over-budget refusal as the client path (subprocess corpus pin).

## Tests (TDD order)

1. `test_serving_classify.py` (subprocess corpus): gate question →
   glob round → union → `needs_self_files`; over-budget refusal names
   classify.py; "how does resolve pick the seat?" → grounded from
   resolve.py; collision (workspace + serve-owned) refuses naming both;
   glob-failed refuses; glob-unavailable proceeds serve-owned-only.
2. `test_chain_plan.py`: row test + ordering test for the new Step.
3. resolve/shape/form_gate/emit pass-through tests for `needs_self_files`.
4. Caller tests: confinement invariant (traversal/absolute/symlink),
   budget parity, re-entry bound, no ClientToolCall on the wire for
   self-reads.
5. `test_serving_ensemble_endpoint.py` (hermetic HTTP): gate question end
   to end — final content grounded/refusing with zero read tool_calls for
   the serve-owned path.

## Live exit (RIG)

Real OpenCode against this checkout: "how does resolve pick the seat?"
answers grounded in resolve.py's actual content; "how does classify decide
routing?" refuses over-budget naming the real file (no speculation). Both
logged; dogfood entry if serve-shaped. Ladder rerun (13-turn battery, no
regression) + trajectory row + author-independent adversarial review with
a wrong-accept hunt (does any self-read ground the wrong file, escape the
project dir, or dodge the budget?).

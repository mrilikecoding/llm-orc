# Conformance Scan Report — Cycle 7 Loop-back #7

**Scanned against:** `docs/agentic-serving/decisions/adr-039-content-anchor.md`
**Codebase:** `src/llm_orc/agentic/`
**Date:** 2026-06-09

## Summary

- **ADRs checked:** 1 (ADR-039)
- **Conforming:** 2 (structural reuse) + 0 net-new mechanisms
- **Violations found:** 0 (the mechanism is not built — all gaps are expected-BUILD)
- **Expected-BUILD items:** 5
- **Incidental gaps:** 1

## Conformance Debt Table

| ID | Violation | Type | Location | Resolution |
|----|-----------|------|----------|------------|
| V-01 | No content-anchor injection in callee dispatch context — callee receives task + form directive only, no produced-sibling signatures | expected-BUILD | `loop_driver.py:645–652` (`_delegate_generation`) | Augment `_delegate_generation` to read produced siblings from the artifact store and append their signatures to the `input` field before dispatching |
| V-02 | No signature-extraction utility exists anywhere in the codebase (Python AST-based or otherwise) | expected-BUILD | entire `src/llm_orc/` — confirmed by grep for `import ast`, `FunctionDef`, `ast.parse`, `extract.*signature` | Build `src/llm_orc/agentic/sibling_signature_extractor.py` (or equivalent) as a net-new component that reads a produced file, parses with `ast`, and returns public function/class signatures + brief docstrings |
| V-03 | `LoopDriver` has no reference to `SessionArtifactStore` — it cannot read produced deliverables at callee-dispatch time | expected-BUILD | `loop_driver.py:330–352` (`LoopDriver.__init__`) | Inject `SessionArtifactStore` (or a narrow protocol) into `LoopDriver.__init__` alongside the existing `SessionActionRecord`; or provide a produced-deliverable listing seam via the action record |
| V-04 | `SessionActionRecord` records `(action_kind, target_path, result)` only — does not record the `ArtifactReference` or the artifact path required to source the produced file for signature extraction | expected-BUILD | `session_action_record.py:33–44` (`ActionRecord` dataclass); `loop_driver.py:477–479` (`record_action` call) | Extend `ActionRecord` with an optional `artifact_reference: ArtifactReference | None` field (the ADR-037 extensible meta-record seam this module explicitly documents), and populate it from the `ApplyWork` envelope on the generation path |
| V-05 | No FC for anchor presence, anchor sourcing, or signatures form is checkable from the current dispatch-context inspection surface — the `TurnDecision` event carries `delegated_ensemble` and `action` but not the dispatch-context payload | expected-BUILD | `loop_driver.py:222–280` (`TurnDecision` dataclass) | Add a `content_anchor_present: bool` field to `TurnDecision` (or a separate event) so the FC "refutable from dispatch-context inspection" can be verified instrumentally without log archaeology |

## Structural Reuse — What ADR-039 Can Lean On

These items conform or provide clean reuse surfaces the BUILD work inherits without modification:

**R-01 — `SessionArtifactStore.read_deliverable` is production-ready.**
`session_artifact_store.py:264–291`. The read-side accessor ADR-039 names explicitly already exists and is wired into `ArtifactBridge`. It is UTF-8-faithful, raises `ArtifactNotFoundError` on missing artifacts, and its path resolution (`_resolve_disk_path`) is the structural inverse of `write_deliverable`. No changes needed to source produced-file content for signature extraction.

**R-02 — The injection seam is clean and correctly identified.**
`loop_driver.py:619–653` (`_delegate_generation`). The method already composes the `input` string as `f"{task}\n\n{directive}"` before passing it to `InternalToolCall`. The ADR-039 anchor appends after the directive in the same composition step — no new code path, no new method boundary required. The seam is a single string concatenation in one private method.

## Notes

**V-03 is the structural prerequisite for V-01.** `LoopDriver` needs a way to enumerate produced siblings for the current session. Two composition paths are viable: (a) inject `SessionArtifactStore` into `LoopDriver` directly and look up produced paths by iterating the session directory, or (b) extend `SessionActionRecord` (V-04) to carry `ArtifactReference` on generation records so `LoopDriver` can recover the artifact path without touching the filesystem layout directly. Path (b) is lower-coupling and consistent with the existing `SessionActionRecord`-as-seam doctrine (`session_action_record.py:1–16`), which explicitly notes the write-log schema is "extensible" with a false-stop trigger for enrichment. The BUILD design amendment should choose a path and record it.

**V-04 records the underlying information gap.** The `ActionRecord` already carries `target_path` (the client-facing file path, e.g. `converters.py`) but not the artifact store path (`agentic-sessions/<session_id>/<dispatch_id>/...`). The artifact store path is what `read_deliverable` needs. The join between a client file path and its artifact store reference is currently not recorded anywhere accessible to `LoopDriver` at dispatch time.

**V-02 is net-new BUILD with no existing structural precedent in the codebase.** Grep over the full `src/llm_orc/` tree found zero uses of `import ast`, `ast.parse`, `FunctionDef`, or any signature-extraction pattern. The extractor is a genuinely new component. ADR-039 notes it is language-specific (Python AST for `.py` files; full-content fallback for other types) and that it is correctness-critical (a wrong anchor resolves 0/10, below the unanchored baseline). The BUILD scenario set should include an extractor unit test with a real Python source fixture to confirm the correct-names property before integration.

**V-05 is optional but strengthens the discharge gate.** ADR-039's FC "refutable from dispatch-context inspection" is stated as a scenario-level criterion; an observable field on `TurnDecision` makes it testable in the existing event-substrate harness without reading the raw dispatch payload. This is a secondary BUILD concern, not a blocker.

**Selection-policy scope note.** ADR-039 §Decision leaves the multi-sibling selection policy (all produced siblings versus a dependency-inferred subset) as a BUILD detail. Spike ξ validated injecting the single relevant sibling. For the discharge-gate re-run (5-file temperature library), a minimal policy — inject all prior produced siblings for the session — is the simplest conforming implementation and avoids a dependency-inference mechanism that is unmeasured. The BUILD design amendment should commit a concrete policy and note it as a scoped BUILD decision, not an ADR amendment.

**`ArtifactBridge` is not involved.** The ADR injects signatures into the dispatch *input* (what the callee ensemble receives as its task), not into the marshalled output. `ArtifactBridge` (`artifact_bridge.py`) operates on the output side and is not a target for this work item.

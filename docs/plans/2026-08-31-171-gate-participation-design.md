# #171 — the deliverable must participate in its own acceptance

Lead design brief, 2026-08-31 — DRAFT, awaiting reviewer pre-flight
before implementation (this changes gate semantics; the fork below is
real). Issue #171 has the measured evidence: two of three target shapes
ship, and content is unconstrained (`x = 1`, wrong function, `# TODO`
all accept; on `re-fix` the write clobbers real client content).

## The mechanism, grounded

- `accept_gather.py:184` (and `refix_gather.py:149`) derive
  `target_file` from `_FILE_RE` over the REQUIREMENT TEXT; when the
  requirement names no .py file, `target_file=""` and nothing is
  shadowed.
- `accept_executor._materialize` writes workspace files, then the
  target-file shadow (only when non-empty), then the deliverable to
  `solution.py` and tests to `tests.py`. Tests satisfied by workspace
  modules pass against ANY deliverable.
- `adequacy_check.py` reads the TESTS only (value-bearing assert rule)
  and never sees the code, so `tests_adequate` is structurally
  independent of what shipped.
- Meanwhile emit's write destination is the ROUTING DECISION's `file`
  key (default `solution.py`), which the gate chain never sees. The
  sandbox's shadow target and the client's write destination are two
  classifiers that can disagree — the #177 shape at the gate seam.

## Invariant (proposed)

`accept: true` is a claim about the shipped deliverable: the tests that
passed must have been able to observe it, at the destination the client
will receive. Concretely: (a) in the sandbox, every import path by which
the tests can reach the deliverable's destination module resolves to the
DELIVERABLE, and (b) the executed tests reference at least one name the
deliverable defines.

## The fork

1. **(A) Unconditional shadow at the resolved destination.** Plumb the
   routing decision's `file` into the gather nodes; shadow it always,
   `_FILE_RE` demoted to fallback. Deterministic; closes the
   shadow-miss shapes (`target_file=""`, requirement naming a different
   file). Does NOT close the participation gap: tests that never touch
   the deliverable still pass against the workspace.
2. **(B) Participation check.** Extend the adequacy seam to see the
   candidate: refuse when the tests reference NO name the candidate
   defines (via the deliverable's module or `solution`). Closes the
   workspace-satisfied and junk-content shapes statically, and #169's
   smoke-test gap at the same seam. Alone, it is spoofable at runtime:
   a referenced NAME can bind to a stale workspace module when the
   shadow misses — so B without A certifies the wrong bytes.
3. **Escalation on signal (#119)** — not reached for; 1 and 2 are not
   ruled out.

**Recommendation: A + B together.** A makes the runtime bind the tests
to the deliverable's bytes; B makes vacuous suites refuse. Each patches
the other's blind side; both are deterministic (doctrine 9). Runtime
does most of B's work once A holds (a junk deliverable breaks a genuine
reference with AttributeError), so B stays minimal: name-overlap only,
no quality judgment (the issue's own boundary against #110 creep).

## Over-refusal exposure (the pre-flight question)

Doctrine 7: the degenerate optimum is refusing everything. Before
implementation, B's predicate must be run over the recorded live corpus
(`.llm-orc/.serve-trace/turns.jsonl` and the dogfood records — the
issue counts 52 recorded writes) and the wrong-reject rate reported.
Known suspect shapes for the pre-flight to weigh:

- Tests exercising the deliverable ONLY via `solution` import while the
  turn resolves a different destination (A shadows both — measure).
- Mutation-pattern tests (`add_todo(todos, "x")`) whose only referenced
  names are workspace-defined containers.
- Held rounds (#100: the carry's sentinel block IS the spec) — whose
  names do held tests reference?
- `_inject_workspace_imports` preludes adding imports the tests did not
  write — do injected names count as references? (They must not.)
- re-fix: the candidate is a FIX of an existing module and defines the
  same names — overlap passes; confirm no shape where the fix defines
  ONLY new helper names the visible test does not call.

## Seams

`accept_gather.py` / `refix_gather.py` (destination plumbing + gather
the decision's `file`), `accept_executor._materialize` (unconditional
shadow), `adequacy_check.py` (candidate input + name-overlap rule; the
value-bearing rule unchanged), refusal vocabulary in the envelope
(honest reason: the tests never exercise the deliverable; path-free per
#168).

## Instruments (from the issue, plus the fork's own)

1. A workspace-satisfied suite does not accept a deliverable the tests
   never touched (all three measured target shapes). Red today.
2. A healthy build whose tests DO exercise the deliverable still
   accepts.
3. End to end through the real serving chain (#155's lesson).
4. The re-fix counterpart: `# TODO: implement the fix` never clobbers
   client content.
5. Sandbox-destination agreement: the file the tests ran against is the
   file emit writes (the two-classifier pin).
6. The corpus replay: wrong-reject rate over recorded real turns, with
   the named suspect shapes as fixtures.

## Out of scope

Artifact QUALITY judgments (#110), widening #166/#169's emptiness
guards, #119 escalation.

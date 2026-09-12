# #171 — the deliverable must participate in its own acceptance

Lead design brief v2, 2026-08-31, rewritten after the measured
pre-flight REDESIGN verdict. v1 (in git) proposed unconditional
destination shadowing (A) plus a static tests-reference-the-candidate
rule (B). The pre-flight drove both through the real
`accept_gather → accept_executor → adequacy_check → accept_gate` chain:
A+B still shipped 8/8 constructed non-participation shapes, B is
structurally absent on two of the four gated routes, and A moved 0 of
32 recorded live build turns while reversing #98 on write-tests. Both
are dropped. Issue #171 has the original evidence.

## Invariant (unchanged from v1)

`accept: true` is a claim about the shipped deliverable: the tests that
passed must have been able to observe it. A gate whose ground truth
never touched the deliverable must refuse.

## Why no static rule can work (measured)

`accept_executor_runner.run_tests` execs code and tests in ONE shared
namespace (`exec(code, ns)` then `exec(tests, ns)`); any import in the
tests rebinds the deliverable's names afterwards, and every workspace
module is importable. `import inventory; inventory.add_item(...)`,
`import inventory as inv`, `from inventory import *`, and facade
re-exports are ordinary dialects no name rule distinguishes from a
deliverable reference — binding is a runtime fact. Worse, the
self-inflicted case: `accept_gather._inject_workspace_imports` PREPENDS
`from inventory import add_item` when the tests reference `add_item`
bare, actively diverting a reference that would have hit the
deliverable (WA-2b; the most common turn shape, destination
`solution.py`).

## The mechanism: runtime ablation control in `accept_executor`

When the suite passes and a NON-EMPTY deliverable exists, run one
control child: same workspace, same repaired tests, the deliverable's
bytes ABSENT from every destination they were written to (`solution.py`
and the shadow target both). If the control still passes, the
deliverable never participated — `accept` is False with the honest,
path-free reason (the tests never exercise the deliverable). A control
that fails or crashes proves necessity; proceed to accept.

Why this seam: `accept_executor.py` is the only node on all four gated
shapes (`build-gated-round`, `build-code-round` held rounds, `re-fix`,
`write-tests-round`), so held rounds and re-fix are covered without
touching the ensemble graph — the two routes the adequacy seam
structurally cannot see (`build-code-round.yaml` and `re-fix.yaml`
declare no judge agent; `accept_gate._resolve_adequacy` carries round
1's verdict on held rounds).

Conditions and bounds:

- **Skip when the deliverable is empty** (write-tests turns carry
  `code: ""` from `tests_gather` — an ablation there is vacuous and
  would refuse every write-tests turn; emptiness is already #166/#169's
  refusal).
- **Run the control only on the would-accept path** (tests already
  failing need no ablation).
- **Budget:** the control is one extra child inside `_run_children`'s
  aggregate wall budget — account for it there, per-child timeout
  applies. Measured cost: median 0.19s against a 0.38s non-model chain.
- **Named bound (pin as documented behavior):** the ablation proves the
  bytes were NECESSARY, not that behavior was observed — a suite
  asserting `len(open("helpers.py").read()) > 0` passes the control
  with a junk deliverable. The value-bearing adequacy rule is the other
  half of the invariant and already exists; neither subsumes the other.
- **Named bound (review F4, control granularity ≠ real-run
  granularity):** the control always runs as ONE combined process; the
  real suite runs per-test isolated (a fresh subprocess, fresh
  workspace copy, per test). A workspace module with cross-test state
  can pass every test in isolation yet fail when the SAME tests run
  together in the control's one process, for a reason unrelated to the
  deliverable — the control then reads "necessary" and non-
  participating junk ships. Measured live rate: 0/46 recorded turns hit
  this shape; matching granularity would cost one extra per-test-
  isolated subprocess set on every ablation run for a divergence not
  yet observed live, so this is recorded rather than closed.
- **Named bound (review round 2, F-3, module-LOAD granularity ≠
  assertion granularity):** participation is decided at module-LOAD
  granularity — any top-level `from solution import X` in the tests
  makes the control fail on ImportError, "proving" necessity, even when
  every ASSERTION in the tests actually routes through a workspace
  module instead. Repro: `code = "def discount(price, pct):\n    return
  0\n"`, `tests = "from solution import discount\nimport pricing\ndef
  test_d():\n    assert pricing.discount(200, 10) == 180\n"`, workspace
  `pricing.py` holding the correct implementation → `accept: true` on
  both base and this branch, with a wrong `discount` that the one real
  assertion never touches. Pinned as documented behavior
  (`test_named_bound_load_granularity_ships_wrong_code_via_a_workspace_module`
  in `test_serving_gate_participation.py`), not fixed — matching
  granularity would mean tracing which names each assertion actually
  reaches at runtime, not just which names load, a materially bigger
  mechanism than this control.

## Two companion slices (independent, same arc)

1. **re-fix's smoke test becomes surface-derived.** `refix_select`'s
   `_SMOKE_TEST` is `def ...(): pass` — it references no name, so on
   the smoke-only path participation is unsatisfiable and today
   `x = 1` / `import os` / an unrelated def still accept and clobber
   (post-#173). Derive the smoke test from the PRIOR module's top-level
   names (`refix_gather` already holds `prior_code` and `target_file`):
   a candidate that drops the surface fails it. Measured: real fix
   passes, all five junk shapes fail. Recorded bound: a fix that
   intentionally drops a public name refuses.

   **Round 2 review, F-2:** the surface was every top-level def/class
   name, underscore-prefixed ones included, so inlining or deleting a
   private helper refused a legitimate fix for dropping a name nothing
   public ever promised. Fixed: the surface is PUBLIC names only (a
   leading underscore, dunders included, is excluded).

   **Round 2 review, F-1, real scope of the surface-less fallback:**
   when the prior module has no public top-level def/class at all
   (constants-only settings, a dict-only rates table, an `__init__`
   re-export module), the smoke-only bar degrades to "loads cleanly"
   alone — and the ablation control's empty-code run satisfies that
   identically, so `participates` was False for EVERY candidate against
   this whole class and the route could never converge. Fixed by
   falling back to the PRE-#171 bar (loads cleanly, plus #173's
   inertness whitelist) for exactly this case — the participation gate
   is bypassed only when `smoke_surface_empty` is set, never for a
   prior that has a real surface to check. Measured scope of the
   fallback: a constants-only one-value fix accepts, a dict-only rate
   change accepts, and `x = 1`-style junk against a surface-less prior
   ALSO accepts now (unchanged from pre-#171: `Assign` isn't on #173's
   inert whitelist, and this fix does not widen that whitelist) — the
   fallback is a return to the pre-#171 bar, not a new, stricter one.
2. **`_inject_workspace_imports` must not divert.** Skip injection for
   any name the CANDIDATE defines (both call sites — tests and code).
   Two-line guard; removes the self-inflicted diversion class (WA-2/2b).

## Explicitly not doing

- A (unconditional destination shadow): 0/32 measured effect on build
  turns (the two `_FILE_RE` copies are byte-identical over the same
  text; the turn payload carries no `file` key), and write-tests sets
  `target_file=""` on purpose (#98) — A there is a regression.
- B (static reference rule): defeated by import dialects; absent on
  held/re-fix routes; sees `code: ""` on every write-tests turn.
- #119 escalation: not reached for.
- Artifact QUALITY (#110), widening #166/#169.

## Instruments

1. The issue's table closes: all three target shapes and all four
   junk-content shapes refuse via the ablation (red today), with the
   already-refused cells unchanged.
2. Healthy builds still accept: the pre-flight's OK-1..7 shapes, plus
   the corpus expectation — 0 wrong-rejects held over the 71
   recorded/labelled contracts (13 build/re-fix replays, 42 write-tests
   replays, 16 judge_adequacy fixtures).

   **Correction (review round 1):** the pre-flight's own "0
   wrong-rejects over 71" figure was itself measured with the
   destination's stale copy WITHHELD at the control (skip the shadow)
   rather than emptied — production shape includes it. Under that
   control, 14/55 recorded pairs flip accept→refuse (two are genuine
   live additive-edit turns whose suite covers only the unchanged half
   of the deliverable, F3 below). The briefed control — shadow the
   target file with EMPTY bytes too, never skip it — gives 13/13 on the
   build/re-fix replay subset with the stale copy correctly present.
3. The import-dialect wrong-accept set (`import X` + attribute, aliased
   import, star import, facade re-export, injected-import diversion)
   all refuse — these killed the static rule and are the fixture set
   that proves the ablation is doing the work.
4. Held-round participation pin and re-fix participation pin — the two
   routes the adequacy seam cannot see each carry their own end-to-end
   pin (#155's lesson).
5. The re-fix counterpart: `# TODO: implement the fix` never clobbers
   client content; the surface-derived smoke test fails every junk
   shape and passes the real fix.
6. write-tests turns are untouched: empty-candidate skip pinned.
7. End to end through the real serving chain, and the ablation's
   refusal reason is path-free (#168 discipline).

Regression instruments: full `make test` (511 instruments + doc-drift),
`tests/unit/serving/` accept/adequacy/refix suites, #166 #169 #173
pins, #98's write-tests shadowing pin.

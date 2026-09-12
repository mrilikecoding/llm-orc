# #182 — the existing-repo shape, slices D and B-1

Lead design brief v1, 2026-09-11. Evidence: `docs/plans/2026-09-11-daily-driver-probe/`
(seven live turns through real OpenCode on a seeded package; 1/7, the one
being the greenfield control). Issue #182 names four mechanisms and five
slices. This brief covers the two smallest, in the order the harm ranks
them; the rest wait on what these two teach.

## Invariant (from #182)

A build turn in a non-empty workspace ships a deliverable that is
gate-verified in the workspace's own layout, at the destination the ask
names. A turn that cannot see enough of the workspace to do that asks or
refuses; it never mints a sibling module.

## Slice D — discovery before an unnamed-file build (turn 6's harm)

Today: `classify._discovery` runs a glob round only when `wants_existing`
(tests-primary, or a build verb plus an "existing" verb like fix/update)
AND the ask has a module stem in one of three exact phrasings. An ask
like "add priority support to the todo app: TodoStore.add takes ..., the
CLI add command accepts --priority ..." has a build verb, no named file,
no stem phrasing, so `named_file == ""`, `file == "solution.py"`, and the
serve builds a parallel implementation beside the package. The gate
accepts it (it is right about its own tests) and the client writes it.
That is a shipped deliverable that does not advance the ask — the
task-level wrong-accept the doctrine's 2x2 exists to count.

Mechanism (deterministic, classify only):

1. An unnamed-file build turn (`has_build_signal`, `_extract_file(task)`
   empty, not `is_explain`, not a run/recap route) with NO `[globbed ...]`
   block visible in the rendered context requests ONE glob round
   (`**/*.py`, the existing `needs_glob` seam, same 50-path cap).
2. With the listing in hand (the existing MATCH step, extended):
   - listing empty, or every path is `test_*`-named → greenfield as today
     (`solution.py`). The 13-turn ladder never reaches this branch: all
     thirteen prompts name a file, so it is unchanged by construction.
   - exactly one non-test `.py` whose basename-stem or whose
     `_basename_components` intersect the ask's identifiers (CamelCase
     names like `TodoStore`, snake identifiers, the stem phrasings) → that
     path becomes the turn's named file; the existing read seam takes over.
   - otherwise (several candidates, or none matching) → the honest
     refusal the truncated/multi-match branches already word: name the
     files, and say which files the listing holds (capped at the first N
     non-test paths). Nothing is minted.
3. A truncated listing keeps #148's wording.

Bound (record, do not close): a workspace whose only `.py` files are
tests still mints `solution.py`; an ask whose identifiers match no file
refuses rather than guesses — a daily driver will hit this on "add a
helper for X" asks in a package with no obvious home, and the refusal
names the listing so the user can name the file. #123's multi-file
shape is not attempted here.

Instruments:

1. The probe's turn-6 ask against the probe's seed listing → glob
   requested (red today: `solution.py` minted with no glob).
2. Same ask with the listing rendered → refusal naming `todo/cli.py` and
   `todo/storage.py`, `file == ""`, `build == false`; NO `solution.py`
   destination anywhere in the decision (the harm pin: mutant restores
   the greenfield fallback → red).
3. "add a remove method to TodoStore" with the same listing → unique
   identifier match → `named_file == "todo/storage.py"` and the read
   request fires (the existing seam).
4. Empty listing → `solution.py` as today; the ladder's thirteen prompts
   → routing byte-identical to main (a table pin over all thirteen).
5. Live: turn 6 of the probe re-driven through real OpenCode on the
   merged serve — zero `write` tool_use, the refusal on the wire, no file
   on disk; then the same ask naming the two files ships (or refuses
   honestly under #123's bound — record which).

Implementation notes (measured, 2026-09-12):

- `needs_glob` carries the stem `"py"`, not the literal `**/*.py"` this
  brief names — the caller's glob-pattern template (`_glob_pattern`,
  `serving_ensemble_caller.py`) only accepts a single identifier-shaped
  stem (or a comma-joined brace form), never a bare extension. Every
  non-test `.py` basename's own extension trivially contains the substring
  `"py"`, so `**/*py*` (the template any module-stem turn already issues)
  discovers the whole `.py` surface through the SAME single-stem seam,
  with no second pattern shape added to the caller.
- The MATCH step's "intersect the ask's identifiers" turned out to need
  more than exact component equality: `TodoStore` → `{todo, store}` does
  not literally intersect `storage.py`'s `{storage}`. Shipped a narrow,
  documented stand-in (`_shares_root`): components equal outright, or both
  ≥ 4 chars and sharing a 4-char prefix — enough to relate `store`/
  `storage` without a real stemmer. Below that floor only exact equality
  counts (`cli` must be named exactly).
- Refusal wording is match-count-shaped, not "always the whole listing":
  zero matches names the full non-test `.py` listing (nothing more
  specific to point at); two or more matches names only the matched
  candidates (more actionable, and keeps a non-matching file like
  `todo/__init__.py` out of the refusal).
- The mechanism turned out broader than "an existing-verb build with no
  stem match": turn 6's own ask uses "Add", not a fix/update-style verb,
  so the fallback fires for ANY unnamed-file build once the old
  stem/`wants_existing` path yields nothing — including plain "write/add/
  create" asks with empty context. That changed three existing classify
  unit tests' first-round routing (`test_build_turn_routes_to_the_code_generation_seat`,
  the renamed `test_fresh_create_module_turn_globs_once_then_falls_back_to_code_seat`,
  `test_normal_decisions_carry_empty_glob_fields`); each got a
  same-behavior round-trip variant plus (where the old test's own name
  asserted the retired invariant) a new test naming the intentional
  change.
- Fixed a latent gap the new fallback exposed rather than introduced:
  `_discovery` re-derived `_extract_file(task)` instead of accepting the
  turn's already-known `named_file` (e.g. `turn["file"]` set directly,
  never appearing in the task text) — a named-via-field turn could have
  reached the new branch and globbed for a file already known. `_discovery`
  now takes `named_file` and skips discovery whenever it is set, from
  whichever source.
- `_files_to_request`'s `wants_existing` gate now also treats a glob MATCH
  as existing (`bool(glob_file)`), a no-op for the old stem path (it was
  already true there) — without it the matched file's read request never
  fired for a build with no fix/update verb, contradicting "the existing
  read seam takes over".
- The `file` field's blanket `named_file or "solution.py"` default is left
  in place for a PENDING glob round (needed elsewhere —
  `test_conceptual_explain_never_gates_despite_the_solution_py_default`
  pins it for bare-symbol explain discovery) but is suppressed specifically
  when `glob_failed` is set: a refusal must never carry a `solution.py`
  destination even inertly, which is what instrument 2's harm pin actually
  checks.
- `docs/serving.md`'s classify row was not touched — its one-line summary
  ("deterministic where the signal is structural … emits `needs_decider`
  when not") already omits discovery mechanics for the pre-existing
  module-stem case, so this slice adds no new drift there.

## Slice B-1 — an edit never drops the prior module's surface (turn 2)

Today: on a build whose target the turn read (`[read todo/storage.py]`
in context), `code-generator`'s prompt asks for "the code change
directly"; on a 40-line class the coder emitted the one method and the
gate ran that fragment as the whole module. The ladder's turn 2 ("add a
complete_todo function to todo.py") passes only because the prior module
is five lines and the coder rewrites it whole.

Mechanism: reuse #171's surface check (`refix_select._smoke_test`'s
public top-level names, post-rework) as a deterministic input on the
build-gated route when the target file's prior body is visible in
context: a deliverable that drops any public top-level name the prior
module defined is refused with the honest reason ("the deliverable no
longer defines X, Y; an edit must ship the whole file") and that reason
enters the held round, where the existing convergent-fix loop lets the
coder re-emit the whole file. The prompt gets one sentence for the edit
case ("when a file's current content is shown, output the complete
updated file"); the guard is what makes the turn honest when the prompt
is ignored (doctrine 2: structure, not a third rule).

Bound: a deliberate removal of a public name refuses (same as #171's
re-fix bound); private names are free.

Instruments: the probe's turn-2 seat output (the fragment) against the
seed's prior body → refused with the surface reason (red today: accepted
into the gate as a module and refused for the wrong reason); a whole-file
edit that keeps every prior name and adds `remove` → accepts; the ladder
turn-2 shape → unchanged; live: probe turn 2 re-driven → `write
todo/storage.py` whose content keeps `add`, `list`, `complete` and adds
`remove`, and the seed's own `tests/test_storage.py` still passes.

Sequencing: B-1 lands after the #171 rework merges (it reuses the surface
helper; do not fork it).

## Explicitly not doing here

Path fidelity in the gate sandbox (slice A), code+tests per turn (#123 /
slice C), the command registry (#124 / slice E), and running the
workspace's own suite inside the gate (B-2). Each waits on D and B-1's
live rows.

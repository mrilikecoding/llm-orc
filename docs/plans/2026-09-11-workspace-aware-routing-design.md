# Workspace-aware routing — arc 1 of the existing-repo shape

Lead design brief v1, 2026-09-11 (evening). Subsumes #185, the residual
of #182 slice D, and the routing half of #123. Evidence: live row 10 in
`docs/plans/2026-09-11-182-d-live-rows/` (a fresh-session "add a method to
the class in todo/storage.py" wrote the file without reading it) and the
probe `docs/plans/2026-09-11-daily-driver-probe/` (turns 1-3, 6, 9).

## Invariant

Whether a build's destination already exists is a fact about the client
workspace, decided from the workspace listing, never from the verb the
ask used. A build whose destination exists reads it before building; a
build whose destination does not exist builds fresh; a build with no
destination matches exactly or asks. `classify` holds the only routing
decision, so this lands in classify alone.

## Mechanism (deterministic, classify only)

1. **One listing per session.** A build turn (named or unnamed
   destination) with no `[globbed ...]` block visible in the rendered
   context requests ONE glob round through the seam slice D added (stem
   `py`, pattern `**/*py*`, 50-path cap). A listing from an earlier turn
   in the rendered window counts; a listing is requested at most once per
   turn (D's termination pin stays).
2. **Existence from the listing.** For a NAMED destination: present in
   the listing (path equality after normalising the client's absolute
   prefix, or basename equality when the ask names a bare basename and
   exactly one listed path has it) → the existing read seam fires
   (`needs_files`), exactly as a fix-verb ask does today; absent →
   greenfield build as today. `_EXISTING_RE` stops deciding existence;
   keep it only where it selects the fix CHAIN (`_FIX_VERB_RE` is
   separate and untouched).
3. **Unnamed destination**: D's exact-match rule, unchanged.
4. **Truncated listing** (#148): a named destination that is not in the
   truncated listing is UNKNOWN, not absent — request the read anyway; a
   failed read then falls through to greenfield (see 5). Never claim
   absence from a cut listing.
5. **A failed read of a named destination is not a refusal.** Today a
   read failure refuses ("could not read X: client read failed", the
   turn-9 shape). Under this arc, when the READ was requested because of
   the listing and the client reports the file absent, the build proceeds
   greenfield; when the file was in the listing and the read failed for
   any other reason (cap, permissions), refuse as today. The
   ladder's turn 9 ("write tests for existing phantom.py") is tests-
   primary and must keep refusing: `phantom.py` is not in the listing and
   a tests-for-existing ask has nothing to build fresh — pin it.

## Bounds (record, do not close)

- A bare-basename ask matching two listed paths (`lib/storage.py`,
  `todo/storage.py`) refuses naming both (honest ask), as D does.
- The listing is `.py`-only; a non-Python destination (`README.md`,
  `config.yaml`) keeps today's verb-based behaviour — record it.
- `#123` code+tests per turn is NOT here; the tests-primary routing keeps
  D's and today's behaviour. Arc 2 carries the destination paths.

## Instruments (each red first; drive the real classify script as the
existing tests do; every pin goes red under the mutant that restores the
verb rule — apply, run, revert, state it in the commit body)

1. Row 10's ask, empty context → `needs_glob` set, no build this round.
2. Same ask with the seed listing → `needs_files == ["todo/storage.py"]`,
   `build` false this round, no `write`; mutant: restore `_EXISTING_RE`
   as the existence signal → red.
3. Same ask with a listing lacking `todo/storage.py` → greenfield build
   to `todo/storage.py` as today.
4. "write a function that adds two numbers in add.py" with a listing
   holding `add.py` → read requested (existence, not verb); with a
   listing lacking it → build as today.
5. Ladder identity: run all thirteen prompts with EMPTY context through
   main's classify and this branch's; the FIRST-round decision may now
   differ only by `needs_glob` on the build prompts (state exactly which
   prompts gain a glob round and pin the table); after a rendered EMPTY
   listing every decision is byte-identical to main.
6. Turn 9's shape (`write tests for existing phantom.py`, listing without
   it) refuses as today.
7. Truncated listing + named destination → read requested, never
   "absent".
8. Reason hygiene: no server path, username, or pattern internals in
   any refusal; client listing paths may appear as the client gave them.

## Live rows (the exit gate, after merge, real OpenCode on the probe seed)

- Row 10's ask in a fresh session → `glob`, `read todo/storage.py`, then a
  `write todo/storage.py` whose content keeps `_load`, `_save`, `add`,
  `list`, `complete` and adds `remove`; the seed's `tests/test_storage.py`
  still passes (B-1's guard now has its prior).
- Turn 7 of the probe (greenfield `add.py`) → one extra glob round, then
  the same write.
- The 13-turn ladder: 13 exits 0, deliverables unchanged (rerun once).

## Required self-tests in the implementer's report (agreed 2026-09-11)

Mutant evidence for every new pin; a consumer trace for any new field
classify emits (emit, the caller's render, the trace); a fresh-input
wrong-match hunt on the path-normalisation step (absolute client paths,
`./` prefixes, Windows separators are out of scope but must not crash).
One independent review round follows; a report without the three
self-tests is returned unread.

## Implementation notes (measured, 2026-09-11)

- A fix/update-verb-led ask (`_EXISTING_RE` fires) is UNTOUCHED by this
  arc: it still requests the read directly, with no glob round, exactly
  as before. The brief's "exactly as a fix-verb ask does today" describes
  the OUTCOME the new existence signal reaches for a non-fix-verb ask,
  not a claim that fix-verb asks now also glob first — an existing pin
  (`test_fix_turn_without_a_write_takes_the_read_seam_not_the_chain`,
  empty context, target `need-files` in one round) would have gone red
  under the other reading, and the ladder's turns 7 and 13 (both
  fix-verb-led) confirm it live: neither gains a glob round.
- Ladder-identity table (instrument 5, empty context, independent per
  prompt): turns 1 ("write a todo item... in todo.py"), 2 ("add a
  complete_todo function to todo.py"), and 6 ("create storage.py...")
  gain a first-round `needs_glob` — each names a `.py` file with a build
  verb and no fix verb. All ten others are unaffected: turn 3 is an
  explain (discovery never fires under `is_explain`), turns 4/8/9 are
  tests-primary (out of scope, #123 carries it), turns 5/10 are
  memory/recall, turns 7/13 are fix-verb-led (see above), turn 11 is a
  run, and turn 12 already discovers via its own module-stem phrasing
  (unchanged by this arc). After a rendered empty listing, all three
  reconverge to byte-identical with main's single round.
- The read-reports-absent distinction (invariant 5) needed a new helper:
  `_attempt_reason` (the existing read-failure reader) collapses every
  "(failed)" read variant to the fixed string "client read failed",
  discarding the client's actual wording — until this arc no caller
  needed to tell "file not found" apart from any other read failure (a
  fix-verb ask already knew the file existed, so any failed read refused).
  A listing-driven read can legitimately come back absent, so a new
  `_read_reports_absent` reads the raw trailing text on the
  `[read ... (failed)]` line directly and checks for the client's own
  "File not found" prefix (`_render_read_block`'s exact wording) —
  everything else on that line still refuses as before.
- A NAMED destination's matched value is never rewritten to the client's
  absolute listing path, even when the ask gave only a bare basename that
  resolves via basename equality: the ask's own naming is preserved
  verbatim (`glob_file = named_file`, unchanged) so the destination write
  path stays what the user asked for — `todo/storage.py`, not
  `/Users/.../todo/storage.py` — matching the live row 10 exit-gate
  wording ("a `write todo/storage.py`"), not the client's own absolute
  form. This differs from the UNNAMED-file build seam (slice D), which
  has nothing else to name the destination with and so rewrites to the
  discovered listing path.

## Implementation notes (measured, 2026-09-12, independent review round)

- Review finding 1 (BLOCKED, fixed): `_named_destination_matches`'s suffix
  match requires the listing path to be AT LEAST as deep as the ask's own
  naming — a listing SHALLOWER than the ask (bare basenames while the ask
  names a directory-qualified path, e.g. a workspace-root/glob-anchor
  mismatch) found zero matches and reported the destination ABSENT,
  reopening row 10's blind-overwrite harm through a path-depth gap rather
  than the verb gap. Fixed with a basename-equality fallback in
  `_named_build_discovery` when the suffix match is empty and the ask
  names a directory: exactly one basename match is existence UNKNOWN (not
  confirmed), so it takes the same path as a truncated listing — request
  the read of the ask's OWN path, never rewritten to the listing's
  shallower one; two or more basename matches refuse naming them, same as
  the existing bare-basename ambiguity bound.

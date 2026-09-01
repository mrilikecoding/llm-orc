# #177 — one file-vs-inline predicate, three consumers

Lead design brief, 2026-08-31. Issue #177 carries the evidence (the
resolver calls a bare CWD-file name content while ScriptAgent executes
the file; #163's cache identity trusted the resolver and reproduced the
#160 stale serve; the fail-closed patch in `agent_runner.py` covers the
symptom). Direction (1) from the issue, endorsed there: widen the
file-vs-inline classification to agree with EXECUTION, and make every
consumer ask one call. (Landed as `resolve_and_classify`, which
subsumed and then replaced the `is_inline_content` predicate this brief
originally named; the syntactic clause alone lives on as the private
`_has_path_syntax`.) Environment: ANY.

## Invariant

Exactly one predicate decides file-vs-inline for a `script:` reference,
and it describes what actually executes. No consumer stats the
filesystem to re-derive the answer. Today's execution behaviour is
preserved for every reference shape that executes today.

## The predicate

A reference is a FILE when it has path syntax (a separator or a
`SCRIPT_EXTENSIONS` suffix — today's rule) OR when, bare, it names an
existing file relative to the process CWD. Otherwise it is inline
content. The widening makes the predicate filesystem-dependent by one
stat on bare names; that is already the price `agent_runner.py` pays for
the fail-closed patch, so nothing gets slower.

## Execution-preservation constraints (the traps)

1. **CWD wins over library paths for bare names.** Today a bare name
   that names a CWD file is returned verbatim by the resolver (classified
   inline) and ScriptAgent executes the CWD file. If `_resolve_uncached`
   were simply flipped to search library paths for these, a name present
   in BOTH places would silently switch which script runs. So: a bare
   name naming an existing CWD file resolves to itself (the relative
   path), BEFORE any search-path logic. Pin this with a both-places
   probe.
2. **A bare name naming nothing stays inline** — resolver returns it
   verbatim, agent executes it as content. Unchanged.
3. **Slash-carrying references keep today's semantics** (search paths,
   `ScriptNotFoundError` when absent). Inline content that contains a
   separator already misclassifies as a path today; that is a
   pre-existing bound, out of scope here — do not widen or narrow it.
4. **The vanished-file race fails closed.** Today ScriptAgent's
   `os.path.exists` falls back to `_execute_inline_script` when a
   resolved PATH is gone at execution time — handing a path to `bash -c`,
   which executes whatever the name resolves to on PATH. Closing that
   door alone is not enough: a bare, slashless name handed to the FILE
   branch's own interpreter (`bash <name>`, or any execvp-style call) is
   subject to the SAME PATH search once the name is not found in CWD —
   round 2 review measured a same-named PATH impostor running and
   reporting success after the file vanished post-classification. So the
   resolved value for a bare CWD file is anchored (`./name`), never
   verbatim: a vanished file then fails LOUDLY in either branch, because
   the anchored name carries a separator and cannot be re-resolved via
   PATH. Pin it on the bare shape — an absolute or already slash-carrying
   reference never had this hole, since `_has_path_syntax` makes it
   unflippable regardless of the file's later state.

## Seams

- `resolver.py` — widen the classification (`is_inline_content` and
  `_resolve_uncached` per trap 1; later unified into `resolve_and_classify`,
  with the syntactic clause split out to the private `_has_path_syntax`);
  rewrite the docstring paragraphs that describe the disagreement (they
  name #177 as open — the doc-drift check in `make test` reads these
  files, so stale prose is a red build, rule 19).
- `script_agent.py` — the three `os.path.exists(resolved_script)` sites
  (`_run_script`, `execute_with_schema_json`, the interactive site)
  consume the predicate's answer instead of statting. Thread the
  classification from resolve time; a small structural prep commit
  (`refactor:`) before the behavioural one is the expected shape.
- `agent_runner.py` — the #163 fail-closed disagreement patch comes OUT;
  the identity for a bare-name CWD file becomes the digest of the file's
  bytes. Its removal must not reopen the stale serve: re-run #177's own
  measured repro (edit the file between runs; the identity must change)
  as a pinned test. Update the long #163 comment block — it documents
  the disagreement as live.

## Instruments

1. A bare-name reference naming a CWD file classifies FILE in the
   resolver and in ScriptAgent — one predicate, no disagreement. Red
   today.
2. It is cacheable under a digest of the file's bytes; the edited-file
   stale-serve probe from the issue goes green (run 2 serves the edit).
3. Inline content naming nothing still executes as content and stays
   cacheable (over-refusal direction).
4. A `.ts`/`.mjs` file in the project root still executes as a file.
5. Both-places bare name: the CWD file still wins (trap 1).
6. Vanished-file race: a file-classified reference whose file vanishes
   fails loudly in EITHER branch — no inline fallback, and the FILE
   branch's own interpreter cannot re-resolve an anchored bare name via
   PATH (trap 4, revised by round 2 review after a same-named PATH
   impostor ran and reported success).

Regression instruments: full `make test` (511 measurement instruments
included; the doc-drift check will bite on stale comments),
`tests/unit/` resolver + script_agent + agent_runner cache-identity
suites, and #163's cache pins.

## Out of scope

`SCRIPT_EXTENSIONS` membership, slash-carrying inline content (bound,
trap 3), #161 cache purity, #162 import misses.

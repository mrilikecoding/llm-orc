# #160 — the cache key identifies the script's BYTES (design)

Status: pre-flight. Issue: #160, found during the #159 review.

## Mechanism

`ScriptCache._generate_cache_key` (cache.py:65-84) hashes what it calls
`script_content`, and `agent_runner.py:117-119` passes
`agent_config.script`. That field is a REFERENCE, and in every ensemble
in this repo it is a path (`script: scripts/agentic_serving/resolve.py`).
So the key identifies the file's name, never its contents.

Consequence on a fresh install, where `ScriptCacheConfig` ships
`enabled=True, ttl_seconds=3600`: edit a script, re-run, and get the
PRE-EDIT result for up to an hour. With `persist_to_artifacts` the entry
lives under `./.llm-orc/cache` and survives a restart, so the stale
result crosses processes.

This is strictly worse than the cached-failure replay fixed in #159, and
#159's predicate cannot touch it: a stale SUCCESS is cached correctly
under any failure-skipping rule.

The field is genuinely dual-purpose, which is why the bug is easy to
miss. `resolve_script_path` returns inline content verbatim for a
non-path reference, so `script: "echo hello"` legitimately IS its own
content and hashing it is correct. Only the path case is wrong.

## Change

Identify a script by its resolved path AND the sha256 of its bytes when
the reference resolves to a file; keep hashing the reference itself when
it is inline content.

Both halves of the pair matter. Bytes alone would collide two different
scripts with identical contents, which is usually harmless but not when
a script reads its own `__file__` or a sibling relative path. Path alone
is the bug.

An unresolvable reference falls back to the reference string rather than
raising: cache-key computation must not be the thing that reports a
missing script, and execution a moment later produces the proper error.

The not-a-file test uses `os.path.isfile`. This paragraph has been
wrong twice in opposite directions, so it is now measured. The guard
buys nothing for inline content longer than `PATH_MAX`: `read_bytes`
raises `OSError` and the `except` already catches it. It buys three
things the `except` cannot. Inline content carrying a NUL byte raises
`ValueError`, not `OSError`, so it escapes and kills the run. A FIFO
resolves fine and is not a regular file, and `read_bytes` blocks on
open until a writer appears, so without the guard computing a cache key
HANGS the agent. A character device such as `/dev/zero` reads unbounded.
The last two are why this is `isfile` and not `exists`.

Two consequences worth naming rather than discovering. Hyphen and
underscore forms of a primitive reference (`primitives/file-ops/...` and
`primitives/file_ops/...`, both live in the corpus) resolve to one file
and therefore collapse to ONE key, where today they are two. And a local
script shadowing a library one now gets a distinct key, where today the
pre-shadow result is served.

### Also: interactive agents stop being cached, at BOTH get and set

The same review found that the human's ANSWER is not part of the key
(`agent_runner.py:125-128` keys on `input_data` and `parameters` only),
so a second identical interactive agent in the same executor replays the
first person's answer without prompting. Caching an agent whose entire
purpose is to ask a human is wrong at any key granularity, so those skip
the cache entirely. `_requires_user_input` already exists on the runner.

Skipping at the `get` alone is not enough AFTER this change, and the new
key is the reason: several references can now resolve to one file and
one key, while `_requires_user_input` still judges each reference
separately. An interactive agent could write an entry that a
non-interactive alias then hits, replaying a human's answer to a
different agent. Under `persist_to_artifacts` it would also write that
answer to a JSON file on disk for the TTL.

The predicate is coarse in both directions (inline content containing
`input(` matches; a custom `scripts/ask.py` calling `input()` does not),
but it cannot be wrong here in a way the ROUTING is not already wrong,
because `agent_runner.py:234,249` use the same predicate to decide
whether an agent prompts at all.

## The default: shipping `enabled: False` after all

An earlier draft of this design kept the default on, arguing the flip
was a mitigation for an unsound key and that fixing the key removed the
need. **Pre-flight falsified that premise by demonstration**, and the
decision reverses.

Fixing the key does not make the cache sound, because it assumes every
script is a pure function of (bytes, input, parameters). Two of the six
registered primitives are not, and they are the two most-used in this
repo's ensembles. Measured through the real runner with shipped
defaults:

- `primitives/file-ops/write_file.py` (10 ensemble uses): run, delete
  the output file, run again with identical parameters. The second run
  reports `{"success": true, ...}` from cache and **the file does not
  exist**. A cache hit ELIDES A WRITE and reports success.
- `primitives/file-ops/read_file.py` (16 ensemble uses): run, change the
  file it reads, run again. Returns the old content — this issue's own
  staleness one level out, on an UNDECLARED input that no cache key can
  reach.

Both survive this fix untouched. So the honest cost/benefit is: the
benefit is small (the cache only pays for sequential repeats inside one
executor, which ensembles rarely have, and #158's lack of in-flight
dedup means it misses the concurrent siblings where it would pay), and
the default cost is demonstrably "a write silently does not happen".

Shipping a default that can silently elide a write is not defensible
while waiting for a purity mechanism, so `ScriptCacheConfig.enabled`
becomes `False`. This repo already sets it false locally, so nothing
here changes. #161 carries the purity requirement and the opt-in
mechanism that would justify turning it back on.

**Round 2: the first attempt at that flip was inert, and review caught
it.** `EnsembleExecutor._load_script_cache_config` restated every
default rather than reading them off the dataclass, and its copy said
`cache_config.get("enabled", True)`. So flipping the dataclass changed
nothing for any project without an explicit opt-out — which is every
fresh install, since `templates/global-config.yaml` writes no
`script_cache` block and `load_performance_config` has no such key in
its defaults. A fresh install still elided writes. The pin was no help
because it asserted the dataclass field, the one the runtime never
read; and the inertness is invisible from inside this repo precisely
because `.llm-orc/config.yaml` sets the key explicitly.

The fix removes the duplicated defaults: `_load_script_cache_config`
now reads each fallback off `ScriptCacheConfig()`. The lesson is
narrower than "add a pin" — a restated default is a second source of
truth, and this one had already drifted.

The key fix lands regardless: it is correct on its own terms, and it is
a precondition for any future re-enable.

## Invariant

Two executions share a cache entry only if they ran the same script
BYTES with the same input and parameters.

Read the converse too, because the invariant is easy to over-hear:
sharing a key does NOT mean the cached result is still valid. A script's
undeclared inputs and side effects sit outside the key by construction
(#161). This makes the key honest; it does not make a hit safe.

And read "the same script BYTES" narrowly: it means the ENTRY FILE's
bytes. Review demonstrated the gap that hides in the plural. Sixteen
scripts in `.llm-orc/scripts/agentic_serving/` import a sibling
`_helpers` module; editing that sibling leaves the entry file's digest
unchanged, so the pre-edit result is served for the TTL. That is #160's
own bug, one file over, and this change does not reach it. Closing it
means hashing the import closure, which is a different piece of
machinery than a digest, so it is recorded here rather than attempted.

## Where the identity is computed

In the RUNNER, not in `ScriptCache`, keeping the cache a generic
key/value store — the same separation the #159 review argued for when it
kept the failure predicate out of `set`.

`script_content` stays the RAW reference for
`_validate_primitive_output` (`agent_runner.py:134`), which needs it:
`_normalize_script_ref` returns `None` for an identity string, so
rebinding the local would silently disable the primitive schema check on
every cache hit, with no pin catching it. Only the two cache calls take
the identity.

That touches one existing test, `test_the_entry_carries_no_lying_success_field`,
which reaches into the cache with a raw path and must use the identity
instead.

## Regression instruments

1. **Editing a script invalidates its entry**: run, rewrite the file
   with different output, run again, and the second result reflects the
   edit. Red today — this is the issue's reproduction.
2. **An unchanged script still hits**, so the fix does not degrade into
   "never cache". Assert on `get_stats()`.
3. **Inline content still caches** (`script: "echo hello"`), which is
   the case the current code gets right and a naive "always hash the
   file" would break.
4. **Two scripts with identical bytes at different paths do not share an
   entry**, pinning the path half of the pair.
5. **An unresolvable script reference does not raise from the cache
   path**, pinning the fallback.
6. **An interactive agent is never cached**, asserting `sets == 0` and
   `hits == 0`. Needs a real file whose name contains `get_user_input.py`
   (the predicate reads the reference, and `_execute_interactive`
   requires the script to exist); `input()` raises `EOFError` under
   pytest and is handled, so do not try to observe a prompt.
7. **A PROJECT-RELATIVE reference is invalidated by an edit.** The
   critical pin, and the one absent from the first draft: if the
   key-time resolver is built without threading `project_dir`, a
   project-relative reference fails to resolve, the identity falls back
   to the reference string, and this whole change ships INERT — while
   pins 1-6 all stay green, because the natural harness uses an absolute
   `tmp_path` reference that resolves either way. Project-relative is
   the shape every shipped ensemble uses.
8. **The cross-process case**: `persist_to_artifacts=True`, run, edit
   the script, then read through a FRESH `ScriptCache`. This is the
   worst case the issue names, and the only pin proving the key is a
   pure function of bytes rather than of process state.

9. **Inline content with a NUL byte does not raise.** Not in the draft;
   added because mutation verification showed removing the
   `os.path.isfile` guard killed nothing. That is the guard's only
   unique job, per the measurement above.

### Round 2, added after review

10. **An interactive agent never reads a NON-interactive alias's entry.**
    Review found the guard at the `get` was unpinned: with only the
    `set` guarded nothing is ever written, so instrument 6's `hits == 0`
    was trivially true and the get-side guard could be deleted with the
    whole suite green. The alias is live in a shipped install rather
    than contrived — the resolver's hyphen-to-underscore normalization
    makes `primitives/user-interaction/get-user-input.py` resolve to the
    packaged interactive primitive while `requires_user_input()` answers
    False for it.
11. **A script edited mid-run does not poison the old bytes**, pinning
    the re-read described in the bounds below.
12. **A project with no `script_cache` block gets a DISABLED cache**,
    through the real `ExecutorFactory`. This is the pin the flip needed
    and did not have.
13. **An explicit opt-in still wins**, so reading defaults off the
    dataclass cannot be mistidied into a constant.

Instrument 8 also gained a second half. Review showed its edit
assertions passed identically when persistence was broken in EITHER
direction, because a total persistence failure produces the same
observation as correct invalidation; it now watches an unchanged script
hit across a fresh cache too.

Mutation-verified. The one that matters most is instrument 7's —
building the key-time resolver without `project_dir` kills only
`test_a_project_relative_reference_is_also_invalidated`. An earlier
draft said the change would then be inert "for every shipped ensemble";
review corrected that. `ScriptResolver._get_search_paths` falls back to
`Path(os.getcwd())`, so with cwd at the project root the fix keeps
working; it goes inert where cwd differs from `project_dir`, which is
the serve, nested executors, and tests. The pin still earns its place.
Dropping the digest half and making `_cache_identity` a no-op are
indistinguishable to the suite (both kill the same three), which is the
correct signature for "reverted to the bug".

Note that `test_script_cache.py:74` is already named
`test_cache_invalidation_on_script_content_change` and passes today,
because it feeds inline content. Nobody should read the existing suite
as evidence of coverage here.

## Known bounds

- On-disk entries are orphaned on EVERY edit, not once. The old key can
  never be recomputed, `_load_from_artifacts` only unlinks on a TTL
  check for that exact key, and `clear()` has no caller in `src/`, so
  under `persist_to_artifacts` the cache directory grows without bound —
  one orphan per (script version, input, parameters). `max_size` bounds
  only the in-memory OrderedDict; eviction does not delete files. Small
  files and an opt-in setting, but a cleanup is a real follow-up rather
  than a one-time invalidation as an earlier draft claimed.
- A script edited mid-run used to store its new output under the OLD
  digest, since the digest is taken before the subprocess opens the
  file. An earlier draft called that self-correcting on the next run.
  Review demonstrated it is not: the old-bytes key stays poisoned for
  the full TTL, and two executions end up sharing an entry although they
  ran different bytes, which violates the invariant above outright
  rather than merely serving something stale. Now closed by re-reading
  the identity after the run and skipping the `set` if the bytes moved.
  The window is not eliminated, only the wrong entry is.
- One resolution plus one file read per script-agent execution, and a
  second on the `set` path only (the mid-run re-read above). An earlier
  draft said "computed at both `get` and `set`", which was wrong: the
  identity was computed exactly once per `execute`. Measured rather than
  assumed, since the draft recorded it as a thing to check: 0.136 ms
  mean for `classify.py` at 103KB, 0.096 ms for `emit.py`, 0.001 ms for
  an inline reference, against 20.9 ms for a bare `python -c pass`
  spawn. Roughly 0.7% of one subprocess spawn. On a cache HIT it is
  0.14 ms of new work where the old code did none, which is still
  noise.
- Still no in-flight deduplication (#158's bound), so N identical
  concurrent agents each miss and each run.

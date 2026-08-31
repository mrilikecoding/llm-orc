# #163 — an undigestable script is not cached (design)

Status: pre-flight. Issue: #163, filed from the #160 review.

## What is measured

**The degradation reproduces, end to end, with a stale serve.** With a
transient `OSError` at hash time and nothing else wrong, the identity drops
to path-only, the entry is written under it, and the EDITED script is served
the pre-edit result:

```
identity under EMFILE: '/tmp/.../probe.py'      (no digest — the #160 shape)
run 1                : {"v": "one"}
run 2 (edited to two): {"v": "one"}
stats                : hits 1, misses 1, sets 1
```

**The issue's named trigger is REFUTED.** #163 hypothesised a script that is
executable but not readable (mode 711, another owner), running "through
`_script_command`'s bare `[script_path]` branch". Two things are wrong with
that:

1. `ScriptAgentRunner` does not use `_script_command`. `script_agent.py`'s
   interpreter map runs `.py` under the host python and defaults everything
   else to `bash` — and every one of those interpreters must READ the file.
2. So an unreadable script cannot execute. Demonstrated: a locally compiled
   binary at mode 0311 runs fine from a shell (`{"v": "one"}`, exit 0) while
   `read_bytes` raises `PermissionError`; through the runner it fails with
   exit 126, `bash: ...: Permission denied`. #159 never caches a failure, so
   that path writes no entry at all.

**What IS reachable is a transient `OSError` while execution still
succeeds**: fd exhaustion (`EMFILE`/`ENFILE`), `EIO`/`ESTALE` on a network
filesystem, or an ENOENT race where the file is replaced between the resolve
and the read. Fd pressure is the plausible one here, and #158 made it more
so — script agents now run concurrently on a dedicated thread pool, so the
interpreters' file opens overlap where they used to be serialized. (The
identity's own `read_bytes` runs synchronously on the event loop and cannot
overlap another; the fd pressure it feels comes from the pool beside it.)

Severity is unchanged from the issue's: the bad key persists for the 3600s
TTL, and under `persist_to_artifacts` it crosses processes.

## Change

`_cache_identity` returns `str | None`. `None` means "this run has no
identity that names the script's bytes", and the run is then neither read
from nor written to the cache.

Five rows:

| reference | identity |
|---|---|
| the resolver calls it inline content AND it names nothing in the CWD | `script_ref` |
| the resolver calls it inline content BUT it names a CWD entry | `None` |
| it denotes a file that stats regular and digests | `f"{resolved}:{digest}"` |
| it denotes a file and ANYTHING goes wrong — resolve, stat (any errno), non-regular, unreadable | `None` |
| the reference is empty (a non-script agent) | `None` |

An earlier draft carried a per-row "filesystem calls" column, added to show
the table had been measured. Review round 6 found two of its five numbers
wrong: they were measured against ABSOLUTE references and stated as each
row's general answer, while the shape every shipped ensemble uses is
project-relative. Re-measured — an absolute hit costs 2 stats, a
project-relative hit 5, and an unresolvable project-relative reference 10,
because `_get_search_paths` probes before the search begins. That count is a
property of the resolver's search walk and of how a project is configured,
not of the identity rule, so stating it per row could only ever be an
instance presented as a rule. The column is gone; what it was really there
to record is the next paragraph.

**Inline content costs one `os.path.exists`** (measured: one `exists` and the
one `stat` inside it, for `echo hello`, a NUL byte and a 5000-character
string alike). The CLASSIFICATION is syntactic and pre-filesystem; the ANSWER
is not, because of the #177 disagreement below. A draft claimed these shapes
"never reach the filesystem", in the same commit that made that false.

The inline row is not a fallback: `resolve_script_path` returns inline
content verbatim, so `script: "echo hello"` genuinely IS its own bytes.

**The RESOLVER classifies; the identity only names bytes** (review round 3).
The errno rule that round 2 introduced opened a fourth site: `ENOENT` at the
identity's own stat, after a healthy resolve, answered the bare path for a
file that had just vanished — one of the three triggers this fix is named
for. It could not be repaired by trimming the errno set, because `ENOENT` is
also exactly what makes `script: "echo hello"` cacheable. One errno was being
asked two questions, and it cannot answer both.

The information lives one level up. `ScriptResolver` returns a reference
verbatim only through its final fall-through; every other successful return
is a file it found. So the resolver now answers `is_inline_content`, the
identity asks it BEFORE touching the filesystem, and from that point there is
no classification left to get wrong: the reference denotes a file, so the
resolve, the stat and the read all answer `None` and only a regular file that
digests produces an identity. `_ABSENT_ERRNOS` and the `ValueError` branch
are gone with it.

**The guard asks one question, at one site** (review round 2). The first two
drafts asked "which call threw" and so moved landing sites twice: round 1
moved it from `read_bytes` to the resolve, and round 2 found a third site
four lines below — `os.path.isfile`/`os.path.exists` are `genericpath`, which
SWALLOWS `OSError` and answers `False`, so an `EIO`/`ESTALE` stat fell
straight through to the bare path. The shape asked "did we establish that this is a
regular file we can name" via ONE `os.stat` with the errno deciding — which
round 3 then replaced again, because an errno cannot tell inline content
from a vanished file.

**The resolve boundary is inside the rule too** (review round 1). The fix
first landed only at `read_bytes`, but `resolve_script_path` stats the file
BEFORE the read and `Path.exists()` swallows only
ENOENT/ENOTDIR/EBADF/ELOOP — `EIO` and `ESTALE` propagate. Two of the three
triggers named above therefore land at the resolve, not the read, where
`except Exception: return script_ref` was the #160 key again. Review
demonstrated it as a live stale serve with the same numbers as the
reproduction above. Refusing to cache an unresolvable reference costs
nothing: execution resolves through the SAME resolver a moment later, so a
resolution failure is a run failure and #159 already declines to cache those.

**Scope decision, deliberate.** The issue asks only about the `read_bytes`
raise. The non-regular-file row is included because it degrades identically
and today has a pin asserting the degradation
(`test_a_fifo_reference_does_not_hang_the_agent` asserts a path-only
identity). Shipping a fix whose
stated invariant is already violated one branch over is how invariants drift
(doctrine 4: state the invariant, not the instance). That pin's real subject
— computing an identity must not HANG on a FIFO — is unchanged; only its
identity assertion moves.

Not in scope: #161 (primitive purity) and #162 (imports are not covered by
the identity). Both are about what the digest COVERS; this is about what
happens when there is no digest at all.

## Invariant

The script cache is read and written only under an identity that names the
script's bytes. When the bytes cannot be named, the run is not cached.

## Regression instruments

Fifteen, named by test. The previous version had item 8 and item 12 pointing
at the same test, item 2 covering two, item 3 naming none, and a note whose
account of the composition was wrong — the count came out right because two
of those errors cancelled.

Reconciling with the collector: `pytest --collect-only` reports **18** across
the three new classes, because item 11 is parametrized over eight cases. So
10 single tests + 1 parametrized instrument = 11 new, plus 4 pre-existing
tests this arc modified = 15.

The pre-existing four are the ones `git diff main...HEAD` shows a body change
for, mapped hunk by hunk to the enclosing test. Round 5 also said four and
got the SET wrong: it listed `test_inline_content_still_caches`, which is
byte-identical to main, and omitted
`test_the_entry_carries_no_lying_success_field`, which gained an assertion.
One swap, not one addition — round 6 read it as a missing fifth because two
of the diff's hunks fall inside the same test. The count survives only
because the errors cancel again, which is exactly why this list is named by
test and mapped to hunks rather than counted.

**New (`TestUndigestableScriptIsNotCached`, `TestAResolveFailureIsAlsoUndigestable`, `TestIsInlineContent`):**

1. `test_a_transient_read_failure_serves_no_stale_result` — the issue's
   reproduction, end to end: a transient `OSError` at hash time used to
   install a path-only key and serve the pre-edit output.
2. `test_nothing_is_written_under_a_digestless_identity` — the WRITE half.
3. `test_an_existing_entry_is_not_consulted_without_a_digest` — the READ
   half, observed as the CALL, since a get that ran would simply miss.
4. `test_becoming_undigestable_mid_run_stores_nothing` — a post-run `None`
   must fail closed rather than compare equal.
5. `test_a_resolve_failure_is_not_cacheable` — round 1's site, by mechanism
   after round 3 showed the end-to-end version could not fail.
6. `test_an_enoent_after_a_healthy_resolve_is_not_cacheable` — round 3's
   blocker, the ENOENT race the design names as a trigger.
7. `test_a_stat_failure_after_a_healthy_resolve_is_not_cacheable` — round
   2's blocker, isolated by mechanism because counting calls does not
   survive a mutant that adds stats.
8. `test_a_bare_name_that_is_a_file_serves_no_stale_result` — round 4's
   blocker and a regression round 3 introduced. End to end, because the
   unit answer looks fine in isolation.
9. `test_a_symlinked_script_still_digests` — the over-refusal direction for
   the stat, asserted against the target's real sha256.
10. `test_an_empty_reference_never_touches_the_cache` — a non-script agent
    has no bytes to name.
11. `TestIsInlineContent::test_the_classification` — eight cases over the
    predicate the whole fix now rests on. Round 5: its `"/"` clause was
    deletable with 3992 tests green.

**Pre-existing, modified by this arc:**

12. `test_a_fifo_reference_does_not_hang_the_agent` — the watchdog is the
    point and is unchanged; its identity assertion moved to `None`.
13. `test_a_project_relative_reference_is_also_invalidated` — round 4 added
    `sets == 2`, because dropping `project_dir` now yields `None` and the
    edit assertions were being satisfied by an ABSENT cache.
14. `test_inline_content_with_a_nul_byte_does_not_raise` — retitled in round
    5; the property is that `os.path.exists` answers for the awkward shapes
    rather than raising, not that they stay off the filesystem.
15. `test_the_entry_carries_no_lying_success_field` — gained
    `assert identity is not None`, since the identity is now nullable and a
    `None` would make the entry it looks up impossible rather than absent.

`test_inline_content_still_caches` is NOT in this list. Round 5 put it here;
its body is byte-identical to main. It is load-bearing for this arc — it is
what discriminates the inline classification — but it is a pre-existing pin
this arc relies on rather than one it changed, and the list is about the
latter.

## Known bounds

- Does not make an undigestable script cacheable. It converts a silent stale
  serve into a cache miss, which costs one execution.
- The permission case is refuted as a trigger for a POISONED entry, but the
  identity still degrades there; it simply never reaches a `set` because the
  execution fails first. The guard covers it anyway rather than relying on
  that coincidence holding.
- Says nothing about what the digest covers (#161, #162).
- **A legitimately inline reference that collides with a CWD entry is
  silently uncacheable.** The cost of failing closed on the #177
  disagreement. It can only refuse, never serve wrong, and no shipped
  `script:` value can hit it — none is extensionless. Two carry no
  separator (`aggregator.py` in `adr-swarm-review.yaml`,
  `test_simple_input.py` in `testing/simple-input-test.yaml`; derive:
  `grep -rn --include="*.yaml" --include="*.yml" -E
  "^[[:space:]]*script:[[:space:]]*[^/\\ ]*$" .` — two rows), but the
  `SCRIPT_EXTENSIONS` clause classifies both as
  paths. So the bound is live only for a future extensionless bare-word
  ref (`script: helper`): a project using one in a directory that happens
  to contain a file of that name loses caching with no error. A debug log
  names it.
- **The whole change is inert on a default install.** `ScriptCacheConfig`
  ships `enabled = False` (#160), and `cacheable` gates the identity
  computation entirely, so none of this runs until a project opts in. The
  severity above — a bad key held for the TTL, crossing processes under
  `persist_to_artifacts` — is real only for those projects.
- The `ScriptNotFoundError` special case was dropped for a better reason
  than the docstring first gave. "A resolution failure is a run failure" is
  falsified by this branch's own pin, where the run succeeds while the
  identity's resolve failed. The real argument: `ScriptNotFoundError`
  subclasses `FileNotFoundError` and is what the resolver raises during an
  ENOENT race — one of the three named triggers — so keeping such a
  reference cacheable would have re-opened the path-only key for it.

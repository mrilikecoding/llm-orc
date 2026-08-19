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
so — script agents now run concurrently on a dedicated thread pool, so many
whole-file reads overlap where they used to be serialized.

Severity is unchanged from the issue's: the bad key persists for the 3600s
TTL, and under `persist_to_artifacts` it crosses processes.

## Change

`_cache_identity` returns `str | None`. `None` means "this run has no
identity that names the script's bytes", and the run is then neither read
from nor written to the cache.

Three branches, one rule:

| reference | identity |
|---|---|
| the resolver calls it inline content (no separator, no script extension) | `script_ref` |
| it denotes a file, stats as regular, and digests | `f"{resolved}:{digest}"` |
| it denotes a file and ANYTHING goes wrong — resolve, stat (any errno), non-regular, unreadable | `None` |
| the reference is empty (a non-script agent) | `None` |

Three rows, because after the classification there is only one question left.
A NUL byte, a string longer than PATH_MAX and `echo hello` are all inline by
the same rule and never reach the filesystem; a FIFO, a vanished file, a
stale mount and an unreadable file are all "a file we cannot name".

The last row is not a fallback: `resolve_script_path` returns inline content
verbatim, so `script: "echo hello"` genuinely IS its own bytes, and an
unresolvable reference must not have its missing-script error reported by the
cache-key path. `os.path.exists` is what separates that row from the
non-regular-file row; it is safe on every shape the current docstring worries
about (NUL byte, longer than PATH_MAX, empty) — measured, all return `False`
rather than raising.

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
straight through to the bare path. The shape now asks "did we establish that
this is a regular file we can name": ONE `os.stat`, with the errno deciding
whether nothing is there (inline content, a missing script) or something is
there that we failed to look at. One stat also removes the isfile/exists
TOCTOU.

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
and today has a pin asserting the degradation (`test_a_fifo_reference_does_
not_hang_the_agent` asserts a path-only identity). Shipping a fix whose
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

Every one must go RED under deletion of the guard it pins — the recurring
failure in this corpus is a pin that cannot fail (#156, #160 round 2).

1. **A transient `OSError` at hash time serves no stale result.** The
   reproduction above, as a test: raise from `read_bytes`, run, edit, run —
   the second run returns the new bytes' output and `hits == 0`.
2. **The skip at the GET is pinned separately from the skip at the SET.**
   #160's round 2 found a get-side skip that was deletable with the whole
   suite green; two clauses need two pins.
3. **A digestable script still caches.** The pin that stops this becoming
   "never cache anything" — the degradation #160's review named explicitly.
4. **An identity that becomes undigestable MID-RUN is not stored.**
   `edited_mid_run` compares the post-run identity to the pre-run one; a
   `None` on the second call must fail closed rather than compare equal.
5. **Inline content still caches**, so the `exists` split does not swallow
   the legitimate no-digest row.
6. **A FIFO still does not hang, and now does not cache.** The existing pin's
   watchdog is kept; its identity assertion changes.
7. **A vanished file after a HEALTHY resolve is not cacheable** — round 3's
   blocker, the `ENOENT` race the design names as a trigger.
8. **A symlink to a real script still digests**, the over-refusal direction
   for the stat.
9. **A stat failure after a HEALTHY resolve is not cacheable** (round 2's
   blocker). Isolated by MECHANISM, not by call order or caller name: the
   resolver checks with `Path.exists`, the identity stats with `os.stat`, so
   patching them apart makes the resolve succeed and only the identity's own
   stat fail. Counting calls would not do — reinstating the genericpath
   prelude ADDS stats and shifts the numbering, which is how that mutant
   survived a first attempt at this pin.
10. **A resolve failure is not cacheable.** Round 1's site, pinned by
    mechanism after round 3 showed the end-to-end version could not fail:
    scripting the fault at the resolver's first stat left the POST-run
    identity healthy, so `edited_mid_run` suppressed the write for an
    unrelated reason and reverting the guard survived the whole suite.
    Asserting the ANSWER rather than a downstream side effect makes the
    schedule irrelevant.

## Known bounds

- Does not make an undigestable script cacheable. It converts a silent stale
  serve into a cache miss, which costs one execution.
- The permission case is refuted as a trigger for a POISONED entry, but the
  identity still degrades there; it simply never reaches a `set` because the
  execution fails first. The guard covers it anyway rather than relying on
  that coincidence holding.
- Says nothing about what the digest covers (#161, #162).
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

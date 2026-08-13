# Serve-native dot-dir self-reference (#144) — design

**Status:** revised 2026-08-13 after reviewer pre-flight (PROCEED-WITH-
CHANGES, 3 blockers + 4 majors, all adjudicated below). This revision is
the source of truth; the pre-flight findings live in the session record
and issue #144.

**Slice:** extend explain-discovery so serve-owned scripts (the project's
own `.llm-orc/scripts/agentic_serving/*.py`) are discoverable and readable
without the client seam. OpenCode glob cannot list dot-dirs (verified
2026-07-14). Direction pinned by the issue: serve-native, never rides the
client seam.

**Evidence base:** `2026-08-13-serve-owned-discovery-findings.md` — the
shipped filename-subset rule holds at serve-owned scope (6/12 exactly-one,
0 wrong-file on the serve-flavored battery), five discovered files admit
under the read budget, classify.py refuses at 35,044 > 35,000 (the #145
pin), and both rescue paths (content-grep, AST sections) are refuted. The
slice grounds five real self-referential questions and converts the
literal gate question from speculation to an honest refusal naming the
true file. The "answers grounded" exit for the classify question stays
open on #144, blocked by the whale — unblockers #106 / #151 / chunked
reads.

## Pre-flight resolutions (bind the mechanism below)

1. **Opt-in flag, default OFF (closes the contamination blocker).**
   Serve-owned discovery runs only when `serving.self_reference: true` is
   set in the project's `.llm-orc` config. Any served project carries
   generically named scripts (shape.py, resolve.py, emit.py); with the
   union always on, "how does shape affect the output?" in a matplotlib
   project would ground the serve's seat-prompt builder. No deterministic
   wording gate can separate that from the real gate question (spike 1
   territory), so the boundary is declarative: OFF everywhere by default
   (zero behavior change for every served project), ON in this repo
   (committed config — self-hosting is the rung). Enabling it generally
   is future work gated on a negative-control battery. The caller reads
   the flag via `_load_config` and threads it to classify as a new turn
   field `self_reference: bool`.
2. **Union timing (closes the truncation blocker).** When the turn is
   explain-discovery and glob is in play, the glob round always runs
   first, exactly as today. Serve-owned candidates are unioned only when
   a COMPLETE listing is present. Listing truncated → serve-owned
   grounding is disabled for the turn (existing truncated-listing
   behavior unchanged; #148 semantics mirrored). No short-circuit on
   pass 1: a serve-owned match never suppresses the glob round, because
   the workspace half of the union (and the collision refusal) needs it.
3. **Full-path keys for the self namespace (closes the collision
   blocker).** Self-read visibility, attempted-tracking, and rendering
   all key on the `.llm-orc`-rooted relative path
   (`.llm-orc/scripts/agentic_serving/resolve.py`), never the basename.
   A visible workspace `resolve.py` must not satisfy a self-read of the
   serve's resolve.py, and a failed/over-budget self-read must not poison
   `attempted` for a workspace file of the same basename. The rendered
   header (`assistant: [read .llm-orc/scripts/agentic_serving/resolve.py]`)
   must stay parse-compatible with `_VISIBLE_HEADER_RE` /
   `_READ_ATTEMPT_RE` (implementer verifies the path charset admits the
   leading dot; extending the regex charset is in-scope, adding a new
   variant string is not).
4. **No glob-unavailable branch in this slice.** classify cannot see tool
   advertisement (it never crosses the `_serve` payload seam), and the
   toolless meta-call gate (`_aux_reply`) makes the zero-tool claim
   unreachable. Scope claim corrected: this slice serves clients with a
   working glob (OpenCode). Glob-failed keeps failing closed.
5. **Single-whale over-budget wording.** When the over-budget refusal
   composes and NO other files hold the budget, the message states that
   the file alone exceeds the session read budget (today's wording —
   "held by other files, start a fresh session" — is false in that case).
   General honesty fix; reachable on the client seam too; lands with its
   own pin test.
6. **Named-file self-reads (closes the cross-turn evaporation major).**
   Self-read blocks are synthesized server-side and never echo back from
   the client, so they evaporate between turns. The named-file explain
   path therefore learns the self namespace: an explain that names a file
   resolving into the enumerated serve-owned set (flag on) derives a
   self-read instead of firing not-grounded. Follow-ups stay coherent.
7. **Refuse-on-unrecognized terminal outcome (closes the skew major).**
   `_outcome_chunks`' terminal branch currently write-defaults; an old
   caller meeting a new outcome key would emit a junk write. The branch
   changes to an honest refusal on unrecognized outcomes (own test), and
   the scripts/caller pairing note rides the release.
8. **Confinement invariant, corrected.** Membership, not prefix: the
   caller re-derives the enumerated set (`project_dir/scripts/
   agentic_serving/*.py`), resolves BOTH the candidate and the set
   entries (`Path.resolve()`; macOS `/var`→`/private/var` in fixtures),
   and admits a self-read only if the resolved path is IN the set.
   `is_relative_to`/set-membership, never `str.startswith`.
9. **Trusted bytes stay trusted.** Self-reads do not pass through the
   client-wire normalizer heuristics (failed-read prefix sniff would
   misfire on a script starting with "Error"). The caller wraps disk
   content the way the spike did and renders via `_render_read_block`
   directly, so the budget projection matches the measured 35,044 pin.
10. **Bookkeeping.** Self-read re-entry marks its turn trace
    (`self_read_round: N`) so ladder instruments don't over-count turns.
    Disk read is synchronous (precedent `_load_emit_reject_prefixes` is
    sync; the whale is 80KB). Refusal and attribution wording uses
    `.llm-orc`-rooted paths everywhere. The multi-self-match refusal
    (two serve-owned candidates) keeps the exactly-one-or-refuse
    conversion but its wording names where the candidates live (not
    "workspace listing").

## Mechanism

1. **Serve-owned candidate rule (classify.py, new `_self_candidates`):**
   when `self_reference` is on, enumerate
   `sorted(Path(__file__).parent.glob("*.py"))` — the operative script
   set for THIS serve instance (fixtures copytree the scripts, verified).
   Apply the byte-identical filename-subset rule used by
   `_explain_glob_candidates`. No new discovery heuristic.
2. **Union at candidate time (`_explain_discover`):** after a complete
   glob listing: candidates = workspace candidates ∪ serve-owned
   candidates. Exactly-one-or-refuse over the union:
   - one, workspace → existing `needs_files` (client read, unchanged);
   - one, serve-owned → new signal `needs_self_files` (`.llm-orc`-rooted
     path);
   - zero → conceptual fall-through (unchanged, deliberate);
   - two+ → refuse-with-candidates naming both halves honestly.
3. **New chain step:** `SignalBundle.needs_self_files` + guard + Step row
   in CHAIN_EXPLAIN beside the explain need-files row (before
   `explainer`; first-match placement is the semantics). Dispatch shape
   `need-self-files.yaml` + echo script, threaded through resolve →
   shape → form_gate → emit like every seam field.
4. **Caller executes the read natively:** on the self-read outcome, no
   ClientToolCall. The caller checks set-membership (resolution 8), reads
   the file synchronously, renders per resolution 9, appends the block to
   the conversation, and re-enters the pipeline in-process. Rendering,
   the 96KB cap, and the token budget run through the SAME
   `_render_read_block` / `_budget_read_blocks` path as client reads —
   budget parity is non-negotiable; classify.py must refuse over-budget
   here exactly as on the client seam (with resolution 5's wording).
5. **Termination:** visible-or-attempted after one re-entry means no
   re-request (pre-flight verified), plus the deterministic backstop: at
   most 3 self-read re-entries per turn, exceeded → fail-closed refusal.
   No model judgment anywhere.

## Invariants (rule 6) and regression instruments

- **Confinement:** a self-read only ever reads a file in the enumerated
  serve-owned set. Instrument: caller tests feeding traversal (`../`),
  absolute, symlink-escape, and non-enumerated-but-inside-project paths,
  asserting refusal and no read.
- **Budget parity:** classify.py through the self-read path refuses
  over-budget exactly as on the client seam. Instrument: subprocess
  corpus pin + caller render test against the 35,044/35,000 bound.
- **Namespace isolation:** a visible workspace file never satisfies, and
  a failed self-read never poisons, the other namespace. Instrument:
  collision tests from pre-flight finding 2's scenarios.
- **Default-off:** with the flag absent, behavior is byte-identical to
  today. Instrument: corpus regression probes run in a flag-off fixture.

## Tests (TDD order)

1. `test_serving_classify.py` (subprocess corpus): flag-off = unchanged
   behavior; flag-on gate question → glob round → union →
   `needs_self_files`; whale over-budget refusal names
   `.llm-orc/scripts/agentic_serving/classify.py` with resolution 5's
   wording; "how does resolve pick the seat?" → grounded from
   resolve.py; workspace/serve-owned collision refuses naming both;
   two-serve-owned refusal wording; truncated listing disables the
   union; named-file follow-up self-reads; glob-failed refuses.
2. `test_chain_plan.py`: row test + ordering test for the new Step.
3. resolve/shape/form_gate/emit pass-through tests for `needs_self_files`.
4. Caller tests: confinement set (traversal/absolute/symlink/
   non-enumerated), budget parity, re-entry bound, no ClientToolCall for
   self-reads, refuse-on-unrecognized terminal outcome, trace marking.
5. `test_serving_ensemble_endpoint.py` (hermetic HTTP): flag-on fixture
   (config written into the tmp project) gate question end to end — zero
   read tool_calls on the wire for the serve-owned path; flag-off
   fixture unchanged.

## Live exit (RIG)

Real OpenCode against this checkout (flag committed on): "how does
resolve pick the seat?" answers grounded in resolve.py's actual content;
"how does classify decide routing?" refuses over-budget naming the real
file with the single-whale wording (no speculation). Both logged; dogfood
entry if serve-shaped. Ladder rerun (13-turn battery, no regression) +
trajectory row + author-independent adversarial review with a
wrong-accept hunt (does any self-read ground the wrong file, cross a
namespace, escape the enumerated set, or dodge the budget?).

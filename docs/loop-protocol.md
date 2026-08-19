# Loop protocol

Operating rules for any autonomous or delegated session working this repo.
This artifact governs; session memories and model-specific configuration
mirror it at best. Every rule below was paid for with a measured failure
(provenance: git history, the roadmap archive, run READMEs).

## Work structure

1. Every unit of work has a GitHub issue. Issues carry exactly one
   `epic:*` label. No untracked tasks.
2. The roadmap (`docs/serving-roadmap.md`) lists outcomes and
   issue-linked steps only. Narrative goes in run records and design
   docs, not the plan.
3. The roadmap State section is rewritten, never appended.

## Merge gate

4. No merge to main without an author-independent adversarial review
   APPROVE (wrong-accept hunt). The reviewer re-verifies fixes to its own
   findings; new review rounds get fresh demonstrating inputs, not
   re-reads.
5. Any honesty/scoring claim (J-tier) comes from an independent scorer
   against the frozen rubric. Author-scored passes close nothing.
6. Fix adjudications by the session lead follow three rules:
   evidence-gated leniency (no absorbing/default behavior without a
   demonstrating capture in the repo; otherwise fail-loud), reviewer
   pre-flight on design-changing directions (one cheap exchange before an
   implementer round), and every brief names the invariant plus the
   regression instrument that pins it.

## Instruments and runs

7. Before first use of an instrument (battery, capture, driver): read the
   entire script header, verify setup against a known-good baseline
   record (e.g. truth-00 manifest match). When a precondition bites, the
   fix is a deterministic guard in the instrument, shipped immediately.
8. Run outputs are evidence. Never delete them; move aside under a
   `discarded-` name with a note. Battery refuses dirty out dirs by
   construction.
9. After every `git push`: watch the triggered workflows to conclusion
   (`gh run watch` or equivalent). A red run is the next work item, not
   background noise.

## Dogfooding

10. Discrete serve-shaped checks from the session's own work route
    through `opencode run -m llm-orc/agentic` first; honest refusals are
    data. Log every attempt in `docs/dogfood-log.md`. Scope grows toward
    the serve owning loop subtasks (the north star, reflexively).

## Scope

13. A finding surfaced while working an arc is FIXED in that arc when it is
    small and clearly defined: one predicate, one wording, one guard, one
    check matching a pattern already in the tree. Filing is for work that
    genuinely changes scope — a different subsystem, a behaviour change
    needing its own live validation, or an arc-sized design.

    These are NOT reasons to defer: "it needs its own review round" (reviews
    re-run, and the round is cheaper than the queue); "it would invalidate
    the review so far" (process is not the deliverable); "it is
    pre-existing" (it is inside the diff's blast radius now, and nobody else
    is looking at it). Paid for: one session closed five issues and filed
    eight, at least four of which were one-line changes it had already
    diagnosed.

14. An arc that files more issues than it closes is scoped to a symptom.
    Name the general fix in the arc's record and take it, or say plainly why
    the symptom is the right place to stop. A growing queue is not progress
    toward the north star; it is progress away from it.

## Claims and instruments

15. Every factual claim in an artifact — a count, a call count, a behaviour,
    a list, a table row — is DERIVED by running something, and the deriving
    command sits next to the claim. Paid for: a design doc's instrument
    section was wrong in four consecutive rounds, and every correction was
    written from what the author meant the code to do rather than from the
    code. A bare count is the weakest form and fails silently when two
    errors cancel (measured, twice); name the items so one command checks
    the claim.

16. A mutation is not evidence until the mutant is confirmed live: assert
    the anchor occurs exactly once, then exercise the mutated function
    before trusting a green run. Paid for: an anchor string occurred three
    times in one file and `replace(..., 1)` mutated an unrelated function
    362 lines away, producing three false "this pin cannot fail" readings.

17. A probe is validated on a known-good input before its output is trusted
    — rule 7, extended from batteries to the ad-hoc scripts a session writes
    for itself. Paid for: a shell `echo` mangled JSON escapes, the node
    under test returned all-defaults, and the broken probe was nearly
    recorded as a falsified reproduction.

18. Two fixes of one defect class exhaust the instance budget; the third is
    structural or it is not made. State the invariant the structure enforces
    and let the old instances fall out of it. Paid for: one guard moved
    landing sites five times, and another leaked through seven channels,
    twice inside the function written to close the previous one.

19. Doc and code drift silently: no test fails when prose describes deleted
    code. `make lint` runs `scripts/check_doc_drift.py`, which resolves
    every `test_*` name a design doc mentions. Extend the check rather than
    re-deriving the claim by hand.

## Style

11. Commits: conventional prefixes, structural/behavioral separated, no
    AI attribution of any kind, no session links.
12. Issue and doc prose: outcome first, steps linked to issues, no
    narrative filler.

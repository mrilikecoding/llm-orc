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

## Style

11. Commits: conventional prefixes, structural/behavioral separated, no
    AI attribution of any kind, no session links.
12. Issue and doc prose: outcome first, steps linked to issues, no
    narrative filler.

# grep->read pattern-derivation spikes (2026-07-14)

Question: can a DETERMINISTIC token->grep rule surface the right file for real
llm-orc meta-task questions, without overfitting a stopword list to the gate?
Method: 8 real questions, extract identifier tokens minus a general-English
stopword set (no code terms), grep src + .llm-orc + tests on the real repo.

## Spike 1 (bare-mention grep) — REFUTED
A question about a central concept names the repo's MOST common strings:
- "how does classify decide routing?" -> classify=100, decide=52, routing=64 files.
- chain=207, gate=274, build=139, tool=372, read=206 ...
So bare-token grep refuses even the roadmap's own gate question. No stopword
list fixes this without being tuned to the gate = the overfit to avoid.

## Spike 2 (name/def signal) — filename-stem is the one clean signal
Ranking by "where a symbol is NAMED/DEFINED" not "mentioned":
- FILENAME-stem is reliably right: classify->classify.py, accept/gate->accept_gate.py,
  chain->chain_plan.py, executor->accept_executor.py, write->write_file.py.
- def-site (`def/class <token>`) is noisy: test files reference every symbol.
- a naive "count of tokens hitting a file" rank favors test_*.py (they mention
  many symbols) — do NOT rank that way.

## Conclusion (drove the design reframe)
- Reliable deterministic signal = filename stem, which the serve ALREADY has
  (discovery glob: charset stem, .py, not test_*, one-or-refuse).
- FIRST slice = glob->read grounded-explain (file/module-named questions).
- Content-grep (symbols that are not filenames: routing->_route,
  normalize->a method) = NEXT rung; its NL->pattern derivation goes to a
  guarded model (Approach B) + charset-sanitize + grep find-or-refuse backstop,
  because deterministic content extraction is demonstrably too blunt.

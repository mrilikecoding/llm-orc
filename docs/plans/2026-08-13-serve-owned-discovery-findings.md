# Spike 3: serve-owned discovery signals (#144) — findings

Script: `2026-08-13-serve-owned-discovery-spike.py` (run against the real
repo, 31 serve-owned scripts, the live `_READ_TOKEN_BUDGET = 35000` and the
real `_render_read_block`/`projected_tokens_v2` path).

## Results

1. **Filename-subset (the shipped `_explain_glob_candidates` rule) holds at
   serve-owned scope.** 6/12 battery questions resolve to exactly one
   candidate; the other 6 yield NONE (fall through, today's behavior). Zero
   wrong-file across the battery. No new discovery rule is needed — #144 is
   the same rule over a serve-native enumeration.
2. **Five of the six discovered files ADMIT under the read budget:**
   accept_gate 1977, form_gate 1745, shape 2581, resolve 3190, emit 5469
   tokens. Serve-native read grounds them immediately.
3. **The literal gate question discovers classify.py and it REFUSES:**
   35,044 tokens > 35,000. This is the #145 pin (classify.py refuses
   over-budget by design), reproduced through the real render path.
4. **Content-grep is still refuted at 31-file scale** (spike 1's refutation
   rescoped): gate=25 files, shape=25, seat=21, emit=22, accept=19.
   Exactly-one-or-refuse over content hits fails on serve-owned scope too.
5. **Deterministic AST section fallback is refuted for the gate question:**
   no top-level def/class in classify.py has a name containing
   classify/decide/routing — there is no cheap within-file section to
   extract. (S5 fired only where S4 refused, i.e. only classify.py.)

## Consequence for #144

Implement the slice: serve-native lookup (self-enumeration of the scripts
dir) + serve-native read through the SAME render/budget discipline. The
gate question upgrades from conceptual speculation to an honest over-budget
refusal naming `.llm-orc/scripts/agentic_serving/classify.py`; five other
self-referential questions answer grounded. The literal "answers grounded"
exit for the classify question is blocked by the whale file itself, not by
discovery; unblockers already on the roadmap: #106 (move shape/table data
out of classify.py), #151 (server-queried real window), or the deferred
chunked-read rung. Recorded on #144.

# Spike Ω-2b — Runtime control: recovery (adapter-side retry)

**Status:** RAN 2026-06-29. Recovery mechanism PASS; task-correctness OPEN
(the deterministic gate is too weak for "converged" to mean "correct").
See `docs/agentic-serving/proposals/ensemble-spike-sequence.md` §4b.

This is the recovery half of Ω-2b. Dynamic dispatch + the OpenCode smoke
test (the other two thirds of §4b) are not yet built — held pending the
gate-strength decision this run surfaced.

## Question

Does adapter-side retry on validate failure (bespoke's ADR-041 self-healing
re-dispatch, relocated from in-process to between-turn) make the 3-file task
converge where the Ω-2 form could not (0 of 9 runs)? No engine primitive.

## Shape

The Ω-2 ensemble + harness, extended with:
- recovery-aware `parse` + `dispatch_shim` (thread the exact ast error back
  to the coder verbatim, bypassing the plan LLM's paraphrase);
- an adapter-side retry loop in the harness (N=2 retries/file, then
  drop-and-continue so partial convergence is measurable);
- a marshal that always emits the terminal write (the Ω-2 marshal dropped
  the last file).

Artifacts: `.llm-orc/ensembles/spike-omega-2b/agent-turn-omega2b.yaml`,
`.llm-orc/scripts/spike-omega-2b/{parse,dispatch_shim,marshal}.py`,
`scratch/spike-omega-2b/run_recovery.py`. Reuses `spike-omega/validate.py`
and `code-generator-omega`.

## Result — two runs

| | run1 (marshal bug) | run2 (marshal fixed) |
|---|---|---|
| files on disk | 2 of 3 (README dropped) | 3 of 3 |
| converters.py | correct, 1 retry | correct, 1 retry |
| cli.py | **wrong**: re-implements funcs, not a CLI, no import | real argparse CLI, imports converters, **3 attempts** — but wrong command structure (`c-to-f` top-level, not under `convert`), so the specified `cli.py convert c-to-f 25` errors |
| README.md | validated then dropped | **broken**: Python source in a `.md`, gate has no `.md` check |
| retries used | converters ×1 | converters ×1, cli ×2 (one a 413s empty-content blowup) |
| wall-clock | 405s (6.75 min) | 935s (15.6 min) |
| functional correctness | ~1 of 3 | ~1.5 of 3 |

## Findings

### #1 — Recovery works, and does real work. (PASS, mechanism)

Every run, at least one file failed `ast.parse` on attempt 1 and recovered
on a later attempt (converters.py both runs; cli.py needed all 3 in run2).
The exact-error hint + "re-emit only this file, first byte is first byte"
steers the coder off the form bleeds (shell prefixes, indentation, prose).
The retry lives entirely in the adapter; the hint rides the existing request
fields; the substrate stays the cross-run store. **No engine primitive** —
the §8 boundary absorbs the recovery loop, as predicted. This is the thing
Ω-2 (no recovery) could not do.

### #2 — "Converged" overstates correctness; the gate is the bottleneck.

`validate` = `ast.parse` on `.py`, nothing on `.md`. So:
- cli.py run2 passes the gate but doesn't implement the specified command
  interface — form-valid, semantically off-spec.
- README.md run2 passes the gate while being Python code in a `.md` file —
  the gate has no markdown check at all.
- The grep-based coherence check is unreliable both ways: run1 false-positive
  (copy-paste counted as "references APIs"); run2 it aligns with a real
  import but still can't see the command-structure bug or the README bug.

So all-3-validated-and-on-disk ≠ correct. Real correctness was ~1.5 of 3.
**The deterministic-first §5 "verification BEATS frontier" claim holds only
for the narrow form (parse) dimension.** Spec-conformance and content
correctness need destination-aware structural gates (cli.py must expose the
named commands; README.md must be Markdown), which is substantial per-
deliverable deterministic cruft — the same gate stack the bespoke built
(FormGate, ArtifactBridge destination-validity), re-implemented as scripts.

### #3 — cli.py is high-variance, not systematically wrong.

Run1 it was a wrong re-implementation; run2 a near-correct CLI. The content
quality swings run-to-run. Recovery + retries improve the odds of landing
*something form-valid* but do not guarantee *correct*. The variance, not a
fixed defect, is the story.

### #4 — Latency is now quantified and damning for long-horizon (B).

6.75 min (run1, one retry) vs 15.6 min (run2, three retries on cli.py incl. a
413s single-turn empty-content qwen3 blowup) — for a 3-file toy. The retry
loop that rescues form failures also multiplies latency, with high run-to-run
variance the §5 long-horizon bet has to survive. A form taking 6-16 min to
produce ~1.5/3 correct files on a toy is not yet competitive with single-
context frontier; Ω-4 would inherit this.

## The (A)-vs-(B) read this run sharpens

Recovery validates the adapter-boundary thesis structurally (good for (B):
runtime control fits the §8 adapter, no engine primitive). But empirically
the ensemble form is not yet competitive: the deterministic gate is the
reliability bottleneck and closing it means rebuilding the bespoke's gate
stack as scripts, and latency is 3-6× bespoke with high variance. The
mechanism is well-demonstrated; the value proposition (reliability, latency)
is not there yet. On outcomes this still weakly favors (A); (B)'s appeal
remains architectural (unification, agent-as-data).

## What stays open

- **Gate strength** (the decision this run surfaces): destination-aware
  structural checks vs. the current `ast.parse`-only gate. Deterministic
  (per the standing determinism-over-carve-outs preference), not an LLM judge.
- **Dynamic dispatch** (§4b item 2): runtime capability selection,
  adapter-mediated. Mechanism test, orthogonal to gate quality.
- **OpenCode smoke** (§4b item 3): the real-client contract check.
- Whether to continue the Ω sequence toward Ω-4 or step back to the
  (A)-vs-(B) fork with this evidence in hand.

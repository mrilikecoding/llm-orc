# Judge false-reject measurement (#84, first half of the gate integrity pair)

**Problem:** with retries wired (#89) and the held TDD round shipped
(#100), judge adequacy on round 1 is the accept path's entry condition —
and live evidence shows conservative/unstable adequacy verdicts are the
visible bottleneck (2026-07-08 min-stack turn: executor passes, judge
still rejects; 2026-07-09: both rounds judged inadequate on a trivial
dataclass task). Nothing measures the judge, so tuning it is guesswork.

**Approved sequence (2026-07-09):** measure (#84) → tune on the data →
then the #98 test-writing shape on the tuned judge.

## 1. Judge extraction (structural, behavior-preserving)

The adequacy judge moves from an inline model node in
`build-gated-round.yaml` to its own `adequacy-judge` ensemble (same
profile `agentic-tier-cheap-general`, `think: false`, prompt verbatim),
referenced from the round like `test-writer` / `code-generator` are.
One definition, three consumers: the round, the harness, the future #98
shape. The round keeps `input_scope: dependencies` on the reference —
verify the engine supports that combination first; if not, close that
engine gap (AS-11: extend the engine, never work around it).

## 2. Fixtures

`benchmarks/judge_adequacy/fixtures.yaml` — ~14 labeled
`{name, class, requirement, code, tests, expected_adequate}` entries:

| Class | Label | What it probes |
|---|---|---|
| adequate-clear | accept | behavior tests with real values |
| adequate-terse | accept | short but catching — the false-reject bait |
| inadequate-trivial | reject | assert True / reflection checks |
| inadequate-tautological | reject | pass any implementation |

Plus a failing-but-adequate pair (wrong code, good tests): adequacy is a
property of the tests, independent of pass status.

## 3. Harness

`benchmarks/judge_adequacy/run.py`:

- Per fixture × N samples (default 8), sequential (32GB rig).
- Contract composed by the REAL `accept_executor.py` (subprocess), then
  judged by the real `adequacy-judge` ensemble through the engine.
- **Fidelity requirement:** the judge sees byte-identical input to what
  it sees inside the round — reuse the engine's dependency-rendering
  path, never a reimplementation.
- Output: timestamped JSONL under `benchmarks/judge_adequacy/runs/`
  (retained artifacts) + a summary: false-reject rate and false-accept
  rate per class, flip variance per fixture.

## 4. Tuning loop

Success gate: false-reject on adequate classes < ~10% while inadequate
rejection stays ≥ ~90%. Lever order: judge prompt calibration first;
ADR-048 §5 AND-vs-weighted gate composition only if prompt work cannot
get there. Every change reruns the harness; results land in the roadmap.

## 5. CI tests (hermetic, no model)

- Wiring: the round's judge node references `adequacy-judge`; the prompt
  lives in exactly one file.
- Harness mechanics: fixture loading, contract composition, verdict
  parsing, summary math.
- Fixture sanity: each fixture's code/tests execute as labeled
  (adequate-clear passes, failing-but-adequate fails, etc.).

## Out of scope

The fuller ADR-048 adversarial harness (artifact-influence channel,
builder/judge weight correlation) stays on #84 for a later increment;
this measures the false-reject bottleneck specifically. The #98 shape is
the next increment, on the tuned judge.

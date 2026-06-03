# Cycle 7 loop-back #2 DECIDE — Spike χ: deliverable-shaping reliability (Finding D, where-fork)

**Date:** 2026-06-02
**Cost:** $0 (local qwen3:8b via `code-generator`; provider status confirmed
ollama-routed, no Zen).
**Question.** The Finding D I/O-contract has a where-fork: enforce the
client-tool deliverable form **at the bridge** (deterministic shaper on the
ensemble's natural output — keeps capability ensembles destination-agnostic) or
**at the ensemble** (bare-output contract reaches the model — couples the
ensemble to file-production). The bridge-side open risk is *deterministic-
extraction reliability*. This spike grounds that risk on real cheap-model output
before the ADR picks a direction (practitioner directive: ground the fork
first).

## Method

Four `code-generator` runs on qwen3:8b:
- **Run A (unconstrained, ×3):** natural code-generation prompts (palindrome,
  Stack class) + the captured fibonacci output from WP-LB-G. Apply a deterministic
  shaper (extract fenced code blocks) and score extraction ambiguity.
- **Run B (bare-output instruction, ×1):** the same kind of task with an explicit
  "output ONLY the raw file bytes, no fences, no prose" instruction. Score whether
  the cheap model emits bare content.

Shaper-analysis script + captured outputs: `scratch/spike-chi-deliverable-shaping/`.

## Results

| Sample | Fenced blocks | Shaper verdict |
|---|---|---|
| fibonacci (captured, synthesizer) | 2 | **ambiguous** — impl + "Example Usage" |
| χ-P1 palindrome (coder; synthesizer timed out) | 2 | **ambiguous** — impl + "Example Usage" |
| χ-P2 Stack (synthesizer) | 1 | unambiguous — extract the one fence |
| χ-P3 bare-output instruction (synthesizer) | 0 | already bare — `def celsius_to_fahrenheit…`, directly writable |

**Run B output (verbatim):**
```
def celsius_to_fahrenheit(c):
    """Convert Celsius to Fahrenheit."""
    return (c * 9/5) + 32
```
No fences, no prose, no example block. Perfect bare file content.

## Findings

**F-χ.1 — the bridge-side deterministic shaper is fragile.** Multi-fence
ambiguity appears in **2 of 3** unconstrained outputs: code-generator's default
habit is implementation + a separate "Example Usage" fence. A single-fence shaper
fails on those; a first/largest-fence heuristic rescues these specific cases but
is fragile by construction — it misfires on test files (the test code may not be
largest/first), multi-file deliverables (multiple legitimate fences), and prose
deliverables (no fence at all). Deterministic extraction from unconstrained
markdown has no robust general rule.

**F-χ.2 — the ensemble-side bare-output contract is reliable.** χ-P3: qwen3:8b
complied perfectly with an explicit bare-output instruction, emitting exactly the
file bytes. Corroborates Spike φ Run 2 (claim-extractor complied with its format
contract once it reached the prompt). The cheap model reliably *produces* the
right form when instructed — getting a contract to the model works; shaping
unconstrained output after the fact does not.

**F-χ.3 — the synthesizer can fail (χ-P1, 300s timeout).** Even a good contract
does not prevent dispatch failure. Two consequences: (a) the Client-Tool-Action
Terminal must degrade on dispatch failure (it already does — WP-LB-C); (b) the
**D1 extraction fix must fall back to the last *successful* agent**, not blindly
"the last agent" — a failed terminal node has `response: null`.

**F-χ.4 — the evidence resolves the where-fork toward a synthesis.** Neither raw
pole is right: bridge-side deterministic shaping is fragile (F-χ.1), and statically
coupling the ensemble YAML to file-production erodes the destination-agnostic
reusability that ADR-025 capability ensembles depend on. The grounded synthesis:
**deliver a destination-keyed bare-output contract *to the model* at the
marshalling boundary.** The loop-driver / terminal, which knows the deliverable's
destination client tool at dispatch time (callee `invoke_ensemble`, ADR-033/034),
composes a form-directive for that tool (`write` → bare file bytes; `bash` → bare
command) and includes it in the dispatch instruction. This:
- gets the contract to the model (reliable per F-χ.2 / φ Run 2) rather than
  trusting fragile post-hoc extraction (F-χ.1);
- keeps the ensemble YAML **destination-agnostic** — the directive is injected
  per-dispatch by the framework, not baked into the ensemble (preserves
  reusability);
- is structurally consistent with the cycle's thesis (the *framework* guarantees
  the contract is present; cf. AS-9 / ADR-033 single-step enforcer) while being
  lighter than `output_schema` reject-and-retry.

It also reframes D2a: the contract-delivery mechanism is **not** "wire the
ensemble's generic `default_task` through" (that is destination-blind) — it is
"the marshalling boundary composes a destination-appropriate output directive."

## Spike χ.2 — bare-output compliance breadth (practitioner directive: ground the n=2 caveat)

Four more $0 local runs across harder deliverable types:

| Deliverable type | Run | Bare-output compliance |
|---|---|---|
| Single function | φ-P3 temperature | ✓ clean |
| Larger single module (imports + dataclass + class + 3 methods) | χ-P4 inventory | ✓ clean — directly writable, no fences/notes |
| `bash` command (destination = `bash` tool) | χ-P5 | ✓ clean — `find . -type f -name "*.py" -mtime -1` |
| Structured prose (claim bullets) | φ Run 2 | ✓ clean |
| **Multi-file in one dispatch** | χ-P6 | ✗ **breaks** |

**F-χ.5 — the bare-output contract is reliable at single-deliverable granularity,
and multi-file-in-one-dispatch is the wrong granularity, not a contract gap.**
Single-deliverable compliance is now n=4 across varied types (function, larger
module, bash command, structured prose) — all clean. The one break, χ-P6 (two
files asked of one dispatch), produced an ad-hoc `filename\ncontent` convention a
`write` cannot consume, and the synthesizer re-added prose ("**Critic's Notes:**").
The architecture already answers this: **one dispatch → one client-tool
deliverable**; multi-file is the loop-driver's *across-turn* decomposition (one
`write` per turn), which is callee-native across-turn composition (ADR-033 F3-1).
χ-P6 is what violating that granularity looks like. So the ADR states the
granularity invariant explicitly rather than reaching for a structured multi-file
contract.

This narrows the carried caveat: single-deliverable bare-output is well-grounded
(n=4); the remaining PLAY/first-deployment target is narrower — sustained
compliance over long multi-turn trajectories (axis-2), and the granularity
invariant holding under a real loop-driver that must decompose multi-file work
across turns (not cram it into one dispatch).

## Disposition (for the ADR)

- **Primary direction: contract-to-model at the marshalling boundary** (F-χ.4).
  The bridge composes a destination-keyed form-directive and injects it into the
  dispatch; the ensemble stays destination-agnostic.
- **Bridge-side deterministic shaping is rejected as the primary mechanism**
  (fragile, F-χ.1). It survives only as an optional defense-in-depth backstop
  (e.g., strip an outer fence if one slips through) if PLAY shows residual
  non-compliance — not as the contract itself.
- **D1 extraction fix** lands as BUILD, shaped to the above, and must fall back to
  the last successful agent (F-χ.3).

**Caveats carried to the ADR / PLAY (narrowed by χ.2).** Single-deliverable
bare-output compliance is now n=4 across varied types (function, larger module,
bash command, structured prose) — well-grounded, not n=2. The granularity
invariant (one dispatch → one deliverable; multi-file across turns) is the ADR's
answer to the χ-P6 break. The remaining PLAY/first-deployment targets are
narrower: (a) sustained compliance over long multi-turn trajectories (axis-2,
ADR-033 §6b); (b) the granularity invariant holding under a real loop-driver
decomposing multi-file work across turns; (c) escalated-tier behavior. The
synthesizer-timeout case (F-χ.3) is a latency/reliability observation for the
local cheap tier, separate from form compliance.

## Artifacts retained

`scratch/spike-chi-deliverable-shaping/` — shaper-analysis script + captured
outputs. Retained until corpus close (spike-artifact-retention discipline).

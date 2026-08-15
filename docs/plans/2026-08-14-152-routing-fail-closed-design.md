# #152 — Failed routing node fails closed (design)

Status: pre-flight. Issue: #152. Capture: the #144 gate misfire
(`docs/plans/2026-08-13-144-live-gate/README.md`,
`discarded-gate-classify-badpath.jsonl`).

## Mechanism (grounded)

A serving script node that crashes does NOT fail at the engine level:
`script_agent.py:161-168` catches the nonzero exit and RETURNS
`{"success": false, "error": "Script failed with exit code N", ...}` as a
normal response, so the ensemble records the node `status: success` and
`dependency_resolver._extract_dependency_results_as_dict` passes the
failure envelope to dependents as a readable dep.

shape.py then picks it (`deps.get("resolve") or deps.get("classify")`),
parses a non-empty dict with no `kind`/`build` keys, and the build
default `decision.get("build", decision.get("kind") != "explanation")`
lands True → build=True, file="solution.py", content="" (the seat is
dead too), `ast.parse("")` passes form_gate → emit ships a junk empty
write. Observed live 2026-08-13 (bad-PATH serve: resolve.py imports
`llm_orc` — `shape_catalog`, resolve.py:21 — so bare python3 kills it;
classify/shape/form_gate/emit are stdlib-only and kept running.
Pre-flight correction: seat_contract.py also imports `llm_orc` and
third-party `yaml` (seat_contract.py:25-29), so the bad-PATH replay
kills it too — harmless for the cascade, since a dead seat_contract
yields `seat_admitted=None` via shape.py:63-69 and no gate fires, but
the live-gate record must expect both crashes).

A second path to the same invariant violation (pre-flight finding 1,
the laundering hole): a crashed CLASSIFY with a healthy resolve.
resolve.py:107-110 json-parses classify's failure envelope with no
readability check → `needs_decider` absent → else branch emits
`target: ""` with build=False (resolve.py:164-166) — the `target` KEY is
present, so a key-presence gate passes it — the seat dispatches on `""`
and fails deterministically, and the wire ships a silent empty prose
finish. The same `target: ""` state pre-exists via an out-of-set decider
target (resolve.py:100-101). Both are routing failures and must refuse.

## Invariant

A turn with no READABLE routing decision must fail CLOSED: no build, no
prose passthrough, an honest deterministic refusal on the wire.

Readable = parses to a dict with a NON-EMPTY `target` AND at least one
of `build`/`kind` present. Both producers always emit all three from
single output sites (resolve.py:197,198,201; classify.py:2391,2396 —
one `print` each, no early exits; classify.py's module docstring names
the contract set). Non-emptiness closes the laundering hole above — no
legitimate live state reaches shape with an empty target (classify
leaves it empty only on defer, and resolve always overwrites on the
defer path except out-of-set, which is itself a routing failure). The
build/kind presence requirement (pre-flight finding 3) keeps the
retained build default from ever firing on a decision that carried
`target` but lost the rest — without it, positive completeness would
rest on "no failure shape can carry stdout keys" instead of on the
gate. This is the #153 positive-completeness doctrine applied to
routing: an unknown future failure shape fails closed instead of
sailing past an error-shape denylist.

## Deviation from the issue's pinned direction (pre-flight question)

The issue says "fall back to classify, else refuse". Two facts found in
grounding change that:

1. classify is NOT in shape's runtime deps (`serving.yaml:131` wires
   `[resolve, seat, seat_contract]`). The classify branch in shape.py is
   the unit-test-harness / backward-compat path; it cannot fire live.
2. A live classify fallback would be UNSAFE for content-bearing routes:
   when resolve is unreadable the seat is poisoned too (it dispatched on
   `${resolve.target}` with resolve-supplied `dispatch_input`), so a
   classify-routed build decision + dead-seat empty content passes
   form_gate (`ast.parse("")` succeeds) and recreates the junk write
   through a legitimate-looking decision. Only content-free seam routes
   (needs_*/recall_answer/not_grounded) would be safe, and carving that
   subset into shape duplicates emit's seam priority — drift risk for a
   fallback whose only demonstrated trigger is operator
   misconfiguration, which a loud refusal surfaces on turn 1 instead of
   hiding behind degraded service (fail-loud, protocol rule 6).

Direction: keep the source-preference chain readable(resolve) else
readable(classify) — harmless live, keeps the test harness and any
pre-resolve ensemble wiring valid — and REFUSE when neither is
readable. Do NOT add classify to shape's depends_on.

## Change

1. `shape.py`: replace the truthiness fallback with readability-gated
   selection; when no readable decision, emit `build=False` plus a new
   `routing_failed` reason field (deterministic text; includes the
   failure envelope's one-line `error` when present).
2. `form_gate.py`: pass `routing_failed` through (one line, matching the
   existing named-field style).
3. `emit.py`: `routing_failed` refuses FIRST in `_seam_outcome`, with
   the existing non-minting `TERMINALS["refused"]` prefix — an
   unreadable decision makes `is_build_ask` unknowable, so never mint a
   build-scoped ledger entry (under-report, never misreport; the
   documented safe direction). No new terminal, no caller change: the
   caller already recognizes "Refused: " as a non-minting prose finish.

Wire text (draft): `Refused: serving pipeline error: no readable
routing decision this turn (<error>); nothing was built or written.`
The parenthetical carries the failure envelope's one-line `error` field
when the unreadable dep parsed to a dict carrying one, else it is
omitted.

## Regression instruments

- shape: failure envelope (the captured `{"success": false, "error":
  "Script failed with exit code 1", "stderr": ...}` shape) as the
  resolve dep, readable classify absent → `build is False` AND
  `routing_failed` set. Mutant kill: reverting the readability gate
  flips build to True.
- shape: `target: ""` decision (the laundered classify-crash /
  out-of-set-decider state) → refusal. Pins key-presence vs.
  non-emptiness explicitly (pre-flight finding 4 — the exact line an
  implementer drifts on).
- shape: decision with non-empty `target` but neither `build` nor
  `kind` → refusal (the build default must never fire past the gate).
- shape: failure-envelope resolve + readable classify → classify's
  decision routes (the harness/back-compat path stays alive).
- resolve: classify failure envelope in deps → resolve emits
  `target: ""` (documents the laundering handoff shape refuses on).
- shape: existing target-less test decisions updated to carry `target`
  (they now mirror the real wire; both producers always emit it). The
  existing `test_unreadable_routing_decision_fails_closed_to_prose`
  strengthens from prose-passthrough to refusal.
- form_gate: `routing_failed` passthrough.
- emit: `routing_failed` → `{"finish": true}` with the plain
  `Refused: ` prefix; asserts no `file` key (never a write).
- endpoint: full pipeline with resolve made to crash (exit 1) → the
  response is the refusal, and no write reaches the client.
- endpoint: full pipeline with classify made to crash (exit 1) → the
  refusal, never the silent empty finish (the laundering hole pinned
  end to end).

## Live gate (exit)

Replay the demonstrated capture conditions against the fix: restart the
serve WITHOUT the venv PATH prefix (bare python3 → resolve AND
seat_contract crash; the latter is inert, `seat_admitted=None`), issue
the gate-2 ask from the #144 record via `opencode run`, observe the
honest refusal on the wire and NO file write; restart the good serve;
13-turn ladder battery stays 13/13, 0 dishonest. Run record under
`docs/plans/2026-08-14-152-live-gate/`.

## Known bounds

- Degraded-service tax: a turn classify could have routed content-free
  (e.g. a recall answer) now refuses when resolve crashes. Accepted:
  the only demonstrated trigger is a misconfigured serve, and the
  refusal surfaces it immediately.
- The `target` discriminator couples shape to the producers' contract
  key. Drift direction is safe-loud: a producer that stops emitting
  `target` makes every turn refuse, which the endpoint suite and any
  live turn catch instantly.
- Behavior change beyond the crash paths: the out-of-set-decider state
  (resolve.py:100-101, `target: ""`) previously shipped a silent empty
  prose finish through a deterministic dispatch failure; it now refuses
  honestly. That state is a routing failure by resolve's own docstring
  ("empty/unknown still fails deterministically") — the refusal is the
  honest rendering of what already happened.

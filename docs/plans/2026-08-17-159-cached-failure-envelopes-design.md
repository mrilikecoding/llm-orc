# #159 — failure envelopes stop being cached (design)

Status: pre-flight. Issue: #159, found during the #157 review and made
more reachable by #157 (timeouts now actually fire) and #158 (identical
concurrent agents no longer collapse into one execution).

## Mechanism (reproduced)

`ScriptAgentRunner.execute` caches EVERY result
(`agent_runner.py:95-101`), and `ScriptCacheConfig` defaults to
`enabled=True, ttl_seconds=3600`. `ScriptCache.get` returns the entry's
`"output"` verbatim, so a failure is replayed for an hour on the same
`(script, input, parameters)` key:

```
cache defaults -> enabled: True ttl: 3600
run1: 1.01s -> {"success": false, "error": "Script timed out after 1 seconds"}
run2: 0.00s -> {"success": false, "error": "Script timed out after 1 seconds"}
stats: {'hits': 1, 'misses': 1, ...}
```

The concrete worry is a transient failure: one rate-limited
`web_searcher` call, or one timeout under momentary load, poisons that
key for an hour. This repo sets `script_cache.enabled: false`, so it is
not live here — which also means dogfooding will never surface it.

**And the worst case is cross-PROCESS, not one in-process hour.** With
`persist_to_artifacts: true` the cache writes to `./.llm-orc/cache`
(`cache.py:200,231`; every `ArtifactManager()` defaults to `Path(".")`),
so a cached failure survives a restart for the full TTL and is shared
between a parent executor and its children. That is where this fix earns
the most.

The entry also carries a hardcoded `"success": True`
(`agent_runner.py:100`) that nothing reads: `agent_runner.py:83-87` is
the cache's only consumer and takes only `"output"`. A field that always
says success, on an entry that may hold a failure, is worth deleting on
its own.

## Change

Do not cache a response the script reports as unsuccessful; drop the
dead `"success": True` from the entry.

"Unsuccessful" is TWO clauses, because one is not enough:

```python
if isinstance(parsed, dict) and (
    not parsed.get("success", True) or parsed.get("error")
):
    skip caching
```

- **Truthiness with a `True` default**, not `is False`: catches
  `{"success": 0}` and `{"success": null}`, while a bare
  `not parsed.get("success")` would stop caching EVERYTHING that omits
  the key.
- **The `error` clause is what catches the issue's own motivating
  example.** `web_searcher` emits `{"error": "rate_limited", ...}` on
  every failure path and returns 0, so no exception envelope fires and
  no `success` key exists. Pre-flight measured that a `success`-only
  predicate leaves that key poisoned exactly as today. More broadly,
  **0 of 33 scripts in `.llm-orc/scripts/agentic_serving/` emit a
  boolean `success`** — the five that mention success use ADR-024's
  `"status": "success"` — so a `success`-only rule would catch nothing
  in the repo's largest live corpus beyond the framework's own exception
  envelopes.

**The response is not always a `str`.** `_parse_output` returns
`json.loads(output)` verbatim, and `execute` returns it unwrapped when
it is not a dict, so a script printing `[1,2,3]`, `null`, `42`, or
`true` yields a `list`/`None`/`int`/`bool`. Calling `json.loads` on
those raises `TypeError` and would kill an agent run that works today,
so the predicate guards `isinstance(response, str)` first.

## The judgement call: which failures

The rule keys on the envelope, so it cannot distinguish an INSTRUMENT
failure (timeout, nonzero exit, resolution error) from a DOMAIN one (a
validation script legitimately reporting `{"success": false, "reason":
...}`). Both stop being cached.

That asymmetry is deliberate. Caching a transient failure is a
CORRECTNESS problem for an hour; re-running a deterministic domain
failure is a performance cost on a script that already ran in
milliseconds. Erring toward re-execution is the safe direction, and the
alternative — marking instrument failures at the source so the runner
can tell them apart — changes the envelope shape, which is wire-visible
and read by the serve's ask-outcome ledger.

Responses that do not parse as JSON, or that carry no `success` key, are
cached as they are today. `_parse_output` wraps prose and empty output
as `{"success": true, ...}`, so the common paths are unaffected.

## Invariant

A result the script itself reports as unsuccessful is never served from
cache.

## Regression instruments

Assert on `ScriptCache.get_stats()` (`cache.py:170-184`), never on wall
time: it reports `hits`/`misses`/`sets` exactly, cannot flake, and
distinguishes "did not cache" from "cached but the read was slow" —
which a timing assertion cannot.

1. **The reproduction, inverted**: two runs of a failing script agent
   with the cache ENABLED end at `hits == 0, sets == 0`. Today they end
   at `hits=1, misses=1, sets=1`. Uses `sys.exit(3)` rather than a real
   timeout: instant, same predicate branch.
2. **Successes still cache**: `hits == 1, sets == 1`.
3. **No-`success`-key and prose responses still cache.** This is the pin
   that matters most, and pin 2 CANNOT do its job: an implementer
   writing `not parsed.get("success")` without the `True` default stops
   caching every response lacking the key — all 33 serving scripts,
   every prose response, every schema-path result — and pin 2 still
   passes because its script emits `{"success": true}`. That is exactly
   the "never cache anything" degradation pin 2 was written to prevent.
4. **A JSON-array response runs twice without raising**, covering the
   non-`str` crash path above.
5. **The entry carries no lying success field**, asserted on the cached
   dict's shape.

## Known bounds

- A deterministic failure re-executes every time. Every shipped producer
  of `success: false` is cheap (json_extract's decode error,
  read_file/write_file's one filesystem call, the interactive
  primitives, two pure-compute library scripts). But the bound a reader
  actually cares about is what re-executes now, which the ERROR clause
  dominates — and `web_searcher.py:90` is a network script whose every
  failure path is `{"error": ..., "backend": ...}`. A repeated failing
  search on an identical query within one executor now makes N API calls
  instead of 1. Small in practice, and the alternative is serving a
  stale rate-limit for an hour.
- **ADR-024 `status` envelopes are invisible to the predicate**, which
  never looks at `status`. Correct today only by accident: every
  envelope builder in the repo hardcodes `"status": "success"`, and the
  only `"status": "error"` producer is a service-layer response that
  never enters the script cache. A pin for it would pin the accident.
- **The `error` clause assumes `error` means "a failure message"**,
  which is convention rather than contract. A success carrying a truthy
  non-string `error` — a count, a findings list, a standard-error float
  in a stats script — stops being cached. Nothing shipped does this;
  the cost if something did is performance, never correctness.
- `{"success": "false"}` as a STRING escapes the predicate — a shell
  script doing `echo "{\"success\": \"$ok\"}"` produces it. Low
  probability, not worth machinery.
- A nested envelope (`{"success": true, "data": {"success": false}}`)
  is cached as a success. No shipped or repo-local script produces one.
- **This is a mitigation, not the fix for the subsystem.** #160 records
  the deeper hazard found in the same review: the cache key hashes the
  script PATH rather than its bytes, so editing a script serves the
  pre-edit result for the TTL — a stale SUCCESS, which no
  failure-skipping predicate can catch. That issue carries the
  `enabled: True` default question too.
- This does not add in-flight deduplication, so N identical concurrent
  agents still run N times (#158's recorded bound). Separate machinery,
  separate change.
- Nothing in this repo exercises the cache, since `script_cache.enabled`
  is false locally. The pins are the only coverage, which is an argument
  for them being end-to-end through the runner rather than unit-level on
  the cache.

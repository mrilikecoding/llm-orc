# Design: embeddings on the serve

Written 2026-09-16 after the vault skills found `/v1/embeddings` answering
405 on the ng-mini serve. Approved shape: the router serves embeddings
natively, so llm-orc renders an embedding seat into the preset and
forwards the OpenAI-shaped call. No Ollama; no new dependency.

## Spike (done, on the mini)

A router on a spare port with the preset section below loaded
`nomic-ai/nomic-embed-text-v1.5-GGUF:Q8_0` and answered
`POST /v1/embeddings` for two inputs with 768-dim vectors (llama-server
b10964, router mode, `--models-max 2`).

    [nomic-embed-text]
    hf-repo = nomic-ai/nomic-embed-text-v1.5-GGUF:Q8_0
    embeddings = true
    pooling = mean
    c = 8192

## Shape

1. **Preset renderer** (`providers/llama_server.py`, `render_preset` and
   `_collect`): besides `options.num_ctx` -> `c`, pass through an
   allowlist of per-model options from a profile's `options`:
   `embeddings` (bool -> `true`/`false`) and `pooling` (one of
   `none, mean, cls, last, rank`). Anything else in `options` is ignored
   as today. Allowlist, not passthrough: an unknown key must never reach
   the router, which exits at startup on an unrecognized preset option
   and takes every seat down with it.
2. **Profile** in the repo's `.llm-orc/profiles/local-nomic-embed-text.yaml`:
   `provider: llama-server`, `model: nomic-embed-text`,
   `hf_repo: nomic-ai/nomic-embed-text-v1.5-GGUF:Q8_0`,
   `options: {embeddings: true, pooling: mean, num_ctx: 8192}`.
3. **Route** `POST /v1/embeddings` (new `web/api/v1_embeddings.py`,
   mounted like `v1_models`): accept the OpenAI body (`model`, `input`
   as string or list of strings, optional `encoding_format`,
   `dimensions` passed through if present), forward it unchanged to
   `<router>/v1/embeddings` (router URL from the same `LLAMA_SERVER_URL`
   resolution the factory uses), return the router's JSON and status.
   `model` may be a llama-server profile id (`local-nomic-embed-text`,
   resolved through the profile to its `model`) or a router model name
   (`nomic-embed-text`); the forwarded body carries the router name.
   404 with an OpenAI-style error body when it resolves to nothing that
   is a rendered preset model (read the rendered preset's model list the
   same way `/api/models` does). Router unreachable -> 503, same
   wording as `/api/models/{name}/pull`. No streaming.
4. **`/v1/models`** unchanged in code: it lists the operator allowlist
   from `.llm-orc/config.yaml`; adding `local-nomic-embed-text` there on
   the mini makes the embedding model appear for the vault skills'
   pre-run check. `GET /api/models` already reports its load state.
5. **Deploy**: the mini's plist gains `--models-max 2` so the embedding
   seat stays resident beside the chat seat (about 150 MB). The profile
   reaches the mini by `git pull` in the checkout (#196).

## Tests (hermetic)

- `render_preset`: a profile with `options.embeddings: true, pooling:
  mean` emits both lines; a profile without emits neither; an unknown
  option key is ignored; `pooling: bogus` is dropped (or rejected,
  pick one and pin it).
- `/v1/embeddings`: forwards body and returns the stub router's JSON;
  a profile id resolves to the router model name in the forwarded body;
  unknown model -> 404 without touching the router (assert the stub
  saw no request); router down -> 503; string `input` and list `input`
  both forwarded verbatim. Use the existing stub-router pattern from
  the `/api/models` tests; the autouse guard forbids the real binary.
- Existing `tests/unit/web` and `tests/unit/providers` keep passing.

## Acceptance on the rig

Through `https://llm-orc.homelab.nate.green/v1/embeddings`: two inputs
with nomic prefixes return two 768-dim vectors; `/api/models` shows
`nomic-embed-text` `loaded` while the chat seat stays `loaded`
(models-max 2); a chat call afterwards does not reload the chat seat.

## Alignment with the vault skills (svalbard session, 2026-09-16)

Callers are script agents inside `vault-` ensembles on the mini, hitting
the serve on loopback, batches of 32 to 64, whole-note chunks, batch and
overnight only; Plexus invokes those ensembles over REST execute; the
Claude skills only manage ensembles over MCP. No vector store here:
Plexus is the record. nomic-embed-text-v1.5 at 768 dims is accepted.

## Out of scope

A vector store. Batching or chunking policy (caller's). Listing
embedding models under `/v1/models`. Reranking (`pooling: rank`)
beyond letting the option through.

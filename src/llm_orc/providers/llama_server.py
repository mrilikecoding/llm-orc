"""llama-server router ownership (#90).

llm-orc owns the local inference process: it renders a llama-server
*preset* from its own model profiles (one section per distinct model,
each naming its GGUF source on Hugging Face) and supervises one router
process that lazy-loads models by name. The preset is the contract
between the profile set and the inference process; nothing else in the
project knows how a model gets onto the box.
"""

import json
import signal
import subprocess
import tempfile
import time
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType
from typing import IO, Any

LLAMA_SERVER_PROVIDER = "llama-server"

#: Global section every model instance inherits. Metal takes every layer
#: on Apple unified memory; jinja is what parses tool calls; flash-attn
#: buys KV headroom (research scoping doc, llama-server backend). The
#: context size is the window the serve's truncation backstop assumes
#: (turn_trace.WINDOW); a profile's ``options.num_ctx`` overrides it per
#: model. Context size is now project-controlled, not inherited from a
#: desktop app's environment (#90 eval, risk 7).
PRESET_DEFAULTS: dict[str, str] = {
    "c": "40960",
    "flash-attn": "on",
    "jinja": "true",
    "n-gpu-layers": "999",
}

#: Resident models at once. One is memory-safe on the 32 GB target rig
#: at the default window (an 8b at 40960 is ~11 GB resident); the ladder
#: never swapped models (#90 eval, 1e), so the escalation stall is rare.
DEFAULT_MODELS_MAX = 1


@dataclass(frozen=True)
class RenderedPreset:
    """The preset text plus what it covers and what it had to leave out."""

    text: str
    models: list[str] = field(default_factory=list)
    missing_source: list[str] = field(default_factory=list)


def _served_model(profile: Mapping[str, Any]) -> str | None:
    """The model name a llama-server profile routes to, or None when the
    profile is for another provider. A colon is refused: the router
    rewrites ``name:tag`` as a Hugging Face tag (spike 2026-09-16)."""
    if profile.get("provider") != LLAMA_SERVER_PROVIDER:
        return None
    model = str(profile.get("model") or "")
    if not model:
        return None
    if ":" in model:
        raise ValueError(
            f"llama-server model name {model!r} contains a colon; the router "
            "rewrites 'name:tag' as a Hugging Face tag (use 'name-tag')"
        )
    if "/" in model:
        raise ValueError(
            f"llama-server model name {model!r} contains a slash; the router "
            "lists raw cache entries as 'user/repo:tag' and the client hides them"
        )
    return model


def _record_scalar(
    store: dict[str, Any], model: str, value: Any, *, label: str
) -> None:
    """One value per model name for a given option; two different ones
    for the same model is a configuration error."""
    if model in store and store[model] != value:
        raise ValueError(
            f"model {model!r} has conflicting {label}: {store[model]!r} and {value!r}"
        )
    store[model] = value


def _record_source(sources: dict[str, str], model: str, repo: Any) -> None:
    """One source per model name; two different ones is a config error."""
    if not repo:
        return
    _record_scalar(sources, model, str(repo), label="hf_repo sources")


#: Allowlisted per-model ``options`` passed through to the preset besides
#: ``num_ctx``. Allowlist, not passthrough: an unknown key must never reach
#: the router, which exits at startup on an unrecognized preset option and
#: takes every seat down with it (docs/plans/2026-09-16-embeddings-on-the-
#: serve.md). ``pooling`` is further restricted to the router's own value
#: set; anything else is dropped rather than rejected, the same treatment
#: an unlisted option key gets.
_POOLING_VALUES = {"none", "mean", "cls", "last", "rank"}


def _collect(
    profiles: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, str], dict[str, int], dict[str, bool], dict[str, str], set[str]]:
    """Sources, context sizes, and embedding options per llama-server model
    name, plus every model name seen (so the sourceless ones can be
    reported)."""
    sources: dict[str, str] = {}
    contexts: dict[str, int] = {}
    embeddings: dict[str, bool] = {}
    pooling: dict[str, str] = {}
    seen: set[str] = set()
    for profile in profiles.values():
        model = _served_model(profile)
        if model is None:
            continue
        seen.add(model)
        _record_source(sources, model, profile.get("hf_repo"))
        options = profile.get("options") or {}
        num_ctx = options.get("num_ctx")
        if isinstance(num_ctx, int):
            contexts[model] = max(num_ctx, contexts.get(model, 0))
        embeddings_flag = options.get("embeddings")
        if isinstance(embeddings_flag, bool):
            _record_scalar(
                embeddings, model, embeddings_flag, label="embeddings option"
            )
        pooling_value = options.get("pooling")
        if pooling_value in _POOLING_VALUES:
            _record_scalar(pooling, model, pooling_value, label="pooling option")
    return sources, contexts, embeddings, pooling, seen


def render_preset(
    profiles: Mapping[str, Mapping[str, Any]],
    *,
    defaults: Mapping[str, str] | None = None,
) -> RenderedPreset:
    """Render the router preset for every llama-server profile.

    Sections are keyed by model name (the router routes on the request's
    ``model`` field). A model needs a source (``hf_repo``) to be loadable;
    one without is reported in ``missing_source`` rather than emitted as
    an invalid section. Two profiles naming one model with different
    sources -- or different ``embeddings``/``pooling`` options -- is a
    configuration error, not something to pick between.
    """
    sources, contexts, embeddings, pooling, seen = _collect(profiles)
    models = sorted(sources)
    missing = sorted(seen - set(sources))

    lines = ["version = 1", "", "[*]"]
    for key, value in sorted((defaults or PRESET_DEFAULTS).items()):
        lines.append(f"{key} = {value}")
    for model in models:
        lines += ["", f"[{model}]"]
        if model in contexts:
            lines.append(f"c = {contexts[model]}")
        if model in embeddings:
            lines.append(f"embeddings = {'true' if embeddings[model] else 'false'}")
        if model in pooling:
            lines.append(f"pooling = {pooling[model]}")
        lines.append(f"hf-repo = {sources[model]}")
    return RenderedPreset("\n".join(lines) + "\n", models, missing)


def _is_preset_model(model: Mapping[str, Any]) -> bool:
    model_id = str(model.get("id", ""))
    return model_id != "default" and "/" not in model_id


#: The router is reached directly, never through an HTTP proxy from the
#: environment: a proxied loopback request fails or times out (CI runners
#: carry proxy variables; reproduced locally with ``http_proxy`` set).
_DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class LlamaServerClient:
    """The router's management (``/models``) and OpenAI-compatible
    (``/v1/...``) surface, whether the router is one this process owns
    or one reached by URL."""

    def __init__(self, root_url: str) -> None:
        self.root_url = root_url.rstrip("/")

    @classmethod
    def from_base_url(cls, base_url: str) -> "LlamaServerClient":
        """From the OpenAI-compatible base URL the factory uses
        (``.../v1``) to the router root it hangs off."""
        root = base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[: -len("/v1")]
        return cls(root)

    def models(self) -> list[dict[str, Any]]:
        """The router's model list with per-model load status."""
        with _DIRECT.open(f"{self.root_url}/models", timeout=5) as resp:
            data = json.load(resp)
        models = data.get("data", [])
        # Router mode lists a ``default`` entry for its own command line
        # and one raw ``user/repo:tag`` entry per cached Hugging Face file
        # (e2e 2026-09-16); neither is a preset model, and this build
        # rejects ``dedup-cache-models``, so they are hidden here.
        return [m for m in models if isinstance(m, dict) and _is_preset_model(m)]

    def load(self, model: str) -> None:
        """Ask the router to load (downloading if needed) one model."""
        body = json.dumps({"model": model}).encode()
        request = urllib.request.Request(
            f"{self.root_url}/models/load",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with _DIRECT.open(request, timeout=3600) as resp:
            json.load(resp)

    def embeddings(self, body: Mapping[str, Any], *, timeout: float) -> tuple[int, Any]:
        """Forward an OpenAI-compatible embeddings request to the router,
        returning its status code and JSON body as-is.

        The router's own error responses (a 4xx/5xx with a JSON body) are
        returned rather than raised; only a connection failure (the router
        unreachable) propagates as ``OSError``, matching ``models``/``load``.
        """
        data = json.dumps(dict(body)).encode()
        request = urllib.request.Request(
            f"{self.root_url}/v1/embeddings",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with _DIRECT.open(request, timeout=timeout) as resp:
                return resp.status, json.load(resp)
        except urllib.request.HTTPError as e:
            return e.code, json.load(e)


class LlamaServerSupervisor:
    """One router process, owned for the life of the serve.

    ``start`` spawns the router and waits until ``GET /models`` answers,
    so callers never race a half-started backend; a router that exits
    before listening (bad preset, missing binary) surfaces as an error
    with its exit code rather than as a later connection refusal.
    """

    def __init__(
        self,
        *,
        preset_path: Path,
        binary: str | Path = "llama-server",
        host: str = "127.0.0.1",
        port: int = 8080,
        models_max: int = DEFAULT_MODELS_MAX,
    ) -> None:
        self.preset_path = Path(preset_path)
        self.binary = str(binary)
        self.host = host
        self.port = port
        self.models_max = models_max
        self._process: subprocess.Popen[bytes] | None = None
        self._stderr: IO[bytes] | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    @property
    def client(self) -> "LlamaServerClient":
        return LlamaServerClient(f"http://{self.host}:{self.port}")

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def command(self) -> list[str]:
        return [
            self.binary,
            "--models-preset",
            str(self.preset_path),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--models-max",
            str(self.models_max),
            "--no-webui",
        ]

    def start(self, *, timeout_s: float = 60.0) -> None:
        """Spawn the router and block until it answers ``/models``."""
        self._stderr = tempfile.TemporaryFile()
        self._process = subprocess.Popen(  # noqa: S603 - argv built here
            self.command(), stdout=subprocess.DEVNULL, stderr=self._stderr
        )
        deadline = time.monotonic() + timeout_s
        last_error = ""
        while time.monotonic() < deadline:
            code = self._process.poll()
            if code is not None:
                self._process = None
                raise RuntimeError(
                    f"llama-server exited with code {code} before listening "
                    f"(command: {' '.join(self.command())})\n"
                    f"last stderr:\n{self._stderr_tail()}"
                )
            try:
                self.models()
                return
            except (OSError, ValueError) as e:
                last_error = f"{type(e).__name__}: {e}"
                time.sleep(0.1)
        tail = self._stderr_tail()
        self.stop()
        raise RuntimeError(
            f"llama-server not ready after {timeout_s:.0f}s "
            f"(command: {' '.join(self.command())}; last probe error: "
            f"{last_error})\nlast stderr:\n{tail}"
        )

    def _stderr_tail(self, lines: int = 8) -> str:
        if self._stderr is None:
            return ""
        self._stderr.seek(0)
        text = self._stderr.read().decode(errors="replace")
        return "\n".join(text.strip().splitlines()[-lines:])

    def stop(self) -> None:
        if self._process is None:
            return
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        self._process = None

    def models(self) -> list[dict[str, Any]]:
        """The router's model list with per-model load status."""
        return self.client.models()

    def load(self, model: str) -> None:
        """Ask the router to load (downloading if needed) one model."""
        self.client.load(model)


PRESET_FILENAME = "llama-server.ini"


def start_router_from_config(
    config_manager: Any,
    *,
    binary: str | Path = "llama-server",
    host: str = "127.0.0.1",
    port: int = 8080,
    models_max: int = DEFAULT_MODELS_MAX,
    timeout_s: float = 60.0,
) -> LlamaServerSupervisor:
    """Render the preset from the project's profiles and start the router.

    The preset lands next to the profiles it was rendered from (the
    local ``.llm-orc`` when there is one, else the global config dir) so
    an operator can read exactly what the router was given.
    """
    profiles = config_manager.get_model_profiles()
    rendered = render_preset(profiles)
    config_dir = Path(
        config_manager.local_config_dir or config_manager.global_config_dir
    )
    config_dir.mkdir(parents=True, exist_ok=True)
    preset_path = config_dir / PRESET_FILENAME
    preset_path.write_text(rendered.text)

    supervisor = LlamaServerSupervisor(
        preset_path=preset_path,
        binary=binary,
        host=host,
        port=port,
        models_max=models_max,
    )
    supervisor.start(timeout_s=timeout_s)
    return supervisor


def install_signal_stop(
    supervisor: LlamaServerSupervisor,
) -> Callable[[int, FrameType | None], None]:
    """Stop the router when the serve is signalled to exit.

    uvicorn captures SIGTERM/SIGINT while it runs and, on exit, restores
    the handlers it found and re-raises the signal, so the process dies
    before any ``finally`` around ``uvicorn.run`` (e2e 2026-09-16: router
    alive after SIGTERM to the serve). Installing this handler first makes
    it the one the re-raise reaches; it stops the router, then lets the
    signal take its default course so the exit status stays honest.
    """

    def _stop_then_die(signum: int, _frame: FrameType | None) -> None:
        supervisor.stop()
        signal.signal(signum, signal.SIG_DFL)
        signal.raise_signal(signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _stop_then_die)
    return _stop_then_die

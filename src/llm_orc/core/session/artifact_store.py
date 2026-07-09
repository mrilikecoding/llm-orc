"""Session Artifact Store — substrate-routing for capability-ensemble deliverables.

NOT YET WIRED into the serving path: reserved for roadmap
Stage 2 (lossless session record, issue #82) -- see
docs/serving-roadmap.md (issue #95 marker).

Per ``docs/serving.md`` §Session Artifact
Store (L1, *new in Cycle 6 per ADR-025*). Owns:

* The ``.llm-orc/agentic-sessions/<session_id>/<dispatch_id>/<deliverable>.<ext>``
  path layout — session-scoped grouping makes session-level review,
  cleanup, and (future) Plexus ingestion natural.
* The ``<session_id>`` format
  (``<iso-8601-datetime>-<short-uuid>`` — chronological filesystem
  sort + same-second disambiguation).
* The typed :class:`ArtifactReference` shape consumers find at
  ``envelope.artifacts[0]`` per ADR-024 + ADR-025.
* Retention semantics enforcement — ``session`` artifacts removed at
  session close; ``durable`` artifacts persist indefinitely;
  ``ephemeral`` artifacts removed at the orchestrator's next turn.

This module is the L1-substrate. It does not depend on Session
Registry — Session Registry calls ``cleanup_session`` at session-close
per the L3 → L1 direction. It does not depend on Orchestrator Tool
Dispatch — Tool Dispatch calls ``write_deliverable`` per the L2 → L1
direction. It does depend on the filesystem and on
:class:`DispatchEventSubstrate` *conceptually* (the ``<dispatch_id>``
segment is the same value the substrate allocates), though the path-
construction code here takes ``dispatch_id`` as a parameter — the
substrate is the single source of truth for the identifier, this
module is a faithful consumer.

Cycle 6 WP-E scope. The substrate-routing pathway in Tool Dispatch
(``invoke_ensemble`` interposition step 8 per system-design.agents.md
§Orchestrator Tool Dispatch Cycle 6 amendment) writes through this
module when the dispatched ensemble's ``output_substrate == "artifact"``.
"""

from __future__ import annotations

import datetime
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from llm_orc.models.structural_errors import LlmOrcStructuralError

__all__ = [
    "ArtifactNotFoundError",
    "ArtifactReference",
    "Retention",
    "SessionArtifactStore",
    "new_session_id",
]


Retention = Literal["session", "durable", "ephemeral"]
"""Lifetime category per ADR-025 §"Retention semantics".

``session`` — cleaned up at session close (the default for
substantive-but-not-promoted deliverables).
``durable`` — persists indefinitely (operator-managed lifecycle).
``ephemeral`` — cleaned up at the orchestrator's next turn (used for
intermediate-stage deliverables that downstream stages consume
immediately).
"""


_CONTENT_TYPE_EXTENSIONS: dict[str, str] = {
    "application/python": ".py",
    "text/x-python": ".py",
    "text/markdown": ".md",
    "application/json": ".json",
    "text/plain": ".txt",
    "text/html": ".html",
    "text/css": ".css",
    "application/javascript": ".js",
    "text/x-yaml": ".yaml",
    "application/yaml": ".yaml",
}
"""Content-type → file extension map for ``write_deliverable`` path
construction. Unknown types fall back to ``.bin`` so the path is
always deterministic — the deliverable is still on disk and downstream
consumers identify the type via :attr:`ArtifactReference.content_type`,
not by extension."""

_DEFAULT_EXTENSION = ".bin"

_ARTIFACT_ROOT_PREFIX = "agentic-sessions"
"""``ArtifactReference.path`` prefix per ADR-025 §"Artifact reference
fields" — the path is rooted at ``.llm-orc/`` (i.e., relative to the
filesystem root the operator scopes agentic-serving under), so the
envelope payload reads as ``agentic-sessions/<session_id>/...``."""


def _coerce_retention(value: str) -> Retention | None:
    """Narrow ``value`` to a :data:`Retention` literal or return ``None``.

    Used by :meth:`SessionArtifactStore._read_dispatch_retention` to
    discriminate marker contents into the typed retention space without
    spreading a chained-ternary across the caller.
    """
    if value == "session":
        return "session"
    if value == "durable":
        return "durable"
    if value == "ephemeral":
        return "ephemeral"
    return None


def new_session_id() -> str:
    """Return a fresh session identifier per ADR-025 §"Session-dir location".

    Format: ``<iso-8601-datetime>-<short-uuid>`` (e.g.,
    ``2026-05-15T14:32:08Z-a7f3``). The ISO-8601 prefix gives natural
    chronological sort on the filesystem; the four-hex-char UUID
    suffix disambiguates same-second sessions.
    """
    now = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    short_uuid = uuid.uuid4().hex[:4]
    return f"{now}-{short_uuid}"


@dataclass(frozen=True)
class ArtifactReference:
    """Typed reference to a substrate-routed deliverable.

    Per ADR-025 §"Artifact reference fields" — the shape consumers find
    at ``envelope.artifacts[0]``. Frozen so envelope consumers across
    the ReAct loop's observation surface can rely on field identity.
    """

    path: str
    """Path to the deliverable, relative to ``.llm-orc/`` (e.g.,
    ``agentic-sessions/<session_id>/<dispatch_id>/<deliverable>.<ext>``).
    Clients running on the same machine as the serve read the artifact
    from this path directly — the architecture's local-first
    commitment, per ADR-025 §"Client access to artifacts"."""

    content_type: str
    """MIME type of the deliverable (e.g., ``application/python``,
    ``text/markdown``, ``application/json``). Downstream consumers
    discriminate by this rather than by the path extension because
    unknown content-types fall back to ``.bin`` for path
    determinism."""

    size_bytes: int
    """Byte length of the deliverable on disk."""

    summary: str
    """One-line human-readable summary the operator (or calibration-
    gate critic under summary-only evaluation) sees at the envelope
    layer. The substrate produces a default summary; ensemble
    synthesizers MAY override with a more specific one."""

    retention: Retention
    """Lifetime category — drives ``cleanup_session`` /
    ``cleanup_ephemeral`` behavior per ADR-025 §"Cleanup"."""


class ArtifactNotFoundError(LlmOrcStructuralError):
    """Raised when ``read_deliverable`` cannot resolve a reference to a
    file on disk.

    The deliverable's :class:`ArtifactReference` points at a path that
    is absent under the store root — typically because the artifact's
    retention window closed (it was cleaned up) or the reference is
    stale. The Client-Tool-Action Terminal (ADR-034 / WP-LB-C) catches
    this and degrades to a dispatch-failure text completion rather than
    emit a malformed tool call (system-design.agents.md
    §Client-Tool-Action Terminal error handling).
    """

    def __init__(self, *, path: str) -> None:
        super().__init__(
            f"deliverable not found at reference path {path!r}",
            error_kind="artifact_not_found",
            recovery_action_required="operator_intervention_required",
            operator_diagnostic=(
                f"No deliverable on disk for reference path {path!r}; the "
                "artifact may have been cleaned up or the reference is stale."
            ),
        )


class SessionArtifactStore:
    """Substrate-routing store for capability-ensemble deliverables.

    Construction takes the operator-configured root directory
    (``OrchestratorConfig.observability.agentic_sessions_root`` —
    defaults to ``.llm-orc/agentic-sessions/`` per ADR-025). The store
    creates the per-session and per-dispatch directories on demand;
    operators do not pre-create them.

    The store does not allocate ``session_id`` or ``dispatch_id``
    itself — both are inputs from upstream owners (Session Registry
    for session_id; :class:`DispatchEventSubstrate` for dispatch_id).
    The single-source-of-truth correlation property the FC-22
    cross-surface check verifies is structurally preserved: the same
    ``dispatch_id`` value flows from the substrate into the artifact
    path's segment without rederivation.
    """

    def __init__(self, *, agentic_sessions_root: Path) -> None:
        self._root = Path(agentic_sessions_root)

    def write_deliverable(
        self,
        *,
        session_id: str,
        dispatch_id: str,
        deliverable_name: str,
        content: str | bytes,
        content_type: str,
        retention: Retention = "session",
        summary: str | None = None,
    ) -> ArtifactReference:
        """Write a capability-ensemble deliverable to the session-dir path.

        Creates intermediate directories as needed. ``content`` accepts
        either ``bytes`` (binary deliverables) or ``str`` (text
        deliverables; UTF-8 encoded on write). Returns the
        :class:`ArtifactReference` ready for envelope construction —
        the caller populates ``envelope.artifacts[0]`` with this value.

        ``retention`` defaults to ``session`` per ADR-025 §"Retention
        semantics" — the documented default for substrate-routed
        deliverables not flagged for explicit preservation.

        ``summary`` defaults to a substrate-produced description
        (``f"{deliverable_name}.{ext}: <size> bytes"``); ensemble
        synthesizers may pass a more specific summary they produced
        during generation.
        """
        extension = _CONTENT_TYPE_EXTENSIONS.get(content_type, _DEFAULT_EXTENSION)
        filename = f"{deliverable_name}{extension}"
        dispatch_dir = self._root / session_id / dispatch_id
        dispatch_dir.mkdir(parents=True, exist_ok=True)
        disk_path = dispatch_dir / filename

        if isinstance(content, str):
            disk_path.write_text(content, encoding="utf-8")
            size_bytes = len(content.encode("utf-8"))
        else:
            disk_path.write_bytes(content)
            size_bytes = len(content)

        # Per-artifact retention marker — read by cleanup_session and
        # cleanup_ephemeral so retention judgments survive process
        # restart and do not require an in-memory registry.
        marker_path = disk_path.with_name(f"{disk_path.name}.retention")
        marker_path.write_text(retention, encoding="utf-8")

        envelope_path = "/".join(
            (_ARTIFACT_ROOT_PREFIX, session_id, dispatch_id, filename)
        )
        return ArtifactReference(
            path=envelope_path,
            content_type=content_type,
            size_bytes=size_bytes,
            summary=summary or f"{filename}: {size_bytes} bytes",
            retention=retention,
        )

    def read_deliverable(self, reference: ArtifactReference) -> str | bytes:
        """Read a substrate-routed deliverable's content (read-side accessor).

        The first read-side API on the formerly write-only store (ADR-034
        §Decision 3; conformance-scan advisory #4). Resolves
        ``reference.path`` to its on-disk location — the structural
        inverse of :meth:`write_deliverable`'s path construction — and
        returns the deliverable content. UTF-8-decodable content returns
        as ``str`` (the common case: code, markdown, JSON); content that
        is not valid UTF-8 returns as ``bytes`` (binary deliverables).

        The content is byte-faithful to what was written: the artifact-
        bridge marshals it into a client tool-call ``content`` argument
        with fidelity, NOT a summary or paraphrase (the FC-49 contract;
        AS-7 result-summarization is upstream and does not apply here).

        Raises :class:`ArtifactNotFoundError` when the reference does not
        resolve to a file on disk (the artifact was cleaned up or the
        reference is stale).
        """
        disk_path = self._resolve_disk_path(reference.path)
        if not disk_path.is_file():
            raise ArtifactNotFoundError(path=reference.path)
        raw = disk_path.read_bytes()
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw

    def _resolve_disk_path(self, envelope_path: str) -> Path:
        """Map an ``ArtifactReference.path`` back to its on-disk location.

        :meth:`write_deliverable` constructs the envelope path by
        prefixing the session-relative segments with
        :data:`_ARTIFACT_ROOT_PREFIX` (``agentic-sessions``) — the
        ``.llm-orc/``-relative shape consumers see at
        ``envelope.artifacts[0]``. The store root already points at the
        ``agentic-sessions`` directory, so resolution strips the prefix
        segment and rejoins the remainder under the root (rather than a
        naive ``self._root / envelope_path``, which would double the
        prefix segment).
        """
        prefix = f"{_ARTIFACT_ROOT_PREFIX}/"
        relative = (
            envelope_path[len(prefix) :]
            if envelope_path.startswith(prefix)
            else envelope_path
        )
        return self._root / relative

    def cleanup_session(self, session_id: str) -> None:
        """Remove ``retention: session`` artifacts under the session-dir.

        Per ADR-025 §"Session-close cleanup removes retention: session
        artifacts" scenario:

        * Each per-dispatch directory containing only ``session``-
          retention artifacts is removed entirely.
        * Per-dispatch directories holding any ``durable`` artifact are
          preserved (the durable artifact stays on disk).
        * ``ephemeral`` artifacts are NOT touched by this method —
          ephemeral retention is the orchestrator-next-turn boundary,
          not the session-close boundary; :meth:`cleanup_ephemeral` is
          the corresponding lifecycle hook.
        * The session directory itself is removed when no per-dispatch
          directory remains; otherwise it persists alongside its
          durable contents.

        Idempotent — calling on an absent session-dir is a no-op so
        request-close paths can always fire it without an existence
        check.
        """
        session_dir = self._root / session_id
        if not session_dir.exists():
            return
        retention_index = self._read_retention_index(session_dir)
        for dispatch_dir in sorted(session_dir.iterdir()):
            if not dispatch_dir.is_dir():
                continue
            self._cleanup_dispatch_dir(
                dispatch_dir=dispatch_dir,
                retention_index=retention_index,
                remove_retentions={"session"},
            )
        # Remove session dir if no per-dispatch dir survives.
        if not any(session_dir.iterdir()):
            session_dir.rmdir()

    def cleanup_ephemeral(self, session_id: str) -> None:
        """Remove ``retention: ephemeral`` artifacts under the session-dir.

        Triggered at the orchestrator's next turn per ADR-025 §"Cleanup"
        — ephemeral artifacts are intermediate-stage deliverables whose
        consuming stage runs immediately; once the next turn starts,
        the ephemeral artifact is no longer needed.

        Does NOT touch ``session`` or ``durable`` retention artifacts
        (composes with :meth:`cleanup_session` — together they cover
        the full retention triplet).
        """
        session_dir = self._root / session_id
        if not session_dir.exists():
            return
        retention_index = self._read_retention_index(session_dir)
        for dispatch_dir in sorted(session_dir.iterdir()):
            if not dispatch_dir.is_dir():
                continue
            self._cleanup_dispatch_dir(
                dispatch_dir=dispatch_dir,
                retention_index=retention_index,
                remove_retentions={"ephemeral"},
            )

    def _read_retention_index(self, session_dir: Path) -> dict[Path, Retention]:
        """Read the per-artifact retention metadata for the session.

        Implementation: each :meth:`write_deliverable` writes a
        sibling ``.retention`` marker file alongside the artifact
        carrying the retention literal. The marker is the lightest-
        weight per-artifact metadata surface that does not require an
        in-memory registry (which would be lost on serve restart and
        wrong on multi-process deployments — though current scope is
        single-process, the marker is forward-compatible).
        """
        index: dict[Path, Retention] = {}
        for dispatch_dir in session_dir.iterdir():
            if dispatch_dir.is_dir():
                self._read_dispatch_retention(dispatch_dir, index)
        return index

    @staticmethod
    def _read_dispatch_retention(
        dispatch_dir: Path, index: dict[Path, Retention]
    ) -> None:
        """Populate ``index`` with retentions for one dispatch directory.

        Extracted from :meth:`_read_retention_index` to keep the outer
        method within the project's complexipy gate (15). Each
        ``<filename>.retention`` marker carries a single retention
        literal; the marker's stem identifies the artifact whose
        retention it records.
        """
        for marker in dispatch_dir.glob("*.retention"):
            value = marker.read_text(encoding="utf-8").strip()
            retention = _coerce_retention(value)
            if retention is not None:
                # Marker name pattern: ``<filename>.retention``. The
                # matched artifact path drops the .retention suffix.
                index[marker.with_suffix("")] = retention

    @staticmethod
    def _cleanup_dispatch_dir(
        *,
        dispatch_dir: Path,
        retention_index: dict[Path, Retention],
        remove_retentions: set[Retention],
    ) -> None:
        """Remove artifacts in ``dispatch_dir`` whose retention matches.

        If the dispatch_dir contains no surviving artifact after
        removal, the directory itself is removed.
        """
        for artifact_path in sorted(dispatch_dir.iterdir()):
            if artifact_path.suffix == ".retention":
                continue
            retention = retention_index.get(artifact_path)
            if retention is not None and retention in remove_retentions:
                marker = artifact_path.with_name(f"{artifact_path.name}.retention")
                artifact_path.unlink()
                if marker.exists():
                    marker.unlink()
        # If no non-marker files remain, remove the dispatch directory.
        surviving = [p for p in dispatch_dir.iterdir() if p.suffix != ".retention"]
        if not surviving:
            # Remove any lingering markers + the dir itself.
            shutil.rmtree(dispatch_dir)

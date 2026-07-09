"""Session Artifact Store unit tests (Cycle 6 WP-E, ADR-025)."""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest

from llm_orc.core.session.artifact_store import (
    ArtifactNotFoundError,
    ArtifactReference,
    SessionArtifactStore,
    new_session_id,
)


class TestNewSessionId:
    """``<session_id>`` format helper.

    Per ADR-025 §"Session-dir location": format
    ``<iso-8601-datetime>-<short-uuid>`` (e.g.,
    ``2026-05-15T14:32:08Z-a7f3``). The ISO-8601 prefix gives natural
    chronological sort on the filesystem; the short UUID
    disambiguates same-second sessions.
    """

    def test_new_session_id_matches_documented_format(self) -> None:
        sid = new_session_id()
        assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z-[0-9a-f]{4}$", sid), (
            f"session_id {sid!r} does not match the ADR-025 documented "
            "format <iso-8601-datetime>-<short-uuid>"
        )

    def test_new_session_id_unique_per_call(self) -> None:
        """Same-second calls disambiguate via the short-uuid suffix."""
        sids = {new_session_id() for _ in range(20)}
        assert len(sids) == 20, "session_id collisions in 20 same-second calls"


class TestArtifactReference:
    """ADR-025 §"Artifact reference fields" — typed envelope payload.

    The reference shape is frozen-dataclass so envelope.artifacts[0]
    consumers can rely on field identity.
    """

    def test_artifact_reference_field_set(self) -> None:
        ref = ArtifactReference(
            path="agentic-sessions/2026-05-15T14:32:08Z-a7f3/dispatch-001/buf.py",
            content_type="application/python",
            size_bytes=1247,
            summary="Class CircularBuffer with iter and len protocol; 24 lines.",
            retention="session",
        )
        assert ref.path.startswith("agentic-sessions/")
        assert ref.content_type == "application/python"
        assert ref.size_bytes == 1247
        assert ref.retention == "session"

    def test_artifact_reference_is_frozen(self) -> None:
        ref = ArtifactReference(
            path="x",
            content_type="text/plain",
            size_bytes=0,
            summary="",
            retention="session",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            ref.size_bytes = 1  # type: ignore[misc]


class TestWriteDeliverable:
    """``write_deliverable`` produces the ADR-025 path layout + reference."""

    def test_write_deliverable_writes_under_session_dispatch_directory(
        self, tmp_path: Path
    ) -> None:
        store = SessionArtifactStore(agentic_sessions_root=tmp_path)
        ref = store.write_deliverable(
            session_id="2026-05-15T14:32:08Z-a7f3",
            dispatch_id="dispatch-001",
            deliverable_name="circular_buffer",
            content="class CircularBuffer:\n    pass\n",
            content_type="application/python",
        )

        # Filesystem invariant — content lands at the ADR-025 path layout.
        expected_disk_path = (
            tmp_path
            / "2026-05-15T14:32:08Z-a7f3"
            / "dispatch-001"
            / "circular_buffer.py"
        )
        assert expected_disk_path.exists()
        assert expected_disk_path.read_text() == "class CircularBuffer:\n    pass\n"

        # ArtifactReference invariant — the envelope payload's path is
        # relative to .llm-orc/ (i.e., relative to the parent of
        # agentic-sessions/). When the store is rooted at tmp_path
        # directly, ``path`` carries the session-relative shape per
        # ADR-025 §"Artifact reference fields".
        assert ref.path == (
            "agentic-sessions/2026-05-15T14:32:08Z-a7f3/dispatch-001/circular_buffer.py"
        )
        assert ref.content_type == "application/python"
        assert ref.size_bytes == len(b"class CircularBuffer:\n    pass\n")
        assert ref.retention == "session"  # default per ADR-025

    def test_write_deliverable_honors_retention_when_supplied(
        self, tmp_path: Path
    ) -> None:
        store = SessionArtifactStore(agentic_sessions_root=tmp_path)
        ref = store.write_deliverable(
            session_id="sess-1",
            dispatch_id="dispatch-001",
            deliverable_name="report",
            content="durable content",
            content_type="text/markdown",
            retention="durable",
        )
        assert ref.retention == "durable"

    def test_write_deliverable_chooses_extension_from_content_type(
        self, tmp_path: Path
    ) -> None:
        store = SessionArtifactStore(agentic_sessions_root=tmp_path)
        cases = [
            ("application/python", "code", ".py"),
            ("text/markdown", "doc", ".md"),
            ("application/json", "claims", ".json"),
            ("text/plain", "notes", ".txt"),
        ]
        for content_type, name, expected_ext in cases:
            ref = store.write_deliverable(
                session_id="sess-ext",
                dispatch_id="dispatch-001",
                deliverable_name=name,
                content="payload",
                content_type=content_type,
            )
            assert ref.path.endswith(f"{name}{expected_ext}"), (
                f"content_type {content_type!r} should produce ext "
                f"{expected_ext!r}; got path {ref.path!r}"
            )

    def test_write_deliverable_falls_back_to_bin_for_unknown_content_type(
        self, tmp_path: Path
    ) -> None:
        """Unknown content-types take ``.bin`` so the path is always
        deterministic — the deliverable is still on disk; downstream
        consumers identify type via ArtifactReference.content_type.
        """
        store = SessionArtifactStore(agentic_sessions_root=tmp_path)
        ref = store.write_deliverable(
            session_id="sess-bin",
            dispatch_id="dispatch-001",
            deliverable_name="blob",
            content=b"\x00\x01\x02",
            content_type="application/x-custom",
        )
        assert ref.path.endswith("blob.bin")

    def test_write_deliverable_accepts_bytes_and_str_content(
        self, tmp_path: Path
    ) -> None:
        """``content`` accepts ``bytes`` (binary deliverables) and ``str``
        (text deliverables); both round-trip correctly to disk and the
        size_bytes reflects the byte length.
        """
        store = SessionArtifactStore(agentic_sessions_root=tmp_path)
        text_ref = store.write_deliverable(
            session_id="sess-mixed",
            dispatch_id="dispatch-001",
            deliverable_name="text",
            content="hello",
            content_type="text/plain",
        )
        bytes_ref = store.write_deliverable(
            session_id="sess-mixed",
            dispatch_id="dispatch-002",
            deliverable_name="blob",
            content=b"\xff\xfe\xfd",
            content_type="application/octet-stream",
        )
        assert text_ref.size_bytes == 5
        assert bytes_ref.size_bytes == 3

    def test_write_deliverable_creates_parent_directories(self, tmp_path: Path) -> None:
        """The store creates intermediate directories on demand —
        operators do not pre-create per-session or per-dispatch dirs."""
        nested_root = tmp_path / "deeply" / "nested"
        store = SessionArtifactStore(agentic_sessions_root=nested_root)
        store.write_deliverable(
            session_id="sess-deep",
            dispatch_id="dispatch-001",
            deliverable_name="x",
            content="ok",
            content_type="text/plain",
        )
        assert (nested_root / "sess-deep" / "dispatch-001" / "x.txt").exists()


class TestCleanupSession:
    """``cleanup_session`` honors retention per ADR-025 §"Cleanup"."""

    def test_cleanup_session_removes_session_retention_artifacts(
        self, tmp_path: Path
    ) -> None:
        store = SessionArtifactStore(agentic_sessions_root=tmp_path)
        store.write_deliverable(
            session_id="sess-clean",
            dispatch_id="dispatch-001",
            deliverable_name="x",
            content="session-scoped",
            content_type="text/plain",
            retention="session",
        )
        store.write_deliverable(
            session_id="sess-clean",
            dispatch_id="dispatch-002",
            deliverable_name="y",
            content="also session",
            content_type="text/plain",
            retention="session",
        )

        store.cleanup_session("sess-clean")

        # Session dir removed entirely when no durable artifacts remain.
        assert not (tmp_path / "sess-clean").exists()

    def test_cleanup_session_preserves_durable_artifacts(self, tmp_path: Path) -> None:
        store = SessionArtifactStore(agentic_sessions_root=tmp_path)
        store.write_deliverable(
            session_id="sess-mixed",
            dispatch_id="dispatch-001",
            deliverable_name="ephemeral_x",
            content="session-scoped",
            content_type="text/plain",
            retention="session",
        )
        durable_ref = store.write_deliverable(
            session_id="sess-mixed",
            dispatch_id="dispatch-002",
            deliverable_name="durable_y",
            content="keep me",
            content_type="text/plain",
            retention="durable",
        )

        store.cleanup_session("sess-mixed")

        # Session retention artifact is gone; per-dispatch dir is gone
        # for the cleaned dispatch.
        assert not (tmp_path / "sess-mixed" / "dispatch-001").exists()
        # Durable artifact remains on disk.
        durable_disk_path = tmp_path / Path(durable_ref.path).relative_to(
            "agentic-sessions"
        )
        assert durable_disk_path.exists()
        assert durable_disk_path.read_text() == "keep me"
        # Session dir remains because at least one durable artifact
        # persists per ADR-025 §"Session-close cleanup removes
        # retention: session artifacts" scenario.
        assert (tmp_path / "sess-mixed").exists()

    def test_cleanup_session_is_noop_when_session_dir_absent(
        self, tmp_path: Path
    ) -> None:
        """A session that never wrote any artifact (or whose dir was
        already cleaned) does not raise on cleanup — the call is
        idempotent so request-close paths can always fire it without
        existence checks."""
        store = SessionArtifactStore(agentic_sessions_root=tmp_path)
        store.cleanup_session("never-existed")
        # No exception, no created dir.
        assert not (tmp_path / "never-existed").exists()

    def test_cleanup_session_removes_only_session_retention(
        self, tmp_path: Path
    ) -> None:
        """An ``ephemeral`` artifact is NOT removed by session-close
        cleanup — ephemeral retention is cleaned at the orchestrator's
        next turn (cleanup_ephemeral), not at session close."""
        store = SessionArtifactStore(agentic_sessions_root=tmp_path)
        store.write_deliverable(
            session_id="sess-eph",
            dispatch_id="dispatch-001",
            deliverable_name="x",
            content="ephemeral",
            content_type="text/plain",
            retention="ephemeral",
        )
        store.cleanup_session("sess-eph")
        # Ephemeral artifact still present after session-close (not
        # session-retention, so cleanup_session is not its trigger).
        assert (tmp_path / "sess-eph" / "dispatch-001" / "x.txt").exists()


class TestCleanupEphemeral:
    """``cleanup_ephemeral`` removes ephemeral artifacts at next turn."""

    def test_cleanup_ephemeral_removes_ephemeral_artifacts_for_session(
        self, tmp_path: Path
    ) -> None:
        store = SessionArtifactStore(agentic_sessions_root=tmp_path)
        store.write_deliverable(
            session_id="sess-eph",
            dispatch_id="dispatch-001",
            deliverable_name="x",
            content="ephemeral",
            content_type="text/plain",
            retention="ephemeral",
        )
        store.write_deliverable(
            session_id="sess-eph",
            dispatch_id="dispatch-002",
            deliverable_name="y",
            content="durable",
            content_type="text/plain",
            retention="durable",
        )

        store.cleanup_ephemeral("sess-eph")

        # Ephemeral removed.
        assert not (tmp_path / "sess-eph" / "dispatch-001" / "x.txt").exists()
        # Durable preserved.
        assert (tmp_path / "sess-eph" / "dispatch-002" / "y.txt").exists()

    def test_cleanup_ephemeral_does_not_remove_session_retention(
        self, tmp_path: Path
    ) -> None:
        store = SessionArtifactStore(agentic_sessions_root=tmp_path)
        store.write_deliverable(
            session_id="sess-keep-session",
            dispatch_id="dispatch-001",
            deliverable_name="x",
            content="session",
            content_type="text/plain",
            retention="session",
        )

        store.cleanup_ephemeral("sess-keep-session")
        # Session-retention artifact is NOT touched by cleanup_ephemeral.
        assert (tmp_path / "sess-keep-session" / "dispatch-001" / "x.txt").exists()


class TestReadDeliverable:
    """``read_deliverable`` — the read-side accessor (Cycle 7 loop-back,
    ADR-034 §Decision 3 / WP-LB-D).

    The first read-side API on the formerly write-only store (advisory
    #4 — the highest-priority BUILD design dependency; the artifact-
    bridge terminal chain cannot close without it). Resolution is the
    structural inverse of ``write_deliverable``'s path construction:
    ``ArtifactReference.path`` carries the ``agentic-sessions/``-prefixed,
    ``.llm-orc/``-relative shape, so the accessor strips the prefix and
    rejoins under the store root.
    """

    def test_read_deliverable_round_trips_text_content(self, tmp_path: Path) -> None:
        """A text deliverable reads back byte-identical to what was
        written — full fidelity, the FC-49 contract at the store layer."""
        store = SessionArtifactStore(agentic_sessions_root=tmp_path)
        content = "def add(a: int, b: int) -> int:\n    return a + b\n"
        ref = store.write_deliverable(
            session_id="2026-06-02T10:00:00Z-aa11",
            dispatch_id="dispatch-001",
            deliverable_name="calc",
            content=content,
            content_type="application/python",
        )
        assert store.read_deliverable(ref) == content

    def test_read_deliverable_round_trips_binary_content(self, tmp_path: Path) -> None:
        """Non-UTF-8 bytes read back as ``bytes`` (the decode-probe falls
        back to raw bytes), so the accessor is faithful for binary
        deliverables, not only text."""
        store = SessionArtifactStore(agentic_sessions_root=tmp_path)
        content = b"\xff\xfe\xfd\x00\x01"
        ref = store.write_deliverable(
            session_id="sess-bin",
            dispatch_id="dispatch-001",
            deliverable_name="blob",
            content=content,
            content_type="application/octet-stream",
        )
        assert store.read_deliverable(ref) == content

    def test_read_deliverable_raises_artifact_not_found_for_missing_path(
        self, tmp_path: Path
    ) -> None:
        """An unresolvable reference raises the typed structural error so
        the terminal (WP-LB-C) can degrade to a dispatch-failure text
        completion rather than emit a malformed tool call (FC-48 /
        integration-contract error handling)."""
        store = SessionArtifactStore(agentic_sessions_root=tmp_path)
        dangling = ArtifactReference(
            path="agentic-sessions/sess-x/dispatch-001/gone.py",
            content_type="application/python",
            size_bytes=10,
            summary="missing",
            retention="session",
        )
        with pytest.raises(ArtifactNotFoundError) as exc_info:
            store.read_deliverable(dangling)
        assert exc_info.value.error_kind == "artifact_not_found"

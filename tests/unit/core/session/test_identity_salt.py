"""Tests for hash_identity_for_wire (SF5).

The wire header must never carry a SessionIdentity's raw value (for
user_field identities, that's the client's OpenAI `user` field
verbatim) but must stay stable for the same identity across turns.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_orc.core.session import identity_salt


@pytest.fixture(autouse=True)
def isolated_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Every test gets its own salt file, never the developer's real one."""
    config_dir = tmp_path / "llm-orc"
    monkeypatch.setattr(identity_salt, "resolve_global_config_dir", lambda: config_dir)
    return config_dir


class TestHashIdentityForWire:
    def test_hash_never_equals_the_raw_value(self) -> None:
        raw = "raw-client-user-id"

        assert identity_salt.hash_identity_for_wire(raw) != raw

    def test_hash_is_stable_for_the_same_identity(self) -> None:
        raw = "conversation-abc123"

        first = identity_salt.hash_identity_for_wire(raw)
        second = identity_salt.hash_identity_for_wire(raw)

        assert first == second

    def test_different_identities_hash_differently(self) -> None:
        first = identity_salt.hash_identity_for_wire("identity-one")
        second = identity_salt.hash_identity_for_wire("identity-two")

        assert first != second

    def test_salt_is_persisted_under_the_global_config_dir(
        self, isolated_config_dir: Path
    ) -> None:
        identity_salt.hash_identity_for_wire("some-identity")

        salt_file = isolated_config_dir / "session_id_salt"
        assert salt_file.exists()

    def test_salt_file_is_owner_only_permissions(
        self, isolated_config_dir: Path
    ) -> None:
        identity_salt.hash_identity_for_wire("some-identity")

        salt_file = isolated_config_dir / "session_id_salt"
        assert oct(salt_file.stat().st_mode)[-3:] == "600"

    def test_salt_is_reused_across_calls_not_regenerated(
        self, isolated_config_dir: Path
    ) -> None:
        """A regenerated salt on every call would make the hash
        unstable across turns - the salt file must be read back, not
        rewritten, once it exists."""
        identity_salt.hash_identity_for_wire("first-call")
        salt_file = isolated_config_dir / "session_id_salt"
        salt_after_first_call = salt_file.read_bytes()

        identity_salt.hash_identity_for_wire("second-call")
        salt_after_second_call = salt_file.read_bytes()

        assert salt_after_first_call == salt_after_second_call

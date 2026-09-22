"""Salted hashing for a SessionIdentity value that crosses the wire.

SF5 (serving-roadmap): the serve resolves a caller's SessionIdentity
(``core/session/registry.py``) and, for the ``user_field`` method, that
identity's ``value`` is the client's OpenAI ``user`` field VERBATIM.
Threading that raw value into ``x-opencode-session`` would send a
client-supplied identifier straight to a third party (opencode.ai).

This module hashes the identity value with a salt generated once per
install and persisted under the same local config directory
credentials/the encryption key already live in
(``ConfigurationManager.global_config_dir`` /
``resolve_global_config_dir()``) — never hardcoded, never sent anywhere.
The result is stable for a given identity value (same salt, same input)
but never equal to the raw value and not reversible to it without the
local salt file.
"""

from __future__ import annotations

import hashlib
import os

from llm_orc.core.config.config_manager import resolve_global_config_dir

_SALT_FILENAME = "session_id_salt"


def _get_or_create_salt() -> bytes:
    """Read the install's session-id salt, generating and persisting a
    fresh random one on first use."""
    config_dir = resolve_global_config_dir()
    salt_file = config_dir / _SALT_FILENAME

    if salt_file.exists():
        return salt_file.read_bytes()

    config_dir.mkdir(parents=True, exist_ok=True)
    salt = os.urandom(32)
    salt_file.write_bytes(salt)
    os.chmod(salt_file, 0o600)
    return salt


def hash_identity_for_wire(identity_value: str) -> str:
    """A stable, non-reversible id for ``identity_value`` safe to send
    on the wire (e.g. as ``x-opencode-session``).

    Salted with a random value generated once per install, so the hash
    can't be correlated across installs or reversed to the raw
    identity, but stays stable across turns of the same conversation
    (same identity value + same local salt -> same hash).
    """
    salt = _get_or_create_salt()
    return hashlib.sha256(salt + identity_value.encode("utf-8")).hexdigest()

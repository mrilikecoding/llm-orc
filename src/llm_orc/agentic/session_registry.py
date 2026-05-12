"""Session Registry — identifies and continues multi-request Sessions.

Per `docs/agentic-serving/system-design.md` §Session Registry (L3) and
the Cycle 4 extension in `system-design.agents.md` §Session Registry
(per ADR-013). Reconstructs orchestrator state from the
OpenAI-compatible chat conversation; tracks cumulative turn and token
accounting; resolves cluster determination at session-start and
governs structured-handoff artifact set activation per ADR-013
disposition (i) (cross-cluster ambiguity defaults to required).
Persistence of artifacts themselves lives in
:mod:`llm_orc.agentic.session_artifacts` — Session Registry owns the
session-shape decisions; the artifact module owns the write-gate
enforcement.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    # ChatMessage lives in session_start so the Orchestrator Runtime (and
    # any other FC-4-constrained consumer) can import it without reaching
    # into this module. TYPE_CHECKING guards the otherwise-circular
    # import: session_start imports SessionIdentity and SessionState
    # from here for its SessionContext definition.
    from llm_orc.agentic.session_start import ChatMessage

IdentityMethod = Literal["user_field", "message_prefix", "cold_start"]
Cluster = Literal["cluster_1", "cluster_2", "cluster_3"]

_VALID_CLUSTERS: frozenset[Cluster] = frozenset(("cluster_1", "cluster_2", "cluster_3"))
_AMBIGUOUS_DEFAULT: Cluster = "cluster_2"


def resolve_cluster(declaration: str | list[str] | None) -> Cluster:
    """Resolve a Cluster from an operator's session-start declaration.

    Per ADR-013 §"Cross-cluster sessions" disposition (i): cross-cluster
    ambiguity defaults to ``cluster_2`` (required-artifact-set behavior).
    Single explicit declarations resolve to the named cluster; multi-
    cluster declarations, empty declarations, ``None``, and unrecognized
    cluster names all fall back to ``cluster_2`` so the artifact set is
    active for the session.
    """
    if isinstance(declaration, list):
        if len(declaration) == 1 and declaration[0] in _VALID_CLUSTERS:
            return _as_cluster(declaration[0])
        return _AMBIGUOUS_DEFAULT
    if declaration is None:
        return _AMBIGUOUS_DEFAULT
    if declaration in _VALID_CLUSTERS:
        return _as_cluster(declaration)
    return _AMBIGUOUS_DEFAULT


def _as_cluster(value: str) -> Cluster:
    """Narrow ``str`` to ``Cluster`` after membership check."""
    if value == "cluster_1":
        return "cluster_1"
    if value == "cluster_2":
        return "cluster_2"
    return "cluster_3"


def requires_structured_handoff_artifacts(
    cluster: Cluster,
    *,
    opt_in: bool = False,
) -> bool:
    """Whether the structured-handoff artifact set is required for ``cluster``.

    Per ADR-013 §"Cluster-conditional applicability": Cluster 2 always
    requires the artifact set; Cluster 1 and Cluster 3 are supported
    but optional, with operator opt-in via explicit configuration.
    """
    if cluster == "cluster_2":
        return True
    return opt_in


@dataclass(frozen=True)
class SessionIdentity:
    """Identifies a Session across HTTP requests.

    Derivation-method-agnostic per the Serving Layer → Session Registry
    integration contract: the identity value may come from the
    OpenAI `user` field, a message-prefix hash, or (future) an
    explicit session-id header. The method is retained so consumers
    can reason about identity stability.
    """

    value: str
    method: IdentityMethod


@dataclass
class SessionState:
    """Mutable per-Session accounting.

    Tracks cumulative turn count (ReAct iterations) and cumulative
    token spend, summed across requests that share a SessionIdentity.
    Budget, Autonomy, and Calibration state live in their own modules
    and read this state through Session Registry contracts.
    """

    identity: SessionIdentity
    turn_count: int = 0
    token_spend: int = 0

    def record_iteration(self, tokens: int) -> None:
        """Record one ReAct iteration's contribution to the Session."""
        self.turn_count += 1
        self.token_spend += tokens


class SessionRegistry:
    """Per-process registry of active Sessions.

    Resolves identity from request features and returns the canonical
    mutable state object for a given identity. Cold-start requests
    (no user field, no user message) produce a fresh identity so that
    each such request is treated as its own Session.
    """

    def __init__(self) -> None:
        self._states: dict[SessionIdentity, SessionState] = {}

    def resolve_identity(
        self,
        *,
        messages: list[ChatMessage],
        user_field: str | None,
    ) -> SessionIdentity:
        if user_field is not None:
            return SessionIdentity(value=user_field, method="user_field")

        first_user = next((m for m in messages if m.role == "user"), None)
        if first_user is None:
            return SessionIdentity(value=uuid.uuid4().hex, method="cold_start")

        # Tolerate None content (e.g., malformed user message) so
        # identity derivation never raises on the request path.
        content = first_user.content or ""
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return SessionIdentity(value=digest, method="message_prefix")

    def get_or_create_state(self, identity: SessionIdentity) -> SessionState:
        existing = self._states.get(identity)
        if existing is not None:
            return existing
        created = SessionState(identity=identity)
        self._states[identity] = created
        return created

    def resolve_session_cluster(
        self,
        *,
        declaration: str | list[str] | None,
    ) -> Cluster:
        """Resolve cluster at session-start per ADR-013 disposition (i).

        Thin wrapper over :func:`resolve_cluster`; the registry boundary
        exists so consumers (Serving Layer at session-start) call into
        Session Registry rather than reaching into module-level
        helpers — keeping cluster determination owned by the module the
        responsibility matrix assigns it to.
        """
        return resolve_cluster(declaration)

    def session_requires_artifact_set(
        self,
        cluster: Cluster,
        *,
        opt_in: bool = False,
    ) -> bool:
        """Whether the artifact set is required for ``cluster``.

        Thin wrapper over :func:`requires_structured_handoff_artifacts`
        co-located on the registry so callers do not import both the
        registry and module-level helper for the same decision.
        """
        return requires_structured_handoff_artifacts(cluster, opt_in=opt_in)

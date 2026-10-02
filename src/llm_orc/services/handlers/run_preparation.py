"""What a run does between its request and its executor (Arc 4).

The pieces the execution handler's one preparation step is built from:
the refusal and its envelope (ruling 10), the rows an unmet binding adds
to the report (ruling 3), the bind keys the closure does not name, and
the pull (ruling 4). The pull trusts only what it observed itself: a
dependency is resolved when the router answered ``loaded``, and the
listing is never re-read to decide.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from llm_orc.core.config.closure import Dependency
from llm_orc.services.handlers.preflight import (
    RESOLVE,
    DependencyReport,
    DependencyStatus,
)

NOT_EQUIPPED = "not_equipped"
INVALID_REQUEST = "invalid_request"


class RunRefusedError(Exception):
    """The run is refused before any agent starts."""

    def __init__(
        self,
        kind: str,
        message: str,
        dependencies: Sequence[DependencyReport] = (),
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.dependencies = list(dependencies)

    @property
    def error(self) -> dict[str, Any]:
        """The ``error`` object of the envelope."""
        return {
            "kind": self.kind,
            "message": self.message,
            "dependencies": [d.model_dump(mode="json") for d in self.dependencies],
        }

    def envelope(self) -> dict[str, Any]:
        """The result a refused run returns (ruling 10)."""
        return {
            "status": "error",
            "has_errors": True,
            "results": {},
            "deliverable": None,
            "error": self.error,
        }


def unmet_binding_rows(unmet: Sequence[tuple[str, str]]) -> list[DependencyReport]:
    """A ``missing_profile`` row for each bind target the host lacks,
    reached ``via`` the binding, whether or not the host has the key."""
    status = DependencyStatus.MISSING_PROFILE
    return [
        DependencyReport(
            kind="profile",
            name=target,
            via=[f"bind:{key}"],
            status=status,
            resolve=RESOLVE[status],
            detail=f"no profile named {target!r} (bind target of {key!r})",
        )
        for key, target in unmet
    ]


def unnamed_bind_keys(
    bind: Mapping[str, str], dependencies: Sequence[Dependency]
) -> list[str]:
    """Bind keys no profile dependency in the closure names: a
    misspelled key must not run on the host's profile of the intended
    name."""
    named = {d.name for d in dependencies if d.kind == "profile"}
    return [key for key in bind if key not in named]

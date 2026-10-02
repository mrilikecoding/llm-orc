"""What a run does between its request and its executor (Arc 4).

The pieces the execution handler's one preparation step is built from:
the refusal and its envelope (ruling 10), the rows an unmet binding adds
to the report (ruling 3), the bind keys the closure does not name, and
the pull (ruling 4). The pull trusts only what it observed itself: a
dependency is resolved when the router answered ``loaded``, and the
listing is never re-read to decide.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from llm_orc.services.handlers.preflight import DependencyReport

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

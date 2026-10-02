"""What a run does between its request and its executor (Arc 4).

The pieces the execution handler's one preparation step is built from:
the refusal and its envelope (ruling 10), the rows an unmet binding adds
to the report (ruling 3), the bind keys the closure does not name, and
the pull (ruling 4). The pull trusts only what it observed itself: a
dependency is resolved when the router answered ``loaded``, and the
listing is never re-read to decide.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

from llm_orc.core.config.closure import Dependency
from llm_orc.providers.llama_server import PULL_POLL_S, PULL_TIMEOUT_S, router_client
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


def _model_of(report: DependencyReport, profiles: Mapping[str, Any]) -> str:
    if report.kind == "model":
        return report.name
    return str(profiles.get(report.name, {}).get("model", ""))


async def pull_pullable(
    reports: Sequence[DependencyReport], profiles: Mapping[str, Any]
) -> tuple[list[DependencyReport], list[str]]:
    """Pull each ``pullable`` model once, off the event loop.

    Returns the reports (a dependency whose model answered ``loaded`` is
    ``ready``; any other answer leaves it ``pullable`` with the observed
    status in its detail) and the models that loaded.
    """
    models = sorted(
        {
            _model_of(r, profiles)
            for r in reports
            if r.status is DependencyStatus.PULLABLE
        }
        - {""}
    )
    outcomes = {model: await _pull_one(model) for model in models}
    loaded = [m for m, (ok, _) in outcomes.items() if ok]
    updated: list[DependencyReport] = []
    for report in reports:
        outcome = outcomes.get(_model_of(report, profiles))
        if report.status is not DependencyStatus.PULLABLE or outcome is None:
            updated.append(report)
        elif outcome[0]:
            updated.append(
                report.model_copy(
                    update={
                        "status": DependencyStatus.READY,
                        "resolve": RESOLVE[DependencyStatus.READY],
                        "detail": outcome[1],
                    }
                )
            )
        else:
            updated.append(
                report.model_copy(update={"detail": f"{report.detail}; {outcome[1]}"})
            )
    return updated, loaded


async def _pull_one(model: str) -> tuple[bool, str]:
    """``(loaded, what was observed)`` for one pull."""
    client = router_client()
    try:
        result = await asyncio.to_thread(
            client.pull, model, timeout_s=PULL_TIMEOUT_S, poll_s=PULL_POLL_S
        )
    except (OSError, ValueError) as e:
        return False, f"pull of {model!r} failed: {type(e).__name__}: {e}"
    status = str(result["status"])
    if status == "loaded":
        return True, f"pulled: model {model!r} loaded"
    extra = ""
    if result.get("failed"):
        extra = f" (failed, exit code {result.get('exit_code')})"
    return False, f"pull of {model!r} ended {status}{extra}"

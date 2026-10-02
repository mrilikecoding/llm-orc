"""A mock service whose preparation step hands out a given executor.

``invoke`` runs inside ``service.prepared_run``. The gate itself is
pinned through the real service in test_invoke_prepared_run.py; these
tests pin what the command does around it (lookup, input, display,
concurrency), so the step is replaced by one that finds the root with
the lookup the command supplies and yields it with the executor.
"""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import Mock

from llm_orc.services.handlers.execution_handler import PreparedRun


def use_prepared_run(service: Mock, executor: Any) -> None:
    @asynccontextmanager
    async def prepared_run(
        request: dict[str, Any], lookup: Callable[[str], Any]
    ) -> AsyncIterator[PreparedRun]:
        yield PreparedRun(lookup(request["ensemble_name"]), executor)

    service.prepared_run = prepared_run

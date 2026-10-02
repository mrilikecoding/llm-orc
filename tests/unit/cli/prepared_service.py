"""A mock service whose preparation step hands out a given executor.

``invoke`` runs inside ``service.prepared_run``. The gate itself is
pinned through the real service in test_invoke_prepared_run.py; these
tests pin what the command does around it (lookup, input, display,
concurrency), so the step is replaced by one that finds the root with
the lookup the command supplies and yields it with the executor. A root
the lookup does not find is the real step's ``RootNotFoundError``.
"""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import Mock

from llm_orc.services.handlers.execution_handler import PreparedRun
from llm_orc.services.handlers.run_preparation import RootNotFoundError


def use_prepared_run(service: Mock, executor: Any) -> None:
    @asynccontextmanager
    async def prepared_run(
        request: dict[str, Any], lookup: Callable[[str], Any]
    ) -> AsyncIterator[PreparedRun]:
        name = request["ensemble_name"]
        config = lookup(name)
        if config is None:
            raise RootNotFoundError(f"Ensemble does not exist: {name}", name)
        yield PreparedRun(config, executor)

    service.prepared_run = prepared_run

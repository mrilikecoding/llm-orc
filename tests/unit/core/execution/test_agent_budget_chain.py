"""The agent's budget bounds the HTTP read, driven through the real chain.

Runner -> factory -> OpenAICompatibleModel -> httpx, against a local server
that answers after a delay, with the pool's read default set shorter than
that delay. The agent's ``timeout_seconds`` decides: a budget longer than
the delay succeeds, a shorter one fails with a message that names the
budget. Remote host 2026-10-10: a compile died at the client's 180 s
default while the agent had 900, with ``error: ''`` as the only record.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from unittest.mock import Mock

import pytest

from llm_orc.core.auth.authentication import CredentialStorage
from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.execution.runners.llm_runner import LlmAgentRunner
from llm_orc.core.execution.usage_collector import UsageCollector
from llm_orc.core.models.model_factory import ModelFactory
from llm_orc.models.base import HTTPConnectionPool
from llm_orc.schemas.agent_config import LlmAgentConfig

DELAY_S = 1.6
POOL_READ_S = 0.2


async def _slow_chat_completions(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    """Read one request, wait, answer like /v1/chat/completions."""
    while True:
        line = await reader.readline()
        if not line or line in (b"\r\n", b"\n"):
            break
    await asyncio.sleep(DELAY_S)
    body = json.dumps(
        {
            "choices": [{"message": {"content": "compiled"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
    ).encode()
    writer.write(
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
        + f"Content-Length: {len(body)}\r\n".encode()
        + b"Connection: close\r\n\r\n"
        + body
    )
    await writer.drain()
    writer.close()


@pytest.fixture
async def slow_server() -> AsyncIterator[str]:
    server = await asyncio.start_server(_slow_chat_completions, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        server.close()
        await server.wait_closed()


@pytest.fixture
async def short_pool_read() -> AsyncIterator[None]:
    saved = HTTPConnectionPool._performance_config
    HTTPConnectionPool.configure(
        {"concurrency": {"request_timeout": {"read": POOL_READ_S}}}
    )
    try:
        yield
    finally:
        await HTTPConnectionPool.close()
        HTTPConnectionPool._performance_config = saved


def _runner(base_url: str) -> LlmAgentRunner:
    config_manager = Mock(spec=ConfigurationManager)
    profile = {"model": "slow", "provider": "llama-server", "base_url": base_url}
    config_manager.get_model_profiles.return_value = {"slow-seat": profile}
    config_manager.get_model_profile.return_value = profile
    config_manager.resolve_model_profile.return_value = ("slow", "llama-server")
    storage = Mock(spec=CredentialStorage)
    storage.get_auth_method.return_value = None
    return LlmAgentRunner(
        model_factory=ModelFactory(config_manager, storage),
        config_manager=config_manager,
        usage_collector=UsageCollector(),
        emit_event=lambda _n, _d: None,
        classify_failure=lambda _e: "unknown",
        performance_config={"execution": {"default_timeout": 60}},
    )


@pytest.mark.asyncio
async def test_a_budget_longer_than_the_answer_succeeds(
    slow_server: str, short_pool_read: None
) -> None:
    runner = _runner(slow_server)
    agent = LlmAgentConfig(
        name="compiler", model_profile="slow-seat", timeout_seconds=5
    )

    response, _, _ = await runner.execute(agent, "compile this")

    assert response == "compiled"


@pytest.mark.asyncio
async def test_a_budget_shorter_than_the_answer_names_itself(
    slow_server: str, short_pool_read: None
) -> None:
    runner = _runner(slow_server)
    agent = LlmAgentConfig(
        name="compiler", model_profile="slow-seat", timeout_seconds=1
    )

    with pytest.raises(Exception, match=r"slow at .* did not answer within 1 s"):
        await runner.execute(agent, "compile this")

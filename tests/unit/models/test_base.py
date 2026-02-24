"""Tests for base model infrastructure."""

from collections.abc import Generator
from unittest.mock import AsyncMock, Mock, patch

import pytest

from llm_orc.models.base import HTTPConnectionPool, ModelInterface


@pytest.fixture(autouse=True)
def mock_httpx() -> Generator[Mock, None, None]:
    """Mock httpx to avoid real network operations in tests."""
    mock_client = Mock()
    mock_client.is_closed = False
    mock_client.aclose = AsyncMock()

    with patch("httpx.AsyncClient", return_value=mock_client):
        with patch("httpx.Limits"):
            with patch("httpx.Timeout"):
                yield mock_client


class TestHTTPConnectionPool:
    """Test HTTP connection pool functionality."""

    def setup_method(self) -> None:
        """Reset singleton instance before each test."""
        HTTPConnectionPool._instance = None
        HTTPConnectionPool._httpx_client = None
        HTTPConnectionPool._performance_config = None

    def test_singleton_pattern(self) -> None:
        """Test that HTTPConnectionPool implements singleton pattern."""
        # Given & When
        pool1 = HTTPConnectionPool()
        pool2 = HTTPConnectionPool()

        # Then
        assert pool1 is pool2
        assert HTTPConnectionPool._instance is pool1

    @pytest.mark.asyncio
    async def test_get_httpx_client_creates_client(self) -> None:
        """Test that get_httpx_client creates and returns httpx client."""
        # When
        client = HTTPConnectionPool.get_httpx_client()

        # Then
        assert client is not None
        assert HTTPConnectionPool._httpx_client is client

        # Cleanup
        await HTTPConnectionPool.close()

    @pytest.mark.asyncio
    async def test_get_httpx_client_reuses_existing(self) -> None:
        """Test that get_httpx_client reuses existing client."""
        # Given
        client1 = HTTPConnectionPool.get_httpx_client()

        # When
        client2 = HTTPConnectionPool.get_httpx_client()

        # Then
        assert client1 is client2

        # Cleanup
        await HTTPConnectionPool.close()

    @pytest.mark.asyncio
    async def test_get_httpx_client_uses_defaults_when_not_configured(
        self,
    ) -> None:
        """Test get_httpx_client works with defaults when configure() not called."""
        # Given - no configure() call, _performance_config is None
        assert HTTPConnectionPool._performance_config is None

        # When
        client = HTTPConnectionPool.get_httpx_client()

        # Then - should create client without error; class config stays None
        assert client is not None
        assert HTTPConnectionPool._httpx_client is client

        # Cleanup
        await HTTPConnectionPool.close()

    @pytest.mark.asyncio
    async def test_configure_sets_performance_config(self) -> None:
        """Test that configure() sets the performance config on the class."""
        # Given
        custom_config: dict[str, object] = {
            "concurrency": {
                "connection_pool": {
                    "max_connections": 50,
                    "max_keepalive": 10,
                    "keepalive_expiry": 15,
                }
            }
        }

        # When
        HTTPConnectionPool.configure(custom_config)

        # Then
        assert HTTPConnectionPool._performance_config is custom_config

        # Cleanup
        await HTTPConnectionPool.close()

    @pytest.mark.asyncio
    async def test_configure_closes_existing_client(self, mock_httpx: Mock) -> None:
        """Test that configure() closes any existing open client."""
        # Given - create a client first
        HTTPConnectionPool.get_httpx_client()
        assert HTTPConnectionPool._httpx_client is not None

        # When
        HTTPConnectionPool.configure({"concurrency": {"connection_pool": {}}})

        # Then - existing client should be closed synchronously and cleared
        mock_httpx.close.assert_called_once()
        assert HTTPConnectionPool._httpx_client is None

        # Cleanup
        await HTTPConnectionPool.close()

    @pytest.mark.asyncio
    async def test_configure_no_client_does_not_error(self) -> None:
        """Test that configure() works when no client exists yet."""
        # Given - no client created
        assert HTTPConnectionPool._httpx_client is None

        # When - should not raise
        HTTPConnectionPool.configure({"concurrency": {"connection_pool": {}}})

        # Then
        assert HTTPConnectionPool._performance_config is not None

        # Cleanup
        await HTTPConnectionPool.close()

    @pytest.mark.asyncio
    async def test_close_client(self) -> None:
        """Test closing the HTTP client."""
        # Given
        client = HTTPConnectionPool.get_httpx_client()
        assert client is not None

        # When
        await HTTPConnectionPool.close()

        # Then
        assert HTTPConnectionPool._httpx_client is None

    @pytest.mark.asyncio
    async def test_close_already_closed(self) -> None:
        """Test closing when client is already closed."""
        # Given - no client created yet
        assert HTTPConnectionPool._httpx_client is None

        # When - should not raise exception
        await HTTPConnectionPool.close()

        # Then
        assert HTTPConnectionPool._httpx_client is None

    @pytest.mark.asyncio
    async def test_get_httpx_client_recreates_after_close(
        self, mock_httpx: Mock
    ) -> None:
        """Test that get_httpx_client recreates client after it's been closed."""
        # Given - create and close a client
        client1 = HTTPConnectionPool.get_httpx_client()
        await client1.aclose()
        # Simulate closed state by changing the mock behavior
        mock_httpx.is_closed = True

        # Reset the class variable to force recreation
        HTTPConnectionPool._httpx_client = None

        # When - get client again (should create new client)
        client2 = HTTPConnectionPool.get_httpx_client()

        # Then - should create new client
        assert client2 is not None

        # Cleanup
        await HTTPConnectionPool.close()


class ConcreteModel(ModelInterface):
    """Concrete implementation for testing."""

    @property
    def name(self) -> str:
        return "test-model"

    async def generate_response(self, message: str, role_prompt: str) -> str:
        return "Test response"


class TestModelInterface:
    """Test ModelInterface abstract base class."""

    def test_model_interface_cannot_be_instantiated(self) -> None:
        """Test that ModelInterface cannot be instantiated directly."""
        # When/Then - abstract class cannot be instantiated
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            ModelInterface()  # type: ignore[abstract]

    def test_model_interface_initialization(self) -> None:
        """Test ModelInterface initialization."""
        # When
        model = ConcreteModel()

        # Then
        assert model._last_usage is None
        assert model._conversation_history == []

    def test_conversation_history_methods(self) -> None:
        """Test conversation history management."""
        # Given
        model = ConcreteModel()

        # When
        model.add_to_conversation("user", "Hello")
        model.add_to_conversation("assistant", "Hi there")

        # Then
        history = model.get_conversation_history()
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "Hello"}
        assert history[1] == {"role": "assistant", "content": "Hi there"}

        # Test history is a copy (not reference)
        history.append({"role": "user", "content": "Modified"})
        assert len(model.get_conversation_history()) == 2

    def test_clear_conversation_history(self) -> None:
        """Test clearing conversation history."""
        # Given
        model = ConcreteModel()
        model.add_to_conversation("user", "Hello")
        assert len(model._conversation_history) == 1

        # When
        model.clear_conversation_history()

        # Then
        assert len(model._conversation_history) == 0
        assert model.get_conversation_history() == []

    def test_usage_recording(self) -> None:
        """Test usage metrics recording."""
        # Given
        model = ConcreteModel()

        # When
        model._record_usage(
            input_tokens=100,
            output_tokens=50,
            duration_ms=1500,
            cost_usd=0.01,
            model_name="custom-model",
        )

        # Then
        usage = model.get_last_usage()
        assert usage is not None
        assert usage["input_tokens"] == 100
        assert usage["output_tokens"] == 50
        assert usage["total_tokens"] == 150
        assert usage["cost_usd"] == 0.01
        assert usage["duration_ms"] == 1500
        assert usage["model"] == "custom-model"

    def test_usage_recording_default_model_name(self) -> None:
        """Test usage recording with default model name."""
        # Given
        model = ConcreteModel()

        # When
        model._record_usage(
            input_tokens=100, output_tokens=50, duration_ms=1500, cost_usd=0.01
        )

        # Then
        usage = model.get_last_usage()
        assert usage is not None
        assert usage["model"] == "test-model"  # Should use model.name

    def test_get_last_usage_none_initially(self) -> None:
        """Test that get_last_usage returns None initially."""
        # Given
        model = ConcreteModel()

        # When
        usage = model.get_last_usage()

        # Then
        assert usage is None

"""Tests for base model infrastructure."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from llm_orc.models.base import HTTPConnectionPool, ModelInterface


@pytest.fixture(autouse=True)
def mock_httpx():
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
    async def test_config_loading_fallback(self) -> None:
        """Test fallback when configuration loading fails."""
        # Reset singleton to ensure clean state
        HTTPConnectionPool._instance = None
        HTTPConnectionPool._httpx_client = None
        HTTPConnectionPool._performance_config = None

        # Given - mock config loading failure at import time
        with patch(
            "llm_orc.core.config.config_manager.ConfigurationManager"
        ) as mock_config:
            mock_config.side_effect = Exception("Config load failed")

            # When
            client = HTTPConnectionPool.get_httpx_client()

            # Then - should use fallback config
            assert client is not None
            assert HTTPConnectionPool._performance_config is not None
            assert (
                HTTPConnectionPool._performance_config["concurrency"][
                    "connection_pool"
                ]["max_connections"]
                == 100
            )

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
    async def test_get_httpx_client_recreates_after_close(self, mock_httpx) -> None:
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

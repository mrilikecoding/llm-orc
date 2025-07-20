"""Base classes and shared infrastructure for LLM models."""

from abc import ABC, abstractmethod
from typing import Any


class HTTPConnectionPool:
    """Manages shared HTTP connections for better performance."""

    _instance: "HTTPConnectionPool | None" = None
    _httpx_client: Any = None  # httpx.AsyncClient
    _performance_config: dict[str, Any] | None = None

    def __new__(cls) -> "HTTPConnectionPool":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_httpx_client(cls) -> Any:
        """Get or create a shared httpx client with connection pooling."""
        # Ensure singleton instance exists
        if cls._instance is None:
            cls._instance = cls()

        if cls._httpx_client is None or cls._httpx_client.is_closed:
            import httpx

            # Load performance configuration
            if cls._performance_config is None:
                try:
                    from llm_orc.config import ConfigurationManager

                    config_manager = ConfigurationManager()
                    cls._performance_config = config_manager.load_performance_config()
                except Exception:
                    # Fallback to defaults if configuration loading fails
                    cls._performance_config = {
                        "concurrency": {
                            "connection_pool": {
                                "max_connections": 100,
                                "max_keepalive": 20,
                                "keepalive_expiry": 30,
                            }
                        }
                    }

            # Get connection pool settings from configuration
            pool_config = cls._performance_config.get("concurrency", {}).get(
                "connection_pool", {}
            )

            # Configure connection pooling for better performance
            limits = httpx.Limits(
                max_connections=pool_config.get("max_connections", 100),
                max_keepalive_connections=pool_config.get("max_keepalive", 20),
                keepalive_expiry=pool_config.get("keepalive_expiry", 30.0),
            )

            timeout = httpx.Timeout(
                connect=10.0,  # Connection timeout
                read=30.0,  # Read timeout
                write=10.0,  # Write timeout
                pool=5.0,  # Pool timeout
            )

            cls._httpx_client = httpx.AsyncClient(
                limits=limits,
                timeout=timeout,
                headers={
                    "User-Agent": "llm-orc/1.0",
                },
            )

        return cls._httpx_client

    @classmethod
    async def close(cls) -> None:
        """Close the shared HTTP client."""
        if cls._httpx_client is not None and not cls._httpx_client.is_closed:
            await cls._httpx_client.aclose()
            cls._httpx_client = None


class ModelInterface(ABC):
    """Abstract interface for LLM models."""

    def __init__(self) -> None:
        self._last_usage: dict[str, Any] | None = None
        self._conversation_history: list[dict[str, str]] = []

    @property
    @abstractmethod
    def name(self) -> str:
        """Model name identifier."""
        pass

    @abstractmethod
    async def generate_response(self, message: str, role_prompt: str) -> str:
        """Generate a response from the model."""
        pass

    def get_conversation_history(self) -> list[dict[str, str]]:
        """Get the conversation history."""
        return self._conversation_history.copy()

    def clear_conversation_history(self) -> None:
        """Clear the conversation history."""
        self._conversation_history.clear()

    def add_to_conversation(self, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        self._conversation_history.append({"role": role, "content": content})

    def get_last_usage(self) -> dict[str, Any] | None:
        """Get usage metrics from the last API call."""
        return self._last_usage

    def _record_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        duration_ms: int,
        cost_usd: float = 0.0,
        model_name: str = "",
    ) -> None:
        """Record usage metrics for the last API call."""
        self._last_usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost_usd": cost_usd,
            "duration_ms": duration_ms,
            "model": model_name or self.name,
        }

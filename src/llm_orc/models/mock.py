"""Mock model for testing without external services."""

from llm_orc.models.base import ModelInterface


class MockModel(ModelInterface):
    """Mock model implementation for testing."""

    def __init__(self, model_name: str = "mock") -> None:
        """Initialize mock model.

        Args:
            model_name: Name identifier for the mock model
        """
        super().__init__()
        self._model_name = model_name

    @property
    def name(self) -> str:
        """Model name identifier."""
        return self._model_name

    async def generate_response(self, message: str, role_prompt: str) -> str:
        """Generate a mock response echoing input with keywords."""
        return (
            f"Analysis of the data shows interesting patterns "
            f"and trends. The centrality metrics reveal key "
            f"structures in the network. "
            f"Context: {str(message)[:100]}"
        )

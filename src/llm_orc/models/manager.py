"""Model manager for LLM Orchestra."""

from typing import Any

from llm_orc.models.anthropic import ClaudeModel, ClaudeCLIModel, OAuthClaudeModel
from llm_orc.models.base import ModelInterface


class ModelManager:
    """Manages model instances and selection."""

    def __init__(self) -> None:
        self.models: dict[str, ModelInterface] = {}

    def register_model(self, key: str, model: ModelInterface) -> None:
        """Register a model instance."""
        self.models[key] = model

    def register_oauth_claude_model(
        self,
        key: str,
        access_token: str,
        refresh_token: str | None = None,
        client_id: str | None = None,
        model: str = "claude-3-5-sonnet-20241022",
        credential_storage: Any = None,
        provider_key: str | None = None,
        expires_at: int | None = None,
    ) -> None:
        """Register an OAuth-authenticated Claude model."""
        oauth_model = OAuthClaudeModel(
            access_token=access_token,
            refresh_token=refresh_token,
            client_id=client_id,
            model=model,
            credential_storage=credential_storage,
            provider_key=provider_key,
            expires_at=expires_at,
        )
        self.models[key] = oauth_model

    def register_claude_model(
        self, key: str, api_key: str, model: str = "claude-3-5-sonnet-20241022"
    ) -> None:
        """Register an API key-authenticated Claude model."""
        claude_model = ClaudeModel(api_key=api_key, model=model)
        self.models[key] = claude_model

    def register_claude_cli_model(
        self, key: str, claude_path: str, model: str = "claude-3-5-sonnet-20241022"
    ) -> None:
        """Register a Claude CLI model."""
        claude_cli_model = ClaudeCLIModel(claude_path=claude_path, model=model)
        self.models[key] = claude_cli_model

    def get_model(self, key: str) -> ModelInterface:
        """Retrieve a registered model."""
        if key not in self.models:
            raise KeyError(f"Model '{key}' not found")
        return self.models[key]

    def list_models(self) -> dict[str, str]:
        """List all registered models."""
        return {key: model.name for key, model in self.models.items()}

    def get_oauth_models(self) -> dict[str, OAuthClaudeModel]:
        """Get all registered OAuth-authenticated models."""
        return {
            key: model
            for key, model in self.models.items()
            if isinstance(model, OAuthClaudeModel)
        }

    def get_api_key_models(self) -> dict[str, ClaudeModel]:
        """Get all registered API key-authenticated models."""
        return {
            key: model
            for key, model in self.models.items()
            if isinstance(model, ClaudeModel)
        }

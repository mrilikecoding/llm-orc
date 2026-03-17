"""Unit tests for provider status Pydantic models."""

from llm_orc.providers.status_types import (
    AgentRunnability,
    AgentStatus,
    CloudProviderStatus,
    EndpointStatus,
    EnsembleRunnability,
    OllamaProviderStatus,
    OpenAICompatibleStatus,
)


class TestAgentStatus:
    """Tests for AgentStatus enum."""

    def test_available_value(self) -> None:
        """Available status has correct string value."""
        assert AgentStatus.AVAILABLE.value == "available"

    def test_missing_profile_value(self) -> None:
        """Missing profile status has correct string value."""
        assert AgentStatus.MISSING_PROFILE.value == "missing_profile"

    def test_provider_unavailable_value(self) -> None:
        """Provider unavailable status has correct string value."""
        assert AgentStatus.PROVIDER_UNAVAILABLE.value == "provider_unavailable"

    def test_model_unavailable_value(self) -> None:
        """Model unavailable status has correct string value."""
        assert AgentStatus.MODEL_UNAVAILABLE.value == "model_unavailable"


class TestOllamaProviderStatus:
    """Tests for OllamaProviderStatus model."""

    def test_default_construction(self) -> None:
        """Default construction has empty models and zero count."""
        status = OllamaProviderStatus(available=False)
        assert status.available is False
        assert status.models == []
        assert status.model_count == 0
        assert status.reason == ""

    def test_available_with_models(self) -> None:
        """Available status with models populates correctly."""
        status = OllamaProviderStatus(
            available=True,
            models=["llama3:latest", "mistral:latest"],
            model_count=2,
        )
        assert status.available is True
        assert len(status.models) == 2
        assert status.model_count == 2

    def test_model_dump_roundtrip(self) -> None:
        """Serialization produces expected dict shape."""
        status = OllamaProviderStatus(
            available=True,
            models=["llama3:latest"],
            model_count=1,
        )
        dumped = status.model_dump()
        assert dumped == {
            "available": True,
            "models": ["llama3:latest"],
            "model_count": 1,
            "reason": "",
        }

    def test_unavailable_dump_matches_existing_shape(self) -> None:
        """Unavailable dump matches the existing dict shape."""
        status = OllamaProviderStatus(
            available=False,
            reason="Ollama not reachable: ConnectError",
        )
        dumped = status.model_dump()
        assert dumped["available"] is False
        assert dumped["reason"] == "Ollama not reachable: ConnectError"
        assert dumped["models"] == []


class TestCloudProviderStatus:
    """Tests for CloudProviderStatus model."""

    def test_configured(self) -> None:
        """Configured provider is available."""
        status = CloudProviderStatus(available=True, reason="configured")
        assert status.available is True
        assert status.reason == "configured"

    def test_not_configured(self) -> None:
        """Unconfigured provider is not available."""
        status = CloudProviderStatus(available=False, reason="not configured")
        dumped = status.model_dump()
        assert dumped == {"available": False, "reason": "not configured"}


class TestEndpointStatus:
    """Tests for EndpointStatus model."""

    def test_default_construction(self) -> None:
        """Default construction has empty lists."""
        status = EndpointStatus(base_url="http://localhost:11434/v1", available=True)
        assert status.models == []
        assert status.profiles == []
        assert status.reason == ""

    def test_full_construction(self) -> None:
        """Full construction populates all fields."""
        status = EndpointStatus(
            base_url="http://localhost:11434/v1",
            available=True,
            models=["llama3", "mistral"],
            profiles=["local-fast", "local-smart"],
        )
        dumped = status.model_dump()
        assert dumped["base_url"] == "http://localhost:11434/v1"
        assert len(dumped["models"]) == 2
        assert len(dumped["profiles"]) == 2


class TestOpenAICompatibleStatus:
    """Tests for OpenAICompatibleStatus model."""

    def test_default_construction(self) -> None:
        """Default construction has empty endpoints and models."""
        status = OpenAICompatibleStatus(available=False)
        assert status.endpoints == []
        assert status.models == []
        assert status.model_count == 0

    def test_with_endpoints(self) -> None:
        """Status with endpoints aggregates correctly."""
        endpoint = EndpointStatus(
            base_url="http://localhost:11434/v1",
            available=True,
            models=["llama3", "mistral"],
            profiles=["local-fast"],
        )
        status = OpenAICompatibleStatus(
            available=True,
            endpoints=[endpoint],
            models=["llama3", "mistral"],
            model_count=2,
        )
        dumped = status.model_dump()
        assert dumped["available"] is True
        assert len(dumped["endpoints"]) == 1
        assert dumped["model_count"] == 2


class TestAgentRunnability:
    """Tests for AgentRunnability model."""

    def test_default_available(self) -> None:
        """Default status is available."""
        agent = AgentRunnability(name="agent1", profile="fast")
        assert agent.status == AgentStatus.AVAILABLE
        assert agent.provider == ""
        assert agent.alternatives == []

    def test_missing_profile(self) -> None:
        """Missing profile status with alternatives."""
        agent = AgentRunnability(
            name="agent1",
            profile="nonexistent",
            status=AgentStatus.MISSING_PROFILE,
            alternatives=["local-fast"],
        )
        dumped = agent.model_dump()
        assert dumped["status"] == "missing_profile"
        assert dumped["alternatives"] == ["local-fast"]

    def test_dump_matches_existing_dict_shape(self) -> None:
        """model_dump matches the existing dict shape from _check_agent_runnable."""
        agent = AgentRunnability(
            name="agent1",
            profile="fast",
            provider="ollama",
            status=AgentStatus.AVAILABLE,
            alternatives=[],
        )
        dumped = agent.model_dump()
        assert dumped == {
            "name": "agent1",
            "profile": "fast",
            "provider": "ollama",
            "status": "available",
            "alternatives": [],
        }


class TestEnsembleRunnability:
    """Tests for EnsembleRunnability model."""

    def test_runnable_ensemble(self) -> None:
        """Runnable ensemble with all agents available."""
        agent = AgentRunnability(
            name="agent1",
            profile="fast",
            provider="ollama",
        )
        ensemble = EnsembleRunnability(
            ensemble="test-ensemble",
            runnable=True,
            agents=[agent],
        )
        dumped = ensemble.model_dump()
        assert dumped["ensemble"] == "test-ensemble"
        assert dumped["runnable"] is True
        assert len(dumped["agents"]) == 1
        assert dumped["agents"][0]["status"] == "available"

    def test_not_runnable_ensemble(self) -> None:
        """Non-runnable ensemble with unavailable agent."""
        agent = AgentRunnability(
            name="agent1",
            profile="missing",
            status=AgentStatus.MISSING_PROFILE,
        )
        ensemble = EnsembleRunnability(
            ensemble="test-ensemble",
            runnable=False,
            agents=[agent],
        )
        assert ensemble.runnable is False
        assert ensemble.agents[0].status == AgentStatus.MISSING_PROFILE

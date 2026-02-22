"""Tests for the ensembles API endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient


class TestEnsemblesAPI:
    """Tests for /api/ensembles endpoints."""

    def test_list_ensembles_returns_list(self, client: TestClient) -> None:
        """Test that GET /api/ensembles returns a list."""
        with patch("llm_orc.web.api.ensembles.get_orchestra_service") as mock_get_svc:
            mock_service = MagicMock()
            mock_service.read_ensembles = AsyncMock(
                return_value=[
                    {"name": "test-ensemble", "description": "Test", "source": "local"}
                ]
            )
            mock_get_svc.return_value = mock_service

            response = client.get("/api/ensembles")

            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            assert len(data) == 1
            assert data[0]["name"] == "test-ensemble"

    def test_get_ensemble_returns_detail(self, client: TestClient) -> None:
        """Test that GET /api/ensembles/{name} returns ensemble detail."""
        with patch("llm_orc.web.api.ensembles.get_orchestra_service") as mock_get_svc:
            mock_service = MagicMock()
            mock_service.read_ensemble = AsyncMock(
                return_value={
                    "name": "test-ensemble",
                    "description": "Test ensemble",
                    "agents": [{"name": "agent1", "model_profile": "default"}],
                }
            )
            mock_get_svc.return_value = mock_service

            response = client.get("/api/ensembles/test-ensemble")

            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "test-ensemble"
            assert "agents" in data

    def test_get_ensemble_not_found(self, client: TestClient) -> None:
        """Test that GET /api/ensembles/{name} returns 404 for missing ensemble."""
        with patch("llm_orc.web.api.ensembles.get_orchestra_service") as mock_get_svc:
            mock_service = MagicMock()
            mock_service.read_ensemble = AsyncMock(return_value=None)
            mock_get_svc.return_value = mock_service

            response = client.get("/api/ensembles/nonexistent")

            assert response.status_code == 404

    def test_execute_ensemble_returns_result(self, client: TestClient) -> None:
        """Test that POST /api/ensembles/{name}/execute returns result."""
        with patch("llm_orc.web.api.ensembles.get_orchestra_service") as mock_get_svc:
            mock_service = MagicMock()
            mock_service.invoke = AsyncMock(
                return_value={
                    "status": "success",
                    "results": {"agent1": {"response": "Test output"}},
                }
            )
            mock_get_svc.return_value = mock_service

            response = client.post(
                "/api/ensembles/test-ensemble/execute",
                json={"input": "Test input"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"

    def test_validate_ensemble_returns_validation(self, client: TestClient) -> None:
        """Test that POST /api/ensembles/{name}/validate returns validation."""
        with patch("llm_orc.web.api.ensembles.get_orchestra_service") as mock_get_svc:
            mock_service = MagicMock()
            mock_service.validate_ensemble = AsyncMock(
                return_value={"valid": True, "details": {"errors": []}}
            )
            mock_get_svc.return_value = mock_service

            response = client.post("/api/ensembles/test-ensemble/validate")

            assert response.status_code == 200
            data = response.json()
            assert data["valid"] is True

    def test_check_runnable_returns_status(self, client: TestClient) -> None:
        """Test that GET /api/ensembles/{name}/runnable returns status."""
        with patch("llm_orc.web.api.ensembles.get_orchestra_service") as mock_get_svc:
            mock_service = MagicMock()
            mock_service.check_ensemble_runnable = AsyncMock(
                return_value={
                    "ensemble": "test-ensemble",
                    "runnable": True,
                    "agents": [
                        {
                            "name": "agent1",
                            "profile": "fast",
                            "provider": "ollama",
                            "status": "available",
                            "alternatives": [],
                        }
                    ],
                }
            )
            mock_get_svc.return_value = mock_service

            response = client.get("/api/ensembles/test-ensemble/runnable")

            assert response.status_code == 200
            data = response.json()
            assert data["ensemble"] == "test-ensemble"
            assert data["runnable"] is True
            assert len(data["agents"]) == 1
            assert data["agents"][0]["status"] == "available"

    def test_check_runnable_with_unavailable_agents(self, client: TestClient) -> None:
        """Test that runnable endpoint shows unavailable agents correctly."""
        with patch("llm_orc.web.api.ensembles.get_orchestra_service") as mock_get_svc:
            mock_service = MagicMock()
            mock_service.check_ensemble_runnable = AsyncMock(
                return_value={
                    "ensemble": "test-ensemble",
                    "runnable": False,
                    "agents": [
                        {
                            "name": "agent1",
                            "profile": "cloud-profile",
                            "provider": "anthropic-api",
                            "status": "provider_unavailable",
                            "alternatives": ["llama3:latest", "gemma2:9b"],
                        }
                    ],
                }
            )
            mock_get_svc.return_value = mock_service

            response = client.get("/api/ensembles/test-ensemble/runnable")

            assert response.status_code == 200
            data = response.json()
            assert data["runnable"] is False
            assert data["agents"][0]["status"] == "provider_unavailable"
            assert len(data["agents"][0]["alternatives"]) == 2

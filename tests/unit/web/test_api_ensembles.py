"""Tests for the ensembles API endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from llm_orc.web.server import create_app


class TestEnsemblesAPI:
    """Tests for /api/ensembles endpoints."""

    def test_list_ensembles_returns_list(self) -> None:
        """Test that GET /api/ensembles returns a list."""
        app = create_app()
        client = TestClient(app)

        with patch("llm_orc.web.api.ensembles.get_mcp_server") as mock_get_mcp:
            mock_server = MagicMock()
            mock_server._read_ensembles_resource = AsyncMock(
                return_value=[
                    {"name": "test-ensemble", "description": "Test", "source": "local"}
                ]
            )
            mock_get_mcp.return_value = mock_server

            response = client.get("/api/ensembles")

            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            assert len(data) == 1
            assert data[0]["name"] == "test-ensemble"

    def test_get_ensemble_returns_detail(self) -> None:
        """Test that GET /api/ensembles/{name} returns ensemble detail."""
        app = create_app()
        client = TestClient(app)

        with patch("llm_orc.web.api.ensembles.get_mcp_server") as mock_get_mcp:
            mock_server = MagicMock()
            mock_server._read_ensemble_resource = AsyncMock(
                return_value={
                    "name": "test-ensemble",
                    "description": "Test ensemble",
                    "agents": [{"name": "agent1", "model_profile": "default"}],
                }
            )
            mock_get_mcp.return_value = mock_server

            response = client.get("/api/ensembles/test-ensemble")

            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "test-ensemble"
            assert "agents" in data

    def test_get_ensemble_not_found(self) -> None:
        """Test that GET /api/ensembles/{name} returns 404 for missing ensemble."""
        app = create_app()
        client = TestClient(app)

        with patch("llm_orc.web.api.ensembles.get_mcp_server") as mock_get_mcp:
            mock_server = MagicMock()
            mock_server._read_ensemble_resource = AsyncMock(return_value=None)
            mock_get_mcp.return_value = mock_server

            response = client.get("/api/ensembles/nonexistent")

            assert response.status_code == 404

    def test_execute_ensemble_returns_result(self) -> None:
        """Test that POST /api/ensembles/{name}/execute returns result."""
        app = create_app()
        client = TestClient(app)

        with patch("llm_orc.web.api.ensembles.get_mcp_server") as mock_get_mcp:
            mock_server = MagicMock()
            mock_server._invoke_tool = AsyncMock(
                return_value={
                    "status": "success",
                    "results": {"agent1": {"response": "Test output"}},
                }
            )
            mock_get_mcp.return_value = mock_server

            response = client.post(
                "/api/ensembles/test-ensemble/execute",
                json={"input": "Test input"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"

    def test_validate_ensemble_returns_validation(self) -> None:
        """Test that POST /api/ensembles/{name}/validate returns validation."""
        app = create_app()
        client = TestClient(app)

        with patch("llm_orc.web.api.ensembles.get_mcp_server") as mock_get_mcp:
            mock_server = MagicMock()
            mock_server._validate_ensemble_tool = AsyncMock(
                return_value={"valid": True, "details": {"errors": []}}
            )
            mock_get_mcp.return_value = mock_server

            response = client.post("/api/ensembles/test-ensemble/validate")

            assert response.status_code == 200
            data = response.json()
            assert data["valid"] is True

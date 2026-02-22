"""Tests for the artifacts API endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient


class TestArtifactsAPI:
    """Tests for /api/artifacts endpoints."""

    def test_list_artifacts_returns_list(self, client: TestClient) -> None:
        """GET /api/artifacts returns all ensembles with artifacts."""
        with patch("llm_orc.web.api.artifacts.get_mcp_server") as mock_get_mcp:
            mock_server = MagicMock()
            mock_server.artifact_manager.list_ensembles.return_value = [
                {"ensemble": "my-ensemble", "count": 3}
            ]
            mock_get_mcp.return_value = mock_server

            response = client.get("/api/artifacts")

            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            assert data[0]["ensemble"] == "my-ensemble"

    def test_get_ensemble_artifacts_returns_list(self, client: TestClient) -> None:
        """GET /api/artifacts/{ensemble} returns artifacts for that ensemble."""
        with patch("llm_orc.web.api.artifacts.get_mcp_server") as mock_get_mcp:
            mock_server = MagicMock()
            mock_server._read_artifacts_resource = AsyncMock(
                return_value=[{"id": "abc123", "timestamp": "2025-01-01"}]
            )
            mock_get_mcp.return_value = mock_server

            response = client.get("/api/artifacts/my-ensemble")

            assert response.status_code == 200
            data = response.json()
            assert data[0]["id"] == "abc123"

    def test_get_artifact_returns_detail(self, client: TestClient) -> None:
        """GET /api/artifacts/{ensemble}/{artifact_id} returns artifact detail."""
        with patch("llm_orc.web.api.artifacts.get_mcp_server") as mock_get_mcp:
            mock_server = MagicMock()
            mock_server._read_artifact_resource = AsyncMock(
                return_value={"id": "abc123", "status": "completed"}
            )
            mock_get_mcp.return_value = mock_server

            response = client.get("/api/artifacts/my-ensemble/abc123")

            assert response.status_code == 200
            data = response.json()
            assert data["id"] == "abc123"

    def test_get_artifact_returns_404_when_not_found(
        self, client: TestClient
    ) -> None:
        """GET /api/artifacts/{ensemble}/{artifact_id} returns 404 when missing."""
        with patch("llm_orc.web.api.artifacts.get_mcp_server") as mock_get_mcp:
            mock_server = MagicMock()
            mock_server._read_artifact_resource = AsyncMock(return_value=None)
            mock_get_mcp.return_value = mock_server

            response = client.get("/api/artifacts/my-ensemble/missing")

            assert response.status_code == 404

    def test_delete_artifact_returns_result(self, client: TestClient) -> None:
        """DELETE /api/artifacts/{ensemble}/{artifact_id} deletes the artifact."""
        with patch("llm_orc.web.api.artifacts.get_mcp_server") as mock_get_mcp:
            mock_server = MagicMock()
            mock_server._delete_artifact_tool = AsyncMock(
                return_value={"status": "deleted", "artifact_id": "my-ensemble/abc123"}
            )
            mock_get_mcp.return_value = mock_server

            response = client.delete("/api/artifacts/my-ensemble/abc123")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "deleted"

    def test_analyze_artifact_returns_result(self, client: TestClient) -> None:
        """POST /api/artifacts/{artifact_id}/analyze returns analysis."""
        with patch("llm_orc.web.api.artifacts.get_mcp_server") as mock_get_mcp:
            mock_server = MagicMock()
            mock_server._analyze_execution_tool = AsyncMock(
                return_value={"summary": "2 agents ran successfully"}
            )
            mock_get_mcp.return_value = mock_server

            response = client.post("/api/artifacts/abc123/analyze")

            assert response.status_code == 200
            data = response.json()
            assert "summary" in data

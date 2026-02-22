"""Tests for the scripts API endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient


class TestScriptsAPI:
    """Tests for /api/scripts endpoints."""

    def test_list_scripts_returns_result(self, client: TestClient) -> None:
        """GET /api/scripts returns the script handler's result."""
        with patch("llm_orc.web.api.scripts.get_mcp_server") as mock_get_mcp:
            mock_server = MagicMock()
            mock_server._script_handler.list_scripts = AsyncMock(
                return_value={"scripts": {"data_transform": ["json_extract"]}}
            )
            mock_get_mcp.return_value = mock_server

            response = client.get("/api/scripts")

            assert response.status_code == 200
            data = response.json()
            assert "scripts" in data

    def test_get_script_returns_detail(self, client: TestClient) -> None:
        """GET /api/scripts/{category}/{name} returns script detail."""
        with patch("llm_orc.web.api.scripts.get_mcp_server") as mock_get_mcp:
            mock_server = MagicMock()
            mock_server._script_handler.get_script = AsyncMock(
                return_value={"name": "json_extract", "category": "data_transform"}
            )
            mock_get_mcp.return_value = mock_server

            response = client.get("/api/scripts/data_transform/json_extract")

            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "json_extract"

    def test_test_script_returns_result(self, client: TestClient) -> None:
        """POST /api/scripts/{category}/{name}/test returns test output."""
        with patch("llm_orc.web.api.scripts.get_mcp_server") as mock_get_mcp:
            mock_server = MagicMock()
            mock_server._script_handler.test_script = AsyncMock(
                return_value={"output": "42", "success": True}
            )
            mock_get_mcp.return_value = mock_server

            response = client.post(
                "/api/scripts/data_transform/json_extract/test",
                json={"input": '{"value": 42}'},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

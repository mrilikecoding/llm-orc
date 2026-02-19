"""Tests for the profiles API endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient


class TestProfilesAPI:
    """Tests for /api/profiles endpoints."""

    def test_list_profiles_returns_list(self, client: TestClient) -> None:
        """Test that GET /api/profiles returns a list."""
        with patch("llm_orc.web.api.profiles.get_mcp_server") as mock_get_mcp:
            mock_server = MagicMock()
            mock_server._read_profiles_resource = AsyncMock(
                return_value=[
                    {"name": "default", "provider": "ollama", "model": "llama3"}
                ]
            )
            mock_get_mcp.return_value = mock_server

            response = client.get("/api/profiles")

            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            assert len(data) == 1
            assert data[0]["name"] == "default"

    def test_create_profile_success(self, client: TestClient) -> None:
        """Test that POST /api/profiles creates a profile."""
        with patch("llm_orc.web.api.profiles.get_mcp_server") as mock_get_mcp:
            mock_server = MagicMock()
            mock_server._create_profile_tool = AsyncMock(
                return_value={"status": "created", "name": "new-profile"}
            )
            mock_get_mcp.return_value = mock_server

            response = client.post(
                "/api/profiles",
                json={"name": "new-profile", "provider": "ollama", "model": "gemma2"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "created"

    def test_delete_profile_success(self, client: TestClient) -> None:
        """Test that DELETE /api/profiles/{name} deletes a profile."""
        with patch("llm_orc.web.api.profiles.get_mcp_server") as mock_get_mcp:
            mock_server = MagicMock()
            mock_server._delete_profile_tool = AsyncMock(
                return_value={"status": "deleted", "name": "old-profile"}
            )
            mock_get_mcp.return_value = mock_server

            response = client.delete("/api/profiles/old-profile")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "deleted"

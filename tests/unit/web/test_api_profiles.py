"""Tests for the profiles API endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient


class TestProfilesAPI:
    """Tests for /api/profiles endpoints."""

    def test_list_profiles_returns_list(self, client: TestClient) -> None:
        """Test that GET /api/profiles returns a list."""
        with patch("llm_orc.web.api.profiles.get_orchestra_service") as mock_get_svc:
            mock_service = MagicMock()
            mock_service.read_profiles = AsyncMock(
                return_value=[
                    {"name": "default", "provider": "ollama", "model": "llama3"}
                ]
            )
            mock_get_svc.return_value = mock_service

            response = client.get("/api/profiles")

            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            assert len(data) == 1
            assert data[0]["name"] == "default"

    def test_create_profile_success(self, client: TestClient) -> None:
        """Test that POST /api/profiles creates a profile."""
        with patch("llm_orc.web.api.profiles.get_orchestra_service") as mock_get_svc:
            mock_service = MagicMock()
            mock_service.create_profile = AsyncMock(
                return_value={"status": "created", "name": "new-profile"}
            )
            mock_get_svc.return_value = mock_service

            response = client.post(
                "/api/profiles",
                json={"name": "new-profile", "provider": "ollama", "model": "gemma2"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "created"

    def test_update_profile_success(self, client: TestClient) -> None:
        """Test that PUT /api/profiles/{name} updates a profile."""
        with patch("llm_orc.web.api.profiles.get_orchestra_service") as mock_get_svc:
            mock_service = MagicMock()
            mock_service.update_profile = AsyncMock(
                return_value={"status": "updated", "name": "my-profile"}
            )
            mock_get_svc.return_value = mock_service

            response = client.put(
                "/api/profiles/my-profile",
                json={"model": "llama3.2"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "updated"

    def test_delete_profile_success(self, client: TestClient) -> None:
        """Test that DELETE /api/profiles/{name} deletes a profile."""
        with patch("llm_orc.web.api.profiles.get_orchestra_service") as mock_get_svc:
            mock_service = MagicMock()
            mock_service.delete_profile = AsyncMock(
                return_value={"status": "deleted", "name": "old-profile"}
            )
            mock_get_svc.return_value = mock_service

            response = client.delete("/api/profiles/old-profile")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "deleted"

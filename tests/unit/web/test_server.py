"""Tests for the web server module."""

from fastapi.testclient import TestClient

from llm_orc.web.server import create_app


class TestWebServer:
    """Tests for web server functionality."""

    def test_create_app_returns_fastapi_instance(self) -> None:
        """Test that create_app returns a FastAPI application."""
        from fastapi import FastAPI

        app = create_app()
        assert isinstance(app, FastAPI)

    def test_health_endpoint_returns_ok(self) -> None:
        """Test that /health returns healthy status."""
        app = create_app()
        client = TestClient(app)

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_root_endpoint_returns_api_info(self) -> None:
        """Test that / returns API information."""
        app = create_app()
        client = TestClient(app)

        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "llm-orc"
        assert "version" in data
        assert "endpoints" in data

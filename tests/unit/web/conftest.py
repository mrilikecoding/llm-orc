"""Shared fixtures for web API tests."""

import pytest
from fastapi.testclient import TestClient

from llm_orc.web.server import create_app


@pytest.fixture
def client() -> TestClient:
    """Create a TestClient for the web app."""
    return TestClient(create_app())

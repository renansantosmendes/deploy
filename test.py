import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Hello": "World new version"}


def test_api_route():
    response = client.get("/api")
    assert response.status_code == 200
    assert response.json() == {"key": "value"}
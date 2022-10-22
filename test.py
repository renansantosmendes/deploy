import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from fastapi.testclient import TestClient
from main import app, model, scaler

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Hello": "World"}


# def test_api_route():
#     response = client.get("/api/predict")
#     assert response.status_code == 200
#     assert response.json() == {"key": "value"}
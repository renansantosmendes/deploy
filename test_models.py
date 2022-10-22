import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from fastapi.testclient import TestClient
from main import app, model, scaler

client = TestClient(app)


def test_model_instance():
    assert isinstance(model, RandomForestClassifier)


def test_scaler_instance():
    assert isinstance(scaler, StandardScaler)

"""
Tests for the FastAPI inference app.
Uses TestClient (no running server needed).
The model is loaded from HuggingFace hub if no local model exists.
"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Patch model loading so tests don't need a GPU or trained weights
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def client():
    """Return a TestClient with a mocked model (no real inference needed)."""
    import app.main as main_module

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    # Simulate logits: ham=0.1, spam=0.9
    import torch

    mock_logits = MagicMock()
    mock_logits.logits = torch.tensor([[0.1, 0.9]])
    mock_model.return_value = mock_logits
    mock_model.eval = MagicMock()

    mock_tokenizer.return_value = {
        "input_ids": torch.zeros(1, 128, dtype=torch.long),
        "attention_mask": torch.ones(1, 128, dtype=torch.long),
    }

    main_module._model = mock_model
    main_module._tokenizer = mock_tokenizer
    main_module._model_path = "mock/model"

    with TestClient(main_module.app) as c:
        yield c


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "model" in body


# ---------------------------------------------------------------------------
# Single prediction
# ---------------------------------------------------------------------------
def test_predict_returns_label_and_score(client):
    r = client.post("/predict", json={"text": "Win a FREE prize now!"})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in {"ham", "spam"}
    assert 0.0 <= body["score"] <= 1.0
    assert body["latency_ms"] >= 0


def test_predict_empty_text_rejected(client):
    r = client.post("/predict", json={"text": ""})
    assert r.status_code == 422  # validation error


def test_predict_too_long_text_rejected(client):
    r = client.post("/predict", json={"text": "x" * 1601})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------
def test_batch_predict(client):
    r = client.post(
        "/predict/batch",
        json={"texts": ["Hello friend", "CLAIM your FREE cash now!"]},
    )
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert len(preds) == 2
    for p in preds:
        assert p["label"] in {"ham", "spam"}
        assert 0.0 <= p["score"] <= 1.0


def test_batch_too_many_items_rejected(client):
    r = client.post(
        "/predict/batch",
        json={"texts": ["msg"] * 33},  # max is 32
    )
    assert r.status_code == 422


def test_batch_empty_list_rejected(client):
    r = client.post("/predict/batch", json={"texts": []})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Docs endpoint
# ---------------------------------------------------------------------------
def test_docs_available(client):
    r = client.get("/docs")
    assert r.status_code == 200

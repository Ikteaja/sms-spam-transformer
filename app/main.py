"""
Phase 8 — FastAPI inference server.

Endpoints:
  POST /predict          → classify a single SMS message
  POST /predict/batch    → classify up to 32 messages at once
  GET  /health           → liveness check
  GET  /docs             → auto-generated Swagger UI (built-in)

Run:
  uvicorn app.main:app --reload

The server loads the best model at startup (frozen > best fallback).
Model path can be overridden with env var MODEL_DIR.
"""

import os
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_DEFAULT_MODEL_DIRS = [Path("models/frozen"), Path("models/best")]
MODEL_DIR = Path(os.getenv("MODEL_DIR", ""))

LABEL_MAP = {0: "ham", 1: "spam"}
MAX_LENGTH = 128
MAX_BATCH = 32

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SMS Spam Classifier",
    description="Transfer-learning DistilBERT fine-tuned on the UCI SMS Spam dataset.",
    version="1.0.0",
)

# Loaded at startup
_model: DistilBertForSequenceClassification = None
_tokenizer: DistilBertTokenizerFast = None
_model_path: str = ""


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------
@app.on_event("startup")
def load_model():
    global _model, _tokenizer, _model_path

    candidates = ([MODEL_DIR] if MODEL_DIR.name else []) + _DEFAULT_MODEL_DIRS
    chosen = next((p for p in candidates if p.exists()), None)
    if chosen is None:
        raise RuntimeError(
            "No trained model found. Run 04_train.py (and optionally 06_freeze_tune.py) first."
        )

    _model_path = str(chosen)
    _tokenizer = DistilBertTokenizerFast.from_pretrained(_model_path)
    _model = DistilBertForSequenceClassification.from_pretrained(_model_path)
    _model.eval()
    print(f"Model loaded from: {_model_path}")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1600, example="Congratulations! You won a free iPhone. Click here now!")


class PredictResponse(BaseModel):
    label: str
    score: float = Field(..., description="Confidence for the predicted label (0–1)")
    latency_ms: float


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=MAX_BATCH)


class BatchPredictResponse(BaseModel):
    predictions: List[PredictResponse]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _infer(texts: List[str]) -> List[dict]:
    encoding = _tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = _model(**encoding).logits
    elapsed_ms = (time.perf_counter() - t0) * 1000

    probs = torch.softmax(logits, dim=-1).numpy()
    per_msg_ms = elapsed_ms / len(texts)

    results = []
    for p in probs:
        label_id = int(np.argmax(p))
        results.append(
            {
                "label": LABEL_MAP[label_id],
                "score": float(p[label_id]),
                "latency_ms": round(per_msg_ms, 2),
            }
        )
    return results


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model": _model_path}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    result = _infer([req.text])[0]
    return PredictResponse(**result)


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(req: BatchPredictRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    results = _infer(req.texts)
    return BatchPredictResponse(predictions=[PredictResponse(**r) for r in results])

"""
Smoke tests for the ML pipeline scripts.
Validates that tokenisation + model loading work on a tiny sample
without needing the full dataset or a trained model.
"""

import json
from pathlib import Path

import pytest
import torch


def test_tokeniser_output_shape():
    """Tokeniser must produce correct tensor shapes."""
    from transformers import DistilBertTokenizerFast
    tok = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    enc = tok(
        ["Free prize now!", "Hey how are you?"],
        padding="max_length", truncation=True, max_length=128,
        return_tensors="pt",
    )
    assert enc["input_ids"].shape == (2, 128)
    assert enc["attention_mask"].shape == (2, 128)


def test_model_forward_pass():
    """DistilBERT must produce 2-class logits without error."""
    from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
    tok   = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )
    model.eval()
    enc = tok("test message", return_tensors="pt", truncation=True,
              padding="max_length", max_length=128)
    with torch.no_grad():
        out = model(**enc)
    assert out.logits.shape == (1, 2)


def test_label_map_valid():
    """label_map.json must exist and map 0→ham 1→spam after tokenisation."""
    label_map_path = Path("data/processed/label_map.json")
    if not label_map_path.exists():
        pytest.skip("data/processed not found — run make train first")
    lm = json.loads(label_map_path.read_text())
    assert lm.get("0") == "ham"
    assert lm.get("1") == "spam"


def test_processed_dataset_splits():
    """Processed dataset must have train/val/test splits."""
    from datasets import load_from_disk
    ds_path = Path("data/processed")
    if not ds_path.exists():
        pytest.skip("data/processed not found — run make train first")
    ds = load_from_disk(str(ds_path))
    assert "train" in ds and "val" in ds and "test" in ds
    for split in ["train", "val", "test"]:
        assert len(ds[split]) > 0, f"{split} split is empty"

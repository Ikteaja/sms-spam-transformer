"""
Tests for data quality — run in CI after the data download step.
These tests will SKIP if spam.csv has not been downloaded yet.
"""

from pathlib import Path

import pandas as pd
import pytest

CSV_PATH = Path("data/raw/spam.csv")


@pytest.fixture(scope="module")
def df():
    if not CSV_PATH.exists():
        pytest.skip("spam.csv not found — run make data first")
    raw = pd.read_csv(CSV_PATH, encoding="latin-1")
    if "v1" in raw.columns:
        raw = raw[["v1", "v2"]].rename(columns={"v1": "label", "v2": "text"})
    raw["label"] = raw["label"].str.strip().str.lower()
    return raw


def test_minimum_row_count(df):
    assert len(df) >= 5000, f"Expected >= 5000 rows, got {len(df)}"


def test_no_null_values(df):
    nulls = df[["label", "text"]].isnull().sum()
    assert nulls.sum() == 0, f"Null values found:\n{nulls}"


def test_no_empty_text(df):
    empty = (df["text"].str.strip() == "").sum()
    assert empty == 0, f"{empty} empty text messages found"


def test_valid_labels_only(df):
    invalid = df[~df["label"].isin({"ham", "spam"})]["label"].unique()
    assert len(invalid) == 0, f"Invalid labels found: {invalid}"


def test_both_classes_present(df):
    classes = set(df["label"].unique())
    assert "ham" in classes and "spam" in classes, f"Missing class(es): {classes}"


def test_class_imbalance_within_bounds(df):
    counts = df["label"].value_counts()
    ratio = counts.max() / counts.min()
    # Ratio should be < 20:1 — anything beyond that needs resampling
    assert ratio < 20, f"Extreme class imbalance: {ratio:.1f}:1"


def test_text_length_sanity(df):
    lengths = df["text"].str.len()
    very_short = (lengths < 2).sum()
    assert very_short == 0, f"{very_short} messages shorter than 2 characters"


def test_columns_present(df):
    assert "label" in df.columns
    assert "text" in df.columns

"""
Phase 3 — Tokenise and split.

Steps:
  1. Load spam.csv, encode labels (ham=0, spam=1)
  2. Stratified split → 70 % train / 15 % val / 15 % test
  3. Tokenise with DistilBertTokenizerFast (max_length=128, padding, truncation)
  4. Save HuggingFace Dataset to data/processed/

Output:
  data/processed/{train,val,test}/  (arrow format)
  data/processed/label_map.json
"""

import json
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast

RAW_CSV = Path("data/raw/spam.csv")
PROCESSED_DIR = Path("data/processed")
MODEL_CHECKPOINT = "distilbert-base-uncased"
MAX_LENGTH = 128
RANDOM_SEED = 42


def load_and_clean(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, encoding="latin-1")
        if "v1" in df.columns:
            df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "text"})
    except Exception:
        df = pd.read_csv(path)

    df = df[["label", "text"]].dropna()
    df["label"] = df["label"].str.strip().str.lower()
    df["label_id"] = (df["label"] == "spam").astype(int)
    return df.reset_index(drop=True)


def split(df: pd.DataFrame):
    # 70 / 15 / 15 stratified split
    train_df, tmp_df = train_test_split(
        df, test_size=0.30, stratify=df["label_id"], random_state=RANDOM_SEED
    )
    val_df, test_df = train_test_split(
        tmp_df, test_size=0.50, stratify=tmp_df["label_id"], random_state=RANDOM_SEED
    )
    return train_df, val_df, test_df


def tokenise(dataset: DatasetDict, tokenizer) -> DatasetDict:
    def _tok(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )

    return dataset.map(_tok, batched=True)


def main() -> None:
    if not RAW_CSV.exists():
        print(f"Missing {RAW_CSV} — run 01_download_data.py first.")
        return

    print("Loading data …")
    df = load_and_clean(RAW_CSV)
    print(
        f"  Total rows: {len(df)}  |  spam: {df['label_id'].sum()}  |  ham: {(df['label_id']==0).sum()}"
    )

    print("Splitting 70/15/15 …")
    train_df, val_df, test_df = split(df)
    print(f"  Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    # Build HuggingFace DatasetDict
    raw_ds = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df[["text", "label_id"]], preserve_index=False),
            "val": Dataset.from_pandas(val_df[["text", "label_id"]], preserve_index=False),
            "test": Dataset.from_pandas(test_df[["text", "label_id"]], preserve_index=False),
        }
    )
    # Rename label_id → labels (expected by Trainer)
    raw_ds = raw_ds.rename_column("label_id", "labels")

    print(f"Loading tokenizer: {MODEL_CHECKPOINT} …")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_CHECKPOINT)

    print("Tokenising …")
    tokenised_ds = tokenise(raw_ds, tokenizer)
    tokenised_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    tokenised_ds.save_to_disk(str(PROCESSED_DIR))
    print(f"Saved tokenised dataset → {PROCESSED_DIR}")

    label_map = {"0": "ham", "1": "spam"}
    with open(PROCESSED_DIR / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)
    print("Saved label_map.json")


if __name__ == "__main__":
    main()

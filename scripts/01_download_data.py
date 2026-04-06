"""
Phase 1 — Download the UCI SMS Spam Collection dataset.

Two paths:
  A) Kaggle CLI  (needs ~/.kaggle/kaggle.json)
  B) Direct UCI URL fallback (no credentials needed)

Output: data/raw/spam.csv
"""

import urllib.request
import zipfile
from pathlib import Path

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

KAGGLE_DATASET = "uciml/sms-spam-collection-dataset"
UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/" "00228/smsspamcollection.zip"
OUTPUT_CSV = RAW_DIR / "spam.csv"


def download_via_kaggle() -> bool:
    try:
        import kaggle  # noqa: F401

        print("Kaggle credentials found — downloading via Kaggle API …")
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(KAGGLE_DATASET, path=str(RAW_DIR), unzip=True)
        # The Kaggle zip extracts to spam.csv directly
        if OUTPUT_CSV.exists():
            print(f"Saved: {OUTPUT_CSV}")
            return True
    except Exception as exc:
        print(f"Kaggle download skipped: {exc}")
    return False


def download_via_uci() -> bool:
    import pandas as pd

    zip_path = RAW_DIR / "smsspamcollection.zip"
    print("Downloading from UCI repository …")
    try:
        urllib.request.urlretrieve(UCI_URL, zip_path)
    except Exception as exc:
        print(f"UCI download failed: {exc}")
        return False

    with zipfile.ZipFile(zip_path) as z:
        with z.open("SMSSpamCollection") as f:
            lines = f.read().decode("utf-8", errors="replace").splitlines()

    rows = [line.split("\t", 1) for line in lines if "\t" in line]
    df = pd.DataFrame(rows, columns=["label", "text"])
    df.to_csv(OUTPUT_CSV, index=False)
    zip_path.unlink()
    print(f"Saved {len(df)} rows → {OUTPUT_CSV}")
    return True


def main() -> None:
    if OUTPUT_CSV.exists():
        print(f"{OUTPUT_CSV} already exists — skipping download.")
        return

    if not download_via_kaggle():
        if not download_via_uci():
            raise RuntimeError(
                "Could not download the dataset. " "Place spam.csv manually in data/raw/."
            )


if __name__ == "__main__":
    main()

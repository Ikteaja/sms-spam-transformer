"""
Phase 2 — Exploratory Data Analysis.

Prints:
  • Class distribution (count + percentage)
  • Basic text length statistics per class
  • Top 10 most frequent words per class
  • Saves figures to data/eda_*.png
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RAW_CSV = Path("data/raw/spam.csv")
EDA_DIR = Path("data")
EDA_DIR.mkdir(exist_ok=True)


def load_data() -> pd.DataFrame:
    # Kaggle version has extra unnamed columns; UCI version has only 2.
    try:
        df = pd.read_csv(RAW_CSV, encoding="latin-1")
        # Kaggle CSV has columns: v1, v2, Unnamed: 2, …
        if "v1" in df.columns:
            df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "text"})
    except Exception:
        df = pd.read_csv(RAW_CSV)

    df = df[["label", "text"]].dropna()
    df["label"] = df["label"].str.strip().str.lower()
    return df


def class_distribution(df: pd.DataFrame) -> None:
    counts = df["label"].value_counts()
    pct = df["label"].value_counts(normalize=True) * 100
    print("\n=== Class distribution ===")
    for cls in counts.index:
        print(f"  {cls:5s}: {counts[cls]:5d}  ({pct[cls]:.1f} %)")

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(x=counts.index, y=counts.values, palette=["steelblue", "salmon"], ax=ax)
    ax.set_title("Class distribution")
    ax.set_ylabel("Count")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 30, str(v), ha="center", fontsize=11)
    fig.tight_layout()
    fig.savefig(EDA_DIR / "eda_class_dist.png", dpi=120)
    plt.close(fig)
    print("Saved: data/eda_class_dist.png")


def text_length_stats(df: pd.DataFrame) -> None:
    df = df.copy()
    df["length"] = df["text"].str.len()
    print("\n=== Text length stats (characters) ===")
    print(df.groupby("label")["length"].describe().round(1).to_string())

    fig, ax = plt.subplots(figsize=(7, 4))
    for label, grp in df.groupby("label"):
        grp["length"].plot.hist(bins=50, alpha=0.6, label=label, ax=ax)
    ax.set_xlabel("Message length (chars)")
    ax.set_title("Text length distribution by class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(EDA_DIR / "eda_text_length.png", dpi=120)
    plt.close(fig)
    print("Saved: data/eda_text_length.png")


def top_words(df: pd.DataFrame, n: int = 10) -> None:
    import re
    from collections import Counter

    stopwords = {
        "i",
        "a",
        "the",
        "to",
        "and",
        "is",
        "in",
        "it",
        "of",
        "you",
        "my",
        "me",
        "we",
        "he",
        "she",
        "are",
        "be",
        "was",
        "for",
        "on",
        "that",
        "this",
        "with",
        "do",
        "have",
        "your",
        "but",
        "not",
        "at",
        "or",
        "from",
        "an",
        "as",
        "so",
        "if",
        "up",
        "its",
        "no",
        "by",
        "had",
        "has",
        "all",
        "any",
        "there",
    }

    print(f"\n=== Top {n} words per class (stopwords removed) ===")
    for label, grp in df.groupby("label"):
        words = re.findall(r"[a-z]+", grp["text"].str.lower().str.cat(sep=" "))
        words = [w for w in words if w not in stopwords]
        top = Counter(words).most_common(n)
        print(f"\n  {label.upper()}:")
        for word, count in top:
            print(f"    {word:20s} {count}")


def main() -> None:
    if not RAW_CSV.exists():
        print(f"Missing {RAW_CSV} — run 01_download_data.py first.")
        return

    df = load_data()
    print(f"Loaded {len(df)} rows from {RAW_CSV}")

    class_distribution(df)
    text_length_stats(df)
    top_words(df)
    print("\nEDA complete.")


if __name__ == "__main__":
    main()

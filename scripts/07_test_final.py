"""
Phase 7 — One-shot final evaluation on the held-out test split.

IMPORTANT: Run this ONLY once, after all hyperparameter choices are locked.
Touching the test set multiple times invalidates the reported numbers.

Prints final accuracy, F1, precision, recall and saves:
  data/test_confusion_matrix.png
  data/test_report.txt
"""

from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
from datasets import load_from_disk
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

PROCESSED_DIR = Path("data/processed")
# Use frozen model if it exists, otherwise fall back to full fine-tuned best
FROZEN_DIR = Path("models/frozen")
BEST_DIR = Path("models/best")
LABEL_NAMES = ["ham", "spam"]
MLFLOW_EXPERIMENT = "sms-spam-distilbert"


def model_dir() -> Path:
    if FROZEN_DIR.exists():
        print(f"Using frozen model: {FROZEN_DIR}")
        return FROZEN_DIR
    print(f"Using best model: {BEST_DIR}")
    return BEST_DIR


def plot_cm(y_true, y_pred, out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Oranges",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Test Split (final)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    mdir = model_dir()
    if not mdir.exists():
        print("No trained model found. Run 04_train.py (and 06_freeze_tune.py) first.")
        return
    if not PROCESSED_DIR.exists():
        print("Processed dataset missing. Run 03_tokenise.py first.")
        return

    print("Loading test split …")
    ds = load_from_disk(str(PROCESSED_DIR))
    ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    model = DistilBertForSequenceClassification.from_pretrained(str(mdir))
    tokenizer = DistilBertTokenizerFast.from_pretrained(str(mdir))

    eval_args = TrainingArguments(
        output_dir="models/test_tmp",
        per_device_eval_batch_size=64,
        report_to="none",
    )
    trainer = Trainer(model=model, args=eval_args, tokenizer=tokenizer)

    predictions = trainer.predict(ds["test"])
    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_true = predictions.label_ids

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="binary")
    prec = precision_score(y_true, y_pred, average="binary")
    rec = recall_score(y_true, y_pred, average="binary")
    report = classification_report(y_true, y_pred, target_names=LABEL_NAMES)

    print("\n========== FINAL TEST RESULTS ==========")
    print(report)
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 (spam) : {f1:.4f}")
    print("=========================================\n")

    report_path = Path("data/test_report.txt")
    report_path.write_text(
        f"Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1: {f1:.4f}\n\n{report}"
    )
    print(f"Saved: {report_path}")

    plot_cm(y_true, y_pred, Path("data/test_confusion_matrix.png"))

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="final-test"):
        mlflow.log_metrics(
            {"test_accuracy": acc, "test_f1": f1, "test_precision": prec, "test_recall": rec}
        )
        mlflow.log_artifact(str(report_path))
        mlflow.log_artifact("data/test_confusion_matrix.png")


if __name__ == "__main__":
    main()

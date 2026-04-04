"""
Phase 5 — Evaluate the best checkpoint on the validation split.

Prints:
  • Accuracy, precision, recall, F1
  • Full classification report
  • Confusion matrix (saved to data/confusion_matrix.png)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datasets import load_from_disk
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

PROCESSED_DIR = Path("data/processed")
BEST_MODEL_DIR = Path("models/best")
LABEL_NAMES = ["ham", "spam"]


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="binary"),
    }


def plot_confusion_matrix(y_true, y_pred) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Validation Split")
    fig.tight_layout()
    fig.savefig("data/confusion_matrix.png", dpi=120)
    plt.close(fig)
    print("Saved: data/confusion_matrix.png")


def main() -> None:
    for required in [PROCESSED_DIR, BEST_MODEL_DIR]:
        if not required.exists():
            print(f"Missing {required} — run prior scripts first.")
            return

    print("Loading dataset and model …")
    ds = load_from_disk(str(PROCESSED_DIR))
    ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    model = DistilBertForSequenceClassification.from_pretrained(str(BEST_MODEL_DIR))
    tokenizer = DistilBertTokenizerFast.from_pretrained(str(BEST_MODEL_DIR))

    # Use Trainer in eval-only mode
    eval_args = TrainingArguments(
        output_dir="models/eval_tmp",
        per_device_eval_batch_size=64,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=eval_args,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("\n--- Validation split ---")
    predictions = trainer.predict(ds["val"])
    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_true = predictions.label_ids

    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES))
    plot_confusion_matrix(y_true, y_pred)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="binary")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 (spam): {f1:.4f}")


if __name__ == "__main__":
    main()

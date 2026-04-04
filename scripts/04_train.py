"""
Phase 4 — Fine-tune DistilBERT for sequence classification.

Strategy:
  • All layers unfrozen for the first run (full fine-tune)
  • 3 epochs, batch size 32, AdamW lr=2e-5
  • Best checkpoint saved to models/best/
  • Every run is tracked with MLflow

Run:
  python scripts/04_train.py
  python scripts/04_train.py --epochs 5 --lr 3e-5
"""

import argparse
from pathlib import Path

import mlflow
import numpy as np
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")
CHECKPOINT = "distilbert-base-uncased"
MLFLOW_EXPERIMENT = "sms-spam-distilbert"


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="binary"),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--weight_decay", type=float, default=0.01)
    return p.parse_args()


def main():
    args = parse_args()

    if not PROCESSED_DIR.exists():
        print(f"Missing {PROCESSED_DIR} — run 03_tokenise.py first.")
        return

    print("Loading tokenised dataset …")
    ds = load_from_disk(str(PROCESSED_DIR))
    ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    print(f"Loading model: {CHECKPOINT} …")
    model = DistilBertForSequenceClassification.from_pretrained(
        CHECKPOINT, num_labels=2
    )
    tokenizer = DistilBertTokenizerFast.from_pretrained(CHECKPOINT)

    output_dir = MODEL_DIR / "checkpoints"
    best_dir = MODEL_DIR / "best"
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=str(MODEL_DIR / "logs"),
        logging_steps=50,
        report_to="none",       # MLflow handled manually below
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # MLflow tracking
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="full-fine-tune"):
        mlflow.log_params(
            {
                "model": CHECKPOINT,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "warmup_steps": args.warmup_steps,
                "weight_decay": args.weight_decay,
                "frozen_layers": "none",
            }
        )

        print(f"\nTraining for {args.epochs} epoch(s) …")
        train_result = trainer.train()

        # Log training metrics
        mlflow.log_metrics(
            {
                "train_loss": train_result.training_loss,
                "train_runtime_s": train_result.metrics["train_runtime"],
            }
        )

        # Evaluate on validation set
        val_metrics = trainer.evaluate(ds["val"])
        mlflow.log_metrics(
            {
                "val_accuracy": val_metrics["eval_accuracy"],
                "val_f1": val_metrics["eval_f1"],
                "val_loss": val_metrics["eval_loss"],
            }
        )

        print(f"\nValidation → accuracy: {val_metrics['eval_accuracy']:.4f}  f1: {val_metrics['eval_f1']:.4f}")

        # Save best model
        trainer.save_model(str(best_dir))
        tokenizer.save_pretrained(str(best_dir))
        mlflow.log_artifact(str(best_dir))
        print(f"Best model saved → {best_dir}")


if __name__ == "__main__":
    main()

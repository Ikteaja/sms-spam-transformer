"""
Phase 6 — Freeze lower transformer layers, re-train only the classifier head
           (and optionally the top-N transformer layers).

Why: After full fine-tuning we have a good representation. Freezing layers:
  • Speeds up future experiments by 10–20x
  • Only 3.5 % of parameters updated → much less risk of overfitting
  • Good baseline for ablation studies

Frozen layers: all distilbert.transformer.layer[0..4] (5 of 6)
Trainable   : distilbert.transformer.layer[5] + pre_classifier + classifier

Output: models/frozen/
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
BEST_MODEL_DIR = Path("models/best")
FROZEN_DIR = Path("models/frozen")
MLFLOW_EXPERIMENT = "sms-spam-distilbert"


def freeze_layers(model, num_frozen: int = 5) -> int:
    """Freeze the first `num_frozen` transformer blocks (0-indexed)."""
    frozen_params = 0
    for i in range(num_frozen):
        layer = model.distilbert.transformer.layer[i]
        for param in layer.parameters():
            param.requires_grad = False
            frozen_params += param.numel()

    # Also freeze embeddings
    for param in model.distilbert.embeddings.parameters():
        param.requires_grad = False
        frozen_params += param.numel()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Frozen  : {frozen_params:,} params")
    print(f"  Trainable: {trainable:,} / {total:,}  ({100*trainable/total:.1f} %)")
    return num_frozen


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
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument(
        "--num_frozen", type=int, default=5, help="Number of transformer layers to freeze (0-5)"
    )
    return p.parse_args()


def main():
    args = parse_args()

    for required in [PROCESSED_DIR, BEST_MODEL_DIR]:
        if not required.exists():
            print(f"Missing {required} — run prior scripts first.")
            return

    print("Loading dataset and model …")
    ds = load_from_disk(str(PROCESSED_DIR))
    ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    model = DistilBertForSequenceClassification.from_pretrained(str(BEST_MODEL_DIR))
    tokenizer = DistilBertTokenizerFast.from_pretrained(str(BEST_MODEL_DIR))

    print(f"\nFreezing {args.num_frozen} transformer layer(s) …")
    freeze_layers(model, num_frozen=args.num_frozen)

    output_dir = Path("models/frozen_checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",  # renamed from evaluation_strategy in transformers 4.46+
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
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

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"frozen-{args.num_frozen}-layers"):
        mlflow.log_params(
            {
                "model": str(BEST_MODEL_DIR),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "frozen_layers": args.num_frozen,
            }
        )

        print(f"\nTraining for {args.epochs} epoch(s) with frozen layers …")
        train_result = trainer.train()

        val_metrics = trainer.evaluate(ds["val"])
        mlflow.log_metrics(
            {
                "train_loss": train_result.training_loss,
                "val_accuracy": val_metrics["eval_accuracy"],
                "val_f1": val_metrics["eval_f1"],
                "val_loss": val_metrics["eval_loss"],
            }
        )

        print(
            f"\nValidation → accuracy: {val_metrics['eval_accuracy']:.4f}  f1: {val_metrics['eval_f1']:.4f}"
        )

        FROZEN_DIR.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(FROZEN_DIR))
        tokenizer.save_pretrained(str(FROZEN_DIR))
        mlflow.log_artifact(str(FROZEN_DIR))
        print(f"Frozen model saved → {FROZEN_DIR}")


if __name__ == "__main__":
    main()

"""
MLflow helper — compare all runs in the sms-spam-distilbert experiment.

Usage:
  python scripts/mlflow_compare.py
  python scripts/mlflow_compare.py --metric val_f1
  python scripts/mlflow_compare.py --top 5
"""

import argparse

import mlflow
import pandas as pd

EXPERIMENT_NAME = "sms-spam-distilbert"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--metric", default="val_f1", help="Metric to sort by")
    p.add_argument("--top", type=int, default=10, help="Number of runs to show")
    return p.parse_args()


def main():
    args = parse_args()
    client = mlflow.MlflowClient()

    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        print(f"Experiment '{EXPERIMENT_NAME}' not found. Run training scripts first.")
        return

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=[f"metrics.{args.metric} DESC"],
        max_results=args.top,
    )

    if not runs:
        print("No runs found.")
        return

    rows = []
    for run in runs:
        rows.append(
            {
                "run_id": run.info.run_id[:8],
                "run_name": run.info.run_name,
                "status": run.info.status,
                **{k: round(v, 4) for k, v in run.data.metrics.items()},
                **{k: v for k, v in run.data.params.items()},
            }
        )

    df = pd.DataFrame(rows)
    # Sort by requested metric if present
    if args.metric in df.columns:
        df = df.sort_values(args.metric, ascending=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(f"\n=== Top {len(df)} runs sorted by {args.metric} ===\n")
    print(df.to_string(index=False))
    print(f"\nMLflow UI: run `mlflow ui` then open http://localhost:5000")


if __name__ == "__main__":
    main()

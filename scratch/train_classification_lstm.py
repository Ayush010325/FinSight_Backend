"""
train_classification_lstm.py
==============================
End-to-end runner for Global Classification LSTM training.

Trains the model and reports:
    - Overall Accuracy, Precision, Recall, F1
    - Per-stock breakdown table
    - Comparison vs. Global Regression LSTM baseline (48.29% dir. accuracy)

Usage:
    python scratch/train_classification_lstm.py
"""

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_classification_lstm")

from database.database import SessionLocal
from services.global_classification_lstm_service import train_classification_model

BASELINE_DIR_ACCURACY = 0.4829  # Previous Global LSTM regression model


def main():
    logger.info("=" * 70)
    logger.info("Global Classification LSTM — Training Pipeline")
    logger.info("=" * 70)

    db = SessionLocal()
    try:
        results = train_classification_model(
            db=db,
            model_version="classification_v1",
            epochs=60,
            batch_size=32,
            lookback=10,
        )
    finally:
        db.close()

    if results.get("status") != "completed":
        logger.error("Training did not complete successfully.")
        return

    overall = results["overall_metrics"]
    per_stock = results["per_stock_metrics"]

    print("\n" + "=" * 70)
    print("GLOBAL CLASSIFICATION LSTM — RESULTS")
    print("=" * 70)

    print(f"\nOverall Metrics:")
    print(f"  Train Samples : {results['global_train_samples']}")
    print(f"  BCE Loss      : {results['loss']:.6f}")
    print(f"  Accuracy      : {overall['accuracy']:.4f}  ({overall['accuracy']:.2%})")
    print(f"  Precision     : {overall['precision']:.4f}")
    print(f"  Recall        : {overall['recall']:.4f}")
    print(f"  F1 Score      : {overall['f1']:.4f}")

    delta = overall["accuracy"] - BASELINE_DIR_ACCURACY
    arrow = "UP" if delta >= 0 else "DOWN"
    print(f"\n  vs. Regression Baseline ({BASELINE_DIR_ACCURACY:.2%}): [{arrow}] {abs(delta):.2%}")

    print(f"\nPer-Stock Breakdown:")
    print(f"{'Symbol':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 62)

    accs = []
    for sym, m in sorted(per_stock.items(), key=lambda x: x[1]["accuracy"], reverse=True):
        accs.append(m["accuracy"])
        print(
            f"{sym:<20} "
            f"{m['accuracy']:>10.4f} "
            f"{m['precision']:>10.4f} "
            f"{m['recall']:>10.4f} "
            f"{m['f1']:>10.4f}"
        )

    print("-" * 62)
    print(f"{'Average':<20} {sum(accs)/len(accs):>10.4f}")
    print("\nModel saved to: models/lstm_classification_global.pt")
    print("Scalers saved to: models/scaler_cls_*.pkl")
    print("=" * 70)


if __name__ == "__main__":
    main()

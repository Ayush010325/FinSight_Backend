import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.WARNING)  # Suppress noisy startup logs

from database.database import SessionLocal
from database.models_db import PredictionRecord, Stock, ModelRecord
from services import db_service


def main():
    db = SessionLocal()

    print()
    print("=" * 70)
    print("   FinSightAI — Prediction Analysis Report")
    print("=" * 70)

    print("\n[1] LATEST PREDICTIONS PER STOCK")
    print(f"{'Symbol':<20} {'Date':>12} {'Direction':>10} {'Probability':>12} {'Confidence':>12}  {'Model'}")
    print("-" * 85)

    all_up = 0
    all_down = 0
    all_confidences = []
    high_confidence = []
    model_versions_used = {}

    stocks = db.query(Stock).filter(Stock.is_active == True).order_by(Stock.symbol).all()

    for s in stocks:
        latest = db_service.get_latest_prediction(db, s.symbol, model_version_prefix="classification")
        if latest is None:
            print(f"{s.symbol:<20} {'No prediction yet':>40}")
            continue

        payload = latest.predictions or {}
        direction = payload.get("direction", "N/A")
        prob = payload.get("probability", 0.0)
        conf = payload.get("confidence", 0.0)
        date_str = str(latest.prediction_date)
        version_short = latest.model_version.replace("classification_v1_", "cls_") if latest.model_version else "N/A"

        print(
            f"{s.symbol:<20} {date_str:>12} {direction:>10} "
            f"{prob:>12.4f} {conf:>12.1%}  {version_short}"
        )

        if direction == "UP":
            all_up += 1
        elif direction == "DOWN":
            all_down += 1

        all_confidences.append(conf)
        if conf >= 0.3:
            high_confidence.append(s.symbol)

        mv = latest.model_version or "unknown"
        model_versions_used[mv] = model_versions_used.get(mv, 0) + 1

    print()
    print("[2] CONFIDENCE DISTRIBUTION")
    if all_confidences:
        print(f"  Min        : {min(all_confidences):.1%}")
        print(f"  Average    : {sum(all_confidences)/len(all_confidences):.1%}")
        print(f"  Max        : {max(all_confidences):.1%}")
        buckets = {
            "Very Low  (0-10%)": 0,
            "Low       (10-20%)": 0,
            "Medium    (20-40%)": 0,
            "High      (40-60%)": 0,
            "Very High (>60%)": 0,
        }
        for c in all_confidences:
            if c < 0.10:
                buckets["Very Low  (0-10%)"] += 1
            elif c < 0.20:
                buckets["Low       (10-20%)"] += 1
            elif c < 0.40:
                buckets["Medium    (20-40%)"] += 1
            elif c < 0.60:
                buckets["High      (40-60%)"] += 1
            else:
                buckets["Very High (>60%)"] += 1
        print()
        for label, count in buckets.items():
            bar = "#" * count
            print(f"  {label}: {bar} ({count})")
        print(f"\n  High Confidence stocks (>= 30%): {high_confidence if high_confidence else 'None'}")
    else:
        print("  No predictions found.")

    print()
    print("[3] UP vs DOWN DISTRIBUTION")
    total = all_up + all_down
    if total > 0:
        print(f"  UP   : {all_up:>3} stocks  ({all_up/total:.0%})")
        print(f"  DOWN : {all_down:>3} stocks  ({all_down/total:.0%})")
        print(f"  Total: {total:>3} stocks")
    else:
        print("  No predictions found.")

    print()
    print("[4] MODEL VERSION USAGE")
    if model_versions_used:
        for version, count in sorted(model_versions_used.items()):
            print(f"  {version}: {count} stocks")
    else:
        print("  No models used yet.")

    print()
    print("[5] HISTORICAL PREDICTION COUNT (all time)")
    print(f"{'Symbol':<20} {'Count':>8}  {'Latest Date':>12}")
    print("-" * 44)
    for s in stocks:
        history = db_service.get_prediction_history(db, s.symbol, limit=1000)
        if not history:
            print(f"{s.symbol:<20} {'0':>8}  {'—':>12}")
            continue
        latest_date = str(history[0].prediction_date)
        print(f"{s.symbol:<20} {len(history):>8}  {latest_date:>12}")

    total_preds = db.query(PredictionRecord).count()
    print()
    print(f"[6] TOTAL PREDICTIONS IN DATABASE: {total_preds}")

    # Model registry accuracy reference
    print()
    print("[7] MODEL REGISTRY ACCURACY REFERENCE (classification models)")
    print(f"{'Symbol':<20} {'Accuracy':>10}  {'Version'}")
    print("-" * 55)
    cls_models = db.query(ModelRecord).filter(
        ModelRecord.is_active == True,
        ModelRecord.model_version.like("classification%")
    ).order_by(ModelRecord.symbol).all()
    for m in cls_models:
        print(f"{m.symbol:<20} {m.accuracy:>10.1%}  {m.model_version}")

    db.close()
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()

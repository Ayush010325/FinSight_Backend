import argparse
import logging
import os
import sys
import time
from datetime import date, datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_predictions")

from database.database import SessionLocal
from services.prediction_service import predict_all_stocks


def parse_args():
    parser = argparse.ArgumentParser(description="FinSightAI Batch Prediction Runner")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Prediction date (YYYY-MM-DD). Defaults to today.",
    )
    return parser.parse_args()


def print_banner():
    print()
    print("=" * 70)
    print("   FinSightAI - Batch Prediction Runner")
    print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


def print_summary(output: dict):
    results = output.get("results", [])
    success = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") == "failed"]

    print()
    print("=" * 70)
    print("PREDICTION RESULTS")
    print("=" * 70)
    print(f"  Prediction Date : {output.get('prediction_date', 'N/A')}")
    print(f"  Stocks Processed: {output.get('stocks_processed', 0)}")
    print(f"  Successful      : {output.get('success_count', 0)}")
    print(f"  Failed          : {output.get('failure_count', 0)}")
    print(f"  Execution Time  : {output.get('execution_time_seconds', 0):.3f}s")
    print()

    if success:
        print(f"{'Symbol':<16} {'Signal':<12} {'Conviction':<11} {'Direction':>9} {'Probability':>12} {'Confidence':>11}  {'Model Version'}")
        print("-" * 88)
        for r in sorted(success, key=lambda x: x["symbol"]):
            conf_pct = f"{r['confidence']:.1%}"
            signal = r.get("signal", "N/A")
            conviction = r.get("conviction", "N/A")
            print(
                f"{r['symbol']:<16} "
                f"{signal:<12} "
                f"{conviction:<11} "
                f"{r['direction']:>9} "
                f"{r['probability']:>12.4f} "
                f"{conf_pct:>11}  "
                f"{r.get('model_version', 'N/A')}"
            )
        print("-" * 88)

    if failed:
        print()
        print(f"FAILED PREDICTIONS ({len(failed)}):")
        for r in failed:
            print(f"  {r['symbol']:<20} Error: {r.get('error', 'Unknown')}")

    if success:
        strong_buys = sum(1 for r in success if r.get("signal") == "STRONG_BUY")
        buys = sum(1 for r in success if r.get("signal") == "BUY")
        holds = sum(1 for r in success if r.get("signal") == "HOLD")
        sells = sum(1 for r in success if r.get("signal") == "SELL")
        strong_sells = sum(1 for r in success if r.get("signal") == "STRONG_SELL")
        high_conv = [r for r in success if r.get("conviction") == "HIGH"]

        print()
        print("CONVICTION & ACTIONABLE SIGNALS:")
        print(f"  STRONG BUY   : {strong_buys} stocks")
        print(f"  BUY          : {buys} stocks")
        print(f"  HOLD/NEUTRAL : {holds} stocks  (Noise filtered)")
        print(f"  SELL         : {sells} stocks")
        print(f"  STRONG SELL  : {strong_sells} stocks")
        print(f"  High-Conviction Trades : {len(high_conv)}/{len(success)} stocks")

    print()
    print("All predictions saved to predictions table.")
    print("=" * 88)
    print()


def main():
    args = parse_args()

    if args.date:
        try:
            prediction_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            logger.error(f"Invalid date format: '{args.date}'. Use YYYY-MM-DD.")
            sys.exit(1)
    else:
        prediction_date = date.today()

    print_banner()
    logger.info(f"Running batch predictions for date={prediction_date}")

    db = SessionLocal()
    try:
        output = predict_all_stocks(db=db, prediction_date=prediction_date)
    finally:
        db.close()

    print_summary(output)

    if output.get("failure_count", 0) > 0:
        if output.get("success_count", 0) == 0:
            logger.error("All predictions failed.")
            sys.exit(1)
        else:
            logger.warning(f"{output['failure_count']} prediction(s) failed.")
            sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()

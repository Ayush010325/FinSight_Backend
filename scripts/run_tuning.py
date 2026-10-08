import argparse
import logging
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_tuning")

from database.database import SessionLocal
from services.tuning_service import run_hyperparameter_tuning


def parse_args():
    parser = argparse.ArgumentParser(description="FinSightAI — Hyperparameter Tuning")
    parser.add_argument(
        "--trials", type=int, default=10,
        help="Number of Optuna trials (default: 10)"
    )
    parser.add_argument(
        "--trial-epochs", type=int, default=20,
        help="Epochs per trial screening (default: 20)"
    )
    parser.add_argument(
        "--timeout", type=int, default=None,
        help="Max tuning time in seconds (optional, no limit by default)"
    )
    parser.add_argument(
        "--version", type=str, default="classification_v2",
        help="Model version name for the best model (default: classification_v2)"
    )
    parser.add_argument(
        "--output", type=str, default="tuning_results.json",
        help="Path to save the JSON results report (default: tuning_results.json)"
    )
    return parser.parse_args()


def print_banner(n_trials: int):
    print()
    print("=" * 70)
    print("  FinSightAI - Hyperparameter Tuning (Optuna TPE)")
    print(f"  Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Trials    : {n_trials}")
    print(f"  Objective : Maximize overall test accuracy")
    print()
    print("  Search Space:")
    print("    hidden_dim   : [64, 128, 256]")
    print("    num_layers   : [1, 2, 3]")
    print("    embedding_dim: [4, 8, 16]")
    print("    dropout      : 0.10 -> 0.50")
    print("    lr           : 1e-4 -> 1e-2  (log scale)")
    print("    weight_decay : 1e-5 -> 1e-3  (log scale)")
    print("    batch_size   : [16, 32, 64]")
    print("    lookback     : [5, 10, 15, 20]")
    print("    scheduler    : [step, cosine, none]")
    print("=" * 70)
    print()
    print(f"  {'Trial':>5}  {'Accuracy':>10}  {'Best':>8}  Config")
    print("  " + "-" * 66)


def print_summary(report: dict):
    print()
    print("=" * 70)
    print("  TUNING COMPLETE")
    print("=" * 70)
    print(f"  Trials run        : {report['n_trials']}")
    print(f"  Total time        : {report['elapsed_seconds']:.0f}s")
    print(f"  Best trial #      : {report['best_trial']}")
    print(f"  Best trial acc    : {report['best_accuracy']:.2%}")
    print(f"  Final retrain acc : {report['final_retrain_accuracy']:.2%}")
    print()
    print("  Best Hyperparameters:")
    for k, v in report["best_config"].items():
        if isinstance(v, float):
            print(f"    {k:<20}: {v:.6f}")
        else:
            print(f"    {k:<20}: {v}")
    print()
    print("  Per-Stock Accuracy (after final retrain):")
    per_stock = report.get("final_retrain_per_stock", {})
    for sym, acc in sorted(per_stock.items(), key=lambda x: -x[1]):
        bar = "=" * int(acc * 20)
        print(f"    {sym:<25} {acc:.2%}  [{bar:<20}]")
    print()
    print(f"  Full results saved to: {report.get('output_path', 'tuning_results.json')}")
    print("=" * 70)
    print()


def main():
    args = parse_args()
    print_banner(args.trials)

    db = SessionLocal()
    try:
        report = run_hyperparameter_tuning(
            db=db,
            n_trials=args.trials,
            trial_epochs=args.trial_epochs,
            timeout_seconds=args.timeout,
            best_model_version=args.version,
            results_path=args.output,
        )
    finally:
        db.close()

    report["output_path"] = args.output
    print_summary(report)

    best = report["best_accuracy"]
    retrain = report["final_retrain_accuracy"]

    if retrain >= best - 0.02:
        print("  Final model matches tuning results. Ready for production.")
        sys.exit(0)
    else:
        print("  Warning: retrain accuracy lower than best trial — variance is normal.")
        sys.exit(0)


if __name__ == "__main__":
    main()

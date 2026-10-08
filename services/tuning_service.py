import json
import logging
import os
import pickle
from datetime import datetime
from typing import Any, Dict, Optional

import torch
import optuna
from sqlalchemy.orm import Session

from services.global_classification_lstm_service import (
    DEFAULT_CONFIG,
    train_with_config,
    MinMaxScalerCustom,
)
from services import db_service

logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _objective(trial: optuna.Trial, db: Session, trial_epochs: int = 20) -> float:
    config = {
        "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "embedding_dim": trial.suggest_categorical("embedding_dim", [4, 8, 16]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5, step=0.05),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "lookback": trial.suggest_categorical("lookback", [5, 10, 15, 20]),
        "scheduler": trial.suggest_categorical("scheduler", ["step", "cosine", "none"]),
        "epochs": trial_epochs,
    }

    try:
        result = train_with_config(db=db, config=config, save=False)
        acc = result["overall_metrics"]["accuracy"]
        return acc
    except Exception as e:
        logger.warning(f"Trial {trial.number} failed: {e}")
        raise optuna.exceptions.TrialPruned()


def run_hyperparameter_tuning(
    db: Session,
    n_trials: int = 10,
    trial_epochs: int = 20,
    timeout_seconds: Optional[int] = None,
    best_model_version: str = "classification_v2",
    results_path: str = "tuning_results.json",
) -> Dict[str, Any]:
    logger.info(f"Starting Optuna HPO study | n_trials={n_trials} | trial_epochs={trial_epochs}")
    started_at = datetime.utcnow()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name="lstm_classification_hpo",
    )

    trial_log = []

    def _trial_callback(study: optuna.Study, trial: optuna.Trial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            entry = {
                "trial": trial.number + 1,
                "accuracy": round(trial.value, 4),
                "params": trial.params,
                "best_so_far": round(study.best_value, 4),
            }
            trial_log.append(entry)
            print(
                f"  Trial {trial.number + 1:>3}/{n_trials} | "
                f"acc={trial.value:.4f} | "
                f"best={study.best_value:.4f} | "
                f"hidden={trial.params.get('hidden_dim')} "
                f"layers={trial.params.get('num_layers')} "
                f"lr={trial.params.get('lr'):.5f} "
                f"lookback={trial.params.get('lookback')}",
                flush=True,
            )

    study.optimize(
        lambda trial: _objective(trial, db, trial_epochs=trial_epochs),
        n_trials=n_trials,
        timeout=timeout_seconds,
        callbacks=[_trial_callback],
        show_progress_bar=False,
        gc_after_trial=True,
    )

    best_params = study.best_params
    best_accuracy = study.best_value

    print(f"\n  Best accuracy: {best_accuracy:.2%}", flush=True)
    print(f"  Best config: {best_params}", flush=True)
    print(f"\n  Retraining with best config (60 epochs) and saving as '{best_model_version}'...", flush=True)

    best_config = {**DEFAULT_CONFIG, **best_params, "epochs": 60}
    final_result = train_with_config(
        db=db,
        config=best_config,
        model_version=best_model_version,
        save=True,
    )

    elapsed = (datetime.utcnow() - started_at).total_seconds()

    all_trials = [
        {
            "trial": t.number + 1,
            "accuracy": round(t.value, 4) if t.value is not None else None,
            "params": t.params,
            "state": str(t.state),
        }
        for t in study.trials
    ]

    report = {
        "study_name": study.study_name,
        "n_trials": n_trials,
        "elapsed_seconds": round(elapsed, 1),
        "best_trial": study.best_trial.number + 1,
        "best_accuracy": round(best_accuracy, 4),
        "best_config": best_params,
        "final_retrain_accuracy": round(
            final_result["overall_metrics"]["accuracy"], 4
        ),
        "final_retrain_per_stock": {
            sym: round(m["accuracy"], 4)
            for sym, m in final_result["per_stock_metrics"].items()
        },
        "all_trials": all_trials,
    }

    with open(results_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(
        f"HPO complete | best_accuracy={best_accuracy:.2%} | "
        f"retrain_accuracy={final_result['overall_metrics']['accuracy']:.2%} | "
        f"elapsed={elapsed:.0f}s"
    )

    return report

import os
import pickle
import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sqlalchemy.orm import Session

import sys
import types

from config.stocks import NIFTY_STOCKS
from database.models_db import ModelRecord, Stock, TrainingDataset, PredictionRecord
from services import db_service
from services.global_classification_lstm_service import (
    GlobalClassificationLSTM,
    CLASSIFICATION_FEATURES,
    BASE_RECORD_FEATURES,
    MinMaxScalerCustom,
)

if "services.lstm_training_service" not in sys.modules:
    _compat_mod = types.ModuleType("services.lstm_training_service")
    _compat_mod.MinMaxScalerCustom = MinMaxScalerCustom
    sys.modules["services.lstm_training_service"] = _compat_mod

logger = logging.getLogger(__name__)

LOOKBACK = 10
NUM_STOCKS = 10
EMBEDDING_DIM = 8
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.3
NUM_FEATURES = len(CLASSIFICATION_FEATURES)


class PredictionError(Exception):
    pass


def load_active_model(db: Session, symbol: str) -> ModelRecord:
    record = db.query(ModelRecord).filter(
        ModelRecord.symbol == symbol,
        ModelRecord.is_active == True,
        ModelRecord.model_version.like("classification%"),
    ).order_by(ModelRecord.accuracy.desc()).first()

    if record is None:
        record = db.query(ModelRecord).filter(
            ModelRecord.symbol == symbol,
            ModelRecord.is_active == True,
        ).order_by(ModelRecord.created_at.desc()).first()

    if record is None:
        raise PredictionError(
            f"No active model found for {symbol} in model_registry. "
            "Please train a model first."
        )

    if not record.model_path or not os.path.exists(record.model_path):
        raise PredictionError(
            f"Model file not found at '{record.model_path}' for {symbol}. "
            "Ensure models/ directory is intact."
        )

    logger.debug(f"Loaded model record for {symbol}: version={record.model_version}")
    return record


def load_scaler(scaler_path: str, symbol: str):
    if not scaler_path or not os.path.exists(scaler_path):
        raise PredictionError(
            f"Scaler file not found at '{scaler_path}' for {symbol}. "
            "Ensure models/scaler_cls_*.pkl files are present."
        )

    try:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    except Exception as e:
        raise PredictionError(f"Failed to load scaler from '{scaler_path}': {e}")

    return scaler


def load_global_model(model_path: str) -> GlobalClassificationLSTM:
    try:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    except Exception as e:
        raise PredictionError(f"Failed to load model weights from '{model_path}': {e}")

    # Determine input dim and hidden dim from state_dict if present
    in_dim = NUM_FEATURES
    h_dim = HIDDEN_DIM
    n_layers = NUM_LAYERS
    use_attn = True

    if "lstm.weight_ih_l0" in state_dict:
        in_dim = state_dict["lstm.weight_ih_l0"].shape[1]
        h_dim = state_dict["lstm.weight_ih_l0"].shape[0] // 4
    if "lstm.weight_ih_l1" in state_dict:
        n_layers = 2
    elif "lstm.weight_ih_l2" in state_dict:
        n_layers = 3
    else:
        n_layers = 1

    if "attention.projection.0.weight" not in state_dict:
        use_attn = False

    model = GlobalClassificationLSTM(
        input_dim=in_dim,
        hidden_dim=h_dim,
        num_layers=n_layers,
        num_stocks=NUM_STOCKS,
        embedding_dim=EMBEDDING_DIM,
        dropout=DROPOUT,
        use_attention=use_attn,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def get_stock_index(db: Session, symbol: str) -> int:
    stocks = db.query(Stock).filter(Stock.is_active == True).order_by(Stock.id.asc()).all()
    for idx, s in enumerate(stocks):
        if s.symbol == symbol:
            return idx
    raise PredictionError(
        f"Symbol '{symbol}' not found among active stocks. "
        f"Active stocks: {[s.symbol for s in stocks]}"
    )


def get_latest_feature_sequence(
    db: Session,
    symbol: str,
    lookback: int = LOOKBACK,
) -> pd.DataFrame:
    stock = db.query(Stock).filter(Stock.symbol == symbol).first()
    if stock is None:
        raise PredictionError(f"Stock '{symbol}' not registered in the stocks table.")

    records = (
        db.query(TrainingDataset)
        .filter(TrainingDataset.stock_id == stock.id)
        .order_by(TrainingDataset.date.desc())
        .limit(lookback)
        .all()
    )

    if len(records) < lookback:
        raise PredictionError(
            f"Insufficient data for {symbol}: need {lookback} rows, "
            f"found {len(records)} in training_dataset."
        )

    records = list(reversed(records))

    rows = []
    for r in records:
        row = {"date": r.date}
        for col in BASE_RECORD_FEATURES:
            row[col] = getattr(r, col, None)
        rows.append(row)

    df = pd.DataFrame(rows)

    for col in BASE_RECORD_FEATURES:
        if col in df.columns and df[col].isna().any():
            median = df[col].median()
            df[col] = df[col].fillna(median if not pd.isna(median) else 0.0)

    bb_width = (df["bb_upper"] - df["bb_lower"]).replace(0, 1.0)
    df["bb_pct_b"] = (df["bb_middle"] - df["bb_lower"]) / bb_width

    ema50_safe = df["ema50"].replace(0, 1.0)
    df["ema_trend_20_50"] = (df["ema20"] - df["ema50"]) / ema50_safe

    ema100_safe = df["ema100"].replace(0, 1.0)
    df["ema_trend_50_100"] = (df["ema50"] - df["ema100"]) / ema100_safe

    for col in CLASSIFICATION_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
        if df[col].isna().any():
            median = df[col].median()
            df[col] = df[col].fillna(median if not pd.isna(median) else 0.0)

    return df


def prepare_prediction_input(
    df: pd.DataFrame,
    scaler,
    stock_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    feature_values = df[CLASSIFICATION_FEATURES].values
    scaled = scaler.transform(feature_values)

    x_tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)
    idx_tensor = torch.tensor([stock_idx], dtype=torch.long)
    return x_tensor, idx_tensor


def predict_direction(
    model: GlobalClassificationLSTM,
    x_tensor: torch.Tensor,
    idx_tensor: torch.Tensor,
    high_threshold: float = 0.60,
    moderate_threshold: float = 0.54,
) -> Dict[str, Any]:
    with torch.no_grad():
        logit = model(x_tensor, idx_tensor)

    probability = float(logit.cpu().numpy().flatten()[0])
    raw_confidence = abs(probability - 0.5) * 2.0

    if probability >= high_threshold:
        signal = "STRONG_BUY"
        conviction = "HIGH"
        direction = "UP"
    elif probability >= moderate_threshold:
        signal = "BUY"
        conviction = "MODERATE"
        direction = "UP"
    elif probability <= (1.0 - high_threshold):
        signal = "STRONG_SELL"
        conviction = "HIGH"
        direction = "DOWN"
    elif probability <= (1.0 - moderate_threshold):
        signal = "SELL"
        conviction = "MODERATE"
        direction = "DOWN"
    else:
        signal = "HOLD"
        conviction = "LOW"
        direction = "NEUTRAL"

    return {
        "probability": round(probability, 6),
        "direction": direction,
        "signal": signal,
        "conviction": conviction,
        "confidence": round(raw_confidence, 6),
    }


def save_prediction(
    db: Session,
    symbol: str,
    model_version: str,
    prediction_date: date,
    probability: float,
    direction: str,
    confidence: float,
    signal: str = "HOLD",
    conviction: str = "LOW",
) -> PredictionRecord:
    payload = {
        "direction": direction,
        "signal": signal,
        "conviction": conviction,
        "probability": probability,
        "confidence": confidence,
        "predicted_at": datetime.utcnow().isoformat(),
    }

    existing = (
        db.query(PredictionRecord)
        .filter(
            PredictionRecord.symbol == symbol,
            PredictionRecord.prediction_date == prediction_date,
            PredictionRecord.model_version == model_version,
        )
        .first()
    )

    if existing:
        existing.predictions = payload
        db.commit()
        db.refresh(existing)
        logger.debug(f"Updated prediction for {symbol} on {prediction_date}")
        return existing

    record = PredictionRecord(
        symbol=symbol,
        model_version=model_version,
        prediction_date=prediction_date,
        predictions=payload,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    logger.debug(f"Inserted new prediction for {symbol} on {prediction_date}")
    return record


def predict_stock(
    db: Session,
    symbol: str,
    model: Optional[GlobalClassificationLSTM] = None,
    prediction_date: Optional[date] = None,
    lookback: int = LOOKBACK,
) -> Dict[str, Any]:
    if prediction_date is None:
        prediction_date = date.today()

    try:
        model_record = load_active_model(db, symbol)
        scaler = load_scaler(model_record.scaler_path, symbol)

        if model is None:
            model = load_global_model(model_record.model_path)

        df = get_latest_feature_sequence(db, symbol, lookback)
        stock_idx = get_stock_index(db, symbol)
        x_tensor, idx_tensor = prepare_prediction_input(df, scaler, stock_idx)
        result = predict_direction(model, x_tensor, idx_tensor)

        save_prediction(
            db=db,
            symbol=symbol,
            model_version=model_record.model_version,
            prediction_date=prediction_date,
            probability=result["probability"],
            direction=result["direction"],
            confidence=result["confidence"],
            signal=result["signal"],
            conviction=result["conviction"],
        )

        logger.info(
            f"[{symbol}] Prediction: {result['signal']} ({result['direction']}) "
            f"prob={result['probability']:.4f}, "
            f"conviction={result['conviction']}"
        )

        return {
            "symbol": symbol,
            "status": "success",
            "direction": result["direction"],
            "signal": result["signal"],
            "conviction": result["conviction"],
            "probability": result["probability"],
            "confidence": result["confidence"],
            "model_version": model_record.model_version,
            "prediction_date": prediction_date.isoformat(),
        }

    except PredictionError as e:
        logger.error(f"[{symbol}] Prediction failed: {e}")
        return {
            "symbol": symbol,
            "status": "failed",
            "error": str(e),
            "prediction_date": prediction_date.isoformat() if prediction_date else None,
        }
    except Exception as e:
        logger.error(f"[{symbol}] Unexpected error during prediction: {e}", exc_info=True)
        return {
            "symbol": symbol,
            "status": "failed",
            "error": f"Unexpected error: {e}",
            "prediction_date": prediction_date.isoformat() if prediction_date else None,
        }


def predict_all_stocks(
    db: Session,
    prediction_date: Optional[date] = None,
    lookback: int = LOOKBACK,
) -> Dict[str, Any]:
    if prediction_date is None:
        prediction_date = date.today()

    started_at = datetime.utcnow()
    logger.info(f"Starting batch prediction for all stocks. Date={prediction_date}")

    sample_record = db.query(ModelRecord).filter(
        ModelRecord.is_active == True,
        ModelRecord.model_version.like("classification%"),
    ).first()

    if sample_record is None:
        return {
            "status": "failed",
            "error": "No active classification model found in model_registry.",
            "results": [],
        }

    try:
        shared_model = load_global_model(sample_record.model_path)
        logger.info(f"Shared model loaded from {sample_record.model_path}")
    except PredictionError as e:
        return {
            "status": "failed",
            "error": str(e),
            "results": [],
        }

    active_stocks = db.query(Stock).filter(Stock.is_active == True).order_by(Stock.id.asc()).all()

    results = []
    success_count = 0
    failure_count = 0

    for stock in active_stocks:
        result = predict_stock(
            db=db,
            symbol=stock.symbol,
            model=shared_model,
            prediction_date=prediction_date,
            lookback=lookback,
        )
        results.append(result)
        if result["status"] == "success":
            success_count += 1
        else:
            failure_count += 1

    elapsed = (datetime.utcnow() - started_at).total_seconds()

    logger.info(
        f"Batch prediction complete. "
        f"Success={success_count}, Failures={failure_count}, "
        f"Time={elapsed:.2f}s"
    )

    return {
        "status": "completed",
        "prediction_date": prediction_date.isoformat(),
        "stocks_processed": len(active_stocks),
        "success_count": success_count,
        "failure_count": failure_count,
        "execution_time_seconds": round(elapsed, 3),
        "results": results,
    }

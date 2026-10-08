import os
import logging
import pickle
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sqlalchemy.orm import Session

from database.models_db import TrainingDataset, Stock
from services import db_service

logger = logging.getLogger(__name__)

BASE_RECORD_FEATURES = [
    "rsi14", "macd", "macd_signal", "macd_hist", "atr14",
    "daily_return", "rolling_volatility20", "relative_volume",
    "momentum5", "momentum10", "momentum20",
    "avg_positive", "avg_negative", "net_sentiment", "news_count",
    "sentiment_3d_avg", "sentiment_7d_avg", "sentiment_change", "news_surprise",
    "nifty_return", "nifty_rsi14", "nifty_ema20", "nifty_ema50", "nifty_volatility20",
    "market_regime", "news_present_flag",
    "ema20", "ema50", "ema100", "bb_upper", "bb_middle", "bb_lower",
]

CLASSIFICATION_FEATURES = [
    "rsi14", "macd", "macd_signal", "macd_hist", "atr14",
    "daily_return", "rolling_volatility20", "relative_volume",
    "momentum5", "momentum10", "momentum20",
    "avg_positive", "avg_negative", "net_sentiment", "news_count",
    "sentiment_3d_avg", "sentiment_7d_avg", "sentiment_change", "news_surprise",
    "nifty_return", "nifty_rsi14", "nifty_ema20", "nifty_ema50", "nifty_volatility20",
    "market_regime", "news_present_flag",
    "bb_pct_b", "ema_trend_20_50", "ema_trend_50_100",
]

TARGET_COL = "target_direction_t_plus_1"


class MinMaxScalerCustom:

    def __init__(self):
        self.min_ = None
        self.max_ = None
        self.range_ = None

    def fit(self, X: np.ndarray):
        self.min_ = np.nanmin(X, axis=0)
        self.max_ = np.nanmax(X, axis=0)
        self.range_ = self.max_ - self.min_
        self.range_[self.range_ == 0.0] = 1.0

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.min_) / self.range_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


def prepare_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    lookback: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    X_raw = df[feature_cols].values
    y_raw = df[target_col].values

    X_seq, y_seq = [], []

    for i in range(lookback, len(df)):
        X_seq.append(X_raw[i - lookback + 1 : i + 1])
        y_seq.append([y_raw[i]])

    return np.array(X_seq), np.array(y_seq)


class TemporalAttention(nn.Module):

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, max(16, hidden_dim // 2)),
            nn.Tanh(),
            nn.Linear(max(16, hidden_dim // 2), 1, bias=False),
        )

    def forward(self, lstm_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.projection(lstm_outputs)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(lstm_outputs * weights, dim=1)
        return context, weights.squeeze(-1)


class GlobalClassificationLSTM(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_stocks: int = 10,
        embedding_dim: int = 8,
        dropout: float = 0.3,
        use_attention: bool = True,
    ):
        super().__init__()
        self.use_attention = use_attention
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        if use_attention:
            self.attention = TemporalAttention(hidden_dim)
        self.embedding = nn.Embedding(num_embeddings=num_stocks, embedding_dim=embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim + embedding_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, stock_idx: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        if self.use_attention:
            rep, _ = self.attention(lstm_out)
        else:
            rep = lstm_out[:, -1, :]
        embedded = self.embedding(stock_idx)
        combined = torch.cat([rep, embedded], dim=1)
        out = self.dropout(combined)
        out = self.fc(out)
        return self.sigmoid(out)


class EarlyStopping:

    def __init__(self, patience: int = 7, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.best_state_dict = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_state_dict = {k: v.clone().detach() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def _load_stock_df(records: list) -> pd.DataFrame:
    rows = []
    for r in records:
        row = {"date": r.date, TARGET_COL: getattr(r, TARGET_COL, None)}
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


def _compute_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray) -> Dict[str, float]:
    y_pred = (y_pred_prob >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


DEFAULT_CONFIG = {
    "hidden_dim": 64,
    "num_layers": 2,
    "embedding_dim": 8,
    "dropout": 0.3,
    "lr": 0.0015,
    "weight_decay": 1.2e-5,
    "batch_size": 16,
    "lookback": 10,
    "epochs": 30,
    "scheduler": "cosine",
    "use_attention": True,
    "patience": 7,
}


def train_with_config(
    db: Session,
    config: Dict[str, Any],
    model_version: str = "classification_v1",
    save: bool = True,
) -> Dict[str, Any]:
    hidden_dim = config.get("hidden_dim", 64)
    num_layers = config.get("num_layers", 2)
    embedding_dim = config.get("embedding_dim", 8)
    dropout = config.get("dropout", 0.3)
    lr = config.get("lr", 0.0015)
    weight_decay = config.get("weight_decay", 1.2e-5)
    batch_size = config.get("batch_size", 16)
    lookback = config.get("lookback", 10)
    epochs = config.get("epochs", 30)
    scheduler_type = config.get("scheduler", "cosine")
    use_attention = config.get("use_attention", True)
    patience = config.get("patience", 7)

    started_at = datetime.utcnow()
    logger.info(
        f"Training | version={model_version} hidden={hidden_dim} "
        f"layers={num_layers} lr={lr:.5f} lookback={lookback} epochs={epochs} attention={use_attention}"
    )

    stocks = db.query(Stock).filter(Stock.is_active == True).order_by(Stock.id.asc()).all()
    if not stocks:
        raise ValueError("No active stocks found.")

    stock_idx_map = {s.id: idx for idx, s in enumerate(stocks)}
    num_stocks = len(stocks)
    feature_cols = CLASSIFICATION_FEATURES

    train_x_list, train_y_list, train_idx_list = [], [], []
    stock_test_data: Dict[str, Tuple] = {}
    stock_meta: Dict[str, Dict] = {}

    for s in stocks:
        records = db.query(TrainingDataset).filter(
            TrainingDataset.stock_id == s.id
        ).order_by(TrainingDataset.date.asc()).all()

        if len(records) < lookback + 15:
            logger.warning(f"Skipping {s.symbol}: only {len(records)} rows.")
            continue

        valid_records = [r for r in records if getattr(r, TARGET_COL, None) is not None]
        if len(valid_records) < lookback + 15:
            logger.warning(f"Skipping {s.symbol}: insufficient labelled rows.")
            continue

        df = _load_stock_df(valid_records)

        latest_date = df["date"].max()
        if isinstance(latest_date, str):
            latest_date = datetime.strptime(latest_date, "%Y-%m-%d").date()

        split_idx = int(len(df) * 0.8)
        df_train = df.iloc[:split_idx].copy()
        df_test = df.iloc[split_idx:].copy()

        scaler = MinMaxScalerCustom()
        scaler.fit(df_train[feature_cols].values)
        df_train[feature_cols] = scaler.transform(df_train[feature_cols].values)
        df_test[feature_cols] = scaler.transform(df_test[feature_cols].values)

        X_tr, y_tr = prepare_sequences(df_train, feature_cols, TARGET_COL, lookback)
        X_te, y_te = prepare_sequences(df_test, feature_cols, TARGET_COL, lookback)

        if len(X_tr) == 0 or len(X_te) == 0:
            logger.warning(f"Sequence generation failed for {s.symbol}. Skipping.")
            continue

        s_idx = stock_idx_map[s.id]
        train_x_list.append(X_tr)
        train_y_list.append(y_tr)
        train_idx_list.append(np.full((len(X_tr),), s_idx, dtype=np.int64))

        stock_test_data[s.symbol] = (X_te, y_te, s_idx, scaler)
        stock_meta[s.symbol] = {"train_samples": len(X_tr), "trained_until": latest_date}

    if not train_x_list:
        raise ValueError("No stocks had sufficient sequences to train.")

    X_train = np.concatenate(train_x_list, axis=0)
    y_train = np.concatenate(train_y_list, axis=0)
    idx_train = np.concatenate(train_idx_list, axis=0)

    num_samples = len(X_train)
    val_split = int(num_samples * 0.85)

    np.random.seed(42)
    shuffled_indices = np.random.permutation(num_samples)
    tr_idx, val_idx = shuffled_indices[:val_split], shuffled_indices[val_split:]

    X_train_t = torch.tensor(X_train[tr_idx], dtype=torch.float32)
    y_train_t = torch.tensor(y_train[tr_idx].reshape(-1, 1), dtype=torch.float32)
    idx_train_t = torch.tensor(idx_train[tr_idx], dtype=torch.long)

    X_val_t = torch.tensor(X_train[val_idx], dtype=torch.float32)
    y_val_t = torch.tensor(y_train[val_idx].reshape(-1, 1), dtype=torch.float32)
    idx_val_t = torch.tensor(idx_train[val_idx], dtype=torch.long)

    os.makedirs("models", exist_ok=True)
    model_path = "models/lstm_classification_global.pt"

    if save:
        db_run = db_service.start_training_run(db, "GLOBAL_CLS", model_version, started_at)
        run_id = db_run.id
    else:
        run_id = None

    model = GlobalClassificationLSTM(
        input_dim=len(feature_cols),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_stocks=num_stocks,
        embedding_dim=embedding_dim,
        dropout=dropout,
        use_attention=use_attention,
    )

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    elif scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        scheduler = None

    early_stopping = EarlyStopping(patience=patience, min_delta=1e-4)
    final_loss = 0.0

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_train_t.size(0))
        epoch_loss = 0.0

        for i in range(0, X_train_t.size(0), batch_size):
            indices = permutation[i: i + batch_size]
            batch_x = X_train_t[indices]
            batch_y = y_train_t[indices]
            batch_idx = idx_train_t[indices]

            optimizer.zero_grad()
            outputs = model(batch_x, batch_idx)
            loss = criterion(outputs, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(batch_x)

        if scheduler is not None:
            scheduler.step()

        train_loss = epoch_loss / len(X_train_t)

        model.eval()
        with torch.no_grad():
            val_out = model(X_val_t, idx_val_t)
            val_loss = criterion(val_out, y_val_t).item()

        if early_stopping.step(val_loss, model):
            logger.info(f"Early stopping triggered at epoch {epoch + 1} | best_val_loss={early_stopping.best_loss:.5f}")
            break

        final_loss = train_loss
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            logger.info(f"Epoch [{epoch + 1}/{epochs}] Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

    if early_stopping.best_state_dict is not None:
        model.load_state_dict(early_stopping.best_state_dict)

    if save:
        torch.save(model.state_dict(), model_path)

    model.eval()
    per_stock_results: Dict[str, Any] = {}
    overall_y_pred: List[float] = []
    overall_y_true: List[float] = []

    for symbol, (X_te, y_te, s_idx, scaler) in stock_test_data.items():
        X_te_t = torch.tensor(X_te, dtype=torch.float32)
        idx_te_t = torch.full((len(X_te),), s_idx, dtype=torch.long)

        with torch.no_grad():
            preds_prob = model(X_te_t, idx_te_t).cpu().numpy().flatten()

        y_true_arr = y_te.flatten()
        metrics = _compute_metrics(y_true_arr, preds_prob)
        per_stock_results[symbol] = metrics
        overall_y_pred.extend(preds_prob.tolist())
        overall_y_true.extend(y_true_arr.tolist())

        if save:
            scaler_path = f"models/scaler_cls_{symbol.replace('.', '_')}.pkl"
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)

            meta = stock_meta[symbol]
            db_service.register_successful_training(
                db=db,
                symbol=symbol,
                model_version=f"{model_version}_{symbol.replace('.', '_')}",
                model_path=model_path,
                scaler_path=scaler_path,
                accuracy=metrics["accuracy"],
                trained_until=meta["trained_until"],
                run_id=None,
                epochs=epochs,
                train_samples=meta["train_samples"],
                loss=final_loss,
                started_at=started_at,
                completed_at=datetime.utcnow(),
            )

    overall_metrics = _compute_metrics(np.array(overall_y_true), np.array(overall_y_pred))

    if save and run_id is not None:
        db_service.complete_training_run(
            db=db,
            run_id=run_id,
            epochs=epochs,
            train_samples=num_samples,
            validation_accuracy=overall_metrics["accuracy"],
            loss=final_loss,
            completed_at=datetime.utcnow(),
        )

    logger.info(f"Done | Accuracy={overall_metrics['accuracy']:.2%} F1={overall_metrics['f1']:.4f}")

    return {
        "status": "completed",
        "global_train_samples": num_samples,
        "loss": final_loss,
        "overall_metrics": overall_metrics,
        "per_stock_metrics": per_stock_results,
        "model": model if not save else None,
        "stock_test_data": stock_test_data if not save else None,
    }


def train_classification_model(
    db: Session,
    model_version: str = "classification_v3",
    epochs: int = 30,
    batch_size: int = 16,
    lookback: int = 10,
    hidden_dim: int = 64,
    num_layers: int = 2,
    lr: float = 0.0015,
) -> Dict[str, Any]:
    config = {
        **DEFAULT_CONFIG,
        "epochs": epochs,
        "batch_size": batch_size,
        "lookback": lookback,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "lr": lr,
    }
    result = train_with_config(db=db, config=config, model_version=model_version, save=True)
    return {k: v for k, v in result.items() if k not in ("model", "stock_test_data")}

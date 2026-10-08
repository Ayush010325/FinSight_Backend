import logging
from datetime import date, datetime
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session

from database.models_db import FeatureStore, DailySentimentFeatures, TrainingDataset, NiftyFeatures

logger = logging.getLogger(__name__)


def load_feature_store(db: Session, stock_id: Optional[int] = None) -> pd.DataFrame:
    query = db.query(FeatureStore)
    if stock_id is not None:
        query = query.filter(FeatureStore.stock_id == stock_id)
    records = query.order_by(FeatureStore.date.asc()).all()

    if not records:
        logger.warning("No records found in feature_store")
        return pd.DataFrame()

    df = pd.DataFrame([{
        "stock_id": r.stock_id,
        "date": r.date,
        "ema20": r.ema20,
        "ema50": r.ema50,
        "ema100": r.ema100,
        "sma20": r.sma20,
        "sma50": r.sma50,
        "rsi14": r.rsi14,
        "macd": r.macd,
        "macd_signal": r.macd_signal,
        "macd_hist": r.macd_hist,
        "bb_upper": r.bb_upper,
        "bb_middle": r.bb_middle,
        "bb_lower": r.bb_lower,
        "atr14": r.atr14,
        "daily_return": r.daily_return,
        "log_return": r.log_return,
        "rolling_volatility20": r.rolling_volatility20,
        "relative_volume": r.relative_volume,
        "momentum5": r.momentum5,
        "momentum10": r.momentum10,
        "momentum20": r.momentum20,
        "target_close_t_plus_1": r.target_close_t_plus_1,
        "target_return_t_plus_1": r.target_return_t_plus_1
    } for r in records])
    return df


def load_sentiment_features(db: Session, stock_id: Optional[int] = None) -> pd.DataFrame:
    query = db.query(DailySentimentFeatures)
    if stock_id is not None:
        query = query.filter(DailySentimentFeatures.stock_id == stock_id)
    records = query.order_by(DailySentimentFeatures.date.asc()).all()

    if not records:
        logger.info("No records found in daily_sentiment_features")
        return pd.DataFrame()

    df = pd.DataFrame([{
        "stock_id": r.stock_id,
        "date": r.date,
        "avg_positive": r.avg_positive,
        "avg_negative": r.avg_negative,
        "avg_neutral": r.avg_neutral,
        "news_count": r.news_count,
        "net_sentiment": r.net_sentiment
    } for r in records])
    return df


def load_nifty_features(db: Session) -> pd.DataFrame:
    records = db.query(NiftyFeatures).order_by(NiftyFeatures.date.asc()).all()

    if not records:
        logger.warning("No records found in nifty_features — NIFTY context will be zeros.")
        return pd.DataFrame()

    df = pd.DataFrame([{
        "date": r.date,
        "nifty_return": r.nifty_return,
        "nifty_rsi14": r.nifty_rsi14,
        "nifty_ema20": r.nifty_ema20,
        "nifty_ema50": r.nifty_ema50,
        "nifty_volatility20": r.nifty_volatility20,
    } for r in records])
    return df


def _add_sentiment_momentum(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["stock_id", "date"]).copy()

    def _rolling_per_stock(group: pd.DataFrame) -> pd.DataFrame:
        g = group.sort_values("date").copy()
        g["sentiment_3d_avg"] = g["net_sentiment"].rolling(window=3, min_periods=1).mean()
        g["sentiment_7d_avg"] = g["net_sentiment"].rolling(window=7, min_periods=1).mean()
        g["sentiment_change"] = g["net_sentiment"] - g["net_sentiment"].shift(1)

        rolling_30 = g["news_count"].rolling(window=30, min_periods=1).mean()
        news_surprise = np.where(rolling_30 > 0, g["news_count"] / rolling_30, 1.0)
        g["news_surprise"] = np.clip(news_surprise, 0.0, 10.0)

        return g

    df = df.groupby("stock_id", group_keys=False).apply(_rolling_per_stock)
    df["sentiment_change"] = df["sentiment_change"].fillna(0.0)
    return df


def merge_features(
    df_features: pd.DataFrame,
    df_sentiment: pd.DataFrame,
    df_nifty: pd.DataFrame,
) -> pd.DataFrame:
    if df_features.empty:
        return pd.DataFrame()

    if df_sentiment.empty:
        df_merged = df_features.copy()
        df_merged["avg_positive"] = 0.0
        df_merged["avg_negative"] = 0.0
        df_merged["avg_neutral"] = 1.0
        df_merged["news_count"] = 0
        df_merged["net_sentiment"] = 0.0
        df_merged["news_present_flag"] = 0
    else:
        df_merged = pd.merge(df_features, df_sentiment, on=["stock_id", "date"], how="left")
        df_merged["news_present_flag"] = np.where(
            df_merged["news_count"].notna() & (df_merged["news_count"] > 0), 1, 0
        )
        df_merged["avg_positive"] = df_merged["avg_positive"].fillna(0.0)
        df_merged["avg_negative"] = df_merged["avg_negative"].fillna(0.0)
        df_merged["avg_neutral"] = df_merged["avg_neutral"].fillna(1.0)
        df_merged["news_count"] = df_merged["news_count"].fillna(0).astype(int)
        df_merged["net_sentiment"] = df_merged["net_sentiment"].fillna(0.0)

    df_merged = _add_sentiment_momentum(df_merged)

    if not df_nifty.empty:
        df_nifty["date"] = pd.to_datetime(df_nifty["date"]).dt.date
        df_merged = pd.merge(df_merged, df_nifty, on="date", how="left")
    else:
        for col in ["nifty_return", "nifty_rsi14", "nifty_ema20", "nifty_ema50", "nifty_volatility20"]:
            df_merged[col] = 0.0

    nifty_cols = ["nifty_return", "nifty_rsi14", "nifty_ema20", "nifty_ema50", "nifty_volatility20"]
    for col in nifty_cols:
        df_merged[col] = df_merged[col].ffill().fillna(0.0)

    df_merged["market_regime"] = np.where(
        df_merged["nifty_ema20"] > df_merged["nifty_ema50"], 1, 0
    ).astype(int)

    return df_merged


def validate_dataset(df_merged: pd.DataFrame) -> pd.DataFrame:
    if df_merged.empty:
        return df_merged

    df_valid = df_merged.dropna(subset=["target_close_t_plus_1"]).copy()

    df_valid["target_direction_t_plus_1"] = np.where(
        df_valid["target_return_t_plus_1"] > 0, 1, 0
    ).astype(int)

    required_cols = [
        "ema20", "ema50", "ema100", "sma20", "sma50", "rsi14",
        "macd", "macd_signal", "macd_hist", "bb_upper", "bb_middle", "bb_lower",
        "atr14", "daily_return", "log_return", "rolling_volatility20",
        "relative_volume", "momentum5", "momentum10", "momentum20",
        "target_close_t_plus_1", "target_return_t_plus_1"
    ]
    existing_cols = [col for col in required_cols if col in df_valid.columns]
    if existing_cols:
        df_valid = df_valid.dropna(subset=existing_cols).copy()

    return df_valid


# ─────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────

def save_dataset(db: Session, df_dataset: pd.DataFrame) -> dict:
    inserted = 0
    updated = 0
    errors = 0

    for _, row in df_dataset.iterrows():
        trade_date = row["date"]
        if isinstance(trade_date, str):
            trade_date = datetime.strptime(trade_date, "%Y-%m-%d").date()

        stock_id = int(row["stock_id"])

        def clean_val(val):
            if pd.isna(val) or val is None or (isinstance(val, float) and np.isinf(val)):
                return None
            return float(val)

        def clean_int(val, default=0):
            try:
                if pd.isna(val):
                    return default
                return int(val)
            except Exception:
                return default

        existing = db.query(TrainingDataset).filter(
            TrainingDataset.stock_id == stock_id,
            TrainingDataset.date == trade_date
        ).first()

        kwargs = dict(
            ema20=clean_val(row.get("ema20")),
            ema50=clean_val(row.get("ema50")),
            ema100=clean_val(row.get("ema100")),
            sma20=clean_val(row.get("sma20")),
            sma50=clean_val(row.get("sma50")),
            rsi14=clean_val(row.get("rsi14")),
            macd=clean_val(row.get("macd")),
            macd_signal=clean_val(row.get("macd_signal")),
            macd_hist=clean_val(row.get("macd_hist")),
            bb_upper=clean_val(row.get("bb_upper")),
            bb_middle=clean_val(row.get("bb_middle")),
            bb_lower=clean_val(row.get("bb_lower")),
            atr14=clean_val(row.get("atr14")),
            daily_return=clean_val(row.get("daily_return")),
            log_return=clean_val(row.get("log_return")),
            rolling_volatility20=clean_val(row.get("rolling_volatility20")),
            relative_volume=clean_val(row.get("relative_volume")),
            momentum5=clean_val(row.get("momentum5")),
            momentum10=clean_val(row.get("momentum10")),
            momentum20=clean_val(row.get("momentum20")),
            avg_positive=clean_val(row.get("avg_positive")),
            avg_negative=clean_val(row.get("avg_negative")),
            avg_neutral=clean_val(row.get("avg_neutral")),
            news_count=clean_int(row.get("news_count", 0)),
            net_sentiment=clean_val(row.get("net_sentiment")),
            news_present_flag=clean_int(row.get("news_present_flag", 0)),
            nifty_return=clean_val(row.get("nifty_return")),
            nifty_rsi14=clean_val(row.get("nifty_rsi14")),
            nifty_ema20=clean_val(row.get("nifty_ema20")),
            nifty_ema50=clean_val(row.get("nifty_ema50")),
            nifty_volatility20=clean_val(row.get("nifty_volatility20")),
            sentiment_3d_avg=clean_val(row.get("sentiment_3d_avg")),
            sentiment_7d_avg=clean_val(row.get("sentiment_7d_avg")),
            sentiment_change=clean_val(row.get("sentiment_change")),
            news_surprise=clean_val(row.get("news_surprise")),
            market_regime=clean_int(row.get("market_regime", 0)),
            target_close_t_plus_1=clean_val(row.get("target_close_t_plus_1")),
            target_return_t_plus_1=clean_val(row.get("target_return_t_plus_1")),
            target_direction_t_plus_1=clean_int(row.get("target_direction_t_plus_1", 0)),
        )

        if existing:
            for k, v in kwargs.items():
                setattr(existing, k, v)
            updated += 1
        else:
            db_row = TrainingDataset(stock_id=stock_id, date=trade_date, **kwargs)
            db.add(db_row)
            inserted += 1

    try:
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to commit dataset batch: {e}")
        errors = len(df_dataset)
        inserted = 0
        updated = 0

    return {"inserted": inserted, "updated": updated, "errors": errors}


def build_training_dataset(db: Session) -> dict:
    logger.info("Starting build_training_dataset...")

    try:
        deleted = db.query(TrainingDataset).delete()
        db.commit()
        logger.info(f"Cleared {deleted} existing records in training_dataset.")
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to clear training_dataset: {e}")

    df_features = load_feature_store(db)
    df_sentiment = load_sentiment_features(db)
    df_nifty = load_nifty_features(db)

    if df_features.empty:
        logger.warning("Feature store is empty. Cannot build training dataset.")
        return {"processed": 0, "inserted": 0, "updated": 0, "errors": 0}

    df_merged = merge_features(df_features, df_sentiment, df_nifty)
    df_valid = validate_dataset(df_merged)

    save_res = save_dataset(db, df_valid)

    summary = {
        "processed": len(df_valid),
        "inserted": save_res["inserted"],
        "updated": save_res["updated"],
        "errors": save_res["errors"],
    }

    logger.info(
        f"Training dataset build complete: "
        f"Processed: {summary['processed']}, "
        f"Inserted: {summary['inserted']}, "
        f"Updated: {summary['updated']}, "
        f"Errors: {summary['errors']}"
    )
    return summary

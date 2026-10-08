import logging
from datetime import date, datetime, timedelta
from typing import Optional
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sqlalchemy.orm import Session

from config.stocks import NIFTY_INDEX_SYMBOL
from database.models_db import NiftyFeatures

logger = logging.getLogger(__name__)


def get_latest_nifty_date(db: Session) -> Optional[date]:
    latest = db.query(NiftyFeatures).order_by(NiftyFeatures.date.desc()).first()
    return latest.date if latest else None


def download_nifty_data(start_date: date, end_date: date) -> pd.DataFrame:
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = (end_date + timedelta(days=1)).strftime("%Y-%m-%d")
    logger.info(f"Downloading NIFTY data ({NIFTY_INDEX_SYMBOL}) from {start_str} to {end_str}")

    ticker = yf.Ticker(NIFTY_INDEX_SYMBOL)
    df = ticker.history(start=start_str, end=end_str)

    if df.empty:
        logger.warning("No NIFTY data returned from yfinance.")
        return pd.DataFrame()

    df = df.reset_index()
    df["trade_date"] = df["Date"].dt.date
    df = df[["trade_date", "Close"]].rename(columns={"Close": "close"})
    df = df.sort_values("trade_date").reset_index(drop=True)
    return df


def compute_nifty_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "close" not in df.columns:
        return pd.DataFrame()

    df = df.copy()

    df["nifty_return"] = df["close"].pct_change()
    df["nifty_rsi14"] = ta.momentum.RSIIndicator(close=df["close"], window=14).rsi()
    df["nifty_ema20"] = ta.trend.EMAIndicator(close=df["close"], window=20).ema_indicator()
    df["nifty_ema50"] = ta.trend.EMAIndicator(close=df["close"], window=50).ema_indicator()
    df["nifty_volatility20"] = df["nifty_return"].rolling(window=20).std()

    df = df.rename(columns={"trade_date": "date"})
    return df[["date", "nifty_return", "nifty_rsi14", "nifty_ema20", "nifty_ema50", "nifty_volatility20"]]


def save_nifty_features(db: Session, df: pd.DataFrame) -> dict:
    inserted = 0
    updated = 0
    errors = 0

    df = df.dropna(subset=["nifty_rsi14", "nifty_ema50"]).copy()

    for _, row in df.iterrows():
        trade_date = row["date"]
        if isinstance(trade_date, str):
            trade_date = datetime.strptime(trade_date, "%Y-%m-%d").date()

        def clean(val):
            if pd.isna(val) or np.isinf(val):
                return None
            return float(val)

        try:
            existing = db.query(NiftyFeatures).filter(NiftyFeatures.date == trade_date).first()
            if existing:
                existing.nifty_return = clean(row["nifty_return"])
                existing.nifty_rsi14 = clean(row["nifty_rsi14"])
                existing.nifty_ema20 = clean(row["nifty_ema20"])
                existing.nifty_ema50 = clean(row["nifty_ema50"])
                existing.nifty_volatility20 = clean(row["nifty_volatility20"])
                updated += 1
            else:
                db_row = NiftyFeatures(
                    date=trade_date,
                    nifty_return=clean(row["nifty_return"]),
                    nifty_rsi14=clean(row["nifty_rsi14"]),
                    nifty_ema20=clean(row["nifty_ema20"]),
                    nifty_ema50=clean(row["nifty_ema50"]),
                    nifty_volatility20=clean(row["nifty_volatility20"]),
                )
                db.add(db_row)
                inserted += 1
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to upsert NIFTY feature for {trade_date}: {e}")
            errors += 1
            continue

    try:
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to commit NIFTY features batch: {e}")
        errors += inserted + updated
        inserted = 0
        updated = 0

    return {"inserted": inserted, "updated": updated, "errors": errors}


def update_nifty_features(db: Session) -> dict:
    today = date.today()
    latest_date = get_latest_nifty_date(db)

    if latest_date:
        start_date = latest_date - timedelta(days=60)
    else:
        start_date = today - timedelta(days=365 * 2)

    if start_date >= today:
        logger.info("NIFTY features are already up to date.")
        return {"downloaded": 0, "inserted": 0, "updated": 0, "errors": 0}

    raw_df = download_nifty_data(start_date, today)
    if raw_df.empty:
        return {"downloaded": 0, "inserted": 0, "updated": 0, "errors": 1}

    feature_df = compute_nifty_features(raw_df)
    if feature_df.empty:
        logger.warning("No features computed from NIFTY data.")
        return {"downloaded": len(raw_df), "inserted": 0, "updated": 0, "errors": 1}

    # On incremental updates, only process dates newer than latest_date
    if latest_date:
        feature_df = feature_df[feature_df["date"] > latest_date]

    save_res = save_nifty_features(db, feature_df)

    summary = {
        "downloaded": len(raw_df),
        "inserted": save_res["inserted"],
        "updated": save_res["updated"],
        "errors": save_res["errors"],
    }
    logger.info(
        f"NIFTY features update complete: "
        f"Downloaded={summary['downloaded']}, "
        f"Inserted={summary['inserted']}, "
        f"Updated={summary['updated']}, "
        f"Errors={summary['errors']}"
    )
    return summary

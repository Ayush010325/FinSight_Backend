import logging
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import ta
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from database.models_db import Stock, HistoricalPrice, FeatureStore
from services.market_data_service import get_stock_id

logger = logging.getLogger(__name__)

def load_price_history(db: Session, symbol: str, start_date: Optional[date] = None) -> pd.DataFrame:
    query = db.query(HistoricalPrice).filter(HistoricalPrice.symbol == symbol)
    if start_date:
        query = query.filter(HistoricalPrice.trade_date >= start_date)
    records = query.order_by(HistoricalPrice.trade_date.asc()).all()
    
    if not records:
        logger.warning(f"No price records found for {symbol}")
        return pd.DataFrame()
        
    df = pd.DataFrame([{
        "trade_date": r.trade_date,
        "open": r.open,
        "high": r.high,
        "low": r.low,
        "close": r.close,
        "volume": r.volume
    } for r in records])
    return df

def calculate_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        return df

    df['ema20'] = ta.trend.ema_indicator(close=df['close'], window=20)
    df['ema50'] = ta.trend.ema_indicator(close=df['close'], window=50)
    df['ema100'] = ta.trend.ema_indicator(close=df['close'], window=100)

    df['sma20'] = ta.trend.sma_indicator(close=df['close'], window=20)
    df['sma50'] = ta.trend.sma_indicator(close=df['close'], window=50)
    return df

def calculate_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        return df

    df['rsi14'] = ta.momentum.rsi(close=df['close'], window=14)

    macd_obj = ta.trend.MACD(close=df['close'])
    df['macd'] = macd_obj.macd()
    df['macd_signal'] = macd_obj.macd_signal()
    df['macd_hist'] = macd_obj.macd_diff()

    df['momentum5'] = df['close'].diff(5)
    df['momentum10'] = df['close'].diff(10)
    df['momentum20'] = df['close'].diff(20)
    return df

def calculate_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        return df

    bb_obj = ta.volatility.BollingerBands(close=df['close'], window=20)
    df['bb_upper'] = bb_obj.bollinger_hband()
    df['bb_middle'] = bb_obj.bollinger_mavg()
    df['bb_lower'] = bb_obj.bollinger_lband()

    df['atr14'] = ta.volatility.average_true_range(
        high=df['high'], low=df['low'], close=df['close'], window=14
    )
    return df

def calculate_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        return df

    df['volume_ma20'] = df['volume'].rolling(window=20).mean()

    df['relative_volume'] = np.where(
        df['volume_ma20'] > 0,
        df['volume'] / df['volume_ma20'],
        np.nan
    )
    
    df['volume_change'] = df['volume'].pct_change()
    return df

def calculate_return_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        return df
        
    df['daily_return'] = df['close'].pct_change()
    
    df['log_return'] = np.where(
        (df['close'].shift(1) > 0) & (df['close'] > 0),
        np.log(df['close'] / df['close'].shift(1)),
        np.nan
    )
    
    df['rolling_volatility20'] = df['daily_return'].rolling(window=20).std()
    return df

def calculate_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        return df

    df['target_close_t_plus_1'] = df['close'].shift(-1)

    df['target_return_t_plus_1'] = df['daily_return'].shift(-1)
    return df

def build_feature_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = calculate_trend_features(df)
    df = calculate_momentum_features(df)
    df = calculate_volatility_features(df)
    df = calculate_volume_features(df)
    df = calculate_return_features(df)
    df = calculate_target_variable(df)
    return df

def save_features(db: Session, symbol: str, feature_df: pd.DataFrame) -> dict:
    inserted = 0
    updated = 0
    errors = 0
    
    try:
        stock_id = get_stock_id(db, symbol)
    except ValueError as e:
        logger.error(f"Cannot save features for {symbol}: {e}")
        return {"inserted": 0, "updated": 0, "errors": len(feature_df)}
        
    for _, row in feature_df.iterrows():
        trade_date = row['trade_date']
        if isinstance(trade_date, str):
            trade_date = datetime.strptime(trade_date, "%Y-%m-%d").date()
            
        def clean_val(val):
            if pd.isna(val) or val is None or np.isinf(val):
                return None
            return float(val)
            
        existing = db.query(FeatureStore).filter(
            FeatureStore.stock_id == stock_id,
            FeatureStore.date == trade_date
        ).first()
        
        if existing:
            existing.ema20 = clean_val(row.get('ema20'))
            existing.ema50 = clean_val(row.get('ema50'))
            existing.ema100 = clean_val(row.get('ema100'))
            existing.sma20 = clean_val(row.get('sma20'))
            existing.sma50 = clean_val(row.get('sma50'))
            existing.rsi14 = clean_val(row.get('rsi14'))
            existing.macd = clean_val(row.get('macd'))
            existing.macd_signal = clean_val(row.get('macd_signal'))
            existing.macd_hist = clean_val(row.get('macd_hist'))
            existing.bb_upper = clean_val(row.get('bb_upper'))
            existing.bb_middle = clean_val(row.get('bb_middle'))
            existing.bb_lower = clean_val(row.get('bb_lower'))
            existing.atr14 = clean_val(row.get('atr14'))
            existing.daily_return = clean_val(row.get('daily_return'))
            existing.log_return = clean_val(row.get('log_return'))
            existing.rolling_volatility20 = clean_val(row.get('rolling_volatility20'))
            existing.volume_ma20 = clean_val(row.get('volume_ma20'))
            existing.relative_volume = clean_val(row.get('relative_volume'))
            existing.volume_change = clean_val(row.get('volume_change'))
            existing.momentum5 = clean_val(row.get('momentum5'))
            existing.momentum10 = clean_val(row.get('momentum10'))
            existing.momentum20 = clean_val(row.get('momentum20'))
            existing.target_close_t_plus_1 = clean_val(row.get('target_close_t_plus_1'))
            existing.target_return_t_plus_1 = clean_val(row.get('target_return_t_plus_1'))
            
            row_sent = clean_val(row.get('sentiment_score'))
            if row_sent is not None:
                existing.sentiment_score = row_sent
                
            try:
                db.commit()
                updated += 1
            except Exception as e:
                db.rollback()
                logger.error(f"Failed to update features for {symbol} on {trade_date}: {e}")
                errors += 1
        else:
            db_feat = FeatureStore(
                stock_id=stock_id,
                date=trade_date,
                ema20=clean_val(row.get('ema20')),
                ema50=clean_val(row.get('ema50')),
                ema100=clean_val(row.get('ema100')),
                sma20=clean_val(row.get('sma20')),
                sma50=clean_val(row.get('sma50')),
                rsi14=clean_val(row.get('rsi14')),
                macd=clean_val(row.get('macd')),
                macd_signal=clean_val(row.get('macd_signal')),
                macd_hist=clean_val(row.get('macd_hist')),
                bb_upper=clean_val(row.get('bb_upper')),
                bb_middle=clean_val(row.get('bb_middle')),
                bb_lower=clean_val(row.get('bb_lower')),
                atr14=clean_val(row.get('atr14')),
                daily_return=clean_val(row.get('daily_return')),
                log_return=clean_val(row.get('log_return')),
                rolling_volatility20=clean_val(row.get('rolling_volatility20')),
                volume_ma20=clean_val(row.get('volume_ma20')),
                relative_volume=clean_val(row.get('relative_volume')),
                volume_change=clean_val(row.get('volume_change')),
                momentum5=clean_val(row.get('momentum5')),
                momentum10=clean_val(row.get('momentum10')),
                momentum20=clean_val(row.get('momentum20')),
                target_close_t_plus_1=clean_val(row.get('target_close_t_plus_1')),
                target_return_t_plus_1=clean_val(row.get('target_return_t_plus_1')),
                sentiment_score=clean_val(row.get('sentiment_score'))
            )
            
            try:
                db.add(db_feat)
                db.commit()
                inserted += 1
            except Exception as e:
                db.rollback()
                logger.error(f"Failed to insert features for {symbol} on {trade_date}: {e}")
                errors += 1
                
    return {"inserted": inserted, "updated": updated, "errors": errors}

def generate_stock_features(db: Session, symbol: str) -> dict:
    logger.info(f"--- Starting feature engineering for {symbol} ---")
    
    try:
        stock_id = get_stock_id(db, symbol)
    except ValueError as e:
        logger.error(f"Stock symbol {symbol} is not registered or configured: {e}")
        return {"processed": 0, "inserted": 0, "updated": 0, "errors": 1}

    latest = db.query(FeatureStore).filter(
        FeatureStore.stock_id == stock_id
    ).order_by(FeatureStore.date.desc()).first()
    
    latest_feature_date = latest.date if latest else None
    
    if latest_feature_date:
        start_date = latest_feature_date - timedelta(days=150)
        df_price = load_price_history(db, symbol, start_date)
    else:
        df_price = load_price_history(db, symbol)
        
    if df_price.empty:
        logger.warning(f"No price history found for {symbol}, skipping features.")
        return {"processed": 0, "inserted": 0, "updated": 0, "errors": 0}
        
    df_features = build_feature_dataframe(df_price)
    
    if latest_feature_date:
        save_start_date = latest_feature_date - timedelta(days=3)
        df_to_save = df_features[df_features['trade_date'] >= save_start_date]
    else:
        df_to_save = df_features
        
    if df_to_save.empty:
        return {"processed": 0, "inserted": 0, "updated": 0, "errors": 0}
        
    save_results = save_features(db, symbol, df_to_save)
    
    summary = {
        "processed": len(df_to_save),
        "inserted": save_results["inserted"],
        "updated": save_results["updated"],
        "errors": save_results["errors"]
    }
    
    logger.info(
        f"Feature engineering completed for {symbol}: "
        f"Processed: {summary['processed']}, "
        f"Inserted: {summary['inserted']}, "
        f"Updated: {summary['updated']}, "
        f"Errors: {summary['errors']}"
    )
    return summary

def generate_all_features(db: Session) -> dict:
    from config.stocks import NIFTY_STOCKS
    logger.info("Starting feature engineering for all stocks...")
    
    overall_processed = 0
    overall_inserted = 0
    overall_updated = 0
    overall_errors = 0
    
    for stock_config in NIFTY_STOCKS:
        symbol = stock_config["symbol"]
        results = generate_stock_features(db, symbol)
        overall_processed += results["processed"]
        overall_inserted += results["inserted"]
        overall_updated += results["updated"]
        overall_errors += results["errors"]
        
    summary = {
        "processed": overall_processed,
        "inserted": overall_inserted,
        "updated": overall_updated,
        "errors": overall_errors
    }
    
    logger.info(
        f"--- All stocks feature engineering completed --- "
        f"Overall Processed: {overall_processed}, "
        f"Overall Inserted: {overall_inserted}, "
        f"Overall Updated: {overall_updated}, "
        f"Overall Errors: {overall_errors}"
    )
    return summary
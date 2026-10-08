import logging
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional
import yfinance as yf
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from config.stocks import NIFTY_STOCKS
from database.models_db import Stock, HistoricalPrice

logger = logging.getLogger(__name__)

def get_stock_id(db: Session, symbol: str) -> int:
    stock = db.query(Stock).filter(Stock.symbol == symbol).first()
    if stock:
        return stock.id
    stock_config = next((s for s in NIFTY_STOCKS if s["symbol"] == symbol), None)
    if not stock_config:
        raise ValueError(f"Stock symbol {symbol} is not configured in config/stocks.py")
    new_stock = Stock(
        symbol=symbol,
        company_name=stock_config["name"],
        sector=stock_config["sector"],
        industry=stock_config["industry"],
        is_active=True
    )
    db.add(new_stock)
    db.commit()
    db.refresh(new_stock)
    logger.info(f"Registered new stock: {symbol} (ID: {new_stock.id})")
    return new_stock.id

def get_latest_price_date(db: Session, symbol: str) -> Optional[date]:
    latest_price = db.query(HistoricalPrice).filter(
        HistoricalPrice.symbol == symbol
    ).order_by(HistoricalPrice.trade_date.desc()).first()
    return latest_price.trade_date if latest_price else None

def download_historical_data(symbol: str, start_date: date, end_date: date) -> List[Dict[str, Any]]:
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = (end_date + timedelta(days=1)).strftime("%Y-%m-%d")
    logger.info(f"Downloading data for {symbol} from {start_str} to {end_str}")
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_str, end=end_str)
    if df.empty:
        logger.warning(f"No data returned from yfinance for {symbol} in date range.")
        return []  
    records = []
    for index, row in df.iterrows():
        trade_date = index.date()
        records.append({
            "trade_date": trade_date,
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": int(row["Volume"])
        })
    return records

def validate_price_data(row: dict) -> bool:
    if row.get("trade_date") is None:
        logger.error(f"Invalid record: trade_date is NULL. Row: {row}")
        return False
    if row.get("close") is None or row.get("close") <= 0:
        logger.error(f"Invalid record: close price is <= 0 or NULL. Row: {row}")
        return False
    if row.get("volume") is None or row.get("volume") < 0:
        logger.error(f"Invalid record: volume is < 0 or NULL. Row: {row}")
        return False
    return True

def save_price_data(db: Session, symbol: str, prices: List[dict], data_source: str = "YahooFinance") -> dict:
    inserted = 0
    skipped = 0
    errors = 0
    for price in prices:
        hp = HistoricalPrice(
            symbol=symbol,
            trade_date=price["trade_date"],
            open=price.get("open"),
            high=price.get("high"),
            low=price.get("low"),
            close=price.get("close"),
            volume=price.get("volume"),
            data_source=data_source
        )
        try:
            db.add(hp)
            db.commit()
            inserted += 1
        except IntegrityError:
            db.rollback()
            skipped += 1
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to insert record {price} for {symbol}: {e}")
            errors += 1
            
    return {"inserted": inserted, "skipped": skipped, "errors": errors}

def update_stock_history(db: Session, symbol: str) -> dict:
    logger.info(f"--- Starting incremental update for {symbol} ---")
    try:
        get_stock_id(db, symbol)
    except ValueError as e:
        logger.error(f"Skipping update for {symbol}: {e}")
        return {"downloaded": 0, "inserted": 0, "skipped": 0, "errors": 1}
        
    latest_date = get_latest_price_date(db, symbol)
    today = date.today()
    if latest_date:
        start_date = latest_date + timedelta(days=1)
    else:
        start_date = today - timedelta(days=365 * 2)
    if start_date >= today:
        logger.info(f"Stock {symbol} is already up to date. Latest date: {latest_date}")
        return {"downloaded": 0, "inserted": 0, "skipped": 0, "errors": 0}
        
    downloaded_records = download_historical_data(symbol, start_date, today)
    downloaded_count = len(downloaded_records)
    
    valid_records = [r for r in downloaded_records if validate_price_data(r)]
    errors_count = downloaded_count - len(valid_records)
    
    save_results = save_price_data(db, symbol, valid_records)
    
    summary = {
        "downloaded": downloaded_count,
        "inserted": save_results["inserted"],
        "skipped": save_results["skipped"],
        "errors": save_results["errors"] + errors_count
    }
    
    logger.info(
        f"Update completed for {symbol}: "
        f"Downloaded: {summary['downloaded']}, "
        f"Inserted: {summary['inserted']}, "
        f"Skipped: {summary['skipped']}, "
        f"Errors: {summary['errors']}"
    )
    return summary

def update_all_stocks(db: Session) -> None:
    logger.info("Starting historical market data update for all stocks...")
    overall_downloaded = 0
    overall_inserted = 0
    overall_skipped = 0
    overall_errors = 0
    
    for stock_config in NIFTY_STOCKS:
        symbol = stock_config["symbol"]
        results = update_stock_history(db, symbol)
        overall_downloaded += results["downloaded"]
        overall_inserted += results["inserted"]
        overall_skipped += results["skipped"]
        overall_errors += results["errors"]
        
    logger.info(
        f"--- All stocks update completed --- "
        f"Overall Downloaded: {overall_downloaded}, "
        f"Overall Inserted: {overall_inserted}, "
        f"Overall Skipped: {overall_skipped}, "
        f"Overall Errors: {overall_errors}"
    )

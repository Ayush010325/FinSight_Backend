import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import get_db_connection

TABLES = {
    "stocks": """
        CREATE TABLE IF NOT EXISTS stocks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT UNIQUE NOT NULL,
            company_name TEXT NOT NULL,
            sector TEXT,
            industry TEXT,
            is_active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """,
    "historical_prices": """
        CREATE TABLE IF NOT EXISTS historical_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            trade_date DATE NOT NULL,
            open REAL, high REAL, low REAL, close REAL, volume BIGINT,
            data_source TEXT DEFAULT 'YahooFinance',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, trade_date)
        );
    """,
    "news_articles": """
        CREATE TABLE IF NOT EXISTS news_articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT,
            source TEXT,
            published_at TIMESTAMP,
            article_url TEXT UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """,
    "sentiment_scores": """
        CREATE TABLE IF NOT EXISTS sentiment_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_id INTEGER NOT NULL,
            positive_score REAL,
            negative_score REAL,
            neutral_score REAL,
            sentiment_label TEXT,
            sentiment_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(article_id) REFERENCES news_articles(id)
        );
    """,
    "feature_store": """
        CREATE TABLE IF NOT EXISTS feature_store (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stock_id INTEGER NOT NULL,
            date DATE NOT NULL,
            ema20 REAL,
            ema50 REAL,
            ema100 REAL,
            sma20 REAL,
            sma50 REAL,
            rsi14 REAL,
            macd REAL,
            macd_signal REAL,
            macd_hist REAL,
            bb_upper REAL,
            bb_middle REAL,
            bb_lower REAL,
            atr14 REAL,
            daily_return REAL,
            log_return REAL,
            rolling_volatility20 REAL,
            volume_ma20 REAL,
            relative_volume REAL,
            volume_change REAL,
            momentum5 REAL,
            momentum10 REAL,
            momentum20 REAL,
            target_close_t_plus_1 REAL,
            target_return_t_plus_1 REAL,
            sentiment_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(stock_id, date),
            FOREIGN KEY(stock_id) REFERENCES stocks(id)
        );
    """,
    "model_registry": """
        CREATE TABLE IF NOT EXISTS model_registry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            model_version TEXT UNIQUE,
            model_path TEXT,
            scaler_path TEXT,
            accuracy REAL,
            trained_until DATE,
            is_active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """,
    "training_runs": """
        CREATE TABLE IF NOT EXISTS training_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            model_version TEXT,
            status TEXT DEFAULT 'queued',
            epochs INTEGER,
            train_samples INTEGER,
            validation_accuracy REAL,
            loss REAL,
            started_at TIMESTAMP,
            completed_at TIMESTAMP
        );
    """,
    "predictions": """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            model_version TEXT,
            prediction_date DATE,
            predictions TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """,
    "prediction_results": """
        CREATE TABLE IF NOT EXISTS prediction_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id INTEGER,
            actual_day_1 TEXT,
            actual_day_2 TEXT,
            actual_day_3 TEXT,
            actual_day_4 TEXT,
            actual_day_5 TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(prediction_id) REFERENCES predictions(id)
        );
    """,
    "daily_sentiment_features": """
        CREATE TABLE IF NOT EXISTS daily_sentiment_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stock_id INTEGER NOT NULL,
            date DATE NOT NULL,
            avg_positive REAL,
            avg_negative REAL,
            avg_neutral REAL,
            news_count INTEGER,
            net_sentiment REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(stock_id, date),
            FOREIGN KEY(stock_id) REFERENCES stocks(id)
        );
    """,
    "nifty_features": """
        CREATE TABLE IF NOT EXISTS nifty_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE NOT NULL UNIQUE,
            nifty_return REAL,
            nifty_rsi14 REAL,
            nifty_ema20 REAL,
            nifty_ema50 REAL,
            nifty_volatility20 REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """,
    "training_dataset": """
        CREATE TABLE IF NOT EXISTS training_dataset (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stock_id INTEGER NOT NULL,
            date DATE NOT NULL,
            ema20 REAL, ema50 REAL, ema100 REAL,
            sma20 REAL, sma50 REAL,
            rsi14 REAL,
            macd REAL, macd_signal REAL, macd_hist REAL,
            bb_upper REAL, bb_middle REAL, bb_lower REAL,
            atr14 REAL,
            daily_return REAL, log_return REAL,
            rolling_volatility20 REAL,
            relative_volume REAL,
            momentum5 REAL, momentum10 REAL, momentum20 REAL,
            avg_positive REAL, avg_negative REAL, avg_neutral REAL,
            news_count INTEGER, net_sentiment REAL,
            news_present_flag INTEGER,
            nifty_return REAL, nifty_rsi14 REAL,
            nifty_ema20 REAL, nifty_ema50 REAL,
            nifty_volatility20 REAL,
            sentiment_3d_avg REAL, sentiment_7d_avg REAL,
            sentiment_change REAL,
            news_surprise REAL,
            market_regime INTEGER,
            target_close_t_plus_1 REAL,
            target_return_t_plus_1 REAL,
            target_direction_t_plus_1 INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(stock_id, date),
            FOREIGN KEY(stock_id) REFERENCES stocks(id)
        );
    """
}

def init_database():
    print("Initializing database...")
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        for table_name, create_sql in TABLES.items():
            cursor.execute(create_sql)
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

if __name__ == "__main__":
    init_database()

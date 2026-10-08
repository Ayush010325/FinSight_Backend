from sqlalchemy import Column, Integer, String, Float, Boolean, Date, DateTime, JSON
from datetime import datetime
from database.database import Base

class ModelRecord(Base):
    __tablename__ = "model_registry"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, nullable=False)
    model_version = Column(String, unique=True, nullable=True)
    model_path = Column(String, nullable=True)
    scaler_path = Column(String, nullable=True)
    accuracy = Column(Float, nullable=True)
    trained_until = Column(Date, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class TrainingJobRecord(Base):
    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, nullable=True)
    model_version = Column(String, nullable=True)
    status = Column(String, default="queued")
    epochs = Column(Integer, nullable=True)
    train_samples = Column(Integer, nullable=True)
    validation_accuracy = Column(Float, nullable=True)
    loss = Column(Float, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, nullable=True)
    model_version = Column(String, nullable=True)
    prediction_date = Column(Date, nullable=True)
    predictions = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Stock(Base):
    __tablename__ = "stocks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, unique=True, nullable=False)
    company_name = Column(String, nullable=False)
    sector = Column(String, nullable=True)
    industry = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class HistoricalPrice(Base):
    __tablename__ = "historical_prices"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, nullable=False)
    trade_date = Column(Date, nullable=False)
    open = Column(Float, nullable=True)
    high = Column(Float, nullable=True)
    low = Column(Float, nullable=True)
    close = Column(Float, nullable=True)
    volume = Column(Integer, nullable=True)
    data_source = Column(String, default="YahooFinance")
    created_at = Column(DateTime, default=datetime.utcnow)

class NewsArticle(Base):
    __tablename__ = "news_articles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, nullable=False)
    title = Column(String, nullable=False)
    description = Column(String, nullable=True)
    source = Column(String, nullable=True)
    published_at = Column(DateTime, nullable=True)
    article_url = Column(String, unique=True, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class FeatureStore(Base):
    __tablename__ = "feature_store"

    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_id = Column(Integer, nullable=False)
    date = Column(Date, nullable=False)
    ema20 = Column(Float, nullable=True)
    ema50 = Column(Float, nullable=True)
    ema100 = Column(Float, nullable=True)
    sma20 = Column(Float, nullable=True)
    sma50 = Column(Float, nullable=True)
    rsi14 = Column(Float, nullable=True)
    macd = Column(Float, nullable=True)
    macd_signal = Column(Float, nullable=True)
    macd_hist = Column(Float, nullable=True)
    bb_upper = Column(Float, nullable=True)
    bb_middle = Column(Float, nullable=True)
    bb_lower = Column(Float, nullable=True)
    atr14 = Column(Float, nullable=True)
    daily_return = Column(Float, nullable=True)
    log_return = Column(Float, nullable=True)
    rolling_volatility20 = Column(Float, nullable=True)
    volume_ma20 = Column(Float, nullable=True)
    relative_volume = Column(Float, nullable=True)
    volume_change = Column(Float, nullable=True)
    momentum5 = Column(Float, nullable=True)
    momentum10 = Column(Float, nullable=True)
    momentum20 = Column(Float, nullable=True)
    target_close_t_plus_1 = Column(Float, nullable=True)
    target_return_t_plus_1 = Column(Float, nullable=True)
    sentiment_score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class SentimentScore(Base):
    __tablename__ = "sentiment_scores"

    id = Column(Integer, primary_key=True, autoincrement=True)
    article_id = Column(Integer, nullable=False)
    positive_score = Column(Float, nullable=True)
    negative_score = Column(Float, nullable=True)
    neutral_score = Column(Float, nullable=True)
    sentiment_label = Column(String, nullable=True)
    sentiment_score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class DailySentimentFeatures(Base):
    __tablename__ = "daily_sentiment_features"

    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_id = Column(Integer, nullable=False)
    date = Column(Date, nullable=False)
    avg_positive = Column(Float, nullable=True)
    avg_negative = Column(Float, nullable=True)
    avg_neutral = Column(Float, nullable=True)
    news_count = Column(Integer, nullable=True)
    net_sentiment = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class NiftyFeatures(Base):
    __tablename__ = "nifty_features"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, unique=True)
    nifty_return = Column(Float, nullable=True)
    nifty_rsi14 = Column(Float, nullable=True)
    nifty_ema20 = Column(Float, nullable=True)
    nifty_ema50 = Column(Float, nullable=True)
    nifty_volatility20 = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class TrainingDataset(Base):
    __tablename__ = "training_dataset"

    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_id = Column(Integer, nullable=False)
    date = Column(Date, nullable=False)
    ema20 = Column(Float, nullable=True)
    ema50 = Column(Float, nullable=True)
    ema100 = Column(Float, nullable=True)
    sma20 = Column(Float, nullable=True)
    sma50 = Column(Float, nullable=True)
    rsi14 = Column(Float, nullable=True)
    macd = Column(Float, nullable=True)
    macd_signal = Column(Float, nullable=True)
    macd_hist = Column(Float, nullable=True)
    bb_upper = Column(Float, nullable=True)
    bb_middle = Column(Float, nullable=True)
    bb_lower = Column(Float, nullable=True)
    atr14 = Column(Float, nullable=True)
    daily_return = Column(Float, nullable=True)
    log_return = Column(Float, nullable=True)
    rolling_volatility20 = Column(Float, nullable=True)
    relative_volume = Column(Float, nullable=True)
    momentum5 = Column(Float, nullable=True)
    momentum10 = Column(Float, nullable=True)
    momentum20 = Column(Float, nullable=True)
    avg_positive = Column(Float, nullable=True)
    avg_negative = Column(Float, nullable=True)
    avg_neutral = Column(Float, nullable=True)
    news_count = Column(Integer, nullable=True)
    net_sentiment = Column(Float, nullable=True)
    news_present_flag = Column(Integer, nullable=True)
    nifty_return = Column(Float, nullable=True)
    nifty_rsi14 = Column(Float, nullable=True)
    nifty_ema20 = Column(Float, nullable=True)
    nifty_ema50 = Column(Float, nullable=True)
    nifty_volatility20 = Column(Float, nullable=True)
    sentiment_3d_avg = Column(Float, nullable=True)
    sentiment_7d_avg = Column(Float, nullable=True)
    sentiment_change = Column(Float, nullable=True)
    news_surprise = Column(Float, nullable=True)
    market_regime = Column(Integer, nullable=True)
    target_close_t_plus_1 = Column(Float, nullable=True)
    target_return_t_plus_1 = Column(Float, nullable=True)
    target_direction_t_plus_1 = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)




import logging
from datetime import date, datetime
from typing import List, Dict, Any, Optional
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sqlalchemy.orm import Session
from sqlalchemy import func

from database.models_db import Stock, NewsArticle, SentimentScore, DailySentimentFeatures

logger = logging.getLogger(__name__)

def load_unsentimented_articles(db: Session) -> List[NewsArticle]:
    return db.query(NewsArticle).outerjoin(
        SentimentScore, NewsArticle.id == SentimentScore.article_id
    ).filter(SentimentScore.article_id == None).all()

def load_finbert_model() -> tuple:
    logger.info("Loading ProsusAI FinBERT model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return tokenizer, model

def predict_sentiment(text: str, tokenizer, model) -> dict:
    if not text.strip():
        return {
            "label": "Neutral",
            "positive": 0.0,
            "negative": 0.0,
            "neutral": 1.0,
            "confidence": 1.0
        }
        
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        
    probs = F.softmax(outputs.logits, dim=-1).numpy()[0]
    
    positive_score = float(probs[0])
    negative_score = float(probs[1])
    neutral_score = float(probs[2])
    
    labels = ["Positive", "Negative", "Neutral"]
    max_idx = probs.argmax()
    label = labels[max_idx]
    confidence = float(probs[max_idx])
    
    return {
        "label": label,
        "positive": positive_score,
        "negative": negative_score,
        "neutral": neutral_score,
        "confidence": confidence
    }

def save_sentiment(
    db: Session,
    article_id: int,
    label: str,
    positive: float,
    negative: float,
    neutral: float,
    confidence: float
) -> SentimentScore:
    db_score = SentimentScore(
        article_id=article_id,
        sentiment_label=label,
        positive_score=positive,
        negative_score=negative,
        neutral_score=neutral,
        sentiment_score=confidence
    )
    try:
        db.add(db_score)
        db.commit()
        db.refresh(db_score)
        return db_score
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to save sentiment score for article {article_id}: {e}")
        raise e

def process_article(db: Session, article: NewsArticle, tokenizer, model) -> dict:
    text = f"{article.title}. {article.description or ''}".strip()
    res = predict_sentiment(text, tokenizer, model)
    save_sentiment(
        db,
        article_id=article.id,
        label=res["label"],
        positive=res["positive"],
        negative=res["negative"],
        neutral=res["neutral"],
        confidence=res["confidence"]
    )
    return res

def process_all_articles(db: Session) -> dict:
    articles = load_unsentimented_articles(db)
    if not articles:
        logger.info("No unsentimented articles found.")
        return {"processed": 0, "inserted": 0, "errors": 0}
        
    logger.info(f"Found {len(articles)} unsentimented articles.")
    tokenizer, model = load_finbert_model()
    
    inserted = 0
    errors = 0
    for i, art in enumerate(articles, 1):
        if i % 50 == 0 or i == len(articles):
            logger.info(f"Processing progress: {i}/{len(articles)} articles completed.")
        try:
            process_article(db, art, tokenizer, model)
            inserted += 1
        except Exception as e:
            logger.error(f"Error processing article ID {art.id}: {e}")
            errors += 1
            
    return {"processed": len(articles), "inserted": inserted, "errors": errors}

def aggregate_daily_sentiment(db: Session) -> dict:
    logger.info("Aggregating daily sentiment scores...")

    results = db.query(
        Stock.id.label("stock_id"),
        func.date(NewsArticle.published_at).label("date_str"),
        func.avg(SentimentScore.positive_score).label("avg_pos"),
        func.avg(SentimentScore.negative_score).label("avg_neg"),
        func.avg(SentimentScore.neutral_score).label("avg_neu"),
        func.count(NewsArticle.id).label("cnt")
    ).join(
        SentimentScore, NewsArticle.id == SentimentScore.article_id
    ).join(
        Stock, NewsArticle.symbol == Stock.symbol
    ).filter(
        NewsArticle.published_at != None
    ).group_by(
        Stock.id,
        func.date(NewsArticle.published_at)
    ).all()
    
    inserted = 0
    updated = 0
    errors = 0
    
    for r in results:
        try:
            trade_date = datetime.strptime(r.date_str, "%Y-%m-%d").date()
            avg_positive = float(r.avg_pos) if r.avg_pos is not None else 0.0
            avg_negative = float(r.avg_neg) if r.avg_neg is not None else 0.0
            avg_neutral = float(r.avg_neu) if r.avg_neu is not None else 0.0
            news_count = int(r.cnt)
            net_sentiment = avg_positive - avg_negative
            
            existing = db.query(DailySentimentFeatures).filter(
                DailySentimentFeatures.stock_id == r.stock_id,
                DailySentimentFeatures.date == trade_date
            ).first()
            
            if existing:
                existing.avg_positive = avg_positive
                existing.avg_negative = avg_negative
                existing.avg_neutral = avg_neutral
                existing.news_count = news_count
                existing.net_sentiment = net_sentiment
                updated += 1
            else:
                db_feat = DailySentimentFeatures(
                    stock_id=r.stock_id,
                    date=trade_date,
                    avg_positive=avg_positive,
                    avg_negative=avg_negative,
                    avg_neutral=avg_neutral,
                    news_count=news_count,
                    net_sentiment=net_sentiment
                )
                db.add(db_feat)
                inserted += 1
                
            db.commit()
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to upsert daily sentiment feature: {e}")
            errors += 1
            
    summary = {
        "inserted": inserted,
        "updated": updated,
        "errors": errors
    }
    logger.info(
        f"Daily sentiment aggregation completed: "
        f"Inserted: {inserted}, Updated: {updated}, Errors: {errors}"
    )
    return summary

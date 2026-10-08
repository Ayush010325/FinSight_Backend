import logging
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
import email.utils
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func

from config.stocks import NIFTY_STOCKS
from database.models_db import NewsArticle

logger = logging.getLogger(__name__)

def build_rss_query(company_name: str) -> str:
    query = f'"{company_name}" stock OR "{company_name}" NSE'
    encoded_query = urllib.parse.quote(query)
    return f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"

def fetch_rss_xml(url: str) -> str:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    )
    with urllib.request.urlopen(req, timeout=15) as response:
        return response.read().decode("utf-8")

def parse_rss_xml(xml_content: str) -> List[Dict[str, Any]]:
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        logger.error(f"Failed to parse XML content: {e}")
        return []

    channel = root.find("channel")
    if channel is None:
        return []

    items = []
    for item in channel.findall("item"):
        title = item.findtext("title", "")
        link = item.findtext("link", "")
        pub_date = item.findtext("pubDate", "")
        description = item.findtext("description", "")
        
        source_elem = item.find("source")
        publisher = source_elem.text if source_elem is not None else ""
        
        items.append({
            "title": title,
            "link": link,
            "pubDate": pub_date,
            "description": description,
            "publisher": publisher
        })
    return items

def parse_article(entry: dict, symbol: str) -> dict:
    published_at = None
    if entry.get("pubDate"):
        try:
            dt = email.utils.parsedate_to_datetime(entry["pubDate"])
            published_at = dt.astimezone(timezone.utc).replace(tzinfo=None)
        except Exception as e:
            logger.debug(f"Failed to parse published date '{entry.get('pubDate')}': {e}")

    return {
        "symbol": symbol,
        "headline": entry.get("title", "").strip(),
        "summary": entry.get("description", "").strip(),
        "url": entry.get("link", "").strip(),
        "publisher": entry.get("publisher", "").strip(),
        "published_at": published_at
    }

def validate_article(article: dict) -> bool:
    headline = article.get("headline", "")
    url = article.get("url", "")
    published_at = article.get("published_at")

    if not headline:
        logger.error("Validation error: headline is empty.")
        return False
    if len(headline) < 10:
        logger.error(f"Validation error: headline '{headline}' is less than 10 characters.")
        return False
    if not url:
        logger.error("Validation error: url is empty.")
        return False
    if published_at is None:
        logger.error("Validation error: published_at date is null.")
        return False

    return True

def get_latest_news_timestamp(db: Session, symbol: str) -> Optional[datetime]:
    latest = db.query(NewsArticle).filter(
        NewsArticle.symbol == symbol
    ).order_by(NewsArticle.published_at.desc()).first()
    return latest.published_at if latest else None

def save_articles(db: Session, symbol: str, articles: List[dict]) -> dict:
    inserted = 0
    skipped = 0
    errors = 0

    for article in articles:
        url_dup = db.query(NewsArticle).filter(NewsArticle.article_url == article["url"]).first()
        if url_dup:
            skipped += 1
            continue
        fallback_dup = db.query(NewsArticle).filter(
            NewsArticle.symbol == symbol,
            NewsArticle.title == article["headline"],
            NewsArticle.published_at == article["published_at"]
        ).first()
        if fallback_dup:
            skipped += 1
            continue

        db_article = NewsArticle(
            symbol=symbol,
            title=article["headline"],
            description=article.get("summary"),
            source=article.get("publisher"),
            published_at=article["published_at"],
            article_url=article["url"]
        )
        try:
            db.add(db_article)
            db.commit()
            inserted += 1
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to save article '{article['headline']}' to database: {e}")
            errors += 1

    return {"inserted": inserted, "skipped": skipped, "errors": errors}

def update_stock_news(db: Session, symbol: str, company_name: str) -> dict:
    logger.info(f"--- Starting News Ingestion for {symbol} ---")

    latest_ts = get_latest_news_timestamp(db, symbol)

    url = build_rss_query(company_name)
    try:
        xml_content = fetch_rss_xml(url)
        raw_items = parse_rss_xml(xml_content)
    except Exception as e:
        logger.error(f"Failed to fetch news feed for {symbol}: {e}")
        return {"retrieved": 0, "inserted": 0, "skipped": 0, "errors": 1}
        
    retrieved_count = len(raw_items)
    
    parsed_articles = [parse_article(item, symbol) for item in raw_items]
    valid_articles = [art for art in parsed_articles if validate_article(art)]
    validation_failures = len(parsed_articles) - len(valid_articles)

    newer_articles = []
    skipped_by_date = 0
    for art in valid_articles:
        if latest_ts and art["published_at"] <= latest_ts:
            skipped_by_date += 1
            continue
        newer_articles.append(art)

    save_results = save_articles(db, symbol, newer_articles)
    
    summary = {
        "retrieved": retrieved_count,
        "inserted": save_results["inserted"],
        "skipped": save_results["skipped"] + skipped_by_date,
        "errors": save_results["errors"] + validation_failures
    }
    
    logger.info(f"symbol: {symbol}")
    logger.info(f"Retrieved: {summary['retrieved']}")
    logger.info(f"Inserted: {summary['inserted']}")
    logger.info(f"Skipped: {summary['skipped']}")
    logger.info(f"Errors: {summary['errors']}")
    
    return summary

def update_all_news(db: Session) -> None:
    logger.info("Starting financial news update for all stocks...")
    overall_retrieved = 0
    overall_inserted = 0
    overall_skipped = 0
    overall_errors = 0
    
    for stock_config in NIFTY_STOCKS:
        symbol = stock_config["symbol"]
        company_name = stock_config["name"]
        results = update_stock_news(db, symbol, company_name)
        overall_retrieved += results["retrieved"]
        overall_inserted += results["inserted"]
        overall_skipped += results["skipped"]
        overall_errors += results["errors"]
        
    logger.info(
        f"--- All stocks news update completed --- "
        f"Overall Retrieved: {overall_retrieved}, "
        f"Overall Inserted: {overall_inserted}, "
        f"Overall Skipped: {overall_skipped}, "
        f"Overall Errors: {overall_errors}"
    )

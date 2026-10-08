"""
run_enhanced_pipeline.py
=========================
End-to-end runner for the enhanced feature pipeline:

    1. Run DB migration (ADD COLUMN for new training_dataset columns)
    2. Run NIFTY 50 data ingestion & feature computation
    3. Rebuild training_dataset with all new features
    4. Report counts and sample rows

Usage:
    python scratch/run_enhanced_pipeline.py
"""

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_enhanced_pipeline")

from database.database import SessionLocal
from database.connection import get_db_connection
from services.nifty_service import update_nifty_features
from services.dataset_builder import build_training_dataset
from database.models_db import NiftyFeatures, TrainingDataset


def run_migration():
    """
    SQLite ALTER TABLE migration to add new columns if they don't already exist.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    new_columns = [
        ("nifty_return", "REAL"),
        ("nifty_rsi14", "REAL"),
        ("nifty_ema20", "REAL"),
        ("nifty_ema50", "REAL"),
        ("nifty_volatility20", "REAL"),
        ("sentiment_3d_avg", "REAL"),
        ("sentiment_7d_avg", "REAL"),
        ("sentiment_change", "REAL"),
        ("news_surprise", "REAL"),
        ("market_regime", "INTEGER"),
        ("target_direction_t_plus_1", "INTEGER"),
    ]

    for col_name, col_type in new_columns:
        try:
            cursor.execute(f"ALTER TABLE training_dataset ADD COLUMN {col_name} {col_type};")
            logger.info(f"  [MIGRATION] Added column training_dataset.{col_name} ({col_type})")
        except Exception:
            # Column already exists — safe to ignore
            pass

    # Create nifty_features table if it doesn't exist
    cursor.execute("""
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
    """)
    conn.commit()
    conn.close()
    logger.info("Migration complete.")


def main():
    logger.info("=" * 60)
    logger.info("Enhanced Feature Pipeline — Start")
    logger.info("=" * 60)

    # Step 0: Run DB migration (add new columns safely)
    logger.info("\n[Step 0] Running database migration...")
    run_migration()

    db = SessionLocal()
    try:
        # Step 1: NIFTY 50 features
        logger.info("\n[Step 1] Downloading NIFTY 50 data & computing market features...")
        nifty_res = update_nifty_features(db)
        logger.info(f"  NIFTY result: {nifty_res}")

        # Verify nifty_features table
        nifty_count = db.query(NiftyFeatures).count()
        logger.info(f"  nifty_features table rows: {nifty_count}")

        # Step 2: Rebuild training_dataset with all features
        logger.info("\n[Step 2] Rebuilding training_dataset (full enhanced feature set)...")
        ds_res = build_training_dataset(db)
        logger.info(f"  Dataset result: {ds_res}")

        # Step 3: Verify
        total_rows = db.query(TrainingDataset).count()
        logger.info(f"\n[Step 3] Verification:")
        logger.info(f"  training_dataset total rows: {total_rows}")

        # Sample a few rows and check new columns
        sample = db.query(TrainingDataset).limit(5).all()
        for row in sample:
            nifty_rsi = f"{row.nifty_rsi14:.4f}" if row.nifty_rsi14 is not None else "NULL"
            ns = f"{row.news_surprise:.4f}" if row.news_surprise is not None else "NULL"
            s3d = f"{row.sentiment_3d_avg:.4f}" if row.sentiment_3d_avg is not None else "NULL"
            logger.info(
                f"  Date={row.date}, stock_id={row.stock_id}, "
                f"nifty_rsi14={nifty_rsi}, "
                f"market_regime={row.market_regime}, "
                f"news_surprise={ns}, "
                f"sentiment_3d_avg={s3d}, "
                f"target_direction={row.target_direction_t_plus_1}"
            )

        # Check nulls in new columns
        logger.info("\n[Step 4] Null check on new columns:")
        from sqlalchemy import func
        for col_name in ["nifty_return", "nifty_rsi14", "market_regime", "sentiment_3d_avg", "news_surprise", "target_direction_t_plus_1"]:
            col_attr = getattr(TrainingDataset, col_name)
            null_count = db.query(func.count()).filter(col_attr == None).scalar()
            logger.info(f"  {col_name}: {null_count} NULLs / {total_rows} total")

    finally:
        db.close()

    logger.info("\n" + "=" * 60)
    logger.info("Enhanced Feature Pipeline — Complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

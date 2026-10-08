import os
import sqlite3
from contextlib import contextmanager

DB_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(DB_DIR, "stock_predictor.db")

def get_db_connection(db_path: str = None) -> sqlite3.Connection:
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    conn.execute("PRAGMA foreign_keys = ON;")
    
    return conn

@contextmanager
def get_db_cursor(db_path: str = None):
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        yield cursor
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

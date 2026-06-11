import sqlite3
import hashlib
import logging
from datetime import datetime
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AuditLogger")

DB_PATH = "autopilot_analytics.db"

def init_audit_db(db_path: str = DB_PATH) -> None:
    """Initializes the production immutable request transaction audit log table."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # Core audit schema matching exact infrastructure specs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS request_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                prompt_hash TEXT NOT NULL,
                complexity_tier_assigned INTEGER NOT NULL,
                routed_model TEXT NOT NULL,
                cost REAL NOT NULL,
                latency REAL NOT NULL,
                quality_score REAL,
                was_escalated INTEGER NOT NULL CHECK (was_escalated IN (0, 1))
            )
        """)
        
        # Index on prompt_hash is vital for quick lookups, deduplication, 
        # and downstream semantic cache mapping metrics.
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_prompt_hash ON request_logs(prompt_hash);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON request_logs(timestamp);")
        conn.commit()
    logger.info("Audit database storage engine and indices verified successfully.")

def generate_prompt_hash(prompt: str) -> str:
    """Generates a secure, deterministic SHA-256 hex signature from the raw prompt string."""
    cleaned_prompt = prompt.strip().encode('utf-8')
    return hashlib.sha256(cleaned_prompt).hexdigest()

def record_request_telemetry(
    prompt: str,
    complexity_tier_assigned: int,
    routed_model: str,
    cost: float,
    latency: float,
    quality_score: Optional[float],
    was_escalated: bool,
    db_path: str = DB_PATH
) -> str:
    """
    Persists a comprehensive record of a request execution event into the database.
    Catches exceptions locally to isolate core request loops from log engine faults.
    """
    prompt_hash = generate_prompt_hash(prompt)
    escalated_flag = 1 if was_escalated else 0
    
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO request_logs (
                    prompt_hash, complexity_tier_assigned, routed_model, 
                    cost, latency, quality_score, was_escalated
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                prompt_hash,
                complexity_tier_assigned,
                routed_model,
                cost,
                latency,
                quality_score,
                escalated_flag
            ))
            conn.commit()
        logger.info(f"[TELEMETRY LOGGED] Saved audit trace for prompt hash {prompt_hash[:12]}...")
    except Exception as e:
        # Fault isolation pattern: telemetry tracking errors must never crash live services
        logger.error(f"[TELEMETRY CRITICAL FAILURE] Failed to write audit trail record: {str(e)}")
        
    return prompt_hash

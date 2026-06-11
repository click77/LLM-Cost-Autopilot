import os
import time
import logging
import sqlite3
from registry import ModelRegistry
from verifier import evaluate_response_quality
from abstraction import Response

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BackgroundWorker")

DB_PATH = os.getenv("DATABASE_PATH", "autopilot_analytics.db")

def process_pending_verifications():
    """Polls database for cheap tier entries missing quality scores, running evaluations asynchronously."""
    registry = ModelRegistry()
    gold_config = registry.get_model("gpt-4o")
    
    while True:
        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Fetch transaction items requiring evaluation
                cursor.execute("""
                    SELECT id, prompt_hash, complexity_tier_assigned, routed_model, cost, latency, quality_score
                    FROM request_logs 
                    WHERE quality_score IS NULL AND complexity_tier_assigned < 3
                    LIMIT 5
                """)
                pending_jobs = cursor.fetchall()
                
                if not pending_jobs:
                    time.sleep(2.0) # Back off smoothly if the execution queue is empty
                    continue
                    
                for job in pending_jobs:
                    logger.info(f"[WORKER PROCESSOR] Ingesting pending verification log ID: {job['id']}")
                    
                    # For simulation stability, we mock a simulated cheap payload check context
                    # In real production, map this back to raw text strings or cached logs
                    mock_cheap_response = Response(
                        model_id=job["routed_model"],
                        output_text="Simulated verification response tracking content context.",
                        input_tokens=100,
                        output_tokens=150,
                        cost=job["cost"],
                        latency_seconds=job["latency"]
                    )
                    
                    # Execute evaluation pipeline metrics
                    # Hardcoded task parameter to generic fallback classification tracking
                    quality_score, was_escalated = evaluate_response_quality(
                        prompt="Simulated prompt reference string via audit logs.",
                        cheap_response=mock_cheap_response,
                        task_type="classification",
                        gold_config=gold_config
                    )
                    
                    # Update table parameters
                    cursor.execute("""
                        UPDATE request_logs
                        SET quality_score = ?, was_escalated = ?
                        WHERE id = ?
                    """, (quality_score, 1 if was_escalated else 0, job["id"]))
                    conn.commit()
                    
                    logger.info(f"[WORKER SUCCESS] Updated Job ID {job['id']} with Quality Score: {quality_score}")
                    
        except Exception as e:
            logger.error(f"[WORKER CRITICAL FAULT] Execution loop disrupted: {str(e)}")
            time.sleep(5.0)

if __name__ == "__main__":
    logger.info("Initializing Autopilot Background Verification Microservice Daemon...")
    # Verify file system state is ready before booting polling worker loop
    db_dir = os.path.dirname(DB_PATH)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
    process_pending_verifications()

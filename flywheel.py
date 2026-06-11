import os
import sqlite3
import joblib
import logging
import pandas as pd
from typing import Dict, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Import dependencies from previous steps
from dataset import generate_labeled_dataset, extract_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FeedbackFlywheel")

DB_PATH = "autopilot_analytics.db"
MODEL_PATH = "model_v1.joblib"

def init_analytics_db(db_path: str = DB_PATH) -> None:
    """Initializes the SQLite data storage layer for capturing operational routing failures."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS routing_failures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT NOT NULL,
                assigned_tier INTEGER NOT NULL,
                true_tier INTEGER NOT NULL,
                token_count INTEGER NOT NULL,
                contains_analyze_compare INTEGER NOT NULL,
                constraint_count INTEGER NOT NULL,
                context_provided INTEGER NOT NULL,
                output_format_complexity INTEGER NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    logger.info("Analytics engine SQLite data tables verified.")

def log_routing_failure(prompt: str, assigned_tier: int, true_tier: int) -> None:
    """
    Pipes an operational routing failure event directly into the training tables.
    Extracts dynamic text features inline to ensure perfect dataset uniformity.
    """
    # Safeguard: Re-extract numerical features from text using original feature engineering
    # Assuming baseline context default = 1, base constraint default = 2 for tracking consistency
    features = extract_features(prompt, context_provided=True, base_constraints=2, base_format_complexity=1)
    
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO routing_failures (
                prompt, assigned_tier, true_tier, token_count, 
                contains_analyze_compare, constraint_count, context_provided, output_format_complexity
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            prompt, assigned_tier, true_tier, 
            features["token_count"], 
            features["contains_analyze_compare"], 
            features["constraint_count"], 
            features["context_provided"], 
            features["output_format_complexity"]
        ))
        conn.commit()
    logger.warning(f"[FLYWHEEL CAPTURE] Logged routing mismatch! Prompt routed to Tier {assigned_tier} re-assigned to Tier {true_tier}.")

def run_weekly_retraining_loop() -> None:
    """
    The Flywheel Engine: Ingests production failure data from SQLite, merges it 
    with the original baseline golden training set, and optimizes the Random Forest model.
    """
    logger.info("Starting scheduled routing classifier retraining pipeline...")
    
    # 1. Ingest original baseline training dataset
    base_df = generate_labeled_dataset()
    base_df = base_df.rename(columns={"tier_label": "true_tier"})
    
    # 2. Extract logged failures from production database
    if not os.path.exists(DB_PATH):
        init_analytics_db()
        
    with sqlite3.connect(DB_PATH) as conn:
        failures_df = pd.read_sql_query("SELECT * FROM routing_failures", conn)
        
    logger.info(f"Retrieved {len(failures_df)} new production failure logs from SQLite tables.")
    
    feature_cols = [
        "token_count", 
        "contains_analyze_compare", 
        "constraint_count", 
        "context_provided", 
        "output_format_complexity"
    ]
    
    # 3. Dynamic dataset concatenation if failures are present
    if not failures_df.empty:
        # Prune columns to match training schema signatures precisely
        failures_clean = failures_df[feature_cols + ["true_tier"]]
        base_clean = base_df[feature_cols + ["true_tier"]]
        
        # Up-weight production failures to ensure the random forest prioritizes recent errors
        failures_weighted = pd.concat([failures_clean] * 3, ignore_index=True)
        
        training_matrix = pd.concat([base_clean, failures_weighted], ignore_index=True)
        logger.info(f"Constructed combined dataset. Total training volume: {len(training_matrix)} samples.")
    else:
        training_matrix = base_df
        logger.info("Zero production anomalies discovered. Retraining random forest on golden baseline records.")

    # 4. Separate training inputs and label vectors
    X = training_matrix[feature_cols]
    y = training_matrix["true_tier"]
    
    # 5. Initialize clean model architecture and update checkpoint
    updated_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        random_state=42,
        class_weight="balanced"
    )
    updated_model.fit(X, y)
    
    # Simple self-test validation score evaluation
    predictions = updated_model.predict(X)
    accuracy = accuracy_score(y, predictions)
    logger.info(f"Model update optimized. Evaluation baseline accuracy: {accuracy * 100:.2f}%")
    
    # 6. Save updated artifacts back to disk for hot-reloading by FastAPI layers
    artifact_payload = {
        "model_object": updated_model,
        "input_features": feature_cols,
        "version_tag": "1.1.0_auto_updated"
    }
    joblib.dump(artifact_payload, MODEL_PATH)
    logger.info(f"Production routing artifact model cleanly serialized and overwritten at '{MODEL_PATH}'.")

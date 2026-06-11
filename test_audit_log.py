import sqlite3
import pandas as pd
from audit_log import init_audit_db, record_request_telemetry, generate_prompt_hash

print("=== STARTING COST AUTOPILOT AUDIT TELEMETRY SYSTEM VALIDATION ===")

# 1. Initialize DB tables and indices
init_audit_db()

# 2. Simulate standard user interactions running across various configurations
test_records = [
    {
        "prompt": "Calculate quarterly revenue churn across our EMEA division.",
        "tier": 2,
        "model": "claude-3-haiku",
        "cost": 0.00021,
        "latency": 0.32,
        "score": 4.5,
        "escalated": False
    },
    {
        "prompt": "Write an enterprise multi-tenant database connection pool manager in Rust.",
        "tier": 3,
        "model": "gpt-4o",
        "cost": 0.01450,
        "latency": 2.14,
        "score": 4.9,
        "escalated": True
    },
    {
        "prompt": "Calculate quarterly revenue churn across our EMEA division.", # Duplicate prompt to check hash consistency
        "tier": 2,
        "model": "gpt-4o-mini",
        "cost": 0.00045,
        "latency": 0.41,
        "score": 4.7,
        "escalated": False
    }
]

print("\n[Piping live application telemetry metrics directly into SQLite database...]")
for record in test_records:
    record_request_telemetry(
        prompt=record["prompt"],
        complexity_tier_assigned=record["tier"],
        routed_model=record["model"],
        cost=record["cost"],
        latency=record["latency"],
        quality_score=record["score"],
        was_escalated=record["escalated"]
    )

# 3. Read back and verify formatting with Pandas to confirm integrity of schema signatures
print("\n=== VERIFYING RECOVERED AUDIT TRAIL DATA CHUNKS ===")
with sqlite3.connect("autopilot_analytics.db") as conn:
    df = pd.read_sql_query("SELECT * FROM request_logs", conn)

print(df.to_string(index=False))

# 4. Check that identical prompts yielded identical hashes
hash_1 = generate_prompt_hash("Calculate quarterly revenue churn across our EMEA division.")
hash_2 = df.iloc[0]['prompt_hash']
hash_3 = df.iloc[2]['prompt_hash']

assert hash_1 == hash_2 == hash_3, "Cryptographic prompt hash mismatch occurred!"
print("\n[Verification Success]: Identical text inputs match perfectly across sequential logs.")
print("=== TELEMETRY TRAIL INTEGRITY VERIFIED ===")

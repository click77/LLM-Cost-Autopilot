import sqlite3
import random
from datetime import datetime, timedelta

DB_PATH = "autopilot_analytics.db"

def seed_telemetry_database():
    """Populates the database with realistic time-series transaction metrics across 14 days."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        
        # Ensure target log table layout is present
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
                was_escalated INTEGER NOT NULL
            )
        """)
        
        print("[SEED WORKER] Generating 1,500 mock transaction records across a 14-day history window...")
        
        base_time = datetime.now() - timedelta(days=14)
        models = ["claude-3-haiku", "gpt-4o-mini", "gpt-4o"]
        
        for i in range(1500):
            # Advance times incrementally to simulate continuous active day/night variations
            record_time = base_time + timedelta(
                days=random.randint(0, 13), 
                hours=random.randint(0, 23), 
                minutes=random.randint(0, 59)
            )
            
            # Weighted baseline choice modeling: 60% simple tier, 30% moderate tier, 10% highly complex
            dice_roll = random.random()
            if dice_roll < 0.60:
                tier = 1
                model = "claude-3-haiku"
                cost = random.uniform(0.0001, 0.0003)
                latency = random.uniform(0.15, 0.40)
                # High chance of solid quality score, tiny chance of poor score resulting in escalation
                if random.random() < 0.04:
                    quality_score = random.uniform(2.1, 3.4)
                    was_escalated = 1
                else:
                    quality_score = random.uniform(4.2, 5.0)
                    was_escalated = 0
            elif dice_roll < 0.90:
                tier = 2
                model = "gpt-4o-mini"
                cost = random.uniform(0.0003, 0.0009)
                latency = random.uniform(0.25, 0.65)
                if random.random() < 0.06:
                    quality_score = random.uniform(2.5, 3.5)
                    was_escalated = 1
                else:
                    quality_score = random.uniform(4.0, 4.9)
                    was_escalated = 0
            else:
                tier = 3
                model = "gpt-4o"
                cost = random.uniform(0.0120, 0.0280)  # Significantly more expensive
                latency = random.uniform(0.90, 2.30)
                quality_score = random.uniform(4.5, 5.0)  # High tier rarely misses
                was_escalated = 0
                
            # If an item was escalated via guardrail loop, adjust tracking cost metrics
            if was_escalated == 1:
                model = "gpt-4o (Escalated)"
                cost += random.uniform(0.0150, 0.0250)
                latency += random.uniform(1.0, 2.1)
                quality_score = random.uniform(4.6, 5.0) # Upgraded response quality fixed
                
            p_hash = f"hash_{random.getrandbits(64):x}"
            
            cursor.execute("""
                INSERT INTO request_logs (
                    timestamp, prompt_hash, complexity_tier_assigned, routed_model, 
                    cost, latency, quality_score, was_escalated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (record_time.strftime('%Y-%m-%d %H:%M:%S'), p_hash, tier, model, cost, latency, quality_score, was_escalated))
            
        conn.commit()
    print("[SEED SUCCESS] DB seeding phase executed completely.")

if __name__ == "__main__":
    seed_telemetry_database()

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Import the generation matrix built in Step 2
from dataset import generate_labeled_dataset

def train_routing_classifier(model_output_path: str = "model_v1.joblib"):
    print("==========================================================")
    print("     PHASE 2, STEP 3: LLM ROUTING CLASSIFIER TRAINING")
    print("==========================================================\n")
    
    # 1. Ingest the hand-labeled dataset
    print("[1/5] Ingesting feature-engineered dataset matrix...")
    df = generate_labeled_dataset()
    
    # Define exact input space features and target vector labels
    feature_cols = [
        "token_count", 
        "contains_analyze_compare", 
        "constraint_count", 
        "context_provided", 
        "output_format_complexity"
    ]
    X = df[feature_cols]
    y = df["tier_label"]
    
    # 2. Execute a stratified split to ensure perfectly uniform class ratios
    print("[2/5] Constructing stratified 80/20 train-test split splits...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # 3. Initialize and fit the random forest classifier
    print("[3/5] Training Random Forest routing skeleton model...")
    # Balanced weights protect minority data if failure feedback loops skew logs later
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)
    
    # 4. Evaluate performance parameters on the held-out data block
    print("[4/5] Executing inference validation on held-out test block...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[1, 2, 3])
    report = classification_report(y_test, y_pred, labels=[1, 2, 3])
    
    print(f"\n-> Held-Out Test Set Accuracy Baseline: {accuracy * 100:.2f}%")
    if accuracy >= 0.80:
        print("   [SUCCESS] V1 Performance exceeds the 80% routing threshold.")
    else:
        print("   [CRITICAL WARNING] Model V1 falls short of acceptable routing accuracy bounds.")
        
    print("\n--- Model Precision & Recall Report ---")
    print(report)
    
    print("--- Detailed Confusion Matrix Matrix ---")
    print("                 Predicted Tier 1   Predicted Tier 2   Predicted Tier 3")
    print(f"Actual Tier 1 |        {cm[0][0]:<14}     {cm[0][1]:<14}     {cm[0][2]:<14}")
    print(f"Actual Tier 2 |        {cm[1][0]:<14}     {cm[1][1]:<14}     {cm[1][2]:<14}")
    print(f"Actual Tier 3 |        {cm[2][0]:<14}     {cm[2][1]:<14}     {cm[2][2]:<14}")
    
    # 5. Under-Routing Safety Violation Audit
    # Under-routing (Actual Tier 3 -> Predicted Tier 1) causes system outages/failures
    # because complex code or multi-step reasoning is routed to an inadequate LLM engine.
    under_routing_faults = cm[2][0] 
    print("\n--- Under-Routing Risk Evaluation ---")
    print(f"-> Catastrophic Under-Routing Violations (Tier 3 -> Tier 1): {under_routing_faults}")
    
    if under_routing_faults > 0:
        print("   [ALERT - HIGH RISK] Tier 3 items are bleeding directly into Tier 1 models!")
        print("   Resolution Strategy: Inject sample_weight penalization to increase Tier 3 costs.")
    else:
        print("   [PASS] Zero Under-Routing breaches encountered. Safety constraints are active.")
        
    # 6. Serialize and store the compiled asset package
    print(f"\n[5/5] Serializing production artifact payload to: {model_output_path}...")
    artifact_payload = {
        "model_object": model,
        "input_features": feature_cols,
        "version_tag": "1.0.0"
    }
    joblib.dump(artifact_payload, model_output_path)
    print("\n==========================================================")
    print(" TRAINING PHASE SKELETON RECOVERED: ARTIFACT READY FOR API")
    print("==========================================================")

if __name__ == "__main__":
    train_routing_classifier()

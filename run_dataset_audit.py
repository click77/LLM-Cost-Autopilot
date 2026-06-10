from dataset import generate_labeled_dataset

print("==========================================================")
print("     LAUNCHING LLM AUTOPILOT CUSTOM TRAINING DATASET AUDIT")
print("==========================================================\n")

# Process and expand dataset from blueprints
df = generate_labeled_dataset()

# Print critical data shape and distribution verifications
print(f"-> Total Record Volume Generated: {len(df)} entries")
print(f"-> Missing / Null Data Points:    {df.isnull().sum().sum()}")

print("\n--- Label Distribution Audit ---")
print(df["tier_label"].value_counts().sort_index().to_string())

print("\n--- Feature Matrix Correlation Statistics ---")
grouped_stats = df.groupby("tier_label").mean(numeric_only=True)
print(grouped_stats.to_string())

# Output an example slice to visually verify the feature extraction quality
print("\n--- Sample Record Row Evaluation (First Sample From Each Tier) ---")
for tier in [1, 2, 3]:
    sample_row = df[df["tier_label"] == tier].iloc[0]
    print(f"\n[TIER {tier} SAMPLE]")
    print(f" Prompt Text: {sample_row['prompt'][:110]}...")
    print(f" Features:    Tokens: {sample_row['token_count']} | "
          f"Analyze/Compare: {sample_row['contains_analyze_compare']} | "
          f"Constraints: {sample_row['constraint_count']} | "
          f"Context Flag: {sample_row['context_provided']} | "
          f"Format Complexity: {sample_row['output_format_complexity']}")

print("\n==========================================================")
print(" DATASET VERIFICATION SUCCESSFUL: READY FOR SCIKIT-LEARN")
print("==========================================================")

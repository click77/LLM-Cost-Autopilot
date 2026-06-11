from thresholds import QualityThresholds

print("=== Testing Extraction Threshold ===")
# Mock: Cheap model missed the 'amount' key
cheap_json = '{"name": "Acme Corp", "date": "2026-10-12"}'
exp_json = '{"name": "Acme Corp", "date": "2026-10-12", "amount": 1500.00}'
res_ext = QualityThresholds.verify_extraction(cheap_json, exp_json)
print(f"Passed: {res_ext.passed} | Reason: {res_ext.reason}")

print("\n=== Testing Classification Threshold ===")
# Mock: Cheap model added conversational filler, but got the label right
cheap_class = "Based on the text, the category is High_Urgency."
exp_class = "highurgency"
res_class = QualityThresholds.verify_classification(cheap_class, exp_class)
print(f"Passed: {res_class.passed} | Reason: {res_class.reason}")

print("\n=== Testing Summarization LLM-Judge Prompt ===")
# Mock: Generating the payload for the async verifier
prompt = QualityThresholds.build_llm_judge_prompt(
    "The server crashed at 3AM due to an out-of-memory error caused by a memory leak in the Redis cache.", 
    "Server went down because of Redis memory leak."
)
print(prompt)

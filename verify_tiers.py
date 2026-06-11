from features import extract_prompt_features

# Sample cases perfectly mapping to our 3 explicit tiers
test_cases = {
    "Tier 1 (Simple)": "Extract all email addresses from this text: 'Please contact support@company.com or sales@company.com for inquiries.' Format as JSON.",
    "Tier 2 (Moderate)": "Summarize the following customer support log into 3 bullet points. Classify whether the user sentiment is frustrated or neutral.",
    "Tier 3 (Complex)": "Write a Python script that implements an async pipeline with exponential backoff retries. Evaluate the memory footprint of using a thread pool versus asyncio, and think step-by-step through the failure scenarios."
}

print("=== Complexity Boundaries Diagnostic Check ===")
for tier_name, prompt in test_cases.items():
    metrics = extract_prompt_features(prompt)
    print(f"\nTarget: {tier_name}")
    print(f" -> Token Size: {metrics['token_count']} words approx")
    print(f" -> Keyword Densities: T1: {metrics['tier1_keyword_density']} | T2: {metrics['tier2_keyword_density']} | T3: {metrics['tier3_keyword_density']}")
    print(f" -> Constraint Count: {metrics['constraint_count']}")
    print(f" -> Has Code Syntax Signal: {metrics['contains_code_syntax']}")

from registry import ModelRegistry

# 1. Instantiate the active registry mapping
registry = ModelRegistry()

# 2. Extract a low-cost model configuration and a frontier reasoning model configuration
cheap_model = registry.get_model("gpt-4o-mini")
expensive_model = registry.get_model("gpt-4o")

# 3. Simulate an operations request tracking 1,500 input tokens and 500 generated output tokens
prompt_in = 1500
completion_out = 500

cost_mini = cheap_model.calculate_cost(prompt_in, completion_out)
cost_4o = expensive_model.calculate_cost(prompt_in, completion_out)

print(f"=== Model Architecture Check ===")
print(f"Routed to Mini: ${cost_mini:.6f} via {cheap_model.provider} (Tier: {cheap_model.quality_tier})")
print(f"Routed to Full: ${cost_4o:.6f} via {expensive_model.provider} (Tier: {expensive_model.quality_tier})")
print(f"Immediate Arbitrage Savings Delta: ${cost_4o - cost_mini:.6f}")

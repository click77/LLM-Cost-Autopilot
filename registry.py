from dataclasses import dataclass
from typing import Dict

@dataclass
class ModelConfig:
    provider: str               # e.g., "openai", "anthropic", "ollama"
    model_id: str               # The exact identifier passed to the API or provider
    cost_per_input_token: float  # Absolute cost for 1 incoming prompt token
    cost_per_output_token: float # Absolute cost for 1 outgoing completion token
    avg_latency_seconds: float   # Historical/expected average time to first token/completion
    quality_tier: str           # Must be strict: "high", "medium", or "low"

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Computes the precise monetary cost of a request based on usage metrics.
        This provides the exact mathematical figures required for the cost dashboard.
        """
        input_cost = input_tokens * self.cost_per_input_token
        output_cost = output_tokens * self.cost_per_output_token
        return input_cost + output_cost


class ModelRegistry:
    def __init__(self):
        # Initializing registry using current standard pricing models
        self._registry: Dict[str, ModelConfig] = {
            
            # --- HIGH QUALITY TIER ---
            "gpt-4o": ModelConfig(
                provider="openai",
                model_id="gpt-4o",
                cost_per_input_token=2.50 / 1_000_000,    # $2.50 per 1M tokens
                cost_per_output_token=10.00 / 1_000_000,  # $10.00 per 1M tokens
                avg_latency_seconds=2.2,
                quality_tier="high"
            ),
            "claude-sonnet": ModelConfig(
                provider="anthropic",
                model_id="claude-3-5-sonnet-latest",       # Maps to current stable Sonnet
                cost_per_input_token=3.00 / 1_000_000,    # $3.00 per 1M tokens
                cost_per_output_token=15.00 / 1_000_000,  # $15.00 per 1M tokens
                avg_latency_seconds=2.5,
                quality_tier="high"
            ),
            
            # --- MEDIUM QUALITY TIER ---
            "gpt-4o-mini": ModelConfig(
                provider="openai",
                model_id="gpt-4o-mini",
                cost_per_input_token=0.15 / 1_000_000,    # $0.15 per 1M tokens
                cost_per_output_token=0.60 / 1_000_000,  # $0.60 per 1M tokens
                avg_latency_seconds=0.8,
                quality_tier="medium"
            ),
            "claude-haiku": ModelConfig(
                provider="anthropic",
                model_id="claude-3-5-haiku-latest",        # Maps to current stable Haiku
                cost_per_input_token=0.80 / 1_000_000,    # $0.80 per 1M tokens
                cost_per_output_token=4.00 / 1_000_000,   # $4.00 per 1M tokens
                avg_latency_seconds=0.7,
                quality_tier="medium"
            ),
            
            # --- LOW QUALITY TIER ---
            "local-llama": ModelConfig(
                provider="ollama",
                model_id="llama3.1:8b",                    # Standard engineering baseline for local serving
                cost_per_input_token=0.0,                  # Free local compute resource
                cost_per_output_token=0.0,                 # Free local compute resource
                avg_latency_seconds=1.2,                   # Hardware dependent (assumes mid-tier local GPU/Mac setup)
                quality_tier="low"
            )
        }

    def get_model(self, key: str) -> ModelConfig:
        """Retrieves a model's config schema safely."""
        if key not in self._registry:
            raise KeyError(f"Model keys must be explicitly listed in registry. {key} not found.")
        return self._registry[key]

    def list_all_models(self) -> Dict[str, ModelConfig]:
        """Exposes the full registry map for configuration and metadata diagnostics."""
        return self._registry

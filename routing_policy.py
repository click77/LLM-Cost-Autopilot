import os
import yaml
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

# Define Type-Safe Structures for the Policy Layer Configuration
class ModelTarget(BaseModel):
    tier_name: str
    description: str
    provider: str
    model_id: str
    fallback_model_id: str
    max_tokens: int
    temperature: float

class RuntimePolicies(BaseModel):
    retry_attempts: int
    timeout_seconds: float
    fallback_on_http_codes: List[int]

class RoutingPolicyConfig(BaseModel):
    version: str
    last_modified: str
    routing_map: Dict[int, ModelTarget]
    runtime_policies: RuntimePolicies


class PolicyEngine:
    """ Manages runtime evaluation of the Cost Autopilot's model routing matrix. """
    
    def __init__(self, filepath: str = "routing_policy.yaml"):
        self.filepath = filepath
        self.policy: Optional[RoutingPolicyConfig] = None
        self.load_policy()

    def load_policy(self) -> None:
        """ Parses and structural-validates raw configuration matrix assets from disk. """
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Critical System Failure: Policy file missing at {self.filepath}")
            
        with open(self.filepath, "r") as stream:
            try:
                raw_data = yaml.safe_load(stream)
                # Parse through Pydantic data schemas to block structural failures
                self.policy = RoutingPolicyConfig(**raw_data)
                print(f"[POLICY MATCH] Policy Configuration v{self.policy.version} successfully parsed.")
            except Exception as e:
                raise ValueError(f"Structural Validation Error inside routing_policy.yaml: {e}")

    def resolve_route(self, tier: int) -> ModelTarget:
        """ Returns the exact target infrastructure data package required for an inference run. """
        if self.policy is None:
            self.load_policy()
            
        if tier not in self.policy.routing_map:
            print(f"[UNRECOGNIZED TIER] Warning: Requested tier {tier} undefined. Safe-failing to Tier 3 fallback.")
            return self.policy.routing_map[3] # Fail-safe upward to secure query completion integrity
            
        return self.policy.routing_map[tier]

# Local evaluation suite for checking configuration runtime behavior
if __name__ == "__main__":
    print("--- Running Local Routing Map Policy Test ---")
    engine = PolicyEngine()
    
    # Simulate routing decisions across classified incoming complexity levels
    for target_tier in [1, 2, 3, 99]:
        route = engine.resolve_route(target_tier)
        print(f"\n[Input Complexity Classified]: Tier {target_tier}")
        print(f" -> Routed Target System     : {route.tier_name}")
        print(f" -> Provider Target Endpoint : {route.provider.upper()}")
        print(f" -> Active Production Model  : {route.model_id}")
        print(f" -> Backup Resiliency Engine : {route.fallback_model_id}")

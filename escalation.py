import logging
import json
from typing import Optional, Dict, Any
from dataclasses import dataclass

# Import our previously built components
from registry import ModelRegistry
from routing_policy import PolicyEngine
from abstraction import send_request, Response
from thresholds import QualityThresholds, VerificationResult

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EscalationEngine")

@dataclass
class EscalateLog:
    original_model: str
    escalated_model: str
    cost_delta: float
    quality_gap_score: float

class AutoEscalator:
    def __init__(self, registry: ModelRegistry, policy: PolicyEngine):
        self.registry = registry
        self.policy = policy

    def _evaluate_quality_sync(self, original_prompt: str, cheap_resp: Response, task_type: str) -> VerificationResult:
        """Runs a synchronous quality check to determine if escalation is necessary."""
        tier3_config = self.registry.get_model("gpt-4o")
        
        # We need the gold standard to compare against for extraction/classification
        if task_type in ["extraction", "classification"]:
            gold_resp = send_request(original_prompt, tier3_config)
            if task_type == "extraction":
                return QualityThresholds.verify_extraction(cheap_resp.output_text, gold_resp.output_text)
            else:
                return QualityThresholds.verify_classification(cheap_resp.output_text, gold_resp.output_text)
                
        elif task_type == "summarization":
            judge_prompt = QualityThresholds.build_llm_judge_prompt(original_prompt, cheap_resp.output_text)
            judge_resp = send_request(judge_prompt, tier3_config)
            try:
                data = json.loads(judge_resp.output_text)
                score = float(data.get("score", 0))
                passed = score >= QualityThresholds.MIN_SUMMARY_SCORE
                return VerificationResult(passed, score, data.get("reason", ""), "summarization")
            except Exception:
                return VerificationResult(False, 0.0, "Judge formatting failed", "summarization")
        
        return VerificationResult(True, 1.0, "Unknown task type, bypassing verification", task_type)

    def execute_with_escalation(
        self, 
        prompt: str, 
        predicted_tier: int, 
        task_type: str, 
        strict_sla: bool = True
    ) -> Dict[str, Any]:
        """
        Executes the prompt. If strict_sla is False, it verifies and auto-escalates on failure.
        If strict_sla is True, it returns immediately and assumes async verification handles logs.
        """
        # 1. Resolve initial routing policy
        initial_route = self.policy.resolve_route(predicted_tier)
        cheap_model_config = self.registry.get_model(initial_route.model_id)
        
        logger.info(f"Executing Tier {predicted_tier} via {cheap_model_config.model_id}...")
        cheap_response = send_request(prompt, cheap_model_config)

        # 2. Strict SLA (Real-time): Return immediately, let async verifier handle the rest
        if strict_sla:
            logger.info("Strict SLA active. Returning immediately without sync-escalation.")
            return {
                "final_response": cheap_response,
                "escalated": False,
                "log": None
            }

        # 3. Flexible SLA (Batch/Async): Synchronously verify the output
        logger.info("Flexible SLA active. Running inline quality verification...")
        verification = self._evaluate_quality_sync(prompt, cheap_response, task_type)

        if verification.passed:
            logger.info(f"Quality check passed (Score: {verification.score}). No escalation needed.")
            return {
                "final_response": cheap_response,
                "escalated": False,
                "log": None
            }

        # 4. Escalation Triggered: Route to fallback model
        logger.warning(f"Quality failure detected (Score: {verification.score}). Reason: {verification.reason}")
        fallback_model_config = self.registry.get_model(initial_route.fallback_model_id)
        
        logger.info(f"ESCALATING to {fallback_model_config.model_id}...")
        escalated_response = send_request(prompt, fallback_model_config)

        # 5. Build the strict audit log for future ML tuning
        cost_delta = escalated_response.cost + cheap_response.cost # Total cost incurred by doing both
        quality_gap = QualityThresholds.MIN_SUMMARY_SCORE - verification.score if task_type == "summarization" else 1.0

        escalation_log = EscalateLog(
            original_model=cheap_model_config.model_id,
            escalated_model=fallback_model_config.model_id,
            cost_delta=cost_delta,
            quality_gap_score=quality_gap
        )

        return {
            "final_response": escalated_response,
            "escalated": True,
            "log": escalation_log
        }

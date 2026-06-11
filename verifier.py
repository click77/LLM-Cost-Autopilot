import asyncio
import logging
import json
from typing import Optional

# Import the core components built in Phase 1 and Phase 3
from abstraction import send_request, Response
from registry import ModelConfig
from thresholds import QualityThresholds, VerificationResult

# Configure standard logging for the verifier
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AsyncVerifier")

async def run_verification_job(
    original_prompt: str,
    cheap_response: Response,
    task_type: str,
    tier3_config: ModelConfig
) -> None:
    """
    Executes the async background verification loop. 
    Compares the cheap model's output against a Tier 3 gold standard.
    """
    logger.info(f"[VERIFIER START] Auditing {cheap_response.model_id} for '{task_type}' task...")

    try:
        # 1. Fetch the Gold Standard (Tier 3) response
        # We use asyncio.to_thread to prevent the synchronous API call from blocking FastAPI
        gold_response = await asyncio.to_thread(send_request, original_prompt, tier3_config)

        # 2. Route to the strict threshold evaluation logic
        result: Optional[VerificationResult] = None

        if task_type == "extraction":
            result = QualityThresholds.verify_extraction(
                cheap_output=cheap_response.output_text, 
                expensive_output=gold_response.output_text
            )

        elif task_type == "classification":
            result = QualityThresholds.verify_classification(
                cheap_output=cheap_response.output_text, 
                expensive_output=gold_response.output_text
            )

        elif task_type == "summarization":
            # Summarization requires the Tier 3 model to act as a judge
            judge_prompt = QualityThresholds.build_llm_judge_prompt(
                original_prompt=original_prompt, 
                cheap_output=cheap_response.output_text
            )
            
            # Fire the judge prompt to the Tier 3 model
            judge_response = await asyncio.to_thread(send_request, judge_prompt, tier3_config)
            
            try:
                # Parse the exact JSON contract demanded from the judge
                judge_data = json.loads(judge_response.output_text)
                score = float(judge_data.get("score", 0))
                reason = judge_data.get("reason", "No reason provided")
                passed = score >= QualityThresholds.MIN_SUMMARY_SCORE
                result = VerificationResult(passed, score, reason, "summarization")
            except json.JSONDecodeError:
                result = VerificationResult(False, 0.0, "Judge model returned invalid JSON structure.", "summarization")
        
        else:
            logger.warning(f"Unknown task type '{task_type}' passed to verifier. Skipping.")
            return

        # 3. Log the final routing outcome and calculate the arbitrage capture
        if result.passed:
            savings = gold_response.cost - cheap_response.cost
            logger.info(f"[VERIFIER PASS] {cheap_response.model_id} output verified. Captured Arbitration Savings: ${savings:.6f}")
        else:
            logger.error(f"[VERIFIER FAIL] Routing failure detected! Reason: {result.reason}")
            # NOTE: In Phase 4, we will append this exact failure to the SQLite database
            # to feed back into the scikit-learn classifier training loop.

    except Exception as e:
        logger.error(f"[VERIFIER ERROR] Background job execution failed: {str(e)}")

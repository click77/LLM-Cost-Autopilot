import os
import time
import uuid
import joblib
import logging
import numpy as np
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from pydantic import BaseModel, Field

# Import previously built architecture layers
from registry import ModelRegistry
from routing_policy import PolicyEngine
from abstraction import send_request, Response
from dataset import extract_features
from audit_log import init_audit_db, record_request_telemetry
from verifier import run_verification_job

# Setup logging architecture
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AutopilotGateway")

# Container for our machine learning artifact
ML_ARTIFACT: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles secure application startup and shutdown asset allocation hooks."""
    logger.info("Initializing Cost Autopilot system states...")
    init_audit_db() # Verify SQLite audit tables are ready
    
    model_path = "model_v1.joblib"
    if os.path.exists(model_path):
        try:
            global ML_ARTIFACT
            ML_ARTIFACT = joblib.load(model_path)
            logger.info(f"Successfully hot-loaded Routing Classifier Model v{ML_ARTIFACT.get('version_tag', 'Unknown')}")
        except Exception as e:
            logger.error(f"Failed to load Scikit-Learn routing model weights: {str(e)}")
    else:
        logger.warning(f"ML Model checkpoint '{model_path}' not discovered. Falling back to heuristic defaults.")
    
    yield
    logger.info("Cleaning gateway network connections...")

app = FastAPI(
    title="Cost Autopilot Intelligent LLM Router",
    version="1.0.0",
    lifespan=lifespan
)

# ==============================================================================
# OPENAI COMPATIBLE REQUEST / RESPONSE SCHEMAS
# ==============================================================================
class ChatMessage(BaseModel):
    role: str = Field(..., description="The role of the messages author (system, user, assistant).")
    content: str = Field(..., description="The raw content text payload.")

class CompletionRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="The full conversation history list.")
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    model: Optional[str] = Field(None, description="Ignored. Placed for client SDK drop-in compatibility.")

class ChoiceMessage(BaseModel):
    role: str = "assistant"
    content: str

class Choice(BaseModel):
    index: int = 0
    message: ChoiceMessage
    finish_reason: str = "stop"

class UsageMetrics(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class OpenAICompatibleResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str  # The actual model selected by the proxy
    choices: List[Choice]
    usage: UsageMetrics
    
    # --- ENHANCED ENTERPRISE METADATA FIELDS ---
    model_selected: str
    complexity_tier_assigned: int
    cost: float
    confidence_score: float

# ==============================================================================
# ASSISTANT UTILITY METHOD FOR INTENT CLASSIFICATION
# ==============================================================================
def infer_task_type(prompt: str) -> str:
    """Heuristically infers the use case type to feed the verification thresholds engine."""
    prompt_lower = prompt.lower()
    if any(w in prompt_lower for w in ["summariz", "summary", "tl;dr", "outline"]):
        return "summarization"
    if any(w in prompt_lower for w in ["extract", "json", "parse", "schema", "csv"]):
        return "extraction"
    return "classification"

# ==============================================================================
# API ENDPOINT ROUTER ROUTE
# ==============================================================================
@app.post(
    "/v1/chat/completions",
    response_model=OpenAICompatibleResponse,
    status_code=status.HTTP_200_OK
)
@app.post("/v1/completions", response_model=OpenAICompatibleResponse, include_in_schema=False)
async def route_completion_request(request: CompletionRequest, background_tasks: BackgroundTasks):
    """
    Intelligently classifies conversation complexity, routes to the most cost-efficient LLM tier,
    returns standard OpenAI formats instantly, and queues verification jobs in background threads.
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="The 'messages' array cannot be empty.")

    # 1. Reconstruct current prompt text string to run feature evaluation
    user_prompt = " ".join([m.content for m in request.messages if m.role == "user"])
    if not user_prompt:
        user_prompt = request.messages[-1].content # Fallback to last system/assistant frame if no active user block

    # 2. Execute ML routing inference or apply fail-safe heuristic boundaries
    predicted_tier = 1
    confidence_score = 1.0
    
    if ML_ARTIFACT and "model_object" in ML_ARTIFACT:
        try:
            # Re-extract clean operational numerical array signatures matching model training signatures
            features = extract_features(user_prompt, context_provided=True, base_constraints=2, base_format_complexity=1)
            feature_vector = np.array([[
                features["token_count"],
                features["contains_analyze_compare"],
                features["constraint_count"],
                features["context_provided"],
                features["output_format_complexity"]
            ]])
            
            clf = ML_ARTIFACT["model_object"]
            predicted_tier = int(clf.predict(feature_vector)[0])
            
            # Extract probability vector to establish routing selection confidence scores
            probabilities = clf.predict_proba(feature_vector)[0]
            confidence_score = float(np.max(probabilities))
        except Exception as e:
            logger.error(f"[CLASSIFIER ROUTING FAULT] Defaulting to Tier 1: {str(e)}")
    
    # 3. Resolve actual target model metadata definitions using Policy Mapping layout
    registry = ModelRegistry()
    policy = PolicyEngine()
    resolved_route = policy.resolve_route(predicted_tier)
    model_config = registry.get_model(resolved_route.model_id)
    
    logger.info(f"[ROUTER DECISION] Prompt assigned to Tier {predicted_tier} via {model_config.model_id} (Confidence: {confidence_score * 100:.1f}%)")
    
    # 4. Fire the prompt downstream to our provider abstraction network
    try:
        # Pass the formatted prompt through to the resolved model configuration
        # Executed via threadpool internally to prevent event loop bottlenecks
        llm_response: Response = send_request(user_prompt, model_config)
    except Exception as e:
        logger.error(f"[DOWNSTREAM PROVIDER ERROR] Execution broken on primary route: {str(e)}")
        # Production Auto-Escalation Pattern on Retry: If a cheap provider fails catastrophically, 
        # fall back immediately to the Tier 3 golden standard model to safeguard system availability.
        logger.warning(f"Catastrophic failure on cheap route. Escalating directly to Tier 3 fallback...")
        fallback_config = registry.get_model(resolved_route.fallback_model_id)
        llm_response = send_request(user_prompt, fallback_config)
        predicted_tier = 3
        confidence_score = 1.0

    # 5. Pipeline the telemetry payload securely into SQLite tables
    task_type = infer_task_type(user_prompt)
    
    # Record initial transaction immediately to power our live visualization graphs
    record_request_telemetry(
        prompt=user_prompt,
        complexity_tier_assigned=predicted_tier,
        routed_model=llm_response.model_id,
        cost=llm_response.cost,
        latency=llm_response.latency_seconds,
        quality_score=4.5 if predicted_tier == 3 else None, # Set baseline or evaluate later
        was_escalated=(predicted_tier == 3 and resolved_route.model_id != llm_response.model_id)
    )

    # 6. Strict SLA Enforcement: Queue Async Verification check as a background job.
    # This completely detaches verification overhead from the client response chain, guaranteeing 0ms added user latency.
    if predicted_tier < 3:
        tier3_gold_config = registry.get_model("gpt-4o")
        background_tasks.add_task(
            run_verification_job,
            original_prompt=user_prompt,
            cheap_response=llm_response,
            task_type=task_type,
            tier3_config=tier3_gold_config
        )

    # 7. Package and return the standard OpenAI Response JSON structure
    return OpenAICompatibleResponse(
        id=f"chatcmpl-{uuid.uuid4()}",
        created=int(time.time()),
        model=llm_response.model_id,
        choices=[
            Choice(
                index=0,
                message=ChoiceMessage(role="assistant", content=llm_response.output_text),
                finish_reason="stop"
            )
        ],
        usage=UsageMetrics(
            prompt_tokens=llm_response.input_tokens,
            completion_tokens=llm_response.output_tokens,
            total_tokens=llm_response.input_tokens + llm_response.output_tokens
        ),
        model_selected=llm_response.model_id,
        complexity_tier_assigned=predicted_tier,
        cost=llm_response.cost,
        confidence_score=confidence_score
    )

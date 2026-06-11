import os
from fastapi.testclient import TestClient
from main import app

# Ensure local runtime mocks are bypassed gracefully
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "mock-key")

client = TestClient(app)

def test_router_endpoint_execution():
    print("\n=== STARTING ENDPOINT CONTRACT VALIDATION ===")
    
    # Mocking standard conversational request body matching OpenAI client standards
    payload = {
        "messages": [
            {"role": "system", "content": "You are a precise data extractor."},
            {"role": "user", "content": "Extract the company names: 'Google bought YouTube in 2006 for $1.65B.' Return as clean JSON."}
        ],
        "temperature": 0.2
    }
    
    # Fire payload to the proxy gateway
    response = client.post("/v1/chat/completions", json=payload)
    
    assert response.status_code == 200, f"API failed with status code: {response.status_code}"
    
    data = response.json()
    print(f"Server Routing Response Status -> {response.status_code}")
    print(f"Selected Infrastructure Model -> {data['model']}")
    print(f"Assigned Complexity Tier      -> {data['complexity_tier_assigned']}")
    print(f"Proxy Routing Confidence       -> {data['confidence_score'] * 100:.2f}%")
    print(f"Calculated Executed Cost       -> ${data['cost']:.6f}")
    print(f"Output Content Payload         -> {data['choices'][0]['message']['content'][:60]}...")
    
    # Assert structural compliance to verify OpenAI contracts remain unbroken
    assert "choices" in data
    assert "usage" in data
    assert "model_selected" in data
    print("=== API ENDPOINT CONTRACT PASSED ===")

if __name__ == "__main__":
    test_router_endpoint_execution()

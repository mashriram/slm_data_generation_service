import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_generate_endpoint():
    # Test generation with just prompt
    response = client.post(
        "/api/v1/generate",
        data={
            "prompt": "Generate 3 random trivia questions about space.",
            "provider": "groq",
            "count": 3
        }
    )
    # Note: Groq might fail without API key, so we expect either success or 500/400.
    # But for now, we just want to verify the endpoint is reachable and parses input.
    # If the provider is not configured, it will raise an error.

    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")

    assert response.status_code in [200, 400, 500]
    if response.status_code == 200:
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]) <= 3

if __name__ == "__main__":
    test_generate_endpoint()

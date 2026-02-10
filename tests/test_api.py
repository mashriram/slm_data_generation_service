from fastapi.testclient import TestClient
from app.main import app
from unittest.mock import patch, AsyncMock

client = TestClient(app)

def test_models_endpoint():
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "providers" in data
    assert "groq" in data["providers"]

@patch("app.api.endpoints.generation.AgentGenerator")
def test_generate_endpoint_mocked(MockAgentGenerator):
    # Mock instance and generate method
    mock_instance = MockAgentGenerator.return_value
    mock_instance.generate = AsyncMock(return_value=[{"question": "Q", "answer": "A"}])

    response = client.post(
        "/api/v1/generate",
        data={
            "prompt": "Test prompt",
            "provider": "groq",
            "count": 1
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["generated_count"] == 1
    assert data["data"][0]["question"] == "Q"

@patch("app.api.endpoints.generation.AgentGenerator")
def test_generate_with_options(MockAgentGenerator):
    mock_instance = MockAgentGenerator.return_value
    mock_instance.generate = AsyncMock(return_value=[{"q": "1", "a": "1"}])

    response = client.post(
        "/api/v1/generate",
        data={
            "prompt": "Test",
            "provider": "groq",
            "conserve_tokens": True,
            "rate_limit": 10,
            "hf_repo_id": "test/repo",
            "hf_token": "secret"
        }
    )
    assert response.status_code == 200
    # verify mock called with correct args
    args, kwargs = mock_instance.generate.call_args
    assert kwargs["conserve_tokens"] is True
    assert kwargs["rate_limit"] == 10

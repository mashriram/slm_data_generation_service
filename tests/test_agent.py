import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from app.services.agent_generator import AgentGenerator
from fastapi import UploadFile

@pytest.mark.asyncio
async def test_agentic_generation_mocked():
    # Mock LLMProviderFactory to return a mock LLM
    with patch("app.services.agent_generator.LLMProviderFactory") as MockFactory:
        mock_llm = MagicMock()
        MockFactory.return_value.llm = mock_llm

        # Mock create_agent to return a mock agent
        with patch("app.services.agent_generator.create_agent") as MockCreateAgent:
            mock_agent = MockCreateAgent.return_value
            # Mock invoke result. The agent returns a dict with "messages".
            # We mock the last message content.
            mock_message = MagicMock()
            mock_message.content = '{"qa_pairs": [{"question": "Q1", "answer": "A1"}]}'

            mock_agent.ainvoke = AsyncMock(return_value={"messages": [mock_message]})

            generator = AgentGenerator(provider="openai")

            # Run generation in agentic mode
            result = await generator.generate(
                prompt="Test prompt",
                files=[],
                demo_file=None,
                count=1,
                agentic=True
            )

            assert len(result) == 1
            assert result[0]["question"] == "Q1"
            assert MockCreateAgent.called

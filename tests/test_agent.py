import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from app.services.agent_generator import AgentGenerator
from fastapi import UploadFile

@pytest.mark.asyncio
async def test_agentic_generation():
    # Mock LLMProviderFactory to return a mock LLM
    with patch("app.services.agent_generator.LLMProviderFactory") as MockFactory:
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm # Mock bind_tools support
        MockFactory.return_value.llm = mock_llm

        # Mock AgentExecutor.ainvoke to return mocked output
        with patch("app.services.agent_generator.AgentExecutor") as MockExecutor:
            mock_executor_instance = MockExecutor.return_value
            mock_executor_instance.ainvoke = AsyncMock(return_value={"output": '{"qa_pairs": [{"question": "Q1", "answer": "A1"}]}'})

            generator = AgentGenerator(provider="openai")

            # Run generation in agentic mode
            result = await generator.generate(
                prompt="Test prompt",
                files=[], # No files for simplicity
                demo_file=None,
                count=1,
                agentic=True
            )

            print(f"Agentic Result: {result}")
            assert len(result) == 1
            assert result[0]["question"] == "Q1"
            assert MockExecutor.called

if __name__ == "__main__":
    asyncio.run(test_agentic_generation())

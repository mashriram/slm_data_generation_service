from unittest.mock import patch

import pytest

from app.services.generative_modifier import GenerativeDatasetModifier


@pytest.mark.asyncio
async def test_init():
    with patch("app.services.generative_modifier.LLMProviderFactory") as MockFactory:
        modifier = GenerativeDatasetModifier(provider="openai", api_key="key")
        MockFactory.assert_called_with("openai", model_name=None, api_key="key")
        assert modifier.llm == MockFactory.return_value.llm

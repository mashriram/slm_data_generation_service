import pytest
import yaml
from httpx import AsyncClient, ASGITransport
from app.main import app
from unittest.mock import MagicMock, AsyncMock
from datasets import Dataset

@pytest.mark.asyncio
async def test_generate_finetuning_data(mocker):
    # 1. Mock file system and network dependencies
    # Mock httpx to avoid downloading the script
    mock_response = AsyncMock()
    mock_response.text = "class Pipeline: pass"
    mock_response.raise_for_status = MagicMock()
    mock_async_client_instance = AsyncMock()
    mock_async_client_instance.__aenter__.return_value.get.return_value = mock_response
    mocker.patch('httpx.AsyncClient', return_value=mock_async_client_instance)

    # Mock tempfile creation for config
    mocker.patch('app.services.finetuning_generator.Path.write_text')
    mocker.patch('builtins.open', mocker.mock_open())
    mocker.patch('yaml.dump')

    # 2. Mock the dynamic module loading and the Pipeline class
    sample_data = [{"generated_text": "This is a test."}]
    mock_dataset = Dataset.from_list(sample_data)

    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.run.return_value = mock_dataset

    mock_pipeline_class = MagicMock(return_value=mock_pipeline_instance)

    mock_aisheets_module = MagicMock()
    mock_aisheets_module.Pipeline = mock_pipeline_class

    mock_spec = MagicMock()
    mock_spec.loader.exec_module = MagicMock()
    mocker.patch('importlib.util.spec_from_file_location', return_value=mock_spec)
    mocker.patch('importlib.util.module_from_spec', return_value=mock_aisheets_module)

    # 3. Run the test
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/api/v1/generate-finetuning-data/",
            json={"prompt": "test prompt", "model_type": "test_model"},
        )

    # 4. Assertions
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["success"] is True
    assert response_data["data"] == sample_data

    mock_pipeline_class.assert_called_once()
    call_args = mock_pipeline_class.call_args
    assert call_args.kwargs['repo_id'] == "fka/awesome-chatgpt-prompts"
    assert call_args.kwargs['num_rows'] == 5
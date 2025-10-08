import logging
import httpx
import yaml
import tempfile
import importlib.util
from pathlib import Path
import asyncio

from app.schemas.finetuning import FinetuningRequest
from app.utils.exceptions import DataGenerationError

logger = logging.getLogger(__name__)

class FinetuningGenerator:
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.script_url = "https://github.com/huggingface/aisheets/raw/refs/heads/main/scripts/extend_dataset/with_inference_client.py"

    async def _download_script(self) -> Path:
        """Downloads the AI Sheets script to a temporary file."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(self.script_url)
                response.raise_for_status()

            script_path = Path(tempfile.gettempdir()) / "aisheets_script.py"
            script_path.write_text(response.text)

            logger.info(f"Successfully downloaded AI Sheets script to {script_path}")
            return script_path
        except httpx.RequestError as e:
            logger.error(f"Failed to download AI Sheets script: {e}")
            raise DataGenerationError("Could not download the data generation script.")

    def _create_temp_config(self, prompt: str) -> Path:
        """Creates a temporary YAML config file based on the user's prompt."""
        config_data = {
            "columns": {
                "generated_text": {
                    "prompt": f"Based on the following, {prompt}: {{{{act}}}}",
                    "modelProvider": "huggingface",
                    "modelName": self.model_type,
                    "columnsReferences": ["act"]
                }
            }
        }

        try:
            config_path = Path(tempfile.gettempdir()) / "config.yml"
            with open(config_path, "w") as f:
                yaml.dump(config_data, f)
            logger.info(f"Successfully created temporary config file at {config_path}")
            return config_path
        except Exception as e:
            logger.error(f"Failed to create temporary config file: {e}")
            raise DataGenerationError("Could not create the generation configuration.")

    async def generate_data(self, request: FinetuningRequest) -> dict:
        logger.info(f"Generating finetuning data for model {self.model_type} with prompt: '{request.prompt}'")

        try:
            script_path = await self._download_script()
            config_path = self._create_temp_config(request.prompt)

            spec = importlib.util.spec_from_file_location("aisheets_script", script_path)
            aisheets_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(aisheets_module)

            pipeline = aisheets_module.Pipeline(
                repo_id="fka/awesome-chatgpt-prompts",
                split="train",
                config=str(config_path),
                num_rows=5,
                debug=False
            )

            loop = asyncio.get_running_loop()
            dataset = await loop.run_in_executor(None, pipeline.run)

            generated_data = dataset.to_list()

            return {"status": "success", "data": generated_data}

        except Exception as e:
            logger.exception(f"An unexpected error occurred during data generation: {e}")
            raise DataGenerationError(f"An internal error occurred: {e}")
import logging
import asyncio
import json
from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from app.services.llm_provider import LLMProviderFactory, QAList
from app.services.hf_service import HuggingFaceService

logger = logging.getLogger(__name__)

class ColumnData(BaseModel):
    values: List[str]

class NewRows(BaseModel):
    rows: List[Dict[str, Any]]

class GenerativeDatasetModifier:
    def __init__(self, provider: str, model: Optional[str] = None, api_key: Optional[str] = None):
        # We need to update LLMProviderFactory to accept api_key if we want to support it.
        # Currently it reads from settings.
        # Plan Step 3 will update LLMProviderFactory.
        # For now assuming we pass it or it uses env.
        self.llm_factory = LLMProviderFactory(provider, model_name=model)
        if api_key:
            # Inject API key if supported by specific provider implementation or environment
            # This requires LLMProviderFactory update.
            pass

        self.llm = self.llm_factory.llm

    async def generate_column(
        self,
        repo_id: str,
        config_name: str,
        split: str,
        hf_token: str,
        new_column_name: str,
        instruction: str,
        sample_rows: int = 5
    ) -> List[Any]:
        """
        Generates values for a new column for ALL rows in the dataset.
        WARNING: This can be expensive for large datasets.
        """
        try:
            # 1. Load Dataset (we need the data to generate context)
            # Use HuggingFaceService or datasets library directly
            from datasets import load_dataset
            # Log in
            from huggingface_hub import login
            login(token=hf_token, add_to_git_credential=False)

            ds = load_dataset(repo_id, name=config_name if config_name != "default" else None, split=split)
            logger.info(f"Loaded dataset {repo_id} with {len(ds)} rows.")

            # 2. Generate values batch by batch
            generated_values = []

            # Batch processing
            BATCH_SIZE = 10

            # Prompt Template
            prompt = ChatPromptTemplate.from_template(
                """
                You are a data annotation expert.
                Task: Generate a value for a new column '{col_name}' based on the provided row data.
                Description of new column: {instruction}

                Input Rows:
                {rows}

                Return strictly a JSON object with a key 'values' containing a list of {batch_size} strings (one for each row in order).
                """
            )
            parser = JsonOutputParser(pydantic_object=ColumnData)
            chain = prompt | self.llm | parser

            total_rows = len(ds)
            # Iterate
            for i in range(0, total_rows, BATCH_SIZE):
                batch = ds[i : i + BATCH_SIZE]
                # batch is a dict of lists: {'col1': [v1, v2], 'col2': [v1, v2]}
                # We need list of dicts for prompt
                batch_rows = [dict(zip(batch, t)) for t in zip(*batch.values())]

                try:
                    response = await chain.ainvoke({
                        "col_name": new_column_name,
                        "instruction": instruction,
                        "rows": json.dumps(batch_rows, default=str),
                        "batch_size": len(batch_rows)
                    })

                    if response and "values" in response:
                        vals = response["values"]
                        if len(vals) != len(batch_rows):
                            logger.warning(f"Batch {i}: Generated {len(vals)} values for {len(batch_rows)} rows. Padding/Truncating.")
                            # Handle mismatch
                            if len(vals) < len(batch_rows):
                                vals.extend([""] * (len(batch_rows) - len(vals)))
                            else:
                                vals = vals[:len(batch_rows)]
                        generated_values.extend(vals)
                    else:
                        generated_values.extend(["Error"] * len(batch_rows))

                except Exception as e:
                    logger.error(f"Error processing batch {i}: {e}")
                    generated_values.extend(["Error"] * len(batch_rows))

            return generated_values

        except Exception as e:
            logger.error(f"Generate column failed: {e}")
            raise e

    async def generate_rows(
        self,
        repo_id: str,
        config_name: str,
        split: str,
        hf_token: str,
        num_rows: int,
        prompt_instruction: str
    ) -> List[Dict]:
        """
        Generates N new rows matching the schema of the dataset.
        """
        try:
            # 1. Analyze Schema from a few rows
            from datasets import load_dataset
            from huggingface_hub import login
            login(token=hf_token, add_to_git_credential=False)

            ds = load_dataset(repo_id, name=config_name if config_name != "default" else None, split=split, streaming=True)
            # Get 3 examples
            examples = list(ds.take(3))
            schema_keys = list(examples[0].keys()) if examples else []

            # 2. Generate
            # We generate in batches
            BATCH_SIZE = 10 # Generate 10 at a time
            generated_data = []

            prompt = ChatPromptTemplate.from_template(
                """
                You are a synthetic data generator.
                Task: Generate {batch_size} new rows for a dataset with the following schema: {keys}.

                Instructions: {instruction}

                Reference Examples (follow this format):
                {examples}

                Return strictly a JSON object with a key 'rows' containing a list of {batch_size} objects.
                """
            )
            parser = JsonOutputParser(pydantic_object=NewRows)
            chain = prompt | self.llm | parser

            generated_count = 0
            while generated_count < num_rows:
                current_batch = min(BATCH_SIZE, num_rows - generated_count)

                try:
                    response = await chain.ainvoke({
                        "batch_size": current_batch,
                        "keys": str(schema_keys),
                        "instruction": prompt_instruction,
                        "examples": json.dumps(examples, default=str)
                    })

                    if response and "rows" in response:
                        rows = response["rows"]
                        generated_data.extend(rows)
                        generated_count += len(rows)
                    else:
                        logger.warning("Empty response from LLM")

                except Exception as e:
                    logger.error(f"Error generating rows: {e}")
                    break # Stop on error to avoid infinite loop

            return generated_data[:num_rows]

        except Exception as e:
            logger.error(f"Generate rows failed: {e}")
            raise e

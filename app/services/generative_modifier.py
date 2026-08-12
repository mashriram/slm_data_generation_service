import json
import logging
from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from app.services.llm_provider import LLMProviderFactory

logger = logging.getLogger(__name__)


class ColumnData(BaseModel):
    values: List[str]


class NewRows(BaseModel):
    rows: List[Dict[str, Any]]


class GenerativeDatasetModifier:
    def __init__(
        self, provider: str, model: Optional[str] = None, api_key: Optional[str] = None
    ):
        # We need to update LLMProviderFactory to accept api_key if we want to support it.
        # Currently it reads from settings.
        # Plan Step 3 will update LLMProviderFactory.
        # For now assuming we pass it or it uses env.
        self.llm_factory = LLMProviderFactory(
            provider, model_name=model, api_key=api_key
        )
        if api_key:
            # Inject API key if supported by specific provider implementation or environment
            # This requires LLMProviderFactory update.
            pass

        self.llm = self.llm_factory.llm

    def _analyze_statistics(self, ds_sample) -> str:
        """
        Analyzes statistics of the dataset sample to guide generation.
        """
        try:
            import pandas as pd
            df = pd.DataFrame(ds_sample)
            stats_summary = []
            
            for col in df.columns:
                try:
                    stats_summary.append(f"Column: {col}")
                    if pd.api.types.is_numeric_dtype(df[col]):
                        desc = df[col].describe()
                        stats_summary.append(f"  - Type: Numeric")
                        stats_summary.append(f"  - Range: {desc['min']} to {desc['max']}")
                        stats_summary.append(f"  - Mean: {desc['mean']:.2f}")
                    else:
                        stats_summary.append(f"  - Type: Categorical/Text")
                        unique_count = df[col].nunique()
                        stats_summary.append(f"  - Unique Values: {unique_count}")
                        if unique_count < 20:
                            stats_summary.append(f"  - Values: {list(df[col].unique())}")
                        else:
                             stats_summary.append(f"  - Example Values: {list(df[col].head(5))}")
                except Exception:
                    pass
            
            return "\n".join(stats_summary)
        except Exception as e:
            logger.warning(f"Failed to analyze statistics: {e}")
            return "No statistics available."

    async def generate_column(
        self,
        repo_id: str,
        config_name: str,
        split: str,
        hf_token: str,
        new_column_name: str,
        instruction: str,
        sample_rows: int = 5,
        limit: Optional[int] = None,
    ) -> List[Any]:
        """
        Generates values for a new column for ALL rows in the dataset (or top N if limit is set).
        WARNING: This can be expensive for large datasets.
        """
        try:
            # 1. Load Dataset (we need the data to generate context)
            # Use HuggingFaceService or datasets library directly
            from datasets import load_dataset
            from huggingface_hub import login

            login(token=hf_token, add_to_git_credential=False)

            if limit:
                # Use streaming to avoid downloading full dataset
                logger.info(f"Loading dataset with streaming=True and limit={limit}")
                iterable_ds = load_dataset(
                    repo_id,
                    name=config_name if config_name != "default" else None,
                    split=split,
                    streaming=True
                )
                sliced_iterable = iterable_ds.take(limit)
                data = list(sliced_iterable)
                from datasets import Dataset
                ds = Dataset.from_list(data, features=iterable_ds.features)
            else:
                ds = load_dataset(
                    repo_id,
                    name=config_name if config_name != "default" else None,
                    split=split,
                )
            logger.info(f"Loaded dataset {repo_id} with {len(ds)} rows.")

            # Analyze Statistics of existing data to respect distribution
            # Take a sample for stats
            sample_for_stats = ds.select(range(min(len(ds), 100)))
            stats_info = self._analyze_statistics(sample_for_stats)
            logger.info(f"Dataset Statistics:\n{stats_info}")

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
                
                Existing Data Statistics (to maintain consistency/context):
                {stats}

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
                    response = await chain.ainvoke(
                        {
                            "col_name": new_column_name,
                            "instruction": instruction,
                            "stats": stats_info,
                            "rows": json.dumps(batch_rows, default=str),
                            "batch_size": len(batch_rows),
                        }
                    )

                    if response and "values" in response:
                        vals = response["values"]
                        if len(vals) != len(batch_rows):
                            logger.warning(
                                f"Batch {i}: Generated {len(vals)} values for {len(batch_rows)} rows. Padding/Truncating."
                            )
                            # Handle mismatch
                            if len(vals) < len(batch_rows):
                                vals.extend([""] * (len(batch_rows) - len(vals)))
                            else:
                                vals = vals[: len(batch_rows)]
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
        prompt_instruction: str,
    ) -> List[Dict]:
        """
        Generates N new rows matching the schema of the dataset.
        """
        try:
            # 1. Analyze Schema from a few rows
            from datasets import load_dataset
            from huggingface_hub import login

            login(token=hf_token, add_to_git_credential=False)

            ds = load_dataset(
                repo_id,
                name=config_name if config_name != "default" else None,
                split=split,
                streaming=True,
            )
            # Get examples
            # Reduce to 1 example to avoid Token Rate Limits (Groq 413) on large datasets
            examples = list(ds.take(1))
            schema_keys = list(examples[0].keys()) if examples else []
            
            # Analyze statistics from these examples (better if not streaming, but we don't want to download full ds if huge)
            # For simplicity, we analyze statistics of the few examples we have
            stats_info = self._analyze_statistics(examples)

            # Truncate examples for prompt to save tokens
            truncated_examples = []
            for ex in examples:
                trunc_ex = {}
                for k, v in ex.items():
                    s_v = str(v)
                    if len(s_v) > 500:
                        trunc_ex[k] = s_v[:500] + "...(truncated)"
                    else:
                        trunc_ex[k] = v
                truncated_examples.append(trunc_ex)

            # 2. Generate
            # We generate in batches
            BATCH_SIZE = 5 # Reduce batch size to save tokens
            generated_data = []

            prompt = ChatPromptTemplate.from_template(
                """
                You are a synthetic data generator.
                Task: Generate {batch_size} new rows for a dataset with the following schema: {keys}.

                Instructions: {instruction}
                
                Statistical Context of Existing Data:
                {stats}

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
                    response = await chain.ainvoke(
                        {
                            "batch_size": current_batch,
                            "keys": str(schema_keys),
                            "instruction": prompt_instruction,
                            "stats": stats_info,
                            "examples": json.dumps(truncated_examples, default=str),
                        }
                    )

                    if response and "rows" in response:
                        rows = response["rows"]
                        generated_data.extend(rows)
                        generated_count += len(rows)
                    else:
                        logger.warning("Empty response from LLM")

                except Exception as e:
                    logger.error(f"Error generating rows: {e}")
                    break  # Stop on error to avoid infinite loop

            return generated_data[:num_rows]

        except Exception as e:
            logger.error(f"Generate rows failed: {e}")
            raise e

import logging
import pandas as pd
from typing import List, Dict, Optional
from datasets import Dataset, load_dataset, concatenate_datasets
from huggingface_hub import HfApi, login

logger = logging.getLogger(__name__)

class HuggingFaceService:
    @staticmethod
    def push_dataset(
        data: List[Dict],
        repo_id: str,
        token: str,
        private: bool = False,
        append: bool = False
    ) -> str:
        """
        Pushes a list of dictionaries as a dataset to Hugging Face Hub.
        """
        try:
            logger.info(f"Pushing dataset to {repo_id} (Append: {append})")
            login(token=token, add_to_git_credential=False)

            new_dataset = Dataset.from_list(data)

            if append:
                try:
                    existing_dataset = load_dataset(repo_id, split="train")
                    # Align features if necessary or just concatenate
                    combined_dataset = concatenate_datasets([existing_dataset, new_dataset])
                    combined_dataset.push_to_hub(repo_id, private=private)
                except Exception as e:
                    logger.warning(f"Could not load existing dataset {repo_id} to append. Creating new. Error: {e}")
                    new_dataset.push_to_hub(repo_id, private=private)
            else:
                new_dataset.push_to_hub(repo_id, private=private)

            return f"Successfully pushed to https://huggingface.co/datasets/{repo_id}"
        except Exception as e:
            logger.error(f"Failed to push dataset to HF: {e}")
            raise e

    @staticmethod
    def modify_dataset(
        repo_id: str,
        new_data: List[Dict],
        token: str,
        operation: str = "append_rows"
    ) -> str:
        """
        Modifies an existing HF dataset.
        Operations:
        - 'append_rows': Adds new rows.
        - 'add_column': Adds a new column (requires matching length or join key logic - complex, assumes simple index match for now).
        """
        try:
            logger.info(f"Modifying dataset {repo_id} with operation {operation}")
            login(token=token, add_to_git_credential=False)

            # Load existing
            ds = load_dataset(repo_id, split="train")

            if operation == "append_rows":
                new_ds = Dataset.from_list(new_data)
                combined = concatenate_datasets([ds, new_ds])
                combined.push_to_hub(repo_id)

            elif operation == "add_column":
                # Expect new_data to be a list of dicts like [{"new_col_name": val}, ...]
                # Only works if len(new_data) == len(ds)
                if len(new_data) != len(ds):
                    raise ValueError(f"Length mismatch for adding column. Dataset: {len(ds)}, New Data: {len(new_data)}")

                # Convert new_data to dataframe to easily extract columns
                df_new = pd.DataFrame(new_data)
                for col in df_new.columns:
                    ds = ds.add_column(col, df_new[col].tolist())

                ds.push_to_hub(repo_id)

            else:
                raise ValueError(f"Unknown operation: {operation}")

            return f"Successfully modified {repo_id}"
        except Exception as e:
            logger.error(f"Failed to modify dataset: {e}")
            raise e

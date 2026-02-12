import logging
import pandas as pd
from typing import List, Dict, Optional, Union, Any
from datasets import Dataset, load_dataset, concatenate_datasets, get_dataset_config_names, get_dataset_split_names
from huggingface_hub import HfApi, login

logger = logging.getLogger(__name__)

class HuggingFaceService:
    @staticmethod
    def get_dataset_info(repo_id: str, token: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetches available configs and splits for a dataset.
        """
        try:
            if token:
                login(token=token, add_to_git_credential=False)

            configs = get_dataset_config_names(repo_id, token=token)
            # If default config (or no specific config required), it often returns ['default'] or similar,
            # or sometimes just the split names if we call get_dataset_split_names(repo_id).

            # If we have multiple configs, we can't easily get splits without picking one.
            # We return configs and let the UI ask for one.

            info = {
                "configs": configs,
                "splits": {}
            }

            # If there is only one config or 'default', try to fetch splits for it
            if configs:
                default_config = configs[0]
                try:
                    splits = get_dataset_split_names(repo_id, config_name=default_config, token=token)
                    info["splits"][default_config] = splits
                except Exception as e:
                    logger.warning(f"Could not fetch splits for config {default_config}: {e}")

            return info
        except Exception as e:
            # If get_dataset_config_names fails, it might be a dataset without configs (just default)
            # Or authentication error.
            logger.error(f"Failed to get dataset info: {e}")
            try:
                # Try getting splits assuming default config
                splits = get_dataset_split_names(repo_id, token=token)
                return {"configs": ["default"], "splits": {"default": splits}}
            except Exception as e2:
                logger.error(f"Failed to get splits fallback: {e2}")
                raise e

    @staticmethod
    def push_dataset(
        data: List[Dict],
        repo_id: str,
        token: str,
        private: bool = False,
        append: bool = False,
        config_name: Optional[str] = None,
        split: str = "train"
    ) -> str:
        """
        Pushes a list of dictionaries as a dataset to Hugging Face Hub.
        """
        try:
            logger.info(f"Pushing dataset to {repo_id} (Append: {append}, Config: {config_name}, Split: {split})")
            login(token=token, add_to_git_credential=False)

            new_dataset = Dataset.from_list(data)

            if append:
                try:
                    # Load existing with specific config/split
                    existing_dataset = load_dataset(repo_id, name=config_name, split=split)
                    # Align features if necessary or just concatenate
                    combined_dataset = concatenate_datasets([existing_dataset, new_dataset])

                    # Pushing combined dataset to a specific config/split is tricky via push_to_hub
                    # because push_to_hub typically updates the main parquet files.
                    # We usually specify `config_name` in push_to_hub arguments if supported,
                    # but standard usage is `ds.push_to_hub(repo_id, config_name=...)`.

                    push_args = {"repo_id": repo_id, "private": private}
                    if config_name and config_name != "default":
                        push_args["config_name"] = config_name
                    if split:
                        push_args["split"] = split

                    combined_dataset.push_to_hub(**push_args)
                except Exception as e:
                    logger.warning(f"Could not load existing dataset {repo_id} to append. Creating new. Error: {e}")
                    # Create new
                    push_args = {"repo_id": repo_id, "private": private}
                    if config_name and config_name != "default":
                        push_args["config_name"] = config_name
                    if split:
                        push_args["split"] = split
                    new_dataset.push_to_hub(**push_args)
            else:
                push_args = {"repo_id": repo_id, "private": private}
                if config_name and config_name != "default":
                    push_args["config_name"] = config_name
                if split:
                    push_args["split"] = split
                new_dataset.push_to_hub(**push_args)

            return f"Successfully pushed to https://huggingface.co/datasets/{repo_id}"
        except Exception as e:
            logger.error(f"Failed to push dataset to HF: {e}")
            raise e

    @staticmethod
    def modify_dataset(
        repo_id: str,
        token: str,
        new_data: Optional[List[Dict]] = None,
        operation: str = "append_rows",
        config_name: Optional[str] = None,
        split: str = "train",
        new_column_name: Optional[str] = None,
        new_column_data: Optional[List[Any]] = None
    ) -> str:
        """
        Modifies an existing HF dataset.
        Operations:
        - 'append_rows': Adds new rows (requires `new_data`).
        - 'add_column': Adds a new column (requires `new_column_name` and `new_column_data`).
        """
        try:
            logger.info(f"Modifying dataset {repo_id} config={config_name} split={split} op={operation}")
            login(token=token, add_to_git_credential=False)

            # Load existing
            ds = load_dataset(repo_id, name=config_name, split=split)

            if operation == "append_rows":
                if not new_data:
                    raise ValueError("new_data is required for append_rows")

                new_ds = Dataset.from_list(new_data)

                # Verify schema compatibility roughly
                # Concatenate
                combined = concatenate_datasets([ds, new_ds])

                push_args = {"repo_id": repo_id, "config_name": config_name, "split": split}
                # Remove None values
                push_args = {k: v for k, v in push_args.items() if v is not None and v != "default"}

                combined.push_to_hub(**push_args)

            elif operation == "add_column":
                if not new_column_name or new_column_data is None:
                    raise ValueError("new_column_name and new_column_data are required for add_column")

                if len(new_column_data) != len(ds):
                    raise ValueError(f"Length mismatch. Dataset: {len(ds)}, Column Data: {len(new_column_data)}")

                ds = ds.add_column(new_column_name, new_column_data)

                push_args = {"repo_id": repo_id, "config_name": config_name, "split": split}
                push_args = {k: v for k, v in push_args.items() if v is not None and v != "default"}

                ds.push_to_hub(**push_args)

            else:
                raise ValueError(f"Unknown operation: {operation}")

            return f"Successfully modified {repo_id}"
        except Exception as e:
            logger.error(f"Failed to modify dataset: {e}")
            raise e

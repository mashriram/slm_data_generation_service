"""Service module for Gradio UI API interactions."""

import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Configuration from environment variables
API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")


class GradioAPIService:
    """Service for making API calls from Gradio UI."""

    @staticmethod
    def get_models() -> dict:
        """Fetch available models from the API."""
        try:
            response = requests.get(f"{API_URL}/models", timeout=5)
            if response.status_code == 200:
                return response.json()
        except (requests.RequestException, ValueError) as e:
            logger.warning(f"Failed to fetch models: {e}")
        return {
            "providers": ["groq", "openai", "google", "huggingface"],
            "configured": [],
            "defaults": {},
        }

    @staticmethod
    def set_token(provider: str, token: str) -> str:
        """Set an API token for a provider."""
        try:
            response = requests.post(f"{API_URL}/tokens", json={"provider": provider, "token": token}, timeout=10)
            if response.status_code == 200:
                return f"Success: {response.json().get('message')}"
            else:
                return f"Error: {response.text}"
        except Exception as e:
            return f"Failed to set token: {e}"

    @staticmethod
    def delete_token(provider: str) -> str:
        """Delete an API token for a provider."""
        try:
            response = requests.delete(f"{API_URL}/tokens/{provider}", timeout=10)
            if response.status_code == 200:
                return f"Success: {response.json().get('message')}"
            else:
                return f"Error: {response.text}"
        except Exception as e:
            return f"Failed to delete token: {e}"

    @staticmethod
    def get_repo_info(repo_id: str, token: Optional[str]) -> Tuple[dict, dict]:
        """Fetch repository configuration and splits from Hugging Face.

        Returns:
            Tuple of (configs_dict, splits_dict) containing configuration info.
        """
        if not repo_id:
            return {}, {}
        try:
            response = requests.get(
                f"{API_URL}/dataset/info",
                params={"repo_id": repo_id, "token": token},
                timeout=10,
            )
            if response.status_code == 200:
                info = response.json()
                configs = info.get("configs", ["default"])
                splits_map = info.get("splits", {})
                return {
                    "configs": configs,
                    "default_config": configs[0] if configs else None,
                }, {
                    "splits_map": splits_map,
                    "default_splits": splits_map.get(
                        "default", ["train", "test", "validation"]
                    ),
                }
        except requests.RequestException as e:
            logger.error(f"Error fetching repo info: {e}")
        return {}, {}

    @staticmethod
    def update_splits(
        repo_id: str, config_name: Optional[str], token: Optional[str]
    ) -> list:
        """Fetch splits for a specific configuration.

        Args:
            repo_id: Hugging Face repository ID
            config_name: Specific configuration name to get splits for
            token: Optional authentication token

        Returns:
            List of available splits for the given config.
        """
        if not repo_id or not config_name:
            return ["train"]

        try:
            response = requests.get(
                f"{API_URL}/dataset/info",
                params={"repo_id": repo_id, "token": token},
                timeout=10,
            )
            if response.status_code == 200:
                info = response.json()
                splits_map = info.get("splits", {})

                # Get splits for the specific config
                if config_name in splits_map:
                    return splits_map[config_name]
                # Fallback to default config splits
                elif "default" in splits_map:
                    return splits_map["default"]
                # Final fallback
                return ["train", "test", "validation"]
        except requests.RequestException as e:
            logger.error(f"Error fetching splits for config {config_name}: {e}")

        return ["train"]

    @staticmethod
    def generate_data(
        prompt: str,
        files: Optional[List],
        demo_file: Optional[str],
        provider: str,
        model: Optional[str],
        api_key: Optional[str],
        temperature: float,
        count: int,
        agentic: bool,
        use_rag: bool,
        mcp_servers: Optional[str],
        conserve_tokens: bool,
        rate_limit: Optional[int],
        hf_repo: Optional[str],
        hf_token: Optional[str],
        hf_private: bool,
        hf_append: bool,
        hf_config: Optional[str],
        hf_split: Optional[str],
    ) -> Tuple[pd.DataFrame, str, str]:
        """Generate synthetic data and optionally export to Hugging Face.

        Args:
            prompt: Generation instruction prompt
            files: List of file paths to include as context
            demo_file: Path to demo CSV file for few-shot examples
            provider: LLM provider (groq, openai, google, huggingface)
            model: Model name to use
            api_key: API key for the provider
            temperature: Sampling temperature (0.0-1.0)
            count: Number of data pairs to generate
            agentic: Whether to use agentic mode
            use_rag: Whether to use RAG
            mcp_servers: JSON list of MCP servers
            conserve_tokens: Whether to conserve tokens
            rate_limit: Rate limit in requests per minute
            hf_repo: Hugging Face repository ID for export
            hf_token: Hugging Face write token
            hf_private: Whether to make the repo private
            hf_append: Whether to append to existing dataset
            hf_config: Configuration name for HF dataset
            hf_split: Split name for HF dataset

        Returns:
            Tuple of (dataframe, json_output, status_message)
        """
        url = f"{API_URL}/generate"

        # Prepare form data
        data = {
            "prompt": prompt,
            "provider": provider,
            "temperature": float(temperature),
            "count": int(count),
            "agentic": agentic,
            "use_rag": use_rag,
            "mcp_servers": mcp_servers if mcp_servers else None,
            "conserve_tokens": conserve_tokens,
            "rate_limit": int(rate_limit) if rate_limit else 0,
            "hf_repo_id": hf_repo if hf_repo else None,
            "hf_token": hf_token if hf_token else None,
            "hf_private": hf_private,
            "hf_append": hf_append,
            "hf_config": hf_config,
            "hf_split": hf_split,
        }

        if model:
            data["model"] = model
        if api_key:
            data["api_key"] = api_key

        files_payload = []
        file_handles = []  # Keep track of file handles for cleanup

        try:
            if files:
                for f in files:
                    try:
                        file_obj = open(f, "rb")
                        file_handles.append(file_obj)
                        file_name = Path(f).name
                        files_payload.append(
                            ("files", (file_name, file_obj, "application/octet-stream"))
                        )
                    except (IOError, OSError) as e:
                        return (
                            pd.DataFrame(),
                            "{}",
                            f"Error reading file {f}: {e}",
                        )

            if demo_file:
                try:
                    demo_obj = open(demo_file, "rb")
                    file_handles.append(demo_obj)
                    demo_name = Path(demo_file).name
                    files_payload.append(
                        ("demo_file", (demo_name, demo_obj, "text/csv"))
                    )
                except (IOError, OSError) as e:
                    return (
                        pd.DataFrame(),
                        "{}",
                        f"Error reading demo file {demo_file}: {e}",
                    )

            response = requests.post(url, data=data, files=files_payload, timeout=120)

            if response.status_code == 200:
                result = response.json()
                df = pd.DataFrame(result.get("data", []))
                json_output = json.dumps(result.get("data", []), indent=2)
                msg = result.get("message", "Success")
                return df, json_output, msg
            else:
                try:
                    detail = response.json().get("detail", response.text)
                except (ValueError, KeyError):
                    detail = response.text
                return (
                    pd.DataFrame(),
                    "{}",
                    f"Error {response.status_code}: {detail}",
                )

        except requests.RequestException as e:
            return pd.DataFrame(), "{}", f"Request Error: {str(e)}"
        except Exception as e:
            return pd.DataFrame(), "{}", f"Unexpected Error: {str(e)}"
        finally:
            # Ensure all file handles are closed
            for file_obj in file_handles:
                try:
                    file_obj.close()
                except Exception:
                    pass

    @staticmethod
    def modify_hf_dataset(
        repo_id: str,
        token: Optional[str],
        config: Union[str, List[str], None],
        split: Union[str, List[str], None],
        data_json: Optional[str],
        operation: str,
        gen_mode: bool,
        gen_instruction: Optional[str],
        gen_new_col: Optional[str],
        gen_rows: Optional[float],
        provider: str,
        api_key: Optional[str],
    ) -> str:
        """Modify an existing Hugging Face dataset by adding columns or rows.
        Supports multi-select for config and split.
        """
        url = f"{API_URL}/dataset/modify"
        
        # Normalize to lists
        configs = config if isinstance(config, list) else [config]
        splits = split if isinstance(split, list) else [split]
        
        if not configs or configs == [None]:
            configs = ["default"]
        if not splits or splits == [None]:
            splits = ["train"]

        results = []
        
        for cfg in configs:
            curr_config = cfg if cfg else None
            for spl in splits:
                curr_split = spl if spl else "train"
                
                try:
                    payload = {
                        "repo_id": repo_id,
                        "token": token,
                        "operation": operation,
                        "config_name": curr_config,
                        "split": curr_split,
                        "generative_mode": gen_mode,
                        "provider": provider,
                        "api_key": api_key,
                    }

                    # Handle operation-specific fields
                    if operation == "add_column":
                        if not gen_new_col:
                            results.append(f"[{curr_config or 'def'}/{curr_split}] Error: New column name is required")
                            continue
                        payload["new_column_name"] = gen_new_col

                        if gen_mode:
                            if not gen_instruction:
                                results.append(f"[{curr_config or 'def'}/{curr_split}] Error: Instruction is required")
                                continue
                            payload["instruction"] = gen_instruction
                        else:
                            if not data_json:
                                results.append(f"[{curr_config or 'def'}/{curr_split}] Error: Column data is required")
                                continue
                            try:
                                payload["data"] = json.loads(data_json)
                            except json.JSONDecodeError:
                                results.append(f"[{curr_config or 'def'}/{curr_split}] Error: Invalid JSON")
                                continue

                    elif operation == "append_rows":
                        payload["num_rows"] = int(gen_rows) if gen_rows else 10

                        if gen_mode:
                            if not gen_instruction:
                                results.append(f"[{curr_config or 'def'}/{curr_split}] Error: Instruction is required")
                                continue
                            payload["instruction"] = gen_instruction
                        else:
                            if not data_json:
                                results.append(f"[{curr_config or 'def'}/{curr_split}] Error: Data is required")
                                continue
                            try:
                                payload["new_data"] = json.loads(data_json)
                            except json.JSONDecodeError:
                                results.append(f"[{curr_config or 'def'}/{curr_split}] Error: Invalid JSON")
                                continue

                    response = requests.post(url, json=payload, timeout=120)
                    
                    status_prefix = f"[{curr_config or 'def'}/{curr_split}]"
                    if response.status_code == 200:
                        results.append(f"{status_prefix} Success: {response.json().get('message')}")
                    else:
                        try:
                            detail = response.json().get("detail", response.text)
                        except:
                            detail = response.text
                        results.append(f"{status_prefix} Error: {detail}")
                        
                except requests.RequestException as e:
                    results.append(f"[{curr_config or 'def'}/{curr_split}] Request Error: {str(e)}")
                except Exception as e:
                     results.append(f"[{curr_config or 'def'}/{curr_split}] Error: {str(e)}")

        return "\n".join(results)

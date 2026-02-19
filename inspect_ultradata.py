import sys
import os
from huggingface_hub import HfApi, login
from datasets import get_dataset_config_names, get_dataset_split_names

# Add current directory to path
sys.path.append(os.getcwd())

HF_TOKEN = "TO_CHANGE before testing api_key`"
REPO_ID = "openbmb/UltraData-Math"

def inspect():
    print(f"Inspecting {REPO_ID} with provided token...")
    try:
        login(token=HF_TOKEN, add_to_git_credential=False)
        
        configs = get_dataset_config_names(REPO_ID, token=HF_TOKEN)
        print(f"Configs found ({len(configs)}): {configs}")
        
        for config in configs[:3]: # Check first 3
            try:
                splits = get_dataset_split_names(REPO_ID, config_name=config, token=HF_TOKEN)
                print(f"  Config '{config}' splits: {splits}")
            except Exception as e:
                print(f"  Failed to get splits for '{config}': {e}")
                
    except Exception as e:
        print(f"Inspection failed: {e}")

if __name__ == "__main__":
    inspect()

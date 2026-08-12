import sys
import os
import time

# Add current directory to path
sys.path.append(os.getcwd())

from app.services.gradio_service import GradioAPIService

HF_TOKEN = "TO_CHANGE before testing api_key`" # User provided token
# We'll use a public repo to test *reading* first, and then *simulated* writing if possible.
# But modification requires write access.
# We will use "openbmb/UltraData-Math" as the target repo, but we expect it to FAIL on push (403 Forbidden)
# This is enough to verify that the service *attempts* to push to the correct configs/splits.
# We will intercept the log or check the error message.

REPO_ID = "openbmb/UltraData-Math"

def test_multiselect():
    print("--- Testing Multi-select logic in GradioAPIService ---")
    
    # Select 2 configs and 1 split, or 1 config and 2 splits
    # Let's try "default" config (or None) and two splits ["train", "test"] if they exist?
    # Inspect script showed: 'UltraData-Math-L3-Multi-Style-Synthetic' has 'train'
    # 'UltraData-Math-L3-QA-Synthetic' has 'train'
    
    configs = ["UltraData-Math-L3-Multi-Style-Synthetic", "UltraData-Math-L3-QA-Synthetic"]
    splits = ["train"] # Both have train
    
    print(f"Target Configs: {configs}")
    print(f"Target Splits: {splits}")
    
    # We'll try to "Validation-Only" (dry run) if possible? No, the service makes real requests.
    # We expect 403 Forbidden because we don't own the repo.
    # But getting 403 for *each* config proves the loop works!
    
    print("\nAttempting 'add_column' operation (Targeting protected repo -> Expect Clone)...")
    
    # Manual data add
    data = '[{"val": 1}, {"val": 2}, {"val": 3}]' # Dummy
    
    result = GradioAPIService.modify_hf_dataset(
        repo_id=REPO_ID,
        token=HF_TOKEN,
        config=configs,
        split=splits,
        data_json=data,
        operation="add_column",
        gen_mode=False,
        gen_instruction=None,
        gen_new_col="test_col",
        gen_rows=None,
        provider="groq",
        api_key=None
    )
    
    print("\n--- Result ---")
    print(result)
    
    # Check if we see 2 attempts and cloning messages
    # Expected message might contain "Cloned to" or "Permission denied on original"
    
    if "UltraData-Math-L3-Multi-Style-Synthetic" in result and "UltraData-Math-L3-QA-Synthetic" in result:
        if "Cloned to" in result or "Permission denied" in result:
             print("\nSUCCESS: Loop Logic Verified AND Cloning behavior triggered.")
        else:
             print("\nPARTIAL SUCCESS: Loop verified but Cloning message not found (maybe it succeeded? Unlikely).")
    else:
        print("\nFAILURE: Did not see both configs in output.")

if __name__ == "__main__":
    test_multiselect()

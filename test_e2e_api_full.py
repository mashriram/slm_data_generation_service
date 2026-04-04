import requests
import json
import time
import sys

# Configuration
API_URL = "http://localhost:8000/api/v1"
REPO_ID = "openbmb/UltraData-Math"
# Use a specific config to be precise
CONFIG = "UltraData-Math-L3-Multi-Style-Synthetic"
SPLIT = "train"
HF_TOKEN = "TO_CHANGE before testing api_key`" # User provided token
GROQ_API_KEY = "TO_CHANGE before testing api_key" # User provided token

def print_result(step_name, response):
    print(f"\n--- {step_name} ---")
    if response.status_code == 200:
        data = response.json()
        print("SUCCESS")
        print(json.dumps(data, indent=2))
        return data
    else:
        print(f"FAILED (Status {response.status_code})")
        print(response.text)
        return None

def test_add_column():
    print(f"\n[1] Testing ADD COLUMN (Generative) on {REPO_ID}")
    print("    Expected behavior: Clone repo (if not exists) -> Load 10 rows -> Generate Column -> Push")
    
    payload = {
        "repo_id": REPO_ID,
        "token": HF_TOKEN,
        "operation": "add_column",
        "config_name": CONFIG,
        "split": SPLIT,
        "generative_mode": True,
        "provider": "groq",
        "api_key": GROQ_API_KEY,
        "new_column_name": "complexity_rating",
        "instruction": "Rate the complexity of this math problem on a scale of 1-10. Return only the number.",
        "num_rows": 10 # LIMIT functionality we just added
    }
    
    start_time = time.time()
    try:
        response = requests.post(f"{API_URL}/dataset/modify", json=payload, timeout=300)
    except Exception as e:
        print(f"Request failed: {e}")
        return None
        
    print(f"    Time taken: {time.time() - start_time:.2f}s")
    return print_result("Add Column response", response)

def test_append_rows():
    print(f"\n[2] Testing APPEND ROWS (Generative) on {REPO_ID}")
    print("    Expected behavior: Clone repo (if not already done) -> Load -> Generate 3 rows -> Push")
    
    # We target the SAME repo/config. 
    # If the previous step cloned it, this step might hit the CLONED repo if we updated the logic to return the new repo ID?
    # The API returns a message saying "Cloned to X". 
    # But for this test, we are passing the ORIGINAL repo ID again.
    # The service logic I wrote:
    # "Attempting to clone... Check if target repo exists... IF EXISTS, USE IT."
    # So passing the original repo ID again SHOULD works! It will fail permission, catch 403, check clone, see it exists, and use it.
    
    payload = {
        "repo_id": REPO_ID,
        "token": HF_TOKEN,
        "operation": "append_rows",
        "config_name": CONFIG,
        "split": SPLIT,
        "generative_mode": True,
        "provider": "groq",
        "api_key": GROQ_API_KEY,
        "instruction": "Generate a math problem involving calculus.",
        "num_rows": 3 # Generate 3 rows
    }
    
    start_time = time.time()
    try:
        response = requests.post(f"{API_URL}/dataset/modify", json=payload, timeout=300)
    except Exception as e:
        print(f"Request failed: {e}")
        return None
        
    print(f"    Time taken: {time.time() - start_time:.2f}s")
    return print_result("Append Rows response", response)

def main():
    print("=== STARTING END-TO-END API TEST ===")
    
    # 1. Add Column
    res1 = test_add_column()
    if not res1:
        print("\nCRITICAL: Add Column failed. Aborting.")
        return

    # 2. Append Rows
    res2 = test_append_rows()
    
    print("\n=== TEST COMPLETE ===")
    if res1 and res2:
        print("VERDICT: PASS")
    else:
        print("VERDICT: PARTIAL FAIL")

if __name__ == "__main__":
    main()

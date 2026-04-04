import asyncio
import sys
import os
import json
from pathlib import Path

# Add current directory to path
sys.path.append(os.getcwd())

from app.core.token_manager import TokenManager
from app.services.agent_generator import AgentGenerator
from app.core.config import get_settings

# Get key from args or hardcode for this test session (user provided it)
GROQ_API_KEY = "TO_CHANGE before testing api_key"

async def main():
    print("--- Starting End-to-End Generation Test ---")
    
    # 1. Setup Token
    print(f"1. Configuring Token for Groq...")
    tm = TokenManager()
    tm.set_token("groq", GROQ_API_KEY)
    
    # Verify token was set
    saved_token = tm.get_token("groq")
    if saved_token == GROQ_API_KEY:
        print("   Token successfully stored in TokenManager.")
    else:
        print(f"   WARNING: Stored token mismatch or failed. Got: {saved_token[:5]}...")

    # 2. Initialize Generator
    print("\n2. Initializing AgentGenerator with Groq...")
    try:
        # We don't pass api_key explicitly to force it to use TokenManager (which LLMProviderFactory uses)
        # Note: LLMProviderFactory loads from TokenManager if api_key is None.
        generator = AgentGenerator(provider="groq", model="llama-3.3-70b-versatile")
        print("   Generator initialized.")
    except Exception as e:
        print(f"   FATAL: Failed to initialize generator: {e}")
        return

    # 3. Generate Data
    print("\n3. Generating Data (Prompt: 'Generate 3 Python coding interview questions')...")
    instruction = "Generate 3 Python coding interview questions with answers."
    try:
        # Generate 3 pairs
        results = await generator.generate(
            prompt=instruction,
            files=[],
            demo_file=None,
            count=3,
            agentic=False
        )
        
        print(f"\n   Generation Complete. Received {len(results)} items.")
        print(json.dumps(results, indent=2))
        
        if len(results) > 0:
            print("\n   SUCCESS: Data generated successfully.")
        else:
            print("\n   FAILURE: No data generated.")
            
    except Exception as e:
        print(f"   FATAL: Generation failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())

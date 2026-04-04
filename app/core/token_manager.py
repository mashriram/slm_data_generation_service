import sqlite3
import logging
from typing import Dict, List, Optional
import os

logger = logging.getLogger(__name__)

DB_PATH = "tokens.db"

class TokenManager:
    """
    Manages API tokens for different LLM providers using a local SQLite database.
    """
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Initialize the database table if it doesn't exist."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS api_tokens (
                        provider TEXT PRIMARY KEY,
                        token TEXT NOT NULL
                    )
                """)
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to initialize token database: {e}")

    def set_token(self, provider: str, token: str) -> None:
        """Save or update a token for a provider."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO api_tokens (provider, token)
                    VALUES (?, ?)
                """, (provider.lower(), token))
                conn.commit()
            logger.info(f"Token set for provider: {provider}")
        except Exception as e:
            logger.error(f"Failed to set token for {provider}: {e}")
            raise e

    def get_token(self, provider: str) -> Optional[str]:
        """Retrieve a token for a provider. Checks DB first, then Environment."""
        # 1. Check Database
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT token FROM api_tokens WHERE provider = ?", (provider.lower(),))
                row = cursor.fetchone()
                if row:
                    return row[0]
        except Exception as e:
            logger.warning(f"Failed to read token from DB for {provider}: {e}")

        # 2. Fallback to Environment Variables
        env_var_map = {
            "openai": "OPENAI_API_KEY",
            "groq": "GROQ_API_KEY",
            "google": "GOOGLE_API_KEY",
            "huggingface": "HF_TOKEN"
        }
        env_var = env_var_map.get(provider.lower())
        if env_var:
           return os.getenv(env_var)
        
        return None

    def list_providers(self) -> List[str]:
        """List providers that have configuration (either in DB or Env)."""
        providers = set()
        
        # From DB
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT provider FROM api_tokens")
                rows = cursor.fetchall()
                for row in rows:
                    providers.add(row[0])
        except Exception:
            pass

        # From Env
        env_map = {
            "openai": "OPENAI_API_KEY",
            "groq": "GROQ_API_KEY",
            "google": "GOOGLE_API_KEY",
            "huggingface": "HF_TOKEN"
        }
        for prov, var in env_map.items():
            if os.getenv(var):
                providers.add(prov)

        return list(providers)

    def delete_token(self, provider: str) -> bool:
        """Remove a token from the database."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM api_tokens WHERE provider = ?", (provider.lower(),))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to delete token for {provider}: {e}")
            return False

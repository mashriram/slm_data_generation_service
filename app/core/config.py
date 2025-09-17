# app/core/config.py
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """

    APP_NAME: str = "SLM Data Generation Service"
    APP_VERSION: str = "1.0.0"

    # Supported LLM Providers
    LLMProviderEnum = Literal["groq", "openai", "google", "huggingface"]

    # API Keys - these should be set in the .env file
    GROQ_API_KEY: str | None = None
    OPENAI_API_KEY: str | None = None
    GOOGLE_API_KEY: str | None = None
    HUGGINGFACE_API_TOKEN: str | None = None

    # Default model names for each provider
    GROQ_MODEL_NAME: str = "llama3-8b-8192"
    OPENAI_MODEL_NAME: str = "gpt-4o-mini"
    GOOGLE_MODEL_NAME: str = "gemini-1.5-flash-latest"
    HUGGINGFACE_MODEL_NAME: str = "google/flan-t5-large"

    # Generation settings
    TARGET_QA_PAIRS: int = 10000
    QA_BATCH_SIZE: int = 20  # Number of QAs to generate in one LLM call

    # Text processing settings
    CHUNK_SIZE: int = 4000
    CHUNK_OVERLAP: int = 200

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Returns a cached instance of the Settings object."""
    return Settings()

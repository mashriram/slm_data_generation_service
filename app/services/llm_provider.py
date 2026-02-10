import logging

from langchain_core.language_models.base import BaseLanguageModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFacePipeline
from langchain_openai import ChatOpenAI

from app.core.config import Settings, get_settings
from app.utils.exceptions import LLMProviderError

logger = logging.getLogger(__name__)


class LLMProviderFactory:
    """Factory to get a configured LangChain LLM instance."""

    def __init__(self, provider: Settings.LLMProviderEnum):
        self.settings = get_settings()
        self.provider = provider
        logger.info(f"Initializing LLM provider: {self.provider}")

    def get_llm(self) -> BaseLanguageModel:
        """Selects and configures the appropriate LLM based on the provider."""
        try:
            if self.provider == "groq":
                if not self.settings.GROQ_API_KEY:
                    raise LLMProviderError(
                        "GROQ_API_KEY is not set in the environment."
                    )
                return ChatGroq(
                    temperature=0.7,
                    groq_api_key=self.settings.GROQ_API_KEY,
                    model_name=self.settings.GROQ_MODEL_NAME,
                )

            elif self.provider == "openai":
                if not self.settings.OPENAI_API_KEY:
                    raise LLMProviderError(
                        "OPENAI_API_KEY is not set in the environment."
                    )
                return ChatOpenAI(
                    temperature=0.7,
                    openai_api_key=self.settings.OPENAI_API_KEY,
                    model=self.settings.OPENAI_MODEL_NAME,
                )

            elif self.provider == "google":
                if not self.settings.GOOGLE_API_KEY:
                    raise LLMProviderError(
                        "GOOGLE_API_KEY is not set in the environment."
                    )
                return ChatGoogleGenerativeAI(
                    temperature=0.7,
                    google_api_key=self.settings.GOOGLE_API_KEY,
                    model=self.settings.GOOGLE_MODEL_NAME,
                    convert_system_message_to_human=True,
                )

            elif self.provider == "huggingface":
                return HuggingFacePipeline.from_model_id(
                    model_id=self.settings.HUGGINGFACE_MODEL_NAME,
                    task="text2text-generation",
                    pipeline_kwargs={"max_new_tokens": 512},
                )

            else:
                # This case should ideally not be reached due to Pydantic validation
                raise LLMProviderError(f"Unsupported LLM provider: {self.provider}")

        except ImportError as e:
            raise LLMProviderError(
                f"Missing dependencies for {self.provider}. Please ensure all required packages are installed. Details: {e}"
            )
        except Exception as e:
            raise LLMProviderError(
                f"Failed to initialize LLM provider '{self.provider}': {e}"
            )

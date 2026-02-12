import logging
from typing import List, Optional

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFacePipeline
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr

from app.core.config import LLMProviderEnum, get_settings
from app.utils.exceptions import LLMProviderError

logger = logging.getLogger(__name__)


# Pydantic model for structured output from the LLM
class QAPair(BaseModel):
    question: str = Field(description="A question about the provided context.")
    answer: str = Field(
        description="The answer to the question, derived strictly from the context."
    )


class QAList(BaseModel):
    qa_pairs: List[QAPair]


class LLMProviderFactory:
    """Factory to create and configure LLM chains."""

    def __init__(
        self,
        provider: LLMProviderEnum,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
    ):
        self.settings = get_settings()
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = api_key
        self.parser = JsonOutputParser(pydantic_object=QAList)
        # Note: self.chain is primarily for backward compatibility or simple use cases.
        # AgentGenerator might use self.llm directly.
        self.prompt = self._create_prompt()
        self.llm = self._get_llm()
        self.chain = self.prompt | self.llm | self.parser
        logger.info(
            f"Initialized LLM provider: {self.provider} (Model: {self.model_name or 'default'})"
        )

    def _create_prompt(self):
        """Creates the prompt template for QA generation."""
        return ChatPromptTemplate.from_template(
            """
            You are an expert at creating high-quality question-answer pairs for training a language model.
            Based *only* on the provided context below, generate exactly {num_questions} diverse and insightful question-answer pairs.
            The questions should cover a range of topics from the text. The answers must be accurate and concise, directly extracted or synthesized from the information in the context.
            Do not use any external knowledge.
            
            Return the pairs in a JSON object with a single key 'qa_pairs' containing a list of objects, where each object has a 'question' and 'answer' key.
            
            Context:
            ---
            {context}
            ---
            
            Format Instructions:
            {format_instructions}
            """
        )

    def _get_llm(self):
        """Selects and configures the appropriate LLM based on the provider."""
        try:
            if self.provider == "groq":
                key = self.api_key or self.settings.GROQ_API_KEY
                if not key:
                    raise LLMProviderError("GROQ_API_KEY is not set.")
                return ChatGroq(
                    temperature=self.temperature,
                    groq_api_key=key,
                    model_name=self.model_name or self.settings.GROQ_MODEL_NAME,
                )

            elif self.provider == "openai":
                key = self.api_key or self.settings.OPENAI_API_KEY
                if not key:
                    raise LLMProviderError("OPENAI_API_KEY is not set.")
                return ChatOpenAI(
                    temperature=self.temperature,
                    openai_api_key=key,
                    model=self.model_name or self.settings.OPENAI_MODEL_NAME,
                )

            elif self.provider == "google":
                key = self.api_key or self.settings.GOOGLE_API_KEY
                if not key:
                    raise LLMProviderError("GOOGLE_API_KEY is not set.")
                return ChatGoogleGenerativeAI(
                    temperature=self.temperature,
                    api_key=SecretStr(self.settings.GOOGLE_API_KEY),
                    model=self.model_name or self.settings.GOOGLE_MODEL_NAME,
                )

            elif self.provider == "huggingface":
                # Note: This runs the model locally. Requires `transformers` and `torch`.
                # API Token might be used for login if needed, but Pipeline usually loads locally.
                return HuggingFacePipeline.from_model_id(
                    model_id=self.model_name or self.settings.HUGGINGFACE_MODEL_NAME,
                    task="text2text-generation",
                    pipeline_kwargs={"max_new_tokens": 512},
                )

            else:
                raise LLMProviderError(f"Unsupported LLM provider: {self.provider}")
        except ImportError as e:
            raise LLMProviderError(
                f"Missing dependencies for {self.provider}. Please install them. Details: {e}"
            )
        except Exception as e:
            raise LLMProviderError(
                f"Failed to initialize LLM provider '{self.provider}': {e}"
            )

    async def generate_qa(self, context: str, num_questions: int) -> dict:
        """Invokes the chain to generate QA pairs."""
        try:
            return await self.chain.ainvoke(
                {
                    "context": context,
                    "num_questions": num_questions,
                    "format_instructions": self.parser.get_format_instructions(),
                }
            )
        except Exception as e:
            logger.error(f"Error during LLM invocation with {self.provider}: {e}")
            raise LLMProviderError(
                f"Failed to generate QA pairs using {self.provider}."
            )

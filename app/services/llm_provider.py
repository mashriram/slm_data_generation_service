import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFacePipeline
from typing import List, Type

from app.core.config import get_settings, LLMProviderEnum
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

    def __init__(self, provider: LLMProviderEnum):
        self.settings = get_settings()
        self.provider = provider
        self.parser = JsonOutputParser(pydantic_object=QAList)
        self.prompt = self._create_prompt()
        self.llm = self._get_llm()
        self.chain = self.prompt | self.llm | self.parser
        logger.info(f"Initialized LLM provider: {self.provider}")

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
                if not self.settings.GROQ_API_KEY:
                    raise LLMProviderError("GROQ_API_KEY is not set.")
                return ChatGroq(
                    temperature=0.7,
                    groq_api_key=self.settings.GROQ_API_KEY,
                    model_name=self.settings.GROQ_MODEL_NAME,
                )

            elif self.provider == "openai":
                if not self.settings.OPENAI_API_KEY:
                    raise LLMProviderError("OPENAI_API_KEY is not set.")
                return ChatOpenAI(
                    temperature=0.7,
                    openai_api_key=self.settings.OPENAI_API_KEY,
                    model=self.settings.OPENAI_MODEL_NAME,
                )

            elif self.provider == "google":
                if not self.settings.GOOGLE_API_KEY:
                    raise LLMProviderError("GOOGLE_API_KEY is not set.")
                return ChatGoogleGenerativeAI(
                    temperature=0.7,
                    google_api_key=self.settings.GOOGLE_API_KEY,
                    model=self.settings.GOOGLE_MODEL_NAME,
                )

            elif self.provider == "huggingface":
                # Note: This runs the model locally. Requires `transformers` and `torch`.
                return HuggingFacePipeline.from_model_id(
                    model_id=self.settings.HUGGINGFACE_MODEL_NAME,
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

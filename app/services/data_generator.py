import asyncio
import logging
import random

from fastapi import UploadFile
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings
from app.schemas.qa import QAItem
from app.services.llm_provider import LLMProviderFactory
from app.services.text_extractor import TextExtractor
from app.utils.exceptions import DataGenerationError, FileExtractionError

logger = logging.getLogger(__name__)


class QAGenerator:
    """Orchestrates the generation of question-answer pairs from a document."""

    def __init__(self, provider: str):
        self.settings = get_settings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.CHUNK_SIZE,
            chunk_overlap=self.settings.CHUNK_OVERLAP,
        )
        self.llm_provider = LLMProviderFactory(provider).llm
        self.llm_chain = LLMProviderFactory(provider).chain

    async def generate_from_file(self, file: UploadFile) -> List[QAItem]:
        """
        Main method to process a file and generate a QA dataset.
        """
        logger.info(
            f"Starting QA generation for '{file.filename}' with target of {self.settings.TARGET_QA_PAIRS} pairs."
        )

        try:
            # 1. Extract text from document
            full_text = await TextExtractor.extract(file)

            # 2. Split text into manageable chunks
            text_chunks = self.text_splitter.split_text(full_text)
            if not text_chunks:
                raise DataGenerationError(
                    "Text could not be split into chunks. The document might be too short."
                )
            logger.info(f"Document split into {len(text_chunks)} chunks.")

            # 3. Generate QA pairs in parallel
            all_qa_pairs = []
            num_chunks = len(text_chunks)
            # Calculate how many questions to generate per chunk
            q_per_chunk = max(1, self.settings.QA_BATCH_SIZE)
            total_needed = self.settings.TARGET_QA_PAIRS

            # Create concurrent generation tasks
            tasks = []
            generated_count = 0
            while generated_count < total_needed:
                # Randomly sample a chunk to promote diversity
                chunk = random.choice(text_chunks)
                task = self._generate_for_chunk(chunk, q_per_chunk)
                tasks.append(task)
                generated_count += q_per_chunk

            # Execute tasks concurrently
            results = await asyncio.gather(*tasks)

            for result in results:
                if result and result.get("qa_pairs"):
                    all_qa_pairs.extend(result["qa_pairs"])

            if not all_qa_pairs:
                raise DataGenerationError(
                    "LLM failed to generate any QA pairs from the document."
                )

            logger.info(f"Successfully generated {len(all_qa_pairs)} QA pairs.")
            # Trim to the target number
            return [
                QAItem(**pair) for pair in all_qa_pairs[: self.settings.TARGET_QA_PAIRS]
            ]

        except (FileExtractionError, DataGenerationError) as e:
            # Pass these specific errors up
            raise e
        except Exception as e:
            logger.exception(f"An unexpected error occurred during QA generation: {e}")
            raise DataGenerationError(f"An internal error stopped QA generation: {e}")

    async def _generate_for_chunk(self, chunk: str, num_questions: int) -> dict:
        """Invokes the LLM chain for a single text chunk."""
        try:
            response = await self.llm_chain.ainvoke(
                {
                    "context": chunk,
                    "num_questions": num_questions,
                    "format_instructions": LLMProviderFactory.parser.get_format_instructions(),
                }
            )
            return response
        except Exception as e:
            logger.error(f"Failed to generate QA for a chunk: {e}")
            return None  # Return None on failure to avoid crashing the whole process

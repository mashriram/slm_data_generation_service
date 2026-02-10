import logging
from typing import List

from datasets import Dataset, Features, Sequence, Value
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings
from app.schemas.qa import QAItem
from app.services.llm_provider import LLMProviderFactory
from app.utils.exceptions import DataGenerationError

logger = logging.getLogger(__name__)


class AISheetsGenerator:
    """
    Orchestrates QA generation using the official datasets.generate() method.
    """

    def __init__(self, provider: str):
        self.settings = get_settings()
        self.llm = LLMProviderFactory(provider).get_llm()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.CHUNK_SIZE,
            chunk_overlap=self.settings.CHUNK_OVERLAP,
        )

    async def generate(self, full_text: str) -> List[QAItem]:
        logger.info("Starting QA generation with datasets.generate().")

        # 1. Split text into chunks
        text_chunks = self.text_splitter.split_text(full_text)
        if not text_chunks:
            raise DataGenerationError(
                "Text could not be split into chunks. The document is likely too short."
            )
        logger.info(f"Document split into {len(text_chunks)} chunks.")

        # 2. Create a Hugging Face Dataset from the chunks
        initial_dataset = Dataset.from_list(
            [{"text_chunk": chunk} for chunk in text_chunks]
        )

        # 3. Define the features (the structure of the output) for the generated data
        qa_features = Features(
            {
                "question": Value("string"),
                "answer": Value("string"),
            }
        )

        # Calculate how many chunks to process to reach the target number of QA pairs
        num_chunks_to_process = (
            self.settings.TARGET_QA_PAIRS // self.settings.QA_PER_CHUNK
        ) + 1

        # Select a subset of chunks to process to avoid unnecessary API calls and costs
        processing_dataset = initial_dataset.select(
            range(min(len(initial_dataset), num_chunks_to_process))
        )
        logger.info(
            f"Processing {len(processing_dataset)} chunks to meet target of {self.settings.TARGET_QA_PAIRS} pairs."
        )

        try:
            # 4. Use dataset.generate() to create the QA pairs. This is the core logic.
            # It will apply the prompt and LLM to each row in `processing_dataset`.
            generated_dataset = processing_dataset.generate(
                "question_answer_pairs",  # Name of the new column to create
                Sequence(
                    feature=qa_features
                ),  # The expected output structure is a list of QA pairs
                llm=self.llm,
                prompt_template=(
                    "You are an expert at creating high-quality question-answer pairs for training a language model.\n"
                    "Based **only** on the provided context below, generate {num_examples} diverse and insightful question-answer pairs.\n"
                    "The questions should cover various aspects of the text. The answers must be accurate and concise, directly derived from the information in the context.\n"
                    "Do not use any external knowledge. Do not number the pairs.\n\n"
                    "## Context:\n"
                    "------------\n"
                    "{text_chunk}\n"
                    "------------\n"
                ),
                num_examples_per_prompt=self.settings.QA_PER_CHUNK,
            )
            logger.info("datasets.generate() process completed successfully.")

            # 5. Flatten the results from the nested structure into a simple list
            all_qa_pairs = []
            for row in generated_dataset["question_answer_pairs"]:
                for qa_pair in row:
                    all_qa_pairs.append(QAItem(**qa_pair))

            if not all_qa_pairs:
                raise DataGenerationError(
                    "The LLM failed to generate any QA pairs from the provided document."
                )

            # 6. Return the list, trimmed to the exact target number
            return all_qa_pairs[: self.settings.TARGET_QA_PAIRS]

        except Exception as e:
            logger.exception(f"An error occurred during datasets.generate(): {e}")
            raise DataGenerationError(
                f"The generation process failed. This can be due to an LLM provider error (e.g., bad API key, rate limit) or an issue with the document content. Details: {e}"
            )

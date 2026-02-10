import logging

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status

from app.core.config import Settings
from app.schemas.qa import QAGenerationResponse
from app.services.aisheets_generator import AISheetsGenerator
from app.services.text_extractor import TextExtractor
from app.utils.exceptions import (
    DataGenerationError,
    FileExtractionError,
    LLMProviderError,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/generate-qa/",
    response_model=QAGenerationResponse,
    summary="Generate Question-Answer Pairs from a Document",
    tags=["QA Generation"],
)
async def generate_qa(
    llm_provider: Settings.LLMProviderEnum = Query(
        "groq", description="The LLM provider to use for generation."
    ),
    file: UploadFile = File(
        ..., description="The document file (PDF or DOCX) to process."
    ),
):
    """
    This endpoint extracts text from a document and uses the `datasets-generation`
    library to generate a high-quality dataset of question-answer pairs.
    """
    logger.info(
        f"Received request to generate QA from '{file.filename}' using '{llm_provider}'."
    )
    try:
        # 1. Extract text
        full_text = await TextExtractor.extract(file)

        # 2. Initialize generator
        generator = AISheetsGenerator(provider=llm_provider)

        # 3. Generate dataset
        qa_pairs = await generator.generate(full_text)

        logger.info(
            f"Successfully generated {len(qa_pairs)} QA pairs for '{file.filename}'."
        )
        return QAGenerationResponse(generated_pairs=len(qa_pairs), data=qa_pairs)
    except (FileExtractionError, DataGenerationError, LLMProviderError) as e:
        logger.error(f"Generation failed for '{file.filename}': {e.detail}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.detail)
    except Exception as e:
        logger.exception(
            f"An unexpected internal error occurred while processing '{file.filename}': {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal server error occurred.",
        )

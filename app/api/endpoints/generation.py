import logging
from typing import List
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Query

from app.core.config import get_settings, Settings
from app.schemas.qa import QAItem, QAGenerationResponse
from app.services.data_generator import QAGenerator
from app.utils.exceptions import (
    DataGenerationError,
    FileExtractionError,
    LLMProviderError,
)

router = APIRouter()
logger = logging.getLogger(__name__)
settings = get_settings()


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
    This endpoint extracts text from a document and uses a powerful LLM to generate
    a dataset of approximately 10,000 question-answer pairs.
    """
    logger.info(
        f"Received request to generate QA from '{file.filename}' using '{llm_provider}'."
    )
    try:
        generator = QAGenerator(provider=llm_provider)
        qa_pairs = await generator.generate_from_file(file)

        response = QAGenerationResponse(generated_pairs=len(qa_pairs), data=qa_pairs)
        return response
    except (FileExtractionError, DataGenerationError, LLMProviderError) as e:
        logger.error(f"Generation failed for '{file.filename}': {e.detail}")
        raise HTTPException(status_code=400, detail=e.detail)
    except Exception as e:
        logger.exception(
            f"An unexpected error occurred while generating QA for '{file.filename}': {e}"
        )
        raise HTTPException(
            status_code=500, detail="An internal server error occurred."
        )

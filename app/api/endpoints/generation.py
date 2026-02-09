# app/api/endpoints/generation.py
import logging
import json
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, Query, Body
from pydantic import BaseModel

from app.core.config import get_settings, LLMProviderEnum
from app.services.agent_generator import AgentGenerator
from app.utils.exceptions import DataGenerationError, LLMProviderError

router = APIRouter()
logger = logging.getLogger(__name__)
settings = get_settings()

class GenerationResponse(BaseModel):
    success: bool
    generated_count: int
    data: List[dict]

@router.post(
    "/generate",
    response_model=GenerationResponse,
    summary="Generate Data from Prompt and/or Files",
    tags=["Generation"],
)
async def generate_data(
    prompt: str = Form(..., description="The instruction prompt for data generation."),
    files: List[UploadFile] = File(None, description="Source files (PDF, DOCX, CSV) to be used as context."),
    demo_file: UploadFile = File(None, description="A CSV file containing few-shot examples (columns: question, answer)."),
    provider: str = Form("groq", description="LLM Provider (groq, openai, google, huggingface)"),
    model: Optional[str] = Form(None, description="Specific model name to use."),
    temperature: float = Form(0.7, description="Temperature for generation (0.0 to 1.0)."),
    count: int = Form(10, description="Number of items to generate."),
    agentic: bool = Form(False, description="Enable agentic behavior (experimental)."),
    mcp_servers: Optional[str] = Form(None, description="JSON string list of MCP server URLs (optional)."),
):
    """
    Unified endpoint to generate synthetic data.
    - **Prompt**: Instructions for what to generate.
    - **Files**: Context documents.
    - **Demo File**: Few-shot examples.
    - **Agentic**: If true, uses an agent to determine execution path (currently experimental).
    """
    logger.info(f"Received generation request. Provider: {provider}, Model: {model}, Count: {count}, Agentic: {agentic}")

    try:
        # Parse MCP servers if provided
        parsed_mcp_servers = []
        if mcp_servers:
            try:
                parsed_mcp_servers = json.loads(mcp_servers)
                if not isinstance(parsed_mcp_servers, list):
                    raise ValueError("MCP servers must be a list of strings.")
            except Exception as e:
                logger.warning(f"Failed to parse MCP servers: {e}")

        generator = AgentGenerator(
            provider=provider,
            model=model,
            temperature=temperature
        )

        # Ensure files list is not None (FastAPI might pass None if no files)
        files = files or []

        generated_data = await generator.generate(
            prompt=prompt,
            files=files,
            demo_file=demo_file,
            count=count,
            agentic=agentic,
            mcp_servers=parsed_mcp_servers
        )

        return GenerationResponse(
            success=True,
            generated_count=len(generated_data),
            data=generated_data
        )

    except (LLMProviderError, DataGenerationError) as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected generation failure: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/models", summary="List Available Models", tags=["Info"])
def list_models():
    """
    Returns a list of supported providers and their default models.
    """
    return {
        "providers": ["groq", "openai", "google", "huggingface"],
        "defaults": {
            "groq": settings.GROQ_MODEL_NAME,
            "openai": settings.OPENAI_MODEL_NAME,
            "google": settings.GOOGLE_MODEL_NAME,
            "huggingface": settings.HUGGINGFACE_MODEL_NAME,
        }
    }

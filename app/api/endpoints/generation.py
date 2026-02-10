# app/api/endpoints/generation.py
import logging
import json
from typing import List, Optional, Dict
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, Query, Body
from pydantic import BaseModel

from app.core.config import get_settings, LLMProviderEnum
from app.services.agent_generator import AgentGenerator
from app.services.hf_service import HuggingFaceService
from app.utils.exceptions import DataGenerationError, LLMProviderError

router = APIRouter()
logger = logging.getLogger(__name__)
settings = get_settings()

class GenerationResponse(BaseModel):
    success: bool
    generated_count: int
    data: List[dict]
    message: Optional[str] = None

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
    agentic: bool = Form(False, description="Enable agentic behavior (uses tools/RAG)."),
    use_rag: bool = Form(False, description="Enable RAG for document querying (requires agentic mode)."),
    conserve_tokens: bool = Form(False, description="Optimize context to conserve tokens."),
    rate_limit: int = Form(0, description="Rate limit (requests per minute). 0 for unlimited."),
    mcp_servers: Optional[str] = Form(None, description="JSON string list of MCP server URLs (optional)."),
    # HF Options
    hf_repo_id: Optional[str] = Form(None, description="Hugging Face Repo ID to push to (e.g., 'username/dataset')."),
    hf_token: Optional[str] = Form(None, description="Hugging Face Write Token."),
    hf_private: bool = Form(False, description="Make HF dataset private."),
    hf_append: bool = Form(False, description="Append to existing HF dataset instead of overwrite.")
):
    """
    Unified endpoint to generate synthetic data.
    """
    logger.info(f"Received generation request. Provider: {provider}, Rate Limit: {rate_limit}")

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

        files = files or []

        generated_data = await generator.generate(
            prompt=prompt,
            files=files,
            demo_file=demo_file,
            count=count,
            agentic=agentic,
            mcp_servers=parsed_mcp_servers,
            use_rag=use_rag,
            conserve_tokens=conserve_tokens,
            rate_limit=rate_limit
        )

        message = "Data generated successfully."

        # Handle HF Push
        if hf_repo_id and hf_token:
            if generated_data:
                try:
                    hf_msg = HuggingFaceService.push_dataset(
                        data=generated_data,
                        repo_id=hf_repo_id,
                        token=hf_token,
                        private=hf_private,
                        append=hf_append
                    )
                    message += f" {hf_msg}"
                except Exception as e:
                    message += f" Warning: HF Push failed: {str(e)}"
            else:
                message += " Warning: No data generated to push."

        return GenerationResponse(
            success=True,
            generated_count=len(generated_data),
            data=generated_data,
            message=message
        )

    except (LLMProviderError, DataGenerationError) as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected generation failure: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

class DatasetModifyRequest(BaseModel):
    repo_id: str
    token: str
    data: List[Dict]
    operation: str = "append_rows" # or "add_column"

@router.post("/dataset/modify", summary="Modify Existing HF Dataset", tags=["Dataset"])
async def modify_dataset(request: DatasetModifyRequest):
    try:
        msg = HuggingFaceService.modify_dataset(
            repo_id=request.repo_id,
            new_data=request.data,
            token=request.token,
            operation=request.operation
        )
        return {"success": True, "message": msg}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

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

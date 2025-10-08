import logging
from fastapi import APIRouter, Depends, HTTPException
from app.schemas.finetuning import FinetuningRequest, FinetuningResponse
from app.services.finetuning_generator import FinetuningGenerator

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post(
    "/generate-finetuning-data/",
    response_model=FinetuningResponse,
    summary="Generate Finetuning Data",
    tags=["Finetuning"],
)
async def generate_finetuning_data(request: FinetuningRequest):
    """
    This endpoint generates finetuning data based on a prompt and model type.
    """
    try:
        generator = FinetuningGenerator(model_type=request.model_type)
        result = await generator.generate_data(request)
        return FinetuningResponse(success=True, data=result["data"])
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        raise HTTPException(
            status_code=500, detail="An internal server error occurred."
        )
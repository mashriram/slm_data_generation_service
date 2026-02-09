from pydantic import BaseModel, Field
from typing import List, Dict, Any

class FinetuningRequest(BaseModel):
    prompt: str = Field(..., description="The prompt to use for generating the finetuning data.")
    model_type: str = Field(..., description="The model type to use for the generation.")

class FinetuningResponse(BaseModel):
    success: bool
    data: List[Dict[str, Any]]
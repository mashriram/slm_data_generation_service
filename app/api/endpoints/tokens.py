from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.token_manager import TokenManager

router = APIRouter()
token_manager = TokenManager()

class TokenSetRequest(BaseModel):
    provider: str
    token: str

class ProviderListResponse(BaseModel):
    providers: List[str]

@router.post("/tokens", response_model=dict)
def set_token(request: TokenSetRequest):
    """
    Set or update the API token for a specific provider.
    """
    try:
        token_manager.set_token(request.provider, request.token)
        return {"message": f"Token set for {request.provider}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tokens", response_model=ProviderListResponse)
def list_providers():
    """
    List all providers that have a configured token (either in DB or Env).
    """
    try:
        providers = token_manager.list_providers()
        return {"providers": providers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/tokens/{provider}", response_model=dict)
def delete_token(provider: str):
    """
    Delete the token for a specific provider from the database.
    Note: Cannot delete tokens set via environment variables.
    """
    try:
        if token_manager.delete_token(provider):
            return {"message": f"Token deleted for {provider}"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete token")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

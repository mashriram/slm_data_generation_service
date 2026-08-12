from fastapi import APIRouter
from app.api.endpoints import generation, tokens

api_router = APIRouter()
api_router.include_router(generation.router, prefix="/v1")
api_router.include_router(tokens.router, prefix="/v1", tags=["Tokens"])

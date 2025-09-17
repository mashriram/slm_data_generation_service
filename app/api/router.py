# app/api/router.py
from fastapi import APIRouter

from app.api.endpoints import generation

api_router = APIRouter()
api_router.include_router(generation.router, prefix="/v1")

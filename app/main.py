# app/main.py
import logging
import asyncio
from typing import List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import get_settings
from app.core.logging_utils import configure_logging, log_queue

# Configure logging
configure_logging()

settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="A service to generate Question-Answer datasets from documents using various LLM providers.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")


@app.get("/", tags=["Health Check"])
def read_root():
    return {
        "status": "ok",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
    }


@app.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            if log_queue:
                # Send all pending logs
                while log_queue:
                    log_entry = log_queue.popleft()
                    await websocket.send_text(log_entry)

            # Efficiently wait before checking again
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        logging.info("WebSocket client disconnected")

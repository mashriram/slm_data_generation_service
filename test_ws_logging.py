import asyncio
import logging
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

async def test_ws():
    with client.websocket_connect("/ws/logs") as websocket:
        # Trigger a log message by hitting the health check endpoint
        client.get("/")

        # Read logs from websocket
        # We might receive multiple log entries (start up logs, the trigger log, etc.)
        found = False
        for _ in range(5):
            try:
                data = websocket.receive_text()
                print(f"Received log: {data}")
                if "GET /" in data or "200" in data or "Health Check" in data:
                    found = True
                    break
            except Exception:
                break

        assert found, "Did not receive expected log message via WebSocket"

if __name__ == "__main__":
    # Configure logging to ensure it's set up
    logging.getLogger().info("Starting test...")
    # Run the async test
    asyncio.run(test_ws())

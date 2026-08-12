import asyncio
import logging
import json
from collections import deque
from typing import Deque

# Global queue to hold log records
log_queue: Deque[str] = deque(maxlen=1000)


class QueueHandler(logging.Handler):
    """
    This handler sends events to a queue.
    """

    def __init__(self):
        super().__init__()

    def emit(self, record):
        try:
            msg = self.format(record)
            log_queue.append(msg)
        except Exception:
            self.handleError(record)

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)

def configure_logging():
    """Configures the root logger to use the QueueHandler."""
    handler = QueueHandler()
    formatter = JSONFormatter()
    handler.setFormatter(formatter)

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    # Ensure standard output still gets logs (as text or JSON, let's keep text for console readability)
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.INFO)

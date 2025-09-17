# app/utils/exceptions.py

class FileExtractionError(Exception):
    """Custom exception for errors during text extraction from files."""
    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(self.detail)

class LLMProviderError(Exception):
    """Custom exception for errors related to the LLM provider."""
    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(self.detail)

class DataGenerationError(Exception):
    """Custom exception for errors during the QA generation process."""
    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(self.detail)```

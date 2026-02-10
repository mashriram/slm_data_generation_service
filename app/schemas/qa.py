from typing import List

from pydantic import BaseModel, Field


class QAItem(BaseModel):
    """
    Defines the structure of a single generated Question-Answer pair.
    """

    question: str = Field(
        ..., description="The generated question based on the document's content."
    )
    answer: str = Field(
        ..., description="The corresponding answer to the generated question."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is the primary benefit of using the datasets-generation library?",
                "answer": "The primary benefit is its ability to reliably generate large datasets by applying a generation function over a dataset, handling batching and retries automatically.",
            }
        }


class QAGenerationResponse(BaseModel):
    """
    The response model for the QA generation endpoint.
    """

    generated_pairs: int
    data: List[QAItem]

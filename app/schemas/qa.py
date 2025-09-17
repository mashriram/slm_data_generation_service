# app/schemas/qa.py
from pydantic import BaseModel, Field
from typing import List


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
                "question": "What is the primary benefit of using LangChain?",
                "answer": "LangChain's primary benefit is its ability to chain together different language model components, enabling the creation of more complex and powerful applications.",
            }
        }


class QAGenerationResponse(BaseModel):
    """
    The response model for the QA generation endpoint.
    """

    generated_pairs: int
    data: List[QAItem]

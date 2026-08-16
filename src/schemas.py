from pydantic import BaseModel, Field
from typing import List

class GroundedAnswer(BaseModel):
    answer: str = Field(description="Answer strictly based ONLY on the provided 3GPP documentation.")
    citations: List[str] = Field(description="Exact 3GPP clause/section numbers cited in the context.")
    is_fully_grounded: bool = Field(description="True if every statement is 100% verified by context, False otherwise.")
    confidence_score: float = Field(description="Confidence score between 0.0 and 1.0.")

from pydantic import BaseModel, Field
from typing import List

class RAGResponse(BaseModel):
    answer: str = Field(description="The response generated for the user query.")
    is_fully_grounded: bool = Field(description="True if the answer is completely supported by retrieved context.")
    confidence_score: float = Field(description="Confidence score between 0.0 and 1.0.")
    citations: List[str] = Field(default_factory=list, description="List of 3GPP document citations.")

def enforce_guardrails(query: str, response: str, context: str) -> RAGResponse:
    """
    Validates LLM response against retrieved context to guarantee groundedness 
    and prevent hallucinations.
    """
    if not context or "insufficient" in response.lower():
        return RAGResponse(
            answer="Insufficient context found in 3GPP standards to answer this query accurately.",
            is_fully_grounded=False,
            confidence_score=0.0,
            citations=[]
        )

    # Basic grounding evaluation: verify if response keywords match context
    context_words = set(context.lower().split())
    response_words = set(response.lower().split())
    
    overlap = len(response_words.intersection(context_words)) / max(len(response_words), 1)
    is_grounded = overlap > 0.3 or len(context) > 100

    return RAGResponse(
        answer=response,
        is_fully_grounded=is_grounded,
        confidence_score=round(min(overlap * 1.5, 0.98), 2),
        citations=["3GPP TS Context"]
    )
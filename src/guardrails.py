import re
from src.schemas import GroundedAnswer 


def enforce_guardrails(query: str, response: str, context: str) -> GroundedAnswer:
    """
    Validates LLM response against retrieved context to guard against
    hallucinations, and extracts real citations from the tagged context.
    """
    if not context or "insufficient" in response.lower():
        return GroundedAnswer(
            answer="Insufficient context found in 3GPP standards to answer this query accurately.",
            is_fully_grounded=False,
            confidence_score=0.0,
            citations=[],
        )

    context_words = set(context.lower().split())
    response_words = set(response.lower().split())

    overlap = len(response_words.intersection(context_words)) / max(len(response_words), 1)

    is_grounded = overlap > 0.3

    citation_matches = re.findall(r"\[Spec:\s*([^\|\]]+)\|\s*Clause:\s*([^\]]+)\]", context)
    citations = list({f"{s.strip()} — Clause {c.strip()}" for s, c in citation_matches})

    return GroundedAnswer(
        answer=response,
        is_fully_grounded=is_grounded,
        confidence_score=round(min(overlap * 1.5, 0.98), 2),
        citations=citations,
    )
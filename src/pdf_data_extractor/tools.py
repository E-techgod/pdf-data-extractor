from typing import Literal, TypedDict


DocumentType = Literal[
    "invoice",
    "resume",
    "receipt",
    "report",
    "generic",
]


class DocumentClassification(TypedDict):
    document_type: DocumentType
    reason: str


def classify_document(document_text: str) -> DocumentClassification:
    """
    Classify a document using simple keyword rules.

    This is the local Python function that the LLM will eventually call.
    The function must remain deterministic and independent from the LLM API.
    """
    normalized_text = document_text.strip().lower()

    if not normalized_text:
        raise ValueError("document_text cannot be empty")

    keyword_groups: dict[DocumentType, tuple[str, ...]] = {
        "invoice": (
            "invoice number",
            "invoice #",
            "bill to",
            "amount due",
            "payment due",
        ),
        "resume": (
            "work experience",
            "professional experience",
            "education",
            "skills",
            "employment history",
        ),
        "receipt": (
            "receipt",
            "subtotal",
            "sales tax",
            "change due",
            "payment method",
        ),
        "report": (
            "executive summary",
            "findings",
            "methodology",
            "recommendations",
            "conclusion",
        ),
    }

    scores: dict[DocumentType, int] = {
        document_type: sum(
            keyword in normalized_text
            for keyword in keywords
        )
        for document_type, keywords in keyword_groups.items()
    }

    best_type = max(scores, key=scores.get)

    if scores[best_type] == 0:
        return {
            "document_type": "generic",
            "reason": "No recognized document-specific keywords were found.",
        }

    return {
        "document_type": best_type,
        "reason": (
            f"Detected {scores[best_type]} keyword match(es) "
            f"associated with {best_type} documents."
        ),
    }
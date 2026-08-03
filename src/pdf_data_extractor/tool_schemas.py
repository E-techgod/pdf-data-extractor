from typing import Any


CLASSIFY_DOCUMENT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "classify_document",
        "description": (
            "Classify a document as an invoice, resume, receipt, "
            "report, or generic document using the document "
            "already provided in the conversation context."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
}


TOOLS: list[dict[str, Any]] = [
    CLASSIFY_DOCUMENT_TOOL,
]

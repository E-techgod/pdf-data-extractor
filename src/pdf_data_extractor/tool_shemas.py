from typing import Any


CLASSIFY_DOCUMENT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "classify_document", # Must match python funtion name 
        "description": (
            "Classify a document as an invoice, resume, receipt, "
            "report, or generic document."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "document_text": {
                    "type": "string",
                    "description": (
                        "The complete extracted text of the document "
                        "that needs to be classified."
                    ),
                }
            },
            "required": ["document_text"],
            "additionalProperties": False, # prevents the model from inventing unsupported arguments 
        },
    },
}


TOOLS: list[dict[str, Any]] = [
    CLASSIFY_DOCUMENT_TOOL,
]